#include "graphics/objects/printer/cylindrical_printer_object.h"

// Local
#include "utilities/constants.h"
#include "utilities/mathutils.h"
#include "graphics/objects/part_object.h"
#include "graphics/support/shape_factory.h"
#include "graphics/objects/axes_object.h"
#include "graphics/objects/cylinder/cylinder_plane_object.h"

namespace ORNL {
    CylindricalPrinterObject::CylindricalPrinterObject(BaseView* view, QSharedPointer<SettingsBase> sb, bool is_true_volume) : PrinterObject(is_true_volume) {
        this->setSettings(sb);
        this->updateMembers();

        std::vector<float> vertices;
        std::vector<float> colors;

        ShapeFactory::createBuildVolumeCylinder(m_radius, m_height, m_x_grid, m_y_grid, Constants::Colors::kBlack, vertices, colors);

        std::vector<float> tmp_norm;
        this->populateGL(view, vertices, tmp_norm, colors, GL_LINES);

        // Axes
        auto goax = QSharedPointer<AxesObject>::create(view, m_radius * 0.4);
        goax->translate(this->minimum());

        // Floor plane
        auto gopl = QSharedPointer<CylinderPlaneObject>::create(view, m_radius, Constants::Colors::kBlue.lighter(160));
        gopl->translate(m_floor_center);
        gopl->setUnderneath(true);

        m_axes = goax;
        m_floor_plane = gopl;

        this->adoptChild(goax);
        this->adoptChild(gopl);

        this->translateAbsolute(-this->printerCenter());

        // Seams
        this->createSeams();
        this->updateSeams();
    }

    QVector3D CylindricalPrinterObject::printerCenter() {
        return m_floor_center + this->translation();
    }

    QList<QSharedPointer<PartObject>> CylindricalPrinterObject::externalParts() {
        QList<QSharedPointer<PartObject>> ret;

        float printerMinZ = this->minimum().z();
        float printerMaxZ = this->maximum().z();
        float radiusSquared = m_radius * m_radius;
        QVector3D origin = this->printerCenter();

        for(auto& child : this->allChildren()) {
            auto gop = child.dynamicCast<PartObject>();
            if (gop.isNull()) continue;

            QVector3D partMin = gop->minimum();
            QVector3D partMax = gop->maximum();

            if (qPow(partMin.x() - origin.x(), 2) + qPow(partMin.y() - origin.y(), 2) > radiusSquared ||
               (!MathUtils::glEquals(partMin.z(), printerMinZ) && partMin.z() < printerMinZ) ||
                qPow(partMax.x() - origin.x(), 2) + qPow(partMax.y() - origin.y(), 2) > radiusSquared ||
               (!MathUtils::glEquals(partMax.z(), printerMaxZ) && partMax.z() > printerMaxZ))
                ret.append(gop);
        }

        /* More accurate but slower.
        for(auto& child : this->allChildren()) {
            auto gop = child.dynamicCast<PartObject>();
            if (gop.isNull()) continue;

            for(Triangle tri : gop->triangles()) {
                for (uint i = 0; i < 3; i++) {
                    QVector3D pt = tri[i];

                    float dist = qPow(pt.x() - origin.x(), 2) + qPow(pt.y() - origin.y(), 2);
                    float z = pt.z();

                    if (dist > radiusSquared ||
                       (!MathUtils::glEquals(z, printerMinZ) && z < printerMinZ) ||
                       (!MathUtils::glEquals(z, printerMaxZ) && z > printerMaxZ)) {

                       ret.append(gop);
                       // A goto in its natural habitat.
                       goto next_child;
                    }
                }
            }

            next_child:;
        }
        */

        return ret;
    }

    void CylindricalPrinterObject::updateMembers() {
        QSharedPointer<SettingsBase> sb = this->getSettings();

        m_x_grid = 0;
        m_y_grid = 0;

        if(sb->setting<bool>(Constants::PrinterSettings::Dimensions::kEnableGridX)) {
            m_x_grid = sb->setting<float>(Constants::PrinterSettings::Dimensions::kGridXDistance);
        }

        if(sb->setting<bool>(Constants::PrinterSettings::Dimensions::kEnableGridY)) {
            m_y_grid = sb->setting<float>(Constants::PrinterSettings::Dimensions::kGridYDistance);
        }

        m_radius = sb->setting<float>(Constants::PrinterSettings::Dimensions::kOuterRadius);

        QVector3D min(0.0, 0.0, sb->setting<float>(Constants::PrinterSettings::Dimensions::kZMin));
        QVector3D max(0.0, 0.0, sb->setting<float>(Constants::PrinterSettings::Dimensions::kZMax));

        // If this is not a true volume (i.e. in the part view) then we draw a box starting at (x_min, y_min, z_min) where z_min > 0
        if(!isTrueVolume())
        {
            // If z-min < 0 then we add the extra space to the top of the box
            if(min.z() < 0.0)
            {
                max.setZ(max.z() + qFabs(min.z()));
                min.setZ(0);
            }

            max.setZ(max.z() - min.z());

            min.setZ(0);
        }

        if(sb->setting<bool>(Constants::PrinterSettings::Dimensions::kEnableW)) {
            max.setZ(max.z() + std::abs(  sb->setting<float>(Constants::PrinterSettings::Dimensions::kWMax)
                                        - sb->setting<float>(Constants::PrinterSettings::Dimensions::kWMin)));
        }

        m_height = max.z() - min.z();

        m_radius *= Constants::OpenGL::kObjectToView;
        m_height *= Constants::OpenGL::kObjectToView;
        m_x_grid *= Constants::OpenGL::kObjectToView;
        m_y_grid *= Constants::OpenGL::kObjectToView;

        m_floor_center = QVector3D(0.0f, 0.0f, 0.0f);

        m_printer_max_dims.setX(m_radius);
        m_printer_max_dims.setY(m_radius);
        m_printer_max_dims.setZ(max.z() - min.z());
    }

    void CylindricalPrinterObject::updateGeometry() {
        std::vector<float> vertices;
        std::vector<float> colors;

        ShapeFactory::createBuildVolumeCylinder(m_radius, m_height, m_x_grid, m_y_grid, Constants::Colors::kBlack, vertices, colors);

        this->replaceVertices(vertices);
        this->replaceColors(colors);

        QVector3D translation = this->translation();

        this->translateAbsolute(QVector3D(0, 0, 0));

        m_axes->updateDimensions(m_radius * 0.4);
        m_floor_plane->updateDimensions(m_radius);

        m_axes->translateAbsolute(this->minimum());
        m_floor_plane->translateAbsolute(m_floor_center);

        this->translateAbsolute(translation);
    }
}
