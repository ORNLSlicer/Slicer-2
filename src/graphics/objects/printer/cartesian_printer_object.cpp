#include "graphics/objects/printer/cartesian_printer_object.h"

// Local
#include "utilities/constants.h"
#include "utilities/mathutils.h"
#include "graphics/objects/part_object.h"
#include "graphics/support/shape_factory.h"
#include "graphics/objects/axes_object.h"
#include "graphics/objects/cube/plane_object.h"

namespace ORNL
{
    CartesianPrinterObject::CartesianPrinterObject(BaseView* view, QSharedPointer<SettingsBase> sb, bool is_true_volume) : PrinterObject(is_true_volume)
    {
        this->setSettings(sb);
        this->updateMembers();

        std::vector<float> vertices;
        std::vector<float> colors;

        ShapeFactory::createBuildVolumeRectangle(m_min, m_max, m_x_grid, m_x_grid_offset, m_y_grid, m_y_grid_offset, Constants::Colors::kBlack, vertices, colors);

        std::vector<float> tmp_norm;
        this->populateGL(view, vertices, tmp_norm, colors, GL_LINES);

        float length = m_max.x() - m_min.x();
        float width  = m_max.y() - m_min.y();

        // Axes
        auto goax = QSharedPointer<AxesObject>::create(view, std::fmin(length, width) * 0.4);
        goax->translate(m_min);

        // Floor plane
        auto gopl = QSharedPointer<PlaneObject>::create(view, m_max.x() - m_min.x(), m_max.y() - m_min.x(), Constants::Colors::kBlue.lighter(160));
        gopl->translate(m_floor_center);
        gopl->setUnderneath(true);

        m_axes = goax;
        m_floor_plane = gopl;

        this->adoptChild(goax);
        this->adoptChild(gopl);

        // Seams
        this->createSeams();
        this->updateSeams();
    }

    QVector3D CartesianPrinterObject::printerCenter()
    {
        return this->translation() + m_floor_center;
    }

    QList<QSharedPointer<PartObject>> CartesianPrinterObject::externalParts()
    {
        QList<QSharedPointer<PartObject>> ret;
        QVector3D printerMin = this->minimum();
        QVector3D printerMax = this->maximum();

        for(auto& child : this->allChildren()) {
            auto gop = child.dynamicCast<PartObject>();
            if (gop.isNull()) continue;

            QVector3D partMin = gop->minimum();
            QVector3D partMax = gop->maximum();

            if ((!MathUtils::glEquals(partMin.x(), printerMin.x()) && partMin.x() < printerMin.x()) ||
                (!MathUtils::glEquals(partMin.y(), printerMin.y()) && partMin.y() < printerMin.y()) ||
                (!MathUtils::glEquals(partMin.z(), printerMin.z()) && partMin.z() < printerMin.z()) ||
                (!MathUtils::glEquals(partMax.x(), printerMax.x()) && partMax.x() > printerMax.x()) ||
                (!MathUtils::glEquals(partMax.y(), printerMax.y()) && partMax.y() > printerMax.y()) ||
                (!MathUtils::glEquals(partMax.z(), printerMax.z()) && partMax.z() > printerMax.z()))
                ret.append(gop);
        }

        /* More accurate but slower.
        for(auto& child : this->allChildren()) {
            auto gop = child.dynamicCast<PartObject>();
            if (gop.isNull()) continue;

            for(Triangle tri : gop->triangles()) {
                for (uint i = 0; i < 3; i++) {
                    QVector3D pt = tri[i];

                    if ((!MathUtils::glEquals(pt.x(), printerMin.x()) && pt.x() < printerMin.x()) ||
                        (!MathUtils::glEquals(pt.y(), printerMin.y()) && pt.y() < printerMin.y()) ||
                        (!MathUtils::glEquals(pt.z(), printerMin.z()) && pt.z() < printerMin.z()) ||
                        (!MathUtils::glEquals(pt.x(), printerMax.x()) && pt.x() > printerMax.x()) ||
                        (!MathUtils::glEquals(pt.y(), printerMax.y()) && pt.y() > printerMax.y()) ||
                        (!MathUtils::glEquals(pt.z(), printerMax.z()) && pt.z() > printerMax.z())) {

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

    void CartesianPrinterObject::updateMembers()
    {
        QSharedPointer<SettingsBase> sb = this->getSettings();

        m_x_grid = 0;
        m_x_grid_offset = 0;
        m_y_grid = 0;
        m_y_grid_offset = 0;

        if(sb->setting<bool>(Constants::PrinterSettings::Dimensions::kEnableGridX)) {
            m_x_grid = sb->setting<float>(Constants::PrinterSettings::Dimensions::kGridXDistance);
            m_x_grid_offset = sb->setting<float>(Constants::PrinterSettings::Dimensions::kGridXOffset);
        }

        if(sb->setting<bool>(Constants::PrinterSettings::Dimensions::kEnableGridY)) {
            m_y_grid = sb->setting<float>(Constants::PrinterSettings::Dimensions::kGridYDistance);
            m_y_grid_offset = sb->setting<float>(Constants::PrinterSettings::Dimensions::kGridYOffset);
        }

        m_min = QVector3D(sb->setting<float>(Constants::PrinterSettings::Dimensions::kXMin),
                          sb->setting<float>(Constants::PrinterSettings::Dimensions::kYMin),
                          sb->setting<float>(Constants::PrinterSettings::Dimensions::kZMin));

        m_max = QVector3D(sb->setting<float>(Constants::PrinterSettings::Dimensions::kXMax),
                          sb->setting<float>(Constants::PrinterSettings::Dimensions::kYMax),
                          sb->setting<float>(Constants::PrinterSettings::Dimensions::kZMax));

        // If this is not a true volume (i.e. in the part view) then we draw a box starting at (x_min, y_min, z_min) where z_min > 0
        if(!isTrueVolume())
        {
            // If z-min < 0 then we add the extra space to the top of the box
            if(m_min.z() < 0.0)
            {
                m_max.setZ(m_max.z() + qFabs(m_min.z()));
                m_min.setZ(0);
            }

            m_max.setZ(m_max.z() - m_min.z());

            m_min.setZ(0);
        }

        if(sb->setting<bool>(Constants::PrinterSettings::Dimensions::kEnableW)) {
            m_max.setZ(m_max.z() + std::abs(  sb->setting<float>(Constants::PrinterSettings::Dimensions::kWMax)
                                            - sb->setting<float>(Constants::PrinterSettings::Dimensions::kWMin)));
        }

        m_min *= Constants::OpenGL::kObjectToView;
        m_max *= Constants::OpenGL::kObjectToView;
        m_x_grid *= Constants::OpenGL::kObjectToView;
        m_x_grid_offset *= Constants::OpenGL::kObjectToView;
        m_y_grid *= Constants::OpenGL::kObjectToView;
        m_y_grid_offset *= Constants::OpenGL::kObjectToView;

        m_floor_center = QVector3D((m_min.x() + m_max.x()) / 2.0f, (m_min.y() + m_max.y()) / 2.0f, m_min.z());

        m_printer_max_dims.setX((m_max.x() - m_min.x()));
        m_printer_max_dims.setY((m_max.y() - m_min.y()));
        m_printer_max_dims.setZ((m_max.z() - m_min.z()));
    }

    void CartesianPrinterObject::updateGeometry()
    {
        std::vector<float> vertices;
        std::vector<float> colors;

        ShapeFactory::createBuildVolumeRectangle(m_min, m_max, m_x_grid, m_x_grid_offset, m_y_grid, m_y_grid_offset, Constants::Colors::kBlack, vertices, colors);

        this->replaceVertices(vertices);
        this->replaceColors(colors);

        QVector3D translation = this->translation();

        this->translateAbsolute(QVector3D(0, 0, 0));

        float length = m_max.x() - m_min.x();
        float width  = m_max.y() - m_min.y();

        m_axes->updateDimensions(std::fmin(length, width) * 0.4);
        m_floor_plane->updateDimensions(length, width);

        m_axes->translateAbsolute(m_min);
        m_floor_plane->translateAbsolute(m_floor_center);

        this->translateAbsolute(translation);
    }
}
