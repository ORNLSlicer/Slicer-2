#include "graphics/objects/part_object.h"

// Local
#include "managers/settings/settings_manager.h"
#include "part/part.h"
#include "utilities/mathutils.h"
#include "geometry/mesh/mesh_base.h"
#include "graphics/base_view.h"
#include "graphics/objects/arrow_object.h"
#include "graphics/objects/axes_object.h"
#include "graphics/objects/text_object.h"
#include "graphics/objects/cube/plane_object.h"

namespace ORNL {
    PartObject::PartObject(BaseView* view, QSharedPointer<Part> p, ushort render_mode) {
        m_part = p;
        QSharedPointer<MeshBase> mesh = p->rootMesh();

        // Vertices
        float* vert_float;
        unsigned int vert_size;

        std::tie(vert_float, vert_size) = mesh->glVertexArray();
        std::vector<float> vertices = std::vector<float>(vert_float, vert_float + (vert_size / sizeof(float)));

        // Scale to OpenGL space
        for(int i = 0; i < vert_size / sizeof(float); i += 3)
        {
            vertices[i]       = (vertices[i]) * Constants::OpenGL::kObjectToView;
            vertices[i + 1]   = (vertices[i + 1]) * Constants::OpenGL::kObjectToView;
            vertices[i + 2]   = (vertices[i + 2]) * Constants::OpenGL::kObjectToView;
        }

        // Normals
        float* normals_float;
        unsigned int normal_size;

        std::tie(normals_float, normal_size) = mesh->glNormalArray();
        std::vector<float> normals = std::vector<float>(normals_float, normals_float + (normal_size / sizeof(float)));

        // Colors
        // Setup colors based on mesh type
        switch(mesh->type())
        {
            case(MeshType::kBuild):
                m_base_color = Constants::Colors::kLightBlue;
                m_selected_color = Constants::Colors::kGreen;
                break;
            case(MeshType::kClipping):
                m_base_color = Constants::Colors::kRed;
                m_selected_color = Constants::Colors::kOrange;
                break;
            case(MeshType::kSettings):
                m_base_color = Constants::Colors::kBlue;
                m_selected_color = Constants::Colors::kPurple;
                break;
            case(MeshType::kEmbossSubmesh):
                m_base_color = Constants::Colors::kPurple;
                m_selected_color = Constants::Colors::kBlue;

                this->setLocked(true);
                break;
        }

        m_color = m_base_color;

        int totalColors = 4 * (vertices.size() / 3);
        std::vector<float> colors;
        colors.resize(totalColors);
        for(int i = 0; i < totalColors; i += 4)
        {
            colors[i]     = m_base_color.redF();
            colors[i + 1] = m_base_color.greenF();
            colors[i + 2] = m_base_color.blueF();
            colors[i + 3] = m_base_color.alphaF();
        }

        // Float buffers have been copied into std::vector by now.
        delete vert_float, normals_float;

        this->populateGL(view, vertices, normals, colors, render_mode);

        // Make a label.
        auto got = QSharedPointer<TextObject>::create(this->view(), m_part->name());
        got->setOnTop(true);
        got->hide();

        this->adoptChild(got);
        m_label_object = got;

        // Make axis.
        float length = this->maximum().x() - this->minimum().x();
        float width  = this->maximum().y() - this->minimum().y();
        float depth  = this->maximum().z() - this->minimum().z();
        auto goax = QSharedPointer<AxesObject>::create(this->view(), std::fmax(std::fmax(length, width), depth));
        goax->setOnTop(true);
        goax->hide();

        this->adoptChild(goax);
        m_axes_object = goax;
        m_axes_object->translateAbsolute(this->minimum());

        // Make slicing plane.
        float max_dim = std::fmax(length, std::fmax(width, depth));
        max_dim += max_dim * 0.20;
        auto gos = QSharedPointer<PlaneObject>::create(this->view(), max_dim, max_dim);
        gos->setLockedRotation(true);
        gos->hide();

        this->adoptChild(gos);
        m_plane_object = gos;

        // Object translation happens last since it will callback (see translationCallback()) this object and everything
        // needs to be setup first.
        QVector3D trans;
        QQuaternion rot;
        QVector3D scale;

        std::tie(trans, rot, scale) = MathUtils::decomposeTransformMatrix(mesh->transformation());

        trans *= Constants::OpenGL::kObjectToView;

        this->setTransformation(MathUtils::composeTransformMatrix(trans, rot, scale), true);
    }

    QSet<QSharedPointer<PartObject>> PartObject::select()
    {
        QSet<QSharedPointer<PartObject>> ret;

        // Cannot have both parent and this selected at the same time, so unselect it.
        QSharedPointer<PartObject> p = this->parent().dynamicCast<PartObject>();
        if (!p.isNull()) {
            while (!p.isNull()) {
                if (p->m_selected) {
                    ret.insert(p);
                    p->unselect();
                }

                p = p->parent().dynamicCast<PartObject>();
            }

            this->arrow()->show();

            m_axes_object->show();
        }

        m_color = m_selected_color;
        this->paint(m_color);

        // Select children but give them a different highlighting.
        for (auto& cp : m_part_children) {
            ret.unite(cp->select());
            cp->m_color = Constants::Colors::kYellow;
            cp->m_color.setAlpha(cp->transparency());
            cp->paint(cp->m_color);
            cp->m_selected = false;

            cp->arrow()->show();

            ret.insert(cp);
        }

        m_selected = true;

        return ret;
    }

    void PartObject::unselect() {
        m_color = m_base_color;
        this->paint(m_color);

        for (auto& cp : m_part_children) {
            cp->arrow()->hide();
            cp->unselect();
        }

        QSharedPointer<PartObject> p = this->parent().dynamicCast<PartObject>();
        if (!p.isNull())
        {
            this->arrow()->hide();
            m_axes_object->show();
        }

        m_selected = false;
    }

    void PartObject::highlight() {
        this->paint(m_color.lighter());
    }

    void PartObject::unhighlight() {
        this->paint(m_color);
    }

    void PartObject::setTransparency(uint trans) {
        m_transparency = trans;
        m_base_color.setAlpha(trans);
        m_selected_color.setAlpha(trans);

        m_color = (m_selected) ? m_selected_color : m_base_color;

        this->paint(m_color);
    }

    uint PartObject::transparency() {
        return m_transparency;
    }

    void PartObject::setMeshTypeColor(MeshType type) {
        switch(type)
        {
            case(MeshType::kBuild):
                m_base_color = Constants::Colors::kLightBlue;
                m_selected_color = Constants::Colors::kGreen;
                break;
            case(MeshType::kClipping):
                m_base_color = Constants::Colors::kRed;
                m_selected_color = Constants::Colors::kOrange;
                break;
            case(MeshType::kSettings):
                m_base_color = Constants::Colors::kBlue;
                m_selected_color = Constants::Colors::kPurple;
                break;
            case(MeshType::kEmbossSubmesh):
                m_base_color = Constants::Colors::kPurple;
                m_selected_color = Constants::Colors::kBlue;
                break;
        }

        m_color = (m_selected) ? m_selected_color : m_base_color;
        this->paint(m_color);
    }

    void PartObject::setRenderMode(ushort mode)
    {
        renderMode() = mode;
        render();
    }

    QSharedPointer<ArrowObject> PartObject::arrow() {
        return m_arrow_object;
    }

    QSharedPointer<TextObject> PartObject::label() {
        return m_label_object;
    }

    QSharedPointer<AxesObject> PartObject::axes() {
        return m_axes_object;
    }

    QSharedPointer<PlaneObject> PartObject::plane() {
        return m_plane_object;
    }

    void PartObject::showOverhang(bool show) {
        m_overhang_shown = show;
        if (show) this->overhangUpdate();
        else this->paint(m_color);
    }

    void PartObject::setOverhangAngle(Angle a) {
        m_overhang_angle = a;
        if (m_overhang_shown) this->overhangUpdate();
        else this->paint(m_color);
    }

    QString PartObject::name() {
        return m_part->name();
    }

    QSharedPointer<Part> PartObject::part() {
        return m_part;
    }

    void PartObject::overhangUpdate() {
        // From old graphics part.
        QMatrix4x4 objectTransform = this->transformation();

        QMatrix4x4 rot;
        rot.rotate(this->rotation());

        const std::vector<float>& vertices = this->vertices();
        const std::vector<float>& normals = this->normals();
        const std::vector<float>& colors = this->colors();

        std::vector<float> new_colors;
        new_colors.resize(colors.size());

        QColor color_to_set;
        int j = 0;

        QSharedPointer<SettingsBase> global_sb = GSM->getGlobal();
        Angle stacking_pitch = global_sb->setting<Angle>(Constants::ExperimentalSettings::SlicingAngle::kStackingDirectionPitch);
        Angle stacking_yaw   = global_sb->setting<Angle>(Constants::ExperimentalSettings::SlicingAngle::kStackingDirectionYaw);
        Angle stacking_roll  = global_sb->setting<Angle>(Constants::ExperimentalSettings::SlicingAngle::kStackingDirectionRoll);

        // Build a normal vector using the quaternion
        QVector3D stacking_axis(0, 0, 1);
        QQuaternion quaternion = MathUtils::CreateQuaternion(stacking_pitch, stacking_yaw, stacking_roll);
        stacking_axis = quaternion.rotatedVector(stacking_axis).normalized();

        //! \todo Move this to GLSL shader to avoid the copies between CPU and graphics.
        for (int i = 0; i < vertices.size(); i += 9) {
            QVector3D p1 = objectTransform * QVector3D(vertices[i + 0], vertices[i + 1], vertices[i + 2]);
            QVector3D p2 = objectTransform * QVector3D(vertices[i + 3], vertices[i + 4], vertices[i + 5]);
            QVector3D p3 = objectTransform * QVector3D(vertices[i + 6], vertices[i + 7], vertices[i + 8]);

            QVector3D normal = rot * QVector3D(normals[i], normals[i + 1], normals[i + 2]);

            color_to_set = m_color;

            auto upward = QVector3D::dotProduct(stacking_axis, normal);

            //Z pointing down
            if (upward <= 0.0)
            {
                float faceAngle;
                auto val = QVector3D::dotProduct(stacking_axis, normal) / (normal.length() * stacking_axis.length());

                if (val <= -1.0)
                    faceAngle = M_PI;
                else if (val >= 1.0)
                    faceAngle = 0;
                else
                    faceAngle = qAcos(val);

                faceAngle = faceAngle - (M_PI / 2.0);

                if (faceAngle > m_overhang_angle)
                    color_to_set = Constants::Colors::kRed;
            }

            new_colors[j + 0]  = color_to_set.redF();
            new_colors[j + 1]  = color_to_set.greenF();
            new_colors[j + 2]  = color_to_set.blueF();
            new_colors[j + 3]  = color_to_set.alphaF();

            new_colors[j + 4]  = color_to_set.redF();
            new_colors[j + 5]  = color_to_set.greenF();
            new_colors[j + 6]  = color_to_set.blueF();
            new_colors[j + 7]  = color_to_set.alphaF();

            new_colors[j + 8]  = color_to_set.redF();
            new_colors[j + 9]  = color_to_set.greenF();
            new_colors[j + 10] = color_to_set.blueF();
            new_colors[j + 11] = color_to_set.alphaF();

            j += 12;
        }

        this->updateColors(new_colors);
    }

    void PartObject::transformationCallback() {
        if (!m_arrow_object.isNull()) m_arrow_object->updateEndpoints();
        for (auto& goc : m_part_children) {
            if (!goc->arrow().isNull()) goc->arrow()->updateEndpoints();
        }

        QVector3D label_pos = this->center();
        label_pos.setZ(this->maximum().z() + 0.5);
        m_label_object->translateAbsolute(label_pos, false);

        m_plane_object->scaleAbsolute(QVector3D(this->scaling().x(), this->scaling().y(), 0));

        float length = this->minimumBoundingBox()[MBB_FLB].distanceToPoint(this->minimumBoundingBox()[MBB_FRB]);
        float width  = this->minimumBoundingBox()[MBB_FLB].distanceToPoint(this->minimumBoundingBox()[MBB_BLB]);
        float depth  = this->minimumBoundingBox()[MBB_FLT].distanceToPoint(this->minimumBoundingBox()[MBB_FLB]);
        m_axes_object->updateDimensions(std::fmax(std::fmax(length, width), depth));

        if (m_overhang_shown) this->overhangUpdate();
    }

    void PartObject::adoptChildCallback(QSharedPointer<GraphicsObject> child) {
        // We only care about new part objects.
        QSharedPointer<PartObject> goc = child.dynamicCast<PartObject>();
        if (goc.isNull()) return;

        m_part_children.insert(goc);

        // Make an arrow. Currently, arrows are only between parts. If we want arrows between other
        // objects, it might be worth the time to move this to the graphics object for the general case.
        auto goa = QSharedPointer<ArrowObject>::create(this->view(), goc, this->sharedFromThis());
        goa->setOnTop(true);
        goa->hide();

        goc->m_arrow_object = goa;
        goc->adoptChild(goa);
    }

    void PartObject::orphanChildCallback(QSharedPointer<GraphicsObject> child) {
        // We only care about removed part objects.
        QSharedPointer<PartObject> goc = child.dynamicCast<PartObject>();
        if (goc.isNull()) return;

        m_part_children.remove(goc);

        goc->m_arrow_object.reset();
    }

    void PartObject::paint(QColor color) {
        color.setAlpha(m_transparency);

        this->GraphicsObject::paint(color);
        if (m_overhang_shown) this->overhangUpdate();
    }
}
