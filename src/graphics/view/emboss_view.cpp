#include "graphics/view/emboss_view.h"

// Local
#include "part/part.h"
#include "utilities/mathutils.h"
#include "graphics/support/part_picker.h"
#include "graphics/objects/grid_object.h"
#include "graphics/objects/cube/plane_object.h"
#include "graphics/objects/part_object.h"
#include "graphics/objects/axes_object.h"

namespace ORNL {
    EmbossView::EmbossView() {
        // NOP
    }

    void EmbossView::clear() {
        m_plane->orphanChild(m_base_object);

        m_base_object.reset();
        m_emboss_objects.clear();

        //m_state.part_rot_normal.clear();

        this->update();
    }

    void EmbossView::setBasePart(QSharedPointer<Part> p) {
        m_plane->orphanChild(m_base_object);

        m_emboss_objects.clear();

        m_base_object.reset();
        m_base_object = QSharedPointer<PartObject>::create(this, p);
        m_base_object->translate(QVector3D(0, 0, -m_base_object->minimum().z()));

        m_plane->adoptChild(m_base_object);

        m_base_object->setTransparency(127);

        this->update();
    }

    void EmbossView::addEmbossPart(QSharedPointer<Part> p) {
        if (m_base_object.isNull()) return;
        auto gop = QSharedPointer<PartObject>::create(this, p);

        gop->translateAbsolute(m_base_object->center());
        gop->translate(QVector3D(0, -100, 0));

        // Snap the part to the base object.
        float dist;
        Triangle tri;
        QVector3D intersect;

        std::tie(dist, tri, intersect) = PartPicker::castRay(gop->center(), QVector3D(0, 1, 0), m_base_object->triangles());

        if (dist == Constants::Limits::Maximums::kInfFloat) gop->translate(QVector3D(0, 100, 0));
        else {
            gop->translateAbsolute(intersect);

            // Find the normal of the picked triangle.
            QVector3D u = tri.b - tri.a;
            QVector3D v = tri.c - tri.a;

            QVector3D norm = QVector3D::normal(u, v);

            gop->rotateAbsolute(QQuaternion::fromDirection(norm, QVector3D(0, 0, 1)));
        }

        m_emboss_objects.insert(gop);
        m_base_object->adoptChild(gop);

        //m_state.part_rot_normal[gop] = QQuaternion();

        this->update();
    }

    QVector<std::pair<QString, QVector3D>> EmbossView::scaleBasePart(QVector3D s) {
        if (m_base_object.isNull()) return QVector<std::pair<QString, QVector3D>>();

        m_base_object->scaleAbsolute(s);

        QVector3D trans = m_plane->center();
        trans.setZ(m_base_object->center().z() - m_base_object->minimum().z());
        m_base_object->translateAbsolute(trans);

        this->update();

        // Return the updated part scales.
        QVector<std::pair<QString, QVector3D>> ret;
        for (auto& gop : m_emboss_objects) {
            ret.append(std::make_pair(gop->name(), gop->scaling()));
        }

        return ret;
    }

    void EmbossView::scaleSelectedPart(QVector3D s) {
        if (m_selected_object.isNull()) return;

        m_selected_object->scaleAbsolute(s);

        this->update();
    }

    void EmbossView::selectPart(QString name) {
        if (!m_selected_object.isNull()) m_selected_object->unselect();

        for (auto& gop : m_emboss_objects) {
            if (name == gop->name()) {
                m_selected_object = gop;
                break;
            }
        }

        m_selected_object->select();

        this->update();
    }

    QSharedPointer<Part> EmbossView::getBase() {
        if (m_base_object.isNull()) return nullptr;

        for (auto& child : m_base_object->part()->children()) {
            m_base_object->part()->orphanChild(child);
        }

        for (auto& gop : m_emboss_objects) {
            auto p = gop->part();

            QVector3D    gop_translation = gop->translation();
            QQuaternion  gop_rotation    = gop->rotation();
            QVector3D    gop_scale       = gop->scaling();

            gop_translation *= Constants::OpenGL::kViewToObject;

            QMatrix4x4 object_transformation = MathUtils::composeTransformMatrix(gop_translation, gop_rotation, gop_scale);

            p->setTransformation(object_transformation);

            m_base_object->part()->adoptChild(p);
        }

        QVector3D    base_translation = m_base_object->translation();
        QQuaternion  base_rotation    = m_base_object->rotation();
        QVector3D    base_scale       = m_base_object->scaling();

        base_translation *= Constants::OpenGL::kViewToObject;

        QMatrix4x4 base_transformation = MathUtils::composeTransformMatrix(base_translation, base_rotation, base_scale);

        m_base_object->part()->setTransformation(base_transformation);

        return m_base_object->part();
    }

    void EmbossView::initView() {
        int lp_len = 1000;

        m_grid = QSharedPointer<GridObject>::create(this, lp_len, lp_len, lp_len / 400, lp_len / 400);

        m_plane = QSharedPointer<PlaneObject>::create(this, lp_len, lp_len, QColor(128, 128, 128, 102));
        m_plane->setUnderneath(true);

        this->addObject(m_grid);
        this->addObject(m_plane);
    }

    void EmbossView::handleMouseMove(QPointF mouse_ndc_pos) {
        auto picked_part = this->pickPart(mouse_ndc_pos, m_emboss_objects);

        QCursor c = QCursor(Qt::ArrowCursor);

        if (!picked_part.isNull() && m_selected_object == picked_part) {
            c = QCursor(Qt::OpenHandCursor);
        }

        // Highlighting of objects.
        if (m_state.highlighted_part.isNull() && !picked_part.isNull()) {
            picked_part->highlight();
            m_state.highlighted_part = picked_part;
            this->update();
        }
        else if (!m_state.highlighted_part.isNull() && m_state.highlighted_part != picked_part) {
            m_state.highlighted_part->unhighlight();
            m_state.highlighted_part.reset();
            this->update();
        }

        this->setCursor(c);
        if (m_state.translating) {
            m_state.translating = false;
        }
    }

    void EmbossView::handleLeftClick(QPointF mouse_ndc_pos) {
        auto picked_part = this->pickPart(mouse_ndc_pos, m_emboss_objects);

        if (!m_selected_object.isNull()) {
            if (picked_part != m_selected_object) {
                m_selected_object->unselect();
                m_selected_object.reset();

                emit selected("");
            }
            else {
                m_state.translating = true;
                this->setCursor(QCursor(Qt::ArrowCursor));
            }
        }

        this->update();
    }

    void EmbossView::handleLeftDoubleClick(QPointF mouse_ndc_pos) {
        auto picked_part = this->pickPart(mouse_ndc_pos, m_emboss_objects);

        if (!picked_part.isNull()) {
            // Unselect
            if (m_selected_object == picked_part) {
                picked_part->unselect();
                m_selected_object.reset();
            }
            // Select
            else {
                m_selected_object = picked_part;
                this->setCursor(QCursor(Qt::OpenHandCursor));
                m_selected_object->select();
                m_state.translating = true;

                emit selected(m_selected_object->name());
            }
        }

        this->update();
    }

    void EmbossView::handleLeftMove(QPointF mouse_ndc_pos) {
        if (m_selected_object.isNull()) return;
        if (!m_state.translating) return;

        QVector3D intersect;
        Triangle tri;
        std::tie(std::ignore, tri, intersect) = PartPicker::pickDistanceTriangleAndIntersection(this->projectionMatrix(), this->viewMatrix(), mouse_ndc_pos, m_base_object->triangles());

        // If there's no intersection, there is no move.
        if (std::fabs(intersect.x()) == Constants::Limits::Maximums::kInfFloat) return;
        if (std::fabs(intersect.y()) == Constants::Limits::Maximums::kInfFloat) return;
        if (std::fabs(intersect.z()) == Constants::Limits::Maximums::kInfFloat) return;

        // Find the normal of the picked triangle.
        QVector3D u = tri.b - tri.a;
        QVector3D v = tri.c - tri.a;
        QVector3D norm = QVector3D::normal(u, v);

        m_selected_object->rotateAbsolute(QQuaternion::fromDirection(norm, QVector3D(0, 0, 1)));
        //m_selected_object->rotateAbsolute(m_state.part_rot_normal[m_selected_object] * QQuaternion::fromDirection(norm, QVector3D(0, 0, 1)));

        // Actual translation.
        m_selected_object->translateAbsolute(intersect);

        this->update();
    }

    void EmbossView::handleLeftRelease(QPointF mouse_ndc_pos) {
        m_state.translating = false;

        this->update();
    }

    void EmbossView::handleRightClick(QPointF mouse_ndc_pos, QPointF global_pos) {
        auto p = this->pickPart(mouse_ndc_pos, m_emboss_objects);
        if (p.isNull() || p != m_selected_object) {
            this->BaseView::handleRightClick(mouse_ndc_pos, global_pos);
            return;
        }

        m_state.rotating = true;
        m_state.rotate_start = mouse_ndc_pos;

        m_state.part_rot_start = p->rotation();
        p->axes()->show();

        m_selected_object = p;
    }

    void EmbossView::handleRightMove(QPointF mouse_ndc_pos) {
        if (!m_state.rotating) {
            this->BaseView::handleRightMove(mouse_ndc_pos);
            return;
        }

        QPointF delta = m_state.rotate_start - mouse_ndc_pos;

        // Actual rotation
        QVector3D r = QVector3D(delta.y() * 90, 0, -delta.x() * 90);

        QMatrix4x4 view_mtrx = this->viewMatrix();

        QVector3D right = QVector3D(view_mtrx(0,0), view_mtrx(0,1), view_mtrx(0,2));
        QVector3D up = QVector3D(0, 0, 1);

        right.setZ(0);
        right.normalize();

        QQuaternion qr = QQuaternion::fromAxisAndAngle(right, r.x());
        qr *= QQuaternion::fromAxisAndAngle(up, r.z());

        //qDebug() << "Applied rotation" << qr.toEulerAngles();

        m_selected_object->rotateAbsolute(qr * m_state.part_rot_start);

        this->update();
    }

    void EmbossView::handleRightRelease(QPointF mouse_ndc_pos, QPointF global_pos) {
        if (!m_state.rotating) {
            this->BaseView::handleRightRelease(mouse_ndc_pos, global_pos);
            return;
        }

        QQuaternion r = m_selected_object->rotation();
        QVector3D er = r.toEulerAngles();

        // Snap to intervals of 15 degrees.
        er.setX(MathUtils::snap(er.x(), 15));
        er.setY(MathUtils::snap(er.y(), 15));
        er.setZ(MathUtils::snap(er.z(), 15));

        QQuaternion sr = QQuaternion::fromEulerAngles(er);

        if (std::fabs(sr.z()) >= 90) {
            sr.setX(sr.x() + 180.f);
            sr.setY(180.f - sr.y());
            sr.setZ(sr.z() + 180.f);
        }

        m_selected_object->rotateAbsolute(QQuaternion::fromEulerAngles(er));
        m_selected_object->axes()->hide();

        // Store the normal for the part to keep in the same orientation while dragging.
        //QQuaternion q = m_selected_object->rotation() * m_state.part_rot_start.inverted();
        //m_state.part_rot_normal[m_selected_object] = q * m_state.part_rot_normal[m_selected_object].inverted();

        m_state.rotating = false;
        m_state.part_rot_start = QQuaternion();
        m_state.rotate_start = QPointF();

        this->update();
    }

    void EmbossView::translateCamera(QVector3D v, bool absolute) {
        if (absolute) {
            m_grid->translateAbsolute(v);
            m_plane->translateAbsolute(v);
        }
        else {
            m_grid->translate(v);
            m_plane->translate(v);
        }
    }

    QSharedPointer<PartObject> EmbossView::pickPart(const QPointF& mouse_ndc_pos, QSet<QSharedPointer<PartObject> > object_set) {
        float min_dist = Constants::Limits::Maximums::kMaxFloat;

        QSharedPointer<PartObject> picked_part;

        for(auto& gop : object_set) {
            float dist = PartPicker::pickDistance(this->projectionMatrix(), this->viewMatrix(), mouse_ndc_pos, gop->triangles());

            if (dist < min_dist) {
                min_dist = dist;
                picked_part = gop;
            }
        }

        return picked_part;
    }
}
