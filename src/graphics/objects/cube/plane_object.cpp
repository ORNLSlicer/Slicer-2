#include "graphics/objects/cube/plane_object.h"

// Local
#include "graphics/support/shape_factory.h"

namespace ORNL {
    PlaneObject::PlaneObject(BaseView* view, float length, float width, QColor color) : CubeObject(view, length, width, 0.01, color) {
        m_starting_length = length;
        m_starting_width  = width;
        m_color = color;
    }

    void PlaneObject::setLockedRotationAngle(Angle pitch, Angle yaw, Angle roll) {
        m_locked_rotation = QQuaternion::fromEulerAngles(pitch.to(deg), yaw.to(deg), roll.to(deg));
        this->rotateAbsolute(m_locked_rotation);
    }

    void PlaneObject::setLockedRotation(bool lock) {
        m_rotation_toggle = lock;
    }

    void PlaneObject::updateDimensions(float length, float width) {
        this->scaleAbsolute(QVector3D(length / m_starting_length, width / m_starting_width, 1));
    }

    void PlaneObject::transformationCallback() {
        if (m_rotation_toggle) this->rotateAbsolute(m_locked_rotation);
    }
}
