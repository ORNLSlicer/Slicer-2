#include "graphics/objects/sphere_object.h"

// Local
#include "graphics/support/shape_factory.h"

namespace ORNL {
    SphereObject::SphereObject(BaseView* view, float radius, QColor color, ushort render_mode) {
        m_radius = radius;

        std::vector<float> vertices;
        std::vector<float> normals;
        std::vector<float> colors;

        ShapeFactory::createSphere(m_radius, 25, 25, QMatrix4x4(), color, vertices, colors, normals);

        this->populateGL(view, vertices, normals, colors, render_mode);
    }
}
