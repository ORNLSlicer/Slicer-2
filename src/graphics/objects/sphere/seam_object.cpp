#include "graphics/objects/sphere/seam_object.h"

namespace ORNL {
    SeamObject::SeamObject(BaseView* view, QColor color) : SphereObject(view, .25, color, GL_TRIANGLES) {
        this->setOnTop(true);
    }
}
