#ifndef SEAM_OBJECT_H_
#define SEAM_OBJECT_H_

#include "graphics/objects/sphere_object.h"

namespace ORNL {
    /*!
     * \brief Wrapper derived class for the sphere object with a default radius / mode.
     *
     * For use as a seam graphic.
     */
    class SeamObject : public SphereObject {
        public:
            //! \brief Constructor.
            //! \param view: View to render to.
            //! \param color: Color of the object.
            SeamObject(BaseView* view, QColor color);
    };
}

#endif // SEAM_OBJECT_H_
