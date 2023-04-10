#ifndef PREVIEW_VIEW_H
#define PREVIEW_VIEW_H

#include "graphics/base_view.h"

// Local
#include "units/unit.h"

namespace ORNL {
    // Forward
    class Part;
    class GridObject;
    class PlaneObject;
    class PartObject;

    /*!
     * \brief A simple view that displays a single part and optionally its slicing plane.
     *
     * PreviewView is intended to be used in loading dialogs and other obvious locations.
     */
    class PreviewView : public BaseView {
        Q_OBJECT
        public:
            //! \brief Constructor.
            PreviewView();

        public slots:
            //! \brief Sets the part to be shown.
            //! \param p: Part.
            void setPart(QSharedPointer<Part> p);

            //! \brief Clear displayed part.
            void clearPart();

            //! \brief Shows the slicing plane for the loaded part.
            void showSlicingPlane(bool show);

            //! \brief Set slicing plane rotation.
            void setSlicingPlaneRotation(Angle pitch, Angle yaw, Angle roll);

        protected:
            //! \brief Initalize the view (floor and grid).
            void initView();

            //! \brief There is no interactivity with the preview besides camera manipulation.
            //!        This function does nothing.
            void handleLeftClick(QPointF mouse_ndc_pos);

        private:
            //! \brief Grid shown below object.
            QSharedPointer<GridObject> m_grid;
            //! \brief Plane shown below object.
            QSharedPointer<PlaneObject> m_plane;

            //! \brief Shown object.
            QSharedPointer<PartObject> m_gop;
    };
}

#endif // PREVIEW_VIEW_H
