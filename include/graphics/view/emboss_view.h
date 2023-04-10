#ifndef EMBOSS_VIEW_H
#define EMBOSS_VIEW_H

// Local
#include <graphics/base_view.h>

namespace ORNL {
    // Forward
    class Part;
    class PartObject;
    class PlaneObject;
    class GridObject;

    /*!
     * \brief View that allows for parts to be translated across a base object's surface.
     *
     * This view is used as a preprocessor for the part view in the emboss import dialog. Embossed objects are manipulated
     * in this view and locked when added to the part view.
     */
    class EmbossView : public BaseView {
        Q_OBJECT
        public:
            //! \brief Constructor.
            EmbossView();

        public slots:
            //! \brief Clears the view of all objects.
            void clear();

            //! \brief Sets the part to be used as a base.
            void setBasePart(QSharedPointer<Part> p);
            //! \brief Adds embossing parts to the view. Added objects appear on the front of the base object.
            //!        This function cannot be used until a base object is set.
            void addEmbossPart(QSharedPointer<Part> p);

            //! \brief Scales the base part to the passed value.
            //! \return Vector of scale changes to embossed parts.
            //! \todo Use model instead.
            QVector<std::pair<QString, QVector3D>> scaleBasePart(QVector3D s);
            //! \brief Scales the selected emboss part.
            void scaleSelectedPart(QVector3D s);
            //! \brief Selects the part with the given name.
            void selectPart(QString name);

            //! \brief Gets the base part with all current manipulations and children.
            QSharedPointer<Part> getBase();

        signals:
            //! \brief Signal that a part was selected.
            void selected(QString name);

        protected:
            //! \brief Initalizes the view with the printer and the associated objects.
            void initView();

            //! \brief Handles the translation of the printer in the volume due to camera movement.
            //! \param v: Translation vector
            //! \param absolute: If this translation is relative to (0, 0, 0).
            void translateCamera(QVector3D v, bool absolute);

            //! \brief Handles the following: Object highlighting
            void handleMouseMove(QPointF mouse_ndc_pos);

            //! \brief Handles the following: Object deselection
            void handleLeftClick(QPointF mouse_ndc_pos);
            //! \brief Handles the following: Object selection
            void handleLeftDoubleClick(QPointF mouse_ndc_pos);
            //! \brief Handles the following: Object translation
            void handleLeftMove(QPointF mouse_ndc_pos);
            //! \brief Handles the following: Translation finalization
            void handleLeftRelease(QPointF mouse_ndc_pos);

            //! \brief Handles the following: Object rotation setup
            void handleRightClick(QPointF mouse_ndc_pos, QPointF global_pos);
            //! \brief Handles the following: Object rotation
            void handleRightMove(QPointF mouse_ndc_pos);
            //! \brief Handles the following: Object rotation snapping and finalization
            void handleRightRelease(QPointF mouse_ndc_pos, QPointF global_pos);

        private:
            //! \brief Picks a part using the mouse cursor.
            //! \param mouse_ndc_pos: Cursor normalized location.
            //! \param object_set: Set of objects to search through.
            QSharedPointer<PartObject> pickPart(const QPointF& mouse_ndc_pos, QSet<QSharedPointer<PartObject>> object_set);

            //! \brief Current view state.
            struct {
                //! \brief Part rotation normals for use with dragging. Unused currently.
                QMap<QSharedPointer<PartObject>, QQuaternion> part_rot_normal;
                //! \brief The currently highlighted part.
                QSharedPointer<PartObject> highlighted_part;

                //! \brief If the view is currently translating or not.
                bool translating = false;

                //! \brief If the view is currently rotating or not.
                bool rotating = false;
                //! \brief Starting mouse position for rotation.
                QPointF rotate_start;
                //! \brief Starting quaternion for part rotation.
                QQuaternion part_rot_start;
            } m_state;

            //! \brief The currently selected object.
            QSharedPointer<PartObject> m_selected_object;

            //! \brief The grid shown underneath the objects.
            QSharedPointer<GridObject> m_grid;
            //! \brief The plane shown underneath the objects.
            QSharedPointer<PlaneObject> m_plane;

            //! \brief The base object for the view.
            QSharedPointer<PartObject> m_base_object;

            //! \brief The objects to be embossed.
            QSet<QSharedPointer<PartObject>> m_emboss_objects;
    };
} // Namespace ORNL

#endif // EMBOSS_VIEW_H
