#ifndef PARTMETAITEM_H_
#define PARTMETAITEM_H_

// Qt
#include <QObject>

// Local
#include "part/part.h"
#include "graphics/objects/part_object.h"

namespace ORNL {
    // Forward
    class PartMetaModel;

    /*!
     * \brief The core of the PartMetaModel: PartMetaItem.
     *
     * Changes to this class are broadcast to the model which are then propagated to all hooked up view classes.
     */
    class PartMetaItem : public QObject, public QEnableSharedFromThis<PartMetaItem> {
        Q_OBJECT

        private:
            //! \enum PartMetaUpdateType
            //! \brief Type of update to model.
            enum class PartMetaUpdateType {
                kAddUpdate,
                kReloadUpdate,
                kRemoveUpdate,
                kParentingUpdate,
                kSelectionUpdate,
                kVisualUpdate,
                kTransformUpdate
            };

        public:
            //! \brief Constructor.
            //! \param p: Part to creat from.
            PartMetaItem(QSharedPointer<Part> p);

            //! \brief Reloads from source STL file this item in the model.
            void reloadInModel();

            //! \brief Removes this item from the model.
            void removeFromModel();

            //! \brief Sets the item selected.
            void setSelected(bool toggle);
            //! \brief Gets selection status.
            bool isSelected();

            //! \brief Sets the mesh type.
            void setMeshType(MeshType mt);
            //! \brief Gets the mesh type.
            MeshType meshType();

            //! \brief Sets the transparency.
            void setTransparency(uint val);
            //! \brief Gets the transparency.
            uint transparency();

            //! \brief sets model to draw as a wireframe
            //! \param show show the wireframe
            void setWireframe(bool show);

            //! \brief sets model to draw as a solid wireframe
            //! \param show show the solidWireframe
            void setSolidWireframe(bool show);

            //! \brief the render mode to use (either normal or triangles)
            //! \return the render mode
            ushort renderMode();

            //! \brief The state of the choice of the shader program this object is using
            //! (m_shader_program vs m_shader_program2)
            //! \return the state of the choice
            bool solidWireframeMode();

            //! \brief The state of whether the object is being rendered in wireframe or not
            //! \return the state of the choice
            bool wireframeMode();

            //! \brief Sets the translation.
            void setTranslation(QVector3D t);
            void translate(QVector3D delta_t);
            //! \brief Gets the translation.
            QVector3D translation();
            //! \brief Sets the rotation.
            void setRotation(QQuaternion r);
            void rotate(QQuaternion delta_r);
            //! \brief Gets the rotation.
            QQuaternion rotation();
            //! \brief Sets the scale.
            void setScale(QVector3D s);
            void scale(QVector3D delta_s);
            //! \brief Gets the scale.
            QVector3D scaling();
            //! \brief Sets the transform.
            void setTransformation(QMatrix4x4 m);
            //! \brief Gets the transform.
            QMatrix4x4 transformation();

            //! \brief resets to the original transform of this object
            void resetTransformation();

            //! \brief Adopts a child item.
            void adoptChild(QSharedPointer<PartMetaItem> c);
            //! \brief Children of this item.
            QList<QSharedPointer<PartMetaItem>> children();

            //! \brief Removes a child.
            void orphanChild(QSharedPointer<PartMetaItem> c);

            //! \brief Sets the parent of this child.
            void setParent(QSharedPointer<PartMetaItem> p);
            //! \brief Parent item.
            QSharedPointer<PartMetaItem> parent();

            //! \brief Sets the index in the scales combo.
            //! \todo Theres a better way to do this.
            void setScaleUnitIndex(uint idx);
            //! \brief Scale unit index.
            uint scaleUnitIndex();

            //! \brief Sets the graphics for this item.
            void setGraphicsPart(QSharedPointer<PartObject> gop);
            //! \brief Gets the graphisc for this item.
            QSharedPointer<PartObject> graphicsPart();

            //! \brief Get the part pointer.
            QSharedPointer<Part> part();

        signals:
            //! \brief Signal that this item was modified.
            void modified(PartMetaUpdateType type);

        private:
            //! \brief Model needs access to update type / model setting.
            friend class PartMetaModel;

            //! \brief Sets the model to use for this item.
            void setModel(QSharedPointer<PartMetaModel> m);

            //! \brief Status
            bool m_selected;
            MeshType m_type;
            uint m_transparency;
            bool m_solid_wireframe_mode = false;
            bool m_wireframe_mode = false;
            ushort m_render_mode = GL_TRIANGLES;
            uint m_scale_unit_index = 0;

            //! \brief Transform
            QVector3D m_translation;
            QQuaternion m_rotation;
            QVector3D m_scale;
            QMatrix4x4 m_transformation;

            //! \brief Parenting
            QSharedPointer<PartMetaItem> m_parent = nullptr;
            QList<QSharedPointer<PartMetaItem>> m_children;

            //! \brief Other representations.
            QSharedPointer<Part> m_part;
            QSharedPointer<PartObject> m_graphics_part;

            //! \brief Model
            QSharedPointer<PartMetaModel> m_model;
    };

}

#endif // PARTMETAITEM_H_
