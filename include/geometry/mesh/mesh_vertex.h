#ifndef MESHVERTEX_H
#define MESHVERTEX_H

//Qt
#include <QVector>
#include <QVector3D>

// Libraries
#ifndef __CUDACC__
#include "geometry/mesh/advanced/mesh_types.h"
#endif

namespace ORNL
{
    /*!
     * \class MeshVertex
     *
     * \brief MeshVertex type to be used in a Mesh.
     *
     * Keeps track of which faces connect to it.
     */
    class MeshVertex
    {
    public:
        /*!
         * \brief Constructor
         *
         * \param location The location of this vertex
         * \param normal The normal vector for this vertex
         */
        MeshVertex(const QVector3D& location = QVector3D(), const QVector3D& normal = QVector3D());

        // Transform this vertex by applying the passed matrix.
        void transform(const QMatrix4x4& transform);

        #ifndef __CUDACC__
        //! \brief Converts the mesh vertex to a CGAL cartesian point
        MeshTypes::Point_3 toPoint3() const;
        #endif

        QVector3D location;  //!< Location of the vertex
        QVector3D normal;
        QVector<int> connected_faces;  //!< list of the indices of connected faces
    };

    /*!
     * \brief comparison operator used for QMap
     * \param lhs Left side of the less than operator
     * \param rhs Right side of the less then operator
     * \return Whether \p lhs is less than \p rhs
     */
//    bool operator<(const MeshVertex& lhs, const MeshVertex& rhs);
}  // namespace ORNL
#endif  // MESHVERTEX_H
