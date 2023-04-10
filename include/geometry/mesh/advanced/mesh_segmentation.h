#ifndef MESH_SEGMENTATION_H
#define MESH_SEGMENTATION_H

// Locals
#include "geometry/mesh/closed_mesh.h"

namespace ORNL
{
    //!
    //! \brief The MeshSegmenter class cuts meshes into printable sub-sections
    //!
    class MeshSegmenter
    {
    public:
        //!
        //! \brief default constructor
        //!
        MeshSegmenter();

        //!
        //! \brief splitMeshIntoSections: divides a mesh into printable sub-sections
        //! \param mesh: the mesh to divide
        //! \return a ordered list of sub-sections
        //!
        QVector<MeshTypes::Polyhedron> splitMeshIntoSections(MeshTypes::Polyhedron mesh);

    private:

        //!
        //! \brief perturbPlane: randomly shifts a plane along its normal and randomly shifts its normal direction
        //! \note this is non-deterministic
        //! \param plane: the plane to shift
        //!
        void perturbPlane(Plane& plane);

        //!
        //! \brief getPlaneOnCurve: builds a plane given a curves.
        //! \note This is non-deterministic
        //! \param curve: a polygon to find the plane for
        //! \return a plane for the curve
        //!
        Plane getPlaneOnCurve(Polygon& curve);

        //!
        //! \brief computeSDFValues: compute diameter functions for a mesh using its skeleton
        //! \param map: the property map to fill
        //! \param mesh: the mesh to compute
        //!
        void computeSDFValues(ClosedMesh::FacetPropMap<double>& map, MeshTypes::Polyhedron& mesh);
    };
}

#endif // MESH_SEGMENTATION_H
