#ifndef CROSSSECTION_H
#define CROSSSECTION_H

// Local
#include "geometry/polygon_list.h"
#include "geometry/plane.h"
#include "geometry/mesh/mesh_base.h"
#include "configs/settings_base.h"

namespace ORNL {
    class CrossSectionSegment;

    //! \brief provides access to functions that cross section meshes
    namespace CrossSection
    {
        //! \brief Computes the cross-section for a mesh and a plane.
        //! \todo Although the external interface to cross section has been simplified, the internals still need some cleanup.
        //! \param QSharedPointer<Mesh> pointer to the mesh to be cross-sectioned
        //! \param Plane* the plane used to generate a cross-section
        //! \param Point* layers must be shifted before rotated, used to return the shift amount to the layer
        //! \param QVector3D* the average normal for the cross-section given the faces that are intersected
        PolygonList doCrossSection(QSharedPointer<MeshBase> mesh, Plane& slicing_plane, Point& shift, QVector3D& averageNormal, QSharedPointer<SettingsBase> sb);

        //! \brief finds the center of the slicing plane within the bounding box
        //! \param mesh the mesh
        //! \param slicing_plane the plane
        //! \return the center
        Point findSlicingPlaneMidPoint(QSharedPointer<MeshBase> mesh, Plane& slicing_plane);

        //! \brief finds intersection of segment between two points and a plane
        //! \param vertex0 the start point of the segment
        //! \param vertex1 the end point of the segment
        //! \param slicing_plane the plane
        //! \return the intersection point
        Point findIntersection(Point& vertex0, Point& vertex1, Plane& slicing_plane);
    }
}

#endif // CROSSSECTION_H
