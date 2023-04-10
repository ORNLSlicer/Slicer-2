#ifndef MESHSKELETON_H
#define MESHSKELETON_H

// CGAL Includes
#include <CGAL/extract_mean_curvature_flow_skeleton.h>

// Local Includes
#include "geometry/mesh/closed_mesh.h"
#include "geometry/plane.h"
#include "geometry/point.h"

namespace ORNL {
    //! \brief a class that calculates the mean curvature flow skeleton on a mesh
    //!        this is done in the following order:
    //!         1. Compute skeleton
    //!         2. Order skeleton points based on Z- > X -> Y height
    //!         3. Extend skeleton to edges of mesh using ray traces
    //!         4. Fetching the approximate distance along a bezier curve fit
    //!
    class MeshSkeleton
    {
    public:
        //! \brief CGAL type
        typedef CGAL::Mean_curvature_flow_skeletonization<MeshTypes::SimpleCartesian::Polyhedron> Skeletonization;

        //! \brief Default Constructor
        MeshSkeleton();

        //! \brief Constructor
        //! \param a closed triangulated mesh
        //! \pre the mesh must be triangulated, closed and not contain self intersections
        MeshSkeleton(QSharedPointer<ClosedMesh> mesh);

        //! \brief Constructor
        //! \param a closed triangulated mesh
        //! \pre the mesh must be triangulated, closed and not contain self intersections
        MeshSkeleton(MeshTypes::Polyhedron mesh);

        //! \brief computes the MCF skeletonization of a mesh
        void compute();

        //! \brief orders the points of the computed skeleton so a bezier can be calculated
        //! \note this is done in this order: Z -> X -> Y
        void order();

        //! \brief extends the skeleton curves to the edges of the part
        //! \note this is done with 2 ray traces that are project along the normal vector of the start and end planes
        void extend();

        //! \brief Gets a the final approximate plane on the mesh
        //! \note this the normal points away from the skeleton
        //! \return the final plane
        Plane getFinalPlane();

        //! \brief Gets a the first approximate plane on the mesh
        //! \note this the normal points away from the skeleton
        //! \return the first plane
        Plane getFirstPlane();

        //! \brief Gets the first point on the skeleton curve
        //! \return the first skeleton point
        Point getFirstPoint();

        //! \brief Gets the last point on the skeleton curve
        //! \return the last skeleton point
        Point getLastPoint();

        //! \brief Returns the internal CGAL skeleton structure
        //! \returns the internal CGAL type
        Skeletonization::Skeleton getSkeleton();

        //! \brief Sets the last plane for the skeleton
        //! \param p: the last plane
        void setPlane(Plane& p);

        //! \brief Iterates through the skeleton curve, calculating the plane at an approximate distance
        //! \param d: the distance to move along the curve
        //! \return the plane at that point
        Plane findNextPlane(Distance d);

    private:
        //! \brief Default mesh type
        QSharedPointer<ClosedMesh> m_mesh;

        //! \brief The CGAL mesh to find the skeleton of
        MeshTypes::SimpleCartesian::Polyhedron m_polyhedron;

        //! \brief The points that make of the skeleton
        QVector<Point> m_skeleton;

        //! \brief Internal CGAL type for skeleton
        Skeletonization::Skeleton m_skeletonization;

        //! \brief the last plane we iterate through
        Plane m_last_plane;

        //! \brief the time on the interval from 0 to 1 along the bezier curve
        double m_time = 0.0;

        //! \brief Gets a point on the skeleton's bezier curve
        //! \param a value on the interval 0 to 1 that represents interpolation along the curve
        //! \return the approximate point on the bezier curve at time t
        Point getPointOnBezierCurve(double t);

    };
}
#endif // MESHSKELETON_H
