#ifndef CLOSEDMESH_H
#define CLOSEDMESH_H

#include "geometry/mesh/mesh_base.h"
#include "geometry/mesh/advanced/mesh_types.h"
#include "geometry/polygon.h"
#include "geometry/polyline.h"
#include "geometry/segments/line.h"
#include "geometry/plane.h"
#include "geometry/mesh/open_mesh.h"

//from CGAL Example - clean later
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>
#include <CGAL/Surface_mesh_shortest_path.h>
#include <CGAL/Random.h>
#include <boost/lexical_cast.hpp>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iterator>
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Polyhedron_3<Kernel, CGAL::Polyhedron_items_with_id_3> Triangle_mesh3;
typedef CGAL::Surface_mesh_shortest_path_traits<Kernel, Triangle_mesh3> Traits3;
typedef CGAL::Surface_mesh_shortest_path<Traits3> Surface_mesh_shortest_path3;
typedef boost::graph_traits<Triangle_mesh3> Graph_traits3;
typedef Graph_traits3::vertex_iterator vertex_iterator3;
typedef Graph_traits3::face_iterator face_iterator3;

namespace ORNL
{
    //! \class ClosedMesh
    //! \brief A class that represents a closed 3D volume
    class ClosedMesh : public MeshBase
    {
    public:
        //! \brief Default constructor
        ClosedMesh();

        //! \brief Full constructor
        //! \param name the name
        //! \param path the file path
        //! \param vertices the vertices
        //! \param faces the faces
        //! \param type the type
        ClosedMesh(const QString& name, const QString& path,
                   const QVector<MeshVertex> &vertices, const QVector<MeshFace> &faces,
                   MeshType type = MeshType::kBuild);

        //! \brief Contructor for vertices and faces
        //! \param vertices the vertices
        //! \param faces the faces
        ClosedMesh(const QVector<MeshVertex> &vertices, const QVector<MeshFace> &faces);

        //! \brief Contructor to build a mesh from the CGAL type
        //! \param poly the CGAL polyhedron
        //! \param name the name
        //! \param file the file path
        ClosedMesh(MeshTypes::Polyhedron poly, QString name = QString(), QString file = QString());

        //! \brief Copy contructor
        //! \param mesh
        ClosedMesh(QSharedPointer<ClosedMesh> mesh);

        //! \brief returns the internal cgal type for this mesh
        //! \return a cgal polyhedron
        MeshTypes::Polyhedron polyhedron();

        //! \brief centers this mesh about its-self
        void center() override;

        //! \brief computes the optimal(tight) bouding box for this part and return the 8 points that form it
        //! \note this returns the tightest bounding box, \see boundingBox() for axis-aligned version
        //! \return eight points that form the bouding box
        QVector<Point> optimalBoundingBox() override;

        //! \brief computes the axis-aligned bouding box for this part and return the 8 points that form it
        //! \note this returns axis-aligned bounding box, \see optimalBoundingBox() for tight bounding box
        //! \return eight points that form the bouding box
        QVector<Point> boundingBox() override;

        //! \brief computes the surface area for the mesh
        //! \return the surface area
        Area area() override;

        //! \brief computes the volume of the mesh
        //! \return the volume of the mesh
        Volume volume();

        //! \brief computes intersections between a line (defined by two points) and this mesh
        //! \param start the start point
        //! \param end the end point
        //! \return a list of intersections
        QVector<Point> intersect(Point start, Point end);

        //! \brief computes intersections between a line segment and this mesh
        //! \param line the line
        //! \return a list of intersections
        QVector<Point> intersect(LineSegment line);

        //! \brief computes all intersections between a polyline and this mesh
        //! \param line the polyline
        //! \return a list of intersections
        QVector<Point> intersect(Polyline line);

        //! \brief computes all intersections between a path and this mesh
        //! \param path the path
        //! \return a list of intersections
        QVector<Point> intersect(Path path);

        //! \brief computes all intersections between a polygon and this mesh
        //! \param poly the polygon
        //! \return a list of intersections
        QVector<Point> intersect(Polygon poly);

        //! \brief intersects a plane with this mesh
        //! \note this can result in points, polylines, or polygons
        //! \param plane the plane to intersect with
        //! \return a list of polylines (includes single points) and polygons that result
        std::pair<QVector<Polyline>, QVector<Polygon>> intersect(Plane plane) override;

        //! \brief subtracts the clipper mesh from this
        //! \param clipper: the mesh to subtract
        void difference(ClosedMesh& clipper);

        //! \brief intersects mesh with this
        //! \param mesh_to_intersect: mesh to intersect with this
        void intersection(ClosedMesh mesh_to_intersect);

        //! \brief unions mesh with this
        //! \param mesh_to_union: mesh to union with this
        void mesh_union(ClosedMesh mesh_to_union);

        //! \brief converts this mesh to an open mesh
        //! \return a pointer to the new open mesh
        QSharedPointer<OpenMesh> toOpenMesh();

        //! \brief divideMeshAlongPlane: splits a mesh into two sections along a curve
        //! \pre Must be a triangulated and closed mesh
        //! \param plane: the plane to cut with
        //! \return the first mesh is the negative side of the plane and the second is the positive
        std::pair<ClosedMesh, ClosedMesh> splitWithPlane(Plane plane);

        //! \brief getCrossSectionalArea finds the cross sectional area of a plane though a mesh.
        //! \note this also reports if it intersects with more then one section of the this.
        //!       This is used to detect if it intersected with a printed section.
        //! \pre Must be a triangulated and closed mesh
        //! \param plane: the plane to cut the body with
        //! \return a bool that reports if it intersects with more then one section and the area
        std::pair<bool, Area> crossSectionalArea(Plane plane, QVector<Polygon> boundary_curves);

        //! \brief converts mesh vertices and mesh faces to a CGAL polyhedron
        //! \pre Must be a triangulated and closed mesh
        //! \return CGAL polyhedron is 3D cartesian space
        static MeshTypes::Polyhedron PolyhedronFromVerticesAndFaces(QVector<MeshVertex>& vertices, QVector<MeshFace>& faces);

        //! \brief toFacesAndVertices: converts a CGAL mesh to faces and vertices
        //! \pre Must be a triangulated and closed mesh
        //! \param mesh: the CGAL mesh polyhedron to be converted
        //! \return a list of vertices and faces that represent the object as a pair
        static std::pair<QVector<MeshVertex>, QVector<MeshFace>> FacesAndVerticesFromPolyhedron(MeshTypes::Polyhedron& mesh);

        //! \brief cleans a supplied polyhedron
        //! \param polyhedron the object to clean
        static void CleanPolyhedron(MeshTypes::Polyhedron& polyhedron);

        MeshTypes::SurfaceMesh extractUpwardFaces() override;

        std::vector<Traits3::Point_3> shortestPath();
        void Sandbox();

    private:
        //! \brief converts vertices and faces into polyhedron, used to keep the two in sync
        void convert() override;

        //! \brief a utility function to check if a polygon curve intersects with a plane
        //! \param plane: the plane who's point we will compare
        //! \param boundary_curves: the list of curves to check
        //! \return if the curve intersected the plane
        static bool CheckIntersectingCurves(Plane& plane, QVector<Polygon> boundary_curves);

        //! \brief Instructs CGAL how to build a polyhedron mesh from vertices and faces
        template <class HDS>
        class MeshBuilder : public CGAL::Modifier_base<HDS>
        {
            public:
                //! \brief Builds a polyhedron incrementally using faces
                //! \param vertices: the mesh's vertices
                //! \param faces: the mesh's faces
                MeshBuilder(QVector<MeshVertex>& vertices, QVector<MeshFace>& faces) : m_vertices(vertices), m_faces(faces) {};
                void operator()(HDS& hds) {
                    CGAL::Polyhedron_incremental_builder_3<HDS> builder(hds, true);

                    builder.begin_surface(m_vertices.length(), m_faces.length());

                    // Add all vertices
                    for(MeshVertex v : m_vertices)
                    {
                        builder.add_vertex(v.toPoint3());
                    }

                    // Add all faces and connect to vertices
                    for(MeshFace f : m_faces)
                    {
                        builder.begin_facet();
                        builder.add_vertex_to_facet(f.vertex_index[0]);
                        builder.add_vertex_to_facet(f.vertex_index[1]);
                        builder.add_vertex_to_facet(f.vertex_index[2]);
                        builder.end_facet();
                    }

                    builder.end_surface();
                }
            private:
                QVector<MeshVertex> m_vertices;
                QVector<MeshFace> m_faces;
        };

        //! \brief the CGAL representation of this mesh
        MeshTypes::Polyhedron m_representation;

        //! \brief the original CGAL representation of this mesh
        MeshTypes::Polyhedron m_original_representation;
    };
}

#endif // CLOSEDMESH_H
