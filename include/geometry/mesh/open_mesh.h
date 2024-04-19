#ifndef OPENMESH_H
#define OPENMESH_H

#include "geometry/mesh/mesh_base.h"
#include "geometry/mesh/advanced/mesh_types.h"
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Surface_mesh_shortest_path.h>
#include <boost/foreach.hpp>
#include <Eigen/Dense>

// Define CGAL kernel and types
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;
typedef K::Vector_3 Vector_3;
typedef Mesh::Property_map<Mesh::Vertex_index, double> Vertex_distance_map;

//from CGAL Example - clean later
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_shortest_path.h>
#include <CGAL/Random.h>
#include <boost/lexical_cast.hpp>
#include <cstdlib>
#include <iostream>
#include <fstream>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Surface_mesh<Kernel::Point_3> Triangle_mesh;
typedef CGAL::Surface_mesh_shortest_path_traits<Kernel, Triangle_mesh> Traits;
typedef CGAL::Surface_mesh_shortest_path<Traits> Surface_mesh_shortest_path;
typedef boost::graph_traits<Triangle_mesh> Graph_traits;
typedef Graph_traits::vertex_iterator vertex_iterator;
typedef Graph_traits::face_iterator face_iterator;

namespace ORNL
{
    //! \class OpenMesh
    //! \brief A class that represents an open 3D volume
    class OpenMesh : public MeshBase
    {
    public:
        //! \brief Default constructor
        OpenMesh();

        //! \brief Full constructor
        //! \param name the name
        //! \param path the file path
        //! \param vertices the vertices
        //! \param faces the faces
        //! \param type the type
        OpenMesh(const QString& name, const QString& path,
                   const QVector<MeshVertex> &vertices, const QVector<MeshFace> &faces,
                   MeshType type = MeshType::kBuild);

        //! \brief Contructor for vertices and faces
        //! \param vertices the vertices
        //! \param faces the faces
        OpenMesh(const QVector<MeshVertex> &vertices, const QVector<MeshFace> &faces);

        //! \brief Contructor to build a mesh from the CGAL type
        //! \param poly the CGAL polyhedron
        //! \param name the name
        //! \param file the file path
        OpenMesh(MeshTypes::SurfaceMesh poly, QString name = QString(), QString file = QString());

        //! \brief Copy contructor
        //! \param mesh
        OpenMesh(QSharedPointer<OpenMesh> mesh);

        //! \brief returns the internal cgal mesh for this object
        //! \return the cgal surface mesh
        MeshTypes::SurfaceMesh surface_mesh();

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

        //! \brief makes a surface mesh of just the upward normal faces
        //! \return a surface mesh
        MeshTypes::SurfaceMesh extractUpwardFaces() override;

        //! \brief intersects a plane with this mesh
        //! \note this can result in points, polylines, or polygons
        //! \param plane the plane to intersect with
        //! \return a list of polylines (includes single points) and polygons that result
        std::pair<QVector<Polyline>, QVector<Polygon>> intersect(Plane plane) override;

        //! \brief converts vertices and faces to a surface mesh
        //! \param vertices the vertices
        //! \param faces the faces
        //! \return a surface mesh
        static MeshTypes::SurfaceMesh SurfaceMeshFromVerticesAndFaces(const QVector<MeshVertex> &vertices, const QVector<MeshFace> &faces);

        //! \brief converts a surface mesh into vertices and faces
        //! \param sm the surface mesh
        //! \return a pair of vertices and faces
        static std::pair<QVector<MeshVertex>, QVector<MeshFace>> VerticesAndFacesFromSurfaceMesh(MeshTypes::SurfaceMesh &sm);

        //! \brief Builds and returns a surface mesh from a point cloud
        //! \param file_path the file to build the cloud from
        //! \return a pointer to the new mesh if it could be loaded
        static QSharedPointer<OpenMesh> BuildMeshFromPointCloud(const QString& file_path);

        void Sandbox();
        std::vector<Traits::Point_3> shortestPath();

    private:
        //! \brief converts vertices and faces into polyhedron, used to keep the two in sync
        void convert() override;

        //! \brief loads points from a .matrix file
        //! \param file_path the file
        //! \return points in a point cloud
        static QVector<MeshTypes::Point_3> loadMatrixFile(const QString& file_path);

        //! \brief loads points from a .xyz file
        //! \param file_path the file
        //! \return points in a point cloud
        static QVector<MeshTypes::Point_3> loadXYZFile(const QString& file_path);

        //! \struct ConstructPointCloud
        //! \brief A trait used to build a point cloud mesh
        struct ConstructPointCloud
        {
          MeshTypes::SurfaceMesh& mesh;
          template < typename PointIterator>
          ConstructPointCloud(MeshTypes::SurfaceMesh& mesh,PointIterator b, PointIterator e) : mesh(mesh)
          {
            for(; b != e; ++b){
              boost::graph_traits<MeshTypes::SurfaceMesh>::vertex_descriptor v;
              v = add_vertex(mesh);
              mesh.point(v) = *b;
            }
          }
          ConstructPointCloud& operator=(const std::array<std::size_t,3> f)
          {
            typedef boost::graph_traits<MeshTypes::SurfaceMesh>::vertex_descriptor vertex_descriptor;
            typedef boost::graph_traits<MeshTypes::SurfaceMesh>::vertices_size_type size_type;
            mesh.add_face(vertex_descriptor(static_cast<size_type>(f[0])),
                          vertex_descriptor(static_cast<size_type>(f[1])),
                          vertex_descriptor(static_cast<size_type>(f[2])));
            return *this;
          }
          ConstructPointCloud& operator*() { return *this; }
          ConstructPointCloud& operator++() { return *this; }
          ConstructPointCloud operator++(int) { return *this; }
        };

        //! \brief the CGAL representation of this mesh
        MeshTypes::SurfaceMesh m_representation;

        //! \brief the original CGAL representation of this mesh
        MeshTypes::SurfaceMesh m_original_representation;
    };
}

#endif // OPENMESH_H
