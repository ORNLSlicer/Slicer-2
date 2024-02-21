#include "geometry/mesh/advanced/parameterization.h"

// Local
#include "managers/gpu_manager.h"

// CGAL
#include <CGAL/boost/graph/copy_face_graph.h>
#include <CGAL/Surface_mesh_parameterization/Two_vertices_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/ARAP_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Error_code.h>
#include <CGAL/Surface_mesh_parameterization/parameterize.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/border.h>

namespace ORNL {

    Parameterization::Parameterization() {}

    Parameterization::Parameterization(MeshTypes::SurfaceMesh mesh, Distance3D dims)
    {
        CGAL::copy_face_graph(mesh, m_sm);

        m_dimensions = dims;

        buildConformalUVMap();
    }

    QPair<bool, Point> Parameterization::fromUV(MeshTypes::Point_2 p)
    {
        MeshTypes::SimpleCartesian::Point_2 point(p.x(), p.y());

        // Find matching faces
        QPair<bool, unsigned long> result;
        if(GPU->use())
        {
            #ifdef NVCC_FOUND
            result = m_gpu->findFace(Point(p));
            #endif
        }else
        {
            result = findFace(point);
        }

        bool found = result.first;
        auto face = static_cast<MeshTypes::SimpleCartesian::SM_FaceDescriptor>(result.second);

        if(!found) // The point was not a face
            return QPair<bool, Point>(false, Point());

        // Extract points
        QVector<MeshTypes::SimpleCartesian::Point_2> points;
        for(auto v_id : vertices_around_face(m_sm.halfedge(face), m_sm))
        {
            points.push_back(m_uv_map[v_id]);
        }
        assert(points.size() == 3);

        // Use Barycentric coords to determine new xyz
        QVector3D barycentric_coords = calculateBarycentricCords(points, point);

        // Extract vertices
        QVector<Point> face_points;
        for(auto v_id : vertices_around_face(m_sm.halfedge(face), m_sm))
        {
            face_points.push_back(Point(m_sm.point(v_id).x(), m_sm.point(v_id).y(), m_sm.point(v_id).z()));
        }

        // Use barycentric coords to determine real point location
        Point transformed_point = calculateRealFromBarycentric(face_points, barycentric_coords);

        return QPair<bool, Point>(true, transformed_point);
    }

    Polygon Parameterization::getBoarderPolygon()
    {
        return m_boarder_polygon;
    }

    Distance3D Parameterization::getDimensions()
    {
        return m_dimensions;
    }

    void Parameterization::buildConformalUVMap()
    {
        // CGAL type used to store the UV coordinates
        typedef MeshTypes::SimpleCartesian::SurfaceMesh::Property_map<MeshTypes::SimpleCartesian::SM_VertexDescriptor, MeshTypes::SimpleCartesian::Point_2>  UV_Map;

        UV_Map uv_map;
        uv_map = m_sm.add_property_map<MeshTypes::SimpleCartesian::SM_VertexDescriptor, MeshTypes::SimpleCartesian::Point_2>("v:uv").first;

        auto boarder = CGAL::Polygon_mesh_processing::longest_border(m_sm).first;

        typedef CGAL::Surface_mesh_parameterization::Two_vertices_parameterizer_3<MeshTypes::SimpleCartesian::SurfaceMesh>  Border_parameterizer;
        typedef CGAL::Surface_mesh_parameterization::ARAP_parameterizer_3<MeshTypes::SimpleCartesian::SurfaceMesh, Border_parameterizer> Parameterizer;

        // Build UV Map
        CGAL::Surface_mesh_parameterization::Error_code err = CGAL::Surface_mesh_parameterization::parameterize(m_sm, Parameterizer(), boarder, uv_map);
        assert(err == 0);

        //! \warning This workaround to copy points from CGAL map to a QT is needed due to a broken copy constructor within CGAL
        //!       this might be fixed in future versions of CGAL.
        for(MeshTypes::SimpleCartesian::SM_VertexDescriptor v : m_sm.vertices())
        {
            m_uv_map[v] = uv_map[v];
        }

        // Parameterize surface mesh's vertices
        MeshTypes::SimpleCartesian::SurfaceMesh parameterized_sm;
        CGAL::copy_face_graph(m_sm, parameterized_sm);
        for(auto v : parameterized_sm.vertices())
        {
            MeshTypes::SimpleCartesian::Point_3 new_point(m_uv_map[v].x(), m_uv_map[v].y(), 0);
            parameterized_sm.point(v) = new_point;
        }

        // Extract boarder points to build polygon
        QVector<Point> boarder_points;
        auto h = boarder;
        do
        {
            auto t = parameterized_sm.point(CGAL::target(h, parameterized_sm));
            boarder_points.push_back(Point(t.x(), t.y(), t.z()));
            h = CGAL::next(h, parameterized_sm);
        }while(h != boarder);

        m_boarder_polygon = Polygon(boarder_points);

        // Scale polygon back to real space
        for(Point& p : m_boarder_polygon)
        {
            p.x(p.x() * m_dimensions.x());
            p.y(p.y() * m_dimensions.y());
        }

        // Store on GPU
        if(GPU->use())
        {
            #ifdef NVCC_FOUND
            QVector<CUDA::GPUFace> host_faces;

            for(auto face : CGAL::faces(m_sm))
            {
                QVector<ORNL::MeshTypes::SimpleCartesian::Point_2> points;
                for(auto v_id : vertices_around_face(m_sm.halfedge(face), m_sm))
                {
                    points.push_back(m_uv_map[v_id]);
                }

                ORNL::Point a(points[0].x(), points[0].y(), 0);
                ORNL::Point b(points[1].x(), points[1].y(), 0);
                ORNL::Point c(points[2].x(), points[2].y(), 0);

                ORNL::MeshTypes::SimpleCartesian::SM_FaceDescriptor face_id = face;

                CUDA::GPUFace new_face(a, b, c, static_cast<unsigned long>(face));
                host_faces.push_back(new_face);
            }

            m_gpu = QSharedPointer<CUDA::GPUFaceFinder>::create(host_faces);
            #endif
        }
    }

    double Parameterization::calculateAreaofTriangle(MeshTypes::SimpleCartesian::Point_2 p0, MeshTypes::SimpleCartesian::Point_2 p1, MeshTypes::SimpleCartesian::Point_2 p2)
    {
        return qFabs(((p0.x() * (p1.y() - p2.y()) +
                       p1.x() * (p2.y() - p0.y()) +
                       p2.x() * (p0.y() - p1.y())) / 2.0));
    }

    QVector3D Parameterization::calculateBarycentricCords(QVector<MeshTypes::SimpleCartesian::Point_2> triangle, MeshTypes::SimpleCartesian::Point_2 point)
    {
        float tri_area = calculateAreaofTriangle(triangle[0], triangle[1], triangle[2]);
        float u = calculateAreaofTriangle(point, triangle[1], triangle[2]) / tri_area;
        float v = calculateAreaofTriangle(triangle[0], point, triangle[2]) / tri_area;
        float w = calculateAreaofTriangle(triangle[0], triangle[1], point) / tri_area;

         QVector3D bary;
         bary[0] = u;
         bary[1] = v;
         bary[2] = w;

         return bary;
    }

    Point Parameterization::calculateRealFromBarycentric(QVector<Point> triangle, QVector3D barycentric_cords)
    {
        Point ua = triangle[0] * barycentric_cords[0];
        Point vb = triangle[1] * barycentric_cords[1];
        Point wc = triangle[2] * barycentric_cords[2];

        return ua + vb + wc;
    }

    QPair<bool, unsigned long> Parameterization::findFace(MeshTypes::SimpleCartesian::Point_2 point)
    {
        auto face = std::find_if(CGAL::faces(m_sm).begin(), CGAL::faces(m_sm).end(), [&point, this](MeshTypes::SimpleCartesian::SM_FaceDescriptor face)
        {
            // Extract vertices
            QVector<MeshTypes::SimpleCartesian::Point_2> points;
            for(auto v_id : vertices_around_face(m_sm.halfedge(face), m_sm))
            {
                points.push_back(m_uv_map[v_id]);
            }
            assert(points.size() == 3);

            // Determine barycentric coords
            QVector3D barycentric_cords = calculateBarycentricCords(points, point);

            double area = (barycentric_cords.x() + barycentric_cords.y() + barycentric_cords.z());

            // Fuzzy value of 0.001 is used here
            return area <= 1.001;
        });

        bool found = face != CGAL::faces(m_sm).end();

        return QPair<bool, unsigned long>(found, static_cast<unsigned long>(*face));
    }
}
