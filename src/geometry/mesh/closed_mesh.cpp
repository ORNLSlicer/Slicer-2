#include "geometry/mesh/closed_mesh.h"

#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/clip.h>
#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_slicer.h>
#include <CGAL/optimal_bounding_box.h>
#include <CGAL/boost/graph/Face_filtered_graph.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/Polygon_mesh_processing/transform.h>

#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/boost/graph/helpers.h>


#include <CGAL/Shape_detection/Efficient_RANSAC.h>
#include <CGAL/Point_with_normal_3.h>
#include <CGAL/property_map.h>
#include <QTextStream>

typedef ORNL::MeshTypes::Kernel Kernel;
typedef Kernel::FT FT;
typedef std::pair<Kernel::Point_3, Kernel::Vector_3> Point_with_normal;
typedef std::vector<Point_with_normal> Pwn_vector;
typedef CGAL::First_of_pair_property_map<Point_with_normal>  Point_map;
typedef CGAL::Second_of_pair_property_map<Point_with_normal> Normal_map;
typedef CGAL::Shape_detection::Efficient_RANSAC_traits
<Kernel, Pwn_vector, Point_map, Normal_map>             surfTraits;
typedef CGAL::Shape_detection::Efficient_RANSAC<surfTraits> Efficient_ransac;
typedef CGAL::Shape_detection::Cone<surfTraits>             Cone;
typedef CGAL::Shape_detection::Cylinder<surfTraits>         Cylinder;
typedef CGAL::Shape_detection::Plane<surfTraits>            Planer;
typedef CGAL::Shape_detection::Sphere<surfTraits>           Sphere;
typedef CGAL::Shape_detection::Torus<surfTraits>            Torus;

namespace ORNL
{

    std::vector<Traits3::Point_3> ClosedMesh::shortestPath()
    {
        /* shortestPath is an algorithm to compute geodesic shortest paths on a triangulated surface mesh.
         * The algorithm implemented in this package builds a data structure to efficiently answer
         * queries of the following form: Given a triangulated surface mesh M, a set of source points S on
         * M, and a target point t also on M, find a shortest path λ between t and any element in S, where
         * λ is constrained to the surface of M.
         * See https://doc.cgal.org/latest/Surface_mesh_shortest_path/index.html#Chapter_Surface_mesh_shortest_path
         * example 3.2*/

        Triangle_mesh3 tmesh = m_representation;

        // initialize indices of vertices, halfedges and faces
        CGAL::set_halfedgeds_items_id(tmesh);

        /* A source point can be specified using either a vertex of the input surface mesh or a face of the input
         * surface mesh with some barycentric coordinates. Given a point p that lies inside a triangle face (A,B,C),
         * its barycentric coordinates are a weight triple (b0,b1,b2) such that p=b0\cdotA+b1\cdotB+b2\cdotC, and
         * b0+b1+b2=1.*/
        // first, pick a face (this one is random)
        const unsigned int randSeed = 5;
        CGAL::Random rand(randSeed);
        const int target_face_index = rand.get_int(0, static_cast<int>(num_faces(tmesh)));
        face_iterator3 face_it = CGAL::faces(tmesh).first;
        std::advance(face_it,target_face_index);
        // then, define a barycentric coordinates inside the face
        Traits3::Barycentric_coordinates face_location = {{0.25, 0.5, 0.25}}; //CHECK outside of given examples

        // construct a shortest path query object and add a source point
        Surface_mesh_shortest_path3 shortest_paths(tmesh);
        shortest_paths.add_source_point(*face_it, face_location);
        // For all vertices in the tmesh, compute the points of the shortest path to the source point and write
        // them into a file readable using the CGAL Polyhedron demo
        std::ofstream output("shortest_paths_with_id.polylines.txt");
        vertex_iterator3 vit, vit_end;
        std::vector<Traits3::Point_3> points;
        for ( boost::tie(vit, vit_end) = CGAL::vertices(tmesh);
             vit != vit_end; ++vit)
        {
            shortest_paths.shortest_path_points_to_source_points(*vit, std::back_inserter(points));
            // print the points
            output << points.size() << " ";
            for (std::size_t i = 0; i < points.size(); ++i)
                output << " " << points[i];
            output << std::endl;
        }
        return points;
    }

    ClosedMesh::ClosedMesh() : MeshBase()
    {
        m_is_closed = true;
    }

    ClosedMesh::ClosedMesh(const QVector<MeshVertex> &vertices, const QVector<MeshFace> &faces) : MeshBase(vertices, faces)
    {
        m_representation = MeshTypes::Polyhedron(PolyhedronFromVerticesAndFaces(m_vertices, m_faces));
        m_original_representation = MeshTypes::Polyhedron(m_representation); // Create a copy
        updateDims();
        m_is_closed = true;
    }

    ClosedMesh::ClosedMesh(const QString& name, const QString& path,
                           const QVector<MeshVertex> &vertices, const QVector<MeshFace> &faces,
                           MeshType type) : MeshBase(vertices, faces, name, path, type)
    {
        m_representation = MeshTypes::Polyhedron(PolyhedronFromVerticesAndFaces(m_vertices, m_faces));
        m_original_representation = MeshTypes::Polyhedron(m_representation); // Create a copy
        updateDims();
        m_is_closed = true;
    }

    ClosedMesh::ClosedMesh(MeshTypes::Polyhedron poly, QString name, QString file) : MeshBase()
    {
        m_name = name;
        m_file = file;

        CGAL::copy_face_graph(poly, m_representation);

        auto vertices_and_faces = FacesAndVerticesFromPolyhedron(m_representation);
        m_vertices_original = m_vertices = m_vertices_aligned = vertices_and_faces.first;
        m_faces_original = m_faces = m_faces_aligned = vertices_and_faces.second;
        CGAL::copy_face_graph(m_representation, m_original_representation);
        updateDims();
        m_original_dimensions = m_dimensions;
        m_is_closed = true;
    }

    ClosedMesh::ClosedMesh(QSharedPointer<ClosedMesh> mesh) : MeshBase(mesh)
    {
        m_representation = mesh->m_representation;
        m_original_representation = mesh->m_original_representation;
        m_is_closed = true;
    }

    MeshTypes::Polyhedron ClosedMesh::polyhedron()
    {
        return m_representation;
    }

    void ClosedMesh::center()
    {
        //! Compose translation
        CGAL::Aff_transformation_3<MeshTypes::Kernel> translation(CGAL::Translation(), -centroid().toVector_3());

        //! Apply translation to CGAL mesh
        CGAL::Polygon_mesh_processing::transform(translation, m_original_representation);
        CGAL::Polygon_mesh_processing::transform(translation, m_representation);

        //! Update mesh vertices
        size_t i = 0;
        for (const auto &v : m_original_representation.points())
        {
            m_vertices[i].location = m_vertices_aligned[i].location = m_vertices_original[i].location = QVector3D(v.x(), v.y(), v.z());
            i++;
        }
    }

    QVector<Point> ClosedMesh::optimalBoundingBox()
    {
        std::array<MeshTypes::Point_3, 8> obb_points;
        CGAL::oriented_bounding_box(m_representation, obb_points,
                                    CGAL::parameters::use_convex_hull(true));
        QVector<Point> points;
        return points;
    }

    QVector<Point> ClosedMesh::boundingBox()
    {
        MeshTypes::Polyhedron_AABB_Tree tree(CGAL::faces(m_representation).first, CGAL::faces(m_representation).second, m_representation);
        auto box = tree.bbox();

        QVector<Point> points;

        points.push_back(Point(box.xmin(), box.ymin(), box.zmin()));
        points.push_back(Point(box.xmax(), box.ymin(), box.zmin()));
        points.push_back(Point(box.xmax(), box.ymax(), box.zmin()));
        points.push_back(Point(box.xmin(), box.ymax(), box.zmin()));
        points.push_back(Point(box.xmin(), box.ymin(), box.zmax()));
        points.push_back(Point(box.xmax(), box.ymin(), box.zmax()));
        points.push_back(Point(box.xmax(), box.ymax(), box.zmax()));
        points.push_back(Point(box.xmin(), box.ymax(), box.zmax()));

        return points;
    }

    Area ClosedMesh::area()
    {
        double area = 0.0;
        area = CGAL::Polygon_mesh_processing::area(m_representation);
        Area a;
        return a.from(area, (micron * micron));
    }

    Volume ClosedMesh::volume()
    {
        double volume = 0.0;
        volume = CGAL::Polygon_mesh_processing::volume(m_representation);
        Volume v;
        return v.from(volume, (micron * micron * micron));
    }

    QVector<Point> ClosedMesh::intersect(Point start, Point end)
    {
        MeshTypes::Polyhedron_AABB_Tree tree(CGAL::faces(m_representation).first, CGAL::faces(m_representation).second, m_representation);
        MeshTypes::Kernel::Segment_3 segment(start.toCartesian3D(), end.toCartesian3D());

        std::list<MeshTypes::Polyhedron_Segment_intersection> intersections;
        tree.all_intersections(segment, std::back_inserter(intersections));

        QVector<Point> intersection_points;
        for(auto intersection : intersections)
        {
            auto point  = boost::get<MeshTypes::Point_3>(&(intersection->first));
            intersection_points.push_back(Point(*point));
        }
        return intersection_points;
    }

    QVector<Point> ClosedMesh::intersect(LineSegment line)
    {
        return intersect(line.start(), line.end());
    }

    QVector<Point> ClosedMesh::intersect(Polyline line)
    {
        QVector<Point> intersections;
        for(int i = 1, end = line.size(); i < end; ++i)
            intersections.append(intersect(line[i - 1], line[i]));

        return intersections;
    }

    QVector<Point> ClosedMesh::intersect(Path path)
    {
        QVector<Point> intersections;
        for(auto& seg : path)
            intersections.append(intersect(seg->start(), seg->end()));

        if(path.front()->start() == path.back()->end()) // Is this path is a closed loop
            intersections.append(intersect(path.back()->end(), path.front()->start()));

        return intersections;
    }

    QVector<Point> ClosedMesh::intersect(Polygon poly)
    {
        QVector<Point> intersections;
        for(int i = 1, end = poly.size(); i < end; ++i)
            intersections.append(intersect(poly[i - 1], poly[i]));

        intersections.append(intersect(poly.last(), poly.first())); // Add loop back line

        return intersections;
    }

    std::pair<QVector<Polyline>, QVector<Polygon>> ClosedMesh::intersect(Plane plane)
    {
        // Slicer constructor from the mesh
        CGAL::Polygon_mesh_slicer<MeshTypes::Polyhedron, MeshTypes::Kernel> slicer(m_representation);

        QVector<std::vector<MeshTypes::Kernel::Point_3>> cgal_polylines;
        slicer(plane.toCGALPlane(), std::back_inserter(cgal_polylines));

        QVector<Polyline> result_polylines;
        QVector<Polygon> result_polygons;

        for(auto& cgal_polyline : cgal_polylines)
        {
            if(cgal_polyline.front() == cgal_polyline.back()) // Can this be closed into a polygon?
            {
                Polygon new_polygon;
                for(auto& point : cgal_polyline)
                {
                    new_polygon.append(Point::FromCGALPoint(point));
                }
                result_polygons.push_back(new_polygon);
            }
            else
            {
                Polyline new_line;
                for(auto& point : cgal_polyline)
                {
                    new_line.append(Point::FromCGALPoint(point));
                }
                result_polylines.push_back(new_line);
            }
        }
        return std::make_pair(result_polylines, result_polygons);
    }

    void ClosedMesh::difference(ClosedMesh &clipper)
    {
        auto clip = clipper.polyhedron();
        CGAL::Polygon_mesh_processing::corefine_and_compute_difference(m_representation, clip, m_representation);

        // Convert back to a mesh
        auto vertices_and_faces = FacesAndVerticesFromPolyhedron(m_representation);
        m_vertices = vertices_and_faces.first;
        m_faces = vertices_and_faces.second;

        updateDims();
    }

    void ClosedMesh::intersection(ClosedMesh mesh_to_intersect)
    {
        // Convert both to CGAL polyhedrons
        MeshTypes::Polyhedron subject = polyhedron();
        MeshTypes::Polyhedron clip = mesh_to_intersect.polyhedron();

        MeshTypes::Polyhedron out;
        CGAL::Polygon_mesh_processing::corefine_and_compute_intersection(subject, clip, out);

        // Convert back to a mesh
        auto vertices_and_faces = FacesAndVerticesFromPolyhedron(out);
        m_vertices = vertices_and_faces.first;
        m_faces = vertices_and_faces.second;

        updateDims();
    }

    void ClosedMesh::mesh_union(ClosedMesh mesh_to_union)
    {
        // Convert both to CGAL polyhedrons
        MeshTypes::Polyhedron subject = polyhedron();
        MeshTypes::Polyhedron clip = mesh_to_union.polyhedron();

        MeshTypes::Polyhedron out;
        CGAL::Polygon_mesh_processing::corefine_and_compute_union(subject, clip, out);

        // Convert back to a mesh
        auto vertices_and_faces = FacesAndVerticesFromPolyhedron(out);
        m_vertices = vertices_and_faces.first;
        m_faces = vertices_and_faces.second;

        updateDims();
    }

    QSharedPointer<OpenMesh> ClosedMesh::toOpenMesh()
    {
        MeshTypes::SurfaceMesh output;
        CGAL::copy_face_graph(m_representation, output);
        auto new_mesh = QSharedPointer<OpenMesh>::create(output, this->name(), this->path());
        return new_mesh;
    }

    MeshTypes::SurfaceMesh ClosedMesh::extractUpwardFaces()
    {
        CGAL::set_halfedgeds_items_id(m_representation);
        std::vector<std::size_t> segment_ids(CGAL::num_faces(m_representation));
        FacetPropMap<std::size_t> segment_property_map(segment_ids);

        for(auto face : CGAL::faces(m_representation))
        {
            QVector<QVector3D> points;

            // Compute normal
            auto halfedge_index = face->facet_begin();
            do {
                auto point = halfedge_index->vertex()->point();
                points.push_back(QVector3D(point.x(), point.y(), point.z()));
            } while (++halfedge_index != face->facet_begin());

            QVector3D normal = QVector3D::crossProduct(points[1] - points[0],
                                                       points[2] - points[0]).normalized();
            // This face is upward facing
            if(normal.z() > 0.0)
                segment_property_map[face] = 1; // Set a flag
            else
                segment_property_map[face] = 0;
        }

        typedef CGAL::Face_filtered_graph<MeshTypes::Polyhedron> FilteredMesh;
        FilteredMesh segment_mesh(m_representation, 0, segment_property_map);
        segment_mesh.set_selected_faces(1, segment_property_map);

        MeshTypes::SurfaceMesh output;
        CGAL::copy_face_graph(segment_mesh, output);
        return output;
    }

    std::pair<ClosedMesh, ClosedMesh> ClosedMesh::splitWithPlane(Plane plane)
    {
        MeshTypes::Polyhedron internal_mesh = m_representation;
        MeshTypes::Polyhedron negative_mesh;
        MeshTypes::Polyhedron positive_mesh;

        CGAL::copy_face_graph(internal_mesh, negative_mesh);
        CGAL::copy_face_graph(internal_mesh, positive_mesh);

        CGAL::Polygon_mesh_processing::clip(negative_mesh, plane.toCGALPlane(), CGAL::Polygon_mesh_processing::parameters::clip_volume(true));

        // Invert plane normal
        plane.normal(-plane.normal());

        CGAL::Polygon_mesh_processing::clip(positive_mesh, plane.toCGALPlane(), CGAL::Polygon_mesh_processing::parameters::clip_volume(true));

        return std::pair<ClosedMesh, ClosedMesh>(ClosedMesh(negative_mesh), ClosedMesh(positive_mesh));
    }

    std::pair<bool, Area> ClosedMesh::crossSectionalArea(Plane plane, QVector<Polygon> boundary_curves)
    {
        // Take cross-section to find area
        std::list<std::vector<MeshTypes::Point_3>> cross_section;
        CGAL::Polygon_mesh_slicer<MeshTypes::Polyhedron, MeshTypes::Kernel> slicer(m_representation);
        slicer(plane.toCGALPlane(), std::back_inserter(cross_section));

        Area a = 0;
        bool intersecting = false;
        if(cross_section.size() == 0) // No cross section generated
            a = 0;
        else
        {
            for(auto section : cross_section)
            {
                auto list = cross_section.front();
                Polyline polyline(list);
                a += polyline.close().area();
            }
        }

        // We also need to check to see if it intersects with any other cuts
        return std::pair<bool, Area>(intersecting || CheckIntersectingCurves(plane, boundary_curves), a);
    }

    MeshTypes::Polyhedron ClosedMesh::PolyhedronFromVerticesAndFaces(QVector<MeshVertex>& vertices, QVector<MeshFace>& faces)
    {
        MeshTypes::Polyhedron mesh;
        MeshBuilder<MeshTypes::HalfedgeDescriptor> builder(vertices, faces);
        mesh.delegate(builder);
        return mesh;
    }

    std::pair<QVector<MeshVertex>, QVector<MeshFace>> ClosedMesh::FacesAndVerticesFromPolyhedron(MeshTypes::Polyhedron &mesh)
    {
        QVector<MeshFace> mesh_faces;
        mesh_faces.reserve(CGAL::faces(mesh).size());

        QVector<MeshVertex> mesh_vertices;
        mesh_vertices.reserve(CGAL::vertices(mesh).size());

        //! \note Because a surface mesh is indexed and not pointer based, it is faster to let CGAL index the polyhedron
        //!       and then convert it to faces and vertices.
        MeshTypes::SurfaceMesh sm;

        CGAL::copy_face_graph(mesh, sm);

        //! Compute normals of mesh faces and vertices
        auto vnormals = sm.add_property_map<MeshTypes::SM_VertexDescriptor, MeshTypes::Vector_3>("v:normals", CGAL::NULL_VECTOR).first;
        auto fnormals = sm.add_property_map<MeshTypes::SM_FaceDescriptor, MeshTypes::Vector_3>("f:normals", CGAL::NULL_VECTOR).first;
        CGAL::Polygon_mesh_processing::compute_normals(sm, vnormals, fnormals);

        for(auto v : sm.vertices())
            mesh_vertices.push_back(MeshVertex(Point(sm.point(v)).toQVector3D(), QVector3D(vnormals[v].x(), vnormals[v].y(), vnormals[v].z())));

        for(auto f : CGAL::faces(sm))
        {
            int vertex_index = 0;
            MeshFace face;
            for(auto v_id : vertices_around_face(sm.halfedge(f), sm))
            {
                face.vertex_index[vertex_index] = v_id;
                vertex_index++;
            }

            face.normal = QVector3D(fnormals[f].x(), fnormals[f].y(), fnormals[f].z());

            mesh_vertices[face.vertex_index[0]].connected_faces.push_back(mesh_faces.size());
            mesh_vertices[face.vertex_index[1]].connected_faces.push_back(mesh_faces.size());
            mesh_vertices[face.vertex_index[2]].connected_faces.push_back(mesh_faces.size());

            mesh_faces.push_back(face);
        }

        for(int faceIndex = 0; faceIndex < mesh_faces.size(); faceIndex++)
        {
            MeshFace& face = mesh_faces[faceIndex];
            face.connected_face_index[0] = GetFaceIdxWithPoints(face.vertex_index[0], face.vertex_index[1], faceIndex, mesh_vertices);
            face.connected_face_index[1] = GetFaceIdxWithPoints(face.vertex_index[1], face.vertex_index[2], faceIndex, mesh_vertices);
            face.connected_face_index[2] = GetFaceIdxWithPoints(face.vertex_index[2], face.vertex_index[0], faceIndex, mesh_vertices);
        }

        return std::pair<QVector<MeshVertex>, QVector<MeshFace>>(mesh_vertices, mesh_faces);
    }

    void ClosedMesh::CleanPolyhedron(MeshTypes::Polyhedron &polyhedron)
    {
        CGAL::Polygon_mesh_processing::remove_degenerate_faces(polyhedron);
        CGAL::Polygon_mesh_processing::remove_isolated_vertices(polyhedron);
        CGAL::Polygon_mesh_processing::remove_connected_components_of_negligible_size(polyhedron);
        CGAL::Polygon_mesh_processing::experimental::remove_self_intersections(polyhedron);

        // If the mesh is not closed, fill holes
        if(!polyhedron.is_closed())
        {
            for(boost::graph_traits<MeshTypes::Polyhedron>::halfedge_descriptor h : halfedges(polyhedron))
            {
                if(CGAL::is_border(h, polyhedron))
                {
                    std::vector<boost::graph_traits<MeshTypes::Polyhedron>::face_descriptor>  patch_facets;
                    std::vector<boost::graph_traits<MeshTypes::Polyhedron>::vertex_descriptor> patch_vertices;
                    bool success = std::get<0>(CGAL::Polygon_mesh_processing::triangulate_refine_and_fair_hole(polyhedron, h,
                                                std::back_inserter(patch_facets),
                                                std::back_inserter(patch_vertices),
                                                CGAL::parameters::vertex_point_map(get(CGAL::vertex_point, polyhedron)).geom_traits(MeshTypes::Kernel())));
                }
            }
        }
    }

    void ClosedMesh::convert()
    {
        m_representation = MeshTypes::Polyhedron(PolyhedronFromVerticesAndFaces(m_vertices, m_faces));
    }

    bool ClosedMesh::CheckIntersectingCurves(Plane &plane, QVector<Polygon> boundary_curves)
    {
        for(Polygon curve : boundary_curves)
        {
            for(Point p : curve)
            {
                if(plane.evaluatePoint(p) >= 0)
                    return true;
            }
        }
        return false;
    }

    void ClosedMesh::Sandbox()
    {
        // Points with normals.
        Pwn_vector points;

        for (MeshVertex vertex : this->vertices())
        {
            QVector3D normal = vertex.normal;
            Kernel::Vector_3 converted_normal(normal.x(), normal.y(), normal.z());
            points.push_back(std::make_pair(vertex.toPoint3(), converted_normal));

        }

        // Instantiate shape detection engine.
        Efficient_ransac ransac;
        // Provide input data.
        ransac.set_input(points);
        // Register shapes for detection.
        ransac.add_shape_factory<Planer>();
        ransac.add_shape_factory<Sphere>();
        ransac.add_shape_factory<Cylinder>();
        ransac.add_shape_factory<Cone>();
        ransac.add_shape_factory<Torus>();
        // Set parameters for shape detection.
        Efficient_ransac::Parameters parameters;
        // Set probability to miss the largest primitive at each iteration.
        parameters.probability = 0.05;
        // Detect shapes with at least 200 points.
        parameters.min_points = 200;
        // Set maximum Euclidean distance between a point and a shape.
        parameters.epsilon = 0.002;
        // Set maximum Euclidean distance between points to be clustered.
        parameters.cluster_epsilon = 0.01;
        // Set maximum normal deviation.
        // 0.9 < dot(surface_normal, point_normal);
        parameters.normal_threshold = 0.9;
        // Detect shapes.
        ransac.detect(parameters);
        // Print number of detected shapes and unassigned points.
        //std::cout << ransac.shapes().end() - ransac.shapes().begin()
        //          << " detected shapes, "
        //          << ransac.number_of_unassigned_points()
        //          << " unassigned points." << std::endl;
        // Efficient_ransac::shapes() provides
        // an iterator range to the detected shapes.
        Efficient_ransac::Shape_range shapes = ransac.shapes();
        Efficient_ransac::Shape_range::iterator it = shapes.begin();
        while (it != shapes.end()) {
            // Get specific parameters depending on the detected shape.
            if (Planer* plane = dynamic_cast<Planer*>(it->get())) {
                Kernel::Vector_3 normal = plane->plane_normal();
                QVector3D temp(normal.x(), normal.y(),normal.z());
                qDebug() << "Plane with normal " << temp;
            } else if (Cylinder* cyl = dynamic_cast<Cylinder*>(it->get())) {
                Kernel::Line_3 axis = cyl->axis();
                FT radius = cyl->radius();
                //std::cout << "Cylinder with axis "
                //          << axis << " and radius " << radius << std::endl;
            } else {
                // Print the parameters of the detected shape.
                // This function is available for any type of shape.
                //std::cout << (*it)->info() << std::endl;
            }
            // Proceed with the next detected shape.
            it++;
        }
    }
}
