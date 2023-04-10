#include "geometry/mesh/advanced/mesh_segmentation.h"

//Qt
#include <QRandomGenerator>

// CGAL
#include <CGAL/mesh_segmentation.h>

// Locals
#include "geometry/mesh/advanced/mesh_skeleton.h"
#include "geometry/mesh/advanced/mesh_breadth_first_search.h"

namespace ORNL
{
    MeshSegmenter::MeshSegmenter(){ }

    QVector<MeshTypes::Polyhedron> MeshSegmenter::splitMeshIntoSections(MeshTypes::Polyhedron mesh)
    {
        QVector<MeshTypes::Polyhedron> subsections;
        CGAL::set_halfedgeds_items_id(mesh);

        std::vector<double> sdf_values(num_faces(mesh));
        ClosedMesh::FacetPropMap<double> sdf_property_map(sdf_values);
        computeSDFValues(sdf_property_map, mesh);

        std::vector<std::size_t> segment_ids(num_faces(mesh));
        ClosedMesh::FacetPropMap<std::size_t> segment_property_map(segment_ids);

        // Segment the mesh using default parameters
        size_t number_of_segments = CGAL::segmentation_from_sdf_values(mesh, sdf_property_map, segment_property_map, 5);
        subsections.reserve(number_of_segments);

        if(number_of_segments == 1)
        {
            subsections.push_back(mesh);
            return subsections;
        }

        MeshBreadthFirstSearch::FilteredGraph segment_mesh(mesh, 0, segment_property_map);

        MeshBreadthFirstSearch bfs(number_of_segments);
        for(std::size_t id = 0; id < number_of_segments; ++id)
        {
            // Select faces for a given id
            if(id > 0)
                segment_mesh.set_selected_faces(id, segment_property_map);

            bfs.buildVertex(segment_mesh, id);
        }
        bfs.connectVertices();

        QVector<Polygon> curves = bfs.compute();

        MeshTypes::Polyhedron construction_mesh;
        CGAL::copy_face_graph(mesh, construction_mesh);

        while(curves.size() > 0)
        {
            Polygon boundary_curve = curves.front();
            curves.pop_front();

            // Fetch boundary curve and get plane from random points
            Plane original_plane = getPlaneOnCurve(boundary_curve);
            if(original_plane.normal().z() < 0)
                original_plane.normal(-original_plane.normal());

            std::pair<bool, Area> cross_section_result = ClosedMesh(construction_mesh).crossSectionalArea(original_plane, curves);
            bool intersecting = cross_section_result.first;
            Area original_cross_sectional_area = cross_section_result.second;

            Plane new_plane = original_plane;
            Area new_cross_sectional_area = original_cross_sectional_area;

            while(intersecting) // The plane will intersect with a printed section or is otherwise not valid
            {
                // Generate random perturbation data to shift plane
                perturbPlane(new_plane);

                cross_section_result = ClosedMesh(construction_mesh).crossSectionalArea(new_plane, curves);
                intersecting = cross_section_result.first;
                new_cross_sectional_area = cross_section_result.second;

                if(((new_cross_sectional_area / original_cross_sectional_area) < 0.85) ||
                   ((new_cross_sectional_area / original_cross_sectional_area) > 1.15)) // If the area has changed more then 15% reset and try again
                {
                    new_plane = original_plane;
                    new_cross_sectional_area = original_cross_sectional_area;
                    intersecting = true;
                }
            }

            // Simply divide mesh
            auto split_operation = ClosedMesh(construction_mesh).splitWithPlane(new_plane);

            // Add negative side to construction mesh
            construction_mesh = split_operation.first.polyhedron();

            // Push positive section
            subsections.push_front(split_operation.second.polyhedron());
        }

        subsections.push_front(construction_mesh);

        return subsections;
    }

    void MeshSegmenter::perturbPlane(Plane& plane)
    {
        // Randomly move forward or backward along normal
        int distanceSign = 1;
        bool toDo = QRandomGenerator::global()->generate() % 2;
        if(toDo) // Move backward
            distanceSign = -1;

        double d = (QRandomGenerator::global()->generate() % 2000);
        plane.shiftAlongNormal(distanceSign * d); // Move between 0 and 2000 microns

        // Rotate normal by up to 5 deg in random directions
        // Convert to spherical
        double rho = qSqrt(qPow(plane.normal().x(), 2) + qPow(plane.normal().y(), 2) + qPow(plane.normal().z(), 2));
        double theta = qAtan(plane.normal().x() / plane.normal().y());
        double phi = qAcos(plane.normal().z() / rho);

        // Randomly add or subtract
        // between 0 and 5 random degrees (0 to 0.0872665 rad) to theta and phi
        double fract = (static_cast<double>((QRandomGenerator::global()->generate() % 872)) / 10000.0);
        toDo = QRandomGenerator::global()->generate() % 2;
        if(toDo)
            theta += fract;
        else
            theta -= fract;
        fract = (static_cast<double>((QRandomGenerator::global()->generate() % 872)) / 10000.0);
        toDo = QRandomGenerator::global()->generate() % 2;
        if(toDo)
            phi += fract;
        else
            phi -= fract;

        // Convert back to rectangular
        plane.normal(QVector3D(rho * qSin(phi) * qSin(theta), rho * qSin(phi) * qCos(theta), rho * qCos(phi)));

        if(plane.normal().z() < 0)
            plane.normal(-plane.normal());
    }

    Plane MeshSegmenter::getPlaneOnCurve(Polygon& curve)
    {
        // Get random 3 points from curve and make plane
        // This ensures they are not the same 3 points or colinear
        Point p0, p1, p2;
        QVector3D cross;
        QVector3D zero = {0,0,0};
        do
        {
            QVector3D v0, v1;
            p0 = curve.at(QRandomGenerator::global()->generate() % curve.size());
            p1 = curve.at(QRandomGenerator::global()->generate() % curve.size());
            p2 = curve.at(QRandomGenerator::global()->generate() % curve.size());
            v0 = (p1 - p0).toQVector3D();
            v1 = (p2 - p0).toQVector3D();
            cross = QVector3D::crossProduct(v0, v1);
        }while(p0 == p1 || p0 == p2 || p1 == p2 || cross == zero);

        Plane plane(p0, p1, p2);

        // The plane needs to point up
        if(plane.normal().z() < 0)
            plane.normal(-plane.normal());

        return plane;
    }

    void MeshSegmenter::computeSDFValues(ClosedMesh::FacetPropMap<double>& map, MeshTypes::Polyhedron& mesh)
    {
        MeshSkeleton skeleton(mesh);
        skeleton.compute();
        auto skeletonization = skeleton.getSkeleton();

        //for each input vertex compute its distance to the skeleton
        std::vector<double> distances(num_vertices(mesh));
        for(auto v : CGAL::make_range(vertices(skeletonization)))
        {
            const MeshTypes::SimpleCartesian::Polyhedron::Point_3& skel_pt = skeletonization[v].point;
            for(auto mesh_v : skeletonization[v].vertices)
            {
                const  MeshTypes::SimpleCartesian::Polyhedron::Point_3& mesh_pt = mesh_v->point();
                distances[mesh_v->id()] = std::sqrt(CGAL::squared_distance(skel_pt, mesh_pt));
            }
        }

        // compute sdf values with skeleton
        for(auto f : faces(mesh))
        {
            double dist = 0;
            for(auto hd : halfedges_around_face(halfedge(f, mesh), mesh))
                dist += distances[target(hd, mesh)->id()];
            map[f] = dist / 3.;
        }

        // post-process the sdf values
        CGAL::sdf_values_postprocessing(mesh, map);
    }
}
