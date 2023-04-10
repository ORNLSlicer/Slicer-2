#include "geometry/mesh/advanced/mesh_breadth_first_search.h"

// CGAL
#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <CGAL/Polygon_mesh_processing/border.h>

namespace ORNL
{
    MeshBreadthFirstSearch::MeshBreadthFirstSearch(size_t number_of_nodes) : m_number_of_nodes(number_of_nodes)
    {
        m_graph = Graph(number_of_nodes);
    }

    void MeshBreadthFirstSearch::buildVertex(MeshBreadthFirstSearch::FilteredGraph& segment_mesh, int id)
    {
        MeshTypes::Polyhedron out;
        CGAL::copy_face_graph(segment_mesh, out);

        QVector<Polygon> boundary_curves;
        QVector<HalfedgeDescriptor> holes;
        CGAL::Polygon_mesh_processing::extract_boundary_cycles(out, std::back_inserter(holes));
        for(HalfedgeDescriptor h : holes)
        {
            QVector<Point> curve;
            HalfedgeDescriptor first = h;
            do
            {
                curve.push_back(Point(h->vertex()->point()));
                h = h->next();
            } while(h->next() != first);
            boundary_curves.push_back(Polygon(curve));
        }
        m_vertex_map.insert(id, boundary_curves);

        setVertex(id);

        m_submesh_map.insert(id, out);
    }

    void MeshBreadthFirstSearch::connectVertices()
    {
        // Add edges to graph
        for(VertexDescriptor vertexADescriptor : boost::make_iterator_range(vertices(m_graph)))
        {
            int a_id = m_graph[vertexADescriptor];

            for(size_t vertexBDescriptor = (vertexADescriptor + 1); vertexBDescriptor < m_graph.vertex_set().size(); vertexBDescriptor++)
            {
                if(vertexADescriptor == vertexBDescriptor) // The vertex is the same, therefore no edge
                    continue;

                // Check to see if mesh A and B intersect. If they do create an edge
                int b_id = m_graph[vertexBDescriptor];

                typedef std::vector<MeshTypes::Polyhedron::Point_3> Polyline;

                for(Polygon polygon : m_vertex_map[a_id])
                {
                    Polyline polyline;
                    for(Point p : polygon)
                    {
                        polyline.push_back(p.toCartesian3D());
                    }
                    bool intersecting = CGAL::Polygon_mesh_processing::do_intersect(m_submesh_map[b_id], polyline);

                    if(intersecting)
                        addEdge(vertexADescriptor, vertexBDescriptor, polygon);
                }

            }
        }
    }

    QVector<Polygon> MeshBreadthFirstSearch::compute()
    {
        QVector<Polygon> curves;
        Visitor visitor(curves);

        boost::breadth_first_search(m_graph, m_number_of_nodes - 1, boost::visitor(visitor));
        return curves;
    }

    void MeshBreadthFirstSearch::setVertex(int id)
    {
        Graph::vertex_descriptor a = boost::vertex(id, m_graph);
        m_graph[a] = id;
    }

    void MeshBreadthFirstSearch::addEdge(MeshBreadthFirstSearch::VertexDescriptor source, MeshBreadthFirstSearch::VertexDescriptor target, Polygon value)
    {
        boost::add_edge(source, target, value, m_graph);
    }

    void MeshBreadthFirstSearch::Visitor::tree_edge(const MeshBreadthFirstSearch::Graph::edge_descriptor &e,
                                                   const MeshBreadthFirstSearch::Graph &g) const
    {
        Polygon boundaryCurve = g[e];
        m_edges.push_front(boundaryCurve);
    }
}
