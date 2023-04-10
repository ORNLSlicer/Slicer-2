#ifndef BREADTH_FIRST_SEARCH_H
#define BREADTH_FIRST_SEARCH_H

// Boost
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>

// CGAL
#include <CGAL/boost/graph/Face_filtered_graph.h>

// Locals
#include "geometry/polygon.h"
#include "geometry/mesh/closed_mesh.h"

namespace ORNL
{
    //!
    //! \brief The MeshBreadthFirstSearch class determines the printing order for a list of sub-meshes. It does this by
    //!        first building a graph where a sub-mesh is a node and boundary curve is the edge. It then traverses and records
    //!        the order of discovered edges.
    //!
    class MeshBreadthFirstSearch
    {
    public:
        /*!
         * \brief CGAL Types
         */
        typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, int, Polygon> Graph;
        typedef Graph::vertex_descriptor VertexDescriptor;
        typedef CGAL::Face_filtered_graph<MeshTypes::Polyhedron> FilteredGraph;
        typedef boost::graph_traits<MeshTypes::Polyhedron>::halfedge_descriptor HalfedgeDescriptor;

        //!
        //! \brief computes a BFS where the edges are boundary curves between mesh sub-sections;
        //! \param number_of_nodes: the number of nodes/ vertices in the graph
        //!
        MeshBreadthFirstSearch(size_t number_of_nodes);

        //!
        //! \brief builds a node/ vertex in the graph
        //! \param segment_mesh: the mesh that this node represents
        //! \param id: the id of the node
        //!
        void buildVertex(FilteredGraph& segment_mesh, int id);

        //!
        //! \brief connects vertices/ nodes based on which meshes intersect. If two meshes intersect, then
        //!        an edge that stores their boundary curve is created
        //!
        void connectVertices();

        //!
        //! \brief computes the breadth first search
        //! \return returns an ordered list of boundary curves
        //!
        QVector<Polygon> compute();

    private:
        //!
        //! \brief the undirected graph structure
        //!
        Graph m_graph;

        //!
        //! \brief holds the vertices that were added to the graph in order
        //!
        QMap<int, QVector<Polygon>> m_vertex_map;

        //!
        //! \brief holds the sub-meshes that were added in order
        //!
        QMap<int, MeshTypes::Polyhedron> m_submesh_map;

        //!
        //! \brief the number of nodes in the graph
        //!
        size_t m_number_of_nodes;

        //!
        //! \brief sets a verted based on id
        //! \param id: the number of the vertex
        //!
        void setVertex(int id);

        //!
        //! \brief adds an edge to the graph
        //! \param source: from nodes
        //! \param target: destination node
        //! \param value: the boundary curve between the two nodes/ sub-meshes
        //!
        void addEdge(VertexDescriptor source, VertexDescriptor target, Polygon value);

        //!
        //! \brief The Visitor struct is used by boost to traverse the graph. tree_edge() is called whenever a new edge is discovered
        //!
        struct Visitor : boost::default_bfs_visitor{
            Visitor(QVector<Polygon>& edges) : m_edges(edges) {};
            void tree_edge(const Graph::edge_descriptor &e, const Graph &g) const;
            QVector<Polygon>& m_edges;
        };
    };
}
#endif // BREADTH_FIRST_SEARCH_H
