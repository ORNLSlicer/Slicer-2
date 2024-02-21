// Local
#include "step/layer/regions/skeleton.h"
#include "geometry/segments/line.h"
#include "optimizers/polyline_order_optimizer.h"
#include "geometry/path_modifier.h"
#include "utilities/mathutils.h"

// Boost
#include "boost/polygon/voronoi.hpp"
#include "boost/graph/undirected_dfs.hpp"


using boost::polygon::voronoi_diagram;

typedef boost::polygon::point_data<int> point;
typedef voronoi_diagram<double>::cell_type cell_type;
typedef voronoi_diagram<double>::edge_type edge_type;
typedef voronoi_diagram<double>::vertex_type vertex_type;

template <>
struct boost::polygon::geometry_concept<ORNL::Point> {typedef point_concept type;};

template <>
struct boost::polygon::point_traits<ORNL::Point>
{
  typedef int coordinate_type;

  static inline coordinate_type get(const ORNL::Point& point, orientation_2d orient)
  {
    return (orient == HORIZONTAL) ? point.x() : point.y();
  }
};

template <>
struct boost::polygon::geometry_concept<ORNL::Polyline> {typedef segment_concept type;};

template <>
struct boost::polygon::segment_traits<ORNL::Polyline>
{
  typedef int coordinate_type;
  typedef ORNL::Point point_type;

  static inline point_type get(const ORNL::Polyline& segment, direction_1d dir)
  {
    return dir.to_int() ? segment.first() : segment.last();
  }
};

namespace ORNL {

    Skeleton::Skeleton(const QSharedPointer<SettingsBase>& sb, const int index, const QVector<SettingsPolygon>& settings_polygons,
                       const SingleExternalGridInfo& gridInfo, bool isWireFed)
        : RegionBase(sb, index, settings_polygons, gridInfo)
    {
        m_wire_region = isWireFed;
    }

    QString Skeleton::writeGCode(QSharedPointer<WriterBase> writer)
    {
        QString gcode;
        gcode += writer->writeBeforeRegion(RegionType::kSkeleton);
        for (Path path : m_paths) {
            gcode += writer->writeBeforePath(RegionType::kSkeleton);
            for (QSharedPointer<SegmentBase> segment : path.getSegments()) {
                gcode += segment->writeGCode(writer);
            }
            gcode += writer->writeAfterPath(RegionType::kSkeleton);
        }
        gcode += writer->writeAfterRegion(RegionType::kSkeleton);
        return gcode;
    }

    void Skeleton::compute(uint layer_num, QSharedPointer<SyncManager>& sync)
    {
        m_paths.clear();

        setMaterialNumber(m_sb->setting<int>(Constants::MaterialSettings::MultiMaterial::kPerimterNum));

        incorporateLostGeometry();

        if (!m_geometry.isEmpty())
        {
            simplifyInputGeometry(layer_num);

            SkeletonInput input = static_cast<SkeletonInput>(m_sb->setting<int>(Constants::ProfileSettings::Skeleton::kSkeletonInput));
            switch(input)
            {
            case SkeletonInput::kSegments:
                computeSegmentVoronoi();
                break;
            case SkeletonInput::kPoints:
                computePointVoronoi();
                break;
            }

            if (!m_skeleton_geometry.isEmpty())
            {
                extractSkeletonPaths();
                simplifyOutputGeometry();

                //! Uncomment to inspect Skeleton structure. See instructions in header file.
                //inspectSkeleton(layer_num);

                if(m_computed_anchor_lines.size() != 0)
                {
                    m_computed_geometry.push_front(m_computed_anchor_lines.first());
                    m_computed_geometry.push_back(m_computed_anchor_lines.last());
                }
            }
            else
                qDebug() << "\t\tNo permitted skeletons generated from geometry on layer " << layer_num;
        }
        else
            qDebug() << "\t\tNo geometry for skeletons to compute";
    }

    void Skeleton::computeSegmentVoronoi()
    {
        //! Gather input geometry
        QVector<Polyline> segments;
        for (Polygon &poly : m_geometry)
        {
            for (uint i = 0, max = poly.size() - 1; i < max; ++i)
                segments += Polyline({poly[i], poly[i + 1]});

            segments += Polyline({poly.last(), poly.first()});
        }

        //! Construct Voronoi Diagram
        voronoi_diagram<double> vd;
        construct_voronoi(segments.begin(), segments.end(), &vd);

        //! Gather skeleton segments
        for (voronoi_diagram<double>::const_cell_iterator it = vd.cells().begin(); it != vd.cells().end(); ++it)
        {
            const cell_type& cell = *it;

            if (!cell.is_degenerate())
            {
                const edge_type* edge = cell.incident_edge();

                do
                {
                    if (edge->is_primary())
                    {
                        if (edge->is_finite())
                        {
                            //! Ensures skeleton segments are only generated once since voronoi diagram implements twin edges
                            if (edge->cell()->source_index() < edge->twin()->cell()->source_index())
                            {
                                Polyline source1 = segments[edge->cell()->source_index()];
                                Polyline source2 = segments[edge->twin()->cell()->source_index()];

                                //! Ensures skeleton segments run parallel to the border geometry
                                if (source1.first() != source2.first() && source1.first() != source2.last() && source1.last() != source2.first() && source1.last() != source2.last())
                                {
                                    Point pt1(edge->vertex0()->x(), edge->vertex0()->y());
                                    Point pt2(edge->vertex1()->x(), edge->vertex1()->y());

                                    //! Ensure skeleton segments fit within the border geometry
                                    m_skeleton_geometry += m_geometry & Polyline({pt1, pt2});
                                }
                            }
                        }
                    }

                    edge = edge->next();
                } while (edge != cell.incident_edge());
            }
        }

        //! Remove any skeleton segments that may overlap with border geometry
        //! Extremely rare and may be unnecessary with input geometry cleaning
        QVector<Polyline>::iterator segment_iter = m_skeleton_geometry.begin(), segment_iter_end = m_skeleton_geometry.end();
        while (segment_iter != segment_iter_end)
        {
            if (segments.contains(*segment_iter) || segments.contains(segment_iter->reverse()))
                segment_iter = m_skeleton_geometry.erase(segment_iter);
            else
                ++segment_iter;
        }
    }

    void Skeleton::computePointVoronoi()
    {
        //! Gather input geometry
        QVector<Point> points;
        for (Polygon &poly : m_geometry)
            points += poly;

        //! Construct Voronoi Diagram
        voronoi_diagram<double> vd;
        construct_voronoi(points.begin(), points.end(), &vd);

        //! Gather skeleton segments
        for (voronoi_diagram<double>::const_cell_iterator it = vd.cells().begin(); it != vd.cells().end(); ++it)
        {
            const cell_type& cell = *it;
            const edge_type* edge = cell.incident_edge();

            do
            {
                if (edge->is_primary())
                {
                    if (edge->is_finite())
                    {
                        //! Ensures skeleton segments are only generated once since voronoi diagram implements twin edges
                        if (edge->cell()->source_index() < edge->twin()->cell()->source_index())
                        {
                            Point start(edge->vertex0()->x(), edge->vertex0()->y());
                            Point end(edge->vertex1()->x(), edge->vertex1()->y());

                            //! Ensure skeleton segments fit within the border geometry
                            m_skeleton_geometry += m_geometry & Polyline({start, end});
                        }
                    }
                    else //! Infinite edge case
                    {
                        const vertex_type* v0 = edge->vertex0();

                        //! Ensures skeleton segments are only generated once since voronoi diagram implements twin edges
                        if (v0)
                        {
                            Point start(v0->x(), v0->y());

                            //! Determine segment end point from segment seeds
                            Point seed1 = points[edge->cell()->source_index()];
                            Point seed2 = points[edge->twin()->cell()->source_index()];
                            Point end((seed1.x() + seed2.x()) / 2, (seed1.y() + seed2.y()) / 2);

                            //! Generate skeleton segments that fit within the geometry
                            m_skeleton_geometry += m_geometry & Polyline({start, end});
                        }
                    }
                }

                edge = edge->next();
            } while (edge != cell.incident_edge());
        }
    }

    void Skeleton::incorporateLostGeometry()
    {
        //! Integrate lost geometry with m_geometry, ensuring they share no common geometry
        for (Polygon &poly : m_geometry.lost_geometry)
            if ((m_geometry & poly).isEmpty())
                m_geometry += poly;
    }

    void Skeleton::simplifyInputGeometry(const uint& layer_num)
    {
        Distance cleaning_dist = m_sb->setting<Distance>(Constants::ProfileSettings::Skeleton::kSkeletonInputCleaningDistance);

        //! Too large of a cleaning distance may decimate inner/outer polygons such that they
        //! contain no points or intersect each other. Check that cleaning distance is appropriate.
        bool valid = true;
        for (Point& p1 : m_geometry.first())
        {
            for (Polygon& poly : m_geometry.mid(1))
            {
                for (Point& p2 : poly)
                {
                    if (cleaning_dist > p1.distance(p2))
                    {
                        valid = false;
                        break;
                    }
                }

                if (!valid)
                    break;
            }

            if (!valid)
                break;
        }

        if (valid)
            m_geometry = m_geometry.cleanPolygons(cleaning_dist);
        else
            qDebug() << "Layer " << layer_num << " Skeleton input geometry cleaning distance too large.";


        //! Chamfer long axis corner of triangles
        for (Polygon &poly : m_geometry)
        {
            if (poly.size() == 3)
            {
                Point A = poly[0];
                Point B = poly[1];
                Point C = poly[2];

                Angle ABC = MathUtils::internalAngle(A, B, C);
                Angle BCA = MathUtils::internalAngle(B, C, A);
                Angle CAB = MathUtils::internalAngle(C, A, B);

                if (ABC < BCA && ABC < CAB)
                {
                    poly = MathUtils::chamferCorner(A, B, C, 10);
                    poly.prepend(A);
                    poly.append(C);
                }
                else if (BCA < ABC && BCA < CAB)
                {
                    poly = MathUtils::chamferCorner(B, C, A, 10);
                    poly.prepend(B);
                    poly.append(A);
                }
                else
                {
                    poly = MathUtils::chamferCorner(C, A, B, 10);
                    poly.prepend(C);
                    poly.append(B);
                }
            }
        }

        //! Chamfer sharp corners
        Angle threshold = m_sb->setting<Angle>(Constants::ProfileSettings::Skeleton::kSkeletonInputChamferingAngle);
        for (Polygon &geometry : m_geometry)
        {
            Polyline poly = geometry.toPolyline();
            Polygon new_geometry;

            for (uint i = 1, max = poly.size() - 1; i < max; ++i)
                if (MathUtils::internalAngle(poly[i - 1], poly[i], poly[i + 1]) < threshold)
                    new_geometry.append(MathUtils::chamferCorner(poly[i - 1], poly[i], poly[i + 1], 10));
                else
                    new_geometry.append(poly[i]);

            if (MathUtils::internalAngle(geometry.last(), geometry[0], geometry[1]) < threshold)
                new_geometry.append(MathUtils::chamferCorner(geometry.last(), geometry[0], geometry[1], 10));
            else
                new_geometry.append(geometry.first());

            geometry = new_geometry;
        }
    }

    void Skeleton::simplifyOutputGeometry()
    {
        Distance cleaning_distance = m_sb->setting<Distance>(Constants::ProfileSettings::Skeleton::kSkeletonOutputCleaningDistance);
        for (Polyline& poly_line : m_computed_geometry)
            poly_line = poly_line.simplify(cleaning_distance);
    }

    void Skeleton::extractSkeletonPaths()
    {
        generateSkeletonGraph();
        extractCycles();
        extractSimplePaths();
    }

    void Skeleton::generateSkeletonGraph()
    {
        //! Tracks points that have been added to the graph
        QMap<Point, SkeletonVertex> vertices;

        for (Polyline &edge :m_skeleton_geometry)
        {
            SkeletonVertex v0;
            if (vertices.contains(edge.first()))
            {
                v0 = vertices[edge.first()];
            }
            else
            {
                v0 = add_vertex(edge.first(), m_skeleton_graph);
                vertices.insert(edge.first(), v0);
            }

            SkeletonVertex v1;
            if (vertices.contains(edge.last()))
            {
                v1 = vertices[edge.last()];
            }
            else
            {
                v1 = add_vertex(edge.last(), m_skeleton_graph);
                vertices.insert(edge.last(), v1);
            }

            add_edge(v0, v1, edge, m_skeleton_graph);
        }
    }

    void Skeleton::cleanSkeletonGraph(Distance cleaning_distance)
    {
        Distance distSqrd = std::pow(cleaning_distance(), 2);

        Vertex_Iter vi, vi_end, v_root;
        boost::tie(vi, vi_end) = vertices(m_skeleton_graph);

        //! Create vertex modification map
        QMap<SkeletonVertex, bool> vertex_mod_map;
        while (vi != vi_end)
        {
            vertex_mod_map.insert(*vi, false);
            ++vi;
        }

        boost::tie(vi, vi_end) = vertices(m_skeleton_graph);
        while (vi != vi_end)
        {
            v_root = vi;
            vi++;

            //! Clean skeleton graph according to ClipperLib2's cleanPolygons function.
            //! Vertices are analyzed in sets of 3: v0--v_root--v1 with pivot vertices (degree != 2) being ignored.
            //! If a vertice has undergone a modification it is marked so as not to be modified again. This prevents
            //! skeletal drifting.
            if (boost::degree(*v_root, m_skeleton_graph) == 2 && !vertex_mod_map[*v_root])
            {
                Out_Edge_Iter e0, e1;
                boost::tie(e0, e1) = boost::out_edges(*v_root, m_skeleton_graph);
                --e1;

                SkeletonVertex v0 = boost::target(*e0, m_skeleton_graph);
                SkeletonVertex v1 = boost::target(*e1, m_skeleton_graph);

                if (MathUtils::pointsAreClose(m_skeleton_graph[*v_root], m_skeleton_graph[v0], distSqrd)
                        || MathUtils::pointsAreClose(m_skeleton_graph[*v_root], m_skeleton_graph[v1], distSqrd)
                        || MathUtils::pointsAreClose(m_skeleton_graph[v0], m_skeleton_graph[v1], distSqrd)
                        || MathUtils::slopesNearCollinear(m_skeleton_graph[v0], m_skeleton_graph[*v_root], m_skeleton_graph[v1], distSqrd))
                {
                    add_edge(v0, v1, {Polyline({m_skeleton_graph[v0], m_skeleton_graph[v1]})}, m_skeleton_graph);
                    remove_edge(*e0, m_skeleton_graph);
                    remove_edge(*e1, m_skeleton_graph);
                    remove_vertex(*v_root, m_skeleton_graph);

                    vertex_mod_map[v0] = true;
                    vertex_mod_map[v1] = true;
                    boost::tie(vi, vi_end) = vertices(m_skeleton_graph);
                }
                else
                {
                    vertex_mod_map[*v_root] = true;
                }
            }
        }
    }

    template <typename TVertex, typename TEdge, typename TGraph>
    struct dfs_visitor : public boost::dfs_visitor<>
    {
        dfs_visitor(QVector<Edge> &longest_path_) : longest_path(longest_path_) {}

        using colormap = std::map<SkeletonGraph::vertex_descriptor, boost::default_color_type>;
        colormap vertex_coloring;

        using edgeColorMap = std::map<SkeletonGraph::edge_descriptor, boost::default_color_type>;
        edgeColorMap  edge_coloring;

        void tree_edge(const TEdge &e, const TGraph &g)
        {
            TVertex source = boost::source(e,g), target = boost::target(e, g);

            predecessor_map[target] = predecessor_map[source];
            predecessor_map[target].push(source);

            if (typeid(g) == typeid(SubGraph) && boost::degree(target, g) == 1)
                getSimplePath(target, g);
        }

        void back_edge(const TEdge &e, const TGraph &g)
        {
            TVertex source = boost::source(e, g), target = boost::target(e, g);

            predecessor_map[target] = predecessor_map[source];
            predecessor_map[target].push(source);

            getCycle(target, g);
        }

        void getCycle(const TVertex &v, const TGraph &g)
        {
            TVertex source = v, target;

            QVector<Edge> cycle;
            Distance cycle_length = 0;
            while (predecessor_map[v].top() != v)
            {
                target = predecessor_map[v].pop();
                TEdge e = edge(source, target, g).first;
                cycle += e;
                cycle_length += g[e].length();
                source = target;
            }

            TEdge e = edge(source, v, g).first;
            cycle += e;
            cycle_length += g[e].length();

            if (cycle_length > longest_path_length)
            {
                longest_path = cycle;
                longest_path_length = cycle_length;
            }
        }

        void getSimplePath(const TVertex v, const TGraph &g)
        {
            TVertex source = v, target;

            QVector<Edge> simple_path;
            Distance length = 0;
            while (!predecessor_map[v].isEmpty())
            {
                target = predecessor_map[v].pop();
                TEdge e = edge(source, target, g).first;
                length += g[e].length();
                simple_path += e;
                source = target;
            }

            if (length > longest_path_length)
            {
                longest_path = simple_path;
                longest_path_length = length;
            }
        }

    private:
        QMap<SkeletonVertex, QStack<SkeletonVertex>> predecessor_map;
        QVector<Edge> &longest_path;
        Distance longest_path_length = 0;
    };

    void Skeleton::extractCycles()
    {
        //! Extract cycles in order of longest to shortest
        while (true)
        {
            QVector<Edge> longest_cycle;
            dfs_visitor<SkeletonVertex, Edge, SkeletonGraph> vis(longest_cycle);
            boost::undirected_dfs(m_skeleton_graph, vis, make_assoc_property_map(vis.vertex_coloring), make_assoc_property_map(vis.edge_coloring));

            if (!longest_cycle.isEmpty())
                extractPath(longest_cycle);
            else
                break; //! All cycles have been removed

            //check if cycle removal empties graph
            if(boost::num_edges(m_skeleton_graph) == 0 || boost::num_vertices(m_skeleton_graph) == 0)
                break;
        }
    }

    void Skeleton::extractSimplePaths()
    {
        //! Extract simple paths in order of longest to shortest
        while (boost::num_edges(m_skeleton_graph) > 0)
        {
            //! Create Vertex Index Map. Needed for SubGraph Map.
            std::map<SkeletonVertex, int> vertex_index_map;
            boost::associative_property_map<std::map<SkeletonVertex, int>> index_map(vertex_index_map);
            Vertex_Iter v, v_end;
            boost::tie(v, v_end) = boost::vertices(m_skeleton_graph);
            for(int i = 0; v != v_end; ++v, ++i)
                boost::put(index_map, *v, i);

            //! Create SubGraph Map
            std::map<SkeletonVertex, int> vertex_subgraph_map;
            boost::associative_property_map<std::map<SkeletonVertex, int>> subgraph_map(vertex_subgraph_map);
            int number_of_subgraphs = boost::connected_components(m_skeleton_graph, subgraph_map, boost::vertex_index_map(index_map));

            //! Create SubGraph
            SubGraph subgraph(m_skeleton_graph, boost::keep_all(), subgraph_filter(QMap<SkeletonVertex, int>(vertex_subgraph_map)));

            //! Find open edge to start dfs from
            SubGraph::vertex_iterator vi, vi_end;
            boost::tie(vi, vi_end) = vertices(subgraph);
            SubGraph::vertex_descriptor start = *vi;
            while (vi != vi_end)
            {
                if (boost::degree(*vi, subgraph) == 1)
                {
                    start = *vi;
                    break;
                }

                ++vi;
            }

            //! Collect longest simple path
            QVector<Edge> longest_simple_path;
            dfs_visitor<SubGraph::vertex_descriptor, SubGraph::edge_descriptor, SubGraph> vis(longest_simple_path);
            boost::undirected_dfs(subgraph, vis, make_assoc_property_map(vis.vertex_coloring), make_assoc_property_map(vis.edge_coloring), start);

            if (!longest_simple_path.isEmpty())
                extractPath(longest_simple_path);
        }
    }

    void Skeleton::extractPath(QVector<Edge> path_)
    {
        Polyline path;
        Distance min_path_length = m_sb->setting<Distance>(Constants::ProfileSettings::Skeleton::kMinPathLength);

        Edge e = path_.takeFirst();
        path << m_skeleton_graph[e.m_source] << m_skeleton_graph[e.m_target];
        remove_edge(e, m_skeleton_graph);

        //! Remove isolated vertices
        if (boost::degree(e.m_source, m_skeleton_graph) == 0)
            remove_vertex(e.m_source, m_skeleton_graph);
        if (boost::degree(e.m_target, m_skeleton_graph) == 0)
            remove_vertex(e.m_target, m_skeleton_graph);

        for (Edge &e : path_)
        {
            path << m_skeleton_graph[e.m_target];
            remove_edge(e, m_skeleton_graph);

            //! Remove isolated vertices
            if (boost::degree(e.m_source, m_skeleton_graph) == 0)
                remove_vertex(e.m_source, m_skeleton_graph);
            if (boost::degree(e.m_target, m_skeleton_graph) == 0)
                remove_vertex(e.m_target, m_skeleton_graph);
        }

        //! Ensure path meets minimum path length requirement
        if (path.length() > min_path_length)
            m_computed_geometry += path;
    }

    void Skeleton::getSkeleton()
    {
        Distance min_path_length = m_sb->setting<Distance>(Constants::ProfileSettings::Skeleton::kMinPathLength);

        //! Locate pivot vertices with degree != 2
        QVector<SkeletonVertex> pivots;
        for (SkeletonVertex &vertex : boost::make_iterator_range(vertices(m_skeleton_graph)))
            if (boost::degree(vertex, m_skeleton_graph) != 2)
                pivots += vertex;

        //! If no pivots exist, then the skeleton must be a single closed loop contour and there is no need to check
        //! its path length.
        if (!pivots.isEmpty())
        {
            for (SkeletonVertex &pivot : pivots)
            {
                while (boost::degree(pivot, m_skeleton_graph) > 0)
                {
                    Out_Edge_Iter edge = boost::out_edges(pivot, m_skeleton_graph).first;
                    SkeletonVertex target = boost::target(*edge, m_skeleton_graph);

                    Polyline path({m_skeleton_graph[pivot], m_skeleton_graph[target]});
                    remove_edge(edge, m_skeleton_graph);

                    if (boost::degree(target, m_skeleton_graph) != 0)
                    {
                        do
                        {
                            edge = boost::out_edges(target, m_skeleton_graph).first;
                            target = boost::target(*edge, m_skeleton_graph);

                            path += m_skeleton_graph[target];
                            remove_edge(edge, m_skeleton_graph);
                        } while (!pivots.contains(target) && boost::degree(target, m_skeleton_graph) != 0);
                    }

                    //! Ensure path meets minimum path length requirement
                    if (path.length() >= min_path_length)
                        m_computed_geometry += path;
                }
            }
        }
    }

    void Skeleton::inspectSkeleton(const uint& layer_num)
    {
        static QMutex lock;
        QMutexLocker locker(& lock);

        #define precision_qDebug() qDebug() << fixed << qSetRealNumberPrecision(1)

        //! Print input geometry
        qDebug() << "Layer " << layer_num << "Input Geometry:";
        for (Polygon &poly : m_geometry)
        {
            for (uint i = 0, max = poly.size() - 1; i < max; ++i)
                precision_qDebug() << "polygon((" << poly[i].x() << "," << poly[i].y() << "),(" << poly[i + 1].x() << "," << poly[i + 1].y() << "))";

            precision_qDebug() << "polygon((" << poly.first().x() << "," << poly.first().y() << "),(" << poly.last().x() << "," << poly.last().y() << "))";
        }

        //! Print skeleton geometry
        qDebug() << "Layer " << layer_num << "Skeleton Geometry";
        for (Polyline &seg : m_computed_geometry)
            for (uint i = 0, max = seg.size() - 1; i < max; ++i)
                precision_qDebug() << "polygon((" << seg[i].x() << "," << seg[i].y() << "),(" << seg[i + 1].x() << "," << seg[i + 1].y() << "))";
    }

    void Skeleton::inspectSkeletonGraph()
    {
        static QMutex lock;
        QMutexLocker locker(& lock);

        #define precision_qDebug() qDebug() << fixed << qSetRealNumberPrecision(1)

        //! Print input geometry
        qDebug() << "Input Geometry";
        for (Polygon &poly : m_geometry)
        {
            for (uint i = 0, max = poly.size() - 1; i < max; ++i)
                precision_qDebug() << "polygon((" << poly[i].x() << "," << poly[i].y() << "),(" << poly[i + 1].x() << "," << poly[i + 1].y() << "))";

            precision_qDebug() << "polygon((" << poly.first().x() << "," << poly.first().y() << "),(" << poly.last().x() << "," << poly.last().y() << "))";
        }

        //! Print skeleton graph
        qDebug() << "Skeleton Graph";
        boost::graph_traits<SkeletonGraph>::edge_iterator edge_iter, edge_iter_end;
        boost::tie(edge_iter,edge_iter_end) = edges(m_skeleton_graph);
        while (edge_iter != edge_iter_end)
        {
            precision_qDebug() << "polygon((" << m_skeleton_graph[edge_iter->m_source].x() << "," << m_skeleton_graph[edge_iter->m_source].y() << "),(" << m_skeleton_graph[edge_iter->m_target].x() << "," << m_skeleton_graph[edge_iter->m_target].y() << "))";
            edge_iter++;
        }
    }

    QVector<QSharedPointer<LineSegment>> Skeleton::adaptBeadWidth(const Point &start, const Point &end)
    {
        //! Define discretization variables
        Distance disc_dist = m_sb->setting<Distance>(Constants::ProfileSettings::Skeleton::kSkeletonAdaptDiscretizationDistance);
        Distance length = start.distance(end);
        int N = std::max(1.0, length() / disc_dist());
        Distance dx = (end.x() - start.x()) / N;
        Distance dy = (end.y() - start.y()) / N;


        //! Discretize segment into set of nodes (location, bead width)
        QVector<QPair<Point, Distance>> nodes(N + 1);
        std::generate(nodes.begin(), nodes.end(), [&, n = 0] () mutable
        {
            Point node(start.x() + n * dx, start.y() + n * dy, 0);
            n++;

            Distance min_dist = std::numeric_limits<double>::max();
            for (Polygon &poly : m_geometry)
            {
                Polyline boundary = poly.toPolyline();

                for (uint i = 0, max = boundary.size() - 1; i < max; ++i)
                    min_dist = std::min(min_dist(), (double)get<0>(MathUtils::findClosestPointOnSegment(boundary[i], boundary[i + 1], node)));
            }

            return qMakePair(node, min_dist * 2.0); //! * 2.0 for bead width
        });
        nodes.last().first = end; //! Ensure last node is end point


        //! Retrieve control bead width and speed to define linear relationship
        Distance control_width = m_sb->setting<Distance>(Constants::ProfileSettings::Skeleton::kBeadWidth);
        Velocity control_speed = m_sb->setting<Velocity>(Constants::ProfileSettings::Skeleton::kSpeed);
        float m = -(control_speed / control_width)();
        float b = 2 * control_speed();


        //! Combine consecutive subsegments whose bead widths are within delta of each other
        Distance delta = control_width / 100;
        QVector<QSharedPointer<LineSegment>> segments;
        QPair<Point,Distance> segment_start = nodes.first();
        for (uint i = 1; i < N; ++i)
        {
            QPair<Point,Distance> segment_end = nodes[i];

            Distance lower = segment_start.second - delta, upper = segment_start.second + delta;
            if (segment_end.second < lower || segment_end.second > upper)
            {
                Distance bead_width = std::min(segment_start.second(), segment_end.second());
                Velocity speed = std::max(control_speed() / 100, m * bead_width() + b); //! Clamp min speed to 1% of control speed

                QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(segment_start.first, segment_end.first);
                segment->getSb()->setSetting(Constants::SegmentSettings::kWidth, bead_width);
                segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, speed);

                segments.append(segment);

                segment_start = segment_end;
            }
        }


        //! Ensure last subsegment end point is equal to segment end point
        if (segment_start.first != nodes.last().first)
        {
            Distance bead_width = std::min(segment_start.second(), nodes.last().second());
            Velocity speed = std::max(control_speed() / 100, m * bead_width() + b); //! Clamp min speed to 1% of control speed

            QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(segment_start.first, nodes.last().first);
            segment->getSb()->setSetting(Constants::SegmentSettings::kWidth, bead_width);
            segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, speed);

            segments.append(segment);
        }

        return segments;
    }

    Path Skeleton::createPath(Polyline line)
    {
        Distance width                  = m_sb->setting< Distance >(Constants::ProfileSettings::Skeleton::kBeadWidth);
        Distance height                 = m_sb->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight);
        Velocity speed                  = m_sb->setting< Velocity >(Constants::ProfileSettings::Skeleton::kSpeed);
        Acceleration acceleration       = m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kSkeleton);
        AngularVelocity extruder_speed  = m_sb->setting< AngularVelocity >(Constants::ProfileSettings::Skeleton::kExtruderSpeed);
        int material_number             = m_sb->setting< int >(Constants::MaterialSettings::MultiMaterial::kPerimterNum);

        Path newPath;

        for (uint i = 0, end = line.size() - 1; i < end; ++i)
        {
            QVector<QSharedPointer<LineSegment>> segments;

            if(m_sb->setting< bool >(Constants::ProfileSettings::Skeleton::kSkeletonAdapt) //! Adaptive bead width
                    && m_sb->setting<Distance>(Constants::ProfileSettings::Skeleton::kSkeletonAdaptDiscretizationDistance) > 0)
            {
                segments = adaptBeadWidth(line[i], line[i + 1]);
            }
            else //! Static bead width
            {
                QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(line[i], line[i + 1]);
                segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            width);
                segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            speed);
                segments += segment;
            }

            for (QSharedPointer<LineSegment> &segment : segments)
            {
                segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           height);
                segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            acceleration);
                segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    extruder_speed);
                segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   material_number);
                segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       RegionType::kSkeleton);

                if(m_computed_anchor_lines.size() != 0)
                    segment->getSb()->setSetting(Constants::SegmentSettings::kWireFeed, true);

                newPath.append(segment);
            }
        }

        return newPath;
    }

    void Skeleton::setAnchorWireFeed(QVector<Polyline> anchor_lines)
    {
        m_computed_anchor_lines = anchor_lines;
    }

    void Skeleton::optimize(int layerNumber, Point& current_location, QVector<Path>& innerMostClosedContour, QVector<Path>& outerMostClosedContour, bool& shouldNextPathBeCCW)
    {
        PolylineOrderOptimizer poo(current_location, layerNumber);

        PathOrderOptimization pathOrderOptimization = static_cast<PathOrderOptimization>(
                    this->getSb()->setting<int>(Constants::ProfileSettings::Optimizations::kPathOrder));
        if(pathOrderOptimization == PathOrderOptimization::kCustomPoint)
        {
            Point startOverride(getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPathXLocation),
                                getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPathYLocation));

            poo.setStartOverride(startOverride);
        }

        PointOrderOptimization pointOrderOptimization = static_cast<PointOrderOptimization>(
                    this->getSb()->setting<int>(Constants::ProfileSettings::Optimizations::kPointOrder));

        if(pointOrderOptimization == PointOrderOptimization::kCustomPoint)
        {
            Point startOverride(getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPointXLocation),
                                getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPointYLocation));

            poo.setStartPointOverride(startOverride);
        }

        //! Uncomment if erroneous skeletons are being generated outside geometry
        if (!outerMostClosedContour.isEmpty())
        {
            PolygonList _outerMostClosedContour;
            for (const Path &path : outerMostClosedContour)
                _outerMostClosedContour += Polygon(path);

            //! Removes skeletons generated outside outerMostClosedContour
            QVector<Polyline> containedPaths;
            for (Polyline line : m_computed_geometry)
                if (std::all_of(line.begin(), line.end(), [_outerMostClosedContour] (const Point &pt) mutable {return _outerMostClosedContour.inside(pt);}))
                    containedPaths += line;
            m_computed_geometry = containedPaths;
        }

        poo.setPointParameters(pointOrderOptimization, getSb()->setting<bool>(Constants::ProfileSettings::Optimizations::kMinDistanceEnabled),
                               getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kMinDistanceThreshold),
                               getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kConsecutiveDistanceThreshold),
                               getSb()->setting<bool>(Constants::ProfileSettings::Optimizations::kLocalRandomnessEnable),
                               getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kLocalRandomnessRadius));

        poo.setGeometryToEvaluate(m_computed_geometry, RegionType::kSkeleton, static_cast<PathOrderOptimization>(m_sb->setting<int>(Constants::ProfileSettings::Optimizations::kPathOrder)));

        while(poo.getCurrentPolylineCount() > 0)
        {
            Polyline result = poo.linkNextPolyline();
            if(result.size() > 0)
            {
                Path newPath = createPath(result);
                QVector<Path> paths = breakPath(newPath);
                if(paths.size() > 0)
                {
                    for(Path path : paths)
                    {
                        QVector<Path> temp_path;
                        calculateModifiers(path, m_sb->setting<bool>(Constants::PrinterSettings::MachineSetup::kSupportG3), temp_path);
                        PathModifierGenerator::GenerateTravel(path, current_location, m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed));
                        current_location = path.back()->end();
                        m_paths.push_back(path);
                    }
                }
            }
        }
    }

    QVector<Path> Skeleton::breakPath(Path path)
    {
        QVector<Path> paths;
        //! Filter adapted path by removing segments whose widths are not within the tolerated range
        if (m_sb->setting< bool >(Constants::ProfileSettings::Skeleton::kSkeletonAdapt))
        {
            Distance min_width = m_sb->setting< Distance >(Constants::ProfileSettings::Skeleton::kSkeletonMinWidth);
            Distance max_width = m_sb->setting< Distance >(Constants::ProfileSettings::Skeleton::kSkeletonMaxWidth);

            Path filtered_path;

            for (QSharedPointer<SegmentBase>& segment : path)
            {
                Distance width = segment->getSb()->setting<Distance>(Constants::SegmentSettings::kWidth);

                if (width >= min_width && width <= max_width) //! Within tolerated range
                {
                    filtered_path.append(segment);
                }
                else if (width > max_width)
                {
                    segment->getSb()->setSetting(Constants::SegmentSettings::kWidth, max_width);
                    filtered_path.append(segment);
                }
                else //! Outside tolerated range
                {
                    if (filtered_path.size() > 0)
                    {
                        if (filtered_path.isClosed())
                            filtered_path.setCCW(Polygon(filtered_path).orientation());

                        paths.append(filtered_path);
                        filtered_path.clear();
                    }
                }
            }

            if (filtered_path.size() > 0)
            {
                if (filtered_path.isClosed())
                    filtered_path.setCCW(Polygon(filtered_path).orientation());

                paths.append(filtered_path);
            }
        }
        else //! Static bead width
        {
            if (path.isClosed())
                path.setCCW(Polygon(path).orientation());

            paths.append(path);
        }

        return paths;
    }

    void Skeleton::calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour)
    {
        if(m_sb->setting<bool>(Constants::ExperimentalSettings::Ramping::kTrajectoryAngleEnabled))
        {
            PathModifierGenerator::GenerateTrajectorySlowdown(path, m_sb);
        }

        if(m_sb->setting<bool>(Constants::MaterialSettings::Slowdown::kSkeletonEnable))
        {
            PathModifierGenerator::GenerateSlowdown(path, m_sb->setting<Distance>(Constants::MaterialSettings::Slowdown::kSkeletonDistance),
                                                    m_sb->setting<Distance>(Constants::MaterialSettings::Slowdown::kSkeletonLiftDistance),
                                                    m_sb->setting<Distance>(Constants::MaterialSettings::Slowdown::kSkeletonCutoffDistance),
                                                    m_sb->setting<Velocity>(Constants::MaterialSettings::Slowdown::kSkeletonSpeed),
                                                    m_sb->setting<AngularVelocity>(Constants::MaterialSettings::Slowdown::kSkeletonExtruderSpeed),
                                                    m_sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableWidthHeight),
                                                    m_sb->setting<double>(Constants::MaterialSettings::Slowdown::kSlowDownAreaModifier));
        }

        if(m_sb->setting<bool>(Constants::MaterialSettings::TipWipe::kSkeletonEnable))
        {
            if(static_cast<TipWipeDirection>(m_sb->setting<int>(Constants::MaterialSettings::TipWipe::kSkeletonDirection)) == TipWipeDirection::kForward ||
                    static_cast<TipWipeDirection>(m_sb->setting<int>(Constants::MaterialSettings::TipWipe::kSkeletonDirection)) == TipWipeDirection::kOptimal)
                PathModifierGenerator::GenerateForwardTipWipeOpenLoop(path, PathModifiers::kForwardTipWipe, m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkeletonDistance),
                                                                      m_sb->setting<Velocity>(Constants::MaterialSettings::TipWipe::kSkeletonSpeed),
                                                                      m_sb->setting<AngularVelocity>(Constants::MaterialSettings::TipWipe::kSkeletonExtruderSpeed),
                                                                      m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkeletonLiftHeight),
                                                                      m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkeletonCutoffDistance));
            else if(static_cast<TipWipeDirection>(m_sb->setting<int>(Constants::MaterialSettings::TipWipe::kSkeletonDirection)) == TipWipeDirection::kAngled)
            {
                PathModifierGenerator::GenerateTipWipe(path, PathModifiers::kAngledTipWipe, m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkeletonDistance),
                                                       m_sb->setting<Velocity>(Constants::MaterialSettings::TipWipe::kSkeletonSpeed),
                                                       m_sb->setting<Angle>(Constants::MaterialSettings::TipWipe::kSkeletonAngle),
                                                       m_sb->setting<AngularVelocity>(Constants::MaterialSettings::TipWipe::kSkeletonExtruderSpeed),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkeletonLiftHeight),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkeletonCutoffDistance));
            }
            else
                PathModifierGenerator::GenerateTipWipe(path, PathModifiers::kReverseTipWipe, m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkeletonDistance),
                                                       m_sb->setting<Velocity>(Constants::MaterialSettings::TipWipe::kSkeletonSpeed),
                                                       m_sb->setting<Angle>(Constants::MaterialSettings::TipWipe::kSkeletonAngle),
                                                       m_sb->setting<AngularVelocity>(Constants::MaterialSettings::TipWipe::kSkeletonExtruderSpeed),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkeletonLiftHeight),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkeletonCutoffDistance));
        }

        if(m_sb->setting<bool>(Constants::MaterialSettings::Startup::kSkeletonEnable))
        {
            if(m_sb->setting<bool>(Constants::MaterialSettings::Startup::kSkeletonRampUpEnable))
            {
                PathModifierGenerator::GenerateInitialStartupWithRampUp(path, m_sb->setting<Distance>(Constants::MaterialSettings::Startup::kSkeletonDistance),
                                                              m_sb->setting<Velocity>(Constants::MaterialSettings::Startup::kSkeletonSpeed),
                                                              m_sb->setting<Velocity>(Constants::ProfileSettings::Skeleton::kSpeed),
                                                              m_sb->setting<AngularVelocity>(Constants::MaterialSettings::Startup::kSkeletonExtruderSpeed),
                                                              m_sb->setting<AngularVelocity>(Constants::ProfileSettings::Skeleton::kExtruderSpeed),
                                                              m_sb->setting<int>(Constants::MaterialSettings::Startup::kSkeletonSteps),
                                                              m_sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableWidthHeight),
                                                              m_sb->setting<double>(Constants::MaterialSettings::Startup::kStartUpAreaModifier));
            }
            else
            {
                PathModifierGenerator::GenerateInitialStartup(path, m_sb->setting<Distance>(Constants::MaterialSettings::Startup::kSkeletonDistance),
                                                              m_sb->setting<Velocity>(Constants::MaterialSettings::Startup::kSkeletonSpeed),
                                                              m_sb->setting<AngularVelocity>(Constants::MaterialSettings::Startup::kSkeletonExtruderSpeed),
                                                              m_sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableWidthHeight),
                                                              m_sb->setting<double>(Constants::MaterialSettings::Startup::kStartUpAreaModifier));
            }
        }
    }
}
