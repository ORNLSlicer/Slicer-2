#ifndef SKELETON_H
#define SKELETON_H

// Local
#include "step/layer/regions/region_base.h"
#include "geometry/segments/line.h"

// System
#include <QStack>

// Boost
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/graph_traits.hpp"
#include "boost/property_map/property_map.hpp"
#include "boost/graph/connected_components.hpp"
#include "boost/graph/filtered_graph.hpp"
#include "boost/graph/subgraph.hpp"

namespace ORNL {
    typedef boost::adjacency_list<boost::listS, boost::listS, boost::undirectedS, Point, Polyline> SkeletonGraph;
    typedef boost::graph_traits<SkeletonGraph>::vertex_descriptor SkeletonVertex;
    typedef boost::graph_traits<SkeletonGraph>::vertex_iterator Vertex_Iter;
    typedef boost::graph_traits<SkeletonGraph>::edge_descriptor Edge;
    typedef boost::graph_traits<SkeletonGraph>::out_edge_iterator Out_Edge_Iter;

    struct subgraph_filter {
        subgraph_filter() = default;

        subgraph_filter(QMap<SkeletonVertex, int> vertex_subgraph_map_) : vertex_subgraph_map(vertex_subgraph_map_){}

        bool operator()(const SkeletonVertex &v) const {
            return vertex_subgraph_map[v] == 0;
        }

        QMap<SkeletonVertex, int> vertex_subgraph_map;
    };

    typedef boost::filtered_graph<SkeletonGraph, boost::keep_all, subgraph_filter> SubGraph;

    class Skeleton : public RegionBase  {
        public:
            //! \brief Constructor
            //! \param sb: the settings
            //! \param index: index for region order
            //! \param settings_polygons: a vector of settings polygons to apply
            //! \param gridInfo: optional external file information
            Skeleton(const QSharedPointer<SettingsBase>& sb, const int index, const QVector<SettingsPolygon>& settings_polygons,
                     const SingleExternalGridInfo& gridInfo, bool iswireFed = false);

            //! \brief Writes the gcode for the skeleton
            //! \param writer is the instance of the Writer Base to be used for writing skeleton region GCode
            QString writeGCode(QSharedPointer<WriterBase> writer) override;

            //! \brief Computes the skeleton region
            void compute(uint layer_num, QSharedPointer<SyncManager>& sync) override;

            //! \brief Computes a Voronoi Diagram from a set of segments
            void computeSegmentVoronoi();

            //! \brief Computes a Voronoi Diagram from a set of points (Experimental)
            void computePointVoronoi();

            //! \brief Incorporates lost geometry into m_geometry
            void incorporateLostGeometry();

            //! \brief Cleans input geometry using ClipperLib2's cleanPolygons function
            void simplifyInputGeometry(const uint& layer_num);

            //! \brief Cleans output geometry according to ClipperLib2's cleanPolygons function
            void simplifyOutputGeometry();

            //! \brief Generates a graph representation of skeleton geometry for cleaning
            void generateSkeletonGraph();

            //! \brief Cleans graph of skeleton geometry according to ClipperLib's CleanPolygon function
            //! \param cleaning_distance is the determinant distance used for cleaning the skeleton graph
            //! \note Deprecated: cleaning now takes place in simplifyOutputGeometry
            void cleanSkeletonGraph(Distance cleaning_distance);

            //! \brief Extracts cycles from m_skeleton_graph
            void extractCycles();

            //! \brief Extracts simple paths from m_skeleton_graph
            void extractSimplePaths();

            //! \brief Extracts passed path from m_skeleton_graph
            //! \param path is the computed skeleton path to be extracted from the skeleton graph and placed in m_computed_geometry
            void extractPath(QVector<Edge> path);

            //! \brief Quick, unoptimized extraction of paths from m_skeleton_graph.
            //! Can be used in place of extractSkeletonPaths().
            void getSkeleton();

            /*!
             * \brief Used for internal inspection of skeleton structure contained in m_computed_geometry.
             * Due to the underlyting nature of Voronoi Diagrams, small skeleton segments may be generated
             * that will not be visible in the UI. To properly inspect these segments and their source geometry,
             * follow these instructions:
             *
             * 1.   Call inspectSkeleton() after cleanOutputGeometry() is called within compute().
             *
             * 2.   The ouput will appear as:
             *
             *      Input Geometry
             *      polygon((x1,y1),(x2,y2))
             *      ...
             *      Skeleton Geometry
             *      polygon((x1,y1),(x2,y2))
             *      ...
             *
             *      Copy this entire output including the "Input Geometry" and "Skeleton Geometry" text.
             *      It will help you later identify which a segment belongs to.
             *
             * 3.   Go to desmos.com/calculator
             *
             * 4.   On the left-hand side of the application you will see an area to input data.
             *      Select the first input row and paste the output from inspectSkeleton() using ctrl + v.
             *
             * 5.   You will see that the input has been entered but not initialized.
             *      To initialize the input, scroll all the way to the top input row and select it.
             *      Now hold down the tab button until you have iterated through all input rows.
             *      The input should now be initialized.
             *
             * 6.   The graph will be centered at (0,0) however the input will most likely be at a much grater scale.
             *      Continue to zoom out until the input becomes visible. Once visible, you can then zoom in on the
             *      skeleton structure to inspect it in detail.
             *
             *      Additionally, you may find it easier to inspect the skeleton by reversing the contrast of the application.
             *      On the right-hand side of the application select the wrench icon and then select Reverse Contrast.
             */
            void inspectSkeleton(const uint& layer_num);

            /*!
             * \brief Used for internal inspection of m_skeleton_graph.
             * To be used after generateSkeletonGraph is called and before skeleton extraction is complete.
             * Follows same instructions listed above.
             */
            void inspectSkeletonGraph();

            /*!
             * \brief Adapts skeleton beadWidth to fill remaining area.
             * Assumes skeleton beadWidth cannot be adapted to more than 1.9 times avgBeadWidth
             * \param start: start of skeleton segment to be adapted
             * \param end: end of skeleton segment to be adapted
             * \return Returns a vector of skeleton segments with adapted beadWidths expressed by their speed
             */
            QVector<QSharedPointer<LineSegment>> adaptBeadWidth(const Point &start, const Point &end);

            //! \brief Optimizes the region.
            //! \param layerNumber: current layer number
            //! \param innerMostClosedContour: used for subsequent path modifiers
            //! \param outerMostClosedContour: used for subsequent path modifiers
            //! \param current_location: most recent location
            //! \param shouldNextPathBeCCW: state as to CW or CCW of previous path for use with additional DOF
            void optimize(int layerNumber, Point& current_location, QVector<Path>& innerMostClosedContour,
                          QVector<Path>& outerMostClosedContour, bool& shouldNextPathBeCCW) override;

            //! \brief Creates paths for the skeleton region.
            //! \param line: polyline representing path
            //! \return Polyline converted to path
            Path createPath(Polyline line) override;

            //! \brief Filters adapted paths by clamping or removing segments whose bead widths are not within the
            //! allowable range.
            //! \param path: the path to be filtered.
            //! \return Returns a vector of paths with bead widths that are within the allowable range.
            QVector<Path> filterPath(Path& path);

            //! \brief Sets pathing for anchor lines
            //! \param anchor_lines: polylines for wire feed
            void setAnchorWireFeed(QVector<Polyline> anchor_lines);

        private:
            //! \brief Creates modifiers
            //! \param path Current path to add modifiers to
            //! \param supportsG3 Whether or not G2/G3 is supported for spiral lift
            //! \param innerMostClosedContour used for Prestarts (currently only skins/infill)
            void calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour) override;

            //! \brief Holds raw skeleton geometry produced by the Voronoi Generator
            QVector<Polyline> m_skeleton_geometry;

            //! \brief Holds graph representation of skeleton geometry
            SkeletonGraph m_skeleton_graph;

            //! \brief Holds computed geometry
            QVector<Polyline> m_computed_geometry;

            //! \brief Whether or not skeletons belong to wire fed areas or not
            bool m_wire_region;

            //! \brief Precomputed paths for wire feed at anchors
            QVector<Polyline> m_computed_anchor_lines;
    };
}

#endif // SKELETON_H
