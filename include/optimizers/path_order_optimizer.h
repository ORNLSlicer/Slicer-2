//! \author Charles Wade

#ifndef PATHORDEROPTIMIZER_H
#define PATHORDEROPTIMIZER_H

#include "utilities/enums.h"
#include "configs/settings_base.h"
#include "geometry/path.h"
#include "geometry/point.h"
#include "geometry/polygon_list.h"

namespace ORNL {

    class TravelSegment;
    /*!
     * \class PathOrderOptimizer
     * \brief This class is used to link together paths with travel segments.
     *
     * \note These are how the optimizations are handled:
     * \note Non-implemented optimizations default to shortest distance
     * \list kShortestDistance: finds the closest point to link to
     * \list kLargestDistance: finds the farthest point to link to
     * \list kLeastRecentlyVisited: consecutively links to vertices inside a layer to guarantee
     *       the link to point was the least recently visited
     * \list kNextClosest: not implemented
     * \list kApproximateShortest: not implemented
     * \list kShortestDistance_DP: not implemented
     * \list kRandom: links to a random vertex each time
     * \list kConsecutive: links to the next vertex consecutivly based on layer number.
     *       after first point in layer is linked to, uses kShortestDistance.
     */
    class PathOrderOptimizer{
        public:
             //! \brief Constructor
             //! \note Layer number is only needed if kConsecutive linking is used
            PathOrderOptimizer(Point& start, uint layer_number, const QSharedPointer<SettingsBase>& sb);

            //! \brief Links next path based on path's region type: infill/skin, skeleton, or perimeter/inset
            //! \param paths: Already existing paths to consider for intersection. Used by infill/skin patterns
            //! to determine whether to insert travels or build moves between paths.
            //! \return Linked path with travel
            Path linkNextPath(QVector<Path> paths = QVector<Path>());

            //! \brief Set paths to evaluate
            //! \param paths Copy of paths to evaluate
            void setPathsToEvaluate(QVector<Path> paths);

            //! \brief Set parameters (when computing infill)
            //! \param infillPattern Pattern to evaluate
            //! \param border_geometry Geometry to consider for intersection testing
            void setParameters(InfillPatterns infillPattern, PolygonList border_geometry);

            //! \brief Set parameters (when computing least recently visited optimization strategy)
            //! \param previousIslands: list of previously visited islands
            void setParameters(PolygonList previousIslands);

            //! \brief Set parameters (when using additional DOF for perimeter/inset)
            //! \param shouldNextPathBeCCW: whether or not next path should be CW or CCW after travel is determined
            void setParameters(bool shouldNextPathBeCCW);

            //! \brief Gets remaining paths
            //! \return current paths remaining
            int getCurrentPathCount();

            //! \brief Gets current location
            //! \return current location
            Point& getCurrentLocation();

            //! \brief Gets whether current path is CW or CCW
            //! \return CW or CCW status
            bool getCurrentCCW();

            //! \brief Link path as part of spiral (works only for a single perimeter)
            //! \param path: Path to link
            //! \param last_spiral: whether last region was spiralized or not
            //! \return spiralized path
            Path linkSpiralPath2D(bool last_spiral);
   			//!\brief Link 3D spiral paths generated using Directed Perimeters & Layer Spiralization
            //! \param spiral_paths: spiral paths to link
            void linkSpiralPaths3D(QVector<Path>& spiral_paths);

            //! \brief Set start override for seam selection (custom start point)
            //! \param pt Seam point
            void setStartOverride(Point pt);

        private:

            //! \brief Links the POO position to the given path with a travel (used for closed contours)
            Path linkTo();

            //! \brief Links together and orders next, single infill path
            //! \param paths Paths to consider
            //! \return Next path linked via travel
            Path linkNextInfillPath(QVector<Path>& paths);

            //! \brief Links together and orders next, single skeleton path
            //! \return Next path linked via travel
            Path linkNextSkeletonPath();

            //! \brief Links single path in line infill
            //! \return Next path linked via travel
            Path linkNextInfillLines(QVector<Path>& paths);

            //! \brief Links single path in concentric infill
            //! \return Next path linked via travel
            Path linkNextInfillConcentric();

            //! \brief Determines whether a link intersects with the infill or border geometry
            //! \param link_start: start of link
            //! \param link_end: end of link
            //! \param infill_geometry: existing infill pathing to test against
            //! \param border_geometry: boundary geometry to test against
            //! \return whether or not intersection occurs
            bool linkIntersects(Point link_start, Point link_end, QVector<Path> infill_geometry, PolygonList border_geometry);

            //! \brief Finds the index and end of the the closest path to m_current_location
            //! \return Index for closest path and whether or not to link to beginning or end
            QPair<int, bool> closestOpenPath(QVector<Path> paths);

            //! \brief Adds a travel segment to the index point and shuffles the paths to be in order
            void addTravel(int index, Path& path);

            //! \brief Links to a path using shortest or longest distance
            //! \param shortest: whether to look for shortest or longest (shortest by default)
            //! \return Index for vertex in closest path and index for path itself
            QPair<int, int> findShortestOrLongestDistance(bool shortest = true);

            //! \brief Consecutively links to vertices inside a layer
            //! \return Index for vertex in closest path and index for path itself
            QPair<int, int> linkToLeastRecentlyVisited();

            //! \brief Links to a random vertex each time
            //! \return Index for vertex in closest path and index for path itself
            QPair<int, int> linkToRandom();

            //! \brief Links to the next vertex consecutivly based on layer number
            //! \return Index for vertex in closest path and index for path itself
            QPair<int, int> linkToConsecutive();

            //! \brief Sets rotation of path based on internal status of the previous paths' CW or CCW status
            void setRotation(Path& path);

            //! \brief Settings the optimizer will use.
            QSharedPointer<SettingsBase> m_sb;

            //! \brief The last point we linked
            Point& m_current_location;

            //! \brief The layer number the POO is on. Used by consecutive linker
            uint m_layer_number;

            //! \brief The layer number links completed by the least recently visited linker
            uint m_links_done;

            //! \brief Internal copy of the paths to evaluated
            QVector<Path> m_paths;

            //! \brief Internal copy of the pattern to evaluate
            InfillPatterns m_pattern;

            //! \brief Internal copy of the border geometry to test for intersections
            PolygonList m_border_geometry;

            //! \brief Track whether the override has been used or not.  Allows reset between regions
            bool m_override_used;

            //! \brief Override location to use for linking instead of current location
            Point m_override_location;

            //! \brief Current region type for path. Determines which version of link is called
            RegionType m_current_region_type;

            //! \brief CW or CCW status for next path to determine which direction to rotate
            bool m_should_next_path_be_ccw;
    };
}

#endif // PATHORDEROPTIMIZER_H

////! \author Charles Wade

//#ifndef PATHORDEROPTIMIZER_H
//#define PATHORDEROPTIMIZER_H

//#include "utilities/enums.h"
//#include "configs/settings_base.h"
//#include "geometry/path.h"
//#include "geometry/point.h"
//#include "geometry/polygon_list.h"

//namespace ORNL {

//    class TravelSegment;
//    /*!
//     * \class PathOrderOptimizer
//     * \brief This class is used to link together paths with travel segments.
//     *
//     * \note These are how the optimizations are handled:
//     * \note Non-implemented optimizations default to shortest distance
//     * \list kShortestDistance: finds the closest point to link to
//     * \list kLargestDistance: finds the farthest point to link to
//     * \list kLeastRecentlyVisited: consecutively links to vertices inside a layer to guarantee
//     *       the link to point was the least recently visited
//     * \list kNextClosest: not implemented
//     * \list kApproximateShortest: not implemented
//     * \list kShortestDistance_DP: not implemented
//     * \list kRandom: links to a random vertex each time
//     * \list kConsecutive: links to the next vertex consecutivly based on layer number.
//     *       after first point in layer is linked to, uses kShortestDistance.
//     */
//    class PathOrderOptimizer{
//        public:
//             //! \brief Constructor
//             //! \note Layer number is only needed if kConsecutive linking is used
//            PathOrderOptimizer(Point& start, uint layer_number, const QSharedPointer<SettingsBase>& sb);

//            //! \brief Links the POO position to the given path with a travel (used for concentric paths where there is
//            //! no question as to the ordering of paths
//            void linkTo(Path& path);

//            //! \brief (Mode 1) Links together and orders infill paths and returns a single path
//            //! Includes travels but does not allow modifiers to be easily added in the middle
//            //! \param paths Vector of paths to evaluate and link
//            //! \param infillPattern Pattern to evaluate
//            //! \param border_geometry Geometry to consider for intersection testing (whether to add travels or small infill segment)
//            void linkInfill(QVector<Path>& paths, InfillPatterns& infillPattern, PolygonList& border_geometry);

//            //! \brief (Mode 2) Links together and orders next, single infill path
//            //! Includes travels and allows modifiers to be easily added in the middle
//            //! Must have additional internal information set via calls to setPaths and setParameters
//            //! \param paths Paths to consider
//            //! \return Next path linked via travel
//            Path linkNextInfillPath(QVector<Path>& paths);

//            //! \brief (Mode 2) Set paths to evaluate
//            //! \param paths Copy of paths to evaluate
//            void setPathsToEvaluate(QVector<Path> paths);

//            //! \brief (Mode 2) Set parameters
//            //! \param infillPattern Pattern to evaluate
//            //! \param border_geometry Geometry to consider for intersection testing
//            void setInfillParameters(InfillPatterns infillPattern, PolygonList border_geometry);

//            //! \brief Gets remaining paths.  Necessary to externally control looping via Mode 2
//            //! \return current paths remaining
//            int getCurrentPathCount();

//            //! \brief (Mode 1) Links together and orders skeleton paths and returns a single path
//            //! Includes travels but does not allow modifiers to be easily added in the middle
//            //! \param paths Vector of paths to evaluate and link
//            void linkSkeletonPaths(QVector<Path> &paths);

//            //! \brief (Mode 2) Links together and orders next, single infill path
//            //! Includes travels and allows modifiers to be easily added in the middle
//            //! \return Next path linked via travel
//            Path linkNextSkeletonPath();

//            //! \brief (Mode 2) Gets current location
//            //! \return current location
//            Point& getCurrentLocation();

//            //! \brief Link path as part of spiral (works only for a single perimeter)
//            //! \param path: Path to link
//            //! \param last_spiral: whether last region was spiralized or not
//            void linkSpiral(Path& path, bool last_spiral);

//            //! \brief Set start override for seam selection (custom start point)
//            //! \param pt Seam point
//            void setStartOverride(Point pt);

//            //! \brief Reset override to use it again between regions
//            void resetOverride();
//        private:

//            //! \brief Find the closest point in a path and reorder path so that point is index 0
//            //! \param path Path to evaluate
//            void findClosestPointAndReorder(Path& path);

//            //! \brief Links current location to closest infill path
//            QVector<Path> linkInfillLines(QVector<Path> paths, PolygonList border_geometry);

//            //! \brief Links single path in line infill
//            Path linkNextInfillLines(QVector<Path>& paths);

//            //! \brief Links infill in the same fashion as insets
//            QVector<Path> linkInfillConcentric(QVector<Path> paths);

//             //! \brief Links single path in concentric infill
//            Path linkNextInfillConcentric();

//            //! \brief Links the honeycomb infill
//            QVector<Path> linkInfillHoneycomb(QVector<Path> paths, PolygonList border_geometry);

//            //! \brief Links single path in honeycomb infill
//            Path linkNextInfillHoneycomb();

//            //! \brief Determines whether a link intersects with the infill or border geometry
//            bool linkIntersects(Point link_start, Point link_end, QVector<Path> infill_geometry, PolygonList border_geometry);

//            //! \brief Finds the index of the closest path to m_current_location
//            int closestPath(QVector<Path> paths);

//            //! \brief Finds the index and end of the the closest path to m_current_location
//            QPair<int, bool> closestOpenPath(QVector<Path> paths);

//            //! \brief Adds a travel segment to the index point and shuffles the paths to be in order
//            void addTravel(int index, Path& path);

//            //! \brief Links to a path using based on the shortest distance to a point
//            void linkToShortestDistance(Path& path);

//            //! \brief Links to a path using based on the largest distance to a point
//            void linkToLargestDistance(Path& path);

//            //! \brief Consecutively links to vertices inside a layer
//            void linkToLeastRecentlyVisited(Path& path);

//            //! \brief Links to a random vertex each time
//            void linkToRandom(Path& path);

//            //! \brief Links to the next vertex consecutivly based on layer number
//            void linkToConsecutive(Path& path);

//            //! \brief Settings the optimizer will use.
//            QSharedPointer<SettingsBase> m_sb;

//            //! \brief The last point we linked
//            Point& m_current_location;

//            //! \brief If we have linked the first point druring on a layer
//            bool m_first_point_linked = false;

//            //! \brief The layer number the POO is on. Used by consecutive linker
//            uint m_layer_number;

//            //! \brief The layer number links completed by the least recently visited linker
//            uint m_links_done;

//            //! \brief Internal copy of the paths to evaluated (used for mode 2 where an individual
//            //! path is asked for one at a time)
//            QVector<Path> m_paths;

//            //! \brief Internal copy of the pattern to evaluate (used for mode 2 where an individual
//            //! path is asked for one at a time)
//            InfillPatterns m_pattern;

//            //! \brief Internal copy of the border geometry to test for intersections (used for mode 2 where an individual
//            //! path is asked for one at a time)
//            PolygonList m_border_geometry;

//            //! \brief Track whether the override has been used or not.  Allows reset between regions
//            bool m_override_used;

//            //! \brief Override location to use for linking instead of current location
//            Point m_override_location;
//    };
//}

//#endif // PATHORDEROPTIMIZER_H
