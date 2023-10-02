//Header
#include "optimizers/path_order_optimizer.h"

//Qt
#include <QRandomGenerator>

//Local
#include <algorithms/knn.h>
#include "geometry/segments/travel.h"
#include "geometry/segments/line.h"
#include "utilities/mathutils.h"
#include "geometry/polygon_list.h"

namespace ORNL
{
    PathOrderOptimizer::PathOrderOptimizer(Point& start, uint layer_number, const QSharedPointer<SettingsBase>& sb)
        : m_current_location(start)
        , m_layer_number(layer_number)
        , m_sb(sb)
        , m_override_used(false)
    {
        m_layer_num = layer_number;
    }

    Point& PathOrderOptimizer::getCurrentLocation()
    {
        return m_current_location;
    }

    int PathOrderOptimizer::getCurrentPathCount()
    {
        return m_paths.size();
    }

    bool PathOrderOptimizer::getCurrentCCW()
    {
        return m_should_next_path_be_ccw;
    }

    void PathOrderOptimizer::setPathsToEvaluate(QVector<Path> paths)
    {
        m_paths = paths;
        for(Path& path : m_paths)
            path.removeTravels();

        if(paths.size() > 0)
            m_current_region_type = paths.front().getSegments().front()->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType);
        else
            m_current_region_type = RegionType::kUnknown;
    }

    void PathOrderOptimizer::setParameters(InfillPatterns infillPattern, PolygonList border_geometry)
    {
        m_pattern = infillPattern;
        m_border_geometry = border_geometry;
    }

    void PathOrderOptimizer::setParameters(PolygonList previousIslands)
    {

    }

    void PathOrderOptimizer::setParameters(bool shouldNextPathBeCCW)
    {
        m_should_next_path_be_ccw = shouldNextPathBeCCW;
    }

    void PathOrderOptimizer::setStartOverride(Point pt)
    {
        m_override_location = pt;
        m_override_used = true;
    }

    Path PathOrderOptimizer::linkNextPath(QVector<Path> paths)
    {
        if(m_paths.size() > 0)
        {
            switch(m_current_region_type)
            {
                case RegionType::kInfill:
                case RegionType::kSkin:
                    return linkNextInfillPath(paths);
                break;

                case RegionType::kSkeleton:
                    return linkNextSkeletonPath();
                break;

                default:
                    return linkTo();
                break;
            }
        }
    }

    Path PathOrderOptimizer::linkNextInfillPath(QVector<Path>& paths)
    {
        Point savedLocation = m_current_location;

        bool infill_enabled = m_sb->setting<bool>(Constants::ProfileSettings::Infill::kEnable);
        bool alternating_lines = m_sb->setting<bool>(Constants::ProfileSettings::Infill::kEnableAlternatingLines);

        Path nextPath;
        switch (m_pattern)
        {
            case InfillPatterns::kLines:
                if((infill_enabled && alternating_lines))
                    nextPath = linkNextInfillTravel(paths);
                else
                    nextPath = linkNextInfillLines(paths);
            break;
            case InfillPatterns::kGrid:
                nextPath = linkNextInfillLines(paths);
            break;
            case InfillPatterns::kConcentric:
            case InfillPatterns::kInsideOutConcentric:
                nextPath = linkNextInfillConcentric();
            break;
            case InfillPatterns::kTriangles:
                nextPath = linkNextInfillLines(paths);
            break;
            case InfillPatterns::kHexagonsAndTriangles:
                nextPath = linkNextInfillLines(paths);
            break;
            case InfillPatterns::kHoneycomb:
                nextPath = linkNextInfillLines(paths);
            break;
            case InfillPatterns::kRadialHatch:
                nextPath = linkNextInfillConcentric();
            break;
            default:
                nextPath = linkNextInfillLines(paths);
            break;
        }

        Distance minDist;
        if (nextPath.back()->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType) == RegionType::kInfill)
            minDist = m_sb->setting< Distance >(Constants::ProfileSettings::Infill::kMinPathLength);
        else if (nextPath.back()->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType) == RegionType::kSkin)
            minDist = m_sb->setting< Distance >(Constants::ProfileSettings::Skin::kMinPathLength);

        if(nextPath.calculateLengthNoTravel() < minDist)
            nextPath.clear();

        //if there are no segments reset start location
        if (nextPath.size() > 0)
            m_current_location = nextPath.back()->end();
        else
            m_current_location = savedLocation;

        return nextPath;
    }

    Path PathOrderOptimizer::linkNextInfillTravel(QVector<Path>& paths)
    {
        Path new_path;
        QPair<int, bool> indexAndStart = closestOpenPath(m_paths);
        int index = indexAndStart.first;

        //if false, indicates index is closest if you start at the end point, so reverse
        if(indexAndStart.second == false)
            m_paths[index].reverseSegments();

        QSharedPointer<TravelSegment> travel_segment = QSharedPointer<TravelSegment>::create(m_current_location, m_paths[index].front()->start());
        Velocity velocity = m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed);
        travel_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, velocity);
        new_path.append(travel_segment);

        for (QSharedPointer<SegmentBase> seg : m_paths[index])
            new_path.append(seg);

        m_current_location = new_path.back()->end();
        m_paths.remove(index);

        QVector<Path> empty_paths;
        PolygonList empty_polygon_list;

        for(int i = 0, end = m_paths.size(); i < end; ++i)
        {
            if(m_paths.size() > 0)
            {
                indexAndStart = closestOpenPath(m_paths);
                index = indexAndStart.first;
                //if false, indicates index is closest if you start at the end point, so reverse
                if(indexAndStart.second == false)
                    m_paths[index].reverseSegments();

                Point link_start = m_current_location;
                Point link_end = m_paths[index].front()->start();

                // Create a travel
                QSharedPointer<TravelSegment> travel = QSharedPointer<TravelSegment>::create(link_start, link_end);
                Velocity velocity = m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed);
                travel->getSb()->setSetting(Constants::SegmentSettings::kSpeed, velocity);

                if (!(linkIntersects(link_start, link_end, empty_paths, m_border_geometry) || linkIntersects(link_start, link_end, m_paths, empty_polygon_list) ||
                      linkIntersects(link_start, link_end, paths, empty_polygon_list) || linkIntersects(link_start, link_end, QVector<Path> { new_path }, empty_polygon_list))
                        && link_start.distance(link_end) < m_sb->setting< int >(Constants::ProfileSettings::Travel::kMinLength))
                {
                    travel->setLiftType(TravelLiftType::kNoLift);
                }

                new_path.append(travel);

                for (QSharedPointer<SegmentBase> seg : m_paths[index])
                    new_path.append(seg);

                m_current_location = new_path.back()->end();
                m_paths.remove(index);

                i = 0;
            }
        }

        return new_path;
    }

    Path PathOrderOptimizer::linkNextInfillLines(QVector<Path>& paths)
    {
        //! Gather settings for line segment links
        Distance bead_width = m_paths.front().front()->getSb()->setting<Distance>(Constants::SegmentSettings::kWidth);
        Distance layer_height = m_paths.front().front()->getSb()->setting<Distance>(Constants::SegmentSettings::kHeight);
        Velocity speed = m_paths.front().front()->getSb()->setting<Velocity>(Constants::SegmentSettings::kSpeed);
        Acceleration acceleration = m_paths.front().front()->getSb()->setting<Acceleration>(Constants::SegmentSettings::kAccel);
        AngularVelocity extruder_speed = m_paths.front().front()->getSb()->setting<AngularVelocity>(Constants::SegmentSettings::kExtruderSpeed);

        Path new_path;
        QPair<int, bool> indexAndStart = closestOpenPath(m_paths);
        int index = indexAndStart.first;

        //if false, indicates index is closest if you start at the end point, so reverse
        if(indexAndStart.second == false)
            m_paths[index].reverseSegments();

        QVector<Path> empty_paths;
        PolygonList empty_polygon_list;

        QSharedPointer<TravelSegment> travel_segment = QSharedPointer<TravelSegment>::create(m_current_location, m_paths[index].front()->start());

        if(m_sb->setting<bool>(Constants::ProfileSettings::Infill::kEnableAlternatingLines) && m_sb->setting<bool>(Constants::ProfileSettings::Infill::kEnable))
        {
            if (!(linkIntersects(m_current_location, m_paths[index].front()->start(), empty_paths, m_border_geometry) || linkIntersects(m_current_location, m_paths[index].front()->start(), m_paths, empty_polygon_list) ||
                  linkIntersects(m_current_location, m_paths[index].front()->start(), paths, empty_polygon_list) || linkIntersects(m_current_location, m_paths[index].front()->start(), QVector<Path> { new_path }, empty_polygon_list)))
            {
                travel_segment->setLiftType(TravelLiftType::kNoLift);
            }
        }

        Velocity velocity = m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed);
        travel_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, velocity);
        new_path.append(travel_segment);

        for (QSharedPointer<SegmentBase> seg : m_paths[index])
            new_path.append(seg);

        m_current_location = new_path.back()->end();
        m_paths.remove(index);

        for(int i = 0, end = m_paths.size(); i < end; ++i)
        {
            if(m_paths.size() > 0)
            {
                indexAndStart = closestOpenPath(m_paths);
                index = indexAndStart.first;
                //if false, indicates index is closest if you start at the end point, so reverse
                if(indexAndStart.second == false)
                    m_paths[index].reverseSegments();

                Point link_start = m_current_location;
                Point link_end = m_paths[index].front()->start();

                // If link intersects the border geometry, always travel. If link intersects the infill/skin paths, check versus minimum travel distance to see if travel or link is needed

                if (!(linkIntersects(link_start, link_end, empty_paths, m_border_geometry) || linkIntersects(link_start, link_end, m_paths, empty_polygon_list) ||
                      linkIntersects(link_start, link_end, paths, empty_polygon_list) || linkIntersects(link_start, link_end, QVector<Path> { new_path }, empty_polygon_list))
                        && link_start.distance(link_end) < m_sb->setting< int >(Constants::ProfileSettings::Travel::kMinLength))
                {
                    QSharedPointer<LineSegment> line_segment = QSharedPointer<LineSegment>::create(link_start, link_end);

                    line_segment->getSb()->setSetting(Constants::SegmentSettings::kWidth, bead_width);
                    line_segment->getSb()->setSetting(Constants::SegmentSettings::kHeight, layer_height);
                    line_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, speed);
                    line_segment->getSb()->setSetting(Constants::SegmentSettings::kAccel, acceleration);
                    line_segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed, extruder_speed);
                    line_segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,
                                                      m_paths[index].front()->getSb()->setting<int>(Constants::SegmentSettings::kMaterialNumber));
                    line_segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,
                                                      m_paths[index].front()->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType));

                    new_path.append(line_segment);

                    for (QSharedPointer<SegmentBase> seg : m_paths[index])
                        new_path.append(seg);

                    m_current_location = new_path.back()->end();
                    m_paths.remove(index);

                    i = 0;
                }
            }
        }

        return new_path;
    }

    Path PathOrderOptimizer::linkNextInfillConcentric()
    {
        Path new_path;

        //! \note Future work: travels aren't always needed between concentric paths, only needed when link distance is longer than
        //! \note travel distance or when link/travel segment crosses paths or border geometry
        if(m_paths.size() > 0)
            new_path = linkTo();

        return new_path;
    }

    Path PathOrderOptimizer::linkNextSkeletonPath()
    {
        Path new_path;
        if(!m_paths.isEmpty())
        {
            QPair<int, bool> location = closestOpenPath(m_paths);
            int index = location.first;
            bool start = location.second;

            new_path.setCCW(m_paths[index].getCCW());

            if (start)
            {
                QSharedPointer<TravelSegment> travel_segment = QSharedPointer<TravelSegment>::create(m_current_location, m_paths[index].front()->start());
                Velocity velocity = m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed);
                travel_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, velocity);
                new_path.append(travel_segment);

                for (QSharedPointer<SegmentBase> seg : m_paths[index])
                    new_path.append(seg);
            }
            else // End
            {
                QSharedPointer<TravelSegment> travel_segment = QSharedPointer<TravelSegment>::create(m_current_location, m_paths[index].back()->end());
                Velocity velocity = m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed);
                travel_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, velocity);
                new_path.append(travel_segment);

                QList<QSharedPointer<SegmentBase>> segments = m_paths[index].getSegments();
                while (!segments.isEmpty())
                {
                    QSharedPointer<SegmentBase> seg = segments.back();
                    seg->reverse();
                    new_path.append(seg);
                    segments.removeLast();
                }
            }
            m_current_location = new_path.back()->end();
            m_paths.remove(index);
        }
        return new_path;
    }

    bool PathOrderOptimizer::linkIntersects(Point link_start, Point link_end, QVector<Path> infill_geometry, PolygonList border_geometry)
    {
        //! Check for possible intersections with infill geometry
        for (Path path: infill_geometry)
            for (QSharedPointer<SegmentBase> seg : path)
                if (link_start != seg->end() && link_end != seg->start())
                    if (seg.dynamicCast<TravelSegment>().isNull() && MathUtils::intersect(link_start, link_end, seg->start(), seg->end()))
                        return true;

        //! Check for possible intersections with border geometry
        for (Polygon poly : border_geometry)
        {
            for (int i = 0, end = poly.size() - 1; i < end; ++i)
                if (MathUtils::intersect(link_start, link_end, poly[i], poly[i + 1]))
                    return true;

            //! Check last line of polygon
            if (MathUtils::intersect(link_start, link_end, poly.last(), poly.first()))
                return true;
        }

        return false;
    }


    QPair<int, bool> PathOrderOptimizer::closestOpenPath(QVector<Path> paths)
    {
        Distance shortest = Distance(std::numeric_limits<float>::max());

        int index = 0;
        bool start;

        for (int i = 0, end = paths.size(); i < end; ++i)
        {
            if (m_current_location.distance(paths[i].front()->start()) < shortest)
            {
                shortest = m_current_location.distance(paths[i].front()->start());
                index = i;
                start = true;
            }

            if (m_current_location.distance(paths[i].back()->end()) < shortest)
            {
                shortest = m_current_location.distance(paths[i].back()->end());
                index = i;
                start = false;
            }
        }
        return QPair<int, bool>(index, start);
    }

    void PathOrderOptimizer::addTravel(int index, Path &path)
    {
        Point current_start = path[index]->start();
        if(m_sb->setting<bool>(Constants::ProfileSettings::Optimizations::kCustomLocationRandomnessEnable))
        {
            Distance radius = m_sb->setting<Distance>(Constants::ProfileSettings::Optimizations::kCustomLocationLocalRandomnessRadius);
            QList<QSharedPointer<SegmentBase>> segments = path.getSegments();
            QVector<int> candidates;

            for(int i = 0; i < segments.size(); ++i)
            {
                if(segments[i]->start().distance(current_start) < radius)
                    candidates.push_back(i);
            }
            if(candidates.size() > 1)
                index = candidates[QRandomGenerator::global()->bounded(candidates.size())];
        }

        QSharedPointer<TravelSegment> travel_segment = QSharedPointer<TravelSegment>::create(m_current_location, path[index]->start());
        Velocity velocity = m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed);
        travel_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, velocity);

        if(path.size() > 0 && path[0]->getSb()->contains(Constants::SegmentSettings::kTilt))
        {
                travel_segment->getSb()->setSetting(Constants::SegmentSettings::kTilt,
                                                    path[0]->getSb()->setting<QVector<QVector3D>>(Constants::SegmentSettings::kTilt));
                travel_segment->getSb()->setSetting(Constants::SegmentSettings::kCCW,
                                                    path[0]->getSb()->setting<bool>(Constants::SegmentSettings::kCCW));
        }

        m_current_location = path[index]->start();
        for(int i = 0; i < index; ++i)
        {
            path.move(0, path.size() - 1);
        }
        path.prepend(travel_segment);
    }

    Path PathOrderOptimizer::linkTo()
    {
        QPair<int, int> result;
        PathOrderOptimization orderOptimization = static_cast<PathOrderOptimization>(m_sb->setting<int>(Constants::ProfileSettings::Optimizations::kPathOrder));
        switch (orderOptimization) {
            case PathOrderOptimization::kNextClosest:
                result = findShortestOrLongestDistance();
                break;
            case PathOrderOptimization::kNextFarthest:
                result = findShortestOrLongestDistance(false);
                break;
            case PathOrderOptimization::kLeastRecentlyVisited:
                result = linkToLeastRecentlyVisited();
                break;
            case PathOrderOptimization::kRandom:
                result = linkToRandom();
                break;
            case PathOrderOptimization::kConsecutive:
                result = linkToConsecutive();
                break;
            default:
                result = findShortestOrLongestDistance();
                break;
        }

        Path nextPath = m_paths[result.second];
        m_paths.removeAt(result.second);

        setRotation(nextPath);

        addTravel(result.first, nextPath);

        m_current_location = nextPath.back()->end();
        return nextPath;
    }

    QPair<int, int> PathOrderOptimizer::findShortestOrLongestDistance(bool shortest)
    {
        Point queryPoint;
        if(m_override_used)
            queryPoint = m_override_location;
        else
            queryPoint = m_current_location;

        int pathIndex, startIndex;
        Distance closest;
        if(shortest)
            closest = Distance(DBL_MAX);

        for(int i = 0, end = m_paths.size(); i < end; ++i)
        {
            for(int j = 0, end2 = m_paths[i].size(); j < end2; ++j)
            {
                QSharedPointer<SegmentBase> seg = m_paths[i][j];
                Distance closestSegment = MathUtils::distanceFromLineSegSqrd(queryPoint, seg->start(), seg->end());
                if(shortest)
                {
                    if (closestSegment < closest)
                    {
                        closest = closestSegment;
                        pathIndex = i;
                        if(seg->start().distance(queryPoint) < seg->end().distance(queryPoint))
                            startIndex = j;
                        else
                            startIndex = (j + 1) % end2;
                    }
                }
                else
                {
                    if (closestSegment > closest)
                    {
                        closest = closestSegment;
                        pathIndex = i;
                        if(seg->start().distance(queryPoint) > seg->end().distance(queryPoint))
                            startIndex = j;
                        else
                            startIndex = (j + 1) % end2;
                    }
                }
            }
        }

        return QPair<int, int>(startIndex, pathIndex);
    }

    QPair<int, int> PathOrderOptimizer::linkToLeastRecentlyVisited()
    {
        int startIndex = m_links_done % m_paths.size();
        QPair<int, int> results = findShortestOrLongestDistance();
        results.first = startIndex;
        m_links_done++;
        return results;
    }

    QPair<int, int> PathOrderOptimizer::linkToRandom()
    {
        int pathIndex = QRandomGenerator::global()->bounded(m_paths.size());
        int startIndex = QRandomGenerator::global()->bounded(m_paths[pathIndex].size());
        return QPair<int, int>(startIndex, pathIndex);
    }

    QPair<int, int> PathOrderOptimizer::linkToConsecutive()
    {
        QPair<int, int> results = findShortestOrLongestDistance();

        int startIndex = m_layer_number - 2;
        if(startIndex < 0)
            startIndex += m_paths[results.second].size();
        else
            startIndex %= m_paths[results.second].size();

        int previousIndex = startIndex;

        Distance dist;
        Distance minDist = m_sb->setting<Distance>(Constants::ProfileSettings::Optimizations::kConsecutiveDistanceThreshold);
        do
        {
            startIndex = (startIndex + 1) % m_paths[results.second].size();
            QSharedPointer<SegmentBase> previous = m_paths[results.second].at(previousIndex);
            QSharedPointer<SegmentBase> current = m_paths[results.second].at(startIndex);
            dist += previous->start().distance(current->start());

            //looped through whole path
            if(startIndex == previousIndex)
            {
                break;
            }

        } while(dist < minDist);

        results.first = startIndex;
        return results;
    }

    Path PathOrderOptimizer::linkSpiralPath2D(bool last_spiral)
    {
        QPair<int, int> results = findShortestOrLongestDistance();
        Path newPath = m_paths[results.second];
        m_paths.removeAt(results.second);

        for(int i = 0; i < results.first; ++i)
            newPath.move(0, newPath.size() - 1);

        Distance layer_height = m_sb->setting<Distance>(Constants::ProfileSettings::Layer::kLayerHeight);

        Distance pathLength = newPath.calculateLength();
        Distance currentLength;

        int start = 0;
        if(!last_spiral)
        {
            addTravel(0, newPath);
            start = 1;
        }

        for(int end = newPath.size(); start < end; ++start)
        {
            QSharedPointer<SegmentBase> seg = newPath[start];
            Point startPt = seg->start();
            Point endPt = seg->end();

            currentLength += startPt.distance(m_current_location);
            startPt.z(startPt.z() + ((currentLength / pathLength) * layer_height)());
            seg->setStart(startPt);

            m_current_location.x(startPt.x());
            m_current_location.y(startPt.y());

            currentLength += endPt.distance(m_current_location);
            endPt.z(endPt.z() + ((currentLength / pathLength) * layer_height)());
            seg->setEnd(endPt);

            m_current_location.x(endPt.x());
            m_current_location.y(endPt.y());

        }
        return newPath;
    }

    void PathOrderOptimizer::linkSpiralPaths3D(QVector<Path>& spiral_paths)
    {
        Path linked_path;
        QVector3D tilt(0, 0, 0);

        for (Path& spiral_path : spiral_paths)
        {
            //! If spiral path end is closest to current location, reverse path
            if (m_current_location.distance(spiral_path.back()->end()) < m_current_location.distance(spiral_path.front()->start()))
                spiral_path.reverseSegments();

            //! Add travel segment link & set tilt to average of segments being linked
            QSharedPointer<TravelSegment> travel_segment = QSharedPointer<TravelSegment>::create(m_current_location, spiral_path.front()->start());
            travel_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed));
            tilt += spiral_path.front()->getSb()->setting<QVector<QVector3D>>(Constants::SegmentSettings::kTilt)[0];
            tilt.normalize();
            travel_segment->getSb()->setSetting(Constants::SegmentSettings::kTilt, QVector<QVector3D>{tilt, tilt});
            linked_path.append(travel_segment);

            //! Add spiral path to linked path
            for (QSharedPointer<SegmentBase>& segment : spiral_path)
                linked_path.append(segment);

            tilt = linked_path.back()->getSb()->setting<QVector<QVector3D>>(Constants::SegmentSettings::kTilt)[0];
            m_current_location = linked_path.back()->end();
        }

        spiral_paths.clear();
        spiral_paths.append(linked_path);
    }

    void PathOrderOptimizer::setRotation(Path& path)
    {
        Point rotation_origin = Point(m_sb->setting<Distance>(Constants::ExperimentalSettings::RotationOrigin::kXOffset),
                                      m_sb->setting<Distance>(Constants::ExperimentalSettings::RotationOrigin::kYOffset));
        bool shouldRotate = m_sb->setting<bool>(Constants::PrinterSettings::MachineSetup::kSupportsE2);

        if(shouldRotate)
        {
            if(path.getCCW() != m_should_next_path_be_ccw)
            {
                path.reverseSegments();
                path.setCCW(!path.getCCW());
            }
            m_should_next_path_be_ccw = !m_should_next_path_be_ccw;

            for(QSharedPointer<SegmentBase> seg : path.getSegments())
            {
                seg->getSb()->setSetting(Constants::SegmentSettings::kRotation, MathUtils::internalAngle(seg->start(), rotation_origin, seg->end()));
            }

        }
        if(m_sb->setting<bool>(Constants::PrinterSettings::MachineSetup::kSupportsE1))
        {
            for(QSharedPointer<SegmentBase>& seg : path.getSegments())
            {
                if(path.getCCW())
                {
                    Point newPoint = seg->start();
                    newPoint.reverseNormals();
                    seg->setStart(newPoint);
                }
                seg->getSb()->setSetting(Constants::SegmentSettings::kTilt, seg->start().getNormals());
                seg->getSb()->setSetting(Constants::SegmentSettings::kCCW, path.getCCW());
            }
        }
    }
}

////Header
//#include "optimizers/path_order_optimizer.h"

////Qt
//#include <QRandomGenerator>

////Local
//#include <algorithms/knn.h>
//#include "geometry/segments/travel.h"
//#include "geometry/segments/line.h"
//#include "utilities/mathutils.h"
//#include "geometry/polygon_list.h"

//namespace ORNL
//{
//    PathOrderOptimizer::PathOrderOptimizer(Point& start, uint layer_number, const QSharedPointer<SettingsBase>& sb)
//        : m_sb(sb)
//        , m_current_location(start)
//        , m_layer_number(layer_number)
//        , m_override_used(true)
//    {
//    }

//    Point& PathOrderOptimizer::getCurrentLocation()
//    {
//        return m_current_location;
//    }

//    int PathOrderOptimizer::getCurrentPathCount()
//    {
//        return m_paths.size();
//    }

//    void PathOrderOptimizer::setPathsToEvaluate(QVector<Path> paths)
//    {
//        m_paths = paths;
//        for(Path& path : m_paths)
//            path.removeTravels();
//    }

//    void PathOrderOptimizer::setInfillParameters(InfillPatterns infillPattern, PolygonList border_geometry)
//    {
//        m_pattern = infillPattern;
//        m_border_geometry = border_geometry;
//    }

//    void PathOrderOptimizer::setStartOverride(Point pt)
//    {
//        m_override_location = pt;
//    }

//    void PathOrderOptimizer::resetOverride()
//    {
//        m_override_used = false;
//    }

//    Path PathOrderOptimizer::linkNextInfillPath(QVector<Path>& paths)
//    {
//        //! \note Only run optimization if we have paths
//        if(m_paths.size() > 0)
//        {
//            Point savedLocation = m_current_location;

//            Path nextPath;
//            switch (m_pattern)
//            {
//                case InfillPatterns::kLines:
//                    nextPath = linkNextInfillLines(paths);
//                    break;
//                case InfillPatterns::kGrid:
//                      nextPath = linkNextInfillLines(paths);
//                    break;
//                case InfillPatterns::kConcentric:
//                case InfillPatterns::kInsideOutConcentric:
//                    nextPath = linkNextInfillConcentric();
//                    break;
//                case InfillPatterns::kTriangles:
//                      nextPath = linkNextInfillLines(paths);
//                    break;
//                case InfillPatterns::kHexagonsAndTriangles:
//                      nextPath = linkNextInfillLines(paths);
//                    break;
//                case InfillPatterns::kHoneycomb:
//                    nextPath = linkNextInfillLines(paths);
//                    break;
//                case InfillPatterns::kRadialHatch:
//                    nextPath = linkNextInfillConcentric();
//                    break;
//                default:
//                    nextPath = linkNextInfillLines(paths);
//                    break;
//            }

//            Distance minDist;
//            if (nextPath.back()->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType) == RegionType::kInfill)
//                minDist = m_sb->setting< Distance >(Constants::ProfileSettings::Infill::kMinPathLength);
//            else if (nextPath.back()->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType) == RegionType::kSkin)
//                minDist = m_sb->setting< Distance >(Constants::ProfileSettings::Skin::kMinPathLength);

//            if(nextPath.calculateLengthNoTravel() < minDist)
//                nextPath.clear();

//            //if there are no segments reset start location
//            if (nextPath.size() > 0)
//                m_current_location = nextPath.back()->end();
//            else
//                m_current_location = savedLocation;

//            return nextPath;
//        }
//    }

//    void PathOrderOptimizer::linkInfill(QVector<Path>& paths, InfillPatterns& infillPattern, PolygonList& border_geometry)
//    {
//        //! \note Only run optimization if we have paths
//        if(paths.size() > 0)
//        {
//            for(Path& path : paths)
//                path.removeTravels();

//            Point savedLocation = m_current_location;

//            QVector<Path> combinedPaths;
//            switch (infillPattern)
//            {
//                case InfillPatterns::kLines:
//                    combinedPaths = linkInfillLines(paths, border_geometry);
//                    break;
//                case InfillPatterns::kGrid:
//                    combinedPaths = linkInfillLines(paths, border_geometry);
//                    break;
//                case InfillPatterns::kConcentric:
//                case InfillPatterns::kInsideOutConcentric:
//                    combinedPaths = linkInfillConcentric(paths);
//                    break;
//                case InfillPatterns::kTriangles:
//                    combinedPaths = linkInfillLines(paths, border_geometry);
//                    break;
//                case InfillPatterns::kHexagonsAndTriangles:
//                    combinedPaths = linkInfillLines(paths, border_geometry);
//                    break;
//                case InfillPatterns::kHoneycomb:
//                    combinedPaths = linkInfillHoneycomb(paths, border_geometry);
//                    break;
//                case InfillPatterns::kRadialHatch:
//                    combinedPaths = linkInfillConcentric(paths);
//                    break;
//                default:
//                    combinedPaths = linkInfillLines(paths, border_geometry);
//                    break;
//            }

//            Distance minDist;
//            if (combinedPaths.back().back()->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType) == RegionType::kInfill)
//                minDist = m_sb->setting< Distance >(Constants::ProfileSettings::Infill::kMinPathLength);
//            else if (combinedPaths.back().back()->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType) == RegionType::kSkin)
//                minDist = m_sb->setting< Distance >(Constants::ProfileSettings::Skin::kMinPathLength);

//            for(int i = combinedPaths.size() - 1; i >= 0; --i)
//            {
//                if (combinedPaths[i].calculateLengthNoTravel() < minDist)
//                {
//                    combinedPaths.remove(i);
//                }
//            }

//            //if there are no segments reset start location
//            if (combinedPaths.size() > 0)
//                m_current_location = combinedPaths.back().back()->end();
//            else
//                m_current_location = savedLocation;

//            paths.clear();
//            for (Path path : combinedPaths)
//                paths.append(path);
//        }
//    }

//    QVector<Path> PathOrderOptimizer::linkInfillLines(QVector<Path> paths, PolygonList border_geometry)
//    {
//        //! Gather settings for line segment links
//        Distance bead_width = paths.front().front()->getSb()->setting<Distance>(Constants::SegmentSettings::kWidth);
//        Distance layer_height = paths.front().front()->getSb()->setting<Distance>(Constants::SegmentSettings::kHeight);
//        Velocity speed = paths.front().front()->getSb()->setting<Velocity>(Constants::SegmentSettings::kSpeed);
//        Acceleration acceleration = paths.front().front()->getSb()->setting<Acceleration>(Constants::SegmentSettings::kAccel);
//        AngularVelocity extruder_speed = paths.front().front()->getSb()->setting<AngularVelocity>(Constants::SegmentSettings::kExtruderSpeed);

//        QVector<Path> new_paths;
//        new_paths.reserve(paths.size());

//        Path linked_path;
//        int index = closestPath(paths);

//        QSharedPointer<TravelSegment> travel_segment = QSharedPointer<TravelSegment>::create(m_current_location, paths[index].front()->start());
//        Velocity velocity = m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed);
//        travel_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, velocity);
//        linked_path.append(travel_segment);

//        for (QSharedPointer<SegmentBase> seg : paths[index])
//            linked_path.append(seg);

//        m_current_location = linked_path.back()->end();
//        paths.remove(index);
//        new_paths.append(linked_path);

//        QVector<Path> empty_paths;
//        PolygonList empty_polygon_list;

//        while (paths.size() > 0)
//        {
//            QPair<int, bool> indexAndStart = closestOpenPath(paths);
//            int index = indexAndStart.first;
//            //if false, indicates index is closest if you start at the end point, so reverse
//            if(indexAndStart.second == false)
//                paths[index].reverseSegments();

//            Point link_start = m_current_location;
//            Point link_end = paths[index].front()->start();

//            // If link intersects the border geometry, always travel. If link intersects the infill/skin paths, check versus minimum travel distance to see if travel or link is needed
//            if ((linkIntersects(link_start, link_end, empty_paths, border_geometry) || linkIntersects(link_start, link_end, paths, empty_polygon_list) ||
//                linkIntersects(link_start, link_end, new_paths, empty_polygon_list)) && link_start.distance(link_end) > m_sb->setting< int >(Constants::ProfileSettings::Travel::kMinLength))
//            {
//                QSharedPointer<TravelSegment> travel_segment = QSharedPointer<TravelSegment>::create(link_start, link_end);
//                Velocity velocity = m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed);
//                travel_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, velocity);

//                Path temp_path;
//                temp_path.append(travel_segment);

//                for (QSharedPointer<SegmentBase> seg : paths[index])
//                    temp_path.append(seg);

//                m_current_location = temp_path.back()->end();
//                paths.remove(index);
//                new_paths.append(temp_path);
//            }
//            else
//            {
//                QSharedPointer<LineSegment> line_segment = QSharedPointer<LineSegment>::create(link_start, link_end);

//                line_segment->getSb()->setSetting(Constants::SegmentSettings::kWidth, bead_width);
//                line_segment->getSb()->setSetting(Constants::SegmentSettings::kHeight, layer_height);
//                line_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, speed);
//                line_segment->getSb()->setSetting(Constants::SegmentSettings::kAccel, acceleration);
//                line_segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed, extruder_speed);
//                line_segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,
//                                                  paths[index].front()->getSb()->setting<int>(Constants::SegmentSettings::kMaterialNumber));
//                line_segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,
//                                                  paths[index].front()->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType));

//                new_paths.last().append(line_segment);

//                for (QSharedPointer<SegmentBase> seg : paths[index])
//                    new_paths.last().append(seg);

//                m_current_location = new_paths.last().back()->end();
//                paths.remove(index);
//            }
//        }

//        new_paths.squeeze();
//        return new_paths;
//    }

//    Path PathOrderOptimizer::linkNextInfillLines(QVector<Path>& paths)
//    {
//        //! Gather settings for line segment links
//        Distance bead_width = m_paths.front().front()->getSb()->setting<Distance>(Constants::SegmentSettings::kWidth);
//        Distance layer_height = m_paths.front().front()->getSb()->setting<Distance>(Constants::SegmentSettings::kHeight);
//        Velocity speed = m_paths.front().front()->getSb()->setting<Velocity>(Constants::SegmentSettings::kSpeed);
//        Acceleration acceleration = m_paths.front().front()->getSb()->setting<Acceleration>(Constants::SegmentSettings::kAccel);
//        AngularVelocity extruder_speed = m_paths.front().front()->getSb()->setting<AngularVelocity>(Constants::SegmentSettings::kExtruderSpeed);

//        Path new_path;
//        QPair<int, bool> indexAndStart = closestOpenPath(m_paths);
//        int index = indexAndStart.first;
//        //if false, indicates index is closest if you start at the end point, so reverse
//        if(indexAndStart.second == false)
//            m_paths[index].reverseSegments();

//        QSharedPointer<TravelSegment> travel_segment = QSharedPointer<TravelSegment>::create(m_current_location, m_paths[index].front()->start());
//        Velocity velocity = m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed);
//        travel_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, velocity);
//        new_path.append(travel_segment);

//        for (QSharedPointer<SegmentBase> seg : m_paths[index])
//            new_path.append(seg);

//        m_current_location = new_path.back()->end();
//        m_paths.remove(index);

//        QVector<Path> empty_paths;
//        PolygonList empty_polygon_list;

//        for(int i = 0, end = m_paths.size(); i < end; ++i)
//        {
//            if(m_paths.size() > 0)
//            {
//                indexAndStart = closestOpenPath(m_paths);
//                index = indexAndStart.first;
//                //if false, indicates index is closest if you start at the end point, so reverse
//                if(indexAndStart.second == false)
//                    m_paths[index].reverseSegments();

//                Point link_start = m_current_location;
//                Point link_end = m_paths[index].front()->start();

//                // If link intersects the border geometry, always travel. If link intersects the infill/skin paths, check versus minimum travel distance to see if travel or link is needed

//                if (!(linkIntersects(link_start, link_end, empty_paths, m_border_geometry) || linkIntersects(link_start, link_end, m_paths, empty_polygon_list) ||
//                      linkIntersects(link_start, link_end, paths, empty_polygon_list) || linkIntersects(link_start, link_end, QVector<Path> { new_path }, empty_polygon_list))
//                        && link_start.distance(link_end) < m_sb->setting< int >(Constants::ProfileSettings::Travel::kMinLength))
//                {
//                    QSharedPointer<LineSegment> line_segment = QSharedPointer<LineSegment>::create(link_start, link_end);

//                    line_segment->getSb()->setSetting(Constants::SegmentSettings::kWidth, bead_width);
//                    line_segment->getSb()->setSetting(Constants::SegmentSettings::kHeight, layer_height);
//                    line_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, speed);
//                    line_segment->getSb()->setSetting(Constants::SegmentSettings::kAccel, acceleration);
//                    line_segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed, extruder_speed);
//                    line_segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,
//                                                      m_paths[index].front()->getSb()->setting<int>(Constants::SegmentSettings::kMaterialNumber));
//                    line_segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,
//                                                      m_paths[index].front()->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType));

//                    new_path.append(line_segment);

//                    for (QSharedPointer<SegmentBase> seg : m_paths[index])
//                        new_path.append(seg);

//                    m_current_location = new_path.back()->end();
//                    m_paths.remove(index);

//                    i = 0;
//                }
//            }
//        }

//        return new_path;
//    }

//    QVector<Path> PathOrderOptimizer::linkInfillConcentric(QVector<Path> paths)
//    {
//        Path combined_path;
//        QVector<Path> new_paths;
//        new_paths.reserve(paths.size());

//        //! \note Future work: travels aren't always needed between concentric paths, only needed when link distance is longer than
//        //! \note travel distance or when link/travel segment crosses paths or border geometry

//        for(Path& path : paths)
//        {
//            this->linkTo(path);
//            for (QSharedPointer<SegmentBase>& segment : path)
//            {
//                combined_path.append(segment);
//            }
//            new_paths.append(combined_path);
//            combined_path.clear();
//        }

//        new_paths.squeeze();
//        return new_paths;
//    }

//    Path PathOrderOptimizer::linkNextInfillConcentric()
//    {
//        Path new_path;

//        //! \note Future work: travels aren't always needed between concentric paths, only needed when link distance is longer than
//        //! \note travel distance or when link/travel segment crosses paths or border geometry
//        if(m_paths.size() > 0)
//        {
//            this->linkTo(m_paths[0]);
//            new_path = m_paths[0];
//            m_paths.pop_front();
//        }
//        return new_path;
//    }

//    QVector<Path> PathOrderOptimizer::linkInfillHoneycomb(QVector<Path> paths, PolygonList border_geometry)
//    {
//        Path combined_path;
//        QVector<Path> new_paths;
//        new_paths.reserve(paths.size());

//        //! \note Converts vector of paths to a single linked path
//        while(paths.size()>0)
//        {
//            QVector<Point> starts;
//            QVector<Point> ends;
//            QVector<Point> query;
//            query.push_back(m_current_location);

//            for(Path path : paths)
//            {
//                starts.push_back(path.front()->start());
//                ends.push_back(path.back()->end());
//            }

//            kNN knnStart(starts,query,1);
//            kNN knnEnd(ends,query,1);

//            knnStart.execute();
//            knnEnd.execute();

//            Distance startDist = knnStart.getNearestDistances().first();
//            Distance endDist = knnEnd.getNearestDistances().first();

//            int indexShortestPath;

//            Path closest_path;
//            if(startDist < endDist)
//            {
//                indexShortestPath = knnStart.getNearestIndices().first();
//                closest_path = paths[indexShortestPath];
//            }else
//            {
//                indexShortestPath = knnEnd.getNearestIndices().first();
//                closest_path = paths[indexShortestPath];

//                //! \note Flips all segments and paths over
//                for (QSharedPointer<SegmentBase> segment : closest_path) {
//                    segment->reverse();
//                }
//                std::reverse(closest_path.begin(),closest_path.end());
//            }

//            //! \note If linking path is longer than the minimum travel distance or if linking path intersects the border, add a travel
//            QVector<Path> empty_paths;
//            if (m_current_location.distance(closest_path.front()->start()) > m_sb->setting< int >(Constants::ProfileSettings::Travel::kMinLength) ||
//                    linkIntersects(m_current_location, closest_path.front()->start(), empty_paths, border_geometry))
//            {
//                QSharedPointer<TravelSegment> travel_segment = QSharedPointer<TravelSegment>::create(m_current_location, closest_path.front()->start());
//                Velocity velocity = m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed);
//                travel_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, velocity);
//                combined_path.append(travel_segment);
//            }
//            else
//            {
//                QSharedPointer<LineSegment> link_segment = QSharedPointer<LineSegment>::create(m_current_location, closest_path.front()->start());
//                link_segment->getSb()->setSetting(Constants::SegmentSettings::kWidth, closest_path.front()->getSb()->setting<Distance>(Constants::SegmentSettings::kWidth));
//                link_segment->getSb()->setSetting(Constants::SegmentSettings::kHeight, closest_path.front()->getSb()->setting<Distance>(Constants::SegmentSettings::kHeight));
//                link_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, closest_path.front()->getSb()->setting<Velocity>(Constants::SegmentSettings::kSpeed));
//                link_segment->getSb()->setSetting(Constants::SegmentSettings::kAccel, closest_path.front()->getSb()->setting<Acceleration>(Constants::SegmentSettings::kAccel));
//                link_segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed, closest_path.front()->getSb()->setting<AngularVelocity>(Constants::SegmentSettings::kExtruderSpeed));
//                link_segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,
//                                                  closest_path.front()->getSb()->setting<int>(Constants::SegmentSettings::kMaterialNumber));
//                link_segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,
//                                                  closest_path.front()->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType));

//                combined_path.append(link_segment);
//            }


//            for(QSharedPointer<SegmentBase> segment : closest_path)
//            {
//                combined_path.append(segment);
//            }


//            paths.removeAt(indexShortestPath);
//            m_current_location = combined_path.back()->end();

//        }
//        paths.push_back(combined_path);
//        new_paths.append(combined_path);
//        new_paths.squeeze();
//        return new_paths;
//    }

//    void PathOrderOptimizer::linkSkeletonPaths(QVector<Path> &paths)
//    {
//        Path linked_path;

//        while (!paths.isEmpty())
//        {
//            QPair<int, bool> location = closestOpenPath(paths);
//            int index = location.first;
//            bool start = location.second;

//            if (start)
//            {
//                QSharedPointer<TravelSegment> travel_segment = QSharedPointer<TravelSegment>::create(m_current_location, paths[index].front()->start());
//                Velocity velocity = m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed);
//                travel_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, velocity);
//                linked_path.append(travel_segment);

//                for (QSharedPointer<SegmentBase> seg : paths[index])
//                    linked_path.append(seg);

//                m_current_location = linked_path.back()->end();
//                paths.remove(index);
//            }
//            else // End
//            {
//                QSharedPointer<TravelSegment> travel_segment = QSharedPointer<TravelSegment>::create(m_current_location, paths[index].back()->end());
//                Velocity velocity = m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed);
//                travel_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, velocity);
//                linked_path.append(travel_segment);

//                QList<QSharedPointer<SegmentBase>> segments = paths[index].getSegments();
//                while (!segments.isEmpty())
//                {
//                    QSharedPointer<SegmentBase> seg = segments.back();
//                    seg->reverse();
//                    linked_path.append(seg);
//                    segments.removeLast();
//                }

//                m_current_location = linked_path.front()->start();
//                paths.remove(index);
//            }
//        }

//        paths.append(linked_path);
//    }

//    Path PathOrderOptimizer::linkNextSkeletonPath()
//    {
//        Path new_path;
//        if(!m_paths.isEmpty())
//        {
//            QPair<int, bool> location = closestOpenPath(m_paths);
//            int index = location.first;
//            bool start = location.second;

//            if (start)
//            {
//                QSharedPointer<TravelSegment> travel_segment = QSharedPointer<TravelSegment>::create(m_current_location, m_paths[index].front()->start());
//                Velocity velocity = m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed);
//                travel_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, velocity);
//                new_path.append(travel_segment);

//                for (QSharedPointer<SegmentBase> seg : m_paths[index])
//                    new_path.append(seg);
//            }
//            else // End
//            {
//                QSharedPointer<TravelSegment> travel_segment = QSharedPointer<TravelSegment>::create(m_current_location, m_paths[index].back()->end());
//                Velocity velocity = m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed);
//                travel_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, velocity);
//                new_path.append(travel_segment);

//                QList<QSharedPointer<SegmentBase>> segments = m_paths[index].getSegments();
//                while (!segments.isEmpty())
//                {
//                    QSharedPointer<SegmentBase> seg = segments.back();
//                    seg->reverse();
//                    new_path.append(seg);
//                    segments.removeLast();
//                }
//            }
//            m_current_location = new_path.back()->end();
//            m_paths.remove(index);
//        }
//        return new_path;
//    }

//    bool PathOrderOptimizer::linkIntersects(Point link_start, Point link_end, QVector<Path> infill_geometry, PolygonList border_geometry)
//    {
//        //! Check for possible intersections with infill geometry
//        for (Path path: infill_geometry)
//            for (QSharedPointer<SegmentBase> seg : path)
//                if (link_start != seg->end() && link_end != seg->start())
//                    if (seg.dynamicCast<TravelSegment>().isNull() && MathUtils::intersect(link_start, link_end, seg->start(), seg->end()))
//                        return true;

//        //! Check for possible intersections with border geometry
//        for (Polygon poly : border_geometry)
//        {
//            for (int i = 0, end = poly.size() - 1; i < end; ++i)
//                if (MathUtils::intersect(link_start, link_end, poly[i], poly[i + 1]))
//                    return true;

//            //! Check last line of polygon
//            if (MathUtils::intersect(link_start, link_end, poly.last(), poly.first()))
//                return true;
//        }

//        return false;
//    }

//    int PathOrderOptimizer::closestPath(QVector<Path> paths)
//    {
//        Distance shortest = Distance(std::numeric_limits<float>::max());

//        int index = 0;

//        //! Find closest path to current location
//        for (int i = 0, end = paths.size(); i < end; ++i)
//        {
//            if (m_current_location.distance(paths[i].front()->start()) < shortest)
//            {
//                shortest = m_current_location.distance(paths[i].front()->start());
//                index = i;
//            }
//        }

//        return index;
//    }

//    QPair<int, bool> PathOrderOptimizer::closestOpenPath(QVector<Path> paths)
//    {
//        Distance shortest = Distance(std::numeric_limits<float>::max());

//        int index = 0;
//        bool start;

//        for (int i = 0, end = paths.size(); i < end; ++i)
//        {
//            if (m_current_location.distance(paths[i].front()->start()) < shortest)
//            {
//                shortest = m_current_location.distance(paths[i].front()->start());
//                index = i;
//                start = true;
//            }

//            if (m_current_location.distance(paths[i].back()->end()) < shortest)
//            {
//                shortest = m_current_location.distance(paths[i].back()->end());
//                index = i;
//                start = false;
//            }
//        }
//        return QPair<int, bool>(index, start);
//    }

//    void PathOrderOptimizer::addTravel(int index, Path &path)
//    {
//        Point current_start = path[index]->start();
//        if(m_sb->setting<bool>(Constants::ProfileSettings::Optimizations::kCustomLocationRandomnessEnable))
//        {
//            Distance radius = m_sb->setting<Distance>(Constants::ProfileSettings::Optimizations::kCustomLocationLocalRandomnessRadius);
//            QList<QSharedPointer<SegmentBase>> segments = path.getSegments();
//            QVector<int> candidates;

//            for(int i = 0; i < segments.size(); ++i)
//            {
//                if(segments[i]->start().distance(current_start) < radius)
//                    candidates.push_back(i);
//            }
//            if(candidates.size() > 1)
//                index = candidates[QRandomGenerator::global()->bounded(candidates.size())];
//        }

//        QSharedPointer<TravelSegment> travel_segment = QSharedPointer<TravelSegment>::create(m_current_location, path[index]->start());
//        Velocity velocity = m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed);
//        travel_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, velocity);

//        if(path.size() > 0 && path[0]->getSb()->contains(Constants::SegmentSettings::kTilt))
//        {
//                travel_segment->getSb()->setSetting(Constants::SegmentSettings::kTilt,
//                                                    path[0]->getSb()->setting<QVector<QVector3D>>(Constants::SegmentSettings::kTilt));
//                travel_segment->getSb()->setSetting(Constants::SegmentSettings::kCCW,
//                                                    path[0]->getSb()->setting<bool>(Constants::SegmentSettings::kCCW));
//        }

//        m_current_location = path[index]->start();
//        for(int i = 0; i < index; ++i)
//        {
//            path.move(0, path.size() - 1);
//        }
//        path.prepend(travel_segment);
//    }

//    void PathOrderOptimizer::linkTo(Path &path)
//    {
//        path.removeTravels();

//        PathOrderOptimization orderOptimization = static_cast<PathOrderOptimization>(m_sb->setting<int>(Constants::ProfileSettings::Optimizations::kPathOrder));
//        switch (orderOptimization) {
//            case PathOrderOptimization::kNextClosest:
//                linkToShortestDistance(path);
//                break;
//            case PathOrderOptimization::kNextFarthest:
//                linkToLargestDistance(path);
//                break;
//            case PathOrderOptimization::kLeastRecentlyVisited:
//                linkToLeastRecentlyVisited(path);
//                break;
//            case PathOrderOptimization::kRandom:
//                linkToRandom(path);
//                break;
//            case PathOrderOptimization::kConsecutive:
//                linkToConsecutive(path);
//                break;
//            default:
//                linkToShortestDistance(path);
//                break;
//        }

//        m_current_location = path.back()->end();
//    }

//    void PathOrderOptimizer::linkToShortestDistance(Path& path)
//    {
//        QVector<Point> startPoints;
//        QVector<Point> queryPoints;
//        for (QSharedPointer<SegmentBase> seg : path)
//        {
//            startPoints.push_back(seg->start());
//        }
//        if(!m_override_used)
//        {
//            queryPoints.push_back(m_override_location);
//            m_override_used = true;
//        }
//        else
//            queryPoints.push_back(m_current_location);

//        kNN kNN(startPoints, queryPoints, 1);
//        kNN.execute();

//        int closestStartIndex = kNN.getNearestIndices().first();
//        this->addTravel(closestStartIndex, path);
//    }

//    void PathOrderOptimizer::linkToLargestDistance(Path& path)
//    {
//        Distance longest = 0.0;
//        int longestStartIndex = 0;

//        for(int i = 0, end = path.size(); i < end; ++i)
//        {
//            QSharedPointer<SegmentBase> seg = path[i];
//            if(dynamic_cast<LineSegment*>(seg.get()))
//            {
//                Distance currDist = seg->start().distance(m_current_location);
//                if(currDist > longest)
//                {
//                    longest = currDist;
//                    longestStartIndex = i;
//                }
//            }
//        }

//        this->addTravel(longestStartIndex, path);
//    }

//    void PathOrderOptimizer::linkToLeastRecentlyVisited(Path& path)
//    {
//        int startIndex = m_links_done % path.size();

//        this->addTravel(startIndex, path);

//        m_links_done++;
//    }

//    void PathOrderOptimizer::linkToRandom(Path& path)
//    {
//        int randomStartIndex = QRandomGenerator::global()->bounded(path.size());

//        this->addTravel(randomStartIndex, path);
//    }

//    void PathOrderOptimizer::linkToConsecutive(Path& path)
//    {
//        if(m_first_point_linked){
//            this->linkToShortestDistance(path);
//        }else{
//            int startIndex = m_layer_number - 2;
//            if(startIndex < 0)
//                startIndex += path.size();
//            else
//                startIndex %= path.size();

//            int previousIndex = startIndex;

//            Distance dist;
//            Distance minDist = m_sb->setting<Distance>(Constants::ProfileSettings::Optimizations::kConsecutiveDistanceThreshold);
//            do
//            {
//                startIndex = (startIndex + 1) % path.size();
//                QSharedPointer<SegmentBase> previous = path.at(previousIndex);
//                QSharedPointer<SegmentBase> current = path.at(startIndex);
//                dist += previous->start().distance(current->start());

//                //looped through whole path
//                if(startIndex == previousIndex)
//                {
//                    startIndex = previousIndex;
//                    break;
//                }

//            } while(dist < minDist);

//            this->addTravel(startIndex, path);

//            m_first_point_linked = true;
//        }
//    }

//    void PathOrderOptimizer::linkSpiral(Path& path, bool last_spiral)
//    {
//        findClosestPointAndReorder(path);

//        Distance layer_height = m_sb->setting<Distance>(Constants::ProfileSettings::Layer::kLayerHeight);

//        Distance pathLength = path.calculateLength();
//        Distance currentLength;

//        int start = 0;
//        if(!last_spiral)
//        {
//            addTravel(0, path);
//            start = 1;
//        }

//        for(int end = path.size(); start < end; ++start)
//        {
//            QSharedPointer<SegmentBase> seg = path[start];
//            Point startPt = seg->start();
//            Point endPt = seg->end();

//            currentLength += startPt.distance(m_current_location);
//            startPt.z(startPt.z() + ((currentLength / pathLength) * layer_height)());
//            seg->setStart(startPt);

//            m_current_location.x(startPt.x());
//            m_current_location.y(startPt.y());

//            currentLength += endPt.distance(m_current_location);
//            endPt.z(endPt.z() + ((currentLength / pathLength) * layer_height)());
//            seg->setEnd(endPt);

//            m_current_location.x(endPt.x());
//            m_current_location.y(endPt.y());

//        }
//    }

//    void PathOrderOptimizer::findClosestPointAndReorder(Path& path)
//    {
//        Distance minDist(std::numeric_limits<double>::max());
//        int index;
//        int totalSegments = path.size();
//        for(int i = 0; i < totalSegments; ++i)
//        {
//            Distance currentMin = path[i]->start().distance(m_current_location);
//            if(currentMin < minDist)
//            {
//                index = i;
//                minDist = currentMin;
//            }
//        }

//        for(int i = 0; i < index; ++i)
//            path.move(0, totalSegments - 1);
//    }
//}
