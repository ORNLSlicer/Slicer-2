// Main Module
#include <QtMath>

#include "geometry/path_modifier.h"
#include "geometry/segments/line.h"
#include "units/unit.h"
#include "configs/settings_base.h"
#include "geometry/segment_base.h"
#include "utilities/mathutils.h"
#include "geometry/segments/arc.h"

namespace ORNL {

    void PathModifierGenerator::GeneratePreStart(Path &path, Distance prestartDistance, Velocity prestartSpeed, AngularVelocity prestartExtruderSpeed, QVector<Path>& outerPath)
    {
        Point closest;
        float dist = INT_MAX;
        int pathIndex, segmentIndex, firstNonTravelSegment = 0;
        Point firstPoint = path[firstNonTravelSegment]->start();
        if (dynamic_cast<TravelSegment*>(path[firstNonTravelSegment].data()))
        {
            firstNonTravelSegment = 1;
            firstPoint = path[firstNonTravelSegment]->start();
        }

        for(int i = 0, totalContours = outerPath.size(); i < totalContours; ++i)
        {
            for(int j = 0, totalSegments = outerPath[i].size(); j < totalSegments; ++j)
            {
                if (!dynamic_cast<TravelSegment*>(outerPath[i][j].data()))
                {
                    Point tempClosest;
                    float tempDist;
                    std::tie(tempDist, tempClosest) = MathUtils::findClosestPointOnSegment(outerPath[i][j]->start(), outerPath[i][j]->end(), firstPoint);
                    if(tempDist < dist)
                    {
                        closest = tempClosest;
                        dist = tempDist;
                        pathIndex = i;
                        segmentIndex = j;
                    }
                }
            }
        }

        //move to segment
        QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(closest, firstPoint);

        segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            path[firstNonTravelSegment]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth));
        segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           path[firstNonTravelSegment]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight));
        segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            prestartSpeed);
        segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            path[firstNonTravelSegment]->getSb()->setting< Acceleration >(Constants::SegmentSettings::kAccel));
        segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    prestartExtruderSpeed);
        segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       path[firstNonTravelSegment]->getSb()->setting< RegionType >(Constants::SegmentSettings::kRegionType));
        segment->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers,    PathModifiers::kPrestart);

        path.insert(1, segment);

        Distance nextSegmentDistStart = outerPath[pathIndex][segmentIndex]->end().distance(closest);
        Distance nextSegmentDistEnd = outerPath[pathIndex][segmentIndex]->start().distance(closest);

        prestartDistance -= firstPoint.distance(closest);

        bool forward = nextSegmentDistEnd > nextSegmentDistStart ? true : false;
        if(forward)
        {
            if(outerPath[pathIndex][segmentIndex]->end() == closest)
                ++segmentIndex;

            while(prestartDistance > 0)
            {
                if (!dynamic_cast<TravelSegment*>(outerPath[pathIndex][segmentIndex].data()))
                {
                    Distance nextSegmentDist = closest.distance(outerPath[pathIndex][segmentIndex]->end());
                    prestartDistance -= nextSegmentDist;

                    Point end;
                    if(prestartDistance >= 0)
                    {
                        end = outerPath[pathIndex][segmentIndex]->end();
                    }
                    else
                    {
                        float percentage = 1 - (-prestartDistance() / nextSegmentDist());
                        end = Point((1.0 - percentage) * outerPath[pathIndex][segmentIndex]->start().x() + percentage * outerPath[pathIndex][segmentIndex]->end().x(),
                                    (1.0 - percentage) * outerPath[pathIndex][segmentIndex]->start().y() + percentage * outerPath[pathIndex][segmentIndex]->end().y());
                    }

                    QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(end, closest);

                    segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            path[firstNonTravelSegment]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth));
                    segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           path[firstNonTravelSegment]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight));
                    segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            prestartSpeed);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            path[firstNonTravelSegment]->getSb()->setting< Acceleration >(Constants::SegmentSettings::kAccel));
                    segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    prestartExtruderSpeed);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       path[firstNonTravelSegment]->getSb()->setting< RegionType >(Constants::SegmentSettings::kRegionType));
                    segment->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers,    PathModifiers::kPrestart);

                    path.insert(1, segment);
                    closest = end;
                }
                segmentIndex = (segmentIndex + 1) % outerPath[pathIndex].size();

            }
            path[0]->setEnd(Point(closest.x(), closest.y(), closest.z()));
        }
        else {
            while(prestartDistance > 0)
            {
                if (!dynamic_cast<TravelSegment*>(outerPath[pathIndex][segmentIndex].data()))
                {
                    Distance nextSegmentDist = closest.distance(outerPath[pathIndex][segmentIndex]->start());
                    prestartDistance -= nextSegmentDist;

                    Point end;
                    if(prestartDistance >= 0)
                    {
                        end = outerPath[pathIndex][segmentIndex]->start();
                    }
                    else
                    {
                        float percentage = 1 - (-prestartDistance() / nextSegmentDist());
                        end = Point((1.0 - percentage) * outerPath[pathIndex][segmentIndex]->end().x() + percentage * outerPath[pathIndex][segmentIndex]->start().x(),
                                    (1.0 - percentage) * outerPath[pathIndex][segmentIndex]->end().y() + percentage * outerPath[pathIndex][segmentIndex]->start().y());
                    }

                    QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(end, closest);

                    segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            path[firstNonTravelSegment]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth));
                    segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           path[firstNonTravelSegment]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight));
                    segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            prestartSpeed);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            path[firstNonTravelSegment]->getSb()->setting< Acceleration >(Constants::SegmentSettings::kAccel));
                    segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    prestartExtruderSpeed);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       path[firstNonTravelSegment]->getSb()->setting< RegionType >(Constants::SegmentSettings::kRegionType));
                    segment->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers,    PathModifiers::kPrestart);

                    path.insert(1, segment);
                    closest = end;
                }

                segmentIndex -= 1;
                if(segmentIndex < 0)
                    segmentIndex = outerPath[pathIndex].size() - 1;
            }
            path[0]->setEnd(Point(closest.x(), closest.y(), closest.z()));
        }
    }

    void PathModifierGenerator::GenerateInitialStartup(Path& path, Distance startDistance, Velocity startSpeed, AngularVelocity extruderSpeed, bool enableWidthHeight, double areaMultiplier)
    {
        int currentIndex = 0;
        while(startDistance > 0)
        {
            while (TravelSegment* ts = dynamic_cast<TravelSegment*>(path[currentIndex].data()))
            {
                ++currentIndex;
            }
            RegionType regionType = path[currentIndex]->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType);

            Distance nextSegmentDist = path[currentIndex]->start().distance(path[currentIndex]->end());
            startDistance -= nextSegmentDist;

            if(startDistance >= 0)
            {
                //Update Width and Height if using Width and Height mode
                if (enableWidthHeight)
                {
                    Distance tempWidth = path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth) * qSqrt(areaMultiplier/100);
                    Distance tempHeight = path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight) * qSqrt(areaMultiplier/100);
                    path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kWidth, tempWidth);
                    path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kHeight, tempHeight);
                }
                path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kSpeed, startSpeed);
                path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed, extruderSpeed);
                path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers, PathModifiers::kInitialStartup);

            }
            else
            {
                float percentage = 1 - (-startDistance() / nextSegmentDist());
                Point end = Point((1.0 - percentage) * path[currentIndex]->start().x() + percentage * path[currentIndex]->end().x(),
                            (1.0 - percentage) * path[currentIndex]->start().y() + percentage * path[currentIndex]->end().y());

                Point oldStart = path[currentIndex]->start();
                path[currentIndex]->setStart(end);

                QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(oldStart, end);

                segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth));
                segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight));
                segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            startSpeed);
                segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            path[currentIndex]->getSb()->setting< Acceleration >(Constants::SegmentSettings::kAccel));
                segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    extruderSpeed);
                segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       regionType);
                segment->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers,    PathModifiers::kInitialStartup);
                segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   path[currentIndex]->getSb()->setting< int >(Constants::SegmentSettings::kMaterialNumber));
                segment->getSb()->setSetting(Constants::SegmentSettings::kExtruders,        path[currentIndex]->getSb()->setting<QVector<int>>(Constants::SegmentSettings::kExtruders));

                //Update Width and Height if using Width and Height mode
                if (enableWidthHeight)
                {
                    areaMultiplier = qSqrt(areaMultiplier/100);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth) * areaMultiplier);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight) * areaMultiplier);
                }

                path.insert(currentIndex, segment);

            }

            currentIndex = (currentIndex + 1) % path.size();
        }
    }

    void PathModifierGenerator::GenerateInitialStartupWithRampUp(Path& path, Distance startDistance, Velocity startSpeed, Velocity endSpeed, AngularVelocity startExtruderSpeed, AngularVelocity endExtruderSpeed, int steps, bool enableWidthHeight, double areaMultiplier)
    {
        int currentIndex = 0;
        Distance stepDistance = startDistance / steps;
        AngularVelocity rpmStep = (endExtruderSpeed - startExtruderSpeed) / steps;
        Velocity speedStep = (endSpeed - startSpeed) / steps;

        //Loop through once to do the standard initial startup pathing
        while(startDistance > 0)
        {
            while (TravelSegment* ts = dynamic_cast<TravelSegment*>(path[currentIndex].data()))
            {
                ++currentIndex;
            }
            RegionType regionType = path[currentIndex]->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType);

            Distance nextSegmentDist = path[currentIndex]->start().distance(path[currentIndex]->end());
            startDistance -= nextSegmentDist;

            if(startDistance >= 0)
            {
                //Update Width and Height if using Width and Height mode
                if (enableWidthHeight)
                {
                    Distance tempWidth = path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth) * qSqrt(areaMultiplier/100);
                    Distance tempHeight = path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight) * qSqrt(areaMultiplier/100);
                    path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kWidth, tempWidth);
                    path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kHeight, tempHeight);
                }
                path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kSpeed, startSpeed);
                path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed, startExtruderSpeed);
                path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers, PathModifiers::kInitialStartup);
            }
            else
            {
                float percentage = 1 - (-startDistance() / nextSegmentDist());
                Point end = Point((1.0 - percentage) * path[currentIndex]->start().x() + percentage * path[currentIndex]->end().x(),
                            (1.0 - percentage) * path[currentIndex]->start().y() + percentage * path[currentIndex]->end().y());

                Point oldStart = path[currentIndex]->start();
                path[currentIndex]->setStart(end);

                QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(oldStart, end);

                segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth));
                segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight));
                segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            startSpeed);
                segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            path[currentIndex]->getSb()->setting< Acceleration >(Constants::SegmentSettings::kAccel));
                segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    startExtruderSpeed);
                segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       regionType);
                segment->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers,    PathModifiers::kInitialStartup);
                segment->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers,    PathModifiers::kInitialStartup);
                segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   path[currentIndex]->getSb()->setting< int >(Constants::SegmentSettings::kMaterialNumber));
                segment->getSb()->setSetting(Constants::SegmentSettings::kExtruders,        path[currentIndex]->getSb()->setting<QVector<int>>(Constants::SegmentSettings::kExtruders));

                //Update Width and Height if using Width and Height mode
                if (enableWidthHeight)
                {
                    areaMultiplier = qSqrt(areaMultiplier/100);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth) * areaMultiplier);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight) * areaMultiplier);
                }

                path.insert(currentIndex, segment);

            }

            currentIndex = (currentIndex + 1) % path.size();
        }

        //Loop through the initial startup pathing to break into smaller moves with corrected extruder speed
        currentIndex = 0;
        while (TravelSegment* ts = dynamic_cast<TravelSegment*>(path[currentIndex].data()))
        {
            ++currentIndex;
        }
        Distance currentDistance;
        AngularVelocity currentExtruderSpeed;
        Velocity currentSpeed;
        for (int j = 1; j < steps; j++)
        {
            currentDistance = stepDistance;
            currentExtruderSpeed = startExtruderSpeed + rpmStep * (j - 1);
            currentSpeed = startSpeed + speedStep * (j - 1);
            while(currentDistance > 0)
            {
                RegionType regionType = path[currentIndex]->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType);

                Distance nextSegmentDist = path[currentIndex]->start().distance(path[currentIndex]->end());
                currentDistance -= nextSegmentDist;

                if(currentDistance >= 0)
                {
                    path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kSpeed, currentSpeed);
                    path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed, currentExtruderSpeed);
                    path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers, PathModifiers::kInitialStartup);
                }
                else
                {
                    float percentage = 1 - (-currentDistance() / nextSegmentDist());
                    Point end = Point((1.0 - percentage) * path[currentIndex]->start().x() + percentage * path[currentIndex]->end().x(),
                                (1.0 - percentage) * path[currentIndex]->start().y() + percentage * path[currentIndex]->end().y());

                    Point oldStart = path[currentIndex]->start();
                    path[currentIndex]->setStart(end);

                    QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(oldStart, end);

                    segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth));
                    segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight));
                    segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            currentSpeed);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            path[currentIndex]->getSb()->setting< Acceleration >(Constants::SegmentSettings::kAccel));
                    segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    currentExtruderSpeed);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       regionType);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers,    PathModifiers::kInitialStartup);

                    path.insert(currentIndex, segment);

                }

                currentIndex = (currentIndex + 1) % path.size();
            }
        }
        //After going through all steps, any remaining initial startup segments don't need start/end edited but need updated RPM and speed
        while(path[currentIndex]->getSb()->setting<PathModifiers>(Constants::SegmentSettings::kPathModifiers) == PathModifiers::kInitialStartup)
        {
            path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kSpeed,    startSpeed + speedStep * (steps - 1));
            path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    startExtruderSpeed + rpmStep * (steps - 1));
            currentIndex++;
        }
    }

    void PathModifierGenerator::GenerateSlowdown(Path& path, Distance slowDownDistance, Distance slowDownLiftDistance, Distance slowDownCutoffDistance, Velocity slowDownSpeed, AngularVelocity extruderSpeed, bool enableWidthHeight, double areaMultiplier)
    {
        int currentIndex = path.size() - 1;
        bool isClosed = false;
        Distance tempDistance = slowDownDistance;
        Point newEnd;
        if(path[currentIndex]->end() == path[0]->start())
            isClosed = true;

        PathModifiers current_mod;
        if(extruderSpeed <= 0)
            current_mod = PathModifiers::kCoasting;
        else
            current_mod = PathModifiers::kSlowDown;

        while(tempDistance > 0 && ((currentIndex >= 0 && !isClosed) || isClosed))
        {
            Distance newZIncrement = tempDistance / slowDownDistance * slowDownLiftDistance;
            newEnd = Point(path[currentIndex]->end().x(), path[currentIndex]->end().y(),path[currentIndex]->end().z() + newZIncrement);

            //Update start point of the move following this one, so that start points have correct Z value
            if(tempDistance != slowDownDistance && currentIndex + 1 < path.size())
            {
                path[currentIndex + 1]->setStart(Point(path[currentIndex+1]->start().x(),
                                                 path[currentIndex+1]->start().y(),
                                                 newEnd.z()));
            }

            Distance nextSegmentDist = path[currentIndex]->end().distance(path[currentIndex]->start());
            tempDistance -= nextSegmentDist;            

            if(tempDistance >= 0)
            {                
                //Update Width and Height if using Width and Height mode
                if (enableWidthHeight)
                {
                    Distance tempWidth = path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth) * qSqrt(areaMultiplier/100);
                    Distance tempHeight = path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight) * qSqrt(areaMultiplier/100);
                    path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kWidth, tempWidth);
                    path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kHeight, tempHeight);
                }
                path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kSpeed, slowDownSpeed);
                path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed, extruderSpeed);
                path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers, current_mod);
                path[currentIndex]->setEnd(newEnd);
            }
            else
            {
                float percentage = 1 - (-tempDistance() / nextSegmentDist());
                Point end = Point((1.0 - percentage) * path[currentIndex]->end().x() + percentage * path[currentIndex]->start().x(),
                            (1.0 - percentage) * path[currentIndex]->end().y() + percentage * path[currentIndex]->start().y());

                Point oldEnd = path[currentIndex]->end();
                path[currentIndex]->setEnd(end);

                QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(end, newEnd);

                segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth));
                segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight));
                segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            slowDownSpeed);
                segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            path[currentIndex]->getSb()->setting< Acceleration >(Constants::SegmentSettings::kAccel));
                segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    extruderSpeed);
                segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       path[currentIndex]->getSb()->setting< RegionType >(Constants::SegmentSettings::kRegionType));
                segment->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers,    current_mod);
                segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   path[currentIndex]->getSb()->setting< int >(Constants::SegmentSettings::kMaterialNumber));
                segment->getSb()->setSetting(Constants::SegmentSettings::kExtruders,        path[currentIndex]->getSb()->setting<QVector<int>>(Constants::SegmentSettings::kExtruders));

                //Update Width and Height if using Width and Height mode
                if (enableWidthHeight)
                {
                    areaMultiplier = qSqrt(areaMultiplier/100);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth) * areaMultiplier);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight) * areaMultiplier);
                }

                path.insert(currentIndex + 1, segment);
            }

            currentIndex -= 1;
            if(currentIndex < 0)
                currentIndex = path.size() - 1;
        }

        // Step through loop again if a cutoff distance is needed
        currentIndex = path.size() - 1;
        current_mod = PathModifiers::kCoasting;
        while(slowDownCutoffDistance > 0 && ((currentIndex >= 1 && !isClosed) || isClosed))
        {
            Distance nextSegmentDist = path[currentIndex]->end().distance(path[currentIndex]->start());
            slowDownCutoffDistance -= nextSegmentDist;

            if(slowDownCutoffDistance >= 0)
            {
                path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kSpeed, slowDownSpeed);
                path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed, 0);
                path[currentIndex]->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers, current_mod);
            }
            else
            {
                float percentage = 1 - (-slowDownCutoffDistance() / nextSegmentDist());
                Point end = Point((1.0 - percentage) * path[currentIndex]->end().x() + percentage * path[currentIndex]->start().x(),
                            (1.0 - percentage) * path[currentIndex]->end().y() + percentage * path[currentIndex]->start().y(),
                            (1.0 - percentage) * path[currentIndex]->end().z() + percentage * path[currentIndex]->start().z());

                Point oldEnd = path[currentIndex]->end();
                path[currentIndex]->setEnd(end);

                QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(end, oldEnd);

                segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth));
                segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight));
                segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            slowDownSpeed);
                segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            path[currentIndex]->getSb()->setting< Acceleration >(Constants::SegmentSettings::kAccel));
                segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    0);
                segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       path[currentIndex]->getSb()->setting< RegionType >(Constants::SegmentSettings::kRegionType));
                segment->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers,    current_mod);
                segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   path[currentIndex]->getSb()->setting< int >(Constants::SegmentSettings::kMaterialNumber));

                path.insert(currentIndex + 1, segment);

            }

            currentIndex -= 1;
            if(currentIndex < 0)
                currentIndex = path.size() - 1;
        }
    }

    void PathModifierGenerator::GenerateLayerLeadIn(Path& path, Point leadIn, QSharedPointer<SettingsBase> sb)
    {
        int firstBuildSegmentIndex = 0;
        if (dynamic_cast<TravelSegment*>(path[firstBuildSegmentIndex].data()))
            firstBuildSegmentIndex = 1;

        QSharedPointer<SegmentBase> firstBuildSegment = path[firstBuildSegmentIndex];
        QSharedPointer<LineSegment> leadInSegment = QSharedPointer<LineSegment>::create(leadIn, firstBuildSegment->start());
        leadInSegment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            firstBuildSegment->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth));
        leadInSegment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           firstBuildSegment->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight));
        leadInSegment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            firstBuildSegment->getSb()->setting< Velocity >(Constants::SegmentSettings::kSpeed));
        leadInSegment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            firstBuildSegment->getSb()->setting< Acceleration >(Constants::SegmentSettings::kAccel));
        leadInSegment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    firstBuildSegment->getSb()->setting< AngularVelocity >(Constants::SegmentSettings::kExtruderSpeed));
        leadInSegment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       firstBuildSegment->getSb()->setting< RegionType >(Constants::SegmentSettings::kRegionType));
        leadInSegment->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers,    PathModifiers::kLeadIn);
        path.insert(firstBuildSegmentIndex, leadInSegment);

        path[0]->setEnd(leadIn);
    }

    void PathModifierGenerator::GenerateTrajectorySlowdown(Path& path, QSharedPointer<SettingsBase> sb)
    {
        Angle trajactoryAngleThresh = sb->setting<Angle>(Constants::ExperimentalSettings::Ramping::kTrajectoryAngleThreshold);

        //if the threshold angle set to zero ignores the calculations and returns
        if(trajactoryAngleThresh <= 0) return;

        Distance rampDownLength = sb->setting<Distance>(Constants::ExperimentalSettings::Ramping::kTrajectoryAngleRampDownDistance);
        Distance rampUpLength = sb->setting<Distance>(Constants::ExperimentalSettings::Ramping::kTrajectoryAngleRampUpDistance);
        Velocity speedSlowDown = sb->setting<Distance>(Constants::ExperimentalSettings::Ramping::kTrajectoryAngleSpeedSlowDown)();
        AngularVelocity extruderSpeedSlowDown = sb->setting<Distance>(Constants::ExperimentalSettings::Ramping::kTrajectoryAngleExtruderSpeedSlowDown)();
        Velocity speedUp = sb->setting<Distance>(Constants::ExperimentalSettings::Ramping::kTrajectoryAngleSpeedUp)();
        AngularVelocity extruderSpeedUp = sb->setting<Distance>(Constants::ExperimentalSettings::Ramping::kTrajectoryAngleExtruderSpeedUp)();

        for(int pathIndex = 0, end = path.size() - 1; pathIndex < end; ++pathIndex)
        {
            if(!path[pathIndex]->isPrintingSegment() || !path[pathIndex+1]->isPrintingSegment()) continue;

            Point startPoint = path[pathIndex]->start();
            Point connectingPoint = path[pathIndex]->end();
            Point endPoint = path[pathIndex+1]->end();

            double seg1X = connectingPoint.x() - startPoint.x();
            double seg1Y = connectingPoint.y() - startPoint.y();
            double seg2X = endPoint.x() - connectingPoint.x();
            double seg2Y = endPoint.y() - connectingPoint.y();
            double seg1Length = qSqrt(seg1X * seg1X + seg1Y * seg1Y);
            double seg2Length = qSqrt(seg2X * seg2X + seg2Y * seg2Y);

            double cosTheta = (seg1X * seg2X + seg1Y * seg2Y) / (seg1Length * seg2Length);
            cosTheta = qMin(1.0, cosTheta);
            cosTheta = qMax(-1.0, cosTheta);
            double theta = qAbs(M_PI - qAcos(cosTheta));

            if(theta < trajactoryAngleThresh)
            {
                bool segmentSplitted = false;
                GenerateRamp(path, segmentSplitted, pathIndex, PathModifiers::kRampingDown, rampDownLength, speedSlowDown, extruderSpeedSlowDown);
                if(segmentSplitted){
                    ++pathIndex;
                    ++end;
                }

                segmentSplitted = false;
                GenerateRamp(path, segmentSplitted, pathIndex + 1, PathModifiers::kRampingUp, rampUpLength, speedUp, extruderSpeedUp);
                if(segmentSplitted){
                    ++pathIndex;
                    ++end;
                }
            }
        }
    }

    //to generate tip wipe
    void PathModifierGenerator::GenerateTipWipe(Path &path, PathModifiers modifiers, Distance wipeDistance, Velocity wipeSpeed, Angle wipeAngle, AngularVelocity extruderSpeed, Distance tipWipeLiftDistance, Distance tipWipeCutoffDistance)
    {
        tipWipeDistanceCovered = 0;

        if(static_cast<int>(modifiers & PathModifiers::kForwardTipWipe) != 0)
        {
            int currentIndex = 0;
            Distance cumulativeDistance=0;
            Distance wipeLength = wipeDistance;
            for (QSharedPointer<SegmentBase> segment : path.getSegments())
            {
                if (TravelSegment* ts = dynamic_cast<TravelSegment*>(segment.data()))
                {
                    ++currentIndex;
                }
                else
                {
                   break;
                }
            }
            while(wipeDistance > 0)
            {
                Distance nextSegmentDist = path[currentIndex]->length();

                if(nextSegmentDist == 0)
                    break;

                wipeDistance -= nextSegmentDist;
                cumulativeDistance += nextSegmentDist;
                Distance tempZ;

                Point end;
                if(wipeDistance >= 0)
                {
                    end = Point(path[currentIndex]->end().x(), path[currentIndex]->end().y(), path[currentIndex]->end().z() + (tipWipeLiftDistance * cumulativeDistance/wipeLength));
                }
                else
                {
                    float percentage = 1 - (-wipeDistance() / nextSegmentDist());
                    end = Point((1.0 - percentage) * path[currentIndex]->start().x() + percentage * path[currentIndex]->end().x(),
                                (1.0 - percentage) * path[currentIndex]->start().y() + percentage * path[currentIndex]->end().y(),
                                path[currentIndex]->end().z() + tipWipeLiftDistance);
                }
                generateTipWipeSegment(path, path[currentIndex]->start(), end,
                             path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth),
                             path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight),
                             wipeSpeed,
                             path[currentIndex]->getSb()->setting< Acceleration >(Constants::SegmentSettings::kAccel),
                             extruderSpeed,
                             path[currentIndex]->getSb()->setting< RegionType >(Constants::SegmentSettings::kRegionType),
                             PathModifiers::kForwardTipWipe,
                             path[currentIndex]->getSb()->setting< int >(Constants::SegmentSettings::kMaterialNumber),
                             path[currentIndex]->getSb()->setting<QVector<int>>(Constants::SegmentSettings::kExtruders),
                             tipWipeCutoffDistance);

                currentIndex = (currentIndex + 1) % path.size();
            }
        }
        else if(static_cast<int>(modifiers & PathModifiers::kAngledTipWipe) != 0)
        {
            int currentIndex = path.size() - 1;
            // Find difference in X, and Y, between start and end of last segment in path
            float diff_x = path[currentIndex]->end().x() - path[currentIndex]->start().x();
            float diff_y= path[currentIndex]->end().y() - path[currentIndex]->start().y();
            // Calculate the angle of the last segment of the path
            Angle current_angle = atan2(diff_y, diff_x);
            // Add the wipe angle
            Angle new_angle = current_angle + wipeAngle;
            //Find the new X and Y location to wipe to
            Distance new_x = wipeDistance * cos(new_angle) + path[currentIndex]->end().x();
            Distance new_y = wipeDistance * sin(new_angle) + path[currentIndex]->end().y();
            Point end = Point(new_x, new_y, path[currentIndex]->end().z() + tipWipeLiftDistance);

            generateTipWipeSegment(path, path[currentIndex]->end(), end,
                         path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth),
                         path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight),
                         wipeSpeed,
                         path[currentIndex]->getSb()->setting< Acceleration >(Constants::SegmentSettings::kAccel),
                         extruderSpeed,
                         path[currentIndex]->getSb()->setting< RegionType >(Constants::SegmentSettings::kRegionType),
                         PathModifiers::kAngledTipWipe,
                         path[currentIndex]->getSb()->setting< int >(Constants::SegmentSettings::kMaterialNumber),
                         path[currentIndex]->getSb()->setting<QVector<int>>(Constants::SegmentSettings::kExtruders),
                         tipWipeCutoffDistance);
        }
        else
        {
            int currentIndex = path.size() - 1;
            Distance cumulativeDistance = 0;
            Distance wipeLength = wipeDistance;
            bool isClosed = false;
            if(path[currentIndex]->end() == path[0]->start())
                isClosed = true;

            while(wipeDistance > 0 && ((currentIndex >= 0 && !isClosed) || isClosed))
            {
                Distance nextSegmentDist = path[currentIndex]->end().distance(path[currentIndex]->start());
                wipeDistance -= nextSegmentDist;
                cumulativeDistance += nextSegmentDist;

                Point end;
                if(wipeDistance >= 0)
                {
                    end = Point(path[currentIndex]->start().x(), path[currentIndex]->start().y(), path[currentIndex]->start().z() + (tipWipeLiftDistance * cumulativeDistance/wipeLength));
                }
                else
                {
                    float percentage = 1 - (-wipeDistance() / nextSegmentDist());
                    end = Point((1.0 - percentage) * path[currentIndex]->end().x() + percentage * path[currentIndex]->start().x(),
                                (1.0 - percentage) * path[currentIndex]->end().y() + percentage * path[currentIndex]->start().y(),
                                path[currentIndex]->start().z() + tipWipeLiftDistance);
                }

                generateTipWipeSegment(path, path[currentIndex]->end(), end,
                             path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth),
                             path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight),
                             wipeSpeed,
                             path[currentIndex]->getSb()->setting< Acceleration >(Constants::SegmentSettings::kAccel),
                             extruderSpeed,
                             path[currentIndex]->getSb()->setting< RegionType >(Constants::SegmentSettings::kRegionType),
                             PathModifiers::kReverseTipWipe,
                             path[currentIndex]->getSb()->setting< int >(Constants::SegmentSettings::kMaterialNumber),
                             path[currentIndex]->getSb()->setting<QVector<int>>(Constants::SegmentSettings::kExtruders),
                             tipWipeCutoffDistance);

                currentIndex -= 1;
                if(currentIndex < 0)
                    currentIndex = path.size() - 1;
            }
        }
    }

    void PathModifierGenerator::GenerateTipWipe(Path &path, PathModifiers modifiers, Distance wipeDistance, Velocity wipeSpeed, QVector<Path>& outerPath, Angle wipeAngle, AngularVelocity extruderSpeed, Distance tipWipeLiftDistance, Distance tipWipeCutoffDistance)
    {
        tipWipeDistanceCovered = 0;

        //can only go forward, connect to inset
        Point closest;
        float dist = INT_MAX;
        int pathIndex, segmentIndex;
        Point finalPoint = path[path.size() - 1]->end();
        int finalIndex = path.size() - 1;
        for(int i = 0, totalContours = outerPath.size(); i < totalContours; ++i)
        {
            for(int j = 0, totalSegments = outerPath[i].size(); j < totalSegments; ++j)
            {
                Point tempClosest;
                float tempDist;
                std::tie(tempDist, tempClosest) = MathUtils::findClosestPointOnSegment(outerPath[i][j]->start(), outerPath[i][j]->end(), finalPoint);
                if(tempDist < dist)
                {
                    closest = tempClosest;
                    dist = tempDist;
                    segmentIndex = j;
                    pathIndex = i;
                }
            }
        }

        if(modifiers == PathModifiers::kForwardTipWipe)
        {
            if (dist > path[finalIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth))
            {
                GenerateForwardTipWipeOpenLoop(path, modifiers, wipeDistance, wipeSpeed, extruderSpeed, tipWipeLiftDistance, tipWipeCutoffDistance, false);
            }
            else
            {
                //move to segment
                Distance new_Z = closest.z() + (tipWipeLiftDistance * finalPoint.distance(closest)/wipeDistance);
                Point new_end = Point(closest.x(), closest.y(), new_Z);
                Distance cumulative_distance = 0;
                Distance wipeLength = wipeDistance;
                generateTipWipeSegment(path, finalPoint, new_end,
                             path[finalIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth),
                             path[finalIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight),
                             wipeSpeed,
                             path[finalIndex]->getSb()->setting< Acceleration >(Constants::SegmentSettings::kAccel),
                             extruderSpeed,
                             path[finalIndex]->getSb()->setting< RegionType >(Constants::SegmentSettings::kRegionType),
                             PathModifiers::kForwardTipWipe,
                             path[finalIndex]->getSb()->setting< int >(Constants::SegmentSettings::kMaterialNumber),
                             path[finalIndex]->getSb()->setting<QVector<int>>(Constants::SegmentSettings::kExtruders),
                             tipWipeCutoffDistance);

                Distance nextSegmentDistEnd = outerPath[pathIndex][segmentIndex]->end().distance(closest);
                Distance nextSegmentDistStart = outerPath[pathIndex][segmentIndex]->start().distance(closest);

                wipeDistance -= finalPoint.distance(closest);
                cumulative_distance += finalPoint.distance(closest);

                if(nextSegmentDistStart > nextSegmentDistEnd)
                {
                    //int currentIndex = index;
                    while(wipeDistance > 0)
                    {
                        Distance nextSegmentDist = closest.distance(outerPath[pathIndex][segmentIndex]->end());
                        wipeDistance -= nextSegmentDist;
                        cumulative_distance += nextSegmentDist;

                        Point end;
                        if(wipeDistance >= 0)
                        {
                            end = Point(outerPath[pathIndex][segmentIndex]->end().x(), outerPath[pathIndex][segmentIndex]->end().y(), outerPath[pathIndex][segmentIndex]->end().z() + (tipWipeLiftDistance * cumulative_distance/wipeLength));
                        }
                        else
                        {
                            float percentage = 1 - (-wipeDistance() / nextSegmentDist());
                            end = Point((1.0 - percentage) * closest.x() + percentage * outerPath[pathIndex][segmentIndex]->end().x(),
                                        (1.0 - percentage) * closest.y() + percentage * outerPath[pathIndex][segmentIndex]->end().y(),
                                        outerPath[pathIndex][segmentIndex]->end().z() + tipWipeLiftDistance);
                        }

                        generateTipWipeSegment(path, closest, end,
                                     path[finalIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth),
                                     path[finalIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight),
                                     wipeSpeed,
                                     path[finalIndex]->getSb()->setting< Acceleration >(Constants::SegmentSettings::kAccel),
                                     extruderSpeed,
                                     path[finalIndex]->getSb()->setting< RegionType >(Constants::SegmentSettings::kRegionType),
                                     PathModifiers::kForwardTipWipe,
                                     path[finalIndex]->getSb()->setting<int>(Constants::SegmentSettings::kMaterialNumber),
                                     path[finalIndex]->getSb()->setting<QVector<int>>(Constants::SegmentSettings::kExtruders),
                                     tipWipeCutoffDistance);

                        segmentIndex = (segmentIndex + 1) % outerPath[pathIndex].size();
                        closest = end;
                    }
                }
                else {
                    while(wipeDistance > 0)
                    {
                        Distance nextSegmentDist = closest.distance(outerPath[pathIndex][segmentIndex]->start());
                        wipeDistance -= nextSegmentDist;
                        cumulative_distance += nextSegmentDist;

                        Point end;
                        if(wipeDistance >= 0)
                        {
                            end = Point(outerPath[pathIndex][segmentIndex]->end().x(), outerPath[pathIndex][segmentIndex]->end().y(), outerPath[pathIndex][segmentIndex]->end().z() + (tipWipeLiftDistance * cumulative_distance/wipeLength));
                        }
                        else
                        {
                            float percentage = 1 - (-wipeDistance() / nextSegmentDist());
                            end = Point((1.0 - percentage) * closest.x() + percentage * outerPath[pathIndex][segmentIndex]->start().x(),
                                        (1.0 - percentage) * closest.y() + percentage * outerPath[pathIndex][segmentIndex]->start().y());
                        }

                        generateTipWipeSegment(path, closest, end,
                                     path[finalIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth),
                                     path[finalIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight),
                                     wipeSpeed,
                                     path[finalIndex]->getSb()->setting< Acceleration >(Constants::SegmentSettings::kAccel),
                                     extruderSpeed,
                                     path[finalIndex]->getSb()->setting< RegionType >(Constants::SegmentSettings::kRegionType),
                                     PathModifiers::kForwardTipWipe,
                                     path[finalIndex]->getSb()->setting<int>(Constants::SegmentSettings::kMaterialNumber),
                                     path[finalIndex]->getSb()->setting<QVector<int>>(Constants::SegmentSettings::kExtruders),
                                     tipWipeCutoffDistance);

                        segmentIndex -= 1;
                        if(segmentIndex < 0)
                            segmentIndex = outerPath[pathIndex].size() - 1;

                        closest = end;
                    }
                }
            }
        }
        else
        {
            GenerateTipWipe(path, modifiers, wipeDistance, wipeSpeed, wipeAngle, extruderSpeed, tipWipeLiftDistance, tipWipeCutoffDistance);
        }
    }

    void PathModifierGenerator::GenerateForwardTipWipeOpenLoop(Path &path, PathModifiers modifiers, Distance wipeDistance, Velocity wipeSpeed, AngularVelocity extruderSpeed, Distance tipWipeLiftDistance, Distance tipWipeCutoffDistance, bool clearTipWipDistanceCovered)
    {
        if(clearTipWipDistanceCovered)
            tipWipeDistanceCovered = 0;

        int currentIndex = path.size() - 1;
        Distance length = path[currentIndex]->end().distance(path[currentIndex]->start());
        Distance X = path[currentIndex]->end().x() + (path[currentIndex]->end().x() - path[currentIndex]->start().x()) / length() * wipeDistance();
        Distance Y = path[currentIndex]->end().y() + (path[currentIndex]->end().y() - path[currentIndex]->start().y()) / length() * wipeDistance();
        Distance Z = path[currentIndex]->end().z() + tipWipeLiftDistance;
        Point end(X, Y, Z);

        generateTipWipeSegment(path, path[currentIndex]->end(), end,
             path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kWidth),
             path[currentIndex]->getSb()->setting< Distance >(Constants::SegmentSettings::kHeight),
             wipeSpeed,
             path[currentIndex]->getSb()->setting< Acceleration >(Constants::SegmentSettings::kAccel),
             extruderSpeed,
             path[currentIndex]->getSb()->setting< RegionType >(Constants::SegmentSettings::kRegionType),
             PathModifiers::kForwardTipWipe,
             path[currentIndex]->getSb()->setting< int >(Constants::SegmentSettings::kMaterialNumber),
             path[currentIndex]->getSb()->setting<QVector<int>>(Constants::SegmentSettings::kExtruders),
             tipWipeCutoffDistance);
    }

    void PathModifierGenerator::GenerateSpiralLift(Path& path, Distance spiralWidth, Distance spiralHeight, int spiralPoints, Velocity spiralLiftVelocity, bool supportsG3)
    {
        Point startPoint = path.back()->end();

        if(supportsG3)
        {
            Point spiral_start_point(startPoint.x() + spiralWidth, startPoint.y(), startPoint.z());
            Point spiral_end_point(startPoint.x() + spiralWidth + spiralHeight * qCos(355.0 * M_PI / 180), startPoint.y() + spiralWidth + spiralHeight * qSin(355.0 * M_PI / 180), startPoint.z() + spiralHeight);
            Point center_point(startPoint.x(), startPoint.y());

            writeSegment(path, startPoint, spiral_start_point,
                         path.back()->getSb()->setting<Distance>(Constants::SegmentSettings::kWidth),
                         spiralHeight,
                         spiralLiftVelocity,
                         path.back()->getSb()->setting< Acceleration >(Constants::SegmentSettings::kAccel),
                         .0f,
                         path.back()->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType),
                         PathModifiers::kSpiralLift,
                         path.back()->getSb()->setting<int>(Constants::SegmentSettings::kMaterialNumber),
                         path.back()->getSb()->setting<QVector<int>>(Constants::SegmentSettings::kExtruders));

            writeArcSegment(path, spiral_start_point, spiral_end_point, center_point, 355, false,
                            path.back()->getSb()->setting<Distance>(Constants::SegmentSettings::kWidth),
                            path.back()->getSb()->setting<Distance>(Constants::SegmentSettings::kHeight),
                            spiralLiftVelocity,
                            path.back()->getSb()->setting<Acceleration>(Constants::SegmentSettings::kAccel),
                            .0f,
                            path.back()->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType),
                            PathModifiers::kSpiralLift,
                            path.back()->getSb()->setting<int>(Constants::SegmentSettings::kMaterialNumber),
                            path.back()->getSb()->setting<QVector<int>>(Constants::SegmentSettings::kExtruders));
        }
        else
        {
            float currentZ = startPoint.z();
            Point newStart = startPoint;
            for(int i = 0; i < spiralPoints; ++i)
            {
                Point newEnd(startPoint.x() - (float(i) / float(spiralPoints)*qCos(i*M_PI / 8.0)*spiralWidth),
                             startPoint.y() - (float(i) / float(spiralPoints)*qSin(i*M_PI / 8.0)*spiralWidth),
                             currentZ);

                currentZ += spiralHeight() / float(spiralPoints);

                writeSegment(path, newStart, newEnd,
                             path.back()->getSb()->setting<Distance>(Constants::SegmentSettings::kWidth),
                             spiralHeight,
                             spiralLiftVelocity,
                             path.back()->getSb()->setting< Acceleration >(Constants::SegmentSettings::kAccel),
                             .0f,
                             path.back()->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType),
                             PathModifiers::kSpiralLift,
                             path.back()->getSb()->setting<int>(Constants::SegmentSettings::kMaterialNumber),
                             path.back()->getSb()->setting<QVector<int>>(Constants::SegmentSettings::kExtruders));

                newStart = newEnd;
            }
        }
    }

    void PathModifierGenerator::writeSegment(Path &path, Point start, Point end, Distance width, Distance height, Velocity speed, Acceleration acceleration,
                                             AngularVelocity extruder_speed, RegionType regionType, PathModifiers pathModifiers, int materialNumber,
                                             QVector<int> extruders)
    {
        QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(start, end);

        segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            width);
        segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           height);
        segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            speed);
        segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            acceleration);
        segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    extruder_speed);
        segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       regionType);
        segment->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers,    pathModifiers);
        segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   materialNumber);
        segment->getSb()->setSetting(Constants::SegmentSettings::kExtruders,        extruders);

        path.append(segment);
    }

    void PathModifierGenerator::writeArcSegment(Path &path, Point start, Point end, Point center, Angle angle, bool ccw, Distance width, Distance height,
                                                Velocity speed, Acceleration acceleration, AngularVelocity extruder_speed, RegionType regionType,
                                                PathModifiers path_modifiers, int materialNumber, QVector<int> extruders)
    {
        QSharedPointer<ArcSegment> segment = QSharedPointer<ArcSegment>::create(start, end, center, angle, ccw);

        segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            width);
        segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           height);
        segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            speed);
        segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            acceleration);
        segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    extruder_speed);
        segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       regionType);
        segment->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers,    path_modifiers);
        segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   materialNumber);
        segment->getSb()->setSetting(Constants::SegmentSettings::kExtruders,        extruders);

        path.append(segment);
    }

    void PathModifierGenerator::GenerateRamp(Path& path, bool& segmentSplitted, int segmentIndex, PathModifiers pathModifiers,
                                             Distance rampLength, Velocity speed, AngularVelocity extruderSpeed)
    {
        Distance rampLengthCovered = 0;
        int endIndex = path.size() - 1;
        bool rampDown = pathModifiers == PathModifiers::kRampingDown;

        while(rampLengthCovered < rampLength)
        {
            if(rampDown){
                if(segmentIndex < 0) break;
            }else {
                if(segmentIndex > endIndex) break;
            }

            QSharedPointer<SegmentBase> segment = path[segmentIndex];

            PathModifiers segPM = segment->getSb()->setting<PathModifiers>(Constants::SegmentSettings::kPathModifiers);
            if(segPM == PathModifiers::kRampingUp || segPM == PathModifiers::kRampingDown) break;

            if(segment->length() > (rampLength - rampLengthCovered)){
                double newPDist = ((rampLength - rampLengthCovered) / segment->length())();
                if(!rampDown) newPDist = 1 - newPDist;

                Point startP = segment->start();
                Point endP = segment->end();
                Point newPV = Point((startP.x() - endP.x()) * newPDist,
                                    (startP.y() - endP.y()) * newPDist,
                                    (startP.z() - endP.z()) * newPDist);
                Point newP = Point(endP.x() + newPV.x(),
                                   endP.y() + newPV.y(),
                                   endP.z() + newPV.z());

                segment->setEnd(newP);

                QSharedPointer<LineSegment> newSegment = QSharedPointer<LineSegment>::create(newP, endP);
                newSegment->getSb()->setSetting(Constants::SegmentSettings::kWidth,
                                             segment->getSb()->setting<Distance>(Constants::SegmentSettings::kWidth));
                newSegment->getSb()->setSetting(Constants::SegmentSettings::kHeight,
                                             segment->getSb()->setting<Distance>(Constants::SegmentSettings::kHeight));
                newSegment->getSb()->setSetting(Constants::SegmentSettings::kAccel,
                                             segment->getSb()->setting<Acceleration>(Constants::SegmentSettings::kAccel));
                newSegment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,
                                             segment->getSb()->setting<int>(Constants::SegmentSettings::kMaterialNumber));

                RegionType regionType = segment->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType);
                if(regionType == RegionType::kUnknown)
                    regionType = path[segmentIndex+1]->getSb()->setting<RegionType>(Constants::SegmentSettings::kRegionType);
                newSegment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       regionType);

                if(rampDown){
                    newSegment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            speed);
                    newSegment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    extruderSpeed);
                    newSegment->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers,    pathModifiers);
                }else{
                    newSegment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,
                                                    segment->getSb()->setting<Velocity>(Constants::SegmentSettings::kSpeed));
                    newSegment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,
                                                    segment->getSb()->setting<AngularVelocity>(Constants::SegmentSettings::kExtruderSpeed));

                    segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            speed);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    extruderSpeed);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers,    pathModifiers);
                }

                path.insert(segmentIndex + 1, newSegment);

                segmentSplitted = true;
                break;
            }
            else
            {
                segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,         speed);
                segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed, extruderSpeed);
                segment->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers, pathModifiers);
            }

            rampLengthCovered += segment->length();
            segmentIndex += rampDown ? -1 : 1;
        };
    }

    void PathModifierGenerator::generateTipWipeSegment(Path &path, Point start, Point end, Distance width, Distance height, Velocity speed, Acceleration acceleration,
                                             AngularVelocity extruder_speed, RegionType regionType, PathModifiers pathModifiers, int materialNumber,
                                             QVector<int> extruders, Distance tipWipeCutoffDistance){
        if(tipWipeCutoffDistance > 0){
            Distance length = end.distance(start);

            if(tipWipeDistanceCovered >= tipWipeCutoffDistance){
                extruder_speed = 0;
            }
            else if(tipWipeDistanceCovered() + length() - tipWipeCutoffDistance() > 0.09){
                auto ratio = (tipWipeCutoffDistance() - tipWipeDistanceCovered()) / length();
                Point hopPoint(start.x() + ((end.x() - start.x()) * ratio),
                               start.y() + ((end.y() - start.y()) * ratio), end.z());

                writeSegment(path, start, hopPoint, width, height, speed, acceleration, extruder_speed,
                             regionType, pathModifiers, materialNumber, extruders);

                start = hopPoint;
                extruder_speed = 0;
            }

            tipWipeDistanceCovered += length;
        }

        writeSegment(path, start, end, width, height, speed, acceleration, extruder_speed,
                     regionType, pathModifiers, materialNumber, extruders);
    }

    Distance PathModifierGenerator::tipWipeDistanceCovered = 0;
}
