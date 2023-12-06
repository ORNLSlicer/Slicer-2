//Header
#include "optimizers/point_order_optimizer.h"

//Qt
#include <QRandomGenerator>

//Local
#include <algorithms/knn.h>

namespace ORNL
{
    int PointOrderOptimizer::linkToPoint(Point current_location, Path path, uint layer_number, const QSharedPointer<SettingsBase>& sb)
    {
        int result;
        PointOrderOptimization pointOrderOptimization = static_cast<PointOrderOptimization>(sb->setting<int>(Constants::ProfileSettings::Optimizations::kPointOrder));

        switch(pointOrderOptimization)
        {
            case PointOrderOptimization::kNextClosest:
                result = findShortestOrLongestDistance(path, current_location,
                                                       sb->setting<bool>(Constants::ProfileSettings::Optimizations::kMinDistanceEnabled),
                                                       sb->setting<Distance>(Constants::ProfileSettings::Optimizations::kMinDistanceThreshold));
            break;
            case PointOrderOptimization::kNextFarthest:
                result = findShortestOrLongestDistance(path, current_location,
                                                       sb->setting<bool>(Constants::ProfileSettings::Optimizations::kMinDistanceEnabled),
                                                       sb->setting<Distance>(Constants::ProfileSettings::Optimizations::kMinDistanceThreshold),
                                                       false);
            break;
            case PointOrderOptimization::kRandom:
                result = linkToRandom(path);
            break;
            case PointOrderOptimization::kConsecutive:
                result = linkToConsecutive(path, layer_number, sb->setting<Distance>(Constants::ProfileSettings::Optimizations::kConsecutiveDistanceThreshold));
            break;
            default:
                result = findShortestOrLongestDistance(path, current_location, false, Distance(0));
            break;
        }

        if(sb->setting<bool>(Constants::ProfileSettings::Optimizations::kLocalRandomnessEnable))
            result = computePerturbation(path, path[result]->start(), sb->setting<Distance>(Constants::ProfileSettings::Optimizations::kLocalRandomnessRadius));

        return result;
    }

    int PointOrderOptimizer::findShortestOrLongestDistance(Path path, Point startPoint, bool minThresholdEnable, Distance minThreshold, bool shortest)
    {
        int pointIndex = -1;
        Distance closest;
        if(shortest)
            closest = Distance(DBL_MAX);
        if(minThresholdEnable)
            closest = Distance(minThreshold);

        for(int i = 0, end = path.size(); i < end; ++i)
        {
            Distance dist = path[i]->start().distance(startPoint);
            if(shortest)
            {
                if (dist < closest)
                {
                    closest = dist;
                    pointIndex = i;
                }
            }
            else
            {
                if (dist > closest)
                {
                    closest = dist;
                    pointIndex = i;
                }
            }
        }

        //if no candidates found it's because nothing met threshold, so find farthest point
        if(pointIndex == -1)
            pointIndex = findShortestOrLongestDistance(path, startPoint, false, Distance(0), false);

        return pointIndex;
    }

    int PointOrderOptimizer::linkToRandom(Path path)
    {
        return QRandomGenerator::global()->bounded(path.size());
    }

    int PointOrderOptimizer::linkToConsecutive(Path path, uint layer_number, Distance minDist)
    {
        int startIndex = layer_number - 2;
        if(startIndex < 0)
            startIndex += path.size();
        else
            startIndex %= path.size();

        int previousIndex = startIndex;

        Distance dist;
        do
        {
            startIndex = (startIndex + 1) % path.size();
            QSharedPointer<SegmentBase> previous = path.at(previousIndex);
            QSharedPointer<SegmentBase> current = path.at(startIndex);
            dist += previous->start().distance(current->start());

            //looped through whole path
            if(startIndex == previousIndex)
            {
                break;
            }

        } while(dist < minDist);

        return startIndex;
    }

    int PointOrderOptimizer::computePerturbation(Path path, Point current_start, Distance radius)
    {
        QList<QSharedPointer<SegmentBase>> segments = path.getSegments();
        QVector<int> candidates;

        for(int i = 0; i < segments.size(); ++i)
        {
            if(segments[i]->start().distance(current_start) < radius)
                candidates.push_back(i);
        }

        return candidates[QRandomGenerator::global()->bounded(candidates.size())];
    }
}
