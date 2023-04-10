#include "geometry/curve_fitting.h"

// Locals
#include "utilities/mathutils.h"
#include "managers/gpu_manager.h"
#include "geometry/segments/line.h"

#ifdef NVCC_FOUND
#include "geometry/gpu/gpu_curve_fitting.h"
#endif

namespace ORNL
{
    void CurveFitting::Fit(Path &path, QSharedPointer<SettingsBase> layer_settings)
    {
        bool use_arcs = layer_settings->setting<bool>(Constants::ExperimentalSettings::CurveFitting::kEnableArcFitting);
        bool use_splines = layer_settings->setting<bool>(Constants::ExperimentalSettings::CurveFitting::kEnableSplineFitting);

        cleanPath(path, 50); // Remove segments less than 50 microns and stitch

        #ifdef ARC_DEBUG_INFO
        writeDebugData(path);
        #endif

        Path new_path;
        auto angle_partitions = PartitionByAngle(path, layer_settings); // Determine sections based on angle breaks

        QVector<QPair<int, Path>> curve_partitions;
        for(auto& p : angle_partitions)
        {
            if(p.size() >= 2)
                curve_partitions.append(PartitionByCurvature(p, use_arcs, use_splines)); // Determine type of lines (regular, arc, or spline)
            else
                curve_partitions.append(qMakePair(0, p));
        }

        #ifdef ARC_DEBUG_INFO
        writeDebugGroups(curve_partitions);
        #endif

        // Fit according to type
        for(auto partition : curve_partitions)
        {
            switch(partition.first)
            {
            case(0): // Line
                new_path.append(partition.second.front());
                break;
            case(1): // Arc
                for(const auto& seg : DetermineBestArcFit(partition.second, layer_settings))
                    new_path.append(seg);
                break;
            case(2): // Spline
                QVector<BezierSegment> spline;
                SplineTraverse(spline, partition.second, 0, partition.second.size() - 1);

                if(spline.size() > 0)
                {
                    for(auto seg: spline)
                    {
                        seg.setSb(partition.second.front()->getSb());
                        new_path.append(QSharedPointer<BezierSegment>::create(seg));
                    }
                }else // Not enough segments to fit, so insert line segment
                {
                    new_path.append(partition.second.front());
                }

                break;
            }
        }

        path = new_path;
    }

    #ifdef ARC_DEBUG_INFO
    void CurveFitting::writeDebugData(Path &path)
    {
        QFile file("arc_data.csv");
        file.open(QIODevice::WriteOnly);
        QTextStream stream(&file);

        Distance running_length = 0.0;
        for(int i = 0; i < path.size(); ++i)
        {
            auto next_segment = path[(i + 1) % path.size()];
            auto this_segment = path[i];
            double curvature = ArcSegment::SignedCurvature(this_segment->start(), this_segment->end(), next_segment->end());
            running_length += this_segment->length();
            stream << QString::number(curvature) << "," <<
                      QString::number(MathUtils::internalAngle(this_segment->start(), this_segment->end(), next_segment->end()).to(degrees)) << "," <<
                      QString::number(MathUtils::signedInternalAngle(this_segment->start(), this_segment->end(), next_segment->end()).to(degrees)) << "," <<
                      QString::number(running_length()) << "," <<
                      this_segment->start().toCSVString() << "," <<
                      this_segment->end().toCSVString() << "," <<
                      next_segment->end().toCSVString() << Qt::endl;
        }

        file.close();
    }
    #endif

    #ifdef ARC_DEBUG_INFO
    void CurveFitting::writeDebugGroups(QVector<QPair<int, Path> > groups)
    {
        QFile group_file("arc_groups.csv");
        group_file.open(QIODevice::WriteOnly);
        QTextStream group_stream(&group_file);
        int group_number = 0;
        for(auto group : groups)
        {
            for(auto segment : group.second)
            {
                group_stream << group_number << "," << segment->start().toCSVString() << Qt::endl;
            }
            group_number++;
        }
        group_file.close();
    }
    #endif

    void CurveFitting::cleanPath(Path &path, Distance threshold)
    {
        // Clean up path by removing any sements of very small length
        Path cleaned_path;
        for(int i = 0, end = path.size(); i < end; ++i)
        {
            auto segment = path[i];

            if(segment->length() < threshold) // Prune small segments
            {
                if(i != 0 && i != (end - 1))
                    path[i - 1]->end() = path[i + 1]->start(); // Stitch last and next semgments
            }
            else
                cleaned_path.append(segment);
        }
        path = cleaned_path;
    }

    QVector<Path> CurveFitting::PartitionByAngle(Path &path, QSharedPointer<SettingsBase> layer_settings)
    {
        QVector<Path> paths;
        Path new_path;

        int end = path.size();
        new_path.append(path.front());

        Distance max_dist = layer_settings->setting<Distance>(Constants::ExperimentalSettings::CurveFitting::kMaxCurveSegmentDistance);
        Angle min_angle = layer_settings->setting<Angle>(Constants::ExperimentalSettings::CurveFitting::kMinCurveAngle);

        for(int seg_index = 1; seg_index < end; ++seg_index)
        {
            auto last_segment = path[seg_index - 1];
            auto segment = path[seg_index];

            Angle angle = MathUtils::internalAngle(last_segment->start(), segment->start(), segment->end());

            if(segment->length() > max_dist || // If the segment is too long
                    last_segment->length() > max_dist ||
                    angle < layer_settings->setting<Angle>(Constants::ExperimentalSettings::CurveFitting::kMinCurveAngle) || // If the angle is too sharp
                    last_segment->getSb()->json() != segment->getSb()->json()) // If it changes settings regions

            {
                // These points cannot support a curve, so it must be a split
                paths.append(new_path);
                new_path.clear(); // Reset our new path
            }

            new_path.append(segment);
        }

        paths.append(new_path);

        // If the first and last partitions link together, and it's not a circle then prepend the last to the first
        if(paths.size() >= 2 && MathUtils::internalAngle(paths.back().back()->start(), paths.first().front()->start(), paths.first().front()->end()) >= min_angle)
        {
            auto last_path = paths.last();
            paths.pop_back();

            std::reverse(last_path.begin(), last_path.end());

            for(auto seg : last_path)
                paths.first().prepend(seg);
        }

        return paths;
    }

    double Variance(QVector<double> samples)
    {
         int size = samples.size();

         double variance = 0;
         double t = samples[0];
         for (int i = 1; i < size; i++)
         {
              t += samples[i];
              double diff = ((i + 1) * samples[i]) - t;
              variance += (diff * diff) / ((i + 1.0) *i);
         }

         return variance / (size - 1);
    }

    double StandardDeviation(QVector<double> samples)
    {
         return qSqrt(Variance(samples));
    }

    double Mean(QVector<double> samples)
    {
        double total = 0.0;
        for(auto value : samples)
        {
            total += value;
        }
        return total / double(samples.size());
    }

    QVector<QPair<int, Path>> CurveFitting::PartitionByCurvature(Path &path, bool use_arcs, bool use_splines)
    {
        bool closed = path.front()->start() == path.back()->end();
        QVector<QPair<int, Path>> sections;

        // ST.DEV
        QVector<double> derivative_of_curvatres;
        double last_curve = ArcSegment::SignedCurvature(path.front(), *(path.begin() + 1));
        for(int i = 0; i < path.size(); ++i)
        {
            auto next_segment = path[(i + 1) % path.size()];
            auto this_segment = path[i];
            double curvature = ArcSegment::SignedCurvature(this_segment->start(), this_segment->end(), next_segment->end());
            derivative_of_curvatres.push_back(curvature - last_curve);
            last_curve = curvature;
        }
        double factor = 2;
        double curvature_stddev = StandardDeviation(derivative_of_curvatres);
        double curvature_mean = Mean(derivative_of_curvatres);
        double curve_max = curvature_mean + (curvature_stddev * factor);
        double curve_min = curvature_mean - (curvature_stddev * factor);

        Path curr_path;

        double last_curvature = ArcSegment::SignedCurvature(path.front(), *(path.begin() + 1));

        for(int i = 0, end = path.size() - 1; i < end; ++i)
        {
            auto this_segment = path[i];
            auto next_segment = path[i + 1];
            double curvature = ArcSegment::SignedCurvature(this_segment, next_segment);

            double derivative_of_curvatre = curvature - last_curvature;

            if((derivative_of_curvatre <= curve_max) && (derivative_of_curvatre >= curve_min) && use_arcs)
            {
                curr_path.append(this_segment);
            }else // This is a break point
            {
                // Add accumulated segments to total
                if(curr_path.size() == 1) // This is just a single segment, therefore a line
                    sections.append(qMakePair(0, curr_path));
                else
                    sections.append(qMakePair(1, curr_path));
                // TODO: fix splines

                curr_path.clear();
                curr_path.append(this_segment); // Start the next section by adding this segment
            }

            last_curvature = curvature;
        }

        // The last segment needs added
        curr_path.append(path.back());
        if(curr_path.size() == 1) // This is just a single segment, therefore a line
            sections.append(qMakePair(0, curr_path));
        else
            sections.append(qMakePair(1, curr_path));

        if(closed && !sections.isEmpty()) // This is a closed loop and this is not a circle
        {

        }

        return sections;
    }

    std::pair<QSharedPointer<ArcSegment>, double> CurveFitting::FitArcToSection(Path path, int start_index, int end_index)
    {
        QSharedPointer<ArcSegment> best_fit = nullptr;
        double min_error = std::numeric_limits<double>::max();
        Point start_point = path[start_index]->start();
        Point end_point = path[end_index]->end();

        bool is_circle = false;

        if(start_point == end_point) // If the start and the end are the same, then this arc will be a circle, use the last point to approximate
        {
            end_index--;
            end_point = path[end_index]->end();
            is_circle = true;
        }

        #ifdef NVCC_FOUND
        if(GPU->use())
        {
            auto result = CUDA::GPUCurveFitting::FindBestArcFit(path, start_index, end_index);
            best_fit = QSharedPointer<ArcSegment>::create(result.first);
            min_error = result.second;
        }else
        {
        #endif
            // Find center point that will result in the best fit
            for(int i = start_index; i < end_index; ++i) // Skip the first and last vertices, they are always used
            {
                ArcSegment segment(start_point, path[i]->end(), end_point);

                auto arc_error = ArcError(start_index, end_index, path, segment);

                if(arc_error < min_error)
                {
                    min_error = arc_error;
                    best_fit = QSharedPointer<ArcSegment>::create(segment);
                }
            }
        #ifdef NVCC_FOUND
        }
        #endif

        if(best_fit != nullptr)
        {
            if(is_circle)
            {
                best_fit->setEnd(path[end_index + 1]->end());
                best_fit->setAngle(2.0 * M_PI);
            }

            best_fit->setSb(path[start_index]->getSb());
        }

        return std::pair<QSharedPointer<ArcSegment>, double>(best_fit, min_error);
    }

    double CurveFitting::ArcError(int start, int end, Path& path, ArcSegment& fit_segment)
    {
        Distance error;

        Point center = fit_segment.center();
        Distance radius = center.distance(fit_segment.start());
        double arc_end_angle = MathUtils::angleFromXAxis(fit_segment.center(), fit_segment.end());
        double arc_start_angle = MathUtils::angleFromXAxis(fit_segment.center(), fit_segment.start());

        for(int i = start; i < end; ++i)
        {
            auto segment = path[i];
            double segment_angle = MathUtils::angleFromXAxis(fit_segment.center(), segment->end());

            if(fit_segment.counterclockwise() && arc_start_angle > arc_end_angle)
            {
                if((segment_angle < arc_end_angle) || (segment_angle > arc_start_angle))
                    error += qFabs(radius() - center.distance(segment->end())());
                else
                    error += qFabs(radius() + center.distance(segment->end())());
            }else if(!fit_segment.counterclockwise() && arc_start_angle < arc_end_angle)
            {
                if((segment_angle < arc_start_angle) || (segment_angle > arc_end_angle))
                    error += qFabs(radius() - center.distance(segment->end())());
                else
                    error += qFabs(radius() + center.distance(segment->end())());
            }
            else
            {
                if((segment_angle > arc_end_angle && segment_angle < arc_start_angle) || (segment_angle < arc_end_angle && segment_angle > arc_start_angle))
                    error += qFabs(radius() - center.distance(segment->end())());
                else
                    error += qFabs(radius() + center.distance(segment->end())());
            }
        }

        return error();
    }

    Path CurveFitting::DetermineBestArcFit(Path& path, QSharedPointer<SettingsBase> layer_settings)
    {
        Path best_fit_path;
        double min_error = std::numeric_limits<double>::max();

        best_fit_path = FitPathToArc(path, path.size()).first;


        if(best_fit_path.size() == 0 || (best_fit_path.calculateLength() / path.calculateLength()) > 2)
            best_fit_path = path;

        return best_fit_path;
    }

    std::pair<Path, double> CurveFitting::FitPathToArc(const Path& path, int dim)
    {
        Path new_path;
        double total_error = 0.0;
        for(int i = 0, max_index = path.size() - (path.size() % (dim + 1)) - (dim + 1); i <= max_index; i += (dim + 1))
        {
            int end_segment_index = i + dim;

            auto arc_fit = FitArcToSection(path, i, end_segment_index);
            if(arc_fit.first != nullptr)
                new_path.append(arc_fit.first);
            total_error += arc_fit.second;
        }

        // Determine how many segments need to be added
        int segments_to_add = path.size() % (dim + 1);
        if(segments_to_add >= 2) // FitArc with smaller arc
        {
            auto arc_fit = FitArcToSection(path, path.size() - segments_to_add, path.size() - 1);
            new_path.append(arc_fit.first);
            total_error += arc_fit.second;
        }else if(segments_to_add == 1)
        {
            new_path.append(path.back());
        }

        return std::pair<Path, double>(new_path, total_error);
    }

    void CurveFitting::SplineTraverse(QVector<BezierSegment>& spline, Path& parent_path, int start, int end)
    {
        if(start == end) // Stop recursion if we have run out of points to fit
            return;

        int middle = qFloor((end + start) / 2.0);
        auto fit = BezierSegment::Fit(start, end, parent_path);

        int size_before = spline.size(); // Remember how many segments existed before more recursion/ adding

        if(fit.second <= 500000.0) // This value may need to be tuned further
            spline.append(fit.first);
        else
        {
            if((middle - start) >= 1) // Do left side
                SplineTraverse(spline, parent_path, start, middle);

            if((end - middle) >= 1) // Do right side
                SplineTraverse(spline, parent_path, middle + 1, end);
        }

        if(size_before == spline.size()) // If no further dividing was possible, add this as out best fit
            spline.append(fit.first);
    }
}
