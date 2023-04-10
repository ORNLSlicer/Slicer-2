#include "geometry/gpu/gpu_curve_fitting.h"

// CUDA
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

namespace CUDA
{
    namespace GPUCurveFitting
    {
        CPU_ONLY
        QPair<ORNL::ArcSegment, double> FindBestArcFit(ORNL::Path& path, int start_index, int end_index)
        {
            int size = (end_index - start_index) + 2; // Total number of points

            if(size == 3) // No need to find best fit when there are is only one option
            {
                return qMakePair(ORNL::ArcSegment(path[start_index]->start(), path[start_index]->end(), path[end_index]->end()), 0.0);
            }else
            {
                // Copy path points to GPU
                thrust::host_vector<GPUPoint> h_points;
                for(int i = start_index; i < end_index; ++i)
                {
                    auto seg = path[i];
                    h_points.push_back(GPUPoint(seg->end()));
                }
                GPUPoint start = static_cast<GPUPoint>(path[start_index]->start());
                GPUPoint end = static_cast<GPUPoint>(path[end_index]->end());

                thrust::device_vector<GPUPoint> d_points = h_points;

                // Array of size dim to store results
                thrust::device_vector<Arc> d_arcs(size - 2);

                GPUPoint * gpu_points = thrust::raw_pointer_cast(d_points.data());
                Arc * arcs = thrust::raw_pointer_cast(d_arcs.data());
                ComputeArcError<<<1, size - 2>>>(gpu_points, arcs, size, start, end);
                cudaDeviceSynchronize();

                auto result = thrust::min_element(thrust::device, d_arcs.begin(), d_arcs.end());
                Arc best_fit = *result;

                return qMakePair(ORNL::ArcSegment(static_cast<ORNL::Point>(best_fit.start), static_cast<ORNL::Point>(best_fit.end),
                                                  static_cast<ORNL::Point>(best_fit.center), best_fit.ccw), best_fit.error);
            }
        }

        GPU_LAUNCHABLE
        void ComputeArcError(GPUPoint *points, Arc *arcs, int size, GPUPoint start, GPUPoint end)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            GPUPoint middle = points[index];

            // Calculate center
            // Find the perpendicular bisector of start -> middle and end -> middle
            GPUPoint mid_start_middle = start;
            GPUPoint mid_end_middle = end;
            mid_start_middle.moveTowards(middle, start.distance(middle)/ 2.0f);
            mid_end_middle.moveTowards(middle, end.distance(middle)/ 2.0f);

            double slope_start_middle = 0.0;
            double slope_end_middle = 0.0;
            double x = 0.0;
            double y = 0.0;

            if(!equals(middle.x(),start.x()) && !equals(end.x(), middle.x())) // If neither of the lines are vertical
            {
                slope_start_middle = (middle.y() - start.y()) / (middle.x() - start.x());
                slope_end_middle = (middle.y() - end.y()) / (middle.x() - end.x());
                x = (((slope_start_middle * slope_end_middle) * (start.y() - end.y())) + (slope_end_middle * (start.x() + middle.x())) - (slope_start_middle * (middle.x() + end.x()))) / (2 * (slope_end_middle - slope_start_middle));
                y = (-1 / slope_start_middle) * (x - ((start.x() + middle.x()) / 2)) + ((start.y() + middle.y()) / 2);
            }else if(equals(middle.x(),start.x()) && !equals(end.x(), middle.x())) // The start- > middle line is vertical
            {
                slope_end_middle = (end.y() - middle.y()) / (end.x() - middle.x());
                y = (middle.y() + start.y()) / 2;
                x = (y - ((end.y() + middle.y()) / 2)) * (slope_end_middle / -1) + ((end.x() + middle.x()) / 2);
            }else if(!equals(middle.x(),start.x()) && equals(end.x(), middle.x()))
            {
                slope_start_middle = (middle.y() - start.y()) / (middle.x() - start.x());
                y = (middle.y() + end.y()) / 2;
                x = (y - ((start.y() + middle.y()) / 2)) * (slope_start_middle / -1) + ((start.x() + middle.x()) / 2);
            } // else Both are vertical, therefore co-linear

            GPUPoint center =  GPUPoint(x,y,((end.z() - start.z()) / 2) + start.z());

            Arc arc(start, center, end, orientation(start, middle, end) == 2);

            double error = 0.0;

            double radius = center.distance(arc.start);
            double arc_end_angle = angleFromXAxis(arc.center, arc.end);
            double arc_start_angle = angleFromXAxis(arc.center, arc.start);

            int end_index = size - 1;
            for(int i = 0; i < end_index; ++i)
            {
                GPUPoint end = points[i];
                double segment_angle = angleFromXAxis(arc.center, end);

                if(arc.ccw && arc_start_angle > arc_end_angle)
                {
                    if((segment_angle < arc_end_angle) || (segment_angle > arc_start_angle))
                        error += fabs(radius - center.distance(end));
                    else
                        error += fabs(radius + center.distance(end));
                }else if(!arc.ccw && arc_start_angle < arc_end_angle)
                {
                    if((segment_angle < arc_start_angle) || (segment_angle > arc_end_angle))
                        error += fabs(radius - center.distance(end));
                    else
                        error += fabs(radius + center.distance(end));
                }
                else
                {
                    if((segment_angle > arc_end_angle && segment_angle < arc_start_angle) || (segment_angle < arc_end_angle && segment_angle > arc_start_angle))
                        error += fabs(radius - center.distance(end));
                    else
                        error += fabs(radius + center.distance(end));
                }
            }

            arc.error = error;
            arcs[index] = arc;
        }

        GPU_CPU_CODE
        double angleFromXAxis(GPUPoint A, GPUPoint B)
        {
            double result;
            double dx = B.x() - A.x();
            double dy = B.y() - A.y();

            if(dx > 0 && dy > 0)
                result = atan(dy / dx);
            else if(dx < 0 && dy > 0)
                result =  3.141592654f + atan(dy / dx);
            else if(dx < 0 && dy < 0)
                result =  3.141592654f + atan(dy / dx);
            else if(dx > 0 && dy < 0)
                result = (2.0 *  3.141592654f) + atan(dy / dx);
            else if(equals(dx,0))
            {
                if(dy >= 0)
                    result =  3.141592654f / 2.0;
                else
                    result = (3.0 *  3.141592654f) / 2.0;
            }
            else if(equals(dy,0))
            {
                if(dx >= 0)
                    result = 0.00;
                else
                    result =  3.141592654f;
            }
            else
                result = NAN;

            return result;
        }

        GPU_CPU_CODE
        int orientation(GPUPoint p, GPUPoint q, GPUPoint r)
        {
            float val = (q.y() - p.y()) * (r.x() - q.x()) - (q.x() - p.x()) * (r.y() - q.y());

            if (val == 0.0f) // co-linear
                return 0;

            return (val > 0)? 1: 2; // clockwise or counter-clockwise
        }

        GPU_CPU_CODE
        bool equals(double a, double b)
        {
            return fabs(a - b) < 0.00001;
        }
    }
}
