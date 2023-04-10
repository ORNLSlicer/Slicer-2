#ifdef NVCC_FOUND

#ifndef GPU_CURVE_FITTING
#define GPU_CURVE_FITTING

// Local
#include "cuda/cuda_macros.h"
#include "geometry/path.h"
#include "geometry/segments/arc.h"
#include "geometry/gpu/gpu_point.h"

namespace CUDA
{
    namespace GPUCurveFitting
    {
        //! \brief fits the best arc segment across a set of points using the gpu
        //! \param path the parent path
        //! \param start_index the start of the fit
        //! \param end_index the end of the fit
        //! \return a pair of the best fit and its error
        CPU_ONLY
        QPair<ORNL::ArcSegment, double> FindBestArcFit(ORNL::Path& path, int start_index, int end_index);

        //! \struct Arc
        //! \brief a slimmed-down arc type to be stored on the GPU
        struct Arc
        {
            //! \brief constructor
            //! \param _start the start point
            //! \param _center the center point
            //! \param _end the end point
            //! \param _ccw if this arc is counterclockwise
            GPU_CPU_CODE
            Arc(GPUPoint _start, GPUPoint _center, GPUPoint _end, bool _ccw) :
                start(_start), center(_center), end(_end), ccw(_ccw) {}

            //! \brief default constructor
            GPU_CPU_CODE
            Arc() {}

            //! \brief greater than operator that compares arcs based on error
            //! \param rhs the right hand side
            //! \return if the left hand side is greater than right hand side
            GPU_CPU_CODE
            bool operator>(const Arc& rhs) const
            {
                return error > rhs.error;
            }

            //! \brief less than operator that compares arcs based on error
            //! \param rhs the right hand side
            //! \return if the left hand side is less than right hand side
            GPU_CPU_CODE
            bool operator<(const Arc& rhs) const
            {
                return error < rhs.error;
            }

            //! \brief internal variables
            double error;
            GPUPoint start;
            GPUPoint end;
            GPUPoint center;
            bool ccw;
        };

        //! \brief CUDA Kernel that computes an all the options of an arc fit's mid point in parallel
        //! \param points list of points in path
        //! \param arcs the arcs to save
        //! \param size the number of point
        //! \param start the start point of the path
        //! \param end the end point of the path
        GPU_LAUNCHABLE
        void ComputeArcError(GPUPoint *points, Arc *arcs, int size, GPUPoint start, GPUPoint end);

        //! \brief device code implementation of angle from axis function in math utils
        //! \param A the first point
        //! \param B the second point
        //! \return the angle
        GPU_CPU_CODE
        double angleFromXAxis(GPUPoint A, GPUPoint B);

        //! \brief Determines the orientation of 3 points with respect to each other. Helper function for intersect.
        //! \param p first point
        //! \param q second point
        //! \param r third point
        //! \return the winding of the points
        GPU_CPU_CODE
        int orientation(GPUPoint p, GPUPoint q, GPUPoint r);

        //! \brief device capable fuzzy compare
        //! \param a first value
        //! \param b second value
        //! \return if a and b are the same
        GPU_CPU_CODE
        bool equals(double a, double b);
    }
}

#endif // GPU_CURVE_FITTING
#endif // NVCC_FOUND
