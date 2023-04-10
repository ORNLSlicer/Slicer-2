#ifdef NVCC_FOUND

#ifndef GPU_POLYGON
#define GPU_POLYGON

#include "cuda/cuda_macros.h"

// CUDA
#include <thrust/device_vector.h>

// Local
#include "geometry/gpu/gpu_point.h"
#include "geometry/polygon.h"
#include "geometry/point.h"

namespace CUDA
{
    /*!
     * \class GPUPolygon
     * \brief is a polygon the exists on the GPU
     * \note conversion to and from this type incurs costly GPU memory copies
     */
    class GPUPolygon
    {
    public:

        GPU_CPU_CODE
        //! \brief Constructor
        //! \note This call is GPU and CPU safe
        GPUPolygon();

        CPU_ONLY
        //! \brief CPU Conversion Constructor
        //! \param cpu_polygon the polygon to GPU to the GPU
        //! \note This call is ONLY CPU safe
        GPUPolygon(ORNL::Polygon& cpu_polygon);

        CPU_ONLY
        //! \brief CPU Conversion Constructor
        //! \param cpu_points the points to copy to the GPU
        //! \note This call is ONLY CPU safe
        GPUPolygon(QVector<ORNL::Point> cpu_points);

        CPU_ONLY
        //! \brief CPU Conversion operator
        //! \note This call is ONLY CPU safe
        operator ORNL::Polygon();

        CPU_ONLY
        //! \brief Equality operator
        //! \param rhs: the other polygon
        //! \return is all the points match
        //! \note This call is ONLY CPU safe
        bool operator==(GPUPolygon& rhs);

        CPU_ONLY
        //! \brief Equality not operator
        //! \param rhs: the other polygon
        //! \return is any point is not the same between the polygons
        //! \note This call is ONLY CPU safe
        bool operator!=(GPUPolygon& rhs);

        CPU_ONLY
        //! \brief Accessor
        //! \param the index to access
        //! \return a point
        //! \note This call is ONLY CPU safe
        GPUPoint operator[](int index);

        CPU_ONLY
        //! \brief fetchs the points as a CPU copy
        //! \return a list of the points
        //! \note This call is ONLY CPU safe
        QVector<ORNL::Point>& points();

        CPU_ONLY
        //! \brief a GPU vector of points
        //! \return a thrust vector of points
        //! \note This call is ONLY CPU safe
        thrust::device_vector<GPUPoint>& devicePoints();

        CPU_ONLY
        //! \brief the number of points in the polygon
        //! \return the number of points
        //! \note This call is ONLY CPU safe
        int size();

    private:
        //! \brief the thrust GPU vector that holds the points
        thrust::device_vector<GPUPoint> m_dev_vector;
    };

    GPU_LAUNCHABLE
    //! \brief an external GPU kernel to check equality in parallel
    //! \param a: raw GPU pointer
    //! \param b: raw GPU pointer
    //! \param equal: flag for if they are equal
    //! \param size: the size of the polygons. MUST be the same
    //! \note This call is ONLY GPU launch-able
    void equalityKernel(GPUPoint* a, GPUPoint* b, bool& equal, unsigned long size);
}

#endif // GPU_POLYGON
#endif // NVCC_FOUND
