#include "geometry/gpu/gpu_polygon.h"

#include <thrust/copy.h>
#include <thrust/host_vector.h>

namespace CUDA
{
    GPU_CPU_CODE
    GPUPolygon::GPUPolygon() {};

    CPU_ONLY
    GPUPolygon::GPUPolygon(ORNL::Polygon &cpu_polygon)
    {
        thrust::host_vector<GPUPoint> host_vector;
        for(auto& point : cpu_polygon)
            host_vector.push_back(static_cast<GPUPoint>(point));
        m_dev_vector = host_vector;
    }

    CPU_ONLY
    GPUPolygon::GPUPolygon(QVector<ORNL::Point> cpu_points)
    {
        thrust::host_vector<GPUPoint> host_vector;
        for(auto& point : cpu_points)
            host_vector.push_back(static_cast<GPUPoint>(point));
        m_dev_vector = host_vector;
    }

    CPU_ONLY
    GPUPolygon::operator ORNL::Polygon()
    {
        QVector<ORNL::Point> points;
        thrust::copy(m_dev_vector.begin(), m_dev_vector.end(), points.begin());
        return ORNL::Polygon(points);
    }

    CPU_ONLY
    bool GPUPolygon::operator==(GPUPolygon &rhs)
    {
        if(m_dev_vector.size() != rhs.size())
            return false;

        GPUPoint* a_ptr = thrust::raw_pointer_cast(m_dev_vector.data());
        GPUPoint* b_ptr = thrust::raw_pointer_cast(rhs.devicePoints().data());

        bool equal = true;
        equalityKernel<<<1, 1024>>>(a_ptr, b_ptr, equal, rhs.size());
        cudaDeviceSynchronize();

        return equal;
    }

    CPU_ONLY
    bool GPUPolygon::operator!=(GPUPolygon &rhs) {
        return !(*this == rhs);
    }

    CPU_ONLY
    GPUPoint GPUPolygon::operator[](int index)
    {
        return m_dev_vector[index];
    }

    CPU_ONLY
    int GPUPolygon::size() {
        return m_dev_vector.size();
    }

    CPU_ONLY
    QVector<ORNL::Point> &GPUPolygon::points()
    {
        QVector<ORNL::Point> points;
        thrust::copy(m_dev_vector.begin(), m_dev_vector.end(), points.begin());
        return points;
    }

    CPU_ONLY
    thrust::device_vector<GPUPoint> &GPUPolygon::devicePoints() {
        return m_dev_vector;
    }

    GPU_LAUNCHABLE
    void equalityKernel(GPUPoint* a, GPUPoint* b, bool &equal, unsigned long size)
    {
        unsigned long thread_id = blockIdx.x * blockDim.x + threadIdx.x;

        if(thread_id < size)
        {
            if(a[thread_id] != b[thread_id])
                equal = false;
        }
    }
}
