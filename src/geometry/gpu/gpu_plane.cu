#include "geometry/gpu/gpu_plane.h"

namespace CUDA
{
    GPU_CPU_CODE
    GPUPlane::GPUPlane() = default;

    GPU_CPU_CODE
    GPUPlane::GPUPlane(GPUPoint& _point, GPUVector3D _vector)
    {
        m_point = _point;
        m_normal = _vector;
        m_normal.normalize();
    }

    CPU_ONLY
    GPUPlane::GPUPlane(ORNL::Plane& plane)
    {
        auto p = plane.point();
        m_point = GPUPoint(p);
        auto n = plane.normal();
        m_normal = GPUVector3D(n);
    }

    GPU_CPU_CODE
    double GPUPlane::evaluatePoint(const GPUPoint& p) const
    {
        /*
           for normal vector <a, b, c> and point (x0, y0, z0)
           the equation for a plane is
           a(x - x0) + b(y - y0) + c(z - z0) = 0
           returns the value of the left side of the equ when
           point is substituted for x, y, z
        */
        double dx = p.x() - m_point.x();
        double dy = p.y() - m_point.y();
        double dz = p.z() - m_point.z();

        return (m_normal.x() * dx) + (m_normal.y() * dy) + (m_normal.z() * dz);
    }

    GPU_CPU_CODE
    double GPUPlane::distanceToPoint(GPUPoint point)
    {
        return evaluatePoint(point) / m_normal.length();
    }

    GPU_CPU_CODE
    GPUPoint GPUPlane::point() const
    {
        return m_point;
    }

    GPU_CPU_CODE
    GPUVector3D GPUPlane::normal() const
    {
        return m_normal;
    }
}
