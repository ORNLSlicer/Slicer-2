#include "geometry/gpu/gpu_point.h"

#include <QtMath>
#include <float.h>

namespace CUDA
{
    GPU_CPU_CODE
    GPUPoint::GPUPoint() {}

    CPU_ONLY
    GPUPoint::GPUPoint(ORNL::Point& p)
    {
        m_x = p.x();
        m_y = p.y();
        m_z = p.z();
    }

    GPU_CPU_CODE
    GPUPoint::GPUPoint(float x, float y, float z)
    {
        m_x = x;
        m_y = y;
        m_z = z;
    }

    CPU_ONLY
    GPUPoint::operator ORNL::Point()
    {
        return ORNL::Point(m_x, m_y, m_z);
    }

    GPU_CPU_CODE
    float GPUPoint::distance(GPUPoint& rhs)
    {
        return sqrtf(pow((double) (rhs.x() - m_x), 2.0) +
                     pow((double) (rhs.y() - m_y), 2.0) +
                     pow((double) (rhs.z() - m_z), 2.0));
    }

    GPU_CPU_CODE
    float GPUPoint::dot(GPUPoint& rhs)
    {
        return m_x * rhs.m_x + m_y * rhs.m_y + m_z * rhs.m_z;
    }

    GPU_CPU_CODE
    GPUPoint GPUPoint::cross(GPUPoint& rhs)
    {
        return GPUPoint(m_y * rhs.m_z - m_z * rhs.m_y,
                        m_z * rhs.m_x - m_x * rhs.m_z,
                        m_x * rhs.m_y - m_y * rhs.m_x);
    }

    GPU_CPU_CODE
    void GPUPoint::moveTowards(GPUPoint &target, double dist)
    {
        #ifdef DEVICE_CODE_COMPILATION
            float a = target.x() - m_x;
            float b = target.y() - m_y;
            float c = target.z() - m_z;

            float len = norm3d(a, b, c);

            a /= len;
            b /= len;
            c /= len;

            a *= dist;
            b *= dist;
            c *= dist;

            m_x += a;
            m_y += b;
            m_z += c;
        #else
            QVector3D norm = QVector3D(target.x() - m_x, target.y() - m_y, target.z() - m_z).normalized();

            norm *= dist;

            m_x += norm.x();
            m_y += norm.y();
            m_y += norm.y();
        #endif
    }

    GPU_CPU_CODE
    bool GPUPoint::operator==(GPUPoint &rhs) {
        #ifdef DEVICE_CODE_COMPILATION
        return fabs(m_x - rhs.x()) < FLT_EPSILON &&
               fabs(m_y - rhs.y()) < FLT_EPSILON &&
               fabs(m_z - rhs.z()) < FLT_EPSILON;
        #else
            return qFuzzyCompare(m_x - rhs.x(), 0) &&
                   qFuzzyCompare(m_y - rhs.y(), 0) &&
                   qFuzzyCompare(m_z - rhs.z(), 0);
        #endif
    }

    GPU_CPU_CODE
    bool GPUPoint::operator!=(GPUPoint &rhs) {
        #ifdef DEVICE_CODE_COMPILATION
                return !(fabs(m_x - rhs.x()) < FLT_EPSILON &&
                         fabs(m_y - rhs.y()) < FLT_EPSILON &&
                         fabs(m_z - rhs.z()) < FLT_EPSILON);
        #else
                return !(qFuzzyCompare(m_x - rhs.x(), 0) &&
                         qFuzzyCompare(m_y - rhs.y(), 0) &&
                         qFuzzyCompare(m_z - rhs.z(), 0));
        #endif
    }

    GPU_CPU_CODE
    GPUPoint GPUPoint::operator+(GPUPoint &point) {
        return GPUPoint(m_x + point.x(), m_y + point.y(), m_z + point.z());
    }

    GPU_CPU_CODE
    GPUPoint GPUPoint::operator+=(GPUPoint &rhs) {
        m_x += rhs.m_x;
        m_y += rhs.m_y;
        m_z += rhs.m_z;
        return *this;
    }

    GPU_CPU_CODE
    GPUPoint GPUPoint::operator-(GPUPoint &rhs) {
        return GPUPoint(m_x - rhs.m_x, m_y - rhs.m_y, m_z - rhs.m_z);
    }

    GPU_CPU_CODE
    GPUPoint GPUPoint::operator-=(GPUPoint &rhs)
    {
        m_x -= rhs.m_x;
        m_y -= rhs.m_y;
        m_z -= rhs.m_z;
        return *this;
    }

    GPU_CPU_CODE
    GPUPoint GPUPoint::operator*(float rhs) {
        return GPUPoint(rhs * m_x, rhs * m_y, rhs * m_z);
    }

    GPU_CPU_CODE
    GPUPoint GPUPoint::operator*=(float rhs) {
        m_x *= rhs;
        m_y *= rhs;
        m_z *= rhs;
        return *this;
    }

    GPU_CPU_CODE
    GPUPoint GPUPoint::operator/(float rhs) {
        return GPUPoint(m_x / rhs, m_y / rhs, m_z / rhs);
    }

    GPU_CPU_CODE GPUPoint GPUPoint::operator/=(float rhs) {
        m_x /= rhs;
        m_y /= rhs;
        m_z /= rhs;
        return *this;
    }

    GPU_CPU_CODE
    float GPUPoint::x()
    {
        return m_x;
    }

    GPU_CPU_CODE
    float GPUPoint::x() const {
        return m_x;
    }

    GPU_CPU_CODE
    void GPUPoint::x(float x)
    {
        m_x = x;
    }

    GPU_CPU_CODE
    float GPUPoint::y()
    {
        return m_y;
    }

    GPU_CPU_CODE
    float GPUPoint::y() const
    {
        return m_y;
    }

    GPU_CPU_CODE
    void GPUPoint::y(float y)
    {
        m_y = y;
    }

    GPU_CPU_CODE
    float GPUPoint::z()
    {
        return m_z;
    }

    GPU_CPU_CODE
    float GPUPoint::z() const
    {
        return m_z;
    }

    GPU_CPU_CODE
    void GPUPoint::z(float z)
    {
        m_z = z;
    }



}


