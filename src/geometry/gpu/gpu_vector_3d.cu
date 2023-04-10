#include "geometry/gpu/gpu_vector_3d.h"

#include <QtMath>
#include <float.h>

namespace CUDA
{
    GPU_CPU_CODE
    GPUVector3D::GPUVector3D() {}

    CPU_ONLY
    GPUVector3D::GPUVector3D(QVector3D &vec)
    {
        m_x = vec.x();
        m_y = vec.y();
        m_z = vec.z();
    }

    CPU_ONLY
    GPUVector3D::GPUVector3D(ORNL::Point& p)
    {
        m_x = p.x();
        m_y = p.y();
        m_z = p.z();
    }

    GPU_CPU_CODE
    GPUVector3D::GPUVector3D(GPUPoint &p)
    {
        m_x = p.x();
        m_y = p.y();
        m_z = p.z();
    }

    GPU_CPU_CODE
    GPUVector3D::GPUVector3D(float x, float y, float z)
    {
        m_x = x;
        m_y = y;
        m_z = z;
    }

    CPU_ONLY
    GPUVector3D::operator QVector3D()
    {
        return QVector3D(m_x, m_y, m_z);
    }

    GPUPoint GPUVector3D::toGPUPoint()
    {
        return GPUPoint(m_x, m_y, m_z);
    }

    double GPUVector3D::length()
    {
        double len = m_x * m_x +
                     m_y * m_y +
                     m_z * m_z;

        return std::sqrt(len);
    }

    GPU_CPU_CODE
    float GPUVector3D::dot(GPUVector3D& rhs)
    {
        return m_x * rhs.m_x + m_y * rhs.m_y + m_z * rhs.m_z;
    }

    GPU_CPU_CODE
    GPUVector3D GPUVector3D::cross(GPUVector3D& rhs)
    {
        return GPUVector3D(m_y * rhs.m_z - m_z * rhs.m_y,
                           m_z * rhs.m_x - m_x * rhs.m_z,
                           m_x * rhs.m_y - m_y * rhs.m_x);
    }

    GPU_CPU_CODE
    void GPUVector3D::normalize()
    {
        #ifdef DEVICE_CODE_COMPILATION
        float len = norm3d(m_x, m_y, m_z);
        m_x /= len;
        m_y /= len;
        m_z /= len;
        #else
            QVector3D temp(*this);
            temp.normalize();
            m_x = temp.x();
            m_y = temp.y();
            m_z = temp.z();
        #endif
    }

    GPUVector3D GPUVector3D::normalized()
    {
        GPUVector3D temp(*this);
        temp.normalize();
        return temp;
    }

    GPU_CPU_CODE
    bool GPUVector3D::operator==(GPUVector3D &rhs) {
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
    bool GPUVector3D::operator!=(GPUVector3D &rhs) {
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
    GPUVector3D GPUVector3D::operator+(GPUVector3D &point) {
        return GPUVector3D(m_x + point.x(), m_y + point.y(), m_z + point.z());
    }

    GPU_CPU_CODE
    GPUVector3D GPUVector3D::operator+=(GPUVector3D &rhs) {
        m_x += rhs.m_x;
        m_y += rhs.m_y;
        m_z += rhs.m_z;
        return *this;
    }

    GPU_CPU_CODE
    GPUVector3D GPUVector3D::operator-(GPUVector3D &rhs) {
        return GPUVector3D(m_x - rhs.m_x, m_y - rhs.m_y, m_z - rhs.m_z);
    }

    GPU_CPU_CODE
    GPUVector3D GPUVector3D::operator-=(GPUVector3D &rhs)
    {
        m_x -= rhs.m_x;
        m_y -= rhs.m_y;
        m_z -= rhs.m_z;
        return *this;
    }

    GPU_CPU_CODE
    GPUVector3D GPUVector3D::operator*(float rhs) {
        return GPUVector3D(rhs * m_x, rhs * m_y, rhs * m_z);
    }

    GPU_CPU_CODE
    GPUVector3D GPUVector3D::operator*=(float rhs) {
        m_x *= rhs;
        m_y *= rhs;
        m_z *= rhs;
        return *this;
    }

    GPU_CPU_CODE
    GPUVector3D GPUVector3D::operator/(float rhs) {
        return GPUVector3D(m_x / rhs, m_y / rhs, m_z / rhs);
    }

    GPU_CPU_CODE GPUVector3D GPUVector3D::operator/=(float rhs) {
        m_x /= rhs;
        m_y /= rhs;
        m_z /= rhs;
        return *this;
    }

    GPU_CPU_CODE
    float GPUVector3D::x()
    {
        return m_x;
    }

    GPU_CPU_CODE
    float GPUVector3D::x() const {
        return m_x;
    }

    GPU_CPU_CODE
    void GPUVector3D::x(float x)
    {
        m_x = x;
    }

    GPU_CPU_CODE
    float GPUVector3D::y()
    {
        return m_y;
    }

    GPU_CPU_CODE
    float GPUVector3D::y() const
    {
        return m_y;
    }

    GPU_CPU_CODE
    void GPUVector3D::y(float y)
    {
        m_y = y;
    }

    GPU_CPU_CODE
    float GPUVector3D::z()
    {
        return m_z;
    }

    GPU_CPU_CODE
    float GPUVector3D::z() const
    {
        return m_z;
    }

    GPU_CPU_CODE
    void GPUVector3D::z(float z)
    {
        m_z = z;
    }
}


