#include "geometry/gpu/gpu_quaternion.h"

namespace CUDA
{
    GPU_CPU_CODE
    GPUQuaternion::GPUQuaternion() = default;

    CPU_ONLY
    GPUQuaternion::GPUQuaternion(QQuaternion q)
    {
        wp = q.scalar();
        xp = q.x();
        yp = q.y();
        zp = q.z();
    }

    GPU_CPU_CODE
    GPUQuaternion::GPUQuaternion(double w, double x, double y, double z)
    {
        wp = w;
        xp = x;
        yp = y;
        zp = z;
    }

    GPU_CPU_CODE
    GPUQuaternion GPUQuaternion::multiply(const GPUQuaternion &q1, const GPUQuaternion& q2)
    {
        float yy = (q1.wp - q1.yp) * (q2.wp + q2.zp);
        float zz = (q1.wp + q1.yp) * (q2.wp - q2.zp);
        float ww = (q1.zp + q1.xp) * (q2.xp + q2.yp);
        float xx = ww + yy + zz;
        float qq = 0.5f * (xx + (q1.zp - q1.xp) * (q2.xp - q2.yp));

        float w = qq - ww + (q1.zp - q1.yp) * (q2.yp - q2.zp);
        float x = qq - xx + (q1.xp + q1.wp) * (q2.xp + q2.wp);
        float y = qq - yy + (q1.wp - q1.xp) * (q2.yp + q2.zp);
        float z = qq - zz + (q1.zp + q1.yp) * (q2.wp - q2.xp);

        return {w, x, y, z};
    }

    GPU_CPU_CODE
    GPUQuaternion GPUQuaternion::conjugated() const
    {
        return {wp, -xp, -yp, -zp};
    }

    GPU_CPU_CODE
    GPUVector3D GPUQuaternion::rotatedVector(GPUVector3D vector)
    {
        auto a = multiply(*this, GPUQuaternion(0, vector.x(), vector.y(), vector.z()));
        auto b = multiply(a, conjugated());
        return GPUVector3D(b.x(), b.y(), b.z());
    }

    GPU_CPU_CODE
    double GPUQuaternion::w()
    {
        return wp;
    }

    GPU_CPU_CODE
    double GPUQuaternion::x()
    {
        return xp;
    }

    GPU_CPU_CODE
    double GPUQuaternion::y()
    {
        return yp;
    }

    GPU_CPU_CODE
    double GPUQuaternion::z()
    {
        return zp;
    }
}
