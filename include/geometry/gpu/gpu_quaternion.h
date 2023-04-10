#ifdef NVCC_FOUND
#ifndef GPUQUATERNION_H
#define GPUQUATERNION_H

#include "cuda/cuda_macros.h"
#include "geometry/gpu/gpu_vector_3d.h"
#include <QQuaternion>

namespace CUDA
{
    //! \class GPUQuaternion
    //! \brief a GPU safe quaternion class
    class GPUQuaternion
    {
    public:
        //! \brief Constructor
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUQuaternion();

        //! \brief Conversion Constructor
        //! \note This call is ONLY CPU safe
        CPU_ONLY
        explicit GPUQuaternion(QQuaternion q);

        //! \brief Conversion Constructor
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUQuaternion(double w, double x, double y, double z);

        //! \brief multiples two GPUQuaternion
        //! \param q1 right hand side
        //! \param q2 left hand side
        //! \return q1 * q2
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        static GPUQuaternion multiply(const GPUQuaternion &q1, const GPUQuaternion& q2);

        //! \brief conjugates this GPUQuaternion
        //! \return this GPUQuaternion conjugated
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUQuaternion conjugated() const;

        //! \brief returns the vector rotated by this quaternion
        //! \param vector the vector to rotate
        //! \return a rotated vector
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUVector3D rotatedVector(GPUVector3D vector);

        //! \brief gets the W value
        //! \return W as a double
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        double w();

        //! \brief gets the X value
        //! \return X as a double
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        double x();

        //! \brief gets the Y value
        //! \return Y as a double
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        double y();

        //! \brief gets the Z value
        //! \return Z as a double
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        double z();

    private:
        double wp;
        double xp;
        double yp;
        double zp;
    };
}

#endif // GPUQUATERNION_H
#endif // NVCC_FOUND
