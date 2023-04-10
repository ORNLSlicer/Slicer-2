#ifdef NVCC_FOUND

#ifndef GPUPLANE_H
#define GPUPLANE_H

#include "cuda/cuda_macros.h"
#include "geometry/plane.h"
#include "geometry/gpu/gpu_point.h"
#include "geometry/gpu/gpu_vector_3d.h"

namespace CUDA
{
    //! \class GPUPlane
    //! \brief a plane class that is GPU-safe
    class GPUPlane
    {
    public:
        //! \brief Constructor
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUPlane();

        //! \brief Conversion Constructor
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUPlane(GPUPoint& _point, GPUVector3D _vector);

        //! \brief Conversion Constructor
        //! \note This call is ONLY CPU safe
        CPU_ONLY
        explicit GPUPlane(ORNL::Plane& plane);

        //! \brief Determines if a point is (<0 == bellow), (0 == on), or (>0 above) the plane
        //! \return result of the plane equation
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        double evaluatePoint(const GPUPoint& p) const;

        //! \brief Returns the distance between the plane and the given point
        //! \return distance to point
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        double distanceToPoint(GPUPoint point);

        //! \brief gets the point of the plane
        //! \return a GPUPoint
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUPoint point() const;

        //! \brief gets the normal of the plane
        //! \return a GPUVector3D
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUVector3D normal() const;

    private:
        GPUPoint m_point;
        GPUVector3D m_normal;
    };
}
#endif // GPUPLANE_H
#endif // NVCC_FOUND
