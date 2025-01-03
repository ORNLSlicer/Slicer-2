#ifdef NVCC_FOUND

#ifndef GPU_POINT
#define GPU_POINT

#include "geometry/point.h"
#include "cuda/cuda_macros.h"

namespace CUDA
{
    /*!
     * \class GPUPoint
     * \brief point class that exists on the GPU
     */
    class GPUPoint
    {
    public:

        GPU_CPU_CODE
        //! \brief Constructor
        //! \note This call is GPU and CPU safe
        GPUPoint();

        CPU_ONLY
        //! \brief Conversion Constructor
        //! \note This call is ONLY CPU safe
        explicit GPUPoint(ORNL::Point& p);

        GPU_CPU_CODE
        //! \brief Constructor
        //! \note This call is GPU and CPU safe
        GPUPoint(float x, float y, float z);

        CPU_ONLY
        //! \brief Operator
        //! \note This call is ONLY CPU safe
        operator ORNL::Point();

        GPU_CPU_CODE
        //! \brief computes the distance between two points
        //! \param rhs the target point
        //! \return the distance
        //! \note This call is GPU and CPU safe
        float distance(GPUPoint& rhs);

        GPU_CPU_CODE
        //! \brief computes the dot product between two points
        //! \param rhs: the second point
        //! \return the dot product
        //! \note This call is GPU and CPU safe
        float dot(GPUPoint& rhs);

        GPU_CPU_CODE
        //! \brief computes the cross product between two points
        //! \param rhs: the second point
        //! \return the cross product
        //! \note This call is GPU and CPU safe
        GPUPoint cross(GPUPoint& rhs);

        GPU_CPU_CODE
        //! \brief moves this point towards another one a given distance
        //! \param target: the other point
        //! \param dist: the distance to move along a vector
        //! \note This call is GPU and CPU safe
        void moveTowards(GPUPoint& target, double dist);

        GPU_CPU_CODE
        //! \brief addition operator
        //! \param a GPU point
        //! \return a GPU point
        //! \note This call is GPU and CPU safe
        GPUPoint operator+(GPUPoint& point);

        GPU_CPU_CODE
        //! \brief addition equals operator
        //! \param a GPU point
        //! \return a GPU point
        //! \note This call is GPU and CPU safe
        GPUPoint operator+=(GPUPoint& rhs);

        GPU_CPU_CODE
        //! \brief subtraction operator
        //! \param a GPU point
        //! \return a GPU point
        //! \note This call is GPU and CPU safe
        GPUPoint operator-(GPUPoint& rhs);

        GPU_CPU_CODE
        //! \brief subtraction equals operator
        //! \param a GPU point
        //! \return a GPU point
        //! \note This call is GPU and CPU safe
        GPUPoint operator-=(GPUPoint& rhs);

        GPU_CPU_CODE
        //! \brief multiplication operator
        //! \param a GPU point
        //! \return a GPU point
        //! \note This call is GPU and CPU safe
        GPUPoint operator*(float rhs);

        GPU_CPU_CODE
        //! \brief multiplication equals operator
        //! \param a GPU point
        //! \return a GPU point
        //! \note This call is GPU and CPU safe
        GPUPoint operator*=(float rhs);

        GPU_CPU_CODE
        //! \brief division operator
        //! \param a GPU point
        //! \return a GPU point
        //! \note This call is GPU and CPU safe
        GPUPoint operator/(float rhs);

        GPU_CPU_CODE
        //! \brief division equals operator
        //! \param a GPU point
        //! \return a GPU point
        //! \note This call is GPU and CPU safe
        GPUPoint operator/=(float rhs);

        GPU_CPU_CODE
        //! \brief equality operator
        //! \param a GPU point
        //! \return is the points are the same
        //! \note This call is GPU and CPU safe
        bool operator==(GPUPoint& rhs);

        GPU_CPU_CODE
        //! \brief not equals operator
        //! \param a GPU point
        //! \return is the points are not the same
        //! \note This call is GPU and CPU safe
        bool operator!=(GPUPoint& rhs);

        GPU_CPU_CODE
        //! \brief gets the X component
        //! \return the X component
        //! \note This call is GPU and CPU safe
        float x();

        GPU_CPU_CODE
        //! \brief gets the X component
        //! \return the X component
        //! \note This call is GPU and CPU safe
        float x() const;

        GPU_CPU_CODE
        //! \brief sets the X component
        //! \param the X component
        //! \note This call is GPU and CPU safe
        void x(float x);

        GPU_CPU_CODE
        //! \brief gets the Y component
        //! \return the Y component
        //! \note This call is GPU and CPU safe
        float y();

        GPU_CPU_CODE
        //! \brief gets the Y component
        //! \return the Y component
        //! \note This call is GPU and CPU safe
        float y() const;

        GPU_CPU_CODE
        //! \brief sets the Y component
        //! \param the Y component
        //! \note This call is GPU and CPU safe
        void y(float y);

        GPU_CPU_CODE
        //! \brief gets the Z component
        //! \return the Z component
        //! \note This call is GPU and CPU safe
        float z();

        GPU_CPU_CODE
        //! \brief gets the Z component
        //! \return the Z component
        //! \note This call is GPU and CPU safe
        float z() const;

        GPU_CPU_CODE
        //! \brief sets the Z component
        //! \param the Z component
        //! \note This call is GPU and CPU safe
        void z(float z);

    private:
        //! \brief the XYZ components of the point
        float m_x;
        float m_y;
        float m_z;
    };
}

#endif // GPU_POINT
#endif // NVCC_FOUND
