#ifdef NVCC_FOUND
#ifndef GPU_VECTOR_3D
#define GPU_VECTOR_3D

#include "cuda/cuda_macros.h"
#include "geometry/point.h"
#include "geometry/gpu/gpu_point.h"

namespace CUDA
{
     //! \class GPUVector3D
     //! \brief 3D vector class that exists on the GPU
    class GPUVector3D
    {
    public:
        //! \brief Constructor
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUVector3D();

        //! \brief Conversion Constructor
        //! \param vec a qt vector
        //! \note This call is ONLY CPU safe
        CPU_ONLY
        explicit GPUVector3D(QVector3D& vec);

        //! \brief Conversion Constructor
        //! \param p an regular point
        //! \note This call is ONLY CPU safe
        CPU_ONLY
        explicit GPUVector3D(ORNL::Point& p);

        //! \brief Conversion Constructor
        //! \param p a GPU point
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        explicit GPUVector3D(GPUPoint& p);

        //! \brief Conversion Constructor
        //! \param x x coordinate
        //! \param y y coordinate
        //! \param z z coordinate
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUVector3D(float x, float y, float z);

        //! \brief Conversion Operator
        //! \return a QT vector from this class
        //! \note This call is ONLY CPU safe
        CPU_ONLY
        explicit operator QVector3D();

        //! \brief Converts this class to a GPUPoint
        //! \return a GPUPoint
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUPoint toGPUPoint();

        //! \brief computes the length of the vector
        //! \return the length
        //! \note after calling normalize() this will always be 1
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        double length();

        //! \brief computes the dot product between two vectors
        //! \param rhs: the second vector
        //! \return the dot product
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        float dot(GPUVector3D& rhs);

        //! \brief computes the cross product between two vectors
        //! \param rhs the second vector
        //! \return the cross product
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUVector3D cross(GPUVector3D& rhs);

        //! \brief normalizes the vector to a unit length of 1
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        void normalize();

        //! \brief gets this vector as a normalized vector to a unit length of 1
        //! \return normalized GPUVector3D
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUVector3D normalized();

        //! \brief addition operator
        //! \param a GPUVector3D
        //! \return a GPUVector3D
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUVector3D operator+(GPUVector3D& point);

        //! \brief addition equals operator
        //! \param a GPUVector3D
        //! \return a GPUVector3D
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUVector3D operator+=(GPUVector3D& rhs);

        //! \brief subtraction operator
        //! \param a GPUVector3D
        //! \return a GPUVector3D
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUVector3D operator-(GPUVector3D& rhs);

        //! \brief subtraction equals operator
        //! \param a GPUVector3D
        //! \return a GPUVector3D
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUVector3D operator-=(GPUVector3D& rhs);

        //! \brief multiplication operator
        //! \param a GPUVector3D
        //! \return a GPUVector3D
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUVector3D operator*(float rhs);

        //! \brief multiplication equals operator
        //! \param a GPUVector3D
        //! \return a GPUVector3D
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUVector3D operator*=(float rhs);

        //! \brief division operator
        //! \param a GPUVector3D
        //! \return a GPUVector3D
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUVector3D operator/(float rhs);

        //! \brief division equals operator
        //! \param a GPUVector3D
        //! \return a GPUVector3D
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUVector3D operator/=(float rhs);

        //! \brief equality operator
        //! \param a GPUVector3D
        //! \return is the vectors are the same
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        bool operator==(GPUVector3D& rhs);

        //! \brief not equals operator
        //! \param a GPUVector3D
        //! \return if the vectors are not the same
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        bool operator!=(GPUVector3D& rhs);

        //! \brief gets the X component
        //! \return the X component
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        float x();

        //! \brief gets the X component
        //! \return the X component
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        float x() const;

        //! \brief sets the X component
        //! \param the X component
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        void x(float x);

        //! \brief gets the Y component
        //! \return the Y component
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        float y();

        //! \brief gets the Y component
        //! \return the Y component
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        float y() const;

        //! \brief sets the Y component
        //! \param the Y component
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        void y(float y);

        //! \brief gets the Z component
        //! \return the Z component
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        float z();

        //! \brief gets the Z component
        //! \return the Z component
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        float z() const;

        //! \brief sets the Z component
        //! \param the Z component
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        void z(float z);

    private:
        //! \brief the XYZ components of the vector
        float m_x;
        float m_y;
        float m_z;
    };
}

#endif // GPU_VECTOR_3D
#endif // NVCC_FOUND
