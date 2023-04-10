#ifdef NVCC_FOUND

#ifndef GPU_FACE_FINDER
#define GPU_FACE_FINDER

// Locals
#include "cuda/cuda_macros.h"
#include "geometry/mesh/gpu/gpu_face.h"

namespace CUDA
{
    /*!
     * \class GPUFaceFinder uses CUDA to find if a point in on face in parallel
     */
    class GPUFaceFinder
    {
    public:
        //! \brief Default Constructor
        GPUFaceFinder() = default;

        CPU_ONLY
        //! \brief Constructor
        //! \param host_faces: the faces on the GPU
        GPUFaceFinder(QVector<GPUFace>& host_faces);

        CPU_ONLY
        //! \brief Destructor to free GPU memory
        ~GPUFaceFinder();

        //! \brief finds if and what face a 2D point is on within the surface mesh
        //! \param point: a 2D point
        //! \return a tuple with: bool for it was on the surface, a 3D vector of barycentric coords and the face
        CPU_ONLY
        QPair<bool, unsigned long> findFace(ORNL::Point point);

    private:
        //! \brief Faces and UV points stored on the GPU
        GPUFace * m_face_ptr;

        //! \brief the number of faces
        int m_num_faces;
    };
}

#endif // GPU_FACE_FINDER
#endif // NVCC_FOUND