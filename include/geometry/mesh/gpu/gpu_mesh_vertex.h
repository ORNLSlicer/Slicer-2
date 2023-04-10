#ifdef NVCC_FOUND
#ifndef GPUMESHVERTEX_H
#define GPUMESHVERTEX_H

#include "geometry/mesh/mesh_vertex.h"
#include "cuda/cuda_macros.h"

#include "geometry/gpu/gpu_point.h"
#include "geometry/gpu/gpu_vector_3d.h"

namespace CUDA
{
    //! \class GPUMeshVertex
    //! \brief a GPU safe MeshVertex
    class GPUMeshVertex
    {
    public:
        //! \brief Constructor
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUMeshVertex();

        //! \brief Conversion Constructor
        //! \param vertex the regular MeshVertex
        //! \note This call is ONLY CPU safe
        CPU_ONLY
        explicit GPUMeshVertex(ORNL::MeshVertex vertex);

        //! \brief converts this to a regular MeshVertex
        //! \return regular MeshVertex
        //! \note This call is ONLY CPU safe
        CPU_ONLY
        ORNL::MeshVertex toMeshVertex();

        //! \brief the point location of this vertex
        GPUPoint location;

        //! \brief the normal of this point
        GPUVector3D normal;
    };
}

#endif // GPUMESHVERTEX_H
#endif // NVCC_FOUND
