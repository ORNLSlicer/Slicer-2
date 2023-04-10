#ifdef NVCC_FOUND

#ifndef GPUMESHFACE_H
#define GPUMESHFACE_H

#include "cuda/cuda_macros.h"
#include "geometry/mesh/mesh_face.h"
#include "geometry/gpu/gpu_vector_3d.h"

namespace CUDA
{
    //! \class GPUMeshFace
    //! \brief a GPU safe MeshFace
    class GPUMeshFace
    {
    public:
        //! \brief Constructor
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUMeshFace();

        //! \brief Conversion Constructor
        //! \note This call is ONLY CPU safe
        CPU_ONLY
        explicit GPUMeshFace(ORNL::MeshFace& face);

        //! \brief vertices in counter-clockwise ordering
        int vertex_index[3];

        //! \brief faces that are connected to this one
        int connected_face_index[3];  //!< same ordering as vertex_index
                                      //!(connected_face is connected via vertex
                                      //! 0 and
                                      //! 1, etc)

        //! \brief the index of this face in the mesh face array
        int face_index;

        //! \brief the normal of this face
        GPUVector3D normal;

        //! \brief if this face should be ignored in cross-sectioning
        bool ignore = false;
    };
}

#endif // GPUMESHFACE_H
#endif // NVCC_FOUND
