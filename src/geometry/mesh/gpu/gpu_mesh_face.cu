#include "geometry/mesh/gpu/gpu_mesh_face.h"

namespace CUDA
{
    GPU_CPU_CODE
    GPUMeshFace::GPUMeshFace()
    {

    }

    CPU_ONLY
    GPUMeshFace::GPUMeshFace(ORNL::MeshFace& face)
    {
        vertex_index[0] = face.vertex_index[0];
        vertex_index[1] = face.vertex_index[1];
        vertex_index[2] = face.vertex_index[2];

        connected_face_index[0] = face.connected_face_index[0];
        connected_face_index[1] = face.connected_face_index[1];
        connected_face_index[2] = face.connected_face_index[2];

        ignore = face.ignore;

        normal = GPUVector3D(face.normal);
    }
}
