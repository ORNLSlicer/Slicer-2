#include "geometry/mesh/gpu/gpu_mesh_vertex.h"

namespace CUDA
{
    GPU_CPU_CODE
    GPUMeshVertex::GPUMeshVertex() {}

    CPU_ONLY
    GPUMeshVertex::GPUMeshVertex(ORNL::MeshVertex vertex)
    {
        location = GPUPoint(vertex.location.x(), vertex.location.y(), vertex.location.z());
        normal = GPUVector3D(vertex.normal);
    }

    CPU_ONLY
    ORNL::MeshVertex GPUMeshVertex::toMeshVertex()
    {
        ORNL::MeshVertex vertex;
        vertex.normal = static_cast<QVector3D>(normal);
        vertex.location = static_cast<QVector3D>(GPUVector3D(location));
        return vertex;
    }
}
