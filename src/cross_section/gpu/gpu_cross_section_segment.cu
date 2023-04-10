#include "cross_section/gpu/gpu_cross_section_segment.h"

namespace CUDA
{
    GPU_CPU_CODE
    GPUCrossSectionSegment::GPUCrossSectionSegment()
    {
        face_index         = -1;
        end_other_face_idx = -1;
        added_to_polygon   = false;
    }

    CPU_ONLY
    ORNL::CrossSectionSegment GPUCrossSectionSegment::toCrossSectionSegment()
    {
        ORNL::CrossSectionSegment segment;
        segment.start = static_cast<ORNL::Point>(start);
        segment.end = static_cast<ORNL::Point>(end);
        segment.face_index = face_index;
        segment.end_other_face_idx = end_other_face_idx;
        segment.end_vertex = new ORNL::MeshVertex(end_vertex.toMeshVertex());
        segment.added_to_polygon = added_to_polygon;
        segment.normal = static_cast<QVector3D>(normal);
        return segment;
    }
}
