#ifdef NVCC_FOUND
#ifndef GPUCROSSSECTIONSEGMENT_H
#define GPUCROSSSECTIONSEGMENT_H

#include "cuda/cuda_macros.h"
#include "geometry/mesh/mesh_vertex.h"
#include "cross_section/cross_section_segment.h"
#include "geometry/gpu/gpu_point.h"
#include "geometry/mesh/gpu/gpu_mesh_vertex.h"

namespace CUDA
{
    //! \class GPUCrossSectionSegment
    //! \brief a GPU safe version of CrossSectionSegment
    class GPUCrossSectionSegment
    {
    public:
        //! \brief Constructor
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        GPUCrossSectionSegment();

        //! \brief converts this back to a regular CrossSectionSegment
        //! \note This call is ONLY CPU safe
        CPU_ONLY
        ORNL::CrossSectionSegment toCrossSectionSegment();

        //! \brief if this segment contains data
        bool valid = false;

        //! \brief Start and end points
        GPUPoint start, end;

        //! \brief Index of face
        int face_index;

        //! \brief Index of other face connected via the edge that created end
        int end_other_face_idx;

        //! \brief End vertex
        GPUMeshVertex end_vertex;

        //! \brief Whether or not segment has been added to polygon
        bool added_to_polygon;

        //! \brief Face normal
        GPUVector3D normal;
    };
}

#endif // GPUCROSSSECTIONSEGMENT_H
#endif
