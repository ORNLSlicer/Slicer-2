#ifdef NVCC_FOUND

#ifndef GPU_CROSSSECTION_H
#define GPU_CROSSSECTION_H

// Local
#include "geometry/polygon_list.h"
#include "geometry/plane.h"
#include "geometry/mesh/mesh_face.h"
#include "geometry/mesh/mesh_vertex.h"
#include "configs/settings_base.h"
#include "cuda/cuda_macros.h"
#include "cross_section/cross_section_segment.h"

// CUDA
#include "cross_section/gpu/gpu_cross_section_segment.h"
#include "geometry/gpu/gpu_point.h"
#include "geometry/gpu/gpu_vector_3d.h"
#include "geometry/gpu/gpu_plane.h"
#include "geometry/gpu/gpu_quaternion.h"
#include "geometry/mesh/gpu/gpu_mesh_face.h"
#include "geometry/mesh/gpu/gpu_mesh_vertex.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace CUDA
{
    class GPUCrossSectioner
    {
    public:
        //! \brief GPU implementation of cross-sectioning
        //! \param vertices the vertices of the mesh
        //! \param faces the faces of the mesh
        //! \param mesh_min the min point on the mesh
        //! \param mesh_max the max point on the mesh
        //! \param sb the settings to use
        //! \note This call is ONLY CPU safe
        CPU_ONLY
        GPUCrossSectioner(QVector<ORNL::MeshVertex>& vertices, QVector<ORNL::MeshFace>& faces,
                          ORNL::Point mesh_min, ORNL::Point mesh_max,
                          QSharedPointer<ORNL::SettingsBase> sb);

        //! \brief Destructor
        //! \note This call is ONLY CPU safe
        CPU_ONLY
        ~GPUCrossSectioner();

        //! \brief performs a cross-section on the mesh with a plane
        //! \param slicing_plane the slicing plane
        //! \param shift modified value for how much shift was added
        //! \return a list of island geometry
        //! \note This call is ONLY CPU safe
        CPU_ONLY
        ORNL::PolygonList doCrossSectionGPU(ORNL::Plane& slicing_plane, ORNL::Point& shift);

        //! \brief finds the intersection point between a line formed between vertex0 and vertex1 with plane
        //! \param vertex0 the first point of the line
        //! \param vertex1 the second point of the line
        //! \param plane the plane to intersect with
        //! \return the intersection point
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        static GPUPoint findIntersectionGPU(const GPUPoint &vertex0, const GPUPoint &vertex1, const GPUPlane& plane);

        //! \brief rotates and shifts a point
        //! \param point_to_rotate point
        //! \param shift_amount how much to also shift by
        //! \param rotation the rotation
        //! \return the new point
        //! \note This call is GPU and CPU safe
        GPU_CPU_CODE
        static GPUPoint rotatePointGPU(GPUPoint point_to_rotate, GPUVector3D& shift_amount, GPUQuaternion& rotation);

        //! \brief finds the mid point of the mesh given a plane
        //! \param mesh_min the min point of the mesh
        //! \param mesh_max the max point of the mesh
        //! \param plane the plane
        //! \return the mid point
        CPU_ONLY
        static GPUPoint findSlicingPlaneMidPointGPU(GPUPoint &mesh_min, GPUPoint &mesh_max, GPUPlane& plane);

        //! \struct GPU_Intersection
        //! \brief CUDA call to transform GPUMeshFace into GPUCrossSectionSegment
        //! \note This struct should only be used with thrust::transform
        struct GPU_Intersection
        {
        public:
            GPUPlane plane;
            GPUMeshVertex* vertices;
            GPUPoint shift;
            GPUQuaternion rotation;

            //! \brief Constructor
            //! \param _plane the plane
            //! \param _vertices pointer to the vertices of the mesh ON THE GPU
            //! \param _shift the shift amount
            //! \param _rotation the rotation to apply
            //! \note This call should only be used with thrust::transform
            GPU_CPU_CODE
            GPU_Intersection(GPUPlane _plane, GPUMeshVertex* _vertices,
                           GPUPoint _shift, GPUQuaternion _rotation)
                           : plane(_plane), vertices(_vertices),
                             shift(_shift), rotation(_rotation) {}

            //! \brief operator to call the conversion
            //! \param face the face to transform
            //! \return a GPUCrossSectionSegment that is either valid == true (intersection) or valid == false (no intersection)
            //! \note This call should only be used with thrust::transform
            GPU_CPU_CODE
            GPUCrossSectionSegment operator()(GPUMeshFace face);
        };

    private:
        thrust::host_vector<GPUMeshFace> h_faces;
        thrust::host_vector<GPUMeshVertex> h_vertices;
        thrust::device_vector<GPUMeshVertex> d_vertices;
        thrust::device_vector<GPUMeshFace> d_faces;
        GPUMeshVertex* d_vertices_ptr;

        GPUPoint d_mesh_min;
        GPUPoint d_mesh_max;

        QSharedPointer<ORNL::SettingsBase> m_sb;
    };
}

#endif // GPU_CROSSSECTION_H
#endif // NVCC_FOUND
