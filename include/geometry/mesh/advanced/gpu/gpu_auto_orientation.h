#ifdef NVCC_FOUND
#ifndef GPUAUTOORIENTATION_H
#define GPUAUTOORIENTATION_H

#include "geometry/plane.h"
#include "geometry/mesh/mesh_face.h"
#include "geometry/mesh/mesh_vertex.h"

#include "geometry/mesh/gpu/gpu_mesh_vertex.h"
#include "geometry/mesh/gpu/gpu_mesh_face.h"
#include "geometry/gpu/gpu_point.h"
#include "geometry/gpu/gpu_plane.h"
#include "geometry/gpu/gpu_quaternion.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace CUDA
{
    namespace GPUAutoOrientation
    {
        //! \struct CandidateOrientation
        //! \brief a potential orientation for an object to be printed in
        //! \note this is a GPU safe struct
        struct CandidateOrientation
        {
        public:
            //! \brief Constructor
            //! \param _plane a CandidateOrientation is defined by a plane (derived from a face on the convex hull)
            GPU_CPU_CODE
            CandidateOrientation(GPUPlane _plane) : plane(_plane) {}

            GPUPlane plane;
            double area = 0.0;
            double support_volume = 0.0;
        };

        //! \class GPUVolumeCalculator
        //! \brief a GPU based estimate of support volume
        class GPUVolumeCalculator
        {
        public:
            //! \brief Constructor
            //! \param faces the faces on the mesh
            //! \param vertices the vertices in the mesh
            //! \param angle_threshold the critical angle that does not need supports
            //! \note This call is ONLY CPU safe
            CPU_ONLY
            GPUVolumeCalculator(QVector<ORNL::MeshFace> faces, QVector<ORNL::MeshVertex> vertices, double angle_threshold);

            //! \brief computes the support volume for a given plane/ orientation
            //! \param stacking_axis the axis to "stack" or slice along
            //! \param plane the plane that defines what the "bottom" of the mesh will be
            //! \return the support volume in cubic microns
            //! \note This call is ONLY CPU safe
            CPU_ONLY
            double ComputeSupportVolume(QVector3D stacking_axis, ORNL::Plane& plane);


            //! \struct SupportVolume
            //! \brief Operator call to compute the support a specific face will required
            //! \note this should only be used with thrust::transform
            struct SupportVolume
            {
                GPUVector3D stacking_axis;
                double threshold;
                GPUMeshVertex* vertices;
                GPUPlane plane;
                GPUQuaternion quat;

                //! \brief Constructor
                //! \param _vertices the vertices in the mesh
                //! \param _stacking_axis the stacking axis to calculate based on
                //! \param _threshold the angle threshold
                //! \param _plane the orientation
                //! \param _quat the quaternion that represents the rotation of the part
                //! \note this should only be used with thrust::transform
                SupportVolume(GPUMeshVertex* _vertices, GPUVector3D _stacking_axis, double _threshold, GPUPlane _plane, GPUQuaternion _quat)
                                : vertices(_vertices), stacking_axis(_stacking_axis), threshold(_threshold), plane(_plane), quat(_quat) {}

                //! \brief computes the support volume for a speific face
                //! \param face the face
                //! \return the support volume required to support this face
                //! \note this should only be used with thrust::transform
                GPU_CPU_CODE
                double operator()(GPUMeshFace face);
            };

        private:
            //! \brief Computes area of a triangle in 3D
            //! \param a point
            //! \param b point
            //! \param c point
            //! \return the area in squared microns
            GPU_CPU_CODE
            static double AreaOfTriangle(GPUPoint& a, GPUPoint& b, GPUPoint& c);

            //! \brief fuzzy compare with custom threshold on GPU
            //! \param a the first value
            //! \param b the second value
            //! \param epsilon the error threshold
            //! \return if the values equal each other
            GPU_CPU_CODE
            static bool Equals(double a, double b, double epsilon);

            //! \brief computes the colume of rectangular pyramid from points in counterclockwise order
            //! \param base_1 base point
            //! \param base_2 base point
            //! \param base_3 base point
            //! \param base_4 base point
            //! \param tip the top/ top of the pyramid
            //! \return the volume of the pyramid
            GPU_CPU_CODE
            static double VolumeOfPyramid(GPUPoint& base_1, GPUPoint& base_2, GPUPoint& base_3, GPUPoint& base_4, GPUPoint& tip);

            //! \brief a floating point error save version of acos(double value)
            //! \param value on the interval [-1, 1]
            //! \return arc cosine on the interval [0, 2 * pi]
            //! \note this function is GPU and CPU safe
            GPU_CPU_CODE
            static double SafeACos(double value);

            thrust::host_vector<GPUMeshFace> h_faces;
            thrust::device_vector<GPUMeshFace> d_faces;

            thrust::host_vector<GPUMeshVertex> h_vertices;
            thrust::device_vector<GPUMeshVertex> d_vertices;
            GPUMeshVertex* d_vertices_ptr;
            double m_angle_threshold;
        };
    }
}

#endif // GPUAUTOORIENTATION_H
#endif // NVCC_FOUND
