#include "geometry/mesh/advanced/gpu/gpu_auto_orientation.h"

#include "geometry/gpu/gpu_vector_3d.h"
#include "geometry/gpu/gpu_plane.h"

#include <thrust/execution_policy.h>

#include "utilities/mathutils.h"

namespace CUDA
{
    namespace GPUAutoOrientation
    {
        CPU_ONLY
        GPUVolumeCalculator::GPUVolumeCalculator(QVector<ORNL::MeshFace> faces, QVector<ORNL::MeshVertex> vertices, double angle_threshold)
        {
            // Copy faces to GPU
            for(int i =  0, end = faces.size(); i < end; ++i)
            {
                h_faces.push_back(GPUMeshFace(faces[i]));
                h_faces.back().face_index = i;
            }
            d_faces = h_faces;

            // Copy vertices to GPU
            for(int i =  0, end = vertices.size(); i < end; ++i)
                h_vertices.push_back(GPUMeshVertex(vertices[i]));
            d_vertices = h_vertices;

            d_vertices_ptr = thrust::raw_pointer_cast(&d_vertices[0]);

            m_angle_threshold = angle_threshold;
        }

        CPU_ONLY
        double GPUVolumeCalculator::ComputeSupportVolume(QVector3D stacking_axis, ORNL::Plane& plane)
        {
            GPUPlane device_plane(plane);
            GPUVector3D d_stacking_axis(stacking_axis);

            thrust::device_vector<double> d_volumes(h_faces.size());

            // Determine required rotation
            auto picked_vector = plane.normal();
            picked_vector *= -1;
            auto h_quat = ORNL::MathUtils::CreateQuaternion(picked_vector, QVector3D(0, 0, 1));
            GPUQuaternion d_quat(h_quat);

            thrust::transform(thrust::device, d_faces.begin(), d_faces.end(), d_volumes.begin(),
                                                SupportVolume(d_vertices_ptr, d_stacking_axis, m_angle_threshold, device_plane, d_quat));
            double volume = thrust::reduce(thrust::device, d_volumes.begin(), d_volumes.end(),
                                           (double) 0.0, thrust::plus<double>());
            return volume;
        }

        GPU_CPU_CODE
        double GPUVolumeCalculator::AreaOfTriangle(GPUPoint& a, GPUPoint& b, GPUPoint& c)
        {
            GPUVector3D a_vec(a);
            GPUVector3D b_vec(b);
            GPUVector3D c_vec(c);

            GPUVector3D ab = b_vec - a_vec;
            GPUVector3D ac = c_vec - a_vec;

            GPUVector3D cross = ab.cross(ac);
            return 0.5 * cross.length();
        }

        GPU_CPU_CODE
        bool GPUVolumeCalculator::Equals(double a, double b, double epsilon)
        {
            return (a == b) || (std::abs(a - b) < epsilon);
        }

        GPU_CPU_CODE
        double GPUVolumeCalculator::VolumeOfPyramid(GPUPoint& base_1, GPUPoint& base_2, GPUPoint& base_3, GPUPoint& base_4, GPUPoint& tip)
        {
            GPUPoint a = base_2 - base_1;
            GPUPoint b = base_3 - base_1;
            GPUVector3D vec_a(a);
            GPUVector3D vec_b(b);

            GPUVector3D normal = vec_a.cross(vec_b);
            double dx = tip.x() - base_1.x();
            double dy = tip.y() - base_1.y();
            double dz = tip.z() - base_1.z();
            double height = fabs((normal.x() * dx) + (normal.y() * dy) + (normal.z() * dz)) / normal.length();

            double area_base = AreaOfTriangle(base_1, base_2, base_3) + AreaOfTriangle(base_1, base_3, base_4);

            return (area_base * height) / 3.0;
        }

        GPU_CPU_CODE
        double GPUVolumeCalculator::SafeACos(double value)
        {
            if (value <= -1.0)
                return 3.141592654f;
            else if (value >= 1.0)
                return 0;
            else
                return acos(value);
        }

        GPU_CPU_CODE
        double GPUVolumeCalculator::SupportVolume::operator()(GPUMeshFace face)
        {
            auto rotated_face_normal = quat.rotatedVector(face.normal);

            double face_angle = SafeACos(stacking_axis.dot(rotated_face_normal) / (rotated_face_normal.length() * stacking_axis.length()));
            face_angle = face_angle - 1.57079632679489661923; // PI / 2 (literal value here because of CUDA)
            double total_volume = 0.0;

            auto upward = stacking_axis.dot(rotated_face_normal); // This is positive if upward

            if(face_angle > threshold && upward <= 0)
            {
                GPUPoint v0 = vertices[face.vertex_index[0]].location;
                GPUPoint v1 = vertices[face.vertex_index[1]].location;
                GPUPoint v2 = vertices[face.vertex_index[2]].location;

                double v0_dist = fabs(plane.distanceToPoint(v0));
                double v1_dist = fabs(plane.distanceToPoint(v1));
                double v2_dist = fabs(plane.distanceToPoint(v2));

                // Find the min dist of the 3 points to the plane
                GPUPoint tip = v0;
                GPUPoint base_3;
                GPUPoint base_4;
                double min_dist = v0_dist;

                if(v1_dist < min_dist)
                {
                    min_dist = v1_dist;
                    tip = v1;
                }

                if(v2_dist < min_dist)
                {
                    min_dist = v2_dist;
                    tip = v2;
                }

                if(tip == v0)
                {
                    base_3 = v1;
                    base_4 = v2;
                }else if(tip == v1)
                {
                    base_3 = v2;
                    base_4 = v0;
                }else
                {
                    base_3 = v0;
                    base_4 = v1;
                }

                double dist_3 = fabs(plane.distanceToPoint(base_3)) - min_dist;
                double dist_4 = fabs(plane.distanceToPoint(base_4)) - min_dist;

                auto a = (plane.normal().normalized() * dist_3).toGPUPoint();
                GPUPoint base_2 = base_3 + a;
                auto b = (plane.normal().normalized() * dist_4).toGPUPoint();
                GPUPoint base_1 = base_4 + b;
                double tri_area = AreaOfTriangle(base_1, base_2, tip);

                double pyramid_volume;
                int s = 0;
                if(Equals(dist_3, 0, 1) && Equals(dist_4, 0, 1)) // Are these vectors parallel
                {
                    pyramid_volume = 0;
                }
                else if(Equals(dist_3, 0, 1) || Equals(dist_4, 0, 1))
                {
                    double height = dist_3 + dist_4;
                    pyramid_volume = (height * tri_area) / 3.0;
                    s = 1;
                }
                else
                {
                    pyramid_volume = VolumeOfPyramid(base_1, base_2, base_3, base_4, tip); // This is the volume of the upper section (rectangular pyramid)
                    s = 2;
                }

                double prism_volume = tri_area * min_dist; // This is the volume of the lower section (triangular prism)
                total_volume = prism_volume + pyramid_volume;
            }

            return total_volume;
        }
    }
}
