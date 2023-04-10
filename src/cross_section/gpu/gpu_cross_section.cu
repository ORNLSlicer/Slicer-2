// Main Module
#include "cross_section/gpu/gpu_cross_section.h"

// Local
#include "cross_section/cross_section_object.h"
#include "cross_section/cross_section_segment.h"
#include "utilities/mathutils.h"
#include "managers/gpu_manager.h"

#include <thrust/transform.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <cstdio>

namespace CUDA
{
    CPU_ONLY
    GPUCrossSectioner::GPUCrossSectioner(QVector<ORNL::MeshVertex>& vertices, QVector<ORNL::MeshFace>& faces,
                                         ORNL::Point mesh_min, ORNL::Point mesh_max,
                                         QSharedPointer<ORNL::SettingsBase> sb)
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

        d_mesh_min = GPUPoint(mesh_min);
        d_mesh_max = GPUPoint(mesh_max);

        m_sb = sb;
        d_vertices_ptr = thrust::raw_pointer_cast(&d_vertices[0]);
    }

    CPU_ONLY
    GPUCrossSectioner::~GPUCrossSectioner(){}

    CPU_ONLY
    ORNL::PolygonList GPUCrossSectioner::doCrossSectionGPU(ORNL::Plane &slicing_plane, ORNL::Point &shift)
    {
        // Device variables;
        GPUPlane device_plane = GPUPlane(slicing_plane);
        GPUPoint device_shift = GPUPoint(findSlicingPlaneMidPointGPU(d_mesh_min, d_mesh_max, device_plane));
        shift = static_cast<ORNL::Point>(device_shift);
        GPUQuaternion device_rotation = GPUQuaternion(ORNL::MathUtils::CreateQuaternion(slicing_plane.normal(), QVector3D(0, 0, 1)));

        thrust::device_vector<GPUCrossSectionSegment> d_segments(h_faces.size());
        thrust::transform(thrust::device, d_faces.begin(), d_faces.end(), d_segments.begin(),
                          GPU_Intersection(device_plane, d_vertices_ptr, device_shift, device_rotation));
        int intersections = thrust::count_if(thrust::device, d_segments.begin(), d_segments.end(), [] GPU_CPU_CODE (const GPUCrossSectionSegment& segment){
             return segment.valid;
        });
        thrust::device_vector<GPUCrossSectionSegment> d_results(intersections);
        thrust::copy_if(thrust::device, d_segments.begin(), d_segments.end(), d_results.begin(), [] GPU_CPU_CODE (const GPUCrossSectionSegment& segment){
            return segment.valid;
        });

        thrust::host_vector<GPUCrossSectionSegment> h_results = d_results; // Copy results to host

        ORNL::CrossSectionObject cs(m_sb);
        //add segment to cross-section
        for(int i = 0; i < h_results.size(); ++i)
        {
            auto segment = h_results[i];
            if(segment.valid)
            {
                ORNL::CrossSectionSegment new_seg = segment.toCrossSectionSegment();
                cs.insertFaceToSegment(segment.face_index, cs.segments().size());
                cs.addSegment(new_seg);
            }
        }
        return cs.makePolygons();
    }

    GPU_CPU_CODE
    GPUPoint GPUCrossSectioner::findIntersectionGPU(const GPUPoint &vertex0, const GPUPoint &vertex1, const GPUPlane& plane)
    {
        //slope of the line
        double dx = vertex1.x() - vertex0.x();
        double dy = vertex1.y() - vertex0.y();
        double dz = vertex1.z() - vertex0.z();

        // diff between plane and vertex0
        double plane_dx = plane.point().x() - vertex0.x();
        double plane_dy = plane.point().y() - vertex0.y();
        double plane_dz = plane.point().z() - vertex0.z();

        double numerator = (plane.normal().x() * plane_dx)
                         + (plane.normal().y() * plane_dy)
                         + (plane.normal().z() * plane_dz);

        double denominator = (plane.normal().x() * dx)
                           + (plane.normal().y() * dy)
                           + (plane.normal().z() * dz);

        double t = numerator / denominator;

        GPUPoint intersection_point;
        intersection_point.x(dx * t + vertex0.x());
        intersection_point.y(dy * t + vertex0.y());
        intersection_point.z(dz * t + vertex0.z());

        return intersection_point;
    }

    GPU_CPU_CODE
    GPUPoint GPUCrossSectioner::rotatePointGPU(GPUPoint point_to_rotate, GPUVector3D& shift_amount, GPUQuaternion& q)
    {
        GPUVector3D point_vec(point_to_rotate);

        //shift to rotate around origin
        point_vec = point_vec - shift_amount;

        GPUQuaternion p(0, point_vec.x(), point_vec.y(), point_vec.z());
        GPUQuaternion temp = GPUQuaternion::multiply(q, p);
        GPUQuaternion rotated = GPUQuaternion::multiply(temp, q.conjugated());

        GPUPoint result(rotated.x() + shift_amount.x(), rotated.y() + shift_amount.y(), rotated.z());

        return result; // will un-rotate and un-shift later, just before gcode writing
    }

    CPU_ONLY
    GPUPoint GPUCrossSectioner::findSlicingPlaneMidPointGPU(GPUPoint &mesh_min, GPUPoint &mesh_max, GPUPlane& plane)
    {
        GPUPoint center = (mesh_min + mesh_max) / 2 ; //center of the bounding box
        ORNL::Distance max_dimension = mesh_min.distance(mesh_max);
        GPUVector3D plane_normal = plane.normal().normalized() * max_dimension();
        auto p = plane_normal.toGPUPoint();

        //calculate a 'center_line' that is parallel to the slicing plane normal and runs through the center
        //end points are arbitrary as long as they are outside the bounding box
        GPUPoint center_line_min(center);
        center_line_min = center_line_min - p;
        GPUPoint center_line_max(center);
        center_line_max = center_line_max + p;

        //the plane midpoint is the intersection of the center_normal line and the plane
        center = findIntersectionGPU(center_line_min, center_line_max, plane);
        return center;
    }

    GPU_CPU_CODE
    GPUCrossSectionSegment GPUCrossSectioner::GPU_Intersection::operator()(GPUMeshFace face)
    {
        // Skip ignored faces
        if(face.ignore)
            return GPUCrossSectionSegment();

        const GPUMeshVertex& v0 = vertices[face.vertex_index[0]];
        const GPUMeshVertex& v1 = vertices[face.vertex_index[1]];
        const GPUMeshVertex& v2 = vertices[face.vertex_index[2]];

        //the points of the triangle
        GPUPoint p0 = v0.location;
        GPUPoint p1 = v1.location;
        GPUPoint p2 = v2.location;

        GPUCrossSectionSegment segment;
        int end_edge_idx = -1;

        /*
           Each point is evaluated in the plane equation to
           determine which side of the plane the point is on.
           If pt_eval > 0, it is above the plane
                      < 0, below the plane
                      = 0, on the plane
        */
        double p0_eval = plane.evaluatePoint(p0);
        double p1_eval = plane.evaluatePoint(p1);
        double p2_eval = plane.evaluatePoint(p2);

        bool intersection = false;
        //Check each orientation of points & plane that intersect
        if(p0_eval < 0 && p1_eval >= 0 && p2_eval >= 0)
        {
            // p2   p1
            // --------
            //   p0
            end_edge_idx = 0;
            segment.start = findIntersectionGPU(p0, p2, plane);
            segment.end   = findIntersectionGPU(p0, p1, plane);
            if(p1_eval == 0)
            {
                segment.end_vertex = v1;
            }
            intersection = true;
        }
        else if(p0_eval > 0 && p1_eval < 0 && p2_eval < 0)
        {
            //   p0
            // --------
            // p1  p2

            end_edge_idx = 2;
            segment.start = findIntersectionGPU(p1, p0, plane);
            segment.end   = findIntersectionGPU(p2, p0, plane);
            intersection = true;
        }
        else if(p1_eval < 0 && p0_eval >= 0 && p2_eval >= 0)
        {
            // p0   p2
            // --------
            //   p1
            end_edge_idx = 1;
            segment.start = findIntersectionGPU(p1, p0, plane);
            segment.end   = findIntersectionGPU(p1, p2, plane);
            if(p2_eval == 0)
            {
                segment.end_vertex = v2;
            }
            intersection = true;
        }
        else if(p1_eval > 0 && p0_eval < 0 && p2_eval < 0)
        {
            //   p1
            // --------
            // p2  p0
            end_edge_idx = 0;
            segment.start = findIntersectionGPU(p2, p1, plane);
            segment.end   = findIntersectionGPU(p0, p1, plane);
            intersection = true;
        }
        else if(p2_eval < 0 && p1_eval >= 0 && p0_eval >= 0)
        {
            // p1   p0
            // --------
            //   p2
            end_edge_idx = 2;
            segment.start = findIntersectionGPU(p2, p1, plane);
            segment.end   = findIntersectionGPU(p2, p0, plane);
            if(p0_eval == 0)
            {
                segment.end_vertex = v0;
            }
            intersection = true;
        }
        else if(p2_eval > 0 && p1_eval < 0 && p0_eval < 0)
        {
            //   p2
            // --------
            // p0  p1
            end_edge_idx = 1;
            segment.start = findIntersectionGPU(p0, p2, plane);
            segment.end   = findIntersectionGPU(p1, p2, plane);
            intersection = true;
        }
        else
        {
            // Not all cases create a segment, because a point of a face
            // could create just a dot, and two touching faces
            //  on the slice would create two segments
        }

        /*
         * Polygon operations that follow cross sectioning are done in 2 dimensions.
         * To allow that, shift and rotate each segment so that the cross-section will be flat in 2D
         * Polymer_slicer save shift and rotation amount to layer so that this is un-done before gcode writing
        */

        if(intersection)
        {
            //define rotation using the direction of the slicing plane
            segment.valid = true;
            auto s = GPUVector3D(shift);
            segment.start = rotatePointGPU(segment.start, s, rotation);
            segment.end   = rotatePointGPU(segment.end, s, rotation);

            segment.face_index         = face.face_index;
            segment.end_other_face_idx = face.connected_face_index[end_edge_idx];
            segment.added_to_polygon   = false;
            segment.normal             = face.normal;
        }else
        {
            segment.valid = false;
        }
        return segment;
    }
}
