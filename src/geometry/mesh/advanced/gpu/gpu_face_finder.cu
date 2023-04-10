// Main Module
#include "geometry/mesh/advanced/gpu/gpu_face_finder.h"

// Locals
#include "geometry/point.h"

// Thrust Lib
#include <thrust/find.h>
#include <thrust/execution_policy.h>

namespace CUDA
{
    CPU_ONLY
    GPUFaceFinder::GPUFaceFinder(QVector<GPUFace>& host_faces)
    {
        cudaMalloc((void **) &m_face_ptr, host_faces.size() * sizeof(GPUFace));

        cudaMemcpy(m_face_ptr, host_faces.begin(), sizeof(GPUFace) * host_faces.size(), cudaMemcpyHostToDevice);

        m_num_faces = host_faces.size();
    }

    CPU_ONLY
    GPUFaceFinder::~GPUFaceFinder()
    {
        cudaFree(m_face_ptr);
    }

    CPU_ONLY
    QPair<bool, unsigned long> GPUFaceFinder::findFace(ORNL::Point point)
    {
        GPUPoint gpu_point(point);

        GPUFace * face = thrust::find_if(thrust::device, m_face_ptr, m_face_ptr + m_num_faces, [gpu_point] GPU_ONLY (auto face) {
            // Calculate barycentric
            float tri_area = fabs(((face.a().x() * ( face.b().y() - face.c().y()) +
                                    face.b().x() * (face.c().y() - face.a().y()) +
                                    face.c().x() * (face.a().y() -  face.b().y())) / 2.0));

            float u =  fabs(((gpu_point.x() * (face.b().y() - face.c().y()) +
                              face.b().x() * (face.c().y() - gpu_point.y()) +
                              face.c().x() * (gpu_point.y() - face.b().y())) / 2.0))
                       / tri_area;

            float v = fabs(((face.a().x() * (gpu_point.y() - face.c().y()) +
                             gpu_point.x() * (face.c().y() - face.a().y()) +
                             face.c().x() * (face.a().y() - gpu_point.y())) / 2.0))
                      / tri_area;

            float w = fabs(((face.a().x() * (face.b().y() - gpu_point.y()) +
                             face.b().x() * (gpu_point.y() - face.a().y()) +
                             gpu_point.x() * (face.a().y() - face.b().y())) / 2.0))
                      / tri_area;

            // If the point on the face?
            return ((u + v + w) <= 1.001);
        });

        bool found = face != (m_face_ptr + m_num_faces);

        if(found)
        {
            auto * host_face = new GPUFace;
            cudaMemcpy(host_face, face, sizeof(GPUFace), cudaMemcpyDeviceToHost);

            return QPair<bool, unsigned long>(found, host_face->faceID());
        }
        else
            return QPair<bool, unsigned long>(found, 0);
    }
}
