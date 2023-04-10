#include "geometry/mesh/gpu/gpu_face.h"

namespace CUDA
{
    GPU_CPU_CODE
    GPUFace::GPUFace(const GPUFace &face)
    {
        m_a = face.a();
        m_b = face.b();
        m_c = face.c();
        m_face_id = face.faceID();
    }

    CPU_ONLY
    CUDA::GPUFace::GPUFace(ORNL::Point a, ORNL::Point b, ORNL::Point c, unsigned long face_id)
    : m_a(a), m_b(b), m_c(c), m_face_id(face_id)
    {

    }

    GPU_CPU_CODE
    GPUFace::GPUFace(const ORNL::MeshFace& face, const QVector<ORNL::MeshVertex> &vertices, int index)
    {
        m_a = GPUPoint(vertices[face.vertex_index[0]].location.x(), vertices[face.vertex_index[0]].location.y(), vertices[face.vertex_index[0]].location.z());
        m_b = GPUPoint(vertices[face.vertex_index[1]].location.x(), vertices[face.vertex_index[1]].location.y(), vertices[face.vertex_index[1]].location.z());
        m_c = GPUPoint(vertices[face.vertex_index[2]].location.x(), vertices[face.vertex_index[2]].location.y(), vertices[face.vertex_index[2]].location.z());
        m_face_id = index;
    }

    GPU_CPU_CODE
    GPUPoint& GPUFace::a()
    {
        return m_a;
    }

    GPU_CPU_CODE
    const GPUPoint GPUFace::a() const
    {
        return m_a;
    }

    GPU_CPU_CODE
    void GPUFace::a(GPUPoint &a)
    {
        m_a = a;
    }

    GPU_CPU_CODE
    GPUPoint &GPUFace::b()
    {
        return m_b;
    }

    GPU_CPU_CODE
    const GPUPoint GPUFace::b() const
    {
        return m_b;
    }

    GPU_CPU_CODE
    void GPUFace::b(GPUPoint &b)
    {
        m_b = b;
    }

    GPU_CPU_CODE
    GPUPoint &GPUFace::c()
    {
        return m_c;
    }

    GPU_CPU_CODE
    const GPUPoint GPUFace::c() const
    {
        return m_c;
    }

    GPU_CPU_CODE
    void GPUFace::c(GPUPoint &c)
    {
        m_c = c;
    }

    GPU_CPU_CODE
    unsigned long &GPUFace::faceID()
    {
        return m_face_id;
    }

    GPU_CPU_CODE
    const unsigned long GPUFace::faceID() const
    {
        return m_face_id;
    }

    GPU_CPU_CODE
    void GPUFace::faceID(unsigned long &face_id)
    {
        m_face_id = face_id;
    }
}


