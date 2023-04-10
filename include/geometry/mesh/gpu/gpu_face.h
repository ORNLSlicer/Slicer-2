#ifdef NVCC_FOUND

#ifndef GPU_FACE
#define GPU_FACE

#include "geometry/mesh/mesh_face.h"
#include "geometry/mesh/mesh_vertex.h"
#include "geometry/gpu/gpu_point.h"

namespace CUDA
{
    /*!
     * \class GPUFace
     * \brief a face that exists on the GPU
     * \note this class is only used for conformal mapping \see GPUMeshFace for replacment for a MeshFace
     */
    class GPUFace
    {
    public:
        GPU_CPU_CODE
        //! \brief Constructor
        GPUFace() = default;

        GPU_CPU_CODE
        //! \brief Copy Constructor
        //! \brief a face to copy
        GPUFace(const GPUFace& face);

        CPU_ONLY
        //! Constructor
        //! \param a: the first point
        //! \param b: the second point
        //! \param c: the third point
        //! \param face_id: the id of the face in the mesh
        GPUFace(ORNL::Point a, ORNL::Point b, ORNL::Point c, unsigned long face_id);

        CPU_ONLY
        GPUFace(const ORNL::MeshFace& face, const QVector<ORNL::MeshVertex> &vertices, int index);

        GPU_CPU_CODE
        //! \brief get the first point
        //! \return a GPU point
        GPUPoint& a();

        GPU_CPU_CODE
        //! \brief get the first point
        //! \return a GPU point
        const GPUPoint a() const;

        GPU_CPU_CODE
        //! \brief sets the first point
        //! \param GPU point
        void a(GPUPoint& a);

        GPU_CPU_CODE
        //! \brief get the second point
        //! \return a GPU point
        GPUPoint& b();

        GPU_CPU_CODE
        //! \brief get the second point
        //! \return a GPU point
        const GPUPoint b() const;

        GPU_CPU_CODE
        //! \brief sets the second point
        //! \param GPU point
        void b(GPUPoint& b);

        GPU_CPU_CODE
        //! \brief get the third point
        //! \return a GPU point
        GPUPoint& c();

        GPU_CPU_CODE
        //! \brief get the third point
        //! \return a GPU point
        const GPUPoint c() const;

        GPU_CPU_CODE
        //! \brief sets the third point
        //! \param GPU point
        void c(GPUPoint& c);

        GPU_CPU_CODE
        //! \brief get the face ID
        //! \return a face id
        unsigned long& faceID();

        GPU_CPU_CODE
        //! \brief get the face ID
        //! \return a face id
        const unsigned long faceID() const;

        GPU_CPU_CODE
        //! \brief sets the face ID
        //! \param a face ID
        void faceID(unsigned long& face_id);

    private:
        //! \brief points that make of the face's triangle
        GPUPoint m_a;
        GPUPoint m_b;
        GPUPoint m_c;

        // \brief the CGAL id of this face in the mesh
        unsigned long m_face_id;
    };
}

#endif // GPU_FACE
#endif // NVCC_FOUND
