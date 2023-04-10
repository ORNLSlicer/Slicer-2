#ifndef PARAMETERIZATION_H
#define PARAMETERIZATION_H

#include "geometry/mesh/open_mesh.h"

#ifdef NVCC_FOUND
#include "geometry/mesh/advanced/gpu/gpu_face_finder.h"
#endif

namespace ORNL {

    /*!
     * \brief The Parameterization class provides surface conformal mapping methods to print on uneven surfaces.
     */
    class Parameterization
    {
    public:

        //! \brief Default constructor
        Parameterization();

        //! \brief Constructor that performs a UV mapping
        //! \param mesh: a surface mesh
        //! \param dims: size of part
        Parameterization(MeshTypes::SurfaceMesh mesh, Distance3D dims);

        //! \brief converts a UV coordinate to real space and determines if it lies on the surface.
        //! \param point
        //! \return a pair where:
        //!             first is a bool siginifying if the point was on the surface
        //!             second is the point is real space
        QPair<bool, Point> fromUV(MeshTypes::Point_2 point);

        //! \brief gets the polygon of the unpacked surface in 2D
        //! \return a polygon
        Polygon getBoarderPolygon();

        //! \brief gets the dimensions of the unpacked surface
        //! \return the dimensions of the surface mesh
        Distance3D getDimensions();

    private:

        //! \brief builds a conformal mapping using the as rigid as possible method
        void buildConformalUVMap();

        //! \brief calculates the positive area of a 2D triangle
        //! \param p0: first point
        //! \param p1: the second point
        //! \param p2: the third point
        //! \return the area of the triangle
        double calculateAreaofTriangle(MeshTypes::SimpleCartesian::Point_2 p0, MeshTypes::SimpleCartesian::Point_2 p1, MeshTypes::SimpleCartesian::Point_2 p2);

        //! \brief calculates the barycentric coordinates for a point within a triangle
        //! \param triangle: 3 2D vertices
        //! \param point: a 2D point
        //! \return barycentric coordinates
        QVector3D calculateBarycentricCords(QVector<MeshTypes::SimpleCartesian::Point_2> triangle, MeshTypes::SimpleCartesian::Point_2 point);

        //! \brief finds a point in real space given a face/ triangle and barycentric coordinates using the UV map
        //! \param triangle: 3 3D points
        //! \param barycentric_cords: barycentric values
        //! \return a 3D point in real space
        Point calculateRealFromBarycentric(QVector<Point> triangle, QVector3D barycentric_cords);

        //! \brief finds if and what face a 2D point is on within the surface mesh
        //! \param point: a 2D point
        //! \return a pair with: bool for it was on the surface, and the id of the face
        QPair<bool, unsigned long> findFace(MeshTypes::SimpleCartesian::Point_2 point);

        //! \brief UV map that corresponds a vertex in the surface mesh to a UV coordinate
        QMap<MeshTypes::SimpleCartesian::SM_VertexDescriptor, MeshTypes::SimpleCartesian::Point_2> m_uv_map;

        //! \brief the surface mesh
        MeshTypes::SimpleCartesian::SurfaceMesh m_sm;

        //! \brief the unpacked 2D representation of the mesh
        Polygon m_boarder_polygon;

        //! \brief the dims of the unpacked 2D representation of the mesh
        Distance3D m_dimensions;

        #ifdef NVCC_FOUND
        //! \brief the gpu finder
        QSharedPointer<CUDA::GPUFaceFinder> m_gpu;
        #endif
    };

}

#endif // PARAMETERIZATION_H
