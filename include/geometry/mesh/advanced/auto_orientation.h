#ifndef AUTOORIENTATION_H
#define AUTOORIENTATION_H

#include "geometry/mesh/mesh_base.h"
#include "geometry/mesh/closed_mesh.h"

namespace ORNL
{
    //! \class AutoOrientation
    //! \brief An algorithm to automatically orient parts
    class AutoOrientation
    {
    public:
        //! \struct CandidateOrientation
        struct CandidateOrientation
        {
        public:
            //! \brief Constructor
            //! \param _plane the plane that defines the orientation
            //! \param _build_vector the stacking dir of the layers
            CandidateOrientation(Plane _plane, QVector3D _build_vector = QVector3D(0, 0, 1)) : plane(_plane), build_vector(_build_vector) {}

            Plane plane;
            Area area = 0.0;
            Volume support_volume = 0.0;
            QVector3D build_vector = QVector3D(0, 0, 1);
        };

        //! \brief Constructor
        //! \param mesh the mesh
        //! \param slicing_plane the plane
        AutoOrientation(ClosedMesh mesh, Plane slicing_plane);

        //! \brief orients the parts using the algorithm
        //! \pre Slicer 2 was compiled with GPU support and there is GPU in the system
        //! \return the rotation to place the part at the optimal rotation
        void orient();

        //! \brief gets the results of the algorithm
        //! \pre orient() has been called
        //! \return candidate orientations
        QVector<CandidateOrientation> getResults();

        //! \brief looks up a candidate orientation by area and volume
        //! \pre orient() has been called
        //! \param area the area (in micron)
        //! \param volume the volume (in micron)
        //! \return the candidate orientation
        CandidateOrientation getOrientationForValues(double area, double volume);

        //! \brief gets the recommended orientation
        //! \note a recommended orientation is considered one that is both the global min of support volume and global max of contour
        //! \pre orient() has been called
        //! \return the recommended orientation
        CandidateOrientation getRecommendedOrientation();

        //! \brief determines the rotation required to rotate an object from it default pos to that of a candidate orientation
        //! \param orentation where we want to rotate to
        //! \return the rotation required
        static QQuaternion GetRotationForOrientation(CandidateOrientation orentation);

    private:
        //! \brief computes the convex hull of the mesh
        //! \param mesh input mesh
        //! \return convex hull as a mesh
        ClosedMesh computeConvexHull(ClosedMesh mesh);

        //! \brief the mesh to run on
        ClosedMesh m_mesh;

        //! \brief list of valid options
        QVector<CandidateOrientation> m_candidate_orientations;
    };
}

#endif // AUTOORIENTATION_H
