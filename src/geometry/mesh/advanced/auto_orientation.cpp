#include "geometry/mesh/advanced/auto_orientation.h"

// CGAL
#include <CGAL/Polyhedron_3.h>
#include <CGAL/convex_hull_3.h>

// Local
#include "geometry/mesh/advanced/mesh_types.h"
#include "managers/settings/settings_manager.h"
#include "utilities/mathutils.h"

// CUDA
#ifdef NVCC_FOUND
#include "geometry/mesh/advanced/gpu/gpu_auto_orientation.h"
#include "cross_section/gpu/gpu_cross_section.h"
#endif

namespace ORNL
{
    AutoOrientation::AutoOrientation(ClosedMesh mesh, Plane slicing_plane)
    {
        m_mesh = mesh;
    }

    void AutoOrientation::orient()
    {
        #ifdef NVCC_FOUND
        // Compute convex hull to help find valid set of faces
        auto convex_hull = computeConvexHull(m_mesh);

        auto faces = m_mesh.faces();
        auto vertices = m_mesh.vertices();
        auto mesh_min = m_mesh.min();
        auto mesh_max = m_mesh.max();

        // Extract unique faces from convex hull
        // Note: a set of faces might not actually be on the mesh, this is why we keep track if a CandidateOrientation is only on the hull
        // Only valid faces to rotate to will be on the convex hull
        auto faces_on_hull = convex_hull.faces();
        auto points_in_hull = convex_hull.vertices();
        for(auto face : faces_on_hull)
        {
            Plane plane(Point(points_in_hull[face.vertex_index[0]].location), face.normal);

            bool match_found = false;

            int i = 0;
            for(int end = m_candidate_orientations.size(); i < end; ++i)
            {
                if(m_candidate_orientations[i].plane == plane) // Do their planes match
                {
                    match_found = true;
                    break;
                }
            }

            if(!match_found)
                m_candidate_orientations.append(CandidateOrientation(plane));
        }

        QSharedPointer<SettingsBase> global_sb = GSM->getGlobal();
        Angle stacking_pitch = global_sb->setting<Angle>(Constants::ProfileSettings::SlicingAngle::kStackingDirectionPitch);
        Angle stacking_yaw   = global_sb->setting<Angle>(Constants::ProfileSettings::SlicingAngle::kStackingDirectionYaw);
        Angle stacking_roll  = global_sb->setting<Angle>(Constants::ProfileSettings::SlicingAngle::kStackingDirectionRoll);

        // Build a normal vector using the quaternion
        QVector3D settings_build_vector(0, 0, 1);
        QQuaternion quaternion = MathUtils::CreateQuaternion(stacking_pitch, stacking_yaw, stacking_roll);
        settings_build_vector = quaternion.rotatedVector(settings_build_vector).normalized();

        const int num_slice = 25;

        if(settings_build_vector != QVector3D(0, 0, 1))
        {
            double rad_slice = (2.0f * M_PI) / num_slice;
            QQuaternion slice_rot = MathUtils::CreateQuaternion(Angle(0.0), Angle(0.0), Angle(rad_slice));

            QVector<CandidateOrientation> new_orientations;

            for(auto orientation : m_candidate_orientations)
            {
                auto build_vector = settings_build_vector;
                for(int i = 0; i < num_slice; ++i)
                {
                    build_vector = slice_rot.rotatedVector(build_vector).normalized();

                    new_orientations.push_back(CandidateOrientation(orientation.plane, build_vector));
                }

            }

            m_candidate_orientations = new_orientations;
        }

        auto size = m_candidate_orientations.size();

        auto layer_height = GSM->getGlobal()->setting<Distance>(Constants::ProfileSettings::Layer::kLayerHeight);
        auto settings = GSM->getGlobal();
        settings->setSetting(Constants::ProfileSettings::SpecialModes::kSmoothing, false); // Disable smoothing, we don't need it here

        CUDA::GPUCrossSectioner *cs = new CUDA::GPUCrossSectioner(vertices, faces, mesh_min, mesh_max, settings);
        // Take cross-sections to find area
        #pragma omp parallel for
        for(int i = 0; i < size; i++)
        {
            auto& orientation = m_candidate_orientations[i];
            Point shift;
            auto plane = orientation.plane;
            plane.normal(plane.normal() * -1);
            plane.shiftAlongNormal(layer_height() / 2.0);

            if(plane.normal() == QVector3D(0,0,-1))
                plane.normal(plane.normal() * -1);

            auto cross_section = cs->doCrossSectionGPU(plane, shift);
            double total_area = 0.0;
            auto num_polygons = cross_section.size();
            Q_ASSERT(num_polygons > 0);

            #pragma omp parallel for shared(cross_section) reduction(+: total_area)
            for(int j = 0; j < num_polygons; j++)
            {
                auto& polygon = cross_section[j];
                total_area += polygon.area()();
            }
            orientation.area = total_area;
        }
        delete cs;


        // Compute support volume for each orientation
        Angle threshold = GSM->getGlobal()->setting<Angle>(Constants::ProfileSettings::Support::kThresholdAngle);
        CUDA::GPUAutoOrientation::GPUVolumeCalculator * volume_calc = new CUDA::GPUAutoOrientation::GPUVolumeCalculator(faces, vertices, threshold());

        #pragma omp parallel for
        for(int i = 0; i < size; i++)
        {
            auto& orientation = m_candidate_orientations[i];
            Plane plane = orientation.plane;
            QVector3D build_vector = orientation.build_vector;

            double volume = volume_calc->ComputeSupportVolume(build_vector, plane);

            orientation.support_volume = Volume(volume);
        }
#endif
    }

    QVector<AutoOrientation::CandidateOrientation> AutoOrientation::getResults()
    {
        return m_candidate_orientations;
    }

    AutoOrientation::CandidateOrientation AutoOrientation::getOrientationForValues(double area, double volume)
    {
        for(auto orentation : m_candidate_orientations)
        {
            if(qFuzzyCompare(orentation.area(), area) && qFuzzyCompare(orentation.support_volume(), volume))
                return orentation;
        }
        return CandidateOrientation(Plane());
    }

    AutoOrientation::CandidateOrientation AutoOrientation::getRecommendedOrientation()
    {
        double max_contour = 0.0;
        double min_contour = std::numeric_limits<double>::max();
        double max_support = 0.0;
        double min_support = std::numeric_limits<double>::max();

        for(auto& orentation : m_candidate_orientations)
        {
            if(orentation.area > max_contour)
            {
                max_contour = orentation.area();
            }

            if(orentation.area < min_contour)
            {
                min_contour = orentation.area();
            }

            if(orentation.support_volume > max_support)
            {
                max_support = orentation.support_volume();
            }

            if(orentation.support_volume < min_support)
            {
                min_support = orentation.support_volume();
            }
        }

        double percentage = 0.03;

        double contour_range = max_contour - min_contour;
        double support_range = max_support - min_support;

        std::sort(m_candidate_orientations.begin(), m_candidate_orientations.end(), [](const CandidateOrientation& lhs,
                                                                                       const CandidateOrientation& rhs)
        {
            return lhs.support_volume < rhs.support_volume;
        });

        auto best = m_candidate_orientations[0];

        if(!qFuzzyCompare(best.area(), max_contour))
        {
            for(int i = 1; i < m_candidate_orientations.size(); ++i)
            {
                auto target = m_candidate_orientations[i];

                if(((target.support_volume - best.support_volume)() / support_range) <= percentage)
                {
                    if(((target.area - best.area)() / contour_range) >= percentage)
                        best = target;
                }
                else
                    break;
            }
        }

        return best;



//        // Find max contour
//        auto max_contour = std::max_element(m_candidate_orientations.begin(), m_candidate_orientations.end(), [](const CandidateOrientation& lhs,
//                                                                                                                 const CandidateOrientation& rhs)
//        {
//            return lhs.area < rhs.area;
//        });

//        // Find min support
//        auto max_support = std::min_element(m_candidate_orientations.begin(), m_candidate_orientations.end(), [](const CandidateOrientation& lhs,
//                                                                                                                 const CandidateOrientation& rhs)
//        {
//            return lhs.support_volume < rhs.support_volume;
//        });

//        if(max_contour->plane == max_support->plane)
//            return max_contour;
//        else
//            return nullptr;

    }

    QQuaternion AutoOrientation::GetRotationForOrientation(CandidateOrientation orientation)
    {
        // // Fetch default axis from settings
        // QSharedPointer<SettingsBase> global_sb = GSM->getGlobal();
        // Angle stacking_pitch = global_sb->setting<Angle>(Constants::ProfileSettings::SlicingAngle::kStackingDirectionPitch);
        // Angle stacking_yaw   = global_sb->setting<Angle>(Constants::ProfileSettings::SlicingAngle::kStackingDirectionYaw);
        // Angle stacking_roll  = global_sb->setting<Angle>(Constants::ProfileSettings::SlicingAngle::kStackingDirectionRoll);

        // // Build a normal vector using the quaternion
        // QVector3D settings_build_vector(0, 0, 1);
        // QQuaternion quaternion = MathUtils::CreateQuaternion(stacking_pitch, stacking_yaw, stacking_roll);
        // settings_build_vector = quaternion.rotatedVector(settings_build_vector).normalized();
        // settings_build_vector.setZ(0.0);

        // auto rotation_vector = orientation.build_vector;
        // rotation_vector.setZ(0.0);

        // auto angle = MathUtils::signedInternalAngle(Point(settings_build_vector), Point(0, 0, 0), Point(rotation_vector));

        // auto axis_angle = MathUtils::AxisAngleToQuat(-orientation.plane.normal(), angle());

        // // Determine required rotation for the face selected
        // auto picked_vector = orientation.plane.normal();
        // picked_vector *= -1;
        // return MathUtils::CreateQuaternion(picked_vector, QVector3D(0, 0, 1)) * axis_angle;
        return QQuaternion();
    }

    ClosedMesh AutoOrientation::computeConvexHull(ClosedMesh mesh)
    {
        auto polyhedron = mesh.polyhedron();
        MeshTypes::Polyhedron hull;
        CGAL::convex_hull_3(polyhedron.points_begin(), polyhedron.points_end(), hull);
        ClosedMesh convex_hull(hull);
        return convex_hull;
    }
}
