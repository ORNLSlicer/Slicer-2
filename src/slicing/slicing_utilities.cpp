#include "slicing/slicing_utilities.h"

// Locals
#include "step/layer/island/island_base.h"
#include "step/layer/layer.h"
#include "step/layer/scan_layer.h"
#include "utilities/mathutils.h"
#include "cross_section/cross_section.h"
#include "step/step.h"

namespace ORNL
{

    QVector<QSharedPointer<MeshBase>> SlicingUtilities::GetMeshesByType(QMap<QString, QSharedPointer<Part> > parts, MeshType mt)
    {
        QVector<QSharedPointer<MeshBase>> meshes;
        for(QSharedPointer<Part> part : parts)
        {
            if(part->rootMesh()->type() == mt)
                meshes.push_back(part->rootMesh());
        }
        return meshes;
    }

    QVector<QSharedPointer<Part>> SlicingUtilities::GetPartsByType(QMap<QString, QSharedPointer<Part>> parts, MeshType mt)
    {
        QVector<QSharedPointer<Part>> found_parts;
        for(QSharedPointer<Part> part : parts)
        {
            if(part->rootMesh()->type() == mt)
                found_parts.push_back(part);
        }
        return found_parts;
    }

    void SlicingUtilities::ClipMesh(QSharedPointer<MeshBase> mesh, QVector<QSharedPointer<MeshBase>> clippers)
    {
        for (QSharedPointer<MeshBase> clipper : clippers)
        {
            auto closed_clipper = dynamic_cast<ClosedMesh*>(clipper.get());
            auto closed_mesh = dynamic_cast<ClosedMesh*>(mesh.get());
            if(closed_clipper != nullptr && closed_mesh != nullptr)
                closed_mesh->difference(*closed_clipper);
        }
    }

    void SlicingUtilities::IntersectMesh(QSharedPointer<ClosedMesh> mesh, QSharedPointer<ClosedMesh> intersect)
    {
        mesh->intersection(*intersect);
    }

    void SlicingUtilities::UnionMesh(QSharedPointer<ClosedMesh> mesh, QSharedPointer<ClosedMesh> to_union)
    {
        mesh->mesh_union(to_union);
    }

    int SlicingUtilities::GetPartStart(QSharedPointer<Part> part, int current_steps)
    {
        int part_start = 0;
        if(part->countStepPairs() > 0)
        {
            while(part_start < current_steps && !part->stepGroupContains(part_start, StepType::kLayer))
                ++part_start;
        }
        return part_start;
    }

    std::tuple<Plane, Point, Point> SlicingUtilities::GetDefaultSlicingAxis(QSharedPointer<SettingsBase> sb, QSharedPointer<MeshBase> mesh, QSharedPointer<MeshSkeleton> skeleton)
    {
        Plane slicing_plane = Plane(mesh->min(), QVector3D(0, 0, 1)); // default plane at min of bounding box and horizontal

        // Get slicing settings, adjust slicing plane accordingly
        Axis  slicing_axis = static_cast<Axis>(sb->setting<int>(Constants::ProfileSettings::SlicingAngle::kSlicingAxis));
        Angle slicing_plane_pitch = sb->setting<Angle>(Constants::ProfileSettings::SlicingAngle::kStackingDirectionPitch);
        Angle slicing_plane_yaw   = sb->setting<Angle>(Constants::ProfileSettings::SlicingAngle::kStackingDirectionYaw);
        Angle slicing_plane_roll  = sb->setting<Angle>(Constants::ProfileSettings::SlicingAngle::kStackingDirectionRoll);
        QQuaternion quaternion = MathUtils::CreateQuaternion(slicing_plane_pitch, slicing_plane_yaw, slicing_plane_roll);
        slicing_plane.rotate(quaternion);

        // invert the slicing plane normal if necessary
        // normal must be a positive direction for comparisons to determine which side of the plane a point is on
        QVector3D axis_vector;
        switch (slicing_axis) {
            case Axis::kX:
                axis_vector = QVector3D(1, 0, 0);
                break;
            case Axis::kY:
                axis_vector = QVector3D(0, 1, 0);
                break;
            case Axis::kZ:
                axis_vector = QVector3D(0, 0, 1);
                break;
        }

        if (QVector3D::dotProduct(slicing_plane.normal(), axis_vector) < 0 ) {
            slicing_plane.normal(slicing_plane.normal() * -1);
        }

        // Find the min & max located on the mesh
        Point mesh_min, mesh_max;

        std::tie(mesh_min, mesh_max) = mesh->getAxisExtrema(slicing_plane.normal());

        // Move slicing plane to start at min on the part
        slicing_plane.point(mesh_min);

        return std::tuple<Plane, Point, Point>(slicing_plane, mesh_min, mesh_max);
    }

    void SlicingUtilities::ShiftSlicingPlane(QSharedPointer<SettingsBase> sb, Plane &slicing_plane, Distance last_height, QSharedPointer<MeshSkeleton> skeleton)
    {
        Distance layer_height = sb->setting<Distance>(Constants::ProfileSettings::Layer::kLayerHeight);

        // move the plane by translating its point along the appropriate axis
        if (sb->setting<bool>(Constants::ProfileSettings::SlicingAngle::kEnableCustomAxis)) {
            Axis slicing_axis = static_cast<Axis>(sb->setting<int>(Constants::ProfileSettings::SlicingAngle::kSlicingAxis));
            switch (slicing_axis) {
                case Axis::kX:
                    slicing_plane.shiftX((layer_height() / 2.) + (last_height() / 2.));
                    break;
                case Axis::kY:
                    slicing_plane.shiftY((layer_height() / 2.) + (last_height() / 2.));
                    break;
                case Axis::kZ:
                    slicing_plane.shiftZ((layer_height() / 2.) + (last_height() / 2.));
                    break;
            }
        }
        else {
            slicing_plane.shiftAlongNormal((layer_height() / 2.) + (last_height() / 2.));
        }
    }

    bool SlicingUtilities::doPartsOverlap(QVector<QSharedPointer<Part>> parts, Plane slicing_plane)
    {
        // Cross-section parts
        QVector<Polygon> polygons;
        for(auto part : parts)
        {
            // GCC doesn't like taking the address of a temporary.
            Point tmp_point;
            QVector3D tmp_vec;

            PolygonList geometry = CrossSection::doCrossSection(part->rootMesh(), slicing_plane, tmp_point, tmp_vec, part->getSb());

            // Since settings meshes are always rectangular prisms there is only a single island
            if(!geometry.isEmpty())
            {
                polygons.push_back(geometry.first());
            }
        }

        // Polygon in Polygon test
        bool overlap = false;
        for(int i = 0, end = polygons.size(); i < end; ++i)
        {
            for(int j = i + 1; j < end; ++j)
            {
                Polygon first = polygons[i];
                Polygon second = polygons[j];

                if(first.overlaps(second))
                {
                    overlap = true;
                    break;
                }
            }
        }

        return overlap;
    }
}
