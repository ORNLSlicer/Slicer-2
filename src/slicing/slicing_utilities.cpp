#include "slicing/slicing_utilities.h"

// Locals
#include "cross_section/cross_section.h"
#include "step/layer/island/island_base.h"
#include "step/layer/layer.h"
#include "step/layer/scan_layer.h"
#include "step/step.h"
#include "utilities/mathutils.h"

namespace ORNL {

QVector<QSharedPointer<MeshBase>> SlicingUtilities::GetMeshesByType(QMap<QString, QSharedPointer<Part>> parts,
                                                                    MeshType mt) {
    QVector<QSharedPointer<MeshBase>> meshes;
    for (QSharedPointer<Part> part : parts) {
        if (part->rootMesh()->type() == mt)
            meshes.push_back(part->rootMesh());
    }
    return meshes;
}

QVector<QSharedPointer<Part>> SlicingUtilities::GetPartsByType(QMap<QString, QSharedPointer<Part>> parts, MeshType mt) {
    QVector<QSharedPointer<Part>> found_parts;
    for (QSharedPointer<Part> part : parts) {
        if (part->rootMesh()->type() == mt)
            found_parts.push_back(part);
    }
    return found_parts;
}

void SlicingUtilities::ClipMesh(QSharedPointer<MeshBase> mesh, QVector<QSharedPointer<MeshBase>> clippers) {
    for (QSharedPointer<MeshBase> clipper : clippers) {
        auto closed_clipper = dynamic_cast<ClosedMesh*>(clipper.get());
        auto closed_mesh = dynamic_cast<ClosedMesh*>(mesh.get());
        if (closed_clipper != nullptr && closed_mesh != nullptr)
            closed_mesh->difference(*closed_clipper);
    }
}

void SlicingUtilities::IntersectMesh(QSharedPointer<ClosedMesh> mesh, QSharedPointer<ClosedMesh> intersect) {
    mesh->intersection(*intersect);
}

void SlicingUtilities::UnionMesh(QSharedPointer<ClosedMesh> mesh, QSharedPointer<ClosedMesh> to_union) {
    mesh->mesh_union(to_union);
}

int SlicingUtilities::GetPartStart(QSharedPointer<Part> part, int current_steps) {
    int part_start = 0;
    if (part->countStepPairs() > 0) {
        while (part_start < current_steps && !part->stepGroupContains(part_start, StepType::kLayer))
            ++part_start;
    }
    return part_start;
}

std::tuple<Plane, Point, Point> SlicingUtilities::GetDefaultSlicingAxis(QSharedPointer<SettingsBase> sb,
                                                                        QSharedPointer<MeshBase> mesh,
                                                                        QSharedPointer<MeshSkeleton> skeleton) {
    // Retrieve the slicing plane normal
    QVector3D slicing_plane_normal = {
        sb->setting<float>(Constants::ProfileSettings::SlicingAngle::kSlicingPlaneNormalX),
        sb->setting<float>(Constants::ProfileSettings::SlicingAngle::kSlicingPlaneNormalY),
        sb->setting<float>(Constants::ProfileSettings::SlicingAngle::kSlicingPlaneNormalZ)};
    slicing_plane_normal.normalize();

    // Retrieve the mesh extrema along the slicing plane normal
    auto [min, max] = mesh->getAxisExtrema(slicing_plane_normal);

    // Create the slicing plane
    Plane slicing_plane(min, slicing_plane_normal);

    return {slicing_plane, min, max};
}

void SlicingUtilities::ShiftSlicingPlane(QSharedPointer<SettingsBase> sb, Plane& slicing_plane, Distance last_height,
                                         QSharedPointer<MeshSkeleton> skeleton) {
    // Retrieve the layer height
    const Distance& layer_height = sb->setting<Distance>(Constants::ProfileSettings::Layer::kLayerHeight);

    // Shift the slicing plane along the normal by half the layer height
    slicing_plane.shiftAlongNormal((layer_height() / 2.) + (last_height() / 2.));
}

bool SlicingUtilities::doPartsOverlap(QVector<QSharedPointer<Part>> parts, Plane slicing_plane) {
    // Cross-section parts
    QVector<Polygon> polygons;
    for (auto part : parts) {
        // GCC doesn't like taking the address of a temporary.
        Point tmp_point;
        QVector3D tmp_vec;

        PolygonList geometry =
            CrossSection::doCrossSection(part->rootMesh(), slicing_plane, tmp_point, tmp_vec, part->getSb());

        // Since settings meshes are always rectangular prisms there is only a single island
        if (!geometry.isEmpty()) {
            polygons.push_back(geometry.first());
        }
    }

    // Polygon in Polygon test
    bool overlap = false;
    for (int i = 0, end = polygons.size(); i < end; ++i) {
        for (int j = i + 1; j < end; ++j) {
            Polygon first = polygons[i];
            Polygon second = polygons[j];

            if (first.overlaps(second)) {
                overlap = true;
                break;
            }
        }
    }

    return overlap;
}
} // namespace ORNL
