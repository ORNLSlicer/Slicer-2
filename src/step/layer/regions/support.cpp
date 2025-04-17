#include "step/layer/regions/support.h"

#include "geometry/path_modifier.h"
#include "geometry/segments/line.h"
#include "optimizers/polyline_order_optimizer.h"
#include "utilities/enums.h"

namespace ORNL {
Support::Support(const QSharedPointer<SettingsBase>& sb, const QVector<SettingsPolygon>& settings_polygons)
    : RegionBase(sb, settings_polygons) {
    // NOP
}

QString Support::writeGCode(QSharedPointer<WriterBase> writer) {
    QString gcode;
    gcode += writer->writeBeforeRegion(RegionType::kSupport);
    for (Path path : m_paths) {
        gcode += writer->writeBeforePath(RegionType::kSupport);
        for (QSharedPointer<SegmentBase> segment : path.getSegments()) {
            gcode += segment->writeGCode(writer);
        }
        gcode += writer->writeAfterPath(RegionType::kSupport);
    }
    gcode += writer->writeAfterRegion(RegionType::kSupport);
    return gcode;
}

void Support::compute(uint layer_num, QSharedPointer<SyncManager>& sync) {
    m_paths.clear();

    setMaterialNumber(m_sb->setting<int>(Constants::MaterialSettings::MultiMaterial::kPerimterNum));

    Distance bead_width = m_sb->setting<Distance>(Constants::ProfileSettings::Layer::kBeadWidth);
    Distance line_spacing = m_sb->setting<Distance>(Constants::ProfileSettings::Support::kLineSpacing);
    Area min_infill_area = m_sb->setting<Area>(Constants::ProfileSettings::Support::kMinInfillArea);
    InfillPatterns infill_pattern =
        static_cast<InfillPatterns>(m_sb->setting<int>(Constants::ProfileSettings::Support::kPattern));

    PolygonList perimeter_geometry = m_geometry.offset(-(bead_width / 2.0));

    m_geometry = perimeter_geometry;

    for (Polygon poly : perimeter_geometry) {
        Polyline line = poly.toPolyline();
        line.pop_back();
        m_computed_perimeter_geometry.push_back(line);
    }

    // Determine whether or not to generate support infill
    if (m_geometry.netArea() > min_infill_area) {
        switch (infill_pattern) {
            case InfillPatterns::kLines:
                computeLine(line_spacing); // Default rotation angle = 0 deg
                break;
            case InfillPatterns::kGrid:
                computeGrid(line_spacing); // Default rotation angle = 0 deg
                break;
            case InfillPatterns::kConcentric:
            case InfillPatterns::kInsideOutConcentric:
            case InfillPatterns::kTriangles:
            case InfillPatterns::kHexagonsAndTriangles:
            case InfillPatterns::kHoneycomb:
            case InfillPatterns::kRadialHatch:
                break;
        }
    }
}

void Support::computeLine(Distance line_spacing, Angle rotation) {
    Distance bead_width = m_sb->setting<Distance>(Constants::ProfileSettings::Layer::kBeadWidth);

    //! Create a non-path border for the infill pattern
    PolygonList border_polygons = m_geometry.offset(-bead_width);

    PolygonList rotated_polygon = border_polygons.rotate(rotation);

    //! Get the bounding box for the polygons. outline_minimum is the minimum
    Point outline_minimum = rotated_polygon.min();
    Point outline_maximum = rotated_polygon.max();

    //! The result we get after intersecting the polygons with the grid lines
    QVector<Polyline> cutlines;

    //! The space left over after all the max number of cutlines are generated
    Distance freeSpace = (outline_maximum.toDistance3D().x - outline_minimum.toDistance3D().x) % line_spacing;

    //! start at the bounding box's minimum x value and go all the way to the bounding box's maximum x value.
    //! As we go along, every "line_spacing" distance we intersect the polygons with the grid lines
    for (Distance x = outline_minimum.toDistance3D().x + (freeSpace / 2); x < outline_maximum.toDistance3D().x;
         x += line_spacing) {
        //! Create the grid lines
        Polyline cutline;
        cutline << Point(x(), outline_minimum.y());
        cutline << Point(x(), outline_maximum.y());

        //! Intersect the polygons and the gridlines and store them
        //! \note This calls ClipperLib
        cutlines += rotated_polygon & cutline;
    }

    //! Unrotate polygons
    for (int i = 0; i < cutlines.size(); i++) {
        cutlines[i] = cutlines[i].rotate(-rotation);
        if (i % 2 == 0)
            cutlines[i] = cutlines[i].reverse();
    }

    m_computed_infill_geometry.append(cutlines);
}

void Support::computeGrid(Distance line_spacing, Angle rotation) {
    //! Call computeLine with our base rotation
    computeLine(line_spacing, rotation);

    //! Call computeLine with our base rotation plus 90 deg
    computeLine(line_spacing, rotation + 90 * deg);
}

void Support::optimize(int layerNumber, Point& current_location, QVector<Path>& innerMostClosedContour,
                       QVector<Path>& outerMostClosedContour, bool& shouldNextPathBeCCW) {
    PolylineOrderOptimizer poo(current_location, layerNumber);

    PathOrderOptimization pathOrderOptimization = static_cast<PathOrderOptimization>(
        this->getSb()->setting<int>(Constants::ProfileSettings::Optimizations::kPathOrder));
    if (pathOrderOptimization == PathOrderOptimization::kCustomPoint) {
        Point startOverride(getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPathXLocation),
                            getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPathYLocation));

        poo.setStartOverride(startOverride);
    }

    PointOrderOptimization pointOrderOptimization = static_cast<PointOrderOptimization>(
        this->getSb()->setting<int>(Constants::ProfileSettings::Optimizations::kPointOrder));

    if (pointOrderOptimization == PointOrderOptimization::kCustomPoint) {
        Point startOverride(getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPointXLocation),
                            getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPointYLocation));

        poo.setStartPointOverride(startOverride);
    }

    poo.setPointParameters(
        pointOrderOptimization, getSb()->setting<bool>(Constants::ProfileSettings::Optimizations::kMinDistanceEnabled),
        getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kMinDistanceThreshold),
        getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kConsecutiveDistanceThreshold),
        getSb()->setting<bool>(Constants::ProfileSettings::Optimizations::kLocalRandomnessEnable),
        getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kLocalRandomnessRadius));

    m_paths.clear();

    poo.setGeometryToEvaluate(
        m_computed_perimeter_geometry, RegionType::kSupport,
        static_cast<PathOrderOptimization>(m_sb->setting<int>(Constants::ProfileSettings::Optimizations::kPathOrder)));

    while (poo.getCurrentPolylineCount() > 0) {
        Polyline result = poo.linkNextPolyline();
        Path newPath = createPath(result);

        if (newPath.size() > 0) {
            PathModifierGenerator::GenerateTravel(newPath, current_location,
                                                  m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed));
            current_location = newPath.back()->end();
            m_paths.push_back(newPath);
        }
    }

    poo.setInfillParameters(
        static_cast<InfillPatterns>(m_sb->setting<int>(Constants::ProfileSettings::Support::kPattern)), m_geometry,
        getSb()->setting<Distance>(Constants::ProfileSettings::Infill::kMinPathLength),
        getSb()->setting<Distance>(Constants::ProfileSettings::Travel::kMinLength));

    poo.setGeometryToEvaluate(
        m_computed_infill_geometry, RegionType::kInfill,
        static_cast<PathOrderOptimization>(m_sb->setting<int>(Constants::ProfileSettings::Optimizations::kPathOrder)));

    QVector<Polyline> previouslyLinkedLines;
    while (poo.getCurrentPolylineCount() > 0) {
        Polyline result = poo.linkNextPolyline(previouslyLinkedLines);
        if (result.size() > 0) {
            Path newPath = createPath(result);
            if (newPath.size() > 0) {
                PathModifierGenerator::GenerateTravel(
                    newPath, current_location, m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed));
                current_location = newPath.back()->end();
                previouslyLinkedLines.push_back(result);
                m_paths.push_back(newPath);
            }
        }
    }
}

void Support::calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour) {
    // NOP
}

Path Support::createPath(Polyline line) {
    Distance bead_width = m_sb->setting<Distance>(Constants::ProfileSettings::Layer::kBeadWidth);
    Distance layer_height = m_sb->setting<Distance>(Constants::ProfileSettings::Layer::kLayerHeight);
    Velocity speed = m_sb->setting<Velocity>(Constants::ProfileSettings::Layer::kSpeed);
    Acceleration acceleration = m_sb->setting<Acceleration>(Constants::PrinterSettings::Acceleration::kSupport);
    AngularVelocity extruder_speed = m_sb->setting<AngularVelocity>(Constants::ProfileSettings::Layer::kExtruderSpeed);
    int material_number = m_sb->setting<int>(Constants::MaterialSettings::MultiMaterial::kPerimterNum);

    Path newPath;

    QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(line.first(), line.last());

    segment->getSb()->setSetting(Constants::SegmentSettings::kWidth, bead_width);
    segment->getSb()->setSetting(Constants::SegmentSettings::kHeight, layer_height);
    segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed, speed);
    segment->getSb()->setSetting(Constants::SegmentSettings::kAccel, acceleration);
    segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed, extruder_speed);
    segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber, material_number);
    segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType, RegionType::kSupport);

    newPath.append(segment);

    if (newPath.calculateLength() > m_sb->setting<Distance>(Constants::ProfileSettings::Layer::kMinExtrudeLength))
        return newPath;
    else
        return Path();
}
} // namespace ORNL
