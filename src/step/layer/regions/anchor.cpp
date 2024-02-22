#include "step/layer/regions/anchor.h"
#include "geometry/segments/line.h"
#include "optimizers/polyline_order_optimizer.h"
#include "geometry/path_modifier.h"

namespace ORNL {
    Anchor::Anchor(const QSharedPointer<SettingsBase>& sb, const QVector<SettingsPolygon>& settings_polygons) : RegionBase(sb, settings_polygons) {
        // NOP
    }

    QString Anchor::writeGCode(QSharedPointer<WriterBase> writer) {
        QString gcode;
        gcode += writer->writeBeforeRegion(RegionType::kAnchor);
        for (Path path : m_paths) {
            gcode += writer->writeBeforePath(RegionType::kAnchor);
            for (QSharedPointer<SegmentBase> segment : path.getSegments()) {
                gcode += segment->writeGCode(writer);
            }
            gcode += writer->writeAfterPath(RegionType::kAnchor);
        }
        gcode += writer->writeAfterRegion(RegionType::kAnchor);
        return gcode;
    }

    void Anchor::compute(uint layer_num, QSharedPointer<SyncManager>& sync){
        m_paths.clear();

        Distance beadWidth = m_sb->setting<Distance>(Constants::ProfileSettings::Perimeter::kBeadWidth);
        int rings = m_sb->setting<int>(Constants::ProfileSettings::Perimeter::kCount);

        PolygonList path_line = m_geometry.offset(-beadWidth / 2);

        int ring_nr = 0;
        while (!path_line.isEmpty() && ring_nr < rings)
        {
            for(Polygon poly : path_line)
            {
                Polyline line = poly.toPolyline();
                line.pop_back();
                m_computed_geometry.push_back(line);
            }

            path_line = path_line.offset(-beadWidth, -beadWidth / 2);
            ring_nr++;
        }
    }

    void Anchor::optimize(int layerNumber, Point& current_location, QVector<Path>& innerMostClosedContour, QVector<Path>& outerMostClosedContour, bool& shouldNextPathBeCCW)
    {
        PolylineOrderOptimizer poo(current_location, layerNumber);

        PathOrderOptimization pathOrderOptimization = static_cast<PathOrderOptimization>(
                    this->getSb()->setting<int>(Constants::ProfileSettings::Optimizations::kPathOrder));
        if(pathOrderOptimization == PathOrderOptimization::kCustomPoint)
        {
            Point startOverride(getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPathXLocation),
                                getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPathYLocation));

            poo.setStartOverride(startOverride);
        }

        PointOrderOptimization pointOrderOptimization = static_cast<PointOrderOptimization>(
                    this->getSb()->setting<int>(Constants::ProfileSettings::Optimizations::kPointOrder));

        if(pointOrderOptimization == PointOrderOptimization::kCustomPoint)
        {
            Point startOverride(getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPointXLocation),
                                getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPointYLocation));

            poo.setStartPointOverride(startOverride);
        }

        poo.setPointParameters(pointOrderOptimization, getSb()->setting<bool>(Constants::ProfileSettings::Optimizations::kMinDistanceEnabled),
                               getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kMinDistanceThreshold),
                               getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kConsecutiveDistanceThreshold),
                               getSb()->setting<bool>(Constants::ProfileSettings::Optimizations::kLocalRandomnessEnable),
                               getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kLocalRandomnessRadius));

        m_paths.clear();

        poo.setGeometryToEvaluate(m_computed_geometry, RegionType::kAnchor, static_cast<PathOrderOptimization>(m_sb->setting<int>(Constants::ProfileSettings::Optimizations::kPathOrder)));

        while(poo.getCurrentPolylineCount() > 0)
        {
            Polyline result = poo.linkNextPolyline();
            Path newPath = createPath(result);

            if(newPath.size() > 0)
            {
                QVector<Path> temp_path;
                calculateModifiers(newPath, m_sb->setting<bool>(Constants::PrinterSettings::MachineSetup::kSupportG3), temp_path);
                PathModifierGenerator::GenerateTravel(newPath, current_location, m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed));
                current_location = newPath.back()->end();
                m_paths.push_back(newPath);
            }
        }
    }

    void Anchor::calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour)
    {
        //NOP
    }

    Path Anchor::createPath(Polyline line)
    {
        for(int i = 0, end = line.size(); i < end; ++i)
        {
            Path new_path;

            Distance default_width                  = m_sb->setting< Distance >(Constants::ProfileSettings::Perimeter::kBeadWidth);
            Distance default_height                 = m_sb->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight);
            Velocity default_speed                  = m_sb->setting< Velocity >(Constants::ProfileSettings::Perimeter::kSpeed);
            Acceleration default_acceleration       = m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kPerimeter);
            AngularVelocity default_extruder_speed  = m_sb->setting< AngularVelocity >(Constants::ProfileSettings::Perimeter::kExtruderSpeed);
            int material_number                     = m_sb->setting< int >(Constants::MaterialSettings::MultiMaterial::kPerimterNum);

            QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(line[i], line[(i + 1) % end]);
            segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            default_width);
            segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           default_height);
            segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            default_speed);
            segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            default_acceleration);
            segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    default_extruder_speed);
            segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   material_number);
            segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       RegionType::kPerimeter);

            new_path.append(segment);

            if (new_path.calculateLength() > m_sb->setting< Distance >(Constants::ProfileSettings::Perimeter::kMinPathLength))
               return new_path;
            else
                return Path();
        }
    }
}
