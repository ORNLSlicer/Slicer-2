#include "step/layer/regions/raft.h"
#include "geometry/segments/line.h"
#include "optimizers/polyline_order_optimizer.h"
#include "geometry/pattern_generator.h"
#include "geometry/path_modifier.h"

namespace ORNL {
    Raft::Raft(const QSharedPointer<SettingsBase>& sb, const QVector<SettingsPolygon>& settings_polygons) : RegionBase(sb, settings_polygons) {
        // NOP
    }

    QString Raft::writeGCode(QSharedPointer<WriterBase> writer) {
        QString gcode;
        gcode += writer->writeBeforeRegion(RegionType::kRaft);
        for (Path path : m_paths) {
            gcode += writer->writeBeforePath(RegionType::kRaft);
            for (QSharedPointer<SegmentBase> segment : path.getSegments()) {
                gcode += segment->writeGCode(writer);
            }
            gcode += writer->writeAfterPath(RegionType::kRaft);
        }
        gcode += writer->writeAfterRegion(RegionType::kRaft);
        return gcode;
    }

    void Raft::compute(uint layer_num, QSharedPointer<SyncManager>& sync){
        m_paths.clear();

        setMaterialNumber(m_sb->setting<int>(Constants::MaterialSettings::MultiMaterial::kPerimterNum));

        Distance nozzle_offset = this->getSb()->setting<Distance>(Constants::MaterialSettings::PlatformAdhesion::kRaftBeadWidth);

        m_computed_geometry.append(PatternGenerator::GenerateLines(m_geometry.offset(-nozzle_offset / 2), nozzle_offset, Angle(), false));
    }

    void Raft::optimize(int layerNumber, Point& current_location, QVector<Path>& innerMostClosedContour, QVector<Path>& outerMostClosedContour, bool& shouldNextPathBeCCW)
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
        poo.setInfillParameters(InfillPatterns::kLines, m_geometry, getSb()->setting<Distance>(Constants::ProfileSettings::Infill::kMinPathLength),
                          getSb()->setting<Distance>(Constants::ProfileSettings::Travel::kMinLength));

        poo.setPointParameters(pointOrderOptimization, getSb()->setting<bool>(Constants::ProfileSettings::Optimizations::kMinDistanceEnabled),
                               getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kMinDistanceThreshold),
                               getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kConsecutiveDistanceThreshold),
                               getSb()->setting<bool>(Constants::ProfileSettings::Optimizations::kLocalRandomnessEnable),
                               getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kLocalRandomnessRadius));

        poo.setGeometryToEvaluate(m_computed_geometry, RegionType::kInfill, static_cast<PathOrderOptimization>(m_sb->setting<int>(Constants::ProfileSettings::Optimizations::kPathOrder)));

        QVector<Polyline> previouslyLinkedLines;
        while(poo.getCurrentPolylineCount() > 0)
        {
            Polyline result = poo.linkNextPolyline(previouslyLinkedLines);
            if(result.size() > 0)
            {
                Path newPath = createPath(result);
                if(newPath.size() > 0)
                {
                    PathModifierGenerator::GenerateTravel(newPath, current_location, m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed));
                    current_location = newPath.back()->end();
                    previouslyLinkedLines.push_back(result);
                    m_paths.push_back(newPath);
                }
            }
        }
    }

    void Raft::calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour)
    {
        //NOP
    }

    Path Raft::createPath(Polyline line)
    {
        // Populate the paths.
        Distance default_width                  = this->getSb()->setting< Distance >(Constants::MaterialSettings::PlatformAdhesion::kRaftBeadWidth);
        Distance default_height                 = this->getSb()->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight);
        Velocity default_speed                  = m_sb->setting< Velocity >(Constants::ProfileSettings::Layer::kSpeed);
        Acceleration default_acceleration       = m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kDefault);
        AngularVelocity default_extruder_speed  = m_sb->setting< AngularVelocity >(Constants::ProfileSettings::Layer::kExtruderSpeed);
        int material_number             = m_sb->setting< int >(Constants::MaterialSettings::MultiMaterial::kInfillNum);

        Path newPath;
        for (int i = 0, end = line.size() - 1; i < end; ++i)
        {
            QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(line[i], line[i + 1]);

            segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            default_width);
            segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           default_height);
            segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            default_speed);
            segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            default_acceleration);
            segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    default_extruder_speed);
            segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   material_number);
            segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       RegionType::kRaft);

            newPath.append(segment);
        }

        if (newPath.calculateLength() > m_sb->setting< Distance >(Constants::ProfileSettings::Layer::kMinExtrudeLength))
            return newPath;
        else
            return Path();
    }
}
