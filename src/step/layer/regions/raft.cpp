#include "step/layer/regions/raft.h"
#include "geometry/segments/line.h"
#include "optimizers/path_order_optimizer.h"
#include "geometry/pattern_generator.h"

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

        this->createPaths();
    }

    void Raft::optimize(QSharedPointer<PathOrderOptimizer> poo, Point& current_location, QVector<Path>& innerMostClosedContour, QVector<Path>& outerMostClosedContour, bool& shouldNextPathBeCCW)
    {
        InfillPatterns infillPattern = InfillPatterns::kLines;
        poo->setPathsToEvaluate(m_paths);
        poo->setParameters(infillPattern, m_geometry);

        QVector<Path> new_paths;
        while(poo->getCurrentPathCount() > 0)
        {
            Path new_path = poo->linkNextPath(new_paths);
            if(new_path.size() > 0)
            {
                new_paths.append(new_path);
            }
        }

        m_paths = new_paths;
    }

    void Raft::calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour, Point& current_location)
    {
        //NOP
    }

    void Raft::createPaths()
    {
        // Populate the paths.
        Distance default_width                  = this->getSb()->setting< Distance >(Constants::MaterialSettings::PlatformAdhesion::kRaftBeadWidth);
        Distance default_height                 = this->getSb()->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight);
        Velocity default_speed                  = m_sb->setting< Velocity >(Constants::ProfileSettings::Layer::kSpeed);
        Acceleration default_acceleration       = m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kDefault);
        AngularVelocity default_extruder_speed  = m_sb->setting< AngularVelocity >(Constants::ProfileSettings::Layer::kExtruderSpeed);
        int material_number             = m_sb->setting< int >(Constants::MaterialSettings::MultiMaterial::kInfillNum);

        for(int i = 0, end = m_computed_geometry.size(); i < end; ++i)
        {
            QVector<Polyline> polylineList = m_computed_geometry[i];

            for(Polyline polyline : polylineList)
            {
                Path newPath;
                for (int j = 0, polyEnd = polyline.size() - 1; j < polyEnd; ++j)
                {
                    QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(polyline[j], polyline[j + 1]);

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
                    m_paths.append(newPath);
            }
        }
    }
}
