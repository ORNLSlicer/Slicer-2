#include "step/layer/regions/anchor.h"
#include "geometry/segments/line.h"
#include "optimizers/path_order_optimizer.h"

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
            m_computed_geometry.append(path_line);
            path_line = path_line.offset(-beadWidth, -beadWidth / 2);
            ring_nr++;
        }

        this->createPaths();
    }

    void Anchor::optimize(QSharedPointer<PathOrderOptimizer> poo, Point& current_location, QVector<Path>& innerMostClosedContour, QVector<Path>& outerMostClosedContour, bool& shouldNextPathBeCCW)
    {
        poo->setPathsToEvaluate(m_paths);
        m_paths.clear();
        while(poo->getCurrentPathCount() > 0)
            m_paths.push_back(poo->linkNextPath());
    }

    void Anchor::calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour, Point& current_location)
    {
        //NOP
    }

    void Anchor::createPaths()
    {
        for(int i = 0, end = m_computed_geometry.size(); i < end; ++i)
        {
            PolygonList& polygon_list = m_computed_geometry[i];
            for (Polygon polygon : polygon_list)
            {
                Path new_path;

                Distance default_width                  = m_sb->setting< Distance >(Constants::ProfileSettings::Perimeter::kBeadWidth);
                Distance default_height                 = m_sb->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight);
                Velocity default_speed                  = m_sb->setting< Velocity >(Constants::ProfileSettings::Perimeter::kSpeed);
                Acceleration default_acceleration       = m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kPerimeter);
                AngularVelocity default_extruder_speed  = m_sb->setting< AngularVelocity >(Constants::ProfileSettings::Perimeter::kExtruderSpeed);
                int material_number                     = m_sb->setting< int >(Constants::MaterialSettings::MultiMaterial::kPerimterNum);

                for (int j = 0, end_cond = polygon.size(); j <  end_cond; ++j)
                {
                    Point start = polygon[j];
                    Point end = polygon[(j + 1) % end_cond];
                    QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(start, end);

                    // Add final segment
                    segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            default_width);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           default_height);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            default_speed);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            default_acceleration);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    default_extruder_speed);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   material_number);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       RegionType::kPerimeter);

                    new_path.append(segment);
                }

                if (new_path.calculateLength() > m_sb->setting< Distance >(Constants::ProfileSettings::Perimeter::kMinPathLength))
                    m_paths.append(new_path);
            }
        }
    }
}
