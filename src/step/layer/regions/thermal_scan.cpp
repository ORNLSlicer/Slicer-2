#include "step/layer/regions/thermal_scan.h"

#include <geometry/segments/scan.h>

#include <optimizers/path_order_optimizer.h>

namespace ORNL {
    ThermalScan::ThermalScan(const QSharedPointer<SettingsBase>& sb, const QVector<SettingsPolygon>& settings_polygons) : RegionBase(sb, settings_polygons) {
        // NOP
    }

    QString ThermalScan::writeGCode(QSharedPointer<WriterBase> writer) {
        QString gcode;
        gcode += writer->writeBeforeRegion(RegionType::kThermalScan);
        for (Path path : m_paths) {
            gcode += writer->writeBeforePath(RegionType::kThermalScan);
            for (QSharedPointer<SegmentBase> segment : path.getSegments()) {
                gcode += segment->writeGCode(writer);
            }
            gcode += writer->writeAfterPath(RegionType::kThermalScan);
        }
        gcode += writer->writeAfterRegion(RegionType::kThermalScan);
        return gcode;
    }

    void ThermalScan::compute(uint layer_num, QSharedPointer<SyncManager>& sync)
    {
        m_paths.clear();

        Distance x_offset = m_sb->setting<Distance>(Constants::ProfileSettings::ThermalScanner::kThermalScannerXOffset);
        Distance y_offset = m_sb->setting<Distance>(Constants::ProfileSettings::ThermalScanner::kThermalScannerYOffset);

        Point start, end = m_geometry.boundingRectCenter();
        end.x(end.x() - x_offset);
        end.y(end.y() - y_offset);

        m_computed_geometry = Polyline({start, end});

        this->createPaths();
    }

    void ThermalScan::optimize(QSharedPointer<PathOrderOptimizer> poo, Point& current_location, QVector<Path>& innerMostClosedContour, QVector<Path>& outerMostClosedContour, bool& shouldNextPathBeCCW)
    {
        //NOP?
    }

    void ThermalScan::calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour, Point& current_location)
    {
       //NOP
    }

    void ThermalScan::createPaths()
    {
        Path newPath;

        QSharedPointer<ScanSegment> segment = QSharedPointer<ScanSegment>::create(m_computed_geometry.first(), m_computed_geometry.last());
        segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       RegionType::kThermalScan);

        newPath.append(segment);

        m_paths.append(newPath);
    }
}

