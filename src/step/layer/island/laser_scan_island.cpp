// Main Module
#include "step/layer/island/laser_scan_island.h"

// Local
#include "step/layer/regions/laser_scan.h"

namespace ORNL {
    LaserScanIsland::LaserScanIsland(const PolygonList& geometry, const QSharedPointer<SettingsBase>& sb, const QVector<SettingsPolygon>& settings_polygons) : IslandBase(geometry, sb, settings_polygons) {
        // Stage @
        this->addRegion(QSharedPointer<LaserScan>::create(sb, settings_polygons));
        m_island_type = IslandType::kLaserScan;
    }

    void LaserScanIsland::optimize(int layerNumber, Point& currentLocation, QVector<QSharedPointer<RegionBase>>& previousRegions)
    {
        bool unused = true;
        for (QSharedPointer<RegionBase> r : m_regions)
        {
            QVector<Path> tmp_path;
            r->optimize(layerNumber, currentLocation, tmp_path, tmp_path, unused);
        }
    }
}
