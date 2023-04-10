// Main Module
#include "step/layer/island/brim_island.h"

// Local
#include "step/layer/regions/brim.h"
#include "managers/settings/settings_manager.h"

namespace ORNL {
    BrimIsland::BrimIsland(const PolygonList& geometry, const QSharedPointer<SettingsBase>& sb, const QVector<SettingsPolygon>& settings_polygons) : IslandBase(geometry, sb, settings_polygons) {
        this->addRegion(QSharedPointer<Brim>::create(sb, settings_polygons));
        m_island_type = IslandType::kBrim;
    }

    void BrimIsland::optimize(QSharedPointer<PathOrderOptimizer> poo, Point& currentLocation, QVector<QSharedPointer<RegionBase>>& previousRegions)
    {
        bool unused = true;
        for (QSharedPointer<RegionBase> r : m_regions)
        {
            QVector<Path> tmp_path;
            r->optimize(poo, currentLocation, tmp_path, tmp_path, unused);
        }
    }
}  // namespace ORNL
