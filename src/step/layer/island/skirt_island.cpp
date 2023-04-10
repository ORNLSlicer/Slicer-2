// Main Module
#include "step/layer/island/skirt_island.h"
#include "step/layer/regions/skirt.h"

namespace ORNL {
    SkirtIsland::SkirtIsland(const PolygonList& geometry, const QSharedPointer<SettingsBase>& sb, const QVector<SettingsPolygon>& settings_polygons) : IslandBase(geometry, sb, settings_polygons) {
        this->addRegion(QSharedPointer<Skirt>::create(sb, settings_polygons));
        m_island_type = IslandType::kSkirt;
    }

    void SkirtIsland::optimize(QSharedPointer<PathOrderOptimizer> poo, Point& currentLocation, QVector<QSharedPointer<RegionBase>>& previousRegions)
    {
        bool unused = true;
        for (QSharedPointer<RegionBase> r : m_regions)
        {
            QVector<Path> tmp_path;
            r->optimize(poo, currentLocation, tmp_path, tmp_path, unused);
        }
    }
}  // namespace ORNL
