// Main Module
#include "step/layer/island/wire_feed_island.h"

// Local
#include "step/layer/regions/skeleton.h"
#include "managers/settings/settings_manager.h"

namespace ORNL {
    WireFeedIsland::WireFeedIsland(const PolygonList& geometry, const QSharedPointer<SettingsBase>& sb, const QVector<SettingsPolygon>& settings_polygons,
                                 const SingleExternalGridInfo& gridInfo) :
        IslandBase(geometry, sb, settings_polygons, gridInfo)
    {
        this->addRegion(QSharedPointer<Skeleton>::create(sb, 0, settings_polygons, gridInfo, true));
        m_island_type = IslandType::kWireFeed;
    }

    void WireFeedIsland::optimize(QSharedPointer<PathOrderOptimizer> poo, Point &currentLocation,
                                  QVector<QSharedPointer<RegionBase>>& previousRegions)
    {

    }

    void WireFeedIsland::setAnchorWireFeed(QVector<Polyline> anchor_lines)
    {
        QSharedPointer<Skeleton> wire_feed_region = getRegion(RegionType::kSkeleton).dynamicCast<Skeleton>();
        wire_feed_region->setAnchorWireFeed(anchor_lines);
    }

}  // namespace ORNL
