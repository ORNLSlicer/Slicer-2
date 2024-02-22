// Main Module
#include "step/layer/island/wire_feed_island.h"

// Local
#include "step/layer/regions/skeleton.h"
#include "managers/settings/settings_manager.h"
#include "geometry/path_modifier.h"

namespace ORNL {
    WireFeedIsland::WireFeedIsland(const PolygonList& geometry, const QSharedPointer<SettingsBase>& sb, const QVector<SettingsPolygon>& settings_polygons,
                                 const SingleExternalGridInfo& gridInfo) :
        IslandBase(geometry, sb, settings_polygons, gridInfo)
    {
        this->addRegion(QSharedPointer<Skeleton>::create(sb, 0, settings_polygons, gridInfo, true));
        m_island_type = IslandType::kWireFeed;
    }

    void WireFeedIsland::optimize(int layerNumber, Point &currentLocation,
                                  QVector<QSharedPointer<RegionBase>>& previousRegions)
    {
        previousRegions.push_back(getRegion(RegionType::kSkeleton));
        QSharedPointer<Skeleton> wire_feed_region = getRegion(RegionType::kSkeleton).dynamicCast<Skeleton>();
        bool empty;
        QVector<Path> empty_path;

        wire_feed_region->optimize(layerNumber, currentLocation, empty_path, empty_path, empty);
        QVector<Path>& paths = wire_feed_region->getPaths();

        for(Path& path : paths)
        {
            PathModifierGenerator::GenerateInitialStartup(path, m_sb->setting<Distance>(Constants::ExperimentalSettings::WireFeed::kWirePrestartDistance),
                                                          m_sb->setting< Velocity >(Constants::ProfileSettings::Skeleton::kSpeed),
                                                          AngularVelocity(), false, 0.0);

            PathModifierGenerator::GenerateSlowdown(path, Distance(), Distance(),
                                                    m_sb->setting<Distance>(Constants::ExperimentalSettings::WireFeed::kWireCutoffDistance),
                                                    m_sb->setting< Velocity >(Constants::ProfileSettings::Skeleton::kSpeed),
                                                    m_sb->setting< AngularVelocity >(Constants::ProfileSettings::Skeleton::kExtruderSpeed),
                                                    false, 0.0);

            QSharedPointer<SegmentBase> nextToLast = path.getSegments()[path.getSegments().size() - 1];
            nextToLast->getSb()->setSetting(Constants::SegmentSettings::kFinalWireCoast, true);

            for(int i = path.getSegments().size() - 1; i >=0; --i)
            {
                QSharedPointer<SegmentBase> seg = path.getSegments()[i];
//                qDebug() << seg->getSb()->setting<PathModifiers>(Constants::SegmentSettings::kPathModifiers);
                if(seg->getSb()->setting<PathModifiers>(Constants::SegmentSettings::kPathModifiers) != PathModifiers::kCoasting)
                {
                    seg->getSb()->setSetting(Constants::SegmentSettings::kFinalWireFeed, true);
                    break;
                }
            }

            for(int i = 0, end = path.getSegments().size(); i < end; ++i)
            {
                QSharedPointer<SegmentBase> seg = path.getSegments()[i];
                if (dynamic_cast<TravelSegment*>(seg.data()))
                {
                    seg->getSb()->setSetting(Constants::SegmentSettings::kWireFeed,
                                             m_sb->setting<Distance>(Constants::ExperimentalSettings::WireFeed::kWireStickoutDistance));

                    seg->getSb()->setSetting(Constants::SegmentSettings::kSpeed,
                                             m_sb->setting<Velocity>(Constants::ExperimentalSettings::WireFeed::kInitialTravelSpeed));
                }
                else
                    seg->getSb()->setSetting(Constants::SegmentSettings::kWireFeed, seg->start().distance(seg->end()));

//                if(seg->getSb()->contains(Constants::SegmentSettings::kPathModifiers))
//                {
//                    PathModifiers mods = seg->getSb()->setting<PathModifiers>(Constants::SegmentSettings::kPathModifiers);
//                    if(seg->getSb()->setting<PathModifiers>(Constants::SegmentSettings::kPathModifiers) == PathModifiers::kInitialStartup)
//                    {
//                        //seg->getSb()->remove(Constants::SegmentSettings::kPathModifiers);
//                        seg->getSb()->setSetting(Constants::SegmentSettings::kWireFeed, seg->start().distance(seg->end()));

//                    }

//                    if(seg->getSb()->setting<PathModifiers>(Constants::SegmentSettings::kPathModifiers) == PathModifiers::kCoasting)
//                    {
//                        //seg->getSb()->remove(Constants::SegmentSettings::kPathModifiers);
//                        seg->getSb()->setSetting(Constants::SegmentSettings::kWireFeed, seg->start().distance(seg->end()));

//                        //seg->getSb()->setSetting(Constants::SegmentSettings::kFinalWireFeed, true);
//                        //path.getSegments()[i-1]->getSb()->setSetting(Constants::SegmentSettings::kFinalWireFeed, true);
//                    }
//                }
            }
        }

//        travel_segment->getSb()->setSetting(Constants::SegmentSettings::kWireFeed, true);
  //      paths.last().back()->getSb()->setSetting(Constants::SegmentSettings::kFinalWireFeed, true);

    }

    void WireFeedIsland::setAnchorWireFeed(QVector<Polyline> anchor_lines)
    {
        QSharedPointer<Skeleton> wire_feed_region = getRegion(RegionType::kSkeleton).dynamicCast<Skeleton>();
        wire_feed_region->setAnchorWireFeed(anchor_lines);
    }

}  // namespace ORNL
