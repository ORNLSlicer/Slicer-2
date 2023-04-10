#ifndef WIRE_FEED_ISLAND_H
#define WIRE_FEED_ISLAND_H

// Local
#include "step/layer/island/island_base.h"

namespace ORNL {
    /*!
     * \class WireFeedIsland
     * \brief Island that polymer builds use.
     */
    class WireFeedIsland : public IslandBase {
        public:
            //! \brief Constructor
            //! \param geometry: the outlines
            //! \param sb: the settings
            //! \param settings_polygons: a vector of settings polygons to apply
            //! \param gridInfo: optional external file information
            WireFeedIsland(const PolygonList& geometry, const QSharedPointer<SettingsBase>& sb, const QVector<SettingsPolygon>& settings_polygons,
                          const SingleExternalGridInfo& gridInfo = SingleExternalGridInfo());

            //! \brief Override from base. Filters down to individual regions to add
            //! travels and apply path modifiers
            void optimize(QSharedPointer<PathOrderOptimizer> poo, Point& currentLocation, QVector<QSharedPointer<RegionBase>>& previousRegions) override;

            //! \brief Sets extra wire feed for anchors to be merged with skeleton
            //! \param anchor_lines: polylines represents wire feed pathing
            void setAnchorWireFeed(QVector<Polyline> anchor_lines);
    };
}  // namespace ORNL
#endif  // WIRE_FEED_ISLAND_H
