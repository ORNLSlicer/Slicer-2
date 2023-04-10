#ifndef LASERSCANISLAND_H
#define LASERSCANISLAND_H

// Local
#include "step/layer/island/island_base.h"

namespace ORNL {
    /*!
     * \class LaserScanIsland
     * \brief Island that laser scans are composed of.
     */
    class LaserScanIsland : public IslandBase {
        public:
            //! \brief Constructor
            //! \param geometry: the outlines
            //! \param sb: the settings
            //! \param settings_polygons: a vector of settings polygons to apply
            LaserScanIsland(const PolygonList& geometry, const QSharedPointer<SettingsBase>& sb, const QVector<SettingsPolygon>& settings_polygons);

            //! \brief Override from base. Filters down to individual regions to add
            //! travels and apply path modifiers
            void optimize(QSharedPointer<PathOrderOptimizer> poo, Point& currentLocation, QVector<QSharedPointer<RegionBase>>& previousRegions) override;
    };
}
#endif  // LASERSCANISLAND_H
