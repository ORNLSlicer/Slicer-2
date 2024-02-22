#ifndef THERMAL_SCAN_ISLAND_H
#define THERMAL_SCAN_ISLAND_H

// Local
#include "step/layer/island/island_base.h"

namespace ORNL {
    /*!
     * \class ThermalScanIsland
     * \brief Island that thermal scans are composed of.
     */
    class ThermalScanIsland : public IslandBase {
        public:
            //! \brief Constructor
            //! \param geometry: the outlines
            //! \param sb: the settings
            //! \param settings_polygons: a vector of settings polygons to apply
            ThermalScanIsland(const PolygonList& geometry, const QSharedPointer<SettingsBase>& sb, const QVector<SettingsPolygon>& settings_polygons);

            //! \brief Override from base. Filters down to individual regions to add
            //! travels and apply path modifiers
            void optimize(int layerNumber, Point& currentLocation, QVector<QSharedPointer<RegionBase>>& previousRegions) override;
    };
}
#endif // THERMAL_SCAN_ISLAND_H
