#ifndef SUPPORTISLAND_H
#define SUPPORTISLAND_H

// Local
#include "step/layer/island/island_base.h"

namespace ORNL {
    /*!
     * \class SupportIsland
     * \brief Island that supports are composed of.
     */
    class SupportIsland : public IslandBase {
        public:
            //! \brief Constructor
            //! \param geometry: the outlines
            //! \param sb: the settings
            //! \param settings_polygons: a vector of settings polygons to apply
            SupportIsland(const PolygonList& geometry, const QSharedPointer<SettingsBase>& sb, const QVector<SettingsPolygon>& settings_polygons);

            //! \brief Override from base. Filters down to individual regions to add
            //! travels and apply path modifiers
            void optimize(QSharedPointer<PathOrderOptimizer> poo, Point& currentLocation, QVector<QSharedPointer<RegionBase>>& previousRegions) override;
    };

}

#endif //SUPPORTISLAND_H
