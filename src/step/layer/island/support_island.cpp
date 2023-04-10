// Main Module
#include "step/layer/island/support_island.h"

// Local
#include "step/layer/regions/support.h"
#include "step/layer/regions/infill.h"

namespace ORNL {
    SupportIsland::SupportIsland(const PolygonList& geometry, const QSharedPointer<SettingsBase>& sb, const QVector<SettingsPolygon>& settings_polygons) : IslandBase(geometry, sb, settings_polygons) {
        // Stage 0
        this->addRegion(QSharedPointer<Support>::create(sb, settings_polygons));
        m_island_type = IslandType::kSupport;
    }

    void SupportIsland::optimize(QSharedPointer<PathOrderOptimizer> poo, Point& currentLocation, QVector<QSharedPointer<RegionBase>>& previousRegions)
    {
        bool unused = true;
        for (QSharedPointer<RegionBase> r : m_regions)
        {
            QVector<Path> tmp_path;
            r->optimize(poo, currentLocation, tmp_path, tmp_path, unused);

            if(r->getPaths().size() > 0)
                previousRegions.push_back(r);

            if(m_sb->setting<bool>(Constants::MaterialSettings::MultiMaterial::kEnable) &&
               m_sb->setting<Distance>(Constants::MaterialSettings::MultiMaterial::kTransitionDistance) > 0 &&
                    !m_sb->setting<bool>(Constants::ExperimentalSettings::MultiNozzle::kEnableMultiNozzleMultiMaterial))
            calculateMultiMaterialTransitions(previousRegions);
        }
    }
}
