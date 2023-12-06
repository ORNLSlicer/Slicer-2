// Main Module
#include "step/layer/island/polymer_island.h"

// Local
#include "step/layer/regions/perimeter.h"
#include "step/layer/regions/inset.h"
#include "step/layer/regions/skin.h"
#include "step/layer/regions/infill.h"
#include "step/layer/regions/ironing.h"
#include "step/layer/regions/skeleton.h"
#include "managers/settings/settings_manager.h"

namespace ORNL {
    PolymerIsland::PolymerIsland(const PolygonList& geometry, const QSharedPointer<SettingsBase>& sb, const QVector<SettingsPolygon>& settings_polygons,
                                 const SingleExternalGridInfo& gridInfo, const PolygonList& uncut_geometry) :
        IslandBase(geometry, sb, settings_polygons, gridInfo)
    {
        bool enable_perimeter = this->getSb()->setting<bool>(Constants::ProfileSettings::Perimeter::kEnable);
        bool enable_inset     = this->getSb()->setting<bool>(Constants::ProfileSettings::Inset::kEnable);
        bool enable_skin      = this->getSb()->setting<bool>(Constants::ProfileSettings::Skin::kEnable);
        bool enable_infill    = this->getSb()->setting<bool>(Constants::ProfileSettings::Infill::kEnable);
        bool enable_ironing   = this->getSb()->setting<bool>(Constants::ExperimentalSettings::Ironing::kEnable);
        bool enable_skeleton  = this->getSb()->setting<bool>(Constants::ProfileSettings::Skeleton::kEnable);

        QList<QString> order = this->getSb()->setting<QList<QString>>(Constants::ProfileSettings::Ordering::kRegionOrder);
        QList<RegionType> regionOrder;
        for(QString str : order)
            regionOrder.push_back(fromString(str.toUpper()));

        for(int i = regionOrder.size() - 1; i >= 0; --i)
        {
            if((regionOrder[i] == RegionType::kPerimeter && !enable_perimeter) ||
               (regionOrder[i] == RegionType::kInset && !enable_inset) ||
               (regionOrder[i] == RegionType::kSkin && !enable_skin) ||
               (regionOrder[i] == RegionType::kInfill && !enable_infill) ||
               (regionOrder[i] == RegionType::kSkeleton && !enable_skeleton))
                regionOrder.removeAt(i);
        }

        if (enable_perimeter) this->addRegion(QSharedPointer<Perimeter>::create(sb, regionOrder.indexOf(RegionType::kPerimeter), settings_polygons, gridInfo, uncut_geometry));
        if (enable_inset)     this->addRegion(QSharedPointer<Inset    >::create(sb, regionOrder.indexOf(RegionType::kInset), settings_polygons, gridInfo));
        if (enable_skin)      this->addRegion(QSharedPointer<Skin     >::create(sb, regionOrder.indexOf(RegionType::kSkin), settings_polygons, gridInfo));
        if (enable_infill)    this->addRegion(QSharedPointer<Infill   >::create(sb, regionOrder.indexOf(RegionType::kInfill), settings_polygons, gridInfo));
        if (enable_skeleton)  this->addRegion(QSharedPointer<Skeleton >::create(sb, regionOrder.indexOf(RegionType::kSkeleton), settings_polygons, gridInfo));

        if (enable_infill && enable_ironing){
            InfillPatterns infill_pattern = static_cast<InfillPatterns>(sb->setting<int>(Constants::ProfileSettings::Infill::kPattern));
            if(infill_pattern == InfillPatterns::kLines || infill_pattern == InfillPatterns::kGrid)
                this->addRegion(QSharedPointer<Ironing>::create(sb, regionOrder.size(), settings_polygons, gridInfo));
        }

        m_island_type = IslandType::kPolymer;
    }

    void PolymerIsland::optimize(QSharedPointer<PathOrderOptimizer> poo, Point &currentLocation,
                                 QVector<QSharedPointer<RegionBase>>& previousRegions)
    {
        QSharedPointer<Inset> insets = getRegion(RegionType::kInset).dynamicCast<Inset>();
        QSharedPointer<Perimeter> perimeters = getRegion(RegionType::kPerimeter).dynamicCast<Perimeter>();

        QVector<Path> innerMostClosedContour;
        if(insets != nullptr)
            innerMostClosedContour = insets->getInnerMostPathSet();
        else if(perimeters != nullptr)
            innerMostClosedContour = perimeters->getInnerMostPathSet();

        QVector<Path> outerMostClosedContour;
        if(perimeters != nullptr)
            outerMostClosedContour = perimeters->getOuterMostPathSet();
        else if(insets != nullptr)
            outerMostClosedContour = insets->getOuterMostPathSet();


        PathOrderOptimization pathOrderOptimization = static_cast<PathOrderOptimization>(
                    this->getSb()->setting<int>(Constants::ProfileSettings::Optimizations::kPathOrder));

        if(pathOrderOptimization == PathOrderOptimization::kCustomPoint)
        {
            Point startOverride(getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPathXLocation),
                                getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPathYLocation));

            poo->setStartOverride(startOverride);
        }

        PointOrderOptimization pointOrderOptimization = static_cast<PointOrderOptimization>(
                    this->getSb()->setting<int>(Constants::ProfileSettings::Optimizations::kPointOrder));

        if(pointOrderOptimization == PointOrderOptimization::kCustomPoint)
        {
            Point startOverride(getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPointXLocation),
                                getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPointYLocation));

            poo->setStartPointOverride(startOverride);
        }

        bool shouldNextPathBeCCW = true;
        if(previousRegions.size() != 0 && previousRegions.last()->getPaths().size() != 0)
            shouldNextPathBeCCW = !previousRegions.last()->getPaths().last().getCCW();

        bool wasLastSpiral = false;

        for (QSharedPointer<RegionBase> r : m_regions)
        {
            if(previousRegions.size() > 0)
                wasLastSpiral = previousRegions.last()->getSb()->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableSpiralize);

            r->setLastSpiral(wasLastSpiral);

            r->optimize(poo, currentLocation, innerMostClosedContour, outerMostClosedContour, shouldNextPathBeCCW);

            if(r->getPaths().size() > 0)
                previousRegions.push_back(r);

            if(m_sb->setting<bool>(Constants::MaterialSettings::MultiMaterial::kEnable) &&
               m_sb->setting<Distance>(Constants::MaterialSettings::MultiMaterial::kTransitionDistance) > 0 &&
               !m_sb->setting<bool>(Constants::ExperimentalSettings::MultiNozzle::kEnableMultiNozzleMultiMaterial))
            {
                // multi-material without multi-nozzle requires material transitions
                calculateMultiMaterialTransitions(previousRegions);
            }
        }
    }

    void PolymerIsland::reorderRegions()
    {
        std::sort(m_regions.begin(), m_regions.end(),
                  [](auto const &a, auto const &b) { return a->getIndex() < b->getIndex(); });
    }
}  // namespace ORNL
