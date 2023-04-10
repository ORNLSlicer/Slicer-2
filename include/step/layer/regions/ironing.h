#ifndef IRONING_H
#define IRONING_H

#include "step/layer/regions/region_base.h"

namespace ORNL {
    class Ironing : public RegionBase {
        public:
            //! \brief Constructor
            //! \param sb: the settings
            //! \param index: index for region order
            //! \param settings_polygons: a vector of settings polygons to apply
            //! \param gridInfo: optional external file information
            Ironing(const QSharedPointer<SettingsBase>& sb, const int index, const QVector<SettingsPolygon>& settings_polygons, const SingleExternalGridInfo& gridInfo);

            //! \brief Writes the gcode for the perimeter.
            //! \param writer Writer type to use for gcode output
            QString writeGCode(QSharedPointer<WriterBase> writer) override;

            //! \brief Computes the perimeter region.
            void compute(uint layer_num, QSharedPointer<SyncManager>& sync) override;

            //! \brief Optimizes the region.
            //! \param poo: currently loaded path optimizer
            //! \param innerMostClosedContour: used for subsequent path modifiers
            //! \param outerMostClosedContour: used for subsequent path modifiers
            //! \param current_location: most recent location
            //! \param shouldNextPathBeCCW: state as to CW or CCW of previous path for use with additional DOF
            void optimize(QSharedPointer<PathOrderOptimizer> poo, Point& current_location,
                          QVector<Path>& innerMostClosedContour, QVector<Path>& outerMostClosedContour,
                          bool& shouldNextPathBeCCW) override;

            //! \brief Creates paths for the Ironing region.
            void createPaths() override;

            //! \brief Adds geometry from a layer above, if exists.
            //! \param poly_list geometry to add
            void addUpperGeometry(const PolygonList& poly_list);

        private:
            //! \brief Creates modifiers
            //! \param path Current path to add modifiers to
            //! \param supportsG3 Whether or not G2/G3 is supported for spiral lift
            //! \param innerMostClosedContour used for Prestarts (currently only skins/Ironing)
            //! \param current_location used to update start points of travels after modifiers are added
            void calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour, Point& current_location) override;

            //! \brief fills a set of geometry with Ironing according to settings
            //! \param geometry: what to fill
            //! \param sb: the settings to apply
            void fillGeometry(PolygonList geometry, const QSharedPointer<SettingsBase>& sb);

            //! \brief Holds the computed geometry before it is converted into paths
            QVector<QVector<Polyline>> m_computed_geometry;

            //! \brief Holds the computed paths for any settings regions that insersect Ironing region
            QVector<QVector<Path>> m_region_paths;

            //! \brief Holds a copy of the geometry for later optimization
            PolygonList m_geometry_copy;

            //! \brief The geometry above the current layer, if exixts.
            QVector<PolygonList> m_upper_geometry;
    };
}


#endif // IRONING_H
