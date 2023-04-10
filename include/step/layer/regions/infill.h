#ifndef INFILL_H
#define INFILL_H

// Local
#include "step/layer/regions/region_base.h"

namespace ORNL {
    class Infill : public RegionBase {
        public:
            //! \brief Constructor
            //! \param sb: the settings
            //! \param index: index for region order
            //! \param settings_polygons: a vector of settings polygons to apply
            //! \param gridInfo: optional external file information
            Infill(const QSharedPointer<SettingsBase>& sb, const int index, const QVector<SettingsPolygon>& settings_polygons, const SingleExternalGridInfo& gridInfo);

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

            //! \brief Creates paths for the infill region.
            void createPaths();

        private:
            //! \brief Creates modifiers
            //! \param path Current path to add modifiers to
            //! \param supportsG3 Whether or not G2/G3 is supported for spiral lift
            //! \param innerMostClosedContour used for Prestarts (currently only skins/infill)
            //! \param current_location used to update start points of travels after modifiers are added
            void calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour, Point& current_location) override;

            //! \brief fills a set of geometry with infill according to settings
            //! \param geometry: what to fill
            //! \param sb: the settings to apply
            void fillGeometry(PolygonList geometry, const QSharedPointer<SettingsBase>& sb);

            //! \brief determines if the infill settings of two settingbase are the same
            //! \param a: the first set of settings
            //! \param b: the second set of settings
            //! \return if they are the same
            bool settingsSame(QSharedPointer<SettingsBase> a, QSharedPointer<SettingsBase> b);

            //! \brief Applies external grid to infill segments
            //! \param seg: Segment to evaluate in grid
            //! \return vector of new segments generated from original segment after applying grid
            QVector<QSharedPointer<SegmentBase>> applyGrid(QSharedPointer<SegmentBase> seg);

            //! \brief Calculates average grid value for current segment and returns recipe index
            //! \param start: Start point of segment
            //! \param end: End point of segment
            //! \param xMax: Max x value of grid
            //! \param yMax: Max y value of grid
            //! \return recipe index for average grid value
            int getBlendVal(Point start, Point end, int xMax, int yMax);

            //! \brief Holds the computed geometry before it is converted into paths
            QVector<QVector<Polyline>> m_computed_geometry;

            //! \brief Holds the computed paths for any settings regions that insersect infill region
            QVector<QVector<Path>> m_region_paths;

            //! \brief Holds a copy of the geometry for later optimization
            PolygonList m_geometry_copy;
    };
}

#endif // INFILL_H
