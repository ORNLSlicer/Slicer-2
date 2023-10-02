#ifndef PERIMETER_H
#define PERIMETER_H

// Local
#include "step/layer/regions/region_base.h"

namespace ORNL {
    class Perimeter : public RegionBase {
        public:
            //! \brief Constructor
            //! \param sb: the settings
            //! \param index: index for region order
            //! \param settings_polygons: a vector of settings polygons to apply
            //! \param gridInfo: optional external file information
            //! \param uncut_geometry: original geometry before setting region cutting
            Perimeter(const QSharedPointer<SettingsBase>& sb, const int index, const QVector<SettingsPolygon>& settings_polygons,
                      const SingleExternalGridInfo& gridInfo, PolygonList uncut_geometry);

            //! \brief Writes the gcode for the perimeter.
            //! \param writer Writer type to use for gcode output
            QString writeGCode(QSharedPointer<WriterBase> writer) override;

            //! \brief Computes the perimeter region.
            void compute(uint layer_num, QSharedPointer<SyncManager>& sync) override;

            //! \brief Computes the perimeter in one direction: outward/inward from the inner/outermost border contour
            //! \param bead_width: bead width of perimeter
            //! \param rings: number of perimeter contours
            void computeDirected(Distance bead_width, int rings);

            //! \brief Optimizes the region.
            //! \param poo: currently loaded path optimizer
            //! \param innerMostClosedContour: used for subsequent path modifiers
            //! \param outerMostClosedContour: used for subsequent path modifiers
            //! \param current_location: most recent location
            //! \param shouldNextPathBeCCW: state as to CW or CCW of previous path for use with additional DOF
            void optimize(QSharedPointer<PathOrderOptimizer> poo, Point& current_location,
                          QVector<Path>& innerMostClosedContour, QVector<Path>& outerMostClosedContour,
                          bool& shouldNextPathBeCCW) override;

            //! \brief Creates paths for the perimeter region.
            void createPaths();

            #ifdef HAVE_SINGLE_PATH
            //! \brief Sets the single path geometry
            //! \param sp_geometry: the new geometry
            void setSinglePathGeometry(QVector<SinglePath::PolygonList> sp_geometry);

            //! \brief Creates single paths for this region, plus any connected ones from insets
            void createSinglePaths();
            #endif

            //! \brief Sets the layer count
            //! \param layer_count: The total number of layers contained within the part that this region belongs to
            void setLayerCount(uint layer_count);

            //!\brief Returns the set of paths representing the outermost contours
            //! \return a list of paths of outermost perimeter contours
            QVector<Path>& getOuterMostPathSet();

            //!\brief Returns the set of paths representing the innermost contours
            //! \return a list of paths of innermost perimeter contours
            QVector<Path>& getInnerMostPathSet();

            //! \brief gets the computed geometry, used for single path
            //! \return the computed geometry
            QVector<PolygonList> getComputedGeometry();

        private:
            //! \brief Creates modifiers
            //! \param path Current path to add modifiers to
            //! \param supportsG3 Whether or not G2/G3 is supported for spiral lift
            //! \param innerMostClosedContour used for Prestarts (currently only skins/infill)
            //! \param current_location used to update start points of travels after modifiers are added
            void calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour, Point& current_location) override;

            //! \brief Holds the computed geometry before it is converted into paths
            QVector<PolygonList> m_computed_geometry;

            #ifdef HAVE_SINGLE_PATH
            //! \brief Holds the single path geometry before it is converted into paths
            QVector<SinglePath::PolygonList> m_single_path_geometry;
            #endif

            //! \brief Holds the first set of perimeter generated to provide for later
            //! optimizations and path modifiers
            QVector<Path> m_outer_most_path_set;

            //! \brief Holds the last set of perimeter generated to provide for later
            //! optimizations and path modifiers
            QVector<Path> m_inner_most_path_set;

            //! \brief Holds the total number of layers contained within the part that this region belongs to
            uint m_layer_count;

            //! \brief Holds the layer number that we are currently on
            uint m_layer_num;
    };
}

#endif // PERIMETER_H
