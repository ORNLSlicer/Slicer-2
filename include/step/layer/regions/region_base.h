#ifndef REGIONBASE_H
#define REGIONBASE_H

// Qt
#include <QObject>

#include <gcode/writers/writer_base.h>

#include <geometry/mesh/advanced/parameterization.h>


// Local
#include "geometry/polygon_list.h"
#include "geometry/path.h"
#include "geometry/settings_polygon.h"
#include "configs/settings_base.h"
#include "managers/sync/sync_manager.h"
#include "external_files/external_grid.h"

namespace ORNL {
    class PathOrderOptimizer;
    /*!
     * \class RegionBase
     * \brief Base class for all region types.
     * \note For more information about the abstract slicing architecture, see the documentation.
     */
    class RegionBase {
        public:
            //! \brief Constructor
            //! \param sb: the settings
            //! \param index: index for region order
            //! \param settings_polygons: a vector of settings polygons to apply
            //! \param gridInfo: optional external file information
            RegionBase(const QSharedPointer<SettingsBase>& sb, const int index, const QVector<SettingsPolygon>& settings_polygons,
                       const SingleExternalGridInfo& gridInfo = SingleExternalGridInfo(), PolygonList uncut_geometry = PolygonList());

            //! \brief Constructor
            //! \param sb: the settings
            //! \param settings_polygons: a vector of settings polygons to apply
            //! \param gridInfo: optional external file information
            RegionBase(const QSharedPointer<SettingsBase>& sb, const QVector<SettingsPolygon>& settings_polygons, const SingleExternalGridInfo& gridInfo = SingleExternalGridInfo());

            //! \brief Destructor
            virtual ~RegionBase() = default;

            //! \brief Writes the GCode for this region.
            //! \param writer: writer for gcode syntax
            virtual QString writeGCode(QSharedPointer<WriterBase> writer) = 0;

            //! \brief Performs the computation for this region.
            //! \param layer_num: current layer number
            //! \param sync: Sync token for linking layers together
            virtual void compute(uint layer_num, QSharedPointer<SyncManager>& sync) = 0;

            //! \brief Performs the optimization for this region.
            //! \param poo: path optimizer that controls optimization behavior
            //! \param current_location: current location
            //! \param innerMostClosedContour: inner most contour used in conjunction with path modifiers
            //! \param outerMostClosedContour: used for subsequent path modifiers
            //! \param shouldNextPathBeCCW: CW or CCW state of last contour when using additional DOF
            virtual void optimize(QSharedPointer<PathOrderOptimizer> poo, Point& current_location,
                                  QVector<Path>& innerMostClosedContour, QVector<Path>& outerMostClosedContour,
                                  bool& shouldNextPathBeCCW) = 0;

            //! \brief Get the paths generated from this region.
            //! \return Reference to region paths
            QVector<Path>& getPaths();

            //! \brief Reverse path ordering in this region.
            void reversePaths();

            //! \brief Get the geometry for the region.
            //! \return Copy of geometry
            PolygonList getGeometry() const;

            //! \brief Set the geometry;
            //! \param geometry: geometry to set
            void setGeometry(const PolygonList& geometry);

            //! \brief Get the settings that the region will use.
            //! \return Pointer to settings base
            QSharedPointer<SettingsBase> getSb() const;

            //! \brief Set the settings that the region will use.
            //! \param sb: Pointer to settings base to set
            void setSb(const QSharedPointer<SettingsBase>& sb);

            //! \brief applies the conformal mapping
            //! \param parameterization is the UV map to map with
            //! \param normal_offset is the normal to shift by
            void applyMapping(QSharedPointer<Parameterization> parameterization, QVector3D normal_offset);

            //! \brief transforms the region by rotating by then quaternion, then shifting
            void transform(QQuaternion rotation, Point shift);

            //! \brief returns the minimun z-value of a region
            //! \return minimum z value
            float getMinZ();

            //! \brief return index that represents region order
            //! \return region order index
            int getIndex();

            //! \brief returns the material number of a region
            //! \return material number
            int getMaterialNumber();

            //! \brief set the material number of a region
            //! \param material_number: material number to set
            void setMaterialNumber(int material_number);

            //! \brief Update material numbers for the segments in each region
            //! \param transition_distance: distance needed for material transition
            //! \param next_material_number: material number to set for transition segments
            void calculateMultiMaterialTransition(Distance& transition_distance, int next_material_number);

            //! \brief adds nozzle to list of nozzles that should be on when this region prints
            //! \param nozzle number, indexed at 0
            void addNozzle(int nozzle);

            //! \brief adjusts regions according to multiple-nozzle settings
            void adjustMultiNozzle();

            //! \brief Sets whether last region was spiralized or not (path optimizer needs this info)
            //! \param spiral: whether or not last region was spiralized
            void setLastSpiral(bool spiral);

        protected:
            //! \brief Generates paths for the region.
            virtual void createPaths() = 0;

            //! \brief Adds a path to the region.
            //! \param path: path to append
            void appendPath(const Path& path);

            //! \brief adds the modifiers for each region
            //! \param path: path to add modifiers to
            //! \param supportsG3: whether or not the system supports G3 command
            //! \param innerMostClosedContour: inner most closed contour for use with path modifiers/open paths
            //! \param current_location: used to update start points of travel segments after modifiers
            virtual void calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour, Point& current_location) = 0;

            //! \brief The geometery this region will work on.
            PolygonList m_geometry;

            //! \brief The resultant paths of the computation.
            QVector<Path> m_paths;

            //! \brief The settings the region will use.
            QSharedPointer<SettingsBase> m_sb;

            //! \brief The settings polygon this region may use
            QVector<SettingsPolygon> m_settings_polygons;
			
            //! \brief The material used for the region.
            int m_material_number;

            //! \brief External grid information
            SingleExternalGridInfo m_grid_info;

            //! \brief Index for order
            int m_index;

            //! \brief Whether last region was spiralized
            bool m_was_last_region_spiral;
			
            //! \brief Uncut geometry to modify pathing
            PolygonList m_uncut_geometry;
    };
}

#endif //REGIONBASE_H
