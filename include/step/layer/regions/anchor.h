#ifndef ANCHOR_H
#define ANCHOR_H

// Local
#include "step/layer/regions/region_base.h"

namespace ORNL {
    class Anchor : public RegionBase {
        public:
            //! \brief Constructor
            //! \param sb: the settings
            //! \param settings_polygons: a vector of settings polygons to apply
            Anchor(const QSharedPointer<SettingsBase>& sb, const QVector<SettingsPolygon>& settings_polygons);

            //! \brief Writes the gcode for the raft.
            //! \param writer Writer type to use for gcode output
            QString writeGCode(QSharedPointer<WriterBase> writer) override;

            //! \brief Computes the raft region.
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

            //! \brief Creates paths for the Raft region.
            void createPaths();

        private:
            //! \brief Creates modifiers
            //! \param path Current path to add modifiers to
            //! \param supportsG3 Whether or not G2/G3 is supported for spiral lift
            //! \param innerMostClosedContour used for Prestarts (currently only skins/infill)
            //! \param current_location used to update start points of travels after modifiers are added
            void calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour, Point& current_location) override;

            //! \brief Holds the computed geometry before it is converted into paths
            QVector<PolygonList> m_computed_geometry;
    };
}

#endif // ANCHOR_H
