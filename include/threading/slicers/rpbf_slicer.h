#ifndef RPBF_SLICER_H
#define RPBF_SLICER_H

// Local
#include "threading/traditional_ast.h"

#include "geometry/mesh/closed_mesh.h"
#include "geometry/mesh/open_mesh.h"
#include "geometry/mesh/mesh_base.h"
#include "step/layer/layer.h"

#include "step/layer/island/powder_sector_island.h"
namespace ORNL{

    /*!
     * \brief The RPBFSlicer supports unique build constraints: multi-nozzle metal powder
     * with circular, rotating build volume
     */
    class RPBFSlicer : public TraditionalAST
    {
        public:
            //! \brief Constructor
            RPBFSlicer(QString gcodeLocation);

        protected:
            //! \brief Creates layers and regions
            //! \param opt_data optional sensor data
            void preProcess(nlohmann::json opt_data = nlohmann::json()) override;

            //! \brief Connects paths
            //! \param opt_data optional sensor data
            void postProcess(nlohmann::json opt_data = nlohmann::json()) override;

            //! \brief Writes layers/ regions to file
            void writeGCode() override;

        private:
            //! \brief Splits geometry into sectors
            //! \param perimeters: global vector of all perimeters represented as polylines
            //! \param layer_specific_settings: settings base
            //! \param infill_geometry: polygonal representation of remaining area for infill
            //! \param sectors: output vector with necessary vector information - perimeters, starting vector, sector angle rotation, and infill area
            void splitIntoSectors(QVector<Polyline> perimeters, QSharedPointer<SettingsBase> layer_specific_settings, PolygonList infill_geometry, QVector<QVector<SectorInformation>>& sectors, int layer_count);

    };
}

#endif // RPBF_SLICER_H
