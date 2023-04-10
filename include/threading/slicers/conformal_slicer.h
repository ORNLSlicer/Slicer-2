#ifndef CONFORMAL_SLICER_H
#define CONFORMAL_SLICER_H

// Local
#include "threading/traditional_ast.h"

#include <boost/property_map/property_map.hpp>
#include "geometry/mesh/mesh_base.h"
#include <step/layer/layer.h>
#include "geometry/mesh/advanced/parameterization.h"

namespace ORNL{
    /*!
     * \brief The ConformalSlicer builds regions on top of a surface
     */
    class ConformalSlicer : public TraditionalAST
    {
        public:
            //! \brief Constructor
            ConformalSlicer(QString gcodeLocation);

        protected:
            //! \brief Creates layers and regions and computes conformal mapping
            //! \param opt_data optional sensor data
            void preProcess(nlohmann::json opt_data = nlohmann::json()) override;

            //! \brief Connects paths and maps geometry onto the surface
            //! \param opt_data optional sensor data
            void postProcess(nlohmann::json opt_data = nlohmann::json()) override;

            //! \brief Writes layers/ regions to file
            void writeGCode() override;

        private:
    };
}

#endif // CONFORMAL_SLICER_H
