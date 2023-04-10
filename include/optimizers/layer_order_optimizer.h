#ifndef LAYER_ORDER_OPTIMIZER_H
#define LAYER_ORDER_OPTIMIZER_H

//Local
#include "part/part.h"
#include "step/global_layer.h"

namespace ORNL
{
    /*!
     * \class LayerOrderOptimizer
     * \brief Creates and orders global layers according to the user-selected schema
     */
    class LayerOrderOptimizer
    {
        public:

            //! \brief creates a single global layer
            //! \param build_parts: list of parts with steps
            //! \note used by real time slicers
            static QSharedPointer<GlobalLayer> populateStep(QVector<QSharedPointer<Part>> build_parts);

            //! \brief Creates and orders global layers
            //! \param global_sb: global settings base
            //! \param build_parts: list of build parts to access steps
            static QList<QSharedPointer<GlobalLayer>> populateSteps(QSharedPointer<SettingsBase> global_sb, QVector<QSharedPointer<Part>> build_parts);

    };
}


#endif // LAYER_ORDER_OPTIMIZER_H
