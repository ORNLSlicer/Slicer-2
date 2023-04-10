#ifndef SHEET_LAMINATION_SLICER_H
#define SHEET_LAMINATION_SLICER_H

#include <QDir>

// Local
#include "threading/traditional_ast.h"

#include "step/layer/layer.h"
#include "step/global_layer.h"
#include "optimizers/layer_order_optimizer.h"

namespace ORNL {
    class SheetLaminationSlicer : public TraditionalAST {
    public:
        //! \brief SheetLaminationSlicer
        //! \param gcodeLocation
        SheetLaminationSlicer(QString gcodeLocation);

    protected:
        //! \brief Creates layer steps by performing cross-sections.
        //! \param opt_data optional sensor data
        void preProcess(nlohmann::json opt_data = nlohmann::json()) override;

        //! \brief Post processing including support, etc.
        //! \param opt_data optional sensor data
        void postProcess(nlohmann::json opt_data = nlohmann::json()) override;

        //! \brief Parent override.  Writes out gcode.
        //! \param file: File pointer to write gcode out to
        //! \param base: WriterBase that creates actual gcode output
        void writeGCode() override;

    private:
        //! \brief list of all islands
        QVector<PolygonList> m_islands;

        //! \brief list of all z values that translate convenietly throws out
        QVector<float> m_island_z_values;

        //! \brief list of offsets in x y movement
        QVector<QVector<float>> m_offsets;

        //! \brief number of print beds or "layers" that the final packing takes up
        int m_resize_counter;

        //! \brief index of first island of each layer
        QVector<int> m_layer_changes;

        //! \brief list of layer numbers for each island
        QList<int> m_layer_list;

        //! \brief cached layer settings
        QList<QSharedPointer<SettingsBase>> m_saved_layer_settings;
    };
}

#endif // SHEET_LAMINATION_SLICER_H
