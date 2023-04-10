#ifndef DXF_LOADER_H
#define DXF_LOADER_H

// Qt
#include <QThread>
#include <QTextCharFormat>
#include <QRegularExpression>

#include "gcode/gcode_command.h"
#include "utilities/enums.h"
#include "graphics/base_view.h"
#include "geometry/segment_base.h"
#include "gcode/parsers/sheet_lamination_parser.h"
#include <geometry/point.h>

namespace ORNL
{
    /*!
     * \class DXFLoader
     * \brief The DXFLoader class. This is the DXF equivalent of the GCodeLoader.
     * It's a threaded class that loads dxf, invokes appropriate parser,
     * and generates visualization.
     */
    class DXFLoader : public QThread {
        Q_OBJECT
        public:
            //! \brief Constructor
            //! \param filename: Name of file to load
            //! \param alterFile: boolean to determine whether to allow file alteration via
            //! minimum layer time enforcement defined in the file
            //! alterFile is not currently used
            DXFLoader(QString filename, bool alterFile);

            //! \brief Function that is run when start is called on this thread.
            void run() override;

        signals:

            //! \brief signal to view with info for visualization
            //! \param layers: vector of graphics layers for visualizaiton.  Information is
            //! provided to OpenGL view
            void dxfLoadedVisualization(QVector<QVector<QSharedPointer<SegmentBase>>> layers);

            //! \brief signal to UI with info for text and text font color
            //! \param text: text from file to display
            //! \param fontColors: Hash with formats to define colors for all lines of text. Allows lookup for each line into the hash for appropriate format.
            //! \param layerFirstLineNumbers: line numbers for BEGINNING LAYER for each layer to jump the cursor to appropriate line when spinbox moves up and down.
            //! \param layerSkipLineNumbers: line numbers to skip applying formatting to hash based on visualization settings
            void dxfLoadedText(QString text, QHash<QString, QTextCharFormat> fontColors, QList<int> layerFirstLineNumbers, QSet<int> layerSkipLineNumbers);

            //! \brief Emits error signal
            //! \param msg: Qstring error message
            void error(QString msg);

            //! \brief signal to layer time window with info
            //! \param layertimes: List of layer times for display
            void forwardInfoToLayerTimeWindow(QList<QList<Time>> layer_times, Time min_layer_time, Time max_layer_time, bool adjusted_layer_time);

            //! \brief signal to export window with info
            //! \param filename: temp dxf filename to copy
            //! \param meta: Currently selected meta
            void forwardInfoToBuildExportWindow(QString filename, GcodeMeta meta);

            //! \brief signal to main window with info
            //! \param parsingInfo: Formatted QString with various pieces of info to display in
            //!  main_window's status window.  Info includes: time, volume, weight, and material info.
            void forwardInfoToMainWindow(QString parsingInfo);

            //! \brief update slice dialog with current status
            //! \param type: Current step process
            //! \param percentComplete: Current completion percentage
            void updateDialog(StatusUpdateStepType type, int percentComplete);

        public slots:

            //! \brief Set cancel flag as received from slice dialog
            void cancelSlice();

            //! \brief Forward update from parser to update slice dialog with current status
            //! \param type: Current step process
            //! \param percentComplete: Current completion percentage
            void forwardDialogUpdate(StatusUpdateStepType type, int percentComplete);

        private:

            //! \brief Filename.
            QString m_filename;

            //! \brief dxf parser set to appropriate syntax once header is parsed
            //! for now, the only one that exists is the sheet lamination parser
            QScopedPointer<SheetLaminationParser> m_parser;

            //! \brief Meta info determined from file
            //! this class is inaccurately named now but it's what we're using
            GcodeMeta m_selected_meta;

            //! \brief Original lines in dxf file used as keys for text font hash
            //! m_lines contains all uppercase version to simplify text comparison
            QStringList m_lines, m_original_lines;

            //! \brief bool to indicate whether dxf should be adjusted for minimal layer time
            bool m_adjust_file;

            //! \brief Settings for visualization
            //! \brief Origin adjusted for offsets in settings, needed to undo adjustment in dxf
            QVector3D m_origin;
            //! \brief Offset for z
            float m_z_offset;
            //! \brief flag to indicate if process should cancel
            bool m_should_cancel;

    };
}
#endif // DXF_LOADER_H
