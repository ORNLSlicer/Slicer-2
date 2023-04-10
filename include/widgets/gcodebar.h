#ifndef GCODEBAR_H
#define GCODEBAR_H

// Qt
#include <QWidget>
#include <QFrame>
#include <QLabel>
#include <QLineEdit>
#include <QComboBox>
#include <QSpinBox>
#include <QToolButton>
#include <QCheckBox>
#include <QGridLayout>

// Local
#include "widgets/gcodetextboxwidget.h"
#include "utilities/enums.h"

namespace ORNL {
    class GcodeBar : public QWidget {
        Q_OBJECT
        public:
            explicit GcodeBar(QWidget *parent = nullptr);

        signals:
            //! \brief Signals to update layer range
            //! \brief new_value: value to set lower/upper to
            void lowerLayerUpdated(int new_value);
            void upperLayerUpdated(int new_value);

            //! \brief Signal to notify GcodeView which segments to add/remove highlights to
            //! \param linesToAdd: segments to highlight
            //! \param linesToRemove: segments to unhighlight
            void lineChanged(QList<int> linesToAdd, QList<int> linesToRemove);

            //! \brief Signal to notify GcodeView to show/hide segment types
            //! \param segmentType: Type to change visibility
            //! \param checked: whether or not to show/hide segment type
            void forwardVisibilityChange(SegmentDisplayType segmentType, bool checked);

            //! \brief Signal to notify GcodeView to change width of segments
            //! \param use_true_width: if true segment width should be used
            void forwardSegmentWidthChange(bool use_true_width);

            //! \brief signal to refresh GCode view
            //! \param fileName: temporary file name for the modified GCode content
            //! \param needToModifyGCode: indicates if it is necessary to modify GCode based on minimum layer time requirement
            void refreshGCode(QString fileName, bool needToModifyGCode);

            //! \brief signal to remove highlight on a highlightable line upon search
            void removeHighlight();

        public slots:
            //! \brief necessary information passing to GcodeBar object after gcode loading
            //! \param text: GCode content
            //! \param fontColors: hash that contains color/highlight information for all lines
            //! \param layerFirstLineNumbers: layer numbers for all layer beginning lines
            //! \param layerSkipLineNumbers: line numbers to skip highlighting if visualization reduction setting is enabled
            void updateGcodeText(QString text, QHash<QString, QTextCharFormat> fontColors, QList<int> layerFirstLineNumbers, QSet<int> layerSkipLineNumbers);

            //! \brief Sets layer maxes
            //! \brief new_value: value to set max to
            void setMaxLayer(int max_value);

            //! \brief Set which lines to add/remove highlights to
            //! \param linesToAdd: segments to highlight
            //! \param linesToRemove: segments to unhighlight
            //! \param shouldCenter: whether or not to center line in text widget
            void setLineNumber(QList<int> linesToAdd, QList<int> linesToRemove, bool shouldCenter);

            //! \brief Forward which lines to add/remove highlights to
            //! \param linesToAdd: segments to highlight
            //! \param linesToRemove: segments to unhighlight
            void forwardLineChange(QList<int> linesToAdd, QList<int> linesToRemove);

            //! \brief move the cursor of GCode Editor to the corresponding layer
            //! \param layer_number: the layer number where the cursor is supposed to move to
            void moveToLayer(int layer_number);

            //! \brief search in GCode text, when Enter/Return is clicked in the search bar
            void search();

            //! \brief update GCode View when Refresh button on GCode Editor is clicked
            void updateGCodeView();

            //! \brief enable or disable the refresh button
            //! \param change: boolean to indicate if the refresh button should be enabled or disabled
            void updateRefreshButton(bool change);

        private slots:
            //! \brief update slots to keep spinboxes and sliders in sync and respect lock
            //! \param new_value: new potential layer
            void updateLowerSpin(int new_value);
            void updateLowerSlider(int new_value);
            void updateUpperSpin(int new_value);
            void updateUpperSlider(int new_value);

        private:
            // Setup the static widgets and their layouts.
            void setupWidget();

            // 1. Setup the widgets.
            void setupSubWidgets();
            // 2. Setup the layouts and insert their children.
            void setupLayouts();
            // 3. Setup the actions for the window.
            void setupActions();
            // 4. Setup the insertions for all UI elements.
            void setupInsert();
            // 5. Setup the events for the various widgets.
            void setupEvents();

            //! \brief Sets lower/upper layer values
            //! \brief new_value: value to set lower/upper spinbox/slider to
            void forwardLowerLayerUpdate(int new_value);
            void forwardUpperLayerUpdate(int new_value);

            // Layout
            QGridLayout *m_layout;

            // Separators
            QFrame *m_search_separator;
            QFrame *m_view_separator;

            // Main Widgets
            QLineEdit *m_search_bar;
            QToolButton *m_refresh_btn;
            GcodeTextBoxWidget *m_view;
            QComboBox *m_view_sel;
            QCheckBox *m_hide_travel;
            QCheckBox *m_hide_support;
            QCheckBox *m_true_width;
            QCheckBox *m_lock_layer;
            QSpinBox *m_layer_lower;
            QSpinBox *m_layer_upper;
            QSlider *m_layer_lower_slider;
            QSlider *m_layer_upper_slider;
            QLabel *m_lower_label;
            QLabel *m_upper_label;

            //! \brief store "BEGINNING LAYER" line numbers for all layers
            QList<int> m_layer_first_line_numbers;

            //! \brief indicate format change induced by search, other than content edit
            bool m_search_only_change;

            //! \brief number of Enter/Return hits, reset to 0 upon GCode loading
            int m_search_count;

            //! \brief If layers are locked, distance between lower/upper bound
            int m_lock_distance;
    };
} // Namespace ORNL

#endif // GCODEBAR_H
