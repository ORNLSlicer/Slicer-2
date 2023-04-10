#include <QStandardPaths>
#include <QDir>
#include <QTextDocumentWriter>
#include <QPushButton>
#include "widgets/gcodebar.h"
#include "managers/preferences_manager.h"

namespace ORNL {
    GcodeBar::GcodeBar(QWidget *parent) : QWidget(parent) {
        this->setupWidget();
        m_lock_distance = 0;
    }

    void GcodeBar::updateGcodeText(QString text, QHash<QString, QTextCharFormat> fontColors, QList<int> layerFirstLineNumbers, QSet<int> layerSkipLineNumbers)
    {
        //new gcode so reset any previous highlight, forward along the font colors first, then add text
        //it is necessary to forward font colors first as the highlighter overrides font colors as text is added
        //so it must know which colors to set before adding text
        m_view->resetHighlight();
        m_view->setHighlighterColors(fontColors, layerSkipLineNumbers);
        m_view->setPlainText(text);
        m_layer_first_line_numbers = layerFirstLineNumbers;
        m_view->setLayerFirstLineNumbers(m_layer_first_line_numbers);
        m_refresh_btn->setEnabled(false);

        //preserve the last search. No harm if search string is empty
        m_search_count = 0;
        search();
    }

    void GcodeBar::updateLowerSpin(int new_value)
    {
        if(m_lock_layer->isChecked())
        {
            if(m_layer_lower->value() > m_layer_lower->maximum() - m_lock_distance)
            {
                new_value = m_layer_lower->maximum() - m_lock_distance;
                m_layer_lower->blockSignals(true);
                m_layer_lower->setValue(new_value);
                m_layer_lower->blockSignals(false);
            }
        }
        m_layer_lower_slider->blockSignals(true);
        m_layer_lower_slider->setValue(new_value);
        m_layer_lower_slider->blockSignals(false);
        forwardLowerLayerUpdate(new_value);
    }

    void GcodeBar::updateLowerSlider(int new_value)
    {
        if(m_lock_layer->isChecked())
        {
            if(m_layer_lower_slider->value() > m_layer_lower_slider->maximum() - m_lock_distance)
            {
                new_value = m_layer_lower_slider->maximum() - m_lock_distance;
                m_layer_lower_slider->blockSignals(true);
                m_layer_lower_slider->setValue(new_value);
                m_layer_lower_slider->blockSignals(false);
            }
        }
        m_layer_lower->blockSignals(true);
        m_layer_lower->setValue(new_value);
        m_layer_lower->blockSignals(false);
        forwardLowerLayerUpdate(new_value);
    }

    void GcodeBar::forwardLowerLayerUpdate(int new_value)
    {
        if(m_lock_layer->isChecked())
            m_layer_upper->setValue(new_value + m_lock_distance);
        //Before we forward, we should check if the new upper layer value is less than the lower layer,
        //and if so, set the lower layer to the upper layer.
        else if(new_value > m_layer_upper->value())
        {
            //Note that this will cause the slot GcodeBar::forwardUpperLayerUpdate to be invoked
            m_layer_upper->setValue(new_value);
        }
        emit lowerLayerUpdated(new_value);
    }

    void GcodeBar::updateUpperSpin(int new_value)
    {
        if(m_lock_layer->isChecked())
        {
            if(m_layer_upper->value() < m_layer_upper->minimum() + m_lock_distance)
            {
                new_value = m_layer_upper->minimum() + m_lock_distance;
                m_layer_upper->blockSignals(true);
                m_layer_upper->setValue(m_layer_upper->minimum() + m_lock_distance);
                m_layer_upper->blockSignals(false);
            }
        }
        m_layer_upper_slider->blockSignals(true);
        m_layer_upper_slider->setValue(new_value);
        m_layer_upper_slider->blockSignals(false);
        forwardUpperLayerUpdate(new_value);
    }

    void GcodeBar::updateUpperSlider(int new_value)
    {
        if(m_lock_layer->isChecked())
        {
            if(m_layer_upper_slider->value() < m_layer_upper_slider->minimum() + m_lock_distance)
            {
                new_value = m_layer_upper_slider->minimum() + m_lock_distance;
                m_layer_upper_slider->blockSignals(true);
                m_layer_upper_slider->setValue(m_layer_upper_slider->minimum() + m_lock_distance);
                m_layer_upper_slider->blockSignals(false);
            }
        }
        m_layer_upper->blockSignals(true);
        m_layer_upper->setValue(new_value);
        m_layer_upper->blockSignals(false);
        forwardUpperLayerUpdate(new_value);
    }


    void GcodeBar::forwardUpperLayerUpdate(int new_value)
    {
        if(m_lock_layer->isChecked())
            m_layer_lower->setValue(new_value - m_lock_distance);
        //Before we forward, we should check if the new upper layer value is less than the lower layer,
        //and if so, set the lower layer to the upper layer.
        else if(new_value < m_layer_lower->value())
        {
            //Note that this will cause the slot GcodeBar::forwardLowerLayerUpdate to be invoked
            m_layer_lower->setValue(new_value);
        }
        emit upperLayerUpdated(new_value);
    }

    void GcodeBar::forwardLineChange(QList<int> linesToAdd, QList<int> linesToRemove)
    {
        this->setLineNumber(linesToAdd, linesToRemove, false);
        emit lineChanged(linesToAdd, linesToRemove);

        if(linesToAdd.count() == 1 && linesToAdd[0] > 0){
            int layerNumber = 0;
            for (int layerStartLine : m_layer_first_line_numbers) {
                if(layerStartLine > linesToAdd[0] + 1)
                    break;
                ++layerNumber;
            }
            --layerNumber;

            if(layerNumber < m_layer_lower->value()){
                updateLowerSpin(layerNumber);
                updateLowerSlider(layerNumber);
            }
            else if(layerNumber > m_layer_upper->value()){
                updateUpperSpin(layerNumber);
                updateUpperSlider(layerNumber);
            }
        }
    }

    void GcodeBar::moveToLayer(int layerNo)
    {
        //if moveToLayer() is called by either clicking a visible line or an up/down arrow, do not move the line to the top
        if(!m_view->getCursorManualMove() && layerNo >= 0 && layerNo < m_layer_first_line_numbers.size())
        {
            m_view->moveCursorToLine(m_layer_first_line_numbers.at(layerNo));
        }
    }

    void GcodeBar::setMaxLayer(int max_value)
    {
        m_layer_lower->setMaximum(max_value);
        m_layer_lower_slider->setMaximum(max_value);
        m_layer_upper->setMaximum(max_value);
        m_layer_upper_slider->setMaximum(max_value);
    }

    void GcodeBar::setLineNumber(QList<int> linesToAdd, QList<int> linesToRemove, bool shouldCenter)
    {
        m_view->highlightLine(linesToAdd, linesToRemove, shouldCenter);
    }

    void GcodeBar::setupWidget() {
        this->setupSubWidgets();
        this->setupLayouts();
        this->setupActions();
        this->setupInsert();
        this->setupEvents();
    }

    void GcodeBar::setupSubWidgets() {
        // Search Separator
        m_search_separator = new QFrame(this);
        m_search_separator->setFrameShape(QFrame::HLine);
        m_search_separator->setFrameShadow(QFrame::Sunken);

        // View Separator
        m_view_separator = new QFrame(this);
        m_view_separator->setFrameShape(QFrame::HLine);
        m_view_separator->setFrameShadow(QFrame::Sunken);

        // Search View
        m_search_bar = new QLineEdit(this);
        m_search_bar->setPlaceholderText("Search through GCode...");
        m_search_bar->setToolTip("Search through GCode ...");

        // Refresh button
        m_refresh_btn = new QToolButton(this);
        m_refresh_btn->setIcon(QIcon(":/icons/file_refresh_black.png"));
        m_refresh_btn->setToolTip("Refresh");
        m_refresh_btn->setEnabled(false);

        // Main View
        m_view = new GcodeTextBoxWidget(this);

        // View Selection
        m_view_sel = new QComboBox(this);

        // Hide Travel CheckBox
        m_hide_travel = new QCheckBox(this);
        m_hide_travel->setText("Hide Travel");
        m_hide_travel->setChecked(PM->getHideTravelPreference());

        // Hide Support CheckBox
        m_hide_support = new QCheckBox(this);
        m_hide_support->setText("Hide Support");
        m_hide_support->setChecked(PM->getHideSupportPreference());

        // True Bead Widths CheckBox
        m_true_width = new QCheckBox(this);
        m_true_width->setText("True Bead Widths");
        m_true_width->setChecked(PM->getUseTrueWidthsPreference());

        // Lower Layer Spinbox
        m_layer_lower = new QSpinBox(this);
        m_layer_lower->setMinimum(0);
        m_layer_lower->setValue(0);

        // Upper Layer Spinbox
        m_layer_upper = new QSpinBox(this);
        m_layer_upper->setMinimum(0);
        m_layer_upper->setValue(1);

        // Lock checkbox
        m_lock_layer = new QCheckBox(this);
        m_lock_layer->setText("Lock Layer Range");

        // Layer Sliders
        m_layer_lower_slider = new QSlider(Qt::Horizontal, this);
        m_layer_upper_slider = new QSlider(Qt::Horizontal, this);
        m_layer_upper_slider->setValue(1);

        // Layer Labels
        m_lower_label = new QLabel("Lower Layer:", this);
        m_upper_label = new QLabel("Upper Layer:", this);
    }

    void GcodeBar::setupLayouts() {
        // Main Layout
        m_layout = new QGridLayout(this);
        m_layout->setSpacing(6);
        m_layout->setContentsMargins(11, 11, 11, 11);
    }

    void GcodeBar::setupActions() {
    }

    void GcodeBar::setupInsert() {
        m_layout->addWidget(m_search_bar, 0, 0, 1, 5);
        m_layout->addWidget(m_refresh_btn, 0, 5, 1, 1);

        m_layout->addWidget(m_search_separator, 1, 0, 1, 5);

        m_layout->addWidget(m_view, 2, 0, 1, 5);

        m_layout->addWidget(m_view_sel, 3, 0, 1, 5);

        m_layout->addWidget(m_view_separator, 4, 0, 1, 5);

        m_layout->addWidget(m_hide_travel, 5, 0, 1, 1);
        m_layout->addWidget(m_hide_support, 5, 1, 1, 1);
        m_layout->addWidget(m_true_width, 5, 2, 1, 1);

        m_layout->addWidget(m_lower_label, 6, 0, 1, 1);
        m_layout->addWidget(m_layer_lower, 6, 1, 1, 1);
        m_layout->addWidget(m_layer_lower_slider, 6, 2, 1, 2);
        m_layout->addWidget(m_lock_layer, 6, 4, 2, 1);

        m_layout->addWidget(m_upper_label, 7, 0, 1, 1);
        m_layout->addWidget(m_layer_upper, 7, 1, 1, 1);
        m_layout->addWidget(m_layer_upper_slider, 7, 2, 1, 2);
    }

    void GcodeBar::setupEvents() {
        connect(m_layer_lower, QOverload<int>::of(&QSpinBox::valueChanged), this, &GcodeBar::updateLowerSpin);
        connect(m_layer_lower_slider, &QSlider::valueChanged, this, &GcodeBar::updateLowerSlider);
        connect(m_layer_upper, QOverload<int>::of(&QSpinBox::valueChanged), this, &GcodeBar::updateUpperSpin);
        connect(m_layer_upper_slider, &QSlider::valueChanged, this, &GcodeBar::updateUpperSlider);

        connect(this, &GcodeBar::lowerLayerUpdated, this, &GcodeBar::moveToLayer);
        connect(m_search_bar, &QLineEdit::returnPressed, this, &GcodeBar::search);
        connect(m_refresh_btn, &QPushButton::clicked, this, &GcodeBar::updateGCodeView);
        connect(m_view, &GcodeTextBoxWidget::lineChange, this, &GcodeBar::forwardLineChange);
        connect(m_view, &QPlainTextEdit::modificationChanged, this, &GcodeBar::updateRefreshButton);

        connect(m_hide_travel, &QCheckBox::clicked,
                [this] { emit forwardVisibilityChange(SegmentDisplayType::kTravel, m_hide_travel->isChecked());
                         PM->setHideTravelPreference(m_hide_travel->isChecked()); });

        connect(m_hide_support, &QCheckBox::clicked,
                [this] { emit forwardVisibilityChange(SegmentDisplayType::kSupport, m_hide_support->isChecked());
                         PM->setHideSupportPreference(m_hide_support->isChecked()); });

        connect(m_true_width, &QCheckBox::clicked,
                [this] { emit forwardSegmentWidthChange(m_true_width->isChecked());
                         PM->setUseTrueWidthsPreference(m_true_width->isChecked());});

        connect(m_lock_layer, &QCheckBox::clicked,
                [this] { if(m_lock_layer->isChecked()) m_lock_distance = m_layer_upper->value() - m_layer_lower->value(); });
    }

    void GcodeBar::updateGCodeView()
    {
        QDir appPath(QStandardPaths::writableLocation(QStandardPaths::AppDataLocation));
        QString tempFileName = appPath.filePath("gcode_output_temp");

        QTextDocumentWriter tempWriter;
        tempWriter.setFileName(tempFileName);
        tempWriter.setFormat("plaintext");
        tempWriter.write(m_view->document());

        //load the file. "false" is to prevent adjusting GCode based on minimum layer time
        emit refreshGCode(tempFileName, false);
    }

    void GcodeBar::search()
    {
        m_search_only_change = true;
        QString searchString = m_search_bar->text().trimmed();
        //perform search even if search string is empty - otherwise highlights from the previous search stay
        m_view->search(searchString, m_search_count);
        m_search_only_change = false;
        ++m_search_count;

        emit removeHighlight();
    }

    void GcodeBar::updateRefreshButton(bool change)
    {
        //Don't enable the refresh button if the change is for formatting resulted from a search
        //QPlainTextEdit::modificationChanged is for all types of modifications. No signal available for content change only
        if(!m_search_only_change)
            m_refresh_btn->setEnabled(change);
    }
}
