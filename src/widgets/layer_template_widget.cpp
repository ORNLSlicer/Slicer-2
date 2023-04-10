#include "widgets/layer_template_widget.h"

namespace ORNL
{
    LayerTemplateWidget::LayerTemplateWidget(QWidget *parent)
        : QWidget{parent}
    {
        m_most_recently_edited_template = "<no template selected>";
        this->setupUi();
    }

    void LayerTemplateWidget::setupUi() {
        this->setupWindow();
        this->setupWidgets();
        this->setupLayouts();
        this->setupEvents();
        loadLayerBar();
    }

    void LayerTemplateWidget::setupWindow() {
        this->setWindowTitle("Set/Edit Layer Bar Template");
    }

    void LayerTemplateWidget::setupWidgets(){
        m_layout = new QVBoxLayout(this);
        m_search = new QLineEdit();
        m_search->setPlaceholderText("Search for setting...");
        m_templates = new QComboBox();
        m_current_folder = new QLabel();
        m_current_folder->setText("Currently searching in : <Not Set> for additional setting files");
        m_buttons = new QHBoxLayout;
        m_new = new QPushButton("New");
        m_edit = new QPushButton("Edit");
    }

    void LayerTemplateWidget::setupLayouts(){
        m_buttons->addWidget(m_new);
        m_buttons->addWidget(m_edit);
        m_layout->addWidget(m_search);
        m_layout->addWidget(m_templates);
        m_layout->addWidget(m_current_folder);
        m_layout->addStretch();
        m_layout->addLayout(m_buttons);
    }

    void LayerTemplateWidget::setupEvents(){
        connect(m_templates, &QComboBox::currentTextChanged, this, &LayerTemplateWidget::changeLayerBarSettings );
        connect(m_new, &QPushButton::pressed, this, &LayerTemplateWidget::addNewTemplate);
        connect(m_edit, &QPushButton::pressed, this, &LayerTemplateWidget::editTemplate);        
    }

    void LayerTemplateWidget::setCurrentFolder(QString path){
        m_current_folder->setText("Currently searching in: " % path % " for additional setting files");
    }

    void LayerTemplateWidget::loadLayerBar(){
        m_templates->clear(); //clear existing items from drop down and load all in template from settings manager
        m_templates->addItem("<no template selected>");  //If no template being applied. User manually enters all layers.
        QMap<QString, QVector<SettingsRange>> layer_bar_template = GSM->getAllLayerBarTemplates();
        for(auto& key:layer_bar_template.keys()){
            m_templates->addItem(key);
            }
        m_templates->setCurrentText(m_most_recently_edited_template);
    }

    void LayerTemplateWidget::changeLayerBarSettings(QString layer_template){
        if(layer_template == "<no template selected>"){ //Clear everything if <no template selected>.
            GSM->setCurrentTemplate(layer_template);
            GSM->clearTemplate();
            emit layerbarTemplateChanged();  //signal to clear deleted layers in LayerBar
            return;
            }
        QMap<QString, QVector<SettingsRange>> layer_bar_template = GSM->getAllLayerBarTemplates();
        //use name of template from drop down to find selected template in settings manager.
        GSM->setCurrentTemplate(layer_template);
        GSM->loadLayerSettingsFromTemplate(layer_bar_template[layer_template]);
        emit layerbarTemplateChanged();  //signal to apply new template.
    }

    void LayerTemplateWidget::addNewTemplate(){
        TemplateLayerDialog dialog;
        if (!dialog.exec()) return;
    }

    void LayerTemplateWidget::editTemplate(){
        m_most_recently_edited_template = m_templates->currentText();
        QMap<QString, QVector<SettingsRange>> layer_bar_template = GSM->getAllLayerBarTemplates();
        //Find selected template in settings manager. Open dialog to edit and save.
        QVector<SettingsRange> current_template = layer_bar_template[m_templates->currentText()];
        TemplateLayerDialog dialog(nullptr, current_template, m_templates->currentText());
        if (!dialog.exec()) return;
    }

}
