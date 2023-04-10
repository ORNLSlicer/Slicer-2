#include "windows/dialogs/template_layer_save.h"
#include "managers/session_manager.h"

namespace ORNL{

    TemplateLayerDialog::TemplateLayerDialog(QWidget* parent): QDialog(parent), m_filename(QDir::homePath() + "/layers.s2l")
        {
        this->setupUi();
    }

    TemplateLayerDialog::TemplateLayerDialog(QWidget* parent, QVector<SettingsRange> selected_template, QString template_name): QDialog(parent), m_filename(QDir::homePath() + "/" + template_name + ".s2l")
        {  //Layerbar template file format is s2l. Located in layerbartemplates folder.
        this->setupUi();
        m_settings_ranges = selected_template; //load in ranges from selected template
        for(int i = 0, end = m_settings_ranges.size(); i < end; ++i){
            if(m_settings_ranges[i].low() == m_settings_ranges[i].high()){
                std::string whole_layer = "Layer " + std::to_string(m_settings_ranges[i].low()+1);
                QString new_layer = QString::fromStdString(whole_layer);
                m_ranges_list->addItem(new_layer);
            }
            else {
                int min = std::min(m_settings_ranges[i].low(), m_settings_ranges[i].high()), max=std::max(m_settings_ranges[i].low(), m_settings_ranges[i].high());
                std::string whole_layer = "Layers " + std::to_string(min+1) + " - " +std::to_string(max+1);
                QString new_layer = QString::fromStdString(whole_layer);
                m_ranges_list->addItem(new_layer);
            }
        }
    }

    void TemplateLayerDialog::setupUi() {
        this->setupWindow();
        this->setupWidgets();
        this->setupLayouts();
        this->setupEvents();
    }

    void TemplateLayerDialog::setupWindow() {
        this->setWindowTitle("Edit/Create Layer Bar Template");
        this->setMinimumSize(900, 900);
    }

    void TemplateLayerDialog::setupWidgets() {
        m_new = new QPushButton("New");
        m_delete = new QPushButton("Delete");
        m_save = new QPushButton("Save");
        m_cancel = new QPushButton("Cancel");
        m_layout = new QHBoxLayout(this);
        m_save_label = new QLabel(this);
        m_save_label->setText("Save to:");
        m_left_buttons = new QHBoxLayout();
        m_right_buttons = new QHBoxLayout();
        m_leftside = new QVBoxLayout();
        m_rightside = new QVBoxLayout();
        m_ranges_list = new QListWidget();
        m_range_label = new QLabel();
        m_edit = new QLineEdit(this);
        m_edit->setText(m_filename);
        m_edit->setReadOnly(true);
        m_browse_btn = new QToolButton(this);
        m_browse_btn->setText("...");
        m_button_container = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel);
        m_save = m_button_container->button(QDialogButtonBox::Save);
        m_cancel = m_button_container->button(QDialogButtonBox::Cancel);
    }

    void TemplateLayerDialog::setupLayouts() {
        m_left_buttons->addWidget(m_new);
        m_left_buttons->addWidget(m_delete);
        m_range_label->setText("Ranges");
        m_leftside->addWidget(m_range_label);
        m_leftside->addWidget(m_ranges_list);
        m_leftside->addLayout(m_left_buttons);
        m_layout->addLayout(m_leftside);
        addSettingBar();
        m_right_buttons->addWidget(m_save_label);
        m_right_buttons->addWidget(m_edit);
        m_right_buttons->addWidget(m_browse_btn);
        m_rightside->addLayout(m_right_buttons);
        m_rightside->addWidget(m_button_container);
        m_layout->addLayout(m_leftside);
        m_layout->addLayout(m_rightside);
    }

    void TemplateLayerDialog::setupEvents() {
        connect(m_save, &QPushButton::pressed, this, &TemplateLayerDialog::saveLayerTemplate);
        connect(m_cancel, &QPushButton::pressed, this, &TemplateLayerDialog::reject);
        connect(m_new, &QPushButton::pressed, this, &TemplateLayerDialog::addNewLayer);
        connect(m_delete, &QPushButton::pressed, this, &TemplateLayerDialog::deleteLayer);
        connect(m_browse_btn, &QToolButton::clicked, this, &TemplateLayerDialog::fileDialog);
        connect(m_ranges_list, &QListWidget::itemClicked, this, &TemplateLayerDialog::setCurrentLayerSettings);

    }

    void TemplateLayerDialog::fileDialog() {
        QFileDialog save_dialog;
        save_dialog.setWindowTitle("Save Template Location");
        save_dialog.setAcceptMode(QFileDialog::AcceptSave);
        save_dialog.setNameFilters(QStringList() << "Slicer 2 Configuration/Template File (*.s2l)" << "Any Files (*)");
        save_dialog.setDirectory(m_filename);
        save_dialog.setDefaultSuffix("s2l");

        if (!save_dialog.exec()) return;

        m_filename = save_dialog.selectedFiles().first();
        m_edit->setText(m_filename);
    }

    void TemplateLayerDialog::addNewLayer(){
        QDialog dialog(this);
        QFormLayout form(&dialog);
        dialog.setWindowTitle("Add New Range of Layer Settings");

        // Input for first and last in range
        QDoubleSpinBox* pairFirst = new QDoubleSpinBox(&dialog); //start of range
        QDoubleSpinBox* pairLast = new QDoubleSpinBox(&dialog); //end of range
        pairFirst->setRange(1, 1000);
        pairLast->setRange(1, 1000);
        pairFirst->setDecimals(0);
        pairLast->setDecimals(0);
        QGroupBox* groupBox = new QGroupBox(tr("Indicate which layers to pair"));
        QGridLayout* grid = new QGridLayout;
        grid->addWidget(new QLabel("Begin layer:"), 1, 1);
        grid->addWidget(new QLabel("End layer:"), 2, 1);
        grid->addWidget(pairFirst, 1, 2);
        grid->addWidget(pairLast, 2, 2);
        grid->setColumnStretch(2, 2);
        groupBox->setLayout(grid);
        groupBox->setEnabled(true);
        form.addRow(groupBox);
        QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
            Qt::Horizontal, &dialog);
        form.addRow(&buttonBox);
        QObject::connect(&buttonBox, SIGNAL(accepted()), &dialog, SLOT(accept()));
        QObject::connect(&buttonBox, SIGNAL(rejected()), &dialog, SLOT(reject()));
        if (dialog.exec() == QDialog::Accepted){
            int first = pairFirst->value() - 1; //index is 1 less than displayed layer number
            int last = pairLast->value() - 1;
            bool found = false;
            for(int i = 0, end = m_settings_ranges.size(); i < end; ++i){
                //Check if range already exists. If user enters higher layer number as low, reverse them.
                if((m_settings_ranges[i].low() == first && m_settings_ranges[i].high() == last) || (m_settings_ranges[i].low() == last&&m_settings_ranges[i].high() == first)){
                    found = true;
                    QMessageBox error;
                    error.setText("Range already exists.");
                    error.exec();
                    break;
                }
            }
            if(!found){
                if(first == last){ //If single layer
                    SettingsRange new_range(first, last, "");
                    m_settings_ranges.push_back(new_range);
                    std::sort(m_settings_ranges.begin(), m_settings_ranges.end(), sortByMin); //sort after inserting new layer
                    int index = findRange(new_range); //find index after sort
                    std::string whole_layer = "Layer " + std::to_string(first+1);
                    QString new_layer = QString::fromStdString(whole_layer);
                    m_ranges_list->insertItem(index, new_layer); //insert into list at same index as in array
                }
                else { //if range
                    int min = std::min(first, last), max = std::max(first, last); //if higher number entered as low, reverse them
                    SettingsRange new_range(min, max, "");
                    m_settings_ranges.push_back(new_range);
                    std::sort(m_settings_ranges.begin(), m_settings_ranges.end(), sortByMin);
                    int index = findRange(new_range);
                    std::string whole_layer = "Layers " + std::to_string(min+1) + " - " + std::to_string(max + 1);
                    QString new_layer = QString::fromStdString(whole_layer);
                    m_ranges_list->insertItem(index, new_layer);
                }
            }
       }
        dialog.close();
    }

    void TemplateLayerDialog::deleteLayer(){
        int index = m_ranges_list->currentRow();
        m_settings_ranges.remove(index); //delete from both array and list
        QListWidgetItem *it = m_ranges_list->takeItem(index);
        delete it;
    }

    void TemplateLayerDialog::addSettingBar(){
        settingbar = new SettingBar(CSM->getMostRecentSettingHistory());
        settingbar->setEnabled(false);
        m_rightside->addWidget(settingbar);
    }

    void TemplateLayerDialog::setCurrentLayerSettings(QListWidgetItem* item){
        settingbar->setEnabled(true);
        int index = m_ranges_list->currentRow(); //get currently selected row
        QString label_name = item->text();
        QList<QSharedPointer<SettingsBase>> list_of_settings;
        QSharedPointer<SettingsBase> current_layer_settings = m_settings_ranges[index].getSb();
        list_of_settings.append(current_layer_settings);
        QPair<QString, QList<QSharedPointer<SettingsBase>>> name_with_settings = qMakePair(label_name, list_of_settings);
        settingbar->settingsBasesSelected(name_with_settings);
    }

      bool TemplateLayerDialog::sortByMin(SettingsRange& a, SettingsRange& b){
        if(a.low() == b.low()){ //if lows equal, sort by high
            return a.high() < b.high();
        }
        else{  //othewise sort by low
            return a.low() < b.low();
        }
    }

    int TemplateLayerDialog::findRange(SettingsRange new_range){
        //find index of range;
        for(int i = 0, end = m_settings_ranges.size(); i < end; ++i){
            if(m_settings_ranges[i].low() == new_range.low()&&m_settings_ranges[i].high() == new_range.high()){
                return i;
            }
        }
        return -1;
    }

    void TemplateLayerDialog::saveLayerTemplate(){
        fifojson j_array = fifojson::array({});
        QFile file(m_filename);
        QString file_path = file.fileName();
        file.open(QIODevice::WriteOnly| QIODevice::Text);
        //Loop through ranges array and load each object into json
        for(int i = 0, end = m_settings_ranges.size(); i < end; ++i){
            fifojson j;
            j["low"] = m_settings_ranges[i].low();
            j["high"] = m_settings_ranges[i].high();
            j["settings"] = m_settings_ranges[i].getSb()->json();
            j_array.push_back(j);
        }
        file.write(j_array.dump(4).c_str());
        file.close();
        GSM->loadGlobalLayerBarTemplate(file_path, true); //Once file created, load all templates to drop down
        close();
    }

}


