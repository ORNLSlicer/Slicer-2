#include "windows/remote_connectivity.h"

#include <QTabWidget>
#include <QTableWidget>
#include <QHeaderView>
#include <QLabel>
#include <QGridLayout>
#include <QFile>
#include <QFileDialog>
#include <QCoreApplication>
#include <QPushButton>
#include <QComboBox>
#include <QCheckBox>
#include <QMessageBox>
#include <QStringBuilder>

#include "utilities/enums.h"
#include "managers/preferences_manager.h"

namespace ORNL
{

    RemoteConnectivity::RemoteConnectivity(QWidget* parent) : QTabWidget()
    {
        setWindowTitle("Slicer 2: Remote Connectivity");

        m_parent = parent;
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/slicer2.pnh"), QSize(), QIcon::Normal, QIcon::Off);
        setWindowIcon(icon);

        createTCPTab();
        createHTTPTab();
    }

    void RemoteConnectivity::createTCPTab()
    {
        QWidget* tcpServer = new QWidget();
        QVBoxLayout* tcpLayout = new QVBoxLayout();
        tcpServer->setLayout(tcpLayout);

        QGroupBox *groupBox = new QGroupBox("Major Slicer 2 Steps");
        QVBoxLayout *vbox = new QVBoxLayout();
        groupBox->setLayout(vbox);

        QCheckBox* m_pre_processing_checkbox = new QCheckBox("After Pre-Processing");
        m_pre_processing_checkbox->setChecked(PM->getStepConnectivity(StatusUpdateStepType::kPreProcess));
        connect(m_pre_processing_checkbox, &QCheckBox::toggled,
                [this] (bool toggle) { PM->setStepConnectivity(StatusUpdateStepType::kPreProcess, toggle);
                                       emit this->setStepConnectivity(StatusUpdateStepType::kPreProcess, toggle); });

        QCheckBox* m_computation_checkbox = new QCheckBox("After Computation");
        setStepConnectivity(StatusUpdateStepType::kCompute, m_computation_checkbox->isChecked());
        connect(m_computation_checkbox, &QCheckBox::toggled,
                [this] (bool toggle) { PM->setStepConnectivity(StatusUpdateStepType::kCompute, toggle);
                                       emit this->setStepConnectivity(StatusUpdateStepType::kCompute, toggle); });

        QCheckBox* m_post_processing_checkbox = new QCheckBox("After Post-Processing");
        setStepConnectivity(StatusUpdateStepType::kPostProcess, m_post_processing_checkbox->isChecked());
        connect(m_post_processing_checkbox, &QCheckBox::toggled,
                [this] (bool toggle) { PM->setStepConnectivity(StatusUpdateStepType::kPostProcess, toggle);
                                       emit this->setStepConnectivity(StatusUpdateStepType::kPostProcess, toggle); });

        QCheckBox* m_gcode_processing_checkbox = new QCheckBox("After Gcode Generation");
        setStepConnectivity(StatusUpdateStepType::kGcodeGeneraton, m_gcode_processing_checkbox->isChecked());
        connect(m_gcode_processing_checkbox, &QCheckBox::toggled,
                [this] (bool toggle) { PM->setStepConnectivity(StatusUpdateStepType::kGcodeGeneraton, toggle);
                                       emit this->setStepConnectivity(StatusUpdateStepType::kGcodeGeneraton, toggle); });

        QCheckBox* m_gcode_parsing_checkbox = new QCheckBox("After Gcode Parsing");
        setStepConnectivity(StatusUpdateStepType::kGcodeParsing, m_gcode_parsing_checkbox->isChecked());
        connect(m_gcode_parsing_checkbox, &QCheckBox::toggled,
                [this] (bool toggle) { PM->setStepConnectivity(StatusUpdateStepType::kGcodeParsing, toggle);
                                       emit this->setStepConnectivity(StatusUpdateStepType::kGcodeParsing, toggle); });

        vbox->addWidget(m_pre_processing_checkbox);
        vbox->addWidget(m_computation_checkbox);
        vbox->addWidget(m_post_processing_checkbox);
        vbox->addWidget(m_gcode_processing_checkbox);
        vbox->addWidget(m_gcode_parsing_checkbox);

        tcpLayout->addWidget(groupBox);

        QGroupBox *groupBox2 = new QGroupBox("Connection Information:");
        QVBoxLayout *vbox2 = new QVBoxLayout();
        QHBoxLayout *hbox = new QHBoxLayout();
        groupBox2->setLayout(vbox2);

        QSpinBox* m_tcp_port_box = new QSpinBox();
        m_tcp_port_box->setMinimum(1);
        m_tcp_port_box->setMaximum(65535);
        m_tcp_port_box->setValue(PM->getTCPServerPort());
        connect(m_tcp_port_box, QOverload<int>::of(&QSpinBox::valueChanged), [this] (int val) { PM->setTCPServerPort(val); });

        hbox->addWidget(new QLabel("Port: "));
        hbox->addWidget(m_tcp_port_box);
        hbox->addStretch();
        vbox2->addLayout(hbox);

        QPushButton* m_tcp_server_restart_button = new QPushButton("Start/Restart Server");
        connect(m_tcp_server_restart_button, &QPushButton::pressed, [=] () { emit this->restartTcpServer(m_tcp_port_box->value());});
        vbox2->addWidget(m_tcp_server_restart_button);

        QCheckBox* m_tcp_autostart_checkbox = new QCheckBox("Automatically start server on startup");
        m_tcp_autostart_checkbox->setChecked(PM->getTcpServerAutoStart());
        connect(m_tcp_autostart_checkbox, &QCheckBox::clicked, [this] (bool checked) { PM->setTcpServerAutoStart(checked);});
        vbox2->addWidget(m_tcp_autostart_checkbox);

        tcpLayout->addWidget(groupBox2);

        this->addTab(tcpServer, "TCP Server Config");
    }

    void RemoteConnectivity::createHTTPTab()
    {
        QWidget* httpServer = new QWidget();
        QGridLayout* gridLayout = new QGridLayout();
        httpServer->setLayout(gridLayout);

        gridLayout->addWidget(new QLabel("Current Configuration"), 0, 0, 1, 3, Qt::AlignCenter);

        m_table = new QTableWidget();
        m_table->setSelectionBehavior(QAbstractItemView::SelectRows);
        m_table->setSelectionMode(QAbstractItemView::SingleSelection);
        m_table->setDragEnabled(false);
        m_table->insertColumn(0);
        m_table->insertColumn(1);
        m_table->insertColumn(2);
        m_table->verticalHeader()->hide();
        m_table->setAcceptDrops(false);
        m_table->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        m_table->horizontalHeader()->setHighlightSections(false);

        gridLayout->addWidget(m_table, 1, 0, 1, 3, Qt::AlignCenter);

        gridLayout->addWidget(new QLabel("Current port: "), 2, 0, 1, 1, Qt::AlignCenter);
        m_port_box = new QSpinBox();
        m_port_box->setMinimum(1);
        m_port_box->setMaximum(65535);

        gridLayout->addWidget(m_port_box, 2, 1);

        m_restart_button = new QPushButton("Restart/Update server");
        connect(m_restart_button, &QPushButton::pressed, [this] () { emit this->restartServer(createJsonConfig());});

        gridLayout->addWidget(m_restart_button, 3, 0);

        QPushButton* loadConfigButton = new QPushButton("Load Config");
        connect(loadConfigButton, &QPushButton::pressed, [this] () { setConfig(
                        QFileDialog::getOpenFileName(this, "Load Config", QCoreApplication::applicationDirPath() % "/http_config", "Config Files (*.config)"));});

        gridLayout->addWidget(loadConfigButton, 3, 1);

        QPushButton* m_save = new QPushButton("Save config");
        connect(m_save, &QPushButton::pressed, this, &RemoteConnectivity::saveConfig);

        gridLayout->addWidget(m_save, 3, 2);

        this->addTab(httpServer, "Http Server Config");
    }

    RemoteConnectivity::~RemoteConnectivity()
    {

    }

    void RemoteConnectivity::setConfig(QString filename)
    {
        if(filename != "")
        {
            m_table->clear();

            QFile file(filename);
            file.open(QIODevice::ReadOnly);
            QString settings = file.readAll();
            fifojson j = fifojson::parse(settings.toStdString());

            m_table->setHorizontalHeaderLabels(QStringList { "Identifier", "End Point Type", "Physical End Point"} );
            int i = 0;
            for (auto& el : j.at("endpoints").items())
            {
                for(auto& endpoint : el.value().items())
                {
                    m_table->insertRow(i);

                    QComboBox* combo = new QComboBox();
                    combo->addItem("status");
                    combo->addItem("api");
                    combo->addItem("generateUploadId");
                    combo->addItem("uploadSTL");

                    QString key = QString::fromStdString(endpoint.key());
                    combo->setCurrentIndex(combo->findText(key));
                    m_table->setCellWidget(i, 0, combo);

                    QVector<QString> endTypes = endpoint.value()["end_type"];
                    QWidget* checkboxWidget = new QWidget();
                    QHBoxLayout* checkboxLayout = new QHBoxLayout(checkboxWidget);
                    QCheckBox* getBox = new QCheckBox("GET");
                    if(endTypes.contains("GET"))
                        getBox->setChecked(true);

                    QCheckBox* postBox = new QCheckBox("POST");
                    if(endTypes.contains("POST"))
                        postBox->setChecked(true);

                    QCheckBox* putBox = new QCheckBox("PUT");
                    if(endTypes.contains("PUT"))
                        putBox->setChecked(true);

                    checkboxLayout->addWidget(getBox);
                    checkboxLayout->addWidget(postBox);
                    checkboxLayout->addWidget(putBox);

                    m_table->setCellWidget(i, 1, checkboxWidget);

                    QString physicalEnd = endpoint.value()["physical_end"];
                    m_table->setItem(i, 2, new QTableWidgetItem(physicalEnd));

                    ++i;
                }
            }
            m_port_box->setValue(j["port"]);
            m_table->resizeColumnsToContents();
            m_table->resizeRowsToContents();

            //Width of table
            int iWidth = 0;
            for (int i = 0; i < m_table->columnCount(); ++i)
            {
                iWidth += m_table->horizontalHeader()->sectionSize(i);
            }
            //+ 2 to include boundary lines as headers do not account for them
            iWidth += m_table->verticalHeader()->width() + 2;
            m_table->setMinimumWidth(iWidth);

            //Height of table
            int iHeight = 0;
            for (int i = 0; i < m_table->rowCount(); ++i)
            {
                iHeight += m_table->verticalHeader()->sectionSize(i);
            }
            //+ 2 to include boundary lines as headers do not account for them
            iHeight += m_table->horizontalHeader()->height() + 2;

            m_table->setMinimumSize(QSize(iWidth + 2, iHeight + 2));
        }
    }

    void RemoteConnectivity::saveConfig()
    {
        QString filename = QFileDialog::getSaveFileName(this, "Save Config",
                                                        QCoreApplication::applicationDirPath() % "/http_config", "Config Files (*.config)");
        if(filename != QString())
        {
            fifojson config = createJsonConfig();
            QFile file(filename);
            file.open(QIODevice::WriteOnly | QIODevice::Truncate);
            file.write(config.dump().c_str());
            file.close();
        }
    }

    fifojson RemoteConnectivity::createJsonConfig()
    {
        fifojson j;
        j["port"] = m_port_box->value();
        fifojson array = fifojson::array();
        int rows = m_table->rowCount();
        for(int i = 0; i < rows; ++i)
        {
            QList<QCheckBox*> checkboxes = m_table->cellWidget(i, 1)->findChildren<QCheckBox*>();
            fifojson endpointTypes = fifojson::array();
            for(QCheckBox* check : checkboxes)
            {
                if(check->isChecked())
                    endpointTypes.push_back(check->text());
            }

            fifojson endPointEntry;
            endPointEntry[static_cast<QComboBox*>(m_table->cellWidget(i, 0))->currentText().toStdString()] = {
            {"end_type", endpointTypes}, {"physical_end", m_table->item(i, 2)->text().toStdString()}};

            array.push_back(endPointEntry);
        }
        j["endpoints"] = array;
        return j;
    }

    void RemoteConnectivity::restartComplete(bool success)
    {
        QString message;
        if(success)
            message = "Server restarted successfully";
        else
            message = "Server restart failed";

        QMessageBox::information(this, "Server Restart", message);
    }

    void RemoteConnectivity::closeEvent(QCloseEvent *event)
    {
        //set the focus back to the main window
        if(m_parent->isMinimized())
        {
            m_parent->showNormal();
        }
        //setFocus() doesn't deliver, but activateWindow() serves the purpose
        m_parent->activateWindow();
    }
}
