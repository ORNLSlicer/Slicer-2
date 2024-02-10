#include "windows/about.h"

#include <QGridLayout>
#include <QLabel>
#include <QDesktopServices>
#include <QApplication>
#include <QUrl>
#include <QIcon>
#include <QFile>

#include "managers/gpu_manager.h"
#include "utilities/qt_json_conversion.h"
#include "units/unit.h"

namespace ORNL
{
    AboutWindow::AboutWindow(QWidget *parent) : QWidget()
    {
        //make it behave the same as "About Qt" window: The window is modal to the application and blocks input to all windows
        setWindowModality(Qt::ApplicationModal);

        setWindowTitle("About ORNL Slicer 2");
        setWindowIcon(QIcon(":/icons/slicer2.png"));

        QGridLayout *layout = new QGridLayout();

        QLabel *mdf = new QLabel();
        QPixmap mdfIcon(":/icons/mdf_color.png");
        mdf->setPixmap(mdfIcon.scaled(450, 100, Qt::KeepAspectRatio));
        layout->addWidget(mdf, 0, 0, 1, 2);

        QFont f = QApplication::font();
        f.setPointSize(10);

        QFile versions(":/configs/versions.conf");
        versions.open(QIODevice::ReadOnly);
        QString version_string = versions.readAll();
        fifojson version_data = json::parse(version_string.toStdString());
        QString version = version_data["slicer_2_version"];

        QLabel *lblVersion = new QLabel(version);
        lblVersion->setFont(f);
        layout->addWidget(lblVersion, 1, 0, 1, 2, Qt::AlignCenter);

        layout->addWidget(new QLabel("Compiled on " + QString(APP_COMPILE_TIME) + ((GPU->hasSupport()) ? " with GPU support" : " without GPU support")), 2, 0, 1, 2, Qt::AlignCenter);

        QGridLayout *gpu_layout = new QGridLayout();
        if(GPU->hasSupport())
        {
            #ifdef NVCC_FOUND
            if(GPU->count() > 0)
            {
                auto gpu_label = new QLabel("Compatible GPUs Detected:");
                QFont font = gpu_label->font();
                font.setWeight(QFont::Bold);
                gpu_label->setFont(font);
                gpu_layout->addWidget(gpu_label, 0, 0, 1, 2, Qt::AlignCenter);
                auto device_ids = GPU->getDeviceIds();
                for (int i = 0, end = GPU->count(); i < end; ++i)
                {
                    auto info = GPU->getDeviceInformation(device_ids[i]);
                    gpu_layout->addWidget(new QLabel(QString(info.name) + " with compute version " + QString::number(info.major) + "." + QString::number(info.minor)), i + 1, 0, 1, 2, Qt::AlignCenter);
                }

            }else
                gpu_layout->addWidget(new QLabel("No compatible GPU detected"), 0, 0, 1, 2, Qt::AlignCenter);
            #endif
        }

        layout->addLayout(gpu_layout, 3, 0, 1, 2, Qt::AlignCenter);

        QLabel *lblContact = new QLabel("Questions or comments? Email us at: <a href='mailto:borishmc@ornl.gov'>borishmc@ornl.gov</a> or "
                                        "<a href='mailto:roschliac@ornl.gov'>roschliac@ornl.gov</a>");
        connect(lblContact, &QLabel::linkActivated, [](const QString &link) { QDesktopServices::openUrl(QUrl(link));});
        layout->addWidget(lblContact, 4, 0, 1, 2, Qt::AlignCenter);

        QLabel *bugReport = new QLabel("<a href='https://github.com/ORNLSlicer/Slicer-2/issues'>Bug report?</a>");
        connect(bugReport, &QLabel::linkActivated, [](const QString &link) { QDesktopServices::openUrl(QUrl(link));});
        layout->addWidget(bugReport, 6, 0, 1, 2, Qt::AlignCenter);

        layout->addWidget(new QLabel("Copyright © 2024"), 7, 0, 1, 2, Qt::AlignCenter);

        this->setLayout(layout);
    }

    AboutWindow::~AboutWindow()
    {

    }
}
