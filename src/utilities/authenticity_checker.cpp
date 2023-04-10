#include "utilities/authenticity_checker.h"
#include <QFile>
#include <qsettings.h>
#include <QMessageBox>
#include <QTimer>

#include "utilities/qt_json_conversion.h"

namespace ORNL
{
    AuthenticityChecker::AuthenticityChecker(QWidget* parent) : m_parent(parent)
    {
        QFile versions(":/configs/versions.conf");
        versions.open(QIODevice::ReadOnly);
        QString version_string = versions.readAll();
        fifojson version_data = fifojson::parse(version_string.toStdString());
        m_current_version = QString::fromStdString(version_data["slicer_2_version"]);
    }

    void AuthenticityChecker::startCheck()
    {
        checkRegistry();
    }

    void AuthenticityChecker::checkRegistry()
    {
        QSettings timeConfig("HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Services\\W32Time\\Config", QSettings::NativeFormat);
        QSettings timeParam("HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Services\\W32Time\\Parameters", QSettings::NativeFormat);
        QString syncType = timeParam.value("Type").toString();


        // We are setup to sync at some point
        if(syncType != "NoSync")
        {
            QString regVal = timeConfig.value("LastKnownGoodTime").toString();
            int windows_tick = 10000000;
            long long sec_to_unix_epoch = 11644473600LL;
            QDateTime lastGoodTime = QDateTime::fromTime_t((regVal.toULongLong() / windows_tick - sec_to_unix_epoch));
            QDateTime sysTime = QDateTime::currentDateTime();
            float hour = (sysTime.toSecsSinceEpoch() - lastGoodTime.toSecsSinceEpoch()) / 60.0 / 60.0;
            if(hour <= 24)
            {
                // Trust system time if synced at least once in the last 24 hours
                QDateTime compileTime = QDateTime::fromString(QString(APP_COMPILE_TIME), "MM-dd-yyyy");
                int daysElapsed = (sysTime.toSecsSinceEpoch() - compileTime.toSecsSinceEpoch()) / 86400;
                if(daysElapsed > 90)
                {
                    QString msg = "The time period for " + m_current_version + " has expired.  Please contact Alex Roschli (roschliac@ornl.gov) or Michael Borish (borishmc@ornl.gov) for a newer version.";
                    if(m_parent == nullptr)
                        qInfo() << msg;
                    else
                        QMessageBox::critical(m_parent, "ORNL Slicer 2 - Trial Expired", msg);

                    emit done(false);
                }
                else
                {
                    if(daysElapsed > 76)
                    {
                        QString msg = "This version of Slicer 2 will expire in " + QString::number(90 - daysElapsed) + " days.  Please contact Alex Roschli (roschliac@ornl.gov) or Michael Borish (borishmc@ornl.gov) for a newer version.";
                        if(m_parent == nullptr)
                            qInfo() << msg;
                        else
                            QMessageBox::critical(m_parent, "ORNL Slicer 2 - Trial Ending Soon", msg);
                    }
                    emit done(true);
                }
            }
            else
            {
                //try to go check NIST's NTP server cluster
                startNISTCheck();
            }
        }
        else
            startNISTCheck();
    }

    void AuthenticityChecker::startNISTCheck()
    {
        m_upd_socket = new QUdpSocket(this);

        QAbstractSocket::connect(m_upd_socket, &QUdpSocket::readyRead, [this] (){

            QByteArray newTime;
            newTime = m_upd_socket->readAll();

            if(newTime.size() == 48)
            {
                unsigned long highWord = ((uchar)newTime[40] << 8) + (uchar)newTime[41];
                unsigned long lowWord = ((uchar)newTime[42] << 8) + (uchar)newTime[43];

                // combine the four bytes (two words) into a long integer
                // this is NTP time (seconds since Jan 1 1900):
                unsigned long secsSince1900 = highWord << 16 | lowWord;
                const unsigned long seventyYears = 2208988800UL;

                // subtract seventy years:
                unsigned long epoch = secsSince1900 - seventyYears;
                m_nist_time = QDateTime::fromSecsSinceEpoch(epoch);
            }
            nistCheckDone();
        });

        QAbstractSocket::connect(m_upd_socket, &QUdpSocket::connected, [this] () {  QByteArray timeRequest(48, 0);
                                                                   timeRequest[0] = '\x23';
                                                                   m_upd_socket->write(timeRequest);
        });

        m_upd_socket->connectToHost("time.nist.gov", 123);

        m_duration_timeout = new QTimer();
        QAbstractSocket::connect(m_duration_timeout, &QTimer::timeout, this, &AuthenticityChecker::nistCheckDone);
        m_duration_timeout->setInterval(11000);
        m_duration_timeout->start();
    }

    void AuthenticityChecker::nistCheckDone()
    {
        m_duration_timeout->stop();
        m_upd_socket->abort();

        if(m_nist_time.isValid())
        {
            QDateTime compileTime = QDateTime::fromString(QString(APP_COMPILE_TIME), "MM-dd-yyyy");
            int daysElapsed = (m_nist_time.toSecsSinceEpoch() - compileTime.toSecsSinceEpoch()) / 86400;
            if(daysElapsed > 90)
            {
                QString msg = "The time period for " + m_current_version + " has expired.  Please contact Alex Roschli (roschliac@ornl.gov) or Michael Borish (borishmc@ornl.gov) for a newer version.";
                if(m_parent == nullptr)
                    qInfo() << msg;
                else
                    QMessageBox::critical(m_parent, "ORNL Slicer 2 - Trial Expired", msg);

                emit done(false);
            }
            else
            {
                if(daysElapsed > 76)
                {
                    QString msg = "This version of Slicer 2 will expire in " + QString::number(90 - daysElapsed) + " days.  Please contact Alex Roschli (roschliac@ornl.gov) or Michael Borish (borishmc@ornl.gov) for a newer version.";
                    if(m_parent == nullptr)
                        qInfo() << msg;
                    else
                        QMessageBox::critical(m_parent, "ORNL Slicer 2 - Trial Ending Soon", msg);
                }
               emit done(true);
            }
        }
        else
        {
            //either error, timeout, or no network connection
            //windows is also on manual, which means it cannot sync
            QString msg = m_current_version + " could not verify critical information about this PC.  Please contact Alex Roschli (roschliac@ornl.gov) or Michael Borish (borishmc@ornl.gov) for assistance.";
            if(m_parent == nullptr)
                qInfo() << msg;
            else
                QMessageBox::critical(m_parent, "ORNL Slicer 2 - Critical Verification", msg);

            emit done(false);
        }
    }
}

