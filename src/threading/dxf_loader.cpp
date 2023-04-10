#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <QStringBuilder>
#include <QDebug>
#include <QDateTime>

#include "threading/dxf_loader.h"
#include "managers/settings/settings_manager.h"
#include "utilities/mathutils.h"

#include "gcode/parsers/sheet_lamination_parser.h"

namespace ORNL
{
    DXFLoader::DXFLoader(QString filename, bool alterFile) :
        m_filename(filename)
        , m_adjust_file(alterFile)
        , m_should_cancel(false)
    {

    }

    void DXFLoader::run()
    {
        m_lines.clear();
        m_original_lines.clear();

        bool disableVisualization = GSM->getGlobal()->setting<bool>(Constants::ExperimentalSettings::GcodeVisualization::kDisableVisualization);
        int layerSkip = GSM->getGlobal()->setting<int>(Constants::ExperimentalSettings::GcodeVisualization::kVisualizationSkip);

        if(!m_filename.isEmpty() && (!disableVisualization || m_adjust_file)) {
            //Try-catch is necessary to prevent a crash when the GCode refresh button is clicked after an erroneous modification
            try {
            //read in entire file and separate into lines
            QString text;
            QFile inputFile(m_filename);
            if (inputFile.open(QIODevice::ReadOnly | QIODevice::Text))
            {
                QTextStream in(&inputFile);
                text = in.readAll();
                m_original_lines = text.split("\n");
                m_lines = text.toUpper().split("\n");
                inputFile.close();
            }
            else
            {
                qDebug() << "Error: " << m_filename << " is not readable!";
                return;
            }

            Time min_time(0), max_time(0), total_time(0), total_adjusted_time(0);
            Time min_layer_time = 0, max_layer_time = 0;
            if (m_adjust_file && GSM->getGlobal()->setting< int >(Constants::MaterialSettings::Cooling::kForceMinLayerTime)){
                min_layer_time = GSM->getGlobal()->setting<Time>(Constants::MaterialSettings::Cooling::kMinLayerTime)();
                max_layer_time = GSM->getGlobal()->setting<Time>(Constants::MaterialSettings::Cooling::kMaxLayerTime)();
            }

            QString weightInfo = "No statistics calculated";
            //parse header looking for syntax
            m_selected_meta = GcodeMetaList::SheetLaminationMeta;
            if(!disableVisualization)
            {
                // At this stage in the gcode_loader.cpp, signals from the common parser were connected to coresponding slots in the loader
                // However, since we're not using the common parser, we'll just emit these signals at some other time
                emit forwardInfoToBuildExportWindow(m_filename, m_selected_meta);

                QVector<QVector<QSharedPointer<SegmentBase>>> layers = m_parser->parse(m_original_lines);
                emit dxfLoadedVisualization(layers);

                double total_length = 0;
                for (int i = 0; i < layers.length(); i++)
                {
                    for (int j = 0; j < layers[i].length(); j++)
                    {
                        total_length += layers[i][j]->length()();
                    }
                }

                forwardInfoToMainWindow("Done.\nTotal Length: " % QString::number(total_length) % " in");

                emit updateDialog(StatusUpdateStepType::kVisualization, 100);
                emit dxfLoadedText(text, QHash<QString, QTextCharFormat>(), QList<int>(), QSet<int>());
            }
            else
            {
                emit forwardInfoToBuildExportWindow(m_filename, m_selected_meta);
                emit forwardInfoToMainWindow("Done.");
                emit updateDialog(StatusUpdateStepType::kVisualization, 100);
                emit dxfLoadedVisualization(QVector<QVector<QSharedPointer<SegmentBase>>>());
                emit dxfLoadedText(text, QHash<QString, QTextCharFormat>(), QList<int>(), QSet<int>());
            }

            QString openingDelim = m_selected_meta.m_comment_starting_delimiter;
            QString closingDelim = m_selected_meta.m_comment_ending_delimiter;

            //if we are allowed to adjust file, add header block and write out file
            //also takes care of situation in which layer times were adjusted
            if(m_adjust_file)
            {
                QFile tempFile(m_filename % "temp");
                if (tempFile.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text))
                {
                    QTextStream out(&tempFile);
                    for(QString& line : m_original_lines)
                    {
                        out << line << "\n";
                    }
                    tempFile.close();
                    bool ret = QFile::remove(m_filename);
                    QString tempStr = tempFile.fileName();

                    ret = QFile::rename(tempFile.fileName(), m_filename);
                }
            }
            }
            catch (ExceptionBase& exception)
            {
                QString message = "Error parsing DXF: " + QString(exception.what());
                emit error(message);
            }

        }
        else
        {
            qDebug() << "Error: " << m_filename << " does not exist or is not set to be loaded!";
            return;
        }
    }

    void DXFLoader::cancelSlice()
    {
        qDebug() << "Warning: parser does not yet have the functionality to cancel!";
    }

    void DXFLoader::forwardDialogUpdate(StatusUpdateStepType type, int percentComplete)
    {
        emit updateDialog(type, percentComplete);
    }
}
