// Header
#include "console/main_control.h"

// Qt

// Local
#include "managers/session_manager.h"
#include "threading/gcode_rpbf_saver.h"
#include "utilities/authenticity_checker.h"
#include "gcode/gcode_meta.h"

namespace ORNL {

    MainControl::MainControl(QSharedPointer<SettingsBase> options) : QObject() {
        GSM->setConsoleSettings(options);
        m_options = options;
        AuthenticityChecker* authChecker = new AuthenticityChecker(nullptr);
        connect(authChecker, &AuthenticityChecker::done, this, [this](bool ok){
            if(ok)
                continueStartup();
            else
                emit finished();
        });

        authChecker->startCheck();
    }

    void MainControl::continueStartup()
    {
        if(m_options->contains(Constants::ConsoleOptionStrings::kInputGlobalSettings))
            GSM->consoleConstructActiveGlobal(m_options->setting<QString>(Constants::ConsoleOptionStrings::kInputGlobalSettings));

        connect(CSM.get(), &SessionManager::partAdded, this, &MainControl::loadComplete);
        connect(CSM.get(), &SessionManager::forwardSliceComplete, this, &MainControl::sliceComplete);
        connect(CSM.get(), &SessionManager::updateDialog, this, &MainControl::displayProgress);
    }

    void MainControl::run()
    {
        int stlCount = m_options->setting<int>(Constants::ConsoleOptionStrings::kInputStlCount);
        if(stlCount > 0)
        {
            m_parts_to_load = stlCount;
            for(int i = 0; i < stlCount; ++i)
            {
                if(m_options->contains(Constants::ConsoleOptionStrings::kInputSTLTransform))
                {
                    QFile file(m_options->setting<QString>(Constants::ConsoleOptionStrings::kInputSTLTransform));
                    file.open(QIODevice::ReadOnly);
                    QTextStream in(&file);
                    QString transforms = in.readAll();
                    file.close();
                    fifojson j = fifojson::parse(transforms.toStdString());

                    for (auto it : j[Constants::Settings::Session::kParts].items())
                        CSM->loadModel(m_options->setting<QString>(Constants::ConsoleOptionStrings::kInputStlFiles, i), false, MeshType::kBuild, true);

                    CSM->loadPartsJson(j);
                }
                else

                    CSM->loadModel(m_options->setting<QString>(Constants::ConsoleOptionStrings::kInputStlFiles, i), false, MeshType::kBuild, true);
            }
        }
        else
        {
            connect(CSM.get(), &SessionManager::totalPartsInProject, this, &MainControl::partsInProject);
            CSM->loadSession(false, m_options->setting<QString>(Constants::ConsoleOptionStrings::kInputProjectFile));
        }
    }

    void MainControl::partsInProject(int total)
    {
        m_parts_to_load = total;
    }

    void MainControl::loadComplete()
    {
        --m_parts_to_load;

        if(m_parts_to_load == 0)
            CSM->doSlice();
    }

    void MainControl::sliceComplete(QString filepath, bool alterFile)
    {
        if(m_options->setting<bool>(Constants::ConsoleOptionStrings::kRealTimeMode))
        {
            auto meta = GcodeMetaList::createMapping()[GSM->getGlobal()->setting<int>(Constants::ExperimentalSettings::PrinterConfig::kSlicerType)];
            updateOutputInformation(filepath, meta);
            gcodeParseComplete();
        }else
        {
            if (filepath.mid(filepath.lastIndexOf(".") + 1) == "dxf")
            { // Check if this is dxf file or not
                DXFLoader* loader_two = new DXFLoader(filepath, alterFile);
            } else
            {
                GCodeLoader* loader = new GCodeLoader(filepath, alterFile);
                connect(loader, &GCodeLoader::finished, loader, &GCodeLoader::deleteLater);
                connect(loader, &GCodeLoader::forwardInfoToBuildExportWindow, this, &MainControl::updateOutputInformation);
                connect(loader, &GCodeLoader::finished, this, &MainControl::gcodeParseComplete);
                connect(loader, &GCodeLoader::updateDialog, this, &MainControl::displayProgress);
                loader->start();
            }
        }
    }

    void MainControl::updateOutputInformation(QString tempLocation, GcodeMeta meta)
    {
        m_temp_location = tempLocation;
        m_selected_meta = meta;
    }

    void MainControl::gcodeParseComplete()
    {
        QFileInfo info(m_options->setting<QString>(Constants::ConsoleOptionStrings::kOutputLocation));
        QString partName = info.baseName();
        QString filepath = info.absolutePath();

        //if bundling files, create folder based on name
        //            if(m_bundle_files_checkbox->isChecked())
        //            {
        //                filepath = filepath % '/' % partName;
        //                QDir dir(filepath);
        //                dir.mkdir(filepath);
        //            }

        QString gcodeFileName = filepath % '/' % partName % m_selected_meta.m_file_suffix;
        if(QFile::exists(gcodeFileName))
            QFile::remove(gcodeFileName);

        QString projectFileName = filepath % '/' % partName % ".s2p";

        QString text;
        QFile inputFile(m_temp_location);
        if (inputFile.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            QTextStream in(&inputFile);
            text = in.readAll();
            inputFile.close();
        }

//                    text.prepend(m_selected_meta.m_comment_starting_delimiter % "Sliced by: " % m_operator_input->text() % m_selected_meta.m_comment_ending_delimiter % "\n" %
//                                 m_selected_meta.m_comment_starting_delimiter % "Slicing notes: " % m_description_input->toPlainText() % m_selected_meta.m_comment_ending_delimiter % "\n");

//                    if(CSM->sensorFilesGenerated())
//                    {
//                        bool overwrite = false;
//                        bool hasAnswered = false;
//                        text.replace("#TEMPFILENAME#", partName + "-");

//                        QFileInfo fi(m_location);
//                        QDir tempDir = fi.absoluteDir();
//                        QFile sensorFile(tempDir.absolutePath() % "/scan_output-0.dat");
//                        if(sensorFile.exists())
//                        {
//                            QString outputLoc = filepath % '/' % partName % "-0.dat";

//                            if(QFile::exists(outputLoc))
//                            {
//                                QMessageBox::StandardButton reply = QMessageBox::question(
//                                 this, "Warning", "Auxiliary file(s) already exists.  Do you wish to overwrite?", QMessageBox::Yes | QMessageBox::No);

//                                if(reply == QMessageBox::Yes)
//                                {
//                                    QFile::remove(outputLoc);
//                                    QFile::copy(sensorFile.fileName(), outputLoc);
//                                    overwrite = true;
//                                }
//                                hasAnswered = true;
//                            }
//                            else
//                                QFile::copy(sensorFile.fileName(), outputLoc);
//                        }

//                        int i = 1;
//                        bool moreFound = true;
//                        while(moreFound)
//                        {
//                            moreFound = false;
//                            QFile sensorFile(tempDir.absolutePath() % "/scan_output-" % QString::number(i) % ".dat");
//                            if(sensorFile.exists())
//                            {
//                                QString outputLoc = filepath % '/' % partName % "-" % QString::number(i) % ".dat";

//                                if(!hasAnswered && QFile::exists(outputLoc))
//                                {
//                                    QMessageBox::StandardButton reply = QMessageBox::question(
//                                     this, "Warning", "Auxiliary file(s) already exists.  Do you wish to overwrite?", QMessageBox::Yes | QMessageBox::No);

//                                    if(reply == QMessageBox::Yes)
//                                    {
//                                        QFile::remove(outputLoc);
//                                        QFile::copy(sensorFile.fileName(), outputLoc);
//                                        overwrite = true;
//                                    }
//                                    hasAnswered = true;
//                                }
//                                else if(hasAnswered && QFile::exists(outputLoc))
//                                {
//                                    if(overwrite)
//                                    {
//                                        QFile::remove(outputLoc);
//                                        QFile::copy(sensorFile.fileName(), outputLoc);
//                                    }
//                                    else
//                                        continue;
//                                }
//                                else
//                                    QFile::copy(sensorFile.fileName(), outputLoc);

//                                moreFound = true;
//                            }
//                            ++i;
//                        }


        if(m_selected_meta == GcodeMetaList::RPBFMeta &&
                static_cast<SlicerType>(GSM->getGlobal()->setting<int>(Constants::ExperimentalSettings::PrinterConfig::kSlicerType))
                != SlicerType::kRealTimeRPBF)
        {
            Angle clockInRad = GSM->getGlobal()->setting<Angle>(Constants::ExperimentalSettings::RPBFSlicing::kClockingAngle);
            bool use_sector_offsetting = GSM->getGlobal()->setting<bool>(Constants::ExperimentalSettings::RPBFSlicing::kSectorOffsettingEnable);
            Angle sector_width = GSM->getGlobal()->setting<Angle>(Constants::ExperimentalSettings::RPBFSlicing::kSectorSize);

            GCodeRPBFSaver* saver = new GCodeRPBFSaver(m_temp_location, filepath, gcodeFileName, text, m_selected_meta, clockInRad(), use_sector_offsetting, sector_width);
            connect(saver, &GCodeRPBFSaver::finished, saver, &GCodeRPBFSaver::deleteLater);
            connect(saver, &GCodeRPBFSaver::finished, this, [this, filepath, partName] () {
                //qInfo() << gcodeFileName % " has been succesfully exported to " % m_output_path;
                emit finished(); });
            saver->start();
        }
        else
        {
            QFile tempFile(m_temp_location % "temp");
            if (tempFile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
            {
                QTextStream out(&tempFile);
                out << text;
                tempFile.close();

                QFile::rename(tempFile.fileName(), gcodeFileName);
                //qInfo() << filename % " has been succesfully exported to " % m_output_path;
                emit finished();
            }
        }
    }

    void MainControl::displayProgress(StatusUpdateStepType type, int percentage)
    {
        if(!m_options->setting<bool>(Constants::ConsoleOptionStrings::kRealTimeMode))
        {
            if(m_last_step_type != type)
                m_last_step_type = type;
            else
                std::cout << "\r";

            std::cout << toString(type).toStdString() << " " << percentage;

            if(percentage == 100)
                std::cout << "\n";
        }else
        {
            if(type == StatusUpdateStepType::kRealTimeLayerCompleted)
            {
                if(percentage == -1)
                    std::cout << "\n" << "Slice complete";
                else
                    std::cout << "\r" << toString(type).toStdString() << " " << percentage; // Percentage is actually a layer number here
            }
        }
    }
}
