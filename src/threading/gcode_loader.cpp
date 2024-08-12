// Header
#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <QStringBuilder>
#include <QDebug>

#include "gcode/gcode_command.h"
#include "threading/gcode_loader.h"

#include "graphics/objects/gcode_object.h"
#include "managers/settings/settings_manager.h"
#include "managers/session_manager.h"

#include "gcode/gcode_meta.h"
#include "gcode/parsers/cincinnati_parser.h"
#include "gcode/parsers/common_parser.h"
#include "gcode/parsers/parser_base.h"
#include "gcode/parsers/GKN_parser.h"
//#include "gcode/parsers/haas_parser.h"
#include "gcode/parsers/ingersoll_parser.h"
#include "gcode/parsers/marlin_parser.h"
#include "gcode/parsers/mazak_parser.h"
//#include "gcode/parsers/hurco_parser.h"
//#include "gcode/parsers/dmg_dmu_parser.h"
//#include "gcode/parsers/romi_fanuc_parser.h"
//#include "gcode/parsers/gudel_parser.h"
#include "gcode/parsers/beam_parser.h"
#include "gcode/parsers/mvp_parser.h"
#include "gcode/parsers/rpbf_parser.h"
#include "gcode/parsers/siemens_parser.h"
#include "gcode/parsers/aerobasic_parser.h"
#include "gcode/parsers/adamantine_parser.h"

#include "geometry/segments/arc.h"
#include "geometry/segments/bezier.h"
#include "geometry/segments/line.h"

#include "net_functions/http_client.h"

#include "utilities/mathutils.h"
#include "managers/preferences_manager.h"

namespace ORNL
{
    GCodeLoader::GCodeLoader(QString filename, bool alterFile) :
      m_filename(filename)
      , m_adjust_file(alterFile)
      , m_should_cancel(false)
    {
        m_prestart = QStringMatcher(Constants::PathModifierStrings::kPrestart.toUpper());
        m_initial_startup = QStringMatcher(Constants::PathModifierStrings::kInitialStartup.toUpper());
        m_slowdown = QStringMatcher(Constants::PathModifierStrings::kSlowDown.toUpper());
        m_forward_tipwipe = QStringMatcher(Constants::PathModifierStrings::kForwardTipWipe.toUpper());
        m_reverse_tipwipe = QStringMatcher(Constants::PathModifierStrings::kReverseTipWipe.toUpper());
        m_angled_tipwipe = QStringMatcher(Constants::PathModifierStrings::kAngledTipWipe.toUpper());
        m_coasting = QStringMatcher(Constants::PathModifierStrings::kCoasting.toUpper());
        m_spirallift = QStringMatcher(Constants::PathModifierStrings::kSpiralLift.toUpper());
        m_rampingup = QStringMatcher(Constants::PathModifierStrings::kRampingUp.toUpper());
        m_rampingdown = QStringMatcher(Constants::PathModifierStrings::kRampingDown.toUpper());
        m_leadin = QStringMatcher(Constants::PathModifierStrings::kLeadIn.toUpper());

        m_modifier_colors.push_back(PM->getVisualizationColor(VisualizationColors::kPrestart));
        m_modifier_colors.push_back(PM->getVisualizationColor(VisualizationColors::kInitialStartup));
        m_modifier_colors.push_back(PM->getVisualizationColor(VisualizationColors::kSlowDown));
        m_modifier_colors.push_back(PM->getVisualizationColor(VisualizationColors::kTipWipeForward));
        m_modifier_colors.push_back(PM->getVisualizationColor(VisualizationColors::kTipWipeReverse));
        m_modifier_colors.push_back(PM->getVisualizationColor(VisualizationColors::kTipWipeAngled));
        m_modifier_colors.push_back(PM->getVisualizationColor(VisualizationColors::kCoasting));
        m_modifier_colors.push_back(PM->getVisualizationColor(VisualizationColors::kSpiralLift));
        m_modifier_colors.push_back(PM->getVisualizationColor(VisualizationColors::kRampingUp));
        m_modifier_colors.push_back(PM->getVisualizationColor(VisualizationColors::kRampingDown));
        m_modifier_colors.push_back(PM->getVisualizationColor(VisualizationColors::kLeadIn));

        m_perimeter = QStringMatcher(Constants::RegionTypeStrings::kPerimeter.toUpper());
        m_perimeter_embossing = QStringMatcher(Constants::RegionTypeStrings::kPerimeterEmbossing.toUpper());
        m_inset = QStringMatcher(Constants::RegionTypeStrings::kInset.toUpper());
        m_inset_embossing = QStringMatcher(Constants::RegionTypeStrings::kInsetEmbossing.toUpper());
        m_infill = QStringMatcher(Constants::RegionTypeStrings::kInfill.toUpper());
        m_ironing = QStringMatcher(Constants::RegionTypeStrings::kIroning.toUpper());
        m_skin = QStringMatcher(Constants::RegionTypeStrings::kSkin.toUpper());
        m_skeleton = QStringMatcher(Constants::RegionTypeStrings::kSkeleton.toUpper());
        m_support = QStringMatcher(Constants::RegionTypeStrings::kSupport.toUpper());
        m_support_roof = QStringMatcher(Constants::RegionTypeStrings::kSupportRoof.toUpper());
        m_travel = QStringMatcher(Constants::RegionTypeStrings::kTravel.toUpper());
        m_raft = QStringMatcher(Constants::RegionTypeStrings::kRaft.toUpper());
        m_brim = QStringMatcher(Constants::RegionTypeStrings::kBrim.toUpper());
        m_skirt = QStringMatcher(Constants::RegionTypeStrings::kSkirt.toUpper());
        m_laserscan = QStringMatcher(Constants::RegionTypeStrings::kLaserScan.toUpper());
        m_thermalscan = QStringMatcher(Constants::RegionTypeStrings::kThermalScan.toUpper());

        m_color_space_conversion = 1.0 / 255.0;
        m_layer_pattern = QRegularExpression("W*(\\d+)W*");
    }

    QString GCodeLoader::additionalExportComments()
    {
        QString openingDelim = m_selected_meta.m_comment_starting_delimiter;
        QString closingDelim = m_selected_meta.m_comment_ending_delimiter;

        QString partMinTranslation;
        QVector3D translationMin;
        int index = 0;
        for (QSharedPointer<Part> part : CSM->parts()) {
            auto transformation = part->rootMesh()->transformation();
            QVector3D translation, scale;
            QQuaternion rotation;
            std::tie(translation, rotation, scale) = MathUtils::decomposeTransformMatrix(transformation);
            if(++index == 1 || translation.x() < translationMin.x() || translation.y() < translationMin.y())
                translationMin = translation;
        }
        if(index > 0)
            partMinTranslation = openingDelim % "Part Translation: X" %
                    QString::number(translationMin.x()/1000000, 'f', 4) % ", Y" %
                    QString::number(translationMin.y()/1000000, 'f', 4) % ", Z" %
                    QString::number(translationMin.z()/1000000, 'f', 4) % " m" % closingDelim % "\n";

        /*QString travelTypes  = openingDelim % "Travel Types:";
        QString travelColors = openingDelim % "Travel Colors:";
        for(const auto& color : PM->getVisualizationHexColors()){
            travelTypes  = travelTypes  % " " % QString::fromStdString(color.first);
            travelColors = travelColors % " " % QString::fromStdString(color.second).right(6);
        }
        travelTypes  = travelTypes  % closingDelim % "\n";
        travelColors = travelColors % closingDelim % "\n";*/

        return partMinTranslation; // % travelTypes % travelColors;
    }

    void GCodeLoader::savePartsModelObjFile()
    {
        QFileInfo info(m_filename);
        QString strObjFile = info.absoluteDir().absolutePath() + "/" + info.baseName() + ".obj";
        QFile::remove(strObjFile);
        QFile objFile(strObjFile);
        if (objFile.open(QIODevice::WriteOnly | QIODevice::Text))
        {
            QTextStream out(&objFile);

            int vertIndex = 1;
            for (QSharedPointer<Part> part : CSM->parts()) {
                auto verts = part->rootMesh()->vertices();
                auto faces = part->rootMesh()->faces();

                out << "# Object name\no " << part->rootMesh()->name() << "\n" << "# Begin list of vertices\n";

                for (MeshVertex& vertex : verts) {
                    out << "v " << QString::number(vertex.location.x()/1000, 'f', 4) <<
                           " "  << QString::number(vertex.location.y()/1000, 'f', 4) <<
                           " "  << QString::number(vertex.location.z()/1000, 'f', 4) << " 1.0\n";
                }

                out << "# End list of vertices\n# Begin list of faces\n";

                for(MeshFace& face : faces) {
                    out << "f " << face.vertex_index[0] + vertIndex <<
                           " "  << face.vertex_index[1] + vertIndex <<
                           " "  << face.vertex_index[2] + vertIndex << "\n";
                }

                out << "# End list of faces\n# End Object " << part->rootMesh()->name() << "\n\n";

                vertIndex += verts.length();
            }

            objFile.close();
        }
        else {
            return;
        }

        if(PM->getKatanaSendOutput()){
            QThread *thread = QThread::create(sendGcodeModelObjFile,
                                          PM->getKatanaTCPIp(), PM->getKatanaTCPPort(),
                                          toString(m_selected_meta.m_syntax_id), m_filename, strObjFile);

            connect (thread, &QThread::finished, thread, &QThread::deleteLater);
            thread->start();
        }
    }

    void GCodeLoader::sendGcodeModelObjFile(QString host, int port, QString machineName, QString gcodeFilePath, QString objFilePath)
    {
        TCPConnection client;
        client.setupNew(host, port);
        QThread::sleep(1);

        if(client.isReady()){
            client.write("Machine Printer: " + machineName);
            QThread::sleep(1);

            client.write("GCode File Path: " + gcodeFilePath);
            QThread::sleep(1);

            client.write("Model File Path: " + objFilePath);
            QThread::sleep(1);

            client.close();
        }
    }

    void GCodeLoader::run() {
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
            QString weightInfo = "No statistics calculated";
            //parse header looking for syntax
            setParser(m_original_lines, m_lines);
            if(!disableVisualization)
            {
                connect(m_parser.get(), &CommonParser::statusUpdate, this, &GCodeLoader::forwardDialogUpdate);
                connect(m_parser.get(), &CommonParser::forwardInfoToMainWindow, this, &GCodeLoader::forwardInfoToMainWindow);

                m_parser->parseHeader();
                QHash<QString, double> visualizationSettings = m_parser->parseFooter();
                //GSM->getGlobal()->setting<int>(Constants::PrinterSettings::Visualization::kLayerSkip);
                QList<QList<GcodeCommand>> m_motion_commands = m_parser->parseLines(layerSkip);
                if(m_parser->getWasModified())
                {
                    text = m_original_lines.join("\n");
                    m_lines = text.toUpper().split("\n");
                }

                QList<QList<Time>> layer_times = m_parser->getLayerTimes();
                QList<double> layer_FR_modifiers = m_parser->getLayerFeedRateModifiers();
                QList<Volume> layer_volumes = m_parser->getLayerVolumes();

                Volume total_volume;
                min_time = INT_MAX;
                max_time = INT_MIN;
                for(int i = 0; i < layer_times.size(); ++i)
                {
                    Time& current = layer_times[i][0]; // layer time is the max of the extruders time
                    for (auto extruder_time : layer_times[i])
                        current = qMax(current, extruder_time);

                    min_time = qMin(current, min_time);
                    max_time = qMax(current, max_time);

                    // add current layer to total printing and adjusted time
                    total_time += current;
                    total_adjusted_time += current / layer_FR_modifiers[i];
                    total_volume += layer_volumes[i];
                }

                PrintMaterial m_material = static_cast<PrintMaterial>((int)
                                                visualizationSettings[Constants::MaterialSettings::Density::kMaterialType]);

                Density materialDensity = ((m_material == PrintMaterial::kOther) ?
                                               (visualizationSettings[Constants::MaterialSettings::Density::kDensity]) :
                                                toDensityValue(m_material));

                Mass total_mass = total_volume * materialDensity;

                //forward to layer_times_window
                emit forwardInfoToLayerTimeWindow(layer_times, layer_FR_modifiers, ForceMinimumLayerTime::kSlow_Feedrate ==
                        static_cast<ForceMinimumLayerTime>(GSM->getGlobal()->setting<int>(Constants::MaterialSettings::Cooling::kForceMinLayerTimeMethod)));

                weightInfo = QString::number((total_mass / m_selected_meta.m_mass_unit)()) % " " % m_selected_meta.m_mass_unit.toString();

                //forward to build_log_export
                emit forwardInfoToBuildExportWindow(m_filename, m_selected_meta);

                QString keyInfo = "GCode file: " % m_filename % "\n"
                        % "Total Time Estimate: " % MathUtils::formattedTimeSpan(total_time()) % "\n";

                if (m_adjust_file && total_adjusted_time > 0 &&
                        GSM->getGlobal()->setting< int >(Constants::MaterialSettings::Cooling::kForceMinLayerTime))
                        keyInfo = keyInfo % "Total Adjusted Time: " % MathUtils::formattedTimeSpan(total_adjusted_time()) % "\n";

                double volumeValue = total_volume() / pow<3>(PM->getDistanceUnit())();
                double distanceValue = (m_parser->getTotalDistance() / PM->getDistanceUnit())();
                double printingDistanceValue = (m_parser->getPrintingDistance() / PM->getDistanceUnit())();
                double travelDistanceValue = (m_parser->getTravelDistance() / PM->getDistanceUnit())();
                double massValue = (total_mass / PM->getMassUnit())();
                keyInfo = keyInfo % "Volume: " % QString::number(volumeValue) % " " % PM->getDistanceUnit().toString() % "³\n"
                        % "Printing Distance: " % QString::number(printingDistanceValue) % " " % PM->getDistanceUnit().toString() % "\n"
                        % "Travel Distance: " % QString::number(travelDistanceValue) % " " % PM->getDistanceUnit().toString() % "\n"
                        % "Total Distance: " % QString::number(distanceValue) % " " % PM->getDistanceUnit().toString() % "\n"
                        % "Approximate Weight (" % toString(m_material) % "): "
                        % QString::number(massValue) % " " % PM->getMassUnit().toString() % "\n";

                QTime qt(0, 0);
                qt = qt.addMSecs(CSM->getSliceTimeElapsed());
                keyInfo = keyInfo % "Total Slice Time (excluding gcode writing/parsing): " % qt.toString("hh:mm:ss.zzz");

                emit forwardInfoToMainWindow(keyInfo);

                //reduce size to fit screen appropriately, found through testing
                //ratio reduction based on a combination of scale, matrix transformation impacts, and aesthetics (ie. personal opinion)
                m_segment_width = visualizationSettings[Constants::ProfileSettings::Layer::kBeadWidth] * Constants::OpenGL::kObjectToView;// * .25f;

                Distance x_dist, y_dist = 0.0;
                m_start_pos = QVector3D((x_dist() + visualizationSettings[Constants::PrinterSettings::Dimensions::kXOffset]) * Constants::OpenGL::kObjectToView,
                                        (y_dist() + visualizationSettings[Constants::PrinterSettings::Dimensions::kYOffset]) * Constants::OpenGL::kObjectToView, .0f);

                m_origin = QVector3D(x_dist() + visualizationSettings[Constants::PrinterSettings::Dimensions::kXOffset],
                                     y_dist() + visualizationSettings[Constants::PrinterSettings::Dimensions::kYOffset], .0f);

                auto min_z = GSM->getGlobal()->setting<Distance>(Constants::PrinterSettings::Dimensions::kZMin);
                auto z_offset = visualizationSettings[Constants::PrinterSettings::Dimensions::kZOffset];

                m_x_offset = visualizationSettings[Constants::PrinterSettings::Dimensions::kXOffset];
                m_y_offset = visualizationSettings[Constants::PrinterSettings::Dimensions::kYOffset];
                m_z_offset = (min_z - z_offset)() * Constants::OpenGL::kObjectToView;
                m_table_offset = 0.0f;
                m_prev_table_offset = 0.0f;

                //reserve more memory than the hash will need to guarantee no reallocation
                QHash<QString, QTextCharFormat> fontColors;
                fontColors.reserve(m_lines.size());

                QVector<QVector<QSharedPointer<SegmentBase>>> layers;

                int currentLayer = 0, totalLayer = m_motion_commands.size();
                for(QList<GcodeCommand> layerCommands : m_motion_commands)
                {
                    QVector<QSharedPointer<SegmentBase>> layer;

                    for(GcodeCommand command : layerCommands)
                    {
                        QColor lineColor(PM->getVisualizationColor(VisualizationColors::kUnknown));
                        if(fontColors.contains(command.getComment()))
                        {
                            lineColor = fontColors[command.getComment()].foreground().color();
                        }
                        else if(!command.getComment().isEmpty())
                        {
                            lineColor = determineFontColor(command.getComment());
                            QTextCharFormat format;
                            format.setForeground(lineColor);
                            fontColors.insert(m_original_lines[command.getLineNumber()], format);
                        }

                        QVector<QSharedPointer<SegmentBase>> generated_segments;

                        if(m_selected_meta.hasTravels)
                            generated_segments = generateVisualSegment(command.getLineNumber() + 1, currentLayer, lineColor, command.getCommandID(), command.getParameters(), command.getExtrudersOn(), command.getExtruderOffsets(), command.getExtrudersSpeed(), true, command.getComment());
                        else
                            generated_segments = generateVisualSegment(command.getLineNumber() + 1, currentLayer, lineColor, command.getCommandID(), command.getParameters(), command.getExtrudersOn(), command.getExtruderOffsets(), command.getExtrudersSpeed(), false, command.getComment(), command.getOptionalParameters());

                        layer.append(generated_segments);
                    }
                    layers.push_back(layer);
                    ++currentLayer;

                    emit updateDialog(StatusUpdateStepType::kVisualization, (double)currentLayer / (double)totalLayer * 100);
                    if(m_should_cancel)
                        return;
                }

                //emit vector for visualization
                emit gcodeLoadedVisualization(layers);
                //very likely to have allocated too much memory, free extra
                fontColors.squeeze();
                //send text and font colors for display, and line numbers for easy editor navigation
                emit gcodeLoadedText(text, fontColors, m_parser->getLayerStartLines(), m_parser->getLayerSkipLines());
            }
            else
            {
                emit forwardInfoToBuildExportWindow(m_filename, m_selected_meta);
                emit forwardInfoToMainWindow("GCode file: " % m_filename % "\n");
                emit updateDialog(StatusUpdateStepType::kVisualization, 100);
                emit gcodeLoadedVisualization(QVector<QVector<QSharedPointer<SegmentBase>>>());
                emit gcodeLoadedText(text, QHash<QString, QTextCharFormat>(), QList<int>(), QSet<int>());
            }

            QString openingDelim = m_selected_meta.m_comment_starting_delimiter;
            QString closingDelim = m_selected_meta.m_comment_ending_delimiter;
            QString additionalHeaderBlock =
            //openingDelim % "Sliced on: " % QDateTime::currentDateTime().toString("MM/dd/yyyy") % closingDelim % "\n" %
            //openingDelim % "Expected Weight: " % weightInfo % closingDelim % "\n" %
            openingDelim % "Expected Build Time: " % MathUtils::formattedTimeSpan(total_time()) % closingDelim % "\n" %
            //openingDelim % "Minimum Layer Time: " % MathUtils::formattedTimeSpan(min_time()) % closingDelim % "\n" %
            openingDelim % "Maximum Layer Time: " % MathUtils::formattedTimeSpan(max_time()) % closingDelim % "\n" %
            openingDelim % "XYZ Translation Data: " % QString::number(m_origin.x()) % ", " %
                    QString::number(m_origin.y()) % ", " % QString::number(m_z_offset) % closingDelim % "\n" % additionalExportComments();

            //if we are allowed to adjust file, add header block and write out file
            //also takes care of situation in which layer times were adjusted
            if(m_adjust_file)
            {
                QFile tempFile(m_filename % "temp");
                if (tempFile.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text))
                {
                    QTextStream out(&tempFile);
                    out << additionalHeaderBlock;
                    for(QString& line : m_original_lines)
                    {
                        out << line << "\n";
                    }
                    tempFile.close();
                    bool ret = QFile::remove(m_filename);
                    QString tempStr = tempFile.fileName();

                    ret = QFile::rename(tempFile.fileName(), m_filename);
                }

                if(PM->getKatanaSendOutput())
                    savePartsModelObjFile();
            }
            }
            catch (ExceptionBase& exception)
            {
                QString message = "Error parsing GCode: " + QString(exception.what());
                emit error(message);
            }

        }
    }

    void GCodeLoader::cancelSlice()
    {
        m_should_cancel = true;
        if(m_parser.get() != nullptr)
            m_parser->cancelSlice();
    }

    void GCodeLoader::forwardDialogUpdate(StatusUpdateStepType type, int percentComplete)
    {
        emit updateDialog(type, percentComplete);
    }

    //at this moment, parsing the header is simply to find the syntax
    void GCodeLoader::setParser(QStringList& originalLines, QStringList& lines)
    {
        int m_current_line = 0;
        bool foundSyntax = false;
        QStringMatcher syntaxIdentifier1("G-CODE SYNTAX");
        QStringMatcher syntaxIdentifier2("GCODE SYNTAX");
        while(m_current_line < m_lines.size())
        {
            if(syntaxIdentifier1.indexIn(m_lines[m_current_line]) != -1 || syntaxIdentifier2.indexIn(m_lines[m_current_line]) != -1)
            {
                if(m_lines[m_current_line].contains(toString(GcodeSyntax::k5AxisMarlin).toUpper())) {
                    m_parser.reset(new MarlinParser(GcodeMetaList::MarlinMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::MarlinMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kAML3D).toUpper())) {
                    m_parser.reset(new CincinnatiParser(GcodeMetaList::AML3DMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::AML3DMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kBeam).toUpper())) {
                    m_parser.reset(new BeamParser(GcodeMetaList::BeamMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::BeamMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kCincinnati).toUpper())) {
                    m_parser.reset(new CincinnatiParser(GcodeMetaList::CincinnatiMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::CincinnatiMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kDmgDmu).toUpper())) {
                     m_parser.reset(new CommonParser(GcodeMetaList::DmgDmuAndBeamMeta, m_adjust_file, originalLines, lines));
                     m_selected_meta = GcodeMetaList::DmgDmuAndBeamMeta;
                 }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kGKN).toUpper())) {
                    m_parser.reset(new GKNParser(GcodeMetaList::GKNMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::GKNMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kGudel).toUpper())) {
                    m_parser.reset(new CommonParser(GcodeMetaList::GudelMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::GudelMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kHaasInch).toUpper())) {
                    m_parser.reset(new CommonParser(GcodeMetaList::HaasInchMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::HaasInchMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kHaasMetric).toUpper())) {
                    m_parser.reset(new CommonParser(GcodeMetaList::HaasMetricMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::HaasMetricMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kHurco).toUpper())) {
                     m_parser.reset(new CommonParser(GcodeMetaList::HurcoMeta, m_adjust_file, originalLines, lines));
                     m_selected_meta = GcodeMetaList::HurcoMeta;
                 }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kIngersoll).toUpper())) {
                     m_parser.reset(new IngersollParser(GcodeMetaList::IngersollMeta, m_adjust_file, originalLines, lines));
                     m_selected_meta = GcodeMetaList::IngersollMeta;
                 }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kKraussMaffei).toUpper())) {
                     m_parser.reset(new MarlinParser(GcodeMetaList::KraussMaffeiMeta, m_adjust_file, originalLines, lines));
                     m_selected_meta = GcodeMetaList::KraussMaffeiMeta;
                 }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kMarlinPellet).toUpper())) {
                    m_parser.reset(new MarlinParser(GcodeMetaList::MarlinMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::MarlinMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kMarlin).toUpper())) {
                    m_parser.reset(new MarlinParser(GcodeMetaList::MarlinMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::MarlinMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kMach4).toUpper())) { // Mach4 uses Marlin
                    m_parser.reset(new MarlinParser(GcodeMetaList::MarlinMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::MarlinMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kRepRap).toUpper())) { // RepRap uses Marlin
                    m_parser.reset(new MarlinParser(GcodeMetaList::RepRapMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::RepRapMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kMazak).toUpper())) {
                    m_parser.reset(new MazakParser(GcodeMetaList::MazakMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::MazakMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kMeld).toUpper())) {
                    m_parser.reset(new CommonParser(GcodeMetaList::MeldMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::MeldMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kMeltio).toUpper())) {
                    m_parser.reset(new CommonParser(GcodeMetaList::MeltioMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::MeltioMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kMVP).toUpper())) {
                    m_parser.reset(new MVPParser(GcodeMetaList::MVPMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::MVPMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kOkuma).toUpper())) {
                    m_parser.reset(new CommonParser(GcodeMetaList::HaasMetricMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::ORNLMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kORNL).toUpper())) {
                    m_parser.reset(new CommonParser(GcodeMetaList::ORNLMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::ORNLMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kRomiFanuc).toUpper())) {
                     m_parser.reset(new CommonParser(GcodeMetaList::RomiFanucMeta, m_adjust_file, originalLines, lines));
                     m_selected_meta = GcodeMetaList::RomiFanucMeta;
                 }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kRPBF).toUpper())) {
                    m_parser.reset(new RPBFParser(GcodeMetaList::RPBFMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::RPBFMeta;
                }
                 else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kSandia).toUpper())) {
                    m_parser.reset(new CommonParser(GcodeMetaList::SandiaMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::SandiaMeta;
                 }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kSiemens).toUpper())) {
                    m_parser.reset(new SiemensParser(GcodeMetaList::SiemensMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::SiemensMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kSkyBaam).toUpper())) {
                     m_parser.reset(new CincinnatiParser(GcodeMetaList::SkyBaamMeta, m_adjust_file, originalLines, lines));
                     m_selected_meta = GcodeMetaList::SkyBaamMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kThermwood).toUpper())) {
                    m_parser.reset(new CincinnatiParser(GcodeMetaList::CincinnatiMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::CincinnatiMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kTormach).toUpper())) {
                    m_parser.reset(new CincinnatiParser(GcodeMetaList::TormachMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::TormachMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kAeroBasic).toUpper())) {
                    m_parser.reset(new AeroBasicParser(GcodeMetaList::AeroBasicMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::AeroBasicMeta;
                }
                else if(m_lines[m_current_line].contains(toString(GcodeSyntax::kAdamantine).toUpper())) {
                    m_parser.reset(new AdamantineParser(GcodeMetaList::AdamantineMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::AdamantineMeta;
                }
//                else if(m_lines[m_current_line].contains("WOLF"))
//                    m_syntax = GcodeSyntax::kWolf;
                else
                {
                    qDebug() << "Warning: unknown syntax";
                    m_parser.reset(new CommonParser(GcodeMetaList::MarlinMeta, m_adjust_file, originalLines, lines));
                    m_selected_meta = GcodeMetaList::MarlinMeta;
                }
                foundSyntax = true;
            }
            ++m_current_line;
            if(foundSyntax)
                break;
        }

        if(!foundSyntax)
        {
            qDebug() << "No syntax definition found: attempting common";
            m_parser.reset(new CommonParser(GcodeMetaList::MarlinMeta, m_adjust_file, originalLines, lines));
        }
    }

    QColor GCodeLoader::determineFontColor(const QString& comment)
    {
        if(m_prestart.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kPrestart);
        if(m_initial_startup.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kInitialStartup);
        if(m_slowdown.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kSlowDown);
        if(m_forward_tipwipe.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kTipWipeForward);
        if(m_reverse_tipwipe.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kTipWipeReverse);
        if(m_angled_tipwipe.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kTipWipeAngled);
        if(m_coasting.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kCoasting);
        if(m_spirallift.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kSpiralLift);
        if(m_rampingup.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kRampingUp);
        if(m_rampingdown.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kRampingDown);
        if(m_leadin.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kLeadIn);
        if(m_perimeter_embossing.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kEmbossing);
        if(m_perimeter.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kPerimeter);
        if(m_inset_embossing.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kEmbossing);
        if(m_inset.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kInset);
        if(m_infill.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kInfill);
        if(m_ironing.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kIroning);
        if(m_skin.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kSkin);
        if(m_skeleton.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kSkeleton);
        if(m_support.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kSupport);
        if(m_travel.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kTravel);
        if(m_raft.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kRaft);
        if(m_brim.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kBrim);
        if(m_skirt.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kSkirt);
        if(m_laserscan.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kLaserScan);
        if(m_thermalscan.indexIn(comment) != -1)
            return PM->getVisualizationColor(VisualizationColors::kThermalScan);

        return PM->getVisualizationColor(VisualizationColors::kUnknown);
    }

    QVector<QSharedPointer<SegmentBase> > GCodeLoader::generateVisualSegment(int line_num, int layer_num, const QColor& color, int command_id,
                                            const QMap<char, double>& parameters, QVector<bool> extruders_on, QVector<Point> extruder_offsets,
                                            double extruders_speed, bool is_travel, QString comment, const QMap<char, double>& optional_parameters)
    {
        // Parameters for drawing and placing each segment in the world correctly
        QVector3D end_pos = m_start_pos;
        QVector3D info_end_pos = m_info_start_pos;
        bool info_speed_set = false;

        if (parameters.contains('F')){
            info_speed_set = true;
            m_info_speed = QString().asprintf("%0.4f", (Velocity(parameters['F']) / PM->getVelocityUnit())()) % " " % PM->getVelocityUnitText();
        }
        if (parameters.contains('S')){
            m_info_extruder_speed = QString().asprintf("%0.4f", (AngularVelocity(parameters['S']) / m_selected_meta.m_angular_velocity_unit)()) % " rpm";
        }

        if (parameters.contains('W'))
        {
            m_prev_table_offset = m_table_offset;
            m_table_offset = parameters['W'] * Constants::OpenGL::kObjectToView;

            //we don't draw segments for commands that are just table shifts, so only update end_pos if XY changes too
            if (parameters.contains('X') || parameters.contains('Y') )
            {
                end_pos.setZ(m_start_pos.z() + m_prev_table_offset - m_table_offset);
                m_prev_table_offset = parameters['W'] * Constants::OpenGL::kObjectToView; //accounted for the table offset, so no need for prev
            }
        }

        //! \note optional_parameters hold start locations for gcommand.  Syntaxes that use this do not have travels.

        // Always draw segments if X and Y are specified (which should be all segments except table-shifts between layers)
        if (parameters.contains('X'))
        {
            info_end_pos.setX(parameters['X'] + m_x_offset);
            end_pos.setX(((parameters['X'] + m_x_offset) * Constants::OpenGL::kObjectToView));
            if(optional_parameters.contains('X'))
                m_start_pos.setX(((optional_parameters['X']) * Constants::OpenGL::kObjectToView));
        }
        if (parameters.contains('Y'))
        {
            info_end_pos.setY(parameters['Y'] + m_y_offset);
            end_pos.setY(((parameters['Y'] + m_y_offset) * Constants::OpenGL::kObjectToView));
            if(optional_parameters.contains('Y'))
                m_start_pos.setY(((optional_parameters['Y']) * Constants::OpenGL::kObjectToView));
        }

        // Z is special! Must account for the table movement and the fact that z=0
        // is not on the floor of the printer
        if (parameters.contains('Z'))
        {
            info_end_pos.setZ(parameters['Z']);
            end_pos.setZ((parameters['Z'] * Constants::OpenGL::kObjectToView) + m_z_offset - m_table_offset);
            if(optional_parameters.contains('Z'))
                m_start_pos.setZ(((optional_parameters['Z']) * Constants::OpenGL::kObjectToView));

            //if the z is specified, it accounts for any table shift that might have happened in this or any previous command
            m_prev_table_offset = m_table_offset;
        }
        // Z wasn't specified, but it might need updated because of a table shift in a previous command
        // (if the table shift was in this command, we would've accounted for it already)
        else if (!qFuzzyCompare(m_prev_table_offset, m_table_offset) && (parameters.contains('X') || parameters.contains('Y')))
        {
            info_end_pos.setZ(m_info_start_pos.z() + m_prev_table_offset - m_table_offset);
            end_pos.setZ(m_start_pos.z() + m_prev_table_offset - m_table_offset);

            //we've accounted for the change in w, don't want to do it again until the table moves (w_offset changes) again
            m_prev_table_offset = m_table_offset;
        }

        QVector<QSharedPointer<SegmentBase>> generated_segments;

        for (int i = 0, end = extruders_on.size(); i < end; ++i)
        {
            if (extruders_on[i] || is_travel)
            {
                QSharedPointer<SegmentBase> segment;

                SegmentDisplayType type;

                if(color == PM->getVisualizationColor(VisualizationColors::kTravel))
                    type = SegmentDisplayType::kTravel;
                else if(color == PM->getVisualizationColor(VisualizationColors::kSupport))
                    type = SegmentDisplayType::kSupport;
                else
                    type = SegmentDisplayType::kLine;

                float display_height = 0;

                QVector3D extruder_offset = extruder_offsets[i].toQVector3D() * Constants::OpenGL::kObjectToView;

                // Builds and draws segments according to their type (Line, Arc, Spline)
                if(command_id == 2 || command_id == 3) // G2 clockwise arc & G3 counter-clockwise arc
                {
                    // Parse extra params
                    Point center;

                    if(parameters.contains('R') && !parameters.contains('I') && !parameters.contains('J'))
                    {
                        Distance R(parameters['R'] * Constants::OpenGL::kObjectToView);
                        Distance H = m_start_pos.distanceToPoint(end_pos) / 2.0;
                        Distance L = qSqrt(((R * R) - (H * H))());

                        center = m_start_pos;
                        center.moveTowards(end_pos, H);

                        QVector3D se = end_pos - m_start_pos;
                        QVector3D perp(se.y(), -se.x(), se.z());
                        perp.normalize();
                        perp *= L();

                        if(command_id == 2)
                            center += perp;
                        else
                            center -= perp;

                        segment = QSharedPointer<ArcSegment>::create(m_start_pos + extruder_offset, end_pos + extruder_offset, center + extruder_offset, (command_id == 3));

                    }else if (parameters.contains('I') && parameters.contains('J'))
                    {
                        // Determine center from I, J
                        center.x(m_start_pos.x() + ((parameters['I']) * Constants::OpenGL::kObjectToView));
                        center.y(m_start_pos.y() + ((parameters['J']) * Constants::OpenGL::kObjectToView));

                        if(parameters.contains('K'))
                            center.z(m_start_pos.z() + ((parameters['K']) * Constants::OpenGL::kObjectToView));
                        else
                            center.z(m_start_pos.z());

                        segment = QSharedPointer<ArcSegment>::create(m_start_pos + extruder_offset, end_pos + extruder_offset, center + extruder_offset, (command_id == 3));
                    }
                }else if(command_id == 5) // G5 splines
                {
                    Point control_a;
                    Point control_b;

                    if(parameters.contains('I') && parameters.contains('J') && parameters.contains('P') && parameters.contains('Q'))
                    {
                        control_a.x(m_start_pos.x() + ((parameters['I']) *  Constants::OpenGL::kObjectToView));
                        control_a.y(m_start_pos.y() + ((parameters['J']) * Constants::OpenGL::kObjectToView));
                        control_b.x(end_pos.x() + ((parameters['P']) * Constants::OpenGL::kObjectToView));
                        control_b.y(end_pos.y() + ((parameters['Q']) * Constants::OpenGL::kObjectToView));

                        segment = QSharedPointer<BezierSegment>::create(m_start_pos + extruder_offset, control_a + extruder_offset, control_b + extruder_offset, end_pos + extruder_offset);
                    }
                }
                else // G0, G1, or anything else is drawn as a line
                {
                    display_height = m_start_pos.distanceToPoint(end_pos);
                    segment = QSharedPointer<LineSegment>::create(m_start_pos + extruder_offset, end_pos - m_start_pos);
                }

                if(m_modifier_colors.contains(color))
                    segment->setGCodeInfo(m_segment_width * 1.1, display_height, type, color, line_num, layer_num);
                else
                    segment->setGCodeInfo(m_segment_width, display_height, type, color, line_num, layer_num);

                segment->m_segment_info_meta.type = comment;
                segment->m_segment_info_meta.start = m_info_start_pos;
                segment->m_segment_info_meta.end = info_end_pos;
                segment->m_segment_info_meta.speed = m_info_speed;
                if(!extruders_on[i] && !info_speed_set)
                    segment->m_segment_info_meta.speed = "";
                segment->m_segment_info_meta.extruderSpeed = extruders_on[i] ?
                        (m_info_extruder_speed.isEmpty() ?
                            QString().asprintf("%0.4f", extruders_speed) % " rpm" :
                            m_info_extruder_speed) : "";
                segment->m_segment_info_meta.length =
                        QString().asprintf("%0.2f", (Distance(m_info_start_pos.distanceToPoint(info_end_pos)) / PM->getDistanceUnit())()) % " " % PM->getDistanceUnitText();

                generated_segments.append(segment);
            }
        }
        //Update our start position for the next command
        m_start_pos = end_pos;
        m_info_start_pos = info_end_pos;

        return generated_segments;
    }
}  // namespace ORNL
