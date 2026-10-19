#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QStringList>
#include <QTemporaryDir>
#include <iostream>

#include "configs/settings_base.h"
#include "gcode/gcode_layer_export.h"
#include "gcode/writers/cincinnati_writer.h"
#include "gcode/writers/haas_writer.h"
#include "gcode/writers/marlin_writer.h"
#include "gcode/writers/wolf_writer.h"
#include "threading/gcode_layer_saver.h"
#include "utilities/constants.h"

namespace {
QSharedPointer<ORNL::SettingsBase> shutdownSettings() {
    auto settings = QSharedPointer<ORNL::SettingsBase>::create();
    settings->setSetting(ORNL::PS::Travel::kFinalLiftDistance, 2.0 * ORNL::mm);
    settings->setSetting(ORNL::PS::Slicing::kSlicePlaneNormalZ, 1.0);
    settings->setSetting(ORNL::PRS::MachineSpeed::kZSpeed, 5.0 * ORNL::mm / ORNL::s);
    settings->setSetting(ORNL::PRS::GCode::kEndCode, QString("M400"));
    settings->setSetting(ORNL::PRS::Dimensions::kWMax, 100.0 * ORNL::mm);
    settings->setSetting(ORNL::PRS::Dimensions::kInitialW, 100.0 * ORNL::mm);
    settings->setSetting(ORNL::PRS::Dimensions::kEnableDoffing, 1);
    settings->setSetting(ORNL::PRS::Dimensions::kDoffingHeight, 10.0 * ORNL::mm);
    return settings;
}

QString layerText(const ORNL::GcodeMeta& meta, int layer) {
    return meta.m_comment_starting_delimiter + "BEGINNING LAYER: " + QString::number(layer) +
           meta.m_comment_ending_delimiter + "\nG1 X" + QString::number(layer) + '\n';
}

QString readFile(const QString& path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) { return {}; }
    return QString::fromUtf8(file.readAll());
}
}  // namespace

int main(int argc, char* argv[]) {
    QCoreApplication app(argc, argv);
    bool passed       = true;
    const auto expect = [&passed](bool condition, const char* message) {
        if (!condition) {
            std::cerr << message << '\n';
            passed = false;
        }
    };

    const QString gcode =
        "; machine header\n"
        ";BEGINNING LAYER: 1\n"
        "G1 X1\n"
        ";BEGINNING LAYER: 2\n"
        "G1 X2\n"
        ";BEGIN LAYER EXPORT SHUTDOWN\n"
        "M30\n"
        ";END LAYER EXPORT SHUTDOWN\n";
    const QVector<ORNL::GcodeLayer> layers = ORNL::splitGcodeIntoLayers(gcode, ";");

    expect(layers.size() == 2, "Expected one output for each G-code layer.");
    if (layers.size() == 2) {
        expect(layers[0].number == 1 && layers[1].number == 2,
               "Expected output layers to retain their source layer numbers.");
        expect(layers[0].text == "; machine header\n;BEGINNING LAYER: 1\nG1 X1\nM30\n",
               "Expected the original header, first layer, and shutdown in the first output.");
        expect(layers[1].text == "; machine header\n" + gcode.mid(gcode.indexOf(";BEGINNING LAYER: 2")),
               "Expected the final layer and its complete original shutdown to be preserved.");
    }

    // Use actual writer output so dialect commands and final-position exclusions are tested together.
    const auto check_writer = [&expect](ORNL::WriterBase& writer, const ORNL::GcodeMeta& meta,
                                        const QStringList& required_commands) {
        writer.writeInitialSetup(ORNL::Distance(0), ORNL::Distance(0), ORNL::Distance(0), ORNL::Distance(0), 2);
        writer.setCurrentPosition(ORNL::Point(30.0 * ORNL::mm, 40.0 * ORNL::mm, 50.0 * ORNL::mm));
        const QString shutdown = writer.writeShutdownWithLayerMarkers();
        const QString header =
            meta.m_comment_starting_delimiter + "machine header" + meta.m_comment_ending_delimiter + '\n';
        const QString footer =
            meta.m_comment_starting_delimiter + "settings footer" + meta.m_comment_ending_delimiter + '\n';
        const QString source = header + layerText(meta, 1) + layerText(meta, 2) + shutdown + footer;
        QString error        = "old error";
        const auto outputs   = ORNL::splitGcodeIntoLayers(source, meta.m_comment_starting_delimiter, &error);
        expect(error.isEmpty(), "Expected valid shutdown metadata to clear the error output.");
        expect(outputs.size() == 2, "Expected two layers from actual writer shutdown output.");
        if (outputs.size() != 2) { return QString(); }
        for (const QString& command : required_commands) {
            expect(outputs[0].text.count(command) == 1 && outputs[1].text.count(command) == 1,
                   "Expected every layer to contain each dialect shutdown command exactly once.");
        }
        expect(outputs[0].text.startsWith(header + layerText(meta, 1)),
               "Expected the header and first layer to be preserved.");
        expect(!outputs[0].text.contains("TRAVEL FINAL LIFT") && !outputs[0].text.contains("JOG W TO DOFFING LOCATION"),
               "Expected non-final layers to exclude moves calculated from the final writer position.");
        expect(outputs[1].text == header + layerText(meta, 2) + shutdown + footer,
               "Expected the last layer's shutdown moves and settings footer to remain unchanged.");
        expect(outputs[1].text.contains("TRAVEL FINAL LIFT"),
               "Expected the last layer to retain the configured final lift.");
        return source;
    };

    const auto settings = shutdownSettings();
    ORNL::MarlinWriter marlin(ORNL::GcodeMetaList::MarlinMeta, settings);
    const QString marlin_source =
        check_writer(marlin, ORNL::GcodeMetaList::MarlinMeta, {"M400", "M104 S0", "M140 S0", "M84"});
    ORNL::HaasWriter haas(ORNL::GcodeMetaList::HaasMetricMeta, settings);
    check_writer(haas, ORNL::GcodeMetaList::HaasMetricMeta, {"M400", "M19", "M30", "%"});
    ORNL::CincinnatiWriter cincinnati(ORNL::GcodeMetaList::CincinnatiMeta, settings);
    const QString cincinnati_source =
        check_writer(cincinnati, ORNL::GcodeMetaList::CincinnatiMeta, {"M400", "M5", "M30"});
    expect(cincinnati_source.contains("JOG W TO DOFFING LOCATION"),
           "Expected the Cincinnati test to exercise the state-dependent doffing move.");
    ORNL::WolfWriter wolf(ORNL::GcodeMetaList::WolfMeta, settings);
    check_writer(wolf, ORNL::GcodeMetaList::WolfMeta, {});

    const QString legacy = "; header\n;BEGINNING LAYER: 1\nG1 X1\n;BEGINNING LAYER: 2\nG1 X2\nM30\n";
    QString error;
    expect(ORNL::splitGcodeIntoLayers(legacy, ";", &error).isEmpty() && error.contains("Re-slice"),
           "Expected legacy multi-layer input to require re-slicing instead of producing unterminated files.");
    const QString single_layer = ";BEGINNING LAYER: 1\nG1 X1\nM30\n";
    const auto single_output   = ORNL::splitGcodeIntoLayers(single_layer, ";", &error);
    expect(single_output.size() == 1 && single_output[0].text == single_layer && error.isEmpty(),
           "Expected a single layer's existing shutdown to be sufficient without markers.");

    QString malformed = gcode;
    malformed.remove(";END LAYER EXPORT SHUTDOWN\n");
    expect(ORNL::splitGcodeIntoLayers(malformed, ";", &error).isEmpty() && !error.isEmpty(),
           "Expected an unterminated shutdown block to fail before exporting.");
    malformed = gcode;
    malformed.replace("M30\n", ";BEGIN FINAL POSITION MOVES\nM30\n");
    expect(ORNL::splitGcodeIntoLayers(malformed, ";", &error).isEmpty(),
           "Expected an unterminated final-position block to fail before exporting.");

    QString buffered_layer = gcode;
    buffered_layer.replace(
        "M30\n", ";BEGIN FINAL POSITION MOVES\n;BEGINNING LAYER: 99\nG0 X99\n;END FINAL POSITION MOVES\nM30\n");
    const auto buffered_outputs = ORNL::splitGcodeIntoLayers(buffered_layer, ";");
    expect(buffered_outputs.size() == 2 && !buffered_outputs[0].text.contains("99"),
           "Expected buffered final-layer markers and moves to remain in the final layer only.");

    QString crlf = gcode;
    crlf.replace("\n", "\r\n");
    const auto crlf_outputs = ORNL::splitGcodeIntoLayers(crlf, ";");
    expect(crlf_outputs.size() == 2 && crlf_outputs[0].text.endsWith("M30\r\n"),
           "Expected CRLF shutdown commands to be copied intact.");

    QTemporaryDir directory;
    expect(directory.isValid(), "Expected a temporary output directory.");
    if (directory.isValid()) {
        const QString stale_path = QDir(directory.path()).filePath("part_layer_99.gcode");
        QFile stale_file(stale_path);
        expect(stale_file.open(QIODevice::WriteOnly), "Expected a pre-existing layer file for the worker test.");
        stale_file.write("original\n");
        stale_file.close();
        ORNL::GCodeLayerSaver invalid_saver(directory.path(), "part", legacy, ";", ".gcode");
        QString worker_error;
        QObject::connect(&invalid_saver, &ORNL::GCodeLayerSaver::completed,
                         [&worker_error](const QString& message) { worker_error = message; });
        invalid_saver.run();
        expect(!worker_error.isEmpty() && readFile(stale_path) == "original\n" &&
                   QDir(directory.path()).entryList(QDir::Files).size() == 1,
               "Expected rejected shutdown metadata to leave existing files intact and write no new layers.");

        ORNL::GCodeLayerSaver saver(directory.path(), "part", marlin_source, ";", ".gcode");
        worker_error = "not completed";
        QObject::connect(&saver, &ORNL::GCodeLayerSaver::completed,
                         [&worker_error](const QString& message) { worker_error = message; });
        saver.run();
        expect(worker_error.isEmpty(), "Expected the worker to report success for marked writer output.");
        const QString first_file = readFile(QDir(directory.path()).filePath("part_layer_1.gcode"));
        expect(first_file.contains("M104 S0") && first_file.contains("M140 S0") && first_file.contains("M84") &&
                   !first_file.contains("TRAVEL FINAL LIFT"),
               "Expected the saved non-final layer to shut down heaters and motors without the final-position lift.");
    }

    const QVector<ORNL::GcodeLayer> no_layers = ORNL::splitGcodeIntoLayers("; machine header\nG1 X1\n", ";");
    expect(no_layers.isEmpty(), "Expected G-code without layer markers to produce no layers.");

    expect(ORNL::isGcodeLayerFileName("part_layer_12.gcode", "part", ".gcode"),
           "Expected a generated layer file name to be recognized.");
    expect(ORNL::isGcodeLayerFileName("part[1]_layer_12.gcode", "part[1]", ".gcode"),
           "Expected special characters in the part name to be matched literally.");
    expect(!ORNL::isGcodeLayerFileName("part_layer_12.gcode.bak", "part", ".gcode"),
           "Expected files with a different suffix not to be recognized.");
    expect(!ORNL::isGcodeLayerFileName("part_layer_notes.gcode", "part", ".gcode"),
           "Expected non-numeric layer names not to be recognized.");
    expect(!ORNL::isGcodeLayerFileName("part_layer_012.gcode", "part", ".gcode"),
           "Expected names not produced by the exporter not to be recognized.");

    return passed ? 0 : 1;
}
