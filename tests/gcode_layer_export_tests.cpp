#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QRegularExpression>
#include <QStringList>
#include <QTemporaryDir>
#include <iostream>

#include "configs/settings_base.h"
#include "gcode/gcode_layer_export.h"
#include "gcode/writers/arc_specialties_writer.h"
#include "gcode/writers/cincinnati_writer.h"
#include "gcode/writers/haas_writer.h"
#include "gcode/writers/marlin_writer.h"
#include "gcode/writers/mvp_writer.h"
#include "gcode/writers/wolf_writer.h"
#include "geometry/segments/arc.h"
#include "geometry/segments/line.h"
#include "geometry/segments/travel.h"
#include "threading/gcode_layer_saver.h"
#include "utilities/constants.h"

namespace {
QSharedPointer<ORNL::SettingsBase> writerSettings() {
    auto settings = QSharedPointer<ORNL::SettingsBase>::create();
    settings->setSetting(ORNL::PS::Travel::kFinalLiftDistance, 2.0 * ORNL::mm);
    settings->setSetting(ORNL::PS::Slicing::kSlicePlaneNormalZ, 1.0);
    settings->setSetting(ORNL::PS::Layer::kLayerHeight, 10.0 * ORNL::mm);
    settings->setSetting(ORNL::PRS::MachineSpeed::kZSpeed, 5.0 * ORNL::mm / ORNL::s);
    settings->setSetting(ORNL::PS::Travel::kSpeed, 10.0 * ORNL::mm / ORNL::s);
    settings->setSetting(ORNL::PRS::GCode::kEndCode, QString("M400"));
    settings->setSetting(ORNL::PRS::GCode::kEnableStartupCode, 1);
    settings->setSetting(ORNL::PRS::Dimensions::kWMax, 100.0 * ORNL::mm);
    settings->setSetting(ORNL::PRS::Dimensions::kInitialW, 100.0 * ORNL::mm);
    settings->setSetting(ORNL::PRS::Dimensions::kZOffset, -150.0 * ORNL::mm);
    settings->setSetting(ORNL::PRS::Dimensions::kLayerChangeAxis, ORNL::LayerChange::kBoth_Z_and_W);
    settings->setSetting(ORNL::SS::kSpeed, 5.0 * ORNL::mm / ORNL::s);
    settings->setSetting(ORNL::SS::kExtruderSpeed, 10);
    settings->setSetting(ORNL::SS::kRegionType, ORNL::RegionType::kPerimeter);
    settings->setSetting(ORNL::SS::kPathModifiers, ORNL::PathModifiers::kNone);
    settings->setSetting(ORNL::PRS::MachineSpeed::kGearRatio, 1.0);
    return settings;
}

QString paths(const QSharedPointer<ORNL::WriterBase>& writer, const QSharedPointer<ORNL::SettingsBase>& settings,
              int layer, bool travel, bool arc = false) {
    const ORNL::Point start(10.0 * ORNL::mm, 20.0 * ORNL::mm, (layer + 1) * 10.0 * ORNL::mm);
    const ORNL::Point end(30.0 * ORNL::mm, 40.0 * ORNL::mm, start.z());
    QString result = writer->writeBeforeRegion(ORNL::RegionType::kPerimeter);
    result += writer->writeBeforePath(ORNL::RegionType::kPerimeter);
    if (travel) {
        ORNL::TravelSegment approach(ORNL::Point(0, 0, start.z() - 10.0 * ORNL::mm()), start,
                                     ORNL::TravelLiftType::kLiftUpOnly);
        approach.setSb(settings);
        result += approach.writeGCode(writer);
    }
    if (arc) {
        ORNL::ArcSegment segment(start, end, ORNL::Point(start.x(), end.y(), start.z()), false);
        segment.setSb(settings);
        result += segment.writeGCode(writer);
    }
    else {
        ORNL::LineSegment segment(start, end);
        segment.setSb(settings);
        result += segment.writeGCode(writer);
    }
    result += writer->writeAfterPath(ORNL::RegionType::kPerimeter);
    result += writer->writeAfterRegion(ORNL::RegionType::kPerimeter);
    return result;
}

template <typename Writer>
QString program(const ORNL::GcodeMeta& meta, const QSharedPointer<ORNL::SettingsBase>& settings, bool travel = true,
                bool arc = false, bool metadata = true, int empty_layer = -1) {
    auto writer      = QSharedPointer<Writer>::create(meta, settings);
    const auto setup = [](const QSharedPointer<Writer>& target) { return target->writeInitialSetup(0, 0, 0, 0, 3); };
    QString source   = setup(writer);
    source += writer->writeCommentLine(ORNL::kLayerHeaderEnd);
    for (int layer = 0; layer < 3; ++layer) {
        source += writer->writeLayerChange(layer);
        source += writer->writeBeforeLayer((layer + 1) * 10.0 * ORNL::mm(), settings);
        source += layer == empty_layer ? writer->writeEmptyStep() : paths(writer, settings, layer, travel, arc);
        source += writer->writeAfterLayer();

        if (metadata) {
            auto independent = QSharedPointer<Writer>::create(meta, settings);
            setup(independent);
            const QString standalone =
                independent->writeStandaloneLayer(layer, (layer + 1) * 10.0 * ORNL::mm(), settings, [&]() {
                    return layer == empty_layer ? independent->writeEmptyStep()
                                                : paths(independent, settings, layer, travel, arc);
                });
            source += writer->writeLayerExportProgram(standalone);
        }
    }
    source += writer->writeAfterPart();
    source += writer->writeShutdownWithLayerMarkers();
    return source + meta.m_comment_starting_delimiter + "settings footer" + meta.m_comment_ending_delimiter + '\n';
}

QString withoutMetadata(const QString& source) {
    QStringList result;
    bool metadata = false;
    for (const QString& line : source.split('\n')) {
        if (line.contains(ORNL::kLayerProgramBegin)) {
            metadata = true;
            continue;
        }
        if (line.contains(ORNL::kLayerProgramEnd)) {
            metadata = false;
            continue;
        }
        if (!metadata) { result.push_back(line); }
    }
    return result.join('\n');
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
    const auto settings         = writerSettings();
    const QString marlin_source = program<ORNL::MarlinWriter>(ORNL::GcodeMetaList::MarlinMeta, settings);
    QString error               = "old error";
    auto layers                 = ORNL::splitGcodeIntoLayers(marlin_source, ";", &error);
    expect(error.isEmpty() && layers.size() == 3, "Expected three independently generated layers and no error.");
    if (layers.size() == 3) {
        const QString header = marlin_source.left(marlin_source.indexOf(";END LAYER EXPORT HEADER"));
        for (int i = 0; i < layers.size(); ++i) {
            expect(layers[i].number == i + 1 && layers[i].text.startsWith(header),
                   "Expected each layer's number and complete original header.");
            for (const QString& command : {QString("M400"), QString("M104 S0"), QString("M140 S0"), QString("M84")}) {
                const int lift = layers[i].text.indexOf("TRAVEL FINAL LIFT");
                expect(lift >= 0 && layers[i].text.mid(lift).count(command) == 1,
                       "Expected a complete dialect shutdown exactly once after the final lift.");
            }
            expect(layers[i].text.contains("TRAVEL FINAL LIFT") && !layers[i].text.contains(ORNL::kLayerProgramData),
                   "Expected a real final lift with all encoded metadata removed.");
        }
        expect(layers.last().text.endsWith(";settings footer\n"), "Expected the final settings footer to survive.");
    }

    const QString cincinnati_source = program<ORNL::CincinnatiWriter>(ORNL::GcodeMetaList::CincinnatiMeta, settings);
    expect(withoutMetadata(cincinnati_source) ==
               program<ORNL::CincinnatiWriter>(ORNL::GcodeMetaList::CincinnatiMeta, settings, true, false, false),
           "Expected standalone generation to leave continuous Cincinnati commands unchanged.");
    layers = ORNL::splitGcodeIntoLayers(cincinnati_source, "(", &error);
    expect(error.isEmpty() && layers.size() == 3, "Expected parenthesized metadata to decode correctly.");
    if (layers.size() == 3) {
        const QString body = layers[1].text.mid(layers[1].text.indexOf("(BEGINNING LAYER: 2)"));
        // Fresh W=100, one 10 mm layer change => W=90. At part Z=20, Z = 20 - 150 + 90 = -40 mm.
        const QString expected_z = " Z" + QString::number((-40.0 * ORNL::mm).to(ORNL::in), 'f', 4);
        const QString expected_w = " W" + QString::number((90.0 * ORNL::mm).to(ORNL::in), 'f', 4);
        expect(body.contains(expected_z) && body.contains(expected_w) &&
                   body.indexOf(expected_z) < body.indexOf("TURN EXTRUDER ON"),
               "Expected the second standalone layer to establish its own Z/W coordinates despite zero travel lift.");
        expect(body.contains("M5") && body.contains("M30"), "Expected Cincinnati's own shutdown.");
    }

    const QString mvp_source = program<ORNL::MVPWriter>(ORNL::GcodeMetaList::MVPMeta, settings);
    expect(withoutMetadata(mvp_source) ==
               program<ORNL::MVPWriter>(ORNL::GcodeMetaList::MVPMeta, settings, true, false, false),
           "Expected standalone generation to leave continuous MVP commands unchanged.");
    layers = ORNL::splitGcodeIntoLayers(mvp_source, ";");
    expect(layers.size() == 3, "Expected three MVP layers.");
    if (layers.size() == 3) {
        for (int i = 0; i < layers.size(); ++i) {
            const QString text   = layers[i].text;
            const int lift       = text.indexOf("TRAVEL FINAL LIFT Z");
            const int park       = text.indexOf("G0 X0 Y0", lift);
            const QString lift_z = " Z" + QString::number(-150.0 - (i + 1) * 10.0 - 2.0, 'f', 4);
            expect(lift >= 0 && park > lift && text.left(lift).endsWith(lift_z + " ;"),
                   "Expected each MVP layer's own endpoint plus clearance before its XY park.");
        }
    }

    settings->setSetting(ORNL::PS::Slicing::kSlicePlaneNormalX, 1.0);
    layers = ORNL::splitGcodeIntoLayers(program<ORNL::MVPWriter>(ORNL::GcodeMetaList::MVPMeta, settings), ";");
    const QRegularExpression tilted_lift("G0 X([0-9.]+) Y([0-9.]+) Z(-[0-9.]+) ;TRAVEL FINAL LIFT");
    const auto tilted_match = layers.size() == 3 ? tilted_lift.match(layers[0].text) : QRegularExpressionMatch();
    // Point coordinates use floats in internal units; compare within one micron, rather than formatting digits.
    expect(tilted_match.hasMatch() && qAbs(tilted_match.captured(1).toDouble() - 31.41421356) < 0.001 &&
               qAbs(tilted_match.captured(2).toDouble() - 40.0) < 0.001 &&
               qAbs(tilted_match.captured(3).toDouble() + 161.41421356) < 0.001,
           "Expected final clearance along the configured slicing normal from the layer endpoint.");
    settings->setSetting(ORNL::PS::Slicing::kSlicePlaneNormalX, 0.0);

    settings->setSetting(ORNL::PS::Travel::kFinalLiftDistance, 0.0 * ORNL::mm);
    layers = ORNL::splitGcodeIntoLayers(program<ORNL::MVPWriter>(ORNL::GcodeMetaList::MVPMeta, settings), ";");
    expect(layers.size() == 3 && !layers[0].text.contains("TRAVEL FINAL LIFT"),
           "Expected zero configured final lift to remain zero.");
    settings->setSetting(ORNL::PS::Travel::kFinalLiftDistance, 2.0 * ORNL::mm);

    // Spiral layers can begin with a printing segment, without a preceding TravelSegment.
    settings->setSetting(ORNL::PS::SpecialModes::kEnableSpiralize, true);
    layers = ORNL::splitGcodeIntoLayers(program<ORNL::MVPWriter>(ORNL::GcodeMetaList::MVPMeta, settings, false), ";");
    expect(layers.size() == 3 && layers[1].text.contains("TURN GUN ON") &&
               layers[1].text.contains("TURN EXTRUDER ON") &&
               layers[1].text.indexOf(";TRAVEL") < layers[1].text.indexOf("TURN GUN ON"),
           "Expected a fresh approach and pump startup for a standalone spiral layer.");
    layers = ORNL::splitGcodeIntoLayers(
        program<ORNL::CincinnatiWriter>(ORNL::GcodeMetaList::CincinnatiMeta, settings, false, true), "(");
    expect(layers.size() == 3 && layers[1].text.contains("TRAVEL") && layers[1].text.contains("G2"),
           "Expected an approach before an arc-only standalone layer.");
    settings->setSetting(ORNL::PS::SpecialModes::kEnableSpiralize, false);

    expect(ORNL::splitGcodeIntoLayers(program<ORNL::HaasWriter>(ORNL::GcodeMetaList::HaasMetricMeta, settings), "(")
                   .size() == 3,
           "Expected Haas standalone output.");
    expect(
        ORNL::splitGcodeIntoLayers(program<ORNL::WolfWriter>(ORNL::GcodeMetaList::WolfMeta, settings), ";").size() == 3,
        "Expected Wolf standalone output.");

    layers = ORNL::splitGcodeIntoLayers(
        program<ORNL::ArcSpecialtiesWriter>(ORNL::GcodeMetaList::ArcSpecialtiesMeta, settings, true, false, true, 0),
        ";");
    expect(layers.size() == 3 && layers[0].text.contains("INTENTIONALLY BLANK") && layers[1].text.contains("#KIN ID"),
           "Expected deferred layer markers and an initial empty robot layer to retain independent startup state.");

    const QString legacy = "; header\n;BEGINNING LAYER: 1\nG1 X1\n;BEGINNING LAYER: 2\nG1 X2\nM30\n";
    expect(ORNL::splitGcodeIntoLayers(legacy, ";", &error).isEmpty() && error.contains("Re-slice"),
           "Expected legacy multi-layer input to require re-slicing.");
    const QString marked_legacy =
        legacy.left(legacy.lastIndexOf("M30")) + ";BEGIN LAYER EXPORT SHUTDOWN\nM30\n;END LAYER EXPORT SHUTDOWN\n";
    expect(ORNL::splitGcodeIntoLayers(marked_legacy, ";", &error).isEmpty() && error.contains("Re-slice"),
           "Expected old shutdown markers alone to be insufficient to reconstruct standalone entry and clearance.");
    const QString single = ";BEGINNING LAYER: 1\nG1 X1\nM30\n";
    layers               = ORNL::splitGcodeIntoLayers(single, ";", &error);
    expect(layers.size() == 1 && layers[0].text == single && error.isEmpty(),
           "Expected an existing genuinely single-layer program to remain intact.");

    QString malformed = marlin_source;
    malformed.remove(";END LAYER EXPORT PROGRAM\n");
    expect(ORNL::splitGcodeIntoLayers(malformed, ";", &error).isEmpty() && !error.isEmpty(),
           "Expected truncated standalone metadata to fail before exporting.");
    malformed = marlin_source;
    malformed.replace("LAYER EXPORT DATA: ", "LAYER EXPORT DATA: !");
    expect(ORNL::splitGcodeIntoLayers(malformed, ";", &error).isEmpty(), "Expected corrupt encoded data to fail.");
    malformed = marlin_source;
    malformed.replace(";BEGINNING LAYER: 2", ";BEGINNING LAYER: 99");
    expect(ORNL::splitGcodeIntoLayers(malformed, ";", &error).isEmpty(), "Expected mismatched layer identity to fail.");
    malformed = marlin_source;
    malformed.remove(";END LAYER EXPORT SHUTDOWN\n");
    expect(ORNL::splitGcodeIntoLayers(malformed, ";", &error).isEmpty(), "Expected a missing shutdown end to fail.");

    QString crlf = marlin_source;
    crlf.replace("\n", "\r\n");
    expect(ORNL::splitGcodeIntoLayers(crlf, ";").size() == 3, "Expected CRLF metadata to decode.");
    expect(ORNL::splitGcodeIntoLayers("; header\nG1 X1\n", ";").isEmpty(), "Expected no layers without markers.");

    QTemporaryDir directory;
    expect(directory.isValid(), "Expected a temporary output directory.");
    if (directory.isValid()) {
        const QString stale_path = QDir(directory.path()).filePath("part_layer_99.gcode");
        QFile stale(stale_path);
        expect(stale.open(QIODevice::WriteOnly), "Expected a pre-existing layer file for the worker test.");
        stale.write("original\n");
        stale.close();
        ORNL::GCodeLayerSaver rejected(directory.path(), "part", malformed, ";", ".gcode");
        QString worker_error;
        QObject::connect(&rejected, &ORNL::GCodeLayerSaver::completed,
                         [&worker_error](const QString& message) { worker_error = message; });
        rejected.run();
        expect(!worker_error.isEmpty() && readFile(stale_path) == "original\n" &&
                   QDir(directory.path()).entryList(QDir::Files).size() == 1,
               "Expected invalid metadata to preserve existing files and produce no outputs.");

        ORNL::GCodeLayerSaver saver(directory.path(), "part", mvp_source, ";", ".mpf");
        worker_error = "not completed";
        QObject::connect(&saver, &ORNL::GCodeLayerSaver::completed,
                         [&worker_error](const QString& message) { worker_error = message; });
        saver.run();
        const QString saved = readFile(QDir(directory.path()).filePath("part_layer_1.mpf"));
        expect(worker_error.isEmpty() && saved.contains("TRAVEL FINAL LIFT Z") && saved.contains("G0 X0 Y0"),
               "Expected the worker to persist each layer's clearance and parking commands.");
    }

    expect(ORNL::isGcodeLayerFileName("part_layer_12.gcode", "part", ".gcode"), "Expected a generated name.");
    expect(ORNL::isGcodeLayerFileName("part[1]_layer_12.gcode", "part[1]", ".gcode"), "Expected literal part names.");
    expect(!ORNL::isGcodeLayerFileName("part_layer_12.gcode.bak", "part", ".gcode"), "Expected an exact suffix.");
    expect(!ORNL::isGcodeLayerFileName("part_layer_notes.gcode", "part", ".gcode"), "Expected numeric layer names.");
    expect(!ORNL::isGcodeLayerFileName("part_layer_012.gcode", "part", ".gcode"), "Expected canonical layer numbers.");
    return passed ? 0 : 1;
}
