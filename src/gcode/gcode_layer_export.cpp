#include "gcode/gcode_layer_export.h"

#include <QRegularExpression>
#include <QStringList>

#include "gcode/gcode_meta.h"

namespace ORNL {
QString layerExportProgram(const QString& program, const GcodeMeta& meta) {
    const QByteArray encoded = qCompress(program.toUtf8()).toBase64();
    const auto comment       = [&meta](const QString& value) {
        return meta.m_comment_starting_delimiter + value + meta.m_comment_ending_delimiter + '\n';
    };
    QString result = comment(kLayerProgramBegin);
    // Keep metadata safe for parenthesized comments and bounded controller line buffers.
    for (qsizetype offset = 0; offset < encoded.size(); offset += 48) {
        result += comment(QString(kLayerProgramData) + QString::fromLatin1(encoded.mid(offset, 48)));
    }
    return result + comment(kLayerProgramEnd);
}

QVector<GcodeLayer> splitGcodeIntoLayers(const QString& text, const QString& comment_starting_delimiter,
                                         QString* error) {
    if (error) { error->clear(); }
    const auto fail = [error](const QString& message) {
        if (error) { *error = message; }
        return QVector<GcodeLayer>();
    };
    const QString invalid =
        "Missing or invalid standalone layer programs. Re-slice before exporting individual layers.";
    if (comment_starting_delimiter.isEmpty()) { return fail("Could not find any G-code layer markers to export."); }

    const QRegularExpression layer_marker(
        "^\\s*" + QRegularExpression::escape(comment_starting_delimiter) + "\\s*BEGINNING\\s+LAYER\\s*:\\s*(\\d+)\\b",
        QRegularExpression::CaseInsensitiveOption | QRegularExpression::MultilineOption);
    const QStringList lines = text.split('\n');
    QVector<int> source_numbers;
    int header_end = -1;
    QVector<GcodeLayer> layers;
    QByteArray encoded;
    bool in_program        = false;
    bool in_shutdown       = false;
    bool in_final_position = false;
    bool shutdown_complete = false;
    int footer_start       = lines.size();

    for (int line_index = 0; line_index < lines.size(); ++line_index) {
        QString marker = lines[line_index].trimmed();
        if (marker.startsWith(comment_starting_delimiter)) {
            marker.remove(0, comment_starting_delimiter.size());
            if (comment_starting_delimiter == "(" && marker.endsWith(')')) { marker.chop(1); }
            marker = marker.trimmed();
        }
        else { marker.clear(); }

        if (marker == kLayerProgramBegin) {
            if (in_program || in_shutdown || shutdown_complete || header_end < 0) { return fail(invalid); }
            in_program = true;
            encoded.clear();
            continue;
        }
        if (marker == kLayerProgramEnd) {
            if (!in_program) { return fail(invalid); }
            const auto decoded = QByteArray::fromBase64Encoding(encoded, QByteArray::AbortOnBase64DecodingErrors);
            if (!decoded) { return fail(invalid); }
            const QByteArray program = qUncompress(decoded.decoded);
            if (program.isEmpty()) { return fail(invalid); }
            const QString standalone = QString::fromUtf8(program);
            auto matches             = layer_marker.globalMatch(standalone);
            if (!matches.hasNext()) { return fail(invalid); }
            bool converted   = false;
            const int number = matches.next().captured(1).toInt(&converted);
            if (!converted || matches.hasNext()) { return fail(invalid); }
            layers.push_back({number, standalone});
            in_program = false;
            continue;
        }
        if (in_program) {
            if (!marker.startsWith(kLayerProgramData)) { return fail(invalid); }
            encoded += marker.mid(QString(kLayerProgramData).size()).toLatin1();
            continue;
        }
        if (marker.startsWith(kLayerProgramData)) { return fail(invalid); }

        if (marker == kLayerHeaderEnd) {
            if (header_end >= 0 || !source_numbers.isEmpty() || !layers.isEmpty()) { return fail(invalid); }
            header_end = line_index;
            continue;
        }

        const auto match = layer_marker.match(lines[line_index]);
        if (match.hasMatch()) {
            bool converted   = false;
            const int number = match.captured(1).toInt(&converted);
            if (!converted || shutdown_complete) { return fail(invalid); }
            source_numbers.push_back(number);
        }

        if (marker == kLayerShutdownBegin) {
            if (in_shutdown || shutdown_complete) { return fail(invalid); }
            in_shutdown = true;
            continue;
        }
        if (marker == kLayerShutdownEnd) {
            if (!in_shutdown || in_final_position) { return fail(invalid); }
            in_shutdown       = false;
            shutdown_complete = true;
            footer_start      = line_index + 1;
            continue;
        }
        if (marker == kFinalPositionBegin) {
            if (!in_shutdown || in_final_position) { return fail(invalid); }
            in_final_position = true;
            continue;
        }
        if (marker == kFinalPositionEnd) {
            if (!in_shutdown || !in_final_position) { return fail(invalid); }
            in_final_position = false;
            continue;
        }
    }

    if (source_numbers.isEmpty()) { return fail("Could not find any G-code layer markers to export."); }
    if (in_program || in_shutdown || in_final_position) { return fail(invalid); }
    if (source_numbers.size() == 1 && layers.isEmpty()) {
        // A genuinely single-layer program already has its own startup and shutdown state.
        return {{source_numbers.first(), text}};
    }
    if (!shutdown_complete || header_end < 0 || layers.size() != source_numbers.size()) { return fail(invalid); }

    const QString header = lines.mid(0, header_end).join('\n');
    for (qsizetype i = 0; i < layers.size(); ++i) {
        if (layers[i].number != source_numbers[i]) { return fail(invalid); }
        if (header_end > 0) { layers[i].text.prepend(header + '\n'); }
    }
    layers.last().text += lines.mid(footer_start).join('\n');
    return layers;
}

bool isGcodeLayerFileName(const QString& file_name, const QString& part_name, const QString& suffix) {
    const QString prefix = part_name + "_layer_";
    if (!file_name.startsWith(prefix) || !file_name.endsWith(suffix)) { return false; }

    const qsizetype number_length = file_name.size() - prefix.size() - suffix.size();
    if (number_length <= 0) { return false; }

    const QString number_text = file_name.mid(prefix.size(), number_length);
    bool converted            = false;
    const int layer_number    = number_text.toInt(&converted);
    return converted && layer_number >= 0 && QString::number(layer_number) == number_text;
}
}  // namespace ORNL
