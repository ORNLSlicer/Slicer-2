#pragma once

#include <QString>
#include <QVector>

namespace ORNL {
struct GcodeMeta;
//! \brief Writer markers identifying standalone programs and the continuous program shutdown.
inline constexpr char kLayerHeaderEnd[]     = "END LAYER EXPORT HEADER";
inline constexpr char kLayerProgramBegin[]  = "BEGIN LAYER EXPORT PROGRAM";
inline constexpr char kLayerProgramEnd[]    = "END LAYER EXPORT PROGRAM";
inline constexpr char kLayerProgramData[]   = "LAYER EXPORT DATA: ";
inline constexpr char kLayerShutdownBegin[] = "BEGIN LAYER EXPORT SHUTDOWN";
inline constexpr char kLayerShutdownEnd[]   = "END LAYER EXPORT SHUTDOWN";
inline constexpr char kFinalPositionBegin[] = "BEGIN FINAL POSITION MOVES";
inline constexpr char kFinalPositionEnd[]   = "END FINAL POSITION MOVES";

//! \brief Encodes an independently generated layer as inert comments in the continuous program.
QString layerExportProgram(const QString& program, const GcodeMeta& meta);

//! \brief A single layer extracted from a generated G-code file.
struct GcodeLayer {
    int number;
    QString text;
};

//! \brief Splits generated G-code into layer files while preserving the original header.
//! \param text Complete G-code text, including its header.
//! \param comment_starting_delimiter G-code comment delimiter used before layer markers.
//! \param error Optional error output, cleared on success.
//! \return One entry per layer, with the shared header and independently generated entry/path/shutdown commands.
//! Multi-layer input without standalone program metadata must be re-sliced; no output is returned on invalid metadata.
QVector<GcodeLayer> splitGcodeIntoLayers(const QString& text, const QString& comment_starting_delimiter,
                                         QString* error = nullptr);

//! \brief Returns whether a file name matches the name generated for an individually exported layer.
//! \param file_name File name without its directory.
//! \param part_name Base name selected for the export.
//! \param suffix G-code file suffix, including its leading period.
bool isGcodeLayerFileName(const QString& file_name, const QString& part_name, const QString& suffix);
}  // namespace ORNL
