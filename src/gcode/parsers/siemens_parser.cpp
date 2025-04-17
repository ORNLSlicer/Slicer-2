#include "gcode/parsers/siemens_parser.h"

#include <QString>
#include <QStringList>
#include <QVector>

#include "exceptions/exceptions.h"
#include "units/unit.h"

namespace ORNL
{
    SiemensParser::SiemensParser(GcodeMeta meta, bool allowLayerAlter, QStringList& lines, QStringList& upperLines)
        : CommonParser(meta, allowLayerAlter, lines, upperLines)
    {
        config();
    }

    void SiemensParser::config()
    {
        CommonParser::config();

        addCommandMapping(
            "BEAD_AREA",
            std::bind(
                &SiemensParser::BeadAreaHandler, this, std::placeholders::_1));
        addCommandMapping(
            "WHEN TRUE DO EXTR_END=2.0",
            std::bind(
                &SiemensParser::ExtruderOffHandler, this, std::placeholders::_1));
    }

    void SiemensParser::BeadAreaHandler(QVector<QString> params)
    {
        //redirect - essentially M3 command
        CommonParser::M3Handler(params);
    }

    void SiemensParser::ExtruderOffHandler(QVector<QString> params)
    {
        //redirect - essentially M5 command
        CommonParser::M5Handler(params);
    }
}  // namespace ORNL
