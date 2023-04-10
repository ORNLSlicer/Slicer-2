#include "gcode/parsers/ingersoll_parser.h"

#include <QString>
#include <QStringBuilder>

#include "exceptions/exceptions.h"
#include "units/unit.h"

namespace ORNL
{
    IngersollParser::IngersollParser(GcodeMeta meta, bool allowLayerAlter, QStringList& lines, QStringList& upperLines)
        : CommonParser(meta, allowLayerAlter, lines, upperLines)
        , m_meta_copy(meta)
        , m_lines_copy(lines)
        , m_upper_lines_copy(upperLines)
    {
        m_insertions = 0;
    }

    void IngersollParser::config()
    {
        CommonParser::config();
    }

    void IngersollParser::AdjustFeedrate(double modifier)
    {
        int insertIndex = this->getCurrentLine() - 1 + m_insertions;

        QString modifierText = "OVR_LAYER_TIME=" % QString::number(modifier, 'f', 4) % " " % getCommentStartDelimiter()
                % "MODIFY FEEDRATE" % getCommentEndDelimiter();
        m_lines_copy.insert(insertIndex, modifierText);
        ++m_insertions;
    }

}  // namespace ORNL
