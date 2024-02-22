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
        m_ua_parameter = QString("UA");
        m_h_parameter = QChar('H');
    }

    void IngersollParser::config()
    {
        CommonParser::config();

        addCommandMapping(
            "G0",
            std::bind(
                &IngersollParser::G0Handler, this, std::placeholders::_1));

        addCommandMapping(
            "G1",
            std::bind(
                &IngersollParser::G1Handler, this, std::placeholders::_1));
    }

    void IngersollParser::G0Handler(QVector<QStringRef> params)
    {
        QVector<QStringRef> interestedParams;
        for(QStringRef& p : params)
        {
            if(!p.startsWith(m_ua_parameter))
                interestedParams.push_back(p);
        }

        CommonParser::G0Handler(interestedParams);
    }

    void IngersollParser::G1Handler(QVector<QStringRef> params)
    {
        QVector<QStringRef> interestedParams;
        for(QStringRef& p : params)
        {
            if(p[0] != m_h_parameter && !p.startsWith(m_ua_parameter))
                interestedParams.push_back(p);
        }

        CommonParser::G1Handler(interestedParams);
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
