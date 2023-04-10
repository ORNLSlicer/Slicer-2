#if 0
#include "gcode/parsers/wolf_parser.h"

#include <QString>

namespace ORNL
{
    WolfParser::WolfParser(): CommonParser(mm, minute, lbm, degree, mm / minute, mm / s / s, rev / minute )
    {}

    void WolfParser::config()
    {
        CommonParser::config();

        //add syntax specific bindings
        addCommandMapping(
            "T1",
            std::bind(
                &WolfParser::T1Handler, this, std::placeholders::_1));
        addCommandMapping(
            "M1",
            std::bind(
                &WolfParser::M1Handler, this, std::placeholders::_1));
        addCommandMapping(
            "M2",
            std::bind(
                &WolfParser::M2Handler, this, std::placeholders::_1));
        addCommandMapping(
            "M5",
            std::bind(
                &WolfParser::M5Handler, this, std::placeholders::_1));
        addCommandMapping(
            "M6",
            std::bind(
                &WolfParser::M6Handler, this, std::placeholders::_1));
        addCommandMapping(
            "M101",
            std::bind(
                &WolfParser::M101Handler, this, std::placeholders::_1));
        addCommandMapping(
            "M103",
            std::bind(
                &WolfParser::M103Handler, this, std::placeholders::_1));
    }

    void WolfParser::M101Handler(QStringList& params)
    {
        m_extruder_ON = true;
    }

    void WolfParser::M103Handler(QStringList& params)
    {
        m_extruder_ON = false;
    }

    void WolfParser::T1Handler(QStringList& params)
    {
    }
    void WolfParser::M1Handler(QStringList& params)
    {
    }
    void WolfParser::M2Handler(QStringList& params)
    {
    }
    void WolfParser::M5Handler(QStringList& params)
    {
    }
    void WolfParser::M6Handler(QStringList& params)
    {
    }
}  // namespace ORNL
#endif
