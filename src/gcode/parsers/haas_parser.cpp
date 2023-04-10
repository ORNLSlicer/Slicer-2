#if 0
#include "gcode/parsers/haas_parser.h"

namespace ORNL
{
    HaasParser::HaasParser(GcodeMeta meta, bool allowLayerAlter) : CommonParser(meta, allowLayerAlter)
    {
        config();
    }

    void HaasParser::config()
    {
        CommonParser::config();

        //Haas syntax specific
        addCommandMapping(
            "M3",
            std::bind(
                &HaasParser::M3Handler, this, std::placeholders::_1));
        addCommandMapping(
            "M5",
            std::bind(
                &HaasParser::M5Handler, this, std::placeholders::_1));
    }

    void HaasParser::M3Handler(QVector<QStringRef> params)
    {
        if(params.size() != 1)
        {
            //throwing errors deemed too restrictive, for now, if improperly formatted, skip command
            return;
        }

        m_extruder_ON = true;
    }

    void HaasParser::M5Handler(QVector<QStringRef> params)
    {
        if (!params.empty())
        {
            //throwing errors deemed too restrictive, for now, if improperly formatted, skip command
            return;
        }

        m_extruder_ON = false;
    }
}
#endif
