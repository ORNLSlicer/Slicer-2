#if 0
#include "gcode/parsers/gudel_parser.h"

namespace ORNL
{
    GudelParser::GudelParser(GcodeMeta meta, bool allowLayerAlter) : CommonParser(meta, allowLayerAlter)
    {
        config();
    }

    void GudelParser::config()
    {
        CommonParser::config();

        //Gudel syntax specific
        addCommandMapping(
            "M3",
            std::bind(
                &GudelParser::M3Handler, this, std::placeholders::_1));
        addCommandMapping(
            "M5",
            std::bind(
                &GudelParser::M5Handler, this, std::placeholders::_1));
    }

    void GudelParser::M3Handler(QVector<QString> params)
    {
        if(params.size() != 1)
        {
            //throwing errors deemed too restrictive, for now, if improperly formatted, skip command
            return;
        }

        m_extruder_ON = true;
    }

    void GudelParser::M5Handler(QVector<QString> params)
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
