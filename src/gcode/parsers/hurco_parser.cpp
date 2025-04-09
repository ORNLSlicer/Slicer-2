#if 0
#include "gcode/parsers/hurco_parser.h"

namespace ORNL
{
    HurcoParser::HurcoParser(GcodeMeta meta, bool allowLayerAlter) : CommonParser(meta, allowLayerAlter)
    {
        config();
    }

    void HurcoParser::config()
    {
        CommonParser::config();

        //Haas syntax specific
        addCommandMapping(
            "M3",
            std::bind(
                &HurcoParser::M3Handler, this, std::placeholders::_1));
        addCommandMapping(
            "M5",
            std::bind(
                &HurcoParser::M5Handler, this, std::placeholders::_1));
    } 

    void HurcoParser::M3Handler(QVector<QString> params)
    {
        if(params.size() != 1)
        {
            //throwing errors deemed too restrictive, for now, if improperly formatted, skip command
            return;
        }

        m_extruder_ON = true;
    }

    void HurcoParser::M5Handler(QVector<QString> params)
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
