#if 0
#include "gcode/parsers/romi_fanuc_parser.h"

namespace ORNL
{
    RomiFanucParser::RomiFanucParser(GcodeMeta meta, bool allowLayerAlter) : CommonParser(meta, allowLayerAlter)
    {
        config();
    }

    void RomiFanucParser::config()
    {
        CommonParser::config();

        //Romi_fanuc syntax specific
        addCommandMapping(
            "M3",
            std::bind(
                &RomiFanucParser::M3Handler, this, std::placeholders::_1));
        addCommandMapping(
            "M5",
            std::bind(
                &RomiFanucParser::M5Handler, this, std::placeholders::_1));
    }

    void RomiFanucParser::M3Handler(QVector<QStringRef> params)
    {
        if(params.size() != 1)
        {
            //throwing errors deemed too restrictive, for now, if improperly formatted, skip command
            return;
        }

        m_extruder_ON = true;
    }

    void RomiFanucParser::M5Handler(QVector<QStringRef> params)
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
