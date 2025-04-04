#if 0
#include "gcode/parsers/dmg_dmu_parser.h"

namespace ORNL
{
    DmgDmuParser::DmgDmuParser(GcodeMeta meta, bool allowLayerAlter) : CommonParser(meta, allowLayerAlter)
    {
        config();
    }

    void DmgDmuParser::config()
    {
        CommonParser::config();

        //DMG-DMU syntax specific
        addCommandMapping(
            "M3",
            std::bind(
                &DmgDmuParser::M3Handler, this, std::placeholders::_1));
        addCommandMapping(
            "M5",
            std::bind(
                &DmgDmuParser::M5Handler, this, std::placeholders::_1));
    }

    void DmgDmuParser::M3Handler(QVector<QString> params)
    {
        if(params.size() != 1)
        {
            //throwing errors deemed too restrictive, for now, if improperly formatted, skip command
//            QString exceptionString;
//            QTextStream(&exceptionString)
//                << "M5 command should have no parameters . Error occured on "
//                   "GCode line "
//                << m_current_gcode_command.getLineNumber() << Qt::endl
//                << "."
//                << "With GCode command string: " << getCurrentCommandString();
//            throw IllegalParameterException(exceptionString);

            return;
        }
        m_extruder_ON = true;
    }

    void DmgDmuParser::M5Handler(QVector<QString> params)
    {
        if (!params.empty())
        {
            //throwing errors deemed too restrictive, for now, if improperly formatted, skip command
//            QString exceptionString;
//            QTextStream(&exceptionString)
//                << "M5 command should have no parameters . Error occured on "
//                   "GCode line "
//                << m_current_gcode_command.getLineNumber() << Qt::endl
//                << "."
//                << "With GCode command string: " << getCurrentCommandString();
//            throw IllegalParameterException(exceptionString);

            return;
        }

        m_extruder_ON = false;
    }
}
#endif
