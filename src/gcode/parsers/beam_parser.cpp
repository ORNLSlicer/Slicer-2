#include "gcode/parsers/beam_parser.h"

namespace ORNL
{
    BeamParser::BeamParser(GcodeMeta meta, bool allowLayerAlter, QStringList& lines, QStringList& upperLines)
        : CommonParser(meta, allowLayerAlter, lines, upperLines)
    {
        config();
    }

    void BeamParser::config()
    {
        CommonParser::config();

        //BEAM syntax specific
        addCommandMapping(
            "M110",
            std::bind(
                &BeamParser::M110Handler, this, std::placeholders::_1));
        addCommandMapping(
            "M111",
            std::bind(
                &BeamParser::M111Handler, this, std::placeholders::_1));

    }

    void BeamParser::M110Handler(QVector<QString> params)
    {
        if (!params.empty())
        {
            //throwing errors deemed too restrictive, for now, if improperly formatted, skip command
//            QString exceptionString;
//            QTextStream(&exceptionString)
//                << "M110 command should have no parameters . Error occured on "
//                   "GCode line "
//                << m_current_gcode_command.getLineNumber() << Qt::endl
//                << "."
//                << "With GCode command string: " << getCurrentCommandString();
//            throw IllegalParameterException(exceptionString);

            return;
        }

        m_extruders_on[0] = true;
    }

    void BeamParser::M111Handler(QVector<QString> params)
    {
        if (!params.empty())
        {
            //throwing errors deemed too restrictive, for now, if improperly formatted, skip command
//            QString exceptionString;
//            QTextStream(&exceptionString)
//                << "M111 command should have no parameters . Error occured on "
//                   "GCode line "
//                << m_current_gcode_command.getLineNumber() << Qt::endl
//                << "."
//                << "With GCode command string: " << getCurrentCommandString();
//            throw IllegalParameterException(exceptionString);

            return;
        }

        m_extruders_on[0] = false;
    }

}
