#include <QStringBuilder>

#include "gcode/parsers/mazak_parser.h"

namespace ORNL
{
    MazakParser::MazakParser(GcodeMeta meta, bool allowLayerAlter, QStringList& lines, QStringList& upperLines)
        : CommonParser(meta, allowLayerAlter, lines, upperLines)
    {
        m_f_parameter = QChar('F');
        m_feedrate_reference = "#981";
        config();
    }

    void MazakParser::config()
    {
        CommonParser::config();

        //Mazak syntax specific
        addCommandMapping(
            "G1",
            std::bind(
                &MazakParser::G1Handler, this, std::placeholders::_1));
        addCommandMapping(
            "#981",
            std::bind(
                &MazakParser::FeedRateHandler, this, std::placeholders::_1));
        addCommandMapping(
            "G441",
            std::bind(
                &MazakParser::G441Handler, this, std::placeholders::_1));
        addCommandMapping(
            "G442",
            std::bind(
                &MazakParser::G442Handler, this, std::placeholders::_1));
    }

    void MazakParser::G1Handler(QVector<QStringRef> params)
    {
        for(QStringRef& p : params)
        {
            if(p[0] == m_f_parameter)
            {
                QStringRef val = p.mid(1).trimmed();
                if(val == m_feedrate_reference)
                {
                    p = m_feedrate.midRef(0);
                    break;
                }
            }
        }

        CommonParser::G1Handler(params);
    }

    //G441 (Turn Laser ON)
    void MazakParser::G441Handler(QVector<QStringRef> params)
    {
        if (!params.empty())
        {
            //throwing errors deemed too restrictive, for now, if improperly formatted, skip command
//            QString exceptionString;
//            QTextStream(&exceptionString)
//                << "G441 command should have no parameters . Error occured on "
//                   "GCode line "
//                << m_current_gcode_command.getLineNumber() << endl
//                << "."
//                << "With GCode command string: " << getCurrentCommandString();
//            throw IllegalParameterException(exceptionString);

            return;
        }

        m_extruders_on[0] = true;
    }

    //G442 (LASER OFF)
    void MazakParser::G442Handler(QVector<QStringRef> params)
    {
        if (!params.empty())
        {
            //throwing errors deemed too restrictive, for now, if improperly formatted, skip command
//            QString exceptionString;
//            QTextStream(&exceptionString)
//                << "G442 command should have no parameters . Error occured on "
//                   "GCode line "
//                << m_current_gcode_command.getLineNumber() << endl
//                << "."
//                << "With GCode command string: " << getCurrentCommandString();
//            throw IllegalParameterException(exceptionString);

            return;
        }

        m_extruders_on[0] = false;
    }

    void MazakParser::FeedRateHandler(QVector<QStringRef> params)
    {
        m_feedrate = "F" % params[1].toString();
    }
}
