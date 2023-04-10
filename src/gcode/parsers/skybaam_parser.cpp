#if 0
#include "gcode/parsers/skybaam_parser.h"

#include <QString>
#include <QStringList>
#include <QVector>
#include <QTextStream>

#include "exceptions/exceptions.h"
#include "units/unit.h"

namespace ORNL
{
    SkyBaamParser::SkyBaamParser() : CommonParser(in, minute, lbm, degree, in / minute, in / s / s, rev / minute )
    {}

    void SkyBaamParser::config()
    {
        CommonParser::config();

        addCommandMapping(
            "M3",
            std::bind(
                &SkyBaamParser::M3Handler, this, std::placeholders::_1));
        addCommandMapping(
            "M5",
            std::bind(
                &SkyBaamParser::M5Handler, this, std::placeholders::_1));
        addCommandMapping(
            "M30",
            std::bind(
                &SkyBaamParser::M30Handler, this, std::placeholders::_1));
        addCommandMapping(
            "M68",
            std::bind(
                &SkyBaamParser::M68Handler, this, std::placeholders::_1));
        addCommandMapping("T",
                          std::bind(&SkyBaamParser::ToolChangeHandler,
                                    this,
                                    std::placeholders::_1));

        setLineCommentString("");
        setBlockCommentDelimiters("(", ")");
        //setDistanceConversion(inch);
    }

    void SkyBaamParser::M3Handler(QStringList& params)
    {
        char current_parameter;
        int current_value;
        bool no_error, s_not_used = true;

        if (params.empty())
        {
            QString exceptionString;
            QTextStream(&exceptionString)
                << "No parameters for command M3, on line number "
                << m_current_gcode_command.getLineNumber()
                << ". Need at least one for this command." << endl
                << "GCode command string: " << getCurrentCommandString();
            throw IllegalParameterException(exceptionString);
        }


        for (QStringList::Iterator i = params.begin(); i != params.end(); i++)
        {
            // Retriving the first character in the QString and making it a char
            current_parameter = (*i).at(0).toLatin1();
            current_value     = (*i).right((*i).size() - 1).toFloat(&no_error);
            if (!no_error)
            {
                throwIntegerConversionErrorException();
            }

            m_current_gcode_command.addParameter(current_parameter,
                                                 current_value);

            switch (current_parameter)
            {
                case ('S'):
                case ('s'):
                    if (s_not_used)
                    {
                        // TODO: Set this to be in RPM
                        setSpindleSpeed(current_value);
                        s_not_used = false;
                    }
                    else
                    {
                        throwMultipleParameterException(current_parameter);
                    }
                    break;
                default:
                    QString exceptionString;
                    QTextStream(&exceptionString)
                        << "Error: Unknown parameter " << *i
                        << " on GCode line "
                        << m_current_gcode_command.getLineNumber()
                        << ", for GCode command M3" << endl
                        << "With GCode command string: "
                        << getCurrentCommandString();
                    throw IllegalParameterException(exceptionString);
                    break;
            }
        }

        if (s_not_used)
        {
            QString exceptionString;
            QTextStream(&exceptionString)
                << "Error not all required parameters passed for GCode command "
                   "on line  "
                << m_current_gcode_command.getLineNumber() << endl
                << "With GCode command string: " << getCurrentCommandString();
            throw IllegalParameterException(exceptionString);
        }

        m_extruder_ON = true;
    }

    void SkyBaamParser::M5Handler(QStringList& params)
    {
        if (!params.empty())
        {
            QString exceptionString;
            QTextStream(&exceptionString)
                << "M5 command should have no parameters . Error occured on "
                   "GCode line "
                << m_current_gcode_command.getLineNumber() << endl
                << "."
                << "With GCode command string: " << getCurrentCommandString();
            throw IllegalParameterException(exceptionString);
        }

        m_extruder_ON = false;
    }

    void SkyBaamParser::M30Handler(QStringList& params)
    {
        if (!params.empty())
        {
            QString exceptionString;
            QTextStream(&exceptionString)
                << "M30 command should have no parameters . Error occured on "
                   "GCode line "
                << m_current_gcode_command.getLineNumber() << endl
                << "."
                << "With GCode command string: " << getCurrentCommandString();
            throw IllegalParameterException(exceptionString);
        }
    }

   void SkyBaamParser::M68Handler(QStringList& params)
    {
        if (!params.empty())
        {
            QString exceptionString;
            QTextStream(&exceptionString)
                << "M68 command should have no parameters . Error occured on "
                   "GCode line "
                << m_current_gcode_command.getLineNumber() << endl
                << "."
                << "With GCode command string: " << getCurrentCommandString();
            throw IllegalParameterException(exceptionString);
        }

        m_park = true;
    }

    void SkyBaamParser::ToolChangeHandler(QStringList& params)
    {
        if (!params.empty())
        {
            QString exceptionString;
            QTextStream(&exceptionString)
                << "Tool change command should have no parameters . Error "
                   "occured on GCode line "
                << m_current_gcode_command.getLineNumber() << endl
                << "With GCode command string: " << getCurrentCommandString();
            throw IllegalParameterException(exceptionString);
        }
    }
}  // namespace ORNL
#endif
