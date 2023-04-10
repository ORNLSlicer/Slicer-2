#include "gcode/parsers/GKN_parser.h"
#include <QStringBuilder>

namespace ORNL
{
  GKNParser::GKNParser(GcodeMeta meta, bool allowLayerAlter, QStringList& lines, QStringList& upperLines)
      : CommonParser(meta, allowLayerAlter, lines, upperLines)
  {
      m_x_param = QLatin1String("X");
      m_y_param = QLatin1String(" Y");
      m_z_param = QLatin1String(" Z");
      m_f_param = QLatin1String(" F");
      m_p_param = QLatin1String("P");

      config();
  }

  void GKNParser::config()
  {
      CommonParser::config();

      //override G0, G1, and G4 of common parser
      addCommandMapping(
          "G0",
          std::bind(
              &GKNParser::G0Handler, this, std::placeholders::_1));
      addCommandMapping(
          "G1",
          std::bind(
              &GKNParser::G1Handler, this, std::placeholders::_1));
      addCommandMapping(
          "G4",
          std::bind(
              &GKNParser::G4Handler, this, std::placeholders::_1));

      //GKN syntax specific
      addCommandMapping(
          "M3",
          std::bind(
              &GKNParser::M3Handler, this, std::placeholders::_1));
      addCommandMapping(
          "M5",
          std::bind(
              &GKNParser::M5Handler, this, std::placeholders::_1));
      addCommandMapping(
          "M6",
          std::bind(
              &GKNParser::M6Handler, this, std::placeholders::_1));
      addCommandMapping(
          "M7",
          std::bind(
              &GKNParser::M7Handler, this, std::placeholders::_1));

      //multiple print speeds can be defined
      addCommandMapping(
          "PRINT_SPEED",
          std::bind(
              &GKNParser::PrintSpeedHandler, this, std::placeholders::_1));
      addCommandMapping(
          "PRINT_SPEED1",
          std::bind(
              &GKNParser::PrintSpeedHandler, this, std::placeholders::_1));
      addCommandMapping(
          "PRINT_SPEED2",
          std::bind(
              &GKNParser::PrintSpeedHandler, this, std::placeholders::_1));
      addCommandMapping(
          "PRINT_SPEED3",
          std::bind(
              &GKNParser::PrintSpeedHandler, this, std::placeholders::_1));
      addCommandMapping(
          "PRINT_SPEED4",
          std::bind(
              &GKNParser::PrintSpeedHandler, this, std::placeholders::_1));

      addCommandMapping(
          "RAPID_SPEED",
          std::bind(
              &GKNParser::RapidSpeedHandler, this, std::placeholders::_1));
      addCommandMapping(
          "SCAN_SPEED",
          std::bind(
              &GKNParser::ScanSpeedHandler, this, std::placeholders::_1));
  }

  void GKNParser::G0Handler(QVector<QStringRef> params)
  {
      QString newParameterList = m_x_param % params[0] % m_y_param % params[1] % m_z_param % params[2];

      CommonParser::G0Handler(newParameterList.splitRef(' '));
  }

  void GKNParser::G1Handler(QVector<QStringRef> params)
  {
      //sometimes speed is a hard-coded value, other times it is a variable to be filled in
      QString speed = params[8].toString();
      if(m_default_values.contains(speed))
          speed = m_default_values[speed];

      QString newParameterList = m_x_param % params[0] % m_y_param % params[1] %
              m_z_param % params[2] % m_f_param % speed;

      CommonParser::G1Handler(newParameterList.splitRef(' '));
  }

  void GKNParser::G4Handler(QVector<QStringRef> params)
  {
      if (params.size() != 1)
      {
          //throwing errors deemed too restrictive, for now, if improperly formatted, skip command
//          QString exceptionString;
//          QTextStream(&exceptionString)
//              << "G4 command should have one parameter . Error occured on "
//                 "GCode line "
//              << m_current_gcode_command.getLineNumber() << endl
//              << "."
//              << "With GCode command string: " << getCurrentCommandString();
//          throw IllegalParameterException(exceptionString);

          return;
      }

      QString newParam = m_p_param % params[0].toString();

      CommonParser::G4Handler(QVector<QStringRef> { newParam.midRef(0) });
  }

  void GKNParser::M3Handler(QVector<QStringRef> params)
  {
      //NOP
  }

  void GKNParser::M5Handler(QVector<QStringRef> params)
  {
      //NOP
  }

  void GKNParser::M6Handler(QVector<QStringRef> params)
  {
      if(params.size() != 1)
      {
          //throwing errors deemed too restrictive, for now, if improperly formatted, skip command
          return;
      }
      m_extruders_on[0] = true;
  }

  void GKNParser::M7Handler(QVector<QStringRef> params)
  {
      if (!params.empty())
      {
          //throwing errors deemed too restrictive, for now, if improperly formatted, skip command
//          QString exceptionString;
//          QTextStream(&exceptionString)
//              << "M7 command should have no parameters . Error occured on "
//                 "GCode line "
//              << m_current_gcode_command.getLineNumber() << endl
//              << "."
//              << "With GCode command string: " << getCurrentCommandString();
//          throw IllegalParameterException(exceptionString);
          return;
      }

      m_extruders_on[0] = false;
  }

  //assume we find them in order
  void GKNParser::PrintSpeedHandler(QVector<QStringRef> params)
  {
      if(m_default_values.size() == 0)
          m_default_values.insert("PRINT_SPEED", params[1].toString());
      else
          m_default_values.insert("PRINT_SPEED" % QString::number(m_default_values.size()), params[1].toString());
  }

  void GKNParser::RapidSpeedHandler(QVector<QStringRef> params)
  {
      m_default_values.insert("RAPID_SPEED", params[1].toString());
  }

  void GKNParser::ScanSpeedHandler(QVector<QStringRef> params)
  {
      m_default_values.insert("SCAN_SPEED", params[1].toString());
  }
}
