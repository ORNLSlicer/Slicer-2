#include "gcode/parsers/RPBF_parser.h"
#include <QStringBuilder>

namespace ORNL
{
  RPBFParser::RPBFParser(GcodeMeta meta, bool allowLayerAlter, QStringList& lines, QStringList& upperLines)
      : CommonParser(meta, allowLayerAlter, lines, upperLines)
      , m_meta_copy(meta)
      , m_lines_copy(lines)
      , m_upper_lines_copy(upperLines)
  {
      m_x_param = QLatin1String("X");
      m_y_param = QLatin1String(" Y");
      m_comma = QLatin1String(", ");
      m_space = QLatin1String(" ");
      m_polyline = QLatin1String("$$POLYLINE/");
      m_hatches = QLatin1String("$$HATCHES/");

      config();

      QStringMatcher polyline("$$POLYLINE");
      QStringMatcher hatches("$$HATCHES");
      int totalExpansion = upperLines.size();
      for(QString line : upperLines)
      {
          if(polyline.indexIn(line) != -1 || hatches.indexIn(line) != -1)
          {
              QVector<QStringRef> parameters = line.splitRef(',');
              totalExpansion += parameters[1].toInt();
          }
      }

      lines.reserve(totalExpansion);
      upperLines.reserve(totalExpansion);

      setModified();
  }

  void RPBFParser::config()
  {
      CommonParser::config();

      addCommandMapping(
          "$$POLYLINE",
          std::bind(
              &RPBFParser::PolylinesHandler, this, std::placeholders::_1));
      addCommandMapping(
          "$$HATCHES",
          std::bind(
              &RPBFParser::HatchesHandler, this, std::placeholders::_1));
  }

  //assume we find them in order
  void RPBFParser::PolylinesHandler(QVector<QStringRef> params)
  {

      //line requires expansion
      if(params.size() > 9)
      {
          int currentIndex = this->getCurrentLine();
          QVector<QStringRef> prefix {params[0], params[1]};
          int j = 2;
          m_lines_copy[currentIndex] = SegmentHelper(m_polyline, Constants::RegionTypeStrings::kPerimeter, prefix,
                                                     QVector<QStringRef> { params[j], params[j + 1], params[j + 2], params[j + 3], params[j + 4], params[j + 5], params[j + 6] });
          m_upper_lines_copy[currentIndex] = m_lines_copy[currentIndex];
          ++currentIndex;
          j += 7;

          for(int end = params.size(); j < end; j += 7)
          {
              QString newLine = SegmentHelper(m_polyline, Constants::RegionTypeStrings::kPerimeter, prefix,
                                              QVector<QStringRef> { params[j], params[j + 1], params[j + 2], params[j + 3], params[j + 4], params[j + 5], params[j + 6] });
              m_lines_copy.insert(currentIndex, newLine);
              m_upper_lines_copy.insert(currentIndex, newLine);
              ++currentIndex;
          }
          this->alterCurrentEndLine(params[1].toInt() - 1);
      }

      m_extruders_on[0] = true;
      QString newParameterList = m_x_param % params[4] % m_y_param % params[5];
      QString optionalParameterList = m_x_param % params[2] % m_y_param % params[3];
      CommonParser::G1HandlerHelper(newParameterList.splitRef(' '), optionalParameterList.splitRef(' '));
  }

  void RPBFParser::HatchesHandler(QVector<QStringRef> params)
  {
      //line requires expansion
      if(params.size() > 9)
      {
          int currentIndex = this->getCurrentLine();
          QVector<QStringRef> prefix {params[0], params[1]};
          int j = 2;
          m_lines_copy[currentIndex] = SegmentHelper(m_hatches, Constants::RegionTypeStrings::kInfill, prefix,
                                                     QVector<QStringRef> { params[j], params[j + 1], params[j + 2], params[j + 3], params[j + 4], params[j + 5], params[j + 6] });
          m_upper_lines_copy[currentIndex] = m_lines_copy[currentIndex];
          ++currentIndex;
          j += 7;

          for(int end = params.size(); j < end; j += 7)
          {
              QString newLine = SegmentHelper(m_hatches, Constants::RegionTypeStrings::kInfill, prefix,
                                              QVector<QStringRef> { params[j], params[j + 1], params[j + 2], params[j + 3], params[j + 4], params[j + 5], params[j + 6] });
              m_lines_copy.insert(currentIndex, newLine);
              m_upper_lines_copy.insert(currentIndex, newLine);
              ++currentIndex;
          }
          this->alterCurrentEndLine(params[1].toInt() - 1);
      }

      m_extruders_on[0] = true;
      QString newParameterList = m_x_param % params[4] % m_y_param % params[5];
      QString optionalParameterList = m_x_param % params[2] % m_y_param % params[3];
      CommonParser::G1HandlerHelper(newParameterList.splitRef(' '), optionalParameterList.splitRef(' '));
  }

  QString RPBFParser::SegmentHelper(QString command, QString type, QVector<QStringRef> prefix, QVector<QStringRef> params)
  {
     return command % prefix[0] % m_comma % prefix[1] % m_comma % params[0] % m_comma % params[1] % m_comma % params[2]
        % m_comma % params[3] % m_comma % params[4] % m_comma % params[5] % m_comma % params[6] % m_space %
        m_meta_copy.m_comment_starting_delimiter % type % m_meta_copy.m_comment_ending_delimiter;
  }
}
