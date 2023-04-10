#if 0
#include "gcode/gcode_parser.h"

#include <QDebug>
#include <QIODevice>
#include <QVariant>
#include <QVector>
#include <QtGlobal>
#include <functional>

#include "exceptions/exceptions.h"
#include "gcode/gcode_command.h"
//#include "gcode/parsers/beam_parser.h"
//#include "gcode/parsers/blue_gantry_parser.h"
#include "gcode/parsers/cincinnati_parser.h"
#include "gcode/parsers/common_parser.h"
//#include "gcode/parsers/dmg_dmu_parser.h"
//#include "gcode/parsers/ingersoll_parser.h"
//#include "gcode/parsers/gkn_parser.h"
//#include "gcode/parsers/gudel_parser.h"
//#include "gcode/parsers/haas_inch_parser.h"
//#include "gcode/parsers/haas_metric_parser.h"
//#include "gcode/parsers/hurco_parser.h"
//#include "gcode/parsers/marlin_parser.h"
//#include "gcode/parsers/mazak_parser.h"
//#include "gcode/parsers/mvp_parser.h"
//#include "gcode/parsers/romi_fanuc_parser.h"
//#include "gcode/parsers/siemens_parser.h"
//#include "gcode/parsers/skybaam_parser.h"
//#include "gcode/parsers/wolf_parser.h"
//#include "utilities/constants.h"
//#include <windows/main_window.h>
//#include "gcode/gcode_vitalstat.h"
#include "gcode/gcode_meta.h"

namespace ORNL
{
    GcodeParser::GcodeParser()
        : m_current_parser(
              nullptr,
              std::bind(&GcodeParser::freeParser, this, std::placeholders::_1))
    {
        selectParser(GcodeSyntax::kCommon);
    }

    GcodeParser::GcodeParser(GcodeSyntax id)
        : m_current_parser(
              nullptr,
              std::bind(&GcodeParser::freeParser, this, std::placeholders::_1))
    {
        selectParser(id);
    }

    GcodeCommand GcodeParser::parseLine(const QString& line)
    {
        if(m_current_parser == nullptr)
        {
            throwParserNotSetException();
        }
        return m_current_parser->parseCommand(line);
    }

//    GcodeCommand GcodeParser::parseLine(const QString& line, int line_number)
//    {
//        if(m_current_parser == nullptr)
//        {
//            throwParserNotSetException();
//        }

//        return m_current_parser->parseCommand(line, line_number);
//    }

    QPair< GcodeCommand, GcodeCommand > GcodeParser::updateLine(
        const QString& line,
        int line_number,
        const QString& previous_line,
        const QString& next_line)
    {
        GcodeCommand l, r;
        if(m_current_parser == nullptr)
        {
            throwParserNotSetException();
        }

        // Sets the internal state of the parser.
        m_current_parser->parseCommand(previous_line);
        l = m_current_parser->parseCommand(line, line_number);
        r = m_current_parser->parseCommand(next_line, line_number + 1);

        return QPair< GcodeCommand, GcodeCommand >(l, r);
    }

//    QVector< GcodeCommand > GcodeParser::parse(QStringList::iterator begin,
//                                               QStringList::iterator end,
//                                               quint64 line_number)
//    {
//        if(m_current_parser == nullptr)
//        {
//            throwParserNotSetException();
//        }

//        QVector< GcodeCommand > command_list;
//        for(; begin != end; begin++, line_number++)
//        {
//            command_list.push_back(
//                m_current_parser->parseCommand(*begin, line_number));
//        }

//        return command_list;
//    }

    void GcodeParser::selectParser(GcodeSyntax parserID)
    {
        if(m_current_parser_id == parserID)
        {
            return;
        }

        switch(parserID)
        {            
//            case GcodeSyntax::kBeam:
//                m_current_parser.reset(new BeamParser());
//                break;
//            case GcodeSyntax::kCommon:
//                m_current_parser.reset(new CommonParser());
//                break;
//            case GcodeSyntax::kCincinnati:
//                m_current_parser.reset(new CincinnatiParser(GcodeMetaList::CincinnatiMeta));
//                break;
//            case GcodeSyntax::kDmgDmu:
//                m_current_parser.reset(new DmgDmuParser());
//                break;
//            case GcodeSyntax::kGKN:
//                m_current_parser.reset(new GKNParser());
//                break;
//            case GcodeSyntax::kGudel:
//                m_current_parser.reset(new GudelParser());
//                break;
//            case GcodeSyntax::kHaasInch:
//                m_current_parser.reset(new HaasInchParser());
//                break;
//            case GcodeSyntax::kHaasMetric:
//                m_current_parser.reset(new HaasMetricParser());
//                break;
//            case GcodeSyntax::kHurco:
//                m_current_parser.reset(new HurcoParser());
//                break;
//            case GcodeSyntax::kIngersoll:
//                m_current_parser.reset(new IngersollParser());
//                break;
//            case GcodeSyntax::kMarlin:
//                m_current_parser.reset(new MarlinParser());
//                break;
//            case GcodeSyntax::kMazak:
//                m_current_parser.reset(new MazakParser());
//                break;
//            case GcodeSyntax::kMVP:
//                m_current_parser.reset(new MVPParser());
//                break;
//            case GcodeSyntax::kRomiFanuc:
//                m_current_parser.reset(new RomiFanucParser());
//                break;
//            case GcodeSyntax::kSiemens:
//                m_current_parser.reset(new SiemensParser());
//                break;
//            case GcodeSyntax::kSkyBaam:
//                m_current_parser.reset(new SkyBaamParser());
//                break;
//            case GcodeSyntax::kWolf:
//                m_current_parser.reset(new WolfParser());
//                break;

            default:
                throwParserNotSetException();
                break;
        }

       // selectMachine(parserID);
        m_current_parser->config();
    }

    void GcodeParser::freeParser(CommonParser *parser)
    {
        if(parser != nullptr)
        {
            delete parser;
        }

        parser = nullptr;
    }

    void GcodeParser::throwParserNotSetException()
    {
        QString exceptionString;
        QTextStream(&exceptionString)
            << "No parser selected to parse the GCode.";
        throw ParserNotSetException(exceptionString);
    }

//    void GcodeParser::selectMachine(GcodeSyntax id)
//    {
//        // TODO: This, once the strings are preset and are decided on.
//    }

    QString GcodeParser::getBlockCommentOpeningDelimiter()
    {
        return m_current_parser->getBlockCommentOpeningDelimiter();
    }

    Distance GcodeParser::getDistanceUnit()
    {
        return m_current_parser->getDistanceUnit();
    }
}  // namespace ORNL
#endif
