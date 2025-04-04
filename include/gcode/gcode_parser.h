#if 0
#ifndef GCODEPARSER_H
#define GCODEPARSER_H
#    include <QSharedPointer>
#    include <QString>
#    include <QVector>
#    include <functional>
#    include <memory>

#    include "gcode/gcode_command.h"
#    include "utilities/enums.h"

namespace ORNL
{
    class CommonParser;

    class GcodeParser : private QObject
    {
    public:
        //! \brief Creates a GcodeParser with no specified List to act upon.
        GcodeParser();

        //! \brief Creates a GcodeParser associated with the specified line of
        //! gcode and parser.
        //!
        //! \param id Parser to use for parsing the Gcode file.
        GcodeParser(GcodeSyntax id);

        //! \brief Destroys the GcodeParser
        ~GcodeParser() = default;

        //! \todo Need to link all the repeated exceptions/params/returns with
        //! copydoc so there isnt a massive amount of
        //!       repitition and cumbersome changes if changed/refactored.

        //! \brief Parses a single line of input. The line number is assumed to
        //! be 1.
        //!
        //! \param line Gcode string to be parsed.
        //!
        //! \return GcodeCommand split into separate pieces.
        //!
        //!
        //! \throws MultipleParameterException Thrown if multiple of the same
        //! parameters exist in the same Gcode command.
        //! \throws IllegalParameterException Thrown if on of the parameters i
        //! snot part of the Gcode command.
        //! \throws IllegalArgumentException
        //! Thrown if one of the Gcode commands is not part of the selected
        //! parser.
        //! \throws ParserNotSetException Thrown if no parser is set
        //! then this exception is thrown.
        GcodeCommand parseLine(const QString& line);

        //! \brief Parses a single line of input.
        //!
        //! \param line Gcode string to be parsed.
        //! \param line_number Line number that the gcodecommand will be set to.
        //!
        //! \return GcodeCommand split into separate pieces.
        //!
        //!
        //! \throws MultipleParameterException Thrown if multiple of the same
        //! parameters exist in the same Gcode command.
        //! \throws IllegalParameterException Thrown if on of the parameters i
        //! snot part of the Gcode command.
        //! \throws IllegalArgumentException
        //! Thrown if one of the Gcode commands is not part of the selected
        //! parser.
        //! \throws ParserNotSetException Thrown if no parser is set
        //! then this exception is thrown.
        //GcodeCommand parseLine(const QString& line, int line_number);

        //! \brief Updates a command in provided.
        //!
        //! \param line Gcode string to be parsed.
        //! \param line_number Line number that the gcodecommand will be set to.
        //! \param previous_command Previous GcodeCommand to base this command
        //! off of. \note This sets the state of the parser since Gcode is a
        //! stateful language.
        //! \param next_command Next command to update since
        //! the value of the previous command was changed.
        //!
        //! \return GcodeCommand split into separate pieces.
        //!
        //!
        //! \throws MultipleParameterException Thrown if multiple of the same
        //! parameters exist in the same Gcode command.
        //! \throws IllegalParameterException Thrown if on of the parameters i
        //! snot part of the Gcode command.
        //! \throws IllegalArgumentException
        //! Thrown if one of the Gcode commands is not part of the selected
        //! parser.
        //! \throws ParserNotSetException Thrown if no parser is set
        //! then this exception is thrown.
        QPair< GcodeCommand, GcodeCommand > updateLine(
            const QString& line,
            int line_number,
            const QString& previous_command,
            const QString& next_command = QString());

        //! \brief Parses a list of Gcode strings over the specified range.
        //!
        //! \param begin Beginning iteratior of the range.
        //! \param end Ending iterator of the range to be parsed but not
        //! including.
        //! \param line_number Line number where the range begins.
        //!
        //! \return A list of GcodeCommands split into separate pieces.
        //!
        //!
        //! \throws MultipleParameterException Thrown if multiple of the same
        //! parameters exist in the same Gcode command.
        //! \throws IllegalParameterException Thrown if on of the parameters i
        //! snot part of the Gcode command.
        //! \throws IllegalArgumentException
        //! Thrown if one of the Gcode commands is not part of the selected
        //! parser.
        //! \throws ParserNotSetException Thrown if no parser is set
        //! then this exception is thrown.
//        QVector< GcodeCommand > parse(QStringList::iterator begin,
//                                      QStringList::iterator end,
//                                      quint64 line_number = 1);

        //! \brief Selects the parser to use.
        //!
        //! \param parserID GcodeSyntax enum to choose which parser to use.
        void selectParser(GcodeSyntax parserID);

        //bool parserSet() const;

        QString getBlockCommentOpeningDelimiter();

        Distance getDistanceUnit();

    private:
        //! \brief Helper function that frees the specified parser
        void freeParser(CommonParser *parser);

        //! \brief Helper function to throw a ParserNotSetException.
        //!
        //! \throws ParserNotSet Occurs when the function is called.
        void throwParserNotSetException();

        //! \brief Helper function that translates the machine name to a parser.
        //void selectMachine(GcodeSyntax id);

        std::unique_ptr< CommonParser, std::function< void(CommonParser *) > >
            m_current_parser;

        GcodeSyntax m_current_parser_id;

        //QString m_current_machine;

        //QVector<int> m_commandCounts;
        QList<QString> m_input;
        QList<QString> m_upper_input;
        int m_current_line;
        int m_current_end_line;
    };
}  // namespace ORNL

#endif  // GCODEPARSER_H
#endif
