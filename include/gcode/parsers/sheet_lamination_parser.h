#ifndef SHEET_LAMINATION_PARSER_H
#define SHEET_LAMINATION_PARSER_H

#include <QSharedPointer>
#include <QVector>

#include "gcode/gcode_command.h"
#include "gcode/gcode_meta.h"
#include "geometry/point.h"
#include "geometry/segments/line.h"

namespace ORNL
{
    class SheetLaminationParser : public QObject
    {
        public:
            SheetLaminationParser();

            // ParserBase interface
            // Unlike every other parser, this parser not only doesn't implement the Common Parser
            // In fact, it doesn't even implement Parser Base
            // Eventually, Parser Base will likely need to be rewritten so that SheetLaminationParser can inherit it without conflict

            //! \brief Takes in a QStringList (a listified version of the DXF file) and parses it into the final 2D vector of displayable segment bases
            //! \param lines: original DXF file
            //! \return list of layers which themselves are list of lines
            QVector<QVector<QSharedPointer<SegmentBase>>> parse(QStringList& lines);

            QString getStats();

    };
}
#endif // SHEET_LAMINATION_PARSER_H
