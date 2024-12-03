#include "gcode/parsers/sheet_lamination_parser.h"
#include <QStringBuilder>

#include "QString"
#include "QStringList"
#include "string"
#include "utilities/enums.h"
#include "QColor"

#include "managers/settings/settings_manager.h"

namespace ORNL
{
    SheetLaminationParser::SheetLaminationParser()
    {

    }

    QVector<QVector<QSharedPointer<SegmentBase>>> SheetLaminationParser::parse(QStringList& lines)
    {
        QSharedPointer<SettingsBase> global_sb = QSharedPointer<SettingsBase>::create(*GSM->getGlobal());

        GcodeMeta meta = GcodeMetaList::SheetLaminationMeta;
        QVector<QVector<QSharedPointer<SegmentBase>>> rv;
        QVector<QSharedPointer<SegmentBase>> newLayer;
        int curr_layer = 0;

        rv.push_back(newLayer); // blank 0th layer
        rv.push_back(newLayer); // 1st layer

        int curr_line = 0;
        while (curr_line < lines.length())
        {
            QString line = lines[curr_line];
            if (line == "LINE")
            {
                if (lines[curr_line+8].toFloat() > curr_layer)
                { // make a new layer with every increase in Z (30, 31)
                    curr_layer++; // 30
                    rv.push_back(newLayer);
                }
                // if we find a LINE, we now know where all the information for the segment is in relation to that line (like 20, 30, 11, etc...)
                // Also, I'm fully aware that Clipper and therefore the Point object throws out 0 values in its constructor
                // I want them to be read anyway
                float x_one = lines[curr_line+4].toFloat() * Constants::OpenGL::kObjectToView * meta.m_distance_unit();
                float y_one = lines[curr_line+6].toFloat() * Constants::OpenGL::kObjectToView * meta.m_distance_unit();
                float z_one = lines[curr_line+8].toFloat() * Constants::OpenGL::kObjectToView * meta.m_distance_unit();
                float x_two = lines[curr_line+10].toFloat() * Constants::OpenGL::kObjectToView * meta.m_distance_unit();
                float y_two = lines[curr_line+12].toFloat() * Constants::OpenGL::kObjectToView * meta.m_distance_unit();
                float z_two = lines[curr_line+14].toFloat() * Constants::OpenGL::kObjectToView * meta.m_distance_unit();
                Point point_one = Point(x_one, y_one, z_one);
                Point point_two = Point(x_two, y_two, z_two);
                float segment_length = point_one.distance(point_two)();
                QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(LineSegment(point_one, point_two - point_one));
                float segment_height = global_sb->setting<Distance>(Constants::ProfileSettings::Layer::kLayerHeight)();
                if (lines[curr_line+2] == "POLYGON")
                {
                    segment->setGCodeInfo(0.1, segment_length, segment_height, SegmentDisplayType::kLine, QColor(118,0,255,255), (uint) curr_line, (uint) curr_layer);
                } else
                {
                    segment->setGCodeInfo(0.1, segment_length, segment_height, SegmentDisplayType::kTravel, QColor(127,127,127,255), (uint) curr_line, (uint) curr_layer);
                }
                rv.last().push_back(segment);
                //i += however many dxf lines it takes to represent a segment
                curr_line+=15;
            }
            curr_line++;
        }
        return rv;
    }

    QString SheetLaminationParser::getStats()
    {
        return "Done!";
    }


}
