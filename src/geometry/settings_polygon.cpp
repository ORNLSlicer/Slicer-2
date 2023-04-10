#include "geometry/settings_polygon.h"

// Locals
#include <cross_section/cross_section.h>
#include "utilities/mathutils.h"

namespace ORNL
{
    SettingsPolygon::SettingsPolygon(Polygon geometry, QSharedPointer<SettingsBase> &sb)
    {
        m_sb = sb;

        // Set geometry from cs.
        if(!geometry.isEmpty()) {
            for(auto& point : geometry) {
                this->append(point);
            }
        }
    }


    QVector<Point> SettingsPolygon::clipLine(Point start, Point end)
    {
        ClipperLib2::PolyTree poly_tree;
        ClipperLib2::Paths paths;

        ClipperLib2::Path line;
        line.push_back(start.toIntPoint());
        line.push_back(end.toIntPoint());

        ClipperLib2::Clipper clipper;
        clipper.AddPath(line, ClipperLib2::ptSubject, false);
        clipper.AddPath(operator()(), ClipperLib2::ptClip, true);

        clipper.Execute(ClipperLib2::ctIntersection,
                        poly_tree,
                        ClipperLib2::pftNonZero,
                        ClipperLib2::pftNonZero);
        ClipperLib2::OpenPathsFromPolyTree(poly_tree, paths);

        //qDebug() << "Start:" << start.toQVector3D() << "End:" << end.toQVector3D();

        QVector<Point> rv;
        for (ClipperLib2::Path path : paths)
        {
            for(auto int_point : path)
            {
                Point p(int_point);

                //qDebug() << "Attempting to add point:" << p.toQVector2D();

                if(p != start && p != end)
                {
                    p.setSettings(m_sb);
                    rv.push_back(p);
                }
            }
        }

        /*
        qDebug() << "Clipper lib found the following intersections:";
        for (auto& p : rv) {
            qDebug() << p.toQVector2D();
        }
        */

        // Check to make sure the line does not intersect at a polygon's vertex
        /*
        for(auto& point : (*this))
        {
            // Add point if it is on the line
            if(MathUtils::onSegment(start, point, end))
                if(!rv.contains(point)) // Only add the point if it is not already accounted for
                {
                    point.setSettings(m_sb);
                    rv.push_back(point);
                }
        }
        */

        // Check to make sure start or end of the segment does not lay on the polygon's boarder
        /*
        for(int i = 0, end_cond = this->size(); i < end_cond; ++i)
        {
            Point poly_seg_start = this->at(i);
            Point poly_seg_end = this->at((i + 1) % this->size());

            if(MathUtils::onSegment(poly_seg_start, start, poly_seg_end))
                if(!rv.contains(start))
                {
                    qDebug() << "Adding start due to lying on border.";
                    start.setSettings(m_sb);
                    rv.push_back(start);
                }

            if(MathUtils::onSegment(poly_seg_start, end, poly_seg_end))
                if(!rv.contains(end))
                {
                    qDebug() << "Adding end due to lying on border.";
                    end.setSettings(m_sb);
                    rv.push_back(end);
                }
        }
        */

        return rv;
    }

    QSharedPointer<SettingsBase> SettingsPolygon::getSettings()
    {
        return m_sb;
    }
}
