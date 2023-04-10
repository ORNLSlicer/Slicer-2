#ifndef SAVEPOLYGONSTOFILE_H
#define SAVEPOLYGONSTOFILE_H

#include <QString>
#include <QTextStream>

#include "../../geometry/polygon_list.h"
#include "circlepack.h"

namespace ORNL
{
    //! \brief save polygonlist to a file including boundary and interior polygon
    namespace SavePolygons
    {
        void savePolygonsToFile(Polygon* polygon, QString file_path);
        void savePolygonsToFile(PolygonList* polygons, QString file_path);
        Polygon readBoundaryPolygonFile(QString file_path);
    }
}

#endif // SAVEPOLYGONSTOFILE_H


