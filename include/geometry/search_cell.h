#ifndef SEARCH_CELL_H
#define SEARCH_CELL_H

#include "geometry/point.h"

namespace ORNL {
    //! \class SearchCell
    //! \brief object representing a square cell used to find the pole of inaccessibility
    class SearchCell
    {
    public:
        SearchCell();

        //! \brief distance from center_point to the edge of the relevant PolygonList, negative if outside
        float distance;

        //! \brief distance between center_point and min_point
        float radius;

        //! \brief minimum point of the search cell
        Point min_point;

        //! \brief maximum point of the search cell
        Point max_point;

        //! \brief exact center of the search cell
        Point center_point;
    };
}

#endif // SEARCH_CELL_H
