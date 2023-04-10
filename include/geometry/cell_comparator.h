#ifndef CELL_COMPARATOR_H
#define CELL_COMPARATOR_H

#include "geometry/search_cell.h"

namespace ORNL
{
    //! \class CellComparator
    //! \brief comparator class used to sort the SearchCell priority queue in PolygonList's PoleOfInaccessibility method
    class CellComparator
    {
    public:
        CellComparator();

        //! \brief compares two search cells based on potential distance
        //! \param cell_a
        //! \param cell_b
        //! \return true if potential distance of cell_a is less than that of cell_b, false otherwise
        bool operator()(SearchCell& cell_a, SearchCell& cell_b);
    };
}

#endif // CELL_COMPARATOR_H
