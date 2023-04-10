#include "geometry/cell_comparator.h"

namespace ORNL
{
    CellComparator::CellComparator()
    {}

    bool CellComparator::operator()(SearchCell &cell_a, SearchCell &cell_b)
    {
        return (cell_a.distance + cell_a.radius) < (cell_b.distance + cell_b.radius);
    }
}
