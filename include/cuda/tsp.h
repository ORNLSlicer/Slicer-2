#ifndef TSP
#define TSP

/*!
 * This class is the CUDA backend to the TSP solver
 */
namespace ORNL {
    //! \brief Computes the shortest or longest TSP for a N given X and Y coordinates and a start index
    int * compute_tsp(float * h_x, float * h_y, int size, int startIndex, bool shortest);
}

#endif
