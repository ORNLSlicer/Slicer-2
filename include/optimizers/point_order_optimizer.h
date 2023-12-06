#ifndef POINTORDEROPTIMIZER_H
#define POINTORDEROPTIMIZER_H

#include "utilities/enums.h"
#include "configs/settings_base.h"
#include "geometry/path.h"
#include "geometry/point.h"

namespace ORNL {

    /*!
     * \class PointOrderOptimizer
     * \brief This class is used to select specific points on a path to link to
     *
     * \note These are how the optimizations are handled:
     * \note Non-implemented optimizations default to shortest distance
     * \list Next Closest – Pick the point on the path that is closest to the current location.
     * \list Next Farthest – Pick the point on the path that is farthest from the current location.
     * \list Random – Randomly select a point on each path.
     * \list Consecutive – Move the start point a minimum user defined distance from the previous layer’s start point.
     * \list Custom Point – User defines a custom point, and the optimizer picks the point on each path that is closest to the defined point.
     */

    class PointOrderOptimizer{
        public:

            //! \brief Constructor
            //! \note Layer number is only needed if kConsecutive linking is used
            //! \param current_location: The current location
            //! \param path: The path to link to
            //! \param layer_number: The layer number of the current path
            //! \param sb: The settings base to use
            //! \return The index of the point to link to
            static int linkToPoint(Point current_location, Path path, uint layer_number, const QSharedPointer<SettingsBase>& sb);

        private:

            //! \brief Finds the shortest or longest distance between the current location and the points in the path
            //! \param path: The path to find the shortest or longest distance in
            //! \param startPoint: The current location
            //! \param minThresholdEnable: Whether or not to use the minimum threshold
            //! \param minThreshold: The minimum threshold to use
            //! \param shortest: Whether or not to find the shortest distance
            //! \return The index of the point with the shortest or longest distance
            static int findShortestOrLongestDistance(Path path, Point startPoint, bool minThresholdEnable, Distance minThreshold, bool shortest = true);

            //! \brief Links to a random point in the path
            //! \param path: The path to link to
            //! \return The index of the point to link to
            static int linkToRandom(Path path);

            //! \brief Links to the next consecutive point in the path based on the layer number
            //! \param path: The path to link to
            //! \param layer_number: The layer number of the current path
            //! \param minDist: The minimum distance the consecutive point must be from the previous layer's start point
            //! \return The index of the point to link to
            static int linkToConsecutive(Path path, uint layer_number, Distance minDist);

            //! \brief Computes the perturbation of the point after the current optimization scheme is run
            //! \param path: The path to select candidates for perturbation
            //! \param current_start: The current start point
            //! \param radius: The radius of the perturbation
            //! \return The index of the point to link to
            static int computePerturbation(Path path, Point current_start, Distance radius);
    };
}

#endif // POINTORDEROPTIMIZER_H
