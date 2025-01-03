#ifndef CURVE_FITTING_H
#define CURVE_FITTING_H

#include "geometry/path.h"
#include "geometry/segments/arc.h"
#include "geometry/segments/bezier.h"

//#define ARC_DEBUG_INFO

namespace ORNL
{
    //! \class CurveFitting
    //! \brief Provides a function to fit paths along a curves
    class CurveFitting
    {
    public:
        //! \brief fits a path with both arcs and splines (depending on settings)
        //! \pre all segments in the path must be line segments
        //! \note not all line segments are guaranteed to be replaced with curves
        //! \param path the path to fit to
        //! \param layer_settings the layer settings
        static void Fit(Path& path, QSharedPointer<SettingsBase> layer_settings);

    private:
        #ifdef ARC_DEBUG_INFO
        //! \brief saves debug info to file for graphs
        //! \param path the path to save info for
        static void writeDebugData(Path& path);

        //! \brief writes debug groups to file for graphs
        //! \param groups the groups
        static void writeDebugGroups(QVector<QPair<int, Path>> groups);
        #endif

        //! \brief removes segments that are smaller than a threshold
        //! \param path the path to clean
        //! \param threshold the cutoff size
        static void cleanPath(Path& path, Distance threshold);

        //! \brief breaks a given path up into sections that might support a curve based on angle
        //! \param path the path to break up
        //! \param layer_settings the layer settings
        //! \return a list of paths, each might be able to support a curve based on angle
        static QVector<Path> PartitionByAngle(Path& path, QSharedPointer<SettingsBase> layer_settings);

        //! \brief partitions based on the radius of curvature
        //! \param path the path to break up
        //! \param use_arcs if arcs are enabled
        //! \param use_spline if splines are enabled
        //! \return a list of paths, some of which will be able to support a curve based on curvature
        static QVector<QPair<int, Path>> PartitionByCurvature(Path& path, bool use_arcs, bool use_splines);

        //! \brief fits a given path to a curve-contour using up to path.size() number of sub-curves
        //! \param path the path to fit
        //! \param layer_settings the layer settings
        //! \return a path containing arcs and lines that now make up the new curve
        static Path DetermineBestArcFit(Path& path, QSharedPointer<SettingsBase> layer_settings);

        //! \brief fits a specific section of a path
        //! \param path the whole path
        //! \param start_index the segment to start fitting at
        //! \param end_index the segment to end fitting at
        //! \return the fit path and it's error
        static std::pair<QSharedPointer<ArcSegment>, double> FitArcToSection(Path path, int start_index, int end_index);

        //! \brief computes a combined error score that includes least-squares and arc length
        //! \param begin the segment the fit started at
        //! \param end the segment the fit ended at
        //! \param fit_segment the arc segment that was fit
        //! \return an error rating
        static double ArcError(int start, int end, Path& path, ArcSegment& fit_segment);

        //! \brief fits a path with dim number of arcs
        //! \param path the path to fit
        //! \param dim the number of sub-arcs to fit the path with
        //! \return a fit path and its error value
        static std::pair<Path, double> FitPathToArc(const Path& path, int dim);

        //! \brief recursive function used to bisect paths to fit splines
        //! \param spline a list of bezier segments that are building the spline
        //! \param parent_path the paths we are fitting to
        //! \param start the start index for this fit in the parent_path
        //! \param end the end index for this fit in the parent_path
        static void SplineTraverse(QVector<BezierSegment>& spline, Path& parent_path, int start, int end);
    };
}

#endif //CURVE_FITTING_H
