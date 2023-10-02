#ifndef POLYGON_H
#define POLYGON_H

//Libraries
#include "clipper.hpp"

// Single Path Lib
#ifdef HAVE_SINGLE_PATH
#include "single_path/geometry/polygon.h"
#include "single_path/geometry/point.h"
#endif


//Local
#include "geometry/point.h"
#include "geometry/path.h"
#include "units/derivative_units.h"

namespace ORNL
{
    class PolygonList;
    class Polyline;

    /*!
     * \class Polygon
     * \brief An individual polygon
     *
     * An individual polygon represented as a vector of points
     */
    class Polygon : public QVector< Point >
    {
    public:
        // QVector constructors
        using QVector<Point>::QVector;

        //! \brief Constructor
        Polygon() = default;

        //! \brief Conversion Constructor
        Polygon(const QVector< Point >& path);

        //! \brief Conversion Constructor
        Polygon(const QVector< ClipperLib2::IntPoint >& path);

        //! \brief Conversion Constructor
        Polygon(const ClipperLib2::Path& path);

        //! \brief Conversion Constructor
        Polygon(const Path& path);

        //! \brief Conversion Constructor
        Polygon(const QVector< Distance2D >& path);

        //! \brief Conversion Constructor
        Polygon(const QVector<QPair<double, double>>& pts);

        #ifdef HAVE_SINGLE_PATH
        //! \brief Conversion constructor
        Polygon(SinglePath::Polygon& polygon);

        //! \brief Conversion operator
        operator SinglePath::Polygon() const;
        #endif

        /*!
         * \brief Checks the orientation of the Polygon
         *
         * Checks the orientation of the polygon based on the area.
         * [Uses
         * Clipper](http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib2/Functions/Orientation.htm)
         *
         * \return
         *        - True: counterclockwise (outer polygon)
         *        - False: clockwise (hole polygon)
         */
        bool orientation() const;

        /*!
         * \brief Offsets the polygons
         *
         * Offsets the Polygon by moving the vertices along their angle
         * bisectors. Can seperate into multiple Polygon's contained by the
         * return value.
         */
        PolygonList offset(
            Distance distance,
            ClipperLib2::JoinType joinType = ClipperLib2::jtMiter) const;

        //! \brief The length of each of sides of the polygon summed
        int64_t polygonLength() const;

        /*!
         * \brief returns whether the length of the sides of the polygon is
         *        shorter than `checkLength`
         */
        bool shorterThan(Distance rhs) const;

        //! \brief Returns the center point of the polygon's bounding rect
        Point boundingRectCenter() const;

        //! \brief Smallest x, y, and z coordinates (may not be from the same
        //! point)
        Point min() const;

        //! \brief Largest x, y, and z coordinates (may not be from the same
        //! point)
        Point max() const;

        //! \brief Rotate the polygon around the origin
        Polygon rotate(const Angle& angle, const QVector3D& axis = {0, 0, 1}) const;

        //! \brief Rotate the polygon around a specified point
        Polygon rotateAround(const Point& center,
                             const Angle& angle,
                             const QVector3D& axis = {0, 0, 1}) const;

        //! \brief Rotates the polygon around its center
        Polygon rotateAroundCenter(const Angle& angle,
                                   const QVector3D& axis = {0, 0, 1}) const;

        //! \brief Restores point normals to geometry after it has been modified
        //! \param all_polys: List of Polygons to retrieve normals from
        //! \param offset: Whether normals are being restored after an offset operation.
        //! If true: normals will be restored from the closest point found within all_polys.
        //! If false: normals will be restored from exact matching point found within all_polys.
        //! If no exact match can be found, the bisecting normal will be computed and assigned.
        void restoreNormals(QVector<Polygon> all_polys, bool offset = false);

        //! \brief Reverses the direction of normals for the points of this polygon
        Polygon reverseNormalDirections();

        //! \brief shifts every point in the polygon by the vector
        //! \param vector of direction/length to shift
        Polygon translate(QVector3D shift);

        /*!
         * \brief Returns whether the point is inside the polygon
         */
        bool inside(const Point& point, bool border_result = false) const;

        //! \brief tests two polygons to see if they overlap
        //! \param p: the second polygon
        //! \return if the polygons overlap
        bool overlaps(const Polygon& p);

        /*!
         * \brief The area of the polygon
         *
         * - Positve: counterclockwise (outer polygon)
         * - Negative: clockwise (hole polygon)
         */
        Area area() const;

        //! \brief Returns the center of mass of the polygon
        Point centerOfMass() const;

        //! \brief Returns the closest point to `rhs`
        Point closestPointTo(const Point& rhs) const;

        // TODO: add psmooth
        //! \brief
        Polygon smooth(Distance removeLength);

        //! \brief Remove all mid collinear points
        Polygon simplify(Angle tolerance);

        /*!
         * \brief Removes vertices:
         * that join co-linear edges, or join edges that are almost co-linear
         * that are within the specified distance of an adjacent vertex
         * that are within the specified distance of a semi-adjacent vertex together with their out-lying vertices
         */
        Polygon cleanPolygon(const Distance distance = 10);

        //Yifan-CirclePack
        ClipperLib2::Path getPath();

        //! \brief Converts polygon to polyline
        //! \return Polyline that represents the underlying polygon
        Polyline toPolyline();

        /*!
         * \brief Add an instance of Polygon to this Polygon to make Polygons.
         *        Any overlapping Polygon are merged into one.
         */
        PolygonList operator+(const PolygonList& rhs);

        /*!
         * \brief Add an instance of Polygon to this Polygon to make Polygons.
         *        Any overlapping Polygon are merged into one.
         */
        PolygonList operator+(const Polygon& rhs);

        //! \brief Subtracts an instance of Polygon from this Polygon to make
        //! Polygons.
        PolygonList operator-(const PolygonList& rhs);

        //! \brief Subtracts an instance of Polygon from this Polygon to make
        //! Polygons.
        PolygonList operator-(const Polygon& rhs);

        /*!
         * \brief Unions an instance of Polygon to this Polygon to make
         * Polygons. Any overlapping Polygon are merged into one.
         */
        PolygonList operator|(const PolygonList& rhs);

        /*!
         * \brief Union an instance of Polygon to this Polygon to make Polygons.
         *        Any overlapping Polygon are merged into one.
         */
        PolygonList operator|(const Polygon& rhs);

        /*!
         * \brief Returns the intersection of this Polygon and other Polygons
         *        as 0 or more Polygon's
         */
        PolygonList operator&(const PolygonList& rhs);

        /*!
         * \brief Returns the intersection of this Polygon and other Polygons
         *        as 0 or more Polygons
         */
        PolygonList operator&(const Polygon& rhs);

        /*!
         * \brief Returns the intersection of this polygon and a polygon as
         *        0 or more polylines
         */
        QVector< Polyline > operator&(const Polyline& rhs);

        /*!
         * \brief Returns the xor of this Polygon and other Polygons
         *        as 0 or more Polygons
         */
        PolygonList operator^(const PolygonList& rhs);

        /*!
         * \brief Returns the xor of this Polygon and other Polygons
         *        as 0 or more Polygons
         */
        PolygonList operator^(const Polygon& rhs);

        //! \brief Checks if two polygons are identical
        bool operator==(const Polygon& rhs) const;

        // friend classes can use each other's private functions/members
        friend class PolygonList;
        friend class Polyline;

        //! \brief operator for use with ClipperLib2
        ClipperLib2::Path operator()() const;

        //! \brief generates a No Fit Polygon for the orbiting polygon
        //!        as a guide for where it can be placed so that it is
        //!        always touching (but never intersecting) the stationary polygon
        //! \param stationary
        //! \param orbiting
        //! \param inner: generates inner NFP when set to true
        //! \return No-Fit-Polygon
        Polygon noFitPolygon(Polygon stationary, Polygon orbiting, bool inner);

        //! \brief aligns a specified vertex of a polygon with a provided point
        //! \param poly
        //! \param target
        //! \param vertex
        //! \return aligned polygon
        Polygon alignVertex(Polygon poly, Point target, int vertex);

    };  // Class Polygon
}  // namespace ORNL
#endif  // POLYGON_H
