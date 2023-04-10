//Main Module
#include "geometry/polygon.h"

//Local
#include "geometry/point.h"
#include "geometry/polygon_list.h"
#include "geometry/polyline.h"
#include "utilities/mathutils.h"

namespace ORNL
{
    Polygon::Polygon(const QVector< Point >& path)
    {
        for (Point point : path)
        {
            append(point);
        }
    }

    Polygon::Polygon(const QVector< ClipperLib2::IntPoint >& path)
    {
        for (const ClipperLib2::IntPoint& point : path)
        {
            append(Point(point));
        }
    }

    Polygon::Polygon(const ClipperLib2::Path& path)
    {
        for (const ClipperLib2::IntPoint& point : path)
        {
            append(Point(point));
        }
    }

    Polygon::Polygon(const Path& path)
    {
        for (int i = 0; i < path.size(); i++)
        {
            append(path[i]->start());
        }
    }

    Polygon::Polygon(const QVector< Distance2D >& path)
    {
        for (Distance2D point : path)
        {
            append(Point(point));
        }
    }

    Polygon::Polygon(const QVector<QPair<double, double>>& pts)
    {
        for(QPair<double, double> pt : pts)
            append(Point(pt.first, pt.second));
    }

    #ifdef HAVE_SINGLE_PATH
    Polygon::Polygon(SinglePath::Polygon &polygon)
    {
        for(SinglePath::Point point : polygon)
        {
            append(point);
        }
    }

    ORNL::Polygon::operator SinglePath::Polygon() const
    {
        SinglePath::Polygon polygon;
        for(Point point : *this)
        {
            polygon.append(point);
        }
        return polygon;
    }
    #endif

    bool Polygon::orientation() const
    {
        return ClipperLib2::Orientation((*this)());
    }

    PolygonList Polygon::offset(Distance distance,
                                ClipperLib2::JoinType joinType) const
    {
        ClipperLib2::Paths paths;
        ClipperLib2::ClipperOffset clipper;
        clipper.AddPath(operator()(), joinType, ClipperLib2::etClosedPolygon);
        clipper.Execute(paths, distance());
        PolygonList polygons(paths);

        polygons.restoreNormals(QVector<Polygon>{*this}, true);

        return polygons;
    }

    int64_t Polygon::polygonLength() const
    {
        int64_t rv = 0;
        Point prev = operator[](size() - 1);
        for (int i = 0; i < size(); i++)
        {
            Point current = (*this)[i];
            rv += current.distance(prev)();
        }
        return rv;
    }

    bool Polygon::shorterThan(Distance rhs) const
    {
        return polygonLength() < rhs();
    }

    Point Polygon::boundingRectCenter() const
    {
        Point minimum = min();
        Point maximum = max();

        return (maximum + minimum) / 2.0;
    }

    Point Polygon::min() const
    {
        Point rv(std::numeric_limits< float >::max(),
                 std::numeric_limits< float >::max(),
                 std::numeric_limits< float >::max());
        for (const Point& point : *this)
        {
            if (point.x() < rv.x())
            {
                rv.x(point.x());
            }

            if (point.y() < rv.y())
            {
                rv.y(point.y());
            }

            if (point.z() < rv.z())
            {
                rv.z(point.z());
            }
        }
        return rv;
    }

    Point Polygon::max() const
    {
        Point rv(std::numeric_limits< float >::lowest(),
                 std::numeric_limits< float >::lowest(),
                 std::numeric_limits< float >::lowest());
        for (const Point& point : *this)
        {
            if (point.x() > rv.x())
            {
                rv.x(point.x());
            }

            if (point.y() > rv.y())
            {
                rv.y(point.y());
            }

            if (point.z() > rv.z())
            {
                rv.z(point.z());
            }
        }
        return rv;
    }

    Polygon Polygon::rotate(const Angle& rotation_angle, const QVector3D& axis) const
    {
        return rotateAround({0, 0, 0}, rotation_angle, axis);
    }

    Polygon Polygon::rotateAround(const Point& center,
                                  const Angle& rotation_angle,
                                  const QVector3D& axis) const
    {
        Polygon polygon;

        for (Point point : *this)
        {
            polygon.append(point.rotateAround(center, rotation_angle, axis));
        }
        return polygon;
    }

    Polygon Polygon::rotateAroundCenter(const Angle& angle, const QVector3D& axis) const
    {
        return rotateAround(boundingRectCenter(), angle, axis);
    }

    void Polygon::restoreNormals(QVector<Polygon> all_polys, bool offset)
    {
        if (offset) //! Offset operation: assign normals of closest point
        {
            for (Point& p1 : *this)
            {
                Distance min_dist = Distance(std::numeric_limits<float>::max());

                for (Polygon& poly : all_polys)
                {
                    Point p2 = poly.closestPointTo(p1);

                    if (p1.distance(p2) < min_dist)
                    {
                        min_dist = p1.distance(p2);
                        p1.setNormals(p2.getNormals());
                    }
                }
            }
        }
        else //! Clipping operation: assign normals of exact point. If point can't be found, compute bisecting normal.
        {
            auto index = [] (uint i, uint last)
            {
                uint ret = i;
                if (i < 0)
                    ret = last;
                else if (i > last)
                    ret = 0;

                return ret;
            };

            for(uint i = 0, size = this->size(); i < size; ++i)
            {
                bool found = false;
                for (Polygon& poly : all_polys)
                {
                    for (Point& p : poly)
                    {
                        if((*this)[i] == p)
                        {
                            (*this)[i].setNormals(p.getNormals());
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }

                if (!found) //! Compute bisecting normal
                {
                    uint last = (*this).size() - 1;

                    QVector3D unit_z {0, 0, 1};
                    QVector3D prev = ((*this)[index(i - 1, last)] - (*this)[i]).toQVector3D().normalized();
                    QVector3D next = ((*this)[index(i + 1, last)] - (*this)[i]).toQVector3D().normalized();

                    QVector3D normal = (QVector3D::crossProduct(unit_z, prev) + QVector3D::crossProduct(next, unit_z)).normalized();
                    (*this)[i].setNormals(QVector<QVector3D>{normal, normal});
                }
            }
        }
    }

    Polygon Polygon::reverseNormalDirections()
    {
        for (Point& point : *this)
            point.reverseNormalDirections();

        return *this;
    }

    Polygon Polygon::translate(QVector3D shift)
    {
        Polygon shifted_polygon;
        shifted_polygon.reserve(size());
        for (Point point : *this)
        {
            shifted_polygon.append(point + shift.toPoint());
        }
        return shifted_polygon;
    }

    bool Polygon::inside(const Point& point, bool border_result) const
    {
        int res = ClipperLib2::PointInPolygon(point.toIntPoint(), (*this)());
        if (res == -1)
        {
            return border_result;
        }
        return res == 1;
    }

    bool Polygon::overlaps(const Polygon& p){
        ClipperLib2::Clipper clipper;

        clipper.AddPath((*this)(), ClipperLib2::ptSubject, true);
        clipper.AddPath(p(), ClipperLib2::ptClip, true);

        ClipperLib2::Paths solution;
        clipper.Execute(ClipperLib2::ctIntersection,
                        solution,
                        ClipperLib2::pftNonZero,
                        ClipperLib2::pftNonZero);

        return !solution.empty();
    }

    Area Polygon::area() const
    {
        return Area(ClipperLib2::Area((*this)()));
    }

    Point Polygon::centerOfMass() const
    {
        double x = 0, y = 0;
        Point p0 = back();
        for (const Point& p1 : *this)
        {
            double second_factor = (p0.x() * p1.y()) - (p1.x() * p0.y());

            x += double(p0.x() + p1.x()) * second_factor;
            y += double(p0.y() + p1.y()) * second_factor;
            p0 = p1;
        }

        Area a = area();

        x = x / 6 / a();
        y = y / 6 / a();

        return Point(x, y);
    }

    Point Polygon::closestPointTo(const Point& rhs) const
    {
        Point ret         = rhs;
        Distance bestDist = Distance(std::numeric_limits< float >::max());
        for (Point point : (*this))
        {
            Distance dist = rhs.distance(point);
            if (dist < bestDist)
            {
                ret      = point;
                bestDist = dist;
            }
        }
        return ret;
    }

    Polygon Polygon::simplify(Angle tolerance)
    {
        Polygon polygon;
        polygon.reserve(size());
        Point current, next;
        Point prev = operator[](size() - 1);
        for (int i = 0; i < size(); i++)
        {
            current = (*this)[i];
            next = (*this)[(i+1) % size()];
            if(!MathUtils::nearCollinear(prev, current, next, tolerance))
            {
                polygon.append(current);
            }
            prev = current;
        }
        polygon.squeeze();
        return polygon;
    }

    Polygon Polygon::cleanPolygon(const Distance distance)
    {
        ClipperLib2::Path path;
        ClipperLib2::CleanPolygon((*this)(), path, distance());
        Polygon result(path);

        result.restoreNormals({*this});

        return result;
    }

    Polyline Polygon::toPolyline()
    {
        Polyline newLine;
        newLine.reserve(this->size());

        for(auto it = this->begin(), end = this->end(); it != end; ++it)
            newLine.push_back(*it);

        newLine.push_back(this->at(0));

        return newLine;
    }

    PolygonList Polygon::operator+(const PolygonList& rhs)
    {
        QVector<Polygon> all_polys = QVector<Polygon>{*this} + QVector<Polygon>{rhs};

        ClipperLib2::Paths paths;
        ClipperLib2::Clipper clipper;
        clipper.AddPath((*this)(), ClipperLib2::ptSubject, true);
        clipper.AddPaths(rhs(), ClipperLib2::ptSubject, true);
        clipper.Execute(ClipperLib2::ctUnion,
                        paths,
                        ClipperLib2::pftNonZero,
                        ClipperLib2::pftNonZero);
        PolygonList result(paths);

        result.restoreNormals(all_polys);

        return result;
    }

    PolygonList Polygon::operator+(const Polygon& rhs)
    {
        QVector<Polygon> all_polys = QVector<Polygon>{*this} + QVector<Polygon>{rhs};

        ClipperLib2::Paths paths;
        ClipperLib2::Clipper clipper;
        clipper.AddPath((*this)(), ClipperLib2::ptSubject, true);
        clipper.AddPath(rhs(), ClipperLib2::ptSubject, true);
        clipper.Execute(ClipperLib2::ctUnion,
                        paths,
                        ClipperLib2::pftNonZero,
                        ClipperLib2::pftNonZero);
        PolygonList result(paths);

        result.restoreNormals(all_polys);

        return result;
    }

    PolygonList Polygon::operator-(const PolygonList& other)
    {
        QVector<Polygon> all_polys = QVector<Polygon>{*this} + QVector<Polygon>{PolygonList(other).reverseNormalDirections()};

        ClipperLib2::Paths paths;
        ClipperLib2::Clipper clipper;
        clipper.AddPath((*this)(), ClipperLib2::ptSubject, true);
        clipper.AddPaths(other(), ClipperLib2::ptClip, true);
        clipper.Execute(ClipperLib2::ctDifference,
                        paths,
                        ClipperLib2::pftNonZero,
                        ClipperLib2::pftNonZero);
        PolygonList result(paths);

        result.restoreNormals(all_polys);

        return result;
    }

    PolygonList Polygon::operator-(const Polygon& rhs)
    {
        QVector<Polygon> all_polys = QVector<Polygon>{*this} + QVector<Polygon>{Polygon(rhs).reverseNormalDirections()};

        ClipperLib2::Paths paths;
        ClipperLib2::Clipper clipper;
        clipper.AddPath((*this)(), ClipperLib2::ptSubject, true);
        clipper.AddPath(rhs(), ClipperLib2::ptClip, true);
        clipper.Execute(ClipperLib2::ctDifference,
                        paths,
                        ClipperLib2::pftNonZero,
                        ClipperLib2::pftNonZero);
        PolygonList result(paths);

        result.restoreNormals(all_polys);

        return result;
    }

    PolygonList Polygon::operator|(const PolygonList& rhs)
    {
        QVector<Polygon> all_polys = QVector<Polygon>{*this} + QVector<Polygon>{rhs};

        ClipperLib2::Paths paths;
        ClipperLib2::Clipper clipper;
        clipper.AddPath((*this)(), ClipperLib2::ptSubject, true);
        clipper.AddPaths(rhs(), ClipperLib2::ptSubject, true);
        clipper.Execute(ClipperLib2::ctUnion,
                        paths,
                        ClipperLib2::pftNonZero,
                        ClipperLib2::pftNonZero);
        PolygonList result(paths);

        result.restoreNormals(all_polys);

        return result;
    }

    PolygonList Polygon::operator|(const Polygon& rhs)
    {
        ClipperLib2::Paths paths;
        ClipperLib2::Clipper clipper;
        clipper.AddPath((*this)(), ClipperLib2::ptSubject, true);
        clipper.AddPath(rhs(), ClipperLib2::ptSubject, true);
        clipper.Execute(ClipperLib2::ctUnion,
                        paths,
                        ClipperLib2::pftNonZero,
                        ClipperLib2::pftNonZero);
        PolygonList result(paths);

        result.restoreNormals({*this, rhs});

        return result;
    }

    PolygonList Polygon::operator&(const PolygonList& rhs)
    {
        QVector<Polygon> all_polys = QVector<Polygon>{*this} + QVector<Polygon>{rhs};

        ClipperLib2::Paths paths;
        ClipperLib2::Clipper clipper;
        clipper.AddPath((*this)(), ClipperLib2::ptSubject, true);
        clipper.AddPaths(rhs(), ClipperLib2::ptClip, true);
        clipper.Execute(ClipperLib2::ctIntersection,
                        paths,
                        ClipperLib2::pftNonZero,
                        ClipperLib2::pftNonZero);
        PolygonList result(paths);

        result.restoreNormals(all_polys);

        return result;
    }

    PolygonList Polygon::operator&(const Polygon& rhs)
    {
        ClipperLib2::Paths paths;
        ClipperLib2::Clipper clipper;
        clipper.AddPath((*this)(), ClipperLib2::ptSubject, true);
        clipper.AddPath(rhs(), ClipperLib2::ptClip, true);
        clipper.Execute(ClipperLib2::ctIntersection,
                        paths,
                        ClipperLib2::pftNonZero,
                        ClipperLib2::pftNonZero);
        PolygonList result(paths);

        result.restoreNormals({*this, rhs});

        return result;
    }

    QVector< Polyline > Polygon::operator&(const Polyline& rhs)
    {
        ClipperLib2::PolyTree poly_tree;
        ClipperLib2::Paths paths;
        ClipperLib2::Clipper clipper;
        clipper.AddPath(rhs(), ClipperLib2::ptSubject, false);
        clipper.AddPath(operator()(), ClipperLib2::ptClip, true);
        clipper.Execute(ClipperLib2::ctIntersection,
                        poly_tree,
                        ClipperLib2::pftNonZero,
                        ClipperLib2::pftNonZero);
        ClipperLib2::OpenPathsFromPolyTree(poly_tree, paths);

        QVector< Polyline > rv;
        for (ClipperLib2::Path path : paths)
        {
            rv += Polyline(path);
        }
        return rv;
    }

    PolygonList Polygon::operator^(const PolygonList& rhs)
    {
        ClipperLib2::Paths paths;
        ClipperLib2::Clipper clipper;
        clipper.AddPath((*this)(), ClipperLib2::ptSubject, true);
        clipper.AddPaths(rhs(), ClipperLib2::ptClip, true);
        clipper.Execute(ClipperLib2::ctXor,
                        paths,
                        ClipperLib2::pftNonZero,
                        ClipperLib2::pftNonZero);
        PolygonList result(paths);

        return result;
    }

    PolygonList Polygon::operator^(const Polygon& rhs)
    {
        ClipperLib2::Paths paths;
        ClipperLib2::Clipper clipper;
        clipper.AddPath((*this)(), ClipperLib2::ptSubject, true);
        clipper.AddPath(rhs(), ClipperLib2::ptClip, true);
        clipper.Execute(ClipperLib2::ctXor,
                        paths,
                        ClipperLib2::pftNonZero,
                        ClipperLib2::pftNonZero);
        PolygonList result(paths);

        return result;
    }

    ClipperLib2::Path Polygon::operator()() const
    {
        ClipperLib2::Path path;
        for (Point point : (*this))
        {
            path.push_back(point.toIntPoint());
        }
        return path;
    }

    Polygon Polygon::noFitPolygon(Polygon stationary, Polygon orbiting, bool inner) // for use in the sheet lamination slicer
    {
        Polygon stationary_copy = Polygon::alignVertex(stationary, Point(0,0,0), 0);
        Polygon orbiting_copy = Polygon::alignVertex(orbiting, stationary_copy[0], 0);
        ClipperLib2::Path stationaryPath = stationary_copy.getPath(); //convert to clipper paths
        ClipperLib2::Path orbiting_copy_rotated_path = orbiting_copy.rotateAround(Point(stationary_copy[0].x(), stationary_copy[0].y(), 0), 180 * deg).getPath(); //rotate orbiting by 180 to made Minkowski difference
        ClipperLib2::Paths minkowski_paths;
        ClipperLib2::MinkowskiSum(stationaryPath, orbiting_copy_rotated_path, minkowski_paths, true);
        Polygon nfp = Polygon::alignVertex(minkowski_paths[int(inner)], stationary[0] + minkowski_paths[int(inner)][0], 0); // now we need to move the NFP back to align with the original stationary
        return nfp;
    }

    Polygon Polygon::alignVertex(Polygon poly, Point polyTarget, int polyVertex) // for use in the sheet lamination slicer // polyVertex usually only needs to be set to 0
    {
        Polygon polyCopy = poly;
        float polyCopyX = poly[polyVertex].x();
        float polyCopyY = poly[polyVertex].y();
        for (int i = 0; i < poly.length(); i++)
        {
            polyCopy[i].x(polyCopy[i].x() - polyCopyX + polyTarget.x());
            polyCopy[i].y(polyCopy[i].y() - polyCopyY + polyTarget.y());
        }
        return polyCopy;
    }

    bool Polygon::operator==(const Polygon &rhs) const
    {
        if(rhs.size() != this->size()) // they both must be the same size
            return false;

        for(int index = 0; index < this->size(); index++)
        {
            if(this->at(index) != rhs[index]) // The points differ
                return false;
        }
        return true;
    }

    ClipperLib2::Path Polygon::getPath()
    {
      return this->operator ()();
    }
}  // namespace ORNL
