#ifndef SHAPE_FACTORY_H
#define SHAPE_FACTORY_H

#include <vector>

#include <QMatrix4x4>
#include <QColor>
#include <units/unit.h>
#include <geometry/point.h>

namespace ORNL
{
    //! \brief "Class" for creating various geometries. Input vectors are appended to so shapes can be built additively.
    class ShapeFactory
    {
    public:
        ShapeFactory();

        /*! \brief Append the data for a rectangular prism to input vectors
         *
         *  @param length Total X dimension
         *  @param width Total Y dimension
         *  @param height Total Z dimension
         *  @param transform Matrix to apply to each vertex
         *  @param color Color of prism
         *  @param vertices Vector of vertices to append the new vertices to
         *  @param colors Vector of colors to append the new colors to
         *  @param normals Vector of normals to append the new normals to
         */
        static void createRectangle(float length, float width, float height, const QMatrix4x4& transform, const QColor& color, std::vector<float>& vertices, std::vector<float>& colors, std::vector<float>& normals);

        /*! \brief Append the data for a torus to input vectors
         *
         *  @param inner_radius Inner radius of torus
         *  @param outer_radius Outer radius of torus
         *  @param transform Matrix to apply to each vertex
         *  @param color Color of torus
         *  @param vertices Vector of vertices to append the new vertices to
         *  @param colors Vector of colors to append the new colors to
         *  @param normals Vector of normals to append the new normals to
         */
        static void createTorus(float inner_radius, float outer_radius, const QMatrix4x4& transform, const QColor& color, std::vector<float>& vertices, std::vector<float>& colors, std::vector<float>& normals);

        /*! \brief Append the data for a cylinder to input vectors
         *
         *  @param radius Radius of cylinder
         *  @param height Height of cylinder
         *  @param transform Matrix to apply to each vertex
         *  @param color Color of cylinder
         *  @param vertices Vector of vertices to append the new vertices to
         *  @param colors Vector of colors to append the new colors to
         *  @param normals Vector of normals to append the new normals to
         */
        static void createCylinder(float radius, float height, const QMatrix4x4& transform, const QColor& color, std::vector<float>& vertices, std::vector<float>& colors, std::vector<float>& normals);

        /*! \brief Append the data for a sphere to input vectors
         *
         *  @param radius: Radius of sphere
         *  @param sectorCount: Number of horizontal sectors that make up sphere
         *  @param stackCount: Number of vertical sectors that make up sphere
         *  @param transform: Matrix to apply to each vertex
         *  @param color: Color of sphere
         *  @param vertices: Vector of vertices to append the new vertices to
         *  @param colors: Vector of colors to append the new colors to
         *  @param normals: Vector of normals to append the new normals to
         */
        static void createSphere(float radius, int sectorCount, int stackCount, const QMatrix4x4& transform, const QColor& color,
                                        std::vector<float>& vertices, std::vector<float>& colors, std::vector<float>& normals);

        /*! \brief Append the data for a gcode cylinder to input vectors
         *
         *  A gcode cylinder is a cylinder with the arcs from PI/4 to 3PI/4 and 5PI/4 to 7PI/4 replaced with flat lines. It looks like a cylinder that
         *  has been squished flat on two sides.
         *  This is the public interface to the gcode graphics which translates the start and displacement into a transform matrix usable by the private
         *  function of the same name.
         *  @param width Radius of gcode cylinder
         *  @param height Height of gcode cylinder
         *  @param start Vector indicating where the center of the base of this cylinder should be
         *  @param displacement Vector indicating the direction and length of this cylinder
         *  @param color Color of gcode cylinder
         *  @param vertices Vector of vertices to append the new vertices to
         *  @param colors Vector of colors to append the new colors to
         *  @param normals Vector of normals to append the new normals to
         */
        static void createGcodeCylinder(float width, float height, const QVector3D& start, const QVector3D& displacement, const QColor& color, std::vector<float>& vertices, std::vector<float>& colors, std::vector<float>& normals);

        /*!
         * \brief appends the data for a arc cylinder to input vectors
         * \note this is an overload that automatically computes the transform
         * @param cylinder_height how thick of a cylinder to draw
         * @param start the start point of the cylinder
         * @param center the center point of the arc
         * @param angle the angle of the arc
         * @param transform the transformation matrix to place the arc at the correct point
         * @param color the color to draw the arc as
         * @param vertices Vector of vertices to append the new vertices to
         * @param colors Vector of colors to append the new colors to
         * @param normals Vector of normals to append the new normals to
         */
        static void createArcCylinder(const float cylinder_height, const Point& start, const Point& center, const Point& end, bool is_ccw, const QColor& color, std::vector<float>& vertices, std::vector<float>& colors, std::vector<float>& normals);

        /*!
         * \brief appends the data for a spline cylinder for input values
         * @param cylinder_height how thick of a cylinder to draw
         * @param start the start point of the cylinder
         * @param control_a the first control point
         * @param control_b the second control point
         * @param end the end point of the spline
         * @param color the color to draw the arc as
         * @param vertices Vector of vertices to append the new vertices to
         * @param colors Vector of colors to append the new colors to
         * @param normals Vector of normals to append the new normals to
         */
        static void createSplineCylinder(const float cylinder_height, const Point& start, const Point& control_a, const Point& control_b, const Point& end, const QColor& color, std::vector<float>& vertices, std::vector<float>& colors, std::vector<float>& normals);

        /*! \brief Append the data for a cone to input vectors
         *
         *  @param radius Radius of base of cone
         *  @param height Height from base to tip
         *  @param transform Matrix to apply to each vertex
         *  @param color Color of cone
         *  @param vertices Vector of vertices to append the new vertices to
         *  @param colors Vector of colors to append the new colors to
         *  @param normals Vector of normals to append the new normals to
         */
        static void createCone(float radius, float height, const QMatrix4x4& transform, const QColor& color, std::vector<float>& vertices, std::vector<float>& colors, std::vector<float>& normals);

        /*! \brief Append the data for a full gimbal (all 6 components) to input vectors
         *
         *  Constructs 3 torii for rotation gimbals and 3 pairs of cylinders and cones to make arrows for the translation gimbals
         *  @param inner_radius Inner radius of torii
         *  @param outer_radius Outer radius of torii
         *  @param center Starting position of gimbal relative to local coordinates of part it is being built around
         *  @param vertices Vector of vertices to append the new vertices to
         *  @param colors Vector of colors to append the new colors to
         *  @param normals Vector of normals to append the new normals to
         */
        static void createFullGimbal(float inner_radius, float outer_radius, const QVector3D& center, std::vector<float>& vertices, std::vector<float>& colors, std::vector<float>& normals);

        /*! \brief Append the data for a xy-gimbal to the input vectors
         *
         *  Constructs one torus in the xy-plane and 2 pairs of cylinders and cones to make arrows in the x and y direction.
         *  @param inner_radius Inner radius of torii
         *  @param outer_radius Outer radius of torii
         *  @param center Starting position of gimbal relative to local coordinates of part it is being built around
         *  @param vertices Vector of vertices to append the new vertices to
         *  @param colors Vector of colors to append the new colors to
         *  @param normals Vector of normals to append the new normals to
         */
        static void createXYGimbal(float inner_radius, float outer_radius, const QVector3D& center, std::vector<float>& vertices, std::vector<float>& colors, std::vector<float>& normals);

        /*! \brief Append the data for a xz-gimbal to the input vectors
         *
         *  Constructs one torus in the xz-plane and 2 pairs of cylinders and cones to make arrows in the x and z direction.
         *  @param inner_radius Inner radius of torii
         *  @param outer_radius Outer radius of torii
         *  @param center Starting position of gimbal relative to local coordinates of part it is being built around
         *  @param vertices Vector of vertices to append the new vertices to
         *  @param colors Vector of colors to append the new colors to
         *  @param normals Vector of normals to append the new normals to
         */
        static void createXZGimbal(float inner_radius, float outer_radius, const QVector3D& center, std::vector<float>& vertices, std::vector<float>& colors, std::vector<float>& normals);

        /*! \brief Create rectangle for use as a plane.
         *
         *  Constructs a wire frame rectangular prism build volume representation
         *  @param length: length of plane
         *  @param width: length of plane
         *  @param x_grid_dist: Distance between grid lines in x direction
         *  @param y_grid_dist: Distance between grid lines in y direction
         *  @param color Color of resulting volume
         *  @param vertices Vector of vertices to append the new vertices to
         *  @param colors Vector of colors to append the new colors to
         */
        static void createGridPlane(float length, float width, float x_grid_dist, float y_grid_dist, const QColor& color, std::vector<float>& vertices, std::vector<float>& colors);

        /*! \brief Create rectangle for build volume representation
         *
         *  Constructs a wire frame rectangular prism build volume representation
         *  @param min Min value of rectangle
         *  @param max Max value of rectangle
         *  @param x_grid_dist: Distance between grid lines in x direction
         *  @param x_grid_offset: Distance to offset the first grid line from the minimum X
         *  @param y_grid_dist: Distance between grid lines in y direction
         *  @param y_grid_offset: Distance to offset the first grid line from the minimum Y
         *  @param color Color of resulting volume
         *  @param vertices Vector of vertices to append the new vertices to
         *  @param colors Vector of colors to append the new colors to
         */
        static void createBuildVolumeRectangle(QVector3D min, QVector3D max, float x_grid_dist, float x_grid_offset, float y_grid_dist,
                                               float y_grid_offset, const QColor& color, std::vector<float>& vertices, std::vector<float>& colors);

        /*! \brief Create cylinder for build volume representation
         *
         *  Constructs a wire frame cylindrical build volume representation
         *  @param radius Radius of circle for top/bottom of cylinder
         *  @param height Height of cylinder
         *  @param x_grid_dist: Distance between grid lines in x direction
         *  @param y_grid_dist: Distance between grid lines in y direction
         *  @param color Color of resulting volume
         *  @param vertices Vector of vertices to append the new vertices to
         *  @param colors Vector of colors to append the new colors to
         */
        static void createBuildVolumeCylinder(float radius, float height, float x_grid_dist, float y_grid_dist,
                                              const QColor& color, std::vector<float>& vertices, std::vector<float>& colors);

        /*! \brief Create torus for build volume representation
         *
         *  Constructs a wire frame toroidal build volume representation
         *  @param outerRadius Radius of outer circle for top/bottom of cylinder
         *  @param innerRadius Radius of iner circle for top/bottom of cylinder
         *  @param height Height of torus
         *  @param x_grid_dist: Distance between grid lines in x direction
         *  @param y_grid_dist: Distance between grid lines in y direction
         *  @param color Color of resulting volume
         *  @param vertices Vector of vertices to append the new vertices to
         *  @param colors Vector of colors to append the new colors to
         */
        static void createBuildVolumeToroidal(float outerRadius, float innerRadius, float height, float x_grid_dist, float y_grid_dist,
                                              const QColor& color, std::vector<float>& vertices, std::vector<float>& colors);

        //! \brief Constructs an arrow.
        static void createArrow(QVector3D begin, QVector3D end, const QColor& color, std::vector<float>& vertices, std::vector<float>& colors);

    private:

        /*! \brief Helper function to append the data for a gcode cylinder to input vector
         *
         *  A gcode cylinder is a cylinder with the arcs from PI/4 to 3PI/4 and 5PI/4 to 7PI/4 replaced with flat lines. It looks like a cylinder that
         *  has been squished flat on two sides.
         *  @param width Width of rectangular cross section
         *  @param height Height of rectangular cross section
         *  @param transform Matrix to apply to each vertex
         *  @param color Color of gcode cylinder
         *  @param vertices Vector of vertices to append the new vertices to
         *  @param colors Vector of colors to append the new colors to
         *  @param normals Vector of normals to append the new normals to
         */
        static void createGcodeCylinder(float width, float height, const QMatrix4x4& transform, const QColor& color, std::vector<float>& vertices, std::vector<float>& colors, std::vector<float>& normals);

        /*!
         * \brief appends the data for a clockwise (G2) arc cylinder to input vectors
         * @param cylinder_height how thick of a cylinder to draw
         * @param start the start point of the cylinder
         * @param center the center point of the arc
         * @param angle the angle of the arc
         * @param transform the transformation matrix to place the arc at the correct point
         * @param color the color to draw the arc as
         * @param vertices Vector of vertices to append the new vertices to
         * @param colors Vector of colors to append the new colors to
         * @param normals Vector of normals to append the new normals to
         */
        static void createArcCylinder(float cylinder_height, const Point& start, const Point& center, const Point& end, const QMatrix4x4& transform, const QColor& color, std::vector<float>& vertices, std::vector<float>& colors, std::vector<float>& normals);

        /*!
         * \brief appends the data for a counter-clockwise (G3) arc cylinder to input vectors
         * @param cylinder_height how thick of a cylinder to draw
         * @param start the start point of the cylinder
         * @param center the center point of the arc
         * @param angle the angle of the arc
         * @param transform the transformation matrix to place the arc at the correct point
         * @param color the color to draw the arc as
         * @param vertices Vector of vertices to append the new vertices to
         * @param colors Vector of colors to append the new colors to
         * @param normals Vector of normals to append the new normals to
         */
        static void createArcCylinderCCW(float cylinder_height, const Point& start, const Point& center, const Point& end, const QMatrix4x4& transform, const QColor& color, std::vector<float>& vertices, std::vector<float>& colors, std::vector<float>& normals);

        //! \brief Get axis of rotation if we want to rotate vector a to vector b
        static QVector3D getAxis(QVector3D a, QVector3D b);

        //! \brief Find angle to rotate a around axis to align a with b
        static float getAngle(QVector3D a, QVector3D b, QVector3D axis);

        //! adds three vectors to array and computes normal/ colors
        //! \param a the first vector
        //! \param b the second vector
        //! \param c the third vector
        //! \param color the color to draw as
        //! \param vertices Vector of vertices to append the new vertices to
        //! \param colors Vector of colors to append the new colors to
        //! \param normals Vector of normals to append the new normals to
        static void appendTriangle(QVector3D a, QVector3D b, QVector3D c, const QColor& color, std::vector<float>& vertices, std::vector<float>& colors, std::vector<float>& normals);
    };
} //Namespace ORNL
#endif // SHAPE_FACTORY_H
