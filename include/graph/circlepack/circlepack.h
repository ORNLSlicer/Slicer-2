#ifndef CIRCLEPACK_H
#define CIRCLEPACK_H

#include <cmath>
#include <QEnableSharedFromThis>
#include <QMap>
#include <QQueue>
#include <QSharedPointer>
#include <QSet>
#include <QtAlgorithms>
#include <QVector>
#include <cfloat>

#include "../../geometry/path.h"
#include "../../geometry/point.h"
#include "../../geometry/polygon.h"
#include "../../geometry/polygon_list.h"
#include "../face.h"
#include "../faceedgeiterator.h"
#include "../field.h"
#include "../graph.h"
#include "../halfedge.h"
#include "../vertex.h"
#include "../vertexedgeiterator.h"
#include "clipper.hpp"

namespace ORNL
{
    const int64_t POS_INF = 1000000000;
    const int64_t NEG_INF = -1000000000;
    const double SQRT_2 = 1.41421356237309;
    const double SQRT_3 = 1.73205080756888;

    const int64_t DEFAULT_LOAD_SCALE = 1000;
    const int64_t DEFAULT_ITERATIONS = 3000;
    const double DEFAULT_ADJUSTMENT_FACTOR = 0.01;
    const double DEFAULT_MIN_INVERSIVE_DIST = 0.25;
    const double DEFAULT_MAX_INVERSIVE_DIST = 4.00;
    const double DEFAULT_MARGIN_RATIO = 0.1;
    const int64_t DEFAULT_MIN_NEIGHBORS = 2;
    const int64_t DEFAULT_MIN_NEIGHBORS_FOR_SECONDARY = 3;
    const int64_t BASE_BINARY_SEARCH_ITERATIONS = 5;
    const int64_t SEARCHES_PER_MAGNITUDE = 4;
    const int64_t PACKING_SCALE_UP_FACTOR = 1000;
    const int64_t TAB_RADIUS_RATIO = 4;

    enum BoundaryMode
    {
        CENTER, TANGENT
    };
    enum VertexStatus
    {
        UNVISITED = 0, PLACED = -1, PROCESSED = 2
    };
    enum Orientation
    {
        CLOCKWISE = 1,
        COUNTER_CLOCKWISE = -1,
        HORIZONTAL = 2,
        VERTICAL = -2,
        VERTICAL_HEX = -4
    };
    enum BoundaryTabType
    {
        NONE = 0, EDGE_MIDPOINT = 1, CIRCLES_TANGENT = 2
    };


    class CirclePack
    {
    public:
        CirclePack();

        /**
        *\brief generateUniformSquareGrid, generateUniformHexGrid
        *These functions take in as parameters the x and y coordinates of the first vertex in the graph,
        *the length of the edges in the graph, and the number of rows and columns in the graph. For the
        *square version, the output is a uniform square grid with secondary vertices included by default
        *in the interstices. Optionally, passing in false to the “include_secondary” parameter will
        *leave these out. For the hex version, the output is a uniform hexagonal grid composed of equilateral
        *triangles. Optionally, passing in true to the “set_secondary” parameter will set specific vertices
        *in the hex grid as secondary vertices so that all secondary vertices are separated from neighboring
        *secondary vertices by exactly one line of vertices.
        */

        QSharedPointer<Graph> generateUniformSquareGrid(int64_t edge_length, int64_t x_offset,
                int64_t y_offset, int64_t rows, int64_t columns,
                bool include_secondary = true);

        QSharedPointer<Graph> generateUniformHexGrid(int64_t edge_length, int64_t x_offset,
                int64_t y_offset, int64_t rows, int64_t columns,
                bool set_secondary = false);

        /**
        * \brief generateContainedSquareGrid, generateContainedHexGrid
        * These functions generate square and hex grids, respectively, within a boundary specified by the
        * polygons parameter with grid edges of the specified length. The overloaded versions provide the
        * option to directly specify the offsets, rows, and columns for generating the initial grids of
        * which a subset contained within the boundary is returned. The simple versions guarantee that the
        * space within the boundary will be filled as fully as possible, while the overloaded versions do
        * not guarantee this since the user will be directly controlling the generation of the initial grids.
        * Also, in the CirclePack object there is a “boundary_mode” field with possible values TANGENT and
        * CENTER. By default, when a CirclePack object is created, the “boundary_mode” field is set to
        * TANGENT. These modes will be explained later in the section “Fitting Graphs to Boundaries”, but if
        * the “boundary_mode” is set to CENTER, then extra vertices that touch inside of the boundaries (but
        * have centers outside) are also included in the returned graph. The optional parameters “include_secondary”
        * and “set_secondary” are explained in section “Uniform Graphs”.
        */

        QSharedPointer<Graph> generateContainedSquareGrid(PolygonList* polygons,
                int64_t edge_length, bool include_secondary = true);

        QSharedPointer<Graph> generateContainedSquareGrid(PolygonList* polygons,
                int64_t edge_length, int64_t x_offset, int64_t y_offset,
                int64_t rows, int64_t columns, bool include_secondary = true);

        QSharedPointer<Graph> generateContainedHexGrid(PolygonList* polygons,
                int64_t edge_length, bool set_secondary = false);

        QSharedPointer<Graph> generateContainedHexGrid(PolygonList* polygons,
                int64_t edge_length, int64_t x_offset, int64_t y_offset,
                int64_t rows, int64_t columns, bool set_secondary = false);
        /**
        * \brief generateEnvelopingSquareGrid, generateEnvelopingHexGrid
        * These functions generate square and hex grids, respectively, that envelop the boundary
        * specified by the polygons parameter with grid edges of the specified length. The size
        * of the envelop is determined by the margin parameter which specifies the desired
        * distance between the outermost vertices and the boundary. The optional parameters
        * “include_secondary” and “set_secondary” are explained in section “Uniform Graphs”.
        * These functions tend to be the most useful when refinements are desired. This is because
        * refinements do not guarantee that the overall shape of the graph will remain the same
        * after packing, but if only internal vertices that are relatively at a distance from the
        * boundary are refined then the overall shape tends to remain very similar. However, we
        * don’t want a huge margin either because that slows down the packing process. Thus when
        * using these methods, a good margin tends to be around 2-3 times the edge length.
        */

        QSharedPointer<Graph> generateEnvelopingSquareGrid(PolygonList* polygons,
                int64_t edge_length, int64_t margin, bool include_secondary = true);

        QSharedPointer<Graph> generateEnvelopingHexGrid(PolygonList* polygons,
                int64_t edge_length, int64_t margin, bool set_secondary = false);

        /**
        * \brief fitGraphToBoundary
        * This function adjusts the locations and radii of the vertices in the given graph to
        * minimize circle overlaps while also maximizing the conformity of the overall graph shape
        * to the given boundary specified by the polygons. Usage notes: This function tends to
        * work best with graphs that already fit inside the boundary such as those generated by
        * the contained graph generators. This function usually does not work well with graphs
        * that have large hollow spaces internally which are typical of many dual graphs.
        */

        void fitGraphToBoundary(QSharedPointer<Graph> graph, PolygonList* polygons,
                bool pin = true);

        void fitGraphToBoundary(QSharedPointer<Graph> graph, PolygonList* polygons,
                int64_t min_radius, int64_t max_radius, bool pin = true);
        /**
        * \brief FieldPackGraph ??
        *
        *
        *
        *
        */

        void FieldPackGraph(QSharedPointer<Graph> graph, Field* field,
                int64_t min_radius, int64_t max_radius, int64_t radius_delta);

        /**
        * \brief pruneGraph
        * This function continuously removes all isolated vertices, vertices with less than the
        * specified minimum number of neighbors, and secondary vertices with less than the specified
        * minimum number of neighbors until no more vertices are valid for removal. The default
        * numbers are 2 and 3 for regular vertices and secondary vertices, respectively.
        */

        void pruneGraph(QSharedPointer<Graph> graph, int64_t min_neighbors = DEFAULT_MIN_NEIGHBORS,
                int64_t min_neighbors_secondary =
                DEFAULT_MIN_NEIGHBORS_FOR_SECONDARY);
        /**
        * \brief removeInternalSecondary
        * This function removes all non-boundary vertices that are marked as secondary from the given graph.
        */

        void removeInternalSecondary(QSharedPointer<Graph> graph);

        /**
        * \brief removeVertexSelection
        * This function removes from the graph the subset of its vertices given in the removal set.
        * This function is actually extremely versatile as it can remove any selection of vertices
        * from the graph. Also this function maintains a proper DCEL graph, which is a very important
        * property for a large portion of the code in CirclePack.
        */

        void removeVertexSelection(QSharedPointer<Graph> graph, QSet< QSharedPointer<Vertex> > removal_set);

        /**
        * \brief cookieGraphToShape
        * This function removes from the graph all of its vertices that are located outside the boundary
        * specified by the polygons parameter. The function maintains the DCEL structure of the graph.
        */

        void cookieGraphToShape(QSharedPointer<Graph> graph, PolygonList* polygons);

        /**
        * \brief clipGraphToShape
        * This function creates and returns a new graph that is the clipped version of the input graph.
        * More specifically, all edges in the input graph are treated as single line segments that are
        * clipped by the boundary shape specified by the clip parameter and then the resulting clipped
        * line segments are connected together where they share endpoints. The returned graph is a DCEL
        * except for the fact that faces are not set up (i.e. no interior faces and all edges have exterior
        * face as leftFace even if visually they are on the “inside”). The overloaded polygons version
        * just converts the polygons to a Paths object, which is easily done since they are practically
        * the same.
        */

        QSharedPointer<Graph> clipGraphToShape(QSharedPointer<Graph> graph, ClipperLib2::Paths clip);

        QSharedPointer<Graph> clipGraphToShape(QSharedPointer<Graph> graph, PolygonList* polygons);

        /**
        * \brief pack
        * This function takes in a graph and applies a circle packing algorithm to modify all the radii
        * and locations of vertices in the graph so that the resulting graph is a circle packing. The
        * circle placing sub-procedure lays out the circles in clockwise orientation by default, but
        * counterclockwise can be specified. This version uses default values for the iteration limit and
        * tolerance (1000 and 0.0000001 respectively). Important note: This process is a major bottleneck
        * in the execution time by far, since this process can take 1-10 minutes depending on the number
        * of vertices in the graph and the desired accuracy.
        */

        void pack(QSharedPointer<Graph> graph, Orientation orientation = CLOCKWISE);

        /**
        * \brief pack
        * It is possible to speed up the packing process by loosening the tolerance or reducing the iteration
        * limit. If there are less than 1000 vertices and packing radii are in the range [10mm, 100mm] then
        * the tolerance can be reduced to 0.000001 and the results will still be sufficient. Conversely, if
        * more accuracy is desired then the tolerance can be tightened if the user doesn’t mind waiting longer.
        */

        void pack(QSharedPointer<Graph> graph, int64_t iteration_limit, double tolerance,
                Orientation orientation = CLOCKWISE);

        /**
        * \brief getNumOddDegreeVertices
        * Returns the number of vertices with odd degree.
        */
        int getNumOddDegreeVertices(QSharedPointer<Graph> graph);

        /**
        * \brief generatePaths
        * Generates and returns paths for traversing the whole graph with the minimum number of lifting.
        */
        QVector<QVector<QPair<int, int>>> generatePaths(QSharedPointer<Graph> graph);

        /**
        * \brief addBarycenterToFace
        * This function adds a new vertex to the specified face of the given graph. The vertex is placed at
        * the centroid of the face and is given a default radius of half the average of the radii of the corner
        * vertices of the face. In less confusing terms, it looks like it is in the center of the face with an
        * okay radius. The new vertex forms an edge with each corner vertex of the face similar to the spokes on
        * a bicycle wheel. Also the original face is split into N faces where N is the number of edges that
        * formed the original face. This function maintains a proper DCEL.Usage notes: This function is step 1
        * in the refinement process, but I’m providing direct access to this function in case the advanced user
        * may find this function to be of use in a more general scenario.
        */

        QSharedPointer<Vertex> addBarycenterToFace(QSharedPointer<Graph> graph, QSharedPointer<Face> face);

        /**
        * \brief flipEdge
        * This function flips the edge specified by the half edge and its twin. In a triangulation, an interior
        * edge is shared by two triangular faces. To 'flip' the edge, we re-orient the edge so that it forms the
        * other diagonal in the union of those two faces. This process removes nothing and adds nothing to the
        * graph, but simply just rearranges references. Note: Make sure the edge provided is not a boundary edge,
        * since for our purposes, forming an edge through the exterior face is undesirable. Also the edge is
        * flipped counter-clockwise and flipping 4 times returns the graph back the way it was before any flipping
        * was done. Usage notes: This function is step 2 in the refinement process, but I’m providing direct access
        * to this function in case the advanced user may find this function to be of use on its own.
        */

        void flipEdge(QSharedPointer<HalfEdge> h);

        /**
        * \brief eraseBoundaryEdge
        * This function removes the edge specified by the half edge and its twin from the given graph. This function
        * maintains a proper DCEL; however, removal of the two half edges and their associated interior face does
        * leave an “index gap” in the set of keys associated with the half edges and faces in the graph. For example,
        * if a graph had 5 faces with keys 0, 1, 2, 3, and 4 and face 2 is removed, then the set of face keys becomes
        * 0, 1, 3, 4. This may or may not be a problem depending on how the graph is further used. If it is necessary
        * to fix the “index gap” then simply call the remap functions in the Graph class. The three functions are
        * remapVertices(), remapHalfEdges(), and remapFaces(). Call whichever ones you need. Note: This function is
        * step 3 in the refinement process that only occurs when vertices near the graph boundary are refined and the
        * boundary edges are then erased instead of flipped.
        */

        void eraseBoundaryEdge(QSharedPointer<Graph> graph, QSharedPointer<HalfEdge> h);


        void localRefine(QSharedPointer<Graph> graph, QSet< QSharedPointer<Vertex> > vertices);

        /**
        * @brief localRefine
        * This function takes in as parameters the graph to be refined and a boundary specified by the polygons.
        * The input graph is altered so that all the faces around the set of graph vertices that were inside the
        * boundary are refined. Note that the graphs now look terrible. This is why packing, explained in the
        * section “Packing Graphs after Refining”, must be done right after.
        */

        void localRefine(QSharedPointer<Graph> graph, PolygonList* polygons);

        /**
        * \brief localRefine
        * This function takes in as parameters the graph to be refined, a boundary specified by the polygons,
        * and a threshold distance. The input graph is altered so that all the faces around the set of graph
        * vertices that were within threshold distance of the boundary are refined. Note that the graphs now look
        * terrible. This is why packing, explained in the section “Packing Graphs after Refining”, must be done
        * right after.
        */

        void localRefine(QSharedPointer<Graph> graph, PolygonList* polygons,
                int64_t threshold);

        /**
        * \brief generateDualGraph
        * This function creates and returns a new graph that is the dual of the input graph.
        */


        QSharedPointer<Graph> generateDualGraph(QSharedPointer<Graph> graph, BoundaryTabType tab_type = NONE);

        /**
        * \brief pointWithinThresholdOfBoundary
        * It is the bool to check if a point is within threshold of the boundary.
        */

        bool pointWithinThresholdOfBoundary(Point p, int64_t threshold,
                PolygonList* polygons);
        /**
        * \brief selectVerticesInside, selectVerticesOutside, selectVerticesCloseToBoundary, vertexSetUnion
        * \brief vertexSetIntersection, vertexSetDifference
        * These functions provide an easy way to select groups of vertices in the graph.
        * The first three functions return a subset of vertices in the given graph that
        * are inside, outside, or within the specified threshold distance of the boundary
        * specified by the polygons. The last three functions perform set operations on
        * vertex sets. The union operation combines both input sets into one set (through
        * the uniqueness property of sets, there will not be duplicates). The intersection
        * operation returns the set of vertices that were present in both input sets. The
        * difference operation returns the set of vertices in the first input set that
        * were not in the second input set. Usage Notes: The vertex sets obtained from these
        * functions can be used to selectively refine vertices in graphs or selectively
        * delete vertices from graphs among other possible uses.Other ways to select vertices
        * can be manually directly selecting specific vertices from a graph; right now this
        * is not user friendly, but perhaps when a graphical user interface is built for all
        * this stuff then users can point and click on circles in a graphical display of the
        *  graphs to select or deselect vertices.
        */

        QSet< QSharedPointer<Vertex> > selectVerticesInside(QSharedPointer<Graph> graph,
                PolygonList* polygons);

        QSet< QSharedPointer<Vertex> > selectVerticesOutside(QSharedPointer<Graph> graph,
                PolygonList* polygons);

        QSet< QSharedPointer<Vertex> > selectVerticesCloseToBoundary(QSharedPointer<Graph> graph,
                PolygonList* polygons, int64_t threshold);

        QSet< QSharedPointer<Vertex> > vertexSetUnion(QSet< QSharedPointer<Vertex> > set1, QSet< QSharedPointer<Vertex> > set2);

        QSet< QSharedPointer<Vertex> > vertexSetIntersection(QSet< QSharedPointer<Vertex> > set1, QSet< QSharedPointer<Vertex> > set2);

        QSet< QSharedPointer<Vertex> > vertexSetDifference(QSet< QSharedPointer<Vertex> > set1, QSet< QSharedPointer<Vertex> > set2);


        /**
        * \brief translateGraph, scaleGraph, rotateGraph
        * These utility functions provide an easy way to perform scalings, translations, and
        * rotations on graphs.
        */

        void translateGraph(QSharedPointer<Graph> graph, int64_t x_offset, int64_t y_offset);
        void scaleGraph(QSharedPointer<Graph> graph, int64_t scale);
        void rotateGraph(QSharedPointer<Graph> graph, double degrees, Orientation orientation =
                CLOCKWISE);


        int64_t getLoadScale();

        /**
        * \brief getIterations, getAdjustmentFactor
        * Return, the number of iterations and the adjustment factor on the forces applied to circle
        * centers and radii for each iteration are member fields in the CirclePack class.
        */

        int64_t getIterations();
        double getAdjustmentFactor();

        /**
        * \brief getMinInversiveDist, getMaxInversiveDist
        * \return maximum and minimum inverse distance measure throughout the graph.
        */

        double getMinInversiveDist();
        double getMaxInversiveDist();

        /**
        * \brief getBoundaryMode
        * \return Boundary Mode for circle packing . There are two possible
        * output CENTER and TANGENT.
        */
        BoundaryMode getBoundaryMode();


        void setLoadScale(int64_t load_scale);

        /**
        * \brief setIterations, setAdjustmentFactor
        * The procedure involves an iterative process that runs as a simulation of forces
        * between vertices with their neighboring vertices and vertices with the boundary.
        * The number of iterations and the adjustment factor on the forces applied to circle
        * centers and radii for each iteration are member fields in the CirclePack class.
        * By default, the number of iterations is set to 3000 and adjustment factor is set to
        * 0.01.
        */

        void setIterations(int64_t iterations);
        void setAdjustmentFactor(double adjustment_factor);

        /**
        * \brief setMinInversiveDist, setMaxInversiveDist
        * Inversive distances measure and describe the locations of two circles relative to each other.
        * If two circles are exactly tangent and disjoint, then their inversive distance is exactly 1.
        * If two circles are far apart, then their inversive distance is much greater than 1. If two
        * circles are overlapping, then their inversive distance is between -1 and 1. More information
        * about inversive distances can be easily found online (there’s a nice Wikipedia article on it
        * too). Since the objective is to make all the circles of the vertices in the graph tangent and
        * disjoint, we want all the inversive distances to be 1. However, this may not be possible so
        * we relax the constraints on vertices in the graph that are designated as secondary in order
        * to better fit the given boundary. These relaxed constraints are specified by minimum and
        * maximum values for inversive distances to secondary vertices. By default, these member fields
        * in the CirclePack class are set to 0.25 and 4.00 respectively.
        */

        void setMinInversiveDist(double min_inversive_dist);
        void setMaxInversiveDist(double max_inversive_dist);

        /**
        * \brief setBoundaryMode
        * The boundary vertices can be fitted to the boundary such that their circle centers are located
        * on the boundary or their circles are tangent to the boundary. This is specified by a member
        * field within the CirclePack class. By default, the boundary mode is set to TANGENT.
        */
        void setBoundaryMode(BoundaryMode boundary_mode);

    private:
        int64_t load_scale;
        int64_t iterations;
        double adjustment_factor;
        double min_inversive_dist;
        double max_inversive_dist;
        BoundaryMode boundary_mode;

        //void setPackingRadii(QSharedPointer<Graph> graph);

        void setPackingRadii(QSharedPointer<Graph> graph, int64_t iteration_limit,
                double tolerance);

        void layOutCircles(QSharedPointer<Graph> graph, Orientation orientation = CLOCKWISE);

        void calculateForce(QSharedPointer<Graph> graph, QSharedPointer<Vertex> vertex,
                PolygonList* polygons, Point& center_force,
                int64_t& radius_force);

        void calculateForce(QSharedPointer<Graph> graph, QSharedPointer<Vertex> vertex,
                Field* field, int64_t radius_delta,
                Point& center_force, int64_t& radius_force);

        void applyForce(QSharedPointer<Graph> graph, int64_t vertex_key, Point center_force,
                int64_t radius_force);

        void applyForce(QSharedPointer<Graph> graph, int64_t vertex_key, Point center_force,
                int64_t radius_force, int64_t min_radius, int64_t max_radius);
    };

}
#endif
