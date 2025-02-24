/**
 * The packing algorithm optimized by Yifan Wang on 07/20/2018
 */

#include "graph/circlepack/circlepack.h"

namespace ORNL {

CirclePack::CirclePack() {
    // Program internally uses micrometers, input/output may be in millimeters.
    load_scale = DEFAULT_LOAD_SCALE;

    // Fitting to boundary procedure will use these parameters.
    iterations = DEFAULT_ITERATIONS;
    adjustment_factor = DEFAULT_ADJUSTMENT_FACTOR;

    // Allows for a certain degree of circle overlap.
    min_inversive_dist = DEFAULT_MIN_INVERSIVE_DIST;
    max_inversive_dist = DEFAULT_MAX_INVERSIVE_DIST;

    // Set default boundary mode.
    boundary_mode = TANGENT;
}

QSharedPointer<HalfEdge> addHalfEdges(QSharedPointer<Graph> graph, QSharedPointer<Vertex> origin,
                                      QSharedPointer<Vertex> twin_origin, QSharedPointer<Face> face,
                                      QSharedPointer<Face> twin_face, int64_t& edge_key) {
    QSharedPointer<HalfEdge> h(new HalfEdge());
    h->origin(origin);
    origin->incidentEdge(h);
    h->leftFace(face);
    h->next(nullptr);
    h->previous(nullptr);
    h->key(edge_key);
    graph->addEdge(edge_key, h);
    edge_key++;

    QSharedPointer<HalfEdge> h_twin(new HalfEdge());
    h_twin->origin(twin_origin);
    twin_origin->incidentEdge(h_twin);
    h_twin->leftFace(twin_face);
    h->setTwin(h_twin);
    h_twin->next(nullptr);
    h_twin->previous(nullptr);
    h_twin->key(edge_key);
    graph->addEdge(edge_key, h_twin);
    edge_key++;

    return h;
}

QSharedPointer<HalfEdge> addHalfEdges(QSharedPointer<Graph> graph, QSharedPointer<Vertex> origin,
                                      QSharedPointer<Vertex> twin_origin, int64_t& edge_key) {
    return addHalfEdges(graph, origin, twin_origin, graph->exterior(), graph->exterior(), edge_key);
}

/**
 * pointInsidePolygon - Returns true if point p is inside the polygon.
 *     The original version of this code was written by W. Randolph Franklin
 *     a professor at RPI.
 */
bool pointInsidePolygon(Point p, Polygon polygon) {
    bool isInside = false;
    int64_t n = polygon.size();
    for (int64_t i = 0, j = n - 1; i < n; j = i++) {
        Point a = polygon[i];
        Point b = polygon[j];
        if (((a.y() > p.y()) != (b.y() > p.y())) &&
            (p.x() < (b.x() - a.x()) * (p.y() - a.y()) / (b.y() - a.y()) + a.x()))
            isInside = !isInside;
    }
    return isInside;
}

/**
 * pointInsidePolygons - Returns true if point p is inside the polygons. The
 *     first polygon represents the exterior boundary and the remaining polygons
 *     represent holes in the interior.
 */
bool pointInsidePolygons(Point p, PolygonList* polygons) {
    if (polygons->size() < 1)
        return false;

    // Check that it is inside the exterior polygon (the first one).
    if (!pointInsidePolygon(p, (*polygons)[0]))
        return false;

    // Check that it is outside the interior polygons (remaining polygons).
    for (int64_t i = 1; i < polygons->size(); i++)
        if (pointInsidePolygon(p, (*polygons)[i]))
            return false;

    return true;
}

Point closestPointOnSegment(Point a, Point b, Point p) {
    Point projection;
    int64_t dist_squared = static_cast<int64_t>(
        std::pow(a.distance(b)(), 2.0)); //(std::pow(b.x()-a.x(), 2.0) + std::pow(b.y()-a.y(), 2.0));//vSize2(b - a);
    // printf("%f %f %f %f \n", a.x(), a.y(),b.x(), b.y());
    // printf("%I64d \n", dist_squared);
    if (dist_squared == 0) // Point a and Point b are the same point.
    {
        return a; // Closest point to Point p is that point itself.
    }
    Point pa = p - a;
    Point ba = b - a;
    double x = ((pa).dot(ba)) / (double)dist_squared; // dot(p - a, b - a) / (double)dist_squared;

    if (x < 0)
        projection = a;
    else if (x > 1)
        projection = b;
    else
        projection = (b - a) * x + a;

    return projection;
}

/**
 * pointWithinThresholdOfBoundary - Returns true if the given point is within
 *     the threshold distance of the boundary shape specified by the polygons,
 *     and returns false if otherwise.
 */
bool CirclePack::pointWithinThresholdOfBoundary(Point p, int64_t threshold, PolygonList* polygons) {
    for (int64_t i = 0; i < polygons->size(); i++) // Check each polygon.
    {
        int64_t m = (*polygons)[i].size();
        for (int64_t j = 0; j < m; j++) // Test each segment in current polygon.
        {
            Point q = closestPointOnSegment((*polygons)[i][j], (*polygons)[i][(j + 1) % m], p);

            if (static_cast<int64_t>(p.distance(q)()) <= threshold) {
                return true; // Point p is within threshold distance of point q.
            }
        }
    }
    return false;
}

QSet<QSharedPointer<Vertex>> CirclePack::selectVerticesInside(QSharedPointer<Graph> graph, PolygonList* polygons) {
    QSet<QSharedPointer<Vertex>> selection;

    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices(); it++)
        if (pointInsidePolygons(graph->getCircle(it.key())->center(), polygons))
            selection.insert(it.value());

    return selection;
}

QSet<QSharedPointer<Vertex>> CirclePack::selectVerticesOutside(QSharedPointer<Graph> graph, PolygonList* polygons) {
    QSet<QSharedPointer<Vertex>> selection;

    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices(); it++)
        if (!pointInsidePolygons(graph->getCircle(it.key())->center(), polygons))
            selection.insert(it.value());

    return selection;
}

QSet<QSharedPointer<Vertex>> CirclePack::selectVerticesCloseToBoundary(QSharedPointer<Graph> graph,
                                                                       PolygonList* polygons, int64_t threshold) {
    QSet<QSharedPointer<Vertex>> selection;

    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices(); it++) {
        Point center = graph->getCircle(it.key())->center();
        if (pointWithinThresholdOfBoundary(center, threshold, polygons))
            selection.insert(it.value());
    }

    return selection;
}

/**
 * vertexSetUnion - Returns a new set of vertices that is the union of the
 *     two given sets.
 */
QSet<QSharedPointer<Vertex>> CirclePack::vertexSetUnion(QSet<QSharedPointer<Vertex>> set1,
                                                        QSet<QSharedPointer<Vertex>> set2) {
    QSet<QSharedPointer<Vertex>> selection;

    for (QSharedPointer<Vertex> vertex : set1)
        selection.insert(vertex);
    for (QSharedPointer<Vertex> vertex : set2)
        selection.insert(vertex);

    return selection;
}

/**
 * vertexSetIntersection - Returns a new set of vertices that is the
 *     intersection of the two given sets.
 */
QSet<QSharedPointer<Vertex>> CirclePack::vertexSetIntersection(QSet<QSharedPointer<Vertex>> set1,
                                                               QSet<QSharedPointer<Vertex>> set2) {
    QSet<QSharedPointer<Vertex>> selection;

    for (QSharedPointer<Vertex> vertex : set1) // Vertex in set1.
        if (set2.find(vertex) != set2.end())   // Vertex also in set2.
            selection.insert(vertex);

    return selection;
}

/**
 * vertexSetDifference - Returns a new set of vertices that contains the
 *     elements in the first set that are not in the second set.
 */
QSet<QSharedPointer<Vertex>> CirclePack::vertexSetDifference(QSet<QSharedPointer<Vertex>> set1,
                                                             QSet<QSharedPointer<Vertex>> set2) {
    QSet<QSharedPointer<Vertex>> selection;

    for (QSharedPointer<Vertex> vertex : set1) // Vertex in set1.
        if (set2.find(vertex) == set2.end())   // Vertex not in set2.
            selection.insert(vertex);

    return selection;
}

/**
 * pruneGraph - Removes all isolated vertices, vertices with less than the
 *     specified minimum number of neighbors, and also secondary vertices with
 *     less than the specified minimum number of neighbors.
 */
void CirclePack::pruneGraph(QSharedPointer<Graph> graph, int64_t min_neighbors, int64_t min_neighbors_secondary) {
    while (true) // Keep pruning until no more vertices need to be pruned.
    {
        QSet<QSharedPointer<Vertex>> pruning_list;

        for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices();
             it++) {
            if (it.value()->incidentEdge() == nullptr)
                pruning_list.insert(it.value());

            else if (it.value()->outDegrees() < min_neighbors)
                pruning_list.insert(it.value());

            else if (graph->getCircle(it.key())->isSecondary() && it.value()->outDegrees() < min_neighbors_secondary)
                pruning_list.insert(it.value());
        }

        if (pruning_list.empty())
            break; // We're done.

        removeVertexSelection(graph, pruning_list);
    }
}

/**
 * calculateBoundingBoxSettings - Calculate the x and y offsets and the number
 *     of required rows and columns, given the length of edges and margin, for
 *     the bounding box that envelops the given polygons. The x and y offsets
 *     are returned together as a point, while the rows and columns are set by
 *     assigning to the passed by reference rows and columns parameters.
 */
Point calculateBoundingBoxSettings(PolygonList* polygons, int64_t edge_length, int64_t& rows, int64_t& columns,
                                   int64_t margin = NEG_INF) {
    int64_t min_x = POS_INF;
    int64_t min_y = POS_INF;
    int64_t max_x = 0;
    int64_t max_y = 0;

    int64_t size = polygons->size();
    for (int64_t i = 0; i < size; i++) {
        for (Point point : (*polygons)[i]) {
            if (point.x() > max_x)
                max_x = point.x();
            if (point.x() < min_x)
                min_x = point.x();
            if (point.y() > max_y)
                max_y = point.y();
            if (point.y() < min_y)
                min_y = point.y();
        }
    }

    int64_t x_dist = std::abs(max_x - min_x);
    int64_t y_dist = std::abs(max_y - min_y);

    if (margin == NEG_INF) // Then use default margin.
    {
        double factor = 1 + (2 * DEFAULT_MARGIN_RATIO);
        rows = ((y_dist * factor) / edge_length) + 1;
        columns = ((x_dist * factor) / edge_length) + 1;

        int64_t x_offset = min_x - (DEFAULT_MARGIN_RATIO * x_dist);
        int64_t y_offset = min_y - (DEFAULT_MARGIN_RATIO * y_dist);
        return Point(x_offset, y_offset);
    }
    else // Use margin provided.
    {
        rows = ((y_dist + 2 * margin) / edge_length) + 1;
        columns = ((x_dist + 2 * margin) / edge_length) + 1;

        int64_t x_offset = min_x - margin;
        int64_t y_offset = min_y - margin;
        return Point(x_offset, y_offset);
    }
}

/**
 * addDCELRow - Adds a row of n vertices to the graph starting at the given
 *     x and y offsets. Individual vertices are separated from their neighbors
 *     by the distance specified by the edge length parameter. The row of
 *     vertices is a fully set up DCEL graph. By default, the row of vertices
 *     are aligned horizontally, but it can be specified to be vertical instead,
 *     in other words, a column.
 */
QVector<QSharedPointer<Vertex>> addDCELRow(QSharedPointer<Graph> graph, int64_t edge_length, int64_t x_offset,
                                           int64_t y_offset, int64_t n, Orientation orientation = HORIZONTAL) {
    QVector<QSharedPointer<Vertex>> vertices = QVector<QSharedPointer<Vertex>>(n);

    int64_t vertex_key = graph->getNumVertices();
    int64_t edge_key = graph->getNumHalfEdges();
    int64_t radius = edge_length / 2;

    for (int64_t i = 0; i < n; i++) // Create the vertices and edges.
    {
        // Create the vertex.
        vertices[i] = graph->addVertex(vertex_key);

        // Add planar embedding real data to the vertex.
        int64_t x_coord = x_offset;
        int64_t y_coord = y_offset;
        if (orientation == HORIZONTAL) {
            x_coord += i * edge_length;
        }
        else if (orientation == VERTICAL_HEX) {
            if (i % 2 == 1)
                x_coord -= radius;
            y_coord += i * radius * SQRT_3;
        }
        else // orientation == VERTICAL
        {
            y_coord += i * edge_length;
        }
        Point center = Point(x_coord, y_coord);
        QSharedPointer<Circle> circle(new Circle(center, radius));
        graph->setData(vertex_key++, circle);

        // Connect only if there is a previous vertex to connect to.
        if (i > 0)
            addHalfEdges(graph, vertices[i - 1], vertices[i], edge_key);
    }

    if (n > 1) // Then there will be at least 1 edge.
    {
        // Set a boundary edge for the exterior face.
        graph->exterior()->boundary(vertices[0]->incidentEdge());

        // Next of 1st vertex incident's twin == 1st vertex's incident.
        vertices[0]->incidentEdge()->twin()->setNext(vertices[0]->incidentEdge());

        // Set the remaining perimeter connections.
        for (int64_t i = 0; i < n - 1; i++)
            vertices[i]->incidentEdge()->setNext(vertices[i + 1]->incidentEdge());
        for (int64_t i = n - 2; i > 0; i--)
            vertices[i]->incidentEdge()->twin()->setNext(vertices[i - 1]->incidentEdge()->twin());
    }

    return vertices;
}

QSharedPointer<Graph> CirclePack::generateUniformSquareGrid(int64_t edge_length, int64_t x_offset, int64_t y_offset,
                                                            int64_t rows, int64_t columns, bool include_secondary) {
    QSharedPointer<Graph> graph(new Graph());

    // Special case, no vertices - empty graph.
    if (rows <= 0 || columns <= 0)
        return graph;
    if (columns == 1) // Special case, single column graph.
    {
        addDCELRow(graph, edge_length, x_offset, y_offset, rows, VERTICAL);
        return graph;
    }

    QVector<QVector<QSharedPointer<Vertex>>> vertex_array =
        QVector<QVector<QSharedPointer<Vertex>>>(rows, QVector<QSharedPointer<Vertex>>(columns));

    // Build first row as a special case.
    vertex_array[0] = addDCELRow(graph, edge_length, x_offset, y_offset, columns);

    int64_t vertex_key = graph->getNumVertices();
    int64_t edge_key = graph->getNumHalfEdges();
    int64_t face_key = graph->getNumFaces() - 1;
    int64_t radius = edge_length / 2;

    for (int64_t r = 1; r < rows; r++) // Add each consecutive row.
    {
        int64_t current_row_y = r * edge_length + y_offset;

        QSharedPointer<HalfEdge> h = graph->getHalfEdge(vertex_array[r - 1][1], vertex_array[r - 1][0]);

        // Create the left initial vertex of this row.
        vertex_array[r][0] = graph->addVertex(vertex_key);

        // Add planar embedding real data to the vertex.
        Point center = Point(x_offset, current_row_y);
        QSharedPointer<Circle> circle(new Circle(center, radius));
        graph->setData(vertex_key++, circle);

        // Add its associated halfEdges and set up the proper connections.
        addHalfEdges(graph, vertex_array[r][0], vertex_array[r - 1][0], edge_key);
        vertex_array[r][0]->incidentEdge()->setNext(h->next());
        vertex_array[r][0]->incidentEdge()->twin()->setPrevious(h);
        vertex_array[r][0]->incidentEdge()->twin()->setNext(vertex_array[r][0]->incidentEdge());

        for (int64_t c = 1; c < columns; c++) // Add following column vertices.
        {
            // Create the new vertex.
            vertex_array[r][c] = graph->addVertex(vertex_key);

            // Add planar embedding real data to the vertex.
            Point center = Point(c * edge_length + x_offset, current_row_y);
            QSharedPointer<Circle> circle(new Circle(center, radius));
            graph->setData(vertex_key++, circle);

            // Create the new face.
            QSharedPointer<Face> face = graph->addFace(face_key++, h);
            h->leftFace(face);
            h->next()->leftFace(face);

            // Create the 2 edges (4 halfEdges).
            QSharedPointer<HalfEdge> h1 =
                addHalfEdges(graph, vertex_array[r][c - 1], vertex_array[r][c], face, graph->exterior(), edge_key);
            QSharedPointer<HalfEdge> h3 =
                addHalfEdges(graph, vertex_array[r][c], vertex_array[r - 1][c], face, graph->exterior(), edge_key);

            // Set up the edge connections.
            h1->twin()->setNext(h->next()->next());
            h->next()->setNext(h1);
            h1->setNext(h3);
            h3->twin()->setNext(h1->twin());

            h = h->previous(); // Move h over to the right
            h3->setNext(h->next());
            h->setNext(h3->twin());
        }
    }

    if (include_secondary) {
        QSet<QSharedPointer<Face>> interior_faces;
        for (QMap<int, QSharedPointer<Face>>::iterator it = graph->beginFaces(); it != graph->endFaces(); it++)
            if (it.value() != graph->exterior()) // Select all interior faces.
                interior_faces.insert(it.value());

        // Add a secondary vertex/circle to every interior face.
        for (QSharedPointer<Face> face : interior_faces) {
            QSharedPointer<Vertex> secondary_vertex = addBarycenterToFace(graph, face);
            QSharedPointer<Circle> circle = graph->getCircle(secondary_vertex->key());
            // Default radius set in addBarycenterToFace is not right, fix it.
            circle->radius(circle->radius() * 2 * (SQRT_2 - 1.0));
            circle->isSecondary(true);
        }
    }

    return graph;
}

QSharedPointer<Graph> CirclePack::generateContainedSquareGrid(PolygonList* polygons, int64_t edge_length,
                                                              int64_t x_offset, int64_t y_offset, int64_t rows,
                                                              int64_t columns, bool include_secondary) {
    QSharedPointer<Graph> graph =
        generateUniformSquareGrid(edge_length, x_offset, y_offset, rows, columns, include_secondary);

    QSet<QSharedPointer<Vertex>> removal_set = selectVerticesOutside(graph, polygons);

    if (boundary_mode == CENTER) // Include boundary vertices for better fit.
    {
        QSet<QSharedPointer<Vertex>> boundary_set = selectVerticesCloseToBoundary(graph, polygons, edge_length / 2);
        removal_set = vertexSetDifference(removal_set, boundary_set);
    }

    removeVertexSelection(graph, removal_set);

    pruneGraph(graph);

    return graph;
}

QSharedPointer<Graph> CirclePack::generateContainedSquareGrid(PolygonList* polygons, int64_t edge_length,
                                                              bool include_secondary) {
    int64_t rows = 0;
    int64_t columns = 0;
    Point offset = calculateBoundingBoxSettings(polygons, edge_length, rows, columns);

    QSharedPointer<Graph> graph =
        generateUniformSquareGrid(edge_length, offset.x(), offset.y(), rows, columns, include_secondary);

    QSet<QSharedPointer<Vertex>> removal_set = selectVerticesOutside(graph, polygons);

    if (boundary_mode == CENTER) // Include boundary vertices for better fit.
    {
        QSet<QSharedPointer<Vertex>> boundary_set = selectVerticesCloseToBoundary(graph, polygons, edge_length / 2);
        removal_set = vertexSetDifference(removal_set, boundary_set);
    }

    removeVertexSelection(graph, removal_set);

    pruneGraph(graph);

    return graph;
}

QSharedPointer<Graph> CirclePack::generateEnvelopingSquareGrid(PolygonList* polygons, int64_t edge_length,
                                                               int64_t margin, bool include_secondary) {
    int64_t rows = 0;
    int64_t columns = 0;
    Point offset = calculateBoundingBoxSettings(polygons, edge_length, rows, columns, 2 * margin);

    QSharedPointer<Graph> graph =
        generateUniformSquareGrid(edge_length, offset.x(), offset.y(), rows, columns, include_secondary);

    QSet<QSharedPointer<Vertex>> external_set = selectVerticesOutside(graph, polygons);
    QSet<QSharedPointer<Vertex>> margin_set = selectVerticesCloseToBoundary(graph, polygons, margin);
    QSet<QSharedPointer<Vertex>> removal_set = vertexSetDifference(external_set, margin_set);

    removeVertexSelection(graph, removal_set);

    pruneGraph(graph);

    return graph;
}

QSharedPointer<Graph> CirclePack::generateUniformHexGrid(int64_t edge_length, int64_t x_offset, int64_t y_offset,
                                                         int64_t rows, int64_t columns, bool set_secondary) {
    QSharedPointer<Graph> graph(new Graph());

    // Special case, no vertices - empty graph.
    if (rows <= 0 || columns <= 0)
        return graph;
    if (columns == 1) // Special case, single column graph.
    {
        addDCELRow(graph, edge_length, x_offset, y_offset, rows, VERTICAL_HEX);
        return graph;
    }

    QVector<QVector<QSharedPointer<Vertex>>> vertex_array =
        QVector<QVector<QSharedPointer<Vertex>>>(rows, QVector<QSharedPointer<Vertex>>(columns));

    // Build first row as a special case.
    vertex_array[0] = addDCELRow(graph, edge_length, x_offset, y_offset, columns);

    int64_t vertex_key = graph->getNumVertices();
    int64_t edge_key = graph->getNumHalfEdges();
    int64_t face_key = graph->getNumFaces() - 1;
    int64_t radius = edge_length / 2;

    for (int64_t r = 1; r < rows; r++) // Add each consecutive row.
    {
        int64_t current_row_y = (r * radius * SQRT_3) + y_offset;

        if (r % 2 == 1) // Odd row (zero-based, so second row would be odd).
        {
            int64_t row_initial_x = x_offset - radius;

            QSharedPointer<HalfEdge> h = graph->getHalfEdge(vertex_array[r - 1][1], vertex_array[r - 1][0]);

            // Create the left initial vertex of this row.
            vertex_array[r][0] = graph->addVertex(vertex_key);

            // Add planar embedding real data to the vertex.
            Point center = Point(row_initial_x, current_row_y);
            QSharedPointer<Circle> circle(new Circle(center, radius));
            graph->setData(vertex_key++, circle);

            // Add its associated halfEdges and set up the proper connections.
            addHalfEdges(graph, vertex_array[r][0], vertex_array[r - 1][0], edge_key);
            vertex_array[r][0]->incidentEdge()->setNext(h->next());
            vertex_array[r][0]->incidentEdge()->twin()->setPrevious(h);
            vertex_array[r][0]->incidentEdge()->twin()->setNext(vertex_array[r][0]->incidentEdge());

            for (int64_t c = 1; c < columns; c++) // Add consecutive column vertices.
            {
                // Create the new vertex.
                vertex_array[r][c] = graph->addVertex(vertex_key);

                // Add planar embedding real data to the vertex.
                Point center = Point(row_initial_x + c * edge_length, current_row_y);
                QSharedPointer<Circle> circle(new Circle(center, radius));
                graph->setData(vertex_key++, circle);

                // Create the 2 new faces.
                QSharedPointer<Face> f1 = graph->addFace(face_key++, h->next());
                h->next()->leftFace(f1);
                QSharedPointer<Face> f2 = graph->addFace(face_key++, h);
                h->leftFace(f2);

                // Create the 3 edges (6 halfEdges).
                QSharedPointer<HalfEdge> h1 =
                    addHalfEdges(graph, vertex_array[r][c - 1], vertex_array[r][c], f1, graph->exterior(), edge_key);
                QSharedPointer<HalfEdge> h3 =
                    addHalfEdges(graph, vertex_array[r][c], vertex_array[r - 1][c - 1], f1, f2, edge_key);
                QSharedPointer<HalfEdge> h5 =
                    addHalfEdges(graph, vertex_array[r][c], vertex_array[r - 1][c], f2, graph->exterior(), edge_key);

                // Set up the edge connections.
                h1->twin()->setNext(h->next()->next());
                h->next()->setNext(h1);
                h1->setNext(h3);
                h3->setNext(h->next());

                h->setNext(h3->twin());
                h3->twin()->setNext(h5);
                h5->twin()->setNext(h1->twin());

                h = h->previous(); // Move h over to the right
                h5->setNext(h->next());
                h->setNext(h5->twin());
            }
        }
        else // Even row.
        {
            int64_t row_initial_x = x_offset;

            QSharedPointer<HalfEdge> h =
                graph->getHalfEdge(vertex_array[r - 1][columns - 1], vertex_array[r - 1][columns - 2]);

            // Create the right initial vertex of this row.
            vertex_array[r][columns - 1] = graph->addVertex(vertex_key);

            // Add planar embedding real data to the vertex.
            Point center = Point(row_initial_x + (columns - 1) * edge_length, current_row_y);
            QSharedPointer<Circle> circle(new Circle(center, radius));
            graph->setData(vertex_key++, circle);

            // Add its associated halfEdges and set up the proper connections.
            addHalfEdges(graph, vertex_array[r][columns - 1], vertex_array[r - 1][columns - 1], edge_key);
            vertex_array[r][columns - 1]->incidentEdge()->twin()->setPrevious(h->previous());
            vertex_array[r][columns - 1]->incidentEdge()->twin()->setNext(vertex_array[r][columns - 1]->incidentEdge());
            vertex_array[r][columns - 1]->incidentEdge()->setNext(h);

            for (int64_t c = columns - 2; c >= 0; c--) // Add consecutive column vertices.
            {
                // Create the new vertex.
                vertex_array[r][c] = graph->addVertex(vertex_key);

                // Add planar embedding real data to the vertex.
                Point center = Point(row_initial_x + c * edge_length, current_row_y);
                QSharedPointer<Circle> circle(new Circle(center, radius));
                graph->setData(vertex_key++, circle);

                // Create the 2 new faces.
                QSharedPointer<Face> f1 = graph->addFace(face_key++, h->previous());
                h->previous()->leftFace(f1);
                QSharedPointer<Face> f2 = graph->addFace(face_key++, h);
                h->leftFace(f2);

                // Create the 3 edges (6 halfEdges).
                QSharedPointer<HalfEdge> h1 =
                    addHalfEdges(graph, vertex_array[r][c], vertex_array[r][c + 1], f1, graph->exterior(), edge_key);
                QSharedPointer<HalfEdge> h3 =
                    addHalfEdges(graph, vertex_array[r - 1][c + 1], vertex_array[r][c], f1, f2, edge_key);
                QSharedPointer<HalfEdge> h5 =
                    addHalfEdges(graph, vertex_array[r - 1][c], vertex_array[r][c], f2, graph->exterior(), edge_key);

                // Set up the edge connections
                h1->twin()->setPrevious(h->previous()->previous());
                h->previous()->setPrevious(h1);
                h3->setPrevious(h->previous());
                h1->setPrevious(h3);

                h->setPrevious(h3->twin());
                h3->twin()->setPrevious(h5);
                h5->twin()->setPrevious(h1->twin());

                h = h->next(); // Move h over to the left
                h5->setPrevious(h->previous());
                h->setPrevious(h5->twin());
            }
        }
    }

    if (set_secondary) {
        for (int64_t r = 0; r < rows; r++) {
            if (r % 2 == 0) // Even row.
            {
                for (int64_t c = 0; c < columns; c++) {
                    if (c % 3 == 2) // Set every 3rd vertex as secondary.
                    {
                        int64_t key = vertex_array[r][c]->key();
                        graph->getCircle(key)->isSecondary(true);
                    }
                }
            }
            else // Odd row.
            {
                for (int64_t c = 0; c < columns; c++) {
                    if (c % 3 == 1) // Set every 3rd vertex as secondary.
                    {
                        int64_t key = vertex_array[r][c]->key();
                        graph->getCircle(key)->isSecondary(true);
                    }
                }
            }
        }
    }

    return graph;
}

QSharedPointer<Graph> CirclePack::generateContainedHexGrid(PolygonList* polygons, int64_t edge_length, int64_t x_offset,
                                                           int64_t y_offset, int64_t rows, int64_t columns,
                                                           bool set_secondary) {
    QSharedPointer<Graph> graph = generateUniformHexGrid(edge_length, x_offset, y_offset, rows, columns, set_secondary);

    QSet<QSharedPointer<Vertex>> removal_set = selectVerticesOutside(graph, polygons);

    if (boundary_mode == CENTER) // Include boundary vertices for better fit.
    {
        QSet<QSharedPointer<Vertex>> boundary_set = selectVerticesCloseToBoundary(graph, polygons, edge_length / 2);
        removal_set = vertexSetDifference(removal_set, boundary_set);
    }

    removeVertexSelection(graph, removal_set);

    pruneGraph(graph);

    return graph;
}

QSharedPointer<Graph> CirclePack::generateContainedHexGrid(PolygonList* polygons, int64_t edge_length,
                                                           bool set_secondary) {
    int64_t rows = 0;
    int64_t columns = 0;
    Point offset = calculateBoundingBoxSettings(polygons, edge_length, rows, columns);

    QSharedPointer<Graph> graph =
        generateUniformHexGrid(edge_length, offset.x(), offset.y(), rows, columns, set_secondary);

    QSet<QSharedPointer<Vertex>> removal_set = selectVerticesOutside(graph, polygons);

    if (boundary_mode == CENTER) // Include boundary vertices for better fit.
    {
        QSet<QSharedPointer<Vertex>> boundary_set = selectVerticesCloseToBoundary(graph, polygons, edge_length / 2);
        removal_set = vertexSetDifference(removal_set, boundary_set);
    }

    removeVertexSelection(graph, removal_set);

    pruneGraph(graph);

    return graph;
}

QSharedPointer<Graph> CirclePack::generateEnvelopingHexGrid(PolygonList* polygons, int64_t edge_length, int64_t margin,
                                                            bool set_secondary) {
    int64_t rows = 0;
    int64_t columns = 0;
    Point offset = calculateBoundingBoxSettings(polygons, edge_length, rows, columns, 2 * margin);
    // printf("%f %f %d %d\n",offset.x(), offset.y(), rows, columns);

    QSharedPointer<Graph> graph =
        generateUniformHexGrid(edge_length, offset.x(), offset.y(), rows, columns, set_secondary);
    // printf("%d %d %d %d\n", graph->vMap.size(), graph->cMap.size(), graph->eMap.size(), graph->fMap.size());
    // printf("%d\n", selectVerticesOutside(graph, polygons).size());
    QSet<QSharedPointer<Vertex>> external_set = selectVerticesOutside(graph, polygons);
    //    //printf("%d\n", selectVerticesCloseToBoundary(graph, polygons, margin).size());
    QSet<QSharedPointer<Vertex>> margin_set = selectVerticesCloseToBoundary(graph, polygons, margin);
    //    //printf("%d\n", vertexSetDifference(external_set, margin_set).size());
    QSet<QSharedPointer<Vertex>> removal_set = vertexSetDifference(external_set, margin_set);

    //    qDebug()<< removal_set.size()<< "\n" <<endl;
    removeVertexSelection(graph, removal_set);

    pruneGraph(graph);

    // printf("%d %d %d %d\n", graph->vMap.size(), graph->cMap.size(), graph->eMap.size(), graph->fMap.size());

    return graph;
}

void CirclePack::calculateForce(QSharedPointer<Graph> graph, QSharedPointer<Vertex> vertex, PolygonList* polygons,
                                Point& center_force, int64_t& radius_force) {
    Point force = Point(0, 0);
    int64_t total_radius_force = 0;
    int64_t num_forces = 0;
    QSharedPointer<Circle> other_circle; // = nullptr;
    double inv_dist = 0.0;

    QSharedPointer<Circle> this_circle = graph->getCircle(vertex->key());
    Point this_other_circle = Point(0, 0);
    Point this_circle_min = Point(0, 0);

    // Sum up all the forces from neighboring vertices.
    for (QSharedPointer<Vertex> neighbor : vertex->neighbors()) {
        other_circle = graph->getCircle(neighbor->key());
        inv_dist = this_circle->inversiveDistance(other_circle);

        this_other_circle = this_circle->center() - other_circle->center();

        if ((!other_circle->isSecondary() || graph->isBoundary(neighbor)) ||
            (inv_dist < min_inversive_dist || inv_dist > max_inversive_dist)) {
            int64_t dist = static_cast<int64_t>(this_circle->center().distance(other_circle->center())());
            int64_t x = this_circle->radius() + other_circle->radius() - dist;

            force = force + this_other_circle.normal(x);
            total_radius_force -= x;
            num_forces++;
        }
    }

    // If circle is a boundary circle, add extra force towards nearest polygon edge.
    if (graph->isBoundary(vertex)) {
        int64_t min_distance = POS_INF;
        Point min_p = Point(POS_INF, POS_INF);
        Point min_q = Point(POS_INF, POS_INF);
        int64_t poly_id = this_circle->pinPolygonId();
        int64_t vert_id = this_circle->pinVertexId();
        int64_t n = polygons->size();

        if (poly_id < 0 || poly_id >= n || vert_id < 0 || vert_id >= (*polygons)[poly_id].size()) {
            // Find the closest polygon edge.
            for (int64_t i = 0; i < n; i++) {
                Polygon current_polygon = (*polygons)[i];
                int64_t m = current_polygon.size();
                for (int64_t j = 0; j < m; j++) {
                    Point p0 = current_polygon[j];
                    Point p1 = current_polygon[(j + 1) % m];

                    Point p = closestPointOnSegment(p0, p1, this_circle->center());

                    int64_t dist = this_circle->center().distance(p)() - this_circle->radius();

                    if (dist < min_distance) {
                        min_distance = dist;
                        min_p = p;
                    }
                }
            }
        }
        else if (boundary_mode == TANGENT) {
            int64_t m = (*polygons)[poly_id].size();

            Point p0 = (*polygons)[poly_id][(vert_id - 1 + m) % m];
            Point p1 = (*polygons)[poly_id][vert_id];
            Point p2 = (*polygons)[poly_id][(vert_id + 1) % m];

            min_p = closestPointOnSegment(p0, p1, this_circle->center());
            min_q = closestPointOnSegment(p1, p2, this_circle->center());
        }
        else if (boundary_mode == CENTER) {
            min_p = (*polygons)[poly_id][vert_id];
        }

        if (min_p.x() != POS_INF && min_p.y() != POS_INF) {
            int64_t x = -this_circle->center().distance(min_p)();
            if (boundary_mode == TANGENT) {
                x += this_circle->radius();
                radius_force -= x;
                num_forces++;
            }
            this_circle_min = this_circle->center() - min_p;
            force = force + this_circle_min.normal(x);
        }

        if (min_q.x() != POS_INF && min_q.y() != POS_INF) {
            int64_t x = -this_circle->center().distance(min_q)();
            if (boundary_mode == TANGENT) {
                x += this_circle->radius();
                radius_force -= x;
                num_forces++;
            }
            this_circle_min = this_circle->center() - min_q;
            force = force + this_circle_min.normal(x);
        }
    }

    center_force = force;
    radius_force = total_radius_force / num_forces;
}

// With new Field Value of the function

void CirclePack::calculateForce(QSharedPointer<Graph> graph, QSharedPointer<Vertex> vertex, Field* field,
                                int64_t radius_delta, Point& center_force, int64_t& radius_force) {
    Point force = Point(0, 0);
    int64_t total_radius_force = 0;
    int64_t num_forces = 0;
    QSharedPointer<Circle> other_circle = nullptr;
    double inv_dist = 0.0;
    int64_t average_neighbor_radius = 0;

    Point this_other_circle = Point(0, 0);

    QSharedPointer<Circle> this_circle = graph->getCircle(vertex->key());

    // Sum up all the forces from neighboring vertices.
    for (QSharedPointer<Vertex> neighbor : vertex->neighbors()) {
        other_circle = graph->getCircle(neighbor->key());
        average_neighbor_radius += other_circle->radius();
        inv_dist = this_circle->inversiveDistance(other_circle);

        if ((!other_circle->isSecondary() || graph->isBoundary(neighbor)) ||
            (inv_dist < min_inversive_dist || inv_dist > max_inversive_dist)) {
            int64_t dist = static_cast<int64_t>(this_circle->center().distance(other_circle->center())());
            int64_t x = this_circle->radius() + other_circle->radius() - dist;
            this_other_circle = this_circle->center() - other_circle->center();

            force = force + this_other_circle.normal(x);
            total_radius_force -= x;
            num_forces++;
        }
    }
    average_neighbor_radius /= vertex->inDegrees();

    // Add force on radius from field.
    {
        // double field_value = field->getField(this_circle->center);
        // double numerator = field_value - field->getMinField();
        // double denominator = field->getMaxField() - field->getMinField();

        double normalized_field = 2 * (field->getField(graph->getCircle(vertex->key())) - 0.5);
        int64_t field_force = radius_delta * normalized_field;

        // Decrease force because higher field => smaller radius (shrink).
        total_radius_force -= field_force;
        num_forces++;
    }

    { total_radius_force += average_neighbor_radius - this_circle->radius(); }

    if (static_cast<int64_t>(force.distance()()) > this_circle->radius()) {
        center_force = force.normal(this_circle->radius());
        total_radius_force -= this_circle->radius() * 0.2;
    }
    else {
        center_force = force;
    }
    radius_force = total_radius_force / num_forces;
}

/**
 * applyForce - Applies the given forces to the specified vertex in the graph.
 */
void CirclePack::applyForce(QSharedPointer<Graph> graph, int64_t vertex_key, Point center_force, int64_t radius_force) {
    QSharedPointer<Circle> this_circle = graph->getCircle(vertex_key);
    Point scaled_center_force = center_force / (1.0 / adjustment_factor);
    int64_t scaled_radius_force = radius_force / (1.0 / adjustment_factor);
    this_circle->center(this_circle->center() + scaled_center_force);
    this_circle->radius(this_circle->radius() + scaled_radius_force);
}

/**
 * applyForce - Applies the given forces to the specified vertex in the graph.
 *     This overloaded version bounds the radius within the range specified by
 *     the minimum radius and the maximum radius parameters.
 */
void CirclePack::applyForce(QSharedPointer<Graph> graph, int64_t vertex_key, Point center_force, int64_t radius_force,
                            int64_t min_radius, int64_t max_radius) {
    QSharedPointer<Circle> this_circle = graph->getCircle(vertex_key);
    Point scaled_center_force = center_force / (1.0 / adjustment_factor);
    int64_t scaled_radius_force = radius_force / (1.0 / adjustment_factor);
    this_circle->center(this_circle->center() + scaled_center_force);
    int64_t new_radius = this_circle->radius() + scaled_radius_force;

    if (new_radius < min_radius)
        this_circle->radius(min_radius);
    else if (new_radius > max_radius)
        this_circle->radius(max_radius);
    else
        this_circle->radius(new_radius);
}

void pinCirclesToPolygonVertices(QSharedPointer<Graph> graph, PolygonList* polygons) {
    for (int64_t i = 0; i < polygons->size(); i++) // For each polygon.
    {
        Polygon current_polygon = (*polygons)[i];

        // For each vertex in the current polygon, pin the closest circle to it.
        for (int64_t j = 0; j < current_polygon.size(); j++) {
            Point point = current_polygon[j];
            QSharedPointer<Circle> closest_circle; // = nullptr;
            int64_t min_distance = POS_INF;

            // Find the closest circle to this polygon vertex.
            for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices();
                 it++) {
                QSharedPointer<Circle> current_circle = it.value()->data<VertexLocationCircleData>()->m_circle;
                if (graph->containsVertex(it.key())) {
                    if (graph->isBoundary(graph->getVertex(it.key()))) {
                        int64_t dist = current_circle->center().distance(point)();
                        if (dist < min_distance) {
                            min_distance = dist;
                            closest_circle = current_circle;
                        }
                    }
                }
            }

            // If there exists a closest circle, pin it to the vertex.
            if (closest_circle != nullptr) {
                closest_circle->pinPolygonId(i);
                closest_circle->pinVertexId(j);
            }
        }
    }
}

void CirclePack::fitGraphToBoundary(QSharedPointer<Graph> graph, PolygonList* polygons, bool pin) {
    if (pin == true)
        pinCirclesToPolygonVertices(graph, polygons);

    for (int64_t i = 0; i < iterations; i++) {
        for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices();
             it++) {
            Point center_force;
            int64_t radius_force = 0;
            calculateForce(graph, it.value(), polygons, center_force, radius_force);
            applyForce(graph, it.key(), center_force, radius_force);
        }
    }
}

void CirclePack::fitGraphToBoundary(QSharedPointer<Graph> graph, PolygonList* polygons, int64_t min_radius,
                                    int64_t max_radius, bool pin) {
    if (pin == true)
        pinCirclesToPolygonVertices(graph, polygons);

    for (int64_t i = 0; i < iterations; i++) {
        for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices();
             it++) {
            Point center_force;
            int64_t radius_force = 0;
            calculateForce(graph, it.value(), polygons, center_force, radius_force);
            applyForce(graph, it.key(), center_force, radius_force, min_radius, max_radius);
        }
    }
}

void CirclePack::FieldPackGraph(QSharedPointer<Graph> graph, Field* field, int64_t min_radius, int64_t max_radius,
                                int64_t radius_delta) {
    for (int64_t i = 0; i < iterations; i++) {
        for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices();
             it++) {
            Point center_force;
            int64_t radius_force = 0;
            calculateForce(graph, it.value(), field, radius_delta, center_force, radius_force);
            applyForce(graph, it.key(), center_force, radius_force, min_radius, max_radius);
        }
    }
}

/**
 * angleSum - Calculates the angle sum at the vertex specified by the
 *     given key. The angle sum is defined as the total angle that is
 *     covered by the vertex's neighbors if each of their respective
 *     circles were laid tangent to the circle of the vertex and to the
 *     circles of other neighbors around the vertex.
 */
double angleSum(QSharedPointer<Graph> graph, int64_t vertex_key) {
    double angle_sum = 0;

    // int64_t r = graph->cMap(vertex_key)->radius;
    int64_t r = graph->getVertex(vertex_key)->data<VertexLocationCircleData>()->m_circle->radius();
    QSharedPointer<HalfEdge> start = graph->getVertex(vertex_key)->incidentEdge();

    int64_t neighbor_key = start->previous()->origin()->key();
    // int64_t r1 = graph->cMap(neighbor_key)->radius;
    int64_t r1 = graph->getVertex(neighbor_key)->data<VertexLocationCircleData>()->m_circle->radius();
    double a = double(r1) / (r + r1);

    QSharedPointer<HalfEdge> current = start;
    do // Go through each of the neighbors.
    {
        neighbor_key = current->twin()->origin()->key();

        // int64_t r2 = graph->cMap(neighbor_key)->radius;
        int64_t r2 = graph->getVertex(neighbor_key)->data<VertexLocationCircleData>()->m_circle->radius();
        double b = double(r2) / (r + r2);
        angle_sum += std::asin(std::sqrt(a * b));
        a = b;

        current = current->twin()->next();
    } while (current != start);

    // Double the angle sum to complete the computation before returning it.
    return 2 * angle_sum;
}

/*
void adjustVertexRadius(QSharedPointer<Graph> graph, int64_t vertex_key, double angle_sum,
        double target_angle_sum)
{
    //int64_t current_radius = graph->cMap(vertex_key)->radius;
    int64_t current_radius = graph->getVertex(vertex_key)->data<VertexLocationCircleData>()->m_circle->radius();
    int64_t upper_bound = current_radius * 2;
    int64_t lower_bound = current_radius / 2;

    if (angle_sum < target_angle_sum) // Angle sum too small, radius too big.
    {
        //graph->cMap(vertex_key)->radius = lower_bound;
        graph->getVertex(vertex_key)->data<VertexLocationCircleData>()->m_circle->radius(lower_bound);
        // Check if we stepped past target.
        if (angleSum(graph, vertex_key) <= target_angle_sum)
            return; // The new value for radius is fine, so return.
        else
            upper_bound = current_radius;
    }
    else // Angle sum too big, radius too small.
    {
        //graph->cMap(vertex_key)->radius = upper_bound;
        graph->getVertex(vertex_key)->data<VertexLocationCircleData>()->m_circle->radius(upper_bound);
        // Check if we stepped past target.
        if (angleSum(graph, vertex_key) >= target_angle_sum)
            return; // The new value for radius is fine, so return.
        else
            lower_bound = current_radius;
    }

    int64_t binary_search_iterations = BASE_BINARY_SEARCH_ITERATIONS;
    // Increase the number of iterations as the error becomes really small.
    double error = std::abs(angle_sum - target_angle_sum);
    if (error < 1.0)
    {
        binary_search_iterations += SEARCHES_PER_MAGNITUDE * std::abs(std::log10(error));
    }

    // New value stepped past the target, search for a value within range.
    for (int64_t i = 0; i < binary_search_iterations; i++)
    {
        int64_t mid = (lower_bound + upper_bound) / 2;
        //graph->cMap(vertex_key)->radius = mid;
        graph->getVertex(vertex_key)->data<VertexLocationCircleData>()->m_circle->radius(mid);
        if (angleSum(graph, vertex_key) > target_angle_sum)
            lower_bound = mid; // Radius too small, increase the lower bound.
        else
            upper_bound = mid; // Radius too big, decrease the upper bound.
    }
}

void CirclePack::setPackingRadii(QSharedPointer<Graph> graph, int64_t iteration_limit,
        double tolerance)
{
    // To prevent nan errors, check that all radii have been initialized.
    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices(); it++)
    {
        if (it.value()->data<VertexLocationCircleData>()->m_circle->radius() <= 0)
        {
            qDebug()<< "Error: A vertex with radius <= 0 was detected. ";
            qDebug()<< "All radii must be initialized to some positive number.\n";
            return;
        }
    }

    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices(); it++) //
Scale up for better integer precision.
    {
        //it.value()->radius *= PACKING_SCALE_UP_FACTOR;
        int64_t radius= it.value()->data<VertexLocationCircleData>()->m_circle->radius();
        it.value()->data<VertexLocationCircleData>()->m_circle->radius( radius * PACKING_SCALE_UP_FACTOR);
    }

    // Prevent infinite loops in case desired accuracy can't be reached.
    int64_t iteration_count = 0;

    double average_error = tolerance;
    double target_angle_sum = 2 * M_PI;

    // Store list of internal vertices' keys.
    QVector<int64_t> internal_v_keys;
    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices(); it++)
        if (!graph->isBoundary(it.value()))
            internal_v_keys.push_back(it.key());

    int64_t n = internal_v_keys.size(); // n = number of internal vertices.

    do // Adjust radii of internal vertices until packing criteria is met.
    {
        iteration_count++;

        double total_error = 0;
        for (int64_t i = 0; i < n; i++) // Adjust each vertex's radius.
        {
            double angle_sum = angleSum(graph, internal_v_keys[i]);
            double error_of_current_v = std::abs(angle_sum - target_angle_sum);
            total_error += error_of_current_v;

            if (error_of_current_v > average_error)
            {
                adjustVertexRadius(graph, internal_v_keys[i], angle_sum,
                        target_angle_sum);
            }
        }
        average_error = total_error / n;
        // Cut average error down to a third to ensure vertices w/ close to
        // average error are still adjusted.
        //cerr << "Average error = " << average_error << endl;
        average_error /= 3;
    }
    while (average_error > tolerance && iteration_count < iteration_limit);
    //cerr << "Iteration count = " << iteration_count << endl;

    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices(); it++) //
Undo the scale up.
    {
        int64_t radius = it.value()->data<VertexLocationCircleData>()->m_circle->radius();
        it.value()->data<VertexLocationCircleData>()->m_circle->radius(radius/PACKING_SCALE_UP_FACTOR);
        //it.value()->radius /= PACKING_SCALE_UP_FACTOR;
    }
}

void CirclePack::setPackingRadii(QSharedPointer<Graph> graph)
{
    // Call the actual working method with default settings.
    int64_t iteration_limit = 1000;
    double tolerance = 0.0000001;
    setPackingRadii(graph, iteration_limit, tolerance);
}
*/

/**
 * findThirdCenter - Given two circles with known centers and radii and the
 *     radius of a third circle, this returns the center of the third circle if
 *     the third circle was laid tangent to the other two in the specified
 *     orientation going from circle1 to circle2 to the third circle.
 *     Note: The two circles must be mutually disjoint and tangent. Also due to
 *     usage of integer points, results may be inaccurate if the x's, y's, and
 *     radii input to this method are less than 100, but this should not be an
 *     issue as the program uses microns internally and 1mm = 1000 microns.
 */
Point findThirdCenter(QSharedPointer<Circle> circle1, QSharedPointer<Circle> circle2, int64_t radius,
                      Orientation orientation) {
    Point A = circle1->center();
    Point B = circle2->center();
    Point C = Point(0, 0); // Initialize the center of the third circle.
    int64_t rA = circle1->radius();
    int64_t rB = circle2->radius();

    if (A == B) // Perform error checks.
    {
        qDebug() << "The two given circles should not have the same center.\n";
        return C;
    }
    if (rA <= 0 || rB <= 0) {
        qDebug() << "The two given circles should have positive nonzero radii.\n";
        return C;
    }

    Point vectAB = B - A; // Compute vector going from point A to point B.

    // Find the angle between the two edges CA and AB in the triangle ABC.
    double a = rB + radius;
    double b = rA + radius;
    double c = rA + rB;
    double theta = std::acos((b * b + c * c - a * a) / (2 * b * c)); // Law of Cosines

    // Rotate vector AB by theta around the origin to get vector AC.
    int64_t x = vectAB.x();
    int64_t y = vectAB.y();
    Point vectAC;
    if (orientation == CLOCKWISE) // Rotate clockwise.
        vectAC = Point(x * std::cos(theta) + y * std::sin(theta), y * std::cos(theta) - x * std::sin(theta));
    else // Rotate counter-clockwise.
        vectAC = Point(x * std::cos(theta) - y * std::sin(theta), x * std::sin(theta) + y * std::cos(theta));

    // Re-scale vector AC to its proper length.
    int64_t magnitude = (int64_t)std::sqrt(vectAC.x() * vectAC.x() + vectAC.y() * vectAC.y());
    int64_t proper_length = b;
    vectAC = vectAC * proper_length / magnitude;

    C = A + vectAC; // Calculate C by adding vector AC to A.

    return C;
}

QSharedPointer<HalfEdge> getAlphaHalfEdge(QSharedPointer<Graph> graph) {
    QSharedPointer<HalfEdge> alpha; // = nullptr;

    /**
    for (auto const &it : graph->eMap)
    {
        if (!graph->isBoundary(it.second->origin))
        {
            alpha = it.second;
            break;
        }
    }
    */

    if (alpha == nullptr) // Provide a backup in case one can't be found.
    {
        for (QMap<int, QSharedPointer<HalfEdge>>::iterator it = graph->beginEdges(); it != graph->endEdges();
             it++) // Find a non-boundary halfEdge.
        {
            if (it.value()->leftFace() != graph->exterior()) {
                alpha = it.value();
                break;
            }
        }
    }

    return alpha;
}

QVector<QSharedPointer<Face>> getFaceOrder(QSharedPointer<Graph> graph) {
    QVector<QSharedPointer<Face>> face_order;
    QMap<int64_t, QSharedPointer<HalfEdge>> incident_edge;
    QMap<int64_t, VertexStatus> vertex_status;

    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices();
         it++) // Mark all vertices as unvisited.
    {
        vertex_status[it.key()] = UNVISITED;
    }

    QSharedPointer<HalfEdge> alpha = getAlphaHalfEdge(graph); // Get a starting edge.
    if (alpha == nullptr) {
        qDebug() << "No non-boundary halfEdges were found!" << endl;
        return face_order; // Return empty list of faces.
    }

    // Add alpha's leftFace as the first face.
    face_order.push_back(alpha->leftFace());
    alpha->leftFace()->boundary(alpha); // <- This assignment is essential!

    incident_edge[alpha->origin()->key()] = alpha;
    incident_edge[alpha->target()->key()] = alpha->next();
    incident_edge[alpha->next()->target()->key()] = alpha->next()->next();
    vertex_status[alpha->origin()->key()] = PLACED;
    vertex_status[alpha->target()->key()] = PLACED;
    vertex_status[alpha->next()->target()->key()] = PLACED;

    QQueue<int64_t> queue; // List of keys of vertices to be processed.
    queue.enqueue(alpha->origin()->key());
    queue.enqueue(alpha->target()->key());
    queue.enqueue(alpha->next()->target()->key());
    int64_t vertex_count = 3; // Number of vertices placed.

    while (!queue.empty()) // This is basically a Breadth-First-Search.
    {
        int64_t vertex_key = queue.front();
        queue.dequeue();

        if (vertex_status[vertex_key] == PLACED) {
            QSharedPointer<HalfEdge> start = incident_edge[vertex_key];

            QSharedPointer<HalfEdge> current = start->previous()->twin();
            while (true) // Visit each out edge in counter-clockwise order.
            {
                if (current == start)
                    break;

                if (current->leftFace() != graph->exterior() && vertex_status[current->target()->key()] != UNVISITED &&
                    vertex_status[current->next()->target()->key()] == UNVISITED) {
                    // If all conditions met, then we found a new vertex.
                    int64_t v = current->next()->target()->key();
                    vertex_status[v] = PLACED;
                    queue.enqueue(v);
                    vertex_count++;

                    // Identify an edge with the new vertex.
                    incident_edge[v] = current->next()->next();

                    // Store this face in the layout list.
                    face_order.push_back(current->leftFace());
                    current->leftFace()->boundary(current); // Essential step!
                }

                current = current->previous()->twin();
            }

            current = start->twin();
            while (true) // Visit each in edge in clockwise order.
            {
                if (current->leftFace() != graph->exterior() && vertex_status[current->origin()->key()] != UNVISITED &&
                    vertex_status[current->next()->target()->key()] == UNVISITED) {
                    // If all conditions met, then we found a new vertex.
                    int64_t v = current->next()->target()->key();
                    vertex_status[v] = PLACED;
                    queue.enqueue(v);
                    vertex_count++;

                    // Identify an edge with the new vertex.
                    incident_edge[v] = current->next()->next();

                    // Store this face in the layout list.
                    face_order.push_back(current->leftFace());
                    current->leftFace()->boundary(current); // Essential step!
                }

                current = current->next()->twin();
                if (current->twin() == start)
                    break;
            }

            vertex_status[vertex_key] = PROCESSED;
        }
    }

    int64_t num_vertices = graph->getNumVertices();
    if (vertex_count != num_vertices) // Error check.
    {
        qDebug() << "Missed some vertex?" << endl;
        qDebug() << "Visited " << vertex_count << "/" << num_vertices << endl;
    }

    return face_order;
}

/**
 * setSingleTangencyCenter - Sets the center for the specified vertex in the
 *     graph. The vertex has only a single tangency to another vertex, so it
 *     can technically be set anywhere on a circle around the other vertex that
 *     it is attached to. However, we do want to avoid overlapping this vertex's
 *     circle with any other vertex's circle if at all possible. Thus, we set
 *     the vertex between the previous and next circles of the other vertex such
 *     that the orientation going from previous->this vertex->next is clockwise.
 */
void setSingleTangencyCenter(QSharedPointer<Graph> graph, int64_t vertex_key) {
    QSharedPointer<HalfEdge> h = graph->getVertex(vertex_key)->incidentEdge();
    QSharedPointer<Circle> vertex = graph->getVertex(vertex_key)->data<VertexLocationCircleData>()->m_circle;
    QSharedPointer<Circle> other = graph->getVertex(h->target()->key())->data<VertexLocationCircleData>()->m_circle;
    QSharedPointer<Circle> previous =
        graph->getVertex(h->twin()->previous()->origin()->key())->data<VertexLocationCircleData>()->m_circle;
    QSharedPointer<Circle> next =
        graph->getVertex(h->next()->target()->key())->data<VertexLocationCircleData>()->m_circle;

    Point vectOP = previous->center() - other->center();
    Point vectON = next->center() - other->center();
    Point vectOV = vectOP + vectON;

    // Determine if vectOV needs to go in the opposite direction.
    double dot_product = (vectOP.x() * vectON.x()) + (vectOP.y() * vectON.y());
    double determinant = (vectOP.x() * vectON.y()) - (vectOP.y() * vectON.x());
    double theta = atan2(determinant, dot_product); // theta in range (-pi, pi].
    if (theta < 0) {
        vectOV = vectOV * -1;
    }

    // Re-scale to proper length.
    int64_t magnitude = vectOV.distance()();
    int64_t proper_length = other->radius() + vertex->radius();
    vectOV = vectOV * proper_length / magnitude;

    vertex->center(other->center() + vectOV); // Set the vertex center.
}

void CirclePack::layOutCircles(QSharedPointer<Graph> graph, Orientation orientation) {
    QVector<QSharedPointer<Face>> face_order = getFaceOrder(graph);

    if (face_order.size() == 0)
        return; // No faces to lay out!

    QMap<int64_t, bool> placed;
    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices();
         it++) // Mark all vertices as not yet placed.
        placed[it.key()] = false;

    // Lay out the first face.
    int64_t v = face_order[0]->boundary()->origin()->key();
    int64_t u = face_order[0]->boundary()->target()->key();
    int64_t w = face_order[0]->boundary()->next()->target()->key();

    // Keep the 1st circle where it is.
    QSharedPointer<Circle> c1 = graph->getVertex(v)->data<VertexLocationCircleData>()->m_circle;
    placed[v] = true;

    // Lay the 2nd circle tangent to the 1st circle preserving direction.
    QSharedPointer<Circle> c2 = graph->getVertex(u)->data<VertexLocationCircleData>()->m_circle;
    Point vectC1C2 = c2->center() - c1->center();
    int64_t magnitude = vectC1C2.distance()();
    vectC1C2 = vectC1C2 * (c1->radius() + c2->radius()) / magnitude;
    c2->center(c1->center() + vectC1C2);
    placed[u] = true;

    // Lay the 3rd circle tangent to the first two.
    QSharedPointer<Circle> c3 = graph->getVertex(w)->data<VertexLocationCircleData>()->m_circle;
    c3->center(findThirdCenter(c1, c2, c3->radius(), orientation));

    for (QSharedPointer<Face> face : face_order) // Now lay out the rest of the faces.
    {
        v = face->boundary()->origin()->key();
        u = face->boundary()->target()->key();
        w = face->boundary()->next()->target()->key();

        c1 = graph->getVertex(v)->data<VertexLocationCircleData>()->m_circle;
        c2 = graph->getVertex(u)->data<VertexLocationCircleData>()->m_circle;
        c3 = graph->getVertex(w)->data<VertexLocationCircleData>()->m_circle;

        // The main work is done here in setting the center of the 3rd circle.
        c3->center(findThirdCenter(c1, c2, c3->radius(), orientation));
        placed[w] = true;
    }

    // Handle unplaced vertices (they were not a corner of any internal face).
    for (QMap<int64_t, bool>::iterator it = placed.begin(); it != placed.end(); it++) {
        if (it.value() == false) // This vertex was not placed.
        {
            setSingleTangencyCenter(graph, it.key()); // Place it.
        }
    }
}

/**
 *  adjustVertexRadius - Directly modifies the vertex radius in the graph to
 *                       better fit the target angle sum.
 *
 *  Created by Yifan Wang on 07/20/2018.
 *
 *  Reference: Charles R. Collins, Kenneth Stephenson, "A circle packing algorithm", Computational Geometry, Volume 25,
 * Issue 3, 2003
 */
void adjustVertexRadius(QSharedPointer<Graph> graph, int64_t vertex_key, double angle_sum, double target_angle_sum) {
    // double angle_sum = angleSum(graph, vertex_key);
    int64_t k = graph->getVertex(vertex_key)->neighbors().size();
    double beta = std::sin(angle_sum / (2 * k));
    double delta = std::sin(target_angle_sum / (2 * k));
    double r_hat = beta / (1 - beta) * graph->getCircle(vertex_key)->radius();
    double r_new = (1 - delta) / delta * r_hat;
    graph->getCircle(vertex_key)->radius(r_new);
}

/**
 *  setPackingRadii - Sets radii of all circles in the graph.
 *
 *  Created by Yifan Wang on 07/20/2018.
 *
 *  Reference: Charles R. Collins, Kenneth Stephenson, "A circle packing algorithm", Computational Geometry, Volume 25,
 * Issue 3, 2003
 */
void CirclePack::setPackingRadii(QSharedPointer<Graph> graph, int64_t iteration_limit, double tolerance) {
    // To prevent nan errors, check that all radii have been initialized.
    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices(); it++) {
        if (graph->getCircle(it.key())->radius() <= 0) {
            qDebug() << "Error: A vertex with radius <= 0 was detected. ";
            qDebug() << "All radii must be initialized to some positive number.\n";
            return;
        }
    }

    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices();
         it++) // Scale up for better integer precision.
    {
        graph->getCircle(it.key())->radius(graph->getCircle(it.key())->radius() * PACKING_SCALE_UP_FACTOR);
    }

    // Prevent infinite loops in case desired accuracy can't be reached.
    int64_t iteration_count = 0;

    double c = tolerance + 0.01;
    double c0 = c + 1;

    double target_angle_sum = 2 * M_PI;

    // Store list of internal vertices' keys.
    QVector<int64_t> internal_v_keys;
    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices(); it++)
        if (!graph->isBoundary(it.value()))
            internal_v_keys.push_back(it.key());

    int64_t n = internal_v_keys.size(); // n = number of internal vertices.

    while ((std::abs(c - c0) > tolerance && c > tolerance) && iteration_count <= iteration_limit) {
        iteration_count++;
        c0 = c;
        c = 0.0;

        for (int64_t i = 0; i < n; i++) // Adjust each vertex's radius.
        {
            double angle_sum = angleSum(graph, internal_v_keys[i]);
            adjustVertexRadius(graph, internal_v_keys[i], angle_sum, target_angle_sum);
            double error_of_current_v = angle_sum - target_angle_sum;
            c += error_of_current_v * error_of_current_v;
        }
        c = std::sqrt(c);
    }

    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices();
         it++) // Undo the scale up.
    {
        graph->getCircle(it.key())->radius(graph->getCircle(it.key())->radius() / PACKING_SCALE_UP_FACTOR);
    }
}

/**
 *  pack - Given a triangulation in the form of a graph along with defined
 *     boundary conditions and set radii for boundary vertices, this method
 *     modifies the given graph into a circle packing that is unique to the
 *     provided input combinatorics and boundary settings up to planar
 *     rotations and translations. The optional orientation parameter specifies
 *     whether the circles should be laid out clockwise or counterclockwise. By
 *     default it is clockwise.
 *
 *  Created by Yifan Wang on 07/20/2018.
 */
void CirclePack::pack(QSharedPointer<Graph> graph, Orientation orientation) {
    int64_t iteration_limit = 1000;
    double tolerance = 0.0000001;
    setPackingRadii(graph, iteration_limit, tolerance);
    layOutCircles(graph, orientation);
}

/**
 *  pack - This overloaded version also takes in the iteration limit and
 *     tolerance that it uses when setting the packing radii.
 *
 *  Created by Yifan Wang on 07/20/2018.
 */
void CirclePack::pack(QSharedPointer<Graph> graph, int64_t iteration_limit, double tolerance, Orientation orientation) {
    setPackingRadii(graph, iteration_limit, tolerance);
    layOutCircles(graph, orientation);
}

/**
 * getNumOddDegreeVertices - Returns the number of vertices with odd degree.
 *
 * Created by Yifan Wang on 08/01/2018.
 */
int CirclePack::getNumOddDegreeVertices(QSharedPointer<Graph> graph) {
    int n = graph->getNumVertices();
    int count = 0;
    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices(); it++) {
        if (graph->getVertex(it.key())->neighbors().size() % 2 == 1) {
            count++;
        }
    }
    return count;
}

/**
 * getNumAvailableDirectionsFromCurrentVertex - Returns the number of available directions
 *     we can go for the next step. If an edge has been visited, it would not be available
 *     for the next step since we should not visit the same edge multi-times.
 *     In short, this function returns the number of unvisited edges starting from specific vertex.
 *
 * Created by Yifan Wang on 08/01/2018.
 */
int getNumAvailableDirectionsFromCurrentVertex(QSharedPointer<Graph> graph, int cur_key, QVector<int>& visited_edges,
                                               QVector<int> boundary) {
    QSharedPointer<Vertex> cur_vertex = graph->getVertex(cur_key);
    int count = 0;
    for (QSharedPointer<HalfEdge> direction : cur_vertex->outEdges()) {
        if (visited_edges.indexOf(direction->key()) == -1) // direction is not in visited_edges
        {
            count++;
        }
    }
    return count;
}

/**
 * getClosestBoundryVertex - Returns the vertex on boundary which is closest to specific vertex.
 *
 * Created by Yifan Wang on 08/01/2018.
 */
QPair<int, Distance> getClosestBoundryVertex(QSharedPointer<Graph> graph, int v_key, QVector<int> boundary) {
    Distance min_distance(FLT_MAX);
    int closest_u_key = -1;
    for (int u_key : boundary) {
        Distance distance = graph->getCircle(v_key)->center().distance(graph->getCircle(u_key)->center());
        if (distance < min_distance) {
            min_distance = distance;
            closest_u_key = u_key;
        }
    }
    return QPair<int, Distance>(closest_u_key, min_distance);
}

/**
 * getVertexClosestToBoundary - In vertex key list "vertices",
 *      selects and returns the vertex that is closest to boundary.
 *      The chosen vertex may be on boundary or off boundary.
 *      If a vertex with degree 2 or more is on boundary, it can be chosen;
 *      if a vertex with degree 1 is on bouundary, it cannot be chosen.
 *      In short, here we only select the vertex cloest to boundary out of those
 *      which are in list "vertices" and not dead-end.
 *
 * Created by Yifan Wang on 08/01/2018.
 */
int getVertexClosestToBoundary(QSharedPointer<Graph> graph, QVector<int> vertices, QVector<int>& visited_edges,
                               QVector<int> boundary) {
    Distance min_distance(FLT_MAX);
    int closest_vertex = -1;
    for (int cur_v_key : vertices) {
        QPair<int, Distance> closest_boundary_v_dist = getClosestBoundryVertex(graph, cur_v_key, boundary);
        int boundary_v_key = closest_boundary_v_dist.first;
        Distance min_dist_to_boundary = closest_boundary_v_dist.second;
        if (min_dist_to_boundary <
            min_distance) { // If a point having only one out edge is the closest point to boundary,
                            // it is on the boundary, and we will not choose it as the next point.
            min_distance = min_dist_to_boundary;
            closest_vertex = cur_v_key;
        }
    }
    return closest_vertex;
}

/**
 * getNextVertexOnPath - Calculates and returns the next vertex we should visit
 *     on current traversal path.
 *
 * Created by Yifan Wang on 08/01/2018.
 */
int getNextVertexOnPath(QSharedPointer<Graph> graph, int start_key, QVector<int>& visited_edges,
                        QVector<int> boundary) { // If no proper next vertex, return -1
    QSharedPointer<Vertex> start = graph->getVertex(start_key);
    QVector<QSharedPointer<Vertex>> neighbors = start->neighbors();
    if (neighbors.size() == 0) {
        return -1;
    }
    QVector<int> next_vertices;
    for (QSharedPointer<Vertex> neighbor : neighbors) {
        QSharedPointer<HalfEdge> edge = graph->getHalfEdge(start, neighbor);
        if (visited_edges.indexOf(edge->key()) == -1 &&
            visited_edges.indexOf(edge->twin()->key()) ==
                -1) { // The edge between current center vertex and this neighbor has not been visited
            next_vertices.push_back(neighbor->key());
        }
    }
    int next_vertex = getVertexClosestToBoundary(graph, next_vertices, visited_edges, boundary);
    return next_vertex;
}

/**
 * generateLongestPath - Generates and returns the almost longest path starting from specific vertex.
 *
 * Created by Yifan Wang on 08/01/2018.
 */
QVector<QPair<int, int>> generateLongestPath(QSharedPointer<Graph> graph, int start_key, QVector<int>& visited_edges,
                                             QVector<int>& unvisited_edges, QVector<int> boundary) {
    int next_vertex_key = getNextVertexOnPath(graph, start_key, visited_edges, boundary);
    QVector<QPair<int, int>> path;
    while (next_vertex_key >= 0) {
        visited_edges.push_back(graph->getHalfEdge(start_key, next_vertex_key)->key());
        visited_edges.push_back(graph->getHalfEdge(next_vertex_key, start_key)->key());
        unvisited_edges.remove(unvisited_edges.indexOf(graph->getHalfEdge(start_key, next_vertex_key)->key()));
        unvisited_edges.remove(unvisited_edges.indexOf(graph->getHalfEdge(next_vertex_key, start_key)->key()));

        path.push_back(QPair<int, int>(start_key, next_vertex_key));
        start_key = next_vertex_key;
        next_vertex_key = getNextVertexOnPath(graph, start_key, visited_edges, boundary);
    }
    return path;
}

/**
 * generatePaths - Generates and returns paths for traversing the whole graph.
 *                 The returning vector contains several vectors where each vector is a continuous path.
 *                 Each continuous path contains sevaral pairs where each pair corresponds to an edge in graph.
 *                 The first integer in a pair is vertex key of the starting vertex and the second integer is vertex key
 * of the ending vertex on this edge. For example, {(1,3), (3,5), (5,10)} represents the continuous path 1->3->5->10. So
 * size of returning vector is one plus the number of lifting during traversing the whole graph.
 *
 * Created by Yifan Wang on 08/01/2018.
 */
QVector<QVector<QPair<int, int>>> CirclePack::generatePaths(QSharedPointer<Graph> graph) {
    int start_key = -1;
    QVector<QVector<QPair<int, int>>> paths;
    QVector<int> boundary;
    QVector<int> visited_edges;
    QVector<int> unvisited_edges;
    for (QMap<int, QSharedPointer<HalfEdge>>::iterator it = graph->beginEdges(); it != graph->endEdges(); it++) {
        unvisited_edges.push_back(it.key());
    }
    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices(); it++) {
        if (graph->isBoundary(it.value())) {
            boundary.push_back(it.key());
        }
    }
    if (boundary.size() > 0) {
        start_key = boundary[0];
    }
    if (start_key == -1) {
        qDebug() << "Error in generatePath()" << endl;
        return paths;
    }

    while (unvisited_edges.size() > 0) {
        QVector<QPair<int, int>> longest_path =
            generateLongestPath(graph, start_key, visited_edges, unvisited_edges, boundary);
        // qDebug() << "Longest path length: " << longest_path.size() << endl;
        paths.push_back(longest_path);
        QVector<int> possible_vertices;
        for (int e : unvisited_edges) {
            if (possible_vertices.indexOf(graph->getHalfEdge(e)->origin()->key()) == -1 &&
                getNumAvailableDirectionsFromCurrentVertex(graph, graph->getHalfEdge(e)->origin()->key(), visited_edges,
                                                           boundary) ==
                    1) // Choose the points which have only one unvisited out edge as possible points to be the start
                       // point in next longest path.
            {
                possible_vertices.push_back(graph->getHalfEdge(e)->origin()->key());
            }
        }
        if (possible_vertices.size() == 0) {
            return paths;
        }
        // start_key = getVertexClosestToBoundary(graph, possible_vertices, visited_edges, boundary);
        // if (start_key == -1)
        //{
        start_key = possible_vertices[0]; // Much faster than start_key = getVertexClosestToBoundary(...)
        //}
    }

    return paths;
}

/**
 * calculateCentroid - Calculates and returns the centroid of the given face in
 *     the graph, treating the face as a polygon. This function uses the
 *     centroid of a polygon formula displayed on Wikipedia.
 */
Point calculateCentroid(QSharedPointer<Graph> graph, QSharedPointer<Face> face) {
    QVector<Point> vertices;

    { // Get all the corner vertices of the face.
        QSharedPointer<HalfEdge> start = face->boundary();
        QSharedPointer<HalfEdge> current = start;
        do {
            QSharedPointer<Circle> circle = graph->getCircle(current->origin()->key());
            if (circle == nullptr) {
                qDebug() << "Error: Vertex " << current->origin()->key();
                qDebug() << " has no Circle associated with it." << endl;
                return Point(0, 0);
            }
            vertices.push_back(circle->center());
            current = current->next();
        } while (current != start);
    }

    int64_t n = vertices.size();
    if (n < 3) // Polygon must have at least 3 vertices.
    {
        qDebug() << "Error: Face has only " << n << " vertices." << endl;
        return Point(0, 0);
    }
    else if (n == 3) // Special case; calculate centroid of triangle.
    {
        int64_t total_x = vertices[0].x() + vertices[1].x() + vertices[2].x();
        int64_t total_y = vertices[0].y() + vertices[1].y() + vertices[2].y();
        return Point(total_x / 3, total_y / 3);
    }
    else // Calculate centroid of polygon.
    {
        int64_t centroid_x = 0;
        int64_t centroid_y = 0;
        int64_t signed_area = 0;
        for (int64_t i = 0; i < n; i++) {
            int64_t xi = vertices[i].x();
            int64_t yi = vertices[i].y();
            int64_t xii = vertices[(i + 1) % n].x();
            int64_t yii = vertices[(i + 1) % n].y();
            int64_t factor = (xi * yii - xii * yi);
            centroid_x += (xi + xii) * factor;
            centroid_y += (yi + yii) * factor;
            signed_area += factor;
        }
        signed_area *= 3;
        return Point(centroid_x / signed_area, centroid_y / signed_area);
    }
}

/**
 * calculateAverageRadius - Returns the average radius of the corner vertices
 *     of the specified face in the graph.
 */
int64_t calculateAverageRadius(QSharedPointer<Graph> graph, QSharedPointer<Face> face) {
    int64_t total_radius = 0;
    int64_t num_vertices = 0;

    FaceEdgeIterator it(face);
    while (it.hasNext()) {
        QSharedPointer<Vertex> corner_vertex = it.getNext()->origin();
        total_radius += graph->getVertex(corner_vertex->key())->data<VertexLocationCircleData>()->m_circle->radius();
        num_vertices++;
    }

    if (num_vertices > 0) {
        return total_radius / num_vertices;
    }
    else {
        qDebug() << "Error: Face has no boundary edges!" << endl;
        return 0;
    }
}

/**
 * addBarycenterToFace - Adds a barycenter to the specified face of the graph.
 *     A barycenter is defined as a vertex inside a face that is connected to
 *     every corner vertex of the face. The barycenter is set at the centroid
 *     of the face and the radius of the circle associated with the barycenter
 *     is half the average of all of its neighbors. If the face has E edges,
 *     then addition of the barycenter will create 1 new vertex, create E - 1
 *     new faces, and create and set up 2*E new halfEdges. There are E - 1 new
 *     faces because the old face is reused as one of the new ones in order to
 *     avoid having to re-map all the faces in the graph. Re-mapping would
 *     otherwise be necessary since the face key of the new faces is determined
 *     by the number of existing faces in the graph and thus deleting a face
 *     would cause overlaps in the face keys of new faces leading to multiple
 *     serious issues.
 *
 * @param Graph* graph must be a properly set up DCEL.
 * @param Face* face must have a boundary halfEdge != nullptr, else segfault.
 *
 * @return Pointer to the newly created barycenter or nullptr if one of the
 * parameters was a nullptr.
 */
QSharedPointer<Vertex> CirclePack::addBarycenterToFace(QSharedPointer<Graph> graph, QSharedPointer<Face> face) {
    if (graph == nullptr || face == nullptr)
        return nullptr;

    int64_t barycenter_key = graph->getNumVertices();
    int64_t edge_key = graph->getNumHalfEdges();
    int64_t face_key = graph->getNumFaces() - 1;

    QSharedPointer<Vertex> barycenter = graph->addVertex(barycenter_key);

    // Add planar embedding real data to the vertex.
    Point center = calculateCentroid(graph, face);
    int64_t average_radius = calculateAverageRadius(graph, face);
    QSharedPointer<Circle> circle(new Circle(center, average_radius / 2));
    graph->setData(barycenter_key, circle);

    QVector<QSharedPointer<HalfEdge>> boundary_edges; // Edges of the original face.
    QVector<QSharedPointer<HalfEdge>> spoke_edges;    // Edges from barycenter to corner vertices.

    QSharedPointer<HalfEdge> start_edge = face->boundary();
    QSharedPointer<HalfEdge> current_edge = start_edge;
    do { // For each boundary edge of the face.
        boundary_edges.push_back(current_edge);

        QSharedPointer<HalfEdge> spoke_edge(new HalfEdge()); // Set up new spoke edge.
        spoke_edge->origin(barycenter);
        spoke_edge->key(edge_key);
        // graph->eMap[edge_key] = spoke_edge;
        graph->addEdge(edge_key, spoke_edge);
        edge_key++;

        QSharedPointer<HalfEdge> spoke_edge_twin(new HalfEdge()); // Set up its twin.
        spoke_edge_twin->origin(current_edge->origin());
        spoke_edge_twin->key(edge_key);
        // graph->eMap[edge_key] = spoke_edge_twin;
        graph->addEdge(edge_key, spoke_edge_twin);
        edge_key++;

        spoke_edge->setTwin(spoke_edge_twin);

        spoke_edges.push_back(spoke_edge);

        current_edge = current_edge->next();
    } while (current_edge != start_edge);

    int64_t n = boundary_edges.size();
    for (int64_t i = 0; i < n; i++) // Set up each new face.
    {
        QSharedPointer<HalfEdge> boundary_edge = boundary_edges[i];
        QSharedPointer<HalfEdge> spoke_edge = spoke_edges[i];
        QSharedPointer<HalfEdge> next_spoke_edge = spoke_edges[(i + 1) % n]->twin();

        // Create the new face and set up references.
        QSharedPointer<Face> new_face = nullptr;
        if (i == 0) // Re-use old face as the first of the new ones.
        {
            face->boundary(boundary_edge);
            new_face = face;
        }
        else {
            new_face = graph->addFace(face_key++, boundary_edge);
        }
        boundary_edge->leftFace(new_face);
        spoke_edge->leftFace(new_face);
        next_spoke_edge->leftFace(new_face);

        // Connect the halfEdges around the new face.
        spoke_edge->setNext(boundary_edge);
        boundary_edge->setNext(next_spoke_edge);
        next_spoke_edge->setNext(spoke_edge);
    }

    barycenter->incidentEdge(spoke_edges[0]);

    return barycenter;
}

/**
 * flipEdge - Flips the edge specified by the halfEdge and its twin. In a
 *     triangulation, an interior edge is shared by two triangular faces. To
 *     'flip' the edge, we re-orient the edge so that it forms the other
 *     diagonal in the union of those two faces. This process removes nothing
 *     and adds nothing to the graph, but simply just rearranges pointers.
 *     Note: Make sure the edge provided is not a boundary edge, since for our
 *     purposes, forming an edge through the exterior face is undesirable.
 *     Also the edge is flipped counter-clockwise and 4 flips == 0 flips.
 */
void CirclePack::flipEdge(QSharedPointer<HalfEdge> h) {
    QSharedPointer<Face> left_face = h->leftFace();
    QSharedPointer<Face> right_face = h->twin()->leftFace();

    QSharedPointer<HalfEdge> h_prev = h->previous();
    QSharedPointer<HalfEdge> h_next = h->next();
    QSharedPointer<HalfEdge> twin_prev = h->twin()->previous();
    QSharedPointer<HalfEdge> twin_next = h->twin()->next();

    // Make sure the edge end vertices don't use the edge as the incident edge.
    h->origin()->incidentEdge(twin_next);
    h->target()->incidentEdge(h_next);

    // Fix edge origins.
    h->origin(twin_prev->origin());
    h->twin()->origin(h_prev->origin());

    // Fix edge connections for left face.
    h->setNext(h_prev);
    h_prev->setNext(twin_next);
    twin_next->setNext(h);

    // Fix face references for left face.
    left_face->boundary(h);
    twin_next->leftFace(left_face);

    // Fix edge connections for right face.
    h->twin()->setNext(twin_prev);
    twin_prev->setNext(h_next);
    h_next->setNext(h->twin());

    // Fix face references for right face.
    right_face->boundary(h->twin());
    h_next->leftFace(right_face);
}

/**
 * eraseBoundaryEdge - Erases the specified boundary edge in the graph and
 *     properly updates the affected half edges. This method removes 2 halfEdges
 *     and the 1 interior face that was associated with the removed edge. Thus
 *     there will now be an 'index gap' in the graph for both faces and
 *     halfEdges that may need to be fixed for certain operations on the graph.
 */
void CirclePack::eraseBoundaryEdge(QSharedPointer<Graph> graph, QSharedPointer<HalfEdge> h) {
    if (h->twin()->leftFace() == graph->exterior())
        h = h->twin(); // We want h to be the exterior halfEdge of the edge.

    QSharedPointer<Face> interior_face = h->twin()->leftFace();
    if (interior_face == graph->exterior()) // Error check.
    {
        qDebug() << "Error: Both faces of this edge are exterior." << endl;
        return;
    }
    graph->removeFace(interior_face);

    // Make sure the edge end vertices don't use the edge as the incident edge.
    if (h->twin()->next() == h)
        h->origin()->incidentEdge(nullptr);
    else
        h->origin()->incidentEdge(h->twin()->next());
    if (h->next() == h->twin())
        h->target()->incidentEdge(nullptr);
    else
        h->target()->incidentEdge(h->next());

    // Make sure the exterior face doesn't use the edge as its boundary.
    if (h->next() == h->twin())
        graph->exterior()->boundary(nullptr);
    else
        graph->exterior()->boundary(h->next());

    // Fix connections of the exterior halfEdges.
    h->previous()->setNext(h->twin()->next());
    h->twin()->previous()->setNext(h->next());

    // Remove both halfEdges of the boundary edge.
    // graph->eMap.erase(h->key);
    // graph->eMap.remove(h->key);
    graph->eraseEdge(h->key());
    // graph->eMap.erase(h->twin->key);
    graph->eraseEdge(h->twin()->key());
    // delete h->twin;
    h->twin().clear();
    // h->twin = nullptr;
    // delete h;
    // h = nullptr;
    h.clear();
}

/**
 * localRefine - Given a graph and a set of vertices, refine all the faces
 *     in the graph that are tangent to one or more vertices in the set. The
 *     basic process of refinement is adding barycenters to all the selected
 *     faces, flipping all interior edges that connected the vertices in the
 *     set to instead connect the new barycenters, and then finally removing
 *     any boundary edges (edges on the exterior of the entire graph) that were
 *     tangent to vertices in the set.
 */
void CirclePack::localRefine(QSharedPointer<Graph> graph, QSet<QSharedPointer<Vertex>> vertices) {
    QSet<QSharedPointer<Face>> faces; // Using set to avoid duplicates
    QVector<QSharedPointer<HalfEdge>> interior_edges;
    QVector<QSharedPointer<HalfEdge>> boundary_edges;

    for (QSharedPointer<Vertex> vertex : vertices) // Store all tangent faces and halfEdges.
    {
        QSharedPointer<HalfEdge> start = vertex->incidentEdge();
        QSharedPointer<HalfEdge> current = start;
        do {
            if (current->leftFace() != graph->exterior())
                faces.insert(current->leftFace());

            QSharedPointer<Vertex> neighbor = current->target();
            // Prevent adding both a halfEdge AND its twin by checking this.
            if (vertex->key() < neighbor->key() || vertices.find(neighbor) == vertices.end()) {
                if (current->leftFace() == graph->exterior() || current->twin()->leftFace() == graph->exterior())
                    boundary_edges.push_back(current);
                else
                    interior_edges.push_back(current);
            }

            current = current->twin()->next();
        } while (current != start);
    }

    for (QSharedPointer<Face> face : faces) {
        addBarycenterToFace(graph, face);
    }

    for (QSharedPointer<HalfEdge> interior_edge : interior_edges) {
        flipEdge(interior_edge);
    }

    for (QSharedPointer<HalfEdge> boundary_edge : boundary_edges) {
        eraseBoundaryEdge(graph, boundary_edge);
    }

    if (boundary_edges.size() > 0) // Edges have been deleted from the graph.
    {
        graph->remapHalfEdges(); // Re-map to remove 'index gaps'.
        graph->remapFaces();
    }
}

/**
 * localRefine - Given a graph, refine all the vertices in the graph that are
 *     inside the boundary shape specified by the polygons parameter.
 */
void CirclePack::localRefine(QSharedPointer<Graph> graph, PolygonList* polygons) {
    QSet<QSharedPointer<Vertex>> selection = selectVerticesInside(graph, polygons);
    localRefine(graph, selection);
}

/**
 * localRefine - Given a graph, refine all the vertices in the graph that are
 *     within threshold distance of the boundary shape specified by the polygons
 *     parameter.
 */
void CirclePack::localRefine(QSharedPointer<Graph> graph, PolygonList* polygons, int64_t threshold) {
    QSet<QSharedPointer<Vertex>> selection = selectVerticesCloseToBoundary(graph, polygons, threshold);
    localRefine(graph, selection);
}

/**
 * addBoundaryTab - Add a boundary tab of the specified type to the dual graph
 *     at its specified vertex going towards the specified boundary halfEdge of
 *     the original graph. Addition of the boundary tab includes adding a new
 *     vertex to the dual graph as the other end-point of the boundary tab.
 */
QSharedPointer<HalfEdge> addBoundaryTab(BoundaryTabType tab_type, QSharedPointer<Graph> dual,
                                        QSharedPointer<Vertex> vertex, QSharedPointer<Graph> graph,
                                        QSharedPointer<HalfEdge> h, int64_t& edge_key) {
    QSharedPointer<Vertex> target = dual->addVertex(dual->getNumVertices());

    QSharedPointer<Circle> circle1 = graph->getVertex(h->origin()->key())->data<VertexLocationCircleData>()->m_circle;
    QSharedPointer<Circle> circle2 = graph->getVertex(h->target()->key())->data<VertexLocationCircleData>()->m_circle;

    // Determine location of the end-point vertex of the tab.
    Point target_center = Point(0, 0);
    int64_t radius = (circle1->radius() + circle2->radius()) / TAB_RADIUS_RATIO;
    if (tab_type == EDGE_MIDPOINT) {
        target_center = (circle1->center() + circle2->center()) / 2;
    }
    else if (tab_type == CIRCLES_TANGENT) {
        Point vectC1C2 = circle2->center() - circle1->center();
        int64_t magnitude = vectC1C2.distance()();
        Point vectC1Target = vectC1C2 * circle1->radius() / magnitude;
        target_center = circle1->center() + vectC1Target;
    }
    QSharedPointer<Circle> circle(new Circle(target_center, radius));
    dual->setData(target, circle);

    return addHalfEdges(dual, vertex, target, edge_key);
}

/**
 * generateDualGraph - Returns a new DCEL graph that is the dual of the input
 *  DCEL graph. The dual will have 1 vertex for every face, 1 face for every
 *  internal vertex, and 1 edge for every internal edge. If "boundary tabs" are
 *  also included, then there will also be a "boundary tab" edge for every
 *  boundary edge in the input graph along with additional vertices that form
 *  the leaf end-points of those boundary tabs. The location of the additional
 *  vertices is specified by the input parameter.
 *  Note: The boundary edges of the dual graph faces move counter-clockwise.
 */
QSharedPointer<Graph> CirclePack::generateDualGraph(QSharedPointer<Graph> graph, BoundaryTabType tab_type) {
    QSharedPointer<Graph> dual(new Graph());

    // Mapping of input graph face keys to dual graph vertices.
    QMap<int64_t, QSharedPointer<Vertex>> vertex_map;

    // List of incident halfEdges for each vertex in the dual graph.
    QMap<int64_t, QVector<QSharedPointer<HalfEdge>>> adj_edge_list;

    // Mapping of dual edges to the input graph edges that they cross over.
    QMap<QSharedPointer<HalfEdge>, QSharedPointer<HalfEdge>> cross_edge;

    // Step 1: Add a dual vertex for each face.
    int64_t vertex_key = 0;
    for (QMap<int, QSharedPointer<Face>>::iterator it = graph->beginFaces(); it != graph->endFaces(); it++) {
        if (it.value() == graph->exterior())
            continue; // Skip the exterior face.

        QSharedPointer<Vertex> vertex = dual->addVertex(vertex_key);

        // Add planar embedding real data to the vertex.
        Point center = calculateCentroid(graph, it.value());
        int64_t average_radius = calculateAverageRadius(graph, it.value());
        QSharedPointer<Circle> circle(new Circle(center, average_radius / 2));
        dual->setData(vertex_key, circle);

        // Create empty list of incident halfEdges for this vertex.
        adj_edge_list[vertex_key] = QVector<QSharedPointer<HalfEdge>>();

        vertex_map[it.key()] = vertex;
        vertex_key++;
    }

    // Step 2: Add all dual graph edges.
    int64_t edge_key = 0;
    for (QMap<int, QSharedPointer<Face>>::iterator it = graph->beginFaces(); it != graph->endFaces(); it++) {
        if (it.value() == graph->exterior())
            continue; // Skip the exterior face.

        FaceEdgeIterator face_edges(it.value());
        while (face_edges.hasNext()) {
            QSharedPointer<HalfEdge> h = face_edges.getNext();
            if (h->leftFace() == graph->exterior() || h->twin()->leftFace() == graph->exterior()) // Boundary edge.
            {
                // Skip boundary edges if boundary tabs are not requested.
                if (tab_type == NONE)
                    continue;

                QSharedPointer<Vertex> vertex = vertex_map[it.key()];
                QSharedPointer<HalfEdge> tab = addBoundaryTab(tab_type, dual, vertex, graph, h, edge_key);
                adj_edge_list[vertex->key()].push_back(tab);
                adj_edge_list[tab->target()->key()] = QVector<QSharedPointer<HalfEdge>>();
                adj_edge_list[tab->target()->key()].push_back(tab->twin());
                cross_edge[tab] = h->twin();
                cross_edge[tab->twin()] = h;
            }
            else // Interior edge.
            {
                // Interior edges will appear twice; This prevent duplicates.
                if (it.key() > h->twin()->leftFace()->key())
                    continue;

                QSharedPointer<Vertex> vertex = vertex_map[it.key()];
                QSharedPointer<Vertex> neighbor = vertex_map[h->twin()->leftFace()->key()];
                QSharedPointer<HalfEdge> edge = addHalfEdges(dual, vertex, neighbor, edge_key);
                adj_edge_list[vertex->key()].push_back(edge);
                adj_edge_list[neighbor->key()].push_back(edge->twin());
                cross_edge[edge] = h->twin();
                cross_edge[edge->twin()] = h;
            }
        }
    }

    // Step 3: Add a dual graph face for each input graph internal vertex.
    int64_t face_key = 0;
    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices(); it++) {
        if (graph->isBoundary(it.value()))
            continue; // Skip boundary vertices.

        QSharedPointer<Face> face = dual->addFace(face_key, nullptr);

        // Identify the boundary edges for this face.
        QVector<QSharedPointer<HalfEdge>> boundary_edges;
        QVector<QSharedPointer<HalfEdge>> out_edges = it.value()->outEdges();
        int64_t n = out_edges.size();
        for (int64_t i = 0; i < n; i++) {
            // out_edges is in clockwise order, so we link backwards for CCW.
            QSharedPointer<Vertex> origin = vertex_map[out_edges[(i + 1) % n]->leftFace()->key()];
            QSharedPointer<Vertex> target = vertex_map[out_edges[i]->leftFace()->key()];
            for (QSharedPointer<HalfEdge> h : adj_edge_list[origin->key()]) {
                if (h->target() == target) {
                    boundary_edges.push_back(h);
                    break;
                }
            }
        }

        n = boundary_edges.size();
        for (int64_t i = 0; i < n; i++) // Connect boundary edges around face.
        {
            boundary_edges[i]->leftFace(face);

            // This looks weird but the next IS the previous (CCW around face).
            boundary_edges[i]->setPrevious(boundary_edges[(i + 1) % n]);
        }
        face->boundary(boundary_edges[0]);

        face_key++;
    }

    // Step 4: Connect boundary edges for exterior face.
    QSharedPointer<HalfEdge> start; // = nullptr;
    for (QMap<int, QSharedPointer<HalfEdge>>::iterator it = dual->beginEdges(); it != dual->endEdges();
         it++) // Find a boundary edge to start on.
    {
        if (it.value()->leftFace() == dual->exterior()) {
            start = it.value();
            break;
        }
    }

    if (start == nullptr) {
        qDebug() << "Error: Dual graph has no boundary edges!" << endl;
        return nullptr;
    }

    QSharedPointer<HalfEdge> current = start;
    do {
        // Find the next boundary edge.
        if (cross_edge[current]->leftFace() == graph->exterior()) {
            // Current edge is a boundary tab; Should be only 1 choice for next.
            QSharedPointer<HalfEdge> next = adj_edge_list[current->target()->key()][0];
            current->setNext(next);
            current = next;
            continue;
        }

        if (tab_type == NONE) // No boundary tabs.
        {
            // Only interior edges have a corresponding dual edge.
            QSharedPointer<HalfEdge> boundary_edge = cross_edge[current]->previous();
            while (boundary_edge->twin()->leftFace() == graph->exterior())
                boundary_edge = boundary_edge->previous();
            QSharedPointer<Vertex> target = vertex_map[boundary_edge->twin()->leftFace()->key()];
            for (QSharedPointer<HalfEdge> next : adj_edge_list[current->target()->key()]) {
                if (next->target() == target) {
                    current->setNext(next);
                    current = next;
                    break;
                }
            }
        }
        else // Boundary tabs are included.
        {
            // All input graph face edges will have a corresponding dual edge.
            for (QSharedPointer<HalfEdge> next : adj_edge_list[current->target()->key()]) {
                if (cross_edge[next]->twin() == cross_edge[current]->previous()) {
                    current->setNext(next);
                    current = next;
                    break;
                }
            }
        }
    } while (current != start);

    return dual;
}

/**
 * cookieGraphToShape - Cookie cuts the graph using the cookie cutter shape
 *     specified by the polygons. The first polygon determines the outer
 *     boundary of the shape and the remaining polygons determine "holes" within
 *     the shape. Vertices that lie outside the cookie cutter shape are removed
 *     along with all the edges that have those vertices as either end-point.
 *     Also, all the interior faces associated with those vertices are also
 *     removed and the exposed interior edges have the exterior face set as
 *     their leftFace.
 */
void CirclePack::cookieGraphToShape(QSharedPointer<Graph> graph, PolygonList* polygons) {
    removeVertexSelection(graph, selectVerticesOutside(graph, polygons));
}

void CirclePack::removeVertexSelection(QSharedPointer<Graph> graph, QSet<QSharedPointer<Vertex>> removal_set) {
    // First remove all faces associated with the set of vertices to be removed.
    for (QSharedPointer<Vertex> vertex : removal_set) {
        for (QSharedPointer<HalfEdge> halfEdge : vertex->outEdges()) {
            if (halfEdge->leftFace() != graph->exterior()) {
                graph->removeFace(halfEdge->leftFace());
            }
        }
    }

    int64_t vertices_removed = 0;
    for (QSharedPointer<Vertex> vertex : removal_set) {
        QSharedPointer<HalfEdge> start = vertex->incidentEdge();

        // Check if this edge still exists. It may have been deleted already
        // during the removal process of an earlier vertex.
        if (start != nullptr && start->twin() != nullptr) {
            QSharedPointer<HalfEdge> current = start->previous()->twin();
            current->twin()->previous()->setNext(current->next());

            // Delete edges associated with this vertex.
            while (current != start) {
                // Fix neighbor and remaining edge connections.
                if (current->next() == current->twin())
                    current->target()->incidentEdge(nullptr);
                else
                    current->target()->incidentEdge(current->next());
                QSharedPointer<HalfEdge> prev = current; // Save reference to current edge.
                current = current->previous()->twin();   // Move to the next edge.
                current->twin()->previous()->setNext(current->next());

                // Delete previously traversed edges of this vertex.
                // graph->eMap.erase(prev->twin->key);
                graph->eraseEdge(prev->twin()->key());
                prev->twin().clear();
                // delete prev->twin;
                // prev->twin = nullptr;

                // graph->eMap.erase(prev->key);
                graph->eraseEdge(prev->key());
                prev.clear();
                // delete prev;
                // prev = nullptr;
            }

            // Start must be removed last, in order for edge traversal to work.
            if (start->next() == start->twin()) // start->twin will soon be deleted.
                start->target()->incidentEdge(nullptr);
            else
                start->target()->incidentEdge(start->next());

            // graph->eMap.erase(start->twin->key);
            // graph->eMap.remove(start->twin()->key());
            graph->eraseEdge(start->twin()->key());
            // delete start->twin;
            start->twin().clear();
            // start->twin = nullptr;

            // graph->eMap.erase(start->key);
            // graph->eMap.remove(start->key);
            graph->eraseEdge(start->key());
            start.clear();
            // delete start;
            // start = nullptr;
        }

        // Delete the vertex and its associated data.
        // graph->vMap.erase(vertex->key);
        // graph->vMap.remove(vertex->key);
        graph->eraseVertex(vertex->key());
        // QSharedPointer<Circle> circle = graph->cMap(vertex->key());
        // graph->cMap.remove(vertex->key);
        // graph->eraseCircle(vertex->key());
        // circle.clear();
        // delete circle;
        // graph->pMap.erase(vertex->key);
        // graph->pMap.remove(vertex->key);
        // graph->erasePoint(vertex->key());
        // delete vertex;
        vertex.clear();

        vertices_removed++;
    }

    if (vertices_removed > 0) {
        // Boundary edge of exterior may have been removed, find it a new one.
        QSharedPointer<HalfEdge> h; // = nullptr;
        for (QMap<int, QSharedPointer<HalfEdge>>::iterator it = graph->beginEdges(); it != graph->endEdges(); it++)
            if (it.value()->leftFace() == graph->exterior())
                h = it.value();
        graph->exterior()->boundary(h);

        // We must re-map to fix "index gaps".
        graph->remapVertices();
        graph->remapHalfEdges();
        graph->remapFaces();
    }
}

/**
 * clipGraphToShape - This version takes in a Polygons object, converts it to a
 *     Paths object and calls the other method to do the actual clipping.
 */
QSharedPointer<Graph> CirclePack::clipGraphToShape(QSharedPointer<Graph> graph, PolygonList* polygons) {
    ClipperLib2::Paths clip;
    for (int64_t i = 0; i < polygons->size(); i++) {
        clip.push_back((*polygons)[i].getPath());
    }
    return clipGraphToShape(graph, clip);
}

/**
 * clipGraphToShape - Returns a new graph that is the clipped version of the
 *     input graph. The clipping shape is determined by the paths with the
 *     first path being the outer boundary of the shape and the remaining
 *     paths being "holes" in the shape. The returned graph is a DCEL;
 *     however, the faces are not set up (every edge has exterior as leftFace).
 */
QSharedPointer<Graph> CirclePack::clipGraphToShape(QSharedPointer<Graph> graph, ClipperLib2::Paths clip) {
    ClipperLib2::Clipper clipper;

    // Load subject paths (edges) from the graph.
    ClipperLib2::Paths subject;
    for (QMap<int, QSharedPointer<HalfEdge>>::iterator it = graph->beginEdges(); it != graph->endEdges();
         it++) // Add all edges of the graph.
    {
        int64_t origin_key = it.value()->origin()->key();
        int64_t target_key = it.value()->target()->key();

        if (origin_key < target_key)
            continue; // Prevent duplicates.

        Point origin = graph->getVertex(origin_key)->data<VertexLocationCircleData>()->m_circle->center();
        Point target = graph->getVertex(target_key)->data<VertexLocationCircleData>()->m_circle->center();

        ClipperLib2::Path edge = {origin.toIntPoint(), target.toIntPoint()};
        subject.push_back(edge);
    }

    // Add subject and clip paths to the clipper.
    bool valid = clipper.AddPaths(subject, ClipperLib2::ptSubject, false);
    if (!valid) {
        qDebug() << "Error: Subject paths are invalid for clipping!" << endl;
        return nullptr;
    }

    valid = clipper.AddPaths(clip, ClipperLib2::ptClip, true);
    if (!valid) {
        qDebug() << "Error: Clipping paths are invalid for clipping!" << endl;
        return nullptr;
    }

    // Perform the clipping.
    ClipperLib2::PolyTree solution;
    clipper.Execute(ClipperLib2::ctIntersection, solution, ClipperLib2::pftEvenOdd, ClipperLib2::pftEvenOdd);
    ClipperLib2::Paths solution_paths;
    ClipperLib2::PolyTreeToPaths(solution, solution_paths);

    // Extract all the vertex location data in the clipping.
    QVector<Point> points;
    for (ClipperLib2::Path path : solution_paths) {
        // The checks prevent duplicate points.
        if (qFind(points.begin(), points.end(), path[0]) == points.end())
            points.push_back(path[0]);

        if (qFind(points.begin(), points.end(), path[1]) == points.end())
            points.push_back(path[1]);
    }

    // Save the clipping as a graph.
    QSharedPointer<Graph> clipped_graph(new Graph());
    int64_t vertex_key = 0;
    for (Point point : points) // Add vertices.
    {
        QSharedPointer<Vertex> vertex = clipped_graph->addVertex(vertex_key);
        int64_t radius = 1 * load_scale; // Temporary default radius.
        QSharedPointer<Circle> circle(new Circle(point, radius));
        clipped_graph->setData(vertex_key, circle);
        vertex_key++;
    }

    int64_t edge_key = 0;
    for (ClipperLib2::Path path : solution_paths) // Add edges.
    {
        QSharedPointer<Vertex> origin; // = nullptr;
        QSharedPointer<Vertex> target; // = nullptr;
        for (QMap<int, QSharedPointer<Vertex>>::iterator it = clipped_graph->beginVertices();
             it != clipped_graph->endVertices(); it++) // Find origin vertex.
        {
            if (it.value()->data<VertexLocationCircleData>()->m_circle->center() == path[0]) {
                origin = clipped_graph->getVertex(it.key());
                break;
            }
        }
        for (QMap<int, QSharedPointer<Vertex>>::iterator it = clipped_graph->beginVertices();
             it != clipped_graph->endVertices(); it++) // Find target vertex.
        {
            if (it.value()->data<VertexLocationCircleData>()->m_circle->center() == path[1]) {
                target = clipped_graph->getVertex(it.key());
                break;
            }
        }
        if (origin != nullptr && target != nullptr) // Connect them.
        {
            addHalfEdges(clipped_graph, origin, target, edge_key);
        }
    }

    // Set next and previous pointers of the edges to allow edge traversal.
    for (QMap<int, QSharedPointer<Vertex>>::iterator it = clipped_graph->beginVertices();
         it != clipped_graph->endVertices(); it++) {
        QVector<QSharedPointer<HalfEdge>> in_edges;
        for (QMap<int, QSharedPointer<HalfEdge>>::iterator it2 = clipped_graph->beginEdges();
             it2 != clipped_graph->endEdges(); it2++) {
            if (it2.value()->target() == it.value()) {
                in_edges.push_back(it2.value());
            }
        }

        int64_t n = in_edges.size();
        for (int i = 0; i < n; i++) {
            in_edges[i]->setNext(in_edges[(i + 1) % n]->twin());
        }
    }

    return clipped_graph;
}

/**
 * removeInternalSecondary - Removes the internal secondary vertices of the
 *     graph and maintains the DCEL.
 */
void CirclePack::removeInternalSecondary(QSharedPointer<Graph> graph) {
    QSet<QSharedPointer<Vertex>> removal_set;

    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices(); it++) {
        //        QMap<int, QSharedPointer<Circle>>::iterator it2 = graph->cMap()->find(it.key());

        //        if (it2 != graph->cMap()->end())
        //        {
        //            if (it2.value()->isSecondary && !graph->isBoundary(it.value()))
        //            {
        //                removal_set.insert(it.value());
        //            }
        //        }

        if (it.value()->data<VertexLocationCircleData>()->m_circle->isSecondary() && !graph->isBoundary(it.value()))

        {
            removal_set.insert(it.value());
        }
    }

    removeVertexSelection(graph, removal_set);
}

void CirclePack::translateGraph(QSharedPointer<Graph> graph, int64_t x_offset, int64_t y_offset) {
    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices(); it++) {
        QSharedPointer<Circle> circle = it.value()->data<VertexLocationCircleData>()->m_circle;
        // qDebug() << circle->center().x();
        // qDebug() << circle->center().x() + static_cast<double>(x_offset);
        Point new_center(0, 0);
        new_center.x(circle->center().x() + static_cast<double>(x_offset));
        new_center.y(circle->center().y() + static_cast<double>(y_offset));
        circle->center(new_center);
        // qDebug() << circle->center().x() << " " << circle->center().y();
        // qDebug() << "==========";
    }
}

void CirclePack::scaleGraph(QSharedPointer<Graph> graph, int64_t scale) {

    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices(); it++) {
        QSharedPointer<Circle> circle = it.value()->data<VertexLocationCircleData>()->m_circle;
        Point new_center(0, 0);
        // qDebug() << circle->center().x() << " " << circle->center().y();
        new_center.x(circle->center().x() * scale);
        new_center.y(circle->center().y() * scale);
        circle->center(new_center);
        circle->radius(circle->radius() * scale);
    }
}

void CirclePack::rotateGraph(QSharedPointer<Graph> graph, double degrees, Orientation orientation) {

    double radians = M_PI / 180.0 * degrees; // Convert degrees to radians.
    for (QMap<int, QSharedPointer<Vertex>>::iterator it = graph->beginVertices(); it != graph->endVertices(); it++) {
        QSharedPointer<Circle> circle = it.value()->data<VertexLocationCircleData>()->m_circle;
        int64_t x = circle->center().x();
        int64_t y = circle->center().y();
        Point new_center;
        if (orientation == CLOCKWISE) // Rotate clockwise.
            new_center =
                Point(x * std::cos(radians) + y * std::sin(radians), y * std::cos(radians) - x * std::sin(radians));
        else // Rotate counter-clockwise.
            new_center =
                Point(x * std::cos(radians) - y * std::sin(radians), x * std::sin(radians) + y * std::cos(radians));
        circle->center(new_center);
    }
}

int64_t CirclePack::getLoadScale() { return load_scale; }
int64_t CirclePack::getIterations() { return iterations; }
double CirclePack::getAdjustmentFactor() { return adjustment_factor; }
double CirclePack::getMinInversiveDist() { return min_inversive_dist; }
double CirclePack::getMaxInversiveDist() { return max_inversive_dist; }
BoundaryMode CirclePack::getBoundaryMode() { return boundary_mode; }

void CirclePack::setLoadScale(int64_t load_scale) { this->load_scale = load_scale; }
void CirclePack::setIterations(int64_t iterations) { this->iterations = iterations; }
void CirclePack::setAdjustmentFactor(double adjustment_factor) { this->adjustment_factor = adjustment_factor; }
void CirclePack::setMinInversiveDist(double min_inversive_dist) { this->min_inversive_dist = min_inversive_dist; }
void CirclePack::setMaxInversiveDist(double max_inversive_dist) { this->max_inversive_dist = max_inversive_dist; }
void CirclePack::setBoundaryMode(BoundaryMode boundary_mode) { this->boundary_mode = boundary_mode; }

} // namespace ORNL
