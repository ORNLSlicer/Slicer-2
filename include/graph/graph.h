#ifndef GRAPH_H
#define GRAPH_H

#include <QEnableSharedFromThis>
#include <QMap>
#include <QSharedPointer>
#include <QVector>

#include "../geometry/point.h"
#include "circle.h"
#include "face.h"
#include "faceedgeiterator.h"
#include "halfedge.h"
#include "vertex.h"
#include "vertexedgeiterator.h"

namespace ORNL
{
    /*!
    * \class Graph
    * \brief A graph is representation of a set of objects where some pairs of objects
    *        are connected by links. The interconnected objects are represented by points termed as
    *        vertices, and the links that connect the vertices are called edges.
    */

    class Graph: public QEnableSharedFromThis<Graph>
    {
    public:
        Graph();
        ~Graph();
        /*!
        * \brief addVertex New vertex is added to a graph object with a unique id(key).
        * \param key Is a unique integer value associated with vertex.
        * \return Shared pointer points to new vertex added.
        */
        QSharedPointer<Vertex> addVertex(int key);
        /*!
        * \brief removeVertex Vertex is removed from graph based on its key value. Associated
        * edges are also removed.
        * \param key Is a unique integer value associated with vertex.
        */
        void removeVertex(int key);
        /*!
        * \brief removeVertex Vertex is removed from graph based on key of vertex v. Associated
        * edges are also removed.
        * \param v Shared pointer points to vertex object.
        */
        void removeVertex(QSharedPointer<Vertex> v);

    //    //! \brief Add associated point to vertex.
    //    void setData(int key, Point p);
        /*!
        * \brief setData Assigning point object to vertex with unique id (key).
        * \param key Unique integer value associated with vertex.
        * \param p Shared pointer points to point object.
        */
        void setData(int key, QSharedPointer<Point> p);

//      void setData(int key, int x, int y, int z);
//      void setData(QSharedPointer<Vertex> v, Point p);
//      void setData(QSharedPointer<Vertex> v, int x, int y, int z);

        /*!
        * \brief setData Assigning circle object to vertex with unique id (key).
        * \param key Unique integer value associated with vertex.
        * \param c Shared pointer points to circle object.
        */

        void setData(int key, QSharedPointer<Circle> c);

        /*!
        * \brief setData Assigning circle object to vertex based on its unique id (key).
        * \param v Shared pointer points to vertex object.
        * \param c Shared pointer points to circle object.
        */

        void setData(QSharedPointer<Vertex> v, QSharedPointer<Circle> c);

//      /*!
//      * \brief removeCircle circle is removed from vertex data. Also vertex associated
//      * with vertex data is removed.
//      * \param key Unique integer value associated with vertex .
//      */
//      void removeCircle(int key);

        /*!
        * \brief addFace  New face is added to a graph object with a unique id(key) and edge data.
        * \param key Is a unique integer value associated with new face.
        * \param edge Shared pointer points to edge object.
        * \return Shared Pointer points to new face object.
        */
        QSharedPointer<Face> addFace(int key, QSharedPointer<HalfEdge> edge);

        /*!
        * \brief addFace  New face is added to a graph object with edge data at the end.
        * \param edge Shared pointer points to edge object.
        * \return Shared pointer points to new face object.
        */

        QSharedPointer<Face> addFace(QSharedPointer<HalfEdge> edge);

        /*!
        * \brief removeFace Face is removed from graph based on face key( unique id).
        * \param key Is a unique integer value associated with face.
        */

        void removeFace(int key);

        /*!
        * \brief removeFace Face is removed from graph based on face key( unique id).
        * \param face Shared Pointer points to face object.
        */

        void removeFace(QSharedPointer<Face> face);

        /*!
        * \brief connectVertices Creates two edges, first one from origin to twinOrigin and assign face as leftface.
        * Second one is a twin edge, from twinOrigin to origin and assign twinFace as leftface.
        * Leftface is a face on left side of edge.
        * \param originKey Is a unique integer value (key) associated with origin vertex of edge.
        * \param twinOriginKey Is a unique integer value (key) associated with origin vertex of twin edge.
        * \param faceKey Is a unique integer value (key) associated with face on left side of edge.
        * \param twinFaceKey Is a unique integer value associated with face on left side of twin edge.
        * \return Shared pointer points to edge. Note: Not it's twin edge.
        */

        QSharedPointer<HalfEdge> connectVertices(int originKey, int twinOriginKey, int faceKey, int twinFaceKey);

        /*!
        * \brief connectVertices Creates two edges, first one from origin to twinOrigin and assign face as leftface.
        * Second one is a twin edge, from twinOrigin to origin and assign twinFace as leftface.
        * Leftface is a face on left side of edge.
        * \param originKey Is a unique integer value (key) associated with origin vertex of edge.
        * \param twinOriginKey Is a unique integer value (key) associated with origin vertex of twin edge.
        * \param face Face on left side of edge.
        * \param twinFace Face on left side of twin edge.
        * \return Shared pointer points to edge. Note: Not it's twin edge.
        */

        QSharedPointer<HalfEdge> connectVertices(int originKey, int twinOriginKey, QSharedPointer<Face> face, QSharedPointer<Face> twinFace);

        /*!
        * \brief connectVertices Creates two edges, first one from origin to twinOrigin and assign face as leftface.
        * Second one is a twin edge, from twinOrigin to origin and assign twinFace as leftface.
        * Leftface is a face on left side of edge.
        * \param origin  Origin vertex of edge.
        * \param twinOrigin Origin vertex of twin edge.
        * \param faceKey Is a unique integer value (key) associated with face on left side of edge.
        * \param twinFaceKey Is a unique integer value (key) associated with face on left side of twin edge.
        * \return Shared pointer points to edge. Note: Not it's twin edge.
        */

        QSharedPointer<HalfEdge> connectVertices(QSharedPointer<Vertex> origin, QSharedPointer<Vertex> twinOrigin, int faceKey, int twinFaceKey);

        /*!
        * \brief connectVertices Creates two edges, first one from origin to twinOrigin and assign face as leftface.
        * Second one is a twin edge, from twinOrigin to origin and assign twinFace as leftface.
        * Leftface is a face on left side of edge.
        * \param origin Origin vertex of edge.
        * \param twinOrigin Origin vertex of twin edge.
        * \param face Face on left side of edge.
        * \param twinFace Face on left side of twin edge.
        * \return Shared pointer points to edge. Note: Not it's twin edge.
        */

        QSharedPointer<HalfEdge> connectVertices(QSharedPointer<Vertex> origin, QSharedPointer<Vertex> twinOrigin, QSharedPointer<Face> face, QSharedPointer<Face> twinFace);

        /*!
        * \brief removeEdge Set edge origin incidence to next edge, connected to origin. Set twin edge origin incidence
        * to next twin edge connected to origin. Merge left faces of edge and twin edge. We have a new face now. Set
        * Boundary of new face. Connect the next and previous of those around the removed edge. Now remove edge and its
        * twin edge.
        * \param key Unique id of a edge to remove from graph.
        */

        void removeEdge(int key);

        /*!
        * \brief removeEdge Set edge origin incidence to next edge, connected to origin. Set twin edge origin incidence
        * to next twin edge connected to origin. Merge left faces of edge and twin edge. We have a new face now. Set
        * Boundary of new face. Connect the next and previous of those around the removed edge. Now remove edge and its
        * twin edge.
        * \param edge HalfEdge to remove from graph.
        */

        void removeEdge(QSharedPointer<HalfEdge> edge);



//      void addWeightToHalfEdge(int key, float weight);

//      /*!
//      * \brief addWeightToHalfEdge
//      * \param key
//      * \param weight
//      */

//      void addWeightToHalfEdge(QSharedPointer<HalfEdge> key, float weight);

        /*!
        * \brief splitFace First find the edge that has leftface as face. Create edge and twin edge between u and v. Assign
        * leftface to both edge and its twin edge. Method connectVertices is used to create edge and there face. Set boundary
        * edge for faces created by connectVertices. Set next and previous edges for created edges. Remove input face.
        * \param v origin of halfedge we are using to split.
        * \param u origin of twin of halfedge we are using to split.
        * \param face face we are spliting.
        * \return halfedge. Note: Not twin of HalfEdge.
        */

        QSharedPointer<HalfEdge> splitFace(QSharedPointer<Vertex> v, QSharedPointer<Vertex> u, QSharedPointer<Face> face);

        /*!
        * \brief joinFaces Join face and face2. Find edge shared by face and face2 and remove that edge.
        * \param face Smallest convex piece with edges and vertices in a graph is a face.
        * \param face2 Same definition as face. But face2 is adjacent to a face.
        * \return Pointer points to an face generated, by joining face and face2.
        */

        QSharedPointer<Face> joinFaces(QSharedPointer<Face> face, QSharedPointer<Face> face2);

        /*!
        * \brief addTriangularFace Create face with three edges and three vertices. key1, key2 and key3 are associated with
        * vertices of a face generated.
        * \param key1 Vertex key of a vertex belongs to a face.
        * \param key2 Vertex key of a vertex belongs to a face.
        * \param key3 Vertex Key of a vertex belongs to a face.
        * \return Face generated.
        */

        QSharedPointer<Face> addTriangularFace(int key1, int key2, int key3);

        /*!
        * \brief addTriangularFace Create face with three edges and three vertices v1, v2, v3.
        * \param v1 Vertex.
        * \param v2 Vertex.
        * \param v3 Vertex.
        * \return Face generated.
        */

        QSharedPointer<Face> addTriangularFace(QSharedPointer<Vertex> v1, QSharedPointer<Vertex> v2, QSharedPointer<Vertex> v3);

        /*!
        * \brief getHalfEdge It returns an edge with a unique id key.
        * \param key Unique id of an edge.
        * \return Edge.
        */

        QSharedPointer<HalfEdge> getHalfEdge(int key);
        /*!
        * \brief getHalfEdge It returns an edge with origin vertex key v and target vertex key u.
        * \param v Unique id (key) for vertex. v is origin of an edge.
        * \param u Unique id (key) for vertex. u is target of an edge.
        * \return Edge.
        */
        QSharedPointer<HalfEdge> getHalfEdge(int v, int u);

        /*!
        * \brief getHalfEdge It returns an edge with origin vertex v and target vertex u.
        * \param v origin of an edge.
        * \param u target of an edge.
        * \return Edge.
        */

        QSharedPointer<HalfEdge> getHalfEdge(QSharedPointer<Vertex> v, QSharedPointer<Vertex> u);

        /*!
        * \brief getFace It returns a face with unique id (key) associated with it.
        * \param key Unique id (key) of a face.
        * \return face.
        */

        QSharedPointer<Face> getFace(int key);
        //! \brief Return circle based on circle key.
        /*!
        * \brief getCircle It return circle with unique id (key) associated with vertex.
        * \param key Unique id of a vertex.
        * \return circle
        */
        QSharedPointer<Circle> getCircle(int key);

        /*!
        * \brief getNumVertices It returns number of vertices.
        * \return number of vertices.
        */

        int getNumVertices();

        /*!
        * \brief getNumHalfEdges It returns number of halfedges.
        * \return number of halfedges.
        */

        int getNumHalfEdges();

        /*!
        * \brief getNumEdges It returns number of edges. Number of edges are  half of total number of halfedges.
        * \return number of edges.
        */

        int getNumEdges(); // # of HalfEdges / 2

        /*!
        * \brief getNumFaces It returns number of faces.
        * \return number of faces.
        */

        int getNumFaces();

//      /*!
//      * \brief manageUnhandledTriangles
//      */
//      void manageUnhandledTriangles();
//      /*!
//      * \brief getNumUnhandledTriangles
//      * \return
//      */
//      int getNumUnhandledTriangles();

        /*!
        * \brief clear
        */

        void clear();

//      QSharedPointer<Graph> copy();

        /**
        * remapVertices - Re-maps all the vertices in the graph so that the keys of the
        *     vertices make up all the numbers from 1 to the total number of vertices.
        *     In addition the points and circles associated with each vertex also have
        *     to be re-mapped as well to the new ids.
        */
        void remapVertices();
        /**
        * remapHalfEdges - Re-maps all the halfEdges in the graph so that the keys of
        *     the halfEdges make up all the numbers from 1 to the total number of
        *     halfEdges.
        */
        void remapHalfEdges();
        /**
        * remapFaces - Re-maps all the faces in the graph so that the keys of the faces
        *     make up all the numbers from 1 to the total number of faces.
        */
        void remapFaces();
        /**
        * isBoundary - A vertex is a boundary vertex if it is an end-point of a
        *     boundary edge or if it is an isolated vertex.
        */

        bool isBoundary(QSharedPointer<Vertex> vertex);
        /**
        * isBoundary - A halfEdge is a boundary halfEdge if it or its twin has the
        *     exterior face as its leftFace.
        */

        bool isBoundary(QSharedPointer<HalfEdge> halfEdge);
        /**
        * isBoundary - A face is a boundary face if it has a boundary edge on one
        *     of its boundaries.
        */

        bool isBoundary(QSharedPointer<Face> face);

        //! \brief Set vertex map
        //! Qt's containers are implicitly shared value classes.
        //! Passing them around by pointers is superfluous.
        //void vMap(QMap<int, QSharedPointer<Vertex> > & vMap);

        //! \brief Return vertex map
        //QMap<int, QSharedPointer<Vertex> > * vMap();

        //! \brief return vertex from vmap
        QSharedPointer<Vertex> getVertex(int key);

        //! \brief return vmap size
        //int numVertices();

        //! \brief Bool return to check, if vmap contains vertex
        bool containsVertex(int key);

        //! \brief Begining, End of vMap iterator
        QMap<int, QSharedPointer<Vertex> > ::iterator beginVertices();
        QMap<int, QSharedPointer<Vertex> > ::iterator endVertices();

        void eraseVertex(int key);

        //! \brief Set edge map
        //void eMap(QMap<int, QSharedPointer<HalfEdge> > & eMap);

        //! \brief Return edge map
        //QMap<int, QSharedPointer<HalfEdge> >  eMap();
        //QMap<int, QSharedPointer<HalfEdge> > * eMap();

        //! \brief size of emap
        //int numEdges();

        //! brief Bool return to check, if emap contains edge
        bool containsEdge(int key);

        //! Add a new edge to emap
        void addEdge(int key, QSharedPointer<HalfEdge> edge);

        //! Remove an edge from emap
        void eraseEdge(int key);

        //! \brief Begining, End of eMap iterator
        QMap<int, QSharedPointer<HalfEdge> > ::iterator beginEdges();
        QMap<int, QSharedPointer<HalfEdge> > ::iterator endEdges();

        //! \brief Set circle map associated with vertex
        // void cMap(QMap<int, QSharedPointer<Circle> > & cMap);

        //! \brief Return circle map associated with vertex
        //QMap<int, QSharedPointer<Circle> > * cMap();

        //! \brief Return circle associted with some key in circle map
        //QSharedPointer<Circle> cMap(int key);

        //! \brief erase the circle
        //void eraseCircle(int key);

        //! \brief Set face map
        // void fMap(QMap<int, QSharedPointer<Face> > & fMap);

        //! \brief Return face map
        //QMap<int, QSharedPointer<Face> > * fMap();

        //! brief Bool return to check, if fmap contains face
        bool containsFace(int key);

        //! \brief Begining, End of fMap iterator
        QMap<int, QSharedPointer<Face> > ::iterator beginFaces();

        QMap<int, QSharedPointer<Face> > ::iterator endFaces();

//      //! \brief Set point map associated with vertex
//       void pMap(QMap<int, Point>& pMap);

//      //! \brief Return point map associted with vertex
//      QMap<int, Point>* pMap();

        void erasePoint(int key);

//      //! \brief Set unhandledTriangleCount
//      void unhandledTriangleCount(int unhandled_triangle_count);

//      //! \brief Return unhandledTriangleCount
//      int unhandledTriangleCount();

//      //! \brief Set unhandledTriangle
//      void unhandledTriangles(QList< QSharedPointer<Vertex> >  unhandled_triangles);

//      //! \brief Return unhandledTriangle
//      QList< QSharedPointer<Vertex> >  unhandledTriangles();

        //! \brief Set exterior face
        QSharedPointer<Face> exterior();

        //! \brief Return exterior face
        void exterior(QSharedPointer<Face> exterior);


    private:

        QMap<int, QSharedPointer<Vertex> >  m_vertices;
//      QMap<int, QSharedPointer<Circle> >  m_cMap;
        QMap<int, QSharedPointer<HalfEdge> >  m_edges;
        QMap<int, QSharedPointer<Face> >  m_faces;
//      QMap<int, Point> m_pMap;
//      int m_unhandled_triangles_count;
//      QList< QSharedPointer<Vertex> >  m_unhandled_triangles;
        QSharedPointer<Face> m_exterior;

    };

}
#endif // GRAPH_H
