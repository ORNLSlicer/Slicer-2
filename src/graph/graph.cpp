#include "graph/graph.h"

namespace ORNL
{
    
    Graph::Graph()
    {
        m_exterior=QSharedPointer<Face>(new Face(-1, nullptr));
        m_faces[m_exterior->key()] = m_exterior;
        //m_unhandled_triangles_count = 0;
    }
    
    Graph::~Graph()
    {
        clear();
    }
    
    QSharedPointer<Vertex> Graph::addVertex(int key)
    {
        m_vertices[key] = QSharedPointer<Vertex>(new Vertex(key));
        return m_vertices[key];
    }
    
    void Graph::removeVertex(int key)
    {
        removeVertex(m_vertices[key]);
    }
    
    void Graph::removeVertex(QSharedPointer<Vertex> v)
    {
        QVector< QSharedPointer<HalfEdge> >  inward = v->inEdges();
    
        for (QSharedPointer<HalfEdge> in : inward)
        {
            removeEdge(in);
        }
    
        // Twin taken care of in removeEdge
    
        //pMap.erase(v->key);
        //vMap.erase(v->key);
        //m_pMap.remove(v->key());
        m_vertices.remove(v->key());
        v.clear();
        //delete v;
        //v = nullptr;
    }
    
    //void Graph::setData(int key, int x, int y, int z)
    //{
    //    //  m_pMap[key] = Point(x, y, z);
    //    if(m_vertices[key]->data<VertexLocationCircleData>().isNull())
    //    {
    //     QSharedPointer<Point> point(new Point(x, y, z));
    //     QSharedPointer<VertexLocationCircleData> vertexLocationData(new VertexLocationCircleData(point));
    //     m_vertices[key]->data<VertexLocationCircleData>(vertexLocationData);
    //    }
    //    else
    //    {
    //     QSharedPointer<Point> point(new Point(x, y, z));
    //     m_vertices[key]->data<VertexLocationCircleData>()->m_point = point;
    //    }
    
    //}
    
    //void Graph::setData(int key, Point p)
    //{
    // //    m_pMap[key] = p;
    //    if(m_vertices[key]->data<VertexLocationCircleData>().isNull())
    //    {
    //     QSharedPointer<Point>point(&p);
    //     QSharedPointer<VertexLocationCircleData> vertexLocationData(new VertexLocationCircleData(point));
    //     m_vertices[key]->data<VertexLocationCircleData>(vertexLocationData);
    //    }
    //    else
    //    {
    //     QSharedPointer<Point>point(&p);
    //     m_vertices[key]->data<VertexLocationCircleData>()->m_point = point;
    //    }
    
    //}
    
    void Graph::setData(int key, QSharedPointer<Point> p)
    {
     //   m_pMap[key] = p;
        if(m_vertices[key]->data<VertexLocationCircleData>().isNull())
        {
         QSharedPointer<VertexLocationCircleData> vertexLocationData(new VertexLocationCircleData(p));
         m_vertices[key]->data<VertexLocationCircleData>(vertexLocationData);
        }
        else
        {
         m_vertices[key]->data<VertexLocationCircleData>()->m_point = p;
        }
    }
    
    //void Graph::addPointToVertex(int key, QSharedPointer<VertexData> vertexData)
    //{
    //    //m_pMap[key] = p;
    //    m_vertices[key]->data(vertexData);
    //}
    
    //void Graph::setData(QSharedPointer<Vertex> v, int x, int y, int z)
    //{
    //    //  m_pMap[v->key()] = Point(x, y, z);
    //    if(m_vertices[v->key()]->data<VertexLocationCircleData>().isNull())
    //    {
    //     QSharedPointer<Point> point(new Point(x, y, z));
    //     QSharedPointer<VertexLocationCircleData> vertexLocationData(new VertexLocationCircleData(point));
    //     m_vertices[v->key()]->data<VertexLocationCircleData>(vertexLocationData);
    //    }
    //    else
    //    {
    //     QSharedPointer<Point> point(new Point(x, y, z));
    //     m_vertices[v->key()]->data<VertexLocationCircleData>()->m_point = point;
    //    }
    //}
    
    //void Graph::setData(QSharedPointer<Vertex> v, Point p)
    //{
    //     // m_pMap[v->key()] = p;
    //    if(m_vertices[v->key()]->data<VertexLocationCircleData>().isNull())
    //    {
    //        QSharedPointer<Point>point(&p);
    //        QSharedPointer<VertexLocationCircleData> vertexLocationData(new VertexLocationCircleData(point));
    //        m_vertices[v->key()]->data<VertexLocationCircleData>(vertexLocationData);
    //    }
    //    else
    //    {
    //        QSharedPointer<Point>point(&p);
    //        m_vertices[v->key()]->data<VertexLocationCircleData>()->m_point = point;
    //    }
    //}
    
    void Graph::setData(int key, QSharedPointer<Circle> c)
    {
    //    m_cMap[key] = c;
        if(m_vertices[key]->data<VertexLocationCircleData>().isNull())
        {
            QSharedPointer<VertexLocationCircleData> vertexCircleData(new VertexLocationCircleData(c));
            m_vertices[key]->data<VertexLocationCircleData>(vertexCircleData);
        }
        else
        {
            m_vertices[key]->data<VertexLocationCircleData>()->m_circle = c;
        }
    }
    
    void Graph::setData(QSharedPointer<Vertex> v, QSharedPointer<Circle> c)
    {
        //  m_cMap[v->key()] = c;
        if(m_vertices[v->key()]->data<VertexLocationCircleData>().isNull())
        {
            QSharedPointer<VertexLocationCircleData> vertexCircleData(new VertexLocationCircleData(c));
            m_vertices[v->key()]->data<VertexLocationCircleData>(vertexCircleData);
        }
        else
        {
            m_vertices[v->key()]->data<VertexLocationCircleData>()->m_circle = c;
        }
    
       //  QSharedPointer<VertexCircleData> vertexCircleData(new VertexCircleData(c));
       //  m_vertices[v->key()]->data<VertexCircleData>(vertexCircleData);
    }
    
    QSharedPointer<Face> Graph::addFace(int key, QSharedPointer<HalfEdge> edge)
    {
        m_faces[key] = QSharedPointer<Face>(new Face(key, edge));
        return m_faces[key];
    }
    
    QSharedPointer<Face> Graph::addFace(QSharedPointer<HalfEdge> edge)
    {
        int key = 0;
        while (m_faces.find(key) != m_faces.end()) key++;
        return addFace(key, edge);
    }
    
    void Graph::removeFace(int key)
    {
        return removeFace(m_faces[key]);
    }
    
    void Graph::removeFace(QSharedPointer<Face> face)
    {
        FaceEdgeIterator it(face);
        while (it.hasNext())
        {
            it.getNext()->leftFace(m_exterior);
        }
    
        m_faces.remove(face->key());
        face.clear();
        //delete face;
        //face = nullptr;
    }
    
    QSharedPointer<HalfEdge> Graph::connectVertices(int originKey, int twinOriginKey, int faceKey, int twinFaceKey)
    {
        return connectVertices(m_vertices[originKey], m_vertices[twinOriginKey], m_faces[faceKey], m_faces[twinFaceKey]);
    }
    
    QSharedPointer<HalfEdge> Graph::connectVertices(int originKey, int twinOriginKey, QSharedPointer<Face> face, QSharedPointer<Face> twinFace)
    {
        return connectVertices(m_vertices[originKey], m_vertices[twinOriginKey], face, twinFace);
    }
    
    QSharedPointer<HalfEdge> Graph::connectVertices(QSharedPointer<Vertex> origin, QSharedPointer<Vertex> twinOrigin, int faceKey, int twinFaceKey)
    {
        return connectVertices(origin, twinOrigin, m_faces[faceKey], m_faces[twinFaceKey]);
    }
    
    QSharedPointer<HalfEdge> Graph::connectVertices(QSharedPointer<Vertex> origin, QSharedPointer<Vertex> twinOrigin, QSharedPointer<Face> face, QSharedPointer<Face> twinFace)
    {
        QSharedPointer<HalfEdge> h1(new HalfEdge());
    
        int key = 0;
    
        while (m_edges.find(key) != m_edges.end()) key++;
        h1->key(key);
    
        m_edges[key] = h1;
    
        h1->origin(origin);
        h1->leftFace(face);
        face->boundary(h1);
        origin->incidentEdge(h1);
    
        QSharedPointer<HalfEdge> h2(new HalfEdge());
        while (m_edges.find(key) != m_edges.end()) key++;
        h2->key(key);
    
        h2->setTwin(h1);
        h2->origin(twinOrigin);
        h2->leftFace(twinFace);
        twinFace->boundary(h2);
        twinOrigin->incidentEdge(h2);
    
        m_edges[key] = h2;
    
        return h1;
    }
    
    void Graph::removeEdge(int key)
    {
        return removeEdge(m_edges[key]);
    }
    
    void Graph::removeEdge(QSharedPointer<HalfEdge> edge)
    {
        // Set new origin incident edge
        if (edge->origin()->incidentEdge() == edge)
        {
            VertexEdgeIterator it(edge->origin());
            QSharedPointer<HalfEdge> newIncident = it.getNext();
            while (it.hasNext() && newIncident == edge)
            {
                newIncident = it.getNext();
            }
    
            if (newIncident == edge)
            {
                edge->origin()->incidentEdge(NULL);
            }
            else
            {
                edge->origin()->incidentEdge(newIncident);
            }
        }
    
        // Set new twin origin incident edge
        if (edge->twin()->origin()->incidentEdge() == edge->twin())
        {
            VertexEdgeIterator it(edge->twin()->origin());
            QSharedPointer<HalfEdge> newIncident = it.getNext();
            while (it.hasNext() && newIncident == edge->twin())
            {
                newIncident = it.getNext();
            }
    
            if (newIncident == edge->twin())
            {
                edge->twin()->origin()->incidentEdge(NULL);
            }
            else
            {
                edge->twin()->origin()->incidentEdge(newIncident);
            }
        }
    
        // merge f2 (twin->leftFace) into f1 (edge->leftFace)
        FaceEdgeIterator it(edge->twin()->leftFace());
        while (it.hasNext())
        {
            it.getNext()->leftFace(edge->leftFace());
        }
    
        // Connect the next and previous of those around the removed edge
        edge->next()->setPrevious(edge->twin()->previous());
        edge->twin()->next()->setPrevious(edge->previous());
    
        // Set the face boundary
        if (edge->leftFace()->boundary() == edge)
        {
            FaceEdgeIterator it(edge->leftFace());
            QSharedPointer<HalfEdge> newBoundary = it.getNext();
            while (it.hasNext() && newBoundary == edge)
            {
                newBoundary = it.getNext();
            }
            if (newBoundary == edge)
            {
                edge->leftFace()->boundary(nullptr);
            }
            else
            {
                edge->leftFace()->boundary(newBoundary);
            }
        }
    
        // Remove edge and twin from map
        m_edges.remove(edge->key());
        m_edges.remove(edge->twin()->key());
    
        // delete edge and twin
        edge->twin().clear();
        //edge->twin() = nullptr;
        edge.clear();
        //edge = nullptr;
    }
    
    //void Graph::addWeightToHalfEdge(int key, float weight)
    //{
    //    m_edges[key]->weight(weight);
    //}
    
    //void Graph::addWeightToHalfEdge(QSharedPointer<HalfEdge> e, float weight)
    //{
    //    e->weight(weight);
    //}
    
    QSharedPointer<HalfEdge> Graph::splitFace(QSharedPointer<Vertex> v, QSharedPointer<Vertex> u, QSharedPointer<Face> face)
    {
    
        QSharedPointer<HalfEdge> h = u->incidentEdge();
        if (h->leftFace() != face)
        {
            VertexEdgeIterator it(u);
            while (it.hasNext() && h->leftFace() != face)
            {
                h = it.getNext();
            }
        }
    
    
        QSharedPointer<Face> f1(addFace(NULL));
        QSharedPointer<Face> f2(addFace(NULL));
    
        QSharedPointer<HalfEdge> h1 = connectVertices(u, v, f1, f2);
        QSharedPointer<HalfEdge> h2 = h1->twin();
        f1->boundary(h1);
        f2->boundary(h2);
        h2->setNext(h->next());
        h1->setPrevious(h);
    
        QSharedPointer<HalfEdge> i = h2;
        while (true)
        {
            i->leftFace(f2);
            if (i->target() == v)
            {
                break;
            }
            i = i->next();
        }
    
        h1->setNext(i->next());
        i->setNext(h2);
        i = h1;
        do
        {
            i->leftFace(f1);
            i = i->next();
        } while (i->target() != u);
    
        m_faces.remove(face->key());
        face.clear();
        //delete face;
    
        return h1;
    }
    
    QSharedPointer<Face> Graph::joinFaces(QSharedPointer<Face> face, QSharedPointer<Face> face2)
    {
        FaceEdgeIterator it(face);
        FaceEdgeIterator it2(face2);
        QSharedPointer<HalfEdge> edge = it.getNext();
        while (it.hasNext())
        {
            while (it2.hasNext())
            {
                if (edge == it2.getNext())
                {
                    removeEdge(edge);
                    break;
                }
            }
            it2.reset();
            edge = it.getNext();
        }
        return face;
    }
    
    QSharedPointer<Face> Graph::addTriangularFace(int v1, int v2, int v3)
    {
        return addTriangularFace(m_vertices[v1], m_vertices[v2], m_vertices[v3]);
    }
    
    QSharedPointer<Face> Graph::addTriangularFace(QSharedPointer<Vertex> v1, QSharedPointer<Vertex> v2, QSharedPointer<Vertex> v3)
    {
        QSharedPointer<HalfEdge> e1 = sharedFromThis()->getHalfEdge(v1, v2);
        QSharedPointer<HalfEdge> e2 = sharedFromThis()->getHalfEdge(v2, v3);
        QSharedPointer<HalfEdge> e3 = sharedFromThis()->getHalfEdge(v3, v1);
    
        int unusedVertices = 0;
        if (v1->incidentEdge() == NULL) unusedVertices++;
        if (v2->incidentEdge() == NULL) unusedVertices++;
        if (v3->incidentEdge() == NULL) unusedVertices++;
    
        int readyEdges = 0;
        if (e1 != NULL) readyEdges++;
        if (e2 != NULL) readyEdges++;
        if (e3 != NULL) readyEdges++;
    
        QSharedPointer<Face> face = NULL;
    
        //the most simple case, all vertices has degree 0
        //create 3 edges, and link then
        if (unusedVertices == 3 && readyEdges == 0)
        {
            face = addFace(NULL);
    
            e1 = connectVertices(v1, v2, face, m_exterior);
            v1->incidentEdge(e1);
            face->boundary(e1);
    
            e2 = connectVertices(v2, v3, face, m_exterior);
            v2->incidentEdge(e2);
    
            e3 = connectVertices(v3, v1, face, m_exterior);
            v3->incidentEdge(e3);
    
            e1->setNext(e2);
            e2->setNext(e3);
            e3->setNext(e1);
    
            e1->twin()->setNext(e3->twin());
            e3->twin()->setNext(e2->twin());
            e2->twin()->setNext(e1->twin());
        }
    
        //there are one vertex that has been used by another triangle.
        //Create the 3 edges, and link to the old triangle.
        else if (unusedVertices == 2 && readyEdges == 0)
        {
            if (v2->incidentEdge() != NULL)
            {
                QSharedPointer<Vertex> vt = v1;
                v1 = v2;
                v2 = v3;
                v3 = vt;
            }
            else if (v3->incidentEdge() != NULL)
            {
                QSharedPointer<Vertex> vt = v1;
                v1 = v3;
                v3 = v2;
                v2 = vt;
            }
            QSharedPointer<HalfEdge> ei1 = v1->incidentEdge();
            if (ei1 != NULL)
            {
                QSharedPointer<HalfEdge> ei1n = ei1->next();
    
                face = addFace(NULL);
    
                e1 = connectVertices(v1, v2, face, ei1->leftFace());
                face->boundary(e1);
    
                e2 = connectVertices(v2, v3, face, ei1->leftFace());
                v2->incidentEdge(e2);
    
                e3 = connectVertices(v3, v1, face, ei1->leftFace());
                v3->incidentEdge(e3);
    
                e1->setNext(e2);
                e2->setNext(e3);
                e3->setNext(e1);
    
                ei1->setNext(e3->twin());
                e3->twin()->setNext(e2->twin());
                e2->twin()->setNext(e1->twin());
                e1->twin()->setNext(ei1n);
            }
        }
    
        //2 vertices are already in use by 2 distinct triangles, and
        //one vertex doesn't have been used yet.
        else if (unusedVertices == 1 && readyEdges == 0)
        {
            if (v3->incidentEdge() == NULL)
            {
                QSharedPointer<Vertex> vt = v1;
                v1 = v2;
                v2 = v3;
                v3 = vt;
            }
            else if (v1->incidentEdge() == NULL)
            {
                QSharedPointer<Vertex> vt = v1;
                v1 = v3;
                v3 = v2;
                v2 = vt;
            }
    
            QSharedPointer<HalfEdge> ei1 = v1->incidentEdge();
            QSharedPointer<HalfEdge> ei3 = v3->incidentEdge();
            if (ei1 != NULL && ei3 != NULL)
            {
                QSharedPointer<HalfEdge> ei1n = ei1->next();
                QSharedPointer<HalfEdge> ei3n = ei3->next();
    
                face = addFace(NULL);
    
                e1 = connectVertices(v1, v2, face, ei1n->leftFace());
                e2 = connectVertices(v2, v3, face, ei3->leftFace());
                e3 = connectVertices(v3, v1, face, ei1->leftFace());
    
                face->boundary(e1);
                v2->incidentEdge(e2);
    
                e1->setNext(e2);
                e2->setNext(e3);
                e3->setNext(e1);
    
                ei1->setNext(e3->twin());
                e3->twin()->setNext(ei3n);
    
                ei3->setNext(e2->twin());
                e2->twin()->setNext(e1->twin());
                e1->twin()->setNext(ei1n);
            }
        }
    
        //two of the vertices already have one edge connected between then,
        //and one vertex has not been used yet.
        else if (unusedVertices == 1 && readyEdges == 1)
        {
            //rotate pointers, so the v1 and v2 are connected, and v3 is unused
            if (e2 != NULL)
            {
                QSharedPointer<Vertex> vt = v1;
                v1 = v2;
                v2 = v3;
                v3 = vt;
                QSharedPointer<HalfEdge> et = e1;
                e1 = e2;
                e2 = e3;
                e3 = et;
            }
            else if (e3 != NULL)
            {
                QSharedPointer<Vertex> vt = v1;
                v1 = v3;
                v3 = v2;
                v2 = vt;
                QSharedPointer<HalfEdge> et = e1;
                e1 = e3;
                e3 = e2;
                e2 = et;
            }
    
            if (e1->leftFace() == NULL)
            {
                QSharedPointer<HalfEdge> e1n = e1->next();
                QSharedPointer<HalfEdge> e1p = e1->previous();
    
                face = addFace(e1);
    
                e2 = connectVertices(v2, v3, face, e1n->leftFace());
                e3 = connectVertices(v3, v1, face, e1p->leftFace());
    
                e1->leftFace(face);
                v3->incidentEdge(e3);
    
                e1->setNext(e2);
                e2->setNext(e3);
                e3->setNext(e1);
    
                e1p->setNext(e3->twin());
                e3->twin()->setNext(e2->twin());
                e2->twin()->setNext(e1n);
            }
        }
    
        //all the tree vertices are used by different triangles, no
        //ready edge available.
        else if (unusedVertices == 0 && readyEdges == 0)
        {
            QSharedPointer<HalfEdge> ei1 = v1->incidentEdge();
            QSharedPointer<HalfEdge> ei2 = v2->incidentEdge();
            QSharedPointer<HalfEdge> ei3 = v3->incidentEdge();
    
            if (ei1 != NULL && ei2 != NULL && ei3 != NULL)
            {
    
                QSharedPointer<HalfEdge> ei1n = ei1->next();
                QSharedPointer<HalfEdge> ei2n = ei2->next();
                QSharedPointer<HalfEdge> ei3n = ei3->next();
    
                face = addFace(NULL);
    
                e1 = connectVertices(v1, v2, face, ei2->leftFace());
                e2 = connectVertices(v2, v3, face, ei3->leftFace());
                e3 = connectVertices(v3, v1, face, ei1->leftFace());
    
                face->boundary(e1);
    
                e1->setNext(e2);
                e2->setNext(e3);
                e3->setNext(e1);
    
                ei1->setNext(e3->twin());
                e3->twin()->setNext(ei3n);
    
                ei3->setNext(e2->twin());
                e2->twin()->setNext(ei2n);
    
                ei2->setNext(e1->twin());
                e1->twin()->setNext(ei1n);
            }
        }
    
        //all vertices are used, and two of then are used by the
        //same triangle. In this case, one edge will be shared with
        //other triangle.
        else if (unusedVertices == 0 && readyEdges == 1)
        {
            if (e2 != NULL)
            {
                QSharedPointer<Vertex> vt = v1;
                v1 = v2;
                v2 = v3;
                v3 = vt;
                QSharedPointer<HalfEdge> et = e1;
                e1 = e2;
                e2 = e3;
                e3 = et;
            }
            else if (e3 != NULL)
            {
                QSharedPointer<Vertex> vt = v1;
                v1 = v3;
                v3 = v2;
                v2 = vt;
                QSharedPointer<HalfEdge> et = e1;
                e1 = e3;
                e3 = e2;
                e2 = et;
            }
    
            QSharedPointer<HalfEdge> ei3 = v3->incidentEdge();
            if (ei3 && e1->leftFace() == NULL)
            {
                QSharedPointer<HalfEdge> e1p = e1->previous();
                QSharedPointer<HalfEdge> e1n = e1->next();
                QSharedPointer<HalfEdge> ei3n = ei3->next();
    
                face = addFace(e1);
    
                e2 = connectVertices(v2, v3, face, ei3->leftFace());
                e3 = connectVertices(v3, v1, face, e1p->leftFace());
    
                e1->leftFace(face);
                e2->leftFace(face);
                e3->leftFace(face);
    
                e1->setNext(e2);
                e2->setNext(e3);
                e3->setNext(e1);
    
                e1p->setNext(e3->twin());
                e3->twin()->setNext(ei3n);
    
                ei3->setNext(e2->twin());
                e2->twin()->setNext(e1n);
            }
        }
    
        //The three vertices are used, but there is an edge missing.
        //will create one edge to 'fill' the hole and expand the mesh
        else if (unusedVertices == 0 && readyEdges == 2)
        {
            if (e1 == NULL)
            {
                QSharedPointer<Vertex> vt = v1;
                v1 = v3;
                v3 = v2;
                v2 = vt;
                QSharedPointer<HalfEdge> et = e1;
                e1 = e3;
                e3 = e2;
                e2 = et;
            }
            else if (e3 == NULL)
            {
                QSharedPointer<Vertex> vt = v1;
                v1 = v2;
                v2 = v3;
                v3 = vt;
                QSharedPointer<HalfEdge> et = e1;
                e1 = e2;
                e2 = e3;
                e3 = et;
            }
    
            if (e1->leftFace() == NULL && e3->leftFace() == NULL)
            {
                QSharedPointer<HalfEdge> e1n = e1->next();
                QSharedPointer<HalfEdge> e3p = e3->previous();
    
                if (e3->next() == e1 && e1->previous() == e3)
                {
                    face = addFace(e1);
    
                    e2 = connectVertices(v2, v3, face, e1n->leftFace());
    
                    e1->leftFace(face);
                    e3->leftFace(face);
    
                    e1->setNext(e2);
                    e2->setNext(e3);
                    e3->setNext(e1);
    
                    e3p->setNext(e2->twin());
                    e2->twin()->setNext(e1n);
                }
                else
                {
                    qDebug() << "outro erro" << endl;
                }
            }
        }
    
        //A practically closed face, need only that the internal edges
        //point to a new face. A easy case =)
        else if (unusedVertices == 0 && readyEdges == 3)
        {
            if (e1->leftFace() == NULL && e2->leftFace() == NULL && e3->leftFace() == NULL)
            {
                if (e1->next() == e2 && e2->next() == e3 && e3->next() == e1)
                {
                    face = addFace(e1);
                    e1->leftFace(face);
                    e2->leftFace(face);
                    e3->leftFace(face);
                }
                else
                {
                    qDebug() << "mais outro erro" << endl;
                }
            }
        }
    
        return face;
    }
    
    //QSharedPointer<Vertex> Graph::getVertex(int key)
    //{
    //    if (m_vertices.find(key) != m_vertices.end())
    //    {
    //        return m_vertices[key];
    //    }
    //    return nullptr;
    //}
    
    QSharedPointer<HalfEdge> Graph::getHalfEdge(int key)
    {
        if (m_edges.find(key) != m_edges.end())
        {
            return m_edges[key];
        }
    
        return nullptr;
    }
    
    QSharedPointer<HalfEdge> Graph::getHalfEdge(int v, int u)
    {
        return getHalfEdge(m_vertices[v], m_vertices[u]);
    }
    
    QSharedPointer<HalfEdge> Graph::getHalfEdge(QSharedPointer<Vertex> v, QSharedPointer<Vertex> u)
    {
        VertexEdgeIterator it(v);
        while (it.hasNext())
        {
            QSharedPointer<HalfEdge> edge = it.getNext();
            if (edge->target() == u)
            {
                return edge;
            }
        }
        return nullptr;
    }
    
    QSharedPointer<Face> Graph::getFace(int key)
    {
        if (m_faces.find(key) != m_faces.end())
        {
            return m_faces[key];
        }
    
        return nullptr;
    }
    
    int Graph::getNumVertices()
    {
        return m_vertices.size();
    }
    
    int Graph::getNumHalfEdges()
    {
        return m_edges.size();
    }
    
    int Graph::getNumEdges() // # of HalfEdges / 2
    {
        return m_edges.size();
    }
    
    int Graph::getNumFaces()
    {
        return m_faces.size();
    }
    
    QSharedPointer<Circle> Graph::getCircle(int key)
    {
        QSharedPointer<VertexLocationCircleData> data = m_vertices[key]->data<VertexLocationCircleData>();
    
        if (!data.isNull())
        {
            return m_vertices[key]->data<VertexLocationCircleData>()->m_circle;
        }
    
        return nullptr;
    }
    
    //QSharedPointer<Circle> Graph::getCircle(int key)
    //{
    //    if (m_cMap.find(key) != m_cMap.end())
    //    {
    //        return m_cMap[key];
    //    }
    
    //    return nullptr;
    //}
    
    //void Graph::removeCircle(int key)
    //{
    //    QSharedPointer<VertexLocationCircleData> data = m_vertices[key]->data<VertexLocationCircleData>();
    
    //    if(!data.isNull())
    //    {
    //        QMap<int, QSharedPointer<Vertex> > ::iterator it = m_vertices.find(key);
    
    //        if (it == m_vertices.end())
    //        {
    //            qDebug() << "There is no circle associated with that id." << endl;
    //            return;
    //        }
    
    //        //QSharedPointer<Circle> circle = it.value();
    //        m_vertices.remove(key);
    //        //circle.clear();
    //    }
    //    //delete circle;
    //}
    
    //void Graph::removeCircle(int key)
    //{
    //    QMap<int, QSharedPointer<Circle> > ::iterator it = m_cMap.find(key);
    
    //    if (it == m_cMap.end())
    //    {
    //        qDebug() << "There is no circle associated with that id." << endl;
    //        return;
    //    }
    
    //    QSharedPointer<Circle> circle = it.value();
    //    m_cMap.remove(key);
    //    circle.clear();
    //    //delete circle;
    //}
    
    //void Graph::manageUnhandledTriangles()
    //{
    //    if (m_unhandled_triangles_count * 3 != m_unhandled_triangles.size())
    //    {
    //        qDebug() << "The number of unhandled triangles should be equal to the number of unhandled vertices*3!" << endl;
    //        return;
    //    }
    
    //    unsigned int curTriangle = 0;
    //    while (!m_unhandled_triangles.empty())
    //    {
    //        if (curTriangle == m_unhandled_triangles_count)
    //        {
    //            unsigned int remainingTriangles = m_unhandled_triangles.size() / 3;
    //            if (remainingTriangles < m_unhandled_triangles_count)
    //            {
    //                m_unhandled_triangles_count = remainingTriangles;
    //                curTriangle = 0;
    //            }
    //            else
    //            {
    //                qDebug() << "There are triangles that cannot be added to the mesh!" << endl;
    //                return;
    //            }
    //        }
    
    //        QSharedPointer<Vertex> v1 = m_unhandled_triangles.front();
    //        m_unhandled_triangles.pop_front();
    //        QSharedPointer<Vertex> v2 = m_unhandled_triangles.front();
    //        m_unhandled_triangles.pop_front();
    //        QSharedPointer<Vertex> v3 = m_unhandled_triangles.front();
    //        m_unhandled_triangles.pop_front();
    
    //        QSharedPointer<Face> face = addTriangularFace(v1, v2, v3);
    //        if (face == NULL)
    //        {
    //            m_unhandled_triangles_count--;
    //        }
    
    //        curTriangle++;
    //    }
    //}
    
    //int Graph::getNumUnhandledTriangles()
    //{
    //    return m_unhandled_triangles_count;
    //}
    
    void Graph::clear()
    {
        m_vertices.clear();
        m_edges.clear();
        m_faces.clear();
        //m_pMap.clear();
        //m_cMap.clear();
        //m_unhandled_triangles.clear();
        //m_unhandled_triangles_count = 0;
    }
    
    //QSharedPointer<Graph> Graph::copy()
    //{
    //    QSharedPointer<Graph> newGraph(new Graph());
    //    for (QMap<int, QSharedPointer<Vertex> > ::iterator it = m_vertices.begin(); it != m_vertices.end(); it++)
    //    {
    //        newGraph->addVertex(it.key());
    
    //        QSharedPointer<VertexLocationData> data = it.value()->data<VertexLocationData>();
    
    //        if (!data.isNull())
    //        {
    //            newGraph->addPointToVertex(it.key(), it.value()->data());
    //        }
    
    // //        if (m_pMap.find(it.key()) != m_pMap.end())
    // //        {
    // //            newGraph->addPointToVertex(it.key(), m_pMap[it.key()]);
    // //        }
    
    //        if (m_cMap.find(it.key()) != m_cMap.end())
    //        {
    //            newGraph->addCircleToVertex(it.key(), m_cMap[it.key()]);
    //        }
    //    }
    
    //    for (QMap<int, QSharedPointer<Face> > ::iterator it = m_faces.begin(); it != m_faces.end(); it++)
    //    {
    //        newGraph->m_faces[it.key()] = QSharedPointer<Face>(new Face(it.key(), nullptr));
    //    }
    
    //    for (QMap<int, QSharedPointer<HalfEdge> > ::iterator it = m_edges.begin(); it != m_edges.end(); it++)
    //    {
    //        newGraph->connectVertices(it.value()->origin()->key(), it.value()->target()->key(), it.value()->leftFace()->key(), it.value()->twin()->leftFace()->key());
    //    }
    
    //    for (QMap<int, QSharedPointer<Vertex> > ::iterator it = m_vertices.begin(); it != m_vertices.end(); it++)
    //    {
    //        newGraph->m_vertices[it.key()]->incidentEdge(newGraph->m_edges[it.value()->incidentEdge()->key()]);
    //    }
    
    //    for (QMap<int, QSharedPointer<HalfEdge> > ::iterator it = m_edges.begin(); it != m_edges.end(); it++)
    //    {
    //        newGraph->m_edges[it.key()]->setTwin(newGraph->m_edges[it.value()->twin()->key()]);
    //        newGraph->m_edges[it.key()]->setNext(newGraph->m_edges[it.value()->next()->key()]);
    //    }
    
    //    return newGraph;
    //}
    
    /**
     * remapVertices - Re-maps all the vertices in the graph so that the keys of the
     *     vertices make up all the numbers from 1 to the total number of vertices.
     *     In addition the points and circles associated with each vertex also have
     *     to be re-mapped as well to the new ids.
     */
    void Graph::remapVertices()
    {
        QMap<int, QSharedPointer<Vertex> >  newVMap;
        QMap<int, Point> newPMap;
        QMap<int, QSharedPointer<Circle> >  newCMap;
    
        int key = 0;
        for (QMap<int, QSharedPointer<Vertex> > ::iterator it = m_vertices.begin(); it!=m_vertices.end(); it++)
        {
            newVMap[key] = it.value();
            it.value()->key(key);
            QSharedPointer<VertexLocationCircleData> data = it.value()->data<VertexLocationCircleData>();
    
            if (!data.isNull())
            {
                newVMap[key]->data(it.value()->data());
            }
    
    //        if (m_pMap.find(it.key()) != m_pMap.end())
    //        {
    //            newPMap[key] = m_pMap[it.key()];
    //        }
    //        if (it.value()->vertexLocatCircData()->m_circle)
    //        {
    //            newVMap[key]->vertexLocatCircData()->setData(it.value()->vertexLocatCircData()->m_circle);
    //        }
    
    //        if (m_cMap.find(it.key()) != m_cMap.end())
    //        {
    //            newCMap[key] = m_cMap[it.key()];
    //        }
            key++;
        }
    
        m_vertices.clear();
        m_vertices = newVMap;
    //    m_pMap.clear();
    //    m_pMap = newPMap;
    //    m_cMap.clear();
    //    m_cMap = newCMap;
    
    }
    
    /**
     * remapHalfEdges - Re-maps all the halfEdges in the graph so that the keys of
     *     the halfEdges make up all the numbers from 1 to the total number of
     *     halfEdges.
     */
    void Graph::remapHalfEdges()
    {
        QMap<int, QSharedPointer<HalfEdge> >  newEMap;
    
        int key = 0;
        for (QMap<int, QSharedPointer<HalfEdge> > ::iterator it = m_edges.begin(); it != m_edges.end(); it++)
        {
            newEMap[key] = it.value();
            it.value()->key(key++);
        }
    
        m_edges.clear();
        m_edges = newEMap;
    }
    
    /**
     * remapFaces - Re-maps all the faces in the graph so that the keys of the faces
     *     make up all the numbers from 1 to the total number of faces.
     */
    void Graph::remapFaces()
    {
        QMap<int, QSharedPointer<Face> >  newFMap;
    
        newFMap[m_exterior->key()] = m_exterior; // Exterior is a special case.
    
        int key = 0;
        for (QMap<int, QSharedPointer<Face> > ::iterator it = m_faces.begin(); it != m_faces.end(); it++)
        {
            if (it.value() != m_exterior)
            {
                newFMap[key] = it.value();
                it.value()->key(key++);
            }
        }
    
        m_faces.clear();
        m_faces = newFMap;
    }
    
    /**
     * isBoundary - A halfEdge is a boundary halfEdge if it or its twin has the
     *     exterior face as its leftFace.
     */
    bool Graph::isBoundary(QSharedPointer<HalfEdge> halfEdge)
    {
        return halfEdge->leftFace() == m_exterior;
    }
    
    /**
     * isBoundary - A vertex is a boundary vertex if it is an end-point of a
     *     boundary edge or if it is an isolated vertex.
     */
    bool Graph::isBoundary(QSharedPointer<Vertex>vertex)
    {
        if (vertex->incidentEdge() == nullptr) // Isolated vertex.
            return true;
    
        for (QSharedPointer<HalfEdge> edge : vertex->outEdges()) // Check its edges.
            if (isBoundary(edge))
                return true;
    
        return false;
    }
    
    /**
     * isBoundary - A face is a boundary face if it has a boundary edge on one
     *     of its boundaries.
     */
    bool Graph::isBoundary(QSharedPointer<Face> face)
    {
        for (QSharedPointer<HalfEdge> edge : face->getBoundaryEdges())
            if (isBoundary(edge))
                return true;
    
        return false;
    }
    
    //void Graph::vMap(QMap<int, QSharedPointer<Vertex> > & vMap)
    //{
    //    m_vertices = vMap;
    //}
    
    QSharedPointer<Vertex> Graph::getVertex(int key)
    {
        if(m_vertices.contains(key)) return m_vertices[key];
        else return nullptr;
    }
    
    //QMap<int, QSharedPointer<Vertex> > * Graph::vMap()
    //{
    //    return &m_vertices;
    //}
    
    //int Graph::sizevMap()
    //{
    //    return m_vertices.size();
    //}
    
    bool Graph::containsVertex(int key)
    {
        return m_vertices.contains(key);
    }
    
    void Graph::eraseVertex(int key)
    {
    
        m_vertices.remove(key);
    }
    
    QMap<int, QSharedPointer<Vertex> > ::iterator Graph::beginVertices()
    {
        return m_vertices.begin();
    }
    
    QMap<int, QSharedPointer<Vertex> > ::iterator Graph::endVertices()
    {
        return m_vertices.end();
    }
    
    
    
    //void Graph::eMap(QMap<int, QSharedPointer<HalfEdge> > & eMap)
    //{
    //    m_edges = eMap;
    //}
    
    //QMap<int, QSharedPointer<HalfEdge> >  Graph::eMap()
    //{
    //    return m_edges;
    //}
    
    //QMap<int, QSharedPointer<HalfEdge> > * Graph::eMap()
    //{
    //    return &m_edges ;
    //}
    
    bool Graph::containsEdge(int key)
    {
        return m_edges.contains(key);
    }
    
    //int Graph::sizeeMap()
    //{
    //    return m_edges.size();
    //}
    
    void Graph::addEdge(int key, QSharedPointer<HalfEdge> edge)
    {
        m_edges[key] = edge;
    }
    
    void Graph::eraseEdge(int key)
    {
        m_edges.remove(key);
    }
    
    QMap<int, QSharedPointer<HalfEdge> > ::iterator Graph::beginEdges()
    {
        return m_edges.begin();
    }
    
    
    QMap<int, QSharedPointer<HalfEdge> > ::iterator Graph::endEdges()
    {
        return m_edges.end();
    }
    
    //void Graph::cMap(QMap<int, QSharedPointer<Circle> > & cMap)
    //{
    //    m_cMap = cMap;
    //}
    
    //QMap<int, QSharedPointer<Circle> > * Graph::cMap()
    //{
    //    return &m_cMap;
    //}
    //QSharedPointer<Circle> Graph::cMap(int key)
    //{
    //    return m_cMap[key];
    //}
    
    //void Graph::eraseCircle(int key)
    //{
    //    m_cMap.remove(key);
    //}
    
    
    //void Graph::fMap(QMap<int, QSharedPointer<Face> > & fMap)
    //{
    //    m_faces = fMap;
    //}
    
    //QMap<int, QSharedPointer<Face> > * Graph::fMap()
    //{
    //    return &m_faces;
    //}
    
    bool Graph::containsFace(int key)
    {
        return m_faces.contains(key);
    }
    
    QMap<int, QSharedPointer<Face> > ::iterator Graph::beginFaces()
    {
        return m_faces.begin();
    }
    QMap<int, QSharedPointer<Face> > ::iterator Graph::endFaces()
    {
        return m_faces.end();
    }
    
    //void Graph::pMap(QMap<int, Point>& pMap)
    //{
    //    m_pMap = pMap;
    //}
    
    //QMap<int, Point>* Graph::pMap()
    //{
    //    return &m_pMap;
    //}
    
    void Graph::erasePoint(int key)
    {
    //    m_pMap.remove(key);
    
        if(m_vertices.contains(key))
        {
            QSharedPointer<VertexLocationData> data = m_vertices[key]->data<VertexLocationData>();
    
                if(!data.isNull())
                {
                    data.clear();
                }
        }
    
    }
    
    //void Graph::unhandledTriangleCount(int unhandled_triangle_count)
    //{
    //    m_unhandled_triangles_count = unhandled_triangle_count;
    //}
    
    //int Graph::unhandledTriangleCount()
    //{
    //    return m_unhandled_triangles_count;
    //}
    
    //void Graph::unhandledTriangles(QList< QSharedPointer<Vertex> >  unhandled_triangles)
    //{
    //    m_unhandled_triangles = unhandled_triangles;
    //}
    
    //QList< QSharedPointer<Vertex> >  Graph::unhandledTriangles()
    //{
    //    return m_unhandled_triangles;
    //}
    
    QSharedPointer<Face> Graph::exterior()
    {
        return m_exterior;
    }
    
    void Graph::exterior(QSharedPointer<Face> exterior)
    {
        m_exterior = exterior;
    }
    
}
