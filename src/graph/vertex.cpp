#include "graph/vertex.h"

namespace ORNL 
{
    
    
    Vertex::Vertex(int key)
    {
        m_key=key;
        m_incident_edge = nullptr;
        m_vertex_data = nullptr;
    }
    
    int Vertex::inDegrees()
    {
        return inEdges().size();
    }
    
    int Vertex::outDegrees()
    {
        return outEdges().size();
    }
    
    QVector< QSharedPointer<HalfEdge> >  Vertex::inEdges()
    {
        QVector< QSharedPointer<HalfEdge> >  inEdges;
        VertexEdgeIterator it(sharedFromThis());
        while (it.hasNext())
        {
            inEdges.push_back(it.getNext()->twin());
        }
        return inEdges;
    }
    
    QVector< QSharedPointer<HalfEdge> >  Vertex::outEdges()
    {
        QVector< QSharedPointer<HalfEdge> >  outEdges;
        VertexEdgeIterator it(sharedFromThis());
        while (it.hasNext())
        {
            outEdges.push_back(it.getNext());
        }
        return outEdges;
    }
    
    QVector< QSharedPointer<Vertex> >  Vertex::neighbors()
    {
        QVector< QSharedPointer<Vertex> >  neighbors;
    
        QVector< QSharedPointer<HalfEdge> >  inwardEdges = inEdges();
        for (QSharedPointer<HalfEdge> i : inwardEdges)
        {
            neighbors.push_back(i->origin());
        }
    
        return neighbors;
    }
    void Vertex::key(int key)
    {
        m_key = key;
    }
    int Vertex::key()
    {
        return m_key;
    }
    QSharedPointer<HalfEdge> Vertex::incidentEdge()
    {
        return m_incident_edge;
    }
    void Vertex::incidentEdge(QSharedPointer<HalfEdge> incident_edge)
    {
        m_incident_edge = incident_edge;
    }
    
    void Vertex::data(QSharedPointer<VertexData> vertex_data)
    {
        m_vertex_data = vertex_data;
    }
    
    QSharedPointer<VertexData> Vertex::data()
    {
        return m_vertex_data;
    }
}
