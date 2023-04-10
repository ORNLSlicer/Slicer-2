#include "graph/vertexedgeiterator.h"

namespace ORNL 
{
    
    VertexEdgeIterator::VertexEdgeIterator(QSharedPointer<Vertex> v)
    {
        reset(v);
    }
    
    bool VertexEdgeIterator::hasNext()
    {
        return m_nextEdge != nullptr;
    }
    
    // Returns the next edge in the sequence of NULL if the sequence has ended.
    QSharedPointer<HalfEdge> VertexEdgeIterator::getNext()
    {
        QSharedPointer<HalfEdge> next = nextEdge();
    
        if (nextEdge())
        {
            QSharedPointer<HalfEdge> twin = nextEdge()->twin();
            nextEdge(twin->next());
        }
    
        if (nextEdge() == startEdge())
        {
            nextEdge(nullptr);
        }
    
        return next;
    }
    
    // Makes the iterator start again, from the first edge of the sequence.
    void VertexEdgeIterator::reset()
    {
        nextEdge(startEdge());
    }
    
    void VertexEdgeIterator::reset(QSharedPointer<Vertex> v)
    {
        startEdge(v->incidentEdge());
        reset();
    }
    
    QSharedPointer<HalfEdge> VertexEdgeIterator::startEdge()
    {
    
        return m_startEdge;
    }
    
    void VertexEdgeIterator::startEdge(QSharedPointer<HalfEdge> startEdge)
    {
        m_startEdge = startEdge;
    }
    
    QSharedPointer<HalfEdge> VertexEdgeIterator::nextEdge()
    {
        return m_nextEdge;
    }
    
    void VertexEdgeIterator::nextEdge(QSharedPointer<HalfEdge> nextEdge)
    {
        m_nextEdge = nextEdge;
    }
    
}
