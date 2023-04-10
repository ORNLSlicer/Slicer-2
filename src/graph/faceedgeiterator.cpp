#include "graph/faceedgeiterator.h"

namespace ORNL 
{
    
    FaceEdgeIterator::FaceEdgeIterator(QSharedPointer<Face> face)
    {
        reset(face);
    }
    
    // Check if there is any edge in the sequence of edges.
    bool FaceEdgeIterator::hasNext()
    {
        return nextEdge() != nullptr;
    }
    
    QSharedPointer<HalfEdge> FaceEdgeIterator::getNext()
    {
        QSharedPointer<HalfEdge> next = nextEdge();
    
        if (nextEdge())
        {
            nextEdge(next->next());
        }
    
        if ( nextEdge() == startEdge() )
        {
            nextEdge(nullptr);
        }
    
        return next;
    }
    
    // Makes the iterator start again, from the first edge of the sequence.
    void FaceEdgeIterator::reset()
    {
        nextEdge(startEdge());
    }
    
    void FaceEdgeIterator::reset(QSharedPointer<Face> face)
    {
        startEdge(face->boundary());
        reset();
    }
    
    void FaceEdgeIterator::startEdge(QSharedPointer<HalfEdge> start_edge)
    {
        m_start_edge = start_edge;
    }
    
    void FaceEdgeIterator::nextEdge(QSharedPointer<HalfEdge> next_edge)
    {
        m_next_edge =  next_edge ;
    }
    
    QSharedPointer<HalfEdge> FaceEdgeIterator::startEdge()
    {
        return m_start_edge;
    }
    
    QSharedPointer<HalfEdge> FaceEdgeIterator::nextEdge()
    {
        return m_next_edge;
    }
    
    
}
