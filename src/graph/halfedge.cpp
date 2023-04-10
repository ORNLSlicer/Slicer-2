#include "graph/halfedge.h"

namespace ORNL 
{
    
    HalfEdge::HalfEdge()
    {
    }
    
    QSharedPointer<Vertex> HalfEdge::target()
    {
        return twin()->origin();
    }
    
    void HalfEdge::setTwin(QSharedPointer<HalfEdge> newTwin)
    {
        twin(newTwin);
        newTwin->twin(sharedFromThis());
    }
    
    void HalfEdge::setNext(QSharedPointer<HalfEdge> newNext)
    {
        next(newNext);
        newNext->previous(sharedFromThis());
    }
    
    void HalfEdge::setPrevious(QSharedPointer<HalfEdge> newPrevious)
    {
        previous(newPrevious);
        newPrevious->next(sharedFromThis());
    }
    
    QSharedPointer<HalfEdge> HalfEdge::twin()
    {
        return m_twin;
    }
    
    QSharedPointer<HalfEdge> HalfEdge::next()
    {
        return m_next;
    }
    
    QSharedPointer<HalfEdge> HalfEdge::previous()
    {
        return m_previous;
    }
    
    //QSharedPointer<Vertex> HalfEdge::origin()
    //{
    //    return m_origin;
    //}
    
    QSharedPointer<Face> HalfEdge::leftFace()
    {
        return m_left_face;
    }
    
    float HalfEdge::weight()
    {
        return m_weight;
    }
    
    QVector<int> HalfEdge::path()
    {
        return m_path;
    }
    
    int HalfEdge::key()
    {
        return m_key;
    }
    
    void HalfEdge::twin( QSharedPointer<HalfEdge> twin)
    {
        m_twin = twin;
    }
    void HalfEdge::next( QSharedPointer<HalfEdge> next)
    {
        m_next = next;
    }
    void HalfEdge::previous( QSharedPointer<HalfEdge> previous)
    {
        m_previous = previous;
    }
    void HalfEdge::origin(  QSharedPointer<Vertex> origin)
    {
        m_origin = origin;
    }
    
    QSharedPointer<Vertex> HalfEdge::origin()
    {
        return m_origin;
    }
    void HalfEdge::leftFace(  QSharedPointer<Face> left_face)
    {
        m_left_face = left_face;
    }
    void HalfEdge::weight( float weight)
    {
        m_weight= weight;
    }
    
    void HalfEdge::path( QVector<int> path)
    {
        m_path = path;
    }
    void HalfEdge::key( int key)
    {
        m_key = key;
    }
}
