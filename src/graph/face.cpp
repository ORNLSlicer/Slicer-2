#include "graph/face.h"

namespace ORNL 
{
    
    Face::Face()
    {
    
    }
    
    Face::Face(int key, QSharedPointer<HalfEdge> edge) :
            m_key(key)
    {
        boundary(edge);
    }
    
    int Face::getEdgeCount()
    {
        int rv = 0;
        QSharedPointer<HalfEdge> start = boundary();
        rv++;
        QSharedPointer<HalfEdge> h = start->next();
        while (start != h)
        {
            rv++;
            h = h->next();
        }
    
        return rv;
    }
    
    QVector< QSharedPointer<HalfEdge> > Face::getBoundaryEdges()
    {
        QSharedPointer<HalfEdge> start = boundary();
        QVector< QSharedPointer<HalfEdge> > rv;
        rv.push_back(boundary());
        QSharedPointer<HalfEdge> h = start->next();
        while (start != h)
        {
            rv.push_back(h);
            h = h->next();
        }
    
        return rv;
    }
    
    void Face::key(int key)
    {
        m_key = key;
    }
    
    void Face::boundary(QSharedPointer<HalfEdge> boundary)
    {
        m_boundary = boundary;
    }
    
    int Face::key()
    {
        return m_key;
    }
    QSharedPointer<HalfEdge> Face::boundary()
    {
        return m_boundary;
    }
    
    
}
