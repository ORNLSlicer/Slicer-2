#ifndef VERTEXEDGEITERATOR_H
#define VERTEXEDGEITERATOR_H

#include <QSharedPointer>
#include <QVector>

#include "halfedge.h"
#include "vertex.h"

namespace ORNL
{

    class HalfEdge;
    class Vertex;
    /*!
    * \class VertexEdgeIterator
    * \brief VertexEdgeIterator is halfedge iterator of a graph.
    */

    class VertexEdgeIterator
    {
    public:
        VertexEdgeIterator();
        VertexEdgeIterator(QSharedPointer<Vertex> v);
        //! \brief Check if there is any edge in the sequence of edges.
        bool hasNext();
        //! \brief Returns the next edge in the sequence of NULL if the sequence has ended.
        QSharedPointer<HalfEdge> getNext();
        //! \brief Makes the iterator start again, from the first edge of the sequence.
        void reset();
        //! \brief Makes the iterator start again, from an edge incident to vertex v of the sequence.
        void reset(QSharedPointer<Vertex> v);
        //! \brief Return start edge
        QSharedPointer<HalfEdge> startEdge();
        //! \brief Set the start edge
        void startEdge(QSharedPointer<HalfEdge> startEdge);
        //! \brief Return the next edge
        QSharedPointer<HalfEdge> nextEdge();
        //! brief Set the next edge
        void nextEdge(QSharedPointer<HalfEdge> nextEdge);
    private:
        QSharedPointer<HalfEdge> m_startEdge;
        QSharedPointer<HalfEdge> m_nextEdge;

    };

}
#endif // VERTEXEDGEITERATOR_H
