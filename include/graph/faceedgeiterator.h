#ifndef FACEEDGEITERATOR_H
#define FACEEDGEITERATOR_H

#include <QSharedPointer>
#include <QVector>

#include "face.h"
#include "halfedge.h"

namespace ORNL
{
    /*!
    * \class FaceEdgeIterator
    * \brief FaceEdgeIterator is halfedge iterator for any face of a graph.
    */

    class FaceEdgeIterator
    {
    public:

        FaceEdgeIterator(QSharedPointer<Face> face);

        //!
        //! \brief hasNext  Check if there is next halfedge in a sequence of edges.
        //!

        bool hasNext();

        //!
        //! \brief getNext Find and return next halfedge.
        //! \return next halfedge.
        //!

        QSharedPointer<HalfEdge> getNext();

        //!
        //! \brief reset Makes iterator start again, from first halfedge of face.
        //!

        void reset();

        //!
        //! \brief reset Makes iterator start again, from boundary halfedge of face.
        //! \param face face we wants to reset.
        //!

        void reset(QSharedPointer<Face> face);

        // //! \brief Return startEdge, nextEdge.
        //!
        //! \brief startEdge Return first edge in a sequence of edges of face.
        //! \return First edge in a sequence of edges of a face.
        //!

        QSharedPointer<HalfEdge> startEdge();

        //!
        //! \brief nextEdge Returns next edge in a sequence of edges of a face.
        //! \return next edge in a sequence of edges of a face.
        //!

        QSharedPointer<HalfEdge> nextEdge();

        //!
        //! \brief startEdge Sets first edge in a sequence of edges of face.
        //! \param start_edge first edge in a sequence of edges of face.
        //!

        void startEdge(QSharedPointer<HalfEdge> start_edge);

        //!
        //! \brief nextEdge Sets next edge in a sequence of edges of face.
        //! \param next_edge next edge in a sequence of edges of face.
        //!

        void nextEdge(QSharedPointer<HalfEdge> next_edge);

    private:

        QSharedPointer<HalfEdge> m_start_edge;

        QSharedPointer<HalfEdge> m_next_edge;
    };

}
#endif // FACEEDGEITERATOR_H
