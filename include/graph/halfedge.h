#ifndef HALFEDGE_H
#define HALFEDGE_H

#include <QEnableSharedFromThis>
#include <QSharedPointer>
#include <QVector>

#include "face.h"
#include "vertex.h"

namespace ORNL
{
    class Vertex;
    class Face;
    /*!
    * \class HalfEdge
    * \brief HalfEdge of a graph is a connection between to vertices pointing in one direction.
    */

    class HalfEdge:public QEnableSharedFromThis<HalfEdge>
    {
    public:
        HalfEdge();
        QSharedPointer<Vertex> target();
        // //! \brief sets twin edge of an edge in a graph
        //!
        //! \brief setTwin
        //! \param newTwin
        //!
        void setTwin(QSharedPointer<HalfEdge> newTwin);
        // //! \brief sets next edge in a graph
        //!
        //! \brief setNext
        //! \param newNext
        //!
        void setNext(QSharedPointer<HalfEdge> newNext);
        // //! \brief sets previous edge of an edge in a graph
        //!
        //! \brief setPrevious
        //! \param newPrevious
        //!
        void setPrevious(QSharedPointer<HalfEdge> newPrevious);
        // //! \brief return twin, next, previous, leftface, origin, weight, path.
        //!
        //! \brief twin
        //! \return
        //!
        QSharedPointer<HalfEdge> twin();
        //!
        //! \brief next
        //! \return
        //!
        QSharedPointer<HalfEdge> next();
        //!
        //! \brief previous
        //! \return
        //!
        QSharedPointer<HalfEdge> previous();
        //!
        //! \brief origin
        //! \return
        //!
        QSharedPointer<Vertex> origin();
        //!
        //! \brief leftFace
        //! \return
        //!
        QSharedPointer<Face> leftFace();
        //!
        //! \brief weight
        //! \return
        //!
        float weight();
        //!
        //! \brief path
        //! \return
        //!
        QVector<int> path();
        //!
        //! \brief key
        //! \return
        //!
        int key();
        // //! \brief set twin, next, previous, leftface, origin, weight, path.
        //!
        //! \brief twin
        //! \param twin
        //!
        void twin( QSharedPointer<HalfEdge> twin);
        //!
        //! \brief next
        //! \param next
        //!
        void next( QSharedPointer<HalfEdge> next);
        //!
        //! \brief previous
        //! \param previous
        //!
        void previous( QSharedPointer<HalfEdge> previous);
        //!
        //! \brief origin
        //! \param origin
        //!
        void origin(  QSharedPointer<Vertex> origin);

//      QSharedPointer<Vertex> origin();
        //!
        //! \brief leftFace
        //! \param left_face
        //!
        void leftFace(  QSharedPointer<Face> left_face);
        //!
        //! \brief weight
        //! \param weight
        //!
        void weight( float weight);
        //!
        //! \brief path
        //! \param path
        //!
        void path( QVector<int> path);
        //!
        //! \brief key
        //! \param key
        //!
        void key( int key);

    private:
        int m_key;
        QSharedPointer<HalfEdge> m_twin;
        QSharedPointer<HalfEdge> m_next;
        QSharedPointer<HalfEdge> m_previous;
        QSharedPointer<Vertex> m_origin;
        QSharedPointer<Face> m_left_face;
        float m_weight;
        QVector<int> m_path;
    };

}
#endif // HALFEDGE_H
