#ifndef FACE_H
#define FACE_H

#include "halfedge.h"

namespace ORNL
{
    class HalfEdge;
    /*!
    * \class Face
    * \brief Face in a graph is smallest region bounded by edges
    */

    class Face
    {
    public:
        Face();
        Face(int key, QSharedPointer<HalfEdge> edge);

        //!
        //! \brief getEdgeCount Count Edge of the boundary.
        //! \return Number of boundary edges.
        //!

        int getEdgeCount();

        //!
        //! \brief getBoundaryEdges Finds all boundary edges
        //! \return Container containing pointers to all boundary edges
        //!

        QVector< QSharedPointer<HalfEdge> > getBoundaryEdges();

        //!
        //! \brief key
        //! \return Face unique id (key).
        //!

        int key();

        //!
        //! \brief boundary Finds boundary edge of face.
        //! \return boundary edge
        //!

        QSharedPointer<HalfEdge> boundary();

        //!
        //! \brief key Sets unique id (key) of face.
        //! \param key Unique id of a face.
        //!

        void key(int key);

        //!
        //! \brief boundary Sets boundary of face.
        //! \param boundary boundary of face.
        //!

        void boundary(QSharedPointer<HalfEdge> boundary);

    private:

        int m_key;

        QSharedPointer<HalfEdge> m_boundary;

    };

}

#endif // FACE_H
