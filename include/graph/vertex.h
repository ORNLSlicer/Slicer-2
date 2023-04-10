#ifndef VERTEX_H
#define VERTEX_H

#include <QEnableSharedFromThis>
#include <QSharedPointer>
#include <QVector>

#include "halfedge.h"
#include "vertexdata.h"
#include "vertexedgeiterator.h"

namespace ORNL
{
    class HalfEdge;

    /*!
    * \class Vertex
    * \brief Vertex of a graph is each individual object in it.
    */

    class Vertex:public QEnableSharedFromThis<Vertex>
    {
    public:

        Vertex()=default;

        Vertex(int key);
        //! \brief indegree of a vertex is the number of edges pointing to that vertex
        int inDegrees();

        //! \brief outdegree of a vertex is the number of edges pointing away from vertex
        int outDegrees();

        //! \brief Return list of Half edges pointing away from vertex
        QVector< QSharedPointer<HalfEdge> > inEdges();

        //! \brief Return list of Half edges pointing toward vertex
        QVector< QSharedPointer<HalfEdge> > outEdges();

        //! \brief Return list of neighboring vertex
        QVector< QSharedPointer<Vertex> > neighbors();

        //! \brief Return incident edge
        QSharedPointer<HalfEdge> incidentEdge();

        //! \brief set incident edge
        void incidentEdge(QSharedPointer<HalfEdge> incidentEdge);

        //! \brief set key
        void key(int key);

        //! \brief Return key
        int key();

        //! \brief Set vertex data
        void data(QSharedPointer<VertexData> vertex_data);

        //! \brief Return Location of vertex.
        QSharedPointer<VertexData> data();

        template <typename T>
        QSharedPointer<T> data()
        {
            return m_vertex_data.dynamicCast<T>();
        }

        template <typename T>
        void data(QSharedPointer<T> vertexData)
        {
            // The dynamicCast() causes issues during compilation for some reason.
            //m_vertex_data = vertexData.dynamicCast<VertexData>();
            m_vertex_data = vertexData;
        }

    private:
        int m_key;
        QSharedPointer<HalfEdge> m_incident_edge;
        QSharedPointer<VertexData> m_vertex_data;
    };

}
#endif // VERTEX_H
