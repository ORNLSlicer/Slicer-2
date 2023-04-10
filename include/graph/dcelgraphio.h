#ifndef DCELGRAPHIO_H
#define DCELGRAPHIO_H

#include <QFile>
#include <QString>
#include <QTextStream>

#include "face.h"
#include "graph.h"
#include "halfedge.h"
#include "vertex.h"

namespace ORNL
{
    //! \brief DCELGraph saving utilities
    namespace DCELGraph
    {
        const QString END_OF_LINE_STR = "\n";

        //!
        //! \brief savedcelgraph save graph to a file in dcel format.
        //! \param graph Input graph.
        //! \param file_path Location of output dcel file.
        //!
        void saveDCELGraph(QSharedPointer<Graph> graph, QString file_path);
        void saveDCELGraphNew(QSharedPointer<Graph> graph, QString file_path);

        //! \brief Load graph data from a dcel file.
        QSharedPointer<Graph> loadDCELGraph(QString file_path);
    }
}

#endif // DCELGRAPHIO_H


