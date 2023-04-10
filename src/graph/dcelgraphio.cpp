#include "graph/dcelgraphio.h"

namespace ORNL
{
    namespace DCELGraph
    {
        void saveDCELGraph(QSharedPointer<Graph> graph, QString file_path)
        {
            qDebug()<<"Saving graph to " << file_path << " ..." ;

            QFile file(file_path);

            if (!file.open(QFile::WriteOnly | QFile::Truncate | QFile::Text)) {
                qDebug()<<"Could not open file: "<<file_path<<"\n";
                exit (EXIT_FAILURE);
            }

            QTextStream out(&file);

    //      ofstream out(file_path);
    //      if (!out) // IO Error check.
    //      {
    //          cerr << "File: \"" << file_path << "\" could not be opened!" << endl;
    //          return;
    //      }

            // Write the vertex data.
            out << "<VERTICES>" << END_OF_LINE_STR;
            out << graph->getNumVertices() << END_OF_LINE_STR;
            for(QMap<int, QSharedPointer<Vertex> >::iterator it = graph->beginVertices(); it != graph->endVertices(); it++)
            {
                out << it.key(); // Write out vertex key;

                QSharedPointer<Circle> circle = graph->getCircle(it.key());
                if (circle != nullptr)
                {
                    out << " " << int(circle->center().x());
                    out << " " << int(circle->center().y());
                    out << " " << int(circle->radius());
                    out << " " << int(circle->isSecondary());
                }
                else // No planar embedding data for this vertex.
                {
                    out << " 0 0 -1 0"; // Write default values.
                }

                out << END_OF_LINE_STR;
            }
            out << "</VERTICES>" << END_OF_LINE_STR;

            // Write the edge data.
            out << "<EDGES>" << END_OF_LINE_STR;
            out << int(0.5*graph->getNumEdges()) << END_OF_LINE_STR;
            QSet<int64_t> processed_halfEdges; // Use this to avoid duplicates.
            for (QMap<int, QSharedPointer<HalfEdge> >::iterator it = graph->beginEdges(); it != graph->endEdges(); it++)
            {
                if (processed_halfEdges.find(it.key()) != processed_halfEdges.end())
                    continue;

                out << it.key(); // Write out halfEdge key.
                out << " " << it.value()->twin()->key(); // Write out twin's key.
                out << " " << it.value()->origin()->key();
                out << " " << it.value()->target()->key();

                processed_halfEdges.insert(it.key());
                processed_halfEdges.insert(it.value()->twin()->key());

                out << END_OF_LINE_STR;
            }
            out << "</EDGES>" << END_OF_LINE_STR;

            // Write out the face data.
            out << "<FACES>" << END_OF_LINE_STR;
            out << graph->getNumFaces() << END_OF_LINE_STR;

            // Write out the exterior face data first.
            out << graph->exterior()->key();
            if (graph->getNumEdges() > 0 && graph->exterior()->boundary() != nullptr)
            {
                out << " " << graph->exterior()->getEdgeCount();
                for (QSharedPointer<HalfEdge> h : graph->exterior()->getBoundaryEdges())
                {
                    out << " " << h->key();
                }
            }
            else
            {
                out << " 0";
            }
            out << END_OF_LINE_STR;

            for (QMap<int, QSharedPointer<Face> >::iterator it = graph->beginFaces(); it != graph->endFaces(); it++) // Write out the remaining faces.
            {
                if (it.value() == graph->exterior()) continue;

                out << it.key();

                out << " " << it.value()->getEdgeCount();
                for (QSharedPointer<HalfEdge> h : it.value()->getBoundaryEdges())
                {
                    out << " " << h->key();
                }
                out << END_OF_LINE_STR;
            }
            out << "</FACES>" << END_OF_LINE_STR;

            qDebug() << "Graph has been saved to " << file_path << "!" ;

            file.close();

        }

        void saveDCELGraphNew(QSharedPointer<Graph> graph, QString file_path)
        {
            qDebug()<<"Saving graph to " << file_path << " ..." ;

            QFile file(file_path);

            if (!file.open(QFile::WriteOnly | QFile::Truncate | QFile::Text)) {
                qDebug()<<"Could not open file: "<<file_path<<"\n";
                exit (EXIT_FAILURE);
            }

            QTextStream out(&file);

    //      ofstream out(file_path);
    //      if (!out) // IO Error check.
    //      {
    //          cerr << "File: \"" << file_path << "\" could not be opened!" << endl;
    //          return;
    //      }

            // Write the vertex data.
            out << "<VERTICES>" << END_OF_LINE_STR;
            out << graph->getNumVertices() << END_OF_LINE_STR;
            for(QMap<int, QSharedPointer<Vertex> >::iterator it = graph->beginVertices(); it != graph->endVertices(); it++)
            {
                out << it.key(); // Write out vertex key;

                QSharedPointer<Circle> circle = graph->getCircle(it.key());
                if (circle != nullptr)
                {
                    out << " " << int(circle->center().x());
                    out << " " << int(circle->center().y());
                    out << " " << int(circle->radius());
                    out << " " << int(circle->isSecondary());
                }
                else // No planar embedding data for this vertex.
                {
                    out << " 0 0 -1 0"; // Write default values.
                }

                out << END_OF_LINE_STR;
            }
            out << "</VERTICES>" << END_OF_LINE_STR;

            // Write the edge data.
            out << "<EDGES>" << END_OF_LINE_STR;
            out << int(0.5*graph->getNumEdges()) << END_OF_LINE_STR;
            QSet<int64_t> processed_halfEdges; // Use this to avoid duplicates.
            for (QMap<int, QSharedPointer<HalfEdge> >::iterator it = graph->beginEdges(); it != graph->endEdges(); it++)
            {
                if (processed_halfEdges.find(it.key()) != processed_halfEdges.end())
                    continue;

                out << it.key(); // Write out halfEdge key.
                out << " " << it.value()->twin()->key(); // Write out twin's key.
                out << " " << it.value()->origin()->key();
                out << " " << it.value()->target()->key();

                processed_halfEdges.insert(it.key());
                processed_halfEdges.insert(it.value()->twin()->key());

                out << END_OF_LINE_STR;
            }
            out << "</EDGES>" << END_OF_LINE_STR;

            // Write out the face data.
            out << "<FACES>" << END_OF_LINE_STR;
            out << graph->getNumFaces() << END_OF_LINE_STR;

            // Write out the exterior face data first.
            out << graph->exterior()->key();
            if (graph->getNumEdges() > 0 && graph->exterior()->boundary() != nullptr)
            {
                out << " " << graph->exterior()->getEdgeCount();
                for (QSharedPointer<HalfEdge> h : graph->exterior()->getBoundaryEdges())
                {
                    out << " " << h->key();
                }
            }
            else
            {
                out << " 0";
            }
            out << END_OF_LINE_STR;

            for (QMap<int, QSharedPointer<Face> >::iterator it = graph->beginFaces(); it != graph->endFaces(); it++) // Write out the remaining faces.
            {
                if (it.value() == graph->exterior()) continue;

                out << it.key();

                out << " " << it.value()->getEdgeCount();
                for (QSharedPointer<HalfEdge> h : it.value()->getBoundaryEdges())
                {
                    out << " " << h->key();
                }
                out << END_OF_LINE_STR;
            }
            out << "</FACES>" << END_OF_LINE_STR;

            QList<QSharedPointer<HalfEdge> > edgeusedlist ;

            QList<QSharedPointer<HalfEdge> > faceedgelist ;

            QSharedPointer<HalfEdge> nextedge;

            bool nextedgecheck ;

            int faceindex = 0;

            for (QMap<int, QSharedPointer<HalfEdge> >::iterator it = graph->beginEdges(); it!= graph->endEdges(); it++)
            {

                if(edgeusedlist.contains(it.value())== false)
                {
                    edgeusedlist.append(it.value());

                    faceedgelist.append(it.value());

                    nextedgecheck = true ;

                    nextedge = it.value();

                    out << faceindex <<" ";

                    while(nextedgecheck)
                    {
                        out << nextedge->key() << " " ;

                        if(nextedge==it.value())
                        {
                            nextedgecheck = false;

                        }
                        else
                        {
                            edgeusedlist.append(nextedge);
                        }

                        nextedge = nextedge->next() ;
                    }

                    faceindex++;

                    out<< endl ;

                    faceedgelist.clear();

                }
            }
            edgeusedlist.clear();

            qDebug() << "Graph has been saved to " << file_path << "!" ;

            file.close();
        }


        QSharedPointer<Graph> loadDCELGraph(QString file_path)
        {
            QSharedPointer<Graph> graph = QSharedPointer<Graph>(new Graph());
            QFile file(file_path);
            if (!file.open(QFile::ReadOnly | QFile::Truncate | QFile::Text)) // IO Error check.
            {
                qDebug()<<"Could not open file: "<<file_path<<"\n";
                return graph;
            }

            QTextStream in(&file);

            // Read in the vertex data.
            QString text = "";
            while (!in.atEnd() && !text.contains("<VERTICES>"))
                in >> text;
            if (text != "<VERTICES>") return graph;

            int64_t num_vertices = 0;
            in >> num_vertices;
            for (int64_t i = 0; i < num_vertices; i++)
            {
                int64_t vertex_key = -1;
                int64_t x_coord = 0;
                int64_t y_coord = 0;
                double radius = 0;
                bool isSecondary = false;
                int tmp_isSecondary = 0;
                // QTextStream cannot read or write bool type variable.
                // So we have to use tmp to store the bool variable in another type, like int (0, 1).
                in >> vertex_key >> x_coord >> y_coord >> radius >> tmp_isSecondary;
                if (tmp_isSecondary) isSecondary = true;
                else isSecondary = false;
                //qDebug() << vertex_key << " " << x_coord << " " << y_coord << " " << radius << " "  << isSecondary;

                graph->addVertex(vertex_key);
                Point center = Point(x_coord, y_coord);
                QSharedPointer<Circle> circle = QSharedPointer<Circle>(new Circle(center, radius));
                circle->isSecondary(isSecondary);
                graph->setData(vertex_key, circle);
            }

            // Read in the edge data.
            while (!in.atEnd() && !text.contains("<EDGES>"))
                in >> text;
            if (text != "<EDGES>") return graph;

            int64_t num_edges = 0;
            in >> num_edges;
            for (int64_t i = 0; i < num_edges; i++)
            {
                int64_t halfEdge_key = -1;
                int64_t twin_key = -1;
                int64_t origin_key = -1;
                int64_t target_key = -1;
                in >> halfEdge_key >> twin_key >> origin_key >> target_key;

                QSharedPointer<Vertex> origin = graph->getVertex(origin_key);
                QSharedPointer<Vertex> target = graph->getVertex(target_key);

                QSharedPointer<HalfEdge> h = QSharedPointer<HalfEdge>(new HalfEdge());
                h->origin(origin);
                origin->incidentEdge(h);
                h->leftFace(graph->exterior());
                h->next(nullptr);
                h->previous(nullptr);
                h->key(halfEdge_key);
                //graph->eMap[halfEdge_key] = h;
                graph->addEdge(halfEdge_key, h);

                QSharedPointer<HalfEdge> h_twin = QSharedPointer<HalfEdge>(new HalfEdge());
                h_twin->origin(target);
                target->incidentEdge(h_twin);
                h_twin->leftFace(graph->exterior());
                h_twin->next(nullptr);
                h_twin->previous(nullptr);
                h->setTwin(h_twin);
                h_twin->key(twin_key);
                //graph->eMap[twin_key] = h_twin;
                graph->addEdge(twin_key, h_twin);
            }

            // Read in the face data.
            while (!in.atEnd() && !text.contains("<FACES>"))
                in >> text;
            if (text != "<FACES>") return graph;

            int64_t num_faces = 0;
            in >> num_faces;
            for (int64_t i = 0; i < num_faces; i++)
            {
                int64_t face_key = -1;
                int64_t num_boundary_edges = 0;
                in >> face_key >> num_boundary_edges;

                QSharedPointer<Face> face = nullptr;
                if (face_key == -1) // Exterior face already exists.
                    face = graph->exterior();
                else
                    face = graph->addFace(face_key, nullptr);

                QVector<int64_t> edge_keys;
                for (int64_t j = 0; j < num_boundary_edges; j++)
                {
                    int64_t boundary_edge_key = -1;
                    in >> boundary_edge_key;
                    edge_keys.push_back(boundary_edge_key);
                }

                int64_t m = edge_keys.size();
                for (int64_t j = 0; j < m; j++)
                {
                    QSharedPointer<HalfEdge> current = graph->getHalfEdge(edge_keys[j]);
                    QSharedPointer<HalfEdge> next = graph->getHalfEdge(edge_keys[(j + 1) % m]);
                    current->setNext(next);
                    current->leftFace(face);
                }

                if (m > 0) face->boundary(graph->getHalfEdge(edge_keys[0]));
            }

            return graph;
        }
    }

} // namespace ORNL
