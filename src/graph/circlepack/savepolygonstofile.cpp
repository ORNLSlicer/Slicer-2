#include "graph/circlepack/savepolygonstofile.h"

namespace ORNL 
{
    namespace SavePolygons
    {
        void savePolygonsToFile(Polygon* polygon, QString file_path)
        {
            PolygonList polygons;
            polygons += *polygon;
            savePolygonsToFile(&polygons, file_path);
        }

        void savePolygonsToFile(PolygonList* polygons, QString file_path)
        {

            QFile file(file_path);

            if (!file.open(QFile::WriteOnly | QFile::Truncate | QFile::Text)) {
                qDebug()<<"Could not open file: "<<file_path<<"\n";
                exit (EXIT_FAILURE);
            }

            QTextStream out(&file);

        //    ofstream out(file_path);
        //    if (!out) // IO Error check.
        //    {
        //        cerr << "File: \"" << file_path << "\" could not be opened." << endl;
        //        return;
        //    }

            int n = polygons->size();
            if (n > 0)
            {
                // Write out the external polygon first.
                out << "EXTERNAL"<<endl;
                int m = (*polygons)[0].size();
                for (int j = 0; j < m; j++)
                {
                    Point p = (*polygons)[0][j];
                    out << (p.x() / DEFAULT_LOAD_SCALE);
                    out << " ";
                    out << (p.y() / DEFAULT_LOAD_SCALE)<< endl;
                    //out << endl;
                }

                // Need to repeat the starting point
                Point p = (*polygons)[0][0];
                out << (p.x()/ DEFAULT_LOAD_SCALE);
                out << " ";
                out << (p.y()/ DEFAULT_LOAD_SCALE)<< endl;
                //out << endl;

                // Write out the remaining internal polygons that represent holes.
                for (int i = 1; i < n; i++)
                {
                    out << "INTERNAL\n";
                    int m = (*polygons)[i].size();
                    for (int j = 0; j < m; j++)
                    {
                        Point p = (*polygons)[i][j];
                        out << (p.x() / DEFAULT_LOAD_SCALE);
                        out << " ";
                        out << (p.y() / DEFAULT_LOAD_SCALE)<<endl;
                        //out << endl;
                    }
                }
                  }
            file.close();
            qDebug()<< "Polygons have been saved to " << file_path << "." << endl;
        }

        Polygon readBoundaryPolygonFile(QString file_path)
        {
            Polygon polygon;

            QFile file(file_path);

            if (!file.open(QIODevice::ReadOnly | QIODevice::Text)){
                qDebug()<<"Could not open file: \\"<<file_path<<"\n";
                exit (EXIT_FAILURE);
            }

            QTextStream in(&file);

            while(!in.atEnd())
            {
              QString line = in.readLine();
              if(QString::compare(line, "EXTERNAL", Qt::CaseInsensitive)==0) continue;
              QStringList strlist = line.split(' ');
              //qDebug()<<line<<"\n";
              //qDebug()<<strlist[0].toInt()<<" "<<strlist[1].toInt()<<"\n";
              //qDebug()<<strlist.size()<<"\n";
              if(strlist.size()!=2)
               {
                  qDebug("error in polygon file");
                  exit (EXIT_FAILURE);
               }
              else{
                  polygon.push_back(Point(strlist[0].toInt() ,strlist[1].toInt()));
               }

            }

           file.close();

           return polygon;

        }

    }
}
