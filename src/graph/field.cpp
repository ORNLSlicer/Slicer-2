#include "graph/field.h"

namespace ORNL 
{
    
    
    Field::Field(){
        double threshold = 0.5;
    }
    
    void Field::loadfieldfile(QString file_path){
    
        // This is used for scaling from inches to microns assuming the units in file are in inches
        // This need to be automated by the information specified in interface. We also need to be careful
        // while by looking at number of digits each number has.
        QFile file(file_path);
    
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)){
            qDebug()<<"Could not open file: \\"<<file_path<<"\n";
            exit (EXIT_FAILURE);
        }
        else{
            qDebug()<<"field file is loaded: \\"<<"\n";
        }
    
        QTextStream in(&file);
    
    
        double scale = 10000.00;
        QPair<int, int> key;
        int filelength = 0;
        while(!in.atEnd())
        {
            QString line = in.readLine();
            QStringList strlist = line.split('\t');
            //qDebug()<< line <<"\n";
            //qDebug()<< strlist[0]<<"\n";
            key = qMakePair(strlist[1].toDouble()*scale, strlist[2].toDouble()*scale);
            //qDebug()<<strlist[1].toDouble()*scale<<" "<<strlist[2].toDouble()*scale<<"\n";
            if(filelength==0){
                //fieldvalue[key] = strlist[4].toDouble();
                fieldValue(key, strlist[4].toDouble());
            }
            else{
                if(fieldValue(key)<strlist[4].toDouble()){
                    fieldValue(key, strlist[4].toDouble());
                    //fieldvalue[key] = strlist[4].toDouble();
                }
            }
    
            filelength = filelength + 1;
        }
    }
    
    double Field::getField(QSharedPointer<Circle> circle)
    {
        int lowerx = static_cast<int>(circle->center().x()) - static_cast<int>(circle->radius());
        int upperx = static_cast<int>(circle->center().x()) + static_cast<int>(circle->radius());
        int lowery = static_cast<int>(circle->center().y()) - static_cast<int>(circle->radius());
        int uppery = static_cast<int>(circle->center().y()) + static_cast<int>(circle->radius());
        double max = 0;
    
        for(QMap< QPair<int,int>,double>::iterator it = m_field_value.begin(); it != m_field_value.end(); ++it) {
    
            if(it.key().first <= upperx && it.key().first >=lowerx && it.key().second >=lowery && it.key().second<=uppery)
            {
                if(it.value() > max)
                {
                    max = it.value();
                }
            }
    
        }
        return max;
    }
    
    
    QSet< QSharedPointer<Vertex> >  Field::selectVerticesOverFieldThreshold(QSharedPointer<Graph> graph, double threshold)
    {
      QSet< QSharedPointer<Vertex> >  selection;
      double temp_value;
    
      for (QMap<int, QSharedPointer<Vertex> > ::iterator it = graph->beginVertices(); it!=graph->endVertices(); it++) // Check each vertex of the graph.// graph->vMap gives us the vertex key.
      {
        temp_value = this->getField(graph->getCircle(it.key()));
    
        if (temp_value > threshold)
        {
          //vertexfield[it.first] = temp_value ;
          selection.insert(it.value());
        }
      }
      return selection;
    }
    
    void Field::normalizeField()
    {
        // Maximum  Value
        double maxn =0;
        for(QMap< QPair<int,int>,double>::iterator it = m_field_value.begin(); it != m_field_value.end(); ++it) {
            if(it.value() > maxn)
                {
                    maxn = it.value();
                }
        }
    
        qDebug()<<"MAXVALUE"<<" " << maxn<<"\n"<<endl;
    
        // Minimum Value
        double minn =1.1E16;
        for(QMap< QPair<int,int>,double>::iterator it = m_field_value.begin(); it != m_field_value.end(); ++it) {
            if(it.value() < minn)
                {
                    minn = it.value();
                }
        }
    
    
        qDebug()<<"MINVALUE"<<" " << minn<<"\n"<<endl;
    
    
        for(QMap< QPair<int,int>,double>::iterator it = m_field_value.begin(); it != m_field_value.end(); ++it) {
    
            if(maxn-minn > 0.1){
            it.value()  = (it.value() - minn) / (maxn-minn);
            }
            else{
            it.value()  = it.value() / maxn;
            }
    
        }
    }
    
    QMap< QPair<int, int>, double> Field::fieldValue()
    {
        return m_field_value;
    }
    double Field::fieldValue(QPair<int, int> key)
    {
        return m_field_value[key];
    }
    void Field::fieldValue(QMap< QPair<int, int>, double> fieldValue)
    {
        m_field_value = fieldValue ;
    }
    
    void Field::fieldValue(QPair<int, int> key, double value)
    {
        m_field_value[key] = value;
    }
    
}
