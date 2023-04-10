#include "graph/vertexdata.h"

namespace ORNL 
{
    
    VertexLocationData::VertexLocationData(QSharedPointer<Point> point)
    {
        m_point = point;
    }
    
    VertexCircleData::VertexCircleData(QSharedPointer<Circle>circle)
    {
        m_circle = circle;
    }
    
    VertexLocationCircleData::VertexLocationCircleData(QSharedPointer<Point> point)
    {
        m_point = point;
    }
    
    VertexLocationCircleData::VertexLocationCircleData(QSharedPointer<Circle> circle)
    {
        m_circle = circle;
    }
    
    VertexLocationCircleData::VertexLocationCircleData(QSharedPointer<Point> point, QSharedPointer<Circle> circle)
    {
        m_point =  point;
        m_circle = circle;
    }
    
}
