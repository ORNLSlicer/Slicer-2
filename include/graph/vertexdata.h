#ifndef VERTEXDATA_H
#define VERTEXDATA_H

#include <QSharedPointer>

#include "../geometry/point.h"
#include "circle.h"

namespace ORNL
{

    class VertexData
    {
    public:
        VertexData()=default;
        virtual ~VertexData(){}
    };

    class VertexLocationData : public VertexData
    {
    public:
        VertexLocationData()=default;
        VertexLocationData(QSharedPointer<Point>point);
        QSharedPointer<Point> m_point;
    };

    class VertexCircleData : public VertexData
    {
    public:
        VertexCircleData()=default;
        VertexCircleData(QSharedPointer<Circle>circle);
        QSharedPointer<Circle> m_circle;
    };

    class VertexLocationCircleData : public VertexData
    {
    public:
        VertexLocationCircleData()=default;
        VertexLocationCircleData(QSharedPointer<Point>point);
        VertexLocationCircleData(QSharedPointer<Circle>circle);
        VertexLocationCircleData(QSharedPointer<Point>point, QSharedPointer<Circle>circle);
        QSharedPointer<Circle> m_circle;
        QSharedPointer<Point> m_point;
    };

}
#endif // VERTEXDATA_H
