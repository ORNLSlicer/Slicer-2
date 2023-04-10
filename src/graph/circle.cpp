#include "graph/circle.h"

namespace ORNL 
{
    
    
    Circle::Circle(Point center, int64_t radius) :
            m_center(center), m_radius(radius)
    {
        m_pin_polygon_id = -1;
        m_pin_vertex_id = -1;
        m_is_secondary = false;
    }
    
    
    // * inversiveDistance - Make sure the passed in argument is not a null pointer,
    // *     since there is no sentinel value for this function that would not overlap
    // *     with the range of possible values for inversive distances and thus there
    // *     is no proper value to be returned from a null check here.
    
    void Circle::radius(int64_t radius)
    {
        m_radius = radius;
    }
    
    int64_t Circle::radius()
    {
        return m_radius;
    }
    
    void Circle::center(Point center)
    {
        m_center = center;
    }
    
    Point Circle::center()
    {
        return m_center;
    }
    
    void Circle::pinPolygonId(int64_t pin_polygon_id)
    {
        m_pin_polygon_id = pin_polygon_id;
    }
    
    int64_t Circle::pinPolygonId()
    {
        return m_pin_polygon_id;
    }
    
    void Circle::pinVertexId(int64_t pin_vertex_id)
    {
        m_pin_vertex_id = pin_vertex_id;
    }
    
    int64_t Circle::pinVertexId()
    {
        return m_pin_vertex_id;
    }
    
    bool Circle::isSecondary()
    {
        return m_is_secondary;
    }
    
    void Circle::isSecondary(bool is_secondary)
    {
        m_is_secondary = is_secondary;
    }
    double Circle::inversiveDistance(QSharedPointer<Circle> other)
    {
        double d = m_center.distance(other->center())();
        return ((d * d - m_radius * m_radius - other->radius() * other->radius())
                / (2 * m_radius * other->radius()));
    }
    
    //void Circle::Radius(int64_t r)
    //{
    //    radius = r;
    //}
}
