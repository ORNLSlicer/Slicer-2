#ifndef LINE_H
#define LINE_H

// Local
#include "geometry/segment_base.h"

namespace ORNL {
/*!
 *  \class LineSegment
 *  \brief Segment type for linear movements.
 */
class LineSegment : public SegmentBase {
  public:
    //! \brief Constructor.
    LineSegment(Point start, Point end);

    //! \brief Populates the passed OpenGL buffers with float data for a line.
    //! \param vertices: OpenGL vertex array to append to.
    //! \param normals: OpenGL normal array to append to.
    //! \param colors: OpenGL color array to append to.
    void createGraphic(std::vector<float>& vertices, std::vector<float>& normals, std::vector<float>& colors) override;

    //! \brief Clone
    QSharedPointer<SegmentBase> clone() const override;

    //! \brief Write Gcode for a line segment.
    QString writeGCode(QSharedPointer<WriterBase> writer) override;

    //! \brief returns minimum z-coordinate of the line
    float getMinZ() override;
};
} // namespace ORNL

#endif // LINE_H
