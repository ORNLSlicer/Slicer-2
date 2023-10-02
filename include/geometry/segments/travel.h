#ifndef TRAVEL_H
#define TRAVEL_H

#include "geometry/segment_base.h"

#include <gcode/writers/writer_base.h>

namespace ORNL {
    /*!
     *  \class TravelSegment
     *  \brief Segment for travel movements.
     */
    class TravelSegment : public SegmentBase {
        public:
            //! \brief Constructor
            //! \param start point
            //! \param end point
            //! \param liftType type of lift to apply to travel (only raise, only lower, both, none)
            //! Default lift type is both
            TravelSegment(Point start, Point end, TravelLiftType liftType = TravelLiftType::kBoth);

            //! \brief Set the travel lift type
            void setLiftType(TravelLiftType newLiftType);

            //! \brief Clone
            QSharedPointer<SegmentBase> clone() const;

            //! \brief Write gcode for travel segment.
            QString writeGCode(QSharedPointer<WriterBase> writer);

            //! \brief returns minimum z-coordinate of the travel
            float getMinZ() override;

        private:
            //! private type to track what type of lift to apply to this particular travel
            TravelLiftType m_lift_type;
    };
}


#endif // TRAVEL_H
