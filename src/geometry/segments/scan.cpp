// Main Module
#include "geometry/segments/scan.h"

// Local
#include "configs/settings_base.h"

namespace ORNL {
    ScanSegment::ScanSegment(Point start, Point end) : SegmentBase(start, end) {
        // NOP
    }

    QSharedPointer<SegmentBase> ScanSegment::clone() const
    {
        return QSharedPointer<ScanSegment>::create(*this);
    }

    QString ScanSegment::writeGCode(QSharedPointer<WriterBase> writer) {
        Velocity speed = this->getSb()->setting<Velocity>(Constants::SegmentSettings::kSpeed);

        return writer->writeScan(m_end, speed, m_on_off);
    }

    void ScanSegment::setDataCollection(bool on) {
           m_on_off = on;
       }

    float ScanSegment::getMinZ()
    {
        //don't want this min to impact the min for an entire layer, but still need to return something
        //so return the largest possible number
        return std::numeric_limits<float>::max();
    }
}
