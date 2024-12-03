// Main Module
#include "geometry/segment_base.h"
#include <gcode/writers/writer_base.h>
#include "geometry/segments/travel.h"

// Local
#include "configs/settings_base.h"

namespace ORNL {
    SegmentBase::SegmentBase(Point start, Point end) : m_start(start), m_end(end), m_sb(new SettingsBase())
    {
        m_sb->setSetting(Constants::SegmentSettings::kIsRegionStartSegment, false);

        m_sb->setSetting(Constants::SegmentSettings::kRegionType, RegionType::kUnknown);
        m_sb->setSetting(Constants::SegmentSettings::kPathModifiers, PathModifiers::kNone);

        m_non_build_modifiers = PathModifiers::kCoasting | PathModifiers::kForwardTipWipe |
                PathModifiers::kPerimeterTipWipe | PathModifiers::kReverseTipWipe | PathModifiers::kSpiralLift;

        //default to extruder 0
        QVector<int> nozzles = QVector<int>();
        nozzles.append(0);
        m_sb->setSetting(Constants::SegmentSettings::kExtruders, nozzles);
    }

    Point SegmentBase::start() const {
        return m_start;
    }

    Point SegmentBase::midpoint() const
    {
        return (m_start + m_end) / 2.;
    }

    Point SegmentBase::end() const {
        return m_end;
    }

    uint SegmentBase::layerNumber() {
        return m_layer_num;
    }

    uint SegmentBase::lineNumber() {
        return m_line_num;
    }

    float SegmentBase::displayWidth() {
        return m_display_width;
    }

    float SegmentBase::displayHeight() {
        return m_display_height;
    }

    SegmentDisplayType SegmentBase::displayType() {
        return m_display_type;
    }

    QColor SegmentBase::color() {
        return m_color;
    }

    void SegmentBase::setGCodeInfo(float display_width, float display_length, float display_height, SegmentDisplayType type, QColor color, uint line_num, uint layer_num) {
        m_display_width = display_width;
        m_display_length = display_length;
        m_display_height = display_height;
        m_display_type = type;
        m_color = color;
        m_line_num = line_num;
        m_layer_num = layer_num;
    }

    float SegmentBase::getGCodeWidth()
    {
        return m_display_width;
    }

    void SegmentBase::setGCodeWidth(float display_width)
    {
        m_display_width = display_width;
    }

    void SegmentBase::createGraphic(std::vector<float>& vertices, std::vector<float>& normals, std::vector<float>& colors) {
        // NOP
    }

    void SegmentBase::setStart(Point start) {
        m_start = start;
    }

    void SegmentBase::setEnd(Point end) {
        m_end = end;
    }

    void SegmentBase::reverse() {
        std::swap(m_start, m_end);

        QVector<QVector3D> normals = m_sb->setting<QVector<QVector3D>>(Constants::SegmentSettings::kTilt);
        if (!normals.isEmpty())
        {
            normals.swapItemsAt(0, 1);
            m_sb->setSetting(Constants::SegmentSettings::kTilt, normals);
        }
    }

    QSharedPointer<SettingsBase> SegmentBase::getSb() const {
        return m_sb;
    }

    void SegmentBase::setSb(const QSharedPointer<SettingsBase>& sb) {
        m_sb = sb;
    }

    void SegmentBase::rotate(QQuaternion rotation)
    {
        //rotate each point
        QVector3D start_vec = m_start.toQVector3D();
        QVector3D result_start = rotation.rotatedVector(start_vec);
        m_start = Point(result_start);

        QVector3D end_vec = m_end.toQVector3D();
        QVector3D result_end = rotation.rotatedVector(end_vec);
        m_end = Point(result_end);
    }

    void SegmentBase::shift(Point shift)
    {
        m_start = m_start + shift;
        m_end   = m_end   + shift;
    }

    bool SegmentBase::isPrintingSegment()
    {
        if (dynamic_cast<TravelSegment*>(this) != nullptr ||
                (int)(m_sb->setting<uint>(Constants::SegmentSettings::kPathModifiers) & (uint)m_non_build_modifiers) != 0)
            return false;

        return true;
    }

    void SegmentBase::setNozzles(QVector<int> nozzles)
    {
        m_sb->setSetting<QVector<int>>("extruders", nozzles);
    }

    void SegmentBase::addNozzle(int nozzle)
    {
        QVector<int> nozzles = m_sb->setting<QVector<int>>(Constants::SegmentSettings::kExtruders);
        if (!nozzles.contains(nozzle))
        {
            nozzles.append(nozzle);
            m_sb->setSetting<QVector<int>>(Constants::SegmentSettings::kExtruders, nozzles);
        }
    }

    Distance SegmentBase::length()
    {
        return m_start.distance(m_end);
    }
}  // namespace ORNL
