#pragma once

#include <QSharedPointer>
#include <QVector>

#include <functional>

namespace ORNL {
class SegmentBase;
}

namespace ORNL::GCodeSegmentFilter {
//! \brief Returns whether a segment is a non-build path modifier rather than deposited part material.
bool isNonBuildModifierSegment(const QSharedPointer<SegmentBase>& segment);

//! \brief Tags non-external printable and non-build modifier segments so the preview can hide them.
//! \param gcode The layers of G-code segments to analyze and tag.
//! \param should_cancel Optional callback returning true if processing should be cancelled early.
void tagInternalSegments(const QVector<QVector<QSharedPointer<SegmentBase>>>& gcode,
                         const std::function<bool()>& should_cancel = nullptr);
}  // namespace ORNL::GCodeSegmentFilter
