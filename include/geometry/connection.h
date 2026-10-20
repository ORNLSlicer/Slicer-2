#pragma once

#include "configs/settings_base.h"
#include "geometry/point.h"
#include "utilities/constants.h"
#include "utilities/enums.h"

namespace ORNL {
namespace Connection {
//! \brief Whether the edge leaving this point connects two contour loops.
inline bool startsAt(Point point) {
    const auto settings = point.getSettings();
    return settings != nullptr && settings->contains(SS::kIsConnection) && settings->setting<bool>(SS::kIsConnection);
}

//! \brief Marks the outgoing edge without changing settings shared with the original loop.
inline void markStart(Point& point) {
    const auto original = point.getSettings();
    auto settings =
        original == nullptr ? QSharedPointer<SettingsBase>::create() : QSharedPointer<SettingsBase>::create(*original);
    settings->setSetting(SS::kIsConnection, true);
    point.setSettings(settings);
}

//! \brief Carries connection identity through splits made by path modifiers.
inline void copySettings(const QSharedPointer<SettingsBase>& destination, const QSharedPointer<SettingsBase>& source) {
    if (source->contains(SS::kIsConnection)) {
        destination->setSetting(SS::kIsConnection, source->setting<bool>(SS::kIsConnection));
    }
    else if (source->contains(SS::kPathModifiers) &&
             source->setting<PathModifiers>(SS::kPathModifiers) == PathModifiers::kConnection) {
        destination->setSetting(SS::kIsConnection, true);
    }
    if (source->contains(SS::kConnectionRegion)) {
        destination->setSetting(SS::kConnectionRegion, source->setting<RegionType>(SS::kConnectionRegion));
    }
}
}  // namespace Connection
}  // namespace ORNL
