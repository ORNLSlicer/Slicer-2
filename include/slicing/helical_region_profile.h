#pragma once

#include <QString>
#include <QVector>

#include "units/unit.h"
#include "utilities/enums.h"

namespace ORNL {
struct HelicalRegionProfileBand {
    RegionType region_type   = RegionType::kUnknown;
    double start_revolutions = 0.0;
    double revolutions       = 0.0;
    Distance start_z;
    Distance pitch;

    double endRevolutions() const;
    Distance endZ() const;
};

struct HelicalRegionProfile {
    Distance start_z;
    Distance available_top_z;
    QVector<HelicalRegionProfileBand> bands;

    bool isEmpty() const;
    double totalRevolutions() const;
    Distance generatedTopZ() const;
    Distance minPitch() const;
    Distance zAtRevolutions(double revolutions) const;
    double revolutionsAtZ(Distance z) const;
    RegionType regionAtRevolutions(double revolutions) const;
};

struct HelicalRegionProfileParameters {
    Distance start_z;
    Distance top_z;
    Distance bead_width;
    int perimeter_revolutions = 0;
    int inset_revolutions     = 0;
    Distance perimeter_stepover;
    Distance inset_stepover;
    Distance infill_stepover;
    HelicalInfillRevolutionsRounding infill_revolutions_rounding = HelicalInfillRevolutionsRounding::kRound;
};

struct HelicalRegionProfileResult {
    HelicalRegionProfile profile;
    QString reason;

    bool valid() const;
};

HelicalRegionProfileResult buildHelicalRegionProfile(const HelicalRegionProfileParameters& params);
}  // namespace ORNL
