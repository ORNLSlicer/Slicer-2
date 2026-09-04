#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "slicing/helical_region_profile.h"
#include "units/unit.h"
#include "utilities/enums.h"

namespace {
bool expect(bool condition, const std::string& message) {
    if (condition) { return true; }

    std::cerr << message << '\n';
    return false;
}

bool near(double actual, double expected, double tolerance = 1.0e-6) {
    return std::abs(actual - expected) <= tolerance;
}

bool nearDistance(ORNL::Distance actual, ORNL::Distance expected, double tolerance_mm = 1.0e-6) {
    return near(actual.to(ORNL::mm), expected.to(ORNL::mm), tolerance_mm);
}

ORNL::HelicalRegionProfileParameters defaultParams() {
    ORNL::HelicalRegionProfileParameters params;
    params.start_z                     = 0.0 * ORNL::mm;
    params.top_z                       = 42.0 * ORNL::mm;
    params.bead_width                  = 4.0 * ORNL::mm;
    params.perimeter_revolutions       = 1;
    params.inset_revolutions           = 1;
    params.perimeter_stepover          = 2.0 * ORNL::mm;
    params.inset_stepover              = 3.0 * ORNL::mm;
    params.infill_stepover             = 4.0 * ORNL::mm;
    params.infill_revolutions_rounding = ORNL::HelicalInfillRevolutionsRounding::kRound;
    return params;
}

bool buildsOrderedProfileWithCumulativeZ() {
    const ORNL::HelicalRegionProfileResult result = ORNL::buildHelicalRegionProfile(defaultParams());
    if (!result.valid() || result.profile.bands.size() != 5) { return false; }

    const QVector<ORNL::RegionType> expected_regions {ORNL::RegionType::kPerimeter, ORNL::RegionType::kInset,
                                                      ORNL::RegionType::kInfill, ORNL::RegionType::kInset,
                                                      ORNL::RegionType::kPerimeter};
    const QVector<double> expected_start_revolutions {0.0, 1.0, 2.0, 10.0, 11.0};
    const QVector<ORNL::Distance> expected_start_z {0.0 * ORNL::mm, 2.0 * ORNL::mm, 5.0 * ORNL::mm, 37.0 * ORNL::mm,
                                                    40.0 * ORNL::mm};

    bool passed = true;
    for (int i = 0, end = result.profile.bands.size(); i < end; ++i) {
        passed &= result.profile.bands[i].region_type == expected_regions[i];
        passed &= near(result.profile.bands[i].start_revolutions, expected_start_revolutions[i]);
        passed &= nearDistance(result.profile.bands[i].start_z, expected_start_z[i]);
    }

    return passed && near(result.profile.totalRevolutions(), 12.0) &&
           nearDistance(result.profile.generatedTopZ(), 42.0 * ORNL::mm);
}

bool usesBeadWidthOnlyForZeroStepovers() {
    ORNL::HelicalRegionProfileParameters params = defaultParams();
    params.perimeter_revolutions                = 1;
    params.inset_revolutions                    = 0;
    params.perimeter_stepover                   = 0.0 * ORNL::mm;
    params.infill_stepover                      = 0.0 * ORNL::mm;

    const ORNL::HelicalRegionProfileResult result = ORNL::buildHelicalRegionProfile(params);
    if (!result.valid() || result.profile.bands.size() != 3) { return false; }

    return nearDistance(result.profile.bands[0].pitch, 4.0 * ORNL::mm) &&
           nearDistance(result.profile.bands[1].pitch, 4.0 * ORNL::mm);
}

bool roundsDerivedInfillRevolutions() {
    ORNL::HelicalRegionProfileParameters params = defaultParams();
    params.top_z                                = 9.0 * ORNL::mm;
    params.perimeter_revolutions                = 0;
    params.inset_revolutions                    = 0;
    params.infill_stepover                      = 4.0 * ORNL::mm;

    params.infill_revolutions_rounding                  = ORNL::HelicalInfillRevolutionsRounding::kRound;
    const ORNL::HelicalRegionProfileResult round_result = ORNL::buildHelicalRegionProfile(params);

    params.infill_revolutions_rounding                  = ORNL::HelicalInfillRevolutionsRounding::kFloor;
    const ORNL::HelicalRegionProfileResult floor_result = ORNL::buildHelicalRegionProfile(params);

    params.infill_revolutions_rounding                 = ORNL::HelicalInfillRevolutionsRounding::kCeil;
    const ORNL::HelicalRegionProfileResult ceil_result = ORNL::buildHelicalRegionProfile(params);

    return round_result.valid() && floor_result.valid() && ceil_result.valid() &&
           near(round_result.profile.totalRevolutions(), 2.0) && near(floor_result.profile.totalRevolutions(), 2.0) &&
           near(ceil_result.profile.totalRevolutions(), 3.0);
}

bool allowsRoundedHeightOverrun() {
    ORNL::HelicalRegionProfileParameters params = defaultParams();
    params.top_z                                = 10.0 * ORNL::mm;
    params.perimeter_revolutions                = 0;
    params.inset_revolutions                    = 0;
    params.infill_stepover                      = 4.0 * ORNL::mm;

    params.infill_revolutions_rounding                  = ORNL::HelicalInfillRevolutionsRounding::kRound;
    const ORNL::HelicalRegionProfileResult round_result = ORNL::buildHelicalRegionProfile(params);

    params.infill_revolutions_rounding                  = ORNL::HelicalInfillRevolutionsRounding::kFloor;
    const ORNL::HelicalRegionProfileResult floor_result = ORNL::buildHelicalRegionProfile(params);

    params.infill_revolutions_rounding                 = ORNL::HelicalInfillRevolutionsRounding::kCeil;
    const ORNL::HelicalRegionProfileResult ceil_result = ORNL::buildHelicalRegionProfile(params);

    return round_result.valid() && floor_result.valid() && ceil_result.valid() &&
           nearDistance(round_result.profile.generatedTopZ(), 12.0 * ORNL::mm) &&
           nearDistance(floor_result.profile.generatedTopZ(), 8.0 * ORNL::mm) &&
           nearDistance(ceil_result.profile.generatedTopZ(), 12.0 * ORNL::mm);
}

bool clampsNonpositiveDerivedInfillToNoInfill() {
    ORNL::HelicalRegionProfileParameters params = defaultParams();
    params.top_z                                = 4.0 * ORNL::mm;
    params.perimeter_revolutions                = 1;
    params.inset_revolutions                    = 1;
    params.perimeter_stepover                   = 2.0 * ORNL::mm;
    params.inset_stepover                       = 3.0 * ORNL::mm;
    params.infill_stepover                      = 4.0 * ORNL::mm;

    const ORNL::HelicalRegionProfileResult result = ORNL::buildHelicalRegionProfile(params);
    if (!result.valid() || result.profile.bands.size() != 4) { return false; }

    for (const ORNL::HelicalRegionProfileBand& band : result.profile.bands) {
        if (band.region_type == ORNL::RegionType::kInfill) { return false; }
    }

    return nearDistance(result.profile.generatedTopZ(), 10.0 * ORNL::mm);
}

bool rejectsInvalidInputsWithReason() {
    ORNL::HelicalRegionProfileParameters params                 = defaultParams();
    params.perimeter_revolutions                                = -1;
    const ORNL::HelicalRegionProfileResult negative_revolutions = ORNL::buildHelicalRegionProfile(params);

    params                                                   = defaultParams();
    params.inset_stepover                                    = -1.0 * ORNL::mm;
    const ORNL::HelicalRegionProfileResult negative_stepover = ORNL::buildHelicalRegionProfile(params);

    return !negative_revolutions.valid() && !negative_revolutions.reason.isEmpty() && !negative_stepover.valid() &&
           !negative_stepover.reason.isEmpty();
}
}  // namespace

int main() {
    bool passed = true;

    passed &= expect(buildsOrderedProfileWithCumulativeZ(),
                     "Expected perimeter/inset/infill/inset/perimeter profile with cumulative Z.");
    passed &=
        expect(usesBeadWidthOnlyForZeroStepovers(), "Expected zero stepovers to fall back to Default Bead Width.");
    passed &= expect(roundsDerivedInfillRevolutions(), "Expected Round, Floor, and Ceil infill revolution behavior.");
    passed &=
        expect(allowsRoundedHeightOverrun(), "Expected Round and Ceil profile heights to overrun available top Z.");
    passed &= expect(clampsNonpositiveDerivedInfillToNoInfill(),
                     "Expected nonpositive derived infill revolutions to remove only the infill band.");
    passed &= expect(rejectsInvalidInputsWithReason(), "Expected invalid helical profile inputs to include a reason.");

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
