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

double infillRevolutions(const ORNL::HelicalRegionProfile& profile) {
    for (const ORNL::HelicalRegionProfileBand& band : profile.bands) {
        if (band.region_type == ORNL::RegionType::kInfill) { return band.revolutions; }
    }

    return 0.0;
}

bool matchesRegionSequence(const ORNL::HelicalRegionProfile& profile, const QVector<ORNL::RegionType>& regions) {
    if (profile.bands.size() != regions.size()) { return false; }

    for (int i = 0, end = regions.size(); i < end; ++i) {
        if (profile.bands[i].region_type != regions[i]) { return false; }
    }

    return true;
}

bool retainedProfileAnchorsBottomShellsToClippedStart() {
    const ORNL::HelicalRegionProfileResult result =
        ORNL::buildRetainedHelicalRegionProfile(defaultParams(), 10.0 * ORNL::mm, 42.0 * ORNL::mm);
    if (!result.valid() || result.profile.bands.size() != 5) { return false; }

    return result.profile.bands[0].region_type == ORNL::RegionType::kPerimeter &&
           near(result.profile.bands[0].start_revolutions, 0.0) &&
           nearDistance(result.profile.bands[0].start_z, 10.0 * ORNL::mm) &&
           result.profile.bands[1].region_type == ORNL::RegionType::kInset &&
           nearDistance(result.profile.bands[1].start_z, 12.0 * ORNL::mm);
}

bool retainedProfileRespectsInfillRevolutionsRounding() {
    ORNL::HelicalRegionProfileParameters params = defaultParams();
    params.infill_revolutions_rounding          = ORNL::HelicalInfillRevolutionsRounding::kRound;
    const ORNL::HelicalRegionProfileResult round_result =
        ORNL::buildRetainedHelicalRegionProfile(params, 10.0 * ORNL::mm, 42.0 * ORNL::mm);

    params.infill_revolutions_rounding = ORNL::HelicalInfillRevolutionsRounding::kFloor;
    const ORNL::HelicalRegionProfileResult floor_result =
        ORNL::buildRetainedHelicalRegionProfile(params, 10.0 * ORNL::mm, 42.0 * ORNL::mm);

    params.infill_revolutions_rounding = ORNL::HelicalInfillRevolutionsRounding::kCeil;
    const ORNL::HelicalRegionProfileResult ceil_result =
        ORNL::buildRetainedHelicalRegionProfile(params, 10.0 * ORNL::mm, 42.0 * ORNL::mm);

    return round_result.valid() && floor_result.valid() && ceil_result.valid() &&
           near(infillRevolutions(round_result.profile), 6.0) && near(infillRevolutions(floor_result.profile), 5.0) &&
           near(infillRevolutions(ceil_result.profile), 6.0) &&
           nearDistance(round_result.profile.generatedTopZ(), 44.0 * ORNL::mm) &&
           nearDistance(floor_result.profile.generatedTopZ(), 40.0 * ORNL::mm) &&
           nearDistance(ceil_result.profile.generatedTopZ(), 44.0 * ORNL::mm);
}

bool retainedLastFullKeepsTopPerimeterWithoutInsets() {
    ORNL::HelicalRegionProfileParameters params = defaultParams();
    params.inset_revolutions                    = 0;
    params.perimeter_stepover                   = 4.0 * ORNL::mm;
    params.inset_stepover                       = 4.0 * ORNL::mm;
    params.infill_stepover                      = 4.0 * ORNL::mm;

    const ORNL::HelicalRegionProfileResult result = ORNL::buildRetainedHelicalRegionProfile(
        params, 0.0 * ORNL::mm, 23.2 * ORNL::mm, ORNL::HelicalPathZClipRounding::kLastFullRevolution);
    if (!result.valid()) { return false; }

    return matchesRegionSequence(result.profile,
                                 QVector<ORNL::RegionType> {ORNL::RegionType::kPerimeter, ORNL::RegionType::kInfill,
                                                            ORNL::RegionType::kPerimeter}) &&
           near(result.profile.bands.last().start_revolutions, 4.0) &&
           nearDistance(result.profile.generatedTopZ(), 20.0 * ORNL::mm) &&
           near(result.profile.totalRevolutions(), 5.0);
}

bool retainedLastFullKeepsTopInsetAndPerimeter() {
    ORNL::HelicalRegionProfileParameters params = defaultParams();
    params.perimeter_stepover                   = 4.0 * ORNL::mm;
    params.inset_stepover                       = 4.0 * ORNL::mm;
    params.infill_stepover                      = 4.0 * ORNL::mm;

    const ORNL::HelicalRegionProfileResult result = ORNL::buildRetainedHelicalRegionProfile(
        params, 0.0 * ORNL::mm, 23.2 * ORNL::mm, ORNL::HelicalPathZClipRounding::kLastFullRevolution);
    if (!result.valid()) { return false; }

    return matchesRegionSequence(result.profile,
                                 QVector<ORNL::RegionType> {ORNL::RegionType::kPerimeter, ORNL::RegionType::kInset,
                                                            ORNL::RegionType::kInfill, ORNL::RegionType::kInset,
                                                            ORNL::RegionType::kPerimeter}) &&
           near(result.profile.bands[3].start_revolutions, 3.0) &&
           near(result.profile.bands[4].start_revolutions, 4.0) &&
           nearDistance(result.profile.generatedTopZ(), 20.0 * ORNL::mm) &&
           near(result.profile.totalRevolutions(), 5.0);
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
    passed &= expect(retainedProfileAnchorsBottomShellsToClippedStart(),
                     "Expected retained clipped starts to rebuild lower shell bands from the retained start.");
    passed &= expect(retainedProfileRespectsInfillRevolutionsRounding(),
                     "Expected retained clipped profiles to respect infill revolutions rounding.");
    passed &= expect(retainedLastFullKeepsTopPerimeterWithoutInsets(),
                     "Expected last-full retained clipping to keep the final revolution as perimeter.");
    passed &= expect(retainedLastFullKeepsTopInsetAndPerimeter(),
                     "Expected last-full retained clipping to keep the second-last inset and final perimeter.");

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
