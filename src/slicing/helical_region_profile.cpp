#include "slicing/helical_region_profile.h"

#include <QStringBuilder>
#include <algorithm>
#include <cmath>
#include <limits>

#include "units/unit.h"
#include "utilities/enums.h"

namespace ORNL {
namespace {
constexpr double kRevolutionTolerance = 1.0e-9;

Distance resolvedPitch(const QString& label, Distance stepover, Distance bead_width, QString& reason) {
    if (stepover < 0) {
        reason = label % " cannot be negative. Use 0 to fall back to Default Bead Width.";
        return Distance();
    }

    if (stepover() == 0.0) { return bead_width; }

    return stepover;
}

double roundedInfillRevolutions(double revolutions, HelicalInfillRevolutionsRounding rounding) {
    switch (rounding) {
        case HelicalInfillRevolutionsRounding::kFloor:
            return std::floor(revolutions + kRevolutionTolerance);
        case HelicalInfillRevolutionsRounding::kCeil:
            return std::ceil(revolutions - kRevolutionTolerance);
        case HelicalInfillRevolutionsRounding::kRound:
        default:
            return std::round(revolutions);
    }
}

void appendBand(HelicalRegionProfile& profile, RegionType region_type, int revolutions, Distance pitch) {
    if (revolutions <= 0) { return; }

    const HelicalRegionProfileBand previous =
        profile.bands.isEmpty() ? HelicalRegionProfileBand {} : profile.bands.back();
    const double start_revolutions = profile.bands.isEmpty() ? 0.0 : previous.start_revolutions + previous.revolutions;
    const Distance start_z         = profile.bands.isEmpty() ? profile.start_z : previous.endZ();

    profile.bands.push_back(
        HelicalRegionProfileBand {region_type, start_revolutions, static_cast<double>(revolutions), start_z, pitch});
}
}  // namespace

double HelicalRegionProfileBand::endRevolutions() const {
    return start_revolutions + revolutions;
}

Distance HelicalRegionProfileBand::endZ() const {
    return start_z + pitch * revolutions;
}

bool HelicalRegionProfile::isEmpty() const {
    return bands.isEmpty();
}

double HelicalRegionProfile::totalRevolutions() const {
    return bands.isEmpty() ? 0.0 : bands.back().endRevolutions();
}

Distance HelicalRegionProfile::generatedTopZ() const {
    return bands.isEmpty() ? start_z : bands.back().endZ();
}

Distance HelicalRegionProfile::minPitch() const {
    if (bands.isEmpty()) { return Distance(); }

    Distance result = bands.first().pitch;
    for (const HelicalRegionProfileBand& band : bands) { result = std::min(result, band.pitch); }
    return result;
}

Distance HelicalRegionProfile::zAtRevolutions(double revolutions) const {
    if (bands.isEmpty() || revolutions <= 0.0) { return start_z; }

    for (const HelicalRegionProfileBand& band : bands) {
        if (revolutions <= band.endRevolutions() + kRevolutionTolerance) {
            const double band_revolutions = std::clamp(revolutions - band.start_revolutions, 0.0, band.revolutions);
            return band.start_z + band.pitch * band_revolutions;
        }
    }

    return generatedTopZ();
}

double HelicalRegionProfile::revolutionsAtZ(Distance z) const {
    if (bands.isEmpty() || z <= start_z) { return 0.0; }

    for (const HelicalRegionProfileBand& band : bands) {
        if (z <= band.endZ()) {
            if (band.pitch <= 0) { return band.start_revolutions; }

            const double band_revolutions = std::clamp(((z - band.start_z) / band.pitch)(), 0.0, band.revolutions);
            return band.start_revolutions + band_revolutions;
        }
    }

    return totalRevolutions();
}

RegionType HelicalRegionProfile::regionAtRevolutions(double revolutions) const {
    if (bands.isEmpty()) { return RegionType::kUnknown; }

    for (const HelicalRegionProfileBand& band : bands) {
        if (revolutions <= band.endRevolutions() + kRevolutionTolerance) { return band.region_type; }
    }

    return bands.back().region_type;
}

bool HelicalRegionProfileResult::valid() const {
    return reason.isEmpty() && !profile.isEmpty();
}

HelicalRegionProfileResult buildHelicalRegionProfile(const HelicalRegionProfileParameters& params) {
    HelicalRegionProfileResult result;
    result.profile.start_z         = params.start_z;
    result.profile.available_top_z = params.top_z;

    if (params.top_z <= params.start_z) {
        result.reason = "Helical profile top Z must be greater than start Z.";
        return result;
    }

    if (params.bead_width <= 0) {
        result.reason = "Default Bead Width must be greater than zero for helical region pitch fallback.";
        return result;
    }

    if (params.perimeter_revolutions < 0) {
        result.reason = "Perimeter Revolutions cannot be negative.";
        return result;
    }

    if (params.inset_revolutions < 0) {
        result.reason = "Inset Revolutions cannot be negative.";
        return result;
    }

    QString reason;
    const Distance perimeter_pitch =
        resolvedPitch("Perimeter Stepover", params.perimeter_stepover, params.bead_width, reason);
    if (!reason.isEmpty()) {
        result.reason = reason;
        return result;
    }

    const Distance inset_pitch = resolvedPitch("Inset Stepover", params.inset_stepover, params.bead_width, reason);
    if (!reason.isEmpty()) {
        result.reason = reason;
        return result;
    }

    const Distance infill_pitch = resolvedPitch("Infill Stepover", params.infill_stepover, params.bead_width, reason);
    if (!reason.isEmpty()) {
        result.reason = reason;
        return result;
    }

    const Distance available_height     = params.top_z - params.start_z;
    const Distance shell_height         = perimeter_pitch * static_cast<double>(params.perimeter_revolutions * 2) +
                                          inset_pitch * static_cast<double>(params.inset_revolutions * 2);
    const double raw_infill_revolutions = ((available_height - shell_height) / infill_pitch)();
    const double rounded_infill_revolutions =
        std::max(0.0, roundedInfillRevolutions(raw_infill_revolutions, params.infill_revolutions_rounding));
    const int infill_revolutions = static_cast<int>(rounded_infill_revolutions);

    appendBand(result.profile, RegionType::kPerimeter, params.perimeter_revolutions, perimeter_pitch);
    appendBand(result.profile, RegionType::kInset, params.inset_revolutions, inset_pitch);
    appendBand(result.profile, RegionType::kInfill, infill_revolutions, infill_pitch);
    appendBand(result.profile, RegionType::kInset, params.inset_revolutions, inset_pitch);
    appendBand(result.profile, RegionType::kPerimeter, params.perimeter_revolutions, perimeter_pitch);

    if (result.profile.isEmpty()) { result.reason = "Helical region profile contains no positive-revolution bands."; }

    return result;
}
}  // namespace ORNL
