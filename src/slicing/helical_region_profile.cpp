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

struct PendingProfileBand {
    RegionType region_type    = RegionType::kUnknown;
    double revolutions        = 0.0;
    Distance configured_pitch = 0.0 * mm;
};

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

double roundedProfileRevolutions(double raw_revolutions, double max_revolutions, HelicalPathZClipRounding rounding) {
    double revolutions = raw_revolutions;
    if (rounding == HelicalPathZClipRounding::kCompleteRevolution) {
        revolutions = std::ceil(raw_revolutions - kRevolutionTolerance);
    }
    else if (rounding == HelicalPathZClipRounding::kLastFullRevolution) {
        revolutions = std::floor(raw_revolutions + kRevolutionTolerance);
    }

    return std::clamp(revolutions, 0.0, max_revolutions);
}

void appendBand(HelicalRegionProfile& profile, RegionType region_type, double revolutions, Distance pitch) {
    if (revolutions <= kRevolutionTolerance) { return; }

    const HelicalRegionProfileBand previous =
        profile.bands.isEmpty() ? HelicalRegionProfileBand {} : profile.bands.back();
    const double start_revolutions = profile.bands.isEmpty() ? 0.0 : previous.start_revolutions + previous.revolutions;
    const Distance start_z         = profile.bands.isEmpty() ? profile.start_z : previous.endZ();

    profile.bands.push_back(HelicalRegionProfileBand {region_type, start_revolutions, revolutions, start_z, pitch});
}

void appendBand(HelicalRegionProfile& profile, RegionType region_type, int revolutions, Distance pitch) {
    appendBand(profile, region_type, static_cast<double>(revolutions), pitch);
}

void addBoundary(QVector<double>& boundaries, double boundary, double total_revolutions) {
    const double clamped_boundary = std::clamp(boundary, 0.0, total_revolutions);
    const auto existing = std::find_if(boundaries.cbegin(), boundaries.cend(), [clamped_boundary](double value) {
        return std::abs(value - clamped_boundary) <= kRevolutionTolerance;
    });
    if (existing == boundaries.cend()) { boundaries.push_back(clamped_boundary); }
}

RegionType boundedRegionType(double revolutions, double total_revolutions, int perimeter_revolutions,
                             int inset_revolutions) {
    if (perimeter_revolutions > 0 &&
        revolutions >= total_revolutions - static_cast<double>(perimeter_revolutions) - kRevolutionTolerance) {
        return RegionType::kPerimeter;
    }

    if (inset_revolutions > 0 && revolutions >= total_revolutions -
                                                    static_cast<double>(perimeter_revolutions + inset_revolutions) -
                                                    kRevolutionTolerance) {
        return RegionType::kInset;
    }

    if (perimeter_revolutions > 0 && revolutions < static_cast<double>(perimeter_revolutions) - kRevolutionTolerance) {
        return RegionType::kPerimeter;
    }

    if (inset_revolutions > 0 &&
        revolutions < static_cast<double>(perimeter_revolutions + inset_revolutions) - kRevolutionTolerance) {
        return RegionType::kInset;
    }

    return RegionType::kInfill;
}

Distance configuredPitchForRegion(RegionType region_type, Distance perimeter_pitch, Distance inset_pitch,
                                  Distance infill_pitch) {
    switch (region_type) {
        case RegionType::kPerimeter:
            return perimeter_pitch;
        case RegionType::kInset:
            return inset_pitch;
        case RegionType::kInfill:
        default:
            return infill_pitch;
    }
}

void appendPendingBand(QVector<PendingProfileBand>& bands, RegionType region_type, double revolutions,
                       Distance configured_pitch) {
    if (revolutions <= kRevolutionTolerance) { return; }

    if (!bands.isEmpty() && bands.back().region_type == region_type &&
        bands.back().configured_pitch == configured_pitch) {
        bands.back().revolutions += revolutions;
        return;
    }

    bands.push_back(PendingProfileBand {region_type, revolutions, configured_pitch});
}

HelicalRegionProfileResult buildRetainedProfileForRoundedRevolutions(const HelicalRegionProfileParameters& params,
                                                                     Distance start_z, Distance top_z,
                                                                     double total_revolutions) {
    HelicalRegionProfileResult result;
    result.profile.start_z         = start_z;
    result.profile.available_top_z = top_z;

    if (top_z <= start_z) {
        result.reason = "Helical retained profile top Z must be greater than start Z.";
        return result;
    }

    if (total_revolutions <= kRevolutionTolerance) {
        result.reason = "Helical retained profile contains no positive full revolutions after z clipping.";
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

    QVector<double> boundaries;
    addBoundary(boundaries, 0.0, total_revolutions);
    addBoundary(boundaries, static_cast<double>(params.perimeter_revolutions), total_revolutions);
    addBoundary(boundaries, static_cast<double>(params.perimeter_revolutions + params.inset_revolutions),
                total_revolutions);
    addBoundary(boundaries,
                total_revolutions - static_cast<double>(params.perimeter_revolutions + params.inset_revolutions),
                total_revolutions);
    addBoundary(boundaries, total_revolutions - static_cast<double>(params.perimeter_revolutions), total_revolutions);
    addBoundary(boundaries, total_revolutions, total_revolutions);
    std::sort(boundaries.begin(), boundaries.end());

    QVector<PendingProfileBand> pending_bands;
    for (int i = 1, end = boundaries.size(); i < end; ++i) {
        const double run_start = boundaries[i - 1];
        const double run_end   = boundaries[i];
        if (run_end <= run_start + kRevolutionTolerance) { continue; }

        const double midpoint = (run_start + run_end) / 2.0;
        const RegionType region_type =
            boundedRegionType(midpoint, total_revolutions, params.perimeter_revolutions, params.inset_revolutions);
        const Distance configured_pitch =
            configuredPitchForRegion(region_type, perimeter_pitch, inset_pitch, infill_pitch);
        appendPendingBand(pending_bands, region_type, run_end - run_start, configured_pitch);
    }

    const Distance available_height = top_z - start_z;
    Distance fixed_region_height;
    double infill_revolutions = 0.0;
    for (const PendingProfileBand& band : pending_bands) {
        if (band.region_type == RegionType::kInfill) { infill_revolutions += band.revolutions; }
        else { fixed_region_height += band.configured_pitch * band.revolutions; }
    }

    Distance bounded_infill_pitch = infill_pitch;
    if (infill_revolutions > kRevolutionTolerance) {
        // Z clip rounding has already selected a hard full-revolution endpoint.
        // Keep shell pitches fixed and let only the middle infill span absorb the remaining height.
        const Distance infill_height = available_height - fixed_region_height;
        if (infill_height <= 0) {
            result.reason = "Helical retained profile shell regions exceed the rounded z clip height.";
            return result;
        }

        bounded_infill_pitch = infill_height / infill_revolutions;
    }
    else if (fixed_region_height > available_height) {
        result.reason = "Helical retained profile shell regions exceed the rounded z clip height.";
        return result;
    }

    for (const PendingProfileBand& band : pending_bands) {
        const Distance pitch = band.region_type == RegionType::kInfill ? bounded_infill_pitch : band.configured_pitch;
        appendBand(result.profile, band.region_type, band.revolutions, pitch);
    }

    if (result.profile.isEmpty()) { result.reason = "Helical region profile contains no positive-revolution bands."; }

    return result;
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

HelicalRegionProfileResult buildRetainedHelicalRegionProfile(HelicalRegionProfileParameters params, Distance start_z,
                                                             Distance top_z) {
    params.start_z = start_z;
    params.top_z   = top_z;
    return buildHelicalRegionProfile(params);
}

HelicalRegionProfileResult buildRetainedHelicalRegionProfile(HelicalRegionProfileParameters params, Distance start_z,
                                                             Distance top_z, HelicalPathZClipRounding z_clip_rounding) {
    HelicalRegionProfileResult result = buildRetainedHelicalRegionProfile(params, start_z, top_z);
    if (!result.valid() || z_clip_rounding == HelicalPathZClipRounding::kExactIntersection) { return result; }

    const double raw_top_revolutions = result.profile.revolutionsAtZ(top_z);
    const double rounded_top_revolutions =
        roundedProfileRevolutions(raw_top_revolutions, result.profile.totalRevolutions(), z_clip_rounding);
    if (rounded_top_revolutions <= kRevolutionTolerance) {
        result.reason = "Helical retained profile contains no positive full revolutions after z clipping.";
        return result;
    }

    const Distance rounded_top_z = result.profile.zAtRevolutions(rounded_top_revolutions);
    return buildRetainedProfileForRoundedRevolutions(params, start_z, rounded_top_z, rounded_top_revolutions);
}
}  // namespace ORNL
