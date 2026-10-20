#include <QSharedPointer>
#include <cstdlib>

#include "configs/settings_base.h"
#include "geometry/connection.h"
#include "geometry/path.h"
#include "geometry/path_modifier.h"
#include "geometry/point.h"
#include "geometry/segments/line.h"
#include "test_utils.h"
#include "units/unit.h"
#include "utilities/constants.h"

namespace {

void appendLine(ORNL::Path& path, const ORNL::Point& start, const ORNL::Point& end) {
    path.append(QSharedPointer<ORNL::LineSegment>::create(start, end));
}

QSharedPointer<ORNL::SettingsBase> sharpCornerSettings(ORNL::Distance close_points_threshold, bool enabled = true) {
    QSharedPointer<ORNL::SettingsBase> settings = QSharedPointer<ORNL::SettingsBase>::create();
    settings->setSetting(ORNL::PS::SpecialModes::kEnableSharpCornerExtension, enabled);
    settings->setSetting(ORNL::PS::SpecialModes::kSharpCornerExtensionAngle, ORNL::Angle(M_PI / 3.0));
    settings->setSetting(ORNL::PS::SpecialModes::kSharpCornerExtensionDistance, ORNL::Distance(2.0));
    settings->setSetting(ORNL::PS::SpecialModes::kSharpCornerClosePointsThreshold, close_points_threshold);
    settings->setSetting(ORNL::PS::SpecialModes::kSharpCornerSharpeningLegLength, ORNL::Distance(4.0));
    return settings;
}
}  // namespace

int main() {
    bool passed = true;

    ORNL::Path direct_corner;
    appendLine(direct_corner, ORNL::Point(-10.0f, 5.0f, 0.0f), ORNL::Point(0.0f, 0.0f, 0.0f));
    appendLine(direct_corner, ORNL::Point(0.0f, 0.0f, 0.0f), ORNL::Point(-10.0f, -5.0f, 0.0f));

    ORNL::PathModifierGenerator::GenerateSharpCornerExtension(direct_corner, sharpCornerSettings(ORNL::Distance()));
    passed &= ORNL::Testing::expect(direct_corner.size() == 4,
                                    "Expected direct sharp corner to insert two sharpened segments.");
    passed &= ORNL::Testing::expect(ORNL::Testing::near2DPoint(direct_corner[0]->end(), -3.5777087, 1.7888544),
                                    "Expected direct corner previous leg to be cut back.");
    passed &= ORNL::Testing::expect(ORNL::Testing::near2DPoint(direct_corner[1]->end(), 2.0, 0.0),
                                    "Expected direct corner merge point to extend along the bisector.");
    passed &= ORNL::Testing::expect(ORNL::Testing::near2DPoint(direct_corner[2]->end(), -3.5777087, -1.7888544),
                                    "Expected direct corner next leg to be cut back.");
    passed &= ORNL::Testing::expect(direct_corner[1]->start() == direct_corner[0]->end() &&
                                        direct_corner[2]->start() == direct_corner[1]->end() &&
                                        direct_corner[3]->start() == direct_corner[2]->end(),
                                    "Expected direct sharpened corner to remain continuous.");

    ORNL::Path connected_corner;
    appendLine(connected_corner, ORNL::Point(-10.0f, 5.0f, 0.0f), ORNL::Point(-1.0f, 0.5f, 0.0f));
    appendLine(connected_corner, ORNL::Point(-1.0f, 0.5f, 0.0f), ORNL::Point(-1.0f, -0.5f, 0.0f));
    appendLine(connected_corner, ORNL::Point(-1.0f, -0.5f, 0.0f), ORNL::Point(-10.0f, -5.0f, 0.0f));
    connected_corner[1]->getSb()->setSetting(ORNL::SS::kIsConnection, true);
    connected_corner[1]->getSb()->setSetting(ORNL::SS::kConnectionRegion, ORNL::RegionType::kInset);

    ORNL::PathModifierGenerator::GenerateSharpCornerExtension(connected_corner,
                                                              sharpCornerSettings(ORNL::Distance(2.0)));
    passed &= ORNL::Testing::expect(connected_corner.size() == 4,
                                    "Expected close connector to be removed and replaced by sharpened geometry.");
    if (connected_corner.size() == 4) {
        for (int i = 0; i < connected_corner.size(); ++i) {
            const auto settings = connected_corner[i]->getSb();
            const bool is_connection =
                settings->contains(ORNL::SS::kIsConnection) && settings->setting<bool>(ORNL::SS::kIsConnection);
            passed &= ORNL::Testing::expect(is_connection == (i == 1 || i == 2),
                                            "Only the two sharpened replacement legs should be connections.");
            if (is_connection) {
                passed &= ORNL::Testing::expect(
                    settings->setting<ORNL::RegionType>(ORNL::SS::kConnectionRegion) == ORNL::RegionType::kInset,
                    "Sharpened replacements must retain the connection destination.");
            }
        }
    }
    passed &= ORNL::Testing::expect(ORNL::Testing::near2DPoint(connected_corner[0]->end(), -4.5777087, 2.2888544),
                                    "Expected connected corner previous leg to be cut back.");
    passed &= ORNL::Testing::expect(ORNL::Testing::near2DPoint(connected_corner[1]->end(), 2.0, 0.0),
                                    "Expected connected corner merge point to extend from the leg-axis intersection.");
    passed &= ORNL::Testing::expect(ORNL::Testing::near2DPoint(connected_corner[2]->end(), -4.5777087, -2.2888544),
                                    "Expected connected corner next leg to be cut back.");
    passed &= ORNL::Testing::expect(connected_corner[1]->start() == connected_corner[0]->end() &&
                                        connected_corner[2]->start() == connected_corner[1]->end() &&
                                        connected_corner[3]->start() == connected_corner[2]->end(),
                                    "Expected connected sharpened corner to remain continuous.");

    ORNL::Path threshold_rejected_corner;
    appendLine(threshold_rejected_corner, ORNL::Point(-10.0f, 5.0f, 0.0f), ORNL::Point(-1.0f, 0.5f, 0.0f));
    appendLine(threshold_rejected_corner, ORNL::Point(-1.0f, 0.5f, 0.0f), ORNL::Point(-1.0f, -0.5f, 0.0f));
    appendLine(threshold_rejected_corner, ORNL::Point(-1.0f, -0.5f, 0.0f), ORNL::Point(-10.0f, -5.0f, 0.0f));

    ORNL::PathModifierGenerator::GenerateSharpCornerExtension(threshold_rejected_corner,
                                                              sharpCornerSettings(ORNL::Distance(0.5)));
    passed &= ORNL::Testing::expect(threshold_rejected_corner.size() == 3,
                                    "Expected connector longer than close-points threshold to remain unchanged.");

    ORNL::Path local_corner;
    appendLine(local_corner, ORNL::Point(-10.0f, 5.0f, 0.0f), ORNL::Point(0.0f, 0.0f, 0.0f));
    appendLine(local_corner, ORNL::Point(0.0f, 0.0f, 0.0f), ORNL::Point(-10.0f, -5.0f, 0.0f));

    QSharedPointer<ORNL::SettingsBase> local_settings = sharpCornerSettings(ORNL::Distance());
    local_corner[0]->getSb()->populate(local_settings);
    local_corner[1]->getSb()->populate(local_settings);

    ORNL::PathModifierGenerator::GenerateSharpCornerExtension(local_corner,
                                                              sharpCornerSettings(ORNL::Distance(), false));
    passed &=
        ORNL::Testing::expect(local_corner.size() == 4,
                              "Expected segment-local sharp-corner settings to override disabled fallback settings.");

    ORNL::Path connection;
    appendLine(connection, ORNL::Point(0.0f, 0.0f, 0.0f), ORNL::Point(10.0f, 0.0f, 0.0f));
    auto connection_settings = connection[0]->getSb();
    connection_settings->setSetting(ORNL::SS::kIsConnection, true);
    connection_settings->setSetting(ORNL::SS::kConnectionRegion, ORNL::RegionType::kInset);
    connection_settings->setSetting(ORNL::SS::kWidth, ORNL::Distance(1));
    connection_settings->setSetting(ORNL::SS::kHeight, ORNL::Distance(1));
    connection_settings->setSetting(ORNL::SS::kSpeed, ORNL::Velocity(10));
    connection_settings->setSetting(ORNL::SS::kAccel, ORNL::Acceleration(10));
    connection_settings->setSetting(ORNL::SS::kExtruderSpeed, ORNL::AngularVelocity(10));
    connection_settings->setSetting(ORNL::SS::kMaterialNumber, 0);
    connection_settings->setSetting(ORNL::SS::kRegionType, ORNL::RegionType::kPerimeter);
    ORNL::PathModifierGenerator::GenerateInitialStartupWithRampUp(connection, ORNL::Distance(4), ORNL::Velocity(2),
                                                                  ORNL::Velocity(5), ORNL::AngularVelocity(2),
                                                                  ORNL::AngularVelocity(5), 2, false, 100);
    ORNL::PathModifierGenerator::GenerateSlowdown(connection, ORNL::Distance(2), ORNL::Distance(0), ORNL::Distance(1),
                                                  ORNL::Velocity(3), ORNL::AngularVelocity(3), false, 100);
    passed &= ORNL::Testing::expect(connection.size() >= 5, "Expected startup and slowdown to split the connection.");
    for (const auto& segment : connection) {
        passed &= ORNL::Testing::expect(
            segment->getSb()->contains(ORNL::SS::kIsConnection) &&
                segment->getSb()->setting<bool>(ORNL::SS::kIsConnection) &&
                segment->getSb()->setting<ORNL::RegionType>(ORNL::SS::kConnectionRegion) == ORNL::RegionType::kInset,
            "Every startup, slowdown and cutoff fragment must retain its connection identity and destination.");
    }
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
