#include <QCoreApplication>
#include <array>
#include <cstdlib>
#include <stdexcept>

#include "configs/settings_base.h"
#include "managers/settings/settings_version_control.h"
#include "test_utils.h"
#include "utilities/constants.h"
#include "utilities/enums.h"

namespace {
constexpr float kTolerance = 1.0e-3f;

bool connectedContourSettingsMigrate() {
    struct SettingRename {
        const char* legacy_key;
        QString current_key;
        bool value;
    };
    const std::array<SettingRename, 7> renames {{
        {"spiral_perimeter", ORNL::PS::Perimeter::kConnectPerimeters, true},
        {"spiral_perimeter_complete_path_before_connecting", ORNL::PS::Perimeter::kCompleteLoopBeforeConnecting, false},
        {"spiral_perimeter_branch_after_tip_wipe", ORNL::PS::Perimeter::kConnectAfterTipWipe, true},
        {"spiral_perimeter_connect_to_insets", ORNL::PS::Perimeter::kConnectToInsets, true},
        {"spiral_inset", ORNL::PS::Inset::kConnectInsets, true},
        {"spiral_inset_complete_path_before_connecting", ORNL::PS::Inset::kCompleteLoopBeforeConnecting, true},
        {"spiral_inset_branch_after_tip_wipe", ORNL::PS::Inset::kConnectAfterTipWipe, false},
    }};
    fifojson legacy = fifojson::object({{"enable_spiral_perimeter", true}, {"unrelated_setting", 42}});
    for (const auto& rename : renames) { legacy[rename.legacy_key] = rename.value; }
    fifojson collision                                               = legacy;
    collision[ORNL::PS::Perimeter::kConnectPerimeters.toStdString()] = false;

    fifojson document;
    document[ORNL::Constants::SettingFileStrings::kHeader][ORNL::Constants::SettingFileStrings::kVersion] = 14.0;
    document[ORNL::Constants::SettingFileStrings::kSettings] = fifojson::array({legacy, collision});
    double version                                           = 14.0;
    ORNL::SettingsVersionControl::rollSettingsForward(version, document);
    bool passed = ORNL::Testing::expect(
        version == 15.0 &&
            document[ORNL::Constants::SettingFileStrings::kHeader][ORNL::Constants::SettingFileStrings::kVersion] ==
                15.0,
        "Connected contour migration should update both version fields.");
    for (const auto& group : document[ORNL::Constants::SettingFileStrings::kSettings]) {
        for (const auto& rename : renames) {
            passed &= ORNL::Testing::expect(
                !group.contains(rename.legacy_key) && group.contains(rename.current_key.toStdString()),
                "Legacy contour keys should be replaced in every group.");
        }
        passed &=
            ORNL::Testing::expect(group.at("enable_spiral_perimeter") == true && group.at("unrelated_setting") == 42,
                                  "Contour migration must preserve spiral lift and unrelated settings.");
    }
    const auto& migrated = document[ORNL::Constants::SettingFileStrings::kSettings][0];
    for (const auto& rename : renames) {
        passed &= ORNL::Testing::expect(migrated.at(rename.current_key.toStdString()) == rename.value,
                                        "Contour migration must retain both enabled and disabled values.");
    }
    passed &= ORNL::Testing::expect(document[ORNL::Constants::SettingFileStrings::kSettings][1].at(
                                        ORNL::PS::Perimeter::kConnectPerimeters.toStdString()) == false,
                                    "Current settings must take precedence when legacy keys coexist.");
    const fifojson before_repeat = document;
    ORNL::SettingsVersionControl::rollSettingsForward(version, document);
    passed &= ORNL::Testing::expect(document == before_repeat, "Settings migration should be idempotent.");

    ORNL::SettingsBase local_settings;
    local_settings.json(fifojson::array({legacy}));
    passed &= ORNL::Testing::expect(local_settings.json()[0] == migrated,
                                    "Unversioned part and layer-range settings must use the same contour migration.");
    local_settings.setSetting(ORNL::PS::Perimeter::kConnectPerimeters, false);
    local_settings.populate(fifojson::array({legacy}));
    passed &= ORNL::Testing::expect(local_settings.setting<bool>(ORNL::PS::Perimeter::kConnectPerimeters),
                                    "Legacy overrides must be renamed before merging onto current defaults.");
    return passed;
}
}  // namespace

int main(int argc, char* argv[]) {
    QCoreApplication app(argc, argv);

    if (!connectedContourSettingsMigrate()) return EXIT_FAILURE;

    const std::string inner_radius_key = ORNL::PS::Slicing::kCylinderInnerRadius.toStdString();
    const std::string layer_height_key = ORNL::PS::Layer::kLayerHeight.toStdString();
    const std::string arc_length_key   = ORNL::PS::Helical::kHelicalToolStartArcLengthOffset.toStdString();
    const double inner_radius          = (100.0 * ORNL::mm)();
    const double layer_height          = (2.0 * ORNL::mm)();
    const double first_radius          = inner_radius + (layer_height / 2.0);

    fifojson v10_settings;
    v10_settings[ORNL::Constants::SettingFileStrings::kHeader][ORNL::Constants::SettingFileStrings::kVersion] = 10.0;
    v10_settings[ORNL::Constants::SettingFileStrings::kSettings] =
        fifojson::array({fifojson::object({{"helical_path_start_angle", 1.74532925},
                                           {inner_radius_key, inner_radius},
                                           {layer_height_key, layer_height}})});
    double v10_version = 10.0;
    ORNL::SettingsVersionControl::rollSettingsForward(v10_version, v10_settings);

    const fifojson v10_group = v10_settings[ORNL::Constants::SettingFileStrings::kSettings].at(0);
    if (!ORNL::Testing::expect(
            v10_group.contains(arc_length_key) && !v10_group.contains("helical_path_start_angle") &&
                !v10_group.contains("helical_tool_start_angle_offset") &&
                ORNL::Testing::near(v10_group.at(arc_length_key).get<double>(),
                                    first_radius * (1.74532925 - (90.0 * ORNL::degree)()), kTolerance),
            "Did not roll v10 helical_path_start_angle to a first-radius arc length."))
        return EXIT_FAILURE;

    fifojson v11_settings;
    v11_settings[ORNL::Constants::SettingFileStrings::kHeader][ORNL::Constants::SettingFileStrings::kVersion] = 11.0;
    v11_settings[ORNL::Constants::SettingFileStrings::kSettings] =
        fifojson::array({fifojson::object({{"helical_start_angle_offset", -0.20943951},
                                           {inner_radius_key, inner_radius},
                                           {layer_height_key, layer_height}})});
    double v11_version = 11.0;
    ORNL::SettingsVersionControl::rollSettingsForward(v11_version, v11_settings);

    const fifojson v11_group = v11_settings[ORNL::Constants::SettingFileStrings::kSettings].at(0);
    if (!ORNL::Testing::expect(v11_group.contains(arc_length_key) &&
                                   !v11_group.contains("helical_start_angle_offset") &&
                                   ORNL::Testing::near(v11_group.at(arc_length_key).get<double>(),
                                                       first_radius * (-12.0 * ORNL::degree)(), kTolerance),
                               "Did not roll v11 helical_start_angle_offset to a first-radius arc length."))
        return EXIT_FAILURE;

    const std::string z_key = ORNL::PRS::Dimensions::kUseVariableForZ.toStdString();
    fifojson v12_settings;
    v12_settings[ORNL::Constants::SettingFileStrings::kHeader][ORNL::Constants::SettingFileStrings::kVersion] = 12.0;
    v12_settings[ORNL::Constants::SettingFileStrings::kSettings] = fifojson::array({fifojson::object({{z_key, true}})});
    double v12_version                                           = 12.0;
    ORNL::SettingsVersionControl::rollSettingsForward(v12_version, v12_settings);

    const fifojson v12_group = v12_settings[ORNL::Constants::SettingFileStrings::kSettings].at(0);
    if (!ORNL::Testing::expect(v12_group.contains(z_key) && v12_group.at(z_key).is_number_integer() &&
                                   v12_group.at(z_key).get<int>() == static_cast<int>(ORNL::VariableZ::kVar200),
                               "Did not roll boolean variable_for_z forward to enumeration index kVar200."))
        return EXIT_FAILURE;

    fifojson v13_settings;
    v13_settings[ORNL::Constants::SettingFileStrings::kHeader][ORNL::Constants::SettingFileStrings::kVersion] = 13.0;
    v13_settings[ORNL::Constants::SettingFileStrings::kSettings] =
        fifojson::array({fifojson::object({{"helical_tool_start_angle_offset", (-12.0 * ORNL::degree)()},
                                           {inner_radius_key, inner_radius},
                                           {layer_height_key, layer_height}})});
    double v13_version = 13.0;
    ORNL::SettingsVersionControl::rollSettingsForward(v13_version, v13_settings);
    const fifojson v13_group = v13_settings[ORNL::Constants::SettingFileStrings::kSettings].at(0);
    if (!ORNL::Testing::expect(v13_group.contains(arc_length_key) &&
                                   !v13_group.contains("helical_tool_start_angle_offset") &&
                                   ORNL::Testing::near(v13_group.at(arc_length_key).get<double>(),
                                                       first_radius * (-12.0 * ORNL::degree)(), kTolerance),
                               "Did not roll v13 helical tool angle offset to a first-radius arc length."))
        return EXIT_FAILURE;

    fifojson invalid_settings;
    invalid_settings[ORNL::Constants::SettingFileStrings::kHeader][ORNL::Constants::SettingFileStrings::kVersion] =
        13.0;
    invalid_settings[ORNL::Constants::SettingFileStrings::kSettings] =
        fifojson::array({fifojson::object({{"helical_tool_start_angle_offset", (5.0 * ORNL::degree)()}})});
    double invalid_version = 13.0;
    bool invalid_rejected  = false;
    try {
        ORNL::SettingsVersionControl::rollSettingsForward(invalid_version, invalid_settings);
    } catch (const std::invalid_argument&) { invalid_rejected = true; }
    if (!ORNL::Testing::expect(invalid_rejected && invalid_version == 13.0,
                               "Migrated a nonzero helical angle without reference geometry."))
        return EXIT_FAILURE;

    const double expected_version = 15.0;
    if (!ORNL::Testing::expect(v10_version == expected_version && v11_version == expected_version &&
                                   v12_version == expected_version && v13_version == expected_version,
                               "Did not roll setting files forward."))
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
