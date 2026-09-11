#include <QCoreApplication>
#include <QEventLoop>
#include <QFileInfo>
#include <QStandardPaths>
#include <QStringList>
#include <QTemporaryDir>
#include <QTimer>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

#include <zip/zip.h>

#include "geometry/mesh/closed_mesh.h"
#include "geometry/mesh/mesh_factory.h"
#include "managers/session_manager.h"
#include "managers/settings/settings_manager.h"
#include "threading/session_loader.h"
#include "utilities/constants.h"

namespace {
bool expect(bool condition, const char* message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

bool writeZipEntry(zip_t* zip, const std::string& name, const std::string& text) {
    if (zip_entry_open(zip, name.c_str()) < 0) return false;

    int result = zip_entry_write(zip, text.c_str(), text.size());
    zip_entry_close(zip);
    return result >= 0;
}

bool writeProjectWithOldGlobal(const QString& path) {
    QByteArray path_bytes = path.toUtf8();
    zip_t* zip            = zip_open(path_bytes.constData(), ZIP_DEFAULT_COMPRESSION_LEVEL, 'w');
    if (zip == nullptr) return false;

    fifojson global = fifojson::object();
    global[ORNL::Constants::SettingFileStrings::kHeader][ORNL::Constants::SettingFileStrings::kVersion] = 2.0;
    global[ORNL::Constants::SettingFileStrings::kSettings] =
        fifojson::array({fifojson::object({{"slicer_type", 0}, {"slicing_vector_x", 0.25}})});

    fifojson session                                    = fifojson::object();
    session[ORNL::Constants::Settings::Session::kParts] = fifojson::object();

    bool success = writeZipEntry(zip, ORNL::Constants::Settings::Session::Files::kGlobal, global.dump(4)) &&
                   writeZipEntry(zip, ORNL::Constants::Settings::Session::Files::kSession, session.dump(4)) &&
                   writeZipEntry(zip, ORNL::Constants::Settings::Session::Files::kLocal, fifojson::array().dump(4));

    zip_close(zip);
    return success;
}

std::optional<fifojson> readGlobalFromProject(const QString& path) {
    QByteArray path_bytes = path.toUtf8();
    zip_t* zip            = zip_open(path_bytes.constData(), ZIP_DEFAULT_COMPRESSION_LEVEL, 'r');
    if (zip == nullptr) return std::nullopt;

    void* buffer       = nullptr;
    size_t buffer_size = 0;
    if (zip_entry_open(zip, ORNL::Constants::Settings::Session::Files::kGlobal.c_str()) < 0 ||
        zip_entry_read(zip, &buffer, &buffer_size) < 0) {
        zip_close(zip);
        return std::nullopt;
    }

    zip_entry_close(zip);
    std::string text(static_cast<char*>(buffer), buffer_size);
    free(buffer);
    zip_close(zip);

    return fifojson::parse(text);
}
}  // namespace

int main(int argc, char* argv[]) {
    QCoreApplication app(argc, argv);
    QStandardPaths::setTestModeEnabled(true);

    QTemporaryDir temp_dir;
    if (!expect(temp_dir.isValid(), "Could not create temporary directory.")) return EXIT_FAILURE;

    auto session = ORNL::CSM;
    session->clearRecentFiles();

    const QString project_one     = temp_dir.path() + "/alpha.s2p";
    const QString project_two     = temp_dir.path() + "/beta.S2P";
    const QString project_ignored = temp_dir.path() + "/notes.txt";

    session->addRecentProjectFile(project_one);
    session->addRecentProjectFile(project_two);
    session->addRecentProjectFile(project_one);
    session->addRecentProjectFile(project_ignored);

    QStringList recent_projects = session->getRecentProjectFiles();
    if (!expect(recent_projects.size() == 2, "Recent project history should deduplicate and filter by extension."))
        return EXIT_FAILURE;
    if (!expect(recent_projects[0] == QFileInfo(project_one).absoluteFilePath(),
                "Most recently reopened project should move to the front."))
        return EXIT_FAILURE;
    if (!expect(recent_projects[1] == QFileInfo(project_two).absoluteFilePath(),
                "Previous project should remain after a duplicate is moved."))
        return EXIT_FAILURE;

    for (int i = 0; i < 12; ++i) { session->addRecentModelFile(temp_dir.path() + QString("/model-%1.stl").arg(i)); }
    session->addRecentModelFile(temp_dir.path() + "/fixture.s2p");

    QStringList recent_models = session->getRecentModelFiles();
    if (!expect(recent_models.size() == 10, "Recent model history should be capped at ten files.")) return EXIT_FAILURE;
    if (!expect(recent_models[0] == QFileInfo(temp_dir.path() + "/model-11.stl").absoluteFilePath(),
                "Newest model should be first in recent model history."))
        return EXIT_FAILURE;
    if (!expect(recent_models.back() == QFileInfo(temp_dir.path() + "/model-2.stl").absoluteFilePath(),
                "Recent model history should drop entries past the cap."))
        return EXIT_FAILURE;

    const QString step_model = temp_dir.path() + "/bracket.step";
    session->addRecentModelFile(step_model);
    recent_models = session->getRecentModelFiles();
    if (!expect(recent_models[0] == QFileInfo(step_model).absoluteFilePath(),
                "STEP model should be accepted in recent model history."))
        return EXIT_FAILURE;

    session->removeRecentFile(step_model);
    if (!expect(!session->getRecentModelFiles().contains(QFileInfo(step_model).absoluteFilePath()),
                "Removed recent model should not remain in history."))
        return EXIT_FAILURE;

    session->clearRecentFiles();
    if (!expect(session->getRecentProjectFiles().isEmpty() && session->getRecentModelFiles().isEmpty(),
                "Clearing recent files should clear both project and model history."))
        return EXIT_FAILURE;

    const QString clipping_model = temp_dir.path() + "/clipping.stl";
    const QString settings_model = temp_dir.path() + "/settings.stl";
    const QString build_model    = temp_dir.path() + "/build.stl";
    session->addRecentModelFile(clipping_model, ORNL::MeshType::kClipping);
    session->addRecentModelFile(settings_model, ORNL::MeshType::kSettings);
    session->addRecentModelFile(build_model, ORNL::MeshType::kBuild);

    recent_models = session->getRecentModelFiles();
    if (!expect(recent_models.size() == 1, "Only build models should be added to recent model history."))
        return EXIT_FAILURE;
    if (!expect(recent_models[0] == QFileInfo(build_model).absoluteFilePath(),
                "Build model should be retained after non-build model history entries are ignored."))
        return EXIT_FAILURE;

    QString project_path = temp_dir.path() + "/old-settings-project.s2p";
    if (!expect(writeProjectWithOldGlobal(project_path), "Could not write project fixture.")) return EXIT_FAILURE;

    ORNL::SessionLoader* loader = session->loadSession(false, project_path, false);
    if (!expect(loader != nullptr, "CLI-style session load rejected the old project before auto-updating settings."))
        return EXIT_FAILURE;

    QEventLoop loop;
    bool finished  = false;
    bool succeeded = false;
    QObject::connect(loader, &ORNL::SessionLoader::loadSucceeded, &loop, [&succeeded]() { succeeded = true; });
    QObject::connect(loader, &ORNL::SessionLoader::finished, &loop, [&finished, &loop]() {
        finished = true;
        loop.quit();
    });
    QTimer::singleShot(5000, &loop, [&loop]() { loop.quit(); });
    loop.exec();

    if (!expect(finished, "Timed out waiting for CLI-style session load to finish.")) return EXIT_FAILURE;
    if (!expect(succeeded, "CLI-style session load did not complete successfully.")) return EXIT_FAILURE;

    std::optional<fifojson> archived_global = readGlobalFromProject(project_path);
    if (!expect(archived_global.has_value(), "Could not read archived global settings from project."))
        return EXIT_FAILURE;

    double archived_version =
        (*archived_global)[ORNL::Constants::SettingFileStrings::kHeader][ORNL::Constants::SettingFileStrings::kVersion];
    if (!expect(archived_version == 2.0, "CLI-style session load unexpectedly modified the project archive."))
        return EXIT_FAILURE;

    double loaded_slice_normal_x =
        ORNL::GSM->getGlobal()->setting<double>(ORNL::Constants::ProfileSettings::Slicing::kSlicePlaneNormalX);
    if (!expect(loaded_slice_normal_x == 0.25,
                "CLI-style session load did not use migrated global settings in memory."))
        return EXIT_FAILURE;

    // Test part renaming and project saving/loading
    session->clearParts();
    auto mesh1 = QSharedPointer<ORNL::ClosedMesh>::create(
        ORNL::MeshFactory::CreateBoxMesh(ORNL::Distance(10), ORNL::Distance(10), ORNL::Distance(10)));
    mesh1->setName("part_alpha");
    session->addPart(mesh1);

    auto mesh2 = QSharedPointer<ORNL::ClosedMesh>::create(
        ORNL::MeshFactory::CreateBoxMesh(ORNL::Distance(10), ORNL::Distance(10), ORNL::Distance(10)));
    mesh2->setName("part_beta");
    session->addPart(mesh2);

    if (!expect(session->parts().size() == 2, "Session should have exactly two parts.")) return EXIT_FAILURE;

    QSharedPointer<ORNL::Part> part1 = session->getPart("part_alpha");
    QSharedPointer<ORNL::Part> part2 = session->getPart("part_beta");
    if (!expect(!part1.isNull() && !part2.isNull(), "Added parts should be accessible by name.")) return EXIT_FAILURE;

    // Availability checks
    if (!expect(!session->isPartNameAvailable("part_beta", part1), "part_beta should not be available for part1."))
        return EXIT_FAILURE;
    if (!expect(session->isPartNameAvailable("part_alpha", part1), "Current name should be reported available for self."))
        return EXIT_FAILURE;
    if (!expect(session->isPartNameAvailable("part_renamed", part1), "Unused name should be available."))
        return EXIT_FAILURE;

    // Reject rename to duplicate name
    if (!expect(!session->renamePart(part1, "part_beta"), "Renaming to existing part name must fail."))
        return EXIT_FAILURE;
    if (!expect(part1->name() == "part_alpha", "Part name should remain unchanged after failed rename."))
        return EXIT_FAILURE;

    // Successful rename
    if (!expect(session->renamePart(part1, "part_renamed"), "Renaming to available name should succeed."))
        return EXIT_FAILURE;
    if (!expect(part1->name() == "part_renamed", "Part internal name should be updated.")) return EXIT_FAILURE;
    if (!expect(!part1->rootMesh().isNull() && part1->rootMesh()->name() == "part_renamed",
                "Root mesh name should be updated."))
        return EXIT_FAILURE;
    if (!expect(session->getPart("part_alpha").isNull(), "Old name lookup should return null.")) return EXIT_FAILURE;
    if (!expect(session->getPart("part_renamed") == part1, "New name lookup should return the renamed part."))
        return EXIT_FAILURE;
    if (!expect(session->parts().contains("part_renamed"), "parts() map should contain new name.")) return EXIT_FAILURE;
    if (!expect(!session->parts().contains("part_alpha"), "parts() map should not contain old name."))
        return EXIT_FAILURE;

    // Save session with renamed part and reload to verify persistence
    QString renamed_project_path = temp_dir.path() + "/renamed-part-project.s2p";
    ORNL::SessionLoader* save_loader = session->saveSession(renamed_project_path, false);
    if (!expect(save_loader != nullptr, "Save session loader could not be created.")) return EXIT_FAILURE;

    QEventLoop save_loop;
    bool save_finished  = false;
    bool save_succeeded = false;
    QObject::connect(save_loader, &ORNL::SessionLoader::saveSucceeded, &save_loop,
                     [&save_succeeded]() { save_succeeded = true; });
    QObject::connect(save_loader, &ORNL::SessionLoader::finished, &save_loop, [&save_finished, &save_loop]() {
        save_finished = true;
        save_loop.quit();
    });
    QTimer::singleShot(5000, &save_loop, [&save_loop]() { save_loop.quit(); });
    save_loop.exec();

    if (!expect(save_finished, "Timed out waiting for session save to finish.")) return EXIT_FAILURE;
    if (!expect(save_succeeded, "Session save did not complete successfully.")) return EXIT_FAILURE;

    // Reload project
    ORNL::SessionLoader* load_loader = session->loadSession(true, renamed_project_path, false);
    if (!expect(load_loader != nullptr, "Load session loader could not be created.")) return EXIT_FAILURE;

    QEventLoop load_loop;
    bool load_finished  = false;
    bool load_succeeded = false;
    QObject::connect(load_loader, &ORNL::SessionLoader::loadSucceeded, &load_loop,
                     [&load_succeeded]() { load_succeeded = true; });
    QObject::connect(load_loader, &ORNL::SessionLoader::finished, &load_loop, [&load_finished, &load_loop]() {
        load_finished = true;
        load_loop.quit();
    });
    QTimer::singleShot(5000, &load_loop, [&load_loop]() { load_loop.quit(); });
    load_loop.exec();

    if (!expect(load_finished, "Timed out waiting for reloaded session to finish.")) return EXIT_FAILURE;
    if (!expect(load_succeeded, "Session load did not complete successfully.")) return EXIT_FAILURE;

    if (!expect(!session->getPart("part_renamed").isNull(), "Renamed part should exist after project reload."))
        return EXIT_FAILURE;
    if (!expect(!session->getPart("part_beta").isNull(), "Unchanged part should exist after project reload."))
        return EXIT_FAILURE;
    if (!expect(session->getPart("part_alpha").isNull(), "Old part name should not exist after project reload."))
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
