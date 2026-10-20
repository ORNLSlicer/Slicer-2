"""Dependency-free version/release tests, including real Git, CMake and Nix readers."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/version.py"
spec = importlib.util.spec_from_file_location("release_version", SCRIPT)
version = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = version
sys.dont_write_bytecode = True
spec.loader.exec_module(version)
V = version.Version.parse


class VersionTests(unittest.TestCase):
    def test_schema_and_roundtrip(self):
        for value in ("0.0.0", "1.9.3", "2.1.0", "2.2.0-rc.1+build.05", "2.2.0+build.05"):
            with self.subTest(value=value):
                parsed = V(value)
                self.assertEqual(str(parsed), value)
                self.assertEqual(version.read_version(json.dumps(parsed.document())), parsed)

    def test_invalid_schema(self):
        valid = V("2.1.0").document()
        for data in ([], {}, valid | {"extra": ""}, valid | {"major": 2},
                     valid | {"major": "3"}, valid | {"minor": "01"},
                     valid | {"patch": "-1"}, valid | {"suffix": "rc.01"},
                     valid | {"suffix": "rc..1"}, valid | {"suffix": "+"},
                     valid | {"suffix": "rc.1+"}, valid | {"suffix": "réc.1"},
                     valid | {"suffix": "rc;1"}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                version.read_version(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            version.read_version('{"major":"2","major":"1","minor":"0","patch":"0","suffix":""}')
        with self.assertRaises(ValueError):
            version.read_version("{")

    def test_semver_precedence(self):
        ordered = ["2.2.0-alpha", "2.2.0-alpha.1", "2.2.0-alpha.beta", "2.2.0-beta",
                   "2.2.0-beta.2", "2.2.0-beta.11", "2.2.0-rc.9", "2.2.0-rc.10", "2.2.0"]
        for previous, current in zip(ordered, ordered[1:]):
            self.assertLess(V(previous).precedence, V(current).precedence)
            version.validate_transition(V(previous), V(current))
        self.assertEqual(V("2.2.0+a").precedence, V("2.2.0+b").precedence)

    def test_transitions(self):
        for previous, current in (("2.1.0", "2.1.0"), ("2.1.0", "2.1.1"),
                                  ("2.1.9", "2.2.0-rc.1"), ("1.9.8", "2.0.0")):
            version.validate_transition(V(previous), V(current))
        for previous, current in (("2.1.0", "2.0.9"), ("2.1.0", "2.1.0+build.1"),
                                  ("2.1.0", "2.1.0-rc.1"), ("2.1.0", "2.1.2"),
                                  ("2.1.0", "2.3.0"), ("2.1.0", "2.2.1")):
            with self.subTest(previous=previous, current=current), self.assertRaises(ValueError):
                version.validate_transition(V(previous), V(current))
        with self.assertRaisesRegex(ValueError, "MAJOR"):
            version.bump_core(V("2.1.0"), "major")

    def test_commit_selection(self):
        self.assertEqual(version.classify(["docs: guide", "chore: release"]), (None, False))
        self.assertEqual(version.classify(["perf(mesh): speed up", "fix: bug"]), ("patch", False))
        self.assertEqual(version.classify(["feat(ui): panel", "fix: bug"]), ("minor", False))
        self.assertEqual(version.classify(["refactor(api)!: remove call"]), (None, True))
        self.assertEqual(version.classify(["fix: change\n\nBREAKING CHANGE: old API removed"]), ("patch", True))
        self.assertEqual(version.classify(["fix: change\n\nBREAKING-CHANGE: old API removed"]), ("patch", True))
        self.assertEqual(version.classify(["not conventional"]), (None, False))


class GitReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.run_git("init", "-q")
        self.run_git("config", "user.name", "Version Test")
        self.run_git("config", "user.email", "version@example.invalid")
        self.write_version("2.1.0")
        self.commit("chore: baseline")
        self.run_git("tag", "v2.1.0")

    def run_git(self, *args):
        return subprocess.run(["git", *args], cwd=self.repo, capture_output=True, text=True, check=True).stdout.strip()

    def write_version(self, value):
        (self.repo / "version.json").write_text(json.dumps(V(value).document(), indent=4) + "\n")

    def commit(self, message):
        self.run_git("add", ".")
        self.run_git("commit", "-qm", message, "--allow-empty")

    def cli(self, *args, ok=True):
        result = subprocess.run([sys.executable, str(SCRIPT), *args], cwd=self.repo, capture_output=True, text=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_auto_and_overrides(self):
        self.commit("fix: bug")
        self.assertEqual(json.loads(self.cli("prepare").stdout)["version"], "2.1.1")
        self.commit("feat: feature")
        self.assertEqual(json.loads(self.cli("prepare").stdout)["version"], "2.2.0")
        self.assertEqual(json.loads(self.cli("prepare", "--bump", "patch").stdout)["version"], "2.1.1")
        self.assertIn("MAJOR", self.cli("prepare", "--bump", "major", ok=False).stderr)

    def test_breaking_override_requires_note(self):
        self.commit("refactor!: remove API")
        for options in ([], ["--bump", "minor"], ["--bump", "patch", "--compatibility-note", "migration"]):
            self.assertIn("breaking changes", self.cli("prepare", *options, ok=False).stderr)
        plan = json.loads(self.cli("prepare", "--bump", "minor", "--compatibility-note", "Use new API").stdout)
        self.assertEqual(plan["version"], "2.2.0")
        self.assertTrue(plan["breaking"])
        self.assertEqual(plan["compatibility_note"], "Use new API")

    def test_breaking_major_below_ceiling(self):
        self.run_git("tag", "-d", "v2.1.0")
        self.write_version("1.9.0")
        self.commit("chore: historical baseline")
        self.run_git("tag", "v1.9.0")
        self.commit("feat!: new API")
        self.assertEqual(json.loads(self.cli("prepare").stdout)["version"], "2.0.0")
        self.cli("prepare", "--bump", "patch", ok=False)
        self.cli("prepare", "--suffix", "rc.1", "--write")
        self.commit("chore: major prerelease")
        self.run_git("tag", "v2.0.0-rc.1")
        self.assertEqual(json.loads(self.cli("prepare").stdout)["version"], "2.0.0")

    def test_dry_run_write_and_dirty_guards(self):
        self.commit("fix: bug")
        original = (self.repo / "version.json").read_bytes()
        (self.repo / "unrelated.txt").write_text("keep me")
        self.cli("prepare", "--dry-run")
        self.assertEqual((self.repo / "version.json").read_bytes(), original)
        (self.repo / "version.json").write_bytes(original + b"\n")
        self.assertIn("refusing to overwrite", self.cli("prepare", "--write", ok=False).stderr)
        self.run_git("add", "version.json")
        self.assertIn("refusing to overwrite", self.cli("prepare", "--write", ok=False).stderr)
        self.run_git("reset", "-q", "HEAD", "--", "version.json")
        (self.repo / "version.json").write_bytes(original)
        self.cli("prepare", "--write")
        self.assertEqual((self.repo / "unrelated.txt").read_text(), "keep me")
        self.assertEqual(self.run_git("diff", "--name-only"), "version.json")
        self.assertEqual(self.run_git("diff", "--cached", "--name-only"), "")
        self.assertIn("tag the pending release", self.cli("prepare", "--write", ok=False).stderr)

    def test_prerelease_cycle_and_promotion(self):
        self.commit("feat: feature")
        for suffix in ("rc.1+build.5", "rc.2"):
            result = self.cli("prepare", "--suffix", suffix, "--write")
            target = json.loads(result.stdout)["version"]
            self.commit("chore(release): " + target)
            self.run_git("tag", "v" + target)
        self.assertEqual(json.loads(self.cli("prepare").stdout)["version"], "2.2.0")
        self.assertIn("version must increase", self.cli("prepare", "--suffix", "rc.1", ok=False).stderr)

    def test_feature_after_patch_prerelease(self):
        self.commit("fix: bug")
        self.cli("prepare", "--suffix", "rc.1", "--write")
        self.commit("chore: release")
        self.run_git("tag", "v2.1.1-rc.1")
        self.commit("feat: feature")
        self.assertEqual(json.loads(self.cli("prepare").stdout)["version"], "2.2.0")

    def test_missing_and_invalid_tags(self):
        self.run_git("tag", "-d", "v2.1.0")
        self.run_git("tag", "v1.3.001+rel")
        self.assertIn("no eligible canonical", self.cli("prepare", ok=False).stderr)
        self.run_git("tag", "v2.0.0")
        self.assertIn("disagrees", self.cli("prepare", ok=False).stderr)

    def test_ambiguous_metadata_tags(self):
        self.run_git("tag", "v2.1.0+other")
        self.assertIn("ambiguous", self.cli("prepare", ok=False).stderr)

    def test_release_branch_tag_with_identical_tree(self):
        branch = self.run_git("branch", "--show-current")
        self.run_git("tag", "-d", "v2.1.0")
        self.run_git("checkout", "-qb", "release")
        self.commit("chore: release branch commit")
        self.run_git("tag", "v2.1.0")
        self.run_git("checkout", "-q", branch)
        self.commit("fix: bug")
        self.assertEqual(json.loads(self.cli("prepare").stdout)["version"], "2.1.1")

    def test_tag_with_unreleased_changes_is_not_a_baseline(self):
        branch = self.run_git("branch", "--show-current")
        self.run_git("checkout", "-qb", "future")
        self.write_version("2.2.0")
        self.commit("feat: unreleased feature")
        self.run_git("tag", "v2.2.0")
        self.run_git("checkout", "-q", branch)
        self.commit("fix: bug")
        self.assertEqual(json.loads(self.cli("prepare").stdout)["version"], "2.1.1")

    def test_no_relevant_commits(self):
        self.commit("docs: guide")
        self.assertIn("no release-relevant", self.cli("prepare", ok=False).stderr)
        self.assertEqual(json.loads(self.cli("prepare", "--bump", "patch").stdout)["version"], "2.1.1")

    def test_git_ref_restrictions(self):
        self.commit("feat: feature")
        for suffix in ("rc.lock", "+build.lock"):
            self.assertIn("valid Git release tag", self.cli("prepare", "--suffix", suffix, "--write", ok=False).stderr)
        self.assertEqual(self.run_git("diff", "--name-only"), "")

    def test_validation_cli(self):
        self.cli("validate", "--base", "HEAD", "--release-tag", "v2.1.0")
        self.cli("validate", "--release-tag", "v2.1.1", ok=False)
        self.cli("validate", "--base", "missing", ok=False)
        self.write_version("2.3.0")
        self.cli("validate", "--base", "HEAD", ok=False)
        self.write_version("2.2.0")
        self.cli("validate", "--base", "HEAD")

    def test_release_validation_excludes_current_tag(self):
        self.commit("feat: feature")
        self.cli("prepare", "--suffix", "rc.1", "--write")
        self.commit("chore: prerelease")
        self.run_git("tag", "v2.2.0-rc.1")
        self.cli("validate", "--release-tag", "v2.2.0-rc.1")
        self.cli("prepare", "--write")
        self.commit("chore: stable release")
        self.run_git("tag", "v2.2.0")
        self.cli("validate", "--release-tag", "v2.2.0")
        self.write_version("2.4.0")
        self.commit("chore: invalid release")
        self.run_git("tag", "v2.4.0")
        self.assertIn("invalid transition", self.cli("validate", "--release-tag", "v2.4.0", ok=False).stderr)


class ConsumerTests(unittest.TestCase):
    def test_cmake_and_nix_agree(self):
        self.assertIsNotNone(shutil.which("cmake"), "CMake is required for reader tests")
        nix = shutil.which("nix-instantiate")
        self.assertIsNotNone(nix, "Nix is required for reader tests")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cmake").mkdir()
            shutil.copy(ROOT / "cmake/version.cmake", root / "cmake/version.cmake")
            driver = root / "check.cmake"
            driver.write_text(
                'include("${CMAKE_CURRENT_LIST_DIR}/cmake/version.cmake")\n'
                'FetchVersion()\nfile(WRITE "${CMAKE_CURRENT_LIST_DIR}/full" "${RESULT_VAR}")\n'
                'FetchCMakeFriendlyVersion()\nfile(WRITE "${CMAKE_CURRENT_LIST_DIR}/numeric" "${RESULT_VAR}")\n'
            )
            valid = V("2.1.0").document()
            fixtures = [V(value).document() for value in (
                "0.0.0", "2.1.0", "2.2.0-rc.1", "2.2.0-rc.1+build.05", "2.2.0+build.05"
            )] + [valid | change for change in (
                {"major": "3"}, {"major": 2}, {"minor": "01"}, {"patch": "-1"}, {"minor": ""},
                {"suffix": "rc.01"}, {"suffix": "rc..1"}, {"suffix": "+"},
                {"suffix": "rc.1+"}, {"suffix": "rc;1"}, {"suffix": "rc_1"},
                {"extra": ""}
            )] + [{}, []]
            for data in fixtures:
                with self.subTest(data=data):
                    path = root / "version.json"
                    path.write_text(json.dumps(data))
                    try:
                        expected = version.read_version(path.read_text())
                    except ValueError:
                        expected = None
                    cmake = subprocess.run(["cmake", "-P", str(driver)], capture_output=True, text=True)
                    expression = f"import {ROOT / 'nix/version.nix'} {path}"
                    nix_result = subprocess.run([nix, "--eval", "--strict", "--json", "--expr", expression],
                                                capture_output=True, text=True)
                    for result in (cmake, nix_result):
                        self.assertEqual(result.returncode == 0, expected is not None, result.stdout + result.stderr)
                    if expected:
                        self.assertEqual((root / "full").read_text(), str(expected))
                        self.assertEqual((root / "numeric").read_text(), ".".join(map(str, expected.core)))
                        self.assertEqual(json.loads(nix_result.stdout), str(expected))


if __name__ == "__main__":
    unittest.main()
