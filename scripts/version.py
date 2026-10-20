#!/usr/bin/env python3
"""Validate the authoritative version and prepare reviewable release bumps."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys


NUMBER = r"(?:0|[1-9][0-9]*)"
IDENTIFIERS = r"[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*"
SEMVER = re.compile(
    rf"({NUMBER})\.({NUMBER})\.({NUMBER})(?:-({IDENTIFIERS}))?(?:\+({IDENTIFIERS}))?"
)
HEADER = re.compile(r"([a-z]+)(?:\([^\r\n]+\))?(!)?: .+")


@dataclass(frozen=True)
class Version:
    major: int
    minor: int
    patch: int
    prerelease: str = ""
    build: str = ""

    @classmethod
    def parse(cls, value: str) -> Version:
        match = SEMVER.fullmatch(value)
        if not match:
            raise ValueError(f"invalid SemVer: {value!r}")
        major, minor, patch, prerelease, build = match.groups()
        if int(major) > 2:
            raise ValueError("MAJOR must not exceed 2")
        for identifier in (prerelease or "").split("."):
            if identifier.isdigit() and len(identifier) > 1 and identifier[0] == "0":
                raise ValueError("numeric prerelease identifiers must not have leading zeros")
        return cls(int(major), int(minor), int(patch), prerelease or "", build or "")

    @property
    def core(self) -> tuple[int, int, int]:
        return self.major, self.minor, self.patch

    @property
    def precedence(self) -> tuple:
        identifiers = tuple(
            (0, int(part)) if part.isdigit() else (1, part)
            for part in self.prerelease.split(".")
        ) if self.prerelease else ()
        return self.core, not self.prerelease, identifiers

    def __str__(self) -> str:
        return (
            ".".join(map(str, self.core))
            + (f"-{self.prerelease}" if self.prerelease else "")
            + (f"+{self.build}" if self.build else "")
        )

    def document(self) -> dict[str, str]:
        return dict(zip(("major", "minor", "patch"), map(str, self.core))) | {
            "suffix": self.prerelease + (f"+{self.build}" if self.build else "")
        }


def unique_keys(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate version key: {key}")
        result[key] = value
    return result


def read_version(content: str) -> Version:
    data = json.loads(content, object_pairs_hook=unique_keys)
    if not isinstance(data, dict) or set(data) != {"major", "minor", "patch", "suffix"}:
        raise ValueError("version.json must contain exactly major, minor, patch, suffix")
    if any(not isinstance(value, str) for value in data.values()):
        raise ValueError("all version.json values must be strings")
    suffix = data["suffix"]
    value = ".".join(data[key] for key in ("major", "minor", "patch"))
    value += ("" if suffix.startswith("+") else "-") + suffix if suffix else ""
    return Version.parse(value)


def bump_core(version: Version, level: str) -> Version:
    major, minor, patch = version.core
    if level == "major":
        return Version.parse(f"{major + 1}.0.0")
    if level == "minor":
        return Version(major, minor + 1, 0)
    return Version(major, minor, patch + 1)


def validate_transition(previous: Version, current: Version) -> None:
    if previous == current:
        return
    if current.precedence <= previous.precedence:
        raise ValueError("version must increase; build metadata does not change precedence")
    if current.core == previous.core:
        return  # Increasing prerelease or promotion to stable.
    allowed = {bump_core(previous, level).core for level in ("patch", "minor")}
    if previous.major < 2:
        allowed.add(bump_core(previous, "major").core)
    if current.core not in allowed:
        raise ValueError("invalid transition: increment one component and reset lower components")


def classify(messages: list[str]) -> tuple[str | None, bool]:
    level = None
    breaking = False
    for message in messages:
        header = HEADER.fullmatch(message.splitlines()[0]) if message.splitlines() else None
        breaking |= bool(header and header[2]) or bool(
            re.search(r"(?m)^BREAKING[ -]CHANGE:[ \t]+\S", message)
        )
        if header:
            if header[1] == "feat":
                level = "minor"
            elif header[1] in {"fix", "perf"} and level is None:
                level = "patch"
    return level, breaking


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], text=True, capture_output=True)
    if result.returncode:
        raise ValueError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def version_at(ref: str) -> Version:
    sha = git("rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")
    return read_version(git("show", f"{sha}:version.json"))


def release_tags(exclude: str = "") -> list[tuple[Version, str]]:
    tags = []
    for tag in git("tag", "--list", "v*").splitlines():
        if tag == exclude:
            continue
        try:
            version = Version.parse(tag[1:])
        except ValueError:
            continue  # Historical pre-SemVer tags are not release baselines.
        # Releases historically tag a merge on the release branch. Its tree must
        # exactly match the shared develop ancestor if the tag is not an ancestor.
        try:
            sha = git("rev-parse", f"refs/tags/{tag}^{{commit}}")
            ancestor = git("merge-base", "HEAD", sha)
        except ValueError:
            continue
        if git("rev-parse", f"{sha}^{{tree}}") != git("rev-parse", f"{ancestor}^{{tree}}"):
            continue
        tags.append((version, tag))
    tags.sort(key=lambda item: item[0].precedence)
    for left, right in zip(tags, tags[1:]):
        if left[0].precedence == right[0].precedence:
            raise ValueError(f"ambiguous release tags: {left[1]} and {right[1]}")
    return tags


def prepare(current: Version, bump: str, suffix: str, compatibility_note: str) -> dict:
    tags = release_tags()
    if not tags:
        raise ValueError("no eligible canonical release tag found")
    stable_tags = [(version, tag) for version, tag in tags if not version.prerelease]
    if not stable_tags:
        raise ValueError("no eligible stable release tag found")
    baseline, base_tag = stable_tags[-1]
    latest, latest_tag = tags[-1]
    for version, tag in {(baseline, base_tag), (latest, latest_tag)}:
        if version_at(f"refs/tags/{tag}") != version:
            raise ValueError(f"{tag} disagrees with its version.json")
    if current != latest:
        raise ValueError(f"version.json must match latest eligible tag {latest_tag}; tag the pending release first")
    base_sha = git("rev-parse", f"refs/tags/{base_tag}^{{commit}}")
    messages = git("log", "--format=%B%x00", f"{base_sha}..HEAD").split("\0")
    inferred, breaking = classify([message.strip() for message in messages if message.strip()])
    if breaking and baseline.major == 2:
        if bump != "minor" or not compatibility_note.strip():
            raise ValueError("breaking changes at MAJOR 2 require --bump minor and --compatibility-note")
    level = ("major" if breaking else inferred) if bump == "auto" else bump
    if level is None:
        raise ValueError("no release-relevant commits; choose an explicit --bump if a release is needed")
    if breaking and baseline.major < 2 and level != "major":
        raise ValueError("breaking changes require a major bump below MAJOR 2")
    candidate = bump_core(baseline, level)
    candidate = read_version(json.dumps(candidate.document() | {"suffix": suffix}))
    try:
        git("check-ref-format", f"refs/tags/v{candidate}")
        git("check-ref-format", f"refs/heads/release/{candidate}")
    except ValueError as exc:
        raise ValueError(f"version {candidate} cannot form a valid Git release tag/branch (for example, .lock is forbidden)") from exc
    validate_transition(baseline, candidate)
    validate_transition(current, candidate)
    if candidate.precedence <= current.precedence:
        raise ValueError("prepared release must be newer than the latest release")
    return {
        "base": base_tag,
        "previous": str(current),
        "version": str(candidate),
        "bump": level,
        "breaking": breaking,
        "compatibility_note": compatibility_note.strip(),
        "document": candidate.document(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate schema, optional transition and release tag")
    validate.add_argument("--base", help="Git ref whose version.json is the previous version")
    validate.add_argument("--release-tag", help="require this tag to equal v<version.json>")
    release = commands.add_parser("prepare", help="calculate a release from eligible stable tags")
    release.add_argument("--bump", choices=("auto", "patch", "minor", "major"), default="auto")
    release.add_argument("--suffix", default="", help="prerelease and/or +build metadata")
    release.add_argument("--compatibility-note", default="")
    mode = release.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="update only version.json")
    mode.add_argument("--dry-run", action="store_true", help="default: print plan without writing")
    args = parser.parse_args()
    try:
        root = Path(git("rev-parse", "--show-toplevel"))
        path = root / "version.json"
        current = read_version(path.read_text())
        if args.command == "validate":
            if args.base:
                validate_transition(version_at(args.base), current)
            if args.release_tag is not None and args.release_tag != f"v{current}":
                raise ValueError(f"release tag must equal v{current}")
            if args.release_tag:
                previous_tags = release_tags(exclude=args.release_tag)
                if previous_tags:
                    previous = previous_tags[-1][0]
                    validate_transition(previous, current)
                    if current.precedence <= previous.precedence:
                        raise ValueError("release must be newer than previous eligible release tags")
            print(current)
        else:
            plan = prepare(current, args.bump, args.suffix, args.compatibility_note)
            if args.write:
                if git("status", "--porcelain", "--", ":(top)version.json"):
                    raise ValueError("refusing to overwrite staged or unstaged version.json edits")
                path.write_text(json.dumps(plan["document"], indent=4) + "\n")
            print(json.dumps(plan, indent=2))
    except (ValueError, OSError) as exc:
        print(f"version: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
