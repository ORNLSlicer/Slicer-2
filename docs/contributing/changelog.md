# Maintaining the changelog

[`CHANGELOG.md`](../../CHANGELOG.md) at the repository root records notable changes
for users, developers, and downstream integrations. It loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and
[The Good Docs Project changelog template](https://www.thegooddocsproject.dev/template/changelog).
The same file is bundled in the application under **Help > Changelog**; rebuilding
updates the bundled copy, which remains available offline.

## Pull requests

Update the changelog in any PR introducing discoverable behavior, features, fixes,
or integration changes. Add entries under `## Unreleased`, without a date or release
highlights. Keep entries focused on outcomes and explain compatibility or migration
steps when needed. Start each bullet with a verb, such as `Added`, `Changed`,
`Fixed`, or `Removed`, and link to the relevant issue or PR when available.

Use only relevant sections, in this order: `Added`, `Changed`, `Deprecated`,
`Removed`, `Fixed`, `Security`, `Breaking changes`. Omit empty sections and
placeholder bullets such as `None.`. An empty `## Unreleased` heading immediately
after a release is expected. Released entries remain newest first, separated by
`---`.

Do not bump `version.json` in ordinary feature or fix PRs or assign a tentative
version to Unreleased. Follow the existing [versioning policy](versioning.md):
release preparation selects the version from Conventional Commits, MAJOR is
capped at 2, and breaking changes at the ceiling require an explicit MINOR bump
and compatibility note. Changelog prose does not determine the automatic bump.

## Preparing release notes

The **Prepare release** workflow initially changes only `version.json`. Finalize
the changelog manually on that release PR's branch before merging it:

1. Use the full version selected by release preparation, including any prerelease
   or build suffix, and the planned release date. Rename `## Unreleased` to
   `## <version> - YYYY-MM-DD` (for example, `## 2.2.0 - 2026-10-20`). The heading
   omits the tag's leading `v` and must match `version.json`.
2. Add `### Release highlights` directly below that heading with 2–5 bullets
   summarizing the key changes. Review all entries against the release contents;
   carry the release PR's compatibility note into `### Breaking changes` with
   concrete upgrade steps.
3. Insert a new empty `## Unreleased` above the release, with `---` separating it
   from the released section. Omit unused category headings. Check that older
   releases remain in reverse chronological order and are separated by `---`.
4. Commit the changelog on the release branch and review it with the version bump.
   If the planned release date changes, correct it before tagging. The application
   built from that commit will contain the finalized history.
5. After merge, tag and publish as described in [Versioning and Releases](versioning.md).
   Copy the released section's highlights and categorized entries into the GitHub
   release notes. These steps are manual; the workflow does not generate or publish
   changelog entries or release notes.
