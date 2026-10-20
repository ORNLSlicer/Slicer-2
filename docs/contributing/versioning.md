# Versioning and release preparation

`version.json` is the authoritative product version for CMake, runtime diagnostics,
G-code/project headers, Nix packages, AppImages, and the Windows installer.
It contains exactly four **string** fields: `major`, `minor`, `patch`, `suffix`.
Components are canonical nonnegative integers without leading zeros. MAJOR must
never exceed `2`; ORNLSlicer's renamed product line starts at `2.0.0`.

The full version follows SemVer:

| `suffix` | Full version, given numeric components `2.2.0` |
| --- | --- |
| `""` | `2.2.0` |
| `"rc.1"` | `2.2.0-rc.1` |
| `"rc.1+build.5"` | `2.2.0-rc.1+build.5` |
| `"+build.5"` | `2.2.0+build.5` |

Identifiers contain ASCII letters, digits, or hyphens, separated by dots.
Numeric prerelease identifiers cannot have leading zeros; build identifiers can.
Build metadata does not affect precedence, so changing only metadata is not a
release bump. CMake's numeric `project(VERSION)` omits the suffix; its generated
runtime version and Nix use the same full string.
Release candidates must also form valid Git tags/branch names; for example,
suffixes ending in `.lock` are rejected before any write.

## Bump rules

Release preparation analyzes full Conventional Commit messages since the latest
eligible stable release tag. The highest applicable level wins:

| Commit | Automatic bump |
| --- | --- |
| `fix: ...`, `perf: ...` | PATCH |
| `feat: ...` | MINOR |
| `type!: ...`, `type(scope)!: ...`, or `BREAKING CHANGE:` / `BREAKING-CHANGE:` footer | MAJOR below `2`; explicit review at `2` |
| Other types or nonconventional messages | No automatic bump |

A PATCH increments PATCH. A MINOR increments MINOR and resets PATCH to zero.
A MAJOR increments MAJOR and resets MINOR/PATCH to zero. The command never
produces MAJOR `3`. At the ceiling, breaking changes require **both** `--bump minor`
and a nonempty `--compatibility-note` describing the impact and migration. The
note is included in the generated PR and commit. Maintainers must review it and
carry the compatibility information into release notes; a MINOR release is not
a promise of backward compatibility under this capped policy.

Explicit `--bump patch|minor|major` overrides automatic inference, including when
there are no release-relevant commits. A PATCH override cannot bypass breaking
change review. Below MAJOR `2`, breaking changes require a MAJOR bump.

Tags use `v<full-version>`. Eligible tags are ancestors of the current branch,
or release-branch tags whose tree exactly matches their shared ancestor with the
current branch. The latter supports the existing `develop` → release-branch merge
history. Unrelated releases and historical noncanonical tags such as
`v1.3.001+rel` are excluded. Baselines are ordered by SemVer precedence, not date
or lexical order. Equal-precedence tags with different metadata are ambiguous
and rejected. The selected tags must agree with their own `version.json`.

Prereleases use the stable baseline for bump selection and the latest eligible
version for monotonicity. For example, a feature after `v2.1.0` can produce
`2.2.0-rc.1`, then `2.2.0-rc.2`, then `2.2.0`. A feature added after a patch
prerelease escalates the target to the next MINOR. Each prerelease must be tagged
before preparing its successor. The checked-in version must match the latest
eligible tag; a merged version bump must be released/tagged before another bump.

## Local commands

Run from the repository root with full Git history and tags available:

```sh
python3 scripts/version.py validate
python3 scripts/version.py validate --base origin/develop
python3 scripts/version.py prepare --dry-run
python3 scripts/version.py prepare --suffix rc.1 --dry-run
python3 scripts/version.py prepare --bump minor --compatibility-note 'Describe migration here' --dry-run
python3 scripts/version.py prepare --write
python3 tests/version_tests.py
```

`prepare` defaults to a dry run and prints a JSON plan. Only `--write` changes
`version.json`; it refuses staged or unstaged edits to that file and does not
stage, commit, or modify unrelated files. Errors print to stderr and return a
nonzero exit status. Missing eligible stable tags, mismatched versions, invalid
suffixes, ambiguous tags, and non-increasing releases require correction rather
than an automatic fallback. Fetch missing tags/history before retrying.

Validation allows an unchanged version during ordinary development. A changed
version must increase in SemVer precedence and either increment one numeric
component by one (resetting lower components) or advance/promote a prerelease
with the same numeric components. Metadata-only changes, decreases, skipped
components, invalid resets, malformed JSON, and MAJOR above `2` are rejected.

The tests require Python 3.9+, Git, CMake 3.25+, and Nix (`nix-instantiate`). They
exercise real temporary Git repositories and compare the Python, CMake and pure
Nix readers without compiling the application or downloading its dependencies.

## GitHub workflow

1. Run **Prepare release** (`prepare-release.yml`) against `develop`. It defaults
   to a dry run; inspect the calculated version, baseline, and compatibility note.
2. Run again with `dry_run=false` to create `codex/release-<version>` and a PR into
   `develop`. Select an explicit bump/note when needed. The workflow changes only
   `version.json`, runs focused checks, and dispatches **Nix CI** on the release
   branch. It never force-pushes or replaces an existing branch. Review an existing
   PR instead of repeating preparation; close/delete an abandoned branch before
   recreating it.
3. Finalize [`CHANGELOG.md`](../../CHANGELOG.md) on the release PR's branch using
   the selected version, release date, highlights, and compatibility notes; follow
   [Maintaining the Changelog](changelog.md). This is a manual step after the
   workflow's version-only commit. Review and merge the PR after CI passes.
   Create `v<full-version>` on the release
   commit or release-branch merge containing it, then publish the GitHub release.
   Tag validation runs before Nix builds/cache publication. Copy compatibility
   notes and the finalized changelog section into the release notes.

The workflow files must be present on the repository's default branch before
GitHub permits dispatch. Repository Actions settings must allow GitHub Actions
to create pull requests. The workflow uses the repository `GITHUB_TOKEN` with
contents, pull-request, and Actions write permissions; it needs no additional
credential. Because bot-token pushes/PR creation do not start ordinary workflows,
it explicitly uses `workflow_dispatch` to start CI, which GitHub permits with
that token ([GitHub documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)).

If PR creation or CI dispatch fails, the run fails visibly and preserves the
already-pushed release branch. Open the PR or dispatch `ci.yml` on that branch
manually after correcting permissions. A rerun refuses to overwrite it.
CI validates product versions on pushes, PRs into `develop`, and published
releases. The release tag must equal `v<version.json>` and exceed previous
eligible release tags; it is excluded when finding its predecessor.
