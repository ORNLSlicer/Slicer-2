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

## Release branches

`develop` is the integration branch and remains the repository's default branch.
`main` contains released code. Prepare each version through a temporary
`codex/release-<version>` PR into `develop`, then promote `develop` into `main`
with a merge commit. Do not squash or rebase the promotion: version-baseline
discovery relies on the shared history and matching release tree. Do not add
release-only edits while merging into `main`; make those changes through
`develop` first. Historical releases on `master` remain unchanged.

A separate stabilization branch is only needed when new development must
continue during release testing. Merge its fixes and finalized version/changelog
back into `develop` before promotion. For routine releases, hold new feature
merges between finalizing the release PR and promoting `develop`.

## Repository setup

The workflow files must be present on the repository's default branch before
GitHub permits dispatch. In **Settings > Actions > General > Workflow
permissions**, enable **Allow GitHub Actions to create and approve pull
requests** and save. This setting permits release PR creation; the workflow does
not approve or merge PRs. The default token permissions may remain read-only:
the preparation and publication jobs request their required write permissions
explicitly. If the checkbox is unavailable or disabled by policy, ask an
organization/enterprise administrator to enable it. See
[GitHub's repository Actions settings](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository).

Enable **Allow merge commits** in **Settings > General > Pull Requests**.
In **Settings > Rules > Rulesets**, edit the active rules targeting `main`:
disable **Require linear history** and allow merge commits in the pull-request
rule's allowed merge methods. A rebase-only rule still blocks promotion even
when the repository permits merge commits. Organization-owned rules require an
organization administrator to make the corresponding change. If a ruleset
covers both `main` and `develop`, split the branch-specific rules to retain
`develop`'s existing merge policy while allowing merge commits on `main`.

Keep PR review requirements and require successful version validation for
promotion PRs: select the actual job check after a **Version validation** PR
run, rather than the workflow display name. **Nix CI** builds branch pushes
and the final `main` commit.
Keep `CACHIX_AUTH_TOKEN` configured as described in
[Cachix Workflow](cachix.md).

## GitHub workflow

1. Run **Prepare release** (`prepare-release.yml`) against `develop`. It defaults
   to a dry run; inspect the calculated version, baseline, and compatibility note.
   Review the changelog and source for breaking changes even when inference says
   `breaking=false`: unmarked commit messages cannot identify them. At MAJOR 2,
   select `bump=minor` and provide a compatibility note for such changes too.
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
4. Open a promotion PR with base `main` and head `develop`. Wait for version
   validation and the Nix CI run for the reviewed `develop` commit to pass, then
   use **Create a merge commit**. Confirm Nix CI passes on the resulting `main`
   commit and that its tree matches the reviewed release contents.
5. Create `v<full-version>` on that exact `main` commit, then publish the GitHub
   release using that tag. Do not rely on the release form's default target,
   which is `develop`. Copy compatibility notes and the finalized changelog
   section into the release notes. Creating a tag alone does not publish
   downloads; the `release: published` event starts publication.
6. Confirm every job in the release-triggered **Nix CI** run passes. Version/tag
   validation runs before builds and cache publication. After both platform
   builds, Cachix pins, and Linux/macOS flake checks succeed, the **publish** job
   downloads that run's artifacts and attaches three files to the release:
   the Linux `.appimage`, Windows installer `.exe`, and Windows portable `.zip`.
   The ZIP contains the complete `ornlslicer/` tree; extract it before running
   `ornlslicer/bin/ornlslicer.exe`. Linux users must mark the downloaded AppImage
   executable with `chmod +x <filename>.appimage`. Verify all three release
   downloads and the [four Cachix pins](cachix.md#release-retention).

CI validates product versions on pushes, PRs into `develop` or `main`, and
published releases. The release tag must equal `v<version.json>` and exceed
previous eligible release tags; it is excluded when finding its predecessor.
The publication job alone receives GitHub contents-write permission; platform
builds retain read-only GitHub tokens. Preparation explicitly dispatches Nix CI
because pushes made with `GITHUB_TOKEN` do not trigger push workflows
([GitHub documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)).

## Recovering a failed run

If preparation reports `GitHub Actions is not permitted to create or approve
pull requests`, enable the repository setting above. When preparation has already
pushed `codex/release-<version>`, preserve it and open the PR manually:

```sh
RELEASE_VERSION='<full-version>'
gh pr create --base develop --head "codex/release-${RELEASE_VERSION}" \
  --title "chore(release): prepare ${RELEASE_VERSION}" --body-file release-body.md
gh workflow run ci.yml --ref "codex/release-${RELEASE_VERSION}"
```

Replace `<full-version>` with the calculated version and write `release-body.md`
with the dry-run plan and reviewed compatibility note first. If the PR already
exists but CI dispatch failed, run only the second command. Re-running
preparation refuses to overwrite an existing branch; it does not resume it.

If a release build, pin, or upload fails, fix the reported cause and rerun the
failed jobs in the original release-triggered run. A manual `ci.yml` dispatch
only builds artifacts and does not publish assets or pins. Running release jobs
are isolated from branch/tag-push concurrency and are not canceled by newer runs.
Uploads use `gh release upload --clobber` to replace matching asset names on a
rerun. Windows download names include the workflow run number, which is stable
across rerun attempts. Do not move a published tag to fix a release; publish a
new version when the released source must change.
