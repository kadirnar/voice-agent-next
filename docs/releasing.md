# Releasing

How voice-agent-next versions, changelogs and PyPI releases work, and the one-time setup
the repository owner does before the first release. Releases are **tag-driven**: pushing a
`v<version>` tag runs [`.github/workflows/release.yml`](../.github/workflows/release.yml),
which builds, verifies and publishes. Nothing is published from a branch, a pull request or
a manual run. Pull requests that touch `pyproject.toml`, `cliff.toml` or the workflow run
its build and verification jobs (without macOS unless labelled `ci:full`).

## Versioning

* **One source of truth: `project.version` in `pyproject.toml`.** Bump it with
  [`uv version`](https://docs.astral.sh/uv/reference/cli/#uv-version), which also updates
  `uv.lock`. At runtime, `voice_agent_next.__version__` (and `van version`) read the installed
  package metadata, so they always match the wheel.
* **[Semantic Versioning](https://semver.org/), 0.x for now**: while the major version is 0,
  a minor bump (`0.2.0`) may break APIs and a patch bump (`0.1.1`) must not. Pre-releases
  use PEP 440 suffixes: `0.2.0a1`, `0.2.0b1`, `0.2.0rc1`.
* **`main` carries a development version** (`0.1.0.dev0`, then `0.2.0.dev0` after the
  0.1.0 release). The release workflow refuses to publish a `.dev` version.
* **The tag must equal the version**: tag `v0.1.0` for `version = "0.1.0"`. The workflow
  fails before building otherwise.

??? note "Why a static version and not hatch-vcs (versions from git tags)"
    A version derived from tags needs the full git history everywhere a package is built
    (`fetch-depth: 0` in every CI job, no shallow clones, a `PKG-INFO` fallback for sdists)
    and changes the editable install's version on every commit, which `uv sync` does not
    notice. A static version is visible in review, works in any checkout, and `uv version
    --bump` makes bumping one command. The workflow's tag check gives the same guarantee
    as hatch-vcs: the published version is the tagged one.

## Changelog

[`CHANGELOG.md`](changelog.md) is
generated from the [Conventional Commits](https://www.conventionalcommits.org/) history by
[git-cliff](https://git-cliff.org/) with the repository's `cliff.toml`. Squash-merged pull
request titles are the commits, so **the PR title is the changelog entry**: write it for
users (`feat(stt): Deepgram Nova-3 streaming STT`).

| Commit type | Section |
| --- | --- |
| `feat` | Features |
| `fix` | Bug fixes |
| `perf` | Performance |
| `refactor` | Refactoring |
| `docs` | Documentation (roadmap / progress notes are skipped) |
| `test` | Tests |
| `build`, `ci` | Build and CI |
| `chore(deps…)` / other `chore` | Dependencies / Miscellaneous |
| `chore(release)`, `style` | not listed |

Entries are sorted by scope inside each section; `(#25)` references become links, and a
`!` after the type (`feat(session)!:`) or a `BREAKING CHANGE:` footer marks an entry
**breaking**. Commits that are not Conventional Commits are left out.

```bash
uvx git-cliff@2.14.2 --unreleased            # preview what the next release contains
uvx git-cliff@2.14.2 --tag v0.2.0 -o CHANGELOG.md   # write it as the 0.2.0 section
```

The generated file is a starting point: edit wording, merge related entries or add a
highlights paragraph in the release pull request before tagging. Later regenerations
rewrite the file, so keep hand edits for the release commit.

## One-time setup (repository owner)

These steps need owner rights on GitHub and a PyPI account with 2FA; nothing in the
repository can do them.

### 1. PyPI trusted publisher

Trusted publishing lets the workflow upload with a short-lived OIDC token instead of an API
token stored in GitHub secrets. The project does not exist on PyPI yet, so register a
**pending** publisher (it creates the project on the first upload):

1. Sign in to <https://pypi.org>, go to **Your account → Publishing**.
2. Under **Add a new pending publisher**, choose **GitHub** and enter:

    | Field | Value |
    | --- | --- |
    | PyPI Project Name | `voice-agent-next` |
    | Owner | `kadirnar` |
    | Repository name | `voice-agent-next` |
    | Workflow name | `release.yml` |
    | Environment name | `pypi` |

3. Optional, for dry runs: do the same on <https://test.pypi.org> (a separate account) with
   environment name **`testpypi`**.

After the first release the publisher shows up under the project's **Settings →
Publishing**; nothing else changes. If the workflow file or environment is ever renamed,
update the publisher too, or uploads fail with `invalid-publisher`.

### 2. GitHub environments

In the repository, **Settings → Environments**:

* **`pypi`**: add yourself as a **required reviewer** (every upload waits for your
  approval), and under **Deployment branches and tags** choose *Selected branches and
  tags* with the tag rule `v*`.
* **`testpypi`** (optional): no reviewer needed; allow the `main` branch.

### 3. Protect release tags

**Settings → Rules → Rulesets → New tag ruleset**: target `v*`, restrict creation,
update and deletion to maintainers (bypass list: yourself). A pushed `v*` tag is what starts
a release.

### 4. Before the first public release

* The repository is private: the README links on PyPI point at GitHub and the
  documentation site, which only work once they are public.
* The PyPI upload also stores [PEP 740](https://peps.python.org/pep-0740/) attestations
  (Sigstore), which record the repository and workflow name in a public transparency log.

## Making a release

1. **Check `main`**: CI green, including macOS (it runs on every push to `main`).
2. **Release pull request** from a branch such as `release/v0.2.0`:

    ```bash
    uv version 0.2.0                  # or: uv version --bump minor
    uvx git-cliff@2.14.2 --tag v0.2.0 -o CHANGELOG.md
    # review and edit CHANGELOG.md, then:
    uv run pytest -q tests/test_packaging.py
    ```

    Title it `chore(release): v0.2.0` (release commits are left out of the changelog) and
    merge it. If other pull requests land first, regenerate the changelog before merging.

3. **Dry run** (recommended): **Actions → Release → Run workflow** on `main`. It builds the
   sdist and wheel, runs `twine check --strict`, checks their contents, installs the wheel
   in clean virtual environments on Linux (Python 3.11 and 3.14), macOS and Windows, runs
   `import voice_agent_next`, `van --help`, `van version` and `van providers`, installs the
   sdist, and puts the release notes in the run summary. Tick **testpypi** to also upload
   to TestPyPI (a version can only be uploaded once there too).
4. **Tag the merge commit and push the tag**:

    ```bash
    git switch main && git pull
    git tag -a v0.2.0 -m "v0.2.0"
    git push origin v0.2.0
    ```

5. **Approve the `pypi` deployment** when the run asks for it. The workflow then uploads to
   PyPI and creates the GitHub Release `v0.2.0` with the changelog section as notes and the
   sdist and wheel attached; versions with `a`, `b` or `rc` are marked pre-release.
6. **Open the next development cycle** with a pull request `chore(release): start
   0.3.0.dev0` (`uv version 0.3.0.dev0`).

### When something fails

* **Before the upload** (tag/version mismatch, missing changelog section, a verification
  job): fix it on `main`, delete the tag (`git push origin :refs/tags/v0.2.0`,
  `git tag -d v0.2.0`) and tag the fixed commit.
* **The upload failed** (e.g. publisher misconfigured): fix the setting and *re-run failed
  jobs*; the build artifacts are reused.
* **The GitHub Release step failed after the upload**: re-run that job.
* **A broken version reached PyPI**: files on PyPI are immutable and a version number can
  never be reused. Yank it (project **Settings → Releases → Yank**) and release a patch
  version. Never move a tag that has been published.

## Checking a build locally

```bash
uv build --no-sources                          # dist/*.tar.gz and dist/*.whl
uvx twine==7.0.0 check --strict dist/*
uv run pytest -q tests/test_packaging.py       # metadata and contents, built in-process
```

The sdist contains `src/`, `pyproject.toml`, `README.md`, `CHANGELOG.md` and `LICENSE`, and
the wheel only the `voice_agent_next` package (with `py.typed` and the benchmark smoke
data). Tests, docs, examples and benchmark suites stay in the repository. On PyPI, relative
README links are rewritten to absolute GitHub links at build time
([hatch-fancy-pypi-readme](https://github.com/hynek/hatch-fancy-pypi-readme), configured in
`pyproject.toml`).
