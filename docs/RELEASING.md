# Releasing pyroid

Owner-only instructions. Everything here needs repository-admin or PyPI-owner rights; no automation does it
for you. Do the one-time setup (sections 1–4) once, before the first 1.0 push to `main`.

How releases work: `.github/workflows/release.yml` runs [release-please](https://github.com/googleapis/release-please)
on every push to `main`. release-please keeps a release PR open (`chore: release X.Y.Z`) that bumps the version in
`Cargo.toml`/`Cargo.lock` and updates `CHANGELOG.md` from Conventional Commits. Merging that PR tags `vX.Y.Z`,
creates the GitHub Release, and the same workflow run then builds 24 wheels and the sdist, tests every wheel in
a clean venv, attests build provenance, and waits for your approval in the `pypi` environment before publishing
with Trusted Publishing (no API tokens; PEP 740 attestations are uploaded with the files).

## 1. GitHub repository settings

1. **Settings → Actions → General**
   - Workflow permissions: **Read repository contents and packages permissions** (the workflow asks for
     more per job).
   - Tick **Allow GitHub Actions to create and approve pull requests**. release-please needs this to open
     the release PR with the default `GITHUB_TOKEN`.
2. **Settings → Secrets and variables → Actions:** delete the old `PYPI_API_TOKEN` secret (0.7.x used it), and
   revoke that token on PyPI (Account settings → API tokens). Nothing in the new workflow uses a token.
3. **Labels** used by the issue forms: create `parity` (the `bug` and `enhancement` labels exist by default).
4. **Discussions** (optional): the issue chooser links "Ask a question" to Discussions. Either enable
   Discussions (Settings → General → Features) or remove that link from `.github/ISSUE_TEMPLATE/config.yml`.

## 2. Environments

**Settings → Environments → New environment**, twice:

| Environment | Required reviewers | Deployment branches and tags | Used by |
|---|---|---|---|
| `pypi` | you | **Selected branches and tags:** `main` only | job `publish to PyPI` |
| `testpypi` | you | **Selected branches and tags:** `main` (add a branch pattern if you want dry runs from branches) | job `publish to TestPyPI` |

For both, leave **Allow administrators to bypass configured protection rules** unticked, and don't add any
secrets. "Prevent self-review" must stay off if you are the only reviewer.

## 3. Trusted publishers

The publisher must match the workflow exactly: owner `ao`, repository `pyroid`, workflow `release.yml`, and the
environment name.

- **PyPI** — `pyroid` already exists there (0.7.x), so this is a normal publisher, not a pending one:
  <https://pypi.org/manage/project/pyroid/settings/publishing/> → **Add a new publisher → GitHub Actions**:
  owner `ao`, repository `pyroid`, workflow name `release.yml`, environment name `pypi`.
- **TestPyPI** — `pyroid` doesn't exist on TestPyPI yet (checked 2026-11-05), so register a **pending
  publisher**: <https://test.pypi.org/manage/account/publishing/> → **Add a new pending publisher**: PyPI project
  name `pyroid`, owner `ao`, repository `pyroid`, workflow name `release.yml`, environment name `testpypi`. The
  first successful upload creates the project. (A TestPyPI account is separate from your PyPI account.)

Docs: [adding a publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/),
[pending publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## 4. Private vulnerability reporting

**Settings → Advanced Security** (called "Code security" on some accounts) → **Private vulnerability reporting
→ Enable.** `SECURITY.md` and the issue chooser send reporters to
<https://github.com/ao/pyroid/security/advisories/new>, which only works once this is on.
[GitHub docs](https://docs.github.com/en/code-security/security-advisories/working-with-repository-security-advisories/configuring-private-vulnerability-reporting-for-a-repository).

Also confirm the Code of Conduct contact (`CODE_OF_CONDUCT.md`, "Reporting an Issue"): it currently names
support@ataiva.com, the address in the package metadata. Change it if conduct reports should go elsewhere.

## 5. First release dry run (TestPyPI)

After `next` is merged into `main` and pushed (PLAN Phase 4, step 1):

1. **Actions → Release → Run workflow**, branch `main`, tick **testpypi**.
2. Wait for the build, test and `check distributions` jobs. The run summary lists the 25 files.
3. Approve the `testpypi` deployment when the run asks for it.
4. Check the result from a clean venv (versions come from `Cargo.toml`; `1.0.0-dev` is `1.0.0.dev0` on
   TestPyPI):

   ```bash
   uv venv /tmp/pyroid-tp && uv pip install --python /tmp/pyroid-tp --index-url https://test.pypi.org/simple/ pyroid==1.0.0.dev0
   /tmp/pyroid-tp/bin/python -c "import pyroid; print(pyroid.__version__)"
   ```

   On <https://test.pypi.org/project/pyroid/#files>, each file should show a "Verified" provenance badge
   (PEP 740).
5. TestPyPI never accepts the same filename twice, even after you delete a release. To dry-run again, change the
   pre-release version in `Cargo.toml` first (e.g. `1.0.0-dev.1`, then `cargo update --workspace`) in a
   regular PR.

## 6. Releasing 1.0.0

1. On the first push to `main`, release-please reads the commits since the `v0.7.0` tag and opens
   **`chore(main): release 1.0.0`**. `release-please-config.json` pins `"release-as": "1.0.0"`, so the first
   release is 1.0.0 however the history is merged. The breaking `chore!:` commit would also give 1.0.0, but a
   squash merge could drop its `!`.
2. Review the PR: the root `Cargo.toml` (`[workspace.package] version`) and the `pyroid-core`/`pyroid-py`
   entries in `Cargo.lock` say `1.0.0`, `.release-please-manifest.json` says `1.0.0`, and `CHANGELOG.md` has the
   generated 1.0.0 section. The hand-written `## 1.0.0 (unreleased)` stub ends up just below the generated
   section. Delete the stub in a normal commit on `main` (release-please refreshes the PR), or straight after
   the release.
3. CI doesn't run on the release PR itself, because PRs opened with `GITHUB_TOKEN` don't trigger workflows.
   That's acceptable: the PR only changes versions and the changelog, and the same build and test matrix runs
   again after the merge, before anything is published. To get checks on the PR anyway, close and reopen it.
4. Merge the release PR. The `Release` run on `main` tags `v1.0.0`, creates the GitHub Release, builds and tests
   everything, attests provenance, and then waits at **publish to PyPI**. Approve the `pypi` deployment.
5. Afterwards:
   - **Remove `"release-as": "1.0.0"`** from `release-please-config.json` (commit `chore: stop pinning
     release-as`). It is sticky: while it is there, every later release PR proposes 1.0.0 again (verified
     offline with release-please 17.6.0).
   - Check <https://pypi.org/project/pyroid/1.0.0/#files>: 25 files, each with provenance.
   - Optionally verify a file's GitHub attestation:
     `gh attestation verify pyroid-1.0.0-cp311-abi3-*.whl --repo ao/pyroid`.

## 7. Later releases

Merge PRs with Conventional Commit titles (see CONTRIBUTING.md). `feat` gives a minor bump, `fix`/`perf` a
patch, and `!`/`BREAKING CHANGE` a major. Merge the release PR when you want to ship, then approve `pypi`.

- **Pre-releases:** add a `Release-As: 1.1.0-rc.1` footer to a commit on `main` (the wheels become `1.1.0rc1`),
  and use `Release-As: 1.1.0` for the final release.
- **How the version is bumped:** `release-please-config.json` uses `release-type: simple` with TOML
  `extra-files`. The `rust` release type can't handle a virtual workspace whose crates use
  `version.workspace = true`: it throws, and no release PR appears. The extra files update
  `[workspace.package].version` in `Cargo.toml` and the `pyroid-core` and `pyroid-py` entries in `Cargo.lock`.
  If you rename a crate or add a workspace crate, add it to the `Cargo.lock` filter, or `--locked` builds fail
  on the release commit. The log line `file version.txt did not exist` is expected and harmless.
- **A job failed after the tag was created:** if it was flaky, use **Re-run failed jobs** on the same run (it
  keeps the release-please outputs, so it can still publish). If the code is broken, fix it on `main` with a
  `fix:` commit and ship the next patch release; don't move or reuse the tag.
- **Never** publish from a laptop, and don't add an API token as a fallback.
