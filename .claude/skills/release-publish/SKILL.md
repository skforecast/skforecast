---
name: release-publish
description: Publishes a skforecast release to PyPI through the GitHub release and the publish-pypi.yml workflow (Trusted Publishing), after a rehearsal on TestPyPI. Use when the release branch X.Y.x is ready to be released.
disable-model-invocation: true
argument-hint: "<version to release, e.g. 0.27.0>"
---

# Publish a release

Version: $ARGUMENTS (ask for it if empty). Release branch: `X.Y.x` (e.g. `0.27.0` → `0.27.x`).

Nothing is uploaded by hand with `twine`: `.github/workflows/publish-pypi.yml` builds, checks and uploads with Trusted Publishing when the GitHub release is published, and waits for a maintainer to approve the deployment. Approving a deployment, pushing a tag and publishing the release are the user's: propose each command and wait for the go-ahead; never approve a deployment yourself.

## 1. Preconditions

- `version` in `pyproject.toml` and `__version__` in `skforecast/__init__.py` are $ARGUMENTS on `origin/X.Y.x` (the workflow fails otherwise).
- The section of the version in `docs/releases/releases.md` went through `/release-note consolidate` and its heading has the release date, not `In development`.
- The regression test of the datasets ran on this version (`SKFORECAST_TEST_DATASETS=1 pytest skforecast/datasets/tests/tests_datasets/test_fetch_dataset_all_datasets.py -q`); it does not run in CI.
- The version is not on PyPI yet: `curl -s -o /dev/null -w '%{http_code}' https://pypi.org/pypi/skforecast/$ARGUMENTS/json` answers `404`.

## 2. Pull request X.Y.x to main

Open it (or find it) and wait for every check: unit tests, the tests with the minimum versions of the dependencies and the coverage. Do not merge it yet. Write down the SHA of its head (`gh pr view <n> --json headRefOid`); the next steps use that exact commit, not the name of the branch, so that a later push to `X.Y.x` cannot slip into the release.

## 3. Rehearsal on TestPyPI

A rehearsal starts in either of two ways:

- A tag `test-publish-*` on any commit (works from any branch, and is the way to rehearse the exact SHA of the release):

  ```bash
  git tag test-publish-<version>-<n> <sha>        # n = 1, 2... one tag per rehearsal
  git push origin test-publish-<version>-<n>
  ```

- A manual run: Actions → Publish to PyPI or TestPyPI → Run workflow, choosing the branch (or `gh workflow run publish-pypi.yml --ref <branch>`). GitHub only offers it once `publish-pypi.yml` is on the default branch, `main`.

The run, titled "Rehearsal on TestPyPI (<ref>)", uploads `<version>.dev<run number>` to TestPyPI and never touches PyPI: only a published GitHub release goes there. Approve the deployment `testpypi` (the run → Review deployments) and wait for the job "Install from TestPyPI". If it fails, fix it on `X.Y.x` through a pull request and rehearse again on the new head. Then delete the tag:

```bash
git push origin --delete test-publish-<version>-<n>
git tag -d test-publish-<version>-<n>
```

## 4. GitHub release

On the same commit that passed the rehearsal and the pull request checks.

The workflow only checks the tag, the prerelease flag and the branch once the release is published, and by then Zenodo has archived it with a DOI. Run the same checks first, in a checkout of `<sha>` (the script only reads the files for `pypi`; it needs `packaging`):

```bash
git fetch origin X.Y.x && git switch --detach <sha>
python tools/release/prepare_version.py pypi v<version> false   # true only for a prerelease
gh api repos/skforecast/skforecast/compare/X.Y.x...<sha> --jq .status   # identical or behind
```

Then create and publish the release:

```bash
gh release create v<version> --target <sha> --title "v<version>" \
  --notes "See https://skforecast.org/<version>/releases/releases for more information on this new release."
```

Publishing it starts the workflow and Zenodo archives the release (a new DOI). The workflow checks that the tag is `v` plus the version of both files, that the release is marked as a prerelease only for a prerelease version (`0.27.0rc1`) and that the commit is on `X.Y.x`. The run is titled "Publish v<version> to PyPI". Follow it with `gh run watch`, approve the deployment `pypi` and wait for "Install from PyPI", which installs the version from PyPI and runs a forecast.

## 5. After the upload

- Merge the pull request `X.Y.x` → `main`.
- conda-forge: the bot opens a pull request in `conda-forge/skforecast-feedstock` from the sdist on PyPI. It updates the version and the sha256, not the dependencies: compare `requirements/run` with `dependencies` in `pyproject.toml` (and `python_min` with `requires-python`) before merging it.
- Check that Zenodo lists the new version.

## If something fails

- Before the approval of `pypi`: reject the deployment, delete the release and its tag (`gh release delete v<version> --cleanup-tag`), fix on `X.Y.x` and start again from step 2. Zenodo has already archived the published release: review the new version there.
- After the upload: PyPI never accepts the same file name twice, even after deleting it. If only part of the upload went through, "Re-run failed jobs" finishes it (`skip-existing` keeps what is there and the last job checks that the index serves the files built). "Re-run all jobs" does not work there: it runs the build again, which fails because the version is already on the index (the same holds for a rehearsal). A broken release is fixed with a new patch version; yank the broken one on PyPI if users should not install it.
