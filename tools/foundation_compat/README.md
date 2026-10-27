# Foundation backends compatibility

Checks that each foundation model adapter of `skforecast/foundation` still works with its backend library (`chronos-forecasting`, `timesfm`, `uni2ts`, `tabicl`, `tabpfn-time-series`, `tfc-t0`, `synthefy-nori`, `tsicl`).

Those libraries release often and change their API without notice, and the unit tests of the adapters use fakes of them, so they stay green when a backend changes. This check installs the real backend of each adapter in its own environment and runs two levels of tests against it.

| Path | Purpose |
|:-----|:--------|
| `adapters.toml` | Single source: one entry per environment, with the adapter, the backend package, the model and the secrets. |
| `run.py` | Creates the environment of one adapter with uv, runs the tests and writes `results/<adapter>.json`. |
| `summarize.py` | Turns the JSON results into a Markdown table. |
| `tests/contracts.py` | Symbols and arguments of the backend that each adapter uses. |
| `tests/test_contract.py` | Level 1: the installed backend still exposes them. No weights, seconds. |
| `tests/test_smoke.py` | Level 2: forecasts with the real weights through the public API. Minutes. |

The tests are outside the package on purpose: they are not in `testpaths`, so the unit test suite never collects them.

## Run it locally

Only Python >=3.11 and `uv` are needed. The script creates and removes its own environment.

```bash
python tools/foundation_compat/run.py --adapter t0
python tools/foundation_compat/run.py --adapter moirai --level contract
python tools/foundation_compat/summarize.py results
```

On macOS the result may differ from CI (Linux): `tfc-t0` and `timesfm` ship an MLX runtime with other signatures. CI is the reference.

## Environment

The backend is installed from `backend_package`, as a user does today: skforecast does not pin its backends, so there is no set of pinned versions to check. The installed version is compared with the latest one on PyPI: if the resolver held the backend back, the environment fails ("held back"), because it would be testing an old version.

skforecast is installed from the checkout with its core dependencies only, in the same resolution as the backend.

## Workflow

`.github/workflows/foundation-backends.yml` runs one job per adapter, which fails when its environment fails, and a `Summary` job that writes the table in the summary of the run. The last run that passed shows the last version of each backend known to work.

| Trigger | Code checked | Notes |
|:--------|:-------------|:---------|
| Schedule (Mondays) | The highest release branch (`X.Y.x`), where the development happens | A failure opens or updates the issue labeled `backend-compat`; the first run in which every environment passes closes it. |
| Pull request to `main` (the release one) that touches `skforecast/foundation/**` or this folder | The branch of the pull request | A failing check does not prevent the merge unless it is marked as required. |
| Manual (`workflow_dispatch`) | Any branch | |

Pull requests to the release branches (`*.x`) do not run the workflow: their changes are checked by the next scheduled run. Neither do the pull requests of Dependabot. To check a change of an adapter right away, run `run.py` locally or launch the workflow by hand on the branch (Actions > Foundation Backends Compatibility > Run workflow), which is possible once the workflow is on `main`.

Secrets of the repository: `HF_TOKEN` and `TABPFN_TOKEN`. They are exposed to third-party packages that are not pinned, so use read-only tokens of a dedicated account. Without `TABPFN_TOKEN` (for example in pull requests from forks), the smoke tests of TabPFN are reported as skipped.

## Maintenance

- **New adapter**: add its entry to `adapters.toml`, its calls to `tests/contracts.py` and its name to the matrix of the workflow. The unit test `skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py` fails until the first and the last are done, and the contract test until the second is.
- **An adapter changes how it calls its backend**: update `tests/contracts.py`.
- **A backend breaks**: fix the adapter, or restrict the version in its `backend_package` attribute (and in `adapters.toml`) until it is fixed.
