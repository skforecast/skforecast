# Add a `backend` argument to `save_forecaster` / `load_forecaster`

## Context

Today `save_forecaster` / `load_forecaster` ([skforecast/utils/utils.py:2391-2513](../skforecast/utils/utils.py#L2391-L2513))
only use **joblib**, and the file extension is hard-forced to `.joblib`. The
exploratory notebook [dev/export_forecaster_backends.ipynb](export_forecaster_backends.ipynb)
validated three additional serialization backends on a fitted forecaster:

- **pickle** — stdlib, no dependency (marginal gain over joblib, but harmless).
- **cloudpickle** — serializes custom `weight_func` and user-defined
  `window_features` classes *by value*, embedding them in the file. This removes
  the current workaround of exporting custom functions as separate `.py` files
  and the "save your custom classes manually" warning.
- **skops** — a *secure* format (no arbitrary code execution on load), ideal for
  sharing models. It cannot natively serialize pandas objects backed by a
  `DatetimeIndex`, so `last_window_` and `training_range_` must be **decomposed**
  before dump and **reconstructed** after load.

**This is a worthwhile feature.** The strongest motivators are cloudpickle
(self-contained custom functions) and skops (secure model sharing). pickle is
included for completeness as agreed. The default backend stays `joblib`, so
existing behavior is unchanged.

Decisions confirmed with the user:
- Support **all four** backends: `joblib` (default), `pickle`, `cloudpickle`, `skops`.
- For `backend='cloudpickle'`: **skip** the `.py`-file export and the custom-class
  warning (functions/classes are embedded in the file).
- skops decompose/reconstruct must be **non-destructive** and cover **all
  forecaster types** (single-series DataFrame/Series + Index, multi-series dicts,
  and `ForecasterStats`).

## Risks / fragility

- The skops path is the bulk of the work. The notebook prototype **mutates the
  live forecaster** (`forecaster.last_window_ = ...`) before dumping — that would
  corrupt the user's in-memory object. The real implementation must operate on a
  `copy.deepcopy` of the forecaster (or restore originals in a `finally`).
- `last_window_` / `training_range_` shapes differ by forecaster type — the
  helper must branch on `dict` vs `pd.DataFrame` vs `pd.Series` vs `pd.Index`.

## Changes

### 1. `save_forecaster` — [skforecast/utils/utils.py:2391](../skforecast/utils/utils.py#L2391)

- Add `backend: str = 'joblib'` parameter (document the four options in the
  NumPy-style docstring — read
  [.github/instructions/docstrings.instructions.md](../.github/instructions/docstrings.instructions.md) first).
- Validate `backend` against `{'joblib', 'pickle', 'cloudpickle', 'skops'}`;
  raise `ValueError` otherwise.
- Replace the hard `.with_suffix('.joblib')` with a backend→extension map:
  `joblib→.joblib`, `pickle→.pkl`, `cloudpickle→.cloudpickle`, `skops→.skops`.
- Gate optional backends behind `check_optional_dependency(...)`
  ([utils.py:2553](../skforecast/utils/utils.py#L2553)) for `cloudpickle` and `skops`.
- Dispatch the dump:
  - `joblib` → `joblib.dump` (unchanged).
  - `pickle` → `pickle.dump`.
  - `cloudpickle` → `cloudpickle.dump`; **skip** the `weight_func` `.py` export
    and the `window_features` custom-class warning blocks.
  - `skops` → deepcopy forecaster, call new `_skops_decompose(fc_copy)`, then
    `skops.io.dump(fc_copy, file_name)`. Never touch the original object.
- Keep the existing custom-function `.py` export + warnings for
  `joblib`/`pickle`/`skops` (unchanged behavior for those).

### 2. `load_forecaster` — [skforecast/utils/utils.py:2468](../skforecast/utils/utils.py#L2468)

- Add `backend: str | None = None`. When `None`, infer from the file suffix
  (`.joblib`→joblib, `.pkl`/`.pickle`→pickle, `.cloudpickle`→cloudpickle,
  `.skops`→skops); raise `ValueError` on an unrecognized extension.
- Gate `cloudpickle`/`skops` with `check_optional_dependency`.
- Dispatch the load:
  - `joblib`/`pickle`/`cloudpickle` → respective `.load`.
  - `skops` → `get_untrusted_types(file=...)` then `load(file=..., trusted=...)`,
    then `_skops_reconstruct(forecaster)` to rebuild `last_window_` /
    `training_range_`.
- Keep the existing skforecast-version mismatch warning and `summary()` call.

### 3. New helpers — `skforecast/utils/utils.py`

Add two module-private helpers next to the save/load functions:

- `_skops_decompose(forecaster)` — convert non-skops-serializable pandas
  attributes into plain dict/list form. Branch on type:
  - `last_window_` is `pd.DataFrame` (Recursive/Direct/DirectMultiVariate) →
    `to_dict(orient="split")` + store `index` as `[str(ts)...]` and
    `freq = index.freqstr`.
  - `last_window_` is `pd.Series` (`ForecasterStats`) → analogous Series form.
  - `last_window_` is `dict` (MultiSeries) → apply per value.
  - `training_range_` is `pd.Index` → `[str(ts)...]`; if `dict`, apply per value.
  Store a small marker (e.g. the original type) so reconstruct can invert
  precisely. Operates in place on the **deepcopy** passed in.
- `_skops_reconstruct(forecaster)` — inverse: rebuild DataFrame/Series with
  `pd.to_datetime(index)` + `.asfreq(freq)`, and `pd.DatetimeIndex(...)` for
  ranges, handling the dict (multi-series) case symmetrically. Tolerate
  non-datetime indexes (e.g. `RangeIndex`) gracefully.

### 4. Dependencies

- Add `cloudpickle` and `skops` to the `optional_dependencies` dict at
  [utils.py:49-62](../skforecast/utils/utils.py#L49-L62) under a new extra
  (e.g. `'serialization'`) **and** mirror them in
  [pyproject.toml](../pyproject.toml) `[project.optional-dependencies]` (plus
  `all`/`full`/`test`). A test enforces consistency between the two — keep them
  in sync.
- Import `pickle` at the top of utils.py; import `cloudpickle`/`skops` lazily
  inside the dispatch branches (after `check_optional_dependency`), following the
  try/except pattern used elsewhere.

### 5. Tests — [skforecast/utils/tests/tests_utils/test_save_load_forecaster.py](../skforecast/utils/tests/tests_utils/test_save_load_forecaster.py)

Read [.github/instructions/testing.instructions.md](../.github/instructions/testing.instructions.md) first. Add:
- Parametrized round-trip test over `backend in {joblib, pickle, cloudpickle, skops}`
  reusing the existing attribute-comparison logic (lines 59-88), for
  `ForecasterRecursive`, `ForecasterRecursiveMultiSeries`, and `ForecasterStats`
  (covers Series / dict / DataFrame `last_window_` shapes).
- skops: assert predictions match before-save vs after-load, and assert the
  **original in-memory forecaster is not mutated** by `save_forecaster`.
- `load_forecaster(backend=None)` extension auto-detection for each extension.
- cloudpickle backend: a forecaster with a custom `weight_func` round-trips
  without needing a `.py` file on disk, and **no** `.py` file / custom-class
  warning is emitted.
- `ValueError` on invalid `backend` and on unrecognized extension when
  `backend=None`.
- Skip skops/cloudpickle tests cleanly if the package is missing
  (`pytest.importorskip`).

### 6. Docs

Update [docs/user_guides/save-load-forecaster.ipynb](../docs/user_guides/save-load-forecaster.ipynb)
with a short section per backend and a note that skops is the secure option and
cloudpickle embeds custom functions. (Optional, can follow the code.)

## Verification

Using the `skforecast_22_py13` conda env (per project memory):

1. `pip install cloudpickle skops` in that env (confirm with user before installing).
2. Run the targeted tests:
   `pytest skforecast/utils/tests/tests_utils/test_save_load_forecaster.py -vv`
3. Manual smoke test mirroring the notebook: fit a `ForecasterRecursive`, then for
   each backend `save_forecaster(fc, "fc", backend=...)` → `load_forecaster("fc.<ext>")`
   → `predict(steps=1, exog=...)` and confirm identical predictions; confirm the
   original `fc.last_window_` is unchanged after the skops save.
4. Repeat the round-trip for `ForecasterRecursiveMultiSeries` (dict attributes)
   and `ForecasterStats` (Series `last_window_`) under skops.
5. Run the optional-dependency consistency test to confirm pyproject/dict sync:
   `pytest skforecast/utils/tests -k optional_dependency -vv`.
