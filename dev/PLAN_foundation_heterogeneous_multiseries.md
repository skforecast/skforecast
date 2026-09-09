# PLAN: Foundation forecasters with heterogeneous multi-series input


## 1. Problem statement

`ForecasterFoundation` / `FoundationModel` cannot predict or backtest a multi-series dataset
when the series differ from each other in one or more of these ways:

1. different lengths (including series that end before the end of the common span),
2. a different subset of exogenous columns per series,
3. a series that contains NaN in its target (interior or trailing).

Reproduction script: `dev/foundation-multiseries-multiexog.py`

The script loads `demo_multi_series` + `demo_multi_series_exog`, drops some exog columns
for two series, and calls `fit` / `predict(steps=5, exog=...)` and `backtesting_foundation`
for every covariate-aware foundation model.

### Dataset characteristics

| series  | full index               | train len | target NaNs (full) | exog columns                                             |
|---------|--------------------------|----------:|-------------------:|----------------------------------------------------------|
| id_1000 | 2016-01-01 .. 2016-12-31 |       213 |                  0 | sin_day_of_week, cos_day_of_week                         |
| id_1001 | 2016-07-02 .. 2016-12-31 |        30 |                  0 | sin_day_of_week, cos_day_of_week, air_temperature, wind_speed |
| id_1002 | 2016-01-01 .. 2016-07-01 |       183 |                  0 | sin_day_of_week, cos_day_of_week, air_temperature, wind_speed |
| id_1003 | 2016-01-01 .. 2016-12-31 |       213 |                146 | sin_day_of_week, air_temperature, wind_speed             |
| id_1004 | 2016-05-02 .. 2016-08-31 |        91 |                  0 | sin_day_of_week, cos_day_of_week, air_temperature, wind_speed |

`end_train = 2016-07-31`. The dataset exercises every axis at once: mixed lengths, two series
that end before the end of the span (id_1002, id_1004), mixed exog subsets (2, 3 or 4
columns), and one series (id_1003) with NaN blocks in the target, including trailing NaN
inside some backtesting folds.

### Observed errors

`predict(steps=5)`:

| model_id                          | error                                                                                                  |
|-----------------------------------|--------------------------------------------------------------------------------------------------------|
| autogluon/chronos-2-small         | `All past_covariates must have same keys. Expected [...2 cols...], got [...4 cols...] at index 1. Heterogeneous lists are not supported.` |
| google/timesfm-3.0-pytorch        | OK (verified in `skforecast_py14`, timesfm 3.0.1; see diagnosis)                                       |
| google/timesfm-2.5-200m-pytorch   | OK (ignores exog, `allow_exog=False`)                                                                   |
| Salesforce/moirai-2.0-R-small     | `uni2ts is required` (backend not installed; also ignores exog)                                         |
| soda-inria/tabicl                 | `boolean index did not match indexed array along axis 0; size of axis is 32 but ... is 30` (notebook run, not reproduced: `tabicl[forecast]` not installed) |
| priorlabs/tabpfn-ts               | OK (notebook run)                                                                                       |
| theforecastingcompany/t0-alpha    | OK (already pools columns with NaN; see diagnosis)                                                      |
| Synthefy/Nori                     | `Input y contains NaN.` (notebook run)                                                                  |
| taharnbl/TS-ICL                   | `Expected all samples in covars dicts to have identical covariate keys.` (notebook run)                 |

`backtesting_foundation(steps=24)` fails for every model, with or without exog:

| model_id                        | error                                                                                     |
|---------------------------------|-------------------------------------------------------------------------------------------|
| autogluon/chronos-2-small       | `KeyError: 'id_1002'` in `ChronosAdapter.predict` (`exog[name]`, key missing)             |
| google/timesfm-3.0-pytorch      | `all input arrays must have the same shape` in `predict_batch` (fold 1)                   |
| google/timesfm-2.5-200m-pytorch | `KeyError: 'id_1002'` in `_calculate_metrics_backtesting_multiseries` (no exog needed)    |

---

## 2. Diagnosis

Two independent problems overlap in the script. The first one is real heterogeneity that the
backends cannot batch. The second one is the backtesting path feeding the adapters inputs that
violate the invariants the `predict` path guarantees. The original version of this plan
attributed the TimesFM 3.0 failure to the first problem; experiments show it belongs to the
second.

### 2.1 Backend batch constraints (real heterogeneity)

- **A. Identical covariate KEYS across the batch.** Chronos (`ChronosAdapter.predict`) and
  TS-ICL (`TSICLAdapter.predict`) build one `inputs_list` of per-series dicts and pass the
  whole list to a single backend call. Chronos' `_validate_list_of_dicts` rejects any list
  whose elements carry different `past_covariates` / `future_covariates` keys, or that
  disagree on whether a future covariate is available. TS-ICL has the same rule. TimesFM 3.0
  (`_predict_v3`) already groups by column signature for the same reason.

- **B. NaN-filled column union is not neutral.** Padding every series to the union of columns
  with NaN (the pattern T0 uses) is not a valid workaround for Chronos. Measured with
  `chronos-2-small` on CPU, `steps=5`:

  | experiment                                                       | max abs diff |
  |------------------------------------------------------------------|-------------:|
  | id_1000 alone vs id_1000 plus 2 all-NaN columns                  |         45.1 |
  | id_1003 alone vs 5-series batch with NaN-filled column union     |        378.2 |
  | same-signature batch vs each series alone (`cross_learning=False`) |       0.003 |

  Values are around 1300, so a NaN column shifts the forecast by about 3 to 30 percent,
  while grouping by signature is numerically neutral. T0 is the exception: its backend
  defines NaN as "covariate absent" and `T0Adapter` already relies on it.

- **C. TabICL (unverified).** `tabicl.forecast` drops NaN-target rows itself
  (`_impute_missing_targets`) and intersects context/future columns (`_align_covariates`), so
  the notebook error may not come from the NaN-filled union frame. Reproduce with
  `tabicl[forecast]` installed before deciding whether grouping fixes it.

- **D. Nori rejects NaN in the target.** `NoriAdapter.predict` loops per series (no batching
  problem) and feeds `y_ctx = series.to_numpy()` to `NoriRegressor.fit`, which runs an
  sklearn-style NaN check. A backend-specific limitation, not a batching issue.

- **E. Not a failure.** Moirai and TimesFM 2.5 drop exog (`allow_exog=False`).

### 2.2 Backtesting path violates the adapter input invariants

`_backtesting_foundation` calls `FoundationModel.predict(..., check_inputs=False)`, which
skips `_prepare_future_exog` and passes the fold slices straight to the adapter. Instrumenting
the real loop (`context_length=100`, TimesFM 2.5 so nothing crashes) shows, in fold 1:

```
id_1003: ctx 2016-04-29..2016-08-06 n=100 | ctx_exog 2016-05-17..2016-08-24 n=100 | fut_exog 08-25..09-17 n=24
id_1004: ctx 2016-05-17..2016-08-24 n=100 | ctx_exog 2016-05-17..2016-08-24 n=100 | fut_exog 08-25..08-31 n=7
id_1002: ctx 2016-03-24..2016-07-01 n=100 | ctx_exog 2016-03-24..2016-07-01 n=100 | fut_exog None
-> pred id_1003: 2016-08-07..2016-08-30     (fold test window is 08-25..09-17)
-> pred id_1002: 2016-07-02..2016-07-25     (repeated in every fold)
```

Three distinct defects, none of them adapter-specific:

1. **Context trimmed, exog not.** `_extract_data_folds_multiseries` trims each series to its
   last valid value (built for ML forecasters, where the last window is separate) but slices
   exog by fold dates. `context` and `context_exog` end up with the same length and different
   dates. This is the TimesFM 3.0 crash: `past_future` arrays get widths `ctx_i + 24` with
   `ctx_i` taken from exog, not from the target, so `np.stack` fails. Padding inside the
   adapter (the previous plan's option A) would have hidden a misalignment instead of fixing it.
2. **Future exog not aligned to the horizon.** id_1004 arrives with 7 rows for 24 steps and
   id_1002 with no key at all. Chronos requires `len(future_covariates) == prediction_length`
   and consistent availability across the batch; `ChronosAdapter` and `TSICLAdapter` also
   index `exog[name]` and raise `KeyError`.
3. **Predictions dated from the end of the trimmed context.** `FoundationModel.predict`
   expands the index from `context[name].index[-1]`, so a trimmed series is predicted at the
   wrong dates. Series that already ended (id_1002) are predicted in every fold at dates with
   no actual values; the inner merge in `_calculate_metrics_backtesting_multiseries` drops
   them and `get_group` raises `KeyError`. This happens with every model, exog or not.

Both TimesFM 3.0 facts that matter: `predict(steps=5)` succeeds with contexts of 30, 183 and
91 rows in the same group because `predict_batch` left-pads every query to the batch context
length (`_Query.format`). Different lengths within a group need no handling in skforecast to
avoid a crash; they are not numerically neutral when covariates are present, see section 7.

---

## 3. Design: one contract in `FoundationModel`, thin adapters

Guiding rule: adapters translate already-normalized per-series dicts into a backend call and
nothing else. No adapter contains grouping, padding, reindexing or per-series loops that exist
to work around batch constraints. Everything that reasons about series lives once in
`FoundationModel`, driven by small declarative attributes on the adapter classes.

### 3.1 Adapter contract

Two new class attributes, next to the existing `allow_exog` and
`supports_past_only_covariates`:

| attribute                         | Chronos, TS-ICL, TabICL, TimesFM 3.0 | T0, TabPFN, Nori, Moirai, TimesFM 2.5 |
|-----------------------------------|--------------------------------------|---------------------------------------|
| `supports_heterogeneous_covariates` | `False`                              | `True`                                |
| `supports_nan_in_series`         | `True`                               | `True` (Nori drops NaN rows itself)   |

`FoundationModel` exposes both through read-only properties, like
`supports_past_only_covariates` today.

Invariants every adapter can rely on, in every code path:

1. `context[name]` and `context_exog[name]` share the same index (same dates, same length).
2. `exog[name]` has exactly `steps` rows aligned to the forecast horizon, or is `None`.
3. When `supports_heterogeneous_covariates` is `False`, every series in one call has the same
   covariate signature (same past-only columns, same future columns).
4. When `supports_nan_in_series` is `False`, no context contains NaN.

The central layer only splits and aligns. It never imputes, drops or fabricates values or
columns; anything that changes values stays in the adapter and is documented there (today:
Nori's NaN-row drop, T0's NaN column pooling).

### 3.2 New private helpers in `FoundationModel`

- `get_exog_signature(context_exog, exog) -> tuple[tuple, tuple]`: sorted past-only
  columns and sorted future columns of one series; `((), ())` when it has none. Handles
  DataFrame, Series (by `.name`) and `None`. Moved from
  `TimesFMAdapter._v3_covariate_signature`. Its future-without-history `ValueError` was
  dropped, not moved: the function is purely descriptive and the check lives only in
  `_check_exog_columns` (user-facing path). See section 7.
- `_group_series_by_exog_signature(series_names_in, context_exog, exog) -> list[list[str]]`:
  groups (`series_groups`) in first-seen order, series names within a group in input order.
- `_align_context_exog(context, context_exog) -> dict`: for each series with exog, reindex
  `context_exog[name]` to `context[name].index`. Rows outside the context are dropped, rows
  missing in exog become NaN (and are reported with `MissingValuesWarning`).

`_prepare_future_exog` and `_check_exog_columns` stay as they are.

### 3.3 `FoundationModel.predict`

Current flow: normalize context, filter `levels`, drop exog if `allow_exog` is false, prepare
future exog only when `check_inputs`, one adapter call, build the long DataFrame. New flow:

1. Normalize context and filter `levels` (unchanged).
2. Drop exog when the adapter does not accept it (unchanged).
3. Alignment, regardless of `check_inputs`: `_align_context_exog`, then
   `_prepare_future_exog` (it already accepts dicts and reindexes to the horizon, NaN-filling
   gaps). `check_inputs=False` now skips only type validation and `_check_exog_columns`,
   which holds by construction in backtesting.
4. NaN-in-series check: if `supports_nan_in_series` is `False` and any context has NaN,
   raise `ValueError` naming the series.
5. Grouping: `series_groups = _group_series_by_exog_signature(...)` when
   `supports_heterogeneous_covariates` is `False`, else `[series_names_in]`. Call `adapter.predict` once
   per group with the sub-dicts of its series (`series_names_group`) and merge the returned dicts.
6. Build the long DataFrame (unchanged; `series_names_in` order is preserved because the
   merged dict is read in that order).

### 3.4 Adapter changes (all subtractive)

- `TimesFMAdapter._predict_v3`: remove the grouping loop and `_v3_covariate_signature`;
  keep `_build_v3_covariates` and the `predict_batch` call. Declare the two attributes.
- `ChronosAdapter`, `TSICLAdapter`: `exog[name]` and `context_exog[name]` become `.get(name)`.
  Declare the attributes. Docstring and changelog note: `cross_learning` applies within a
  covariate-signature group (numerically neutral for same-signature series, and the only
  behaviour the backend allows for heterogeneous ones).
- `TabICLAdapter`: declare the attributes. No code change unless the reproduction in 2.1.C
  shows the grouping is not enough.
- `NoriAdapter`: keep `supports_nan_in_series=True` and drop NaN-target rows (and rows with
  NaN in `X_ctx`) before the in-context `fit`; raise a clear `ValueError` naming the series if
  no valid rows remain. This stays local because Nori's running-index features use absolute
  offsets, so dropping interior rows is correct for that backend and nowhere else.
- `T0Adapter`, `TabPFNAdapter`, `MoiraiAdapter`: declare the attributes only.

### 3.5 `_backtesting_foundation`

The foundation backtest stops reusing the ML-oriented trailing-NaN trim and builds its own
context per fold from the aligned `series_dict`:

- **Context** for series `k` in a fold = `series[k].loc[first_valid : train_loc_end]`, trailing
  NaN included, then `.iloc[-context_length:]`. Prediction dates therefore always start at the
  fold's test start (id_1003 in fold 1 ends on 2016-08-24, not 08-06).
- **Inclusion rule**: a series is predicted in a fold only if it has at least one valid value
  in the fold's test window and at least one valid value in its context. id_1002 is never
  predicted; id_1004 is predicted in folds 0 and 1 only. This removes the wrong-date
  predictions and the metrics `KeyError`. A fold where no requested level passes the rule is
  skipped with a `MissingValuesWarning` and contributes an empty frame, exactly as
  `_backtesting_forecaster_multiseries` does, so the output keeps its columns and the metric
  of a never predicted level is `None`.
- **Exog**: `context_exog` and `exog_test` are sliced by fold dates exactly as today and passed
  with `check_inputs=False`; step 3 of `predict` aligns them. id_1004 in fold 1 arrives with
  7 rows and reaches the adapter with 24, the last 17 NaN, with `MissingValuesWarning` unless
  `suppress_warnings=True`.
- `_calculate_metrics_backtesting_multiseries`: if a level has no rows left after the inner
  merge, return `None` for its metrics instead of raising `KeyError` (defensive; the inclusion
  rule should already prevent it).

`_extract_data_folds_multiseries` is not modified; the foundation loop keeps using it for fold
bookkeeping and exog slicing only.

### 3.6 What does not change

`adapter.predict` signature and return type, `ForecasterFoundation`, the long-format output,
`backtesting_foundation`'s public signature, `bayesian_search_foundation`.

---

## 4. Testing

Adapters accept a `model=` / `pipeline=` / `module=` injection for a mock backend and
`fixtures_adapters.py` holds shared fixtures.

1. **Grouping, once, at `FoundationModel` level.** A mock adapter with
   `supports_heterogeneous_covariates=False` that records every call. Fixture: mixed lengths,
   exog subsets of 2, 3 and 4 columns, one series with past-only columns, one series without
   exog. Assert: number of calls equals number of distinct signatures; every call is
   homogeneous; every requested series appears exactly once in the merged output, in input
   order; a mock with the attribute `True` receives a single call.
2. **Alignment in both paths.** `context_exog` longer than `context`, future exog shorter than
   `steps`, missing exog key: assert the adapter receives invariants 1 and 2 with
   `check_inputs=True` and with `check_inputs=False`.
3. **NaN in series.** Mock with `supports_nan_in_series=False` and a NaN context raises
   `ValueError` naming the series.
4. **Backend equivalence (marked `slow`, real weights, Chronos and TimesFM 3.0).** Grouped
   call equals native per-group call; a homogeneous dataset equals a single native call.
   These are the guard against the abstraction drifting from the libraries, and they should
   run in `unit-tests-latest-deps.yml`.
5. **Backtesting regression.** Fixture with one series with trailing NaN inside a fold, one
   series ending before the end of the span, one with incomplete future exog. Assert every
   prediction date lies inside its fold's test window, the ended series is absent from the
   folds after its end, and metrics are computed for every level that has predictions.
6. **TimesFM 3.0 unit test**: mixed-length members of one group produce the same per-series
   output as batch-size-1 calls on the same mock (documents that no padding is needed).
7. **Nori**: NaN-target series fits on the non-NaN rows only (mock's received `y` has no NaN
   and the expected length); all-NaN target raises a clear error.

Manual check: `dev/foundation-multiseries-multiexog.py` for every installed backend, both
`predict` and `backtesting_foundation`.

```bash
pytest skforecast/foundation/tests/ skforecast/model_selection/tests/tests_validation/test_backtesting_foundation.py -vv
```

---

## 5. Risks and open questions

- **Batch composition effects.** Which batch a series lands in can change results slightly
  (0.003 measured for Chronos; not yet measured for TimesFM 3.0, whose batch context is
  rounded to the longest member). Test 4 quantifies it per release.
- **NaN filling of future exog is a data change skforecast makes before the backend sees it.**
  It only ever adds missing rows to columns the series already has, never columns. Each
  adapter docstring must state what its backend does with NaN in covariates (Chronos: missing;
  TimesFM 3.0: linear interpolation; TabICL: warning).
- **Inclusion rule semantics.** Excluding a series from folds where it has no actual values is
  a behaviour change for anyone who relied on the old (wrong-dated) predictions. Document in
  `changelog.md`.
- **TabICL root cause** is unverified (2.1.C). If the notebook error persists after grouping,
  it is a separate issue.
- **Direct adapter calls** bypass the central layer and get no grouping. Acceptable: adapters
  are private, and the backend error messages are explicit.
- **Performance.** One backend call per distinct signature, the minimum the backends allow.
  Homogeneous datasets still make a single call.

---

## 6. Rollout order

1. `FoundationModel`: attributes, helpers, new `predict` flow, tests 1 to 3. `_predict_v3`
   loses its loop; Chronos and TS-ICL switch to `.get`. Land with `changelog.md` entry.
2. `_backtesting_foundation`: own context construction, inclusion rule, metrics guard,
   test 5.
3. Backend equivalence tests (4) and the TimesFM 3.0 mixed-length test (6).
4. Nori NaN-row drop (7).
5. TabICL: reproduce with `tabicl[forecast]`; act only if grouping is not enough.
6. Docs: adapter docstrings (NaN handling, `cross_learning` scope), user guide note that
   foundation forecasters accept per-series lengths and exog subsets.

This work goes before the split of `TimesFMAdapter` into separate 2.5 and 3.0 classes: the
split then moves less code and inherits tests 4 and 6 as its safety net.

---

## 7. Decisions after review (2026-09-09)

Review of the implementation against real backends (Chronos-2 small, TimesFM 2.5 and 3.0 on
CPU, `context_length=200`, dataset of section 1, `steps=5`). `predict` and
`backtesting_foundation` succeed on the three; every prediction is dated inside its fold;
id_1002 is never predicted, id_1004 only in folds 0 and 1, id_1003 is skipped in folds 1 and 2.
TabICL remains unverified (`tabicl[forecast]` not installed).

### 7.1 Batch composition is not neutral in TimesFM 3.0 with covariates. Decision: document, no code change.

Max abs difference of the forecast of id_1001 (30 rows) predicted alone versus in the same
covariate-signature group as another series. Series scale is about 3400.

| backend                                   | with exog | without exog |
|-------------------------------------------|----------:|-------------:|
| Chronos-2 small (`cross_learning=False`)  |     0.001 |          n/a |
| TimesFM 3.0, grouped with id_1004 (91)    |      74.1 |        0.002 |
| TimesFM 3.0, grouped with id_1002 (183)   |      74.2 |        0.002 |

The effect appears only when covariates are present (the adapter passes `padding_mode="edge"`
and the backend left-pads the shorter series to the batch context length) and does not grow
with the length gap. It is about 2 percent of the scale, it is a backend property, and it also
existed before this work (the grouping was already inside `TimesFMAdapter`). Grouping by
context length as well would multiply backend calls for a difference of this size, so it is
not done. The `_predict_v3` docstring already states that shorter series are left-padded by
the backend; that is the documented behaviour.

### 7.2 Future-without-history check lives only in `_check_exog_columns`. Decision: keep as is.

Section 3.2 originally planned to move the `ValueError` raised by
`TimesFMAdapter._v3_covariate_signature` into the central helper so that it also covered
`check_inputs=False`. The implementation dropped it instead: `get_exog_signature` never
raises (unit test `future_without_history` documents it) and the check runs only in
`_check_exog_columns` on the user-facing path. The backtesting path cannot produce a future
column without history because both slices come from the same DataFrame. A direct adapter call
with such input now fails inside `_build_v3_covariates` with a `KeyError` instead of the
former explicit message; adapters are private, so this is accepted.

### 7.3 Open points deliberately left out of this work

- Nori: `_prepare_future_exog` NaN-fills a future `exog` with gaps and `NoriRegressor.predict`
  may reject NaN. Not verified (backend not installed).
- In backtesting, a series whose exog ends mid-way changes covariate signature between folds
  (its columns become past-only) without a warning, because `_check_exog_columns` is skipped
  on that path.
- The metrics guard added in `_calculate_metrics_backtesting_multiseries` is defensive only
  and has no dedicated test.
- The user guide notebook (rollout step 6) is not updated yet.
