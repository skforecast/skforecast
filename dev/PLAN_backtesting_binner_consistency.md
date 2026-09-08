# Keep the residual binner consistent with `out_sample_residuals_by_bin_` during backtesting

## Problem

When `backtesting_forecaster` (and `backtesting_forecaster_multiseries`) runs with
`interval` set, `use_in_sample_residuals=False` and `use_binned_residuals=True`:

1. `out_sample_residuals_by_bin_` is captured once before any `fit()` and restored
   unchanged after every `fit()` (`_validation.py`, lines ~509-531 and ~201-204;
   multiseries: ~1291-1323 and ~936-939).
2. Every `fit()` refits the `QuantileBinner` on the in-sample predictions of the
   current training set, even when `store_in_sample_residuals=False`
   (`_forecaster_recursive.py`, `fit()` ~1376-1390 calls
   `_binning_in_sample_residuals`, which does `self.binner.fit(y_pred)` at ~1447).
3. At prediction time the bin index of each prediction is computed with the
   refitted binner (`_recursive_predict_bootstrapping` ~1877, conformal path ~2346)
   and used as a key into the restored residual dictionary.

Result: the dictionary keys refer to the partition of the binner the user had when
calling `set_out_sample_residuals()`, while predictions are routed with a different
partition. Bin `k` means two different prediction ranges on each side of the lookup.

## When it manifests

- Documented workflow (forecaster fitted on train+val, backtesting with
  `initial_train_size = len(train+val)`, `refit=False`): the internal fit
  reproduces the same binner. No effect (0% of predictions rerouted).
- `refit=True` (expanding or rolling window), or `initial_train_size` different from
  the data the user fitted on: 17% to 65% of test predictions were assigned to a
  different bin than under the original partition (bike_sharing, items_sales,
  fuel_consumption, h2o; see `dev/exp_backtesting_binner_consistency.py`).
- Measured effect on interval quality (coverage, width, Winkler score): within
  noise in every scenario. This is a coherence fix, not an accuracy fix.

## Decision

Restore `binner` and `binner_intervals_` together with `out_sample_residuals_by_bin_`
after every `fit()` inside backtesting, following the pattern already used for the
residuals. No new state is stored in the forecaster and no public API changes.

> **NOTE:** the residuals are not re-binned with the refitted binner because the
> forecaster does not store the `y_pred` each out-of-sample residual came from
> (`set_out_sample_residuals()` keeps only the residuals and the derived per-bin
> dictionary). Re-binning the same residuals with each fold's binner would be more
> correct than the fix chosen here, since it would reflect the prediction
> distribution of the model actually used in that fold. It requires storing the
> `y_pred` paired with each residual (bounded to 10_000 values, about 80 KB per
> series). It was measured to give intervals indistinguishable from both the
> current behaviour and the chosen fix, so it is deferred. If `y_pred` is ever
> stored for another reason (for example to allow changing `n_bins` after
> `set_out_sample_residuals()` or to simplify `append=True`), switch backtesting to
> re-binning instead of restoring the binner.

## Implementation plan

### 1. `backtesting_forecaster` (`skforecast/model_selection/_validation.py`)

- Where `out_sample_residuals_by_bin_` is captured (~509-514), also capture
  `binner_ = deepcopy(forecaster.binner)` and
  `binner_intervals_ = forecaster.binner_intervals_`.
  Use `deepcopy`: `_binning_in_sample_residuals` calls `self.binner.fit()` in place
  on the same `QuantileBinner` object, so a plain reference would be mutated by the
  first `fit()`.
- After the initial fit (~531), restore them alongside the residual dictionary.
- Add both to `kwargs_fit_predict_forecaster` (~565).
- In `_fit_predict_forecaster` (~104): add the two parameters, document them, and
  restore them after `forecaster.fit()` inside the `if fold[5] is True` block
  (~201-204).
- Only needed when `use_binned_residuals=True`; keep the `None` guard pattern.
- This also covers `ForecasterDirect`, which uses the same function and the same
  `binner` / `binner_intervals_` attributes.

### 2. `backtesting_forecaster_multiseries` (same file)

- Same capture (~1291-1296), restore after initial fit (~1323), kwargs (~1375) and
  restore in `_fit_predict_forecaster_multiseries` (~936-939).
- `binner` and `binner_intervals_` are dicts keyed by level (plus `_unknown_level`).
  `fit()` replaces the dict (`self.binner = {}`) and creates new `QuantileBinner`
  objects, so a reference to the old dict survives, but use `deepcopy` anyway for
  symmetry and safety.
- Covers `ForecasterRecursiveMultiSeries` and `ForecasterDirectMultiVariate`.

### 3. Tests (`skforecast/model_selection/tests/tests_validation/`)

- Unit test for `_fit_predict_forecaster` (and the multiseries counterpart): pass a
  fold with refit, a forecaster with known `binner_intervals_`, and assert that
  after the call `forecaster.binner_intervals_` equals the original and that
  `forecaster.binner.intervals_` matches it.
- Integration test: `backtesting_forecaster` with `refit=True`,
  `interval=[0.05, 0.95]`, `use_in_sample_residuals=False`,
  `use_binned_residuals=True`, on a series where the training data of the folds
  differs from the data used in `set_out_sample_residuals()`. Compare against
  expected predictions computed with the original binner. Update any existing
  expected values that change because of the fix.
- Run both bootstrapping and conformal paths.

### 4. Docs

- One sentence in the changelog. No user guide changes needed; the documented
  workflow is unaffected.

## Out of scope

- Re-binning out-of-sample residuals per fold (see NOTE above).
- Refitting the binner on out-of-sample predictions in `set_out_sample_residuals()`.
  Measured on 12 dataset and estimator combinations: no consistent gain over the
  current behaviour, and no binning at all was the most robust option when fewer
  than a few hundred out-of-sample residuals are available.
- Skipping the in-sample `estimator.predict(X_train)` in `fit()` when only
  out-of-sample residuals will be used (about 6% of fit time per fold on
  bike_sharing with 300 trees). Possible follow-up, needs an internal flag.
