# Plan: `bayesian_search_foundation`

## Context

The cost of zero-shot foundation models is dominated by the context window (attention scales quadratically with context_length, and memory footprint scales linearly). The new function bayesian_search_foundation lets practitioners empirically identify the shortest context length and lightest adapter configuration that preserves forecasting accuracy on their data, while reducing inference-time compute, memory, and energy consumption in production. This directly supports the responsible and sustainable adoption of large pre-trained time-series models within the PyData ecosystem.

`skforecast` can tune hyperparameters of ML and multi-series forecasters via
`bayesian_search_forecaster` / `bayesian_search_forecaster_multiseries`
(Optuna-based), but there is no equivalent for `ForecasterFoundation`. Foundation
models are zero-shot (no weight training), yet their **inference-time
configuration** materially affects accuracy, above all `context_length`, plus
adapter-specific knobs (`point_estimate`, `cross_learning`, `temporal_features`,
`mode`, ...). Today users must hand-loop over these settings and call
`backtesting_foundation` manually.

This adds `bayesian_search_foundation`: an Optuna search that, for each trial,
applies a sampled configuration via `set_params` and evaluates it with
`backtesting_foundation`, returning a ranked results DataFrame and the Optuna
`study` (identical output contract to the existing search functions).

**Decided scope (confirmed with user):**
- `bayesian_search_foundation` only (no grid/random variant for now).
- Point metrics only in v1 (metric computed on the median forecast, mirroring
  `bayesian_search_forecaster`). No `quantiles`/CRPS objective plumbing.
- Generic passthrough search space: `search_space(trial)` may return any dict of
  adapter parameters; forwarded through `set_params`, with the adapter itself
  validating unknown keys. Per-adapter tunable params documented in the docstring.

## Design constraints unique to foundation search (vs. the generic template)

Mirror `bayesian_search_forecaster_multiseries` ([_search.py:1573](skforecast/model_selection/_search.py#L1573)),
but adapt for foundation semantics:

- **No `lags` / `set_lags`.** Foundation forecasters have no lags. Drop the
  `'lags'` special-casing entirely; the whole `search_space` dict goes to
  `set_params`. Drop the `lags` column from the results DataFrame.
- **`TimeSeriesFold` only.** `backtesting_foundation` accepts only `TimeSeriesFold`
  and `check_backtesting_input` requires an integer `initial_train_size` for
  `ForecasterFoundation`. Reject `OneStepAheadFold` with a clear `TypeError`
  (so there is no one-step-ahead / split-cache branch to implement).
- **No `n_jobs`.** `backtesting_foundation` is not parallelized across folds; omit
  the argument.
- **`series` (not `y`).** Foundation is inherently multi-series-capable; accept
  `series: pd.Series | pd.DataFrame | dict` and support `levels` +
  `add_aggregated_metric`, exactly like `backtesting_foundation`.
- **Deep-copy behavior.** Reuse `deepcopy_forecaster` at the top (consistent with
  the other search functions). For real models `sklearn.clone` drops the loaded
  pipeline, which then reloads lazily **once** on the copy and is reused across
  trials (changing only `context_length` on Chronos/TabICL/TabPFN does not reset
  the pipeline, so this stays cheap). Tests patch this symbol with plain
  `deepcopy` — see Testing.
- **`return_best` refit** uses the foundation API:
  `forecaster.set_params(best_params)` then `forecaster.fit(series=series, exog=exog)`
  (note `series=`, and `ForecasterFoundation.fit` has no `store_in_sample_residuals`
  argument — do not pass it).

## Implementation

### 1. `skforecast/model_selection/_search.py` — new public function

Add `bayesian_search_foundation` (place it after
`bayesian_search_forecaster_multiseries`). Signature:

```python
def bayesian_search_foundation(
    forecaster: object,
    series: pd.Series | pd.DataFrame | dict,
    cv: TimeSeriesFold,
    search_space: Callable,
    metric: str | Callable | list[str | Callable],
    aggregate_metric: str | list[str] | None = None,
    levels: str | list[str] | None = None,
    exog: pd.Series | pd.DataFrame | dict | None = None,
    n_trials: int = 20,
    random_state: int = 123,
    return_best: bool = True,
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False,
    output_file: str | None = None,
    kwargs_create_study: dict | None = None,
    kwargs_study_optimize: dict | None = None,
) -> tuple[pd.DataFrame, object]:
```

Body, mirroring the TimeSeriesFold path of `bayesian_search_forecaster_multiseries`:

1. **Validate.** `forecaster_search = deepcopy_forecaster(forecaster)`. Type guard:
   raise `TypeError` unless `type(forecaster).__name__ == 'ForecasterFoundation'`
   (mirrors `backtesting_foundation` at [_validation.py:2673](skforecast/model_selection/_validation.py#L2673)).
   Raise `TypeError` unless `type(cv).__name__ == 'TimeSeriesFold'`. If
   `return_best and exog is not None`, length-check exog vs series.
2. **Metric / levels / aggregation setup.** Reuse the exact block from the
   multiseries search: `aggregate_metric` defaults to
   `['weighted_average', 'average', 'pooling']`; normalize `metric` to a list and
   build `metric_names`; `levels = _initialize_levels_model_selection_multiseries(...)`;
   `add_aggregated_metric = len(levels) > 1`, expanding `metric_names` to
   `f"{name}__{agg}"` when true.
3. **Objective closure** `_objective(trial, ...)` (loop-invariants bound as default
   kwargs):
   - `sample = search_space(trial)`; enforce `sample.keys() == trial.params.keys()`
     (`ValueError` otherwise) — same contract as the generic search.
   - `forecaster_search.set_params(sample)` (whole dict; no lags split).
   - `metrics, _ = backtesting_foundation(forecaster=forecaster_search, series=series, cv=cv, exog=exog, levels=levels, metric=metric, add_aggregated_metric=add_aggregated_metric, verbose=verbose, show_progress=False, suppress_warnings=suppress_warnings)`.
   - Filter/reshape metrics into one row of `metric_names` columns (copy the
     multiseries logic: filter `metrics['levels']` on `aggregate_metric` when
     `add_aggregated_metric` else on `levels[0]`; transpose+stack).
   - `trial.set_user_attr(name, value)` for each; `return metrics.loc[0, metric_names[0]]`.
4. **Study + optimize.** Identical to the generic search: default
   `TPESampler(multivariate=True, group=True, seed=random_state)`,
   direction from `__skforecast_tags__['forecaster_task']` (regression -> minimize),
   optuna logging/`output_file` handling, `study.optimize(...)` inside the
   categorical-warning suppression block.
5. **Results DataFrame.** Iterate `TrialState.COMPLETE` trials. Columns:
   `trial_number`, `levels`, `params`, then metric columns, then expanded param
   columns via `results['params'].apply(pd.Series)`. **No `lags` column.** Sort by
   `metric_names[0]` (ascending for regression). `params` = `trial.params` verbatim.
6. **`return_best`.** On the original `forecaster`:
   `forecaster.set_params(best_params)`; `forecaster.fit(series=series, exog=exog)`;
   optional verbose print of best config/metric.
7. `return results, study`.

Docstring: full NumPy-style. Document that only `TimeSeriesFold` is supported,
that there are no lags, and include a per-adapter tunable-parameter reference
(context_length everywhere; Chronos `cross_learning`; TabICL/TabPFN
`point_estimate`, `temporal_features`; TabPFN `mode`; TimesFM `max_horizon`) with
a note that changing `model_id`/`device*`/`torch_dtype` (and `context_length` on
TimesFM/Moirai) forces a model reload and is expensive.

### 2. `skforecast/model_selection/__init__.py`

Add `bayesian_search_foundation` to the `_search` import block and to `__all__`.

### 3. Verify `check_backtesting_input` path

No change expected. `backtesting_foundation` already runs `check_backtesting_input`
internally, and the objective calls `backtesting_foundation` (not the private
core), so validation is inherited. Confirm during implementation that no
search-specific validation helper needs a `ForecasterFoundation` branch.

## Testing

New file: `skforecast/model_selection/tests/tests_search/test_bayesian_search_foundation.py`.

Follow the existing foundation test pattern:
- Import fixtures from
  `skforecast.foundation.tests.tests_forecaster_foundation.fixtures_forecaster_foundation`
  (`FakePipeline`, `make_forecaster`, `y`, `series_wide`, `exog`). The
  `FakePipeline` makes predictions deterministic (`q_0.5 -> 0.5`), so metrics are
  known constants.
- **Critical:** patch the deepcopy symbol so the injected fake pipeline survives
  cloning:
  `patch("skforecast.model_selection._search.deepcopy_forecaster", side_effect=deepcopy)`.
  (Confirm the exact import name `_search.py` uses and patch that symbol.)
- Silence Optuna/tqdm as in
  [test_bayesian_search_forecaster.py:32-33](skforecast/model_selection/tests/tests_search/test_bayesian_search_forecaster.py#L32).

Cases:
1. Single series (`y`) + `TimeSeriesFold`: `search_space` varying `context_length`
   (e.g. `trial.suggest_categorical('context_length', [24, 48])`); assert results
   DataFrame shape/columns (`trial_number`, `levels`, `params`, metric, expanded),
   n_trials rows, sorted ascending, and that `study` is returned.
2. Multi-series (`series_wide`) with `levels`/`add_aggregated_metric`; assert
   aggregated metric column names.
3. `search_space` key mismatch -> `ValueError`.
4. `cv` is `OneStepAheadFold` -> `TypeError`.
5. Non-`ForecasterFoundation` forecaster -> `TypeError`.
6. `return_best=True`: original forecaster ends fitted with best params
   (assert `forecaster.context_length`/adapter param equals the best trial's).

Run:
```
pytest skforecast/model_selection/tests/tests_search/test_bayesian_search_foundation.py -vv
```
(Use conda env `skforecast_22_py13`; call its `python.exe` directly with
`PYTHONIOENCODING=utf-8` to avoid the known `conda run` unicode crash on Windows.)

## Verification (end-to-end)

After implementation, in the env python, run a real end-to-end smoke test with the
`FakePipeline`-backed forecaster (no model download) exercising a 3-trial
`context_length` search over a small series and confirming `return_best` refits
the original forecaster. Then run the full search + backtesting test suites:
```
pytest skforecast/model_selection/tests/tests_search/ -vv
pytest skforecast/model_selection/tests/tests_validation/test_backtesting_foundation.py -vv
```

## Out of scope (v1)

- `grid_search_foundation` / `random_search_foundation` (can follow the same pattern later).
- Probabilistic-metric (CRPS/pinball) objective — would require wiring `quantiles`
  through `backtesting_foundation`'s metric computation.
- `OneStepAheadFold` support (incompatible with zero-shot foundation semantics).
- Documentation site pages / user guide updates (code + docstring + tests only here).
