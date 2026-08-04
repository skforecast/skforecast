# Implementation Plan: `grid_search_equivalent_date`

## Goal

Provide a dedicated model-selection function to evaluate different
`ForecasterEquivalentDate` baseline configurations via time series backtesting
and rank them by a metric. The generic `grid_search_forecaster` is incompatible
because it assumes an sklearn estimator (`estimator.set_params`) plus a
`lags_grid`, neither of which this forecaster has.

Backtesting already supports `ForecasterEquivalentDate`
(`backtesting_forecaster` accepts it), so only the search/orchestration layer is
missing.

## Scope decisions (agreed)

- **Only `grid_search_equivalent_date`.** Do NOT implement
  `random_search_equivalent_date`. The search space (`offset`, `n_offsets`,
  `agg_func`) is tiny and discrete; random sampling adds no value and can produce
  duplicated configurations. Over-engineering, skip it.
- **Option C for `param_grid`:** accept both a `dict` (Cartesian product) and a
  `list` of configuration dicts (explicit configurations, no cross product),
  normalizing scalar values internally.
- **No private helper.** Implement all logic inline inside
  `grid_search_equivalent_date`. Do NOT mirror the
  `grid_search_stats` -> `_evaluate_grid_hyperparameters_stats` split.

## Why Option C (coupled parameters)

`offset` and `n_offsets` are coupled: the meaningful unit is the *pair* (it
defines the window semantics). A pure Cartesian grid produces semantically
meaningless combinations. Example:

```python
# User wants only:
#   7-day moving average       -> offset=1, n_offsets=7
#   mean of lag-7 and lag-14   -> offset=7, n_offsets=2

# dict form (Cartesian) would ALSO test the unwanted (1, 2) and (7, 7):
param_grid = {'offset': [1, 7], 'n_offsets': [7, 2]}   # 4 combos, 2 unwanted

# list form (explicit configurations) tests exactly what is wanted:
param_grid = [
    {'offset': 1, 'n_offsets': 7},
    {'offset': 7, 'n_offsets': 2},
]
```

Both forms must be supported:

- `dict` of lists  -> Cartesian product (convenient when combinations ARE
  meaningful, e.g. sweeping `agg_func` over a set of pairs).
- `list` of dicts  -> each dict is one full configuration; scalar values allowed.
  Per-config sub-sweeps also allowed, e.g.
  `[{'offset': 7, 'n_offsets': 2, 'agg_func': [np.mean, np.median]}]`.

## Changes

### 1. Add `set_params` to `ForecasterEquivalentDate`

File: `skforecast/recursive/_forecaster_equivalent_date.py`

Add a `set_params(self, params: dict)` method (mirrors the `ForecasterStats`
convention: plain dict input, resets forecaster to an unfitted state).

Behavior:
- Accept a dict with any of: `offset`, `n_offsets`, `agg_func`.
- Update the corresponding attributes.
- Recompute `window_size = self.offset * self.n_offsets` for the `int` offset
  case (the `DateOffset` case is recomputed inside `fit`, so recomputing the raw
  product here is a harmless pre-fit placeholder, consistent with `__init__`).
- Reset `self.is_fitted = False`.

Docstring: NumPy-style, follow `.github/instructions/docstrings.instructions.md`.

```python
def set_params(self, params: dict[str, object]) -> None:
    """
    Set new values to the parameters of the forecaster. After calling this
    method, the forecaster is reset to an unfitted state. The `fit` method must
    be called before prediction.

    Parameters
    ----------
    params : dict
        Parameter values. Valid keys are `offset`, `n_offsets` and `agg_func`.

    Returns
    -------
    None
    """
    allowed = {'offset', 'n_offsets', 'agg_func'}
    invalid = set(params) - allowed
    if invalid:
        warnings.warn(
            f"Unknown parameters {sorted(invalid)} will be ignored. "
            f"Valid parameters are {sorted(allowed)}.",
            IgnoredArgumentWarning
        )
    for k, v in params.items():
        if k in allowed:
            setattr(self, k, v)

    if isinstance(self.offset, int):
        self.window_size = self.offset * self.n_offsets

    self.is_fitted = False
```

Note: import `IgnoredArgumentWarning` from `..exceptions` (add to the existing
import block).

### 2. Add `grid_search_equivalent_date` (all logic inline)

File: `skforecast/model_selection/_search.py`

Signature (mirror `grid_search_stats`, minus the private-helper split):

```python
def grid_search_equivalent_date(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold,
    param_grid: dict | list[dict],
    metric: str | Callable | list[str | Callable],
    return_best: bool = True,
    n_jobs: int | str = 'auto',
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False,
    output_file: str | None = None
) -> pd.DataFrame:
```

Inline logic:

1. **Type guard.**
   ```python
   if type(forecaster).__name__ != 'ForecasterEquivalentDate':
       raise TypeError(
           "`forecaster` must be of type `ForecasterEquivalentDate`, for all "
           "other types of forecasters use the functions available in the "
           "`model_selection` module."
       )
   ```

2. **Normalize `param_grid` into a flat list of scalar-valued config dicts.**
   - If `param_grid` is a `dict` -> `list(ParameterGrid(param_grid))`
     (Cartesian, standard behavior).
   - If `param_grid` is a `list` -> for each element, wrap scalar values into
     singleton lists and expand with `ParameterGrid`, then concatenate. This
     supports both plain configs `{'offset': 1, 'n_offsets': 7}` and per-config
     sub-sweeps `{'offset': 7, 'n_offsets': 2, 'agg_func': [np.mean, np.median]}`.

   ```python
   if isinstance(param_grid, dict):
       param_grid = list(ParameterGrid(param_grid))
   elif isinstance(param_grid, list):
       expanded = []
       for config in param_grid:
           if not isinstance(config, dict):
               raise TypeError(
                   "When `param_grid` is a list, each element must be a dict "
                   f"of parameters. Got {type(config).__name__}."
               )
           normalized = {
               k: (v if isinstance(v, (list, np.ndarray)) else [v])
               for k, v in config.items()
           }
           expanded.extend(list(ParameterGrid(normalized)))
       param_grid = expanded
   else:
       raise TypeError(
           f"`param_grid` must be a dict or a list of dicts. "
           f"Got {type(param_grid).__name__}."
       )
   ```

   Note: `agg_func` values are callables; keep them scalar unless the user wraps
   them in a list intentionally. The `isinstance(v, (list, np.ndarray))` check
   treats a bare callable as a scalar (correct).

3. **Metric bookkeeping, dedup, progress bar, output-file reset** -> copy from
   `_evaluate_grid_hyperparameters_stats` (metric list normalization,
   `metric_dict`, uniqueness check, `tqdm`, `os.remove(output_file)`).

4. **Search loop** over the normalized `param_grid`, using
   `backtesting_forecaster` (NOT `backtesting_stats`), on a
   `deepcopy_forecaster(forecaster)`:
   ```python
   forecaster_search = deepcopy_forecaster(forecaster)
   ...
   for params in param_grid:
       try:
           forecaster_search.set_params(params)
           metric_values = backtesting_forecaster(
               forecaster        = forecaster_search,
               y                 = y,
               cv                = cv,
               metric            = metric,
               exog              = None,
               interval          = None,
               n_jobs            = n_jobs,
               verbose           = verbose,
               show_progress     = False,
               suppress_warnings = suppress_warnings
           )[0]
       except Exception as e:
           warnings.warn(f"Parameters skipped: {params}. {e}", RuntimeWarning)
           continue
       metric_values = metric_values.iloc[0, :].to_list()
       ...
   ```
   `exog` is intentionally omitted from the public signature: this forecaster
   does not support exogenous variables (`supports_exog=False`).

5. **Assemble results DataFrame**: `params` column + one column per metric +
   expanded param columns (`results['params'].apply(pd.Series)`), sorted
   ascending by the first metric. Same empty-results warning as the stats
   version.

6. **`return_best`**: pick `results.loc[0, 'params']`, call
   `forecaster.set_params(best_params)` then `forecaster.fit(y=y)` (no `exog`),
   optional verbose summary.

Decorate with `@manage_warnings` (same as the stats evaluator).

### 3. Exports

File: `skforecast/model_selection/__init__.py`
- Add `grid_search_equivalent_date` to the `from ._search import (...)` block and
  to `__all__`.

## Tests

Directory: `skforecast/model_selection/tests/tests_search/`
(follow `.github/instructions/testing.instructions.md`).

New file: `test_grid_search_equivalent_date.py`. Cases:

- `TypeError` when `forecaster` is not a `ForecasterEquivalentDate`.
- `TypeError` when `param_grid` is neither dict nor list, and when a list element
  is not a dict.
- **dict form** (Cartesian): `{'offset': [1, 7], 'n_offsets': [2, 7]}` produces 4
  rows.
- **list form** (explicit configs): `[{'offset': 1, 'n_offsets': 7},
  {'offset': 7, 'n_offsets': 2}]` produces exactly 2 rows (the key coupling
  requirement).
- **list form with per-config sub-sweep**: one dict with
  `agg_func=[np.mean, np.median]` expands to 2 rows.
- Results sorted ascending by first metric; expected metric values asserted
  against precomputed numbers (hard-coded expected values per testing
  conventions).
- `return_best=True` refits `forecaster` with best params and leaves it fitted
  (`forecaster.is_fitted is True`, params match best row).
- `output_file` is written and matches the results.
- Multiple metrics (list) -> one column per metric; duplicated metric names ->
  `ValueError`.
- `set_params` unit tests on `ForecasterEquivalentDate`: updates attributes,
  recomputes `window_size` for int offset, resets `is_fitted`, warns on unknown
  keys.

## Docs (follow-up, optional in this PR)

- Add `grid_search_equivalent_date` to the model-selection API reference page.
- Add a short example in the baseline user guide showing the list-of-configs form
  with the moving-average vs lag-mean example, emphasizing the coupled-parameter
  semantics.

## Out of scope

- `random_search_equivalent_date` (intentionally not implemented).
- `bayesian_search_equivalent_date` (unnecessary for a tiny discrete space).
- Multi-estimator support on the forecaster.
