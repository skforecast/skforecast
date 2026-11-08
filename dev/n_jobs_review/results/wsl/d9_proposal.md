# D9: the LightGBM fast path ignores `n_jobs` (proposal, not applied)

Branch: `fix/lgbm-fast-predict-num-threads` from `origin/0.27.x` (`--no-track`).

## Code: `skforecast/utils/utils.py`, `_build_predict_function`

```diff
     if estimator_name == 'LGBMRegressor':
         booster = estimator.booster_
+        # NOTE: `booster_.predict` does not know `n_jobs`: without `num_threads`
+        # it uses all OpenMP threads. Same value as `LGBMRegressor.predict`.
+        num_threads = estimator._process_n_jobs(estimator.n_jobs)
 
         def predict_fn(X):
-            return booster.predict(X)
+            return booster.predict(X, num_threads=num_threads)
 
         return predict_fn
```

- `_process_n_jobs` is private, but identical and used the same way by
  `LGBMRegressor.predict` from LightGBM 4.0.0 (minimum allowed by
  skforecast) to 4.7.0.
- The value is computed once, when the function is built (after each fit),
  as `predict` would compute it on every call.
- The docstring list of fast paths gets: "`LGBMRegressor` (`booster_.predict`,
  with the same `num_threads` as `LGBMRegressor.predict`)".

## Test: `skforecast/utils/tests/tests_utils/test_build_predict_function.py`

```python
@pytest.mark.parametrize("n_jobs", [1, 2, -1, None], ids=lambda x: f"n_jobs={x}")
def test_build_predict_function_lightgbm_uses_num_threads_of_estimator_predict(
    n_jobs, regression_data
):
    """
    Test that the predict function of an LGBMRegressor passes to the booster
    the same `num_threads` as `estimator.predict`, derived from `n_jobs`.
    Without it, the booster uses all the OpenMP threads of the machine.
    """
    X, y = regression_data
    estimator = LGBMRegressor(n_estimators=10, n_jobs=n_jobs, verbose=-1)
    estimator.fit(X, y)

    booster = estimator.booster_
    original_predict = booster.predict
    calls = []

    def spy_predict(data, **kwargs):
        calls.append(kwargs)
        return original_predict(data, **kwargs)

    booster.predict = spy_predict
    predict_fn = _build_predict_function(estimator)
    predict_fn(X[:1])
    estimator.predict(X[:1])

    assert len(calls) == 2
    assert calls[0]["num_threads"] == calls[1]["num_threads"]
```

## Release note (0.27.0, Fixed)

"The predictions of forecasters with `LGBMRegressor` now respect its `n_jobs`:
the fast prediction path used all the threads of the machine, which made
recursive predictions with `n_jobs=1` up to 40 times slower on Linux virtual
machines (WSL2)." (numbers to confirm with D7)

## Evidence (2026-11-08, LightGBM 4.7.0, 3000 single-row predictions)

| Call | Windows | WSL2 |
|---|---|---|
| `estimator.predict` | 0.353 ms | 0.318 ms |
| `booster_.predict(X, num_threads=1)` | 0.043 ms | 0.038 ms |
| `booster_.predict(X)` (current) | 0.060 ms, +21 threads | 6.581 ms, +21 threads |

V1 cell (Recursive LGBM(1) L, no refit, n_jobs=1) in WSL2: 16.0 s -> 0.37 s.
