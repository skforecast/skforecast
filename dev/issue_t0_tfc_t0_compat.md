# Issue draft: T0Adapter is incompatible with tfc-t0 >= 0.4

**Title:** `T0Adapter` fails with `tfc-t0>=0.4`: `predict()` got an unexpected keyword argument `context`

**Labels:** bug, foundation

---

## Description

`T0Adapter` in skforecast 0.26.0 does not work with the current releases of
the backend library `tfc-t0`. Every call to `predict`, `predict_interval` or
`backtesting_foundation` with `model_id="theforecastingcompany/t0-alpha"`
fails when `tfc-t0` 0.4.0 or 0.5.0 (the latest on PyPI) is installed.

Users following the install instructions (`pip install tfc-t0`) get the latest
version, so T0 is broken out of the box.

## Reproduce

```python
# pip install skforecast==0.26.0 tfc-t0==0.5.0
from skforecast.datasets import fetch_dataset
from skforecast.foundation import FoundationModel, ForecasterFoundation

data = fetch_dataset(name="vic_electricity").resample("h").mean(numeric_only=True)
forecaster = ForecasterFoundation(
    estimator=FoundationModel(model_id="theforecastingcompany/t0-alpha", context_length=500)
)
forecaster.fit(series=data["Demand"])
forecaster.predict(steps=24)
```

```
File .../skforecast/foundation/_adapters.py:3724, in T0Adapter.predict(...)
-> 3724 forecast = self._model.predict(
   3725     context           = context_batch,
   3726     horizon           = steps,
   3727     quantiles         = query_levels,
   3728     future_covariates = future_covariates,
   3729 )

TypeError: T0Forecaster.predict() got an unexpected keyword argument 'context'
```

## Cause

`T0Adapter.predict` calls `T0Forecaster.predict(context=..., quantiles=...)`.
The `tfc-t0` API changed:

| tfc-t0 | Released | 1st argument of `predict` | Quantiles argument | Works with skforecast 0.26.0 |
|---|---|---|---|---|
| 0.3.1 and older | up to 2026-09-01 | `context` | `quantiles` | Yes |
| 0.3.2 | 2026-09-09 | `model_input` (TimeSeries) or `context` (array) overloads | `quantiles` | Yes (last working version) |
| 0.4.0 | 2026-09-15 | `model_input` | `quantiles` | No |
| 0.5.0 | 2026-09-17 | `model_input` | `quantile_levels` | No |

Unchanged in 0.5.0: `horizon`, `future_covariates` (keyword), and the returned
`Forecast.quantiles` attribute used by the adapter.

## Workaround (0.26.0)

```bash
pip install "tfc-t0<0.4"
```

Verified: the user guide `foundation-forecasting-models.ipynb` runs end to end
with `tfc-t0==0.3.2`, `torch==2.9.1` (CPU) and skforecast 0.26.0 (backtesting
MAE 145.53).

## Proposed fix (next release)

1. Adapt `T0Adapter.predict` to the new API: pass the context positionally
   (`model_input`) and use `quantile_levels`. Set a minimum version
   (`backend_package = "tfc-t0>=0.5"`) and raise a clear `ImportError` when an
   older version is installed, as `TimesFM3Adapter._load_model` already does
   for `timesfm<3`.
2. Check that `future_covariates` keeps the same shape contract
   (`[batch, n_future_covariates, context + horizon]`) in 0.5.0, and that the
   left NaN padding of shorter series is still read as missing.
3. Update the fake `T0Forecaster` in
   `skforecast/foundation/tests/tests_foundation_models/fixtures_adapters.py`
   to the new signature, and the expected `backend_package` in
   `test_get_model_info.py`.
4. Update the install command in the user guide table, in
   `skills/foundation-forecasting/SKILL.md` and, if needed, in
   `tools/ai/llms-base.txt`; then regenerate the AI context files.
5. Release note in `docs/releases/releases.md`.

Alternative if the backend API keeps changing: pin `backend_package` to a
tested range (`tfc-t0>=0.5,<0.6`) and widen it after each check.
