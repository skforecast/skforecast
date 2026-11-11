"""
Execute the skeleton of the report with the real backend (h2ofm 0.43.1 and
the TimeH2O weights) and with the fake of `fake_timeh2o.py`:

0. Contract and parameters (no weights): required class attributes of
   PR #1450, `get_params`, `set_params`, `clone`, validation, reserved
   `predict_kwargs`, `ImportError`, `LicenseWarning`, `get_model_info`, and
   the calls proposed for `tools/foundation_compat/tests/contracts.py`
   against the signatures of h2ofm.
1. Smoke test: `FoundationModel`, `ForecasterFoundation` (`predict`,
   `predict_interval`, `predict_quantiles`) and `backtesting_foundation`
   with exog.
2. Heterogeneous batch (lengths, exog columns, NaN): one backend call, no
   padding by `FoundationModel`, each series equal to the series alone.
3. Quantiles: any level, requested order, duplicates, 0 and 1.
4. Covariate dtypes: float, int, bool, nullable `Int64` and `boolean` with
   NA, `category`, strings, a Series exog, past-only columns, inf.
5. Devices: MPS falls back to CPU with a warning that points to the user's
   call (torch patched), `batch_size` reaches the backend.
5b. Warning for flat or nearly flat known covariates: constant, nearly
   constant, boolean, varying, labels, unchanged future, NaN history,
   category, columns only in the history or only in the future, `None`
   inputs, mixed dtypes, the threshold, the message, the location of the
   warning, and no warning with the tiny checkpoint.
5c. Boolean columns that reach the adapter as `object` (a missing context
   timestamp filled by `FoundationModel`) are passed as 0/1 numbers.
6. Tiny checkpoint with exog (the backend ignores it with a warning).
7. The fake backend with the skeleton.

The skeleton relies on PR #1450 (skforecast/skforecast#1450, merged in
`0.27.x` on 2026-11-05), so it must run against a skforecast that includes
it (`0.27.x` from commit fc7efd0).

Usage, from this folder after `extract_skeleton.py`:

    HF_HUB_OFFLINE=1 python -W ignore run_checks.py > run_checks.log 2>&1

By default it checks the code extracted from the report and registers it at
runtime, overwriting any `TimeH2OAdapter` of the installed package. Once the
adapter is implemented, `--package` checks the package instead: it imports
the adapter, the helper and `FakeTimeH2O` from skforecast and only asserts
that the adapter and its licence entry are registered (no extraction needed):

    HF_HUB_OFFLINE=1 python -W ignore run_checks.py --package > run_checks_package.log 2>&1
"""
import inspect
import re
import sys
import warnings
import numpy as np
import pandas as pd

PACKAGE = "--package" in sys.argv[1:]
if PACKAGE:
    from skforecast.foundation import _adapters as sk
    from skforecast.foundation._utils import _warn_if_flat_known_covariates
    from skforecast.foundation.tests.tests_foundation_models.fixtures_adapters import (
        FakeTimeH2O,
    )
else:
    sys.path.insert(0, ".")
    import timeh2o_skeleton as sk
    from license_entry import LICENSE_ENTRY
    from shared_changes import _warn_if_flat_known_covariates
    from fake_timeh2o import FakeTimeH2O
from sklearn.base import clone  # noqa: E402
from skforecast.exceptions import (  # noqa: E402
    FeatureOutOfRangeWarning, IgnoredArgumentWarning, LicenseWarning,
)
from skforecast.foundation import (  # noqa: E402
    FoundationModel, ForecasterFoundation, _adapters, _utils, get_model_info,
)
from skforecast.foundation._adapter_base import _AdapterBase  # noqa: E402
from skforecast.model_selection import TimeSeriesFold, backtesting_foundation  # noqa: E402

TimeH2OAdapter = sk.TimeH2OAdapter
V1, TINY = "h2oai/timeh2o-v1", "h2oai/timeh2o-v1-tiny"
STEPS = 24

assert "requires_equal_length_context" in _AdapterBase._REQUIRED_CLASS_ATTRIBUTES, (
    "run against a skforecast that includes PR #1450 (see the docstring)"
)
if PACKAGE:
    assert _adapters._ADAPTER_REGISTRY["h2oai/timeh2o"] is TimeH2OAdapter
    assert "h2oai/timeh2o" in _utils._MODEL_LICENSES
else:
    _adapters._ADAPTER_REGISTRY["h2oai/timeh2o"] = TimeH2OAdapter
    _utils._MODEL_LICENSES.update(LICENSE_ENTRY)
print(f"mode: {'package' if PACKAGE else 'report skeleton'}", flush=True)

ok = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f": {detail}" if detail else ""), flush=True)
    ok.append(bool(cond))


def raises(fn, exc_type, pattern=None):
    try:
        fn()
    except exc_type as exc:
        return pattern is None or re.search(pattern, str(exc)) is not None, str(exc)[:160]
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"
    return False, "no error"


def hourly(n, seed=0, start="2024-01-01", name="y"):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 50 + 0.01 * t + 5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n)
    return pd.Series(values, index=pd.date_range(start, periods=n, freq="h"), name=name)


# 0. Contract and parameters
# ==============================================================================
missing = [a for a in _AdapterBase._REQUIRED_CLASS_ATTRIBUTES if a not in vars(TimeH2OAdapter)]
check("every required class attribute is declared (PR #1450 contract)", not missing, str(missing))
check("requires_equal_length_context is False and supports_nan_in_series True",
      TimeH2OAdapter.requires_equal_length_context is False and TimeH2OAdapter.supports_nan_in_series)
a = TimeH2OAdapter(V1)
params = a.get_params()
check("get_params", set(params) == {"model_id", "context_length", "device", "batch_size", "predict_kwargs"}
      and params["predict_kwargs"] is None and params["device"] == "auto"
      and params["batch_size"] == 256, str(params))
for name, kwargs, pattern in [
    ("prefix", dict(model_id="h2oai/h2ogpt-4096-llama2-7b"), "must start with 'h2oai/timeh2o'"),
    ("context_length", dict(model_id=V1, context_length=0), "context_length"),
    ("batch_size", dict(model_id=V1, batch_size=0), "batch_size"),
    ("reserved predict_kwargs", dict(model_id=V1, predict_kwargs={"future_covariates": None}),
     "cannot include"),
]:
    passed, detail = raises(lambda: TimeH2OAdapter(**kwargs), ValueError, pattern)
    check(f"init rejects {name}", passed, detail)
a._model = object()
a.set_params(context_length=512, predict_kwargs={"patch_ensemble": False})
check("set_params context_length and predict_kwargs keep the model", a._model is not None)
for key, value in [("device", "cpu"), ("batch_size", 64), ("model_id", TINY)]:
    a._model = object()
    a.set_params(**{key: value})
    check(f"set_params {key} resets the model", a._model is None)
passed, detail = raises(lambda: a.set_params(predict_kwargs={"quantile_levels": [0.5]}), ValueError, "cannot include")
check("set_params rejects reserved predict_kwargs", passed, detail)
passed, detail = raises(lambda: a.set_params(revision="main"), ValueError, "Invalid parameter")
check("set_params rejects unknown parameters", passed, detail)
fm_clone = clone(FoundationModel(V1, context_length=1024, batch_size=32, predict_kwargs={"patch_ensemble": False}))
check("clone round trip", fm_clone.adapter.get_params() == {
    "model_id": V1, "context_length": 1024, "device": "auto", "batch_size": 32,
    "predict_kwargs": {"patch_ensemble": False}}, str(fm_clone.adapter.get_params()))
saved = sys.modules.get("h2ofm")
sys.modules["h2ofm"] = None
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    passed, detail = raises(lambda: TimeH2OAdapter(V1)._load_model(), ImportError,
                            re.escape('pip install "h2ofm>=0.43"'))
check("ImportError without h2ofm and no LicenseWarning",
      passed and not any(issubclass(x.category, LicenseWarning) for x in w), detail)
if saved is not None:
    sys.modules["h2ofm"] = saved
else:
    del sys.modules["h2ofm"]
info = get_model_info(V1)
check("get_model_info", (
    info.adapter, info.allow_exog, info.supports_past_only_covariates,
    info.supports_categorical_covariates, info.supports_heterogeneous_covariates,
    info.supported_quantiles, info.requires_hf_auth, info.weights_repo_id,
    info.license, info.commercial_use_restricted,
) == ("TimeH2OAdapter", True, False, True, True, None, True, V1,
      "h2o-model-non-commercial-licence-v1.0", True), str(info))
check("get_model_info of the tiny checkpoint", get_model_info(TINY).license_url
      == "https://huggingface.co/h2oai/timeh2o-v1/blob/main/LICENSE", get_model_info(TINY).license_url)

import h2ofm  # noqa: E402
from h2ofm.timeh2o import Forecast  # noqa: E402

CONTRACT = [
    (h2ofm.TimeH2O.from_pretrained, 1, ["device", "batch_size"]),
    (h2ofm.TimeH2O.forecast_many, 1, ["horizon", "future_covariates", "quantile_levels"]),
    (Forecast, 0, ["quantiles", "quantile_levels"]),
]
for target, n_positional, kwargs in CONTRACT:
    signature = inspect.signature(target)
    try:
        if target is h2ofm.TimeH2O.forecast_many:
            signature.bind_partial(None, *[None] * n_positional, **dict.fromkeys(kwargs))
        else:
            signature.bind_partial(*[None] * n_positional, **dict.fromkeys(kwargs))
        bound = all(k in signature.parameters for k in kwargs)
    except TypeError as exc:
        bound = False
        print(exc)
    check(f"contract: {target.__qualname__} accepts {n_positional} positional and {kwargs}", bound)

# 1. Smoke test with the real backend
# ==============================================================================
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    model = FoundationModel(V1, device="cpu")
    y = hourly(800)
    model.fit(series=y)
    pred = model.predict(steps=STEPS)
lic = [x for x in w if issubclass(x.category, LicenseWarning)]
check("LicenseWarning on load", len(lic) == 1 and "H2O Model Non-Commercial Licence" in str(lic[0].message)
      and "blob/main/LICENSE" in str(lic[0].message), str(lic[0].message) if lic else "")
expected_index = pd.date_range(y.index[-1] + pd.Timedelta(hours=1), periods=STEPS, freq="h")
check("FoundationModel.predict output", list(pred.columns) == ["level", "pred"]
      and pred.index.equals(expected_index) and np.isfinite(pred["pred"]).all())
backend = model.adapter._model
direct = backend.forecast(y.to_numpy(), horizon=STEPS).point
check("point forecast equals forecast.point of the backend",
      np.allclose(pred["pred"].to_numpy(), direct, atol=1e-5),
      f"max abs diff {np.max(np.abs(pred['pred'].to_numpy() - direct)):.2e}")

forecaster = ForecasterFoundation(estimator=FoundationModel(V1, device="cpu"))
forecaster.estimator.adapter._model = backend
idx = pd.date_range("2024-01-01", periods=900 + STEPS, freq="h")
exog_all = pd.DataFrame({
    "temp": 15 + 8 * np.sin(2 * np.pi * np.arange(900 + STEPS) / 24),
    "holiday": (idx.dayofweek >= 5),
}, index=idx)
y2 = hourly(900) + 0.5 * exog_all["temp"].iloc[:900].to_numpy()
forecaster.fit(series=y2, exog=exog_all.iloc[:900])
p = forecaster.predict(steps=STEPS, exog=exog_all.iloc[900:])
pi = forecaster.predict_interval(steps=STEPS, exog=exog_all.iloc[900:], interval=[0.1, 0.9])
pq = forecaster.predict_quantiles(steps=STEPS, exog=exog_all.iloc[900:], quantiles=[0.05, 0.5, 0.95])
check("ForecasterFoundation predict, predict_interval and predict_quantiles", (
    np.isfinite(p["pred"]).all() and (pi["lower_bound"] <= pi["upper_bound"]).all()
    and list(pq.columns) == ["level", "q_0.05", "q_0.5", "q_0.95"]
    and np.allclose(pq["q_0.5"], p["pred"])))
cv = TimeSeriesFold(steps=12, initial_train_size=900 - 24, refit=False)
metric, bt = backtesting_foundation(
    forecaster=forecaster, series=y2, exog=exog_all.iloc[:900], cv=cv,
    metric="mean_absolute_error", show_progress=False,
)
check("backtesting_foundation with exog",
      len(bt) == 24 and np.isfinite(metric["mean_absolute_error"].iloc[0])
      and metric["n_evaluated"].iloc[0] == 24,
      f"MAE {metric['mean_absolute_error'].iloc[0]:.3f}")

# 2. Heterogeneous batch
# ==============================================================================
lengths = {"s1": 900, "s2": 400, "s3": 700}
series, context_exog, exog = {}, {}, {}
for i, (name, n) in enumerate(lengths.items()):
    s = hourly(n, seed=i, start=f"2024-01-0{i + 1}", name=name)
    if name == "s2":
        s.iloc[100:130] = np.nan
    full_idx = pd.date_range(s.index[0], periods=n + STEPS, freq="h")
    frame = pd.DataFrame({
        "temp": 15 + 8 * np.sin(2 * np.pi * np.arange(n + STEPS) / 24),
        "kind": np.where(np.arange(n + STEPS) % 168 < 120, "work", "weekend"),
    }, index=full_idx)
    cols = {"s1": ["temp", "kind"], "s2": ["temp"], "s3": None}[name]
    series[name] = s
    context_exog[name] = frame.iloc[:n][cols] if cols else None
    exog[name] = frame.iloc[n:][cols] if cols else None
fm = FoundationModel(V1, device="cpu")
fm.adapter._model = backend
received = []
original_predict = fm.adapter.predict


def spy(**kwargs):
    received.append({k: len(v) for k, v in kwargs["context"].items()})
    return original_predict(**kwargs)


fm.adapter.predict = spy
batch = fm.predict(steps=STEPS, context=series, context_exog=context_exog, exog=exog, quantiles=[0.1, 0.5, 0.9])
check("heterogeneous batch: one adapter call, unpadded lengths",
      received == [lengths], str(received))
diffs = {}
for name in series:
    alone = fm.predict(steps=STEPS, context={name: series[name]}, context_exog={name: context_exog[name]},
                       exog={name: exog[name]}, quantiles=[0.1, 0.5, 0.9])
    b = batch[batch["level"] == name]
    diffs[name] = float(np.max(np.abs(b.iloc[:, 1:].to_numpy() - alone.iloc[:, 1:].to_numpy())))
check("heterogeneous batch equals each series alone", max(diffs.values()) < 1e-3,
      str({k: f"{v:.1e}" for k, v in diffs.items()}))

# 3. Quantiles
# ==============================================================================
q = fm.predict(steps=STEPS, context=y, quantiles=[0.9, 0.01, 0.33, 0.5])
check("any level, in the requested order", list(q.columns) == ["level", "q_0.9", "q_0.01", "q_0.33", "q_0.5"]
      and (q["q_0.01"] < q["q_0.33"]).all() and (q["q_0.33"] < q["q_0.5"]).all() and (q["q_0.5"] < q["q_0.9"]).all())
q_dup = fm.predict(steps=STEPS, context=y, quantiles=[0.5, 0.1, 0.5])
check("duplicated levels: FoundationModel keeps one column per level",
      list(q_dup.columns) == ["level", "q_0.5", "q_0.1"]
      and np.allclose(q_dup["q_0.5"].to_numpy(), q["q_0.5"].to_numpy()), str(list(q_dup.columns)))
passed, detail = raises(lambda: fm.predict(steps=STEPS, context=y, quantiles=[0.0, 0.5]), ValueError,
                        "strictly between 0 and 1")
check("quantile 0 raises the ValueError of the backend", passed, detail)
passed, detail = raises(lambda: fm.predict(steps=STEPS, context=y, quantiles=[0.5, 1.0]), ValueError,
                        "strictly between 0 and 1")
check("quantile 1 raises the ValueError of the backend", passed, detail)

# 4. Covariate dtypes
# ==============================================================================
n = 600
idx = pd.date_range("2024-01-01", periods=n + STEPS, freq="h")
rng = np.random.default_rng(0)
frame = pd.DataFrame({
    "f": rng.normal(size=n + STEPS),
    "i": rng.integers(0, 5, n + STEPS),
    "b": rng.random(n + STEPS) < 0.2,
    "i_na": pd.array(rng.integers(0, 5, n + STEPS), dtype="Int64"),
    "b_na": pd.array(rng.random(n + STEPS) < 0.2, dtype="boolean"),
    "c": pd.Categorical(rng.choice(["a", "b", "c"], n + STEPS)),
    "s": rng.choice(["x", "y"], n + STEPS).astype(object),
}, index=idx)
frame.loc[idx[5], "i_na"] = pd.NA
frame.loc[idx[n + 2], "b_na"] = pd.NA
frame.loc[idx[7], "s"] = None
frame.loc[idx[9], "f"] = np.nan
yy = hourly(n)
fm_cov = FoundationModel(V1, device="cpu")
fake = FakeTimeH2O()
fm_cov.adapter._model = fake
fm_cov.predict(steps=STEPS, context=yy, context_exog=frame.iloc[:n], exog=frame.iloc[n:])
rows = fake.last_future_covariates[0]
kinds = {
    col: "numeric" if (pd.api.types.is_numeric_dtype(r) and not pd.api.types.is_bool_dtype(r)) else "labels"
    for col, r in zip(sorted(frame.columns), rows)
}
check("dtypes: numeric for float, int, Int64, bool and boolean (cast to 0/1); labels for category and strings",
      kinds == {"b": "numeric", "b_na": "numeric", "c": "labels", "f": "numeric", "i": "numeric",
                "i_na": "numeric", "s": "labels"} and all(len(r) == n + STEPS for r in rows)
      and rows[1].isna().sum() == 1 and set(rows[0].unique()) <= {0.0, 1.0}, str(kinds))
fm_cov.adapter._model = backend
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    out = fm_cov.predict(steps=STEPS, context=yy, context_exog=frame.iloc[:n], exog=frame.iloc[n:],
                         quantiles=[0.1, 0.5, 0.9])
# FoundationModel raises a MissingValuesWarning when the future exog holds NaN
# (the NA of `b_na`), even if no timestamp is missing; only h2ofm warnings count.
backend_w = [x for x in w if x.category is UserWarning]
check("real backend with every dtype, no warning from h2ofm",
      np.isfinite(out.iloc[:, 1:].to_numpy()).all() and not backend_w,
      str([str(x.message)[:80] for x in backend_w]))
out_series = fm_cov.predict(steps=STEPS, context=yy, context_exog=frame.iloc[:n]["f"], exog=frame.iloc[n:]["f"])
check("Series exog", np.isfinite(out_series["pred"]).all())
fm_cov.adapter._model = fake
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    fm_cov.predict(steps=STEPS, context=yy, context_exog=frame.iloc[:n][["f", "s"]], exog=frame.iloc[n:][["f"]])
check("past-only column: IgnoredArgumentWarning and not forwarded",
      any(issubclass(x.category, IgnoredArgumentWarning) for x in w)
      and len(fake.last_future_covariates[0]) == 1 and fake.last_past_covariates is None)
fm_cov.adapter._model = backend
bad = frame.copy()
bad.loc[idx[3], "f"] = np.inf
passed, detail = raises(lambda: fm_cov.predict(steps=STEPS, context=yy, context_exog=bad.iloc[:n][["f"]],
                                               exog=bad.iloc[n:][["f"]]), ValueError, "infinity")
check("inf in a covariate raises the ValueError of the backend", passed, detail)

# 5. Devices and batch_size
# ==============================================================================
import torch  # noqa: E402

forecaster_mps = ForecasterFoundation(estimator=FoundationModel(TINY))
forecaster_mps.fit(series=y)
cuda, mps = torch.cuda.is_available, torch.backends.mps.is_available
torch.cuda.is_available = lambda: False
torch.backends.mps.is_available = lambda: True
try:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        forecaster_mps.predict(steps=STEPS)
        line = inspect.currentframe().f_lineno - 1
finally:
    torch.cuda.is_available, torch.backends.mps.is_available = cuda, mps
mps_w = [x for x in w if "MPS device is not supported by TimeH2O" in str(x.message)]
check("device='auto' on MPS falls back to CPU with a warning at the user's call",
      len(mps_w) == 1 and mps_w[0].filename == __file__ and mps_w[0].lineno == line
      and str(forecaster_mps.estimator.adapter._model.device) == "cpu",
      f"{mps_w[0].filename.split('/')[-1]}:{mps_w[0].lineno}" if mps_w else str([str(x.message)[:60] for x in w]))
adapter_mps = TimeH2OAdapter(TINY, device="mps:0", batch_size=8)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    adapter_mps._load_model()
check("device='mps:0' falls back to CPU, batch_size reaches the backend",
      str(adapter_mps._model.device) == "cpu" and adapter_mps._model.batch_size == 8
      and any("MPS" in str(x.message) for x in w))

# 5b. Warning for flat or nearly flat known covariates (fake backend, the
# helper called directly and the tiny checkpoint)
# ==============================================================================
def flat_warnings(history, future, name="x"):
    """
    Run `predict` with one covariate and return the FeatureOutOfRangeWarnings.
    """
    n_hist = len(history)
    idx_cov = pd.date_range("2024-01-01", periods=n_hist + STEPS, freq="h")
    frame_cov = pd.DataFrame({name: pd.concat([pd.Series(history), pd.Series(future)], ignore_index=True).to_numpy()},
                             index=idx_cov)
    model_flat = FoundationModel(V1, model=FakeTimeH2O())
    yy_flat = pd.Series(np.arange(n_hist, dtype=float), index=idx_cov[:n_hist], name="y")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model_flat.predict(steps=STEPS, context=yy_flat, context_exog=frame_cov.iloc[:n_hist],
                           exog=frame_cov.iloc[n_hist:])
    return [x for x in w if issubclass(x.category, FeatureOutOfRangeWarning)]


rng_flat = np.random.default_rng(1)
n_flat = 500
cases_flat = {
    "constant price that changes": (np.full(n_flat, 2.5), np.full(STEPS, 2.56), True),
    "price with noise 1e-4 that changes by 0.06": (2.5 + rng_flat.normal(0, 1e-4, n_flat), np.full(STEPS, 2.56), True),
    "bool indicator False over the context": (np.zeros(n_flat, dtype=bool), np.r_[[True, True], np.zeros(STEPS - 2, dtype=bool)], True),
    "varying covariate": (np.sin(np.arange(n_flat) / 10), np.sin(np.arange(n_flat, n_flat + STEPS) / 10), False),
    "indicator with 50 ones in the context": (np.r_[np.ones(50), np.zeros(n_flat - 50)], np.ones(STEPS), False),
    "constant price that does not change": (np.full(n_flat, 2.5), np.full(STEPS, 2.5), False),
    "history all NaN": (np.full(n_flat, np.nan), np.full(STEPS, 2.56), False),
    "labels (strings)": (np.full(n_flat, "a", dtype=object), np.full(STEPS, "b", dtype=object), False),
}
for case, (history, future, expected) in cases_flat.items():
    found = flat_warnings(history, future)
    check(f"flat covariate warning, {case}: {'warns' if expected else 'no warning'}",
          bool(found) == expected and (not found or "('y', 'x')" in str(found[0].message)),
          str(found[0].message)[:90] if found else "")
forecaster_flat = ForecasterFoundation(estimator=FoundationModel(V1))
forecaster_flat.estimator.adapter._model = FakeTimeH2O()
idx_ff = pd.date_range("2024-01-01", periods=300 + STEPS, freq="h")
exog_ff = pd.DataFrame({"price": np.r_[np.full(300, 2.5), np.full(STEPS, 2.56)]}, index=idx_ff)
forecaster_flat.fit(series=pd.Series(np.arange(300.0), index=idx_ff[:300], name="y"), exog=exog_ff.iloc[:300])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    forecaster_flat.predict(steps=STEPS, exog=exog_ff.iloc[300:])
    line_ff = inspect.currentframe().f_lineno - 1
found = [x for x in w if issubclass(x.category, FeatureOutOfRangeWarning)]
check("flat covariate warning points to the user's call to ForecasterFoundation.predict",
      len(found) == 1 and found[0].filename == __file__ and found[0].lineno == line_ff,
      f"{found[0].filename.split('/')[-1]}:{found[0].lineno}" if found else "no warning")

def helper_warnings(context_exog, exog, **kwargs):
    """
    Call the helper directly and return the FeatureOutOfRangeWarnings.
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_if_flat_known_covariates(context_exog=context_exog, exog=exog, model_name="TimeH2O",
                                       remedy="REMEDY", stacklevel=2, **kwargs)
    return [x for x in w if issubclass(x.category, FeatureOutOfRangeWarning)]


flat_ctx = {"y": pd.DataFrame({"p": np.full(100, 2.5), "q": np.sin(np.arange(100.0)), "only_past": 1.0})}
flat_fut = {"y": pd.DataFrame({"p": np.full(10, 2.56), "q": np.zeros(10), "only_future": 9.0})}
found = helper_warnings(flat_ctx, flat_fut)
check("helper: message with the model name, the remedy and only the flat column",
      len(found) == 1 and "TimeH2O" in str(found[0].message) and "REMEDY" in str(found[0].message)
      and "('y', 'p')" in str(found[0].message) and "'q'" not in str(found[0].message)
      and "only_" not in str(found[0].message), str(found[0].message)[:120] if found else "")
check("helper: None inputs give no warning",
      not helper_warnings(None, flat_fut) and not helper_warnings(flat_ctx, None)
      and not helper_warnings({"y": None}, flat_fut))
cat_ctx = {"y": pd.DataFrame({"c": pd.Categorical(["a"] * 100)})}
cat_fut = {"y": pd.DataFrame({"c": pd.Categorical(["b"] * 10)})}
check("helper: category column gives no warning", not helper_warnings(cat_ctx, cat_fut))
mixed_ctx = {"y": pd.DataFrame({"m": ["a"] * 100})}
mixed_fut = {"y": pd.DataFrame({"m": np.ones(10)})}
check("helper: string history and numeric future give no error and no warning",
      not helper_warnings(mixed_ctx, mixed_fut))
rng_thr = np.random.default_rng(3)
noisy_ctx = {"y": pd.DataFrame({"p": 2.5 + rng_thr.normal(0, 0.01, 1000)})}
jump_fut = {"y": pd.DataFrame({"p": np.full(10, 2.5 + 6 * noisy_ctx["y"]["p"].std())})}
check("helper: a jump of about 6 deviations warns with max_deviation=5 and not with 20",
      bool(helper_warnings(noisy_ctx, jump_fut, max_deviation=5.0)) and not helper_warnings(noisy_ctx, jump_fut))
object_bool_ctx = {"y": pd.DataFrame({"h": pd.Series([False] * 99 + [None], dtype=object)})}
object_bool_fut = {"y": pd.DataFrame({"h": [True] * 10})}
check("helper: object boolean history (False and a missing value) is checked as 0/1",
      bool(helper_warnings(object_bool_ctx, object_bool_fut)))
fm_tiny_flat = FoundationModel(TINY, device="cpu")
idx_t = pd.date_range("2024-01-01", periods=300 + STEPS, freq="h")
exog_t = pd.DataFrame({"price": np.r_[np.full(300, 2.5), np.full(STEPS, 2.56)]}, index=idx_t)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    fm_tiny_flat.predict(steps=STEPS, context=pd.Series(np.arange(300.0), index=idx_t[:300], name="y"),
                         context_exog=exog_t.iloc[:300], exog=exog_t.iloc[300:])
check("no flat-covariate warning with the tiny checkpoint (real backend)",
      not [x for x in w if issubclass(x.category, FeatureOutOfRangeWarning)])
fake_tiny = FakeTimeH2O(variate_attention=False)
fm_fake_tiny = FoundationModel(V1, model=fake_tiny)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    fm_fake_tiny.predict(steps=STEPS, context=pd.Series(np.arange(300.0), index=idx_t[:300], name="y"),
                         context_exog=exog_t.iloc[:300], exog=exog_t.iloc[300:])
check("no flat-covariate warning when config.variate_attention is False (fake)",
      not [x for x in w if issubclass(x.category, FeatureOutOfRangeWarning)])

# 5c. Boolean column that reaches the adapter as object
# ==============================================================================
fake_obj = FakeTimeH2O()
fm_obj = FoundationModel(V1, model=fake_obj)
idx_o = pd.date_range("2024-01-01", periods=200 + STEPS, freq="h")
exog_o = pd.DataFrame({"flag": np.arange(200 + STEPS) % 7 == 0}, index=idx_o)
context_exog_o = exog_o.iloc[:200].drop(idx_o[50])  # a missing timestamp: NaN after alignment
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fm_obj.predict(steps=STEPS, context=pd.Series(np.arange(200.0), index=idx_o[:200], name="y"),
                   context_exog=context_exog_o, exog=exog_o.iloc[200:])
row_o = fake_obj.last_future_covariates[0][0]
check("bool column with a missing context timestamp is passed as 0/1 float with NaN",
      pd.api.types.is_float_dtype(row_o) and row_o.isna().sum() == 1
      and set(row_o.dropna().unique()) == {0.0, 1.0}, f"{row_o.dtype}, NaN {row_o.isna().sum()}")

# 6. Tiny checkpoint with exog
# ==============================================================================
fm_tiny = FoundationModel(TINY, device="cpu")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    with_exog = fm_tiny.predict(steps=STEPS, context=y2, context_exog=exog_all.iloc[:900][["temp"]],
                                exog=exog_all.iloc[900:][["temp"]])
without = fm_tiny.predict(steps=STEPS, context=y2)
tiny_w = [x for x in w if "does not use future covariates" in str(x.message)]
check("tiny ignores exog with the warning of the backend",
      np.array_equal(with_exog["pred"].to_numpy(), without["pred"].to_numpy()) and len(tiny_w) == 1,
      f"{tiny_w[0].filename.split('/')[-1]}:{tiny_w[0].lineno}" if tiny_w else "")

# 7. Fake backend
# ==============================================================================
fake = FakeTimeH2O()
fm_fake = FoundationModel(V1, model=fake)
out = fm_fake.predict(steps=STEPS, context=series, context_exog=context_exog, exog=exog,
                      quantiles=[0.9, 0.1, 0.9])
check("fake: one call, levels deduplicated and sorted, columns in the requested order",
      len(fake.calls) == 1 and fake.last_quantile_levels == [0.1, 0.9]
      and np.allclose(out["q_0.9"], 0.9) and np.allclose(out["q_0.1"], 0.1))
check("fake: future covariates per series (2 rows, 1 row, None)",
      [None if r is None else len(r) for r in fake.last_future_covariates] == [2, 1, None])
point = fm_fake.predict(steps=STEPS, context=y)
check("fake: point forecast is the 0.5 column", fake.last_quantile_levels == [0.5]
      and np.allclose(point["pred"], 0.5))
fm_fake.adapter.set_params(predict_kwargs={"patch_ensemble": False, "clip_nonnegative": True})
fm_fake.predict(steps=STEPS, context=y)
check("fake: predict_kwargs forwarded", fake.last_kwargs["patch_ensemble"] is False
      and fake.last_kwargs["clip_nonnegative"] is True)
fm_fake.adapter.set_params(predict_kwargs={"unknown": 1})
passed, detail = raises(lambda: fm_fake.predict(steps=STEPS, context=y), TypeError, "unexpected keyword")
check("fake: unknown predict_kwargs key raises TypeError", passed, detail)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
