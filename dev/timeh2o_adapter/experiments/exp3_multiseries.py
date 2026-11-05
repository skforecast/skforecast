"""
Experiment 3: the prototype adapter through `FoundationModel`.

1. Heterogeneous batch: three series of different lengths (one with NaN) and
   different exog columns (numeric and categorical, numeric only, none),
   predicted together against each series alone. Number of calls to
   `forecast_many` with `supports_heterogeneous_covariates=True` (one call)
   and `False` (one call per signature).
2. Quantiles: levels outside the trained grid, duplicated levels, 0 and 1,
   and `predict_interval` through `ForecasterFoundation`.
3. Cost with and without covariates (numeric, numeric and categorical), with
   the default `patch_ensemble=True` and with `False`, and with the tiny
   checkpoint (which ignores covariates).

Usage: HF_HUB_OFFLINE=1 python exp3_multiseries.py > exp3.log 2>&1
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from common import hourly_series, peak_rss_gb, versions  # noqa: E402
from timeh2o_proto import TimeH2OAdapter, register  # noqa: E402
from skforecast.foundation import FoundationModel, ForecasterFoundation  # noqa: E402

register()
print(versions(), flush=True)
STEPS = 24
V1, TINY = "h2oai/timeh2o-v1", "h2oai/timeh2o-v1-tiny"


def count_calls(model: FoundationModel) -> list:
    """
    Wrap `forecast_many` of the loaded backend to record the batch sizes.
    """
    model.adapter._load_model()
    backend = model.adapter._model
    original = backend.forecast_many
    calls = []

    def wrapper(contexts, *args, **kwargs):
        calls.append(len(contexts))
        return original(contexts, *args, **kwargs)

    backend.forecast_many = wrapper
    return calls


# 1. Heterogeneous batch
# ==============================================================================
rng = np.random.default_rng(0)
lengths = {"s1": 900, "s2": 400, "s3": 700}
series, context_exog, exog = {}, {}, {}
for i, (name, n) in enumerate(lengths.items()):
    idx = pd.date_range("2024-01-01", periods=n + STEPS, freq="h")[-(n + STEPS):]
    idx = idx + pd.Timedelta(hours=int(37 * i))  # different end dates
    temp = 15 + 8 * np.sin(2 * np.pi * np.arange(n + STEPS) / 24) + rng.normal(0, 1, n + STEPS)
    kind = np.where(np.arange(n + STEPS) % 168 < 120, "work", "weekend")
    y = hourly_series(n, seed=i) + 0.5 * temp[:n] + 3 * (kind[:n] == "weekend")
    if name == "s2":
        y[100:130] = np.nan
    series[name] = pd.Series(y, index=idx[:n], name=name)
    frame = pd.DataFrame({"temp": temp, "kind": kind}, index=idx)
    if name == "s1":
        cols = ["temp", "kind"]
    elif name == "s2":
        cols = ["temp"]
    else:
        cols = None
    context_exog[name] = frame.iloc[:n][cols] if cols else None
    exog[name] = frame.iloc[n:][cols] if cols else None

model = FoundationModel(V1, context_length=4096, device="cpu")
calls = count_calls(model)
batch = model.predict(
    steps=STEPS, context=series, context_exog=context_exog, exog=exog,
    quantiles=[0.1, 0.5, 0.9],
)
print(f"1. heterogeneous batch: calls to forecast_many {calls}")
for name in series:
    alone = model.predict(
        steps=STEPS, context={name: series[name]},
        context_exog={name: context_exog[name]}, exog={name: exog[name]},
        quantiles=[0.1, 0.5, 0.9],
    )
    b = batch[batch["level"] == name]
    diff = np.max(np.abs(b[["q_0.1", "q_0.5", "q_0.9"]].to_numpy()
                         - alone[["q_0.1", "q_0.5", "q_0.9"]].to_numpy()))
    same_index = b.index.equals(alone.index)
    print(f"1. {name} (n={lengths[name]}, exog {list(exog[name].columns) if exog[name] is not None else None}): "
          f"batch vs alone max abs diff {diff:.2e} | same index {same_index}")


class GroupedTimeH2OAdapter(TimeH2OAdapter):
    """
    The prototype with `supports_heterogeneous_covariates=False`.
    """
    SUPPORTED_QUANTILES = None
    allow_exog = True
    supports_past_only_covariates = False
    supports_categorical_covariates = True
    supports_heterogeneous_covariates = False
    requires_equal_length_context = False
    supports_nan_in_series = True
    requires_hf_auth = True
    requires_provider_auth = False
    weights_repo_id = None
    weights_in_hf_cache = True
    backend_package = "h2ofm>=0.43"
    default_model_id = V1


grouped = FoundationModel(V1, context_length=4096, device="cpu")
grouped.adapter = GroupedTimeH2OAdapter(V1, context_length=4096, device="cpu",
                                        model=model.adapter._model)
calls_grouped = count_calls(grouped)
batch_grouped = grouped.predict(
    steps=STEPS, context=series, context_exog=context_exog, exog=exog,
    quantiles=[0.1, 0.5, 0.9],
)
diff = np.max(np.abs(batch_grouped[["q_0.1", "q_0.5", "q_0.9"]].to_numpy()
                     - batch[["q_0.1", "q_0.5", "q_0.9"]].to_numpy()))
print(f"1. grouped by signature: calls {calls_grouped} | max abs diff vs one call {diff:.2e}")

# 2. Quantiles
# ==============================================================================
y = pd.Series(hourly_series(600), index=pd.date_range("2024-01-01", periods=600, freq="h"), name="y")
q = model.predict(steps=STEPS, context=y, quantiles=[0.33, 0.02, 0.98, 0.5])
print(f"2. levels [0.33, 0.02, 0.98, 0.5]: columns {list(q.columns)} | "
      f"q_0.02 < q_0.33 < q_0.5 < q_0.98 at every step: "
      f"{bool(((q['q_0.02'] < q['q_0.33']) & (q['q_0.33'] < q['q_0.5']) & (q['q_0.5'] < q['q_0.98'])).all())}")
levels_seen = []
original_forecast_many = model.adapter._model.forecast_many


def record_levels(*args, **kwargs):
    levels_seen.append(kwargs.get("quantile_levels"))
    return original_forecast_many(*args, **kwargs)


model.adapter._model.forecast_many = record_levels
q_dup = model.predict(steps=STEPS, context=y, quantiles=[0.5, 0.1, 0.5])
model.adapter._model.forecast_many = original_forecast_many
print(f"2. duplicated levels [0.5, 0.1, 0.5]: backend receives {levels_seen[-1]} | "
      f"FoundationModel columns {list(q_dup.columns)} | q_0.5 equals the median "
      f"{np.allclose(q_dup['q_0.5'].to_numpy(), q['q_0.5'].to_numpy())}")
try:
    model.adapter._model.forecast(y.to_numpy(), horizon=STEPS, quantile_levels=[0.5, 0.5])
except ValueError as exc:
    print(f"2. backend with duplicated levels: ValueError: {exc}")
for levels in ([0.0, 0.5], [0.5, 1.0]):
    try:
        model.predict(steps=STEPS, context=y, quantiles=levels)
        print(f"2. quantiles {levels}: no error")
    except ValueError as exc:
        print(f"2. quantiles {levels}: ValueError: {exc}")
forecaster = ForecasterFoundation(estimator=model)
forecaster.fit(series=y)
for interval in ([0.1, 0.9], 0.9, 0.98, [0.01, 0.99]):
    pi = forecaster.predict_interval(steps=STEPS, interval=interval)
    ok = bool((pi["lower_bound"] <= pi["pred"]).all() and (pi["pred"] <= pi["upper_bound"]).all())
    print(f"2. predict_interval(interval={interval}): columns {list(pi.columns)} | ordered {ok}")
point = forecaster.predict(steps=STEPS)
print(f"2. predict() equals the q_0.5 column: "
      f"{np.allclose(point['pred'].to_numpy(), q['q_0.5'].to_numpy())}")

# 3. Cost with and without covariates
# ==============================================================================
CTX = 512


def make_inputs(n_series: int, n_cov: int):
    """
    `n_series` hourly series of `CTX` points with 0, 1 (numeric) or 2
    (numeric and categorical) future covariates.
    """
    idx = pd.date_range("2024-01-01", periods=CTX + STEPS, freq="h")
    ctx, ctx_exog, fut_exog = {}, {}, {}
    for i in range(n_series):
        name = f"s{i}"
        ctx[name] = pd.Series(hourly_series(CTX, seed=i), index=idx[:CTX], name=name)
        frame = pd.DataFrame({
            "temp": 15 + 8 * np.sin(2 * np.pi * np.arange(CTX + STEPS) / 24),
            "kind": np.where(np.arange(CTX + STEPS) % 168 < 120, "work", "weekend"),
        }, index=idx)
        cols = ["temp", "kind"][:n_cov]
        ctx_exog[name] = frame.iloc[:CTX][cols] if cols else None
        fut_exog[name] = frame.iloc[CTX:][cols] if cols else None
    return ctx, (ctx_exog if n_cov else None), (fut_exog if n_cov else None)


cases = [
    (V1, True, 20), (V1, False, 20), (V1, False, 100), (TINY, None, 100),
]
for model_id, patch_ensemble, n_series in cases:
    kwargs = {} if patch_ensemble is None else {"predict_kwargs": {"patch_ensemble": patch_ensemble}}
    m = FoundationModel(model_id, context_length=CTX, device="cpu", **kwargs)
    for n_cov in (0, 1, 2):
        ctx, ctx_exog, fut_exog = make_inputs(n_series, n_cov)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m.predict(steps=STEPS, context=ctx, context_exog=ctx_exog, exog=fut_exog)  # warm-up
            start = time.perf_counter()
            m.predict(steps=STEPS, context=ctx, context_exog=ctx_exog, exog=fut_exog)
            seconds = time.perf_counter() - start
        backend_warnings = sorted({str(x.message)[:70] for x in w})
        print(f"3. {model_id} patch_ensemble={patch_ensemble} {n_series} series x {CTX}, "
              f"{n_cov} covariates: {seconds:.2f} s | peak RSS {peak_rss_gb():.2f} GB"
              f"{' | warnings ' + str(backend_warnings) if backend_warnings else ''}", flush=True)
