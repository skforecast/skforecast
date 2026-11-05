"""
Experiment 4: covariates.

1. Past-only covariates: directly to the backend (ignored with a warning by
   the v1 checkpoints) and through `FoundationModel` with the prototype
   (`IgnoredArgumentWarning`, not forwarded).
2. Numeric, boolean and string covariates on a synthetic series whose level
   depends on them: the backend encodes booleans and strings as categoricals
   (target encoding against the history of the series). Boolean passed as
   bool against the same column as float 0/1; labels against integer codes;
   a label not seen in the history; missing labels.
3. NaN in covariates: some NaN in the history or in the horizon, and a
   covariate whose whole history is NaN (dropped by the backend).
4. Infinite values in a covariate and in the series.
5. The case of mistralai/timestral#7: 32 daily Poisson(2) series, 1024
   points of context and 28 steps; a price that is flat at 2.50 over the
   context and moves to 2.56 in the horizon, a 0/1 indicator that is 0 over
   the context and 1 at two horizon steps, the same with their history set
   to NaN, and a price that already moves in the context.
6. A covariate with the tiny checkpoint (no cross-variate attention).

Usage: HF_HUB_OFFLINE=1 python exp4_covariates.py > exp4.log 2>&1
"""
import sys
import warnings
import numpy as np
import pandas as pd
from h2ofm import TimeH2O

sys.path.insert(0, ".")
from common import hourly_series, versions  # noqa: E402
from timeh2o_proto import register  # noqa: E402
from skforecast.foundation import FoundationModel  # noqa: E402

register()
print(versions(), flush=True)
V1, TINY = "h2oai/timeh2o-v1", "h2oai/timeh2o-v1-tiny"
model = TimeH2O.from_pretrained(V1, device="cpu")
STEPS = 24


def call(fn):
    """
    Run `fn` and return its result and the messages of the warnings raised.
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn()
    return out, [f"{x.category.__name__}: {str(x.message)[:110]}" for x in w]


# 1. Past-only covariates
# ==============================================================================
y = hourly_series(700)
past = np.sin(np.arange(700) / 7)
base = model.forecast(y, horizon=STEPS).quantiles
with_past, w = call(lambda: model.forecast(y, horizon=STEPS, past_covariates=past).quantiles)
print(f"1. backend, past_covariates: max abs diff vs none {np.max(np.abs(with_past - base)):.2e} | warnings {w}")

idx = pd.date_range("2024-01-01", periods=700 + STEPS, freq="h")
s = pd.Series(y, index=idx[:700], name="y")
frame = pd.DataFrame({"past": np.sin(np.arange(724) / 7), "fut": np.cos(np.arange(724) / 5)}, index=idx)
fm = FoundationModel(V1, context_length=4096, device="cpu")
fm.adapter._model = model
p_none = fm.predict(steps=STEPS, context=s, quantiles=[0.1, 0.5, 0.9])
p_past, w = call(lambda: fm.predict(steps=STEPS, context=s, context_exog=frame.iloc[:700][["past"]], quantiles=[0.1, 0.5, 0.9]))
print(f"1. FoundationModel, past-only column: max abs diff vs none "
      f"{np.max(np.abs(p_past.iloc[:, 1:].to_numpy() - p_none.iloc[:, 1:].to_numpy())):.2e} | warnings {w}")
p_both, w = call(lambda: fm.predict(steps=STEPS, context=s, context_exog=frame.iloc[:700], exog=frame.iloc[700:][["fut"]], quantiles=[0.1, 0.5, 0.9]))
p_fut, _ = call(lambda: fm.predict(steps=STEPS, context=s, context_exog=frame.iloc[:700][["fut"]], exog=frame.iloc[700:][["fut"]], quantiles=[0.1, 0.5, 0.9]))
print(f"1. FoundationModel, past-only + future column vs future column only: max abs diff "
      f"{np.max(np.abs(p_both.iloc[:, 1:].to_numpy() - p_fut.iloc[:, 1:].to_numpy())):.2e} | warnings {w}")

# 2. Numeric, boolean and string covariates
# ==============================================================================
rng = np.random.default_rng(1)
n = 1000
total = n + STEPS
flag = rng.random(total) < 0.1
flag[n + 3] = flag[n + 15] = True
labels = rng.choice(["low", "mid", "high"], total)
effect = {"low": -5.0, "mid": 0.0, "high": 5.0}
signal = 50 + 3 * np.sin(2 * np.pi * np.arange(total) / 24) + rng.normal(0, 0.5, total)
y_flag = signal + 8 * flag
y_label = signal + np.array([effect[x] for x in labels])
truth_flag, truth_label = y_flag[n:], y_label[n:]


def mae(forecast, truth):
    return float(np.mean(np.abs(forecast.point - truth)))


none_flag = model.forecast(y_flag[:n], horizon=STEPS)
as_bool = model.forecast(y_flag[:n], horizon=STEPS, future_covariates=flag)
as_float = model.forecast(y_flag[:n], horizon=STEPS, future_covariates=flag.astype(float))
print(f"2. indicator (level +8 when True, 10% of the steps): MAE none {mae(none_flag, truth_flag):.3f} "
      f"| bool (categorical) {mae(as_bool, truth_flag):.3f} | float 0/1 (numeric) {mae(as_float, truth_flag):.3f} "
      f"| max abs diff bool vs float {np.max(np.abs(as_bool.point - as_float.point)):.3f}")
none_label = model.forecast(y_label[:n], horizon=STEPS)
as_labels = model.forecast(y_label[:n], horizon=STEPS, future_covariates=[labels.astype(object)])
codes = pd.factorize(labels)[0].astype(float)
as_codes = model.forecast(y_label[:n], horizon=STEPS, future_covariates=[codes])
as_category = model.forecast(y_label[:n], horizon=STEPS, future_covariates=[pd.Series(labels, dtype="category")])
print(f"2. labels low/mid/high (-5/0/+5): MAE none {mae(none_label, truth_label):.3f} "
      f"| strings (categorical) {mae(as_labels, truth_label):.3f} | integer codes (numeric) "
      f"{mae(as_codes, truth_label):.3f} | category dtype equals strings "
      f"{np.array_equal(as_category.quantiles, as_labels.quantiles)}")
unseen = labels.astype(object).copy()
unseen[n:] = "never seen"
missing = labels.astype(object).copy()
missing[n + 2] = None
as_unseen = model.forecast(y_label[:n], horizon=STEPS, future_covariates=[unseen])
as_missing, w = call(lambda: model.forecast(y_label[:n], horizon=STEPS, future_covariates=[missing]))
print(f"2. label not seen in the history over the whole horizon: MAE {mae(as_unseen, truth_label):.3f} "
      f"| one missing label (None): finite {np.isfinite(as_missing.quantiles).all()} warnings {w}")

# 3. NaN in covariates
# ==============================================================================
temp = 15 + 8 * np.sin(2 * np.pi * np.arange(total) / 24)
y_temp = signal + 0.8 * temp
none_temp = model.forecast(y_temp[:n], horizon=STEPS)
full = model.forecast(y_temp[:n], horizon=STEPS, future_covariates=temp)
some_nan = temp.copy()
some_nan[rng.choice(n, 100, replace=False)] = np.nan
some_nan[n + 5:n + 8] = np.nan
with_nan = model.forecast(y_temp[:n], horizon=STEPS, future_covariates=some_nan)
history_nan = temp.copy()
history_nan[:n] = np.nan
with_history_nan = model.forecast(y_temp[:n], horizon=STEPS, future_covariates=history_nan)
truth_temp = y_temp[n:]
print(f"3. numeric covariate: MAE none {mae(none_temp, truth_temp):.3f} | full {mae(full, truth_temp):.3f} "
      f"| 100 NaN in the history and 3 in the horizon {mae(with_nan, truth_temp):.3f} "
      f"| whole history NaN {mae(with_history_nan, truth_temp):.3f} "
      f"(equal to no covariate: {np.array_equal(with_history_nan.quantiles, none_temp.quantiles)})")

# 4. Infinite values
# ==============================================================================
inf_cov = temp.copy()
inf_cov[10] = np.inf
for name, fn in {
    "covariate with inf": lambda: model.forecast(y_temp[:n], horizon=STEPS, future_covariates=inf_cov),
    "series with inf": lambda: model.forecast(np.r_[y_temp[:n - 1], np.inf], horizon=STEPS),
}.items():
    try:
        fn()
        print(f"4. {name}: no error")
    except ValueError as exc:
        print(f"4. {name}: ValueError: {exc}")
s_inf = pd.Series(np.r_[y_temp[:n - 1], np.inf], index=pd.date_range("2024-01-01", periods=n, freq="h"), name="y")
try:
    fm.predict(steps=STEPS, context=s_inf)
    print("4. FoundationModel, series with inf: no error")
except Exception as exc:
    print(f"4. FoundationModel, series with inf: {type(exc).__name__}: {str(exc)[:150]}")

# 5. Flat future covariates (mistralai/timestral#7)
# ==============================================================================
rng = np.random.default_rng(7)
T, H = 1024, 28
counts = rng.poisson(2, (32, T)).astype(float)
flat = np.full((32, T + H), 2.50)
flat[:, T + 5:] = 2.56
moving = np.full((32, T + H), 2.50)
moving[:, T - 300:] = 2.56
moving[:, T + 5:] = 2.60
flat_masked = flat.copy()
flat_masked[:, :T] = np.nan
indicator = np.zeros((32, T + H))
indicator[:, T + 10:T + 12] = 1.0
indicator_masked = indicator.copy()
indicator_masked[:, :T] = np.nan
levels = [0.1, 0.5, 0.9, 0.99]
cases = {
    "no covariate": None,
    "flat price": flat,
    "price that moves in the context": moving,
    "flat price, history set to NaN": flat_masked,
    "indicator 0/1 (float)": indicator,
    "indicator 0/1 (bool, categorical)": indicator.astype(bool),
    "indicator, history set to NaN": indicator_masked,
}
for name, cov in cases.items():
    fc = model.forecast_many(
        list(counts), horizon=H,
        future_covariates=None if cov is None else list(cov),
        quantile_levels=levels,
    )
    q = np.stack([f.quantiles for f in fc])
    print(f"5. {name:36s} median max {q[..., 1].max():8.2f} | q0.9 max {q[..., 2].max():8.2f} "
          f"| q0.99 max {q[..., 3].max():8.2f} | series with q0.99 > 100: "
          f"{int((q[..., 3].max(-1) > 100).sum())}/32", flush=True)

# 6. Tiny checkpoint with a covariate
# ==============================================================================
tiny = TimeH2O.from_pretrained(TINY, device="cpu")
tiny_none = tiny.forecast(y_temp[:n], horizon=STEPS)
tiny_cov, w = call(lambda: tiny.forecast(y_temp[:n], horizon=STEPS, future_covariates=temp))
print(f"6. tiny with a future covariate: equal to no covariate "
      f"{np.array_equal(tiny_cov.quantiles, tiny_none.quantiles)} | warnings {w}")
_, w = call(lambda: tiny.forecast(y_temp[:n], horizon=STEPS, future_covariates=temp))
print(f"6. tiny, second call with a covariate: warnings {w}")
