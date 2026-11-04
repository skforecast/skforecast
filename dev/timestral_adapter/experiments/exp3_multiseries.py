"""
Multi-series and covariate checks of the Timestral prototype through
FoundationModel:

1. Heterogeneous series (different lengths, exog columns, NaN) and a
   categorical covariate: predictions in one call equal the predictions of
   each series alone.
2. Cost of a batch with covariates as the number of series grows (variate
   attention is quadratic in series x (1 + covariates), mistralai/timestral#6).
3. Default symmetric averaging with a sign-fixed covariate (mistralai/timestral#7).
"""
import sys
import time
import tracemalloc
import warnings
import numpy as np
import pandas as pd
import resource

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from timestral_proto import register
from skforecast.foundation import FoundationModel

register()
rng = np.random.default_rng(1)
idx = pd.date_range("2024-01-01", periods=900, freq="h")
t = np.arange(900)


def make_series(n, phase):
    return pd.Series(
        100 + 20 * np.sin(2 * np.pi * (t[-n:] + phase) / 24) + rng.standard_normal(n).cumsum(),
        index=idx[-n:],
    )


# 1. Heterogeneous input
series = {"a": make_series(900, 0), "b": make_series(400, 5), "c": make_series(700, 9)}
series["c"].iloc[100:120] = np.nan
full_idx = pd.date_range(idx[0], periods=900 + 24, freq="h")
temp = pd.Series(15 + 8 * np.sin(2 * np.pi * np.arange(924) / 24), index=full_idx, name="temp")
dow = pd.Series(full_idx.day_name(), index=full_idx, name="dow").astype("category")
exog = {
    "a": pd.concat([temp, dow], axis=1),
    "b": pd.concat([temp, dow], axis=1),
    "c": temp.to_frame(),
}
hist = {k: v.loc[: idx[-1]] for k, v in exog.items()}
fut = {k: v.loc[idx[-1]:].iloc[1:] for k, v in exog.items()}

model = FoundationModel("mistralai/Timestral-28M", device="cpu", context_length=2048)
model.fit(series=series, exog=hist)
together = model.predict(steps=24, exog=fut, quantiles=[0.1, 0.5, 0.9])
max_diff = 0.0
for name in series:
    alone = FoundationModel("mistralai/Timestral-28M", device="cpu", context_length=2048)
    alone.adapter._model = model.adapter._model
    alone.fit(series=series[name], exog=hist[name])
    p = alone.predict(steps=24, exog=fut[name], quantiles=[0.1, 0.5, 0.9])
    q = together.loc[together["level"] == name, ["q_0.1", "q_0.5", "q_0.9"]].to_numpy()
    max_diff = max(max_diff, np.abs(q - p[["q_0.1", "q_0.5", "q_0.9"]].to_numpy()).max())
print(f"1. heterogeneous batch vs alone, max abs diff: {max_diff:.2e}")
print(together.groupby("level")[["q_0.5"]].mean().round(2).to_dict())

# Unsupported quantile
try:
    model.predict(steps=24, exog=fut, quantiles=[0.33])
except ValueError as e:
    print("   quantile 0.33 ->", str(e)[:90])

# 2. Batch cost with 2 covariates (numeric + categorical) as series grow
fc = model.adapter._model
for n_series in [10, 100, 200, 400]:
    ctx = np.stack([make_series(512, i).to_numpy() for i in range(n_series)])[:, None, :]
    known = np.stack([np.stack([temp.to_numpy()[:536], (np.arange(536) % 7).astype(float)])] * n_series)
    t0 = time.perf_counter()
    fc.forecast(ctx, prediction_length=24, known_covariates=known[:, :1],
                categorical_known_covariates=known[:, 1:])
    elapsed = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"2. {n_series:4d} series x (1 target + 2 cov) x 512: {elapsed:6.2f}s  peak RSS {rss:,.0f} MB")

# 3. Symmetric averaging with a sign-fixed, piecewise-constant known covariate
y = pd.Series(rng.poisson(2, 1024).astype(float), index=pd.date_range("2020-01-01", periods=1024, freq="D"))
fidx = pd.date_range(y.index[0], periods=1024 + 28, freq="D")
price = pd.Series(2.50, index=fidx, name="price")
price.iloc[1024 + 5:] = 2.56
for sym in [True, False]:
    m = FoundationModel("mistralai/Timestral-209M", device="cpu",
                        predict_kwargs={"symmetric_averaging": sym})
    m.fit(series=y, exog=price.loc[: y.index[-1]])
    p = m.predict(steps=28, exog=price.loc[y.index[-1]:].iloc[1:], quantiles=[0.5, 0.99])
    print(f"3. symmetric_averaging={sym}: median max {p['q_0.5'].max():.2f}, q99 max {p['q_0.99'].max():.2f}")
