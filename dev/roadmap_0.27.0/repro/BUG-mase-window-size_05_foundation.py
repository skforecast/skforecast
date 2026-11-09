"""
BUG-mase-window-size: scaled metrics of `backtesting_foundation` with a real
model (Chronos-2 small, CPU), compared with the reference per forecast origin:
each error is scaled with the one-step naive forecast over the values before
the end of the last window of its fold, `initial_train_size + fold * steps`
(the `gap` is not known when predicting).

Run with PYTHONPATH pointing to the code to test (0.27.x or the fix). With
0.27.x, the reference does not match: the scale is the union of the training
sets of all the folds (`refit` is forced to True).
"""
import platform
import warnings
import numpy as np
import pandas as pd
import skforecast
from skforecast.datasets import load_demo_dataset
from skforecast.foundation import FoundationModel, ForecasterFoundation
from skforecast.model_selection import TimeSeriesFold, backtesting_foundation
from skforecast.metrics import mean_absolute_scaled_error, root_mean_squared_scaled_error

warnings.filterwarnings("ignore")
import torch  # noqa: E402

print(
    f"Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, "
    f"torch {torch.__version__}, skforecast from {skforecast.__file__}"
)

y = load_demo_dataset(verbose=False)
N_TRAIN, STEPS = len(y) - 36, 12
received = []


def custom_metric(y_true, y_pred, y_train):
    received.append(len(y_train))
    return 0.0


METRICS = ["mean_absolute_error", mean_absolute_scaled_error, root_mean_squared_scaled_error]
forecaster = ForecasterFoundation(
    estimator=FoundationModel(model_id="autogluon/chronos-2-small", context_length=2048, device_map="cpu")
)


def reference(values, predictions, pred_column="pred"):
    predictions = predictions.loc[predictions["y_true"].notna()]
    errors = (predictions["y_true"] - predictions[pred_column]).to_numpy()
    origins = N_TRAIN + STEPS * predictions["fold"].to_numpy()
    scale_abs = np.array([np.nanmean(np.abs(np.diff(values[:o]))) for o in origins])
    scale_sq = np.array([np.nanmean(np.diff(values[:o]) ** 2) for o in origins])
    q_abs, q_sq = np.abs(errors) / scale_abs, errors ** 2 / scale_sq
    return q_abs, q_sq


def row(name, metric, q_abs, q_sq):
    mase, rmsse = metric["mean_absolute_scaled_error"], metric["root_mean_squared_scaled_error"]
    ok = np.isclose(mase, q_abs.mean()) and np.isclose(rmsse, np.sqrt(q_sq.mean()))
    print(f"| {name} | {metric['mean_absolute_error']:.4f} | {mase:.4f} | {q_abs.mean():.4f} | "
          f"{rmsse:.4f} | {np.sqrt(q_sq.mean()):.4f} | {'yes' if ok else 'NO'} |")


print("\n| Case | MAE | MASE | MASE ref. | RMSSE | RMSSE ref. | Equal |")
print("|---|---|---|---|---|---|---|")

# Single series ---------------------------------------------------------------
for name, kwargs in [("one series", {}), ("one series, gap=2", {"gap": 2})]:
    metric, predictions = backtesting_foundation(
        forecaster=forecaster, series=y,
        cv=TimeSeriesFold(steps=STEPS, initial_train_size=N_TRAIN, **kwargs),
        metric=METRICS + [custom_metric], show_progress=False, suppress_warnings=True,
    )
    row(name, metric.iloc[0], *reference(y.to_numpy(), predictions))

metric, predictions = backtesting_foundation(
    forecaster=forecaster, series=y, cv=TimeSeriesFold(steps=STEPS, initial_train_size=N_TRAIN),
    metric=METRICS, quantiles=[0.1, 0.5, 0.9], show_progress=False, suppress_warnings=True,
)
row("one series, quantiles (q_0.5)", metric.iloc[0], *reference(y.to_numpy(), predictions, "q_0.5"))
print(f"\nLength of `y_train` received by the custom metric: {sorted(set(received))} (N_TRAIN={N_TRAIN})")

# Multiple series: s3 starts 30 dates later ------------------------------------
rng = np.random.default_rng(1)
s1 = y.rename("s1")
s2 = (2 * y + rng.normal(scale=0.05, size=len(y))).rename("s2")
s3 = (1.5 * y).rename("s3").iloc[30:]
wide = pd.concat([s1, s2, s3], axis=1)
as_dict = {"s1": s1, "s2": s2, "s3": s3}

print("\n| Case | MAE | MASE | MASE ref. | RMSSE | RMSSE ref. | Equal |")
print("|---|---|---|---|---|---|---|")
results = {}
for name, data in [("wide", wide), ("dict", as_dict)]:
    metric, predictions = backtesting_foundation(
        forecaster=forecaster, series=data, cv=TimeSeriesFold(steps=STEPS, initial_train_size=N_TRAIN),
        metric=METRICS, show_progress=False, suppress_warnings=True,
    )
    metric = metric.set_index("levels")
    results[name] = metric
    all_abs, all_sq = [], []
    for level in ["s1", "s2", "s3"]:
        q_abs, q_sq = reference(wide[level].to_numpy(), predictions.loc[predictions["level"] == level])
        all_abs.append(q_abs)
        all_sq.append(q_sq)
        row(f"{name}, {level}", metric.loc[level], q_abs, q_sq)
    row(f"{name}, pooling", metric.loc["pooling"], np.concatenate(all_abs), np.concatenate(all_sq))
    print(f"| {name}, weighted_average | | {metric.loc['weighted_average', 'mean_absolute_scaled_error']:.4f} | | | | |")

same = np.allclose(
    results["wide"][["mean_absolute_scaled_error", "root_mean_squared_scaled_error"]].to_numpy(dtype=float),
    results["dict"][["mean_absolute_scaled_error", "root_mean_squared_scaled_error"]].to_numpy(dtype=float),
)
print(f"\nWide DataFrame and dict give the same scaled metrics: {same}")
