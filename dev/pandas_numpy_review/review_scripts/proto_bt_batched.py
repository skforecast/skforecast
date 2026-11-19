"""
Is the "ceiling" of the study (`scripts/ceiling_bt_ms.py`) a ceiling? Without
refit all the folds use the same model, so they can be predicted in ONE
recursive pass: the folds are stacked as extra rows (n_folds * n_series rows per
step) and the estimator is called `steps` times instead of `n_folds * steps`.

Same data as `ceiling_bt_ms.py`. Compares, for the numpy loop only (no
validation, no output, as the ceiling): fold by fold (the ceiling) against all
folds in one pass, with 1 thread and with the threads of the estimator.

Usage: python proto_bt_batched.py N_SERIES EXOG(none|float|mixed) EST(ridge|lgbm) [N_FOLDS] [STEPS]
"""
import _env  # noqa: F401
import sys
import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from skforecast.recursive import ForecasterRecursiveMultiSeries

warnings.simplefilter("ignore")
n_series = int(sys.argv[1]); exog_kind = sys.argv[2]; est = sys.argv[3]
n_folds = int(sys.argv[4]) if len(sys.argv) > 4 else 50
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 24
n_rows = 1500; lags = 24
rng = np.random.default_rng(1)
idx = pd.date_range("2020-01-01", periods=n_rows, freq="h")
names = [f"series_{i:04d}" for i in range(n_series)]
values = np.cumsum(rng.normal(size=(n_rows, n_series)), axis=0) + 100
series = {k: pd.Series(values[:, j], index=idx, name=k) for j, k in enumerate(names)}
DOW = np.array(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])


def make_exog(j):
    df = pd.DataFrame({"temp": rng.normal(size=n_rows), "promo": rng.normal(size=n_rows)}, index=idx)
    if exog_kind == "mixed":
        df["hour"] = idx.hour.to_numpy().astype("int64")
        df["dow"] = pd.Categorical(DOW[idx.dayofweek.to_numpy()], categories=list(DOW))
        df["shop_type"] = np.where(rng.random(n_rows) < 0.5, "big", "small")
    return df


exog = None if exog_kind == "none" else {k: make_exog(j) for j, k in enumerate(names)}
initial = n_rows - n_folds * steps


class Window:
    def __init__(self, values):
        self.values = values

    def to_numpy(self):
        return self.values


def measure(n_threads):
    if est == "ridge":
        estimator = Ridge()
    else:
        estimator = LGBMRegressor(n_estimators=50, verbose=-1, n_jobs=n_threads, random_state=1)
    f = ForecasterRecursiveMultiSeries(estimator, lags=lags, encoding="ordinal")
    f.fit(series={k: v.iloc[:initial] for k, v in series.items()},
          exog=None if exog is None else {k: v.iloc[:initial] for k, v in exog.items()})
    ex3 = None
    if exog is not None:
        cat = f.categorical_features_names_in_ or []
        ex3 = np.empty((n_rows, n_series, len(f.X_train_exog_names_out_)))
        for j, k in enumerate(names):
            df = exog[k][f.exog_names_in_]
            if cat:
                df = df.copy(deep=False)
                df[cat] = f.categorical_encoder.transform(df[cat])
            ex3[:, j, :] = df.to_numpy(dtype=float)

    def fold_by_fold():
        out = np.empty((n_folds, steps, n_series))
        for i in range(n_folds):
            t0 = initial + i * steps
            ed = None if ex3 is None else {s + 1: ex3[t0 + s] for s in range(steps)}
            out[i] = f._recursive_predict(
                steps=steps, levels=names, last_window=Window(values[t0 - lags:t0]),
                exog_values_dict=ed,
            )
        return out

    starts = initial + steps * np.arange(n_folds)
    levels_all = names * n_folds

    def all_folds_one_pass():
        # (lags, n_folds * n_series): the windows of all folds side by side.
        lw = np.concatenate([values[t0 - lags:t0] for t0 in starts], axis=1)
        ed = None
        if ex3 is not None:
            ed = {
                s + 1: ex3[starts + s].reshape(n_folds * n_series, -1)
                for s in range(steps)
            }
        out = f._recursive_predict(
            steps=steps, levels=levels_all, last_window=Window(lw), exog_values_dict=ed
        )
        return out.reshape(steps, n_folds, n_series).transpose(1, 0, 2)

    res = {}
    for name, fn in (("fold by fold (ceiling)", fold_by_fold), ("all folds, one pass", all_folds_one_pass)):
        best = np.inf
        for _ in range(3):
            t = time.perf_counter(); out = fn(); best = min(best, time.perf_counter() - t)
        res[name] = (best, out)
    a, b = res["fold by fold (ceiling)"], res["all folds, one pass"]
    print(
        f"  threads={n_threads}: fold by fold {a[0] / n_folds * 1000:7.3f} ms per fold | "
        f"one pass {b[0] / n_folds * 1000:7.3f} ms per fold ({a[0] / b[0]:.1f}x) | "
        f"identical={np.array_equal(a[1], b[1])} max abs diff={np.abs(a[1] - b[1]).max():.3g}"
    )


print(f"n_series={n_series} exog={exog_kind} est={est} folds={n_folds} steps={steps}")
measure(1)
if est == "lgbm":
    measure(4)
print(_env.versions())
