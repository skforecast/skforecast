"""
BUG-catboost-refit-categorical with real data: one (case, variant) per process.

Usage: python run_variant.py prepare            (downloads and caches the data)
       python run_variant.py <case> <variant>

Run it through driver.py: a variant with `n_jobs` greater than 1 can block
forever. Set `PYTHONPATH` to the repository root if the installed skforecast is
not this code (the first line printed is the package used). Everything is
written under `_out/` (or the folder in the environment variable `CB_WORK`):
the cache of the data and the predictions, in
preds_pandas<major><CB_TAG>/<case>__<variant>.pkl (`CB_TAG` is an optional
suffix, used for the runs with the installed 0.27.0).

Cases (`CASES`): bike_sharing with the periods of categorical-features.ipynb
(bikeS, 14 daily folds) and with one year of training (bikeL, 8 weekly
folds), vic_electricity (hourly, 8 weekly folds) and store_sales (10 series,
8 weekly folds). CatBoost with 100 iterations, except the case `default1000`.

Variants:
  ref, ref_s7     backtesting with refit and `n_jobs=1`, cloning the estimator
                  before every `fit` (a new CatBoost object in each fold). The
                  reference. `ref_s7` uses another seed (scale of the noise).
  manual          new forecaster in each fold, without the backtesting function
                  (single series; validates `ref`).
  bt1, bt2        backtesting with refit, `n_jobs=1` and `n_jobs=2`.
  prefit1         forecaster fitted before with the first half of the initial
                  window, then backtesting with refit and `n_jobs=1`.
  norefit         new forecaster, backtesting without refit.
  prefit_norefit  forecaster fitted before, backtesting without refit.
  oh_ref, oh_bt1  `ref` and `bt1` with `one_hot_max_size=255`.
  bt_auto         backtesting with refit and the default `n_jobs='auto'`
                  (CatBoost with `thread_count=1`).
"""
import os
import pickle
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import skforecast
from catboost import CatBoostRegressor
from sklearn.base import clone
from skforecast.datasets import fetch_dataset
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.model_selection import (
    TimeSeriesFold, backtesting_forecaster, backtesting_forecaster_multiseries,
)

warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("CB_WORK", HERE / "_out"))
CACHE = WORK / "cache"
OUT = WORK / f"preds_pandas{pd.__version__.split('.')[0]}{os.environ.get('CB_TAG', '')}"
THREADS = 4
GUIDE = ["holiday", "weather", "temp", "hum"]
CALENDAR = GUIDE + ["hour", "weekday"]


def cached(name, builder):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}_pandas{pd.__version__.split('.')[0]}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    obj = builder()
    pickle.dump(obj, open(path, "wb"))
    return obj


def _bike():
    data = fetch_dataset(name="bike_sharing", raw=True, verbose=False)
    data["date_time"] = pd.to_datetime(data["date_time"], format="%Y-%m-%d %H:%M:%S")
    data = data.set_index("date_time").sort_index().asfreq("h")
    data["holiday"] = data["holiday"].astype(int).astype(str)
    for col in ("weather", "hour", "weekday"):
        data[col] = data[col].astype(str)
    return data


def _vic():
    d = fetch_dataset(name="vic_electricity", raw=True, verbose=False)
    d["Time"] = pd.to_datetime(d["Time"], format="%Y-%m-%dT%H:%M:%SZ")
    d = d.set_index("Time").sort_index().asfreq("30min")
    h = d[["Demand", "Temperature"]].resample("h").mean()
    h["Holiday"] = d["Holiday"].resample("h").max().astype(int).astype(str)
    h["weekday"] = h.index.dayofweek.astype(str)
    h["hour"] = h.index.hour.astype(str)
    return h


def _store():
    d = fetch_dataset(name="store_sales", raw=True, verbose=False)
    d["date"] = pd.to_datetime(d["date"], format="%Y-%m-%d")
    d = d[d["item"] == 1]
    series = d.pivot(index="date", columns="store", values="sales").asfreq("D")
    series.columns = [f"store_{c}" for c in series.columns]
    return series.astype(float)


def data_bikeS(cols, n_test_days=14):
    # Periods of categorical-features.ipynb: train June and July 2012.
    end = pd.Timestamp("2012-08-01") + pd.Timedelta(days=n_test_days)
    d = cached("bike", _bike).loc["2012-06-01":end - pd.Timedelta(hours=1)]
    return d["users"], d[cols], len(d.loc[:"2012-07-31 23:59"])


def data_bikeL(cols):
    d = cached("bike", _bike).loc["2011-07-01":"2012-08-25 23:59"]
    return d["users"], d[cols], len(d.loc[:"2012-06-30 23:59"])


def data_vic(cols):
    h = cached("vic", _vic).loc["2013-01-01":"2014-02-25 23:59"]
    return h["Demand"], h[cols], len(h.loc[:"2013-12-31 23:59"])


def data_store(with_exog):
    series = cached("store", _store).loc["2015-01-01":"2017-02-25"]
    exog = None
    if with_exog:
        exog = pd.DataFrame(
            {
                "weekday": series.index.dayofweek.astype(str),
                "month": series.index.month.astype(str),
            },
            index=series.index,
        )
    return series, exog, len(series.loc[:"2016-12-31"])


IT100 = {"iterations": 100}
CASES = {
    # name: (kind, data builder, steps, lags, fixed_train_size, estimator kwargs, encoding)
    "bikeS_guide_fixed": ("single", lambda: data_bikeS(GUIDE), 24, 24, True, IT100, None),
    "bikeS_guide_expanding": ("single", lambda: data_bikeS(GUIDE), 24, 24, False, IT100, None),
    "bikeS_calendar_fixed": ("single", lambda: data_bikeS(CALENDAR), 24, 24, True, IT100, None),
    "bikeS_calendar_expanding": ("single", lambda: data_bikeS(CALENDAR), 24, 24, False, IT100, None),
    "bikeS_guide_fixed_default1000": ("single", lambda: data_bikeS(GUIDE, 4), 24, 24, True, {}, None),
    "bikeL_calendar_fixed": ("single", lambda: data_bikeL(CALENDAR), 168, 24, True, IT100, None),
    "vic_calendar_fixed": ("single", lambda: data_vic(["Temperature", "Holiday", "weekday", "hour"]), 168, 24, True, IT100, None),
    "store_ordinal_exog_fixed": ("multi", lambda: data_store(True), 7, 14, True, IT100, "ordinal"),
    "store_ordcat_exog_fixed": ("multi", lambda: data_store(True), 7, 14, True, IT100, "ordinal_category"),
    "store_ordcat_noexog_fixed": ("multi", lambda: data_store(False), 7, 14, True, IT100, "ordinal_category"),
}


def cb(seed=123, **kw):
    return CatBoostRegressor(
        **{"random_seed": seed, "verbose": 0, "allow_writing_files": False,
           "thread_count": THREADS, **kw},
    )


@contextmanager
def clone_before_fit(cls):
    """Reference: the estimator is a new object in every `fit`."""
    original = cls.fit

    def fit(self, *args, **kwargs):
        self.estimator = clone(self.estimator)
        return original(self, *args, **kwargs)

    cls.fit = fit
    try:
        yield
    finally:
        cls.fit = original


def to_wide(preds):
    if "level" in preds.columns:
        wide = preds.pivot(columns="level", values="pred")
        fold = preds.loc[~preds.index.duplicated(), "fold"]
    else:
        wide = preds[["pred"]].rename(columns={"pred": "y"})
        fold = preds["fold"]
    return wide.sort_index(), fold.sort_index()


def main(case, variant):
    kind, builder, steps, lags, fixed, est_kw, encoding = CASES[case]
    data, exog, initial = builder()
    seed = 7 if variant.endswith("_s7") else 123
    if variant.startswith("oh_"):
        est_kw = {**est_kw, "one_hot_max_size": 255}
    if variant == "bt_auto":
        est_kw = {**est_kw, "thread_count": 1}
    estimator = cb(seed=seed, **est_kw)
    refit = "norefit" not in variant
    n_jobs = {"bt2": 2, "bt_auto": "auto"}.get(variant, 1)
    cv = TimeSeriesFold(
        steps=steps, initial_train_size=initial, refit=refit,
        fixed_train_size=fixed if refit else True,
    )
    half = initial // 2
    sub = (lambda a, b: None) if exog is None else (lambda a, b: exog.iloc[a:b])
    print(skforecast.__version__, skforecast.__file__, "pandas", pd.__version__, flush=True)
    if variant == "bt_auto":
        from skforecast.model_selection._utils import select_n_jobs_backtesting
        probe = ForecasterRecursive(estimator=estimator, lags=lags)
        print(
            "select_n_jobs_backtesting:",
            select_n_jobs_backtesting(forecaster=probe, refit=True), flush=True,
        )
    t0 = time.perf_counter()

    if variant == "manual":
        out = []
        for start in range(initial, len(data), steps):
            first = start - initial if fixed else 0
            f = ForecasterRecursive(estimator=cb(seed=seed, **est_kw), lags=lags)
            f.fit(y=data.iloc[first:start], exog=sub(first, start))
            n = min(steps, len(data) - start)
            out.append(f.predict(steps=n, exog=sub(start, start + n)))
        wide, fold = pd.concat(out).to_frame("y"), None
    else:
        if kind == "single":
            cls = ForecasterRecursive
            forecaster = cls(estimator=estimator, lags=lags)
            if variant.startswith("prefit"):
                forecaster.fit(y=data.iloc[:half], exog=sub(0, half))

            def run():
                return backtesting_forecaster(
                    forecaster=forecaster, y=data, exog=exog, cv=cv,
                    metric="mean_absolute_error", n_jobs=n_jobs,
                    show_progress=False, verbose=False,
                )
        else:
            cls = ForecasterRecursiveMultiSeries
            forecaster = cls(estimator=estimator, lags=lags, encoding=encoding)
            if variant.startswith("prefit"):
                forecaster.fit(series=data.iloc[:half], exog=sub(0, half))

            def run():
                return backtesting_forecaster_multiseries(
                    forecaster=forecaster, series=data, exog=exog, cv=cv,
                    metric="mean_absolute_error", n_jobs=n_jobs,
                    show_progress=False, verbose=False,
                )
        if variant in ("ref", "ref_s7", "oh_ref"):
            with clone_before_fit(cls):
                _, preds = run()
        else:
            _, preds = run()
        wide, fold = to_wide(preds)

    OUT.mkdir(parents=True, exist_ok=True)
    truth = data.to_frame("y") if kind == "single" else data
    res = {
        "case": case, "variant": variant, "pandas": pd.__version__,
        "wide": wide, "fold": fold, "truth": truth.loc[wide.index, wide.columns],
        "seconds": round(time.perf_counter() - t0, 1),
    }
    pickle.dump(res, open(OUT / f"{case}__{variant}.pkl", "wb"))
    mae = float(np.mean(np.abs(wide.to_numpy() - res["truth"].to_numpy())))
    print(case, variant, "MAE", round(mae, 4), "seconds", res["seconds"], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        for name, builder in (("bike", _bike), ("vic", _vic), ("store", _store)):
            print(name, cached(name, builder).shape, flush=True)
    else:
        main(sys.argv[1], sys.argv[2])
