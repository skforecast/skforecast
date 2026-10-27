"""
BUG-search-short-series, experiment 5: prototype of option c2 with the real
skforecast `fit` and `predict` (no changes in the library), fold by fold.

c2 = train as today (only series with more than `window_size` values) and
predict every series with some history, padding the last window with NaN. A
series that is not in the training set is an unknown level for the forecaster.

Variants:
  drop      : today's rule (validated against backtesting_forecaster_multiseries).
  c2_pooled : c2 as skforecast does it today with unknown levels: the series is
              scaled with the `_unknown_level` transformer (all series pooled).
  c2_own    : c2 where every series is scaled with its own values, even if it
              does not add training rows (series scaled outside skforecast and
              `transformer_series=None`).

Parts (first argument):
  synthetic : data of experiments 2 to 4 (5 long and 15 short monthly series),
              validation, 4 estimators, encodings, window_features, intervals
              and differentiation.
  m5        : M5 store CA_1, 40 products on sale since 2011 and 60 launched in
              2014, backtesting over the second half of 2014.

Usage: python BUG-search-short-series_05_c2_prototype.py synthetic|m5 [m5.parquet]
The M5 parquet is the output of `fetch_dataset("m5")` saved with `to_parquet`
(reset_index), about 116 MB to download.
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursiveMultiSeries

warnings.simplefilter("ignore")
pd.set_option("display.width", 200)

ESTIMATORS = {
    "HGBR": lambda: HistGradientBoostingRegressor(max_iter=200, random_state=123),
    "LightGBM": lambda: LGBMRegressor(n_estimators=200, random_state=123, verbose=-1),
    "XGBoost": lambda: XGBRegressor(n_estimators=200, random_state=123),
    "CatBoost": lambda: CatBoostRegressor(
        iterations=200, random_state=123, verbose=0, allow_writing_files=False
    ),
}

# ---------------------------------------------------------------------------
# RollingFeatures.transform (predict path) ignores NaN and `min_periods`, while
# transform_batch (train path) uses pandas rolling with `min_periods`. This
# patch makes the predict path return NaN when the window has fewer observed
# values than `min_periods`, as in training.
# ---------------------------------------------------------------------------
_original_transform = RollingFeatures.transform


def _transform_consistent(self, X):
    out = _original_transform(self, X)
    X2 = X[:, np.newaxis] if X.ndim == 1 else X
    for j, (w, mp) in enumerate(zip(self.window_sizes, self.min_periods)):
        n_obs = (~np.isnan(X2[-w:])).sum(axis=0)
        if out.ndim == 1:
            if n_obs[0] < mp:
                out[j] = np.nan
        else:
            out[n_obs < mp, j] = np.nan
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def synthetic_data(seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=96, freq="MS")
    t = np.arange(96)

    def make(start, level, noise):
        y = level * (1 + 0.15 * np.sin(2 * np.pi * t / 12)) + rng.normal(0, noise * level, 96)
        return pd.Series(y, index=idx).loc[start:]

    series = {}
    for i in range(5):
        series[f"long_{i}"] = make("2016-01-01", rng.uniform(50, 150), 0.05)
    starts = pd.date_range("2019-01-01", "2021-01-01", freq="MS")
    for i in range(15):
        series[f"new_{i:02d}"] = make(rng.choice(starts), rng.uniform(50, 150), 0.15)
    series = {k: v.asfreq("MS") for k, v in series.items()}
    cv = dict(steps=12, initial_train_size="2021-06-01", refit=True)
    return series, idx, cv


def m5_data(path, seed=123):
    df = pd.read_parquet(
        path, columns=["id", "datetime", "sales", "sell_price", "store_id"],
        filters=[("store_id", "==", "CA_1")]
    )
    df["id"] = df["id"].astype(str)
    df = df[df["datetime"] <= "2014-12-31"]
    launch = df[df["sell_price"].notna()].groupby("id")["datetime"].min()
    rng = np.random.default_rng(seed)
    old = launch[launch <= "2011-02-01"].index.to_numpy()
    new = launch[(launch >= "2014-01-01") & (launch <= "2014-11-01")].index.to_numpy()
    ids = list(rng.choice(old, 40, replace=False)) + list(rng.choice(new, 60, replace=False))
    df = df[df["id"].isin(ids)]
    idx = pd.date_range(df["datetime"].min(), "2014-12-31", freq="D")
    series = {}
    for k, g in df.groupby("id"):
        s = g.set_index("datetime")["sales"].astype(float).reindex(idx)
        series[k] = s.loc[launch[k]:].asfreq("D")
    cv = dict(steps=28, initial_train_size="2014-06-30", refit=True)
    return series, idx, cv


# ---------------------------------------------------------------------------
# Prototype backtesting
# ---------------------------------------------------------------------------
def get_folds(series, idx, cv):
    Y = pd.DataFrame(series).reindex(idx)
    return TimeSeriesFold(**cv, window_size=1, verbose=False).split(X=Y, as_pandas=True)


def evaluation_set(series, idx, folds):
    """(level, fold, date) with an observed value and some history before the fold."""
    rows = []
    for _, f in folds.iterrows():
        tr_start, tr_end = idx[f["train_start"]], idx[f["train_end"] - 1]
        te = idx[f["test_start"]:f["test_end"]]
        for k, s in series.items():
            hist = s.loc[tr_start:tr_end].dropna()
            if hist.empty:
                continue
            y = s.reindex(te).dropna()
            for d, v in y.items():
                rows.append((k, f["fold"], d, v, hist.iloc[-1], hist.iloc[-28:].mean()))
    return pd.DataFrame(
        rows, columns=["level", "fold", "date", "y", "naive_last", "naive_mean28"]
    )


def proto_backtest(series, idx, folds, make_forecaster, variant, interval=None):
    scaling = "own" if variant == "c2_own" else "skforecast"
    recs = []
    for _, f in folds.iterrows():
        tr_start, tr_end = idx[f["train_start"]], idx[f["train_end"] - 1]
        te = idx[f["test_start"]:f["test_end"]]
        forecaster = make_forecaster(scaling)
        ws = forecaster.window_size
        train = {k: s.loc[tr_start:tr_end].dropna() for k, s in series.items()}
        train = {k: v for k, v in train.items() if len(v) > 0}
        if scaling == "own":
            stats = {k: (v.mean(), v.std(ddof=0) if v.std(ddof=0) > 0 else 1.0)
                     for k, v in train.items()}
            train = {k: (v - stats[k][0]) / stats[k][1] for k, v in train.items()}
        in_train = {k: v for k, v in train.items() if len(v) > ws}
        if not in_train:
            continue
        forecaster.fit(series=in_train, store_in_sample_residuals=interval is not None,
                       suppress_warnings=True)
        levels = list(in_train) if variant == "drop" else list(train)
        lw_index = idx[f["train_end"] - ws:f["train_end"]]
        last_window = pd.DataFrame({k: train[k].reindex(lw_index) for k in levels})
        last_window.index.freq = idx.freq
        if interval is None:
            pred = forecaster.predict(steps=len(te), levels=levels,
                                      last_window=last_window, suppress_warnings=True)
        else:
            pred = forecaster.predict_interval(
                steps=len(te), levels=levels, last_window=last_window,
                method="conformal", interval=interval, suppress_warnings=True
            )
        pred = pred.reset_index(names="date")
        if scaling == "own":
            for col in [c for c in ["pred", "lower_bound", "upper_bound"] if c in pred]:
                mu = pred["level"].map(lambda k: stats[k][0])
                sd = pred["level"].map(lambda k: stats[k][1])
                pred[col] = pred[col] * sd + mu
        pred["fold"] = f["fold"]
        recs.append(pred)
    return pd.concat(recs, ignore_index=True)


def score(E, pred):
    df = E.merge(pred, on=["level", "fold", "date"], how="left")
    covered = df["pred"].notna()
    mae = (df.loc[covered, "pred"] - df.loc[covered, "y"]).abs().groupby(
        df.loc[covered, "level"]).mean().mean()
    return df, covered.mean(), mae


def make_factory(est_name, lags, encoding="ordinal", window_features=None,
                 differentiation=None):
    def factory(scaling):
        return ForecasterRecursiveMultiSeries(
            estimator          = ESTIMATORS[est_name](),
            lags               = lags,
            window_features    = window_features,
            encoding           = encoding,
            transformer_series = None if scaling == "own" else StandardScaler(),
            differentiation    = differentiation,
        )
    return factory


# ---------------------------------------------------------------------------
# Part: synthetic
# ---------------------------------------------------------------------------
def part_synthetic():
    seeds = [42, 0, 1]

    # 1. Validation of the prototype against skforecast
    series, idx, cv = synthetic_data(42)
    folds = get_folds(series, idx, cv)
    for lags in [12, 24]:
        proto = proto_backtest(series, idx, folds, make_factory("LightGBM", lags), "drop")
        f = make_factory("LightGBM", lags)("skforecast")
        _, bt = backtesting_forecaster_multiseries(
            f, series, TimeSeriesFold(**cv), "mean_absolute_error",
            n_jobs=1, show_progress=False, suppress_warnings=True)
        bt = bt.reset_index(names="date")
        m = bt.merge(proto, on=["date", "level", "fold"], suffixes=("_bt", "_proto"))
        print(f"[validation] lags={lags}: rows backtesting={len(bt)}, prototype={len(proto)}, "
              f"matched={len(m)}, max abs diff={np.abs(m['pred_bt'] - m['pred_proto']).max():.2e}")

    # 2. Estimators x variants (ordinal encoding)
    rows = []
    for est_name in ESTIMATORS:
        for seed in seeds:
            series, idx, cv = synthetic_data(seed)
            folds = get_folds(series, idx, cv)
            E = evaluation_set(series, idx, folds)
            res = {}
            for lags in [12, 24]:
                for variant in ["drop", "c2_pooled", "c2_own"]:
                    pred = proto_backtest(series, idx, folds, make_factory(est_name, lags), variant)
                    res[(variant, lags)] = score(E, pred)
            cold = ~res[("drop", 24)][0]["pred"].notna().to_numpy()
            row = {"estimator": est_name, "seed": seed,
                   "cold_naive": (E.loc[cold, "naive_last"] - E.loc[cold, "y"]).abs().mean()}
            for variant in ["drop", "c2_pooled", "c2_own"]:
                row[f"adv24_{variant}"] = res[(variant, 12)][2] - res[(variant, 24)][2]
            for variant in ["c2_pooled", "c2_own"]:
                d = res[(variant, 24)][0][cold]
                row[f"cold_{variant}"] = (d["pred"] - d["y"]).abs().mean()
            row["coverage24_drop"] = res[("drop", 24)][1]
            row["coverage24_c2"] = res[("c2_pooled", 24)][1]
            rows.append(row)
        print(f"done {est_name}", flush=True)
    res = pd.DataFrame(rows)
    print("\n[estimators] lags=24 vs lags=12, MAE (cold = points that lags=24 drops today)")
    print(res.round(2).to_string(index=False))
    print("\nMean over seeds:")
    print(res.drop(columns="seed").groupby("estimator", sort=False).mean().round(2).to_string())

    # 3. Encodings (LightGBM and HGBR), cold-start MAE with lags=24
    rows = []
    for est_name in ["LightGBM", "HGBR"]:
        for encoding in ["ordinal", "ordinal_category", "onehot", None]:
            for seed in seeds:
                series, idx, cv = synthetic_data(seed)
                folds = get_folds(series, idx, cv)
                E = evaluation_set(series, idx, folds)
                drop = score(E, proto_backtest(series, idx, folds, make_factory(est_name, 24, encoding), "drop"))[0]
                cold = ~drop["pred"].notna().to_numpy()
                row = {"estimator": est_name, "encoding": str(encoding), "seed": seed}
                for variant in ["c2_pooled", "c2_own"]:
                    try:
                        pred = proto_backtest(series, idx, folds, make_factory(est_name, 24, encoding), variant)
                        d = score(E, pred)[0][cold]
                        row[f"cold_{variant}"] = (d["pred"] - d["y"]).abs().mean()
                    except Exception as e:
                        row[f"cold_{variant}"] = np.nan
                        row["error"] = f"{type(e).__name__}: {str(e)[:80]}"
                rows.append(row)
    res = pd.DataFrame(rows)
    print("\n[encodings] cold-start MAE, lags=24, mean over seeds")
    cols = [c for c in ["cold_c2_pooled", "cold_c2_own"] if c in res]
    print(res.groupby(["estimator", "encoding"], sort=False)[cols].mean().round(2).to_string())
    if "error" in res:
        print(res.dropna(subset=["error"])[["estimator", "encoding", "error"]].drop_duplicates().to_string(index=False))

    # 4. window_features: lags=6 + RollingFeatures(mean, std, sum, 24), window 24
    rows = []
    for patched in [False, True]:
        RollingFeatures.transform = _transform_consistent if patched else _original_transform
        for seed in seeds:
            series, idx, cv = synthetic_data(seed)
            folds = get_folds(series, idx, cv)
            E = evaluation_set(series, idx, folds)
            wf = RollingFeatures(stats=["mean", "std", "sum"], window_sizes=24)
            fac = make_factory("LightGBM", 6, window_features=wf)
            drop = score(E, proto_backtest(series, idx, folds, fac, "drop"))
            cold = ~drop[0]["pred"].notna().to_numpy()
            row = {"predict path": "NaN if < min_periods (patched)" if patched else "ignores NaN (today)",
                   "seed": seed, "coverage_drop": drop[1]}
            for variant in ["c2_pooled", "c2_own"]:
                d = score(E, proto_backtest(series, idx, folds, fac, variant))[0][cold]
                row[f"cold_{variant}"] = (d["pred"] - d["y"]).abs().mean()
            row["cold_naive"] = (E.loc[cold, "naive_last"] - E.loc[cold, "y"]).abs().mean()
            rows.append(row)
    RollingFeatures.transform = _original_transform
    res = pd.DataFrame(rows)
    print("\n[window_features] LightGBM, lags=6 + roll mean/std/sum 24, cold-start MAE, mean over seeds")
    print(res.drop(columns="seed").groupby("predict path", sort=False).mean().round(2).to_string())

    # 5. Intervals: conformal [0.1, 0.9] with in-sample residuals, LightGBM, lags=24
    rows = []
    for seed in seeds:
        series, idx, cv = synthetic_data(seed)
        folds = get_folds(series, idx, cv)
        E = evaluation_set(series, idx, folds)
        drop = score(E, proto_backtest(series, idx, folds, make_factory("LightGBM", 24), "drop"))[0]
        cold = ~drop["pred"].notna().to_numpy()
        for variant in ["c2_pooled", "c2_own"]:
            df = score(E, proto_backtest(series, idx, folds, make_factory("LightGBM", 24), variant,
                                         interval=[0.1, 0.9]))[0]
            inside = (df["y"] >= df["lower_bound"]) & (df["y"] <= df["upper_bound"])
            width = df["upper_bound"] - df["lower_bound"]
            rows.append({"variant": variant, "seed": seed,
                         "coverage_covered_today": inside[~cold].mean(),
                         "coverage_cold": inside[cold].mean(),
                         "width_covered_today": width[~cold].mean(),
                         "width_cold": width[cold].mean()})
    res = pd.DataFrame(rows)
    print("\n[intervals] conformal 80%, LightGBM, lags=24, mean over seeds")
    print(res.drop(columns="seed").groupby("variant").mean().round(3).to_string())

    # 6. Differentiation with a padded last window, LightGBM, lags=12
    rows = []
    for seed in seeds:
        series, idx, cv = synthetic_data(seed)
        folds = get_folds(series, idx, cv)
        E = evaluation_set(series, idx, folds)
        for diff in [None, 1]:
            fac = make_factory("LightGBM", 12, differentiation=diff)
            drop = score(E, proto_backtest(series, idx, folds, fac, "drop"))
            cold = ~drop[0]["pred"].notna().to_numpy()
            row = {"differentiation": str(diff), "seed": seed, "MAE drop (covered)": drop[2]}
            for variant in ["c2_pooled", "c2_own"]:
                pred = proto_backtest(series, idx, folds, fac, variant)
                d = score(E, pred)[0][cold]
                row[f"cold_{variant}"] = (d["pred"] - d["y"]).abs().mean()
                row[f"NaN preds {variant}"] = pred["pred"].isna().mean()
            row["cold_naive"] = (E.loc[cold, "naive_last"] - E.loc[cold, "y"]).abs().mean()
            rows.append(row)
    res = pd.DataFrame(rows)
    print("\n[differentiation] LightGBM, lags=12, mean over seeds")
    print(res.drop(columns="seed").groupby("differentiation").mean().round(2).to_string())


# ---------------------------------------------------------------------------
# Part: m5
# ---------------------------------------------------------------------------
def part_m5(path):
    series, idx, cv = m5_data(path)
    folds = get_folds(series, idx, cv)
    E = evaluation_set(series, idx, folds)
    n_new = sum(1 for s in series.values() if s.index[0] >= pd.Timestamp("2014-01-01"))
    print(f"[m5] {len(series)} series ({n_new} launched in 2014), {len(folds)} folds, "
          f"{len(E)} test points")
    lags_grid = [7, 28, 56]
    rows, res = [], {}
    for est_name in ["LightGBM", "HGBR"]:
        for lags in lags_grid:
            for variant in ["drop", "c2_pooled", "c2_own"]:
                t0 = time.time()
                pred = proto_backtest(series, idx, folds, make_factory(est_name, lags), variant)
                res[(est_name, variant, lags)] = score(E, pred)
                print(f"done {est_name} lags={lags} {variant} ({time.time() - t0:.0f}s)", flush=True)
        cold = ~res[(est_name, "drop", 56)][0]["pred"].notna().to_numpy()
        for variant in ["drop", "c2_pooled", "c2_own"]:
            for lags in lags_grid:
                df, cov, mae = res[(est_name, variant, lags)]
                d = df[cold]
                rows.append({"estimator": est_name, "variant": variant, "lags": lags,
                             "coverage": cov, "MAE per series": mae,
                             "MAE cold (lags=56 drops)": (d["pred"] - d["y"]).abs().mean()
                             if variant != "drop" else np.nan})
        rows.append({"estimator": est_name, "variant": "naive_last", "lags": np.nan,
                     "MAE cold (lags=56 drops)": (E.loc[cold, "naive_last"] - E.loc[cold, "y"]).abs().mean()})
        rows.append({"estimator": est_name, "variant": "naive_mean28", "lags": np.nan,
                     "MAE cold (lags=56 drops)": (E.loc[cold, "naive_mean28"] - E.loc[cold, "y"]).abs().mean()})
        print(f"cold-start points: {cold.sum()} of {len(cold)}")
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    if part == "synthetic":
        part_synthetic()
    else:
        part_m5(sys.argv[2])
