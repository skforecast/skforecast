"""
T3: learn the bin edges of `fit` with a random sample of the rows of X_train
instead of predicting on all of them.

Part 1 (cost): share of `fit` spent in the prediction over X_train, and what a
sample would cost instead.
Part 2 (edges): how far the edges learned with a sample are from the edges
learned with all the rows.

Nothing in the package is modified: the pieces of `fit` are reproduced here and
checked against the fitted forecaster.

python dev/preprocessing_speedups/t3_cost_edges.py --part multi --sizes S,M --estimators lgbm,hgbr
python dev/preprocessing_speedups/t3_cost_edges.py --part single --estimators lgbm,hgbr
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from data import SIZES, env_report, make_data, make_estimator, make_forecaster  # noqa: E402

from skforecast.preprocessing import QuantileBinner, RollingFeatures  # noqa: E402
from skforecast.recursive import ForecasterRecursive  # noqa: E402

warnings.filterwarnings("ignore")

BINNER_KWARGS = {
    "n_bins": 10, "method": "linear", "subsample": 200000,
    "random_state": 789654, "dtype": np.float64,
}


def best_of(func, repeat):
    times = []
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = func()
        times.append(time.perf_counter() - t0)
    return min(times), out


def interleaved(fit_default, fit_noprob, repeat):
    """Best time of each variant, alternating them to share the machine noise."""
    times_default, times_noprob, f = [], [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        f = fit_default()
        times_default.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        fit_noprob()
        times_noprob.append(time.perf_counter() - t0)
    print(
        "fit times (s) default:", [round(x, 2) for x in times_default],
        "| without the block:", [round(x, 2) for x in times_noprob],
    )
    return min(times_default), min(times_noprob), f


def fit_binner(y_pred):
    binner = QuantileBinner(**BINNER_KWARGS)
    binner.fit(y_pred)
    return binner


def edge_metrics(binner_full, binner_sample, y_pred_full):
    """
    Share of the predictions of the level (all rows) that fall in each bin of
    the sampled binner (ideal: 1 / n_bins each), and share of rows whose bin
    changes with respect to the binner learned with all the rows.
    """
    bins_full = binner_full.transform(y_pred_full).astype(int)
    bins_sample = binner_sample.transform(y_pred_full).astype(int)
    shares = np.bincount(bins_sample, minlength=binner_sample.n_bins_) / len(y_pred_full)
    changed = float(np.mean(bins_full != bins_sample))
    return shares.min(), shares.max(), changed, binner_sample.n_bins_


def multiseries(size, est_name, repeat_fit, caps, fracs, seeds):
    print(f"\n=== ForecasterRecursiveMultiSeries, size {size}, {est_name} ===")
    s, e = make_data(**SIZES[size])

    def fit_default():
        f = make_forecaster(est_name)
        f.fit(series=s, exog=e)
        return f

    def fit_noprob():
        f = make_forecaster(est_name)
        f._probabilistic_mode = False
        f.fit(series=s, exog=e)
        return f

    t_default, t_noprob, f = interleaved(fit_default, fit_noprob, repeat_fit)

    X_train = f._create_train_X_y(series=s, exog=e, store_last_window=False)[0]
    level_slices = f._get_level_row_slices(X_train=X_train)
    levels = list(level_slices.keys())
    n_rows = len(X_train)
    rows_level = np.array([level_slices[k].stop - level_slices[k].start for k in levels])
    print(
        f"rows X_train: {n_rows:,} | series: {len(levels)} | rows per series: "
        f"min {rows_level.min()}, median {int(np.median(rows_level))}, max {rows_level.max()}"
    )

    t_pred_full, y_pred = best_of(lambda: f.estimator.predict(X_train), 3)

    def binning_full():
        binners = {k: fit_binner(y_pred[level_slices[k]]) for k in levels}
        binners["_unknown_level"] = fit_binner(y_pred)
        return binners

    t_bin_full, binners_full = best_of(binning_full, 3)
    same = all(
        binners_full[k].intervals_ == f.binner_intervals_[k] for k in binners_full
    )
    print(f"edges reproduced equal to the fitted forecaster: {same}")
    block = t_default - t_noprob
    print(
        f"fit default {t_default:.2f} s | fit without the block {t_noprob:.2f} s | "
        f"block {block:.2f} s ({100 * block / t_default:.1f} % of fit) | "
        f"predict(X_train) alone {t_pred_full:.2f} s | binners {t_bin_full * 1e3:.0f} ms"
    )

    def sample_indices(rng, cap=None, frac=None):
        """Row positions sampled per level, without replacement, sorted."""
        idx, slices, start = [], {}, 0
        for k in levels:
            sl = level_slices[k]
            n = sl.stop - sl.start
            m = min(n, cap) if cap is not None else max(int(round(n * frac)), 1)
            pos = np.sort(rng.choice(n, size=m, replace=False)) + sl.start
            idx.append(pos)
            slices[k] = slice(start, start + m)
            start += m
        return np.concatenate(idx), slices

    variants = [("cap", c) for c in caps] + [("frac", fr) for fr in fracs]
    rows = []
    for kind, value in variants:
        mins, maxs, changes, n_bins_red, times, n_sampled = [], [], [], 0, [], 0
        for seed in seeds:
            rng = np.random.default_rng(seed)

            def block_sample():
                kw = {"cap": value} if kind == "cap" else {"frac": value}
                idx, slices = sample_indices(rng, **kw)
                y_pred_s = f.estimator.predict(X_train.iloc[idx])
                binners = {k: fit_binner(y_pred_s[slices[k]]) for k in levels}
                binners["_unknown_level"] = fit_binner(y_pred_s)
                return idx, binners

            t0 = time.perf_counter()
            idx, binners_s = block_sample()
            times.append(time.perf_counter() - t0)
            n_sampled = len(idx)
            for k in levels:
                mn, mx, ch, nb = edge_metrics(
                    binners_full[k], binners_s[k], y_pred[level_slices[k]]
                )
                mins.append(mn)
                maxs.append(mx)
                changes.append(ch)
                n_bins_red += nb < binners_full[k].n_bins_
        t_s = min(times)
        rows.append(
            {
                "sample": f"{kind} {value}",
                "rows predicted": f"{n_sampled:,} ({100 * n_sampled / n_rows:.0f} %)",
                "block (s)": round(t_s, 2),
                "fit (s)": round(t_noprob + t_s, 2),
                "fit vs default": f"{100 * (t_noprob + t_s) / t_default - 100:+.1f} %",
                "bin share min (mean)": f"{100 * np.mean(mins):.1f} %",
                "bin share max (mean)": f"{100 * np.mean(maxs):.1f} %",
                "smallest bin (worst)": f"{100 * np.min(mins):.1f} %",
                "rows changing bin": f"{100 * np.mean(changes):.1f} %",
                "levels with fewer bins": n_bins_red,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


def single_series(n, est_name, repeat_fit, sizes, seeds):
    print(f"\n=== ForecasterRecursive, {n:,} observations, {est_name} ===")
    rng = np.random.default_rng(0)
    t = np.arange(n)
    season = np.sin(2 * np.pi * t / 24)
    y = pd.Series(
        50 + 5 * season + 0.1 * np.cumsum(rng.normal(0, 0.3, n)) + rng.normal(0, 1, n),
        index=pd.date_range("2000-01-01", periods=n, freq="h"),
        name="y",
    )
    exog = pd.DataFrame(
        {"e1": season + rng.normal(0, 0.1, n), "e2": rng.normal(0, 1, n)}, index=y.index
    )

    def make():
        return ForecasterRecursive(
            estimator=make_estimator(est_name),
            lags=48,
            window_features=RollingFeatures(stats=["mean", "std", "min"], window_sizes=24),
        )

    def fit_default():
        f = make()
        f.fit(y=y, exog=exog)
        return f

    def fit_noprob():
        f = make()
        f._probabilistic_mode = False
        f.fit(y=y, exog=exog)
        return f

    t_default, t_noprob, f = interleaved(fit_default, fit_noprob, repeat_fit)
    X_train, y_train = f.create_train_X_y(y=y, exog=exog)
    X_np = X_train.to_numpy()
    t_pred_full, y_pred = best_of(lambda: f.estimator.predict(X_np), 3)
    binner_full = fit_binner(y_pred)
    block = t_default - t_noprob
    print(
        f"rows X_train: {len(X_np):,} | fit default {t_default:.2f} s | without the "
        f"block {t_noprob:.2f} s | block {block:.3f} s ({100 * block / t_default:.1f} % "
        f"of fit) | predict(X_train) alone {t_pred_full:.3f} s"
    )
    rows = []
    for m in sizes:
        if m >= len(X_np):
            continue
        mins, maxs, changes, times = [], [], [], []
        for seed in seeds:
            rng_s = np.random.default_rng(seed)
            t0 = time.perf_counter()
            idx = np.sort(rng_s.choice(len(X_np), size=m, replace=False))
            y_pred_s = f.estimator.predict(X_np[idx])
            binner_s = fit_binner(y_pred_s)
            times.append(time.perf_counter() - t0)
            mn, mx, ch, _ = edge_metrics(binner_full, binner_s, y_pred)
            mins.append(mn)
            maxs.append(mx)
            changes.append(ch)
        t_s = min(times)
        rows.append(
            {
                "rows predicted": f"{m:,} ({100 * m / len(X_np):.0f} %)",
                "block (s)": round(t_s, 3),
                "fit vs default": f"{100 * (t_noprob + t_s) / t_default - 100:+.1f} %",
                "bin share min (mean)": f"{100 * np.mean(mins):.2f} %",
                "bin share max (mean)": f"{100 * np.mean(maxs):.2f} %",
                "rows changing bin": f"{100 * np.mean(changes):.2f} %",
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="multi")
    ap.add_argument("--sizes", default="S,M")
    ap.add_argument("--estimators", default="lgbm,hgbr")
    args = ap.parse_args()
    print(env_report())
    seeds = [1, 2, 3]
    if args.part == "multi":
        for size in args.sizes.split(","):
            for est in args.estimators.split(","):
                multiseries(
                    size, est, repeat_fit=3 if size == "S" else 2,
                    caps=[2000, 1000, 500, 250], fracs=[0.5, 0.25, 0.1], seeds=seeds,
                )
    else:
        for n in (20_000, 200_000, 1_000_000):
            for est in args.estimators.split(","):
                single_series(
                    n, est, repeat_fit=3 if n <= 200_000 else 2,
                    sizes=[5_000, 10_000, 20_000, 50_000, 200_000], seeds=seeds,
                )
