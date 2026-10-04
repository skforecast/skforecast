"""
Bit-identity snapshots of `ForecasterRecursiveMultiSeries` outputs, used as the safety net
of the internal `fit()` optimizations (`dev/PLAN_multiseries_fit_optimizations.md`,
section 4.1).

For every case the script computes the training matrices, the fitted state, the
predictions (point, intervals, bootstrapping), the sample weights, the residuals set
by `set_in_sample_residuals` and the one-step-ahead split, and stores a fingerprint of
each object in `results/snapshot/<case>.json`: sha1 of the raw bytes of every array /
column, plus shapes, dtypes, names and the order of dict keys. Floats are stored with
`repr`, so the comparison is bitwise. Residual containers are compared with
`np.asarray` so that a `Series` and an `ndarray` with the same values match.

Snapshots depend on the environment (LightGBM, numpy, pandas versions): they are not
versioned (`.gitignore`). Generate them on the base code, then check after each change:

    python dev/profiling_multiseries_fit/11_snapshot_outputs.py            # generate
    python dev/profiling_multiseries_fit/11_snapshot_outputs.py --check    # compare
    python dev/profiling_multiseries_fit/11_snapshot_outputs.py --quick    # 100 series only
"""
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import (
    RESULTS_DIR, SCENARIOS, build_forecaster, env_info, future_exog, make_data, print_env,
)

SNAPSHOT_DIR = RESULTS_DIR / "snapshot"
STEPS = 10
N_BOOT = 50


def weight_func(index):
    """Module-level weight function (its source must be retrievable)."""
    return np.where(index >= pd.Timestamp("2016-06-01"), 1.0, 0.5)


# ------------------------------------------------------------------ fingerprints
def _sha1_array(a: np.ndarray) -> str:
    if a.dtype == object:
        return hashlib.sha1(repr(a.tolist()).encode()).hexdigest()
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def fp(obj):
    """JSON-able fingerprint that is equal iff the objects are bit-identical."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return repr(float(obj))
    if isinstance(obj, np.ndarray):
        return {"ndarray": str(obj.dtype), "shape": list(obj.shape), "sha1": _sha1_array(obj)}
    if isinstance(obj, pd.Index):
        out = {"index": type(obj).__name__, "dtype": str(obj.dtype), "name": fp(obj.name),
               "values": fp(np.asarray(obj))}
        if isinstance(obj, pd.DatetimeIndex):
            out["freq"] = str(obj.freq)
        return out
    if isinstance(obj, pd.Series):
        if isinstance(obj.dtype, pd.CategoricalDtype):
            values = {"categories": fp(obj.cat.categories), "ordered": obj.cat.ordered,
                      "codes": fp(obj.cat.codes.to_numpy())}
        else:
            values = fp(obj.to_numpy())
        return {"series": fp(obj.name), "dtype": str(obj.dtype), "index": fp(obj.index),
                "values": values}
    if isinstance(obj, pd.DataFrame):
        return {"frame": [fp(c) for c in obj.columns],
                "dtypes": [str(d) for d in obj.dtypes],
                "index": fp(obj.index),
                "columns": [fp(obj.iloc[:, j])["values"] for j in range(obj.shape[1])]}
    if isinstance(obj, dict):
        return {"dict": [[str(k), fp(v)] for k, v in obj.items()]}
    if isinstance(obj, (list, tuple)):
        return [fp(x) for x in obj]
    return repr(obj)


def residuals_as_arrays(residuals):
    """Same values, container-agnostic (`Series` and `ndarray` compare equal)."""
    if residuals is None:
        return None
    out = {}
    for k, v in residuals.items():
        if isinstance(v, dict):
            out[k] = {b: np.asarray(r) for b, r in v.items()}
        else:
            out[k] = None if v is None else np.asarray(v)
    return out


def diff(a, b, path="", out=None):
    """Paths where two fingerprints differ."""
    out = [] if out is None else out
    if type(a) is not type(b):
        out.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
    elif isinstance(a, dict):
        if list(a) != list(b):
            out.append(f"{path}: keys {list(a)} != {list(b)}")
        for k in a:
            if k in b:
                diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(f"{path}: {str(a)[:80]} != {str(b)[:80]}")
    return out


# ------------------------------------------------------------------ cases
def _data(scenario, n_series, n_obs, unequal=False, nans=False):
    data = make_data(n_series=n_series, n_obs=n_obs, exog=SCENARIOS[scenario])
    series = {k: v.copy() for k, v in data["series_dict"].items()}
    exog = data["exog_dict"]
    if exog is not None:
        exog = {k: v.copy() for k, v in exog.items()}
    names = list(series)
    if unequal:
        # Different start dates (same end date so that all levels are predicted).
        for i, k in enumerate(names):
            series[k] = series[k].iloc[(i * 37) % (n_obs // 3):]
    if nans:
        rng = np.random.default_rng(321)
        for k in names[::3]:
            pos = rng.choice(np.arange(100, n_obs - 100), size=n_obs // 20, replace=False)
            series[k].iloc[pos] = np.nan
        if exog is not None:
            for k in names[1::4]:
                pos = rng.choice(np.arange(100, n_obs - 100), size=n_obs // 50, replace=False)
                exog[k].iloc[pos, 0] = np.nan
            exog[names[-1]] = None  # this series loses all its rows with dropna
    return series, exog


def cases(quick: bool) -> dict:
    """name -> (data kwargs, forecaster kwargs)."""
    q = dict(n_series=100, n_obs=1000)
    out = {
        "A100": (dict(scenario="A", **q), {}),
        "B100": (dict(scenario="B", **q), {}),
        "C100": (dict(scenario="C", **q), {}),
        "A100_ordinal_category": (dict(scenario="A", **q), dict(encoding="ordinal_category")),
        "C100_ordinal_category": (dict(scenario="C", **q), dict(encoding="ordinal_category")),
        "A100_onehot": (dict(scenario="A", **q), dict(encoding="onehot")),
        "B100_onehot": (dict(scenario="B", **q), dict(encoding="onehot")),
        "A100_none": (dict(scenario="A", **q), dict(encoding=None)),
        "B100_none": (dict(scenario="B", **q), dict(encoding=None)),
        "A100_unequal_weights": (
            dict(scenario="A", unequal=True, **q),
            dict(series_weights={f"series_{i:04d}": 2.0 for i in range(0, 100, 3)},
                 weight_func=weight_func)),
        "A100_unequal_onehot_weights": (
            dict(scenario="A", unequal=True, **q),
            dict(encoding="onehot",
                 series_weights={f"series_{i:04d}": 2.0 for i in range(0, 100, 3)},
                 weight_func=weight_func)),
        "A100_unequal_none_weights": (
            dict(scenario="A", unequal=True, **q),
            dict(encoding=None,
                 series_weights={f"series_{i:04d}": 2.0 for i in range(0, 100, 3)})),
        "B100_nan_dropna": (dict(scenario="B", nans=True, **q), dict(dropna_from_series=True)),
        "B100_nan_keep": (dict(scenario="B", nans=True, **q), dict(dropna_from_series=False)),
        "A100_scaler_diff": (dict(scenario="A", **q),
                             dict(transformer_series=StandardScaler(), differentiation=1)),
        "A100_no_window_features": (dict(scenario="A", **q), dict(window_features=None)),
    }
    if not quick:
        full = dict(n_series=500, n_obs=2000)
        out = {"A": (dict(scenario="A", **full), {}),
               "B": (dict(scenario="B", **full), {}),
               "C": (dict(scenario="C", **full), {}), **out}
    return out


def _safe(fn):
    try:
        return fp(fn())
    except Exception as e:  # errors are part of the snapshot too
        return {"error": f"{type(e).__name__}: {e}"[:300]}


def snapshot_case(data_kwargs: dict, forecaster_kwargs: dict) -> dict:
    series, exog = _data(**data_kwargs)
    ex = future_exog(exog, STEPS)
    snap = {}

    f = build_forecaster(**forecaster_kwargs)
    X_train, y_train = f.create_train_X_y(series=series, exog=exog)
    snap["X_train"] = fp(X_train)
    snap["y_train"] = fp(y_train)

    f.fit(series=series, exog=exog, store_in_sample_residuals=True)
    snap["X_train_series_names_in_"] = fp(f.X_train_series_names_in_)
    snap["X_train_features_names_out_"] = fp(f.X_train_features_names_out_)
    snap["binner_intervals_"] = fp(f.binner_intervals_)
    snap["in_sample_residuals_"] = fp(residuals_as_arrays(f.in_sample_residuals_))
    snap["in_sample_residuals_by_bin_"] = fp(residuals_as_arrays(f.in_sample_residuals_by_bin_))
    snap["last_window_"] = fp(f.last_window_)
    snap["feature_importances"] = _safe(lambda: f.get_feature_importances())
    if f.series_weights is not None or f.weight_func is not None:
        # Internal X_train, as in fit(): with encoding=None the public one has no
        # `_level_skforecast` column.
        X_train_internal = f._create_train_X_y(
            series=series, exog=exog, store_last_window=False)[0]
        snap["sample_weights"] = _safe(lambda: f.create_sample_weights(
            series_names_in_=f.series_names_in_, X_train=X_train_internal))
    snap["predict"] = _safe(lambda: f.predict(steps=STEPS, exog=ex))
    for method in ("bootstrapping", "conformal"):
        snap[f"predict_interval_{method}"] = _safe(lambda: f.predict_interval(
            steps=STEPS, exog=ex, method=method, interval=[0.1, 0.9], n_boot=N_BOOT,
            random_state=123))
    for binned in (True, False):
        snap[f"predict_bootstrapping_binned={binned}"] = _safe(lambda: f.predict_bootstrapping(
            steps=STEPS, exog=ex, n_boot=N_BOOT, use_binned_residuals=binned,
            random_state=123))

    g = build_forecaster(**forecaster_kwargs)
    g.fit(series=series, exog=exog, store_in_sample_residuals=False)
    g.set_in_sample_residuals(series=series, exog=exog)
    snap["set_in_sample_residuals.binner_intervals_"] = fp(g.binner_intervals_)
    snap["set_in_sample_residuals.in_sample_residuals_"] = fp(
        residuals_as_arrays(g.in_sample_residuals_))
    snap["set_in_sample_residuals.in_sample_residuals_by_bin_"] = fp(
        residuals_as_arrays(g.in_sample_residuals_by_bin_))

    h = build_forecaster(**forecaster_kwargs)
    initial_train_size = int(data_kwargs["n_obs"] * 0.8)
    snap["train_test_split_one_step_ahead"] = _safe(
        lambda: h._train_test_split_one_step_ahead(
            series=series, initial_train_size=initial_train_size, exog=exog))

    return json.loads(json.dumps(snap))


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare against stored snapshots")
    ap.add_argument("--quick", action="store_true", help="skip the 500 x 2000 cases")
    ap.add_argument("--cases", nargs="+", default=None, help="subset of case names")
    args = ap.parse_args()
    print_env()

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    all_cases = cases(quick=args.quick)
    names = args.cases or list(all_cases)
    failed = []
    for name in names:
        t0 = time.perf_counter()
        snap = snapshot_case(*all_cases[name])
        path = SNAPSHOT_DIR / f"{name}.json"
        if args.check:
            stored = json.loads(path.read_text(encoding="utf-8"))["snapshot"]
            problems = diff(stored, snap, name)
            status = "OK" if not problems else f"{len(problems)} DIFFERENCES"
            if problems:
                failed.append(name)
                for p in problems[:20]:
                    print(f"    {p}")
        else:
            path.write_text(
                json.dumps({"env": env_info(), "snapshot": snap}, indent=1), encoding="utf-8")
            status = f"saved {path.name}"
        print(f"{name:32s} {time.perf_counter() - t0:7.1f} s  {status}")

    if args.check:
        print("\nALL IDENTICAL" if not failed else f"\nFAILED: {failed}")
        raise SystemExit(1 if failed else 0)
