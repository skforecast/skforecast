"""
Prototype: in-sample residual stage of `fit()` with contiguous slices per level instead
of one boolean mask per level (O(levels x rows) -> O(rows)).

Two measurements, both same-process, interleaved A/B, bit-identical outputs asserted
before timing:

1. Component: the residual loop alone (masks vs slices) on a fitted forecaster's
   `X_train` / `y_pred`, with `store_in_sample_residuals` False and True.
2. `fit()` total: `fit_replica()` below is a line-by-line copy of
   `ForecasterRecursiveMultiSeries.fit` (7d849ed2b, lines 2020-2177) with the residual
   loop switchable. `fit_replica(mode='mask')` is first checked to produce exactly the
   same fitted state as the real `fit()`; then mask vs slice are compared.

    python dev/profiling_multiseries_fit/proto_residual_slices.py --scenarios A C --reps 5
"""
import argparse
import time

import numpy as np
import pandas as pd

from common import (
    SCENARIOS, ab_interleaved, assert_identical_fits, build_forecaster, fmt, inputs_for,
    make_data, md_table, print_env, save_json,
)
from skforecast.utils import (
    cast_catboost_categorical_columns_dataframe, configure_estimator_categorical_features,
)


# ------------------------------------------------------------------ the alternative
def level_slices(codes: np.ndarray, encoding_mapping: dict, levels: list[str]) -> dict[str, slice]:
    """
    Contiguous row block of each level. `_create_train_X_y` fills the rows level by level
    (offset loop, lines 1262-1287), and the two later row drops (`iloc[mask]`) preserve
    order, so each level is one block. A code found in two blocks means the assumption
    broke: raise instead of silently falling back.
    """
    if len(codes) == 0:
        return {}
    cut = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [len(codes)]))
    by_code = {}
    for s, e in zip(starts, ends):
        c = codes[s]
        if c in by_code:
            raise RuntimeError(f"rows of level code {c} are not contiguous")
        by_code[c] = slice(int(s), int(e))
    return {lvl: by_code[encoding_mapping[lvl]] for lvl in levels if encoding_mapping[lvl] in by_code}


def residual_loop(f, X_train, y_train, y_pred, levels, mode, store, random_state=123):
    """The loop of fit() lines 2142-2165 in both variants."""
    f.binner, f.binner_intervals_ = {}, {}
    f.in_sample_residuals_, f.in_sample_residuals_by_bin_ = {}, {}
    if mode == "mask":
        for level in levels:
            if f.encoding == "onehot":
                mask = X_train[level].to_numpy() == 1.
            else:
                mask = X_train["_level_skforecast"].to_numpy() == f.encoding_mapping_[level]
            f._binning_in_sample_residuals(level=level, y_true=y_train[mask], y_pred=y_pred[mask],
                                           store_in_sample_residuals=store, random_state=random_state)
    else:
        if f.encoding == "onehot":
            keys = list(f.encoding_mapping_.keys())
            codes = X_train[keys].to_numpy() @ np.arange(len(keys))
        else:
            codes = X_train["_level_skforecast"].to_numpy()
        for level, sl in level_slices(codes, f.encoding_mapping_, levels).items():
            f._binning_in_sample_residuals(level=level, y_true=y_train[sl], y_pred=y_pred[sl],
                                           store_in_sample_residuals=store, random_state=random_state)
    f._binning_in_sample_residuals(level="_unknown_level", y_true=y_train, y_pred=y_pred,
                                   store_in_sample_residuals=store, random_state=random_state)


def fit_replica(self, series, exog=None, store_last_window=True, store_in_sample_residuals=False,
                random_state=123, mode="mask"):
    """Copy of ForecasterRecursiveMultiSeries.fit (7d849ed2b) with a switchable residual loop."""
    self.last_window_ = None
    self.index_type_ = None
    self.index_freq_ = None
    self.training_range_ = None
    self.series_names_in_ = None
    self.exog_in_ = False
    self.exog_names_in_ = None
    self.exog_type_in_ = None
    self.exog_dtypes_in_ = None
    self.exog_dtypes_out_ = None
    self.categorical_features_names_in_ = None
    self.X_train_series_names_in_ = None
    self.X_train_window_features_names_out_ = None
    self.X_train_calendar_features_names_out_ = None
    self.X_train_exog_names_out_ = None
    self.X_train_features_names_out_ = None
    self.encoding_mapping_ = {}
    self.in_sample_residuals_ = None
    self.in_sample_residuals_by_bin_ = None
    self.out_sample_residuals_ = None
    self.out_sample_residuals_by_bin_ = None
    self.binner = {}
    self.binner_intervals_ = {}
    self.is_fitted = False
    self.fit_date = None

    (X_train, y_train, series_indexes, series_names_in_, X_train_series_names_in_,
     exog_names_in_, categorical_features_names_in_, X_train_window_features_names_out_,
     X_train_calendar_features_names_out_, X_train_exog_names_out_, exog_dtypes_in_,
     exog_dtypes_out_, last_window_) = self._create_train_X_y(
        series=series, exog=exog, store_last_window=store_last_window)

    sample_weight = self.create_sample_weights(series_names_in_=series_names_in_, X_train=X_train)
    X_train_estimator = X_train if self.encoding is not None else X_train.drop(columns="_level_skforecast")
    X_train_features_names_out_ = X_train_estimator.columns.to_list()

    if self.categorical_features is not None:
        all_categorical_names = list(categorical_features_names_in_) if categorical_features_names_in_ else []
        if self.encoding == "ordinal_category":
            all_categorical_names.append("_level_skforecast")
        fit_kwargs = configure_estimator_categorical_features(
            estimator=self.estimator, categorical_features_names_in_=all_categorical_names,
            X_train_features_names_out_=X_train_features_names_out_, fit_kwargs={**self.fit_kwargs})
    else:
        fit_kwargs = {**self.fit_kwargs}
    X_train_estimator = cast_catboost_categorical_columns_dataframe(
        X=X_train_estimator, fit_kwargs=fit_kwargs, estimator=self.estimator,
        feature_names=X_train_features_names_out_)
    if sample_weight is not None:
        self.estimator.fit(X=X_train_estimator, y=y_train, sample_weight=sample_weight, **fit_kwargs)
    else:
        self.estimator.fit(X=X_train_estimator, y=y_train, **fit_kwargs)

    self.series_names_in_ = series_names_in_
    self.X_train_series_names_in_ = X_train_series_names_in_
    self.X_train_window_features_names_out_ = X_train_window_features_names_out_
    self.X_train_calendar_features_names_out_ = X_train_calendar_features_names_out_
    self.X_train_features_names_out_ = X_train_features_names_out_
    self.is_fitted = True
    self.fit_date = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M:%S")
    self.training_range_ = {k: v[[0, -1]] for k, v in series_indexes.items()}
    self.index_type_ = type(series_indexes[series_names_in_[0]])
    if isinstance(series_indexes[series_names_in_[0]], pd.DatetimeIndex):
        self.index_freq_ = series_indexes[series_names_in_[0]].freq
    else:
        self.index_freq_ = series_indexes[series_names_in_[0]].step
    if exog is not None and X_train_exog_names_out_ is not None:
        self.exog_in_ = True
        self.exog_names_in_ = exog_names_in_
        self.exog_type_in_ = type(exog)
        self.exog_dtypes_in_ = exog_dtypes_in_
        self.exog_dtypes_out_ = exog_dtypes_out_
        self.categorical_features_names_in_ = categorical_features_names_in_
        self.X_train_exog_names_out_ = X_train_exog_names_out_

    self.in_sample_residuals_ = {}
    self.in_sample_residuals_by_bin_ = {}
    if self._probabilistic_mode is not False:
        y_train = y_train.to_numpy()
        y_pred = self.estimator.predict(X_train_estimator)
        levels = X_train_series_names_in_ if self.encoding is not None else []
        residual_loop(self, X_train, y_train, y_pred, levels, mode, store_in_sample_residuals, random_state)

    if not store_in_sample_residuals:
        if self.encoding is not None:
            for level in X_train_series_names_in_:
                self.in_sample_residuals_[level] = None
                self.in_sample_residuals_by_bin_[level] = None
        self.in_sample_residuals_["_unknown_level"] = None
        self.in_sample_residuals_by_bin_["_unknown_level"] = None
    if store_last_window:
        self.last_window_ = last_window_


# ------------------------------------------------------------------ checks and timing
def same_residual_state(fa, fb):
    assert fa.binner_intervals_ == fb.binner_intervals_
    assert list(fa.binner_intervals_) == list(fb.binner_intervals_)  # same insertion order
    assert fa.in_sample_residuals_.keys() == fb.in_sample_residuals_.keys()
    for k in fa.in_sample_residuals_:
        a, b = fa.in_sample_residuals_[k], fb.in_sample_residuals_[k]
        if a is None or b is None:
            assert a is b
        else:
            np.testing.assert_array_equal(a, b)
    for k in fa.in_sample_residuals_by_bin_:
        a, b = fa.in_sample_residuals_by_bin_[k], fb.in_sample_residuals_by_bin_[k]
        if a is None or b is None:
            assert a is b
        else:
            assert a.keys() == b.keys()
            for bin_ in a:
                np.testing.assert_array_equal(a[bin_], b[bin_])


def run(scenario, n_jobs, reps, encoding="ordinal"):
    data = make_data(exog=SCENARIOS[scenario])
    series, exog = inputs_for(data, "dict", "dict")
    print(f"\n=== scenario {scenario}, encoding={encoding}, n_jobs={n_jobs} ===")
    out = {"scenario": scenario, "encoding": encoding}

    # --- 1. component
    f = build_forecaster(n_jobs=n_jobs, encoding=encoding)
    f.fit(series=series, exog=exog)
    X_train, y_train = f.create_train_X_y(series=series, exog=exog)
    y_np = y_train.to_numpy()
    y_pred = f.estimator.predict(X_train)
    levels = f.X_train_series_names_in_
    print(f"rows={len(X_train):,} levels={len(levels)}")
    for store in (False, True):
        fa, fb = build_forecaster(encoding=encoding), build_forecaster(encoding=encoding)
        for g in (fa, fb):
            g.encoding_mapping_ = dict(f.encoding_mapping_)
            g.encoding = encoding
        residual_loop(fa, X_train, y_np, y_pred, levels, "mask", store)
        residual_loop(fb, X_train, y_np, y_pred, levels, "slice", store)
        same_residual_state(fa, fb)
        ab = ab_interleaved(
            lambda: residual_loop(fa, X_train, y_np, y_pred, levels, "mask", store),
            lambda: residual_loop(fb, X_train, y_np, y_pred, levels, "slice", store),
            reps=reps, label_a="mask", label_b="slice")
        print(f"component store_in_sample_residuals={store}: mask {fmt(ab['mask']['median'])} "
              f"| slice {fmt(ab['slice']['median'])} | x{ab['speedup_median']:.1f} "
              f"| saving {fmt(ab['mask']['median'] - ab['slice']['median'])}  [bit-identical OK]")
        out[f"component_store={store}"] = ab

    # masks alone (no binner) to isolate the O(levels x rows) term
    codes = X_train["_level_skforecast"].to_numpy() if encoding != "onehot" else None
    def masks_only():
        for lvl in levels:
            m = (X_train[lvl].to_numpy() == 1.) if encoding == "onehot" else (codes == f.encoding_mapping_[lvl])
            y_np[m]; y_pred[m]
    def slices_only():
        c = (X_train[list(f.encoding_mapping_)].to_numpy() @ np.arange(len(f.encoding_mapping_))
             if encoding == "onehot" else codes)
        for lvl, sl in level_slices(c, f.encoding_mapping_, levels).items():
            y_np[sl]; y_pred[sl]
    ab = ab_interleaved(masks_only, slices_only, reps=reps, label_a="mask", label_b="slice")
    print(f"masks alone (no binner): mask {fmt(ab['mask']['median'])} | slice {fmt(ab['slice']['median'])} "
          f"| x{ab['speedup_median']:.1f}")
    out["masks_only"] = ab

    # --- 2. fit() total
    ref = build_forecaster(n_jobs=n_jobs, encoding=encoding)
    ref.fit(series=series, exog=exog)
    fm = build_forecaster(n_jobs=n_jobs, encoding=encoding)
    fit_replica(fm, series, exog, mode="mask")
    assert_identical_fits(ref, fm, series, exog)
    same_residual_state(ref, fm)
    fs = build_forecaster(n_jobs=n_jobs, encoding=encoding)
    fit_replica(fs, series, exog, mode="slice")
    assert_identical_fits(ref, fs, series, exog)
    same_residual_state(ref, fs)
    print("fit_replica(mask) == fit() and fit_replica(slice) == fit(): bit-identical OK")
    ab = ab_interleaved(
        lambda: fit_replica(build_forecaster(n_jobs=n_jobs, encoding=encoding), series, exog, mode="mask"),
        lambda: fit_replica(build_forecaster(n_jobs=n_jobs, encoding=encoding), series, exog, mode="slice"),
        reps=reps, label_a="mask", label_b="slice")
    print(f"fit() total: mask {fmt(ab['mask']['median'])} (min {ab['mask']['min']:.3f}) | "
          f"slice {fmt(ab['slice']['median'])} (min {ab['slice']['min']:.3f}) | x{ab['speedup_median']:.3f} "
          f"| saving {fmt(ab['mask']['median'] - ab['slice']['median'])} "
          f"= {100 * (1 - ab['slice']['median'] / ab['mask']['median']):.1f}%")
    out["fit_total"] = ab
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["A", "C"], choices=list(SCENARIOS))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--onehot", action="store_true", help="also run encoding='onehot'")
    args = ap.parse_args()
    print_env()
    results = [run(sc, args.n_jobs, args.reps) for sc in args.scenarios]
    if args.onehot:
        results += [run(sc, args.n_jobs, args.reps, encoding="onehot") for sc in args.scenarios]
    print(f"\nsaved {save_json(results, 'proto_residual_slices.json')}")
