"""Does the estimator receive in `predict` the same row it received in `fit`?

A recording estimator stores what it gets. A forecaster fitted with the whole
series gives the training row of (series, t). A second forecaster fitted up to
t-1 predicts one step: the row it passes to `predict` must be the same.
"""
import sys
import warnings
import numpy as np
import pandas as pd
import skforecast
print(skforecast.__file__, flush=True)
from sklearn.base import BaseEstimator, RegressorMixin
from skforecast.recursive import ForecasterRecursiveMultiSeries
import common as C


class Rec(BaseEstimator, RegressorMixin):
    def fit(self, X, y, sample_weight=None):
        self.fit_type_ = type(X).__name__
        self.fit_dtypes_ = (
            sorted(set(map(str, X.dtypes))) if isinstance(X, pd.DataFrame)
            else [str(X.dtype)]
        )
        self.X_fit_ = X.copy()
        self.pred_ = []
        return self

    def predict(self, X):
        self.pred_.append((type(X).__name__, np.array(X, copy=True)))
        return np.zeros(len(X))


def to_float(X):
    """What a numeric estimator would read from the matrix."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float, na_value=np.nan)
    return np.asarray(X, dtype=float)


def to_codes(X):
    """What LightGBM/XGBoost read from a DataFrame: codes of `category`."""
    if isinstance(X, pd.DataFrame):
        X = X.copy()
        for c in X.columns:
            if isinstance(X[c].dtype, pd.CategoricalDtype):
                X[c] = X[c].cat.codes
        return X.to_numpy(dtype=float, na_value=np.nan)
    return np.asarray(X, dtype=float)


n_ok = n_bad = n_err = 0
rows = []
for as_dict in (False, True):
    dtypes = C.DTYPES if not as_dict else ["float", "cat_int_vals", "mixed", "nullable"]
    for dtype_name in dtypes:
        series, exog, exog_fut, cat_like, cat_cols = C.make_data(dtype_name, as_dict=as_dict)
        exog_cols = list(exog.columns) if not as_dict else list(exog["s_a"].columns)
        for cf_mode in C.CATFEATS:
            cf = C.catfeat_value(cf_mode, cat_like, exog_cols)
            if cf == []:
                continue
            for enc in C.ENCODINGS:
                key = (dtype_name, cf_mode, enc, as_dict)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        f1 = ForecasterRecursiveMultiSeries(
                            estimator=Rec(), lags=3, encoding=enc,
                            categorical_features=cf,
                        )
                        f1.fit(series=series, exog=exog)
                        # last training row of each series = time n-1
                        if as_dict:
                            series_cut = {k: v.iloc[:-1] for k, v in series.items()}
                            exog_cut = {k: v.iloc[:-1] for k, v in exog.items()}
                            exog_last = {k: v.iloc[-1:] for k, v in exog.items()}
                        else:
                            series_cut = series.iloc[:-1]
                            exog_cut = exog.iloc[:-1]
                            exog_last = exog.iloc[-1:]
                        f2 = ForecasterRecursiveMultiSeries(
                            estimator=Rec(), lags=3, encoding=enc,
                            categorical_features=cf,
                        )
                        f2.fit(series=series_cut, exog=exog_cut)
                        f2.estimator.pred_.clear()
                        f2.predict(steps=1, exog=exog_last)
                    X_fit = f1.estimator.X_fit_
                    # Rows of the last date, one per series, in the training matrix
                    _, y_tr = f1.create_train_X_y(series=series, exog=exog)
                    last_date = y_tr.index.max()
                    last_pos = np.flatnonzero(y_tr.index == last_date)
                    Xf = X_fit.iloc[last_pos] if isinstance(X_fit, pd.DataFrame) else X_fit[last_pos]
                    ptype, Xp = f2.estimator.pred_[-1]
                    same_values = np.array_equal(to_float(Xf), np.asarray(Xp, dtype=float), equal_nan=True)
                    same_codes = np.array_equal(to_codes(Xf), np.asarray(Xp, dtype=float), equal_nan=True)
                    rows.append((key, f1.estimator.fit_type_, f1.estimator.fit_dtypes_, ptype, same_values, same_codes))
                    if same_values and same_codes:
                        n_ok += 1
                    else:
                        n_bad += 1
                        print("DIFF", key, "fit:", f1.estimator.fit_type_, f1.estimator.fit_dtypes_,
                              "| same as values:", same_values, "| same as codes (what a boosting reads from a DataFrame):", same_codes)
                except Exception as e:  # noqa: BLE001
                    n_err += 1
                    rows.append((key, "ERR", type(e).__name__, str(e)[:100]))
import collections
print("fit input types:", collections.Counter(r[1] for r in rows))
print("ok", n_ok, "diff", n_bad, "errors", n_err)
errs = collections.Counter((r[0][0], r[0][1], r[2], r[3][:70]) for r in rows if r[1] == "ERR")
for k, v in sorted(errs.items()):
    print("  ERR", v, k)
