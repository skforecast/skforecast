"""
Prototypes of the two preprocessing speedups of
`dev/preprocessing_speedups/PLAN.md`, applied at runtime (monkeypatching)
so that production code is not modified. They are references for the
implementation, not the implementation itself.

P1  FastOrdinalEncoder: encodes pandas `category` columns with their integer
    codes and a lookup table against `categories_`. Same output as the
    forecaster's `OrdinalEncoder` (bit-identical), which converts the columns
    to object arrays and maps them with Python dictionaries.
P2  fast_transform_numpy: `transform_numpy` without scikit-learn's input
    validation for a fitted `StandardScaler` (same arithmetic and order of
    operations as `StandardScaler.transform` / `inverse_transform`).
"""

from __future__ import annotations

import contextlib
import importlib
import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import skforecast.utils.utils as _utils

# --------------------------------------------------------------------- P1


class FastOrdinalEncoder:
    """
    Drop-in for the `categorical_encoder` of the forecasters. The fitted
    state lives in a scikit-learn `OrdinalEncoder` (`self.sk`), fitted on
    the categories present only, so `categories_` is the same as fitting it
    on the whole data. Columns that are not `category` fall back to
    scikit-learn.
    """

    def __init__(self):
        self.sk = OrdinalEncoder(
            dtype=float,
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        ).set_output(transform="pandas")

    @property
    def categories_(self):
        return self.sk.categories_

    @staticmethod
    def _all_categorical(X):
        return all(isinstance(d, pd.CategoricalDtype) for d in X.dtypes)

    def fit(self, X: pd.DataFrame):
        if not self._all_categorical(X):
            self.sk.fit(X)
            return self
        # NOTE: One row per category present (and one missing value, if any)
        # is taken from each column, so scikit-learn receives the same dtype
        # as with the whole data and `categories_` is identical, values and
        # dtype (e.g. int64 for integer categories). Columns are padded with
        # their first row: repeated values do not change `categories_`.
        positions = {}
        for col in X.columns:
            _, first_rows = np.unique(X[col].cat.codes.to_numpy(), return_index=True)
            positions[col] = first_rows
        n = max(len(p) for p in positions.values())
        small = pd.DataFrame(
            {
                col: X[col].iloc[np.r_[p, np.full(n - len(p), p[0])]].reset_index(drop=True)
                for col, p in positions.items()
            }
        )
        self.sk.fit(small)
        return self

    def _lookup(self, j, dtype):
        """Position in `categories_[j]` of each category of `dtype`, NaN if unseen."""
        pos = pd.Index(self.sk.categories_[j]).get_indexer(dtype.categories)
        lookup = pos.astype(float)
        lookup[pos < 0] = np.nan
        # Code -1 (missing value) indexes the last element: NaN.
        return np.append(lookup, np.nan)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._all_categorical(X):
            return self.sk.transform(X)
        out = {
            col: self._lookup(j, X[col].dtype)[X[col].cat.codes.to_numpy()]
            for j, col in enumerate(X.columns)
        }
        return pd.DataFrame(out, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


@contextlib.contextmanager
def use_fast_encoder(forecaster):
    """Replace the forecaster's encoder (kept after the block: predict needs it)."""
    forecaster.categorical_encoder = FastOrdinalEncoder()
    yield forecaster


# --------------------------------------------------------------------- P2

_ORIG_TRANSFORM_NUMPY = _utils.transform_numpy

# Modules that import `transform_numpy` by name.
_MODULES = [
    "skforecast.recursive._forecaster_recursive_multiseries",
    "skforecast.recursive._forecaster_recursive",
    "skforecast.direct._forecaster_direct",
    "skforecast.direct._forecaster_direct_multivariate",
    "skforecast.model_selection._utils",
]


def fast_transform_numpy(
    array, transformer, fit=False, inverse_transform=False, force_single_column=False
):
    """
    Shortcut for a fitted `StandardScaler` with one feature, no fitting and a
    float64 array. Anything else goes to the current `transform_numpy`.
    An `inf` in the input also goes to the current `transform_numpy`, which
    raises the same error as scikit-learn (point 5 of T1 in PLAN.md).
    """
    if (
        not fit
        and type(transformer) is StandardScaler
        and isinstance(array, np.ndarray)
        and array.dtype == np.float64
        and getattr(transformer, "n_features_in_", None) == 1
        and hasattr(transformer, "scale_")
        and (inverse_transform or array.ndim == 1)
    ):
        out = array.copy()
        if inverse_transform:
            if transformer.with_std:
                out *= transformer.scale_[0]
            if transformer.with_mean:
                out += transformer.mean_[0]
        else:
            if transformer.with_mean:
                out -= transformer.mean_[0]
            if transformer.with_std:
                out /= transformer.scale_[0]
        # NOTE: Same check as the fast path of linear models in
        # `_build_predict_function`: the input is only inspected when the sum
        # of the result is not finite (NaN, which scikit-learn accepts, or inf).
        if not math.isfinite(out.sum()) and np.isinf(array).any():
            return _ORIG_TRANSFORM_NUMPY(
                array=array, transformer=transformer, fit=fit,
                inverse_transform=inverse_transform,
                force_single_column=force_single_column,
            )
        return out
    return _ORIG_TRANSFORM_NUMPY(
        array=array, transformer=transformer, fit=fit,
        inverse_transform=inverse_transform, force_single_column=force_single_column,
    )


@contextlib.contextmanager
def use_fast_scaler():
    """Patch `transform_numpy` in the modules that use it while active."""
    mods = [importlib.import_module(m) for m in _MODULES]
    for m in mods:
        m.transform_numpy = fast_transform_numpy
    try:
        yield
    finally:
        for m in mods:
            m.transform_numpy = _ORIG_TRANSFORM_NUMPY
