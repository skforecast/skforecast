"""
Review of T1: empty inputs through the public API. Today scikit-learn rejects
an empty array inside `transform_numpy` (`ValueError`); a shortcut without a
check of the size accepts it.

python dev/preprocessing_speedups/review_t1_empty_inputs.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from prototypes import fast_transform_numpy
from review_t1_common import (
    ORIG_TRANSFORM_NUMPY,
    PACKAGE_TRANSFORM_NUMPY,
    patched,
    proposed,
)

from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
idx = pd.date_range("2024-01-01", periods=200, freq="h")
y = pd.Series(50 + rng.normal(0, 3, 200).cumsum(), index=idx, name="y")
series = pd.DataFrame({f"s{i}": rng.normal(0, 2, 200).cumsum() for i in range(2)}, index=idx)
m = None
def flows():
    global m
    f = ForecasterRecursive(Ridge(), lags=5, transformer_y=StandardScaler())
    f.fit(y)
    yield "ForecasterRecursive.set_out_sample_residuals(empty arrays)", lambda: f.set_out_sample_residuals(y_true=np.array([]), y_pred=np.array([]))
    yield "ForecasterRecursive.set_out_sample_residuals(empty Series)", lambda: f.set_out_sample_residuals(y_true=pd.Series([], dtype=float), y_pred=pd.Series([], dtype=float))
    m = ForecasterRecursiveMultiSeries(Ridge(), lags=5, transformer_series=StandardScaler())
    m.fit(series)
    yield "ForecasterRecursiveMultiSeries.set_out_sample_residuals(empty arrays)", lambda: m.set_out_sample_residuals(y_true={"s0": np.array([])}, y_pred={"s0": np.array([])})
    yield "ForecasterRecursiveMultiSeries.predict(last_window empty)", lambda: m.predict(steps=3, last_window=series.iloc[:0])
for label, fn in flows():
    row = []
    for name, func in (("current", ORIG_TRANSFORM_NUMPY), ("PACKAGE", PACKAGE_TRANSFORM_NUMPY),
                       ("PROTO", fast_transform_numpy), ("PROPOSED", proposed)):
        with patched(func):
            try:
                fn()
                row.append(f"{name}: no error")
                if "MultiSeries.set_out" in label:
                    row.append(f"      out_sample_residuals_ after the call: { {k: (None if v is None else v.shape) for k, v in m.out_sample_residuals_.items()} }")
                    try:
                        m.predict_interval(steps=3, levels=["s0"], method="conformal", use_in_sample_residuals=False)
                        row.append("      predict_interval(use_in_sample_residuals=False): no error")
                    except Exception as exc:
                        row.append(f"      predict_interval(use_in_sample_residuals=False): {type(exc).__name__}: {str(exc)[:110]}")
            except Exception as exc:
                row.append(f"{name}: {type(exc).__name__}: {str(exc)[:95]}")
    print(label)
    for r in row:
        print("   ", r)
