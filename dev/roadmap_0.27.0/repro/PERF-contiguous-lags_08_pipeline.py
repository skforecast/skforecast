"""
PERF-contiguous-lags, script 08: `ForecasterRecursive.fit` with a
`Pipeline(StandardScaler(), estimator)`, with the training matrix as it is today
(view) and C-contiguous (prototype). The scaler receives the view and passes a
new array to the estimator, so the copy of the forecaster only helps the scaler.

Same method as script 03 (it imports its `run_case`).

Run from the root of the repository:
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_08_pipeline.py
"""

import importlib.util
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skforecast.recursive import ForecasterRecursive

spec = importlib.util.spec_from_file_location(
    "b03", Path(__file__).with_name("PERF-contiguous-lags_03_fit.py")
)
b03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b03)

print("| estimator | rows x lags | view (ms) | C (ms) | C / view | max abs diff |")
print("|---|---|---|---|---|---|")
for name, make in [
    ("Pipeline(StandardScaler, Ridge)", lambda: make_pipeline(StandardScaler(), Ridge())),
    ("Pipeline(StandardScaler, LinearRegression)",
     lambda: make_pipeline(StandardScaler(), LinearRegression())),
]:
    for n_rows, lags in [(1_000, 24), (20_000, 24), (20_000, 168), (200_000, 168)]:
        y = b03.make_y(n_rows, lags)
        res = b03.run_case(
            "recursive", name,
            lambda: ForecasterRecursive(estimator=make(), lags=lags), y,
        )
        print(f"| {name} | {n_rows:,} x {lags} | {res['min_off_ms']:.2f} | "
              f"{res['min_on_ms']:.2f} | {res['ratio_min']:.3f} | "
              f"{res['max_abs_diff']:.1e} |", flush=True)
