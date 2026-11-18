"""
Review of T1: scaler calls that do not go through `transform_numpy`. The
one-step-ahead validation of `ForecasterRecursiveMultiSeries` calls
`transformer.inverse_transform` with a DataFrame, twice per series and
candidate (`skforecast/model_selection/_utils.py`, lines 3046 and 3049), so a
shortcut inside `transform_numpy` does not reach it.

python dev/preprocessing_speedups/review_t1_one_step_ahead.py --size M
"""

from __future__ import annotations

import argparse
import time
import warnings

from data import SIZES, env_report, make_data, make_forecaster
from review_t1_common import ORIG_TRANSFORM_NUMPY, patched
from sklearn.preprocessing import StandardScaler

from skforecast.model_selection import OneStepAheadFold, grid_search_forecaster_multiseries

warnings.filterwarnings("ignore")


class Timer:
    def __init__(self, func):
        self.func, self.total, self.calls = func, 0.0, 0

    def __call__(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return self.func(*args, **kwargs)
        finally:
            self.total += time.perf_counter() - t0
            self.calls += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="M")
    args = ap.parse_args()
    print(env_report())
    s, e = make_data(**SIZES[args.size])
    n_span = max(len(v) for v in s.values())
    forecaster = make_forecaster("ridge")
    cv = OneStepAheadFold(initial_train_size=n_span - 168)

    # Calls made through `transform_numpy` are counted apart from the calls
    # to the method of the class made from outside it.
    in_numpy = Timer(ORIG_TRANSFORM_NUMPY)
    original = StandardScaler.inverse_transform
    direct = Timer(original)
    inside = [False]

    def counted_transform_numpy(*args, **kwargs):
        inside[0] = True
        try:
            return in_numpy(*args, **kwargs)
        finally:
            inside[0] = False

    def spy(self, X, copy=None):
        if inside[0]:
            return original(self, X, copy=copy)
        return direct(self, X, copy=copy)

    StandardScaler.inverse_transform = spy
    try:
        with patched(counted_transform_numpy):
            t0 = time.perf_counter()
            grid_search_forecaster_multiseries(
                forecaster, series=s, exog=e, cv=cv, param_grid={"alpha": [0.1, 1.0, 10.0]},
                lags_grid=[48], metric="mean_absolute_error", return_best=False,
                show_progress=False, verbose=False, suppress_warnings=True)
            total = time.perf_counter() - t0
    finally:
        StandardScaler.inverse_transform = original
    print(f"\ngrid_search_forecaster_multiseries, OneStepAheadFold, size {args.size} "
          f"({len(s)} series), Ridge, 3 candidates: {total:.2f} s")
    print(f"calls to transform_numpy: {in_numpy.calls} ({in_numpy.total:.3f} s)")
    print(f"direct calls to StandardScaler.inverse_transform: {direct.calls}, "
          f"{direct.total:.3f} s ({direct.total / total:.1%} of the search, "
          f"{direct.total / max(direct.calls, 1) * 1e6:.0f} us per call)")


if __name__ == "__main__":
    main()
