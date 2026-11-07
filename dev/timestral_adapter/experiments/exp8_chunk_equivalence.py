"""
PR C: predictions of `FoundationModel` with Timestral are the same with and
without chunks (`max_series_per_call`), up to float32 rounding, and time of
both paths through the package (padding and covariate building included).

230 daily series of 300 to 700 points with one known numeric covariate, 14
steps, `Timestral-28M`. The attribute of the adapter instance is set to
`None` (one call), 50 and 100. Run from the repository root, with the
environment that has `timestral` installed and `PYTHONPATH` set to the root:
    python -W ignore dev/timestral_adapter/experiments/exp8_chunk_equivalence.py cpu > exp8_cpu.log
    python -W ignore dev/timestral_adapter/experiments/exp8_chunk_equivalence.py cuda > exp8_cuda.log
"""
import sys
import time
import numpy as np
import pandas as pd
import skforecast
from skforecast.foundation import FoundationModel

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
STEPS = 14
rng = np.random.default_rng(8)
print(f"skforecast {skforecast.__version__} from {skforecast.__file__}, device {DEVICE}")

series, exog_hist, exog_future = {}, {}, {}
for i in range(230):
    length = int(rng.integers(300, 701))
    index = pd.date_range(end="2024-12-31", periods=length, freq="D")
    future = pd.date_range("2025-01-01", periods=STEPS, freq="D")
    t = np.arange(length + STEPS)
    temp = 15 + 8 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 1, length + STEPS)
    y = 50 + 0.8 * temp[:length] + rng.normal(0, 3, length).cumsum() * 0.1
    series[f"s{i}"] = pd.Series(y, index=index)
    exog_hist[f"s{i}"] = pd.DataFrame({"temp": temp[:length]}, index=index)
    exog_future[f"s{i}"] = pd.DataFrame({"temp": temp[length:]}, index=future)


def run(max_series_per_call):
    model = FoundationModel(
        "mistralai/Timestral-28M", context_length=4096, device=DEVICE
    )
    model.adapter.max_series_per_call = max_series_per_call
    model.fit(series=series, exog=exog_hist)
    model.predict(steps=STEPS, exog=exog_future, levels=["s0", "s1"])  # warm-up
    t0 = time.perf_counter()
    predictions = model.predict(
        steps=STEPS, exog=exog_future, quantiles=[0.1, 0.5, 0.9]
    )
    return predictions, time.perf_counter() - t0


single, t_single = run(None)
values_single = single[["q_0.1", "q_0.5", "q_0.9"]].to_numpy()
scale = np.abs(values_single).max()
print(f"one call: {t_single:.2f} s")
for chunk in [50, 100]:
    chunked, t_chunked = run(chunk)
    pd.testing.assert_index_equal(chunked.index, single.index)
    assert chunked["level"].tolist() == single["level"].tolist()
    values_chunked = chunked[["q_0.1", "q_0.5", "q_0.9"]].to_numpy()
    abs_diff = np.abs(values_chunked - values_single)
    print(
        f"chunks of {chunk}: {t_chunked:.2f} s; max abs diff {abs_diff.max():.2e}, "
        f"relative to the largest prediction {abs_diff.max() / scale:.2e}, "
        f"median abs diff {np.median(abs_diff):.2e}"
    )
