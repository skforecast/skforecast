"""
Cost of the proposed left padding in `FoundationModel` (`_left_pad_context`,
extracted from the report into skeleton/shared_changes.py) compared with the
numpy padding that T0 does today: 400 series of 500 to 4096 hourly points
with 2 exog columns and different end timestamps. Checks that the stacked
target and exog arrays are identical.

Run `python skeleton/extract_skeleton.py ../research_timestral_adapter.md`
first (from this folder: `cd skeleton && python extract_skeleton.py
../../research_timestral_adapter.md`).
"""
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "skeleton")
from shared_changes import _left_pad_context

rng = np.random.default_rng(0)
n_series, max_len = 400, 4096
context, cexog = {}, {}
for i in range(n_series):
    length = int(rng.integers(500, max_len + 1))
    end = pd.Timestamp("2014-12-01") - pd.Timedelta(hours=int(rng.integers(0, 500)))
    idx = pd.date_range(end=end, periods=length, freq="h")
    context[f"s{i}"] = pd.Series(rng.normal(size=length), index=idx)
    cexog[f"s{i}"] = pd.DataFrame(
        {"a": rng.normal(size=length), "b": rng.normal(size=length)}, index=idx
    )


def pad_numpy(context, cexog):
    width = max(len(s) for s in context.values())
    names = list(context)
    b = np.full((len(names), width), np.nan)
    e = np.full((len(names), 2, width), np.nan)
    for i, k in enumerate(names):
        v = context[k].to_numpy()
        b[i, width - len(v):] = v
        ev = cexog[k].to_numpy().T
        e[i, :, width - ev.shape[1]:] = ev
    return b, e


for name, f in (("_left_pad_context (pandas)", _left_pad_context), ("numpy (T0 today)", pad_numpy)):
    t = time.perf_counter()
    for _ in range(5):
        f(context, cexog)
    print(f"{name}: {(time.perf_counter() - t) / 5 * 1000:.1f} ms per call "
          f"(400 series, 500 to 4096 points, 2 exog columns)")

c2, e2 = _left_pad_context(context, cexog)
b1, e1 = pad_numpy(context, cexog)
b2 = np.stack([c2[k].to_numpy() for k in context])
e2 = np.stack([e2[k].to_numpy().T for k in context])
print("identical after stacking: target", np.array_equal(b1, b2, equal_nan=True),
      "| exog", np.array_equal(e1, e2, equal_nan=True),
      "| distinct end timestamps:", len({s.index[-1] for s in context.values()}))
