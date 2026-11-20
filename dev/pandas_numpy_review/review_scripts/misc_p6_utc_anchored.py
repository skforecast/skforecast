"""P6: `_is_utc_anchored_index` depends on the rows of the window, so a value
cached from another window (or from the whole index) is not always the same."""
import _env
import numpy as np
import pandas as pd
from skforecast.utils.utils import _is_utc_anchored_index, expand_index

tz = "Europe/Madrid"
cases = {
    "local midnight (calendar days)": pd.date_range("2021-01-01", periods=200, freq="D", tz=tz),
    "UTC midnight shown in Madrid": pd.date_range("2021-01-01", periods=200, freq="D", tz="UTC").tz_convert(tz),
    "06:00 UTC shown in Madrid": pd.date_range("2021-01-01 06:00", periods=200, freq="D", tz="UTC").tz_convert(tz),
    "UTC index": pd.date_range("2021-01-01", periods=200, freq="D", tz="UTC"),
    "fixed offset +02:00, UTC midnight": pd.date_range("2021-01-01", periods=200, freq="D", tz="UTC").tz_convert("Etc/GMT-2"),
}
ws = 14
for name, idx in cases.items():
    whole = _is_utc_anchored_index(idx, idx.freq)
    per_window = [_is_utc_anchored_index(idx[i:i + ws], idx.freq) for i in range(0, len(idx) - ws, 7)]
    # effect on the output of expand_index if the whole-index answer were reused
    n_diff = 0
    for i in range(0, len(idx) - ws - 5, 7):
        w = idx[i:i + ws]
        cur = expand_index(w, steps=5)
        truth = idx[i + ws:i + ws + 5]
        n_diff += int(not cur.equals(pd.DatetimeIndex(truth, freq=None)) and not np.array_equal(cur.asi8, truth.asi8))
    print(f"{name:36s} whole index={whole!s:5s} per window: True x{sum(per_window)}, False x{len(per_window) - sum(per_window)}"
          f" | windows where today's expand_index != real continuation of the index: {n_diff}")

print("\nexact cheaper formulation (integers, no normalize): same answer?")


def fast(index, freq):
    if index.tz is None or len(index) == 0:
        return False
    if isinstance(freq, pd.offsets.Tick) and not isinstance(freq, pd.offsets.Day):
        return False
    day = 86_400 * {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}[index.unit]
    t_utc = index.asi8 % day
    if (t_utc != t_utc[0]).any():
        return False
    t_loc = index.tz_localize(None).asi8 % day
    if (t_loc != t_loc[0]).any():
        return True
    return bool(t_utc[0] == 0 and t_loc[0] != 0)


ok = True
for name, idx in cases.items():
    for i in range(0, len(idx) - ws):
        w = idx[i:i + ws]
        ok &= fast(w, idx.freq) == _is_utc_anchored_index(w, idx.freq)
    for unit in ("s", "ms", "us", "ns"):
        ok &= fast(idx.as_unit(unit), idx.freq) == _is_utc_anchored_index(idx.as_unit(unit), idx.freq)
pre = pd.date_range("1960-01-01", periods=400, freq="D", tz=tz)  # negative epoch values
ok &= all(fast(pre[i:i + ws], pre.freq) == _is_utc_anchored_index(pre[i:i + ws], pre.freq) for i in range(380))
print("fast == current on all windows/units:", ok)
print(_env.versions())
