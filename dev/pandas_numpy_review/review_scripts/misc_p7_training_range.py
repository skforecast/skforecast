"""P7: v[[0, -1]] vs v[np.array([0, -1])] on an Index."""
import _env
import pickle
import numpy as np
import pandas as pd

d = pd.date_range("2020-01-01", periods=10, freq="D")
cases = {
    "DatetimeIndex freq=D": d,
    "DatetimeIndex freq=None": pd.DatetimeIndex(d.to_numpy()),
    "DatetimeIndex tz=UTC": d.tz_localize("UTC"),
    "DatetimeIndex tz=Madrid, named": d.tz_localize("Europe/Madrid").rename("date"),
    "DatetimeIndex unit=s": d.as_unit("s"),
    "DatetimeIndex freq=MS": pd.date_range("2020-01-01", periods=10, freq="MS"),
    "DatetimeIndex freq=h, len 1": pd.date_range("2020-01-01", periods=1, freq="h"),
    "DatetimeIndex len 2 freq=D": d[:2],
    "RangeIndex(10)": pd.RangeIndex(10),
    "RangeIndex(5, 50, 5) named": pd.RangeIndex(5, 50, 5, name="pos"),
    "RangeIndex len 1": pd.RangeIndex(1),
    "RangeIndex len 2": pd.RangeIndex(3, 5),
    "Index int64": pd.Index([3, 4, 5, 6]),
}
pos = np.array([0, -1])
for name, v in cases.items():
    a, b = v[[0, -1]], v[pos]
    c = v[np.array([0, -1], dtype=np.intp)]
    print(
        f"{name:32s} list->{type(a).__name__}/{a.dtype}/freq={getattr(a, 'freq', '-')!s:6s} "
        f"array->{type(b).__name__}/{b.dtype}/freq={getattr(b, 'freq', '-')!s:6s} "
        f"| identical={a.identical(b)} same type={type(a) is type(b)} "
        f"same freq={getattr(a, 'freq', None) == getattr(b, 'freq', None)} "
        f"same repr={repr(a) == repr(b)} same pickle={pickle.dumps(a) == pickle.dumps(b)} "
        f"to_list eq={a.to_list() == b.to_list()} intp ok={a.identical(c)}"
    )
print(_env.versions())
