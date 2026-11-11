"""
BUG-07: the long-format MultiIndex input of ForecasterRecursiveMultiSeries
only works when the MultiIndex comes from `reshape_series_wide_to_long`.

A MultiIndex built by the user (`df.set_index(['series_id', 'date'])`, the
usual way to load long panel data from a CSV or a database) always fails in
`check_preprocess_series` because the second level of the MultiIndex never
carries a `freq`.

Run: python dev/roadmap_0.27.0/repro/BUG-07.py
"""

import io
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import skforecast
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster_multiseries,
)
from skforecast.preprocessing import (
    reshape_series_long_to_dict,
    reshape_series_wide_to_long,
)
from skforecast.recursive import ForecasterRecursiveMultiSeries

warnings.simplefilter("ignore")
print(f"skforecast {skforecast.__version__}, pandas {pd.__version__}\n")

# Synthetic panel: 3 daily series, 'b' starts later (different lengths)
rng = np.random.default_rng(123)
idx = pd.date_range("2024-01-01", periods=120, freq="D")
wide = pd.DataFrame(
    {
        "a": 10 + rng.normal(size=120).cumsum(),
        "b": 20 + rng.normal(size=120).cumsum(),
        "c": 30 + rng.normal(size=120).cumsum(),
    },
    index=idx,
)
wide.loc[: "2024-01-20", "b"] = np.nan

# Long CSV, as it typically arrives from a file or a database
long_csv = (
    wide.rename_axis("date")
    .reset_index()
    .melt(id_vars="date", var_name="series_id", value_name="sales")
    .dropna()
)
buffer = io.StringIO()
long_csv.to_csv(buffer, index=False)
buffer.seek(0)
df = pd.read_csv(buffer, parse_dates=["date"])


def make_forecaster():
    return ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=7)


def try_fit(label, series):
    level1 = series.index.levels[1]
    print(f"[{label}]")
    print(f"  index.levels[1].freq = {level1.freq!r}")
    try:
        f = make_forecaster()
        f.fit(series=series)
        pred = f.predict(steps=3)
        print(f"  fit/predict OK, predictions shape {pred.shape}")
        return pred
    except Exception as e:  # noqa: BLE001
        print(f"  {type(e).__name__}: {e}")
        return None
    finally:
        print()


# 1) Documented route: MultiIndex from reshape_series_wide_to_long -> OK
series_long_doc = reshape_series_wide_to_long(data=wide.copy())
pred_ref = try_fit("1. reshape_series_wide_to_long (guide)", series_long_doc)

# 2) User route: long CSV + set_index -> FAILS
series_long_user = df.set_index(["series_id", "date"])[["sales"]]
try_fit("2. read_csv + set_index(['series_id', 'date'])", series_long_user)

# 3) Round trip of the guide's own data (cell 10 shows reset_index()) -> FAILS
roundtrip = series_long_doc.reset_index().set_index(["series_id", "datetime"])
try_fit("3. guide data: reset_index() + set_index()", roundtrip)

# 4) Even after sorting the index -> FAILS
try_fit("4. set_index + sort_index()", series_long_user.sort_index())

# 5) backtesting_forecaster_multiseries with the user MultiIndex -> FAILS
print("[5. backtesting_forecaster_multiseries with set_index MultiIndex]")
try:
    cv = TimeSeriesFold(steps=7, initial_train_size=90, refit=False)
    backtesting_forecaster_multiseries(
        forecaster=make_forecaster(),
        series=series_long_user,
        cv=cv,
        metric="mean_absolute_error",
        show_progress=False,
    )
    print("  OK\n")
except Exception as e:  # noqa: BLE001
    print(f"  {type(e).__name__}: {e}\n")

# 6) Workaround A: dict via reshape_series_long_to_dict (needs freq) -> OK
series_dict = reshape_series_long_to_dict(
    data=df, series_id="series_id", index="date", values="sales", freq="D"
)
print("[6. workaround reshape_series_long_to_dict(..., freq='D')]")
f = make_forecaster()
f.fit(series=series_dict)
pred_dict = f.predict(steps=3)
print(f"  fit/predict OK, same predictions as route 1: "
      f"{np.allclose(pred_dict['pred'].to_numpy(), pred_ref['pred'].to_numpy())}\n")

# 7) Workaround B: set the freq of level 1 by hand -> OK (not discoverable)
fixed = series_long_user.copy()
fixed.index = fixed.index.set_levels(
    pd.DatetimeIndex(fixed.index.levels[1], freq="infer"), level=1
)
pred_fixed = try_fit("7. set_levels(..., freq='infer') by hand", fixed)
if pred_fixed is not None:
    print(f"  same predictions as route 1: "
          f"{np.allclose(pred_fixed['pred'].to_numpy(), pred_ref['pred'].to_numpy())}\n")

# 8) Proposed fix (per-group inferred freq + asfreq) on the user MultiIndex
print("[8. sketch of fix: infer freq per group before the freq check]")
series_dict_fix = {}
for sid, g in series_long_user.groupby(level=0, sort=True):
    s = g.iloc[:, 0].droplevel(0).rename(sid)
    s.index.name = None
    freq = s.index.freq or pd.infer_freq(s.index)
    series_dict_fix[sid] = s.asfreq(freq)
    print(f"  {sid}: inferred freq {freq!r}, len {len(s)}")
f = make_forecaster()
f.fit(series=series_dict_fix)
pred_fix = f.predict(steps=3)
print(f"  same predictions as route 1: "
      f"{np.allclose(pred_fix['pred'].to_numpy(), pred_ref['pred'].to_numpy())}\n")

print("OBSERVED: routes 2-5 (any user-built MultiIndex) raise ValueError, "
      "only route 1 (reshape_series_wide_to_long) works.")
print("EXPECTED: a long DataFrame with a (series_id, DatetimeIndex) MultiIndex "
      "and regular per-series dates fits like route 1 (frequency inferred per "
      "series), or fails with a MultiIndex-specific message pointing to "
      "reshape_series_long_to_dict(..., freq=...).")
