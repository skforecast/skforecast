"""P8: in the multiseries fit with a wide `exog`, is `exog_dict[k]` the same
object for every series when `exog_dict[k].iloc[window_size:]` is taken?"""
import _env
import warnings
import numpy as np
import pandas as pd
from skforecast.utils import check_preprocess_exog_multiseries, align_series_and_exog_multiseries

warnings.simplefilter("ignore")
n = 50
idx = pd.date_range("2020-01-01", periods=n, freq="D")
rng = np.random.default_rng(0)
exog = pd.DataFrame({"e1": rng.normal(size=n), "e2": rng.normal(size=n)}, index=idx)
full = {k: pd.Series(rng.normal(size=n), index=idx, name=k) for k in "abc"}

lead_nan = {k: v.copy() for k, v in full.items()}; lead_nan["b"].iloc[:5] = np.nan
inner_nan = {k: v.copy() for k, v in full.items()}; inner_nan["b"].iloc[20:25] = np.nan
shorter = dict(full); shorter["c"] = full["c"].iloc[10:]
cases = {
    "all series with the index of exog": (full, exog),
    "series 'b' with leading NaN (trimmed)": (lead_nan, exog),
    "series 'b' with interior NaN": (inner_nan, exog),
    "series 'c' starts later": (shorter, exog),
    "exog longer than the series": ({k: v.iloc[:40] for k, v in full.items()}, exog),
    "exog with index not sorted": (full, exog.iloc[::-1]),
    "dict exog built with the same DataFrame": (full, {k: exog for k in "abc"}),
}
for name, (series, ex) in cases.items():
    series_dict = {k: v.copy() for k, v in series.items()}
    exog_dict, _ = check_preprocess_exog_multiseries(
        series_names_in_=list(series_dict), series_index_type=pd.DatetimeIndex,
        exog=ex, exog_dict={k: None for k in series_dict})
    ids_before = {id(v) for v in exog_dict.values()}
    series_dict, exog_dict = align_series_and_exog_multiseries(series_dict=series_dict, exog_dict=exog_dict)
    ids_after = [id(v) for v in exog_dict.values()]
    lens = [len(v) for v in exog_dict.values()]
    print(f"{name:42s} distinct objects after check={len(ids_before)} after align={len(set(ids_after))}"
          f" | rows per series={lens} | is the user's object={any(v is ex for v in exog_dict.values())}")
print(_env.versions())
