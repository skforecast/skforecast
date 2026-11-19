"""P9: `nunique` over a DataFrame of dtypes (today) vs comparing dtype objects
with `==` vs comparing `str(dtype)`. Also the real check with two series."""
import _env
import warnings
import numpy as np
import pandas as pd
from skforecast.utils import check_preprocess_exog_multiseries

warnings.simplefilter("ignore")
CD = pd.CategoricalDtype


def sdtype(storage, na):
    try:
        return pd.StringDtype(storage, na_value=na)
    except Exception as e:  # no pyarrow, or pandas without `na_value`
        return None


pairs = {
    "category same cats": (CD(["a", "b"]), CD(["a", "b"])),
    "category other order (unordered)": (CD(["a", "b"]), CD(["b", "a"])),
    "category other order (ordered)": (CD(["a", "b"], ordered=True), CD(["b", "a"], ordered=True)),
    "category ordered vs unordered": (CD(["a", "b"], ordered=True), CD(["a", "b"])),
    "category extra category": (CD(["a", "b"]), CD(["a", "b", "c"])),
    "category int vs float cats (1 == 1.0)": (CD([1, 2]), CD([1.0, 2.0])),
    "category int64 vs int32 cats": (CD(pd.Index([1, 2], dtype="int64")), CD(pd.Index([1, 2], dtype="int32"))),
    "category str vs object cats": (CD(pd.Index(["a", "b"], dtype=object)), CD(pd.Index(["a", "b"]))),
    "int64 vs Int64": (np.dtype("int64"), pd.Int64Dtype()),
    "int64 vs int32": (np.dtype("int64"), np.dtype("int32")),
    "float64 vs Float64": (np.dtype("float64"), pd.Float64Dtype()),
    "datetime64[ns] vs [us]": (np.dtype("M8[ns]"), np.dtype("M8[us]")),
    "datetime tz UTC vs naive": (pd.DatetimeTZDtype("ns", "UTC"), np.dtype("M8[ns]")),
    "object vs str(python, nan)": (np.dtype(object), sdtype("python", np.nan)),
    "str(python, nan) vs str(pyarrow, nan)": (sdtype("python", np.nan), sdtype("pyarrow", np.nan)),
    "str(python, nan) vs string(python, NA)": (sdtype("python", np.nan), sdtype("python", pd.NA)),
    "string(python, NA) vs string(pyarrow, NA)": (sdtype("python", pd.NA), sdtype("pyarrow", pd.NA)),
    "bool vs boolean": (np.dtype(bool), pd.BooleanDtype()),
}
is_text = pd.api.types.is_string_dtype
print(f"{'pair':44s} nunique  a==b   str==  hash==  today(after text rule)")
for name, (a, b) in pairs.items():
    if a is None or b is None:
        print(f"{name:44s} (dtype not available in this environment)")
        continue
    nun = pd.DataFrame({"s1": pd.Series({"e": a}), "s2": pd.Series({"e": b})}).nunique(axis=1)["e"]
    nun_text = pd.DataFrame({k: pd.Series({"e": d}).map(lambda x: "text" if is_text(x) else x)
                             for k, d in (("s1", a), ("s2", b))}).nunique(axis=1)["e"]
    eq = bool(a == b)
    flag = "" if (nun == 1) == eq else "  <-- `==` and nunique disagree"
    print(f"{name:44s} {nun:^7d}  {eq!s:5s}  {str(a) == str(b)!s:5s}  {hash(a) == hash(b)!s:5s}   "
          f"{'accept' if nun_text == 1 else 'TypeError'}{flag}")

print("\n--- real function: two series, one exog column ---")
idx = pd.date_range("2020", periods=4, freq="D")


def real(col1, col2):
    exog = {"s1": pd.DataFrame({"e": col1}, index=idx), "s2": pd.DataFrame({"e": col2}, index=idx)}
    try:
        check_preprocess_exog_multiseries(
            series_names_in_=["s1", "s2"], series_index_type=pd.DatetimeIndex,
            exog=exog, exog_dict={"s1": None, "s2": None})
        return "accept"
    except TypeError as e:
        return "TypeError"


C = pd.Categorical
data = {
    "category other order": (C(list("abab"), categories=["a", "b"]), C(list("abab"), categories=["b", "a"])),
    "category extra unused category": (C(list("abab"), categories=["a", "b"]), C(list("abab"), categories=["a", "b", "c"])),
    "text vs all-missing text column": (["a", "b", "a", "b"], [None, None, None, None]),
    "int64 vs float64 (a NaN in one series)": ([1, 2, 3, 4], [1, 2, np.nan, 4]),
    "different column sets (e only in s1)": None,
}
for name, cols in data.items():
    if cols is None:
        exog = {"s1": pd.DataFrame({"e": [1, 2, 3, 4], "g": 1.0}, index=idx), "s2": pd.DataFrame({"g": [1.0] * 4}, index=idx)}
        try:
            check_preprocess_exog_multiseries(series_names_in_=["s1", "s2"], series_index_type=pd.DatetimeIndex,
                                              exog=exog, exog_dict={"s1": None, "s2": None})
            r = "accept"
        except TypeError:
            r = "TypeError"
        naive = exog["s1"].dtypes.tolist() == exog["s2"].dtypes.tolist()
        print(f"{name:44s} today={r}  naive whole-list comparison equal={naive}")
    else:
        print(f"{name:44s} today={real(*cols)}")
print(_env.versions())
