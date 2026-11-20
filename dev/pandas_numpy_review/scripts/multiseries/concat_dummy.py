import warnings, numpy as np, pandas as pd
idx = pd.date_range("2020-01-01", periods=5, freq="D")
a = pd.DataFrame({"e1": np.arange(5.), "c1": pd.Categorical(list("abcab"), categories=list("abc")), "s": list("xyzxy"), "i": np.arange(5)}, index=idx)
d = pd.Series(np.nan, index=idx, name="_dummy")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    r = pd.concat([a, d], axis=0).drop(columns="_dummy")
print(pd.__version__, dict(r.dtypes.astype(str)), [str(x.message)[:80] for x in w])
