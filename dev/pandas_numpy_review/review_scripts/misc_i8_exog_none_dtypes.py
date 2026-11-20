"""I8: multiseries, one series without exog -> dtypes after the concat."""
import _env
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries

n = 30
idx = pd.date_range("2020-01-01", periods=n, freq="D")
fut = pd.date_range(idx[-1], periods=4, freq="D")[1:]
series = {k: pd.Series(np.arange(n, dtype=float) + i, index=idx, name=k) for i, k in enumerate("abc")}


def make_exog(index, kind):
    m = len(index)
    col = {
        "int": np.arange(m, dtype="int64"),
        "bool": np.arange(m) % 2 == 0,
        "category": pd.Categorical(np.arange(m) % 3),
        "Int64": pd.array(np.arange(m), dtype="Int64"),
        "str": np.array(["x", "y", "z"])[np.arange(m) % 3],
    }[kind]
    return pd.DataFrame({"e": col}, index=index)


def short(d):
    return {k: str(v) for k, v in d.items()} if d is not None else None


for kind in ("int", "bool", "category", "Int64", "str"):
    for cat_feat in (None, "auto"):
        if kind == "str" and cat_feat is None:
            continue
        for label, missing in (("all series with exog", ()), ("series 'c' = None", ("c",))):
            exog = {k: (None if k in missing else make_exog(idx, kind)) for k in series}
            exog_p = {k: make_exog(fut, kind) for k in series}  # all levels have exog at predict
            est = LGBMRegressor(verbose=-1, n_estimators=5, min_child_samples=2)
            f = ForecasterRecursiveMultiSeries(est, lags=3, categorical_features=cat_feat)
            print(f"## {kind:8s} categorical_features={cat_feat!s:5s} {label}")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    X, _ = f.create_train_X_y(series=series, exog=exog)
                    print("   X_train['e'] dtype:", X["e"].dtype)
                    f.fit(series=series, exog=exog)
                    print("   in :", short(f.exog_dtypes_in_), "| out:", short(f.exog_dtypes_out_))
                    wfit = sorted({x.category.__name__ for x in w}); w.clear()
                    p = f.predict(steps=3, exog=exog_p)
                    print("   fit warnings:", wfit, "| predict warnings:", sorted({x.category.__name__ for x in w}))
                    print("   pred head:", np.round(p["pred"].to_numpy()[:3], 6))
                except Exception as e:
                    print(f"   {type(e).__name__}: {str(e).splitlines()[0][:110]}")
print(_env.versions())
