# Prototype for doubt 2: can the public `create_train_X_y` DataFrame be rebuilt
# from a float64 array (numpy-first build) plus the stored dtypes?
import warnings
import numpy as np, pandas as pd
import skforecast; print(skforecast.__file__, "pandas", pd.__version__)
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.utils import _train_matrix_to_numpy
warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
n = 60
idx = pd.date_range("2020", periods=n, freq="D")
s = {"a": pd.Series(rng.normal(size=n), index=idx, name="a"),
     "b": pd.Series(rng.normal(size=n - 10), index=idx[10:], name="b"),
     "c": pd.Series(rng.normal(size=n - 20), index=idx[20:], name="c")}
int_na = pd.array(rng.integers(0, 5, size=n), dtype="Int64"); int_na[::9] = pd.NA
full = pd.DataFrame({
    "f64": rng.normal(size=n),
    "f32": rng.normal(size=n).astype("float32"),
    "i64": rng.integers(0, 5, size=n),
    "i8": rng.integers(0, 5, size=n).astype("int8"),
    "bool": rng.integers(0, 2, size=n).astype(bool),
    "Int64": int_na,
    "boolean": pd.array(rng.integers(0, 2, size=n).astype(bool), dtype="boolean"),
    "arrow_f": pd.array(rng.normal(size=n), dtype="double[pyarrow]"),
    "arrow_i": pd.array(rng.integers(0, 5, size=n), dtype="int64[pyarrow]"),
    "cat_int": pd.Categorical(rng.choice([10, 20, 30], size=n), categories=[10, 20, 30]),
    "txt": rng.choice(["x", "y"], size=n),
}, index=idx)
cases = {
    "all series same exog": {k: full.loc[v.index] for k, v in s.items()},
    "series c without exog": {"a": full, "b": full.loc[s["b"].index], "c": None},
    "series c lacks i64/bool/cat_int/Int64": {"a": full, "b": full.loc[s["b"].index],
        "c": full.loc[s["c"].index].drop(columns=["i64", "bool", "cat_int", "Int64"])},
}
for encoding in ["ordinal", "ordinal_category", "onehot", None]:
    for cname, exog in cases.items():
        for cf in ["auto", None]:
            ex = exog if cf == "auto" else {k: (None if v is None else v.drop(columns="txt")) for k, v in exog.items()}
            f = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=3, encoding=encoding,
                                               categorical_features=cf)
            X, y = f.create_train_X_y(series=s, exog=ex)
            try:
                arr = _train_matrix_to_numpy(X)
                X2 = pd.DataFrame(arr, columns=X.columns, index=X.index).astype(X.dtypes.to_dict())
                pd.testing.assert_frame_equal(X, X2)
                res = "round trip identical"
            except Exception as e:
                res = f"DIFFERENT: {type(e).__name__}: {str(e)[:160]}"
            dt = {c: str(d) for c, d in X.dtypes.items() if str(d) != "float64"}
            print(f"enc={str(encoding):17s} cf={str(cf):5s} {cname:38s} -> {res}")
            if cname.startswith("series c lacks") and encoding == "ordinal" :
                print("      non-float64 dtypes after concat:", dt)
