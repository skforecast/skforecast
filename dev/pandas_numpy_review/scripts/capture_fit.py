import sys, warnings, numpy as np, pandas as pd
warnings.simplefilter("ignore")
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate

captured = {}
def spy(est):
    return est
for cls in (Ridge, LGBMRegressor):
    def make(orig):
        def fit(self, X, y, **kw):
            captured['X'] = X; captured['kw'] = {k: v for k, v in kw.items() if k != 'sample_weight'}
            return orig(self, X, y, **kw)
        return fit
    cls.fit = make(cls.fit)

n = 200
idx = pd.date_range("2020", periods=n, freq="D")
rng = np.random.default_rng(0)
y = pd.Series(rng.normal(size=n), index=idx, name="y")
cases = {
 "float": pd.DataFrame({"a": rng.normal(size=n)}, index=idx),
 "float+int": pd.DataFrame({"a": rng.normal(size=n), "b": np.arange(n)}, index=idx),
 "float+bool": pd.DataFrame({"a": rng.normal(size=n), "b": np.arange(n) % 2 == 0}, index=idx),
 "float+cat": pd.DataFrame({"a": rng.normal(size=n), "c": pd.Categorical(np.array(list("xyz"))[np.arange(n) % 3])}, index=idx),
 "float+str": pd.DataFrame({"a": rng.normal(size=n), "c": np.array(list("xyz"))[np.arange(n) % 3]}, index=idx),
 "float+Int64": pd.DataFrame({"a": rng.normal(size=n), "b": pd.array(np.arange(n), dtype="Int64")}, index=idx),
}
def desc(X):
    if isinstance(X, pd.DataFrame):
        return f"DataFrame dtypes={sorted(set(map(str, X.dtypes)))}"
    return f"ndarray dtype={X.dtype}"
for est_name, mk in [("Ridge", lambda: Ridge()), ("LGBM", lambda: LGBMRegressor(verbose=-1, n_estimators=5))]:
    for name, ex in cases.items():
        out = []
        for fname, mkf in [
            ("Rec", lambda e: ForecasterRecursive(e, lags=3)),
            ("Dir", lambda e: ForecasterDirect(e, lags=3, steps=2)),
            ("MS", lambda e: ForecasterRecursiveMultiSeries(e, lags=3)),
            ("DMV", lambda e: ForecasterDirectMultiVariate(e, level="s1", lags=3, steps=2)),
        ]:
            f = mkf(spy(mk()))
            try:
                if fname == "Rec" or fname == "Dir":
                    f.fit(y=y, exog=ex)
                elif fname == "MS":
                    f.fit(series=pd.DataFrame({"s1": y, "s2": y * 2}), exog=ex)
                else:
                    f.fit(series=pd.DataFrame({"s1": y, "s2": y * 2}), exog=ex)
                # for Direct, estimator is cloned per step: inspect estimators_
                if True:
                    out.append(f"{fname}: {desc(captured['X'])} kw={list(captured['kw'])}")
            except Exception as e:
                out.append(f"{fname}: ERR {type(e).__name__}: {str(e)[:80]}")
        print(est_name, name); [print("   ", o) for o in out]
