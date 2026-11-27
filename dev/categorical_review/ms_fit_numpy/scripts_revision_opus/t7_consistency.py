"""T7: consistency with the other forecasters: user categorical config by name
with categorical_features=None, and category exog not managed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
banner()
from skforecast.recursive import ForecasterRecursiveMultiSeries, ForecasterRecursive
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

s = make_series("df")
ex = make_exog_frame(pd.date_range("2020-01-01", periods=125, freq="D"), ["f", "ci0"])
mk_lgbm = lambda **kw: LGBMRegressor(n_estimators=10, verbose=-1, **kw)
cases = [
    ("lgbm fit_kwargs by name", lambda: mk_lgbm(), {"categorical_feature": ["ci0"]}),
    ("lgbm ctor by name", lambda: mk_lgbm(categorical_feature=["ci0"]), None),
    ("hgb ctor by name", lambda: HistGradientBoostingRegressor(max_iter=10, categorical_features=["ci0"]), None),
    ("lgbm plain (category dtype)", lambda: mk_lgbm(), None),
]
for tag, mk, fk in cases:
    for cls in ["MultiSeries", "Recursive", "Direct", "DirectMultiVariate"]:
        if cls == "MultiSeries":
            f = ForecasterRecursiveMultiSeries(mk(), lags=3, categorical_features=None, fit_kwargs=fk)
            fit = lambda: f.fit(series=s, exog=ex.iloc[:120])
        elif cls == "Recursive":
            f = ForecasterRecursive(mk(), lags=3, categorical_features=None, fit_kwargs=fk)
            fit = lambda: f.fit(y=s["s0"], exog=ex.iloc[:120])
        elif cls == "Direct":
            f = ForecasterDirect(mk(), lags=3, steps=2, categorical_features=None, fit_kwargs=fk)
            fit = lambda: f.fit(y=s["s0"], exog=ex.iloc[:120])
        else:
            f = ForecasterDirectMultiVariate(mk(), level="s0", lags=3, steps=2, categorical_features=None, fit_kwargs=fk)
            fit = lambda: f.fit(series=s, exog=ex.iloc[:120])
        st, r, w = run(fit)
        print(f"{tag:30s} {cls:20s} fit={st} {r[:110] if st=='err' else ''} warns={w}")
