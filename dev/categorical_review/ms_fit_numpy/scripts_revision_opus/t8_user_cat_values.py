"""T8: categorical_features=None and a `category` column configured by the user
as a native categorical feature (by name). The branch passes its VALUES. Check
relabel invariance (a categorical model must not depend on the labels) with
negative and large integer categories."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
banner()
from skforecast.recursive import ForecasterRecursiveMultiSeries
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

series, exog_train, exog_pred = make_case("df", ["f", "ci"], effect_col="ci", exog_kind="df")
def relabel(e, labels):
    return e.assign(ci=pd.Categorical(np.array(labels)[e["ci"].cat.codes.to_numpy()]))
LABELS = {"codes 0,1,2": [0, 1, 2], "10,20,30": [10, 20, 30], "neg -5,0,7": [-5, 0, 7],
          "large 10^6": [1_000_000, 2_000_000, 3_000_000]}
for name, mk, fk in [
    ("lgbm fit_kwargs", lambda: LGBMRegressor(n_estimators=20, verbose=-1, random_state=1, min_data_per_group=5, cat_smooth=1), {"categorical_feature": ["ci"]}),
    ("xgb feature_types", lambda: XGBRegressor(n_estimators=20, random_state=1, enable_categorical=True, feature_types=["q"] * 5 + ["c"]), None),
    ("hgb names", lambda: HistGradientBoostingRegressor(max_iter=20, random_state=1, categorical_features=["ci"]), None),
]:
    ref = None
    for lab_name, labels in LABELS.items():
        f = ForecasterRecursiveMultiSeries(mk(), lags=3, encoding="ordinal", categorical_features=None, fit_kwargs=fk)
        st, r, w = run(lambda: f.fit(series=series, exog=relabel(exog_train, labels)))
        if st == "err":
            print(f"{name:18s} {lab_name:12s} fit ERR {r[:150]}"); continue
        st, p, w2 = run(lambda: f.predict(5, exog=relabel(exog_pred, labels))["pred"].to_numpy())
        if st == "err":
            print(f"{name:18s} {lab_name:12s} predict ERR {p[:150]}"); continue
        if ref is None:
            ref = p
        print(f"{name:18s} {lab_name:12s} same_as_codes={np.allclose(p, ref)} max|d|={np.max(np.abs(p-ref)):.4f} warns={sorted(set(w)|set(w2))}")
