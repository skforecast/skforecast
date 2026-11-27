"""T9: release-note claims: special characters in names, copy_X=False with
non-float exog, Pipeline that selects columns by name, feature_names_in_."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
banner()
from skforecast.recursive import ForecasterRecursiveMultiSeries
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

series, exog_train, exog_pred = make_case("df", ["f", "i"], exog_kind="df")
s2 = series.rename(columns={"s0": "a[1]", "s1": "b<2>"})
e2 = exog_train.rename(columns={"f": "temp[C]"}); e2p = exog_pred.rename(columns={"f": "temp[C]"})
for est in [lambda: LGBMRegressor(n_estimators=5, verbose=-1), lambda: XGBRegressor(n_estimators=5)]:
    for enc in ["onehot", "ordinal"]:
        f = ForecasterRecursiveMultiSeries(est(), lags=3, encoding=enc)
        st, r, w = run(lambda: f.fit(series=s2, exog=e2))
        print("special names", type(f.estimator).__name__, enc, st, r[:90] if st == "err" else "")

for ex_kind, ex, exp in [("float exog", exog_train.astype({"i": float}), None), ("int exog", exog_train, None)]:
    f = ForecasterRecursiveMultiSeries(LinearRegression(copy_X=False), lags=3,
                                       transformer_series=StandardScaler())
    st, r, w = run(lambda: f.fit(series=series, exog=ex))
    print("copy_X=False", ex_kind, st, r[:120] if st == "err" else "")

ct = make_column_transformer((StandardScaler(), ["f"]), remainder="passthrough")
f = ForecasterRecursiveMultiSeries(make_pipeline(ct, LinearRegression()), lags=3)
st, r, w = run(lambda: f.fit(series=series, exog=exog_train))
print("Pipeline selects by name: fit", st, r[:120] if st == "err" else "")
if st == "ok":
    st, r, w = run(lambda: f.predict(3, exog=exog_pred))
    print("   predict", st, r[:120] if st == "err" else "")

f = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
f.fit(series=series, exog=exog_train, suppress_warnings=True)
print("feature_names_in_ present:", hasattr(f.estimator, "feature_names_in_"))
f = ForecasterRecursiveMultiSeries(LGBMRegressor(n_estimators=5, verbose=-1), lags=3)
f.fit(series=series, exog=exog_train, suppress_warnings=True)
print("LGBM feature_name_[:5]:", f.estimator.feature_name_[:5])
print("get_feature_importances ok:", f.get_feature_importances().shape)
