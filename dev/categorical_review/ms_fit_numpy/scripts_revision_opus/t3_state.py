"""T3: user categorical config on estimators, state between fits, warnings,
in-sample residuals vs estimator predict, exception mid-fit."""
import sys, os, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
banner()
from skforecast.recursive import ForecasterRecursiveMultiSeries
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

series, exog_train, exog_pred = make_case("df", ["f", "ci0", "i"], exog_kind="df")
# ci0: codes == values, so base is right: compare base vs branch on user config
def show(tag, f, fn_fit, pred=True):
    st, r, w = run(fn_fit)
    msg = f"{tag:60s} fit={st} {r if st=='err' else ''} warn={w}"
    if st == "ok" and pred:
        st2, p, w2 = run(lambda: f.predict(5, exog=exog_pred))
        msg += f" pred={np.round(p['pred'].to_numpy()[:4], 6).tolist() if st2=='ok' else p} w2={w2}"
        try:
            msg += f" cat_params={ {k: v for k, v in f.estimator.get_params().items() if k in ('feature_types','enable_categorical','categorical_features','categorical_feature','cat_features')} }"
        except Exception as e:
            msg += f" params err {e}"
    print(msg, flush=True)

cfgs = [
    ("lgbm fit_kwargs names ci0", lambda: LGBMRegressor(n_estimators=20, verbose=-1, random_state=1), {"categorical_feature": ["ci0"]}),
    ("lgbm fit_kwargs pos [5]", lambda: LGBMRegressor(n_estimators=20, verbose=-1, random_state=1), {"categorical_feature": [5]}),
    ("lgbm ctor names ci0", lambda: LGBMRegressor(n_estimators=20, verbose=-1, random_state=1, categorical_feature=["ci0"]), None),
    ("lgbm ctor 'name:ci0'", lambda: LGBMRegressor(n_estimators=20, verbose=-1, random_state=1, categorical_feature="name:ci0"), None),
    ("lgbm ctor alias cat_feature=[5]", lambda: LGBMRegressor(n_estimators=20, verbose=-1, random_state=1, cat_feature=[5]), None),
    ("lgbm default (category dtype auto)", lambda: LGBMRegressor(n_estimators=20, verbose=-1, random_state=1), None),
    ("cat fit_kwargs names ci0", lambda: CatBoostRegressor(iterations=20, verbose=0, random_seed=1, allow_writing_files=False, thread_count=1), {"cat_features": ["ci0"]}),
    ("cat ctor names ci0", lambda: CatBoostRegressor(iterations=20, verbose=0, random_seed=1, allow_writing_files=False, thread_count=1, cat_features=("ci0",)), None),
    ("cat ctor pos [5]", lambda: CatBoostRegressor(iterations=20, verbose=0, random_seed=1, allow_writing_files=False, thread_count=1, cat_features=(5,)), None),
    ("xgb feature_types", lambda: XGBRegressor(n_estimators=20, random_state=1, enable_categorical=True, feature_types=["q","q","q","q","q","c","q"]), None),
    ("xgb enable_categorical only", lambda: XGBRegressor(n_estimators=20, random_state=1, enable_categorical=True), None),
    ("hgb names", lambda: HistGradientBoostingRegressor(max_iter=20, random_state=1, categorical_features=["ci0"]), None),
    ("hgb bool mask", lambda: HistGradientBoostingRegressor(max_iter=20, random_state=1, categorical_features=[False]*5+[True, False]), None),
    ("hgb from_dtype", lambda: HistGradientBoostingRegressor(max_iter=20, random_state=1), None),
]
for enc in ["ordinal", "ordinal_category"]:
    for tag, mk, fk in cfgs:
        if "pos [4]" in tag or "feature_types" in tag or "bool mask" in tag:
            # positions computed for ordinal (lags 0-2, level 3, f 4?) -> recompute
            pass
        f = ForecasterRecursiveMultiSeries(mk(), lags=3, encoding=enc, categorical_features=None, fit_kwargs=fk)
        show(f"{enc} | {tag}", f, lambda: f.fit(series=series, exog=exog_train))

print("names:", ForecasterRecursiveMultiSeries(LGBMRegressor(verbose=-1), lags=3).fit(series=series, exog=exog_train, suppress_warnings=True) or "")

# ---- state: set_lags + second fit with XGB/HGB + ordinal_category + None
for mk in [lambda: XGBRegressor(n_estimators=10, random_state=1),
           lambda: HistGradientBoostingRegressor(max_iter=10, random_state=1)]:
    f = ForecasterRecursiveMultiSeries(mk(), lags=3, encoding="ordinal_category", categorical_features=None)
    f.fit(series=series, exog=exog_train, suppress_warnings=True, store_in_sample_residuals=True)
    p1 = f.predict(3, exog=exog_pred, suppress_warnings=True)["pred"].to_numpy()
    f.set_lags(5); f.fit(series=series, exog=exog_train, suppress_warnings=True, store_in_sample_residuals=True)
    f.set_lags(3); f.fit(series=series, exog=exog_train, suppress_warnings=True, store_in_sample_residuals=True)
    p2 = f.predict(3, exog=exog_pred, suppress_warnings=True)["pred"].to_numpy()
    g = ForecasterRecursiveMultiSeries(mk(), lags=3, encoding="ordinal_category", categorical_features=None)
    g.fit(series=series, exog=exog_train, suppress_warnings=True)
    p3 = g.predict(3, exog=exog_pred, suppress_warnings=True)["pred"].to_numpy()
    est = type(f.estimator).__name__
    # in-sample residuals vs the booster/model prediction on the training matrix
    X, y = f.create_train_X_y(series=series, exog=exog_train, suppress_warnings=True)
    Xn = X.astype(float).to_numpy() if hasattr(X, 'to_numpy') else X
    yp = f.estimator.predict(Xn)
    res_all = np.concatenate([np.asarray(v) for k, v in f.in_sample_residuals_.items() if k != '_unknown_level'])
    print(est, "refit-stable", np.array_equal(p1, p2), "fresh==refit", np.array_equal(p1, p3),
          "params after", {k: v for k, v in f.estimator.get_params().items() if k in ("feature_types", "enable_categorical", "categorical_features")},
          "| residual vs predict(numpy) max|d|:",
          float(np.max(np.abs(np.sort(res_all) - np.sort((y.to_numpy() - yp))))) )
    if est == "XGBRegressor":
        yb = f.estimator.get_booster().inplace_predict(Xn)
        print("   XGB estimator.predict vs booster.inplace_predict max|d|:", float(np.max(np.abs(yp - yb))))
        # Force the non-inplace path of XGBRegressor.predict (DMatrix built from get_params)
        import xgboost as xgb
        dm = xgb.DMatrix(Xn, feature_types=f.estimator.get_params()["feature_types"],
                         enable_categorical=f.estimator.get_params()["enable_categorical"])
        yd = f.estimator.get_booster().predict(dm)
        print("   XGB DMatrix(params after restore) predict vs inplace max|d|:", float(np.max(np.abs(yd - yb))))

# ---- forecaster.set_params changes estimator params between fits
f = ForecasterRecursiveMultiSeries(XGBRegressor(n_estimators=10), lags=3, encoding="ordinal_category", categorical_features=None)
f.fit(series=series, exog=exog_train, suppress_warnings=True)
f.set_params({"max_depth": 2})
st, r, w = run(lambda: f.fit(series=series, exog=exog_train, suppress_warnings=True))
print("xgb set_params refit", st, r if st == "err" else "")
