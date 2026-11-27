"""T1: the reported bug. category exog with values 10,20,30 not managed by the
forecaster. Compare predictions with the same exog passed as float (the
'right' numeric model), and with categorical_features='auto'."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
banner()
from skforecast.recursive import ForecasterRecursiveMultiSeries

series, exog_train, exog_pred = make_case("df", ["f", "ci"], effect_col="ci", exog_kind="df")
as_float = lambda e: e.assign(ci=e["ci"].astype(float))
for est in ["lgbm", "xgb", "cat", "hgb", "ridge", "ttr_lgbm"]:
    for enc in ["ordinal", "ordinal_category"]:
        for cf in [None, ["f"]]:
            f1 = ForecasterRecursiveMultiSeries(make_estimator(est), lags=3, encoding=enc, categorical_features=cf)
            f2 = ForecasterRecursiveMultiSeries(make_estimator(est), lags=3, encoding=enc, categorical_features=cf)
            st, r, w = run(lambda: f1.fit(series=series, exog=exog_train))
            if st == "err":
                print(est, enc, cf, "fit err", r); continue
            f2.fit(series=series, exog=as_float(exog_train), suppress_warnings=True)
            p1 = f1.predict(5, exog=exog_pred, suppress_warnings=True)["pred"].to_numpy()
            p2 = f2.predict(5, exog=as_float(exog_pred), suppress_warnings=True)["pred"].to_numpy()
            # truth: underlying signal of last obs + effect, just to see which is plausible
            print(f"{est:9s} {str(enc):17s} cf={str(cf):6s} equal_to_float_model={np.allclose(p1, p2)} "
                  f"max|diff|={np.max(np.abs(p1-p2)):.4f} warn={w}")
