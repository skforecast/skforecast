"""T1b: was the base wrong for each estimator? A model that reads a category
consistently must give the same predictions when its categories are relabeled
(10,20,30 -> 0,1,2) without changing the order."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
banner()
from skforecast.recursive import ForecasterRecursiveMultiSeries

series, exog_train, exog_pred = make_case("df", ["f", "ci"], effect_col="ci", exog_kind="df")
relabel = lambda e: e.assign(ci=pd.Categorical(e["ci"].cat.codes.to_numpy()))
for est in ["lgbm", "xgb", "hgb", "ridge", "ttr_lgbm"]:
    for enc in ["ordinal", "ordinal_category"]:
        for cf in [None, ["f"]]:
            f1 = ForecasterRecursiveMultiSeries(make_estimator(est), lags=3, encoding=enc, categorical_features=cf)
            f2 = ForecasterRecursiveMultiSeries(make_estimator(est), lags=3, encoding=enc, categorical_features=cf)
            try:
                f1.fit(series=series, exog=exog_train, suppress_warnings=True)
                f2.fit(series=series, exog=relabel(exog_train), suppress_warnings=True)
                p1 = f1.predict(5, exog=exog_pred, suppress_warnings=True)["pred"].to_numpy()
                p2 = f2.predict(5, exog=relabel(exog_pred), suppress_warnings=True)["pred"].to_numpy()
                print(f"{est:9s} {enc:17s} cf={str(cf):6s} invariant_to_relabel={np.allclose(p1, p2)} max|d|={np.max(np.abs(p1-p2)):.4f}")
            except Exception as e:
                print(est, enc, cf, "err", str(e)[:80])
