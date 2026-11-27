"""T6: user code that passes the DataFrame of create_train_X_y to
forecaster.estimator (the SHAP / explainability pattern). Does
estimator.predict(X_train) give the in-sample fitted values (y_train - residual)?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
banner()
from skforecast.recursive import ForecasterRecursiveMultiSeries

for exname, cols, cf in [("num", ["f", "i", "b"], "auto"), ("ci_auto", ["f", "ci"], "auto"),
                         ("ci_none", ["f", "ci"], None)]:
    series, exog_train, _ = make_case("df", cols, effect_col="ci" if "ci" in cols else None, exog_kind="df")
    for est in ["lgbm", "xgb", "hgb", "ridge", "cat"]:
        for enc in ["ordinal", "ordinal_category"]:
            f = ForecasterRecursiveMultiSeries(make_estimator(est), lags=3, encoding=enc, categorical_features=cf)
            st, r, w = run(lambda: f.fit(series=series, exog=exog_train, suppress_warnings=True))
            if st == "err":
                print(exname, est, enc, "fit err", r[:80]); continue
            X, y = f.create_train_X_y(series=series, exog=exog_train, suppress_warnings=True)
            st, yp, w = run(lambda: np.asarray(f.estimator.predict(X)).ravel())
            if st == "err":
                print(f"{exname:8s} {est:6s} {enc:17s} estimator.predict(X_train DataFrame) ERROR {yp[:120]}"); continue
            # fitted values the forecaster itself computed
            Xn = X.to_numpy(dtype=float)
            st2, yref, _ = run(lambda: np.asarray(f.estimator.predict(Xn if est != 'cat' else X)).ravel())
            print(f"{exname:8s} {est:6s} {enc:17s} max|pred(DataFrame)-pred(numpy)|="
                  f"{(np.max(np.abs(yp - yref)) if st2=='ok' else yref[:60])} warns={w}")
