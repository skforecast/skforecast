"""Main battery: fit/predict/intervals/residuals across estimators, encodings,
categorical_features modes and exog dtypes. Usage: python battery.py out.pkl"""
import sys, os, pickle, itertools, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
banner()
from skforecast.recursive import ForecasterRecursiveMultiSeries

ESTIMATORS = ["lgbm", "xgb", "cat", "hgb", "ridge", "lr", "rf", "pipe_ridge",
              "pipe_lgbm", "ttr_lgbm"]
ENCODINGS = ["ordinal", "ordinal_category", "onehot", None]
EXOGS = {
    "num": ["f", "i", "b"],
    "ci": ["f", "ci"],
    "ci0": ["f", "ci0"],
    "cneg": ["f", "cneg"],
    "mixstr": ["f", "ci", "cs", "o"],
    "cf": ["f", "cf"],
    "nullable": ["f", "nIfull", "nF", "nB"],
    "nullableNA": ["f", "nI"],
    "pyarrow": ["f", "paI", "paF", "paB"],
    "str": ["f", "str"],
    "dt": ["f", "dt"],
}
NONNUM = {"ci", "ci0", "cneg", "cs", "o", "str", "cf"}

def cf_modes(cols):
    nonnum = [c for c in cols if c in NONNUM]
    modes = {"auto": "auto", "none": None}
    if nonnum:
        modes["full"] = list(nonnum)
        inc = [c for c in nonnum if c not in ("ci", "ci0", "cneg", "cf")]
        modes["incomplete"] = inc if inc else ["i"] if "i" in cols else ["f"]
    return modes

only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
results = {}
for series_kind in ["dict", "df"]:
    for ex_name, cols in EXOGS.items():
        effect = next((c for c in cols if c in ("ci", "cneg")), None)
        series, exog_train, exog_pred = make_case(series_kind, cols, effect_col=effect)
        for est in ESTIMATORS:
            if only and est not in only:
                continue
            for enc in ENCODINGS:
                for cfn, cf in cf_modes(cols).items():
                    key = (series_kind, ex_name, est, enc, cfn)
                    f = ForecasterRecursiveMultiSeries(
                        estimator=make_estimator(est), lags=3, encoding=enc,
                        categorical_features=cf)
                    st, r, w = run(lambda: f.fit(series=series, exog=exog_train, store_in_sample_residuals=True))
                    out = {"fit": (st, r if st == "err" else None, w)}
                    if st == "ok":
                        out["names"] = list(f.X_train_features_names_out_)
                        st2, p, w2 = run(lambda: f.predict(steps=5, exog=exog_pred))
                        out["pred"] = (st2, p.to_numpy() if st2 == "ok" else p, w2)
                        res = f.in_sample_residuals_
                        out["res"] = {k: np.asarray(v) for k, v in res.items()} if res else None
                        st3, pi, w3 = run(lambda: f.predict_interval(
                            steps=5, exog=exog_pred, method="bootstrapping",
                            interval=[0.1, 0.9], n_boot=5, use_in_sample_residuals=True))
                        out["pi"] = (st3, pi.to_numpy()[:, 1:].astype(float) if st3 == "ok" else pi, w3)
                        st4, r4, w4 = run(lambda: f.set_in_sample_residuals(series=series, exog=exog_train))
                        res2 = f.in_sample_residuals_
                        out["res2"] = (st4, {k: np.asarray(v) for k, v in res2.items()} if (st4 == "ok" and res2) else r4)
                    results[key] = out
        print(series_kind, ex_name, "done", flush=True)

with open(sys.argv[1], "wb") as fh:
    pickle.dump(results, fh)
print("n cases", len(results))
