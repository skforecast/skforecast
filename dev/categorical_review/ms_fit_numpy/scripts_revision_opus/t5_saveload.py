"""T5: save in one version, load in another. Usage:
python t5_saveload.py save <dir>   (run with base)  -> saves forecasters + predictions
python t5_saveload.py load <dir>   (run with branch) -> loads, predicts, set_in_sample_residuals
python t5_saveload.py roundtrip <dir> (branch only) -> save+load in same version"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
banner()
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.utils import save_forecaster, load_forecaster

mode, d = sys.argv[1], sys.argv[2]
os.makedirs(d, exist_ok=True)
series, exog_train, exog_pred = make_case("dict", ["f", "i", "ci0"])
CFGS = [("lgbm", "ordinal_category", "auto"), ("xgb", "ordinal_category", "auto"),
        ("cat", "ordinal_category", "auto"), ("hgb", "ordinal", None), ("lgbm", "ordinal", None),
        ("ridge", "onehot", "auto")]
for est, enc, cf in CFGS:
    name = f"{est}_{enc}_{cf}"
    path = os.path.join(d, name + ".joblib")
    if mode in ("save", "roundtrip"):
        f = ForecasterRecursiveMultiSeries(make_estimator(est), lags=3, encoding=enc, categorical_features=cf)
        f.fit(series=series, exog=exog_train, suppress_warnings=True, store_in_sample_residuals=True)
        p = f.predict(5, exog=exog_pred, suppress_warnings=True)["pred"].to_numpy()
        res = np.concatenate([np.asarray(v) for v in f.in_sample_residuals_.values()])
        save_forecaster(f, path, save_custom_functions=False, verbose=False)
        pickle.dump((p, res), open(path + ".pred.pkl", "wb"))
    if mode in ("load", "roundtrip"):
        p0, res0 = pickle.load(open(path + ".pred.pkl", "rb"))
        st, f, w = run(lambda: load_forecaster(path, verbose=False))
        if st == "err":
            print(name, "load err", f); continue
        st, p, w = run(lambda: f.predict(5, exog=exog_pred)["pred"].to_numpy())
        st2, r2, w2 = run(lambda: f.set_in_sample_residuals(series=series, exog=exog_train))
        res1 = np.concatenate([np.asarray(v) for v in f.in_sample_residuals_.values()]) if st2 == "ok" else None
        print(f"{name:32s} predict={st} same_pred={st=='ok' and np.array_equal(p, p0)} warn={w} | "
              f"set_in_sample_residuals={st2} {r2 if st2=='err' else ''} same_res="
              f"{res1 is not None and np.allclose(np.sort(res1), np.sort(res0))} warn={w2}", flush=True)
