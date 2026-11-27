"""T4: weights, backtesting with/without refit (n_jobs 1 and 2), conformal
intervals, other forecasters. Writes a pickle with results to compare base vs
branch. Usage: python t4_misc.py out.pkl"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
banner()
from skforecast.recursive import ForecasterRecursiveMultiSeries, ForecasterRecursive
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries, backtesting_forecaster

R = {}
series, exog_train, exog_pred = make_case("dict", ["f", "i", "b", "ci", "cs"], effect_col="ci")
def wf(index):
    return np.where(index.dayofweek == 0, 0.5, 1.0)
for est in ["lgbm", "xgb", "cat", "hgb", "ridge"]:
    for enc in ["ordinal", "ordinal_category", "onehot"]:
        for weights in ["none", "weight_func", "series_weights"]:
            kw = {}
            if weights == "weight_func":
                kw["weight_func"] = wf
            if weights == "series_weights":
                kw["series_weights"] = {"s0": 2.0, "s1": 1.0, "s2": 0.5, "s3": 1.0}
            f = ForecasterRecursiveMultiSeries(make_estimator(est), lags=3, encoding=enc, **kw)
            st, r, w = run(lambda: f.fit(series=series, exog=exog_train))
            p = f.predict(5, exog=exog_pred, suppress_warnings=True)["pred"].to_numpy() if st == "ok" else r
            R[("w", est, enc, weights)] = p
            if weights == "none" and st == "ok":
                st2, r2, _ = run(lambda: f.predict_interval(5, exog=exog_pred, method="conformal", interval=[0.1, 0.9], suppress_warnings=True))
                R[("conformal", est, enc)] = r2.iloc[:, 1:].to_numpy().astype(float) if st2 == "ok" else r2
                st2, r2, _ = run(lambda: f.predict_quantiles(5, exog=exog_pred, quantiles=[0.1, 0.5], n_boot=10, suppress_warnings=True))
                R[("quant", est, enc)] = r2.iloc[:, 1:].to_numpy().astype(float) if st2 == "ok" else r2
        for refit in [False, True]:
            for n_jobs in [1, 2]:
                if est == "cat" and n_jobs == 2:
                    continue
                f = ForecasterRecursiveMultiSeries(make_estimator(est), lags=3, encoding=enc)
                cv = TimeSeriesFold(steps=4, initial_train_size=80, refit=refit)
                st, r, w = run(lambda: backtesting_forecaster_multiseries(
                    f, series, cv=cv, exog={k: pd.concat([v, exog_pred[k]]) for k, v in exog_train.items()},
                    metric="mean_absolute_error", n_jobs=n_jobs, show_progress=False, suppress_warnings=True,
                    interval=[0.1, 0.9], interval_method="conformal"))
                R[("bt", est, enc, refit, n_jobs)] = r[1].iloc[:, 1:].to_numpy().astype(float) if st == "ok" else r
        print(est, enc, flush=True)

# Other forecasters: ForecasterRecursive / Direct / DirectMultiVariate with category exog
s = make_series("df")
exf = make_exog_frame(pd.date_range("2020-01-01", periods=125, freq="D"), ["f", "ci", "cs"])
for cls in ["rec", "dir"]:
    for est in ["lgbm", "xgb", "hgb", "ridge", "cat"]:
        for cf in ["auto", None]:
            exo = exf if cf == "auto" else exf.drop(columns="cs")
            if cls == "rec":
                f = ForecasterRecursive(make_estimator(est), lags=3, categorical_features=cf)
            else:
                f = ForecasterDirect(make_estimator(est), lags=3, steps=3, categorical_features=cf)
            st, r, w = run(lambda: f.fit(y=s["s0"], exog=exo.iloc[:120]))
            R[(cls, est, cf)] = f.predict(3, exog=exo.iloc[120:]).to_numpy() if st == "ok" else r
            R[(cls, est, cf, "w")] = w
for est in ["lgbm", "xgb", "hgb", "ridge"]:
    for cf in ["auto", None]:
        exo = exf if cf == "auto" else exf.drop(columns="cs")
        f = ForecasterDirectMultiVariate(make_estimator(est), level="s0", steps=3, lags=3, categorical_features=cf)
        st, r, w = run(lambda: f.fit(series=s, exog=exo.iloc[:120]))
        R[("dmv", est, cf)] = f.predict(exog=exo.iloc[120:])["pred"].to_numpy() if st == "ok" else r
        R[("dmv", est, cf, "w")] = w

pickle.dump(R, open(sys.argv[1], "wb"))
print("n", len(R))
