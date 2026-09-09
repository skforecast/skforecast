"""
Does the binner refit inside backtesting (use_in_sample_residuals=False) matter?
Current: fit() refits the binner every fold; out_sample_residuals_by_bin_ keys refer to the
user's original binner. Fixed: restore the original binner after each fit.
"""
import warnings; warnings.filterwarnings("ignore")
from copy import deepcopy
import numpy as np, pandas as pd
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold
from skforecast.datasets import fetch_dataset
from skforecast.metrics import winkler_score

INTERVAL = [0.05, 0.95]
recorded = []          # intervals_ seen after each fit (before restore)
RESTORE = {"on": False, "mode": "current"}
OOS = {}
_orig_fit = ForecasterRecursive.fit
def patched_fit(self, *a, **k):
    b, bi = deepcopy(self.binner), self.binner_intervals_
    r = _orig_fit(self, *a, **k)
    recorded.append(deepcopy(self.binner))
    if RESTORE["mode"] == "fixed" and bi is not None:
        self.binner, self.binner_intervals_ = b, bi
    if RESTORE["mode"] == "rebin":
        # re-bin the same OOS residuals with the NEW binner
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.set_out_sample_residuals(y_true=OOS["y_true"], y_pred=OOS["y_pred"])
        self._rebinned = self.out_sample_residuals_by_bin_
    return r
ForecasterRecursive.fit = patched_fit
_orig_pb = ForecasterRecursive.predict_bootstrapping
def patched_pb(self, *a, **k):
    if RESTORE["mode"] == "rebin":
        self.out_sample_residuals_by_bin_ = self._rebinned
    return _orig_pb(self, *a, **k)
ForecasterRecursive.predict_bootstrapping = patched_pb

def edges(binner):
    iv = binner.intervals_
    return np.array([iv[k][0] for k in sorted(iv)][1:])  # interior lower edges

def scenario(name, y, lags, steps, fit_frac, calib_frac, bt_train_frac, refit, fixed_train_size=False):
    n = len(y); n_fit = int(n * fit_frac); n_cal = int(n * calib_frac); n_bt = int(n * bt_train_frac)
    est = LGBMRegressor(n_estimators=150, learning_rate=0.05, verbose=-1, random_state=123)
    f = ForecasterRecursive(estimator=est, lags=lags)
    f.fit(y=y.iloc[:n_fit], store_in_sample_residuals=True)
    orig_binner = deepcopy(f.binner)
    # OOS residuals: model trained on the data before the calibration tramo, predicting it
    n_res_train = n_fit if n_cal > 0 else int(n * 0.6)
    n_res_end = n_fit + n_cal if n_cal > 0 else n_fit
    cv_cal = TimeSeriesFold(steps=steps, initial_train_size=n_res_train, refit=False)
    _, pv = backtesting_forecaster(forecaster=f, y=y.iloc[:n_res_end], cv=cv_cal,
                                   metric="mean_absolute_error", show_progress=False)
    f.set_out_sample_residuals(y_true=y.loc[pv.index], y_pred=pv["pred"])
    OOS["y_true"], OOS["y_pred"] = y.loc[pv.index], pv["pred"]

    cv = TimeSeriesFold(steps=steps, initial_train_size=n_bt, refit=refit, fixed_train_size=fixed_train_size)
    out = {}
    for mode in ["current", "fixed", "rebin"]:
        RESTORE["mode"] = mode; recorded.clear()
        _, p = backtesting_forecaster(forecaster=f, y=y, cv=cv, metric="mean_absolute_error",
                                      interval=INTERVAL, interval_method="bootstrapping", n_boot=200,
                                      use_in_sample_residuals=False, use_binned_residuals=True,
                                      random_state=123, show_progress=False)
        yt = y.loc[p.index]
        cov = ((yt >= p.lower_bound) & (yt <= p.upper_bound)).mean()
        wk = winkler_score(yt, p.lower_bound, p.upper_bound, alpha=0.1) / yt.abs().mean()
        width = (p.upper_bound - p.lower_bound).mean() / yt.abs().mean()
        # routing mismatch: bin of point predictions under refit binners vs original
        e0 = edges(orig_binner); rng_ = e0[-1] - e0[0]
        shifts = [np.abs(edges(b) - e0).max() / rng_ for b in recorded]
        preds = p["pred"].to_numpy()
        b0 = orig_binner.transform(preds).astype(int)
        mism = np.mean([(b.transform(preds).astype(int) != b0).mean() for b in recorded])
        out[mode] = dict(cov=cov, width=width, winkler=wk, n_fits=len(recorded),
                         max_edge_shift=max(shifts), mean_edge_shift=np.mean(shifts), routing_mismatch=mism)
    print(f"\n=== {name}")
    for m, d in out.items():
        print(f"  {m:8s} cov={d['cov']:.3f} width={d['width']:.3f} winkler={d['winkler']:.3f}")

bike = fetch_dataset("bike_sharing", verbose=False)["users"].asfreq("h").iloc[-6000:].astype(float)
h2o = fetch_dataset("h2o", verbose=False)["x"].asfreq("MS").astype(float)
fuel = fetch_dataset("fuel_consumption", verbose=False)["Gasolinas"].asfreq("MS").astype(float)
items = fetch_dataset("items_sales", verbose=False)["item_1"].asfreq("D").astype(float)

# Case 1 (documented workflow): fit on train+val, backtest initial_train_size = train+val, refit=False
# Case 2: same but refit=True expanding window
scenario("bike | refit=True expanding", bike, 24, 24, 0.6, 0.2, 0.8, True)
scenario("bike | refit=True rolling (fixed_train_size)", bike, 24, 24, 0.6, 0.2, 0.8, True, True)
# Case 3: user fit on train only, backtest trains on train+val (bigger train, no refit)
scenario("bike | fit on train, backtest train+val, refit=False", bike, 24, 24, 0.6, 0.2, 0.8, False)
scenario("h2o | refit=True expanding", h2o, 12, 12, 0.6, 0.2, 0.8, True)
scenario("h2o | fit on train, backtest train+val, refit=False", h2o, 12, 12, 0.6, 0.2, 0.8, False)
scenario("fuel | refit=True expanding", fuel, 12, 12, 0.6, 0.2, 0.8, True)
scenario("fuel | fit on train, backtest train+val, refit=False", fuel, 12, 12, 0.6, 0.2, 0.8, False)
scenario("items | refit=True expanding", items, 14, 7, 0.6, 0.2, 0.8, True)
scenario("items | fit on train, backtest train+val, refit=False", items, 14, 7, 0.6, 0.2, 0.8, False)
