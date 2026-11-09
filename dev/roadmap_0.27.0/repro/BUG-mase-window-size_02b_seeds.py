"""Experiment 2b: many seeds, synthetic series with burn-in (stationary from t=0)."""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import importlib.util
from pathlib import Path


def _load(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_search = _load("BUG-mase-window-size_02_search.py")
search, rank_summary, LAGS_GRID = _search.search, _search.rank_summary, _search.LAGS_GRID

def synthetic(seed, n=400, trend_var=0.0, burn=200):
    rng = np.random.default_rng(seed)
    N = n + burn
    t = np.arange(N) - burn
    sd = np.where(t < 0, 1.0, 1.0 + trend_var * np.clip(t, 0, None) / n)
    e = np.zeros(N)
    for i in range(1, N):
        e[i] = 0.7 * e[i - 1] + rng.normal(scale=sd[i])
    y = 10 + 3 * np.sin(2 * np.pi * t / 12) + e
    return pd.Series(y[burn:], index=pd.date_range("2000-01-01", periods=n, freq="MS"), name="y")

print("| series | fold | seeds | best MASE != best MAE | median MAE loss of MASE pick (when it differs) | max MAE loss | best RMSSE != best MSE | mean Kendall tau MASE~MAE | mean n_lags best MASE / MAE | scale(ws=48)/scale(ws=0) mean [min, max] |")
print("|---|---|---|---|---|---|---|---|---|---|")
for label, tv in [("constant variance", 0.0), ("variance grows x3", 2.0), ("variance shrinks x1/3", -0.667)]:
    for kind in ["TimeSeriesFold", "OneStepAheadFold"]:
        d1 = d2 = 0; losses = []; taus = []; bm = []; ba = []; ratios = []
        for s in range(30):
            y = synthetic(s, trend_var=tv)
            ytr = y.iloc[:300].to_numpy()
            ratios.append(np.mean(np.abs(np.diff(ytr[48:]))) / np.mean(np.abs(np.diff(ytr))))
            res = search(y, 300, kind)
            by_mase, by_mae, by_rmsse, by_mse, tau, _ = rank_summary(res)
            if by_mase[0] != by_mae[0]:
                d1 += 1
                mae = res["mean_absolute_error"]
                losses.append(100 * (mae[by_mase[0]] / mae[by_mae[0]] - 1))
            d2 += by_rmsse[0] != by_mse[0]
            taus.append(tau); bm.append(by_mase[0]); ba.append(by_mae[0])
        med = f"{np.median(losses):.2f}%" if losses else "-"
        mx = f"{np.max(losses):.2f}%" if losses else "-"
        print(f"| {label} | {kind} | 30 | {d1} | {med} | {mx} | {d2} | {np.mean(taus):.3f} | {np.mean(bm):.1f} / {np.mean(ba):.1f} | {np.mean(ratios):.3f} [{np.min(ratios):.3f}, {np.max(ratios):.3f}] |")
