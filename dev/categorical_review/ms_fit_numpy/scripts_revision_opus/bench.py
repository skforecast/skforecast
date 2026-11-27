"""Benchmark one (estimator, exog kind, encoding) config in a fresh process.
Usage: python bench.py <est> <exogkind> <encoding> <reps>  -> prints one JSON line."""
import os, sys, time, json, ctypes, tracemalloc, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
h = ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess())
ctypes.windll.kernel32.SetProcessAffinityMask(h, 1 << 12)
ctypes.windll.kernel32.SetPriorityClass(h, 0x00000080)  # HIGH
import numpy as np, pandas as pd, sklearn
warnings.filterwarnings("ignore")
import skforecast
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import (TimeSeriesFold, OneStepAheadFold,
    backtesting_forecaster_multiseries, grid_search_forecaster_multiseries)
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

est_name, exkind, enc, reps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
enc = None if enc == "None" else enc
N_SERIES, N = 300, 400
rng = np.random.default_rng(0)
idx = pd.date_range("2020-01-01", periods=N + 24, freq="D")
series = pd.DataFrame(rng.normal(size=(N + 24, N_SERIES)).cumsum(axis=0),
                      index=idx, columns=[f"s{i}" for i in range(N_SERIES)])
ex = pd.DataFrame({"f1": rng.normal(size=N + 24), "f2": rng.normal(size=N + 24)}, index=idx)
if exkind == "mixed":
    ex["i"] = rng.integers(0, 7, N + 24)
    ex["b"] = rng.integers(0, 2, N + 24).astype(bool)
    ex["c"] = pd.Categorical(np.array(["a", "b", "c", "d"])[rng.integers(0, 4, N + 24)])
series_train = series.iloc[:N]
exog_train, exog_pred = ex.iloc[:N], ex.iloc[N:]

def mk():
    if est_name == "lgbm":
        e = LGBMRegressor(n_estimators=50, verbose=-1, n_jobs=1, random_state=1)
    elif est_name == "xgb":
        e = XGBRegressor(n_estimators=50, n_jobs=1, random_state=1)
    elif est_name == "ridge":
        e = Ridge()
    elif est_name == "hgb":
        e = HistGradientBoostingRegressor(max_iter=50, random_state=1)
    return ForecasterRecursiveMultiSeries(e, lags=24, encoding=enc)

def t_fit():
    f = mk(); t = time.perf_counter(); f.fit(series_train, exog=exog_train); return time.perf_counter() - t, f
def t_pred(f):
    t = time.perf_counter(); f.predict(24, exog=exog_pred); return time.perf_counter() - t
def t_bt():
    f = mk(); cv = TimeSeriesFold(steps=24, initial_train_size=N - 72, refit=False)
    t = time.perf_counter()
    backtesting_forecaster_multiseries(f, series_train, cv=cv, exog=exog_train, n_jobs=1,
                                       show_progress=False, metric="mean_absolute_error")
    return time.perf_counter() - t
def t_osa():
    f = mk(); cv = OneStepAheadFold(initial_train_size=N - 72)
    grid = {"ridge": {"alpha": [0.1, 1, 10]}}.get(est_name, {"max_depth": [3, 5, 7]} if est_name != "hgb" else {"max_depth": [3, 5, 7]})
    t = time.perf_counter()
    grid_search_forecaster_multiseries(f, series_train, cv=cv, exog=exog_train, param_grid=grid,
                                       lags_grid=[24], metric="mean_absolute_error",
                                       return_best=False, verbose=False, show_progress=False)
    return time.perf_counter() - t

out = {"skf": os.path.dirname(skforecast.__file__),
       "py": sys.version.split()[0], "np": np.__version__, "pd": pd.__version__, "skl": sklearn.__version__}
# warmup
dt, f = t_fit(); t_pred(f)
res = {"fit": [], "pred": [], "bt": [], "osa": []}
for _ in range(reps):
    dt, f = t_fit(); res["fit"].append(dt)
    res["pred"].append(t_pred(f))
    res["bt"].append(t_bt())
    res["osa"].append(t_osa())
out.update({k: sorted(v) for k, v in res.items()})
# memory: tracemalloc peak (python + numpy allocations) of fit and osa
tracemalloc.start(); f = mk(); f.fit(series_train, exog=exog_train)
_, peak = tracemalloc.get_traced_memory(); tracemalloc.stop(); out["mem_fit_MB"] = peak / 2**20
tracemalloc.start(); t_osa()
_, peak = tracemalloc.get_traced_memory(); tracemalloc.stop(); out["mem_osa_MB"] = peak / 2**20
class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [(n, ctypes.c_size_t) for n in
                ("PeakWorkingSetSize", "WorkingSetSize", "a", "b", "c", "d", "PagefileUsage", "PeakPagefileUsage")]
c = PMC(); c.cb = ctypes.sizeof(PMC)
ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb)
out["peak_ws_MB"] = c.PeakWorkingSetSize / 2**20
print(json.dumps(out))
