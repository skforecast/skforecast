import warnings, numpy as np, pandas as pd, time
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.preprocessing import RollingFeatures
import skforecast.recursive._forecaster_recursive_multiseries as M
rng = np.random.default_rng(0); LEN=1000; N=100
idx = pd.date_range("2020-01-01", periods=LEN, freq="D")
series = {f"s{i:03d}": pd.Series(rng.normal(size=LEN).cumsum(), index=idx) for i in range(N)}
acc = {"wf": 0.0, "tb": 0.0}
o = M.ForecasterRecursiveMultiSeries._create_window_features
def w(self, *a, **k):
    t0 = time.perf_counter(); r = o(self, *a, **k); acc["wf"] += time.perf_counter() - t0; return r
M.ForecasterRecursiveMultiSeries._create_window_features = w
rf = RollingFeatures(stats=["mean", "std", "min"], window_sizes=[7, 7, 7])
otb = type(rf).transform_batch
def wtb(self, *a, **k):
    t0 = time.perf_counter(); r = otb(self, *a, **k); acc["tb"] += time.perf_counter() - t0; return r
type(rf).transform_batch = wtb
best = None
for _ in range(4):
    acc.update(wf=0.0, tb=0.0)
    f = ForecasterRecursiveMultiSeries(Ridge(), lags=24, window_features=rf)
    t0 = time.perf_counter(); f.fit(series); t = time.perf_counter() - t0
    if best is None or t < best[0]: best = (t, dict(acc))
t, a = best
print(pd.__version__, f"fit {t*1e3:.1f} ms; _create_window_features {a['wf']*1e3:.1f} ms of which transform_batch {a['tb']*1e3:.1f} ms -> wrapper overhead {(a['wf']-a['tb'])*1e3:.1f} ms")
