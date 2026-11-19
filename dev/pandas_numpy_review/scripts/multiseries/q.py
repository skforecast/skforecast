import timeit, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursiveMultiSeries
rng = np.random.default_rng(0); LEN=1000; N=100
idx = pd.date_range("2020-01-01", periods=LEN, freq="D")
series = {f"s{i:03d}": pd.Series(rng.normal(size=LEN).cumsum(), index=idx) for i in range(N)}
f = ForecasterRecursiveMultiSeries(Ridge(), lags=24)
f.fit(series, store_in_sample_residuals=True)
def t(fn, n=3): return min(timeit.repeat(fn, number=n, repeat=3))/n*1e3
tb = t(lambda: f.predict_bootstrapping(steps=24, n_boot=250))
tp = t(lambda: f.predict(steps=24))
ti = t(lambda: f.predict_interval(steps=24, n_boot=250, interval=[0.05, 0.95], method="bootstrapping"))
tq = t(lambda: f.predict_quantiles(steps=24, n_boot=250, quantiles=[0.05, 0.95]))
b = f.predict_bootstrapping(steps=24, n_boot=250)
tqp = t(lambda: b.iloc[:, 1:].quantile(q=[0.05, 0.95], axis=1).transpose())
v = b.iloc[:, 1:].to_numpy()
tqn = t(lambda: np.quantile(v, [0.05, 0.95], axis=1).T)
same = np.array_equal(b.iloc[:, 1:].quantile(q=[0.05, 0.95], axis=1).transpose().to_numpy(), np.quantile(v, [0.05, 0.95], axis=1).T)
print(pd.__version__, f"predict {tp:.1f} ms | predict_bootstrapping {tb:.1f} | predict_interval(boot) {ti:.1f} | predict_quantiles {tq:.1f}")
print(f"  DataFrame.quantile(axis=1) {tqp:.1f} ms vs np.quantile {tqn:.1f} ms; bitwise same: {same}")
import cProfile, pstats
from collections import defaultdict
pr = cProfile.Profile(); pr.enable(); f.predict_interval(steps=24, n_boot=250, interval=[0.05, 0.95], method="bootstrapping"); pr.disable()
agg = defaultdict(float); tot = 0
for (fn, ln, func), (cc, nc, tt, ct, _) in pstats.Stats(pr).stats.items():
    tot += tt
    lib = "pandas" if "/pandas/" in fn else "numpy" if "/numpy/" in fn else "sklearn" if "/sklearn/" in fn else "skforecast" if "/skforecast/" in fn else ("pandas(C)" if "pandas" in func else "numpy(C)" if ("numpy" in func or "ndarray" in func or "ufunc" in func) else "other")
    agg[lib] += tt
print({k: f"{100*v/tot:.0f}%" for k, v in sorted(agg.items(), key=lambda kv: -kv[1])}, f"total {tot*1e3:.0f} ms")
