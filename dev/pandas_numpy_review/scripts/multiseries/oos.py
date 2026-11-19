import timeit, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursiveMultiSeries
import skforecast.recursive._forecaster_recursive_multiseries as M
rng = np.random.default_rng(0); LEN=1000; N=100
idx = pd.date_range("2020-01-01", periods=LEN, freq="D")
series = {f"s{i:03d}": pd.Series(rng.normal(size=LEN).cumsum(), index=idx) for i in range(N)}
f = ForecasterRecursiveMultiSeries(Ridge(), lags=24)
f.fit(series)
yt = {k: rng.normal(size=1000) for k in series}; yp = {k: rng.normal(size=1000) for k in series}
t = min(timeit.repeat(lambda: f.set_out_sample_residuals(yt, yp), number=3, repeat=3))/3*1e3
calls = {"tr": 0.0, "bin": 0.0}
import time
ot, ob = M.ForecasterRecursiveMultiSeries._transform_out_sample_residuals, M.ForecasterRecursiveMultiSeries._binning_out_sample_residuals
def wt(self, *a, **k):
    t0=time.perf_counter(); r=ot(self,*a,**k); calls["tr"]+=time.perf_counter()-t0; return r
def wb(self, *a, **k):
    t0=time.perf_counter(); r=ob(self,*a,**k); calls["bin"]+=time.perf_counter()-t0; return r
M.ForecasterRecursiveMultiSeries._transform_out_sample_residuals = wt
M.ForecasterRecursiveMultiSeries._binning_out_sample_residuals = wb
f.set_out_sample_residuals(yt, yp)
data = pd.DataFrame({'residuals': yt['s000']}); data['bin'] = rng.integers(0, 10, 1000)
tg = min(timeit.repeat(lambda: data.groupby('bin')['residuals'].apply(np.array).to_dict(), number=20, repeat=3))/20*1e6
b = data['bin'].to_numpy(); r = data['residuals'].to_numpy()
def alt():
    o = np.argsort(b, kind='stable'); u, s = np.unique(b[o], return_index=True)
    return dict(zip(u.tolist(), np.split(r[o], s[1:])))
ta = min(timeit.repeat(alt, number=20, repeat=3))/20*1e6
print(pd.__version__, f"set_out_sample_residuals 100 levels x 1000: {t:.1f} ms; transform {calls['tr']*1e3:.1f} ms, binning {calls['bin']*1e3:.1f} ms; groupby-apply {tg:.0f} us/level vs numpy split {ta:.0f} us/level")
