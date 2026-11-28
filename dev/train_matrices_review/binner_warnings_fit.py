import time, warnings, functools
import numpy as np, pandas as pd
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.exceptions import IgnoredArgumentWarning

rng = np.random.default_rng(0)
n_series, n_days = 2000, 300
idx = pd.date_range('2020-01-01', periods=n_days, freq='D')
# intermittent demand: 60 % of the series with sales on 5 % of the days
rates = np.where(rng.random(n_series) < 0.6, 0.05, 2.0)
series = {f's{i}': pd.Series(rng.poisson(rates[i], n_days).astype(float), index=idx, name=f's{i}')
          for i in range(n_series)}
f = ForecasterRecursiveMultiSeries(estimator=LGBMRegressor(n_estimators=100, verbose=-1), lags=14)
acc = {'t': 0.0}
orig = f._binning_in_sample_residuals
def timed(*a, **k):
    t = time.perf_counter(); r = orig(*a, **k); acc['t'] += time.perf_counter() - t; return r
f._binning_in_sample_residuals = timed
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always', IgnoredArgumentWarning)
    t = time.perf_counter(); f.fit(series=series); t_fit = time.perf_counter() - t
binw = [x for x in w if 'number of bins has been reduced' in str(x.message)]
print(f'fit {t_fit:.2f}s, binning {acc["t"]:.2f}s ({acc["t"]/t_fit:.0%}); bin warnings emitted: {len(binw)}, '
      f'distinct texts: {len({str(x.message) for x in binw})}')
