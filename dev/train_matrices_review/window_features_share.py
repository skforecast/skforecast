import time, functools
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirect
from skforecast.preprocessing import RollingFeatures

acc = {}
def wrap(cls, name):
    orig = getattr(cls, name)
    @functools.wraps(orig)
    def w(*a, **k):
        t = time.perf_counter(); r = orig(*a, **k)
        acc[name] = acc.get(name, 0) + time.perf_counter() - t
        return r
    setattr(cls, name, w)

for n in ['_create_window_features', '_create_lags', '_create_train_X_y_single_series']:
    wrap(ForecasterRecursiveMultiSeries, n)
for n in ['_filter_nan_X_y_step', '_create_train_X_y_step']:
    wrap(ForecasterDirect, n)

rng = np.random.default_rng(0)
lags = list(range(1, 15)) + [21, 28]
def wf():
    return RollingFeatures(stats=['mean', 'mean', 'mean', 'std', 'min', 'max'],
                           window_sizes=[7, 28, 56, 28, 7, 7])
for n_series, n_days in [(5000, 300), (500, 1000)]:
    idx = pd.date_range('2020-01-01', periods=n_days, freq='D')
    series = {f's{i}': pd.Series(rng.poisson(3, n_days).astype(float), index=idx, name=f's{i}')
              for i in range(n_series)}
    for use_wf in [False, True]:
        best = None
        for _ in range(2):
            acc.clear()
            f = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=lags,
                                               window_features=wf() if use_wf else None)
            t = time.perf_counter(); f.create_train_X_y(series=series); tot = time.perf_counter() - t
            if best is None or tot < best[0]:
                best = (tot, dict(acc))
        tot, a = best
        print(f'MS {n_series}x{n_days} wf={use_wf}: total {tot:.2f}s | ' +
              ' | '.join(f'{k} {v:.2f}s ({v/tot:.0%})' for k, v in a.items()))

n, steps = 20000, 24
idx = pd.date_range('2000-01-01', periods=n, freq='h')
y = pd.Series(rng.normal(size=n).cumsum(), index=idx, name='y')
exog = pd.DataFrame(rng.normal(size=(n, 10)), index=idx, columns=[f'x{i}' for i in range(10)])
for dropna in [False, True]:
    best = None
    for _ in range(2):
        acc.clear()
        f = ForecasterDirect(estimator=Ridge(), lags=24, steps=steps, n_jobs=1,
                             dropna_from_series=dropna)
        t = time.perf_counter(); f.fit(y=y, exog=exog); tot = time.perf_counter() - t
        if best is None or tot < best[0]:
            best = (tot, dict(acc))
    tot, a = best
    print(f'Direct n={n} steps={steps} exog=10 dropna={dropna}: fit {tot:.2f}s | ' +
          ' | '.join(f'{k} {v:.3f}s ({v/tot:.0%})' for k, v in a.items()))
