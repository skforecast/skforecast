import time, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.preprocessing import RollingFeatures
warnings.simplefilter('ignore')

rng = np.random.default_rng(0)
n_series, n_days = 5000, 300
idx = pd.date_range('2020-01-01', periods=n_days, freq='D')
wide = pd.DataFrame(rng.poisson(3, (n_days, n_series)).astype(float), index=idx,
                    columns=[f's{i}' for i in range(n_series)])
series = {c: wide[c].rename(c) for c in wide.columns}
lags = list(range(1, 15)) + [21, 28]
stats = ['mean', 'mean', 'mean', 'std', 'min', 'max']
sizes = [7, 28, 56, 28, 7, 7]

def make(wf=True):
    return ForecasterRecursiveMultiSeries(
        estimator=LGBMRegressor(n_estimators=200, verbose=-1, random_state=0),
        lags=lags, window_features=RollingFeatures(stats=stats, window_sizes=sizes) if wf else None)

for wf in [False, True]:
    f = make(wf)
    t = time.perf_counter(); X, y = f.create_train_X_y(series=series); t_ctx = time.perf_counter() - t
    t = time.perf_counter(); f.fit(series=series); t_fit = time.perf_counter() - t
    est = LGBMRegressor(n_estimators=200, verbose=-1, random_state=0)
    t = time.perf_counter(); est.fit(X, y); t_est = time.perf_counter() - t
    t = time.perf_counter(); f.predict(steps=28); t_pred = time.perf_counter() - t
    print(f'wf={wf}: rows {len(X)}, cols {X.shape[1]} | create_train_X_y {t_ctx:.2f}s | '
          f'fit {t_fit:.2f}s (LightGBM alone {t_est:.2f}s) | predict 28 steps {t_pred:.2f}s')

# Prototype: the same 6 rolling stats for all series at once on the wide frame
t = time.perf_counter()
out = []
for s, w in zip(stats, sizes):
    r = wide.rolling(window=w, min_periods=w)
    out.append(getattr(r, s)().to_numpy())
t_wide = time.perf_counter() - t
print(f'wide-frame rolling of all series at once: {t_wide:.2f}s')
