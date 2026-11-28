import time, warnings
import numpy as np
from skforecast.preprocessing import QuantileBinner

rng = np.random.default_rng(0)
kw = {'n_bins': 10, 'method': 'linear', 'subsample': 200000, 'random_state': 789654, 'dtype': np.float64}
q = np.linspace(0, 100, 11)

def scenario(name, preds):
    # 1. Current: one QuantileBinner per series
    n_warn = 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        t = time.perf_counter()
        binners = []
        for p in preds:
            b = QuantileBinner(**kw); b.fit(p); binners.append(b)
        t_loop = time.perf_counter() - t
        n_warn = len(w)
    # parts of the current loop
    t = time.perf_counter()
    for p in preds:
        np.percentile(p, q, method='linear')
    t_pct = time.perf_counter() - t
    t = time.perf_counter()
    for _ in preds:
        QuantileBinner(**kw)
    t_init = time.perf_counter() - t
    # 2. Vectorized edges: one np.percentile per group of series with the same length
    t = time.perf_counter()
    lengths = np.array([len(p) for p in preds])
    edges = np.empty((len(preds), 11))
    for L in np.unique(lengths):
        ids = np.flatnonzero(lengths == L)
        edges[ids] = np.percentile(np.stack([preds[i] for i in ids]), q, axis=1, method='linear').T
    # rows that need the slow path: duplicated edges
    strictly_increasing = (np.diff(edges, axis=1) > 0).all(axis=1)
    t_vec = time.perf_counter() - t
    same = all(np.array_equal(edges[i], np.percentile(preds[i], q, method='linear')) for i in range(len(preds)))
    n_fast = int(strictly_increasing.sum())
    print(f'{name}: loop {t_loop:.3f}s (np.percentile {t_pct:.3f}s, __init__ {t_init:.3f}s, '
          f'{n_warn} warnings) | vectorized edges {t_vec:.3f}s | identical edges: {same} | '
          f'series without duplicated edges: {n_fast}/{len(preds)}')

# Continuous predictions (typical regression), 5000 series of 272 rows
scenario('5000 x 272 continuous', [rng.normal(size=272) for _ in range(5000)])
# Lengths between 100 and 1000
scenario('5000 x 100..1000 continuous', [rng.normal(size=rng.integers(100, 1001)) for _ in range(5000)])
# Intermittent demand: predictions with many repeated values (as LightGBM gives on counts)
scenario('5000 x 272 repeated values', [np.round(rng.gamma(0.3, 1, 272), 1) for _ in range(5000)])
