import time, warnings
import numpy as np, pandas as pd
from skforecast.preprocessing import QuantileBinner

kw = {'n_bins': 10, 'method': 'linear', 'subsample': 200000, 'random_state': 789654, 'dtype': np.float64}
q = np.linspace(0, 100, 11)

def loop_fit(preds):
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for p in preds:
            b = QuantileBinner(**kw); b.fit(p); out.append(b)
    return out

def from_edges(bin_edges):
    b = QuantileBinner.__new__(QuantileBinner)
    b.n_bins, b.method, b.subsample, b.dtype, b.random_state = 10, 'linear', 200000, np.float64, 789654
    b.bin_edges_ = bin_edges; b.n_bins_ = len(bin_edges) - 1
    b.internal_edges_ = bin_edges[1:-1]
    b.intervals_ = dict(enumerate(zip(bin_edges[:-1].tolist(), bin_edges[1:].tolist())))
    return b

def vectorized_fit(preds):
    """Edges for all series with one np.percentile per length; clean rows fully
    vectorized, rows with duplicated edges or empty bins with the per-series rules."""
    lengths = np.array([len(p) for p in preds])
    out = [None] * len(preds)
    n_reduced = 0
    for L in np.unique(lengths):
        ids = np.flatnonzero(lengths == L)
        X = np.stack([preds[i] for i in ids])                       # (m, L)
        E = np.percentile(X, q, axis=1, method='linear').T          # (m, 11)
        clean = (np.diff(E, axis=1) > 0).all(axis=1)
        # counts per bin of the clean rows: number of values below each internal edge
        Xc, Ec = X[clean], E[clean]
        n_lt = (Xc[:, :, None] < Ec[:, None, 1:-1]).sum(axis=1)       # (mc, 9)
        cum = np.concatenate([np.zeros((len(Xc), 1), int), n_lt, np.full((len(Xc), 1), L)], axis=1)
        no_empty = (np.diff(cum, axis=1) > 0).all(axis=1)
        clean_ids = np.flatnonzero(clean)
        for j, row in zip(clean_ids[no_empty], Ec[no_empty]):
            out[ids[j]] = from_edges(row)
        # the rest: same rules as QuantileBinner.fit, on 11 edges already computed
        rest = np.setdiff1d(np.arange(len(ids)), clean_ids[no_empty])
        for j in rest:
            x, e = X[j], np.unique(E[j])
            if len(e) == 1:
                e = np.array([e.item(), e.item()])
            counts = np.bincount(np.searchsorted(e[1:-1], x, side='right'), minlength=len(e) - 1)
            if not counts.all():
                e = np.concatenate([e[:1], e[1:][counts > 0]])
            out[ids[j]] = from_edges(e)
            n_reduced += len(e) - 1 != 10
    return out, n_reduced

rng = np.random.default_rng(0)
scen = {
    'continuous, 272 rows': [rng.normal(size=272) for _ in range(5000)],
    'continuous, 100..1000 rows': [rng.normal(size=rng.integers(100, 1001)) for _ in range(5000)],
    'repeated values, 272 rows': [np.round(rng.gamma(0.3, 1, 272), 1) for _ in range(5000)],
    'half constant, 272 rows': [np.zeros(272) if i % 2 else rng.normal(size=272) for i in range(5000)],
}
for name, preds in scen.items():
    t = time.perf_counter(); ref = loop_fit(preds); t_loop = time.perf_counter() - t
    t = time.perf_counter(); new, n_red = vectorized_fit(preds); t_vec = time.perf_counter() - t
    ok = all(np.array_equal(a.bin_edges_, b.bin_edges_) and a.n_bins_ == b.n_bins_ and a.intervals_ == b.intervals_
             for a, b in zip(ref, new))
    x = rng.normal(size=1000)
    ok_t = all(np.array_equal(a.transform(x), b.transform(x)) for a, b in zip(ref[:200], new[:200]))
    print(f'{name}: loop {t_loop:.3f}s | vectorized {t_vec:.3f}s (x{t_loop / t_vec:.1f}) | '
          f'identical edges, n_bins_, intervals_: {ok} | identical transform: {ok_t} | series with fewer bins: {n_red}')
