import sys, warnings, timeit
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection._utils import _extract_data_folds_multiseries
from skforecast.utils import check_predict_input

warnings.filterwarnings("ignore")


def t(label, fn, n=1000):
    r = min(timeit.repeat(fn, number=n, repeat=5)) / n
    print(f"   {label:72s} {r*1e6:8.1f} us")
    return r


N = 2000
for nser in (10, 20):
    idx = pd.date_range("2020-01-01", periods=N + 24, freq="h")
    rng = np.random.default_rng(0)
    series = pd.DataFrame(rng.normal(size=(len(idx), nser)).cumsum(axis=0), index=idx,
                          columns=[f"s{i}" for i in range(nser)])
    f = ForecasterDirectMultiVariate(Ridge(), level="s0", steps=24, lags=24, n_jobs=1)
    f.fit(series.iloc[:N])
    cols = f.X_train_series_names_in_
    ws = f.window_size
    # a fold last window as backtesting builds it
    lw = series.iloc[N - ws:N]
    print(f"\n== DMV {nser} series, no exog")
    t("predict(last_window=lw)", lambda: f.predict(last_window=lw))
    t("predict(last_window=lw, check_inputs=False)", lambda: f.predict(last_window=lw, check_inputs=False))
    t("check_predict_input", lambda: check_predict_input(
        forecaster_name="ForecasterDirectMultiVariate", steps=list(range(1, 25)), is_fitted=True,
        exog_in_=False, index_type_=f.index_type_, index_freq_=f.index_freq_, window_size=ws,
        last_window=lw, exog=None, exog_names_in_=None, max_step=24, series_names_in_=cols))
    t("  of which lw[series_names_in_] (list getitem)", lambda: lw[cols])
    t("  of which pd.isna(lw[cols].to_numpy()).any()", lambda: pd.isna(lw[cols].to_numpy()).any())
    def cur():
        l2 = lw.iloc[-ws:, lw.columns.get_indexer(cols)].copy()
        return [l2[c].to_numpy(copy=True) for c in cols]
    t("current extraction: iloc+get_indexer+copy, per-series [col].to_numpy(copy)", cur)
    def alt():
        pos = lw.columns.get_indexer(cols)
        v = lw.to_numpy(dtype=float)  # one block
        v = v[-ws:, pos]               # fancy index -> new array (copy)
        return [v[:, i] for i in range(v.shape[1])]
    t("alt: to_numpy(dtype=float) once, numpy slicing", alt)
    a = cur(); b = alt()
    print("   same:", all(np.array_equal(x, z) for x, z in zip(a, b)))
    t("lw.columns.get_indexer(cols)", lambda: lw.columns.get_indexer(cols))
    t("lw.to_numpy(dtype=float)", lambda: lw.to_numpy(dtype=float))
    # backtesting fold extraction
    cv = TimeSeriesFold(steps=24, initial_train_size=N - 1200, refit=False, verbose=False)
    folds = cv.split(X=series.iloc[:N], as_pandas=False)
    span_index = cv._extract_index(X=series.iloc[:N])
    st = series.iloc[:800]
    t("  per fold: series.iloc[lw] + .drop(columns=[])",
      lambda: series.iloc[1000:1024].drop(columns=[]))
    t("  per fold: series.iloc[lw]", lambda: series.iloc[1000:1024])
