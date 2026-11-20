import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from skforecast.utils import expand_index, input_to_frame
from skforecast.utils.utils import _is_utc_anchored_index, check_extract_values_and_index

y, exog = make_data(exog_kind="num")
ex = exog.iloc[-24:]
lw = y.iloc[-48:-24]
print("columns dtype:", ex.columns.dtype)
cols_list = ["e1", "e2", "e3"]
res = {}
res["exog.columns.tolist() (arrow str)"] = timeit(lambda: ex.columns.tolist(), n=2000)
res["list(exog.columns)"] = timeit(lambda: list(ex.columns), n=2000)
res["exog.columns.equals(Index)"] = timeit(lambda: ex.columns.equals(pd.Index(cols_list)), n=2000)
cols_obj = pd.Index(cols_list, dtype=object)
res["obj Index.tolist()"] = timeit(lambda: cols_obj.tolist(), n=2000)
res["exog.dtypes.tolist()"] = timeit(lambda: ex.dtypes.tolist(), n=2000)
res["[b.dtype for b in exog._mgr...] alt: exog.dtypes (Series)"] = timeit(lambda: ex.dtypes, n=2000)
res["exog.isna().to_numpy().any()"] = timeit(lambda: ex.isna().to_numpy().any(), n=2000)
res["np.isnan(exog.to_numpy()).any()"] = timeit(lambda: np.isnan(ex.to_numpy()).any(), n=2000)
res["exog.to_numpy()"] = timeit(lambda: ex.to_numpy(), n=2000)
res["input_to_frame(exog)"] = timeit(lambda: input_to_frame(ex, 'exog'), n=2000)
res["expand_index(lw.index, 24)"] = timeit(lambda: expand_index(lw.index, 24), n=2000)
res["pd.date_range(start, periods=25, freq)[1:]"] = timeit(lambda: pd.date_range(start=lw.index[-1], periods=25, freq=lw.index.freq)[1:], n=2000)
res["_is_utc_anchored_index (naive)"] = timeit(lambda: _is_utc_anchored_index(lw.index, lw.index.freq), n=2000)
pidx = expand_index(lw.index, 24)
res["pd.Series(arr, index=pidx, name)"] = timeit(lambda: pd.Series(np.zeros(24), index=pidx, name="pred"), n=2000)
res["pd.DataFrame(arr 24x3, index, cols)"] = timeit(lambda: pd.DataFrame(np.zeros((24, 3)), index=pidx, columns=["pred", "lower_bound", "upper_bound"]), n=2000)
res["lw.to_numpy(copy=True).ravel()"] = timeit(lambda: lw.to_numpy(copy=True).ravel(), n=2000)
res["check_extract_values_and_index(lw, no values)"] = timeit(lambda: check_extract_values_and_index(lw, return_values=False), n=2000)
res["lw.iloc[-24:]"] = timeit(lambda: lw.iloc[-24:], n=2000)
res["y.iloc[a:b] (fold slice)"] = timeit(lambda: y.iloc[1000:1024], n=2000)
res["exog.iloc[a:b] (fold slice)"] = timeit(lambda: exog.iloc[1000:1024], n=2000)
res["pd.isna(lw.to_numpy()).any()"] = timeit(lambda: pd.isna(lw.to_numpy()).any(), n=2000)
res["exog.index[0] == pidx[0]"] = timeit(lambda: ex.index[0] == pidx[0], n=2000)
res["exog.index.freq == pidx.freq"] = timeit(lambda: ex.index.freq == pidx.freq, n=2000)
res["lw.index.freq == 'D' (str compare)"] = timeit(lambda: lw.index.freq == "D", n=2000)
for k, v in res.items():
    print(f"{v*1e6:9.2f} us  {k}")

# boot quantile
boot = pd.DataFrame(np.random.default_rng(0).normal(size=(24, 250)), index=pidx)
print(f"{timeit(lambda: boot.quantile(q=[0.05,0.95], axis=1).transpose(), n=500)*1e6:9.2f} us  DataFrame.quantile axis=1 (24x250)")
arr = boot.to_numpy()
print(f"{timeit(lambda: np.quantile(arr, [0.05,0.95], axis=1).T, n=500)*1e6:9.2f} us  np.quantile axis=1")
print(f"{timeit(lambda: pd.DataFrame(arr, index=pidx, columns=[f'pred_boot_{i}' for i in range(250)]), n=500)*1e6:9.2f} us  pd.DataFrame boot (24x250) + col names")
