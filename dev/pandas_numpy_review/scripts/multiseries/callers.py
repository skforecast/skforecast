import sys, cProfile, pstats, warnings, gc
warnings.filterwarnings("ignore")
sys.argv = ["x", sys.argv[1], sys.argv[2], "ridge", "1"]
exec(open("prof_fit.py").read().split("# ---------------------------------------------------------------- patches")[0])
from skforecast.recursive import ForecasterRecursiveMultiSeries
fc = ForecasterRecursiveMultiSeries(estimator=make_est(), lags=24)
fc.fit(series=series, exog=exog, suppress_warnings=True)
gc.disable()
pr = cProfile.Profile(); pr.enable()
fc = ForecasterRecursiveMultiSeries(estimator=make_est(), lags=24)
fc.fit(series=series, exog=exog, suppress_warnings=True)
pr.disable()
st = pstats.Stats(pr)
st.sort_stats("cumulative")
st.print_callers(r"datetimelike.py:1438|indexing.py:1327|check_array_indexer|generic.py:6800\(copy\)|_ordinal|percentile")
