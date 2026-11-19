import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import grid_search_forecaster, TimeSeriesFold, OneStepAheadFold

top = int(sys.argv[2]) if len(sys.argv) > 2 else 0
for kind in sys.argv[1].split(","):
    y, exog = make_data(exog_kind=None if kind == "none" else kind)
    for name in ["Ridge", "LGBM"]:
        est = estimators()[name]
        grid = {"alpha": [0.1, 1, 10]} if name == "Ridge" else {"n_estimators": [50, 100], "learning_rate": [0.1]}
        lags_grid = [7, 24]
        for cvname in ["OSA", "TSF"]:
            f = ForecasterRecursive(estimator=est, lags=LAGS)
            if cvname == "OSA":
                cv = OneStepAheadFold(initial_train_size=N - 50 * STEPS)
            else:
                cv = TimeSeriesFold(steps=STEPS, initial_train_size=N - 50 * STEPS, refit=False)
            kw = dict(forecaster=f, y=y, cv=cv, exog=exog, param_grid=grid, lags_grid=lags_grid,
                      metric="mean_absolute_error", return_best=False, n_jobs=1,
                      show_progress=False, verbose=False)
            t = timeit(lambda: grid_search_forecaster(**kw), n=2, warm=1)
            tot, out, st = profile_breakdown(lambda: grid_search_forecaster(**kw), n_rep=1)
            ncand = len(lags_grid) * np.prod([len(v) for v in grid.values()])
            print(f"exog={kind} {name} {cvname}: grid_search {ncand} cand = {t*1e3:.1f}ms ({t*1e3/ncand:.1f} ms/cand) | profile {fmt_breakdown(tot, out)}")
            if top:
                st.sort_stats("cumulative").print_stats(top)
