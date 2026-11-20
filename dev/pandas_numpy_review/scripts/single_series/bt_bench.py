import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold

kinds = sys.argv[1].split(",") if len(sys.argv) > 1 else ["none", "num", "cat"]
refit = (sys.argv[2] == "refit") if len(sys.argv) > 2 else False
interval = None
if len(sys.argv) > 3 and sys.argv[3] != "none":
    interval = [0.1, 0.9] if sys.argv[3] == "boot" else "conformal"
top = int(sys.argv[4]) if len(sys.argv) > 4 else 0

for kind in kinds:
    y, exog = make_data(exog_kind=None if kind == "none" else kind)
    for name, est in estimators().items():
        f = ForecasterRecursive(estimator=est, lags=LAGS)
        cv = TimeSeriesFold(steps=STEPS, initial_train_size=N - 50 * STEPS, refit=refit)
        kw = dict(forecaster=f, y=y, cv=cv, exog=exog, metric="mean_absolute_error",
                  n_jobs=1, show_progress=False)
        if interval == "conformal":
            kw.update(interval=0.8, interval_method="conformal")
        elif interval is not None:
            kw.update(interval=interval, n_boot=100)
        t = timeit(lambda: backtesting_forecaster(**kw), n=3, warm=1)
        # time of the initial fit alone
        ff = ForecasterRecursive(estimator=estimators()[name], lags=LAGS)
        tfit = timeit(lambda: ff.fit(y.iloc[:N - 50*STEPS], exog=None if exog is None else exog.iloc[:N - 50*STEPS]), n=3, warm=1)
        tot, out, st = profile_breakdown(lambda: backtesting_forecaster(**kw), n_rep=1)
        print(f"exog={kind} {name} refit={refit} interval={interval}: backtesting={t*1e3:.1f}ms (initial fit {tfit*1e3:.1f}ms) | profile {fmt_breakdown(tot, out)}")
        if top:
            st.sort_stats("cumulative").print_stats(top)
