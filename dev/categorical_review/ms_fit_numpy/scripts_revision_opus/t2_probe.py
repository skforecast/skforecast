import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import OneStepAheadFold, grid_search_forecaster_multiseries
series, exog_train, _ = make_case("df", ["f", "i", "b"], exog_kind="df")
f = ForecasterRecursiveMultiSeries(make_estimator("lgbm"), lags=3)
r = grid_search_forecaster_multiseries(f, series, cv=OneStepAheadFold(initial_train_size=80), exog=exog_train,
    param_grid={}, lags_grid=[3], metric="mean_absolute_error", aggregate_metric="average",
    return_best=False, verbose=False, show_progress=False)
print(r.columns.tolist())
