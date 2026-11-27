"""T10: how many DataTypeWarning does a backtesting with refit (and a one-step-ahead
grid search) issue with categorical_features=None and a category exog? And with
suppress_warnings=True?"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
banner()
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import TimeSeriesFold, OneStepAheadFold, backtesting_forecaster_multiseries, grid_search_forecaster_multiseries
from skforecast.exceptions import DataTypeWarning

series, exog_full, _ = make_case("df", ["f", "ci"], effect_col="ci", exog_kind="df")
for sup in [False, True]:
    f = ForecasterRecursiveMultiSeries(make_estimator("ridge"), lags=3, categorical_features=None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        backtesting_forecaster_multiseries(f, series, cv=TimeSeriesFold(steps=5, initial_train_size=80, refit=True),
                                           exog=exog_full, metric="mean_absolute_error", n_jobs=1,
                                           show_progress=False, suppress_warnings=sup)
    print("backtesting refit (8 folds) suppress_warnings=", sup, "DataTypeWarning count:",
          sum(issubclass(x.category, DataTypeWarning) for x in w))
    f = ForecasterRecursiveMultiSeries(make_estimator("ridge"), lags=3, categorical_features=None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        grid_search_forecaster_multiseries(f, series, cv=OneStepAheadFold(initial_train_size=80), exog=exog_full,
                                           param_grid={"alpha": [0.1, 1, 10]}, lags_grid=[3, 5],
                                           metric="mean_absolute_error", return_best=True, verbose=False,
                                           show_progress=False, suppress_warnings=sup)
    print("grid OSA 2 lags x 3 + return_best, suppress_warnings=", sup, "DataTypeWarning count:",
          sum(issubclass(x.category, DataTypeWarning) for x in w))
