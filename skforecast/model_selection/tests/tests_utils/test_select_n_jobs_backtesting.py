# Unit test select_n_jobs_backtesting
# ==============================================================================
import pytest
from joblib import cpu_count
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from skforecast.model_selection._utils import select_n_jobs_backtesting
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.stats import Sarimax
from skforecast.recursive import ForecasterStats
from skforecast.recursive import ForecasterEquivalentDate


@pytest.mark.parametrize("forecaster, refit, n_jobs_expected", 
    [(ForecasterRecursive(estimator=LinearRegression(), lags=2), True, 1),
     (ForecasterRecursive(estimator=make_pipeline(StandardScaler(), LinearRegression()), lags=2), True, 1),
     (ForecasterRecursive(estimator=LinearRegression(), lags=2), 2, 1),
     (ForecasterRecursive(estimator=LinearRegression(), lags=2), False, 1),
     (ForecasterRecursive(estimator=HistGradientBoostingRegressor(), lags=2), 1, cpu_count() - 1),
     (ForecasterRecursive(estimator=HistGradientBoostingRegressor(), lags=2), 2, 1),
     (ForecasterRecursive(estimator=HistGradientBoostingRegressor(), lags=2), 0, cpu_count() - 1),
     (ForecasterRecursive(estimator=LGBMRegressor(n_jobs=1), lags=2), True, cpu_count() - 1),
     (ForecasterRecursive(estimator=LGBMRegressor(n_jobs=1), lags=2), False, cpu_count() - 1),
     (ForecasterRecursive(estimator=LGBMRegressor(), lags=2), True, 1),
     (ForecasterRecursive(estimator=LGBMRegressor(), lags=2), False, 1),
     (ForecasterDirect(estimator=LinearRegression(), steps=3, lags=2), True, 1),
     (ForecasterDirect(estimator=LinearRegression(), steps=3, lags=2), 2, 1),
     (ForecasterDirect(estimator=LinearRegression(), steps=3, lags=2), False, 1),
     (ForecasterDirect(estimator=HistGradientBoostingRegressor(), steps=3, lags=2), 1, 1),
     (ForecasterDirect(estimator=HistGradientBoostingRegressor(), steps=3, lags=2), 2, 1),
     (ForecasterDirect(estimator=HistGradientBoostingRegressor(), steps=3, lags=2), 0, 1),
     (ForecasterDirect(estimator=LGBMRegressor(), steps=3, lags=2), True, 1),
     (ForecasterDirect(estimator=LGBMRegressor(), steps=3, lags=2), False, 1),
     (ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=2), True, cpu_count() - 1),
     (ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=2), 2, 1),
     (ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=2), False, cpu_count() - 1),
     (ForecasterRecursiveMultiSeries(estimator=HistGradientBoostingRegressor(), lags=2), 1, cpu_count() - 1),
     (ForecasterRecursiveMultiSeries(estimator=HistGradientBoostingRegressor(), lags=2), 2, 1),
     (ForecasterRecursiveMultiSeries(estimator=HistGradientBoostingRegressor(), lags=2), 0, cpu_count() - 1),
     (ForecasterRecursiveMultiSeries(estimator=LGBMRegressor(n_jobs=1), lags=2), True, cpu_count() - 1),
     (ForecasterRecursiveMultiSeries(estimator=LGBMRegressor(n_jobs=1), lags=2), False, cpu_count() - 1),
     (ForecasterRecursiveMultiSeries(estimator=LGBMRegressor(), lags=2), True, 1),
     (ForecasterRecursiveMultiSeries(estimator=LGBMRegressor(), lags=2), False, 1),
     (ForecasterStats(estimator=Sarimax((1, 0, 1))), True, 1),
     (ForecasterStats(estimator=Sarimax((1, 0, 1))), 2, 1),
     (ForecasterStats(estimator=Sarimax((1, 0, 1))), False, 1),
     (ForecasterEquivalentDate(2), True, 1),
     (ForecasterEquivalentDate(2), 2, 1),
     (ForecasterEquivalentDate(2), False, 1)
], 
ids=lambda info: f'info: {info}')
def test_select_n_jobs_backtesting(forecaster, refit, n_jobs_expected):
    """
    Test select_n_jobs_backtesting
    """
    n_jobs = select_n_jobs_backtesting(
                 forecaster = forecaster, 
                 refit      = refit
             )
    
    assert n_jobs == n_jobs_expected
