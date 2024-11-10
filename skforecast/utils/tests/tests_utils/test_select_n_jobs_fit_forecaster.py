# Unit test select_n_jobs_fit_forecaster
# ==============================================================================
import pytest
from joblib import cpu_count
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from skforecast.utils.utils import select_n_jobs_fit_forecaster


@pytest.mark.parametrize("forecaster_name, regressor, n_jobs_expected", 
    [('ForecasterDirect', LinearRegression(), 1),
     ('ForecasterDirect', HistGradientBoostingRegressor(), cpu_count() - 1),
     ('ForecasterDirect', LGBMRegressor(), 1),
     ('ForecasterDirect', LGBMRegressor(n_jobs=1), cpu_count() - 1),
     ('ForecasterDirectMultiVariate', LinearRegression(), 1),
     ('ForecasterDirectMultiVariate', HistGradientBoostingRegressor(), cpu_count() - 1),
     ('ForecasterDirectMultiVariate', LGBMRegressor(), 1),
     ('ForecasterDirectMultiVariate', LGBMRegressor(n_jobs=1), cpu_count() - 1),
     ('ForecasterRecursive', LinearRegression(), 1),
     ('ForecasterRecursive', HistGradientBoostingRegressor(), 1),
     ('ForecasterRecursive', LGBMRegressor(), 1),
     ('ForecasterRecursive', LGBMRegressor(n_jobs=1), 1)
], 
ids=lambda info: f'info: {info}')
def test_select_n_jobs_fit_forecaster(forecaster_name, regressor, n_jobs_expected):
    """
    Test select_n_jobs_fit_forecaster
    """
    n_jobs = select_n_jobs_fit_forecaster(
                 forecaster_name = forecaster_name, 
                 regressor       = regressor
             )
    
    assert n_jobs == n_jobs_expected
