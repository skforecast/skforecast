# Unit test select_n_jobs_fit_forecaster
# ==============================================================================
import pytest
from joblib import cpu_count
from skforecast.utils.utils import select_n_jobs_fit_forecaster


@pytest.mark.parametrize("forecaster_name, regressor_name, n_jobs_expected", 
    [('ForecasterDirect', 'LinearRegression', 1),
    ('ForecasterDirect', 'LGBMRegressor', 1),
    ('ForecasterDirect', 'RandomForestRegressor', cpu_count() - 1),
    ('ForecasterDirectMultiVariate', 'LinearRegression', 1),
    ('ForecasterDirectMultiVariate', 'LGBMRegressor', 1),
    ('ForecasterDirectMultiVariate', 'RandomForestRegressor', cpu_count() - 1),
    ('ForecasterRecursive', 'LinearRegression', 1),
    ('ForecasterRecursive', 'RandomForestRegressor', 1),
    ('ForecasterRecursive', 'LGBMRegressor', 1)
], 
ids=lambda info: f'info: {info}')
def test_select_n_jobs_fit_forecaster(forecaster_name, regressor_name, n_jobs_expected):
    """
    Test select_n_jobs_fit_forecaster
    """
    n_jobs = select_n_jobs_fit_forecaster(
                 forecaster_name = forecaster_name, 
                 regressor_name  = regressor_name
             )
    
    assert n_jobs == n_jobs_expected