# Unit test backtesting_forecaster
# ==============================================================================
import re
import pytest
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection._split import TimeSeriesFold

# Fixtures
from ..fixtures_model_selection import y


def test_backtesting_forecaster_TypeError_when_forecaster_not_supported_types():
    """
    Test TypeError is raised in backtesting_forecaster if Forecaster is not one 
    of the types supported by the function.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    forecaters_allowed = [
        'ForecasterRecursive', 
        'ForecasterDirect',
        'ForecasterEquivalentDate'
    ]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y[:-12]),
            window_size           = None,
            differentiation       = None,
            refit                 = False,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    err_msg = re.escape(
        f"`forecaster` must be of type {forecaters_allowed}, for all other types of "
        f" forecasters use the functions available in the other `model_selection` "
        f"modules."
    )
    
    with pytest.raises(TypeError, match = err_msg):
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            cv                    = cv,
            metric                = 'mean_absolute_error',
            exog                  = None,
            verbose               = False,
            show_progress         = False
        )
