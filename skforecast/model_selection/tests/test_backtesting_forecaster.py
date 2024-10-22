# Unit test backtesting_forecaster
# ==============================================================================
import re
import pytest
from sklearn.linear_model import Ridge
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection._split import TimeSeriesFold

# Fixtures
from .fixtures_model_selection import y


def test_backtesting_forecaster_TypeError_when_forecaster_not_supported_types():
    """
    Test TypeError is raised in backtesting_forecaster if Forecaster is not one 
    of the types supported by the function.
    """
    forecaster = ForecasterAutoregMultiSeries(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    forecaters_allowed = [
        'ForecasterAutoreg', 
        'ForecasterAutoregDirect',
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
        (f"`forecaster` must be of type {forecaters_allowed}, for all other types of "
         f" forecasters use the functions available in the other `model_selection` "
         f"modules.")
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


def test_backtesting_forecaster_ValueError_when_ForecasterAutoregDirect_not_enough_steps():
    """
    Test ValueError is raised in backtesting_forecaster when there is not enough 
    steps to predict steps+gap in a ForecasterAutoregDirect.
    """
    forecaster = ForecasterAutoregDirect(
                    regressor = Ridge(random_state=123),
                    lags      = 5,
                    steps     = 5
                 )
    cv = TimeSeriesFold(
            steps                 = 1,
            initial_train_size    = len(y[:-12]),
            window_size           = None,
            differentiation       = None,
            refit                 = False,
            fixed_train_size      = True,
            gap                   = 5,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    err_msg = re.escape(
        (f"When using a ForecasterAutoregDirect, the combination of steps "
         f"+ gap ({cv.steps + cv.gap}) cannot be greater than the `steps` parameter "
         f"declared when the forecaster is initialized ({forecaster.max_step}).")
    )
    with pytest.raises(ValueError, match = err_msg):
        backtesting_forecaster(
            forecaster            = forecaster,
            y                     = y,
            metric                = 'mean_absolute_error',
            exog                  = None,
            cv                    = cv,
            verbose               = False,
            show_progress         = False
        )