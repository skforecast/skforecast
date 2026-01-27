# Unit test _check_append_last_window ForecasterStats
# ==============================================================================
import re
import pytest
import pandas as pd

from skforecast.stats import Sarimax, Arar
from skforecast.recursive import ForecasterStats
from skforecast.exceptions import IgnoredArgumentWarning

# Fixtures
from .fixtures_forecaster_stats import y
from .fixtures_forecaster_stats import y_lw
from .fixtures_forecaster_stats import exog
from .fixtures_forecaster_stats import exog_lw


def test_check_append_last_window_NotImplementedError_when_no_sarimax_estimator():
    """
    Test NotImplementedError is raised when no SARIMAX estimator is present.
    """
    forecaster = ForecasterStats(estimator=Arar())
    forecaster.fit(y=y)
    
    err_msg = re.escape(
        "Prediction with `last_window` parameter is only supported for "
        "skforecast.Sarimax estimator. The forecaster does not contain any "
        "estimator that supports this feature."
    )
    with pytest.raises(NotImplementedError, match=err_msg):
        forecaster._check_append_last_window(
            steps            = 5,
            last_window      = y_lw,
            last_window_exog = None
        )


def test_check_append_last_window_IgnoredArgumentWarning_when_mixed_estimators():
    """
    Test IgnoredArgumentWarning is raised when there are estimators that 
    don't support last_window.
    """
    forecaster = ForecasterStats(
        estimator=[Sarimax(order=(1, 0, 0)), Arar()]
    )
    forecaster.fit(y=y)
    
    warn_msg = re.escape(
        f"Prediction with `last_window` is not implemented for estimators: ['skforecast.Arar']. "
        f"These estimators will be skipped. Available estimators for prediction "
        f"using `last_window` are: {list(forecaster.estimators_support_last_window)}."
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        forecaster._check_append_last_window(
            steps            = 5,
            last_window      = y_lw,
            last_window_exog = None
        )


def test_check_append_last_window_returns_correct_prediction_index():
    """
    Test that _check_append_last_window returns correct prediction_index 
    and updates extended_index_.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 0)))
    forecaster.fit(y=y)
    
    steps = 5
    prediction_index = forecaster._check_append_last_window(
        steps            = steps,
        last_window      = y_lw,
        last_window_exog = None
    )
    expected_prediction_index = pd.RangeIndex(start=100, stop=105, step=1)
    
    assert forecaster.extended_index_ is not None
    assert forecaster.extended_index_[-1] == 99
    pd.testing.assert_index_equal(prediction_index, expected_prediction_index)


def test_check_append_last_window_with_exog():
    """
    Test that _check_append_last_window works correctly with exogenous variables.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 0)))
    forecaster.fit(y=y, exog=exog)
    
    steps = 3
    prediction_index = forecaster._check_append_last_window(
        steps            = steps,
        last_window      = y_lw,
        last_window_exog = exog_lw
    )
    expected_prediction_index = pd.RangeIndex(start=100, stop=103, step=1)
    
    assert forecaster.extended_index_ is not None
    assert forecaster.extended_index_[-1] == 99
    pd.testing.assert_index_equal(prediction_index, expected_prediction_index)
