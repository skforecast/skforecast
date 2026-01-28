# Unit test reduce_memory ForecasterStats
# ==============================================================================
import re
import pytest
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.stats import Sarimax, Arima, Arar, Ets
from skforecast.recursive import ForecasterStats
from skforecast.exceptions import IgnoredArgumentWarning

# Fixtures
from .fixtures_forecaster_stats import y


def test_NotFittedError_is_raised_when_forecaster_is_not_fitted():
    """
    Test NotFittedError is raised when calling reduce_memory() and 
    forecaster is not fitted.
    """
    forecaster = ForecasterStats(estimator=Arar())

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `reduce_memory()`."
    )
    with pytest.raises(NotFittedError, match=err_msg):
        forecaster.reduce_memory()


def test_reduce_memory_warning_when_estimator_not_supported():
    """
    Test IgnoredArgumentWarning is raised when estimator does not support
    reduce_memory (e.g., Sarimax).
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 0)))
    forecaster.fit(y=y)

    warn_msg = re.escape(
        "Memory reduction is not implemented for estimators: ['skforecast.Sarimax']. "
        "These estimators will be skipped. Available estimators for memory "
        "reduction are:"
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        forecaster.reduce_memory()


@pytest.mark.parametrize(
    'estimator',
    [Arima(order=(1, 0, 0)), Arar(), Ets(model='ANN')],
    ids=['Arima', 'Arar', 'Ets']
)
def test_reduce_memory_supported_estimators(estimator):
    """
    Test reduce_memory method with supported estimators (Arima, Arar, Ets).
    Verifies is_memory_reduced is set to True and predictions still work.
    """
    forecaster = ForecasterStats(estimator=estimator)
    forecaster.fit(y=y)
    
    # Verify estimator is not memory reduced before
    assert not forecaster.estimators_[0].is_memory_reduced
    
    # Get predictions before reduce_memory
    predictions_before = forecaster.predict(steps=5)
    
    forecaster.reduce_memory()
    
    # Verify is_memory_reduced is True after reduce_memory
    assert forecaster.estimators_[0].is_memory_reduced
    
    # Predictions should still work and be the same
    predictions_after = forecaster.predict(steps=5)
    pd.testing.assert_series_equal(predictions_before, predictions_after)


def test_reduce_memory_multiple_estimators_mixed_support():
    """
    Test reduce_memory method with multiple estimators where some support it
    and some do not. Warning should be raised for unsupported estimators.
    """
    forecaster = ForecasterStats(
        estimator=[Sarimax(order=(1, 0, 0)), Arima(order=(1, 0, 0)), Arar(), Ets(model='ANN')]
    )
    forecaster.fit(y=y)
    
    # Verify supported estimators are not memory reduced before
    for i in [1, 2, 3]:  # Arima, Arar, Ets
        assert not forecaster.estimators_[i].is_memory_reduced
    
    warn_msg = re.escape(
        "Memory reduction is not implemented for estimators: ['skforecast.Sarimax']. "
        "These estimators will be skipped. Available estimators for memory "
        "reduction are:"
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        forecaster.reduce_memory()
    
    # Verify is_memory_reduced is True only for supported estimators
    for i in [1, 2, 3]:  # Arima, Arar, Ets
        assert forecaster.estimators_[i].is_memory_reduced
