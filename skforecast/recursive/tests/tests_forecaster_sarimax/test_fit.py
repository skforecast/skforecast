# Unit test fit ForecasterSarimax
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.sarimax import Sarimax
from skforecast.recursive import ForecasterSarimax

# Fixtures
from .fixtures_forecaster_sarimax import y
from .fixtures_forecaster_sarimax import y_datetime


def test_fit_ValueError_when_len_exog_is_not_the_same_as_len_y():
    """
    Raise ValueError if the length of `exog` is different from the length of `y`.
    """
    y = pd.Series(data=np.arange(10))
    exog = pd.Series(data=np.arange(11))
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 1, 1)))

    err_msg = re.escape(
                  (f"`exog` must have same number of samples as `y`. "
                   f"length `exog`: ({len(exog)}), length `y`: ({len(y)})")
              )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(y=y, exog=exog)


def test_forecaster_DatetimeIndex_index_freq_stored():
    """
    Test serie_with_DatetimeIndex.index.freqstr is stored in forecaster.index_freq_.
    """
    serie_with_DatetimeIndex = pd.Series(
        data  = [1, 2, 3, 4, 5],
        index = pd.date_range(start='2022-01-01', periods=5)
    )
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 0, 0)))
    forecaster.fit(y=serie_with_DatetimeIndex)
    expected = serie_with_DatetimeIndex.index.freqstr
    results = forecaster.index_freq_

    assert results == expected


@pytest.mark.parametrize("suppress_warnings", 
                         [True, False], 
                         ids = lambda v: f'suppress_warnings: {v}')
def test_forecaster_index_step_stored_with_suppress_warnings(suppress_warnings):
    """
    Test serie without DatetimeIndex, step is stored in forecaster.index_freq_.
    """
    y = pd.Series(data=np.arange(10))
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 0, 0)))
    forecaster.fit(y=y, suppress_warnings=suppress_warnings)
    expected = y.index.step
    results = forecaster.index_freq_

    assert results == expected


@pytest.mark.parametrize("store_last_window", 
                         [True, False], 
                         ids=lambda lw: f'store_last_window: {lw}')
def test_fit_last_window_stored(store_last_window):
    """
    Test that values of last window are stored after fitting.
    """
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 0, 0)))
    forecaster.fit(y=pd.Series(np.arange(50)), 
                   store_last_window=store_last_window)
    expected = pd.Series(np.arange(50))

    if store_last_window:
        pd.testing.assert_series_equal(forecaster.last_window_, expected)
    else:
        assert forecaster.last_window_ is None


@pytest.mark.parametrize("y          , idx", 
                         [(y         , pd.RangeIndex(start=0, stop=50)), 
                          (y_datetime, pd.date_range(start='2000', periods=50, freq='YE'))], 
                         ids = lambda values: f'y, index: {type(values)}')
def test_fit_extended_index_stored(y, idx):
    """
    Test that values of self.regressor.arima_res_.fittedvalues.index are 
    stored after fitting in forecaster.extended_index_.
    """
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 0, 0)))
    forecaster.fit(y=y)

    pd.testing.assert_index_equal(forecaster.extended_index_, idx)
