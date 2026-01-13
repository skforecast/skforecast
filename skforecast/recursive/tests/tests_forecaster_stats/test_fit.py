# Unit test fit ForecasterStats
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.stats import Sarimax, Arar, Arima, Ets
from skforecast.recursive import ForecasterStats
from skforecast.exceptions import IgnoredArgumentWarning

# Fixtures
from .fixtures_forecaster_stats import y
from .fixtures_forecaster_stats import y_datetime


def test_fit_ValueError_when_len_exog_is_not_the_same_as_len_y():
    """
    Raise ValueError if the length of `exog` is different from the length of `y`.
    """
    y = pd.Series(data=np.arange(10), name='y')
    exog = pd.Series(data=np.arange(11), name='exog')
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 1, 1)))

    err_msg = re.escape(
        f"`exog` must have same number of samples as `y`. "
        f"length `exog`: ({len(exog)}), length `y`: ({len(y)})"
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.fit(y=y, exog=exog)


def test_IgnoredArgumentWarning_when_estimators_do_not_support_exog():
    """
    Test IgnoredArgumentWarning is raised when a estimators do not 
    support exog.
    """
    y = pd.Series(data=np.arange(10), name='y')
    exog = pd.Series(data=np.arange(10), name='exog')
    forecaster = ForecasterStats(estimator=Ets())

    warn_msg = re.escape(
        "The following estimators do not support exogenous variables "
        "and will ignore them during fit: ['skforecast.Ets']"
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        forecaster.fit(y=y, exog=exog)
    
    estimators = [Arima(order=(1, 0, 1)), Ets()]
    forecaster = ForecasterStats(estimator=estimators)

    # Use regex to match estimator_id pattern like Ets(ZZZ)
    warn_msg = re.escape(
        "The following estimators do not support exogenous variables "
        "and will ignore them during fit: ['skforecast.Ets']"
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        forecaster.fit(y=y, exog=exog)


def test_forecaster_y_exog_features_stored():
    """
    Test forecaster stores y and exog features after fitting.
    """
    y = pd.Series(data=np.arange(10), name='y_sarimax')
    exog = pd.Series(data=np.arange(10), name='exog')

    estimators = [Sarimax(order=(1, 1, 1)), Ets()]
    forecaster = ForecasterStats(estimator=estimators)
    forecaster.fit(y=y, exog=exog)

    is_fitted = True
    estimator_ids = ['skforecast.Sarimax', 'skforecast.Ets']
    estimator_names_ = ['Sarimax(1,1,1)(0,0,0)[0]', 'Ets(AAN)']
    series_name_in_ = 'y_sarimax'
    exog_in_ = True
    exog_type_in_ = type(exog)
    exog_names_in_ = ['exog']
    exog_dtypes_in_ = {'exog': exog.dtype}
    exog_dtypes_out_ = {'exog': exog.dtype}
    X_train_exog_names_out_ = ['exog']
    
    assert forecaster.is_fitted == is_fitted
    assert forecaster.estimator_ids == estimator_ids
    assert forecaster.estimator_names_ == estimator_names_
    assert forecaster.series_name_in_ == series_name_in_
    assert forecaster.exog_in_ == exog_in_
    assert forecaster.exog_type_in_ == exog_type_in_
    assert forecaster.exog_names_in_ == exog_names_in_
    assert forecaster.exog_dtypes_in_ == exog_dtypes_in_
    assert forecaster.exog_dtypes_out_ == exog_dtypes_out_
    assert forecaster.X_train_exog_names_out_ == X_train_exog_names_out_


def test_forecaster_DatetimeIndex_index_freq_stored():
    """
    Test serie_with_DatetimeIndex.index.freq is stored in forecaster.index_freq_.
    """
    serie_with_DatetimeIndex = pd.Series(
        data  = [1, 2, 3, 4, 5],
        index = pd.date_range(start='2022-01-01', periods=5)
    )
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 0)))
    forecaster.fit(y=serie_with_DatetimeIndex)
    expected = serie_with_DatetimeIndex.index.freq
    results = forecaster.index_freq_

    assert results == expected


@pytest.mark.parametrize("suppress_warnings", 
                         [True, False], 
                         ids=lambda v: f'suppress_warnings: {v}')
def test_forecaster_index_step_stored_with_suppress_warnings(suppress_warnings):
    """
    Test serie without DatetimeIndex, step is stored in forecaster.index_freq_.
    """
    y = pd.Series(data=np.arange(10))
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 0)))
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
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 0)))
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
                         ids=lambda values: f'y, index: {type(values)}')
def test_fit_extended_index_stored_with_sarimax(y, idx):
    """
    Test that extended_index_ is stored from SARIMAX fitted values index.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 0)))
    forecaster.fit(y=y)

    pd.testing.assert_index_equal(forecaster.extended_index_, idx)


def test_fit_extended_index_stored_without_sarimax():
    """
    Test that extended_index_ is stored from y.index when no SARIMAX estimator.
    """
    y = pd.Series(data=np.arange(50), name='y')
    forecaster = ForecasterStats(estimator=Arima(order=(1, 0, 0)))
    forecaster.fit(y=y)

    pd.testing.assert_index_equal(forecaster.extended_index_, y.index)


def test_fit_all_estimators_fitted_multiple():
    """
    Test that all estimators are fitted when using multiple estimators.
    """
    y = pd.Series(data=np.arange(50), name='y')
    estimators = [Arima(order=(1, 0, 1)), Arar(), Ets()]
    forecaster = ForecasterStats(estimator=estimators)
    forecaster.fit(y=y)

    assert forecaster.is_fitted
    assert len(forecaster.estimators_) == 3
    # estimators_ contains fresh copies, different from original estimators
    for i, est in enumerate(forecaster.estimators_):
        assert est is not forecaster.estimators[i]


def test_fit_estimator_names_updated_after_fit():
    """
    Test that estimator_names_ are regenerated after fit (some models like 
    Ets update their estimator_id after fitting).
    """
    y = pd.Series(data=np.arange(50), name='y')
    forecaster = ForecasterStats(estimator=Ets())
    
    # Before fit, [None]
    names_before = forecaster.estimator_names_.copy()
    
    forecaster.fit(y=y)
    
    # After fit, Ets should have updated name with model config
    names_after = forecaster.estimator_names_
    
    # The name should be different (Ets updates its estimator_id after fit)
    assert names_before != names_after
    # The new name should contain error/trend/season config like 'Ets(AAN)'
    assert 'Ets(' in names_after[0]
