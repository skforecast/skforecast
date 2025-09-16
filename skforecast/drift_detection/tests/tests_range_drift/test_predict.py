# Unit test predict
# ==============================================================================
import pytest
import pandas as pd
import numpy as np
import warnings
from skforecast.drift_detection import RangeDriftDetector
from skforecast.exceptions import (
    FeatureOutOfRangeWarning,
    UnknownLevelWarning,
    MissingExogWarning
)


def test_predict_not_fitted():
    """
    Test predict raises RuntimeError when detector is not fitted.
    """
    detector = RangeDriftDetector()
    with pytest.raises(RuntimeError, match="Model is not fitted yet."):
        detector.predict()


def test_predict_invalid_last_window_type():
    """
    Test predict raises TypeError for invalid last_window type.
    """
    detector = RangeDriftDetector()
    detector.is_fitted = True
    with pytest.raises(TypeError, match="last_window must be a pandas DataFrame, Series, dict or None."):
        detector.predict(last_window="invalid")


def test_predict_invalid_exog_type():
    """
    Test predict raises TypeError for invalid exog type.
    """
    detector = RangeDriftDetector()
    detector.is_fitted = True 
    with pytest.raises(TypeError, match="Exogenous variables must be a pandas DataFrame, Series, dict or None."):
        detector.predict(exog="invalid")


def test_predict_no_out_of_range():
    """
    Test predict with data within training ranges.
    """
    
    series = pd.Series([1, 2, 3, 4, 5], name='series_1')
    exog = pd.DataFrame({'exog1': [10, 20, 30]})
    detector = RangeDriftDetector()
    detector.fit(series=series, exog=exog)

    last_window = pd.Series([2, 3, 4], name='series_1')
    exog_pred = pd.DataFrame({'exog1': [15, 25]})

    flag, out_series, out_exog = detector.predict(
        last_window=last_window, exog=exog_pred, verbose=False
    )

    assert flag is False
    assert out_series == []
    assert out_exog == []


def test_predict_out_of_range_series_pandas_series():
    """
    Test predict with series values out of training range.
    Series is a single pd.Series.
    """
    series = pd.Series([1, 2, 3, 4, 5], name='series_1')
    detector = RangeDriftDetector()
    detector.fit(series=series)

    last_window = pd.Series([0, 6], name='series_1')

    with pytest.warns(FeatureOutOfRangeWarning):
        flag, out_series, out_exog = detector.predict(
            last_window=last_window, verbose=False
        )

    assert flag is True
    assert out_series == ['series_1']
    assert out_exog == []


def test_predict_out_of_range_series_pandas_dataframe():
    """
    Test predict with series values out of training range.
    Series is a pandas dataframe with multiple columns.
    """
    series = pd.DataFrame({
        'series_1': [1, 2, 3, 4, 5],
        'series_2': [10, 20, 30, 40, 50],
        'series_3': [100, 200, 300, 400, 500]
    })
    detector = RangeDriftDetector()
    detector.fit(series=series)

    last_window = pd.DataFrame({
        'series_1': [0, 6],  # out of range
        'series_2': [15, 25],  # in range
        'series_3': [600, 250]  # out of range
    })

    with pytest.warns(FeatureOutOfRangeWarning):
        flag, out_series, out_exog = detector.predict(
            last_window=last_window, verbose=False
        )

    assert flag is True
    assert out_series == ['series_1', 'series_3']
    assert out_exog == []


def test_predict_out_of_range_series_pandas_dataframe_multiindex():
    """
    Test predict with series values out of training range.
    Series is a pandas dataframe multiindex.
    """
    index = pd.MultiIndex.from_product(
                [['series_1', 'series_2', 'series_3'], range(3)],
                names=['series', 'time']
            )
    series = pd.DataFrame({
        'value': [1, 2, 3, 10, 20, 30, 100, 200, 300]
    }, index=index) 

    detector = RangeDriftDetector()
    detector.fit(series=series)

    last_window = pd.DataFrame({
        'series_1': [0, 6],  # out of range
        'series_2': [15, 25],  # in range
        'series_3': [600, 700]  # out of range
    })

    with pytest.warns(FeatureOutOfRangeWarning):
        flag, out_series, out_exog = detector.predict(
            last_window=last_window, verbose=False
        )

    assert flag is True
    assert out_series == ['series_1', 'series_3']
    assert out_exog == []


def test_predict_out_of_range_exog_pandas_series(): 
    """
    Test predict with exogenous values out of training range.
    """
    series = pd.Series([1, 2, 3], name='series_1')
    exog = pd.DataFrame({'exog1': [10, 20, 30]}, index=[0, 1, 2])
    detector = RangeDriftDetector()
    detector.fit(series=series, exog=exog)

    exog_pred = pd.DataFrame({'exog1': [5, 35]}, index=[0, 1])

    with pytest.warns(FeatureOutOfRangeWarning):
        flag, out_series, out_exog = detector.predict(
            exog=exog_pred, verbose=False
        )

    assert flag is True
    assert out_series == []
    assert out_exog == ['exog1']


def test_predict_out_of_range_exog_pandas_dataframe():
    """
    Test predict with exogenous values out of training range when exog is a
    DataFrame with multiple columns.
    """
    series = pd.Series([1, 2, 3], name='series_1')
    exog = pd.DataFrame({'exog1': [10, 20, 30], 'exog2': [100, 200, 300]}, index=[0, 1, 2])
    detector = RangeDriftDetector()
    detector.fit(series=series, exog=exog)

    exog_pred = pd.DataFrame({
        'exog1': [5, 35],  # out of range
        'exog2': [95, 205]},  # out of range
        index=[0, 1]
    )

    with pytest.warns(FeatureOutOfRangeWarning):
        flag, out_series, out_exog = detector.predict(
            exog=exog_pred, verbose=False
        )

    assert flag is True
    assert out_series == []
    assert out_exog == ['exog1', 'exog2']


def test_predict_out_of_range_exog_pandas_dataframe_multiindex():
    """
    Test predict with exogenous values out of training range when exog is a
    DataFrame with MultiIndex.
    """
    series = pd.Series([1, 2, 3], name='series_1')
    index = pd.MultiIndex.from_product(
                [['series_1', 'series_2', 'series_3'], range(3)],
                names=['series', 'time']
            )
    exog = pd.DataFrame(
        {'exog1': [10, 20, 30, 100, 200, 300, 1000, 2000, 3000],
        'exog2': [-10, -20, -30, -100, -200, -300, -1000, -2000, -3000]},
        index=index
    )
    detector = RangeDriftDetector()
    detector.fit(series=series, exog=exog)

    index_pred = pd.MultiIndex.from_product(
        [['series_1', 'series_2', 'series_3'], range(2)],
        names=['series', 'time']
    )
    # Series 1, exog 1: in range
    # Series 1, exog 2: out of range
    # Series 2, exog 1: out of range
    # Series 2, exog 2: in range
    # Series 3, exog 1: out of range
    # Series 3, exog 2: out of range
    exog_pred = pd.DataFrame(
        {'exog1': [15, 25, 400, 500, 4000, 5000],
         'exog2': [-15, -25, -200, -200, -4000, -5000]},
        index=index_pred
    )

    with pytest.warns(FeatureOutOfRangeWarning):
        flag, out_series, out_exog = detector.predict(
            exog=exog_pred, verbose=False
        )

    assert flag is True
    assert out_series == []
    assert out_exog == ['exog1', 'exog1', 'exog2']


def test_predict_unknown_series():
    """
    Test predict with series not seen during training.
    """

    series = pd.Series([1, 2, 3], name='series_1')
    detector = RangeDriftDetector()
    detector.fit(series=series)
    last_window = pd.Series([1, 2], name='unknown_series')

    msg = "'unknown_series' was not seen during training. Its range is unknown."
    with pytest.warns(UnknownLevelWarning, match=msg):
        flag, out_series, out_exog = detector.predict(
            last_window=last_window, verbose=False
        )

    assert flag is False
    assert out_series == []
    assert out_exog == []


def test_predict_unknown_exog():
    """
    Test predict with exogenous variable not seen during training.
    """
    series = pd.Series([1, 2, 3], name='series_1')
    exog = pd.DataFrame({'exog1': [10, 20, 30]}, index=[0, 1, 2])
    detector = RangeDriftDetector()
    detector.fit(series=series, exog=exog)

    exog_pred = pd.DataFrame({'unknown_exog': [1, 2]}, index=[0, 1])
    msg = "'unknown_exog' was not seen during training. Its range is unknown."
    with pytest.warns(MissingExogWarning, match=msg):
        flag, out_series, out_exog = detector.predict(
            exog=exog_pred, verbose=False
        )

    assert flag is False
    assert out_series == []
    assert out_exog == []


def test_predict_suppress_warnings():
    """
    Test predict with suppress_warnings=True suppresses warnings.
    """
    # Fit detector
    series = pd.Series([1, 2, 3], name='series_1')
    detector = RangeDriftDetector()
    detector.fit(series=series)

    # Test data out of range
    last_window = pd.Series([0], name='series_1')

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        flag, out_series, out_exog = detector.predict(
            last_window=last_window, suppress_warnings=True, verbose=False
        )


def test_predict_with_dict_inputs_series_and_exog():
    """
    Test predict with dict inputs for both series and exog.
    """
    # Fit detector
    series = {
        'series_1': pd.Series([1, 2, 3], name='series_1'),
        'series_2': pd.Series([10, 20, 30], name='series_2')
    }
    exog = {
        'series_1': pd.DataFrame({'exog_1': [10, 20], 'exog_2': [-10, -20]}, index=[0, 1]),
        'series_2': pd.DataFrame({'exog_1': [100, 200], 'exog_2': [-100, -200]}, index=[0, 1])
    }
    detector = RangeDriftDetector()
    detector.fit(series=series, exog=exog)

    last_window = {
        'series_1': pd.Series([-10, 0, -5], name='series_1'), # out of range
        'series_2': pd.Series([15, 16, 17], name='series_2')  # in range
    }
    exog_pred = {
        'series_1': pd.DataFrame({
            'exog_1': [11, 12, 13], # in range
            'exog_2': [-5, -1, -0]  # out of range
        }, index=[0, 1, 2]),
        'series_2': pd.DataFrame({
            'exog_1': [100, 110, 120],  # in range
            'exog_2': [-50, -60, -70]   # out of range
        }, index=[0, 1, 2])
    }

    with pytest.warns(FeatureOutOfRangeWarning):
        flag, out_series, out_exog = detector.predict(
            last_window=last_window, exog=exog_pred, verbose=False
        )

    assert flag is True
    assert out_series == ['series_1']
    assert out_exog == ['exog_2', 'exog_2']