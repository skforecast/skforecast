# Unit test fit
# ==============================================================================
import pytest
import pandas as pd
import numpy as np
import warnings
from skforecast.drift_detection import RangeDriftDetector


def test_fit_series():
    """
    Test fit with pandas Series.
    """
    series = pd.Series([1, 2, 3, 4, 5], name='test_series')
    detector = RangeDriftDetector()
    detector.fit(series=series)

    assert detector.is_fitted_ is True
    assert detector.series_names_in_ == ['test_series']
    assert detector.series_values_range_ == {'test_series': (1.0, 5.0)}
    assert detector.exog_values_range_ is None
    assert detector.exog_names_in_ is None


def test_fit_dataframe():
    """
    Test fit with pandas DataFrame.
    """
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    detector = RangeDriftDetector()
    detector.fit(series=df)

    assert detector.is_fitted_ is True
    assert set(detector.series_names_in_) == {'col1', 'col2'}
    assert detector.series_values_range_ == {'col1': (1.0, 3.0), 'col2': (4.0, 6.0)}


def test_fit_dict():
    """
    Test fit with dict of Series.
    """
    series = {
        's1': pd.Series([1, 2, 3], name='s1'),
        's2': pd.Series([10, 20, 30], name='s2')
    }
    detector = RangeDriftDetector()
    detector.fit(series=series)

    assert detector.is_fitted_ is True
    assert set(detector.series_names_in_) == {'s1', 's2'}
    assert detector.series_values_range_ == {
        's1': (1.0, 3.0),
        's2': (10.0, 30.0)
    }


def test_fit_with_exog_series():
    """
    Test fit with exogenous Series.
    """
    series = pd.Series([1, 2, 3], name='series1')
    exog = pd.Series([10, 20, 30], name='exog1')
    detector = RangeDriftDetector()
    detector.fit(series=series, exog=exog)

    assert detector.is_fitted_ is True
    assert detector.series_names_in_ == ['series1']
    assert detector.series_values_range_ == {'series1': (1.0, 3.0)}
    assert detector.exog_names_in_ == ['exog1']
    assert detector.exog_values_range_ == {'exog1': (10.0, 30.0)}


def test_fit_with_exog_dataframe():
    """
    Test fit with exogenous DataFrame.
    """
    series = pd.Series([1, 2, 3], name='series1')
    exog = pd.DataFrame({'exog1': [10, 20, 30], 'exog2': [100, 200, 300]})
    detector = RangeDriftDetector()
    detector.fit(series=series, exog=exog)

    assert detector.is_fitted_ is True
    assert detector.exog_names_in_ == ['exog1', 'exog2']
    assert detector.exog_values_range_ == {
        'exog1': (10.0, 30.0),
        'exog2': (100.0, 300.0)
    }


def test_fit_with_exog_dict():
    """
    Test fit with exogenous dict.
    """
    series = pd.Series([1, 2, 3], name='series1')
    exog = {
        's1': pd.DataFrame({'exog1': [10, 20], 'exog2': [100, 200]}, index=[0, 1])
    }
    detector = RangeDriftDetector()
    detector.fit(series=series, exog=exog)

    assert detector.is_fitted_ is True
    assert detector.exog_names_in_ == ['exog1', 'exog2']
    expected_ranges = {
        'exog1': (10.0, 20.0),
        'exog2': (100.0, 200.0)
    }
    assert detector.exog_values_range_ == {'s1': expected_ranges}


def test_fit_deprecated_y_argument():
    """
    Test fit with deprecated 'y' argument.
    """
    series = pd.Series([1, 2, 3], name='test_series')
    detector = RangeDriftDetector()

    with pytest.warns(DeprecationWarning, match="'y' is deprecated"):
        detector.fit(y=series)

    assert detector.is_fitted_ is True
    assert detector.series_names_in_ == ['test_series']


def test_fit_series_none():
    """
    Test fit raises ValueError when series is None.
    """
    detector = RangeDriftDetector()
    with pytest.raises(ValueError, match="`series` cannot be None"):
        detector.fit(series=None)


def test_fit_both_series_and_y():
    """
    Test fit raises TypeError when both series and y are provided.
    """
    series = pd.Series([1, 2, 3], name='test_series')
    detector = RangeDriftDetector()
    with pytest.raises(TypeError, match="Cannot specify both 'series' and 'y'"):
        detector.fit(series=series, y=series)


def test_fit_invalid_series_type():
    """
    Test fit raises TypeError for invalid series type.
    """
    detector = RangeDriftDetector()
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame, Series or dict"):
        detector.fit(series="invalid")


def test_fit_invalid_exog_type():
    """
    Test fit raises TypeError for invalid exog type.
    """
    series = pd.Series([1, 2, 3], name='test_series')
    detector = RangeDriftDetector()
    with pytest.raises(TypeError, match="Exogenous variables must be a pandas DataFrame, Series or dict"):
        detector.fit(series=series, exog="invalid")


def test_fit_categorical_series():
    """
    Test fit with categorical Series.
    """
    series = pd.Series(['a', 'b', 'c', 'a'], name='cat_series')
    detector = RangeDriftDetector()
    detector.fit(series=series)

    assert detector.is_fitted_ is True
    assert detector.series_names_in_ == ['cat_series']
    assert detector.series_values_range_ == {'cat_series': {'a', 'b', 'c'}}


def test_fit_mixed_types_dataframe():
    """
    Test fit with DataFrame containing numeric and categorical columns.
    """
    df = pd.DataFrame({
        'numeric': [1, 2, 3],
        'categorical': ['x', 'y', 'z']
    })
    detector = RangeDriftDetector()
    detector.fit(series=df)

    assert detector.is_fitted_ is True
    assert set(detector.series_names_in_) == {'numeric', 'categorical'}
    assert detector.series_values_range_ == {
        'numeric': (1.0, 3.0),
        'categorical': {'x', 'y', 'z'}
    }


def test_fit_exog_names_deduplication():
    """
    Test that exog_names_in_ removes duplicates.
    """
    series = pd.Series([1, 2, 3], name='series1')
    exog = {
        's1': pd.DataFrame({'exog1': [10, 20], 'exog2': [100, 200]}, index=[0, 1]),
        's2': pd.DataFrame({'exog1': [15, 25], 'exog3': [150, 250]}, index=[0, 1])
    }
    detector = RangeDriftDetector()
    detector.fit(series=series, exog=exog)

    assert set(detector.exog_names_in_) == {'exog1', 'exog2', 'exog3'}
