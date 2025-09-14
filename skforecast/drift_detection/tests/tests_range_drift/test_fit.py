# Unit test fit
# ==============================================================================
import pytest
import pandas as pd
import numpy as np
import warnings
from skforecast.drift_detection import RangeDriftDetector


def test_fit_series_and_exog_single_series():
    """
    Test fit with series and exog as single Series.
    """
    series = pd.Series([1, 2, 3, 4, 5], name='y')
    exog = pd.Series([10, 20, 30, 40, 50], name='exog')
    detector = RangeDriftDetector()
    detector.fit(series=series, exog=exog)

    assert detector.is_fitted_ is True
    assert detector.series_names_in_ == ['y']
    assert detector.series_values_range_ == {'y': (1.0, 5.0)}
    assert detector.exog_values_range_ == {'exog': (10.0, 50.0)}
    assert detector.exog_names_in_ == ['exog']


def test_fit_series_and_exog_df():
    """
    Test fit with series and exog as DataFrames.
    """
    series = pd.DataFrame({
        'series_1': [1, 2, 3, 4, 5],
        'series_2': [10, 20, 30, 40, 50]
    })
    exog = pd.DataFrame({
        'exog_1': [10, 20, 30, 40, 50],
        'exog_2': ['A', 'B', 'C', 'D', 'E']
    })
    detector = RangeDriftDetector()
    detector.fit(series=series, exog=exog)

    assert detector.is_fitted_ is True
    assert detector.series_names_in_ == ['series_1', 'series_2']
    assert detector.series_values_range_ == {
        'series_1': (1.0, 5.0),
        'series_2': (10.0, 50.0)
    }
    assert detector.exog_names_in_ == ['exog_1', 'exog_2']
    assert detector.exog_values_range_ == {
        'exog_1': (10.0, 50.0),
        'exog_2': set(['A', 'B', 'C', 'D', 'E'])
    }


def test_fit_series_and_exog_df_multindex():
    """
    Test fit with series and exog as DataFrames with MultiIndex.
    """
    index = pd.MultiIndex.from_product(
        [["series_1", "series_2", "series_3"], range(3)], names=["series", "time"]
    )
    series = pd.DataFrame({"value": range(9)}, index=index)
    series = series.stack().to_frame().swaplevel(0, 1).sort_index()
    exog = pd.DataFrame(
        {
            "exog_1": [10, 20, 30, 100, 200, 300, 1000, 2000, 3000],
            "exog_2": ["A", "B", "C", "D", "E", "A", "B", "C", "D"],
        },
        index=index,
    )

    detector = RangeDriftDetector()
    detector.fit(series=series, exog=exog)

    assert detector.is_fitted_ is True
    assert detector.series_names_in_ == ["series_1", "series_2", "series_3"]
    assert detector.series_values_range_ == {
        "series_1": (1.0, 5.0),
        "series_2": (10.0, 50.0),
        "series_3": (100.0, 500.0),
    }
    assert detector.exog_names_in_ == ["exog_1", "exog_2"]
    assert detector.exog_values_range_ == {
        "series_1": {"exog_1": (np.int64(10), np.int64(30)), "exog_2": {"A", "B", "C"}},
        "series_2": {
            "exog_1": (np.int64(100), np.int64(300)),
            "exog_2": {"A", "D", "E"},
        },
        "series_3": {
            "exog_1": (np.int64(1000), np.int64(3000)),
            "exog_2": {"B", "C", "D"},
        },
    }

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
