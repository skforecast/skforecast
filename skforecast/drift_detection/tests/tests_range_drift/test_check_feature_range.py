# Unit test _get_features_range
# ==============================================================================
import pytest
import pandas as pd
import numpy as np
from skforecast.drift_detection import RangeDriftDetector


def test_check_feature_range_numeric_within_range():
    """
    Test _check_feature_range with numeric features within training range.
    """
    # Values within range
    feature_range = (1, 10)
    X = pd.Series([2, 3, 4, 5])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is False

    # Boundary values
    X = pd.Series([1, 10])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is False

    # Single value within range
    X = pd.Series([5])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is False


def test_check_feature_range_numeric_outside_range():
    """
    Test _check_feature_range with numeric features outside training range.
    """
    feature_range = (1, 10)

    # Below minimum
    X = pd.Series([0, 2, 3])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is True

    # Above maximum
    X = pd.Series([5, 11, 12])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is True

    # Both below and above
    X = pd.Series([0, 15])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is True

    # Only below minimum
    X = pd.Series([-5, -1])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is True

    # Only above maximum
    X = pd.Series([11, 20])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is True


def test_check_feature_range_numeric_with_floats():
    """
    Test _check_feature_range with float values.
    """
    feature_range = (1.5, 9.5)

    # Within range
    X = pd.Series([2.0, 3.5, 8.9])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is False

    # Outside range
    X = pd.Series([1.0, 10.0])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is True


def test_check_feature_range_categorical_within_range():
    """
    Test _check_feature_range with categorical features within training range.
    """
    feature_range = {'a', 'b', 'c'}

    # All values seen during training
    X = pd.Series(['a', 'b', 'c', 'a'])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is False

    # Subset of training values
    X = pd.Series(['a', 'b'])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is False

    # Single value
    X = pd.Series(['c'])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is False


def test_check_feature_range_categorical_outside_range():
    """
    Test _check_feature_range with categorical features containing unseen values.
    """
    feature_range = {'a', 'b', 'c'}

    # One unseen value
    X = pd.Series(['a', 'b', 'd'])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is True

    # All unseen values
    X = pd.Series(['d', 'e', 'f'])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is True

    # Mix of seen and unseen
    X = pd.Series(['a', 'x', 'b', 'y'])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is True


def test_check_feature_range_with_NaN_values():
    """
    Test _check_feature_range with NaN values in data.
    """
    # Numeric with NaN - NaN should not affect the check
    feature_range = (1, 10)
    X = pd.Series([2, np.nan, 5])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is False

    X = pd.Series([0, np.nan, 15])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is True

    # Categorical with NaN - NaN is treated as a unique value
    feature_range = {'a', 'b', 'c'}
    X = pd.Series(['a', np.nan, 'b'])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is False


def test_check_feature_range_single_value_ranges():
    """
    Test _check_feature_range with single value ranges.
    """
    # Numeric range with same min/max
    feature_range = (5, 5)
    X = pd.Series([5])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is False

    X = pd.Series([4])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is True

    # Categorical with single value
    feature_range = {'a'}
    X = pd.Series(['a'])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is False

    X = pd.Series(['b'])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is True


def test_check_feature_range_negative_values():
    """
    Test _check_feature_range with negative values.
    """
    feature_range = (-10, -1)

    # Within range
    X = pd.Series([-5, -3, -7])
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is False

    # Outside range
    X = pd.Series([-15, -12])  # Below min
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is True

    X = pd.Series([0, 1])  # Above max
    result = RangeDriftDetector._check_feature_range(feature_range, X)
    assert result is True
