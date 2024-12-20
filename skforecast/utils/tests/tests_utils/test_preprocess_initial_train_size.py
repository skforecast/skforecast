import pytest
import pandas as pd
from skforecast.utils.utils import _preprocess_initial_train_size

# Unit test preprocess_initial_train_size
# ==============================================================================

def test_integer_initial_train_size_returns_same_integer():
    y = pd.Series([1, 2, 3])
    result = _preprocess_initial_train_size(y, 2)
    assert result == 2

def test_date_initial_train_size_with_datetimeindex():
    y = pd.Series(
        [1, 2, 3, 4, 5],
        index=pd.date_range('2020-01-01', periods=5, freq='D')
    )
    result = _preprocess_initial_train_size(y, '2020-01-02')
    assert result == 2

def test_timestamp_initial_train_size_with_datetimeindex():
    y = pd.Series(
        [1, 2, 3, 4, 5],
        index=pd.date_range('2020-01-01', periods=5, freq='D')
    )
    result = _preprocess_initial_train_size(y, pd.Timestamp('2020-01-03'))
    assert result == 3

def test_string_initial_train_size_invalid_date():
    y = pd.Series(
        [1, 2, 3],
        index=pd.date_range('2020-01-01', periods=3, freq='D')
    )
    with pytest.raises(ValueError):
        _preprocess_initial_train_size(y, 'invalid-date')

def test_initial_train_size_not_in_index():
    y = pd.Series(
        [1, 2, 3],
        index=pd.date_range('2020-01-01', periods=3, freq='D')
    )
    with pytest.raises(ValueError):
        _preprocess_initial_train_size(y, '2020-01-04')

def test_initial_train_size_non_datetimeindex_with_string():
    y = pd.Series(
        [1, 2, 3],
        index=[10, 20, 30]
    )
    with pytest.raises(TypeError):
        _preprocess_initial_train_size(y, '2020-01-02')

def test_initial_train_size_invalid_type():
    y = pd.Series([1, 2, 3])
    with pytest.raises(TypeError):
        _preprocess_initial_train_size(y, [1, 2, 3])

