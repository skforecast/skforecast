# Unit test date_to_index_position
# ==============================================================================
import re
import pytest
import pandas as pd
from skforecast.utils import date_to_index_position


def test_TypeError_date_to_index_position_when_index_is_not_DatetimeIndex():
    """
    Test TypeError is raised when `date_input` is a date but the index is not 
    a DatetimeIndex.
    """
    index = pd.RangeIndex(start=0, stop=3, step=1)
    
    err_msg = re.escape(
        "Index must be a pandas DatetimeIndex when `steps` is not an integer. "
        "Check input series or last window."
    )
    with pytest.raises(TypeError, match=err_msg):
        date_to_index_position(index, date_input='1990-01-10')


def test_ValueError_date_to_index_position_when_date_is_before_last_index():
    """
    Test ValueError is raised when the provided date is earlier than or equal 
    to the last date in index.
    """
    index = pd.date_range(start='1990-01-01', periods=3, freq='D')
    
    err_msg = re.escape(
        "The provided date must be later than the last date in the index."
    )
    with pytest.raises(ValueError, match=err_msg):
        date_to_index_position(index, date_input='1990-01-02')


def test_TypeError_date_to_index_position_when_date_input_is_not_int_str_or_Timestamp():
    """
    Test TypeError is raised when `date_input` is not a int, str or pd.Timestamp.
    """
    index = pd.date_range(start='1990-01-01', periods=3, freq='D')
    date_input = 2.5
    date_literal = 'initial_train_size'
    
    err_msg = re.escape(
        "`initial_train_size` must be an integer, string, or pandas Timestamp."
    )
    with pytest.raises(TypeError, match=err_msg):
        date_to_index_position(index, date_input=date_input, date_literal=date_literal)


@pytest.mark.parametrize("date_input", 
                         ['1990-01-07', pd.Timestamp('1990-01-07'), 4], 
                         ids = lambda date_input: f'date_input: {type(date_input)}')
def test_output_date_to_index_position_with_different_date_input_types(date_input):
    """
    Test values returned by date_to_index_position with different date_input types.
    """
    index = pd.date_range(start='1990-01-01', periods=3, freq='D')
    results = date_to_index_position(index, date_input=date_input)

    expected = 4
    
    assert results == expected


def test_output_date_to_index_position_when_date_input_is_string_date_with_kwargs_pd_to_datetime():
    """
    Test values returned by date_to_index_position when `date_input` is a string 
    date and `kwargs_pd_to_datetime` are passed.
    """
    index = pd.date_range(start='1990-01-01', periods=3, freq='D')
    results = date_to_index_position(
        index, date_input='1990-07-01', kwargs_pd_to_datetime={'format': '%Y-%d-%m'}
    )
    
    expected = 4
    
    assert results == expected
