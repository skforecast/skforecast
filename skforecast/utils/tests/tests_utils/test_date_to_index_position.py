# Unit test date_to_index_position
# ==============================================================================
import re
import pytest
import pandas as pd
from skforecast.utils import date_to_index_position


def test_ValueError_date_to_index_position_when_method_not_valid():
    """
    Test ValueError is raised when `method` is not 'prediction' or 'validation'.
    """
    index = pd.date_range(start='1990-01-01', periods=3, freq='D')
    
    err_msg = re.escape("`method` must be 'prediction' or 'validation'.")
    with pytest.raises(ValueError, match=err_msg):
        date_to_index_position(index, date_input='1990-01-10', method='not_valid')


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
        "If `steps` is a date, it must be greater than the last date "
        "in the index."
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
        date_to_index_position(
            index=index, date_input=date_input, date_literal=date_literal
        )


@pytest.mark.parametrize("date_input", 
                         ['1990-01-07', pd.Timestamp('1990-01-07'), 4], 
                         ids = lambda date_input: f'date_input: {type(date_input)}')
def test_output_date_to_index_position_with_different_date_input_types(date_input):
    """
    Test values returned by date_to_index_position with different date_input types.
    """
    index = pd.date_range(start='1990-01-01', periods=3, freq='D')
    results = date_to_index_position(index=index, date_input=date_input)

    expected = 4
    
    assert results == expected


def test_output_date_to_index_position_when_date_input_is_string_date_with_kwargs_pd_to_datetime():
    """
    Test values returned by date_to_index_position when `date_input` is a string 
    date and `kwargs_pd_to_datetime` are passed.
    """
    index = pd.date_range(start='1990-01-01', periods=3, freq='D')
    results = date_to_index_position(
        index=index, date_input='1990-07-01', kwargs_pd_to_datetime={'format': '%Y-%d-%m'}
    )
    
    expected = 4
    
    assert results == expected


def test_ValueError_date_to_index_position_when_date_is_out_of_range_and_method_is_validation():
    """
    Test ValueError is raised when date_input is out of the index range
    and method is 'validation'.
    """
    index = pd.date_range(start='1990-01-01', periods=3, freq='D')
    
    err_msg = re.escape(
        "If `initial_train_size` is a date, it must be within the index "
        "range, between the first and the last date (both included)."
    )
    with pytest.raises(ValueError, match=err_msg):
        date_to_index_position(
            index        = index,
            date_input   = '1990-01-10',
            method       = 'validation',
            date_literal ='initial_train_size'
        )


def test_output_date_to_index_position_when_date_in_range_and_method_is_validation():
    """
    Test correct position is returned when date_input is within the index range
    and method is 'validation'.
    """
    index = pd.date_range(start='1990-01-01', periods=5, freq='D')
    results = date_to_index_position(
                  index        = index,
                  date_input   = '1990-01-03',
                  method       = 'validation',
                  date_literal ='initial_train_size'
              )

    expected = 3  # iloc position within the range

    assert results == expected


def test_output_date_to_index_position_when_date_is_first_date_and_method_is_validation():
    """
    Test it returns the correct position when date_input is exactly the first date
    in the index and method is 'validation'.
    """
    index = pd.date_range(start='1990-01-01', periods=5, freq='D')
    results = date_to_index_position(
        index=index, date_input='1990-01-01', method='validation', date_literal ='initial_train_size'
    )

    assert results == 1  # iloc position within the range


@pytest.mark.parametrize(
    'start, utc_anchored',
    [('2025-10-01', True), ('2026-03-01', True),
     ('2025-10-01', False), ('2026-03-01', False)],
    ids=lambda x: f'{x}'
)
def test_output_date_to_index_position_when_index_is_tz_aware_and_crosses_dst_change(
    start, utc_anchored
):
    """
    Test it returns the correct position when the index is timezone-aware and
    the range between the index and `date_input` crosses a daylight saving
    change, both when the index advances in fixed UTC steps (created in UTC
    and converted to a local timezone) and in local calendar steps.
    """
    if utc_anchored:
        index = pd.date_range(
            start=start, periods=50, freq='D', tz='UTC'
        ).tz_convert('Europe/Madrid')
    else:
        index = pd.date_range(start=start, periods=50, freq='D', tz='Europe/Madrid')

    results_prediction = date_to_index_position(
        index=index[:20], date_input=index[45], method='prediction'
    )
    results_validation = date_to_index_position(
        index=index, date_input=index[45], method='validation',
        date_literal='initial_train_size'
    )

    assert results_prediction == 26
    assert results_validation == 46


@pytest.mark.parametrize(
    "date_input",
    ['2024-01-05 08:00', pd.Timestamp('2024-01-05 07:00', tz='UTC')],
    ids=['without_time_zone', 'other_time_zone']
)
def test_output_date_to_index_position_when_index_is_tz_aware_and_date_has_other_tz(
    date_input
):
    """
    Test that, with a timezone-aware index, a date without time zone is
    interpreted in the time zone of the index and a date with another time
    zone is converted to it ('2024-01-05 07:00' UTC is '2024-01-05 08:00' in
    Europe/Madrid).
    """
    index = pd.date_range(start='2024-01-01', periods=120, freq='h', tz='Europe/Madrid')

    results_prediction = date_to_index_position(
        index=index[:100], date_input=date_input, method='prediction'
    )
    results_validation = date_to_index_position(
        index=index, date_input=date_input, method='validation',
        date_literal='initial_train_size'
    )

    assert results_prediction == 5
    assert results_validation == 105


def test_ValueError_date_to_index_position_when_date_has_tz_and_index_has_not():
    """
    Test ValueError is raised when `date_input` has a time zone and the index
    does not.
    """
    index = pd.date_range(start='1990-01-01', periods=5, freq='D')

    err_msg = re.escape(
        "`initial_train_size` has a time zone (UTC), but the index has none. "
        "Use a date without time zone."
    )
    with pytest.raises(ValueError, match=err_msg):
        date_to_index_position(
            index        = index,
            date_input   = pd.Timestamp('1990-01-03', tz='UTC'),
            method       = 'validation',
            date_literal = 'initial_train_size'
        )


@pytest.mark.parametrize(
    "index, date_input, expected",
    [(pd.DatetimeIndex(pd.date_range('2020-01-01', periods=100, freq='h').to_list()),
      '2020-01-02 00:00', 25),
     (pd.DatetimeIndex(['2020-01-01', '2020-01-03', '2020-01-04', '2020-01-10']),
      '2020-01-05', 3)],
    ids=['hourly_without_freq', 'irregular']
)
def test_output_date_to_index_position_when_index_has_no_freq_and_method_is_validation(
    index, date_input, expected
):
    """
    Test that, with method 'validation', the position is the number of dates
    in the index up to `date_input` (included) when the index has no frequency.
    """
    assert index.freq is None

    results = date_to_index_position(
                  index        = index,
                  date_input   = date_input,
                  method       = 'validation',
                  date_literal = 'initial_train_size'
              )

    assert results == expected


def test_ValueError_date_to_index_position_when_no_freq_and_method_is_prediction():
    """
    Test ValueError is raised when the index has no frequency and method is
    'prediction', since the number of steps cannot be computed.
    """
    index = pd.DatetimeIndex(
        pd.date_range('2020-01-01', periods=10, freq='h').to_list()
    )

    err_msg = re.escape(
        "If `steps` is a date, the index must have a frequency to compute "
        "the number of steps until that date."
    )
    with pytest.raises(ValueError, match=err_msg):
        date_to_index_position(
            index=index, date_input='2020-01-01 15:00', method='prediction'
        )
