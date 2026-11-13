# Unit test _infer_freq_from_index
# ==============================================================================
import pytest
import pandas as pd
from skforecast.utils.utils import _infer_freq_from_index


def _without_freq(index):
    """
    Return the same dates without frequency.
    """
    return pd.DatetimeIndex(index.to_list())


@pytest.mark.parametrize(
    'index, expected',
    [
        (pd.date_range('2025-01-01', periods=10, freq='h'), pd.offsets.Hour()),
        (pd.date_range('2025-01-01', periods=10, freq='MS'), pd.offsets.MonthBegin()),
        (
            pd.date_range('2025-10-15', periods=20, freq='D', tz='Europe/Madrid'),
            pd.offsets.Day()
        ),
        (
            pd.date_range('2025-06-01', periods=8, freq='MS', tz='UTC').tz_convert('America/New_York'),
            pd.offsets.MonthBegin()
        ),
        (
            pd.date_range('2025-10-15', periods=20, freq='D', tz='UTC').tz_convert('Europe/Madrid'),
            pd.offsets.Day()
        ),
    ],
    ids=[
        'naive hourly',
        'naive month start',
        'local calendar daily across DST',
        'UTC anchored month start west of UTC',
        'UTC anchored daily across DST',
    ]
)
def test_output_infer_freq_from_index(index, expected):
    """
    Test the frequency is inferred in local time, or on the UTC timestamps
    when the index advances in fixed UTC steps.
    """
    result = _infer_freq_from_index(_without_freq(index))

    assert result == expected


@pytest.mark.parametrize(
    'index',
    [
        pd.date_range('2025-01-01', periods=2, freq='D'),
        pd.DatetimeIndex(['2025-01-01', '2025-01-02', '2025-01-04', '2025-01-05']),
    ],
    ids=['fewer than 3 dates', 'irregular dates']
)
def test_output_infer_freq_from_index_is_None_when_frequency_cannot_be_inferred(index):
    """
    Test None is returned when the index has fewer than 3 dates or they are
    not regularly spaced.
    """
    result = _infer_freq_from_index(_without_freq(index))

    assert result is None
