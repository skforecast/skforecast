# Unit test _check_exog_alignment
# ==============================================================================
import pytest
import pandas as pd
from skforecast.utils.utils import _check_exog_alignment
from skforecast.exceptions import MissingValuesWarning


@pytest.mark.parametrize(
    'exog_index, expected',
    [
        (pd.date_range(start='2020-02-20', periods=5, freq='D'), False),
        (pd.date_range(start='2020-02-01', periods=30, freq='D'), False),
        (pd.DatetimeIndex(['2020-02-20', '2020-02-21', '2020-02-22',
                           '2020-02-23', '2020-02-24']), False),
        (pd.DatetimeIndex(['2020-02-20', '2020-02-22', '2020-02-24',
                           '2020-02-26', '2020-02-28']), True),
    ],
    ids=['same_freq', 'starts_before_first_step', 'no_freq', 'misses_dates']
)
def test_check_exog_alignment_output_when_align_by_index(exog_index, expected):
    """
    Test _check_exog_alignment returns True only when `exog` misses some of the
    dates of the steps predicted, the case in which it issues a
    MissingValuesWarning, and False when `exog` has all of them.
    """
    expected_index = pd.date_range(start='2020-02-20', periods=5, freq='D')

    if expected:
        with pytest.warns(MissingValuesWarning):
            results = _check_exog_alignment(
                exog_name      = '`exog`',
                exog_index     = exog_index,
                expected_index = expected_index,
                align_by_index = True
            )
    else:
        results = _check_exog_alignment(
            exog_name      = '`exog`',
            exog_index     = exog_index,
            expected_index = expected_index,
            align_by_index = True
        )

    assert results is expected


@pytest.mark.parametrize(
    'expected_index',
    [
        pd.date_range(start='2020-02-20', periods=5, freq='D'),
        pd.RangeIndex(start=20, stop=25, step=1),
    ],
    ids=['DatetimeIndex', 'RangeIndex']
)
def test_check_exog_alignment_output_False_when_exog_is_used_by_position(
    expected_index
):
    """
    Test _check_exog_alignment returns False when `exog` is used by position
    (`align_by_index=False`) and its first values follow the steps predicted.
    """
    results = _check_exog_alignment(
        exog_name      = '`exog`',
        exog_index     = expected_index.copy(),
        expected_index = expected_index,
        align_by_index = False
    )

    assert results is False
