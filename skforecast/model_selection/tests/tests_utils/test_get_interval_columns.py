# Unit test _get_interval_columns
# ==============================================================================
import pytest
from scipy.stats import norm
from skforecast.model_selection._utils import _get_interval_columns


@pytest.mark.parametrize(
    "columns, interval, expected",
    [(['fold', 'pred', 'lower_bound', 'upper_bound'], 0.8, ['lower_bound', 'upper_bound']),
     (['fold', 'pred', 'lower_bound', 'upper_bound'], [0.1, 0.9], ['lower_bound', 'upper_bound']),
     (['fold', 'pred', 'q_0.1', 'q_0.5', 'q_0.9'], [0.1, 0.5, 0.9], ['q_0.1', 'q_0.9']),
     (['fold', 'pred', 'q_0.1', 'q_0.5', 'q_0.9'], (0.9, 0.5, 0.1), ['q_0.1', 'q_0.9']),
     (['level', 'fold', 'q_0.5', 'q_0.9'], [0.5, 0.9], ['q_0.5', 'q_0.9']),
     (['fold', 'pred', 'q_0.1', 'q_0.3', 'q_0.9'], [0.1, 0.3, 0.9], None),
     (['fold', 'pred', 'q_0.05', 'q_0.1', 'q_0.9', 'q_0.95'], [0.05, 0.1, 0.9, 0.95], None),
     (['fold', 'pred', 'pred_boot_0', 'pred_boot_1'], 'bootstrapping', None),
     (['fold', 'pred', 'loc', 'scale'], norm, None),
     (['fold', 'pred'], None, None)],
    ids = lambda dt: f'{dt}'
)
def test_get_interval_columns_output(columns, interval, expected):
    """
    Test output of _get_interval_columns: `lower_bound` and `upper_bound` if
    they exist, the quantile columns when `interval` has 2 quantiles
    (optionally with the median), and None otherwise.
    """
    results = _get_interval_columns(columns=columns, interval=interval)

    assert results == expected
