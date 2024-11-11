# Unit test initialize_lags_grid
# ==============================================================================
import re
import pytest
import numpy as np
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection._utils import initialize_lags_grid


def test_TypeError_initialize_lags__rid_when_not_list_dict_or_None():
    """
    Test TypeError is raised when lags_grid is not a list, dict or None.
    """
    forecaster = ForecasterRecursive(regressor=Ridge(random_state=123), lags=2)
    lags_grid = 'not_valid_type'

    err_msg = re.escape(
        (f"`lags_grid` argument must be a list, dict or None. "
         f"Got {type(lags_grid)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        initialize_lags_grid(forecaster, lags_grid)


def test_initialize_lags_grid_when_lags_grid_is_a_list():
    """
    Test initialize_lags_grid when lags_grid is a list.
    """
    forecaster = ForecasterRecursive(regressor=Ridge(random_state=123), lags=2)
    lags_grid = [1, [2, 4], range(3, 5), np.array([3, 7])]
    lags_grid, lags_label = initialize_lags_grid(forecaster, lags_grid)

    lags_grid_expected = {
        '1': 1, 
        '[2, 4]': [2, 4], 
        'range(3, 5)': range(3, 5), 
        '[3 7]': np.array([3, 7])
    }

    assert lags_label == 'values'
    assert lags_grid.keys() == lags_grid_expected.keys()
    for v, v_expected in zip(lags_grid.values(), lags_grid_expected.values()):
        if isinstance(v, np.ndarray):
            assert np.array_equal(v, v_expected)
        elif isinstance(v, range):
            assert list(v) == list(v_expected)
        else:
            assert v == v_expected


@pytest.mark.parametrize("lags, lags_grid_expected",
                         [(3, {'[1, 2, 3]': [1, 2, 3]}), 
                          ([1, 2, 3], {'[1, 2, 3]': [1, 2, 3]}),
                          (range(1, 4), {'[1, 2, 3]': [1, 2, 3]}),
                          (np.array([1, 2, 3]), {'[1, 2, 3]': [1, 2, 3]})],
                         ids=lambda lags: f'lags, lags_grid_expected: {lags}')
def test_initialize_lags_grid_when_lags_grid_is_None(lags, lags_grid_expected):
    """
    Test initialize_lags_grid when lags_grid is None.
    """
    forecaster = ForecasterRecursive(regressor=Ridge(random_state=123), lags=lags)
    lags_grid = None
    lags_grid, lags_label = initialize_lags_grid(forecaster, lags_grid)

    assert lags_label == 'values'
    assert lags_grid.keys() == lags_grid_expected.keys()
    for v, v_expected in zip(lags_grid.values(), lags_grid_expected.values()):
        assert v == v_expected


def test_initialize_lags_grid_when_lags_grid_is_a_dict():
    """
    Test initialize_lags_grid when lags_grid is a dict.
    """
    forecaster = ForecasterRecursive(regressor=Ridge(random_state=123), lags=2)
    lags_grid = {'1': 1, '[2, 4]': [2, 4], 'range(3, 5)': range(3, 5), '[3 7]': np.array([3, 7])}
    lags_grid, lags_label = initialize_lags_grid(forecaster, lags_grid)

    lags_grid_expected = {
        '1': 1, 
        '[2, 4]': [2, 4], 
        'range(3, 5)': range(3, 5), 
        '[3 7]': np.array([3, 7])
    }
    
    assert lags_label == 'keys'
    assert lags_grid.keys() == lags_grid_expected.keys()
    for v, v_expected in zip(lags_grid.values(), lags_grid_expected.values()):
        if isinstance(v, np.ndarray):
            assert np.array_equal(v, v_expected)
        elif isinstance(v, range):
            assert list(v) == list(v_expected)
        else:
            assert v == v_expected