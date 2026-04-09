# Unit test _make_lags_hashable
# ==============================================================================
import pytest
import numpy as np
from skforecast.model_selection._utils import _make_lags_hashable


_SENTINEL = object()


@pytest.mark.parametrize(
    'lags, expected',
    [
        (5, 5),
        (1, 1),
        (0, 0),
    ],
    ids=lambda dt: f'lags: {dt}'
)
def test_make_lags_hashable_output_when_lags_is_int(lags, expected):
    """
    Test that int lags are returned as-is.
    """
    result = _make_lags_hashable(lags, sentinel=_SENTINEL)
    assert result == expected
    assert isinstance(result, int)


@pytest.mark.parametrize(
    'lags, expected',
    [
        ([1, 2, 3], (1, 2, 3)),
        ([5], (5,)),
        ([1, 5, 10, 20], (1, 5, 10, 20)),
        (np.array([1, 2, 3]), (1, 2, 3)),
        (np.array([5]), (5,)),
        (np.array([1, 5, 10, 20]), (1, 5, 10, 20)),
        (range(1, 4), (1, 2, 3)),
        (range(1, 2), (1,)),
        (range(1, 6), (1, 2, 3, 4, 5)),
    ],
    ids=lambda dt: f'lags: {dt}'
)
def test_make_lags_hashable_output_when_lags_is_list_ndarray_or_range(lags, expected):
    """
    Test that list, numpy ndarray, and range lags are converted to tuple.
    """
    result = _make_lags_hashable(lags, sentinel=_SENTINEL)
    assert result == expected
    assert isinstance(result, tuple)


def test_make_lags_hashable_output_when_lags_is_None():
    """
    Test that None lags are returned as None.
    """
    result = _make_lags_hashable(None, sentinel=_SENTINEL)
    assert result is None


def test_make_lags_hashable_output_when_lags_is_sentinel():
    """
    Test that sentinel lags are returned as the same sentinel object.
    Also verifies that different sentinels are independent and that sentinel
    is not confused with None.
    """
    result = _make_lags_hashable(_SENTINEL, sentinel=_SENTINEL)
    assert result is _SENTINEL

    sentinel_a = object()
    sentinel_b = object()
    assert _make_lags_hashable(sentinel_a, sentinel=sentinel_a) is sentinel_a
    assert _make_lags_hashable(sentinel_b, sentinel=sentinel_b) is sentinel_b
    assert sentinel_a is not sentinel_b

    result_none = _make_lags_hashable(None, sentinel=_SENTINEL)
    assert result_none is None
    assert result_none is not _SENTINEL


@pytest.mark.parametrize(
    'lags, expected',
    [
        ({'series_1': 3, 'series_2': 5},
         (('series_1', 3), ('series_2', 5))),
        ({'series_1': [1, 2, 3], 'series_2': [1, 5]},
         (('series_1', (1, 2, 3)), ('series_2', (1, 5)))),
        ({'series_1': np.array([1, 2, 3]), 'series_2': np.array([4, 5])},
         (('series_1', (1, 2, 3)), ('series_2', (4, 5)))),
        ({'series_1': range(1, 4), 'series_2': range(1, 6)},
         (('series_1', (1, 2, 3)), ('series_2', (1, 2, 3, 4, 5)))),
        ({'a': 3, 'b': [1, 2], 'c': np.array([4, 5]), 'd': range(1, 4)},
         (('a', 3), ('b', (1, 2)), ('c', (4, 5)), ('d', (1, 2, 3)))),
        ({'z_series': 1, 'a_series': 2, 'm_series': 3},
         (('a_series', 2), ('m_series', 3), ('z_series', 1))),
    ],
    ids=['int_values', 'list_values', 'ndarray_values', 'range_values',
         'mixed_values', 'keys_sorted']
)
def test_make_lags_hashable_output_when_lags_is_dict(lags, expected):
    """
    Test that dict lags are converted to a sorted tuple of (key, value)
    pairs, with sequence values converted to tuples.
    """
    result = _make_lags_hashable(lags, sentinel=_SENTINEL)
    assert result == expected
    assert isinstance(result, tuple)


def test_make_lags_hashable_result_is_usable_as_dict_key():
    """
    Test that the result of _make_lags_hashable can be used as a dictionary
    key for all supported input types.
    """
    test_cases = [
        5,
        [1, 2, 3],
        np.array([1, 2, 3]),
        range(1, 4),
        None,
        _SENTINEL,
        {'series_1': [1, 2], 'series_2': 3},
    ]
    cache = {}
    for i, lags in enumerate(test_cases):
        key = _make_lags_hashable(lags, sentinel=_SENTINEL)
        cache[key] = i

    assert len(cache) == len(test_cases) - 2  # list, ndarray, range all map to (1, 2, 3)
