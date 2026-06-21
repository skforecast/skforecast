# Unit test skops decompose/compose helpers
# ==============================================================================
import copy
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from ...utils import _decompose_index
from ...utils import _compose_index
from ...utils import _decompose_pandas_object
from ...utils import _compose_pandas_object
from ...utils import _skops_decompose_forecaster
from ...utils import _skops_reconstruct_forecaster
from ....recursive import ForecasterRecursive
from ....recursive import ForecasterRecursiveMultiSeries


@pytest.mark.parametrize(
    "index, expected_payload",
    [
        (
            pd.date_range('2020-01-01', periods=4, freq='MS', name='dt'),
            {
                'index_type': 'datetime',
                'index': [
                    '2020-01-01 00:00:00',
                    '2020-02-01 00:00:00',
                    '2020-03-01 00:00:00',
                    '2020-04-01 00:00:00',
                ],
                'freq': 'MS',
                'index_name': 'dt',
            },
        ),
        (
            pd.RangeIndex(2, 12, 2, name='r'),
            {'index_type': 'range', 'range': [2, 12, 2], 'index_name': 'r'},
        ),
        (
            pd.Index([10, 20, 30], name='x'),
            {'index_type': 'other', 'index': [10, 20, 30], 'index_name': 'x'},
        ),
        (
            pd.Index(['a', 'b', 'c']),
            {'index_type': 'other', 'index': ['a', 'b', 'c'], 'index_name': None},
        ),
    ],
    ids=['datetime', 'range', 'other_int', 'other_object']
)
def test_decompose_index_output(index, expected_payload):
    """
    Test that _decompose_index returns the expected plain-dict payload (only
    str/list/int values, no pandas objects) for each index kind.
    """
    payload = _decompose_index(index)

    assert payload == expected_payload


@pytest.mark.parametrize(
    "index",
    [
        pd.date_range('2020-01-01', periods=4, freq='MS', name='dt'),
        pd.DatetimeIndex(
            pd.to_datetime(['2021-01-01', '2021-06-15', '2021-12-31'])
        ),
        pd.RangeIndex(2, 12, 2, name='r'),
        pd.Index([10, 20, 30], name='x'),
        pd.Index(['a', 'b', 'c']),
    ],
    ids=lambda idx: f'index: {type(idx).__name__}, freq: {getattr(idx, "freqstr", None)}'
)
def test_decompose_compose_index_round_trip(index):
    """
    Test that _compose_index rebuilds an index identical to the original
    produced by _decompose_index, preserving type, name, and frequency.
    """
    rebuilt = _compose_index(_decompose_index(index))

    pd.testing.assert_index_equal(rebuilt, index)
    if isinstance(index, pd.DatetimeIndex):
        assert rebuilt.freq == index.freq


def test_decompose_compose_pandas_object_round_trip_dataframe_datetime():
    """
    Test that a DataFrame with a DatetimeIndex round-trips through
    _decompose_pandas_object and _compose_pandas_object, with `data` stored as
    a numpy array and columns preserved.
    """
    df = pd.DataFrame(
        {'col_1': [1.0, 2.0, 3.0], 'col_2': [4.0, 5.0, 6.0]},
        index=pd.date_range('2020-01-01', periods=3, freq='D', name='dt')
    )

    payload = _decompose_pandas_object(df)
    rebuilt = _compose_pandas_object(payload)

    assert payload['object_type'] == 'DataFrame'
    assert isinstance(payload['data'], np.ndarray)
    pd.testing.assert_frame_equal(rebuilt, df)


def test_decompose_compose_pandas_object_round_trip_dataframe_range():
    """
    Test that a DataFrame with a RangeIndex round-trips through
    _decompose_pandas_object and _compose_pandas_object.
    """
    df = pd.DataFrame(
        {'col_1': [1.0, 2.0, 3.0]},
        index=pd.RangeIndex(0, 3, 1)
    )

    payload = _decompose_pandas_object(df)
    rebuilt = _compose_pandas_object(payload)

    assert payload['object_type'] == 'DataFrame'
    assert isinstance(payload['data'], np.ndarray)
    pd.testing.assert_frame_equal(rebuilt, df)


def test_decompose_compose_pandas_object_round_trip_series():
    """
    Test that a named Series with a DatetimeIndex round-trips through
    _decompose_pandas_object and _compose_pandas_object.
    """
    series = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range('2020-01-01', periods=3, freq='D'),
        name='y'
    )

    payload = _decompose_pandas_object(series)
    rebuilt = _compose_pandas_object(payload)

    assert payload['object_type'] == 'Series'
    assert isinstance(payload['data'], np.ndarray)
    pd.testing.assert_series_equal(rebuilt, series)


def test_decompose_compose_pandas_object_round_trip_index():
    """
    Test that a standalone DatetimeIndex round-trips through
    _decompose_pandas_object and _compose_pandas_object (no `data` key, only the
    index payload and the `object_type` marker).
    """
    index = pd.date_range('2020-01-01', periods=3, freq='D', name='dt')

    payload = _decompose_pandas_object(index)
    rebuilt = _compose_pandas_object(payload)

    assert payload['object_type'] == 'Index'
    assert 'data' not in payload
    pd.testing.assert_index_equal(rebuilt, index)


def test_skops_decompose_reconstruct_forecaster_single_series():
    """
    Test that _skops_decompose_forecaster replaces `last_window_` and
    `training_range_` of a single-series forecaster with plain dicts carrying the
    `object_type` marker, and that _skops_reconstruct_forecaster restores them to
    the original pandas objects.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    rng = np.random.default_rng(12345)
    idx = pd.date_range('2020-01-01', periods=50, freq='D')
    y = pd.Series(rng.normal(size=50), index=idx)
    forecaster.fit(y=y)

    last_window_original = forecaster.last_window_.copy()
    training_range_original = forecaster.training_range_.copy()

    _skops_decompose_forecaster(forecaster)

    assert isinstance(forecaster.last_window_, dict)
    assert forecaster.last_window_['object_type'] == 'DataFrame'
    assert isinstance(forecaster.training_range_, dict)
    assert forecaster.training_range_['object_type'] == 'Index'

    _skops_reconstruct_forecaster(forecaster)

    pd.testing.assert_frame_equal(forecaster.last_window_, last_window_original)
    pd.testing.assert_index_equal(forecaster.training_range_, training_range_original)


def test_skops_decompose_reconstruct_forecaster_multiseries():
    """
    Test that _skops_decompose_forecaster handles the multi-series `dict`
    containers (no top-level `object_type` key, each value decomposed) and that
    _skops_reconstruct_forecaster rebuilds the per-level pandas objects.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(), lags=3, transformer_series=StandardScaler()
    )
    rng = np.random.default_rng(12345)
    series = pd.DataFrame(
        {'serie_1': rng.normal(size=50), 'serie_2': rng.normal(size=50)}
    )
    forecaster.fit(series=series)

    last_window_original = copy.deepcopy(forecaster.last_window_)
    training_range_original = copy.deepcopy(forecaster.training_range_)

    _skops_decompose_forecaster(forecaster)

    assert isinstance(forecaster.last_window_, dict)
    assert 'object_type' not in forecaster.last_window_
    assert all(v['object_type'] == 'Series' for v in forecaster.last_window_.values())
    assert isinstance(forecaster.training_range_, dict)
    assert 'object_type' not in forecaster.training_range_
    assert all(v['object_type'] == 'Index' for v in forecaster.training_range_.values())

    _skops_reconstruct_forecaster(forecaster)

    assert forecaster.last_window_.keys() == last_window_original.keys()
    assert forecaster.training_range_.keys() == training_range_original.keys()
    for k in last_window_original.keys():
        pd.testing.assert_series_equal(
            forecaster.last_window_[k], last_window_original[k]
        )
        pd.testing.assert_index_equal(
            forecaster.training_range_[k], training_range_original[k]
        )
