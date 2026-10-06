# Unit test skops decompose/compose helpers
# ==============================================================================
import re
import copy
import datetime
import zoneinfo
import pytest
import dateutil.tz
import pytz
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
    "tz",
    [
        dateutil.tz.tzoffset(None, 3600),
        pytz.FixedOffset(60),
        datetime.timezone(datetime.timedelta(hours=1), 'CET'),
    ],
    ids=lambda tz: f'tz: {tz!r}'
)
def test_decompose_index_ValueError_when_time_zone_cannot_be_rebuilt(tz):
    """
    Test that _decompose_index raises a ValueError when the time zone of a
    DatetimeIndex cannot be rebuilt from its name, instead of storing a payload
    that cannot be loaded or that is loaded with another time zone (the name
    'CET' of a fixed offset would be rebuilt as the CET zone, which has
    daylight saving time).
    """
    index = pd.date_range('2024-01-01', periods=3, freq='h', tz=tz)

    err_msg = re.escape(
        f"The time zone {index.tz!r} of the index cannot be saved with "
        f"backend='skops' because it cannot be rebuilt from its name "
        f"{str(index.tz)!r}. Convert the index to a named time zone (e.g. "
        f"'Europe/Madrid') or use another backend."
    )
    with pytest.raises(ValueError, match=err_msg):
        _decompose_index(index)


@pytest.mark.parametrize(
    "index, expected_payload",
    [
        (
            pd.date_range('2020-01-01', periods=4, freq='MS', name='dt'),
            {
                'index_type_': 'datetime',
                'index': np.array([
                    1577836800000000000,
                    1580515200000000000,
                    1583020800000000000,
                    1585699200000000000,
                ]),
                'unit': 'ns',
                'tz': None,
                'tz_zoneinfo': False,
                'freq': pd.offsets.MonthBegin(),
                'index_name': 'dt',
            },
        ),
        (
            pd.date_range('2024-03-31', periods=4, freq='h', tz='Europe/Madrid'),
            {
                'index_type_': 'datetime',
                'index': np.array([
                    1711839600000000000,
                    1711843200000000000,
                    1711846800000000000,
                    1711850400000000000,
                ]),
                'unit': 'ns',
                'tz': 'Europe/Madrid',
                'tz_zoneinfo': False,
                'freq': pd.offsets.Hour(),
                'index_name': None,
            },
        ),
        (
            pd.RangeIndex(2, 12, 2, name='r'),
            {'index_type_': 'range', 'range': [2, 12, 2], 'index_name': 'r'},
        ),
        (
            pd.Index([10, 20, 30], name='x'),
            {'index_type_': 'other', 'index': [10, 20, 30], 'index_name': 'x'},
        ),
        (
            pd.Index(['a', 'b', 'c']),
            {'index_type_': 'other', 'index': ['a', 'b', 'c'], 'index_name': None},
        ),
    ],
    ids=['datetime', 'datetime_tz', 'range', 'other_int', 'other_object']
)
def test_decompose_index_output(index, expected_payload):
    """
    Test that _decompose_index returns the expected plain-dict payload for each
    index kind. A DatetimeIndex stores its values as integers since the epoch
    and its time zone by name.
    """
    payload = _decompose_index(index)

    assert payload.keys() == expected_payload.keys()
    for key, expected_value in expected_payload.items():
        if isinstance(expected_value, np.ndarray):
            np.testing.assert_array_equal(payload[key], expected_value)
        else:
            assert payload[key] == expected_value


@pytest.mark.parametrize(
    "index",
    [
        pd.date_range('2020-01-01', periods=4, freq='MS', name='dt'),
        pd.DatetimeIndex(
            pd.to_datetime(['2021-01-01', '2021-06-15', '2021-12-31'])
        ),
        pd.date_range('2024-03-31', periods=4, freq='h', tz='Europe/Madrid'),
        pd.date_range(
            '2024-03-09', periods=3, freq='D', tz=zoneinfo.ZoneInfo('America/New_York')
        ),
        pd.date_range('2024-01-01', periods=3, freq='h', tz='UTC'),
        pd.date_range(
            '2024-01-01',
            periods=3,
            freq='h',
            tz=datetime.timezone(datetime.timedelta(hours=1)),
        ),
        pd.date_range('2024-01-01', periods=3, freq='500ms'),
        pd.date_range('2024-01-01', periods=3, freq='D', unit='s'),
        pd.date_range(
            '2024-12-23',
            periods=4,
            freq=pd.offsets.CustomBusinessDay(holidays=['2024-12-25']),
        ),
        pd.RangeIndex(2, 12, 2, name='r'),
        pd.Index([10, 20, 30], name='x'),
        pd.Index(['a', 'b', 'c']),
    ],
    ids=lambda idx: (
        f'index: {type(idx).__name__}, freq: {getattr(idx, "freqstr", None)}, '
        f'tz: {getattr(idx, "tz", None)}, unit: {getattr(idx, "unit", None)}'
    )
)
def test_decompose_compose_index_round_trip(index):
    """
    Test that _compose_index rebuilds an index identical to the original
    produced by _decompose_index, preserving type, name, unit, time zone (and
    its type) and frequency (including the holidays of a CustomBusinessDay).
    """
    rebuilt = _compose_index(_decompose_index(index))

    pd.testing.assert_index_equal(rebuilt, index)
    if isinstance(index, pd.DatetimeIndex):
        assert rebuilt.freq == index.freq


@pytest.mark.parametrize(
    "payload, expected_index",
    [
        (
            {
                'index_type_': 'datetime',
                'index': ['2020-01-01 00:00:00', '2020-01-02 00:00:00'],
                'freq': 'D',
                'index_name': 'dt',
            },
            pd.date_range('2020-01-01', periods=2, freq='D', name='dt'),
        ),
        (
            {
                'index_type_': 'datetime',
                'index': ['2024-01-01 00:00:00', '2024-01-01 00:00:00.500000'],
                'freq': '500ms',
                'index_name': None,
            },
            pd.date_range('2024-01-01', periods=2, freq='500ms'),
        ),
        (
            {
                'index_type_': 'datetime',
                'index': ['2024-01-01 00:00:00+01:00', '2024-01-01 01:00:00+01:00'],
                'freq': 'h',
                'index_name': None,
            },
            pd.date_range(
                '2024-01-01',
                periods=2,
                freq='h',
                tz=datetime.timezone(datetime.timedelta(hours=1)),
            ),
        ),
        (
            {
                'index_type_': 'datetime',
                'index': ['2024-03-31 01:00:00+01:00', '2024-03-31 03:00:00+02:00'],
                'freq': 'h',
                'index_name': None,
            },
            pd.date_range(
                '2024-03-31 01:00:00',
                periods=2,
                freq='h',
                tz=datetime.timezone(datetime.timedelta(hours=1)),
            ),
        ),
    ],
    ids=['naive', 'fractional_seconds', 'utc_offset', 'mixed_utc_offsets']
)
def test_compose_index_legacy_datetime_payload(payload, expected_index):
    """
    Test that _compose_index rebuilds the DatetimeIndex payloads of files saved
    with skforecast < 0.26, which store the timestamps as strings. Mixed UTC
    offsets (a daylight saving time change) are rebuilt with the offset of the
    first timestamp.
    """
    index = _compose_index(payload)

    pd.testing.assert_index_equal(index, expected_index)
    assert index.freq == expected_index.freq


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

    assert payload['object_type_'] == 'DataFrame'
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

    assert payload['object_type_'] == 'DataFrame'
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

    assert payload['object_type_'] == 'Series'
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

    assert payload['object_type_'] == 'Index'
    assert 'data' not in payload
    pd.testing.assert_index_equal(rebuilt, index)


def test_skops_decompose_reconstruct_forecaster_single_series():
    """
    Test that _skops_decompose_forecaster returns a copy of a single-series
    forecaster whose `last_window_` and `training_range_` are plain dicts
    carrying the `object_type_` marker, without modifying the forecaster, and
    that _skops_reconstruct_forecaster restores them to the original pandas
    objects.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    rng = np.random.default_rng(12345)
    idx = pd.date_range('2020-01-01', periods=50, freq='D')
    y = pd.Series(rng.normal(size=50), index=idx)
    forecaster.fit(y=y)

    last_window_original = forecaster.last_window_
    training_range_original = forecaster.training_range_

    forecaster_decomposed = _skops_decompose_forecaster(forecaster)

    assert forecaster_decomposed is not forecaster
    assert forecaster.last_window_ is last_window_original
    assert forecaster.training_range_ is training_range_original
    assert isinstance(forecaster_decomposed.last_window_, dict)
    assert forecaster_decomposed.last_window_['object_type_'] == 'DataFrame'
    assert isinstance(forecaster_decomposed.training_range_, dict)
    assert forecaster_decomposed.training_range_['object_type_'] == 'Index'

    _skops_reconstruct_forecaster(forecaster_decomposed)

    pd.testing.assert_frame_equal(
        forecaster_decomposed.last_window_, last_window_original
    )
    pd.testing.assert_index_equal(
        forecaster_decomposed.training_range_, training_range_original
    )


def test_skops_decompose_reconstruct_forecaster_multiseries():
    """
    Test that _skops_decompose_forecaster handles the multi-series `dict`
    containers (no top-level `object_type_` key, each value decomposed) without
    modifying the forecaster, and that _skops_reconstruct_forecaster rebuilds
    the per-level pandas objects.
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

    forecaster_decomposed = _skops_decompose_forecaster(forecaster)

    last_window_decomposed = forecaster_decomposed.last_window_
    training_range_decomposed = forecaster_decomposed.training_range_
    assert 'object_type_' not in last_window_decomposed
    assert all(v['object_type_'] == 'Series' for v in last_window_decomposed.values())
    assert 'object_type_' not in training_range_decomposed
    assert all(
        v['object_type_'] == 'Index' for v in training_range_decomposed.values()
    )

    _skops_reconstruct_forecaster(forecaster_decomposed)

    for forecaster_to_check in (forecaster, forecaster_decomposed):
        last_window = forecaster_to_check.last_window_
        training_range = forecaster_to_check.training_range_
        assert last_window.keys() == last_window_original.keys()
        assert training_range.keys() == training_range_original.keys()
        for k in last_window_original.keys():
            pd.testing.assert_series_equal(last_window[k], last_window_original[k])
            pd.testing.assert_index_equal(
                training_range[k], training_range_original[k]
            )
