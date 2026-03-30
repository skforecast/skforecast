# Unit test fetch_dataset
# ==============================================================================
import re
import pytest
import pandas as pd
from unittest.mock import patch
from skforecast.datasets import datasets, fetch_dataset


def test_fetch_dataset_csv_raw_false():
    """
    `fetch_dataset` with a CSV dataset and `raw=False` returns a DataFrame
    with a DatetimeIndex and the expected frequency.
    """
    df = fetch_dataset('h2o', version='latest', raw=False, verbose=False)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (204, 1)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.freq == 'MS'
    assert df.index.is_monotonic_increasing
    assert df.index[0].strftime('%Y-%m-%d') == '1991-07-01'
    assert df.index[-1].strftime('%Y-%m-%d') == '2008-06-01'


def test_fetch_dataset_csv_raw_true():
    """
    `fetch_dataset` with `raw=True` returns a DataFrame without index
    manipulation: the date column is kept as a regular column and no
    frequency is assigned.
    """
    df = fetch_dataset('h2o', version='latest', raw=True, verbose=False)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (204, 2)
    assert not isinstance(df.index, pd.DatetimeIndex)
    assert 'fecha' in df.columns
    assert 'x' in df.columns


def test_fetch_dataset_csv_multiple_series():
    """
    `fetch_dataset` returns the expected shape and frequency for a multi-column
    CSV dataset.
    """
    df = fetch_dataset('items_sales', version='latest', raw=False, verbose=False)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1097, 3)
    assert df.index.freq == 'D'
    assert df.index[0] == pd.Timestamp('2012-01-01')
    assert df.index[-1] == pd.Timestamp('2015-01-01')


def test_fetch_dataset_kwargs_read():
    """
    `fetch_dataset` passes `kwargs_read` through to `pd.read_csv` (for CSV
    datasets) and to `pd.read_parquet` (for parquet datasets).
    """
    # CSV: limit rows with nrows
    df = fetch_dataset(
        'h2o',
        version='latest',
        raw=True,
        kwargs_read={'nrows': 10},
        verbose=False
    )
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (10, 2)

    # Parquet: pass columns kwarg
    mock_df = pd.DataFrame({
        'timestamp': ['2019-01-01 00:00:00', '2019-01-02 00:00:00'],
        'value': [1.0, 2.0]
    })
    with patch('pandas.read_parquet', return_value=mock_df) as mock_read:
        fetch_dataset(
            'm4_daily',
            version='latest',
            raw=True,
            kwargs_read={'columns': ['timestamp', 'value']},
            verbose=False
        )
    call_kwargs = mock_read.call_args
    assert call_kwargs[1].get('columns') == ['timestamp', 'value']


def test_fetch_dataset_parquet_raw_false():
    """
    `fetch_dataset` correctly processes a parquet dataset: calls
    `pd.read_parquet` with the expected URL and applies index/frequency
    preprocessing when `raw=False`.
    """
    mock_df = pd.DataFrame({
        'timestamp': ['2019-01-01 00:00:00', '2019-01-02 00:00:00', '2019-01-03 00:00:00'],
        'value': [1.0, 2.0, 3.0],
        'series_id': ['S1', 'S1', 'S1']
    })

    with patch('pandas.read_parquet', return_value=mock_df) as mock_read:
        df = fetch_dataset('m4_daily', version='latest', raw=False, verbose=False)

    expected_url = (
        'https://raw.githubusercontent.com/skforecast/'
        'skforecast-datasets/main/data/m4_daily.parquet'
    )
    mock_read.assert_called_once_with(expected_url)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.freq == 'D'


def test_fetch_dataset_parquet_raw_true():
    """
    `fetch_dataset` with `raw=True` returns the parquet DataFrame as-is,
    without index processing.
    """
    mock_df = pd.DataFrame({
        'timestamp': ['2019-01-01 00:00:00', '2019-01-02 00:00:00'],
        'value': [1.0, 2.0]
    })

    with patch('pandas.read_parquet', return_value=mock_df):
        df = fetch_dataset('m4_daily', version='latest', raw=True, verbose=False)

    assert isinstance(df, pd.DataFrame)
    assert 'timestamp' in df.columns


def test_fetch_dataset_verbose(capsys):
    """
    `fetch_dataset` prints dataset info when `verbose=True` and produces
    no output when `verbose=False`.
    """
    fetch_dataset('h2o', version='latest', raw=False, verbose=True)
    captured = capsys.readouterr()
    assert 'h2o' in captured.out
    assert '204' in captured.out

    fetch_dataset('h2o', version='latest', raw=False, verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ''


def test_fetch_dataset_invalid_name_raises():
    """
    `fetch_dataset` raises `ValueError` with an informative message when the
    requested dataset name does not exist.
    """
    err_msg = re.escape(
        f"Dataset 'non_existent_dataset' not found. "
        f"Available datasets are: {sorted(datasets)}"
    )
    with pytest.raises(ValueError, match=err_msg):
        fetch_dataset(
            'non_existent_dataset', version='latest', raw=False, verbose=False
        )


def test_fetch_dataset_parquet_already_indexed():
    """
    `fetch_dataset` skips `set_index` when the parquet DataFrame is already
    stored with the date column as the index (i.e. `df.index.name == index_col`).
    """
    mock_df = pd.DataFrame(
        {'value': [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(
            ['2019-01-01', '2019-01-02', '2019-01-03'], name='timestamp'
        )
    )

    with patch('pandas.read_parquet', return_value=mock_df):
        df = fetch_dataset('m4_daily', version='latest', raw=False, verbose=False)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == 'timestamp'
    assert df.index.freq == 'D'


def test_fetch_dataset_duplicate_index_skips_asfreq():
    """
    `fetch_dataset` does not call `asfreq` when the index has duplicate dates,
    which happens with long-format multi-series datasets. The result has a
    DatetimeIndex but no frequency set.
    """
    mock_df = pd.DataFrame({
        'timestamp': [
            '2019-01-01 00:00:00', '2019-01-02 00:00:00', '2019-01-03 00:00:00',
            '2019-01-01 00:00:00', '2019-01-02 00:00:00', '2019-01-03 00:00:00',
        ],
        'series_id': ['S1', 'S1', 'S1', 'S2', 'S2', 'S2'],
        'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })

    with patch('pandas.read_parquet', return_value=mock_df):
        df = fetch_dataset('m4_daily', version='latest', raw=False, verbose=False)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert not df.index.is_unique
    assert df.index.freq is None


def test_fetch_dataset_preprocessing_failure_warns():
    """
    `fetch_dataset` emits a `UserWarning` with an informative message when
    preprocessing fails (e.g., `index_col` is not present in the DataFrame).
    The raw DataFrame is returned unchanged rather than raising an exception.
    """
    mock_df = pd.DataFrame({
        'wrong_col': ['2019-01-01 00:00:00', '2019-01-02 00:00:00'],
        'value': [1.0, 2.0],
    })

    warn_msg = re.escape("Could not preprocess dataset 'm4_daily':")
    with patch('pandas.read_parquet', return_value=mock_df):
        with pytest.warns(UserWarning, match=warn_msg):
            df = fetch_dataset('m4_daily', version='latest', raw=False, verbose=False)

    assert isinstance(df, pd.DataFrame)
    assert 'wrong_col' in df.columns


def test_fetch_dataset_invalid_version_raises():
    """
    `fetch_dataset` raises `ValueError` when the requested version does not
    exist, for both CSV and parquet datasets.
    """
    # CSV dataset
    bad_url = (
        'https://raw.githubusercontent.com/skforecast/'
        'skforecast-datasets/non_existent_version/data/h2o.csv'
    )
    err_msg = re.escape(
        f"Error reading dataset 'h2o' from {bad_url}: HTTP Error 404: Not Found."
    )
    with pytest.raises(ValueError, match=err_msg):
        fetch_dataset(
            'h2o', version='non_existent_version', raw=False, verbose=False
        )

    # Parquet dataset
    with patch('pandas.read_parquet', side_effect=Exception("404 Client Error")):
        with pytest.raises(ValueError, match="Error reading dataset 'm4_daily'"):
            fetch_dataset(
                'm4_daily', version='non_existent_version', raw=False, verbose=False
            )
