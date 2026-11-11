# Unit test load_demo_dataset
# ==============================================================================
import re
import warnings
import pytest
import pandas as pd
from skforecast.datasets import load_demo_dataset


def test_load_demo_dataset_returns_correct_series():
    """
    `load_demo_dataset` returns a pandas Series named `y` with a monthly
    DatetimeIndex named `datetime`, the expected shape and values, and data
    sorted in ascending order.
    """
    df = load_demo_dataset(verbose=False)

    assert isinstance(df, pd.Series)
    assert df.name == 'y'
    assert df.index.name == 'datetime'
    assert df.dtype == float
    assert df.index.freq == 'MS'
    assert df.index.is_monotonic_increasing
    assert df.index[0] == pd.Timestamp('1991-07-01')
    assert df.index[-1] == pd.Timestamp('2008-06-01')
    assert df.shape == (204,)
    assert df.iloc[:5].tolist() == [0.429795, 0.400906, 0.432159, 0.492543, 0.502369]


def test_load_demo_dataset_verbose(capsys):
    """
    `load_demo_dataset` prints dataset info when `verbose=True` and produces
    no output when `verbose=False`.
    """
    load_demo_dataset(verbose=True)
    captured = capsys.readouterr()
    assert 'h2o' in captured.out
    assert '204' in captured.out

    load_demo_dataset(verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ''


def test_load_demo_dataset_version_FutureWarning_once():
    """
    `load_demo_dataset` emits a single `FutureWarning`, pointing to the caller,
    when `version` is passed, and none when it is not.
    """
    warn_msg = re.escape(
        "`version` is deprecated since version 0.27.0 and has no effect: "
        "datasets are always fetched from the main branch of "
        "skforecast-datasets. It will be removed in version 0.29.0."
    )
    with pytest.warns(FutureWarning, match=warn_msg) as record:
        y = load_demo_dataset(version='latest', verbose=False)

    assert len(record) == 1
    assert record[0].filename == __file__
    assert y.shape == (204,)

    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        load_demo_dataset(verbose=False)
