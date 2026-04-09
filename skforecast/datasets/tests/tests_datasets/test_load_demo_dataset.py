# Unit test load_demo_dataset
# ==============================================================================
import pandas as pd
from skforecast.datasets import load_demo_dataset


def test_load_demo_dataset_returns_correct_series():
    """
    `load_demo_dataset` returns a pandas Series with a monthly DatetimeIndex,
    the expected shape, and data sorted in ascending order.
    """
    df = load_demo_dataset(verbose=False)

    assert isinstance(df, pd.Series)
    assert df.index.freq == 'MS'
    assert df.index.is_monotonic_increasing
    assert df.index[0] == pd.Timestamp('1991-07-01')
    assert df.index[-1] == pd.Timestamp('2008-06-01')
    assert df.shape == (204,)


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
