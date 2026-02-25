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


def test_load_demo_dataset_verbose_true_prints_output(capsys):
    """
    `load_demo_dataset` with `verbose=True` prints a panel containing the
    dataset name and shape information to stdout.
    """
    load_demo_dataset(verbose=True)

    captured = capsys.readouterr()
    assert 'h2o' in captured.out
    assert '204' in captured.out


def test_load_demo_dataset_verbose_false_no_output(capsys):
    """
    `load_demo_dataset` with `verbose=False` produces no stdout output.
    """
    load_demo_dataset(verbose=False)

    captured = capsys.readouterr()
    assert captured.out == ''
