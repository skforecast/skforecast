# Unit test load_demo_dataset
# ==============================================================================
import pandas as pd
from skforecast.datasets import load_demo_dataset


def test_load_demo_dataset():
    """
    Test load_demo_dataset function.
    """
    df = load_demo_dataset()
    
    assert isinstance(df, pd.Series)
    assert df.index.freq == 'MS'
    assert df.index.is_monotonic_increasing
    assert df.index[0] == pd.Timestamp('1991-07-01')
    assert df.index[-1] == pd.Timestamp('2008-06-01')
    assert df.shape == (204,)