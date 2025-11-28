# Unit test fetch_dataset
# ==============================================================================
import re
import pytest
import pandas as pd
from skforecast.datasets import datasets, show_datasets_info, fetch_dataset
from skforecast.datasets.datasets import _print_dataset_info


def test_fetch_dataset():
    """
    Test function `fetch_dataset`.
    """

    show_datasets_info()
    show_datasets_info(datasets_names=['not_existing_dataset'])
    
    # Test fetching the 'h2o' dataset
    df = fetch_dataset('h2o', version='latest', raw=False, verbose=False)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (204, 1)
    assert df.index.freq == 'MS'
    assert df.index[0].strftime('%Y-%m-%d') == '1991-07-01'
    assert df.index[-1].strftime('%Y-%m-%d') == '2008-06-01'

    # Test fetching the 'items_sales' dataset
    df = fetch_dataset('items_sales', version='latest', raw=False, verbose=True)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1097, 3)
    assert df.index.freq == 'D'
    assert df.index[0] == pd.Timestamp('2012-01-01')
    assert df.index[-1] == pd.Timestamp('2015-01-01')

    # Test fetching a non-existent dataset
    err_msg = re.escape(
        f"Dataset 'non_existent_dataset' not found. "
        f"Available datasets are: {sorted(datasets.keys())}"
    )
    with pytest.raises(ValueError, match = err_msg):
        _print_dataset_info(dataset_name='non_existent_dataset')

    # Test fetching a non-existent dataset
    err_msg = re.escape(
        f"Dataset 'non_existent_dataset' not found. "
        f"Available datasets are: {sorted(datasets.keys())}"
    )
    with pytest.raises(ValueError, match = err_msg):
        fetch_dataset(
            'non_existent_dataset', version='latest', raw=False, verbose=False
        )

    # Test fetching a dataset with a non-existent version
    bad_url = (
        'https://raw.githubusercontent.com/skforecast/'
        'skforecast-datasets/non_existent_version/data/h2o.csv'
    )
    
    err_msg = re.escape(
        f"Error reading dataset 'h2o' from {bad_url}: HTTP Error 404: Not Found."
    )
    with pytest.raises(ValueError, match = err_msg):
        fetch_dataset(
            'h2o', version='non_existent_version', raw=False, verbose=False
        )
