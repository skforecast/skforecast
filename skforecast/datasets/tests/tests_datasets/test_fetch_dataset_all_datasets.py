# Unit test fetch_dataset on every dataset hosted on GitHub
# ==============================================================================
# The regression test downloads every dataset hosted on GitHub (about 175 MB),
# so it only runs with SKFORECAST_TEST_DATASETS=1. Run it before each release
# and after changing the skforecast-datasets repository:
#
#   SKFORECAST_TEST_DATASETS=1 pytest \
#       skforecast/datasets/tests/tests_datasets/test_fetch_dataset_all_datasets.py -q
#
# In PowerShell: $env:SKFORECAST_TEST_DATASETS = '1'; pytest ...
import os
import warnings
from urllib.parse import urlparse
import pytest
import pandas as pd
from skforecast.datasets import datasets, fetch_dataset
from skforecast.exceptions import MissingValuesWarning

EXPECTED_DATASETS = {
    # name: (rows, cols, freq)
    'h2o': (204, 1, 'MS'),
    'h2o_exog': (195, 3, 'MS'),
    'fuel_consumption': (644, 5, 'MS'),
    'items_sales': (1097, 3, 'D'),
    'air_quality_valencia': (43824, 10, 'h'),
    'air_quality_valencia_no_missing': (43824, 10, 'h'),
    'website_visits': (421, 1, 'D'),
    'bike_sharing': (17544, 11, 'h'),
    'bike_sharing_extended_features': (17352, 90, 'h'),
    'australia_tourism': (24320, 4, None),
    'uk_daily_flights': (1118, 1, 'D'),
    'wikipedia_visits': (2964, 1, 'D'),
    'vic_electricity': (52608, 4, '30min'),
    'vic_electricity_classification': (8736, 5, 'h'),
    'store_sales': (913000, 3, None),
    'bicimad': (3022, 1, 'D'),
    'm4_daily': (10023836, 3, None),
    'm4_hourly': (373372, 3, None),
    'bdg2_daily_sample': (274, 2, 'D'),
    'bdg2_hourly_sample': (6553, 2, 'h'),
    'ett_m1': (69680, 7, '15min'),
    'ett_m2': (69680, 7, '15min'),
    'ett_m2_extended': (69680, 16, '15min'),
    'expenditures_australia': (505, 1, 'MS'),
    'public_transport_madrid': (715, 5, 'D'),
    'turbine_emission': (36733, 11, 'h'),
}

# Datasets whose dates are not complete: setting the frequency introduces rows
# with NaN and emits a `MissingValuesWarning`
DATASETS_WITH_MISSING_DATES = {'wikipedia_visits'}

requires_datasets_download = pytest.mark.skipif(
    os.environ.get('SKFORECAST_TEST_DATASETS') != '1',
    reason='Downloads every dataset hosted on GitHub (about 175 MB).'
)


@pytest.mark.slow
@requires_datasets_download
@pytest.mark.parametrize(
    'name', sorted(EXPECTED_DATASETS), ids=lambda n: f'dataset: {n}'
)
def test_fetch_dataset_all_github_datasets_are_preprocessed(name):
    """
    Every dataset hosted on GitHub is returned with a sorted DatetimeIndex named
    as its `index_col`, the expected shape and the declared frequency (None for
    long-format datasets with repeated dates). No warning is emitted, except
    the `MissingValuesWarning` of the datasets with missing dates.
    """
    rows, cols, freq = EXPECTED_DATASETS[name]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        df = fetch_dataset(name, verbose=False)

    expected_warnings = (
        [MissingValuesWarning] if name in DATASETS_WITH_MISSING_DATES else []
    )
    messages = [f'{w.category.__name__}: {w.message}' for w in record]
    assert [w.category for w in record] == expected_warnings, messages
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == datasets[name]['index_col']
    assert df.index.is_monotonic_increasing
    assert df.shape == (rows, cols)
    assert df.index.freqstr == freq


def test_fetch_dataset_all_github_datasets_are_covered():
    """
    `EXPECTED_DATASETS` lists every dataset of the registry hosted on GitHub,
    so a new dataset cannot be added without its expected shape.
    """
    github = {
        name for name, info in datasets.items()
        if urlparse(info['url']).netloc == 'raw.githubusercontent.com'
    }
    assert set(EXPECTED_DATASETS) == github
    assert DATASETS_WITH_MISSING_DATES <= github
