# Unit test datasets registry
# ==============================================================================
import warnings
from urllib.parse import urlparse
import pytest
import pandas as pd
from pandas.tseries.frequencies import to_offset
from skforecast.datasets import datasets

REQUIRED_KEYS = {
    'url', 'sep', 'index_col', 'date_format', 'freq', 'file_type',
    'description', 'source'
}


@pytest.mark.parametrize('name', sorted(datasets), ids=lambda n: f'dataset: {n}')
def test_datasets_registry_entry_is_consistent(name):
    """
    Every entry of the `datasets` registry has the required keys, a supported
    file type, a separator consistent with the file type, a frequency alias
    accepted by pandas without warnings and a URL with the `{version}`
    placeholder only when hosted on GitHub.
    """
    info = datasets[name]
    assert set(info) == REQUIRED_KEYS
    assert info['file_type'] in ('csv', 'parquet')
    if info['file_type'] == 'csv':
        assert isinstance(info['sep'], str)
    else:
        assert info['sep'] is None
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        to_offset(info['freq'])
    is_github = urlparse(info['url']).netloc == 'raw.githubusercontent.com'
    assert ('{version}' in info['url']) == is_github
    assert info['description'].strip() and info['source'].strip()


@pytest.mark.parametrize('name', sorted(datasets), ids=lambda n: f'dataset: {n}')
def test_datasets_registry_texts_have_no_typos(name):
    """
    The `description` and `source` of every entry of the `datasets` registry
    have no leading or trailing whitespace, no double spaces, no space before
    a line break and no repeated article ('the The').
    """
    for key in ('description', 'source'):
        text = datasets[name][key]
        assert text == text.strip(), key
        assert '  ' not in text, key
        assert ' \n' not in text, key
        assert 'the The' not in text, key


def test_datasets_registry_australia_tourism_quarter_start_freq():
    """
    The frequency of `australia_tourism` is quarter start, which matches the
    dates of the file (1998-01-01, 1998-04-01, ...).
    """
    freq = datasets['australia_tourism']['freq']
    results = pd.date_range('1998-01-01', periods=3, freq=freq)
    expected = pd.to_datetime(['1998-01-01', '1998-04-01', '1998-07-01'])
    assert list(results) == list(expected)
