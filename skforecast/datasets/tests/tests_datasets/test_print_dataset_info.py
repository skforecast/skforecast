# Unit test _print_dataset_info
# ==============================================================================
import re
import pytest
from skforecast.datasets import datasets
from skforecast.datasets.datasets import _print_dataset_info


def test_print_dataset_info_shape(capsys):
    """
    `_print_dataset_info` prints a panel with description, source, and URL.
    When `shape` is None, no shape line appears. When a 2-tuple is given,
    rows and columns are shown. When a 1-tuple is given, only rows are shown.
    """
    # No shape
    _print_dataset_info('h2o')
    captured = capsys.readouterr()
    assert 'h2o' in captured.out
    assert 'Description' in captured.out
    assert 'Source' in captured.out
    assert 'URL' in captured.out
    assert 'Shape' not in captured.out

    # 2D shape
    _print_dataset_info('h2o', shape=(204, 1))
    captured = capsys.readouterr()
    assert '204' in captured.out
    assert 'rows' in captured.out
    assert 'columns' in captured.out

    # 1D shape
    _print_dataset_info('h2o', shape=(204,))
    captured = capsys.readouterr()
    assert '204' in captured.out
    assert 'rows' in captured.out
    assert 'columns' not in captured.out


def test_print_dataset_info_version_substituted_in_url(capsys):
    """
    `_print_dataset_info` substitutes the `version` argument into the URL
    `{version}` placeholder when printing.
    """
    _print_dataset_info('h2o', version='latest')

    captured = capsys.readouterr()
    # 'latest' is normalised to 'main' inside the function
    assert 'main' in captured.out
    assert '{version}' not in captured.out


def test_print_dataset_info_invalid_name_raises():
    """
    `_print_dataset_info` raises `ValueError` with an informative message when
    the dataset name is not in the registry.
    """
    err_msg = re.escape(
        f"Dataset 'non_existent_dataset' not found. "
        f"Available datasets are: {sorted(datasets)}"
    )
    with pytest.raises(ValueError, match=err_msg):
        _print_dataset_info(dataset_name='non_existent_dataset')
