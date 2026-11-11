# Unit test show_datasets_info
# ==============================================================================
import re
import warnings
import pytest
from skforecast.datasets import datasets, show_datasets_info


def test_show_datasets_info_all_datasets(capsys):
    """
    `show_datasets_info` with `datasets_names=None` (default) prints a panel
    for every available dataset.
    """
    show_datasets_info()

    captured = capsys.readouterr()
    for name in datasets:
        assert name in captured.out


def test_show_datasets_info_specific_datasets(capsys):
    """
    `show_datasets_info` with specific dataset names prints only those
    datasets' panels.
    """
    # Single dataset
    show_datasets_info(datasets_names=['h2o'])
    captured = capsys.readouterr()
    assert 'h2o' in captured.out
    assert 'items_sales' not in captured.out

    # Multiple datasets
    show_datasets_info(datasets_names=['h2o', 'items_sales'])
    captured = capsys.readouterr()
    assert 'h2o' in captured.out
    assert 'items_sales' in captured.out


def test_show_datasets_info_string_and_tuple(capsys):
    """
    `show_datasets_info` with a single dataset name as a string prints the
    panel of that dataset instead of iterating over its characters, and a
    tuple of names works as a list.
    """
    show_datasets_info(datasets_names='h2o')
    captured = capsys.readouterr()
    assert 'h2o' in captured.out
    assert 'Description' in captured.out
    assert 'not available' not in captured.out

    show_datasets_info(datasets_names=('h2o', 'items_sales'))
    captured = capsys.readouterr()
    assert 'h2o' in captured.out
    assert 'items_sales' in captured.out
    assert 'not available' not in captured.out


def test_show_datasets_info_empty_list_prints_nothing(capsys):
    """
    `show_datasets_info` with an empty list prints nothing, unlike
    `datasets_names=None`, which prints every dataset.
    """
    show_datasets_info(datasets_names=[])
    captured = capsys.readouterr()
    assert captured.out == ''


def test_show_datasets_info_version_FutureWarning(capsys, monkeypatch):
    """
    `show_datasets_info` emits a `FutureWarning` pointing to the caller when
    `version` is passed, and prints the URL of the main branch.
    """
    monkeypatch.setenv('COLUMNS', '200')
    warn_msg = re.escape(
        "`version` is deprecated since version 0.27.0 and has no effect: "
        "datasets are always fetched from the main branch of "
        "skforecast-datasets. It will be removed in version 0.29.0."
    )
    with pytest.warns(FutureWarning, match=warn_msg) as record:
        show_datasets_info(datasets_names=['h2o'], version='0.14.0')

    captured = capsys.readouterr()
    assert 'skforecast-datasets/main/data/h2o.csv' in captured.out
    assert '0.14.0' not in captured.out
    assert len(record) == 1
    assert record[0].filename == __file__

    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        show_datasets_info(datasets_names=['h2o'])


def test_show_datasets_info_invalid_names(capsys):
    """
    `show_datasets_info` prints a user-friendly message for invalid dataset
    names without raising an exception, and handles mixed valid/invalid lists.
    """
    # Only invalid
    show_datasets_info(datasets_names=['not_a_real_dataset'])
    captured = capsys.readouterr()
    assert 'not_a_real_dataset' in captured.out
    assert 'not available' in captured.out

    # Mixed valid and invalid
    show_datasets_info(datasets_names=['h2o', 'not_a_real_dataset'])
    captured = capsys.readouterr()
    assert 'h2o' in captured.out
    assert 'not_a_real_dataset' in captured.out
    assert 'not available' in captured.out
