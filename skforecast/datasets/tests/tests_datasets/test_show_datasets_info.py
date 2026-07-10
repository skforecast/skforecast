# Unit test show_datasets_info
# ==============================================================================
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


def test_show_datasets_info_version_in_url(capsys):
    """
    `show_datasets_info` substitutes the `version` argument into the URL
    `{version}` placeholder when printing each dataset panel.
    """
    show_datasets_info(datasets_names=['h2o'], version='0.14.0')
    captured = capsys.readouterr()
    assert '0.14.0' in captured.out
    assert '{version}' not in captured.out


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
