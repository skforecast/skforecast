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


def test_show_datasets_info_single_valid_dataset(capsys):
    """
    `show_datasets_info` with a single valid dataset name prints only that
    dataset's panel and not the others.
    """
    show_datasets_info(datasets_names=['h2o'])

    captured = capsys.readouterr()
    assert 'h2o' in captured.out
    assert 'items_sales' not in captured.out


def test_show_datasets_info_multiple_valid_datasets(capsys):
    """
    `show_datasets_info` with multiple valid dataset names prints a panel for
    each requested dataset.
    """
    show_datasets_info(datasets_names=['h2o', 'items_sales'])

    captured = capsys.readouterr()
    assert 'h2o' in captured.out
    assert 'items_sales' in captured.out


def test_show_datasets_info_invalid_name_prints_warning(capsys):
    """
    `show_datasets_info` prints a user-friendly message for any dataset name
    not present in the registry, without raising an exception.
    """
    show_datasets_info(datasets_names=['not_a_real_dataset'])

    captured = capsys.readouterr()
    assert 'not_a_real_dataset' in captured.out
    assert 'not available' in captured.out


def test_show_datasets_info_mixed_valid_and_invalid(capsys):
    """
    `show_datasets_info` handles a list that contains both valid and invalid
    dataset names: valid ones produce panels and invalid ones produce warnings.
    """
    show_datasets_info(datasets_names=['h2o', 'not_a_real_dataset'])

    captured = capsys.readouterr()
    assert 'h2o' in captured.out
    assert 'not_a_real_dataset' in captured.out
    assert 'not available' in captured.out
