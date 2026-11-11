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


def test_print_dataset_info_url_of_main_branch(capsys, monkeypatch):
    """
    `_print_dataset_info` replaces the `{version}` placeholder of the URL with
    the main branch.
    """
    monkeypatch.setenv('COLUMNS', '200')
    _print_dataset_info('h2o')

    captured = capsys.readouterr()
    assert 'skforecast-datasets/main/data/h2o.csv' in captured.out
    assert '{version}' not in captured.out


@pytest.mark.parametrize(
    "columns",
    ['200', '60'],
    ids=lambda x: f'columns: {x}'
)
def test_print_dataset_info_panel_width(capsys, monkeypatch, columns):
    """
    `_print_dataset_info` prints a panel as wide as the panels of the skforecast
    warnings (88 characters) in a wide console, and no wider than the console
    when it is narrower (rich uses one column less in legacy Windows consoles).
    A URL longer than the panel is split across lines without losing any
    character.
    """
    monkeypatch.setenv('COLUMNS', columns)
    _print_dataset_info('bike_sharing_extended_features')

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    widths = {len(line) for line in lines}
    assert len(widths) == 1
    if columns == '200':
        assert widths == {88}
    else:
        assert widths.pop() in (59, 60)

    url = (
        'https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/'
        'data/bike_sharing_extended_features.csv'
    )
    inner_text = ''.join(line[2:-2].rstrip() for line in lines[1:-1])
    assert url in inner_text


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
