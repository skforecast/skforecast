# Unit test _get_y_true_multiseries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.model_selection._utils import _get_y_true_multiseries

# Fixtures
index = pd.date_range("2020-01-01", periods=6, freq="D")
series_wide = pd.DataFrame(
    {"a": [0., 1., 2., 3., 4., 5.], "b": [10., 11., np.nan, 13., 14., 15.]},
    index=index
)
series_dict = {
    "a": series_wide["a"],
    "b": series_wide["b"].iloc[1:4],  # starts late and ends early
}


@pytest.mark.parametrize(
    "series, expected",
    [
        (series_wide, [3., 13., 4., 14., 3., 13., 5., 15.]),
        (series_dict, [3., 13., 4., np.nan, 3., 13., 5., np.nan]),
    ],
    ids=["wide DataFrame", "dict"]
)
def test_get_y_true_multiseries_output(series, expected):
    """
    Test the true value of each (date, level) pair, including repeated pairs
    (overlapping folds), in the order of the index. Dates outside a series
    of the dict are NaN.
    """
    dates = index[[3, 3, 4, 4, 3, 3, 5, 5]]
    levels = ["a", "b"] * 4
    multiindex = pd.MultiIndex.from_arrays([dates, levels], names=["idx", "level"])

    y_true = _get_y_true_multiseries(series=series, index=multiindex)

    np.testing.assert_array_equal(y_true, expected)


def test_get_y_true_multiseries_output_missing_value_and_integer_levels():
    """
    Test that a missing value of the series is NaN, with integer series names.
    """
    series = {0: series_wide["a"], 1: series_wide["b"]}
    multiindex = pd.MultiIndex.from_arrays(
        [index[[2, 2, 1]], [1, 0, 1]], names=["idx", "level"]
    )

    y_true = _get_y_true_multiseries(series=series, index=multiindex)

    np.testing.assert_array_equal(y_true, [np.nan, 2., 11.])


def test_get_y_true_multiseries_output_unused_levels_and_order_of_index_levels():
    """
    Test that the levels of the MultiIndex without rows (kept after filtering
    it) are ignored, and that the levels are found by name, whatever their
    order in the MultiIndex.
    """
    multiindex = pd.MultiIndex.from_arrays(
        [index[[3, 4, 3, 5]], ["a", "a", "b", "b"]], names=["idx", "level"]
    )
    filtered = multiindex[multiindex.get_level_values("level") == "b"]
    swapped = multiindex.swaplevel()

    assert list(filtered.levels[1]) == ["a", "b"]
    np.testing.assert_array_equal(
        _get_y_true_multiseries(series=series_wide, index=filtered), [13., 15.]
    )
    np.testing.assert_array_equal(
        _get_y_true_multiseries(series=series_dict, index=swapped),
        [3., 4., 13., np.nan]
    )
