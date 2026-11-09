# Unit test _naive_scales
# ==============================================================================
import numpy as np
from skforecast.metrics.metrics import _naive_scales


def test_naive_scales_output_equal_to_naive_errors_before_each_origin():
    """
    Test that the scales of each origin are the mean absolute and mean squared
    errors of the one-step naive forecast over `y[:origin]`, with the
    differences next to a missing value left out (a gap is not bridged).
    """
    rng = np.random.default_rng(123)
    y = rng.normal(size=60).cumsum()
    y[[5, 6, 30]] = np.nan
    origins = np.array([3, 6, 7, 8, 31, 32, 45, 60, 45])

    scale_abs, scale_sq = _naive_scales(y=y, origins=origins)

    expected_abs = np.array([np.nanmean(np.abs(np.diff(y[:o]))) for o in origins])
    expected_sq = np.array([np.nanmean(np.diff(y[:o]) ** 2) for o in origins])
    np.testing.assert_array_almost_equal(scale_abs, expected_abs)
    np.testing.assert_array_almost_equal(scale_sq, expected_sq)


def test_naive_scales_output_NaN_when_no_two_consecutive_values_before_origin():
    """
    Test that the scales are NaN when there are no two consecutive observed
    values before the origin (first value, or values separated by a missing
    value), and zero when the series is constant before the origin.
    """
    y = np.array([1., np.nan, 3., 3., 3., 5.])
    origins = np.array([0, 1, 2, 3, 4, 5, 6])

    scale_abs, scale_sq = _naive_scales(y=y, origins=origins)

    # NOTE: The differences are [nan, nan, 0, 0, 2]. Origins 0 to 3 see only
    # missing differences, 4 and 5 the constant values 3, and 6 sees 0, 0, 2.
    expected_abs = np.array([np.nan, np.nan, np.nan, np.nan, 0., 0., 2 / 3])
    expected_sq = np.array([np.nan, np.nan, np.nan, np.nan, 0., 0., 4 / 3])
    np.testing.assert_array_almost_equal(scale_abs, expected_abs)
    np.testing.assert_array_almost_equal(scale_sq, expected_sq)
