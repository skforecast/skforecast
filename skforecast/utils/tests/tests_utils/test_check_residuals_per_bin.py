# Unit test check_residuals_per_bin
# ==============================================================================
import re
import pytest
import warnings
from skforecast.exceptions import ResidualsUsageWarning
from skforecast.utils import check_residuals_per_bin

advice = (
    "With fewer than 10 residuals per bin, prediction intervals estimated "
    "with `use_binned_residuals = True` are likely to be too narrow. Consider "
    "providing more out-of-sample residuals, reducing `n_bins` in the "
    "`binner_kwargs` of the forecaster, or predicting with "
    "`use_binned_residuals = False`."
)


def test_check_residuals_per_bin_ResidualsUsageWarning_when_few_residuals_per_bin():
    """
    Test ResidualsUsageWarning is raised when the average number of residuals
    per bin is lower than `min_residuals_per_bin`.
    """
    warn_msg = re.escape(
        f"Only 48 out-of-sample residuals are available for 10 bins, an "
        f"average of 4.8 residuals per bin. {advice}"
    )
    with pytest.warns(ResidualsUsageWarning, match=warn_msg):
        check_residuals_per_bin(n_residuals=48, n_bins=10)


def test_check_residuals_per_bin_ResidualsUsageWarning_when_few_residuals_per_bin_in_some_levels():
    """
    Test a single ResidualsUsageWarning, listing only the levels with an
    average number of residuals per bin lower than `min_residuals_per_bin`,
    is raised when `n_residuals` and `n_bins` are dicts.
    """
    n_residuals = {"l1": 48, "l2": 100, "l3": 20, "_unknown_level": 168}
    n_bins = {"l1": 10, "l2": 10, "l3": 3, "_unknown_level": 10}

    warn_msg = re.escape(
        f"The out-of-sample residuals of the following levels have, on average, "
        f"fewer than 10 residuals per bin: ['l1', 'l3']. {advice}"
    )
    with pytest.warns(ResidualsUsageWarning, match=warn_msg) as record:
        check_residuals_per_bin(n_residuals=n_residuals, n_bins=n_bins)

    assert len(record) == 1


@pytest.mark.parametrize(
    "n_residuals, n_bins, min_residuals_per_bin",
    [
        (100, 10, 10),
        (48, 10, 4),
        ({"l1": 100, "l2": 30}, {"l1": 10, "l2": 3}, 10),
    ],
    ids=lambda v: f"{v}"
)
def test_check_residuals_per_bin_no_warning_when_enough_residuals_per_bin(
    n_residuals, n_bins, min_residuals_per_bin
):
    """
    Test no warning is raised when the average number of residuals per bin is
    equal to or greater than `min_residuals_per_bin`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_residuals_per_bin(
            n_residuals           = n_residuals,
            n_bins                = n_bins,
            min_residuals_per_bin = min_residuals_per_bin
        )
