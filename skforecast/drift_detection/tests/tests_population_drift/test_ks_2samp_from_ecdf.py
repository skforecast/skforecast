# Unit test ks_2samp_from_ecdf
# ==============================================================================
import re
import pytest
import numpy as np
from scipy.stats import ecdf, ks_2samp
from ..._population_drift import ks_2samp_from_ecdf


def test_ks_2samp_ValueError_when_alternative_not_allowed():
    """
    Test ks_2samp_from_ecdf raises ValueError when alternative is not one of 
    the allowed values.
    """
    
    rng = np.random.default_rng(157839)
    data1 = rng.normal(loc=0, scale=1, size=30)
    data2 = rng.normal(loc=2, scale=3, size=30)

    ecdf1 = ecdf(data1)
    ecdf2 = ecdf(data2)

    error_msg = re.escape(
        "Invalid `alternative`. Must be 'two-sided', 'less', or 'greater'."
    )
    with pytest.raises(ValueError, match=error_msg):
        ks_2samp_from_ecdf(ecdf1, ecdf2, alternative="invalid")


def test_ks_2samp_from_ecdf_output():
    """
    Test that ks_2samp_from_ecdf produces the same statistic as scipy.stats.ks_2samp
    for different alternatives.
    """
    
    rng = np.random.default_rng(157839)
    data1 = rng.normal(loc=0, scale=1, size=500)
    data2 = rng.normal(loc=2, scale=3, size=500)

    ecdf1 = ecdf(data1)
    ecdf2 = ecdf(data2)

    expected_statistic, _ = ks_2samp(data1, data2, alternative="two-sided")
    computed_statistic = ks_2samp_from_ecdf(ecdf1, ecdf2, alternative="two-sided")
    assert np.isclose(expected_statistic, computed_statistic)

    expected_statistic, _ = ks_2samp(data1, data2, alternative="greater")
    computed_statistic = ks_2samp_from_ecdf(ecdf1, ecdf2, alternative="greater")
    assert np.isclose(expected_statistic, computed_statistic)

    expected_statistic, _ = ks_2samp(data1, data2, alternative="less")
    computed_statistic = ks_2samp_from_ecdf(ecdf1, ecdf2, alternative="less")
    assert np.isclose(expected_statistic, computed_statistic)