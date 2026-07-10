# Unit tests for winkler_score
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.metrics import winkler_score


class TestWinklerScoreInputValidation:
    """Tests for input validation in winkler_score."""

    def test_raise_when_y_true_not_array(self):
        lower = np.array([90.0, 180.0])
        upper = np.array([110.0, 220.0])
        msg = "`y_true` must be a 1D numpy array or pandas Series."
        with pytest.raises(TypeError, match=re.escape(msg)):
            winkler_score(y_true="invalid", lower_bound=lower, upper_bound=upper, alpha=0.05)

    def test_raise_when_lower_bound_not_array(self):
        y_true = np.array([100.0, 200.0])
        upper = np.array([110.0, 220.0])
        msg = "`lower_bound` must be a 1D numpy array or pandas Series."
        with pytest.raises(TypeError, match=re.escape(msg)):
            winkler_score(y_true=y_true, lower_bound="invalid", upper_bound=upper, alpha=0.05)

    def test_raise_when_upper_bound_not_array(self):
        y_true = np.array([100.0, 200.0])
        lower = np.array([90.0, 180.0])
        msg = "`upper_bound` must be a 1D numpy array or pandas Series."
        with pytest.raises(TypeError, match=re.escape(msg)):
            winkler_score(y_true=y_true, lower_bound=lower, upper_bound="invalid", alpha=0.05)

    def test_raise_when_alpha_out_of_range(self):
        y_true = np.array([100.0, 200.0])
        lower = np.array([90.0, 180.0])
        upper = np.array([110.0, 220.0])
        msg = "`alpha` must be a float strictly between 0 and 1."
        with pytest.raises(ValueError, match=re.escape(msg)):
            winkler_score(y_true=y_true, lower_bound=lower, upper_bound=upper, alpha=0.0)
        with pytest.raises(ValueError, match=re.escape(msg)):
            winkler_score(y_true=y_true, lower_bound=lower, upper_bound=upper, alpha=1.5)

    def test_raise_when_shapes_mismatch(self):
        y_true = np.array([100.0, 200.0])
        lower = np.array([90.0, 180.0, 140.0])
        upper = np.array([110.0, 220.0])
        msg = "`y_true`, `lower_bound`, and `upper_bound` must have the same shape."
        with pytest.raises(ValueError, match=re.escape(msg)):
            winkler_score(y_true=y_true, lower_bound=lower, upper_bound=upper, alpha=0.05)

    def test_raise_when_empty_array(self):
        msg = "`y_true` must have at least one element."
        with pytest.raises(ValueError, match=re.escape(msg)):
            winkler_score(
                y_true=np.array([]),
                lower_bound=np.array([]),
                upper_bound=np.array([]),
                alpha=0.05,
            )

    def test_raise_when_upper_below_lower(self):
        y_true = np.array([100.0])
        lower = np.array([110.0])
        upper = np.array([90.0])
        msg = "All values in `upper_bound` must be >= corresponding `lower_bound`."
        with pytest.raises(ValueError, match=re.escape(msg)):
            winkler_score(y_true=y_true, lower_bound=lower, upper_bound=upper, alpha=0.05)


class TestWinklerScoreOutput:
    """Tests for the numerical output of winkler_score."""

    def test_perfect_coverage_score_equals_interval_width(self):
        y_true = np.array([100.0, 200.0, 150.0])
        lower_bound = np.array([90.0, 190.0, 140.0])
        upper_bound = np.array([110.0, 210.0, 160.0])
        result = winkler_score(y_true, lower_bound, upper_bound, alpha=0.05)
        assert abs(result - 20.0) < 1e-10

    def test_single_observation_below_lower_bound(self):
        y_true = np.array([80.0])
        lower_bound = np.array([100.0])
        upper_bound = np.array([120.0])
        result = winkler_score(y_true, lower_bound, upper_bound, alpha=0.05)
        expected = 20.0 + (2.0 / 0.05) * (100.0 - 80.0)
        assert abs(result - expected) < 1e-10

    def test_single_observation_above_upper_bound(self):
        y_true = np.array([130.0])
        lower_bound = np.array([100.0])
        upper_bound = np.array([120.0])
        result = winkler_score(y_true, lower_bound, upper_bound, alpha=0.05)
        expected = 20.0 + (2.0 / 0.05) * (130.0 - 120.0)
        assert abs(result - expected) < 1e-10

    def test_mixed_observations_95_interval(self):
        y_true = np.array([100.0, 200.0, 150.0, 80.0])
        lower_bound = np.array([90.0, 180.0, 140.0, 100.0])
        upper_bound = np.array([110.0, 220.0, 160.0, 120.0])
        result = winkler_score(y_true, lower_bound, upper_bound, alpha=0.05)
        expected = (20.0 + 40.0 + 20.0 + 820.0) / 4
        assert abs(result - expected) < 1e-10

    def test_pandas_series_input(self):
        y_true = pd.Series([100.0, 200.0, 150.0])
        lower_bound = pd.Series([90.0, 190.0, 140.0])
        upper_bound = pd.Series([110.0, 210.0, 160.0])
        result = winkler_score(y_true, lower_bound, upper_bound, alpha=0.05)
        assert isinstance(result, float)
        assert result == 20.0

    def test_alpha_80_interval(self):
        y_true = np.array([80.0])
        lower_bound = np.array([100.0])
        upper_bound = np.array([120.0])
        result = winkler_score(y_true, lower_bound, upper_bound, alpha=0.20)
        expected = 20.0 + (2.0 / 0.20) * 20.0
        assert abs(result - expected) < 1e-10

    def test_returns_float(self):
        y_true = np.array([100.0])
        lower_bound = np.array([90.0])
        upper_bound = np.array([110.0])
        result = winkler_score(y_true, lower_bound, upper_bound, alpha=0.05)
        assert isinstance(result, float)
