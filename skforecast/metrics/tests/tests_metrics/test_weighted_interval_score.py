# Unit tests for weighted_interval_score
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.metrics import weighted_interval_score


class TestWeightedIntervalScoreInputValidation:
    """Tests for input validation in weighted_interval_score."""

    def test_raise_when_y_true_not_array(self):
        pf = np.array([98.0, 195.0])
        lb = np.array([[88.0, 80.0], [175.0, 165.0]])
        ub = np.array([[108.0, 118.0], [215.0, 225.0]])
        alphas = np.array([0.20, 0.05])
        msg = "`y_true` must be a 1D numpy array or pandas Series."
        with pytest.raises(TypeError, match=re.escape(msg)):
            weighted_interval_score("invalid", pf, lb, ub, alphas)

    def test_raise_when_point_forecast_not_array(self):
        y_true = np.array([100.0, 200.0])
        lb = np.array([[88.0, 80.0], [175.0, 165.0]])
        ub = np.array([[108.0, 118.0], [215.0, 225.0]])
        alphas = np.array([0.20, 0.05])
        msg = "`point_forecast` must be a 1D numpy array or pandas Series."
        with pytest.raises(TypeError, match=re.escape(msg)):
            weighted_interval_score(y_true, "invalid", lb, ub, alphas)

    def test_raise_when_lower_bounds_not_2d(self):
        y_true = np.array([100.0, 200.0])
        pf = np.array([98.0, 195.0])
        lb_1d = np.array([88.0, 175.0])
        ub = np.array([[108.0, 118.0], [215.0, 225.0]])
        alphas = np.array([0.20, 0.05])
        msg = "`lower_bounds` must be a 2D array with shape (n_observations, n_intervals)."
        with pytest.raises(ValueError, match=re.escape(msg)):
            weighted_interval_score(y_true, pf, lb_1d, ub, alphas)

    def test_raise_when_upper_bounds_not_2d(self):
        y_true = np.array([100.0, 200.0])
        pf = np.array([98.0, 195.0])
        lb = np.array([[88.0, 80.0], [175.0, 165.0]])
        ub_1d = np.array([108.0, 215.0])
        alphas = np.array([0.20, 0.05])
        msg = "`upper_bounds` must be a 2D array with shape (n_observations, n_intervals)."
        with pytest.raises(ValueError, match=re.escape(msg)):
            weighted_interval_score(y_true, pf, lb, ub_1d, alphas)

    def test_raise_when_alphas_not_1d(self):
        y_true = np.array([100.0, 200.0])
        pf = np.array([98.0, 195.0])
        lb = np.array([[88.0, 80.0], [175.0, 165.0]])
        ub = np.array([[108.0, 118.0], [215.0, 225.0]])
        alphas_2d = np.array([[0.20, 0.05]])
        msg = "`alphas` must be a 1D array of significance levels."
        with pytest.raises(ValueError, match=re.escape(msg)):
            weighted_interval_score(y_true, pf, lb, ub, alphas_2d)

    def test_raise_when_alphas_out_of_range(self):
        y_true = np.array([100.0, 200.0])
        pf = np.array([98.0, 195.0])
        lb = np.array([[88.0, 80.0], [175.0, 165.0]])
        ub = np.array([[108.0, 118.0], [215.0, 225.0]])
        msg = "All values in `alphas` must be strictly between 0 and 1."
        with pytest.raises(ValueError, match=re.escape(msg)):
            weighted_interval_score(y_true, pf, lb, ub, np.array([0.0, 0.05]))

    def test_raise_when_lower_bounds_shape_mismatch(self):
        y_true = np.array([100.0, 200.0])
        pf = np.array([98.0, 195.0])
        lb_wrong = np.array([[88.0], [175.0]])
        ub = np.array([[108.0, 118.0], [215.0, 225.0]])
        alphas = np.array([0.20, 0.05])
        msg = "`lower_bounds` must have shape (2, 2). Got (2, 1)."
        with pytest.raises(ValueError, match=re.escape(msg)):
            weighted_interval_score(y_true, pf, lb_wrong, ub, alphas)

    def test_raise_when_empty_y_true(self):
        msg = "`y_true` must have at least one element."
        with pytest.raises(ValueError, match=re.escape(msg)):
            weighted_interval_score(
                np.array([]),
                np.array([]),
                np.zeros((0, 2)),
                np.zeros((0, 2)),
                np.array([0.20, 0.05]),
            )


class TestWeightedIntervalScoreOutput:
    """Tests for the numerical output of weighted_interval_score."""

    def test_single_interval_no_violations(self):
        y_true = np.array([100.0])
        point_forecast = np.array([100.0])
        lower_bounds = np.array([[90.0]])
        upper_bounds = np.array([[110.0]])
        alphas = np.array([0.05])
        result = weighted_interval_score(y_true, point_forecast, lower_bounds, upper_bounds, alphas)
        expected = (1.0 / 1.5) * (0.0 + (0.05 / 2.0) * 20.0)
        assert abs(result - expected) < 1e-10

    def test_two_intervals_all_covered(self):
        """Reference: Bracher et al. (2021), eq. (2)."""
        y_true = np.array([100.0])
        point_forecast = np.array([98.0])
        lower_bounds = np.array([[88.0, 80.0]])
        upper_bounds = np.array([[108.0, 118.0]])
        alphas = np.array([0.20, 0.05])
        result = weighted_interval_score(y_true, point_forecast, lower_bounds, upper_bounds, alphas)
        expected = (1.0 / 2.5) * (0.5 * 2.0 + (0.20 / 2) * 20.0 + (0.05 / 2) * 38.0)
        assert abs(result - expected) < 1e-10

    def test_outer_violation_increases_wis(self):
        y_true = np.array([100.0])
        point_forecast = np.array([100.0])
        lower_bounds = np.array([[88.0, 80.0]])
        upper_bounds = np.array([[108.0, 118.0]])
        alphas = np.array([0.20, 0.05])
        result_inside = weighted_interval_score(y_true, point_forecast, lower_bounds, upper_bounds, alphas)
        y_true_outside = np.array([200.0])
        result_outside = weighted_interval_score(y_true_outside, point_forecast, lower_bounds, upper_bounds, alphas)
        assert result_outside > result_inside

    def test_three_observations_two_intervals(self):
        """Mean WIS reference from Bracher et al. (2021)."""
        y_true = np.array([100.0, 200.0, 150.0])
        point_forecast = np.array([98.0, 195.0, 155.0])
        lower_bounds = np.array([[88.0, 80.0], [175.0, 165.0], [138.0, 128.0]])
        upper_bounds = np.array([[108.0, 118.0], [215.0, 225.0], [168.0, 178.0]])
        alphas = np.array([0.20, 0.05])
        result = weighted_interval_score(y_true, point_forecast, lower_bounds, upper_bounds, alphas)
        wis_0 = (1.0 / 2.5) * (0.5 * 2.0 + 0.10 * 20.0 + 0.025 * 38.0)
        wis_1 = (1.0 / 2.5) * (0.5 * 5.0 + 0.10 * 40.0 + 0.025 * 60.0)
        wis_2 = (1.0 / 2.5) * (0.5 * 5.0 + 0.10 * 30.0 + 0.025 * 50.0)
        expected = (wis_0 + wis_1 + wis_2) / 3.0
        assert abs(result - expected) < 1e-10

    def test_wis_zero_when_perfect_point_and_degenerate_intervals(self):
        y_true = np.array([100.0, 200.0])
        point_forecast = np.array([100.0, 200.0])
        lower_bounds = np.array([[100.0, 100.0], [200.0, 200.0]])
        upper_bounds = np.array([[100.0, 100.0], [200.0, 200.0]])
        alphas = np.array([0.20, 0.05])
        result = weighted_interval_score(y_true, point_forecast, lower_bounds, upper_bounds, alphas)
        assert abs(result) < 1e-10

    def test_pandas_series_inputs_accepted(self):
        y_true = pd.Series([100.0])
        point_forecast = pd.Series([100.0])
        lower_bounds = np.array([[90.0, 80.0]])
        upper_bounds = np.array([[110.0, 120.0]])
        alphas = np.array([0.20, 0.05])
        result = weighted_interval_score(y_true, point_forecast, lower_bounds, upper_bounds, alphas)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_returns_float(self):
        y_true = np.array([100.0])
        point_forecast = np.array([98.0])
        lower_bounds = np.array([[88.0, 80.0]])
        upper_bounds = np.array([[108.0, 118.0]])
        alphas = np.array([0.20, 0.05])
        result = weighted_interval_score(y_true, point_forecast, lower_bounds, upper_bounds, alphas)
        assert isinstance(result, float)

    def test_list_alphas_accepted(self):
        y_true = np.array([100.0])
        point_forecast = np.array([100.0])
        lower_bounds = np.array([[90.0, 80.0]])
        upper_bounds = np.array([[110.0, 120.0]])
        result = weighted_interval_score(y_true, point_forecast, lower_bounds, upper_bounds, [0.20, 0.05])
        assert isinstance(result, float)
