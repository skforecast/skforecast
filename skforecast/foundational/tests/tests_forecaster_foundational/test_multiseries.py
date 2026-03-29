# Unit tests — ForecasterFoundational multi-series mode
# ==============================================================================
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from skforecast.foundational import ForecasterFoundational

from .fixtures_forecaster_foundational import (
    FakePipeline,
    make_forecaster,
    y,
)


# ---------------------------------------------------------------------------
# Shared multi-series fixtures
# ---------------------------------------------------------------------------

INDEX = pd.date_range("2020-01-01", periods=50, freq="ME")

# Wide DataFrame (two series)
series_df = pd.DataFrame(
    {
        "s1": np.arange(50, dtype=float),
        "s2": np.arange(50, 100, dtype=float),
    },
    index=INDEX,
)

# Same data as a dict
series_dict = {
    "s1": pd.Series(np.arange(50, dtype=float), index=INDEX, name="s1"),
    "s2": pd.Series(np.arange(50, 100, dtype=float), index=INDEX, name="s2"),
}

# Exog dict (one entry per series)
exog_dict = {
    "s1": pd.DataFrame({"feat_a": np.arange(50, dtype=float)}, index=INDEX),
    "s2": pd.DataFrame({"feat_a": np.arange(50, dtype=float) * 2}, index=INDEX),
}

# Future exog for predict (5 steps)
FORECAST_INDEX = pd.date_range("2024-07-31", periods=5, freq="ME")
future_exog_df = pd.DataFrame(
    {"feat_a": np.arange(70, 75, dtype=float)},
    index=FORECAST_INDEX,
)
future_exog_dict = {
    "s1": pd.DataFrame({"feat_a": np.arange(70, 75, dtype=float)}, index=FORECAST_INDEX),
    "s2": pd.DataFrame({"feat_a": np.arange(70, 75, dtype=float) * 2}, index=FORECAST_INDEX),
}

# Last-window override (as DataFrame and dict)
LW_INDEX = pd.date_range("2024-03-31", periods=20, freq="ME")
lw_df = pd.DataFrame(
    {
        "s1": np.arange(50, 70, dtype=float),
        "s2": np.arange(100, 120, dtype=float),
    },
    index=LW_INDEX,
)
lw_dict = {
    "s1": pd.Series(np.arange(50, 70, dtype=float), index=LW_INDEX, name="s1"),
    "s2": pd.Series(np.arange(100, 120, dtype=float), index=LW_INDEX, name="s2"),
}


# ===========================================================================
# fit — multi-series mode
# ===========================================================================

class TestFitMultiSeries:

    def test_fit_dataframe_sets_is_multiseries(self):
        forecaster = make_forecaster()
        forecaster.fit(series=series_df)
        assert forecaster._is_multiseries is True

    def test_fit_dict_sets_is_multiseries(self):
        forecaster = make_forecaster()
        forecaster.fit(series=series_dict)
        assert forecaster._is_multiseries is True

    def test_fit_dataframe_stores_series_names_in_(self):
        forecaster = make_forecaster()
        forecaster.fit(series=series_df)
        assert forecaster.series_names_in_ == ["s1", "s2"]

    def test_fit_dict_stores_series_names_in_(self):
        forecaster = make_forecaster()
        forecaster.fit(series=series_dict)
        assert forecaster.series_names_in_ == ["s1", "s2"]

    def test_fit_multiseries_series_name_in_is_None(self):
        forecaster = make_forecaster()
        forecaster.fit(series=series_df)
        assert forecaster.series_name_in_ is None

    def test_fit_multiseries_stores_training_range_as_dict(self):
        forecaster = make_forecaster()
        forecaster.fit(series=series_df)
        assert isinstance(forecaster.training_range_, dict)
        assert set(forecaster.training_range_.keys()) == {"s1", "s2"}
        pd.testing.assert_index_equal(
            forecaster.training_range_["s1"], series_df["s1"].index[[0, -1]]
        )
        pd.testing.assert_index_equal(
            forecaster.training_range_["s2"], series_df["s2"].index[[0, -1]]
        )

    def test_fit_dict_stores_training_range_as_dict(self):
        forecaster = make_forecaster()
        forecaster.fit(series=series_dict)
        assert isinstance(forecaster.training_range_, dict)
        pd.testing.assert_index_equal(
            forecaster.training_range_["s1"], series_dict["s1"].index[[0, -1]]
        )

    def test_fit_multiseries_sets_is_fitted(self):
        forecaster = make_forecaster()
        forecaster.fit(series=series_df)
        assert forecaster.is_fitted is True

    def test_fit_multiseries_stores_index_type(self):
        forecaster = make_forecaster()
        forecaster.fit(series=series_df)
        assert forecaster.index_type_ == pd.DatetimeIndex

    def test_fit_multiseries_stores_index_freq(self):
        forecaster = make_forecaster()
        forecaster.fit(series=series_df)
        assert forecaster.index_freq_ == INDEX.freq

    def test_fit_multiseries_exog_dict_stores_metadata(self):
        forecaster = make_forecaster()
        forecaster.fit(series=series_df, exog=exog_dict)
        assert forecaster.exog_in_ is True
        assert forecaster.exog_names_in_ == ["feat_a"]
        assert forecaster.exog_type_in_ == dict

    def test_fit_multiseries_broadcast_exog_stores_metadata(self):
        forecaster = make_forecaster()
        broadcast_exog = pd.DataFrame({"feat_a": np.arange(50, dtype=float)}, index=INDEX)
        forecaster.fit(series=series_df, exog=broadcast_exog)
        assert forecaster.exog_in_ is True
        assert forecaster.exog_names_in_ == ["feat_a"]

    def test_fit_multiseries_returns_self(self):
        forecaster = make_forecaster()
        result = forecaster.fit(series=series_df)
        assert result is forecaster

    def test_fit_invalid_type_raises_TypeError(self):
        forecaster = make_forecaster()
        with pytest.raises(TypeError, match="`series` must be"):
            forecaster.fit(series=[1, 2, 3])

    def test_fit_refit_clears_multiseries_state(self):
        """Re-fitting with a single series clears _is_multiseries."""
        forecaster = make_forecaster()
        forecaster.fit(series=series_df)
        assert forecaster._is_multiseries is True

        forecaster.fit(series=y)
        assert forecaster._is_multiseries is False
        assert forecaster.series_name_in_ == "y"
        assert forecaster.series_names_in_ == ["y"]


# ===========================================================================
# predict — multi-series mode
# ===========================================================================

class TestPredictMultiSeries:

    def _fitted(self, **kwargs):
        f = make_forecaster(**kwargs)
        f.fit(series=series_df)
        return f

    def test_predict_returns_dataframe(self):
        forecaster = self._fitted()
        result = forecaster.predict(steps=5)
        assert isinstance(result, pd.DataFrame)

    def test_predict_has_level_and_pred_columns(self):
        forecaster = self._fitted()
        result = forecaster.predict(steps=5)
        assert list(result.columns) == ["level", "pred"]

    def test_predict_row_count(self):
        """n_steps × n_series rows."""
        forecaster = self._fitted()
        result = forecaster.predict(steps=5)
        assert len(result) == 5 * 2  # 2 series

    def test_predict_level_values(self):
        """Level column tiles series names over steps."""
        forecaster = self._fitted()
        result = forecaster.predict(steps=3)
        expected_levels = np.tile(["s1", "s2"], 3)
        np.testing.assert_array_equal(result["level"].values, expected_levels)

    def test_predict_with_levels_filter(self):
        forecaster = self._fitted()
        result = forecaster.predict(steps=5, levels=["s1"])
        assert set(result["level"].unique()) == {"s1"}
        assert len(result) == 5

    def test_predict_with_levels_str(self):
        forecaster = self._fitted()
        result = forecaster.predict(steps=5, levels="s2")
        assert set(result["level"].unique()) == {"s2"}

    def test_predict_with_last_window_dataframe(self):
        forecaster = self._fitted()
        result = forecaster.predict(steps=5, last_window=lw_df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["level", "pred"]

    def test_predict_with_last_window_dict(self):
        forecaster = self._fitted()
        result = forecaster.predict(steps=5, last_window=lw_dict)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["level", "pred"]

    def test_unfitted_with_last_window_raises_not_fitted_error(self):
        """predict() raises NotFittedError even when last_window is provided
        and the forecaster has not been fitted."""
        forecaster = make_forecaster()
        with pytest.raises(NotFittedError):
            forecaster.predict(steps=3, last_window=lw_df)


# ===========================================================================
# predict_interval — multi-series mode
# ===========================================================================

class TestPredictIntervalMultiSeries:

    def _fitted(self):
        f = make_forecaster()
        f.fit(series=series_df)
        return f

    def test_predict_interval_returns_dataframe(self):
        forecaster = self._fitted()
        result = forecaster.predict_interval(steps=5)
        assert isinstance(result, pd.DataFrame)

    def test_predict_interval_columns(self):
        forecaster = self._fitted()
        result = forecaster.predict_interval(steps=5)
        assert list(result.columns) == ["level", "pred", "lower_bound", "upper_bound"]

    def test_predict_interval_row_count(self):
        forecaster = self._fitted()
        result = forecaster.predict_interval(steps=5)
        assert len(result) == 5 * 2

    def test_predict_interval_level_values(self):
        forecaster = self._fitted()
        result = forecaster.predict_interval(steps=3)
        expected_levels = np.tile(["s1", "s2"], 3)
        np.testing.assert_array_equal(result["level"].values, expected_levels)

    def test_predict_interval_with_levels_filter(self):
        forecaster = self._fitted()
        result = forecaster.predict_interval(steps=5, levels=["s2"])
        assert set(result["level"].unique()) == {"s2"}
        assert len(result) == 5

    def test_predict_interval_with_levels_str(self):
        forecaster = self._fitted()
        result = forecaster.predict_interval(steps=5, levels="s1")
        assert set(result["level"].unique()) == {"s1"}

    def test_predict_interval_lower_le_pred_le_upper(self):
        """lower_bound <= pred <= upper_bound for every row."""
        forecaster = self._fitted()
        result = forecaster.predict_interval(steps=5)
        assert (result["lower_bound"] <= result["pred"]).all()
        assert (result["pred"] <= result["upper_bound"]).all()

    def test_predict_interval_custom_interval(self):
        forecaster = self._fitted()
        result = forecaster.predict_interval(steps=5, interval=[5, 95])
        assert list(result.columns) == ["level", "pred", "lower_bound", "upper_bound"]

    def test_predict_interval_invalid_interval_length_raises(self):
        forecaster = self._fitted()
        with pytest.raises(ValueError, match="exactly two values"):
            forecaster.predict_interval(steps=5, interval=[10, 50, 90])

    def test_predict_interval_invalid_interval_order_raises(self):
        forecaster = self._fitted()
        with pytest.raises(ValueError, match="0 <= lower < upper <= 100"):
            forecaster.predict_interval(steps=5, interval=[90, 10])

    def test_predict_interval_with_last_window_dataframe(self):
        forecaster = self._fitted()
        result = forecaster.predict_interval(steps=5, last_window=lw_df)
        assert list(result.columns) == ["level", "pred", "lower_bound", "upper_bound"]


# ===========================================================================
# predict_quantiles — multi-series mode
# ===========================================================================

class TestPredictQuantilesMultiSeries:

    def _fitted(self):
        f = make_forecaster()
        f.fit(series=series_df)
        return f

    def test_predict_quantiles_returns_dataframe(self):
        forecaster = self._fitted()
        result = forecaster.predict_quantiles(steps=5)
        assert isinstance(result, pd.DataFrame)

    def test_predict_quantiles_default_columns(self):
        forecaster = self._fitted()
        result = forecaster.predict_quantiles(steps=5)
        assert list(result.columns) == ["level", "q_0.1", "q_0.5", "q_0.9"]

    def test_predict_quantiles_custom_quantiles_columns(self):
        forecaster = self._fitted()
        result = forecaster.predict_quantiles(steps=5, quantiles=[0.25, 0.75])
        assert list(result.columns) == ["level", "q_0.25", "q_0.75"]

    def test_predict_quantiles_row_count(self):
        forecaster = self._fitted()
        result = forecaster.predict_quantiles(steps=5)
        assert len(result) == 5 * 2

    def test_predict_quantiles_level_values(self):
        forecaster = self._fitted()
        result = forecaster.predict_quantiles(steps=3)
        expected_levels = np.tile(["s1", "s2"], 3)
        np.testing.assert_array_equal(result["level"].values, expected_levels)

    def test_predict_quantiles_with_levels_filter(self):
        forecaster = self._fitted()
        result = forecaster.predict_quantiles(steps=5, levels=["s1"])
        assert set(result["level"].unique()) == {"s1"}
        assert len(result) == 5

    def test_predict_quantiles_with_levels_str(self):
        forecaster = self._fitted()
        result = forecaster.predict_quantiles(steps=5, levels="s2")
        assert set(result["level"].unique()) == {"s2"}

    def test_predict_quantiles_values_match_quantile_levels(self):
        """
        FakePipeline sets q_<x> = x for every step, so we can verify values
        exactly.
        """
        forecaster = self._fitted()
        result = forecaster.predict_quantiles(steps=3, quantiles=[0.1, 0.9])
        np.testing.assert_allclose(result["q_0.1"].values, 0.1)
        np.testing.assert_allclose(result["q_0.9"].values, 0.9)

    def test_predict_quantiles_with_last_window_dict(self):
        forecaster = self._fitted()
        result = forecaster.predict_quantiles(steps=5, last_window=lw_dict)
        assert "level" in result.columns

    def test_predict_quantiles_dict_input(self):
        """Dict-based fit also routes through multi-series path."""
        forecaster = make_forecaster()
        forecaster.fit(series=series_dict)
        result = forecaster.predict_quantiles(steps=5)
        assert list(result.columns) == ["level", "q_0.1", "q_0.5", "q_0.9"]
        assert len(result) == 10


# ===========================================================================
# Single-series unchanged (smoke tests after refactor)
# ===========================================================================

class TestSingleSeriesUnchanged:

    def test_fit_single_series_name_in_still_str(self):
        forecaster = make_forecaster()
        forecaster.fit(series=y)
        assert forecaster.series_name_in_ == "y"
        assert forecaster.series_names_in_ == ["y"]
        assert forecaster._is_multiseries is False

    def test_fit_single_training_range_is_index(self):
        forecaster = make_forecaster()
        forecaster.fit(series=y)
        assert isinstance(forecaster.training_range_, pd.Index)
        assert not isinstance(forecaster.training_range_, dict)

    def test_predict_single_returns_series(self):
        forecaster = make_forecaster()
        forecaster.fit(series=y)
        result = forecaster.predict(steps=5)
        assert isinstance(result, pd.Series)
        assert result.name == "pred"

    def test_predict_interval_single_returns_dataframe_no_level_column(self):
        forecaster = make_forecaster()
        forecaster.fit(series=y)
        result = forecaster.predict_interval(steps=5)
        assert list(result.columns) == ["pred", "lower_bound", "upper_bound"]
        assert "level" not in result.columns

    def test_predict_quantiles_single_no_level_column(self):
        forecaster = make_forecaster()
        forecaster.fit(series=y)
        result = forecaster.predict_quantiles(steps=5)
        assert "level" not in result.columns
