# Unit test Chronos2Adapter _build_chronos_input
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.foundational._foundational_models import Chronos2Adapter


# Fixtures
# ==============================================================================
adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")

target = np.arange(30, dtype=float)

past_exog_df = pd.DataFrame(
    {"feat_a": np.arange(30, dtype=float), "feat_b": np.arange(30, dtype=float) * 2},
    index=pd.date_range("2020-01-01", periods=30, freq="ME"),
)

past_exog_series = pd.Series(
    np.arange(30, dtype=float),
    index=pd.date_range("2020-01-01", periods=30, freq="ME"),
    name="feat_a",
)

future_exog_df = pd.DataFrame(
    {"feat_a": np.arange(12, dtype=float), "feat_b": np.arange(12, dtype=float) * 2},
    index=pd.date_range("2022-07-01", periods=12, freq="ME"),
)


# Tests Chronos2Adapter._build_chronos_input
# ==============================================================================
def test_Chronos2Adapter_build_chronos_input_target_only():
    """
    Test _build_chronos_input returns only 'target' key when no exog is passed.
    """
    result = adapter._build_chronos_input(target=target)
    assert set(result.keys()) == {"target"}
    np.testing.assert_array_equal(result["target"], target)


def test_Chronos2Adapter_build_chronos_input_target_is_float64():
    """
    Test _build_chronos_input converts target to float64.
    """
    int_target = np.arange(10, dtype=int)
    result = adapter._build_chronos_input(target=int_target)
    assert result["target"].dtype == np.float64


def test_Chronos2Adapter_build_chronos_input_with_past_exog_dataframe():
    """
    Test _build_chronos_input includes 'past_covariates' when a DataFrame is passed.
    """
    result = adapter._build_chronos_input(target=target, past_exog=past_exog_df)
    assert "past_covariates" in result
    assert set(result["past_covariates"].keys()) == {"feat_a", "feat_b"}
    np.testing.assert_array_almost_equal(
        result["past_covariates"]["feat_a"], past_exog_df["feat_a"].to_numpy()
    )
    np.testing.assert_array_almost_equal(
        result["past_covariates"]["feat_b"], past_exog_df["feat_b"].to_numpy()
    )


def test_Chronos2Adapter_build_chronos_input_with_past_exog_series():
    """
    Test _build_chronos_input wraps a Series into a single-key 'past_covariates'.
    """
    result = adapter._build_chronos_input(target=target, past_exog=past_exog_series)
    assert "past_covariates" in result
    assert set(result["past_covariates"].keys()) == {"feat_a"}
    np.testing.assert_array_almost_equal(
        result["past_covariates"]["feat_a"], past_exog_series.to_numpy()
    )


def test_Chronos2Adapter_build_chronos_input_with_future_exog_dataframe():
    """
    Test _build_chronos_input includes 'future_covariates' when future_exog is passed.
    """
    result = adapter._build_chronos_input(target=target, future_exog=future_exog_df)
    assert "future_covariates" in result
    assert set(result["future_covariates"].keys()) == {"feat_a", "feat_b"}
    np.testing.assert_array_almost_equal(
        result["future_covariates"]["feat_a"], future_exog_df["feat_a"].to_numpy()
    )


def test_Chronos2Adapter_build_chronos_input_no_past_covariates_key_when_no_past_exog():
    """
    Test that 'past_covariates' key is absent when past_exog is None.
    """
    result = adapter._build_chronos_input(target=target, future_exog=future_exog_df)
    assert "past_covariates" not in result


def test_Chronos2Adapter_build_chronos_input_no_future_covariates_key_when_no_future_exog():
    """
    Test that 'future_covariates' key is absent when future_exog is None.
    """
    result = adapter._build_chronos_input(target=target, past_exog=past_exog_df)
    assert "future_covariates" not in result


def test_Chronos2Adapter_build_chronos_input_with_all():
    """
    Test _build_chronos_input with all three inputs present.
    """
    result = adapter._build_chronos_input(
        target=target, past_exog=past_exog_df, future_exog=future_exog_df
    )
    assert set(result.keys()) == {"target", "past_covariates", "future_covariates"}


def test_Chronos2Adapter_build_chronos_input_covariate_values_are_float64():
    """
    Test _build_chronos_input converts covariate arrays to float64.
    """
    int_exog = pd.DataFrame(
        {"feat_a": np.arange(30, dtype=int)},
        index=past_exog_df.index,
    )
    result = adapter._build_chronos_input(target=target, past_exog=int_exog)
    assert result["past_covariates"]["feat_a"].dtype == np.float64
