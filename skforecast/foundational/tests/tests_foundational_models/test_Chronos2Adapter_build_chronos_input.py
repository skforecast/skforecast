# Unit test Chronos2Adapter _build_chronos_input
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.foundational._adapters import Chronos2Adapter


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
    Test _build_chronos_input converts numeric (int) covariate arrays to float64.
    """
    int_exog = pd.DataFrame(
        {"feat_a": np.arange(30, dtype=int)},
        index=past_exog_df.index,
    )
    result = adapter._build_chronos_input(target=target, past_exog=int_exog)
    assert result["past_covariates"]["feat_a"].dtype == np.float64


def test_Chronos2Adapter_build_chronos_input_with_categorical_string_past_exog():
    """
    Test _build_chronos_input preserves string categorical past covariates as an
    object array without casting to float64, enabling Chronos-2 native categorical
    covariate support.
    """
    rng = np.random.default_rng(42)
    cat_exog = pd.DataFrame(
        {"weather": rng.choice(["sunny", "cloudy", "rainy"], size=30)},
        index=past_exog_df.index,
    )
    result = adapter._build_chronos_input(target=target, past_exog=cat_exog)
    assert "past_covariates" in result
    arr = result["past_covariates"]["weather"]
    assert arr.dtype.kind in ("U", "O"), f"Expected unicode or object dtype, got {arr.dtype}"
    np.testing.assert_array_equal(arr, cat_exog["weather"].to_numpy())


def test_Chronos2Adapter_build_chronos_input_with_categorical_string_future_exog():
    """
    Test _build_chronos_input preserves string categorical future covariates as an
    object array without casting to float64.
    """
    rng = np.random.default_rng(42)
    cat_future = pd.DataFrame(
        {"weather": rng.choice(["sunny", "cloudy", "rainy"], size=12)},
        index=future_exog_df.index,
    )
    result = adapter._build_chronos_input(target=target, future_exog=cat_future)
    assert "future_covariates" in result
    arr = result["future_covariates"]["weather"]
    assert arr.dtype.kind in ("U", "O"), f"Expected unicode or object dtype, got {arr.dtype}"
    np.testing.assert_array_equal(arr, cat_future["weather"].to_numpy())


def test_Chronos2Adapter_build_chronos_input_mixed_numeric_and_categorical():
    """
    Test _build_chronos_input handles DataFrames with a mix of numeric and
    categorical columns: numeric columns are cast to float64, string columns are
    left as object arrays.
    """
    rng = np.random.default_rng(42)
    mixed_exog = pd.DataFrame(
        {
            "temperature": np.arange(30, dtype=int),
            "weather": rng.choice(["sunny", "cloudy", "rainy"], size=30),
        },
        index=past_exog_df.index,
    )
    result = adapter._build_chronos_input(target=target, past_exog=mixed_exog)
    assert result["past_covariates"]["temperature"].dtype == np.float64
    assert result["past_covariates"]["weather"].dtype.kind in (
        "U",
        "O",
    ), f"Expected unicode or object dtype, got {result['past_covariates']['weather'].dtype}"


def test_Chronos2Adapter_build_chronos_input_with_pandas_categorical_dtype():
    """
    Test _build_chronos_input handles pandas Categorical dtype columns without
    raising a ValueError and preserves the underlying string values as an object
    array.
    """
    labels = (["spring", "summer", "fall", "winter"] * 7) + ["spring", "summer"]
    cat_exog = pd.DataFrame(
        {"season": pd.Categorical(labels)},
        index=past_exog_df.index,
    )
    result = adapter._build_chronos_input(target=target, past_exog=cat_exog)
    assert "past_covariates" in result
    arr = result["past_covariates"]["season"]
    assert arr.dtype.kind in ("U", "O"), f"Expected unicode or object dtype, got {arr.dtype}"
    np.testing.assert_array_equal(arr, np.array(labels))


def test_Chronos2Adapter_build_chronos_input_with_nullable_integer_dtype():
    """
    Test _build_chronos_input correctly casts pandas nullable integer Series
    (pd.Int64Dtype) to float64, converting pd.NA to np.nan.
    np.asarray() on nullable integer Series produces dtype=object with pd.NA
    sentinels, so the pandas-first branch is required to handle this correctly.
    """
    nullable_int_exog = pd.DataFrame(
        {"feat": pd.array([1, 2, None, 4, 5] * 6, dtype="Int64")},
        index=past_exog_df.index,
    )
    result = adapter._build_chronos_input(target=target, past_exog=nullable_int_exog)
    arr = result["past_covariates"]["feat"]
    assert arr.dtype == np.float64
    assert np.isnan(arr[2])
    np.testing.assert_array_equal(arr[[0, 1, 3, 4]], [1.0, 2.0, 4.0, 5.0])


def test_Chronos2Adapter_build_chronos_input_with_nullable_float_dtype():
    """
    Test _build_chronos_input correctly casts pandas nullable float Series
    (pd.Float64Dtype) to float64, preserving np.nan for pd.NA entries.
    """
    nullable_float_exog = pd.DataFrame(
        {"feat": pd.array([1.1, 2.2, None, 4.4, 5.5] * 6, dtype="Float64")},
        index=past_exog_df.index,
    )
    result = adapter._build_chronos_input(target=target, past_exog=nullable_float_exog)
    arr = result["past_covariates"]["feat"]
    assert arr.dtype == np.float64
    assert np.isnan(arr[2])
    np.testing.assert_allclose(arr[[0, 1, 3, 4]], [1.1, 2.2, 4.4, 5.5])


def test_Chronos2Adapter_build_chronos_input_with_nullable_boolean_dtype():
    """
    Test _build_chronos_input correctly casts pandas nullable boolean Series
    (pd.BooleanDtype) to float64 (True→1.0, False→0.0, pd.NA→np.nan).
    """
    nullable_bool_exog = pd.DataFrame(
        {"feat": pd.array([True, False, None, True, False] * 6, dtype="boolean")},
        index=past_exog_df.index,
    )
    result = adapter._build_chronos_input(target=target, past_exog=nullable_bool_exog)
    arr = result["past_covariates"]["feat"]
    assert arr.dtype == np.float64
    assert np.isnan(arr[2])
    np.testing.assert_array_equal(arr[[0, 1, 3, 4]], [1.0, 0.0, 1.0, 0.0])
