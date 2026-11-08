# Smoke tests of the foundation backends (level 2)
# ==============================================================================
# Forecast with the real weights of the adapter selected with `--backend`,
# through the public API only, to follow the same path as a user. Only
# invariants are checked (shapes, index, finite values): the predicted values
# change with every version of the backend and with the hardware.
import numpy as np
import pandas as pd
import pytest
from skforecast.foundation import (
    ForecasterFoundation,
    FoundationModel,
    get_model_info,
)
from skforecast.model_selection import TimeSeriesFold, backtesting_foundation

CONTEXT_LENGTH = 64
STEPS = 12
N_OBSERVATIONS = 300

# Two hourly series with trend, daily seasonality and noise, and a numeric
# exogenous variable known in the future.
rng = np.random.default_rng(123)
index = pd.date_range("2024-01-01", periods=N_OBSERVATIONS + STEPS, freq="h")
hour = index.hour.to_numpy()
exog_all = pd.DataFrame(
    {"exog_1": np.cos(2 * np.pi * hour / 24) + rng.normal(0, 0.1, len(index))},
    index=index,
)
series_all = pd.DataFrame(
    {
        "series_1": (
            50
            + 0.02 * np.arange(len(index))
            + 5 * np.sin(2 * np.pi * hour / 24)
            + 2 * exog_all["exog_1"].to_numpy()
            + rng.normal(0, 0.5, len(index))
        ),
        "series_2": (
            20
            - 0.01 * np.arange(len(index))
            + 3 * np.cos(2 * np.pi * hour / 24)
            + rng.normal(0, 0.3, len(index))
        ),
    },
    index=index,
)
series = series_all.iloc[:N_OBSERVATIONS]
exog = exog_all.iloc[:N_OBSERVATIONS]
exog_future = exog_all.iloc[N_OBSERVATIONS:]
expected_index = index[N_OBSERVATIONS:]


@pytest.fixture(scope="module")
def model(backend_config):
    """
    FoundationModel of the adapter selected with `--backend`. Shared by all
    the tests so the weights are loaded only once.
    """
    return FoundationModel(
        model_id=backend_config["model_id"], context_length=CONTEXT_LENGTH
    )


def test_model_id_is_served_by_the_adapter(model, backend_config):
    """
    Test that the `model_id` of `adapters.toml` is routed to its adapter.
    """
    assert type(model.adapter).__name__ == backend_config["adapter"]


def test_predict_output_when_single_series(model):
    """
    Test the shape, index and values of predict with a single series.
    """
    forecaster = ForecasterFoundation(estimator=model)
    forecaster.fit(series=series["series_1"])
    predictions = forecaster.predict(steps=STEPS)

    assert predictions.columns.tolist() == ["level", "pred"]
    assert (predictions["level"] == "series_1").all()
    pd.testing.assert_index_equal(
        predictions.index, expected_index, check_names=False
    )
    assert np.isfinite(predictions["pred"].to_numpy(dtype=float)).all()


def test_predict_interval_output_when_single_series(model):
    """
    Test the shape, index and values of predict_interval with a single
    series, and that the bounds are ordered.
    """
    forecaster = ForecasterFoundation(estimator=model)
    forecaster.fit(series=series["series_1"])
    predictions = forecaster.predict_interval(steps=STEPS, interval=[0.1, 0.9])

    assert predictions.columns.tolist() == [
        "level", "pred", "lower_bound", "upper_bound"
    ]
    pd.testing.assert_index_equal(
        predictions.index, expected_index, check_names=False
    )
    values = predictions[["pred", "lower_bound", "upper_bound"]].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert (predictions["lower_bound"] <= predictions["upper_bound"]).all()


def test_predict_output_when_multiple_series(model):
    """
    Test the shape, levels and values of predict with two series.
    """
    forecaster = ForecasterFoundation(estimator=model)
    forecaster.fit(series=series)
    predictions = forecaster.predict(steps=STEPS)

    assert len(predictions) == STEPS * 2
    assert sorted(predictions["level"].unique()) == ["series_1", "series_2"]
    for level in ["series_1", "series_2"]:
        pd.testing.assert_index_equal(
            predictions.index[predictions["level"] == level],
            expected_index,
            check_names=False,
        )
    assert np.isfinite(predictions["pred"].to_numpy(dtype=float)).all()


def test_predict_output_when_exog(model, backend_config):
    """
    Test the shape, index and values of predict with an exogenous variable,
    for the adapters that support them.
    """
    if not get_model_info(backend_config["model_id"]).allow_exog:
        pytest.skip("The adapter does not support exogenous variables.")

    forecaster = ForecasterFoundation(estimator=model)
    forecaster.fit(series=series["series_1"], exog=exog)
    predictions = forecaster.predict(steps=STEPS, exog=exog_future)

    pd.testing.assert_index_equal(
        predictions.index, expected_index, check_names=False
    )
    assert np.isfinite(predictions["pred"].to_numpy(dtype=float)).all()


def test_predict_output_when_t0_unequal_series_and_covariates(model, backend):
    """
    Test T0 accepts shared NaN padding and heterogeneous future covariates.
    """
    if backend != "t0":
        pytest.skip("This regression exercises the T0 runtime contract.")

    short = series["series_2"].iloc[-(CONTEXT_LENGTH // 2):].copy()
    short.iloc[10] = np.nan
    short.iloc[-1] = np.nan
    heterogeneous_series = {"series_1": series["series_1"], "series_2": short}
    past = {
        "series_1": exog,
        "series_2": exog.loc[short.index].rename(columns={"exog_1": "exog_2"}),
    }
    future = {
        "series_1": exog_future,
        "series_2": exog_future.rename(columns={"exog_1": "exog_2"}),
    }
    forecaster = ForecasterFoundation(estimator=model)
    forecaster.fit(series=heterogeneous_series, exog=past)

    predictions = forecaster.predict_interval(
        steps=STEPS, interval=[0.1, 0.9], exog=future
    )

    assert len(predictions) == 2 * STEPS
    assert predictions["level"].unique().tolist() == ["series_1", "series_2"]
    for level in ["series_1", "series_2"]:
        pd.testing.assert_index_equal(
            predictions.index[predictions["level"] == level],
            expected_index,
            check_names=False,
        )
    values = predictions[["pred", "lower_bound", "upper_bound"]].to_numpy(float)
    assert np.isfinite(values).all()
    assert (predictions["lower_bound"] <= predictions["upper_bound"]).all()


@pytest.mark.parametrize("torch_dtype", ["float16", "bfloat16"])
def test_predict_interval_output_when_t0_mixed_precision(
    backend, backend_config, torch_dtype
):
    """
    Test T0's native mixed precision works through the public forecasting API.
    """
    if backend != "t0":
        pytest.skip("This regression exercises T0's mixed-precision runtime.")

    import torch

    model = FoundationModel(
        model_id=backend_config["model_id"],
        context_length=CONTEXT_LENGTH,
        device_map="cpu",
        torch_dtype=getattr(torch, torch_dtype),
    )
    forecaster = ForecasterFoundation(estimator=model)
    forecaster.fit(series=series["series_1"], exog=exog)

    predictions = forecaster.predict_interval(
        steps=STEPS, interval=[0.1, 0.9], exog=exog_future
    )

    pd.testing.assert_index_equal(
        predictions.index, expected_index, check_names=False
    )
    values = predictions[["pred", "lower_bound", "upper_bound"]].to_numpy(float)
    assert np.isfinite(values).all()
    assert (predictions["lower_bound"] <= predictions["upper_bound"]).all()


def test_backtesting_foundation_output_when_two_folds(model):
    """
    Test the shape and values of the metrics and predictions of
    backtesting_foundation with two folds and two series.
    """
    forecaster = ForecasterFoundation(estimator=model)
    cv = TimeSeriesFold(steps=STEPS, initial_train_size=len(series) - 2 * STEPS)
    metrics, predictions = backtesting_foundation(
        forecaster=forecaster,
        series=series,
        cv=cv,
        metric="mean_absolute_error",
        show_progress=False,
    )

    assert len(predictions) == 2 * STEPS * 2
    assert np.isfinite(predictions["pred"].to_numpy(dtype=float)).all()
    assert np.isfinite(metrics["mean_absolute_error"].to_numpy(dtype=float)).all()
