# Unit test fit method with PyTorch backend
# ==============================================================================
import os
import pytest
import numpy as np
import pandas as pd
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning.utils import create_and_compile_model

series = pd.DataFrame(
    {"1": pd.Series(np.arange(5)), "2": pd.Series(np.arange(5))}
)
exog = pd.DataFrame(
    {"exog1": pd.Series(np.arange(5)), "exog2": pd.Series(np.arange(5))}
)
lags = 3
steps = 1
levels = "1"

model = create_and_compile_model(
    series=series, 
    levels=levels,    
    lags=lags,           
    steps=steps,              
    recurrent_layer="LSTM",
    recurrent_units=100,
    recurrent_layers_kwargs={"activation": "relu"},
    dense_units=[128, 64],
    dense_layers_kwargs={"activation": "relu"},
    output_dense_layer_kwargs={"activation": "linear"},
    compile_kwargs={"optimizer": Adam(learning_rate=0.01), "loss": MeanSquaredError()},
)

model_exog = create_and_compile_model(
    series=series, 
    exog=exog,
    levels=levels,    
    lags=lags,           
    steps=steps,              
    recurrent_layer="LSTM",
    recurrent_units=128,
    dense_units=64,
)


# Test case for fitting the forecaster without validation data
def test_fit_without_validation_data():
    """
    Test case for fitting the forecaster without validation data
    """

    forecaster = ForecasterRnn(estimator=model, levels=levels, lags=lags)

    assert forecaster.is_fitted is False

    forecaster.fit(series)

    assert forecaster.is_fitted is True
    assert forecaster.keras_backend_ == "torch"

    assert forecaster.series_names_in_ == ["1", "2"]
    assert forecaster.X_train_series_names_in_ == ["1", "2"]

    assert forecaster.training_range_.to_list() == [0, 4]

    last_window = pd.DataFrame({"1": [2, 3, 4], "2": [2, 3, 4]})
    np.testing.assert_array_almost_equal(forecaster.last_window_, last_window)


# Test case for fitting the forecaster with validation data
def test_fit_with_validation_data():
    """
    Test case for fitting the forecaster with validation data
    """

    series_val = pd.DataFrame(
        {"1": pd.Series(np.arange(5)), "2": pd.Series(np.arange(5))}
    )
    forecaster = ForecasterRnn(
        estimator=model,
        levels=levels,
        fit_kwargs={
            "epochs": 10,  # Number of epochs to train the model.
            "batch_size": 32,  # Batch size to train the model.
            "series_val": series_val,  # Validation data for model training.
        },
        lags=lags
    )
    assert forecaster.is_fitted is False
    forecaster.fit(series)
    assert forecaster.is_fitted is True
    assert forecaster.keras_backend_ == "torch"
    assert forecaster.history_ is not None


def test_fit_with_exog_and_validation_data():
    """
    Test case for fitting the forecaster with validation data
    """

    series_val = pd.DataFrame(
        {"1": pd.Series(np.arange(5)), "2": pd.Series(np.arange(5))}
    )
    exog_val = pd.DataFrame(
        {"exog1": pd.Series(np.arange(5)), "exog2": pd.Series(np.arange(5))}
    )
    forecaster = ForecasterRnn(
        estimator=model_exog,
        levels=levels,
        fit_kwargs={
            "epochs": 10,  # Number of epochs to train the model.
            "batch_size": 32,  # Batch size to train the model.
            "series_val": series_val,  # Validation data for model training.
            "exog_val": exog_val,  # Validation exogenous data for model training.
        },
        lags=lags
    )
    assert forecaster.is_fitted is False
    forecaster.fit(series, exog=exog)
    assert forecaster.is_fitted is True
    assert forecaster.keras_backend_ == "torch"
    assert forecaster.history_ is not None
    assert forecaster.exog_names_in_ == ["exog1", "exog2"]


def test_fit_resets_out_sample_residuals_on_refit():
    """
    Test that out_sample_residuals_ and out_sample_residuals_by_bin_ are reset
    to None when the forecaster is refitted.
    """
    forecaster = ForecasterRnn(estimator=model, levels=levels, lags=lags)
    forecaster.fit(series)
    forecaster.set_out_sample_residuals(
        y_true={"1": np.array([1.0, 2.0, 3.0, 4.0, 5.0])},
        y_pred={"1": np.array([0.0, 0.0, 0.0, 0.0, 0.0])},
    )

    assert forecaster.out_sample_residuals_ is not None
    assert forecaster.out_sample_residuals_by_bin_ is not None

    forecaster.fit(series)

    assert forecaster.out_sample_residuals_ is None
    assert forecaster.out_sample_residuals_by_bin_ is None


def test_fit_populates_binner_and_binned_residuals():
    """
    Test that fit populates binner, binner_intervals_, and in_sample_residuals_by_bin_
    when store_in_sample_residuals=True.
    """
    forecaster = ForecasterRnn(estimator=model, levels=levels, lags=lags)
    forecaster.fit(series, store_in_sample_residuals=True)

    assert isinstance(forecaster.binner, dict)
    assert isinstance(forecaster.binner_intervals_, dict)
    assert isinstance(forecaster.in_sample_residuals_by_bin_, dict)
    for level in forecaster.levels:
        assert level in forecaster.binner
        assert level in forecaster.binner_intervals_
        assert level in forecaster.in_sample_residuals_by_bin_
        assert isinstance(forecaster.in_sample_residuals_by_bin_[level], dict)
        assert len(forecaster.in_sample_residuals_by_bin_[level]) > 0


def test_fit_store_in_sample_residuals_False_stores_None_for_residuals():
    """
    Test that fit with store_in_sample_residuals=False sets residuals to None per level
    but still computes binner and binner_intervals_.
    """
    forecaster = ForecasterRnn(estimator=model, levels=levels, lags=lags)
    forecaster.fit(series, store_in_sample_residuals=False)

    assert isinstance(forecaster.binner, dict)
    assert isinstance(forecaster.binner_intervals_, dict)
    for level in forecaster.levels:
        assert level in forecaster.binner
        assert level in forecaster.binner_intervals_
    # Residuals should be None per level
    for level in forecaster.levels:
        assert forecaster.in_sample_residuals_[level] is None
        assert forecaster.in_sample_residuals_by_bin_[level] is None


@pytest.mark.parametrize(
    "warm_start",
    [False, True],
    ids=lambda warm_start: f"warm_start: {warm_start}",
)
def test_fit_warm_start_resets_or_continues_weights_and_optimizer(warm_start):
    """
    Test that, if `warm_start` is `False`, each fit starts from the initial
    weights with the optimizer reset (including a learning rate changed by a
    callback during training), so two fits with the same seed give the same
    weights and learning rate. If `warm_start` is `True`, the second fit
    continues from the first one.
    """
    forecaster = ForecasterRnn(
        estimator=model,
        levels=levels,
        lags=lags,
        fit_kwargs={
            "epochs": 2,
            "verbose": 0,
            "callbacks": [
                keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr * 0.5)
            ],
        },
        warm_start=warm_start,
    )

    keras.utils.set_random_seed(123)
    forecaster.fit(series)
    weights_first_fit = forecaster.estimator.get_weights()
    lr_first_fit = float(forecaster.estimator.optimizer.learning_rate.numpy())

    keras.utils.set_random_seed(123)
    forecaster.fit(series)
    weights_second_fit = forecaster.estimator.get_weights()
    lr_second_fit = float(forecaster.estimator.optimizer.learning_rate.numpy())

    same_weights = all(
        np.allclose(w_1, w_2)
        for w_1, w_2 in zip(weights_first_fit, weights_second_fit)
    )

    assert np.isclose(lr_first_fit, 0.01 * 0.5**2)
    if warm_start:
        assert not same_weights
        assert np.isclose(lr_second_fit, 0.01 * 0.5**4)
    else:
        assert same_weights
        assert np.isclose(lr_second_fit, lr_first_fit)


def test_fit_estimator_compiled_with_lambda_loss():
    """
    Test that a forecaster whose estimator is compiled with a lambda function
    as loss can be fitted more than once (the compile configuration is
    deserialized to reset the optimizer).
    """
    model_lambda_loss = create_and_compile_model(
        series=series,
        levels=levels,
        lags=lags,
        steps=steps,
        recurrent_units=8,
        dense_units=4,
        compile_kwargs={
            "optimizer": Adam(learning_rate=0.01),
            "loss": lambda y_true, y_pred: ((y_true - y_pred) ** 2).mean(),
        },
    )
    forecaster = ForecasterRnn(
        estimator=model_lambda_loss,
        levels=levels,
        lags=lags,
        fit_kwargs={"epochs": 1, "verbose": 0},
    )
    forecaster.fit(series)
    forecaster.fit(series)

    assert forecaster.is_fitted is True
    assert int(forecaster.estimator.optimizer.iterations.numpy()) == 1
