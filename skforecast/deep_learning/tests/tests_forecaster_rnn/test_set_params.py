# Unit test set_params ForecasterRnn using PyTorch backend
# ==============================================================================
import os
import numpy as np
import pandas as pd
os.environ["KERAS_BACKEND"] = "torch"
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning.utils import create_and_compile_model

series = pd.DataFrame(
    {"1": pd.Series(np.arange(10, dtype=float)), "2": pd.Series(np.arange(10, dtype=float))}
)
lags = 3
steps = 1
levels = "1"

model = create_and_compile_model(
    series=series,
    levels=levels,
    lags=lags,
    steps=steps,
    recurrent_units=8,
    dense_units=4,
    compile_kwargs={"optimizer": Adam(learning_rate=0.01), "loss": MeanSquaredError()},
)


def test_set_params_compiles_with_new_params_and_resets_estimator():
    """
    Test that set_params compiles the estimator with the new parameters, keeps
    the compile parameters not included in `params` (loss), resets the weights
    to the initial ones and the optimizer, and leaves the forecaster unfitted.
    The next fit uses the new parameters.
    """
    forecaster = ForecasterRnn(
        estimator=model, levels=levels, lags=lags, fit_kwargs={"epochs": 2, "verbose": 0}
    )
    initial_weights = forecaster.estimator.get_weights()
    forecaster.fit(series)
    forecaster.set_params({"optimizer": Adam(learning_rate=0.001)})

    assert forecaster.is_fitted is False
    assert isinstance(forecaster.estimator.loss, MeanSquaredError)
    assert isinstance(forecaster.estimator.optimizer, Adam)
    assert np.isclose(float(forecaster.estimator.optimizer.learning_rate.numpy()), 0.001)
    assert int(forecaster.estimator.optimizer.iterations.numpy()) == 0
    assert np.isclose(
        forecaster._initial_compile_config["optimizer"]["config"]["learning_rate"], 0.001
    )
    for weights, expected in zip(forecaster.estimator.get_weights(), initial_weights):
        np.testing.assert_array_equal(weights, expected)

    forecaster.fit(series)

    assert forecaster.is_fitted is True
    assert isinstance(forecaster.estimator.loss, MeanSquaredError)
    assert np.isclose(float(forecaster.estimator.optimizer.learning_rate.numpy()), 0.001)
