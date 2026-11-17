# Unit test grid_search_forecaster_multiseries ForecasterRnn using PyTorch backend
# ==============================================================================
import os
import numpy as np
import pandas as pd
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from skforecast.deep_learning.utils import create_and_compile_model
from skforecast.deep_learning import ForecasterRnn
from skforecast.model_selection._split import TimeSeriesFold
from skforecast.model_selection import grid_search_forecaster_multiseries

series = pd.DataFrame(
    {
        "1": np.arange(50, dtype=float),
        "2": np.arange(50, dtype=float),
    },
    index=pd.date_range("2020-01-01", periods=50, freq="D")
)

model = create_and_compile_model(
            series=series,
            levels=["1", "2"],
            lags=3,
            steps=4,
            recurrent_units=8,
            dense_units=4,
            compile_kwargs={"optimizer": Adam(), "loss": MeanSquaredError()},
        )


def test_grid_search_forecaster_multiseries_ForecasterRnn_with_optimizer_grid():
    """
    Test that grid_search_forecaster_multiseries evaluates every optimizer of
    `param_grid` with ForecasterRnn (through `set_params`), keeps the loss of
    the estimator and, with `return_best=True`, fits the forecaster with the
    optimizer of the best candidate.
    """
    forecaster = ForecasterRnn(
        estimator  = model,
        levels     = ["1", "2"],
        lags       = 3,
        fit_kwargs = {"epochs": 1, "verbose": 0}
    )
    cv = TimeSeriesFold(
            initial_train_size = len(series) - 12,
            steps              = forecaster.max_step,
            refit              = False
         )
    param_grid = {
        "optimizer": [Adam(learning_rate=0.01), Adam(learning_rate=0.001)]
    }

    keras.utils.set_random_seed(123)
    results = grid_search_forecaster_multiseries(
                  forecaster    = forecaster,
                  series        = series,
                  cv            = cv,
                  param_grid    = param_grid,
                  metric        = "mean_absolute_error",
                  return_best   = True,
                  verbose       = False,
                  show_progress = False
              )

    learning_rates = [
        float(optimizer.learning_rate.numpy()) for optimizer in results["optimizer"]
    ]
    best_learning_rate = learning_rates[0]

    assert results.shape[0] == 2
    np.testing.assert_array_almost_equal(sorted(learning_rates), [0.001, 0.01])
    assert list(results.columns) == [
        "levels", "lags", "lags_label", "params",
        "mean_absolute_error__weighted_average", "mean_absolute_error__average",
        "mean_absolute_error__pooling", "n_evaluated", "optimizer"
    ]
    assert results.filter(like="mean_absolute_error").notna().all().all()
    assert results["n_evaluated"].to_list() == [24, 24]
    assert forecaster.is_fitted is True
    assert isinstance(forecaster.estimator.loss, MeanSquaredError)
    assert np.isclose(
        float(forecaster.estimator.optimizer.learning_rate.numpy()),
        best_learning_rate
    )
