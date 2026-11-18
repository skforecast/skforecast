"""
ForecasterRnn: backtesting with intervals, out-of-sample residuals and refit.
Saves the predictions to compare base and branch (PYTHONPATH selects the code).

Needs keras (the minimum versions environment has keras 3.3.0 with torch).
Compare the two pickles with `pandas.testing.assert_frame_equal`.

PYTHONPATH=<base_dir> python dev/preprocessing_speedups/t3_rnn_identity.py <base.pkl>
python dev/preprocessing_speedups/t3_rnn_identity.py <branch.pkl>
"""

import os
import pickle
import sys
import warnings

os.environ.setdefault("KERAS_BACKEND", "torch")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import skforecast  # noqa: E402
from skforecast.deep_learning import ForecasterRnn, create_and_compile_model  # noqa: E402
from skforecast.model_selection import (  # noqa: E402
    TimeSeriesFold,
    backtesting_forecaster_multiseries,
)

rng = np.random.default_rng(0)
n = 300
index = pd.date_range("2020-01-01", periods=n, freq="D")
t = np.arange(n)
series = pd.DataFrame(
    {
        "l1": 10 + np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.3, n),
        "l2": 5 + np.cos(2 * np.pi * t / 7) + rng.normal(0, 0.3, n),
    },
    index=index,
)

import keras  # noqa: E402

keras.utils.set_random_seed(123)
model = create_and_compile_model(
    series=series, levels=["l1", "l2"], lags=7, steps=3,
    recurrent_layer="LSTM", recurrent_units=8, dense_units=8,
)
forecaster = ForecasterRnn(
    estimator=model, levels=["l1", "l2"], lags=7,
    fit_kwargs={"epochs": 3, "batch_size": 32, "verbose": 0},
)
keras.utils.set_random_seed(123)
forecaster.fit(series=series.iloc[:150])
residuals = rng.normal(0, 0.5, size=200)
y_pred = {"l1": rng.uniform(9, 11, 200), "l2": rng.uniform(4, 6, 200)}
y_true = {k: v + residuals for k, v in y_pred.items()}
forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)

cv = TimeSeriesFold(steps=3, initial_train_size=200, refit=10)
keras.utils.set_random_seed(123)
_, pred = backtesting_forecaster_multiseries(
    forecaster=forecaster, series=series, cv=cv, metric="mean_absolute_error",
    interval=[0.1, 0.9], interval_method="conformal",
    use_in_sample_residuals=False, use_binned_residuals=True,
    n_jobs=1, show_progress=False, suppress_warnings=True,
)
with open(sys.argv[1], "wb") as f:
    pickle.dump(pred, f)
print(skforecast.__file__, pred.shape)
