"""
BUG-rnn-old-pickle: a `ForecasterRnn` saved with 0.27.x still predicts in
0.28.x, but `fit`, `set_params` and a backtesting with `initial_train_size`
fail with a bare `AttributeError` (`warm_start`, `_initial_compile_config`,
`_initial_weights`) that does not say the forecaster must be created again.
The release notes document the limitation; the message is what is missing.

Needs keras and torch (KERAS_BACKEND=torch). Usage:
    PYTHONPATH=<worktree of origin/0.27.x> python rnn_old_pickle.py save
    python rnn_old_pickle.py load
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import sys
import warnings

import keras
import numpy as np
import pandas as pd
from _common import OUT, attempt, versions

from skforecast.deep_learning import ForecasterRnn, create_and_compile_model
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster_multiseries,
)
from skforecast.utils import load_forecaster, save_forecaster

warnings.filterwarnings("ignore")
print(versions())
n = 120
series = pd.DataFrame(
    {"a": np.sin(np.arange(n) / 5), "b": np.cos(np.arange(n) / 7)},
    index=pd.date_range("2020-01-01", periods=n, freq="D"),
)
file = str(OUT / "rnn_027.joblib")
if sys.argv[1] == "save":
    keras.utils.set_random_seed(0)
    model = create_and_compile_model(
        series=series, levels=["a"], lags=6, steps=2, recurrent_units=4, dense_units=4
    )
    forecaster = ForecasterRnn(
        model, levels=["a"], lags=6, fit_kwargs={"epochs": 2, "verbose": 0}
    )
    forecaster.fit(series)
    save_forecaster(forecaster, file)
    print(f"saved {file}")
    sys.exit()


def backtest(initial_train_size):
    cv = TimeSeriesFold(steps=2, initial_train_size=initial_train_size, refit=False)
    return backtesting_forecaster_multiseries(
        load_forecaster(file, verbose=False), series=series, levels=["a"], cv=cv,
        metric="mean_absolute_error", show_progress=False,
    )[0].iloc[0, 1]


attempt("predict", lambda: load_forecaster(file, verbose=False).predict().shape)
attempt("backtesting initial_train_size=None", lambda: backtest(None))
attempt("backtesting initial_train_size=100", lambda: backtest(100))
attempt("fit", lambda: load_forecaster(file, verbose=False).fit(series))
attempt(
    "set_params",
    lambda: load_forecaster(file, verbose=False).set_params(
        {"optimizer": keras.optimizers.SGD(0.01)}
    ),
)
