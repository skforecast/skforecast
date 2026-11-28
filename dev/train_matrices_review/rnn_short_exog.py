"""
Reproduce BUG-rnn-short-exog: ForecasterRnn with an `exog` that starts after
the first `window_size` observations of `series`.

`_create_train_X_y` accepts `len(exog) == len(series) - window_size`, but
`exog_train` gets `window_size` fewer rows than `X_train`, and `fit` fails in
keras with `ValueError: Data cardinality is ambiguous`.

Expected output with the bug:
    full exog  : X_train (26, 3, 2), exog_train (26, 2, 1)
    short exog : X_train (26, 3, 2), exog_train (23, 2, 1)
    fit with short exog: ValueError ...
"""

import os

os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np
import pandas as pd
from skforecast.deep_learning import ForecasterRnn, create_and_compile_model

n = 30
lags, steps = 3, 2
idx = pd.date_range('2020-01-01', periods=n, freq='D')
series = pd.DataFrame(
    {'a': np.arange(n, dtype=float), 'b': np.arange(n, dtype=float) * 10}, index=idx
)
exog = pd.DataFrame({'x': np.arange(n, dtype=float) + 1000}, index=idx)

model = create_and_compile_model(
    series=series, levels=['a'], lags=lags, steps=steps, exog=exog,
    recurrent_units=4, dense_units=4
)
forecaster = ForecasterRnn(
    estimator=model, levels=['a'], lags=lags, fit_kwargs={'epochs': 1, 'verbose': 0}
)

X_train, exog_train, *_ = forecaster._create_train_X_y(series=series, exog=exog)
print(f'full exog  : X_train {X_train.shape}, exog_train {exog_train.shape}')

X_train, exog_train, *_ = forecaster._create_train_X_y(
    series=series, exog=exog.iloc[lags:]
)
print(f'short exog : X_train {X_train.shape}, exog_train {exog_train.shape}')

try:
    forecaster.fit(series=series, exog=exog.iloc[lags:])
    print('fit with short exog: OK')
except Exception as e:
    print(f'fit with short exog: {type(e).__name__}: {str(e)[:200]}')
