import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from skforecast.datasets import fetch_dataset
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.recursive import ForecasterRecursive

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

datos = fetch_dataset(name='h2o_exog', raw=True, verbose=False)
datos['fecha'] = pd.to_datetime(datos['fecha'], format='%Y-%m-%d')
datos = datos.set_index('fecha')
datos = datos.asfreq('MS')
datos = datos.sort_index()

y_col = 'y'
exog_cols = ['exog_1', 'exog_2']

steps_test = 36
datos_train = datos[:-steps_test]
datos_test  = datos[-steps_test:]

def build_conv1d_encoder(lookback: int, emb_dim: int) -> keras.Model:
    inp = keras.Input(shape=(lookback, 1))
    x = layers.Conv1D(32, 7, padding="same")(inp)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, 5, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, 3, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    emb = layers.Dense(emb_dim, activation="relu", name="embedding")(x)
    out = layers.Dense(1, name="head")(emb)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model

class Conv1DTabularRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self, lookback: int, emb_dim: int = 32, epochs: int = 20,
        batch_size: int = 128, patience: int = 3, tab_max_iter: int = 600,
        tab_learning_rate: float = 0.05, random_state: int = 42, verbose: int = 0,
    ):
        self.lookback = int(lookback)
        self.emb_dim = int(emb_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.tab_max_iter = int(tab_max_iter)
        self.tab_learning_rate = float(tab_learning_rate)
        self.random_state = int(random_state)
        self.verbose = int(verbose)
        self._encoder = None
        self._emb_model = None
        self._tab_model = None
        self._n_exog_features = None
        self._lag_cols = None
        self._exog_cols = None

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected X to be a pandas DataFrame for fitting.")
        y = np.asarray(y).reshape(-1).astype(np.float32)

        self._n_exog_features = X.shape[1] - self.lookback
        lag_cols_candidate = [c for c in X.columns if str(c).startswith("lag_")]
        self._lag_cols = sorted(lag_cols_candidate, key=lambda s: int(str(s).split("_")[1]))
        self._exog_cols = [c for c in X.columns if c not in self._lag_cols]

        lag_mat = X[self._lag_cols].to_numpy(dtype=np.float32)
        lag_seq = lag_mat.reshape(len(X), self.lookback, 1)

        self._encoder = build_conv1d_encoder(self.lookback, self.emb_dim)
        es = keras.callbacks.EarlyStopping(monitor="loss", patience=self.patience, restore_best_weights=True)
        self._encoder.fit(lag_seq, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=[es])

        self._emb_model = keras.Model(inputs=self._encoder.input, outputs=self._encoder.get_layer("embedding").output)
        emb = self._emb_model.predict(lag_seq, batch_size=512, verbose=0)

        exog_mat = X[self._exog_cols].to_numpy(dtype=np.float32) if self._n_exog_features > 0 else np.zeros((len(X), 0), np.float32)
        X_enriched = np.hstack([exog_mat, emb])

        self._tab_model = HistGradientBoostingRegressor(
            max_depth=6, learning_rate=self.tab_learning_rate, max_iter=self.tab_max_iter, random_state=self.random_state,
        )
        self._tab_model.fit(X_enriched, y)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            lag_mat = X[self._lag_cols].to_numpy(dtype=np.float32)
            exog_mat = X[self._exog_cols].to_numpy(dtype=np.float32) if self._n_exog_features > 0 else np.zeros((len(X), 0), np.float32)
        elif isinstance(X, np.ndarray):
            lag_mat = X[:, :self.lookback].astype(np.float32)
            exog_mat = X[:, self.lookback:].astype(np.float32) if self._n_exog_features > 0 else np.zeros((len(X), 0), np.float32)
        
        lag_seq = lag_mat.reshape(len(lag_mat), self.lookback, 1)
        emb = self._emb_model.predict(lag_seq, batch_size=512, verbose=0)
        X_enriched = np.hstack([exog_mat, emb])
        return self._tab_model.predict(X_enriched)

lookback = 20
estimator = Conv1DTabularRegressor(lookback=lookback, emb_dim=32, epochs=15, batch_size=64, patience=3, verbose=0)

forecaster = ForecasterRecursive(regressor=estimator, lags=lookback)
forecaster.fit(y=datos_train[y_col], exog=datos_train[exog_cols])

predicciones = forecaster.predict(steps=steps_test, exog=datos_test[exog_cols])

cv = TimeSeriesFold(
    steps=12 * 3,
    initial_train_size=len(datos) - 12 * 9,
    fixed_train_size=False,
    refit=True,
)

metrica, predicciones_backtest = backtesting_forecaster(
    forecaster=forecaster, y=datos[y_col], exog=datos[exog_cols], cv=cv, metric='mean_squared_error', verbose=True
)

print(metrica)
print(predicciones_backtest.head())
