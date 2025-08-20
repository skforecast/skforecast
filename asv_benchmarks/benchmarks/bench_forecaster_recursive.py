################################################################################
#                       Benchmarking ForecasterRecursive                       #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

"""
Benchmark suite for ForecasterRecursive class.

This module contains benchmarks for measuring the performance of:
- Training operations (_create_train_X_y and fit)
- Prediction operations (predict and predict_interval)
- Backtesting with and without conformal intervals

The benchmarks are designed to work with ASV (Airspeed Velocity) and measure
both execution time and memory usage across different parameter combinations.
"""

from __future__ import annotations

# Lightweight imports at module level; heavy imports are done in setup()
# (ASV doesn't time imports or setup anyway)
import numpy as np
import pandas as pd

# Common parameters - keep as single-element lists for now to avoid complexity
LEN_SERIES = [2000]
LAGS = [50]
STEPS = 100
RANDOM_STATE = 321


def _make_data(
    len_series: int = 2000, 
    steps: int = STEPS, 
    random_state: int = RANDOM_STATE
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic time series data for benchmarking.
    
    Parameters
    ----------
    len_series : int
        Length of the time series.
    steps : int
        Number of prediction steps to generate.
    random_state : int
        Random seed for reproducibility.
        
    Returns
    -------
    y : pandas Series
        Target time series.
    exog : pandas DataFrame
        Exogenous variables for training.
    exog_prediction : pandas DataFrame
        Exogenous variables for prediction.
    
    """

    rng = np.random.default_rng(random_state)
    y = pd.Series(
        data=rng.normal(loc=20, scale=5, size=len_series),
        index=pd.date_range(start='2010-01-01', periods=len_series, freq='h'),
        name='y'
    )

    # Create exogenous variables with temporal features
    exog = pd.DataFrame(index=y.index)
    exog['day_of_week'] = exog.index.dayofweek
    exog['week_of_year'] = exog.index.isocalendar().week.astype(int)
    exog['month'] = exog.index.month

    # Create future exogenous variables for prediction
    exog_prediction = pd.DataFrame(
        index=pd.date_range(
            start=exog.index.max() + pd.Timedelta(hours=1),
            periods=steps,
            freq='h'
        )
    )
    exog_prediction['day_of_week'] = exog_prediction.index.dayofweek
    exog_prediction['week_of_year'] = exog_prediction.index.isocalendar().week.astype(int)
    exog_prediction['month'] = exog_prediction.index.month

    return y, exog, exog_prediction


# ---------------------------------------------------------------------
# 1) Preparación de matrices de entrenamiento y ajuste (fit)
# ---------------------------------------------------------------------
class TimeForecasterRecursive_Fit:
    """
    Benchmark training operations for ForecasterRecursive.
    
    Measures performance of:
    - _create_train_X_y (matrix preparation)
    - fit (model training)
    
    Does not include prediction operations.
    """

    # ASV parameter configuration
    params = [
        LAGS,       # list of different lag values
        LEN_SERIES  # list of different series lengths
    ]
    param_names = ['lags', 'len_series']

    def setup(self, lags, len_series):
        """
        Set up data and forecaster for benchmarking.
        """
        # Heavy imports here to avoid module-level import overhead
        from sklearn.dummy import DummyRegressor
        from sklearn.preprocessing import StandardScaler
        from skforecast.recursive import ForecasterRecursive

        # Generate data specific to these parameters
        y, exog, _ = _make_data(len_series=len_series)

        self.y = y
        self.exog = exog

        self.forecaster = ForecasterRecursive(
            regressor=DummyRegressor(strategy='constant', constant=1),
            lags=lags,
            transformer_y=StandardScaler(),
            transformer_exog=StandardScaler(),
        )

    def time_create_train_X_y(self, lags, len_series):
        """
        Benchmark _create_train_X_y matrix preparation.
        """
        self.forecaster._create_train_X_y(y=self.y, exog=self.exog)

    def time_fit(self, lags, len_series):
        """
        Benchmark model fitting.
        """
        self.forecaster.fit(y=self.y, exog=self.exog)
    
    def peakmem_fit(self, lags, len_series):
        """
        Measure peak memory usage during model fitting.
        """
        self.forecaster.fit(y=self.y, exog=self.exog)


# ---------------------------------------------------------------------
# 2) Predicción simple e intervalos conformales (modelo ya ajustado)
# ---------------------------------------------------------------------
class TimeForecasterRecursive_Predict:
    """
    Benchmark prediction operations for ForecasterRecursive.
    
    Measures performance of:
    - _create_predict_inputs (input preparation)
    - predict (point forecasts)
    - predict_interval with conformal method
    
    The model is pre-trained in setup (not timed).
    """

    # ASV parameter configuration
    params = [
        LAGS,       # list of different lag values
        LEN_SERIES  # list of different series lengths
    ]
    param_names = ['lags', 'len_series']

    def setup(self, lags, len_series):
        """
        Set up pre-trained forecaster for prediction benchmarking.
        """
        from sklearn.dummy import DummyRegressor
        from sklearn.preprocessing import StandardScaler
        from skforecast.recursive import ForecasterRecursive

        y, exog, exog_pred = _make_data(len_series=len_series)

        # Ensure prediction data matches the required steps
        exog_pred = exog_pred.iloc[:STEPS].copy()

        self.y = y
        self.exog = exog
        self.exog_pred = exog_pred
        self.steps = STEPS

        self.forecaster = ForecasterRecursive(
            regressor=DummyRegressor(strategy='constant', constant=1),
            lags=lags,
            transformer_y=StandardScaler(),
            transformer_exog=StandardScaler(),
        )

        # Pre-train the model (not timed)
        # Store residuals for conformal prediction
        self.forecaster.fit(
            y=self.y,
            exog=self.exog,
            store_in_sample_residuals=True
        )

    def time_create_predict_inputs(self, lags, len_series):
        """
        Benchmark _create_predict_inputs preparation.
        """
        self.forecaster._create_predict_inputs(
            steps=self.steps,
            exog=self.exog_pred,
            check_inputs=True
        )

    def time_predict(self, lags, len_series):
        """
        Benchmark point prediction.
        """
        self.forecaster.predict(steps=self.steps, exog=self.exog_pred)

    def time_predict_interval_conformal(self, lags, len_series):
        """
        Benchmark conformal prediction intervals.
        """
        self.forecaster.predict_interval(
            steps=self.steps,
            exog=self.exog_pred,
            interval=[5, 95],
            method="conformal"
        )
    
    def peakmem_predict(self, lags, len_series):
        """
        Measure peak memory usage during prediction.
        """
        self.forecaster.predict(steps=self.steps, exog=self.exog_pred)

    def peakmem_predict_interval_conformal(self, lags, len_series):
        """
        Measure peak memory usage during conformal prediction.
        """
        self.forecaster.predict_interval(
            steps=self.steps,
            exog=self.exog_pred,
            interval=[5, 95],
            method="conformal"
        )


# ---------------------------------------------------------------------
# 3) Backtesting con y sin intervalos conformales
# ---------------------------------------------------------------------
class TimeForecasterRecursive_Backtesting:
    """
    Benchmark backtesting operations for ForecasterRecursive.
    
    Measures performance of:
    - backtesting_forecaster (standard backtesting)
    - backtesting_forecaster with conformal intervals
    
    Note: These operations can take longer. If you get timeouts in ASV,
    consider reducing data sizes or using `--quick` during development.
    """

    # ASV parameter configuration
    params = [
        LAGS,       # list of different lag values
        LEN_SERIES  # list of different series lengths
    ]
    param_names = ['lags', 'len_series']

    def setup(self, lags, len_series):
        """
        Set up data and backtesting configuration.
        """

        from sklearn.dummy import DummyRegressor
        from sklearn.preprocessing import StandardScaler
        from skforecast.recursive import ForecasterRecursive
        from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster

        y, exog, _ = _make_data(len_series=len_series)

        self.y = y
        self.exog = exog
        self.backtesting_forecaster = backtesting_forecaster

        self.cv = TimeSeriesFold(
            initial_train_size=1200,
            fixed_train_size=True,
            steps=50,
        )

        self.forecaster = ForecasterRecursive(
            regressor=DummyRegressor(strategy='constant', constant=1),
            lags=lags,
            transformer_y=StandardScaler(),
            transformer_exog=StandardScaler(),
        )

    def time_backtesting(self, lags, len_series):
        """
        Benchmark standard backtesting.
        """
        self.backtesting_forecaster(
            forecaster=self.forecaster,
            y=self.y,
            exog=self.exog,
            cv=self.cv,
            metric="mean_squared_error",
            show_progress=False,
            verbose=False
        )

    def time_backtesting_conformal(self, lags, len_series):
        """
        Benchmark backtesting with conformal intervals.
        """
        self.backtesting_forecaster(
            forecaster=self.forecaster,
            y=self.y,
            exog=self.exog,
            cv=self.cv,
            interval=[5, 95],
            interval_method="conformal",
            metric="mean_squared_error",
            show_progress=False,
            verbose=False
        )

    def peakmem_backtesting(self, lags, len_series):
        """
        Measure peak memory usage during backtesting.
        """
        self.backtesting_forecaster(
            forecaster=self.forecaster,
            y=self.y,
            exog=self.exog,
            cv=self.cv,
            metric="mean_squared_error",
            show_progress=False,
            verbose=False
        )

    def peakmem_backtesting_conformal(self, lags, len_series):
        """
        Measure peak memory usage during conformal backtesting.
        """
        self.backtesting_forecaster(
            forecaster=self.forecaster,
            y=self.y,
            exog=self.exog,
            cv=self.cv,
            interval=[5, 95],
            interval_method="conformal",
            metric="mean_squared_error",
            show_progress=False,
            verbose=False
        )
