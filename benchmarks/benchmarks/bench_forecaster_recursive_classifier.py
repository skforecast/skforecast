################################################################################
#                 Benchmarking ForecasterRecursiveClassifier                   #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterRecursiveClassifier
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.utils import check_predict_input
from .benchmark_runner import BenchmarkRunner

# Global config
LEN_SERIES = 2000
STEPS = 100
RANDOM_STATE = 123


def _make_data(
    len_series: int = LEN_SERIES, 
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
    exog_pred : pandas DataFrame
        Exogenous variables for prediction.
    
    """

    rng = np.random.default_rng(random_state)
    y = pd.Series(
        data=rng.random.choice(a=['a', 'b', 'c'], size=len_series),
        index=pd.date_range(start='2010-01-01', periods=len_series, freq='h'),
        name='y'
    )

    # Create exogenous variables with temporal features
    exog = pd.DataFrame(index=y.index)
    exog['day_of_week'] = exog.index.dayofweek
    exog['week_of_year'] = exog.index.isocalendar().week.astype(int)
    exog['month'] = exog.index.month

    # Create future exogenous variables for prediction
    exog_pred = pd.DataFrame(
        index=pd.date_range(
            start=exog.index.max() + pd.Timedelta(hours=1),
            periods=steps,
            freq='h'
        )
    )
    exog_pred['day_of_week'] = exog_pred.index.dayofweek
    exog_pred['week_of_year'] = exog_pred.index.isocalendar().week.astype(int)
    exog_pred['month'] = exog_pred.index.month

    return y, exog, exog_pred


def run_benchmark_ForecasterRecursiveClassifier(output_dir):
    """
    Run all benchmarks for the ForecasterRecursiveClassifier class and save the results.
    """

    # Setup
    # ==========================================================================
    y, exog, exog_pred = _make_data()
    y_values = y.to_numpy()

    forecaster = ForecasterRecursiveClassifier(
        regressor=DummyRegressor(strategy='constant', constant=1.),
        lags=50,
        transformer_exog=StandardScaler(),
    )

    def ForecasterRecursiveClassifier__create_lags(forecaster, y):
        forecaster._create_lags(y=y, X_as_pandas=False, train_index=None)

    def ForecasterRecursiveClassifier__create_train_X_y(forecaster, y, exog):
        forecaster._create_train_X_y(y=y, exog=exog)

    def ForecasterRecursiveClassifier_fit(forecaster, y, exog):
        forecaster.fit(y=y, exog=exog)

    def ForecasterRecursiveClassifier_check_predict_inputs(forecaster, exog):
        check_predict_input(
            forecaster_name = type(forecaster).__name__,
            steps           = STEPS,
            is_fitted       = forecaster.is_fitted,
            exog_in_        = forecaster.exog_in_,
            index_type_     = forecaster.index_type_,
            index_freq_     = forecaster.index_freq_,
            window_size     = forecaster.window_size,
            last_window     = forecaster.last_window_,
            exog            = exog,
            exog_names_in_  = forecaster.exog_names_in_,
            interval        = None
        )

    def ForecasterRecursiveClassifier__create_predict_inputs(forecaster, exog):
        _ = forecaster._create_predict_inputs(
                steps = STEPS,
                exog  = exog
            )

    def ForecasterRecursiveClassifier_predict(forecaster, exog):
        forecaster.predict(steps=STEPS, exog=exog)

    def ForecasterRecursiveClassifier_predict_proba(forecaster, exog):
        forecaster.predict_proba(
            steps = STEPS,
            exog  = exog
        )
        
    def ForecasterRecursiveClassifier_backtesting(forecaster, y, exog):
        cv = TimeSeriesFold(
                 initial_train_size=1200,
                 steps=50,
                 refit=False
             )
        _ = backtesting_forecaster(
                forecaster=forecaster,
                y=y,
                exog=exog,
                cv=cv,
                metric='accuracy_score',
                n_jobs=1,
                show_progress=False
            )

    runner = BenchmarkRunner(repeat=30, output_dir=output_dir)
    _ = runner.benchmark(ForecasterRecursiveClassifier__create_lags, forecaster=forecaster, y=y_values)
    _ = runner.benchmark(ForecasterRecursiveClassifier__create_train_X_y, forecaster=forecaster, y=y, exog=exog)

    runner = BenchmarkRunner(repeat=10, output_dir=output_dir)
    _ = runner.benchmark(ForecasterRecursiveClassifier_fit, forecaster=forecaster, y=y, exog=exog)

    runner = BenchmarkRunner(repeat=30, output_dir=output_dir)
    forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
    _ = runner.benchmark(ForecasterRecursiveClassifier_check_predict_inputs, forecaster=forecaster, exog=exog_pred)
    _ = runner.benchmark(ForecasterRecursiveClassifier__create_predict_inputs, forecaster=forecaster, exog=exog_pred)
    _ = runner.benchmark(ForecasterRecursiveClassifier_predict, forecaster=forecaster, exog=exog_pred)
    _ = runner.benchmark(ForecasterRecursiveClassifier_predict_proba, forecaster=forecaster, exog=exog_pred)

    runner = BenchmarkRunner(repeat=5, output_dir=output_dir)
    _ = runner.benchmark(ForecasterRecursiveClassifier_backtesting, forecaster=forecaster, y=y, exog=exog)
