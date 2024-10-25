# Unit test _predict_and_calculate_metrics_multiseries_one_step_ahead
# ==============================================================================
import pytest
import pandas as pd
import joblib
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection import backtesting_forecaster_multiseries
from skforecast.model_selection._utils import _predict_and_calculate_metrics_multiseries_one_step_ahead
from skforecast.model_selection._split import TimeSeriesFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from skforecast.metrics import mean_absolute_scaled_error

# Fixtures
THIS_DIR = Path(__file__).parent.parent
series = pd.read_parquet(THIS_DIR/'fixture_multi_series_items_sales.parquet')
series = series.asfreq('D')
exog = pd.DataFrame({'day_of_week': series.index.dayofweek}, index = series.index)
series_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series.joblib')
exog_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series_exog.joblib')


@pytest.mark.parametrize(
        "forecaster",
        [
            ForecasterRecursiveMultiSeries(regressor=Ridge(random_state=678), lags=3),
            ForecasterRecursiveMultiSeries(
                regressor=Ridge(random_state=678),
                lags=3,
                transformer_series=None,
            ),
            ForecasterRecursiveMultiSeries(
                regressor=Ridge(random_state=678),
                lags=3,
                transformer_series=StandardScaler(),
                transformer_exog=StandardScaler()
            ),
            ForecasterDirectMultiVariate(
                regressor=Ridge(random_state=678),
                level='item_1',
                lags=3,
                steps=1,
                transformer_series=StandardScaler(),
                transformer_exog=StandardScaler()
            )
        ]
)
def test_predict_and_calculate_metrics_multiseries_one_step_ahead_output_equivalence_to_backtesting(forecaster):
    """
    Test that the output of _predict_and_calculate_metrics_multiseries_one_step_ahead is equivalent to
    the output of backtesting_forecaster_multiseries when steps=1 and refit=False.

    Results are not equivalent if diferentiation is included
    """

    initial_train_size = 927
    metrics = ['mean_absolute_error', mean_absolute_percentage_error, mean_absolute_scaled_error]

    if type(forecaster) is ForecasterRecursiveMultiSeries:
        levels = ['item_1', 'item_2', 'item_3']
    else:
        levels = ['item_1']

    cv = TimeSeriesFold(
            initial_train_size = initial_train_size,
            steps              = 1,
            refit              = False,
         )

    metrics_backtesting, pred_backtesting = backtesting_forecaster_multiseries(
        series=series,
        exog=exog,
        forecaster=forecaster,
        cv=cv,
        metric=metrics,
        levels=levels,
        add_aggregated_metric=True,
        show_progress=False
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_encoding,
        X_test_encoding
    ) = forecaster._train_test_split_one_step_ahead(
            series             = series,
            exog               = exog,
            initial_train_size = initial_train_size,
        )

    metrics_one_step_ahead, pred_one_step_ahead = _predict_and_calculate_metrics_multiseries_one_step_ahead(
        forecaster=forecaster,
        series=series,
        X_train = X_train,
        y_train= y_train,
        X_train_encoding = X_train_encoding,
        X_test = X_test,
        y_test = y_test,
        X_test_encoding = X_test_encoding,
        levels = levels,
        metrics = metrics,
        add_aggregated_metric = True
    )


    pd.testing.assert_frame_equal(metrics_one_step_ahead, metrics_backtesting)
    pd.testing.assert_frame_equal(pred_one_step_ahead, pred_backtesting)


@pytest.mark.parametrize(
        "forecaster",
        [
            ForecasterRecursiveMultiSeries(
                regressor          = LGBMRegressor(random_state=123, verbose=-1),
                lags               = 24,
                encoding           = 'ordinal',
                transformer_series = StandardScaler(),
                transformer_exog   = StandardScaler(),
                weight_func        = None,
                series_weights     = None,
                differentiation    = None,
                dropna_from_series = False,
            )
        ]
)
def test_predict_and_calculate_metrics_multiseries_one_step_ahead_output_equivalence_to_backtesting_when_series_is_dict(forecaster):
    """
    Test that the output of _predict_and_calculate_metrics_multiseries_one_step_ahead is
    equivalent to the output of backtesting_forecaster_multiseries when steps=1 and
    refit=False. Using series and exog as dictionaries.
    Results are not equivalent if diferentiation is included.
    ForecasterMultiVariate is not included because it is not possible to use dictionaries as input.
    """

    initial_train_size = 213
    metrics = ['mean_absolute_error', mean_absolute_percentage_error, mean_absolute_scaled_error]

    if type(forecaster) is ForecasterRecursiveMultiSeries:
        levels = ['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004']
    else:
        levels = ['id_1000']

    cv = TimeSeriesFold(
            initial_train_size = initial_train_size,
            steps              = 1,
            refit              = False,
        )

    metrics_backtesting, pred_backtesting = backtesting_forecaster_multiseries(
        series=series_dict,
        exog=exog_dict,
        forecaster=forecaster,
        cv=cv,
        metric=metrics,
        levels=levels,
        add_aggregated_metric=True,
        show_progress=False
    )


    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_encoding,
        X_test_encoding
    ) = forecaster._train_test_split_one_step_ahead(
            series             = series_dict,
            exog               = exog_dict,
            initial_train_size = initial_train_size,
        )
    metrics_one_step_ahead, pred_one_step_ahead = _predict_and_calculate_metrics_multiseries_one_step_ahead(
        forecaster=forecaster,
        series=series_dict,
        X_train = X_train,
        y_train= y_train,
        X_train_encoding = X_train_encoding,
        X_test = X_test,
        y_test = y_test,
        X_test_encoding = X_test_encoding,
        levels = levels,
        metrics = metrics,
        add_aggregated_metric = True
    )

    pd.testing.assert_frame_equal(metrics_one_step_ahead, metrics_backtesting)
    pd.testing.assert_frame_equal(pred_one_step_ahead, pred_backtesting)