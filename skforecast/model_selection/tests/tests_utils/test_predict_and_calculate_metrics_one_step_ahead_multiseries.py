# Unit test _predict_and_calculate_metrics_one_step_ahead_multiseries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection import backtesting_forecaster_multiseries
from skforecast.model_selection._utils import _predict_and_calculate_metrics_one_step_ahead_multiseries
from skforecast.model_selection._split import TimeSeriesFold
from skforecast.metrics import mean_absolute_scaled_error
from skforecast.utils._pandas_compat import PANDAS_GE_3

# Fixtures
from ..fixtures_model_selection_multiseries import (
    series_wide_dt_item_sales,
    series_dict_range_item_sales,
    series_dict_dt_item_sales,
    exog_wide_range_item_sales,
    exog_wide_dt_item_sales,
    series_dict_nans,
    exog_dict_nans,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore::skforecast.exceptions.MissingExogWarning"),
    pytest.mark.filterwarnings("ignore::skforecast.exceptions.MissingValuesWarning"),
]


# NOTE: Ignored, inputs checks are commented out in the function.
# def test_predict_and_calculate_metrics_one_step_ahead_multiseries_input_types():
#     """
#     Check if function raises errors when input parameters have wrong types.
#     """

#     # Mock inputs
#     forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(random_state=678), lags=3)
    
#     initial_train_size = 927
#     metrics = ['mean_absolute_error']
#     levels = ['item_1', 'item_2', 'item_3']
#     add_aggregated_metric = True

#     (
#         X_train,
#         y_train,
#         X_test,
#         y_test,
#         X_train_encoding,
#         X_test_encoding
#     ) = forecaster._train_test_split_one_step_ahead(
#             series             = series_dict_dt_item_sales,
#             exog               = exog_wide_dt_item_sales,
#             initial_train_size = initial_train_size,
#         )

#     # Test invalid type for series
#     err_msg = re.escape(
#         "`series` must be a pandas DataFrame or a dictionary of pandas DataFrames."
#     )
#     with pytest.raises(TypeError, match=err_msg):
#         _predict_and_calculate_metrics_one_step_ahead_multiseries(
#             forecaster, "invalid_series_type", X_train, y_train, X_test, y_test, 
#             X_train_encoding, X_test_encoding, levels, metrics, add_aggregated_metric
#         )

#     # Test invalid type for X_train
#     X_train_invalid = "invalid_X_train_type"
#     err_msg = re.escape(
#         f"`X_train` must be a pandas DataFrame. Got: {type(X_train_invalid)}"
#     )
#     with pytest.raises(TypeError, match=err_msg):
#         _predict_and_calculate_metrics_one_step_ahead_multiseries(
#             forecaster, series_dict_dt_item_sales, X_train_invalid, y_train, X_test, y_test, 
#             X_train_encoding, X_test_encoding, levels, metrics, add_aggregated_metric
#         )

#     # Test invalid type for y_train
#     y_train_invalid = "invalid_y_train_type"
#     err_msg = re.escape(
#         f"`y_train` must be a pandas Series or a dictionary of pandas Series. "
#         f"Got: {type(y_train_invalid)}"
#     )  
#     with pytest.raises(TypeError, match=err_msg):
#         _predict_and_calculate_metrics_one_step_ahead_multiseries(
#             forecaster, series_dict_dt_item_sales, X_train, y_train_invalid, X_test, y_test, 
#             X_train_encoding, X_test_encoding, levels, metrics, add_aggregated_metric
#         )

#     # Test invalid type for X_test
#     X_test_invalid = "invalid_X_test_type"
#     err_msg = re.escape(
#         f"`X_test` must be a pandas DataFrame. Got: {type(X_test_invalid)}"
#     )
#     with pytest.raises(TypeError, match=err_msg):
#         _predict_and_calculate_metrics_one_step_ahead_multiseries(
#             forecaster, series_dict_dt_item_sales, X_train, y_train, X_test_invalid, y_test, 
#             X_train_encoding, X_test_encoding, levels, metrics, add_aggregated_metric
#         )

#     # Test invalid type for y_test
#     y_test_invalid = "invalid_y_test_type"
#     err_msg = re.escape(
#         f"`y_test` must be a pandas Series or a dictionary of pandas Series. "
#         f"Got: {type(y_test_invalid)}"
#     )
#     with pytest.raises(TypeError, match=err_msg):
#         _predict_and_calculate_metrics_one_step_ahead_multiseries(
#             forecaster, series_dict_dt_item_sales, X_train, y_train, X_test, y_test_invalid, 
#             X_train_encoding, X_test_encoding, levels, metrics, add_aggregated_metric
#         )

#     # Test invalid type for X_train_encoding
#     X_train_encoding_invalid = "invalid_X_train_encoding_type"
#     err_msg = re.escape(
#         f"`X_train_encoding` must be a pandas Series. Got: {type(X_train_encoding_invalid)}"
#     )
#     with pytest.raises(TypeError, match=err_msg):
#         _predict_and_calculate_metrics_one_step_ahead_multiseries(
#             forecaster, series_dict_dt_item_sales, X_train, y_train, X_test, y_test, 
#             X_train_encoding_invalid, X_test_encoding, levels, metrics, add_aggregated_metric
#         )

#     # Test invalid type for X_test_encoding
#     X_test_encoding_invalid = "invalid_X_test_encoding_type"
#     err_msg = re.escape(
#         f"`X_test_encoding` must be a pandas Series. Got: {type(X_test_encoding_invalid)}"
#     )
#     with pytest.raises(TypeError, match=err_msg):
#         _predict_and_calculate_metrics_one_step_ahead_multiseries(
#             forecaster, series_dict_dt_item_sales, X_train, y_train, X_test, y_test, 
#             X_train_encoding, X_test_encoding_invalid, levels, metrics, add_aggregated_metric
#         )

#     # Test invalid type for levels
#     levels_invalid = "invalid_levels_type"
#     err_msg = re.escape(
#         f"`levels` must be a list. Got: {type(levels_invalid)}"
#     )
#     with pytest.raises(TypeError, match=err_msg):
#         _predict_and_calculate_metrics_one_step_ahead_multiseries(
#             forecaster, series_dict_dt_item_sales, X_train, y_train, X_test, y_test, 
#             X_train_encoding, X_test_encoding, levels_invalid, metrics, add_aggregated_metric
#         )

#     # Test invalid type for metrics
#     metrics_invalid = "invalid_metrics_type"
#     err_msg = re.escape(
#         f"`metrics` must be a list. Got: {type(metrics_invalid)}"
#     )
#     with pytest.raises(TypeError, match=err_msg):
#         _predict_and_calculate_metrics_one_step_ahead_multiseries(
#             forecaster, series_dict_dt_item_sales, X_train, y_train, X_test, y_test, 
#             X_train_encoding, X_test_encoding, levels, metrics_invalid, add_aggregated_metric
#         )

#     # Test invalid type for add_aggregated_metric
#     add_aggregated_metric_invalid = "invalid_add_aggregated_metric_type"
#     err_msg = re.escape(
#         f"`add_aggregated_metric` must be a boolean. Got: {type(add_aggregated_metric_invalid)}"
#     )
#     with pytest.raises(TypeError, match=err_msg):
#         _predict_and_calculate_metrics_one_step_ahead_multiseries(
#             forecaster, series_dict_dt_item_sales, X_train, y_train, X_test, y_test, 
#             X_train_encoding, X_test_encoding, levels, metrics, add_aggregated_metric_invalid
#         )


@pytest.mark.parametrize(
    "encoding",
    ['ordinal', 'onehot', None],
    ids=lambda encoding: f'encoding: {encoding}'
)
@pytest.mark.parametrize(
    "estimator",
    [LinearRegression(copy_X=False),
     make_pipeline(StandardScaler(copy=False), Ridge())],
    ids=['LinearRegression(copy_X=False)', 'pipeline StandardScaler(copy=False)']
)
def test_predict_and_calculate_metrics_one_step_ahead_multiseries_ValueError_when_estimator_modifies_X_train_in_place(
    estimator, encoding
):
    """
    Test ValueError is raised with ForecasterRecursiveMultiSeries when the
    estimator modifies the training matrix in place. A search with
    `OneStepAheadFold` reuses the matrix for every candidate, so the metrics
    of the next candidates would be wrong.
    """
    index = pd.date_range(start='2020-01-01', periods=30, freq='D')
    series = {
        f'l{i}': pd.Series(
                     np.arange(30, dtype=float) ** 1.5 + 10 * i, index=index, name=f'l{i}'
                 )
        for i in range(1, 4)
    }
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = estimator,
                     lags      = 3,
                     encoding  = encoding
                 )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_encoding,
        X_test_encoding,
        sample_weight,
        fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
            series             = series,
            initial_train_size = 20,
        )

    err_msg = re.escape(
        "The estimator has modified the training matrix in place during "
        "`fit`. The matrix is used again after training, to calculate the "
        "in-sample residuals or to fit the next candidates of a search "
        "with `OneStepAheadFold`, so the results would be wrong. This "
        "happens with estimators that do not copy their input, such as "
        "`LinearRegression(copy_X=False)` or a pipeline with "
        "`StandardScaler(copy=False)`. Use the default copy behavior of "
        "the estimator (`copy_X=True`, `copy=True`)."
    )
    with pytest.raises(ValueError, match=err_msg):
        _predict_and_calculate_metrics_one_step_ahead_multiseries(
            forecaster       = forecaster,
            series           = series,
            X_train          = X_train,
            y_train          = y_train,
            X_test           = X_test,
            y_test           = y_test,
            X_train_encoding = X_train_encoding,
            X_test_encoding  = X_test_encoding,
            levels           = ['l1', 'l2', 'l3'],
            metrics          = ['mean_absolute_error'],
            sample_weight    = sample_weight,
            fit_kwargs       = fit_kwargs
        )


@pytest.mark.parametrize(
    "forecaster",
    [
        ForecasterRecursiveMultiSeries(
            estimator=Ridge(random_state=678), lags=3, forecaster_id='multiseries_no_transformer'
        ),
        ForecasterDirectMultiVariate(
            estimator=Ridge(random_state=678), steps=1, level='item_1', lags=3, transformer_series=None,
            forecaster_id='multivariate_no_transformer'
        ),
        ForecasterRecursiveMultiSeries(
            estimator=Ridge(random_state=678),
            lags=3,
            transformer_series=StandardScaler(),
            transformer_exog=StandardScaler(),
            forecaster_id='multiseries_transformer'
        ),
        ForecasterDirectMultiVariate(
            estimator=Ridge(random_state=678),
            level='item_1',
            lags=3,
            steps=1,
            transformer_series=StandardScaler(),
            transformer_exog=StandardScaler(),
            forecaster_id='multivariate_transformer'
        ),
        ForecasterRecursiveMultiSeries(
            estimator=Ridge(random_state=678),
            lags=3,
            transformer_series=StandardScaler(),
            transformer_exog=StandardScaler(),
            differentiation=1,
            forecaster_id='multiseries_transformer_diff_1'
        ),
        ForecasterRecursiveMultiSeries(
            estimator=Ridge(random_state=678),
            lags=3,
            differentiation={'item_1': 1, 'item_2': None, 'item_3': 2, '_unknown_level': None},
            forecaster_id='multiseries_diff_dict'
        ),
        ForecasterDirectMultiVariate(
            estimator=Ridge(random_state=678),
            level='item_1',
            lags=3,
            steps=1,
            transformer_series=StandardScaler(),
            transformer_exog=StandardScaler(),
            differentiation=1,
            forecaster_id='multivariate_transformer_diff_1'
        ),
        ForecasterDirectMultiVariate(
            estimator=Ridge(random_state=678), level='item_1', lags=3, steps=1,
            differentiation=2, forecaster_id='multivariate_diff_2'
        )
    ],
ids=lambda forecaster: f'{forecaster.forecaster_id}')
def test_predict_and_calculate_metrics_one_step_ahead_multiseries_output_equivalence_to_backtesting(forecaster):
    """
    Test that the output of _predict_and_calculate_metrics_one_step_ahead_multiseries is equivalent to
    the output of backtesting_forecaster_multiseries when steps=1 and refit=False,
    also with differentiation (each prediction is reverted one step ahead from
    the observed values).
    """

    initial_train_size = 927
    metrics = ['mean_absolute_error', mean_absolute_percentage_error, mean_absolute_scaled_error]

    if type(forecaster) is ForecasterRecursiveMultiSeries:
        levels = ['item_1', 'item_2', 'item_3']
        series = series_dict_dt_item_sales
    else:
        levels = ['item_1']
        series = series_wide_dt_item_sales

    cv = TimeSeriesFold(
             initial_train_size = initial_train_size,
             steps              = 1,
             refit              = False
         )

    metrics_backtesting, pred_backtesting = backtesting_forecaster_multiseries(
        series=series,
        exog=exog_wide_dt_item_sales,
        forecaster=forecaster,
        cv=cv,
        metric=metrics,
        levels=levels,
        add_aggregated_metric=True,
        show_progress=False
    )
    pred_backtesting = (
        pred_backtesting
        .pivot(columns='level', values='pred')
        .rename_axis(None, axis=0)
        .rename_axis(None, axis=1)
        .asfreq('D')
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_encoding,
        X_test_encoding,
        sample_weight,
        fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
            series             = series,
            exog               = exog_wide_dt_item_sales,
            initial_train_size = initial_train_size,
        )

    metrics_one_step_ahead, pred_one_step_ahead = _predict_and_calculate_metrics_one_step_ahead_multiseries(
        forecaster=forecaster,
        series=series,
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        y_test = y_test,
        X_train_encoding = X_train_encoding,
        X_test_encoding = X_test_encoding,
        levels = levels,
        metrics = metrics,
        add_aggregated_metric = True,
        sample_weight = sample_weight,
        fit_kwargs = fit_kwargs
    )

    pd.testing.assert_frame_equal(metrics_one_step_ahead, metrics_backtesting)
    pd.testing.assert_frame_equal(pred_one_step_ahead, pred_backtesting)


@pytest.mark.parametrize(
    "metrics",
    [['mean_absolute_error'], ['mean_absolute_error', mean_absolute_scaled_error]],
    ids=lambda metrics: f'n_metrics: {len(metrics)}'
)
def test_predict_and_calculate_metrics_one_step_ahead_multiseries_output_when_levels_is_a_subset_of_series(metrics):
    """
    Test that, when `levels` is a subset of the series, only the predictions of
    `levels` are pooled and `n_evaluated` of the aggregations is their sum. The
    output is equivalent to the backtesting with steps=1 and refit=False.
    """

    initial_train_size = 927
    levels = ['item_1', 'item_3']
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator=Ridge(random_state=678), lags=3,
                     transformer_series=StandardScaler()
                 )

    cv = TimeSeriesFold(initial_train_size=initial_train_size, steps=1, refit=False)
    metrics_backtesting, _ = backtesting_forecaster_multiseries(
        forecaster=forecaster, series=series_dict_dt_item_sales, cv=cv,
        metric=metrics, levels=levels, show_progress=False
    )

    (
        X_train, y_train, X_test, y_test,
        X_train_encoding, X_test_encoding, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
            series=series_dict_dt_item_sales, initial_train_size=initial_train_size
        )
    metrics_one_step_ahead, _ = _predict_and_calculate_metrics_one_step_ahead_multiseries(
        forecaster=forecaster, series=series_dict_dt_item_sales,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        X_train_encoding=X_train_encoding, X_test_encoding=X_test_encoding,
        levels=levels, metrics=metrics, add_aggregated_metric=True,
        sample_weight=sample_weight, fit_kwargs=fit_kwargs, return_predictions=False
    )

    pd.testing.assert_frame_equal(metrics_one_step_ahead, metrics_backtesting)
    assert metrics_one_step_ahead['n_evaluated'].to_list() == [170, 170, 340, 340, 340]


@pytest.mark.parametrize("differentiation", 
    [1, {'item_1': 1, 'item_2': 1, 'item_3': 1, '_unknown_level': 1}], 
     ids = lambda diff: f'diff: {diff}')
def test_predict_and_calculate_metrics_one_step_ahead_multiseries_ForecasterRecursiveMultiSeries_diff(differentiation):
    """
    Test that the output of _predict_and_calculate_metrics_one_step_ahead_multiseries 
    when ForecasterRecursiveMultiSeries is used with differentiation.
    Each prediction is reverted one step ahead from the observed values, so
    the output is equal to a backtesting with `steps=1` and `refit=False`.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = Ridge(random_state=678),
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = StandardScaler(),
                     differentiation    = differentiation,
                     forecaster_id      = 'multiseries_transformer_diff'
                 )

    initial_train_size = 927
    metrics = ['mean_absolute_error', mean_absolute_percentage_error, mean_absolute_scaled_error]
    levels = ['item_1', 'item_2', 'item_3']

    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_encoding,
        X_test_encoding,
        sample_weight,
        fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
            series             = series_dict_range_item_sales,
            exog               = exog_wide_range_item_sales,
            initial_train_size = initial_train_size,
        )

    results = _predict_and_calculate_metrics_one_step_ahead_multiseries(
        forecaster=forecaster,
        series=series_dict_range_item_sales,
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        y_test = y_test,
        X_train_encoding = X_train_encoding,
        X_test_encoding = X_test_encoding,
        levels = levels,
        metrics = metrics,
        add_aggregated_metric = True,
        sample_weight = sample_weight,
        fit_kwargs = fit_kwargs
    )

    expected = (
        pd.DataFrame(
            {'levels': ['item_1',
                        'item_2',
                        'item_3',
                        'average',
                        'weighted_average',
                        'pooling'],
            'mean_absolute_error': [1.1503178261670213,
                                    2.398111266775402,
                                    3.5819278725450805,
                                    2.3767856551625015,
                                    2.3767856551625015,
                                    2.376785655162501],
            'mean_absolute_percentage_error': [0.05602579819201732,
                                                0.1543886758598446,
                                                0.2357057444158043,
                                                0.14870673948922208,
                                                0.14870673948922206,
                                                0.14870673948922208],
            'mean_absolute_scaled_error': [0.7520986899376542,
                                            1.019192035237137,
                                            0.9645191922741891,
                                            0.9119366391496602,
                                            0.9119366391496602,
                                            0.91193663914966],
            'n_evaluated': [170, 170, 170, 510, 510, 510]}
        ),
        pd.DataFrame(
            data = np.array([[25.60594070, 10.60920540, 11.74557632],
                             [25.41074260, 10.64749734, 10.25576089],
                             [24.69993772, 10.98250168, 13.85087248],
                             [26.03160180,  8.39308526, 12.42333537],
                             [24.05758462,  9.65409401,  8.14457109],
                             [23.30803874, 11.54684248, 14.17761292],
                             [24.19163094,  9.53768226, 12.15819564],
                             [26.06924239,  9.70976235,  8.85802054],
                             [25.61749886,  8.22828660, 11.07709330],
                             [25.89806846,  9.25172802,  8.99597145]]),
            columns = ['item_1', 'item_2', 'item_3'],
            index = pd.RangeIndex(start=927, stop=937, step=1)
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_frame_equal(results[1].head(10), expected[1])


@pytest.mark.parametrize(
    "return_predictions",
    [True, False],
    ids=lambda return_predictions: f'return_predictions={return_predictions}'
)
def test_predict_and_calculate_metrics_one_step_ahead_multiseries_output_ForecasterRecursiveMultiSeries_differentiation_dict(return_predictions):
    """
    Test that the output of _predict_and_calculate_metrics_one_step_ahead_multiseries 
    when ForecasterRecursiveMultiSeries is used with differentiation as dict.
    Each prediction is reverted one step ahead from the observed values, so
    the output is equal to a backtesting with `steps=1` and `refit=False`.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = Ridge(random_state=678),
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = StandardScaler(),
                     differentiation    = {'item_1': 1, 'item_2': None, 'item_3': 2, '_unknown_level': None},
                     forecaster_id      = 'multiseries_transformer_diff'
                 )

    initial_train_size = 927
    metrics = ['mean_absolute_error', mean_absolute_percentage_error, mean_absolute_scaled_error]
    levels = ['item_1', 'item_2', 'item_3']

    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_encoding,
        X_test_encoding,
        sample_weight,
        fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
            series             = series_dict_dt_item_sales,
            exog               = exog_wide_dt_item_sales,
            initial_train_size = initial_train_size,
        )

    results = _predict_and_calculate_metrics_one_step_ahead_multiseries(
        forecaster=forecaster,
        series=series_dict_dt_item_sales,
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        y_test = y_test,
        X_train_encoding = X_train_encoding,
        X_test_encoding = X_test_encoding,
        levels = levels,
        metrics = metrics,
        add_aggregated_metric = True,
        sample_weight = sample_weight,
        fit_kwargs = fit_kwargs,
        return_predictions = return_predictions
    )

    expected = (
        pd.DataFrame(
            {
                "levels": {
                    0: "item_1",
                    1: "item_2",
                    2: "item_3",
                    3: "average",
                    4: "weighted_average",
                    5: "pooling",
                },
                "mean_absolute_error": {
                    0: 1.2494901997609082,
                    1: 3.2454861506987474,
                    2: 6.328693330432144,
                    3: 3.6078898936306003,
                    4: 3.6078898936306003,
                    5: 3.6078898936306,
                },
                "mean_absolute_percentage_error": {
                    0: 0.060374026807567885,
                    1: 0.23902627227111134,
                    2: 0.41383952408047064,
                    3: 0.23774660771971665,
                    4: 0.23774660771971662,
                    5: 0.23774660771971667,
                },
                "mean_absolute_scaled_error": {
                    0: 0.8173020826169635,
                    1: 1.3794997216922271,
                    2: 1.7038654244982896,
                    3: 1.3002224096024932,
                    4: 1.3002224096024935,
                    5: 1.3002224096024935,
                },
                "n_evaluated": {0: 170, 1: 170, 2: 170, 3: 510, 4: 510, 5: 510},
            }
        ),
        pd.DataFrame(
            data = np.array([[25.86570499, 15.18029154,  8.97046675],
                             [26.84840255, 15.07864711, 10.19583831],
                             [25.61634276, 14.60073761, 20.84056163],
                             [25.66454517, 14.55994232, 14.04738353],
                             [23.67624741, 13.60961291,  2.08001327],
                             [22.74431718, 16.14084149, 13.81895425],
                             [22.70407896, 16.13231006, 12.11482165],
                             [26.18485300, 15.46833359,  7.06767977],
                             [27.55737144, 14.83278091, 11.57715035],
                             [27.21146088, 14.01618597,  8.15729678]]),
            columns = ['item_1', 'item_2', 'item_3'],
            index = pd.date_range(start='2014-07-16', periods=10, freq='D', unit='ns')
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    if return_predictions:
        pd.testing.assert_frame_equal(results[1].head(10), expected[1])
    else:
        assert results[1] is None


@pytest.mark.parametrize(
    "return_predictions",
    [True, False],
    ids=lambda return_predictions: f'return_predictions={return_predictions}'
)
def test_predict_and_calculate_metrics_one_step_ahead_multiseries_output_ForecasterDirectMultiVariate_differentiation(return_predictions):
    """
    Test that the output of _predict_and_calculate_metrics_one_step_ahead_multiseries 
    when ForecasterDirectMultiVariate is used with differentiation.
    Each prediction is reverted one step ahead from the observed values, so
    the output is equal to a backtesting with `steps=1` and `refit=False`.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator          = Ridge(random_state=678),
                     level              = 'item_1',
                     lags               = 3,
                     steps              = 1,
                     transformer_series = StandardScaler(),
                     transformer_exog   = StandardScaler(),
                     differentiation    = 1,
                     forecaster_id      = 'multivariate_transformer_diff'
                 )

    initial_train_size = 927
    metrics = ['mean_absolute_error', mean_absolute_percentage_error, mean_absolute_scaled_error]
    levels = ['item_1']

    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_encoding,
        X_test_encoding,
        sample_weight,
        fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
            series             = series_wide_dt_item_sales,
            exog               = exog_wide_dt_item_sales,
            initial_train_size = initial_train_size,
        )

    results = _predict_and_calculate_metrics_one_step_ahead_multiseries(
        forecaster=forecaster,
        series=series_wide_dt_item_sales,
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        y_test = y_test,
        X_train_encoding = X_train_encoding,
        X_test_encoding = X_test_encoding,
        levels = levels,
        metrics = metrics,
        add_aggregated_metric = True,
        sample_weight = sample_weight,
        fit_kwargs = fit_kwargs,
        return_predictions = return_predictions
    )

    expected = (
        pd.DataFrame(
            {'levels': ['item_1'],
             'mean_absolute_error': [0.8927713839844975],
             'mean_absolute_percentage_error': [0.04380407225142365],
             'mean_absolute_scaled_error': [0.5845726494502864],
             'n_evaluated': [170]}
        ),
        pd.DataFrame(
            data = np.array([[26.77436694],
                             [25.74161777],
                             [24.50723727],
                             [25.61056764],
                             [22.09248784],
                             [23.86420683],
                             [24.64356954],
                             [27.25277250],
                             [26.24504265],
                             [26.26450239]]),
            columns = ['item_1'],
            index = pd.date_range(start='2014-07-16', periods=10, freq='D', unit='ns')
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    if return_predictions:
        pd.testing.assert_frame_equal(results[1].head(10), expected[1])
    else:
        assert results[1] is None


@pytest.mark.parametrize(
    "forecaster",
    [
        ForecasterRecursiveMultiSeries(
            estimator          = LGBMRegressor(random_state=123, verbose=-1),
            lags               = 24,
            encoding           = 'ordinal',
            transformer_series = StandardScaler(),
            transformer_exog   = StandardScaler(),
            weight_func        = None,
            series_weights     = None,
            differentiation    = None,
            dropna_from_series = False,
            forecaster_id      = 'multiseries_lgbm'
        ),
        ForecasterRecursiveMultiSeries(
            estimator          = CatBoostRegressor(
                                     iterations          = 50,
                                     random_seed         = 123,
                                     verbose             = 0,
                                     allow_writing_files = False
                                 ),
            lags               = 24,
            encoding           = 'ordinal_category',
            transformer_series = StandardScaler(),
            transformer_exog   = StandardScaler(),
            weight_func        = None,
            series_weights     = None,
            differentiation    = None,
            dropna_from_series = False,
            forecaster_id      = 'multiseries_catboost_ordinal_category'
        )
    ],
ids=lambda forecaster: f'{forecaster.forecaster_id}')
def test_predict_and_calculate_metrics_one_step_ahead_multiseries_equivalence_backtesting_series_is_dict(forecaster):
    """
    Test that the output of _predict_and_calculate_metrics_one_step_ahead_multiseries is
    equivalent to the output of backtesting_forecaster_multiseries when steps=1 and
    refit=False. Using series and exog as dictionaries.
    ForecasterMultiVariate is not included because it is not possible to use dictionaries as input.
    Series 'id_1002' has no data in the test set: with CatBoost and
    encoding='ordinal_category', the codes of the other levels must not shift.
    """

    initial_train_size = 213
    metrics = ['mean_absolute_error', mean_absolute_percentage_error, mean_absolute_scaled_error]
    levels = ['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004']

    cv = TimeSeriesFold(
            initial_train_size = initial_train_size,
            steps              = 1,
            refit              = False,
        )

    metrics_backtesting, pred_backtesting = backtesting_forecaster_multiseries(
        series=series_dict_nans,
        exog=exog_dict_nans,
        forecaster=forecaster,
        cv=cv,
        metric=metrics,
        levels=levels,
        add_aggregated_metric=True,
        show_progress=False
    )
    pred_backtesting = (
        pred_backtesting
        .pivot(columns='level', values='pred')
        .rename_axis(None, axis=0)
        .rename_axis(None, axis=1)
        .asfreq('D')
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_encoding,
        X_test_encoding,
        sample_weight,
        fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
            series             = series_dict_nans,
            exog               = exog_dict_nans,
            initial_train_size = initial_train_size,
        )
    
    metrics_one_step_ahead, pred_one_step_ahead = _predict_and_calculate_metrics_one_step_ahead_multiseries(
        forecaster=forecaster,
        series=series_dict_nans,
        X_train = X_train,
        y_train= y_train,
        X_test = X_test,
        y_test = y_test,
        X_train_encoding = X_train_encoding,
        X_test_encoding = X_test_encoding,
        levels = levels,
        metrics = metrics,
        add_aggregated_metric = True,
        sample_weight = sample_weight,
        fit_kwargs = fit_kwargs
    )

    pd.testing.assert_frame_equal(metrics_one_step_ahead, metrics_backtesting)
    pd.testing.assert_frame_equal(pred_one_step_ahead, pred_backtesting)


@pytest.mark.parametrize(
    "forecaster",
    [
        ForecasterRecursiveMultiSeries(
            estimator          = LGBMRegressor(random_state=123, verbose=-1),
            lags               = 24,
            encoding           = 'ordinal',
            transformer_series = StandardScaler(),
            transformer_exog   = StandardScaler(),
            weight_func        = None,
            series_weights     = None,
            differentiation    = None,
            dropna_from_series = False,
            forecaster_id      = 'multiseries_lgbm'
        )
    ],
ids=lambda forecaster: f'{forecaster.forecaster_id}')
def test_predict_and_calculate_metrics_one_step_ahead_multiseries_equivalence_backtesting_series_is_dict_no_scaled(forecaster):
    """
    Test that the output of _predict_and_calculate_metrics_one_step_ahead_multiseries is
    equivalent to the output of backtesting_forecaster_multiseries when steps=1 and
    refit=False. Using series and exog as dictionaries.
    ForecasterMultiVariate is not included because it is not possible to use dictionaries as input.
    """

    initial_train_size = 213
    metrics = ['mean_absolute_error', mean_absolute_percentage_error]
    levels = ['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004']

    cv = TimeSeriesFold(
            initial_train_size = initial_train_size,
            steps              = 1,
            refit              = False,
        )

    metrics_backtesting, pred_backtesting = backtesting_forecaster_multiseries(
        series=series_dict_nans,
        exog=exog_dict_nans,
        forecaster=forecaster,
        cv=cv,
        metric=metrics,
        levels=levels,
        add_aggregated_metric=True,
        show_progress=False
    )
    pred_backtesting = (
        pred_backtesting
        .pivot(columns='level', values='pred')
        .rename_axis(None, axis=0)
        .rename_axis(None, axis=1)
        .asfreq('D')
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_encoding,
        X_test_encoding,
        sample_weight,
        fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
            series             = series_dict_nans,
            exog               = exog_dict_nans,
            initial_train_size = initial_train_size,
        )
    
    metrics_one_step_ahead, pred_one_step_ahead = _predict_and_calculate_metrics_one_step_ahead_multiseries(
        forecaster=forecaster,
        series=series_dict_nans,
        X_train = X_train,
        y_train= y_train,
        X_test = X_test,
        y_test = y_test,
        X_train_encoding = X_train_encoding,
        X_test_encoding = X_test_encoding,
        levels = levels,
        metrics = metrics,
        add_aggregated_metric = True,
        sample_weight = sample_weight,
        fit_kwargs = fit_kwargs
    )

    pd.testing.assert_frame_equal(metrics_one_step_ahead, metrics_backtesting)
    pd.testing.assert_frame_equal(pred_one_step_ahead, pred_backtesting)


@pytest.mark.parametrize(
    "estimator",
    [
        CatBoostRegressor(
            iterations=20, random_seed=123, verbose=0,
            allow_writing_files=False
        ),
        LGBMRegressor(n_estimators=50, verbose=-1, random_state=123),
        XGBRegressor(n_estimators=50, random_state=123),
        HistGradientBoostingRegressor(max_iter=50, random_state=123),
    ],
    ids=['CatBoostRegressor', 'LGBMRegressor', 'XGBRegressor', 'HistGradientBoostingRegressor']
)
def test_predict_and_calculate_metrics_one_step_ahead_multiseries_equivalence_backtesting_categorical_features_ForecasterRecursiveMultiSeries(estimator):
    """
    Test that the output of _predict_and_calculate_metrics_one_step_ahead_multiseries
    is equivalent to the output of backtesting_forecaster_multiseries when steps=1,
    refit=False, and categorical_features='auto'. Exog contains string and numeric
    columns with a ColumnTransformer (StandardScaler on numeric, passthrough on
    categorical).
    """
    n = len(series_wide_dt_item_sales)
    np.random.seed(42)
    exog_cat = pd.DataFrame(
        {
            'exog_num_1': np.random.rand(n),
            'exog_num_2': np.random.rand(n),
            'exog_cat_1': np.random.choice(['a', 'b', 'c'], size=n),
            'exog_cat_2': np.random.choice(['x', 'y'], size=n),
        },
        index=series_wide_dt_item_sales.index
    )

    transformer_exog = make_column_transformer(
                           (StandardScaler(), make_column_selector(dtype_include=np.number)),
                           remainder='passthrough',
                           verbose_feature_names_out=False,
                       ).set_output(transform='pandas')

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator            = estimator,
                     lags                 = 5,
                     encoding             = 'ordinal_category',
                     transformer_series   = StandardScaler(),
                     transformer_exog     = transformer_exog,
                     categorical_features = 'auto',
                 )

    initial_train_size = 927
    metrics = ['mean_absolute_error']
    levels = ['item_1', 'item_2', 'item_3']
    series = series_dict_dt_item_sales

    cv = TimeSeriesFold(
             initial_train_size = initial_train_size,
             steps              = 1,
             refit              = False,
         )

    metrics_backtesting, pred_backtesting = backtesting_forecaster_multiseries(
        series=series,
        exog=exog_cat,
        forecaster=forecaster,
        cv=cv,
        metric=metrics,
        levels=levels,
        add_aggregated_metric=True,
        show_progress=False
    )
    pred_backtesting = (
        pred_backtesting
        .pivot(columns='level', values='pred')
        .rename_axis(None, axis=0)
        .rename_axis(None, axis=1)
        .asfreq('D')
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_encoding,
        X_test_encoding,
        sample_weight,
        fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
            series             = series,
            exog               = exog_cat,
            initial_train_size = initial_train_size,
        )

    metrics_one_step_ahead, pred_one_step_ahead = _predict_and_calculate_metrics_one_step_ahead_multiseries(
        forecaster=forecaster,
        series=series,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_encoding=X_train_encoding,
        X_test_encoding=X_test_encoding,
        levels=levels,
        metrics=metrics,
        add_aggregated_metric=True,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs
    )

    pd.testing.assert_frame_equal(metrics_one_step_ahead, metrics_backtesting)
    pd.testing.assert_frame_equal(pred_one_step_ahead, pred_backtesting, check_dtype=False)


@pytest.mark.parametrize(
    "estimator",
    [
        CatBoostRegressor(
            iterations=20, random_seed=123, verbose=0,
            allow_writing_files=False
        ),
        LGBMRegressor(n_estimators=50, verbose=-1, random_state=123),
        XGBRegressor(n_estimators=50, random_state=123),
        HistGradientBoostingRegressor(max_iter=50, random_state=123),
    ],
    ids=['CatBoostRegressor', 'LGBMRegressor', 'XGBRegressor', 'HistGradientBoostingRegressor']
)
def test_predict_and_calculate_metrics_one_step_ahead_multiseries_equivalence_backtesting_categorical_features_ForecasterDirectMultiVariate(estimator):
    """
    Test that the output of _predict_and_calculate_metrics_one_step_ahead_multiseries
    is equivalent to the output of backtesting_forecaster_multiseries when steps=1,
    refit=False, and categorical_features='auto'. Exog contains string and numeric
    columns with a ColumnTransformer (StandardScaler on numeric, passthrough on
    categorical).
    """
    n = len(series_wide_dt_item_sales)
    np.random.seed(42)
    exog_cat = pd.DataFrame(
        {
            'exog_num_1': np.random.rand(n),
            'exog_num_2': np.random.rand(n),
            'exog_cat_1': np.random.choice(['a', 'b', 'c'], size=n),
            'exog_cat_2': np.random.choice(['x', 'y'], size=n),
        },
        index=series_wide_dt_item_sales.index
    )

    transformer_exog = make_column_transformer(
                           (StandardScaler(), make_column_selector(dtype_include=np.number)),
                           remainder='passthrough',
                           verbose_feature_names_out=False,
                       ).set_output(transform='pandas')

    forecaster = ForecasterDirectMultiVariate(
                     estimator            = estimator,
                     level                = 'item_1',
                     lags                 = 5,
                     steps                = 1,
                     transformer_series   = StandardScaler(),
                     transformer_exog     = transformer_exog,
                     categorical_features = 'auto',
                 )

    initial_train_size = 927
    metrics = ['mean_absolute_error']
    levels = ['item_1']
    series = series_wide_dt_item_sales

    cv = TimeSeriesFold(
             initial_train_size = initial_train_size,
             steps              = 1,
             refit              = False,
         )

    metrics_backtesting, pred_backtesting = backtesting_forecaster_multiseries(
        series=series,
        exog=exog_cat,
        forecaster=forecaster,
        cv=cv,
        metric=metrics,
        levels=levels,
        add_aggregated_metric=False,
        show_progress=False
    )
    pred_backtesting = (
        pred_backtesting
        .pivot(columns='level', values='pred')
        .rename_axis(None, axis=0)
        .rename_axis(None, axis=1)
        .asfreq('D')
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_encoding,
        X_test_encoding,
        sample_weight,
        fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
            series             = series,
            exog               = exog_cat,
            initial_train_size = initial_train_size,
        )

    metrics_one_step_ahead, pred_one_step_ahead = _predict_and_calculate_metrics_one_step_ahead_multiseries(
        forecaster=forecaster,
        series=series,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_encoding=X_train_encoding,
        X_test_encoding=X_test_encoding,
        levels=levels,
        metrics=metrics,
        add_aggregated_metric=False,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs
    )

    pd.testing.assert_frame_equal(metrics_one_step_ahead, metrics_backtesting)
    pd.testing.assert_frame_equal(pred_one_step_ahead, pred_backtesting, check_dtype=False)


@pytest.mark.parametrize(
    "freq",
    [
        pytest.param(
            "D",
            marks=pytest.mark.skipif(
                PANDAS_GE_3, reason="Since pandas 3, `tz_convert` drops 'D'."
            ),
        ),
        "24h",
    ],
    ids=lambda freq: f"freq: {freq}"
)
def test_predict_and_calculate_metrics_one_step_ahead_multiseries_when_index_is_tz_aware_and_utc_anchored(freq):
    """
    Test the output of _predict_and_calculate_metrics_one_step_ahead_multiseries
    when the series have a timezone-aware index that advances in fixed UTC steps
    (created in UTC and converted to a local timezone) and both the training
    and the test sets cross a daylight saving change. With 'D' the index keeps
    its frequency only in pandas 2; '24h' is the frequency that keeps it in
    both versions.
    """
    index = pd.date_range(
        start="2025-09-01", periods=250, freq=freq, tz="UTC"
    ).tz_convert("Europe/Madrid")
    rng = np.random.default_rng(123)
    series = {
        "l1": pd.Series(rng.normal(size=250), index=index, name="l1"),
        "l2": pd.Series(rng.normal(size=220), index=index[30:], name="l2"),
    }
    initial_train_size = 100
    forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=3)

    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_encoding,
        X_test_encoding,
        sample_weight,
        fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
            series             = series,
            initial_train_size = initial_train_size,
        )

    metrics, predictions = _predict_and_calculate_metrics_one_step_ahead_multiseries(
        forecaster=forecaster,
        series=series,
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        y_test = y_test,
        X_train_encoding = X_train_encoding,
        X_test_encoding = X_test_encoding,
        levels = ["l1", "l2"],
        metrics = ["mean_absolute_error", mean_absolute_scaled_error],
        add_aggregated_metric = False,
        sample_weight = sample_weight,
        fit_kwargs = fit_kwargs,
        return_predictions = True
    )

    pd.testing.assert_index_equal(predictions.index, index[initial_train_size:])
    assert predictions.notna().all(axis=None)
    assert metrics.notna().all(axis=None)


@pytest.mark.parametrize("differentiation", [1, 2], ids=lambda d: f"differentiation: {d}")
def test_predict_and_calculate_metrics_one_step_ahead_multiseries_differentiation_missing_values_in_test(differentiation):
    """
    Test that, with differentiation and missing values in the test set of a
    series, its true values are the observed ones and each prediction is
    built from the previous observed values plus the predicted difference
    (`y[t-1] + d` with order 1, `2 * y[t-1] - y[t-2] + d` with order 2). The
    rows with a missing target are left out of the test set, so the true
    values cannot be rebuilt with the cumulative sum of the differences. (A
    backtesting cannot be the reference: it raises an error when the last
    value before a date is missing.)
    """
    series = {k: v.copy() for k, v in series_dict_dt_item_sales.items()}
    series['item_1'].iloc[[950, 951, 1000]] = np.nan
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = HistGradientBoostingRegressor(max_iter=20, random_state=123),
                     lags               = 7,
                     encoding           = 'ordinal',
                     differentiation    = differentiation,
                     dropna_from_series = True
                 )
    received = {}

    def capture(y_true, y_pred):
        received['y_true'] = np.asarray(y_true)
        received['y_pred'] = np.asarray(y_pred)
        return 0.0

    (
        X_train, y_train, X_test, y_test,
        X_train_encoding, X_test_encoding, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
            series=series, exog=None, initial_train_size=927
        )
    _predict_and_calculate_metrics_one_step_ahead_multiseries(
        forecaster=forecaster, series=series,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        X_train_encoding=X_train_encoding, X_test_encoding=X_test_encoding,
        levels=['item_1'], metrics=[capture], add_aggregated_metric=False,
        sample_weight=sample_weight, fit_kwargs=fit_kwargs, return_predictions=False
    )

    is_item_1 = (np.asarray(X_test_encoding) == 'item_1')
    test_index = y_test.index[is_item_1]
    diff_pred = forecaster.estimator.predict(X_test[is_item_1])
    y_item_1 = series['item_1']
    previous_1 = y_item_1.shift(1).loc[test_index].to_numpy()
    previous_2 = y_item_1.shift(2).loc[test_index].to_numpy()
    if differentiation == 1:
        expected_y_pred = previous_1 + diff_pred
    else:
        expected_y_pred = 2 * previous_1 - previous_2 + diff_pred

    assert len(test_index) < len(y_item_1) - 927
    np.testing.assert_array_almost_equal(
        received['y_true'], y_item_1.loc[test_index].to_numpy()
    )
    np.testing.assert_array_almost_equal(received['y_pred'], expected_y_pred)
