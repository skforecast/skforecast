# Unit test _predict_and_calculate_metrics_one_step_ahead_multiseries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection import backtesting_forecaster_multiseries
from skforecast.model_selection._utils import _predict_and_calculate_metrics_one_step_ahead_multiseries
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


def test_predict_and_calculate_metrics_one_step_ahead_multiseries_input_types():
    """
    Check if function raises errors when input parameters have wrong types.
    """

    # Mock inputs
    forecaster = ForecasterRecursiveMultiSeries(regressor=Ridge(random_state=678), lags=3)
    
    initial_train_size = 927
    metrics = ['mean_absolute_error']
    levels = ['item_1', 'item_2', 'item_3']
    add_aggregated_metric = True

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

    # Test invalid type for series
    err_msg = re.escape(
        "`series` must be a pandas DataFrame or a dictionary of pandas DataFrames."
    )
    with pytest.raises(TypeError, match=err_msg):
        _predict_and_calculate_metrics_one_step_ahead_multiseries(
            forecaster, "invalid_series_type", X_train, y_train, X_test, y_test, 
            X_train_encoding, X_test_encoding, levels, metrics, add_aggregated_metric
        )

    # Test invalid type for X_train
    X_train_invalid = "invalid_X_train_type"
    err_msg = re.escape(
        f"`X_train` must be a pandas DataFrame. Got: {type(X_train_invalid)}"
    )
    with pytest.raises(TypeError, match=err_msg):
        _predict_and_calculate_metrics_one_step_ahead_multiseries(
            forecaster, series, X_train_invalid, y_train, X_test, y_test, 
            X_train_encoding, X_test_encoding, levels, metrics, add_aggregated_metric
        )

    # Test invalid type for y_train
    y_train_invalid = "invalid_y_train_type"
    err_msg = re.escape(
        f"`y_train` must be a pandas Series or a dictionary of pandas Series. "
        f"Got: {type(y_train_invalid)}"
    )  
    with pytest.raises(TypeError, match=err_msg):
        _predict_and_calculate_metrics_one_step_ahead_multiseries(
            forecaster, series, X_train, y_train_invalid, X_test, y_test, 
            X_train_encoding, X_test_encoding, levels, metrics, add_aggregated_metric
        )

    # Test invalid type for X_test
    X_test_invalid = "invalid_X_test_type"
    err_msg = re.escape(
        f"`X_test` must be a pandas DataFrame. Got: {type(X_test_invalid)}"
    )
    with pytest.raises(TypeError, match=err_msg):
        _predict_and_calculate_metrics_one_step_ahead_multiseries(
            forecaster, series, X_train, y_train, X_test_invalid, y_test, 
            X_train_encoding, X_test_encoding, levels, metrics, add_aggregated_metric
        )

    # Test invalid type for y_test
    y_test_invalid = "invalid_y_test_type"
    err_msg = re.escape(
        f"`y_test` must be a pandas Series or a dictionary of pandas Series. "
        f"Got: {type(y_test_invalid)}"
    )
    with pytest.raises(TypeError, match=err_msg):
        _predict_and_calculate_metrics_one_step_ahead_multiseries(
            forecaster, series, X_train, y_train, X_test, y_test_invalid, 
            X_train_encoding, X_test_encoding, levels, metrics, add_aggregated_metric
        )

    # Test invalid type for X_train_encoding
    X_train_encoding_invalid = "invalid_X_train_encoding_type"
    err_msg = re.escape(
        f"`X_train_encoding` must be a pandas Series. Got: {type(X_train_encoding_invalid)}"
    )
    with pytest.raises(TypeError, match=err_msg):
        _predict_and_calculate_metrics_one_step_ahead_multiseries(
            forecaster, series, X_train, y_train, X_test, y_test, 
            X_train_encoding_invalid, X_test_encoding, levels, metrics, add_aggregated_metric
        )

    # Test invalid type for X_test_encoding
    X_test_encoding_invalid = "invalid_X_test_encoding_type"
    err_msg = re.escape(
        f"`X_test_encoding` must be a pandas Series. Got: {type(X_test_encoding_invalid)}"
    )
    with pytest.raises(TypeError, match=err_msg):
        _predict_and_calculate_metrics_one_step_ahead_multiseries(
            forecaster, series, X_train, y_train, X_test, y_test, 
            X_train_encoding, X_test_encoding_invalid, levels, metrics, add_aggregated_metric
        )

    # Test invalid type for levels
    levels_invalid = "invalid_levels_type"
    err_msg = re.escape(
        f"`levels` must be a list. Got: {type(levels_invalid)}"
    )
    with pytest.raises(TypeError, match=err_msg):
        _predict_and_calculate_metrics_one_step_ahead_multiseries(
            forecaster, series, X_train, y_train, X_test, y_test, 
            X_train_encoding, X_test_encoding, levels_invalid, metrics, add_aggregated_metric
        )

    # Test invalid type for metrics
    metrics_invalid = "invalid_metrics_type"
    err_msg = re.escape(
        f"`metrics` must be a list. Got: {type(metrics_invalid)}"
    )
    with pytest.raises(TypeError, match=err_msg):
        _predict_and_calculate_metrics_one_step_ahead_multiseries(
            forecaster, series, X_train, y_train, X_test, y_test, 
            X_train_encoding, X_test_encoding, levels, metrics_invalid, add_aggregated_metric
        )

    # Test invalid type for add_aggregated_metric
    add_aggregated_metric_invalid = "invalid_add_aggregated_metric_type"
    err_msg = re.escape(
        f"`add_aggregated_metric` must be a boolean. Got: {type(add_aggregated_metric_invalid)}"
    )
    with pytest.raises(TypeError, match=err_msg):
        _predict_and_calculate_metrics_one_step_ahead_multiseries(
            forecaster, series, X_train, y_train, X_test, y_test, 
            X_train_encoding, X_test_encoding, levels, metrics, add_aggregated_metric_invalid
        )


@pytest.mark.parametrize(
    "forecaster",
    [
        ForecasterRecursiveMultiSeries(
            regressor=Ridge(random_state=678), lags=3, forecaster_id='multiseries_no_transformer'
        ),
        ForecasterDirectMultiVariate(
            regressor=Ridge(random_state=678), steps=1, level='item_1', lags=3, transformer_series=None,
            forecaster_id='multivariate_no_transformer'
        ),
        ForecasterRecursiveMultiSeries(
            regressor=Ridge(random_state=678),
            lags=3,
            transformer_series=StandardScaler(),
            transformer_exog=StandardScaler(),
            forecaster_id='multiseries_transformer'
        ),
        ForecasterDirectMultiVariate(
            regressor=Ridge(random_state=678),
            level='item_1',
            lags=3,
            steps=1,
            transformer_series=StandardScaler(),
            transformer_exog=StandardScaler(),
            forecaster_id='multivariate_transformer'
        )
    ],
ids=lambda forecaster: f'forecaster: {forecaster.forecaster_id}')
def test_predict_and_calculate_metrics_one_step_ahead_multiseries_output_equivalence_to_backtesting(forecaster):
    """
    Test that the output of _predict_and_calculate_metrics_one_step_ahead_multiseries is equivalent to
    the output of backtesting_forecaster_multiseries when steps=1 and refit=False.

    **Results are not equivalent if differentiation is included**
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
             differentiation    = forecaster.differentiation_max
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
        X_test_encoding
    ) = forecaster._train_test_split_one_step_ahead(
            series             = series,
            exog               = exog,
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
        add_aggregated_metric = True
    )

    pd.testing.assert_frame_equal(metrics_one_step_ahead, metrics_backtesting)
    pd.testing.assert_frame_equal(pred_one_step_ahead, pred_backtesting)


@pytest.mark.parametrize("differentiation", 
    [1, {'item_1': 1, 'item_2': 1, 'item_3': 1, '_unknown_level': 1}], 
     ids = lambda diff: f'differentiation: {diff}')
def test_predict_and_calculate_metrics_one_step_ahead_multiseries_output_ForecasterRecursiveMultiSeries_differentiation(differentiation):
    """
    Test that the output of _predict_and_calculate_metrics_one_step_ahead_multiseries 
    when ForecasterRecursiveMultiSeries is used with differentiation.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = Ridge(random_state=678),
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
        X_test_encoding
    ) = forecaster._train_test_split_one_step_ahead(
            series             = series,
            exog               = exog,
            initial_train_size = initial_train_size,
        )

    results = _predict_and_calculate_metrics_one_step_ahead_multiseries(
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
        add_aggregated_metric = True
    )

    expected = (
        pd.DataFrame(
            {'levels': ['item_1',
                        'item_2',
                        'item_3',
                        'average',
                        'weighted_average',
                        'pooling'],
            'mean_absolute_error': [4.929999429969281,
                                    10.648823118144193,
                                    12.779424665286516,
                                    9.452749071133331,
                                    9.452749071133331,
                                    9.452749071133331],
            'mean_absolute_percentage_error': [0.24360856584323834,
                                                0.6270921373681089,
                                                0.7060429773403681,
                                                0.5255812268505718,
                                                0.5255812268505717,
                                                0.5255812268505717],
            'mean_absolute_scaled_error': [3.2251721028626883,
                                            4.520621275575221,
                                            3.441691599998732,
                                            3.7291616594788803,
                                            3.729161659478881,
                                            3.732657654164705]}
        ),
        pd.DataFrame(
            data = np.array([[25.6059407 , 10.6092054 , 11.74557632],
                             [24.62838843, 10.0671194 , 12.22932432],
                             [24.10287379,  9.51212108, 10.56435278],
                             [23.78506633,  9.17812301,  8.33429508],
                             [23.58213459,  8.57805035,  7.60062965],
                             [24.99011875,  9.57489282,  9.87305383],
                             [26.15161932, 10.22090842, 10.90470084],
                             [25.69506167, 10.94733743, 11.52785388],
                             [24.41084617, 11.46312403, 11.96414091],
                             [23.53329426, 11.05443539, 11.65551743]]),
            columns = ['item_1', 'item_2', 'item_3'],
            index = pd.date_range(start='2014-07-16', periods=10, freq='D')
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_frame_equal(results[1].head(10), expected[1])


def test_predict_and_calculate_metrics_one_step_ahead_multiseries_output_ForecasterRecursiveMultiSeries_differentiation_dict():
    """
    Test that the output of _predict_and_calculate_metrics_one_step_ahead_multiseries 
    when ForecasterRecursiveMultiSeries is used with differentiation as dict.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = Ridge(random_state=678),
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
        X_test_encoding
    ) = forecaster._train_test_split_one_step_ahead(
            series             = series,
            exog               = exog,
            initial_train_size = initial_train_size,
        )

    results = _predict_and_calculate_metrics_one_step_ahead_multiseries(
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
        add_aggregated_metric = True
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
                    0: 3.3727768904389137,
                    1: 3.2454861506987474,
                    2: 125.29749202738702,
                    3: 43.97191835617489,
                    4: 43.971918356174896,
                    5: 43.9719183561749,
                },
                "mean_absolute_percentage_error": {
                    0: 0.16768637909858158,
                    1: 0.23902627227111134,
                    2: 7.127463509732804,
                    3: 2.5113920537008325,
                    4: 2.5113920537008325,
                    5: 2.511392053700832,
                },
                "mean_absolute_scaled_error": {
                    0: 2.204480978912765,
                    1: 1.3765145512362387,
                    2: 33.712615977712886,
                    3: 12.431203835953964,
                    4: 12.431203835953964,
                    5: 17.347393898836998,
                },
            }
        ),
        pd.DataFrame(
            data = np.array([[25.86570499, 15.18029154,  8.97046675],
                             [26.32581267, 15.07864711,  8.59274604],
                             [26.71670307, 14.60073761,  7.7959118 ],
                             [26.03183899, 14.55994232,  7.25551897],
                             [25.44757003, 13.60961291,  5.69205946],
                             [26.29183264, 16.14084149,  3.01541324],
                             [25.96578123, 16.13231006,  2.10568015],
                             [25.6248342 , 15.46833359,  2.92044047],
                             [26.28049127, 14.83278091,  2.2656061 ],
                             [26.71633178, 14.01618597,  1.79968493]]),
            columns = ['item_1', 'item_2', 'item_3'],
            index = pd.date_range(start='2014-07-16', periods=10, freq='D')
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_frame_equal(results[1].head(10), expected[1])


def test_predict_and_calculate_metrics_one_step_ahead_multiseries_output_ForecasterDirectMultiVariate_differentiation():
    """
    Test that the output of _predict_and_calculate_metrics_one_step_ahead_multiseries 
    when ForecasterDirectMultiVariate is used with differentiation.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = Ridge(random_state=678),
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
        X_test_encoding
    ) = forecaster._train_test_split_one_step_ahead(
            series             = series,
            exog               = exog,
            initial_train_size = initial_train_size,
        )

    results = _predict_and_calculate_metrics_one_step_ahead_multiseries(
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
        add_aggregated_metric = True
    )

    expected = (
        pd.DataFrame(
            {'levels': ['item_1'],
             'mean_absolute_error': [3.246937481200051],
             'mean_absolute_percentage_error': [0.1604318346564821],
             'mean_absolute_scaled_error': [2.1241244208766368]}
        ),
        pd.DataFrame(
            data = np.array([[26.77436694],
                             [26.12768984],
                             [25.40947475],
                             [24.67063314],
                             [22.50260461],
                             [24.46675686],
                             [26.08019603],
                             [26.80716849],
                             [26.15049678],
                             [25.63937881]]),
            columns = ['item_1'],
            index = pd.date_range(start='2014-07-16', periods=10, freq='D')
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_frame_equal(results[1].head(10), expected[1])


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
            forecaster_id      = 'multiseries_lgbm'
        )
    ],
ids=lambda forecaster: f'forecaster: {forecaster.forecaster_id}')
def test_predict_and_calculate_metrics_one_step_ahead_multiseries_output_equivalence_to_backtesting_when_series_is_dict(forecaster):
    """
    Test that the output of _predict_and_calculate_metrics_one_step_ahead_multiseries is
    equivalent to the output of backtesting_forecaster_multiseries when steps=1 and
    refit=False. Using series and exog as dictionaries.
    Results are not equivalent if diferentiation is included.
    ForecasterMultiVariate is not included because it is not possible to use dictionaries as input.
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
        series=series_dict,
        exog=exog_dict,
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
        X_test_encoding
    ) = forecaster._train_test_split_one_step_ahead(
            series             = series_dict,
            exog               = exog_dict,
            initial_train_size = initial_train_size,
        )
    
    metrics_one_step_ahead, pred_one_step_ahead = _predict_and_calculate_metrics_one_step_ahead_multiseries(
        forecaster=forecaster,
        series=series_dict,
        X_train = X_train,
        y_train= y_train,
        X_test = X_test,
        y_test = y_test,
        X_train_encoding = X_train_encoding,
        X_test_encoding = X_test_encoding,
        levels = levels,
        metrics = metrics,
        add_aggregated_metric = True
    )

    pd.testing.assert_frame_equal(metrics_one_step_ahead, metrics_backtesting)
    pd.testing.assert_frame_equal(pred_one_step_ahead, pred_backtesting)
