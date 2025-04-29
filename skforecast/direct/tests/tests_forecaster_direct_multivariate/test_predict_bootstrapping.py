# Unit test predict_bootstrapping ForecasterDirectMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor

from skforecast.preprocessing import RollingFeatures
from skforecast.preprocessing import TimeSeriesDifferentiator
from skforecast.direct import ForecasterDirectMultiVariate

# Fixtures
from .fixtures_forecaster_direct_multivariate import series
from .fixtures_forecaster_direct_multivariate import exog
from .fixtures_forecaster_direct_multivariate import exog_predict
from .fixtures_forecaster_direct_multivariate import data  # to test results when using differentiation

transformer_exog = ColumnTransformer(
                       [('scale', StandardScaler(), ['exog_1']),
                        ('onehot', OneHotEncoder(), ['exog_2'])],
                       remainder = 'passthrough',
                       verbose_feature_names_out = False
                   )


def test_predict_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', lags=3, steps=3
    )

    err_msg = re.escape(
        "This Forecaster instance is not fitted yet. Call `fit` with "
        "appropriate arguments before using predict."
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict_bootstrapping(steps=5)


@pytest.mark.parametrize("use_binned_residuals", [True, False], 
                         ids=lambda binned: f'use_binned_residuals: {binned}')
def test_predict_bootstrapping_ValueError_when_out_sample_residuals_is_None(use_binned_residuals):
    """
    Test ValueError is raised when use_in_sample_residuals=False and
    forecaster.out_sample_residuals_ is None.
    """
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', lags=3, steps=2
    )
    forecaster.fit(series=series, store_in_sample_residuals=True)

    if use_binned_residuals:
        literal = "out_sample_residuals_by_bin_"
    else:
        literal = "out_sample_residuals_"

    err_msg = re.escape(
        f"`forecaster.{literal}` is either None or empty. Use "
        f"`use_in_sample_residuals = True` or the `set_out_sample_residuals()` "
        f"method before predicting."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(
            steps=1, use_in_sample_residuals=False, use_binned_residuals=use_binned_residuals
        )


@pytest.mark.parametrize("steps", [2, [1, 2], None], 
                         ids=lambda steps: f'steps: {steps}')
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer(steps):
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     steps              = 2,
                     level              = 'l1',
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog
                 )
    forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
    results = forecaster.predict_bootstrapping(
        steps=steps, exog=exog_predict, n_boot=4, 
        use_in_sample_residuals=True, use_binned_residuals=False
    )
    
    expected = pd.DataFrame(
                   data = np.array([[0.68727676, 0.8016235 , 0.4586416 , 0.69518909],
                                    [0.07063188, 0.60208793, 0.20809564, 0.3706782 ]]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    expected.insert(0, 'level', np.tile(['l1'], 2))
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_False_exog_and_transformer():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     steps              = 2,
                     level              = 'l1',
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog
                 )
    forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    forecaster.out_sample_residuals_by_bin_ = forecaster.in_sample_residuals_by_bin_
    results = forecaster.predict_bootstrapping(
        steps=2, exog=exog_predict, n_boot=4, 
        use_in_sample_residuals=False, use_binned_residuals=False
    )
    
    expected = pd.DataFrame(
                   data = np.array([[0.68727676, 0.8016235 , 0.4586416 , 0.69518909],
                                    [0.07063188, 0.60208793, 0.20809564, 0.3706782 ]]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    expected.insert(0, 'level', np.tile(['l1'], 2))
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_fixed():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals that are fixed.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     steps              = 2,
                     level              = 'l1',
                     lags               = 3,
                     transformer_series = None
                 )
    forecaster.fit(series=series, exog=exog['exog_1'], store_in_sample_residuals=True)
    forecaster.in_sample_residuals_ = {'l1': np.array([1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5])}
    results = forecaster.predict_bootstrapping(
        steps=2, exog=exog_predict['exog_1'], n_boot=4, 
        use_in_sample_residuals=True, use_binned_residuals=False
    )
    
    expected = pd.DataFrame(
                   data = np.array([[1.57457831, 5.57457831, 5.57457831, 1.57457831],
                                    [5.3777698 , 1.3777698 , 1.3777698 , 1.3777698 ]]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    expected.insert(0, 'level', np.tile(['l1'], 2))
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_regressor_is_LinearRegression_with_exog_and_differentiation_is_1_steps_1():
    """
    Test predict_bootstrapping output when using LinearRegression as regressor 
    and differentiation=1 and steps=1.
    """

    arr = data.to_numpy(copy=True)
    series_2 = pd.DataFrame(
        {'l1': arr,
         'l2': arr * 1.6},
        index=data.index
    )

    # Data differentiated
    differentiator = TimeSeriesDifferentiator(order=1)
    series_diff = pd.DataFrame(
        {'l1': differentiator.fit_transform(arr),
         'l2': differentiator.fit_transform(arr * 1.6)},
        index=data.index
    ).dropna()

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff = exog.iloc[1:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=1, lags=15, transformer_series=None
    )
    forecaster_1.fit(
        series=series_diff.loc[:end_train], exog=exog_diff.loc[:end_train], store_in_sample_residuals=True
    )
    boot_predictions_diff = forecaster_1.predict_bootstrapping(
        exog=exog_diff.loc[end_train:], n_boot=10, use_in_sample_residuals=True, use_binned_residuals=False
    )

    # Revert the differentiation
    last_value_train = series_2[['l1']].loc[:end_train].iloc[[-1]]
    boot_predictions_1 = boot_predictions_diff.drop(columns='level').copy()
    last_value_train = pd.DataFrame(
        np.full(shape=(1, 10), fill_value=last_value_train.values[0]), 
        columns = boot_predictions_1.columns, 
        index   = pd.date_range(last_value_train.index[0], periods=1)
    )
    boot_predictions_1 = pd.concat([last_value_train, boot_predictions_1])    
    boot_predictions_1 = boot_predictions_1.sort_index()
    boot_predictions_1 = boot_predictions_1.cumsum(axis=0).iloc[1:,]
    boot_predictions_1 = boot_predictions_1.asfreq('MS')
    boot_predictions_1.insert(0, 'level', np.tile(['l1'], 1))

    forecaster_2 = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=1, lags=15, transformer_series=None, differentiation=1
    )
    forecaster_2.fit(
        series=series_2.loc[:end_train], exog=exog.loc[:end_train], store_in_sample_residuals=True
    )
    boot_predictions_2 = forecaster_2.predict_bootstrapping(
        exog=exog_diff.loc[end_train:], n_boot=10, use_in_sample_residuals=True, use_binned_residuals=False
    )

    pd.testing.assert_frame_equal(boot_predictions_1, boot_predictions_2)


def test_predict_bootstrapping_output_when_regressor_is_LinearRegression_with_exog_and_differentiation_is_1_steps_10():
    """
    Test predict_bootstrapping output when using LinearRegression as regressor 
    and differentiation=1 and steps=10.
    """

    arr = data.to_numpy(copy=True)
    series_2 = pd.DataFrame(
        {'l1': arr,
         'l2': arr * 1.6},
        index=data.index
    )

    # Data differentiated
    differentiator = TimeSeriesDifferentiator(order=1)
    series_diff = pd.DataFrame(
        {'l1': differentiator.fit_transform(arr),
         'l2': differentiator.fit_transform(arr * 1.6)},
        index=data.index
    ).dropna()

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff = exog.iloc[1:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=10, lags=15, transformer_series=None
    )
    forecaster_1.fit(
        series=series_diff.loc[:end_train], exog=exog_diff.loc[:end_train], store_in_sample_residuals=True
    )
    boot_predictions_diff = forecaster_1.predict_bootstrapping(
        exog=exog_diff.loc[end_train:], n_boot=10, use_in_sample_residuals=True, use_binned_residuals=False
    )

    # Revert the differentiation
    last_value_train = series_2[['l1']].loc[:end_train].iloc[[-1]]
    boot_predictions_1 = boot_predictions_diff.drop(columns='level').copy()
    last_value_train = pd.DataFrame(
        np.full(shape=(1, 10), fill_value=last_value_train.values[0]), 
        columns = boot_predictions_1.columns, 
        index   = pd.date_range(last_value_train.index[0], periods=1)
    )
    boot_predictions_1 = pd.concat([last_value_train, boot_predictions_1])    
    boot_predictions_1 = boot_predictions_1.sort_index()
    boot_predictions_1 = boot_predictions_1.cumsum(axis=0).iloc[1:,]
    boot_predictions_1 = boot_predictions_1.asfreq('MS')
    boot_predictions_1.insert(0, 'level', np.tile(['l1'], 10))

    forecaster_2 = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=10, lags=15, transformer_series=None, differentiation=1
    )
    forecaster_2.fit(
        series=series_2.loc[:end_train], exog=exog.loc[:end_train], store_in_sample_residuals=True
    )
    boot_predictions_2 = forecaster_2.predict_bootstrapping(
        exog=exog_diff.loc[end_train:], n_boot=10, use_in_sample_residuals=True, use_binned_residuals=False
    )

    pd.testing.assert_frame_equal(boot_predictions_1, boot_predictions_2)


def test_predict_output_when_window_features_steps_1():
    """
    Test output of predict when regressor is LGBMRegressor and window features
    with steps=1.
    """

    rolling = RollingFeatures(stats=['mean', 'sum'], window_sizes=[3, 5])
    forecaster = ForecasterDirectMultiVariate(
        regressor=LGBMRegressor(verbose=-1, random_state=123), level='l1', 
        steps=1, lags=5, window_features=rolling
    )
    forecaster.fit(series=series, exog=exog['exog_1'], store_in_sample_residuals=True)
    predictions = forecaster.predict_bootstrapping(
        n_boot=10, exog=exog_predict['exog_1'], use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data    = np.array(
                                 [[0.60290414, 0.4125675 , 0.49927603, 0.64983046, 0.49009973,
                                   0.48971982, 0.40030064, 0.3430941 , 0.82287993, 0.67578235]]
                             ),
                   columns = [f"pred_boot_{i}" for i in range(10)],
                   index   = pd.RangeIndex(start=50, stop=51)
               )
    expected.insert(0, 'level', np.tile(['l1'], 1))
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_output_when_window_features_steps_10():
    """
    Test output of predict when regressor is LGBMRegressor and window features
    with steps=10.
    """

    rolling = RollingFeatures(stats=['mean', 'sum'], window_sizes=[3, 5])
    forecaster = ForecasterDirectMultiVariate(
        regressor=LGBMRegressor(verbose=-1, random_state=123), level='l1', 
        steps=10, lags=5, window_features=rolling
    )
    forecaster.fit(series=series, exog=exog['exog_1'], store_in_sample_residuals=True)
    predictions = forecaster.predict_bootstrapping(
        n_boot=10, exog=exog_predict['exog_1'], use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data    = np.array(
                                 [[0.34317802, 0.64776959, 0.44252774, 0.72244338, 0.17452841,
                                    0.42501556, 0.38875996, 0.43489265, 0.66887236, 0.50222644],
                                [0.34888897, 0.1813542 , 0.24348837, 0.71499064, 0.3770138 ,
                                    0.44426026, 0.42297623, 0.6403034 , 0.31816366, 0.25635781],
                                [0.37107296, 0.39877449, 0.43370117, 0.61102351, 0.84943179,
                                    0.83690427, 0.24017543, 0.72904971, 0.43536741, 0.35150872],
                                [0.58509345, 0.83877703, 0.7137504 , 0.57562876, 0.34400862,
                                    0.52405127, 0.30293129, 0.96777975, 0.72154961, 0.17572187],
                                [0.24338025, 0.5347562 , 0.61422834, 0.61575098, 0.66106327,
                                    0.86630916, 0.31941155, 0.50577174, 0.39804426, 0.41112617],
                                [0.0988748 , 0.31547649, 0.83027159, 0.42189523, 0.40316123,
                                    0.71344514, 0.227533  , 0.60577608, 0.41454097, 0.63051606],
                                [0.74188031, 0.60966097, 0.17545176, 0.41697845, 0.4308126 ,
                                    0.06283256, 0.92888472, 0.41697845, 0.48408656, 0.60389823],
                                [0.54283777, 0.32254116, 0.44564736, 0.48303426, 0.30078917,
                                    0.42410373, 0.07449023, 0.32756542, 0.54283777, 0.54283777],
                                [0.33621667, 0.07766116, 0.51371272, 0.51509444, 0.61903383,
                                    0.64692848, 0.37116152, 0.3145088 , 0.70908819, 0.42496032],
                                [0.91812915, 0.47396699, 0.63571333, 0.34410137, 0.65201567,
                                    0.18341508, 0.46122504, 0.43951603, 0.33700121, 0.51837492]]
                             ),
                   columns = [f"pred_boot_{i}" for i in range(10)],
                   index   = pd.RangeIndex(start=50, stop=60)
               )
    expected.insert(0, 'level', np.tile(['l1'], 10))
    
    pd.testing.assert_frame_equal(predictions, expected)
