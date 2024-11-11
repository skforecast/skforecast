# Unit test predict_bootstrapping ForecasterRecursive
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

from skforecast.preprocessing import RollingFeatures
from skforecast.preprocessing import TimeSeriesDifferentiator
from skforecast.recursive import ForecasterRecursive

# Fixtures
from .fixtures_forecaster_recursive import y
from .fixtures_forecaster_recursive import exog
from .fixtures_forecaster_recursive import exog_predict
from .fixtures_forecaster_recursive import data  # to test results when using differentiation


def test_predict_bootstrapping_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)

    err_msg = re.escape(
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1)


def test_predict_bootstrapping_ValueError_when_out_sample_residuals_is_None():
    """
    Test ValueError is raised when use_in_sample_residuals=False and
    forecaster.out_sample_residuals_ is None.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))

    err_msg = re.escape(
        ("`forecaster.out_sample_residuals_` is `None`. Use "
         "`use_in_sample_residuals=True` or the "
         "`set_out_sample_residuals()` method before predicting.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1, use_in_sample_residuals=False)


def test_predict_bootstrapping_ValueError_when_out_sample_residuals_by_bin_is_None():
    """
    Test ValueError is raised when use_in_sample_residuals=False and
    forecaster.out_sample_residuals_by_bin_ is None.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))

    err_msg = re.escape(
        ("`forecaster.out_sample_residuals_by_bin_` is `None`. Use "
         "`use_in_sample_residuals=True` or the "
         "`set_out_sample_residuals()` method before predicting.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1, use_in_sample_residuals=False,
                                         use_binned_residuals=True)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_1_in_sample_residuals_is_True():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    1 step ahead is predicted with exog using in-sample residuals.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3, binner_kwargs={'n_bins': 15})
    forecaster.fit(y=y, exog=exog)
    results = forecaster.predict_bootstrapping(
        steps=1, n_boot=4, exog=exog_predict, use_in_sample_residuals=True
    )

    expected = pd.DataFrame(
                   data    = np.array([[0.76205175, 0.52687879, 0.40505505, 0.77278459]]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=51)
               )

    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_2_in_sample_residuals_is_True():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    2 steps ahead are predicted with exog using in-sample residuals.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3, binner_kwargs={'n_bins': 15})
    forecaster.fit(y=y, exog=exog)
    results = forecaster.predict_bootstrapping(
        steps=2, n_boot=4, exog=exog_predict, use_in_sample_residuals=True
    )

    expected = pd.DataFrame(
                   data    = np.array(
                                 [[0.76205175, 0.52687879, 0.40505505, 0.77278459],
                                  [0.86040207, 0.16150649, 0.11656985, 0.39176265]]
                             ),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
               )

    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_1_in_sample_residuals_is_False():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    1 step ahead is predicted with exog using out-sample residuals.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3, binner_kwargs={'n_bins': 15})
    forecaster.fit(y=y, exog=exog)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    results = forecaster.predict_bootstrapping(
        steps=1, n_boot=4, exog=exog_predict, use_in_sample_residuals=False
    )

    expected = pd.DataFrame(
                   data    = np.array([[0.76205175, 0.52687879, 0.40505505, 0.77278459]]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=51)
               )

    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_2_in_sample_residuals_is_False():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    2 steps ahead are predicted with exog using out-sample residuals.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3, binner_kwargs={'n_bins': 15})
    forecaster.fit(y=y, exog=exog)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    results = forecaster.predict_bootstrapping(
        steps=2, n_boot=4, exog=exog_predict, use_in_sample_residuals=False
    )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.76205175, 0.52687879, 0.40505505, 0.77278459],
                                       [0.86040207, 0.16150649, 0.11656985, 0.39176265]]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
               )

    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer_fixed_to_zero():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included, both
    inputs are transformed and in-sample residuals are fixed to 0.
    """
    forecaster = ForecasterRecursive(
                     regressor        = LinearRegression(),
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                 )
    forecaster.fit(y=y, exog=exog)
    forecaster.in_sample_residuals_ = np.full_like(forecaster.in_sample_residuals_, fill_value=0)
    results = forecaster.predict_bootstrapping(
        steps=2, exog=exog_predict, n_boot=4, use_in_sample_residuals=True
    )
    
    expected = pd.DataFrame(
                   data = np.array([[0.67998879, 0.67998879, 0.67998879, 0.67998879],
                                    [0.46135022, 0.46135022, 0.46135022, 0.46135022]]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed.
    """

    forecaster = ForecasterRecursive(
                     regressor        = LinearRegression(),
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                     binner_kwargs    = {'n_bins': 15}
                 )
    forecaster.fit(y=y, exog=exog)
    results = forecaster.predict_bootstrapping(
        steps=2, exog=exog_predict, n_boot=4, use_in_sample_residuals=True
    )
    
    expected = pd.DataFrame(
                   data    = np.array(
                                 [[0.76205175, 0.52687879, 0.40505505, 0.77278459],
                                  [0.86040207, 0.16150649, 0.11656985, 0.39176265]]
                             ),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
                )

    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_and_differentiation_is_1():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    differentiation is 1.
    """
    # Data differentiated
    diferenciator = TimeSeriesDifferentiator(order=1)
    data_diff = diferenciator.fit_transform(data.to_numpy())
    data_diff = pd.Series(data_diff, index=data.index).dropna()

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff = exog.iloc[1:]
    end_train = '2003-03-01 23:59:00'
    steps = len(data.loc[end_train:])

    forecaster_1 = ForecasterRecursive(regressor=LinearRegression(), lags=15)
    forecaster_1.fit(y=data_diff.loc[:end_train], exog=exog_diff.loc[:end_train])
    boot_predictions_diff = forecaster_1.predict_bootstrapping(
                                steps=steps,
                                exog=exog_diff.loc[end_train:],
                                n_boot=10
                            )
    last_value_train = data.loc[:end_train].iloc[[-1]]
    boot_predictions_1 = boot_predictions_diff.copy()
    boot_predictions_1.loc[last_value_train.index[0]] = last_value_train.values[0]
    boot_predictions_1 = boot_predictions_1.sort_index()
    boot_predictions_1 = boot_predictions_1.cumsum(axis=0).iloc[1:,]
    boot_predictions_1 = boot_predictions_1.asfreq('MS')

    forecaster_2 = ForecasterRecursive(regressor=LinearRegression(), lags=15, differentiation=1)
    forecaster_2.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    boot_predictions_2 = forecaster_2.predict_bootstrapping(
                            steps  = steps,
                            exog   = exog_diff.loc[end_train:],
                            n_boot = 10
                        )

    pd.testing.assert_frame_equal(boot_predictions_1, boot_predictions_2)


def test_predict_bootstrapping_output_when_window_features():
    """
    Test output of predict_bootstrapping when regressor is LGBMRegressor and
    4 steps ahead are predicted with exog and window features using in-sample residuals.
    """
    y_datetime = y.copy()
    y_datetime.index = pd.date_range(start="2001-01-01", periods=len(y), freq="D")
    exog_datetime = exog.copy()
    exog_datetime.index = pd.date_range(start="2001-01-01", periods=len(exog), freq="D")
    exog_predict_datetime = exog_predict.copy()
    exog_predict_datetime.index = pd.date_range(
        start="2001-02-20", periods=len(exog_predict), freq="D"
    )
    rolling = RollingFeatures(stats=["mean", "sum"], window_sizes=[3, 5])
    forecaster = ForecasterRecursive(
        LGBMRegressor(verbose=-1, random_state=123), lags=3, window_features=rolling
    )
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    results = forecaster.predict_bootstrapping(
        steps=4, n_boot=10, exog=exog_predict_datetime, use_in_sample_residuals=True
    )

    expected = pd.DataFrame(
        {
            "pred_boot_0": [
                0.6186600380569245,
                0.2647569741581052,
                0.4005398117513452,
                0.39037942037849277,
            ],
            "pred_boot_1": [
                0.8969725592689648,
                0.6723022197318752,
                0.32295891,
                0.0680509514238431,
            ],
            "pred_boot_2": [
                0.14325578784512172,
                0.18926364363504294,
                0.6157758549401303,
                0.86630916,
            ],
            "pred_boot_3": [
                0.18128852194384876,
                0.4512437336944209,
                0.15682665494013032,
                0.5219030808497911,
            ],
            "pred_boot_4": [
                0.868465651152662,
                0.34317802,
                0.43966008751761915,
                0.4724785875176192,
            ],
            "pred_boot_5": [
                0.5502084895370893,
                0.6900186013644649,
                0.46585817973187515,
                0.43966008751761915,
            ],
            "pred_boot_6": [
                0.2587067892095866,
                0.4313427657899213,
                0.5110143957305437,
                0.4622747804629107,
            ],
            "pred_boot_7": [
                0.5167063442694563,
                0.6723022197318752,
                0.5225711016919676,
                0.5852571421548783,
            ],
            "pred_boot_8": [
                0.2923943620032269,
                0.20674374336691814,
                0.5603167728446299,
                0.5189877024230032,
            ],
            "pred_boot_9": [
                0.42315363705470854,
                0.8408282633075403,
                0.4724785875176192,
                0.28131970830803243,
            ],
        },
        index=pd.date_range(start="2001-02-20", periods=4, freq="D"),
    )

    pd.testing.assert_frame_equal(expected, results)
