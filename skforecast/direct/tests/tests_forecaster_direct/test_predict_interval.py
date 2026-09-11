# Unit test predict_interval ForecasterDirect
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from skforecast.direct import ForecasterDirect
from skforecast.exceptions import IgnoredArgumentWarning

# Fixtures
from .fixtures_forecaster_direct import y
from .fixtures_forecaster_direct import exog
from .fixtures_forecaster_direct import exog_predict
from .fixtures_forecaster_direct import y_intermittent


def test_check_interval_ValueError_when_method_is_not_valid_method():
    """
    Check ValueError is raised when `method` is not 'bootstrapping' or 'conformal'.
    """
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=2, steps=3
    )
    forecaster.fit(y=pd.Series(np.arange(10)), store_in_sample_residuals=True)

    method = 'not_valid_method'
    err_msg = re.escape(
        f"Invalid `method` '{method}'. Choose 'bootstrapping' or 'conformal'."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(steps=1, method=method)


@pytest.mark.parametrize("interval", 
                         [0.90, [0.05, 0.95], (0.05, 0.95)], 
                         ids = lambda value: f'interval: {value}')
def test_predict_interval_output_when_in_sample_residuals_exog_and_transformer(interval):
    """
    Test output of predict_interval when estimator is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed.
    """

    forecaster = ForecasterDirect(
                     estimator        = LinearRegression(),
                     steps            = 2,
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                 )
    forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
    n_boot = 250
    results = forecaster.predict_interval(
                      steps                   = 2,
                      interval                = interval,
                      exog                    = exog_predict,
                      n_boot                  = n_boot,
                      use_in_sample_residuals = True,
                      use_binned_residuals    = False
                  )
    
    expected = pd.DataFrame(
                   data    = np.array([
                                [0.67523588, 0.29721203, 1.07760213],
                                [0.38024988, 0.00222603, 0.78098289]
                            ]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_False_exog_and_transformer():
    """
    Test output of predict_interval when estimator is LinearRegression,
    2 steps are predicted, using out-sample residuals, exog is included and both
    inputs are transformed.
    """

    forecaster = ForecasterDirect(
                     estimator        = LinearRegression(),
                     steps            = 2,
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                 )
    forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    results = forecaster.predict_interval(
                  steps                   = 2,
                  interval                = (0.05, 0.95),
                  exog                    = exog_predict,
                  n_boot                  = 250,
                  use_in_sample_residuals = False,
                  use_binned_residuals    = False
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.67523588, 0.29721203, 1.07760213],
                                        [0.38024988, 0.00222603, 0.78098289]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )

    pd.testing.assert_frame_equal(expected, results)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_5_in_sample_residuals_is_True_binned_residuals_is_True():
    """
    Test output when estimator is LinearRegression 5 step ahead is predicted
    using in sample binned residuals.
    """
    forecaster = ForecasterDirect(
                     estimator     = LinearRegression(),
                     steps         = 5,
                     lags          = 3,
                     transformer_y = StandardScaler()
                 )
    forecaster.fit(y=y, store_in_sample_residuals=True)
    results = forecaster.predict_interval(
        steps=5, interval=(0.05, 0.95), use_in_sample_residuals=True, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                    data    = np.array(
                                [[0.51883519, 0.11786323, 0.96698339],
                                 [0.4584716 , 0.04248806, 0.85897968],
                                 [0.39962743, 0.16612305, 0.91127372],
                                 [0.40452904, 0.17102467, 0.85708924],
                                 [0.41534557, 0.07605488, 0.92699186]]
                            ),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index   = pd.RangeIndex(start=50, stop=55, step=1)
                )

    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_5_in_sample_residuals_is_False_binned_residuals_is_True():
    """
    Test output when estimator is LinearRegression, steps=5, use_in_sample_residuals=False,
    binned_residuals=True.
    """
    forecaster = ForecasterDirect(
                     estimator        = LinearRegression(),
                     steps            = 5,
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                 )
    forecaster.fit(y=y, store_in_sample_residuals=True)
    forecaster.out_sample_residuals_by_bin_ = forecaster.in_sample_residuals_by_bin_
    results = forecaster.predict_interval(
        steps=5, interval=(0.05, 0.95), use_in_sample_residuals=False, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                    data    = np.array(
                                [[0.51883519, 0.11786323, 0.96698339],
                                 [0.4584716 , 0.04248806, 0.85897968],
                                 [0.39962743, 0.16612305, 0.91127372],
                                 [0.40452904, 0.17102467, 0.85708924],
                                 [0.41534557, 0.07605488, 0.92699186]]
                            ),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index   = pd.RangeIndex(start=50, stop=55, step=1)
                )
    
    pd.testing.assert_frame_equal(results, expected)


@pytest.mark.parametrize("interval", 
                         [0.95, (0.025, 0.975)], 
                         ids = lambda value: f'interval: {value}')
def test_predict_interval_conformal_output_when_transform_y(interval):
    """
    Test predict output when using LinearRegression as estimator and StandardScaler
    and conformal prediction.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4 ,  0.49,
                       0.67,  0.54, -0.17,  0.54,  1.49, -2.26, -0.41, -0.64, -0.8 ,
                      -0.61, -0.88])
        )
    forecaster = ForecasterDirect(
                     estimator     = LinearRegression(),
                     steps         = 3,
                     lags          = 3,
                     transformer_y = StandardScaler()
                 )
    forecaster.fit(y=y, store_in_sample_residuals=False)
    forecaster.set_in_sample_residuals(y=y)
    results = forecaster.predict_interval(
        steps=3, method='conformal', interval=interval, 
        use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data = np.array([
                            [-0.07720596, -2.17165565,  2.01724372],
                            [-0.54638907, -2.64083876,  1.54806061],
                            [-0.08892596, -2.18337565,  2.00552372]]),
                   index = pd.RangeIndex(start=20, stop=23, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    
    pd.testing.assert_frame_equal(results, expected)


@pytest.mark.parametrize("interval", 
                         [0.95, (0.025, 0.975)], 
                         ids = lambda value: f'interval: {value}')
def test_predict_interval_conformal_output_when_binned_residuals(interval):
    """
    Test predict output when using LinearRegression as estimator and StandardScaler
    and conformal prediction with binned residuals.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4 ,  0.49,
                       0.67,  0.54, -0.17,  0.54,  1.49, -2.26, -0.41, -0.64, -0.8 ,
                      -0.61, -0.88])
        )
    forecaster = ForecasterDirect(
                     estimator     = LinearRegression(),
                     steps         = 3,
                     lags          = 3,
                     transformer_y = StandardScaler()
                 )
    forecaster.fit(y=y, store_in_sample_residuals=True)
    results = forecaster.predict_interval(
        steps=3, method='conformal', interval=interval, 
        use_in_sample_residuals=True, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [-0.07720596, -1.96803865,  1.81362673],
                              [-0.54638907, -3.1153822 ,  2.02260406],
                              [-0.08892596, -1.97975865,  1.80190673]]),
                   index = pd.RangeIndex(start=20, stop=23, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_bootstrapping_binned_residuals_when_binner_reduces_n_bins():
    """
    Test predict_interval with method 'bootstrapping' and binned residuals when
    the predictions are so concentrated that the binner has to reduce the number
    of bins. Every bin id returned by the binner must have residuals associated
    with it.
    """
    forecaster = ForecasterDirect(estimator=LinearRegression(), lags=5, steps=3)
    warn_msg = re.escape(
        "The number of bins has been reduced from 10 to 9 due to empty bins. "
        "This happens when "
        "the values used to compute the edges of the bins are highly "
        "concentrated or contain many repeated values.",
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        forecaster.fit(y=y_intermittent, store_in_sample_residuals=True)
    results = forecaster.predict_interval(
        steps=3, method='bootstrapping', interval=0.8, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [ 0.56717387, -0.32346175,  3.06027443],
                              [-2.39178993, -3.28242556,  0.10131063],
                              [ 9.61191402,  2.46656439, 23.2188022 ]]),
                   index = pd.date_range(start='2020-03-21', periods=3, freq='D'),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )

    assert forecaster.binner.n_bins_ == 9
    assert sorted(forecaster.in_sample_residuals_by_bin_) == list(range(9))
    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_conformal_binned_residuals_when_binner_reduces_n_bins():
    """
    Test predict_interval with method 'conformal' and binned residuals when the
    predictions are so concentrated that the binner has to reduce the number of
    bins. Every bin id returned by the binner must have residuals associated
    with it.
    """
    forecaster = ForecasterDirect(estimator=LinearRegression(), lags=5, steps=3)
    warn_msg = re.escape(
        "The number of bins has been reduced from 10 to 9 due to empty bins. "
        "This happens when "
        "the values used to compute the edges of the bins are highly "
        "concentrated or contain many repeated values.",
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        forecaster.fit(y=y_intermittent, store_in_sample_residuals=True)
    results = forecaster.predict_interval(
        steps=3, method='conformal', interval=0.8, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [ 0.56717387, -0.96324417,  2.09759192],
                              [-2.39178993, -3.92220797, -0.86137189],
                              [ 9.61191402,  0.44226262, 18.78156542]]),
                   index = pd.date_range(start='2020-03-21', periods=3, freq='D'),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )

    assert forecaster.binner.n_bins_ == 9
    assert sorted(forecaster.in_sample_residuals_by_bin_) == list(range(9))
    pd.testing.assert_frame_equal(results, expected)

