# Unit test predict_interval ForecasterDirectMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.exceptions import IgnoredArgumentWarning

# Fixtures
from .fixtures_forecaster_direct_multivariate import series
from .fixtures_forecaster_direct_multivariate import exog
from .fixtures_forecaster_direct_multivariate import exog_predict
from .fixtures_forecaster_direct_multivariate import series_intermittent

transformer_exog = ColumnTransformer(
                       [('scale', StandardScaler(), ['exog_1']),
                        ('onehot', OneHotEncoder(), ['exog_2'])],
                       remainder = 'passthrough',
                       verbose_feature_names_out = False
                   )


def test_check_interval_ValueError_when_method_is_not_valid_method():
    """
    Check ValueError is raised when `method` is not 'bootstrapping' or 'conformal'.
    """
    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l1', steps=2, lags=3
    )
    forecaster.fit(series=series, store_in_sample_residuals=True)

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
    forecaster = ForecasterDirectMultiVariate(
                     estimator          = LinearRegression(),
                     level              = 'l1',
                     steps              = 2,
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog
                 )
    forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
    n_boot = 250
    results = forecaster.predict_interval(
                    steps                   = 2,
                    exog                    = exog_predict,
                    method                  = 'bootstrapping',
                    interval                = interval,
                    n_boot                  = n_boot,
                    use_in_sample_residuals = True,
                    use_binned_residuals    = False
                )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.61820497, 0.27569583, 0.94783844],
                                        [0.41314101, 0.06233635, 0.74277449]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    expected.insert(0, 'level', np.tile(['l1'], 2))
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_False_exog_and_transformer():
    """
    Test output of predict_interval when estimator is LinearRegression,
    2 steps are predicted, using out-sample residuals, exog is included and both
    inputs are transformed.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator          = LinearRegression(),
                     level              = 'l1',
                     steps              = 2,
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog
                 )
    forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    forecaster.out_sample_residuals_by_bin_ = forecaster.in_sample_residuals_by_bin_    
    n_boot = 250
    results = forecaster.predict_interval(
                    steps                   = 2,
                    exog                    = exog_predict,
                    method                  = 'bootstrapping',
                    interval                = [0.05, 0.95],
                    n_boot                  = n_boot,
                    use_in_sample_residuals = False,
                    use_binned_residuals    = False
                )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.61820497, 0.27569583, 0.94783844],
                                        [0.41314101, 0.06233635, 0.74277449]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    expected.insert(0, 'level', np.tile(['l1'], 2))

    pd.testing.assert_frame_equal(expected, results)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_5_in_sample_residuals_is_True_binned_residuals_is_True():
    """
    Test output when estimator is LinearRegression 5 step ahead is predicted
    using in sample binned residuals.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator          = LinearRegression(),
                     level              = 'l1',
                     steps              = 5,
                     lags               = 3,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results = forecaster.predict_interval(
        steps=5, method='bootstrapping', interval=(0.05, 0.95), 
        use_in_sample_residuals=True, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                    data    = np.array(
                                [[0.58307704, 0.30689592, 0.97766641],
                                 [0.40064856, 0.04683134, 0.63486761],
                                 [0.29394488, 0.13006777, 0.50853662],
                                 [0.41007329, 0.0785547 , 0.85997779],
                                 [0.4390632 , 0.10298379, 0.83922684]]
                            ),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index   = pd.RangeIndex(start=50, stop=55, step=1)
                )
    expected.insert(0, 'level', np.tile(['l1'], forecaster.max_step))

    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_5_in_sample_residuals_is_False_binned_residuals_is_True():
    """
    Test output when estimator is LinearRegression, steps=5, use_in_sample_residuals=False,
    binned_residuals=True.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator          = LinearRegression(),
                     level              = 'l1',
                     steps              = 5,
                     lags               = 3,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    forecaster.out_sample_residuals_by_bin_ = forecaster.in_sample_residuals_by_bin_

    results = forecaster.predict_interval(
        steps=5, method='bootstrapping', interval=(0.05, 0.95), 
        use_in_sample_residuals=False, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                    data    = np.array(
                                [[0.58307704, 0.30689592, 0.97766641],
                                 [0.40064856, 0.04683134, 0.63486761],
                                 [0.29394488, 0.13006777, 0.50853662],
                                 [0.41007329, 0.0785547 , 0.85997779],
                                 [0.4390632 , 0.10298379, 0.83922684]]
                            ),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index   = pd.RangeIndex(start=50, stop=55, step=1)
                )
    expected.insert(0, 'level', np.tile(['l1'], forecaster.max_step))
    
    pd.testing.assert_frame_equal(results, expected)


@pytest.mark.parametrize("interval", 
                         [0.95, (0.025, 0.975)], 
                         ids = lambda value: f'interval: {value}')
def test_predict_interval_conformal_output_when_estimator_is_LinearRegression(interval):
    """
    Test predict output when using LinearRegression as estimator and StandardScaler
    and conformal prediction.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator          = LinearRegression(),
                     level              = 'l1',
                     steps              = 3,
                     lags               = 3,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series, store_in_sample_residuals=False)
    forecaster.set_in_sample_residuals(series=series)
    results = forecaster.predict_interval(
        steps=3, method='conformal', interval=interval, 
        use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data = np.array([[ 0.63114259,  0.22697524,  1.03530995],
                                    [ 0.3800417 , -0.02412565,  0.78420905],
                                    [ 0.33255977, -0.07160758,  0.73672712]]),
                   index = pd.RangeIndex(start=50, stop=53, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    expected.insert(0, 'level', np.tile(['l1'], forecaster.max_step))
    
    pd.testing.assert_frame_equal(results, expected)


@pytest.mark.parametrize("interval", 
                         [0.95, (0.025, 0.975)], 
                         ids = lambda value: f'interval: {value}')
def test_predict_interval_conformal_output_when_binned_residuals(interval):
    """
    Test predict output when using LinearRegression as estimator and StandardScaler
    and conformal prediction with binned residuals.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator          = LinearRegression(),
                     level              = 'l1',
                     steps              = 3,
                     lags               = 3,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results = forecaster.predict_interval(
        steps=3, method='conformal', interval=interval, 
        use_in_sample_residuals=True, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [0.63114259, 0.17603311, 1.08625208],
                              [0.3800417 , 0.12832655, 0.63175685],
                              [0.33255977, 0.08084462, 0.58427492]]),
                   index = pd.RangeIndex(start=50, stop=53, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    expected.insert(0, 'level', np.tile(['l1'], forecaster.max_step))
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_bootstrapping_binned_residuals_when_binner_reduces_n_bins():
    """
    Test predict_interval with method 'bootstrapping' and binned residuals when
    the predictions are so concentrated that the binner has to reduce the number
    of bins. Every bin id returned by the binner must have residuals associated
    with it.
    """
    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), lags=3, steps=3, level='l1'
    )
    warn_msg = re.escape(
        "The number of bins has been reduced from 10 to 9 due to empty bins. "
        "This happens when "
        "the values used to compute the edges of the bins are highly "
        "concentrated or contain many repeated values.",
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        forecaster.fit(series=series_intermittent, store_in_sample_residuals=True)
    # NOTE: The last window of `series_intermittent` is all zeros, the same
    # window that appears many times in the training set. Its scaled prediction
    # therefore coincides exactly with a bin edge, and the bin it falls into
    # depends on floating point noise that varies between machines. A last
    # window with a non zero value keeps the predictions away from any edge.
    last_window = series_intermittent.iloc[-5:-2]
    results = forecaster.predict_interval(
        steps=3, last_window=last_window, method='bootstrapping',
        interval=0.8, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [-2.4748361 , -1.61054424,  3.1273761 ],
                              [15.8793251 ,  4.23219054, 60.87783227],
                              [-2.14747949, -1.32301949,  3.45473271]]),
                   index = pd.date_range(start='2020-03-19', periods=3, freq='D'),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    expected.insert(0, 'level', np.tile(['l1'], forecaster.max_step))

    assert forecaster.binner['l1'].n_bins_ == 9
    assert sorted(forecaster.in_sample_residuals_by_bin_['l1']) == list(range(9))
    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_conformal_binned_residuals_when_binner_reduces_n_bins():
    """
    Test predict_interval with method 'conformal' and binned residuals when the
    predictions are so concentrated that the binner has to reduce the number of
    bins. Every bin id returned by the binner must have residuals associated
    with it.
    """
    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), lags=3, steps=3, level='l1'
    )
    warn_msg = re.escape(
        "The number of bins has been reduced from 10 to 9 due to empty bins. "
        "This happens when "
        "the values used to compute the edges of the bins are highly "
        "concentrated or contain many repeated values.",
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        forecaster.fit(series=series_intermittent, store_in_sample_residuals=True)
    # NOTE: The last window of `series_intermittent` is all zeros, the same
    # window that appears many times in the training set. Its scaled prediction
    # therefore coincides exactly with a bin edge, and the bin it falls into
    # depends on floating point noise that varies between machines. A last
    # window with a non zero value keeps the predictions away from any edge.
    last_window = series_intermittent.iloc[-5:-2]
    results = forecaster.predict_interval(
        steps=3, last_window=last_window, method='conformal',
        interval=0.8, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [-2.4748361 , -7.4929334 ,  2.5432612 ],
                              [15.8793251 ,  2.973606  , 28.78504421],
                              [-2.14747949, -7.16557679,  2.8706178 ]]),
                   index = pd.date_range(start='2020-03-19', periods=3, freq='D'),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    expected.insert(0, 'level', np.tile(['l1'], forecaster.max_step))

    assert forecaster.binner['l1'].n_bins_ == 9
    assert sorted(forecaster.in_sample_residuals_by_bin_['l1']) == list(range(9))
    pd.testing.assert_frame_equal(results, expected)
