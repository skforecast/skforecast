# Unit test predict_interval ForecasterSarimax
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.sarimax import Sarimax
from skforecast.recursive import ForecasterSarimax
from skforecast.utils import expand_index
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Fixtures
from .fixtures_forecaster_sarimax import y
from .fixtures_forecaster_sarimax import y_lw
from .fixtures_forecaster_sarimax import exog
from .fixtures_forecaster_sarimax import exog_lw
from .fixtures_forecaster_sarimax import exog_predict
from .fixtures_forecaster_sarimax import exog_lw_predict
from .fixtures_forecaster_sarimax import y_datetime
from .fixtures_forecaster_sarimax import y_lw_datetime
from .fixtures_forecaster_sarimax import exog_datetime
from .fixtures_forecaster_sarimax import exog_lw_datetime
from .fixtures_forecaster_sarimax import exog_predict_datetime
from .fixtures_forecaster_sarimax import exog_lw_predict_datetime
from .fixtures_forecaster_sarimax import df_exog
from .fixtures_forecaster_sarimax import df_exog_lw
from .fixtures_forecaster_sarimax import df_exog_predict
from .fixtures_forecaster_sarimax import df_exog_lw_predict
from .fixtures_forecaster_sarimax import df_exog_datetime
from .fixtures_forecaster_sarimax import df_exog_lw_datetime
from .fixtures_forecaster_sarimax import df_exog_predict_datetime
from .fixtures_forecaster_sarimax import df_exog_lw_predict_datetime


def test_predict_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 1, 1)))

    err_msg = re.escape(
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict_interval(
            steps = 5, 
            alpha = 0.05
        )


def test_predict_interval_ValueError_when_ForecasterSarimax_last_window_exog_is_not_None_and_last_window_is_not_provided():
    """
    Check ValueError is raised when last_window_exog is not None, but 
    last_window is not provided.
    """
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 1, 1)))
    forecaster.fit(y=y, exog=exog)
    
    err_msg = re.escape(
                ("To make predictions unrelated to the original data, both "
                 "`last_window` and `last_window_exog` must be provided.")
              )   
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(
            steps            = 5, 
            alpha            = 0.05, 
            exog             = exog_predict, 
            last_window      = None, 
            last_window_exog = exog
        )


def test_predict_interval_ValueError_when_ForecasterSarimax_last_window_exog_is_None_and_included_exog_is_true():
    """
    Check ValueError is raised when last_window_exog is None, but included_exog
    is True and last_window is provided.
    """
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 1, 1)))
    forecaster.fit(y=y, exog=exog)
    
    err_msg = re.escape(
                ("Forecaster trained with exogenous variable/s. To make predictions "
                 "unrelated to the original data, same variable/s must be provided "
                 "using `last_window_exog`.")
              )   
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(
            steps            = 5, 
            alpha            = 0.05, 
            exog             = exog_lw_predict, 
            last_window      = y_lw, 
            last_window_exog = None
        )


def test_predict_interval_ValueError_when_interval_is_not_symmetrical():
    """
    Raise ValueError if `interval` is not symmetrical.
    """
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 1, 1)))
    forecaster.fit(y=y)
    alpha = None
    interval_not_symmetrical = [5, 97.5] 

    err_msg = re.escape(
                (f"When using `interval` in ForecasterSarimax, it must be symmetrical. "
                 f"For example, interval of 95% should be as `interval = [2.5, 97.5]`. "
                 f"Got {interval_not_symmetrical}.")
            )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(
            steps    = 5, 
            alpha    = alpha, 
            interval = interval_not_symmetrical
        )


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_skforecast_Sarimax(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax using Sarimax from skforecast.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(maxiter=2000, method='cg', disp=False, order=(3, 2, 0))
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict_interval(steps=5, alpha=alpha, interval=interval)
    expected = pd.DataFrame(
                   data    = np.array([[ 0.53809981,  0.24275351,  0.83344611],
                                       [ 0.53145374,  0.0431938 ,  1.01971368],
                                       [ 0.53763636, -0.12687285,  1.20214556],
                                       [ 0.52281442, -0.35748984,  1.40311868],
                                       [ 0.49770378, -0.64436866,  1.63977622]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=55, step=1)
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_exog(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax with exogenous variables.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(maxiter=1000, method='cg', disp=False, order=(1, 0, 1))
                 )
    forecaster.fit(y=y, exog=exog)
    predictions = forecaster.predict_interval(steps=5, exog=exog_predict, alpha=alpha, interval=interval)
    expected = pd.DataFrame(
                   data    = np.array([[0.59929905, 0.57862017, 0.61997793],
                                       [0.61299725, 0.59202539, 0.63396911],
                                       [0.6287311 , 0.60774224, 0.64971995],
                                       [0.64413557, 0.62314573, 0.66512542],
                                       [0.66195978, 0.64096988, 0.68294969]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=55, step=1)
               )
    
    pd.testing.assert_frame_equal(predictions, expected, atol=0.0001)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_transform_y(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax with a StandardScaler() as transformer_y.
    """
    forecaster = ForecasterSarimax(
                     regressor     = Sarimax(maxiter=1000, method='cg', disp=False, order=(1, 1, 1)),
                     transformer_y = StandardScaler()
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict_interval(steps=5, alpha=alpha, interval=interval)
    expected = pd.DataFrame(
                   data    = np.array([[0.63520867, 0.61383185, 0.6565855 ],
                                       [0.61741115, 0.5894499 , 0.6453724 ],
                                       [0.6330291 , 0.60053638, 0.66552182],
                                       [0.6193238 , 0.58402618, 0.65462142],
                                       [0.63135068, 0.59379186, 0.66890951]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=55, step=1)
               )
    
    pd.testing.assert_frame_equal(predictions, expected, atol=0.0001)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_transform_y_and_transform_exog(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax, StandardScaler
    as transformer_y and transformer_exog as transformer_exog.
    """
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterSarimax(
                     regressor        = Sarimax(maxiter=1000, method='cg', disp=False, order=(1, 0, 1)),
                     transformer_y    = StandardScaler(),
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y, exog=df_exog)
    predictions = forecaster.predict_interval(steps=5, exog=df_exog_predict, alpha=alpha, interval=interval)
    expected = pd.DataFrame(
                   data    = np.array([[0.60687311, 0.50667956, 0.70706666],
                                       [0.62484493, 0.49759696, 0.75209289],
                                       [0.63515416, 0.50776733, 0.762541  ],
                                       [0.67730912, 0.54992148, 0.80469675],
                                       [0.69458838, 0.56720074, 0.82197602]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=55, step=1)
               )
    
    pd.testing.assert_frame_equal(predictions, expected, atol=0.0001)


def test_predict_interval_ValueError_when_last_window_index_does_not_follow_training_set():
    """
    Raise ValueError if `last_window` index does not start at the end 
    of the index seen by the forecaster.
    """
    y_test = pd.Series(data=y_datetime.to_numpy())
    y_test.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    lw_test = pd.Series(data=y_lw_datetime.to_numpy())
    lw_test.index = pd.date_range(start='2022-03-01', periods=50, freq='D')

    forecaster = ForecasterSarimax(regressor = Sarimax(order=(1, 0, 0)))
    forecaster.fit(y=y_test)
    expected_index = expand_index(forecaster.extended_index_, 1)[0]

    err_msg = re.escape(
        (f"To make predictions unrelated to the original data, `last_window` "
         f"has to start at the end of the index seen by the forecaster.\n"
         f"    Series last index         : {forecaster.extended_index_[-1]}.\n"
         f"    Expected index            : {expected_index}.\n"
         f"    `last_window` index start : {lw_test.index[0]}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(
            steps            = 5, 
            alpha            = 0.05,
            last_window      = lw_test,
        )


def test_predict_interval_ValueError_when_last_window_exog_index_does_not_follow_training_set():
    """
    Raise ValueError if `last_window_exog` index does not start at the end 
    of the index seen by the forecaster.
    """
    y_test = pd.Series(data=y_datetime.to_numpy())
    y_test.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    lw_test = pd.Series(data=y_lw_datetime.to_numpy())
    lw_test.index = pd.date_range(start='2022-02-20', periods=50, freq='D')

    exog_test = pd.Series(data=exog_datetime.to_numpy(), name='exog')
    exog_test.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    exog_pred_test = pd.Series(data=exog_predict_datetime.to_numpy(), name='exog')
    exog_pred_test.index = pd.date_range(start='2022-04-11', periods=10, freq='D')
    lw_exog_test = pd.Series(data=exog_lw_datetime.to_numpy(), name='exog')
    lw_exog_test.index = pd.date_range(start='2022-03-01', periods=50, freq='D')

    forecaster = ForecasterSarimax(regressor = Sarimax(order=(1, 0, 0)))
    forecaster.fit(y=y_test, exog=exog_test)
    expected_index = expand_index(forecaster.extended_index_, 1)[0]

    err_msg = re.escape(
        (f"To make predictions unrelated to the original data, `last_window_exog` "
         f"has to start at the end of the index seen by the forecaster.\n"
         f"    Series last index              : {forecaster.extended_index_[-1]}.\n"
         f"    Expected index                 : {expected_index}.\n"
         f"    `last_window_exog` index start : {lw_exog_test.index[0]}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(
            steps            = 5, 
            alpha            = 0.05,
            exog             = exog_pred_test, 
            last_window      = lw_test, 
            last_window_exog = lw_exog_test
        )


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_last_window(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax with `last_window`.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(maxiter=1000, method='cg', disp=False, order=(3, 2, 0))
                 )
    forecaster.fit(y=y_datetime)
    predictions = forecaster.predict_interval(
                      steps            = 5, 
                      alpha            = alpha,
                      interval         = interval,
                      last_window      = y_lw_datetime,
                  )
    
    expected = pd.DataFrame(
                    data = np.array([[0.91877817, 0.62343187, 1.21412446],
                                     [0.98433512, 0.49607518, 1.47259506],
                                     [1.06945921, 0.40495001, 1.73396842],
                                     [1.15605055, 0.27574629, 2.03635481],
                                     [1.22975713, 0.08768469, 2.37182957]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.date_range(start='2100', periods=5, freq='YE')
               )
    
    pd.testing.assert_frame_equal(predictions, expected, atol=0.0001)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_last_window_and_exog(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax with exogenous variables and `last_window`.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(maxiter=1000, method='cg', disp=False, order=(1, 1, 1))
                 )
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    predictions = forecaster.predict_interval(
                      steps            = 5, 
                      alpha            = alpha,
                      interval         = interval,
                      exog             = exog_lw_predict_datetime, 
                      last_window      = y_lw_datetime, 
                      last_window_exog = exog_lw_datetime
                  )

    expected = pd.DataFrame(
                   data    = np.array([[ 0.89386888, -0.84405923,  2.63179699],
                                       [ 0.92919515, -1.45638221,  3.3147725 ],
                                       [ 0.98327241, -1.88128514,  3.84782996],
                                       [ 1.02336286, -2.2399583 ,  4.28668401],
                                       [ 1.05334974, -2.56051157,  4.66721105]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.date_range(start='2100', periods=5, freq='YE')
               )
    
    pd.testing.assert_frame_equal(predictions, expected, atol=0.0001)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values : f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterSarimax_with_last_window_and_exog_and_transformers(alpha, interval):
    """
    Test predict_interval output of ForecasterSarimax with exogenous variables and `last_window`.
    """
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterSarimax(
                     regressor = Sarimax(maxiter=1000, method='cg', disp=False, order=(1, 1, 1)),
                     transformer_y    = StandardScaler(),
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y_datetime, exog=df_exog_datetime)
    predictions = forecaster.predict_interval(
                      steps            = 5, 
                      alpha            = alpha,
                      interval         = interval,
                      exog             = df_exog_lw_predict_datetime, 
                      last_window      = y_lw_datetime, 
                      last_window_exog = df_exog_lw_datetime
                  )

    expected = pd.DataFrame(
                   data    = np.array([[0.61139264, 0.35457567, 0.86820961],
                                       [0.88228163, 0.57163268, 1.19293057],
                                       [0.77749663, 0.42990006, 1.12509319],
                                       [0.94985823, 0.58885008, 1.31086638],
                                       [0.89218798, 0.5184476 , 1.26592836]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.date_range(start='2100', periods=5, freq='YE')
               )
    
    pd.testing.assert_frame_equal(predictions, expected, atol=0.0001)


@pytest.mark.parametrize("y          , idx", 
                         [(y         , pd.RangeIndex(start=0, stop=50)), 
                          (y_datetime, pd.date_range(start='2000', periods=50, freq='YE'))], 
                         ids = lambda values: f'y, index: {type(values)}')
def test_predict_interval_ForecasterSarimax_updates_extended_index_twice(y, idx):
    """
    Test forecaster.extended_index_ is updated when using predict_interval twice.
    """
    y_fit = y.iloc[:30].copy()

    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 0, 0)))
    forecaster.fit(y=y_fit)

    lw_1 = y.iloc[30:40].copy()
    forecaster.predict_interval(steps=5, alpha = 0.05, last_window=lw_1)
    result_1 = forecaster.extended_index_.copy()
    expected_1 = idx[:40]

    lw_2 = y.iloc[40:].copy()
    forecaster.predict_interval(steps=5, alpha = 0.05, last_window=lw_2)

    pd.testing.assert_index_equal(result_1, expected_1)
    pd.testing.assert_index_equal(forecaster.extended_index_, idx)
