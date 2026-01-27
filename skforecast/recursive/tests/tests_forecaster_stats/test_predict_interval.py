# Unit test predict_interval ForecasterStats
# ==============================================================================
import re
import pytest
import platform
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.stats import Sarimax, Arar, Ets
from skforecast.recursive import ForecasterStats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Fixtures
from .fixtures_forecaster_stats import y
from .fixtures_forecaster_stats import y_lw
from .fixtures_forecaster_stats import exog
from .fixtures_forecaster_stats import exog_lw
from .fixtures_forecaster_stats import exog_predict
from .fixtures_forecaster_stats import exog_lw_predict
from .fixtures_forecaster_stats import y_datetime
from .fixtures_forecaster_stats import y_lw_datetime
from .fixtures_forecaster_stats import exog_datetime
from .fixtures_forecaster_stats import exog_lw_datetime
from .fixtures_forecaster_stats import exog_predict_datetime
from .fixtures_forecaster_stats import exog_lw_predict_datetime
from .fixtures_forecaster_stats import df_exog
from .fixtures_forecaster_stats import df_exog_lw
from .fixtures_forecaster_stats import df_exog_predict
from .fixtures_forecaster_stats import df_exog_lw_predict
from .fixtures_forecaster_stats import df_exog_datetime
from .fixtures_forecaster_stats import df_exog_lw_datetime
from .fixtures_forecaster_stats import df_exog_predict_datetime
from .fixtures_forecaster_stats import df_exog_lw_predict_datetime


def test_predict_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 1, 1)))

    err_msg = re.escape(
        "This Forecaster instance is not fitted yet. Call `fit` with "
        "appropriate arguments before using predict."
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict_interval(
            steps = 5, 
            alpha = 0.05
        )


def test_predict_interval_ValueError_when_interval_is_not_symmetrical():
    """
    Raise ValueError if `interval` is not symmetrical.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 1, 1)))
    forecaster.fit(y=y)
    alpha = None
    interval_not_symmetrical = [5, 97.5] 

    err_msg = re.escape(
        f"When using `interval` in ForecasterStats, it must be symmetrical. "
        f"For example, interval of 95% should be as `interval = [2.5, 97.5]`. "
        f"Got {interval_not_symmetrical}."
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
                         ids = lambda values: f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterStats_skforecast_Sarimax(alpha, interval):
    """
    Test predict_interval output of ForecasterStats using Sarimax from skforecast.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(maxiter=2000, method='cg', disp=False, order=(3, 2, 0))
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
                         ids = lambda values: f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterStats_with_exog(alpha, interval):
    """
    Test predict_interval output of ForecasterStats with exogenous variables.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(maxiter=1000, method='cg', disp=False, order=(1, 0, 1))
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
                         ids = lambda values: f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterStats_multiple_estimators_exog(alpha, interval):
    """
    Test predict_interval output of ForecasterStats with multiple estimators
    and exogenous variables. Ets is included to check that bounds are reasonable
    (lower_bound < pred < upper_bound), but cannot be checked against stable values 
    as they differ slightly between runs.
    """
    estimators = [
        Sarimax(order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), maxiter=1000, method='cg', disp=False),
        Arar(),
        Ets(trend='add', seasonal=None) 
    ]
    forecaster = ForecasterStats(estimator=estimators)
    forecaster.fit(y=y, exog=exog)
    predictions = forecaster.predict_interval(
        steps=5, exog=exog_predict, alpha=alpha, interval=interval
    )
    
    # Check estimator and pred columns (stable values)
    expected_estimator_pred = pd.DataFrame(
                   data    = {
                       'estimator_id': ['skforecast.Sarimax', 'skforecast.Arar', 'skforecast.Ets'] * 5,
                       'pred': [0.599299, 0.635100, 0.604989,
                                0.612997, 0.677110, 0.604981,
                                0.628731, 0.765381, 0.604974,
                                0.644136, 0.757966, 0.604969,
                                0.661960, 0.800557, 0.604965], 
                   },
                   index   = pd.Index([50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54])
               )
    pd.testing.assert_frame_equal(
        predictions[['estimator_id', 'pred']], 
        expected_estimator_pred, 
        atol=0.0001
    )
    
    # Check bounds for Sarimax and Arar (stable values) - select by estimator_id
    sarimax_arar_mask = predictions['estimator_id'].isin(['skforecast.Sarimax', 'skforecast.Arar'])
    sarimax_arar_predictions = predictions[sarimax_arar_mask]
    expected_bounds_sarimax_arar = pd.DataFrame(
                   data    = {
                       'lower_bound': [0.578620, 0.566307, 
                                       0.592025, 0.606592,  
                                       0.607742, 0.694777,  
                                       0.623146, 0.687357,  
                                       0.640970, 0.724952], 
                       'upper_bound': [0.619978, 0.703893,  
                                       0.633969, 0.747628,  
                                       0.649720, 0.835985,  
                                       0.665126, 0.828575,  
                                       0.682950, 0.876162], 
                   },
                   index   = pd.Index([50, 50, 51, 51, 52, 52, 53, 53, 54, 54])
               )
    pd.testing.assert_frame_equal(
        sarimax_arar_predictions[['lower_bound', 'upper_bound']], 
        expected_bounds_sarimax_arar, 
        atol=0.0001
    )
    
    # Check that bounds for Ets are reasonable (lower < pred < upper)
    ets_predictions = predictions[predictions['estimator_id'] == 'skforecast.Ets']
    assert (ets_predictions['lower_bound'] < ets_predictions['pred']).all()
    assert (ets_predictions['pred'] < ets_predictions['upper_bound']).all()


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values: f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterStats_with_transform_y(alpha, interval):
    """
    Test predict_interval output of ForecasterStats with a StandardScaler() as transformer_y.
    """
    forecaster = ForecasterStats(
                     estimator     = Sarimax(maxiter=1000, method='cg', disp=False, order=(1, 1, 1)),
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
                         ids = lambda values: f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterStats_multiple_estimators_exog_transform_y(alpha, interval):
    """
    Test predict_interval output of ForecasterStats with multiple estimators, 
    exogenous variables and StandardScaler() as transformer_y. Ets is included 
    to check that bounds are reasonable (lower_bound < pred < upper_bound), but 
    cannot be checked against stable values as they differ slightly between runs.
    """
    estimators = [
        Sarimax(order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), maxiter=1000, method='cg', disp=False),
        Arar(),
        Ets(trend='add', seasonal=None) 
    ]
    forecaster = ForecasterStats(
        estimator=estimators, transformer_y=StandardScaler()
    )
    forecaster.fit(y=y, exog=exog)
    predictions = forecaster.predict_interval(
        steps=5, exog=exog_predict, alpha=alpha, interval=interval
    )
    
    # Check estimator and pred columns (stable values)
    expected_estimator_pred = pd.DataFrame(
                   data    = {
                       'estimator_id': ['skforecast.Sarimax', 'skforecast.Arar', 'skforecast.Ets'] * 5,
                       'pred': [0.611820, 0.635100, 0.693197, 
                                0.613855, 0.677110, 0.693995, 
                                0.613302, 0.765381, 0.694766, 
                                0.613836, 0.757966, 0.695513, 
                                0.613947, 0.800557, 0.696234],
                   },
                   index   = pd.Index([50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54])
               )
    pd.testing.assert_frame_equal(
        predictions[['estimator_id', 'pred']], 
        expected_estimator_pred, 
        atol=0.0001
    )
    
    # Check bounds for Sarimax and Arar (stable values) - select by estimator_id
    sarimax_arar_mask = predictions['estimator_id'].isin(['skforecast.Sarimax', 'skforecast.Arar'])
    sarimax_arar_predictions = predictions[sarimax_arar_mask]
    expected_bounds_sarimax_arar = pd.DataFrame(
                   data    = {
                       'lower_bound': [0.448996, 0.566307, 
                                       0.450212, 0.606592,  
                                       0.449515, 0.694777,  
                                       0.450024, 0.687357,  
                                       0.450130, 0.724952], 
                       'upper_bound': [0.774643, 0.703893,  
                                       0.777497, 0.747628,  
                                       0.777089, 0.835985,  
                                       0.777649, 0.828575,  
                                       0.777764, 0.876162], 
                   },
                   index   = pd.Index([50, 50, 51, 51, 52, 52, 53, 53, 54, 54])
               )
    pd.testing.assert_frame_equal(
        sarimax_arar_predictions[['lower_bound', 'upper_bound']], 
        expected_bounds_sarimax_arar, 
        atol=0.0001
    )
    
    # Check that bounds for Ets are reasonable (lower < pred < upper)
    ets_predictions = predictions[predictions['estimator_id'] == 'skforecast.Ets']
    assert (ets_predictions['lower_bound'] < ets_predictions['pred']).all()
    assert (ets_predictions['pred'] < ets_predictions['upper_bound']).all()


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values: f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterStats_with_transform_y_and_transform_exog(alpha, interval):
    """
    Test predict_interval output of ForecasterStats, StandardScaler
    as transformer_y and transformer_exog as transformer_exog.
    """
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterStats(
                     estimator        = Sarimax(maxiter=1000, method='cg', disp=False, order=(1, 0, 1)),
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


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values: f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterStats_with_last_window(alpha, interval):
    """
    Test predict_interval output of ForecasterStats with `last_window`.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(maxiter=1000, method='cg', disp=False, order=(3, 2, 0))
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
                         ids = lambda values: f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterStats_with_last_window_and_exog(alpha, interval):
    """
    Test predict_interval output of ForecasterStats with exogenous variables and `last_window`.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(maxiter=1000, method='cg', disp=False, order=(1, 1, 1))
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
                         ids = lambda values: f'alpha, interval: {values}')
def test_predict_interval_output_ForecasterStats_with_last_window_and_exog_and_transformers(alpha, interval):
    """
    Test predict_interval output of ForecasterStats with exogenous variables and `last_window`.
    """
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterStats(
                     estimator = Sarimax(maxiter=1000, method='cg', disp=False, order=(1, 1, 1)),
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
def test_predict_interval_ForecasterStats_updates_extended_index_twice(y, idx):
    """
    Test forecaster.extended_index_ is updated when using predict_interval twice.
    """
    y_fit = y.iloc[:30].copy()

    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 0)))
    forecaster.fit(y=y_fit)

    lw_1 = y.iloc[30:40].copy()
    forecaster.predict_interval(steps=5, alpha = 0.05, last_window=lw_1)
    result_1 = forecaster.extended_index_.copy()
    expected_1 = idx[:40]

    lw_2 = y.iloc[40:].copy()
    forecaster.predict_interval(steps=5, alpha = 0.05, last_window=lw_2)

    pd.testing.assert_index_equal(result_1, expected_1)
    pd.testing.assert_index_equal(forecaster.extended_index_, idx)


@pytest.mark.parametrize(
    "estimator, expected_data",
    [
        (
            Sarimax(order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), maxiter=1000, method='cg', disp=False),
            np.array([[0.605353  , 0.352368  , 0.858338  ],
                      [0.596542  , 0.243767  , 0.949317  ],
                      [0.587858  , 0.159896  , 1.015821  ],
                      [0.579301  , 0.089243  , 1.069360  ],
                      [0.570869  , 0.027256  , 1.114482  ],
                      [0.562559  , -0.028434 , 1.153553  ],
                      [0.554371  , -0.079252 , 1.187993  ],
                      [0.546301  , -0.126137 , 1.218740  ],
                      [0.538349  , -0.169750 , 1.246448  ],
                      [0.530513  , -0.210575 , 1.271601  ]])
        ),
        (
            Arar(max_ar_depth=26, max_lag=40),
            np.array([[0.65451694, 0.56798138, 0.7410525 ],
                      [0.69369274, 0.60112468, 0.78626081],
                      [0.8018875 , 0.70848121, 0.8952938 ],
                      [0.82157326, 0.72804665, 0.91509988],
                      [0.87868702, 0.78514306, 0.97223098],
                      [0.88798496, 0.79443849, 0.98153142],
                      [1.01739572, 0.92384889, 1.11094254],
                      [1.02221717, 0.92867029, 1.11576405],
                      [0.5688093 , 0.47526242, 0.66235619],
                      [0.63365663, 0.54010974, 0.72720352]])
        ),
        (
            Ets(model='AAN', damped=False),
            np.array([[0.681232  , 0.290318  , 1.072146  ],
                      [0.679951  , 0.286679  , 1.073223  ],
                      [0.678670  , 0.282610  , 1.074730  ],
                      [0.677389  , 0.278082  , 1.076696  ],
                      [0.676108  , 0.273068  , 1.079148  ],
                      [0.674827  , 0.267544  , 1.082110  ],
                      [0.673546  , 0.261489  , 1.085604  ],
                      [0.672265  , 0.254883  , 1.089647  ],
                      [0.670984  , 0.247713  , 1.094256  ],
                      [0.669703  , 0.239965  , 1.099442  ]])
        ),
    ],
    ids=['Sarimax', 'Arar', 'Ets']
)
def test_predict_interval_output_ForecasterStats_multiple_estimators(estimator, expected_data, y=y):
    """
    Test output of predict_interval when using different estimators (Sarimax, Arar, Ets) in ForecasterStats.
    """
    y = y.copy()
    y.index = pd.date_range(start='2000-01-01', periods=len(y), freq='D')
    forecaster = ForecasterStats(estimator=estimator)
    forecaster.fit(y=y)
    predictions = forecaster.predict_interval(steps=10, alpha=0.05)

    if isinstance(estimator, Sarimax) and platform.system() == 'Windows':
        expected_data = np.array([
            [ 0.63432268, -1.62088311,  2.88952848],
            [ 0.62507372, -3.25642788,  4.50657533],
            [ 0.61595962, -4.3597053 ,  5.59162454],
            [ 0.60697841, -5.2383511 ,  6.45230791],
            [ 0.59812815, -5.98260763,  7.17886392],
            [ 0.58940693, -6.63414807,  7.81296193],
            [ 0.58081288, -7.21639982,  8.37802558],
            [ 0.57234414, -7.7441369 ,  8.88882518],
            [ 0.56399888, -8.2274152 ,  9.35541295],
            [ 0.55577529, -8.67346365,  9.78501423]]
        )

    expected_results = pd.DataFrame(
        data=expected_data,
        columns=['pred', 'lower_bound', 'upper_bound'],
        index=pd.date_range(start="2000-02-20", periods=10, freq='D')
    )

    pd.testing.assert_frame_equal(predictions, expected_results, atol=0.0001)

