# Unit test predict ForecasterStats
# ==============================================================================
import re
import pytest
import platform
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.stats import Sarimax, Arar, Ets
from skforecast.recursive import ForecasterStats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from aeon.forecasting.stats import ARIMA

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
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict(steps=5)


def test_predict_ValueError_when_ForecasterStats_last_window_exog_is_not_None_and_last_window_is_not_provided():
    """
    Check ValueError is raised when last_window_exog is not None, but 
    last_window is not provided.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 1, 1)))
    forecaster.fit(y=y, exog=exog)
    
    err_msg = re.escape(
        ("To make predictions unrelated to the original data, both "
         "`last_window` and `last_window_exog` must be provided.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict(
            steps            = 5, 
            exog             = exog_predict, 
            last_window      = None, 
            last_window_exog = exog
        )


def test_predict_ValueError_when_ForecasterStats_last_window_exog_is_None_and_included_exog_is_true():
    """
    Check ValueError is raised when last_window_exog is None, but included_exog
    is True and last_window is provided.
    """
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 1, 1)))
    forecaster.fit(y=y, exog=exog)
    
    err_msg = re.escape(
        ("Forecaster trained with exogenous variable/s. To make predictions "
         "unrelated to the original data, same variable/s must be provided "
         "using `last_window_exog`.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict(
            steps            = 5, 
            exog             = exog_lw_predict, 
            last_window      = y_lw, 
            last_window_exog = None
        )


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1, 0, 1), 
                            'seasonal_order': (0, 0, 0, 0)}, 
                           {'win': [0.63432268, 0.62507372, 0.61595962, 0.60697841, 0.59812815],
                            'linux': [0.60535333, 0.59654171, 0.58785836, 0.5793014 , 0.570869]}), 
                          ({'order': (1, 1, 1), 
                            'seasonal_order': (1, 1, 1, 2)}, 
                           {'win': [0.5366165, 0.55819701, 0.49539926, 0.51944837, 0.45417575],
                            'linux': [0.5366165 , 0.55819701, 0.49539926, 0.51944837, 0.45417575]})])
def test_predict_output_ForecasterStats_skforecast_Sarimax(kwargs, data):
    """
    Test predict output of ForecasterStats using Sarimax from skforecast.
    """
    system = "win" if platform.system() == "Windows" else 'linux'

    forecaster = ForecasterStats(
                     estimator = Sarimax(maxiter=1000, method='cg', disp=False, **kwargs)
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=5)
    expected = pd.Series(
                   data  = data[system],
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   name  = 'pred'
               )

    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1, 0, 1), 
                            'seasonal_order': (0, 0, 0, 0)}, 
                            [0.59929905, 0.61299725, 0.6287311 , 0.64413557, 0.66195978]), 
                          ({'order': (1, 1, 1), 
                            'seasonal_order': (1, 1, 1, 2)}, 
                            [0.47217517, 0.57747478, 0.58655865, 0.69219403, 0.71031467])])
def test_predict_output_ForecasterStats_with_exog(kwargs, data):
    """
    Test predict output of ForecasterStats with exogenous variables.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(maxiter=1000, method='cg', disp=False, **kwargs)
                 )
    forecaster.fit(y=y, exog=exog)
    predictions = forecaster.predict(steps=5, exog=exog_predict)
    expected = pd.Series(
                   data  = data,
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize(
    "estimator, expected_data",
    [
        (
            Sarimax(order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), maxiter=1000, method='cg', disp=False),
            {'win': [0.60290703, 0.60568721, 0.60451413, 0.6050091, 0.60480025],
             'linux': [0.60290703, 0.60568721, 0.60451413, 0.6050091, 0.60480025]}
        ),
        (
            Arar(),
            {'win': [0.62548412, 0.63711385, 0.70171521, 0.68564555, 0.72810186],
             'linux': [0.62548412, 0.63711385, 0.70171521, 0.68564555, 0.72810186]}
        ),
        (
            Ets(trend='add', seasonal=None),
            {'win': [0.69319696, 0.6939948, 0.69476642, 0.69551268, 0.69623443],
             'linux': [0.69319696, 0.6939948, 0.69476642, 0.69551268, 0.69623443]}
        ),
    ],
    ids=['Sarimax', 'Arar', 'Ets']
)
def test_predict_output_ForecasterStats_with_transform_y(estimator, expected_data):
    """
    Test predict output of ForecasterStats with a StandardScaler() as transformer_y
    for different estimators.
    """
    system = "win" if platform.system() == "Windows" else 'linux'
    
    forecaster = ForecasterStats(
                     estimator     = estimator,
                     transformer_y = StandardScaler()
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=5)
    expected = pd.Series(
                   data  = expected_data[system],
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1, 0, 1), 
                            'seasonal_order': (0, 0, 0, 0)},
                            {'win': [0.60687311, 0.62484493, 0.63515416, 0.67730912, 0.69458838],
                            'linux': [0.60687186, 0.62484336, 0.63515295, 0.67730812, 0.69458769]}
                            )])
def test_predict_output_ForecasterStats_with_transform_y_and_transform_exog(kwargs, data):
    """
    Test predict output of ForecasterStats, StandardScaler
    as transformer_y and transformer_exog as transformer_exog.
    """
    system = "win" if platform.system() == "Windows" else 'linux'
    
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterStats(
                     estimator        = Sarimax(maxiter=1000, method='cg', disp=False, **kwargs),
                     transformer_y    = StandardScaler(),
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y, exog=df_exog)
    predictions = forecaster.predict(steps=5, exog=df_exog_predict)
    expected = pd.Series(
                   data  = data[system],
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


def test_predict_ValueError_when_last_window_index_does_not_follow_training_set():
    """
    Raise ValueError if `last_window` index does not start at the end 
    of the index seen by the forecaster.
    """
    y_test = pd.Series(data=y_datetime.to_numpy())
    y_test.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    lw_test = pd.Series(data=y_lw_datetime.to_numpy())
    lw_test.index = pd.date_range(start='2022-03-01', periods=50, freq='D')

    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 1)))
    forecaster.fit(y=y_test)

    err_msg = re.escape(
        "To make predictions unrelated to the original data, `last_window` "
        "has to start at the end of the index seen by the forecaster.\n"
        "    Series last index         : 2022-02-19 00:00:00.\n"
        "    Expected index            : 2022-02-20 00:00:00.\n"
        "    `last_window` index start : 2022-03-01 00:00:00."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict(steps=5, last_window=lw_test)


def test_predict_ValueError_when_last_window_exog_index_does_not_follow_training_set():
    """
    Raise ValueError if `last_window_exog` index does not start at the end 
    of the index seen by the forecaster.
    """
    y_test = pd.Series(data=y_datetime.to_numpy())
    y_test.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    lw_test = pd.Series(data=y_lw_datetime.to_numpy())
    lw_test.index = pd.date_range(start='2022-02-20', periods=50, freq='D')
    
    exog_test = pd.Series(data=exog_datetime.to_numpy(), name='exog')
    exog_test.index = pd.date_range(start='2022-01-01', periods=50, freq='D', name='exog')
    exog_pred_test = pd.Series(data=exog_predict_datetime.to_numpy(), name='exog')
    exog_pred_test.index = pd.date_range(start='2022-04-11', periods=10, freq='D', name='exog')
    lw_exog_test = pd.Series(data=exog_lw_datetime.to_numpy(), name='exog')
    lw_exog_test.index = pd.date_range(start='2022-03-01', periods=50, freq='D', name='exog')

    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 1)))
    forecaster.fit(y=y_test, exog=exog_test)

    err_msg = re.escape(
        "To make predictions unrelated to the original data, `last_window_exog` "
        "has to start at the end of the index seen by the forecaster.\n"
        "    Series last index              : 2022-02-19 00:00:00.\n"
        "    Expected index                 : 2022-02-20 00:00:00.\n"
        "    `last_window_exog` index start : 2022-03-01 00:00:00."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict(
            steps            = 5, 
            exog             = exog_pred_test, 
            last_window      = lw_test,
            last_window_exog = lw_exog_test
        )


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1, 0, 1), 
                            'seasonal_order': (0, 0, 0, 0)},
                            {'win': [0.89534355, 0.88228868, 0.86942417, 0.85674723, 0.84425513],
                            'linux': [0.84149149, 0.82924261, 0.81717202, 0.80527714, 0.7935554]}),
                          ({'order': (1, 1, 1), 
                            'seasonal_order': (1, 1, 1, 2)},
                            {'win': [0.8079287 , 0.8570154 , 0.82680235, 0.88869186, 0.85565138],
                            'linux': [0.8079287 , 0.8570154 , 0.82680235, 0.88869186, 0.85565138]}
                            )])
def test_predict_output_ForecasterStats_with_last_window(kwargs, data):
    """
    Test predict output of ForecasterStats with `last_window`.
    """
    system = "win" if platform.system() == "Windows" else 'linux'
    
    forecaster = ForecasterStats(
                     estimator = Sarimax(maxiter=1000, method='cg', disp=False, **kwargs)
                 )
    forecaster.fit(y=y_datetime)
    predictions = forecaster.predict(steps=5, last_window=y_lw_datetime)
    expected = pd.Series(
                   data  = data[system],
                   index = pd.date_range(start='2100', periods=5, freq='YE'),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1, 0, 1), 
                            'seasonal_order': (0, 0, 0, 0)}, 
                            [0.70551392, 0.68955739, 0.72525104, 0.74304522, 0.75825646]), 
                          ({'order': (1, 1, 1), 
                            'seasonal_order': (1, 1, 1, 2)}, 
                            [0.94305077, 1.06784624, 1.17677373, 1.30237101, 1.37057143])])
def test_predict_output_ForecasterStats_with_last_window_and_exog(kwargs, data):
    """
    Test predict output of ForecasterStats with exogenous variables and `last_window`.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(maxiter=1000, method='cg', disp=False, **kwargs)
                 )
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    predictions = forecaster.predict(
                      steps            = 5, 
                      exog             = exog_lw_predict_datetime, 
                      last_window      = y_lw_datetime, 
                      last_window_exog = exog_lw_datetime
                  )

    expected = pd.Series(
                   data  = data,
                   index = pd.date_range(start='2100', periods=5, freq='YE'),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected, atol=0.0005)


@pytest.mark.parametrize("kwargs, data", 
                         [({'order': (1, 0, 1), 
                            'seasonal_order': (0, 0, 0, 0)}, 
                            [0.81663903, 0.77783205, 0.80523981, 0.85467197, 0.86644466])
                        ])
def test_predict_output_ForecasterStats_with_last_window_and_exog_and_transformers(kwargs, data):
    """
    Test predict output of ForecasterStats with exogenous variables, `last_window`
    and transformers.
    """
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterStats(
                     estimator        = Sarimax(maxiter=1000, method='cg', disp=False, **kwargs), 
                     transformer_y    = StandardScaler(),
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y_datetime, exog=df_exog_datetime)
    predictions = forecaster.predict(
                      steps            = 5, 
                      exog             = df_exog_lw_predict_datetime, 
                      last_window      = y_lw_datetime, 
                      last_window_exog = df_exog_lw_datetime
                  )

    expected = pd.Series(
                   data  = data,
                   index = pd.date_range(start='2100', periods=5, freq='YE'),
                   name  = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected, atol=0.0001)


@pytest.mark.parametrize("y          , idx", 
                         [(y         , pd.RangeIndex(start=0, stop=50)), 
                          (y_datetime, pd.date_range(start='2000', periods=50, freq='YE'))], 
                         ids = lambda values: f'y, index: {type(values)}')
def test_predict_ForecasterStats_updates_extended_index_twice(y, idx):
    """
    Test forecaster.extended_index_ is updated when using predict twice.
    """
    y_fit = y.iloc[:30].copy()

    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 0)))
    forecaster.fit(y=y_fit)

    lw_1 = y.iloc[30:40].copy()
    forecaster.predict(steps=5, last_window=lw_1)
    result_1 = forecaster.extended_index_.copy()
    expected_1 = idx[:40]

    lw_2 = y.iloc[40:].copy()
    forecaster.predict(steps=5, last_window=lw_2)

    pd.testing.assert_index_equal(result_1, expected_1)
    pd.testing.assert_index_equal(forecaster.extended_index_, idx)


@pytest.mark.parametrize(
    "estimator, expected_data",
    [
        (
            Sarimax(order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), maxiter=1000, method='cg', disp=False),
            [0.60535333, 0.59654171, 0.58785836, 0.5793014, 0.570869]
        ),
        (
            Arar(),
            [0.65451694, 0.69369274, 0.8018875, 0.82157326, 0.87868702]
        ),
        (
            Ets(trend='add', seasonal=None),
            [0.60498897, 0.60498083, 0.60497432, 0.60496911, 0.60496495]
        ),
    ],
    ids=['Sarimax', 'Arar', 'Ets']
)
def test_predict_output_ForecasterStats_different_estimators(estimator, expected_data, y_datetime=y_datetime):
    """
    Test predict output of ForecasterStats with different estimators (Sarimax, Arar, Ets).
    """
    forecaster = ForecasterStats(estimator=estimator)
    forecaster.fit(y=y_datetime)
    predictions = forecaster.predict(steps=5)
    expected = pd.Series(
        data=expected_data,
        index=pd.date_range(start='2050', periods=5, freq='YE'),
        name='pred'
    )
    
    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize(
    "estimator, expected_data",
    [
        (
            Sarimax(order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), maxiter=1000, method='cg', disp=False),
            [
                0.6053533305780869,
                0.5965417125355235,
                0.5878583577874791,
                0.5793013993133179,
                0.5708689972690493,
                0.5625593385917403,
                0.5543706366096868,
                0.5463011306582577,
                0.5383490857013329,
                0.5305127919582501,
            ]
        ),
        (
            Arar(max_ar_depth=26, max_lag=40),
            [
                0.6545169424155248,
                0.693692742846921,
                0.8018875044998569,
                0.8215732635739413,
                0.8786870196348286,
                0.8879849576779746,
                1.0173957185481672,
                1.022217168929162,
                0.5688093026635382,
                0.6336566288056642,
            ]
        ),
        (
            Ets(model='AAN', damped=False),
            [
                0.6812318294683052,
                0.679950896354816,
                0.6786699632413268,
                0.6773890301278376,
                0.6761080970143484,
                0.6748271639008592,
                0.67354623078737,
                0.6722652976738808,
                0.6709843645603916,
                0.6697034314469024,
            ]
        ),
        (
            ARIMA(p=4, d=1, q=1),
            [
                0.68329597,
                0.7153698,
                0.72118068,
                0.71875056,
                0.70465789,
                0.68932917,
                0.68087982,
                0.67748653,
                0.67828572,
                0.68193945,
            ]
        ),
    ],
    ids=['Sarimax', 'Arar', 'Ets', 'aeon_ARIMA']
)
def test_predict_output_ForecasterStats_with_multiple_estimators(estimator, expected_data, y=y):
    """
    Test output of predict when using different estimators (Sarimax, Arar, Ets, aeon ARIMA) in ForecasterStats.
    """
    y = y.copy()
    y.index = pd.date_range(start="2000-01-01", periods=len(y), freq="D")
    forecaster = ForecasterStats(estimator=estimator)
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=10)

    expected_results = pd.Series(
        data=expected_data,
        name="pred",
        index=pd.date_range(start="2000-02-20", periods=10, freq="D"),
    )

    pd.testing.assert_series_equal(predictions, expected_results)
