# Unit test _create_predict_inputs ForecasterSarimax
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.sarimax import Sarimax
from skforecast.recursive import ForecasterSarimax
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


def test_create_predict_inputs_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 1, 1)))

    err_msg = re.escape(
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster._create_predict_inputs(steps=5)


def test_create_predict_inputs_ValueError_when_ForecasterSarimax_last_window_exog_is_not_None_and_last_window_is_not_provided():
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
        forecaster._create_predict_inputs(
            steps            = 5, 
            exog             = exog_predict, 
            last_window      = None, 
            last_window_exog = exog
        )


def test_create_predict_inputs_ValueError_when_ForecasterSarimax_last_window_exog_is_None_and_included_exog_is_true():
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
        forecaster._create_predict_inputs(
            steps            = 5, 
            exog             = exog_lw_predict, 
            last_window      = y_lw, 
            last_window_exog = None
        )


def test_create_predict_inputs_output_ForecasterSarimax():
    """
    Test _create_predict_inputs output of ForecasterSarimax.
    """
    forecaster = ForecasterSarimax(regressor = Sarimax())
    forecaster.fit(y=y)
    results = forecaster._create_predict_inputs(steps=5)

    expected = (
        None,
        None,
        None
    )
    
    assert results == expected


def test_create_predict_inputs_output_ForecasterSarimax_with_exog():
    """
    Test _create_predict_inputs output of ForecasterSarimax with exogenous variables.
    """
    forecaster = ForecasterSarimax(regressor = Sarimax())
    forecaster.fit(y=y, exog=exog)
    results = forecaster._create_predict_inputs(steps=5, exog=exog_predict)

    expected = (
        None,
        None,
        pd.DataFrame(
            {'exog': np.array([1.27501789, 1.31081724, 1.34284965, 1.37614005, 1.41412559])},
            index = pd.RangeIndex(start=50, stop=55, step=1)
        )
    )
    
    assert results[0] == expected[0]
    assert results[1] == expected[1]
    pd.testing.assert_frame_equal(results[2], expected[2])


def test_create_predict_inputs_output_ForecasterSarimax_with_transform_y():
    """
    Test _create_predict_inputs output of ForecasterSarimax with a 
    StandardScaler() as transformer_y.
    """        
    forecaster = ForecasterSarimax(
                     regressor     = Sarimax(),
                     transformer_y = StandardScaler()
                 )
    forecaster.fit(y=y)
    results = forecaster._create_predict_inputs(steps=5)

    expected = (
        None,
        None,
        None
    )
    
    assert results == expected


def test_create_predict_inputs_output_ForecasterSarimax_with_transform_y_and_transform_exog():
    """
    Test _create_predict_inputs output of ForecasterSarimax, StandardScaler
    as transformer_y and transformer_exog as transformer_exog.
    """    
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterSarimax(
                     regressor        = Sarimax(),
                     transformer_y    = StandardScaler(),
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y, exog=df_exog)
    results = forecaster._create_predict_inputs(steps=5, exog=df_exog_predict)

    expected = (
        None,
        None,
        pd.DataFrame(
            {'exog_1': np.array([-0.20710692, 0.10403839, 0.38244384, 0.67178294, 1.00192923]),
             'exog_2_a': np.array([1., 0., 1., 0., 1.]),
             'exog_2_b': np.array([0., 1., 0., 1., 0.])},
            index = pd.RangeIndex(start=50, stop=55, step=1)
        )
    )
    
    assert results[0] == expected[0]
    assert results[1] == expected[1]
    pd.testing.assert_frame_equal(results[2], expected[2])


def test_create_predict_inputs_ValueError_when_last_window_index_does_not_follow_training_set():
    """
    Raise ValueError if `last_window` index does not start at the end 
    of the index seen by the forecaster.
    """
    y_test = pd.Series(data=y_datetime.to_numpy())
    y_test.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    lw_test = pd.Series(data=y_lw_datetime.to_numpy())
    lw_test.index = pd.date_range(start='2022-03-01', periods=50, freq='D')

    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 0, 1)))
    forecaster.fit(y=y_test)

    err_msg = re.escape(
        (f"To make predictions unrelated to the original data, `last_window` "
         f"has to start at the end of the index seen by the forecaster.\n"
         f"    Series last index         : 2022-02-19 00:00:00.\n"
         f"    Expected index            : 2022-02-20 00:00:00.\n"
         f"    `last_window` index start : 2022-03-01 00:00:00.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_predict_inputs(steps=5, last_window=lw_test)


def test_create_predict_inputs_ValueError_when_last_window_exog_index_does_not_follow_training_set():
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

    forecaster = ForecasterSarimax(regressor=Sarimax(order=(1, 0, 1)))
    forecaster.fit(y=y_test, exog=exog_test)

    err_msg = re.escape(
        (f"To make predictions unrelated to the original data, `last_window_exog` "
         f"has to start at the end of the index seen by the forecaster.\n"
         f"    Series last index              : 2022-02-19 00:00:00.\n"
         f"    Expected index                 : 2022-02-20 00:00:00.\n"
         f"    `last_window_exog` index start : 2022-03-01 00:00:00.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_predict_inputs(
            steps            = 5, 
            exog             = exog_pred_test, 
            last_window      = lw_test,
            last_window_exog = lw_exog_test
        )


def test_create_predict_inputs_output_ForecasterSarimax_with_last_window():
    """
    Test _create_predict_inputs output of ForecasterSarimax with `last_window`.
    """    
    forecaster = ForecasterSarimax(regressor = Sarimax())
    forecaster.fit(y=y_datetime)
    results = forecaster._create_predict_inputs(steps=5, last_window=y_lw_datetime)

    expected = (
        pd.Series(
            np.array([0.59418877, 0.70775844, 0.71950195, 0.74432369, 0.80485511,
                      0.78854235, 0.9710894 , 0.84683354, 0.46382252, 0.48527317,
                      0.5280586 , 0.56233647, 0.5885704 , 0.66948036, 0.67799365,
                      0.76299549, 0.79972374, 0.77052192, 0.99438934, 0.80054443,
                      0.49055721, 0.52440799, 0.53664948, 0.55209054, 0.60336564,
                      0.68124538, 0.67807535, 0.79489265, 0.7846239 , 0.8130087 ,
                      0.9777323 , 0.89308148, 0.51269597, 0.65299589, 0.5739764 ,
                      0.63923842, 0.70387188, 0.77064824, 0.84618588, 0.89272889,
                      0.89789988, 0.94728069, 1.05070727, 0.96965567, 0.57329151,
                      0.61850684, 0.61899573, 0.66520922, 0.72652015, 0.85586494]),
            index = pd.date_range(start='2050', periods=50, freq='YE'),
            name = 'y'
        ),
        None,
        None
    )
    
    pd.testing.assert_series_equal(results[0], expected[0])
    assert results[1] == expected[1]
    assert results[2] == expected[2]


def test_create_predict_inputs_output_ForecasterSarimax_with_last_window_and_exog():
    """
    Test _create_predict_inputs output of ForecasterSarimax with exogenous 
    variables and `last_window`.
    """
    forecaster = ForecasterSarimax(regressor = Sarimax())
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    results = forecaster._create_predict_inputs(
                  steps            = 5, 
                  exog             = exog_lw_predict_datetime, 
                  last_window      = y_lw_datetime, 
                  last_window_exog = exog_lw_datetime
              )

    expected = (
        pd.Series(
            np.array([0.59418877, 0.70775844, 0.71950195, 0.74432369, 0.80485511,
                      0.78854235, 0.9710894 , 0.84683354, 0.46382252, 0.48527317,
                      0.5280586 , 0.56233647, 0.5885704 , 0.66948036, 0.67799365,
                      0.76299549, 0.79972374, 0.77052192, 0.99438934, 0.80054443,
                      0.49055721, 0.52440799, 0.53664948, 0.55209054, 0.60336564,
                      0.68124538, 0.67807535, 0.79489265, 0.7846239 , 0.8130087 ,
                      0.9777323 , 0.89308148, 0.51269597, 0.65299589, 0.5739764 ,
                      0.63923842, 0.70387188, 0.77064824, 0.84618588, 0.89272889,
                      0.89789988, 0.94728069, 1.05070727, 0.96965567, 0.57329151,
                      0.61850684, 0.61899573, 0.66520922, 0.72652015, 0.85586494]),
            index = pd.date_range(start='2050', periods=50, freq='YE'),
            name = 'y'
        ),
        pd.DataFrame(
            {'exog': np.array([1.27501789, 1.31081724, 1.34284965, 1.37614005, 1.41412559,
                               1.45299631, 1.5056625 , 1.53112882, 1.47502858, 1.4111122 ,
                               1.35901545, 1.27726486, 1.22561223, 1.2667438 , 1.3052879 ,
                               1.35227527, 1.39975273, 1.43614303, 1.50112483, 1.52563498,
                               1.47114733, 1.41608418, 1.36930969, 1.28084993, 1.24141417,
                               1.27955181, 1.31028528, 1.36193391, 1.40844058, 1.4503692 ,
                               1.50966658, 1.55266781, 1.49622847, 1.46990287, 1.42209641,
                               1.35439763, 1.31655571, 1.36814617, 1.40678416, 1.47053466,
                               1.52226695, 1.57094872, 1.62696052, 1.65165448, 1.587767  ,
                               1.5318884 , 1.4662314 , 1.38913179, 1.34050469, 1.39701938])},
            index = pd.date_range(start='2050', periods=50, freq='YE')
        ),
        pd.DataFrame(
            {'exog': np.array([1.44651487, 1.48776549, 1.54580785, 1.58822301, 1.6196549])},
            index = pd.date_range(start='2100', periods=5, freq='YE')
        )
    )
    
    pd.testing.assert_series_equal(results[0], expected[0])
    pd.testing.assert_frame_equal(results[1], expected[1])
    pd.testing.assert_frame_equal(results[2], expected[2])


def test_create_predict_inputs_output_ForecasterSarimax_with_last_window_and_exog_and_transformers():
    """
    Test _create_predict_inputs output of ForecasterSarimax with exogenous 
    variables, `last_window` and transformers.
    """
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterSarimax(
                     regressor        = Sarimax(), 
                     transformer_y    = StandardScaler(),
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y_datetime, exog=df_exog_datetime)
    results = forecaster._create_predict_inputs(
                  steps            = 5, 
                  exog             = df_exog_lw_predict_datetime, 
                  last_window      = y_lw_datetime, 
                  last_window_exog = df_exog_lw_datetime
              )

    expected = (
        pd.Series(
            np.array([-0.07016482,  0.67641597,  0.75361509,  0.91678748,  1.31470706,
                       1.20747074,  2.4074929 ,  1.59066357, -0.92716245, -0.78615083,
                      -0.50488929, -0.27955449, -0.10709869,  0.42478471,  0.48074911,
                       1.03953158,  1.28097462,  1.0890086 ,  2.56066133,  1.28636965,
                      -0.75141477, -0.52888756, -0.44841483, -0.34690886, -0.0098382 ,
                       0.50212523,  0.48128619,  1.24921615,  1.18171176,  1.36830688,
                       2.45116179,  1.89468685, -0.60587967,  0.31641961, -0.20303627,
                       0.22598118,  0.65086662,  1.089839  ,  1.586406  ,  1.892369  ,
                       1.9263619 ,  2.25097994,  2.9308824 ,  2.39806789, -0.20753858,
                       0.08969656,  0.09291041,  0.39670722,  0.79975112,  1.65003391]),
            index = pd.date_range(start='2050', periods=50, freq='YE'),
            name = 'y'
        ),
        pd.DataFrame(
            data = np.array([[-0.20710692,  1.        ,  0.        ],
                             [ 0.10403839,  0.        ,  1.        ],
                             [ 0.38244384,  1.        ,  0.        ],
                             [ 0.67178294,  0.        ,  1.        ],
                             [ 1.00192923,  1.        ,  0.        ],
                             [ 1.33976895,  0.        ,  1.        ],
                             [ 1.79751016,  1.        ,  0.        ],
                             [ 2.01884731,  0.        ,  1.        ],
                             [ 1.5312595 ,  1.        ,  0.        ],
                             [ 0.97573875,  0.        ,  1.        ],
                             [ 0.52294675,  1.        ,  0.        ],
                             [-0.18757768,  0.        ,  1.        ],
                             [-0.63650967,  1.        ,  0.        ],
                             [-0.27902008,  0.        ,  1.        ],
                             [ 0.05598086,  1.        ,  0.        ],
                             [ 0.46436537,  0.        ,  1.        ],
                             [ 0.87700942,  1.        ,  0.        ],
                             [ 1.19329089,  0.        ,  1.        ],
                             [ 1.7580716 ,  1.        ,  0.        ],
                             [ 1.97109833,  0.        ,  1.        ],
                             [ 1.49752613,  1.        ,  0.        ],
                             [ 1.01895206,  0.        ,  1.        ],
                             [ 0.61241777,  1.        ,  0.        ],
                             [-0.15641852,  0.        ,  1.        ],
                             [-0.4991692 ,  1.        ,  0.        ],
                             [-0.16770096,  0.        ,  1.        ],
                             [ 0.09941493,  1.        ,  0.        ],
                             [ 0.54831216,  0.        ,  1.        ],
                             [ 0.95251872,  1.        ,  0.        ],
                             [ 1.31693577,  0.        ,  1.        ],
                             [ 1.83231109,  1.        ,  0.        ],
                             [ 2.20605059,  0.        ,  1.        ],
                             [ 1.71551554,  1.        ,  0.        ],
                             [ 1.48671007,  0.        ,  1.        ],
                             [ 1.07120656,  1.        ,  0.        ],
                             [ 0.48281158,  0.        ,  1.        ],
                             [ 0.15391354,  1.        ,  0.        ],
                             [ 0.60230519,  0.        ,  1.        ],
                             [ 0.93812216,  1.        ,  0.        ],
                             [ 1.49220119,  0.        ,  1.        ],
                             [ 1.94182554,  1.        ,  0.        ],
                             [ 2.3649367 ,  0.        ,  1.        ],
                             [ 2.85175584,  1.        ,  0.        ],
                             [ 3.06638012,  0.        ,  1.        ],
                             [ 2.51111055,  1.        ,  0.        ],
                             [ 2.0254491 ,  0.        ,  1.        ],
                             [ 1.45479998,  1.        ,  0.        ],
                             [ 0.78469893,  0.        ,  1.        ],
                             [ 0.36206293,  1.        ,  0.        ],
                             [ 0.85325287,  0.        ,  1.        ]]),
            columns = ['exog_1', 'exog_2_a', 'exog_2_b'],
            index = pd.date_range(start='2050', periods=50, freq='YE')
        ),
        pd.DataFrame(
            data = np.array([[1.28343637, 1.        , 0.        ],
                             [1.64196067, 0.        , 1.        ],
                             [2.14642815, 1.        , 0.        ],
                             [2.5150739 , 0.        , 1.        ],
                             [2.78826001, 1.        , 0.        ]]),
            columns = ['exog_1', 'exog_2_a', 'exog_2_b'],
            index = pd.date_range(start='2100', periods=5, freq='YE')
        )
    )
    
    pd.testing.assert_series_equal(results[0], expected[0])
    pd.testing.assert_frame_equal(results[1], expected[1])
    pd.testing.assert_frame_equal(results[2], expected[2])