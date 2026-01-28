# Unit test _direct_predict ForecasterDirect
# ==============================================================================
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from skforecast.direct import ForecasterDirect

# Fixtures
from .fixtures_forecaster_direct import y as y_categorical
from .fixtures_forecaster_direct import exog as exog_categorical
from .fixtures_forecaster_direct import exog_predict as exog_predict_categorical


def test_direct_predict_output_when_estimator_is_LinearRegression():
    """
    Test _direct_predict output when using LinearRegression as estimator.
    """
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=5
    )
    forecaster.fit(y=y_categorical)
    Xs, _, steps, _ = forecaster._create_predict_inputs(steps=5)
    predictions = forecaster._direct_predict(steps=steps, Xs=Xs)

    expected = np.array([0.5531936 , 0.48305122, 0.44981279, 0.4134158 , 0.41986132])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_direct_predict_output_when_estimator_is_LinearRegression_with_exog():
    """
    Test _direct_predict output when using LinearRegression as estimator with exog.
    """
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=5
    )
    forecaster.fit(y=y_categorical, exog=exog_categorical)
    Xs, _, steps, _ = forecaster._create_predict_inputs(steps=5, exog=exog_predict_categorical)
    predictions = forecaster._direct_predict(steps=steps, Xs=Xs)

    expected = np.array([0.65818211, 0.42775674, 0.43971937, 0.39334849, 0.49343381])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_direct_predict_output_when_estimator_is_LGBMRegressor():
    """
    Test _direct_predict output when using LGBMRegressor as estimator.
    """
    forecaster = ForecasterDirect(
        estimator=LGBMRegressor(verbose=-1, random_state=123), lags=5, steps=5
    )
    forecaster.fit(y=y_categorical)
    Xs, _, steps, _ = forecaster._create_predict_inputs(steps=5)
    predictions = forecaster._direct_predict(steps=steps, Xs=Xs)

    expected = np.array([0.53160297, 0.53912094, 0.43082739, 0.44087501, 0.39199316])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_direct_predict_output_when_estimator_is_LGBMRegressor_with_exog():
    """
    Test _direct_predict output when using LGBMRegressor as estimator with exog.
    """
    forecaster = ForecasterDirect(
        estimator=LGBMRegressor(verbose=-1, random_state=123), lags=5, steps=5
    )
    forecaster.fit(y=y_categorical, exog=exog_categorical)
    Xs, _, steps, _ = forecaster._create_predict_inputs(steps=5, exog=exog_predict_categorical)
    predictions = forecaster._direct_predict(steps=steps, Xs=Xs)

    expected = np.array([0.56249088, 0.52088404, 0.41589238, 0.45968771, 0.42306319])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_direct_predict_output_when_estimator_is_RandomForestRegressor():
    """
    Test _direct_predict output when using RandomForestRegressor as estimator.
    """
    forecaster = ForecasterDirect(
        estimator=RandomForestRegressor(n_estimators=10, random_state=123),
        lags=5,
        steps=5
    )
    forecaster.fit(y=y_categorical)
    Xs, _, steps, _ = forecaster._create_predict_inputs(steps=5)
    predictions = forecaster._direct_predict(steps=steps, Xs=Xs)

    expected = np.array([0.58721155, 0.33402273, 0.47910935, 0.59901173, 0.40474317])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_direct_predict_output_when_estimator_is_RandomForestRegressor_with_exog():
    """
    Test _direct_predict output when using RandomForestRegressor as estimator with exog.
    """
    forecaster = ForecasterDirect(
        estimator=RandomForestRegressor(n_estimators=10, random_state=123),
        lags=5,
        steps=5
    )
    forecaster.fit(y=y_categorical, exog=exog_categorical)
    Xs, _, steps, _ = forecaster._create_predict_inputs(steps=5, exog=exog_predict_categorical)
    predictions = forecaster._direct_predict(steps=steps, Xs=Xs)

    expected = np.array([0.76957028, 0.25115545, 0.57398305, 0.54794638, 0.47204635])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_direct_predict_output_when_estimator_is_XGBRegressor():
    """
    Test _direct_predict output when using XGBRegressor as estimator.
    """
    forecaster = ForecasterDirect(
        estimator=XGBRegressor(n_estimators=10, random_state=123, verbosity=0),
        lags=5,
        steps=5
    )
    forecaster.fit(y=y_categorical)
    Xs, _, steps, _ = forecaster._create_predict_inputs(steps=5)
    predictions = forecaster._direct_predict(steps=steps, Xs=Xs)

    expected = np.array([0.71678525, 0.24216469, 0.50616848, 0.6219855 , 0.43826309])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_direct_predict_output_when_estimator_is_XGBRegressor_with_exog():
    """
    Test _direct_predict output when using XGBRegressor as estimator with exog.
    """
    forecaster = ForecasterDirect(
        estimator=XGBRegressor(n_estimators=10, random_state=123, verbosity=0),
        lags=5,
        steps=5
    )
    forecaster.fit(y=y_categorical, exog=exog_categorical)
    Xs, _, steps, _ = forecaster._create_predict_inputs(steps=5, exog=exog_predict_categorical)
    predictions = forecaster._direct_predict(steps=steps, Xs=Xs)

    expected = np.array([0.72550768, 0.25783226, 0.52587408, 0.56275314, 0.33653745])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_direct_predict_output_when_estimator_is_LinearRegression_with_list_steps():
    """
    Test _direct_predict output when using LinearRegression as estimator with 
    list of interspersed steps [1, 3, 5].
    """
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=5, steps=5
    )
    forecaster.fit(y=y_categorical)
    Xs, _, steps, _ = forecaster._create_predict_inputs(steps=[1, 3, 5])
    predictions = forecaster._direct_predict(steps=steps, Xs=Xs)

    expected = np.array([0.5531936 , 0.44981279, 0.41986132])
    
    assert steps == [1, 3, 5]
    np.testing.assert_array_almost_equal(predictions, expected)
