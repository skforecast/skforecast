# Unit test _direct_predict ForecasterDirectMultiVariate
# ==============================================================================
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from skforecast.direct import ForecasterDirectMultiVariate

# Fixtures
from .fixtures_forecaster_direct_multivariate import series
from .fixtures_forecaster_direct_multivariate import exog as exog_categorical
from .fixtures_forecaster_direct_multivariate import exog_predict as exog_predict_categorical


def test_direct_predict_output_when_estimator_is_LinearRegression():
    """
    Test _direct_predict output when using LinearRegression as estimator.
    """
    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l1', lags=5, steps=5,
        transformer_series=None
    )
    forecaster.fit(series=series)
    Xs, _, steps, _ = forecaster._create_predict_inputs(steps=5)
    predictions = forecaster._direct_predict(steps=steps, Xs=Xs)

    expected = np.array([0.60056539, 0.42924504, 0.34173573, 0.44231236, 0.40133213])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_direct_predict_output_when_estimator_is_LinearRegression_with_exog():
    """
    Test _direct_predict output when using LinearRegression as estimator with exog.
    """
    exog = exog_categorical[['exog_1']]
    exog_predict = exog_predict_categorical[['exog_1']]

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l1', lags=5, steps=5,
        transformer_series=None
    )
    forecaster.fit(series=series, exog=exog)
    Xs, _, steps, _ = forecaster._create_predict_inputs(steps=5, exog=exog_predict)
    predictions = forecaster._direct_predict(steps=steps, Xs=Xs)

    expected = np.array([0.57527255, 0.41857122, 0.53297714, 0.51553216, 0.45969602])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_direct_predict_output_when_estimator_is_LGBMRegressor():
    """
    Test _direct_predict output when using LGBMRegressor as estimator.
    """
    forecaster = ForecasterDirectMultiVariate(
        estimator=LGBMRegressor(verbose=-1, random_state=123), level='l1', lags=5, steps=5,
        transformer_series=None
    )
    forecaster.fit(series=series)
    Xs, _, steps, _ = forecaster._create_predict_inputs(steps=5)
    predictions = forecaster._direct_predict(steps=steps, Xs=Xs)

    expected = np.array([0.56942973, 0.51214016, 0.40635497, 0.48158903, 0.49029841])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_direct_predict_output_when_estimator_is_LGBMRegressor_with_exog():
    """
    Test _direct_predict output when using LGBMRegressor as estimator with exog.
    """
    exog = exog_categorical[['exog_1']]
    exog_predict = exog_predict_categorical[['exog_1']]

    forecaster = ForecasterDirectMultiVariate(
        estimator=LGBMRegressor(verbose=-1, random_state=123), level='l1', lags=5, steps=5,
        transformer_series=None
    )
    forecaster.fit(series=series, exog=exog)
    Xs, _, steps, _ = forecaster._create_predict_inputs(steps=5, exog=exog_predict)
    predictions = forecaster._direct_predict(steps=steps, Xs=Xs)

    expected = np.array([0.63485683, 0.49703737, 0.46674625, 0.54557163, 0.51988782])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_direct_predict_output_when_estimator_is_RandomForestRegressor():
    """
    Test _direct_predict output when using RandomForestRegressor as estimator.
    """
    forecaster = ForecasterDirectMultiVariate(
        estimator=RandomForestRegressor(n_estimators=10, random_state=123),
        level='l1', lags=5, steps=5, transformer_series=None
    )
    forecaster.fit(series=series)
    Xs, _, steps, _ = forecaster._create_predict_inputs(steps=5)
    predictions = forecaster._direct_predict(steps=steps, Xs=Xs)

    expected = np.array([0.57157055, 0.47276495, 0.49586004, 0.58315325, 0.40988631])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_direct_predict_output_when_estimator_is_RandomForestRegressor_with_exog():
    """
    Test _direct_predict output when using RandomForestRegressor as estimator with exog.
    """
    exog = exog_categorical[['exog_1']]
    exog_predict = exog_predict_categorical[['exog_1']]

    forecaster = ForecasterDirectMultiVariate(
        estimator=RandomForestRegressor(n_estimators=10, random_state=123),
        level='l1', lags=5, steps=5, transformer_series=None
    )
    forecaster.fit(series=series, exog=exog)
    Xs, _, steps, _ = forecaster._create_predict_inputs(steps=5, exog=exog_predict)
    predictions = forecaster._direct_predict(steps=steps, Xs=Xs)

    expected = np.array([0.58637475, 0.46978312, 0.4462907 , 0.57861253, 0.40648166])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_direct_predict_output_when_estimator_is_XGBRegressor():
    """
    Test _direct_predict output when using XGBRegressor as estimator.
    """
    forecaster = ForecasterDirectMultiVariate(
        estimator=XGBRegressor(n_estimators=10, random_state=123, verbosity=0),
        level='l1', lags=5, steps=5, transformer_series=None
    )
    forecaster.fit(series=series)
    Xs, _, steps, _ = forecaster._create_predict_inputs(steps=5)
    predictions = forecaster._direct_predict(steps=steps, Xs=Xs)

    expected = np.array([0.79249394, 0.36357051, 0.46422222, 0.52599514, 0.39951146])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_direct_predict_output_when_estimator_is_XGBRegressor_with_exog():
    """
    Test _direct_predict output when using XGBRegressor as estimator with exog.
    """
    exog = exog_categorical[['exog_1']]
    exog_predict = exog_predict_categorical[['exog_1']]

    forecaster = ForecasterDirectMultiVariate(
        estimator=XGBRegressor(n_estimators=10, random_state=123, verbosity=0),
        level='l1', lags=5, steps=5, transformer_series=None
    )
    forecaster.fit(series=series, exog=exog)
    Xs, _, steps, _ = forecaster._create_predict_inputs(steps=5, exog=exog_predict)
    predictions = forecaster._direct_predict(steps=steps, Xs=Xs)

    expected = np.array([0.73904371, 0.36545673, 0.47884795, 0.64205378, 0.53937113])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_direct_predict_output_when_estimator_is_LinearRegression_with_list_steps():
    """
    Test _direct_predict output when using LinearRegression as estimator with 
    list of interspersed steps [1, 3, 5].
    """
    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l1', lags=5, steps=5,
        transformer_series=None
    )
    forecaster.fit(series=series)
    Xs, _, steps, _ = forecaster._create_predict_inputs(steps=[1, 3, 5])
    predictions = forecaster._direct_predict(steps=steps, Xs=Xs)

    expected = np.array([0.60056539, 0.34173573, 0.40133213])
    
    assert steps == [1, 3, 5]
    np.testing.assert_array_almost_equal(predictions, expected)
