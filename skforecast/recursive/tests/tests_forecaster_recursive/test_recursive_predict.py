# Unit test _recursive_predict ForecasterRecursive
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from skforecast.preprocessing import RollingFeatures, CalendarFeatures
from skforecast.recursive import ForecasterRecursive

# Fixtures
from .fixtures_forecaster_recursive import y
from .fixtures_forecaster_recursive import exog
from .fixtures_forecaster_recursive import exog_predict


# Subclasses with different class names to force the generic `else` prediction
# branch in _recursive_predict, used as the reference (slow) path vs the fast path
# the uses the fast prediction methods for supported estimators (RandomForestRegressor,
# DecisionTreeRegressor, LGBMRegressor, XGBRegressor, and linear models that inherit from sklearn's LinearModel).
class _SlowRF(RandomForestRegressor):
    pass


class _SlowDT(DecisionTreeRegressor):
    pass


class _SlowLGBM(LGBMRegressor):
    pass


class _SlowXGB(XGBRegressor):
    pass


class _SlowLinear(BaseEstimator, RegressorMixin):
    """
    Wraps a Ridge estimator without inheriting from sklearn's LinearModel, so
    that `isinstance(estimator, LinearModel)` is False and the generic `else`
    branch is used in _recursive_predict.
    """
    def __init__(self, alpha=1.0, random_state=None):
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        self._model = Ridge(alpha=self.alpha, random_state=self.random_state)
        self._model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self._model.predict(X)


def test_recursive_predict_output_when_estimator_is_LinearRegression():
    """
    Test _recursive_predict output when using LinearRegression as estimator.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)))

    last_window_values, exog_values, _, _, _, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    predictions = forecaster._recursive_predict(
                      steps              = 5,
                      last_window_values = last_window_values,
                      exog_values        = exog_values
                  )
    
    expected = np.array([50., 51., 52., 53., 54.])
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_when_estimator_is_Ridge_StandardScaler():
    """
    Test _recursive_predict output when using Ridge as estimator and
    StandardScaler.
    """
    forecaster = ForecasterRecursive(
                     estimator     = Ridge(random_state=123),
                     lags          = [1, 5],
                     transformer_y = StandardScaler()
                 )
    forecaster.fit(y=pd.Series(np.arange(50), name='y'))

    last_window_values, exog_values, _, _, _, _ = (
        forecaster._create_predict_inputs(steps=5)
    )
    predictions = forecaster._recursive_predict(
                      steps              = 5,
                      last_window_values = last_window_values,
                      exog_values        = exog_values
                  )
    
    expected = np.array([1.745476, 1.803196, 1.865844, 1.930923, 1.997202])

    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_window_features_LGBMRegressor():
    """
    Test _recursive_predict output with window features.
    """
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=4)
    forecaster = ForecasterRecursive(
        LGBMRegressor(verbose=-1), lags=3, window_features=rolling
    )
    forecaster.fit(y=y, exog=exog)

    last_window_values, exog_values, _, _, _, _ = (
        forecaster._create_predict_inputs(steps=10, exog=exog_predict)
    )
    predictions = forecaster._recursive_predict(
                      steps              = 10,
                      last_window_values = last_window_values,
                      exog_values        = exog_values
                  )
    
    expected = np.array(
                   [0.584584, 0.487441, 0.483098, 0.483098, 0.580241, 
                    0.584584, 0.584584, 0.487441, 0.483098, 0.483098]
               )
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_window_features_XGBRegressor():
    """
    Test _recursive_predict output with window features.
    """
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=4)
    forecaster = ForecasterRecursive(
        XGBRegressor(random_state=123, verbosity=0), lags=3, window_features=rolling
    )
    forecaster.fit(y=y, exog=exog)

    last_window_values, exog_values, _, _, _, _ = (
        forecaster._create_predict_inputs(steps=10, exog=exog_predict)
    )
    predictions = forecaster._recursive_predict(
                      steps              = 10,
                      last_window_values = last_window_values,
                      exog_values        = exog_values
                  )
    
    expected = np.array(
                   [0.537775, 0.520587, 0.611477, 0.683535, 0.685327, 
                    0.70897 , 0.605805, 0.503862, 0.612269, 0.68454 ]
               )
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_output_with_two_window_features():
    """
    Test _recursive_predict output with 2 window features.
    """
    rolling = RollingFeatures(stats=['mean'], window_sizes=4)
    rolling_2 = RollingFeatures(stats=['median'], window_sizes=4)
    forecaster = ForecasterRecursive(
        LGBMRegressor(verbose=-1), lags=3, window_features=[rolling, rolling_2]
    )
    forecaster.fit(y=y, exog=exog)

    last_window_values, exog_values, _, _, _, _ = (
        forecaster._create_predict_inputs(steps=10, exog=exog_predict)
    )
    predictions = forecaster._recursive_predict(
                      steps              = 10,
                      last_window_values = last_window_values,
                      exog_values        = exog_values
                  )
    
    expected = np.array(
                   [0.584584, 0.487441, 0.483098, 0.483098, 0.580241, 
                    0.584584, 0.584584, 0.487441, 0.483098, 0.483098]
               )
    
    np.testing.assert_array_almost_equal(predictions, expected)


def test_recursive_predict_fast_path_RandomForestRegressor_matches_generic_path():
    """
    Test that the fast prediction path for RandomForestRegressor (using
    tree_.predict directly) produces the same results as the generic sklearn
    predict path (forced via a subclass that bypasses the fast-path branch).
    """
    forecaster_fast = ForecasterRecursive(
        RandomForestRegressor(n_estimators=10, random_state=123), lags=3
    )
    forecaster_fast.fit(y=y)

    forecaster_slow = ForecasterRecursive(
        _SlowRF(n_estimators=10, random_state=123), lags=3
    )
    forecaster_slow.fit(y=y)

    last_window_values_fast, exog_values_fast, _, _, _, _ = (
        forecaster_fast._create_predict_inputs(steps=10)
    )
    last_window_values_slow, exog_values_slow, _, _, _, _ = (
        forecaster_slow._create_predict_inputs(steps=10)
    )

    predictions_fast = forecaster_fast._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_fast,
                           exog_values        = exog_values_fast
                       )
    predictions_slow = forecaster_slow._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_slow,
                           exog_values        = exog_values_slow
                       )

    np.testing.assert_array_almost_equal(predictions_fast, predictions_slow)


def test_recursive_predict_fast_path_RandomForestRegressor_with_exog_window_features_and_calendar():
    """
    Test that the fast prediction path for RandomForestRegressor matches the
    generic sklearn path when exog, window features and calendar features are used.
    Also checked that the same results are obtained when calendar features are used 
    with the CalendarFeatures class or when calendar features are manually created 
    and passed as exog.
    """

    y_datetime = y.copy()
    y_datetime.index = pd.date_range(start='2020-01-01', periods=len(y), freq='D')
    exog_datetime = exog.copy()
    exog_datetime.index = y_datetime.index
    exog_predict_datetime = exog_predict.copy()
    exog_predict_datetime.index = pd.date_range(
        start=y_datetime.index[-1] + pd.Timedelta(days=1), periods=len(exog_predict), freq='D'
    )

    exog_calendar = exog_datetime.to_frame()
    exog_calendar['day_of_week'] = exog_calendar.index.dayofweek
    exog_calendar['weekend'] = exog_calendar['day_of_week'] >= 5
    exog_predict_calendar = exog_predict_datetime.to_frame()
    exog_predict_calendar['day_of_week'] = exog_predict_calendar.index.dayofweek
    exog_predict_calendar['weekend'] = exog_predict_calendar['day_of_week'] >= 5

    rolling = RollingFeatures(stats=['mean', 'std'], window_sizes=4)
    calendar = CalendarFeatures(features=['day_of_week', 'weekend'], encoding=None)

    forecaster_fast = ForecasterRecursive(
        RandomForestRegressor(n_estimators=10, random_state=123),
        lags=3,
        window_features=rolling,
        calendar_features=calendar
    )
    forecaster_fast.fit(y=y_datetime, exog=exog_datetime)

    forecaster_slow = ForecasterRecursive(
        _SlowRF(n_estimators=10, random_state=123),
        lags=3,
        window_features=rolling
    )
    forecaster_slow.fit(y=y_datetime, exog=exog_calendar)

    last_window_values_fast, exog_values_fast, calendar_values_fast, _, _, _ = (
        forecaster_fast._create_predict_inputs(steps=10, exog=exog_predict_datetime)
    )
    last_window_values_slow, exog_values_slow, calendar_values_slow, _, _, _ = (
        forecaster_slow._create_predict_inputs(steps=10, exog=exog_predict_calendar)
    )

    predictions_fast = forecaster_fast._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_fast,
                           exog_values        = exog_values_fast,
                           calendar_values    = calendar_values_fast
                       )
    predictions_slow = forecaster_slow._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_slow,
                           exog_values        = exog_values_slow,
                           calendar_values    = calendar_values_slow
                       )

    np.testing.assert_array_almost_equal(predictions_fast, predictions_slow)


def test_recursive_predict_fast_path_DecisionTreeRegressor_matches_generic_path():
    """
    Test that the fast prediction path for DecisionTreeRegressor (using
    tree_.predict directly) produces the same results as the generic sklearn
    predict path (forced via a subclass that bypasses the fast-path branch).
    """
    forecaster_fast = ForecasterRecursive(
        DecisionTreeRegressor(random_state=123), lags=3
    )
    forecaster_fast.fit(y=y)

    forecaster_slow = ForecasterRecursive(
        _SlowDT(random_state=123), lags=3
    )
    forecaster_slow.fit(y=y)

    last_window_values_fast, exog_values_fast, _, _, _, _ = (
        forecaster_fast._create_predict_inputs(steps=10)
    )
    last_window_values_slow, exog_values_slow, _, _, _, _ = (
        forecaster_slow._create_predict_inputs(steps=10)
    )

    predictions_fast = forecaster_fast._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_fast,
                           exog_values        = exog_values_fast
                       )
    predictions_slow = forecaster_slow._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_slow,
                           exog_values        = exog_values_slow
                       )

    np.testing.assert_array_almost_equal(predictions_fast, predictions_slow)


def test_recursive_predict_fast_path_DecisionTreeRegressor_with_exog_and_window_features():
    """
    Test that the fast prediction path for DecisionTreeRegressor matches the
    generic sklearn path when exog and window features are used.
    """
    rolling = RollingFeatures(stats=['mean', 'std'], window_sizes=4)

    forecaster_fast = ForecasterRecursive(
        DecisionTreeRegressor(random_state=123),
        lags=3,
        window_features=rolling
    )
    forecaster_fast.fit(y=y, exog=exog)

    forecaster_slow = ForecasterRecursive(
        _SlowDT(random_state=123),
        lags=3,
        window_features=rolling
    )
    forecaster_slow.fit(y=y, exog=exog)

    last_window_values_fast, exog_values_fast, _, _, _, _ = (
        forecaster_fast._create_predict_inputs(steps=10, exog=exog_predict)
    )
    last_window_values_slow, exog_values_slow, _, _, _, _ = (
        forecaster_slow._create_predict_inputs(steps=10, exog=exog_predict)
    )

    predictions_fast = forecaster_fast._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_fast,
                           exog_values        = exog_values_fast
                       )
    predictions_slow = forecaster_slow._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_slow,
                           exog_values        = exog_values_slow
                       )

    np.testing.assert_array_almost_equal(predictions_fast, predictions_slow)


def test_recursive_predict_fast_path_LinearRegression_matches_generic_path():
    """
    Test that the fast prediction path for linear models (using np.dot + intercept)
    produces the same results as the generic sklearn predict path (forced via a
    wrapper class that does not inherit from LinearModel).
    """
    forecaster_fast = ForecasterRecursive(
        Ridge(alpha=1.0, random_state=123), lags=3
    )
    forecaster_fast.fit(y=y)

    forecaster_slow = ForecasterRecursive(
        _SlowLinear(alpha=1.0, random_state=123), lags=3
    )
    forecaster_slow.fit(y=y)

    last_window_values_fast, exog_values_fast, _, _, _, _ = (
        forecaster_fast._create_predict_inputs(steps=10)
    )
    last_window_values_slow, exog_values_slow, _, _, _, _ = (
        forecaster_slow._create_predict_inputs(steps=10)
    )

    predictions_fast = forecaster_fast._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_fast,
                           exog_values        = exog_values_fast
                       )
    predictions_slow = forecaster_slow._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_slow,
                           exog_values        = exog_values_slow
                       )

    np.testing.assert_array_almost_equal(predictions_fast, predictions_slow)


def test_recursive_predict_fast_path_LinearRegression_with_exog_and_window_features():
    """
    Test that the fast prediction path for linear models matches the generic
    sklearn path when exog and window features are used.
    """
    rolling = RollingFeatures(stats=['mean', 'std'], window_sizes=4)

    forecaster_fast = ForecasterRecursive(
        Ridge(random_state=123),
        lags=3,
        window_features=rolling
    )
    forecaster_fast.fit(y=y, exog=exog)

    forecaster_slow = ForecasterRecursive(
        _SlowLinear(alpha=1.0, random_state=123),
        lags=3,
        window_features=rolling
    )
    forecaster_slow.fit(y=y, exog=exog)

    last_window_values_fast, exog_values_fast, _, _, _, _ = (
        forecaster_fast._create_predict_inputs(steps=10, exog=exog_predict)
    )
    last_window_values_slow, exog_values_slow, _, _, _, _ = (
        forecaster_slow._create_predict_inputs(steps=10, exog=exog_predict)
    )

    predictions_fast = forecaster_fast._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_fast,
                           exog_values        = exog_values_fast
                       )
    predictions_slow = forecaster_slow._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_slow,
                           exog_values        = exog_values_slow
                       )

    np.testing.assert_array_almost_equal(predictions_fast, predictions_slow)


def test_recursive_predict_fast_path_LGBMRegressor_matches_generic_path():
    """
    Test that the fast prediction path for LGBMRegressor (using booster.predict
    directly) produces the same results as the generic sklearn predict path
    (forced via a subclass that bypasses the fast-path branch).
    """
    forecaster_fast = ForecasterRecursive(
        LGBMRegressor(verbose=-1, random_state=123), lags=3
    )
    forecaster_fast.fit(y=y)

    forecaster_slow = ForecasterRecursive(
        _SlowLGBM(verbose=-1, random_state=123), lags=3
    )
    forecaster_slow.fit(y=y)

    last_window_values_fast, exog_values_fast, _, _, _, _ = (
        forecaster_fast._create_predict_inputs(steps=10)
    )
    last_window_values_slow, exog_values_slow, _, _, _, _ = (
        forecaster_slow._create_predict_inputs(steps=10)
    )

    predictions_fast = forecaster_fast._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_fast,
                           exog_values        = exog_values_fast
                       )
    predictions_slow = forecaster_slow._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_slow,
                           exog_values        = exog_values_slow
                       )

    np.testing.assert_array_almost_equal(predictions_fast, predictions_slow)


def test_recursive_predict_fast_path_LGBMRegressor_with_exog_and_window_features():
    """
    Test that the fast prediction path for LGBMRegressor matches the generic
    sklearn path when exog and window features are used.
    """
    rolling = RollingFeatures(stats=['mean', 'std'], window_sizes=4)

    forecaster_fast = ForecasterRecursive(
        LGBMRegressor(verbose=-1, random_state=123),
        lags=3,
        window_features=rolling
    )
    forecaster_fast.fit(y=y, exog=exog)

    forecaster_slow = ForecasterRecursive(
        _SlowLGBM(verbose=-1, random_state=123),
        lags=3,
        window_features=rolling
    )
    forecaster_slow.fit(y=y, exog=exog)

    last_window_values_fast, exog_values_fast, _, _, _, _ = (
        forecaster_fast._create_predict_inputs(steps=10, exog=exog_predict)
    )
    last_window_values_slow, exog_values_slow, _, _, _, _ = (
        forecaster_slow._create_predict_inputs(steps=10, exog=exog_predict)
    )

    predictions_fast = forecaster_fast._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_fast,
                           exog_values        = exog_values_fast
                       )
    predictions_slow = forecaster_slow._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_slow,
                           exog_values        = exog_values_slow
                       )

    np.testing.assert_array_almost_equal(predictions_fast, predictions_slow)


def test_recursive_predict_fast_path_XGBRegressor_matches_generic_path():
    """
    Test that the fast prediction path for XGBRegressor (using
    booster.inplace_predict directly) produces the same results as the generic
    sklearn predict path (forced via a subclass that bypasses the fast-path
    branch).
    """
    forecaster_fast = ForecasterRecursive(
        XGBRegressor(random_state=123, verbosity=0), lags=3
    )
    forecaster_fast.fit(y=y)

    forecaster_slow = ForecasterRecursive(
        _SlowXGB(random_state=123, verbosity=0), lags=3
    )
    forecaster_slow.fit(y=y)

    last_window_values_fast, exog_values_fast, _, _, _, _ = (
        forecaster_fast._create_predict_inputs(steps=10)
    )
    last_window_values_slow, exog_values_slow, _, _, _, _ = (
        forecaster_slow._create_predict_inputs(steps=10)
    )

    predictions_fast = forecaster_fast._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_fast,
                           exog_values        = exog_values_fast
                       )
    predictions_slow = forecaster_slow._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_slow,
                           exog_values        = exog_values_slow
                       )

    np.testing.assert_array_almost_equal(predictions_fast, predictions_slow)


def test_recursive_predict_fast_path_XGBRegressor_with_exog_and_window_features():
    """
    Test that the fast prediction path for XGBRegressor matches the generic
    sklearn path when exog and window features are used.
    """
    rolling = RollingFeatures(stats=['mean', 'std'], window_sizes=4)

    forecaster_fast = ForecasterRecursive(
        XGBRegressor(random_state=123, verbosity=0),
        lags=3,
        window_features=rolling
    )
    forecaster_fast.fit(y=y, exog=exog)

    forecaster_slow = ForecasterRecursive(
        _SlowXGB(random_state=123, verbosity=0),
        lags=3,
        window_features=rolling
    )
    forecaster_slow.fit(y=y, exog=exog)

    last_window_values_fast, exog_values_fast, _, _, _, _ = (
        forecaster_fast._create_predict_inputs(steps=10, exog=exog_predict)
    )
    last_window_values_slow, exog_values_slow, _, _, _, _ = (
        forecaster_slow._create_predict_inputs(steps=10, exog=exog_predict)
    )

    predictions_fast = forecaster_fast._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_fast,
                           exog_values        = exog_values_fast
                       )
    predictions_slow = forecaster_slow._recursive_predict(
                           steps              = 10,
                           last_window_values = last_window_values_slow,
                           exog_values        = exog_values_slow
                       )

    np.testing.assert_array_almost_equal(predictions_fast, predictions_slow)
