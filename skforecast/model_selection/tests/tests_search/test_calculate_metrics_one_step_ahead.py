# Unit test _calculate_metrics_one_step_ahead
# ==============================================================================
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from skforecast.metrics import mean_absolute_scaled_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.model_selection._search import _calculate_metrics_one_step_ahead
from skforecast.metrics import add_y_train_argument

# Fixtures
from ..fixtures_model_selection import y
from ..fixtures_model_selection import exog


def test_calculate_metrics_one_step_ahead_when_forecaster_autoreg():
    """
    Testing _calculate_metrics_one_step_ahead when forecaster is of type ForecasterRecursive.
    """

    forecaster = ForecasterRecursive(
        regressor=LinearRegression(),
        lags=5,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
        differentiation=1,
    )
    metrics = [
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_absolute_scaled_error,
    ]
    metrics = [add_y_train_argument(metric) for metric in metrics]
    X_train, y_train, X_test, y_test = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=10
    )
    results = _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        y=y,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    results = np.array([float(result) for result in results])
    expected = np.array([0.5516310508466604, 1.2750659053445799, 1.9249024647280362])

    np.testing.assert_array_almost_equal(results, expected)


def test_calculate_metrics_one_step_ahead_when_forecaster_autoreg_direct():
    """
    Testing _calculate_metrics_one_step_ahead when forecaster is of type ForecasterDirect.
    """

    forecaster = ForecasterDirect(
        regressor=LinearRegression(),
        lags=5,
        steps=3,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    )
    metrics = [
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_absolute_scaled_error,
    ]
    metrics = [add_y_train_argument(metric) for metric in metrics]
    X_train, y_train, X_test, y_test = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=10
    )
    results = _calculate_metrics_one_step_ahead(
        forecaster=forecaster,
        y=y,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    results = np.array([float(result) for result in results])
    expected = np.array([0.3277718194807295, 1.3574261666383498, 0.767982227299475])

    np.testing.assert_array_almost_equal(results, expected)
