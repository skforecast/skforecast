# Unit test get_feature_importances ForecasterRecursiveClassifier
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier

from skforecast.preprocessing import RollingFeaturesClassification
from skforecast.recursive import ForecasterRecursiveClassifier

# Fixtures
from .fixtures_forecaster_recursive_classifier import y, y_dt
from .fixtures_forecaster_recursive_classifier import exog, exog_dt


def test_NotFittedError_is_raised_when_forecaster_is_not_fitted():
    """
    Test NotFittedError is raised when calling get_feature_importances() and 
    forecaster is not fitted.
    """
    forecaster = ForecasterRecursiveClassifier(
                     regressor = LogisticRegression(),
                     lags = 3,
                 )

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `get_feature_importances()`."
    )
    with pytest.raises(NotFittedError, match = err_msg):         
        forecaster.get_feature_importances()


def test_output_and_UserWarning_get_feature_importances_when_regressor_no_attributes():
    """
    Test output of get_feature_importances when estimator is HistGradientBoostingClassifier.
    Since HistGradientBoostingClassifier hasn't attributes `feature_importances_` 
    or `coef_, results = None and a UserWarning is issues.
    """
    forecaster = ForecasterRecursiveClassifier(
        HistGradientBoostingClassifier(max_iter=5, max_depth=2, random_state=123), lags=3
    )
    forecaster.fit(y=y)

    estimator = forecaster.regressor
    expected = None

    warn_msg = re.escape(
        f"Impossible to access feature importances for estimator of type "
        f"{type(estimator)}. This method is only valid when the "
        f"estimator stores internally the feature importances in the "
        f"attribute `feature_importances_` or `coef_`."
    )
    with pytest.warns(UserWarning, match = warn_msg):
        results = forecaster.get_feature_importances()
        assert results is expected


def test_output_get_feature_importances_when_regressor_is_RandomForestClassifier():
    """
    Test output of get_feature_importances when regressor is RandomForestClassifier with lags=3
    and it is trained with y.
    """
    forecaster = ForecasterRecursiveClassifier(
        RandomForestClassifier(n_estimators=1, max_depth=2, random_state=123), lags=3
    )
    forecaster.fit(y=y)
    
    results = forecaster.get_feature_importances()
    expected = pd.DataFrame({
                   'feature': ['lag_3', 'lag_2', 'lag_1'],
                   'importance': np.array([0.54353465, 0.37326339, 0.08320197])
               })
    expected.index = results.index 

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_regressor_is_RandomForest_with_exog():
    """
    Test output of get_feature_importances when regressor is RandomForestClassifier 
    with exog.
    """
    forecaster = ForecasterRecursiveClassifier(
        RandomForestClassifier(n_estimators=10, max_depth=2, random_state=123), lags=3
    )
    forecaster.fit(y=y, exog=exog)
    
    results = forecaster.get_feature_importances()
    expected = pd.DataFrame({
                   'feature': ['exog', 'lag_3', 'lag_2', 'lag_1'],
                   'importance': np.array([0.40110841, 0.24458015, 0.2177843, 0.13652714])
               })
    expected.index = results.index 

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_regressor_is_LogisticRegression():
    """
    Test output of get_feature_importances when regressor is LogisticRegression.
    """
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)
    forecaster.fit(y=y)

    results = forecaster.get_feature_importances(sort_importance=False)
    expected = pd.DataFrame(
        {
            "classes": [1, 2, 3],
            "lag_1": [0.23797127758652273, -0.28475321707531215, 0.04678193948878988],
            "lag_2": [0.06713233699919362, -0.019426688294508363, -0.0477056487046847],
            "lag_3": [-0.23490161604338683, -0.13721916783375127, 0.372120783877139],
        }
    )

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_with_exog():
    """
    Test output of get_feature_importances when regressor is LogisticRegression 
    using exog.
    """
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)

    results = forecaster.get_feature_importances(sort_importance=False)
    expected = pd.DataFrame(
        {
            "classes": [1, 2, 3],
            "lag_1": [0.24164681032992197, -0.2813814286253202, 0.03973461829539689],
            "lag_2": [
                0.05463802280761968,
                -0.026731595943493343,
                -0.027906426864126796,
            ],
            "lag_3": [-0.22882712182712814, -0.1359468215076011, 0.36477394333472674],
            "exog": [0.3673160018213961, 0.16152780517129176, -0.5288438069926882],
        }
    )

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_CalibratedClassifierCV():
    """
    Test output of get_feature_importances when using CalibratedClassifierCV with
    LogisticRegression as base estimator.
    """
    forecaster = ForecasterRecursiveClassifier(
                     regressor = CalibratedClassifierCV(LogisticRegression()),
                     lags      = 3
                 )
    forecaster.fit(y=y)

    results = forecaster.get_feature_importances(sort_importance=False)
    expected = pd.DataFrame({
                   'feature': ['lag_1', 'lag_2', 'lag_3'],
                   'importance': np.array([0.166667, 0.166667, 0.166667])
               })

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_window_features():
    """
    Test output of get_feature_importances when regressor is LGMBRegressor with 
    lags=3 and window features.
    """

    rolling = RollingFeaturesClassification(stats=['proportion', 'entropy'], window_sizes=[3, 5])
    forecaster = ForecasterRecursiveClassifier(
        LGBMClassifier(verbose=-1, random_state=123), lags=3, window_features=rolling
    )
    forecaster.fit(y=y_dt, exog=exog_dt)

    results = forecaster.get_feature_importances(sort_importance=False)
    results = results.astype({'importance': int})
    expected = pd.DataFrame(
        {
            "feature": [
                "lag_1",
                "lag_2",
                "lag_3",
                "roll_proportion_3_class_0",
                "roll_proportion_3_class_1",
                "roll_proportion_3_class_2",
                "roll_entropy_5",
                "exog",
            ],
            "importance": [37, 63, 32, 0, 23, 0, 0, 145],
        }
    )

    pd.testing.assert_frame_equal(results, expected)
