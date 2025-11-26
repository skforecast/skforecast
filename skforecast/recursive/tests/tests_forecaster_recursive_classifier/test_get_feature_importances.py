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
                     estimator = LogisticRegression(),
                     lags = 3,
                 )

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `get_feature_importances()`."
    )
    with pytest.raises(NotFittedError, match = err_msg):         
        forecaster.get_feature_importances()


def test_output_and_UserWarning_get_feature_importances_when_CalibratedClassifierCV_no_calibrated_classifiers_():
    """
    Test output of get_feature_importances when estimator is CalibratedClassifierCV.
    and calibrated_classifiers_ is empty.
    """
    forecaster = ForecasterRecursiveClassifier(
        CalibratedClassifierCV(LogisticRegression()), lags=3
    )
    forecaster.is_fitted = True  # To skip the check for fitted forecaster

    warn_msg = re.escape(
        "The CalibratedClassifierCV instance is not fitted or does not "
        "expose 'calibrated_classifiers_'. Unable to retrieve importances."
    )
    with pytest.warns(UserWarning, match = warn_msg):
        assert forecaster.get_feature_importances() is None


def test_output_and_UserWarning_get_feature_importances_when_estimator_no_attributes():
    """
    Test output of get_feature_importances when estimator is HistGradientBoostingClassifier.
    Since HistGradientBoostingClassifier hasn't attributes `feature_importances_` 
    or `coef_, results = None and a UserWarning is issues.
    """
    forecaster = ForecasterRecursiveClassifier(
        HistGradientBoostingClassifier(max_iter=5, max_depth=2, random_state=123), lags=3
    )
    forecaster.fit(y=y)

    estimator = forecaster.estimator

    warn_msg = re.escape(
        f"Impossible to access feature importances for estimator of type "
        f"{type(estimator)}. This method is only valid when the "
        f"estimator stores internally the feature importances in the "
        f"attribute `feature_importances_` or `coef_`."
    )
    with pytest.warns(UserWarning, match = warn_msg):
        assert forecaster.get_feature_importances() is None


def test_output_get_feature_importances_when_estimator_is_RandomForestClassifier():
    """
    Test output of get_feature_importances when estimator is RandomForestClassifier with lags=3
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


def test_output_get_feature_importances_when_estimator_is_RandomForest_with_exog():
    """
    Test output of get_feature_importances when estimator is RandomForestClassifier 
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


def test_output_get_feature_importances_when_estimator_is_LogisticRegression():
    """
    Test output of get_feature_importances when estimator is LogisticRegression.
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
    Test output of get_feature_importances when estimator is LogisticRegression 
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


def test_output_get_feature_importances_when_CalibratedClassifierCV_LogisticRegression():
    """
    Test output of get_feature_importances when using CalibratedClassifierCV with
    LogisticRegression as base estimator.
    """
    forecaster = ForecasterRecursiveClassifier(
                     estimator = CalibratedClassifierCV(LogisticRegression()),
                     lags      = 3
                 )
    forecaster.fit(y=y)

    results = forecaster.get_feature_importances(sort_importance=False)
    expected = pd.DataFrame(
        {
            "cv_fold": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "classes": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            "lag_1": [
                0.17392860358936565,
                -0.3184535617446897,
                0.14452495815532343,
                0.5352421566187048,
                -0.5173562179196608,
                -0.017885938699045225,
                0.03377500382837249,
                -0.26741874378694175,
                0.23364373995857027,
                0.17065538525910254,
                -0.007321092653028191,
                -0.16333429260607474,
                0.2726424414624969,
                -0.3334117120536584,
                0.06076927059116098,
            ],
            "lag_2": [
                -0.01611048290713891,
                0.11077280342639155,
                -0.09466232051925359,
                0.2280201941787555,
                -0.25569262632673506,
                0.027672432147979882,
                0.10076280520266548,
                -0.003800599666816256,
                -0.09696220553585018,
                0.04863953728034529,
                -0.04348611843417003,
                -0.005153418846175248,
                -0.04023988196420971,
                0.04327266081780304,
                -0.003032778853593432,
            ],
            "lag_3": [
                -0.3653103558512643,
                -0.1654339349554506,
                0.530744290806715,
                -0.1725577743251356,
                0.061039588331654,
                0.11151818599348112,
                -0.2058169676549178,
                -0.2336644022888757,
                0.43948136994379405,
                -0.16896234946742456,
                -0.11116568424885509,
                0.2801280337162793,
                -0.25566300391265306,
                -0.21052789543738035,
                0.46619089935003255,
            ],
        }
    )

    pd.testing.assert_frame_equal(results, expected)


def test_output_get_feature_importances_when_CalibratedClassifierCV_LGBMClassifier():
    """
    Test output of get_feature_importances when using CalibratedClassifierCV with
    LGBMClassifier as base estimator, min_child_samples=5 due to small sample size.
    """
    forecaster = ForecasterRecursiveClassifier(
                     estimator = CalibratedClassifierCV(LGBMClassifier(
                         min_child_samples=5, verbose=-1, random_state=123
                     ), cv=3),
                     lags      = 3
                 )
    forecaster.fit(y=y)

    results = forecaster.get_feature_importances(sort_importance=True)
    expected = pd.DataFrame(
        {
            "cv_fold": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "feature": [
                "lag_3",
                "lag_1",
                "lag_2",
                "lag_1",
                "lag_2",
                "lag_3",
                "lag_1",
                "lag_3",
                "lag_2",
            ],
            "importance": [387, 376, 324, 368, 366, 338, 386, 361, 338],
        }
    ).astype({"importance": int})
    expected.index = pd.Index([2, 0, 1, 3, 4, 5, 6, 8, 7], dtype=int)

    pd.testing.assert_frame_equal(results.astype({"importance": int}), expected)


def test_output_get_feature_importances_when_window_features():
    """
    Test output of get_feature_importances when estimator is LGMBRegressor with 
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
