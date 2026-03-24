# Unit test predict_proba ForecasterRecursiveClassifier
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier

from skforecast.preprocessing import RollingFeaturesClassification
from skforecast.recursive import ForecasterRecursiveClassifier

# Fixtures
from .fixtures_forecaster_recursive_classifier import y, y_dt
from .fixtures_forecaster_recursive_classifier import exog, exog_dt, exog_predict, exog_dt_predict


@pytest.mark.parametrize(
    "forecaster_kwargs",
    [
        {"estimator": LogisticRegression(), "lags": 3},
        {"estimator": LogisticRegression(), "lags": 3,
         "window_features": RollingFeaturesClassification(stats=['proportion'], window_sizes=4)},
        {"estimator": LogisticRegression(), "lags": 3,
         "window_features": RollingFeaturesClassification(stats=['proportion'], window_sizes=4),
         "transformer_exog": StandardScaler()},
    ],
    ids=["base", "window_features", "transformers"]
)
def test_predict_proba_does_not_modify_y_exog(forecaster_kwargs):
    """
    Test forecaster.predict_proba does not modify y, exog, exog_predict or last_window.
    """
    y_local = y.copy()
    exog_local = exog.copy()
    exog_predict_local = exog_predict.copy()
    last_window_local = y_local.iloc[-4:].copy()

    y_copy = y_local.copy()
    exog_copy = exog_local.copy()
    exog_predict_copy = exog_predict_local.copy()
    last_window_copy = last_window_local.copy()

    forecaster = ForecasterRecursiveClassifier(**forecaster_kwargs)
    forecaster.fit(y=y_local, exog=exog_local)
    _ = forecaster.predict_proba(steps=5, exog=exog_predict_local, last_window=last_window_local)

    pd.testing.assert_series_equal(y_local, y_copy)
    pd.testing.assert_series_equal(exog_local, exog_copy)
    pd.testing.assert_series_equal(last_window_local, last_window_copy)
    pd.testing.assert_series_equal(exog_predict_local, exog_predict_copy)


def test_predict_proba_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)

    err_msg = re.escape(
        "This Forecaster instance is not fitted yet. Call `fit` with "
        "appropriate arguments before using predict."
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict_proba(steps=5)


def test_predict_proba_AttributeError_when_estimator_does_not_support_predict_proba():
    """
    Test AttributeError is raised when estimator does not support predict_proba.
    """
    forecaster = ForecasterRecursiveClassifier(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10), name='y'))

    err_msg = re.escape(
        f"The estimator {type(forecaster.estimator).__name__} does not have a "
        f"`predict_proba` method. Use a estimator that supports probability "
        f"predictions (e.g., XGBClassifier, HistGradientBoostingClassifier, etc.)."
    )
    with pytest.raises(AttributeError, match = err_msg):
        forecaster.predict_proba(steps=5)


def test_predict_proba_output_when_estimator_is_LogisticRegression():
    """
    Test predict_proba output when using LogisticRegression as estimator.
    """
    y_dummy = pd.Series(
        np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']), 
        index=pd.date_range("2020-01-01", periods=15),
        name='y'
    )

    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)
    forecaster.fit(y=y_dummy)
    predictions = forecaster.predict_proba(steps=5)

    expected = pd.DataFrame(
                   data = np.array([[0.86043129, 0.06978436, 0.06978436],
                                    [0.06978436, 0.86043129, 0.06978436],
                                    [0.06978436, 0.06978436, 0.86043129],
                                    [0.86043129, 0.06978436, 0.06978436],
                                    [0.06978436, 0.86043129, 0.06978436]]),
                   index = pd.date_range("2020-01-16", periods=5),
                   columns = ['a_proba', 'b_proba', 'c_proba']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)

        
def test_predict_proba_output_when_with_exog():
    """
    Test predict_proba output when using LogisticRegression as estimator.
    """
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)
    
    predictions = forecaster.predict_proba(steps=5, exog=exog_predict)

    expected = pd.DataFrame(
                   data = np.array([[0.2605886 , 0.50755887, 0.23185253],
                                    [0.20478879, 0.45058736, 0.34462385],
                                    [0.23614095, 0.49574906, 0.26810999],
                                    [0.23145311, 0.49174114, 0.27680575],
                                    [0.21512009, 0.47646933, 0.30841058]]),
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1_proba', '2_proba', '3_proba']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_proba_output_with_transform_exog():
    """
    Test predict_proba output when using LogisticRegression as estimator and 
    transformer_exog.
    """

    df_exog = pd.DataFrame({
               'col_1': exog.to_numpy(),
               'col_2': ['a', 'a', 'b', 'a', 'b'] * 10}
           )
    df_exog_predict = df_exog.iloc[:10, :].copy()
    df_exog_predict.index = pd.RangeIndex(start=50, stop=60)

    transformer_exog = ColumnTransformer(
                            [('scale', StandardScaler(), ['col_1']),
                             ('onehot', OneHotEncoder(), ['col_2'])],
                            remainder = 'passthrough',
                            verbose_feature_names_out = False
                       )
    forecaster = ForecasterRecursiveClassifier(
                     estimator        = LogisticRegression(),
                     lags             = 5,
                     transformer_exog = transformer_exog,
                 )
    forecaster.fit(y=y, exog=df_exog)
    predictions = forecaster.predict_proba(steps=5, exog=df_exog_predict)

    expected = pd.DataFrame(
                   data = np.array([[0.17873329, 0.34431688, 0.47694983],
                                    [0.48722762, 0.24371881, 0.26905357],
                                    [0.05659829, 0.8275415 , 0.11586022],
                                    [0.2290655 , 0.30146926, 0.46946524],
                                    [0.26096949, 0.50877694, 0.23025358]]),
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1_proba', '2_proba', '3_proba']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize(
    'categorical_features',
    ['auto', ['exog_2', 'exog_3']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_predict_proba_output_when_categorical_features_HistGradientBoostingClassifier(categorical_features):
    """
    Test predict_proba output when using HistGradientBoostingClassifier and
    categorical variables. Both `categorical_features='auto'` and an explicit
    list should produce the same predictions.
    """
    df_exog = pd.DataFrame({'exog_1': exog.to_numpy(),
                            'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
                            'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)})
    
    df_exog_predict = df_exog.iloc[:10, :].copy()
    df_exog_predict.index = pd.RangeIndex(start=50, stop=60)

    forecaster = ForecasterRecursiveClassifier(
                     estimator            = HistGradientBoostingClassifier(
                                                random_state = 123
                                            ),
                     lags                 = 5,
                     categorical_features = categorical_features
                 )
    forecaster.fit(y=y, exog=df_exog)
    predictions = forecaster.predict_proba(steps=10, exog=df_exog_predict)

    expected = pd.DataFrame(
                   data = np.array([[0.14094939, 0.51412759, 0.34492302],
                                    [0.46748341, 0.37135751, 0.16115908],
                                    [0.40762409, 0.36783112, 0.22454478],
                                    [0.10945589, 0.50152688, 0.38901722],
                                    [0.14094939, 0.51412759, 0.34492302],
                                    [0.31361194, 0.31477162, 0.37161644],
                                    [0.2479161 , 0.28266388, 0.46942002],
                                    [0.18460338, 0.60538689, 0.21000973],
                                    [0.21744631, 0.62774431, 0.15480938],
                                    [0.43857413, 0.25746504, 0.30396083]]),
                   index = pd.RangeIndex(start=50, stop=60, step=1),
                   columns = ['1_proba', '2_proba', '3_proba']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize(
    'categorical_features',
    ['auto', ['exog_2', 'exog_3']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_predict_proba_output_when_categorical_features_LGBMClassifier(categorical_features):
    """
    Test predict_proba output when using LGBMClassifier and categorical variables.
    Both `categorical_features='auto'` and an explicit list should produce the
    same predictions.
    """
    df_exog = pd.DataFrame({'exog_1': exog.to_numpy(),
                            'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
                            'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)})
    
    df_exog_predict = df_exog.iloc[:10, :].copy()
    df_exog_predict.index = pd.RangeIndex(start=50, stop=60)

    forecaster = ForecasterRecursiveClassifier(
                     estimator            = LGBMClassifier(verbose=-1, random_state=123),
                     lags                 = 5,
                     categorical_features = categorical_features
                 )
    forecaster.fit(y=y, exog=df_exog)
    predictions = forecaster.predict_proba(steps=10, exog=df_exog_predict)

    expected = pd.DataFrame(
                   data = np.array([[0.17224153, 0.44370028, 0.38405819],
                                    [0.50668773, 0.30518928, 0.18812299],
                                    [0.34780065, 0.40852763, 0.24367172],
                                    [0.16616252, 0.46415355, 0.36968392],
                                    [0.42795775, 0.24426185, 0.3277804 ],
                                    [0.42466306, 0.27703231, 0.29830463],
                                    [0.09774055, 0.49100823, 0.41125122],
                                    [0.23814169, 0.53285177, 0.22900654],
                                    [0.50668773, 0.30518928, 0.18812299],
                                    [0.25145828, 0.34004613, 0.40849559]]),
                   index = pd.RangeIndex(start=50, stop=60, step=1),
                   columns = ['1_proba', '2_proba', '3_proba']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize(
    'categorical_features',
    ['auto', ['exog_2', 'exog_3']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_predict_proba_output_when_categorical_features_LGBMClassifier_auto(categorical_features):
    """
    Test predict_proba output when using LGBMClassifier and categorical variables
    with categorical_features='auto'. Uses FunctionTransformer pipeline to cast
    encoded columns to 'category' dtype via transformer_exog.
    """
    df_exog = pd.DataFrame({'exog_1': exog.to_numpy(),
                            'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
                            'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)})
    
    df_exog_predict = df_exog.iloc[:10, :].copy()
    df_exog_predict.index = pd.RangeIndex(start=50, stop=60)

    pipeline_categorical = make_pipeline(
                               OrdinalEncoder(
                                   dtype=int,
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1,
                                   encoded_missing_value=-1
                               ),
                               FunctionTransformer(
                                   func=lambda x: x.astype('category'),
                                   feature_names_out= 'one-to-one'
                               )
                           )
    transformer_exog = make_column_transformer(
                            (
                                pipeline_categorical,
                                make_column_selector(dtype_exclude=np.number)
                            ),
                            remainder="passthrough",
                            verbose_feature_names_out=False,
                       ).set_output(transform="pandas")
    
    forecaster = ForecasterRecursiveClassifier(
                     estimator            = LGBMClassifier(verbose=-1, random_state=123),
                     lags                 = 5,
                     transformer_exog     = transformer_exog,
                     categorical_features = categorical_features
                 )
    forecaster.fit(y=y, exog=df_exog)
    predictions = forecaster.predict_proba(steps=10, exog=df_exog_predict)

    expected = pd.DataFrame(
                   data = np.array([[0.17224153, 0.44370028, 0.38405819],
                                    [0.50668773, 0.30518928, 0.18812299],
                                    [0.34780065, 0.40852763, 0.24367172],
                                    [0.16616252, 0.46415355, 0.36968392],
                                    [0.42795775, 0.24426185, 0.3277804 ],
                                    [0.42466306, 0.27703231, 0.29830463],
                                    [0.09774055, 0.49100823, 0.41125122],
                                    [0.23814169, 0.53285177, 0.22900654],
                                    [0.50668773, 0.30518928, 0.18812299],
                                    [0.25145828, 0.34004613, 0.40849559]]),
                   index = pd.RangeIndex(start=50, stop=60, step=1),
                   columns = ['1_proba', '2_proba', '3_proba']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize(
    'features_encoding, categorical_features, use_exog_cat, expected_data',
    [
        ('auto', 'auto', True,
         np.array([[0.17224153, 0.44370028, 0.38405819],
                   [0.50668773, 0.30518928, 0.18812299],
                   [0.34780065, 0.40852763, 0.24367172],
                   [0.16616252, 0.46415355, 0.36968392],
                   [0.42795775, 0.24426185, 0.3277804 ],
                   [0.42466306, 0.27703231, 0.29830463],
                   [0.09774055, 0.49100823, 0.41125122],
                   [0.23814169, 0.53285177, 0.22900654],
                   [0.50668773, 0.30518928, 0.18812299],
                   [0.25145828, 0.34004613, 0.40849559]])),
        ('auto', None, False,
         np.array([[0.17224153, 0.44370028, 0.38405819],
                   [0.50668773, 0.30518928, 0.18812299],
                   [0.34780065, 0.40852763, 0.24367172],
                   [0.16616252, 0.46415355, 0.36968392],
                   [0.42795775, 0.24426185, 0.3277804 ],
                   [0.42466306, 0.27703231, 0.29830463],
                   [0.09774055, 0.49100823, 0.41125122],
                   [0.23814169, 0.53285177, 0.22900654],
                   [0.50668773, 0.30518928, 0.18812299],
                   [0.25145828, 0.34004613, 0.40849559]])),
        ('ordinal', 'auto', True,
         np.array([[0.22829797, 0.38950076, 0.38220126],
                   [0.27679603, 0.4837721 , 0.23943188],
                   [0.27679603, 0.4837721 , 0.23943188],
                   [0.13247683, 0.55335888, 0.31416429],
                   [0.22829797, 0.38950076, 0.38220126],
                   [0.22829797, 0.38950076, 0.38220126],
                   [0.22829797, 0.38950076, 0.38220126],
                   [0.27679603, 0.4837721 , 0.23943188],
                   [0.27679603, 0.4837721 , 0.23943188],
                   [0.22829797, 0.38950076, 0.38220126]])),
        ('ordinal', None, False,
         np.array([[0.22829797, 0.38950076, 0.38220126],
                   [0.27679603, 0.4837721 , 0.23943188],
                   [0.27679603, 0.4837721 , 0.23943188],
                   [0.13247683, 0.55335888, 0.31416429],
                   [0.22829797, 0.38950076, 0.38220126],
                   [0.22829797, 0.38950076, 0.38220126],
                   [0.22829797, 0.38950076, 0.38220126],
                   [0.27679603, 0.4837721 , 0.23943188],
                   [0.27679603, 0.4837721 , 0.23943188],
                   [0.22829797, 0.38950076, 0.38220126]])),
    ],
    ids=[
        'autoreg_cat-exog_cat',
        'autoreg_cat-no_exog_cat',
        'no_autoreg_cat-exog_cat',
        'no_autoreg_cat-no_exog_cat',
    ]
)
def test_predict_proba_output_when_features_encoding_and_categorical_features_combinations(
    features_encoding, categorical_features, use_exog_cat, expected_data
):
    """
    Test predict_proba output for all combinations of `features_encoding`
    (autoreg categorical) and `categorical_features` (exog categorical)
    with LGBMClassifier.
    """
    if use_exog_cat:
        df_exog = pd.DataFrame(
            {'exog_1': exog.to_numpy(),
             'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
             'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)}
        )
        df_exog_predict = df_exog.iloc[:10, :].copy()
        df_exog_predict.index = pd.RangeIndex(start=50, stop=60)
    else:
        df_exog = exog
        df_exog_predict = exog_predict

    forecaster = ForecasterRecursiveClassifier(
                     estimator            = LGBMClassifier(verbose=-1, random_state=123),
                     lags                 = 5,
                     features_encoding    = features_encoding,
                     categorical_features = categorical_features
                 )
    forecaster.fit(y=y, exog=df_exog)
    predictions = forecaster.predict_proba(steps=10, exog=df_exog_predict)

    expected = pd.DataFrame(
                   data = expected_data,
                   index = pd.RangeIndex(start=50, stop=60, step=1),
                   columns = ['1_proba', '2_proba', '3_proba']
               )

    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize("steps", 
                         [10, '2020-02-29', pd.to_datetime('2020-02-29')], 
                         ids=lambda steps: f'steps: {steps}')
def test_predict_proba_output_when_window_features(steps):
    """
    Test output of predict_proba when estimator is LGBMClassifier and window features.
    """
    
    rolling = RollingFeaturesClassification(stats=['proportion', 'entropy'], window_sizes=[3, 5])
    forecaster = ForecasterRecursiveClassifier(
        LGBMClassifier(verbose=-1, random_state=123), lags=3, window_features=rolling
    )
    forecaster.fit(y=y_dt, exog=exog_dt)
    predictions = forecaster.predict_proba(steps=steps, exog=exog_dt_predict)

    expected = pd.DataFrame(
                   data = np.array([[0.12076544, 0.42469472, 0.45453984],
                                    [0.22492734, 0.45651712, 0.31855554],
                                    [0.28068232, 0.54339786, 0.17591983],
                                    [0.36106523, 0.39667833, 0.24225644],
                                    [0.42262161, 0.22638072, 0.35099767],
                                    [0.34071633, 0.28389443, 0.37538924],
                                    [0.14714771, 0.56239456, 0.29045773],
                                    [0.14676289, 0.59766833, 0.25556878],
                                    [0.41894907, 0.37194739, 0.20910354],
                                    [0.41684109, 0.34732373, 0.23583518]]),
                   index = pd.date_range("2020-02-20", periods=10),
                   columns = ['1_proba', '2_proba', '3_proba']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)
