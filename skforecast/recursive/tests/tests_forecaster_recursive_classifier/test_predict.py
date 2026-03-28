# Unit test predict ForecasterRecursiveClassifier
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
from sklearn.linear_model import LogisticRegression
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
def test_predict_does_not_modify_y_exog(forecaster_kwargs):
    """
    Test forecaster.predict does not modify y, exog, exog_predict or last_window.
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
    _ = forecaster.predict(steps=5, exog=exog_predict_local, last_window=last_window_local)

    pd.testing.assert_series_equal(y_local, y_copy)
    pd.testing.assert_series_equal(exog_local, exog_copy)
    pd.testing.assert_series_equal(last_window_local, last_window_copy)
    pd.testing.assert_series_equal(exog_predict_local, exog_predict_copy)


def test_predict_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)

    err_msg = re.escape(
        "This Forecaster instance is not fitted yet. Call `fit` with "
        "appropriate arguments before using predict."
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict(steps=5)


def test_predict_output_when_estimator_is_LogisticRegression():
    """
    Test predict output when using LogisticRegression as estimator.
    """
    y_dummy = pd.Series(
        np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']), 
        index=pd.date_range("2020-01-01", periods=15),
        name='y'
    )

    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)
    forecaster.fit(y=y_dummy)
    predictions = forecaster.predict(steps=5)

    expected = pd.Series(
                   data = np.array(['a', 'b', 'c', 'a', 'b']),
                   index = pd.date_range("2020-01-16", periods=5),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)

        
def test_predict_output_when_with_exog():
    """
    Test predict output when using LogisticRegression as estimator.
    """
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)
    
    predictions = forecaster.predict(steps=5, exog=exog_predict)

    expected = pd.Series(
                   data = np.array([2, 2, 2, 2, 2]),
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


def test_predict_output_with_transform_exog():
    """
    Test predict output when using LogisticRegression as estimator and 
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
    predictions = forecaster.predict(steps=5, exog=df_exog_predict)

    expected = pd.Series(
                   data = np.array([3, 1, 2, 3, 2]),
                   index = pd.RangeIndex(start=50, stop=55, step=1),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize(
    'categorical_features',
    ['auto', ['exog_2', 'exog_3']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_predict_output_when_categorical_features_HistGradientBoostingClassifier(categorical_features):
    """
    Test predict output when using HistGradientBoostingClassifier and categorical
    variables managed by the forecaster's `categorical_features` parameter.
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
    predictions = forecaster.predict(steps=10, exog=df_exog_predict)

    expected = pd.Series(
                   data = np.array([2, 1, 1, 2, 2, 3, 3, 2, 2, 1]),
                   index = pd.RangeIndex(start=50, stop=60, step=1),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize(
    'categorical_features',
    ['auto', ['exog_2', 'exog_3']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_predict_output_when_categorical_features_LGBMClassifier(categorical_features):
    """
    Test predict output when using LGBMClassifier and categorical variables
    managed by the forecaster's `categorical_features` parameter.
    """
    df_exog = pd.DataFrame(
        {'exog_1': exog.to_numpy(),
         'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
         'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)}
    )
    
    df_exog_predict = df_exog.iloc[:10, :].copy()
    df_exog_predict.index = pd.RangeIndex(start=50, stop=60)

    forecaster = ForecasterRecursiveClassifier(
                     estimator            = LGBMClassifier(verbose=-1, random_state=123),
                     lags                 = 5,
                     categorical_features = categorical_features
                 )
    forecaster.fit(y=y, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=df_exog_predict)

    expected = pd.Series(
                   data = np.array([2, 1, 2, 2, 1, 1, 2, 2, 1, 3]),
                   index = pd.RangeIndex(start=50, stop=60, step=1),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize(
    'categorical_features',
    ['auto', ['exog_2', 'exog_3']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_predict_output_when_categorical_features_LGBMClassifier_auto(categorical_features):
    """
    Test predict output when using LGBMClassifier and categorical variables with
    categorical_features='auto'. Uses FunctionTransformer pipeline to cast
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
    predictions = forecaster.predict(steps=10, exog=df_exog_predict)

    expected = pd.Series(
                   data = np.array([2, 1, 2, 2, 1, 1, 2, 2, 1, 3]),
                   index = pd.RangeIndex(start=50, stop=60, step=1),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize(
    'features_encoding, categorical_features, use_exog_cat, expected_data',
    [
        ('auto', 'auto', True, np.array([2, 1, 2, 2, 1, 1, 2, 2, 1, 3])),
        ('auto', None, False, np.array([2, 1, 2, 2, 1, 1, 2, 2, 1, 3])),
        ('ordinal', 'auto', True, np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])),
        ('ordinal', None, False, np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])),
    ],
    ids=[
        'autoreg_cat-exog_cat',
        'autoreg_cat-no_exog_cat',
        'no_autoreg_cat-exog_cat',
        'no_autoreg_cat-no_exog_cat',
    ]
)
def test_predict_output_when_features_encoding_and_categorical_features_combinations(
    features_encoding, categorical_features, use_exog_cat, expected_data
):
    """
    Test predict output for all combinations of `features_encoding` (autoreg
    categorical) and `categorical_features` (exog categorical) with LGBMClassifier.
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
    predictions = forecaster.predict(steps=10, exog=df_exog_predict)

    expected = pd.Series(
                   data = expected_data,
                   index = pd.RangeIndex(start=50, stop=60, step=1),
                   name = 'pred'
               )

    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize("steps", 
                         [10, '2020-02-29', pd.to_datetime('2020-02-29')], 
                         ids=lambda steps: f'steps: {steps}')
def test_predict_output_when_window_features(steps):
    """
    Test output of predict when estimator is LGBMClassifier and window features.
    """
    
    rolling = RollingFeaturesClassification(stats=['proportion', 'entropy'], window_sizes=[3, 5])
    forecaster = ForecasterRecursiveClassifier(
        LGBMClassifier(verbose=-1, random_state=123), lags=3, window_features=rolling
    )
    forecaster.fit(y=y_dt, exog=exog_dt)
    predictions = forecaster.predict(steps=steps, exog=exog_dt_predict)

    expected = pd.Series(
                   data = np.array([3, 2, 2, 2, 1, 3, 2, 2, 1, 1]),
                   index = pd.date_range(start='2020-02-20', periods=10, freq='D'),
                   name = 'pred'
               )
    
    pd.testing.assert_series_equal(predictions, expected)
