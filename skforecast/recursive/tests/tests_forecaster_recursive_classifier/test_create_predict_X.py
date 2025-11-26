# Unit test create_predict_X ForecasterRecursiveClassifier
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
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier

from ....exceptions import DataTransformationWarning
from skforecast.utils import transform_numpy
from skforecast.preprocessing import RollingFeaturesClassification
from skforecast.recursive import ForecasterRecursiveClassifier

# Fixtures
from .fixtures_forecaster_recursive_classifier import y, y_dt
from .fixtures_forecaster_recursive_classifier import exog, exog_dt, exog_predict, exog_dt_predict


def test_create_predict_X_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterRecursiveClassifier(
                     estimator = LogisticRegression(),
                     lags      = 5
                 )

    err_msg = re.escape(
        "This Forecaster instance is not fitted yet. Call `fit` with "
        "appropriate arguments before using predict."
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.create_predict_X(steps=5)


def test_create_predict_X_output_when_estimator_is_LogisticRegression():
    """
    Test create_predict_X output when using LogisticRegression as estimator.
    """
    y_dummy = pd.Series(
        np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']), 
        index=pd.date_range("2020-01-01", periods=15),
        name='y'
    )

    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)
    forecaster.fit(y=y_dummy)
    predictions = forecaster.create_predict_X(steps=5)

    expected = pd.DataFrame(
                   data = np.array([[2., 1., 0.],
                                    [0., 2., 1.],
                                    [1., 0., 2.],
                                    [2., 1., 0.],
                                    [0., 2., 1.]]),
                   index = pd.date_range("2020-01-16", periods=5),
                   columns = ['lag_1', 'lag_2', 'lag_3']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_create_predict_X_output_when_with_exog():
    """
    Test create_predict_X output when using LogisticRegression as estimator.
    """
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)
    
    predictions = forecaster.create_predict_X(steps=5, exog=exog_predict)

    expected = pd.DataFrame(
        data=np.array(
            [
                [1.0, 2.0, 0.0, 0.12062867],
                [1.0, 1.0, 2.0, 0.8263408],
                [1.0, 1.0, 1.0, 0.60306013],
                [1.0, 1.0, 1.0, 0.54506801],
                [1.0, 1.0, 1.0, 0.34276383],
            ]
        ),
        index=pd.RangeIndex(start=50, stop=55, step=1),
        columns=["lag_1", "lag_2", "lag_3", "exog"],
    )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_create_predict_X_output_with_transform_exog():
    """
    Test create_predict_X output when using LogisticRegression as estimator and 
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

    warn_msg = re.escape(
        "The output matrix is in the transformed scale due to the "
        "inclusion of transformations (`transformer_exog`) in the Forecaster. "
        "As a result, any predictions generated using this matrix will also "
        "be in the transformed scale. Please refer to the documentation "
        "for more details: "
        "https://skforecast.org/latest/user_guides/training-and-prediction-matrices.html"
    )
    with pytest.warns(DataTransformationWarning, match = warn_msg):
        predictions = forecaster.create_predict_X(steps=5, exog=df_exog_predict)

    expected = pd.DataFrame(
        data=np.array(
            [
                [1.0, 2.0, 0.0, 1.0, 1.0, -1.47636391, 1.0, 0.0],
                [2.0, 1.0, 2.0, 0.0, 1.0, 1.26277054, 1.0, 0.0],
                [0.0, 2.0, 1.0, 2.0, 0.0, 0.3961342, 0.0, 1.0],
                [1.0, 0.0, 2.0, 1.0, 2.0, 0.17104495, 1.0, 0.0],
                [2.0, 1.0, 0.0, 2.0, 1.0, -0.61417373, 0.0, 1.0],
            ]
        ),
        index=pd.RangeIndex(start=50, stop=55, step=1),
        columns=[
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_4",
            "lag_5",
            "col_1",
            "col_2_a",
            "col_2_b",
        ],
    )

    pd.testing.assert_frame_equal(predictions, expected)


def test_create_predict_X_output_when_categorical_features_native_implementation_HistGradientBoostingClassifier():
    """
    Test create_predict_X output when using HistGradientBoostingClassifier and categorical variables.
    """
    df_exog = pd.DataFrame({'exog_1': exog.to_numpy(),
                            'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
                            'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)})

    df_exog_predict = df_exog.iloc[:10, :].copy()
    df_exog_predict.index = pd.RangeIndex(start=50, stop=60)

    categorical_features = df_exog.select_dtypes(exclude=[np.number]).columns.tolist()
    transformer_exog = make_column_transformer(
                           (
                               OrdinalEncoder(
                                   dtype=int,
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1,
                                   encoded_missing_value=-1
                               ),
                               categorical_features
                           ),
                           remainder="passthrough",
                           verbose_feature_names_out=False,
                       ).set_output(transform="pandas")

    forecaster = ForecasterRecursiveClassifier(
                     estimator        = HistGradientBoostingClassifier(
                                            categorical_features = categorical_features,
                                            random_state         = 123
                                        ),
                     lags             = 5,
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y, exog=df_exog)

    warn_msg = re.escape(
        "The output matrix is in the transformed scale due to the "
        "inclusion of transformations (`transformer_exog`) in the Forecaster. "
        "As a result, any predictions generated using this matrix will also "
        "be in the transformed scale. Please refer to the documentation "
        "for more details: "
        "https://skforecast.org/latest/user_guides/training-and-prediction-matrices.html"
    )
    with pytest.warns(DataTransformationWarning, match = warn_msg):
        predictions = forecaster.create_predict_X(steps=10, exog=df_exog_predict)

    expected = pd.DataFrame(
        data=np.array(
            [
                [1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.12062867],
                [1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.8263408],
                [1.0, 1.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.60306013],
                [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 0.54506801],
                [1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 0.34276383],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.30412079],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.41702221],
                [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.68130077],
                [1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 0.87545684],
                [1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 0.51042234],
            ]
        ),
        index=pd.RangeIndex(start=50, stop=60, step=1),
        columns=[
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_4",
            "lag_5",
            "exog_2",
            "exog_3",
            "exog_1",
        ],
    )
    for col in ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']:
        expected[col] = pd.Categorical(
                            values     = expected[col],
                            categories = forecaster.class_codes_,
                            ordered    = False
                        )

    pd.testing.assert_frame_equal(predictions, expected)


def test_create_predict_X_output_when_categorical_features_native_implementation_LGBMClassifier():
    """
    Test create_predict_X output when using LGBMClassifier and categorical variables.
    """
    df_exog = pd.DataFrame({'exog_1': exog.to_numpy(),
                            'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
                            'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)})
    
    df_exog_predict = df_exog.iloc[:10, :].copy()
    df_exog_predict.index = pd.RangeIndex(start=50, stop=60)

    categorical_features = df_exog.select_dtypes(exclude=[np.number]).columns.tolist()
    transformer_exog = make_column_transformer(
                           (
                               OrdinalEncoder(
                                   dtype=int,
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1,
                                   encoded_missing_value=-1
                               ),
                               categorical_features
                           ),
                           remainder="passthrough",
                           verbose_feature_names_out=False,
                       ).set_output(transform="pandas")
    
    forecaster = ForecasterRecursiveClassifier(
                     estimator        = LGBMClassifier(verbose=-1, random_state=123),
                     lags             = 5,
                     transformer_exog = transformer_exog,
                     fit_kwargs       = {'categorical_feature': categorical_features}
                 )
    forecaster.fit(y=y, exog=df_exog)

    warn_msg = re.escape(
        "The output matrix is in the transformed scale due to the "
        "inclusion of transformations (`transformer_exog`) in the Forecaster. "
        "As a result, any predictions generated using this matrix will also "
        "be in the transformed scale. Please refer to the documentation "
        "for more details: "
        "https://skforecast.org/latest/user_guides/training-and-prediction-matrices.html"
    )
    with pytest.warns(DataTransformationWarning, match = warn_msg):
        predictions = forecaster.create_predict_X(steps=10, exog=df_exog_predict)

    expected = pd.DataFrame(
        data=np.array(
            [
                [1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.12062867],
                [1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.8263408],
                [1.0, 1.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.60306013],
                [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 0.54506801],
                [1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 0.34276383],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.30412079],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.41702221],
                [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.68130077],
                [1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 0.87545684],
                [1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 0.51042234],
            ]
        ),
        index=pd.RangeIndex(start=50, stop=60, step=1),
        columns=[
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_4",
            "lag_5",
            "exog_2",
            "exog_3",
            "exog_1",
        ],
    )
    for col in ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']:
        expected[col] = pd.Categorical(
                            values     = expected[col],
                            categories = forecaster.class_codes_,
                            ordered    = False
                        )

    pd.testing.assert_frame_equal(predictions, expected)


def test_create_predict_X_output_when_categorical_features_native_implementation_LGBMClassifier_auto():
    """
    Test create_predict_X output when using LGBMClassifier and categorical variables with 
    categorical_features='auto'.
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
                     estimator        = LGBMClassifier(verbose=-1, random_state=123),
                     lags             = 5,
                     transformer_exog = transformer_exog,
                     fit_kwargs       = {'categorical_feature': 'auto'}
                 )
    forecaster.fit(y=y, exog=df_exog)

    warn_msg = re.escape(
        "The output matrix is in the transformed scale due to the "
        "inclusion of transformations (`transformer_exog`) in the Forecaster. "
        "As a result, any predictions generated using this matrix will also "
        "be in the transformed scale. Please refer to the documentation "
        "for more details: "
        "https://skforecast.org/latest/user_guides/training-and-prediction-matrices.html"
    )
    with pytest.warns(DataTransformationWarning, match = warn_msg):
        predictions = forecaster.create_predict_X(steps=10, exog=df_exog_predict)

    expected = pd.DataFrame(
        data=np.array(
            [
                [1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.12062867],
                [1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.8263408],
                [0.0, 1.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.60306013],
                [1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 0.54506801],
                [1.0, 1.0, 0.0, 1.0, 1.0, 4.0, 4.0, 0.34276383],
                [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.30412079],
                [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.41702221],
                [1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.68130077],
                [1.0, 1.0, 0.0, 0.0, 1.0, 3.0, 3.0, 0.87545684],
                [0.0, 1.0, 1.0, 0.0, 0.0, 4.0, 4.0, 0.51042234],
            ]
        ),
        index=pd.RangeIndex(start=50, stop=60, step=1),
        columns=[
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_4",
            "lag_5",
            "exog_2",
            "exog_3",
            "exog_1",
        ],
    )
    for col in ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']:
        expected[col] = pd.Categorical(
                            values     = expected[col],
                            categories = forecaster.class_codes_,
                            ordered    = False
                        )
    expected['exog_2'] = pd.Categorical(
                             values     = expected['exog_2'],
                             categories = [0, 1, 2, 3, 4],
                             ordered    = False
                         )
    expected['exog_3'] = pd.Categorical(
                             values     = expected['exog_3'],
                             categories = [0, 1, 2, 3, 4],
                             ordered    = False
                         )

    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize("steps", 
                         [10, '2020-02-29', pd.to_datetime('2020-02-29')], 
                         ids=lambda steps: f'steps: {steps}')
def test_create_predict_X_output_when_window_features(steps):
    """
    Test output of create_predict_X when estimator is LGBMClassifier and window features.
    """

    rolling = RollingFeaturesClassification(stats=['proportion', 'entropy'], window_sizes=[3, 5])
    forecaster = ForecasterRecursiveClassifier(
        LGBMClassifier(verbose=-1, random_state=123), lags=3, window_features=rolling
    )
    forecaster.fit(y=y_dt, exog=exog_dt)
    predictions = forecaster.create_predict_X(steps=steps, exog=exog_dt_predict)

    expected = pd.DataFrame(
        data=np.array(
            [
                [
                    1.0,
                    2.0,
                    0.0,
                    0.33333333,
                    0.33333333,
                    0.33333333,
                    1.37095059,
                    0.12062867,
                ],
                [2.0, 1.0, 2.0, 0.0, 0.33333333, 0.66666667, 1.52192809, 0.8263408],
                [1.0, 2.0, 1.0, 0.0, 0.66666667, 0.33333333, 1.52192809, 0.60306013],
                [1.0, 1.0, 2.0, 0.0, 0.66666667, 0.33333333, 0.97095059, 0.54506801],
                [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.72192809, 0.34276383],
                [0.0, 1.0, 1.0, 0.33333333, 0.66666667, 0.0, 1.37095059, 0.30412079],
                [
                    2.0,
                    0.0,
                    1.0,
                    0.33333333,
                    0.33333333,
                    0.33333333,
                    1.37095059,
                    0.41702221,
                ],
                [
                    1.0,
                    2.0,
                    0.0,
                    0.33333333,
                    0.33333333,
                    0.33333333,
                    1.37095059,
                    0.68130077,
                ],
                [1.0, 1.0, 2.0, 0.0, 0.66666667, 0.33333333, 1.37095059, 0.87545684],
                [0.0, 1.0, 1.0, 0.33333333, 0.66666667, 0.0, 1.52192809, 0.51042234],
            ]
        ),
        index=pd.date_range("2020-02-20", periods=10),
        columns=[
            "lag_1",
            "lag_2",
            "lag_3",
            "roll_proportion_3_class_0",
            "roll_proportion_3_class_1",
            "roll_proportion_3_class_2",
            "roll_entropy_5",
            "exog",
        ],
    )
    for col in ['lag_1', 'lag_2', 'lag_3']:
        expected[col] = pd.Categorical(
                            values     = expected[col],
                            categories = forecaster.class_codes_,
                            ordered    = False
                        )

    pd.testing.assert_frame_equal(predictions, expected)
