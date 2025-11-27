# Unit test _create_predict_inputs ForecasterRecursiveClassifier
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.recursive import ForecasterRecursiveClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

# Fixtures
from .fixtures_forecaster_recursive_classifier import y, y_dt
from .fixtures_forecaster_recursive_classifier import exog


def test_create_predict_inputs_NotFittedError_when_fitted_is_False():
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
        forecaster._create_predict_inputs(steps=5)


def test_create_predict_inputs_ValueError_when_last_window_contains_invalid_classes():
    """
    Test ValueError is raised when last_window contains classes not present in training y.
    """
    forecaster = ForecasterRecursiveClassifier(
                     estimator = LogisticRegression(),
                     lags      = 5
                 )
    forecaster.fit(y=y)

    last_window = pd.Series(
        np.array([1, 2, 3, 4, 5]),
        index = pd.RangeIndex(start=45, stop=50),
        name='y'
    )

    valid_classes = set(forecaster.encoding_mapping_.keys())
    unique_values = set(last_window.to_numpy())
    invalid_values = unique_values - valid_classes
    invalid_list = sorted(list(invalid_values))[:5]
    valid_list = sorted(list(valid_classes))[:10]

    err_msg = re.escape(
        f"The `last_window` contains {len(invalid_values)} class label(s) "
        f"not seen during training: {invalid_list}{'...' if len(invalid_values) > 5 else ''}.\n"
        f"Valid class labels (seen during training): {valid_list}"
        f"{'...' if len(valid_classes) > 10 else ''}.\n"
        f"Total valid classes: {len(valid_classes)}."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_predict_inputs(steps=5, last_window=last_window)


def test_create_predict_inputs_when_estimator_is_LogisticRegression():
    """
    Test _create_predict_inputs when using LogisticRegression as estimator.
    """
    forecaster = ForecasterRecursiveClassifier(
                     estimator = LogisticRegression(),
                     lags      = 5
                 )
    forecaster.fit(y=pd.Series(np.arange(50, dtype=float), name='y'))
    results = forecaster._create_predict_inputs(steps=5)

    expected = (
        np.array([45., 46., 47., 48., 49.]),
        None,
        pd.RangeIndex(start=50, stop=55, step=1),
        5
    )
    
    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert results[1] is None
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


def test_create_predict_inputs_when_estimator_is_HistGradientBoostingClassifier():
    """
    Test _create_predict_inputs when using HistGradientBoostingClassifier as estimator.
    """
    forecaster = ForecasterRecursiveClassifier(
                     estimator = HistGradientBoostingClassifier(),
                     lags      = 5
                 )
    forecaster.fit(y=y_dt)
    results = forecaster._create_predict_inputs(steps=5)

    expected = (
        np.array([1, 1, 0, 2, 1]),
        None,
        pd.date_range(start='2020-02-20', periods=5, freq='D'),
        5
    )
    
    np.testing.assert_array_almost_equal(results[0], expected[0])
    assert results[1] is None
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


def test_create_predict_inputs_when_with_transform_exog():
    """
    Test _create_predict_inputs when using LogisticRegression as estimator and
    StandardScaler as transformer_exog.
    """
    exog_dummy = pd.Series(np.array([7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4]), name='exog')
    exog_predict = exog_dummy.copy()
    exog_predict.index = pd.RangeIndex(start=8, stop=16)

    forecaster = ForecasterRecursiveClassifier(
                     estimator        = LogisticRegression(),
                     lags             = 5,
                     transformer_exog = StandardScaler()
                 )
    forecaster.fit(y=y.iloc[:8], exog=exog_dummy)
    results = forecaster._create_predict_inputs(steps=5, exog=exog_predict)

    expected = (
        np.array([1., 1., 0., 1., 1.]),
        np.array([[-1.76425513], [-1.00989936], [0.59254869], [0.45863938], [0.1640389]]),
        pd.RangeIndex(start=8, stop=13, step=1),
        5
    )
    
    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


def test_create_predict_inputs_when_categorical_features_native_implementation_HistGradientBoostingClassifier():
    """
    Test _create_predict_inputs when using HistGradientBoostingClassifier and categorical variables.
    """
    df_exog = pd.DataFrame(
        {'exog_1': exog.to_numpy(),
         'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
         'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)}
    )
    
    exog_predict = df_exog.copy()
    exog_predict.index = pd.RangeIndex(start=50, stop=100)

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
    results = forecaster._create_predict_inputs(steps=10, exog=exog_predict)
    
    expected = (
        np.array([1, 1, 0, 2, 1]),
        np.array([[0.        , 0.        , 0.12062867],
                  [1.        , 1.        , 0.8263408 ],
                  [2.        , 2.        , 0.60306013],
                  [3.        , 3.        , 0.54506801],
                  [4.        , 4.        , 0.34276383],
                  [0.        , 0.        , 0.30412079],
                  [1.        , 1.        , 0.41702221],
                  [2.        , 2.        , 0.68130077],
                  [3.        , 3.        , 0.87545684],
                  [4.        , 4.        , 0.51042234]]),
        pd.RangeIndex(start=50, stop=60, step=1),
        10
    )
    
    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]
