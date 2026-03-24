# Unit test _create_predict_inputs ForecasterRecursiveClassifier
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.recursive import ForecasterRecursiveClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier

from skforecast.preprocessing import RollingFeaturesClassification

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
def test_create_predict_inputs_does_not_modify_y_exog(forecaster_kwargs):
    """
    Test _create_predict_inputs does not modify y, exog, exog_predict or
    last_window.
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
    _ = forecaster._create_predict_inputs(
            steps=5, exog=exog_predict_local, last_window=last_window_local
        )

    pd.testing.assert_series_equal(y_local, y_copy)
    pd.testing.assert_series_equal(exog_local, exog_copy)
    pd.testing.assert_series_equal(last_window_local, last_window_copy)
    pd.testing.assert_series_equal(exog_predict_local, exog_predict_copy)


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


@pytest.mark.parametrize(
    'categorical_features',
    ['auto', ['exog_2', 'exog_3']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_predict_inputs_when_categorical_features_HistGradientBoostingClassifier(categorical_features):
    """
    Test _create_predict_inputs when using HistGradientBoostingClassifier and
    categorical variables managed by the forecaster's `categorical_features`
    parameter.
    """
    df_exog = pd.DataFrame(
        {'exog_1': exog.to_numpy(),
         'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
         'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)}
    )
    
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
    results = forecaster._create_predict_inputs(steps=10, exog=df_exog_predict)
    
    expected = (
        np.array([1, 1, 0, 2, 1]),
        np.array([[0.12062867, 0.        , 0.        ],
                  [0.8263408 , 1.        , 1.        ],
                  [0.60306013, 2.        , 2.        ],
                  [0.54506801, 3.        , 3.        ],
                  [0.34276383, 4.        , 4.        ],
                  [0.30412079, 0.        , 0.        ],
                  [0.41702221, 1.        , 1.        ],
                  [0.68130077, 2.        , 2.        ],
                  [0.87545684, 3.        , 3.        ],
                  [0.51042234, 4.        , 4.        ]]),
        pd.RangeIndex(start=50, stop=60, step=1),
        10
    )
    
    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


@pytest.mark.parametrize(
    'categorical_features',
    ['auto', ['exog_2', 'exog_3']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_predict_inputs_when_categorical_features_LGBMClassifier(categorical_features):
    """
    Test _create_predict_inputs when using LGBMClassifier and categorical
    variables managed by the forecaster's `categorical_features` parameter.
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
    results = forecaster._create_predict_inputs(steps=10, exog=df_exog_predict)
    
    expected = (
        np.array([1, 1, 0, 2, 1]),
        np.array([[0.12062867, 0.        , 0.        ],
                  [0.8263408 , 1.        , 1.        ],
                  [0.60306013, 2.        , 2.        ],
                  [0.54506801, 3.        , 3.        ],
                  [0.34276383, 4.        , 4.        ],
                  [0.30412079, 0.        , 0.        ],
                  [0.41702221, 1.        , 1.        ],
                  [0.68130077, 2.        , 2.        ],
                  [0.87545684, 3.        , 3.        ],
                  [0.51042234, 4.        , 4.        ]]),
        pd.RangeIndex(start=50, stop=60, step=1),
        10
    )
    
    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]


@pytest.mark.parametrize(
    'features_encoding, categorical_features, use_exog_cat, expected_exog',
    [
        ('auto', 'auto', True,
         np.array([[0.12062867, 0.        , 0.        ],
                   [0.8263408 , 1.        , 1.        ],
                   [0.60306013, 2.        , 2.        ],
                   [0.54506801, 3.        , 3.        ],
                   [0.34276383, 4.        , 4.        ],
                   [0.30412079, 0.        , 0.        ],
                   [0.41702221, 1.        , 1.        ],
                   [0.68130077, 2.        , 2.        ],
                   [0.87545684, 3.        , 3.        ],
                   [0.51042234, 4.        , 4.        ]])),
        ('auto', None, False,
         np.array([[0.12062867],
                   [0.8263408 ],
                   [0.60306013],
                   [0.54506801],
                   [0.34276383],
                   [0.30412079],
                   [0.41702221],
                   [0.68130077],
                   [0.87545684],
                   [0.51042234]])),
        ('ordinal', 'auto', True,
         np.array([[0.12062867, 0.        , 0.        ],
                   [0.8263408 , 1.        , 1.        ],
                   [0.60306013, 2.        , 2.        ],
                   [0.54506801, 3.        , 3.        ],
                   [0.34276383, 4.        , 4.        ],
                   [0.30412079, 0.        , 0.        ],
                   [0.41702221, 1.        , 1.        ],
                   [0.68130077, 2.        , 2.        ],
                   [0.87545684, 3.        , 3.        ],
                   [0.51042234, 4.        , 4.        ]])),
        ('ordinal', None, False,
         np.array([[0.12062867],
                   [0.8263408 ],
                   [0.60306013],
                   [0.54506801],
                   [0.34276383],
                   [0.30412079],
                   [0.41702221],
                   [0.68130077],
                   [0.87545684],
                   [0.51042234]])),
    ],
    ids=[
        'autoreg_cat-exog_cat',
        'autoreg_cat-no_exog_cat',
        'no_autoreg_cat-exog_cat',
        'no_autoreg_cat-no_exog_cat',
    ]
)
def test_create_predict_inputs_when_features_encoding_and_categorical_features_combinations(
    features_encoding, categorical_features, use_exog_cat, expected_exog
):
    """
    Test _create_predict_inputs output for all combinations of
    `features_encoding` (autoreg categorical) and `categorical_features`
    (exog categorical) with LGBMClassifier.
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
    results = forecaster._create_predict_inputs(steps=10, exog=df_exog_predict)

    np.testing.assert_array_almost_equal(results[0], np.array([1, 1, 0, 2, 1]))
    np.testing.assert_array_almost_equal(results[1], expected_exog)
    pd.testing.assert_index_equal(results[2], pd.RangeIndex(start=50, stop=60, step=1))
    assert results[3] == 10


@pytest.mark.parametrize(
    "steps",
    [10, '2020-02-29', pd.to_datetime('2020-02-29')],
    ids=lambda steps: f'steps: {steps}'
)
def test_create_predict_inputs_when_window_features(steps):
    """
    Test _create_predict_inputs when estimator is LGBMClassifier and window
    features.
    """
    rolling = RollingFeaturesClassification(
                  stats=['proportion', 'entropy'], window_sizes=[3, 5]
              )
    forecaster = ForecasterRecursiveClassifier(
                     LGBMClassifier(verbose=-1, random_state=123),
                     lags=3, window_features=rolling
                 )
    forecaster.fit(y=y_dt, exog=exog_dt)
    results = forecaster._create_predict_inputs(steps=steps, exog=exog_dt_predict)

    expected = (
        np.array([1, 1, 0, 2, 1]),
        np.array([[0.12062867],
                  [0.8263408 ],
                  [0.60306013],
                  [0.54506801],
                  [0.34276383],
                  [0.30412079],
                  [0.41702221],
                  [0.68130077],
                  [0.87545684],
                  [0.51042234]]),
        pd.date_range(start='2020-02-20', periods=10, freq='D'),
        10
    )

    np.testing.assert_array_almost_equal(results[0], expected[0])
    np.testing.assert_array_almost_equal(results[1], expected[1])
    pd.testing.assert_index_equal(results[2], expected[2])
    assert results[3] == expected[3]
