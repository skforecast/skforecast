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
from catboost import CatBoostClassifier

from skforecast.preprocessing import RollingFeaturesClassification
from skforecast.exceptions import MissingValuesWarning
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
        {"estimator": LogisticRegression(), "lags": 3,
         "transformer_exog": make_pipeline(StandardScaler(copy=False))},
    ],
    ids=["base", "window_features", "transformers", "in_place_transformers"]
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


@pytest.mark.parametrize(
    "transformer_exog",
    [None, FunctionTransformer()],
    ids=["no_transformer_exog", "identity_transformer_exog"]
)
@pytest.mark.parametrize(
    "categorical_features",
    ["auto", ["exog_1", "exog_2", "exog_3"]],
    ids=["auto", "list_with_float_column"]
)
def test_predict_does_not_modify_exog_with_categorical_features(
    categorical_features, transformer_exog
):
    """
    Test forecaster.fit and forecaster.predict do not modify exog and
    exog_predict when the categorical features are encoded without
    `transformer_exog` or with one that returns its input (identity
    `FunctionTransformer`). With the list, the float column `exog_1` is also
    encoded, with codes of its same dtype, while `exog_4`, which pandas
    stores in the same array, is not.
    """
    y_local = pd.Series(np.tile(['x', 'y'], 10), name='y')
    exog_local = pd.DataFrame({
        'exog_1': np.tile([1., 2., 3., 4.], 5),
        'exog_2': pd.Categorical(np.tile(['a', 'b', 'c', 'd'], 5)),
        'exog_3': np.tile(['u', 'v'], 10),
        'exog_4': np.arange(20, dtype=float)
    })
    exog_predict_local = pd.DataFrame(
        {
            'exog_1': [1., 2.],
            'exog_2': pd.Categorical(['a', 'b'], categories=['a', 'b', 'c', 'd']),
            'exog_3': ['u', 'v'],
            'exog_4': [20., 21.]
        },
        index=pd.RangeIndex(20, 22)
    )
    exog_copy = exog_local.copy()
    exog_predict_copy = exog_predict_local.copy()

    forecaster = ForecasterRecursiveClassifier(
                     estimator            = LogisticRegression(),
                     lags                 = 3,
                     transformer_exog     = transformer_exog,
                     categorical_features = categorical_features
                 )
    forecaster.fit(y=y_local, exog=exog_local)
    _ = forecaster.predict(steps=2, exog=exog_predict_local)

    pd.testing.assert_frame_equal(exog_local, exog_copy)
    pd.testing.assert_frame_equal(exog_predict_local, exog_predict_copy)


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
    'categorical_features',
    ['auto', ['exog_2', 'exog_3']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_predict_output_when_categorical_features_CatBoostClassifier(categorical_features):
    """
    Test predict output when using CatBoostClassifier with native categorical
    features: the lags (`features_encoding='auto'`) and the categorical
    exogenous variables. CatBoost requires these columns as integers when
    predicting from a numpy array.
    """
    df_exog = pd.DataFrame(
        {'exog_1': exog.to_numpy(),
         'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
         'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10)}
    )

    df_exog_predict = df_exog.iloc[:10, :].copy()
    df_exog_predict.index = pd.RangeIndex(start=50, stop=60)

    forecaster = ForecasterRecursiveClassifier(
                     estimator            = CatBoostClassifier(
                                                iterations          = 100,
                                                depth               = 3,
                                                random_seed         = 123,
                                                verbose             = 0,
                                                allow_writing_files = False
                                            ),
                     lags                 = 5,
                     categorical_features = categorical_features
                 )
    forecaster.fit(y=y, exog=df_exog)
    predictions = forecaster.predict(steps=10, exog=df_exog_predict)

    expected = pd.Series(
                   data = np.array([2, 1, 2, 2, 2, 3, 2, 2, 1, 2]),
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


def test_predict_output_when_last_window_stored_has_NaN():
    """
    Test predict output when the stored last_window_ contains NaN values.
    Estimator: HistGradientBoostingClassifier (supports NaN natively).
    """
    y_nan = pd.Series(
        data  = [1, 2, 1, 2, 1, 2, 1, np.nan, 2, 1],
        name  = 'y',
        dtype = float
    )
    forecaster = ForecasterRecursiveClassifier(
                     estimator          = HistGradientBoostingClassifier(random_state=123),
                     lags               = 3,
                     dropna_from_series = True
                 )

    warn_msg = re.escape(
        "Some rows of the training matrix have missing values. 3 of 7 rows "
        "are dropped: 1 with a NaN target and 2 with NaN predictors (lags, "
        "window features or exog). They come from interspersed NaNs in `y` "
        "or `exog`. To keep the rows with NaN predictors, set "
        "`forecaster.dropna_from_series = False` (the estimator must "
        "support NaN)."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        forecaster.fit(y=y_nan)

    assert forecaster.last_window_.isna().any().any()

    warn_msg = re.escape(
        "`last_window` has missing values."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        predictions = forecaster.predict(steps=3)

    expected = pd.Series(
                   data  = np.array([1.0, 1.0, 1.0]),
                   index = pd.RangeIndex(start=10, stop=13, step=1),
                   name  = 'pred'
               )

    pd.testing.assert_series_equal(predictions, expected)


def test_predict_output_when_last_window_argument_has_NaN():
    """
    Test predict output when a custom last_window with NaN values is passed
    to the predict method. Estimator: HistGradientBoostingClassifier.
    """
    y_clean = pd.Series(
        data  = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        name  = 'y',
        dtype = float
    )
    forecaster = ForecasterRecursiveClassifier(
                     estimator          = HistGradientBoostingClassifier(random_state=123),
                     lags               = 3,
                     dropna_from_series = False
                 )
    forecaster.fit(y=y_clean)

    last_window_nan = pd.Series(
        data  = [np.nan, 1.0, 2.0],
        index = pd.RangeIndex(start=17, stop=20)
    )

    warn_msg = re.escape(
        "`last_window` has missing values."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        predictions = forecaster.predict(steps=3, last_window=last_window_nan)

    expected = pd.Series(
                   data  = np.array([2.0, 2.0, 2.0]),
                   index = pd.RangeIndex(start=20, stop=23, step=1),
                   name  = 'pred'
               )

    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize(
    'dtype',
    ['bool', 'Int64', 'double[pyarrow]'],
    ids=lambda dtype: f'dtype: {dtype}'
)
def test_predict_proba_output_when_exog_has_bool_nullable_or_pyarrow_column(dtype):
    """
    Test predict_proba output when exog has a bool, nullable or pyarrow column
    next to a float column. The values of exog are a float64 array, not an
    object array, and the predictions are the same for all the dtypes.
    """
    if 'pyarrow' in dtype:
        pytest.importorskip('pyarrow')
    exog_2 = [1, 0, 0, 1, 1] * 10
    if dtype == 'bool':
        exog_2 = [bool(value) for value in exog_2]
    df_exog = pd.DataFrame({
        'exog_1': exog.to_numpy(),
        'exog_2': pd.array(exog_2, dtype=dtype)
    })
    exog_predict = df_exog.iloc[:5].copy()
    exog_predict.index = pd.RangeIndex(start=50, stop=55)

    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)
    forecaster.fit(y=y, exog=df_exog)
    exog_values = forecaster._create_predict_inputs(steps=5, exog=exog_predict)[1]
    predictions = forecaster.predict_proba(steps=5, exog=exog_predict)

    expected = pd.DataFrame(
                   data    = np.array([[0.27483561, 0.46440463, 0.26075975],
                                       [0.19098039, 0.52627509, 0.28274452],
                                       [0.21704984, 0.56452178, 0.21842838],
                                       [0.24245638, 0.44716149, 0.31038213],
                                       [0.22495508, 0.43186334, 0.34318158]]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1_proba', '2_proba', '3_proba']
               )

    assert exog_values.dtype == np.float64
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_proba_output_when_HistGradientBoostingClassifier_and_exog_has_float_and_bool():
    """
    Test predict_proba output when using HistGradientBoostingClassifier and
    exog has a float column and a bool column.
    """
    df_exog = pd.DataFrame({
        'exog_1': exog.to_numpy(),
        'exog_2': [True, False, False, True, True] * 10
    })
    exog_predict = df_exog.iloc[:5].copy()
    exog_predict.index = pd.RangeIndex(start=50, stop=55)

    forecaster = ForecasterRecursiveClassifier(
        HistGradientBoostingClassifier(random_state=123), lags=3
    )
    forecaster.fit(y=y, exog=df_exog)
    predictions = forecaster.predict_proba(steps=5, exog=exog_predict)

    expected = pd.DataFrame(
                   data    = np.array([[0.11439177, 0.51582837, 0.36977986],
                                       [0.38509477, 0.387823  , 0.22708223],
                                       [0.48872209, 0.32246487, 0.18881304],
                                       [0.34361371, 0.4637224 , 0.19266389],
                                       [0.16468338, 0.48653569, 0.34878093]]),
                   index   = pd.RangeIndex(start=50, stop=55, step=1),
                   columns = ['1_proba', '2_proba', '3_proba']
               )

    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize(
    'dtype',
    ['Int64', 'Float64', 'boolean'],
    ids=lambda dtype: f'dtype: {dtype}'
)
def test_predict_MissingValuesWarning_and_ValueError_when_nullable_exog_has_missing_values(dtype):
    """
    Test predict issues a MissingValuesWarning and the estimator raises its
    ValueError of NaN, not a TypeError, when a nullable column of exog has a
    missing value and the estimator does not accept NaN.
    """
    exog_2 = [1, 0, 0, 1, 1] * 10
    if dtype == 'boolean':
        exog_2 = [bool(value) for value in exog_2]
    df_exog = pd.DataFrame({
        'exog_1': exog.to_numpy(),
        'exog_2': pd.array(exog_2, dtype=dtype)
    })
    exog_predict = df_exog.iloc[:5].copy()
    exog_predict.index = pd.RangeIndex(start=50, stop=55)
    exog_predict.loc[51, 'exog_2'] = pd.NA

    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)
    forecaster.fit(y=y, exog=df_exog)

    warn_msg = re.escape("`exog` has missing values.")
    err_msg = re.escape("Input X contains NaN.")
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        with pytest.raises(ValueError, match=err_msg):
            forecaster.predict(steps=5, exog=exog_predict)
