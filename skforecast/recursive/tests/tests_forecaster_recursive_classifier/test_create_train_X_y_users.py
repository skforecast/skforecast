# Unit test create_train_X_y ForecasterRecursiveClassifier
# ==============================================================================
# The public method `create_train_X_y` is a thin wrapper over
# `_create_train_X_y` that converts numpy arrays to pandas
# DataFrame / Series and restores exog dtypes. The numerical
# correctness of X_train and y_train is already tested exhaustively
# in test_create_train_X_y.py (private method). This file only
# validates the conversion: correct types, column names, index,
# dtype restoration for the different exog dtype paths, and the
# `encoded` parameter behavior.
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from skforecast.recursive import ForecasterRecursiveClassifier


def test_create_train_X_y_output_when_encoded_False():
    """
    Test the output of create_train_X_y when encoded is False.
    """
    y = pd.Series(np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']), name='y')
    exog = None
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)
    results = forecaster.create_train_X_y(y=y, exog=exog, encoded=False)

    expected = (
        pd.DataFrame(
            data = np.array([['b', 'a', 'c', 'b', 'a'],
                             ['c', 'b', 'a', 'c', 'b'],
                             ['a', 'c', 'b', 'a', 'c'],
                             ['b', 'a', 'c', 'b', 'a'],
                             ['c', 'b', 'a', 'c', 'b']]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ),
        pd.Series(
            data  = np.array(['c', 'a', 'b', 'c', 'a']),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y'
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_create_train_X_y_output_when_encoded_True():
    """
    Test the output of create_train_X_y when encoded is True.
    """
    y = pd.Series(np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']), name='y')
    exog = None
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)
    results = forecaster.create_train_X_y(y=y, exog=exog, encoded=True)

    expected = (
        pd.DataFrame(
            data = np.array([[1, 0, 2, 1, 0],
                             [2, 1, 0, 2, 1],
                             [0, 2, 1, 0, 2],
                             [1, 0, 2, 1, 0],
                             [2, 1, 0, 2, 1]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ),
        pd.Series(
            data  = np.array([2, 0, 1, 2, 0]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = int
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_create_train_X_y_output_when_encoded_True_as_categorical():
    """
    Test the output of create_train_X_y when encoded is True and lags are categorical.
    """
    y = pd.Series(np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']), name='y')
    exog = None
    forecaster = ForecasterRecursiveClassifier(LGBMClassifier(verbose=-1), lags=5)
    results = forecaster.create_train_X_y(y=y, exog=exog, encoded=True)

    expected = (
        pd.DataFrame(
            data = np.array([[1, 0, 2, 1, 0],
                             [2, 1, 0, 2, 1],
                             [0, 2, 1, 0, 2],
                             [1, 0, 2, 1, 0],
                             [2, 1, 0, 2, 1]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ),
        pd.Series(
            data  = np.array([2, 0, 1, 2, 0]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = int
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_create_train_X_y_output_when_encoded_True_as_categorical_after_fit():
    """
    Test the output of create_train_X_y when encoded is True and lags are 
    categorical and after fitting the forecaster.
    """
    y = pd.Series(np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']), name='y')
    exog = None
    forecaster = ForecasterRecursiveClassifier(LGBMClassifier(verbose=-1), lags=5)
    forecaster.fit(y=y)
    results = forecaster.create_train_X_y(y=y, exog=exog, encoded=True)

    expected = (
        pd.DataFrame(
            data = np.array([[1, 0, 2, 1, 0],
                             [2, 1, 0, 2, 1],
                             [0, 2, 1, 0, 2],
                             [1, 0, 2, 1, 0],
                             [2, 1, 0, 2, 1]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ),
        pd.Series(
            data  = np.array([2, 0, 1, 2, 0]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = int
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


@pytest.mark.parametrize(
    "categorical_features",
    [None, 'auto', ['exog_3', 'exog_5', 'exog_6']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_exog_is_dataframe_of_float_int_category_bool_str(categorical_features):
    """
    Test create_train_X_y restores exog dtypes correctly for float, int,
    category (int values), bool, string, and category (string values) columns.
    Covers all three categorical_features modes: None (category and str kept
    as-is), 'auto' and explicit list (category and str OrdinalEncoded to float).
    Bool columns are never treated as categorical. Lags are encoded class codes
    cast to float by dtype restoration.
    """
    y = pd.Series(
        np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']), name='y'
    )
    exog = pd.DataFrame({
        'exog_1': pd.Series(np.arange(100, 110), dtype=float),
        'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
        'exog_3': pd.Categorical(range(100, 110)),
        'exog_4': pd.Series([True, False] * 5, dtype=bool),
        'exog_5': pd.Series(
            ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'], dtype=str
        ),
    })
    if categorical_features is not None:
        exog['exog_6'] = pd.Categorical(['a', 'b'] * 5)

    forecaster = ForecasterRecursiveClassifier(
        LogisticRegression(), lags=5, categorical_features=categorical_features
    )
    results = forecaster.create_train_X_y(y=y, exog=exog, encoded=True)

    if categorical_features is None:
        exog_3_col = pd.Categorical(range(105, 110), categories=range(100, 110))
        exog_5_col = ['b', 'a', 'b', 'a', 'b']
    else:
        exog_3_col = [5.0, 6.0, 7.0, 8.0, 9.0]
        exog_5_col = [1.0, 0.0, 1.0, 0.0, 1.0]

    expected_X = pd.DataFrame(
        data = np.array([[1., 0., 2., 1., 0., 105., 1005.],
                         [2., 1., 0., 2., 1., 106., 1006.],
                         [0., 2., 1., 0., 2., 107., 1007.],
                         [1., 0., 2., 1., 0., 108., 1008.],
                         [2., 1., 0., 2., 1., 109., 1009.]]),
        index   = pd.RangeIndex(start=5, stop=10, step=1),
        columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
                   'exog_1', 'exog_2']
    ).astype({'exog_1': float, 'exog_2': int}
    ).assign(
        exog_3 = exog_3_col,
        exog_4 = np.array([False, True, False, True, False], dtype=bool),
        exog_5 = exog_5_col,
    )
    if categorical_features is not None:
        expected_X['exog_6'] = [1.0, 0.0, 1.0, 0.0, 1.0]

    expected_y = pd.Series(
        data  = np.array([2, 0, 1, 2, 0]),
        index = pd.RangeIndex(start=5, stop=10, step=1),
        name  = 'y',
        dtype = int
    )

    pd.testing.assert_frame_equal(results[0], expected_X)
    pd.testing.assert_series_equal(results[1], expected_y)


def test_create_train_X_y_output_when_transformer_exog():
    """
    Test create_train_X_y with transformer_exog (ColumnTransformer) to verify
    that columns produced by ColumnTransformer (including one-hot encoded)
    are correctly named and typed in the output DataFrame. Lags are encoded
    class codes cast to float by dtype restoration.
    """
    y = pd.Series(
        np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b']), name='y'
    )
    y.index = pd.date_range('1990-01-01', periods=8, freq='D')
    exog = pd.DataFrame(
               {'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4],
                'col_2': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']},
               index = pd.date_range('1990-01-01', periods=8, freq='D')
           )

    transformer_exog = ColumnTransformer(
                            [('scale', StandardScaler(), ['col_1']),
                             ('onehot', OneHotEncoder(), ['col_2'])],
                            remainder = 'passthrough',
                            verbose_feature_names_out = False
                        )

    forecaster = ForecasterRecursiveClassifier(
                    estimator        = LogisticRegression(),
                    lags             = 5,
                    transformer_exog = transformer_exog
                )
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[1., 0., 2., 1., 0., -0.25107995, 0., 1.],
                             [2., 1., 0., 2., 1.,  1.79326881, 0., 1.],
                             [0., 2., 1., 0., 2.,  0.01673866, 0., 1.]]),
            index   = pd.date_range('1990-01-06', periods=3, freq='D'),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'col_1',
                       'col_2_a', 'col_2_b']
        ),
        pd.Series(
            data  = np.array([2, 0, 1]),
            index = pd.date_range('1990-01-06', periods=3, freq='D'),
            name  = 'y',
            dtype = int
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


@pytest.mark.parametrize(
    "categorical_features",
    ['auto', ['col_2']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_transformer_exog_is_make_column_transformer_and_categorical(categorical_features):
    """
    Test the output of create_train_X_y when using make_column_transformer
    with StandardScaler only for numeric columns and a categorical column
    passed through as remainder. With set_output(transform='pandas'), the
    category dtype is preserved and 'auto' correctly detects col_2.
    OrdinalEncoder maps ['a'..'c'] -> [0.0..2.0]. Lags are encoded class
    codes cast to float by dtype restoration.
    """
    y = pd.Series(
        np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']), name='y'
    )
    y.index = pd.date_range('1990-01-01', periods=10, freq='D')
    exog = pd.DataFrame(
               {'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4, 30.1, 22.3],
                'col_2': pd.Categorical(
                    ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'c']
                )},
               index = pd.date_range('1990-01-01', periods=10, freq='D')
           )

    transformer_exog = make_column_transformer(
                           (StandardScaler(), ['col_1']),
                           remainder='passthrough',
                           verbose_feature_names_out=False
                       ).set_output(transform='pandas')

    forecaster = ForecasterRecursiveClassifier(
                     estimator        = LogisticRegression(),
                     lags             = 5,
                     transformer_exog = transformer_exog,
                     categorical_features = categorical_features
                 )
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array(
                [[1., 0., 2., 1., 0., -0.06706325, 2.],
                 [2., 1., 0., 2., 1.,  2.03670162, 0.],
                 [0., 2., 1., 0., 2.,  0.20853914, 1.],
                 [1., 0., 2., 1., 0., -0.5861144 , 2.],
                 [2., 1., 0., 2., 1., -0.9443975 , 2.]]
            ),
            index   = pd.date_range('1990-01-06', periods=5, freq='D'),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
                       'col_1', 'col_2']
        ),
        pd.Series(
            data  = np.array([2, 0, 1, 2, 0]),
            index = pd.date_range('1990-01-06', periods=5, freq='D'),
            name  = 'y',
            dtype = int
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])

