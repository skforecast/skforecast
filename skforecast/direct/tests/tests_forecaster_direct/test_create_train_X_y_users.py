# Unit test create_train_X_y ForecasterDirect
# ==============================================================================
# The public method `create_train_X_y` is a thin wrapper over
# `_create_train_X_y` that converts numpy arrays to pandas
# DataFrame / Series and restores exog dtypes. The numerical
# correctness of X_train and y_train is already tested exhaustively
# in test_create_train_X_y.py (private method). This file only
# validates the conversion: correct types, column names, index,
# and dtype restoration for the different exog dtype paths.
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from skforecast.direct import ForecasterDirect


def _assert_y_train_dict(results_y, expected_y):
    """Helper to assert y_train dicts match."""
    assert isinstance(results_y, dict)
    assert all(isinstance(x, pd.Series) for x in results_y.values())
    assert results_y.keys() == expected_y.keys()
    for key in expected_y:
        pd.testing.assert_series_equal(results_y[key], expected_y[key])


def test_create_train_X_y_output_when_exog_is_None():
    """
    Test create_train_X_y returns (DataFrame, dict of Series) with correct
    index, columns and dtypes when exog is None (exog_dtypes_out_ is None path).
    """
    y = pd.Series(np.arange(10), dtype=float)
    forecaster = ForecasterDirect(LinearRegression(), lags=5, steps=2)
    results = forecaster.create_train_X_y(y=y, exog=None)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.]]),
            index   = pd.RangeIndex(start=6, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8.]),
                index = pd.RangeIndex(start=5, stop=9, step=1),
                name  = 'y_step_1',
                dtype = float
            ),
         2: pd.Series(
                data  = np.array([6., 7., 8., 9.]),
                index = pd.RangeIndex(start=6, stop=10, step=1),
                name  = 'y_step_2',
                dtype = float
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    _assert_y_train_dict(results[1], expected[1])


@pytest.mark.parametrize(
    "categorical_features",
    [None, 'auto', ['exog_3', 'exog_5', 'exog_6']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_exog_is_dataframe_of_float_int_category_bool_str(categorical_features):
    """
    Test create_train_X_y restores exog dtypes correctly for float, int,
    category (int values), bool, string, and category (string values) columns
    (exog_dtypes_out_ is not None path). Covers all three categorical_features
    modes: None (category and str kept as-is), 'auto' and explicit list
    (category and str OrdinalEncoded to float). Bool columns are never treated
    as categorical.
    """
    y = pd.Series(np.arange(10), dtype=float)
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

    forecaster = ForecasterDirect(
        LinearRegression(), lags=5, steps=2,
        categorical_features=categorical_features
    )
    results = forecaster.create_train_X_y(y=y, exog=exog)

    if categorical_features is None:
        # category kept as pd.Categorical, str kept as object
        exog_3_s1 = pd.Categorical(
            [105, 106, 107, 108], categories=list(range(100, 110))
        )
        exog_5_s1 = ['b', 'a', 'b', 'a']
        exog_3_s2 = pd.Categorical(
            [106, 107, 108, 109], categories=list(range(100, 110))
        )
        exog_5_s2 = ['a', 'b', 'a', 'b']
    else:
        # OrdinalEncoded to float
        exog_3_s1 = [5.0, 6.0, 7.0, 8.0]
        exog_5_s1 = [1.0, 0.0, 1.0, 0.0]
        exog_3_s2 = [6.0, 7.0, 8.0, 9.0]
        exog_5_s2 = [0.0, 1.0, 0.0, 1.0]

    expected_X = pd.DataFrame(
        data = np.array([
            [4., 3., 2., 1., 0., 105., 1005.],
            [5., 4., 3., 2., 1., 106., 1006.],
            [6., 5., 4., 3., 2., 107., 1007.],
            [7., 6., 5., 4., 3., 108., 1008.]
        ]),
        index   = pd.RangeIndex(start=6, stop=10, step=1),
        columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
                   'exog_1_step_1', 'exog_2_step_1']
    ).astype({'exog_1_step_1': float, 'exog_2_step_1': int}
    ).assign(
        exog_3_step_1 = exog_3_s1,
        exog_4_step_1 = np.array([False, True, False, True], dtype=bool),
        exog_5_step_1 = exog_5_s1,
    )

    if categorical_features is not None:
        expected_X['exog_6_step_1'] = [1.0, 0.0, 1.0, 0.0]

    # Step 2 exog columns
    expected_X = expected_X.assign(
        exog_1_step_2 = [106., 107., 108., 109.],
        exog_2_step_2 = pd.array([1006, 1007, 1008, 1009], dtype=int),
    ).assign(
        exog_3_step_2 = exog_3_s2,
        exog_4_step_2 = np.array([True, False, True, False], dtype=bool),
        exog_5_step_2 = exog_5_s2,
    )
    if categorical_features is not None:
        expected_X['exog_6_step_2'] = [0.0, 1.0, 0.0, 1.0]

    expected_y = {
        1: pd.Series(
               data  = np.array([5., 6., 7., 8.]),
               index = pd.RangeIndex(start=5, stop=9, step=1),
               name  = 'y_step_1',
               dtype = float
           ),
        2: pd.Series(
               data  = np.array([6., 7., 8., 9.]),
               index = pd.RangeIndex(start=6, stop=10, step=1),
               name  = 'y_step_2',
               dtype = float
           )
    }

    pd.testing.assert_frame_equal(results[0], expected_X)
    _assert_y_train_dict(results[1], expected_y)


def test_create_train_X_y_output_when_transformer_y_and_transformer_exog():
    """
    Test create_train_X_y with transformer_y and transformer_exog to verify
    that columns produced by ColumnTransformer (including one-hot encoded)
    are correctly named and typed in the output DataFrame.
    """
    y = pd.Series(np.arange(8), dtype=float)
    y.index = pd.date_range('1990-01-01', periods=8, freq='D')
    exog = pd.DataFrame(
               {'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4],
                'col_2': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']},
               index=pd.date_range('1990-01-01', periods=8, freq='D')
           )

    transformer_y = StandardScaler()
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['col_1']),
                            ('onehot', OneHotEncoder(), ['col_2'])],
                           remainder='passthrough',
                           verbose_feature_names_out=False
                       )

    forecaster = ForecasterDirect(
                     estimator        = LinearRegression(),
                     lags             = 5,
                     steps            = 2,
                     transformer_y    = transformer_y,
                     transformer_exog = transformer_exog
                 )
    results = forecaster.create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([
                [0.21821789, -0.21821789, -0.65465367, -1.09108945,
                 -1.52752523, -0.25107995,  0.        ,  1.        ,
                  1.79326881,  0.        ,  1.        ],
                [0.65465367,  0.21821789, -0.21821789, -0.65465367,
                 -1.09108945,  1.79326881,  0.        ,  1.        ,
                  0.01673866,  0.        ,  1.        ]
            ]),
            index   = pd.date_range('1990-01-07', periods=2, freq='D'),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
                       'col_1_step_1', 'col_2_a_step_1', 'col_2_b_step_1',
                       'col_1_step_2', 'col_2_a_step_2', 'col_2_b_step_2']
        ),
        {1: pd.Series(
                data  = np.array([0.65465367, 1.09108945]),
                index = pd.date_range('1990-01-06', periods=2, freq='D'),
                name  = 'y_step_1',
                dtype = float
            ),
         2: pd.Series(
                data  = np.array([1.09108945, 1.52752523]),
                index = pd.date_range('1990-01-07', periods=2, freq='D'),
                name  = 'y_step_2',
                dtype = float
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    _assert_y_train_dict(results[1], expected[1])


@pytest.mark.parametrize(
    "categorical_features",
    ['auto', ['col_2']],
    ids=lambda cf: f'categorical_features: {cf}'
)
def test_create_train_X_y_output_when_transformer_exog_is_make_column_transformer_and_categorical(categorical_features):
    """
    Test the output of create_train_X_y when using make_column_transformer
    with StandardScaler only for numeric columns and a string categorical
    column passed through as remainder. With set_output(transform='pandas'),
    the category dtype is preserved and 'auto' correctly detects col_2.
    OrdinalEncoder maps ['a'..'c'] -> [0.0..2.0].
    """
    y = pd.Series(np.arange(10), dtype=float)
    y.index = pd.date_range('1990-01-01', periods=10, freq='D')
    exog = pd.DataFrame(
               {'col_1': [7.5, 24.4, 60.3, 57.3, 50.7,
                          41.4, 87.2, 47.4, 30.1, 22.3],
                'col_2': pd.Categorical(
                    ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'c']
                )},
               index=pd.date_range('1990-01-01', periods=10, freq='D')
           )

    transformer_exog = make_column_transformer(
                           (StandardScaler(), ['col_1']),
                           remainder='passthrough',
                           verbose_feature_names_out=False
                       ).set_output(transform='pandas')

    forecaster = ForecasterDirect(
                     estimator            = LinearRegression(),
                     lags                 = 5,
                     steps                = 2,
                     transformer_exog     = transformer_exog,
                     categorical_features = categorical_features
                 )
    results = forecaster.create_train_X_y(y=y, exog=exog)

    # OrdinalEncoder maps ['a'..'c'] -> [0..2]
    expected = (
        pd.DataFrame(
            data = np.array([
                [4., 3., 2., 1., 0., -0.06706325, 2., 2.03670162, 0.],
                [5., 4., 3., 2., 1.,  2.03670162, 0., 0.20853914, 1.],
                [6., 5., 4., 3., 2.,  0.20853914, 1., -0.5861144, 2.],
                [7., 6., 5., 4., 3., -0.5861144,  2., -0.9443975, 2.]
            ]),
            index   = pd.date_range('1990-01-07', periods=4, freq='D'),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
                       'col_1_step_1', 'col_2_step_1',
                       'col_1_step_2', 'col_2_step_2']
        ),
        {1: pd.Series(
                data  = np.array([5., 6., 7., 8.]),
                index = pd.date_range('1990-01-06', periods=4, freq='D'),
                name  = 'y_step_1',
                dtype = float
            ),
         2: pd.Series(
                data  = np.array([6., 7., 8., 9.]),
                index = pd.date_range('1990-01-07', periods=4, freq='D'),
                name  = 'y_step_2',
                dtype = float
            )
        }
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    _assert_y_train_dict(results[1], expected[1])
