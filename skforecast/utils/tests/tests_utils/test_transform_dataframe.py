# Unit test transform_dataframe
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from skforecast.utils import transform_dataframe


def test_transform_dataframe_TypeError_when_df_is_not_pandas_DataFrame():
    """
    Test TypeError is raised when df is not a pandas DataFrame.
    """
    df = pd.Series(np.arange(10), name='y')

    err_msg = re.escape(f"`df` argument must be a pandas DataFrame. Got {type(df)}")
    with pytest.raises(TypeError, match = err_msg):
        transform_dataframe(
            df                = df,
            transformer       = None,
            fit               = True,
            inverse_transform = False
        )


def test_transform_dataframe_ValueError_when_transformer_is_ColumnTransformer_and_inverse_transform_is_true():
    """
    Test that transform_dataframe raise ValueError when transformer is ColumnTransformer
    and argument inverse_transform is True.
    """
    df_input = pd.DataFrame({
                   'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4],
                   'col_2': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
               })
    transformer = ColumnTransformer(
                      [('scale', StandardScaler(), ['col_1']),
                       ('onehot', OneHotEncoder(sparse_output=False), ['col_2'])],
                      remainder = 'passthrough',
                      verbose_feature_names_out = False
                  )

    err_msg = re.escape("`inverse_transform` is not available when using ColumnTransformers.")
    with pytest.raises(ValueError, match = err_msg):
        transform_dataframe(
            df                = df_input,
            transformer       = transformer,
            fit               = True,
            inverse_transform = True
        )


def test_transform_dataframe_when_transformer_is_None():
    """
    Test the output of transform_dataframe when transformer is None.
    """
    df_input = pd.DataFrame({
                   'A': [1.16, -0.28, 0.07, 2.4, 0.25, -0.56, -1.42, 1.26, 1.78, -1.49],
                   'B': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4, 2.7, 59.6]
               })  
    expected = df_input
    transformer = None
    results = transform_dataframe(
                  df                = df_input,
                  transformer       = transformer,
                  fit               = True,
                  inverse_transform = False
              )
    
    pd.testing.assert_frame_equal(results, expected)


def test_transform_dataframe_when_transformer_is_StandardScaler():
    """
    Test the output of transform_dataframe when transformer is StandardScaler.
    """
    df_input = pd.DataFrame({
                   'A': [1.16, -0.28, 0.07, 2.4, 0.25, -0.56, -1.42, 1.26, 1.78, -1.49],
                   'B': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4, 2.7, 59.6]
               })
    expected = pd.DataFrame({
                   'A': [0.67596768, -0.47871021, -0.19805933,  1.67027365, -0.0537246 ,
                         -0.70323091, -1.39283021,  0.75615365,  1.17312067, -1.44896038],
                   'B': [-1.47939551, -0.79158852,  0.6694926 ,  0.54739669,  0.27878567,
                         -0.09971166,  1.76428598,  0.14448017, -1.67474897,  0.64100356]
               })
    transformer = StandardScaler()
    results = transform_dataframe(
                  df                = df_input,
                  transformer       = transformer,
                  fit               = True,
                  inverse_transform = False
              )
    
    pd.testing.assert_frame_equal(results, expected)


def test_transform_dataframe_when_transformer_is_OneHotEncoder():
    """
    Test the output of transform_dataframe when transformer is OneHotEncoder.
    """
    df_input = pd.DataFrame({
                   'col_1': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                   'col_2': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
               })
    expected = pd.DataFrame({
                   'col_1_a': [1., 1., 1., 1., 0., 0., 0., 0.],
                   'col_1_b': [0., 0., 0., 0., 1., 1., 1., 1.],
                   'col_2_a': [1., 1., 1., 1., 0., 0., 0., 0.],
                   'col_2_b': [0., 0., 0., 0., 1., 1., 1., 1.]
               })
    transformer = OneHotEncoder()
    results = transform_dataframe(
                  df                = df_input,
                  transformer       = transformer,
                  fit               = True,
                  inverse_transform = False
              )
    
    pd.testing.assert_frame_equal(results, expected)


def test_transform_dataframe_when_transformer_is_ColumnTransformer():
    """
    Test the output of transform_dataframe when transformer is ColumnTransformer.
    """
    df_input = pd.DataFrame({
                   'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4],
                   'col_2': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
               })
    expected = pd.DataFrame({
                   'col_1': [-1.76425513, -1.00989936, 0.59254869, 0.45863938,
                             0.1640389 , -0.25107995, 1.79326881, 0.01673866],
                   'col_2_a': [1., 1., 1., 1., 0., 0., 0., 0.],
                   'col_2_b': [0., 0., 0., 0., 1., 1., 1., 1.]
               })
    transformer = ColumnTransformer(
                      [('scale', StandardScaler(), ['col_1']),
                       ('onehot', OneHotEncoder(), ['col_2'])],
                      remainder = 'passthrough',
                      verbose_feature_names_out = False
                  )
    results = transform_dataframe(
                  df                = df_input,
                  transformer       = transformer,
                  fit               = True,
                  inverse_transform = False
              )
    
    pd.testing.assert_frame_equal(results, expected)


def test_transform_dataframe_when_transformer_expands_columns_without_feature_names():
    """
    Test output column naming when transformer expands columns and has no feature names.
    """
    df_input = pd.DataFrame({'y': np.arange(4, dtype=float)})
    transformer = FunctionTransformer(lambda X: np.c_[X, X**2], validate=False)

    expected = pd.DataFrame(
        {'transformed_0': [0.0, 1.0, 2.0, 3.0], 'transformed_1': [0.0, 1.0, 4.0, 9.0]}
    )

    results = transform_dataframe(
        df=df_input, transformer=transformer, fit=True, inverse_transform=False
    )

    pd.testing.assert_frame_equal(results, expected)


def test_transform_dataframe_when_fit_False_and_inverse_transform():
    """
    Test transform_dataframe with fit=False (pre-fitted transformer) and
    inverse_transform=True, verifying round-trip produces original values.
    """
    df_input = pd.DataFrame({
        'A': [1.16, -0.28, 0.07, 2.4, 0.25],
        'B': [7.5, 24.4, 60.3, 57.3, 50.7]
    })
    transformer = StandardScaler()

    # Fit + transform
    df_transformed = transform_dataframe(
        df=df_input, transformer=transformer, fit=True
    )

    # Transform with fit=False (same data, covers fit=False branch)
    df_transformed_no_fit = transform_dataframe(
        df=df_input, transformer=transformer, fit=False
    )
    pd.testing.assert_frame_equal(df_transformed, df_transformed_no_fit)

    # Inverse transform
    df_recovered = transform_dataframe(
        df=df_transformed, transformer=transformer, inverse_transform=True
    )
    pd.testing.assert_frame_equal(df_recovered, df_input)


def test_transform_dataframe_ValueError_when_force_single_column_with_sparse_output():
    """
    Test that transform_dataframe raises ValueError when force_single_column is
    True and transformer expands columns. Also covers toarray() from sparse.
    """
    df_input = pd.DataFrame({
        'col_1': ['a', 'a', 'b', 'b'],
        'col_2': ['x', 'y', 'x', 'y']
    })
    transformer = OneHotEncoder(sparse_output=True)

    err_msg = re.escape(
        "`transformer_y` and `transformer_series` must return a single column. "
        "The transformer generated 4 columns. "
        "Transformers that expand target series into multiple feature "
        "columns are not supported; use `window_features` or pass "
        "those features through `exog` instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        transform_dataframe(
            df=df_input,
            transformer=transformer,
            fit=True,
            inverse_transform=False,
            force_single_column=True
        )


def test_transform_dataframe_when_transformer_returns_1d_array():
    """
    Test transform_dataframe when transformer returns a 1D array, verifying
    the ndim==1 reshape branch produces a single-column DataFrame.
    """
    df_input = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [4.0, 5.0, 6.0]})
    transformer = FunctionTransformer(func=lambda X: np.sum(X, axis=1), validate=False)

    results = transform_dataframe(
        df=df_input, transformer=transformer, fit=True
    )

    expected = pd.DataFrame({'transformed_0': [5.0, 7.0, 9.0]})
    pd.testing.assert_frame_equal(results, expected)


def test_transform_dataframe_when_set_output_pandas():
    """
    Test transform_dataframe when transformer uses set_output('pandas'),
    verifying DataFrame output preserves DatetimeIndex and dtypes.
    """
    idx = pd.date_range(start='2020-01-01', periods=5, freq='D')
    df_input = pd.DataFrame(
        {'A': [1.16, -0.28, 0.07, 2.4, 0.25],
         'B': [7.5, 24.4, 60.3, 57.3, 50.7]},
        index=idx
    )
    transformer_pandas = StandardScaler().set_output(transform='pandas')
    transformer_default = StandardScaler()

    results = transform_dataframe(
        df=df_input, transformer=transformer_pandas, fit=True
    )
    expected = transform_dataframe(
        df=df_input, transformer=transformer_default, fit=True
    )
    pd.testing.assert_frame_equal(results, expected)
    pd.testing.assert_index_equal(results.index, idx)


def test_transform_dataframe_preserves_categorical_dtypes_with_set_output_pandas():
    """
    Test transform_dataframe preserves categorical dtypes when using a
    ColumnTransformer with set_output('pandas') and multiple categorical columns.
    """
    idx = pd.date_range(start='2020-01-01', periods=50, freq='D')
    df_input = pd.DataFrame(
        {
            'exog_1': np.random.default_rng(123).random(50),
            'exog_2': ['a', 'b', 'c', 'd', 'e'] * 10,
            'exog_3': pd.Categorical(['F', 'G', 'H', 'I', 'J'] * 10),
        },
        index=idx,
    )

    pipeline_categorical = make_pipeline(
        OrdinalEncoder(
            dtype=int,
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            encoded_missing_value=-1,
        ),
        FunctionTransformer(
            func=lambda x: x.astype('category'),
            feature_names_out='one-to-one',
        ),
    )
    transformer = make_column_transformer(
        (pipeline_categorical, make_column_selector(dtype_exclude=np.number)),
        remainder='passthrough',
        verbose_feature_names_out=False,
    ).set_output(transform='pandas')

    results = transform_dataframe(
        df=df_input, transformer=transformer, fit=True
    )

    expected = pd.DataFrame(
        {
            'exog_2': pd.Categorical([0, 1, 2, 3, 4] * 10),
            'exog_3': pd.Categorical([0, 1, 2, 3, 4] * 10),
            'exog_1': np.random.default_rng(123).random(50),
        },
        index=idx,
    )
    pd.testing.assert_frame_equal(results, expected)
