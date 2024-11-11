# Unit test transform_dataframe
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
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
