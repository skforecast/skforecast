# Unit test transform_series
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from skforecast.utils import transform_series


def test_transform_series_TypeError_when_series_is_not_pandas_series():
    """
    Test TypeError is raised when series is not a pandas series.
    """
    series = np.arange(10)

    err_msg = re.escape(f"`series` argument must be a pandas Series. Got {type(series)}")
    with pytest.raises(TypeError, match = err_msg):
        transform_series(
            series            = series,
            transformer       = None,
            fit               = True,
            inverse_transform = False
        )


def test_transform_series_when_transformer_is_None():
    """
    Test the output of transform_series when transformer is None.
    """
    input_series = pd.Series([1.16, -0.28, 0.07, 2.4, 0.25, -0.56, -1.42, 1.26, 1.78, -1.49])
    expected = input_series
    transformer = None
    results = transform_series(
                  series = input_series,
                  transformer = transformer,
                  fit = True,
                  inverse_transform = False
              )
    
    pd.testing.assert_series_equal(results, expected)


def test_transform_series_when_transformer_is_StandardScaler():
    """
    Test the output of transform_series when transformer is StandardScaler.
    """
    input_series = pd.Series([1.16, -0.28, 0.07, 2.4, 0.25, -0.56, -1.42, 1.26, 1.78, -1.49])
    expected = pd.Series([0.67596768, -0.47871021, -0.19805933,  1.67027365, -0.0537246,
                         -0.70323091, -1.39283021,  0.75615365,  1.17312067, -1.44896038],
                         name = 'no_name')
    transformer = StandardScaler()
    results = transform_series(
                  series = input_series,
                  transformer = transformer,
                  fit = True,
                  inverse_transform = False
              )
    
    pd.testing.assert_series_equal(results, expected)


def test_transform_series_when_transformer_is_StandardScaler_and_inverse_transform_is_True():
    """
    Test the output of transform_series when transformer is StandardScaler and
    inverse_transform is True.
    """
    input_1 = pd.Series([1.16, -0.28, 0.07, 2.4, 0.25, -0.56, -1.42, 1.26, 1.78, -1.49],
                      name = 'y')
    transformer = StandardScaler()
    transformer.fit(input_1.to_numpy().reshape(-1, 1))
    input_2 = transformer.transform(input_1.to_numpy().reshape(-1, 1))
    input_2 = pd.Series(input_2.ravel(), name= 'y')
    expected = input_1
    results = transform_series(
                  series = input_2,
                  transformer = transformer,
                  fit = False,
                  inverse_transform = True
              )
    
    pd.testing.assert_series_equal(results, expected)


def test_transform_series_when_transformer_is_OneHotEncoder():
    """
    Test the output of transform_series when transformer is OneHotEncoder.
    """
    input_series = pd.Series(["A"] * 5 + ["B"] * 5, name='exog')
    transformer = OneHotEncoder(sparse_output=False)
    transformer.fit(input_series.to_frame())
    results = transform_series(
                  series = input_series,
                  transformer = transformer,
                  fit = False,
                  inverse_transform = False
              )

    expected = pd.DataFrame(
                data = np.array(
                            [[1., 0.],
                            [1., 0.],
                            [1., 0.],
                            [1., 0.],
                            [1., 0.],
                            [0., 1.],
                            [0., 1.],
                            [0., 1.],
                            [0., 1.],
                            [0., 1.]]
                       ),
                columns = ['exog_A', 'exog_B']
    )
    
    pd.testing.assert_frame_equal(results, expected)


def test_transform_series_when_applied_to_serie_with_different_name_than_the_one_used_to_fit():
    """
    Test the transform series works when the serie to transform has different name
    than the serie used for training.
    """
    training_series = pd.Series([1.16, -0.28, 0.07, 2.4, 0.25, -0.56, -1.42, 1.26, 1.78, -1.49],
                                name = 'y')
    input_series = pd.Series([1.16, -0.28, 0.07, 2.4, 0.25, -0.56, -1.42, 1.26, 1.78, -1.49],
                             name = 'pred')
    expected = pd.Series([0.67596768, -0.47871021, -0.19805933,  1.67027365, -0.0537246,
                         -0.70323091, -1.39283021,  0.75615365,  1.17312067, -1.44896038],
                         name = 'pred')
    transformer = StandardScaler()
    transformer.fit(training_series.to_frame())
    results = transform_series(
                  series = input_series,
                  transformer = transformer,
                  fit = False,
                  inverse_transform = False
              )
    
    pd.testing.assert_series_equal(results, expected)