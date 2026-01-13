# Unit test transform_numpy
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from skforecast.utils import transform_numpy


def test_transform_numpy_when_transformer_is_None():
    """
    Test the output of transform_numpy when transformer is None.
    """
    input_array = np.array([1.16, -0.28, 0.07, 2.4, 0.25, -0.56, -1.42, 1.26, 1.78, -1.49])
    transformer = None
    results = transform_numpy(
                  array             = input_array,
                  transformer       = transformer,
                  fit               = True,
                  inverse_transform = False
              )
    
    expected = input_array
    
    np.testing.assert_array_almost_equal(results, expected)


def test_transform_numpy_TypeError_when_array_is_not_numpy_ndarray():
    """
    Test TypeError is raised when `array` is not a numpy ndarray.
    """
    array = pd.Series(np.arange(10))

    err_msg = re.escape(f"`array` argument must be a numpy ndarray. Got {type(array)}")
    with pytest.raises(TypeError, match = err_msg):
        transform_numpy(
            array             = array,
            transformer       = StandardScaler(),
            fit               = True,
            inverse_transform = False
        )


def test_transform_numpy_ValueError_when_transformer_is_ColumnTransformer_and_inverse_transform_is_true():
    """
    Test that transform_numpy raise ValueError when transformer is ColumnTransformer
    and argument inverse_transform is True.
    """
    array = np.array([
                [7.5, 'a'],
                [24.4, 'a'],
                [60.3, 'a'],
                [57.3, 'a'],
                [50.7, 'b'],
                [41.4, 'b'],
                [87.2, 'b'],
                [47.4, 'b']
            ], dtype=object)
    
    transformer = ColumnTransformer(
                      [('scale', StandardScaler(), 0),
                       ('onehot', OneHotEncoder(sparse_output=False), 1)],
                      remainder = 'passthrough',
                      verbose_feature_names_out = False
                  )

    err_msg = re.escape(
        "`inverse_transform` is not available when using ColumnTransformers."
    )
    with pytest.raises(ValueError, match = err_msg):
        transform_numpy(
            array             = array,
            transformer       = transformer,
            fit               = True,
            inverse_transform = True
        )


def test_transform_numpy_when_transformer_is_StandardScaler():
    """
    Test the output of transform_numpy when transformer is StandardScaler.
    """
    input_array = np.array([1.16, -0.28, 0.07, 2.4, 0.25, -0.56, -1.42, 1.26, 1.78, -1.49])
    transformer = StandardScaler()
    results = transform_numpy(
                  array             = input_array,
                  transformer       = transformer,
                  fit               = True,
                  inverse_transform = False
              )
    
    expected = np.array([
        0.67596768, -0.47871021, -0.19805933,  1.67027365, -0.0537246,
        -0.70323091, -1.39283021,  0.75615365,  1.17312067, -1.44896038]
    )
    
    np.testing.assert_array_almost_equal(results, expected)


def test_transform_numpy_when_transformer_is_StandardScaler_and_inverse_transform_is_True():
    """
    Test the output of transform_numpy when transformer is StandardScaler and
    inverse_transform is True.
    """
    input_1 = np.array([1.16, -0.28, 0.07, 2.4, 0.25, -0.56, -1.42, 1.26, 1.78, -1.49])
    transformer = StandardScaler()
    transformer.fit(input_1.reshape(-1, 1))

    input_2 = transformer.transform(input_1.reshape(-1, 1)).ravel()
    results = transform_numpy(
                  array             = input_2,
                  transformer       = transformer,
                  fit               = False,
                  inverse_transform = True
              )
    
    expected = input_1
    
    np.testing.assert_array_almost_equal(results, expected)


def test_transform_numpy_when_transformer_is_OneHotEncoder():
    """
    Test the output of transform_numpy when transformer is OneHotEncoder.
    """
    input_array = np.array(['A'] * 5 + ['B'] * 5).reshape(-1, 1)
    transformer = OneHotEncoder(sparse_output=False)
    transformer.fit(input_array)

    results = transform_numpy(
                  array             = input_array,
                  transformer       = transformer,
                  fit               = False,
                  inverse_transform = False
              )

    expected = np.array(
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
               )
    
    np.testing.assert_array_almost_equal(results, expected)


def test_transform_numpy_when_transformer_is_ColumnTransformer():
    """
    Test the output of transform_numpy when transformer is ColumnTransformer.
    """
    array = np.array([
                [7.5, 'a'],
                [24.4, 'a'],
                [60.3, 'a'],
                [57.3, 'a'],
                [50.7, 'b'],
                [41.4, 'b'],
                [87.2, 'b'],
                [47.4, 'b']
            ], dtype=object)
    transformer = ColumnTransformer(
                      [('scale', StandardScaler(), [0]),
                       ('onehot', OneHotEncoder(), [1])],
                      remainder = 'passthrough',
                      verbose_feature_names_out = False
                  )
    results = transform_numpy(
                  array             = array,
                  transformer       = transformer,
                  fit               = True,
                  inverse_transform = False
              )
    
    expected = np.array([
        [-1.76425513, 1., 0.],
        [-1.00989936, 1., 0.],
        [ 0.59254869, 1., 0.],
        [ 0.45863938, 1., 0.],
        [ 0.1640389 , 0., 1.],
        [-0.25107995, 0., 1.],
        [ 1.79326881, 0., 1.],
        [ 0.01673866, 0., 1.]
    ])
    
    np.testing.assert_array_almost_equal(results, expected)


def test_transform_numpy_when_transformer_set_output_is_pandas():
    """
    Test the output of transform_numpy when transformer is StandardScaler and
    set_output is pandas.
    """
    input_array = np.array([1.16, -0.28, 0.07, 2.4, 0.25, -0.56, -1.42, 1.26, 1.78, -1.49])
    transformer = StandardScaler().set_output(transform='pandas')
    results = transform_numpy(
                  array             = input_array,
                  transformer       = transformer,
                  fit               = True,
                  inverse_transform = False
              )
    
    expected = np.array([
        0.67596768, -0.47871021, -0.19805933,  1.67027365, -0.0537246,
        -0.70323091, -1.39283021,  0.75615365,  1.17312067, -1.44896038]
    )
    
    np.testing.assert_array_almost_equal(results, expected)


def test_transform_numpy_inverse_transform_multiple_columns_equivalent_to_column_by_column():
    """
    Test that transform_numpy with inverse_transform=True on a 2D array with 
    multiple columns produces the same result as applying inverse_transform 
    column by column with a for loop.
    """
    
    np.random.seed(123)
    train_data = np.random.rand(100)
    transformer = StandardScaler()
    _ = transform_numpy(
            array             = train_data,
            transformer       = transformer,
            fit               = True,
            inverse_transform = False
        )
    
    # Create 2D array with multiple columns
    n_rows = 48
    n_cols = 250
    input_array = np.random.rand(n_rows, n_cols)
    
    # Method 1: Column by column with for loop
    expected = np.empty_like(input_array, order='F')
    for i in range(n_cols):
        expected[:, i] = transformer.inverse_transform(
            input_array[:, i].reshape(-1, 1)
        ).ravel()
    
    # Method 2: Using transform_numpy
    results = transform_numpy(
                  array             = input_array,
                  transformer       = transformer,
                  fit               = False,
                  inverse_transform = True
              )
    
    np.testing.assert_array_almost_equal(results, expected)


def test_transform_numpy_inverse_transform_preserves_shape():
    """
    Test that transform_numpy with inverse_transform=True preserves the 
    original shape of the input array.
    """
    np.random.seed(456)
    train_data = np.random.rand(100, 1)
    transformer = StandardScaler()
    transformer.fit(train_data)
    
    # Test various shapes
    shapes_to_test = [(10, 1), (10, 3), (48, 100), (24, 250)]
    
    for shape in shapes_to_test:
        input_array = np.random.rand(*shape)
        results = transform_numpy(
                      array             = input_array,
                      transformer       = transformer,
                      fit               = False,
                      inverse_transform = True
                  )
        assert results.shape == shape, f"Shape mismatch: expected {shape}, got {results.shape}"
