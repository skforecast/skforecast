# Unit test transform_numpy
# ==============================================================================
import re
import warnings
from copy import deepcopy
import pytest
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from skforecast.utils import transform_numpy


def test_transform_numpy_when_transformer_is_None():
    """
    Test the output of transform_numpy when transformer is None. The input
    array is returned as is.
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
    assert results is input_array


@pytest.mark.parametrize(
    "array, fit",
    [(pd.Series(np.arange(10)), True),
     (pd.Series(np.arange(10, dtype=float)), False),
     (list(np.arange(10, dtype=float)), False)],
    ids=["Series int64, fit=True", "Series float64, fit=False", "list, fit=False"]
)
def test_transform_numpy_TypeError_when_array_is_not_numpy_ndarray(array, fit):
    """
    Test TypeError is raised when `array` is not a numpy ndarray, also with a
    fitted StandardScaler and `fit=False` (fast path).
    """
    transformer = StandardScaler().fit(np.arange(10, dtype=float).reshape(-1, 1))

    err_msg = re.escape(f"`array` argument must be a numpy ndarray. Got {type(array)}")
    with pytest.raises(TypeError, match = err_msg):
        transform_numpy(
            array             = array,
            transformer       = transformer,
            fit               = fit,
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


def test_transform_numpy_ValueError_when_force_single_column_with_sparse_output():
    """
    Test that transform_numpy raises ValueError when force_single_column is True
    and transformer expands columns. Also covers toarray() conversion from sparse
    matrix to dense array.
    """
    input_array = np.array(['A'] * 5 + ['B'] * 5).reshape(-1, 1)
    transformer = OneHotEncoder(sparse_output=True)

    err_msg = re.escape(
        "`transformer_y` and `transformer_series` must return a single column. "
        "The transformer generated 2 columns. "
        "Transformers that expand target series into multiple feature "
        "columns are not supported; use `window_features` or pass "
        "those features through `exog` instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        transform_numpy(
            array=input_array,
            transformer=transformer,
            fit=True,
            inverse_transform=False,
            force_single_column=True
        )


def _multiply_in_place(X):
    """
    Multiply the input by 2 in place and return it.
    """
    X *= 2.
    return X


def _divide_in_place(X):
    """
    Divide the input by 2 in place and return it.
    """
    X /= 2.
    return X


@pytest.mark.parametrize(
    "transformer",
    [StandardScaler(copy=False),
     make_pipeline(StandardScaler(copy=False)),
     FunctionTransformer(func=_multiply_in_place, inverse_func=_divide_in_place),
     FunctionTransformer()],
    ids=['StandardScaler(copy=False)', 'Pipeline with StandardScaler(copy=False)',
         'in-place FunctionTransformer', 'identity FunctionTransformer']
)
def test_transform_numpy_does_not_modify_input_array(transformer):
    """
    Test transform_numpy does not modify the input array, with fit, transform
    and inverse_transform, when the transformer writes into its input or
    returns it as is. The output does not share memory with the input. Also
    with a read-only array and, in the inverse, a 2D array (the fast path
    covers `StandardScaler(copy=False)` with `fit=False`).
    """
    input_array_1d = np.arange(10, dtype=float)
    input_array_read_only = np.arange(10, dtype=float)
    input_array_read_only.flags.writeable = False
    input_array_2d = np.arange(12, dtype=float).reshape(4, 3)

    cases = [
        (input_array_1d, True, False),
        (input_array_1d, False, False),
        (input_array_1d, False, True),
        (input_array_read_only, False, False),
        (input_array_read_only, False, True),
        (input_array_2d, False, True),
    ]
    for input_array, fit, inverse_transform in cases:
        input_array_copy = input_array.copy()
        results = transform_numpy(
                      array             = input_array,
                      transformer       = transformer,
                      fit               = fit,
                      inverse_transform = inverse_transform
                  )
        np.testing.assert_array_equal(input_array, input_array_copy)
        assert not np.shares_memory(results, input_array)


# Fast path: fitted StandardScaler with `fit=False`
# ==============================================================================
y_fit = np.array([1.16, -0.28, 0.07, 2.4, 0.25, -0.56, -1.42, 1.26, 1.78, -1.49])


class StandardScalerSubclass(StandardScaler):
    """
    Subclass of StandardScaler. The fast path applies to the exact class only.
    """


class ScikitLearnMethodCalled(Exception):
    """
    Raised by the patched methods of StandardScaler: the fast path was not taken.
    """


def _raise_scikit_learn_method_called(self, *args, **kwargs):
    """
    Replacement for the `transform` and `inverse_transform` methods of
    StandardScaler.
    """
    raise ScikitLearnMethodCalled("A method of StandardScaler was called.")


def _scikit_learn_output(transformer, array, inverse_transform):
    """
    Output of the scikit-learn method applied to the values of `array` as a
    single column, with the shape of `array`.
    """
    X = array.reshape(-1, 1).copy()
    method = transformer.inverse_transform if inverse_transform else transformer.transform
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="X does not have valid feature names", category=UserWarning
        )
        output = method(X)

    return np.asarray(output).reshape(array.shape)


def _assert_identical_arrays(results, expected):
    """
    Assert two numpy ndarrays have the same type, dtype, shape and bytes. Unlike
    `np.testing.assert_array_equal`, it distinguishes `-0.0` from `0.0`.
    """
    assert type(results) is np.ndarray
    assert results.dtype == expected.dtype
    assert results.shape == expected.shape
    assert results.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "array, inverse_transform",
    [(np.array([1., np.inf, 3.]), False),
     (np.array([1., -np.inf, np.nan]), True),
     (np.array([[1., 2.], [np.nan, np.inf]]), True)],
    ids=["1D +inf transform", "1D -inf and NaN inverse", "2D inf and NaN inverse"]
)
def test_transform_numpy_ValueError_when_StandardScaler_and_array_has_infinite_values(
    array, inverse_transform
):
    """
    Test the fast path leaves arrays with infinite values to scikit-learn, which
    raises its ValueError (the message differs between `transform` and
    `inverse_transform`). With `assume_finite=True`, scikit-learn does not raise
    and the output is the same.
    """
    transformer = StandardScaler().fit(y_fit.reshape(-1, 1))

    err_msg = re.escape(
        f"Input {'' if inverse_transform else 'X '}contains infinity or a value "
        f"too large for dtype('float64')."
    )
    with pytest.raises(ValueError, match = err_msg):
        transform_numpy(
            array             = array,
            transformer       = transformer,
            fit               = False,
            inverse_transform = inverse_transform
        )

    with sklearn.config_context(assume_finite=True):
        results = transform_numpy(
                      array             = array,
                      transformer       = transformer,
                      fit               = False,
                      inverse_transform = inverse_transform
                  )
        expected = _scikit_learn_output(transformer, array, inverse_transform)

    _assert_identical_arrays(results, expected)


@pytest.mark.parametrize(
    "transformer",
    [StandardScaler(), StandardScaler(with_mean=False, with_std=False)],
    ids=["StandardScaler()", "StandardScaler(with_mean=False, with_std=False)"]
)
def test_transform_numpy_NotFittedError_when_StandardScaler_is_not_fitted(transformer):
    """
    Test NotFittedError is raised by scikit-learn when the StandardScaler is not
    fitted and `fit=False`, in both directions.
    """
    err_msg = re.escape("This StandardScaler instance is not fitted yet.")
    for inverse_transform in [False, True]:
        with pytest.raises(NotFittedError, match = err_msg):
            transform_numpy(
                array             = np.array([1., 2., 3.]),
                transformer       = transformer,
                fit               = False,
                inverse_transform = inverse_transform
            )


@pytest.mark.parametrize(
    "array, inverse_transform",
    [(np.array([]), False),
     (np.array([]), True),
     (np.empty((0, 3)), True)],
    ids=["1D transform", "1D inverse", "2D inverse"]
)
def test_transform_numpy_ValueError_when_StandardScaler_and_array_is_empty(
    array, inverse_transform
):
    """
    Test scikit-learn still raises a ValueError with an empty array and a fitted
    StandardScaler (`set_out_sample_residuals` relies on it).
    """
    transformer = StandardScaler().fit(y_fit.reshape(-1, 1))

    err_msg = re.escape(
        "Found array with 0 sample(s) (shape=(0, 1)) while a minimum of 1 is required"
    )
    with pytest.raises(ValueError, match = err_msg):
        transform_numpy(
            array             = array,
            transformer       = transformer,
            fit               = False,
            inverse_transform = inverse_transform
        )


def test_transform_numpy_output_when_StandardScaler_is_fitted_and_fit_is_False():
    """
    Test the output of transform_numpy, in both directions, with a fitted
    StandardScaler and `fit=False` (fast path). Hardcoded values, with NaN and
    `-0.0` in the input.
    """
    transformer = StandardScaler().fit(y_fit.reshape(-1, 1))

    results_transform = transform_numpy(
                            array             = np.array([0.5, -1.5, 3.25, np.nan, -0.0]),
                            transformer       = transformer,
                            fit               = False,
                            inverse_transform = False
                        )
    results_inverse = transform_numpy(
                          array             = np.array([[0.5, -1.5, 3.25],
                                                        [np.nan, -0.0, 1.0]]),
                          transformer       = transformer,
                          fit               = False,
                          inverse_transform = True
                      )

    expected_transform = np.array(
        [0.14674031563113063, -1.4569789808839584, 2.351854348339378, np.nan,
         -0.2541895084976416]
    )
    expected_inverse = np.array(
        [[0.9405505192043384, -1.553651557613015, 4.370078374828199],
         [np.nan, 0.317, 1.5641010384086766]]
    )

    np.testing.assert_array_equal(results_transform, expected_transform, strict=True)
    np.testing.assert_array_equal(results_inverse, expected_inverse, strict=True)


@pytest.mark.parametrize(
    "transformer, X_fit",
    [(StandardScaler(), y_fit.reshape(-1, 1)),
     (StandardScaler(), pd.DataFrame({"y": y_fit})),
     (StandardScaler(copy=False), y_fit.reshape(-1, 1)),
     (StandardScaler().set_output(transform="pandas"), y_fit.reshape(-1, 1)),
     (StandardScaler(with_mean=False), y_fit.reshape(-1, 1)),
     (StandardScaler(with_std=False), y_fit.reshape(-1, 1)),
     (StandardScaler(with_mean=False, with_std=False), y_fit.reshape(-1, 1)),
     (StandardScaler(), np.full((10, 1), 3.5))],
    ids=["fitted with array", "fitted with DataFrame (feature names)",
         "copy=False", "set_output pandas", "with_mean=False", "with_std=False",
         "with_mean=False, with_std=False", "constant series (scale_ = 1)"]
)
def test_transform_numpy_fast_path_output_equal_to_scikit_learn(
    transformer, X_fit, monkeypatch
):
    """
    Test the fast path of a fitted StandardScaler gives the same bytes as the
    scikit-learn method, without calling it (the methods are patched to raise
    once the expected values are computed). 1D arrays in both directions and 2D
    arrays in the inverse, including NaN, `-0.0`, subnormal and large values,
    and arrays that are not C-contiguous or are read-only. The input array and
    the fitted attributes of the transformer are not modified.
    """
    transformer.fit(X_fit)
    mean_before = deepcopy(transformer.mean_)
    scale_before = deepcopy(transformer.scale_)

    rng = np.random.default_rng(123)
    matrix = rng.normal(loc=0.5, scale=2., size=(24, 6))
    cube = rng.normal(loc=0.5, scale=2., size=(24, 3, 6))
    read_only = matrix.copy()
    read_only.flags.writeable = False
    with_nan = matrix[:, 0].copy()
    with_nan[[0, 5, 23]] = np.nan

    arrays_1d = [
        matrix[:, 0].copy(),
        with_nan,
        np.full(5, np.nan),
        np.array([0., -0., 5e-324, -5e-324, 1e-300, 1e300, -1e300]),
        matrix[:, 1],
        matrix[::-1, 2],
        read_only[:, 3],
    ]
    arrays_2d = [
        matrix,
        matrix[:, :3],
        matrix[:, [0]],
        matrix[[0], :],
        np.asfortranarray(matrix),
        cube[:, 1, :],
        read_only,
    ]
    cases = (
        [(array, False) for array in arrays_1d]
        + [(array, True) for array in arrays_1d]
        + [(array, True) for array in arrays_2d]
    )
    expected = [
        _scikit_learn_output(transformer, array, inverse_transform)
        for array, inverse_transform in cases
    ]

    monkeypatch.setattr(StandardScaler, "transform", _raise_scikit_learn_method_called)
    monkeypatch.setattr(
        StandardScaler, "inverse_transform", _raise_scikit_learn_method_called
    )
    for (array, inverse_transform), expected_array in zip(cases, expected):
        array_copy = array.copy()
        results = transform_numpy(
                      array             = array,
                      transformer       = transformer,
                      fit               = False,
                      inverse_transform = inverse_transform
                  )
        _assert_identical_arrays(results, expected_array)
        assert array.tobytes() == array_copy.tobytes()
        assert not np.shares_memory(results, array)

    for before, after in [(mean_before, transformer.mean_),
                          (scale_before, transformer.scale_)]:
        if before is None:
            assert after is None
        else:
            assert after.tobytes() == before.tobytes()


@pytest.mark.parametrize(
    "transformer, X_fit, array, fit, inverse_transform",
    [(StandardScalerSubclass(), y_fit.reshape(-1, 1), y_fit, False, False),
     (make_pipeline(StandardScaler()), y_fit.reshape(-1, 1), y_fit, False, True),
     (StandardScaler(), y_fit.reshape(-1, 1), y_fit.astype(np.float32), False, False),
     (StandardScaler(), y_fit.reshape(-1, 1), y_fit.astype(np.float32), False, True),
     (StandardScaler(), y_fit.reshape(-1, 1), np.arange(10), False, False),
     (StandardScaler(), y_fit.reshape(-1, 1), np.ma.masked_array(y_fit), False, False),
     (StandardScaler(), y_fit.reshape(-1, 1), np.asmatrix(y_fit.reshape(2, 5)), False, True),
     (StandardScaler(), y_fit.reshape(-1, 1), np.array([]), False, False),
     (StandardScaler(), y_fit.reshape(-1, 1), np.array([]), False, True),
     (StandardScaler(), y_fit.reshape(-1, 1), y_fit.reshape(-1, 1), False, False),
     (StandardScaler(), y_fit.reshape(-1, 1), np.array(1.5), False, False),
     (StandardScaler(), y_fit.reshape(-1, 1), np.ones((2, 3, 4)), False, True),
     (StandardScaler(), np.column_stack([y_fit, y_fit]), y_fit, False, True),
     (StandardScaler(), y_fit.reshape(-1, 1), np.array([1., np.inf]), False, False),
     (StandardScaler(), y_fit.reshape(-1, 1), y_fit, True, False),
     (StandardScaler(), None, y_fit, False, False),
     (StandardScaler(with_mean=False, with_std=False).fit(y_fit.reshape(-1, 1))
      .set_params(with_mean=True), None, y_fit, False, False)],
    ids=["subclass", "Pipeline", "float32 transform", "float32 inverse", "int64",
         "MaskedArray transform", "matrix inverse", "empty 1D transform", "empty 1D inverse", "2D transform", "0D transform",
         "3D inverse", "fitted with 2 columns", "inf", "fit=True", "not fitted",
         "with_mean=True but mean_ is None"]
)
def test_transform_numpy_fast_path_not_taken(
    transformer, X_fit, array, fit, inverse_transform, monkeypatch
):
    """
    Test the cases outside the fast path call the scikit-learn method (patched
    to raise a sentinel error), so their result and errors are the same as
    without the fast path.
    """
    if X_fit is not None:
        transformer.fit(X_fit)

    monkeypatch.setattr(StandardScaler, "transform", _raise_scikit_learn_method_called)
    monkeypatch.setattr(
        StandardScaler, "inverse_transform", _raise_scikit_learn_method_called
    )
    with pytest.raises(ScikitLearnMethodCalled):
        transform_numpy(
            array             = array,
            transformer       = transformer,
            fit               = fit,
            inverse_transform = inverse_transform
        )


@pytest.mark.parametrize(
    "dtype, expected_dtype",
    [(np.float32, np.float32), (np.int64, np.float64)],
    ids=["float32", "int64"]
)
def test_transform_numpy_output_dtype_when_StandardScaler_and_array_is_not_float64(
    dtype, expected_dtype
):
    """
    Test the output of scikit-learn is kept for float32 and int64 arrays, which
    do not follow the fast path: float32 stays float32 and int64 becomes float64.
    """
    transformer = StandardScaler().fit(y_fit.reshape(-1, 1))
    array = np.arange(10).astype(dtype)

    for inverse_transform in [False, True]:
        results = transform_numpy(
                      array             = array,
                      transformer       = transformer,
                      fit               = False,
                      inverse_transform = inverse_transform
                  )
        expected = _scikit_learn_output(transformer, array, inverse_transform)
        assert results.dtype == expected_dtype
        _assert_identical_arrays(results, expected)


def test_transform_numpy_fast_path_no_warnings_with_large_finite_values():
    """
    Test the fast path emits no warning with finite values close to the largest
    float64 when the result does not overflow (scikit-learn emits none either).
    The sum of the values overflows, so a check of infinite values based on the
    sum would emit a RuntimeWarning.
    """
    transformer = StandardScaler().fit(y_fit.reshape(-1, 1))
    array = np.array([1e308, 1e308, 1.4e308])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for inverse_transform in [False, True]:
            results = transform_numpy(
                          array             = array,
                          transformer       = transformer,
                          fit               = False,
                          inverse_transform = inverse_transform
                      )
            assert np.isfinite(results).all()
