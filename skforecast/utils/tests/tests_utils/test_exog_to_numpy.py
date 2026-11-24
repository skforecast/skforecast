# Unit test _exog_to_numpy
# ==============================================================================
import decimal
import warnings
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import _exog_to_numpy

numpy_dtypes = ['bool', 'int8', 'int64', 'uint8', 'float32', 'float64']
nullable_dtypes = [
    'boolean', 'Int8', 'Int64', 'UInt8', 'UInt64', 'Float32', 'Float64'
]
pyarrow_dtypes = [
    'bool[pyarrow]', 'int8[pyarrow]', 'int64[pyarrow]', 'uint64[pyarrow]',
    'float[pyarrow]', 'double[pyarrow]'
]


def _column(dtype, values):
    """
    Column of `values` with the given dtype. 'category' creates a category
    with integer categories.
    """
    if dtype == 'category':
        return pd.Categorical(values)
    if 'bool' in dtype:
        values = [bool(v) for v in values]
    return pd.array(values, dtype=dtype)


@pytest.mark.parametrize(
    'dtype, values, expected',
    [('bool', [True, False, True], [1., 0., 1.]),
     ('boolean', [True, False, True], [1., 0., 1.]),
     ('Int64', [3, 1, 2], [3., 1., 2.]),
     ('UInt8', [3, 1, 2], [3., 1., 2.]),
     ('Float32', [3.5, 1.5, 2.5], [3.5, 1.5, 2.5]),
     ('Float64', [3.5, 1.5, 2.5], [3.5, 1.5, 2.5])],
    ids=lambda v: f'{v}'
)
def test_exog_to_numpy_output_is_float64_when_column_next_to_float(dtype, values, expected):
    """
    Test _exog_to_numpy returns a float64 array, instead of the object array
    of `to_numpy()`, when a bool or nullable column is next to a float column.
    """
    exog = pd.DataFrame({
        'a': [0.5, 1.5, 2.5],
        'x': _column(dtype, values)
    })
    results = _exog_to_numpy(exog)

    expected = np.column_stack(([0.5, 1.5, 2.5], expected))

    assert exog.to_numpy().dtype == object
    assert results.dtype == np.float64
    np.testing.assert_array_equal(results, expected)


@pytest.mark.parametrize(
    'dtype, values, expected',
    [('bool[pyarrow]', [True, False, True], [1., 0., 1.]),
     ('int64[pyarrow]', [3, 1, 2], [3., 1., 2.]),
     ('uint8[pyarrow]', [3, 1, 2], [3., 1., 2.]),
     ('float[pyarrow]', [3.5, 1.5, 2.5], [3.5, 1.5, 2.5]),
     ('double[pyarrow]', [3.5, 1.5, 2.5], [3.5, 1.5, 2.5])],
    ids=lambda v: f'{v}'
)
def test_exog_to_numpy_output_is_float64_when_pyarrow_column_next_to_float(dtype, values, expected):
    """
    Test _exog_to_numpy returns a float64 array, instead of the object array
    of `to_numpy()`, when a pyarrow bool, integer or float column is next to
    a float column.
    """
    pytest.importorskip('pyarrow')
    exog = pd.DataFrame({
        'a': [0.5, 1.5, 2.5],
        'x': pd.array(values, dtype=dtype)
    })
    results = _exog_to_numpy(exog)

    expected = np.column_stack(([0.5, 1.5, 2.5], expected))

    assert exog.to_numpy().dtype == object
    assert results.dtype == np.float64
    np.testing.assert_array_equal(results, expected)


@pytest.mark.parametrize(
    'exog, expected',
    [(pd.DataFrame({'a': [True, False, True], 'x': [3, 1, 2]}),
      [[1., 3.], [0., 1.], [1., 2.]]),
     (pd.DataFrame({
         'a': [True, False, True],
         'x': np.array([3.5, 1.5, 2.5], dtype='float32')
      }),
      [[1., 3.5], [0., 1.5], [1., 2.5]]),
     (pd.DataFrame({
         'a': pd.array([3, 1, 2], dtype='Int64'),
         'x': pd.array([6, 4, 5], dtype='Int64')
      }),
      [[3., 6.], [1., 4.], [2., 5.]]),
     (pd.DataFrame({
         'a': pd.array([True, False, True], dtype='boolean'),
         'x': pd.array([False, False, True], dtype='boolean')
      }),
      [[1., 0.], [0., 0.], [1., 1.]]),
     (pd.DataFrame({
         'a': pd.array([3, 1, 2], dtype='Int8'),
         'x': pd.array([3.5, 1.5, 2.5], dtype='Float64')
      }),
      [[3., 3.5], [1., 1.5], [2., 2.5]]),
     (pd.DataFrame({
         'a': pd.Categorical([30, 10, 20]),
         'x': [True, False, True]
      }),
      [[30., 1.], [10., 0.], [20., 1.]]),
     (pd.DataFrame({
         'a': pd.Categorical([30, 10, 20], categories=[30, 20, 10]),
         'x': pd.array([3, 1, 2], dtype='Int64')
      }),
      [[30., 3.], [10., 1.], [20., 2.]])],
    ids=['bool + int', 'bool + float32', 'Int64 x2', 'boolean x2',
         'Int8 + Float64', 'category + bool', 'category + Int64']
)
def test_exog_to_numpy_output_is_float64_when_no_float_column(exog, expected):
    """
    Test _exog_to_numpy returns a float64 array, instead of the object array
    of `to_numpy()`, when numpy booleans are next to numbers, when all the
    columns have the same nullable dtype and when an integer category is next
    to a bool or nullable column.
    """
    results = _exog_to_numpy(exog)

    assert exog.to_numpy().dtype == object
    assert results.dtype == np.float64
    np.testing.assert_array_equal(results, np.array(expected))


@pytest.mark.parametrize(
    'exog, expected',
    [(pd.DataFrame({
         'a': [0.5, 1.5, 2.5],
         'x': pd.array([3, pd.NA, 2], dtype='Int64')
      }),
      [[0.5, 3.], [1.5, np.nan], [2.5, 2.]]),
     (pd.DataFrame({
         'a': [0.5, 1.5, 2.5],
         'x': pd.array([3.5, pd.NA, 2.5], dtype='Float64')
      }),
      [[0.5, 3.5], [1.5, np.nan], [2.5, 2.5]]),
     (pd.DataFrame({
         'a': [0.5, 1.5, 2.5],
         'x': pd.array([True, pd.NA, False], dtype='boolean')
      }),
      [[0.5, 1.], [1.5, np.nan], [2.5, 0.]]),
     (pd.DataFrame({'x': pd.array([True, pd.NA, False], dtype='boolean')}),
      [[1.], [np.nan], [0.]]),
     (pd.DataFrame({'x': pd.array([3, pd.NA, 2], dtype='Int64')}),
      [[3.], [np.nan], [2.]]),
     (pd.DataFrame({
         'a': pd.array([3, pd.NA, 2], dtype='Int64'),
         'x': pd.array([6, 4, 5], dtype='Int64')
      }),
      [[3., 6.], [np.nan, 4.], [2., 5.]]),
     (pd.DataFrame({
         'a': pd.Categorical([30, np.nan, 20]),
         'x': [True, False, True]
      }),
      [[30., 1.], [np.nan, 0.], [20., 1.]]),
     (pd.DataFrame({
         'a': [1, 2, 3],
         'x': pd.Categorical([30, np.nan, 20])
      }),
      [[1., 30.], [2., np.nan], [3., 20.]])],
    ids=['float + Int64', 'float + Float64', 'float + boolean', 'boolean',
         'Int64', 'Int64 x2', 'category + bool', 'int + category']
)
def test_exog_to_numpy_missing_values_are_nan(exog, expected):
    """
    Test _exog_to_numpy returns a float64 array with NaN in the place of the
    missing values of nullable and category columns. With an integer column
    next to an integer category, `exog.to_numpy()` is an integer array where
    the missing value is the minimum of int64.
    """
    results = _exog_to_numpy(exog)

    assert results.dtype == np.float64
    np.testing.assert_array_equal(results, np.array(expected))


@pytest.mark.parametrize(
    'dtype',
    ['int64[pyarrow]', 'double[pyarrow]', 'bool[pyarrow]'],
    ids=lambda dtype: f'dtype: {dtype}'
)
def test_exog_to_numpy_missing_values_of_pyarrow_columns_are_nan(dtype):
    """
    Test _exog_to_numpy returns a float64 array with NaN in the place of the
    missing values of pyarrow columns.
    """
    pytest.importorskip('pyarrow')
    exog = pd.DataFrame({
        'a': [0.5, 1.5, 2.5],
        'x': pd.array([1, None, 0], dtype=dtype)
    })
    results = _exog_to_numpy(exog)

    expected = np.array([[0.5, 1.], [1.5, np.nan], [2.5, 0.]])

    assert results.dtype == np.float64
    np.testing.assert_array_equal(results, expected)


@pytest.mark.parametrize(
    'exog, expected_dtype',
    [(pd.DataFrame({'a': [0.5, 1.5], 'x': [2.5, 3.5]}), 'float64'),
     (pd.DataFrame({'a': [1, 2], 'x': [3, 4]}), 'int64'),
     (pd.DataFrame({'a': [1, 2], 'x': [2.5, 3.5]}), 'float64'),
     (pd.DataFrame({'a': [True, False], 'x': [False, False]}), 'bool'),
     (pd.DataFrame({'a': np.array([0.5, 1.5], dtype='float32')}), 'float32'),
     (pd.DataFrame({
         'a': np.array([1, 2], dtype='int8'),
         'x': np.array([3, 4], dtype='uint8')
      }), 'int16'),
     (pd.DataFrame({'a': ['a', 'b'], 'x': [True, False]}), 'object'),
     (pd.DataFrame({'a': ['7', '8'], 'x': [0.5, 1.5]}), 'object'),
     (pd.DataFrame({
         'a': pd.Categorical(['a', 'b']),
         'x': [True, False]
      }), 'object'),
     (pd.DataFrame({
         'a': pd.Categorical([True, False]),
         'x': [0.5, 1.5]
      }), 'object'),
     (pd.DataFrame({
         'a': pd.Categorical([0.5, 1.5]),
         'x': [True, False]
      }), 'object'),
     (pd.DataFrame({
         'a': pd.date_range('2020-01-01', periods=2),
         'x': [True, False]
      }), 'object'),
     (pd.DataFrame({
         'a': pd.to_timedelta([1, 2], unit='D'),
         'x': [True, False]
      }), 'object'),
     (pd.DataFrame({'a': [1 + 1j, 2 + 2j], 'x': [True, False]}), 'object'),
     (pd.DataFrame({'a': [1 + 1j, 2 + 2j], 'x': [0.5, 1.5]}), 'complex128'),
     (pd.DataFrame({
         'a': pd.Series([1, 2.5], dtype=object),
         'x': [True, False]
      }), 'object'),
     (pd.DataFrame({
         'a': pd.Series(
                  [decimal.Decimal('1.1'), decimal.Decimal('2.2')], dtype=object
              ),
         'x': pd.array([1, 2], dtype='Int64')
      }), 'object'),
     (pd.DataFrame({
         'a': pd.array(['a', 'b'], dtype='string'),
         'x': pd.array([1, 2], dtype='Int64')
      }), 'object')],
    ids=['float', 'int', 'int + float', 'bool', 'float32', 'int8 + uint8',
         'str + bool', 'numeric str + float', 'category str + bool',
         'category bool + float', 'category float + bool', 'datetime + bool',
         'timedelta + bool', 'complex + bool', 'complex + float',
         'object + bool', 'Decimal + Int64', 'string + Int64']
)
def test_exog_to_numpy_output_same_as_to_numpy_when_not_converted(exog, expected_dtype):
    """
    Test _exog_to_numpy returns the same array as `exog.to_numpy()` when the
    numpy dtypes of the columns can be combined by numpy, or when any column
    is not boolean, integer, float or a category with integer categories
    (text, other categories, dates, timedeltas, complex, object).
    """
    results = _exog_to_numpy(exog)
    expected = exog.to_numpy()

    assert results.dtype == np.dtype(expected_dtype)
    assert results.dtype == expected.dtype
    pd.testing.assert_frame_equal(pd.DataFrame(results), pd.DataFrame(expected))


@pytest.mark.parametrize(
    'exog, expected',
    [(pd.DataFrame({
         'a': [0.5, 1.5, 2.5],
         'x': pd.Categorical([30, 10, 20], categories=[30, 20, 10])
      }),
      [[0.5, 30.], [1.5, 10.], [2.5, 20.]]),
     (pd.DataFrame({
         'x': pd.Categorical([30, 10, 20], categories=[30, 20, 10])
      }),
      [[30.], [10.], [20.]]),
     (pd.DataFrame({
         'a': [3, 1, 2],
         'x': pd.Categorical(pd.array([30, 10, 20], dtype='Int32'))
      }),
      [[3., 30.], [1., 10.], [2., 20.]])],
    ids=['float + category', 'category', 'int + category of Int32']
)
def test_exog_to_numpy_values_of_integer_category_are_its_categories(exog, expected):
    """
    Test _exog_to_numpy returns the categories of an integer category, not
    its codes, as float64. They are the values `to_numpy()` returns.
    """
    results = _exog_to_numpy(exog)

    assert results.dtype == np.float64
    np.testing.assert_array_equal(results, np.array(expected))
    np.testing.assert_array_equal(results, exog.to_numpy().astype(float))


def test_exog_to_numpy_output_same_as_to_numpy_when_sparse_column():
    """
    Test _exog_to_numpy returns the same array as `exog.to_numpy()` when a
    column is sparse (float64 in pandas 2 and object in pandas 3).
    """
    exog = pd.DataFrame({
        'a': pd.arrays.SparseArray([0.5, 0., 1.5]),
        'x': [True, False, True]
    })
    results = _exog_to_numpy(exog)
    expected = exog.to_numpy()

    assert results.dtype == expected.dtype
    pd.testing.assert_frame_equal(pd.DataFrame(results), pd.DataFrame(expected))


def test_exog_to_numpy_output_same_as_to_numpy_when_pyarrow_column_not_numeric():
    """
    Test _exog_to_numpy returns the same array as `exog.to_numpy()` when a
    pyarrow column is not boolean, integer or float (decimal, timestamp,
    string, dictionary).
    """
    pa = pytest.importorskip('pyarrow')
    columns = {
        'decimal': pd.array(
            [decimal.Decimal('1.10'), decimal.Decimal('2.20')],
            dtype=pd.ArrowDtype(pa.decimal128(10, 2))
        ),
        'timestamp': pd.array(
            pd.date_range('2020-01-01', periods=2),
            dtype=pd.ArrowDtype(pa.timestamp('ns'))
        ),
        'string': pd.array(['a', 'b'], dtype=pd.ArrowDtype(pa.string())),
        'dictionary': pd.array(
            [1, 2], dtype=pd.ArrowDtype(pa.dictionary(pa.int8(), pa.int64()))
        )
    }
    for column in columns.values():
        exog = pd.DataFrame({'a': [0.5, 1.5], 'x': column})
        results = _exog_to_numpy(exog)
        expected = exog.to_numpy()

        assert results.dtype == object
        pd.testing.assert_frame_equal(
            pd.DataFrame(results), pd.DataFrame(expected)
        )


@pytest.mark.parametrize(
    'dtypes',
    [numpy_dtypes + nullable_dtypes + ['category'], pyarrow_dtypes],
    ids=['numpy, nullable and category', 'pyarrow']
)
def test_exog_to_numpy_no_object_and_no_warnings_with_all_pairs_of_dtypes(dtypes):
    """
    Test _exog_to_numpy returns a numeric array with the same values, without
    warnings, for all the pairs of dtypes it converts. The array is float64
    unless both dtypes are numpy dtypes that numpy combines.
    """
    if dtypes is pyarrow_dtypes:
        pytest.importorskip('pyarrow')
        dtypes = numpy_dtypes + nullable_dtypes + ['category'] + pyarrow_dtypes
        pairs = [
            (dtype_1, dtype_2) for dtype_1 in dtypes for dtype_2 in dtypes
            if dtype_1 in pyarrow_dtypes or dtype_2 in pyarrow_dtypes
        ]
    else:
        pairs = [(dtype_1, dtype_2) for dtype_1 in dtypes for dtype_2 in dtypes]

    expected = np.array([[1., 1.], [0., 0.], [1., 1.]])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for dtype_1, dtype_2 in pairs:
            exog = pd.DataFrame({
                'a': _column(dtype_1, [1, 0, 1]),
                'x': _column(dtype_2, [1, 0, 1])
            })
            results = _exog_to_numpy(exog)

            both_numpy = dtype_1 in numpy_dtypes and dtype_2 in numpy_dtypes
            bool_and_number = (dtype_1 == 'bool') != (dtype_2 == 'bool')
            if both_numpy and not bool_and_number:
                assert results.dtype == exog.to_numpy().dtype, (dtype_1, dtype_2)
            else:
                assert results.dtype == np.float64, (dtype_1, dtype_2)
            np.testing.assert_array_equal(results.astype(float), expected)


def test_exog_to_numpy_same_output_when_dtypes_are_passed():
    """
    Test _exog_to_numpy returns the same array when the dtypes of the columns
    are passed and when they are extracted from `exog`.
    """
    exog_converted = pd.DataFrame({
        'a': [0.5, 1.5, 2.5],
        'x': pd.array([3, pd.NA, 2], dtype='Int64')
    })
    exog_not_converted = pd.DataFrame({
        'a': [0.5, 1.5, 2.5],
        'x': ['a', 'b', 'c']
    })

    for exog in [exog_converted, exog_not_converted]:
        results = _exog_to_numpy(exog, dtypes=exog.dtypes.tolist())
        expected = _exog_to_numpy(exog)

        assert results.dtype == expected.dtype
        pd.testing.assert_frame_equal(
            pd.DataFrame(results), pd.DataFrame(expected)
        )


def test_exog_to_numpy_output_when_duplicated_column_names():
    """
    Test _exog_to_numpy uses the dtype of every column, by position, when
    `exog` has duplicated column names.
    """
    exog = pd.DataFrame(
        {'a': [0.5, 1.5, 2.5], 'x': [True, False, True]}
    )
    exog.columns = ['a', 'a']
    results = _exog_to_numpy(exog)

    expected = np.array([[0.5, 1.], [1.5, 0.], [2.5, 1.]])

    assert results.dtype == np.float64
    np.testing.assert_array_equal(results, expected)


def test_exog_to_numpy_output_when_exog_has_no_rows():
    """
    Test _exog_to_numpy takes the same decision, which only depends on the
    dtypes, when `exog` has no rows.
    """
    exog = pd.DataFrame({
        'a': [0.5, 1.5, 2.5],
        'x': pd.array([3, pd.NA, 2], dtype='Int64')
    })
    results = _exog_to_numpy(exog.iloc[:0])

    assert results.dtype == np.float64
    assert results.shape == (0, 2)


def test_exog_to_numpy_integers_above_2_53_are_rounded():
    """
    Test _exog_to_numpy rounds the integers that float64 cannot represent
    when they are converted.
    """
    exog = pd.DataFrame({
        'a': [0.5, 1.5],
        'x': pd.array([2**53 + 1, 5], dtype='Int64')
    })
    results = _exog_to_numpy(exog)

    expected = np.array([[0.5, 9007199254740992.], [1.5, 5.]])

    assert results.dtype == np.float64
    np.testing.assert_array_equal(results, expected)
