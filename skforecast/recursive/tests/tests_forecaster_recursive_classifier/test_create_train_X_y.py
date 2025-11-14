# Unit test _create_train_X_y ForecasterRecursiveClassifier
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from skforecast.exceptions import MissingValuesWarning
from skforecast.preprocessing import RollingFeaturesClassification
from skforecast.recursive import ForecasterRecursiveClassifier


def test_create_train_X_y_ValueError_when_len_y_less_than_window_size():
    """
    Test ValueError is raised when len(y) <= window_size.
    """
    y = pd.Series(np.arange(5))

    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)
    err_msg = re.escape(
        "Length of `y` must be greater than the maximum window size "
        "needed by the forecaster.\n"
        "    Length `y`: 5.\n"
        "    Max window size: 5.\n"
        "    Lags window size: 5.\n"
        "    Window features window size: None."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(y=y)

    rolling = RollingFeaturesClassification(stats=['mean', 'median'], window_sizes=6)
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=2, window_features=rolling)
    err_msg = re.escape(
        "Length of `y` must be greater than the maximum window size "
        "needed by the forecaster.\n"
        "    Length `y`: 5.\n"
        "    Max window size: 6.\n"
        "    Lags window size: 2.\n"
        "    Window features window size: 6."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(y=y)


def test_create_train_X_y_ValueError_when_y_contains_non_discrete_class_labels():
    """
    Test ValueError is raised when y contains non discrete class labels.
    """
    y = pd.Series(np.array([0.0, 1.5, 2.3, 3.7, 4.1]), name='y')
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=2)

    y_values = y.to_numpy()
    not_allowed = np.mod(y_values, 1) != 0
    examples = ", ".join(map(str, np.unique(y_values[not_allowed])[:5]))
    err_msg = re.escape(
        f"Invalid target for classification: targets must be discrete "
        f"class labels (strings, integers or floats with decimals "
        f"equal to 0). Received float dtype '{y_values.dtype}' with "
        f"decimals (e.g., {examples}). "
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(y=y)


def test_create_train_X_y_ValueError_when_y_only_contains_one_class_label():
    """
    Test ValueError is raised when y contains only one class label.
    """
    y = pd.Series(np.array(["low", "low", "low", "low", "low"]), name='y')
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=2)

    err_msg = re.escape(
        "The target variable must have at least 2 classes. "
        "Found ['low'] class."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(y=y)


def test_create_train_X_y_TypeError_when_exog_is_categorical_of_no_int():
    """
    Test TypeError is raised when exog is categorical with no int values.
    """
    y = pd.Series(np.arange(3))
    exog = pd.Series(['A', 'B', 'C'], name='exog', dtype='category')
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=2)

    err_msg = re.escape(
        "Categorical dtypes in exog must contain only integer values. "
        "See skforecast docs for more info about how to include "
        "categorical features https://skforecast.org/"
        "latest/user_guides/categorical-features.html"
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster._create_train_X_y(y=y, exog=exog)


def test_create_train_X_y_MissingValuesWarning_when_exog_has_missing_values():
    """
    Test _create_train_X_y is issues a MissingValuesWarning when exog has missing values.
    """
    y = pd.Series(np.arange(4))
    exog = pd.Series([1, 2, 3, np.nan], name='exog')
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=2)

    warn_msg = re.escape(
        "`exog` has missing values. Most machine learning models do "
        "not allow missing values. Fitting the forecaster may fail."
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        forecaster._create_train_X_y(y=y, exog=exog)


@pytest.mark.parametrize(
    "y                        , exog", 
    [(pd.Series(np.arange(50), name='y'), pd.Series(np.arange(10), name='exog')), 
     (pd.Series(np.arange(10), name='y'), pd.Series(np.arange(50), name='exog')), 
     (pd.Series(np.arange(10), name='y'), pd.DataFrame(np.arange(50).reshape(25, 2), columns=['exog_1', 'exog_2'])),
     (pd.Series(np.arange(10), index=pd.date_range(start='2022-01-01', periods=10, freq='1D'), name='y'), 
      pd.Series(np.arange(50), index=pd.date_range(start='2022-01-01', periods=50, freq='1D'), name='exog'))
])
def test_create_train_X_y_ValueError_when_len_y_or_len_train_index_is_different_from_len_exog(y, exog):
    """
    Test ValueError is raised when length of y is not equal to length exog or
    length of y - window_size is not equal to length exog.
    """
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)

    len_exog = len(exog)
    len_y = len(y)
    train_index = y.index[forecaster.window_size:]
    len_train_index = len(train_index)
    err_msg = re.escape(
        f"Length of `exog` must be equal to the length of `y` (if index is "
        f"fully aligned) or length of `y` - `window_size` (if `exog` "
        f"starts after the first `window_size` values).\n"
        f"    `exog`              : ({exog.index[0]} -- {exog.index[-1]})  (n={len_exog})\n"
        f"    `y`                 : ({y.index[0]} -- {y.index[-1]})  (n={len_y})\n"
        f"    `y` - `window_size` : ({train_index[0]} -- {train_index[-1]})  (n={len_train_index})"
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster._create_train_X_y(y=y, exog=exog)

  
def test_create_train_X_y_ValueError_when_y_and_exog_have_different_index_but_same_length():
    """
    Test ValueError is raised when y and exog have different index but same length.
    """
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)

    err_msg = re.escape(
        "When `exog` has the same length as `y`, the index of "
        "`exog` must be aligned with the index of `y` "
        "to ensure the correct alignment of values." 
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(
            y    = pd.Series(np.arange(10), index=pd.date_range(start='2022-01-01', periods=10, freq='1D'), name='y'),
            exog = pd.Series(np.arange(10), index=pd.RangeIndex(start=0, stop=10, step=1), name='exog')
        )

  
def test_create_train_X_y_ValueError_when_y_and_exog_have_different_index_and_length_exog_no_window_size():
    """
    Test ValueError is raised when y and exog have different index and
    length exog no window_size.
    """
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)

    err_msg = re.escape(
        "When `exog` doesn't contain the first `window_size` observations, "
        "the index of `exog` must be aligned with the index of `y` minus "
        "the first `window_size` observations to ensure the correct "
        "alignment of values."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.fit(
            y    = pd.Series(np.arange(10), index=pd.date_range(start='2022-01-01', periods=10, freq='1D'), name='y'),
            exog = pd.Series(np.arange(5, 10), index=pd.RangeIndex(start=5, stop=10, step=1), name='exog')
        )


def test_create_train_X_y_output_when_y_is_series_of_int_and_exog_is_None():
    """
    Test the output of _create_train_X_y when exog is None and y is a Series of int.
    """
    y = pd.Series(np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1]), dtype=int, name='y')
    exog = None
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[1., 0., 2., 1., 0.],
                             [2., 1., 0., 2., 1.],
                             [0., 2., 1., 0., 2.],
                             [1., 0., 2., 1., 0.],
                             [2., 1., 0., 2., 1.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ),
        pd.Series(
            data  = np.array([2, 0, 1, 2, 0]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        {
            'classes_': [np.int64(1), np.int64(2), np.int64(3)],
            'class_codes_': [0, 1, 2],
            'n_classes_': 3,
            'encoding_mapping_': {np.int64(1): 0, np.int64(2): 1, np.int64(3): 2}
        },
        None,
        None,
        None,
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
        None,
        None,
        pd.DataFrame(
            data = np.array([3, 1, 2, 3, 1]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['y']
        ),
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        assert results[2][k] == expected[2][k]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    assert results[8] == expected[8]
    pd.testing.assert_frame_equal(results[9], expected[9])


def test_create_train_X_y_output_when_y_is_series_of_str_and_exog_is_None():
    """
    Test the output of _create_train_X_y when exog is None and y is a Series of str.
    """
    y = pd.Series(np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']), name='y')
    exog = None
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[1., 0., 2., 1., 0.],
                             [2., 1., 0., 2., 1.],
                             [0., 2., 1., 0., 2.],
                             [1., 0., 2., 1., 0.],
                             [2., 1., 0., 2., 1.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ),
        pd.Series(
            data  = np.array([2, 0, 1, 2, 0]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        {
            'classes_': ['a', 'b', 'c'],
            'class_codes_': [0, 1, 2],
            'n_classes_': 3,
            'encoding_mapping_': {'a': 0, 'b': 1, 'c': 2}
        },
        None,
        None,
        None,
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
        None,
        None,
        pd.DataFrame(
            data = np.array(['c', 'a', 'b', 'c', 'a']),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['y']
        ),
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        assert results[2][k] == expected[2][k]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    assert results[8] == expected[8]
    pd.testing.assert_frame_equal(results[9], expected[9])


def test_create_train_X_y_output_when_y_is_series_of_int_and_exog_is_None_as_categorical():
    """
    Test the output of _create_train_X_y when exog is None and y is a Series of int
    treated as categorical.
    """
    y = pd.Series(np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1]), dtype=int, name='y')
    exog = None
    forecaster = ForecasterRecursiveClassifier(LGBMClassifier(verbose=-1), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)

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
        ),
        {
            'classes_': [np.int64(1), np.int64(2), np.int64(3)],
            'class_codes_': [0, 1, 2],
            'n_classes_': 3,
            'encoding_mapping_': {np.int64(1): 0, np.int64(2): 1, np.int64(3): 2}
        },
        None,
        None,
        None,
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
        None,
        None,
        pd.DataFrame(
            data = np.array([3, 1, 2, 3, 1]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['y']
        ),
    )

    for col in expected[0].columns:
        expected[0][col] = pd.Categorical(
                               values     = expected[0][col],
                               categories = expected[2]['class_codes_'],
                               ordered    = False
                           )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        assert results[2][k] == expected[2][k]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    assert results[8] == expected[8]
    pd.testing.assert_frame_equal(results[9], expected[9])


def test_create_train_X_y_output_when_y_is_series_of_str_and_exog_is_None_as_categorical():
    """
    Test the output of _create_train_X_y when exog is None and y is a Series of str
    treated as categorical.
    """
    y = pd.Series(np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']), name='y')
    exog = None
    forecaster = ForecasterRecursiveClassifier(LGBMClassifier(verbose=-1), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)

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
        ),
        {
            'classes_': ['a', 'b', 'c'],
            'class_codes_': [0, 1, 2],
            'n_classes_': 3,
            'encoding_mapping_': {'a': 0, 'b': 1, 'c': 2}
        },
        None,
        None,
        None,
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'],
        None,
        None,
        pd.DataFrame(
            data = np.array(['c', 'a', 'b', 'c', 'a']),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['y']
        )
    )

    for col in expected[0].columns:
        expected[0][col] = pd.Categorical(
                               values     = expected[0][col],
                               categories = expected[2]['class_codes_'],
                               ordered    = False
                           )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        assert results[2][k] == expected[2][k]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    assert results[7] == expected[7]
    assert results[8] == expected[8]
    pd.testing.assert_frame_equal(results[9], expected[9])


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_series_of_float_int(dtype):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas series of floats or ints.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=dtype)
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105.],
                             [5., 4., 3., 2., 1., 106.],
                             [6., 5., 4., 3., 2., 107.],
                             [7., 6., 5., 4., 3., 108.],
                             [8., 7., 6., 5., 4., 109.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog']
        ).astype({'exog': float}),
        pd.Series(
            data  = np.array([5., 6., 7., 8., 9.]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        {
            "classes_": [
                np.int64(0),
                np.int64(1),
                np.int64(2),
                np.int64(3),
                np.int64(4),
                np.int64(5),
                np.int64(6),
                np.int64(7),
                np.int64(8),
                np.int64(9),
            ],
            "class_codes_": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "n_classes_": 10,
            "encoding_mapping_": {
                np.int64(0): 0.,
                np.int64(1): 1.,
                np.int64(2): 2.,
                np.int64(3): 3.,
                np.int64(4): 4.,
                np.int64(5): 5.,
                np.int64(6): 6.,
                np.int64(7): 7.,
                np.int64(8): 8.,
                np.int64(9): 9.,
            },
        },
        ['exog'],
        None,
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog'],
        {'exog': exog.dtypes},
        {'exog': exog.dtypes},
        pd.DataFrame(
            data = np.array([5., 6., 7., 8., 9.]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['y']
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        assert results[2][k] == expected[2][k]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    pd.testing.assert_frame_equal(results[9], expected[9])


@pytest.mark.parametrize("datetime_index", 
                         [True, False], 
                         ids = lambda dt: f'datetime_index: {dt}')
@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_series_of_float_int_with_no_window_size(datetime_index, dtype):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas series of floats or ints and no initial window_size
    observations.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.Series(
        np.arange(105, 110), index=pd.RangeIndex(start=5, stop=10, step=1), 
        name='exog', dtype=dtype
    )

    expected_index = pd.RangeIndex(start=5, stop=10, step=1)
    if datetime_index:
        y.index = pd.date_range(start='2022-01-01', periods=10, freq='D')
        exog.index = pd.date_range(start='2022-01-06', periods=5, freq='D')
        expected_index = pd.date_range(start='2022-01-06', periods=5, freq='D')

    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105.],
                             [5., 4., 3., 2., 1., 106.],
                             [6., 5., 4., 3., 2., 107.],
                             [7., 6., 5., 4., 3., 108.],
                             [8., 7., 6., 5., 4., 109.]]),
            index   = expected_index,
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog']
        ).astype({'exog': float}),
        pd.Series(
            data  = np.array([5., 6., 7., 8., 9.]),
            index = expected_index,
            name  = 'y',
            dtype = float
        ),
        {
            "classes_": [
                np.int64(0),
                np.int64(1),
                np.int64(2),
                np.int64(3),
                np.int64(4),
                np.int64(5),
                np.int64(6),
                np.int64(7),
                np.int64(8),
                np.int64(9),
            ],
            "class_codes_": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "n_classes_": 10,
            "encoding_mapping_": {
                np.int64(0): 0.,
                np.int64(1): 1.,
                np.int64(2): 2.,
                np.int64(3): 3.,
                np.int64(4): 4.,
                np.int64(5): 5.,
                np.int64(6): 6.,
                np.int64(7): 7.,
                np.int64(8): 8.,
                np.int64(9): 9.,
            },
        },
        ['exog'],
        None,
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog'],
        {'exog': exog.dtypes},
        {'exog': exog.dtypes},
        pd.DataFrame(
            data = np.array([5., 6., 7., 8., 9.]),
            index   = expected_index,
            columns = ['y']
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        assert results[2][k] == expected[2][k]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    pd.testing.assert_frame_equal(results[9], expected[9])


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe_of_float_int(dtype):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas dataframe with 2 columns of floats or ints.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=dtype),
                         'exog_2': np.arange(1000, 1010, dtype=dtype)})
    
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)        
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105., 1005.],
                             [5., 4., 3., 2., 1., 106., 1006.],
                             [6., 5., 4., 3., 2., 107., 1007.],
                             [7., 6., 5., 4., 3., 108., 1008.],
                             [8., 7., 6., 5., 4., 109., 1009.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1', 'exog_2']
        ).astype({'exog_1': float, 'exog_2': float}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        {
            "classes_": [
                np.int64(0),
                np.int64(1),
                np.int64(2),
                np.int64(3),
                np.int64(4),
                np.int64(5),
                np.int64(6),
                np.int64(7),
                np.int64(8),
                np.int64(9),
            ],
            "class_codes_": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "n_classes_": 10,
            "encoding_mapping_": {
                np.int64(0): 0.,
                np.int64(1): 1.,
                np.int64(2): 2.,
                np.int64(3): 3.,
                np.int64(4): 4.,
                np.int64(5): 5.,
                np.int64(6): 6.,
                np.int64(7): 7.,
                np.int64(8): 8.,
                np.int64(9): 9.,
            },
        },
        ['exog_1', 'exog_2'],
        None,
        ['exog_1', 'exog_2'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1', 'exog_2'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
        pd.DataFrame(
            data = np.array([5., 6., 7., 8., 9.]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['y']
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        assert results[2][k] == expected[2][k]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    pd.testing.assert_frame_equal(results[9], expected[9])


@pytest.mark.parametrize("exog_values, dtype", 
                         [([True]    , bool), 
                          (['string'], str)], 
                         ids = lambda dt: f'values, dtype: {dt}')
def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_series_of_bool_str(exog_values, dtype):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas series of bool or str.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.Series(exog_values * 10, name='exog', dtype=dtype)
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog=exog_values * 5).astype({'exog': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        {
            "classes_": [
                np.int64(0),
                np.int64(1),
                np.int64(2),
                np.int64(3),
                np.int64(4),
                np.int64(5),
                np.int64(6),
                np.int64(7),
                np.int64(8),
                np.int64(9),
            ],
            "class_codes_": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "n_classes_": 10,
            "encoding_mapping_": {
                np.int64(0): 0.,
                np.int64(1): 1.,
                np.int64(2): 2.,
                np.int64(3): 3.,
                np.int64(4): 4.,
                np.int64(5): 5.,
                np.int64(6): 6.,
                np.int64(7): 7.,
                np.int64(8): 8.,
                np.int64(9): 9.,
            },
        },
        ['exog'],
        None,
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog'],
        {'exog': exog.dtypes},
        {'exog': exog.dtypes},
        pd.DataFrame(
            data = np.array([5., 6., 7., 8., 9.]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['y']
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        assert results[2][k] == expected[2][k]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    pd.testing.assert_frame_equal(results[9], expected[9])


@pytest.mark.parametrize("v_exog_1   , v_exog_2  , dtype", 
                         [([True]    , [False]   , bool), 
                          (['string'], ['string'], str)], 
                         ids = lambda dt: f'values, dtype: {dt}')
def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe_of_bool_str(v_exog_1, v_exog_2, dtype):
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas dataframe with two columns of bool or str.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.DataFrame({
               'exog_1': v_exog_1 * 10,
               'exog_2': v_exog_2 * 10,
           })
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog_1=v_exog_1 * 5, exog_2=v_exog_2 * 5).astype({'exog_1': dtype, 'exog_2': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        {
            "classes_": [
                np.int64(0),
                np.int64(1),
                np.int64(2),
                np.int64(3),
                np.int64(4),
                np.int64(5),
                np.int64(6),
                np.int64(7),
                np.int64(8),
                np.int64(9),
            ],
            "class_codes_": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "n_classes_": 10,
            "encoding_mapping_": {
                np.int64(0): 0.,
                np.int64(1): 1.,
                np.int64(2): 2.,
                np.int64(3): 3.,
                np.int64(4): 4.,
                np.int64(5): 5.,
                np.int64(6): 6.,
                np.int64(7): 7.,
                np.int64(8): 8.,
                np.int64(9): 9.,
            },
        },
        ['exog_1', 'exog_2'],
        None,
        ['exog_1', 'exog_2'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1', 'exog_2'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
        pd.DataFrame(
            data = np.array([5., 6., 7., 8., 9.]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['y']
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        assert results[2][k] == expected[2][k]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    pd.testing.assert_frame_equal(results[9], expected[9])


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_series_of_category():
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas series of category.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.Series(range(10), name='exog', dtype='category')
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog=pd.Categorical(range(5, 10), categories=range(10))),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        {
            "classes_": [
                np.int64(0),
                np.int64(1),
                np.int64(2),
                np.int64(3),
                np.int64(4),
                np.int64(5),
                np.int64(6),
                np.int64(7),
                np.int64(8),
                np.int64(9),
            ],
            "class_codes_": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "n_classes_": 10,
            "encoding_mapping_": {
                np.int64(0): 0.,
                np.int64(1): 1.,
                np.int64(2): 2.,
                np.int64(3): 3.,
                np.int64(4): 4.,
                np.int64(5): 5.,
                np.int64(6): 6.,
                np.int64(7): 7.,
                np.int64(8): 8.,
                np.int64(9): 9.,
            },
        },
        ['exog'],
        None,
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog'],
        {'exog': exog.dtypes},
        {'exog': exog.dtypes},
        pd.DataFrame(
            data = np.array([5., 6., 7., 8., 9.]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['y']
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        assert results[2][k] == expected[2][k]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    pd.testing.assert_frame_equal(results[9], expected[9])


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe_of_category():
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas dataframe with two columns of category.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.DataFrame({'exog_1': pd.Categorical(range(10)),
                         'exog_2': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)        
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(
            exog_1=pd.Categorical(range(5, 10), categories=range(10)),
            exog_2=pd.Categorical(range(105, 110), categories=range(100, 110))
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        {
            "classes_": [
                np.int64(0),
                np.int64(1),
                np.int64(2),
                np.int64(3),
                np.int64(4),
                np.int64(5),
                np.int64(6),
                np.int64(7),
                np.int64(8),
                np.int64(9),
            ],
            "class_codes_": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "n_classes_": 10,
            "encoding_mapping_": {
                np.int64(0): 0.,
                np.int64(1): 1.,
                np.int64(2): 2.,
                np.int64(3): 3.,
                np.int64(4): 4.,
                np.int64(5): 5.,
                np.int64(6): 6.,
                np.int64(7): 7.,
                np.int64(8): 8.,
                np.int64(9): 9.,
            },
        },
        ['exog_1', 'exog_2'],
        None,
        ['exog_1', 'exog_2'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1', 'exog_2'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes},
        pd.DataFrame(
            data = np.array([5., 6., 7., 8., 9.]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['y']
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        assert results[2][k] == expected[2][k]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    pd.testing.assert_frame_equal(results[9], expected[9])


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe_of_float_int_category():
    """
    Test the output of _create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas dataframe with 3 columns of float, int, category.
    """
    y = pd.Series(np.arange(10), dtype=float)
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)        
    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105., 1005.],
                             [5., 4., 3., 2., 1., 106., 1006.],
                             [6., 5., 4., 3., 2., 107., 1007.],
                             [7., 6., 5., 4., 3., 108., 1008.],
                             [8., 7., 6., 5., 4., 109., 1009.]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'exog_1', 'exog_2']
        ).astype({'exog_1': float, 
                  'exog_2': int}
        ).assign(exog_3=pd.Categorical(range(105, 110), categories=range(100, 110))
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        {
            "classes_": [
                np.int64(0),
                np.int64(1),
                np.int64(2),
                np.int64(3),
                np.int64(4),
                np.int64(5),
                np.int64(6),
                np.int64(7),
                np.int64(8),
                np.int64(9),
            ],
            "class_codes_": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "n_classes_": 10,
            "encoding_mapping_": {
                np.int64(0): 0.,
                np.int64(1): 1.,
                np.int64(2): 2.,
                np.int64(3): 3.,
                np.int64(4): 4.,
                np.int64(5): 5.,
                np.int64(6): 6.,
                np.int64(7): 7.,
                np.int64(8): 8.,
                np.int64(9): 9.,
            },
        },
        ['exog_1', 'exog_2', 'exog_3'],
        None,
        ['exog_1', 'exog_2', 'exog_3'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1', 'exog_2', 'exog_3'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes, 'exog_3': exog['exog_3'].dtypes},
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes, 'exog_3': exog['exog_3'].dtypes},
        pd.DataFrame(
            data = np.array([5., 6., 7., 8., 9.]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['y']
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        assert results[2][k] == expected[2][k]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    pd.testing.assert_frame_equal(results[9], expected[9])


def test_create_train_X_y_output_when_y_str_as_category_and_exog_is_dataframe_of_float_int_category():
    """
    Test the output of _create_train_X_y when y is str as category and 
    exog is a pandas dataframe with 3 columns of float, int, category.
    """
    y = pd.Series(np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']), name='y')
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})
    
    forecaster = ForecasterRecursiveClassifier(LGBMClassifier(verbose=-1), lags=5)
    results = forecaster._create_train_X_y(y=y, exog=exog)        
    expected = (
        pd.DataFrame(
            data = np.array([[1, 0, 2, 1, 0, 105, 1005],
                             [2, 1, 0, 2, 1, 106, 1006],
                             [0, 2, 1, 0, 2, 107, 1007],
                             [1, 0, 2, 1, 0, 108, 1008],
                             [2, 1, 0, 2, 1, 109, 1009]]),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'exog_1', 'exog_2']
        ).astype({'exog_1': float}
        ).assign(exog_3=pd.Categorical(range(105, 110), categories=range(100, 110))
        ),
        pd.Series(
            data  = np.array([2, 0, 1, 2, 0]),
            index = pd.RangeIndex(start=5, stop=10, step=1),
            name  = 'y',
            dtype = int
        ),
        {
            'classes_': ['a', 'b', 'c'],
            'class_codes_': [0, 1, 2],
            'n_classes_': 3,
            'encoding_mapping_': {'a': 0, 'b': 1, 'c': 2}
        },
        ['exog_1', 'exog_2', 'exog_3'],
        None,
        ['exog_1', 'exog_2', 'exog_3'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'exog_1', 'exog_2', 'exog_3'],
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes, 'exog_3': exog['exog_3'].dtypes},
        {'exog_1': exog['exog_1'].dtypes, 'exog_2': exog['exog_2'].dtypes, 'exog_3': exog['exog_3'].dtypes},
        pd.DataFrame(
            data = np.array(['c', 'a', 'b', 'c', 'a']),
            index   = pd.RangeIndex(start=5, stop=10, step=1),
            columns = ['y']
        )
    )

    for col in ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']:
        expected[0][col] = pd.Categorical(
                               values     = expected[0][col],
                               categories = expected[2]['class_codes_'],
                               ordered    = False
                           )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        assert results[2][k] == expected[2][k]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    pd.testing.assert_frame_equal(results[9], expected[9])


def test_create_train_X_y_output_when_transformer_exog():
    """
    Test the output of _create_train_X_y when using transformer_exog.
    """
    y = pd.Series(np.arange(8), dtype = int)
    y.index = pd.date_range("1990-01-01", periods=8, freq='D')
    exog = pd.DataFrame({
               'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4],
               'col_2': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']},
               index = pd.date_range("1990-01-01", periods=8, freq='D')
           )

    transformer_exog = ColumnTransformer(
                            [('scale', StandardScaler(), ['col_1']),
                             ('onehot', OneHotEncoder(), ['col_2'])],
                            remainder = 'passthrough',
                            verbose_feature_names_out = False
                        )

    forecaster = ForecasterRecursiveClassifier(
                    regressor        = LogisticRegression(),
                    lags             = 5,
                    transformer_exog = transformer_exog
                )
    results = forecaster._create_train_X_y(y=y, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([
                [ 4.        ,  3.        ,  2.        ,  1.        ,  0.        ,
                 -0.25107995,  0.        ,  1.        ],
                [ 5.        ,  4.        ,  3.        ,  2.        ,  1.        ,
                  1.79326881,  0.        ,  1.        ],
                [ 6.        ,  5.        ,  4.        ,  3.        ,  2.        ,
                  0.01673866,  0.        ,  1.        ]]),
            index   = pd.date_range("1990-01-06", periods=3, freq='D'),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'col_1',
                       'col_2_a', 'col_2_b']
        ),
        pd.Series(
            data  = np.array([5., 6., 7.]),
            index = pd.date_range("1990-01-06", periods=3, freq='D'),
            name  = 'y',
            dtype = float
        ),
        {
            "classes_": [
                np.int64(0),
                np.int64(1),
                np.int64(2),
                np.int64(3),
                np.int64(4),
                np.int64(5),
                np.int64(6),
                np.int64(7)
            ],
            "class_codes_": [0., 1., 2., 3., 4., 5., 6., 7.],
            "n_classes_": 8,
            "encoding_mapping_": {
                np.int64(0): 0.,
                np.int64(1): 1.,
                np.int64(2): 2.,
                np.int64(3): 3.,
                np.int64(4): 4.,
                np.int64(5): 5.,
                np.int64(6): 6.,
                np.int64(7): 7.,
            },
        },
        ['col_1', 'col_2'],
        None,
        ['col_1', 'col_2_a', 'col_2_b'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'col_1', 'col_2_a', 'col_2_b'],
        {'col_1': exog['col_1'].dtypes, 'col_2': exog['col_2'].dtypes},
        {'col_1': exog['col_1'].dtypes, 'col_2_a': float, 'col_2_b': float},
        pd.DataFrame(
            data = np.array([3, 4, 5, 6, 7]),
            index   = pd.date_range("1990-01-04", periods=5, freq='D'),
            columns = ['y']
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        assert results[2][k] == expected[2][k]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    pd.testing.assert_frame_equal(results[9], expected[9])


def test_create_train_X_y_output_when_window_features_and_exog():
    """
    Test the output of _create_train_X_y when using window_features and exog 
    with datetime index.
    """
    y_datetime = pd.Series(
        np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'c', 'b', 'a', 'b', 'b', 'c']), 
        index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='y'
    )
    exog_datetime = pd.Series(
        np.arange(100, 115), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='exog', dtype=float
    )
    rolling = RollingFeaturesClassification(
        stats=['proportion', 'mode'], window_sizes=[5, 6]
    )

    forecaster = ForecasterRecursiveClassifier(
        LogisticRegression(), lags=5, window_features=rolling
    )
    results = forecaster._create_train_X_y(y=y_datetime, exog=exog_datetime)
    
    expected = (
        pd.DataFrame(
            data=np.array(
                [
                    [2.0, 1.0, 0.0, 2.0, 1.0, 0.2, 0.4, 0.4, 0.0, 106.0],
                    [0.0, 2.0, 1.0, 0.0, 2.0, 0.4, 0.2, 0.4, 0.0, 107.0],
                    [1.0, 0.0, 2.0, 1.0, 0.0, 0.4, 0.4, 0.2, 0.0, 108.0],
                    [2.0, 1.0, 0.0, 2.0, 1.0, 0.2, 0.4, 0.4, 0.0, 109.0],
                    [2.0, 2.0, 1.0, 0.0, 2.0, 0.2, 0.2, 0.6, 2.0, 110.0],
                    [1.0, 2.0, 2.0, 1.0, 0.0, 0.2, 0.4, 0.4, 2.0, 111.0],
                    [0.0, 1.0, 2.0, 2.0, 1.0, 0.2, 0.4, 0.4, 0.0, 112.0],
                    [1.0, 0.0, 1.0, 2.0, 2.0, 0.2, 0.4, 0.4, 1.0, 113.0],
                    [1.0, 1.0, 0.0, 1.0, 2.0, 0.2, 0.6, 0.2, 1.0, 114.0],
                ]
            ),
            index   = pd.date_range('2000-01-07', periods=9, freq='D'),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
                       'roll_proportion_5_class_0.0', 'roll_proportion_5_class_1.0',
                       'roll_proportion_5_class_2.0', 'roll_mode_6', 'exog']
        ),
        pd.Series(
            data  = np.array([0., 1., 2., 2., 1., 0., 1., 1., 2.]),
            index = pd.date_range('2000-01-07', periods=9, freq='D'),
            name  = 'y',
            dtype = float
        ),
        {
            'classes_': ['a', 'b', 'c'],
            'class_codes_': [0, 1, 2],
            'n_classes_': 3,
            'encoding_mapping_': {'a': 0, 'b': 1, 'c': 2}
        },
        ['exog'],
        ['roll_proportion_5_class_0.0', 'roll_proportion_5_class_1.0',
         'roll_proportion_5_class_2.0', 'roll_mode_6'],
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
         'roll_proportion_5_class_0.0', 'roll_proportion_5_class_1.0',
         'roll_proportion_5_class_2.0', 'roll_mode_6', 'exog'],
        {'exog': exog_datetime.dtypes},
        {'exog': exog_datetime.dtypes},
        pd.DataFrame(
            data = np.array(['c', 'b', 'a', 'b', 'b', 'c']),
            index   = pd.date_range('2000-01-10', periods=6, freq='D'),
            columns = ['y']
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        assert results[2][k] == expected[2][k]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    pd.testing.assert_frame_equal(results[9], expected[9])


def test_create_train_X_y_output_when_two_window_features_and_exog():
    """
    Test the output of _create_train_X_y when using 2 window_features and exog 
    with datetime index.
    """
    y_datetime = pd.Series(
        np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'c', 'b', 'a', 'b', 'b', 'c']), 
        index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='y'
    )
    exog_datetime = pd.Series(
        np.arange(100, 115), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='exog', dtype=float
    )
    rolling = RollingFeaturesClassification(stats='proportion', window_sizes=[5])
    rolling_2 = RollingFeaturesClassification(stats='mode', window_sizes=6)

    forecaster = ForecasterRecursiveClassifier(
        LogisticRegression(), lags=5, window_features=[rolling, rolling_2]
    )
    results = forecaster._create_train_X_y(y=y_datetime, exog=exog_datetime)
    
    expected = (
        pd.DataFrame(
            data=np.array(
                [
                    [2.0, 1.0, 0.0, 2.0, 1.0, 0.2, 0.4, 0.4, 0.0, 106.0],
                    [0.0, 2.0, 1.0, 0.0, 2.0, 0.4, 0.2, 0.4, 0.0, 107.0],
                    [1.0, 0.0, 2.0, 1.0, 0.0, 0.4, 0.4, 0.2, 0.0, 108.0],
                    [2.0, 1.0, 0.0, 2.0, 1.0, 0.2, 0.4, 0.4, 0.0, 109.0],
                    [2.0, 2.0, 1.0, 0.0, 2.0, 0.2, 0.2, 0.6, 2.0, 110.0],
                    [1.0, 2.0, 2.0, 1.0, 0.0, 0.2, 0.4, 0.4, 2.0, 111.0],
                    [0.0, 1.0, 2.0, 2.0, 1.0, 0.2, 0.4, 0.4, 0.0, 112.0],
                    [1.0, 0.0, 1.0, 2.0, 2.0, 0.2, 0.4, 0.4, 1.0, 113.0],
                    [1.0, 1.0, 0.0, 1.0, 2.0, 0.2, 0.6, 0.2, 1.0, 114.0],
                ]
            ),
            index   = pd.date_range('2000-01-07', periods=9, freq='D'),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
                       'roll_proportion_5_class_0.0', 'roll_proportion_5_class_1.0',
                       'roll_proportion_5_class_2.0', 'roll_mode_6', 'exog']
        ),
        pd.Series(
            data  = np.array([0., 1., 2., 2., 1., 0., 1., 1., 2.]),
            index = pd.date_range('2000-01-07', periods=9, freq='D'),
            name  = 'y',
            dtype = float
        ),
        {
            'classes_': ['a', 'b', 'c'],
            'class_codes_': [0, 1, 2],
            'n_classes_': 3,
            'encoding_mapping_': {'a': 0, 'b': 1, 'c': 2}
        },
        ['exog'],
        ['roll_proportion_5_class_0.0', 'roll_proportion_5_class_1.0',
         'roll_proportion_5_class_2.0', 'roll_mode_6'],
        ['exog'],
        ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
         'roll_proportion_5_class_0.0', 'roll_proportion_5_class_1.0',
         'roll_proportion_5_class_2.0', 'roll_mode_6', 'exog'],
        {'exog': exog_datetime.dtypes},
        {'exog': exog_datetime.dtypes},
        pd.DataFrame(
            data = np.array(['c', 'b', 'a', 'b', 'b', 'c']),
            index   = pd.date_range('2000-01-10', periods=6, freq='D'),
            columns = ['y']
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        assert results[2][k] == expected[2][k]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    pd.testing.assert_frame_equal(results[9], expected[9])


def test_create_train_X_y_output_when_window_features_lags_None_and_exog_store_last_window_False():
    """
    Test the output of _create_train_X_y when using window_features and exog 
    with datetime index, lags=None and store_last_window is False.
    """
    y_datetime = pd.Series(
        np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'c', 'b', 'a', 'b', 'b', 'c']), 
        index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='y'
    )
    exog_datetime = pd.Series(
        np.arange(100, 115), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='exog', dtype=float
    )
    rolling = RollingFeaturesClassification(stats='proportion', window_sizes=[5])
    rolling_2 = RollingFeaturesClassification(stats='mode', window_sizes=6)

    forecaster = ForecasterRecursiveClassifier(
        LogisticRegression(), lags=None, window_features=[rolling, rolling_2]
    )
    results = forecaster._create_train_X_y(
        y=y_datetime, exog=exog_datetime, store_last_window=False
    )
    
    expected = (
        pd.DataFrame(
            data=np.array(
                [
                    [0.2, 0.4, 0.4, 0.0, 106.0],
                    [0.4, 0.2, 0.4, 0.0, 107.0],
                    [0.4, 0.4, 0.2, 0.0, 108.0],
                    [0.2, 0.4, 0.4, 0.0, 109.0],
                    [0.2, 0.2, 0.6, 2.0, 110.0],
                    [0.2, 0.4, 0.4, 2.0, 111.0],
                    [0.2, 0.4, 0.4, 0.0, 112.0],
                    [0.2, 0.4, 0.4, 1.0, 113.0],
                    [0.2, 0.6, 0.2, 1.0, 114.0],
                ]
            ),
            index   = pd.date_range('2000-01-07', periods=9, freq='D'),
            columns = ['roll_proportion_5_class_0.0', 'roll_proportion_5_class_1.0',
                       'roll_proportion_5_class_2.0', 'roll_mode_6', 'exog']
        ),
        pd.Series(
            data  = np.array([0., 1., 2., 2., 1., 0., 1., 1., 2.]),
            index = pd.date_range('2000-01-07', periods=9, freq='D'),
            name  = 'y',
            dtype = float
        ),
        {
            'classes_': ['a', 'b', 'c'],
            'class_codes_': [0, 1, 2],
            'n_classes_': 3,
            'encoding_mapping_': {'a': 0, 'b': 1, 'c': 2}
        },
        ['exog'],
        ['roll_proportion_5_class_0.0', 'roll_proportion_5_class_1.0',
         'roll_proportion_5_class_2.0', 'roll_mode_6'],
        ['exog'],
        ['roll_proportion_5_class_0.0', 'roll_proportion_5_class_1.0',
         'roll_proportion_5_class_2.0', 'roll_mode_6', 'exog'],
        {'exog': exog_datetime.dtypes},
        {'exog': exog_datetime.dtypes},
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        assert results[2][k] == expected[2][k]
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    assert results[5] == expected[5]
    assert results[6] == expected[6]
    for k in results[7].keys():
        assert results[7][k] == expected[7][k]
    for k in results[8].keys():
        assert results[8][k] == expected[8][k]
    assert results[9] == expected[9]
