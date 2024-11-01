# Unit test OneStepAheadFold
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.model_selection._split import OneStepAheadFold


@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [[range(0, 70)], [range(70, 100)], True]),
                          (False, [[0, 70], [70, 100], True])], 
                         ids = lambda argument: f'{argument}')
def test_OneStepAhead_split_initial_train_size_and_window_size(capfd, return_all_indexes, expected):
    """
    Test OneStepAhead splits when initial_train_size and window_size are provided.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    cv = OneStepAheadFold(
            initial_train_size = 70,
            window_size        = 3,
            differentiation    = None,
            return_all_indexes = return_all_indexes,
        )
    folds = cv.split(X=y)
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations in train: 70\n"
        "Number of observations in test: 30\n"
        "Training : 2022-01-01 00:00:00 -- 2022-03-12 00:00:00 (n=70)\n"
        "Test     : 2022-03-12 00:00:00 -- 2022-04-10 00:00:00 (n=30)\n\n"
    )

    assert out == expected_out
    assert folds == expected


def test_OneStepAhead_split_initial_train_size_and_window_size_diferentiation_is_1(capfd):
    """
    Test OneStepAhead splits when initial_train_size and window_size are provided, and
    differentiation is 1.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    cv = OneStepAheadFold(
            initial_train_size = 70,
            window_size        = 3,
            differentiation    = 1,
            return_all_indexes = False,
        )
    folds = cv.split(X=y)
    out, _ = capfd.readouterr()
    expected_folds = [[0, 70], [70, 100], True]
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations in train: 69\n"
        "    First 1 observation/s in training set are used for differentiation\n"
        "Number of observations in test: 30\n"
        "Training : 2022-01-02 00:00:00 -- 2022-03-12 00:00:00 (n=69)\n"
        "Test     : 2022-03-12 00:00:00 -- 2022-04-10 00:00:00 (n=30)\n\n"
    )

    assert out == expected_out
    assert folds == expected_folds


def test_OneStepAhead_split_initial_train_size_window_size_return_all_indexes_true_as_pandas_true(capfd):
    """
    Test OneStepAhead splits when initial_train_size and window_size are provided, output as
    pandas DataFrame.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    cv = OneStepAheadFold(
            initial_train_size = 70,
            window_size        = 3,
            differentiation    = None,
            return_all_indexes = True,
        )
    folds = cv.split(X=y, as_pandas=True)
    out, _ = capfd.readouterr()
    expected_folds = pd.DataFrame(
        {
            "fold": {0: 0},
            "train_index": {0: [range(0, 70)]},
            "test_index": {0: [range(70, 100)]},
            "fit_forecaster": {0: True},
        }
    )
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations in train: 70\n"
        "Number of observations in test: 30\n"
        "Training : 2022-01-01 00:00:00 -- 2022-03-12 00:00:00 (n=70)\n"
        "Test     : 2022-03-12 00:00:00 -- 2022-04-10 00:00:00 (n=30)\n\n"
    )

    assert out == expected_out
    pd.testing.assert_frame_equal(folds, expected_folds)


def test_OneStepAhead_split_initial_train_size_window_size_return_all_indexes_false_as_pandas_true(capfd):
    """
    Test OneStepAhead splits when initial_train_size and window_size are provided, output as
    pandas DataFrame.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    cv = OneStepAheadFold(
            initial_train_size = 70,
            window_size        = 3,
            differentiation    = None,
            return_all_indexes = False,
        )
    folds = cv.split(X=y, as_pandas=True)
    out, _ = capfd.readouterr()
    expected_folds = pd.DataFrame(
        {
            'fold': [0],
            'train_start': [0],
            'train_end': [70],
            'test_start': [70],
            'test_end': [100],
            'fit_forecaster': [True]
        }
    )
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations in train: 70\n"
        "Number of observations in test: 30\n"
        "Training : 2022-01-01 00:00:00 -- 2022-03-12 00:00:00 (n=70)\n"
        "Test     : 2022-03-12 00:00:00 -- 2022-04-10 00:00:00 (n=30)\n\n"
    )

    assert out == expected_out
    pd.testing.assert_frame_equal(folds, expected_folds)


def test_onestepaheadfold_split_raise_error_when_X_is_not_series_dataframe_or_dict():
    """
    Test that ValueError is raised when X is not a pd.Series, pd.DataFrame or dict.
    """
    X = np.arange(100)
    cv = OneStepAheadFold(initial_train_size=70)
    msg = (
        f"X must be a pandas Series, DataFrame, Index or a dictionary. Got {type(X)}.")
    with pytest.raises(TypeError, match=msg):
        cv.split(X=X)
