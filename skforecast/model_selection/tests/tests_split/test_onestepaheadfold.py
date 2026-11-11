# Unit test OneStepAheadFold
# ==============================================================================
import re
import warnings
import pytest
import numpy as np
import pandas as pd
from skforecast.model_selection._split import OneStepAheadFold

# Fixtures
from ..fixtures_model_selection_multiseries import series_dict_dt_different_periods


def test_OneStepAheadFold_split_raise_error_when_X_is_not_series_dataframe_or_dict():
    """
    Test that ValueError is raised when X is not a pd.Series, pd.DataFrame or dict.
    """
    X = np.arange(100)
    cv = OneStepAheadFold(initial_train_size=70)
    
    err_msg = re.escape(
        f"X must be a pandas Series, DataFrame, Index or a dictionary. "
        f"Got {type(X)}."
    )
    with pytest.raises(TypeError, match=err_msg):
        cv.split(X=X)


@pytest.mark.parametrize("invalid_date", [
    "2021-12-31",  # Before the first date in the index
    "2022-04-11",  # After the last date in the index
])
def test_OneStepAhead_split_invalid_initial_train_size_date(invalid_date):
    """
    Test that ValueError is raised when initial_train_size date is outside the index range.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start="2022-01-01", periods=100, freq="D")
    cv = OneStepAheadFold(initial_train_size=invalid_date)
    
    err_msg = re.escape(
        "If `initial_train_size` is a date, it must be within the index "
        "range, between the first and the last date (both included)."
    )
    with pytest.raises(ValueError, match=err_msg):
        cv.split(X=y)


@pytest.mark.parametrize("return_all_indexes, expected",
                         [(True, [0, [range(0, 70)], [range(70, 100)], True]),
                          (False, [0, [0, 70], [70, 100], True])], 
                         ids = lambda argument: f'{argument}')
def test_OneStepAhead_split_initial_train_size(capfd, return_all_indexes, expected):
    """
    Test OneStepAhead splits when initial_train_size is provided.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    cv = OneStepAheadFold(
            initial_train_size = 70,
            return_all_indexes = return_all_indexes,
        )
    folds = cv.split(X=y)
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations in train: 70\n"
        "Number of observations in test: 30\n"
        "Training : 2022-01-01 00:00:00 -- 2022-03-11 00:00:00 (n=70)\n"
        "Test     : 2022-03-12 00:00:00 -- 2022-04-10 00:00:00 (n=30)\n\n"
    )

    assert out == expected_out
    assert folds == expected


@pytest.mark.parametrize(
    "deprecated_params",
    [{'differentiation': 1}, {'window_size': 3}, {'window_size': 3, 'differentiation': 1}],
    ids=lambda params: f'{params}'
)
def test_OneStepAheadFold_FutureWarning_when_deprecated_params_are_set(capfd, deprecated_params):
    """
    Test FutureWarning is raised when `differentiation` or `window_size` are
    set, in `__init__` or with `set_params`, and that they have no effect: the
    fold and the printed information are the same as without them, and they
    are not shown in the representation of the object.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    warn_msgs = {
        'differentiation': (
            "`differentiation` is deprecated since version 0.27.0 and has no "
            "effect on the folds of `OneStepAheadFold`: the `window_size` of the "
            "forecaster already includes its differentiation order. It will "
            "be removed in version 0.29.0."
        ),
        'window_size': (
            "`window_size` is deprecated since version 0.27.0 and has no "
            "effect on the fold of `OneStepAheadFold`. It will be removed in "
            "version 0.29.0."
        ),
    }

    with pytest.warns(FutureWarning) as record:
        cv = OneStepAheadFold(initial_train_size=70, **deprecated_params)
    assert sorted(str(w.message) for w in record) == sorted(
        warn_msgs[param] for param in deprecated_params
    )
    folds = cv.split(X=y)
    out, _ = capfd.readouterr()

    cv_default = OneStepAheadFold(initial_train_size=70)
    with pytest.warns(FutureWarning) as record:
        cv_default.set_params(deprecated_params)
    assert len(record) == len(deprecated_params)
    folds_default = cv_default.split(X=y)
    out_default, _ = capfd.readouterr()

    assert folds == folds_default == [0, [0, 70], [70, 100], True]
    assert out == out_default
    assert "Number of observations in train: 70\n" in out
    assert "differentiation" not in out
    assert "window_size" not in repr(cv)
    assert "differentiation" not in repr(cv)
    assert "Differentiation" not in cv._repr_html_()


@pytest.mark.parametrize(
    "window_size", [0, -1, "invalid"], ids=lambda ws: f'window_size: {ws}'
)
def test_OneStepAheadFold_deprecated_window_size_is_not_validated(window_size):
    """
    Test that an invalid `window_size` only raises the FutureWarning, since it
    is deprecated and its value is ignored. Subclasses also raise the warning.
    """
    class CustomOneStepAheadFold(OneStepAheadFold):
        pass

    for cv_class in [OneStepAheadFold, CustomOneStepAheadFold]:
        with pytest.warns(FutureWarning, match="`window_size` is deprecated"):
            cv = cv_class(initial_train_size=70, window_size=window_size)
        assert cv.split(X=pd.Series(np.arange(100))) == [0, [0, 70], [70, 100], True]


def test_OneStepAheadFold_no_FutureWarning_when_deprecated_params_are_None():
    """
    Test no FutureWarning is raised when `differentiation` and `window_size`
    are `None`, in `__init__` or with `set_params`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=FutureWarning)
        cv = OneStepAheadFold(
            initial_train_size=70, window_size=None, differentiation=None
        )
        cv.set_params({'window_size': None, 'differentiation': None})


def test_OneStepAhead_split_initial_train_size_return_all_indexes_true_as_pandas_true(capfd):
    """
    Test OneStepAhead splits when initial_train_size is provided, output as
    pandas DataFrame.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    cv = OneStepAheadFold(
            initial_train_size = 70,
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
        "Training : 2022-01-01 00:00:00 -- 2022-03-11 00:00:00 (n=70)\n"
        "Test     : 2022-03-12 00:00:00 -- 2022-04-10 00:00:00 (n=30)\n\n"
    )

    assert out == expected_out
    pd.testing.assert_frame_equal(folds, expected_folds)


def test_OneStepAhead_split_initial_train_size_return_all_indexes_false_as_pandas_true(capfd):
    """
    Test OneStepAhead splits when initial_train_size is provided, output as
    pandas DataFrame.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start='2022-01-01', periods=100, freq='D')
    cv = OneStepAheadFold(
            initial_train_size = 70,
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
        "Training : 2022-01-01 00:00:00 -- 2022-03-11 00:00:00 (n=70)\n"
        "Test     : 2022-03-12 00:00:00 -- 2022-04-10 00:00:00 (n=30)\n\n"
    )

    assert out == expected_out
    pd.testing.assert_frame_equal(folds, expected_folds)


@pytest.mark.parametrize(
    "initial_train_size, expected",
    [
        (70, [0, [0, 70], [70, 100], True]),
        ("2022-03-11", [0, [0, 70], [70, 100], True]),
        ("2022-03-11 00:00:00", [0, [0, 70], [70, 100], True]),
        (pd.to_datetime("2022-03-11"), [0, [0, 70], [70, 100], True])
    ],
    ids=lambda x: f'initial_train_size={x}',
)
def test_OneStepAhead_split_int_and_date_initial_train_size(capfd, initial_train_size, expected):
    """
    Test OneStepAhead splits with different types for initial_train_size.
    """
    y = pd.Series(np.arange(100))
    y.index = pd.date_range(start="2022-01-01", periods=100, freq="D")
    cv = OneStepAheadFold(initial_train_size=initial_train_size)
    folds = cv.split(X=y)
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations in train: 70\n"
        "Number of observations in test: 30\n"
        "Training : 2022-01-01 00:00:00 -- 2022-03-11 00:00:00 (n=70)\n"
        "Test     : 2022-03-12 00:00:00 -- 2022-04-10 00:00:00 (n=30)\n\n"
    )

    assert out == expected_out
    assert folds == expected
    # `initial_train_size` is immutable user input; the resolved integer position
    # is stored in `initial_train_size_as_int` (split() must not mutate the input).
    assert cv.initial_train_size == initial_train_size
    assert cv.initial_train_size_as_int == 70


@pytest.mark.parametrize(
    "as_pandas",
    [False, True],
    ids=lambda as_pandas: f"as_pandas: {as_pandas}",
)
def test_OneStepAhead_split_dict_of_series_that_do_not_cover_the_same_period(capfd, as_pandas):
    """
    Test OneStepAhead splits when X is a dict of series that do not cover the
    same period. The test set ends at the end of the span of the series (50),
    not at the number of series.
    """
    cv = OneStepAheadFold(initial_train_size=40)
    folds = cv.split(X=series_dict_dt_different_periods, as_pandas=as_pandas)
    out, _ = capfd.readouterr()
    expected_out = (
        "Information of folds\n"
        "--------------------\n"
        "Number of observations in train: 40\n"
        "Number of observations in test: 10\n"
        "Training : 2020-01-01 00:00:00 -- 2020-02-09 00:00:00 (n=40)\n"
        "Test     : 2020-02-10 00:00:00 -- 2020-02-19 00:00:00 (n=10)\n\n"
    )

    assert out == expected_out
    if as_pandas:
        expected_folds = pd.DataFrame(
            {
                'fold': [0],
                'train_start': [0],
                'train_end': [40],
                'test_start': [40],
                'test_end': [50],
                'fit_forecaster': [True]
            }
        )
        pd.testing.assert_frame_equal(folds, expected_folds)
    else:
        assert folds == [0, [0, 40], [40, 50], True]
