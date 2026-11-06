# Unit test _train_test_split_one_step_ahead_search
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.model_selection._split import OneStepAheadFold
from skforecast.model_selection._utils import _train_test_split_one_step_ahead_search

# NOTE: `short` has 5 values in the training set of `initial_train_size=45`.
index = pd.date_range('2020-01-01', periods=60, freq='D')
series = {
    'l1': pd.Series(10 + np.sin(np.arange(60)), index=index),
    'l2': pd.Series(20 + 2 * np.cos(np.arange(60) / 2), index=index),
    'short': pd.Series(5 + np.arange(20) % 3, index=index[-20:], dtype=float),
}


@pytest.mark.parametrize(
    "forecaster, data_name",
    [(ForecasterRecursive(Ridge(), lags=3), 'y'),
     (ForecasterRecursiveMultiSeries(Ridge(), lags=3), 'series'),
     (ForecasterDirectMultiVariate(Ridge(), level='l1', lags=3, steps=1), 'series')],
    ids=lambda x: f"{type(x).__name__ if not isinstance(x, str) else x}"
)
def test_train_test_split_one_step_ahead_search_output_equal_to_forecaster_split(
    forecaster, data_name
):
    """
    Test that the output is the split of the forecaster when the candidate
    can be evaluated.
    """
    if data_name == 'y':
        kwargs = {'y': series['l1']}
    elif type(forecaster).__name__ == 'ForecasterDirectMultiVariate':
        kwargs = {'series': pd.DataFrame({'l1': series['l1'], 'l2': series['l2']})}
    else:
        kwargs = {'series': series}

    split = _train_test_split_one_step_ahead_search(
                forecaster         = forecaster,
                cv                 = OneStepAheadFold(initial_train_size=45),
                initial_train_size = 45,
                levels             = ['l1'],
                **kwargs
            )
    expected = forecaster._train_test_split_one_step_ahead(
                   initial_train_size=45, **kwargs
               )

    assert len(split) == len(expected)
    np.testing.assert_array_equal(np.asarray(split[0]), np.asarray(expected[0]))
    np.testing.assert_array_equal(np.asarray(split[3]), np.asarray(expected[3]))


def test_train_test_split_one_step_ahead_search_ValueError_when_initial_train_size_not_greater_than_window_size():
    """
    Test ValueError is raised when the `window_size` of the candidate is not
    smaller than `initial_train_size`, with the date received.
    """
    forecaster = ForecasterRecursiveMultiSeries(Ridge(), lags=45)

    err_msg = re.escape(
        "`initial_train_size` must be greater than the `window_size` of the "
        "forecaster (45), so that the first training set has data to train with. "
        "Got '2020-02-14', 45 observations in the training set."
    )
    with pytest.raises(ValueError, match=err_msg):
        _train_test_split_one_step_ahead_search(
            forecaster         = forecaster,
            cv                 = OneStepAheadFold(initial_train_size='2020-02-14'),
            initial_train_size = 45,
            series             = series,
            levels             = ['l1']
        )


# NOTE: With `initial_train_size=45` and 8 lags, each of these series cannot be
# evaluated for a different cause.
gap = series['l1'].copy()
gap.iloc[42] = np.nan
series_causes = {
    'l1': series['l1'],
    'short': series['short'],
    'ended': series['l1'].iloc[:40],
    'gap': gap,
    'no_test': series['l1'].iloc[:45],
}


@pytest.mark.parametrize(
    "level, cause",
    [('short', "5 observations in the training set, not more than the "
               "`window_size` (8)"),
     ('ended', "it ends before the end of the training set, so it cannot be "
               "predicted after it"),
     ('gap', "missing values in its last window (the `window_size` "
             "observations before the test set)"),
     ('no_test', "no observed values in the test set")],
    ids=lambda x: x if x in series_causes else ''
)
def test_train_test_split_one_step_ahead_search_ValueError_when_no_level_with_points_in_test(
    level, cause
):
    """
    Test ValueError is raised when no level of `levels` can be evaluated in
    the test set, with the cause of the level: too few observations in the
    training set, a series that ends before the end of the training set,
    missing values in the last window (Ridge does not support them) or no
    observed values in the test set. The other series are trained.
    """
    forecaster = ForecasterRecursiveMultiSeries(Ridge(), lags=8)

    err_msg = re.escape(
        f"No level of `levels` can be evaluated in the test set with this "
        f"`window_size` (8): '{level}': {cause}. A level is evaluated if it has "
        f"more than `window_size` observations in the training set, its last "
        f"window can be used to predict, and it has observed values in the "
        f"test set. Review `levels`, `initial_train_size` of `cv` or the span "
        f"of the series."
    )
    with pytest.raises(ValueError, match=f"^{err_msg}$"):
        _train_test_split_one_step_ahead_search(
            forecaster         = forecaster,
            cv                 = OneStepAheadFold(initial_train_size=45),
            initial_train_size = 45,
            series             = series_causes,
            levels             = [level]
        )


def test_train_test_split_one_step_ahead_search_ValueError_max_levels():
    """
    Test that the error gives the cause of at most `max_levels` levels and
    counts the rest.
    """
    forecaster = ForecasterRecursiveMultiSeries(Ridge(), lags=8)

    err_msg = re.escape(
        "No level of `levels` can be evaluated in the test set with this "
        "`window_size` (8): 'short': 5 observations in the training set, not "
        "more than the `window_size` (8); 'no_test': no observed values in the "
        "test set; and 2 more levels. A level is evaluated if"
    )
    with pytest.raises(ValueError, match=err_msg):
        _train_test_split_one_step_ahead_search(
            forecaster         = forecaster,
            cv                 = OneStepAheadFold(initial_train_size=45),
            initial_train_size = 45,
            series             = series_causes,
            levels             = ['short', 'no_test', 'ended', 'gap'],
            max_levels         = 2
        )
