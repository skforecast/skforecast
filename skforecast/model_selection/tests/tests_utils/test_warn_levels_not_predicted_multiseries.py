# Unit test _warn_levels_not_predicted_multiseries
# ==============================================================================
import re
import warnings
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.exceptions import MissingValuesWarning
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection._utils import _warn_levels_not_predicted_multiseries

# Fixtures
# 'l3' has no observed values in the test set of fold 0.
series_dict_range = {
    'l1': pd.Series(np.arange(20, dtype=float)),
    'l2': pd.Series(np.arange(20, 40, dtype=float)),
    'l3': pd.Series(np.r_[np.full(16, np.nan), np.arange(4, dtype=float)]),
}
span_index = pd.RangeIndex(20)
# Train, last_window, test, test_no_gap, fit_forecaster
folds = [
    [0, [0, 12], [9, 12], [12, 16], [12, 16], True],
    [1, [0, 16], [13, 16], [16, 20], [16, 20], True]
]


def test_warn_levels_not_predicted_multiseries_no_warning_when_not_predicted_levels_have_no_test_values():
    """
    Test no warning is raised when every level is predicted, or the level not
    predicted in a fold has no observed values in its test set.
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=3)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _warn_levels_not_predicted_multiseries(
            forecaster           = forecaster,
            series               = series_dict_range,
            folds                = folds,
            span_index           = span_index,
            levels               = ['l1', 'l2', 'l3'],
            levels_predict_folds = [['l1', 'l2'], ['l1', 'l2', 'l3']],
            levels_train_folds   = [['l1', 'l2'], ['l1', 'l2', 'l3']]
        )


@pytest.mark.parametrize(
    'differentiation, window_size, reason_last_window',
    [(None, 3, 'Ridge does not support them'),
     (1, 4, 'they are not allowed with differentiation')],
    ids=lambda dt: f'differentiation, window_size, reason_last_window: {dt}'
)
def test_warn_levels_not_predicted_multiseries_MissingValuesWarning_when_levels_not_predicted(
    differentiation, window_size, reason_last_window
):
    """
    Test MissingValuesWarning is raised with the levels not predicted in folds
    where they have observed values, grouped by reason. 'l2' is in the training
    set of fold 1 but not predicted (missing values in the last window), 'l3'
    is not in the training set of fold 1, so it is never predicted.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator       = Ridge(),
                     lags            = 3,
                     differentiation = differentiation
                 )

    warn_msg = re.escape(
        "Some levels are not predicted in folds where they have observed values, "
        "so their metrics, and the aggregated metrics, do not include those "
        "folds. Levels never predicted have NaN metrics and are left out of the "
        "aggregated metrics.\n"
        f"  - `window_size` ({window_size}) observations or fewer in the "
        "training set: 'l3' (never predicted).\n"
        f"  - Missing values in the last window, {reason_last_window}: "
        "'l2' (folds 1)."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        _warn_levels_not_predicted_multiseries(
            forecaster           = forecaster,
            series               = series_dict_range,
            folds                = folds,
            span_index           = span_index,
            levels               = ['l1', 'l2', 'l3'],
            levels_predict_folds = [['l1', 'l2'], ['l1']],
            levels_train_folds   = [['l1', 'l2'], ['l1', 'l2']]
        )


def test_warn_levels_not_predicted_multiseries_MissingValuesWarning_last_window_when_levels_train_folds_None():
    """
    Test every level not predicted is attributed to the last window when
    `levels_train_folds` is None (the training set of the folds is not used).
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=3)

    warn_msg = re.escape(
        "  - Missing values in the last window, Ridge does not support them: "
        "'l1' (folds 0), 'l3' (never predicted)."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        _warn_levels_not_predicted_multiseries(
            forecaster           = forecaster,
            series               = series_dict_range,
            folds                = folds,
            span_index           = span_index,
            levels               = ['l1', 'l2', 'l3'],
            levels_predict_folds = [['l2'], ['l1', 'l2']]
        )


def test_warn_levels_not_predicted_multiseries_MissingValuesWarning_lists_skipped_folds():
    """
    Test the folds in which no level is predicted are listed as skipped when a
    level has observed values in their test set (fold 0), and that no warning
    is raised when the skipped fold has nothing to evaluate ('l3' has no
    observed values in the test set of fold 0).
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=3)

    warn_msg = re.escape(
        "  - `window_size` (3) observations or fewer in the training set: "
        "'l1' (folds 0), 'l2' (folds 0).\n"
        "  - Folds skipped, no level can be predicted in them: 0."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        _warn_levels_not_predicted_multiseries(
            forecaster           = forecaster,
            series               = series_dict_range,
            folds                = folds,
            span_index           = span_index,
            levels               = ['l1', 'l2', 'l3'],
            levels_predict_folds = [[], ['l1', 'l2', 'l3']],
            levels_train_folds   = [[], ['l1', 'l2', 'l3']]
        )

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _warn_levels_not_predicted_multiseries(
            forecaster           = forecaster,
            series               = series_dict_range,
            folds                = folds,
            span_index           = span_index,
            levels               = ['l3'],
            levels_predict_folds = [[], ['l3']],
            levels_train_folds   = [[], ['l3']]
        )
