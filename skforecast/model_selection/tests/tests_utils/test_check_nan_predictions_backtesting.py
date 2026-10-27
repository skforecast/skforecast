# Unit test _check_nan_predictions_backtesting
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.model_selection._utils import _check_nan_predictions_backtesting

# Fixtures
index = pd.date_range(start='2020-01-01', periods=5, freq='D')
y_true = pd.Series([1., 2., 3., np.nan, 5.], index=index)


@pytest.mark.parametrize(
    "estimator_id, of_estimator",
    [(None, ""), ('arima', " of estimator 'arima'")],
    ids=lambda x: f"estimator_id: {x}"
)
def test_check_nan_predictions_backtesting_ValueError_when_NaN_predictions_at_observed_dates(
    estimator_id, of_estimator
):
    """
    Test ValueError is raised, with the number of NaN predictions, the first
    date, the likely cause and the remedy, when some predictions are NaN at
    dates with observed values.
    """
    y_pred = pd.Series([1., np.nan, np.nan, 4., 5.], index=index)

    err_msg = re.escape(
        f"2 backtesting predictions{of_estimator} are NaN at dates with "
        f"observed values (first: 2020-01-02 00:00:00), so the metrics cannot "
        f"be calculated. The most likely cause is a missing value in the "
        f"series or in `exog`, used as a predictor of these dates, with an "
        f"estimator that does not accept NaN (for example, a linear model). "
        f"Fill the missing values, or use an estimator that accepts them, for "
        f"example HistGradientBoostingRegressor or LGBMRegressor."
    )
    with pytest.raises(ValueError, match=err_msg):
        _check_nan_predictions_backtesting(
            y_true=y_true, y_pred=y_pred, estimator_id=estimator_id
        )


def test_check_nan_predictions_backtesting_no_error_when_NaN_predictions_at_dates_without_observed_values():
    """
    Test no error is raised when the only NaN prediction is at a date without
    observed value, or when there are no NaN predictions.
    """
    y_pred_nan = pd.Series([1., 2., 3., np.nan, 5.], index=index)
    y_pred = pd.Series([1., 2., 3., 4., 5.], index=index)

    _check_nan_predictions_backtesting(y_true=y_true, y_pred=y_pred_nan)
    _check_nan_predictions_backtesting(y_true=y_true, y_pred=y_pred)
