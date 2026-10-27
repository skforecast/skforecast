# Unit test _all_samples_removed_message
# ==============================================================================
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skforecast.utils import _all_samples_removed_message


def test_all_samples_removed_message_when_dropna_from_series_False():
    """
    Test that only the target is given as the cause when `dropna_from_series`
    is `False`, since the rows with NaN in the predictors are not removed.
    """
    message = _all_samples_removed_message(
                  series_name              = "`y`",
                  dropna_from_series       = False,
                  max_lag                  = 3,
                  max_size_window_features = 5,
                  estimator                = HistGradientBoostingRegressor()
              )

    expected = (
        "All samples have been removed due to NaNs: the target is NaN in "
        "every row. Review `y` or fill its missing values."
    )

    assert message == expected


@pytest.mark.parametrize(
    'series_name, max_lag, max_size_window_features, expected',
    [
        ("`y`", 12, None,
         "All samples have been removed due to NaNs: every row has a missing "
         "value in the target or in the predictors (largest lag: 12). This can "
         "happen when the missing values of `y` are closer together than the "
         "lags, or when `exog` has a missing value in every row. To keep "
         "training samples, fill the missing values of `y` and `exog`, use "
         "fewer lags, or use an estimator that accepts NaNs (for example, "
         "LightGBM or scikit-learn HistGradientBoosting models) and set "
         "`dropna_from_series = False`."),
        ("`series`", None, 7,
         "All samples have been removed due to NaNs: every row has a missing "
         "value in the target or in the predictors (largest window feature: "
         "7). This can happen when the missing values of `series` are closer "
         "together than the window features, or when `exog` has a missing "
         "value in every row. To keep training samples, fill the missing "
         "values of `series` and `exog`, use shorter window features (or a "
         "smaller `min_periods` if they are `RollingFeatures`), or use an "
         "estimator that accepts NaNs (for example, LightGBM or scikit-learn "
         "HistGradientBoosting models) and set `dropna_from_series = False`."),
    ],
    ids=['lags', 'window_features']
)
def test_all_samples_removed_message_when_estimator_without_nan_support(
    series_name, max_lag, max_size_window_features, expected
):
    """
    Test the message when `dropna_from_series` is `True` and the estimator
    does not support NaN values, with lags only and with window features only.
    An estimator that accepts NaNs is suggested and the size of the largest
    lag or window feature is given.
    """
    message = _all_samples_removed_message(
                  series_name              = series_name,
                  dropna_from_series       = True,
                  max_lag                  = max_lag,
                  max_size_window_features = max_size_window_features,
                  estimator                = LinearRegression()
              )

    assert message == expected


@pytest.mark.parametrize(
    'estimator',
    [HistGradientBoostingRegressor(),
     make_pipeline(StandardScaler(), HistGradientBoostingRegressor())],
    ids=lambda est: f'estimator: {type(est).__name__}'
)
def test_all_samples_removed_message_when_estimator_with_nan_support(estimator):
    """
    Test that `dropna_from_series = False` is suggested directly when the
    estimator, or the last step of a Pipeline, natively supports NaN values.
    """
    message = _all_samples_removed_message(
                  series_name              = "`y`",
                  dropna_from_series       = True,
                  max_lag                  = 12,
                  max_size_window_features = 20,
                  estimator                = estimator
              )

    expected = (
        "All samples have been removed due to NaNs: every row has a missing "
        "value in the target or in the predictors (largest lag: 12, largest "
        "window feature: 20). This can happen when the missing values of `y` "
        "are closer together than the lags or window features, or when `exog` "
        "has a missing value in every row. To keep training samples, set "
        "`dropna_from_series = False` (the estimator accepts NaNs), fill the "
        "missing values of `y` and `exog`, or use fewer lags or shorter window "
        "features (or a smaller `min_periods` if they are `RollingFeatures`)."
    )

    assert message == expected
