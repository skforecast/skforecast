# Unit test _drop_series_too_short_refit
# ==============================================================================
import re
import warnings
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection._utils import _drop_series_too_short_refit

index = pd.RangeIndex(30)
series = {
    'l1': pd.Series(np.arange(30, dtype=float), index=index),
    'short': pd.Series([np.nan] * 2 + list(np.arange(6.)) + [np.nan], index=index[:9]),
    'l2': pd.Series(np.arange(30, dtype=float), index=index),
}
exog = {k: pd.DataFrame({'x': np.arange(30.)}, index=index) for k in series}


def test_drop_series_too_short_refit_output_and_warning():
    """
    Test that the series with `window_size` observations or fewer, after
    removing their leading and trailing NaNs, are left out of `series` and of
    `exog` (dict), with a RuntimeWarning that lists them. The input is not
    modified.
    """
    forecaster = ForecasterRecursiveMultiSeries(Ridge(), lags=6)

    warn_msg = re.escape(
        "1 series is left out of the refit of the best candidate: it has "
        "`window_size` (6) observations or fewer, too few to train the "
        "forecaster, and the candidate did not predict it in the search either. "
        "The forecaster returned is not trained with it (not in "
        "`series_names_in_`).\n\nSeries left out: 'short'."
    )
    with pytest.warns(RuntimeWarning, match=f"^{warn_msg}$"):
        series_refit, exog_refit = _drop_series_too_short_refit(
                                       forecaster = forecaster,
                                       series     = series,
                                       exog       = exog
                                   )

    assert list(series_refit) == ['l1', 'l2']
    assert list(exog_refit) == ['l1', 'l2']
    assert list(series) == ['l1', 'short', 'l2']
    assert len(series['short']) == 9


@pytest.mark.parametrize(
    "forecaster",
    [ForecasterRecursiveMultiSeries(Ridge(), lags=5),
     ForecasterDirectMultiVariate(Ridge(), level='l1', lags=8, steps=1)],
    ids=lambda f: f"{type(f).__name__}"
)
def test_drop_series_too_short_refit_unchanged(forecaster):
    """
    Test that `series` and `exog` are returned unchanged, without warning,
    when no series is too short (6 values and 5 lags), and for forecasters
    other than ForecasterRecursiveMultiSeries.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        series_refit, exog_refit = _drop_series_too_short_refit(
                                       forecaster = forecaster,
                                       series     = series,
                                       exog       = exog
                                   )

    assert series_refit is series
    assert exog_refit is exog
