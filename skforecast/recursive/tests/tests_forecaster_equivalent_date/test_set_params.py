# Unit test set_params ForecasterEquivalentDate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd

from skforecast.exceptions import IgnoredArgumentWarning
from skforecast.recursive import ForecasterEquivalentDate

# Fixtures
from .fixtures_forecaster_equivalent_date import y


def test_set_params_updates_attributes_and_window_size():
    """
    Test set_params updates offset, n_offsets and agg_func and recomputes
    window_size for an integer offset.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    forecaster.set_params({'offset': 7, 'n_offsets': 2, 'agg_func': np.median})

    assert forecaster.offset == 7
    assert forecaster.n_offsets == 2
    assert forecaster.agg_func is np.median
    assert forecaster.window_size == 14


def test_set_params_window_size_when_offset_is_date_offset():
    """
    Test set_params recomputes window_size when the new offset is a pandas
    DateOffset, including when the previous offset was an integer and `fit`
    had already converted window_size into a number of steps.
    """
    forecaster = ForecasterEquivalentDate(offset=12, n_offsets=1)
    forecaster.fit(y=y)
    assert forecaster.window_size == 12

    forecaster.set_params(
        {'offset': pd.DateOffset(months=12), 'n_offsets': 2, 'agg_func': np.mean}
    )

    assert forecaster.offset == pd.DateOffset(months=12)
    assert forecaster.n_offsets == 2
    assert forecaster.agg_func is np.mean
    assert forecaster.window_size == pd.DateOffset(months=12) * 2


def test_set_params_resets_is_fitted():
    """
    Test set_params resets the forecaster to an unfitted state.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    forecaster.fit(y=y)
    assert forecaster.is_fitted

    forecaster.set_params({'offset': 7})
    assert not forecaster.is_fitted


def test_set_params_IgnoredArgumentWarning_when_unknown_key():
    """
    Test set_params warns and ignores unknown parameters.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)

    warn_msg = re.escape(
        "Unknown parameters ['lags'] will be ignored. "
        "Valid parameters are ['agg_func', 'n_offsets', 'offset']."
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        forecaster.set_params({'offset': 7, 'lags': 3})

    assert forecaster.offset == 7
    assert not hasattr(forecaster, 'lags')
