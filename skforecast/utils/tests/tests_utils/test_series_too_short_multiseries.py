# Unit test _series_too_short_multiseries
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.utils import align_series_and_exog_multiseries
from skforecast.utils import _series_too_short_multiseries


def test_series_too_short_multiseries_output_after_aligning():
    """
    Test the series with `window_size` observations or fewer, counted after
    removing their leading and trailing NaNs with
    `align_series_and_exog_multiseries`, as in `fit`.
    """
    index = pd.RangeIndex(10)
    series_dict = {
        'long': pd.Series(np.arange(10, dtype=float), index=index),
        'equal': pd.Series(np.arange(4, dtype=float), index=index[:4]),
        'nan_ends': pd.Series([np.nan] * 3 + [1., 2., 3., 4., 5.] + [np.nan] * 2, index=index),
        'one_more': pd.Series(np.arange(5, dtype=float), index=index[5:]),
    }
    series_dict, _ = align_series_and_exog_multiseries(
                         series_dict = series_dict,
                         exog_dict   = {k: None for k in series_dict}
                     )

    results = _series_too_short_multiseries(series_dict=series_dict, window_size=4)

    assert results == ['equal']
    assert _series_too_short_multiseries(series_dict=series_dict, window_size=5) == [
        'equal', 'nan_ends', 'one_more'
    ]
