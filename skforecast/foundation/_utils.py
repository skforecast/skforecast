################################################################################
#                         skforecast.foundation._utils                         #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import pandas as pd
from ..utils import check_preprocess_series


def check_preprocess_series_foundation(
    series: pd.Series | pd.DataFrame | dict[str, pd.Series],
) -> tuple[dict[str, pd.Series], dict[str, pd.Index]]:
    """
    Normalize and validate any supported series format to
    `dict[str, pandas Series]`.

    A `pandas Series` is wrapped in a one-element dict keyed by its
    `.name` (defaulting to `'y'`) before being passed to
    `check_preprocess_series`. All other types are forwarded
    directly.

    Parameters
    ----------
    series : pandas Series, pandas DataFrame, dict
        Input to normalize and validate.

    Returns
    -------
    series_dict : dict
        Normalized and validated series.
    series_indexes : dict
        Index of each series.
    
    """

    if isinstance(series, pd.Series):
        name = series.name if series.name is not None else 'y'
        series = {name: series.rename(name)}

    return check_preprocess_series(series)
