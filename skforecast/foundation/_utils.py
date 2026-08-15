################################################################################
#                         skforecast.foundation._utils                         #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################


from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from ..utils import check_preprocess_series


def _validate_positive_int(name: str, value: Any) -> None:
    """
    Validate that a parameter is a positive integer.

    Parameters
    ----------
    name : str
        Parameter name, used in the raised error message.
    value : Any
        Value to validate.

    Returns
    -------
    None

    """

    if not isinstance(value, int) or value < 1:
        raise ValueError(f"`{name}` must be a positive integer. Got {value!r}.")


def _tensor_to_numpy(values: Any) -> np.ndarray:
    """
    Detach a torch tensor to a numpy array, preserving its native dtype.

    Parameters
    ----------
    values : array-like
        Model output, either a numpy array or a torch tensor.

    Returns
    -------
    array : numpy ndarray
        Numpy array. Torch tensors are detached, moved to CPU, and
        converted, keeping their native dtype.

    """

    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()

    return np.asarray(values)


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
