################################################################################
#                         skforecast.foundation._utils                         #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################


from __future__ import annotations
from typing import Any, Callable
import warnings
import numpy as np
import pandas as pd
from ..utils import check_preprocess_series
from ..exceptions import LicenseWarning


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


def _apply_set_params(
    instance: Any,
    params: dict[str, Any],
    *,
    validate: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    resets: tuple[tuple[set[str], Callable[[], None]], ...] = (),
) -> Any:
    """
    Shared `set_params` skeleton for the foundation-model adapters.

    Rejects keys not present in `instance.get_params()`, validates and
    normalizes the values through the adapter-provided `validate` callback,
    applies only the values that actually change, and invalidates the cached
    artifacts whose trigger keys changed. Value validation is left to each
    adapter (via `validate`) because it is model specific; only the mechanical
    key check, compare-and-reset, and assignment are shared here.

    Parameters
    ----------
    instance : object
        The adapter whose parameters are being set. Its `get_params` keys
        define the set of valid parameters.
    params : dict
        Parameters to set.
    validate : callable, default None
        Callback that receives the parameters (already checked for unknown
        keys) and returns them validated and normalized, raising `ValueError`
        on invalid values. If `None`, the parameters are applied verbatim.
    resets : tuple of (set, callable), default ()
        Each entry pairs a set of trigger keys with a reset callback. A
        callback is invoked once when at least one of its trigger keys is
        among the parameters that actually changed.

    Returns
    -------
    instance : object
        The same adapter, to allow chaining.

    """

    valid = set(instance.get_params())
    invalid = set(params) - valid
    if invalid:
        raise ValueError(
            f"Invalid parameter(s) for {type(instance).__name__}: {sorted(invalid)}. "
            f"Valid parameters are: {sorted(valid)}."
        )

    if validate is not None:
        params = validate(params)

    changed = {
        key: value
        for key, value in params.items()
        if getattr(instance, key) != value
    }
    if changed:
        for trigger_keys, reset in resets:
            if changed.keys() & trigger_keys:
                reset()
        for key, value in changed.items():
            setattr(instance, key, value)

    return instance


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


_NON_COMMERCIAL_LICENSES: dict[str, tuple[str, str]] = {
    "google/timesfm-3.0": (
        "TimesFM Non-Commercial License v1.0",
        "https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE",
    ),
    "Salesforce/moirai": (
        "CC-BY-NC-4.0",
        "https://huggingface.co/Salesforce/moirai-2.0-R-small",
    ),
    "priorlabs/tabpfn": (
        "TabPFN License v1.0 (non-commercial without an enterprise license)",
        "https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/LICENSE",
    ),
    "taharnbl/TS-ICL": (
        "tsicl-v1-license-v1.0 (non-commercial)",
        "https://huggingface.co/taharnbl/TS-ICL",
    ),
}


def _warn_if_non_commercial(model_id: str) -> None:
    """
    Warn when `model_id` resolves to weights released under a non-commercial
    license.

    Looks up `model_id` in `_NON_COMMERCIAL_LICENSES` using longest-prefix
    matching. Model ids that do not match any registered prefix are assumed
    to be unrestricted and no warning is raised.

    Parameters
    ----------
    model_id : str
        Model ID whose weights are about to be loaded.

    Returns
    -------
    None

    """

    best_prefix = None
    for prefix in _NON_COMMERCIAL_LICENSES:
        if model_id.startswith(prefix):
            if best_prefix is None or len(prefix) > len(best_prefix):
                best_prefix = prefix

    if best_prefix is None:
        return

    license_name, license_url = _NON_COMMERCIAL_LICENSES[best_prefix]
    warnings.warn(
        f"The weights for '{model_id}' are released under {license_name}, "
        "which restricts their use to non-commercial or non-production "
        f"purposes. Review the license before deploying. See {license_url}.",
        category=LicenseWarning,
        stacklevel=3,
    )


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
