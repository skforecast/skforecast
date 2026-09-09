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
from ..exceptions import LicenseWarning, MissingValuesWarning


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


def _validate_model_id_prefix(model_id: str, prefix: str, adapter_name: str) -> None:
    """
    Validate that `model_id` starts with the prefix served by an adapter.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID to validate.
    prefix : str
        Prefix the adapter serves, e.g. `"google/timesfm-2.5"`.
    adapter_name : str
        Adapter class name, used in the raised error message.

    Returns
    -------
    None

    """

    if not isinstance(model_id, str) or not model_id.startswith(prefix):
        raise ValueError(
            f"`model_id` must start with {prefix!r} for {adapter_name}. "
            f"Got {model_id!r}."
        )


def _validate_supported_quantiles(
    quantiles: list[float] | tuple[float] | None,
    supported_quantiles: list[float],
    model_name: str,
    tol: float = 1e-9,
) -> list[float] | None:
    """
    Validate that every requested quantile level is one of the fixed levels
    supported by a backend.

    Parameters
    ----------
    quantiles : list, tuple, None
        Requested quantile levels. `None` means point forecast.
    supported_quantiles : list
        Fixed levels supported by the backend, e.g. the adapter's
        `SUPPORTED_QUANTILES`.
    model_name : str
        Backend name used in the raised error message, e.g. `"TimesFM"`.
    tol : float, default 1e-9
        Maximum absolute difference for a requested level to be considered
        equal to a supported one.

    Returns
    -------
    quantile_list : list, None
        `list(quantiles)`, or `None` if `quantiles` is `None`.

    """

    if quantiles is None:
        return None

    quantile_list = list(quantiles)
    for q in quantile_list:
        if not any(
            abs(q - supported_quantile) < tol
            for supported_quantile in supported_quantiles
        ):
            raise ValueError(
                f"{model_name} only supports quantile levels "
                f"{supported_quantiles}. Got {q!r}. "
                f"Quantile interpolation is not supported."
            )

    return quantile_list


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


# License terms last verified against the HuggingFace model card on
# 2026-09-07. This table is not checked automatically against the source, so
# it can go stale silently if a provider changes its license terms; re-verify
# periodically. A `model_id` not listed here is not known to carry a
# commercial-use restriction, it does not mean the license has been confirmed
# permissive.
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
    Warn when `model_id` matches a prefix known to carry a non-commercial
    license.

    Looks up `model_id` in `_NON_COMMERCIAL_LICENSES` using longest-prefix
    matching. Model ids that do not match any registered prefix do not raise
    a warning; this only means the id is not in this registry, it does not
    confirm that the license permits commercial use.

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
        f"The weights for '{model_id}' are released under {license_name}. "
        "Review the license terms before commercial or production use. "
        f"See {license_url}.",
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
        series_name = series.name if series.name is not None else 'y'
        series = {series_name: series.rename(series_name)}

    return check_preprocess_series(series)


def _extract_exog_columns(
    data: pd.DataFrame | pd.Series | None
) -> set:
    """
    Return the set of column names of an exog entry.

    Parameters
    ----------
    data : pandas DataFrame, pandas Series, None
        Exogenous variables of one series. A `pandas Series` contributes its
        `.name`; `None` contributes no columns.

    Returns
    -------
    columns : set
        Column names.

    """

    if data is None:
        return set()
    if isinstance(data, pd.Series):
        return {data.name}

    return set(data.columns)


def get_exog_signature(
    context_exog: pd.DataFrame | pd.Series | None,
    exog: pd.DataFrame | pd.Series | None,
) -> tuple[tuple, tuple]:
    """
    Return the covariate signature `(past_only_cols, fut_cols)` of one
    series.

    Parameters
    ----------
    context_exog : pandas DataFrame, pandas Series, None
        Historical exogenous variables aligned to the context of the series.
    exog : pandas DataFrame, pandas Series, None
        Future-known exogenous variables covering the forecast horizon of
        the series.

    Returns
    -------
    past_only_cols : tuple
        Columns present only in `context_exog`, sorted.
    fut_cols : tuple
        Columns present in `exog`, sorted.

    Notes
    -----
    Both tuples are sorted so that series with the same columns produce the
    same, hashable signature regardless of column order, and so that adapters
    build their covariate arrays in the same column order for every series of
    a group. `((), ())` means the series has no covariates.

    """

    ctx_cols = _extract_exog_columns(context_exog)
    fut_cols = _extract_exog_columns(exog)

    return (
        tuple(sorted(ctx_cols - fut_cols, key=str)),
        tuple(sorted(fut_cols, key=str)),
    )


def group_series_by_exog_signature(
    series_names_in: list[str],
    context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    exog: dict[str, pd.DataFrame | pd.Series | None] | None,
) -> list[list[str]]:
    """
    Group series that share the same covariate signature.

    Parameters
    ----------
    series_names_in : list
        Names of the series, in the order they will be predicted.
    context_exog : dict, None
        Per-series historical exogenous variables. Missing keys or `None`
        values mean the series has no historical exog.
    exog : dict, None
        Per-series future exogenous variables. Missing keys or `None` values
        mean the series has no future exog.

    Returns
    -------
    series_groups : list
        One list of series names per distinct signature, as returned by
        `get_exog_signature`. Groups appear in the order their first series
        appears in `series_names_in`; within a group, names keep the order of
        `series_names_in`.

    """

    groups: dict[tuple[tuple, tuple], list[str]] = {}
    for series_name in series_names_in:
        signature = get_exog_signature(
            context_exog = (
                context_exog.get(series_name) if context_exog is not None else None
            ),
            exog         = exog.get(series_name) if exog is not None else None,
        )
        groups.setdefault(signature, []).append(series_name)

    return list(groups.values())


def align_context_exog(
    context: dict[str, pd.Series],
    context_exog: dict[str, pd.DataFrame | pd.Series | None],
    series_names_in: list[str],
) -> dict[str, pd.DataFrame | None]:
    """
    Align the historical exogenous variables of each series to the index of
    its context.

    Parameters
    ----------
    context : dict
        Per-series context windows.
    context_exog : dict
        Per-series historical exogenous variables. Missing keys or `None`
        values mean the series has no historical exog.
    series_names_in : list
        Series to align. Defines the keys of the output dict.

    Returns
    -------
    context_exog_aligned : dict
        Per-series dict with exactly the keys in `series_names_in`. Each
        non-None value is a pandas DataFrame with the same index as
        `context[series_name]`. Series inputs are coerced to single-column
        DataFrames.

    Notes
    -----
    Rows of `context_exog` outside the context index are dropped. Context
    timestamps missing from `context_exog` are added with NaN and reported
    once with a `MissingValuesWarning`. No columns are ever added or removed.

    """

    context_exog_aligned: dict[str, pd.DataFrame | None] = {}
    nan_filled_series = []
    for series_name in series_names_in:
        series_exog = context_exog.get(series_name)
        if series_exog is None:
            context_exog_aligned[series_name] = None
            continue
        if isinstance(series_exog, pd.Series):
            series_exog = series_exog.to_frame()
        ctx_index = context[series_name].index
        if series_exog.index.equals(ctx_index):
            # Fast path: exog already aligned, no reindex needed.
            context_exog_aligned[series_name] = series_exog
            continue
        if not ctx_index.isin(series_exog.index).all():
            nan_filled_series.append(series_name)
        context_exog_aligned[series_name] = series_exog.reindex(ctx_index)

    if nan_filled_series:
        warnings.warn(
            f"`context_exog` for series {nan_filled_series} has been reindexed "
            f"to match the index of `context`. Missing timestamps were filled "
            f"with NaN.",
            MissingValuesWarning,
            stacklevel=4,
        )

    return context_exog_aligned
