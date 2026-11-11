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
    quantiles: list[float] | tuple[float, ...] | None,
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


def _match_quantile_indices(
    quantile_grid: list[float],
    quantiles: list[float],
    tol: float = 1e-6,
) -> list[int]:
    """
    Map requested quantile levels to column indices in `quantile_grid`.

    Parameters
    ----------
    quantile_grid : list of float
        Quantile levels corresponding to the columns of the model's
        quantile output, in order (e.g. the quantile grid of the loaded
        checkpoint).
    quantiles : list of float
        Requested quantile levels.
    tol : float, default 1e-6
        Maximum allowed absolute difference between a requested
        quantile and its nearest match in `quantile_grid`.

    Returns
    -------
    indices : list of int
        Column index in `quantile_grid` for each entry in
        `quantiles`, in the same order.

    Notes
    -----
    A `ValueError` is raised if a requested quantile has no match
    within `tol`. Matching against the model's own quantile grid
    (synced from the loaded checkpoint) rather than a fixed formula
    keeps the mapping correct even if a checkpoint ships a different
    quantile grid.

    """

    indices = []
    for q in quantiles:
        diffs = [abs(g - q) for g in quantile_grid]
        best_idx = diffs.index(min(diffs))
        if diffs[best_idx] > tol:
            raise ValueError(
                f"Quantile {q} not found in the model's quantile grid "
                f"{quantile_grid} (tolerance {tol})."
            )
        indices.append(best_idx)

    return indices


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


# License of the weights served by every registered `model_id` prefix, as
# `(license, license_url, commercial_use_restricted, warning_name)`:
#
# - `license`: SPDX identifier, or the `license_name` of the HuggingFace model
#   card when the license is not a standard one.
# - `license_url`: link to the license file. `None` for standard licenses,
#   whose link is the model card of the repository the weights come from
#   (see `_get_license_url`), so it follows the `model_id`.
# - `warning_name`: descriptive name shown by `LicenseWarning`. `None` when
#   the license does not restrict commercial use, so no warning is raised.
#
# Every prefix in `_ADAPTER_REGISTRY` must be covered by an entry (checked by
# the tests), so adding an adapter forces an explicit decision about its
# license. The entries are compared with the HuggingFace model cards by
# `tools/check_foundation_models_metadata.py`, which runs on a schedule in CI
# because a provider can change its terms, or a backend switch to newer
# weights, at any time. A `model_id` not covered here is not known to carry a
# commercial-use restriction, it does not mean the license has been confirmed
# permissive.
_MODEL_LICENSES: dict[str, tuple[str, str | None, bool, str | None]] = {
    "amazon/chronos-2": (
        "Apache-2.0",
        None,
        False,
        None,
    ),
    "autogluon/chronos-2": (
        "Apache-2.0",
        None,
        False,
        None,
    ),
    "google/timesfm-2.5": (
        "Apache-2.0",
        None,
        False,
        None,
    ),
    "google/timesfm-3.0": (
        "timesfm-non-commercial-license-v1.0",
        "https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE",
        True,
        "TimesFM Non-Commercial License v1.0",
    ),
    "NX-AI/TiRex-2": (
        "Apache-2.0",
        None,
        False,
        None,
    ),
    "Salesforce/moirai": (
        "CC-BY-NC-4.0",
        None,
        True,
        "CC-BY-NC-4.0",
    ),
    "soda-inria/tabicl": (
        "BSD-3-Clause",
        None,
        False,
        None,
    ),
    # License of the TabPFN-3.5 weights, the version pinned by
    # tabpfn-time-series 1.3. See `TabPFNAdapter.weights_repo_id`.
    "priorlabs/tabpfn": (
        "tabpfn-3-5-license-v1.0",
        "https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE",
        True,
        "TabPFN-3.5 License v1.0 (non-commercial without an enterprise license)",
    ),
    "theforecastingcompany/t0": (
        "Apache-2.0",
        None,
        False,
        None,
    ),
    "Synthefy/Nori": (
        "Apache-2.0",
        None,
        False,
        None,
    ),
    "taharnbl/TS-ICL": (
        "tsicl-v1-license-v1.0",
        "https://huggingface.co/taharnbl/TS-ICL/blob/main/LICENSE",
        True,
        "tsicl-v1-license-v1.0 (non-commercial)",
    ),
    # The model cards declare `license: other` with no `license_name`, and
    # the four checkpoints publish the same LICENSE file.
    "mistralai/Timestral": (
        "other",
        "https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE",
        True,
        "Timestral Modified MIT License (no rights for companies with a monthly "
        "revenue above USD 20 million)",
    ),
}


def _get_model_license(
    model_id: str
) -> tuple[str, str | None, bool, str | None] | None:
    """
    Return the license registered for the weights of `model_id`, if any.

    Looks up `model_id` in `_MODEL_LICENSES` using longest-prefix matching.

    Parameters
    ----------
    model_id : str
        Model ID to look up.

    Returns
    -------
    license_info : tuple, None
        Tuple `(license, license_url, commercial_use_restricted,
        warning_name)`, or `None` when no registered prefix matches
        `model_id`.

    """

    best_prefix = None
    for prefix in _MODEL_LICENSES:
        if model_id.startswith(prefix):
            if best_prefix is None or len(prefix) > len(best_prefix):
                best_prefix = prefix

    if best_prefix is None:
        return None

    return _MODEL_LICENSES[best_prefix]


def _get_license_url(license_url: str | None, weights_repo_id: str) -> str:
    """
    Return the link to the terms of a registered license.

    Parameters
    ----------
    license_url : str, None
        Link registered in `_MODEL_LICENSES`. `None` for standard licenses.
    weights_repo_id : str
        Hugging Face repository the weights are downloaded from.

    Returns
    -------
    license_url : str
        `license_url` when it is registered, otherwise the model card of
        `weights_repo_id`.

    """

    if license_url is None:
        license_url = f"https://huggingface.co/{weights_repo_id}"

    return license_url


def _warn_if_non_commercial(model_id: str) -> None:
    """
    Warn when `model_id` matches a prefix known to carry a non-commercial
    license.

    Looks up `model_id` with `_get_model_license`. Model ids whose registered
    license does not restrict commercial use, or that do not match any
    registered prefix, do not raise a warning; the latter only means the id
    is not in the registry, it does not confirm that the license permits
    commercial use.

    Parameters
    ----------
    model_id : str
        Model ID whose weights are about to be loaded.

    Returns
    -------
    None

    """

    license_info = _get_model_license(model_id)
    if license_info is None or not license_info[2]:
        return

    _, license_url, _, license_name = license_info
    # The licenses that restrict commercial use and have no registered link
    # belong to backends that download the weights from `model_id`.
    license_url = _get_license_url(license_url, weights_repo_id=model_id)
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


def drop_past_only_exog(
    context_exog: dict[str, pd.DataFrame | None],
    exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    series_names_in: list[str],
) -> dict[str, pd.DataFrame | None]:
    """
    Remove, from the historical exogenous variables of each series, the
    columns that have no future values (past-only covariates).

    Parameters
    ----------
    context_exog : dict
        Per-series historical exogenous variables as returned by
        `align_context_exog`: each value is a pandas DataFrame or `None`.
    exog : dict, None
        Per-series future exogenous variables. Missing keys or `None` values
        mean the series has no future exog, so all its historical columns
        are removed.
    series_names_in : list
        Series to filter. Defines the keys of the output dict.

    Returns
    -------
    context_exog_filtered : dict
        Per-series dict with exactly the keys in `series_names_in`. Each
        value holds only the historical columns that are also in the future
        exog of the series, in their original order, or `None` if no column
        is left.

    Notes
    -----
    `FoundationModel.predict` applies it to adapters that declare
    `supports_past_only_covariates=False`, so that they never receive a
    covariate without future values. The input objects are not modified.

    """

    context_exog_filtered: dict[str, pd.DataFrame | None] = {}
    for series_name in series_names_in:
        series_exog = context_exog.get(series_name)
        if series_exog is None:
            context_exog_filtered[series_name] = None
            continue
        future_cols = _extract_exog_columns(
            exog.get(series_name) if exog is not None else None
        )
        cols_to_keep = [col for col in series_exog.columns if col in future_cols]
        if not cols_to_keep:
            context_exog_filtered[series_name] = None
        elif len(cols_to_keep) == len(series_exog.columns):
            context_exog_filtered[series_name] = series_exog
        else:
            context_exog_filtered[series_name] = series_exog[cols_to_keep]

    return context_exog_filtered


def _left_pad_context(
    context: dict[str, pd.Series],
    context_exog: dict[str, pd.DataFrame | None] | None,
) -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame | None] | None]:
    """
    Left-pad the context window of every series with NaN to the length of the
    longest one, so that all of them end at their own forecast origin.

    Parameters
    ----------
    context : dict
        Per-series context windows.
    context_exog : dict, None
        Per-series historical exogenous variables, aligned to the index of
        their context.

    Returns
    -------
    context : dict
        Per-series context windows, all of length `width` (the longest
        context) and indexed by `pandas.RangeIndex(width)`.
    context_exog : dict, None
        Per-series historical exogenous variables padded and indexed in the
        same way, or `None` if `context_exog` is `None`.

    Notes
    -----
    The padding is positional: the original timestamps are replaced by a
    `RangeIndex`, so the result does not depend on the frequency or on the
    time zone of each series (no timestamps are generated, which could not
    exist or be ambiguous around a daylight saving change). Adapters that
    declare `requires_equal_length_context` only use the values;
    `FoundationModel` builds the index of the predictions from the unpadded
    context.

    A padded context series with a NumPy dtype is returned as float (a float
    dtype is kept, any other becomes float64). Boolean exog columns (also the
    nullable `boolean` dtype) are cast to float in every series, because a
    bool column cannot hold NaN (pandas would turn it into an `object`
    column). NumPy integer columns become float only in the series that are
    padded, so within a group an integer column can be int or float depending
    on the series. Nullable extension dtypes (`Int64`, `Float64`), in the
    context and in the exog, keep their dtype and hold `pd.NA` in the padded
    rows. Adapters must only rely on whether a column is numeric. The other
    columns keep their dtype. The input objects are not modified.

    """

    width = max(len(series_values) for series_values in context.values())
    index = pd.RangeIndex(width)
    padded_context: dict[str, pd.Series] = {}
    padded_exog: dict[str, pd.DataFrame | None] | None = (
        {} if context_exog is not None else None
    )
    for series_name, series_values in context.items():
        n_pad = width - len(series_values)
        positions = pd.RangeIndex(n_pad, width)
        if n_pad > 0 and isinstance(series_values.dtype, np.dtype):
            # NumPy dtypes: fill an array by position, about 5 times faster
            # than `reindex`. A float dtype is kept, any other becomes float64.
            dtype = (
                series_values.dtype if series_values.dtype.kind == "f" else np.float64
            )
            values = np.full(width, np.nan, dtype=dtype)
            values[n_pad:] = series_values.to_numpy()
            padded_context[series_name] = pd.Series(
                values, index=index, name=series_values.name
            )
        else:
            series_values = series_values.set_axis(positions)
            padded_context[series_name] = (
                series_values.reindex(index) if n_pad > 0 else series_values
            )
        if padded_exog is not None:
            series_exog = context_exog.get(series_name)
            if series_exog is not None:
                series_exog = series_exog.set_axis(positions)
                for col, dtype in series_exog.dtypes.items():
                    if pd.api.types.is_bool_dtype(dtype):
                        series_exog[col] = series_exog[col].astype(float)
                if n_pad > 0:
                    series_exog = series_exog.reindex(index)
            padded_exog[series_name] = series_exog

    return padded_context, padded_exog
