################################################################################
#                     skforecast.foundational._utils                           #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import warnings
import pandas as pd
from ..exceptions import IgnoredArgumentWarning, InputTypeWarning, MissingExogWarning, MissingValuesWarning


def check_preprocess_series_type(
    series: pd.Series | pd.DataFrame | dict,
) -> tuple[bool, list[str], pd.Series | pd.DataFrame | dict]:
    """
    Validate `series`, normalise long-format DataFrames, and return
    `(is_multiseries, series_names, normalised_series)`.

    Long-format DataFrames (MultiIndex with series IDs in the first level and
    a `DatetimeIndex` in the second level) are converted to a
    `dict[str, pd.Series]` before being returned. All other supported types
    are returned unchanged.

    Parameters
    ----------
    series : pd.Series, pd.DataFrame, or dict of pd.Series
        Input to validate and normalise.

    Returns
    -------
    is_multiseries : bool
    series_names : list of str
    series : pd.Series, pd.DataFrame, or dict of pd.Series
        Normalised series. Guaranteed NOT to be a long-format DataFrame.

    Raises
    ------
    TypeError
        If `series` is a long-format DataFrame whose second MultiIndex level is
        not a `pandas.DatetimeIndex`, or if `series` is an unsupported type.
    """
    if isinstance(series, pd.Series):
        name = series.name if series.name is not None else 'y'
        return False, [name], series

    elif isinstance(series, pd.DataFrame):
        if not isinstance(series.index, pd.MultiIndex):
            # Wide-format DataFrame — each column is one series.
            return True, series.columns.tolist(), series

        # Long-format DataFrame: first level = series IDs, second = timestamps.
        if not isinstance(series.index.levels[1], pd.DatetimeIndex):
            raise TypeError(
                "The second level of the MultiIndex in `series` must be a "
                "pandas DatetimeIndex with the same frequency for each series. "
                f"Found {type(series.index.levels[1])}."
            )

        first_col = series.columns[0]
        if len(series.columns) != 1:
            warnings.warn(
                f"`series` DataFrame has multiple columns. Only the values of "
                f"the first column, '{first_col}', will be used as series "
                f"values. All other columns will be ignored.",
                IgnoredArgumentWarning,
                stacklevel=3,
            )

        series_dict = {
            sid: group[first_col].droplevel(0).rename(sid)
            for sid, group in series.groupby(level=0, sort=False)
        }
        series_ids = list(series_dict.keys())

        warnings.warn(
            "Passing a DataFrame (either wide or long format) as `series` "
            "requires additional internal transformations, which can increase "
            "computational time. It is recommended to use a dictionary of "
            "pandas Series instead. For more details, see: "
            "https://skforecast.org/latest/user_guides/"
            "independent-multi-time-series-forecasting.html#input-data",
            InputTypeWarning,
            stacklevel=3,
        )

        return True, series_ids, series_dict

    elif isinstance(series, dict):
        return True, list(series.keys()), series

    else:
        raise TypeError(
            "`series` must be a pandas Series, a wide pandas DataFrame, a "
            "long-format pandas DataFrame (MultiIndex), or a "
            f"dict[str, pd.Series]. Got {type(series)}."
        )


def check_preprocess_exog_type(
    exog: pd.Series | pd.DataFrame | dict | None,
    series_names_in_: list[str] | None = None,
) -> pd.Series | pd.DataFrame | dict | None:
    """
    Validate `exog` and normalise long-format MultiIndex DataFrames.

    Flat-index `pd.Series` and wide `pd.DataFrame` are returned
    unchanged — they are broadcast to all series by the adapter.  A
    `dict` is returned unchanged.  A long-format `pd.Series` or
    `pd.DataFrame` (MultiIndex, first level = series IDs, second level
    = `DatetimeIndex`) is converted to a `dict[str, pd.DataFrame]`
    and an `InputTypeWarning` is issued.

    Parameters
    ----------
    exog : pd.Series, pd.DataFrame, dict, or None
        Input to validate and normalise.
    series_names_in_ : list of str, optional
        Series names seen at fit time. When provided, a
        `MissingExogWarning` is issued for any series that has no
        corresponding entry in a long-format `exog`.

    Returns
    -------
    pd.Series, pd.DataFrame, dict, or None
        Normalised exog. Guaranteed NOT to be a long-format DataFrame.

    Raises
    ------
    TypeError
        If `exog` is a long-format DataFrame whose second MultiIndex level
        is not a `pandas.DatetimeIndex`, or if `exog` is an unsupported
        type.
    """
    if exog is None or isinstance(exog, dict):
        return exog

    # Coerce a MultiIndex pd.Series to pd.DataFrame so it follows the same
    # long-format path as a MultiIndex DataFrame.
    if isinstance(exog, pd.Series):
        if not isinstance(exog.index, pd.MultiIndex):
            return exog  # flat-index pd.Series — broadcast unchanged
        exog = exog.to_frame()  # fall through to the DataFrame path below

    if isinstance(exog, pd.DataFrame):
        if not isinstance(exog.index, pd.MultiIndex):
            # Wide-format DataFrame — broadcast to all series unchanged.
            return exog

        # Long-format: first level = series IDs, second = timestamps.
        if not isinstance(exog.index.levels[1], pd.DatetimeIndex):
            raise TypeError(
                "The second level of the MultiIndex in `exog` must be a "
                "pandas DatetimeIndex. "
                f"Found {type(exog.index.levels[1])}."
            )

        exog_dict = {
            sid: group.droplevel(0)
            for sid, group in exog.groupby(level=0, sort=False)
        }

        warnings.warn(
            "Passing a long-format DataFrame as `exog` requires additional "
            "internal transformations, which can increase computational time. "
            "It is recommended to use a dictionary of pandas Series or "
            "DataFrames instead. For more details, see: "
            "https://skforecast.org/latest/user_guides/"
            "independent-multi-time-series-forecasting.html#input-data",
            InputTypeWarning,
            stacklevel=3,
        )

        if series_names_in_ is not None:
            missing = [n for n in series_names_in_ if n not in exog_dict]
            if missing:
                warnings.warn(
                    f"The following series are present in `series_names_in_` "
                    f"but have no entry in the long-format `exog`: {missing}. "
                    f"No exogenous variables will be used for these series.",
                    MissingExogWarning,
                    stacklevel=3,
                )

        return exog_dict

    raise TypeError(
        "`exog` must be a pandas Series, a pandas DataFrame, or a "
        f"dict[str, pd.Series | pd.DataFrame | None]. Got {type(exog)}."
    )


def col_names(e: pd.Series | pd.DataFrame) -> list[str]:
    """Return column names of a DataFrame, or the name of a Series, as a list."""
    return e.columns.to_list() if isinstance(e, pd.DataFrame) else [e.name]


def assert_aligned(
    target: pd.Series | pd.DataFrame,
    ref: pd.Series | pd.DataFrame,
    target_label: str,
    ref_label: str,
) -> None:
    """
    Assert that `target` and `ref` have the same length and, when `ref` uses
    a `DatetimeIndex`, that their index values are identical.

    Raises
    ------
    ValueError
        If lengths differ, or if both have a DatetimeIndex that does not match.
    """
    if len(target) != len(ref):
        raise ValueError(
            f"{target_label} must have the same number of observations as "
            f"{ref_label}. "
            f"Got len({target_label}) = {len(target)}, "
            f"len({ref_label}) = {len(ref)}."
        )
    if (
        isinstance(ref.index, pd.DatetimeIndex)
        and not target.index.equals(ref.index)
    ):
        raise ValueError(
            f"The index of {target_label} must be aligned with the index of "
            f"{ref_label}. "
            f"{ref_label} has index from {ref.index[0]} to {ref.index[-1]}. "
            f"{target_label} has index from {target.index[0]} to {target.index[-1]}."
        )


def validate_exog_fit(
    series: pd.Series | pd.DataFrame | dict[str, pd.Series],
    exog: (
        pd.Series
        | pd.DataFrame
        | dict[str, pd.Series | pd.DataFrame | None]
        | None
    ),
    is_multiseries: bool,
) -> dict[str, list[str] | None]:
    """
    Validate `exog` against `series` at fit time and return a per-series
    column-name mapping.

    For single-series mode checks that the length of `exog` matches `series`
    and, when both have a `DatetimeIndex`, that their indices are identical.
    For multi-series mode these checks are applied independently to each
    (series, exog) pair. A `MissingExogWarning` is issued for any series
    in `series` that has no corresponding entry in a dict `exog`.

    Parameters
    ----------
    series : pd.Series, pd.DataFrame, or dict of pd.Series
        Training time series. `pd.DataFrame` is treated as a wide-format
        multi-series input (one column per series).
    exog : pd.Series, pd.DataFrame, dict, or None
        Historical exogenous variables. `None` is accepted and returns a
        mapping of all series names to `None`.
    is_multiseries : bool
        Whether the forecaster is operating in multi-series mode.

    Returns
    -------
    dict[str, list[str] | None]
        Mapping of series name → list of exog column names seen at fit time,
        or `None` if no exog was provided for that series.

    Raises
    ------
    ValueError
        If the length of `exog` (or any per-series value) does not match the
        corresponding series length, or if their `DatetimeIndex` values differ.
    """

    if not is_multiseries:
        series_name = series.name if series.name is not None else 'y'
        if exog is None:
            return {series_name: None}

        assert_aligned(exog, series, "`exog`", "`series`")
        return {series_name: col_names(exog)}

    else:
        # Normalise series to dict[str, pd.Series]
        if isinstance(series, pd.DataFrame):
            series_dict: dict[str, pd.Series] = {
                col: series[col] for col in series.columns
            }
        else:
            series_dict = series

        if exog is None:
            return {name: None for name in series_dict}

        # Normalise exog: broadcast (pd.Series / pd.DataFrame) → expand to dict
        if not isinstance(exog, dict):
            exog_dict_for_val: dict[str, pd.Series | pd.DataFrame | None] = {
                name: exog for name in series_dict
            }
        else:
            exog_dict_for_val = exog

        # Warn about series that have no exog entry in the dict
        if isinstance(exog, dict):
            missing = [
                name
                for name in series_dict
                if name not in exog or exog[name] is None
            ]
            if missing:
                warnings.warn(
                    f"No `exog` for the following series: {missing}. "
                    f"No exogenous variables will be stored for these series.",
                    MissingExogWarning,
                    stacklevel=4,
                )

        exog_names_per_series: dict[str, list[str] | None] = {}
        for name, s in series_dict.items():
            e = exog_dict_for_val.get(name, None)
            if e is None:
                exog_names_per_series[name] = None
                continue

            assert_aligned(e, s, f"`exog` for series '{name}'", f"series['{name}']")
            exog_names_per_series[name] = col_names(e)

        return exog_names_per_series


def validate_last_window_exog(
    last_window_exog: (
        pd.Series
        | pd.DataFrame
        | dict[str, pd.DataFrame | pd.Series | None]
        | None
    ),
    last_window: pd.Series | pd.DataFrame | dict[str, pd.Series] | None,
    exog_in_: bool,
) -> None:
    """
    Validate `last_window_exog` against `last_window` at predict time.

    Checks that the historical exogenous variables provided for a backtesting
    window are properly aligned with the corresponding context window. Issues a
    warning when the forecaster was trained with exog but `last_window_exog`
    is `None`. Raises when lengths or `DatetimeIndex` values do not match.

    Parameters
    ----------
    last_window_exog : pd.Series, pd.DataFrame, dict, or None
        Historical exogenous variables for the context window.
    last_window : pd.Series, pd.DataFrame, dict, or None
        Context window override passed to the predict method.
    exog_in_ : bool
        Whether the forecaster was trained with exogenous variables.

    Raises
    ------
    ValueError
        If `last_window_exog` length or `DatetimeIndex` does not match
        `last_window` for any series.
    """

    if last_window is None:
        return

    # Improvement A: warn if forecaster uses exog but last_window_exog is not provided
    if exog_in_ and last_window_exog is None:
        warnings.warn(
            "Forecaster was trained with exogenous variables but "
            "`last_window_exog` is `None`. The model will receive no historical "
            "exogenous variables (past_covariates) for this prediction window.",
            IgnoredArgumentWarning,
            stacklevel=3,
        )
        return

    if last_window_exog is None:
        return

    is_multi = isinstance(last_window, (pd.DataFrame, dict))

    if not is_multi:
        # Single-series: last_window is pd.Series
        assert_aligned(
            last_window_exog, last_window, "`last_window_exog`", "`last_window`"
        )

    else:
        # Multi-series: normalise both to dicts and validate per series
        if isinstance(last_window, pd.DataFrame):
            lw_dict: dict[str, pd.Series] = {
                col: last_window[col] for col in last_window.columns
            }
        else:
            lw_dict = last_window

        if isinstance(last_window_exog, dict):
            lwe_dict: dict[str, pd.Series | pd.DataFrame | None] = last_window_exog
        elif isinstance(last_window_exog, (pd.Series, pd.DataFrame)):
            # Broadcast: same exog for all context windows
            lwe_dict = {name: last_window_exog for name in lw_dict}
        else:
            lwe_dict = {}

        for name, lw in lw_dict.items():
            lwe = lwe_dict.get(name, None)
            if lwe is None:
                warnings.warn(
                    f"No `last_window_exog` for series '{name}'. The model will "
                    f"receive no historical exogenous variables (past_covariates) "
                    f"for this series.",
                    MissingExogWarning,
                    stacklevel=3,
                )
                continue

            assert_aligned(
                lwe, lw,
                f"`last_window_exog` for series '{name}'",
                f"`last_window['{name}']`",
            )


def align_exog_single(
    e: pd.Series | pd.DataFrame,
    steps: int,
    ref_end,
    index_freq_,
    label: str,
) -> pd.Series | pd.DataFrame:
    """
    Align a single exog array to the forecast horizon.

    For `DatetimeIndex` data reindexes to the exact expected step range,
    NaN-filling any gaps and emitting a `MissingValuesWarning`. For other
    index types trims to `steps` after validating the minimum length and,
    for `RangeIndex`, the start position.

    Parameters
    ----------
    e : pd.Series or pd.DataFrame
        Exog to align.
    steps : int
        Forecast horizon.
    ref_end : scalar
        Last observed index value (last entry of `last_window` or training range).
    index_freq_ : DateOffset, int, or None
        Index frequency.
    label : str
        Human-readable label for `e`, used in error and warning messages.

    Returns
    -------
    pd.Series or pd.DataFrame
        Aligned exog with exactly `steps` rows.
    """
    if isinstance(e.index, pd.DatetimeIndex) and index_freq_ is not None:
        expected_idx = pd.date_range(
            start=ref_end + index_freq_, periods=steps, freq=index_freq_
        )
        e_aligned = e.reindex(expected_idx)
        has_nans = (
            e_aligned.isnull().values.any()
            if isinstance(e_aligned, pd.DataFrame)
            else e_aligned.isnull().any()
        )
        if has_nans:
            warnings.warn(
                f"{label} has been reindexed to match the expected forecast "
                "horizon. Some positions were filled with NaN.",
                MissingValuesWarning,
                stacklevel=4,
            )
        return e_aligned
    else:
        if len(e) < steps:
            raise ValueError(
                f"{label} must have at least {steps} values. Got {len(e)}."
            )
        if isinstance(e.index, pd.RangeIndex) and index_freq_ is not None:
            expected_start = ref_end + index_freq_
            if e.index[0] != expected_start:
                raise ValueError(
                    f"To make predictions {label} must start one step ahead of "
                    "`last_window`.\n"
                    f"    `last_window` ends at: {ref_end}.\n"
                    f"    {label} starts at: {e.index[0]}.\n"
                    f"    Expected index: {expected_start}."
                )
        return e.iloc[:steps]


def validate_exog_predict(
    exog: pd.Series | pd.DataFrame | None,
    steps: int,
    last_window: pd.Series | pd.DataFrame | dict | None,
    exog_names_in_: list[str],
    exog_in_: bool,
    index_freq_: object,
    is_multiseries: bool,
    training_range_: object,
    series_names_in_: list[str],
    exog_names_in_per_series_: dict[str, list[str] | None] | None = None,
) -> pd.Series | pd.DataFrame | None:
    """
    Validate exog presence, column names, and align to the expected forecast horizon.

    Handles both flat (`pd.Series` / `pd.DataFrame`) and dict-type exog
    (multi-series mode).

    Presence mismatches (exog provided but not trained with, or vice-versa)
    always raise a `ValueError`.

    Column-name mismatches always raise a `ValueError` because they
    represent an unrecoverable structural error.

    For `DatetimeIndex` data, length and alignment issues are handled
    gracefully: `exog` is reindexed to the exact forecast horizon so that
    any missing positions are NaN-filled. Most foundational models that accept
    exogenous variables handle NaN natively. A `MissingValuesWarning` is
    issued whenever NaN values are introduced by the reindex.

    For non-DatetimeIndex data a plain length check is kept as a
    `ValueError`.

    For dict-type exog each value is validated and aligned independently.
    A `MissingExogWarning` is issued for any series that has no entry in
    the dict.

    Parameters
    ----------
    exog : pd.Series, pd.DataFrame, or None
        Exogenous variables provided to a predict method.
    steps : int
        Number of steps to forecast.
    last_window : pd.Series, pd.DataFrame, dict, or None
        Context override passed to the same predict call. Used to determine
        the reference end-timestamp for index alignment.
    exog_names_in_ : list of str
        Names of the exogenous variables used during training.
    exog_in_ : bool
        Whether the forecaster was trained with exogenous variables.
    index_freq_ : DateOffset or int
        Frequency of the training index.
    is_multiseries : bool
        Whether the forecaster was trained in multi-series mode.
    training_range_ : pandas Index or dict
        Training range stored by the forecaster.
    series_names_in_ : list of str
        Names of all series seen during training.
    exog_names_in_per_series_ : dict[str, list[str] | None], optional
        Per-series exog column names stored at fit time. When provided, column
        validation for dict exog uses the series-specific expected columns
        instead of the global union in `exog_names_in_`. Entries mapping to
        `None` indicate that the series had no exog at fit time.

    Returns
    -------
    pd.Series, pd.DataFrame, dict, or None
        The (possibly reindexed / NaN-filled) exog aligned to the forecast
        horizon. For dict exog, each per-series value is aligned
        independently and the dict is returned with updated values.
    """
    # Presence checks (unrecoverable — always raise)
    if exog is None and exog_in_:
        raise ValueError(
            "Forecaster trained with exogenous variable/s. "
            "Same variable/s must be provided when predicting."
        )
    if exog is not None and not exog_in_:
        raise ValueError(
            "Forecaster trained without exogenous variable/s. "
            "`exog` must be `None` when predicting."
        )

    if exog is None:
        return exog

    # --- Dict exog: validate and align each series independently ---------------
    if isinstance(exog, dict):
        not_valid = [
            k for k, v in exog.items()
            if not isinstance(v, (pd.Series, pd.DataFrame, type(None)))
        ]
        if not_valid:
            raise TypeError(
                "All values in the `exog` dict must be a pd.Series, a "
                f"pd.DataFrame, or None. Invalid keys: {not_valid}."
            )

        missing_series = [n for n in series_names_in_ if n not in exog]
        if missing_series:
            warnings.warn(
                f"No `exog` for series {missing_series}. All values of the "
                "exogenous variables for these series will be NaN.",
                MissingExogWarning,
                stacklevel=3,
            )

        exog = dict(exog)  # shallow copy — do not mutate the caller's dict
        for sid, e in exog.items():
            if e is None:
                continue

            # Column-name check: use per-series expected columns when available
            cols = col_names(e)
            expected_cols: list[str] | None = (
                exog_names_in_per_series_.get(sid)
                if exog_names_in_per_series_ is not None
                else None
            )
            if expected_cols is None:
                expected_cols = exog_names_in_
            bad_cols = set(cols) - set(expected_cols)
            if bad_cols:
                raise ValueError(
                    f"`exog` for series '{sid}' contains columns not seen "
                    f"during training: {sorted(bad_cols)}. "
                    f"Expected columns: {expected_cols}."
                )

            # Reference end-point for this specific series
            if isinstance(last_window, dict) and last_window.get(sid) is not None:
                ref_end_s = last_window[sid].index[-1]
            elif last_window is not None and isinstance(
                last_window, (pd.Series, pd.DataFrame)
            ):
                ref_end_s = last_window.index[-1]
            elif isinstance(training_range_, dict) and sid in training_range_:
                ref_end_s = training_range_[sid][1]
            else:
                ref_end_s = training_range_[series_names_in_[0]][1]

            # Alignment
            exog[sid] = align_exog_single(
                e, steps, ref_end_s, index_freq_, f"`exog` for series '{sid}'"
            )

        return exog

    # --- Flat exog (pd.Series / pd.DataFrame) ----------------------------------
    if not isinstance(exog, (pd.Series, pd.DataFrame)):
        return exog

    # Check #1: column names match exog_names_in_ (unrecoverable — always raises)
    if isinstance(exog, pd.DataFrame):
        missing_cols = set(exog_names_in_) - set(exog.columns)
        if missing_cols:
            raise ValueError(
                f"Missing columns in `exog`. "
                f"Expected {exog_names_in_}. "
                f"Got {exog.columns.to_list()}."
            )
    else:  # pd.Series
        if exog.name not in exog_names_in_:
            raise ValueError(
                f"'{exog.name}' was not observed during training. "
                f"Exogenous variables must be: {exog_names_in_}."
            )

    # Determine the reference end-point (last known observation)
    if last_window is not None and isinstance(
        last_window, (pd.Series, pd.DataFrame)
    ):
        ref_end = last_window.index[-1]
    elif last_window is not None and isinstance(last_window, dict):
        first_series = next(
            (v for v in last_window.values() if v is not None), None
        )
        if first_series is not None:
            ref_end = first_series.index[-1]
        elif not is_multiseries:
            ref_end = training_range_[1]
        else:
            ref_end = training_range_[series_names_in_[0]][1]
    elif not is_multiseries:
        ref_end = training_range_[1]
    else:
        ref_end = training_range_[series_names_in_[0]][1]

    # Align exog to the forecast horizon.
    return align_exog_single(exog, steps, ref_end, index_freq_, "`exog`")
