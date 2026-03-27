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


def _check_preprocess_series_type(
    series: pd.Series | pd.DataFrame | dict,
) -> tuple[bool, list[str], pd.Series | pd.DataFrame | dict]:
    """
    Validate `series`, normalise long-format DataFrames, and return
    ``(is_multiseries, series_names, normalised_series)``.

    Long-format DataFrames (MultiIndex with series IDs in the first level and
    a ``DatetimeIndex`` in the second level) are converted to a
    ``dict[str, pd.Series]`` before being returned. All other supported types
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
        not a ``pandas.DatetimeIndex``, or if `series` is an unsupported type.
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


def _check_preprocess_exog_type(
    exog: pd.Series | pd.DataFrame | dict | None,
    series_names_in_: list[str] | None = None,
) -> pd.Series | pd.DataFrame | dict | None:
    """
    Validate `exog` and normalise long-format MultiIndex DataFrames.

    Flat-index ``pd.Series`` and wide ``pd.DataFrame`` are returned
    unchanged — they are broadcast to all series by the adapter.  A
    ``dict`` is returned unchanged.  A long-format ``pd.Series`` or
    ``pd.DataFrame`` (MultiIndex, first level = series IDs, second level
    = ``DatetimeIndex``) is converted to a ``dict[str, pd.DataFrame]``
    and an ``InputTypeWarning`` is issued.

    Parameters
    ----------
    exog : pd.Series, pd.DataFrame, dict, or None
        Input to validate and normalise.
    series_names_in_ : list of str, optional
        Series names seen at fit time. When provided, a
        ``MissingExogWarning`` is issued for any series that has no
        corresponding entry in a long-format `exog`.

    Returns
    -------
    pd.Series, pd.DataFrame, dict, or None
        Normalised exog. Guaranteed NOT to be a long-format DataFrame.

    Raises
    ------
    TypeError
        If `exog` is a long-format DataFrame whose second MultiIndex level
        is not a ``pandas.DatetimeIndex``, or if `exog` is an unsupported
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


def _validate_exog_predict(
    exog: pd.Series | pd.DataFrame | None,
    steps: int,
    last_window: pd.Series | pd.DataFrame | dict | None,
    exog_names_in_: list[str],
    exog_in_: bool,
    index_freq_: object,
    is_multiseries: bool,
    training_range_: object,
    series_names_in_: list[str],
) -> pd.Series | pd.DataFrame | None:
    """
    Validate exog presence, column names, and align to the expected forecast horizon.

    Handles both flat (``pd.Series`` / ``pd.DataFrame``) and dict-type exog
    (multi-series mode).

    Presence mismatches (exog provided but not trained with, or vice-versa)
    always raise a ``ValueError``.

    Column-name mismatches always raise a ``ValueError`` because they
    represent an unrecoverable structural error.

    For ``DatetimeIndex`` data, length and alignment issues are handled
    gracefully: ``exog`` is reindexed to the exact forecast horizon so that
    any missing positions are NaN-filled. Most foundational models that accept
    exogenous variables handle NaN natively. A ``MissingValuesWarning`` is
    issued whenever NaN values are introduced by the reindex.

    For non-DatetimeIndex data a plain length check is kept as a
    ``ValueError``.

    For dict-type exog each value is validated and aligned independently.
    A ``MissingExogWarning`` is issued for any series that has no entry in
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

            # Column-name check
            cols = e.columns.to_list() if isinstance(e, pd.DataFrame) else [e.name]
            bad_cols = set(cols) - set(exog_names_in_)
            if bad_cols:
                raise ValueError(
                    f"`exog` for series '{sid}' contains columns not seen "
                    f"during training: {sorted(bad_cols)}. "
                    f"Expected columns: {exog_names_in_}."
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
            if isinstance(e.index, pd.DatetimeIndex) and index_freq_ is not None:
                expected_idx = pd.date_range(
                    start=ref_end_s + index_freq_, periods=steps, freq=index_freq_
                )
                e_aligned = e.reindex(expected_idx)
                has_nans = (
                    e_aligned.isnull().values.any()
                    if isinstance(e_aligned, pd.DataFrame)
                    else e_aligned.isnull().any()
                )
                if has_nans:
                    warnings.warn(
                        f"`exog` for series '{sid}' has been reindexed to "
                        "match the expected forecast horizon. Some positions "
                        "were filled with NaN.",
                        MissingValuesWarning,
                        stacklevel=3,
                    )
                exog[sid] = e_aligned
            else:
                if len(e) < steps:
                    raise ValueError(
                        f"`exog` for series '{sid}' must have at least "
                        f"{steps} values. Got {len(e)}."
                    )
                exog[sid] = e.iloc[:steps]

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

    # For DatetimeIndex: reindex to the exact expected forecast horizon.
    # Missing positions are NaN-filled; excess rows are dropped.
    if isinstance(exog.index, pd.DatetimeIndex) and index_freq_ is not None:
        expected_index = pd.date_range(
            start=ref_end + index_freq_, periods=steps, freq=index_freq_
        )
        exog_aligned = exog.reindex(expected_index)
        has_new_nans = (
            exog_aligned.isnull().values.any()
            if isinstance(exog_aligned, pd.DataFrame)
            else exog_aligned.isnull().any()
        )
        if has_new_nans:
            warnings.warn(
                "`exog` has been reindexed to match the expected forecast "
                "horizon. Some positions were filled with NaN because `exog` "
                f"did not fully cover the {steps}-step horizon starting at "
                f"{expected_index[0]}.",
                MissingValuesWarning,
                stacklevel=3,
            )
        exog = exog_aligned
    else:
        # Non-DatetimeIndex: enforce minimum length, then trim to exactly `steps`
        if len(exog) < steps:
            raise ValueError(
                f"`exog` must have at least as many values as steps predicted, "
                f"{steps}. Got {len(exog)}."
            )
        exog = exog.iloc[:steps]

    return exog
