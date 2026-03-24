################################################################################
#                     skforecast.foundational._utils                           #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import warnings
import pandas as pd
from ..exceptions import IgnoredArgumentWarning, InputTypeWarning, MissingExogWarning


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
