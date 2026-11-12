################################################################################
#                          skforecast.utils._pandas_compat                     #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
"""
Private helpers that isolate the differences between pandas 2 and pandas 3.

pandas 3 enables Copy-on-Write: `concat` and `astype` no longer copy the data
whose dtype does not change, and the `copy` keyword of both is deprecated (it
emits a `Pandas4Warning` in pandas 3.1). Under pandas 2, `copy=False` avoids a
copy that is measurable in the training matrices, so the helpers keep it there.
`pd.infer_freq` returns a string in pandas 2 and will return an offset from
pandas 4 (pandas 3.1 warns unless the option `future.infer_freq_returns_offset`
is set); `infer_freq` returns an offset in every version.

This is the only module of the package that looks at the pandas version.
"""

from __future__ import annotations
from typing import Any
from packaging.version import Version
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

PANDAS_GE_3 = Version(pd.__version__).major >= 3


def concat(
    objs: list[pd.DataFrame | pd.Series], **kwargs: Any
) -> pd.DataFrame | pd.Series:
    """
    `pd.concat` without an eager copy of the data in pandas 2 and pandas 3.

    Parameters
    ----------
    objs : list
        Objects to concatenate.
    **kwargs : Any
        Keyword arguments passed to `pd.concat` (`axis`, `ignore_index`...).
        Do not pass `copy`.

    Returns
    -------
    result : pandas DataFrame, pandas Series
        Concatenated object.

    """

    if not PANDAS_GE_3:
        kwargs["copy"] = False

    return pd.concat(objs, **kwargs)


def astype(
    obj: pd.DataFrame | pd.Series,
    dtypes: dict[str, Any] | str | type | np.dtype,
    changed_only: bool = False
) -> pd.DataFrame | pd.Series:
    """
    `astype` without an eager copy of the data in pandas 2 and pandas 3.

    Parameters
    ----------
    obj : pandas DataFrame, pandas Series
        Object to cast.
    dtypes : dict, dtype
        Target dtypes, as accepted by `astype`.
    changed_only : bool, default False
        Only for a DataFrame and a dict of dtypes: cast only the columns whose
        dtype differs from the target, and return `obj` itself when no column
        changes. It is meant for calls made once per fit; in prediction loops
        the scan of the dtypes costs more than it saves. Ignored when `obj`
        has duplicated column names: every column is cast, as pandas does.

    Returns
    -------
    result : pandas DataFrame, pandas Series
        Cast object, or `obj` when `changed_only=True` and nothing changes.

    """

    if changed_only and obj.columns.is_unique:
        current = obj.dtypes
        dtypes = {
            col: dtype for col, dtype in dtypes.items() if current[col] != dtype
        }
        if not dtypes:
            return obj

    if PANDAS_GE_3:
        return obj.astype(dtypes)

    return obj.astype(dtypes, copy=False)


def infer_freq(index: pd.DatetimeIndex) -> pd.offsets.BaseOffset | None:
    """
    `pd.infer_freq` returning an offset (the behavior of pandas 4) in every
    pandas version, without the deprecation warning of pandas 3.1.

    Parameters
    ----------
    index : pandas DatetimeIndex
        Index to infer the frequency from. As in `pd.infer_freq`, it needs at
        least 3 values.

    Returns
    -------
    freq : pandas BaseOffset, None
        Inferred frequency, or None when no frequency fits the index.

    """

    try:
        with pd.option_context("future.infer_freq_returns_offset", True):
            inferred = pd.infer_freq(index)
    except pd.errors.OptionError:
        # pandas < 3.1: the option does not exist and `pd.infer_freq` returns
        # a string without warnings.
        inferred = pd.infer_freq(index)

    return to_offset(inferred) if inferred is not None else None
