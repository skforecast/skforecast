"""
Draft of the proposed compatibility module `skforecast/utils/_pandas_compat.py`.

It is the ONLY place of the package allowed to look at the pandas version.
Everything else calls these helpers. The draft is kept here (not in the
package) as the reference for the implementation plan of the report.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PANDAS_VERSION = tuple(int(x) for x in pd.__version__.split(".")[:2])
PANDAS_GE_3 = PANDAS_VERSION >= (3, 0)

# pandas 3 emits Pandas4Warning for the `copy` keyword: Copy-on-Write makes it
# meaningless. pandas 2 still needs copy=False to avoid an eager copy.
_COPY_KW: dict[str, Any] = {} if PANDAS_GE_3 else {"copy": False}


def concat(objs, **kwargs) -> pd.DataFrame | pd.Series:
    """pd.concat without an eager copy in pandas 2 and without the deprecated
    keyword in pandas 3."""
    return pd.concat(objs, **_COPY_KW, **kwargs)


def astype(obj, dtypes, **kwargs):
    """astype without an eager copy of the unchanged columns in pandas 2 and
    without the deprecated keyword in pandas 3."""
    return obj.astype(dtypes, **_COPY_KW, **kwargs)


def frame_from_array(data: np.ndarray, **kwargs) -> pd.DataFrame:
    """DataFrame that takes ownership of a private ndarray: never copies in
    either version (pandas 3 copies by default; pandas 2 does not)."""
    return pd.DataFrame(data=data, copy=False, **kwargs)


def writable_array(obj: pd.Series | pd.DataFrame | pd.Index) -> np.ndarray:
    """Array that the caller may modify in place. In pandas 3 to_numpy() may
    return a read-only view; in pandas 2 it may return a writable view of the
    user's data, which must not be modified either. Always a copy, so the
    behaviour and the cost are the same in both versions."""
    return obj.to_numpy(copy=True)


def is_string_dtype(dtype) -> bool:
    """True for the pandas 3 default ``str`` dtype, for ``object`` columns of
    strings (pandas 2) and for ``pd.StringDtype``. Use it instead of
    ``dtype == object`` when the question is "is this a text column"."""
    return pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype)


def dayofweek(index: pd.DatetimeIndex) -> np.ndarray:
    """``DatetimeIndex.dayofweek`` is deprecated in pandas 3.1 (Pandas4Warning)
    in favour of ``day_of_week``, which exists since pandas 1.1."""
    return index.day_of_week


def dayofyear(index: pd.DatetimeIndex) -> np.ndarray:
    return index.day_of_year
