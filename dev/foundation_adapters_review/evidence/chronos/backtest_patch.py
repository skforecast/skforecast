"""
Run the backtesting of the user guide (common/backtest_guide.py) with the
change proposed for `ChronosAdapter._to_covariate_array` applied as a
monkeypatch: a pandas Categorical column is passed to the backend as an
object array, so that Chronos-2 encodes it as a categorical covariate even
when its categories are numbers.

Usage (from the repository root):
    ~/.venvs/fm-review-chronos/bin/python \
        dev/foundation_adapters_review/evidence/chronos/backtest_patch.py [backtest_guide args]
"""

import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from skforecast.foundation._adapters import ChronosAdapter


def _to_covariate_array_patched(col_data):
    if isinstance(col_data, pd.Series):
        if isinstance(col_data.dtype, pd.CategoricalDtype):
            return col_data.astype(object).to_numpy()
        if pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
            return col_data.astype(np.float32).to_numpy()
        return col_data.to_numpy()
    arr = np.asarray(col_data)
    if arr.dtype.kind in ("i", "u", "f", "b"):
        return arr.astype(np.float32)
    return arr


ChronosAdapter._to_covariate_array = staticmethod(_to_covariate_array_patched)
print("ChronosAdapter._to_covariate_array patched (Categorical -> object)")

guide = Path(__file__).resolve().parents[1] / "common" / "backtest_guide.py"
sys.argv = [str(guide), *sys.argv[1:]]
runpy.run_path(str(guide), run_name="__main__")
