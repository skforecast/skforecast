"""
Does `pytest.warns` detect a backend warning emitted inside the adapter's
`redirect_stderr` block today (without the fix proposed in TPF-P1)? If it does,
the unit test proposed in TPF-P1 cannot catch the regression.

Run from the repository root:
    ~/.venvs/fm-review-tabpfn/bin/python -m pytest -q -p no:cacheprovider \
        dev/foundation_adapters_review/evidence/tabpfn/verification/test_p1_pytest_warns.py
"""

import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from skforecast.foundation import FoundationModel


class WarningPipeline:
    def predict_df(self, context_df, future_df, quantiles=None):
        warnings.warn("backend data quality warning", UserWarning)
        index = pd.MultiIndex.from_frame(future_df[["item_id", "timestamp"]])
        out = pd.DataFrame({"target": 1.0}, index=index)
        for q in quantiles:
            out[q] = 1.0
        return out


def make_model():
    idx = pd.date_range("2024-01-01", periods=50, freq="h")
    model = FoundationModel(
        "priorlabs/tabpfn-ts", model=WarningPipeline(), show_progress=False
    )
    model.fit(series=pd.Series(np.arange(50.0), index=idx, name="y"))
    return model


def test_pytest_warns_sees_warning_even_with_redirect():
    model = make_model()
    with pytest.warns(UserWarning, match="backend data quality warning"):
        model.predict(steps=3)


def test_warning_text_does_not_reach_stderr():
    here = os.path.dirname(os.path.abspath(__file__))
    code = (
        "import numpy as np, pandas as pd, sys\n"
        f"sys.path.insert(0, {here!r})\n"
        "from test_p1_pytest_warns import make_model\n"
        "make_model().predict(steps=3)\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "backend data quality warning" not in res.stderr
