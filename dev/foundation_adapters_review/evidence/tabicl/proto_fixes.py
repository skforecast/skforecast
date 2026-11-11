"""
Prototype (monkeypatch, nothing in skforecast/ is modified) of two small fixes
proposed for TabICLAdapter:

  P-quantiles: cast the quantile levels to Python float before calling
               `predict_df` (tabicl rejects int and numpy float32 levels).
  P-warnings:  keep hiding the tqdm bar with the stderr redirection, but record
               the warnings raised inside it and re-emit them once stderr is
               restored, so backend warnings reach the user.

Neither change alters the predicted values: the script checks that the
prediction with float levels equals the one with the cast int levels.
Run from the repository root:
    ~/.venvs/fm-review-tabicl/bin/python dev/foundation_adapters_review/evidence/tabicl/proto_fixes.py
"""

import contextlib
import importlib.metadata as md
import sys
import warnings

import numpy as np
import pandas as pd

import skforecast.foundation._adapters as adapters
from skforecast.foundation import FoundationModel

print("python", sys.version.split()[0], "| tabicl", md.version("tabicl"), "| skforecast", md.version("skforecast"))

_original_redirect = contextlib.redirect_stderr
_original_predict = adapters.TabICLAdapter.predict


@contextlib.contextmanager
def redirect_stderr_keep_warnings(target):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _original_redirect(target):
            yield target
    for w in caught:
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


def predict_cast_quantiles(self, steps, context, context_exog, exog, quantiles):
    if quantiles is not None:
        quantiles = [float(q) for q in quantiles]
    return _original_predict(self, steps, context, context_exog, exog, quantiles)


idx = pd.date_range("2024-01-01", periods=126, freq="h")
y = pd.Series(10 + np.sin(np.arange(126) / 4), index=idx, name="y").iloc[:120]
ex = pd.DataFrame({"x": np.cos(np.arange(126) / 4)}, index=idx)
ex_future = ex.iloc[120:].copy()
ex_future.iloc[0, 0] = np.nan

m = FoundationModel("soda-inria/tabicl", context_length=64)
m.fit(series=y, exog=ex.iloc[:120])
reference = m.predict(steps=6, exog=ex_future, quantiles=[0.0, 0.5, 1.0])

adapters.contextlib = type("patched", (), {
    "redirect_stderr": staticmethod(redirect_stderr_keep_warnings),
    "nullcontext": staticmethod(contextlib.nullcontext),
})
adapters.TabICLAdapter.predict = predict_cast_quantiles

with warnings.catch_warnings(record=True) as seen:
    warnings.simplefilter("always")
    patched = m.predict(steps=6, exog=ex_future, quantiles=[0, 0.5, 1])
print("columns with int levels after cast:", patched.columns.tolist())
print("equal to float levels:", np.array_equal(reference.iloc[:, 1:].to_numpy(), patched.iloc[:, 1:].to_numpy()))
print("warnings re-emitted:", sorted({f"{w.category.__name__}: {str(w.message)[:90]}" for w in seen}))
