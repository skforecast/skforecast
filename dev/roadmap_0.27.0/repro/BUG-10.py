"""
BUG-10: TabICL (and TabPFN-TS, same code) do not ignore past-only covariates
although `IgnoredArgumentWarning` says they are ignored.

`FoundationModel._check_exog_columns` warns that historical exog columns
without future values are ignored when the adapter has
`supports_past_only_covariates=False`, but the TabICL adapter
(`skforecast/foundation/_adapters.py:2679-2685`) concatenates the whole
`context_exog` into the context frame. The prediction changes silently.

Expected: same prediction as without exog.
Observed (skforecast 0.26.0, tabicl 2.2.0): [5.21 ... 5.06] with the "ignored"
columns versus [-17.29 ... -18.22] without exog (true values around -16).

Requires: tabicl (`uv pip install tabicl`).
Run: python dev/roadmap_0.27.0/repro/BUG-10.py
"""

import warnings

import numpy as np
import pandas as pd

from skforecast.foundation import FoundationModel, ForecasterFoundation

rng = np.random.default_rng(0)
n = 300
idx = pd.date_range("2020-01-01", periods=n, freq="D")
x = pd.Series(rng.normal(0, 1, n).cumsum(), index=idx)
y = pd.Series(5 + 2 * x + rng.normal(0, 0.3, n), index=idx, name="y")
exog = pd.DataFrame({"x_observed": x})

forecaster = ForecasterFoundation(
    FoundationModel("soda-inria/tabicl", context_length=128)
)
forecaster.fit(series=y)
end = n - 7

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    pred_past_only = forecaster.predict(
        steps=7, context=y.iloc[:end], context_exog=exog.iloc[:end]
    )["pred"].to_numpy()
warned = [str(x.message)[:90] for x in w if "ignored" in str(x.message)]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pred_no_exog = forecaster.predict(
        steps=7, context=y.iloc[:end]
    )["pred"].to_numpy()

print("Warning emitted      :", warned)
print("Past-only (ignored?) :", pred_past_only.round(2))
print("Without exog         :", pred_no_exog.round(2))
print("True values          :", y.iloc[end:].to_numpy().round(2))
print("Identical (expected True):", np.allclose(pred_past_only, pred_no_exog))
