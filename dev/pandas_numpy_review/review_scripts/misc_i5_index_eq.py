"""I5: `(a == b).all()` (classifier) vs `.equals` (rest) when checking exog index."""
import _env
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveClassifier

warnings.simplefilter("ignore")
n = 40
idx = pd.date_range("2020-01-01", periods=n, freq="D")
y_reg = pd.Series(np.arange(n, dtype=float), index=idx, name="y")
y_clf = pd.Series(np.tile([0, 1, 2, 1], n // 4), index=idx, name="y")

cases = {
    "tz-aware exog vs naive y": idx.tz_localize("UTC"),
    "tz Madrid exog vs UTC-equal instants (y naive)": idx.tz_localize("Europe/Madrid"),
    "shifted 1 day": idx + pd.Timedelta(days=1),
    "same values, freq=None": pd.DatetimeIndex(idx.to_numpy()),
    "same values, other name": idx.rename("other"),
    "same values, datetime64[s]": idx.as_unit("s"),
    "RangeIndex exog vs datetime y": pd.RangeIndex(n),
    "object index of Timestamps": pd.Index(list(idx), dtype=object),
    "string dates index": pd.Index(idx.strftime("%Y-%m-%d")),
}


def run(forecaster, y, exog_index):
    exog = pd.DataFrame({"e": np.arange(n, dtype=float)}, index=exog_index)
    try:
        forecaster.fit(y, exog=exog)
        return "fit OK"
    except Exception as e:
        return f"{type(e).__name__}: {str(e).splitlines()[0][:75]}"


for name, ex_idx in cases.items():
    print(f"## {name}")
    print("   ==      :", end=" ")
    try:
        r = (ex_idx == idx)
        print(type(r).__name__, bool(np.all(r)), "| .all attr:", hasattr(r, "all"))
    except Exception as e:
        print(f"{type(e).__name__}: {str(e)[:60]}")
    print("   equals  :", ex_idx.equals(idx))
    print("   Recursive :", run(ForecasterRecursive(Ridge(), lags=3), y_reg, ex_idx))
    print("   Classifier:", run(ForecasterRecursiveClassifier(LogisticRegression(), lags=3), y_clf, ex_idx))

print("## tz-aware y and naive exog (reverse)")
ya = y_reg.tz_localize("UTC"); yc = y_clf.tz_localize("UTC")
print("   Recursive :", run(ForecasterRecursive(Ridge(), lags=3), ya, idx))
print("   Classifier:", run(ForecasterRecursiveClassifier(LogisticRegression(), lags=3), yc, idx))

print("## raw pandas: == with different lengths / types")
for a, b in [(idx, idx[:-1]), (pd.RangeIndex(5), pd.RangeIndex(4)), (pd.RangeIndex(5), pd.Index([0., 1, 2, 3, 4]))]:
    try:
        print("  ", type(a).__name__, len(a), len(b), "==:", (a == b).all(), "equals:", a.equals(b))
    except Exception as e:
        print("  ", type(a).__name__, len(a), len(b), f"== raises {type(e).__name__}: {e}", "| equals:", a.equals(b))
print(_env.versions())
