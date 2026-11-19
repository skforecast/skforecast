"""P17: skip `.drop(columns=[])` in `_extract_data_folds_multiseries`.
The proposal is simulated by patching `DataFrame.drop` so that an empty list
returns the same object (no copy), then backtesting results are compared."""
import _env
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries

rng = np.random.default_rng(0)
n = 120
idx = pd.date_range("2020-01-01", periods=n, freq="D")
series = pd.DataFrame(rng.normal(size=(n, 3)).cumsum(axis=0) + 50, index=idx, columns=list("abc"))
series_nan = series.copy(); series_nan.iloc[:30, 2] = np.nan  # a column with leading NaN

s = series.iloc[10:20]
d = s.drop(columns=[])
print("drop(columns=[]) returns a new object:", d is not s,
      "| shares memory with the slice:", np.shares_memory(d.to_numpy(), s.to_numpy()),
      "| slice shares memory with `series`:", np.shares_memory(s.to_numpy(), series.to_numpy()),
      "| slice._is_copy set:", getattr(s, "_is_copy", None) is not None,
      "| drop result _is_copy set:", getattr(d, "_is_copy", None) is not None)

orig_drop = pd.DataFrame.drop
n_skipped = 0


def drop_patched(self, *args, **kwargs):
    global n_skipped
    cols = kwargs.get("columns")
    if not args and isinstance(cols, list) and len(cols) == 0 and len(kwargs) == 1:
        n_skipped += 1
        return self
    return orig_drop(self, *args, **kwargs)


def run(make, data, refit, **bt_kw):
    cv = TimeSeriesFold(steps=5, initial_train_size=80, refit=refit)
    with warnings.catch_warnings():
        from skforecast.exceptions import DataTransformationWarning
        warnings.simplefilter("ignore", category=DataTransformationWarning)
        warnings.simplefilter("error", category=FutureWarning)
        for cat in ("SettingWithCopyWarning", "ChainedAssignmentError"):
            c = getattr(pd.errors, cat, None)
            if c is not None:
                warnings.simplefilter("error", category=c)
        m, p = backtesting_forecaster_multiseries(
            forecaster=make(), series=data, cv=cv, metric="mean_absolute_error",
            show_progress=False, suppress_warnings=True, **bt_kw)
    return m, p


makers = {
    "MultiSeries Ridge": lambda: ForecasterRecursiveMultiSeries(Ridge(), lags=5),
    "MultiSeries scaler+diff": lambda: ForecasterRecursiveMultiSeries(
        Ridge(), lags=5, transformer_series=StandardScaler(), differentiation=1),
    "MultiSeries dropna_from_series": lambda: ForecasterRecursiveMultiSeries(Ridge(), lags=5, dropna_from_series=True),
    "DMV": lambda: ForecasterDirectMultiVariate(Ridge(), level="a", lags=5, steps=5, transformer_series=StandardScaler()),
}
for name, make in makers.items():
    for dname, data in (("no NaN", series), ("leading NaN in 'c'", series_nan)):
        if name == "DMV" and dname != "no NaN":
            continue
        for refit in (False, True):
            for extra in ({}, {"interval": [0.1, 0.9], "interval_method": "conformal"}):
                if name == "DMV" and extra:
                    continue
                before = data.copy()
                try:
                    pd.DataFrame.drop = orig_drop
                    m0, p0 = run(make, data, refit, **extra)
                    pd.DataFrame.drop = drop_patched; n_skipped = 0
                    m1, p1 = run(make, data, refit, **extra)
                    res = (f"metrics equal={m0.equals(m1)} predictions equal={p0.equals(p1)} "
                           f"user series untouched={data.equals(before)} drops skipped={n_skipped}")
                except Exception as e:
                    res = f"{type(e).__name__}: {str(e).splitlines()[0][:90]}"
                finally:
                    pd.DataFrame.drop = orig_drop
                print(f"{name:32s} {dname:20s} refit={refit!s:5s} {'interval' if extra else 'point':8s}: {res}")
print(_env.versions())
