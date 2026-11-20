"""I9 (index_freq_ type) and I11 (transformer feature_names_in_) by running."""
import _env
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries, ForecasterEquivalentDate
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate

warnings.simplefilter("ignore")
n = 40
idx = pd.date_range("2020-01-01", periods=n, freq="D")
y = pd.Series(np.arange(n, dtype=float), index=idx, name="sales")
df = pd.DataFrame({"a": y, "b": y * 2})

fs = {
    "Recursive": ForecasterRecursive(Ridge(), lags=3, transformer_y=StandardScaler()),
    "Direct": ForecasterDirect(Ridge(), lags=3, steps=2, transformer_y=StandardScaler()),
    "MultiSeries": ForecasterRecursiveMultiSeries(Ridge(), lags=3, transformer_series=StandardScaler()),
    "DMV": ForecasterDirectMultiVariate(Ridge(), level="a", lags=3, steps=2, transformer_series=StandardScaler()),
    "EquivalentDate": ForecasterEquivalentDate(offset=7),
}
for name, f in fs.items():
    if name in ("MultiSeries", "DMV"):
        f.fit(series=df)
        tr = list(f.transformer_series_.values())[0]
    else:
        f.fit(y=y)
        tr = getattr(f, "transformer_y", None)
    print(f"{name:15s} index_freq_={f.index_freq_!r} ({type(f.index_freq_).__name__})"
          f" | transformer.feature_names_in_={getattr(tr, 'feature_names_in_', 'absent')}")
try:
    from skforecast.recursive import ForecasterStats
    from skforecast.stats import Arima
    f = ForecasterStats(Arima(order=(1, 0, 0)))
    f.fit(y=y)
    print(f"{'Stats':15s} index_freq_={f.index_freq_!r} ({type(f.index_freq_).__name__})")
    print("   freq check used by check_predict_input: Day() == 'D' ->", idx.freq == f.index_freq_)
except Exception as e:
    print("ForecasterStats not run:", type(e).__name__, str(e)[:100])
print(_env.versions())
