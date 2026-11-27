# Does the conversion at the boundary copy the training matrix? (branch tree)
import tracemalloc, warnings
import numpy as np, pandas as pd
import skforecast; print(skforecast.__file__)
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.utils import _train_matrix_to_numpy
print("pandas", pd.__version__, "numpy", np.__version__)
warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
n_series, n = 100, 2000
idx = pd.date_range("2020", periods=n, freq="h")
series = {f"s{i}": pd.Series(rng.normal(size=n), index=idx, name=f"s{i}") for i in range(n_series)}
exogs = {
    "float exog": pd.DataFrame({"e1": rng.normal(size=n), "e2": rng.normal(size=n)}, index=idx),
    "one int exog": pd.DataFrame({"e1": rng.normal(size=n), "e2": rng.integers(0, 5, size=n)}, index=idx),
    "one bool exog": pd.DataFrame({"e1": rng.normal(size=n), "e2": rng.integers(0, 2, size=n).astype(bool)}, index=idx),
    "text exog (auto encodes it)": pd.DataFrame({"e1": rng.normal(size=n), "e2": rng.choice(["a", "b", "c"], size=n)}, index=idx),
}
for encoding in ["ordinal", "ordinal_category", "onehot"]:
    for name, exog in exogs.items():
        if encoding != "ordinal" and name != "float exog":
            continue
        f = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=24, encoding=encoding)
        X, y = f.create_train_X_y(series=series, exog={k: exog for k in series})
        size_mb = X.shape[0] * X.shape[1] * 8 / 1e6
        tracemalloc.start()
        arr = _train_matrix_to_numpy(X)
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        dtypes = sorted({str(d) for d in X.dtypes})
        print(f"encoding={encoding:17s} {name:28s} X={X.shape} ({size_mb:6.1f} MB float64) dtypes={dtypes} "
              f"-> new memory {cur / 1e6:6.1f} MB, peak {peak / 1e6:6.1f} MB, F-contig={arr.flags.f_contiguous} C-contig={arr.flags.c_contiguous}")
