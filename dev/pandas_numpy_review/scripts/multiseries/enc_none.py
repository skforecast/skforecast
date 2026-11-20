import timeit, warnings, numpy as np, pandas as pd, time
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursiveMultiSeries
rng = np.random.default_rng(0); LEN=1000; N=100
idx = pd.date_range("2020-01-01", periods=LEN, freq="D")
series = {f"s{i:03d}": pd.Series(rng.normal(size=LEN).cumsum(), index=idx) for i in range(N)}
for enc in ["ordinal", None, "onehot"]:
    f = ForecasterRecursiveMultiSeries(Ridge(), lags=24, encoding=enc)
    out = f._create_train_X_y(series, store_last_window=False)
    X = out[0]
    td = min(timeit.repeat(lambda: X.drop(columns="_level_skforecast"), number=3, repeat=3))/3*1e3 if enc is None else 0
    ts = []
    for _ in range(5):
        f2 = ForecasterRecursiveMultiSeries(Ridge(), lags=24, encoding=enc); t0=time.perf_counter(); f2.fit(series); ts.append(time.perf_counter()-t0)
    print(enc, "X", X.shape, "blocks", len(X._mgr.blocks), f"fit {min(ts)*1e3:.1f} ms", f"drop {td:.2f} ms" if enc is None else "")
