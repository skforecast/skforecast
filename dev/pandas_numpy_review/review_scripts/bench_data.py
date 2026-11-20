"""Data of the timing scripts: the same generator as `scripts/ceiling_bt_ms.py`."""
import numpy as np
import pandas as pd

DOW = np.array(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])


def make_data(n_series, exog_kind, n_rows=1500, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_rows, freq="h")
    names = [f"series_{i:04d}" for i in range(n_series)]
    values = np.cumsum(rng.normal(size=(n_rows, n_series)), axis=0) + 100
    series = {k: pd.Series(values[:, j], index=idx, name=k) for j, k in enumerate(names)}

    def make_exog():
        df = pd.DataFrame(
            {"temp": rng.normal(size=n_rows), "promo": rng.normal(size=n_rows)}, index=idx
        )
        if exog_kind == "mixed":
            df["hour"] = idx.hour.to_numpy().astype("int64")
            df["dow"] = pd.Categorical(DOW[idx.dayofweek.to_numpy()], categories=list(DOW))
            df["shop_type"] = np.where(rng.random(n_rows) < 0.5, "big", "small")
        return df

    exog = None if exog_kind == "none" else {k: make_exog() for k in names}
    return series, exog
