"""
BUG-search-short-series, experiment 7: share of the test points that each
window size leaves out in the whole M5 dataset (30,490 series), with no model.

A point of a fold is left out when its series has some history before the
fold but `window_size` values or fewer (frontier of #1418). The launch date of
a product is its first day with `sell_price`. Two backtesting periods: the last
84 days (3 folds of 28) and the second half of 2014 (6 folds of 28).

Downloads M5 with fetch_dataset (~116 MB).
"""
import warnings
import pandas as pd
from skforecast.datasets import fetch_dataset

warnings.simplefilter("ignore")
df = fetch_dataset("m5", verbose=False).reset_index()[["id", "datetime", "sell_price"]]
launch = df[df["sell_price"].notna()].groupby("id", observed=True)["datetime"].min()
end = df["datetime"].max()
del df
print(f"{len(launch)} series, data until {end.date()}")


def dropped_share(test_start, n_folds, steps=28, windows=(7, 28, 56, 90, 365)):
    out = {}
    for w in windows:
        total = dropped = 0
        for k in range(n_folds):
            fold_start = test_start + pd.Timedelta(days=k * steps)
            history = (fold_start - launch).dt.days
            total += (history > 0).sum() * steps
            dropped += ((history > 0) & (history <= w)).sum() * steps
        out[f"lags={w}"] = f"{dropped / total:.2%}"
    return out


print("last 84 days      :", dropped_share(end - pd.Timedelta(days=83), n_folds=3))
print("second half 2014  :", dropped_share(pd.Timestamp("2014-07-01"), n_folds=6))
t = end - pd.Timedelta(days=83)
young = ((launch > t - pd.Timedelta(days=365)) & (launch <= t)).mean()
print(f"series launched in the year before {t.date()}: {young:.2%}")
