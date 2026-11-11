"""
QW-11: which calendar features add accuracy, by frequency and by model family.

Every feature set is built from the DatetimeIndex and passed as `exog`, so features
that `CalendarFeatures` does not have yet can be compared on equal terms. Each
configuration is compared with a baseline that uses what `CalendarFeatures` offers
today, with the encoding that suits each model: one-hot for Ridge (a linear model
needs one coefficient per category) and raw integers for LightGBM.

Metric: MAE of a backtesting with refit=False; reported as % change of MAE with
respect to the baseline of the same series and model (negative = better).

Run: python dev/roadmap_0.27.0/repro/QW-11-calendar-analysis.py
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
import lightgbm
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import skforecast
from skforecast.datasets import fetch_dataset, load_demo_dataset
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import create_calendar_features
from skforecast.recursive import ForecasterRecursive

warnings.filterwarnings("ignore")


# Series
# ==============================================================================
def load_series():
    series = {"hourly": {}, "daily": {}, "monthly": {}}
    holidays = {}

    bike = fetch_dataset("bike_sharing", verbose=False)
    series["hourly"]["bike_sharing"] = bike["users"].astype(float)
    holidays["bike_sharing"] = bike["holiday"].astype(int)
    bike_d = bike["users"].resample("D").sum()
    series["daily"]["bike_sharing_daily"] = bike_d
    holidays["bike_sharing_daily"] = bike["holiday"].resample("D").max().astype(int)

    vic = fetch_dataset("vic_electricity", verbose=False)
    vic_h = vic["Demand"].resample("h").sum().loc["2012-01-01":"2014-12-30"]
    series["hourly"]["vic_electricity"] = vic_h
    holidays["vic_electricity"] = vic["Holiday"].resample("h").max().astype(int).loc[vic_h.index]
    vic_d = vic["Demand"].resample("D").sum().loc["2012-01-01":"2014-12-30"]
    series["daily"]["vic_electricity_daily"] = vic_d
    holidays["vic_electricity_daily"] = vic["Holiday"].resample("D").max().astype(int).loc[vic_d.index]

    air = fetch_dataset("air_quality_valencia_no_missing", verbose=False)
    series["hourly"]["air_quality_no2"] = air["no2"].loc["2022-01-01":].astype(float)

    bdg_h = fetch_dataset("bdg2_hourly_sample", verbose=False)
    for col in bdg_h.columns:
        series["hourly"][f"bdg2_hourly_{col}"] = bdg_h[col].astype(float)

    madrid = fetch_dataset("public_transport_madrid", verbose=False)
    for col in ["metro", "bus", "train"]:
        series["daily"][f"madrid_{col}"] = madrid[col].astype(float)

    series["daily"]["bicimad"] = fetch_dataset("bicimad", verbose=False)["users"].loc["2017-01-01":"2019-12-31"].astype(float)
    series["daily"]["wikipedia"] = fetch_dataset("wikipedia_visits", verbose=False)["y"].astype(float)

    store = fetch_dataset("store_sales", verbose=False)
    total = store.groupby(level=0)["sales"].sum().asfreq("D").astype(float)
    series["daily"]["store_sales_total"] = total

    series["monthly"]["h2o"] = load_demo_dataset().astype(float)
    fuel = fetch_dataset("fuel_consumption", verbose=False)
    for col in ["Gasolinas", "Gasoleos", "GLPs"]:
        series["monthly"][f"fuel_{col}"] = fuel[col].loc["1990-01-01":].astype(float)
    series["monthly"]["retail_australia"] = fetch_dataset("expenditures_australia", verbose=False)["turnover"].astype(float)

    for freq in series:
        for name, y in series[freq].items():
            y = y.copy()
            y.name = "y"
            series[freq][name] = y.interpolate(limit_direction="both")
    return series, holidays


# Calendar feature builders (all from the DatetimeIndex)
# ==============================================================================
def onehot(values, categories, prefix, index):
    return pd.get_dummies(
        pd.Categorical(values, categories=categories), prefix=prefix, dtype=int
    ).set_axis(index)


def fourier(values, period, order, prefix, index):
    cols = {}
    for k in range(1, order + 1):
        cols[f"{prefix}_sin{k}"] = np.sin(2 * np.pi * k * values / period)
        cols[f"{prefix}_cos{k}"] = np.cos(2 * np.pi * k * values / period)
    return pd.DataFrame(cols, index=index)


def feature(name, idx, linear, holiday=None):
    """One calendar feature, encoded for a linear model (one-hot) or for trees (int)."""
    raw = {
        "hour": (idx.hour, range(24)),
        "day_of_week": (idx.dayofweek, range(7)),
        "hour_of_week": (idx.dayofweek * 24 + idx.hour, range(168)),
        "month": (idx.month, range(1, 13)),
        "day_of_month": (idx.day, range(1, 32)),
        "week_of_month": ((idx.day - 1) // 7 + 1, range(1, 6)),
    }
    if name in raw:
        values, cats = raw[name]
        if linear:
            return onehot(values, cats, name, idx)
        return pd.DataFrame({name: np.asarray(values)}, index=idx)
    if name.startswith("doy_fourier"):
        order = int(name.removeprefix("doy_fourier"))
        return fourier(np.asarray(idx.dayofyear), 365.25, order, "doy", idx)
    if name.startswith("dow_fourier"):
        order = int(name.removeprefix("dow_fourier"))
        return fourier(np.asarray(idx.dayofweek), 7, order, "dow", idx)
    if name.startswith("hour_fourier"):
        order = int(name.removeprefix("hour_fourier"))
        return fourier(np.asarray(idx.hour), 24, order, "hour", idx)
    if name == "is_month_start":
        return pd.DataFrame({name: idx.is_month_start.astype(int)}, index=idx)
    if name == "is_month_end":
        return pd.DataFrame({name: idx.is_month_end.astype(int)}, index=idx)
    if name == "days_to_month_end":
        values = np.asarray(idx.days_in_month - idx.day)
        if linear:
            return onehot(np.minimum(values, 7), range(8), name, idx)
        return pd.DataFrame({name: values}, index=idx)
    if name == "days_in_month":
        return pd.DataFrame({name: idx.days_in_month.astype(int)}, index=idx)
    if name == "weekend_days_in_month":
        values = [
            int(np.isin(pd.date_range(d, periods=n, freq="D").dayofweek, [5, 6]).sum())
            for d, n in zip(idx, idx.days_in_month)
        ]
        return pd.DataFrame({name: values}, index=idx)
    if name == "weekday_counts":
        rows = [
            np.bincount(pd.date_range(d, periods=n, freq="D").dayofweek, minlength=7)
            for d, n in zip(idx, idx.days_in_month)
        ]
        return pd.DataFrame(rows, index=idx, columns=[f"n_dow_{i}" for i in range(7)])
    if name == "doy_int":
        return pd.DataFrame({name: np.asarray(idx.dayofyear)}, index=idx)
    if name.startswith("spline:"):
        # Existing CalendarFeatures spline encoding: "spline:<feature>" or "spline:<feature>:<n_knots>"
        parts = name.split(":")
        spline_kwargs = {"n_knots": int(parts[2])} if len(parts) == 3 else None
        return create_calendar_features(
            idx, features=[parts[1]], encoding="spline", spline_kwargs=spline_kwargs
        ).add_prefix("spl_")
    if name == "holiday":
        return pd.DataFrame({name: holiday.to_numpy()}, index=idx)
    raise ValueError(name)


def build(features, idx, linear, holiday=None):
    return pd.concat([feature(f, idx, linear, holiday) for f in features], axis=1)


# Configurations by frequency: (name, features). The first one is the baseline.
# ==============================================================================
CONFIGS = {
    "hourly": [
        ("base: day_of_week + hour", ["day_of_week", "hour"]),
        ("hour_of_week instead of dow + hour", ["hour_of_week"]),
        ("base + hour_of_week", ["day_of_week", "hour", "hour_of_week"]),
        ("base + month", ["day_of_week", "hour", "month"]),
        ("base + doy_fourier3", ["day_of_week", "hour", "doy_fourier3"]),
        ("base + holiday", ["day_of_week", "hour", "holiday"]),
        ("ENC: dow + hour fourier K=1 (cyclical today)", ["dow_fourier1", "hour_fourier1"]),
        ("ENC: dow K=3 + hour K=6 fourier", ["dow_fourier3", "hour_fourier6"]),
        ("ENC: dow + hour spline (existing)", ["spline:day_of_week", "spline:hour"]),
    ],
    "daily": [
        ("base: day_of_week + month", ["day_of_week", "month"]),
        ("base + is_month_start", ["day_of_week", "month", "is_month_start"]),
        ("base + is_month_end", ["day_of_week", "month", "is_month_end"]),
        ("base + days_to_month_end", ["day_of_week", "month", "days_to_month_end"]),
        ("base + day_of_month", ["day_of_week", "month", "day_of_month"]),
        ("base + week_of_month", ["day_of_week", "month", "week_of_month"]),
        ("base + holiday", ["day_of_week", "month", "holiday"]),
        ("dow + doy_fourier1 (cyclical today)", ["day_of_week", "doy_fourier1"]),
        ("dow + doy_fourier3", ["day_of_week", "doy_fourier3"]),
        ("dow + doy_fourier6", ["day_of_week", "doy_fourier6"]),
        ("dow + month + doy_fourier3", ["day_of_week", "month", "doy_fourier3"]),
        ("ENC: dow fourier K=1 + month (cyclical dow today)", ["dow_fourier1", "month"]),
        ("dow + day_of_year int (existing)", ["day_of_week", "doy_int"]),
        ("dow + day_of_year spline 13 knots (existing)", ["day_of_week", "spline:day_of_year:13"]),
        ("dow + month + day_of_year spline 13 knots", ["day_of_week", "month", "spline:day_of_year:13"]),
    ],
    "monthly": [
        ("base: month", ["month"]),
        ("base + days_in_month", ["month", "days_in_month"]),
        ("base + weekend_days_in_month", ["month", "weekend_days_in_month"]),
        ("base + weekday_counts (trading days)", ["month", "weekday_counts"]),
    ],
}

SETUP = {
    # lags, steps, n_test
    "hourly": (list(range(1, 25)) + [168], 24, 24 * 56),
    "daily": (list(range(1, 15)), 7, 364),
    "monthly": (list(range(1, 13)), 12, 48),
}

MODELS = {
    "Ridge": lambda: Ridge(alpha=1.0),
    "LightGBM": lambda: LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=123, verbose=-1
    ),
}


def run(y, exog, model, lags, steps, n_test):
    linear = model == "Ridge"
    forecaster = ForecasterRecursive(
        estimator=MODELS[model](),
        lags=lags,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler() if linear else None,
    )
    n_test = min(n_test, len(y) // 4 // steps * steps)
    cv = TimeSeriesFold(steps=steps, initial_train_size=len(y) - n_test, refit=False, verbose=False)
    metric, _ = backtesting_forecaster(
        forecaster, y=y, exog=exog, cv=cv, metric="mean_absolute_error", show_progress=False
    )
    return metric.iloc[0, 0]


def main():
    print(
        f"Python {sys.version.split()[0]}, skforecast {skforecast.__version__}, "
        f"numpy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}, "
        f"lightgbm {lightgbm.__version__}"
    )
    series, holidays = load_series()
    rows = []
    for freq, configs in CONFIGS.items():
        lags, steps, n_test = SETUP[freq]
        for name, y in series[freq].items():
            for model in MODELS:
                linear = model == "Ridge"
                for cfg_name, feats in configs:
                    if "holiday" in feats and name not in holidays:
                        continue
                    exog = build(feats, y.index, linear, holidays.get(name))
                    start = time.perf_counter()
                    mae = run(y, exog, model, lags, steps, n_test)
                    rows.append(dict(freq=freq, series=name, model=model, config=cfg_name,
                                     mae=mae, seconds=time.perf_counter() - start))
            print(f"done {freq} {name}", flush=True)

    res = pd.DataFrame(rows)
    base = res.groupby(["freq", "series", "model"]).apply(
        lambda g: g.loc[g["config"].str.startswith("base:"), "mae"].iloc[0], include_groups=False
    ).rename("base_mae")
    res = res.join(base, on=["freq", "series", "model"])
    res["pct"] = 100 * (res["mae"] / res["base_mae"] - 1)
    res.to_csv("dev/roadmap_0.27.0/repro/QW-11-calendar-analysis.csv", index=False)

    pd.set_option("display.width", 200)
    for freq in CONFIGS:
        sub = res[res["freq"] == freq]
        summary = sub.groupby(["config", "model"], sort=False)["pct"].agg(
            median="median",
            best="min",
            worst="max",
            n_better=lambda s: int((s < -1).sum()),
            n_worse=lambda s: int((s > 1).sum()),
            n="count",
        ).round(1)
        print(f"\n=== {freq}: % change of MAE vs baseline (negative = better; better/worse = beyond 1%)")
        print(summary.to_string())
    print("\n=== Per series (pct)")
    print(res.pivot_table(index=["freq", "config"], columns=["model", "series"], values="pct", sort=False).round(1).to_string())


if __name__ == "__main__":
    main()
