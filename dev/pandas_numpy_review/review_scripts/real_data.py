"""
Real cases for the prototype test: the backtestings of three articles of
cienciadedatos.net, rebuilt with their data, preprocessing, forecasters and
folds (skforecast 0.27.0 in the articles).

- `store`: py44 (multi-series forecasting), 50 items of store 2 of the Kaggle
  store item demand data (`store_sales` in skforecast-datasets), wide
  DataFrame, HistGradientBoosting, no exog.
- `demo`: py44, `demo_multi_series` (5 series of different lengths) with exog
  in a dict and some columns dropped, HistGradientBoosting.
- `bdg2`: py59 (scalable forecasting models), `bdg2_daily`, 1578 buildings,
  22 exog (two text categoricals and a `category`), LightGBM with the best
  parameters of the article's search (600 trees, lags 1 to 62).
- `bdg2_small`: the same data with the forecaster as created before the search
  (LightGBM by default, lags 14).
- `bdg2_steps1`, `bdg2_small_steps1`: variants that are not in the article,
  the same backtesting with one step per fold (a daily evaluation, 61 folds).
- `bike`: py39 (XGBoost, LightGBM and CatBoost), single series, hourly, with
  exog (`bike_sharing_extended_features`, the precomputed version of the
  article's features), LightGBM with categorical `weather`.

Usage:
    python real_data.py download DIR    # once, with pyarrow (bdg2 is parquet)
Then set REAL_DATA_DIR=DIR for `time_real_cases.py`.

LightGBM with `n_jobs=None` (the articles) trains with all the physical cores
and ignores `OMP_NUM_THREADS`: its results then change in the last bits from
one run to the next (also two runs of the current code). `LGBM_N_JOBS=1` fixes
the threads, to compare outputs exactly. `REAL_N_SERIES=N` keeps a fixed random
sample of N buildings in the `bdg2` cases.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LGBM_N_JOBS = (
    {"n_jobs": int(os.environ["LGBM_N_JOBS"])} if os.environ.get("LGBM_N_JOBS") else {}
)
RAW = "https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/"
FILES = {
    "store_sales.csv": RAW + "store_sales.csv",
    "demo_multi_series.csv": RAW + "demo_multi_series.csv",
    "demo_multi_series_exog.csv": RAW + "demo_multi_series_exog.csv",
    "bike_sharing_extended_features.csv": RAW + "bike_sharing_extended_features.csv",
}
BDG2 = "https://huggingface.co/datasets/skforecast/bdg2_daily/resolve/main/bdg2_daily.parquet"


def download(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        pd.read_csv(url).to_csv(out / name, index=False)
    # NOTE: Saved as csv with its dtypes, so that pandas 3 without pyarrow can
    # read it.
    data = pd.read_parquet(BDG2)
    data.to_csv(out / "bdg2_daily.csv", index=False)
    dtypes = {c: str(t) for c, t in data.dtypes.items()}
    (out / "bdg2_daily_dtypes.json").write_text(json.dumps(dtypes))


def _dir():
    return Path(os.environ["REAL_DATA_DIR"])


def case_store():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    from skforecast.model_selection import TimeSeriesFold
    from skforecast.preprocessing import RollingFeatures
    from skforecast.recursive import ForecasterRecursiveMultiSeries

    data = pd.read_csv(_dir() / "store_sales.csv")
    data = data[(data["store"] == 2) & (data["item"].isin(data.item.unique()))].copy()
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
    data = pd.pivot_table(data=data, values="sales", index="date", columns="item")
    data.columns.name = None
    data.columns = [f"item_{col}" for col in data.columns]
    data = data.asfreq("1D")
    end_train, end_val = "2016-05-31 23:59:00", "2017-05-31 23:59:00"
    n_train_val = len(data.loc[:end_train]) + len(data.loc[end_train:end_val])

    def forecaster():
        return ForecasterRecursiveMultiSeries(
            estimator=HistGradientBoostingRegressor(random_state=8523),
            lags=14,
            encoding="ordinal",
            transformer_series=StandardScaler(),
            window_features=RollingFeatures(stats=["mean", "min", "max"], window_sizes=7),
        )

    cv = TimeSeriesFold(steps=7, initial_train_size=n_train_val, refit=False, verbose=False)
    return dict(
        kind="ms", forecaster=forecaster, series=data, exog=None, cv=cv,
        kwargs=dict(metric="mean_absolute_error", add_aggregated_metric=False),
    )


def case_demo():
    from sklearn.ensemble import HistGradientBoostingRegressor

    from skforecast.model_selection import TimeSeriesFold
    from skforecast.preprocessing import (
        RollingFeatures, reshape_exog_long_to_dict, reshape_series_long_to_dict,
    )
    from skforecast.recursive import ForecasterRecursiveMultiSeries

    series = pd.read_csv(_dir() / "demo_multi_series.csv")
    exog = pd.read_csv(_dir() / "demo_multi_series_exog.csv")
    series["timestamp"] = pd.to_datetime(series["timestamp"])
    exog["timestamp"] = pd.to_datetime(exog["timestamp"])
    series_dict = reshape_series_long_to_dict(
        data=series, series_id="series_id", index="timestamp", values="value", freq="D"
    )
    exog_dict = reshape_exog_long_to_dict(
        data=exog, series_id="series_id", index="timestamp", freq="D"
    )
    exog_dict["id_1000"] = exog_dict["id_1000"].drop(columns=["air_temperature", "wind_speed"])
    exog_dict["id_1003"] = exog_dict["id_1003"].drop(columns=["cos_day_of_week"])

    def forecaster():
        return ForecasterRecursiveMultiSeries(
            estimator=HistGradientBoostingRegressor(random_state=123, max_depth=5),
            lags=14,
            window_features=RollingFeatures(stats=["mean", "min", "max"], window_sizes=7),
            encoding="ordinal",
            dropna_from_series=False,
        )

    cv = TimeSeriesFold(steps=7, initial_train_size="2016-07-31 23:59:00", refit=False, verbose=False)
    return dict(
        kind="ms", forecaster=forecaster, series=series_dict, exog=exog_dict, cv=cv,
        kwargs=dict(metric="mean_absolute_error", add_aggregated_metric=False,
                    suppress_warnings=True),
    )


def _bdg2_data():
    from feature_engine.timeseries.forecasting import WindowFeatures

    from skforecast.preprocessing import (
        CalendarFeatures, reshape_exog_long_to_dict, reshape_series_long_to_dict,
    )

    dtypes = json.loads((_dir() / "bdg2_daily_dtypes.json").read_text())
    data = pd.read_csv(_dir() / "bdg2_daily.csv")
    n_series = os.environ.get("REAL_N_SERIES")
    if n_series:
        # NOTE: A fixed random sample of buildings, for the variants that are
        # too slow with the 1578 of the article.
        ids = np.sort(data["building_id"].unique())
        keep = np.random.default_rng(123).choice(ids, size=int(n_series), replace=False)
        data = data[data["building_id"].isin(keep)].copy()
    for col, dtype in dtypes.items():
        if dtype == "category":
            data[col] = data[col].astype("category")
    # As `fetch_dataset('bdg2_daily')`: index `timestamp`.
    data.index = pd.to_datetime(data.pop("timestamp"), format="%Y-%m-%d")
    data = (
        data.groupby("building_id")
        .apply(lambda group: group.asfreq("D", fill_value=np.nan), include_groups=False)
        .reset_index(level=0)
    )
    first = data.drop_duplicates(subset=["building_id"])
    infrequent_types = (
        first["primaryspaceusage"].value_counts().loc[lambda x: x < 100].index.tolist()
    )
    infrequent_subtypes = (
        first["sub_primaryspaceusage"].value_counts().loc[lambda x: x < 50].index.tolist()
    )
    data["primaryspaceusage"] = np.where(
        data["primaryspaceusage"].isin(infrequent_types), "Other", data["primaryspaceusage"]
    )
    data["sub_primaryspaceusage"] = np.where(
        data["sub_primaryspaceusage"].isin(infrequent_subtypes), "Other",
        data["sub_primaryspaceusage"],
    )
    calendar_transformer = CalendarFeatures(
        features=["month", "week", "day_of_week"], encoding="cyclical",
        keep_original_columns=True,
    )
    data = calendar_transformer.fit_transform(data)
    wf_transformer = WindowFeatures(
        variables=["airTemperature", "windSpeed"], window=["7D", "14D"],
        functions=["mean"], freq="D", missing_values="ignore", drop_na=False,
    )
    data = (
        data.groupby("building_id")
        .apply(wf_transformer.fit_transform, include_groups=False)
        .reset_index(level=0)
    )
    exog_features = [
        "primaryspaceusage", "sub_primaryspaceusage", "timezone", "sqm",
        "airTemperature", "cloudCoverage", "dewTemperature", "precipDepth1HR",
        "precipDepth6HR", "seaLvlPressure", "windDirection", "windSpeed",
        "day_of_week_sin", "day_of_week_cos", "week_sin", "week_cos", "month_sin",
        "month_cos", "airTemperature_window_7D_mean", "windSpeed_window_7D_mean",
        "airTemperature_window_14D_mean", "windSpeed_window_14D_mean",
    ]
    series_dict = reshape_series_long_to_dict(
        data=data.reset_index(), series_id="building_id", index="timestamp",
        values="meter_reading", freq="D",
    )
    exog_dict = reshape_exog_long_to_dict(
        data=data[exog_features + ["building_id"]].reset_index(),
        series_id="building_id", index="timestamp", freq="D",
    )
    return series_dict, exog_dict


def case_bdg2(small=False, steps=7):
    from lightgbm import LGBMRegressor

    from skforecast.model_selection import TimeSeriesFold
    from skforecast.preprocessing import RollingFeatures
    from skforecast.recursive import ForecasterRecursiveMultiSeries

    series_dict, exog_dict = _bdg2_data()
    best_params = {
        "n_estimators": 600, "learning_rate": 0.011796607289691044, "num_leaves": 111,
        "max_depth": -1, "min_child_samples": 56, "colsample_bytree": 0.955142717070489,
        "subsample": 0.8785245072941625, "subsample_freq": 2,
        "reg_alpha": 0.05660535748888388, "reg_lambda": 1.1491702674993802,
    }

    def forecaster():
        params = {} if small else best_params
        return ForecasterRecursiveMultiSeries(
            estimator=LGBMRegressor(random_state=8520, verbose=-1, **params, **LGBM_N_JOBS),
            lags=14 if small else 62,
            window_features=RollingFeatures(stats=["mean", "min", "max"], window_sizes=7),
            categorical_features="auto",
            encoding="ordinal",
        )

    cv = TimeSeriesFold(initial_train_size="2017-10-31", steps=steps, refit=False, verbose=False)
    return dict(
        kind="ms", forecaster=forecaster, series=series_dict, exog=exog_dict, cv=cv,
        kwargs=dict(metric="mean_absolute_error", suppress_warnings=True),
    )


def case_bike():
    from lightgbm import LGBMRegressor

    from skforecast.model_selection import TimeSeriesFold
    from skforecast.preprocessing import RollingFeatures
    from skforecast.recursive import ForecasterRecursive

    data = pd.read_csv(_dir() / "bike_sharing_extended_features.csv")
    data.index = pd.to_datetime(data.pop("date_time"), format="%Y-%m-%d %H:%M:%S")
    data = data.asfreq("h").sort_index()
    data["weather"] = data["weather"].astype("category")
    # As the article: drop the rows without rolling features and the last day.
    data = data.iloc[7 * 24:-24]
    exog_features = data.filter(regex="_sin$|_cos$").columns.tolist()
    exog_features += data.filter(regex="^temp_.*").columns.tolist()
    exog_features += data.filter(regex="^holiday_.*").columns.tolist()
    exog_features += ["temp", "holiday", "weather"]
    data = data[["users"] + exog_features]
    data = data.astype({col: np.float32 for col in data.select_dtypes("number").columns})
    end_validation = "2012-08-31 23:59:00"

    def forecaster():
        return ForecasterRecursive(
            estimator=LGBMRegressor(random_state=15926, verbose=-1, **LGBM_N_JOBS),
            lags=72,
            window_features=RollingFeatures(stats=["mean"], window_sizes=24 * 3),
            categorical_features="auto",
        )

    cv = TimeSeriesFold(steps=36, initial_train_size=len(data.loc[:end_validation]), verbose=False)
    return dict(
        kind="single", forecaster=forecaster, y=data["users"], exog=data[exog_features],
        cv=cv, kwargs=dict(metric="mean_absolute_error"),
    )


CASES = {
    "store": case_store,
    "demo": case_demo,
    "bdg2": case_bdg2,
    "bdg2_small": lambda: case_bdg2(small=True),
    "bike": case_bike,
    # Variant: daily evaluation (one step per fold, 61 folds).
    "bdg2_steps1": lambda: case_bdg2(steps=1),
    "bdg2_small_steps1": lambda: case_bdg2(small=True, steps=1),
}

if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "download":
    download(sys.argv[2])
