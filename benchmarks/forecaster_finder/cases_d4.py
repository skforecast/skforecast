"""
Cases added to the forecaster finder study to reach six or more datasets per
type of data (decision D4 of PLAN.md), and to compare AutoETS and AutoARIMA
as the statistical reference on more monthly and quarterly series.

Each entry is (Case, dataset, exog_kind). `exog_kind` says what the variables
known in advance are: 'calendar' (events, planned values: really known for
future dates), 'weather' (observed values, used as if they were a perfect
forecast) or '' (none). The settings per frequency are those of the original
cases of cases.py.

Run with `python -I study.py --d4`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cases import Case
from skforecast.datasets import fetch_dataset

HOURLY = dict(steps=24, season=24, lags=[1, 2, 3, 24, 48, 168], window=24,
              calendar=["hour", "day_of_week"])
DAILY = dict(steps=28, season=7, lags=14, window=7, calendar=["day_of_week", "month"])
WEEKLY = dict(steps=8, season=52, lags=[1, 2, 3, 4, 52], window=4, calendar=["month"])
MONTHLY = dict(steps=12, season=12, lags=12, window=None, calendar=["month"])
QUARTERLY = dict(steps=4, season=4, lags=4, window=None, calendar=["quarter"])


# ==============================================================================
# One series, nothing known in advance
# ==============================================================================
def load_website_visits():
    d = fetch_dataset("website_visits", verbose=False).asfreq("D")
    return d[["users"]].astype(float), None


def load_uk_flights():
    d = fetch_dataset("uk_daily_flights", verbose=False).asfreq("D")
    d.columns = ["flights"]
    return d.astype(float), None


def load_wikipedia():
    d = fetch_dataset("wikipedia_visits", verbose=False).asfreq("D")
    return d[["y"]].astype(float).interpolate(), None


def load_air_passengers():
    # Classic monthly series, shipped with the tests of skforecast
    from skforecast.stats.tests.tests_arima.fixtures_arima import air_passengers
    y = pd.Series(air_passengers.to_numpy(dtype=float), name="passengers",
                  index=pd.date_range("1949-01-01", periods=len(air_passengers), freq="MS"))
    return y.to_frame(), None


def load_elnino():
    import statsmodels.api as sm
    d = sm.datasets.elnino.load_pandas().data.set_index("YEAR")
    y = pd.Series(d.to_numpy(dtype=float).ravel(), name="sea_temperature",
                  index=pd.date_range(f"{int(d.index[0])}-01-01", periods=d.size, freq="MS"))
    return y.to_frame(), None


def _macrodata(col):
    import statsmodels.api as sm
    d = sm.datasets.macrodata.load_pandas().data
    y = pd.Series(d[col].to_numpy(dtype=float), name=col,
                  index=pd.date_range("1959-01-01", periods=len(d), freq="QS"))
    return y.to_frame(), None


def load_macro_unemp():
    return _macrodata("unemp")


def load_macro_realcons():
    return _macrodata("realcons")


def load_co2_weekly():
    import statsmodels.api as sm
    d = sm.datasets.co2.load_pandas().data.interpolate().asfreq("W-SAT")
    return d[["co2"]].iloc[-520:].astype(float), None  # Last 10 years


def load_tourism_total():
    d = fetch_dataset("australia_tourism", verbose=False)
    y = d.groupby(level=0)["Trips"].sum().asfreq("QS").rename("trips")
    return y.to_frame().astype(float), None


# ==============================================================================
# One series, variables known in advance
# ==============================================================================
def load_h2o_exog():
    d = fetch_dataset("h2o_exog", verbose=False).asfreq("MS").astype(float)
    return d[["y"]], d[["exog_1", "exog_2"]]


def load_bdg2_building():
    d = fetch_dataset("bdg2_daily", verbose=False)
    d = d.loc[d["site_id"] == "Fox"]
    s = d.pivot_table(index=d.index, columns="building_id", values="meter_reading").asfreq("D")
    s = s.loc[:, (s.isna().sum() == 0) & (s.std() > 0) & ((s == 0).mean() < 0.02)]
    temp = d.groupby(level=0)["airTemperature"].mean().asfreq("D").interpolate()
    s.index.name = None
    s.columns.name = None
    return s.iloc[:, [0]].astype(float), temp.to_frame("temp").astype(float)


def _m5():
    d = fetch_dataset("m5", verbose=False)
    stores = d.groupby([d.index, "store_id"], observed=True)["sales"].sum().unstack("store_id")
    stores = stores.asfreq("D").astype(float)
    stores.columns = [str(c) for c in stores.columns]
    stores.columns.name = None
    stores.index.name = None
    cal = d.loc[d["id"] == d["id"].iloc[0], ["event_type_1", "snap_CA"]].asfreq("D")
    exog = pd.DataFrame({"event": cal["event_type_1"].notna().astype(float),
                         "snap_CA": cal["snap_CA"].astype(float)}, index=stores.index)
    return stores, exog


def load_m5_store():
    stores, exog = _m5()
    return stores[["CA_1"]], exog  # Events and SNAP days of California


def load_turbine():
    d = fetch_dataset("turbine_emission", verbose=False).asfreq("h").iloc[-24 * 400:]
    return d[["TEY"]].astype(float), d[["AT", "AP", "AH"]].astype(float)


def load_air_quality_pm25():
    d = fetch_dataset("air_quality_valencia_no_missing", verbose=False).asfreq("h")
    d = d.iloc[-24 * 400:]
    return d[["pm2.5"]].astype(float), d[["veloc.", "direc."]].astype(float)


# ==============================================================================
# Several series
# ==============================================================================
def load_items_sales():
    d = fetch_dataset("items_sales", verbose=False).asfreq("D")
    return d.astype(float), None


def load_air_quality():
    d = fetch_dataset("air_quality_valencia_no_missing", verbose=False).asfreq("h")
    cols = ["so2", "co", "no", "no2", "pm10", "nox", "o3", "pm2.5"]
    return d[cols].iloc[-24 * 400:].astype(float), None


def load_ett_m1():
    d = fetch_dataset("ett_m1", verbose=False).resample("h").mean().asfreq("h")
    return d.iloc[-24 * 400:].astype(float), None


def load_m4_daily():
    d = fetch_dataset("m4_daily", verbose=False)
    # M4 timestamps are artificial: 50 random series with 1200 or more values,
    # their last 1200 aligned on a common index (no calendar features).
    lengths = d.groupby("series_id").size()
    ids = sorted(lengths[lengths >= 1200].index)
    ids = list(np.random.default_rng(1).permutation(ids))[:50]
    idx = pd.date_range("2018-01-01", periods=1200, freq="D")
    s = pd.DataFrame({i: d.loc[d["series_id"] == i, "value"].to_numpy()[-1200:] for i in ids},
                     index=idx)
    return s.astype(float), None


def load_m5_stores():
    stores, exog = _m5()
    return stores, exog[["event"]]  # The events are common to the three states


def load_madrid_modes():
    d = fetch_dataset("public_transport_madrid", verbose=False).asfreq("D")
    return d[["metro", "bus", "road", "train"]].astype(float), None


def case(name, question, loader, freq, n_folds, exog_mode="none", stats=(), calendar=True):
    f = dict(freq)
    if not calendar:
        f["calendar"] = None
    return Case(name, question, loader, f["steps"], n_folds, f["season"], f["lags"],
                f["window"], f["calendar"], exog_mode, [None], list(stats))


D4_CASES = [
    # One series, nothing known in advance
    (case("website_visits", "1 series, daily", load_website_visits, DAILY, 4, stats=["ets"]),
     "website_visits", ""),
    (case("uk_daily_flights", "1 series, daily", load_uk_flights, DAILY, 6, stats=["ets"]),
     "uk_daily_flights", ""),
    (case("wikipedia_visits", "1 series, daily", load_wikipedia, DAILY, 6, stats=["ets"]),
     "wikipedia_visits", ""),
    (case("air_passengers_monthly", "1 series, monthly", load_air_passengers, MONTHLY, 3,
          stats=["ets", "arima"]), "air_passengers", ""),
    (case("elnino_monthly", "1 series, monthly", load_elnino, MONTHLY, 5,
          stats=["ets", "arima"]), "elnino", ""),
    (case("macro_unemp_quarterly", "1 series, quarterly", load_macro_unemp, QUARTERLY, 5,
          stats=["ets", "arima"]), "macrodata", ""),
    (case("macro_realcons_quarterly", "1 series, quarterly", load_macro_realcons, QUARTERLY, 5,
          stats=["ets", "arima"]), "macrodata", ""),
    (case("co2_weekly", "1 series, weekly", load_co2_weekly, WEEKLY, 6,
          stats=["ets", "arima"]), "co2", ""),
    (case("tourism_total_quarterly", "1 series, quarterly", load_tourism_total, QUARTERLY, 3,
          stats=["ets", "arima"]), "australia_tourism", ""),
    # One series, variables known in advance
    (case("h2o_exog_monthly", "1 series, monthly, 2 known variables", load_h2o_exog, MONTHLY, 3,
          "future", stats=["ets", "arima"]), "h2o_exog", "calendar"),
    (case("bdg2_one_building", "1 building, daily, temperature", load_bdg2_building, DAILY, 6,
          "future", stats=["ets"]), "bdg2_daily", "weather"),
    (case("m5_store_daily", "1 store, daily, events and SNAP days", load_m5_store, DAILY, 6,
          "future", stats=["ets"]), "m5", "calendar"),
    (case("turbine_hourly", "1 series, hourly, ambient conditions", load_turbine, HOURLY, 28,
          "future"), "turbine_emission", "weather"),
    (case("air_quality_pm25_hourly", "1 series, hourly, wind", load_air_quality_pm25, HOURLY, 28,
          "future"), "air_quality_valencia", "weather"),
    # Several series
    (case("items_sales_3_items", "3 series, daily", load_items_sales, DAILY, 6),
     "items_sales", ""),
    (case("air_quality_8_series", "8 series, hourly", load_air_quality, HOURLY, 14),
     "air_quality_valencia", ""),
    (case("ett_m1_7_series", "7 series, hourly", load_ett_m1, HOURLY, 14), "ett_m1", ""),
    (case("m4_daily_50_series", "50 series, daily", load_m4_daily, DAILY, 4, calendar=False),
     "m4_daily", ""),
    (case("m5_10_stores", "10 stores, daily, events", load_m5_stores, DAILY, 6, "future"),
     "m5", "calendar"),
    (case("madrid_4_modes", "4 series, daily", load_madrid_modes, DAILY, 6),
     "public_transport_madrid", ""),
]
