# Unit test generate_gas_station_panel, generate_station, StationConfig,
# SeasonalEffectsConfig
# ==============================================================================
import re
import warnings
import pytest
import numpy as np
import pandas as pd

from .._synthetic_data import (
    StationConfig,
    SeasonalEffectsConfig,
    generate_gas_station_panel,
    generate_station,
    _easter,
    _get_holidays,
)

TARGET_COLS = [
    "liters_diesel_sold", "liters_gasoline_sold",
    "store_revenue_euros", "carwash_revenue_euros",
]


def test_generate_gas_station_panel_ValueError_when_output_invalid():
    """
    Test that generate_gas_station_panel raises ValueError when `output` is
    not 'long' or 'wide'.
    """
    err_msg = re.escape("`output` must be 'long' or 'wide'.")
    with pytest.raises(ValueError, match=err_msg):
        generate_gas_station_panel(
            "2023-01-01", "2023-01-31", stations=1, output="invalid"
        )


def test_generate_gas_station_panel_output_index_and_columns():
    """
    Test that the long-format panel has a frequency-aware DatetimeIndex and
    the expected set of columns.
    """
    df = generate_gas_station_panel(
        "2023-01-01", "2023-01-31", stations=2, seed=42
    )

    expected_cols = {
        "station_id", "liters_diesel_sold", "liters_gasoline_sold",
        "store_revenue_euros", "carwash_revenue_euros", "price_diesel",
        "price_gasoline", "temperature_c", "precipitation_mm", "is_raining",
        "is_national_holiday", "is_weekend", "is_open", "is_highway_station",
        "is_anomaly",
    }
    assert set(df.columns) == expected_cols
    assert df["station_id"].nunique() == 2
    first = df[df["station_id"] == "station_01"]
    assert first.index.inferred_freq == "h"


def test_generate_gas_station_panel_targets_non_negative():
    """
    Test that all target columns are non-negative, including after anomaly
    injection.
    """
    df = generate_gas_station_panel(
        "2023-01-01", "2023-12-31", stations=3, seed=42
    )
    assert (df[TARGET_COLS] >= 0).all().all()


def test_generate_gas_station_panel_closed_hours_have_zero_sales():
    """
    Test that stations configured to close at night have exactly zero sales
    during closed hours.
    """
    df = generate_gas_station_panel(
        "2023-01-01", "2023-12-31", stations=3, seed=42
    )
    closed = df["is_open"] == 0
    assert closed.any()
    assert (df.loc[closed, TARGET_COLS].to_numpy() == 0).all()


def test_generate_gas_station_panel_holidays_present_each_year():
    """
    Test that at least one national holiday hour is flagged in each calendar
    year, using the default country ('ES').
    """
    df = generate_gas_station_panel(
        "2022-01-01", "2023-12-31", stations=1, seed=42
    )
    for year in (2022, 2023):
        mask = df.index.year == year
        assert df.loc[mask, "is_national_holiday"].sum() > 0


def test_generate_gas_station_panel_deterministic_with_same_seed():
    """
    Test that two calls with the same seed produce an identical panel.
    """
    df1 = generate_gas_station_panel("2023-01-01", "2023-03-31", stations=2, seed=7)
    df2 = generate_gas_station_panel("2023-01-01", "2023-03-31", stations=2, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_gas_station_panel_wide_vs_long_shapes_consistent():
    """
    Test that the wide-format panel has one row per timestamp and
    n_stations * n_feature_columns columns, consistent with the long-format
    panel.
    """
    long_df = generate_gas_station_panel(
        "2023-01-01", "2023-01-31", stations=2, output="long", seed=42
    )
    wide_df = generate_gas_station_panel(
        "2023-01-01", "2023-01-31", stations=2, output="wide", seed=42
    )

    n_stations = long_df["station_id"].nunique()
    n_feature_cols = long_df.shape[1] - 1  # exclude station_id
    assert wide_df.shape[0] == long_df[long_df["station_id"] == "station_01"].shape[0]
    assert wide_df.shape[1] == n_stations * n_feature_cols


def test_station_config_accepts_plain_dict_for_backward_compatibility():
    """
    Test that generate_gas_station_panel accepts plain dicts (with
    StationConfig fields) in addition to StationConfig instances.
    """
    dict_config = {
        "station_id": "custom_station",
        "is_highway": True,
        "base_diesel_per_hr": 600.0,
    }
    object_config = StationConfig(
        station_id="custom_station", is_highway=True, base_diesel_per_hr=600.0
    )

    df_from_dict = generate_gas_station_panel(
        "2023-01-01", "2023-01-31", stations=[dict_config], seed=42
    )
    df_from_object = generate_gas_station_panel(
        "2023-01-01", "2023-01-31", stations=[object_config], seed=42
    )
    pd.testing.assert_frame_equal(df_from_dict, df_from_object)


def test_hourly_temperature_diurnal_cycle():
    """
    Test that mean hourly temperature is coldest near dawn and hottest in
    mid-afternoon.
    """
    df = generate_gas_station_panel(
        "2023-01-01", "2023-12-31", stations=1, seed=42
    )
    by_hour = df.groupby(df.index.hour)["temperature_c"].mean()
    assert by_hour.idxmin() in (2, 3, 4, 5)
    assert by_hour.idxmax() in (14, 15, 16)


def test_sales_lag1_autocorrelation_is_positive():
    """
    Test that total fuel sales show positive lag-1 autocorrelation, i.e. the
    AR(1) demand-momentum factor is having an effect.
    """
    df = generate_gas_station_panel(
        "2023-01-01", "2023-12-31", stations=1, seed=42
    )
    total = df["liters_diesel_sold"] + df["liters_gasoline_sold"]
    assert total.autocorr(lag=1) > 0.1


def test_inject_anomalies_flags_at_least_one_hour():
    """
    Test that at least one hour is flagged as an anomaly over a full year.
    """
    df = generate_gas_station_panel(
        "2023-01-01", "2023-12-31", stations=1, seed=42
    )
    assert df["is_anomaly"].sum() > 0


def test_get_holidays_uses_country_argument():
    """
    Test that _get_holidays returns different, non-empty holiday sets for
    different countries when the `holidays` library is installed.
    """
    pytest.importorskip("holidays")
    es_holidays = _get_holidays("ES", [2023])
    fr_holidays = _get_holidays("FR", [2023])
    assert len(es_holidays) > 0
    assert len(fr_holidays) > 0
    assert es_holidays != fr_holidays


def test_get_holidays_fallback_warns_for_non_spain_country_without_holidays_lib(
    monkeypatch,
):
    """
    Test that _get_holidays falls back to a minimal universal holiday set
    and warns when the `holidays` library is unavailable and the country is
    not Spain.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "holidays":
            raise ImportError("holidays not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    warn_msg = re.escape("falling back to a minimal universal holiday set")
    with pytest.warns(UserWarning, match=warn_msg):
        result = _get_holidays("DE", [2023])

    easter_2023 = _easter(2023)
    from datetime import date, timedelta
    expected = {
        date(2023, 1, 1), date(2023, 12, 25),
        easter_2023, easter_2023 - timedelta(days=2),
    }
    assert result == expected


def test_get_holidays_spain_fallback_does_not_warn_without_holidays_lib(monkeypatch):
    """
    Test that _get_holidays for country='ES' falls back silently (no
    warning) to the hardcoded Spanish holiday table when the `holidays`
    library is unavailable.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "holidays":
            raise ImportError("holidays not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _get_holidays("ES", [2023])

    assert len(result) > 0


def test_seasonal_effects_config_overrides_change_output():
    """
    Test that overriding `summer_months` in SeasonalEffectsConfig changes
    which months receive the summer demand multiplier.
    """
    default_effects = SeasonalEffectsConfig()
    custom_effects = SeasonalEffectsConfig(summer_months=(6,))

    config = StationConfig(station_id="s1", is_highway=True)
    rng_default = np.random.default_rng(123)
    rng_custom = np.random.default_rng(123)
    holidays_set = _get_holidays("ES", [2023])

    df_default = generate_station(
        "2023-06-01", "2023-06-30", config, rng_default, holidays_set, default_effects
    )
    df_custom = generate_station(
        "2023-06-01", "2023-06-30", config, rng_custom, holidays_set, custom_effects
    )

    assert not df_default["liters_gasoline_sold"].equals(df_custom["liters_gasoline_sold"])


def test_seasonal_effects_config_defaults_preserve_expected_patterns():
    """
    Test that the default (Spain-tuned) SeasonalEffectsConfig still produces
    the expected weekly travel patterns: a Sunday diesel dip (heavy-truck
    ban) and a Friday-afternoon highway gasoline boost.
    """
    df = generate_gas_station_panel(
        "2023-01-01", "2023-12-31",
        stations=[StationConfig(station_id="highway_1", is_highway=True)],
        seed=42,
    )

    diesel_by_dow = df.groupby(df.index.dayofweek)["liters_diesel_sold"].mean()
    assert diesel_by_dow.idxmin() == 6  # Sunday

    friday_afternoon = df[(df.index.dayofweek == 4) & (df.index.hour.isin(range(15, 22)))]
    other_hours = df[~((df.index.dayofweek == 4) & (df.index.hour.isin(range(15, 22))))]
    assert friday_afternoon["liters_gasoline_sold"].mean() > other_hours["liters_gasoline_sold"].mean()
