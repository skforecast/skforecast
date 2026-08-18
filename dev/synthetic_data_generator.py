"""
Synthetic hourly gas-station sales generator (Spain).

Context
-------
Generates realistic, messy hourly time series for Spanish gas-station retail,
intended for training and testing forecasting models (XGBoost, Prophet, LSTM)
and skforecast forecasters. The output deliberately mimics real-world dynamics:
bimodal weekday commute peaks vs a single wide weekend peak, Spanish yearly
seasonality (Operacion Salida in July/August, Semana Santa, Christmas), hourly
weather (a diurnal temperature cycle and hourly rain events), fuel prices that
move as a bounded random walk with a Monday effect, autoregressive demand
momentum, and occasional anomalies (supply shocks, price-drop queues, extreme
weather).

What it generates
-----------------
A multi-station panel (mix of highway/urban stations, some open 24/7 and some
that close overnight). Each station-hour row carries multi-product targets
(diesel and gasoline liters, store and car-wash revenue) plus exogenous
features (prices, temperature, precipitation, holiday/weekend/open flags).

Usage
-----
    python dev/synthetic_data_generator.py                                  # defaults + report
    python dev/synthetic_data_generator.py --start 2023-01-01 --end 2024-12-31 --stations 3
    python dev/synthetic_data_generator.py --stations 4 --output wide --csv out.csv
    python dev/synthetic_data_generator.py --selftest                       # run internal checks

    from dev.synthetic_data_generator import generate_gas_station_panel
    df = generate_gas_station_panel("2023-01-01", "2024-12-31", stations=3, seed=42)

Environment note: run inside the project conda env (e.g. `skforecast_24_py13`).
The `holidays` library is used when available; otherwise a self-contained
computation of Spanish national holidays (fixed dates plus Easter-derived
Semana Santa days) is used instead.
"""

from __future__ import annotations
import argparse
from datetime import date, timedelta
import numpy as np
import pandas as pd


# ============================================================================ #
#                                  Holidays                                    #
# ============================================================================ #

def _easter(year: int) -> date:
    """
    Compute the Gregorian Easter Sunday date using the Anonymous Gregorian
    algorithm (computus).

    Parameters
    ----------
    year : int
        Gregorian year.

    Returns
    -------
    easter_sunday : datetime.date
        Date of Easter Sunday for the given year.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _compute_spanish_holidays_fallback(years: list[int]) -> set[date]:
    """
    Self-contained set of Spanish national holidays, used when the `holidays`
    library is not installed.

    Parameters
    ----------
    years : list
        Years to generate holidays for.

    Returns
    -------
    holidays_set : set
        Set of `datetime.date` objects that are national holidays in Spain.

    Notes
    -----
    Includes the fixed nationwide holidays plus the two movable Easter-derived
    days observed in most of Spain (Jueves Santo and Viernes Santo). Regional
    (autonomous community) holidays are not included.
    """
    holidays_set: set[date] = set()
    for year in years:
        holidays_set.update(
            {
                date(year, 1, 1),    # Ano Nuevo
                date(year, 1, 6),    # Epifania (Reyes)
                date(year, 5, 1),    # Dia del Trabajador
                date(year, 8, 15),   # Asuncion
                date(year, 10, 12),  # Fiesta Nacional
                date(year, 11, 1),   # Todos los Santos
                date(year, 12, 6),   # Dia de la Constitucion
                date(year, 12, 8),   # Inmaculada Concepcion
                date(year, 12, 25),  # Navidad
            }
        )
        easter = _easter(year)
        holidays_set.add(easter - timedelta(days=3))  # Jueves Santo
        holidays_set.add(easter - timedelta(days=2))  # Viernes Santo
    return holidays_set


def _get_spanish_holidays(years: list[int], prov: str | None = None) -> set[date]:
    """
    Return the set of Spanish national holiday dates, using the `holidays`
    library when importable and falling back to a self-contained computation
    otherwise.

    Parameters
    ----------
    years : list
        Years to generate holidays for.
    prov : str, default None
        Subdivision (autonomous community) code passed to the `holidays`
        library, e.g. `'MD'`. Ignored by the fallback implementation.

    Returns
    -------
    holidays_set : set
        Set of `datetime.date` objects that are holidays in Spain.
    """
    try:
        import holidays as holidays_lib

        es = holidays_lib.country_holidays("ES", years=years, subdiv=prov)
        return set(es.keys())
    except Exception:
        return _compute_spanish_holidays_fallback(years)


# ============================================================================ #
#                             Exogenous variables                              #
# ============================================================================ #

def _hourly_temperature(
    index: pd.DatetimeIndex,
    rng: np.random.Generator,
    mean_annual: float = 16.0,
    annual_amplitude: float = 11.0,
    diurnal_amplitude: float = 5.5,
) -> np.ndarray:
    """
    Generate an hourly temperature series (Celsius) with an annual cycle, a
    diurnal cycle (coldest before dawn, hottest mid-afternoon), and
    autocorrelated day-to-day noise.

    Parameters
    ----------
    index : pandas DatetimeIndex
        Hourly index for the series.
    rng : numpy.random.Generator
        Random number generator.
    mean_annual : float, default 16.0
        Mean annual temperature.
    annual_amplitude : float, default 11.0
        Amplitude of the annual (seasonal) cycle.
    diurnal_amplitude : float, default 5.5
        Amplitude of the within-day cycle.

    Returns
    -------
    temperature : numpy ndarray
        Hourly temperature in Celsius.
    """
    doy = index.dayofyear.to_numpy()
    hour = index.hour.to_numpy()

    # Annual cycle: peak around mid-July (day ~200), trough in January.
    annual = mean_annual + annual_amplitude * np.sin(
        2 * np.pi * (doy - 109) / 365.25
    )
    # Diurnal cycle: max at 15:00, min at 03:00.
    diurnal = diurnal_amplitude * np.cos(2 * np.pi * (hour - 15) / 24)

    # Autocorrelated daily anomaly (AR(1)) broadcast to the 24 hours of each day.
    n_days = (index.normalize().nunique())
    daily_anom = np.empty(n_days)
    daily_anom[0] = rng.normal(0, 2.5)
    for t in range(1, n_days):
        daily_anom[t] = 0.75 * daily_anom[t - 1] + rng.normal(0, 1.8)
    day_id = (index.normalize().view("int64"))
    _, inverse = np.unique(day_id, return_inverse=True)
    anomaly = daily_anom[inverse]

    hourly_noise = rng.normal(0, 0.6, size=len(index))
    return np.round(annual + diurnal + anomaly + hourly_noise, 1)


def _hourly_precipitation(
    index: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate hourly precipitation (mm) as a wet-spell process: seasonally
    varying probability of rain onset combined with persistence, so rain
    arrives in contiguous spells rather than isolated hours.

    Parameters
    ----------
    index : pandas DatetimeIndex
        Hourly index for the series.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    precipitation : numpy ndarray
        Hourly precipitation in mm (0.0 when dry).
    """
    doy = index.dayofyear.to_numpy()
    # Seasonal onset factor: wetter in spring/autumn, drier in summer.
    season = 0.6 + 0.4 * np.cos(2 * np.pi * (doy - 40) / 365.25)
    season = np.clip(season, 0.2, 1.0)

    n = len(index)
    p_start_base = 0.02   # dry -> wet hourly probability (modulated by season)
    p_stop = 0.30         # wet -> dry hourly probability (drives spell length)

    precip = np.zeros(n)
    raining = False
    u = rng.random(n)
    intensity = rng.exponential(scale=1.4, size=n)
    for t in range(n):
        if raining:
            precip[t] = np.round(0.2 + intensity[t], 1)
            if u[t] < p_stop:
                raining = False
        else:
            if u[t] < p_start_base * season[t] * 24:
                raining = True
                precip[t] = np.round(0.2 + intensity[t], 1)
    return precip


def _daily_fuel_prices(
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
    base_price_diesel: float,
    base_price_gasoline: float,
) -> pd.DataFrame:
    """
    Generate correlated daily diesel and gasoline prices as a bounded,
    mean-reverting random walk with a slight Monday drop.

    Parameters
    ----------
    dates : pandas DatetimeIndex
        Unique, sorted daily dates.
    rng : numpy.random.Generator
        Random number generator.
    base_price_diesel : float
        Long-run mean diesel price per liter.
    base_price_gasoline : float
        Long-run mean gasoline price per liter.

    Returns
    -------
    prices : pandas DataFrame
        Indexed by `dates`, columns `['price_diesel', 'price_gasoline']`.

    Notes
    -----
    A shared latent AR(1) factor drives the common market movement; independent
    idiosyncratic AR(1) factors decorrelate the two fuels partially, so they
    move together without being identical. Mean reversion keeps prices bounded.
    """
    n = len(dates)
    rho = 0.97  # persistence (random-walk-like but mean reverting)

    common = np.empty(n)
    idio_d = np.empty(n)
    idio_g = np.empty(n)
    common[0] = idio_d[0] = idio_g[0] = 0.0
    for t in range(1, n):
        common[t] = rho * common[t - 1] + rng.normal(0, 0.012)
        idio_d[t] = rho * idio_d[t - 1] + rng.normal(0, 0.004)
        idio_g[t] = rho * idio_g[t - 1] + rng.normal(0, 0.004)

    # Slight Monday drop (Spanish "Monday effect").
    dow = dates.dayofweek.to_numpy()
    monday_effect = np.where(dow == 0, -0.008, 0.0)

    price_diesel = base_price_diesel * np.exp(common + idio_d + monday_effect)
    price_gasoline = base_price_gasoline * np.exp(common + idio_g + monday_effect)

    return pd.DataFrame(
        {
            "price_diesel": np.round(price_diesel, 3),
            "price_gasoline": np.round(price_gasoline, 3),
        },
        index=dates,
    )


# ============================================================================ #
#                          Demand shape and momentum                           #
# ============================================================================ #

# Base intraday profiles (relative demand by hour, index 0..23).
_DIESEL_WEEKDAY = np.array(
    [0.1, 0.1, 0.1, 0.2, 0.5, 1.2, 2.5, 2.2, 1.8, 1.5, 1.4, 1.3,
     1.3, 1.3, 1.0, 0.8, 0.8, 0.9, 1.2, 1.0, 0.6, 0.3, 0.2, 0.1]
)
_DIESEL_WEEKEND = np.array(
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0,
     1.0, 1.0, 0.8, 0.6, 0.5, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
)
_GAS_WEEKDAY = np.array(
    [0.1, 0.05, 0.05, 0.05, 0.1, 0.2, 0.8, 1.8, 1.5, 1.2, 1.0, 1.2,
     1.5, 1.5, 1.0, 0.8, 1.0, 1.5, 2.2, 2.0, 1.2, 0.7, 0.4, 0.2]
)
_GAS_WEEKEND = np.array(
    [0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.3, 0.5, 1.0, 1.5, 1.8, 2.2,
     2.5, 2.0, 1.5, 1.2, 1.2, 1.5, 1.8, 1.6, 1.2, 0.8, 0.5, 0.4]
)


def _ar_momentum(
    n: int,
    rng: np.random.Generator,
    phi: float = 0.6,
    sigma: float = 0.12,
) -> np.ndarray:
    """
    Generate a multiplicative demand-momentum factor from a latent AR(1)
    process, so that consecutive hours are correlated (busy hours cluster).

    Parameters
    ----------
    n : int
        Number of hours.
    rng : numpy.random.Generator
        Random number generator.
    phi : float, default 0.6
        AR(1) autoregressive coefficient (persistence of momentum).
    sigma : float, default 0.12
        Standard deviation of the innovation term.

    Returns
    -------
    factor : numpy ndarray
        Positive multiplicative factor with mean approximately 1.
    """
    latent = np.empty(n)
    latent[0] = rng.normal(0, sigma)
    innov = rng.normal(0, sigma, size=n)
    for t in range(1, n):
        latent[t] = phi * latent[t - 1] + innov[t]
    # exp() keeps it positive; subtract half-variance to keep mean near 1.
    stationary_var = sigma ** 2 / (1 - phi ** 2)
    return np.exp(latent - 0.5 * stationary_var)


def _seasonal_calendar_multiplier(
    index: pd.DatetimeIndex,
    holidays_set: set[date],
    is_highway: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build multiplicative demand modifiers capturing Spanish weekly and yearly
    seasonality for diesel and gasoline separately.

    Parameters
    ----------
    index : pandas DatetimeIndex
        Hourly index.
    holidays_set : set
        National holiday dates.
    is_highway : bool
        Whether the station sits on a highway (amplifies holiday travel,
        dampens local weekday effects) versus an urban station.

    Returns
    -------
    diesel_mult, gasoline_mult : tuple of numpy ndarray
        Per-hour multiplicative modifiers for each fuel.
    """
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    month = index.month.to_numpy()
    day = index.day.to_numpy()
    dates = index.normalize().date

    diesel = np.ones(len(index))
    gas = np.ones(len(index))

    salida_up = 1.6 if is_highway else 1.15
    salida_down = 0.75 if is_highway else 0.95

    # Friday afternoon outbound (Operacion Salida) and Sunday evening return.
    friday_out = (dow == 4) & (hour >= 15) & (hour <= 21)
    gas[friday_out] *= salida_up
    diesel[friday_out] *= salida_down
    sunday_return = (dow == 6) & (hour >= 17) & (hour <= 22)
    gas[sunday_return] *= salida_up

    # Sunday heavy-truck ban depresses diesel.
    diesel[dow == 6] *= 0.4

    # Mid-week lull (Tuesday/Wednesday).
    gas[np.isin(dow, [1, 2])] *= 0.9
    diesel[np.isin(dow, [1, 2])] *= 0.95

    # Summer migration (July and August).
    summer = np.isin(month, [7, 8])
    gas[summer] *= 1.4 if is_highway else 0.7
    diesel[summer] *= 1.1 if is_highway else 0.85

    # Semana Santa (week of Easter) and Christmas travel window.
    holiday_flag = np.array([d in holidays_set for d in dates])
    easter_days = {
        d for d in holidays_set
        if d.month in (3, 4) and (d.weekday() in (3, 4))  # Jueves/Viernes Santo
    }
    semana_santa = np.array(
        [any(abs((d - e).days) <= 3 for e in easter_days) for d in dates]
    ) if easter_days else np.zeros(len(index), dtype=bool)
    gas[semana_santa] *= 1.5 if is_highway else 0.85
    diesel[semana_santa] *= 1.15 if is_highway else 0.9

    christmas = ((month == 12) & (day >= 20)) | ((month == 1) & (day <= 6))
    gas[christmas] *= 1.35 if is_highway else 0.85

    # On holidays, weekday commuting collapses (behaves like leisure travel).
    gas[holiday_flag] *= 1.25 if is_highway else 0.8
    diesel[holiday_flag] *= 0.55

    return diesel, gas


# ============================================================================ #
#                                  Anomalies                                    #
# ============================================================================ #

def _inject_anomalies(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n_supply_shocks: int = 3,
    n_price_queues: int = 4,
    n_extreme_weather: int = 3,
) -> pd.DataFrame:
    """
    Inject occasional anomalies into an assembled station DataFrame: supply
    shocks (sales collapse), price-drop queues (demand spike with a price cut),
    and extreme-weather events (heavy rain and suppressed sales).

    Parameters
    ----------
    df : pandas DataFrame
        Station data with the target and exogenous columns already assembled.
    rng : numpy.random.Generator
        Random number generator.
    n_supply_shocks : int, default 3
        Number of supply-shock events over the whole horizon.
    n_price_queues : int, default 4
        Number of price-drop queue events.
    n_extreme_weather : int, default 3
        Number of extreme-weather events.

    Returns
    -------
    df : pandas DataFrame
        The same DataFrame with anomalies applied and `is_anomaly` flagged.
    """
    n = len(df)
    liters_cols = ["liters_diesel_sold", "liters_gasoline_sold"]
    revenue_cols = ["store_revenue_euros", "carwash_revenue_euros"]

    def _block(duration_lo: int, duration_hi: int) -> slice:
        start = int(rng.integers(0, max(1, n - duration_hi)))
        dur = int(rng.integers(duration_lo, duration_hi))
        return slice(start, start + dur)

    # Supply shocks: pumps run dry, sales collapse.
    for _ in range(n_supply_shocks):
        sl = _block(3, 12)
        df.iloc[sl, df.columns.get_indexer(liters_cols + revenue_cols)] = 0.0
        df.iloc[sl, df.columns.get_loc("is_anomaly")] = 1

    # Price-drop queues: a sharp cut triggers a rush.
    for _ in range(n_price_queues):
        sl = _block(2, 6)
        df.iloc[sl, df.columns.get_indexer(["price_diesel", "price_gasoline"])] *= 0.92
        df.iloc[sl, df.columns.get_indexer(liters_cols)] = np.round(
            df.iloc[sl][liters_cols].to_numpy() * rng.uniform(1.5, 2.2)
        )
        df.iloc[sl, df.columns.get_loc("is_anomaly")] = 1

    # Extreme weather: torrential rain, suppressed sales.
    for _ in range(n_extreme_weather):
        sl = _block(4, 10)
        df.iloc[sl, df.columns.get_loc("precipitation_mm")] += rng.uniform(20, 45)
        df.iloc[sl, df.columns.get_loc("is_raining")] = 1
        df.iloc[sl, df.columns.get_indexer(liters_cols)] = np.round(
            df.iloc[sl][liters_cols].to_numpy() * 0.5
        )
        df.iloc[sl, df.columns.get_loc("is_anomaly")] = 1

    return df


# ============================================================================ #
#                              Station generator                               #
# ============================================================================ #

def generate_station(
    start_date: str,
    end_date: str,
    config: dict,
    rng: np.random.Generator,
    holidays_set: set[date],
) -> pd.DataFrame:
    """
    Generate one gas station's hourly multi-product series.

    Parameters
    ----------
    start_date : str
        Inclusive start date (any pandas-parseable date string).
    end_date : str
        Inclusive end date.
    config : dict
        Station configuration with keys `station_id`, `is_highway` (bool),
        `closes_at_night` (bool), `base_diesel_per_hr`, `base_gasoline_per_hr`,
        `base_price_diesel`, `base_price_gasoline`.
    rng : numpy.random.Generator
        Random number generator (one independent stream per station).
    holidays_set : set
        National holiday dates covering the horizon.

    Returns
    -------
    df : pandas DataFrame
        Hourly data indexed by a frequency-aware DatetimeIndex.
    """
    index = pd.date_range(start=start_date, end=end_date, freq="h")
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    dates_norm = index.normalize()

    is_highway = bool(config["is_highway"])
    is_weekend = np.isin(dow, [5, 6]).astype(int)
    holiday_flag = np.array(
        [d in holidays_set for d in dates_norm.date]
    ).astype(int)

    # Opening hours: closing stations shut from 23:00 to 05:59.
    if config["closes_at_night"]:
        is_open = ((hour >= 6) & (hour <= 22)).astype(int)
    else:
        is_open = np.ones(len(index), dtype=int)

    # --- Weather (hourly) ---
    temperature = _hourly_temperature(index, rng)
    precipitation = _hourly_precipitation(index, rng)
    is_raining = (precipitation > 1.0).astype(int)

    # --- Prices (daily random walk broadcast to hours) ---
    unique_days = dates_norm.unique()
    daily_prices = _daily_fuel_prices(
        unique_days, rng, config["base_price_diesel"], config["base_price_gasoline"]
    )
    prices = daily_prices.reindex(dates_norm)
    price_diesel = prices["price_diesel"].to_numpy()
    price_gasoline = prices["price_gasoline"].to_numpy()

    # --- Intraday profiles ---
    diesel_profile = np.where(is_weekend == 0, _DIESEL_WEEKDAY[hour], _DIESEL_WEEKEND[hour])
    gas_profile = np.where(is_weekend == 0, _GAS_WEEKDAY[hour], _GAS_WEEKEND[hour])

    # --- Seasonal / calendar modifiers ---
    diesel_season, gas_season = _seasonal_calendar_multiplier(
        index, holidays_set, is_highway
    )

    # --- Weather impact on demand ---
    temp_boost = 1.0 + np.where(temperature > 28, (temperature - 28) * 0.004, 0.0)
    temp_boost += np.where(temperature < 8, (8 - temperature) * 0.004, 0.0) - 1.0
    temp_boost += 1.0
    # Rain: urban stations lose leisure trips; highway gains long-distance traffic.
    if is_highway:
        rain_mod = np.where(is_raining == 1, 1.08, 1.0)
    else:
        rain_mod = np.where(
            (is_raining == 1) & (is_weekend == 1), 0.85,
            np.where(is_raining == 1, 1.03, 1.0),
        )
    weather_mod = temp_boost * rain_mod

    # --- Price elasticity ---
    diesel_elasticity = (price_diesel / config["base_price_diesel"]) ** -1.2
    gas_elasticity = (price_gasoline / config["base_price_gasoline"]) ** -1.8

    # --- Demand momentum (AR(1)) ---
    momentum_d = _ar_momentum(len(index), rng)
    momentum_g = _ar_momentum(len(index), rng)

    # --- Expected demand and non-negative count draws ---
    expected_diesel = (
        config["base_diesel_per_hr"] * diesel_profile * diesel_season
        * diesel_elasticity * weather_mod * momentum_d
    )
    expected_gas = (
        config["base_gasoline_per_hr"] * gas_profile * gas_season
        * gas_elasticity * weather_mod * momentum_g
    )
    expected_diesel = np.clip(expected_diesel, 0.0, None) * is_open
    expected_gas = np.clip(expected_gas, 0.0, None) * is_open

    liters_diesel = rng.poisson(np.maximum(expected_diesel, 1e-6)).astype(float)
    liters_gasoline = rng.poisson(np.maximum(expected_gas, 1e-6)).astype(float)
    liters_diesel[is_open == 0] = 0.0
    liters_gasoline[is_open == 0] = 0.0

    # --- Non-fuel revenue (continuous euros, lognormal) ---
    total_liters = liters_diesel + liters_gasoline
    carwash_base = np.where(is_raining == 1, 0.0, liters_gasoline * 0.05)
    carwash = np.where(
        carwash_base > 0,
        np.round(carwash_base * rng.lognormal(0, 0.3, size=len(index)), 2),
        0.0,
    )
    night = np.isin(hour, [0, 1, 2, 3, 4, 5])
    store_base = total_liters * 0.10 * np.where(night, 0.2, 1.0)
    store_base *= np.where((is_raining == 1) | (temperature < 5), 0.7, 1.0)
    store = np.round(store_base * rng.lognormal(0, 0.25, size=len(index)), 2)
    carwash[is_open == 0] = 0.0
    store[is_open == 0] = 0.0

    df = pd.DataFrame(
        {
            "station_id": config["station_id"],
            "liters_diesel_sold": liters_diesel,
            "liters_gasoline_sold": liters_gasoline,
            "store_revenue_euros": store,
            "carwash_revenue_euros": carwash,
            "price_diesel": price_diesel,
            "price_gasoline": price_gasoline,
            "temperature_c": temperature,
            "precipitation_mm": precipitation,
            "is_raining": is_raining,
            "is_national_holiday": holiday_flag,
            "is_weekend": is_weekend,
            "is_open": is_open,
            "is_highway_station": int(is_highway),
            "is_anomaly": 0,
        },
        index=index,
    )
    df.index.name = "datetime"

    df = _inject_anomalies(df, rng)

    # Re-assert hard constraints after anomaly injection.
    target_cols = [
        "liters_diesel_sold", "liters_gasoline_sold",
        "store_revenue_euros", "carwash_revenue_euros",
    ]
    df[target_cols] = df[target_cols].clip(lower=0.0)
    df.loc[df["is_open"] == 0, target_cols] = 0.0
    return df


# ============================================================================ #
#                                Panel builder                                 #
# ============================================================================ #

def _default_station_configs(n_stations: int, rng: np.random.Generator) -> list[dict]:
    """
    Build a list of station configurations alternating highway/urban and
    24/7/closing, with jittered base demand and prices.

    Parameters
    ----------
    n_stations : int
        Number of stations to configure.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    configs : list of dict
        Station configuration dictionaries.
    """
    configs = []
    for i in range(n_stations):
        is_highway = i % 2 == 0
        configs.append(
            {
                "station_id": f"station_{i + 1:02d}",
                "is_highway": is_highway,
                "closes_at_night": (i % 3 == 2),  # roughly a third close overnight
                "base_diesel_per_hr": float(np.round(rng.uniform(450, 750), 1)),
                "base_gasoline_per_hr": float(np.round(rng.uniform(300, 550), 1)),
                "base_price_diesel": float(np.round(rng.uniform(1.55, 1.75), 3)),
                "base_price_gasoline": float(np.round(rng.uniform(1.45, 1.65), 3)),
            }
        )
    return configs


def generate_gas_station_panel(
    start_date: str,
    end_date: str,
    stations: int | list[dict] = 3,
    output: str = "long",
    prov: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a multi-station panel of synthetic hourly Spanish gas-station data.

    Parameters
    ----------
    start_date : str
        Inclusive start date.
    end_date : str
        Inclusive end date.
    stations : int or list of dict, default 3
        Number of stations to auto-configure, or an explicit list of station
        configuration dictionaries (see `generate_station`).
    output : str, default 'long'
        `'long'` returns one row per station-hour with a `station_id` column
        and a repeated DatetimeIndex. `'wide'` returns a single DatetimeIndex
        with columns suffixed by station id.
    prov : str, default None
        Subdivision code passed to the `holidays` library, when available.
    seed : int, default 42
        Master seed. Each station receives an independent child RNG stream.

    Returns
    -------
    panel : pandas DataFrame
        The assembled panel, indexed by a frequency-aware DatetimeIndex.

    Notes
    -----
    The `is_national_holiday` column is a 0/1 indicator compatible with
    `skforecast.preprocessing.calculate_distance_from_holiday`.
    """
    if output not in ("long", "wide"):
        raise ValueError("`output` must be 'long' or 'wide'.")

    seed_seq = np.random.SeedSequence(seed)
    config_rng = np.random.default_rng(seed_seq.spawn(1)[0])

    if isinstance(stations, int):
        configs = _default_station_configs(stations, config_rng)
    else:
        configs = stations

    years = list(range(pd.Timestamp(start_date).year, pd.Timestamp(end_date).year + 1))
    holidays_set = _get_spanish_holidays(years, prov=prov)

    child_seeds = seed_seq.spawn(len(configs))
    frames = []
    for cfg, child in zip(configs, child_seeds):
        station_rng = np.random.default_rng(child)
        frames.append(generate_station(start_date, end_date, cfg, station_rng, holidays_set))

    if output == "long":
        return pd.concat(frames, axis=0)

    wide = pd.concat(
        [f.drop(columns="station_id").add_suffix(f"_{f['station_id'].iloc[0]}") for f in frames],
        axis=1,
    )
    return wide


# ============================================================================ #
#                            Reporting and self-test                           #
# ============================================================================ #

def summarize(df: pd.DataFrame) -> None:
    """
    Print a human-readable summary report of a generated panel.

    Parameters
    ----------
    df : pandas DataFrame
        A long-format panel produced by `generate_gas_station_panel`.

    Returns
    -------
    None
    """
    print("=" * 70)
    print("SYNTHETIC GAS-STATION PANEL SUMMARY")
    print("=" * 70)
    print(f"Rows                : {len(df):,}")
    print(f"Date range          : {df.index.min()}  ->  {df.index.max()}")
    print(f"Index frequency     : {df.index.freq}")
    if "station_id" in df.columns:
        print(f"Stations            : {df['station_id'].nunique()} "
              f"({', '.join(map(str, df['station_id'].unique()))})")
    print(f"Holiday hours       : {int(df['is_national_holiday'].sum()):,}")
    print(f"Anomaly hours       : {int(df['is_anomaly'].sum()):,}")

    target_cols = [
        "liters_diesel_sold", "liters_gasoline_sold",
        "store_revenue_euros", "carwash_revenue_euros",
    ]
    print("-" * 70)
    print("Targets (min / mean / max):")
    for col in target_cols:
        print(f"  {col:<24}: {df[col].min():>8.1f} / "
              f"{df[col].mean():>10.2f} / {df[col].max():>10.1f}")

    closed = df["is_open"] == 0
    if closed.any():
        max_sale_when_closed = df.loc[closed, target_cols].to_numpy().max()
        print(f"Max sale when closed: {max_sale_when_closed:.3f} (should be 0)")

    # Lag-1 autocorrelation of total fuel sales (momentum check), first station.
    if "station_id" in df.columns:
        first = df[df["station_id"] == df["station_id"].iloc[0]]
    else:
        first = df
    total = (first["liters_diesel_sold"] + first["liters_gasoline_sold"])
    print(f"Lag-1 autocorr sales: {total.autocorr(lag=1):.3f} (>0 means momentum)")
    print("=" * 70)


def _run_selftest() -> None:
    """
    Run internal consistency checks and raise `AssertionError` on failure.

    Returns
    -------
    None
    """
    df = generate_gas_station_panel("2023-01-01", "2024-12-31", stations=3, seed=42)
    target_cols = [
        "liters_diesel_sold", "liters_gasoline_sold",
        "store_revenue_euros", "carwash_revenue_euros",
    ]

    assert (df[target_cols] >= 0).all().all(), "Targets must be non-negative."

    closed = df["is_open"] == 0
    assert closed.any(), "Expected at least one closed hour among stations."
    assert (df.loc[closed, target_cols].to_numpy() == 0).all(), \
        "Sales must be exactly 0 when the station is closed."

    for year in (2023, 2024):
        mask = df.index.year == year
        assert df.loc[mask, "is_national_holiday"].sum() > 0, \
            f"No holidays found for {year}."

    first = df[df["station_id"] == df["station_id"].iloc[0]]
    by_hour = first.groupby(first.index.hour)["temperature_c"].mean()
    assert by_hour.idxmin() in (2, 3, 4, 5), \
        f"Coldest hour {by_hour.idxmin()} not near dawn."
    assert by_hour.idxmax() in (14, 15, 16), \
        f"Hottest hour {by_hour.idxmax()} not mid-afternoon."

    total = first["liters_diesel_sold"] + first["liters_gasoline_sold"]
    assert total.autocorr(lag=1) > 0.1, "Sales momentum (autocorrelation) too weak."

    assert df["is_anomaly"].sum() > 0, "No anomalies were injected."
    # A long panel repeats timestamps across stations; check a single station.
    assert first.index.inferred_freq == "h", \
        "Per-station index must be regular hourly."

    print("All self-tests passed.")


def main() -> None:
    """
    Command-line entry point: generate a panel, print a report, optionally
    save to CSV, or run the self-test suite.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic hourly Spanish gas-station sales data."
    )
    parser.add_argument("--start", default="2023-01-01", help="Inclusive start date.")
    parser.add_argument("--end", default="2024-12-31", help="Inclusive end date.")
    parser.add_argument("--stations", type=int, default=3, help="Number of stations.")
    parser.add_argument("--output", choices=["long", "wide"], default="long",
                        help="Panel layout.")
    parser.add_argument("--prov", default=None,
                        help="Subdivision code for the holidays library (e.g. MD).")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed.")
    parser.add_argument("--csv", default=None, help="Optional path to save the panel.")
    parser.add_argument("--selftest", action="store_true",
                        help="Run internal consistency checks and exit.")
    args = parser.parse_args()

    if args.selftest:
        _run_selftest()
        return

    df = generate_gas_station_panel(
        start_date=args.start,
        end_date=args.end,
        stations=args.stations,
        output=args.output,
        prov=args.prov,
        seed=args.seed,
    )
    if args.output == "long":
        summarize(df)
    else:
        print(f"Wide panel: {df.shape[0]:,} rows x {df.shape[1]} columns.")

    if args.csv:
        df.to_csv(args.csv)
        print(f"Saved to {args.csv}")


if __name__ == "__main__":
    main()
