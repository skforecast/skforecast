################################################################################
#                          Synthetic Gas-Station Data                          #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################


from __future__ import annotations
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
import numpy as np
import pandas as pd


@dataclass
class StationConfig:
    """
    Configuration for a single synthetic gas station.

    Parameters
    ----------
    station_id : str
        Identifier used in the `station_id` column (and as a suffix in
        wide-format output).
    is_highway : bool, default False
        Whether the station sits on a highway (amplifies holiday/summer
        travel demand) versus an urban station (dominated by local commutes).
    closes_at_night : bool, default False
        If True, the station closes from 23:00 to 05:59. If False, it is
        open 24/7.
    base_diesel_per_hr : float, default 500.0
        Long-run average diesel demand per open hour, before seasonal,
        weather, price, and momentum modifiers.
    base_gasoline_per_hr : float, default 400.0
        Long-run average gasoline demand per open hour, before modifiers.
    base_price_diesel : float, default 1.65
        Long-run mean diesel price per liter.
    base_price_gasoline : float, default 1.55
        Long-run mean gasoline price per liter.
    """

    station_id: str
    is_highway: bool = False
    closes_at_night: bool = False
    base_diesel_per_hr: float = 500.0
    base_gasoline_per_hr: float = 400.0
    base_price_diesel: float = 1.65
    base_price_gasoline: float = 1.55


@dataclass
class SeasonalEffectsConfig:
    """
    Tunable multiplicative modifiers for weekly and yearly demand seasonality.

    Defaults reproduce the originally hand-tuned Spanish demand patterns
    (Operacion Salida in July/August, Sunday heavy-truck ban, an
    Easter-linked travel window, Christmas travel). Only these Spain-tuned
    defaults have been validated; when modeling another country, pass an
    explicit override rather than assuming these values transfer.

    Parameters
    ----------
    summer_months : tuple of int, default (7, 8)
        Calendar months treated as the summer travel season.
    weekend_travel_gas_highway_mult : float, default 1.6
        Gasoline multiplier applied to highway stations during the
        Friday-afternoon outbound and Sunday-evening return travel windows.
    weekend_travel_gas_urban_mult : float, default 1.15
        Same as `weekend_travel_gas_highway_mult`, for urban stations.
    friday_outbound_diesel_highway_mult : float, default 0.75
        Diesel multiplier applied to highway stations on Friday afternoon
        (commercial diesel traffic dips as leisure travel takes over).
    friday_outbound_diesel_urban_mult : float, default 0.95
        Same as `friday_outbound_diesel_highway_mult`, for urban stations.
    sunday_truck_ban : bool, default True
        Whether to apply a Sunday heavy-truck driving ban (depresses diesel
        demand on Sundays).
    sunday_truck_ban_factor : float, default 0.4
        Diesel multiplier applied on Sundays when `sunday_truck_ban` is True.
    midweek_gas_mult : float, default 0.9
        Gasoline multiplier applied on Tuesday/Wednesday (midweek lull).
    midweek_diesel_mult : float, default 0.95
        Diesel multiplier applied on Tuesday/Wednesday (midweek lull).
    summer_gas_highway_mult : float, default 1.4
        Gasoline multiplier for highway stations during `summer_months`.
    summer_gas_urban_mult : float, default 0.7
        Gasoline multiplier for urban stations during `summer_months`
        (local commuting drops as residents travel away).
    summer_diesel_highway_mult : float, default 1.1
        Diesel multiplier for highway stations during `summer_months`.
    summer_diesel_urban_mult : float, default 0.85
        Diesel multiplier for urban stations during `summer_months`.
    semana_santa_days_before : int, default 6
        Number of days before Easter Sunday included in the Easter-linked
        travel window.
    semana_santa_days_after : int, default 1
        Number of days after Easter Sunday included in the Easter-linked
        travel window.
    semana_santa_gas_highway_mult : float, default 1.5
        Gasoline multiplier for highway stations during the Easter-linked
        travel window.
    semana_santa_gas_urban_mult : float, default 0.85
        Gasoline multiplier for urban stations during the Easter-linked
        travel window.
    semana_santa_diesel_highway_mult : float, default 1.15
        Diesel multiplier for highway stations during the Easter-linked
        travel window.
    semana_santa_diesel_urban_mult : float, default 0.9
        Diesel multiplier for urban stations during the Easter-linked
        travel window.
    christmas_gas_highway_mult : float, default 1.35
        Gasoline multiplier for highway stations during the Christmas travel
        window (Dec 20 through Jan 6).
    christmas_gas_urban_mult : float, default 0.85
        Gasoline multiplier for urban stations during the Christmas travel
        window.
    holiday_gas_highway_mult : float, default 1.25
        Gasoline multiplier for highway stations on any national holiday.
    holiday_gas_urban_mult : float, default 0.8
        Gasoline multiplier for urban stations on any national holiday.
    holiday_diesel_mult : float, default 0.55
        Diesel multiplier (both station types) on any national holiday
        (commercial traffic drops).
    """

    summer_months: tuple[int, ...] = (7, 8)
    weekend_travel_gas_highway_mult: float = 1.6
    weekend_travel_gas_urban_mult: float = 1.15
    friday_outbound_diesel_highway_mult: float = 0.75
    friday_outbound_diesel_urban_mult: float = 0.95
    sunday_truck_ban: bool = True
    sunday_truck_ban_factor: float = 0.4
    midweek_gas_mult: float = 0.9
    midweek_diesel_mult: float = 0.95
    summer_gas_highway_mult: float = 1.4
    summer_gas_urban_mult: float = 0.7
    summer_diesel_highway_mult: float = 1.1
    summer_diesel_urban_mult: float = 0.85
    semana_santa_days_before: int = 6
    semana_santa_days_after: int = 1
    semana_santa_gas_highway_mult: float = 1.5
    semana_santa_gas_urban_mult: float = 0.85
    semana_santa_diesel_highway_mult: float = 1.15
    semana_santa_diesel_urban_mult: float = 0.9
    christmas_gas_highway_mult: float = 1.35
    christmas_gas_urban_mult: float = 0.85
    holiday_gas_highway_mult: float = 1.25
    holiday_gas_urban_mult: float = 0.8
    holiday_diesel_mult: float = 0.55


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
    library is not installed and `country='ES'`.

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


def _compute_universal_holidays_fallback(years: list[int]) -> set[date]:
    """
    Minimal, country-agnostic set of holidays used as a last-resort fallback
    when the `holidays` library is not installed and `country != 'ES'`.

    Parameters
    ----------
    years : list
        Years to generate holidays for.

    Returns
    -------
    holidays_set : set
        Set of `datetime.date` objects: New Year's Day, Christmas, Easter
        Sunday, and Good Friday for each year.
    """
    holidays_set: set[date] = set()
    for year in years:
        holidays_set.update({date(year, 1, 1), date(year, 12, 25)})
        easter = _easter(year)
        holidays_set.add(easter)
        holidays_set.add(easter - timedelta(days=2))  # Good Friday
    return holidays_set


def _get_holidays(
    country: str, years: list[int], subdiv: str | None = None
) -> set[date]:
    """
    Return the set of national holiday dates for `country`, using the
    `holidays` library when importable and falling back to a self-contained
    computation otherwise.

    Parameters
    ----------
    country : str
        ISO country code passed to the `holidays` library, e.g. `'ES'`,
        `'FR'`, `'DE'`.
    years : list
        Years to generate holidays for.
    subdiv : str, default None
        Subdivision code passed to the `holidays` library, e.g. `'MD'`.
        Ignored by the fallback implementations.

    Returns
    -------
    holidays_set : set
        Set of `datetime.date` objects that are holidays in `country`.

    Notes
    -----
    If the `holidays` library is not installed and `country != 'ES'`, this
    emits a `UserWarning` and falls back to a minimal universal holiday set,
    not country-specific holidays.
    """
    try:
        import holidays as holidays_lib

        country_cal = holidays_lib.country_holidays(
            country, years=years, subdiv=subdiv
        )
        return set(country_cal.keys())
    except Exception:
        if country == "ES":
            return _compute_spanish_holidays_fallback(years)
        warnings.warn(
            "The `holidays` library is not available: falling back to a "
            "minimal universal holiday set (New Year's Day, Christmas, "
            f"Easter Sunday, Good Friday) for country '{country}'. Install "
            "`holidays` for accurate country-specific holidays.",
            UserWarning,
        )
        return _compute_universal_holidays_fallback(years)


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
    day_id = (index.normalize().astype("int64"))
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
        AR(1) autoregressive coefficient (persistence of momentum). Must
        satisfy `abs(phi) < 1` for a stationary process.
    sigma : float, default 0.12
        Standard deviation of the innovation term.

    Returns
    -------
    factor : numpy ndarray
        Positive multiplicative factor with mean approximately 1.
    """
    if not abs(phi) < 1.0:
        raise ValueError(
            f"`phi` must satisfy abs(phi) < 1.0 for a stationary AR(1) "
            f"process, got {phi}."
        )
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
    seasonal_effects: SeasonalEffectsConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build multiplicative demand modifiers capturing weekly and yearly
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
    seasonal_effects : SeasonalEffectsConfig, default None
        Multiplicative modifiers to use. If None, `SeasonalEffectsConfig()`
        defaults (Spain-tuned) are used.

    Returns
    -------
    diesel_mult, gasoline_mult : tuple of numpy ndarray
        Per-hour multiplicative modifiers for each fuel.
    """
    if seasonal_effects is None:
        seasonal_effects = SeasonalEffectsConfig()
    se = seasonal_effects

    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    month = index.month.to_numpy()
    day = index.day.to_numpy()
    year = index.year.to_numpy()
    dates = index.normalize().date

    diesel = np.ones(len(index))
    gas = np.ones(len(index))

    salida_up = se.weekend_travel_gas_highway_mult if is_highway \
        else se.weekend_travel_gas_urban_mult
    salida_down = se.friday_outbound_diesel_highway_mult if is_highway \
        else se.friday_outbound_diesel_urban_mult

    # Friday afternoon outbound and Sunday evening return travel.
    friday_out = (dow == 4) & (hour >= 15) & (hour <= 21)
    gas[friday_out] *= salida_up
    diesel[friday_out] *= salida_down
    sunday_return = (dow == 6) & (hour >= 17) & (hour <= 22)
    gas[sunday_return] *= salida_up

    # Sunday heavy-truck ban depresses diesel.
    if se.sunday_truck_ban:
        diesel[dow == 6] *= se.sunday_truck_ban_factor

    # Mid-week lull (Tuesday/Wednesday).
    gas[np.isin(dow, [1, 2])] *= se.midweek_gas_mult
    diesel[np.isin(dow, [1, 2])] *= se.midweek_diesel_mult

    # Summer migration.
    summer = np.isin(month, se.summer_months)
    gas[summer] *= se.summer_gas_highway_mult if is_highway else se.summer_gas_urban_mult
    diesel[summer] *= (
        se.summer_diesel_highway_mult if is_highway else se.summer_diesel_urban_mult
    )

    # Easter-linked travel window, computed directly from Easter Sunday so it
    # does not depend on the target country's holiday calendar labeling Holy
    # Week days as national holidays.
    unique_years = np.unique(year)
    easter_by_year = {int(y): _easter(int(y)) for y in unique_years}
    easter_dates = np.array([easter_by_year[y] for y in year])
    days_from_easter = np.array(
        [(d - e).days for d, e in zip(dates, easter_dates)]
    )
    semana_santa = (
        (days_from_easter >= -se.semana_santa_days_before)
        & (days_from_easter <= se.semana_santa_days_after)
    )
    gas[semana_santa] *= (
        se.semana_santa_gas_highway_mult if is_highway else se.semana_santa_gas_urban_mult
    )
    diesel[semana_santa] *= (
        se.semana_santa_diesel_highway_mult if is_highway
        else se.semana_santa_diesel_urban_mult
    )

    christmas = ((month == 12) & (day >= 20)) | ((month == 1) & (day <= 6))
    gas[christmas] *= (
        se.christmas_gas_highway_mult if is_highway else se.christmas_gas_urban_mult
    )

    # On holidays, weekday commuting collapses (behaves like leisure travel).
    holiday_flag = np.array([d in holidays_set for d in dates])
    gas[holiday_flag] *= (
        se.holiday_gas_highway_mult if is_highway else se.holiday_gas_urban_mult
    )
    diesel[holiday_flag] *= se.holiday_diesel_mult

    return diesel, gas


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


def generate_station(
    start_date: str,
    end_date: str,
    config: StationConfig,
    rng: np.random.Generator,
    holidays_set: set[date],
    seasonal_effects: SeasonalEffectsConfig | None = None,
) -> pd.DataFrame:
    """
    Generate one gas station's hourly multi-product series.

    Parameters
    ----------
    start_date : str
        Inclusive start date (any pandas-parseable date string).
    end_date : str
        Inclusive end date.
    config : StationConfig
        Station configuration (see `StationConfig`).
    rng : numpy.random.Generator
        Random number generator (one independent stream per station).
    holidays_set : set
        National holiday dates covering the horizon.
    seasonal_effects : SeasonalEffectsConfig, default None
        Multiplicative seasonal demand modifiers. If None,
        `SeasonalEffectsConfig()` defaults (Spain-tuned) are used.

    Returns
    -------
    df : pandas DataFrame
        Hourly data indexed by a frequency-aware DatetimeIndex.
    """
    if seasonal_effects is None:
        seasonal_effects = SeasonalEffectsConfig()

    index = pd.date_range(start=start_date, end=end_date, freq="h")
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    dates_norm = index.normalize()

    is_highway = config.is_highway
    is_weekend = np.isin(dow, [5, 6]).astype(int)
    holiday_flag = np.array(
        [d in holidays_set for d in dates_norm.date]
    ).astype(int)

    # Opening hours: closing stations shut from 23:00 to 05:59.
    if config.closes_at_night:
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
        unique_days, rng, config.base_price_diesel, config.base_price_gasoline
    )
    prices = daily_prices.reindex(dates_norm)
    price_diesel = prices["price_diesel"].to_numpy()
    price_gasoline = prices["price_gasoline"].to_numpy()

    # --- Intraday profiles ---
    diesel_profile = np.where(is_weekend == 0, _DIESEL_WEEKDAY[hour], _DIESEL_WEEKEND[hour])
    gas_profile = np.where(is_weekend == 0, _GAS_WEEKDAY[hour], _GAS_WEEKEND[hour])

    # --- Seasonal / calendar modifiers ---
    diesel_season, gas_season = _seasonal_calendar_multiplier(
        index, holidays_set, is_highway, seasonal_effects
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
    diesel_elasticity = (price_diesel / config.base_price_diesel) ** -1.2
    gas_elasticity = (price_gasoline / config.base_price_gasoline) ** -1.8

    # --- Demand momentum (AR(1)) ---
    momentum_d = _ar_momentum(len(index), rng)
    momentum_g = _ar_momentum(len(index), rng)

    # --- Expected demand and non-negative count draws ---
    expected_diesel = (
        config.base_diesel_per_hr * diesel_profile * diesel_season
        * diesel_elasticity * weather_mod * momentum_d
    )
    expected_gas = (
        config.base_gasoline_per_hr * gas_profile * gas_season
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
            "station_id": config.station_id,
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


def _default_station_configs(
    n_stations: int, rng: np.random.Generator
) -> list[StationConfig]:
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
    configs : list of StationConfig
        Station configurations.
    """
    configs = []
    for i in range(n_stations):
        configs.append(
            StationConfig(
                station_id=f"station_{i + 1:02d}",
                is_highway=(i % 2 == 0),
                closes_at_night=(i % 3 == 2),  # roughly a third close overnight
                base_diesel_per_hr=float(np.round(rng.uniform(450, 750), 1)),
                base_gasoline_per_hr=float(np.round(rng.uniform(300, 550), 1)),
                base_price_diesel=float(np.round(rng.uniform(1.55, 1.75), 3)),
                base_price_gasoline=float(np.round(rng.uniform(1.45, 1.65), 3)),
            )
        )
    return configs


def generate_gas_station_panel(
    start_date: str,
    end_date: str,
    stations: int | list[StationConfig | dict] = 3,
    output: str = "long",
    country: str = "ES",
    subdiv: str | None = None,
    seasonal_effects: SeasonalEffectsConfig | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a multi-station panel of synthetic hourly gas-station data.

    Produces realistic, messy hourly time series intended for training and
    testing forecasting models and skforecast forecasters: bimodal weekday
    commute peaks vs a single wide weekend peak, yearly seasonality (summer
    travel, a movable Easter-linked travel window, Christmas), hourly weather
    (a diurnal temperature cycle and hourly rain events), fuel prices that
    move as a bounded random walk with a Monday effect, autoregressive demand
    momentum, and occasional anomalies (supply shocks, price-drop queues,
    extreme weather). The panel mixes highway/urban stations, some open 24/7
    and some that close overnight, each carrying multi-product targets
    (diesel and gasoline liters, store and car-wash revenue) plus exogenous
    features (prices, temperature, precipitation, holiday/weekend/open flags).

    Parameters
    ----------
    start_date : str
        Inclusive start date.
    end_date : str
        Inclusive end date.
    stations : int or list of StationConfig or dict, default 3
        Number of stations to auto-configure, or an explicit list of
        `StationConfig` instances (plain dicts with the same fields are also
        accepted for convenience).
    output : str, default 'long'
        `'long'` returns one row per station-hour with a `station_id` column
        and a repeated DatetimeIndex. `'wide'` returns a single DatetimeIndex
        with columns suffixed by station id.
    country : str, default 'ES'
        ISO country code used to resolve national holidays via the
        `holidays` library (when installed). See `_get_holidays`.
    subdiv : str, default None
        Subdivision code passed to the `holidays` library, when available.
    seasonal_effects : SeasonalEffectsConfig, default None
        Multiplicative seasonal demand modifiers (summer travel, weekend
        travel, Sunday truck ban, Easter-linked travel window, Christmas,
        generic holiday effect). If None, `SeasonalEffectsConfig()` defaults
        (Spain-tuned) are used. Pass an explicit instance when modeling a
        country other than Spain.
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

    The `holidays` library is used when available; otherwise a self-contained
    fallback is used instead. For `country='ES'` the fallback reproduces the
    Spanish national holidays (fixed dates plus Easter-derived Semana Santa
    days). For any other country, the fallback degrades to a minimal
    universal set (New Year's Day, Christmas, Easter Sunday, Good Friday) and
    emits a `UserWarning`, since no country-specific holiday tables are
    hand-maintained here.

    Only the default (Spain-tuned) `SeasonalEffectsConfig` values have been
    validated. When modeling another country, pass an explicit
    `SeasonalEffectsConfig` override rather than assuming the defaults
    transfer.
    """
    if output not in ("long", "wide"):
        raise ValueError("`output` must be 'long' or 'wide'.")

    seed_seq = np.random.SeedSequence(seed)
    config_rng = np.random.default_rng(seed_seq.spawn(1)[0])

    if isinstance(stations, int):
        configs = _default_station_configs(stations, config_rng)
    else:
        configs = [
            cfg if isinstance(cfg, StationConfig) else StationConfig(**cfg)
            for cfg in stations
        ]

    years = list(range(pd.Timestamp(start_date).year, pd.Timestamp(end_date).year + 1))
    holidays_set = _get_holidays(country, years, subdiv=subdiv)

    child_seeds = seed_seq.spawn(len(configs))
    frames = []
    for cfg, child in zip(configs, child_seeds):
        station_rng = np.random.default_rng(child)
        frames.append(
            generate_station(
                start_date, end_date, cfg, station_rng, holidays_set, seasonal_effects
            )
        )

    if output == "long":
        return pd.concat(frames, axis=0)

    wide = pd.concat(
        [f.drop(columns="station_id").add_suffix(f"_{f['station_id'].iloc[0]}") for f in frames],
        axis=1,
    )
    return wide


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
