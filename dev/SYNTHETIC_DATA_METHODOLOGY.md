# Synthetic Gas-Station Data: Methodology

This document describes the logic behind
[`skforecast/experimental/_synthetic_data.py`](../skforecast/experimental/_synthetic_data.py),
a generator that produces synthetic hourly gas-station panels for training and
testing forecasting models. It is meant as a reference for understanding *why*
the generated series look the way they do, not as API documentation (see
`docs/api/experimental.md` and the docstrings in the module itself for that).

## 1. Goal and design principles

The generator produces multi-product, multi-station hourly time series that
are realistic enough to exercise a forecaster's ability to handle:

- Multiple seasonalities (daily, weekly, yearly).
- Exogenous drivers (weather, price, calendar).
- Multi-series panels with heterogeneous station behavior.
- Messy real-world artifacts: closed hours, occasional anomalies, correlated
  noise.

Two design principles run through the whole module:

1. **Reproducibility.** Every random draw is derived from a single master
   `seed` via `numpy.random.SeedSequence`, so a given `(seed, parameters)`
   pair always yields byte-identical output.
2. **Composability of effects.** Demand is built as a product of independent
   multiplicative factors (intraday profile x seasonal/calendar x weather x
   price elasticity x momentum), each computed by its own function. This
   keeps every effect easy to reason about, test, and override in isolation.

## 2. Country-agnosticism and what is actually validated

`generate_gas_station_panel` accepts a `country` (ISO code, default `'ES'`)
and `subdiv` argument, and holidays are resolved via the `holidays` library
when it is installed (see Section 4). This makes the *calendar* mechanism
country-agnostic.

The *seasonal demand effects* (Section 5) are a separate concern, controlled
by `SeasonalEffectsConfig`. Its defaults reproduce the originally hand-tuned
Spanish patterns: Operacion Salida (the summer/holiday exodus), the Sunday
heavy-truck driving ban, and an Easter-linked ("Semana Santa") travel surge.
**Only these Spain-tuned defaults have been validated.** Equivalent effects
are not uniform across Europe (e.g. truck bans, school-holiday-driven travel
peaks, and the timing/scale of summer migration all vary by country), so
`country` alone does not make the generated demand patterns realistic for
another country. To model another country's demand seasonality, pass an
explicit `SeasonalEffectsConfig` override.

## 3. Random number generation

- A single `seed` (default 42) seeds a `numpy.random.SeedSequence`.
- One child stream is spawned for building auto-generated `StationConfig`
  jitter (`_default_station_configs`).
- One independent child stream is spawned per station, so stations don't
  share correlated noise and adding/removing a station does not perturb the
  random draws of the others.
- Within a station, the same `numpy.random.Generator` is threaded through
  temperature, precipitation, prices, momentum, and anomaly injection, in a
  fixed call order, so results are deterministic given the seed.

## 4. Calendar: holidays and Easter

- `_easter(year)` implements the Anonymous Gregorian algorithm (computus) to
  compute Easter Sunday for any Gregorian year. This is used both for the
  Easter-linked travel window (Section 5) and for the holiday fallbacks
  below, independent of any specific country's holiday table.
- `_get_holidays(country, years, subdiv)` resolves national holidays:
  - If the `holidays` library is importable, it delegates to
    `holidays.country_holidays(country, years=years, subdiv=subdiv)`.
  - If it is not importable and `country == 'ES'`, it falls back to
    `_compute_spanish_holidays_fallback`: a hardcoded table of Spain's fixed
    nationwide holidays plus the two Easter-derived days observed in most of
    Spain (Jueves Santo, Viernes Santo). This reproduces the generator's
    original (pre-country-parameterization) behavior exactly when `holidays`
    isn't installed.
  - If it is not importable and `country != 'ES'`, it falls back to
    `_compute_universal_holidays_fallback` (New Year's Day, Christmas, Easter
    Sunday, Good Friday only) and raises a `UserWarning`, since no
    country-specific holiday tables are hand-maintained in this module.
- The resulting `holidays_set` (a `set[date]`) is resolved **once** per call
  to `generate_gas_station_panel`, covering every calendar year touched by
  `[start_date, end_date]`, and passed down to every station.

## 5. Seasonal and calendar demand multipliers

`_seasonal_calendar_multiplier` builds two per-hour multiplicative arrays,
one for diesel and one for gasoline, starting from all-ones and applying each
effect below in sequence (effects compound multiplicatively, so a hour that
falls in more than one window, e.g. a Sunday within the Christmas window,
receives more than one adjustment). All magnitudes come from
`SeasonalEffectsConfig`, defaulting to the original Spain-tuned values.

| Effect | Trigger | What happens |
|---|---|---|
| Friday outbound travel | Friday 15:00-21:00 | Gasoline up (leisure travel begins), diesel down (commercial traffic drops) |
| Sunday return travel | Sunday 17:00-22:00 | Gasoline up (return leg of weekend travel) |
| Sunday heavy-truck ban | Every Sunday | Diesel multiplied by `sunday_truck_ban_factor` (default 0.4), if `sunday_truck_ban=True` |
| Midweek lull | Tuesday/Wednesday | Both fuels slightly down |
| Summer migration | Months in `summer_months` (default July/August) | Highway stations up (through-traffic), urban stations down (residents away) |
| Easter-linked travel window | `semana_santa_days_before` days before through `semana_santa_days_after` days after Easter Sunday (computed per calendar year via `_easter`) | Highway up, urban mixed, mirroring the summer pattern at smaller scale |
| Christmas travel | Dec 20 - Jan 6 | Gasoline: highway up, urban down |
| National holiday | Any date in `holidays_set` | Gasoline mixed by station type (leisure vs. commute), diesel down (commercial traffic drops) |

Two points on the mechanics:

- The Easter-linked window is computed **directly from `_easter(year)` for
  each year present in the index**, not from whether Holy Week days appear in
  `holidays_set`. This decouples the effect from a specific country's holiday
  labeling: the window fires even for countries whose holiday calendar
  doesn't mark Holy Week as a bank holiday.
- `is_highway` selects between a highway-tuned and an urban-tuned multiplier
  for every effect that differs by station type, reflecting that highway
  stations are dominated by through-traffic/leisure travel while urban
  stations are dominated by local commuting.

## 6. Intraday demand profile

Two fixed 24-hour relative-demand curves per fuel (`_DIESEL_WEEKDAY`,
`_DIESEL_WEEKEND`, `_GAS_WEEKDAY`, `_GAS_WEEKEND`) encode the base shape of a
day:

- **Diesel weekday**: a single sharp morning peak (06:00-09:00, commercial/
  commute traffic) tapering through the day.
- **Diesel weekend**: flatter, delayed, lower-amplitude (no commercial
  commute peak).
- **Gasoline weekday**: bimodal, a morning peak and a larger evening peak
  (18:00-19:00, commute + leisure).
- **Gasoline weekend**: a single broad late-morning/early-afternoon peak
  (11:00-13:00, leisure driving).

These are dimensionless relative weights, not literal traffic counts; the
`is_weekend` flag selects which curve applies to a given hour, and the curve
is then scaled by every other multiplier described in this document.

## 7. Weather

- **Temperature** (`_hourly_temperature`): additive combination of
  (a) an annual sine cycle peaking in mid-July, (b) a diurnal cosine cycle
  peaking at 15:00 and troughing at 03:00, (c) an AR(1) *daily* anomaly
  (broadcast across all 24 hours of a day, so consecutive days drift
  together rather than jumping), and (d) small i.i.d. hourly noise.
- **Precipitation** (`_hourly_precipitation`): a wet-spell process rather
  than i.i.d. rain draws. A seasonally varying onset probability (wetter in
  spring/autumn, drier in summer, via a cosine curve) governs the dry-to-wet
  transition; once raining, a fixed stop probability governs spell length.
  This produces rain in contiguous multi-hour blocks, matching real
  precipitation patterns, rather than isolated rainy hours.
- **Effect on demand**: `temp_boost` adds a small demand boost for hot
  (>28C, more travel/AC-driven store visits) or cold (<8C, more heating-fuel-
  adjacent behavior modeled generically) hours. Rain affects highway and
  urban stations oppositely: highway stations gain long-distance traffic
  (people avoid trains/driving conditions favor fuel stops), while urban
  stations lose weekend leisure trips but see a small weekday bump.

## 8. Fuel prices

`_daily_fuel_prices` generates one row per calendar day (broadcast to all 24
hours of that day):

- A shared latent AR(1) **common** factor drives market-wide price movement.
- Independent AR(1) **idiosyncratic** factors for diesel and gasoline
  partially decorrelate the two fuels, so they move together without being
  identical.
- All three factors use persistence `rho=0.97`, which keeps the process
  effectively a bounded random walk (mean-reverting but only very slowly) so
  prices don't diverge to unrealistic values over a long horizon.
- A small fixed drop on Mondays models the well-known "Monday effect" in
  Spanish fuel pricing.
- Prices are `base_price * exp(common + idiosyncratic + monday_effect)`, so
  the two fuels each have a distinct long-run mean set by `StationConfig`,
  fluctuating log-normally around it.

Demand responds to price via constant-elasticity terms:
`(price / base_price) ** -elasticity`, with diesel elasticity `-1.2` (less
price-sensitive, more commercial/inelastic demand) and gasoline elasticity
`-1.8` (more price-sensitive, discretionary driving).

## 9. Demand momentum

`_ar_momentum` generates a positive multiplicative factor from a latent AR(1)
process (`phi=0.6`), exponentiated and variance-corrected to have mean
approximately 1. This makes consecutive hours' demand correlated (a busy hour
tends to be followed by another busy hour) beyond what the deterministic
profile alone would produce, giving the target series realistic short-range
autocorrelation. Diesel and gasoline get independent momentum draws.

## 10. Assembling expected demand

For each fuel, expected hourly demand is the product of:

```
base_per_hr (StationConfig) x intraday_profile x seasonal/calendar_multiplier
x price_elasticity x weather_multiplier x momentum_factor
```

clipped at zero and multiplied by `is_open` (closing stations are open
06:00-22:00; 24/7 stations are always open). Actual liters sold are then
drawn as `Poisson(expected_demand)`, which keeps values non-negative,
integer-like, and realistically noisy relative to their mean (higher-volume
hours have proportionally less relative noise, matching count-like real-world
sales data).

## 11. Non-fuel revenue

- **Car wash revenue**: proportional to gasoline liters sold (a proxy for
  footfall), zeroed out during rain (nobody washes a car in the rain), with
  log-normal multiplicative noise.
- **Store revenue**: proportional to total liters sold, reduced at night
  (fewer impulse purchases) and during rain or cold (<5C) weather, with
  log-normal multiplicative noise.

Both are forced to zero when the station is closed.

## 12. Anomaly injection

`_inject_anomalies` runs after the main series is assembled and randomly
places three kinds of contiguous-block events (each `is_anomaly`-flagged):

1. **Supply shocks** (default 3 events, 3-12 hour blocks): all targets forced
   to zero, simulating pumps running dry.
2. **Price-drop queues** (default 4 events, 2-6 hour blocks): prices cut ~8%,
   liters sold multiplied 1.5x-2.2x, simulating a rush triggered by a
   promotional price cut.
3. **Extreme weather** (default 3 events, 4-10 hour blocks): precipitation
   boosted by 20-45mm, `is_raining` forced on, liters sold halved.

After injection, hard constraints are re-asserted (targets clipped at zero;
targets forced to zero during closed hours) so anomalies can never produce
physically inconsistent rows.

## 13. Panel assembly

`generate_gas_station_panel`:

1. Validates `output in {'long', 'wide'}`.
2. Builds station configurations: either `n` auto-generated stations
   (`_default_station_configs`, alternating highway/urban and 24/7/closing,
   with jittered base demand and prices) or an explicit list of
   `StationConfig` instances (plain dicts with the same fields are accepted
   for convenience and converted internally).
3. Resolves `holidays_set` once for every year spanned by the requested date
   range (Section 4).
4. Spawns one independent RNG child stream per station and calls
   `generate_station` for each.
5. Concatenates results: `'long'` stacks stations row-wise with a
   `station_id` column and a repeated `DatetimeIndex`; `'wide'` joins them
   column-wise with each station's columns suffixed by its `station_id`,
   sharing a single `DatetimeIndex`.

## 14. Output schema

Each row (long format) or each station's column group (wide format)
contains:

| Column | Meaning |
|---|---|
| `station_id` | Station identifier (long format only) |
| `liters_diesel_sold`, `liters_gasoline_sold` | Target: fuel volumes |
| `store_revenue_euros`, `carwash_revenue_euros` | Target: non-fuel revenue |
| `price_diesel`, `price_gasoline` | Exogenous: daily prices |
| `temperature_c`, `precipitation_mm`, `is_raining` | Exogenous: weather |
| `is_national_holiday`, `is_weekend` | Exogenous: calendar flags |
| `is_open` | Exogenous: 0/1 station-open flag |
| `is_highway_station` | Exogenous: static station-type flag |
| `is_anomaly` | Diagnostic: 1 during injected anomaly blocks |

`is_national_holiday` is a plain 0/1 indicator, compatible as-is with
`skforecast.preprocessing.calculate_distance_from_holiday`.

## 15. Known limitations / non-goals

- Regional (sub-national) holidays are not modeled in either fallback path;
  only the `holidays` library's own `subdiv` support provides those.
- Seasonal effect magnitudes are hand-tuned to feel realistic for Spain, not
  fit to real sales data; they should not be treated as calibrated
  estimates of true elasticities or seasonal amplitudes.
- The AR(1)/random-walk processes (temperature anomaly, prices, momentum)
  are simple linear-Gaussian approximations, not physical or econometric
  models; they are chosen for plausibility and speed, not fidelity.
