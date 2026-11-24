# Comprobaciones del paquete `holidays` y prototipo de las dos funciones internas
# del plan FE-06 (dev/calendar_holidays/PLAN.md). No toca skforecast.
#
# Uso: python holidays_facts.py
# Para probar otra version de `holidays` sin tocar el entorno:
#   pip install --no-deps --target <carpeta> "holidays==<version>"
#   PYTHONPATH=<carpeta> python holidays_facts.py

import copy
import datetime
import pickle
import subprocess
import sys
import time
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd

import holidays as holidays_pkg


# ----------------------------------------------------------------------------
# Prototipo (especificacion de las funciones internas de la seccion 4 del plan)
# ----------------------------------------------------------------------------
def resolve_holiday_dates(holidays, holidays_kwargs, years):
    """
    Fechas de festivo como array ordenado y sin duplicados de datetime64[D].

    `holidays` es un codigo de pais (str) o una coleccion de fechas.
    """
    if isinstance(holidays, str):
        calendar = holidays_pkg.country_holidays(
            holidays, years=years, **(holidays_kwargs or {})
        )
        dates = np.array(sorted(calendar.keys()), dtype="datetime64[D]")
    else:
        dates = pd.DatetimeIndex(pd.to_datetime(list(holidays)))
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        if dates.hasnans:
            raise ValueError("`holidays` contains missing dates.")
        dates = np.unique(dates.values.astype("datetime64[D]"))

    return dates


def is_holiday(index, holiday_dates):
    """
    1 si la fecha local de cada marca de tiempo es festivo, 0 si no.
    """
    local_index = index.tz_localize(None) if index.tz is not None else index
    days = local_index.values.astype("datetime64[D]")

    return np.isin(days, holiday_dates).astype(int)


def index_years(index):
    """Anos de las fechas locales del indice (no tiene por que estar ordenado)."""
    local_index = index.tz_localize(None) if index.tz is not None else index
    years = local_index.year
    return range(int(years.min()), int(years.max()) + 1)


def reference(index, calendar):
    """Consulta marca a marca con el objeto de `holidays` (lento, de referencia)."""
    return np.fromiter((ts in calendar for ts in index), dtype=int, count=len(index))


def section(title):
    print(f"\n== {title}")


print(
    f"Python {sys.version.split()[0]} | holidays {version('holidays')} | "
    f"pandas {pd.__version__} | numpy {np.__version__}"
)

# ----------------------------------------------------------------------------
section("1. Identidad con la consulta marca a marca")
cases = {
    "diario naive": pd.date_range("2019-01-01", "2024-12-31", freq="D"),
    "horario naive": pd.date_range("2019-01-01", "2024-12-31 23:00", freq="h"),
    "horario Europe/Madrid": pd.date_range(
        "2019-01-01", "2024-12-31 23:00", freq="h", tz="Europe/Madrid"
    ),
    "horario UTC convertido a Madrid": pd.date_range(
        "2019-01-01", "2024-12-31 23:00", freq="h", tz="UTC"
    ).tz_convert("Europe/Madrid"),
    "15 min Europe/Madrid": pd.date_range(
        "2023-01-01", "2024-12-31 23:45", freq="15min", tz="Europe/Madrid"
    ),
    "dias laborables": pd.bdate_range("2019-01-01", "2024-12-31"),
    "sin freq, desordenado no": pd.DatetimeIndex(
        ["2024-01-01", "2024-01-06", "2024-05-02", "2024-07-01"]
    ),
}
for name, index in cases.items():
    dates = resolve_holiday_dates("ES", {"subdiv": "MD"}, index_years(index))
    fast = is_holiday(index, dates)
    slow = reference(index, holidays_pkg.country_holidays("ES", subdiv="MD"))
    print(
        f"{name:34s} n={len(index):6d} unit={index.unit} festivos={fast.sum():5d} "
        f"identico={np.array_equal(fast, slow)}"
    )

# Con zona horaria, la fecha es la local: 2024-12-24 23:30 UTC ya es Navidad en Madrid
ts = pd.DatetimeIndex(["2024-12-24 23:30"], tz="UTC").tz_convert("Europe/Madrid")
dates = resolve_holiday_dates("ES", None, index_years(ts))
print("2024-12-24 23:30 UTC visto desde Madrid es festivo:", is_holiday(ts, dates)[0])

# Cambio de ano: 2024-12-31 23:30 UTC ya es Ano Nuevo de 2025 en Madrid, y el ano
# que hay que pedir al paquete es el de la fecha local
ts = pd.DatetimeIndex(["2024-12-31 23:30"], tz="UTC").tz_convert("Europe/Madrid")
dates = resolve_holiday_dates("ES", None, index_years(ts))
print(
    "2024-12-31 23:30 UTC visto desde Madrid: anos pedidos", list(index_years(ts)),
    "| es festivo:", is_holiday(ts, dates)[0],
)

# ----------------------------------------------------------------------------
section("2. Fechas propias: tipos de entrada")
own = {
    "lista de str": ["2024-01-01", "2024-12-25"],
    "lista de date": [datetime.date(2024, 1, 1), datetime.date(2024, 12, 25)],
    "lista de Timestamp con hora": [
        pd.Timestamp("2024-01-01 10:00"), pd.Timestamp("2024-12-25 18:00")
    ],
    "DatetimeIndex": pd.DatetimeIndex(["2024-01-01", "2024-12-25"]),
    "DatetimeIndex con tz": pd.DatetimeIndex(
        ["2024-01-01", "2024-12-25"], tz="Europe/Madrid"
    ),
    "numpy datetime64": np.array(["2024-01-01", "2024-12-25"], dtype="datetime64[D]"),
    "con duplicados": ["2024-01-01", "2024-01-01", "2024-12-25"],
    "claves de un objeto de holidays": holidays_pkg.country_holidays(
        "ES", years=2024
    ).keys(),
}
index = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h", tz="Europe/Madrid")
for name, values in own.items():
    dates = resolve_holiday_dates(values, None, None)
    print(f"{name:34s} fechas={len(dates):2d} horas festivas={is_holiday(index, dates).sum()}")
try:
    resolve_holiday_dates(["2024-01-01", None], None, None)
except ValueError as e:
    print("con None ->", type(e).__name__, e)

# ----------------------------------------------------------------------------
section("3. Errores del paquete")
tests = {
    "pais inexistente": lambda: holidays_pkg.country_holidays("XX"),
    "subdiv inexistente": lambda: holidays_pkg.country_holidays("ES", subdiv="ZZ"),
    "kwarg desconocido": lambda: holidays_pkg.country_holidays("ES", foo=1),
    "categoria no soportada": lambda: holidays_pkg.country_holidays(
        "ES", categories=("bank",)
    ),
    "codigo en minusculas 'es'": lambda: holidays_pkg.country_holidays("es"),
    "nombre 'Spain'": lambda: holidays_pkg.country_holidays("Spain"),
}
for name, func in tests.items():
    try:
        func()
        print(f"{name:26s} -> sin error")
    except Exception as e:
        print(f"{name:26s} -> {type(e).__name__}: {str(e)[:110]}")
print("alpha-3 'ESP':", len(holidays_pkg.country_holidays("ESP", years=2024)))
try:
    supported = holidays_pkg.list_supported_countries(include_aliases=False)
    print("paises (sin alias):", len(supported), "| 'ES' in supported:", "ES" in supported)
except Exception as e:
    print("list_supported_countries ->", type(e).__name__, e)

# ----------------------------------------------------------------------------
section("4. Anos cubiertos por pais")
for country in ["ES", "US", "GB", "DE", "IN", "CN", "SA", "IL", "JP", "BR", "MX"]:
    cal = holidays_pkg.country_holidays(country)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        counts = {
            y: len(holidays_pkg.country_holidays(country, years=y))
            for y in (1900, 1990, 2024, 2040, 2080, 2150)
        }
    print(
        f"{country}: start_year={getattr(cal, 'start_year', 'n/d')} "
        f"end_year={getattr(cal, 'end_year', 'n/d')} festivos por ano={counts}"
    )
    for message in sorted({str(w.message) for w in caught}):
        print(f"    aviso del paquete: {message}")

# ----------------------------------------------------------------------------
section("5. Coste de resolver las fechas (min de 5)")


def best(func, n=5):
    times = []
    for _ in range(n):
        t = time.perf_counter()
        func()
        times.append(time.perf_counter() - t)
    return min(times) * 1e3


for country, kwargs in [
    ("ES", {"subdiv": "MD"}), ("US", {}), ("DE", {"subdiv": "BY"}), ("IN", {}),
    ("CN", {}), ("SA", {}),
]:
    t1 = best(lambda: resolve_holiday_dates(country, kwargs, [2025]))
    t10 = best(lambda: resolve_holiday_dates(country, kwargs, range(2015, 2025)))
    print(f"{country}: 1 ano {t1:6.2f} ms | 10 anos {t10:6.2f} ms")

index_h = pd.date_range("2015-01-01", "2024-12-31 23:00", freq="h")
dates = resolve_holiday_dates("ES", {"subdiv": "MD"}, index_years(index_h))
print(f"is_holiday, {len(index_h)} horas: {best(lambda: is_holiday(index_h, dates)):.2f} ms")
index_p = pd.date_range("2025-01-01", periods=24, freq="h")
print(
    "transform de 24 pasos (resolver + marcar): "
    f"{best(lambda: is_holiday(index_p, resolve_holiday_dates('ES', {'subdiv': 'MD'}, index_years(index_p)))):.2f} ms"
)

# ----------------------------------------------------------------------------
section("6. Opciones que cambian el resultado")
print(
    "US 2021-07-05 (4 de julio en domingo): observed=True ->",
    "2021-07-05" in holidays_pkg.country_holidays("US"),
    "| observed=False ->",
    "2021-07-05" in holidays_pkg.country_holidays("US", observed=False),
)
print(
    "ES 2024: nacional", len(holidays_pkg.country_holidays("ES", years=2024)),
    "| MD", len(holidays_pkg.country_holidays("ES", subdiv="MD", years=2024)),
    "| CT", len(holidays_pkg.country_holidays("ES", subdiv="CT", years=2024)),
)
for country in ["ES", "US", "DE", "GB", "JP"]:
    cal = holidays_pkg.country_holidays(country)
    print(
        f"{country}: categorias={getattr(cal, 'supported_categories', 'n/d')} "
        f"por defecto={getattr(cal, 'default_categories', getattr(cal, 'categories', 'n/d'))}"
    )
union = holidays_pkg.country_holidays("ES", subdiv="MD", years=2024) + (
    holidays_pkg.country_holidays("ES", subdiv="CT", years=2024)
)
print("union MD + CT 2024:", len(union), "festivos")

# ----------------------------------------------------------------------------
section("7. Persistencia de los parametros con skops")
try:
    import skops.io as sio
    from sklearn.preprocessing import FunctionTransformer

    params = {
        "str": "ES",
        "dict": {"subdiv": "MD", "observed": False},
        "lista de str": ["2024-01-01", "2024-12-25"],
        "lista de date": [datetime.date(2024, 1, 1)],
        "lista de Timestamp": [pd.Timestamp("2024-01-01")],
        "DatetimeIndex": pd.DatetimeIndex(["2024-01-01"]),
        "numpy datetime64": np.array(["2024-01-01"], dtype="datetime64[D]"),
    }
    for name, value in params.items():
        carrier = FunctionTransformer(kw_args={"holidays": value})
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dumped = sio.dumps(carrier)
                untrusted = sio.get_untrusted_types(data=dumped)
            print(f"{name:20s} -> ok, tipos no confiables: {untrusted}")
        except Exception as e:
            print(f"{name:20s} -> {type(e).__name__}: {str(e)[:100]}")
except ImportError:
    print("skops no instalado")

# ----------------------------------------------------------------------------
section("8. El objeto de `holidays` como parametro, y coste del import")
calendar = holidays_pkg.country_holidays("ES", subdiv="MD")
print("repr sin consultar:", repr(calendar), "| len:", len(calendar))
"2026-05-02" in calendar
print("repr tras una consulta: len(repr) =", len(repr(calendar)), "| len:", len(calendar))
print(
    "deepcopy igual:", copy.deepcopy(calendar) == calendar,
    "| pickle igual:", pickle.loads(pickle.dumps(calendar)) == calendar,
)
index_h = pd.date_range("2015-01-01", "2024-12-31 23:00", freq="h")
print(
    f"consulta marca a marca, {len(index_h)} horas: "
    f"{best(lambda: reference(index_h, holidays_pkg.country_holidays('ES', subdiv='MD')), n=3):.0f} ms"
)
code = (
    "import time, pandas; t = time.perf_counter(); import holidays; "
    "print(time.perf_counter() - t)"
)
times = [
    float(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout)
    for _ in range(5)
]
print(
    f"import holidays despues de pandas: min {min(times) * 1e3:.0f} ms, "
    f"max {max(times) * 1e3:.0f} ms (5 procesos nuevos)"
)
