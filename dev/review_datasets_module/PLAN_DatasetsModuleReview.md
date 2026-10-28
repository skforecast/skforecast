# Plan: bugs y mejoras de `skforecast.datasets` y del repositorio `skforecast-datasets`

**Versión del plan:** 2 (2026-10-27). Lo que cambió respecto a la versión 1 está en el apéndice C.
**Base comprobada:** `0.27.x` en `6aee84b` (0.27.0 en desarrollo) más la PR skforecast/skforecast#1402 (`fix/fetch-dataset-website-visits-m5`, head `051b1bd`), fusionada sin cambios el 2026-10-28 (`79efe5281`).
**Entornos usados:** versión 1: Linux, Python 3.12.3, pandas 2.3.3, pyarrow 26.0.0. Versión 2: Windows, `skforecast_27_py13` (Python 3.13, pandas 2.3.3, numpy 2.5.4).

> **Cómo usar este documento**
> 1. Vive en `0.27.x`, en `dev/review_datasets_module/`, para que lo vean todas las ramas y sesiones. Sus cambios se suben directamente a `0.27.x` (solo toca `dev/`: sin PR ni `verify`), nunca dentro de una PR del plan.
> 2. Para ejecutar una PR: «Lee `dev/review_datasets_module/PLAN_DatasetsModuleReview.md` y ejecuta la PR N». La columna «Estado» de la tabla 0.1 dice cuál toca; el flujo de trabajo está en la sección 2.1.
> 3. Todas las PRs parten de `origin/0.27.x`, que ya incluye la 1402.
> 4. Las decisiones están tomadas (sección 3) salvo DEC6 y DEC7, que solo bloquean las PRs 5c y 5d.
> 5. Antes de empezar una PR, repite el apéndice A sobre un clon reciente de `skforecast-datasets`: los ficheros pueden haber cambiado.

---

## 0. Resumen

### 0.1 PRs

| PR | Estado | Repo | Rama | Contenido | Bloqueada por | Nota de versión |
|---|---|---|---|---|---|---|
| **0** | fusionada (#1402, 2026-10-28) | skforecast | `fix/fetch-dataset-website-visits-m5` | PR #1402 (base de todo el plan) | nada | ya incluida |
| **1** | pendiente | skforecast | `fix/datasets-module-small-fixes` | Bugs pequeños, textos del registro, docstrings, test del registro sin red | nada | Fixed |
| **2** | pendiente | skforecast | `feature/datasets-version-deprecation` | Deprecar `version`; `load_demo_dataset` sobre `fetch_dataset` | PR 1 | Changed |
| **3** | pendiente | skforecast | `feature/fetch-dataset-gaps-and-long-format` | Aviso cuando `asfreq` añade o quita filas; orden estable del formato largo; documentación | PR 2 | Changed |
| **4** | pendiente | skforecast | `chore/datasets-regression-test` | Test de regresión contra los ficheros reales, opt-in | PR 3 | ninguna |
| **5a** | pendiente | skforecast-datasets | `main` (commits directos) | README generado y BOM | PR 1 | ninguna |
| **5b** | pendiente | skforecast-datasets / HF | (subida a HF) | Parquet de HF ordenados por fecha | token de HF | se menciona en la release en la que se suba |
| **5c** | pendiente | skforecast-datasets | `main` (commits directos) | Ficheros huérfanos | DEC6 | ninguna |
| **5d** | pendiente | skforecast-datasets | `main` (commits directos) | Licencias | DEC7 | ninguna |

Valores de «Estado»: `pendiente`, `en curso (#NNNN)`, `fusionada (#NNNN, AAAA-MM-DD)`. Se actualiza al abrir y al fusionar cada PR (sección 2.1).

Orden: las PRs 1 a 4 van en secuencia, cada una cuando la anterior esté fusionada, porque tocan `datasets.py` y `test_fetch_dataset.py` (PR 2 y PR 3 no dependen funcionalmente entre sí, pero en paralelo darían conflictos). La PR 4 va la última para fijar el comportamiento final. Las PRs 5 van por su lado: 5a cuando el registro esté estable tras la PR 1; 5b en cualquier momento.

### 0.2 Decisiones

Tomadas el 2026-10-27: DEC1 (deprecar `version`), DEC2 (orden estable y parquet de HF ordenados), DEC3 (avisar), DEC4 (generador en el repo de datos), DEC5 (documentar la zona horaria) y DEC8 (test opt-in). Pendientes: DEC6 (ficheros huérfanos) y DEC7 (licencias). Detalle en la sección 3.

---

## 1. Contexto y hechos verificados

### 1.1 El módulo

`skforecast/datasets/datasets.py` define el registro `datasets` (30 entradas con las claves `url`, `sep`, `index_col`, `date_format`, `freq`, `file_type`, `description`, `source`) y tres funciones públicas: `fetch_dataset`, `load_demo_dataset` y `show_datasets_info`. `skforecast/datasets/__init__.py` exporta las tres y el registro. `_print_dataset_info` es privada.

`fetch_dataset(name, version='latest', raw=False, kwargs_read=None, verbose=True)`:

1. Sustituye `{version}` en la URL por `main` si `version='latest'` (las URL de Hugging Face no llevan `{version}`).
2. Lee CSV (`pd.read_csv(url, sep=sep, **kwargs_read)`) o parquet (`pd.read_parquet(url, **kwargs_read)`); cualquier error se relanza como `ValueError` sin encadenar.
3. Con `raw=False`: pone `index_col` como índice, lo convierte con `pd.to_datetime(format=date_format)`, aplica `asfreq(freq)` **solo si el índice es único** y ordena por índice. Cualquier fallo emite `UserWarning: Could not preprocess dataset ...` y devuelve el DataFrame sin preprocesar.

Cambios de la PR 1402: `date_format` de `website_visits` pasa a `%d/%m/%y`; `index_col` de `m5` pasa a `datetime`; el índice se asigna con `df.index = df.pop(index_col)`; se libera el pool de pyarrow tras leer un parquet; `sort_index` solo se llama si el índice no es monótono creciente.

Consecuencia importante: en los datasets en formato largo (índice no único) `freq` **nunca se aplica**; solo sirve como metadato (README, `show_datasets_info`).

### 1.2 Tests actuales (tras 1402)

23 tests en `skforecast/datasets/tests/tests_datasets/`: `test_fetch_dataset.py` 14, `test_load_demo_dataset.py` 2, `test_print_dataset_info.py` 3, `test_show_datasets_info.py` 4. Nueve descargan ficheros reales de GitHub (`h2o`, `items_sales`, `website_visits` y un 404 deliberado) y ninguno está marcado. Usan comillas simples.

La CI (`unit-tests.yml`, matriz de 5 versiones de Python por 3 sistemas, en las PRs a `main`) ejecuta `pytest` **sin filtro `-m`**: los tests `@pytest.mark.slow` también corren allí.

### 1.3 Tabla de verdad de los 30 datasets

Obtenida cargando cada fichero real con el preprocesado de `0.27.x` + 1402. «Filas» es tras el preprocesado (`wikipedia_visits` gana 59 filas NaN por `asfreq`). «Cols» excluye la columna de índice. Los cuatro de Hugging Face (HF) no están en el repositorio de GitHub.

| name | fichero | index_col | date_format | freq | índice único | ordenado en el fichero | filas | cols |
|---|---|---|---|---|---|---|---|---|
| h2o | h2o.csv | fecha | %Y-%m-%d | MS | sí | sí | 204 | 1 |
| h2o_exog | h2o_exog.csv | fecha | %Y-%m-%d | MS | sí | sí | 195 | 3 |
| fuel_consumption | consumos-combustibles-mensual.csv | Fecha | %Y-%m-%d | MS | sí | sí | 644 | 5 |
| items_sales | simulated_items_sales.csv | date | %Y-%m-%d | D | sí | sí | 1097 | 3 |
| air_quality_valencia | air_quality_valencia.csv | datetime | %Y-%m-%d %H:%M:%S | h | sí | sí | 43824 | 10 |
| air_quality_valencia_no_missing | air_quality_valencia_no_missing.csv | datetime | %Y-%m-%d %H:%M:%S | h | sí | sí | 43824 | 10 |
| website_visits | visitas_por_dia_web_cienciadedatos.csv | date | %d/%m/%y (1402) | D | sí | sí | 421 | 1 |
| bike_sharing | bike_sharing_dataset_clean.csv | date_time | %Y-%m-%d %H:%M:%S | h | sí | sí | 17544 | 11 |
| bike_sharing_extended_features | bike_sharing_extended_features.csv | date_time | %Y-%m-%d %H:%M:%S | h | sí | sí | 17352 | 90 |
| australia_tourism | australia_tourism.csv | date_time | %Y-%m-%d | **Q (debe ser QS)** | no | no | 24320 | 4 |
| uk_daily_flights | uk_daily_flights.csv (con BOM) | Date | %d/%m/%Y | D | sí | sí | 1118 | 1 |
| wikipedia_visits | wikipedia_visits.csv | date | %Y-%m-%d | D | sí (59 días ausentes) | sí | 2964 | 1 |
| vic_electricity | vic_electricity.csv | Time | %Y-%m-%dT%H:%M:%SZ | 30min | sí | sí | 52608 | 4 |
| vic_electricity_classification | vic_electricity_classification.csv | Time | %Y-%m-%d %H:%M:%S | h | sí | sí | 8736 | 5 |
| store_sales | store_sales.csv | date | %Y-%m-%d | D | no | no | 913000 | 3 |
| bicimad | bicimad_users.csv | date | %Y-%m-%d | D | sí | sí | 3022 | 1 |
| m4_daily | m4_daily.parquet | timestamp | (ya datetime) | D | no | no | 10023836 | 3 |
| m4_hourly | m4_hourly.parquet | timestamp | (ya datetime) | h | no | no | 373372 | 3 |
| ashrae_daily | HF ashrae_daily.parquet | timestamp | (ya datetime) | D | no | sí | 444324 | 10 |
| bdg2_daily | HF bdg2_daily.parquet | timestamp | (ya datetime) | D | no | no | 1153518 | 17 |
| bdg2_daily_sample | bdg2_daily_sample.csv | timestamp | %Y-%m-%d | D | sí | sí | 274 | 2 |
| bdg2_hourly | HF bdg2_hourly.parquet | timestamp | (ya datetime) | h | no | no | 27684432 | 17 |
| bdg2_hourly_sample | bdg2_hourly_sample.csv | timestamp | %Y-%m-%d %H:%M:%S | h | sí | sí | 6553 | 2 |
| m5 | HF m5_daily.parquet | datetime (1402) | (ya datetime) | D | no | no | 59181090 | 15 |
| ett_m1 | ETTm1.csv | date | %Y-%m-%d %H:%M:%S | 15min | sí | sí | 69680 | 7 |
| ett_m2 | ETTm2.csv | date | %Y-%m-%d %H:%M:%S | 15min | sí | sí | 69680 | 7 |
| ett_m2_extended | ETTm2_extended.csv | date | %Y-%m-%d %H:%M:%S | 15min | sí | sí | 69680 | 16 |
| expenditures_australia | expenditures_australia.csv | date | %Y-%m-%d | MS | sí | sí | 505 | 1 |
| public_transport_madrid | public-transport-madrid.csv | date | %Y-%m-%d | D | sí | sí | 715 | 5 |
| turbine_emission | turbine_emission.csv | datetime | %Y-%m-%d %H:%M:%S | h | sí | sí | 36733 | 11 |

### 1.4 Otros hechos sobre los datos

- **`skforecast-datasets`**: sin tags y con dos ramas (`main` y `feature_public_transport_madrid`). `version` solo funciona con un nombre de rama; cualquier otro valor da `HTTP Error 404`. Sin CI; contiene `LICENSE`, `README.md` y `data/`. Los 26 ficheros que usa el registro suman unos 175 MB (`m4_daily.parquet` 66 MB, los tres `ETT*` 40 MB, `store_sales.csv` 17 MB, `bike_sharing_extended_features.csv` 16 MB).
- **Siete ficheros de `data/` no están en el registro ni en el README:** `bitcoin.csv`, `demo_multi_series.csv`, `demo_multi_series_exog.csv`, `energy_production_shutdown.csv`, `generacion_energia_US_source.csv`, `intermittent_demand.csv`, `visitas_por_dia_web_cienciadedatos_2020_2021.csv`.
- **README de datos desalineado con el registro:** 18 entradas difieren hoy (19 tras 1402, por `website_visits`) según el apéndice B: formatos de `uk_daily_flights`, `vic_electricity_classification`, `bdg2_daily_sample`; `index_col`, `freq` y `date_format` de `m4_*`; URL de Google Drive en `ashrae_daily`, `bdg2_daily`, `bdg2_hourly` y «(4 partitions)» en `m5`; `date_format: date_format:` en `air_quality_valencia*`; `H` mayúscula en nueve entradas; `sep` vacío en `bdg2_*_sample`; URL con `refs/heads/main` en seis entradas (equivalentes, pero distintas del registro).
- **`australia_tourism`**: 80 trimestres (1998-01-01 a 2017-10-01, inicios de trimestre) y 304 series. La `description` dice «from 1998 Q1 to 2016 Q4»: es **2017 Q4**. `to_offset('Q')` emite `FutureWarning` en pandas 2.3.3 (alias obsoleto, `QE`); `QS` no avisa.
- **`vic_electricity`**: `Time` es UTC (sufijo `Z`) y se parsea como naive; `Date` y `Holiday` son hora local de Melbourne. Primera fila: `2011-12-31T13:00:00Z` con `Date=2012-01-01`. Consecuencia: las variables de calendario derivadas del índice (hora, día de la semana) están desplazadas 10 u 11 horas respecto a la hora local, y `Holiday` no coincide con el día UTC del índice.
- **ETT (`ett_m1`, `ett_m2`, `ett_m2_extended`)**: la `description` dice «between July 2016 and July 2018 ... 70,080 data points»; el fichero tiene 69,680 filas, de 2016-07-01 00:00 a 2018-06-26 19:45.
- **`m4_daily`/`m4_hourly`**: marcas de tiempo a las 12:00 y años de 1959 a 2039 (artefacto del conversor de Monash). **`m5`**: los 30,490 ids terminan en `_evaluation`; el 20.8 % de `sell_price` es NaN. **`bdg2_*` y `ashrae_daily`**: paneles completos, sin duplicados (edificio, fecha).
- **Orden del formato largo:** `sort_index()` usa `quicksort`, que no es estable: dentro de cada fecha se pierde el orden de las series del fichero. Medido en un DataFrame sintético con la forma de `m5` (30,490 series por 1,941 días, una columna float32, Windows, una sola ejecución): `quicksort` 6.9 s sin conservar el orden; `kind='stable'` 5.6 s conservándolo.

### 1.5 Uso en docs, tools y contexto de IA

Lo que importa para cada PR (búsqueda de `fetch_dataset(`, `load_demo_dataset(` y `show_datasets_info(` en `docs/`, `tools/`, `skills/`, `README.md`):

- **`h2o`**: unos 13 notebooks, la mayoría con `raw=True, kwargs_read={"names": ["y", "datetime"], "header": 0}`, igual que `load_demo_dataset`. Todos dependen del orden de columnas `x,fecha` de `h2o.csv`: no reordenarlo nunca.
- **`wikipedia_visits`**: `docs/user_guides/migrate-from-prophet.ipynb` (interpola los 59 días ausentes). Mostrará el aviso de la PR 3.
- **`store_sales`** (formato largo): `docs/user_guides/comparing-forecasting-models.ipynb`, con `pivot_table(...).asfreq("D")`. No depende del orden de filas.
- **`australia_tourism`**: solo `tools/docs/home_page/generate_home_data.py` (pivota y aplica `asfreq("QS")`).
- **`vic_electricity`**: `explainability`, `foundation-forecasting-models`, `comparing-forecasting-models`, `faq/time-series-aggregation`, `faq/parameters-search-backtesting-vs-one-step-ahead`, más `tools/docs/figures/gif_shap_values.ipynb`, `generate_home_data.py` y `docs/javascripts/home.js`.
- **`website_visits`**: `metrics.ipynb` y `faq/parameters-search-backtesting-vs-one-step-ahead.ipynb`, ambos con `raw=True` y `%d/%m/%y`.
- **`docs/user_guides/datasets.ipynb`**: tiene guardada la salida de `show_datasets_info()` (los 30 paneles, con la URL partida por `textwrap`). Queda desactualizada con la PR 1.
- **Ningún** notebook usa `m4_*`, `m5`, `ashrae_daily`, `bdg2_daily` ni `bdg2_hourly`, ni pasa `version`.
- `tools/ai/llms-base.txt` (sección «Available Datasets») tiene ejemplos de `fetch_dataset` y `show_datasets_info()` sin `version`; `skills/` no menciona estas funciones.

---

## 2. Reglas de trabajo

### 2.1 Flujo de trabajo de cada PR

Las PRs se hacen en días distintos, en sesiones sin memoria de la anterior: el plan es la única fuente de estado.

1. **Arrancar:** comprobar en la tabla 0.1 que la PR anterior está fusionada y que no hay otra `en curso`. Actualizar la base: `git fetch origin` y `git switch 0.27.x && git pull --ff-only`.
2. **Crear la rama sin seguimiento**, para que una sincronización de VS Code no empuje a `0.27.x`: `git switch -c <rama> --no-track origin/0.27.x`. La primera subida, con `git push -u origin <rama>`.
3. **Implementar** la sección de la PR, con `verify` antes de cada commit de código, tests o docs. Abrir la PR contra `0.27.x` y pasar su fila de la tabla 0.1 a `en curso (#NNNN)`.
4. **Si la rama se queda atrás**, actualizarla con `git merge origin/0.27.x`, nunca con rebase (el force push está bloqueado).
5. **Si una sesión acaba con la PR a medias**, ejecutar `/handoff` (escribe un `dev/handoff_*.md`) y empezar la siguiente sesión leyendo ese fichero además del plan.
6. **Al fusionar:** pasar la fila a `fusionada (#NNNN, fecha)`, anotar en el apéndice C cualquier desviación del plan y subir el cambio del plan directamente a `0.27.x`. Borrar la rama local.
7. **Antes de la release 0.27.0:** `/release-note consolidate` (las PRs 1 a 3 añaden entradas parecidas) y el test opt-in de la PR 4 con `SKFORECAST_TEST_DATASETS=1`.
8. **Al terminar el plan** (PRs 1 a 5 cerradas, o DEC6/DEC7 descartadas): borrar `dev/review_datasets_module/` de `0.27.x`, o dejarlo como histórico.

### 2.2 Reglas

Las del repo están en `CLAUDE.md` y `tools/ai/ai_context_header.md` y se cargan solas. Lo específico de este plan:

- **Ramas y PRs:** `<type>/<topic>` con `type` en `feature`, `fix`, `docs`, `chore`, `perf`, desde `origin/0.27.x` y contra `0.27.x` (nunca `main`), una a la vez (2.1). Sin force push, sin trailers de IA.
- **Commits:** uno por problema y un último commit con la nota de versión (preferencia, no regla del repo).
- **Estilo:** `datasets.py` y sus tests usan comillas simples; mantenerlas. ruff solo comprueba `E` y `F` (ignora `E501`), pero el código nuevo respeta 88 columnas. Sin guiones largos en comentarios, docstrings ni docs.
- **Docstrings:** `.github/instructions/docstrings.instructions.md`: línea de tipo `name : type, default value` (con espacio antes de los dos puntos), tipos como `pandas DataFrame`, parámetros obsoletos con `**Deprecated**`.
- **Tests:** `.github/instructions/testing.instructions.md`: docstring multilínea en cada test, dos líneas en blanco entre tests, `re.escape` en los `match`, valores esperados escritos a mano. Nunca `-n`, nunca la suite completa. Un test nuevo que descarga datos va marcado (ver DEC8).
- **Hecho:** skill `verify`. Nota de versión: skill `release-note`, sección `## 0.27.0 <small>In development</small>` de `docs/releases/releases.md` (`[fetch_dataset]`, `[load_demo_dataset]` y `[show_datasets_info]` ya están definidos al final). Si cambia una firma pública: skill `ai-context-sync` y `python tools/ai/generate_ai_context_files.py --check`.
- **Notebooks:** las salidas de referencia se generan en macOS; reejecutar en Windows introduce ruido ajeno al cambio. Ejecutar con `tools/docs/execute_notebooks/execute_notebooks.py` (con `PYTHONPATH` en la raíz del repo, un notebook por llamada).

---

## 3. Decisiones

### 3.1 Tomadas (2026-10-27)

| Id | Pregunta | Decisión | Motivo | Se aplica en |
|---|---|---|---|---|
| DEC1 | ¿Qué hacer con `version`? | Deprecar en 0.27.0 (valor por defecto `None`, `FutureWarning` siempre que se pase) y retirar en 0.29.0 haciendo keyword-only los argumentos que siguen | Nunca se publicaron versiones (sin tags), ningún notebook ni ejemplo lo usa y HF lo ignora; versionar los datos costaría un tag por release en cinco repos | PR 2 |
| DEC2 | Formato largo (fechas repetidas): ¿cómo se ordena? | Por fecha con `sort_index(kind='stable')`, y subir a HF `m5`, `bdg2_daily` y `bdg2_hourly` ya ordenados por fecha | Conserva el orden de las series dentro de cada fecha sin cambiar el resultado actual; con los parquet ordenados, `is_monotonic_increasing` evita la copia de `sort_index` y baja el pico de memoria | PR 3 (3.2), PR 5b |
| DEC3 | ¿Avisar cuando `asfreq` cambia las filas? | Sí: `MissingValuesWarning` al insertar filas NaN, `UserWarning` al descartar filas | Hoy los 59 días ausentes de `wikipedia_visits` aparecen como NaN sin aviso | PR 3 (3.1) |
| DEC4 | ¿Dónde vive el generador del README de datos? | En `skforecast-datasets` (`tools/generate_readme.py`), sin CI, ejecutado a mano | El README es del repo de datos; el script importa el registro del skforecast instalado | PR 5a |
| DEC5 | Zona horaria de `vic_electricity` | Documentarla en `description`; el fichero no cambia | Cinco notebooks y dos scripts de docs lo usan, y todas las versiones de skforecast leen el mismo fichero | PR 1 (1.7) |
| DEC8 | ¿Dónde corre el test que descarga todos los datasets? | Test de pytest opt-in (`SKFORECAST_TEST_DATASETS=1`), ejecutado a mano antes de cada release y tras cambiar el repo de datos | `slow` no lo saca de la CI, que descargaría unos 175 MB en 15 trabajos | PR 4 |

### 3.2 Pendientes

#### DEC6. Ficheros huérfanos de `data/`

Para cada uno: registrarlo en `datasets` (con `description`, `source`, `index_col`, `freq`) o solo listarlo en una sección `## Other files` del README. Datos para decidir:

| fichero | columnas | filas | observación |
|---|---|---|---|
| bitcoin.csv | date, open, high, low, close, volume, market cap | 3171 | diario 2013-04-28 a 2022-01-01 |
| demo_multi_series.csv | series_id, timestamp, value | 1074 | formato largo, 5 series `id_1000` a `id_1004`, 2016 |
| demo_multi_series_exog.csv | series_id, timestamp, sin_day_of_week, cos_day_of_week, air_temperature, wind_speed | 1830 | exógenas del anterior |
| energy_production_shutdown.csv | date, production | 1095 | diario 2012-01-01 a 2014-12-30 |
| generacion_energia_US_source.csv | date_time, series, source | 189664 | horario, UTC (`Z`), formato largo por `source` |
| intermittent_demand.csv | date_time, users | 17544 | horario 2011 a 2012, demanda intermitente |
| visitas_por_dia_web_cienciadedatos_2020_2021.csv | date, users | 522 | extensión de `website_visits` hasta 2021-12-04, mismo formato `%d/%m/%y` |

Antes de borrar alguno, buscar su nombre en `docs/`, en el blog de cienciadedatos.net y en notebooks de versiones anteriores: la URL `.../main/data/<fichero>` puede estar enlazada desde fuera.

#### DEC7. Licencias por dataset

Joaquín indica la licencia de cada fuente (Kaggle, UCI, ABS, EUROCONTROL/ONS, tsibbledata, Monash, ...). La sesión solo prepara la estructura (PR 5d).

---

## 4. PR 1: correcciones pequeñas en `datasets.py`

**Rama:** `fix/datasets-module-small-fixes`. **Ficheros:** `skforecast/datasets/datasets.py`, los cuatro ficheros de tests, nuevo `test_datasets_registry.py`, `docs/user_guides/datasets.ipynb` (solo reejecución), `docs/releases/releases.md`. **No cambia ninguna firma pública.**

### 1.1 `australia_tourism`: frecuencia y periodo (metadatos)

- `'freq': 'Q'` pasa a `'freq': 'QS'`: las fechas son inicios de trimestre y `'Q'` es un alias obsoleto (`FutureWarning`). **No cambia los datos devueltos**: el índice no es único y `fetch_dataset` nunca aplica `asfreq` a este dataset. Afecta al metadato que muestran `show_datasets_info` y el README y a quien use `datasets['australia_tourism']['freq']`.
- `description`: «from 1998 Q1 to 2016 Q4» pasa a «from 1998 Q1 to 2017 Q4», y añadir «Long format: one row per region, state, purpose and quarter (304 series).».

Test en el nuevo `test_datasets_registry.py` (sin red):

```python
# Unit test datasets registry
# ==============================================================================
import warnings
import pytest
import pandas as pd
from pandas.tseries.frequencies import to_offset
from skforecast.datasets import datasets

REQUIRED_KEYS = {
    'url', 'sep', 'index_col', 'date_format', 'freq', 'file_type',
    'description', 'source'
}


@pytest.mark.parametrize('name', sorted(datasets), ids=lambda n: f'dataset: {n}')
def test_datasets_registry_entry_is_consistent(name):
    """
    Every entry of the `datasets` registry has the required keys, a supported
    file type, a separator consistent with the file type, a frequency alias
    accepted by pandas without warnings and a URL with the `{version}`
    placeholder only when hosted on GitHub.
    """
    info = datasets[name]
    assert set(info) == REQUIRED_KEYS
    assert info['file_type'] in ('csv', 'parquet')
    if info['file_type'] == 'csv':
        assert isinstance(info['sep'], str)
    else:
        assert info['sep'] is None
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        to_offset(info['freq'])
    assert ('{version}' in info['url']) == ('raw.githubusercontent.com' in info['url'])
    assert info['description'].strip() and info['source'].strip()


def test_datasets_registry_australia_tourism_quarter_start_freq():
    """
    The frequency of `australia_tourism` is quarter start, which matches the
    dates of the file (1998-01-01, 1998-04-01, ...).
    """
    freq = datasets['australia_tourism']['freq']
    results = pd.date_range('1998-01-01', periods=3, freq=freq)
    expected = pd.to_datetime(['1998-01-01', '1998-04-01', '1998-07-01'])
    assert list(results) == list(expected)
```

Comprobar que el primer test falla en la base por el `FutureWarning` de `'Q'` antes de arreglarlo (`git stash push skforecast/datasets/datasets.py`). Si la PR 2 se hace después, la aserción de `{version}` sigue valiendo (el marcador se mantiene hasta 0.29.0).

### 1.2 Tipo de fichero desconocido (robustez)

Validar `file_type` antes de descargar, justo tras comprobar `name`:

```python
    file_type = datasets[name]['file_type']
    if file_type not in ('csv', 'parquet'):
        raise ValueError(
            f"Unknown file type '{file_type}' for dataset '{name}'. "
            f"Supported file types are 'csv' and 'parquet'."
        )
```

Hoy un tipo desconocido acaba en `UnboundLocalError: cannot access local variable 'df'` (precedido de «Could not preprocess» si `raw=False`). Test con `unittest.mock.patch.dict(datasets, {'_tmp': {**datasets['h2o'], 'file_type': 'xlsx'}})` y `pytest.raises(ValueError, match=re.escape(...))`. Sin nota de versión (solo se alcanza con un registro mal escrito).

### 1.3 `sep` dentro de `kwargs_read` (bug)

Hoy `fetch_dataset('h2o', kwargs_read={'sep': ','})` falla con `ValueError: Error reading dataset ... got multiple values for keyword argument 'sep'`. Bloque CSV:

```python
    if file_type == 'csv':
        kwargs_read = {'sep': datasets[name]['sep'], **kwargs_read}
        try:
            df = pd.read_csv(url, **kwargs_read)
```

Gana el valor del usuario. Test: `patch('pandas.read_csv', return_value=mock_df)` y comprobar `mock_read.call_args.kwargs['sep']` (`';'` con `kwargs_read={'sep': ';'}`, `','` sin él); se puede añadir como caso a `test_fetch_dataset_kwargs_read`.

### 1.4 Encadenar excepciones y pista de pyarrow

Los dos `raise ValueError(...)` de lectura pasan a `raise ValueError(...) from e`. En el bloque parquet, distinguir el motor ausente:

```python
    elif file_type == 'parquet':
        try:
            df = pd.read_parquet(url, **kwargs_read)
        except ImportError as e:
            raise ImportError(
                f"Reading dataset '{name}' requires pyarrow or fastparquet: "
                f"pip install pyarrow. Original error: {e}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Error reading dataset '{name}' from {url}: {str(e)}."
            ) from e
```

Tests: `patch('pandas.read_parquet', side_effect=ImportError('Unable to find a usable engine'))` con `pytest.raises(ImportError, match=re.escape('requires pyarrow'))`; y en el caso parquet de `test_fetch_dataset_invalid_version_raises` comprobar `excinfo.value.__cause__ is not None`.

### 1.5 `show_datasets_info` con un string o una lista vacía (bug)

Hoy `datasets_names = datasets_names or sorted(datasets)`: `show_datasets_info('h2o')` imprime tres «not available» (uno por carácter) y `show_datasets_info([])` muestra todos. Sustituir por:

```python
    if datasets_names is None:
        datasets_names = sorted(datasets)
    elif isinstance(datasets_names, str):
        datasets_names = [datasets_names]
    else:
        datasets_names = list(datasets_names)
```

Se acepta cualquier iterable (lista, tupla, `set`, `datasets.keys()`, `pandas Index`), que hoy funciona; rechazar todo lo que no sea lista o tupla rompería esos usos. Un `int` falla con el `TypeError` de `list()`, suficiente. Docstring: `datasets_names : str, list, default None`. Tests en `test_show_datasets_info.py`: string (panel de `h2o`, sin «not available»), lista vacía (`captured.out == ''`), tupla.

### 1.6 Panel de `_print_dataset_info` envuelto dos veces (bug visual)

`textwrap.fill(..., width=80)` corta el texto y rich lo vuelve a envolver al ancho real de la consola: en terminales estrechas quedan palabras sueltas, y la URL se parte en los guiones de `skforecast-datasets` (se ve en las salidas guardadas de `datasets.ipynb` y `migrate-from-prophet.ipynb`). Cambios:

- Eliminar `import textwrap` y los tres `textwrap.fill`.
- Escapar el texto para rich (`from rich.markup import escape`) en `description`, `source` y URL, y marcar la URL como enlace: `f"[bold]URL:[/bold]\n[link={url}]{escape(url)}[/link]"`. Hoy ningún texto rompe el markup (los corchetes de las fuentes ETT empiezan por dígito), pero el registro no debería poder hacerlo.
- Mantener `Panel(content, title=..., expand=False)`.

Test en `test_print_dataset_info.py`: con `monkeypatch.setenv('COLUMNS', '200')` (rich respeta `COLUMNS` sin tty), la URL completa de `bike_sharing_extended_features` aparece entera en una línea de `captured.out`. Los tests existentes deben seguir pasando. En terminales más estrechas que la URL rich la seguirá plegando: el objetivo es quitar el doble envoltorio, no garantizar una línea.

### 1.7 Textos del registro

Todos son cambios de `description` o `source`; el README generado (PR 5a) los hereda, así que conviene hacerlos antes.

- `vic_electricity` (DEC5): añadir «`Time` is in UTC (suffix Z, parsed as a naive datetime); `Date` and `Holiday` refer to local time in Melbourne (UTC+10, UTC+11 in summer).» y el punto final que falta.
- `m4_daily`/`m4_hourly`: «Timestamps are artificial (the M4 series have no real dates): all at 12:00 and spanning 1959 to 2039. Long format: one row per series and timestamp.».
- `m5`: «Series ids keep the `_evaluation` suffix of the Kaggle files; `sell_price` is NaN before the product was first sold. Long format: one row per series and date.».
- `store_sales`, `ashrae_daily`, `bdg2_daily`, `bdg2_hourly`: añadir «Long format: one row per <unidad> and date.» (lo usa la PR 3 en lugar de enumerar los datasets en el docstring).
- ETT (`ett_m1`, `ett_m2`, `ett_m2_extended`): «between July 2016 and July 2018 (... = 70,080 data points)» pasa a «from 2016-07-01 to 2018-06-26 (69,680 data points every 15 minutes)».
- Erratas: «the The Building Data Genome» (cuatro entradas `bdg2_*`); espacios ausentes en `source` de `h2o` («Practice(3rd Edition)», «fpp3package/,https»); «(2022).tsibbledata» (dos entradas); «several air chemical pollutant» (dos entradas, «pollutants»); espacio final en `description` de `h2o`; doble espacio en «Repository  » (`m4_*`).
- `source` de `m4_*` lleva `\n` incrustados: sin `textwrap` (1.6) pasan a ser saltos de línea reales en el panel. Mantenerlos (separan cita, conversión y categorías) y quitar el espacio previo a cada `\n`.
- Comentario de una línea sobre las entradas parquet (opcional): `# date_format is ignored: the column is already datetime64 in the file`.

### 1.8 Docstrings del módulo

Las cuatro funciones usan `name: type` y `df: pandas DataFrame` en `Returns`; la guía pide `name : type, default value`. Corregir todas. Además:

- `show_datasets_info(..., version: str = 'latest')` pasa a `str | int` como las otras dos (si se hace la PR 2, la cambia de nuevo).
- `raw` de `fetch_dataset`: decir que el índice se ordena y que la frecuencia solo se fija si las fechas son únicas (la PR 3 amplía el texto).
- `load_demo_dataset`: la línea de `Returns` es `y : pandas Series`.

### 1.9 Reejecutar `docs/user_guides/datasets.ipynb`

Su salida guardada de `show_datasets_info()` muestra las descripciones viejas y la URL partida. Reejecutar solo ese notebook (idealmente en macOS; en Windows revisar que el diff se limite a los paneles). Los demás notebooks con paneles se actualizarán en su próxima ejecución; no hace falta tocarlos.

### 1.10 Nota de versión (último commit)

Subsección **Fixed** de 0.27.0, una entrada, sin highlight:

```markdown
+ Fixed several issues in <code>[fetch_dataset]</code> and <code>[show_datasets_info]</code>: `kwargs_read` could not include `sep`; a missing parquet engine raised a generic `ValueError` instead of an `ImportError` suggesting `pip install pyarrow`; `show_datasets_info` with a string instead of a list printed one error per character, and with an empty list printed every dataset; the information panel wrapped its text twice in narrow terminals; and the metadata of `australia_tourism` declared the deprecated quarter-end frequency `Q` instead of `QS` and the wrong last quarter. ([#NNNN](https://github.com/skforecast/skforecast/pull/NNNN))
```

Ejecutar el check del paso 8 del skill `release-note`.

### 1.11 Verificación

```bash
ruff check skforecast/datasets/
pytest skforecast/datasets/tests/tests_datasets/test_datasets_registry.py -x -q
pytest skforecast/datasets/tests/tests_datasets/ -q          # necesita red
python tools/ai/generate_ai_context_files.py --check         # no debe cambiar nada
```

Más el apéndice A: «PREPROCESS OK» en los 26 datasets de GitHub y `asfreq_inserted=59` solo en `wikipedia_visits`.

---

## 5. PR 2: deprecar `version` y apoyar `load_demo_dataset` en `fetch_dataset` (DEC1)

**Rama:** `feature/datasets-version-deprecation`.

### 2.1 `load_demo_dataset` sobre `fetch_dataset`

Hoy lee `h2o.csv` por su cuenta con `names=['y', 'datetime']` (depende del orden `x,fecha`), sin el manejo de errores de `fetch_dataset`. Nueva implementación con el mismo resultado exacto (Series `y`, índice `datetime`, freq `MS`, 204 valores float64):

```python
    y = fetch_dataset('h2o', raw=False, verbose=False)['x']
    y = y.rename('y')
    y.index.name = 'datetime'

    if verbose:
        _print_dataset_info('h2o', shape=y.shape)

    return y
```

No pasa `version` a `fetch_dataset`: así el aviso de 2.2 sale una sola vez y desde `load_demo_dataset`.

Test: ampliar `test_load_demo_dataset_returns_correct_series` con `y.name == 'y'`, `y.index.name == 'datetime'`, `y.dtype == float` y los cinco primeros valores escritos a mano (`0.429795, 0.400906, 0.432159, 0.492543, 0.502369`; comprobarlos antes con `fetch_dataset('h2o', raw=True).head()`).

### 2.2 Deprecar `version` (DEC1)

Mismo patrón que `differentiation` en `skforecast/model_selection/_split.py`: **valor por defecto `None` y aviso cuando no es `None`**. Avisar solo para valores distintos de `'latest'` dejaría sin aviso a quien escribe `version='latest'` (todos los tests del módulo lo hacen) y su código fallaría sin previo aviso en 0.29.0.

1. Firmas: `version: str | int | None = None` en `fetch_dataset`, `load_demo_dataset` y `show_datasets_info`. `_print_dataset_info` pierde `version` (privada; siempre `main`).
2. Helper privado, llamado al principio de cada función pública:

```python
def _warn_deprecated_version(version: str | int | None) -> None:
    """
    Warn that `version` is deprecated when it is passed.
    """
    # TODO: Remove in version 0.29.0 together with the `version` argument.
    if version is not None:
        warnings.warn(
            "`version` is deprecated since version 0.27.0 and has no effect: "
            "datasets are always fetched from the main branch of "
            "skforecast-datasets. It will be removed in version 0.29.0.",
            FutureWarning,
            stacklevel=3
        )
```

   Las URL se formatean siempre con `version='main'`.
3. Docstrings: `version : str, int, default None` con «**Deprecated** since version 0.27.0, it has no effect and will be removed in version 0.29.0. Datasets are always fetched from the main branch of skforecast-datasets.».
4. Tests: `test_fetch_dataset_invalid_version_raises` pasa a `test_fetch_dataset_version_FutureWarning` (avisa con `pytest.warns(FutureWarning, match=re.escape(...))` y lee la URL de `main`, comprobado con `patch('pandas.read_csv')`); avisa también con `version='latest'`; sin `version` no avisa. `test_show_datasets_info_version_in_url` y `test_print_dataset_info_version_substituted_in_url`: la URL impresa contiene `/main/`. `load_demo_dataset(version='x')` avisa una sola vez. Quitar `version='latest'` del resto de tests del módulo.
5. Nota de versión, **Changed**:

```markdown
+ The `version` argument of <code>[fetch_dataset]</code>, <code>[load_demo_dataset]</code> and <code>[show_datasets_info]</code> is deprecated and has no effect: datasets are always fetched from the main branch of skforecast-datasets, which has no published versions. Passing it emits a `FutureWarning`; it will be removed in version 0.29.0. ([#NNNN](https://github.com/skforecast/skforecast/pull/NNNN))
```

6. `ai-context-sync`: ni `tools/ai/llms-base.txt` ni `skills/` usan `version`; ejecutar `--check` igualmente.
7. **Al retirarlo en 0.29.0:** `version` es el segundo parámetro posicional. Si solo se borra, `fetch_dataset('h2o', 'latest')` pasaría `'latest'` a `raw` (verdadero) y devolvería datos sin preprocesar sin error. Al retirarlo, hacer keyword-only todo lo que sigue a `name` (`def fetch_dataset(name, *, raw=False, ...)`) y lo mismo en `load_demo_dataset(*, verbose=True)`. Anotarlo en el `TODO`.

---

## 6. PR 3: huecos de `asfreq` y formato largo (DEC2, DEC3)

**Rama:** `feature/fetch-dataset-gaps-and-long-format`.

### 3.1 Aviso cuando `asfreq` cambia las filas (DEC3)

En el bloque `if not raw:` de `fetch_dataset`, sustituir `if df.index.is_unique: df = df.asfreq(freq)` por:

```python
            if df.index.is_unique:
                index_before = df.index
                df = df.asfreq(freq)
                n_inserted = len(df.index.difference(index_before))
                n_dropped = len(index_before.difference(df.index))
                if n_inserted > 0:
                    warnings.warn(
                        f"Dataset '{name}' has {n_inserted} missing dates for "
                        f"frequency '{freq}'. Rows with NaN values have been "
                        f"introduced after setting the frequency.",
                        MissingValuesWarning,
                        stacklevel=2
                    )
                if n_dropped > 0:
                    warnings.warn(
                        f"{n_dropped} rows of dataset '{name}' are not aligned "
                        f"with frequency '{freq}' and have been dropped after "
                        f"setting the frequency. The frequency of the dataset "
                        f"may be wrong, use `raw=True` to get the original data.",
                        stacklevel=2
                    )
```

Import relativo: `from ..exceptions import MissingValuesWarning` (su `__str__` ya añade la línea «You can suppress this warning using ...»). Los avisos salen dentro del `try` del preprocesado pero no disparan el `except`.

Tests (`test_fetch_dataset.py`, `patch('pandas.read_csv', return_value=...)` sobre `h2o`): fechas `1991-07-01`, `1991-08-01`, `1991-10-01` → `pytest.warns(MissingValuesWarning, match=re.escape("has 1 missing dates for frequency 'MS'"))`, 4 filas y NaN en `1991-09-01`; fechas `1991-07-01`, `1991-07-15` → aviso de filas descartadas y 1 fila. `test_fetch_dataset_website_visits_day_first_dates` (1402) convierte `UserWarning` en error: `website_visits` no tiene huecos y sigue pasando.

Notebook: `migrate-from-prophet.ipynb` carga `wikipedia_visits` y mostrará el aviso (el texto del notebook ya explica que interpola los huecos). Reejecutarlo, o dejar que lo recoja la próxima ejecución de docs y decirlo en la PR.

### 3.2 Orden estable del formato largo (DEC2)

`df = df.sort_index()` pasa a `df = df.sort_index(kind='stable')` (dentro del `if not df.index.is_monotonic_increasing:` de 1402). Conserva el orden de las series del fichero dentro de cada fecha y es algo más rápido (sección 1.4). Ampliar `test_fetch_dataset_m5_long_format_unsorted`: con ids `A, A, B, B` y fechas `29, 30, 29, 30`, el resultado es `id == ['A', 'B', 'A', 'B']` y `sales == [1, 3, 2, 4]`.

La otra mitad de DEC2 (parquet de HF ya ordenados) está en la PR 5b y no cambia código.

### 3.3 Documentación

Docstring de `fetch_dataset`, parámetro `raw` y nueva sección `Notes` (sin enumerar datasets, que se quedan desfasados; cada uno lo dice en su `description` desde 1.7):

```
    raw : bool, default False
        If `True`, the raw dataset is returned. If `False`, the column with the
        date/time is set as index, converted to datetime and sorted. When the
        dates are unique, the frequency of the dataset is set with `asfreq`
        (rows with NaN are introduced for missing dates, with a
        `MissingValuesWarning`).

    Notes
    -----
    Datasets with several series in long format (one row per series and date,
    see their description in `show_datasets_info`) have repeated dates in the
    index, so no frequency is set and the rows are sorted by date, keeping the
    order of the series within each date. Use `reshape_series_long_to_dict` or
    `pivot_table` to get one series per key or column.
```

Y una celda markdown en `docs/user_guides/datasets.ipynb`, tras la de `raw=True`, con la misma idea (solo markdown; `execute_notebooks.py ... --check` debe seguir limpio).

### 3.4 Nota de versión, **Changed**

```markdown
+ <code>[fetch_dataset]</code> warns with a `MissingValuesWarning` when setting the frequency introduces rows with NaN for missing dates (today only `wikipedia_visits`, with 59 missing days), keeps the order of the series within each date when sorting long-format datasets, and documents that these datasets have repeated dates and therefore no frequency. ([#NNNN](https://github.com/skforecast/skforecast/pull/NNNN))
```

### 3.5 Verificación

Como la PR 1, más el apéndice A: `wikipedia_visits` muestra el nuevo aviso y ningún otro dataset de GitHub avisa.

---

## 7. PR 4: test de regresión contra los ficheros reales (DEC8)

**Rama:** `chore/datasets-regression-test`. Sin nota de versión. Va después de la PR 3, cuyo comportamiento fija. Nuevo fichero `skforecast/datasets/tests/tests_datasets/test_fetch_dataset_all_datasets.py` (separado para poder lanzarlo solo).

```python
# Unit test fetch_dataset on every dataset hosted on GitHub
# ==============================================================================
import os
import warnings
import pytest
import pandas as pd
from skforecast.datasets import datasets, fetch_dataset

EXPECTED_DATASETS = {
    # name: (rows, cols, freq)
    'h2o': (204, 1, 'MS'),
    'h2o_exog': (195, 3, 'MS'),
    'fuel_consumption': (644, 5, 'MS'),
    'items_sales': (1097, 3, 'D'),
    'air_quality_valencia': (43824, 10, 'h'),
    'air_quality_valencia_no_missing': (43824, 10, 'h'),
    'website_visits': (421, 1, 'D'),
    'bike_sharing': (17544, 11, 'h'),
    'bike_sharing_extended_features': (17352, 90, 'h'),
    'australia_tourism': (24320, 4, None),
    'uk_daily_flights': (1118, 1, 'D'),
    'wikipedia_visits': (2964, 1, 'D'),
    'vic_electricity': (52608, 4, '30min'),
    'vic_electricity_classification': (8736, 5, 'h'),
    'store_sales': (913000, 3, None),
    'bicimad': (3022, 1, 'D'),
    'm4_daily': (10023836, 3, None),
    'm4_hourly': (373372, 3, None),
    'bdg2_daily_sample': (274, 2, 'D'),
    'bdg2_hourly_sample': (6553, 2, 'h'),
    'ett_m1': (69680, 7, '15min'),
    'ett_m2': (69680, 7, '15min'),
    'ett_m2_extended': (69680, 16, '15min'),
    'expenditures_australia': (505, 1, 'MS'),
    'public_transport_madrid': (715, 5, 'D'),
    'turbine_emission': (36733, 11, 'h'),
}

requires_datasets_download = pytest.mark.skipif(
    os.environ.get('SKFORECAST_TEST_DATASETS') != '1',
    reason='Downloads every dataset hosted on GitHub (about 175 MB).'
)


@pytest.mark.slow
@requires_datasets_download
@pytest.mark.parametrize(
    'name', sorted(EXPECTED_DATASETS), ids=lambda n: f'dataset: {n}'
)
def test_fetch_dataset_all_github_datasets_are_preprocessed(name):
    """
    Every dataset hosted on GitHub is returned with a DatetimeIndex named as
    its `index_col`, the expected shape and the declared frequency (None for
    long-format datasets with repeated dates), without the 'Could not
    preprocess' warning.
    """
    rows, cols, freq = EXPECTED_DATASETS[name]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        df = fetch_dataset(name, verbose=False)
    assert not [w for w in record if 'Could not preprocess' in str(w.message)]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == datasets[name]['index_col']
    assert df.index.is_monotonic_increasing
    assert df.shape == (rows, cols)
    assert df.index.freqstr == freq


def test_fetch_dataset_all_github_datasets_are_covered():
    """
    `EXPECTED_DATASETS` lists every dataset of the registry hosted on GitHub,
    so a new dataset cannot be added without its expected shape.
    """
    github = {
        name for name, info in datasets.items()
        if 'raw.githubusercontent.com' in info['url']
    }
    assert set(EXPECTED_DATASETS) == github
```

El segundo test no descarga nada y sí corre en la CI. Ejecución del primero (unos 175 MB, 1 a 2 minutos):

```bash
SKFORECAST_TEST_DATASETS=1 pytest skforecast/datasets/tests/tests_datasets/test_fetch_dataset_all_datasets.py -q
```

En PowerShell: `$env:SKFORECAST_TEST_DATASETS = '1'; pytest ...`. HF queda fuera por tamaño (`ashrae_daily`, 3.6 MB, puede añadirse a mano). Si se registran ficheros huérfanos (DEC6), ampliar la tabla en el mismo cambio. Para endurecerlo, añadir el primer y último valor del índice de cada dataset.

Opcional en la misma PR: marcar con un marcador `network` (declarado en `pyproject.toml`) los nueve tests actuales que descargan, para poder deseleccionarlos sin red con `-m "not network"`. No cambia la CI.

Añadir a la checklist de `/release-bump` el paso «ejecutar el test de regresión de datasets con `SKFORECAST_TEST_DATASETS=1`».

---

## 8. PR 5: repositorio `skforecast-datasets`

Repositorio separado (`https://github.com/skforecast/skforecast-datasets`, rama `main`). Cada punto es un commit. Lo que necesita credenciales de HF o licencias queda para Joaquín.

### 5a. README generado desde el registro (DEC4) y BOM

Nuevo `tools/generate_readme.py` (Python 3.10+, depende solo de `skforecast`):

- `from skforecast.datasets import datasets`; escribe `README.md` con: cabecera (los dos párrafos actuales corregidos y el ejemplo `from skforecast.datasets import fetch_dataset` / `data = fetch_dataset(name="h2o")`); el aviso «Generated by tools/generate_readme.py from the `datasets` registry of skforecast X.Y.Z; do not edit by hand»; y una sección `### <name>` por dataset en orden alfabético con `+ url` (con `{version}` sustituido por `main`), `+ sep`, `+ index_col`, `+ date_format`, `+ freq`, `+ file_type`, `+ description`, `+ source`.
- Sección final `## Other files` con los ficheros huérfanos no registrados (DEC6), una línea cada uno.
- Opción `--check`: regenera en memoria y sale con código 1 si difiere.

La primera ejecución corrige de golpe las entradas desalineadas (sección 1.4). Revisar el diff línea a línea antes del commit.

Commit aparte: quitar el BOM UTF-8 de `data/uk_daily_flights.csv`. pandas ya lo ignoraba, así que `fetch_dataset` no cambia; comprobar con el apéndice A.

**No cambiar** el formato de fecha de `data/visitas_por_dia_web_cienciadedatos.csv`: el registro (1402) y dos notebooks parsean `%d/%m/%y`, y skforecast 0.26 y anteriores leen el mismo fichero de `main`. Si algún día se cambia, publicarlo con otro nombre. Tampoco reordenar las columnas de `h2o.csv` (sección 1.5) ni tocar `vic_electricity.csv` (DEC5).

### 5b. Parquet de HF ordenados por fecha (DEC2, requiere el token de HF de Joaquín)

Descargar `m5_daily.parquet`, `bdg2_daily.parquet` y `bdg2_hourly.parquet`; leer con `pq.read_table`, ordenar con `table.sort_by([(col, 'ascending')])` (`datetime` en m5, `timestamp` en bdg2; conserva los tipos `dictionary`), guardar con el mismo tamaño de row group (57, 2 y 27 row groups), comprobar `is_monotonic_increasing` y subir con `huggingface_hub.upload_file`. Medir después el pico de `fetch_dataset('m5')` (en la PR 1402 fue 10.3 GB): debe bajar porque `sort_index` deja de copiar. Todas las versiones de skforecast leen `main` de HF; las que ordenan devuelven lo mismo, y con `raw=True` cambia el orden de filas (anotarlo en la nota de la release en la que se suba).

### 5c. Ficheros huérfanos (DEC6)

Aplicar lo que Joaquín decida en DEC6: registrar en `skforecast` (PR aparte en skforecast, ampliando `EXPECTED_DATASETS` de la PR 4) o listar en `## Other files`.

### 5d. Licencias (DEC7)

Añadir `+ license:` a cada sección del README. Si además se añade una clave `license` al registro, actualizar `REQUIRED_KEYS` del test 1.1, el generador y `show_datasets_info`.

---

## 9. Fuera de alcance (posibles mejoras futuras)

No se proponen para 0.27.0; se anotan para no perderlas.

- **Caché local:** cada llamada descarga el fichero (66 MB `m4_daily`, cientos de MB `m5`). Un `data_home` como el de scikit-learn (`~/skforecast_data`, variable `SKFORECAST_DATA`) evitaría descargas repetidas y permitiría trabajar sin red.
- **Tiempo de espera:** `pd.read_csv(url)` no tiene timeout; una red lenta bloquea indefinidamente.
- **Metadatos de formato largo en el registro:** una clave como `series_id` permitiría a `fetch_dataset` y a la documentación saber qué datasets son largos sin buscarlo en el texto, y ofrecer `reshape_series_long_to_dict` directamente.
- **pandas 3:** el módulo no se ha revisado con pandas 3 (columnas de texto como `str` en lugar de `object`, alias de frecuencia obsoletos). Incluirlo en el estudio de migración en curso.

---

## 10. Riesgos

- `MissingValuesWarning` (PR 3) es subclase de `UserWarning`: un test o notebook que convierta `UserWarning` en error y cargue `wikipedia_visits` fallará. Hoy solo `migrate-from-prophet.ipynb` lo carga, sin filtros de avisos.
- `show_datasets_info([])` deja de mostrar todos (PR 1). Ningún notebook lo llama así.
- La deprecación de `version` (PR 2) convierte un 404 en un aviso con datos de `main`. En 0.29.0, sin parámetros keyword-only, una llamada posicional cambiaría de significado en silencio (2.2, punto 7).
- El test de regresión (PR 4) depende de que los ficheros del repositorio de datos no cambien; si cambian, actualizar `EXPECTED_DATASETS` en el mismo cambio.
- La reordenación de los parquet de HF (PR 5b) cambia el orden de filas con `raw=True` para todas las versiones de skforecast.
- `docs/user_guides/datasets.ipynb` queda desactualizado si la PR 1 se fusiona sin reejecutarlo (1.9).

---

## Apéndice A. Script de comprobación con un clon local de los datos

Guardar como `dev/review_datasets_module/check_datasets_local.py` (no subir). Necesita un clon de `skforecast-datasets` y el skforecast de la rama en edición importable (`PYTHONPATH=<raíz del repo>`; comprobar `skforecast.__file__`). Aplica el preprocesado de `fetch_dataset` (con 1402) a los ficheros locales.

```python
# Check the datasets registry against a local clone of skforecast-datasets
# ==============================================================================
# Usage: python check_datasets_local.py /path/to/skforecast-datasets/data
import os
import sys
import warnings
import pandas as pd
import skforecast
from skforecast.datasets import datasets

print(skforecast.__file__)
local = sys.argv[1]
for name, info in datasets.items():
    url = info['url']
    if 'raw.githubusercontent.com' not in url:
        print(f"{name}: hosted at {url}, skipped")
        continue
    path = os.path.join(local, url.rsplit('/', 1)[-1])
    df = (pd.read_csv(path, sep=info['sep']) if info['file_type'] == 'csv'
          else pd.read_parquet(path))
    unique = None
    n_before = len(df)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        try:
            if df.index.name != info['index_col']:
                df.index = df.pop(info['index_col'])
            df.index = pd.to_datetime(df.index, format=info['date_format'])
            unique = df.index.is_unique
            if unique:
                df = df.asfreq(info['freq'])
            if not df.index.is_monotonic_increasing:
                df = df.sort_index(kind='stable')
            status = 'PREPROCESS OK'
        except Exception as e:
            status = f'PREPROCESS FAILS: {type(e).__name__}: {e}'
    inserted = len(df) - n_before if status == 'PREPROCESS OK' and unique else 0
    print(
        f"{name}: {status} shape={df.shape} unique={unique} "
        f"freq={getattr(df.index, 'freqstr', None)} asfreq_inserted={inserted} "
        f"warnings={[w.category.__name__ for w in record]}"
    )
```

Resultado esperado con la 1402 fusionada (sin ella, `website_visits` falla por el formato de fecha): «PREPROCESS OK» en los 26, `asfreq_inserted=59` solo en `wikipedia_visits`, `freq=None` en `australia_tourism`, `store_sales`, `m4_daily` y `m4_hourly`, y ningún aviso.

Para clonar sin bajar los 175 MB: `git clone --depth 1 --filter=blob:none --no-checkout <url> <dir>` y después `git -C <dir> checkout HEAD -- data/<fichero>` para los que hagan falta.

## Apéndice B. Comparación del README de datos con el registro

Para repetir la comparación de la sección 1.4 antes de que exista el generador:

```python
# Compare the README of skforecast-datasets with the registry
# ==============================================================================
# Usage: python check_datasets_readme.py /path/to/skforecast-datasets/README.md
import re
import sys
from skforecast.datasets import datasets

text = open(sys.argv[1], encoding='utf-8').read()
readme = {}
for section in re.split(r'^### ', text, flags=re.M)[1:]:
    name, _, body = section.partition('\n')
    readme[name.strip()] = dict(re.findall(r'^\+ (\w+): ?(.*)$', body, flags=re.M))
print('README only:', sorted(set(readme) - set(datasets)))
print('registry only:', sorted(set(datasets) - set(readme)))
differ = set()
for name in sorted(set(readme) & set(datasets)):
    for key in ('url', 'sep', 'index_col', 'date_format', 'freq', 'file_type'):
        expected = '' if datasets[name][key] is None else str(datasets[name][key])
        expected = expected.replace('{version}', 'main')
        got = readme[name].get(key, '<missing>').strip().strip("'\"")
        if got != expected:
            differ.add(name)
            print(f"{name}.{key}: README={got!r} registry={expected!r}")
print(f"{len(differ)} entries differ")
```

## Apéndice C. Cambios respecto a la versión 1 del plan

Comprobados el 2026-10-27 contra el código, los notebooks, la CI y un clon del repositorio de datos.

**Correcciones de hechos**
- `wikipedia_visits` sí se usa en un notebook (`migrate-from-prophet.ipynb`), y `store_sales` (formato largo) en `comparing-forecasting-models.ipynb`. Los recuentos de uso por dataset de la v1 no eran fiables (por ejemplo `h2o` aparece en unos 13 notebooks, casi siempre con `raw=True`); se sustituyen por la lista de usos relevantes (1.5).
- `australia_tourism`: la frecuencia `Q` no cambia los datos devueltos (índice no único, nunca se aplica `asfreq`); es un error de metadatos. Su `description` tiene además mal el último trimestre (2017 Q4, no 2016 Q4).
- `version` sí funciona con nombres de rama (`feature_public_transport_madrid`); lo que nunca ha existido son versiones publicadas.
- La CI ejecuta los tests `slow` (sin filtro `-m`), así que el test de regresión de la v1 habría descargado unos 175 MB (no 150) en cada uno de los 15 trabajos de la matriz.
- El README de datos difiere en 18 entradas hoy (19 tras 1402), no en 15: faltaban las URL con `refs/heads/main` y el `sep` vacío de `bdg2_*_sample`.

**Cambios de diseño**
- Deprecación de `version`: valor por defecto `None` y aviso siempre que se pase, como `differentiation`; la v1 no avisaba con `version='latest'`. Se añade el riesgo posicional al retirarlo y la interacción con `load_demo_dataset`.
- `show_datasets_info`: aceptar cualquier iterable que no sea string; la v1 rompía `set`, `dict_keys` o `pandas Index`, que hoy funcionan.
- Tipo de fichero desconocido: se valida antes de descargar, no después.
- Formato largo: `sort_index(kind='stable')` como parte de DEC2; conserva el orden de las series aunque los parquet no se vuelvan a subir.
- El docstring de `fetch_dataset` no enumera datasets de formato largo; cada `description` lo indica.

**Mejoras que faltaban**
- Erratas y datos incorrectos en `description`/`source` (ETT, `bdg2_*`, `h2o`, `vic_electricity`, `air_quality_*`, `m4_*`).
- Formato de docstrings del módulo (`name : type`).
- Reejecutar `datasets.ipynb` en la PR 1.
- Consecuencia de UTC en `vic_electricity` para las variables de calendario.
- Lista de mejoras fuera de alcance (caché, timeout, metadatos de formato largo, pandas 3).

**Estructura**
- Resumen inicial con tablas de PRs y decisiones.
- Decisiones renombradas a DEC1 a DEC8 (en la v1, D1 a D3 eran a la vez decisiones y apartados de la PR D) y PRs numeradas del 1 al 5, con apartados `n.m`.
- `load_demo_dataset` pasa a la PR 2 (comparte la gestión de `version`); el test de regresión queda solo en la PR 4.
- Ramas con tipos válidos (`change/` y `test/` no lo son).
- Se quitan las referencias a sesiones cloud y a comandos solo de Linux (`/usr/bin/time -v`, `sed -i`).
- El plan pasa a vivir en `0.27.x` (antes, sin seguimiento en git), con una columna «Estado» en la tabla 0.1 y el flujo de trabajo de la sección 2.1, porque las PRs se hacen en días y sesiones distintos.

**Decisiones tomadas (2026-10-27)**
- DEC1, DEC3, DEC4, DEC5 y DEC8 con la opción recomendada; DEC2 con orden estable y además parquet de HF ordenados (la PR 5b deja de ser opcional). Se eliminan las alternativas descartadas: versionar los datos (antigua 2.3 y tag 5e), no ordenar el formato largo, generador en skforecast, reguardar `vic_electricity` en hora local, script o workflow en lugar del test opt-in.
- Quedan pendientes DEC6 (ficheros huérfanos, uno a uno) y DEC7 (licencias), que solo afectan a las PRs 5c y 5d.

**Incidencias**
- 2026-10-28: el commit `9b5f1f3e8` (estudio de pandas 3) borró este plan de `0.27.x`. Se restauró desde `5f4e2f3a6`, con la PR 0 (#1402) marcada como fusionada.
