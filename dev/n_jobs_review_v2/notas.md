# Diario de la revisión de `n_jobs` (v2)

Rama `perf/n-jobs-auto-work-threshold` (base `0.27.x`). Revisión a ciegas: hasta la
fase 3 no se han leído `dev/n_jobs_review/`, `dev/handoff_*.md`, el notebook de la FAQ,
las notas de release sobre `n_jobs` ni los cuerpos de los commits.

Máquina: contenedor Linux en la nube, 4 vCPU compartidas (Intel Xeon @ 2.10GHz),
15 GB RAM. Python 3.12.3, joblib 1.6.0, numpy 2.5.4, pandas 2.3.3,
scikit-learn 1.9.1, scipy 1.18.1, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10,
statsmodels 0.15.0, numba 0.68.1, threadpoolctl 3.8.0.

---

## Fase 1: mapa del código

### 1. Dónde se resuelve `n_jobs`

| Punto | Fichero:función | Regla |
|---|---|---|
| Validación | `model_selection/_utils.py:check_backtesting_input` (l. 564) | `int` o `'auto'`; cualquier entero pasa (0, -2...). |
| Reglas estáticas | `model_selection/_utils.py:select_n_jobs_backtesting` | `refit` entero ≠ 0/1 → 1. Forecaster ∉ {Recursive, RecursiveClassifier, RecursiveMultiSeries} → 1. Último paso de `Pipeline` es subclase de `LinearModel` → 1. `LGBMRegressor`/`LGBMClassifier` con `n_jobs != 1` → 1. Resto → `cpu_count() - 1`. |
| Umbral de trabajo | `model_selection/_utils.py:limit_n_jobs_by_workload` | Solo un backtesting suelto con `'auto'`. `n_fits == 0` (sin refit) → 1; `n_fits * fit_seconds < 2 s` (Linux) / `4 s` (Windows, macOS y resto) → 1; si no, `min(n_jobs, n_tasks)`. |
| Backtesting single | `_validation.py:_backtesting_forecaster` (l. 542-556, 610-656) | `'auto'` → reglas + umbral. Entero: `refit` intermitente fuerza 1 con aviso; `n_jobs <= 0` → `cpu_count()` (también -2, 0). `fit_seconds` = duración del ajuste inicial en el proceso principal. |
| Backtesting multiseries | `_validation.py:_backtesting_forecaster_multiseries` (l. 1392-1406, 1569-1617) | Igual; `n_tasks` excluye los folds saltados; `n_fits = sum(fit_folds)` sin el inicial. |
| Backtesting stats | `_validation.py:_backtesting_stats` (l. 2263-2310) | `refit=False` → 1 (aviso si el usuario pidió otro valor). `'auto'` → `select_n_jobs_backtesting` → **siempre 1** (ForecasterStats no está en la lista). Sin umbral. |
| Búsquedas (grid/random/bayes, single y multiseries) | `_search.py:_evaluate_grid_hyperparameters` (l. 557-567), `bayesian_search_forecaster` (l. 888-895) y sus equivalentes multiseries (l. 1738-1745, 2179-2186) | `'auto'` → `select_n_jobs_backtesting` por candidato y se pasa **un entero** a `backtesting_forecaster`, así que **no se aplica el umbral**: tampoco el caso `n_fits == 0` (refit=False). `OneStepAheadFold` ignora `n_jobs` (secuencial). |
| Búsquedas stats | `_search.py:_evaluate_grid_hyperparameters_stats` (l. 3326) | Pasa `n_jobs` tal cual a `backtesting_stats` → `'auto'` = 1. |
| `grid_search_equivalent_date` | `_search.py` (l. 3616) | `'auto'` → 1. |
| Direct | `direct/_forecaster_direct.py:__init__` (l. 633-643) y `_forecaster_direct_multivariate.py:__init__` (l. 589-599) → `utils/utils.py:select_n_jobs_fit_forecaster` | Se resuelve **una vez, en `__init__`**. `LinearModel` → 1; `LGBMRegressor` con `n_jobs != 1` → 1; resto → `cpu_count() - 1`. Sin umbral de trabajo. `set_params` no lo recalcula. |
| Multiseries recursivo, Recursive, Stats, EquivalentDate, Rnn | — | Sin paralelismo propio en `fit`/`predict` (ni en el bootstrapping de intervalos). |

### 2. Qué se paraleliza y cómo

Siempre `joblib.Parallel(n_jobs=n)` con los valores por defecto: backend `loky`,
`batch_size='auto'`, `pre_dispatch='2*n_jobs'`, `max_nbytes='1M'`, `mmap_mode='r'`.
Sin `prefer`, sin `parallel_config`, sin `threadpoolctl` en el paquete. Única excepción:
`_backtesting_stats` con `Sarimax` usa `mmap_mode='c'` (el filtro de Kalman no se puede
deserializar desde un memmap de solo lectura).

| Qué | Dónde | Datos enviados a cada tarea |
|---|---|---|
| Folds de un backtesting single | `_validation.py:_fit_predict_forecaster` | Trozos de `y`/`exog` por fold (`_prepare_fold_data`: con refit `y_train`+`exog_train`+`exog_test`, sin refit `last_window`+`exog_test`) **+ el forecaster entero ya ajustado** en el proceso principal (con su estimador ajustado, aunque el fold lo vaya a reajustar) + residuos fuera de muestra y binner si hay intervalos. |
| Folds multiseries | `_validation.py:_fit_predict_forecaster_multiseries` | Igual: `series_train`/`exog_train` solo en folds con refit; forecaster ajustado. Todos los trozos se materializan antes del `Parallel` (también con `n_jobs=1`), y en DataFrame `drop(columns=...)` copia. |
| Folds stats | `_validation.py:_fit_predict_forecaster_stats` | **`y` y `exog` completos** en cada tarea + la lista completa de folds + el forecaster. |
| Pasos de Direct | `direct/_forecaster_direct.py:_fit_one_step_estimator` | `self` (el forecaster, con `estimators_` reseteados a clones para no serializar los ajustados) + las matrices completas `X_train_autoreg`, `X_train_exog`, `X_train_calendar` y el dict `y_train` en cada tarea; arrays > 1 MB como memmap. |
| Candidatos de una búsqueda | — | **No** se paralelizan: bucle secuencial, y cada candidato paraleliza sus folds. Los workers de loky se reutilizan entre candidatos (mismo `n_jobs`). |
| Series | — | No se paralelizan. |
| Bootstrapping de intervalos | — | No se paraleliza (vectorizado sobre `n_boot`). |

Con `n_jobs=1`, `Parallel` ejecuta en el mismo proceso y el mismo objeto forecaster
pasa de fold en fold (lo que permite el refit intermitente).

### 3. Reglas de paralelismo anidado

- Detección: solo `LGBMRegressor`/`LGBMClassifier` (por nombre de clase y atributo
  `n_jobs`), tras desenvolver `Pipeline` (`estimator[-1]`). No se desenvuelve
  `TransformedTargetRegressor` ni otros meta-estimadores; subclases de LGBM con otro
  nombre no se detectan.
- No se detectan: `XGBRegressor` (`n_jobs=None` → hilos OpenMP), `CatBoostRegressor`
  (`thread_count=-1`), `HistGradientBoostingRegressor` (OpenMP, sin parámetro),
  `RandomForest/ExtraTrees(n_jobs=-1)` (joblib threads anidado).
- loky (joblib 1.6, `_parallel_backends.py:_prepare_worker_env`) arranca los workers con
  `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMBA_NUM_THREADS`...
  = `max(cpu_count() // n_jobs, 1)` salvo que ya estén en el entorno del padre. Con 4 CPU
  y 3 workers → 1 hilo por worker. Lo que no lea esas variables (CatBoost? LightGBM con
  `n_jobs=None` usa núcleos físicos?) no queda limitado: **a comprobar**.
- `Parallel` anidado dentro de un worker de loky → `ThreadingBackend` (nivel 1).
  `RandomForest(n_jobs=-1)` en un worker usa hilos con `cpu_count()`.
- Predicción LightGBM (`utils.py:_build_predict_function`): `num_threads` del ajuste,
  limitado por `OMP_NUM_THREADS` si está definido. RF/ET predicen árbol a árbol en
  Python (sin hilos).
- numba: solo en `skforecast/stats` (Arima, Ets, diferenciación), `@njit(cache=True)`,
  sin `parallel=True`. En frío, cada worker carga la caché de numba al importar.

### 4. Influencia de los parámetros

- `refit=True`: todos los folds salvo el primero ajustan → `n_fits = n_folds - 1`.
- `refit=False`: en un backtesting suelto, `'auto'` → 1 (no hay ajustes). **En una
  búsqueda, `'auto'` → `cpu_count()-1` igualmente.**
- `refit` entero: siempre 1 (el modelo se reutiliza entre folds).
- `fixed_train_size=False` (defecto): los folds crecen, `fit_seconds` del inicial
  subestima el trabajo (el código lo asume, ~80 %).
- Intervalos: no entran en la estimación de trabajo; con `refit=False` y bootstrapping
  caro, `'auto'` = 1 aunque predecir sea lo pesado.
- Nº de folds / candidatos: `n_jobs` se limita a `n_tasks` solo en el backtesting
  suelto con `'auto'`.
- Tamaño de datos: solo a través de `fit_seconds`.
- Stats: el código fuerza `refit=True` para todo estimador que no sea `Sarimax`
  (aviso `IgnoredArgumentWarning`); `Sarimax` admite `refit=False` y entero. Así que la
  suposición "en stats refit siempre es True" es cierta **salvo para `Sarimax`**.

### 5. Hipótesis (ordenadas por impacto esperado)

- **H1 (búsqueda, refit=False)**: en `grid/random/bayesian_search_forecaster` con
  `TimeSeriesFold(refit=False)` y RF/ET/HGB/XGB/CatBoost, `'auto'` elige `cpu_count()-1`
  para unos folds que solo predicen y envían el forecaster ajustado a los workers: más
  lento que `n_jobs=1`. Es el caso de búsqueda más habitual.
- **H2 (refit=True, forecaster ajustado enviado)**: en folds con refit se envía el
  forecaster con el estimador ya ajustado (RF/ET: MB por árbol), aunque el worker lo
  reajuste. Enviar un clon sin ajustar reduciría la IPC.
- **H3 (`cpu_count()-1`)**: con 2 vCPU (Colab) `'auto'` nunca paraleliza; con 4 deja una
  CPU libre mientras el proceso principal espera. `cpu_count()` podría ser más rápido si
  el principal no está ocupado serializando.
- **H4 (umbral de trabajo)**: el umbral de 2 s (Linux) está calibrado en frío; en
  caliente (notebook) el arranque es casi nulo. Comprobar el coste de arranque en frío y
  en caliente aquí y si el umbral separa bien.
- **H5 (lineales)**: la regla estática "lineal → 1" es correcta para tamaños típicos.
  Pero con `refit=True`, muchos folds y datos grandes, o con `QuantileRegressor`,
  `TheilSenRegressor`, `LassoCV` (subclases de `LinearModel` y caras), pierde. Con el
  umbral de trabajo, la regla estática podría sobrar.
- **H6 (anidado)**: dentro de un worker, HGB/XGB respetan `OMP_NUM_THREADS=1`;
  CatBoost (`thread_count=-1`) y LGBM (`n_jobs=None`) quizá no → sobresuscripción.
  Además, para HGB/XGB/CatBoost con datos medianos, 1 proceso × 4 hilos podría ganar a
  3 procesos × 1 hilo.
- **H7 (Direct)**: sin umbral de trabajo, `'auto'` arranca `cpu_count()-1` workers para
  cualquier estimador no lineal aunque los pasos sean pocos y baratos.
- **H8 (stats)**: con Arima/Ets y `refit=True` (forzado), `'auto'` = 1 siempre; con
  muchos folds el paralelismo podría ganar, a pesar del arranque de numba en frío.
- **H9 (intervalos sin refit)**: `refit=False` + bootstrapping caro → `'auto'` = 1 aunque
  la predicción sea lo pesado.
- **H10 (memoria multiseries)**: los trozos de entrenamiento de todos los folds se
  materializan a la vez en el principal (copias por `drop`), también con `n_jobs=1`.

### 6. Posibles errores de correctitud vistos en la lectura

- E1: `n_jobs` negativo distinto de -1 (p. ej. -2) → `cpu_count()`, no
  `cpu_count()+1+n_jobs` como en joblib; `n_jobs=0` se acepta y da `cpu_count()`.
- E2: `ForecasterDirect`/`DirectMultiVariate` resuelven `'auto'` en `__init__`;
  `set_params` no lo recalcula (p. ej. un LGBM que pasa a `n_jobs=1`).
- E3: inconsistencia de `'auto'` entre backtesting suelto y búsqueda con `refit=False`
  (H1). No es un error de resultados, sí de la promesa "'auto' no es peor que 1".
- E4: `TransformedTargetRegressor(Ridge)` o `(LGBM)` no se desenvuelve: se paraleliza
  como si fuera un modelo caro / sin hilos.
- A comprobar: que las predicciones y métricas en paralelo sean idénticas a `n_jobs=1`.

---

## Fase 2: medidas

Herramientas: `cases.py` (casos), `run_one.py` (ejecuta un caso; registra el `n_jobs`
efectivo de cada `Parallel` parcheando `joblib.Parallel` en los módulos de skforecast;
huella md5 de los números del resultado), `bench.py` (orquestador: frío = un proceso
nuevo por medida; caliente = un proceso que calienta cada config y luego las mide
intercaladas; `--cpus k` = `taskset`). Resultados en `results/<nombre>.csv|md`.
Todas las tablas: mediana y mínimo de 5 repeticiones intercaladas al azar.

### E0. Coste de arranque de los workers (`probe_startup.py`)

Hipótesis: el arranque en frío cuesta segundos en Linux; en caliente, milisegundos.
Experimento: `Parallel` con tareas que importan `skforecast.model_selection._validation`
y `sklearn.ensemble`, proceso nuevo por medida, 1/2/4 CPU con `taskset`.

| CPUs | n_jobs | import skforecast en el principal | frío (med/min) | caliente (med/min) | caliente, 100 tareas triviales |
|---|---|---|---|---|---|
| 4 | 1 | 1.14 s | 0.03 / 0.03 s | 0.1 ms | 0.2 ms |
| 4 | 3 | 1.17 s | 1.30 / 1.23 s | 11 ms | 11 ms |
| 4 | 4 | 1.12 s | 1.31 / 1.27 s | 11 ms | 11 ms |
| 2 | 2 | 1.14 s | 1.31 / 1.25 s | 11 ms | 11 ms |

Conclusión: en este Linux el arranque en frío de los workers cuesta ~1.3 s (lo mismo
con 2, 3 o 4 workers, importan en paralelo) y en caliente ~11 ms por llamada a
`Parallel`. Cualquier paralelización en frío tiene que ahorrar más de ~1.3 s.

### H1. Búsqueda con `refit=False`

Caso: `grid_search_forecaster`, `ForecasterRecursive(lags=24)`, n=2000 horario,
`steps=24`, 20 folds, `refit=False`, 4 candidatos, 4 CPU.

| Estimador (rejilla) | modo | n_jobs=1 med (min) | 'auto' (=3) med (min) | 'auto' / 1 |
|---|---|---|---|---|
| RF (`n_estimators` 20/50 × `max_depth` 5/None) | frío | 1.54 (1.37) | 3.34 (3.27) | 2.2× más lento |
| RF | caliente | 1.52 (1.42) | 1.84 (1.78) | 1.2× más lento |
| HGB (`max_depth` 3/6 × `learning_rate`) | frío | 2.59 (2.21) | 2.82 (2.71) | 1.1× más lento |
| HGB | caliente | 2.38 (2.27) | 1.49 (1.40) | **1.6× más rápido** |
| XGB (`max_depth` × `learning_rate`) | frío | 0.73 (0.68) | 2.75 (2.62) | 3.8× más lento |
| CatBoost (`depth` 4/6 × `iterations` 100/200) | frío | 1.28 (1.16) | 2.50 (2.35) | 2.0× más lento |

Predicciones y métricas idénticas en todas las configuraciones.

Conclusión: H1 confirmada salvo HGB en caliente. En la búsqueda, `'auto'` con
`refit=False` paraleliza folds que solo predicen y pierde: arranque en frío + enviar el
forecaster ajustado a cada tarea. HGB es la excepción porque predecir una fila con HGB
es caro (ver siguiente punto) y en los workers corre con 1 hilo.

### Lateral: hilos OpenMP en la predicción fila a fila

Micro-medida (1500×24, mediana de 5 × 300 llamadas):

| Estimador | 1 fila, hilos por defecto (4) | 1 fila, 1 hilo |
|---|---|---|
| `HistGradientBoostingRegressor.predict` | 1.02 ms | 0.54 ms (`threadpool_limits(1)`) |
| `XGBRegressor` `inplace_predict` | 0.093 ms | 0.107-0.117 ms |
| `CatBoostRegressor.predict` | 0.223 ms | 0.221-0.225 ms |

Conclusión: solo HGB sufre el coste de abrir la región OpenMP con 4 hilos en cada fila:
la predicción recursiva con HGB es ~2× más lenta de lo necesario también con
`n_jobs=1`. skforecast ya limita los hilos de LightGBM al predecir; HGB no.
Siguiente: medir el efecto en un backtesting real limitando los hilos solo al predecir.

Lateral confirmado en un backtesting real (`probe_hgb_predict.py`, HGB, n_jobs=1,
caliente): limitar a 1 hilo OpenMP solo la predicción recursiva baja 0.666 → 0.610 s
con `refit=False` (−8 %) y 5.78 → 5.56 s con `refit=True` (−4 %). Micro-medida repetida:
0.95 → 0.66 ms por fila (−30 %). Efecto real pequeño: **no prioritario**.

### H2. Forecaster ajustado enviado a cada tarea con `refit=True`

Medida directa (forecaster ajustado con 1520 filas, 24 lags):

| estimador | pickle ajustado | dumps+loads | pickle sin ajustar | fit |
|---|---|---|---|---|
| Ridge | 0.01 MB | 0.3 ms | 4 KB | 0.01 s |
| RF (100) | 13.6 MB | 15.5 ms | 5 KB | 1.33 s |
| ET (100) | 21.6 MB | 36.9 ms | 5 KB | 0.64 s |
| HGB | 0.41 MB | 5.8 ms | 5 KB | 0.27 s |
| XGB | 0.37 MB | 2.9 ms | 5 KB | 0.16 s |
| CatBoost | 1.13 MB | 1.8 ms | 4 KB | 1.48 s |
| LGBM | 0.29 MB | 2.3 ms | 5 KB | 0.08 s |

Conclusión: con `refit=True` el sobrecoste por tarea (≤ 40 ms + transferencia) es
≤ 5-6 % del ajuste del fold incluso con RF/ET. H2 es real pero de impacto bajo en el
backtesting con refit. Es el coste dominante cuando el fold solo predice (H1).
Enviar un clon sin ajustar a los folds con refit es barato de implementar, pero
la ganancia esperada es pequeña. No lo mido más.

### H4. Umbral de trabajo de un backtesting suelto (`refit=True`)

Caso: `backtesting_forecaster`, RF con pocos árboles para escalar el trabajo, n=2000,
20 folds, `steps=24`, 4 CPU. `est. work` = `n_fits * fit_seconds` registrado por
`limit_n_jobs_by_workload`.

| árboles | modo | n_jobs=1 | n_jobs=3 | 'auto' (elige) | est. work |
|---|---|---|---|---|---|
| 4 | frío | 1.27 (1.06) | 2.02 (1.98) | 1.28 (1) | — |
| 8 | frío | 2.40 (2.38) | 2.42 (2.33) | 2.40 (1 o 3) | 1.94 |
| 12 | frío | 3.61 (3.40) | 2.96 (2.85) | 2.90 (3) | — |
| 1 | caliente | 0.31 (0.26) | 0.17 (0.15) | 0.36 (1) | 0.32 |
| 4 | caliente | 1.21 (1.13) | 0.54 (0.48) | 1.17 (1) | 0.96 |

Conclusión: en frío el umbral de 2 s está bien calibrado en este Linux: el equilibrio
está en ~2.4 s de trabajo secuencial real, y la carga estimada es ~80 % del tiempo
secuencial (1.94 / 2.40). En caliente paralelizar gana a cualquier tamaño (1.8-2.2×),
así que el umbral deja sin aprovechar hasta ~1.5 s por llamada en un notebook. Es una
pérdida acotada; el umbral cumple el criterio "'auto' nunca mucho peor que 1".

### H3. `cpu_count() - 1` frente a `cpu_count()`

Caso: el de H4 con RF de 25 árboles (≈ 7.3 s secuenciales), frío.

| CPUs | n_jobs=1 | 'auto' | n_jobs=cpu_count |
|---|---|---|---|
| 4 | 7.35 (6.99) | 4.44 (4.31), elige 3 | 3.90 (3.75) con 4 |
| 2 | — | 7.44 (7.17), elige 1 | 5.78 (5.48) con 2 |

Las 4 vCPU de este contenedor se presentan como 4 núcleos sin hyperthreading
(`lscpu`), aunque el hipervisor puede mapearlas a hilos hermanos.

Conclusión: H3 confirmada. Con 4 CPU, 4 workers son un 12 % más rápidos que 3 (el
proceso principal apenas trabaja durante el `Parallel`). Con 2 CPU `'auto'` nunca
paraleliza y deja un 1.29× sin aprovechar. En una máquina con 2 vCPU que sean un
núcleo físico con HT (Colab gratis) la ganancia sería menor: no medible aquí.

### H5. Modelos lineales (regla estática "lineal → 1")

Caso: `backtesting_forecaster`, `refit=True`, `steps=24`, frío, 4 CPU.

| Estimador, datos | folds | n_jobs=1 | n_jobs=3 | 'auto' (=1) | est. work | pico memoria 1 / 3 |
|---|---|---|---|---|---|---|
| Ridge, n=20 000, 48 lags | 50 | 0.62 | 1.82 | 0.55 | 1.06 | — |
| Pipeline(Scaler, Ridge), n=50 000, 168 lags, 5 exog | 10 | 2.33 (2.21) | 2.75 (2.64) | 2.27 | 2.23 | — |
| ídem | 20 | 4.80 (4.30) | 4.08 (3.99) | — | — | — |
| ídem (1 rep) | 100 | 22.7 | 10.7 | 22.1 | 26.1 | 407 MB / 1486 MB |

Las predicciones en paralelo difieren de las secuenciales en ~1e-11 (Pipeline+Ridge,
Ridge) y ~1e-13 (LinearRegression): ruido de coma flotante, no un error. La huella pasa
a redondear a 1e-7.

Conclusión: la regla estática acierta en los tamaños típicos (Ridge con 20 000 filas
y 50 folds tarda 0.6 s), pero deja 2.1× sin aprovechar en backtests lineales grandes
(22 s → 10.7 s). En lineales el equilibrio está más alto que en RF (~3-3.5 s
secuenciales reales frente a ~2.4 s): 3 workers solo rinden ~1.9× (memoria + BLAS de 1
hilo en los workers frente a 4 en el principal). Y para ajustes de milisegundos la carga
estimada sobreestima (1.06 s estimados frente a 0.55 s reales: el ajuste inicial lleva
costes de primera llamada). Sustituir la regla por el umbral actual perdería ~0.4 s
(18 %) cerca del umbral; con un umbral doble para lineales no.

### H6. Paralelismo anidado (hilos de los estimadores dentro de los workers)

Hilos efectivos (`probe_threads.py` y medida de tiempo de ajuste con 20 000×30, 200
árboles; un worker de un `Parallel(n_jobs=3)` en 4 CPU):

| Estimador (parámetro por defecto) | principal | worker de 3 | ¿respeta el límite de loky? |
|---|---|---|---|
| HGB (OpenMP) | 4 | 1 | sí |
| XGB (`n_jobs=None`) | 4 (0.74 s) | 1 (1.58 s ≈ 1.53 s con `n_jobs=1`) | sí |
| CatBoost (`thread_count=-1`) | 4 (0.66 s) | **4** (0.70 s; 1.50 s con 1 hilo) | **no** |
| LightGBM (`n_jobs=None` o -1) | 4 | **4** (`num_threads` del booster) | **no** |
| `n_jobs=-1` anidado (RF) | 4 | **4** (joblib threading anidado) | **no** |

Efecto en un backtesting `refit=True`, n=2000, 24 lags, 20 folds, frío, 4 CPU:

| Estimador | n_jobs=1 | 'auto' | n_jobs=3 | est. work |
|---|---|---|---|---|
| HGB | 6.02 (5.59) | **3.42 (3.29)**, elige 3 | — | 5.70 |
| XGB | **3.13 (3.08)** | 3.94 (3.65), elige 3 | — | 2.90 |
| CatBoost (300 it.) | 9.31 (8.79) | **8.23 (7.81)**, elige 3 | — | 9.44 |
| LGBM, 4 folds (1 rep) | 0.55 | — | **52.8** | — |
| LGBM, 20 folds | — | — | **> 166 s (timeout)** | — |
| `TransformedTargetRegressor(LGBM)`, 20 folds (1 rep) | 2.10 | 1.92, elige 1 (umbral) | — | 1.59 |
| `TransformedTargetRegressor(LGBM 400 árboles)`, 10 folds | 3.53 (3.45) | **> 155 s (timeout)**, elige 3 | — | — |

Conclusiones:
- **LightGBM con sus hilos por defecto dentro de varios workers es catastrófico en este
  Linux (~100× más lento)**: 3 procesos × 4 hilos OpenMP en 4 CPU. `'auto'` lo evita
  si ve el `LGBMRegressor` (directo o último paso de un `Pipeline`), pero **no** si va
  envuelto en un `TransformedTargetRegressor` (u otro meta-estimador): ahí `'auto'`
  supera el umbral y el backtesting pasa de 3.5 s a más de 155 s. Un usuario que pase
  `n_jobs=-1` con un `LGBMRegressor()` por defecto cae en lo mismo sin aviso.
- XGBoost respeta el límite de loky, pero con 1 hilo por worker los 3 procesos solo
  sustituyen a los 4 hilos del secuencial: `'auto'` pierde un 26 % con datos pequeños.
- HGB gana mucho con procesos (1.76×): su paralelismo de hilos es poco eficiente con
  datos pequeños y predecir fila a fila con 4 hilos es lento.
- CatBoost ignora el límite (sobresuscripción 3×4) y aun así gana un 12 %: su
  OpenMP no se degrada como el de LightGBM.
- Siguiente: repetir HGB/XGB/CatBoost con datos medianos (los hilos rinden más).

**Nota de método**: a las 13:42 el usuario fusionó `0.27.x` en la rama (commit 0604fa4:
avisos de los workers re-emitidos en el principal con `_run_recording_worker_warnings`,
matriz C-contigua para lineales). Las medidas anteriores usan el código de 16a519a. Se
repitieron H1-RF (frío: 1.70 s con 1 frente a 3.47 s con 'auto') y H5 (10 folds: 2.38
frente a 2.81; 20 folds: 4.57 frente a 3.64): mismas conclusiones. Todo lo que sigue usa
el código fusionado.

### H6 (cont.). Boosting con datos medianos (n=10 000, 48 lags, 10 folds, refit, frío)

| Estimador | n_jobs=1 | 'auto' (=3) | est. work |
|---|---|---|---|
| XGB | **3.45 (3.19)** | 4.31 (4.04) | 3.09 |
| HGB | 4.56 (4.39) | **3.76 (3.43)** | 4.14 |
| CatBoost (300 it.) | 11.00 (10.75) | **9.97 (9.70)** | 10.04 |

XGB pierde también aquí (−25 %); HGB gana menos que con datos pequeños (1.21× frente a
1.76×); CatBoost gana poco (1.10×) pese a la sobresuscripción.

### Hallazgo: un backtesting paralelo con XGBoost ralentiza 2.5× los ajustes posteriores

Observación (caliente, n=2000, 20 folds, refit): `n_jobs=1` tardaba 8.35 s en un proceso
que ya había ejecutado un backtesting paralelo, frente a 3.13 s en frío. Reproducido en
un proceso: `[1: 3.05, 1: 3.16, 3: 3.75, 1: 7.99, 1: 7.82, 3: 2.78, 1: 7.43]`.

Diagnóstico (`probe_xgb_after_parallel.py` y pruebas en línea):
- Solo lo dispara un backtesting **paralelo de skforecast con XGBoost**: un `Parallel`
  trivial, ajustes de XGB en workers o un backtesting paralelo con Ridge no.
- Los workers ociosos no consumen CPU; no hay avisos re-emitidos; los filtros de avisos
  y `threadpool_info()` del principal no cambian (4 hilos OpenMP).
- cProfile: el tiempo extra está todo en `xgboost.core.Booster.update` (nativo): 1.57 s →
  3.45 s. Cada hilo OpenMP consume la misma CPU pero tarda 2.5× más en reloj (duerme en
  las barreras).
- El proceso pasa de 13 a 16 hilos del sistema (con Ridge se queda en 13): aparece un
  segundo equipo de libgomp.
- Reproducción mínima sin skforecast: `pickle.dumps(xgb_model)` desde un hilo **que sigue
  vivo** → 11 → 15 hilos y ajustes de XGB en el principal 162 → 417 ms; al terminar ese
  hilo, vuelven a 204 ms. Igual con `inplace_predict` desde ese hilo. Con un hilo que
  termina no pasa (libgomp destruye su equipo).

Mecanismo: el `QueueFeederThread` de loky serializa las tareas en el proceso principal.
Al serializar el forecaster ajustado, XGBoost (`save_raw`) abre una región OpenMP en ese
hilo y libgomp le crea un equipo propio de 4 hilos. Con 8 hilos gestionados para 4 CPU,
libgomp pasa a la espera reducida (throttled spin) y los hilos del equipo principal se
duermen en cada barrera. El hilo vive lo que el ejecutor de loky (hasta 300 s
inactivo), así que **cada ajuste secuencial de XGB en ese proceso** (otro backtesting
con `n_jobs=1`, el ajuste inicial de cada candidato de una búsqueda, el reajuste final
con `return_best=True`) va ~2.5× más lento. Las medidas en frío no lo ven. Corrige la
comparación en caliente de XGB: 'auto' caliente 2.77 s frente a `n_jobs=1` limpio
3.13 s, solo un 12 % de ganancia.

Conclusiones:
- Con XGBoost, paralelizar folds no compensa: −25 % en frío (datos pequeños y medianos),
  +12 % en caliente, y además contamina el proceso. Tratarlo como LightGBM
  (`n_jobs != 1` → 1) es la decisión segura.
- Enviar a los folds con refit un forecaster **sin ajustar** (H2) evitaría serializar el
  booster en el hilo de loky: H2 gana una justificación más allá de la IPC.
