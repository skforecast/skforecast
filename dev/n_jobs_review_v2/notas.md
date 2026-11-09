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
| Multiseries recursivo, Recursive, Stats, EquivalentDate, Rnn | n/m | Sin paralelismo propio en `fit`/`predict` (ni en el bootstrapping de intervalos). |

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
| Candidatos de una búsqueda | n/m | **No** se paralelizan: bucle secuencial, y cada candidato paraleliza sus folds. Los workers de loky se reutilizan entre candidatos (mismo `n_jobs`). |
| Series | n/m | No se paralelizan. |
| Bootstrapping de intervalos | n/m | No se paraleliza (vectorizado sobre `n_boot`). |

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
| 4 | frío | 1.27 (1.06) | 2.02 (1.98) | 1.28 (1) | n/m |
| 8 | frío | 2.40 (2.38) | 2.42 (2.33) | 2.40 (1 o 3) | 1.94 |
| 12 | frío | 3.61 (3.40) | 2.96 (2.85) | 2.90 (3) | n/m |
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
| 2 | n/m | 7.44 (7.17), elige 1 | 5.78 (5.48) con 2 |

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
| Ridge, n=20 000, 48 lags | 50 | 0.62 | 1.82 | 0.55 | 1.06 | n/m |
| Pipeline(Scaler, Ridge), n=50 000, 168 lags, 5 exog | 10 | 2.33 (2.21) | 2.75 (2.64) | 2.27 | 2.23 | n/m |
| ídem | 20 | 4.80 (4.30) | 4.08 (3.99) | n/m | n/m | n/m |
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
| HGB | 6.02 (5.59) | **3.42 (3.29)**, elige 3 | n/m | 5.70 |
| XGB | **3.13 (3.08)** | 3.94 (3.65), elige 3 | n/m | 2.90 |
| CatBoost (300 it.) | 9.31 (8.79) | **8.23 (7.81)**, elige 3 | n/m | 9.44 |
| LGBM, 4 folds (1 rep) | 0.55 | n/m | **52.8** | n/m |
| LGBM, 20 folds | n/m | n/m | **> 166 s (timeout)** | n/m |
| `TransformedTargetRegressor(LGBM)`, 20 folds (1 rep) | 2.10 | 1.92, elige 1 (umbral) | n/m | 1.59 |
| `TransformedTargetRegressor(LGBM 400 árboles)`, 10 folds | 3.53 (3.45) | **> 155 s (timeout)**, elige 3 | n/m | n/m |

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

### H7. ForecasterDirect (ajuste de los pasos, `'auto'` sin umbral)

Caso: `ForecasterDirect(...).fit` + `predict`, 24 lags, frío, 4 CPU.

| Estimador, datos, pasos | n_jobs=1 | 'auto' (=3) |
|---|---|---|
| RF 10 árboles, n=500, 6 pasos, 12 lags | **0.19 (0.15)** | 1.51 (1.49) |
| RF 10 árboles, n=2000, 12 pasos | 2.25 (1.98) | 2.23 (2.14) |
| XGB, n=2000, 12 pasos | **1.99 (1.78)** | 2.99 (2.87) |
| HGB, n=2000, 12 pasos | 3.35 (3.17) | **2.50 (2.32)** |

Conclusión: H7 confirmada. Sin umbral, un Direct barato paga el arranque completo
(+1.3 s, 8× en el peor caso medido); el equilibrio está otra vez en ~2.2-2.4 s de
trabajo; XGB pierde un 50 % (hilos frente a procesos, como en el backtesting).

### H8. Estadísticos (`'auto'` = 1 siempre)

`backtesting_stats`, `refit=True`, frío, 4 CPU. Ojo: la **primera** ejecución tras
instalar compila numba (caché en disco) y la infla: el piloto dio Ets(m=24) 8.47 s
con 1 frente a 4.55 s con 3, que con la caché caliente es al revés.

| Modelo, datos | 'auto' (=1) | n_jobs=3 |
|---|---|---|
| Ets(m=1), n=500, 10 folds de 12 | **0.60 (0.52)** | 2.49 (2.43) |
| Ets(m=24) selección automática, n=1500, 10 folds de 24 | **3.28 (2.75)** | 4.37 (4.18) |
| Arima(2,1,1)(1,1,1)24, n=1500, 10 folds (3 reps) | 18.55 (18.55) | **11.92 (10.60)** |

Primer ajuste con selección frente a reajuste con parámetros congelados
(`freeze_params=True`, el defecto): auto-Arima 2.21 s frente a 0.031 s; auto-Ets
1.79 s frente a 0.080 s.

Comprobación de `refit` en stats (`_validation.py:_backtesting_stats`, l. 2263-2277):
todo estimador que no sea `skforecast.stats.Sarimax` se fuerza a `refit=True`
(`IgnoredArgumentWarning`); `Sarimax` admite `refit=False` y entero (y entonces
`n_jobs=1`). La suposición "refit siempre True" es cierta salvo con `Sarimax`.

Conclusión: `'auto'` = 1 acierta en los casos pequeños y medianos (paralelizar pierde
hasta 4×: arranque + carga de numba en cada worker), y deja 1.56× en modelos
estacionales de orden fijo con series largas (o con `freeze_params=False`). Un umbral
basado en el ajuste inicial **no** sirve con `freeze_params=True`: el ajuste inicial
incluye la selección y sobreestima los reajustes 25-70×.

### H9. `refit=False` con intervalos por bootstrapping

RF 100, n=2000, 20 folds, `interval=0.8`, `n_boot=250`, frío: 'auto' (=1) **2.34
(2.20)** frente a 3.56 (3.37) con 3. H9 descartada: el bootstrapping está vectorizado y
`'auto'` = 1 acierta.

### H10. Memoria de los trozos de los folds en multiseries

Ridge, 50 series × 10 000, `refit=True`, `n_jobs=1`: pico 402 MB con 10 folds y 426 MB
con 80. Los trozos son vistas; H10 descartada. El paralelismo sí multiplica la memoria:
mismo caso con 80 folds, 427 MB con 1 frente a 1473 MB con 3 (3.4×), y 12.09 (11.09) s
frente a 8.65 (8.44) s (1.40×).

### Búsquedas con `refit=True` (folds en paralelo, workers reutilizados)

| Rejilla (4 candidatos, n=2000, 10 folds) | n_jobs=1 | 'auto' (=3) |
|---|---|---|
| RF barato (`n_estimators` 5/10 × `max_depth` 5/None) | 3.79 (3.66) | **3.25 (3.11)** |
| XGB (`max_depth` × `learning_rate`) | **4.66 (4.49)** | 5.87 (5.77) |

Con `refit=True` la regla actual funciona (el arranque se amortiza entre candidatos),
salvo con XGB (−26 %).

### H11. Prototipo: candidatos en paralelo en lugar de folds (`refit=False`)

`cases.py:grid_candidates`: cada candidato es un `backtesting_forecaster(n_jobs=1)` y
los candidatos van en un `Parallel`. Mismas métricas que la búsqueda (huella igual).

| Rejilla RF, `refit=False`, 20 folds | búsqueda n_jobs=1 | búsqueda 'auto' (folds) | candidatos ×3 | candidatos ×4 |
|---|---|---|---|---|
| 4 candidatos (ejecuciones separadas) | 4.06 (3.99) | 6.41 (6.28) | 3.40 (3.09) | 2.83 (2.63) |
| 12 candidatos (misma ejecución; 1 = candidatos en serie) | 8.43 (8.11) | n/m | 4.87 (4.67) | 4.28 (4.00) |

Conclusión: con `refit=False`, paralelizar candidatos da 1.7-2.0× frente a `n_jobs=1`
(con arranque en frío incluido) y ~3× frente al `'auto'` actual. Es un cambio de diseño
(avisos, `output_file`, barra de progreso, bayesiana secuencial, límites de hilos para
LightGBM/CatBoost dentro de los workers).

### H2 (cont.). Prototipo "forecaster sin ajustar en los folds con refit"

`probe_unfitted_refit.py` (XGB, 20 folds, refit): ambas variantes dejan el proceso
contaminado (`n_jobs=1` posterior: 7.98 s actual, 8.45 s variante, frente a 3.4-3.6 s
en limpio). El fold 0 no reajusta (se ajusta en el principal) y sigue enviando el
forecaster ajustado. Para evitar la contaminación habría que predecir el fold 0 en el
principal además de enviar copias sin ajustar. La parte RF del prototipo superó el
tiempo límite (100 árboles × 13 ejecuciones, demasiado para un experimento corto): no
se repite porque el ahorro de IPC con refit ya está acotado (≤ 5 %).

---

## Conclusiones de la fase 2 (escritas antes de leer la revisión previa)

Modelo explícito para otras máquinas (estimación, no medida): con `W` el trabajo
secuencial, `S` el arranque en frío (aquí 1.3 s en Linux) y `e` la eficiencia por
worker, `T_par ≈ S + W / (n·e)`. Ajuste aquí: RF `e ≈ 0.78` (7.35 s → 4.44 s con 3),
lineal grande `e ≈ 0.6-0.7`. Equilibrio `W* = S / (1 − 1/(n·e))`: con 3 workers
`W* ≈ 2.3 s` (medido 2.4 s); con 15 workers y el mismo `S`, `W* ≈ 1.4 s`. Con un `S` de
Windows de varios segundos, `W*` sube en proporción. En caliente `S ≈ 0.01 s`.

Veredicto: la estrategia es razonable en el backtesting suelto (el umbral de trabajo
está bien calibrado aquí y protege de lo peor) pero tiene cuatro huecos que hacen a
`'auto'` mucho peor que `n_jobs=1` en casos frecuentes:

1. **Búsquedas con `refit=False`**: `'auto'` paraleliza folds que solo predicen
   (2-3.8× más lento en frío con RF/XGB/CatBoost; HGB es la excepción en caliente).
2. **Estimadores con hilos propios no detectados**: LightGBM dentro de un
   `TransformedTargetRegressor` (≥ 44× más lento: >155 s frente a 3.5 s); XGBoost
   (−25 % a −50 % en backtesting, búsqueda y Direct, y además ralentiza 2.5× los ajustes
   de XGB posteriores del proceso por un segundo equipo de libgomp).
3. **Direct sin umbral**: un Direct barato paga +1.3 s (8×).
4. **`cpu_count() - 1`**: en 2 CPU `'auto'` nunca paraleliza (deja 1.29×); en 4 CPU
   deja un 12 %.

Propuestas, por impacto/riesgo:

- **P1** (alto, riesgo bajo): `select_n_jobs_backtesting` devuelve 1 si `refit` es
  `False` (así lo hereda la búsqueda; el backtesting suelto ya lo hacía con el umbral).
- **P2** (alto, riesgo bajo): ampliar la detección de hilos internos: desenvolver
  `TransformedTargetRegressor` (y meta-estimadores con `regressor`/`estimator`), y tratar
  `XGBRegressor`/`XGBClassifier` con `n_jobs != 1` como LightGBM, en el backtesting y en
  `select_n_jobs_fit_forecaster`. Guardarraíl: aviso cuando el usuario pide `n_jobs > 1`
  explícito con un LightGBM de hilos por defecto. CatBoost y HGB se quedan en paralelo
  (ganan 1.1× y 1.2-1.8×).
- **P3** (medio, riesgo bajo-medio): `cpu_count()` en lugar de `cpu_count() - 1` cuando
  el umbral ya ha decidido paralelizar.
- **P4** (medio, coste medio): umbral de trabajo para Direct (ajustar y cronometrar el
  primer paso en el principal) y para stats solo cuando el ajuste inicial es
  representativo (`freeze_params=False` o sin selección automática).
- **P5** (medio-bajo): lineales: umbral propio (≈ 2× el general) en lugar de "siempre
  1"; gana 1.3-2.1× en backtests lineales grandes a cambio de ~3.4× de memoria.
- **P6** (alto, coste alto): búsquedas con candidatos en paralelo (1.7-2.0× frente a
  `n_jobs=1`, ~3× frente al `'auto'` actual con `refit=False`). Proyecto aparte.
- Descartadas o menores: umbral consciente de workers calientes (≤ 1.5 s por llamada,
  depende de estado privado de loky), limitar hilos de HGB al predecir (−4 a −8 %),
  enviar el forecaster sin ajustar (≤ 5 %; solo útil contra la contaminación de XGB si
  además el fold 0 se predice en el principal), intervalos sin refit (`'auto'` acierta),
  memoria de multiseries (no crece con los folds).

Correctitud: todas las configuraciones dieron predicciones y métricas iguales a
`n_jobs=1` salvo ruido de coma flotante (≤ 4e-11 en lineales con `refit=True`). Errores
de comportamiento: E4 (meta-estimadores no desenvueltos, con la patología de LightGBM),
E1 (`n_jobs=-2` → todos los núcleos), E2 (Direct resuelve `'auto'` en `__init__`).

---

## Fase 3: contraste con la revisión previa (`dev/n_jobs_review/`)

Leídos: `informe_n_jobs.md` (entero), `results/threshold.md`,
`results/strategies_summary.md`, `bench.py` (cómo ejecuta las celdas) y el notebook de la
FAQ (celdas de texto).

Medidas cortas hechas aquí para resolver discrepancias (búsqueda `refit=False`, frío
salvo indicación, 4 CPU):

| Caso | n_jobs=1 | 'auto' (=3) |
|---|---|---|
| HGB, n=5000, 48 lags, **168 pasos**, 10 folds, 2 candidatos | 3.69 (3.56) | **2.80 (2.63)** |
| RF (50/100 árboles), mismo caso | **9.79 (9.34)** | 11.29 (10.96) |
| LGBM(`n_jobs=1`), n=2000, 24 pasos, 20 folds, 4 candidatos | **0.57 (0.51)** | 1.99 (1.94) |
| ídem, caliente | **0.49 (0.45)** | 0.73 (0.67) |

Coincidencias:
- Arranque en frío: WSL2 1.34-1.46 s con 2-4 workers frente a 1.3 s aquí; dominan las
  importaciones. En caliente, despreciable en ambas.
- Umbral de Linux de 2 s: su equilibrio con 21 workers en WSL2 era ~2.8 s reales; aquí
  ~2.4 s con 3 workers. La estimación es ~78-81 % del real en las dos revisiones.
- `refit` entero → 1; lineales pequeños → 1; Stats → 1 en frío; intervalos sin refit →
  1 (su medida con LGBM(1) y la mía con RF dan lo mismo).
- LightGBM con hilos propios dentro de los workers: catastrófico en Linux en las dos
  (>600 s allí, >155 s aquí).
- `cpu_count()-1` frente a `-1`: en su máquina de 4 lógicos (2c/4t) `-1` fue un 5-9 %
  mejor en 3 de 4 celdas (lo llamaron ruido); aquí, un 12 % con 4 CPU. Misma dirección.
- Direct necesita un umbral propio (ellos lo infirieron para Direct S con LGBM(1); aquí
  medido: 0.19 s frente a 1.51 s).

Discrepancias y por qué:
1. **Búsquedas sin refit**. Ellos (22 procesadores, en caliente, tamaño L con 168 pasos,
   HGB y LGBM(1)): `'auto'` gana ×4.7 (HGB) y ×1.5 (LGBM(1)) → "mantener". Aquí (4 CPU):
   pierde con RF, ET, XGB, CatBoost y LGBM(1) (1.5-3.8×, también en caliente con RF y
   LGBM(1)); gana con HGB (en caliente, o en frío con 168 pasos). Causa: **máquina y
   caso**. Paralelizar folds que solo predicen compensa cuando predecir es caro (HGB,
   horizontes largos, muchos workers) y no cuando el modelo pesa mucho (RF/ET: 13-21 MB
   por tarea) o predecir es barato (XGB, CatBoost, LGBM(1) con 24 pasos). Ellos no
   midieron RF/ET/XGB/CatBoost en búsquedas sin refit ni horizontes cortos.
2. **XGBoost**. Ellos: `'auto'` gana ×2.5-4.9 (8 y 22 procesadores, caliente, 168 pasos)
   → "sin regla nueva". Aquí, con 4 CPU: pierde un 25 % en frío y gana un 12 % en
   caliente. Causa probable: **número de núcleos** (con 4 hilos XGB ya usa bien la
   máquina; con 22, procesos de 1 hilo rinden más) y el horizonte (168 pasos). Además,
   sus celdas en caliente se ejecutaban en un solo proceso barajadas (`bench.py`); en
   Linux, una celda XGB con `n_jobs=1` posterior a una paralela estaría inflada 2.5× por
   el segundo equipo de libgomp. En Windows XGBoost usa otro runtime de OpenMP: no
   puedo saber si les afecta. Su conclusión para máquinas grandes es plausible; la mía
   para 4 CPU también.
3. **Contaminación de XGBoost tras un paralelo** (segundo equipo de libgomp): no aparece
   en la revisión previa. Nuevo.
4. **`TransformedTargetRegressor(LGBM)` con `'auto'`**: su sección 8.6 cita
   `TransformedTargetRegressor(Ridge)` como mal clasificado, pero no el caso LightGBM, que
   aquí lleva a la patología de sobresuscripción con `'auto'`. Nuevo.
5. **Búsquedas candidatos frente a folds**: no lo evaluaron (proponían decidir una vez por
   búsqueda). Aquí el prototipo da 1.7-2.0× frente a `n_jobs=1` con `refit=False`.
6. **Stats con `freeze_params`**: no lo consideraron; aquí el ajuste inicial con selección
   es 25-70× más caro que los reajustes, lo que invalida estimar con él.
7. **HGB con muchos hilos**: ellos midieron que HGB secuencial es muy lento con 22 hilos
   (Direct S: 5.5 s frente a 0.6 s en workers; HGB L sin refit 2.94 s con 1 frente a
   0.52 s con 21). Aquí, con 4 hilos, limitar a 1 hilo la predicción solo ahorra un
   4-8 %. En máquinas grandes, el equivalente de D9 (LightGBM) para HGB puede valer mucho
   más: **falta medirlo** allí.

Revisión crítica de sus scripts y conclusiones:
- `strategies.py` evalúa E3 con el tiempo secuencial **medido** como estimación (lo dicen);
  el código real usa `t_fit × n_fits`, que además sobreestima con ajustes de milisegundos
  (Ridge: 1.06 s estimados frente a 0.55 s reales). No afecta hoy porque los lineales
  van a 1 antes del umbral, pero sí a cualquier propuesta que quite la regla lineal.
- Escenario "frío" de V2: mediana en caliente + 72 % del arranque medido. Es una
  aproximación razonable; mis medidas en frío reales coinciden en orden de magnitud.
- Su veredicto "XGBoost, CatBoost, RF/ET con hilos propios: sin regla nueva" se apoya en
  22 procesadores; con 4, XGBoost pierde (arriba) y CatBoost apenas gana (1.1×).

Documentación frente a código (notebook de la FAQ, `llms-base.txt`, skills):
- La tabla de reglas y el umbral del notebook coinciden con el código.
- **Inexacto**: "Without refit ... `'auto'` runs them in one process" y "Do not expect
  gains from n_jobs without refit": cierto en un backtesting suelto, **falso en las
  búsquedas**, donde `'auto'` sí paraleliza folds sin refit.
- **Inexacto**: "Do not expect gains ... with `ForecasterStats`": `'auto'` no lo
  paraleliza, pero un `n_jobs` entero sí paraleliza `backtesting_stats` con refit y gana
  (aquí 1.56× con un Arima estacional; ellos ×2.3-2.5 en caliente).
- **Inexacto**: "XGBoost ... running several of them in each worker makes them compete
  for the same cores": XGBoost respeta el límite de loky (1 hilo por worker); CatBoost,
  LightGBM y `n_jobs=-1` anidado no.
- **Falta**: el aviso de LightGBM habla de `n_jobs=-1` fuera de los workers; no dice que
  `LGBMRegressor()` por defecto con `n_jobs > 1` en skforecast (o dentro de un
  `TransformedTargetRegressor` con `'auto'`) es ~100× más lento en Linux.
- `llms-base.txt` y los skills solo dan firmas (`n_jobs='auto'`): sin contradicciones.

## Revisión de propuestas con el usuario

### P1: descartada

Pregunta del usuario: nunca se ha visto a un usuario envolver un LightGBM; ¿es un escenario
real? Comprobación:
- Issues de GitHub (`lightgbm n_jobs`, `lightgbm slow`, `n_jobs slow`,
  `TransformedTargetRegressor`): ninguna sobre lentitud con LightGBM y `n_jobs`; ninguna
  menciona `TransformedTargetRegressor`. Las que combinan LightGBM y `n_jobs` hablan de los
  hilos del estimador (p. ej. `LGBMRegressor(n_jobs=20)`), no del `n_jobs` de skforecast.
- `docs/` y `skills/`: ningún `TransformedTargetRegressor` (skforecast tiene
  `transformer_y`). El único `n_jobs=-1` junto a LightGBM es el del estimador.
- Para caer en la patología con `LGBMRegressor` suelto hay que cambiar a mano el `'auto'`
  por defecto, que ya lo protege.

Conclusión: la prioricé por gravedad (100×) sin pesar la frecuencia. Descartada; queda
una frase en la FAQ (P4).

### P2: búsquedas multiseries con `refit=False` (H13)

Pregunta: la P2 también cambia el valor por defecto de las búsquedas multiseries, donde
cada fold predice muchas series a la vez y el modelo es más grande. No estaba medido.
`cases.py:grid_ms` no se había ejecutado nunca y fallaba (la búsqueda multiseries
devuelve `mean_absolute_error__weighted_average`, `__average` y `__pooling`): corregido.

`grid_search_forecaster_multiseries`, `ForecasterRecursiveMultiSeries`, 4 CPU, 5 reps,
mismas rejillas que H1 (`results/h13_grid_ms_refit_false.md`):

| Caso | modo | n_jobs=1 med (min) | 'auto' (=3) med (min) | 'auto' / 1 |
|---|---|---|---|---|
| LGBM(`n_jobs=1`), 50 series × 1000, 24 pasos, 20 folds, 4 cand. | frío | **3.27 (3.16)** | 4.97 (4.96) | 1.52× más lento |
| LGBM(`n_jobs=1`), ídem | caliente | **3.29 (3.08)** | 3.78 (3.66) | 1.15× más lento |
| HGB, ídem | frío | 4.83 (4.67) | 5.02 (4.81) | empate (+4 %) |
| HGB, ídem | caliente | 4.68 (4.35) | **3.68 (3.57)** | 1.27× más rápido |
| RF, 20 series × 1000, ídem | frío | **13.05 (12.88)** | 17.34 (16.90) | 1.33× más lento |
| HGB, 50 series × 3000, 168 pasos, 10 folds, 2 cand. | frío | 5.63 (5.58) | **4.58 (4.30)** | 1.23× más rápido |
| HGB, ídem | caliente | 5.30 (5.01) | **2.93 (2.90)** | 1.81× más rápido |

Predicciones y métricas idénticas en todas las configuraciones. Piloto de 1 rep con RF y
50 series: 36.9 s frente a 43.1 s (por eso RF va con 20 series).

Conclusión: el patrón de una sola serie se mantiene en multiseries. `'auto'` pierde con
RF y LGBM(1) (también en caliente) y gana con HGB salvo en frío con 24 pasos; con 168
pasos HGB gana 1.2× en frío y 1.8× en caliente. Con la P2 ese caso pasaría a 1. No medido:
LGBM(1) y RF con 168 pasos en multiseries; cuánto de la ventaja de HGB se debe a predecir
con 4 hilos en el principal (P5).

### P2 y P5: HGB con 1 hilo solo en la predicción (H14)

Pregunta: ¿cuánto de la ventaja de `'auto'` con HGB en búsquedas con `refit=False` se
debe a que en el principal HGB predice cada fila con 4 hilos OpenMP (en un worker, 1)?
Nueva config `1lim` en `run_one.py`: `n_jobs=1` con `threadpool_limits(1, "openmp")`
alrededor de `_recursive_predict` (Recursive y RecursiveMultiSeries); emula la P5.
Horizontes de 24 y 72 pasos (el usuario indica que la mayoría de casos predicen 24-72).
4 CPU, 5 reps, configs intercaladas (`results/h14_hgb_predict_1thread.md`):

| Búsqueda HGB, `refit=False`, 4 candidatos | modo | n_jobs=1 | `1lim` (P5) | 'auto' hoy |
|---|---|---|---|---|
| 1 serie, n=2000, 24 pasos, 20 folds | caliente | 2.42 (2.02) | 1.93 (1.83) | **1.45 (1.42)** |
| ídem | frío | 2.52 (2.33) | **2.03 (1.82)** | 2.84 (2.59) |
| 50 series × 1000, 24 pasos, 20 folds | caliente | 4.64 (4.46) | 4.06 (4.00) | **3.63 (3.56)** |
| ídem | frío | 4.84 (4.69) | **4.07 (3.97)** | 4.99 (4.92) |
| 50 series × 2000, 72 pasos, 10 folds | caliente | 5.76 (5.56) | 4.63 (4.15) | **3.92 (3.83)** |
| ídem | frío | 5.50 (5.28) | **4.74 (4.53)** | 5.23 (5.10) |

Predicciones idénticas en las tres configuraciones de cada caso.

Conclusiones:
- La P5 sola vale más de lo estimado: −16 a −20 % con `n_jobs=1` en búsquedas HGB (la
  estimación anterior, −4 a −8 %, era de un backtesting suelto con más peso del ajuste).
  Beneficia también al backtesting suelto con `refit=False`, donde `'auto'` ya es 1.
- La P5 recupera la mitad o algo más de la ventaja de `'auto'` en caliente (50-61 %).
- P2 + P5 frente a hoy, HGB: en frío, más rápido en los tres casos (10-29 %); en
  caliente, más lento (1.12-1.33×). El peor caso de la P2 baja de 1.27-1.67× (24-72
  pasos, caliente) a 1.12-1.33×.
- En máquinas grandes el efecto de la P5 debería ser mayor (la revisión previa medía el
  secuencial de HGB hasta 9× más lento con 22 hilos), sin medir aquí.

### P5: filas por llamada y coste del límite de hilos (micro-medida)

`HistGradientBoostingRegressor` ajustado con 20 000 × 24 (100 iteraciones), 4 CPU,
mediana de 5 bloques. `predict` con los hilos por defecto (4) frente a
`threadpool_limits(1, "openmp")`:

| filas | 4 hilos ms | 1 hilo ms | 1 / 4 |
|---|---|---|---|
| 1 | 0.523 | 0.335 | 0.64 |
| 10 | 0.521 | 0.257 | 0.49 |
| 50 | 0.526 | 0.408 | 0.78 |
| 100 | 0.721 | 0.569 | 0.79 |
| 250 | 0.714 | 0.849 | 1.19 |
| 500 | 1.155 | 1.293 | 1.12 |
| 1000 | 1.521 | 2.197 | 1.44 |
| 5000 | 4.796 | 9.861 | 2.06 |
| 20000 | 14.629 | 40.264 | 2.75 |

Coste de limitar: `threadpool_limits(...)` (entrar y salir) 1.07 ms, porque vuelve a
inspeccionar las librerías cargadas; `ThreadpoolController()` 0.92 ms (una vez);
`controller.limit(...)` con el controlador ya creado, 0.011 ms.

Conclusión: 1 hilo gana hasta ~100 filas por llamada y pierde desde ~250 (cruce entre
100 y 250 en esta máquina). Con un controlador creado una vez por proceso, limitar en
cada llamada cuesta un 3 % de una predicción de una fila: se puede decidir por llamada
según el número de filas (bootstrapping con `n_boot` filas y multiseries con muchas
series quedan como hoy). `threadpool_limits` en cada llamada costaría 3× la predicción.

### P2 y P5 implementadas en el paquete (sin commit): comprobación (H15)

Decisiones del usuario: P2 aprobada; P5 con `threadpoolctl` importado sin declararlo,
umbral de 100 filas y `ForecasterRecursiveClassifier` fuera. Todo en esta rama y en un
único commit final.

Mismo caso que H1/H14 (HGB, 1 serie, 24 pasos, 20 folds, 4 candidatos, `refit=False`),
con el paquete modificado; `3` = folds en paralelo, lo que hacía `'auto'` antes de la P2
(`results/h15_after_p2_p5.md`):

| modo | n_jobs=1 | `'auto'` (ahora 1) | 3 |
|---|---|---|---|
| caliente | 2.00 (1.65) | 1.85 (1.80) | 1.49 (1.42) |
| frío | 2.14 (1.89) | **1.89 (1.86)** | 2.91 (2.84) |

Antes (H14): `n_jobs=1` 2.42 / 2.52 y `'auto'` 1.45 / 2.84 (caliente / frío). Coincide con
la emulación `1lim` (1.93 / 2.03): en caliente `'auto'` es 1.28× más lento que antes y en
frío 1.50× más rápido. Predicciones idénticas.

### P3: descartada

Propuesta: usar `cpu_count()` workers en lugar de `cpu_count() - 1` cuando `'auto'`
paraleliza (`select_n_jobs_backtesting` y `select_n_jobs_fit_forecaster`). Evidencia
(H3): −12 % con 4 CPU y 1.29× con 2 CPU; neutro con 8 y 22 en la revisión previa.

Al revisarla:
- Con 3 o más CPU solo añade un worker: se paraleliza en los mismos casos que hoy.
- Con 2 CPU cambia la decisión (hoy `'auto'` nunca paraleliza). El backtesting suelto
  tiene el umbral de trabajo, pero Direct y las búsquedas con refit no: un Direct barato
  pagaría el arranque (H7: 0.19 s frente a 1.51 s) y una búsqueda barata, unos 1.3 s una
  vez.
- No medible aquí: 2 vCPU que sean un núcleo físico con HT (Colab gratis), donde la
  ganancia sería menor.

Recomendé aplicarla solo al backtesting y las búsquedas. Decisión del usuario:
descartada; se mantiene `cpu_count() - 1`. Sin cambios en el paquete.

### P6: descartada; P7 y P9: fuera de la revisión

Decisión del usuario: P6 (umbral de trabajo para Direct y Stats) descartada sin medir
Direct con 24-72 pasos. P7 (umbral propio para lineales) y P9 (candidatos en paralelo) no
se revisan. La FAQ (P4) queda para el final. Único punto restante: P8 (XGBoost).

### P8: XGBoost, datos crudos de la revisión previa y contaminación (H16)

**Revisión previa, datos crudos** (`dev/n_jobs_review/results/block_C*.csv`). La máquina
es **Windows 11** (Intel Ultra 9 185H, 16 físicos / 22 lógicos), no Linux: XGBoost usa
allí el OpenMP de MSVC (vcomp), no libgomp. XGB `n_jobs=None` (sus hilos por defecto),
tamaño L, `refit=True`, caliente, mediana de 5:

| Procesadores (afinidad) | `n_jobs=1` | `cpu_count() - 1` | -1 |
|---|---|---|---|
| 4 (2c/4t) | 7.08 | 6.14 (1.15×) | 5.96 |
| 8 (4c/8t) | 10.02 | 3.98 (2.5×) | 5.36 |
| 22 | 7.33 | 2.66 (2.8×) | 2.77 |

Con 4 procesadores paralelizar gana un 15 % allí (caliente) y aquí pierde un 25 % en frío
y gana un 12 % en caliente; con 8 y 22 gana 2.5-2.8×. Una regla "XGBoost → 1" perdería eso
en máquinas medianas y grandes: no compensa.

**¿Qué estimadores contaminan?** `probe_omp_pickle_thread.py`: el modelo ajustado se
serializa (o predice) desde un hilo que sigue vivo, como el `QueueFeederThread` de loky;
después, mediana de 5 ajustes del mismo estimador en el hilo principal (5000 × 30, 4 CPU;
xgboost 3.4.1, lightgbm 4.7.0, scikit-learn 1.9.1, catboost 1.2.10, joblib 1.6.0):

| Estimador | serializar en el hilo | predecir en el hilo | hilos del proceso |
|---|---|---|---|
| XGB (hilos por defecto) | 1.75-1.96× | 2.09× | 11 → 15 |
| XGB `n_jobs=1` | 0.95-1.03× | n/m | 11 → 12 |
| LGBM (hilos por defecto) | 6.01× | 5.91× | 11 → 15 |
| LGBM `n_jobs=1` | 0.94× | 1.05× | 8 → 9 |
| HGB | 1.04× | 7.13× | 11 → 12 / 15 |
| CatBoost | 0.94× | 1.00× | 8 → 9 |

Al terminar el hilo, todo vuelve a la normalidad. Serializar solo crea un segundo equipo
de OpenMP con XGB y LGBM de hilos por defecto. Con `'auto'`, LGBM de hilos por defecto ya
va a 1 job; HGB solo contamina si predice en otro hilo (skforecast no lo hace; loky usa
procesos). Queda XGB con sus hilos por defecto, el valor por defecto de `XGBRegressor`.

**Direct también contamina** (`probe_xgb_direct.py`): tras `ForecasterDirect(XGB, 12
pasos, n_jobs=3).fit`, los ajustes de XGB en el principal pasan de 229 a 472 ms (11 → 16
hilos); con `n_jobs=1`, 234 → 228 ms. Aquí el modelo ajustado vuelve del worker y se
deserializa en un hilo de loky.

**Corrección probada** (`probe_xgb_fix_bytes.py`, `Parallel(n_jobs=3)` de 6 tareas):

| Modo | ajustes de XGB en el principal después | hilos |
|---|---|---|
| enviar el modelo ajustado | 218 → 477 ms | 11 → 16 |
| enviar `cloudpickle.dumps(modelo)` hecho en el principal | 249 → 249 ms | 11 → 13 |
| devolver el modelo ajustado | 247 → 465 ms | 11 → 16 |
| devolver `pickle.dumps(modelo)` y cargarlo en el principal | 235 → 233 ms | 11 → 13 |

Serializar y deserializar en el hilo principal evita el segundo equipo en los dos
sentidos. En skforecast: serializar el forecaster una vez en el principal antes del
`Parallel` del backtesting (single y multiseries), y que el ajuste de los pasos de Direct
devuelva los estimadores serializados. De paso, el forecaster se serializa una vez en
lugar de una por tarea. Loky usa `cloudpickle` si está instalado (joblib 1.6 ya no lo
incluye en `joblib.externals`; joblib 1.3, el mínimo, sí).

Decisión del usuario: P8 aplazada; no se implementa nada por ahora.

### Revisión final de P2 + P5 antes del commit

Diff revisado completo: `_utils.py`, `_search.py`, `utils.py`, los tres tests, la FAQ y
la nota de versión. Comprobaciones:

- `controller.limit(limits=1, user_api="openmp")` con el controlador cacheado: 12-14 µs
  por entrada y salida (3 librerías: 2 OpenBLAS y libgomp). threadpoolctl 3.8 restaura a
  la salida todas las librerías, también OpenBLAS, al valor que tenían (dos llamadas
  ctypes más, sin efecto). Coincide con el "0.01 ms" del docstring.
- HGB 5000 × 24, 4 CPU con carga de fondo, `predict_fn` frente a `estimator.predict`:
  1 fila 0.80, 0.67 y 0.63× con 100, 500 y 1000 iteraciones (la ganancia por árbol es
  el fork/join de OpenMP, así que crece con el número de árboles y con el de núcleos);
  100 filas 1.17, 0.89 y 0.84× (ruidoso, es la zona del cruce); 250 filas sin límite,
  como antes. El umbral de 100 sale de la tabla de la micro-medida de P5 (0.49-0.79×
  hasta 100 filas, 1.19× en 250).
- threadpoolctl: scikit-learn 1.9.1 exige >= 3.5.0 y el mínimo soportado (1.6) exige
  >= 3.1.0; `ThreadpoolController.limit` existe desde 3.0. `ThreadpoolController()`
  emite un `RuntimeWarning` en Linux si libomp y libiomp están cargadas a la vez: una
  vez por proceso (lru_cache), solo en entornos que ya tienen ese riesgo.
- El controlador se crea en la primera predicción con HGB: libgomp de scikit-learn ya
  está cargada (la importa el módulo de HGB). `_build_predict_function` se llama en cada
  `predict`, no se guarda en el forecaster, así que no viaja a los workers.
- `limit_n_jobs_by_workload`: la guarda `n_fits == 0` ya no se alcanza desde `'auto'`
  (select devuelve 1 sin refit). Se mantiene: es el contrato de la función y la cubre su
  test.
- Nota de versión: la ganancia de H14 (`1` frente a `1lim`) es 12.6-20.0 %
  (MS 24 pasos caliente 12.6, frío 16.0; MS 72 pasos 19.6 y 13.7; serie única 20.0 y
  19.5), no "16 to 20%". Propuesta: "13 to 20%".
- Consistencia: los otros tres puntos de `_search.py` remiten al NOTE de grid search;
  skills y `llms-base.txt` no describen las reglas; la FAQ (celdas 22 y 23) ya decía que
  sin refit no hay ganancia. Los tests hermanos de "rules of each candidate" (grid
  multiseries, bayesian, bayesian multiseries) no tienen el caso sin refit; la regla
  está cubierta en `test_select_n_jobs_backtesting.py` para todos los forecasters.
- verify: lint solo el E402 preexistente; 2120 tests del alcance + 259 de `predict` de
  recursive, multiseries, direct y direct multivariate pasan; `page_meta.py` y el check
  del notebook OK (solo cambiaron celdas markdown).

Correcciones aplicadas tras la revisión (aprobadas por el usuario): nota de versión
"13 to 20%"; caso sin refit añadido a los tres tests hermanos de "rules of each
candidate" (grid multiseries, bayesian, bayesian multiseries); NOTE de
`_validation.py` (l. 616 y 1588) precisa "with an integer `n_jobs`".
