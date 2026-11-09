# Revisión de la paralelización (`n_jobs`) de skforecast, v2

Rama `perf/n-jobs-auto-work-threshold` (base `0.27.x`), código medido hasta el commit
`2926b38` (incluye la fusión de `0.27.x` de las 13:42, ver notas). Diario completo, con
cada hipótesis, experimento y conclusión: [`notas.md`](notas.md). Datos crudos:
`results/*.csv` y `results/*.md`. Scripts: `cases.py`, `run_one.py`, `bench.py` y los
`probe_*.py`.

**Máquina**: contenedor Linux en la nube, 4 vCPU compartidas (Intel Xeon @ 2.10 GHz,
`lscpu`: 4 núcleos sin hyperthreading), 15 GB de RAM. Máquinas de 1 y 2 CPU emuladas con
`taskset`. **Versiones**: Python 3.12.3, joblib 1.6.0, numpy 2.5.4, pandas 2.3.3,
scikit-learn 1.9.1, scipy 1.18.1, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10,
statsmodels 0.15.0, numba 0.68.1, threadpoolctl 3.8.0.

**Método**: cada tabla es mediana (mínimo) de 5 repeticiones con las configuraciones
intercaladas al azar dentro de la misma ejecución, salvo que se indique otra cosa. "Frío"
es un proceso nuevo por medida (cada worker importa skforecast, como en un script);
"caliente" reutiliza los workers de loky (como en un notebook). Se comprobó en todas las
medidas que predicciones y métricas en paralelo coinciden con `n_jobs=1` (huella md5 de
los números redondeados a 1e-7).

## 1. Resumen ejecutivo

Veredicto: la estrategia es correcta en lo esencial para un backtesting suelto (umbral bien calibrado en Linux: equilibrio medido en ~2.4 s; sin errores en los resultados), pero `'auto'` sigue siendo mucho peor que `n_jobs=1` en casos frecuentes que no ve. Mejoras más valiosas:
1. **Detectar LightGBM dentro de `TransformedTargetRegressor`** y avisar con `n_jobs` explícito: hoy `'auto'` pasa de 3.5 s a más de 155 s.
2. **Búsquedas con `refit=False` → 1 job**: hoy son 1.5-3.8× más lentas que `n_jobs=1` con RF, XGBoost, CatBoost y LGBM(1) en 4 CPU.
3. **`cpu_count()` en lugar de `cpu_count() - 1`**: −12 % con 4 CPU; con 2 CPU `'auto'` nunca paraleliza y deja 1.29×.
4. **Corregir el notebook de la FAQ** (búsquedas sin refit, Stats, hilos de XGBoost y LightGBM).
5. **Medir en una máquina grande** limitar a 1 hilo la predicción de HGB (aquí −4 a −8 %; allí, probablemente varias veces).

## 2. Cómo funciona hoy la estrategia

### 2.1 Dónde se resuelve `n_jobs`

| Punto | Fichero:función | Regla |
|---|---|---|
| Validación | `model_selection/_utils.py:check_backtesting_input` | Entero o `'auto'`; cualquier entero pasa (0, -2...). |
| Reglas estáticas | `model_selection/_utils.py:select_n_jobs_backtesting` | `refit` entero distinto de 0/1 → 1; forecaster fuera de {Recursive, RecursiveClassifier, RecursiveMultiSeries} → 1; último paso de `Pipeline` subclase de `LinearModel` → 1; `LGBMRegressor`/`LGBMClassifier` con `n_jobs != 1` → 1; resto → `cpu_count() - 1`. |
| Umbral de trabajo | `model_selection/_utils.py:limit_n_jobs_by_workload` | Solo backtesting suelto con `'auto'`: sin folds que reajusten → 1; `n_fits × fit_seconds` < 2 s (Linux) o 4 s (Windows, macOS y resto) → 1; si no, `min(n_jobs, n_tasks)`. |
| Backtesting single | `_validation.py:_backtesting_forecaster` | `'auto'`: reglas + umbral con el ajuste inicial cronometrado en el principal. Entero: `refit` intermitente → 1 con aviso; `n_jobs <= 0` → `cpu_count()`. |
| Backtesting multiseries | `_validation.py:_backtesting_forecaster_multiseries` | Igual; `n_tasks` sin los folds saltados, `n_fits = sum(fit_folds)`. |
| Backtesting stats | `_validation.py:_backtesting_stats` | `refit=False` → 1; `'auto'` → `select_n_jobs_backtesting` → siempre 1; sin umbral. |
| Búsquedas grid/random/bayes (single y multiseries) | `_search.py:_evaluate_grid_hyperparameters`, `bayesian_search_forecaster` y equivalentes multiseries | `'auto'` → reglas por candidato y se pasa un **entero** al backtesting: **sin umbral**, tampoco con `refit=False`. `OneStepAheadFold` ignora `n_jobs`. |
| Búsquedas stats y EquivalentDate | `_search.py:_evaluate_grid_hyperparameters_stats`, `grid_search_equivalent_date` | `'auto'` → 1. |
| Direct y DirectMultiVariate | `__init__` de cada uno → `utils/utils.py:select_n_jobs_fit_forecaster` | Una vez, en `__init__`: `LinearModel` → 1; `LGBMRegressor` con `n_jobs != 1` → 1; resto → `cpu_count() - 1`. Sin umbral. `set_params` no lo recalcula. |

### 2.2 Qué se paraleliza y cómo

Siempre `joblib.Parallel(n_jobs=n)` con los valores por defecto (backend `loky`,
`batch_size='auto'`, `pre_dispatch='2*n_jobs'`, `max_nbytes='1M'`, `mmap_mode='r'`);
`mmap_mode='c'` solo en `_backtesting_stats` con `Sarimax`. Sin `prefer`, sin
`parallel_config`, sin `threadpoolctl`. Desde la fusión de `0.27.x`, cada tarea va
envuelta en `_run_recording_worker_warnings` (los avisos de los workers se re-emiten en
el principal).

| Qué | Función de la tarea | Datos de cada tarea |
|---|---|---|
| Folds de un backtesting single | `_validation.py:_fit_predict_forecaster` | Trozos de `y`/`exog` del fold (`_prepare_fold_data`) + **el forecaster ya ajustado** en el principal (aunque el fold lo reajuste) + residuos y binner si hay intervalos. |
| Folds multiseries | `_validation.py:_fit_predict_forecaster_multiseries` | Igual, con `series_train`/`exog_train` solo en folds con refit. |
| Folds stats | `_validation.py:_fit_predict_forecaster_stats` | `y` y `exog` completos + lista de folds + forecaster. |
| Pasos de Direct | `direct/_forecaster_direct.py:_fit_one_step_estimator` | `self` (con `estimators_` reiniciados a clones) + matrices completas; arrays > 1 MB como memmap. |
| Candidatos, series, bootstrapping | n/m | No se paralelizan. Los candidatos van en serie y cada uno paraleliza sus folds, reutilizando los workers. |

### 2.3 Paralelismo anidado

- **Detección**: solo `LGBMRegressor`/`LGBMClassifier`, por nombre de clase y atributo
  `n_jobs`, tras desenvolver `Pipeline` (último paso). `TransformedTargetRegressor` y otros
  meta-estimadores no se desenvuelven.
- **loky** arranca cada worker con `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`, `NUMBA_NUM_THREADS`... = `max(cpu_count() // n_jobs, 1)`
  (`joblib/_parallel_backends.py:_prepare_worker_env`). Un `Parallel` anidado en un
  worker usa `ThreadingBackend`.
- **Hilos efectivos dentro de un worker de un `Parallel(n_jobs=3)` en 4 CPU** (medido,
  `probe_threads.py`):

  | Estimador (defecto) | Principal | Worker | ¿Respeta el límite de loky? |
  |---|---|---|---|
  | HGB (OpenMP) | 4 | 1 | sí |
  | XGBoost (`n_jobs=None`) | 4 | 1 | sí |
  | CatBoost (`thread_count=-1`) | 4 | **4** | no |
  | LightGBM (`n_jobs=None` o -1) | 4 | **4** | no |
  | `n_jobs=-1` anidado (RF/ET) | 4 | **4** | no |

- **Predicción**: LightGBM predice con los hilos del ajuste, limitados por
  `OMP_NUM_THREADS` (`utils.py:_build_predict_function`); RF/ET predicen árbol a árbol en
  Python; HGB, XGBoost y CatBoost usan sus hilos por defecto.
- **numba**: solo en `skforecast/stats` (Arima, Ets), `@njit(cache=True)` sin
  `parallel=True`. La primera ejecución tras instalar compila y escribe la caché.

### 2.4 Cómo influyen los parámetros

- `refit=True`: `n_fits = n_folds - 1`. `refit=False`: 1 en backtesting suelto, pero
  `cpu_count() - 1` en una búsqueda. `refit` entero: siempre 1.
- `fixed_train_size=False` (defecto): los folds crecen y `fit_seconds` subestima
  (estimación ≈ 80 % del real, medido).
- Intervalos: no entran en la estimación de trabajo.
- Nº de folds/candidatos: `n_jobs` se limita a `n_tasks` solo en el backtesting suelto.
- Stats: `refit` se **fuerza a `True`** para todo estimador que no sea
  `skforecast.stats.Sarimax` (aviso `IgnoredArgumentWarning`, `_validation.py` l.
  2263-2277). `Sarimax` admite `refit=False` y entero (y entonces `n_jobs=1`). La
  suposición "en stats `refit` siempre es `True`" es cierta salvo con `Sarimax`.

## 3. Errores de correctitud

**Resultados**: ninguno. En todas las medidas (487 medidas de 100 configuraciones,
todas las familias), las predicciones y métricas en paralelo coinciden con `n_jobs=1`. La única
diferencia numérica es ruido de coma flotante en lineales con `refit=True`
(Pipeline+Ridge, Ridge: ≤ 4e-11; LinearRegression: ≤ 1e-13).

**Comportamiento** (no cambian los números, sí la promesa de `'auto'`):

| # | Error | Consecuencia medida |
|---|---|---|
| E1 | `'auto'` no desenvuelve `TransformedTargetRegressor` (ni otros meta-estimadores): un LightGBM dentro se trata como un estimador sin hilos. | Con trabajo por encima del umbral, `'auto'` lanza 3 workers × 4 hilos de LightGBM: **> 155 s frente a 3.5 s** con `n_jobs=1`. |
| E2 | Un `n_jobs` explícito > 1 con `LGBMRegressor()` (hilos por defecto) no avisa. | 4 folds: 52.8 s frente a 0.55 s (~100×); 20 folds: > 166 s. |
| E3 | En una búsqueda con `refit=False`, `'auto'` paraleliza aunque el backtesting suelto, con los mismos datos, usaría 1. | 1.5-3.8× más lento que `n_jobs=1` (sección 4.6). |
| E4 | `n_jobs` negativo distinto de -1 (p. ej. -2) o 0 → `cpu_count()`, no `cpu_count() + 1 + n_jobs` como en joblib. | Menor: el usuario que pide "todos menos uno" obtiene todos. |
| E5 | Direct y DirectMultiVariate resuelven `'auto'` en `__init__`; `set_params` no lo recalcula. | Menor: un forecaster guardado en una máquina grande lleva su `n_jobs` a otra. |

## 4. Hallazgos por familia de modelos y escenario

### 4.1 Coste fijo: arrancar los workers

| CPUs | n_jobs | frío | caliente | import de skforecast en el principal |
|---|---|---|---|---|
| 4 | 3 | 1.30 (1.23) s | 11 ms | 1.17 s |
| 4 | 4 | 1.31 (1.27) s | 11 ms | 1.12 s |
| 2 | 2 | 1.31 (1.25) s | 11 ms | 1.14 s |

Umbral de trabajo de un backtesting suelto (RF con pocos árboles para escalar el
trabajo, n=2000, 20 folds, `refit=True`, 4 CPU):

| Árboles | Modo | n_jobs=1 | n_jobs=3 | `'auto'` (elige) | Trabajo estimado |
|---|---|---|---|---|---|
| 4 | frío | 1.27 (1.06) | 2.02 (1.98) | 1.28 (1) | n/m |
| 8 | frío | 2.40 (2.38) | 2.42 (2.33) | 2.40 (1 o 3) | 1.94 |
| 12 | frío | 3.61 (3.40) | 2.96 (2.85) | 2.90 (3) | n/m |
| 1 | caliente | 0.31 (0.26) | 0.17 (0.15) | 0.36 (1) | 0.32 |
| 4 | caliente | 1.21 (1.13) | 0.54 (0.48) | 1.17 (1) | 0.96 |

En frío, el umbral de 2 s cae justo en el equilibrio. En caliente, paralelizar gana a
cualquier tamaño (1.8-2.2×), y el umbral deja hasta ~1.5 s por llamada sin aprovechar en
un notebook. Es una pérdida acotada y compatible con el criterio "nunca mucho peor que 1".

### 4.2 Lineales (LinearRegression, Ridge, Lasso, Pipeline con escalado)

`'auto'` → 1 siempre (regla estática). Backtesting con `refit=True`, frío:

| Caso | Folds | n_jobs=1 | n_jobs=3 | Trabajo estimado | Memoria 1 / 3 |
|---|---|---|---|---|---|
| Ridge, 20 000 filas, 48 lags | 50 | 0.62 | 1.82 | 1.06 | n/m |
| Pipeline(Scaler, Ridge), 50 000 filas, 168 lags, 5 exog | 10 | **2.38 (2.18)** | 2.81 (2.79) | 2.23 | n/m |
| ídem | 20 | 4.57 (4.36) | **3.64 (3.42)** | n/m | n/m |
| ídem (1 rep) | 100 | 22.7 | **10.7** | 26.1 | 407 / 1486 MB |
| Multiseries Ridge, 50 series × 10 000, 24 lags | 80 | 12.09 (11.09) | **8.65 (8.44)** | n/m | 427 / 1473 MB |

- La regla acierta en los tamaños típicos (Ridge, 20 000 filas, 50 folds: 0.6 s).
- Deja 1.3-2.1× en backtests lineales grandes, a cambio de ~3.4× de memoria.
- En lineales el equilibrio está más alto que en árboles (~3-3.5 s reales frente a
  ~2.4 s): 3 workers rinden solo ~1.9× (memoria y BLAS de 1 hilo en los workers).
- Con ajustes de milisegundos, `t_fit × n_fits` **sobreestima** (1.06 s estimados frente
  a 0.55 s reales: el ajuste inicial carga costes de primera llamada). Quitar la regla
  estática y fiarlo todo al umbral actual perdería ~0.4 s (18 %) cerca del umbral.

### 4.3 Árboles de scikit-learn (RandomForest, ExtraTrees, HistGradientBoosting)

| Escenario (4 CPU) | n_jobs=1 | `'auto'` |
|---|---|---|
| RF 25 árboles, backtesting `refit=True`, 20 folds, frío | 7.35 (6.99) | **4.44 (4.31)** (3); con 4 workers 3.90 |
| HGB, backtesting `refit=True`, n=2000, frío | 6.02 (5.59) | **3.42 (3.29)** |
| HGB, backtesting `refit=True`, n=10 000, 48 lags, frío | 4.56 (4.39) | **3.76 (3.43)** |
| RF, búsqueda `refit=False` (4 candidatos), frío | **1.54 (1.37)** | 3.34 (3.27) |
| RF, ídem, caliente | **1.52 (1.42)** | 1.84 (1.78) |
| HGB, búsqueda `refit=False`, frío | **2.59 (2.21)** | 2.82 (2.71) |
| HGB, ídem, caliente | 2.38 (2.27) | **1.49 (1.40)** |
| RF backtesting `refit=False` + bootstrapping (250) | **2.34 (2.20)** (= `'auto'`) | 3.56 (3.37) con 3 |

- RF/ET con refit: `'auto'` acierta. Sin refit en búsquedas pierde: el forecaster
  ajustado pesa 13.6 MB (RF) y 21.6 MB (ET) y viaja en cada tarea.
- HGB es el que más gana con procesos: su paralelismo de hilos rinde poco con datos
  pequeños, y predecir una fila con 4 hilos cuesta 0.95-1.02 ms frente a 0.54-0.66 ms
  con 1. En un backtesting secuencial, limitar a 1 hilo solo la predicción ahorra un
  4-8 % con 4 CPU (en máquinas con muchos hilos puede ser bastante más, sección 6).

### 4.4 LightGBM, XGBoost y CatBoost

| Escenario (4 CPU, n=2000 salvo indicación) | n_jobs=1 | `'auto'` | Otro |
|---|---|---|---|
| LGBM por defecto, backtesting `refit=True`, 4 folds | **0.55** | 1 | n_jobs=3: **52.8 s** |
| `TransformedTargetRegressor(LGBM 400)`, 10 folds | **3.53 (3.45)** | **> 155 s** (elige 3) | n/m |
| XGB, backtesting `refit=True`, frío | **3.13 (3.08)** | 3.94 (3.65) | n/m |
| XGB, ídem n=10 000, 48 lags | **3.45 (3.19)** | 4.31 (4.04) | n/m |
| XGB, ídem n=2000, caliente (`n_jobs=1` sin contaminar) | 3.13 | **2.77 (2.68)** | n/m |
| XGB, búsqueda `refit=True` | **4.66 (4.49)** | 5.87 (5.77) | n/m |
| XGB, búsqueda `refit=False` | **0.73 (0.68)** | 2.75 (2.62) | n/m |
| CatBoost 300 it., backtesting `refit=True` | 9.31 (8.79) | **8.23 (7.81)** | n/m |
| CatBoost, ídem n=10 000 | 11.00 (10.75) | **9.97 (9.70)** | n/m |
| CatBoost, búsqueda `refit=False` | **1.28 (1.16)** | 2.50 (2.35) | n/m |
| LGBM(`n_jobs=1`), búsqueda `refit=False`, frío / caliente | **0.57 / 0.49** | 1.99 / 0.73 | n/m |

- **LightGBM con sus hilos dentro de varios workers es catastrófico en Linux** (~100×):
  la regla lo evita solo si ve el `LGBMRegressor`.
- **XGBoost** respeta el límite de loky: con 4 CPU, 3 procesos de 1 hilo solo sustituyen
  a 4 hilos y `'auto'` pierde un 25-26 % en frío.
- **Hallazgo nuevo: un backtesting paralelo con XGBoost ralentiza 2.5× los ajustes de
  XGBoost posteriores del proceso.** El `QueueFeederThread` de loky serializa el
  forecaster ajustado; `save_raw` abre una región OpenMP en ese hilo y libgomp le crea un
  segundo equipo de hilos. Con 8 hilos gestionados para 4 CPU, libgomp reduce la espera
  activa y los hilos del principal se duermen en cada barrera. Reproducción mínima sin
  skforecast: `pickle.dumps(modelo)` desde un hilo que sigue vivo → ajustes de 162 a
  417 ms; al terminar el hilo, vuelven a 204 ms. Dura lo que el ejecutor de loky (300 s
  sin uso). Afecta al ajuste inicial de cada candidato de una búsqueda, a
  `return_best=True` y a cualquier `n_jobs=1` posterior (3.1 → 7.8 s). Enviar copias sin
  ajustar a los folds con refit no basta: el fold 0 (ajustado en el principal) sigue
  viajando.
- **CatBoost** ignora el límite (3 × 4 hilos) y aun así gana un 10-12 % con refit.

### 4.5 Modelos estadísticos (ForecasterStats)

`backtesting_stats`, `refit=True` (forzado salvo `Sarimax`), frío, caché de numba
caliente:

| Modelo | `'auto'` (= 1) | n_jobs=3 |
|---|---|---|
| Ets(m=1), n=500, 10 folds | **0.60 (0.52)** | 2.49 (2.43) |
| Ets(m=24), selección automática, n=1500, 10 folds | **3.28 (2.75)** | 4.37 (4.18) |
| Arima(2,1,1)(1,1,1)24, n=1500, 10 folds (3 reps) | 18.55 (18.55) | **11.92 (10.60)** |

- `'auto'` = 1 acierta en los casos pequeños y medianos; deja 1.56× en modelos
  estacionales de orden fijo con series largas.
- Con `freeze_params=True` (defecto) el primer ajuste incluye la selección y es 25-70×
  más caro que los reajustes (auto-Arima 2.21 s frente a 0.031 s; auto-Ets 1.79 s frente
  a 0.080 s): **no sirve para estimar el trabajo**.
- La primera ejecución tras instalar compila numba e invierte la conclusión (Ets(m=24):
  8.47 s con 1 frente a 4.55 s con 3). Ojo al medir.

### 4.6 Búsquedas

| Rejilla, 4 CPU | n_jobs=1 | `'auto'` (folds) |
|---|---|---|
| `refit=False`, RF / XGB / CatBoost / LGBM(1), 24 pasos, frío | **1.54 / 0.73 / 1.28 / 0.57** | 3.34 / 2.75 / 2.50 / 1.99 |
| `refit=False`, RF, 168 pasos, frío | **9.79 (9.34)** | 11.29 (10.96) |
| `refit=False`, HGB, 168 pasos, frío | 3.69 (3.56) | **2.80 (2.63)** |
| `refit=True`, RF barato, 10 folds | 3.79 (3.66) | **3.25 (3.11)** |
| `refit=True`, XGB, 10 folds | **4.66 (4.49)** | 5.87 (5.77) |

Prototipo de **candidatos en paralelo** (`cases.py:grid_candidates`, cada candidato un
backtesting con `n_jobs=1`; mismas métricas):

| RF, `refit=False`, 20 folds | candidatos en serie | ×3 workers | ×4 workers |
|---|---|---|---|
| 12 candidatos (misma ejecución) | 8.43 (8.11) | 4.87 (4.67) | **4.28 (4.00)** |

Con `refit=True` la regla actual funciona (el arranque se amortiza entre candidatos),
salvo con XGBoost. Con `refit=False` decide la relación entre el coste de predecir un
fold y el de enviar el modelo ajustado: gana HGB (predicción cara, modelo de 0.4 MB) y
pierde el resto. Paralelizar candidatos da 1.7-2.0× frente a `n_jobs=1` y ~3× frente al
`'auto'` actual.

### 4.7 Direct (ajuste de los pasos, sin umbral)

| `fit` + `predict`, frío | n_jobs=1 | `'auto'` (= 3) |
|---|---|---|
| RF 10 árboles, n=500, 6 pasos | **0.19 (0.15)** | 1.51 (1.49) |
| RF 10 árboles, n=2000, 12 pasos | 2.25 (1.98) | 2.23 (2.14) |
| XGB, n=2000, 12 pasos | **1.99 (1.78)** | 2.99 (2.87) |
| HGB, n=2000, 12 pasos | 3.35 (3.17) | **2.50 (2.32)** |

### 4.8 Otros

- **`cpu_count() - 1` frente a `cpu_count()`** (RF 25 árboles, `refit=True`, frío): 4 CPU:
  3 workers 4.44 s, 4 workers **3.90 s** (−12 %); 2 CPU: `'auto'` (= 1) 7.44 s, 2 workers
  **5.78 s** (1.29×).
- **Memoria en multiseries**: los trozos de los folds son vistas (pico 402 MB con 10
  folds, 426 MB con 80); el paralelismo sí multiplica la memoria (3.4× con 3 workers).
- **Intervalos sin refit**: `'auto'` = 1 acierta (bootstrapping vectorizado).

## 5. Propuestas priorizadas

Orden por impacto en los usuarios, riesgo y coste de mantenimiento. Ninguna cambia la
API pública; todas cambian el valor que `'auto'` elige y llevan nota de versión.

### P1. Desenvolver meta-estimadores para detectar LightGBM, y avisar con `n_jobs` explícito

- **Cambio**: en `select_n_jobs_backtesting` y `select_n_jobs_fit_forecaster`, buscar el
  estimador real a través de `Pipeline` (último paso), `TransformedTargetRegressor`
  (`regressor`) y, en general, atributos `regressor`/`estimator`, recursivamente. Además,
  `IgnoredArgumentWarning` (o un aviso propio) cuando `n_jobs > 1` explícito coincide con
  un LightGBM de hilos por defecto, explicando que conviene `n_jobs=1` en el estimador.
- **Evidencia**: E1 y E2 (sección 3): > 155 s frente a 3.5 s; 52.8 s frente a 0.55 s.
- **Peor caso**: ninguno en tiempo (pasa a 1, que es lo medido como mejor con 4 CPU).
  Con 22 procesadores la revisión previa medía que LightGBM con hilos y 21 workers en
  Windows ganaba ×1.1-1.3: se renuncia a eso, igual que hoy con LightGBM sin envolver.
- **Coste**: bajo (una función auxiliar y tests).
- **Falta medir**: nada para decidir.

### P2. Búsquedas con `refit=False`: `'auto'` → 1

- **Cambio**: `select_n_jobs_backtesting` devuelve 1 si `refit` es `False`. El
  backtesting suelto ya lo hacía con el umbral; la búsqueda lo hereda.
- **Evidencia**: sección 4.6: con 4 CPU, `'auto'` es 1.5-3.8× más lento que `n_jobs=1`
  con RF, XGB, CatBoost y LGBM(1), en frío y (RF, LGBM(1)) en caliente.
- **Peor caso**: HGB y horizontes largos, donde paralelizar predicciones gana: aquí
  1.3-1.6× (2.80 → 3.69 s; 1.49 → 2.38 s en caliente); en la revisión previa, con 22
  procesadores y 168 pasos, HGB 3.42 → 16.21 s y LGBM(1) 1.38 → 2.06 s. Nunca peor que
  `n_jobs=1`. P5 reduciría el caso HGB; el usuario puede pasar un entero.
- **Alternativa (P2b)**: decidir con el trabajo de predicción medido: predecir el fold 0
  en el principal, cronometrarlo y paralelizar el resto solo si
  `t_fold × (n_folds − 1)` supera un umbral. Conserva los casos HGB/horizonte largo;
  coste medio y falta calibrarlo (RF con 168 pasos lo superaría y pierde un 15 %).
- **Coste**: mínimo (una condición, tests de búsquedas, docstrings, notebook).

### P3. `cpu_count()` en lugar de `cpu_count() - 1`

- **Cambio**: el número de workers cuando las reglas y el umbral deciden paralelizar.
- **Evidencia**: −12 % de tiempo con 4 CPU; 1.29× con 2 CPU, donde hoy `'auto'` nunca
  paraleliza. La revisión previa: neutro con 22 y 8 lógicos; 5-9 % mejor con 4 lógicos
  (2c/4t) en 3 de 4 celdas.
- **Peor caso**: con 2 vCPU que sean un núcleo físico con HT (Colab gratis) la ganancia
  será menor; con trabajo justo por encima del umbral, el empate actual. Responsividad
  del escritorio durante cálculos largos (el usuario puede limitar con
  `LOKY_MAX_CPU_COUNT`).
- **Coste**: bajo; recalibrar el umbral no hace falta (el arranque no depende de k
  entre 2 y 4).
- **Falta medir**: 2 vCPU con HT real; máquinas de 6-8 núcleos en Linux.

### P4. Corregir el notebook de la FAQ

- "Without refit ... `'auto'` runs them in one process" y "Do not expect gains from
  `n_jobs` without refit": falso en búsquedas (salvo que se aplique P2).
- "Do not expect gains ... with `ForecasterStats`": con `n_jobs` entero y refit gana
  (1.56× aquí; ×2.3-2.5 en caliente en la revisión previa).
- "XGBoost ... compete for the same cores": XGBoost respeta el límite de loky; CatBoost,
  LightGBM y `n_jobs=-1` anidado no.
- Añadir: `LGBMRegressor()` por defecto con `n_jobs > 1` en skforecast (o envuelto, con
  `'auto'`, hasta P1) es ~100× más lento en Linux; `n_jobs=1` en el estimador.
- **Coste**: bajo. **Riesgo**: nulo.

### P5. HGB: predecir con 1 hilo en el bucle recursivo (medir antes en una máquina grande)

- **Cambio**: igual que D9 para LightGBM: limitar los hilos OpenMP de
  `HistGradientBoostingRegressor` durante la predicción recursiva (por ejemplo, con
  `threadpoolctl`, dependencia de scikit-learn, alrededor del bucle).
- **Evidencia**: aquí, −4 a −8 % en backtesting (micro: 0.95 → 0.66 ms por fila). La
  revisión previa midió HGB secuencial muy lento con 22 hilos (Direct S: 5.5 s frente a
  0.6 s; HGB L sin refit: 2.94 s con 1 frente a 0.52 s con 21 workers): ahí la ganancia
  puede ser de varias veces para todo usuario con `n_jobs=1` o `refit=False`.
- **Peor caso**: predicciones con muchas filas por llamada (bootstrapping, multiseries)
  más lentas con 1 hilo; limitarlo solo a llamadas de una fila lo evita.
- **Falta medir**: el efecto con 8-22 hilos (Windows y Linux).

### P6. Umbral de trabajo para Direct y, con cuidado, para Stats

- **Direct**: ajustar y cronometrar el primer paso en el principal y paralelizar el resto
  solo si `t_paso × (pasos − 1)` supera el umbral. Evita +1.3 s (8×) en Direct baratos;
  el equilibrio medido está otra vez en ~2.2-2.4 s. Se puede aprovechar para resolver
  `'auto'` en `fit` y no en `__init__` (E5), pero cambia el atributo `n_jobs`: con
  cuidado.
- **Stats**: aplicar `limit_n_jobs_by_workload` solo cuando el ajuste inicial sea
  representativo (`freeze_params=False`, o un estimador sin selección automática).
  Ganaría 1.56× en Arima estacional de orden fijo con series largas.
- **Coste**: medio. **Falta medir**: Direct con LGBM(1) y DMV; Stats con
  `freeze_params=False`.

### P7. Lineales: umbral propio en lugar de "siempre 1"

- **Cambio**: que los lineales pasen por el umbral con un mínimo mayor (≈ 2× el general,
  4 s estimados en Linux), por la sobreestimación con ajustes de milisegundos y la peor
  eficiencia de los workers.
- **Evidencia**: 1.26× con 4.6 s secuenciales, 2.1× con 22.7 s (sección 4.2). Cubriría
  también `LinearModel` lentos (`QuantileRegressor`, `TheilSenRegressor`, `LassoCV`),
  que hoy van a 1 sin medir.
- **Peor caso**: ~3.4× de memoria; cerca del umbral, empate.
- **Coste**: bajo-medio. **Prioridad**: media-baja (pocos usuarios con backtests
  lineales de más de 5 s).

### P8. XGBoost en máquinas pequeñas y contaminación de libgomp

- Con 4 CPU, paralelizar XGBoost pierde un 25-50 % (backtesting, búsqueda con refit,
  Direct) y contamina el proceso; con 8-22 procesadores la revisión previa medía
  ×2.5-4.9 a favor. Una regla del tipo "XGBoost con hilos → 1 si `cpu_count() <= 4`"
  depende de la máquina: **no la recomiendo sin medir 6-8 núcleos en Linux**.
- Contra la contaminación: predecir el fold 0 en el principal y enviar copias sin
  ajustar a los folds con refit (de paso, menos IPC). Coste medio; falta medir en
  máquinas grandes y en Windows (otro runtime de OpenMP).

### P9. Búsquedas: candidatos en paralelo (proyecto aparte)

- **Evidencia**: 1.7-2.0× frente a `n_jobs=1` con 12 candidatos y `refit=False`, frío
  incluido; ~3× frente al `'auto'` actual. No envía modelos ajustados.
- **Coste**: alto: avisos por candidato, `output_file`, barra de progreso, la búsqueda
  bayesiana (secuencial por naturaleza), y limitar los hilos de LightGBM/CatBoost dentro
  de los workers.

### Descartadas o sin cambio

- Umbral consciente de workers calientes: gana ≤ 1.5 s por llamada y depende de estado
  privado de loky.
- Enviar el forecaster sin ajustar a los folds con refit, solo: ≤ 5 % de IPC.
- Intervalos con bootstrapping sin refit: `'auto'` = 1 acierta.
- Memoria de multiseries: no crece con los folds.
- `batch_size` y el backend `threading`: no medidos. El bucle recursivo de skforecast es
  Python (GIL), así que `threading` solo ayudaría en la parte nativa del ajuste
  [inferido, sin medida].

## 6. Contraste con la revisión previa (`dev/n_jobs_review/`)

Datos de otra revisión, para plataformas que aquí no se pueden medir: Windows 11 y WSL2
en un Intel Core Ultra 9 185H (16 núcleos físicos, 22 lógicos, híbrido), Python 3.13.

**Coincidencias**

| Punto | Previa | Aquí |
|---|---|---|
| Arranque en frío | Windows 5.7 s (21 workers), WSL2 1.34-1.46 s (2-4) y 2.94 s (21) | 1.3 s (2-4) |
| Equilibrio en Linux | ~2.8 s reales (21 workers) | ~2.4 s reales (3 workers) |
| Estimación / real | mediana 0.78 | 0.81 |
| Sin refit (suelto), intervalos sin refit, `refit` entero, Stats en frío | 1 | 1 |
| LightGBM con hilos en los workers | > 600 s en WSL2 | > 155 s / ~100× |
| `cpu_count()` frente a `-1` con 4 lógicos | `-1` 5-9 % mejor (llamado ruido) | 12 % mejor |
| Direct necesita umbral | inferido | medido (8× en el peor caso) |

**Discrepancias y causa**

1. **Búsquedas sin refit**. Previa: "mantener" (HGB ×4.7, LGBM(1) ×1.5 a favor con 22
   procesadores, caliente, 168 pasos). Aquí pierde con RF, XGB, CatBoost y LGBM(1) en
   4 CPU con 24 pasos (y RF también con 168). Causa: **máquina y caso**; no midieron
   RF/ET/XGB/CatBoost sin refit en búsquedas ni horizontes cortos. Medida corta aquí: HGB
   con 168 pasos también gana con 4 CPU (1.32×), así que su conclusión para HGB se
   sostiene; la mía para el resto, también.
2. **XGBoost**. Previa: sin regla nueva (×2.5-4.9 a favor con 8 y 22). Aquí: −25 % en
   frío con 4 CPU. Causa probable: número de núcleos (con 4 hilos XGBoost ya aprovecha la
   máquina) y horizonte. Revisión crítica de su método: `bench.py` ejecuta las celdas en
   caliente en un solo proceso barajadas; en Linux, una celda XGBoost con `n_jobs=1`
   posterior a una paralela quedaría inflada 2.5× por la contaminación de libgomp. Sus
   celdas XGBoost eran de Windows, con otro runtime de OpenMP: no sé si les afecta.
3. **Hallazgos nuevos aquí**, sin equivalente allí: la contaminación de XGBoost, el caso
   `TransformedTargetRegressor(LGBM)` con `'auto'`, el efecto de `freeze_params` en
   Stats, la sobreestimación con ajustes de milisegundos y el prototipo de candidatos en
   paralelo.
4. **HGB con muchos hilos**: allí el secuencial era hasta 9× más lento por los 22 hilos de
   OpenMP; aquí, con 4, el efecto es pequeño. Explica parte de por qué allí paralelizar
   HGB sin refit ganaba tanto, y motiva P5.

Sus scripts: `strategies.py` evalúa el umbral (E3) con el tiempo secuencial medido como
estimación (lo declaran); el escenario frío es "caliente + 72 % del arranque". Son
aproximaciones razonables y mis medidas en frío reales caen en el mismo orden.

**Documentación frente a código**

- Notebook de la FAQ: la tabla de reglas y el umbral coinciden con el código. Cuatro
  inexactitudes (P4): búsquedas sin refit, Stats, XGBoost, aviso de LightGBM.
- `tools/ai/llms-base.txt` y skills: solo firmas (`n_jobs='auto'`); sin contradicciones.
- Docstrings: el `n_jobs` de las búsquedas dice que usan `select_n_jobs_backtesting`,
  correcto; no dice que con `refit=False` paralelizan aunque el backtesting suelto no lo
  haría (E3).

## 7. Limitaciones

- **Una sola máquina Linux de 4 vCPU compartidas**. El ruido entre ejecuciones llegó al
  10-15 % (por eso todas las comparaciones son dentro de la misma ejecución). Los
  resultados con 1-2 CPU son con `taskset` sobre la misma VM.
- **Sin Windows ni macOS**: el arranque de los workers y el runtime de OpenMP cambian
  (Windows: `spawn` más caro, 5.7 s con 21 workers según la revisión previa; macOS sin
  medir en ninguna de las dos). Para medirlo: `run_one.py`/`bench.py` funcionan en
  cualquier sistema POSIX; en Windows habría que sustituir `taskset` por la afinidad de
  `psutil`. La contaminación de XGBoost hay que comprobarla con
  `probe_xgb_after_parallel.py` en Windows y macOS (libomp de LLVM).
- **Sin máquinas de muchos núcleos**: para 8-22 hay que fiarse de la revisión previa o del
  modelo `T_par ≈ S + W / (n·e)` (aquí `S` = 1.3 s, `e` ≈ 0.78 en RF, ≈ 0.6-0.7 en
  lineales grandes). Es una estimación: con 15 workers y el mismo `S`, el equilibrio
  bajaría a ~1.4 s de trabajo secuencial; en Windows sube con `S`. P3, P5 y P8 necesitan
  esa medida.
- **Experimentos cortos (< 3 min)**: los tamaños grandes (Arima estacional, lineal de 100
  folds) tienen 1-3 repeticiones, indicado en cada tabla.
- **No medido**: Direct con LGBM(1) y DirectMultiVariate, `ForecasterRecursiveClassifier`,
  `QuantileRegressor` y otros `LinearModel` lentos, `batch_size`, backend `threading`,
  `inner_max_num_threads`.
