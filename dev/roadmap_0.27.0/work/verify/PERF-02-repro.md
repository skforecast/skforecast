# PERF-02: paralelismo en stats y límite de hilos BLAS en Ets y Sarimax

**Veredicto: keep (reproducido), esfuerzo S.** Son dos cambios pequeños e independientes. El
más valioso es el límite de hilos BLAS: corrige una degradación silenciosa de 3x a 70x en
cuanto la máquina tiene otra carga, y es un problema aguas arriba (scipy>=1.15) que Optuna,
dependencia del núcleo de skforecast, ya parchea por su cuenta. El `n_jobs='auto'` da 2,5x a 3x
en todo backtesting de Arima, Ets y Arar, que siempre reajustan en cada fold.

Script: `dev/roadmap_0.27.0/repro/PERF-02.py` (partes `env compile workers bt blas pool profile`).
Entorno: Python 3.12.3, numpy 2.5.4, pandas 2.3.3, scipy 1.18.1 (OpenBLAS propio 0.3.31.dev),
numba 0.68.0, statsmodels 0.15.0, scikit-learn 1.9.1, joblib 1.6.0, threadpoolctl 3.7.0,
skforecast 0.26.0. Máquina de 4 núcleos **compartida con otros agentes** (carga media entre 0,4 y
17 durante las medidas; se imprime en cada bloque): los valores absolutos varían mucho, las
comparaciones con y sin límite se hicieron seguidas. Medianas de 3 repeticiones. No se solapa con
`perf/ets-numba-compile` (#1400), que solo toca `_ets_nelder_mead` y `_simulate_ets_jit`.

## Compilación numba (aparte)

Todas las funciones de `skforecast/stats` usan `@njit(cache=True)`. Primer ajuste en un proceso
nuevo: caché vacía Ets 8,97 s y Arima 10,25 s; caché caliente 1,17 s y 0,19 s. Los workers de
loky leen la caché que escribe el proceso padre (el primer `fit` de `_backtesting_stats` se hace
antes de `Parallel`): arrancar 3 workers cuesta 1,91 s una vez por sesión y 0,10 s después.
Las tablas siguientes excluyen la compilación (calentamiento previo).

## 1. Límite de hilos BLAS (lo más importante)

Causa: desde scipy 1.15 el L-BFGS-B está en C y llama al OpenBLAS de scipy, que usa 4 hilos con
vectores minúsculos; si hay más hilos que núcleos libres cada llamada a `setulb` pasa de
microsegundos a milisegundos (scipy/scipy#22438). Lo usan `_optimize_ets`
(`skforecast/stats/exponential_smoothing/_ets_base.py:1052-1059`) y statsmodels en `Sarimax.fit`
(`skforecast/stats/_sarimax.py:434`, `method='lbfgs'` por defecto). Arima usa BFGS y no se ve
afectado (6,59 s frente a 6,00 s, ruido). Perfil con 2 procesos simultáneos: `setulb` 8,34 s de
9,97 s (84%), 3097 llamadas a 2,7 ms. Limitar solo el OpenBLAS de scipy basta (Ets AAA n=300:
0,95 s a 0,065 s); limitar solo el de numpy no cambia nada.

Segundos por ajuste (n=1000), sin límite frente a `threadpool_limits(1, user_api="blas")`:

| Situación | Ets automático | Sarimax(2,1,1)(1,0,1,7) |
|---|---|---|
| 1 proceso, máquina en reposo (carga 0,4) | 1,01 / 0,89 | n/d |
| 1 proceso + 1 proceso ocupado (bucle Python) | 2,77 / 0,92 (3x) | 2,04 / 0,62 (3,3x) |
| 1 proceso + 2 procesos ocupados | 4,75 / 0,91 (5,2x) | 7,93 / 0,62 (13x) |
| 2 ajustes a la vez (dos notebooks) | 7,3 a 8,3 / 0,85 a 0,89 (9x) | 19,5 a 20,4 / 0,66 a 0,74 (28x) |
| 4 ajustes a la vez | 31 a 33 / 1,9 a 2,4 (15x) | 45 / 0,5 a 0,7 (70x) |
| 1 proceso con carga externa 9 (otros agentes) | | 10,82 / 0,66 (16x) |

Patrón "un modelo por serie" (8 series, Ets automático): secuencial 7,35 s; joblib n_jobs=4
1,91 s (loky ya limita los hilos de sus workers); `multiprocessing.Pool(4)` **73,6 s**;
`Pool(4)` con el límite 2,56 s. En reposo el límite no empeora nada.

Precedente: Optuna 5.0 (`optuna/_gp/thread_limiting.py`) limita OpenBLAS a 1 hilo con
scipy>=1.15 citando el mismo issue. En contenedores con cuota de CPU (Kubernetes, Databricks)
OpenBLAS ve todos los núcleos del nodo, así que el caso "más hilos que núcleos" es el normal ahí
(razonado, no medido: `OPENBLAS_NUM_THREADS=16` se recorta a 4 y no sirve para emularlo).

## 2. `n_jobs='auto'` en `backtesting_stats` y búsquedas stats

`select_n_jobs_backtesting` devuelve 1 para `ForecasterStats` (`skforecast/model_selection/_utils.py:706`).
Además `_backtesting_stats` fuerza `refit=True` con cualquier estimador que no sea Sarimax
(`_validation.py:2010-2021`), así que todo backtesting de Arima, Ets o Arar es un ajuste
independiente por fold que hoy corre en serie. `grid_search_stats` y `random_search_stats` pasan
`n_jobs` al mismo código (`_search.py:2541,2640`).

1000 puntos, steps=7, 30 folds, refit=True (predicciones idénticas, `allclose` atol 1e-8, y mismo
MAE en todas las configuraciones):

| Estimador | n_jobs=1 | n_jobs=1 + límite BLAS | n_jobs=3 (auto propuesto) | n_jobs=4 |
|---|---|---|---|---|
| Arima(2,1,1)(1,0,1)[7] | 6,59 s | 6,00 s | 2,59 s (2,5x) | 3,03 s |
| Ets automático | 194 s (194; 219; 44) y 24,1 s en otro momento | 23,3 s | 8,37 s (2,8x frente a 1+límite) | 6,98 s |
| Ets AAA (aproxima el coste con BUG-08 arreglado) | 2,32 s | 2,32 s | 0,95 s (2,4x) | 0,88 s |
| Sarimax(2,1,1)(1,0,1,7) | 20,4 s | 16,1 s | 6,73 s (3,0x) | 5,14 s |

El Ets automático con n_jobs=1 pasa de 24 s a 194 s según la carga de la máquina: es el problema
del punto 1 dentro del flujo por defecto. Con n_jobs>1 desaparece porque loky limita los hilos
BLAS de sus workers.

Matiz: arrancar los workers cuesta unos 1,9 s la primera vez por sesión. Con ajustes baratos y
pocos folds (Ets fijo, unos 0,07 s por fold) la primera llamada puede quedar igual o algo más
lenta; con Arima, Sarimax o modelos automáticos con `freeze_params=False` la ganancia es clara.
Es el mismo compromiso que skforecast ya acepta para `ForecasterRecursive`.

## Arreglo propuesto

1. En `Ets.fit` y `Sarimax.fit` (y en `Arima.fit` por homogeneidad, útil si
   `optim_method="L-BFGS-B"`) envolver el ajuste una sola vez en
   `threadpoolctl.threadpool_limits(1, user_api="blas")`. `threadpoolctl` ya llega como
   dependencia de scikit-learn (no hace falta añadirla a `pyproject.toml`, aunque conviene
   declararla). Coste medido: 1,45 ms por llamada a `threadpool_limits` (no 0,1 ms como decía la
   candidata), despreciable frente a un ajuste de 60 ms o más; con un `ThreadpoolController`
   creado de forma perezosa y reutilizado, 0,007 ms. Test: mismas estimaciones con y sin límite y
   que dentro del ajuste `threadpool_info()` muestre 1 hilo BLAS (por ejemplo con un mock de
   `minimize`).
2. En `select_n_jobs_backtesting`, para `ForecasterStats` devolver `max(1, cpu_count() - 1)` si
   `refit` es `True` o 1, y 1 en otro caso; `ForecasterFoundation` se queda en 1. Actualizar el
   docstring y `test_select_n_jobs_backtesting.py:56` (hoy espera 1). Nota de versión.
3. Volver a medir tras BUG-08 (con `freeze_params` funcionando el coste por fold baja en Ets). Si
   el arranque de workers pesa demasiado, una condición sencilla sería mantener 1 cuando el número
   de folds con reajuste sea pequeño.
