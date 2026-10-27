# Lente STATS: salud y rendimiento de ForecasterStats (0.27.0)

Alcance: flujos habituales de `ForecasterStats` con `Arima` (fijo y automático), `Ets` (fijo y automático) y `Sarimax` (statsmodels): fit, predict, predict_interval, exog, transformer_y, `backtesting_stats` y `grid_search_stats`. Solo correcciones y optimizaciones, sin funcionalidad nueva.

Entorno de medida: Python 3.12.3, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1, scipy 1.18.1 (OpenBLAS 0.3.34), numba 0.68.0, statsmodels 0.15.0; máquina de 4 núcleos compartida con otros agentes (medianas de 2 a 5 repeticiones; la compilación numba se mide aparte con `NUMBA_CACHE_DIR` vacío). Rama 0.26.x (0.26.0).

Scripts: `/tmp/claude-0/-home-user-skforecast/8505cf39-716d-5c4c-adf6-669c7ec65b16/scratchpad/stats/` (`bench.py`, `bench_bt.py`, `repro_freeze.py`, `freeze_after.py`, `threads_ab.py`, `conc.py`, `pool.py`, `repro_overlap.py`, `repro_sarimax_norefit.py`, `cold.py`, `compile_breakdown.py`).

Ya identificado y no repetido aquí: BUG-stats-interval-ignored (alpha=0.05 anula interval), DEP-stats-alpha, PR #1400 (compilación numba de Ets, solo `_ets_nelder_mead`), HANDOFF-STATS-LW, DEP-arima-dictstyle.

## Candidatos (ordenados por valor)

### 1. BUG: `freeze_params=True` (valor por defecto) no tiene ningún efecto en `backtesting_stats` ni en las búsquedas (S)

`_backtesting_stats` congela los parámetros llamando a `estimator._set_params(**best_params_)` sobre `forecaster.estimators_` (skforecast/model_selection/_validation.py:2097-2100), pero cada fold reentrena con `forecaster.fit` (_validation.py:1847), que vuelve a clonar las plantillas sin ajustar `self.estimators` (skforecast/recursive/_forecaster_stats.py:521). La búsqueda automática se repite en todos los folds.

Reproducción (`repro_freeze.py`, 1000 puntos diarios, m=7, 20 folds, refit=True, n_jobs=1, contando llamadas a `auto_arima`/`auto_ets`):

| Estimador | freeze_params | llamadas auto | tiempo | MAE |
|---|---|---|---|---|
| Arima automático | True | 20 | 20,6 s | 1,9137 |
| Arima automático | False | 20 | 19,9 s | 1,9137 |
| Ets automático | True | 20 | 17,6 s | 1,8888 |
| Ets automático | False | 20 | 17,0 s | 1,8888 |

Con la congelación funcionando (simulada en `freeze_after.py`: ajuste automático una vez y backtesting con los parámetros elegidos): Arima 4,4 s (4,7 veces menos), Ets 1,8 s (10 veces menos). Las dos opciones dan hoy el mismo MAE: el usuario paga el coste de `freeze_params=False` creyendo que usa el modo rápido documentado. `grid_search_stats` y `random_search_stats` fuerzan `freeze_params=True` (skforecast/model_selection/_search.py:2779), así que también se ven afectadas cuando la rejilla deja `order=None`.

Arreglo: tras congelar, propagar los parámetros también a las plantillas de la copia del forecaster (`forecaster.estimators[i]._set_params(**forecaster.estimators_[i].best_params_)`), o no reclonar en el fit de cada fold. Test: contar llamadas a `auto_arima` con freeze True (1) y False (n folds).

### 2. RENDIMIENTO: sobresuscripción de hilos BLAS en el ajuste de Ets y Sarimax (S)

El optimizador L-BFGS-B de scipy (usado por `_optimize_ets`, skforecast/stats/exponential_smoothing/_ets_base.py:1052-1059, y por statsmodels en `Sarimax.fit`, skforecast/stats/_sarimax.py:434) llama a OpenBLAS con vectores minúsculos; en cuanto hay más hilos BLAS que núcleos libres, cada llamada a `setulb` pasa de microsegundos a milisegundos. Perfil de `Ets(m=7).fit` con n=3000 y carga en la máquina: 15,66 s de 19,9 s dentro de `setulb`; con `OPENBLAS_NUM_THREADS=1`, 0,033 s.

Medidas (n=1000, m=7):

- Máquina en reposo: sin diferencia (Ets automático 0,98 s frente a 1,02 s; Sarimax 0,72 s frente a 0,63 s).
- Dos procesos a la vez (dos notebooks o dos scripts): Ets automático 12,7 s por ajuste frente a 1,0 s con `threadpool_limits(1)`; Sarimax 11,2 s frente a 0,7 s.
- Patrón habitual de "un modelo estadístico por serie" con 8 series: secuencial 8,7 s; `joblib` n_jobs=4: 4,5 s (joblib limita los hilos de sus workers); `multiprocessing.Pool(4)`: 50,5 s.
- Arima nativo (BFGS) no se ve afectado.

Arreglo: envolver el ajuste en `threadpoolctl.threadpool_limits(1, user_api="blas")` dentro de `Ets.fit` y `Sarimax.fit` (y opcionalmente `Arima.fit`, por si el usuario elige `optim_method="L-BFGS-B"`). `threadpoolctl` ya es dependencia transitiva vía scikit-learn y el contexto cuesta 0,1 ms. Medido con scipy 1.18.1; no comprobado con scipy<1.15 (L-BFGS-B en Fortran).

### 3. QUICK WIN: `n_jobs='auto'` paraleliza el backtesting con refit de ForecasterStats (S)

`select_n_jobs_backtesting` devuelve siempre 1 para `ForecasterStats` (skforecast/model_selection/_utils.py:706), aunque con `refit=True` cada fold es un ajuste independiente y monohilo. Medido (1000 puntos, 30 folds, refit=True, mismo MAE):

| Estimador | n_jobs=1 | n_jobs=4 |
|---|---|---|
| Arima(2,1,1)(1,0,1)[7] | 6,4 s | 3,2 s |
| Ets automático (freeze_params=False) | 24,7 s | 8,0 s |
| Sarimax(2,1,1)(1,0,1,7) | 18,8 s | 6,0 s |

Antes:

```python
metric, preds = backtesting_stats(forecaster, y=y, cv=cv, metric="mean_absolute_error")
# 18,8 s con Sarimax y 30 folds; el usuario no sabe que puede pasar n_jobs=-1
```

Después (mismo código, `n_jobs='auto'` elige `cpu_count() - 1` cuando `refit` es True o 1):

```python
metric, preds = backtesting_stats(forecaster, y=y, cv=cv, metric="mean_absolute_error")
# 6,0 s
```

Mantener 1 con `refit=False` o refit intermitente (ya se fuerza así). Los workers de joblib limitan los hilos BLAS, lo que además evita el problema del punto 2.

### 4. BUG: varios estimadores y `fold_stride < steps` lanza ValueError en `backtesting_stats` (S)

Con salida larga (varias filas por fecha, una por estimador), el filtro de solapamiento `~index.duplicated(keep='last')` (_validation.py:2160-2164) deja solo la última fila de cada fecha, es decir, solo un estimador. Después `metric_values.insert(0, 'estimator_id', ...)` falla (_validation.py:2195). Reproducido (`repro_overlap.py`, Arima + Ets, steps=7, fold_stride=2):

```
ValueError: Length of values (2) does not match length of index (1)
```

Con un único estimador funciona. Arreglo: deduplicar por (índice, estimator_id), igual que el multiserie usa (idx, level).

### 5. BUG: Sarimax con `refit=False` y `fold_stride != steps` falla (S)

`_fit_predict_forecaster_stats` construye `last_window` con `last_window_iloc_end - steps` (_validation.py:1830); con `fold_stride` distinto de `steps` la ventana no empieza donde terminó la anterior y `append` lo rechaza. Reproducido (`repro_sarimax_norefit.py`): fold_stride=3 y 10 dan `ValueError: ... last_window has to start at the end of the index seen by the forecaster`; fold_stride=None funciona. El arreglo (usar el inicio de test del fold anterior, o `fold_stride`) se validó parcheando la función: diferencia máxima 0,0 frente a un bucle manual con `predict(last_window=...)` para strides 3, 7 y 10. Los tests actuales con `fold_stride` solo cubren `refit=True`.

### 6. RENDIMIENTO: compilación numba de Arima en el primer uso (M; amplía BR-perf-ets-numba-compile / PR #1400)

Con caché vacía: primer `fit` de Arima 11,4 a 12,0 s más 1,2 a 1,3 s del primer `predict_interval` (segunda vez: 0,03 a 0,05 s). Ets: 6,2 a 6,7 s, ya atacado por #1400. Desglose (`compile_breakdown.py`, tiempos anidados): `transform_arima_parameters` 4,0 s (incluye `transform_unconstrained_to_ar_params` 2,2 s), `_compute_q0_njit` 3,5 s (incluye `_solve_discrete_lyapunov` 1,2 s), `_arima_kalman_core` 1,7 s, `kalman_forecast_core` 1,0 s; "impl" (sobrecargas de numpy dentro de numba) 6,6 s. Se paga en cada instalación o actualización, en CI, contenedores y entornos sin caché escribible. Aplicar el mismo enfoque que #1400 (bucles explícitos en lugar de expresiones de arrays y `@`) a esas funciones.

## Medidas de referencia (sin bug, máquina en reposo salvo indicación)

Ajuste (mediana de 3, caché numba caliente):

| Modelo | n | fit | predict | predict_interval |
|---|---|---|---|---|
| Arima(2,1,1)(1,0,1)[7] | 1000 / 3000 | 0,31 / 0,37 s | <1 ms | ~1 ms |
| Arima automático m=7 | 1000 / 3000 | 1,75 / 0,40 s (aproximación CSS a partir de n grande) | <1 ms | ~1 ms |
| Ets(AAdA) m=7 | 1000 / 3000 | 0,11 / 0,55 s | <1 ms | ~1 ms |
| Ets automático m=7 | 1000 / 3000 | 0,96 / 2,60 s | <1 ms | ~1 ms (2 ms en modelos multiplicativos, simulación) |
| Sarimax(2,1,1)(1,0,1,7) | 1000 / 3000 | 0,76 / 1,22 s | ~2 ms | ~3 ms |
| Arima(2,0,1)(1,1,1)[24] | 2000 / 5000 | 4,1 / 11,3 s | ~1 ms | ~1 ms |
| Arima automático m=24 | 2000 / 5000 | 1,2 / 4,4 s | ~3 ms | ~3 ms |
| Ets automático m=24 | 2000 / 5000 | 2,5 / 5,8 s | <1 ms | <1 ms |

Backtesting (1000 puntos, 30 folds, n_jobs=1): Arima fijo 6,4 a 7,2 s; Arima automático 29 a 31 s (por el bug 1); Ets automático 25 s; Sarimax refit=True 19 s; Sarimax refit=False 1,0 s (3000 puntos y 50 folds: 3,3 s, el `append` no es un cuello de botella).

## Fricciones observadas (no candidatas)

- Arima(2,0,1)(1,1,1)[24] con 5000 puntos tarda 10,6 s: 729 evaluaciones de la verosimilitud de 14 ms cada una (estado de dimensión 26), el 80 % dentro de gradientes numéricos (`approx_derivative`). Un filtro de Kalman con detección de estado estacionario o gradiente analítico podría reducirlo, pero es esfuerzo L y cambia ligeramente los resultados.
- `Arima(order=None, seasonal_order=None)` con exog emite 35 `RuntimeWarning: invalid value encountered in subtract` (scipy `_numdiff`) y un `UserWarning: CSS optimization convergence issue` procedentes de modelos candidatos de la búsqueda, aunque el modelo final está bien. R los silencia durante la búsqueda. Ruido, no error.
- `Sarimax` sin `transformer_y` sobre una serie con nivel ~50 y sin constante da un MAE distinto (1,79) que con `StandardScaler` (2,04); con Arima el resultado es idéntico con y sin escalado. Es sensibilidad del optimizador de statsmodels, no un bug de skforecast.
- `Ets.predict_interval` nombra las columnas con `int(level*100)`: `interval=[0.05, 0.95]` produce internamente el nivel 89,999 y la clave `lower_89`. Es coherente dentro de Ets (no falla), pero dos niveles muy próximos colisionarían. Caso marginal.
- Lo que funciona bien: exog con Arima y Sarimax, `transformer_y` con intervalos, aviso `IgnoredArgumentWarning` cuando Ets recibe exog, `grid_search_stats` con exog, intervalos de Ets simulados reproducibles (semilla fija), caché numba efectiva entre procesos.
