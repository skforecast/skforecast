# Triaje de candidatos de descubrimiento (roadmap 0.27.0)

Fecha: 2026-10-18. Rama de referencia: `0.26.x` (`__version__ = "0.26.0"`).

Entrada: 39 candidatos de 7 lentes (ml-single, ml-multi, outputs-metrics, foundation,
consistency, competitors, stats). Salida: 23 elementos fusionados para verificar
(26 huecos de verificación) y 10 descartados o aplazados.

Criterios aplicados:

- Fusión de duplicados entre lentes (3 pares exactos) y agrupación de cambios muy
  pequeños del mismo tipo para ahorrar huecos (mensajes de error, paralelismo en stats,
  fallos de `fold_stride` en `backtesting_stats`, aviso de cobertura dentro de la
  función de evaluación).
- Ningún candidato coincide exactamente con lo ya identificado; los que lo amplían
  lo indican en la columna "Amplía".
- Reclasificación: los "quick win" de esfuerzo M que añaden una capacidad nueva pasan
  a feature (función de evaluación, baseline multiserie, espacios de búsqueda).
- Coste de verificación: quick win, bug, perf y feature S = 1 hueco; feature M o L = 2.
- Comprobaciones puntuales hechas en este triaje (grep sobre `0.26.x`):
  `_search.py:514` ordena por tipo de tarea y no por métrica; `_search.py:512-524`
  concatena antes de comprobar si está vacío; `_validation.py:662-667` deduplica con
  `keep='last'`; `_validation.py:513-515` sobrescribe `cv.differentiation` justo después
  del error de `_utils.py:270-288`; `utils.py:4680` solo avisa si `levels` es lista;
  `_forecaster_foundation.py:155` clona el estimador; `_forecaster_foundation.py:946-960`
  construye `pred` con `q_0.5`; no hay `__getstate__` en `skforecast/foundation/`;
  `_validation.py:2097-2100` congela `estimators_`; `_utils.py:706` fija `n_jobs = 1`
  para stats; `metrics.py:52` solo admite la variante escalada de RMSE;
  `_arima_base.py:578` usa `njit(cache=True)`.

## Elementos para verificar (ordenados por prioridad)

| Rango | ID | Tipo | Área | Título | Esf. | Huecos | Lentes | Amplía |
|---|---|---|---|---|---|---|---|---|
| 1 | BUG-01 | bug | model_selection | Avisar de evaluación in-sample cuando `initial_train_size=None` y el forecaster ya está ajustado | S | 1 | ml-single | |
| 2 | BUG-02 | bug | model_selection | Con `fold_stride < steps` la métrica solo mide el horizonte más corto y las búsquedas seleccionan por el error a 1 paso | S | 1 | outputs-metrics | DISC-1119 |
| 3 | BUG-03 | bug | model_selection | Las búsquedas eligen el peor modelo con métricas donde mayor es mejor (`r2_score`) | S | 1 | outputs-metrics | |
| 4 | BUG-04 | bug | foundation | Un pipeline o modelo precargado (Chronos-2 ajustado) se descarta en silencio al crear `ForecasterFoundation` | S | 1 | foundation | PLAN-CKPT |
| 5 | QW-01 | quick_win | recursive (multiseries) | `predict()` con `levels=None` excluye sin aviso las series cuya última ventana termina antes | S | 1 | ml-multi | ISSUE-1406 / PR-1407 (misma familia, en predict) |
| 6 | QW-02 | quick_win | foundation | Avisar cuando se ajustó con exog y `predict` no la recibe (Chronos-2, TimesFM 3, TS-ICL) | S | 1 | foundation, consistency | |
| 7 | QW-03 | quick_win | model_selection | Heredar `differentiation` del forecaster en `TimeSeriesFold` y `OneStepAheadFold` | S | 1 | ml-single, ml-multi | |
| 8 | BUG-05 | bug | model_selection | `grid_search`/`random_search` devuelven un DataFrame vacío y el forecaster sin ajustar cuando fallan todas las combinaciones | S | 1 | ml-single, ml-multi | |
| 9 | QW-04 | quick_win | model_selection | Columna `y_true` en las predicciones de backtesting | S | 1 | ml-multi | GAP-backtesting-interval-metrics, DISC-1119, ISSUE-1409 |
| 10 | BUG-06 | bug | foundation | `pred` distinto según el método: `predict()` da la media y `predict_interval()` / backtesting con cuantiles la mediana | M | 1 | consistency | GAP-foundation-backtesting-interval |
| 11 | QW-05 | quick_win | foundation / utils | `save_forecaster` no debe guardar los pesos del modelo foundation cargados tras un `predict` | S | 1 | foundation | |
| 12 | BUG-07 | bug | recursive (multiseries) / utils | El formato largo con MultiIndex documentado falla con un MultiIndex creado con `set_index` | S | 1 | ml-multi | |
| 13 | BUG-08 | bug | model_selection (stats) | `freeze_params=True` (por defecto) no congela nada en `backtesting_stats` | S | 1 | stats | |
| 14 | QW-06 | quick_win | recursive / direct | Alinear `exog` por índice en `predict` de `ForecasterRecursive` y `ForecasterDirect` | S | 1 | ml-single | |
| 15 | QW-07 | quick_win | utils / model_selection | Mensajes de error accionables: frecuencia inferida, intervalos en percentiles, residuos ausentes en multiserie, función equivocada para el tipo de forecaster | S | 1 | ml-single, ml-multi, consistency | DOC-skills-percentiles-stale, GAP-foundation-backtesting-interval |
| 16 | QW-08 | quick_win | metrics | Aceptar `'root_mean_squared_error'` como string y nombres únicos en `create_mean_pinball_loss` | S | 1 | outputs-metrics | |
| 17 | FE-01 | feature | model_selection / metrics | Función pública de evaluación de predicciones de backtesting por fold, paso, serie y grupos propios, con cobertura, anchura y aviso de infracobertura | M | 2 | outputs-metrics, ml-single | GAP-backtesting-interval-metrics, DISC-1119, ISSUE-1409 |
| 18 | PERF-01 | perf | foundation / model_selection | Agrupar folds en una llamada al modelo y reducir la sobrecarga pandas en `backtesting_foundation` | M | 1 | foundation | |
| 19 | PERF-02 | perf | stats / model_selection (stats) | Paralelismo en stats: `n_jobs='auto'` con `refit=True` y límite de hilos BLAS en el ajuste de Ets y Sarimax | S | 1 | stats | |
| 20 | FE-02 | feature | foundation / model_selection | Covariables solo pasadas en `backtesting_foundation` (`past_only_exog`) | S | 1 | foundation | ISSUE-1131 |
| 21 | FE-03 | feature | preprocessing | Ventanas estacionales en `RollingFeatures` (`season_length`) | M | 2 | competitors | DISC-1153 (relacionado) |
| 22 | FE-04 | feature | model_selection | `ForecasterEquivalentDate` en `backtesting_forecaster_multiseries` | M | 2 | competitors | |
| 23 | BUG-09 | bug | model_selection (stats) | `backtesting_stats` falla con `fold_stride != steps` en dos casos: varios estimadores y Sarimax con `refit=False` | S | 1 | stats | |

Total: 26 huecos.

### Notas para la verificación

- **BUG-01**: medido MAE 0,0036 frente a 0,0826 fuera de muestra (h2o, LGBM lags=12),
  sin ningún aviso. El modo in-sample es legítimo (`backtesting.ipynb` lo usa), así que
  la propuesta es un aviso, no un error. Comprobar también multiserie
  (`_validation.py:1420`) y que el aviso no salte en el uso documentado (o que se pueda
  silenciar con `suppress_warnings`).
- **BUG-02**: medido con bike_sharing: `fold_stride=1` devuelve 32,5 (MAE a 1 paso)
  frente a 63,2 en el horizonte de 24 pasos; `grid_search` cambia el modelo elegido.
  Decidir entre aviso y opción `'all'`; la deduplicación aparece en 5 sitios de
  `_validation.py` (también stats y foundation). Coordinar con BUG-09.
- **BUG-03**: medido `alpha=100` (R2 -0,37) elegido frente a `alpha=0,01` (R2 0,84) y
  reentrenado con `return_best=True`. Comprobar si basta con deducir la dirección de un
  conjunto de nombres conocidos o hace falta un parámetro.
- **BUG-04**: reproducido con FakePipeline: el forecaster predice con el modelo base.
  El fixture de tests ya reconoce que el pipeline no sobrevive al `clone`. Valorar a la
  vez QW-05 (si se conserva la referencia, `__getstate__` debe respetarla).
- **QW-01**: 47 de 50 series predichas en store_sales sin avisos. Ver si conviene un
  aviso nuevo o reutilizar el de `levels` lista.
- **QW-02**: dos lentes con la misma reproducción. Decidir la categoría
  (`MissingExogWarning`, ya existe, o `IgnoredArgumentWarning`) y si añadir
  `suppress_warnings` a `predict*` de `ForecasterFoundation`.
- **QW-03**: `llms-base.txt:340,370` ya dice "set automatically by forecaster" y el
  código generado con esa referencia falla. Mantener el error solo si el usuario fija un
  valor explícito distinto. Revisar el camino `OneStepAheadFold` (`_utils.py:602`) y
  quitar los comentarios "must match" de guías y skills.
- **BUG-05**: aplicar en las 4 funciones (`_search.py:474`, 1481, 2786, 3040). Si
  QW-03 se arregla, el disparador más típico desaparece, pero quedan exog mal alineada,
  parámetros inexistentes o `initial_train_size` mayor que los datos (`s22.py`).
- **QW-04**: comprobar coste en memoria en paneles grandes y si hace falta un flag para
  no romper código que indexa columnas por posición.
- **BUG-06**: afecta a TabICL, Nori, TabPFN con `point_estimate != 'median'` y TimesFM
  2.5. Medido MAE 1,1693 sin cuantiles frente a 1,0834 con cuantiles en la misma
  configuración. Debe resolverse antes o junto a GAP-foundation-backtesting-interval.
- **QW-05**: medido 0,01 MB antes de predecir y 111,81 MB después (chronos-2-small).
- **BUG-07**: con store_sales, `set_index` deja `freq=None` y `inferred_freq='D'`; el
  mensaje habla de diccionarios. Valorar inferir por grupo o error específico.
- **BUG-08**: Arima auto 20,6 s frente a 4,4 s con la congelación funcionando; Ets 17,6 s
  frente a 1,8 s; mismo MAE en ambos modos hoy. Las búsquedas fuerzan
  `freeze_params=True` (`_search.py:2779`), así que también se benefician.
- **QW-06**: `ForecasterRecursiveMultiSeries` ya alinea por índice (`utils.py:1656`).
  Comprobar que no oculta desalineaciones reales (índice sin la fecha esperada, duplicados).
- **QW-07**: cuatro mensajes independientes y triviales; verificar cada uno por separado
  (`utils.py:2046-2050`, `utils.py:1222-1229`, `utils.py:1938-1947`,
  `_validation.py:869,1698,2321`, `_search.py:2946`).
- **QW-08**: sklearn >= 1.6 trae `root_mean_squared_error`. Las dos curvas pinball con el
  mismo nombre rompen las búsquedas con varios cuantiles. WAPE queda opcional.
- **FE-01**: decidir el diseño junto con ISSUE-1409 y DISC-1119 (integrar en la tupla de
  backtesting o función aparte). Incluye el aviso de infracobertura con residuos
  in-sample (cobertura 0,22 a 0,53 frente al 80% nominal con LightGBM).
- **PERF-01**: 5,92 s frente a 1,64 s (3,6x) con una serie y 200 folds, mismas
  predicciones. Comprobar que agrupar folds no altera adaptadores con aprendizaje
  cruzado (`cross_learning`).
- **PERF-02**: `n_jobs` 1 -> 4 con `refit=True`: Ets auto 24,7 s -> 8,0 s, Sarimax
  18,8 s -> 6,0 s. El límite BLAS (`threadpool_limits(1)`) evita el caso 12,7 s frente a
  1,0 s con dos procesos simultáneos. `threadpoolctl` ya llega vía scikit-learn.
- **FE-02**: `predict` ya admite covariables solo pasadas; falta en el backtesting. Sin
  ello, el único backtest honesto es un bucle manual.
- **FE-03**: medido -4,1% de MAE en bike_sharing con medias estacionales 24x7 y 168x4.
  Las clases propias de window features no se guardan con `save_forecaster`.
- **FE-04**: la guía de comparación de modelos hace el bucle por serie a mano.
- **BUG-09**: dos fallos S con arreglo validado (diferencia 0,0 frente a bucle manual en
  el caso de Sarimax). Nicho; agrupar en una sola PR con tests.

## Descartados y aplazados

| Candidato | Lente | Destino | Motivo |
|---|---|---|---|
| grid_search y random_search devuelven un DataFrame vacío en silencio | ml-multi | duplicado | Igual que el candidato de ml-single; fusionado en BUG-05 con su evidencia multiserie. |
| Propagar differentiation del forecaster a TimeSeriesFold | ml-multi | duplicado | Igual que el candidato de ml-single; fusionado en QW-03 (añade el caso dict por serie). |
| Avisar cuando el forecaster foundation se ajustó con exog y predict no la recibe | consistency | duplicado | Igual que el candidato de foundation; fusionado en QW-02. |
| Agregar los avisos por serie en paneles grandes | ml-multi | aplazado | Esfuerzo M; molestia real pero los avisos responden a errores del usuario (exog corta). Abordar junto a HANDOFF-DOUBLE-WARN y HANDOFF-T4. |
| Columna de tiempo por combinación en grid y random search | outputs-metrics | aplazado | Valor bajo y de nicho; bayesian_search ya lo da vía `study.trials_dataframe()`. |
| Calibración conformal de los cuantiles nativos foundation | foundation | aplazado | Esfuerzo M y diseño abierto; primero medir cobertura con ISSUE-1409 / FE-01. |
| Espacios de búsqueda por defecto (`search_space='auto'`) | competitors | aplazado | Esfuerzo M, comodidad; rangos opinables por estimador y mantenimiento continuo. |
| Variables estáticas por serie (`static_features`) | competitors | aplazado | Esfuerzo M a L (fit, predict, backtesting, búsquedas, selección); hay alternativa con exog en dict. |
| Pesos de decaimiento temporal integrados para `weight_func` | competitors | aplazado | Nicho; la ganancia depende de la semivida (+3,2% de MAE con 90 días) y la alternativa son 3 líneas. |
| Reducir la compilación numba de Arima en el primer uso | stats | aplazado | `njit(cache=True)` (`_arima_base.py:578`): se paga una vez por entorno. Seguimiento natural de PR #1400 cuando se fusione. |
