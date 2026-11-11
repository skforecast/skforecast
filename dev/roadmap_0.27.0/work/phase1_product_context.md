# Fase 1: contexto de producto para la 0.27.0

Fecha: 2026-10-17. Base: rama `0.26.x` en `708a8b502`, `__version__ = "0.26.0"`.

Nota de partida: en `docs/releases/releases.md:15` la 0.26.0 figura ya con fecha (`Oct 8, 2026`), no como "In development". La sección `## 0.27.0 <small>In development</small>` todavía no existe en `0.26.x`: la crean por separado dos PR abiertos, #1400 (`perf/ets-numba-compile`) y #1402 (`fix/fetch-dataset-website-visits-m5`). El segundo que se fusione tendrá un conflicto en las notas de versión.

## 1. Resumen por versión

### 0.23.0 (2026-07-08)

- `calendar_features` (`CalendarFeatures`) en los 4 forecasters ML; refactor de codificaciones (`cyclical`, `onehot`, `spline`).
- Adaptadores TabPFN-TS y TFC-T0 (5 adaptadores).
- `acf`, `pacf` y `calculate_lag_autocorrelation` propios (sin statsmodels).
- `save_forecaster`/`load_forecaster` con `backend` (`joblib`, `pickle`, `cloudpickle`, `skops`) y `trusted`.
- API: `interval` pasa a cuantiles 0-1 (percentiles con `FutureWarning`); `level` de Arima/Arar/Ets también.
- `select_features*` devuelven un cuarto elemento, `selected_calendar_features`.
- `calculate_distance_from_holiday` sale de `experimental` y pasa a `preprocessing`.
- Incompatibilidad de modelos serializados (hay que reentrenar).

### 0.24.0 (2026-08-24)

- `bayesian_search_foundation` (ajuste de `context_length` y similares).
- `grid_search_equivalent_date` para el baseline.
- Adaptadores Nori y TS-ICL.
- Métricas `winkler_score` y `weighted_interval_score`.
- `ForecasterRecursiveMultiSeries` conserva niveles con NaN en backtesting si el estimador admite NaN de forma nativa (#1260).
- Bootstrapping con `order='F'` (más rápido con CatBoost).

### 0.25.0 (2026-09-11)

- Foundation con series heterogéneas (longitudes distintas, exog distinta por serie, NaN en el objetivo); atributos `supports_heterogeneous_covariates`, `supports_nan_in_series`, `supports_past_only_covariates`, `allow_exog`.
- Adaptador TimesFM 3.0; `TimesFMAdapter` renombrado a `TimesFM25Adapter`.
- `LicenseWarning` para pesos con licencia no comercial.
- Eliminado el soporte de percentiles en `interval` (forecasters y backtesting) y en `level` (Arima, Arar, Ets): ahora `ValueError`.
- Correcciones en `backtesting_foundation`, `QuantileBinner`, residuos binned multiserie y `crps_from_quantiles` (integración analítica).

### 0.26.0 (2026-10-08)

- `get_model_info` y `list_adapters` (capacidades de los modelos foundation sin instalar el backend).
- Skills instalables en agentes (plugin de Claude Code, `npx skills add`, Context7).
- Rendimiento: `ForecasterRecursiveMultiSeries` (`predict` x2,3 con 500 series, x4,9 con 5000), ExtraTrees x15 en predicción, Arima 1,4 a 2,2 veces más rápido, `approximation` en auto ARIMA (x5), numba cargado de forma perezosa.
- Correcciones grandes de exactitud en STATS: Ets no estimaba los parámetros de suavizado; Arima con 2 o más regresores daba coeficientes erróneos; Box-Cox, drift, `get_params` incompletos, BIC/AICc.
- Correcciones ML: `encoding='onehot'` con series no ordenadas alfabéticamente (predicciones erróneas desde la 0.19), exog ancha multiserie usada por posición, XGBoost con early stopping, `ForecasterDirect` con `differentiation` y `steps` no consecutivos, zonas horarias con cambio de hora.
- `ResidualsUsageWarning` cuando hay menos de 10 residuos por bin de media.
- Mínimos: pandas 2.2, scikit-learn 1.6, numpy 1.26.1, statsmodels 0.13.2, keras 3.3. Eliminado `cast_exog_dtypes`.
- Docs: nueva portada, guía de migración desde Prophet, guía de comparación de modelos, overview de intervalos con sección de evaluación, folds por mes natural en backtesting.

## 2. Trabajo a medias o anunciado

| Id | Qué | Evidencia | Estado |
|---|---|---|---|
| PR-1400 | Ets compila numba un 40% más rápido (7 s frente a 12 s) | PR #1400, rama `perf/ets-numba-compile` | Abierto contra `0.26.x`, ya etiquetado 0.27.0 |
| PR-1402 | `fetch_dataset` con `website_visits` y `m5` sin `DatetimeIndex`; menos memoria en m5 | PR #1402 | Abierto, etiquetado 0.27.0 |
| PR-1407 | Aviso cuando una serie no se predice en ningún fold de `backtesting_forecaster_multiseries` | PR #1407 | Abierto |
| PR-1408 | `KeyError` en `ForecasterRecursiveMultiSeries` con nombres de series enteros | PR #1408 | Abierto |
| PR-1389 | `set_in_sample_residuals` con CatBoost y exog categórica (issue #1354) | PR #1389 | Abierto |
| PR-1388 | Búsqueda de hiperparámetros con `lags=None` y solo window features (issue #1355) | PR #1388 | Abierto |
| PR-1387 | `plot_residuals` ignora `fig` | PR #1387 | Abierto |
| PR-1341 | Adaptador TiRex-2 (sustituye a #1340) | PR #1341, actualizado 2026-10-10 | Abierto; no hay rastro de TiRex en `skforecast/` |
| PLAN-TTM | Adaptador IBM TinyTimeMixer en tres fases (cuantiles r3, exog con checkpoint ajustado, helper de fine-tuning) | `dev/PLAN_TinyTimeMixerAdapter.md` | Plan escrito, sin código |
| PLAN-CKPT | Parámetro común `finetuned_checkpoint` en todos los adaptadores | `dev/PLAN_UnifiedFinetunedCheckpoint.md` | Plan escrito, sin código |
| HANDOFF-T2 | Las categóricas nativas no llegan al estimador dentro de `CalibratedClassifierCV` o `TransformedTargetRegressor` | `dev/handoff_utils_pendiente.md`, tarea T2 | Pendiente, "para la 0.27" |
| HANDOFF-T3 | `Pipeline` con LightGBM o CatBoost y exog categórica: `fit` falla | mismo fichero, T3 | Pendiente de decisión de Joaquín |
| HANDOFF-T4 | Se pierde la configuración categórica del usuario y salen avisos de más | mismo fichero, T4 | Pendiente de decisión de Joaquín |
| HANDOFF-STATS-LW | `ForecasterStats.fit(store_last_window=False)` y luego `predict(last_window=...)` da `AttributeError` | mismo fichero, sección 5 | Sin decidir |
| HANDOFF-DOUBLE-WARN | Doble aviso de NaN en la exog de `predict` en `ForecasterRecursive`/`ForecasterDirect` | mismo fichero, sección 5 | Sin decidir |
| HANDOFF-SAVE-PATH | `save_forecaster` podría devolver la ruta final | mismo fichero, sección 5 | Propuesta sin decidir |

## 3. Deprecaciones programadas

No hay ninguna deprecación con versión de retirada 0.27 explícita. `grep -rniE "deprecat|will be removed|0\.27"` en `skforecast/` (sin `deep_learning` ni tests) solo devuelve:

1. `ArimaResult.__getitem__` con las claves `'arma'` y `'model'` emite `FutureWarning` ("deprecated, use result.order / result.state_space") sin versión de retirada (`skforecast/stats/arima/_arima_base.py:271-290`). Lleva así desde la 0.22. Candidato a retirar en la 0.27 o, al menos, a fijar la versión en el mensaje.
2. El decorador `runtime_deprecated` existe (`skforecast/exceptions/exceptions.py:21`) pero no se usa en ninguna parte del paquete.

Deprecaciones de facto que conviene programar en la 0.27:

- **Python 3.10**: fin de vida en octubre de 2026 (PEP 619). Sigue en `requires-python = ">=3.10"` (`pyproject.toml:65`), en los clasificadores (`pyproject.toml:26`) y es la versión del job de dependencias mínimas (`.github/workflows/unit-tests-min-deps.yml:30`). Precedente: 3.9 se dejó en la 0.19.0. Como mínimo, anunciarlo en la 0.27 y retirarlo en la 0.28.
- **`alpha` en `ForecasterStats.predict_interval` y `backtesting_stats`**: duplica a `interval` y además provoca un resultado silencioso erróneo (ver la sección 6). Es la única API de intervalos que no sigue la convención de cuantiles de la 0.23.

Retiradas ya completadas (no son trabajo): percentiles en `interval`/`level` (0.25), `regressor` (0.22), `ForecasterSarimax` (0.20), `cast_exog_dtypes` (0.26).

## 4. TODOs del código

Solo 9 `TODO` en `skforecast/` (sin `deep_learning`), ninguno con fecha:

| Fichero:línea | TODO | Valor |
|---|---|---|
| `recursive/_forecaster_recursive.py:1289`, `recursive/_forecaster_recursive_multiseries.py:2345` | Crear `reset_forecaster()` para reiniciar atributos | Refactor interno; bajo |
| `recursive/_forecaster_stats.py:219` | Valorar un parámetro `aggregate` que combine las predicciones de varios estimadores | Sería una funcionalidad nueva en STATS (fuera de foco salvo que sea trivial) |
| `recursive/_forecaster_stats.py:626` | Revisar cuando haya varias series | Fuera de foco |
| `preprocessing/_preprocessing.py:2015`, `:2219` | `RollingFeaturesClassification`: `fillna` sin uso y clases por columna | Preparación de un futuro `ForecasterRecursiveMultiSeriesClassifier` |
| `model_selection/_validation.py:1099` | Formato largo con varios niveles en multivariante y `gap > 0` | Revisar si es un caso real |
| `model_selection/_search.py:1320`, `:1722` | Validación de longitud de exog solo para DataFrame ancho | Limpieza pendiente |

## 5. Matriz de capacidades

Verificada con la firma de cada clase y las listas de `skforecast/model_selection/_utils.py:200-235` y `:555-566`. S = sí, N = no, "nat." = nativo del modelo.

| Capacidad | Recursive | Direct | RecursiveMultiSeries | DirectMultiVariate | Classifier | Stats | EquivalentDate | Foundation |
|---|---|---|---|---|---|---|---|---|
| `lags` | S | S | S | S (dict por serie) | S | N | N (`offset`) | N (`context_length`) |
| `window_features` | S | S | S | S | S (`RollingFeaturesClassification`) | N | N | N |
| `calendar_features` | S | S | S | S | N | N | N | N |
| exog | S | S | S | S | S | Arima, Sarimax, Arar | N | Según adaptador (todos salvo TimesFM 2.5 y Moirai-2) |
| `differentiation` | S | S | S (por serie) | S (por serie) | N | dentro del modelo | N | N |
| `transformer_y`/`series` | S | S | S (por serie) | S (por serie) | N | S | N | N |
| `weight_func` | S | S | S (+ `series_weights`) | S | S | N | N | N |
| Bootstrapping | S | S | S | S | N | N | N | N |
| Conformal | S | S | S | S | N | N | S | N |
| Residuos binned | S | S | S | S | N | N | S | N |
| `predict_quantiles` | S | S | S | S | N | N | N | S (nat.) |
| `predict_dist` | S | S | S | S | N | N | N | N |
| `predict_interval` | S | S | S | S | N (`predict_proba`) | S (paramétrico, `alpha`/`interval`) | S (conformal) | S (nat.) |
| Backtesting | `backtesting_forecaster` | ídem | `backtesting_forecaster_multiseries` | ídem | `backtesting_forecaster` | `backtesting_stats` | `backtesting_forecaster` | `backtesting_foundation` (solo `quantiles`, sin `interval`) |
| `return_predictors` | S | S | S | S | S | N | N | N |
| `OneStepAheadFold` | S | S | S | S | S | N | N | N |
| Búsquedas | grid, random, bayes | ídem | grid, random, bayes (`*_multiseries`) | ídem | grid, random, bayes | grid, random (`*_stats`) | grid (`grid_search_equivalent_date`) | solo bayes (`bayesian_search_foundation`, solo `TimeSeriesFold`) |
| `select_features` | S | S | `select_features_multiseries` | `select_features_multiseries` | N | N | N | N |
| skops | S | S | S | S | S | N | S | N |

Asimetrías relevantes para la 0.27:

- Las salidas probabilísticas no son homogéneas: Foundation ofrece `predict_interval` y `predict_quantiles` pero su backtesting solo acepta `quantiles`; Stats ofrece intervalos pero no cuantiles; EquivalentDate solo conformal.
- Ningún backtesting añade métricas de intervalo (cobertura, anchura) ni columna de horizonte; solo `fold` (`_validation.py:660`, `:1471`, `:2140`, `:2574`). `calculate_coverage`, `crps_from_quantiles`, `winkler_score` y `weighted_interval_score` ya existen en `skforecast.metrics`.
- Foundation solo tiene búsqueda bayesiana; no hay `grid_search_foundation` para comparar unos pocos `context_length` o checkpoints.

## 6. Hallazgos colaterales durante la revisión

### 6.1 `ForecasterStats.predict_interval` ignora `interval` (resultado silencioso erróneo)

`alpha` tiene valor por defecto `0.05` (`recursive/_forecaster_stats.py:1010`) y solo se usa `interval` cuando `alpha is None` (`:1078-1084`). Por tanto `predict_interval(steps, interval=[0.1, 0.9])` devuelve el intervalo del 95%, no el del 80%, sin aviso. Reproducido con `Arima(order=(1, 1, 0))`: las tres llamadas (sin argumentos, `interval=[0.1, 0.9]` e `interval=[0.1, 0.9], alpha=0.05`) devuelven exactamente los mismos límites (57,93 / 61,71 en el primer paso). El ejemplo de `skills/statistical-models/SKILL.md:72-75` usa justo `interval=[0.1, 0.9]` y lo presenta como intervalo del 80%. `backtesting_stats` no se ve afectado porque allí `alpha=None` por defecto. Arreglo barato: `alpha=None` por defecto y 0,05 solo si ambos son `None`; y deprecar `alpha`.

### 6.2 Skills desactualizadas sobre percentiles

`skills/prediction-intervals/SKILL.md:55-57`, `:70`, `:265` y `skills/prediction-intervals/references/interval-compatibility.md:23`, `:205` siguen diciendo que los percentiles "still work but emit a FutureWarning". Desde la 0.25 lanzan `ValueError` (comprobado: `predict_interval(interval=[5, 95])` da `ValueError: Lower interval bound (5) must be >= 0 and < 1`). Además, `skills/complete-api-reference/references/forecaster-methods.md:452` marca `set_params` como no disponible en `ForecasterEquivalentDate`, que sí lo tiene, y esa matriz no incluye a Foundation.
