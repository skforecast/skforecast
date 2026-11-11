# FE-01: función pública de evaluación de predicciones de backtesting (lente valor frente a mantenimiento)

**Veredicto: keep_modified.** No crear `evaluate_backtesting` en la 0.27. Casi todo su valor
llega con tres piezas ya identificadas (QW-04 `y_true`, columna `step` de DISC-1119 e
ISSUE-1409) más dos añadidos pequeños propios de FE-01: el aviso de infracobertura dentro del
backtesting y una receta documentada de agregación por grupos. Esfuerzo de lo que queda: S.
Reabrir la función en la 0.28 solo si, con las columnas ya publicadas, siguen llegando peticiones.

## Qué existe hoy (comprobado en 0.26.x)

- No existe ninguna función de evaluación por grupos: grep de `evaluate_backtesting`, `by_step`,
  `by_fold` o `groupby(['fold'` en `skforecast/` sin resultados. Las métricas se calculan solo
  sobre `pred` (`_validation.py:662-692`; multiserie `_utils.py:1067-1291`).
- La salida no lleva metadatos (`grep .attrs` vacío en `model_selection/`): ni nivel nominal del
  intervalo, ni `gap` (las filas del gap se quitan en `_validation.py:340`, `1101`, `1878`), ni tipo
  de residuos. Una función a posteriori tendría que pedir `cv`, `interval` y
  `use_in_sample_residuals` otra vez.
- Formatos de salida que la función tendría que entender (medido, `scratchpad/fe01/formats.py`):
  `['fold','pred']`; `+ lower_bound, upper_bound`; `+ q_0.1 ...`; `+ pred_boot_0 ...`;
  `+ lag_1 ...` con `return_predictors`; multiserie con `level`; stats con varios estimadores con
  `estimator_id`; además foundation y el clasificador (`*_proba`). Y `series` en tres formatos
  (DataFrame ancho, dict, MultiIndex largo).
- La guía de backtesting ya usa una receta pandas por grupos (error por mes, celda 36 de
  `docs/user_guides/backtesting.ipynb`); `summarize` de `comparing-forecasting-models.ipynb`
  (celda 5) son ~40 líneas, sobre todo para reconstruir `y_true` por nivel y MASE.

## Demanda

- #720 (2024, pregunta, 1 comentario): métrica por fold "como `evaluate` de sktime"; respuesta:
  calcularlo a mano. DISC-1119: métricas por horizonte a mano. #1409 (abierta por un mantenedor el
  2026-10-17): cobertura y anchura en las métricas de backtesting; dice expresamente que la
  alineación por nivel en multiserie "is easy to get wrong".
- Competidores: Prophet (`performance_metrics` por horizonte, con `coverage`), Nixtla
  (`cross_validation` devuelve `y` y `cutoff`; `utilsforecast.evaluate` agrupa), sktime
  (`evaluate`, una fila por fold). El patrón común es que la salida de validación traiga el valor
  real y el origen; la agregación es un groupby o una función genérica fina.
- Demanda real: sobre todo por paso y cobertura; por fold, una pregunta en dos años; agrupaciones
  propias, ninguna petición.

## Experimento: lo que queda para el usuario cuando existan `y_true` y `step`

`scratchpad/fe01/after_columns.py` (items_sales, 3 series, Ridge, conformal 80%, 14 folds de 7):
con las dos columnas simuladas, métricas por (level, step) con MAE, cobertura y anchura y MAE por
fold son 5 líneas de pandas estándar, y el MAE por nivel coincide con el de backtesting
(diferencia máxima 0,0). Una función ahorraría 2 o 3 líneas por uso.

Premisa del aviso, reproducida (`scratchpad/fe01/coverage.py`, bike_sharing últimos 120 días,
lags=24, steps=24, 80% nominal; Python 3.12.3, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1,
lightgbm 4.7.0):

| Modelo | Método | Cobertura in-sample | Cobertura out-sample |
|:--|:--|--:|--:|
| Ridge | bootstrapping | 0,81 | 0,92 |
| Ridge | conformal | 0,65 | 0,77 |
| LGBM | bootstrapping | 0,53 | 0,80 |
| LGBM | conformal | 0,26 | 0,62 |

## Coste de mantenimiento de la función completa

- Parseo de 5 familias de salida, 4 formatos de intervalo y 3 formatos de `series`; cada formato
  nuevo de salida (p. ej. QW-04, `step`, foundation con `interval`) obliga a tocarla.
- Le faltan datos de contexto: `gap` para el paso, `interval` para la cobertura nominal, conjunto
  de entrenamiento y `window_size` para MASE/RMSSE por grupo, tipo de residuos para el aviso.
- Choca con BUG-02: la función usaría todos los pares origen-fecha y la métrica de backtesting solo
  el último por fecha; dos MAE distintos para "lo mismo" mientras BUG-02 no se resuelva.
- Estimación: 150 a 250 líneas, 300 a 500 de tests, guía nueva, `llms-base.txt` y tres skills
  (backtesting-configuration, metric-selection, prediction-intervals).

## Versión reducida propuesta

1. Hacer QW-04 (`y_true`) y la columna `step` (DISC-1119) antes que nada.
2. En ISSUE-1409 (cobertura y anchura en la tupla de métricas, por nivel y agregados), emitir
   `ResidualsUsageWarning` (ya existe, `exceptions.py:263`) cuando `use_in_sample_residuals=True`,
   el intervalo tiene dos límites y la cobertura empírica queda más de 0,10 por debajo de la
   nominal, con un mínimo de filas (p. ej. 50) para evitar ruido; sugerir
   `set_out_sample_residuals` y `use_in_sample_residuals=False`. Solo forecasters ML (stats y
   foundation no usan residuos). Descartar filas con `y_true` NaN: `calculate_coverage` no ignora NaN
   (hallazgo de QW-04: 0,63 frente a 0,73).
3. Documentar la receta de 5 líneas por `step`, `fold` y `level` en `backtesting.ipynb` y en las
   skills backtesting-configuration y prediction-intervals; simplificar `summarize` en
   `comparing-forecasting-models.ipynb`.
4. WIS/CRPS para salidas `q_*` o `pred_boot_*`: dentro de #1409 como opción, no aquí.

## Riesgo

Bajo para la versión reducida: aviso nuevo (silenciable con `suppress_warnings`) y documentación.
El cambio de columnas es de QW-04/DISC-1119 y las métricas nuevas de #1409.
