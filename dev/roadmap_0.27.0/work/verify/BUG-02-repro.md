# BUG-02: con `fold_stride < steps` la métrica solo mide el horizonte más corto

**Veredicto: keep (reproducido).** No es un error de código en sentido estricto: la guía
de usuario documenta que se usa "la última predicción de cada observación". Pero la
consecuencia no se explica en ningún sitio y el resultado es silenciosamente engañoso:
con `steps=24, fold_stride=1` la métrica devuelta es el MAE a 1 paso, y todas las
búsquedas seleccionan modelos por ese error.

Script: `dev/roadmap_0.27.0/repro/BUG-02.py` (bike_sharing horario, LGBM, train hasta
2012-08-31, 30 días de test; ~15 s). Entorno: Python 3.12.3, skforecast 0.26.0,
pandas 2.3.3, numpy 2.5.4, scikit-learn 1.9.1, lightgbm 4.7.0.

## Resultado medido

1. `backtesting_forecaster`, lags=24, steps=24, refit=False:

| fold_stride | MAE devuelto | MAE real (todos los pares origen-fecha) | MAE paso 1 | Pasos usados en la métrica |
|---|---|---|---|---|
| 24 | 75.68 | 75.68 | 18.75 | 1..24 |
| 6 | 52.19 | 71.62 | 41.80 | 1..6 |
| 1 | 35.37 | 72.62 | 35.37 | solo 1 |

   Con fold_stride=1 se calculan 17004 predicciones pero la métrica usa 720, todas a
   1 paso. El MAE por paso crece de 35 (paso 1) a 85 (paso 24).

2. `grid_search_forecaster` con steps=24, fold_stride=1:

| lags | MAE búsqueda | MAE real 24 pasos | MAE fold_stride=24 | rango búsqueda | rango real |
|---|---|---|---|---|---|
| 3 | 55.53 | 195.99 | 195.47 | 4 | 4 |
| 24 | 35.37 | 72.62 | 75.68 | **1** | 3 |
| [1,2,24,168] | 42.23 | 69.36 | 68.45 | 3 | 2 |
| [1,2,3,23,24,25,167,168,169] | 35.80 | 58.97 | 59.74 | 2 | **1** |

   La búsqueda elige lags=24 (MAE real 72.6) en lugar de los lags estacionales (59.0):
   un 23% peor en el horizonte que el usuario configuró.

3. `backtesting_forecaster_multiseries`: mismo efecto por nivel (users: devuelto 28.53,
   todos los pares 60.20, paso 1 28.53).

## Causa raíz

`index.duplicated(keep='last')` antes de calcular métricas cuando `cv.overlapping_folds`:

- `skforecast/model_selection/_validation.py:662-667` (`_backtesting_forecaster`)
- `skforecast/model_selection/_validation.py:1488-1493` (`_backtesting_forecaster_multiseries`)
- `skforecast/model_selection/_validation.py:2159-2164` (`_backtesting_stats`)
- `skforecast/model_selection/_validation.py:2643-2648` y `2674-2679` (`_backtesting_foundation`)

Las búsquedas heredan el problema porque llaman a esas funciones
(`_search.py:448, 745, 1446, 1820, 2328, 2771, 3027`).

## Por qué afecta a usuarios normales

- `fold_stride` nació en #764 justo para "previsión a 60 días cada lunes"; el usuario
  pedía evaluar por ejecución de previsión (todas las fechas de cada origen), no por fecha.
- La skill `skills/backtesting-configuration/SKILL.md:202-208` recomienda `fold_stride=1`
  para "máxima cobertura de evaluación con horizonte de 12 pasos": la métrica resultante es
  la de 1 paso.
- skforecast avisa con `OneStepAheadValidationWarning` (`_utils.py:653-658`) cuando se usa
  `OneStepAheadFold`, pero con `fold_stride=1` da exactamente esa métrica, más cara, sin aviso.
- La guía (`docs/user_guides/backtesting.ipynb`, sección "Backtesting using fold stride")
  dice qué predicción se usa, no que la métrica deja de medir el horizonte `steps`.
- Usuarios afectados: nicho moderado (quien usa fold_stride < steps), pero en ellos el
  error afecta a toda métrica y a toda selección de modelos.

## Arreglo propuesto (esfuerzo S)

1. Nuevo parámetro en `TimeSeriesFold` (así lo heredan backtesting y búsquedas), p. ej.
   `overlapping_metric: str = 'last'` con opción `'all'`. En los 5 puntos:
   `if overlapping_folds and cv.overlapping_metric == 'last': ...`. Con `'all'` no se
   deduplica; `_calculate_metrics_backtesting_multiseries` ya admite índice repetido
   (merge many-to-one en `_utils.py:1160`).
2. Mientras el defecto sea `'last'`, aviso al usar fold_stride < steps: "la métrica solo
   evalúa los horizontes 1..fold_stride".
3. No cambiar el defecto sin anunciarlo: la receta de meses naturales (`steps=31`,
   `fold_stride=1`, `skip_folds`) depende de `'last'`.
4. Corregir la skill (`SKILL.md:202-208`), la guía y `llms-base.txt`.
5. Complementa DISC-1119 (columna de paso del horizonte): esa columna haría visible el
   problema, pero no corrige la métrica que usan las búsquedas. Coordinar con BUG-09.
