# QW-08: 'root_mean_squared_error' como string y nombres únicos en create_mean_pinball_loss

**Veredicto: keep_modified** (esfuerzo S; RMSE sin cambio de API; pinball renombra una columna de salida)

## Qué existe hoy

- `skforecast/metrics/metrics.py:45-62`: `allowed_metrics` admite `root_mean_squared_scaled_error` (l. 52) pero no `root_mean_squared_error`. sklearn lo trae desde 1.4 y el mínimo del proyecto es `scikit-learn>=1.6` (`pyproject.toml:58`), así que el import es seguro.
- `skforecast/metrics/metrics.py:555-558`: la función interna de `create_mean_pinball_loss` se llama siempre `mean_pinball_loss_q`, sin el cuantil.
- Las búsquedas ya validan nombres únicos (`_search.py:394-400`, `707-713`, `1373-1376`, `1768-1771`, `2268-2271`, `2748-2754`, `3002-3008`). Los backtesting no: `_validation.py:540-550` construye la lista y `693-695`, `2174-2193`, `2705` usan `m.__name__` como columnas sin comprobar duplicados.
- No aparece en la lista de identificados. Nada de WAPE en el paquete, docs ni skills; la skill `metric-selection` ya recomienda sMAPE (que sí existe como string) cuando hay ceros.

## Reproducción (skforecast 0.26.0, Python 3.12.3, sklearn 1.9.1, pandas 2.3.3, numpy 2.3.x)

- `metric=["root_mean_squared_error", "mean_absolute_error"]` -> `ValueError: Allowed metrics are: [...]`. El "before" con `from sklearn.metrics import root_mean_squared_error` funciona (0.7157 / 0.5944). Fricción real pero pequeña: una línea de import y un error al probar el nombre natural.
- Dos pinball (0.1 y 0.9) en `backtesting_forecaster`: columnas duplicadas `mean_pinball_loss_q`, `mean_pinball_loss_q` (indexar por nombre devuelve un DataFrame).
- Dos pinball en `backtesting_forecaster_multiseries`: `TypeError: Axis must be specified when shapes of a and weights differ` (críptico, sale de `np.average` en `_utils.py:1246`).
- Dos pinball en `grid_search_forecaster`: `ValueError: ... each metric name must be unique`.
- El mismo problema lo tienen dos lambdas (`<lambda>`) o dos `functools.partial` (ambos quedan como `wrapper` tras `add_y_train_argument`).

Matiz: evaluar dos cuantiles distintos sobre la misma predicción puntual tiene poco sentido estadístico (el backtesting aplica las métricas a la columna `pred`). El valor real del nombre con el cuantil es legibilidad (tablas de resultados de búsquedas por cuantil, como en la guía de regresión cuantílica) más que habilitar un flujo nuevo.

## Prototipo medido (worktree desechable, ya borrado)

Cambio: 3 líneas para RMSE (import, lista y diccionario) y 2 para `__name__`/`__qualname__ = f"mean_pinball_loss_q_{alpha}"`.

- `skforecast/metrics/tests`: 80 pasan, 1 falla solo por el texto de la lista de métricas permitidas (`test_get_metric.py:27`).
- RMSE como string funciona en `backtesting_forecaster`, `grid_search_forecaster` con `OneStepAheadFold` y `backtesting_forecaster_multiseries` (por nivel, average, weighted_average y pooling).
- Dos pinball: columnas `mean_pinball_loss_q_0.1`, `mean_pinball_loss_q_0.9`; la búsqueda deja de rechazarlas.

## Ajustes respecto a la propuesta

1. Núcleo: RMSE como string. Aprovechar para corregir las ~24 docstrings de `metric` en `_validation.py` y `_search.py`, que listan solo 6 strings y omiten `median_absolute_error` y `symmetric_mean_absolute_percentage_error` (ya admitidos).
2. Añadir en los backtesting la misma validación de nombres únicos que tienen las búsquedas (unas 6 líneas en `_validation.py` o en `check_backtesting_input`): error claro en lugar de columnas duplicadas o el `TypeError` de multiserie. Cubre lambdas y partials, no solo pinball.
3. Renombrar pinball con el cuantil, anotado en "Changed" de la nota de versión (rompe `results["mean_pinball_loss_q"]`) y reejecutar o retocar la guía `probabilistic-forecasting-quantile-regression.ipynb` (4 apariciones del nombre en salidas).
4. WAPE: sacarlo del quick win (función nueva, tests y docs). sMAPE ya cubre el caso de ceros en la skill; dejarlo para "later" si aparece demanda.
5. Docs: fila RMSE en la tabla de `skills/metric-selection/SKILL.md` (Step 2), `docs/user_guides/metrics.ipynb`, comentario de `metric` en `tools/ai/llms-base.txt` y regenerar contexto IA.

## Riesgo

Bajo. RMSE es aditivo. La validación de nombres únicos solo convierte en error claro lo que hoy ya es salida rota o excepción críptica. El renombrado de pinball cambia el nombre de una columna de salida para usuarios de `create_mean_pinball_loss` (nicho); aceptable en una minor con nota de versión.
