# QW-03: heredar `differentiation` del forecaster en TimeSeriesFold y OneStepAheadFold

**Veredicto: keep** (esfuerzo S, sin romper la API)

## Qué existe hoy

- `skforecast/model_selection/_utils.py:270-288` (`check_backtesting_input`) y `:602-620` (`check_one_step_ahead_input`): si `forecaster.differentiation_max != cv.differentiation` se lanza `ValueError`, también cuando `cv.differentiation` es `None` (valor por defecto).
- Justo después, `_backtesting_forecaster` (`_validation.py:513-518`) y `_backtesting_forecaster_multiseries` (`_validation.py:1286-1291`) hacen `cv.set_params({'window_size': forecaster.window_size, 'differentiation': forecaster.differentiation_max, ...})` sobre una copia del cv. El valor del usuario nunca se usa para calcular nada; `window_size` ya se hereda sin comprobación.
- Con `OneStepAheadFold`, las búsquedas (`_search.py:368,681,1337,1732`) solo usan `cv.initial_train_size`; `_train_test_split_one_step_ahead` y `_calculate_metrics_one_step_ahead` (`_utils.py:814`) usan la diferenciación del forecaster. `cv.differentiation` solo sirve para la comprobación.
- Solo importa si el usuario llama a `cv.split()` a mano (amplía `last_window`); eso no cambia.
- La referencia para IA ya describe el comportamiento propuesto: `tools/ai/llms-base.txt:340,370` y `skills/complete-api-reference/references/model-selection-signatures.md:291,304` dicen "set automatically by forecaster". Las skills no muestran `differentiation` en el cv, así que el código generado con ellas falla en cuanto el forecaster diferencia.
- Las guías dicen lo contrario: `probabilistic-forecasting-metrics.ipynb` (2 celdas, "must match"), `probabilistic-forecasting-conformal-calibration.ipynb` (2 celdas, "Must match") y `time-series-differentiation.ipynb` (1 celda, "must be specified in the cv object").
- No está en la lista de identificados. No hay ningún issue con este mensaje de error (búsqueda en GitHub). El #751 es otro problema, ya resuelto.

## Reproducción (Python 3.12.3, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1)

Serie con tendencia (200 días), `ForecasterRecursive(Ridge(), lags=10, differentiation=1)`, `TimeSeriesFold(steps=10, initial_train_size=150)` sin `differentiation`:

| Flujo | Hoy |
|---|---|
| `backtesting_forecaster` | ValueError |
| `grid_search_forecaster` (TimeSeriesFold) | sin error: 2 RuntimeWarning "Parameters skipped", "No valid parameter combinations found" y **DataFrame vacío**; con `return_best=True` tampoco hay error y el forecaster queda sin ajustar |
| `bayesian_search_forecaster` | "Trial 0 failed" y luego ValueError |
| `grid_search_forecaster` (OneStepAheadFold) | ValueError |
| `ForecasterDirect`, `ForecasterDirectMultiVariate` | ValueError |
| `ForecasterRecursiveMultiSeries` con `{'a': 1, 'b': None, '_unknown_level': 1}` | ValueError; hay que saber que se pasa `differentiation_max` (1) |

El "before" es correcto y la fricción es real. El peor caso es la búsqueda en rejilla, que acaba sin ningún resultado y solo con avisos.

## Prototipo medido (worktree desechable, ya borrado)

Cambio de 2 líneas en `_utils.py`: `if hasattr(forecaster, 'differentiation') and cv.differentiation is not None:` en las dos funciones.

- Los 8 casos anteriores pasan a funcionar. El error se mantiene si el usuario fija un valor distinto (p. ej., forecaster sin diferenciación y `cv.differentiation=1`).
- Resultados **idénticos** (`DataFrame.equals`) a usar `differentiation=1` en el cv: métricas y predicciones del backtesting, rejilla con TimeSeriesFold, rejilla con OneStepAheadFold y backtesting multiserie con diccionario.
- `model_selection/tests/tests_utils/`: 220 pasan. Ningún test espera el error con `cv.differentiation=None`: los 4 tests del mensaje usan valores explícitos distintos (`test_check_backtesting_input.py:183-260`, `test_check_one_step_ahead_input.py:178,213`) y no cambian.

## Alcance del cambio (S)

1. `_utils.py`: 2 líneas (las dos comprobaciones).
2. Docstrings de `differentiation` en `_split.py` (BaseFold, TimeSeriesFold y OneStepAheadFold): "Si es None, en backtesting y en las búsquedas se toma del forecaster (`differentiation_max`)".
3. Tests: un test sin error con `cv.differentiation=None` en cada fichero de comprobación, y uno en `tests_validation` que compare la salida con la que se obtiene con el valor explícito.
4. Docs: cambiar el comentario en 5 celdas de 3 cuadernos (no hace falta volver a ejecutarlos) o quitar el argumento. `llms-base.txt` y `model-selection-signatures.md` ya son correctos; si acaso precisar "(if None, taken from the forecaster)" y regenerar.
5. Nota de versión (Changed, model_selection).

## Riesgo de API

Ninguno que rompa nada: es una relajación (código que antes fallaba ahora funciona) y no cambia ni la salida ni los valores por defecto. Mantener el error con un valor explícito distinto conserva el comportamiento de los tests y del código actual. Otra opción sería quitar la comprobación del todo, porque el valor nunca se usa, pero no aporta nada más y cambia los tests existentes. Afecta también a ForecasterRnn (mismo camino), lo que no es un problema.

## Extra observado (no se incluye en este QW)

La rejilla con TimeSeriesFold repite la misma comprobación en cada combinación dentro de `try/except` (`_search.py:443-475`). Cualquier error de configuración del cv (no solo este) acaba en un DataFrame vacío con avisos, en vez de un error claro al principio. Llamar a `check_backtesting_input` una vez antes del bucle lo arreglaría. Es otro tema, de prioridad baja.
