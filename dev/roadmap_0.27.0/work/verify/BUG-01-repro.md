# BUG-01: verificación

**Veredicto: keep (esfuerzo S).** Se reproduce: backtesting con `initial_train_size=None` (valor por defecto de `TimeSeriesFold`) sobre un forecaster ya ajustado evalúa dentro del periodo de entrenamiento sin ningún aviso. El comportamiento está documentado (docstring y guía "Backtesting on training data"), así que no es un error de cálculo sino la falta de una salvaguarda barata ante una fuga de información silenciosa en el flujo más común (backtesting).

Script: `dev/roadmap_0.27.0/repro/BUG-01.py` (h2o, `ForecasterRecursive(LGBMRegressor(random_state=123), lags=12)`; skforecast 0.26.0, sklearn 1.9.1, lightgbm 4.7.0, pandas 2.3.3).

## Resultados medidos

| Caso | Observado | Referencia fuera de muestra | Avisos |
|---|---|---|---|
| A. `fit(y)` y luego `TimeSeriesFold(steps=12)` | MAE 0.0291; 192 de 192 predicciones dentro de `training_range_` | Mismos últimos 60 obs: in-sample 0.0384 frente a 0.0966 con `initial_train_size=144` (2.5 veces) | ninguno |
| B. `fit(y_train)` y backtesting sobre `y` completa sin `initial_train_size` | MAE 0.0491 que mezcla 132 predicciones in-sample y 60 fuera de muestra | 0.0966 en test (2 veces) | ninguno |
| C. Multiserie (`ForecasterRecursiveMultiSeries`, dict de 2 series) | average 0.0378 (s1 0.0277, s2 0.0479) | average 0.1047 (s1 0.0612, s2 0.1483) | ninguno |
| D. `grid_search_forecaster` con `initial_train_size=None` | No afectado: `set_lags` reinicia `is_fitted` y todas las combinaciones fallan con NotFittedError | | |
| E. Forecaster sin ajustar | `NotFittedError: forecaster must be already trained if no initial_train_size is provided` | | el mensaje empuja a ajustar el forecaster y caer en el caso A |

Corrección a la candidata: el "23 veces" (0.0036 frente a 0.0826) no se reproduce con LightGBM por defecto; comparando el mismo periodo la diferencia medida es de 2.5 veces (caso A) y de 2 veces en el caso mixto B. Sigue siendo una métrica claramente optimista y sin aviso. Con `verbose=True` solo se imprime "An already trained forecaster is to be used" (`_split.py:1257-1261`), sin decir que la evaluación es in-sample.

## Causa raíz

- `skforecast/model_selection/_split.py:106`: `initial_train_size=None` por defecto.
- `skforecast/model_selection/_utils.py:362-380` (`check_backtesting_input`): con `initial_train_size=None` solo comprueba `is_fitted` y `refit`; no compara el periodo de test con `forecaster.training_range_`.
- `skforecast/model_selection/_validation.py:580-597` (un solo modelo) y `:1376-1420` (multiserie): solo se ajusta si `initial_train_size is not None`; si no, se reutiliza el modelo externo (`externally_fitted=True`) y se predice desde `window_size`.

## Arreglo propuesto

En la rama `else` de `check_backtesting_input` (`_utils.py:370-380`), tras comprobar `is_fitted`: calcular la primera fecha de test (`index[window_size + gap]`, ya que con `initial_train_size=None` el split usa `window_size` como tamaño inicial, `_split.py:1014-1015`) y compararla con el final de `forecaster.training_range_` (Index en un solo modelo y en `ForecasterDirectMultiVariate`, dict por serie en `ForecasterRecursiveMultiSeries`). Si hay solape y `suppress_warnings=False`, emitir un aviso (nuevo `DataLeakageWarning(UserWarning)` o reutilizar uno existente) con el número de observaciones solapadas y el rango, e indicar que, si se buscan predicciones in-sample, el aviso puede ignorarse con `suppress_warnings=True`. Cambiar también el texto del NotFittedError para sugerir primero `initial_train_size`. Los tests: forecaster ajustado sobre todo (aviso), sobre una parte (aviso con el número correcto), con datos posteriores al entrenamiento (sin aviso) y multiserie. Volver a ejecutar `docs/user_guides/backtesting.ipynb` (la sección de backtesting in-sample mostrará el aviso, o pasarle `suppress_warnings=True`) y añadir nota en `releases.md`. Las búsquedas no se ven afectadas (caso D).
