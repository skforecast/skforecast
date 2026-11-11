# BUG-05: grid/random search con todas las combinaciones fallidas

**Veredicto: later** (se reproduce, pero no es un resultado silencioso erróneo ni un bug que haga daño; es pulido de UX sobre un comportamiento deliberado y testeado).

## Reproducción

Script: `dev/roadmap_0.27.0/repro/BUG-05.py` (datos sintéticos, Ridge; skforecast 0.26.0, pandas 2.3.3, numpy 2.5.4, scikit-learn 1.9.1).

| Caso | Observado |
|------|-----------|
| A. `grid_search_forecaster`, `differentiation=1` en el forecaster y no en el cv | 4 avisos "Parameters skipped" + 1 "No valid parameter combinations found"; `results.shape=(0, 5)` con `params` duplicada; `is_fitted=False`; `predict` -> `NotFittedError` |
| B. `initial_train_size=500` con 200 observaciones | Igual que A |
| C. Errata en el nombre del parámetro (`alphaa`) | Igual que A (2 avisos) |
| D. `random_search_forecaster` + `OneStepAheadFold`, errata | Igual que A (3 avisos) |
| E. Forecaster ya ajustado (alpha=1234) antes de la búsqueda | Igual, pero `is_fitted=True` y `predict` funciona con el modelo SIN ajustar (alpha=1234) |
| F. `grid_search_forecaster_multiseries` | `(0, 7)`, sin columna duplicada, `is_fitted=False` |
| G/H. `bayesian_search_forecaster(_multiseries)` | Lanza el `ValueError` original (incoherencia con grid/random) |

## Valoración

- Cada aviso "Parameters skipped" incluye el mensaje del error original, así que la causa es visible; no hay métricas ni intervalos erróneos. La afirmación de que "predict falla lejos de la causa" está exagerada.
- El comportamiento es deliberado: se introdujo en marzo de 2026 (commit c66130afa, punto 3 de `dev/review_grid_search.md`) para sustituir un `KeyError`/"No objects to concatenate" poco claro, y está fijado por tests (`test_evaluate_grid_hyperparameters.py:181-229`, `test_evaluate_grid_hyperparameters_multiseries.py:1721,1761`, `test_evaluate_grid_hyperparameters_stats.py:185`, `test_grid_search_equivalent_date.py:360`). Cambiarlo requiere acuerdo de los mantenedores.
- Sí hay argumentos para mejorarlo: scikit-learn lanza `ValueError("All the N fits failed ...")` cuando fallan todos los ajustes (`sklearn/model_selection/_validation.py`, `_warn_or_raise_about_fit_failures`) y solo avisa si fallan algunos; `bayesian_search` ya propaga el error. Con rejillas grandes (p. ej. 3 lags x 27 params) el usuario recibe 81 avisos largos idénticos salvo los parámetros.
- Caso E (forecaster previamente ajustado + `return_best=True`) es el único con efecto silencioso, y aun así con avisos visibles: poco frecuente.
- No hay issues de usuarios sobre esto desde la 0.21.

## Causa raíz

- `skforecast/model_selection/_search.py:474-476`: `except Exception` -> `warnings.warn("Parameters skipped ...")` y `continue`.
- `_search.py:517-524`: `pd.concat([results, results['params'].apply(pd.Series)])` se ejecuta antes del `if results.empty` (de ahí la columna `params` duplicada); después solo avisa y devuelve el DataFrame vacío, sin reajustar el forecaster.
- Mismo patrón en `_search.py:1480-1482` y `1521-1526` (multiserie), `2785-2786` y `2820-2827` (stats), `3039-3040` y `3084-3091` (equivalent date).

## Esbozo del arreglo (S)

1. En las 4 funciones `_evaluate_grid_hyperparameters*`, guardar los errores (`Counter` de mensajes, como sklearn) y la primera excepción.
2. Si no queda ninguna combinación válida: `raise ValueError(f"All {n} parameter combinations failed. ... First error: {first_exc}") from first_exc`, en lugar del aviso y el DataFrame vacío.
3. Mover el chequeo de vacío antes del `pd.concat` (single y stats).
4. Opcional: con `TimeSeriesFold`, validar una sola vez antes del bucle lo que no depende de los parámetros (p. ej. `differentiation` del cv frente al forecaster).
5. Actualizar los 6 tests citados a `pytest.raises` y añadir nota "Changed" en `docs/releases/releases.md` (cambio de comportamiento).

Se puede agrupar con cualquier otro PR de `model_selection`; no justifica un hueco propio en la 0.27.
