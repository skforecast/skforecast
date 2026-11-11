# QW-07: mensajes de error accionables en cuatro puntos de fricción

**Veredicto: keep_modified** (esfuerzo S confirmado con prototipo: +142/-3 líneas en 4 ficheros, 980 tests existentes pasan sin tocarlos).

## Qué existe hoy (0.26.x, comprobado)

- (a) `check_extract_values_and_index` (`skforecast/utils/utils.py:2046-2050`) es el único punto del error de frecuencia (cubre `y`, `series` DataFrame y exog). Tras `read_csv` la `freq` es `None` y `pd.infer_freq` devuelve `'MS'`; con un hueco devuelve `None`. El mensaje no da la receta.
- (b) `check_interval` (`utils.py:1222-1229` y `1260-1264` para `quantiles`): `[10, 90]` da `Lower interval bound (10) must be >= 0 and < 1.` sin pista. Los percentiles se **eliminaron en la 0.25.0 (11-09-2026)** (`docs/releases/releases.md:292`): quien ignoró el FutureWarning de la 0.23/0.24 lo encuentra ahora.
- (c) `check_residuals_input` (`utils.py:1875-1947`): `fit` usa `store_in_sample_residuals=False` por defecto, y en multiserie deja `{'x1': None, 'y': None, '_unknown_level': None}` (DirectMultiVariate: `{'y': None}`). El dict no está vacío, así que no salta el mensaje accionable (el de Recursive/Direct) sino el del bucle final: `Residuals for level 'x1' are None or empty. Check ...`, y con `encoding=None` nombra `'_unknown_level'`. Lo encuentra **cualquier usuario multiserie que llama a `predict_interval` tras un `fit` por defecto**: es el subpunto de más valor.
- (d) Peor de lo descrito: `grid_search_forecaster`/`random_search_forecaster` (y sus `_multiseries`) no validan el tipo; llaman a `_backtesting_forecaster` privado. Medido:
  - Foundation o Stats en `grid/random_search_forecaster`: `AttributeError: ... has no attribute 'lags'` (`_utils.py:73`, `initialize_lags_grid`).
  - RecursiveMultiSeries en `grid_search_forecaster` (o Recursive en `grid_search_forecaster_multiseries`): **sin excepción**, un `RuntimeWarning "Parameters skipped"` por combinación (10 con una rejilla 3x3) y un DataFrame vacío, también con `return_best=True` (`_search.py:474,519`).
  - `bayesian_search_*`: traza de Optuna `Trial 0 failed` y luego el TypeError del backtesting.
  - Corrección al candidato: `backtesting_forecaster` con Foundation ya da TypeError (no AttributeError), pero sin nombrar la función correcta (`_validation.py:866-871,1695-1700,2318-2323,2825-2830`; `_search.py:2235,2722,2943`).
- La skill `skills/troubleshooting-common-errors/SKILL.md` ya trata los cuatro casos como errores típicos (líneas 72-80, 159-180, 184-193), pero cita textos que no coinciden con los mensajes reales (`"The index of the series must be a DatetimeIndex with frequency"`, `"No in-sample residuals stored"`). Las skills de intervalos dicen que los percentiles siguen funcionando (ya en DOC-skills-percentiles-stale).

## Prototipo (worktree temporal, ya borrado; diff en scratchpad/qw07/prototype.diff)

- (a) En la rama de error: `pd.infer_freq` dentro de `try/except` (lanza ValueError con menos de 3 fechas). Si infiere: `Inferred frequency: 'MS'. Use .asfreq('MS') to set it.`; si no: frase sobre huecos/duplicados con `.asfreq(freq)`. Sin coste fuera del error.
- (b) Frase de migración añadida al final de los tres mensajes cuando el valor está en (1, 100].
- (c) Basta con tratar como vacío un dict cuyos valores son todos `None` (2 líneas, en las ramas in-sample y out-sample): se reutiliza el mensaje accionable existente. Además se añade la acción al mensaje por nivel (caso parcial con out-sample). No hace falta recoger todos los niveles.
- (d) Dict `{forecaster: funciones}` y `check_forecaster_type_model_selection` en `model_selection/_utils.py`, llamado al inicio de `_evaluate_grid_hyperparameters`, `bayesian_search_forecaster` y sus versiones multiserie (antes del estudio de Optuna); la lista de funciones se añade al final de los 7 mensajes existentes. Para las búsquedas univariantes la lista permitida es Recursive, Direct y RecursiveClassifier (EquivalentDate tiene `grid_search_equivalent_date`).
- Tests: 615 (utils de intervalos/residuos/índice, intervalos de RecursiveMultiSeries, DirectMultiVariate completo) y 365 (búsquedas y backtesting de model_selection, sin `slow`) pasan sin cambios, porque los tests usan `match` y los textos solo se alargan. Faltan unos 8 tests parametrizados nuevos.

## Riesgo de API

- Sin parámetros ni columnas nuevas. Solo cambian mensajes y, en un uso incorrecto, el tipo de fallo (AttributeError, o DataFrame vacío con avisos, pasa a TypeError inmediato).
- Subclases de forecasters: ya fallan hoy (medido: 0 filas en `grid_search_forecaster`, TypeError en `backtesting_forecaster`), así que la comprobación por nombre de clase no rompe nada nuevo.

## Alcance recomendado

- Hacer los cuatro en un PR. Orden de valor: (c) y (d) primero, luego (b) y (a).
- Actualizar en el mismo PR los textos citados en la skill de troubleshooting y cerrar DOC-skills-percentiles-stale; nota en releases.md (Changed, mensajes de error). No toca `llms-base.txt` salvo los textos de error citados.
- Fuera de alcance: el mensaje de `series` como dict sin frecuencia (otro texto, `check_preprocess_series`) y `level` en percentiles de los estimadores stats.

## Before / after revisados

```python
y = pd.read_csv("data.csv", parse_dates=["date"], index_col="date")["y"]
forecaster.fit(y)
# antes: ValueError: `y` has a pandas DatetimeIndex without a frequency. To avoid this error, set the frequency of the DatetimeIndex.
# después: ... Inferred frequency: 'MS'. Use `.asfreq('MS')` to set it.

forecaster.predict_interval(steps=12, interval=[10, 90])
# antes: ValueError: Lower interval bound (10) must be >= 0 and < 1.
# después: ... Since version 0.23.0, `interval` and `quantiles` are expressed as quantiles in [0, 1], not percentiles: divide the values by 100, e.g. `interval = [0.1, 0.9]` instead of `[10, 90]`.

forecaster_ms.fit(series)                  # store_in_sample_residuals=False por defecto
forecaster_ms.predict_interval(steps=12)
# antes: ValueError: Residuals for level 'x1' are None or empty. Check `forecaster.in_sample_residuals_by_bin_`.  (encoding=None: '_unknown_level')
# después: ValueError: `forecaster.in_sample_residuals_by_bin_` is either None or empty. Use `store_in_sample_residuals = True` when fitting the forecaster or use the `set_in_sample_residuals()` method before predicting.

grid_search_forecaster(forecaster_ms, y, cv, param_grid, metric="mean_absolute_error")
# antes: un RuntimeWarning por combinación y DataFrame vacío, sin excepción
# después: TypeError: `forecaster` must be of type ['ForecasterRecursive', 'ForecasterDirect', 'ForecasterRecursiveClassifier']. Got ForecasterRecursiveMultiSeries. For `ForecasterRecursiveMultiSeries` use: `backtesting_forecaster_multiseries`, `grid_search_forecaster_multiseries`, ...

grid_search_forecaster(forecaster_foundation, y, cv, {"context_length": [256, 512]}, metric="mean_absolute_error")
# antes: AttributeError: 'ForecasterFoundation' object has no attribute 'lags'
# después: TypeError: ... Got ForecasterFoundation. For `ForecasterFoundation` use: `backtesting_foundation`, `bayesian_search_foundation`.
```
