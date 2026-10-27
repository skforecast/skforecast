# BUG-09: `backtesting_stats` falla con `fold_stride != steps` (varios estimadores y Sarimax sin reentrenar)

**Veredicto: keep (reproducido, prioridad baja dentro de STATS, esfuerzo S).** No es un
resultado silenciosamente erróneo: todos los casos lanzan error (o, en `grid_search_stats`,
descartan todos los candidatos con avisos). Pero falla en combinaciones documentadas y
razonables (prever 7 días cada día, comparar varios modelos, Sarimax con `refit=False`, que
la propia librería sugiere en `LongTrainingWarning`). El arreglo son unas 10 líneas en un
fichero y se ha validado con diferencia 0,0 frente a referencias independientes. Además, la
verificación amplía el alcance: **`refit` entero >= 3 con Sarimax falla incluso con
`fold_stride == steps`** (el valor por defecto).

Script: `dev/roadmap_0.27.0/repro/BUG-09.py` (serie diaria sintética con estacionalidad
semanal, 300 días, 42 de backtesting, steps=7; 1 a 3 min). Entorno: Python 3.12.3,
skforecast 0.26.0, pandas 2.3.3, numpy 2.5.4, scikit-learn 1.9.1, statsmodels 0.15.0.

## Resultado medido (código actual)

| Caso | Resultado |
|---|---|
| (a) `[Arima(1,1,1), Ets(AAN)]`, refit=True, fold_stride=7 o 10 | OK |
| (a) igual, fold_stride=2 o 1 | `ValueError: Length of values (2) does not match length of index (1)`, después de hacer todos los ajustes (8 a 20 s perdidos) |
| (b) Sarimax, refit=False, fold_stride=7 (control) | OK, MAE 1,3597 |
| (b) Sarimax, refit=False, fold_stride=1, 3 o 10 | `ValueError: ... last_window has to start at the end of the index seen by the forecaster` |
| (b) Sarimax, refit=3 o 4, fold_stride=7 (por defecto) | mismo ValueError (refit=2 funciona) |
| (b) `grid_search_stats`, Sarimax, refit=False, fold_stride=1 | sin excepción: 0 filas, `RuntimeWarning: Parameters skipped` por cada candidato y "No valid parameter combinations found" |

## Causa raíz

- (a) `skforecast/model_selection/_validation.py:2160-2164`: el filtro de solapamiento
  `~index.duplicated(keep='last')` usa solo el índice de fecha; con salida larga (una fila
  por fecha y estimador) deja una sola fila por fecha, es decir, solo el último estimador.
  El `groupby('estimator_id')` (:2182) produce un único grupo y `metric_values.insert(0,
  'estimator_id', forecaster.estimator_ids)` (:2195) falla. Las búsquedas exigen un único
  estimador (`_search.py:2729`), así que (a) solo afecta a `backtesting_stats`.
- (b) `skforecast/model_selection/_validation.py:1825-1830`: con `refit=False` la ventana es
  `y[test_start - steps : test_start]`, que solo coincide con lo ya visto por el forecaster
  (`extended_index_`, `_forecaster_stats.py:727-737`) si `fold_stride == steps`. Con `refit`
  entero la ventana empieza en `fold[1][1]` (fin del último entrenamiento), que es incorrecto
  desde el segundo fold consecutivo sin reentrenar, porque `append` ya extendió el modelo.
- Los tests con `fold_stride` (`test_backtesting_stats.py:824,894`) solo cubren `refit=True`
  y un estimador; no hay tests de `refit` entero.

## Arreglo propuesto (validado)

```python
# _fit_predict_forecaster_stats: el forecaster ha visto la serie hasta el origen del
# fold anterior, sea cual sea fold_stride o refit.
fold_position = [f[0] for f in folds].index(fold[0])
previous_fold = folds[fold_position - 1] if fold_position > 0 else fold
last_window_iloc_start = previous_fold[3][0]
last_window_iloc_end   = fold[3][0]  # test_iloc_start

# _backtesting_stats: deduplicar por (fecha, estimator_id) con varios estimadores
if forecaster.n_estimators == 1:
    is_duplicated = preds.index.duplicated(keep='last')
else:
    is_duplicated = preds.set_index('estimator_id', append=True).index.duplicated(keep='last')
```

- (a) con el arreglo: MAE de cada estimador idéntico al backtesting del estimador solo
  (Arima 6,2311, Ets 8,3975; diferencia 0,0).
- (b) con el arreglo: predicciones idénticas (diferencia máxima 0,0) a una referencia
  independiente (copia del forecaster ajustado y un único `append` hasta cada origen) para
  refit=False con fold_stride 1, 3, 7 y 10, y para refit=3. `grid_search_stats` devuelve las
  dos filas (MAE 1,1238 y 1,1729).
- Con el parche aplicado en un worktree pasan `test_backtesting_stats.py`,
  `test_grid_search_stats.py` (40 tests) y `test_evaluate_grid_hyperparameters_stats.py` (12).
- Tests nuevos: dos estimadores con fold_stride < steps; Sarimax con refit=False y
  fold_stride < y > steps; Sarimax con refit=3. Coordinar con BUG-02 (misma línea de
  deduplicación: si BUG-02 cambia la semántica de `keep='last'`, aplicarla por estimador).
