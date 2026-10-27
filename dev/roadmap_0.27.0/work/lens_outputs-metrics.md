# Lente outputs-metrics: salidas de backtesting, búsquedas y métricas

Entorno: skforecast 0.26.0 (rama 0.26.x), Python 3.12, numpy 2.5.4, pandas 2.3.3,
scikit-learn 1.9.1, lightgbm instalado. Datos: `fetch_dataset("h2o_exog")`,
`fetch_dataset("bike_sharing")` y `fetch_dataset("items_sales")`. Scripts en
`/tmp/claude-0/-home-user-skforecast/8505cf39-716d-5c4c-adf6-669c7ec65b16/scratchpad/outputs-metrics/`
(s1.py a s9.py).

Qué se ejecutó e inspeccionó: `backtesting_forecaster` (puntual, `interval=0.8`,
cuantiles, `fold_stride` 24/6/1), `backtesting_forecaster_multiseries` (con intervalo
y agregados), `grid_search_forecaster` y `bayesian_search_forecaster` (con
`TimeSeriesFold` y `OneStepAheadFold`, `return_best`, varias métricas),
`grid_search_forecaster_multiseries`, métricas escaladas (MASE) con distintos `refit`.

Qué devuelven hoy:

- `backtesting_*`: tupla `(metric_values, predictions)`. `metric_values` es una sola
  fila (o una fila por serie más `average`, `weighted_average`, `pooling` en multiserie).
  `predictions` tiene `fold`, `pred` (y `level` en multiserie) más las columnas de
  intervalo. No hay columna de paso del horizonte, ni de origen (cutoff), ni del valor
  real, ni métricas por fold.
- `grid/random_search_*`: DataFrame con `lags`, `lags_label`, `params`, una columna por
  métrica (en multiserie `metrica__agregacion`) y una columna por hiperparámetro.
  Sin tiempos ni métricas por serie. `bayesian_search_*` devuelve además el `Study` de
  Optuna.

Ya identificados y no repetidos aquí salvo como extensión: métricas de intervalo en
backtesting (#1409, GAP-backtesting-interval-metrics), columna de paso y métricas por
horizonte (D1119), `backtesting_foundation` solo con `quantiles`.

## Candidatos (ordenados por valor)

### 1. Con folds solapados (`fold_stride < steps`) la métrica solo mide el horizonte más corto, y las búsquedas eligen modelo por el error a 1 paso (bug silencioso, S/M)

`_backtesting_forecaster` calcula la métrica tras quedarse con la última predicción de
cada fecha (`index.duplicated(keep='last')`, `skforecast/model_selection/_validation.py:662-667`).
La última predicción de una fecha es la del fold con origen más reciente, es decir, la
de horizonte más corto. Mismo patrón en multiserie (`_validation.py:1488-1493`), stats
(`:2159-2164`) y foundation (`:2643-2648`, `:2674-2679`), y por tanto en todas las
búsquedas, que llaman a esas funciones. La guía lo menciona en una frase
(`docs/user_guides/backtesting.ipynb`, sección "Backtesting using fold stride"), pero
la consecuencia no se explica: con `fold_stride=1` el número que se devuelve es el MAE
a 1 paso, no el del horizonte `steps` que el usuario configuró. `fold_stride` nació
precisamente para evaluar horizontes largos con orígenes frecuentes
([#764](https://github.com/skforecast/skforecast/issues/764): "evaluate model
performance across different forecasting horizons", "a 60-day forecast every Monday").

Medido (s4.py, bike_sharing horario, LGBM lags=24, `steps=24`, test febrero y marzo 2012):

| fold_stride | MAE devuelto | MAE de todas las predicciones | MAE paso 1 | MAE paso 24 |
|---|---|---|---|---|
| 24 | 67.4 | 67.4 | 11.2 | 25.0 |
| 6 | 45.1 | 62.7 | 36.2 | 77.3 |
| 1 | 32.5 | 63.2 | 32.5 | 70.5 |

Con `fold_stride=1` la métrica devuelta es exactamente el MAE del paso 1 (32.5) y la
mitad del error real del problema a 24 h (63.2).

El sesgo cambia la selección de modelos (s7.py, misma configuración, `fold_stride=1`):

| Modelo | MAE devuelto | MAE en todo el horizonte |
|---|---|---|
| lags=24 | 32.5 | 63.2 |
| lags=[1, 2, 24, 168] | 39.6 | **60.1** |
| lags=3 | 43.5 | 129.7 |

`grid_search_forecaster` con `lags_grid=[3, 24, [1, 2, 24, 168]]` elige `lags=24`,
cuando para el pronóstico a 24 h el mejor es `[1, 2, 24, 168]`; y `lags=3` parece solo un
34% peor cuando en realidad duplica el error.

Antes (sin aviso):
```python
cv = TimeSeriesFold(steps=24, initial_train_size=n_train, refit=False, fold_stride=1)
metric, preds = backtesting_forecaster(forecaster, y=y, cv=cv, metric="mean_absolute_error")
# metric = 32.5 (error a 1 paso)
```
Después (propuesta):
```python
cv = TimeSeriesFold(steps=24, initial_train_size=n_train, refit=False, fold_stride=1)
metric, preds = backtesting_forecaster(forecaster, y=y, cv=cv, metric="mean_absolute_error")
# metric = 63.2: todas las predicciones (cada par origen-fecha) cuentan
# o, si se mantiene el comportamiento actual por defecto:
# UserWarning: fold_stride (1) < steps (24): la métrica usa solo la última predicción
# de cada fecha (horizontes 1 a 1). Use overlapping_metric='all' para evaluar los
# 24 pasos.
```
Cambio: opción (p. ej. en `TimeSeriesFold` o en backtesting) `'last'` (actual) frente a
`'all'` (estándar de rolling origin, como `cross_validation` + `performance_metrics` de
Prophet o `evaluate` de sktime), aviso mientras el valor por defecto sea `'last'`, y
decidir si `'all'` pasa a ser el valor por defecto (cambio de resultados, anunciarlo).
El caso de meses naturales con `skip_folds` (guía, #1401) es el único que se apoya en
`'last'`. Complementa D1119: la columna de paso permite ver el problema, pero la
métrica que usan las búsquedas sigue sesgada.

### 2. Métricas por fold, por paso y por serie a partir de las predicciones de backtesting (quick win, M; amplía D1119, GAP-backtesting-interval-metrics e #1409)

Hoy la única salida es una métrica global (o por serie). Para ver la estabilidad por
fold ([#720](https://github.com/skforecast/skforecast/issues/720), respondida con
"calcúlelo a mano"), el error por paso (D1119) o la cobertura por serie, el usuario
reconstruye el valor real, el paso y los grupos. La guía oficial
`docs/user_guides/comparing-forecasting-models.ipynb` (celda 5) define una función
`summarize` de unas 40 líneas para MAE, MASE por serie, cobertura y tiempo, y la de
backtesting (sección de meses naturales) pide "aplicar el mismo filtro" a mano. En
`docs/user_guides` hay al menos 18 usos del patrón `data.loc[predictions.index, col]`.

Antes (s8.py, multiserie con intervalo y `fold_stride=3`, 15 líneas de pandas):
```python
preds = preds.copy()
preds["step"] = preds.groupby(["fold", "level"]).cumcount() + 1
y_long = series.stack().rename("y_true")
y_long.index.names = ["date", "level"]
preds = preds.rename_axis("date").reset_index().merge(y_long.reset_index(), on=["date", "level"])
rows = []
for (level, step), g in preds.groupby(["level", "step"]):
    rows.append({"level": level, "step": step,
                 "mae": mean_absolute_error(g["y_true"], g["pred"]),
                 "coverage": calculate_coverage(g["y_true"], g["lower_bound"], g["upper_bound"]),
                 "width": (g["upper_bound"] - g["lower_bound"]).mean()})
by_step = pd.DataFrame(rows)
by_fold = preds.groupby("fold").apply(
    lambda g: mean_absolute_error(g["y_true"], g["pred"]), include_groups=False
)
```
Después:
```python
from skforecast.model_selection import evaluate_backtesting  # nombre a decidir

by_step = evaluate_backtesting(preds, series, metric="mean_absolute_error", by=["level", "step"])
by_fold = evaluate_backtesting(preds, series, metric="mean_absolute_error", by="fold")
# Si preds tiene lower_bound/upper_bound o q_*, añade coverage, interval_width y WIS/CRPS.
```
Cambio: una función pública que reciba la salida de cualquier `backtesting_*`
(forecaster, multiserie, stats, foundation), una `y`/`series` y `by` en
{`fold`, `step`, `level`, columnas propias}; añade `y_true`, `step` (posición en el fold
más `gap`) y opcionalmente el origen del fold. Reutiliza `_get_metric`,
`calculate_coverage`, `crps_from_quantiles`, `weighted_interval_score`. No cambia la
firma de los backtesting (la tupla actual se mantiene). Afecta al flujo habitual:
todo usuario que compara modelos o reporta resultados por horizonte.

### 3. Las búsquedas eligen el peor modelo con métricas donde mayor es mejor (`r2_score`) (bug silencioso, S)

El orden se decide por tipo de tarea, no por métrica: regresión ordena ascendente y
clasificación descendente (`skforecast/model_selection/_search.py:514, 916, 1537, 2049,
2427`; Optuna `direction` en `:846, :1953, :2369`). El docstring de `metric` no dice que
un callable deba ser una pérdida (`_search.py:85-93`). `grid/random_search` no tienen
forma de cambiar la dirección; `bayesian_search` sí, vía `kwargs_create_study`.

Reproducido (s6.py, h2o, Ridge, `metric=[r2_score, "mean_absolute_error"]`,
`return_best=True`):
```
`Forecaster` refitted using the best-found lags and parameters ...
  Parameters: {'alpha': 100}
  Backtesting metric: -0.3693007243109365
            params  r2_score  mean_absolute_error
0   {'alpha': 100} -0.369301             0.212630
1     {'alpha': 1}  0.822487             0.079045
2  {'alpha': 0.01}  0.838559             0.071935
forecaster alpha after return_best: 100
```
`bayesian_search_forecaster` con `r2_score` minimiza el R2 y explora `alpha` cerca de
1000 (R2 = -0.62). El caso simétrico afecta a clasificadores con pérdidas
(`log_loss`, `zero_one_loss`, `hamming_loss`), que se ordenarían descendente.

Cambio: deducir la dirección del nombre de la primera métrica (lista de
`*_score` de sklearn donde mayor es mejor: `r2_score`, `explained_variance_score`,
`d2_*`, y pérdidas de clasificación) y, como mínimo, un parámetro `greater_is_better`
o un aviso cuando el nombre termina en `_score` en regresión. Usuarios: nicho
moderado (R2 es habitual entre usuarios de scikit-learn), pero el resultado es
silenciosamente erróneo y `return_best=True` deja el forecaster con el peor modelo.

### 4. `'root_mean_squared_error'` no se acepta como string, y los nombres de pinball se repiten (quick win, S)

`_get_metric` (`skforecast/metrics/metrics.py:45-62`) acepta
`'root_mean_squared_scaled_error'` pero no `'root_mean_squared_error'`, la métrica más
usada junto al MAE (scikit-learn la tiene desde la 1.4; el mínimo es 1.6). Medido:
`metric='root_mean_squared_error'` lanza `ValueError: Allowed metrics are: [...]`.
Además `create_mean_pinball_loss` devuelve siempre `__name__ = 'mean_pinball_loss_q'`
(`metrics.py:555`): con dos cuantiles, `backtesting_forecaster` devuelve dos columnas
con el mismo nombre (observado en s2.py) y las búsquedas fallan con
`ValueError: When metric is a list, each metric name must be unique` (s3.py).

Antes:
```python
from sklearn.metrics import root_mean_squared_error
metric, preds = backtesting_forecaster(..., metric=[root_mean_squared_error, "mean_absolute_error"])
```
Después:
```python
metric, preds = backtesting_forecaster(..., metric=["root_mean_squared_error", "mean_absolute_error"])
# y create_mean_pinball_loss(0.1).__name__ == "mean_pinball_loss_q_0.1"
```
Cambio: añadir la entrada al diccionario de `_get_metric`, docstrings y skill
`metric-selection`; fijar `mean_pinball_loss_q.__name__ = f"mean_pinball_loss_q_{alpha}"`.
Usuarios: flujo habitual (ahorro pequeño pero diario). Valorar en el mismo cambio un
`'weighted_absolute_percentage_error'` (WAPE): MAPE con ceros da 1.8e15 (medido) y es la
métrica estándar en demanda minorista.

### 5. Tiempo de cada combinación en los resultados de `grid_search`/`random_search` (quick win, S)

Las búsquedas no informan del coste de cada configuración; la guía
`comparing-forecasting-models.ipynb` cronometra a mano 13 bloques con
`time.perf_counter()`. Con `bayesian_search` el dato existe en
`study.trials_dataframe()["duration"]`, pero en grid/random no.

Antes:
```python
results = grid_search_forecaster(forecaster, y=y, cv=cv, param_grid={"n_estimators": [100, 500, 1000]}, metric="mean_absolute_error")
# para saber si 1000 árboles compensan hay que repetir cada combinación con un cronómetro
```
Después:
```python
results = grid_search_forecaster(...)
results[["params", "mean_absolute_error", "fit_predict_time"]]
```
Cambio: medir con `perf_counter` cada llamada a backtesting dentro del bucle de
`_evaluate_grid_hyperparameters` (y multiserie, stats) y añadir la columna; en bayesiano,
copiar la duración del trial. Usuarios: nicho moderado (quien ajusta modelos de
boosting grandes).

## Fricciones observadas (no llegan a candidato)

- `verbose=True` en `grid_search_forecaster` imprime el bloque "Information of folds"
  una vez por combinación (3 veces con 3 combinaciones en s6.py): con 100 combinaciones
  es ruido. Bastaría imprimirlo una vez.
- MASE con `initial_train_size=None` (forecaster ya entrenado) devuelve `NaN` con solo
  un `RuntimeWarning: Mean of empty slice` de numpy (s9.py). Modo poco habitual.
- El MASE de backtesting excluye los primeros `window_size` valores del denominador y,
  con `refit=True`, usa la unión de todos los entrenamientos: 0.848 frente a 0.863 con
  `y_train=y[:150]` (refit=False) y 0.798 frente a 0.876 (refit=True) (s9.py). La guía
  de comparación calcula MASE con `y_train = series[level].iloc[:n_train]`, así que sus
  números no coinciden con los de `metric`. Documentar la definición.
- `grid_search_forecaster_multiseries` solo devuelve métricas agregadas: para ver el
  error por serie de la mejor configuración hay que relanzar el backtesting.
- La guía de migración desde Prophet tiene que construir los cutoffs a mano con
  `pd.date_range` para cuadrar con `TimeSeriesFold`; una columna de origen del fold en
  las predicciones (o en `cv.split(as_pandas=True)`) lo evitaría.
- Las predicciones no incluyen el valor real; casi todo análisis posterior empieza con
  `y.loc[preds.index]` (más complejo en multiserie larga y con folds solapados, donde
  el índice se repite).
