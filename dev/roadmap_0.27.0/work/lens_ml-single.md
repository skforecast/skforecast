# Lente ml-single: usuario de ML con una sola serie

Entorno: skforecast 0.26.0 (rama 0.26.x), Python 3.12.3, numpy 2.5.4, pandas 2.3.3,
scikit-learn 1.9.1, lightgbm instalado. Datos: `fetch_dataset("h2o_exog")` y
`fetch_dataset("bike_sharing")` (la red funcionó). Scripts en
`/tmp/claude-0/-home-user-skforecast/8505cf39-716d-5c4c-adf6-669c7ec65b16/scratchpad/ml-single/` (s1.py a s14.py).

Flujos reproducidos: ForecasterRecursive y ForecasterDirect con exógenas,
RollingFeatures, transformer_y/transformer_exog, differentiation,
backtesting_forecaster con TimeSeriesFold (intervalos bootstrapping y conformal,
cuantiles, distribución, bootstrapping crudo), grid_search_forecaster con
lags_grid, bayesian_search_forecaster con lags en el espacio de búsqueda,
predict_interval/predict_quantiles/predict_dist, residuos fuera de muestra con
set_out_sample_residuals.

## Candidatos (ordenados por valor)

### 1. Fuga silenciosa: backtesting con `initial_train_size` por defecto (None) sobre un forecaster ya entrenado con los mismos datos (bug, S)

`TimeSeriesFold(steps=12)` deja `initial_train_size=None` por defecto
(`skforecast/model_selection/_split.py:105`). Con un forecaster ya ajustado,
backtesting no reentrena (`_validation.py:580`) y evalúa sobre datos que el
modelo ya vio, sin ningún aviso. Flujo típico: `forecaster.fit(y)` y luego
backtesting olvidando `initial_train_size`.

Reproducido (s13.py, h2o, LGBM, lags=12):

```
warnings: []
MAE correct (train 150, test 45): 0.0826
MAE with TimeSeriesFold(steps=12) ... on already-fitted forecaster: 0.0036  n preds 183
training range [1992-04-01, 2008-06-01]
```

Métrica 23 veces mejor que la real y ningún aviso. El modo `initial_train_size=None`
es legítimo (la guía de backtesting lo usa para obtener predicciones en el
entrenamiento, `docs/user_guides/backtesting.ipynb`), así que no debe ser error,
pero sí un aviso cuando las fechas predichas caen dentro de
`forecaster.training_range_`.

Antes:
```python
forecaster.fit(y=data["y"])
cv = TimeSeriesFold(steps=12)  # se olvida initial_train_size
metric, preds = backtesting_forecaster(forecaster, y=data["y"], cv=cv,
                                       metric="mean_absolute_error")
# MAE 0.0036, sin aviso
```
Después (mismo código):
```
DataLeakageWarning: 183 de 183 predicciones caen dentro del periodo de
entrenamiento del forecaster (1992-04-01 a 2008-06-01): las métricas son
in-sample. Indique `initial_train_size` en TimeSeriesFold para evaluar fuera
de muestra.
```
Cambio: en `_backtesting_forecaster` (y la versión multiserie, `_validation.py:1420`)
cuando `initial_train_size is None`, comparar `forecaster.training_range_[-1]`
con el primer índice de test y avisar. Una clase de aviso nueva o
`ResidualsUsageWarning`/`UserWarning`.

### 2. `TimeSeriesFold(differentiation=...)` debe repetir la del forecaster, aunque backtesting la sobrescribe (quick win, S)

`check_backtesting_input` lanza error si `cv.differentiation` difiere de
`forecaster.differentiation_max` (`model_selection/_utils.py:270-288`; igual en
`check_one_step_ahead_input`, `_utils.py:602`), pero justo después
`_backtesting_forecaster` copia el cv y fija `differentiation` desde el forecaster
(`_validation.py:513-515`, `1286-1288`), igual que ya hace con `window_size`.
El parámetro del usuario es redundante. Además `tools/ai/llms-base.txt:340,370`
documenta `differentiation=None  # (set automatically by forecaster)`, así que el
código generado siguiendo la referencia falla. Las guías llevan el comentario
`# Must match the forecaster's differentiation`
(probabilistic-forecasting-conformal-calibration, probabilistic-forecasting-metrics).

Reproducido (s1.py, s2.py, s10.py): backtesting_forecaster lanza
`ValueError: The differentiation included in the forecaster (1) differs from the
differentiation included in the cv (None)`; bayesian_search falla en el trial 0;
grid_search salta todas las combinaciones (ver candidato 3).

Antes:
```python
forecaster = ForecasterRecursive(LGBMRegressor(), lags=24, differentiation=1)
cv = TimeSeriesFold(steps=24, initial_train_size=n_train,
                    differentiation=1)  # debe coincidir
```
Después:
```python
forecaster = ForecasterRecursive(LGBMRegressor(), lags=24, differentiation=1)
cv = TimeSeriesFold(steps=24, initial_train_size=n_train)
```
Cambio: si `cv.differentiation is None`, heredar la del forecaster (ya se hace);
error solo si el usuario puso un valor explícito distinto. Actualizar los
comentarios de las guías. Afecta a todo el que usa differentiation (flujo
habitual con árboles y series con tendencia).

### 3. Las búsquedas grid/random se tragan errores de configuración y devuelven un DataFrame vacío sin ajustar el forecaster (bug, S)

`_evaluate_grid_hyperparameters` captura cualquier excepción por combinación y
emite `RuntimeWarning: Parameters skipped` (`_search.py:474-476`). Si el error es
de configuración (igual para todas), el usuario recibe N avisos, un DataFrame vacío
y, aunque `return_best=True`, el forecaster queda sin ajustar.

Reproducido (s10.py, Ridge, differentiation=1, 10 valores de alpha):

```
grid: n warnings 11 result shape (0, 5)
cols ['lags', 'lags_label', 'params', 'mean_absolute_error', 'params']
forecaster fitted: False
```

Además la columna `params` sale duplicada porque el `pd.concat` con
`results['params'].apply(pd.Series)` se hace antes del `if results.empty`
(`_search.py:512-524`). El mismo patrón está en `_search.py:1481/1523`,
`2786/2824` y `3040/3088` (multiserie y stats). bayesian_search sí propaga el
error, así que el comportamiento es incoherente.

Cambio: guardar la primera excepción y, si todas las combinaciones fallan,
relanzarla (`raise ... from e`) en lugar de devolver vacío; mover el chequeo de
vacío antes del concat.

### 4. ForecasterRecursive/Direct no alinean `exog` por índice en predict, ForecasterRecursiveMultiSeries sí (quick win, S)

`check_predict_input` solo activa `align_by_index` para
ForecasterRecursiveMultiSeries (`utils/utils.py:1656`); en el resto exige que
`exog` empiece exactamente en el paso siguiente a `last_window`
(`utils/utils.py:1738-1745`). En producción el usuario tiene un único DataFrame de
exógenas (calendario, previsiones meteo) que cubre pasado y futuro, y debe
recortarlo a mano.

Reproducido (s11.py, s12.py, s14.py): con `exog=data[exog_cols]` (histórico + futuro)
MultiSeries predice 0.8915, 1.0044, 1.0904; Recursive y Direct lanzan
`ValueError: To make predictions exog must start one step ahead of last_window`.
Con el recorte manual, Recursive devuelve exactamente los mismos valores.

Antes:
```python
start = forecaster.last_window_.index[-1] + forecaster.last_window_.index.freq
predictions = forecaster.predict(steps=24, exog=exog_all.loc[start:])
```
Después:
```python
predictions = forecaster.predict(steps=24, exog=exog_all)
```
Cambio: si el índice de `exog` contiene `expected_index[0]`, seleccionar
`exog.loc[expected_index]` en Recursive/Direct (comportamiento idéntico cuando
ya empieza bien, solo añade casos que hoy fallan). Opcional: lo mismo en `fit`
cuando `exog` es más larga que `y` y contiene su índice (hoy error de longitud,
s11.py). Flujo habitual en producción.

### 5. Coste de calibración invisible: intervalos con residuos in-sample muy por debajo de la cobertura nominal (quick win, S; amplía GAP-backtesting-interval-metrics / #1409)

`use_in_sample_residuals=True` es el valor por defecto y, con modelos de alta
capacidad, los intervalos quedan muy estrechos. Medido (s9.py, bike_sharing últimos
120 días, lags=24, intervalo nominal 80%, 60/20/20):

| Modelo | Método | Cobertura in-sample | Cobertura out-sample |
|---|---|---|---|
| Ridge | bootstrapping | 0.81 | 0.93 |
| Ridge | conformal | 0.65 | 0.75 |
| LGBM | bootstrapping | 0.53 | 0.84 |

En h2o con ForecasterDirect + LGBM (s3.py): 0.22 (bootstrapping) y 0.27 (conformal)
frente al 80% nominal. Nada en la salida lo indica.

Antes:
```python
_, preds = backtesting_forecaster(..., interval=[0.1, 0.9])
cov = calculate_coverage(y.loc[preds.index], preds.lower_bound, preds.upper_bound)
```
Después (una vez integrada la cobertura en las métricas, #1409):
```
ResidualsUsageWarning: cobertura empírica 0.53 frente a 0.80 nominal con
residuos in-sample. Use set_out_sample_residuals() y
use_in_sample_residuals=False.
```

### 6. Mensaje de error sin frecuencia: sugerir la frecuencia inferida o el primer hueco (quick win, S)

`check_extract_values_and_index` (`utils/utils.py:2046-2050`) dice solo "set the
frequency". En el caso típico (CSV con `parse_dates`) `pd.infer_freq` devuelve la
frecuencia ('MS' en h2o, s14.py); cuando falta una fecha, el problema real es un
hueco y el mensaje es el mismo (s4.py).

Antes: `ValueError: y has a pandas DatetimeIndex without a frequency. To avoid
this error, set the frequency of the DatetimeIndex.`
Después: `... Inferred frequency: 'MS'. Use y = y.asfreq('MS').` o
`... The index has gaps (first: 1996-06-01 missing); frequency cannot be inferred.`

### 7. Error de `interval` en percentiles sin pista de migración (quick win, S)

`check_interval` (`utils/utils.py:1222-1229`) responde
`Lower interval bound (10) must be >= 0 and < 1.` (s1.py). Código antiguo, blogs y
código generado por LLM siguen usando `[10, 90]`. Añadir: "Since 0.23 `interval`
is expressed as quantiles, e.g. interval=[0.1, 0.9]."

## Otras fricciones observadas (no priorizadas)

- `predict_interval` tras `fit()` falla porque `store_in_sample_residuals=False`
  por defecto; el mensaje es claro pero obliga a reentrenar (s1.py). Las búsquedas con
  `return_best=True` sí guardan residuos: incoherencia menor.
- Valor por defecto de `interval_method`: 'bootstrapping' en backtesting_forecaster,
  'conformal' en backtesting_forecaster_multiseries (`_validation.py:708,1525`).
- La salida de backtesting no trae el valor real: 19 apariciones de
  `data.loc[predictions.index, ...]` en 7 guías. Encaja con la columna de horizonte
  (DISC-1119).
- NaN intercalados en `y` con Ridge: dos MissingValuesWarning y después el error de
  scikit-learn "Input X contains NaN" (s5.py); aceptable.
- Los errores de alineación de `exog` en predict son claros y útiles.
- grid_search con `lags_grid=[6, 12, {...}]` (dict dentro de lista) da TypeError de
  `initialize_lags`; aceptable, la API pide un dict.
