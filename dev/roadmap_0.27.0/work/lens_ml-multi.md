# Lente: usuario ML multiserie (ForecasterRecursiveMultiSeries, ForecasterDirectMultiVariate)

Fecha: 2026-10-17. Rama `0.26.x` (skforecast 0.26.0). Entorno: Python 3.12.3, numpy 2.5.4,
pandas 2.3.3, scikit-learn 1.9.1, lightgbm 4.7.0.

Scripts reproducibles en
`/tmp/claude-0/-home-user-skforecast/8505cf39-716d-5c4c-adf6-669c7ec65b16/scratchpad/ml-multi/`
(`s1.py` a `s22.py`). Datos: sintéticos y `fetch_dataset("store_sales")` (50 series, tiendas 1-2 x
artículos 1-25, convertidas a dict con `reshape_series_long_to_dict`).

Flujo recorrido: dict de series de distinta longitud, dict de exógenas por serie (con categóricas),
los cuatro `encoding`, series nuevas en `predict` con `last_window`, `transformer_series` y
`differentiation` por serie, `predict_interval`, `backtesting_forecaster_multiseries` con
intervalos, `grid_search_forecaster_multiseries` y `bayesian_search_forecaster_multiseries`
(TimeSeriesFold y OneStepAheadFold), `ForecasterDirectMultiVariate` con lags por serie.

Lo que funciona bien (sin fricción): backtesting de 50 series en 1 s; búsqueda bayesiana de 6 trials
en 8 s; `predict` de 1000 series x 28 pasos en 0,19 s (fit 5,1 s); los cuatro `encoding` con series
nuevas en `predict` (avisos claros); exógenas faltantes o cortas en `predict` (avisos claros y relleno
con NaN); categóricas nuevas en `predict` se codifican como NaN.

## Candidatos (ordenados por valor)

### 1. `grid_search`/`random_search` devuelven un DataFrame vacío ante errores de configuración (bug, S)

Las búsquedas por rejilla y aleatorias capturan cualquier excepción por combinación y la convierten en
`RuntimeWarning` (`skforecast/model_selection/_search.py:474-476`, `:1480-1482`, `:2786`, `:3040`).
Si el error es de configuración y afecta a todas las combinaciones, el usuario recibe un DataFrame
vacío (`shape (0, 7)`) y no se lanza nada; con `return_best=True` el forecaster queda sin cambiar
(`:518-523` devuelve antes). La búsqueda bayesiana, en cambio, sí lanza el error. Reproducido
(`s21.py`, `s22.py`):

- `ForecasterRecursiveMultiSeries(..., differentiation=1)` con `TimeSeriesFold(...)` sin
  `differentiation`: `grid multiseries: shape (0, 7)`; `grid single: (0, 5)`;
  `bayes multiseries ValueError ...`.
- `TimeSeriesFold(initial_train_size=500)` con 200 observaciones: `no exception, results shape (0, 7)`.

En OneStepAheadFold hay validación previa (`check_one_step_ahead_input`), pero en TimeSeriesFold no
se llama a `check_backtesting_input` antes del bucle. Arreglo: si todas las combinaciones fallan,
relanzar la primera excepción (encadenada) en vez de avisar; opcionalmente validar `cv` contra el
forecaster una vez antes del bucle.

### 2. `predict()` excluye en silencio las series cuya última ventana acaba antes (quick win, S)

Con `levels=None` (lo habitual), las series cuya última observación no coincide con la máxima se
quitan de la predicción sin aviso; el aviso solo sale si `levels` es una lista
(`skforecast/utils/utils.py:4680`). Caso real: en un panel de ventas, un artículo sin registro el
último día (ventas cero no registradas) desaparece del forecast. Medido con `store_sales` quitando la
fila del último día a 3 artículos (`s13.py`): `series trained: 50 | series predicted: 47`,
`warnings: []`.

Antes:

```python
p = forecaster.predict(steps=7)
faltan = set(forecaster.series_names_in_) - set(p["level"])  # hay que saber que pasa
```

Después:

```python
p = forecaster.predict(steps=7)
# IgnoredArgumentWarning: 3 series no se predicen porque su última ventana termina antes de
# '2017-12-31': {'s1_i3', 's2_i7', 's2_i20'} (última fecha de cada una ...). Use `levels` y
# `last_window` para predecirlas.
```

### 3. El formato largo con MultiIndex documentado falla con cualquier MultiIndex creado por el usuario (bug, S)

`check_preprocess_series` exige que cada grupo conserve la frecuencia del índice
(`skforecast/utils/utils.py:4173-4200`, `:4267-4274`). Eso solo ocurre si el MultiIndex sale de
`reshape_series_wide_to_long` (único caso en tests y en la guía, celda 6 de
`independent-multi-time-series-forecasting.ipynb`). Un `df.set_index(["series_id", "date"])` sobre
datos leídos de CSV o base de datos (formato natural de un panel) siempre falla, y el mensaje habla
de diccionarios: `If series is a dictionary, all series must have a Pandas RangeIndex or
DatetimeIndex with the same step/frequency...` (`s2.py`, `s3.py`; también con series alineadas). Con
`store_sales`, `levels[1].freq: None`, `inferred: {'D'}`: inferir la frecuencia por grupo basta para
que `fit` funcione (comprobado).

Antes:

```python
long = df.set_index(["series_id", "date"])[["sales"]]
forecaster.fit(series=long)  # ValueError engañoso
series = reshape_series_long_to_dict(df, series_id="series_id", index="date",
                                     values="sales", freq="D")
forecaster.fit(series=series)
```

Después:

```python
forecaster.fit(series=df.set_index(["series_id", "date"])[["sales"]])  # frecuencia inferida
```

Si la frecuencia no se puede inferir (huecos), mensaje específico de MultiIndex que sugiera
`reshape_series_long_to_dict(..., freq=...)`.

### 4. Columna con el valor real en las predicciones de backtesting (quick win, S)

`backtesting_forecaster_multiseries` devuelve `level, fold, pred[, lower_bound, upper_bound]` sin el
valor observado, aunque el cálculo de métricas ya hace ese merge
(`skforecast/model_selection/_utils.py:1150-1165`). Para graficar, analizar errores por segmento o
calcular métricas propias hay que reconstruirlo a mano (`s19.py`). Amplía
GAP-backtesting-interval-metrics: con `y_true` en la salida, cobertura, anchura y métricas por paso
salen directas.

Antes:

```python
_, preds = backtesting_forecaster_multiseries(forecaster, series=series, cv=cv, metric="mae")
y = pd.concat(series, names=["level", "date"]).rename("y_true").reset_index()
preds = (preds.rename_axis("date").reset_index()
              .merge(y, on=["level", "date"], how="left").set_index("date"))
```

Después:

```python
_, preds = backtesting_forecaster_multiseries(forecaster, series=series, cv=cv, metric="mae")
preds[["level", "fold", "y_true", "pred"]]
```

### 5. Propagar `differentiation` del forecaster a `TimeSeriesFold` (quick win, S)

Si el forecaster tiene `differentiation`, hay que repetirla en el `cv` o el backtesting lanza
`ValueError: The differentiation included in the forecaster (1) differs from the differentiation
included in the cv (None)` (`skforecast/model_selection/_utils.py:270-288`; `s10.py`, `s20.py`).
Con `differentiation` dict por serie hay que pasar `forecaster.differentiation_max`. Es información
redundante: si `cv.differentiation is None`, se puede tomar del forecaster (copia del `cv`). Esto
además elimina el caso más común del candidato 1. Transversal (también un solo serie).

Antes:

```python
forecaster = ForecasterRecursiveMultiSeries(LGBMRegressor(), lags=14,
                                            differentiation={"a": 1, "b": None, "_unknown_level": 1})
cv = TimeSeriesFold(steps=7, initial_train_size=300,
                    differentiation=forecaster.differentiation_max)
```

Después:

```python
cv = TimeSeriesFold(steps=7, initial_train_size=300)  # se toma del forecaster
```

### 6. Error accionable al pedir intervalos sin residuos en multiserie (quick win, S)

`fit()` tiene `store_in_sample_residuals=False` por defecto. En ForecasterRecursive y
ForecasterDirect, `predict_interval` explica qué hacer; en multiserie el mensaje no lo dice y nombra
una serie arbitraria o `_unknown_level` (`skforecast/utils/utils.py:1938-1947`; `s11.py`):

- RecursiveMultiSeries: `Residuals for level 'x1' are None or empty. Check forecaster.in_sample_residuals_by_bin_.`
- con `encoding=None`: `Residuals for level '_unknown_level' are None or empty...`
- DirectMultiVariate: `Residuals for level 'y' are None or empty...`
- Recursive: `... Use store_in_sample_residuals = True when fitting the forecaster or use the set_in_sample_residuals() method before predicting.`

Antes:

```python
forecaster.fit(series=series)
forecaster.predict_interval(steps=7)  # ValueError no accionable; buscar en la documentación
```

Después:

```python
forecaster.predict_interval(steps=7)
# ValueError: No in-sample residuals are stored. Use `store_in_sample_residuals=True` in fit()
# or call `set_in_sample_residuals()` before predicting.
```

### 7. Ruido de avisos por serie en paneles grandes (quick win, M)

- Un mismo problema genera un aviso por serie: con 200 series y exógenas cortas, `predict` emite 200
  `MissingValuesWarning` (cada uno un recuadro de ~10 líneas) (`skforecast/utils/utils.py:1674-1681`;
  `s6.py`); con 100 series, 101 avisos (`s15.py`).
- `fit()` ajusta un `QuantileBinner` por serie aunque no se pidan intervalos
  (`_forecaster_recursive_multiseries.py:2475-2496`) y cada uno puede avisar `The number of bins has
  been reduced from 10 to N` sin decir de qué serie se trata (`preprocessing/_preprocessing.py:2449-2461`).
  Con 100 series (la mitad intermitentes) y `fit()` por defecto: 5 avisos (`s16.py`); con "always",
  10 sobre 100 series (`s15.py`).

Antes: el usuario añade `warnings.simplefilter("ignore", ...)` y pierde también los avisos útiles.
Después: un aviso agregado por causa con la lista de series afectadas (truncada), y el aviso del
binner solo cuando se van a usar residuos binned, con el nombre de la serie.

## Otras fricciones observadas (no priorizadas)

- `transformer_series` o `differentiation` como dict exigen la clave mágica `'_unknown_level'` aunque
  nunca se predigan series nuevas (`_forecaster_recursive_multiseries.py:543-548`, `:581-586`). Es
  defendible (escala de series nuevas), pero es boilerplate y poco descubrible.
- `fit` acepta dict de series, pero `predict(last_window=...)` solo DataFrame ancho
  (`TypeError: last_window must be a pandas DataFrame. Got dict`; `s12.py`). Nicho.
- `ForecasterDirectMultiVariate` no acepta dict (`TypeError: series must be a pandas DataFrame`);
  limitación conocida. `training_range_` sale con `freq=None`.
- Los resultados de las búsquedas multiserie solo traen métricas agregadas, no por serie: para saber
  qué series van mal hay que repetir el backtesting con los mejores parámetros.
- MAPE con una serie de ceros da `2.9e14` y contamina `average`, `weighted_average` y `pooling` sin
  aviso más allá de un `RuntimeWarning` de numpy (propiedad de la métrica; nicho).
- Con LightGBM `n_jobs=-1` en una máquina compartida, la búsqueda bayesiana pasó de 8 s a más de 8 min
  por sobresuscripción de hilos; no es atribuible a skforecast, pero un consejo en la guía de
  rendimiento evitaría sustos.
- Ya identificados y confirmados de paso: serie sin predicciones en backtesting aparece con NaN en
  métricas sin aviso (#1406); intervalos y métricas por paso ausentes (GAP-backtesting-interval-metrics).
