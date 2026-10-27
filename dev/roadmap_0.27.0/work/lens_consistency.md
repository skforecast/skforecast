# Lente: consistencia ML vs Foundation

Comparación de `ForecasterRecursive`, `ForecasterDirect` y `ForecasterRecursiveMultiSeries`
con `ForecasterFoundation` (rama 0.26.x). Para foundation se usaron los adaptadores falsos
de la suite de tests (`FakePipeline` de Chronos-2) y un `NoriRegressor` falso propio cuya
distribución predictiva es la distribución empírica del contexto (asimétrica).

Scripts (todos ejecutados): `/tmp/claude-0/-home-user-skforecast/8505cf39-716d-5c4c-adf6-669c7ec65b16/scratchpad/consistency/`

- `s1_outputs.py`: columnas y formato de `predict*`, `steps` como fecha, `last_window`, `levels` desconocidos.
- `s2_mean_median.py`: punto de `predict()` frente a `predict_interval()` y métrica de `backtesting_foundation` con y sin `quantiles`.
- `s3_backtest.py`: columnas de backtesting y errores al usar la función equivocada.
- `s4_exog.py`: manejo de `exog` en `predict` (ausente, completa, corta, columnas distintas).
- `s5_direct_date.py`: `steps` como fecha en Direct y MultiSeries.
- `s6_wrong_fn.py`: búsquedas ML con un `ForecasterFoundation`.

## Candidatos (ordenados por valor)

### 1. [bug, M] `pred` cambia de significado entre `predict()` y `predict_interval()` / backtesting con `quantiles`

- `ForecasterFoundation.predict()` pide al adaptador el punto (`quantiles=None`), que en
  TabICL (`point_estimate='mean'` por defecto, `_adapters.py:2147,2529`), Nori
  (`'mean'` por defecto, `_adapters.py:4421,4827`), TabPFN con `point_estimate` distinto de
  `'median'` y TimesFM 2.5 (`point_forecast`, documentado como media, `_adapters.py:660-664,916-922`)
  es la **media**.
- `predict_interval()` construye `pred` con `q_0.5` (`_forecaster_foundation.py:942-959`),
  es decir la **mediana**, siempre.
- `_backtesting_foundation` calcula la métrica sobre `pred` sin `quantiles` y sobre `q_0.5`
  con `quantiles` (`_validation.py:2645-2648,2674-2679`), y con `quantiles` elimina la columna
  `pred` de la salida.
- `bayesian_search_foundation` evalúa sin `quantiles` (`_search.py:2328`), y la skill
  lista `point_estimate` como hiperparámetro ajustable (`skills/foundation-forecasting/references/adapter-parameters.md:194-195`):
  el `point_estimate` elegido en la búsqueda se ignora en cuanto el usuario pide intervalos.
- En ML, `predict_interval()['pred']` es exactamente `predict()` y `backtesting_*` con
  `interval` conserva `pred` y mide sobre él.
- El docstring de `FoundationModel.predict` (`_foundation_model.py:1041,1058`) dice
  "point forecast (median)", falso para esos adaptadores.

Medido (`s2_mean_median.py`, serie lognormal, Nori con `point_estimate='mean'`):

```
predict()['pred']          : [1.438, 1.438, 1.438]
predict_interval()['pred'] : [1.003, 1.003, 1.003]
MAE backtesting sin quantiles: 1.1693
MAE backtesting con quantiles: 1.0834
cols sin quantiles: ['level', 'fold', 'pred']  con quantiles: ['level', 'fold', 'q_0.1', 'q_0.5', 'q_0.9']
```

La misma configuración da dos MAE distintos según se pidan cuantiles o no: resultado
silenciosamente distinto en un flujo habitual (comparar modelos con y sin intervalos).

Propuesta: que el adaptador devuelva el punto junto a los cuantiles cuando se le pidan
(TabICL `predict_df` ya devuelve `target` y cuantiles en la misma llamada; TimesFM 2.5 devuelve
`point_forecast` y `quantile_forecast` juntos; Nori necesita una segunda llamada barata). Con
eso: `predict_interval()['pred'] == predict()['pred']`, `backtesting_foundation(quantiles=...)`
conserva `pred` y mide sobre él, como en ML. Corregir el docstring. Completa
`GAP-foundation-backtesting-interval`: al añadir `interval` a `backtesting_foundation`,
`pred` debe ser el punto de `predict()`.

### 2. [quick win, S] Mensaje claro al usar una función de ML con `ForecasterFoundation`

`grid_search_forecaster`, `random_search_forecaster` y `grid_search_forecaster_multiseries`
fallan con `AttributeError: 'ForecasterFoundation' object has no attribute 'lags'`
(`s6_wrong_fn.py`). `bayesian_search_forecaster(_multiseries)` arranca el estudio de Optuna
y falla dentro del trial con traza larga. `backtesting_forecaster`,
`backtesting_forecaster_multiseries` y `backtesting_stats` dicen "use the other functions
available in the `model_selection` module" sin nombrar cuál (`_validation.py:869,1698,2321`,
`_search.py:2946`). Es el primer intento natural de quien viene de ML.

Antes:

```python
grid_search_forecaster(forecaster_foundation, y, cv, param_grid, metric)
# AttributeError: 'ForecasterFoundation' object has no attribute 'lags'
```

Después:

```python
grid_search_forecaster(forecaster_foundation, y, cv, param_grid, metric)
# TypeError: `forecaster` must be ... Got ForecasterFoundation.
# Use `backtesting_foundation` or `bayesian_search_foundation` instead.
```

Un helper con el mapa tipo de forecaster a función recomendada, llamado al principio de
cada función pública de `model_selection` (antes de crear el estudio de Optuna).

### 3. [quick win, S] Avisar cuando se entrenó con `exog` y `predict` no recibe `exog` (Chronos-2, TimesFM 3, TS-ICL)

En ML, olvidar la `exog` futura lanza `ValueError` ("Forecaster trained with exogenous
variable/s. Same variable/s must be provided when predicting."). En foundation, con los
adaptadores que admiten covariables solo pasadas (`supports_past_only_covariates=True`:
Chronos-2 `_adapters.py:170`, TimesFM 3 `:1154`, TS-ICL `:4028`), la predicción sale sin
error ni aviso y las covariables pasan a usarse solo como pasadas
(`_foundation_model.py:973-975` solo avisa a los adaptadores sin soporte). Medido en
`s4_exog.py`: `FM predict sin exog: OK shape=(3, 2)`, cero avisos. Cambia el modelo en
silencio (la previsión de temperatura o promociones deja de usarse).

Antes:

```python
forecaster.fit(series=y_train, exog=exog_train)
pred = forecaster.predict(steps=7)   # olvida exog: sin aviso, exog solo como pasado
```

Después:

```python
pred = forecaster.predict(steps=7)
# IgnoredArgumentWarning: Forecaster fitted with exog ['temp', 'promo'] but `exog` is
# None: they are used only as past covariates. Pass `exog` with future values to use them.
```

Solo cuando ninguna columna histórica tiene valores futuros (caso típico de olvido); no
afecta al backtesting, que siempre pasa la `exog` del test.

## Fricciones observadas (no priorizadas)

- **Formato de salida en una sola serie**: `ForecasterFoundation.predict(...)` con un
  `pd.Series` devuelve `DataFrame['level', 'pred']`; ML devuelve `Series 'pred'`.
  `predict_interval` añade `level`; `backtesting_foundation` de una serie devuelve
  `['level', 'fold', 'pred']` frente a `['fold', 'pred']` de `backtesting_forecaster`. Está
  documentado en la guía y cambiarlo rompe código: no se propone, pero es la fricción
  más visible al comparar ML y foundation en una sola serie.
- **Intervalo por defecto distinto**: ML `interval=[0.05, 0.95]` y `quantiles=[0.05, 0.5, 0.95]`;
  foundation `[0.1, 0.9]` y `[0.1, 0.5, 0.9]` (`_forecaster_foundation.py:861,981`). Se explica
  porque TimesFM 2.5 solo tiene deciles, pero comparar anchuras con valores por defecto
  compara un 90% con un 80%. Bastaría con mencionarlo en la docstring y en la guía.
- **Nombres de parámetros**: `fit(series=...)` incluso para una serie (ML `fit(y=...)`),
  `context` en lugar de `last_window` en `predict` (aunque existe el alias `last_window_`),
  `backtesting_foundation(series=...)` frente a `backtesting_forecaster(y=...)`.
- **`steps` como fecha**: `ForecasterRecursive.predict(steps='2025-03-01')` funciona;
  foundation lanza `ValueError: steps must be a positive integer`. MultiSeries y Direct
  tampoco lo admiten, así que no es una asimetría propia de foundation.
- **Sin `suppress_warnings` en `predict*` de foundation** (ML Recursive y MultiSeries lo
  tienen): un bucle de predicciones con `exog` corta repite `MissingValuesWarning`.
- **`backtesting_foundation(quantiles=[0.1, 0.9])` añade 0.5 con `IgnoredArgumentWarning`**:
  la categoría no encaja (no se ignora nada, se añade) y desaparecería con el candidato 1.
- **`save_forecaster` con foundation**: ningún adaptador define `__getstate__`, así que
  tras un `predict` se serializan los pesos cargados (`_pipeline` / `_model`). Puede ser
  deseable (despliegue offline) o no (ficheros de cientos de MB); no medido con un modelo real.
- **Mensaje de `levels` inexistente en ML multiserie**: "None of the series {'zzz'} are
  present in `last_window_` attribute. Provide `last_window`..." sugiere pasar
  `last_window` cuando el problema es un nombre mal escrito; foundation da un mensaje mejor
  (`levels ['zzz'] not found in available series ['a', 'b']`).

## Lo que es consistente (comprobado)

- Columnas de multiserie: `level`, `pred`, `lower_bound`, `upper_bound`, `q_<q>` iguales
  en ML multiserie y foundation; backtesting multiserie `['level', 'fold', ...]` y métricas
  con columna `levels` y agregados iguales.
- `interval` como float o lista de cuantiles con la misma validación (`check_interval`).
- `set_params` deja el forecaster sin ajustar en ambos.
- `bayesian_search_foundation` replica la firma de `bayesian_search_forecaster_multiseries`
  (salvo `n_jobs`), con salida de resultados equivalente.

## No cubierto

- Sin torch ni backends reales: la magnitud de la diferencia media/mediana con TabICL,
  Nori o TabPFN reales no se ha medido; la evidencia es el camino de código y el falso.
- Deep learning fuera de alcance.
