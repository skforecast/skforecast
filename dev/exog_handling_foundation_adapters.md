# Gestión de variables exógenas en los adaptadores de `skforecast.foundation`

**Rama**: `refactor/timesfm-3.0`
**Ficheros analizados**: `skforecast/foundation/_forecaster_foundation.py`, `skforecast/foundation/_foundation_model.py`, `skforecast/foundation/_adapters.py`, `skforecast/foundation/_utils.py`, `skforecast/utils/utils.py` (`check_preprocess_series`, `check_preprocess_exog_multiseries`, `align_series_and_exog_multiseries`), `skforecast/model_selection/_validation.py` (`backtesting_foundation`), `skforecast/model_selection/_utils.py` (`_extract_data_folds_multiseries`).
**Backends contrastados**: `chronos-forecasting` (chronos2/dataset.py) y `tabicl` (forecast/forecaster.py) instalados en el entorno `skforecast_py14`. TabPFN-TS, T0, TS-ICL y Nori no están instalados; para ellos se describe lo que hace skforecast y lo que documenta el docstring del adaptador.
**Revisión relacionada**: `dev/review_timesfm_3.0_adapter_round2.md` (hallazgo 5.1 sobre covariables heterogéneas en TimesFM 3.0).

---

## 1. Flujo común a todos los escenarios

Hay tres capas con responsabilidades separadas. Toda la normalización y validación ocurre en la capa intermedia (`FoundationModel`), y los adaptadores reciben siempre el mismo formato canónico:

- `context`: `dict[str, pd.Series]`
- `context_exog`: `dict[str, pd.DataFrame | None] | None`
- `exog`: `dict[str, pd.DataFrame | None] | None`

### 1.1 Capa 1: `ForecasterFoundation`

Fichero: `_forecaster_foundation.py`, métodos `fit` (línea 607) y `predict` (línea 672).

- `fit(series, exog)`: si `exog` no es `None` y `estimator.allow_exog` es `False` (TimesFM 2.5, Moirai-2), emite `IgnoredArgumentWarning` y descarta `exog`. Después delega en `estimator.fit(series, exog)`.
- `predict`, `predict_interval`, `predict_quantiles`: comprueban `NotFittedError` (no ajustado y sin `context`) y delegan en `estimator.predict(steps, context, context_exog, exog, quantiles, levels, check_inputs)`.
- No valida nada sobre columnas ni tipos de exógenas. Todas las propiedades (`exog_in_`, `exog_names_in_`, `exog_names_in_per_series_`, `context_exog_`, ...) son delegaciones al estimador.

### 1.2 Capa 2: `FoundationModel`

Fichero: `_foundation_model.py`.

#### `fit` (línea 519)

1. Llama a `_check_preprocess_context(series, exog)` (línea 433):
   - `check_preprocess_series_foundation` (`_utils.py`): una `pd.Series` se envuelve en `{nombre: serie}` (nombre `'y'` si no tiene). DataFrame ancho, largo (MultiIndex) o dict se convierten a dict mediante `check_preprocess_series`. Exige `DatetimeIndex` con `freq` o `RangeIndex`, y la misma frecuencia en todas las series. Lanza `ValueError` si alguna serie es toda NaN.
   - Si `exog` no es `None`, `check_preprocess_exog_multiseries(series_names_in_, series_index_type, exog, exog_dict={name: None})` (`utils.py:3470`) normaliza a `{serie: DataFrame | None}`:
     - DataFrame o Series plano: se asigna a todas las series por referencia (`{sid: exog for sid in names}`). Debe tener el mismo tipo de índice que las series; si no, `TypeError`.
     - DataFrame largo (MultiIndex): se agrupa por nivel 0; solo se conservan las series conocidas. Emite `InputTypeWarning`.
     - dict: solo se usan las claves presentes en las series; el resto se ignora. Cada valor debe ser Series, DataFrame o `None`; si no, `TypeError`. Las series sin entrada quedan en `None` y se emite `MissingExogWarning` con el texto "All values ... will be NaN" (texto heredado de los forecasters clásicos; en foundation la serie simplemente no tiene bloque de exógenas).
     - Validaciones comunes: `check_exog(allow_nan=True)` por bloque; toda `pd.Series` se convierte a DataFrame de una columna; el tipo de índice debe coincidir con el de las series (`TypeError`); con dict, las columnas compartidas deben tener el mismo dtype en todas las series (`TypeError`); ninguna columna puede llamarse como una serie (`ValueError`).
     - `exog_names_in_` es la unión de columnas de todos los bloques (con dict) o las columnas del DataFrame plano.
   - `align_series_and_exog_multiseries(trim_series_nan=False)`: recorta cada bloque exog al rango de índice de su serie. Si queda vacío, pasa a `None` con `MissingValuesWarning`. Si le faltan fechas, se reindexa al índice de la serie rellenando con NaN, con `MissingValuesWarning`.
   - Ambos dicts se truncan a las últimas `context_length` observaciones (`iloc[-context_length:]`).
2. `adapter.fit(context, context_exog)`: guarda los dos dicts y marca `is_fitted`. Es idéntico en los ocho adaptadores. Moirai ignora `context_exog`.
3. Guarda metadatos: `series_names_in_`, `is_multiple_series_`, `exog_in_`, `exog_names_in_` (unión), `exog_names_in_per_series_` (dict con las columnas de cada serie, `None` para las que no tienen), `exog_type_in_`, `index_type_`, `index_freq_`, `context_range_`.

#### `predict` (línea 876)

1. Si no está ajustado y `context` es `None`: `ValueError`. Valida `steps` y `quantiles`.
2. Resolución del contexto:
   - `context=None`: usa `adapter.context_`, `adapter.context_exog_` y `series_names_in_`. Si el usuario pasó `context_exog`, se ignora con `IgnoredArgumentWarning`.
   - `context` dado y `check_inputs=True`: se repite `_check_preprocess_context(context, context_exog)`, con todas las validaciones de `fit`. Es el modo zero-shot sin `fit`.
   - `context` dado y `check_inputs=False`: los dicts se usan tal cual (ruta interna de backtesting).
3. `levels`: comprueba que existan (`ValueError` si no) y filtra `context` y `context_exog` a las series solicitadas.
4. Exógenas futuras:
   - Si `allow_exog` es `False`: si hay `exog` o `context_exog`, `IgnoredArgumentWarning` y ambos pasan a `None`.
   - Si `allow_exog` es `True` y `check_inputs=True`:
     - `_prepare_future_exog(steps, context, exog, series_names_in)` (línea 664). Con `exog=None` devuelve `{serie: None}`. Si no es Series, DataFrame o dict: `TypeError`. `_exog_to_dict` difunde un DataFrame o Series plano a todas las series, convierte un MultiIndex a dict por serie (con `InputTypeWarning`), y con un dict rellena `None` en las claves ausentes. Después, por serie: Series a DataFrame; con `DatetimeIndex` se reindexa a `pd.date_range(fin_contexto + freq, periods=steps, freq=freq)` (huecos a NaN, un único `MissingValuesWarning` con la lista de series afectadas); con `RangeIndex` u otro índice exige al menos `steps` filas (`ValueError`), que la primera fila sea `fin_contexto + step` (`ValueError`), y recorta a `iloc[:steps]`.
     - `_check_exog_columns(context_exog, exog, series_names_in)`: compara, serie a serie, el conjunto de columnas del contexto con el del futuro (sin importar el orden). Una columna futura sin histórico en la misma serie lanza `ValueError` para todos los adaptadores. Una columna histórica sin futuro es una covariable past-only: si `adapter.supports_past_only_covariates` es `True` (Chronos, TS-ICL, TimesFM 3.0) no pasa nada; si es `False` (TabICL, TabPFN, T0, Nori) se emite `IgnoredArgumentWarning` con las columnas y series afectadas.
   - Con `check_inputs=False` no se ejecuta ninguno de los dos pasos: `exog` llega al adaptador tal cual.
5. **La validación de nombres se hace contra el `context_exog` resuelto, no contra `exog_names_in_`.** `_prepare_future_exog` sigue siendo autocontenido ("does not depend on any metadata stored at fit time") para permitir el modo zero-shot sin `fit`. Con `context=None` el `context_exog` resuelto es el almacenado por `fit`, del que derivan `exog_names_in_per_series_`, así que el efecto es el mismo que validar contra los metadatos. La semántica de una columna de un solo lado la declara cada adaptador con `supports_past_only_covariates`.
6. `adapter.predict(steps, context, context_exog, exog, quantiles)` devuelve `{serie: ndarray(steps, n_q)}`. Se construye el DataFrame largo con columnas `level` y `pred` (o `q_*`), con índice expandido desde el final de cada contexto.

### 1.3 Capa 3: adaptadores

Fichero: `_adapters.py`. `fit` es idéntico en todos (guardar dicts). La diferencia está en `predict`: cómo traducen los dos dicts al formato del backend y cómo reconcilian columnas.

| Adaptador | `allow_exog` | Formato entregado al backend | dtypes aceptados | Reconciliación de columnas | Líneas |
|---|---|---|---|---|---|
| `ChronosAdapter` (Chronos-2) | Sí | Lista con un dict por serie: `target`, `past_covariates` (de `context_exog`) y `future_covariates` (de `exog`), cada uno `{columna: array 1-D}` | Numérico y bool a float32; string, object y categórico se pasan tal cual (Chronos-2 los trata como categóricos) | Ninguna en skforecast; cada serie es un input independiente. El backend (`chronos2/dataset.py:118`) exige que las claves de `future_covariates` sean subconjunto de las de `past_covariates`, y lanza `ValueError` si no | 273-481 |
| `TimesFMAdapter` v2.5 | No | Solo el target | - | - | 986-1048 |
| `TimesFMAdapter` v3.0 | Sí (`allow_exog` y `supports_past_only_covariates` son atributos de instancia, `True` en v3.0) | `past_only_covariates` y `past_future_covariates`: por serie, arrays `(n_cols, ctx)` y `(n_cols, ctx + steps)`; la parte known-future concatena histórico y futuro de la misma columna | Solo numérico y bool a float32; `ValueError` si no | **Por serie, con agrupación por firma**: `_v3_covariate_signature` calcula para cada serie `(past_only_cols, fut_cols)` con sus propias columnas (y lanza `ValueError` si una columna futura no tiene histórico); las series con la misma firma se envían juntas a `predict_batch` y las de firmas distintas en llamadas separadas. No se rellena nada con NaN. `padding_mode="edge"` cuando hay covariables | `_predict_v3`, `_v3_covariate_signature`, `_build_v3_covariates` |
| `MoiraiAdapter` | No | Solo el target | - | - | 1797-1872 |
| `TabICLAdapter` | Sí | `context_df` largo con `item_id`, `timestamp`, `target` y columnas extra; `future_df` largo con `item_id`, `timestamp` y columnas extra. Con `RangeIndex` se sintetiza un índice diario desde 2000-01-01 | Lo que acepte la librería | `pd.concat(ignore_index=True)` une las columnas entre series (NaN donde faltan). La librería (`_align_covariates`) usa solo la intersección contexto ∩ futuro del lote completo y registra un aviso si hay NaN en covariables futuras | 2237-2557 |
| `TabPFNAdapter` | Sí | Igual que TabICL | Lo que acepte la librería | Igual que TabICL en skforecast. Según el docstring, la librería descarta las covariables sin valores futuros (no verificado, librería no instalada) | 2840-3169 |
| `T0Adapter` | Sí | Target `(n_series, ctx_max)` con relleno NaN por la izquierda; `future_covariates` `(n_series, n_cov, ctx_max + steps)` con NaN donde no hay valor | Solo numérico y bool a float32; `ValueError` si no | Las columnas son la unión, en orden de aparición, de las columnas de `exog` futuro de todas las series. Por serie: futuro desde `exog`; histórico desde `context_exog` solo si la columna también está en el futuro, pegado al origen del pronóstico. Una serie sin `exog` futuro tiene la fila entera en NaN. T0 trata NaN como ausente. Si ninguna serie tiene `exog` futuro, `future_covariates=None` aunque haya `context_exog` | 3368-3620 |
| `TSICLAdapter` | Sí | Igual que Chronos: dict por serie con `target`, `past_covariates` y `future_covariates` | Solo numérico y bool a float32; `ValueError` si no | Ninguna; cada serie es independiente. Comportamiento del backend con columnas de un solo lado no verificado | 3837-4055 |
| `NoriAdapter` | Sí | Por serie, matriz tabular `X_ctx` `(ctx, n_feat)` y `X_fut` `(steps, n_feat)`: índice corrido, calendario, Fourier y columnas exógenas. `fit(X_ctx, y_ctx)` en contexto y `predict(X_fut)` | Solo numérico y bool a float32; `ValueError` si no | **Por serie**: `_known_future_columns` devuelve la intersección entre las columnas de `context_exog` y `exog` de esa misma serie, en el orden del contexto. Si cualquiera de los dos es `None`, lista vacía y no se usan covariables | 4332-4694 |

Una columna presente solo en el futuro ya no llega a ningún adaptador por la ruta pública: `_check_exog_columns` la rechaza con `ValueError`. Ante una columna presente solo en el contexto quedan dos grupos, declarados por cada adaptador con el atributo de clase `supports_past_only_covariates`:

1. **La usan como past-only** (`supports_past_only_covariates=True`): Chronos, TS-ICL, TimesFM 3.0.
2. **La descartan** (`supports_past_only_covariates=False`): TabICL, TabPFN (por intersección a nivel de lote, en la librería), Nori (por intersección por serie, en skforecast), T0 (solo usa columnas con futuro). `FoundationModel.predict` avisa con `IgnoredArgumentWarning` antes de llamarlos.

---

## 2. Escenario 1: serie única

Flujo del usuario:

```python
forecaster = ForecasterFoundation(estimator=FoundationModel(model_id, context_length=...))
forecaster.fit(y, exog=exog_hist)
pred = forecaster.predict(steps=12, exog=exog_fut)
```

1. `ForecasterFoundation.fit` comprueba `allow_exog` y delega.
2. `FoundationModel.fit` envuelve `y` en `{'y': y}` (o el nombre de la serie). `exog_hist` se convierte en `{'y': DataFrame}` (una Series pasa a DataFrame de una columna). Se comprueba que el índice es del mismo tipo que el de `y`, se recorta al rango de `y` (reindexando con NaN si faltan fechas), y ambos se truncan a `context_length`.
3. El adaptador guarda `context_` y `context_exog_`. `exog_names_in_` recoge las columnas de `exog_hist`; `exog_names_in_per_series_` es `{'y': [...]}`.
4. En `predict`, sin `context`, se usa lo almacenado. `exog_fut` pasa por `_prepare_future_exog`: se convierte en `{'y': DataFrame}` y se reindexa a los 12 timestamps que siguen al final del contexto. Filas sobrantes se recortan. Fechas ausentes se rellenan con NaN y se avisa. Con `RangeIndex`, se exige que empiece justo después del contexto.
5. `_warn_covariate_column_divergence` no dispara si las columnas coinciden.
6. Cada adaptador construye su input:
   - Chronos y TS-ICL: `past_covariates` con las columnas históricas y `future_covariates` con las futuras.
   - TimesFM 3.0: por cada columna de `exog_fut`, un array `past_future` concatenando histórico y futuro; columnas solo en histórico van a `past_only`.
   - TabICL y TabPFN: un único `item_id` en `context_df` y `future_df`.
   - T0: array `(1, n_cols, ctx + 12)`.
   - Nori: `X_ctx` y `X_fut`, reajuste en contexto y predicción.

En este escenario los siete adaptadores con soporte se comportan como se espera, porque no hay nada que reconciliar.

Variante: `fit(y)` sin exog y `predict(exog=exog_fut)`. `_check_exog_columns` lanza `ValueError` para todos los adaptadores antes de llegar al backend. Ver escenario 4a.

---

## 3. Escenario 2: múltiples series con las mismas exógenas

Flujo del usuario: `series` como DataFrame ancho o dict, `exog` como un solo DataFrame plano (se difunde a todas) o un dict con el mismo conjunto de columnas por serie.

1. En `fit`, un DataFrame plano se asigna a todas las series por referencia, sin copiar. Un dict se valida para que cada columna compartida tenga el mismo dtype en todas las series; si no, `TypeError` (también cuando las categorías de una columna categórica difieren).
2. La alineación se hace por serie. Si las series tienen distinta longitud, cada bloque exog se recorta al rango de su serie.
3. En `predict`, `_exog_to_dict` difunde de nuevo un DataFrame plano a todas las series. La alineación temporal usa el fin del contexto de cada serie, así que si las series terminan en fechas distintas, cada una obtiene su propia ventana de `steps` pasos.
4. Con `levels`, se filtran `context` y `context_exog` antes de preparar el futuro; solo se alinean las series solicitadas.
5. En los adaptadores, como todas las series tienen las mismas columnas, la reconciliación es nula: el agrupado de TimesFM 3.0 y la unión de T0 coinciden con las columnas de cada serie; en TabICL y TabPFN el `concat` no genera NaN.

Detalles de lote:

- Chronos activa `cross_learning` solo si hay más de una serie en el lote.
- T0 procesa todo en una llamada. Si los contextos tienen distinta longitud, rellena por la izquierda con NaN tanto el target como las covariables, de modo que el origen del pronóstico queda alineado al final para todas las series.
- TS-ICL, TabICL y TabPFN pasan el lote completo en una llamada; Nori itera serie a serie.

---

## 4. Escenario 3: múltiples series con subconjuntos distintos de exógenas

Flujo del usuario:

```python
exog = {'a': df[['x1', 'x2']], 'b': df[['x1']], 'c': None}
forecaster.fit(series, exog=exog)
pred = forecaster.predict(steps=12, exog=exog_fut)   # mismo patrón por serie
```

1. En `fit`, el dict se acepta tal cual. La serie `c` recibe `None` y se emite `MissingExogWarning` ("All values ... will be NaN", texto heredado; en foundation `c` simplemente no tiene bloque). `exog_names_in_` queda como `['x1', 'x2']`; `exog_names_in_per_series_` conserva el detalle por serie.
2. En `predict`, el dict futuro se procesa por serie; las claves ausentes reciben `None`. La comparación de columnas es serie a serie, así que no avisa mientras cada serie tenga las mismas columnas en contexto y futuro.
3. Los adaptadores divergen en tres grupos:

**Aislamiento correcto por serie: Chronos, TS-ICL, Nori y TimesFM 3.0.** Chronos y TS-ICL construyen un dict independiente por serie y lo entregan como lista, así que `b` nunca ve `x2` y `c` va sin covariables. Nori calcula la intersección contexto ∩ futuro de cada serie por separado y construye una matriz de características distinta por serie, coherente porque se reajusta en contexto para cada una. TimesFM 3.0 calcula la firma `(past_only_cols, fut_cols)` de cada serie con sus propias columnas, agrupa las series con la misma firma y llama a `predict_batch` una vez por grupo: `a` va en una llamada con `x1` y `x2`, `b` en otra con `x1`, y `c` en una tercera sin covariables. Cada serie recibe exactamente las mismas covariables que si se predijera sola, por lo que `predict(levels=[x]) == predict()[x]`.

Antes de este cambio, `_predict_v3` decidía las columnas a nivel de lote y rellenaba con NaN lo que cada serie no aportaba; el backend interpola esos NaN como ceros o como constante extrapolada, y la predicción de `b` y `c` cambiaba según qué exog tuviera `a` (hasta 1.8 desviaciones típicas de diferencia, medido en `dev/review_timesfm_3.0_adapter_round2.md`, sección 5.1). Ese comportamiento ya no existe.

**Relleno con NaN tratado como ausente: T0.** Las columnas del array son la unión de las columnas futuras de todas las series (`x1`, `x2`). La fila de `b` tiene `x2` entera en NaN, y la fila de `c` está toda en NaN. T0 interpreta NaN como valor ausente; no se fabrica información.

**Punto intermedio: TabICL y TabPFN.** `pd.concat(ignore_index=True)` une las columnas entre series y deja NaN donde una serie no aporta la columna. La librería TabICL calcula después la intersección entre las columnas del contexto y del futuro para el lote completo (no por serie) y registra un aviso por su logger si hay NaN en covariables futuras. Cómo pondera TabICL esos NaN en la regresión tabular es cosa del modelo; no se ha medido.

---

## 5. Escenario 4: discrepancia entre entrenamiento y predicción

`FoundationModel.predict` resuelve las dos situaciones en `_check_exog_columns`, antes de llamar al adaptador, comparando las columnas del `context_exog` resuelto con las del `exog` futuro de cada serie.

### 5.1 Aparece una variable nueva en el futuro

`fit(exog=[x1])` y `predict(exog=[x1, x_new])`.

- `FoundationModel` lanza `ValueError` para los siete adaptadores: "`exog` contains columns with no historical values in the context for series {serie: ['x_new']}". Ningún adaptador recibe la columna.
- Lo mismo ocurre en modo zero-shot con `predict(context=..., exog=...)` sin `context_exog`.
- Lo que hacía cada backend antes de la validación (Chronos `ValueError` desde la librería, TimesFM 3.0 exclusión con `UserWarning`, TabICL/TabPFN/Nori descarte silencioso, T0 uso con pasado NaN) ya no es alcanzable por la ruta pública. `TimesFMAdapter._v3_covariate_signature` conserva un `ValueError` propio como defensa para llamadas directas al adaptador o con `check_inputs=False`.

### 5.2 Falta en el futuro una variable usada en entrenamiento

`fit(exog=[x1, x2])` y `predict(exog=[x1])`, o `predict()` sin `exog`.

- **Chronos, TS-ICL y TimesFM 3.0** (`supports_past_only_covariates=True`): `x2` pasa como covariable past-only, uso legítimo y documentado; `x1` sigue como known-future. No hay aviso. En TimesFM 3.0 la firma se calcula por serie, así que el resultado no depende de que otra serie del lote sí aporte `x2` en el futuro.
- **TabICL, TabPFN, Nori y T0** (`supports_past_only_covariates=False`): `FoundationModel` emite `IgnoredArgumentWarning` ("only uses covariates that also have future values. Historical exog columns without future values are ignored for series {serie: ['x2']}") y el adaptador descarta `x2` como hacía antes (intersección en TabICL/TabPFN/Nori; en T0 el conjunto de columnas se define desde el futuro). Con `exog=None` el aviso lista todas las columnas históricas y se predice sin covariables.

### 5.3 La ruta de backtesting es distinta

`backtesting_foundation` (`_validation.py:2582`) normaliza `exog` una sola vez con las mismas utilidades de `fit` (`check_preprocess_exog_multiseries` y `align_series_and_exog_multiseries`). `_extract_data_folds_multiseries` (`_utils.py:911`) corta cada fold del mismo DataFrame por serie para obtener `context_exog` y `exog_test`; las series con exog `None` reciben `None` en ambos. Las llamadas a `predict` y `predict_quantiles` van con `check_inputs=False`, así que no se ejecutan ni `_prepare_future_exog` ni el aviso de divergencia.

Consecuencias:

- Por construcción, en backtesting las columnas históricas y futuras de cada serie siempre coinciden. Los escenarios 4a y 4b no pueden darse, y por eso `_check_exog_columns` no se ejecuta en esta ruta (hay un comentario en `FoundationModel.predict` que lo documenta).
- El escenario 3 sí puede darse (dict parcial). Con la agrupación por firma de TimesFM 3.0, las métricas de cada serie son las mismas que se obtendrían prediciéndola sola.

---

## 6. Resumen de puntos de diseño

- La capa intermedia unifica el formato y valida los nombres de columnas por serie en `_check_exog_columns`, contra el `context_exog` resuelto (lo que permite el modo zero-shot sin `fit`). Una columna futura sin histórico es `ValueError` para todos los adaptadores.
- Ante una columna histórica sin futuro hay dos grupos, declarados por cada adaptador con `supports_past_only_covariates`: los que la usan como past-only (Chronos, TS-ICL, TimesFM 3.0) y los que la descartan (TabICL, TabPFN, Nori, T0), en cuyo caso `FoundationModel` avisa con `IgnoredArgumentWarning`.
- `_prepare_future_exog` se ocupa de la alineación temporal y `_check_exog_columns` de los nombres; los adaptadores ya no razonan sobre columnas más allá de construir sus arrays con las columnas de cada serie.
- TimesFM 3.0 agrupa las series por firma de covariables y llama a `predict_batch` una vez por grupo, de modo que `predict(levels=[x]) == predict()[x]` con cualquier combinación de exog por serie. Tests: `test_TimesFMAdapter_v3_predict_groups_series_by_covariate_signature`, `test_FoundationModel_v3_predict_levels_matches_batch_covariates` y `test_check_exog_columns.py` (incluida la ausencia de aviso para el uso past-only en Chronos, TS-ICL y TimesFM 3.0).
