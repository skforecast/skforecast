# Handoff: optimizaciones del preprocesado (codificación de categóricas, atajo de escaladores y predicción in-sample en `fit`)

**Fecha:** 2026-11-18.
**Origen:** el estudio de viabilidad de polars en `ForecasterRecursiveMultiSeries` (descartado; ver la sección 1).
**Para:** quien implemente estos cambios más adelante.
**Base comprobada:** `0.28.x` en `c694e19` (2026-11-17). Los números de línea citados son de ese commit: vuelve a localizarlos antes de editar.
**Entorno de medida:** Python 3.12.3, numpy 2.5.5, pandas 3.1.1 (con pyarrow 26.0.1), scikit-learn 1.9.1, numba 0.68.1, lightgbm 4.7.0. CPU Intel Xeon a 2,10 GHz, 4 núcleos lógicos, 15,7 GiB de RAM (contenedor en la nube). La equivalencia se comprobó también con pandas 2.3.3.

> **Cómo usar este documento**
> 1. Empieza por la tarea que toque (T1, T2 o T3): cada una trae el problema, el diseño, los tests, los criterios de aceptación y la nota de versión.
> 2. Las tareas T1 y T2 son independientes y van en PRs separados. T3 es un estudio que termina en una decisión, no en código.
> 3. Antes de empezar, vuelve a ejecutar `dev/preprocessing_speedups/check_equivalence.py` y `bench.py` sobre el `0.28.x` del momento (sección 4), porque el código puede haber cambiado.
> 4. Las validaciones pendientes (sección 3) indican qué tarea bloquean.
> 5. Las decisiones marcadas «decisión de Joaquín» no se implementan hasta que él elija una opción. La recomendada va primero.

---

## 1. Contexto y decisión

Se estudió con números medidos si polars aceleraría `ForecasterRecursiveMultiSeries`. El caso objetivo:

- **Series:** de 100 a 1.000, con 1.000 a 5.000 observaciones horarias cada una y longitudes e inicios distintos.
- **Exógenas:** 5 numéricas y 3 categóricas `category` (de 5 a 50 categorías), unas comunes a todas las series y otras específicas.
- **Configuración:** `lags=48`, `RollingFeatures` con 3 estadísticos y `StandardScaler` como `transformer_series`.
- **Estimadores:** LightGBM, HGBR y Ridge.

**Decisión: no adoptar polars.** Los cuellos de botella no son operaciones tabulares:

- Con LightGBM, el estimador es el 83 % del `fit` en todos los tamaños, así que el techo de Amdahl es ×1,19.
- Medido de extremo a extremo con 1.000 series:
  - polars, `fit` de 37,8 s a 34,3 s (estimado);
  - la alternativa con numpy, 32,9 s. Es una sola repetición: lo esperable es ahorrar unos 3 s, lo que cuesta de menos construir `X_train`.
- En `predict`, solo convertir la entrada a polars ya cuesta más que el `predict` optimizado con numpy.

Los ficheros del estudio de polars se han borrado. Siguen en el historial de la rama local `chore/polars-study`, que no se subió a GitHub, mientras no se borre.

Lo que sí compensa son dos cambios pequeños, sin dependencias nuevas y con resultados idénticos bit a bit (T1 y T2), más un estudio de diseño (T3). Las cifras de partida (`ForecasterRecursiveMultiSeries`, encoding ordinal, entrada como dict, mediana de varias repeticiones):

| Tamaño | Filas de entrenamiento | `fit` LightGBM | `fit` Ridge | Parte de skforecast en `fit` | Codificador en `fit` | `predict` 168 (LightGBM) |
|---|---|---|---|---|---|---|
| S (100 series) | 134.137 | 2,34 s | 0,42 s | 0,4 s | 0,12 s | 0,12 s (0,03 s estimador) |
| M (300 series) | 880.369 | 10,41 s | 2,25 s | 1,7 s | 0,76 s | 0,27 s (0,08 s estimador) |
| L (1.000 series) | 2.895.871 | 37,53 s | 7,61 s | 6,0 s | 2,8 s | 0,97 s (0,22 s estimador) |

**Dónde se va el tiempo de skforecast** (cProfile, L, Ridge):

1. **`OrdinalEncoder` sobre columnas `category`:** convierte cada columna a un array de objetos (`Categorical.__array__`, 1,0 s; con pandas 3 y pyarrow, las categorías de texto están en Arrow) y la mapea con diccionarios de Python (`_map_to_integer`, `_check_unknown` y `_unique_python`, 1,4 s). En el backtesting con refit (L, Ridge) suma 19 s de 71,8 s, más que el propio estimador (14,8 s). Su pico de memoria transitoria es ~0,46 veces el tamaño de `X_train` (0,6 GiB en L).
2. **`StandardScaler` serie a serie:** en cada `predict`, unas 2.000 llamadas (transformar la última ventana e invertir las predicciones de cada serie). Cada una pasa por `validate_data` / `check_array` de scikit-learn: 0,57 s de `transform_numpy` sobre 1,5 s de `predict` medido con cProfile (0,64 s sin perfilador) en L con Ridge.
3. **`fit` predice sobre todo `X_train`** para ajustar los bins de residuos (`_probabilistic_mode='binned'` por defecto), aunque luego no se pidan intervalos: 13 % del `fit` con LightGBM y 17 % con HGBR en L.

---

## 2. Tareas

### T1. Atajo en `transform_numpy` para escaladores conocidos y ajustados

**Problema.** `transform_numpy` (`skforecast/utils/utils.py`, línea 2580) llama a `transformer.transform` o `inverse_transform` en cada serie. Para un `StandardScaler`, el cálculo es una resta y una división, pero scikit-learn valida la entrada en cada llamada (`validate_data`, `check_array`, `_assert_all_finite`), lo que cuesta del orden de 0,1 ms por llamada. En `predict` multiserie hay dos llamadas por serie, y en el backtesting dos por serie y fold.

**Ganancia medida con el prototipo** (`ForecasterRecursiveMultiSeries`, M, solo T1; tabla completa en la sección 4):

- `predict` de 168 pasos: Ridge, de 0,194 a 0,109 s (×1,77); LightGBM, de 0,264 a 0,226 s (×1,16).
- `predict` de 24 pasos: ×1,57 con Ridge y ×1,48 con LightGBM.
- Conformal de 24 pasos: ×1,56 con Ridge y ×1,17 con LightGBM.
- Backtesting con refit (Ridge): de 22,7 a 19,2 s (×1,18).
- Con T2: `predict` de 168 pasos ×2,15 con Ridge y ×1,36 con LightGBM; en L, ×2,1 con Ridge y ×1,8 con LightGBM.

El atajo funciona igual con exógenas de texto, porque no depende de ellas.

**Dónde.** Una sola función, que usan todos los forecasters: `transform_numpy` se llama desde `recursive/_forecaster_recursive_multiseries.py` (7 veces), `recursive/_forecaster_recursive.py` (6), `direct/_forecaster_direct.py` (6), `direct/_forecaster_direct_multivariate.py` (7), `deep_learning/_forecaster_rnn.py` (5), `recursive/_forecaster_stats.py` (2) y `model_selection/_utils.py` (6). El cambio dentro de `transform_numpy` beneficia a todos sin tocarlos.

**Diseño propuesto.** Prototipo de referencia: `fast_transform_numpy` en `dev/preprocessing_speedups/prototypes.py`.

1. **Ruta rápida:** una función privada, por ejemplo `_transform_numpy_known_scaler(array, transformer, inverse_transform)`. Devuelve el array transformado si el atajo aplica y `None` si no. `transform_numpy` la llama justo después de `if transformer is None` y, con `None`, sigue por el camino actual.
2. **El atajo aplica solo si se cumple todo lo siguiente** (cualquier otro caso va por scikit-learn):
   - `fit=False`;
   - `type(transformer) is StandardScaler`: el tipo exacto, para que las subclases sigan el camino actual;
   - el escalador está ajustado: tiene `scale_` (o `with_std=False`) y `mean_` (o `with_mean=False`);
   - `transformer.n_features_in_ == 1` y no tiene `feature_names_in_`;
   - `array.dtype == np.float64`. Con float32, scikit-learn mantiene float32: el camino actual se encarga;
   - `array.ndim == 1`, o `inverse_transform=True` con cualquier `ndim`, porque el camino actual de la inversa multicolumna se reduce a una columna y vuelve.
3. **La misma aritmética, en el mismo orden, que scikit-learn 1.9:**
   - transformación: `out = array.copy()`; `if with_mean: out -= mean_[0]`; `if with_std: out /= scale_[0]`;
   - inversa: `if with_std: out *= scale_[0]`; `if with_mean: out += mean_[0]`.

   Operar con el escalar `mean_[0]` sobre un array 1D da el mismo resultado IEEE que el array de forma `(1,)` de scikit-learn: comprobado bit a bit.
4. **Conservar la copia de la entrada.** La #1504 (`fix/inplace-transformers`) garantiza que `transform_numpy` no modifica el array del llamante. El atajo hace una sola copia; scikit-learn hacía dos.
5. **Valores infinitos (decisión de Joaquín).** scikit-learn lanza `ValueError` con `inf` (`ensure_all_finite="allow-nan"`) y el atajo, tal cual, no. Recomendado: después de calcular, `if not np.isfinite(out[~np.isnan(out)]).all():` volver al camino actual, para que lance el mismo error. Cuesta O(n) sobre arrays pequeños. Alternativa: aceptar la diferencia y documentarla.
6. **Fuera del alcance de T1:** `MinMaxScaler`, `RobustScaler` y los `Pipeline`. Se añaden después de la validación V3.

**Tests.** Fichero nuevo `skforecast/utils/tests/tests_utils/test_transform_numpy_known_scalers.py`, o casos añadidos al de `transform_numpy`. Cada caso compara con `np.testing.assert_array_equal` frente a llamar directamente al método de scikit-learn:

- transformar e invertir, 1D y 2D (inversa);
- las cuatro combinaciones de `with_mean` y `with_std`;
- NaN en la entrada;
- `inf` en la entrada (el mismo error que scikit-learn, si se elige la recomendación del punto 5);
- float32 (dtype de salida float32, por el camino actual);
- la entrada no se modifica;
- una subclase de `StandardScaler` usa el camino actual (por ejemplo, con un `mock` del método);
- un `StandardScaler` sin ajustar lanza el mismo `NotFittedError` que hoy;
- un `Pipeline([("s", StandardScaler())])` usa el camino actual;
- `transformer=None` devuelve la entrada sin copiar, como hoy.

Además: los tests de los módulos de los forecasters afectados y `dev/preprocessing_speedups/check_equivalence.py` (predicciones, intervalos y backtesting bit a bit).

**Aceptación:**
- identidad bit a bit en todos los tests y en `check_equivalence.py`, con pandas 2.3 y 3, y con scikit-learn 1.6 (V2);
- `predict_168` de `ForecasterRecursiveMultiSeries` en M al menos ×1,5 más rápido con Ridge y ×1,1 con LightGBM (el prototipo da ×1,77 y ×1,16);
- ninguna regresión de más del 5 % en `benchmarks/` (V7).

**Nota de versión** (sección *Changed*, área de rendimiento): "`transform_numpy` applies a fitted `StandardScaler` without scikit-learn's input validation, which speeds up `predict`, `predict_interval` and backtesting with `transformer_series` / `transformer_y` (up to N times with M series)." Pon N y M al medir.

---

### T2. Codificación de las exógenas `category` por códigos

**Problema.** Los forecasters codifican las exógenas categóricas con un `OrdinalEncoder` de scikit-learn:

```python
OrdinalEncoder(dtype=float, handle_unknown='use_encoded_value', unknown_value=np.nan, encoded_missing_value=np.nan).set_output(transform="pandas")
```

Con columnas `category`, scikit-learn las convierte a arrays de objetos y las mapea con diccionarios. El resultado equivale a "posición de la categoría en `categories_`", que se obtiene con los códigos enteros de pandas y una tabla de búsqueda.

**Ganancia medida con el prototipo:**
- construcción de `X_train` en L: de 6,41 a 3,43 s (×1,9);
- en M: de 1,71 a 0,97 s con Ridge (×1,76) y de 1,82 a 1,01 s con LightGBM (×1,81);
- `fit` de Ridge en M: de 2,74 a 1,91 s (×1,43);
- backtesting con refit (L, Ridge, junto con T1): de 71,8 a 48,1 s. Lo midió un prototipo anterior que además construía lags y window features por lotes, cambio que no dio ganancia medible;
- debería eliminar el pico de memoria transitoria del codificador (0,6 GiB en L). Está por medir (V9).

**Dónde** (números de línea en `c694e19`):

| Fichero | Creación del codificador | `fit_transform` / `transform` en `_create_train_X_y` | `transform` en la predicción |
|---|---|---|---|
| `recursive/_forecaster_recursive.py` | 406 | 904, 910 | 1648 |
| `recursive/_forecaster_recursive_classifier.py` | 383 | 911, 917 | 1611 |
| `recursive/_forecaster_recursive_multiseries.py` | 534 | 1632, 1638 | 2847 (exog ancha o larga), 2957 (exog en dict) |
| `direct/_forecaster_direct.py` | 597 | 1120, 1126 | 2225 |
| `direct/_forecaster_direct_multivariate.py` | 553 | 1200, 1206 | 2398 |

**Diseño propuesto.** Prototipo de referencia: `FastOrdinalEncoder` en `dev/preprocessing_speedups/prototypes.py`.

1. **No cambiar el atributo público.** `self.categorical_encoder` sigue siendo un `OrdinalEncoder` de scikit-learn con los mismos parámetros. Así no cambian:
   - la serialización (`save_forecaster` / `load_forecaster` con pickle y skops, que reconstruye tipos de scikit-learn conocidos);
   - `__repr__`, `get_params` y la documentación;
   - los forecasters guardados con versiones anteriores, que tienen el codificador ajustado sobre todos los datos y deben poder predecir con la ruta nueva.

   Descartado: una subclase de `OrdinalEncoder`, porque cambia el tipo de un atributo público y obliga a registrar el tipo en skops.
2. **Dos funciones privadas en `skforecast/utils/utils.py`** (nombres orientativos):
   - `_fit_transform_categorical(encoder, X)`: si todas las columnas de `X` son `pd.CategoricalDtype`, ajusta `encoder` sobre una tabla pequeña y codifica con la ruta rápida; si no, `encoder.fit_transform(X)`.
   - `_transform_categorical(encoder, X)`: si todas las columnas son `category`, usa la ruta rápida; si no, `encoder.transform(X)`.

   Los 5 forecasters cambian `self.categorical_encoder.fit_transform(...)` y `.transform(...)` por estas funciones (12 llamadas en total, tabla de arriba).
3. **Ajuste equivalente.**
   - Por cada columna, se toma de los datos una fila de cada categoría presente y, si hay valores faltantes, una fila con NaN: `np.unique(codes, return_index=True)` da la primera posición de cada código, y `-1` es el faltante.
   - Las columnas se rellenan repitiendo su primera fila hasta igualar longitudes.
   - Se ajusta `encoder` sobre esa tabla. scikit-learn ve los mismos valores y el mismo dtype que con los datos completos, así que `categories_` es idéntico, en valores, orden (alfabético, con NaN al final) y dtype (por ejemplo `int64` con categorías enteras), y también lo son `n_features_in_` y `feature_names_in_`.
   - Ojo: la primera versión del prototipo construía la tabla como `object` y obtenía `categories_` de dtype `object` con categorías enteras. La salida era igual, pero el estado ajustado distinto. La versión actual lo evita, y `check_equivalence.py` comprueba ya el dtype.
4. **Codificación rápida.** Por columna:

   ```python
   lookup = pd.Index(encoder.categories_[j]).get_indexer(X[col].cat.categories).astype(float)
   lookup[lookup < 0] = np.nan          # categoría del dtype no vista en el ajuste
   lookup = np.append(lookup, np.nan)   # el código -1 (faltante) cae en esta última posición
   codificada = lookup[X[col].cat.codes.to_numpy()]
   ```

   El resultado es un `DataFrame` float64 con las mismas columnas e índice que devuelve `set_output(transform="pandas")`. Funciona aunque las categorías del dtype estén en otro orden, haya categorías nuevas en `predict` o la columna sea ordenada.
5. **Columnas de texto (`str` u `object`) y mezclas:** en la primera versión, la ruta rápida solo se usa si todas las columnas categóricas son `category`; en cualquier otro caso se usa scikit-learn, como hoy. Una ruta rápida para texto queda para la validación V1.

**Tests.** Fichero nuevo `skforecast/utils/tests/tests_utils/test_categorical_encoding.py`. En cada caso, `assert_frame_equal(check_exact=True)` de la salida y comparación de `categories_` (valores, NaN y dtype) frente a `OrdinalEncoder` directamente:

- categorías de texto y enteras;
- columna ordenada;
- NaN en el ajuste y en la transformación;
- categoría vista en el dtype pero ausente en el ajuste;
- categoría nueva en `transform`;
- categorías del dtype en otro orden en `transform`;
- varias columnas con distinto número de categorías;
- una sola fila;
- columnas mezcladas (`category` y `str`): camino actual;
- pandas 2.3 y 3, con y sin pyarrow (V5).

En los forecasters:
- un test por forecaster que compare `create_train_X_y` y `predict` con y sin la ruta rápida;
- un test de `save_forecaster` / `load_forecaster`, con pickle y con skops, de un forecaster ajustado con la ruta nueva;
- un forecaster ajustado con la versión actual (codificador ajustado con todos los datos) que prediga con la ruta nueva (V8).

**Aceptación:**
- identidad bit a bit de `X_train`, `y_train`, predicciones, intervalos y `categories_` (valores y dtype);
- construcción de `X_train` en M al menos ×1,5 más rápida con exógenas `category`;
- sin regresión con exógenas de texto (deben ir por el mismo camino de hoy);
- `benchmarks/` sin regresiones (V7).

**Nota de versión:** "The categorical exogenous variables with `category` dtype are encoded with their integer codes instead of scikit-learn's `OrdinalEncoder` conversion to object arrays, with the same result. Training matrices are built up to N times faster and use less memory." Pon N al medir.

---

### T3. Estudio: la predicción sobre `X_train` en `fit`

**Qué pasa hoy.** Con `_probabilistic_mode` distinto de `False`, que es el valor por defecto (`"binned"`), `fit`:

- predice sobre todo `X_train` (`self.estimator.predict(X_train_estimator)`);
- ajusta un `QuantileBinner` por serie con esas predicciones (`_binning_in_sample_residuals`);
- guarda los bordes (`binner_intervals_`);
- si `store_in_sample_residuals=True`, guarda una muestra de residuos por bin;
- copia filas de `X_train` para comprobar que el estimador no lo modifica (`_copy_rows_to_check` y `_check_in_place_fit`, ISSUE-1506).

`backtesting_*` ya pone `_probabilistic_mode = False` cuando no hay `interval` (`model_selection/_validation.py`, líneas 752 y 1737), así que no lo pagan ni el backtesting ni las búsquedas basadas en él (sin intervalos). Lo pagan los `fit` directos del usuario. Las búsquedas con `OneStepAheadFold` no pasan por `fit`: confírmalo en el estudio.

**Coste medido** (`ForecasterRecursiveMultiSeries`, `fit` sin `store_in_sample_residuals`):

| Tamaño | LightGBM | HGBR | Ridge |
|---|---|---|---|
| S | 9 % | 10 % | 2 % |
| M | 14 % | 19 % | 3 % |
| L | 13 % (4,8 s) | 17 % (7,7 s) | 3 % |

**Para qué se usa el binner** (`recursive/_forecaster_recursive_multiseries.py`):
- bootstrapping con residuos por bins (líneas 3303 y 3764);
- conformal con residuos por bins (3962);
- `set_out_sample_residuals`, que agrupa los residuos fuera de muestra por los bins del ajuste (4819 y 4938).

La misma lógica está en `ForecasterRecursive` (línea 1394), `ForecasterDirect` (1936), `ForecasterDirectMultiVariate` (2084), `ForecasterRnn` (1134), `ForecasterEquivalentDate` (455) y `ForecasterRecursiveClassifier`.

**Pregunta que hay que contestar.** ¿Se puede evitar este coste cuando el usuario no va a pedir intervalos, sin cambiar ningún resultado cuando sí los pide y sin romper la API?

**Opciones a evaluar:**

| Opción | Descripción | A favor | En contra |
|---|---|---|---|
| O1 | Parámetro explícito (en `__init__` o en `fit`) para no preparar los intervalos | Sin cambios de resultado | Con el valor por defecto actual no gana nada; con otro valor por defecto cambia el comportamiento |
| O2 | No preparar nada en `fit` y calcularlo bajo demanda con `set_in_sample_residuals(series, exog)` (ya existe) la primera vez que se pide un intervalo | Gana en todos los `fit` directos | Cambio de API: `predict_interval` sin preparación previa tendría que lanzar un error claro o recalcular con datos que ya no tiene |
| O3 | Ajustar el binner con una muestra de filas | Gana siempre | Cambia los bordes de los bins y, por tanto, los intervalos. Habría que medir la cobertura |
| O4 | Mantenerlo y documentarlo, con el coste en la guía de rendimiento | Sin riesgo | Sin ganancia |

**Qué medir:**
1. El porcentaje del `fit` en `ForecasterRecursive`, `ForecasterDirect` (que predice con cada estimador de cada paso: el coste puede multiplicarse por `steps`) y `ForecasterDirectMultiVariate`, con LightGBM, HGBR, XGBoost y Ridge.
2. Qué flujos de usuario dependen de la preparación en `fit`: los intervalos justo después de `fit` y `set_out_sample_residuals` con bins. Cuántos aparecen en la documentación (`docs/` y `skills/prediction-intervals/`).
3. Para O3, la diferencia en cobertura y en anchura de los intervalos con varias fracciones de muestreo.

**Criterio de decisión (decisión de Joaquín):** si con la opción elegida el coste sigue siendo de más del 10 % del `fit` con árboles, y el cambio de API se puede introducir con un aviso de deprecación en una versión, avanzar con O2 u O1. Si no, O4.

**Entregable:** una fila en la hoja de ruta (sección 5) con la decisión y, si procede, la tarea de implementación con sus propios tests y nota de versión.

---

## 3. Validaciones pendientes

| ID | Qué validar | Bloquea | Cómo |
|---|---|---|---|
| V1 | Exógenas categóricas de texto (`str` en pandas 3, `object` en pandas 2) | Una posible ampliación de T2 | Con texto, el coste actual es el mismo que con `category` (construcción en M: 1,69 s frente a 1,71 s), y la ruta rápida de T2 no aplica. Prototipar con `pd.factorize` (o `pd.Categorical(col, categories=encoder.categories_[j])` en `transform`) y comprobar la identidad bit a bit con `OrdinalEncoder`, incluido el orden de `categories_` con mayúsculas, acentos y números como texto. Medir con pandas 2.3, pandas 3 sin pyarrow y pandas 3 con pyarrow |
| V2 | scikit-learn 1.6 (versión mínima): la aritmética de `StandardScaler` y el orden de `categories_` de `OrdinalEncoder` pueden cambiar entre versiones | T1, T2 | Entorno con las versiones mínimas (`uv pip install --resolution lowest-direct -e ".[test]"`, ver `unit-tests-min-deps.yml`) y `check_equivalence.py` |
| V3 | Otros escaladores: `MinMaxScaler` (`X *= scale_; X += min_`, `clip`), `RobustScaler` (`center_`, `scale_`, `with_centering`, `with_scaling`) | Ampliación de T1 | Leer su `transform` / `inverse_transform` en la versión mínima y en la última; mismos tests que T1 |
| V4 | El resto de forecasters: `ForecasterRecursive`, `ForecasterDirect`, `ForecasterDirectMultiVariate`, `ForecasterRecursiveClassifier` (T2), `ForecasterRnn` y `ForecasterStats` (T1) | T1, T2 | Medir antes y después con `benchmarks/` y con un caso equivalente al de `bench.py`. `check_equivalence.py` ya cubre `ForecasterRecursive` |
| V5 | Sin pyarrow: con pandas 3 sin pyarrow, las categorías de texto no son Arrow y el codificador actual cuesta menos | T2 (para documentar la ganancia) | `bench.py` en un entorno sin pyarrow. Con pandas 2.3, la construcción en M bajó de 1,34 a 0,93 s con T2 (×1,45), frente a ×1,8 con pandas 3 y pyarrow |
| V6 | Estimadores con categóricas nativas (LightGBM `categorical_feature`, HGBR `categorical_features`, XGBoost `enable_categorical`, CatBoost vía `cast_catboost_categorical_columns_dataframe`) | T2 | Si `X_train` es idéntico bit a bit, su ajuste también lo es. Comprobarlo con un caso por estimador en `check_equivalence.py` (CatBoost convierte las columnas a enteros: confirmar que recibe los mismos códigos) |
| V7 | Suite `benchmarks/` antes y después | T1, T2 | Ver MAINT-bench-pandas3 en la hoja de ruta; diferencias de menos del 5 % son ruido |
| V8 | Serialización y compatibilidad hacia atrás | T2 | Guardar con `save_forecaster` (pickle y skops) un forecaster ajustado con la versión actual, cargarlo con la nueva y predecir: mismas predicciones. Y al revés |
| V9 | Memoria | T2 (para la nota de versión) | `tracemalloc` alrededor de `_create_train_X_y` antes y después: se espera eliminar ~0,46 veces el tamaño de `X_train` |
| V10 | La exógena en dict en `predict` multiserie (`reindex_like` por serie y `pd.concat`, ~0,3 s con 1.000 series) | Ninguna (otra tarea) | Encaja en PERF-multiseries-names-pyarrow, que la menciona sin analizar |

---

## 4. Kit de validación: `dev/preprocessing_speedups/`

| Fichero | Qué hace |
|---|---|
| `data.py` | Datos sintéticos del estudio (tamaños T, S, M y L), el forecaster de referencia y el informe del entorno |
| `prototypes.py` | `FastOrdinalEncoder` (T2) y `fast_transform_numpy` (T1), aplicados en tiempo de ejecución sin tocar `skforecast/` |
| `check_equivalence.py` | Identidad bit a bit de los dos prototipos: base (Ridge y LightGBM, ordinal y onehot), NaN en categóricas, categoría no vista, categorías reordenadas, categorías ordenadas y enteras, exógenas que faltan, columna de texto (camino actual), backtesting con refit y `ForecasterRecursive` |
| `bench.py` | Actual frente a T2 (`P1`), T1 (`P2`) y ambos, en la construcción de `X_train`, `fit`, `predict` (24 y 168), conformal y backtesting (refit `False` y `True`). Mediana, dispersión y primera llamada. `--text-cats` para V1 |
| `results/` | JSON de las mediciones de referencia de este documento |

Comandos (desde la raíz del repositorio):

```bash
python dev/preprocessing_speedups/check_equivalence.py
```

```bash
python dev/preprocessing_speedups/bench.py --size M --est ridge --reps 3
```

Al implementar T1 o T2, el prototipo deja de ser necesario: la comprobación de equivalencia se convierte en los tests de la tarea, y `bench.py` sirve para medir la rama frente a `0.28.x` (ejecutándolo en cada una). Mide siempre con la máquina libre: LightGBM y HGBR usan OpenMP y se degradan mucho con otra carga (en el estudio se vio una repetición de 202 s frente a 4,7 s).

**Resultados de referencia** (M, `bench.py`, pandas 3.1.1 con pyarrow):

| Estimador | Flujo | Actual | T2 (P1) | T1 (P2) | T1 + T2 |
|---|---|---|---|---|---|
| Ridge | `build_X_train` | 1,707 s | 0,971 s (×1,76) | 1,678 s (×1,02) | 0,891 s (×1,92) |
| Ridge | `fit` | 2,735 s | 1,913 s (×1,43) | 2,505 s (×1,09) | 1,678 s (×1,63) |
| Ridge | `predict_24` | 0,221 s | 0,200 s (×1,10) | 0,141 s (×1,57) | 0,146 s (×1,51) |
| Ridge | `predict_168` | 0,194 s | 0,164 s (×1,18) | 0,109 s (×1,77) | 0,090 s (×2,15) |
| Ridge | `interval_conformal_24` | 0,345 s | 0,315 s (×1,10) | 0,222 s (×1,56) | 0,250 s (×1,38) |
| Ridge | `backtesting_refit=False` | 5,417 s | 4,369 s (×1,24) | 5,426 s (×1,00) | 4,729 s (×1,15) |
| Ridge | `backtesting_refit=True` | 22,694 s | 16,430 s (×1,38) | 19,178 s (×1,18) | 15,374 s (×1,48) |
| LightGBM | `build_X_train` | 1,815 s | 1,005 s (×1,81) | 1,936 s (×0,94) | 1,185 s (×1,53) |
| LightGBM | `fit` | 10,545 s | 10,781 s (×0,98) | 10,647 s (×0,99) | 10,105 s (×1,04) |
| LightGBM | `predict_24` | 0,224 s | 0,214 s (×1,05) | 0,152 s (×1,48) | 0,148 s (×1,52) |
| LightGBM | `predict_168` | 0,264 s | 0,249 s (×1,06) | 0,226 s (×1,16) | 0,193 s (×1,36) |
| LightGBM | `interval_conformal_24` | 0,354 s | 0,332 s (×1,07) | 0,302 s (×1,17) | 0,278 s (×1,27) |
| Ridge, categóricas como texto | `build_X_train` | 1,686 s | 1,979 s (×0,85) | 1,854 s (×0,91) | 1,826 s (×0,92) |
| Ridge, categóricas como texto | `fit` | 2,487 s | 2,502 s (×0,99) | 2,450 s (×1,02) | 2,520 s (×0,99) |
| Ridge, categóricas como texto | `predict_24` | 0,264 s | 0,253 s (×1,04) | 0,144 s (×1,83) | 0,153 s (×1,73) |
| Ridge, categóricas como texto | `predict_168` | 0,166 s | 0,177 s (×0,94) | 0,127 s (×1,30) | 0,139 s (×1,19) |
| Ridge, categóricas como texto | `interval_conformal_24` | 0,330 s | 0,440 s (×0,75) | 0,265 s (×1,24) | 0,274 s (×1,20) |

Cómo leerla:

- **Repeticiones:** `build_X_train`, `predict` y conformal son la mediana de 3 repeticiones (2 con LightGBM); `fit` con LightGBM y el backtesting, de 1 (más la primera llamada). Diferencias de menos de un 10 % en esas filas son ruido.
- **Fila de texto `build_X_train` (×0,85):** es ruido. Remedido intercalando las variantes, 7 repeticiones: 1,76 s actual y 1,78 s con P1. Con texto, la ruta rápida no aplica y P1 sigue el camino actual.
- **`fit` con LightGBM no cambia de forma apreciable,** porque manda el estimador. Lo esperable es ahorrar el tiempo de construcción (~0,8 s de 10,5 s en M, ~3 s de 37,5 s en L).

---

## 5. Orden, PRs y hoja de ruta

1. **T1** (rama `perf/transform-numpy-known-scalers`, desde `origin/0.28.x`): el cambio más pequeño y localizado. Validaciones V2 (escalador) y V7.
2. **T2** (rama `perf/categorical-codes-encoding`): V2 (codificador), V4, V5, V6, V8 y V9.
3. **T3** (estudio, sin rama de código hasta la decisión).

Cada PR va contra `0.28.x` (o la rama de la versión en desarrollo), con su nota en `docs/releases/releases.md`, el lint y los tests de los módulos tocados antes del commit. Ninguno cambia la API pública. Si T3 la cambia, hay que actualizar también la referencia de la API (`tools/ai/llms-base.txt`) y la documentación de usuario (`skills/`), y regenerar los ficheros derivados.

Filas propuestas para `dev/roadmap_0.28.0/roadmap.md` (o la hoja de ruta de la versión siguiente), en el formato de la sección de rendimiento:

| ID | Qué | Esfuerzo | Estado | Referencia |
|---|---|---|---|---|
| PERF-transform-numpy-scalers | Atajo en `transform_numpy` para un `StandardScaler` ajustado (misma aritmética, sin la validación de scikit-learn): `predict` multiserie de 168 pasos en M ×1,77 con Ridge y ×1,16 con LightGBM (junto con PERF-categorical-codes, ×2,1 y ×1,8 con 1.000 series) | S | Candidata, prototipo idéntico bit a bit | `dev/handoff_preprocessing_speedups.md` (T1) |
| PERF-categorical-codes | Codificar las exógenas `category` con sus códigos en lugar de `OrdinalEncoder` (mismo `categories_` y misma salida): construcción de `X_train` ×1,8 a ×1,9 con pandas 3, ×1,45 con pandas 2.3; −0,46 veces `X_train` de memoria transitoria | S a M | Candidata, prototipo idéntico bit a bit | `dev/handoff_preprocessing_speedups.md` (T2) |
| STUDY-fit-insample-binning | `fit` predice sobre todo `X_train` para los bins de residuos aunque no se pidan intervalos (13 a 17 % del `fit` con árboles y 1.000 series) | S (estudio) | Pendiente de decisión | `dev/handoff_preprocessing_speedups.md` (T3) |

---

## 6. Riesgos

- **Identidad numérica:** T1 y T2 están diseñados para ser idénticos bit a bit, y el kit lo comprueba. El riesgo está en versiones de scikit-learn distintas de la 1.9.1 (V2). Si una versión cambia la aritmética del escalador o el orden de `categories_`, la ruta rápida debe seguirla o desactivarse para esa versión.
- **Atributos públicos:** `categorical_encoder` sigue siendo el mismo tipo y queda ajustado igual. Ningún atributo cambia de valor.
- **Superficie:** T2 toca 12 llamadas en 5 forecasters; T1 una sola función. Las dos rutas rápidas caen en el camino actual ante cualquier caso que no reconozcan, de modo que el peor caso es "sin ganancia", no "resultado distinto".
- **Expectativas:** con LightGBM o HGBR, el `fit` apenas mejora (lo que ahorra la construcción: ~8 %, dentro del ruido de una repetición) porque manda el estimador. La ganancia visible está en `predict`, los intervalos, los modelos lineales y el backtesting con refit. Conviene decirlo así en la nota de versión.
