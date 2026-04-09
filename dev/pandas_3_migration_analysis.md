# Análisis de Migración a pandas >= 3.0

> Evaluación del impacto de pandas 3.0 en skforecast y plan de compatibilidad dual (pandas >=2.1,<3 y pandas >=3.0)

**Fecha:** Marzo 2026  
**Versión pandas analizada:** 3.0.0 (released 21 enero 2026)  
**Versión skforecast:** 0.21.0+

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Cambios Clave de pandas 3.0](#2-cambios-clave-de-pandas-30)
3. [Análisis por Módulo](#3-análisis-por-módulo)
4. [Patrones Detectados y Acciones](#4-patrones-detectados-y-acciones)
5. [Impacto en Rendimiento](#5-impacto-en-rendimiento)
6. [Estrategia de Compatibilidad Dual](#6-estrategia-de-compatibilidad-dual)
7. [Escenario: Solo pandas 3](#7-escenario-solo-pandas-3)
8. [Plan de Acción Priorizado](#8-plan-de-acción-priorizado)
9. [Cambios Ya Aplicados](#9-cambios-ya-aplicados)

---

## 1. Resumen Ejecutivo

pandas 3.0 introduce cambios arquitectónicos profundos que afectan el rendimiento y comportamiento de skforecast. Los principales son:

| Cambio | Impacto en skforecast | Prioridad |
|--------|----------------------|-----------|
| **Copy-on-Write (CoW) activado por defecto** | Alto — overhead en copias defensivas innecesarias, `.to_numpy(copy=True)`, `pd.concat(copy=False)` | P0 |
| **StringDtype por defecto (PyArrow)** | Medio — validación de dtypes en `check_exog_dtypes`, comparaciones `dtype.name` | P1 |
| **Parámetro `copy` deprecado** | Medio — `pd.concat(copy=False)` emite FutureWarning | P1 |
| **Arrays read-only desde `.to_numpy()`/`.values`** | Medio-Alto — puede romper mutaciones in-place en arrays numpy | P1 |
| **Aliases de frecuencia deprecados** | Bajo — el código ya usa aliases modernos (`'h'`, `'MS'`, `'D'`) | P3 |
| **`inplace` deprecado** | Sin impacto — no se usa `inplace=True` en el código fuente | — |
| **Constructores copian numpy arrays por defecto** | Medio — `pd.DataFrame(data=numpy_array)` ahora copia | P2 |

**Conclusión:** skforecast es compatible funcionalmente con pandas 3.0 (con un fix ya aplicado en `check_exog_dtypes` — ver sección 9), pero experimenta regresión de rendimiento en `fit()` (~10-20%) debido principalmente al cambio en constructores `pd.DataFrame(data=numpy_array)` que ahora copian por defecto. Este overhead es **inevitable e inherente a pandas 3** y no se puede evitar ni siquiera dropando pandas 2.

---

## 2. Cambios Clave de pandas 3.0

### 2.1 Copy-on-Write (CoW)

En pandas 3.0, CoW está activado por defecto. Esto significa:

- **Cualquier indexación** (`.iloc`, `.loc`, `[]`) devuelve una referencia lazy, no una copia.
- **La mutación de una vista** dispara una copia automática (copy-on-write).
- **`.copy()` explícito** sigue funcionando pero es redundante en la mayoría de escenarios — CoW ya protege contra mutations inesperadas.
- **Chained assignment** (`df["col"][0] = val`) lanza `ChainedAssignmentError`.
- **`.to_numpy()` y `.values`** devuelven arrays **read-only** por defecto (para evitar modificar el backing data sin que CoW lo sepa).

**Implicación para skforecast:** Las llamadas `.copy()` defensivas que protegían contra vistas mutables ahora son innecesarias y añaden overhead de copia real donde antes CoW habría evitado la copia.

### 2.2 StringDtype por Defecto

Las columnas de texto ahora usan `pd.StringDtype()` (backed by PyArrow) en lugar de `object`. Esto afecta:

- `dtype.name` devuelve `"string"` en lugar de `"object"` para columnas de texto.
- Comparaciones como `dtype.name.startswith(("int", "float", ...))` fallarán para el nuevo `"string"` dtype — pero esto es deseable porque skforecast ya advierte sobre dtypes no numéricos.
- `exog.dtypes.to_dict()` devuelve tipos `pd.StringDtype()` en lugar de `dtype('O')`.

### 2.3 Parámetro `copy` Deprecado

El parámetro `copy` se ha deprecado en:
- `pd.concat()`
- `pd.DataFrame()`, `pd.Series()` constructores
- `.astype()`, `.reindex()`, `.align()`, etc.

En pandas 3.0 `copy=False` ya no tiene efecto (CoW siempre usa lazy copy) y emite `FutureWarning`.

### 2.4 Arrays Read-Only

`.to_numpy()` y `.values` devuelven arrays read-only. Para obtener un array mutable:
- Usar `.to_numpy(copy=True)` — fuerza copia y devuelve array writable.
- Usar `np.asarray(series)` — devuelve array writable (copia si es necesario).

### 2.5 Aliases de Frecuencia

Deprecated aliases (eliminados en 3.0):

| Deprecated | Nuevo |
|-----------|-------|
| `'H'` | `'h'` |
| `'T'`, `'min'` | `'min'` |
| `'S'` | `'s'` |
| `'L'` | `'ms'` |
| `'U'` | `'us'` |
| `'N'` | `'ns'` |
| `'M'` | `'ME'` |
| `'Q'` | `'QE'` |
| `'Y'`, `'A'` | `'YE'` |
| `'BM'` | `'BME'` |
| `'BQ'` | `'BQE'` |
| `'BA'` | `'BYE'` |

**Estado en skforecast:** ✅ Ya usa aliases modernos (`'h'`, `'MS'`, `'D'`, `'QS'`). No se encontraron aliases deprecados en el código fuente.

### 2.6 Constructores Copian Arrays por Defecto

`pd.DataFrame(data=numpy_array)` y `pd.Series(data=numpy_array)` ahora copian el array por defecto (en pandas 2.x compartían memoria). Esto afecta la construcción de DataFrames desde arrays numpy, que es frecuente en skforecast.

---

## 3. Análisis por Módulo

### 3.1 `recursive/_forecaster_recursive.py`

#### `_create_lags()` (líneas 512-568)

```python
# Usa sliding_window_view — devuelve vista read-only
X_train_lags = np.lib.stride_tricks.sliding_window_view(y_values, ...)
# Fancy indexing con lags no contiguos — crea copia
X_train_lags = X_train_lags[:, self.lags - 1]
```

- **Impacto CoW:** Ninguno directo — opera con numpy puro.
- **Impacto arrays read-only:** `sliding_window_view` siempre devuelve vista read-only. El fancy indexing posterior crea una copia writable. ✅ Sin cambio necesario.

#### `_create_window_features()` (líneas 569-628)

```python
y_window_features = y.iloc[-len_y:].to_numpy()  # línea ~625
```

- **Impacto CoW:** `.iloc[...]` devuelve vista lazy; `.to_numpy()` devuelve array read-only en pandas 3.
- **Riesgo:** Si `RollingFeatures.transform()` intenta mutar el array, fallará.
- **Acción:** Verificar que `RollingFeatures.transform()` no muta el input array.

#### `_create_train_X_y()` (líneas 629-845)

```python
# línea ~802: conversión de exog
exog = exog.to_numpy()

# línea ~810: concatenación
X_train = pd.concat(X_train, axis=1)

# línea ~819-820: construcción de DataFrame desde numpy
X_train = pd.DataFrame(data=X_train, index=train_index, columns=...)
```

- **Impacto CoW:**
  - `pd.concat(X_train, axis=1)` — sin parámetro `copy`, funciona igual. ✅
  - `pd.DataFrame(data=X_train, ...)` — ahora copia el array numpy (antes compartía). **Overhead nuevo.**

- **Impacto arrays read-only:** `exog.to_numpy()` devuelve read-only. Si el array se usa solo para `np.concatenate`, no hay problema (concatenate crea copia). ✅

#### `fit()` (líneas 968-1093)

```python
# línea 1091: copia defensiva de last_window
self.last_window_ = y.iloc[-self.window_size_:].copy()

# línea 1067: almacena frecuencia
self.index_freq_ = y.index.freq
```

- **Impacto CoW:** `.copy()` en línea 1091 es ahora puramente redundante — CoW protege la referencia. **Eliminar `.copy()` evitaría una copia innecesaria.**
- **Nota:** La eliminación de `.copy()` requiere verificar que la referencia lazy no se invalide prematuramente (e.g., si `y` es gc-collected antes de usar `last_window_`). En la práctica, `self.last_window_` mantiene la referencia viva, así que CoW protege correctamente.

#### `_create_predict_inputs()` (líneas 1165-1313)

```python
# línea 1271: copia forzada
last_window_values = last_window.to_numpy(copy=True).ravel()

# línea 1304: extracción exog
exog_values = exog.to_numpy()[:steps]
```

- **Impacto CoW:**
  - `.to_numpy(copy=True)` **fuerza una copia incluso cuando CoW habría evitado una.** Este es uno de los principales contribuyentes a la regresión de rendimiento.
  - `.to_numpy()[:steps]` devuelve un slice read-only. Si se muta después → error.

- **Análisis de necesidad del `copy=True`:** `last_window_values` se **muta in-place** dentro del bucle de predicción recursiva. Necesita ser writable. **El `copy=True` es NECESARIO** para obtener un array mutable.

  Sin embargo, se puede optimizar: en vez de `.to_numpy(copy=True)`, usar `np.array(last_window)` o `last_window.to_numpy().copy()` — ambos producen el mismo resultado. Con pandas 3, `.to_numpy()` devuelve read-only, y hacer `.copy()` sobre el resultado es semánticamente más claro que `to_numpy(copy=True)`.

#### `_recursive_predict()` (líneas 1314-1401)

- Bucle puro numpy, no usa pandas. ✅ Sin impacto.

### 3.2 `recursive/_forecaster_recursive_multiseries.py`

#### `pd.concat(copy=False)` (líneas ~1190-1258)

```python
X_train = pd.concat(X_train, copy=False)    # línea ~1190
y_train = pd.concat(y_train, copy=False)    # línea ~1191
train_index = pd.concat(train_index, copy=False)  # línea ~1220 
X_train_encoding = pd.concat(X_train_encoding, axis=1, copy=False)  # línea ~1258
```

- **Impacto:** `copy=False` está **deprecado** en pandas 3.0. Emitirá `FutureWarning` y será eliminado en futuras versiones.
- **Acción:** Eliminar el parámetro `copy=False`. Con CoW, `pd.concat()` ya hace lazy-copy por defecto. El resultado neto es equivalente.
- **Compatibilidad dual:** En pandas 2.x, omitir `copy` usa `copy=True` por defecto. Esto podría añadir copia extra en pandas 2.x. **Sin embargo**, la diferencia es mínima porque `pd.concat` con DataFrames ya crea nuevos arrays consolidados en la mayoría de casos.

#### Copias defensivas

```python
v.iloc[-self.window_size:].copy()  # línea ~1333 — last_window per series
exog.copy().to_frame()             # línea ~2093
exog.copy()                        # varias ubicaciones
```

- **Impacto CoW:** Copias redundantes bajo CoW.
- **Acción:** Evaluar si se pueden eliminar. Prioridad menor (no en hot path).

### 3.3 `utils/utils.py`

#### `check_extract_values_and_index()` (~línea 536)

```python
y.isna().to_numpy().any()        # devuelve read-only array, pero .any() solo lee → OK
```

#### `check_exog_dtypes()` (líneas 613-686)

```python
valid_dtypes = ("int", "Int", "float", "Float", "uint")
# ...
elif not dtype.name.startswith(valid_dtypes):
    has_invalid_dtype = True
```

- **Impacto StringDtype:** En pandas 3.0, columnas string tienen `dtype.name == "string"`. Esto **NO** empieza con ningún valid_dtype, por lo que emitirá warning. Este es el **comportamiento deseado** — skforecast quiere que exog sea numérica o categórica.
- **Potencial problema:** Si un usuario tiene exog con columnas "string" que antes eran "object", recibirán un nuevo warning que antes no veían con pandas 2.x. Considerar documentarlo.

#### `cast_exog_dtypes()` (líneas 1440-1470)

```python
if initial_dtype == "category" and exog[col].dtypes == float:
    exog[col] = exog[col].astype(int).astype("category")
else:
    exog[col] = exog[col].astype(initial_dtype)
```

- **Impacto CoW:** La asignación `exog[col] = ...` es una **setitem operación** que bajo CoW copia el DataFrame si tiene otras referencias. Esto podría provocar copias inesperadas si `exog` fue pasada por referencia.
- **Impacto StringDtype:** Si `initial_dtype` es un `StringDtype` guardado de una sesión anterior y el nuevo exog tiene "object", el `.astype(StringDtype())` funcionará pero con overhead.

#### `check_predict_input()` (~líneas 1014, 1185)

```python
if not last_window_index.freq == index_freq_:
```

- Sin impacto directo. Comparación de frecuencias funciona igual. ✅

#### `exog_to_direct()` (líneas 1480-1530)

```python
exog_direct = pd.concat(exog_direct, axis=1) if steps > 1 else exog_direct[0]
```

- Sin parámetro `copy`. ✅

#### Copias defensivas en funciones utilitarias

Varias funciones usan `.copy()` en rutas no-críticas para rendimiento. Sin urgencia de modificar.

### 3.4 `preprocessing/preprocessing.py`

#### `RollingFeatures.transform()` (línea ~1626)

```python
rolling_features = pd.concat(rolling_features, axis=1)
```

- Sin `copy` parameter. ✅

#### `reshape_series_wide_to_long()` y similares

```python
series = pd.concat(series, names=index_names).to_frame(series_col_name)
```

- Sin `copy` parameter. ✅
- Estas funciones no están en el hot path de fit/predict.

#### `TimeSeriesDifferentiator`, `QuantileBinner`

- Operan principalmente con numpy arrays. ✅ Sin impacto significativo.

### 3.5 `model_selection/`

#### `_validation.py`

```python
backtest_predictions = pd.concat(backtest_predictions)        # línea ~583
backtest_predictions = pd.concat(backtest_predictions, axis=0) # línea ~1358
```

- Sin `copy` parameter. ✅
- Performance: `pd.concat` de muchas predicciones podría ser más lento con CoW tracking overhead, pero está fuera del hot path de predicción individual.

#### `_search.py`

```python
results = pd.concat([results, results['params'].apply(pd.Series)], axis=1)
```

- **Impacto CoW:** `results['params']` es una vista lazy; `.apply(pd.Series)` materializa. Sin problema funcional.
- Sin `copy` parameter. ✅

### 3.6 `direct/` y `deep_learning/`

- `ForecasterDirect._create_train_X_y()` tiene `pd.concat(X_train, axis=1)` similar al recursivo. Mismo análisis aplica.
- `ForecasterDirectMultiVariate` ídem.
- `ForecasterRnn` — sin patrones problemáticos adicionales.

---

## 4. Patrones Detectados y Acciones

### 4.1 `.copy()` Defensivo (Redundante con CoW)

| Ubicación | Código | Hot Path? | Acción |
|-----------|--------|-----------|--------|
| `_forecaster_recursive.py:1091` | `y.iloc[-ws:].copy()` | Sí (fit) | ⚠️ NO eliminar — ver análisis |
| `_forecaster_recursive_multiseries.py:~1333` | `v.iloc[-ws:].copy()` | Sí (fit multi) | ⚠️ NO eliminar — ver análisis |
| `_forecaster_recursive_multiseries.py:~2093` | `exog.copy().to_frame()` | Sí (predict multi) | ⚠️ NO eliminar — ver análisis |
| Varias en `utils/utils.py` | `series.copy()`, `exog.copy()` | No | Evaluar caso a caso |

**Análisis de compatibilidad dual:** Con pandas 2.x, `.copy()` es **imprescindible** para proteger contra vistas mutables. Ejemplo del riesgo:

```python
# Sin .copy(), en pandas 2.x:
forecaster.fit(y=data)
data.iloc[-1] = 999  # ← Muta el original
forecaster.predict(steps=5)  # last_window_ corrupto — comparte memoria con data
```

En pandas 2.x, `.iloc[-ws:]` devuelve una **vista directa** sobre el array subyacente. Sin `.copy()`, `last_window_` y `data` comparten memoria → la mutación del usuario corrompe las predicciones.

En pandas 3.x (CoW), `.iloc[-ws:]` devuelve una referencia lazy. Si el usuario muta `data`, CoW copia automáticamente → `last_window_` queda protegido sin `.copy()`.

**Conclusión: MANTENER `.copy()` mientras se soporte pandas 2.x.** El overhead en pandas 3 es mínimo (copia de `window_size` elementos, típicamente 24-168 floats ≈ bytes). El riesgo de eliminarlo en pandas 2 es corrupción silenciosa de predicciones.

### 4.2 `.to_numpy(copy=True)` (Copia Forzada)

| Ubicación | Código | Necesario? | Acción |
|-----------|--------|-----------|--------|
| `_forecaster_recursive.py:1271` | `last_window.to_numpy(copy=True).ravel()` | **SÍ** — array se muta en `_recursive_predict` | Mantener, pero considerar `np.array(last_window, copy=True)` |
| `utils/utils.py:check_extract_values_and_index` | `.to_numpy(copy=True)` | Evaluar si el resultado se muta | Si no se muta → cambiar a `.to_numpy()` |

### 4.3 `pd.concat(copy=False)` (Deprecado)

| Ubicación | Código | Acción |
|-----------|--------|--------|
| `_forecaster_recursive_multiseries.py:~1190` | `pd.concat(X_train, copy=False)` | Eliminar `copy=False` |
| `_forecaster_recursive_multiseries.py:~1191` | `pd.concat(y_train, copy=False)` | Eliminar `copy=False` |
| `_forecaster_recursive_multiseries.py:~1220` | `pd.concat(train_index, copy=False)` | Eliminar `copy=False` |
| `_forecaster_recursive_multiseries.py:~1258` | `pd.concat(X_train_encoding, axis=1, copy=False)` | Eliminar `copy=False` |

**Compatibilidad dual:** Eliminar `copy=False` funciona en ambas versiones. En pandas 2.x, `copy` por defecto es `True`, pero el impacto en rendimiento es mínimo ya que `pd.concat` típicamente genera nuevos arrays consolidados.

### 4.4 Constructores `pd.DataFrame(data=numpy_array)`

| Ubicación | Acción |
|-----------|--------|
| `_forecaster_recursive.py:~819` | `pd.DataFrame(data=X_train, ...)` — ahora copia en pandas 3. No hay forma de evitarlo sin `copy=False` (deprecado). Impacto de rendimiento aceptable. |
| `_forecaster_recursive.py:~834` | `pd.Series(data=y_train, ...)` — mismo caso. |

**Nota:** Estos constructores crean objetos que se devuelven al usuario. La copia es deseable para evitar que el usuario mute los datos internos.

### 4.5 StringDtype

| Ubicación | Impacto | Acción |
|-----------|---------|--------|
| `utils/utils.py:check_exog_dtypes()` | Columnas string ahora triggerean warning (antes no, con "object") | Documentar el cambio de comportamiento. Considerar añadir `"string"` a la lista de dtypes que reciben warning explícito. |
| `utils/utils.py:cast_exog_dtypes()` | Almacenar/restaurar dtypes funciona pero puede haber mismatch entre "object" → "string" | Testear con datos de sesiones cruzadas pandas 2/3. |
| `utils/utils.py:get_exog_dtypes()` | Almacena dtypes incluyendo StringDtype | No necesita cambio inmediato. |

### 4.6 Aliases de Frecuencia

**Estado:** ✅ No se encontraron aliases deprecados en el código fuente de skforecast.

Los aliases utilizados (`'h'`, `'MS'`, `'D'`, `'QS'`, `'1D'`, etc.) son todos compatibles con pandas 3.0.

**Nota:** Los tests y benchmarks también usan aliases modernos. Los archivos de documentación y notebooks de ejemplo deberían verificarse por separado.

---

## 5. Impacto en Rendimiento

### 5.1 ¿Por qué pandas 3 es más lento en skforecast?

Los benchmarks con `ForecasterRecursive` muestran regresión en `fit()` y `predict()`. Las causas principales:

1. **CoW reference tracking overhead:** Cada operación de indexación/slicing ahora mantiene metadata de CoW. En operaciones repetitivas (lag creation, window features), este tracking se acumula.

2. **Constructores copian numpy arrays:** `pd.DataFrame(data=array)` y `pd.Series(data=array)` ahora copian por defecto. En `_create_train_X_y`, esto afecta la construcción del DataFrame final desde arrays numpy.

3. **`.copy()` redundante:** Con CoW, `.copy()` fuerza una copia real inmediata. Sin CoW, `.copy()` también forzaba copia, pero el overhead de tracking de CoW se suma.

4. **`.to_numpy(copy=True)`:** Fuerza copia donde CoW habría evitado una. Sin embargo, en `_create_predict_inputs` es necesario porque el array se muta.

### 5.2 Mediciones Esperadas

| Operación | Causa de overhead | Estimación |
|-----------|------------------|------------|
| `fit()` — `_create_lags()` | Constructor DataFrame desde numpy | +5-15% |
| `fit()` — `_create_train_X_y()` | `pd.concat` + constructor DataFrame | +5-10% |
| `fit()` — `last_window_.copy()` | Copia redundante bajo CoW | +1-3% |
| `predict()` — `_create_predict_inputs()` | `.to_numpy(copy=True)` (necesario) | ~0% (ya copiaba) |
| `predict()` — `_recursive_predict()` | Pure numpy | ~0% |

**Total estimado `fit()`:** +10-25% más lento  
**Total estimado `predict()`:** +0-5% más lento (casi todo es numpy)

### 5.3 Optimizaciones Posibles

1. **Maximizar uso de numpy en `_create_train_X_y`**: El flag `X_as_pandas=False` ya existe y crea todo como numpy hasta el final. Verificar que se usa como default en `fit()`.

2. **Eliminar `.copy()` donde es seguro**: Incluso en pandas 2.x, si la referencia original no se muta, `.copy()` es innecesario.

3. **Evitar ida y vuelta pandas↔numpy**: En `_create_window_features`, `y.iloc[...].to_numpy()` hace slice pandas → numpy. Si el dato ya está en numpy, usarlo directamente.

---

## 6. Estrategia de Compatibilidad Dual

### 6.1 Principios

1. **No usar `copy` parameter en `pd.concat`**: Omitirlo funciona en ambas versiones.
2. **No usar `copy` parameter en constructores**: `pd.DataFrame(data=..., copy=False)` evita deprecation warning pero es un no-op en pandas 3.
3. **Mantener `.to_numpy(copy=True)` donde el array se muta**: Necesario en ambas versiones.
4. **Mantener `.copy()` defensivos**: Necesarios para pandas 2.x. Overhead despreciable en pandas 3.
5. **No usar CoW-specific APIs**: No llamar `pd.options.mode.copy_on_write` ni similar.
6. **Usar `try/except TypeError` para `np.issubdtype`**: pandas 3 StringDtype no es interpretable por numpy (ya aplicado — ver sección 9).

### 6.2 Tabla de Compatibilidad de Cada Cambio Propuesto

| Cambio | ¿Seguro en pandas 2.x? | ¿Mejora pandas 3? | Decisión |
|--------|------------------------|-------------------|----------|
| Eliminar `copy=False` de `pd.concat` | ✅ Sí — `concat` consolida de todos modos | ✅ Elimina FutureWarning | **Hacer** |
| Eliminar `.copy()` en last_window_ | ❌ No — rompe protección contra vistas | ✅ Evita copia redundante | **NO hacer** |
| Mantener `.to_numpy(copy=True)` | ✅ Necesario (array se muta) | ✅ Necesario (array read-only) | **No tocar** |
| StringDtype handling | ✅ Compatible | ✅ Compatible | **No tocar** |
| `try/except` en `np.issubdtype` | ✅ `except` never triggers | ✅ Captura TypeError de StringDtype | **Ya aplicado** |
| Constructor DataFrame overhead | N/A | Inevitable | **No controlable** |

### 6.3 Detección de Versión

Si es necesario diferenciar comportamiento:

```python
import pandas as pd
PANDAS_VERSION = tuple(int(x) for x in pd.__version__.split('.')[:2])
PANDAS_GE_3 = PANDAS_VERSION >= (3, 0)
```

**Recomendación:** Evitar branching por versión. Preferir código que funcione en ambas versiones.

### 6.4 Supresión de Warnings

**Recomendación:** NO suprimir warnings. Mejor corregir el código para no emitirlos.

---

## 7. Escenario: Solo pandas 3

¿Qué pasaría si se dropara pandas 2 y se trabajara exclusivamente con pandas >=3?

### 7.1 Lo que se ganaría

**Eliminar TODOS los `.copy()` defensivos** (~10+ ubicaciones en los forecasters):

```python
# Ahora (compatible pandas 2+3):
self.last_window_ = y.iloc[-ws:].copy().to_frame(...)

# Solo pandas 3 — CoW protege automáticamente:
self.last_window_ = y.iloc[-ws:].to_frame(...)
```

Ubicaciones afectadas:
- `ForecasterRecursive.fit()` — `y.iloc[-ws:].copy()`
- `ForecasterRecursiveMultiSeries._create_train_X_y()` — `v.iloc[-ws:].copy()` por cada serie
- `ForecasterRecursiveMultiSeries._create_predict_inputs()` — `exog.copy()`
- `ForecasterDirect.fit()` — `y.iloc[-ws:].copy()`
- `ForecasterDirectMultiVariate.fit()` — `series.iloc[-ws:].copy()`
- `ForecasterRnn.fit()` — `series.iloc[-max_lag:].copy()`
- `ForecasterStats.fit()` — `y.copy()`
- `ForecasterEquivalentDate.fit()` — `y.copy()`

### 7.2 Lo que NO se evitaría

**Constructor `pd.DataFrame(data=numpy_array)` — Inevitable:**

Este es el overhead principal y **no tiene solución en pandas 3**. Es una decisión de diseño de CoW: pandas debe "poseer" sus datos, no compartir con numpy.

```python
# Pandas 2: zero-copy (comparte buffer con numpy)
# Pandas 3: siempre copia (CoW requiere ownership exclusivo)
# No hay parámetro para evitarlo — copy=False está deprecado y es un no-op
```

**`.to_numpy(copy=True)` donde se muta — Sigue necesario:**

Con pandas 3, `.to_numpy()` devuelve read-only. Para mutar el array (necesario en `_recursive_predict`), hay que copiar explícitamente.

### 7.3 Balance Neto

| Flujo | pandas 2 | pandas 3 (solo) | Ganancia vs dual |
|-------|----------|-----------------|------------------|
| Constructor DataFrame | Zero-copy | **Copia** (overhead) | Ninguna |
| `.copy()` defensivos | Copia cada vez | **Eliminables** | ~KB |
| `.iloc` slicing | Vista (zero-copy) | Ref CoW (zero-copy) | ~Igual |
| `.to_numpy(copy=True)` | Copia | Copia | Ninguna |

Ejemplo con 100K filas × 24 lags:

| Operación | pandas 2 | pandas 3-only |
|-----------|----------|---------------|
| `pd.DataFrame(X_train)` | 0 (zero-copy) | **19.2 MB** (copia) |
| `pd.Series(y_train)` | 0 (zero-copy) | **800 KB** (copia) |
| `last_window_.copy()` | **192 B** (copia) | 0 (eliminable) |
| **Total** | **~192 B** | **~20 MB** |

### 7.4 Conclusión

Dropar pandas 2 permitiría eliminar copias defensivas (ahorro de kilobytes), pero el overhead del constructor (megabytes) es **inevitable y domina**. El resultado neto es que **pandas 3-only es inherentemente más lento que pandas 2 en `fit()`**, independientemente de lo que se optimice.

La optimización real que beneficiaría a ambas versiones es **mantener datos en numpy el máximo tiempo posible** y retrasar la conversión a DataFrame, lo cual no depende de la versión de pandas.

---

## 8. Plan de Acción Priorizado

### P0 — Crítico (Afecta funcionalidad)

| # | Acción | Archivos | Compatibilidad | Estado |
|---|--------|----------|---------------|--------|
| 1 | Wrap `np.issubdtype` con `try/except TypeError` en `check_exog_dtypes` | `utils/utils.py` | ✅ pandas 2 + 3 | ✅ **HECHO** |
| 2 | Eliminar `copy=False` de `pd.concat()` | `_forecaster_recursive_multiseries.py` (4 ocurrencias) | ✅ pandas 2 + 3 | Pendiente |
| 3 | Verificar que arrays de `.to_numpy()` no se mutan donde no se usa `copy=True` | `_forecaster_recursive.py`, `utils/utils.py` | ✅ pandas 2 + 3 | Pendiente |

### P1 — Alto (Funcionalidad/Comportamiento)

| # | Acción | Archivos | Compatibilidad | Estado |
|---|--------|----------|---------------|--------|
| 4 | Verificar que `cast_exog_dtypes` maneja transiciones object↔string | `utils/utils.py:1440-1470` | N/A | Pendiente |
| 5 | Añadir tests específicos para pandas 3.0 en CI/CD | `tests/`, CI config | N/A | Pendiente |

### P2 — Medio (Optimización — Mayor impacto)

| # | Acción | Archivos | Compatibilidad | Estado |
|---|--------|----------|---------------|--------|
| 6 | **Refactor: pasar numpy directo al estimador en `fit()` cuando no hay categoricals** (ver Apéndice B) | Todos los forecasters | ✅ pandas 2 + 3 | Pendiente |
| 7 | Documentar cambios de comportamiento para usuarios (StringDtype warnings) | docs, changelog | N/A | Pendiente |

### P3 — Bajo (Mantenimiento)

| # | Acción | Archivos | Compatibilidad | Estado |
|---|--------|----------|---------------|--------|
| 8 | Auditar notebooks y ejemplos para aliases de frecuencia deprecados | `docs/`, `dev/` | N/A | Pendiente |
| 9 | Revisar `.copy()` en funciones utilitarias no-hot-path | `utils/utils.py` | Evaluar caso a caso | Pendiente |

### Acciones Descartadas

| Acción | Razón |
|--------|-------|
| Eliminar `.copy()` en `last_window_` (fit) | ❌ Rompe protección contra vistas en pandas 2.x. Overhead en pandas 3 es despreciable (~bytes). |
| Branching por versión de pandas | ❌ Añade complejidad sin ganancia significativa de rendimiento. |

---

## Apéndice A: Inventario Completo de Patrones

### A.1 Todas las ocurrencias de `pd.concat` en código fuente (no tests)

```
_forecaster_recursive.py:810          pd.concat(X_train, axis=1)
_forecaster_recursive_multiseries.py  pd.concat(X_train, copy=False)           ← CAMBIAR
_forecaster_recursive_multiseries.py  pd.concat(y_train, copy=False)           ← CAMBIAR
_forecaster_recursive_multiseries.py  pd.concat(train_index, copy=False)       ← CAMBIAR
_forecaster_recursive_multiseries.py  pd.concat(X_train_encoding, ..., copy=False)  ← CAMBIAR
_forecaster_direct.py:996             pd.concat(X_train, axis=1)
_forecaster_direct.py:2239            pd.concat((predictions, ...), axis=1)
_forecaster_direct_multivariate.py    pd.concat(X_train, axis=1)
_forecaster_direct_multivariate.py    pd.concat([...], ...)
preprocessing.py:785                  pd.concat(series, names=...)
preprocessing.py:795                  pd.concat(exog, names=...)
preprocessing.py:1626                 pd.concat(rolling_features, axis=1)
utils.py:1512                         pd.concat(exog_direct, axis=1)
_validation.py:276,583,1009,1358,2017 pd.concat(...)
_search.py:514,888,1482,1492,1981,2377 pd.concat(...)
_utils.py:1115,1258,1537,1538,1557,1563 pd.concat(...)
_population_drift.py:871              pd.concat(results, ignore_index=True)
```

### A.2 Todas las ocurrencias de `.copy()` en código fuente (no tests)

Las copias defensivas en el código fuente están distribuidas entre forecasters y utilidades. Las que están en hot paths (fit/predict) son las candidatas principales para optimización.

### A.3 Uso de `.to_numpy()` y `.to_numpy(copy=True)`

- `_forecaster_recursive.py:1271` — `to_numpy(copy=True)` — **NECESARIO** (array se muta en predict loop)
- `utils/utils.py:536,577` — `to_numpy()` seguido de `.any()` — array no se muta → OK
- `_forecaster_recursive.py:802` — `exog.to_numpy()` — array se usa en `np.concatenate` → OK
- `_forecaster_recursive.py:625` — `.to_numpy()` en window features → verificar mutación

---

## 9. Cambios Ya Aplicados

### 9.1 Fix: `np.issubdtype` con StringDtype en `check_exog_dtypes()`

**Archivo:** `utils/utils.py` — función `check_exog_dtypes()`  
**Problema:** En pandas 3, las categorías con valores string usan `pd.StringDtype` en lugar de `object`. Al llamar `np.issubdtype(StringDtype, np.integer)`, numpy lanza `TypeError: Cannot interpret '<StringDtype(na_value=nan)>' as a data type` porque no reconoce el dtype de pandas. Esta excepción se propagaba antes de que skforecast pudiera emitir su propio `TypeError`.

**Fix:** Envolver `np.issubdtype()` en `try/except TypeError` en las dos ramas (DataFrame y Series). Si numpy no puede interpretar el dtype, no es integer → se lanza el `TypeError` correcto de skforecast.

```python
# Antes:
if not np.issubdtype(dtype.categories.dtype, np.integer):
    raise TypeError("Categorical dtypes in exog must contain only integer values...")

# Después:
try:
    is_integer = np.issubdtype(dtype.categories.dtype, np.integer)
except TypeError:
    is_integer = False
if not is_integer:
    raise TypeError("Categorical dtypes in exog must contain only integer values...")
```

**Compatibilidad:** 
- pandas 2.x: `except TypeError` nunca se activa — overhead cero (un `try` sin excepción ~nanosegundos).
- pandas 3.x: Captura el TypeError de numpy y devuelve el error correcto de skforecast.
- Se ejecuta una sola vez durante `fit()` (validación de entrada) — no está en hot path.

**Tests:** 5/5 tests `test_create_train_X_y_TypeError_when_exog_is_categorical_of_no_int` pasan en todos los forecasters.

---

## Apéndice B: Refactor — Pasar Numpy Directo al Estimador en `fit()`

### B.1 Problema

El flujo actual de `fit()` en `ForecasterRecursive` (y los demás forecasters) es:

```
_create_train_X_y(y, exog)
  └─ lags, features → numpy arrays
  └─ np.concatenate(...) → X_train (numpy)
  └─ pd.DataFrame(data=X_train, index=..., columns=...)  ← COPIA en pandas 3 (~20MB para 100K×24)
  └─ pd.Series(data=y_train, index=..., name='y')        ← COPIA en pandas 3 (~800KB)
  └─ return (DataFrame, Series, ...)

fit()
  └─ create_sample_weights(X_train)   ← solo usa X_train.index
  └─ estimator.fit(X=X_train, y=y_train)  ← sklearn convierte a numpy internamente (otra copia)
  └─ y_train.to_numpy()               ← tercera copia para residuos
```

Total de copias innecesarias en pandas 3 (100K filas × 24 lags): **~40 MB**.

### B.2 Solución Propuesta

Separar `_create_train_X_y` en dos niveles:

1. **`_create_train_X_y()`** (uso interno de `fit()`) — devuelve numpy arrays + metadatos.
2. **`create_train_X_y()`** (API pública) — llama al interno y envuelve en DataFrame/Series.

#### Ruta A: Sin exog categórico (`X_as_pandas=False`, ~90% de los casos)

```python
# _create_train_X_y devuelve:
#   X_train: numpy ndarray
#   y_train: numpy ndarray
#   train_index: pandas DatetimeIndex (referencia, sin copia)
#   + metadatos (nombres, dtypes, etc.)

# fit() usa directamente:
sample_weight = self.weight_func(train_index) if self.weight_func else None
self.estimator.fit(X=X_train, y=y_train)  # numpy directo, sin paso por DataFrame
# residuos: y_train ya es numpy, sin .to_numpy()
```

**Copias eliminadas:**
- `pd.DataFrame(data=X_train_np)` → eliminada (~19.2 MB)
- `pd.Series(data=y_train_np)` → eliminada (~800 KB)
- `y_train.to_numpy()` para residuos → eliminada (~800 KB)
- `estimator.fit()` ya no necesita check_array → eliminada (~19.2 MB en sklearn)

#### Ruta B: Con exog categórico (`X_as_pandas=True`)

```python
# _create_train_X_y devuelve:
#   X_train: pandas DataFrame (construido por pd.concat de DataFrames existentes,
#            NO por pd.DataFrame(numpy) → sin copia extra)
#   y_train: numpy ndarray (los lags siguen siendo numpy)
#   train_index: referencia al índice
#   + metadatos

# fit() usa:
sample_weight = self.weight_func(train_index) if self.weight_func else None
self.estimator.fit(X=X_train, y=y_train)  # DataFrame para LightGBM/CatBoost
```

**Sin cambio de overhead:** la ruta B ya construye DataFrames por `pd.concat` de DataFrames existentes, no por `pd.DataFrame(numpy)`. El overhead de pandas 3 en esta ruta es mínimo.

### B.3 Impacto en `create_sample_weights()`

Actual:
```python
def create_sample_weights(self, X_train: pd.DataFrame) -> np.ndarray:
    if self.weight_func is not None:
        sample_weight = self.weight_func(X_train.index)
    return sample_weight
```

La función solo necesita el **índice**, no el DataFrame. Refactor:

```python
def create_sample_weights(self, train_index: pd.Index) -> np.ndarray:
    if self.weight_func is not None:
        sample_weight = self.weight_func(train_index)
    return sample_weight
```

**Compatibilidad API**: `create_sample_weights` se documenta como método interno (prefijo `_` no tiene, pero su uso directo es raro). Si se quiere mantener backward compatibility, se puede aceptar ambos:

```python
def create_sample_weights(self, X_train=None, train_index=None):
    index = train_index if train_index is not None else X_train.index
    ...
```

### B.4 Impacto en Otros Métodos que Llaman `_create_train_X_y`

| Caller | Necesita DataFrame? | Cambio |
|--------|-------------------|--------|
| `fit()` | No (pasa a estimator.fit y residuos) | Usar numpy directo |
| `create_train_X_y()` (público) | Sí (devuelve al usuario) | Envolver numpy en DataFrame al final |
| `train_test_split()` | Sí (devuelve al usuario) | Envolver numpy en DataFrame al final |

### B.5 Estimación de Ganancia

Para 100K filas × 24 lags:

| Versión pandas | Copias actuales | Copias tras refactor | Ahorro |
|----------------|----------------|---------------------|--------|
| **pandas 2** | ~19.2 MB (sklearn check_array) | ~0 MB | **~19 MB** |
| **pandas 3** | ~40 MB (constructor + sklearn + to_numpy) | ~0 MB | **~40 MB** |

El refactor **mejora ambas versiones**, pero el impacto es el doble en pandas 3.

### B.6 Forecasters Afectados

El mismo patrón `_create_train_X_y → DataFrame → estimator.fit` existe en:

- `ForecasterRecursive` — `_forecaster_recursive.py`
- `ForecasterDirect` — `_forecaster_direct.py`
- `ForecasterDirectMultiVariate` — `_forecaster_direct_multivariate.py`
- `ForecasterRecursiveMultiSeries` — `_forecaster_recursive_multiseries.py`
- `ForecasterRecursiveClassifier` — `_forecaster_recursive_classifier.py`

`ForecasterRnn` y `ForecasterStats` tienen flujos diferentes y no se benefician de este refactor.

### B.7 Riesgos

1. **`fit_kwargs`**: Algunos estimadores pueden recibir kwargs que esperan DataFrames (e.g., `eval_set` en LightGBM). Verificar que pasar numpy no rompe nada.
2. **`weight_func`**: Usuarios pueden tener funciones que esperan un DatetimeIndex procedente de un DataFrame. El refactor sigue pasando el index correcto.
3. **Feature names**: sklearn >=1.0 emite warning si el estimador se entrena con feature names (DataFrame) y predice sin ellos o viceversa. Al pasar numpy, no se pasan feature names → verificar que predict también pasa numpy.
