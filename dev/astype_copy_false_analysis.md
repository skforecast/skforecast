# Análisis de `.astype()` en skforecast — Candidatos para `copy=False`

> Fecha: 2026-03-18  
> Rama: `feature_manage_categorical`  
> Scope: Solo código fuente (sin tests)

## Contexto

Cuando `pandas.DataFrame.astype(dtypes)` o `numpy.ndarray.astype(dtype)` se llama sin `copy=False`, **siempre se crea una copia** aunque los tipos ya coincidan. Añadir `copy=False` evita la copia cuando no es necesaria, ahorrando memoria y tiempo de CPU.

---

## Ya tienen `copy=False` (4 sitios) ✅

| Archivo | Línea | Código |
|---------|-------|--------|
| `recursive/_forecaster_recursive.py` | 956 | `X_train.astype(X_train_dtypes, copy=False)` |
| `recursive/_forecaster_recursive.py` | 1755 | `X_predict.astype(X_predict_dtypes, copy=False)` |
| `stats/arar/_arar_base.py` | 177 | `best_phi.astype(float, copy=False)` |
| `stats/arar/_arar_base.py` | 180 | `psi.astype(float, copy=False)` |

---

## CANDIDATOS para añadir `copy=False` 🔶

### Grupo 1: `X_predict.astype(self.exog_dtypes_out_)` — pandas DataFrame

Patrón idéntico en 4 forecasters. Restaura dtypes de exog en la matriz de predicción. Si los dtypes ya coinciden (caso frecuente cuando no hay categoricals), se evita la copia. **Se ejecuta en cada llamada a predict.**

| # | Archivo | Línea | Cambio propuesto |
|---|---------|-------|------------------|
| 1 | `recursive/_forecaster_recursive_multiseries.py` | 2716 | `X_predict.astype(self.exog_dtypes_out_, copy=False)` |
| 2 | `recursive/_forecaster_recursive_classifier.py` | 1604 | `X_predict.astype(self.exog_dtypes_out_, copy=False)` |
| 3 | `direct/_forecaster_direct.py` | 1714 | `X_predict.astype(self.exog_dtypes_out_, copy=False)` |
| 4 | `direct/_forecaster_direct_multivariate.py` | 1946 | `X_predict.astype(self.exog_dtypes_out_, copy=False)` |

**Código actual:**
```python
if categorical_features:
    X_predict = X_predict.astype(self.exog_dtypes_out_)
```

**Código propuesto:**
```python
if categorical_features:
    X_predict = X_predict.astype(self.exog_dtypes_out_, copy=False)
```

---

### Grupo 2: `preprocessing.py` — consolidación de dtypes en `reshape_exog_long_to_dict`

| # | Archivo | Línea | Cambio propuesto |
|---|---------|-------|------------------|
| 5 | `preprocessing/preprocessing.py` | 721 | `v.astype(new_dtypes, copy=False)` |

Algunas columnas de cada DataFrame ya son float; `copy=False` evita copiarlas. Se ejecuta sobre **todos** los DataFrames del dict de exog, multiplicando el efecto.

**Código actual:**
```python
exog_dict = {k: v.astype(new_dtypes) for k, v in exog_dict.items()}
```

**Código propuesto:**
```python
exog_dict = {k: v.astype(new_dtypes, copy=False) for k, v in exog_dict.items()}
```

---

### Grupo 3: `QuantileBinner.transform` — `searchsorted` → dtype configurable

| # | Archivo | Línea | Cambio propuesto |
|---|---------|-------|------------------|
| 6 | `preprocessing/preprocessing.py` | 2554 | `.astype(self.dtype, copy=False)` |

`searchsorted` devuelve `np.intp` (normalmente int64). Si `self.dtype` es int64, la copia sobra.

**Código actual:**
```python
bin_indices = np.searchsorted(
    self.internal_edges_, X, side='right'
).astype(self.dtype)
```

**Código propuesto:**
```python
bin_indices = np.searchsorted(
    self.internal_edges_, X, side='right'
).astype(self.dtype, copy=False)
```

---

### Grupo 4: `_population_drift.py` — histograma sobre datos ya float64

| # | Archivo | Línea | Cambio propuesto |
|---|---------|-------|------------------|
| 7 | `drift_detection/_population_drift.py` | 502 | `ref.astype("float64", copy=False)` |

**Código actual:**
```python
bins_edges = np.histogram_bin_edges(ref.astype("float64"), bins='doane')
```

**Código propuesto:**
```python
bins_edges = np.histogram_bin_edges(ref.astype("float64", copy=False), bins='doane')
```

---

### Grupo 5: `_arima_base.py` — matrices del filtro de Kalman

Estas arrays se pasan a funciones numba que **no mutan** los originales (las copian internamente con `.copy()` o usan operaciones de matriz que crean arrays nuevos). Si ya son float64, la copia sobra.

#### Función `arima_kalman_loglik` (L1541-1543)

| # | Archivo | Línea | Cambio propuesto |
|---|---------|-------|------------------|
| 8 | `stats/arima/_arima_base.py` | 1541 | `model.filtered_state.astype(np.float64, copy=False)` |
| 9 | `stats/arima/_arima_base.py` | 1542 | `model.filtered_covariance.astype(np.float64, copy=False)` |
| 10 | `stats/arima/_arima_base.py` | 1543 | `model.predicted_covariance.astype(np.float64, copy=False)` |

**Código actual:**
```python
a = model.filtered_state.astype(np.float64)
P = model.filtered_covariance.astype(np.float64)
Pn = model.predicted_covariance.astype(np.float64)
```

**Código propuesto:**
```python
a = model.filtered_state.astype(np.float64, copy=False)
P = model.filtered_covariance.astype(np.float64, copy=False)
Pn = model.predicted_covariance.astype(np.float64, copy=False)
```

#### Función `kalman_forecast` (L1746-1750)

| # | Archivo | Línea | Cambio propuesto |
|---|---------|-------|------------------|
| 11 | `stats/arima/_arima_base.py` | 1746 | `ss.transition_matrix.astype(np.float64, copy=False)` |
| 12 | `stats/arima/_arima_base.py` | 1747 | `ss.innovation_covariance.astype(np.float64, copy=False)` |
| 13 | `stats/arima/_arima_base.py` | 1748 | `ss.observation_vector.astype(np.float64, copy=False)` |
| 14 | `stats/arima/_arima_base.py` | 1749 | `ss.filtered_state.astype(np.float64, copy=False)` |
| 15 | `stats/arima/_arima_base.py` | 1750 | `ss.filtered_covariance.astype(np.float64, copy=False)` |

**Código actual:**
```python
T = ss.transition_matrix.astype(np.float64)
V = ss.innovation_covariance.astype(np.float64)
Z = ss.observation_vector.astype(np.float64)
a = ss.filtered_state.astype(np.float64)
P = ss.filtered_covariance.astype(np.float64)
```

**Código propuesto:**
```python
T = ss.transition_matrix.astype(np.float64, copy=False)
V = ss.innovation_covariance.astype(np.float64, copy=False)
Z = ss.observation_vector.astype(np.float64, copy=False)
a = ss.filtered_state.astype(np.float64, copy=False)
P = ss.filtered_covariance.astype(np.float64, copy=False)
```

#### Función `_prepare_exog` (L1808)

| # | Archivo | Línea | Cambio propuesto |
|---|---------|-------|------------------|
| 16 | `stats/arima/_arima_base.py` | 1808 | `exog.values.astype(np.float64, copy=False)` |

**Código actual:**
```python
exog_matrix = exog.values.astype(np.float64)
```

**Código propuesto:**
```python
exog_matrix = exog.values.astype(np.float64, copy=False)
```

---

### Grupo 6: `_arima_base.py` — doble copia `.astype(np.float64).copy()`

Patrón: `c.fixed.astype(np.float64).copy()`. Si `c.fixed` ya es float64, `.astype()` copia y luego `.copy()` copia otra vez (**doble copia**). Con `copy=False` se garantiza exactamente una copia:

| # | Archivo | Línea | Código actual | Cambio propuesto |
|---|---------|-------|---------------|------------------|
| 17 | `stats/arima/_arima_base.py` | 2354 | `c.fixed.astype(np.float64).copy()` | `c.fixed.astype(np.float64, copy=False).copy()` |
| 18 | `stats/arima/_arima_base.py` | 2358 | `c.fixed.astype(np.float64).copy()` | `c.fixed.astype(np.float64, copy=False).copy()` |
| 19 | `stats/arima/_arima_base.py` | 2459 | `c.fixed.astype(np.float64).copy()` | `c.fixed.astype(np.float64, copy=False).copy()` |
| 20 | `stats/arima/_arima_base.py` | 2467 | `c.fixed.astype(np.float64).copy()` | `c.fixed.astype(np.float64, copy=False).copy()` |
| 21 | `stats/arima/_arima_base.py` | 2639 | `c.fixed.astype(np.float64).copy()` | `c.fixed.astype(np.float64, copy=False).copy()` |

---

## NO son candidatos (no cambiar) ❌

| Patrón | Sitios | Razón |
|--------|--------|-------|
| `.binner.transform(...).astype(int)` | 9 sitios (todos los forecasters) | `searchsorted` → int, conversión siempre real |
| `.astype(object)` + `.astype(int)` para CatBoost | `_forecaster_recursive.py` L1165-1166, `utils.py` L2619-2620 | Siempre cambia de tipo |
| `.astype(np.float32)` para `tree_.predict` | `utils.py` L2598, L2608 | float64→float32, siempre real |
| `.astype(str)` para display/repr | `_forecaster_base.py` L78, `plot.py` L157 | Siempre cambia de tipo |
| `.astype('category')` para encoding | `_forecaster_recursive_multiseries.py` L1210 | Siempre cambia de tipo |
| `.astype('timedelta64[D]').astype(int)` | `_experimental.py` L61, L69 | Cadena de conversiones reales |
| `cast_exog_dtypes` en `utils.py` | L1593, L1598, L1600 | Ya tienen guards `if dtype != target` |
| `corr.index.astype('int64')` | `utils.py` L2429 | Conversión de índice, siempre necesaria |
| `(idx.weekday >= 5).astype(int)` | `preprocessing.py` L903 | bool→int, siempre real |
| `getattr(X.index, attr).astype(int)` | `preprocessing.py` L919 | datetime attr→int, siempre real |
| `best_phi = phi.astype(float, copy=True)` | `_arar_base.py` L168 | Copia intencional (snapshot) |

---

## Resumen de impacto

| Prioridad | Grupo | Sitios | Impacto |
|-----------|-------|--------|---------|
| **Alta** | Grupo 1 (X_predict en forecasters) | 4 | Se ejecuta en **cada predict**. Con series largas y muchas exog, ahorra memoria y tiempo |
| **Alta** | Grupo 6 (doble copia en ARIMA) | 5 | Elimina una copia redundante en **cada evaluación del optimizador** |
| **Media** | Grupo 5 (Kalman matrices) | 8 | Se ejecuta en cada paso del filtro Kalman; datos normalmente ya float64 |
| **Media** | Grupo 2 (reshape_exog_long_to_dict) | 1 | Se multiplica por el nº de series en el dict |
| **Baja** | Grupo 3 (QuantileBinner) | 1 | Normalmente dtype ya coincide |
| **Baja** | Grupo 4 (PopulationDriftDetector) | 1 | Solo en fit, una vez |
| **Baja** | L1808 (_prepare_exog) | 1 | Exog normalmente ya float64 |

**Total: 21 sitios candidatos para `copy=False`**
