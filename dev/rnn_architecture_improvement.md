# Mejora de la arquitectura RNN: Unificación del modelo con y sin exógenas

## Resumen

La función `create_and_compile_model` en `skforecast.deep_learning.utils` genera dos arquitecturas **fundamentalmente diferentes** según si se proporcionan variables exógenas o no. Esto causa que **el modelo con exógenas deje de aprender** debido a limitaciones arquitectónicas críticas. La solución propuesta unifica ambas en una sola arquitectura basada en `Dense + Reshape`.

---

## Estado actual

### Dos funciones internas, dos arquitecturas distintas

`create_and_compile_model` delega a:

- `_create_and_compile_model_no_exog()` → usa **Dense + Reshape** (funciona bien)
- `_create_and_compile_model_exog()` → usa **RepeatVector + TimeDistributed** (problemática)

### Arquitectura SIN exógenas (funciona correctamente)

```
series_input: (batch, n_lags, n_series)
       │
       ▼
  RNN layers (LSTM/GRU/SimpleRNN)
  return_sequences=False en última capa
       │
       ▼
  (batch, recurrent_units[-1])        ← vector resumen de la serie temporal
       │
       ▼
  Dense(dense_units)                  ← capas densas regulares
       │
       ▼
  Dense(n_levels * steps)             ← pesos INDEPENDIENTES por (step, level)
       │
       ▼
  Reshape(steps, n_levels)            ← reestructura a formato (batch, steps, n_levels)
```

**Parámetros de la capa de salida:** `dense_units × (n_levels × steps) + (n_levels × steps)`

Ejemplo con `dense_units=64`, `steps=12`, `n_levels=2`:
- `64 × 24 + 24 = 1,560` parámetros
- Cada combinación (step, level) tiene **pesos independientes**

### Arquitectura CON exógenas (problemática)

```
series_input: (batch, n_lags, n_series)     exog_input: (batch, steps, n_exog)
       │                                           │
       ▼                                           │
  RNN layers → (batch, recurrent_units[-1])        │
       │                                           │
       ▼                                           │
  RepeatVector(steps)                              │
       │  → (batch, steps, recurrent_units[-1])    │
       ▼                                           ▼
  Concatenate(axis=-1) ────────────────────────────┘
       │  → (batch, steps, recurrent_units[-1] + n_exog)
       ▼
  TimeDistributed(Dense(dense_units))    ← MISMOS pesos en cada step
       │
       ▼
  TimeDistributed(Dense(n_levels))       ← MISMOS pesos en cada step
       │
       ▼
  output: (batch, steps, n_levels)
```

**Parámetros de la capa de salida:** `dense_units × n_levels + n_levels`

Ejemplo con `dense_units=64`, `steps=12`, `n_levels=2`:
- `64 × 2 + 2 = 130` parámetros (¡12x menos que sin exógenas!)
- Los **mismos pesos** se aplican en todos los steps

---

## Problemas identificados

### 1. RepeatVector + TimeDistributed elimina la capacidad de aprender patrones por step (CRÍTICO)

`RepeatVector(steps)` copia el **mismo vector** a todos los pasos temporales. Luego, `TimeDistributed(Dense)` aplica la **misma matriz de pesos** W en cada step:

$$\hat{y}_t = W \cdot h_t + b$$

Donde $h_t$ para la parte de la serie es **idéntico** en todos los pasos (es el vector repetido). La **única diferenciación** entre step 1 y step 12 proviene de las ~5 variables exógenas concatenadas junto a ~100 features repetidas. Esto produce un ratio señal/ruido de aproximadamente 5:100.

**Consecuencia:** El modelo no puede aprender que la predicción a 1 paso requiere un mapeo diferente que la predicción a 12 pasos. Produce predicciones casi constantes.

### 2. El RNN nunca ve las variables exógenas (ALTO)

Las exógenas se inyectan **después** de todas las capas recurrentes. El LSTM/GRU solo procesa los lags de las series, sin acceso a las exógenas. Relaciones temporales como "la temperatura de ayer afecta las ventas de hoy" son **invisibles** para la red recurrente.

### 3. El gradiente se degrada por conflicto entre steps (MEDIO)

Con `TimeDistributed`, los gradientes de step 1 y step 12 fluyen a través de los **mismos pesos**. Si el mapeo óptimo para step 1 contradice el de step 12 (lo cual es habitual: predicciones a corto plazo vs. largo plazo), los gradientes se cancelan parcialmente. El resultado es un compromiso subóptimo para todos los steps.

### 4. Inconsistencia arquitectónica (MEDIO)

El usuario no espera que añadir exógenas cambie completamente la estrategia de salida del modelo. Esto dificulta la depuración y comparación de experimentos.

### 5. Argumentos mutables como valores por defecto (BAJO)

```python
# Actual (peligroso):
def f(recurrent_layers_kwargs={"activation": "tanh"}):
    ...

# Correcto:
def f(recurrent_layers_kwargs=None):
    if recurrent_layers_kwargs is None:
        recurrent_layers_kwargs = {"activation": "tanh"}
```

Aunque se usa `deepcopy` internamente, los defaults mutables compartidos son un antipatrón conocido de Python.

---

## Solución propuesta

### Arquitectura unificada: Dense + Reshape con Flatten para exógenas

```
series_input: (batch, n_lags, n_series)    exog_input: (batch, steps, n_exog)
       │                                          │  [opcional]
       ▼                                          ▼
  RNN layers                                 Flatten()
  (return_sequences=False)                        │
       │                                          │
       ▼                                          ▼
  (batch, rnn_units)                     (batch, steps * n_exog)
       │                                          │
       └──────────── Concatenate ─────────────────┘
                         │
                         ▼
                  (batch, rnn_units [+ steps * n_exog])
                         │
                         ▼
                   Dense(dense_units)          ← capas densas regulares
                         │
                         ▼
                Dense(n_levels * steps)        ← pesos independientes por step
                         │
                         ▼
                Reshape(steps, n_levels)
```

### Ventajas de la solución

| Aspecto | Actual (exog) | Propuesto |
|---------|---------------|-----------|
| Pesos por step | Compartidos (TimeDistributed) | Independientes (Dense + Reshape) |
| Params capa salida (ejemplo) | 130 | 1,560 |
| Exog visible para Dense | Solo a nivel de step individual | Todas las exog de todos los steps simultáneamente |
| Relaciones cross-step de exog | Imposible | Posible (e.g., exog en step 5 afecta predicción en step 3) |
| Consistencia con modelo sin exog | Diferente arquitectura | Misma arquitectura |
| Código duplicado | ~200 líneas duplicadas | Eliminable con merge de funciones |

### Comparación de parámetros

Para `recurrent_units=100`, `dense_units=64`, `steps=12`, `n_levels=2`, `n_exog=5`:

| Componente | No-exog actual | Exog actual | Propuesto (ambos) |
|------------|---------------|-------------|-------------------|
| LSTM(100) | ~40,800 | ~40,800 | ~40,800 |
| Dense(64) | 6,464 | 8,384¹ | 6,464 / 10,304² |
| Output Dense | 1,560 | 130 | 1,560 |
| Reshape | 0 | 0 | 0 |
| **Total** | **~48,824** | **~49,314** | **~48,824 / ~52,664** |

¹ TimeDistributed(Dense(64)) sobre 105 features: (105×64+64)=6,784, pero compartido  
² Dense(64) sobre (100 + 12×5)=160 features: (160×64+64)=10,304

El incremento de parámetros es marginal comparado con el LSTM y el beneficio es enorme.

---

## Plan de implementación

### Archivos afectados

| Archivo | Cambio | Complejidad |
|---------|--------|-------------|
| `skforecast/deep_learning/utils.py` | Arquitectura de `_create_and_compile_model_exog()` | **Principal** |
| `skforecast/deep_learning/_forecaster_rnn.py` | 1 línea en `__init__()` (extracción de `n_levels_out`) | **Trivial** |
| Tests de arquitectura | Actualizar tests de estructura del modelo | **Moderada** |

### Cambio 1: `utils.py` — `_create_and_compile_model_exog()`

**Reemplazar** (líneas ~300-370):

```python
# ACTUAL:
x = RepeatVector(steps, name="repeat_vector")(x)
if exog is not None:
    x = Concatenate(axis=-1, name="concat_exog")([x, exog_input])
# ... TimeDistributed(Dense) layers ...
output = TimeDistributed(Dense(n_levels), ...)(x)
```

**Por:**

```python
# PROPUESTO:
from keras.layers import Flatten

if exog is not None:
    exog_flat = Flatten(name="exog_flatten")(exog_input)
    x = Concatenate(axis=-1, name="concat_exog")([x, exog_flat])

# Dense layers (regulares, NO TimeDistributed)
for i, units in enumerate(dense_units):
    x = Dense(units, ...)(x)

# Output layer
x = Dense(n_levels * steps, ...)(x)
output = Reshape((steps, n_levels), name="reshape")(x)
```

### Cambio 2: `_forecaster_rnn.py` — `__init__()`

**Reemplazar** (líneas ~320-328):

```python
# ACTUAL:
self.n_levels_out = self.estimator.get_layer('output_dense_td_layer').output.shape[-1]
self.exog_in_ = True if "exog_input" in self.layers_names else False
if self.exog_in_:
    self.n_exog_in = self.estimator.get_layer('exog_input').output.shape[-1]
else:
    self.n_exog_in = None
    # NOTE: This is needed because the Reshape layer changes the output shape
    self.n_levels_out = int(self.n_levels_out / self.max_step)
```

**Por:**

```python
# PROPUESTO:
self.n_levels_out = self.estimator.get_layer('output_dense_td_layer').output.shape[-1]
self.exog_in_ = True if "exog_input" in self.layers_names else False
if self.exog_in_:
    self.n_exog_in = self.estimator.get_layer('exog_input').output.shape[-1]
else:
    self.n_exog_in = None
# Ahora SIEMPRE necesario: ambas arquitecturas usan Dense + Reshape
self.n_levels_out = int(self.n_levels_out / self.max_step)
```

### Cambio 3 (opcional): Unificar en una sola función

Al tener la misma arquitectura, se puede eliminar `_create_and_compile_model_no_exog` y `_create_and_compile_model_exog`, y tener una sola función interna. El bloque condicional se reduce a:

```python
if exog is not None:
    exog_input = Input(shape=(steps, n_exog), name="exog_input")
    inputs.append(exog_input)

# ... RNN layers (idéntico) ...

if exog is not None:
    exog_flat = Flatten(name="exog_flatten")(exog_input)
    x = Concatenate(axis=-1, name="concat_exog")([x, exog_flat])

# ... Dense + Reshape (idéntico) ...
```

Esto elimina ~200 líneas de código de validación duplicado.

---

## Métodos de ForecasterRnn que NO requieren cambios

Los métodos a continuación no se ven afectados porque las **shapes de entrada y salida del modelo no cambian**:

- `fit()` — sigue llamando `model.fit(x=[X_train, exog_train], y=y_train)`
- `predict()` — sigue llamando `model.predict(X)` con salida `(1, steps, n_levels)`
- `_create_predict_inputs()` — sigue construyendo exog con shape `(1, steps, n_exog)`
- `_create_lags()` — manipulación de datos pura, sin awareness del modelo
- `_create_train_X_y()` — produce los mismos arrays con las mismas shapes
- `set_in_sample_residuals()` — usa `model.predict()`, mismas shapes
- `_predict_interval_conformal()` — usa `model.predict()`, mismas shapes
- `set_params()`, `set_fit_kwargs()` — agnósticos a la arquitectura

---

## Mejoras futuras (fuera de scope)

### Exógenas históricas dentro del RNN

Permitir que las exógenas pasadas se alimenten **dentro** del RNN junto con las series:

```
Input: (batch, n_lags, n_series + n_exog_hist) → RNN → ...
```

Esto permitiría al RNN aprender correlaciones temporales cruzadas (e.g., "cuando la temperatura subió hace 2 días, las ventas suben hoy").

### Step embeddings

Añadir un encoding posicional aprendible por paso de predicción. Con `Dense + Reshape` esto es implícito (pesos independientes por step), pero podría ser útil en arquitecturas más complejas.

### Attention mechanism

En lugar de comprimir toda la secuencia en un vector fijo, permitir que la red "atienda" a diferentes pasos del input según el step de predicción.

---

## Referencias

- Keras LSTM: https://keras.io/api/layers/recurrent_layers/lstm/
- Keras Dense: https://keras.io/api/layers/core_layers/dense/
- N-BEATS (Direct multi-output): Oreshkin et al., 2019
- Seq2Seq para time series: Sutskever et al., 2014
