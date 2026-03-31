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

---

## Debate: Causalidad dentro del horizonte de predicción

### El problema con la solución Flatten propuesta

En la solución propuesta, `Flatten()` transforma `exog_input: (batch, steps, n_exog)` en un vector plano `(batch, steps * n_exog)` que se concatena con la salida del RNN antes de las capas Dense. Esto significa que cuando el modelo predice el step 1, tiene acceso simultáneo a los valores de las exógenas en los steps 2, 3, ..., K.

```
exog en step 1: [e₁]
exog en step 2: [e₂]   → Flatten → [e₁, e₂, ..., eₖ] → Dense → predicción step 1
...
exog en step K: [eₖ]                                         ↑
                                            ¿debería e₂..eₖ influir aquí?
```

### Argumento a favor del cross-step (defensa del documento original)

Todas las exógenas del horizonte son **valores conocidos en el momento de inferencia** — el usuario los provee explícitamente en `predict(exog=...)`. No hay *data leakage* en el sentido clásico (no se usa información del futuro que aún no ha ocurrido en producción). Desde esta perspectiva, la red puede aprender relaciones útiles: si se sabe que el precio sube en el step 5, quizás la demanda en el step 3 también se ve afectada.

### Argumentos en contra (causalidad dentro del horizonte)

1. **Overfitting en training:** el modelo puede aprender atajos espurios usando el exog del step 12 para predecir el step 1. Estos atajos no generalizan si la correlación entre exógenas cambia en producción.
2. **Gradientes conflictivos entre steps:** las derivadas de la pérdida en step 1 fluyen a través de los mismos parámetros que las derivadas de step 12, pero usando el vector `[e₁, ..., eₖ]` completo. El mapeo óptimo puede ser contradictorio entre steps.
3. **Principio de diseño:** en series temporales, el mapeo `t → predicción t` debería depender causalmente solo de información disponible *hasta* `t`. Aunque todos los exog sean conocidos, respetar este principio mejora la interpretabilidad y la generalización.
4. **Inconsistencia con el modelo no-exog:** sin exógenas, el modelo no tiene este problema (la Dense opera igual para todos los steps). Con exógenas, se introduce una asimetría informacional entre steps que no existe en el caso base.

---

## Alternativas arquitectónicas

### Opción A — Flatten (propuesta en "Solución propuesta")

```
series_input: (batch, n_lags, n_series)    exog_input: (batch, steps, n_exog)
       │                                          │
       ▼                                          ▼
  RNN layers                               Flatten()
  (return_sequences=False)                        │
       │                                          │
       ▼                                          ▼
  (batch, rnn_units)               (batch, steps * n_exog)
       │                                          │
       └──────────────── Concatenate ─────────────┘
                               │
                    (batch, rnn_units + steps * n_exog)  ← TODAS las exog visibles
                               │
                               ▼
                        Dense(dense_units)
                               │
                               ▼
                    Dense(n_levels * steps)
                               │
                               ▼
                    Reshape(steps, n_levels)
                               │
                               ▼
                    output: (batch, steps, n_levels)
```

**Parámetros Dense de entrada (ejemplo `rnn_units=100`, `steps=12`, `n_exog=5`, `dense_units=64`):**
- Input Dense: `(100 + 60) × 64 + 64 = 10,304`
- Output Dense: `64 × 24 + 24 = 1,560`

| | |
|---|---|
| ✅ | Pesos independientes por step |
| ✅ | Arquitectura unificada con/sin exog |
| ✅ | Implementación simple |
| ❌ | Step `t` accede a exog de steps `t+1 ... K` (no causal dentro del horizonte) |

---

### Opción B — TimeDistributed con Step Embeddings

Mantiene la estructura TimeDistributed actual pero añade un **embedding de posición** (uno por step) al vector de cada timestep, haciendo que los pesos compartidos sean "conscientes del step" sin duplicarlos.

```
series_input: (batch, n_lags, n_series)    exog_input: (batch, steps, n_exog)
       │                                          │
       ▼                                          │
  RNN layers → (batch, rnn_units)                 │
       │                                          │
       ▼                                          │
  RepeatVector(steps)                             │
       │ → (batch, steps, rnn_units)              │
       │                                          │
       │         step_embedding: (1, steps, d_s)  │     ← embedding aprendible
       │         Broadcast → (batch, steps, d_s)  │
       │                           │              │
       ▼                           ▼              ▼
  Concatenate([rnn_repeated, step_embed, exog_input])
       │ → (batch, steps, rnn_units + d_s + n_exog)
       ▼
  TimeDistributed(Dense(dense_units))    ← pesos COMPARTIDOS pero step-aware vía embedding
       │
       ▼
  TimeDistributed(Dense(n_levels))
       │
       ▼
  output: (batch, steps, n_levels)
```

**Nota de implementación:** El step embedding es un `Embedding(input_dim=steps, output_dim=d_s)` aplicado sobre un tensor de índices `[0, 1, ..., steps-1]`, o bien un tensor fijo de posiciones sinusoidales (sin parámetros extra). El resultado se repite en la dimensión de batch con `Lambda` o `tf.broadcast_to`.

| | |
|---|---|
| ✅ | Causal: step `t` solo ve exog en step `t` |
| ✅ | Diferenciación mínima de parámetros entre steps (embedding `d_s` pequeño) |
| ✅ | Fácil de añadir sobre la arquitectura actual |
| ❌ | Pesos Dense aún **compartidos** entre steps (mismo problema raíz, atenuado) |
| ❌ | Arquitectura diferente a la del modelo sin exógenas |
| ❌ | El embedding añade complejidad de implementación sin resolver el problema principal |

---

### Opción C — K cabezas Dense separadas (una por step)

Cada step tiene su propia cabeza Dense completamente independiente. Solo se concatena la exog correspondiente al step `i`, garantizando causalidad. Las K salidas se apilan con `keras.ops.stack` (backend-agnostic: TensorFlow, PyTorch, JAX).

```
series_input: (batch, n_lags, n_series)    exog_input: (batch, steps, n_exog)
       │                                          │
       ▼                                   exog[:,0,:]  exog[:,1,:]  ...  exog[:,K-1,:]
  RNN layers                                    │             │                  │
  (return_sequences=False)                      ▼             ▼                  ▼
       │
       ▼
  (batch, rnn_units)
       │
       ├──── Concatenate([rnn, exog[:,0,:]]) → Dense_head_0(dense_units) → Dense_head_0(n_levels)
       │                                                                           │
       ├──── Concatenate([rnn, exog[:,1,:]]) → Dense_head_1(dense_units) → Dense_head_1(n_levels)
       │                                                                           │
       ├──── ...                                                                   │
       │                                                                           │
       └──── Concatenate([rnn, exog[:,K-1,:]]) → Dense_head_{K-1}(d_u) → Dense_head_{K-1}(n_levels)
                                                                               │
                                                          keras.ops.stack(outputs, axis=1)
                                                                               │
                                                                               ▼
                                                                  output: (batch, steps, n_levels)
```

**Sin exógenas:** el `Concatenate` desaparece; cada cabeza recibe solo `(batch, rnn_units)`. La arquitectura es idéntica al modelo sin exog salvo que los pesos de la capa de salida están explícitamente separados en K Dense independientes en lugar de un único `Dense(n_levels * steps) + Reshape`.

**Parámetros por cabeza (ejemplo `rnn_units=100`, `n_exog=5`, `dense_units=64`, `n_levels=2`):**
- Dense(64): `(100 + 5) × 64 + 64 = 6,784`
- Dense(2): `64 × 2 + 2 = 130`
- Por cabeza: 6,914 parámetros
- Total 12 cabezas: `12 × 6,914 = 82,968` (vs. 10,304 + 1,560 = 11,864 en Opción A)

**Fragmento de implementación Keras:**

```python
from keras import ops

head_outputs = []
for i in range(steps):
    if exog is not None:
        exog_step = exog_input[:, i, :]          # (batch, n_exog)
        h = Concatenate(name=f"concat_step_{i}")([x, exog_step])
    else:
        h = x
    for j, units in enumerate(dense_units):
        h = Dense(units, activation=activation, name=f"dense_step_{i}_{j}")(h)
    h = Dense(n_levels, name=f"output_step_{i}")(h)  # (batch, n_levels)
    head_outputs.append(h)

# Stack: lista de K tensores (batch, n_levels) → (batch, steps, n_levels)
output = ops.stack(head_outputs, axis=1)
```

| | |
|---|---|
| ✅ | Pesos **completamente independientes** por step |
| ✅ | Causal: step `t` solo ve exog en step `t` |
| ✅ | Consistente con modelo sin exog (misma estructura, sin Concatenate) |
| ✅ | Sin `RepeatVector`, `TimeDistributed`, ni `Flatten` |
| ❌ | K × más capas Dense → modelo más grande en parámetros |
| ❌ | Código de construcción del modelo más complejo (bucle explícito) |
| ❌ | Nombres de capas dinámicos (`dense_step_0_0`, ...) complican `get_layer()` en `__init__` |

---

## Tabla comparativa de las tres alternativas

| Aspecto | A — Flatten | B — Step Embeddings | C — K Heads |
|---------|:-----------:|:-------------------:|:-----------:|
| Pesos independientes por step | ✅ | ❌ (compartidos) | ✅ |
| Causalidad dentro del horizonte | ❌ | ✅ | ✅ |
| Consistencia sin/con exog | ✅ | ❌ | ✅ |
| Complejidad de implementación | Baja | Media | Alta |
| Parámetros (ejemplo K=12, E=5) | ~11,864 | similar al actual | ~82,968 |
| Cambios en `_forecaster_rnn.py` | Mínimos | Mínimos | Significativos (get_layer) |
| Elimina código duplicado (~200 líneas) | ✅ | ❌ | ✅ |
| Riesgo de overfitting cross-step | Alto | Bajo | Ninguno |

### Recomendación

- **Si la prioridad es la implementación mínima y la unificación de código:** Opción A, aceptando el tradeoff de causalidad con la justificación de que las exógenas futuras son valores conocidos.
- **Si la prioridad es la corrección arquitectónica estricta:** Opción C, siendo conscientes del mayor coste en parámetros y complejidad de código. Requiere refactorizar cómo `__init__` extrae `n_levels_out` (buscar la capa `output_step_0` en lugar de `output_dense_td_layer`).
- **Opción B no resuelve el problema raíz** (pesos compartidos) y añade complejidad sin beneficio claro respecto a Opción A.

---

## SOLUCIÓN FINAL

Tras evaluar las tres alternativas y revisar el código actual (`utils.py`, `_forecaster_rnn.py` y los tests), la solución final es la **Opción A (Flatten + Dense + Reshape)** con las siguientes mejoras adicionales:

1. **Unificación completa** en una sola función interna (eliminar `_create_and_compile_model_exog` y `_create_and_compile_model_no_exog`)
2. **Renombrado de la capa de salida** de `output_dense_td_layer` a `output_dense_layer` (ya no hay `TimeDistributed`)
3. **Corrección de defaults mutables** en las firmas de función
4. **Eliminación de imports innecesarios** (`RepeatVector`, `TimeDistributed`)

### Justificación de la Opción A sobre la C

El problema real que se resuelve es la **falta de expresividad** (pesos compartidos via TimeDistributed), no la causalidad intra-horizonte. Argumentos clave:

- Las exógenas del horizonte son **valores conocidos** proporcionados por el usuario en `predict(exog=...)`. No hay data leakage.
- El ratio de features exógenas vs. RNN output es típicamente bajo (~5 exog vs. ~100 rnn_units), limitando el riesgo de overfitting cross-step.
- La Opción C multiplica los parámetros por K (7x en el ejemplo), lo cual escala mal para horizontes largos (e.g., `steps=48` → ~330K solo en cabezas).
- La complejidad de implementación y mantenimiento de K cabezas con nombres dinámicos es desproporcionada al beneficio marginal en causalidad.

### Arquitectura unificada

```
series_input: (batch, n_lags, n_series)    exog_input: (batch, steps, n_exog)  [opcional]
       │                                          │
       ▼                                          ▼
  RNN layers                               Flatten()
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
                   Dense(dense_units)          ← capas densas regulares (N capas)
                         │
                         ▼
                Dense(n_levels * steps)        ← pesos independientes por (step, level)
                         │                       nombre: "output_dense_layer"
                         ▼
                Reshape(steps, n_levels)
                         │
                         ▼
                output: (batch, steps, n_levels)
```

### Implementación completa: `utils.py`

#### Imports

Eliminar `RepeatVector` y `TimeDistributed` del bloque de imports. Añadir `Flatten`:

```python
from keras.layers import (
    Input,
    LSTM,
    GRU,
    SimpleRNN,
    Flatten,
    Concatenate,
    Dense,
    Reshape,
)
```

#### Función `create_and_compile_model` (refactorizada)

La función pública ya no delega a dos funciones internas. Todo el código vive en una sola función:

```python
def create_and_compile_model(
    series: pd.DataFrame,
    lags: int | list[int] | np.ndarray[int] | range[int],
    steps: int,
    levels: str | list[str] | tuple[str] | None = None,
    exog: pd.Series | pd.DataFrame | None = None,
    recurrent_layer: str = 'LSTM',
    recurrent_units: int | list[int] | tuple[int] = 100,
    recurrent_layers_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
    dense_units: int | list[int] | tuple[int] | None = 64,
    dense_layers_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
    output_dense_layer_kwargs: dict[str, Any] | None = None,
    compile_kwargs: dict[str, Any] | None = None,
    model_name: str | None = None
) -> keras.models.Model:

    # --- Defaults inmutables ---
    if recurrent_layers_kwargs is None:
        recurrent_layers_kwargs = {'activation': 'tanh'}
    if dense_layers_kwargs is None:
        dense_layers_kwargs = {'activation': 'relu'}
    if output_dense_layer_kwargs is None:
        output_dense_layer_kwargs = {'activation': 'linear'}
    if compile_kwargs is None:
        compile_kwargs = {'optimizer': Adam(), 'loss': MeanSquaredError()}

    # --- Print backend info ---
    keras_backend = keras.backend.backend()
    print(f'keras version: {keras.__version__}')
    print(f'Using backend: {keras_backend}')
    # ... (backend version prints) ...

    # --- Validaciones de series ---
    if not isinstance(series, pd.DataFrame):
        raise TypeError(
            f'`series` must be a pandas DataFrame. Got {type(series)}.'
        )
    n_series = series.shape[1]

    # --- Validaciones de exog ---
    if exog is not None:
        if not isinstance(exog, (pd.Series, pd.DataFrame)):
            raise TypeError(
                f'`exog` must be a pandas Series, DataFrame or None. Got {type(exog)}.'
            )
        exog = input_to_frame(data=exog, input_name='exog')
        n_exog = exog.shape[1]

    # --- Validaciones de lags, steps, levels ---
    lags, _, _ = initialize_lags('ForecasterRnn', lags)
    n_lags = len(lags)
    # ... (mismas validaciones de steps, levels que existen actualmente) ...

    # === INPUTS ===
    series_input = Input(shape=(n_lags, n_series), name='series_input')
    inputs = [series_input]

    if exog is not None:
        exog_input = Input(shape=(steps, n_exog), name='exog_input')
        inputs.append(exog_input)

    x = series_input

    # === RECURRENT LAYERS (idéntico al actual) ===
    if not isinstance(recurrent_units, (list, tuple)):
        recurrent_units = [recurrent_units]

    if isinstance(recurrent_layers_kwargs, dict):
        recurrent_layers_kwargs = [recurrent_layers_kwargs] * len(recurrent_units)
    # ... (validaciones de kwargs) ...

    for i, units in enumerate(recurrent_units):
        return_sequences = i < len(recurrent_units) - 1
        layer_kwargs = deepcopy(recurrent_layers_kwargs[i])
        layer_kwargs.update({'units': units, 'return_sequences': return_sequences})
        if 'name' not in layer_kwargs:
            layer_kwargs['name'] = f'{recurrent_layer.lower()}_{i + 1}'

        if recurrent_layer == 'LSTM':
            x = LSTM(**layer_kwargs)(x)
        elif recurrent_layer == 'GRU':
            x = GRU(**layer_kwargs)(x)
        elif recurrent_layer == 'RNN':
            x = SimpleRNN(**layer_kwargs)(x)
        else:
            raise ValueError(...)

    # === EXOG CONCATENATION (NUEVO) ===
    if exog is not None:
        exog_flat = Flatten(name='exog_flatten')(exog_input)
        x = Concatenate(axis=-1, name='concat_exog')([x, exog_flat])

    # === DENSE LAYERS (regulares, NO TimeDistributed) ===
    if dense_units is not None:
        if not isinstance(dense_units, (list, tuple)):
            dense_units = [dense_units]
        # ... (validaciones de dense_layers_kwargs) ...

        for i, units in enumerate(dense_units):
            layer_kwargs = deepcopy(dense_layers_kwargs[i])
            layer_kwargs.update({'units': units})
            if 'name' not in layer_kwargs:
                layer_kwargs['name'] = f'dense_{i + 1}'
            x = Dense(**layer_kwargs)(x)

    # === OUTPUT LAYER (Dense + Reshape) ===
    output_layer_kwargs = deepcopy(output_dense_layer_kwargs)
    output_layer_kwargs.update({'units': n_levels * steps})
    if 'name' not in output_layer_kwargs:
        output_layer_kwargs['name'] = 'output_dense_layer'

    x = Dense(**output_layer_kwargs)(x)
    output = Reshape((steps, n_levels), name='reshape')(x)

    # === COMPILE ===
    model = Model(inputs=inputs, outputs=output, name=model_name)
    model.compile(**compile_kwargs)

    return model
```

### Implementación completa: `_forecaster_rnn.py` — `__init__()`

Cambio en la extracción de `n_levels_out`. La capa se renombra y la división por `max_step` es ahora **incondicional**:

```python
# ANTES:
self.n_levels_out = self.estimator.get_layer('output_dense_td_layer').output.shape[-1]
self.exog_in_ = True if "exog_input" in self.layers_names else False
if self.exog_in_:
    self.n_exog_in = self.estimator.get_layer('exog_input').output.shape[-1]
else:
    self.n_exog_in = None
    # NOTE: This is needed because the Reshape layer changes the output shape
    self.n_levels_out = int(self.n_levels_out / self.max_step)

# DESPUÉS:
self.n_levels_out = self.estimator.get_layer('output_dense_layer').output.shape[-1]
self.exog_in_ = True if 'exog_input' in self.layers_names else False
if self.exog_in_:
    self.n_exog_in = self.estimator.get_layer('exog_input').output.shape[-1]
else:
    self.n_exog_in = None
# Ambas arquitecturas (con y sin exog) usan Dense(n_levels * steps) + Reshape,
# por lo que SIEMPRE es necesario dividir por max_step.
self.n_levels_out = int(self.n_levels_out / self.max_step)
```

### Tests afectados

Inventario completo de tests que requieren actualización:

| Archivo | Cambio necesario | Motivo |
|---------|-----------------|--------|
| `tests/tests_utils/test_create_and_compile_model_exog_tensorflow.py` | Actualizar tests de estructura de capas | Ya no existen `RepeatVector`, `TimeDistributed`. Ahora hay `Flatten`, `Dense`, `Reshape` |
| `tests/tests_utils/test_create_and_compile_model_exog_torch.py` | Ídem | Mismos tests, backend diferente |
| `tests/tests_utils/test_create_and_compile_model_no_exog_tensorflow.py` | Eliminar o redirigir | La función `_create_and_compile_model_no_exog` desaparece. Los tests pasan a la función unificada |
| `tests/tests_utils/test_create_and_compile_model_no_exog_torch.py` | Eliminar o redirigir | Ídem |
| `tests/tests_forecaster_rnn/test_init.py` | Actualizar `layers_names` assertion | `'output_dense_td_layer'` → `'output_dense_layer'` |
| `tests/tests_forecaster_rnn/test_repr.py` | Revisar si el repr incluye nombres de capas | Posible actualización menor |
| `tests/tests_forecaster_rnn/test_fit_tensorflow.py` | Probablemente sin cambios | Las shapes de entrada/salida no cambian |
| `tests/tests_forecaster_rnn/test_fit_pytorch.py` | Probablemente sin cambios | Ídem |
| `tests/tests_forecaster_rnn/test_predict.py` | Probablemente sin cambios | La output shape `(batch, steps, n_levels)` es idéntica |

**Test crítico a actualizar** en `test_init.py`:

```python
# ANTES:
assert forecaster.layers_names == [
    'series_input', 'lstm_1', 'dense_1', 'output_dense_td_layer', 'reshape'
]

# DESPUÉS:
assert forecaster.layers_names == [
    'series_input', 'lstm_1', 'dense_1', 'output_dense_layer', 'reshape'
]
```

**Test a añadir** — modelo con exog debe tener las mismas capas base + `exog_input`, `exog_flatten`, `concat_exog`:

```python
def test_ForecasterRnn_init_with_exog_layer_names():
    series = pd.DataFrame({'A': np.arange(10, dtype=float)})
    exog = pd.DataFrame({'E': np.arange(10, dtype=float)})
    model = create_and_compile_model(
        series=series, lags=3, steps=2, exog=exog
    )
    forecaster = ForecasterRnn(
        estimator=model, levels='A', lags=3
    )
    assert forecaster.layers_names == [
        'series_input', 'exog_input', 'lstm_1',
        'exog_flatten', 'concat_exog',
        'dense_1', 'output_dense_layer', 'reshape'
    ]
    assert forecaster.exog_in_ is True
    assert forecaster.n_exog_in == 1
    assert forecaster.n_levels_out == 1
```

### Defaults mutables — Solución

Todas las firmas de función cambian de defaults mutables a `None`, con resolución interna:

```python
# ANTES (antipatrón):
def create_and_compile_model(
    ...,
    recurrent_layers_kwargs: ... = {"activation": "tanh"},
    dense_layers_kwargs: ... = {"activation": "relu"},
    output_dense_layer_kwargs: ... = {"activation": "linear"},
    compile_kwargs: ... = {"optimizer": Adam(), "loss": MeanSquaredError()},
):
    ...

# DESPUÉS (correcto):
def create_and_compile_model(
    ...,
    recurrent_layers_kwargs: ... = None,
    dense_layers_kwargs: ... = None,
    output_dense_layer_kwargs: ... = None,
    compile_kwargs: ... = None,
):
    if recurrent_layers_kwargs is None:
        recurrent_layers_kwargs = {'activation': 'tanh'}
    if dense_layers_kwargs is None:
        dense_layers_kwargs = {'activation': 'relu'}
    if output_dense_layer_kwargs is None:
        output_dense_layer_kwargs = {'activation': 'linear'}
    if compile_kwargs is None:
        compile_kwargs = {'optimizer': Adam(), 'loss': MeanSquaredError()}
    ...
```

**Nota sobre backward compatibility:** esto **no es un breaking change** para los usuarios. Cualquier llamada que antes pasaba `recurrent_layers_kwargs={"activation": "tanh"}` explícitamente funciona igual. Cualquier llamada que omitía el argumento obtiene el mismo default.

### Backward compatibility: modelos serializados

Los modelos Keras guardados con `model.save()` bajo la arquitectura actual (RepeatVector + TimeDistributed) **no serán compatibles directamente** con la nueva versión del forecaster. Esto se debe a que:

1. Los nombres de capas cambian (`output_dense_td_layer` → `output_dense_layer`)
2. La estructura interna del modelo es diferente

**Mitigación:**
- Documentar como **breaking change** en el changelog de la versión
- Los pesos del RNN se pueden reutilizar reconstruyendo el modelo con la nueva arquitectura y copiando los pesos de las capas recurrentes (que no cambian)
- Los usuarios deberán re-entrenar la parte Dense del modelo

### Resumen de líneas de código

| Métrica | Antes | Después | Cambio |
|---------|-------|---------|--------|
| `utils.py` — funciones | 3 (`create_and_compile_model` + 2 internas) | 1 (unificada) | -2 funciones |
| `utils.py` — líneas aprox. | ~700 | ~350 | -50% |
| `_forecaster_rnn.py` — cambios | — | 3 líneas en `__init__` | Trivial |
| Imports eliminados | — | `RepeatVector`, `TimeDistributed` | Limpieza |
| Imports añadidos | — | `Flatten` | +1 |
| Tests a actualizar | — | ~6-8 tests | Moderado |
| Tests a añadir | — | ~2-3 tests (exog layer names, consistencia sin/con exog) | Bajo |

### Checklist de implementación

- [ ] Reescribir `create_and_compile_model` como función unificada en `utils.py`
- [ ] Eliminar `_create_and_compile_model_exog` y `_create_and_compile_model_no_exog`
- [ ] Actualizar imports en `utils.py` (quitar `RepeatVector`, `TimeDistributed`; añadir `Flatten`)
- [ ] Corregir defaults mutables en la firma de `create_and_compile_model`
- [ ] Actualizar `__init__` en `_forecaster_rnn.py`: nombre de capa + división incondicional
- [ ] Actualizar `test_init.py`: `layers_names` assertion
- [ ] Unificar tests de utils en ficheros únicos (o actualizar los 4 existentes)
- [ ] Añadir test de consistencia: modelo sin exog y con exog producen la misma estructura base
- [ ] Verificar que `predict()`, `fit()`, `backtesting` pasan sin cambios  
- [ ] Documentar breaking change en changelog
- [ ] Actualizar docstring de `create_and_compile_model` (eliminar referencias a TimeDistributed)
