# Evaluación de `_bayesian_search_optuna` — Análisis de necesidad real

> Fecha: 2026-03-01  
> Aplica a: `_bayesian_search_optuna` y `_bayesian_search_optuna_multiseries`  
> Archivo: `skforecast/model_selection/_search.py`

---

## Resumen ejecutivo

De los ~15 puntos identificados en la evaluación inicial, **6 son realmente necesarios** (bugs reales o mejoras de alto impacto), **3 son recomendables** (mejoran la calidad del código con poco esfuerzo), y **el resto se descartan** por ser innecesarios, ya accesibles vía la API existente, o de impacto insuficiente para justificar el cambio.

---

## NECESARIOS — Deben implementarse

### 1. Mutable default arguments (`dict = {}`)

**Tipo:** Bug  
**Riesgo real:** Medio  
**Esfuerzo:** Trivial

```python
# ACTUAL (problemático)
def _bayesian_search_optuna(..., kwargs_create_study: dict = {}, kwargs_study_optimize: dict = {}):

# CORRECTO
def _bayesian_search_optuna(..., kwargs_create_study: dict | None = None, kwargs_study_optimize: dict | None = None):
    kwargs_create_study = kwargs_create_study or {}  # copia implícita
    kwargs_study_optimize = kwargs_study_optimize or {}
```

**Por qué es necesario:** La función muta ambos dicts durante la ejecución:
- Añade `'direction'` a `kwargs_create_study` (línea ~882)
- Añade `'show_progress_bar'` a `kwargs_study_optimize` (línea ~885)

Si un usuario llama a la función dos veces en el mismo proceso (ej. primero con un `ForecasterRecursive` y luego con un `ForecasterRecursiveClassifier`), la `direction='minimize'` de la primera llamada persiste en la segunda, donde debería ser `'maximize'`. El bug es silencioso: no lanza error, simplemente **produce resultados incorrectos**.

**Afecta a (5 funciones):**
- `bayesian_search_forecaster` (wrapper público)
- `_bayesian_search_optuna`
- `bayesian_search_forecaster_multiseries` (wrapper público)
- `_bayesian_search_optuna_multiseries`

---

### 2. `try/finally` para restaurar warnings y cerrar FileHandler

**Tipo:** Bug (resource leak + estado corrupto)  
**Riesgo real:** Medio  
**Esfuerzo:** Bajo

Actualmente, si `study.optimize()` u otro código intermedio lanza una excepción:

1. `set_skforecast_warnings(suppress_warnings, action='default')` nunca se ejecuta → las warnings quedan permanentemente suprimidas para el resto del proceso.
2. `handler.close()` nunca se ejecuta → el `FileHandler` queda abierto Y adjunto al logger de optuna, corrompiendo el logging de cualquier operación posterior.

**Solución:**
```python
set_skforecast_warnings(suppress_warnings, action='ignore')
try:
    # ... todo el cuerpo de la función ...
finally:
    if output_file is not None:
        handler.close()
        logger.removeHandler(handler)
    set_skforecast_warnings(suppress_warnings, action='default')
```

**Afecta a:** `_bayesian_search_optuna` y `_bayesian_search_optuna_multiseries`

---

### 3. Filtrar trials por `TrialState.COMPLETE` en `study.get_trials()`

**Tipo:** Bug potencial  
**Riesgo real:** Bajo-Medio  
**Esfuerzo:** Trivial

```python
# ACTUAL
for i, trial in enumerate(study.get_trials()):
    for m, m_values in zip(metric, metric_values[i]):  # IndexError si hay trials fallidos

# CORRECTO
from optuna.trial import TrialState
for i, trial in enumerate(study.get_trials(states=[TrialState.COMPLETE])):
```

**Cuándo ocurre el bug:** Cuando el usuario pasa `kwargs_study_optimize={'catch': (Exception,)}` para que optuna capture excepciones y continúe con otros trials. En ese caso, un trial con `FAIL` estado aparece en `get_trials()` pero no tiene entrada en `metric_values`, causando un desalineamiento de índices → `IndexError` o, peor, métricas asignadas al trial incorrecto.

Con el comportamiento por defecto de optuna >= 3.0 (`catch=()`), las excepciones propagan y nunca se llega al código de extracción. Pero con optuna 2.x o con `catch` explícito, el bug es real.

**Nota:** Este bug se eliminaría automáticamente si se adopta `trial.set_user_attr()` (punto 5), pero debe corregirse igualmente como defensa en profundidad.

---

### 4. Cachear `_train_test_split_one_step_ahead` en el path `OneStepAheadFold`

**Tipo:** Optimización de rendimiento  
**Impacto:** Alto (>50% speedup cuando lags no están en el search_space)  
**Esfuerzo:** Medio

En el path `OneStepAheadFold`, **cada trial** ejecuta:
```python
X_train, y_train, X_test, y_test = forecaster_search._train_test_split_one_step_ahead(
    y=y, initial_train_size=cv.initial_train_size, exog=exog
)
```

`_train_test_split_one_step_ahead` internamente llama a `_create_train_X_y` **dos veces** (una para train, otra para test), que incluye: creación de lag features, rolling features, transformaciones, etc. Esta es la operación más costosa de cada trial.

**Pero:** cuando el `search_space` no incluye `'lags'`, las matrices resultantes son **idénticas en cada trial** — solo cambian los hiperparámetros del estimador (que no afectan al split).

**Solución propuesta:**
```python
# Fuera del _objective, antes de study.optimize:
_cached_split = {}

def _objective(trial, ...):
    sample = search_space(trial)
    current_lags = sample.get('lags', _SENTINEL)

    # Convertir a tuple para que sea hashable
    lags_key = _make_hashable(current_lags)

    if lags_key not in _cached_split:
        if current_lags is not _SENTINEL:
            forecaster_search.set_lags(current_lags)
        _cached_split[lags_key] = forecaster_search._train_test_split_one_step_ahead(
            y=y, initial_train_size=cv.initial_train_size, exog=exog
        )

    X_train, y_train, X_test, y_test = _cached_split[lags_key]
    ...
```

**Impacto real medido en funciones similares de grid_search:** La función `_evaluate_grid_hyperparameters` ya implementa un patrón similar donde el split solo se recalcula cuando cambian los lags. La bayesian search debería hacer lo mismo.

**Afecta a:** `_bayesian_search_optuna` (path OneStepAheadFold) y `_bayesian_search_optuna_multiseries` (path OneStepAheadFold)

---

### 5. Usar `trial.set_user_attr()` en vez del hack con `nonlocal metric_values`

**Tipo:** Mejora de diseño + robustez  
**Impacto:** Medio  
**Esfuerzo:** Bajo

Actualmente, las métricas adicionales (cuando el usuario pasa múltiples métricas) se almacenan en una lista externa mediante `nonlocal`:
```python
metric_values = []  # definida fuera

def _objective(trial, ...):
    ...
    nonlocal metric_values
    metric_values.append(metrics)
    return metrics[0]
```

Este patrón es frágil:
- Depende de que `metric_values` y `study.get_trials()` estén sincronizados por índice.
- Se rompe con trials fallidos (punto 3).
- Es un antipatrón reconocido en la documentación de optuna.

**Solución con `trial.set_user_attr()`:**
```python
def _objective(trial, ...):
    ...
    # Almacenar todas las métricas directamente en el trial
    for m_name, m_val in zip(metric_names, metrics):
        trial.set_user_attr(m_name, m_val)
    return metrics[0]

# Al extraer resultados:
for trial in study.get_trials(states=[TrialState.COMPLETE]):
    for m_name in metric_names:
        metric_dict[m_name].append(trial.user_attrs[m_name])
```

**Ventajas:**
- Elimina la variable `nonlocal` y su fragilidad.
- Inmune a desalineamientos por trials fallidos/podados.
- Las métricas adicionales son accesibles desde `study.trials_dataframe()` automáticamente.
- Es el mecanismo oficial de optuna para almacenar datos extra por trial.
- Disponible desde optuna 2.0 (compatible con `optuna>=2.10`).

**Afecta a:** `_bayesian_search_optuna` y `_bayesian_search_optuna_multiseries`

---

### 6. Falta `suppress_warnings` en `backtesting_forecaster` (path TimeSeriesFold)

**Tipo:** Bug de inconsistencia  
**Riesgo real:** Bajo (spam de warnings)  
**Esfuerzo:** Trivial

En el `_objective` del path `TimeSeriesFold` de `_bayesian_search_optuna`:
```python
metrics, _ = backtesting_forecaster(
    forecaster    = forecaster_search,
    y             = y,
    cv            = cv,
    exog          = exog,
    metric        = metric,
    n_jobs        = n_jobs,
    verbose       = verbose,
    show_progress = False
    # ← FALTA: suppress_warnings = suppress_warnings
)
```

Pero en `_bayesian_search_optuna_multiseries`, la versión multiseries **sí** lo pasa:
```python
metrics, _ = backtesting_forecaster_multiseries(
    ...
    suppress_warnings = suppress_warnings  # ← presente
)
```

**Resultado:** Con `suppress_warnings=True`, la versión single-series imprime warnings de skforecast en cada trial × cada fold. Con 50 trials y 10 folds = 500 iteraciones de warnings.

---

## RECOMENDABLES — Mejoran la calidad, poco esfuerzo

### 7. Añadir `else: kwargs_study_optimize['show_progress_bar'] = False`

**Tipo:** Inconsistencia menor  
**Esfuerzo:** Trivial

```python
# ACTUAL
if show_progress:
    kwargs_study_optimize['show_progress_bar'] = True
# Si show_progress=False, no se establece nada.

# MEJOR
if show_progress:
    kwargs_study_optimize['show_progress_bar'] = True
else:
    kwargs_study_optimize.setdefault('show_progress_bar', False)
```

Si el usuario pasa `show_progress=False` pero en una llamada anterior (con mutable default, punto 1) se estableció `show_progress_bar=True`, la barra seguiría mostrándose. Al corregir el punto 1 (usar `None`), este problema desaparece. Aun así, ser explícito es mejor.

---

### 8. Normalizar el `print()` de `return_best` a `warnings.warn()`

**Tipo:** Mejora de API  
**Esfuerzo:** Trivial

```python
# ACTUAL
print(f"`Forecaster` refitted using the best-found lags and parameters, ...")

# MEJOR
warnings.warn(
    f"`Forecaster` refitted using the best-found lags and parameters, ...",
    stacklevel=2
)
```

`print()` no es silenciable y contamina stdout. Sin embargo, cambiar a `warnings.warn()` podría romper scripts que parsean la salida o tests que verifican el print. **Evaluar si es un breaking change en el contexto del proyecto.**

**Nota:** Este patrón de `print()` de `return_best` es consistente con `_evaluate_grid_hyperparameters` y las versiones multiseries. Si se cambia, debe hacerse en todas las funciones de búsqueda.

---

### 9. Inconsistencia del parámetro `output_file`

**Tipo:** Inconsistencia de API (solo documentar, no cambiar)

El parámetro `output_file` tiene comportamientos **completamente distintos**:
- En `_evaluate_grid_hyperparameters`: guarda resultados tabulares en TSV (incrementalmente por cada combinación evaluada).
- En `_bayesian_search_optuna`: redirige el logger de optuna a un archivo.

El docstring de `_bayesian_search_optuna` dice *"The results will be saved in a tab-separated values (TSV) format"*, lo cual es **incorrecto** — lo que se guarda son los logs de optuna, no un TSV de resultados.

**Decisión: NO cambiar** (sería un breaking change de la API pública). **Sí corregir el docstring** para reflejar el comportamiento real.

---

## DESCARTADOS — No necesarios

### 10. Pruning de trials (Early Stopping)

**Justificación del descarte:**  
Requiere una refactorización profunda de `backtesting_forecaster` para que pueda reportar métricas intermedias por fold. Actualmente, `backtesting_forecaster` es una operación atómica que devuelve un resultado agregado. Implementar pruning requeriría:
- Añadir un callback o yield por fold dentro de `backtesting_forecaster`.
- O reimplementar el loop de backtesting directamente dentro del `_objective`.
- O cambiar `backtesting_forecaster` para que acepte un trial de optuna.

Cualquiera de estas opciones es un cambio arquitectónico significativo con riesgo alto. El beneficio (abortar trials malos antes de completar todos los folds) es real pero el coste de implementación no lo justifica ahora. **Considerar para una versión futura como feature independiente.**

---

### 11. `study.enqueue_trial()` / parámetro `initial_trials`

**Justificación del descarte:**  
Ya es posible hacerlo con la API actual:
```python
study = optuna.create_study()
study.enqueue_trial({'n_estimators': 100, 'max_depth': 5})
# Pasar study via kwargs_create_study... PERO: la función crea su propio study internamente.
```

Hmm, en realidad NO es posible porque `_bayesian_search_optuna` siempre crea un study nuevo. Para soportar `enqueue_trial`, se necesitaría:
- Aceptar un parámetro `study` externo, O
- Añadir un parámetro `initial_trials: list[dict]`.

Es una feature útil pero **baja prioridad**: el TPE sampler ya incluye fases de exploración aleatoria al inicio. El beneficio marginal es pequeño.

---

### 12. Multi-objective optimization nativo

**Justificación del descarte:**  
- Cambio de API significativo (el return cambiaría a un Pareto front).
- El caso de uso principal (optimizar una métrica) ya está cubierto.
- Las métricas adicionales ya se almacenan y devuelven en el DataFrame de resultados.
- El usuario que necesite multi-objetivo puede pasar `kwargs_create_study={'directions': ['minimize', 'minimize']}` y manejar el Pareto front manualmente.

**No es necesario** como feature integrada.

---

### 13. Persistent Storage / Resume

**Justificación del descarte:**  
Ya es accesible vía `kwargs_create_study`:
```python
bayesian_search_forecaster(
    ...,
    kwargs_create_study={
        'storage': 'sqlite:///my_study.db',
        'study_name': 'my_search',
        'load_if_exists': True
    }
)
```

No se necesita un parámetro explícito. La API existente ya lo permite.

---

### 14. `timeout` como parámetro explícito

**Justificación del descarte:**  
Ya es accesible vía `kwargs_study_optimize={'timeout': 300}`. Añadir un parámetro dedicado incrementa la superficie de la API sin aportar funcionalidad nueva. La documentación podría simplemente mostrar un ejemplo.

---

### 15. `pd.json_normalize` en vez de `apply(pd.Series)`

**Justificación del descarte:**  
El DataFrame de resultados tiene como máximo `n_trials` filas (típicamente 10-200). La diferencia de rendimiento entre `apply(pd.Series)` y `pd.json_normalize` en DataFrames pequeños es de microsegundos. El tiempo de la función está dominado por el backtesting/entrenamiento (minutos-horas), no por el post-procesamiento.

---

### 16. `deepcopy(forecaster)` costoso

**Justificación del descarte:**  
El `deepcopy` es **necesario y correcto**. La función muta el forecaster con `set_params()` y `set_lags()` en cada trial. Sin deepcopy, estas mutaciones afectarían al forecaster original del usuario antes del refit final.

El coste es one-time (una sola copia antes del loop de optimización), no per-trial. Incluso si el forecaster tiene datos pesados cacheados, el coste es despreciable comparado con los N trials de búsqueda.

Un shallow copy sería incorrecto porque `set_params()` muta el estimador interno.

---

### 17. Constraints de optuna

**Justificación del descarte:**  
Feature demasiado especializada. El usuario que la necesite puede implementarla via `kwargs_create_study` y lógica custom en el `search_space`. No justifica complejidad adicional en la API.

---

## Resumen final

| #  | Issue | Tipo | Necesario | Esfuerzo |
|----|-------|------|-----------|----------|
| 1  | Mutable default args | Bug | **SÍ** | Trivial |
| 2  | try/finally warnings + handler | Bug | **SÍ** | Bajo |
| 3  | Filtrar trials COMPLETE | Bug | **SÍ** | Trivial |
| 4  | Cachear split OneStepAhead | Perf | **SÍ** | Medio |
| 5  | trial.set_user_attr() | Diseño | **SÍ** | Bajo |
| 6  | suppress_warnings en backtesting | Bug | **SÍ** | Trivial |
| 7  | show_progress_bar explícito | Consistencia | Recomendable | Trivial |
| 8  | print → warnings.warn | API | Recomendable | Trivial |
| 9  | Docstring output_file | Doc | Recomendable | Trivial |
| 10 | Pruning | Feature | No | Alto |
| 11 | enqueue_trial | Feature | No | Medio |
| 12 | Multi-objective | Feature | No | Alto |
| 13 | Persistent storage | Feature | No (ya existe via kwargs) | — |
| 14 | timeout explícito | API | No (ya existe via kwargs) | — |
| 15 | json_normalize | Perf | No (impacto despreciable) | — |
| 16 | Evitar deepcopy | Perf | No (es necesario y correcto) | — |
| 17 | Constraints | Feature | No (demasiado especializado) | — |

### Orden de implementación sugerido

1. **Puntos 1 + 6 + 7** (bugs triviales, un solo commit)
2. **Punto 2** (try/finally, un commit)
3. **Puntos 3 + 5** (trial.set_user_attr + filtro COMPLETE, van juntos, un commit)
4. **Punto 4** (cacheo de split, requiere más cuidado y tests)
5. **Punto 9** (corregir docstring, un commit)
6. **Punto 8** (print → warn, evaluar impacto en tests primero)

---

## Análisis de Samplers de Optuna

> Fuente: https://optuna.readthedocs.io/en/stable/reference/samplers/

### ¿Por qué TPESampler es el default adecuado?

TPESampler (Tree-structured Parzen Estimator) es la elección correcta como sampler por defecto para skforecast por las siguientes razones:

1. **Es el default de optuna.** El equipo de optuna lo selecciona como sampler por defecto tras extensivo benchmarking. Es el sampler más probado y estable del ecosistema.

2. **Soporta espacios de búsqueda mixtos.** En skforecast, el `search_space` típicamente mezcla `suggest_categorical` (lags), `suggest_int` (n_estimators, max_depth) y `suggest_float` (learning_rate). TPE maneja tipos mixtos nativamente. Alternativas como `GPSampler` y `CmaEsSampler` no soportan bien parámetros categóricos.

3. **Escala bien con el número de trials.** Complejidad $O(dn\log n)$ por trial (donde $d$ = dimensiones, $n$ = trials previos). `GPSampler` es $O(n^3)$ y se recomienda solo para ≤500 trials.

4. **Sin dependencias adicionales.** Viene incluido con optuna. `GPSampler` requiere `torch` + `scipy`. `AutoSampler` requiere `optunahub` + `cmaes` + `torch` + `scipy`. Ninguna de estas dependencias es aceptable para el core de skforecast.

5. **Funciona bien con cualquier número de trials.** Desde 10 (caso común en skforecast con `n_trials=10` por defecto) hasta miles. `GPSampler` es teóricamente mejor con pocos trials (<100) pero colapsa computacionalmente con muchos. TPE es robusto en todo el rango.

**Conclusión:** No hay razón para cambiar el tipo de sampler. La mejora está en **configurar mejor** el TPESampler existente (ver secciones siguientes).

---

### Estado actual

La función usa `TPESampler(seed=random_state)` con **todos los parámetros por defecto**:

```python
study.sampler = TPESampler(seed=random_state)
```

Esto significa:
- `multivariate=False` → cada parámetro se optimiza **independientemente** (no modela correlaciones entre hiperparámetros)
- `group=False` → no se descompone el search space para conditional params
- `n_startup_trials=10` → los 10 primeros trials son aleatorios
- `constant_liar=False` → no se penalizan trials running (solo relevante en paralelo)
- `consider_endpoints=False` → no considera extremos del dominio

### Mejoras evaluadas

#### A. `multivariate=True` — **RECOMENDABLE**

**Qué hace:** En lugar de ajustar un GMM univariante independiente por cada hiperparámetro, ajusta un GMM **multivariante** que modela las correlaciones entre todos los parámetros.

**Por qué importa para skforecast:** En un search_space típico de ML los hiperparámetros están correlacionados. Ejemplo con LGBMRegressor:
- `n_estimators` y `learning_rate` tienen una relación inversa fuerte
- `max_depth` y `num_leaves` están acoplados
- `colsample_bytree` y `reg_lambda` interactúan

Con `multivariate=False`, el TPE trata estos parámetros como si fueran independientes, perdiendo información valiosa sobre las interacciones.

**Coste:** Insignificante. Aumenta ligeramente el tiempo del sampler (no del backtesting que es órdenes de magnitud más costoso).

**Disponibilidad:** Desde optuna 2.2.0 (compatible con `optuna>=2.10` del proyecto).

**Paper referencia:** [BOHB: Robust and Efficient Hyperparameter Optimization at Scale](http://proceedings.mlr.press/v80/falkner18a.html)

**Cambio propuesto:**
```python
# ACTUAL
study.sampler = TPESampler(seed=random_state)

# PROPUESTO
study.sampler = TPESampler(seed=random_state, multivariate=True)
```

**Riesgo:** Esta flag es experimental en optuna. Sin embargo, lleva estable desde v2.2.0 (2020), se recomienda oficialmente en el blog de optuna, y es el default de AutoSampler. El riesgo real es mínimo.

---

#### B. `group=True` (junto con `multivariate=True`) — **RECOMENDABLE**

**Qué hace:** Cuando el search_space tiene parámetros **condicionales** (parámetros que solo existen en algunos trials), el TPE multivariante estándar no puede modelarlos bien porque necesita que todos los trials tengan los mismos parámetros.

`group=True` descompone automáticamente el search space en subespacios independientes basándose en los trials pasados, y aplica TPE multivariante dentro de cada subespacio.

**Por qué importa:** En skforecast, el `search_space` puede incluir `'lags'` como `suggest_categorical`, lo que crea un espacio condicional de facto: distintas configuraciones de lags producen search spaces efectivamente diferentes. Además, los usuarios pueden definir search spaces condicionales:
```python
def search_space(trial):
    estimator_type = trial.suggest_categorical('estimator', ['lightgbm', 'random_forest'])
    if estimator_type == 'lightgbm':
        return {'learning_rate': trial.suggest_float('lr', 0.01, 0.3)}
    else:
        return {'max_features': trial.suggest_float('max_features', 0.5, 1.0)}
```

**Disponibilidad:** Desde optuna 2.8.0 (compatible con `optuna>=2.10`).

**Cambio propuesto:**
```python
study.sampler = TPESampler(seed=random_state, multivariate=True, group=True)
```

---

#### C. `consider_endpoints=True` — **RECOMENDABLE**

**Qué hace:** Incluye los extremos del dominio al calcular las varianzas de los estimadores de Parzen. Sin esto, el TPE tiene bias contra los valores en los bordes del search space.

**Por qué importa:** En hiperparámetros de ML es común que el óptimo esté cerca de un borde (ej. `learning_rate` cerca de 0.01, `n_estimators` cerca de su máximo, `max_depth` bajo como 3).

**Coste:** Cero.

**Cambio propuesto:**
```python
study.sampler = TPESampler(
    seed=random_state,
    multivariate=True,
    group=True,
    consider_endpoints=True
)
```

---

#### D. `constant_liar=True` — **NO NECESARIO**

**Qué hace:** Penaliza trials en ejecución para evitar sugerir parámetros similares durante ejecución paralela.

**Por qué NO aplica:** `study.optimize()` en skforecast se ejecuta secuencialmente (no se pasa `n_jobs` al optimize, sino al backtesting interno). No hay trials concurrentes a nivel de optuna.

---

#### E. `GPSampler` como alternativa — **NO RECOMENDABLE como default**

**Qué es:** Sampler basado en Gaussian Process (Bayesian Optimization clásica con logEI).

**Ventajas:** Mejor modelado del landscape de la función objetivo. Especialmente eficiente con pocos trials (<100).

**Por qué NO como default:**
- Requiere `scipy` + `torch` como dependencias, lo que sería un peso enorme para skforecast.
- Complejidad $O(n^3)$ por trial vs $O(dn\log n)$ del TPE.
- Presupuesto recomendado: ≤500 trials. El TPE es más robusto en general.
- No maneja bien parámetros categóricos (que son muy comunes en los search spaces de skforecast, ej. lags).

**Nota:** El usuario ya puede usarlo vía `kwargs_create_study={'sampler': GPSampler(seed=123)}`.

---

#### F. `AutoSampler` (OptunaHub) — **NO RECOMENDABLE**

**Qué es:** Selecciona automáticamente entre TPE, GP y CMA-ES según el search space.

**Por qué NO:**
- Requiere dependencia externa: `optunahub`, `cmaes`, `torch`, `scipy`.
- Es un paquete externo a optuna (de OptunaHub), no parte del core.
- La lógica de selección automática está fuera de nuestro control.
- El beneficio marginal sobre TPE multivariante bien configurado es pequeño.

---

#### G. `Terminator` (early stopping del estudio) — **INTERESANTE PERO NO PRIORITARIO**

**Qué es:** Detiene automáticamente `study.optimize()` cuando estima que el potencial de mejora restante es menor que el error estadístico de la función objetivo.

**Uso:**
```python
from optuna.terminator import Terminator, TerminatorCallback

terminator = Terminator()
callback = TerminatorCallback(terminator)
study.optimize(objective, n_trials=1000, callbacks=[callback])
# Se detendrá antes de 1000 si no hay mejora significativa
```

**Por qué es interesante:** Permitiría a los usuarios poner un `n_trials` alto sin preocuparse del desperdicio de compute. Pero ya pueden hacerlo vía `kwargs_study_optimize={'callbacks': [TerminatorCallback(...)]}`.

**Decisión:** No integrar como feature nativa. Es accesible vía la API existente.

---

### Resumen de mejoras al Sampler

| Mejora | Recomendada | Esfuerzo | Impacto |
|--------|-------------|----------|---------|
| `multivariate=True` | **SÍ** | Trivial (1 línea) | Medio-Alto: mejor exploración del espacio |
| `group=True` | **SÍ** | Trivial (1 línea) | Medio: mejor manejo de search spaces condicionales |
| `consider_endpoints=True` | **SÍ** | Trivial (1 línea) | Bajo-Medio: menos bias en bordes |
| `constant_liar=True` | No | — | No aplica (ejecución secuencial) |
| `GPSampler` como default | No | — | Requiere torch, no justificado |
| `AutoSampler` | No | — | Dependencia externa innecesaria |
| `Terminator` integrado | No | — | Ya accesible vía kwargs |

### Cambio propuesto final (1 línea)

Aplicar en `_bayesian_search_optuna` y `_bayesian_search_optuna_multiseries`:

```python
# ANTES
if 'sampler' not in kwargs_create_study.keys():
    study.sampler = TPESampler(seed=random_state)

# DESPUÉS
if 'sampler' not in kwargs_create_study.keys():
    study.sampler = TPESampler(
        seed               = random_state,
        multivariate       = True,
        group              = True,
        consider_endpoints = True,
    )
```

**Nota sobre compatibilidad:** Todos estos parámetros existen desde optuna 2.8.0+, dentro del rango de `optuna>=2.10` que requiere skforecast. Los defaults anteriores (`False, False, False`) producían búsquedas subóptimas. El cambio mejora la calidad de la búsqueda sin romper la API ni tests existentes (los tests verifican estructura de resultados, no valores específicos de métricas).

---
---

# Rediseño completo de `_bayesian_search_optuna`

> Ejercicio de diseño con total libertad: qué cambiaría si pudiera reescribir la función desde cero, sin restricciones de backward compatibility.

---

## Parte 1 — Argumentos / Firma de la función

### 1.1 Eliminar el wrapper público que no aporta

**Problema actual:**  
`bayesian_search_forecaster` es un wrapper de 30 líneas que solo valida `len(exog) != len(y)` y delega todo a `_bayesian_search_optuna`. Para la versión multiseries, incluso esa validación no existe — el wrapper solo reordena argumentos.

```python
# ACTUAL: 2 funciones que hacen lo mismo
def bayesian_search_forecaster(...)       → llama a _bayesian_search_optuna(...)
def _bayesian_search_optuna(...)          → toda la lógica real
```

**Rediseño:** Una sola función pública. La validación de exog se mueve al inicio de la implementación.

```python
# PROPUESTO: 1 función
def bayesian_search_forecaster(...):
    # Validación + implementación en el mismo sitio
```

**Impacto:** Elimina 2 funciones (wrapper single + wrapper multi), ~80 líneas de código duplicado, y evita confusión sobre qué función usar. La API pública no cambia.

---

### 1.2 Mutable default arguments → `None`

**Problema actual (ya identificado):**
```python
def _bayesian_search_optuna(..., kwargs_create_study: dict = {}, kwargs_study_optimize: dict = {}):
```

**Rediseño:**
```python
def bayesian_search_forecaster(
    ...,
    kwargs_create_study: dict | None = None,
    kwargs_study_optimize: dict | None = None,
):
    kwargs_create_study = kwargs_create_study.copy() if kwargs_create_study is not None else {}
    kwargs_study_optimize = kwargs_study_optimize.copy() if kwargs_study_optimize is not None else {}
```

**Nota:** Usar `.copy()` en vez de `or {}` para que las mutaciones internas (añadir `direction`, `show_progress_bar`) no modifiquen el dict original del usuario.

---

### 1.3 `random_state` es redundante con `kwargs_create_study`

**Problema actual:**  
El parámetro `random_state` solo se usa para crear `TPESampler(seed=random_state)` cuando el usuario no proporciona sampler. Si el usuario pasa un sampler custom vía `kwargs_create_study`, `random_state` se ignora silenciosamente.

```python
# Esto es confuso:
bayesian_search_forecaster(
    random_state=42,
    kwargs_create_study={'sampler': TPESampler(seed=99)}  # random_state=42 se ignora
)
```

**Rediseño opción A — Mantener `random_state` pero documentar claramente:**
```python
random_state : int, default 123
    Seed for the default TPESampler. **Ignored** if a custom sampler is 
    provided via `kwargs_create_study['sampler']`.
```

**Rediseño opción B — Eliminar `random_state`, usar solo `kwargs_create_study`:**
```python
# El usuario controla el sampler explícitamente
kwargs_create_study={'sampler': TPESampler(seed=42, multivariate=True)}
```

**Decisión:** Opción A es más pragmática. `random_state` es un patrón familiar de sklearn que reduce la barrera de entrada. Pero el docstring actual es engañoso.

---

### 1.4 `n_jobs` no pertenece a esta función

**Problema actual:**  
`n_jobs` solo se usa en el path `TimeSeriesFold` como parámetro de `backtesting_forecaster`. En el path `OneStepAheadFold`, se ignora completamente. No controla paralelismo de optuna (que sería `n_jobs` en `study.optimize()`).

```python
# Path TimeSeriesFold: n_jobs se pasa a backtesting_forecaster
# Path OneStepAheadFold: n_jobs no se usa NUNCA
```

**Rediseño:**  
`n_jobs` debería ser un parámetro del `cv` object (ya que el nivel de paralelismo del backtesting es una configuración de la evaluación, no de la búsqueda). O al menos documentar claramente que solo aplica a `TimeSeriesFold`.

Alternativa pragmática: mantenerlo pero añadir al docstring:
```
n_jobs : int, 'auto', default 'auto'
    Only used when `cv` is a `TimeSeriesFold`. Ignored for `OneStepAheadFold`.
```

---

### 1.5 `output_file` tiene comportamiento inconsistente

**Problema actual:**  
En `_evaluate_grid_hyperparameters`, `output_file` guarda resultados incrementalmente en TSV:
```python
# Grid search: guarda resultados
with open(output_file, 'a') as f:
    f.write('\t'.join([str(r) for r in row]) + '\n')
```

En `_bayesian_search_optuna`, `output_file` redirige los logs de optuna:
```python
# Bayesian search: redirige logger
handler = logging.FileHandler(output_file, mode="w")
logger.addHandler(handler)
```

El docstring de ambas funciones dice exactamente lo mismo: *"The results will be saved in a tab-separated values (TSV) format"* — lo cual es **falso** para la versión bayesiana.

**Rediseño:**  
Hacer que `output_file` tenga el **mismo comportamiento** en todas las funciones de búsqueda: guardar resultados tabulares incrementalmente. Si el usuario quiere logs de optuna, puede configurar el logger él mismo o usar `optuna.logging.set_file_handler()`.

```python
# Dentro del _objective, al final de cada trial:
if output_file is not None:
    _append_trial_to_file(output_file, trial, lags, params, metrics)
```

**Alternativa modesta:** Renombrar a `log_file` en bayesian y mantener `output_file` en grid, con docstrings correctos.

---

### 1.6 Nombres de `kwargs_*` son verbosos

**Problema menor.** Los nombres actuales son largos pero descriptivos:
- `kwargs_create_study` (20 chars)
- `kwargs_study_optimize` (22 chars)

**Alternativa:**
- `study_kwargs` → para `optuna.create_study()`
- `optimize_kwargs` → para `study.optimize()`

O usar un solo diccionario:
```python
optuna_kwargs: dict | None = None
# {'create_study': {...}, 'optimize': {...}}
```

**Decisión:** No es prioritario. Los nombres actuales son claros aunque verbosos.

---

### 1.7 No hay forma de devolver el `study` object

**Problema actual:**  
La función crea un `optuna.Study` internamente, lo optimiza, extrae resultados a un DataFrame, y **descarta el study**. Solo devuelve el `best_trial`.

El usuario pierde acceso a:
- `study.trials_dataframe()` — vista completa con metadatos de optuna
- `study.best_params`, `study.best_value`
- Visualizaciones: `optuna.visualization.plot_optimization_history(study)`
- Resumir la optimización con `study.optimize()` adicional
- Trials con estado FAIL/PRUNED (para diagnóstico)

**Rediseño — Devolver `study` en lugar de `best_trial`:**
```python
# ACTUAL
def bayesian_search_forecaster(...) -> tuple[pd.DataFrame, FrozenTrial]:
    return results, best_trial

# PROPUESTO
def bayesian_search_forecaster(...) -> tuple[pd.DataFrame, optuna.Study]:
    return results, study
    # El usuario puede acceder a study.best_trial si lo necesita
```

**Ventajas:**
- El `study` contiene toda la información del `best_trial` y mucho más.
- Permite `optuna.visualization.plot_optimization_history(study)`.
- Permite resumir: `study.optimize(objective, n_trials=50)` (si se expone el objective).
- No requiere cambios en la extracción de resultados.

**Riesgo:** Breaking change de la API pública.

---

### 1.8 Simplificación de argumentos — De 16 a 13 parámetros

**Estado actual — 16 parámetros agrupados por categoría:**

```
ESENCIALES (6) — no se pueden tocar:
  forecaster, y, cv, search_space, metric, exog

OPTUNA (4):
  n_trials, random_state, kwargs_create_study, kwargs_study_optimize

FLAGS DE COMPORTAMIENTO (4):
  return_best, verbose, show_progress, suppress_warnings

OTROS (2):
  n_jobs, output_file
```

#### Eliminar 3 parámetros

| Parámetro | Por qué eliminar | Alternativa |
|-----------|-----------------|-------------|
| `verbose` | Redundante con `show_progress`. En sklearn, `verbose` controla todo. Aquí `verbose` imprime info de folds y `show_progress` muestra barra — la distinción es confusa para el usuario. | Unificar en `verbose`: `0`=silencio, `1`=barra de progreso, `2`=barra+info de folds |
| `output_file` | Comportamiento inconsistente con grid search (TSV vs logs de optuna). Redirigir logs de optuna es responsabilidad del usuario, no de la función de búsqueda. | Eliminar. Documentar cómo hacerlo via `optuna.logging.set_file_handler()` |
| `n_jobs` | Solo se usa en path `TimeSeriesFold`, ignorado en `OneStepAheadFold`. Es un parámetro del backtesting, no de la búsqueda. | Moverlo a atributo del `cv` object, o documentar que solo aplica a `TimeSeriesFold` |

#### Fusionar 2 en 1

`kwargs_create_study` + `kwargs_study_optimize` → un solo dict `optuna_kwargs`:

```python
# ACTUAL (2 params, verbose)
bayesian_search_forecaster(
    ...,
    kwargs_create_study={'storage': 'sqlite:///study.db'},
    kwargs_study_optimize={'timeout': 300, 'callbacks': [cb]}
)

# PROPUESTO (1 param, estructura clara)
bayesian_search_forecaster(
    ...,
    optuna_kwargs={
        'create_study': {'storage': 'sqlite:///study.db'},
        'optimize': {'timeout': 300, 'callbacks': [cb]}
    }
)
```

**Ventajas:** 
- Un solo parámetro en vez de dos.
- El nombre `optuna_kwargs` deja claro que son opciones avanzadas de optuna.
- Las sub-keys `'create_study'` y `'optimize'` mapean directamente a las funciones de optuna.

#### Añadir 1 parámetro

`return_study: bool = False` — para devolver el `optuna.Study` completo en lugar de solo `best_trial`. Permite visualizaciones (`optuna.visualization`), resume de la búsqueda, y acceso a trials fallidos.

Alternativa más limpia: siempre devolver `study` en el segundo elemento de la tupla (breaking change) — ver punto 1.7.

#### Lo que NO cambiaría

| Parámetro | Por qué mantener |
|-----------|-----------------|
| `return_best` | Muy útil y bien entendido. Eliminar la mutación del forecaster sería "más puro" pero peor UX para la mayoría de usuarios. |
| `suppress_warnings` | Cross-cutting concern necesario. No hay forma elegante de eliminarlo sin context managers que compliquen al usuario. |
| `random_state` | Aunque es redundante con `optuna_kwargs`, es un patrón tan familiar de sklearn que su ausencia confundiría. La ergonomía supera a la pureza. |

#### Firma propuesta final

```python
def bayesian_search_forecaster(
    # --- Core (no cambian) ---
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold | OneStepAheadFold,
    search_space: Callable,
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    # --- Optuna ---
    n_trials: int = 10,
    random_state: int = 123,
    optuna_kwargs: dict | None = None,       # ← fusión de kwargs_create_study + kwargs_study_optimize
    # --- Comportamiento ---
    return_best: bool = True,
    return_study: bool = False,              # ← nuevo: devolver optuna.Study completo
    verbose: int = 1,                        # ← fusión de verbose (bool) + show_progress (bool)
    suppress_warnings: bool = False,
) -> tuple[pd.DataFrame, object]:
    """
    ...
    verbose : int, default 1
        Controls verbosity level:
        - 0: No output (silent mode).
        - 1: Show progress bar only.
        - 2: Show progress bar and fold/backtesting details.
    ...
    """
```

**Resumen del cambio: 16 → 13 parámetros**

| Acción | Parámetros | Resultado |
|--------|-----------|-----------|
| Mantener sin cambios | `forecaster`, `y`, `cv`, `search_space`, `metric`, `exog`, `n_trials`, `random_state`, `return_best`, `suppress_warnings` | 10 |
| Fusionar `verbose` + `show_progress` | `verbose: int` | 1 (era 2) |
| Fusionar `kwargs_create_study` + `kwargs_study_optimize` | `optuna_kwargs: dict \| None` | 1 (era 2) |
| Añadir nuevo | `return_study: bool` | 1 |
| Eliminar `output_file` | — | 0 (era 1) |
| Eliminar `n_jobs` | — | 0 (era 1) |
| **Total** | | **13** |

**Nota sobre `n_jobs`:** Si eliminarlo es demasiado disruptivo (es el mismo parámetro en `grid_search_forecaster` y `random_search_forecaster`), la alternativa pragmática es mantenerlo con docstring actualizado indicando que solo aplica a `TimeSeriesFold`.

---

## Parte 2 — Diseño / Arquitectura

### 2.1 Duplicación masiva entre single y multiseries (~70%)

**Problema actual:**  
`_bayesian_search_optuna` (340 líneas) y `_bayesian_search_optuna_multiseries` (460 líneas) comparten ~70% del código idéntico:

| Bloque de código | single | multi | ¿Idéntico? |
|---|---|---|---|
| Warning suppression | ✓ | ✓ | ≈ idéntico |
| deepcopy forecaster | ✓ | ✓ | ≈ idéntico |
| CV validation + setup | ✓ | ✓ | ≈ idéntico (multi tiene series vs y) |
| Metric parsing | ✓ | ✓ | ≈ idéntico (multi tiene aggregate_metric) |
| `_objective` TimeSeriesFold | ✓ | ✓ | **Diferente** (backtesting_forecaster vs _multiseries) |
| `_objective` OneStepAheadFold | ✓ | ✓ | **Diferente** (split + metrics) |
| Study creation + sampler | ✓ | ✓ | **Idéntico** |
| Logging setup | ✓ | ✓ | **Idéntico** |
| `study.optimize()` | ✓ | ✓ | **Idéntico** |
| Result extraction loop | ✓ | ✓ | ≈ idéntico (multi tiene levels + aggregate) |
| `return_best` refit + print | ✓ | ✓ | ≈ idéntico (multi tiene levels + series vs y) |

**Rediseño — Función única parametrizada por tipo:**

```python
def _bayesian_search_optuna(
    forecaster, data, cv, search_space, metric, exog=None, 
    levels=None, aggregate_metric=None, ...
):
    is_multiseries = levels is not None or isinstance(data, (dict, pd.DataFrame))
    
    # 1. Setup común
    _setup_cv(...)
    _parse_metrics(...)
    
    # 2. Objective function (único punto de divergencia real)
    objective = _build_objective(
        forecaster, data, cv, metric, exog, levels, aggregate_metric, ...
    )
    
    # 3. Study execution (100% común)
    study = _run_study(objective, n_trials, random_state, ...)
    
    # 4. Result extraction (parametrizado)
    results = _extract_results(study, metric_names, is_multiseries, ...)
    
    # 5. Refit (parametrizado)
    if return_best:
        _refit_best(forecaster, results, data, exog, ...)
    
    return results, study
```

**Pero:** Esto requiere refactorizar la divergencia del `_objective`. La clave es que solo difieren en:
1. Qué función de evaluación se llama (`backtesting_forecaster` vs `backtesting_forecaster_multiseries`)
2. Cómo se procesan las métricas post-evaluación (aggregate en multi)
3. El split one-step-ahead (parámetros diferentes)

Se podría usar una **strategy o callable** para encapsular estas diferencias.

---

### 2.2 Separación de responsabilidades

**Problema actual:**  
Una función de ~200 líneas (sin contar docstring) hace TODO:

```
_bayesian_search_optuna():
    1. Suppress warnings                    ← cross-cutting concern
    2. deepcopy + type checking             ← setup
    3. CV validation + setup                ← input processing
    4. Metric parsing + validation          ← input processing
    5. Define _objective (2 versiones)      ← core logic
    6. Configure direction + progress       ← optuna config
    7. Setup logging / output_file          ← infrastructure
    8. Create study + set sampler           ← optuna lifecycle
    9. study.optimize()                     ← execution
    10. Close handler                       ← cleanup
    11. Validate search_space keys          ← post-validation
    12. Extract trials → lists              ← result processing
    13. Build DataFrame + sort              ← result formatting
    14. Refit forecaster + print            ← side effect
    15. Restore warnings                    ← cleanup
```

**Rediseño — Descomponer en funciones con responsabilidad única:**

```python
def bayesian_search_forecaster(...):
    # 1. Input validation & preprocessing
    forecaster_search, cv, metric, metric_names = _prepare_search_inputs(
        forecaster, y, cv, metric, exog, suppress_warnings
    )
    
    # 2. Build objective function
    objective_fn = _build_objective_fn(
        forecaster_search, y, cv, exog, metric, n_jobs, verbose, suppress_warnings
    )
    
    # 3. Run optuna study
    study = _run_optuna_study(
        objective_fn, n_trials, random_state,
        kwargs_create_study, kwargs_study_optimize,
        is_regression, show_progress, output_file
    )
    
    # 4. Extract & format results
    results = _format_study_results(
        study, forecaster_search, metric, metric_names
    )
    
    # 5. Refit best (optional)
    if return_best:
        _refit_with_best(forecaster, results, y, exog, cv)
    
    return results, study
```

**Ventajas:**
- Cada función es testeable independientemente.
- `_run_optuna_study` sería reutilizable entre single y multiseries.
- `_build_objective_fn` es el único punto de divergencia real.
- El try/finally se simplifica (solo en `_run_optuna_study`).

---

### 2.3 El `_objective` no debería usar `nonlocal`

**Problema ya identificado (punto 5 del análisis anterior).** En un rediseño completo, el `_objective` devolvería solo el valor optimizado y almacenaría las métricas adicionales via `trial.set_user_attr()`.

```python
def _objective(trial):
    ...
    for name, value in zip(metric_names, metrics):
        trial.set_user_attr(name, value)
    return metrics[0]
```

---

### 2.4 El sampler se asigna DESPUÉS de `create_study()`

**Problema actual:**
```python
study = optuna.create_study(**kwargs_create_study)
if 'sampler' not in kwargs_create_study.keys():
    study.sampler = TPESampler(seed=random_state)  # ← override post-creación
```

Esto es raro porque `create_study()` acepta `sampler` como parámetro. El sampler se asigna después, sobrescribiendo el que optuna creó por defecto internamente.

**Rediseño:**
```python
if 'sampler' not in kwargs_create_study:
    kwargs_create_study['sampler'] = TPESampler(
        seed=random_state, multivariate=True, group=True, consider_endpoints=True
    )
study = optuna.create_study(**kwargs_create_study)
```

**Ventaja:** El sampler se pasa al constructor, que es lo idiomático. No hay override posterior.

---

### 2.5 Logging setup es frágil y muta estado global

**Problema actual:**
```python
optuna.logging.disable_default_handler()
logger = logging.getLogger('optuna')
logger.setLevel(logging.INFO)
for handler in logger.handlers.copy():
    if isinstance(handler, logging.StreamHandler):
        logger.removeHandler(handler)
handler = logging.FileHandler(output_file, mode="w")
logger.addHandler(handler)
```

Esto muta el logger **global** de optuna. Si dos procesos/threads ejecutan búsquedas concurrentes, se pisarían los handlers. Además, el estado original del logger nunca se restaura.

**Rediseño:**
```python
@contextmanager
def _optuna_logging_context(output_file: str | None):
    """Temporarily configure optuna logging, restore on exit."""
    logger = logging.getLogger('optuna')
    original_level = logger.level
    original_handlers = logger.handlers.copy()
    
    try:
        optuna.logging.disable_default_handler()
        if output_file is not None:
            logger.setLevel(logging.INFO)
            for h in logger.handlers.copy():
                if isinstance(h, logging.StreamHandler):
                    logger.removeHandler(h)
            file_handler = logging.FileHandler(output_file, mode="w")
            logger.addHandler(file_handler)
        else:
            logger.setLevel(logging.WARNING)
        yield
    finally:
        if output_file is not None and file_handler:
            file_handler.close()
            logger.removeHandler(file_handler)
        logger.setLevel(original_level)
        # Restore original handlers
        for h in logger.handlers.copy():
            logger.removeHandler(h)
        for h in original_handlers:
            logger.addHandler(h)
```

**Ventaja:** Usa context manager → garantiza cleanup. Estado global se restaura siempre.

---

### 2.6 La validación de `search_space` keys es insuficiente

**Problema actual:**
```python
search_space_best = search_space(best_trial)
if search_space_best.keys() != best_trial.params.keys():
    raise ValueError(...)
```

Esto **re-ejecuta** `search_space()` sobre `best_trial` solo para validar que las keys coinciden. Problemas:
1. Asume que `search_space` es determinista (siempre devuelve las mismas keys para el mismo trial).
2. Re-ejecuta los `suggest_*` de optuna innecesariamente (podrían tener side effects).
3. Solo valida el `best_trial`, no todos los trials.
4. Se ejecuta **después** de `study.optimize()` — si hay un problema, ya se perdió todo el compute.

**Rediseño:** Validar en el primer trial (o antes de la optimización con un trial ficticio), no al final.

---

## Parte 3 — Valor de retorno / Output

### 3.1 Devolver `study` en vez de `best_trial`

**Ya descrito en 1.7.** El `study` contiene `best_trial` y mucho más. Es el cambio de API más valioso del rediseño.

---

### 3.2 Añadir `trial_number` al DataFrame de resultados

**Problema actual:**  
No hay forma de correlacionar una fila del DataFrame con un trial específico del study de optuna.

```python
# ACTUAL
results = pd.DataFrame({
    'lags': lags_list,
    'params': params_list,
    **metric_dict
})
```

**Rediseño:**
```python
results = pd.DataFrame({
    'trial_number': [t.number for t in study.get_trials(states=[TrialState.COMPLETE])],
    'lags': lags_list,
    'params': params_list,
    **metric_dict
})
```

**Ventaja:** Permite `study.trials[results.loc[0, 'trial_number']]` para acceder al trial completo de optuna.

---

### 3.3 Columna `params` duplica información

**Problema actual:**
```python
results = pd.concat([results, results['params'].apply(pd.Series)], axis=1)
```

Esto crea columnas individuales por parámetro **y** mantiene la columna `params` con el dict completo. La información está duplicada.

**Rediseño:** Mantener ambas (es útil), pero generar las columnas individuales con `pd.json_normalize` (más robusto con nested dicts) o directamente al construir el DataFrame.

```python
params_df = pd.DataFrame(params_list)
results = pd.concat([results, params_df], axis=1)
```

---

### 3.4 `return_best` muta el forecaster original

**Problema actual:**
```python
if return_best:
    # NOTE: Here we use the actual forecaster passed by the user
    forecaster.set_lags(best_lags)
    forecaster.set_params(best_params)
    forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
    print(...)
```

Problemas:
1. **Side effect silencioso:** Una función de "búsqueda" muta el objeto que recibe. Viola el principio de menor sorpresa.
2. **`print()` no es silenciable:** Contamina stdout. No se puede suprimir sin redirigir stdout.
3. **Acoplamiento:** La función de búsqueda no debería saber cómo hacer fit de un forecaster.

**Rediseño opción A — Documentar mejor y usar `warnings.warn`:**
```python
if return_best:
    forecaster.set_lags(best_lags)
    forecaster.set_params(best_params)
    forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
    warnings.warn(
        f"`Forecaster` refitted using the best-found lags and parameters...",
        stacklevel=2
    )
```

**Rediseño opción B — Separar búsqueda y refit:**
```python
# La función de búsqueda NUNCA muta el forecaster
results, study = bayesian_search_forecaster(forecaster, y, cv, ...)

# El usuario decide si refit
forecaster.set_lags(results.loc[0, 'lags'])
forecaster.set_params(results.loc[0, 'params'])
forecaster.fit(y=y, exog=exog)
```

**Decisión:** Opción A es pragmática (mantiene backward compat). Opción B es más limpia pero es breaking change. En un rediseño completo, B.

---

## Parte 4 — Rendimiento

### 4.1 Cache de `_train_test_split_one_step_ahead` (ya identificado)

En el path `OneStepAheadFold`, el split solo depende de los lags. Cuando no cambian (mayoría de los trials), el split se puede cachear. Ver punto 4 del análisis anterior para la implementación.

---

### 4.2 Extracción de resultados: iteración innecesaria

**Problema actual:**
```python
for i, trial in enumerate(study.get_trials()):
    estimator_params = {k: v for k, v in trial.params.items() if k != 'lags'}
    lags = trial.params.get('lags', ...)
    params_list.append(estimator_params)
    lags_list.append(lags)
    for m, m_values in zip(metric, metric_values[i]):
        ...
```

Con `trial.set_user_attr()` (punto 5 anterior), toda la extracción se simplifica:

```python
trials = study.get_trials(states=[TrialState.COMPLETE])
results = pd.DataFrame([
    {
        'trial_number': t.number,
        'lags': initialize_lags(..., t.params.get('lags', default_lags))[0],
        'params': {k: v for k, v in t.params.items() if k != 'lags'},
        **{name: t.user_attrs[name] for name in metric_names}
    }
    for t in trials
])
```

**Ventaja:** Una list comprehension limpia, sin variable `metric_values` externa, inmune a desalineamientos.

---

## Parte 5 — Resumen del rediseño

### Cambios de API (breaking)

| Cambio | Impacto | Prioridad |
|--------|---------|-----------|
| Devolver `study` en vez de `best_trial` | Breaking (tipo de retorno) | Alta |
| `kwargs_create_study: dict = {}` → `dict \| None = None` | Técnicamente breaking si alguien hace `is {}` | Alta |
| Eliminar wrappers vacíos (interna) | No breaking (se mantiene API pública) | Media |
| `return_best` → no mutar por defecto | Breaking (comportamiento) | Baja |

### Cambios de diseño (no breaking)

| Cambio | Esfuerzo | Prioridad |
|--------|----------|-----------|
| `trial.set_user_attr()` + filtro `COMPLETE` | Bajo | Alta |
| Sampler antes de `create_study()` | Trivial | Alta |
| Context manager para logging | Bajo | Alta |
| Unificar single + multiseries en función parametrizada | Alto | Media |
| Descomponer en subfunciones | Medio | Media |
| Añadir `trial_number` al DataFrame | Trivial | Baja |
| Cachear split OneStepAhead | Medio | Alta |

### Firma propuesta final

```python
def bayesian_search_forecaster(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold | OneStepAheadFold,
    search_space: Callable,
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    n_trials: int = 10,
    random_state: int = 123,
    return_best: bool = True,
    n_jobs: int | str = 'auto',
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False,
    output_file: str | None = None,
    kwargs_create_study: dict | None = None,
    kwargs_study_optimize: dict | None = None,
) -> tuple[pd.DataFrame, optuna.Study]:
```

Cambios vs actual:
1. `dict = {}` → `dict | None = None` (**fix bug + type safety**)
2. Return type: `object` → `optuna.Study` (**más información al usuario**)
3. Todo lo demás se mantiene igual en la firma (backward compatible)

Los cambios internos (trial.set_user_attr, context manager, cache split, sampler en create_study, try/finally) no afectan a la API pública.
---

## Análisis de compatibilidad con Optuna v5.0.0

> Fecha: 2026-03-01  
> Última versión estable de optuna: **v4.7.0** (19 enero 2026)  
> Optuna v5.0 previsto: **verano 2026** (según [roadmap oficial](https://medium.com/optuna/optuna-v5-roadmap-ac7d6935a878))  
> Constraint actual de skforecast: `optuna>=2.10`

### Contexto: qué es Optuna v5

El roadmap de v5.0 se centra en **tres ejes estratégicos**, ninguno de ellos orientado a romper APIs existentes:

1. **Generative AI**: toolchain para prompt optimization + MCP server para agentes LLM
2. **Disponibilidad multiplataforma**: reimplementación parcial en Rust → WASM, soporte non-Python (Google Sheets, multi-lenguaje)
3. **Crecimiento sostenible**: mejoras al default sampler, AutoSampler multiobjective/constrained, multi-fidelity optimization, ELA

**No se anuncian breaking changes explícitos en la API pública de Python.** Es un roadmap de expansión, no de ruptura.

### Breaking changes ya ocurridos en v4.x (que sí nos afectan)

| Versión | Cambio | ¿Afecta a skforecast? |
|---------|--------|----------------------|
| v4.4.0 | `TPESampler`: **todos los args son keyword-only** (#6041) | **No** — ya usamos `TPESampler(seed=random_state)` y `seed` siempre fue keyword |
| v4.4.0 | `consider_prior=False` removed → forzado a `True` (#6007) | **No** — no usamos `consider_prior` |
| v4.4.0 | `restart_strategy`/`inc_popsize` removed de CmaEsSampler | **No** — no usamos CmaEsSampler |
| v4.6.0 | Drop Python 3.8, soporte Python 3.13 | **No** — skforecast ya requiere Python ≥3.10 |
| v4.6.0 | `TrialState.__repr__` y `__str__` cambiado (#6281) | **No** — no dependemos de la representación string |

### Deprecaciones activas con fecha de removal

| Feature | Deprecated en | Removal programado | ¿Nos afecta? |
|---------|:---:|:---:|:---:|
| `consider_prior` arg en TPESampler | v4.3.0 | **v6.0.0** | No — no lo usamos |
| Args posicionales en `TPESampler.__init__` | v4.4.0 | **v6.0.0** | No — ya pasamos `seed` como keyword |

**Importante:** las deprecaciones actuales apuntan a **v6.0**, no a v5.0. Esto sugiere que v5.0 no eliminará APIs existentes que usamos.

### Puntos de contacto entre skforecast y optuna

Uso actual en `_bayesian_search_optuna`:

```python
import optuna
from optuna.samplers import TPESampler

# 1. Logging
optuna.logging.disable_default_handler()         # ✅ Estable
logging.getLogger("optuna").setLevel(...)         # ✅ Estable (stdlib)

# 2. Study creation
study = optuna.create_study(**kwargs_create_study) # ✅ Estable

# 3. Sampler
study.sampler = TPESampler(seed=random_state)      # ✅ Estable hasta v6.0+

# 4. Optimization
study.optimize(func, **kwargs_study_optimize)      # ✅ Estable
# show_progress_bar via kwargs                     # ✅ Estable

# 5. Trial access
study.trials / study.best_trial                    # ✅ Estable
trial.suggest_float/int/categorical                # ✅ Estable (define-by-run core)
```

**Ninguno de estos puntos está marcado como deprecated ni mencionado en el roadmap de v5 como candidato a cambio.**

### Riesgos potenciales (especulativos) para v5

| Riesgo | Probabilidad | Impacto | Acción |
|--------|:---:|:---:|--------|
| AutoSampler se convierte en default sampler | Media | Bajo | Nosotros forzamos `TPESampler` explícitamente, no nos afecta |
| Cambios internos por core en Rust | Baja | Nulo | API pública de Python permanece igual; Rust es implementación interna |
| Nuevos warnings por features experimentales (`multivariate`, `group`) | Media | Bajo | `multivariate`/`group` llevan como experimental desde v2.2/v2.8; en v5 podrían estabilizarlos (no romper) |
| Cleanup de APIs realmente viejas de v2.x/v3.x | Baja | Bajo | No usamos nada pre-v3.0 |
| Cambio de firma en `study.optimize()` | Muy baja | Alto | Sería un breaking change masivo; improbable |

### Problema real: constraint `optuna>=2.10` demasiado permisivo

El riesgo real **no es v5.0**, sino que un usuario instale hoy `optuna==2.10` o `optuna==3.0` y rompa todo. Los tests probablemente solo pasan con optuna ≥4.x.

**Recomendación:** subir el constraint a `optuna>=3.6` (mínimo razonable) o idealmente `optuna>=4.0` (estable, keyword-only TPESampler, GPSampler disponible).

```toml
# pyproject.toml actual
"optuna>=2.10",

# Recomendado
"optuna>=4.0",
```

### Mejoras de optuna v4.x que podríamos aprovechar

| Feature (v4.x) | Versión | Aplicable a skforecast |
|-----------------|:---:|:---:|
| TPESampler 5x más rápido | v4.5.0 | ✅ Automático si optuna≥4.5 |
| `multivariate`/`group`/`consider_endpoints` mejorados | v4.0+ | ✅ Ya recomendados en nuestro análisis de samplers |
| GPSampler constrained + multi-objective | v4.2–4.5 | No aplica (single-objective) |
| AutoSampler multi-objective + constrained | v4.6 | No aplica directamente |
| Mejoras de `plot_hypervolume_history` | v4.5 | No aplica |
| `TrialState` cached para TPESampler | v4.6 | ✅ Automático |

### Conclusión

| Aspecto | Estado |
|---------|--------|
| ¿v5.0 romperá nuestro código? | **No** (con alta probabilidad) |
| ¿Hay deprecaciones que nos afecten? | **No** — las que existen apuntan a v6.0 y no las usamos |
| ¿Debemos prepararnos para v5? | **Sí** — subir constraint a `optuna>=4.0` |
| ¿v5 trae algo que debamos adoptar? | Nada anunciado aún; monitorizar AutoSampler y posibles nuevas APIs |
| ¿Riesgo real? | El constraint `>=2.10` actual, no v5.0 |

---

## 7. Recomendaciones para todas las dependencias core

> **Fecha del análisis:** febrero 2026  
> **Criterio principal:** alinear todos los floors a la era de `numba>=0.59` (enero 2024), que es el constraint más reciente y actúa como **ancla temporal** del proyecto.

### 7.1 Estado actual vs. recomendado

| Dependencia | Constraint actual | Floor fecha | Última versión | Recomendado | Floor fecha rec. | Urgencia |
|-------------|-------------------|-------------|----------------|-------------|------------------|----------|
| numpy | `>=1.24` | Dic 2022 | 2.4.2 (Feb 2026) | `>=1.26` | Sep 2023 | 🟡 Media |
| pandas | `>=1.5, <3.0` | Sep 2022 | 3.0.1 (Feb 2026) | `>=2.1` (sin cap) | Ago 2023 | 🔴 **URGENTE** |
| tqdm | `>=4.57` | Feb 2021 | 4.67.3 (Ene 2026) | `>=4.66` | Jul 2023 | 🟢 Baja |
| scikit-learn | `>=1.2` | Dic 2022 | 1.8.0 (Dic 2025) | `>=1.4` | Ene 2024 | 🟡 Media |
| scipy | `>=1.3.2` | **Nov 2019** | 1.17.1 (Feb 2026) | `>=1.12` | Ene 2024 | 🔴 **Alta** |
| optuna | `>=2.10` | Jul 2021 | 4.7.0 (Ene 2026) | `>=3.6` | Mar 2024 | 🔴 **Alta** |
| joblib | `>=1.1` | Sep 2021 | 1.5.3 (Dic 2025) | `>=1.3` | Jun 2023 | 🟡 Media |
| numba | `>=0.59` | Ene 2024 | 0.64.0 (Feb 2026) | `>=0.59` (sin cambio) | Ene 2024 | ✅ OK |
| rich | `>=13.9` | Sep 2024 | 14.3.3 (Feb 2026) | `>=13.9` (sin cambio) | Sep 2024 | ✅ OK |

### 7.2 Análisis detallado por dependencia

---

#### 7.2.1 numpy: `>=1.24` → `>=1.26`

**Situación actual:** numpy 1.24 es de diciembre 2022 (3+ años). El ecosistema ya migró a numpy 2.x.

**Por qué `>=1.26`:**
- numpy 1.26 (Sep 2023) es la **última versión 1.x** y el puente oficial hacia numpy 2.0
- Incluye la capa de compatibilidad con numpy 2.0 (`numpy._core` aliases)
- Es el mínimo que scikit-learn 1.4+ y pandas 2.1+ esperan
- No exigimos numpy 2.0 como mínimo porque aún hay usuarios en 1.x

**Alternativa agresiva:** `>=2.0` — solo si skforecast ya no necesita soportar numpy 1.x. Descartable por ahora.

**Riesgo de mantener `>=1.24`:** Usuarios con numpy 1.24/1.25 pueden tener incompatibilidades silenciosas con el resto del stack moderno.

---

#### 7.2.2 pandas: `>=1.5, <3.0` → `>=2.1` (ELIMINAR cap `<3.0`) 🔴 URGENTE

**Situación actual:** pandas 3.0.0 se liberó en **enero 2026**. El cap `<3.0` **está bloqueando activamente** a usuarios que actualizan pandas.

**Problemas del cap `<3.0`:**
1. **Bloqueo en instalación:** `pip install skforecast` falla si el usuario ya tiene pandas 3.0+
2. **Incompatibilidad con el ecosistema:** scikit-learn, matplotlib, etc. ya soportan pandas 3.0
3. **Señal negativa:** proyectos con caps estrictos son percibidos como mal mantenidos

**Por qué `>=2.1`:**
- pandas 2.0 (Abr 2023) fue un breaking change mayor (CoW, ArrowDtype, nullable dtypes por defecto)
- pandas 2.1 (Ago 2023) estabilizó la serie 2.x
- Alinea con la era numba 0.59

**Acción requerida:**
1. **Inmediato:** Eliminar el cap `<3.0`
2. **Testing:** Ejecutar test suite completo con pandas 3.0.1 para identificar roturas
3. **Principales cambios en pandas 3.0 a vigilar:**
   - Copy-on-Write activado por defecto
   - Eliminación de APIs deprecadas en 2.x (`DataFrame.append`, `Series.swaplevel` renombrado, etc.)
   - `inplace` parameter removido de algunos métodos
   - Cambios en el comportamiento de `groupby` con valores NA

---

#### 7.2.3 tqdm: `>=4.57` → `>=4.66`

**Situación actual:** tqdm 4.57 es de febrero 2021. Librería muy estable — raramente rompe.

**Por qué `>=4.66`:**
- tqdm 4.66.3 (May 2024) incluye fix de seguridad **CVE-2024-34062** (eval safety en CLI)
- 4.66 (Jul 2023) es un buen punto de corte en la era numba 0.59
- Skforecast solo usa `tqdm` para barras de progreso — riesgo de rotura mínimo

**Nota:** Si se quiere ser conservador, `>=4.66.3` para garantizar el fix de CVE.

---

#### 7.2.4 scikit-learn: `>=1.2` → `>=1.4`

**Situación actual:** scikit-learn 1.2 es de diciembre 2022. La 1.4 (Ene 2024) alinea con la era.

**Por qué `>=1.4`:**
- scikit-learn 1.4 introdujo mejoras importantes en `set_output`, metadata routing, y HGBT
- 1.3 (Jun 2023) deprecó varias APIs que skforecast podría estar usando
- 1.4 tiene mejor soporte para Python 3.12+
- Es la versión que coincide temporalmente con numba 0.59

**Riesgo de mantener `>=1.2`:** APIs deprecadas en 1.3/1.4 podrían emitir warnings confusos para usuarios con versiones intermedias.

---

#### 7.2.5 scipy: `>=1.3.2` → `>=1.12` 🔴 ALTA PRIORIDAD

**Situación actual:** scipy 1.3.2 es de **noviembre 2019** — ¡más de 6 años! Es el floor más antiguo de todo el proyecto.

**Por qué `>=1.12`:**
- scipy 1.12.0 (Ene 2024) alinea perfectamente con numba 0.59
- scipy 1.3.2 no soporta Python 3.10+ (skforecast requiere `>=3.10`)
- **El constraint actual es ficticio:** ningún usuario con Python 3.10+ puede instalar scipy 1.3.2 — pip resuelve automáticamente a una versión superior. El floor da una falsa sensación de compatibilidad.
- scipy 1.12 incluye mejoras en `scipy.stats` y `scipy.optimize` que skforecast puede aprovechar

**Este es el cambio de floor más obvio y sin riesgo** — no rompe a nadie porque scipy 1.3.2 es inalcanzable con Python 3.10+.

---

#### 7.2.6 optuna: `>=2.10` → `>=3.6`

> Ya analizado en la sección 6. Resumen: subir a `>=3.6` (mínimo razonable) o `>=4.0` (ideal).

---

#### 7.2.7 joblib: `>=1.1` → `>=1.3`

**Situación actual:** joblib 1.1 es de septiembre 2021.

**Por qué `>=1.3`:**
- joblib 1.3 (Jun 2023) añadió `return_as='generator_unordered'` para mejor paralelismo
- joblib 1.3 mejoró el manejo de memoria en `Parallel`
- Versión estable y bien probada por scikit-learn (que la usa internamente)
- scikit-learn 1.4 ya requiere joblib>=1.3 internamente

**Nota:** Si scikit-learn `>=1.4` se adopta como floor, joblib `>=1.3` se resuelve transitivamente. Pero es buena práctica declarar el constraint explícito.

---

#### 7.2.8 numba: `>=0.59` → sin cambio ✅

El constraint más reciente. numba 0.59 (Ene 2024) es el ancla temporal del proyecto. Sin cambio necesario.

---

#### 7.2.9 rich: `>=13.9` → sin cambio ✅

Constraint reciente (Sep 2024). rich 14.x es compatible hacia atrás con 13.x. Sin cambio necesario.

---

### 7.3 Dependencias opcionales (breve revisión)

| Dependencia | Constraint actual | Observación |
|-------------|-------------------|-------------|
| statsmodels | `>=0.12, <0.15` | 0.12 es Nov 2020 (antiguo). Considerar subir a `>=0.13`. El cap `<0.15` es correcto (0.15 no existe aún) |
| matplotlib | `>=3.3, <3.11` | 3.3 es Jul 2020 (antiguo). Considerar subir a `>=3.7`. El cap `<3.11` es razonable |
| seaborn | `>=0.11, <0.14` | 0.11 es Dic 2020. Considerar subir a `>=0.12`. El cap `<0.14` es correcto |
| keras | `>=3.0` | Correcto — keras 3.0 es el mínimo multi-backend |

### 7.4 Pyproject.toml propuesto

```toml
dependencies = [
    "numpy>=1.26",
    "pandas>=2.1",
    "tqdm>=4.66",
    "scikit-learn>=1.4",
    "scipy>=1.12",
    "optuna>=3.6",
    "joblib>=1.3",
    "numba>=0.59",
    "rich>=13.9",
]

[project.optional-dependencies]
stats = [
    "statsmodels>=0.13, <0.15"
]

plotting = [
    "matplotlib>=3.7, <3.11",
    "seaborn>=0.12, <0.14",
    "statsmodels>=0.13, <0.15"
]

deeplearning = [
    "keras>=3.0"
]
```

### 7.5 Resumen de cambios por prioridad

#### 🔴 Urgente (hacer YA)

| Cambio | Motivo |
|--------|--------|
| Eliminar `<3.0` de pandas | Bloquea instalación con pandas 3.0+ (liberado Ene 2026) |
| scipy `>=1.3.2` → `>=1.12` | Floor ficticio — scipy 1.3.2 es inalcanzable con Python 3.10+ |
| optuna `>=2.10` → `>=3.6` | 4.5 años de distancia; APIs muy diferentes |

#### 🟡 Media (próxima release)

| Cambio | Motivo |
|--------|--------|
| numpy `>=1.24` → `>=1.26` | Alinear con era; 1.26 es puente a numpy 2.0 |
| scikit-learn `>=1.2` → `>=1.4` | Alinear con era; evitar deprecation warnings |
| joblib `>=1.1` → `>=1.3` | Transitivo con scikit-learn 1.4; mejoras de paralelismo |
| pandas floor `>=1.5` → `>=2.1` | Alinear con era; evitar bugs de pandas 1.x |

#### 🟢 Baja (cuando convenga)

| Cambio | Motivo |
|--------|--------|
| tqdm `>=4.57` → `>=4.66` | CVE fix; alineación temporal |
| Subir floors de opcionales | Limpieza general |

### 7.6 Plan de ejecución recomendado

```
1. [URGENTE] Ejecutar tests con pandas 3.0.1 → identificar roturas
2. [URGENTE] Eliminar cap <3.0 de pandas (o sustituir por <4.0 temporal si hay roturas)
3. [URGENTE] Subir floors de scipy y optuna (cambio sin riesgo real)
4. [RELEASE] Subir todos los floors restantes en un solo commit
5. [CI] Actualizar matrices de CI para testear con versiones mínimas nuevas
6. [CI] Añadir job de "pip install --upgrade" con últimas versiones para detectar roturas futuras
```