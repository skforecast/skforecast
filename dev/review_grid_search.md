# Evaluación de `_evaluate_grid_hyperparameters` — Análisis de necesidad real

> Fecha: 2026-03-03  
> Aplica a: `_evaluate_grid_hyperparameters`, `_evaluate_grid_hyperparameters_multiseries`, `_evaluate_grid_hyperparameters_stats`  
> Archivo: `skforecast/model_selection/_search.py`

---

## Resumen ejecutivo

Se analizan las tres funciones internas de evaluación de grid/random search: single-series (L265-543), multiseries (L1174-1529) y stats (L2210-2395), junto con sus 6 wrappers públicos (`grid_search_forecaster`, `random_search_forecaster`, `grid_search_forecaster_multiseries`, `random_search_forecaster_multiseries`, `grid_search_stats`, `random_search_stats`).

De los 14 puntos identificados, **5 son necesarios** (bugs reales o inconsistencias de alto impacto), **4 son recomendables** (mejoran calidad del código con poco esfuerzo), y **5 se descartan**.

---

## NECESARIOS — Deben implementarse

### 1. ~~`_evaluate_grid_hyperparameters_stats` sin `@manage_warnings`~~ **Done**

**Tipo:** Bug  
**Riesgo real:** Medio  
**Esfuerzo:** Trivial

`_evaluate_grid_hyperparameters_stats` (L2210) es la **única** función interna de evaluación que NO tiene el decorador `@manage_warnings`:

| Función | Decorador |
|---------|-----------|
| `_evaluate_grid_hyperparameters` (L264) | `@manage_warnings` ✓ |
| `_evaluate_grid_hyperparameters_multiseries` (L1173) | `@manage_warnings` ✓ |
| `bayesian_search_forecaster` (L546) | `@manage_warnings` ✓ |
| `bayesian_search_forecaster_multiseries` (L1530) | `@manage_warnings` ✓ |
| `_evaluate_grid_hyperparameters_stats` (L2210) | **FALTA** ✗ |

**Consecuencias:**

1. El parámetro `suppress_warnings` se acepta pero **no suprime** las warnings de skforecast (solo las pasa a `backtesting_stats` internamente, pero las warnings del propio loop de evaluación no se controlan).

2. Las llamadas a `warnings.filterwarnings()` en L2347-2349 **modifican el estado global de warnings** del caller:
```python
warnings.filterwarnings(
    'ignore', category=RuntimeWarning, message="The forecaster will be fit.*"
)
```
Sin `@manage_warnings` (que envuelve en `warnings.catch_warnings()`), este filtro persiste después de la llamada, suprimiendo RuntimeWarnings legítimas en código posterior del usuario.

**Afecta a:** `grid_search_stats` y `random_search_stats` (ambos llaman a `_evaluate_grid_hyperparameters_stats`).

---

### 2. ~~`_evaluate_grid_hyperparameters_stats` no usa `deepcopy_forecaster` — muta el forecaster del usuario~~ **Done**

**Tipo:** Bug  
**Riesgo real:** Alto  
**Esfuerzo:** Bajo

Todas las funciones de evaluación usan `deepcopy_forecaster` para crear una copia del forecaster y operar sobre ella:

| Función | `deepcopy_forecaster` |
|---------|----------------------|
| `_evaluate_grid_hyperparameters` (L348) | ✓ |
| `_evaluate_grid_hyperparameters_multiseries` (L1273) | ✓ |
| `bayesian_search_forecaster` (L651) | ✓ |
| `bayesian_search_forecaster_multiseries` (L1644) | ✓ |
| `_evaluate_grid_hyperparameters_stats` | **FALTA** ✗ |

La función stats opera **directamente** sobre el forecaster del usuario:

```python
# Línea 2338 — opera sobre el forecaster original, no una copia
forecaster.set_params(params)
metric_values = backtesting_stats(forecaster=forecaster, ...)
```

**Consecuencias:**

1. **Si `return_best=False`:** después de la búsqueda, el forecaster del usuario tiene los parámetros de la **última** combinación probada, no los originales. El usuario no espera que su forecaster sea modificado.

2. **Si la búsqueda falla a mitad:** el forecaster queda con parámetros parciales de la última combinación que se intentó (exitosa o no, ya que `set_params` se ejecuta ANTES del `try`).

3. **Contraste con otras funciones:** en `_evaluate_grid_hyperparameters`, la búsqueda usa `forecaster_search` (copia) y solo modifica el `forecaster` original en el path `return_best=True` — comportamiento correcto y esperado.

**Solución:**
```python
forecaster_search = deepcopy_forecaster(forecaster)
# ... usar forecaster_search en el loop ...
if return_best:
    forecaster.set_params(best_params)  # solo aquí modificar el original
    forecaster.fit(...)
```

**Nota:** Verificar si `deepcopy_forecaster` es compatible con `ForecasterStats`. Si no lo es, usar `deepcopy(forecaster)` como alternativa.

---

### 3. ~~Sin protección contra `results` vacío cuando todas las combinaciones fallan~~ **Done**

**Tipo:** Bug (edge case)  
**Riesgo real:** Bajo-Medio  
**Esfuerzo:** Bajo

Si todas las combinaciones de parámetros lanzan excepciones (capturadas por el `try/except` + `continue`), los resultados están vacíos:

**En `_evaluate_grid_hyperparameters` (L501-519):**
```python
results = pd.DataFrame({
    'lags': lags_list,        # lista vacía
    'lags_label': lags_label_list,
    'params': params_list,
    **metric_dict
})
# results es un DataFrame vacío

if return_best:
    best_lags = results.loc[0, 'lags']  # ← KeyError!
```

**En `_evaluate_grid_hyperparameters_multiseries` (L1479):**
```python
results = pd.concat(metrics_list, axis=0)  # metrics_list = [] → ValueError: No objects to concatenate
```

**En `_evaluate_grid_hyperparameters_stats` (L2370-2388):**
```python
results = pd.DataFrame({'params': params_list, **metric_dict})  # vacío
if return_best:
    best_params = results.loc[0, 'params']  # ← KeyError!
```

**Cuándo ocurre:** Cuando el usuario proporciona un `param_grid` cuyos parámetros son incompatibles con el estimador (ej. `max_depth` para un `LinearRegression`), o cuando los datos provocan errores numéricos para todas las combinaciones.

**Solución:** Añadir un check antes de `return_best`:
```python
if results.empty:
    warnings.warn(
        "No valid parameter combinations found. All combinations raised exceptions.",
        RuntimeWarning
    )
    return results
```

**Afecta a:** las 3 funciones `_evaluate_grid_hyperparameters*`.

---

### 4. ~~`param_grid` se re-envuelve en `tqdm` en cada iteración del outer loop — nesting de objetos tqdm~~ **Done**

**Tipo:** Defecto de código  
**Riesgo real:** Bajo (no hay pérdida de datos, pero código incorrecto)  
**Esfuerzo:** Trivial

En `_evaluate_grid_hyperparameters` (L441) y `_evaluate_grid_hyperparameters_multiseries` (L1403):

```python
for lags_k, lags_v in lags_grid_tqdm:        # loop exterior
    ...
    if show_progress:
        param_grid = tqdm(param_grid, desc='params grid', position=1, leave=False)  # L441

    for params in param_grid:                  # loop interior
        ...
```

En la primera iteración del loop exterior, `param_grid` (una lista) se envuelve en un `tqdm`. En la segunda iteración, `param_grid` ya es un `tqdm` (cerrado tras iteración), y se envuelve de nuevo: `tqdm(tqdm(lista))`. En la tercera: `tqdm(tqdm(tqdm(lista)))`, etc.

**¿Por qué no es un bug de pérdida de datos?** Porque `tqdm`, incluso cerrado (`disable=True`), delega a su `self.iterable` en `__iter__`, permitiendo re-iteración de la lista original a través de la cadena de tqdms.

**¿Por qué sí es un defecto?**
1. Crea objetos tqdm anidados innecesariamente (overhead de memoria y CPU).
2. Cada iteración exterior crea un objeto `tqdm` adicional que nunca se recolecta hasta que termina la función.
3. Es código confuso: la variable `param_grid` cambia de tipo (`list` → `tqdm`) como efecto lateral.

**Solución:**
```python
for lags_k, lags_v in lags_grid_tqdm:
    ...
    if show_progress:
        param_grid_tqdm = tqdm(param_grid, desc='params grid', position=1, leave=False)
    else:
        param_grid_tqdm = param_grid

    for params in param_grid_tqdm:
        ...
```

**Afecta a:** `_evaluate_grid_hyperparameters` (L441) y `_evaluate_grid_hyperparameters_multiseries` (L1403).

---

### 5. ~~Multiseries (grid y bayesian) siempre ordena `ascending=True` — falta check `is_regression`~~ **Done**

**Tipo:** Bug latente / Inconsistencia  
**Riesgo real:** Bajo (actualmente todos los multiseries son regresión)  
**Esfuerzo:** Trivial

**Single-series** — correcto:
```python
# _evaluate_grid_hyperparameters L511, bayesian_search_forecaster L885
is_regression = forecaster_search.__skforecast_tags__['forecaster_task'] == 'regression'
...
results.sort_values(by=..., ascending=True if is_regression else False)
```

**Multiseries** — siempre ascending:
```python
# _evaluate_grid_hyperparameters_multiseries L1489
results.sort_values(by=metric_names[0], ascending=True)

# bayesian_search_forecaster_multiseries L1978
results.sort_values(by=metric_names[0], ascending=True)
```

Además, `bayesian_search_forecaster_multiseries` (L1887-1891) NO establece `direction` en el study de optuna. Compárese con `bayesian_search_forecaster` (L816-818) que sí lo hace:
```python
# bayesian_search_forecaster — correcto
if 'direction' not in kwargs_create_study.keys():
    kwargs_create_study['direction'] = 'minimize' if is_regression else 'maximize'

# bayesian_search_forecaster_multiseries — FALTA direction:
kwargs_create_study = kwargs_create_study.copy() if kwargs_create_study is not None else {}
if 'sampler' not in kwargs_create_study:
    kwargs_create_study['sampler'] = TPESampler(...)
# ← No direction. Usa default de optuna ('minimize') que es correcto para regresión, pero no explícito.
```

**Impacto actual:** Nulo, porque `ForecasterRecursiveMultiSeries` y `ForecasterDirectMultiVariate` siempre son regresión. Pero si se añade un forecaster multiseries de clasificación, el sort y la direction serían incorrectos.

**Solución:** Añadir `is_regression` check en las 2 funciones multiseries de grid, y añadir `direction` explícita en `bayesian_search_forecaster_multiseries`.

---

## RECOMENDABLES — Mejoran la calidad, poco esfuerzo

### 6. Enfoque de supresión de warnings inconsistente entre single-series y multiseries

**Tipo:** Inconsistencia  
**Esfuerzo:** Trivial

**Single-series** `_evaluate_grid_hyperparameters` (L476-479) — suprime solo un RuntimeWarning específico:
```python
warnings.filterwarnings(
    'ignore',
    category = RuntimeWarning, 
    message  = "The forecaster will be fit.*"
)
```

**Multiseries** `_evaluate_grid_hyperparameters_multiseries` (L1455-1456) — suprime **todas** las categorías de skforecast:
```python
for warn_category in warn_skforecast_categories:
    warnings.filterwarnings('ignore', category=warn_category)
```

Ambas persiguen el mismo objetivo: suprimir warnings repetitivas dentro del loop de evaluación. Debería usarse un enfoque consistente en ambas. El enfoque de la versión multiseries (suprimir todas las categorías de skforecast) es más robusto.

---

### 7. ~~Posición del check de `exog` para `return_best` inconsistente~~ **Done**

**Tipo:** Inconsistencia menor  
**Esfuerzo:** Trivial

**`_evaluate_grid_hyperparameters`** — check DESPUÉS de `deepcopy_forecaster`:
```python
forecaster_search = deepcopy_forecaster(forecaster)  # L348 — deep copy primero
...
if return_best and exog is not None and (len(exog) != len(y)):  # L385 — check después
    raise ValueError(...)
```

**`bayesian_search_forecaster`** — check ANTES de `deepcopy_forecaster`:
```python
if return_best and exog is not None and (len(exog) != len(y)):  # L648 — check primero
    raise ValueError(...)

forecaster_search = deepcopy_forecaster(forecaster)  # L651 — deep copy después
```

En la versión grid, si el check falla, se ha desperdiciado una deep copy. Es más eficiente validar primero.

---

### 8. ~~`aggregate_metric` con default mutable (lista literal)~~ **Done**

**Tipo:** Anti-patrón menor  
**Esfuerzo:** Trivial

En 4 firmas de funciones:
```python
# L930, L1048, L1182, L1537
aggregate_metric: str | list[str] = ['weighted_average', 'average', 'pooling'],
```

Es un default mutable (lista). En la práctica no causa bugs porque la lista nunca se muta in-place — solo se itera o se reemplaza con `if isinstance(aggregate_metric, str): aggregate_metric = [aggregate_metric]`. Pero viola la convención de Python.

**Solución:**
```python
aggregate_metric: str | list[str] | None = None,
...
if aggregate_metric is None:
    aggregate_metric = ['weighted_average', 'average', 'pooling']
```

**Nota:** Esto sería un cambio en la API pública. Evaluar si se justifica o si es preferible documentar el comportamiento actual.

---

### 9. ~~`type().__name__` vs `__skforecast_tags__` para checks de tipo~~ **Done**

**Tipo:** Inconsistencia de estilo  
**Esfuerzo:** Bajo

**Multiseries** usa string matching:
```python
# L1274
if type(forecaster_search).__name__ == 'ForecasterRecursiveMultiSeries':
# L1644
if forecaster_name == 'ForecasterRecursiveMultiSeries':
```

**Single-series** usa tags:
```python
# L349
is_regression = forecaster_search.__skforecast_tags__['forecaster_task'] == 'regression'
```

El sistema de tags (`__skforecast_tags__`) fue diseñado para abstraer el tipo de forecaster y permitir extensibilidad. Usar `type().__name__` es más frágil. Debería migrarse a tags donde sea posible.

---

## DESCARTADOS — No necesarios

### 10. Cacheo de `_train_test_split_one_step_ahead` en grid search

**Justificación del descarte:**  
En grid search, el split se calcula una vez por configuración de lags (dentro del loop exterior, antes del loop de parámetros). Como cada configuración de lags produce un split diferente, y el grid itera secuencialmente por lags, no hay splits redundantes.

Compárese con bayesian search, donde el mismo lags puede aparecer en múltiples trials aleatoriamente — ahí el cacheo sí es necesario (ya implementado con `_cached_split`).

---

### 11. `pd.json_normalize` en vez de `apply(pd.Series)` en resultados

**Justificación del descarte:**  
Todas las funciones usan `results['params'].apply(pd.Series)` para expandir los diccionarios de parámetros en columnas. El DataFrame de resultados tiene como máximo `len(lags_grid) × len(param_grid)` filas (típicamente 10-200). La diferencia de rendimiento con `pd.json_normalize` es despreciable. El tiempo de las funciones está dominado por el backtesting.

---

### 12. Warning sobre refit redundante en el loop

**Justificación del descarte:**  
La supresión de `"The forecaster will be fit.*"` en L476-479 es necesaria para evitar spam de warnings en cada iteración. El comportamiento es correcto: se suprime después de la primera evaluación exitosa. Se podría mover antes del loop para mayor claridad, pero el efecto es idéntico dado que `@manage_warnings` envuelve todo en `catch_warnings`.

---

### 13. `output_file` check de existencia redundante

**Justificación del descarte:**  
El patrón:
```python
if output_file is not None and os.path.isfile(output_file):
    os.remove(output_file)
...
if not os.path.isfile(output_file):   # primera escritura con header
    with open(output_file, 'w', ...) as f: ...
else:                                  # append sin header
    with open(output_file, 'a', ...) as f: ...
```

Parece redundante (ya se borró al inicio), pero es **defensivo y correcto**: si la primera combinación de parámetros falla (`continue`), el archivo no se crea. La siguiente combinación exitosa necesita detectar que el archivo no existe para escribir el header. El patrón es robusto.

---

### 14. Eficiencia de `deepcopy_forecaster` vs `deepcopy`

**Justificación del descarte:**  
`deepcopy_forecaster` usa `sklearn.base.clone()` para el estimador interno, que es más eficiente que `deepcopy` completo. La copia se hace una sola vez al inicio de la función, no per-iteración. El coste es despreciable comparado con los N×M backtestings.

---

## Resumen final

| #  | Issue | Tipo | Necesario | Esfuerzo | Afecta a |
|----|-------|------|-----------|----------|----------|
| 1  | ~~`_evaluate_grid_hyperparameters_stats` sin `@manage_warnings`~~ **Done** | Bug | **SÍ** | Trivial | `grid_search_stats`, `random_search_stats` |
| 2  | ~~Stats no usa `deepcopy_forecaster` — muta forecaster~~ **Done** | Bug | **SÍ** | Bajo | `grid_search_stats`, `random_search_stats` |
| 3  | ~~Sin protección contra `results` vacío~~ **Done** | Bug | **SÍ** | Bajo | 3 funciones `_evaluate_*` |
| 4  | ~~`param_grid` re-wrapped en tqdm (nesting)~~ **Done** | Defecto | **SÍ** | Trivial | `_evaluate_grid_*`, `_evaluate_grid_*_multiseries` |
| 5  | ~~Multiseries sin `is_regression` check + falta `direction`~~ **Done** | Inconsistencia | **SÍ** | Trivial | `_evaluate_grid_*_multiseries`, `bayesian_search_*_multiseries` |
| 6  | Warning suppression inconsistente single vs multi | Inconsistencia | Recomendable | Trivial | `_evaluate_grid_hyperparameters` |
| 7  | ~~Posición del check `exog`/`return_best`~~ **Done** | Inconsistencia | Recomendable | Trivial | `_evaluate_grid_hyperparameters` |
| 8  | ~~`aggregate_metric` default mutable~~ **Done** | Anti-patrón | Recomendable | Trivial | 4 funciones multiseries |
| 9  | ~~`type().__name__` vs `__skforecast_tags__`~~ **Done** | Estilo | Recomendable | Bajo | Funciones multiseries |
| 10 | Cacheo de split en grid | Perf | No (innecesario) | — | — |
| 11 | `json_normalize` vs `apply(pd.Series)` | Perf | No (despreciable) | — | — |
| 12 | Warning refit en loop | Diseño | No (correcto) | — | — |
| 13 | `output_file` check redundante | Diseño | No (defensivo, correcto) | — | — |
| 14 | Eficiencia de `deepcopy_forecaster` | Perf | No (ya optimizado) | — | — |

### Orden de implementación sugerido

1. **Puntos 1 + 2** (bugs en stats: añadir `@manage_warnings` + `deepcopy_forecaster`, un commit)
2. **Punto 4** (tqdm nesting: usar variable separada `param_grid_tqdm`, un commit)
3. **Punto 5** (añadir `is_regression` + `direction` en multiseries, un commit)
4. **Punto 3** (protección contra results vacío en las 3 funciones, un commit)
5. **Puntos 6 + 7** (inconsistencias menores, un commit)
6. **Punto 9** (migrar `type().__name__` a tags, evaluar impacto primero)
7. **Punto 8** (evaluar si el cambio de API es aceptable)
