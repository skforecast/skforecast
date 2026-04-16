# Plan: Unificar API de intervalos y quantiles

## Problema actual

La API de predicción probabilística tiene tres caminos con convenciones inconsistentes:

| Método | Parámetro | Escala | Columnas output | N bounds |
|--------|-----------|--------|-----------------|----------|
| `predict_interval` | `interval=[5, 95]` | 0–100 | `pred, lower_bound, upper_bound` | Solo 2 |
| `predict_quantiles` | `quantiles=[0.05, 0.5, 0.95]` | 0–1 | `q_0.05, q_0.5, q_0.95` | N |
| `backtesting_forecaster` | `interval=[5, 50, 95]` | 0–100 | `pred, p_5, p_50, p_95` | N |

### Problemas concretos

1. **Dos escalas**: `predict_interval` usa 0–100, `predict_quantiles` usa 0–1.
2. **Tres convenciones de naming**: `lower/upper_bound`, `q_0.05`, `p_5`.
3. **Trampa silenciosa**: `predict_interval(interval=[0.05, 0.95])` pasa validación pero produce resultados incorrectos (quantiles 0.0005 y 0.0095).
4. **Backtesting bypasea `predict_interval`**: reimplementa la conversión percentil→quantil y llama a `predict_quantiles` directamente.
5. **`predict_interval` es limitado**: solo soporta 2 bounds, no N quantiles.

---

## Solución propuesta

### Principios

- **Una sola escala**: quantiles 0–1 (estándar en pandas, scipy, scikit-learn).
- **Dos métodos complementarios** con responsabilidades claras:
  - `predict_interval`: interfaz simple para intervalos de confianza.
  - `predict_quantiles`: interfaz flexible para N quantiles arbitrarios.
- **Naming consistente** entre predict y backtesting.
- **Transición gradual** con `FutureWarning` para no romper código existente.

### Cambios en `predict_interval`

**Antes (v0.22):**
```python
forecaster.predict_interval(
    steps=10,
    interval=[5, 95],       # percentiles 0-100
    method='bootstrapping',
)
# → pred | lower_bound | upper_bound
```

**Después:**
```python
forecaster.predict_interval(
    steps=10,
    interval=[0.05, 0.95],  # quantiles 0-1
    method='bootstrapping',
)
# → pred | lower_bound | upper_bound

# También acepta coverage como float (sin cambio)
forecaster.predict_interval(
    steps=10,
    interval=0.90,           # coverage 0-1 → [0.05, 0.95]
    method='bootstrapping',
)
# → pred | lower_bound | upper_bound
```

**Reglas:**
- `interval` como `float`: coverage 0–1. Se calcula `[0.5 - interval/2, 0.5 + interval/2]`. Sin cambio respecto a la API actual.
- `interval` como `list`/`tuple` de 2 valores: quantiles 0–1. Devuelve `pred, lower_bound, upper_bound`.
- `interval` como `list`/`tuple` de 2 valores todos > 1: se detectan como percentiles legacy, se aplica `/100`, se emite `FutureWarning`.

### Cambios en `predict_quantiles`

**Solo cambia la validación interna, la API no cambia:**

```python
forecaster.predict_quantiles(
    steps=10,
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],  # sin cambio, ya usa 0-1
)
# → q_0.1 | q_0.25 | q_0.5 | q_0.75 | q_0.9
```

Este método se mantiene para el caso de uso de N quantiles arbitrarios. Su interfaz no necesita cambios.

### Cambios en `backtesting_forecaster`

**Antes (v0.22):**
```python
backtesting_forecaster(
    ...,
    interval=[5, 95],           # percentiles 0-100
)
# → pred | lower_bound | upper_bound

backtesting_forecaster(
    ...,
    interval=[10, 50, 90],      # percentiles 0-100
)
# → pred | p_10 | p_50 | p_90
```

**Después:**
```python
backtesting_forecaster(
    ...,
    interval=[0.05, 0.95],      # quantiles 0-1
)
# → pred | lower_bound | upper_bound

backtesting_forecaster(
    ...,
    interval=[0.1, 0.5, 0.9],   # quantiles 0-1
)
# → pred | q_0.1 | q_0.5 | q_0.9
```

**Reglas internas de backtesting:**
- `interval` como `float`: coverage 0–1 → calcula 2 quantiles → llama a `predict_interval`.
- `interval` como `list` de 2 valores: llama a `predict_interval` directamente.
- `interval` como `list` de N > 2 valores: llama a `predict_quantiles` directamente.
- `interval` como `list` con todos valores > 1: `FutureWarning` + `/100` (transición).
- `interval` como `str` (`'bootstrapping'`): sin cambio, llama a `predict_bootstrapping`.
- `interval` como `distribution` object: sin cambio, llama a `predict_dist`.

### Naming de columnas unificado

| Caso | Columnas |
|------|----------|
| 2 quantiles (intervalo) | `pred`, `lower_bound`, `upper_bound` |
| N quantiles | `q_0.1`, `q_0.25`, `q_0.5`, `q_0.75`, `q_0.9` |

Se elimina el prefijo `p_`. Todo usa `q_` con valores en escala 0–1.

---

## Cambios en `check_interval`

La función `check_interval` en `utils.py` necesita actualizarse:

**Antes:**
```python
check_interval(interval=[5, 95])          # valida 0-100
check_interval(quantiles=[0.05, 0.95])    # valida 0-1
check_interval(alpha=0.95)                # valida 0-1
```

**Después:**
```python
check_interval(interval=[0.05, 0.95])     # valida 0-1
check_interval(quantiles=[0.05, 0.95])    # valida 0-1 (sin cambio)
check_interval(alpha=0.95)                # valida 0-1 (sin cambio)
```

Añadir detección de valores legacy (> 1) con `FutureWarning`.

---

## Plan de transición

### Versión N (próxima release)

1. **`predict_interval`**: acepta ambas escalas.
   - Si todos los valores en `interval` son ≤ 1 → nueva API (quantiles 0-1).
   - Si algún valor > 1 → legacy (percentiles 0-100), se emite `FutureWarning`:
     ```
     FutureWarning: Passing `interval` as percentiles (0-100) is deprecated. 
     Use quantiles (0-1) instead. For example, use `interval=[0.05, 0.95]` 
     instead of `interval=[5, 95]`. The old behavior will be removed in 
     skforecast X.X.X.
     ```
   - El default cambia de `[5, 95]` a `[0.025, 0.975]`.

2. **`backtesting_forecaster`**: misma lógica de detección y warning.
   - Columnas para N quantiles cambian de `p_` a `q_`.

3. **`check_interval`**: refactor de la rama `interval=` para validar 0-1 con detección legacy.

4. **`predict_quantiles`**: sin cambios en la API pública.

### Versión N+2

- Se elimina soporte para percentiles 0-100.
- `check_interval(interval=...)` solo acepta 0-1.
- Se eliminan los `FutureWarning`.

---

## Archivos a modificar

### Forecasters (predict_interval, predict_quantiles)
- `skforecast/recursive/_forecaster_recursive.py`
- `skforecast/recursive/_forecaster_recursive_multiseries.py`
- `skforecast/recursive/_forecaster_recursive_classifier.py`
- `skforecast/direct/_forecaster_direct.py`
- `skforecast/direct/_forecaster_direct_multivariate.py`
- `skforecast/deep_learning/_forecaster_rnn.py`

### Validación
- `skforecast/utils/utils.py` → `check_interval`

### Model selection (backtesting)
- `skforecast/model_selection/_validation.py` → `_fit_predict_forecaster`, `_fit_predict_forecaster_multiseries`, `_predict_and_calculate_metrics_one_step_ahead`

### Tests
- `**/tests/test_predict_interval.py`
- `**/tests/test_predict_quantiles.py`
- `**/tests/test_check_interval.py`
- `**/tests/test_backtesting_forecaster*.py`

### Documentación
- `llms-base.txt` / `llms.txt`
- User guides de prediction intervals
- API reference docstrings

---

## Resumen visual

```
                    ANTES                                    DESPUÉS
                    
predict_interval    [5, 95] (0-100)      →    [0.05, 0.95] (0-1)
                    → pred, lower, upper        → pred, lower, upper (sin cambio)

predict_quantiles   [0.05, 0.5, 0.95]    →    [0.05, 0.5, 0.95] (sin cambio)
                    → q_0.05, q_0.5, q_0.95    → q_0.05, q_0.5, q_0.95

backtesting         [5, 95] (0-100)      →    [0.05, 0.95] (0-1)
                    → pred, lower, upper        → pred, lower, upper
                    [10, 50, 90] (0-100)  →    [0.1, 0.5, 0.9] (0-1)
                    → pred, p_10, p_50, p_90    → pred, q_0.1, q_0.5, q_0.9
```

**Resultado final:** una sola escala (0-1), dos métodos con identidad clara, naming consistente, y transición sin ruptura.
