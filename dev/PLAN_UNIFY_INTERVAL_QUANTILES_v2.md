# Plan: Unificar API de intervalos y quantiles (v2)

> Revisión del plan original incorporando: detección de valores mezclados/límite,
> manejo explícito de defaults de cobertura, alcance completo de forecasters
> (incluido Foundation), consumidores de columnas `p_`, y guards de regresión.

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
  - `predict_interval`: interfaz simple para intervalos de confianza (2 bounds).
  - `predict_quantiles`: interfaz flexible para N quantiles arbitrarios.
- **Naming consistente** entre predict y backtesting.
- **Transición gradual** con `FutureWarning` para no romper código existente.
- **Sin cambios silenciosos de comportamiento**: cualquier cambio de cobertura por
  defecto se documenta y, si es posible, se evita.

---

## Reglas de detección de escala (núcleo del plan)

Esta es la parte más delicada. Se define una única función de normalización
(`_normalize_interval_scale`) reutilizada por `predict_interval`, `predict_quantiles`,
`check_interval` y backtesting, para garantizar comportamiento idéntico en todos los caminos.

Dado `interval` como lista/tupla de valores numéricos:

| Caso | Detección | Acción |
|------|-----------|--------|
| Todos los valores en `[0, 1]` | Nueva API | Usar tal cual (quantiles 0–1) |
| Todos los valores en `(1, 100]` | Legacy percentiles | Dividir `/100` + `FutureWarning` |
| **Mezcla** (alguno ≤ 1 y alguno > 1) | Ambiguo | **`ValueError` explícito**, no adivinar |
| Algún valor `< 0` o `> 100` | Inválido | `ValueError` |

**Casos límite definidos explícitamente:**

- `interval=[1, 50]` → mezcla (`1` es quantil válido o percentil legacy). **`ValueError`**
  con mensaje claro pidiendo desambiguar.
- `interval=[0.5, 95]` → mezcla. **`ValueError`**.
- `interval=[5, 95]` → todos en `(1, 100]` → legacy, `/100`, warning.
- `interval=[0.05, 0.95]` → todos en `[0, 1]` → nueva API.

> El valor exacto `1` se trata siempre como quantil (extremo superior válido), nunca
> como percentil, salvo que aparezca junto a otro valor `> 1` (entonces es mezcla → error).

**Mensaje de error para mezcla:**
```
ValueError: `interval` mixes values <= 1 and > 1, so the scale is ambiguous.
Use quantiles in the [0, 1] range, e.g. `interval=[0.05, 0.95]`.
```

**Mensaje de `FutureWarning` (legacy):**
```
FutureWarning: Passing `interval` as percentiles (0-100) is deprecated.
Use quantiles (0-1) instead. For example, use `interval=[0.05, 0.95]`
instead of `interval=[5, 95]`. Percentile support will be removed in
skforecast X.X.X.
```

---

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

# coverage como float (sin cambio de API)
forecaster.predict_interval(
    steps=10,
    interval=0.90,           # coverage 0-1 → [0.05, 0.95]
    method='bootstrapping',
)
# → pred | lower_bound | upper_bound
```

**Reglas:**
- `interval` como `float` en `(0, 1)`: coverage. Se calcula `[0.5 - interval/2, 0.5 + interval/2]`.
- `interval` como `float == 1.0`: **`ValueError`** (cobertura 100% no produce intervalo finito
  con bootstrapping; ambiguo). Documentar.
- `interval` como `list`/`tuple` de 2 valores: aplica reglas de detección de escala arriba.
- Salida siempre `pred, lower_bound, upper_bound`.

**Decisión sobre el default (importante):**

El default actual `[5, 95]` equivale a **90% de cobertura**. Para **no cambiar el
comportamiento por defecto de forma silenciosa**, el nuevo default es `[0.05, 0.95]`
(mismo 90% de cobertura), **no** `[0.025, 0.975]` (95%).

> Cambiar el default a 95% movería el resultado de todos los usuarios que no pasan
> `interval`. Se mantiene 90% para preservar el comportamiento. Si en el futuro se
> quiere 95% como default, hacerlo en una major aparte y anunciarlo.

### Cambios en `predict_quantiles`

La API pública no cambia, solo la validación interna pasa por la función de
normalización compartida.

```python
forecaster.predict_quantiles(
    steps=10,
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],  # sin cambio, ya usa 0-1
)
# → q_0.1 | q_0.25 | q_0.5 | q_0.75 | q_0.9
```

**Nota de diseño (documentar):** `predict_quantiles([0.05, 0.95])` devuelve
`q_0.05, q_0.95`, mientras que `predict_interval([0.05, 0.95])` devuelve
`lower_bound, upper_bound`. Es intencional: `predict_interval` es la interfaz de
intervalo simétrico de confianza; `predict_quantiles` es la interfaz de N quantiles
arbitrarios. Aclararlo en docstrings para evitar confusión.

### Cambios en `backtesting_forecaster`

**Antes (v0.22):**
```python
backtesting_forecaster(..., interval=[5, 95])      # → pred | lower_bound | upper_bound
backtesting_forecaster(..., interval=[10, 50, 90]) # → pred | p_10 | p_50 | p_90
```

**Después:**
```python
backtesting_forecaster(..., interval=[0.05, 0.95]) # → pred | lower_bound | upper_bound
backtesting_forecaster(..., interval=[0.1, 0.5, 0.9]) # → pred | q_0.1 | q_0.5 | q_0.9
```

**Reglas internas de backtesting:**
- `interval` como `float`: coverage 0–1 → 2 quantiles → `predict_interval`.
- `interval` como `list` de 2 valores: `predict_interval` directamente.
- `interval` como `list` de N > 2 valores: `predict_quantiles` directamente.
- Detección de escala y mezcla: misma función compartida (`FutureWarning` legacy / `ValueError` mezcla).
- `interval` como `str` (`'bootstrapping'`): sin cambio.
- `interval` como objeto `distribution`: sin cambio.

### Naming de columnas unificado

| Caso | Columnas |
|------|----------|
| 2 quantiles (intervalo) | `pred`, `lower_bound`, `upper_bound` |
| N quantiles | `q_0.1`, `q_0.25`, `q_0.5`, `q_0.75`, `q_0.9` |

Se elimina el prefijo `p_`. Todo usa `q_` con valores en escala 0–1.

> **Advertencia de migración:** cualquier consumidor que parsee columnas `p_*`
> (métricas, plots, calibradores) debe actualizarse. Ver sección "Consumidores de `p_`".

---

## Cambios en `check_interval`

```python
# Después
check_interval(interval=[0.05, 0.95])     # valida 0-1 (legacy >1 → warning, mezcla → error)
check_interval(quantiles=[0.05, 0.95])    # valida 0-1 (sin cambio)
check_interval(alpha=0.95)                # valida 0-1 (sin cambio)
```

Delegar la rama `interval=` a `_normalize_interval_scale` para detección legacy/mezcla.

---

## Consumidores de columnas `p_` (auditar antes de tocar)

Antes de cambiar el naming hay que localizar y actualizar todo lo que dependa del
prefijo `p_`:

- Funciones de métricas que reciben DataFrames de backtesting con bounds.
- `crps_from_predictions` / `crps_from_quantiles` y utilidades de cobertura.
- `ConformalIntervalCalibrator` (preprocessing) — verificar qué escala/columnas usa.
- Funciones de plotting: `plot_prediction_intervals`, `plot_prediction_distribution`.
- Cualquier test o ejemplo que haga `df['p_5']` / regex `p_\d+`.

Acción: `grep` de `p_` y `lower_bound|upper_bound` en todo el repo y documentar la lista
real antes de empezar la migración.

---

## Alcance completo de forecasters

### Forecasters (predict_interval, predict_quantiles)
- `skforecast/recursive/_forecaster_recursive.py`
- `skforecast/recursive/_forecaster_recursive_multiseries.py`
- `skforecast/recursive/_forecaster_recursive_classifier.py`
- `skforecast/direct/_forecaster_direct.py`
- `skforecast/direct/_forecaster_direct_multivariate.py`
- `skforecast/deep_learning/_forecaster_rnn.py`
- **`skforecast/foundation/_forecaster_foundation.py`** ← añadido. Tiene
  `predict_interval` / `predict_quantiles` nativos; debe usar la misma escala 0–1.

### Validación / utilidades compartidas
- `skforecast/utils/utils.py` → `check_interval`, nueva `_normalize_interval_scale`.

### Preprocessing
- `skforecast/preprocessing/...` → `ConformalIntervalCalibrator` (revisar escala/columnas).

### Model selection (backtesting)
- `skforecast/model_selection/_validation.py` → `_fit_predict_forecaster`,
  `_fit_predict_forecaster_multiseries`, `_predict_and_calculate_metrics_one_step_ahead`.
- **`backtesting_foundation`** ← añadido. Debe enrutar igual que `backtesting_forecaster`.

---

## Plan de transición

### Versión N (próxima release)

1. **`predict_interval`**: acepta ambas escalas vía `_normalize_interval_scale`.
   - Todos los valores en `[0, 1]` → nueva API.
   - Todos en `(1, 100]` → legacy + `FutureWarning`.
   - Mezcla → `ValueError`.
   - Default `[0.05, 0.95]` (mantiene 90% de cobertura, **sin** cambio silencioso).
2. **`backtesting_forecaster` / `backtesting_foundation`**: misma lógica + columnas `q_`.
3. **`check_interval`**: delega en `_normalize_interval_scale`.
4. **`predict_quantiles`**: API pública sin cambios.
5. **Foundation forecasters**: alinear a escala 0–1.

### Versión N+2

- Se elimina soporte para percentiles 0–100.
- `_normalize_interval_scale` deja de aceptar `> 1` (solo `ValueError`).
- Se eliminan los `FutureWarning`.

---

## Tests

### Nuevos / actualizados
- `**/tests/test_predict_interval.py`
- `**/tests/test_predict_quantiles.py`
- `**/tests/test_check_interval.py`
- `**/tests/test_backtesting_forecaster*.py`
- Tests de Foundation (`predict_interval`, `predict_quantiles`, `backtesting_foundation`).

### Guards de regresión (obligatorios)
- **Equivalencia legacy**: `predict_interval([5, 95])` (con warning) debe dar
  **exactamente los mismos números** que la versión anterior. Comparar contra valores
  fijos. Lo mismo para backtesting.
- **Equivalencia de escalas**: `predict_interval([5, 95])` == `predict_interval([0.05, 0.95])`.
- **Default sin cambio**: `predict_interval()` produce 90% de cobertura igual que antes.
- **Casos límite**: `[1, 50]`, `[0.5, 95]`, `[0, 1]`, `interval=1.0` → comportamiento
  definido (error o quantil) verificado con `pytest.raises`.
- **`FutureWarning`**: verificado con `pytest.warns(FutureWarning)` en cada camino.

---

## Documentación

- `llms-base.txt` / `llms.txt` y `docs/llms-full.txt`.
- User guides de prediction intervals (ejemplos con nueva escala 0–1).
- Docstrings con tags `versionchanged` (convención del repo) en cada método tocado.
- Nota de migración en `changelog.md` / release notes: cambio de escala, naming `p_`→`q_`,
  y aclaración de que el default mantiene 90% de cobertura.

---

## Resumen de cambios respecto al plan v1

1. **Detección de mezcla → `ValueError`** explícito (antes indefinido).
2. **`interval=1.0`** definido como `ValueError` (antes ambiguo).
3. **Default `[0.05, 0.95]`** (90%, sin cambio silencioso) en vez de `[0.025, 0.975]` (95%).
4. **Foundation** (`ForecasterFoundation`, `backtesting_foundation`) incluido en el alcance.
5. **`ConformalIntervalCalibrator`** y consumidores de `p_` auditados antes de tocar naming.
6. **Función compartida** `_normalize_interval_scale` para garantizar comportamiento idéntico
   en todos los caminos.
7. **Guards de regresión** explícitos (equivalencia legacy, equivalencia de escalas,
   default, casos límite, warnings).
8. Aclaración documentada de por qué `predict_interval` y `predict_quantiles` difieren en
   nombres de columnas para 2 quantiles.

---

## Resumen visual

```
                    ANTES                                    DESPUÉS

predict_interval    [5, 95] (0-100)      →    [0.05, 0.95] (0-1)
                    → pred, lower, upper        → pred, lower, upper (sin cambio)
                    default 90%                 default 90% (sin cambio de cobertura)

predict_quantiles   [0.05, 0.5, 0.95]    →    [0.05, 0.5, 0.95] (sin cambio)
                    → q_0.05, q_0.5, q_0.95    → q_0.05, q_0.5, q_0.95

backtesting         [5, 95] (0-100)      →    [0.05, 0.95] (0-1)
                    → pred, lower, upper        → pred, lower, upper
                    [10, 50, 90] (0-100)  →    [0.1, 0.5, 0.9] (0-1)
                    → pred, p_10, p_50, p_90    → pred, q_0.1, q_0.5, q_0.9

escala mezcla       (indefinido)         →    ValueError
interval=1.0        (indefinido)         →    ValueError
```

**Resultado final:** una sola escala (0–1), dos métodos con identidad clara, naming
consistente, alcance completo (incluido Foundation), y transición sin ruptura ni cambios
silenciosos de comportamiento.
