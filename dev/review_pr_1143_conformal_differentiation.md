# Review PR #1143: Conformal Prediction Interval Scaling with Differentiation

**PR**: https://github.com/skforecast/skforecast/pull/1143  
**Issue**: https://github.com/skforecast/skforecast/issues/1142  
**Autor**: @hritikkumarpradhan  
**Scope**: Este análisis se centra en el fix para `ForecasterRecursive`.

---

## 1. El Bug

### Código actual (`_forecaster_recursive.py`, líneas 2248–2254)

```python
lower_bound = predictions - correction_factor
upper_bound = predictions + correction_factor
predictions = np.column_stack([predictions, lower_bound, upper_bound])

if differentiator is not None:
    predictions = differentiator.inverse_transform_next_window(predictions)
```

### Qué hace `inverse_transform_next_window` (para d=1)

El método aplica `np.cumsum(X, axis=0) + last_value` a **cada columna** del array.
Cuando se le pasa la matriz `[pred, lower, upper]`:

```
En espacio diferenciado (step h):
    pred_h   = p_h
    lower_h  = p_h - c
    upper_h  = p_h + c

Después de cumsum:
    pred_undiff_h  = sum(p_1..p_h) + last_value
    lower_undiff_h = sum(p_1..p_h) - h*c + last_value
    upper_undiff_h = sum(p_1..p_h) + h*c + last_value

Width_h = upper_undiff_h - lower_undiff_h = 2*h*c
```

**El ancho del intervalo crece linealmente con h**, lo cual es incorrecto.

### Verificación numérica

Con `c = 0.5` y `preds_diff = [0.1, 0.2, 0.15, 0.12, 0.18]`:

| Step | Width (buggy) | Width esperado (d=1) |
|------|--------------|---------------------|
| 1    | 1.00         | 1.00                |
| 2    | 2.00         | 1.41                |
| 3    | 3.00         | 1.73                |
| 4    | 4.00         | 2.00                |
| 5    | 5.00         | 2.24                |

**El bug es real y significativo.** Los intervalos en el paso 5 son 2.2x más anchos de
lo que deberían.

---

## 2. El Fix Propuesto

### Cambio principal

```python
if differentiator is not None:
    predictions = differentiator.inverse_transform_next_window(predictions)

    import scipy.special
    d = self.differentiation
    steps_array = np.arange(1, len(predictions) + 1)
    scaling_factor = np.sqrt(
        np.cumsum(scipy.special.comb(steps_array + d - 2, d - 1)**2)
    )
    correction_factor = correction_factor * scaling_factor

lower_bound = predictions - correction_factor
upper_bound = predictions + correction_factor
predictions = np.column_stack([predictions, lower_bound, upper_bound])
```

### Lógica

1. Aplicar `inverse_transform_next_window` **solo** a las predicciones puntuales (1D).
2. Escalar `correction_factor` por el factor de la representación MA(∞) de $(1-B)^{-d}$.
3. Construir los bounds **después** del escalado.

---

## 3. Análisis Matemático

### 3.1 Representación MA(∞) de $(1-B)^{-d}$

Si los errores de predicción en el espacio diferenciado son i.i.d. con varianza
$\sigma^2$, la inversión de la diferenciación $(1-B)^{-d}$ tiene representación
MA(∞):

$$(1-B)^{-d} = \sum_{j=0}^{\infty} \psi_j B^j$$

donde los coeficientes son:

$$\psi_j = \binom{j + d - 1}{d - 1}$$

La varianza acumulada del error de predicción en el paso $h$ del espacio original es:

$$\text{Var}(\hat{e}_h) = \sigma^2 \sum_{j=0}^{h-1} \psi_j^2$$

Y el factor de escalado es:

$$s_h = \sqrt{\sum_{j=0}^{h-1} \psi_j^2}$$

### 3.2 Casos específicos

**Para d=1:**

$$\psi_j = \binom{j}{0} = 1 \quad \forall j$$
$$s_h = \sqrt{\sum_{j=0}^{h-1} 1} = \sqrt{h}$$

Verificado: `scaling = [1.0, 1.414, 1.732, 2.0, 2.236, 2.449]` = $\sqrt{h}$ ✓

**Para d=2:**

$$\psi_j = \binom{j+1}{1} = j + 1$$
$$s_h = \sqrt{\sum_{j=0}^{h-1} (j+1)^2} = \sqrt{\frac{h(h+1)(2h+1)}{6}}$$

Verificado: el código de la PR con `comb(steps_array + d - 2, d - 1)` produce
`[1, 2, 3, 4, 5, 6]` para d=2. ✓

### 3.3 Discrepancia en el índice de la fórmula

La PR usa:

```python
comb(steps_array + d - 2, d - 1)
```

con `steps_array = np.arange(1, h+1)`, es decir $j \in \{1, 2, ..., h\}$.

Sustituyendo: $\binom{j + d - 2}{d - 1}$ con $j$ empezando en 1.

Para $j=1$: $\binom{d-1}{d-1} = 1$ ✓  
Para $j=2$ (d=1): $\binom{1}{0} = 1$ ✓  
Para $j=2$ (d=2): $\binom{2}{1} = 2$ ✓  

La indexación es correcta: al empezar `steps_array` en 1 (en vez del 0 teórico),
la fórmula `comb(j + d - 2, d - 1)` produce los mismos coeficientes que
$\psi_{j-1} = \binom{j-1+d-1}{d-1}$. ✓

### 3.4 Supuesto clave: ¿son los errores i.i.d. en el espacio diferenciado?

**Este es el punto más importante del análisis.**

La justificación teórica de la PR asume que los errores de predicción en el espacio
diferenciado son i.i.d. Esto es **exacto** solo para un modelo naive (random walk)
y **aproximado** para modelos de regresión recursivos.

En un `ForecasterRecursive` con `differentiation=1`:

1. El modelo predice $\Delta y_{t+h}$ recursivamente.
2. La predicción en $t+h$ usa como input la predicción de $t+h-1$.
3. Los errores se propagan: un error en el paso $h-1$ afecta la predicción del paso $h$.
4. Esto crea correlación serial en los errores, violando el supuesto i.i.d.

Sin embargo:

| Método | Crecimiento del intervalo | Corrección teórica |
|--------|--------------------------|-------------------|
| Actual (bug) | Lineal: $2ch$ | Incorrecto en todos los casos |
| Fix propuesto | $2c\sqrt{h}$ | Exacto para random walk, aproximado para otros |
| Teórico exacto | Depende del modelo | Requeriría propagar errores por simulación |

**Conclusión:** El fix es una mejora sustancial sobre el bug actual. El crecimiento
$\sqrt{h}$ es la aproximación estándar usada en la literatura de series temporales
(e.g., Hyndman & Athanasopoulos, FPP3, §5.5) y es exacta para el caso base. No es
teóricamente perfecta para cualquier modelo recursivo, pero es la convención aceptada.

---

## 4. Comparación con Bootstrapping

El método de bootstrapping **NO tiene este bug** porque:

1. En `predict_bootstrapping`, los residuos se suman a las predicciones recursivamente
   dentro de `_recursive_predict_bootstrapping`, generando trayectorias completas en el
   espacio diferenciado.
2. `inverse_transform_next_window` se aplica a **cada trayectoria bootstrap completa**
   (una columna del array `boot_predictions`), **no** a bounds pre-calculados.
3. Los quantiles se calculan **después** de la transformación inversa, en el espacio
   original.

Esto significa que bootstrapping captura naturalmente la propagación de incertidumbre
a través de los pasos, sin necesidad de un factor de escalado explícito. El flujo es:

```
diferenciado → trayectorias con ruido → inverse_transform (cada trayectoria)
    → espacio original → quantiles → bounds
```

Mientras que conformal (con el fix) es:

```
diferenciado → predicción puntual → inverse_transform (solo preds)
    → espacio original → escalar correction_factor por sqrt(h)
    → bounds
```

---

## 5. Problemas de Implementación

### 5.1 PR basada en versión antigua de master (CRÍTICO)

La PR modifica las líneas usando `self.differentiation` y `self.differentiator`:

```python
if self.differentiation is not None:
    predictions = self.differentiator.inverse_transform_next_window(predictions)
    d = self.differentiation
```

Pero el master actual usa la variable local `differentiator` retornada por
`_create_predict_inputs()`:

```python
if differentiator is not None:
    predictions = differentiator.inverse_transform_next_window(predictions)
```

Esto fue refactorizado para evitar mutar el estado interno del forecaster durante
la predicción. **La PR necesita ser rebaseada sobre master actual y adaptada a esta
interfaz.**

### 5.2 `import scipy.special` dentro de la función (MENOR)

```python
import scipy.special
d = self.differentiation
```

El import se ejecuta en cada llamada a `_predict_interval_conformal`. Debería estar
a nivel de módulo con los demás imports. `scipy` ya es dependencia core
(`scipy>=1.12` en `pyproject.toml`), así que no hay problema de dependencias.

Nota: Python cachea los módulos importados en `sys.modules`, así que el coste real
de re-importar es mínimo (solo un lookup en dict), pero es mala práctica y
inconsistente con el resto del código.

### 5.3 Solo testea `differentiation=1` (IMPORTANTE)

El test actualizado solo verifica d=1:

```python
forecaster = ForecasterRecursive(
    estimator=LinearRegression(), lags=3,
    transformer_y=StandardScaler(), differentiation=1
)
```

La rama de d≥2 queda sin cobertura. Los coeficientes de la representación MA(∞)
cambian significativamente:

- d=1: crecimiento $\sqrt{h}$
- d=2: crecimiento $\approx h^{3/2} / \sqrt{3}$

Un error en la indexación de `comb()` solo se manifestaría con d>1. Falta al menos
un test con `differentiation=2`.

### 5.4 Código duplicado en 5 forecasters (MEJORA)

El mismo bloque de ~12 líneas se copia en:
- `_forecaster_recursive.py`
- `_forecaster_recursive_multiseries.py`
- `_forecaster_direct.py`
- `_forecaster_direct_multivariate.py`
- `_forecaster_rnn.py`

Sería más mantenible extraerlo a una función en `utils/`:

```python
def scale_conformal_correction_factor(
    correction_factor: np.ndarray,
    steps: int,
    differentiation_order: int
) -> np.ndarray:
    from scipy.special import comb
    steps_array = np.arange(1, steps + 1)
    scaling_factor = np.sqrt(
        np.cumsum(comb(steps_array + differentiation_order - 2,
                       differentiation_order - 1)**2)
    )
    return correction_factor * scaling_factor
```

### 5.5 Interacción con residuos binned (NOTA)

Cuando `use_binned_residuals=True`, `correction_factor` es un **vector** (un valor
por step, basado en el bin de la predicción en espacio diferenciado). Al multiplicar
por `scaling_factor`, se asume que cada `c_h` es independiente, lo cual es consistente
con la suposición i.i.d. del método conformal. No hay error aquí, pero es una
limitación inherente del enfoque.

### 5.6 Orden de operaciones: transform_y vs scaling (CORRECTO)

En el código actual (y en el fix), el flujo es:

1. Predicción en espacio diferenciado+transformado
2. Inverse differentiation
3. Inverse transform_y (e.g., StandardScaler)

El fix aplica el scaling entre los pasos 2 y 3. Esto es correcto porque:
- El `correction_factor` se calcula sobre residuos en espacio diferenciado+transformado
- El `scaling_factor` modela la acumulación de varianza por la inversión de diferencias
- La transformación inversa de `transform_y` se aplica uniformemente a pred y bounds

Si `transformer_y` es no lineal (e.g., log), los intervalos resultantes serán
asimétricos, lo cual es el comportamiento correcto.

---

## 6. Resumen

| Aspecto | Evaluación |
|---------|-----------|
| ¿El bug es real? | ✅ Sí, verificado analítica y numéricamente |
| ¿El diagnóstico es correcto? | ✅ Sí, el cumsum sobre bounds produce crecimiento lineal |
| ¿La fórmula matemática es correcta? | ✅ Sí, para el supuesto de errores i.i.d. |
| ¿El supuesto i.i.d. es razonable? | ⚠️ Es una aproximación estándar, no exacta |
| ¿Es mejor que el estado actual? | ✅ Significativamente mejor |
| ¿Necesita rebase sobre master? | ❌ Sí, usa API antigua |
| ¿Import a nivel de módulo? | ❌ Debería moverse |
| ¿Tests suficientes? | ❌ Falta d≥2 |
| ¿Código duplicado? | ⚠️ Mejorable con función auxiliar |

### Acción recomendada

El fix aborda un bug real con una solución matemáticamente sólida. Necesita los
siguientes cambios antes de merge:

1. **Rebasear sobre master actual** y usar `differentiator` (variable local) en vez
   de `self.differentiator` / `self.differentiation`.
2. **Mover `import scipy.special`** a los imports del módulo.
3. **Añadir test con `differentiation=2`** para cubrir la rama de d>1.
4. **Considerar extraer** la lógica de escalado a una función compartida.
