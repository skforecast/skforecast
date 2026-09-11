# `crps_from_predictions`: analisis y detalles pendientes

## Conclusion

**No necesita la correccion aplicada a `crps_from_quantiles`.** El calculo es exacto, no
aproximado.

La diferencia esta en que `y_true` se incluye en la rejilla:

```python
grid = np.concatenate(([y_true], y_pred))
grid = np.sort(grid)
```

Con eso:

- Para `x < min(grid)`: `F(x) = 0` y ademas `x < y_true`, asi que el indicador tambien es 0.
  Integrando = 0.
- Para `x > max(grid)`: `F(x) = 1` y `x > y_true`, indicador 1. Integrando = 0.

Las colas valen exactamente cero, no hace falta el `tail_area` que si necesita
`crps_from_quantiles`. Y como la cdf empirica es una funcion escalon constante entre nodos
consecutivos, y el indicador tambien lo es (porque `y_true` es un nodo), la suma
`diffs * (cdf - indicator)**2` es la integral exacta, no una regla trapezoidal.

## Validacion realizada

Comparacion con la forma cerrada del CRPS empirico,

```
CRPS = (1/n) * sum_i |x_i - y|  -  (1/(2 n^2)) * sum_i sum_j |x_i - x_j|
```

sobre 20 000 casos con escalas entre 1e-3 y 1e4 y niveles entre -1000 y +1000:

```
error relativo mediano : 1.3e-16
error relativo maximo  : 1.6e-15
```

Propiedades comprobadas sobre otros 20 000 casos, 0 fallos en todas:

```
negativo                       0
no_invariante_traslacion       0
no_equivariante_escala         0
no_monotono_lejos_del_rango    0
no_es_mae_si_determinista      0
no_cero_si_perfecto            0
```

Los cuatro escenarios que rompian `crps_from_quantiles` se comportan bien:

```
traslacion   nivel -20 / 0 / 1000 / 10000   -> 0.6111111111 en los cuatro
negativos    [-1550, -1500, -1450], y=-1520 -> 17.777778   (positivo)
y_true=100 fuera del rango [8, 10, 12]      -> 89.1111     (crece lineal)
predicciones [0, 0, 0], y_true=500          -> 500.0000    (= |y - 0|)
```

## Detalles menores encontrados

### 1. Un array vacio devuelve 0.0 en lugar de lanzar un error

`len(y_pred) == 0` provoca `RuntimeWarning: invalid value encountered in divide` y luego
`np.sum` sobre un array vacio devuelve `0.0`, es decir, el mejor CRPS posible a partir de
cero predicciones. El resto de metricas del modulo (`mean_absolute_scaled_error`,
`root_mean_squared_scaled_error`, `symmetric_mean_absolute_percentage_error`,
`winkler_score`) si validan este caso.

Solucion propuesta, junto a las validaciones que ya existen:

```python
if len(y_pred) == 0:
    raise ValueError("`y_pred` must have at least one element.")
```

### 2. `NaN` se propaga en silencio

`crps_from_predictions(5.0, np.array([3.0, np.nan, 7.0]))` devuelve `nan`. En un backtesting
agregado eso contamina la media sin ningun aviso. Con `inf` devuelve `inf`.

Es discutible si conviene validarlo o dejar que se propague (es lo que hacen la mayoria de
metricas de scikit-learn). Como minimo deberia mencionarse en el docstring. Opcion:

```python
if np.isnan(y_pred).any():
    raise ValueError("`y_pred` must not contain NaN values.")
```

### 3. Devuelve `np.float64` en lugar de `float`

Inconsistente con `crps_from_quantiles`, que tras la correccion devuelve `float`, y con
`winkler_score` y `weighted_interval_score`, que hacen `float(...)` explicito. El type hint
declara `-> float`.

Solucion:

```python
return float(crps)
```

## Prioridad

Ninguno de los tres es un bug de calculo. Por orden:

| # | Detalle | Prioridad | Riesgo del cambio |
|---|---------|-----------|-------------------|
| 1 | Validar array vacio | Media | Ninguno, es un caso que hoy da un warning |
| 3 | `float()` en el retorno | Baja | Ninguno |
| 2 | Politica sobre `NaN` | Baja | Podria romper flujos que hoy toleran NaN |
