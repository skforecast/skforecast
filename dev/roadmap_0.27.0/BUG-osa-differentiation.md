# BUG-osa-differentiation: con `differentiation`, el OSA no evalúa predicciones a un paso

Fila del roadmap: ninguna todavía. Detectado en la revisión de la PR #1471 (MASE por origen),
pero no tiene relación con ella: ocurre igual en `0.27.x`.

Diagnóstico (2026-11-09) en local, en el worktree de `fix/mase-scale-per-origin` (#1471) y en
un checkout temporal de `origin/0.27.x` (`ecbe7756a`). Implementado en la misma rama que #1471
(sección 8), por decisión del usuario. Scripts:

- `repro/BUG-osa-differentiation_00_fix_prototype.py`: prototipo del arreglo (sustituye el
  método en tiempo de ejecución, solo cuando lo llaman las dos funciones del OSA).
- `_01_scope.py`: alcance por forecaster, orden de diferenciación, transformador y métrica.
- `_02_search.py`: candidato elegido en búsquedas con datasets de skforecast.
- `_03_pinned_tests.py`: tests que fijan el comportamiento actual (los de `model_selection`
  con el prototipo activado).

Python 3.14.3, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1, lightgbm 4.6.0, scipy 1.18.1.

## 0. Veredicto

**Bug, desde 0.14.0** (la versión que introdujo `OneStepAheadFold`). Con `differentiation`,
las métricas del OSA no se calculan sobre predicciones a un paso, sino sobre una trayectoria
que acumula los errores de todas las predicciones anteriores del test.

- **Causa confirmada en el código** (sección 1): `inverse_transform_next_window` suma las
  diferencias predichas desde el último valor de entrenamiento.
- **Afecta a los cuatro forecasters con `differentiation`**: `ForecasterRecursive`,
  `ForecasterDirect`, `ForecasterRecursiveMultiSeries` y `ForecasterDirectMultiVariate`.
  `ForecasterRecursiveClassifier` no admite `differentiation` (el atributo es siempre
  `None`), así que no le afecta.
- **Afecta a todas las métricas** (MAE, MSE, MAPE, MASE, RMSSE y las propias), con y sin
  transformador. Las métricas salen infladas: entre 1.2 y 9 veces con `differentiation=1`,
  y entre 6 y 5,800 veces con `differentiation=2` (sección 3.1).
- **Cambia el candidato elegido** en las 8 búsquedas medidas, con 4 datasets de skforecast,
  Ridge y LightGBM. El OSA actual no elige nunca el mejor candidato al horizonte real; el
  corregido, en 5 de 8 (sección 3.3). El OSA actual elige siempre candidatos con pocos lags.
- **Con huecos en el test**, también `y_true` sale mal: la suma acumulada se salta las filas
  quitadas (sección 3.2).
- **Los tests fijan el comportamiento incorrecto**: 9 tests de `tests_utils` tienen valores
  esperados calculados con la suma acumulada. Ningún test de equivalencia OSA = backtesting
  usa `differentiation`, y por eso no se detectó (sección 4).

La hipótesis de la revisión es correcta, con un matiz: la predicción a un paso es "el valor
observado anterior más la diferencia predicha" solo para `differentiation=1`. En general es
`nivel_observado_t − diferencia_observada_t + diferencia_predicha_t` (sección 5).

## 1. Causa

`TimeSeriesDifferentiator.inverse_transform_next_window`
(`skforecast/preprocessing/_preprocessing.py:323-361`) da por hecho que `X` es una ventana
futura que empieza justo después de la serie con la que se ajustó, y la reconstruye con sumas
acumuladas:

```python
X_undiff = np.cumsum(X, axis=0, dtype=float) + self.last_values[-1]          # orden 1
X_undiff = np.cumsum(X_undiff, axis=0, dtype=float) + self.last_values[-(i + 1)]  # orden > 1
```

Es lo correcto para `predict`, donde cada paso parte de la predicción anterior. En el OSA no:

- **Ajuste del diferenciador.** `_train_test_split_one_step_ahead`
  (`recursive/_forecaster_recursive.py:1164-1180`, igual en los demás) llama a
  `_create_train_X_y` con `y.iloc[:initial_train_size]`, que ajusta `self.differentiator`
  (`last_values` = último valor de entrenamiento y, con orden 2, última diferencia). Para el
  test usa una **copia** del diferenciador (`_create_train_X_y`, líneas 836-841), que no se
  guarda.
- **`y_true`.** Es la suma acumulada de las diferencias observadas desde el último valor de
  entrenamiento. Da los valores observados, pero solo si las filas del test son consecutivas
  (sección 3.2).
- **`y_pred`.** Es la suma acumulada de las diferencias **predichas**. El nivel predicho en
  `t` es `y_{n} + Σ_{k≤t} d̂_k`, así que el error en `t` es `Σ_{k≤t} (d_k − d̂_k)`: un paseo
  aleatorio de los errores. Con orden 2 se acumula dos veces.

Los tres sitios:

| Función | `0.27.x` | Rama de #1471 |
|---|---|---|
| `_calculate_metrics_one_step_ahead` (una serie, Recursive y Direct) | `_utils.py:1153-1154` | `_utils.py:1164-1165` |
| `_predict_and_calculate_metrics_one_step_ahead_multiseries`, rama `ForecasterDirectMultiVariate` | `_utils.py:2390-2391` | `_utils.py:2591-2592` |
| Misma función, rama `ForecasterRecursiveMultiSeries` | `_utils.py:2525-2530` | `_utils.py:2701-2706` |

Las predicciones que devuelve la función multiserie con `return_predictions=True` también
son las acumuladas, pero las búsquedas la llaman con `return_predictions=False`
(`_search.py:1782` y `:2318`): solo las usan los tests.

## 2. Historia

- La inversión con `inverse_transform_next_window` entró con el OSA: `147227776`
  (2024-10-11, multiserie) y `3c51e04ae` (2024-11-07, una serie), publicados en **0.14.0**.
- Los tests de equivalencia OSA = backtesting con `steps=1` (`test_grid_search_forecaster.py:116`,
  `test_predict_and_calculate_metrics_one_step_ahead_multiseries.py:281`, `:739`, `:830`,
  `:915`, `:1027`) no usan `differentiation`. Los tests que sí la usan comparan con valores
  fijos, calculados con el propio código.
- **La diferencia se conocía, pero se tomó por una limitación y no por un bug.** Cinco
  docstrings de tests de equivalencia lo decían: "Results are not equivalent if
  differentiation is included" (dos en `test_evaluate_grid_hyperparameters_multiseries.py` y
  tres en `test_predict_and_calculate_metrics_one_step_ahead_multiseries.py`).

## 3. Experimentos

### 3.1 Alcance (`_01_scope.py`)

Referencia: un backtesting con `TimeSeriesFold(steps=1, refit=False)`, que predice cada punto
a un paso desde los valores observados con el mismo estimador ajustado. Sin `differentiation`,
OSA y backtesting son iguales (los tests de equivalencia).

- **Datos.** h2o (una serie, Ridge, `lags=12`, `initial_train_size=144`, 60 puntos de test) e
  items_sales (tres series, Ridge, `lags=14`, `initial_train_size=912`, 185 puntos por serie;
  `DirectMultiVariate` con `level='item_1'`).
- **Columnas.** `now`: OSA actual. `fix`: OSA con el prototipo. `now/bt`: cociente entre el
  OSA actual y el backtesting. `fix == bt`: las 5 métricas (MAE, MSE, MAPE, MASE, RMSSE) del
  prototipo coinciden con el backtesting (`rtol=1e-8`). Multiserie: `average` de las series.
- **Forecasters directos con `steps=1`.** Con `steps=3`, OSA y backtesting no usan las mismas
  filas ni siquiera sin `differentiation` (otro asunto, sección 6).

| Forecaster | diff | transformer | MAE now | MAE fix | MAE now/bt | MASE now | MASE fix | RMSSE now/bt | fix == bt |
|---|---|---|---|---|---|---|---|---|---|
| ForecasterRecursive | None | none | 0.0708 | 0.0708 | 1.00 | 0.7636 | 0.7636 | 1.00 | yes |
| ForecasterRecursive | None | StandardScaler | 0.0650 | 0.0650 | 1.00 | 0.6992 | 0.6992 | 1.00 | yes |
| ForecasterRecursive | None | log | 0.0723 | 0.0723 | 1.00 | 0.7784 | 0.7784 | 1.00 | yes |
| ForecasterRecursive | 1 | none | 0.0921 | 0.0768 | 1.20 | 0.9897 | 0.8224 | 1.12 | yes |
| ForecasterRecursive | 1 | StandardScaler | 0.2496 | 0.0663 | 3.77 | 2.6371 | 0.7076 | 3.34 | yes |
| ForecasterRecursive | 1 | log | 0.1429 | 0.0744 | 1.92 | 1.5172 | 0.7944 | 1.72 | yes |
| ForecasterRecursive | 2 | none | 3.0792 | 0.1169 | 26.33 | 32.4342 | 1.2506 | 23.22 | yes |
| ForecasterRecursive | 2 | StandardScaler | 0.9809 | 0.0821 | 11.95 | 10.2807 | 0.8759 | 11.47 | yes |
| ForecasterRecursive | 2 | log | 198.9316 | 0.1182 | 1683.32 | 2050.1486 | 1.2598 | 2325.51 | yes |
| ForecasterDirect | 1, 2 | (3) | = Recursive | = Recursive | = Recursive | = Recursive | = Recursive | = Recursive | yes (9/9) |
| ForecasterRecursiveMultiSeries | None | (3) | 2.05-2.07 | igual | 1.00 | 0.78-0.80 | igual | 1.00 | yes |
| ForecasterRecursiveMultiSeries | 1 | none | 18.0236 | 2.0530 | 8.78 | 7.2169 | 0.7894 | 8.23 | yes |
| ForecasterRecursiveMultiSeries | 1 | StandardScaler | 17.0119 | 2.0823 | 8.17 | 6.8263 | 0.7829 | 7.85 | yes |
| ForecasterRecursiveMultiSeries | 1 | log | 11.2700 | 2.0893 | 5.39 | 5.0488 | 0.8064 | 5.51 | yes |
| ForecasterRecursiveMultiSeries | 2 | none | 209.6909 | 2.2386 | 93.67 | 82.3247 | 0.8598 | 82.99 | yes |
| ForecasterRecursiveMultiSeries | 2 | StandardScaler | 225.9202 | 2.2511 | 100.36 | 92.2364 | 0.8407 | 94.72 | yes |
| ForecasterRecursiveMultiSeries | 2 | log | 13.9683 | 2.3839 | 5.86 | 5.9600 | 0.9144 | 5.22 | yes |
| ForecasterDirectMultiVariate | None | (3) | 0.81-0.83 | igual | 1.00 | 0.53-0.54 | igual | 1.00 | yes |
| ForecasterDirectMultiVariate | 1 | none | 7.2891 | 0.7842 | 9.30 | 4.7807 | 0.5130 | 7.77 | yes |
| ForecasterDirectMultiVariate | 1 | StandardScaler | 7.1793 | 0.7839 | 9.16 | 4.7087 | 0.5129 | 7.66 | yes |
| ForecasterDirectMultiVariate | 1 | log | 4.2758 | 0.8064 | 5.30 | 2.8024 | 0.5274 | 4.56 | yes |
| ForecasterDirectMultiVariate | 2 | none | 168.1918 | 0.8325 | 202.04 | 110.3831 | 0.5446 | 175.61 | yes |
| ForecasterDirectMultiVariate | 2 | StandardScaler | 148.9644 | 0.8412 | 177.09 | 97.7615 | 0.5503 | 154.20 | yes |
| ForecasterDirectMultiVariate | 2 | log | 5915.4457 | 1.0146 | 5830.04 | 3907.7014 | 0.6637 | 9285.25 | yes |

(`ForecasterDirect` con `steps=1` y Ridge da exactamente lo mismo que `ForecasterRecursive`,
fila a fila.)

Lecturas:

- **El prototipo coincide con el backtesting en los 36 casos**, en las 5 métricas. Confirma
  la causa y la fórmula del arreglo.
- **El MASE con la escala por origen (#1471) está afectado igual que el MAE**: el numerador es
  el error acumulado. El MAPE, igual (cociente `now/bt` entre 1.19 y 6,323, no mostrado).
- **El daño depende del sesgo de las diferencias predichas**, que se acumula linealmente: con
  Ridge, `StandardScaler` cambia el efecto de la regularización y h2o pasa de 1.20 a 3.77.
  Con un transformador no lineal (log), el error acumulado se exponencia al deshacer el
  logaritmo.
- **Con `steps=3` en los directos** (segunda tabla del script), el efecto es el mismo:
  `ForecasterDirect` MAE ×1.11 (orden 1) y ×21 (orden 2); `ForecasterDirectMultiVariate`
  ×9.5 y ×204.
- **En `0.27.x`** el MAE de las tres columnas es idéntico en los 36 casos, y el prototipo
  también coincide con el backtesting en los 36. Solo cambia el valor del MASE, por la escala
  antigua (Recursive, orden 1, sin transformador: 1.0127 frente a 0.8444). El ejemplo de la
  revisión da lo mismo en las dos ramas (MAE 4.0758 en el OSA, 0.7770 en el backtesting).

### 3.2 Huecos en el test: también `y_true` sale mal

El OSA quita del test las filas con el objetivo ausente (y, según el estimador, las que
tienen lags ausentes), así que la suma acumulada se salta diferencias: a partir del primer
hueco, `y_true` deja de ser el valor observado. Reconstrucción a mano, igual que la función,
con 3 valores ausentes en el test y `HistGradientBoostingRegressor` (admite NaN en `X`):

| Caso | Filas de test | `y_true` reconstruido ≠ observado | Primer error |
|---|---|---|---|
| `ForecasterRecursive`, h2o, `lags=12`, NaN en 160, 161 y 180 | 55 de 60 | 39 | 2005-02-01 |
| `ForecasterRecursiveMultiSeries`, `item_1`, NaN en 950, 951 y 1000 | 152 de 185 | 114 (error máximo 9.03) | 2014-08-25 |

Por eso el arreglo debe tomar `y_true` de la serie observada, no de la suma acumulada. El
prototipo de `_00` no cubre este caso: usa la suma acumulada para el nivel observado.

### 3.3 Búsquedas: cambia el candidato (`_02_search.py`)

Grid search con `differentiation=1`, cada candidato evaluado de tres formas: `now` (OSA
actual), `fix` (OSA con el prototipo, igual al backtesting con `steps=1`) y `horizon`
(`TimeSeriesFold` con el horizonte del caso, `refit=False`, la evaluación que el OSA
aproxima). Rejillas: Ridge `alpha` ∈ {0.01, 1, 100}; LightGBM (`n_estimators=200`)
`max_depth` ∈ {3, 6} × `learning_rate` ∈ {0.03, 0.1}.

| Caso | Cand. | Elegido `now` | Elegido `fix` | Mejor al horizonte | τ now~horizon | τ fix~horizon | Pérdida al horizonte, `now` | Pérdida, `fix` |
|---|---|---|---|---|---|---|---|---|
| h2o, Ridge, h=12 | 18 | 13 lags, α=1 | 24 lags, α=0.01 | 36 lags, α=1 | 0.31 | 0.73 | 34.4% | 7.7% |
| h2o, LightGBM, h=12 | 16 | 24 lags, lr=0.1, d=6 | 12 lags, lr=0.1, d=6 | 12 lags, lr=0.1, d=6 | 0.48 | 0.57 | 19.7% | 0.0% |
| fuel Gasolinas, Ridge, h=12 | 18 | 24 lags, α=100 | 12 lags, α=100 | 36 lags, α=100 | -0.14 | 0.33 | 11.1% | 16.0% |
| fuel Gasolinas, LightGBM, h=12 | 16 | 24 lags, lr=0.1, d=3 | 24 lags, lr=0.03, d=6 | 12 lags, lr=0.03, d=6 | 0.78 | 0.70 | 6.4% | 22.4% |
| items_sales `item_1`, Ridge, h=7 | 15 | 7 lags, α=100 | 56 lags, α=100 | 56 lags, α=100 | -0.18 | 0.90 | 17.0% | 0.0% |
| bike_sharing `users` (120 días), Ridge, h=24 | 12 | 24 lags, α=100 | 168 lags, α=0.01 | 168 lags, α=0.01 | -1.00 | 0.73 | 19.6% | 0.0% |
| bike_sharing `users`, LightGBM, h=24 | 12 | 24 lags, lr=0.03, d=3 | 168 lags, lr=0.03, d=6 | 168 lags, lr=0.03, d=3 | 0.12 | 0.15 | 20.8% | 25.4% |
| items_sales, 3 series, Ridge, h=7 | 12 | 7 lags, α=100 | 28 lags, α=0.01 | 28 lags, α=0.01 | -0.94 | 0.94 | 8.6% | 0.0% |

"Pérdida al horizonte": MAE al horizonte del candidato elegido, respecto al mejor candidato al
horizonte.

- **El candidato cambia en los 8 casos.** Kendall τ entre los rankings `now` y `fix`: de
  −1.00 a 0.72 (mediana −0.15).
- **El OSA actual no elige nunca el mejor candidato al horizonte; el corregido, en 5 de 8.**
  Pérdida media al horizonte: 17.2 % con `now` y 8.9 % con `fix` (medianas 18.3 % y 3.9 %).
  τ medio con el horizonte: −0.07 con `now` y 0.63 con `fix`.
- **El OSA actual elige pocos lags**: 7, 13 o 24 en los 8 casos (el corregido, entre 12 y
  168), y con Ridge `alpha=100` en 4 de 5. Es coherente con el mecanismo: el candidato más
  suave y regularizado tiene diferencias predichas con menos sesgo acumulado, aunque no
  prediga mejor.
- **En 3 casos el candidato corregido es peor al horizonte** que el actual (fuel con Ridge y
  con LightGBM, bike con LightGBM). Es la limitación conocida del OSA, que mide a un paso, no
  un acierto del bug: el ranking corregido se parece más al del horizonte en 7 de 8 casos.
- **MASE y MAE eligen el mismo candidato** en todos los casos, antes y después: la escala por
  origen es la misma para todos los candidatos (#1471).

## 4. Tests que fijan el comportamiento actual (`_03_pinned_tests.py`)

Con el prototipo activado, `tests_utils` y `tests_search` de `model_selection` dan 725
pasados y **9 fallidos**, todos por valor (por ejemplo, en
`test_calculate_metrics_one_step_ahead_when_ForecasterRecursive` se esperan 0.5516, 1.2751 y
2.2776, y la predicción a un paso da 0.3217, 1.1613 y 1.3129):

- `test_calculate_metrics_one_step_ahead.py`:
  - `test_calculate_metrics_one_step_ahead_when_ForecasterRecursive`
  - `test_calculate_metrics_one_step_ahead_when_ForecasterRecursive_only_differentiation`
  - `test_calculate_metrics_one_step_ahead_when_metric_does_not_need_y_train`
- `test_predict_and_calculate_metrics_one_step_ahead_multiseries.py`:
  - `..._ForecasterRecursiveMultiSeries_diff` (2 casos: `1` y dict)
  - `..._output_ForecasterRecursiveMultiSeries_differentiation_dict` (2 casos)
  - `..._output_ForecasterDirectMultiVariate_differentiation` (2 casos)

Los tests de búsqueda que combinan OSA y `differentiation`
(`test_grid_search_forecaster_cv_differentiation_is_deprecated_and_ignored` y su versión
multiserie) comparan dos ejecuciones entre sí, no valores fijos, así que pasan con y sin el
arreglo. `test_output_evaluate_grid_hyperparameters_ForecasterRecursive_with_differentiation_mocked`
usa `TimeSeriesFold`.

## 5. Arreglo propuesto

**Fórmula.** Para cualquier orden `d`, el nivel en `t` es una combinación fija de los niveles
anteriores más la diferencia de orden `d` en `t`. A un paso, la predicción usa los niveles
anteriores observados, así que:

```
nivel_pred_t = nivel_obs_t − dif_obs_t + dif_pred_t     (en la escala transformada)
```

El error en la escala transformada es el error de la diferencia. Con `d = 1` es
`y_{t−1} + d̂_t`; con `d = 2`, `2·y_{t−1} − y_{t−2} + d̂²_t`. **No basta con invertir la
diferenciación con los valores observados anteriores mediante `inverse_transform_next_window`**:
ese método sigue acumulando dentro de la ventana que recibe. Hay que aplicar la fórmula.

**Dónde.** Un helper privado en `model_selection/_utils.py`, usado en los tres sitios de la
sección 1:

```python
def _one_step_ahead_levels(y_obs, diff_true, diff_pred, transformer):
    # y_obs: valores observados en la escala original (de `y` o `series[level]`
    # en las fechas del test). diff_true, diff_pred: diferencias en la escala
    # transformada (y_test y la salida del estimador).
    level_obs = transform_numpy(y_obs, transformer, fit=False)
    level_pred = level_obs - diff_true + diff_pred
    y_pred = transform_numpy(level_pred, transformer, inverse_transform=True)
    return y_obs, y_pred
```

- **`y_true` de la serie observada**, no de la suma acumulada (sección 3.2). En una serie
  hace falta el índice del test, que añade #1471 (`test_index` de
  `_train_test_split_one_step_ahead`). En el multiserie, `y_test` ya tiene el índice de
  fechas en las dos ramas.
- **Sin `differentiation`** no cambia nada: el camino actual (`inverse_transform` del
  transformador) se mantiene.
- **`transform_numpy`** (`utils/utils.py:2517`) con el transformador ya ajustado
  (`transformer_y`, o `transformer_series_[level]`). En el multiserie, cada nivel usa su
  diferenciador; los niveles con `differentiation=None` en el dict se dejan como están.

**Tests del arreglo:**

- **Equivalencia OSA = backtesting con `steps=1` y `refit=False`** con `differentiation` 1 y
  2, sin y con transformador, para los cuatro forecasters (directos con `steps=1`). Es el
  test que faltaba. Se puede parametrizar a partir de los tests de equivalencia existentes.
- **Huecos en el test:** `y_true` igual a los valores observados.
- **Los 9 tests de la sección 4:** recalcular los valores esperados y comprobar antes con un
  script que coinciden con el backtesting con `steps=1` (o con la fórmula a mano).

**Nota de la versión (Fixed):** "With `OneStepAheadFold` and `differentiation`, the
hyperparameter searches evaluated predictions that accumulated the errors of the previous
ones instead of one-step-ahead predictions. The metrics were inflated (several times with
`differentiation=1`) and the selected candidate could change." Más el enlace a la PR.
Probablemente también un highlight de tipo Fix y una línea en "Before upgrading", porque
cambian los resultados de las búsquedas sin cambiar el código.

**¿En esta PR o en otra?** (Propuesta inicial; al final se implementó en la rama de #1471,
sección 8.) En otra: es un bug independiente, anterior y con su propia nota de
versión. Pero **no desde `0.27.x` tal como está ahora**, sino encima de #1471 (desde
`fix/mase-scale-per-origin` mientras #1471 no se fusione, o desde la base actualizada cuando
se fusione):

- Toca exactamente los mismos bloques de `_calculate_metrics_one_step_ahead` y de la función
  multiserie que #1471, que les cambia la firma (`y` y `test_index`) y el código de justo
  después. Desde `0.27.x`, los conflictos serían seguros.
- Para tomar `y_true` de la serie observada en una serie necesita `test_index`, que añade
  #1471. Desde `0.27.x`, o se duplica ese cambio, o se usa la suma acumulada para `y_true`
  (`nivel_pred = inverse_transform_next_window(y_test) − y_test + y_pred`, una línea), que
  arregla el caso sin huecos pero deja mal el de la sección 3.2.

## 6. Fuera de alcance

- **Forecasters directos con `steps > 1`:** el OSA y el backtesting con `steps=1` difieren
  incluso sin `differentiation` (MAE de `ForecasterDirectMultiVariate`, `lags=14`, `steps=3`,
  items_sales: 0.7770 en el OSA frente a 0.8163 en el backtesting; con `steps=1`, 0.8157 en
  los dos). El ajuste del estimador del paso 1 con `steps=3` usa otras filas que el split del
  OSA. Merece su propio diagnóstico.
- La diferenciación en `predict`, en el backtesting y en `inverse_transform_training`: no se
  ha tocado y coincide con el prototipo (es la referencia).

## 7. Lo que no se ha comprobado

- El helper de la sección 5 con huecos: el prototipo usa la suma acumulada para el nivel
  observado, así que solo valida la fórmula sin huecos.
- Exógenas, `window_features` y `weight_func` con `differentiation` en el OSA. El mecanismo no
  depende de ellos, pero no se han medido.
- `bayesian_search_forecaster` y `random_search_forecaster` no se han ejecutado: llaman a las
  mismas funciones que `grid_search_forecaster` (`_calculate_metrics_one_step_ahead` y la
  multiserie).
- `ForecasterRecursiveMultiSeries` con series que empiezan después de `initial_train_size`
  y `differentiation`: el diferenciador de esa serie no se ajusta con datos de entrenamiento.
- Que el sesgo hacia los lags cortos de 3.3 sea general: son 8 búsquedas, no un estudio con
  semillas.

## 8. Implementación (2026-11-09)

En `fix/mase-scale-per-origin`, la rama de #1471, a petición del usuario.

**Diseño.** Helper privado `_revert_differentiation_one_step_ahead` en `model_selection/_utils.py`,
no un método de `TimeSeriesDifferentiator`:

- **No usa el estado del diferenciador.** La fórmula no depende de `last_values` ni del
  orden, así que un método no usaría `self`.
- **El trabajo real es de orquestación:** sacar los valores observados de la serie, usar el
  transformador ajustado y saltarse los niveles sin diferenciación. Es lo que ya hacen los
  demás helpers del OSA en ese módulo.
- **No amplía la API pública.**
- **Aviso en la clase:** el docstring de `inverse_transform_next_window` dice ahora que sirve
  para predicciones encadenadas, no para predicciones a un paso.

**Cambios:**

- **`_calculate_metrics_one_step_ahead`:** con `differentiation`, el helper toma `y_true` de
  `y` en `test_index` y calcula `y_pred` en la escala original. Sin ella, el camino del
  transformador no cambia.
- **`_predict_and_calculate_metrics_one_step_ahead_multiseries`:** lo mismo en las dos ramas.
  En la de `ForecasterRecursiveMultiSeries`, las posiciones de cada nivel en su serie se
  calculan una vez y se usan para la diferenciación y para las escalas, solo si hacen falta
  (con `differentiation` o con métricas con `y_train`).

**Tests:**

| Tipo | Detalle |
|---|---|
| Actualizados | Los 9 de la sección 4. Antes de sustituir los valores esperados, se comprobó que métricas y predicciones coinciden con `backtesting_forecaster` y `backtesting_forecaster_multiseries` con `steps=1` y `refit=False`. |
| Nuevo, unitario | `test_revert_differentiation_one_step_ahead.py`: orden 1 (el error de cada predicción es el de su diferencia), orden 2 (`2·y[t−1] − y[t−2] + d̂`) y transformador con órdenes 1 y 2. |
| Nuevo, huecos | `test_calculate_metrics_one_step_ahead_differentiation_missing_values_in_test`: `y_true` es el valor observado, y cada predicción es el valor observado anterior más la diferencia predicha. Falla con el código anterior. El backtesting no sirve de referencia aquí: lanza un error cuando falta el último valor antes de una fecha. |
| Equivalencia OSA = backtesting, ampliada con `differentiation` | Búsqueda de una serie: Recursive con orden 1 y `StandardScaler`, Recursive con orden 2, Direct con orden 1. Utils multiserie: MultiSeries con orden 1, dict `{1, None, 2}`, DirectMultiVariate con órdenes 1 y 2. Búsqueda multiserie: MultiSeries y DirectMultiVariate con orden 1. |

Se ha quitado de los cinco docstrings la nota "Results are not equivalent if differentiation
is included".

**Resultado:** `_01_scope.py` da `now == bt` en los 36 casos y con `steps=3` coincide con el
prototipo.

**Nota de la versión:** una entrada en Fixed (junto a las demás del OSA) y una línea en
"Before upgrading", ambas con el enlace a #1471. No se ha añadido un highlight (ya hay 9).

