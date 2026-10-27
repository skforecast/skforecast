# BUG-08: `freeze_params=True` no congela nada en `backtesting_stats` ni en las búsquedas stats

**Veredicto: keep (reproducido).** Esfuerzo S.

## Reproducción

Script: `dev/roadmap_0.27.0/repro/BUG-08.py` (serie diaria sintética de 800 puntos, m=7,
`TimeSeriesFold(steps=15, initial_train_size=500, refit=True)`, 20 folds). Cuenta las
llamadas a `auto_arima` / `auto_ets` envolviéndolas, mide el tiempo y compara la métrica.
Entorno: Python 3.12.3, skforecast 0.26.0, numpy 2.5.4, pandas 2.3.3, numba 0.68.0.

| Caso | Búsquedas automáticas | Tiempo | MAE |
|---|---|---|---|
| Arima auto, `freeze_params=True` (actual) | 20 (esperado 1) | 44.8 s | 1.559485 |
| Arima auto, `freeze_params=False` (actual) | 20 | 43.4 s | 1.559485 |
| Arima auto, `freeze_params=True` (con arreglo) | 1 | 8.5 s (5.3x) | 1.532520 |
| Ets auto, `freeze_params=True` (actual) | 20 (esperado 1) | 13.1 s | 1.526711 |
| Ets auto, `freeze_params=False` (actual) | 20 | 11.4 s | 1.526711 |
| Ets auto, `freeze_params=True` (con arreglo) | 1 | 0.9 s (15x) | 1.536584 |
| `grid_search_stats`, Arima auto, 2 candidatos | 40 (esperado 2) | 76.4 s | |

Hoy `True` y `False` dan predicciones idénticas. Con `False` la columna `estimator_params`
muestra 15 órdenes distintos en 20 folds (de `AutoArima(1,0,1)(0,1,1)[7]` a
`AutoArima(3,0,3)(2,1,2)[7]`): con `True` el usuario cree que el orden está fijo y no lo está.
El primer fold coincide con y sin arreglo (solo cambian los reajustes posteriores).

## Causa raíz

- `skforecast/model_selection/_validation.py:2097-2100`: tras el primer ajuste se congela
  `best_params_` en `forecaster.estimators_` (los clones ajustados).
- `skforecast/model_selection/_validation.py:1847`: cada fold con reajuste llama a
  `forecaster.fit`.
- `skforecast/recursive/_forecaster_stats.py:521`: `fit` reconstruye
  `self.estimators_ = [clone(est) for est in self.estimators]` desde las plantillas sin
  ajustar, que siguen en modo automático. Lo congelado se descarta en el fold 1.
- `skforecast/model_selection/_search.py:2779`: `grid_search_stats` y `random_search_stats`
  fuerzan `freeze_params=True`, así que cada candidato repite la búsqueda en todos los folds.
- Presente desde que existe `freeze_params` (0.20.0, entonces con `copy(est)` en lugar de
  `clone`, mismo efecto). Los tests no lo detectan: el de deriva
  (`test_backtesting_stats.py:1450`) exige precisamente que `True` y `False` coincidan.

## ¿Lo sufre un usuario típico?

Sí. `freeze_params=True` es el valor por defecto y el modo "rápido" documentado
(docstring de `backtesting_stats`, guía `forecasting-statistical-models.ipynb`,
skill `statistical-models`). Cualquiera que haga backtesting con `refit=True` o una búsqueda
con `Arima(order=None)` o `Ets()` (modo automático por defecto de Ets) paga el coste
completo de `freeze_params=False`: 5x más lento con Arima y 15x con Ets en este caso, y 20x
llamadas de más en `grid_search_stats`. No es una fuga ni una métrica errónea: el backtest
resultante es válido para la estrategia "reseleccionar en cada fold", pero no es lo que
promete el parámetro. Encaja en la prioridad de STATS (corrección y rendimiento).

## Arreglo propuesto

En `_backtesting_stats`, justo después del primer `fit` y antes de `Parallel` (para que
también funcione con `n_jobs > 1`), propagar los parámetros a las plantillas de la copia
(ya es una copia de `deepcopy_forecaster`, comprobado que no comparte plantillas con el
objeto del usuario):

```python
if freeze_params and refit:
    for estimator, estimator_ in zip(forecaster.estimators, forecaster.estimators_):
        if getattr(estimator_, "best_params_", None):
            estimator._set_params(**estimator_.best_params_)
            estimator_._set_params(**estimator_.best_params_)
```

Tests: contar llamadas a `auto_arima` / `auto_ets` (1 con `True`, n folds con `False`) y
que con `True` el orden sea el mismo en todos los folds. Hay que revisar (y
probablemente regenerar) los valores esperados de los tests con estimadores automáticos y
`refit=True` (`test_output_backtesting_stats_multiple_estimators_refit_True_with_mocked`
usa `Ets()`; `test_output_backtesting_stats_auto_arima_with_drift_and_exog`) y los de
`grid_search_stats` / `random_search_stats` con modo automático. Nota de versión en
"Fixed": cambia los resultados de quien usa el valor por defecto con modelos automáticos.
