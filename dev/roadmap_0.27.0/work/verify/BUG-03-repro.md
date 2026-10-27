# BUG-03: verificación

**Veredicto: keep (esfuerzo S).** Se reproduce en todas las búsquedas probadas. El orden de `results` y la dirección de Optuna dependen solo del tipo de tarea (regresión minimiza, clasificación maximiza), nunca de la métrica. Con una métrica "mayor es mejor" (`r2_score`) la fila 0 es el peor modelo y `return_best=True` reentrena el forecaster del usuario con él, sin ningún aviso. Hay además un caso más grave que la candidata no recogía (caso C): aunque el usuario fije `kwargs_create_study={'direction': 'maximize'}`, como indica el docstring, Optuna optimiza bien pero la tabla se ordena ascendente y `return_best` reentrena con el peor trial explorado.

Script: `dev/roadmap_0.27.0/repro/BUG-03.py` (h2o, `Ridge` con `lags=12`, `TimeSeriesFold(steps=12, initial_train_size=len(y)-60)`; skforecast 0.26.0, sklearn 1.9.1, optuna 5.0.0, pandas 2.3.3).

## Resultados medidos

| Caso | Observado | Esperado |
|---|---|---|
| A. `grid_search_forecaster`, `metric=[r2_score, 'mean_absolute_error']` | fila 0 `alpha=100` (R2 -0.806, MAE 0.230); `forecaster.estimator.alpha = 100` | `alpha=0.01` (R2 0.857, también el menor MAE 0.066) |
| B. `bayesian_search_forecaster`, `r2_score`, dirección por defecto | estudio `MINIMIZE`; explora alpha ~800-980 (R2 ~ -1.09); reentrena con alpha 978.8 | maximizar R2 (0.857 con alpha ~0.05) |
| C. Igual que B con `kwargs_create_study={'direction': 'maximize'}` | `study.best_value = 0.857` (alpha 0.048), pero fila 0 alpha 766.6 (R2 -1.084) y se reentrena con alpha 766.6 | reentrenar con `study.best_params` |
| D. `grid_search_forecaster_multiseries`, `r2_score` | reentrena con `alpha=100` (R2 ponderado -0.113) | `alpha=0.01` (0.849) |
| E. `ForecasterRecursiveClassifier`, `metric=[zero_one_loss, 'accuracy_score']` | reentrena con `C=0.0001` (pérdida 0.333) | `C=1.0` (pérdida 0.317) |
| Control: `backtesting_forecaster` con `r2_score` | 0.857 / 0.819 / -0.806 | los valores son correctos; solo falla la ordenación |

Corrección a la candidata: el triaje daba R2 -0.369 para `alpha=100`; con este `cv` sale -0.806. La conclusión no cambia.

## ¿Lo encuentra un usuario típico?

- Las métricas en texto son coherentes (regresión: todas pérdidas; clasificación: todas puntuaciones, `skforecast/metrics/metrics.py:45-62`), así que el fallo solo aparece con callables. Pero pasar funciones de `sklearn.metrics` es el uso documentado y `r2_score` es de las métricas más usadas por quien viene de scikit-learn.
- Nada avisa: el docstring de `metric` (`_search.py:85-93`) dice "goodness of fit" y "returns a float", sin dirección; la guía de tuning con métrica personalizada tampoco lo indica.
- Resultado silencioso erróneo en un flujo común (búsqueda con `return_best=True`, valor por defecto): el forecaster final queda entrenado con el peor modelo. El caso C no tiene más alternativa que leer `study.best_params` a mano. No hay issues abiertos al respecto (búsqueda en GitHub sin resultados relevantes).

## Causa raíz

- Ordenación por tarea: `skforecast/model_selection/_search.py:358,514` (grid/random de un solo modelo), `:1529,1537` (grid/random multiserie), `:677,916` (bayesiana), `:1950,2049` (bayesiana multiserie), `:2362,2427` (bayesiana foundation); siempre ascendente en `:2817` (stats) y `:3077` (equivalent date).
- Dirección de Optuna por tarea: `:845-846`, `:1952-1953`, `:2368-2369`. La ordenación posterior no lee `study.direction`, de ahí el caso C.

## Arreglo propuesto

1. Arreglo inmediato sin cambio de API (caso C): en las tres búsquedas bayesianas ordenar con `ascending = study.direction == optuna.study.StudyDirection.MINIMIZE` en lugar de `is_regression`.
2. Dirección deducida de la métrica: helper en `model_selection/_utils.py`, p. ej. `_infer_metric_direction(metric, forecaster_task)`, que use la convención de scikit-learn sobre el nombre de la primera métrica (`*_score` maximizar; `*_error`, `*_loss`, `*_deviance` minimizar) y, si el nombre no encaja (métrica propia), caiga al valor por defecto de la tarea. Opcionalmente respetar un atributo `greater_is_better` en el callable. Usarlo en las 9 ordenaciones y como `direction` por defecto de Optuna (solo si el usuario no la fija).
3. Docstrings de `metric` y `kwargs_create_study` (y skills `hyperparameter-optimization` y `metric-selection`): explicar cómo se decide la dirección.
4. Tests: `r2_score` en grid, random, bayesiana (con y sin `direction`) y multiserie; `zero_one_loss` con el clasificador; nota en `releases.md` (Fixed).

Alternativa mínima si no se quiere deducir la dirección: solo el punto 1 más un aviso cuando el nombre de la métrica termine en `_score` en regresión (o en `_loss`/`_error` en clasificación). Atajo actual para usuarios: pasar la métrica negada (`lambda`/función que devuelva `-r2_score(...)`).
