# FE-01: función de evaluación de predicciones de backtesting (lente riesgo y API)

**Veredicto: keep_modified.** Esfuerzo M solo en la versión reducida; tal como está propuesta
(recibir `y`/`series` en todos los formatos, métricas escaladas, WIS/CRPS y aviso de
infracobertura a posteriori) es L y tiene dos piezas que no se pueden hacer bien fuera del
backtesting.

## Qué existe hoy

- No existe nada parecido: ninguna función pública agrupa métricas por fold, paso o nivel
  (grep de `evaluate`, `summarize`, `by_step`, `step` en `skforecast/` sin resultados). Ninguna
  rama remota ni PR abierto lo toca (#1407 y QW-04 tocan el mismo código de `_validation.py`).
- Las piezas sí existen: `_get_metric` y `add_y_train_argument` (`metrics.py:29,88`),
  `calculate_coverage` (`metrics.py:459`, no ignora NaN), `crps_from_quantiles` (`:364`,
  escalar por observación), `winkler_score` (`:630`, necesita `alpha`),
  `weighted_interval_score` (`:732`).
- La guía `comparing-forecasting-models.ipynb` (celda 5) tiene `summarize` de ~40 líneas; su
  MASE usa `y_train = serie[:n_train]`, que no es lo que calcula el backtesting.

## Riesgos medidos (scripts en `scratchpad/fe01-risk/`)

1. **El nivel nominal se pierde en la salida.** `interval=0.8`, `[0.1, 0.9]`, `[0.05, 0.95]`,
   conformal y `backtesting_stats` con `alpha=0.2` devuelven las mismas columnas
   `['fold', 'pred', 'lower_bound', 'upper_bound']` (`_validation.py:290-293` renombra los dos
   cuantiles). Una función a posteriori no puede saber la cobertura nominal ni si se usaron
   residuos in-sample: **el aviso de infracobertura no se puede colocar en ella**. Debe ir en
   el backtesting junto con ISSUE-1409, donde se conocen `interval` y `use_in_sample_residuals`
   (las búsquedas pasan siempre `interval=None`, `_search.py:454,1454,2778,3033`, así que no les
   afecta).
2. **El paso no se deduce de la salida si hay `gap`.** `cumcount()+1` por fold es correcto con
   `gap=0`, `fold_stride` solapado, fold incompleto y multiserie con series que empiezan tarde o
   terminan antes, pero falla con `gap=3` (pasos reales 4..13) en Recursive y Direct; hace falta
   `+ gap`, que no está en la salida. Solución limpia: columna `step` en la salida del
   backtesting (DISC-1119), no reconstruirla en la función.
3. **Folds solapados: la métrica global no coincide.** El backtesting calcula la métrica solo con
   la última predicción de cada fecha (`keep='last'`, `_validation.py:663-666`, y en
   `:1489,2160,2644,2675`). Con `steps=10` y `fold_stride=1`: métrica del backtesting 0,214
   (solo filas de paso 1), MAE de todas las filas 0,714, paso 10 1,277. Con `fold_stride=5`:
   0,461 frente a 0,731. La función nueva no debe imitar ese filtro (mataría el análisis por
   paso), así que su total no coincidirá con el del backtesting: documentarlo.
4. **Métricas escaladas no reproducibles.** MASE con `refit=True`: backtesting 2,5234;
   `y_train=y[:200]` 2,5312; `y_train=y[7:200]` 2,5650. El backtesting construye `y_train` con la
   unión de los entrenamientos de los folds menos `window_size` (`_validation.py:672-684`,
   `_utils.py:1168-1190`); sin `cv` no se puede replicar. En una primera versión, rechazar
   métricas con `_needs_y_train` con un `ValueError` claro.
5. **Filas sin valor observado.** En multiserie ML, cuando la serie ya terminó, `pred` es NaN pero
   `lower_bound`/`upper_bound` no (medido: 8 de 8 filas), y `calculate_coverage` no ignora NaN.
   La función debe descartar filas con `y_true` o punto NaN antes de cualquier métrica.
6. **Formatos de salida heterogéneos.** Foundation con cuantiles no tiene `pred` (usa `q_0.5`,
   `_validation.py:2652,2684`); bootstrapping da `pred_boot_*`; `predict_dist` da parámetros
   (`loc`, `scale`); el clasificador da `*_proba`; `return_predictors=True` añade columnas.
7. **Coste de CRPS.** `crps_from_quantiles` en bucle: 30 µs por fila (1.500 filas en 0,05 s);
   un panel de 400.000 filas tardaría unos 12 s. Hacerlo opcional o vectorizarlo.

## Prototipo de la versión reducida

`scratchpad/fe01-risk/proto.py`: 35 líneas de lógica que operan solo sobre la salida con
`y_true` (QW-04) y `step`. Por nivel coincide exactamente con las métricas de
`backtesting_forecaster_multiseries` (sin solapamiento). Ejemplo por paso (gap=2, 5 series,
intervalo 80% in-sample conformal): MAE 0,46 en el paso 3 y 1,74 en el paso 9; cobertura de
0,41 a 0,09.

## API menos arriesgada

```python
from skforecast.model_selection import calculate_backtesting_metrics  # nombre a decidir

calculate_backtesting_metrics(
    predictions,                  # salida de cualquier backtesting_* (con y_true y step)
    metric='mean_absolute_error', # str, Callable o lista; sin métricas escaladas en v1
    by=['level', 'step'],         # None, 'fold', 'step', 'level' o columnas de predictions
    interval=None,                # opcional: nivel nominal, solo para Winkler y etiquetas
)
# Devuelve un DataFrame con una fila por grupo: métricas, n, y si hay columnas de intervalo
# coverage e interval_width; con q_* además crps (opcional por coste).
```

- Sin parámetro `y`/`series`: depende de QW-04 (`y_true`) y de la columna `step` (DISC-1119).
  Si QW-04 se rechaza, hay que aceptar `series` en ancho, largo y dict y alinear por nombre
  (ojo PR-1408, IDs enteros): vuelve a ser L.
- Sin aviso propio: el aviso de infracobertura va en el backtesting con ISSUE-1409.
- Total sin `by` calculado sobre todas las filas, documentando la diferencia con el backtesting
  cuando los folds se solapan.
- No interactúa con exog, window features, differentiation, transformers ni encoding: trabaja
  sobre predicciones ya en escala original.

## Orden recomendado

1. QW-04 + columna `step` en el mismo cambio de salida (un solo cambio anunciado, precedente de
   `fold` en 0.18.0).
2. ISSUE-1409: `coverage` e `interval_width` en las métricas del backtesting con dos cotas, y
   aviso si la cobertura queda más de 0,10 por debajo de la nominal (con un mínimo de filas,
   p. ej. 50: con n=100 la desviación típica de la cobertura al 80% es 0,04).
3. FE-01 reducida (M): función, 15 a 20 tests, docstring, `llms-base.txt`, skills
   (`backtesting-configuration`, `prediction-intervals`, `metric-selection`), sustituir
   `summarize` en la guía de comparación y nota de versión.

## Hallazgo lateral (no estaba en la lista)

Con `fold_stride < steps` la métrica del backtesting solo refleja los primeros `fold_stride`
pasos (con `fold_stride=1`, solo el paso 1). Está explicado en `backtesting.ipynb` (celda 24),
pero no en el docstring de `fold_stride` (`_split.py:40-47`), ni en `llms-base.txt:356-358`, y la
skill `backtesting-configuration/SKILL.md:128` recomienda `fold_stride=1` como "Sliding window
evaluation" sin avisarlo. En una búsqueda de hiperparámetros con solapamiento se optimiza para el
horizonte corto. Arreglo mínimo: documentarlo en docstring y skill; la métrica por paso de FE-01
lo hace visible.
