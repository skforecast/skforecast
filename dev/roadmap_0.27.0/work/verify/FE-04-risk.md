# FE-04: `ForecasterEquivalentDate` en `backtesting_forecaster_multiseries` (lente riesgo y compatibilidad de API)

**Veredicto: keep_modified.** Riesgo de API bajo (solo amplía los tipos aceptados, sin parámetros
nuevos ni cambios de valores por defecto ni de salida para los forecasters actuales). El riesgo está
en la corrección (alineación de pliegues, series cortas) y en los conflictos con el trabajo abierto
sobre `_backtesting_forecaster_multiseries`. Esfuerzo M realista si se acota la matriz de parámetros.

## No existe hoy

- `backtesting_forecaster_multiseries` rechaza el forecaster (`skforecast/model_selection/_validation.py:1687-1700`).
  Reproducido: `TypeError ... Got ForecasterEquivalentDate`.
- `check_backtesting_input` lo trata solo como univariante y exige `y` (`_utils.py:200-206,237-240`).
- grep de `EquivalentDate` en `skforecast/` (sin tests): ningún camino multiserie; en ramas
  (`git log --all --grep=equivalent`) nada abierto. Ningún PR abierto lo toca.

## Comprobaciones (scratchpad `FE-04/proto.py`, `checks.py`, `misalign.py`; Python 3.12.3, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1)

1. **Prototipo de 40 líneas** (bucle por nivel + `_calculate_metrics_backtesting_multiseries`): 50
   series diarias, `steps=14`, intervalo conformal: 0,63 s el bucle y 0,09 s las métricas, con
   `average`, `weighted_average` y `pooling`. En un DataFrame ancho sin huecos, las filas
   `(índice, level, fold)` son idénticas a las de `ForecasterRecursiveMultiSeries` con el mismo `cv`.
2. **Series heterogéneas: el bucle por serie con el mismo `cv` no sirve como implementación.** Con
   `initial_train_size=300`, una serie que empieza 50 días tarde se evalúa desde 2020-12-16 (50
   predicciones, 4 pliegues) frente a 2020-10-27 (100, 8 pliegues) en el modelo global; si empieza
   100 días tarde, `ValueError`. Hay que calcular los pliegues una vez sobre el índice común.
3. **Frontera de `window_size`.** `_extract_data_folds_multiseries` conserva series con
   `len(train) >= window_size` (`_utils.py:1027`), pero `ForecasterEquivalentDate.fit` exige
   `len(y) > window_size` (`_forecaster_equivalent_date.py:433`): misma familia que #1410.
4. **`DateOffset`.** `TimeSeriesFold.split` con `window_size` `DateOffset` usa `X.index`
   (`_split.py:985`) y falla con dict o con un `Index` (`AttributeError`, medido); con DataFrame
   funciona. El formato recomendado en multiserie es el dict: arreglo de 1 línea (usar el índice
   extraído) o convertir `window_size` a pasos antes de `split`.
5. **Parámetros.** En serie única: `exog` se ignora en silencio; `return_predictors` ya da
   `ValueError` (`_utils.py:481`); `use_in_sample_residuals=False` exige residuos fuera de muestra;
   `n_jobs='auto'` da 1 (`select_n_jobs_backtesting`).

## API de menor riesgo

- Misma firma pública; solo se añade `ForecasterEquivalentDate` a los tipos aceptados y al docstring.
- Despacho a una función privada nueva (`_backtesting_equivalent_date_multiseries`): no tocar
  `_backtesting_forecaster_multiseries` ni `_fit_predict_forecaster_multiseries`, para no chocar con
  #1407, #1410, #1408 ni con la integración de métricas de intervalo (#1409) y columna de horizonte (D1119).
- `check_preprocess_series` (ancho, largo o dict), pliegues con `cv.split` sobre el índice común,
  un clon por nivel ajustado solo en los pliegues con `refit`, saltar el nivel si
  `len(train) <= window_size` o la última ventana no está completa, y salida y métricas con
  `_calculate_metrics_backtesting_multiseries` (mismo formato: `level`, `fold`, `pred`,
  `lower_bound`, `upper_bound`).
- Matriz: `interval` solo conformal (ya es el valor por defecto en multiserie); `bootstrapping`,
  `return_predictors=True`, `use_in_sample_residuals=False` (un único forecaster guarda residuos de
  una sola serie: aplicarlos a todas daría intervalos erróneos sin aviso) e `initial_train_size=None`
  dan `ValueError` claro; `exog` da `IgnoredArgumentWarning` (permite la misma llamada que el modelo
  global al comparar). `levels`, `add_aggregated_metric`, `n_jobs`, `show_progress`,
  `suppress_warnings` y todo `TimeSeriesFold` (gap, refit entero, `fold_stride`, `skip_folds`) se
  admiten.
- Fuera de la 0.27: `grid_search_equivalent_date` multiserie, el mismo camino para `ForecasterStats`
  y un `ForecasterEquivalentDate` global (cambiaría atributos, residuos y serialización).

## Esfuerzo (M confirmado)

`_validation.py` (comprobación de tipo y función privada, unas 120 líneas), `_utils.py`
(`check_backtesting_input` acepta `series` para este forecaster), `_split.py:985`, unos 12 a 15 tests
(formatos, panel heterogéneo comparado con `ForecasterRecursiveMultiSeries`, `levels`, agregados,
conformal, `DateOffset`, errores), skills `baseline-forecasting` y `backtesting-configuration`,
`model-selection-signatures.md`, `llms-base.txt`, guía de baseline y nota de versión. Hacerlo después
de fusionar #1407, #1410 y #1408 y de integrar las métricas de intervalo, para heredar su salida.

## Hallazgo lateral (fallo nuevo, flujo común)

`backtesting_forecaster(ForecasterEquivalentDate(offset=7), ..., interval=[0.1, 0.9])` con el
`interval_method='bootstrapping'` por defecto lanza `AttributeError: ... no attribute 'predict_quantiles'`
(también con `0.8`; con `'bootstrapping'`, `predict_bootstrapping`). Causa: el forecaster figura en
`forecasters_boot_conformal` (`_utils.py:222-228`) aunque solo admite conformal. Arreglo S: `ValueError`
claro que indique `interval_method='conformal'`. La guía de comparación siempre lo fija a mano.
