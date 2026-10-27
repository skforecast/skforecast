# FE-04: `ForecasterEquivalentDate` en `backtesting_forecaster_multiseries` (lente valor frente a mantenimiento)

**Veredicto: keep_modified (prioridad media-baja, esfuerzo M confirmado).** Hacerlo solo para
`ForecasterEquivalentDate` (no como vía genérica de "modelos locales"), con los pliegues resueltos
una vez sobre el índice común de todas las series. Ese requisito no aparece en el boceto y es lo que
más valor aporta. Mientras tanto, arreglo de docs (S) que vale por sí solo.

## Qué existe hoy (comprobado en 0.26.x)

- `backtesting_forecaster_multiseries` solo acepta `ForecasterRecursiveMultiSeries`,
  `ForecasterDirectMultiVariate` y `ForecasterRnn` (`skforecast/model_selection/_validation.py:1687-1701`).
  Reproducido: `TypeError: forecaster must be of type [...]. Got ForecasterEquivalentDate`.
- `ForecasterEquivalentDate` es de serie única (`_forecaster_equivalent_date.py:240,246`) y
  `check_backtesting_input` lo trata como univariante y exige que `y` sea una Series
  (`skforecast/model_selection/_utils.py:200-206,237-240`).
- Precedente de diseño: `backtesting_foundation` ya ejecuta modo serie única o multiserie en una
  sola función y reutiliza `_calculate_metrics_backtesting_multiseries` (`_utils.py:1067`), que
  necesita los pliegues sobre `span_index` y `window_size` (para `y_train` del MASE, `_utils.py:1170-1190`).
- La guía `comparing-forecasting-models.ipynb` (celda 29) hace el bucle por serie para el baseline
  y repite el mismo bucle para ARIMA y ETS; su función `summarize` calcula MASE por serie y la media a mano.
- La skill `baseline-forecasting` no menciona multiserie.

## Demanda

- Issues y discusiones: búsqueda semántica ("EquivalentDate multiple series baseline multiseries
  backtesting"): ninguna petición explícita. La fricción es la de los propios mantenedores (la guía).
- Competidores: darts (`NaiveSeasonal` sobre listas), statsforecast y autogluon (`SeasonalNaive`)
  evalúan el baseline en un panel con una llamada.
- Contexto: el tema recurrente número 1 de la 0.27 es el backtesting multiserie con series
  heterogéneas (#1406, #1410, #1357...).

## Hallazgo medido: el bucle manual compara periodos distintos sin avisar

`scratchpad/fe04-value/exp.py` (Python 3.12.3, skforecast 0.26.0, numpy 2.5.4, pandas 2.3.3,
scikit-learn 1.9.1). Panel diario de 3 series, `s2` empieza 60 días más tarde,
`TimeSeriesFold(steps=14, initial_train_size=300)`:

| | primera fecha de test de `s2` | predicciones de `s2` |
|:--|:--|--:|
| `backtesting_forecaster_multiseries` (modelo global) | 2022-10-28 | 100 |
| bucle por serie (patrón de la guía) | 2022-12-27 | 40 |
| bucle por serie con `initial_train_size="2022-10-27"` | 2022-10-28 | 100 |

Con un `initial_train_size` entero, cada serie del bucle cuenta posiciones desde su propio inicio:
el baseline se evalúa en otra ventana y con menos puntos que el modelo global, y la comparación
(MASE, mejora frente al naive) sale sesgada sin ningún aviso. Con una fecha se alinea. Una
implementación que llame a `backtesting_forecaster` por nivel con el mismo `cv` reproduce el error.

## Lo que aporta frente a lo que ya existe

- Ergonomía: el bucle son 8 líneas, pero las métricas agregadas (`average`, `weighted_average`,
  `pooling`) y el MASE por serie con su `y_train` hay que calcularlos a mano.
- Corrección: pliegues idénticos a los del modelo global (tabla anterior).
- Rendimiento: no es un argumento. 500 series x 91 días con intervalo conformal: 7,55 s en bucle
  (`exp2.py`), unos 15 ms por serie.
- Atajo sin código (descartado como solución): `ForecasterRecursiveMultiSeries` con un estimador
  que devuelve la columna `lag_7` (`lags=[7]`, `transformer_series=None`, `encoding=None`) da
  exactamente las mismas predicciones que el bucle (diferencia máxima 0,0 en 300 puntos) y además
  las métricas agregadas. Pero no cubre `DateOffset` ni `n_offsets > 1` y es un truco poco
  descubrible: no merece documentarse como receta oficial.

## Coste de mantenimiento

- Código: rama nueva en `backtesting_forecaster_multiseries` (unas 80 a 120 líneas): normalizar
  `series` con `check_preprocess_series`, calcular pliegues con `cv.split` sobre el índice común,
  ajustar y predecir por nivel, saltar con aviso las series sin datos suficientes (coherente con
  #1406/PR-1407 y #1410), construir la salida con `level` y llamar a
  `_calculate_metrics_backtesting_multiseries`. `window_size` con `DateOffset` solo se conoce tras
  `fit`.
- Matriz de parámetros distinta en una función central ya cargada: sin `exog`, sin
  `return_predictors`, solo `conformal`; hay que tocar `check_backtesting_input`, donde el forecaster
  pasaría a ser univariante y multiserie a la vez.
- Tests: unos 12 (formatos ancho, largo y dict; series heterogéneas alineadas con el modelo global;
  `levels`; agregados; intervalo conformal; `DateOffset`; errores de `exog` y `return_predictors`).
- Docs y contexto IA: `skills/baseline-forecasting/SKILL.md`,
  `skills/backtesting-configuration/SKILL.md`,
  `skills/complete-api-reference/references/model-selection-signatures.md`, `llms-base.txt`,
  docstring y la guía de comparación.
- Arrastra peticiones: `grid_search_equivalent_date` multiserie y el mismo camino para
  `ForecasterStats` (la guía repite el bucle con ARIMA y ETS). Las dos quedan fuera de alcance.

## Versión reducida propuesta

1. Docs ya (S, sin código): en `baseline-forecasting` y en la guía, si se hace un bucle por serie,
   usar `initial_train_size` como fecha (la misma que el modelo global) y explicar por qué.
2. Código (M): aceptar solo `ForecasterEquivalentDate` en `backtesting_forecaster_multiseries`,
   con los pliegues sobre el índice común y la reutilización del cálculo de métricas multiserie.
   Admitir `metric`, `levels`, `add_aggregated_metric`, `interval` (solo conformal, como en serie
   única), `n_jobs` y `show_progress`; `exog` y `return_predictors` dan un error claro. Sin
   búsqueda multiserie ni vía genérica de modelos locales en la 0.27. Encaja mejor después de
   integrar las métricas de intervalo en el backtesting (GAP-backtesting-interval-metrics), que
   también resuelve la cobertura del baseline.

## Riesgo

Bajo para la API (solo amplía el tipo aceptado, sin romper nada). Riesgo de corrección en la
alineación de pliegues y en las series cortas; se cubre con un test que compare índice y `fold` de
las predicciones con las de `ForecasterRecursiveMultiSeries` para el mismo `cv` en un panel
heterogéneo.
