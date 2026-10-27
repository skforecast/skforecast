# QW-04: columna `y_true` en las predicciones de backtesting

**Veredicto: keep_modified** (código S; con la actualización mecánica de tests, alrededor de un día)

## Qué existe hoy

- Ninguna función de backtesting devuelve el valor observado. Columnas medidas (0.26.0):
  multiserie `['level', 'fold', 'pred', 'lower_bound', 'upper_bound']`; una serie `['fold', 'pred']`.
- El valor real ya se calcula dentro y se descarta:
  - `_backtesting_forecaster`: `y_true = y.loc[...]` (`_validation.py:669`).
  - `_backtesting_stats`: `_validation.py:2167` y `:2180`.
  - `_backtesting_foundation` (una serie): `_validation.py:2681`.
  - Multiserie (ML y foundation): merge interno en `_calculate_metrics_backtesting_multiseries` (`_utils.py:1149-1166`).
- No hay columna `y_true` en ningún sitio de `skforecast/` (fuera de variables internas), ni en
  `tools/ai/llms-base.txt` ni en las skills. Las skills enseñan el patrón manual
  (`skills/prediction-intervals/SKILL.md:205`, `references/interval-compatibility.md:133`).
- En `docs/user_guides` hay 21 usos de `data.loc[pred*.index, col]` en 8 guías; la mayoría sobre la
  salida de backtesting (residuos, `set_out_sample_residuals`, cobertura, gráficos).
- No hay issue que lo pida explícitamente (búsqueda semántica en GitHub sin resultados directos).
  Es el paso previo de GAP-backtesting-interval-metrics, ISSUE-1409, DISC-1119 y FE-01.

## Reproducción del "before" (skforecast 0.26.0, pandas 2.3.3, numpy 2.5.4)

Scripts: `/tmp/claude-0/-home-user-skforecast/8505cf39-716d-5c4c-adf6-669c7ec65b16/scratchpad/qw04/before.py`, `overhead.py` y `side.py` (3 series dict de longitudes distintas, `fold_stride=5`).

- El snippet del "before" es correcto con entrada dict (440 filas, 360 fechas duplicadas por folds
  solapados, valores verificados uno a uno; 15 `y_true` NaN donde la serie ya no tiene datos).
- Con DataFrame ancho el mismo snippet falla:
  `TypeError: first argument must be an iterable of pandas objects, you passed an object of type "DataFrame"`.
  Hay que cambiar a `series.stack()` con otros nombres de índice. Con formato largo (MultiIndex) es
  otra variante más. La fricción real es mayor que la descrita: el código depende del formato de entrada.
- Una serie: el ahorro es solo una línea (`data.loc[preds.index, "y"]` frente a `preds["y_true"]`).

## Prototipo y coste medido

`reindex` de la serie larga sobre `MultiIndex(level, índice)` reproduce exactamente el merge (dict y
DataFrame ancho). Coste (`overhead.py`, Ridge, `refit=False`):

| Series | `fold_stride` | Filas | Backtesting | `y_true` | Memoria extra |
|---|---|---|---|---|---|
| 1000 | None | 65 000 | 9,1 s | 20 ms (0,2%) | 0,5 MB (sobre 5,0 MB) |
| 1000 | 1 | 434 000 | 35,6 s | 41 ms (0,1%) | 3,5 MB (sobre 33,4 MB) |
| 3000 | None | 195 000 | 24,2 s | 97 ms (0,4%) | 1,6 MB (sobre 15,1 MB) |

Coste despreciable, también dentro de las búsquedas (llaman al backtesting público y descartan las
predicciones). No hace falta un flag por rendimiento.

## Hallazgo lateral (relevante para el diseño)

En `_backtesting_forecaster_multiseries` solo se pone a NaN `pred` cuando la serie no tiene valor
observado (`_validation.py:1480-1486`); `lower_bound`/`upper_bound` (y cuantiles o bootstrapping) se
quedan. `backtesting_foundation` sí enmascara todas las columnas (`_validation.py:2627-2641`). Y
`calculate_coverage` no ignora NaN (`metrics.py:516`, `np.mean` de una comparación con NaN). Medido con
una serie que termina antes: cobertura 0,63 calculada con las filas sin observación frente a 0,73 sin
ellas. Con `y_true` en la salida, el usuario calculará la cobertura directamente y la infraestimará en
silencio si no hace `dropna`.

## Ajustes respecto a la propuesta

1. Sin flag, como cambio de API anunciado, siguiendo el precedente de la columna `fold` (0.18.0,
   `releases.md:725`). Colocar `y_true` justo antes de `pred` (lectura natural `y_true`/`pred`).
2. Un helper privado en `_utils.py` (~20 líneas, `reindex` sobre la serie larga) usado en 5 sitios:
   `_backtesting_forecaster` (`:660`), `_backtesting_forecaster_multiseries` (`:1506`),
   `_backtesting_stats` (`:2140`) y las dos ramas de `_backtesting_foundation` (`:2667`, `:2681`).
   Usar la `series` ya preprocesada para que los nombres coincidan con `level` (ojo a PR-1408, IDs enteros).
3. En el mismo cambio, enmascarar todas las columnas de predicción en el backtesting multiserie ML
   cuando no hay valor observado (como foundation), o como mínimo que las métricas integradas de
   GAP-backtesting-interval-metrics descarten filas con `y_true` NaN.
4. Hacerlo antes o junto a GAP-backtesting-interval-metrics / FE-01: la función de evaluación no
   necesitaría recibir `series` y la cobertura por paso sale de la propia salida.
5. Actualizar docstrings `Returns` de las 4 funciones públicas, `tools/ai/llms-base.txt`, las skills
   con el patrón `data.loc[preds.index]` (ai-context-sync) y la nota de versión.

## Riesgo

Bajo a medio. Rompe código que compara el DataFrame completo, renombra columnas por posición
(`preds.columns = [...]`) o usa `iloc` por posición; también un merge propio que ya llame `y_true` a la
columna (saldrían `y_true_x`/`y_true_y`). Las funciones de `skforecast.plot` usan nombres de columna,
no se ven afectadas. Colisión improbable con `return_predictors=True` si una exógena se llama `y_true`.

## Tamaño

Código ~40 líneas. Tests: ~160 aserciones de predicciones (118 idénticas,
`assert_frame_equal(expected_predictions, backtest_predictions)`, en 6 ficheros de `tests_validation`,
más `test_backtesting.py` de EquivalentDate), actualizables con un script. Los notebooks no necesitan
reejecutarse en el mismo PR.
