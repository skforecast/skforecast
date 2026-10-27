# QW-06: alinear exog por fecha en predict de ForecasterRecursive y ForecasterDirect

**Veredicto: keep_modified** (esfuerzo S confirmado con prototipo).

## Que existe hoy

- `check_predict_input` solo alinea por índice en `ForecasterRecursiveMultiSeries` (`skforecast/utils/utils.py:1656`, `align_by_index`); el resto exige que `exog` empiece justo tras `last_window` (`utils.py:1738-1745`). Decisión reciente y consciente: el PR #1346 (0.26.0) cambió solo MultiSeries y mantuvo por posición Recursive, Direct, Classifier, DirectMultiVariate, Stats y Rnn.
- `ForecasterFoundation` ya alinea con DatetimeIndex (`skforecast/foundation/_foundation_model.py:869-879`) y es estricto con RangeIndex (`:888-897`).
- `fit` de Recursive exige la misma longitud que `y` (`skforecast/recursive/_forecaster_recursive.py:845-853`); MultiSeries también acepta exog más larga en `fit`.
- Las skills ya lo tratan como error típico: `skills/forecasting-single-series/SKILL.md:35` y `skills/troubleshooting-common-errors/SKILL.md:105-112`.

## Comprobación del "before" (0.26.x, datos horarios sintéticos)

- `predict(steps=24, exog=exog_all)` lanza `ValueError: To make predictions exog must start one step ahead of last_window` en Recursive, Direct, Classifier y DirectMultiVariate (también `predict_interval`).
- MultiSeries con el mismo `exog_all` predice sin avisos y da exactamente lo mismo que Recursive con el recorte manual.
- La fricción es real pero acotada: el mensaje de error ya indica la fecha esperada; el arreglo manual es una línea.

## Prototipo (worktree temporal, ya borrado)

- Helper `_select_exog_from_first_step(exog, first_step)` (`index.get_loc` + `iloc`), llamado al inicio del bucle de exog en `check_predict_input` y antes de `transform_dataframe` en `_create_predict_inputs` de Recursive y Direct: +36/-1 líneas en 4 ficheros.
- Resultados idénticos a recortar a mano en `predict`, `predict_interval` (bootstrapping) y `predict_quantiles`. Sin coste medible (exog de 200 000 filas: 1,5 ms frente a 1,7 ms).
- Casos límite correctos: exog que empieza tarde, futuro corto o con huecos siguen dando error; NaN o fechas duplicadas solo en el histórico ya no avisan; índice sin `freq` (CSV) funciona; `last_window` del usuario funciona.
- Tests: 871 pasan en `test_check_predict_input.py` + tests de Recursive y Direct, y 89 en backtesting refit/no_refit; 1 test a adaptar (`test_check_predict_input_ValueError_when_exog_index_does_not_follow_last_window_index_DatetimeIndex`, cuya exog contiene la fecha esperada).

## Riesgo encontrado y modificación

- Con **RangeIndex** el prototipo crea un resultado silencioso erróneo: `y` con índice 0..99 y `exog_future = exog.iloc[100:300].reset_index(drop=True)` (índice 0..199) selecciona las filas etiquetadas 100..123 (originales 200..223). Predicción `[-1.331, -1.229, -3.212]` frente a la correcta `[1.005, 1.979, -0.329]`; en 0.26.x da error.
- Por eso: **alinear solo con DatetimeIndex** (como Foundation); con RangeIndex se mantiene el chequeo estricto.
- Hallazgo lateral: MultiSeries en 0.26.x ya tiene ese resultado silencioso con RangeIndex reseteado (mismos números, sin aviso). Caso poco frecuente; anotarlo, no priorizarlo.
- Incluir también `ForecasterRecursiveClassifier` y `ForecasterDirectMultiVariate` (unas 4 líneas cada uno) para no crear otra asimetría. `ForecasterStats` es opcional (trivial, fuera de foco); `ForecasterRnn` queda fuera de alcance.
- No tocar `fit`: su chequeo de longitud lo usan backtesting y búsquedas (`_search.py:351,669`); sería otro cambio.

## API y documentación

- Sin parámetros nuevos ni columnas nuevas; solo pasan entradas que hoy fallan. No rompe código.
- Actualizar `tools/ai/llms-base.txt:157`, las dos skills citadas, los docstrings de `exog` en los `predict*` y la nota de versión; regenerar el contexto de IA.

## Before / after revisados

```python
# before
forecaster.fit(y=y, exog=exog_all.loc[y.index])
start = forecaster.last_window_.index[-1] + forecaster.last_window_.index.freq
predictions = forecaster.predict(steps=24, exog=exog_all.loc[start:])

# after (solo DatetimeIndex; fit sigue pidiendo la exog del periodo de y)
forecaster.fit(y=y, exog=exog_all.loc[y.index])
predictions = forecaster.predict(steps=24, exog=exog_all)
```

Scripts: /tmp/claude-0/-home-user-skforecast/8505cf39-716d-5c4c-adf6-669c7ec65b16/scratchpad/qw06/ (before.py, after.py, interval.py, range_risk.py, range_risk_ms.py).
