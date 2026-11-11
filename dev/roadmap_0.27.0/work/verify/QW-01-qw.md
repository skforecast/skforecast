# QW-01: predict() con levels=None excluye sin aviso las series que terminan antes

**Veredicto: keep_modified** (esfuerzo S, sin cambio de API)

## Qué existe hoy

- `skforecast/utils/utils.py:4668-4687` (`preprocess_levels_self_last_window_multiseries`): si las `last_window_` terminan en fechas distintas, solo se predicen las que llegan a la fecha máxima. El aviso (`IgnoredArgumentWarning`) solo se emite si `input_levels_is_list` es True (línea 4680). Con `levels=None` la exclusión es silenciosa.
- Comportamiento documentado pero no avisado: docstrings de `levels` en `_forecaster_recursive_multiseries.py:2607,3279,3462,3565,3745,3917,4073,4170` y la guía `docs/user_guides/multi-series-with-different-length-and-different_exog.ipynb` (celda 18, serie `id_1002`). No aparece en `tools/ai/llms-base.txt` ni en las skills.
- Solo afecta a `predict*` y `create_predict_X` sin `last_window` (uso en producción). El backtesting pasa `last_window` explícito (`model_selection/_validation.py:1070-1077`), así que no se ve afectado ni recibiría avisos nuevos.
- No está en la lista de identificados; es la misma familia que ISSUE-1406 / PR-1407 (backtesting), que no lo cubre.

## Reproducción (skforecast 0.26.0, pandas 2.3.3)

Panel de 6 series construido desde formato largo, quitando el registro del último día a 2 artículos:

- dict (`reshape_series_long_to_dict`): `trained=6 predicted=4 missing=['item_3','item_5']`, avisos en predict/predict_interval: `[]`.
- DataFrame ancho por `pivot` (NaN final): mismo resultado, `[]`.
- Con `levels=list(...)`: sí aparece el `IgnoredArgumentWarning`.
- La propia batería de tests cae en este caso: 4 tests existentes con `series_dict` y `levels=None` (`test_create_predict_inputs_output_when_series_and_exog_dict`, `test_predict_interval*_conformal_output_when_series_and_exog_dict`) excluyen series sin aviso.

El "before" es correcto y la fricción es real: hay que saber que pasa y comparar a mano `series_names_in_` con `p['level']`.

## Prototipo medido (worktree desechable)

Quitar `input_levels_is_list and` y mejorar el mensaje (conteo, lista ordenada y truncada a 10, cómo predecirlas). Resultado:

- Tests de `tests_forecaster_recursive_multiseries/`, `test_preprocess_levels_self_last_window_multiseries.py` y `test_plot_prediction_intervals.py`: 724 pasan, 3 fallan solo por el texto del mensaje.
- `tests_validation/` (backtesting): sin cambios (no pasan por esta función).
- `suppress_warnings=True` lo silencia correctamente.

## Ajustes respecto a la propuesta

1. Mensaje determinista y acotado: hoy imprime un `set` (orden aleatorio) sin límite; con cientos de artículos descatalogados sería ilegible. Ordenar y truncar.
2. `predict_interval(method="bootstrapping")` llama a `predict_bootstrapping` y luego a `predict`, así que el aviso sale dos veces (ya ocurre hoy con `levels` lista). Arreglo trivial: llamar al `predict` interno con los niveles ya filtrados.
3. Indicar en el aviso que, si se piden con `levels`, sus predicciones empiezan tras su propia última observación (medido: empiezan el día de la fecha máxima, no el siguiente), para que nadie mezcle horizontes desalineados.
4. Actualizar docstrings de `levels` ("se emite un aviso con las series excluidas"), los 3 tests, un test nuevo con `levels=None`, nota de versión y reejecutar la celda de la guía (mostrará el aviso). Opcional: una línea en `skills/troubleshooting-common-errors`.

## Riesgo

Bajo. No cambia la salida ni los parámetros. Paneles con muchas series descatalogadas verán un aviso en cada `predict`; es informativo y se silencia con `suppress_warnings=True` o pasando `levels`. Solo rompería código que convierta avisos en errores.

## Tamaño

`utils.py` (~15 líneas), `_forecaster_recursive_multiseries.py` (docstrings y 1 línea en `predict_interval`), 3 tests ajustados y 1-2 nuevos. Medio día incluyendo la nota de versión.
