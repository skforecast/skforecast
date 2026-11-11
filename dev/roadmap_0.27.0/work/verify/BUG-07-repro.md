# BUG-07: verificación

**Veredicto: keep (esfuerzo S).** Se reproduce al 100% en 0.26.0 (pandas 2.3.3). No es un resultado silencioso erróneo (falla con error explícito), pero sí un flujo normal: el formato de entrada documentado "DataFrame largo con MultiIndex" solo funciona si el MultiIndex lo genera `reshape_series_wide_to_long`.

Script: `dev/roadmap_0.27.0/repro/BUG-07.py`

## Resultado observado

| Ruta | `levels[1].freq` | Resultado |
|---|---|---|
| 1. `reshape_series_wide_to_long` (la guía) | `<Day>` | fit/predict OK |
| 2. `read_csv` + `set_index(['series_id', 'date'])` | `None` | ValueError |
| 3. Datos de la guía: `reset_index()` + `set_index()` | `None` | ValueError |
| 4. `set_index` + `sort_index()` | `None` | ValueError |
| 5. `backtesting_forecaster_multiseries` con la ruta 2 | `None` | ValueError |
| 6. `reshape_series_long_to_dict(..., freq='D')` | n/a | OK, mismas predicciones que 1 |
| 7. `set_levels(DatetimeIndex(..., freq='infer'), level=1)` a mano | `<Day>` | OK, mismas predicciones que 1 |
| 8. Esbozo del arreglo (inferir freq por grupo + `asfreq`) | n/a | OK, mismas predicciones que 1 |

Mensaje: `If series is a dictionary, all series must have a Pandas RangeIndex or DatetimeIndex with the same step/frequency. If it a MultiIndex DataFrame, the second level must be a DatetimeIndex with the same frequency for each series. Found series with no frequency or step.`

## Causa raíz

- `skforecast/utils/utils.py:4199-4202`: la rama MultiIndex construye cada serie con `group[first_col].droplevel(0)`. `droplevel` hace `levels[1].take(codes)`, que solo conserva `freq` si `levels[1]` ya la tenía. Un MultiIndex creado con `set_index` o `MultiIndex.from_arrays` factoriza las fechas y `levels[1]` nunca tiene `freq`.
- `skforecast/utils/utils.py:4249` y `:4267-4274`: el chequeo común exige `idx.freq` no nulo y lanza el ValueError.
- Los tests solo construyen el MultiIndex con `reshape_series_wide_to_long` (`skforecast/utils/tests/tests_utils/test_check_preprocess_series.py:9`), por eso no se detectó.

## Matices respecto a la propuesta

- El mensaje sí menciona el MultiIndex (no solo diccionarios), pero exige una `freq` en el nivel 1 que el usuario no sabe fijar (la única salida es `set_levels(..., freq=...)`, no documentada). Además tiene la errata "If it a".
- La exog larga creada con `set_index` ya funciona (no se exige `freq`); no hace falta tratarla. Comprobado en fit y predict.
- También afecta a `ForecasterFoundation` y `backtesting_foundation` (`skforecast/foundation/_utils.py:414` reenvía a `check_preprocess_series`) y a las búsquedas multiserie (`_search.py:1309,1711`).
- Incoherencia de documentación: la guía (`independent-multi-time-series-forecasting.ipynb`, celdas 5 y 11) usa el MultiIndex como entrada principal, mientras que `skills/forecasting-multiple-series/SKILL.md:52` dice que el formato largo no se acepta directamente.

## Arreglo propuesto

En la rama MultiIndex de `check_preprocess_series`, tras el `groupby`: si el índice del grupo no tiene `freq`, usar `pd.infer_freq` (con 3 o más observaciones) y `asfreq`; si no se puede inferir (huecos o series muy cortas), lanzar un error específico de MultiIndex que sugiera `reshape_series_long_to_dict(..., freq=...)`. Corregir la errata del mensaje. Tests con MultiIndex creado por `set_index`, series de distinta longitud y un grupo con huecos. Alinear la skill con la guía.
