# FE-03: ventanas estacionales en `RollingFeatures` (lente riesgo y compatibilidad de API)

**Veredicto: keep_modified.** Riesgo bajo si se usa la semántica de tramo (`window_sizes` en pasos
de tiempo, `season_length` como salto). La API propuesta (`window_sizes=[24, 7, 4]` = número de
estaciones) rompe el contrato con los forecasters y se descarta. Esfuerzo M realista (código S,
el resto es tests, docs y contexto IA).

## No existe hoy

- `RollingFeatures` solo corta ventanas contiguas: `X[-window_sizes[j]:]`
  (`skforecast/preprocessing/_preprocessing.py:1604,1657,1673`) y `rolling(closed='left')` en
  `transform_batch` (`:1462-1500`). grep de `season_length`, `SeasonalRolling` o `same hour` en
  `skforecast/`: nada. Lo más cercano es `ForecasterEquivalentDate(offset, n_offsets, agg_func)`,
  pero como baseline, no como predictor.
- Ninguna rama ni PR abierto toca esta parte: solo `origin/feature_synthetic_data_generator` modifica
  `RollingFeatures.transform`, y son optimizaciones antiguas ya presentes en 0.26.x.

## Riesgos comprobados (`scratchpad/fe03-risk/risk_checks.py`, con el prototipo de la lente de valor)

1. **Contrato `window_sizes`.** El único consumidor externo es `initialize_window_features`
   (`skforecast/utils/utils.py:211,229,244`), que calcula `window_size` del forecaster con
   `wf.window_sizes`. Si `window_sizes` fuera el número de estaciones, el forecaster reservaría 7
   filas en lugar de 168 (la lente de valor midió el `ValueError` en `fit`). Con la semántica de tramo
   el atributo no cambia de significado y no hay que tocar ni `utils` ni los forecasters.
2. **Validación de duplicados.** `(stat, window_size)` duplicado lanza `ValueError`
   (`_preprocessing.py:1338-1346`): medido, `mean` contigua y `mean` estacional con el mismo tamaño
   se rechazan. La clave debe ser `(stat, window_size, season_length)`, igual que la de
   `unique_rolling_windows` (`:1184-1205`).
3. **`min_periods`.** Hoy se valida contra `window_sizes` (`:1365-1370`); con estacionalidad pasa a
   contar puntos (`window_sizes // season_length`) y la validación debe cambiar en consecuencia.
4. **Forecasters guardados con 0.26.** Un `RollingFeatures` serializado no tendrá el atributo nuevo:
   medido, `AttributeError` al predecir si `transform` lo lee. Se evita con un valor por defecto a
   nivel de clase (`season_length = None`), sin `__setstate__`.
5. **`ForecasterDirect`.** La ventana se alinea con el origen, no con el paso: medido, el valor de
   `roll_mean_7_s24` es el mismo en los 24 pasos (10,18), mientras que la "misma hora" real para los
   pasos 6 y 12 es 14,14 y 11,22 (en `ForecasterRecursive` coincide exactamente). No es un fallo (los
   lags se comportan igual), pero hay que documentarlo: en Direct es "la hora de t+1".
6. **Multiserie con series cortas.** Con `168 x 4` el `window_size` sube a 672 y `fit` falla entero si
   alguna serie del diccionario es más corta (`ValueError: Length of 'b' must be greater than the
   maximum window size`). Comportamiento previo (igual que `lags=672`), pero esta función lo hará más
   frecuente; documentarlo junto a ISSUE-1410.
7. **Sin interacción problemática** con `differentiation` (medido: `window_size` 169, predicción
   correcta), `transformer_y`, `exog`, `encoding`, bootstrapping (camino 2D) ni búsquedas (las
   window features no se buscan en rejilla). `RollingFeaturesClassification` queda fuera.

## API de menor riesgo

```python
RollingFeatures(
    stats         = ['mean', 'mean', 'mean'],
    window_sizes  = [24, 168, 672],     # tramo en pasos de tiempo (sin cambio de significado)
    season_length = [None, 24, 168],    # None = ventana contigua (comportamiento actual)
)
# roll_mean_168_s24 = media de y[t-24], y[t-48], ..., y[t-168] (7 puntos)
```

- Parámetro nuevo al final de la firma, `None` por defecto: salidas idénticas a 0.26 sin él.
- Reglas: `window_sizes % season_length == 0`; nombres con sufijo `_s{season_length}` solo si se
  usa; `min_periods` en puntos; cachés de NaN de `_transform_vectorized` con clave `(ws, s)`; corte
  `X[-ws::s]` en `transform`; `groupby(fase).rolling(n, closed='left')` en `transform_batch`.
- Ficheros: `skforecast/preprocessing/_preprocessing.py` (`__init__`, `_validate_params`,
  `__repr__`, `_repr_html_`, `transform_batch`, `transform`, `_transform_vectorized`),
  `tests_preprocessing/test_RollingFeatures.py` (equivalencia fila a fila batch/transform, 2D,
  NaN, validación, pickle antiguo), `docs/user_guides/window-features-and-custom-features.ipynb`,
  `tools/ai/llms-base.txt`, `skills/feature-engineering/` (SKILL.md y
  `references/rolling-stats-reference.md`),
  `skills/complete-api-reference/references/preprocessing-signatures.md`, `releases.md`.
- Medir antes de fusionar que `predict` recursivo y `predict_bootstrapping` no empeoran con
  `season_length=None` (camino caliente optimizado).
