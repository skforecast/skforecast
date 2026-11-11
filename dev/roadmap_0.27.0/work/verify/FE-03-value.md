# FE-03: ventanas estacionales en `RollingFeatures` (lente valor frente a mantenimiento)

**Veredicto: keep_modified (prioridad media-baja, esfuerzo M por docs; el código es S).**
Hacerlo, pero con otra semántica de parámetros: `window_sizes` sigue siendo el tramo en pasos de
tiempo y `season_length` indica el salto dentro de ese tramo. Tal como está propuesto
(`window_sizes=[24, 7, 4]` = número de estaciones) rompe el contrato con los forecasters.

## Qué existe hoy (comprobado en 0.26.x)

- `RollingFeatures` solo usa ventanas contiguas: `transform` toma `X[-window_sizes[j]:]`
  (`skforecast/preprocessing/_preprocessing.py:1604`, `:1657`, `:1673`) y `transform_batch` un
  `rolling` normal (`:1462-1500`). grep de `season`, `offset` o `stride` en la clase: nada.
- Lo que sí existe y cubre parte del caso: lags estacionales explícitos (`lags=[..., 48, 72, ...]`)
  y `ForecasterEquivalentDate(offset, n_offsets, agg_func)` como baseline (no como predictor).
- Las clases propias no se guardan con `save_forecaster` (`skforecast/utils/utils.py:3599-3611`).
- Ni la guía `window-features-and-custom-features.ipynb` ni la skill `feature-engineering`
  (`SKILL.md:250-257`) mencionan medias estacionales ni lags estacionales múltiples.

## Contrato con los forecasters (hallazgo)

Los forecasters calculan `window_size` leyendo `wf.window_sizes`, no `max_window_size`
(`skforecast/utils/utils.py:175,211,229,244,259`). Con la API propuesta (`window_sizes=[24, 7, 4]`,
`season_length=[None, 24, 168]`) el forecaster reserva 168 filas en vez de 672 y `fit` falla
(medido: `ValueError: The method transform_batch ... must return a DataFrame with the same number
of rows...`). Habría que reescribir el atributo `window_sizes` distinto del argumento del usuario.
Con la semántica de tramo no hace falta tocar ni el contrato ni los forecasters.

## Demanda

- Issues: búsqueda semántica ("same hour previous days", "seasonal rolling", "window on lagged
  values", "same day of week"): ninguna petición explícita. #797 (2024, cerrada) pedía rolling de
  exógenas, no esto. DISC-1153 es otro estadístico (asimetría), no otra geometría de ventana.
- Competidores: mlforecast trae `SeasonalRollingMean/Std/Min/Max/Quantile`; sktime y darts solo
  ventanas desplazadas contiguas. Es paridad con un competidor, no un hueco generalizado.

## Valor medido frente a lo que ya existe

`scratchpad/fe03-value/exp_fair.py` (prototipo en `proto.py`, subclase de ~40 líneas útiles).
Backtesting `steps=24`, `refit=False`, LightGBM 500 árboles y Ridge, exógenas de calendario y
meteo. Python 3.12.3, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1, lightgbm 4.7.0.

| Configuración | bike_sharing LGBM | vic_electricity LGBM | bike Ridge | vic Ridge |
|:--|--:|--:|--:|--:|
| A: lags 1-24, 168 + roll mean 24, 168 | 49,750 | 248,83 | 71,74 | 481,87 |
| L: A + lags 48..144, 336, 504, 672 (ya existe) | 48,081 | 244,11 | 65,32 | 445,64 |
| B: A + media estacional 24x7 y 168x4 | **47,035** | **239,53** | 66,44 | 451,89 |
| BL: L + medias estacionales | 46,828 | 239,83 | 65,32 | 445,64 |

- La ganancia frente a la línea base justa (L, lags estacionales que ya se pueden poner hoy) es
  **-2,2 % (bike) y -1,9 % (vic)** con árboles, con 2 variables en lugar de 8. El -4,1 % de la
  propuesta se medía contra A, sin lags estacionales.
- Con modelos lineales no aporta nada: la media es combinación lineal de los lags (L = BL exacto).
- Los lags estacionales explícitos ya capturan entre el 50 % y el 60 % de la mejora: hoy es un
  consejo de documentación gratuito.

## Las clases propias son frágiles (evidencia directa)

La clase `SeasonalRolling` del informe de competidores tiene un desajuste silencioso entre
entrenamiento y predicción: `transform_batch` usa 7 puntos, pero `transform` toma todos los
múltiplos de 24 de la última ventana (672 / 24 = 28 puntos). Medido en `check_lens_class.py`:
valor de entrenamiento -64,25 frente a -55,53 en predicción para el mismo instante. Por eso su
MAE (47,691) es peor que el del prototipo correcto (47,035). Es el argumento más fuerte a favor
de integrarlo: el error que comete un usuario experto al escribirlo a mano no da ningún aviso.

## Coste de mantenimiento

- Código acotado a `RollingFeatures`: `__init__` (nombres `roll_mean_168_s24`), `_validate_params`
  (`season_length` int, lista o None; `window_sizes % season_length == 0`), `transform_batch`
  (`groupby(fase).rolling(n, closed='left')` reutiliza `_apply_stat_pandas`, todos los estadísticos,
  incluido `ewm`, en orden cronológico), `transform` y `_transform_vectorized` (corte con paso
  `X[-ws::s]`, clave de caché `(ws, s)`), `__repr__` y `_repr_html_`. Prototipo: diferencia máxima
  entre `transform_batch` y `transform` 3e-14 en 8 estadísticos; camino 2D (bootstrapping) correcto.
- Rendimiento: `transform_batch` en 200 000 puntos, 0,057 s frente a 0,007 s contiguo; en predicción
  el corte con paso no cuesta más que el actual.
- Tests: unos 10 a 15 (validación, nombres, equivalencia batch/transform, NaN, `min_periods`, 2D).
- Docs y contexto IA: sección en `window-features-and-custom-features.ipynb`, `llms-base.txt`,
  `skills/feature-engineering/SKILL.md` y `references/rolling-stats-reference.md`,
  `skills/complete-api-reference/references/preprocessing-signatures.md`.
- Interacciones a documentar: en `ForecasterDirect` la ventana se alinea con el origen (t+1), no con
  el paso h, igual que los lags; `window_size` crece (672 descarta las primeras 672 filas y en
  multiserie deja fuera series cortas, cf. ISSUE-1410); `min_periods` pasa a contar puntos.
- No extender a `RollingFeaturesClassification` en esta versión.

## Versión reducida propuesta

1. `RollingFeatures(stats, window_sizes, ..., season_length=None)`, con `window_sizes` = tramo en
   pasos y `season_length` = salto: `window_sizes=168, season_length=24` es la media de
   `y[t-24], y[t-48], ..., y[t-168]` (7 puntos). Sin estadísticos nuevos ni cambios en forecasters.
2. En la guía y en la skill `feature-engineering`: recomendar lags estacionales múltiples
   (48..144, 336...) para cualquier modelo y las medias estacionales solo para modelos de árboles,
   con la advertencia de Direct. Este punto 2 puede ir solo como mejora de docs aunque el 1 se
   aplace.

## Riesgo

Bajo: parámetro nuevo con valor por defecto que conserva el comportamiento actual, sin cambios de
API ni en los forecasters. El riesgo principal es de corrección en `transform` frente a
`transform_batch`; se cubre con un test de equivalencia fila a fila como el del prototipo.
