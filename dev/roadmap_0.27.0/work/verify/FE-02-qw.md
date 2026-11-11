# FE-02: covariables solo pasadas en `backtesting_foundation` (`past_only_exog`)

**Veredicto: keep_modified** (esfuerzo S confirmado), prioridad media-baja como funcionalidad,
pero destapa un **bug nuevo de mayor prioridad** en TabICL (y por código en TabPFN-TS) que
conviene arreglar antes o en la misma PR (ver abajo).

## Qué se ha comprobado

- **No existe.** Ni `skforecast/model_selection/_validation.py` ni `_search.py` mencionan
  `past_only`. `_backtesting_foundation` pasa en cada fold `exog = exog_test` con todas las
  columnas (`_validation.py:2521`, `2596`, `2606`) y el docstring exige que `exog` cubra el
  horizonte (`_validation.py:2413`, `2775`). `bayesian_search_foundation` reenvía `exog` tal cual
  (`_search.py:2328-2339`). En `predict` sí existe el modo: `get_exog_signature`
  (`foundation/_utils.py:444-482`) trata como solo pasada toda columna de `context_exog` que no
  esté en `exog`; lo anuncian `skills/foundation-forecasting/SKILL.md:128` y
  `tools/ai/llms-base.txt:512`.
- **Fricción real, medida** (`scratchpad/fe02/before_after.py`, Chronos-2 small en CPU, serie
  diaria de 600 puntos, `x_observed` adelanta 3 días a `y`, `weekend` conocida a futuro, 20 folds
  de 7 pasos):

  | Configuración | MAE |
  |:--|--:|
  | `backtesting_foundation` con `x_observed` como futura (usa el futuro) | 1,186 |
  | Bucle manual honesto: `x_observed` solo pasada + `weekend` futura | 2,919 |
  | Simulación de `past_only_exog=['x_observed']` (quitar la columna de `exog_test`) | 2,919 (diferencia máxima con el bucle: 0,0) |
  | Solo `weekend` (descartar `x_observed`) | 3,104 |
  | Sin exog | 3,244 |

  El backtest con la columna como futura sobrestima la precisión 2,5 veces; usarla como solo
  pasada mejora un 6% frente a descartarla. Hoy la única forma de medir esto es el bucle manual,
  y no hay forma de usar `bayesian_search_foundation` (por ejemplo para elegir `context_length`)
  con covariables solo pasadas.
- **Matiz sobre la "fuga de información":** no es un fallo de skforecast. El contrato de `exog` en
  todos los backtestings es "conocida a futuro"; pasar una variable observada a posteriori es un
  error de modelado igual que en `backtesting_forecaster`. Lo correcto es presentarlo como hueco:
  `predict` ofrece un modo que el backtesting y la búsqueda no permiten evaluar. Coincide con lo
  que hacen otras librerías al declarar el tipo de covariable (Darts `past_covariates`, AutoGluon
  `known_covariates_names`).

## Bug nuevo encontrado (no estaba en la lista)

**TabICL no ignora las covariables solo pasadas aunque el aviso diga que sí.**
`FoundationModel._check_exog_columns` (`_foundation_model.py:973-993`) emite
`IgnoredArgumentWarning` ("Historical exog columns without future values are ignored") para
adaptadores con `supports_past_only_covariates=False`, pero nadie elimina esas columnas:
`TabICLAdapter._build_context_df` (`_adapters.py:2679-2685`) las mete en el contexto y el futuro
no las tiene, así que el backend las recibe como NaN en el horizonte.

Medido (`scratchpad/fe02/tabicl_check.py`, TabICL, serie con `y` real alrededor de -16):

- `predict(steps=7, context=y, context_exog=exog)` (con aviso de "ignorada"): `[5.21 5.21 5.22 5.2 5.18 5.14 5.06]`
- `predict(steps=7, context=y)` (sin exog): `[-17.29 -18.07 -18.46 -18.68 -18.66 -18.51 -18.22]`
- Backtest de 5 folds con la columna solo pasada (`scratchpad/fe02/edge.py`): MAE 15,5 frente a 3,6 sin exog.

Afecta a quien ajusta TabICL con exog y predice sin ella o con un subconjunto de columnas.
`TabPFNAdapter._build_context_df` (`_adapters.py:3347-3353`) tiene el mismo patrón (no medido);
T0 (`_adapters.py:3869-3878`) y Nori (`_known_future_columns`, `_adapters.py:4795-4799`) sí
filtran bien. Arreglo centralizado (regla de `foundation.md`): en `FoundationModel.predict`, si
`not adapter.supports_past_only_covariates`, quitar de `context_exog` las columnas sin futuro
antes de llamar al adaptador. Sin este arreglo, FE-02 con TabICL o TabPFN-TS daría basura en
silencio, porque el backtesting usa `check_inputs=False` y ni siquiera sale el aviso.

## Cambios respecto a la propuesta

1. Presentarlo como hueco (evaluar y ajustar covariables solo pasadas), no como fuga.
2. Parámetro `past_only_exog: str | list[str] | None = None` (nombres de columna) en
   `backtesting_foundation` y `bayesian_search_foundation`. Validar que hay `exog` y que cada
   columna existe en la exog de al menos una serie (`ValueError` si no).
3. En `_backtesting_foundation`, por serie: `exog_test[k].drop(columns=..., errors='ignore')`,
   y `None` si queda vacía (probado: con todas las columnas solo pasadas funciona y Chronos-2 da
   MAE 3,675 frente a 3,681 sin exog). `context_exog` las conserva.
4. Si `not forecaster.supports_past_only_covariates`: `IgnoredArgumentWarning` y quitar también
   esas columnas de `context_exog` (o depender del arreglo centralizado del bug anterior).
5. Números de la evidencia: usar los de esta verificación (1,186 / 2,919 / 3,104).

## Esfuerzo y ficheros

`_validation.py` (firma, validación, 10 líneas en el bucle, docstrings de ambas funciones),
`_search.py` (reenvío y docstring), unos 30 a 40 líneas de código. Tests: igualdad con bucle
manual (con el adaptador falso de los tests), validación de columnas, aviso con adaptador sin
soporte, reenvío en la búsqueda. Documentación: `SKILL.md` de foundation-forecasting,
`llms-base.txt`, `releases.md`, regenerar contexto IA. Medio día a un día.

## Riesgo de API

Parámetro nuevo con valor por defecto `None`: no cambia salidas ni columnas. Relación con QW-02
(aviso al predecir sin exog): compatible, el backtesting usa `check_inputs=False`.
