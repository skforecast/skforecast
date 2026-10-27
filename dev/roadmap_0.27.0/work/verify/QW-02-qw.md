# QW-02: aviso cuando un forecaster foundation ajustado con exog predice sin exog

**Veredicto: keep_modified** (esfuerzo S confirmado).

## Qué se ha comprobado

- **Fricción real, medida.** Chronos-2 small, serie diaria con `promo` (efecto +5), `fit(series=y, exog=promo)`:
  `predict(steps=7)` no emite ningún aviso y predice 10,08 / 10,03 los días con promo, frente a 15,08 / 15,27 con `exog`.
  `ForecasterRecursive` en el mismo caso lanza `ValueError: Forecaster trained with exogenous variable/s...`.
  Con el forecaster falso de los tests: ML -> `ValueError`; FM -> `OK shape=(3, 2)` sin aviso.
- **No existe, y además el silencio es una decisión documentada y testeada**, no un descuido:
  - `skforecast/foundation/_foundation_model.py:973-975`: solo se avisa si `supports_past_only_covariates=False`.
  - `skills/foundation-forecasting/references/adapter-parameters.md:226`: "Used as a past-only covariate, no warning" (caso `exog=None`).
  - `skforecast/foundation/tests/tests_foundation_models/test_check_exog_columns.py:182-201`: `test_check_exog_columns_no_warning_for_past_only_adapters` exige cero avisos con `exog=None` y `{"s1": None}`.
  - También lo describen `skills/foundation-forecasting/SKILL.md:128,277`, `tools/ai/llms-base.txt:512` y la tabla de `docs/user_guides/foundation-forecasting-with-heterogeneous-series.ipynb` (celda markdown).
  Por tanto la propuesta invierte un comportamiento documentado: necesita el visto bueno del mantenedor, pero el argumento es sólido (el caso habitual de exog en skforecast es exog conocida a futuro: calendario, promos, festivos; las covariables solo pasadas intencionadas son un caso avanzado que no aparece en ningún ejemplo de la documentación). Otras librerías obligan a declarar qué covariables son futuras (AutoGluon `known_covariates_names`, Darts `past_covariates`/`future_covariates`); skforecast lo infiere por presencia de columnas, de ahí el riesgo de olvido silencioso.
- **Prototipo** (worktree desechable, 20 líneas en `_check_exog_columns`, diff en scratchpad `qw02/prototype.diff`):
  - Aviso en `predict`, `predict_quantiles` (y por tanto `predict_interval`) y en `predict(context=..., context_exog=...)` sin `exog`.
  - Sin aviso con `exog`, sin aviso si el ajuste fue sin exog, sin aviso en `backtesting_foundation` (usa `check_inputs=False` y siempre pasa la exog del horizonte).
  - Tests: `skforecast/foundation/tests` + tests de validación filtrados: 1028 pasan, 6 fallan, todos parametrizaciones del test que exige silencio (casos `None` y `dict_of_None`). Ningún notebook con salidas guardadas predice sin exog tras ajustar con exog, no hay que reejecutar nada.

## Cambios respecto a la propuesta

1. **Dónde:** dentro de `FoundationModel._check_exog_columns` (ya recibe `context_exog` y `exog` normalizados y solo corre con `check_inputs=True`), sin nuevo parámetro. Condición: `exog is None` o todos los valores `None`, adaptador con `supports_past_only_covariates=True` y alguna serie con columnas históricas. Si el usuario pasa un subconjunto de columnas no se avisa (mezcla consciente de pasadas y futuras).
2. **Ámbito:** aplicar también cuando se pasa `context` + `context_exog` sin `exog` (misma semántica, misma regla, más simple de documentar), no solo con `context is None`.
3. **Categoría:** `MissingExogWarning` (`skforecast/exceptions/exceptions.py:213`), que ya añade cómo silenciarla; el mensaje debe decir que, si son covariables solo pasadas a propósito, se puede ignorar.
4. **stacklevel:** usar 4 cuando se llama vía `ForecasterFoundation` para que apunte al código del usuario (hoy el `IgnoredArgumentWarning` vecino apunta a `_forecaster_foundation.py:832`). Detalle menor.
5. **Fuera de este QW:** `suppress_warnings` en `ForecasterFoundation.predict*` (los forecasters ML lo tienen, `_forecaster_recursive.py:1036`): es un hueco de consistencia aparte, no bloquea.

## Ficheros a tocar

`_foundation_model.py` (bloque + Notes de `_check_exog_columns` y de `predict`), `test_check_exog_columns.py` (dividir el test: subconjunto sin aviso; `None`/`dict_of_None` con `MissingExogWarning`), `SKILL.md:128,277`, `adapter-parameters.md:226`, `llms-base.txt:512`, tabla markdown del notebook heterogéneo, `releases.md`, regenerar contexto IA.

## Riesgo de API

Solo un aviso nuevo; no cambia salidas, columnas ni valores por defecto. Quien use covariables solo pasadas a propósito verá el aviso en cada `predict` (se silencia con `warnings.simplefilter('ignore', category=MissingExogWarning)`). Rompería solo suites con `-W error` en ese caso concreto.
