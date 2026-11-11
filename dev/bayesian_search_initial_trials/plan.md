# Plan: argumento `initial_trials` en las búsquedas bayesianas

Fecha: 2026-11-11. Base: rama `0.27.x` (0.27.0 ya etiquetada como `v0.27.0`; la
funcionalidad va a la 0.28.0). Las rutas son relativas a la raíz del repositorio.

Entorno de las comprobaciones: Python 3.12.3 en Linux, numpy 2.5.4, pandas 2.3.3,
scikit-learn 1.9.1, scipy 1.18.1, numba 0.68.1, optuna 5.0.0. Scripts lanzados en línea,
no se conservan.

## Contexto

`bayesian_search_forecaster`, `bayesian_search_forecaster_multiseries` y
`bayesian_search_foundation` solo reciben el espacio de búsqueda (`search_space`). No hay
forma directa de pedir que se evalúen combinaciones concretas (la configuración de
producción, los valores por defecto del modelo, las mejores de una búsqueda anterior).

Hoy solo se consigue con un rodeo poco evidente (crear un `study` en un storage, llamar a
`study.enqueue_trial` y pasarlo con `load_if_exists`), con trampas comprobadas (optuna
5.0.0):

- Una clave que no existe en el `search_space` se ignora sin avisar.
- Un valor categórico (o de `lags`) que no está entre las opciones aborta la búsqueda a
  mitad, con un `ValueError` de Optuna poco claro.
- Los trials encolados se cuentan como "trials anteriores" y quedan fuera del aviso de
  `n_evaluated` y del recuento de avisos.

Resultado buscado: un argumento `initial_trials` que garantiza que esas combinaciones se
evalúan primero, validadas antes de empezar. El valor es la garantía (el resultado nunca
es peor que la combinación dada, con la misma validación), no acelerar la búsqueda: en el
experimento con `n_trials=8` el MAE medio no mejoró (0.9289 con combinación inicial,
0.9210 sin ella).

## Paso previo

La versión 0.27.0 ya está publicada (existe el tag `v0.27.0`, `releases.md` la fecha el
11 de noviembre de 2026 y `__version__` sigue en `0.27.0`). No hay sección "en desarrollo"
para la nota de versión. Antes de implementar hay que ejecutar `/release-bump 0.28.0` y
trabajar sobre la rama de versión resultante, o elegir otra rama y sección.

## Decisiones de diseño

| Decisión | Elección | Motivo |
|---|---|---|
| Nombre | `initial_trials: dict \| list[dict] \| None = None` | Hace pareja con `n_trials` y no sugiere que excluya `lags`. |
| Posición en la firma | Al final, después de `kwargs_study_optimize` | Las llamadas posicionales existentes siguen funcionando. |
| Mecanismo | `study.enqueue_trial(params, skip_if_exists=True)` | API pública de Optuna (>=4.0). No repite combinaciones, tampoco al reanudar un study guardado. |
| Presupuesto | Cuentan dentro de `n_trials` | Convención de Optuna, Ray Tune y hyperopt. `ValueError` si hay más combinaciones que `n_trials`. |
| Combinaciones | Completas: un valor por cada parámetro del `search_space` | Ver abajo. Relajarlo más adelante es compatible hacia atrás. |
| Clave que el `search_space` no usa | `ValueError` antes de buscar | Optuna la ignora en silencio. |
| Categórico o `lags` fuera de las opciones | `ValueError` antes de buscar | Optuna no puede representarlo; hoy rompe la búsqueda a mitad. |
| Numérico fuera de rango | Aviso y se evalúa | Es el comportamiento de Optuna; su aviso llega agrupado por `_SearchWarnings`. Comprobado que TPE multivariante sigue funcionando con ese valor en el historial (30 trials). |
| `results` | Sin columnas nuevas | Los trials iniciales son los primeros `trial_number` de la llamada. |

### Por qué combinaciones completas, validadas con `optuna.trial.FixedTrial`

Cambio respecto al borrador inicial, que permitía combinaciones parciales:

- Con combinaciones parciales en espacios condicionales la validación es arbitraria: se
  midió que `{'rate_drop': 0.3}` (parámetro que solo existe con
  `booster='dart'`) se marcaba como "clave no usada" en 3 de 8 semillas.
- Una combinación parcial es en parte aleatoria, así que debilita la garantía que
  justifica la función. Los casos de uso reales (producción, valores por defecto,
  `study.best_params`) son combinaciones completas.
- `FixedTrial(params)` ejecuta el `search_space` real sin crear un study: no hay sampler,
  ni semilla, ni mensajes de Optuna que silenciar. Falla si falta un parámetro o si un
  categórico no está entre las opciones, y expone `.params` y `.distributions`.
  Comprobado en optuna 5.0.0.

### Por qué validar antes y no dentro de `_objective`

- Hay 5 closures `_objective`; un helper único evita repetir la comprobación.
- `Trial.system_attrs` (donde Optuna guarda los valores fijados) ya no existe en optuna
  5.0: dentro del objetivo no hay forma pública de saber qué valores eran fijos.
- Falla antes de cualquier backtesting costoso, antes de abrir `output_file` y antes de
  `optuna.create_study`: un error no deja un study creado en un storage persistente.

## Ejemplo de uso: la configuración por defecto del regresor

```python
forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(random_state=123, verbose=-1), lags=24
)

def search_space(trial):
    return {
        'lags'         : trial.suggest_categorical('lags', [24, 48, [1, 2, 3, 23, 24]]),
        'n_estimators' : trial.suggest_int('n_estimators', 50, 500),
        'max_depth'    : trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
    }

defaults = forecaster.estimator.get_params()
initial_trials = {
    'lags'         : 24,                          # una de las opciones de `lags`
    'n_estimators' : defaults['n_estimators'],    # 100
    'max_depth'    : defaults['max_depth'],       # -1, fuera de rango: aviso y se evalúa
    'learning_rate': defaults['learning_rate'],   # 0.1
}

results, study = bayesian_search_forecaster(
    forecaster=forecaster, y=y, cv=cv, search_space=search_space,
    metric='mean_absolute_error', n_trials=20, initial_trials=initial_trials,
)
```

Los parámetros que no están en el `search_space` mantienen en todos los trials el valor
del forecaster que se pasa, así que el trial 0 es "el forecaster tal como está".

Si no se sabe qué lags funcionan (caso habitual, con 3 o 4 opciones de lags), la receta
es cruzar los valores por defecto con todas las opciones. Consume 3 o 4 trials, que
ocupan parte de los 10 trials aleatorios iniciales de TPE, y da la referencia de los
valores por defecto con cada opción de lags:

```python
lags_options = [24, 48, [1, 2, 3, 23, 24]]
params_default = {'n_estimators': 100, 'max_depth': -1, 'learning_rate': 0.1}
initial_trials = [{**params_default, 'lags': lags} for lags in lags_options]
```

Esta receta va en la guía de usuario y en el skill `hyperparameter-optimization`.

Límites comprobados (optuna 5.0.0), que hay que documentar y cubrir con tests:

| Regresor | `get_params()` | Qué ocurre |
|---|---|---|
| LightGBM | Valores numéricos (`max_depth=-1`) | Funciona; `-1` da el aviso de fuera de rango. |
| HistGradientBoosting, RandomForest | `max_depth=None` | Optuna no admite `None` en `suggest_int`/`suggest_float`: `ValueError` en la validación previa. Solución: declarar ese parámetro como categórico, `trial.suggest_categorical('max_depth', [None, 3, 5, 10])`. |
| XGBoost, CatBoost | `None` en todo, o `{}` | `get_params()` no devuelve los valores por defecto: hay que escribirlos a mano (por ejemplo `max_depth=6`, `learning_rate=0.3` en XGBoost). |
| Cualquiera | Valor `<= 0` en una distribución `log=True` | Optuna lo rechaza: `ValueError` en la validación previa. |

No se añade un atajo automático (por ejemplo `initial_trials='forecaster'`): fallaría con
`None` y con XGBoost/CatBoost, y obligaría a igualar `lags` de forma semántica. El
mensaje de error de un valor `None` debe sugerir la solución del categórico.

## Implementación

### 1. Helpers en `skforecast/model_selection/_utils.py` (después de `_optimize_study`)

`_check_initial_trials(initial_trials, search_space, n_trials) -> list[dict]`

1. `None` devuelve `[]`; un `dict` pasa a `[dict]`.
2. `TypeError` si no es un `dict` o una lista de `dict` no vacíos. `ValueError` si hay
   más combinaciones que `n_trials`.
3. Convierte los `numpy ndarray` a lista (la columna `lags` de `results` son arrays).
4. Por combinación: `trial = FixedTrial(params)` y `search_space(trial)`, con los
   warnings silenciados (el de fuera de rango lo emite el trial real).
   - Excepción de `search_space` (falta un parámetro, categórico fuera de las opciones,
     tipo no válido): `ValueError` que nombra `initial_trials[i]`, incluye el error
     original (`from e`) y recuerda que los categóricos, `lags` incluido, deben ser
     iguales a una de las opciones.
   - `set(params) - set(trial.params)` no vacío: `ValueError` con las claves sobrantes y
     las que usa el `search_space`.
5. Normaliza cada valor con su distribución
   (`dist.to_external_repr(dist.to_internal_repr(v))`, métodos públicos): en los
   categóricos devuelve la opción; en los numéricos se usa si es igual al valor dado
   (`np.int64(100)` pasa a `100`, `1` pasa a `1.0`) y si no se deja el original (`100.5`
   en un entero, que Optuna avisa como fuera de rango). Así los valores son serializables
   en storages RDB (`np.int64` falla con sqlite, comprobado) y `skip_if_exists` reconoce
   los duplicados.

`_get_finished_trial_numbers(study) -> set[int]`: números de los trials terminados
(`COMPLETE`, `PRUNED`, `FAIL`).

### 2. Las tres funciones de `skforecast/model_selection/_search.py`

Mismo patrón en las tres:

- Firma y docstring: `initial_trials` al final. El docstring (estilo de
  `.github/instructions/docstrings.instructions.md`) explica: combinaciones completas que
  se evalúan primero; cuentan en `n_trials` y también en los `n_startup_trials` de TPE
  (10 combinaciones iniciales eliminan la fase aleatoria); no se repiten si ya están en
  un study cargado; los valores fuera de rango se evalúan con un aviso; para reutilizar
  una búsqueda anterior, usar `study.best_params` o `study.trials[i].params` (conservan
  `lags` tal como se sugirió). Añadir la referencia en el docstring de `n_trials`.
- `initial_trials = _check_initial_trials(...)` junto a las primeras comprobaciones de
  entrada (antes de configurar el logging y de `optuna.create_study`).
- Sustituir `n_trials_previous = len(study.trials)` por
  `trials_previous = _get_finished_trial_numbers(study)`.
- Como primera instrucción dentro del `try` que envuelve a `_optimize_study` (para que el
  `FileHandler` de `output_file` se cierre si algo falla):

  ```python
  for params in initial_trials:
      study.enqueue_trial(params, skip_if_exists=True)
  ```

- Recuento de "trials de esta llamada" con conjuntos, no con una resta:
  `search_warnings.emit(n_candidates=len(_get_finished_trial_numbers(study) - trials_previous))`
  y, en la función de una serie, el filtro del aviso de `n_evaluated` pasa a
  `~results['trial_number'].isin(trials_previous)` (multiseries ya filtra por conjunto).
  Motivo: con `timeout` o `study.stop()` pueden quedar combinaciones encoladas sin
  ejecutar (`WAITING`); la resta actual las cuenta como ejecutadas y, al reanudar el
  study, las excluye.

No se tocan `_optimize_study` ni los `_objective`.

### 3. Tests

Nuevo `skforecast/model_selection/tests/tests_utils/test_check_initial_trials.py`:

- Salida con `None`, con un `dict` y con una lista.
- `TypeError` si no es `dict` o lista de `dict` no vacíos (parametrizado).
- `ValueError`: más combinaciones que `n_trials`; falta un parámetro; clave no usada
  (incluida una rama condicional); categórico fuera de las opciones (también `lags` como
  tupla en vez de lista); `None` en una distribución numérica (el mensaje sugiere el
  categórico); valor `<= 0` en una distribución `log=True`.
- `None` como opción de un categórico se acepta.
- Valores normalizados: escalares numpy, entero a float, `lags` como array;
  `json.dumps` funciona; sin warnings ni mensajes de Optuna. Fuera de rango no da error.

`tests_search/test_bayesian_search_forecaster.py` (con `Ridge`, como los tests de trials
fallidos):

- Las combinaciones se evalúan primero, con `TimeSeriesFold` y `OneStepAheadFold`: los
  trials 0 y 1 tienen los valores dados, `len(results) == n_trials`, y se acepta un
  único `dict`.
- Un `ValueError` de `initial_trials` salta antes de crear el study: el
  `InMemoryStorage` queda sin studies y no se crea `output_file`.
- Study cargado de storage: sin duplicados y recuento "of N trials" correcto.
- Combinaciones no ejecutadas no se cuentan: un callback llama a `study.stop()`
  (determinista, a diferencia de `timeout`).
- Valor fuera de rango: se evalúa y aparece el aviso.

`test_bayesian_search_forecaster_multiseries.py`: los dos primeros, más ampliar el test
de storage existente. `test_bayesian_search_foundation.py`: evaluadas primero y el
`ValueError` (estos tests usan un pipeline simulado, no necesitan torch).

### 4. Documentación y contexto de IA

- `docs/user_guides/hyperparameter-tuning-and-lags-selection.ipynb`: una celda de texto
  y una de código tras el ejemplo bayesiano (la configuración por defecto del regresor,
  con sus límites, y reutilizar `study.best_params`). Ejecutar y comprobar:
  `python tools/docs/execute_notebooks/execute_notebooks.py user_guides/hyperparameter-tuning-and-lags-selection.ipynb`
  y `python tools/docs/execute_notebooks/execute_notebooks.py --check`.
- `docs/releases/releases.md`: entrada en "Added" de la versión en desarrollo (skill
  `release-note`), tras el paso previo.
- Fuentes de contexto de IA (nunca los ficheros generados): `tools/ai/llms-base.txt`,
  `skills/complete-api-reference/references/model-selection-signatures.md` (las tres
  firmas), `skills/hyperparameter-optimization/SKILL.md` y
  `references/search-parameters.md`, `skills/foundation-forecasting/SKILL.md`. Después
  `python tools/ai/generate_ai_context_files.py` y lo mismo con `--check`.

## Fuera del alcance

- Combinaciones parciales (se puede añadir después sin romper compatibilidad).
- Igualar `lags` de forma semántica (`24`, `range(1, 25)` y un array equivalentes): se
  resuelve con la receta de `study.best_params`.

## Verificación

1. `pytest skforecast/model_selection/tests/tests_utils/test_check_initial_trials.py -x -q`
2. `pytest skforecast/model_selection/tests/tests_search/test_bayesian_search_forecaster.py -x -q`,
   y lo mismo para `_multiseries` y `_foundation`; después
   `pytest skforecast/model_selection/tests/tests_search/ -q -k bayesian` una vez.
3. Script manual de extremo a extremo: combinación válida, duplicada, con errata, con
   `lags` fuera de las opciones y fuera de rango; y storage sqlite reanudado
   (`load_if_exists=True`) para ver que no se repiten.
4. Skill `verify` (ruff, tests afectados, contexto de IA, notas de versión) antes de dar
   el cambio por terminado y antes del commit.
