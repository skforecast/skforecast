# Handoff: lo que queda de la revisión de `skforecast/utils/utils.py`

**Fecha:** 2026-10-07, sobre las 20:40 UTC.
**Escrito por:** la sesión de Claude Code de Joaquín Amat Rodrigo, que hizo la revisión y los PRs 1a a 5b.
**Para:** quien siga el trabajo en su propia sesión de Claude Code.
**Base comprobada:** `0.26.x` en `18e5dbaf0` (merge de #1363, 2026-10-07 20:24 UTC). Todas las reproducciones de este documento se han ejecutado sobre ese commit. Entorno: Python 3.12, pandas 2.3.3, numpy 2.5.3, scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.4.1 y catboost 1.2.10.

> **Cómo usar este documento**
> 1. Guárdalo en tu clon como `dev/handoff_utils_pendiente.md`. `CLAUDE.md` le pide a Claude leer un `dev/handoff_*.md` cuando se lo indicas. No lo subas al repo.
> 2. Abre la sesión en la raíz del repo y pide, por ejemplo: «Lee `dev/handoff_utils_pendiente.md` y haz la tarea T2».
> 3. Las tareas son independientes. Cada una trae su reproducción, la causa, el arreglo propuesto, los tests y la nota de versión.
> 4. Antes de empezar una tarea, vuelve a ejecutar su reproducción sobre el `0.26.x` del momento, porque puede haber cambiado.
> 5. Las tareas marcadas «decisión de Joaquín» no se implementan hasta que él elija una opción. La recomendada va primero.

---

## 1. Estado

- **Rama de desarrollo:** `0.26.x` (versión 0.26.0, en desarrollo). Las ramas de trabajo salen de `origin/0.26.x` y los PRs van contra `0.26.x`, nunca contra `main`.
- **Fusionados en esta revisión:**

| PR | Qué arregló |
|---|---|
| skforecast/skforecast#1344 (1a) | Atajos de predicción de `_build_predict_function`: XGBoost con early stopping y `missing`; subclases de modelos de scikit-learn |
| skforecast/skforecast#1345 (1b) | `ForecasterDirect` y `ForecasterDirectMultiVariate` con `differentiation` y `steps` no consecutivos |
| skforecast/skforecast#1346 (1c) | Validación de la exog en `predict` (frecuencia, huecos, exog ancha multiserie alineada por fecha), `last_window` con varias columnas |
| skforecast/skforecast#1348 (6) | `pandas>=2.2` y `scikit-learn>=1.6`, `cast_exog_dtypes` eliminada, `show_versions` ampliada |
| skforecast/skforecast#1349 (2a) | CatBoost con categóricas en `predict` del clasificador, códigos de niveles multiserie, dispositivo del estimador |
| skforecast/skforecast#1350, #1352, #1353 (3a, 3b, 3c) | `save_forecaster`/`load_forecaster`: nombres con puntos, skops (índices con zona horaria, `RollingFeatures`, categóricas, pyarrow, `DateOffset`), `weight_func` exportada con imports, `partial` |
| skforecast/skforecast#1360 (7) | Job de CI con las versiones mínimas de las dependencias |
| skforecast/skforecast#1361 (4) | `transform_series`/`transform_dataframe`, `date_to_index_position`, `steps` y `levels` de otros tipos |
| skforecast/skforecast#1362 (5a) | Validación de entradas: lags y `window_sizes` numpy, `fit_kwargs` sin modificar, dtypes nullable y pyarrow, zonas horarias, exog multiserie, residuos por nivel, `last_window` de `ForecasterRnn`, errores de importación |
| skforecast/skforecast#1363 (5b) | Rendimiento: `predict` multiserie (×4,9 con 5000 series), `check_predict_input`, `multivariate_time_series_corr` con `corrwith`, `ExtraTrees` en el atajo por árbol; `deepcopy_forecaster` sin modificar el original; docstrings |

- **Abiertos:** ninguno de esta revisión. Al revisar el rendimiento de #1363 antes de fusionarlo apareció una regresión, que ya está en `0.26.x` sin publicar: es la tarea T1.
- **Ramas remotas que quedan:** `ci/minimum-versions-job` y `fix/weight-func-export`. Las borra Joaquín.

---

## 2. Reglas de trabajo

**Las del repo.** Están en `CLAUDE.md` y `tools/ai/ai_context_header.md`, y Claude las carga solo. Lo esencial:
- Commits y PRs solo con la identidad del autor: sin `Co-Authored-By` de una IA, sin trailer de sesión y sin «Generated with Claude Code». Un hook lo bloquea.
- No se puede hacer force push. Si hay que corregir algo ya subido, se añade un commit nuevo encima.
- Tests en secuencia (nunca `-n`) y nunca la suite completa. Se ejecutan los tests del código tocado; el skill `verify` dice cuáles.
- Antes de dar algo por terminado, el skill `verify`. Para la nota de versión, el skill `release-note`, que escribe en `docs/releases/releases.md`, sección 0.26.0. Si cambia la API pública o una skill de `skills/`, el skill `ai-context-sync`.
- Docstrings NumPy, líneas de 88 como máximo y sin guiones largos (en dash ni em dash) en comentarios, docstrings ni documentación.
- En las sesiones cloud no hay torch ni keras: `SKFORECAST_CLOUD_DL=1`, o `uv pip install torch "keras>=3.3,<4.0" --torch-backend cpu`. Hay fallos previos que no son tuyos: `foundation` da 10 fallos sin torch y `deep_learning` da 3 en `test_fit_tensorflow.py` sin TensorFlow.

**Las de Joaquín.** Las ha corregido a mano en PRs anteriores:
- Una asignación por línea: nada de `a, b = x, y`, aunque desempaquetar lo que devuelve una función sí vale. Nombres que digan qué hace el argumento.
- Exactamente dos líneas en blanco entre tests.
- Un commit por problema y un último commit con las notas de versión. En la nota, una sola entrada por función o problema; las sub-viñetas `    + ` valen.
- Flujo de cada PR:
  - Implementar en local y enseñar un resumen (commits, tests y desviaciones del plan).
  - Push y PR **en borrador** contra `0.26.x`, solo con su OK.
  - Descripción del PR con `## Description` y `## Verification`, sin el pie «Generated by Claude Code» si el servidor lo añade.
  - Él lo pasa a «ready for review» y lo fusiona con merge commit.
  - A veces sube commits suyos a la rama: haz `git fetch` y revísalos antes de seguir.
- Los cambios de diseño se le presentan con mediciones y una recomendación, y él decide. Antes de cambiar un comportamiento, comprobar qué hacen hoy `fit`, el backtesting y los otros formatos de entrada, y contar en la nota de versión el caso de usuario que cambia.
- Al acabar cada PR pide una revisión final exhaustiva:
  - una prueba de propiedades contra una referencia (por ejemplo, la función antigua);
  - los tests de cada commit pasando en su propio worktree;
  - el rendimiento comparado con la base;
  - el caso de usuario que cambia.

**Lecciones de esta revisión:**
- Probar configuraciones raras: una serie sin lags, transformadores, diferenciación por serie, categóricas, índices con zona horaria, dtypes nullable (`Float64`, `Int64`) y pyarrow, y entradas largas. T1 salió por no medir con tipos nullable largos.
- Rendimiento: comparar la base y el cambio en worktrees separados, con ejecuciones intercaladas y quedándose con el mínimo, porque una sola ejecución varía hasta un 20 %.
- Los tests nuevos deben fallar en la base. Para comprobarlo sin perder el test, usa `git stash push <fichero>` solo con el código.

---

## 3. Resumen de lo pendiente

| Tarea | Hallazgos | Gravedad | ¿Decide Joaquín? | Esfuerzo | PR sugerido |
|---|---|---|---|---|---|
| **T1** | N-19: `check_predict_input` es mucho más lento con `exog` o `last_window` largos de tipos nullable o pyarrow (regresión de #1363, sin publicar) | Media (rendimiento) | No | Bajo | `fix/check-predict-input-nullable-perf` |
| **T2** | M-17, S-03: las categóricas nativas no llegan al estimador dentro de `CalibratedClassifierCV` o `TransformedTargetRegressor` | Media | No | Medio-alto | PR 2b `fix/categorical-wrapped-estimators` |
| **T3** | A-05: `Pipeline` con LightGBM o CatBoost y exog categórica: `fit` falla | Alta (error claro) | **Sí** | Medio | PR 2b |
| **T4** | M-12: se pierde la configuración categórica del usuario y salen avisos de más | Media | **Sí** | Medio | PR 2b |
| **T5** | N-17: en multiserie, una exog con las fechas en orden descendente se descarta y el modelo se entrena sin ella | Media | Recomendación: ordenar | Bajo | PR de arreglos pequeños |
| **T6** | N-16, N-18: `ForecasterStats` con `last_window` de otro nombre, o `last_window_exog` como Series | Baja | No | Bajo | PR de arreglos pequeños |
| **T7** | N-04: doble aviso de NaN en multiserie con exog ancha | Baja (ruido) | No | Bajo | PR de arreglos pequeños |
| **T8** | N-07: la skill de multiserie dice que la exog y las series deben tener el mismo formato | Baja (docs IA) | No | Muy bajo | PR de arreglos pequeños |
| **T9** | N-08: imports sin usar en tests | Trivial | No | Muy bajo | Junto a cualquier PR que toque esos ficheros |

Orden recomendado:
1. T1, porque es una regresión que no debe llegar a la 0.26.0.
2. T2 a T4, que forman el PR 2b.
3. T5 a T8, que forman el PR de arreglos pequeños, con un commit por tarea.

El PR 2b debe empezar por T2, que no necesita decisiones, mientras Joaquín decide T3 y T4.

---

## 4. Tareas

### T1 · `check_predict_input` lento con tipos nullable o pyarrow (N-19)

**Contexto.** El commit `05341f2` de #1363 («Speed up check_predict_input», hallazgo O-05, fusionado en `18e5dbaf0`) cambió la búsqueda de NaN de `df.isna().to_numpy().any()` a `pd.isna(df.to_numpy()).any()`. Con columnas `float64` es más rápido. Pero `DataFrame.to_numpy()` de varias columnas de tipo extensión (`Float64`, `Int64`, `double[pyarrow]`) crea un array `object`, y entonces es mucho más lento. En `last_window`, además, se convierte la tabla entera antes de cortar las últimas `window_size` filas.

**Medido** (4 núcleos, mínimo de 3 ejecuciones intercaladas):

| Comprobación de NaN, 5 columnas | Antes (`df.isna()`) | #1363 (`pd.isna(to_numpy())`) |
|---|---|---|
| `float64`, 24 filas | 10 µs | 3 µs |
| `float64`, 100.000 filas | 183 µs | 166 µs |
| `Float64`, 10.000 filas | 47 µs | 1813 µs (×38) |
| `Float64`, 100.000 filas | 92 µs | 22.744 µs (×247) |
| `double[pyarrow]`, 100.000 filas | 82 µs | 24.075 µs (×294) |
| 4 `float64` + 1 categórica, 100.000 filas | 814 µs | 1211 µs (×1,5) |

| Caso de usuario | `0.26.x` antes de #1363 | Con #1363 | Con #1363 y el arreglo |
|---|---|---|---|
| `check_predict_input`, exog de 24 filas `float64` | 112 µs | 80 µs | 98 µs |
| `ForecasterRecursiveMultiSeries.predict(24)`, 5000 series | 379 ms | 83 ms | 87 ms |
| `ForecasterRecursiveMultiSeries.predict(24)`, 50 series con exog | 3,8 ms | 2,9 ms | 3,0 ms |
| `ForecasterRecursive.predict(24)`, exog `Float64` de 100.000 filas | 13,3 ms | 29,3 ms | 13,4 ms |
| `ForecasterRecursiveMultiSeries.predict(24)`, `last_window` `Float64` de 50.000×50 | 5,5 ms | 116 ms | 7,0 ms |

**Reproducción** (`n19_predict.py`). En `9c6f0fd`, antes de #1363, da `13.5 ms`; en `18e5dbaf0`, `31.6 ms`; con el arreglo, `12.0 ms`.

```python
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.recursive import ForecasterRecursive

warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
idx = pd.date_range("2020-01-01", periods=1000, freq="h")
y = pd.Series(rng.normal(size=1000), index=idx, name="y")
cols = [f"e{i}" for i in range(5)]
exog = pd.DataFrame(rng.normal(size=(1000, 5)), index=idx, columns=cols).astype("Float64")
exog_pred = pd.DataFrame(
    rng.normal(size=(100_000, 5)), columns=cols,
    index=pd.date_range(idx[-1] + idx.freq, periods=100_000, freq="h"),
).astype("Float64")
forecaster = ForecasterRecursive(LinearRegression(), lags=24)
forecaster.fit(y, exog=exog)
times = []
for _ in range(5):
    start = time.perf_counter()
    forecaster.predict(24, exog=exog_pred)
    times.append(time.perf_counter() - start)
print(f"predict(24), exog Float64 with 100000 rows: {min(times) * 1e3:.1f} ms")
```

**Arreglo probado** sobre `18e5dbaf0`: pasan los 201 tests de `test_check_predict_input.py` y de los `test_predict.py` de `ForecasterRecursive` y `ForecasterRecursiveMultiSeries`. Hay que hacer dos cambios en `check_predict_input` (`skforecast/utils/utils.py`):
- En `last_window`, cortar las filas antes de `to_numpy`, y solo cuando sobran. Así, el `last_window_` guardado, que ya tiene `window_size` filas, no paga el `iloc`.
- En `exog`, volver a `DataFrame.isna`, como antes de #1363. El `exog` puede ser largo y de cualquier tipo.

```diff
     last_window_to_check = last_window
+    if forecaster_name != 'ForecasterStats' and len(last_window) > window_size:
+        last_window_to_check = last_window_to_check.iloc[-window_size:]
     if forecaster_name == 'ForecasterRecursiveMultiSeries':
         last_window_to_check = last_window_to_check[levels]
     elif forecaster_name in ['ForecasterDirectMultiVariate', 'ForecasterRnn']:
         last_window_to_check = last_window_to_check[series_names_in_]
     # NOTE: `pd.isna` on the numpy values is faster than `DataFrame.isna` for
-    # the small inputs used to predict.
-    last_window_values = last_window_to_check.to_numpy()
-    if forecaster_name != 'ForecasterStats':
-        last_window_values = last_window_values[-window_size:]
-    if pd.isna(last_window_values).any():
+    # the small inputs used to predict. The rows are cut before `to_numpy`
+    # because it creates an object array with extension dtypes (nullable, pyarrow).
+    if pd.isna(last_window_to_check.to_numpy()).any():
 ...
-            if pd.isna(exog_to_check.to_numpy()).any():
+            if exog_to_check.isna().to_numpy().any():
```

El `iloc` va antes de seleccionar las columnas para que la selección copie solo las filas de la ventana.

**Tests.** El comportamiento no cambia y los tests de NaN de `test_check_predict_input.py` ya lo cubren, también con dtypes nullable. Un test opcional: un `last_window` `Float64` largo con NaN solo fuera de la ventana no avisa, y con NaN dentro sí.

**Nota de versión.** Corregir dos entradas de Changed que añadió #1363:
- La de «The prediction methods of all the forecasters check their inputs faster…»: volver a medir la base y el arreglo, con ejecuciones intercaladas y el mínimo, y poner las cifras nuevas.
- La de `multivariate_time_series_corr`: la mejora con `pearson` («19 ms instead of 22 ms») no se repite (23,5 ms antes y 23,3 ms después). Quitar esa cifra o decir que no cambia; las de `spearman` y `kendall` sí se mantienen.

**Dónde va.** Rama nueva desde `origin/0.26.x`, por ejemplo `fix/check-predict-input-nullable-perf`, con dos commits: el arreglo y la nota de versión. Título sugerido: *Check missing values of long nullable inputs faster in check_predict_input*.

**Un dato más para la descripción del PR.** En #1363 se dijo que el último `last_window` multiserie conserva los dtypes («dtypes kept»), pero eso solo vale para los tipos numpy: las series nullable dan columnas `float64` (antes `Float64`), con predicciones idénticas. No hay que cambiar nada; solo no repetir esa afirmación.

---

### T2 · Categóricas nativas dentro de `CalibratedClassifierCV` y `TransformedTargetRegressor` (M-17, S-03)

**Problema.** Los helpers de categóricas solo desenvuelven un `Pipeline`:
- `ForecasterRecursiveClassifier` con `CalibratedClassifierCV(LGBM | XGB | HGB | CatBoost)` dice `use_native_categoricals=True`, pero los lags, que son códigos de clase, llegan al modelo como numéricos.
- Con `TransformedTargetRegressor` alrededor de un modelo con categóricas nativas, la exog categórica llega también como numérica.
- No hay error ni aviso. Afecta al ejemplo de la guía de clasificación (`autoregressive-classification-forecasting.ipynb`, celda 45, `CalibratedClassifierCV(HistGradientBoostingClassifier)`).

**Reproducción** (`m17_s03.py`):

```python
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import TransformedTargetRegressor
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveClassifier

warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
idx = pd.date_range("2020-01-01", periods=200, freq="D")


def lgbm_feature_kinds(model):
    infos = model.booster_.dump_model()["feature_infos"]
    return ["cat" if len(v["values"]) > 0 else "num" for v in infos.values()]


# M-17: the lags of the classifier are class codes and should be categorical
y_class = pd.Series(rng.choice(["lo", "mid", "hi"], 200), index=idx, name="y")
for estimator in [LGBMClassifier(verbose=-1, n_estimators=10),
                  CalibratedClassifierCV(LGBMClassifier(verbose=-1, n_estimators=10), cv=3)]:
    forecaster = ForecasterRecursiveClassifier(estimator, lags=3)
    forecaster.fit(y_class)
    inner = forecaster.estimator
    if isinstance(inner, CalibratedClassifierCV):
        inner = inner.calibrated_classifiers_[0].estimator
    print(f"M-17 {type(estimator).__name__:22} use_native_categoricals={forecaster.use_native_categoricals} "
          f"lags={lgbm_feature_kinds(inner)}")

# S-03: the categorical exog should reach LightGBM as categorical
cat = pd.Series(rng.choice(["a", "b", "c"], 200), index=idx)
y = pd.Series(rng.normal(size=200) + cat.map({"a": 0, "b": 5, "c": 10}).to_numpy() + 20, index=idx, name="y")
exog = pd.DataFrame({"cat": cat.astype("category")}, index=idx)
for estimator in [LGBMRegressor(verbose=-1, n_estimators=10),
                  TransformedTargetRegressor(LGBMRegressor(verbose=-1, n_estimators=10), func=np.log, inverse_func=np.exp)]:
    forecaster = ForecasterRecursive(estimator, lags=3)
    forecaster.fit(y, exog=exog)
    inner = forecaster.estimator
    if isinstance(inner, TransformedTargetRegressor):
        inner = inner.regressor_
    print(f"S-03 {type(estimator).__name__:26} features (lag_1, lag_2, lag_3, cat)={lgbm_feature_kinds(inner)}")
```

Salida en `18e5dbaf0`:

```text
M-17 LGBMClassifier         use_native_categoricals=True lags=['cat', 'cat', 'cat']
M-17 CalibratedClassifierCV use_native_categoricals=True lags=['num', 'num', 'num']
S-03 LGBMRegressor              features (lag_1, lag_2, lag_3, cat)=['num', 'num', 'num', 'cat']
S-03 TransformedTargetRegressor features (lag_1, lag_2, lag_3, cat)=['num', 'num', 'num', 'num']
```

Lo mismo pasa con XGB, HGB y CatBoost: el modelo interno no tiene ninguna variable categórica.

**Lo que ya se comprobó.**
- `TransformedTargetRegressor` y `CalibratedClassifierCV` reenvían al estimador interno los argumentos de `fit` (`categorical_feature`, `cat_features`), una matriz `object` y los NaN.
- `set_params` sobre `CalibratedClassifierCV.estimator` llega a los clones entrenados, uno por fold.

**Arreglo propuesto.** Un helper privado común en `skforecast/utils/utils.py`:

```python
def _unwrap_estimator(estimator: object, fitted: bool = False) -> object:
    # Pipeline -> último paso (ver T3: la decisión de A-05 puede cambiar esto)
    # TransformedTargetRegressor -> .regressor (sin entrenar) / .regressor_ (entrenado)
    # CalibratedClassifierCV -> .estimator (sin entrenar) /
    #                           .calibrated_classifiers_[0].estimator (entrenado)
```

Hay que usarlo en lugar de los `isinstance(estimator, Pipeline)` de:
- `configure_estimator_categorical_features`;
- `cast_catboost_categorical_columns` y `cast_catboost_categorical_columns_dataframe`;
- `_get_estimator_categorical_set_params` y `_restore_estimator_categorical_set_params`;
- `_get_catboost_cat_feature_indices`, con `fitted=True`;
- `ForecasterRecursiveClassifier._check_categorical_support` (`skforecast/recursive/_forecaster_recursive_classifier.py`), que hoy desenvuelve `CalibratedClassifierCV` a mano y ahí está la incoherencia de M-17.

Para ver todos los llamadores: `grep -rn "configure_estimator_categorical_features\|cast_catboost_categorical_columns\|_estimator_categorical_set_params" skforecast --include=*.py`. Aparecen los cinco forecasters con categóricas, `model_selection/_search.py` y `model_selection/_utils.py` (búsqueda one-step-ahead).

**Requisito, para no romper lo que hoy funciona.** `TransformedTargetRegressor(CatBoost)` y `CalibratedClassifierCV(CatBoost)` predicen bien hoy, con las categóricas como numéricas. Cuando reciban `cat_features` en `fit`, su `predict` necesitará una matriz `object` con las columnas categóricas en `int`.
- La rama de CatBoost de `_build_predict_function` compara hoy `estimator_name == 'CatBoostRegressor'`, así que un `TransformedTargetRegressor` cae en el `estimator.predict(X)` genérico con `float` y fallaría.
- `ForecasterRecursiveClassifier._recursive_predict`, cerca de la línea 1686, usa `_get_catboost_cat_feature_indices(self.estimator)`.

En los dos sitios hay que desenvolver para detectar el CatBoost y hacer el cast antes de llamar al `predict` del envoltorio.

**Fuera de alcance:** `estimator_has_native_nan_support` solo desenvuelve `Pipeline`. Añadir `TransformedTargetRegressor` sería una mejora, no un arreglo.

**Tests:**
- Un `test_unwrap_estimator.py` nuevo: Pipeline, `TransformedTargetRegressor` y `CalibratedClassifierCV`, entrenados y sin entrenar.
- Ampliar `test_configure_estimator_categorical_features.py`, `test_cast_catboost_categorical_columns*.py`, `test_get/restore_estimator_categorical_set_params.py` y `test_get_catboost_cat_feature_indices.py` con los dos envoltorios.
- De punta a punta:
  - `ForecasterRecursive`, `ForecasterDirect`, `ForecasterRecursiveMultiSeries` y `ForecasterDirectMultiVariate` con `TransformedTargetRegressor(LGBM | XGB | HGB | CatBoost)` y exog categórica. El modelo interno debe ver la categórica, y `predict` y el backtesting deben funcionar.
  - `ForecasterRecursiveClassifier` con `CalibratedClassifierCV(...)`: lags categóricos, `predict`, `predict_proba` y backtesting.
  - Búsqueda one-step-ahead con un envoltorio.

**Documentación.** Volver a ejecutar `docs/user_guides/autoregressive-classification-forecasting.ipynb`, porque los resultados de la celda 45 cambiarán. El comando es `python tools/docs/execute_notebooks/execute_notebooks.py <notebook>`; luego revisa el log en `tools/docs/execute_notebooks/logs/`.

**Nota de versión (Fixed).** Con `CalibratedClassifierCV` y `TransformedTargetRegressor`, las variables categóricas y los lags del clasificador se pasaban como numéricos aunque el modelo interno las admite de forma nativa. Ahora se configuran en el modelo interno, así que los resultados de esos modelos cambian.

---

### T3 · `Pipeline` con LightGBM o CatBoost y exog categórica (A-05), decisión de Joaquín

**Problema.** `configure_estimator_categorical_features` desenvuelve el `Pipeline`, pero devuelve `categorical_feature` / `cat_features` sin el prefijo `paso__`, así que `fit` falla. Con XGB y HGB funciona, porque se configuran con `set_params` sobre el último paso.

**Reproducción** (`a05.py`):

```python
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterRecursive

rng = np.random.default_rng(0)
idx = pd.date_range("2020-01-01", periods=200, freq="D")
cat = pd.Series(rng.choice(["a", "b", "c"], 200), index=idx)
y = pd.Series(rng.normal(size=200) + cat.map({"a": 0, "b": 5, "c": 10}).to_numpy(), index=idx, name="y")
exog = pd.DataFrame({"cat": cat.astype("category"), "num": rng.normal(size=200)}, index=idx)
forecaster = ForecasterRecursive(make_pipeline(StandardScaler(), LGBMRegressor(verbose=-1)), lags=3)
forecaster.fit(y, exog=exog)
```

```text
ValueError: Pipeline.fit does not accept the categorical_feature parameter. You can pass parameters to specific steps of your pipeline using the stepname__parameter format, ...
```

Con `CatBoostRegressor` falla igual (`cat_features`), en `ForecasterRecursive` y en `ForecasterDirect`.

**Opciones medidas** (10 categorías; MAE de la predicción):

| Opción | Resultado |
|---|---|
| **(a) Recomendada.** Con un `Pipeline`, no configurar las categóricas nativas: las columnas categóricas pasan con su codificación ordinal, como para cualquier otro estimador | MAE 0,11-0,12 con cualquier paso previo y librería |
| (b) Prefijo `f"{pipeline.steps[-1][0]}__"` en la clave (el plan original) | Descartada: con `StandardScaler` delante de LightGBM, MAE 5,60 frente a 0,12 (LightGBM trunca los códigos escalados y junta categorías). Con `MinMaxScaler`, 5,99 y sin aviso. CatBoost y XGB con escalador fallan. Con `ColumnTransformer` las columnas se reordenan y los índices apuntan a otra columna |
| (c) Mantener el error, con un mensaje más claro | Mínima; el usuario tiene que quitar el `Pipeline` |

**Implementación de (a):**
- `configure_estimator_categorical_features` devuelve `fit_kwargs` sin tocar nada si el estimador es un `Pipeline`. Hay que revisar también la rama de reset.
- `_check_categorical_support` del clasificador devuelve `False` con un `Pipeline`. Así `features_encoding='auto'` usa la codificación ordinal y `'categorical'` da el `ValueError` que ya existe.
- Si se aplica después de T2, `_unwrap_estimator` no debe desenvolver el `Pipeline` para las categóricas, o los llamadores comprueban el `Pipeline` antes.
- Cambia el comportamiento de los `Pipeline` con XGB o HGB, que hoy reciben la configuración nativa. Las categóricas nativas se publicaron en la 0.22.0, así que la nota va en **Changed**.

**Tests:**
- Reescribir `test_pipeline_extracts_last_step_lgbm` y `test_pipeline_extracts_last_step_xgboost`, en `skforecast/utils/tests/tests_utils/test_configure_estimator_categorical_features.py`.
- De punta a punta, `fit` y `predict` con `Pipeline(StandardScaler(), LGBM | CatBoost | XGB | HGB)` y exog categórica, en `ForecasterRecursive` y `ForecasterDirect`.
- El clasificador con un `Pipeline`.
- Buscar en `docs/user_guides/` si alguna guía combina `Pipeline` y categóricas.

---

### T4 · La configuración categórica del usuario se pierde y salen avisos de más (M-12), decisión de Joaquín

**Problema.** `configure_estimator_categorical_features`, con `categorical_features='auto'`, el valor por defecto:
1. Sin columnas categóricas, resetea sin aviso la configuración del usuario: HGB `categorical_features` pasa a `'from_dtype'` y XGB `feature_types` a `None`.
2. En cada reentrenamiento avisa (`IgnoredArgumentWarning`) por el valor que puso el propio skforecast en el `fit` anterior. Se repite en el backtesting con `refit` y en la búsqueda one-step-ahead con `lags_grid`.
3. Contradice la nota de `docs/user_guides/categorical-features.ipynb` (celda 35), que dice que `'auto'` «will have no effect» si no encuentra categóricas.

**Reproducción** (`m12.py`):

```python
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from skforecast.recursive import ForecasterRecursive

rng = np.random.default_rng(0)
idx = pd.date_range("2020-01-01", periods=120, freq="D")
y = pd.Series(rng.normal(size=120), index=idx, name="y")
exog_num = pd.DataFrame({"num": rng.normal(size=120)}, index=idx)
exog_cat = pd.DataFrame({"cat": pd.Categorical(rng.choice(["a", "b"], 120)), "num": rng.normal(size=120)}, index=idx)

# 1) The user configuration is reset without warning when there are no categorical columns
forecaster = ForecasterRecursive(HistGradientBoostingRegressor(categorical_features=[3], max_iter=5), lags=3)
forecaster.fit(y, exog=exog_num)
print("1) categorical_features after fit:", forecaster.estimator.get_params()["categorical_features"])

# 2) Every refit warns about the value that skforecast itself set in the previous fit
forecaster = ForecasterRecursive(HistGradientBoostingRegressor(max_iter=5), lags=3)
for i in range(2):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        forecaster.fit(y, exog=exog_cat)
    print(f"2) fit {i + 1}:", [type(x.message).__name__ for x in w])
```

```text
1) categorical_features after fit: from_dtype
2) fit 1: []
2) fit 2: ['IgnoredArgumentWarning']
```

**Opciones:**
- **(a) Recomendada.** Avisar una vez en `__init__` cuando el usuario trae su propia configuración categórica y `categorical_features` no es `None`. El aviso dice que skforecast gestiona esos parámetros y que, para conservar los suyos, use `categorical_features=None`. Con eso se quitan los avisos de `configure_estimator_categorical_features` en `fit`.
  - Configuración propia: HGB `categorical_features` distinto de `None`/`'from_dtype'`; XGB `feature_types` distinto de `None` o `enable_categorical=True`; `categorical_feature` (LightGBM) o `cat_features` (CatBoost) en `fit_kwargs`.
  - Forecasters con `categorical_features`: `ForecasterRecursive`, `ForecasterDirect`, `ForecasterRecursiveMultiSeries` y `ForecasterDirectMultiVariate`. Además, `ForecasterRecursiveClassifier`, con `features_encoding`.
  - Revisar también `set_fit_kwargs` y `set_params`, que pueden traer la configuración después de `__init__`.
- **(b) Reducida:** avisar solo si el valor previo no es el de por defecto y es distinto del nuevo. Quita el aviso del refit, pero no los de la búsqueda con `lags_grid`, donde los índices cambian con los lags.

Un marcador en el estimador no sirve: `forecaster.set_params` clona el estimador y lo pierde. En las dos opciones hay que corregir la nota de la celda 35: con `'auto'` y sin categóricas, la configuración categórica del estimador se resetea.

**Tests:**
- Los de `test_configure_estimator_categorical_features.py` que comprueban avisos.
- Tests de `__init__` de cada forecaster con configuración propia: avisa una vez, y no vuelve a avisar en `fit`.
- Backtesting con `refit=True` y búsqueda con `lags_grid`: ningún `IgnoredArgumentWarning`.

**Nota de versión:** Fixed.

---

### T5 · Exog con fechas en orden descendente descartada en multiserie (N-17)

**Problema.** En `ForecasterRecursiveMultiSeries`, una exog con las fechas correctas pero en orden descendente se queda vacía al alinearla. El modelo se entrena **sin exog** (`exog_in_=False`) y solo quedan los avisos. Pasa igual con la exog ancha y con el dict.

**Reproducción** (`n17.py`):

```python
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.recursive import ForecasterRecursiveMultiSeries

idx = pd.date_range("2020-01-01", periods=10, freq="D")
series = {"a": pd.Series(np.arange(10.0), index=idx), "b": pd.Series(np.arange(10.0), index=idx)}
exog = pd.DataFrame({"e": np.arange(10.0) * 100}, index=idx[::-1])  # same dates, descending order
for name, exog_input in [
    ("wide", exog), ("dict", {"a": exog, "b": exog}),
    ("wide sorted", exog.sort_index()), ("dict sorted", {"a": exog.sort_index(), "b": exog.sort_index()}),
]:
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        forecaster.fit(series, exog=exog_input)
    categories = sorted({type(x.message).__name__ for x in w} - {"DataTransformationWarning"})
    print(f"{name:12} exog_in_={forecaster.exog_in_!s:5} warnings={categories}")
```

```text
wide         exog_in_=False warnings=['IgnoredArgumentWarning', 'MissingExogWarning', 'MissingValuesWarning']
dict         exog_in_=False warnings=['IgnoredArgumentWarning', 'MissingExogWarning', 'MissingValuesWarning']
wide sorted  exog_in_=True  warnings=['IgnoredArgumentWarning']
dict sorted  exog_in_=True  warnings=['IgnoredArgumentWarning']
```

**Causa.** En `align_series_and_exog_multiseries` (`skforecast/utils/utils.py`), `exog_dict[k].loc[first_valid_index:last_valid_index]` corta por etiqueta. Con un índice descendente, el corte sale vacío y la exog pasa a `None` con el aviso «empty after aligning». La exog ancha se convierte antes en un dict por serie, así que pasa por el mismo sitio.

**Arreglo recomendado.** Ordenar el índice tras la comprobación de duplicados, dentro de `if not series_dict[k].index.equals(exog_dict[k].index):`:

```python
if not exog_dict[k].index.is_monotonic_increasing:
    exog_dict[k] = exog_dict[k].sort_index()
```

Es seguro, porque la alineación es por fecha y los duplicados ya dan error. La alternativa es lanzar un `ValueError` pidiendo un índice ordenado; consúltalo con Joaquín si dudas.

**Comprobar también:**
- `predict` con una exog descendente: `check_predict_input` alinea con `isin` y `_create_predict_inputs` reindexa, así que probablemente funciona, pero hay que testearlo.
- `FoundationModel`, que usa los mismos helpers.

**Tests:**
- En `test_align_series_and_exog_multiseries.py`: una exog descendente da lo mismo que la ordenada.
- `fit` y `predict` multiserie con exog descendente, ancha y dict.

**Nota de versión:** Fixed.

---

### T6 · `ForecasterStats` con `last_window` o `last_window_exog` (N-16, N-18)

**N-16.** `ForecasterStats` con `Sarimax` y un `last_window` cuyo nombre no es el de la serie de entrenamiento falla dentro de statsmodels.

```python
import warnings
import numpy as np
import pandas as pd
from skforecast.recursive import ForecasterStats
from skforecast.stats import Sarimax

warnings.simplefilter("ignore")
idx = pd.date_range("2020-01-01", periods=60, freq="D")
y = pd.Series(np.arange(60.0) + np.random.default_rng(1).normal(size=60), index=idx, name="y")
last_window = pd.Series([61.0, 62.5], index=pd.date_range("2020-03-01", periods=2, freq="D"), name="other_name")
for name in ["y", "other_name"]:
    forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 0)))
    forecaster.fit(y)
    try:
        print(name, forecaster.predict(steps=2, last_window=last_window.rename(name)).round(3).tolist())
    except Exception as e:
        print(name, f"{type(e).__name__}: {e}")
```

```text
y [62.455, 62.409]
other_name ValueError: Columns must match to concatenate along rows.
```

- **Causa.** `ForecasterStats._create_predict_inputs` (`skforecast/recursive/_forecaster_stats.py`) pasa `last_window` con su propio nombre a `Sarimax.append`, en `_check_append_last_window`, y statsmodels concatena por nombre de columna.
- **Arreglo.** Renombrar `last_window` al nombre que tenía `y` en `fit` antes de `append`.
- **Cuidado:** `series_name_in_` vale `'y'` cuando `y` no tenía nombre, pero statsmodels vio `None`. Prueba con `y` con nombre y sin él, y guarda o usa el nombre real.

**N-18.** Un `last_window_exog` Series con un nombre válido, cuando el forecaster se entrenó con más exógenas, también falla en statsmodels:

```python
import warnings
import numpy as np
import pandas as pd
from skforecast.recursive import ForecasterStats
from skforecast.stats import Sarimax

warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
idx = pd.date_range("2020-01-01", periods=63, freq="D")
y = pd.Series(np.arange(63.0) + rng.normal(size=63), index=idx, name="y")
exog = pd.DataFrame({"e1": rng.normal(size=63), "e2": rng.normal(size=63)}, index=idx)
forecaster = ForecasterStats(estimator=Sarimax(order=(1, 0, 0)))
forecaster.fit(y.iloc[:50], exog=exog.iloc[:50])
forecaster.predict(
    steps=3, exog=exog.iloc[60:], last_window=y.iloc[50:60], last_window_exog=exog["e1"].iloc[50:60]
)
```

```text
ValueError: Columns must match to concatenate along rows.
```

- **Causa.** En `check_predict_input`, el bloque de `last_window_exog` solo comprueba las columnas que faltan si es un DataFrame.
- **Arreglo.** El mismo de B-11 para `exog` (en #1362):
  - las columnas de una Series son `[name]`;
  - una Series sin nombre da `ValueError`;
  - y luego viene la comprobación de columnas que faltan, que lanza `ValueError: Missing columns in last_window_exog...`.

**Tests.** En `skforecast/utils/tests/tests_utils/test_check_predict_input.py` y en los tests de `predict` de `ForecasterStats` (`skforecast/recursive/tests/tests_forecaster_stats/`). Comprobar que con el nombre cambiado se obtiene la misma predicción que con el nombre original, y que la Series da el `ValueError` nuevo.

**Nota de versión:** una entrada en Fixed para los dos casos de `ForecasterStats`.

---

### T7 · Doble aviso de NaN en multiserie con exog ancha (N-04)

```python
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.recursive import ForecasterRecursiveMultiSeries

rng = np.random.default_rng(0)
idx = pd.date_range("2020-01-01", periods=50, freq="D")
series = {"a": pd.Series(rng.normal(size=50), index=idx), "b": pd.Series(rng.normal(size=50), index=idx)}
forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
forecaster.fit(series, exog=pd.DataFrame({"e": rng.normal(size=50)}, index=idx))
exog_pred = pd.DataFrame({"e": [1.0, np.nan, 2.0]}, index=pd.date_range("2020-02-20", periods=3, freq="D"))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    forecaster.predict(3, exog=exog_pred)
for x in w:
    if type(x.message).__name__ == "MissingValuesWarning":
        print(str(x.message).splitlines()[0])
```

```text
`exog` has missing values. Most of machine learning models do not allow missing values. Prediction method may fail.
`exog` has missing values. Most machine learning models do not allow missing values. Fitting the forecaster may fail.
```

**Causa.** `ForecasterRecursiveMultiSeries._create_predict_inputs` llama a `check_exog(exog=exog, allow_nan=False)` cerca de la línea 2791, en la rama `else` de la comprobación de dtypes, después de alinear la exog. Ese NaN ya lo avisó `check_predict_input`. Además, el texto habla de «Fitting» dentro de `predict`.

**Arreglo.** Pasar `allow_nan=True` en esa llamada, que entonces solo valida el tipo, o quitarla. Antes, comprueba que cada caso de NaN sigue avisando **una** vez:
- NaN del usuario;
- exog sin todas las fechas de los pasos;
- columnas que faltan (`MissingExogWarning`);
- dict sin algunos niveles.

Si algún NaN que aparece por la alineación solo lo avisaba esta llamada, hay que conservar ese aviso con el texto de `predict`.

**Tests.** Contar los `MissingValuesWarning` en los casos anteriores, con exog ancha y con dict.

**Nota de versión:** Fixed. Es una línea: el aviso de valores que faltan salía dos veces.

---

### T8 · Skill de multiserie: formato de la exog (N-07)

`skills/forecasting-multiple-series/SKILL.md`, línea 33 (fila de la tabla) y línea 157 (Common Mistakes, punto 3), dice que la exog debe tener el mismo formato que `series`, las dos anchas o las dos dict. Es falso:
- La guía `multi-series-with-different-length-and-different_exog.ipynb` (celda 1) tiene una tabla con todas las combinaciones válidas.
- Comprobado en `18e5dbaf0`: un dict de series con una exog ancha entrena y predice.

Hay que corregir las dos líneas: los formatos se pueden combinar y la exog ancha se alinea por fecha con cada serie. Luego `python tools/ai/generate_ai_context_files.py` y `--check` (skill `ai-context-sync`); los ficheros generados no se editan a mano. No necesita nota de versión.

---

### T9 · Imports sin usar en tests (N-08)

`ruff check --select F401,F811` da avisos previos en:
- `skforecast/recursive/tests/tests_forecaster_recursive_multiseries/test_recursive_predict_bootstrapping.py` (4);
- `skforecast/recursive/tests/tests_forecaster_stats/test_predict_interval.py` (7) y `test_repr.py`;
- `skforecast/recursive/tests/tests_forecaster_recursive_classifier/test_create_predict_X.py`;
- `skforecast/utils/tests/tests_utils/test_align_series_and_exog_multiseries.py`, `test_initialize_transformer_series.py` (2) y `test_manage_warnings.py`.

Los `import keras` de los tests de `ForecasterRnn` son probablemente intencionados (el backend): no tocarlos. Arréglalos solo cuando un PR toque esos ficheros; no hace falta un PR solo para esto.

---

## 5. Opcionales y descartados (no hacerlos sin hablarlo)

| Tema | Estado | Datos |
|---|---|---|
| `ExtraTreesRegressor` (#1363) y `RandomForestRegressor` (desde la 0.24.0) con `n_jobs=-1` y muchas filas por llamada | Se deja | El atajo por árbol usa un solo hilo y scikit-learn reparte entre núcleos. MultiSeries 500 series `predict(24)`: 1,1 → 1,4 s. Árboles profundos y 5000 filas: 139 → 464 ms (×3,3 con 4 núcleos). Pero `ForecasterRecursive.predict(24)` con `n_jobs=-1`: 830 → 10 ms; `predict_bootstrapping(n_boot=500)`: 910 → 80 ms; con `n_jobs=None` siempre igual o mejor. Un umbral por filas dependería de la máquina y cambiaría también RandomForest |
| `multivariate_time_series_corr` (#1363) con datos muy pequeños | Se deja | 100 filas y 5 lags: spearman 6,8 → 14 ms, pearson 4,2 → 5,5 ms. Con datos reales mejora (spearman 2000×24: 512 → 128 ms) |
| O-08: cast de CatBoost sin `astype(object)` | Descartado | Gana poco y solo en `fit`; en `predict` es peor |
| N-05: `searchsorted` en lugar de `isin` en `check_predict_input` | Descartado | La comprobación baja de 77 a 59 ms, pero `predict` no cambia |
| N-06: el camino rápido de `_check_exog_alignment` confía en que `freq` sea igual | Teórico | Una exog `freq='24h'` frente a una serie `'D'` con cambio de hora pasaría sin comprobar; sin caso real |
| N-03: Series exog con nombre no visto y una categórica en multiserie | Ya no se reproduce en `18e5dbaf0` | Ahora salen los avisos y NaN, como se espera |
| `estimator_has_native_nan_support` con `TransformedTargetRegressor` | Mejora, no arreglo | Solo desenvuelve `Pipeline` |
| `save_forecaster` devuelve la ruta final | Propuesta sin decidir | |
| `initialize_differentiator_multiseries` puede listar `'_unknown_level'` en su aviso | Menor | No se alcanza desde los forecasters |
| `check_predict_input` con `levels` como `str` | Se deja | El docstring dice `str, list` (Joaquín lo mantuvo en `5e08b2f`); `set('l1')` lo partiría, pero no se alcanza desde los forecasters |

---

## 6. Pendiente de Joaquín (no del compañero)

- **Decisiones:**
  - la opción de A-05 (T3) y la de M-12 (T4);
  - si N-17 (T5) ordena o lanza un error.
- **Tests de `ForecasterRnn` en local,** porque el cloud no tiene torch ni keras: `pytest skforecast/deep_learning/tests/tests_forecaster_rnn -q` sobre `0.26.x`.
- **Ramas remotas por borrar:** `ci/minimum-versions-job` y `fix/weight-func-export`.

---

## 7. Verificación al cerrar esta sesión

- Todas las reproducciones se ejecutaron sobre `18e5dbaf0`, y las salidas de este documento son las reales. La de T1 también sobre `9c6f0fd`, antes de #1363, como referencia.
- Antes de fusionar #1363 se probó su combinación con `9c6f0fd`:
  - `pytest skforecast/utils/tests -q -m "not slow"`: 885 passed;
  - los `test_predict.py` de `ForecasterRecursive` y `ForecasterRecursiveMultiSeries` y el `test_predict_bootstrapping.py` multiserie: 128 passed;
  - la nota de versión no tiene referencias sin definir.
- El arreglo de T1, aplicado sobre `18e5dbaf0`: 201 passed en `test_check_predict_input.py` y en los `test_predict.py` de `ForecasterRecursive` y `ForecasterRecursiveMultiSeries`.
- Para cada tarea, el cierre es el skill `verify`:
  - `ruff check` en los ficheros cambiados;
  - los tests afectados, en secuencia;
  - `python tools/ai/generate_ai_context_files.py --check` si cambia el contexto IA;
  - la nota de versión, comprobada con el script del skill `release-note`.
