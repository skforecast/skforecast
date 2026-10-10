# Revisión de `skforecast/utils/utils.py` (rama 0.26.x)

> **Cómo retomar este trabajo en otra sesión** (actualizado el 2026-10-07, tarde)
>
> **Punto de partida:**
> - Repo `skforecast/skforecast`, rama `0.26.x`, HEAD `02ae46b` (merge del PR 4). La revisión original se hizo sobre `a5fed66`.
> - Hechos y fusionados: **PR 1a, 1b, 1c, 6, 2a, 3a, 3b, 3c, 7 y 4**.
> - **Abiertos en borrador (2026-10-07, con el OK del usuario):** el **PR 5a** skforecast/skforecast#1362 (`fix/input-validation`, 14 commits sobre `02ae46b`, head `40ed950`, §5.11) y el **PR 5b** skforecast/skforecast#1363 (`perf/utils-hot-paths`, 7 commits apilados sobre el 5a, head `5867a2d`, §5.12; depende del 5a). Sesión suscrita a los dos; check-in de seguridad programado.
> - **Queda del plan:** el **PR 2b** (§5.7), que espera las decisiones A-05 y M-12. Hallazgos sueltos sin PR: N-03, N-04, N-07, N-08, N-16 y N-17.
> - Los números de línea de `utils.py` citados en los hallazgos son de `a5fed66` y están desplazados: buscar siempre por nombre de función.
>
> **Cambios en `0.26.x` desde la revisión (comprobado el 2026-10-05, HEAD `2e365b6`):** se fusionó skforecast/skforecast#1343 (`7092ae3`), que añade `_is_utc_anchored_index` y `_date_range_from_index` y toca `expand_index`, `date_to_index_position`, los Direct, el multiserie y `preprocessing`. Los números de línea de `utils.py` a partir de la línea 1913 están desplazados unas 105 líneas. Ese PR **no** resuelve M-09 ni M-10: los dos se siguen reproduciendo en `2e365b6`. Ajustes al plan: ver M-09 y el punto 4 del diseño de A-03.
>
> **Estado de los PRs:**
> - PR 1a: **fusionado** en `0.26.x` (skforecast/skforecast#1344, merge commit `942f3d0`, 2026-10-05). Aprobado por Javier Escobar Ortiz. Incluye A-01 y M-02.
> - PR 1b: **fusionado** en `0.26.x` (skforecast/skforecast#1345, merge commit `8a4367b`, 2026-10-05). El usuario añadió `176006b` en su revisión local: en DirectMV, con la serie objetivo sin lags, `self.differentiation` y el diferenciador devuelto no coincidían (bug mío del 1b); ahora ambas condiciones usan el diferenciador. Desuscrito y check-in `trig_01Ewaqenx4NrYoAKfh1ouaKz` borrado. La rama remota se borró.
> - PR 1c: **fusionado** en `0.26.x` (skforecast/skforecast#1346, merge commit `5012fcb`, 2026-10-05). Commits: `fae5401` (M-09), `fe184cd` (A-03), `ab5c63a` (A-04 b-light), `178607d` (M-03), `54577a8` (notas), `29f707e` (usuario: `exog` vacío), `9a81dc9` (`lenient` -> `align_by_index`), `b711849` (nota de M-09 fusionada con la de #1343). CI verde en el head. Desuscrito y check-in `trig_01FrQa6xtCT3kDwWVBG6QJDC` borrado. La rama remota `fix/predict-exog-validation` sigue existiendo (no borrar sin permiso). Pendiente fuera del PR: tests de `ForecasterRnn` sin ejecutar aquí (sin torch/keras); doble aviso de NaN con exog ancho (ya existía en 0.26.x).
> - PR 6: **fusionado** en `0.26.x` (skforecast/skforecast#1348, merge commit `742d309`, 2026-10-06). Aprobado por Javier Escobar Ortiz. Desuscrito y check-ins borrados. La rama remota `chore/minimum-versions-and-cleanup` sigue existiendo (no borrar sin permiso). Historial: rama `chore/minimum-versions-and-cleanup`, 4 commits sobre `14bf3ae` (`08e3db0`, `3b7fd82`, `8e52a08`, `fea55d7`). Título sin `>=` (la herramienta lo escapaba como `&gt;`). Pie quitado y sesión suscrita a la actividad del PR. El usuario añadió `1dee14b` (highlight con la etiqueta API Change para las versiones mínimas), revisado y correcto. El plan cambió: ver §5.6.
> - PR 2a: **fusionado** en `0.26.x` (skforecast/skforecast#1349, merge commit `59822eb`, 2026-10-06 10:57 UTC), sin commits del usuario (mismo contenido que `3d43b12`). CI verde. Desuscrito y check-ins borrados. El usuario borró la rama remota; queda la local `fix/catboost-predict-and-device`. Ver §5.7.
> - PR 3a: **fusionado** en `0.26.x` (skforecast/skforecast#1350, merge commit `7e352bc`, 2026-10-06), sin commits del usuario. Desuscrito y check-in borrado. La rama remota `fix/save-forecaster-file-names` sigue existiendo (no borrar sin permiso).
> - PR 3b: **fusionado** en `0.26.x` (skforecast/skforecast#1352, merge commit `b08b5b6`, 2026-10-06 15:21 UTC), con un commit del usuario (`4b5c3e1`). CI verde. Desuscrito y check-in borrado. El usuario borró la rama remota; queda la local `fix/skops-persistence` (no borrarla sin permiso).
> - PR 3c: **fusionado** en `0.26.x` (skforecast/skforecast#1353, merge commit `adbf335`, 2026-10-06 16:24 UTC), con un commit del usuario (`4478ba0`: funciones sin código fuente al aviso, ancla de la documentación). CI verde. Desuscrito automáticamente y check-in borrado. La rama remota `fix/weight-func-export` sigue existiendo, igual que la local (no borrar sin permiso).
> - PR 2b, 4, 5a y 5b: sin empezar. El PR 2 está revisado y dividido en 2a y 2b (§5.7); el PR 3 se dividió en 3a, 3b y 3c (§5.8). M-06 y S-04 se hicieron en el 3c, así que salen del 5a. Ramas que siguen existiendo (no borrarlas sin permiso del usuario): remotas `fix/fast-predict-paths` (1a), `fix/predict-exog-validation` (1c), `chore/minimum-versions-and-cleanup` (6), `fix/save-forecaster-file-names` (3a) y `fix/weight-func-export` (3c); locales `fix/direct-differentiation-steps` (1b), `fix/catboost-predict-and-device` (2a), `fix/skops-persistence` (3b) y `fix/weight-func-export` (3c).
> - PR 7 (job de CI con las versiones mínimas): **fusionado** en `0.26.x` (skforecast/skforecast#1360, merge commit `2d102b6`, 2026-10-07 15:34 UTC), con un commit del usuario (`31de197`: `packaging.version` en `test_preprocess_repr`). Desuscrito y check-in borrado. Siguen existiendo las remotas `ci/minimum-versions-job` (5 commits; el borrado desde aquí da 403) y `ci/minimum-versions-job-v2`, y la local `ci/minimum-versions-job-v2`. Ver §5.9.
>
> **Autorizaciones (2026-10-05):**
> - El usuario autoriza crear y subir las ramas del §5.4 y abrir sus PRs **en borrador** contra `0.26.x`. El merge y el paso a "listo para revisión" los hace siempre el usuario.
> - Flujo por PR: implementar en local → punto de control 1 en el chat (commits, tests, desviaciones del plan) → push y PR en borrador solo con el OK del usuario → punto de control 2 en GitHub. En el PR 1c hay una parada extra tras el commit de A-03 (texto del error y tests que cambian).
> - Preferencias de estilo del usuario: **una asignación por línea** (nada de `a, b = x, y`; desempaquetar el retorno de una función sí vale). Descripción de PR con `## Description` y `## Verification`; quitar el pie "Generated by Claude Code" que añade el servidor.
> - Más preferencias aprendidas en los PRs 1b y 1c:
>   - Nombres descriptivos: un argumento debe decir qué hace, no su consecuencia (se cambió `lenient` por `align_by_index`).
>   - Exactamente dos líneas en blanco entre tests (el usuario lo corrigió a mano en el 1b; en el 1c lo cazó la revisión final).
>   - Notas de versión: una sola entrada por función o problema, aunque vengan de PRs o autores distintos (se fusionó la de M-09 con la de #1343). Las sub-viñetas `    + ` son válidas (formato de la entrada de `Ets`).
>   - El usuario revisa los PRs en local con Claude en VS Code y a veces sube commits propios a la rama: hacer `git fetch` y revisarlos antes de seguir.
>   - Cambios de diseño (como A-04) se le presentan con mediciones y una recomendación; él decide. Para cambios de comportamiento, comprobar primero qué hacen hoy `fit`, el backtesting y los otros formatos de entrada.
>   - Al terminar cada PR pide una "revisión final exhaustiva": incluir prueba de propiedades contra una referencia, tests en cada commit, rendimiento frente a la base y caso de usuario que cambia de comportamiento.
> - Paralelismo: como mucho dos PRs abiertos a la vez. 1c sale de `origin/0.26.x` (no de la rama del 1b): toca otras funciones; el único choque esperado es la nota de versión en `releases.md`, que se resuelve con un merge de la base cuando se fusione el primero.
> - Identidad de git: variables `GIT_AUTHOR_*` y `GIT_COMMITTER_*` del entorno cloud (`Joaquín Amat Rodrigo <JoaquinAmatRodrigo@users.noreply.github.com>`); comprobado con `git var`.
>
> **Orden y contenido:** §5.4 tiene el orden de los PRs y los commits de cada uno. El antiguo PR 1 está dividido en 1a, 1b y 1c. Los PRs 1a, 1b, 1c y 6 no tocan las mismas funciones y pueden empezar ya.
>
> **Scripts citados:** los que se mencionan en el informe (`mine*.py`, `v1..v5.py`, `xgb_e2e.py`, `b1/..b5/`, `a03/`) vivían en el scratchpad de la sesión original y **ya no existen**. Cada hallazgo incluye su reproducción mínima y su salida, y el prototipo de A-03 está copiado en su sección; basta para volver a generarlos.
>
> **Entorno:**
> - Las sesiones cloud no instalan torch ni keras. Para S-05 y B-21 (PR 5a, `ForecasterRnn`) hay que instalarlos (`SKFORECAST_CLOUD_DL=1` o `uv pip install torch "keras>=3.0,<4.0" --torch-backend cpu`).
> - lightgbm, xgboost, catboost y skops ya vienen instalados.

**Alcance:** las 54 funciones del módulo (unas 4250 líneas), más los llamadores necesarios para entender cada contrato. Es solo revisión: no se ha modificado nada del repo y `git status` está limpio.

**Estado (2026-10-05, final del día):** las cinco decisiones de diseño están tomadas (§5.1; la 1a se cambió por la opción b-light, ver A-04 y §5.5) y el plan está en §5.2 y §5.4. **Implementados y fusionados: PR 1a, 1b y 1c.** Pendientes: PR 6, 2, 3, 4, 5a y 5b. Los hallazgos afectados por esas decisiones (A-04, M-05, M-16, B-10) y el nuevo N-01 (pandas 2.1) se han actualizado en el texto. El 2026-10-05 el PR 1 se dividió en tres (1a, 1b y 1c; ver §5.4).

**Entorno:**
- Python 3.12, pandas 2.3.3, numpy 2.5.3, scikit-learn 1.9.1.
- lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, skops 0.16.0.
- Para el suelo de versiones, entornos aparte con pandas 2.1.4 + numpy 1.26.4 + sklearn 1.4.2, y sklearn 1.5.2 / 1.6.1.

**Cómo se hizo:**
- **Lectura:** leí el módulo entero.
- **Revisión por bloques:** se revisó por los 5 bloques del encargo. Cada bloque generó sus reproducciones y benchmarks.
- **Verificación cruzada:** he vuelto a ejecutar yo mismo las reproducciones de todos los hallazgos de severidad alta y de la mayoría de los de severidad media. Los que no he podido ejecutar de punta a punta (porque requieren keras o Windows) están marcados.
- **Scripts:** están en `scratchpad/review/` (`mine*.py`, `v1..v5.py`, `xgb_e2e.py`) y en `scratchpad/review/b1..b5/`.

**Tests:** los tests actuales de las funciones revisadas pasan en la rama: 148 + 152 + 114 + 58 en los bloques 1, 4, 3 y 5. Los fixes propuestos se han validado como prototipos en memoria contra los tests existentes; en cada hallazgo se indica qué tests habría que actualizar.

---

## 1. Tabla resumen

**Ningún hallazgo llega a crítico.** No hay ningún fallo en el camino por defecto más común (`ForecasterRecursive` sin exog con un estimador sklearn estándar). Los de severidad alta producen **resultados erróneos sin aviso** o **ficheros guardados que no se pueden recuperar** en configuraciones habituales.

### Bugs confirmados

| ID | Sev. | Función (utils.py:línea) | Resumen |
|---|---|---|---|
| A-01 | alto | `_build_predict_function` :3253 | XGBoost con early stopping, `booster='gblinear'` o `missing` personalizado: el atajo `inplace_predict` da predicciones distintas a `estimator.predict` sin avisar, o falla |
| A-02 | alto | (llamador) `ForecasterDirect*` + `scale_correction_factor_differentiation` | Direct con `differentiation` y `steps` no consecutivos desde 1 (o backtesting con `gap>0`): predicciones e intervalos erróneos sin aviso |
| A-03 | alto | `check_predict_input` :1466-1494 | Se acepta una exog con otra frecuencia o con huecos (solo se compara el primer timestamp): predicciones erróneas sin aviso |
| A-04 | alto | `check_predict_input` :1416/1434/1452/1477 | Multiseries con exog ancha (no dict): el aviso promete "relleno con NaN", pero se usan valores desplazados o la llamada termina en `KeyError` |
| A-05 | alto | `configure_estimator_categorical_features` :563-607 | `Pipeline` con LightGBM o CatBoost al final y exog categórica: `fit` falla con `categorical_features='auto'` (el valor por defecto) |
| A-06 | alto | (llamador) `ForecasterRecursiveClassifier` + `_build_predict_function` :3283 | `CatBoostClassifier` con `features_encoding='auto'` entrena bien, pero `predict`, `predict_proba` y backtesting fallan |
| A-07 | alto | `transform_dataframe` :2341 / `transform_series` :2257 | `transformer_y` o `transformer_exog` = `Pipeline(FunctionTransformer(log1p), StandardScaler())` hace fallar `fit` |
| A-08 | alto | `_decompose_index` / `_compose_index` :2433-2479 | skops + índice tz-aware que cruza un cambio de hora: se guarda bien pero **no se puede cargar**; sin cambio de hora se pierde el nombre de la zona horaria |
| A-09 | alto | `save_forecaster` skops :2753 (causa en `RollingFeatures`) | skops no puede guardar **ningún** forecaster entrenado con `RollingFeatures`; además, todos los backends guardan la serie de entrenamiento completa |
| A-10 | alto | `set_cpu_gpu_device` :3152-3187 | XGB `device='cuda:0'` hace fallar `predict`; `'GPU'` da `KeyError`; LightGBM `'cuda'` se restaura como `'gpu'`, que es otro backend |
| M-01 | medio | `cast_catboost_categorical_columns_dataframe` :754 | `.cat.codes` usa posiciones, no valores: en la búsqueda one-step-ahead multiserie con CatBoost, un nivel recibe la categoría de otro (MAE 6.67 frente a 1.39) |
| M-02 | medio | `_build_predict_function` :3233 | Las subclases de `LinearModel` que sobrescriben `predict` se saltan su lógica (`NonNegRidge` devuelve -6.9) |
| M-03 | medio | `check_predict_input` :1325 | Un `last_window` DataFrame con varias columnas se acepta en los forecasters de una serie; `.ravel()` intercala las columnas |
| M-04 | medio | `prepare_steps_direct` :3870 | `steps=np.int64`, tupla, `range` o array dan `UnboundLocalError`; `steps=0` o `[]` dan `min() iterable argument is empty` |
| M-05 | medio | `check_extract_values_and_index` :1730 | Con pandas 2.1 (dentro del rango soportado), una `y` `Int64` o `Float64` hace fallar `fit` (`isnan` sobre un array object). Se resuelve subiendo a `pandas>=2.2` (decisión 2b) |
| M-06 | medio | `initialize_weights` :297-302 | `weight_func` como `functools.partial` o instancia invocable hace fallar el constructor (`inspect.getsource`) |
| M-07 | medio | `align_series_and_exog_multiseries` :3682 | `np.isnan(pd.NA)`: las series `Float64`, `Int64` o pyarrow con NA al principio o al final hacen fallar `fit` |
| M-08 | medio | `check_residuals_input` :1673 | Valida los residuos de todos los niveles, no solo de los que se predicen: calibrar solo algunas series rompe todos los intervalos |
| M-09 | medio | `expand_index` :2046 | Índice diario tz-aware el día del cambio de hora: el índice de predicción se desplaza 1 h y `predict(exog=...)` lanza un `ValueError` falso |
| M-10 | medio | `date_to_index_position` :1971/1980/1989 | (a) Índice tz-aware con fecha sin zona horaria: `TypeError`. (b) Índice sin `freq`: asume diario y genera folds erróneos sin aviso |
| M-11 | medio | `transform_series` :2247 | `squeeze()` de una sola fila devuelve un escalar: `ForecasterStats.predict(last_window=<1 obs>)` falla con transformers de salida pandas |
| M-12 | medio | `configure_estimator_categorical_features` :569-577 | Con `'auto'` y sin categóricas, sobrescribe sin aviso la configuración del usuario (HGB `categorical_features`, XGB `feature_types`); en cada refit avisa sin motivo |
| M-13 | medio | `_SKLEARN_NAN_TOLERANT_ESTIMATORS` :50-64 | ExtraTree y ExtraTrees solo admiten NaN desde sklearn 1.6, pero skforecast admite >=1.4: backtesting multiserie falla en lugar de descartar el nivel |
| M-14 | medio | `_skops_decompose_forecaster` :2594 | skops no puede guardar forecasters con exog categórica (`CategoricalDtype` en `exog_dtypes_in_`) |
| M-15 | medio | `_compose_index` :2475 | skops con frecuencia por debajo del segundo (`500ms`, `100us`): se guarda pero no se puede cargar |
| M-16 | medio | `save_forecaster` :2709 | `with_suffix` corta los nombres con puntos: `model_v1.1` y `model_v1.2` escriben los dos `model_v1.joblib`, y el segundo **sobrescribe** el primero |
| M-17 | medio | `configure_estimator_categorical_features` y helpers | `CalibratedClassifierCV`: el clasificador activa las categóricas nativas, pero los helpers no desenvuelven el estimador y los lags se tratan como numéricos |
| B-01 | bajo | `check_preprocess_series` :3464 | `sorted(indexes_freq)` lanza un `TypeError` sin relación con el problema (D frente a MS, o Range frente a Datetime) en lugar del `ValueError` informativo |
| B-02 | bajo | `check_select_fit_kwargs` :515 | `del fit_kwargs['sample_weight']` modifica el dict del usuario |
| B-03 | bajo | `initialize_lags` :109/138 | Rechaza `np.int64`; acepta `True`; los lags `uint` provocan overflow en `-window_size` y `last_window_` queda vacío |
| B-04 | bajo | `deepcopy_forecaster` :4106-4158 | Sin `try/finally`: si `deepcopy` falla, el forecaster original queda sin entrenar y sin residuos ni `last_window_` |
| B-05 | bajo | `save_forecaster` skops :2747 | Modifica el forecaster vivo durante el volcado: los `predict` concurrentes fallan |
| B-06 | bajo | `_decompose_index` :2437 | skops pierde los festivos de `CustomBusinessDay` (guarda `freqstr`), y `predict` falla tras cargar |
| B-07 | bajo | `save_forecaster` :2803 | Falso `SaveLoadSkforecastWarning` para la clase propia `RollingFeaturesClassification` |
| B-08 | bajo | `save_forecaster` :2779 | El `.py` de `weight_func` se escribe con la codificación por defecto de la plataforma; en Windows falla con caracteres no ASCII (simulado, no ejecutado en Windows) |
| B-09 | bajo | `check_exog_dtypes` :985-1036 | `TypeError` con categorías `Int32` (las que produce `convert_dtypes`); aviso falso con `UInt8` y `double[pyarrow]` |
| B-10 | bajo | `cast_exog_dtypes` :1802 | Función pública rota: con una Series lanza `AttributeError`, modifica el DataFrame del usuario y pierde las categorías. Ningún código interno la usa |
| B-11 | bajo | `check_predict_input` :1445-1463 | Una Series exog con nombre válido pero sin el resto de columnas da un `KeyError` sin contexto |
| B-12 | bajo | `check_predict_input` :1476 | Un DataFrame vacío dentro de un exog dict multiserie da `IndexError` |
| B-13 | bajo | `check_predict_input` :1337 | `MissingValuesWarning` falso por NaN fuera de la ventana o en niveles que no se predicen |
| B-14 | bajo | `check_preprocess_exog_multiseries` :3524/3594 | Una Series exog sin nombre no se detecta (`to_frame()` antes de `check_exog`) |
| B-15 | bajo | `check_preprocess_exog_multiseries` :3629/3640 | `exog_names_in_` sale de un `set` (orden distinto en cada proceso); las columnas duplicadas dan un mensaje equivocado |
| B-16 | bajo | `check_preprocess_series` | Acepta un dict de series con zonas horarias distintas; luego `predict` falla |
| B-17 | bajo | `transform_series` :2223 | Un `Pipeline` aplicado a una serie con otro nombre da `AttributeError` (`feature_names_in_` no tiene setter) |
| B-18 | bajo | `preprocess_levels_self_last_window_multiseries` :3778 | `levels` como `pd.Index` o array: "truth value is ambiguous" |
| B-19 | bajo | `exog_to_direct*` :1852/1910 | No valida `steps` frente a `len(exog)`: salida llena de NaN o error críptico (solo en llamadas directas a la API pública) |
| B-20 | bajo | `initialize_window_features` :204-225 | `window_sizes=[]` da `max() iterable argument is empty`; acepta `True` |
| B-21 | bajo | `input_to_frame` :1758 | `'exog_val'` (lo pasa `ForecasterRnn`) da `KeyError`. Confirmado a nivel de función; Rnn no ejecutado (falta keras) |
| N-01 | medio | fuera de utils: `preprocessing/_preprocessing.py:425`, `preprocessing/_calendar.py:694-695` | La librería ya usa APIs que solo existen desde pandas 2.2 (`include_groups`, alias `'YE'`/`'QE'`/`'ME'`), pero declara `pandas>=2.1`. Con pandas 2.1.4, `reshape_series_wide_to_long` falla. Se resuelve con la decisión 2b (ver M-05) |
| N-10 | alto | (llamador) `_skops_decompose_forecaster` | skops + exog con dtype pyarrow: se guarda y se carga, pero el primer `predict` mata el proceso (segfault). Ver §5.8 |
| N-11 | medio | (llamador) `_skops_decompose_forecaster` | skops no puede guardar `pd.DateOffset`: `ForecasterEquivalentDate(offset=pd.DateOffset(...))` o una serie con frecuencia `DateOffset`. Ver §5.8 |
| N-12 | medio | `save_forecaster` (exportación de `weight_func`) | El `.py` exportado no incluye los imports que usa la función: el modelo cargado predice, pero al reentrenarlo da `NameError`. Ver PR 3c (§5.8) |
| N-13 | bajo | `plot.py`, `deep_learning/_forecaster_rnn.py` y `deep_learning/utils.py` (import de dependencias opcionales) | Si una dependencia opcional está instalada pero falla al importar, el error dice `No module named '<ruta>'` y esconde la causa real. Ver PR 7 (§5.9) |

### Sospechas (no reproducidas de punta a punta)

| ID | Función | Resumen |
|---|---|---|
| S-01 | `check_predict_input` :1300-1323 | `ForecasterRnn`: `last_window` puede no tener series de entrada que no están en `levels`; `get_indexer` devuelve -1 y se usa la última columna. Confirmado en el check, sin keras para probarlo entero |
| S-02 | `check_select_fit_kwargs` :495 | Los estimadores con `fit(**kwargs)` (`Pipeline`) pierden todos los `fit_kwargs`, y el aviso dice algo falso. Es una decisión de diseño |
| S-03 | categóricas | `TransformedTargetRegressor` alrededor de LGBM, XGB o HGB probablemente tampoco recibe las categóricas nativas (mismo patrón que M-17) |
| S-04 | `save_forecaster` :2780 | `inspect.getsource` / `__name__` con un `weight_func` invocable definido en `__main__` (instancia o `partial`) probablemente falla al guardar. Hoy no se puede alcanzar (el constructor falla antes, M-06): pasa al PR 5a, o al PR 3c si el usuario lo decide (§5.8) |
| S-05 | fuera de utils | `ForecasterRnn` hace `.pop("series_val")` sobre el `fit_kwargs` del usuario (`_forecaster_rnn.py:383,397`) |
| S-06 | `align_series_and_exog_multiseries` :3701 | Una exog de la misma longitud y con etiquetas distintas no se reindexa. En multiserie se detecta después con un mensaje confuso; en los caminos de foundation podría desalinear sin aviso |

### Optimizaciones medidas

| ID | Dónde | Ganancia medida | ¿Merece la pena? |
|---|---|---|---|
| O-01 | `_decompose_index` (viene con el fix de A-08) | `ForecasterEquivalentDate` skops guardar+cargar: **5.30 s / 49.7 MB → 0.019 s / 2.8 MB** | Sí |
| O-02 | `RollingFeatures.rolling_obj` (viene con el fix de A-09) | Fichero joblib **15.6 MB → 8.8 KB**; `deepcopy` 1.98 → 0.24 ms | Sí |
| O-03 | `_build_predict_function`: añadir ExtraTreesRegressor al camino rápido de RandomForest | `predict(100)` **883 → 59 ms (15x)**, salida idéntica | Sí, en el PR 5b (§5.7) |
| O-04 | `preprocess_levels_self_last_window_multiseries` | Multiserie `predict(24)`: 500 series 35 → 12 ms (x2.9); 5000 series 724 → 162 ms (x4.5) | Sí |
| O-05 | `check_predict_input` (pertenencia con `set`, sin `expand_index`, `pd.isna(to_numpy())`) | Por llamada x2.3 a x4.7; backtesting LinReg con exog, 1000 folds: **-12 %** | Sí, riesgo nulo |
| O-06 | `multivariate_time_series_corr` → `corrwith` | x2.3 (pearson) a x12.9 (spearman) | Sí, es barato |
| O-07 | `deepcopy_forecaster` con `ForecasterRnn` (doble copia del modelo Keras) | 9.4 → 5.5 s por llamada de backtesting/búsqueda | Sí (viene con el fix de B-04) |
| O-08 | `cast_catboost_categorical_columns` (evitar `astype(object)`) | `fit` 1585 → 1402 ms con 50k×30; en predict es peor | Opcional, solo en fit |

**Descartadas tras medir:**
- Comprobación de NaN en `check_y`/`check_exog`: ya es la más rápida.
- `clone` por serie en `initialize_transformer_series`: menos del 1 % del fit.
- Preprocesado y alineado multiserie en fit: menos del 4 % incluso con 5000 series.
- Envoltorios de `transform_numpy`/`transform_dataframe`: añaden 3-10 µs.
- HGB `_raw_predict` / `threadpool_limits`: sin ganancia.
- `copy=True` en `check_extract_values_and_index`: es necesario.

---

## 2. Detalle de los bugs confirmados

### A-01 · XGBoost: el atajo de predicción no respeta `best_iteration`, `gblinear` ni `missing`
- **Dónde:** `_build_predict_function`, `utils.py:3253-3259`.
- **Problema:** `booster.inplace_predict(X)` usa todos los árboles e ignora `estimator.missing`. Además no está soportado con `booster='gblinear'`. `XGBRegressor.predict` sí aplica `best_iteration`.
- **Reproducción** (`xgb_e2e.py`, de punta a punta con `ForecasterRecursive`):
  ```python
  f = ForecasterRecursive(XGBRegressor(n_estimators=500, early_stopping_rounds=5, learning_rate=0.3), lags=10,
                          fit_kwargs={"eval_set": [(X_val, y_val)], "verbose": False})
  f.fit(y[:300])
  # best_iteration 8, trees 14
  # forecaster predict (camino rápido) frente a estimator.predict: max abs diff 0.456
  ```
  Además, `gblinear` hace fallar `predict` ("Inplace predict is not supported"), y con `missing=-999` sale -0.586 en lugar de -3.304 (`b4/r11*.py`).
- **Fix:**
  ```diff
  +    if estimator.get_params().get('booster') == 'gblinear':
  +        return lambda X: estimator.predict(X).ravel()
       booster = estimator.get_booster()
  +    try:
  +        iteration_range = (0, estimator.best_iteration + 1)
  +    except AttributeError:
  +        iteration_range = (0, 0)
  +    missing = estimator.missing
       def predict_fn(X):
  -        return booster.inplace_predict(X)
  +        return booster.inplace_predict(X, iteration_range=iteration_range, missing=missing)
  ```
  El coste por fila no cambia (271.7 → 277.2 µs).
- **Riesgo / API:** ninguno. Afecta a todos los forecasters que usan `_build_predict_function` (Recursive, Direct, MultiSeries, DirectMultiVariate).
- **Test:** `test_build_predict_function.py` no cubre estos casos. Añadir casos parametrizados: early stopping, `gblinear` y `missing`.

### A-02 · `ForecasterDirect` / `DirectMultiVariate` con `differentiation` y pasos no consecutivos
- **Dónde:** fuera de utils. `_forecaster_direct.py:2419-2420, 2556, 2674-2679`; `_forecaster_direct_multivariate.py:2601, 2740, 2860`. Se dispara desde `model_selection/_validation.py:265-270` cuando `gap > 0`.
- **Problema:** `_direct_predict` solo predice los pasos pedidos, pero la inversa de la diferenciación (suma acumulada) los trata como si fueran 1..k. Además, `scale_correction_factor_differentiation` recibe `len(predictions)` en lugar de los pasos reales. La función en sí es correcta: comprobada para d=1, 2 y 3.
- **Reproducción** (`v3.py`):
  ```
  ForecasterDirect(Ridge(), steps=5, lags=5, differentiation=1)
  predict(5)[2:]     = [102.875 103.006 102.841]
  predict([3, 4, 5]) = [102.907 103.038 102.874]
  control sin differentiation: iguales = True
  ```
  En backtesting con `gap=2`, cada fold usa la versión errónea, y las métricas de grid search y bayesian search salen mal sin ningún aviso.
- **Fix (en el forecaster):**
  - Con `differentiation` activa, predecir internamente `1..max(steps)` (todos los modelos ya están entrenados), invertir la diferenciación y seleccionar `np.array(steps) - 1`.
  - Para el método conformal: `scale_correction_factor_differentiation(cf, max(steps), d)[np.array(steps) - 1]`.
- **Riesgo / API:** cambian, a mejor, las predicciones de quien ya usa esta combinación. Necesita nota de versión.
- **Test:** añadir una comprobación de `predict(steps=[3,4,5]) == predict(steps=5)[2:]`, y lo mismo para `predict_interval` y para backtesting con `gap`.

### A-03 · La exog con otra frecuencia o con huecos se acepta
- **Dónde:** `check_predict_input`, `utils.py:1466-1494`. Valida el índice de la exog con `ignore_freq=True` y solo compara `exog_index[0]` con la fecha siguiente al final de `last_window`. Todos los forecasters consumen la exog por posición:
  - `ForecasterRecursive`: `_forecaster_recursive.py:1625`, `exog.to_numpy()[:steps]`.
  - `ForecasterRecursiveClassifier`: `:1615`, igual.
  - `ForecasterDirect`: `:2182`, `exog.to_numpy()[:max(steps), :]`.
  - `ForecasterDirectMultiVariate`: `:2355`, igual.
  - `ForecasterStats`: `:775`, `exog.iloc[:steps]`.
  - `ForecasterRnn`: `:1383`, `exog.to_numpy()[:self.max_step]`.

  En cuanto la exog empieza bien pero no sigue la frecuencia, el modelo recibe valores de otras fechas.
- **Reproducción, código actual frente al prototipo** (`a03/cases.py`; `y = 2x`, datos diarios, `LinearRegression`, 5 pasos):

  | Exog pasada a `predict` | Hoy | Con el prototipo |
  |---|---|---|
  | Correcta | `[100, 102, 104, 106, 108]` | igual |
  | Correcta, con 10 filas de más | correcta | igual |
  | Correcta, sin `freq` (leída de CSV) | correcta | igual |
  | Con un hueco **después** de los pasos predichos | correcta | igual (no se usa) |
  | Un día sí y otro no (sin `freq`) | `[100, 102.4, 105.1, 108.1, 111.3]` **sin aviso** | `ValueError` |
  | Con una fecha duplicada | `[100, 102, 103.6, 105.3, 107]` **sin aviso** | `ValueError` |
  | Horaria en lugar de diaria | se usa sin aviso | `ValueError` |
  | Direct, `steps=[3,4,5]`, a saltos | `[104.8, 107.2, 109.6]` (correcto: `[104, 106, 108]`) | `ValueError` |
  | `RangeIndex` con `step=2` en lugar de 1 | se usa sin aviso | `ValueError` |
  | Diaria tz-aware que cruza el cambio de hora | correcta | igual |

- **Diseño:** un helper privado `_check_exog_alignment(exog_name, exog_index, last_window_index, last_step, lenient)`, llamado dentro del bucle de `check_predict_input`.
  1. Solo se comprueban las primeras `last_step` posiciones (`max(steps)` en Direct). Las filas de más y los huecos posteriores no afectan, porque no se usan.
  2. **Vía rápida, O(1), unos 2 µs:** si el índice de la exog tiene la misma `freq` que `last_window` (o el mismo `step` en `RangeIndex`) y empieza donde toca, no puede tener huecos, así que no hace falta más. Es el caso habitual: un índice creado con `date_range` o un `.loc`/`.iloc` de la exog original conservan la `freq`.
  3. **Vía completa, unos 76-82 µs** y casi constante con el horizonte (medido con 24, 168 y 1000 pasos): cuando la exog no tiene `freq` (por ejemplo, leída de un CSV), se construye el índice esperado y se compara con `equals`. El error indica la primera posición que no coincide y cómo corregirlo (punto 6).
  4. **El índice esperado se construye respetando el cambio de hora:** `pd.date_range(start=last_window_index[-1], periods=last_step + 1, freq=freq)[1:]`, en lugar de `index[-1] + n * freq`, que con zona horaria suma 24 h fijas y falla en el día del cambio (es el mismo problema que M-09). **Actualización (2026-10-05):** desde skforecast/skforecast#1343 hay dos convenciones para los índices tz-aware (hora local o anclado en UTC). Para no duplicar esa lógica, el índice esperado se obtiene con `expand_index(last_window_index, steps=last_step)`, ya corregido por M-09 en el commit anterior, en lugar de llamar a `pd.date_range` directamente.
  5. **La comprobación de la posición 0 conserva su mensaje actual** ("must start one step ahead of `last_window`"), para no romper los tests existentes. El mensaje nuevo solo aparece cuando falla una posición posterior.
  6. **El mensaje de error indica la solución** (añadido el 2026-10-04 a petición del usuario). Si faltan fechas sin dato, el usuario puede añadirlas como NaN explícito con `exog.reindex(...)`; `predict` las acepta con el `MissingValuesWarning` habitual. skforecast **no** hace este `reindex` automáticamente, por las razones de la opción b descartada en A-04: esconde errores como una exog horaria en un modelo diario, sería incoherente con `fit` y con LinearRegression da NaN desde el primer hueco. Texto propuesto:
     ```
     ValueError: `exog` must have consecutive values following the frequency of `last_window`
     for the 5 steps predicted. Expected 2020-02-21 at position 1, got 2020-02-22.
     If some dates have no data, add them explicitly as NaN, for example:
     exog = exog.reindex(expand_index(last_window.index, steps=5)).
     ```
     - `steps` se rellena con el valor real de `last_step` (`max(steps)` en Direct).
     - `expand_index` es pública (`skforecast.utils`) y, con el arreglo de M-09 (commit anterior del PR 1c), calcula bien las fechas en los días de cambio de hora.
     - El texto final se ajusta al implementarlo: hay que indicar que `last_window` es, por defecto, `forecaster.last_window_`; en multiserie con exog ancha, `last_window_` es un dict, así que el ejemplo usará la ventana de una de las series.
     - En la rama permisiva (exog dict) no se añade, porque allí el relleno con NaN ya es automático.
- **Prototipo** (`a03/a03_plugin.py`, sin tocar el repo):
  ```python
  def _check_exog_alignment(exog_name, exog_index, last_window_index, last_step, lenient):
      if len(exog_index) < last_step:
          return                                    # la longitud ya se valida antes
      if isinstance(last_window_index, pd.RangeIndex):
          if exog_index.step == last_window_index.step:
              return                                # vía rápida
          expected = pd.RangeIndex(...)             # start, n pasos, mismo step
      else:
          freq = last_window_index.freq
          if exog_index.freq is not None and exog_index.freq == freq:
              return                                # vía rápida
          expected = pd.date_range(start=last_window_index[-1], periods=last_step + 1, freq=freq)[1:]
      actual = exog_index[:last_step]
      if not actual.equals(expected):
          i = (actual != expected).nonzero()[0][0]
          msg = (f"{exog_name} must have consecutive values following the frequency of "
                 f"`last_window` for the {last_step} steps predicted. Expected "
                 f"{expected[i]} at position {i}, got {actual[i]}.")
          if lenient:
              warnings.warn(msg + " Values for the missing dates are filled with NaN.", MissingValuesWarning)
          else:
              raise ValueError(
                  msg + " If some dates have no data, add them explicitly as NaN, for example: "
                  f"exog = exog.reindex(expand_index(last_window.index, steps={last_step}))."
              )
  ```
- **Exog dict en multiserie (rama permisiva, decisión 1a):** se alinea por fecha, así que no da valores desplazados. Pero hoy un hueco produce predicciones NaN **sin ningún aviso**: con exog a saltos para la serie `a`, sale `[120, nan, nan, nan, nan]` y no se emite ningún warning. Con `lenient=True`, el helper emite un `MissingValuesWarning` en lugar de lanzar un error.
- **Interacción con M-09 (cambia el plan):** la comprobación actual de la posición 0 ya da un error falso cuando `last_window` termina el día del cambio de hora (`a03/dst.py`):
  ```
  last date: 2024-03-31 00:00:00+01:00 | exog starts: 2024-04-01 00:00:00+02:00 | expand_index -> 2024-04-01 01:00:00+02:00
  predict -> ValueError To make predictions `exog` must start one step ahead of `last_window`.
  ```
  Por eso **el arreglo de M-09 (`expand_index`) se adelanta al PR 1c**, en el commit anterior a A-03, y `check_predict_input` usará el mismo cálculo del índice esperado.
- **Coste:** `predict(24)` con exog tarda unos 814 µs, y `check_predict_input` hoy unos 193 µs. Con la vía rápida se añaden unos 2 µs (0,3 %); con la completa, unos 76 µs (9 %), y solo cuando la exog no tiene `freq`. En backtesting no hay coste, porque llama a `predict` con `check_inputs=False` (`_validation.py:329, 1090`) y la alineación de la exog con `y` ya se valida al entrenar.
- **Compatibilidad:** con el prototipo activo pasan **2279 tests (0 fallos)**, los mismos que sin él, en: `test_check_predict_input.py` y los tests de `ForecasterRecursive`, `ForecasterDirect`, `ForecasterDirectMultiVariate`, `ForecasterRecursiveClassifier`, `ForecasterStats`, `ForecasterRecursiveMultiSeries` y `ForecasterEquivalentDate` (sin `slow`). Ningún test existente pasa una exog desalineada a propósito.
- **Fuera de alcance:** `ForecasterEquivalentDate` no usa exog. `ForecasterFoundation` y `FoundationModel` no pasan por `check_predict_input`; su validación se revisaría aparte.
- **Riesgo / API:** las entradas que hoy dan predicciones erróneas pasan a lanzar un error. Una exog regular sin `freq`, una exog con filas de más y una exog con huecos después del horizonte siguen siendo válidas. Nota de versión en Fixed.
- **Tests a añadir:**
  - en `test_check_predict_input.py`: horaria, a saltos, duplicada, `RangeIndex` con otro `step`, hueco después del horizonte (no lanza error), sin `freq` (no lanza error), tz-aware que cruza el cambio de hora (no lanza error) y exog dict con hueco (`MissingValuesWarning`). Los tests del `ValueError` comprueban también que el mensaje incluye la pista con `reindex`;
  - un test de "ida y vuelta": la exog a saltos, después de `exog.reindex(expand_index(...))` como sugiere el mensaje, se acepta con `MissingValuesWarning`;
  - a nivel de forecaster: el caso "a saltos" en Recursive y en Direct con `steps=[3,4,5]`.

### A-04 · Multiseries con exog ancha: los avisos prometen NaN, pero se usan datos desplazados
> **Actualización (2026-10-05): implementado en el PR 1c con la opción b-light, no con la a.** La exog ancha se alinea por fecha y columna en `predict`, como en `fit`, en el backtesting y con la exog dict. El texto de abajo es la evaluación original; el porqué del cambio y las mediciones están en §5.5.

- **Dónde:** `check_predict_input`, `utils.py:1416, 1434, 1452, 1477`. Solo el camino **dict** de `_create_predict_inputs` alinea por etiqueta; el ancho es posicional.
- **Reproducción** (`v2.py`):
  ```
  exog empieza 3 días tarde -> warning "Missing values are filled with NaN"
  predicciones nivel 'a': [104.7, 107.7, 110.0, 112.0, 114.0]   correctas: [100.0, 102.0, 104.0, 106.0, 108.0]
  ```
  - Si la exog es más corta que `steps`, se reciclan filas.
  - Si falta una columna, da `KeyError` justo después del aviso de "All values will be NaN".
- **Decisión: opción a** (decisión 1a, §5.1). La exog ancha pasa a ser estricta, como en el resto de forecasters; la exog dict sigue siendo permisiva.
- **Fix:** `lenient = isinstance(exog, dict)`, y usar `if lenient:` en lugar de `if forecaster_name in ['ForecasterRecursiveMultiSeries']:` en las 4 ramas (:1416, :1434, :1452, :1477).
- **Opción b, descartada:** reindexar en el forecaster (`exog.reindex(index=prediction_index, columns=self.exog_names_in_)`). Lo probé en `q1.py` con el ejemplo de "empieza 3 días tarde":
  - LinearRegression devuelve `[nan, nan, nan, nan, nan]`.
  - LightGBM devuelve `[243.8, 189.1, 123.9, …]` en lugar de `[387.5, …]`.

  Es decir, solo convierte el error del usuario en predicciones NaN o degradadas, acompañadas de un aviso.
- **Por qué la opción a no rompe nada que funcione hoy:** todos los casos afectados ya dan valores desplazados, filas reutilizadas o un `KeyError`. El backtesting no se ve afectado, porque convierte la exog ancha a dict internamente (`_validation.py:1707`).
- **Riesgo / API:** es un cambio de comportamiento y necesita nota de versión (sección Changed). Hay que pasar a exog dict estos 5 tests de `test_check_predict_input.py`:
  - `test_check_predict_input_MissingValuesWarning_when_len_exog_is_less_than_steps_MultiSeries` (3 parametrizaciones);
  - `..._MissingExogWarning_when_exog_is_DataFrame_without_columns_in_exog_names_in__MultiSeries`;
  - `..._IgnoredArgumentWarning_when_exog_is_Series_with_name_not_in_exog_names_in__multiseries`.
- **Test:** añadir tests a nivel de forecaster con una exog ancha que empieza tarde y con una más corta que `steps`; los dos deben lanzar `ValueError`.

### A-05 · `Pipeline` con LightGBM o CatBoost y exog categórica: `fit` falla
- **Dónde:** `configure_estimator_categorical_features`, `utils.py:563-607`. Desenvuelve el Pipeline, pero devuelve `categorical_feature` / `cat_features` sin el prefijo `paso__`.
- **Reproducción** (`v4.py`):
  `ForecasterRecursive(make_pipeline(StandardScaler(), LGBMRegressor()), lags=3).fit(y, exog=<categórica>)` lanza `ValueError: Pipeline.fit does not accept the categorical_feature parameter`. Con XGB y HGB funciona.
- **Fix:**
  - Usar el prefijo `f"{estimator.steps[-1][0]}__"` en las claves.
  - En `cast_catboost_*`, buscar la clave con `k.split('__')[-1] == 'cat_features'`.
  - En `_build_predict_function`, desenvolver el Pipeline para CatBoost.
  - Prototipo en `b4/fix_pipeline_prefix.py`: las 4 combinaciones funcionan y dan predicciones idénticas a las del estimador sin Pipeline.
- **Riesgo / API:** solo es válido si los pasos previos del Pipeline mantienen las columnas en su sitio. La alternativa es avisar y no configurar las categóricas nativas.
- **Test:** `test_pipeline_extracts_last_step_lgbm` comprueba la clave errónea y hay que corregirlo. Añadir tests de fit y predict de punta a punta con Pipeline.

### A-06 · `CatBoostClassifier` en `ForecasterRecursiveClassifier`: `predict` falla
- **Dónde:** fuera de utils, en `_forecaster_recursive_classifier.py:1705/1709`. El cast a int en predict solo existe en la rama `CatBoostRegressor` de `_build_predict_function`, y el clasificador no la usa.
- **Reproducción** (`v4.py`): `fit` funciona; `predict` lanza `CatBoostError: 'data' is numpy array of floating point numerical type ... but 'cat_features' parameter specifies nonzero number`. Fallan también `predict_proba` y backtesting.
- **Fix:** generalizar la rama CatBoost de `_build_predict_function` a clasificadores y a `predict_proba`, o aplicar el cast en `_recursive_predict`. Prototipo en `b4/fix_clf_catboost.py`.
- **Test:** no hay ningún test de predict con CatBoostClassifier. Añadir predict, `predict_proba` y backtesting.

### A-07 · `Pipeline(FunctionTransformer, ...)` como transformer: `fit` falla
- **Dónde:** `transform_dataframe`, `utils.py:2341-2345`, y `transform_series`, `utils.py:2257`. `hasattr(transformer, 'get_feature_names_out')` es `True` en `Pipeline` y `ColumnTransformer`, pero la llamada falla si un paso interno no lo implementa.
- **Reproducción** (`v3.py`):
  `ForecasterRecursive(Ridge(), lags=3, transformer_y=make_pipeline(FunctionTransformer(np.log1p, np.expm1), StandardScaler())).fit(y)` lanza `AttributeError: Estimator functiontransformer does not provide get_feature_names_out`. Pasa lo mismo con `transformer_exog` y con `ColumnTransformer`.
- **Fix:**
  ```diff
  +def _safe_feature_names_out(transformer):
  +    if not hasattr(transformer, 'get_feature_names_out'):
  +        return None
  +    try:
  +        return transformer.get_feature_names_out()
  +    except (AttributeError, ValueError, TypeError):
  +        return None
  ```
  Usarlo en las dos funciones, con el fallback actual (`df.columns` o `transformed_i`).
- **Riesgo / API:** ninguno.
- **Test:** añadir un test de `transform_dataframe` con ese Pipeline y otro de `ForecasterRecursive.fit` con ese `transformer_y`.

### A-08 · skops: índice con zona horaria y cambio de hora
- **Dónde:** `_decompose_index` / `_compose_index`, `utils.py:2433-2479`. Se guarda `str(ts)`, que solo conserva el desfase UTC, y se carga con `pd.to_datetime(list)`.
- **Reproducción** (`v5.py`, serie horaria `Europe/Madrid` de enero a abril):
  ```
  save_forecaster(f, "tz", backend="skops")                 -> OK
  load_forecaster("tz.skops", trusted=True)                 -> ValueError: Tz-aware datetime.datetime cannot be converted to datetime64 unless utc=True
  ```
  - `training_range_` (primer y último timestamp) ya cruza el cambio de hora con cualquier serie de unos meses en Europa o EEUU.
  - Sin cambio de hora, `Europe/Madrid` vuelve como `UTC+01:00` y las etiquetas de predicción quedan desplazadas una hora tras el cambio.
  - joblib funciona bien.
- **Fix:** guardar `index.asi8` + `unit` + `tz` y reconstruir con `tz_localize('UTC').tz_convert(tz)`, con un fallback que siga cargando el formato antiguo (`format='ISO8601'` y `utc=True` si los desfases son mixtos). El diff completo está en el informe del bloque 5. Arregla también M-15 y O-01.
- **Riesgo / API:** cambia el formato interno del payload; el fallback mantiene la compatibilidad con los ficheros de 0.23-0.25.
- **Test:** hay que actualizar `test_decompose_index_output[datetime]`. Añadir round trips con cambio de hora y un payload antiguo.

### A-09 · `RollingFeatures` guarda un objeto `Rolling` con toda la serie
- **Dónde:** aflora en `save_forecaster` (`utils.py:2753`). La causa está en `preprocessing/_preprocessing.py:1429-1430` y `:2060-2061`: `transform_batch` guarda `X.rolling(...)` en `self.unique_rolling_windows[k]['rolling_obj']`.
- **Reproducción** (`v5.py`):
  ```
  save_forecaster(ForecasterRecursive(..., window_features=RollingFeatures(['mean'], 7)) entrenado, backend='skops')
  -> TypeError: no default __reduce__ due to non-trivial __cinit__
  type(rolling_obj) tras fit: Rolling
  ```
  - Afecta a Recursive, Direct, MultiSeries, DirectMultiVariate y Classifier.
  - Con joblib, 500k filas: 15.6 MB frente a 8.8 KB.
- **Fix:** dejar los objetos `Rolling` en una variable local dentro de `transform_batch`. La clave `'rolling_obj': None` se mantiene porque los tests la comprueban.
- **Riesgo / API:** ninguno. 76 tests pasan con el cambio emulado.
- **Test:** añadir un round trip skops con `window_features` para cada tipo de forecaster.

### A-10 · `set_cpu_gpu_device`
- **Dónde:** `utils.py:3152-3187`.
- **Reproducción** (`v4.py`, `mine.py`):
  ```
  ForecasterRecursive(XGBRegressor(device='cuda:0')).fit(y); predict -> ValueError: `device` must be 'gpu', 'cpu', 'cuda', or None.
  set_cpu_gpu_device(<XGBRegressor>, 'GPU') -> KeyError: 'GPU'   ('GPU'/'CPU' están en valid_devices pero no en device_values de XGB/LGBM)
  LGBM device='cuda' -> tras predict queda 'gpu' (en LightGBM >= 4 son backends distintos: CUDA frente a OpenCL)
  ```
  - La rama de CatBoost es código muerto: `getattr(est, 'task_type')` siempre es `None`, y `set_params` sobre un modelo entrenado falla en silencio por el `except: pass`.
  - Los llamadores no restauran el dispositivo en un `try/finally`.
- **Fix:**
  - Leer el dispositivo con `estimator.get_params().get(param_name)`.
  - Aceptar cualquier `str` y restaurarlo tal cual: `device_values[...].get(device, device)`.
  - Mapear `'cuda': 'cuda'` en LGBM.
  - Quitar o documentar la rama de CatBoost.
  - Envolver los llamadores en `try/finally`.
- **Test:** añadir un round trip fit → predict con `'cuda:0'` y con LGBM `'cuda'`. No hace falta GPU, porque se puede fijar después de entrenar.

### M-01 · CatBoost + `.cat.codes`: categorías desplazadas en la búsqueda one-step-ahead multiserie
- **Dónde:** `cast_catboost_categorical_columns_dataframe`, `utils.py:754-755`. La causa está en `_forecaster_recursive_multiseries.py:1316`: `pd.Categorical(encoded_values)` sin `categories=`.
- **Reproducción** (`b4/r04b.py`): las categorías de train son `[0, 1, 2]` y las de test `[0, 2]`, así que el nivel `c` se convierte en 1. MAE de `c`: 6.67; con el fix, 1.39. LightGBM, XGB y HGB no están afectados.
- **Fix:**
  - En la causa raíz: `pd.Categorical(encoded_values, categories=range(len(self.encoding_mapping_)))`.
  - En la función, de forma defensiva: con categorías numéricas, usar los valores en lugar de `.cat.codes`.
- **Test:** añadir un caso con categorías numéricas en el que falte un nivel.

### M-02 · Subclases de `LinearModel` con `predict` propio
- **Dónde:** `utils.py:3233`.
- **Reproducción** (`v4.py`): con `class NonNegRidge(Ridge)` cuyo `predict` hace `clip(0)`, `forecaster.predict(5).min()` da `-6.906`.
- **Fix:** `if isinstance(estimator, LinearModel) and type(estimator).__module__.startswith('sklearn.'):`. La misma guarda conviene para RF y DT.
- **Riesgo / API:** ninguno; las subclases pasan al camino genérico.

### M-03 · `last_window` con varias columnas en forecasters de una serie
- **Dónde:** `utils.py:1324-1329`.
- **Reproducción** (`v2.py`): con un `last_window` de 2 columnas sale `[79.4, 97.7, 103.1]` en lugar de `[100.0, 102.0, 104.0]`, sin error. En Direct y EquivalentDate pasa lo mismo.
- **Fix:**
  ```diff
  +        if isinstance(last_window, pd.DataFrame) and last_window.shape[1] != 1:
  +            raise ValueError(f"`last_window` must be a pandas Series or a DataFrame with a single column. "
  +                             f"Got {last_window.shape[1]} columns.")
  ```

### M-04 · `prepare_steps_direct`
- **Dónde:** `utils.py:3870-3887`. Usa `isinstance(steps, int)` y no tiene rama `else`. Además se llama antes de `check_predict_input`, que sí acepta `np.integer`.
- **Reproducción** (`mine2.py`):
  ```
  ForecasterDirect.predict(steps=np.int64(3)) -> UnboundLocalError: cannot access local variable 'steps_direct'
  predict(steps=0) / predict(steps=[])        -> ValueError: min() iterable argument is empty
  ```
  Afecta también a `ForecasterDirectMultiVariate` y `ForecasterRnn`.
- **Fix:** aceptar `(int, np.integer)` con `int(steps)`, añadir un `else: raise TypeError(...)` y lanzar `ValueError` si la lista queda vacía.

### M-05 + N-01 · pandas 2.1: `y` nullable y APIs de pandas 2.2
- **Dónde:**
  - `check_extract_values_and_index`, `utils.py:1730`. Con pandas < 2.2, `to_numpy()` de `Int64`/`Float64` devuelve `object`.
  - N-01: `preprocessing/_preprocessing.py:425` (`groupby(...).apply(..., include_groups=False)`) y `preprocessing/_calendar.py:694-695` (alias `'YE'`, `'QE'`, `'ME'`). Las dos son APIs de pandas 2.2.
- **Reproducción:**
  - `b1/cev1.py`, con pandas 2.1.4: una `y` `Int64` o `Float64` da `TypeError: ufunc 'isnan' not supported`. Con pandas 2.3.3 funciona.
  - Tests de utils, recursive, multiserie, direct y preprocessing con pandas 2.1.4, numpy 1.26.4 y scikit-learn 1.4.2: **1219 passed, 17 failed y 51 errores de recogida**.
    - Algunos errores se deben solo a paquetes opcionales que no instalé en ese entorno.
    - Los demás vienen de `include_groups` en `reshape_series_wide_to_long` y del alias `'YE'`.
- **Contexto:**
  - El CI nunca instala las versiones mínimas: `unit-tests.yml` instala lo último que resuelve y `unit-tests-latest-deps.yml` actualiza.
  - pandas 2.1 no es compatible con numpy 2 ni con Python 3.13, que el CI sí prueba.
- **Decisión: subir el mínimo a `pandas>=2.2`** (decisión 2b, §5.1). M-05 y N-01 desaparecen sin tocar el código; no se añade el arreglo de compatibilidad.
- **Ficheros a cambiar:**
  - `pyproject.toml:56` (`"pandas>=2.2, <3.0"`) y `:121` (`"pandas[parquet]>=2.2"`).
  - `docs/quick-start/how-to-install.md:33`.
  - `tools/ai/ai_context_header.md:39` (línea `Core:`), y después regenerar con el skill `ai-context-sync`. `AGENTS.md` y `.github/copilot-instructions.md` son ficheros generados y no se editan a mano.
  - Nota de versión en `docs/releases/releases.md` (sección Changed).
- **Seguimiento propuesto (no incluido en el plan):** un job de CI con las versiones mínimas (pandas 2.2, numpy 1.26, scikit-learn 1.4). Habría detectado N-01 y M-13. **Documentado como PR 7 (§5.9).**

### M-06 · `weight_func` como `partial` o invocable
- **Dónde:** `initialize_weights`, `utils.py:297-302`.
- **Reproducción** (`v1.py`): `ForecasterRecursive(Ridge(), lags=3, weight_func=functools.partial(w, cutoff=...))` lanza `TypeError: module, class, method, function ... expected, got partial`.
- **Fix:** envolver `inspect.getsource` en `try/except (OSError, TypeError)` y devolver `None`. `source_code_weight_func` solo es informativo.

### M-07 · `align_series_and_exog_multiseries` con dtypes nullable
- **Dónde:** `utils.py:3682`.
- **Reproducción** (`b2/r18*.py`): con `Float64` o `double[pyarrow]` y NA al principio, `fit` falla con `TypeError: boolean value of NA is ambiguous`.
- **Fix:** usar `pd.isna(...)` en lugar de `np.isnan(...)`.

### M-08 · `check_residuals_input` valida niveles que no se predicen
- **Dónde:** `utils.py:1673-1679`.
- **Reproducción** (`b2/r9*.py`): `predict_interval(levels=['a'], use_in_sample_residuals=False)` con residuos solo para `a` da `ValueError: Residuals for level 'b' are None`.
- **Fix:** recorrer solo `levels`, con el fallback a `'_unknown_level'` que usan los forecasters, y mejorar el mensaje.
- **Test:** hay que reescribir `test_check_residuals_input_ValueError_when_residuals_for_some_level_is_None`.

### M-09 · `expand_index` con índice diario tz-aware el día del cambio de hora
- **Dónde:** `utils.py:2046-2050`. `index[-1] + Day` suma 24 h fijas.
- **Reproducción** (`v3.py`): `expand_index(<D Europe/Madrid hasta 2024-03-31>, 2)` da `[2024-04-01 01:00+02:00, 2024-04-02 01:00+02:00]`, cuando debería dar las 00:00.
- **Fix:** `pd.date_range(start=index[-1], periods=steps + 1, freq=freq)[1:]`. Es idéntico en 11 frecuencias sin zona horaria.
- **Actualización (2026-10-05):** skforecast/skforecast#1343 añadió a `expand_index` una rama para los índices "anclados en UTC" (`_is_utc_anchored_index`), pero la rama habitual (índice creado en hora local) sigue usando `index[-1] + freq`, y el bug se reproduce igual en `2e365b6`: `[2024-04-01 01:00+02:00, 2024-04-02 01:00+02:00]`, y `predict` con exog lanza el `ValueError` falso. El fix se aplica solo a esa rama `else`; la rama UTC no lo necesita, porque UTC no tiene cambio de hora. Los tests nuevos de `test_expand_index.py` de ese PR deben seguir pasando.

### M-10 · `date_to_index_position`
- **Dónde:** `utils.py:1971-1989`.
- **Reproducción** (`b3/t_tz_e2e.py`, `t_date2.py`):
  - **Zona horaria:** con una serie tz-aware, `predict(steps='2024-01-06 05:00')` y `TimeSeriesFold(initial_train_size='2024-01-03')` dan `TypeError: Cannot compare tz-naive and tz-aware`.
  - **Sin `freq`:** con un índice horario sin `freq`, `initial_train_size='2020-01-02 00:00'` da **2** en lugar de 25, sin ningún aviso.
- **Fix:**
  - Localizar `target_date` en `index.tz`.
  - En `validation`, usar `int(index.searchsorted(target_date, side='right'))`.
  - En `prediction` sin `freq`, inferirla o lanzar un error explícito.

### M-11 · `transform_series` con una sola fila
- **Dónde:** `utils.py:2247`.
- **Reproducción** (`mine.py`, `b3/t_stats_e2e.py`):
  - `transform_series(<1 fila>, StandardScaler().set_output('pandas'))` devuelve `numpy.float64`.
  - `ForecasterStats(..., transformer_y=...).predict(last_window=<1 obs>)` lanza `TypeError: object of type 'numpy.float64' has no len()`.
- **Fix:** `values_transformed.iloc[:, 0]`.

### M-12 · Las categóricas `'auto'` sobrescriben la configuración del usuario
- **Dónde:** `utils.py:569-577, 622, 638`.
- **Reproducción** (`b4/r07*.py`, `r14*.py`):
  - Con `HistGradientBoostingRegressor(categorical_features=[3])` y sin exog categórica, tras `fit` queda `from_dtype` sin ningún aviso; con XGB `feature_types` pasa lo mismo.
  - En cada refit salta un `IgnoredArgumentWarning` por el valor que puso el propio skforecast.
  - Contradice la nota del user guide (`categorical-features.ipynb`, celda 35).
- **Fix:**
  - Avisar solo si el valor previo no es el de por defecto **y** es distinto del nuevo.
  - En la rama de reset, avisar si el valor previo no es el de por defecto.
  - Corregir la nota del user guide.

### M-13 · ExtraTrees y NaN con sklearn < 1.6
- **Dónde:** `utils.py:50-64` (el comentario también es incorrecto).
- **Reproducción** (`b4/r12*.py`, `r13*.py` en entornos con sklearn 1.4.2 y 1.5.2): `backtesting_forecaster_multiseries` con ExtraTrees y NaN en `last_window` da `ValueError Input X contains NaN`; con Ridge, el nivel se descarta sin problema.
- **Fix:** añadir los 4 nombres ExtraTree* solo si `Version(sklearn.__version__) >= Version('1.6')`.
- **Test:** `test_estimator_has_native_nan_support_true` tiene que depender de la versión.

### M-14 · skops con exog categórica
- **Dónde:** `utils.py:2594`. `exog_dtypes_in_` contiene un `CategoricalDtype`.
- **Reproducción** (`b5/rt5.py`): `TypeError: no default __reduce__ due to non-trivial __cinit__`.
- **Fix:** descomponer y recomponer los dtypes (`_decompose_dtype` / `_compose_dtype`) en `exog_dtypes_in_` y `exog_dtypes_out_`, junto con el fix de B-05.

### M-15 · skops con frecuencia por debajo del segundo
- **Dónde:** `utils.py:2476`. `str(Timestamp)` omite la parte fraccionaria cuando vale cero.
- **Reproducción** (`b5/rt6.py`): con `500ms`, la carga falla con `ValueError: time data "2020-01-06 00:00:39" doesn't match format`.
- **Fix:** lo cubre el fix de A-08.

### M-16 · Nombres de fichero con puntos
- **Dónde:** `save_forecaster`, `utils.py:2709`.
- **Reproducción** (`v5.py`):
  ```
  save 'model_v1.1' (lags=3), save 'model_v1.2' (lags=5)
  ficheros: ['model_v1.joblib'], lags cargados: [1 2 3 4 5]   <- el primer modelo se pierde sin aviso
  ```
- **Decisión: opción a** (decisión 3a, §5.1). Sustituir el sufijo solo si es una extensión conocida (`.joblib`, `.pkl`, `.pickle`, `.cloudpickle`, `.skops`); si no, añadir la del backend.
  ```diff
  -    file_name = Path(file_name).with_suffix(backend_extensions[backend])
  +    file_name = Path(file_name)
  +    if file_name.suffix.lower() in {'.joblib', '.pkl', '.pickle', '.cloudpickle', '.skops'}:
  +        file_name = file_name.with_suffix(backend_extensions[backend])
  +    else:
  +        file_name = file_name.with_name(file_name.name + backend_extensions[backend])
  ```
- **Comportamiento resultante:**

  | `file_name` | Antes | Después |
  |---|---|---|
  | `'model'` | `model.joblib` | `model.joblib` |
  | `'model.pkl'` (joblib) | `model.joblib` | `model.joblib` |
  | `'model_v1.2'` | `model_v1.joblib` | `model_v1.2.joblib` |
  | `'forecaster_2026.10.04'` | `forecaster_2026.10.joblib` | `forecaster_2026.10.04.joblib` |
  | `'model.bin'` | `model.joblib` | `model.bin.joblib` |

- **Riesgo / API:** es un cambio de comportamiento y necesita nota de versión (sección Changed). Hay que:
  - actualizar el docstring de `file_name`;
  - corregir la frase del user guide "regardless of the extension originally passed" (`docs/user_guides/save-load-forecaster.ipynb`, celda 13).

  `tools/ai/llms-base.txt` y `skills/` no mencionan este comportamiento, así que no hay contexto de IA que tocar.
- **Test:** añadir dos casos: `'model_v1.2'` → `model_v1.2.joblib`, y `'f.pkl'` con joblib → `f.joblib`.
- **Mejora opcional (no decidida):** que `save_forecaster` devuelva la ruta final; hoy devuelve `None`.

### M-17 · `CalibratedClassifierCV` y categóricas nativas
- **Dónde:** `configure_estimator_categorical_features` y los helpers `cast_*` y get/restore (solo desenvuelven `Pipeline`).
- **Reproducción** (`b4/r06*.py`): `use_native_categoricals=True`, pero los `feature_infos` de LGBM muestran bins numéricos.
- **Fix:** un helper común `_unwrap_estimator()` que desenvuelva el `Pipeline` y luego `CalibratedClassifierCV.estimator`.

### Bajos (B-01 a B-21)
Para cada uno, la reproducción está en los scripts indicados y el fix es local:

- **B-01** `check_preprocess_series:3464`: `sorted(indexes_freq, key=str)`. Repro: `mine.py`; con `ForecasterRecursiveMultiSeries.fit` y series D + MS da `TypeError '<' not supported between ... MonthBegin and Day`.
- **B-02** `check_select_fit_kwargs:515`: quitar el `del` y filtrar `k != 'sample_weight'` en la comprensión. Repro: `mine.py`; el dict del usuario queda `{'foo': 1}`.
- **B-03** `initialize_lags:109,138-140`:
  - Excluir `bool`; aceptar `np.integer`.
  - Hacer `np.sort(lags).astype(np.int64)` y `max_lag = int(lags[-1])`.
  - Repro: `v1.py`; con lags `uint8`, `window_size=np.uint8(3)`, `last_window_` queda vacío y `predict` falla.
  - Afecta también a `grid_search` con `lags_grid=list(np.arange(...))`.
- **B-04** `deepcopy_forecaster`: sustituir los objetos pesados vía el `memo` de `deepcopy`, sin modificar el original. Prototipo en `b4/fix_deepcopy_memo.py`: 35/35 tests pasan, y además resuelve O-07.
- **B-05** `save_forecaster` skops: descomponer una copia superficial, `copy(forecaster)`. Repro: `b5/race.py` (131 753 errores de `predict` concurrente); con la copia, 0.
- **B-06** `_decompose_index`: guardar `index.freq` en lugar de `freqstr`. Repro: `b5/rt1.py`.
- **B-07** `save_forecaster:2803`: decidir por módulo (`type(wf).__module__.startswith('skforecast.')`).
- **B-08** `save_forecaster:2779`: `open(..., 'w', encoding='utf-8')`.
- **B-09** `check_exog_dtypes`: usar `pd.api.types.is_numeric_dtype` / `is_integer_dtype` en lugar de prefijos de nombre. Prototipo: 30/30 tests pasan.
- **B-10** `cast_exog_dtypes`: es pública (añadida en 0.8.0), no tiene tests, nadie la usa y está marcada `# pragma: no cover`.
  - **Decisión: deprecarla** (decisión 4, §5.1). Se usa el decorador existente `runtime_deprecated` de `skforecast/exceptions/exceptions.py`, con `FutureWarning`. El mensaje indicará la alternativa `exog.astype(forecaster.exog_dtypes_in_)`. Eliminación propuesta: 0.28.
  - No se corrigen los bugs internos.
  - Hay que añadir la nota de versión (sección Deprecated o Changed) y un aviso en `docs/api/utils.md`. Ningún fichero de contexto de IA la menciona.
  - Test: uno solo, que compruebe que se emite el `FutureWarning`.
- **B-11** `check_predict_input`: convertir a frame una Series con nombre válido, para que salte el mensaje "Missing columns".
- **B-12** `check_predict_input:1476`: `if len(exog_index) > 0 and ...`.
- **B-13** `check_predict_input:1337`: buscar NaN solo en `last_window.iloc[-window_size:]` y en las columnas de `levels` (excepto en Stats).
- **B-14** `check_preprocess_exog_multiseries:3524`: llamar a `check_exog` antes de `to_frame()`. La rama `else [exog.name]` de :3638 es inalcanzable.
- **B-15** `check_preprocess_exog_multiseries:3629`: `list(dict.fromkeys(...))`, más una comprobación explícita de columnas duplicadas. Repro: `b2/r14*.py` con `PYTHONHASHSEED=1,2,3` da 3 órdenes distintos. El modelo no se ve afectado, porque multiserie recalcula los nombres; sí lo que expone `FoundationModel.exog_names_in_`.
- **B-16** `check_preprocess_series`: recoger `idx.tz` y lanzar un `ValueError` si hay más de una.
- **B-17** `transform_series:2223`: renombrar la columna de entrada al nombre visto en `fit`, en lugar de modificar el transformer.
- **B-18** `preprocess_levels_self_last_window_multiseries:3778`: `if len(levels) == 0:`.
- **B-19** `exog_to_direct*`: validar `1 <= steps <= len(exog)`.
- **B-20** `initialize_window_features`: excluir `bool` y la lista vacía.
- **B-21** `input_to_frame`: añadir `'exog_val': 'exog'` al mapeo.

---

## 3. Documentación

- **`check_exog`:** el docstring está invertido ("If `allow_nan = True`, issue a warning"; "default True ... If False (default)").
- **`check_y`:** dice "warning message", pero solo se usa en errores.
- **`initialize_differentiator_multiseries`:**
  - El docstring describe `differentiation` (int o dict) en lugar de `differentiator`.
  - Dice "cloning", pero usa `copy`/`deepcopy`.
  - El aviso incluye `'_unknown_level'`; es inalcanzable desde los forecasters, porque el `__init__` exige esa clave.
- **`initialize_weights`:** el mensaje de error dice "ints.Got" (sin espacio) y el test lo comprueba literalmente.
- **`check_predict_input`:**
  - `levels` admite `str` según el docstring, pero `set('l1')` lo parte en caracteres (inalcanzable desde los forecasters).
  - `index_freq_` no es `str`.
  - `window_size` no es solo `max_lag`.
- **`check_preprocess_series`:** "all series must have the same index" es falso (solo tipo y frecuencia).
- **`align_series_and_exog_multiseries`:** `exog_dict` "default None", pero no tiene valor por defecto.
- **`date_to_index_position`:** el mensaje fija "`steps`" e ignora `date_literal`; el mensaje de `validation` dice "greater than / less than", pero acepta los extremos.
- **`exog_to_direct_numpy`:** dice `shape(samples,)`, pero acepta arrays 2D.
- **`multivariate_time_series_corr`:** no dice que `lags=n` incluye el lag 0.
- **`deepcopy_forecaster`:**
  - Dice que usa `copy.copy` para Stats, pero usa `clone`.
  - No menciona que con Rnn copia el modelo entrenado.
- **`set_cpu_gpu_device`:** no tiene secciones Parameters ni Returns.
- **`show_versions`:** omite `scipy` (dependencia core), `statsmodels`, `matplotlib`, `skops` y `cloudpickle`.
- **Comentario en :50-52** sobre el soporte de NaN: es incorrecto para ExtraTrees (ver M-13).
- **`tools/ai/ai_context_header.md`:** dice `matplotlib<3.11`, pero `pyproject.toml` y `optional_dependencies` dicen `<3.12`.

---

## 4. Revisado y sin problemas

- **`manage_warnings`:**
  - `@wraps` conserva los metadatos.
  - Restaura los filtros también tras una excepción.
  - No es thread-safe, pero es inherente a `catch_warnings`, y skforecast paraleliza con procesos.
- **`check_interval`:** el test de simetría `a + b != 1.` no falla con ningún literal decimal razonable. Se probaron 5547 decimales y 2M pares aleatorios.
- **`select_n_jobs_fit_forecaster`:** la heurística se confirma con medidas (LGBM con n_jobs=3 sobresuscrito: 178 s frente a 1 s).
- **`transform_numpy`:** correcto en 1D/2D, sparse, salida pandas, inversa vectorizada y `force_single_column`.
- **`scale_correction_factor_differentiation`:** la matemática es correcta (d=1: √h; d=2: [1, 2.236, 3.742]; d=3: [1, 3.162, 6.782]).
- **`load_forecaster`:** el mapeo de `trusted`, la comprobación de versión y la inferencia del backend son correctos (al margen de lo que hereda de A-08 y M-15).
- **Varias funciones más:** `check_optional_dependency`, `_find_optional_dependency`, `get_style_repr_html`, `get_exog_dtypes`, `initialize_transformer_series`, `prepare_levels_multiseries`, `_get/_restore_estimator_categorical_set_params`.
- **`exog_to_direct*`:** el orden de columnas y la alineación de filas son correctos (aparte de B-19).
- **Atajos LightGBM, RandomForest y DecisionTree** de `_build_predict_function`: equivalentes a `predict`. LightGBM sí respeta `best_iteration`.
- **Falsos positivos descartados:**
  - Lags duplicados: `fit` los detecta con un error claro.
  - Copia superficial del differentiator: es segura.
  - CatBoost con categorías float ("1.0" frente a "1"): consistente entre fit y predict.
  - `device_type` de LGBM: tiene prioridad sobre el alias.

---

## 5. Decisiones y plan de implementación

### 5.1 Decisiones tomadas (2026-10-04)

| # | Pregunta | Decisión | Consecuencias |
|---|---|---|---|
| 1 | Multiserie con exog ancha | **a: estricta.** Solo la exog dict sigue siendo permisiva | Lanza `ValueError` cuando hoy hay valores desplazados o un `KeyError`. Cambian 5 tests de `test_check_predict_input.py`. Nota de versión en Changed. Ver A-04 |
| 1 (revisada) | Multiserie con exog ancha | **b-light** (2026-10-05): alinear por fecha y columna, como dict, `fit` y backtesting | Implementado en el PR 1c. Sustituye a la opción a. Ver §5.5 |
| 2 (ampliada) | Versiones mínimas | **`pandas>=2.2` y `scikit-learn>=1.6`** (2026-10-05). Solo se suben los mínimos y se explica el motivo; sin arreglos de compatibilidad ni job de CI | Resuelve M-05, N-01, M-13 y N-09. Ver §5.6 |
| 2 | `y` nullable con pandas 2.1 | **b: subir el mínimo a `pandas>=2.2`** | Resuelve M-05 y N-01 sin tocar el código. Se cambian `pyproject.toml` (2 líneas), `how-to-install.md` y `ai_context_header.md`, se regeneran los ficheros de contexto de IA y va nota en Changed |
| 3 | Nombres de fichero con puntos | **a: sustituir solo las extensiones conocidas** y añadir en los demás casos | `model_v1.2` → `model_v1.2.joblib`; `model.bin` → `model.bin.joblib`. Se actualizan el docstring y la celda 13 del user guide; nota en Changed. Ver M-16 |
| 4 (revisada) | `cast_exog_dtypes` | **Eliminar** (2026-10-05), sin periodo de deprecación: nadie la usa en ninguna versión publicada desde 0.8.0 | Ver §5.6 |
| 4 | `cast_exog_dtypes` | **Deprecar** con `runtime_deprecated` (`FutureWarning`), eliminación propuesta en 0.28 | No se arreglan sus bugs internos. Se añaden la nota de versión y el aviso en `docs/api/utils.md`. Ver B-10 |
| 5 | Arreglos fuera de `utils.py` | **Sí, se incluyen**, en PRs separados por tema contra `0.26.x` | Entran `direct/`, `recursive/_forecaster_recursive_classifier.py`, `recursive/_forecaster_recursive_multiseries.py`, `recursive/_forecaster_recursive.py`, `preprocessing/` y `deep_learning/_forecaster_rnn.py` |

### 5.2 Plan de implementación (PRs 1a, 1b y 1c hechos; el resto pendiente)

**Normas para todos los PRs:**
- Rama base `0.26.x`.
- Pasar el skill `verify` antes de cada PR: solo los tests afectados, nunca la suite completa.
- Nota de versión en `docs/releases/releases.md`, en la sección de 0.26.
- Corregir los docstrings del §3 en el mismo PR que toca cada función.

Los PRs 1c y 5 tocan `check_predict_input`, así que el 1c va antes. Los PRs 1a y 2 tocan `_build_predict_function`, así que el 1a va antes que el 2. El PR 6 es independiente y puede ir en cualquier momento.

El antiguo PR 1 (predicciones erróneas sin aviso) se divide en tres PRs de máxima prioridad, uno por zona del código. Ver el motivo en §5.4.

#### PR 1a · Atajos de predicción de `_build_predict_function`

| Hallazgo | Ficheros | Tests a tocar |
|---|---|---|
| A-01 XGB: `best_iteration`, `gblinear`, `missing` | `utils.py` (`_build_predict_function`) | Añadir casos en `test_build_predict_function.py` |
| M-02 subclases de `LinearModel` (misma guarda para RF y DT) | `utils.py` | Añadir un caso con subclase |

Nota de versión: Fixed (A-01, M-02).

#### PR 1b · Direct con diferenciación y pasos no consecutivos

| Hallazgo | Ficheros | Tests a tocar |
|---|---|---|
| A-02 Direct + `differentiation` con pasos no consecutivos o `gap` | `direct/_forecaster_direct.py`, `direct/_forecaster_direct_multivariate.py` | Nuevos: `predict(steps=[3,4,5]) == predict(5)[2:]`, intervalos y backtesting con `gap` |

Nota de versión: Fixed (A-02).

#### PR 1c · Validación de la exog y de `last_window` en `predict`

| Hallazgo | Ficheros | Tests a tocar |
|---|---|---|
| M-09 `expand_index` con zona horaria y cambio de hora (adelantado desde el PR 4, porque A-03 lo necesita) | `utils.py` | Casos `Europe/Madrid` en primavera y otoño; `predict` con exog cuando `last_window` acaba el día del cambio |
| A-03 exog con otra frecuencia o con huecos (diseño en A-03) | `utils.py` (`_check_exog_alignment` + `check_predict_input`) | Los listados en A-03 |
| A-04 exog ancha estricta (decisión 1a) | `utils.py` (`check_predict_input`) | Cambiar los 5 tests indicados en A-04; añadir tests a nivel de forecaster |
| M-03 `last_window` con varias columnas | `utils.py` (`check_predict_input`) | Nuevo |

Nota de versión: Fixed (A-03, M-03, M-09) y Changed (A-04).

#### PR 2 · Estimadores: categóricas, CatBoost, NaN y dispositivo

**Dividido el 2026-10-06 en PR 2a y PR 2b, con cambios de diseño en A-05, M-12 y A-10, y O-03 movido al PR 5b. Ver §5.7.** La tabla siguiente es el plan original.

| Hallazgo | Ficheros | Tests a tocar |
|---|---|---|
| A-05 `Pipeline` + LGBM/CatBoost con categóricas | `utils.py` (`configure_estimator_categorical_features`, `cast_catboost_*`, `_build_predict_function`) | Corregir `test_pipeline_extracts_last_step_lgbm`; añadir fit y predict de punta a punta |
| A-06 `CatBoostClassifier` en predict | `recursive/_forecaster_recursive_classifier.py` (o generalizar `_build_predict_function`) | Nuevos: predict, `predict_proba` y backtesting |
| M-01 CatBoost `.cat.codes` | `recursive/_forecaster_recursive_multiseries.py:1316` (causa raíz) y `utils.py` (defensa) | Nuevo, con categorías numéricas en las que falta un nivel |
| M-12 reset silencioso de categóricas y aviso falso en cada refit | `utils.py`; `docs/user_guides/categorical-features.ipynb` (celda 35) | Actualizar `test_histgbr_reset_*` y `test_xgboost_reset_*` |
| M-17 `CalibratedClassifierCV` (y comprobar S-03, `TransformedTargetRegressor`) | `utils.py`: helper común `_unwrap_estimator()` | Nuevo |
| M-13 NaN en ExtraTrees según la versión de sklearn | `utils.py` (`_SKLEARN_NAN_TOLERANT_ESTIMATORS` y su comentario) | `test_estimator_has_native_nan_support_true` debe depender de la versión |
| O-03 ExtraTrees en el camino rápido (**después de M-13**) | `utils.py` (`_build_predict_function`) | Añadir un caso de equivalencia |
| A-10 `set_cpu_gpu_device` | `utils.py`; `try/finally` en los llamadores de `recursive/_forecaster_recursive.py`, multiseries y classifier | Nuevos: round trips con `'cuda:0'` (XGB) y `'cuda'` (LGBM), sin GPU |

Nota de versión: Fixed. Opcional en Enhanced: O-03.

#### PR 3 · Persistencia (`save_forecaster` / `load_forecaster`)

| Hallazgo | Ficheros | Tests a tocar |
|---|---|---|
| A-08 + M-15 + O-01 índice con epochs, unidad y zona horaria, con fallback al formato antiguo | `utils.py` (`_decompose_index`, `_compose_index`) | Actualizar `test_decompose_index_output[datetime]`; añadir round trips con cambio de hora, `500ms` y payload antiguo |
| A-09 + O-02 `RollingFeatures` guarda `Rolling` | `preprocessing/_preprocessing.py` (`RollingFeatures` y `RollingFeaturesClassification`) | Nuevos: skops con `window_features` para cada tipo de forecaster; `rolling_obj is None` tras `fit` |
| B-05 volcar una copia superficial (lo necesita M-14) | `utils.py` (`save_forecaster`) | El test de no mutación ya existe |
| M-14 skops con exog categórica | `utils.py` (`_skops_decompose/reconstruct_forecaster`) | Nuevo, una serie y multiserie |
| M-16 nombres con puntos (decisión 3a) | `utils.py`; `docs/user_guides/save-load-forecaster.ipynb` (celda 13) | Nuevos: `'model_v1.2'` y `'f.pkl'` |
| B-06 festivos de `CustomBusinessDay` | `utils.py` (`_decompose_index`) | Nuevo |
| B-07 aviso falso con `RollingFeaturesClassification` | `utils.py` | Nuevo |
| B-08 `.py` en UTF-8 | `utils.py` | No hace falta test específico |
| S-04 `weight_func` invocable en `__main__` al guardar | `utils.py` | Investigar al implementar M-06 |

Nota de versión: Fixed y Changed (M-16).

#### PR 4 · Transformaciones, índices y fechas

| Hallazgo | Ficheros |
|---|---|
| A-07 `Pipeline(FunctionTransformer, …)` como transformer | `utils.py` (`transform_dataframe`, `transform_series`) |
| M-10 `date_to_index_position` con zona horaria y sin `freq` | `utils.py` |
| M-11 `transform_series` con una sola fila | `utils.py` |
| M-04 `prepare_steps_direct` (`np.integer`, tupla, lista vacía) | `utils.py` |
| B-17 `transform_series` con `Pipeline` y otro nombre | `utils.py` |
| B-18 `levels` como `pd.Index` | `utils.py` |
| B-19 `exog_to_direct*` con `steps` fuera de rango | `utils.py` |

Nota de versión: Fixed.

#### PR 5 · Validación, robustez y optimizaciones (después del PR 1c)

- **Validación de entradas:**
  - Generales: M-06, B-02, B-03, B-09, B-20.
  - Multiserie: M-07, B-01, B-14, B-15, B-16, S-06 (investigar).
  - `check_predict_input`: M-08 (reescribir `test_check_residuals_input_ValueError_when_residuals_for_some_level_is_None`), B-11, B-12, B-13.
  - S-02: solo reformular el aviso.
- **`ForecasterRnn`:** S-01 (en `check_predict_input`, comprobar `series_names_in_` también para Rnn), B-21 (`input_to_frame`) y S-05 (`.pop` sobre el `fit_kwargs` del usuario, en `deep_learning/_forecaster_rnn.py`). Requieren instalar keras en la sesión (`SKFORECAST_CLOUD_DL=1` o `uv pip install torch "keras>=3.0,<4.0" --torch-backend cpu`).
- **`deepcopy_forecaster`:** B-04 + O-07 (versión con `memo`).
- **Optimizaciones:** O-04 (multiserie `predict`), O-05 (`check_predict_input`) y O-06 (`multivariate_time_series_corr`).
- **Documentación:** lo que quede del §3.

Nota de versión: Fixed y Enhanced.

#### PR 6 · Dependencias y deprecaciones (independiente)

- **Decisión 2b:** `pandas>=2.2` en `pyproject.toml:56` y `:121`, `docs/quick-start/how-to-install.md:33` y `tools/ai/ai_context_header.md:39`. Después, regenerar con el skill `ai-context-sync` (`AGENTS.md` y `.github/copilot-instructions.md` no se editan a mano).
- **En el mismo fichero de cabecera:** corregir `matplotlib<3.11` → `<3.12`, para que coincida con `pyproject.toml`.
- **Decisión 4:** deprecar `cast_exog_dtypes` con `runtime_deprecated` y añadir el aviso en `docs/api/utils.md`.
- **`show_versions`:** añadir `scipy`, `statsmodels`, `matplotlib`, `skops` y `cloudpickle`.

Nota de versión: Changed (pandas) y Deprecated (`cast_exog_dtypes`).

### 5.3 Fuera del plan

- **Se deja como está:**
  - `transform_numpy`, `manage_warnings`, `select_n_jobs_fit_forecaster`, `check_interval` y las comprobaciones de NaN.
  - O-08 (cast de CatBoost), porque la ganancia es pequeña y solo en fit.
- **Propuestas para el futuro (no decididas):**
  - ~~Un job de CI con versiones mínimas: pandas 2.2, numpy 1.26, scikit-learn 1.4.~~ Pasa al PR 7 (§5.9).
  - Que `save_forecaster` devuelva la ruta final.

### 5.4 Orden de implementación, commits y PRs

**Orden recomendado** (estado a 2026-10-05: pasos 1 a 3 hechos salvo el PR 6; siguiente, PR 6 y PR 2):

| Paso | PR | Por qué en este orden |
|---|---|---|
| 1 | PR 1a · Atajos de predicción | El más pequeño (2 arreglos) y de mucho impacto: XGBoost con early stopping es una configuración habitual. Es lo único que el PR 2 necesita tener fusionado antes |
| 1 (en paralelo) | PR 6 · Dependencias y deprecaciones | Es pequeño y no comparte código con los demás. Retira M-05 y N-01 del plan |
| 2 | PR 1b · Direct con diferenciación | Solo toca `direct/` y no comparte código con ningún otro PR. Pequeño y autocontenido |
| 3 | PR 1c · Validación de exog en `predict` | El más grande de los tres y el único con un cambio de comportamiento (A-04), así que es el que más revisión necesita. Fija la nueva forma de `check_predict_input`, que los PRs 5a y 5b tocan después |
| 4 | PR 2a · CatBoost y dispositivo; después PR 2b · Categóricas nativas en estimadores envueltos | Tocan `_build_predict_function`, igual que el PR 1a, así que van después de él. El 2a va primero porque arregla los fallos más importantes sin decisiones de diseño (§5.7) |
| 5 | PR 3 · Persistencia | Es una zona independiente (save/load y `RollingFeatures`). Puede ir en paralelo con el PR 2 |
| 6 | PR 4 · Transformaciones e índices | Va antes del PR 5a porque toca helpers (`prepare_steps_direct`, `transform_series`) que el PR 5a usa en sus tests |
| 7 | PR 5a · Validación y robustez | Va después de los PRs 1c y 4, porque toca `check_predict_input` y helpers que esos PRs modifican |
| 8 | PR 5b · Rendimiento, `deepcopy` y docstrings | Va al final: las optimizaciones se miden mejor sobre el código ya corregido, y los docstrings que quedan son los de las funciones ya tocadas |

El PR 1 original se divide en 1a, 1b y 1c (2026-10-05) porque mezclaba tres zonas del código que no tocan las mismas funciones y que piden revisiones distintas:
- **1a:** internos de los estimadores (XGBoost, `LinearModel`, árboles) en `_build_predict_function`;
- **1b:** la matemática de la diferenciación en los forecasters Direct, sin tocar `utils.py`;
- **1c:** la validación de entradas en `predict`, con el único cambio de comportamiento (A-04).

Se pueden desarrollar en paralelo. Los PRs 1a y 1c editan `utils.py`, pero en funciones distintas, así que git fusiona sin conflicto; el único choque es `releases.md`.

El PR 5 original se divide en 5a y 5b porque tenía unos 14 commits, demasiados para revisarlos bien de una vez.

**Convenciones para todos los PRs:**
- **Ramas:** una rama `fix/…` o `chore/…` por PR, creada desde `0.26.x` y con PR contra `0.26.x`. Autorizado por el usuario el 2026-10-05, con PRs en borrador.
- **Un commit por arreglo**, con sus tests en el mismo commit, para que cada commit pase sus tests por sí solo.
- **Mensajes en inglés, en imperativo**, como el historial del repo ("Fix …", "Add …", "Validate …"). Sin trailers de IA.
- **Último commit de cada PR:** la nota de versión en `docs/releases/releases.md`, sección 0.26. Ninguna de las funciones afectadas es nueva en 0.26, así que los bugs van en Fixed. Los cambios de comportamiento y la deprecación van en Changed.
- **Antes de abrir cada PR:**
  - el skill `verify` (solo los tests afectados);
  - `python tools/ai/generate_ai_context_files.py --check`;
  - la descripción con `## Description` y `## Verification`, según el skill `open-pr`.
  El CI no ejecuta los tests unitarios en los PRs contra `0.26.x`, así que los tests locales son la única barrera.
- **Fusión secuencial:** todos los PRs añaden líneas a la misma sección de `releases.md`. Si alguno queda abierto en paralelo, se hace merge de `0.26.x` en su rama; el force push está bloqueado.

#### PR 1a · `fix/fast-predict-paths`
Título: *Fix fast predict paths for XGBoost and subclassed scikit-learn estimators*

| # | Commit | Contenido | Tests |
|---|---|---|---|
| 1 | Respect best_iteration, missing and gblinear in the XGBoost fast predict path | A-01: `_build_predict_function` (rama XGB) | `test_build_predict_function.py`: early stopping, `missing`, `gblinear`; equivalencia con `estimator.predict` |
| 2 | Use fast predict paths only for scikit-learn estimators, not user subclasses | M-02: guarda `__module__.startswith('sklearn.')` en las ramas lineal, RF y DT | Caso con una subclase de `Ridge` que recorta en 0 |
| 3 | Add release notes for fast predict path fixes | Fixed: A-01, M-02 | |

#### PR 1b · `fix/direct-differentiation-steps`
Título: *Fix Direct forecasters with differentiation when steps are not consecutive*

| # | Commit | Contenido | Tests |
|---|---|---|---|
| 1 | Fix Direct forecasters with differentiation when steps are not consecutive from 1 | A-02: `_forecaster_direct.py`, `_forecaster_direct_multivariate.py`. Con `differentiation`, predecir `1..max(steps)`, invertir y seleccionar los pasos pedidos; escalar el factor conformal con los pasos reales | `predict([3,4,5]) == predict(5)[2:]` en predict, intervalos (bootstrapping y conformal) y backtesting con `gap>0`, en los dos forecasters |
| 2 | Add release notes for Direct differentiation fix | Fixed: A-02 | |

#### PR 1c · `fix/predict-exog-validation`
Título: *Validate exog alignment and last_window shape at predict*

| # | Commit | Contenido | Tests |
|---|---|---|---|
| 1 | Keep wall-clock alignment in expand_index for tz-aware indexes across DST | M-09 (adelantado desde el PR 4): `expand_index` pasa a `pd.date_range(start=index[-1], periods=steps + 1, freq=freq)[1:]` | Casos `Europe/Madrid` en primavera y otoño; `predict` con exog cuando `last_window` acaba el día del cambio |
| 2 | Validate that exog follows the last_window frequency without gaps | A-03: nuevo helper `_check_exog_alignment` (vía rápida O(1) con `freq`, vía completa sin `freq`, aviso en la rama dict) llamado desde `check_predict_input`. El `ValueError` incluye la solución: `exog.reindex(expand_index(last_window.index, steps=n))` | Los listados en A-03 (`test_check_predict_input.py` y nivel de forecaster) |
| 3 | Raise an error for misaligned wide-format exog in ForecasterRecursiveMultiSeries | A-04 (decisión 1a): `lenient = isinstance(exog, dict)` | Pasar a exog dict los 5 tests indicados en A-04; tests de forecaster con exog ancha que empieza tarde o es corta |
| 4 | Reject a multi-column last_window in single-series forecasters | M-03: `check_predict_input` | Nuevo test con `last_window` de 2 columnas |
| 5 | Add release notes for predict input validation fixes | Fixed: A-03, M-03, M-09. Changed: A-04 | |

#### PR 6 · `chore/pandas-2.2-and-deprecations` (en paralelo con el PR 1a)
Título: *Require pandas>=2.2 and deprecate cast_exog_dtypes*

| # | Commit | Contenido |
|---|---|---|
| 1 | Require pandas>=2.2 | `pyproject.toml:56` y `:121`, `docs/quick-start/how-to-install.md`, `tools/ai/ai_context_header.md` (también `matplotlib<3.12`). Ficheros de contexto de IA regenerados con `ai-context-sync` en el mismo commit |
| 2 | Deprecate cast_exog_dtypes | B-10 (decisión 4): `@runtime_deprecated(...)` con la alternativa `exog.astype(forecaster.exog_dtypes_in_)`; aviso en `docs/api/utils.md`; test del `FutureWarning` |
| 3 | Report scipy, statsmodels, matplotlib, skops and cloudpickle in show_versions | `show_versions` y su test |
| 4 | Add release notes for dependency and deprecation changes | Changed: pandas, deprecación y `show_versions` |

#### PR 2 · `fix/estimator-categorical-gpu`

**Sustituido por los PRs 2a y 2b (§5.7).** Se conserva como referencia.

Título: *Fix native categorical features, CatBoost, NaN support and device handling for estimators*

| # | Commit | Contenido | Tests |
|---|---|---|---|
| 1 | Add a shared helper to unwrap Pipeline and CalibratedClassifierCV estimators | M-17: `_unwrap_estimator()`, usado por `configure_*`, `cast_catboost_*`, `_get/_restore_*` y `estimator_has_native_nan_support`. Comprobar S-03 (`TransformedTargetRegressor`) | Nuevo, con `CalibratedClassifierCV(LGBMClassifier)` |
| 2 | Pass categorical feature arguments to the last step of a Pipeline | A-05: prefijo `paso__`, búsqueda de la clave en `cast_catboost_*`, CatBoost dentro de Pipeline en `_build_predict_function` | Corregir `test_pipeline_extracts_last_step_lgbm`; fit y predict de punta a punta con Pipeline de LGBM y de CatBoost (Recursive, Direct y MultiSeries) |
| 3 | Cast CatBoost categorical lags at predict in ForecasterRecursiveClassifier | A-06: `_forecaster_recursive_classifier.py` | predict, `predict_proba` y backtesting con `CatBoostClassifier` |
| 4 | Encode multiseries levels with fixed categories so CatBoost codes do not shift | M-01: `_forecaster_recursive_multiseries.py:1316`, más la defensa en `cast_catboost_categorical_columns_dataframe` | Búsqueda one-step-ahead con un nivel ausente en test |
| 5 | Warn when resetting user categorical settings and stop warning on every refit | M-12: `configure_estimator_categorical_features`; nota del user guide `categorical-features.ipynb` (celda 35) | Actualizar `test_histgbr_reset_*` y `test_xgboost_reset_*`; nuevo test sin aviso en el refit |
| ~~6~~ | **Ya no hace falta** (scikit-learn>=1.6 en el PR 6, que también corrige el comentario). ~~Declare ExtraTrees NaN support only for scikit-learn >= 1.6~~ | M-13: `_SKLEARN_NAN_TOLERANT_ESTIMATORS` y su comentario | `test_estimator_has_native_nan_support_true`, dependiente de la versión |
| 7 | Add ExtraTreesRegressor to the per-tree fast predict path | O-03 (requiere el commit 6) | Equivalencia con `estimator.predict` |
| 8 | Restore the original estimator device verbatim after predicting | A-10: `set_cpu_gpu_device` (más su docstring) y `try/finally` en los llamadores de `_forecaster_recursive.py`, multiseries y classifier | Round trips con `'cuda:0'` (XGB) y `'cuda'` (LGBM), sin GPU |
| 9 | Add release notes for estimator fixes | Fixed: A-05, A-06, A-10, M-01, M-12, M-13, M-17. Changed: O-03 (rendimiento) | |

#### PR 3 · `fix/save-load-forecaster`
Título: *Fix skops persistence and file naming in save_forecaster*

**Sustituido por los PRs 3a y 3b del §5.8** (revisión del 2026-10-06). La tabla de abajo es el plan original.

| # | Commit | Contenido | Tests |
|---|---|---|---|
| 1 | Stop storing pandas Rolling objects in RollingFeatures after transform_batch | A-09 + O-02: `preprocessing/_preprocessing.py` (dos clases) | `rolling_obj is None` tras `fit`; round trip skops con `window_features` para los 5 forecasters |
| 2 | Serialize a shallow copy of the forecaster with skops instead of mutating it | B-05: `save_forecaster` | El test existente de no mutación |
| 3 | Store DatetimeIndex as epochs with unit, time zone and frequency for skops | A-08, M-15, B-06, O-01: `_decompose_index` / `_compose_index`, con fallback al payload antiguo | Actualizar `test_decompose_index_output[datetime]`; casos con cambio de hora, `500ms`, `CustomBusinessDay` y payload de 0.25 |
| 4 | Serialize categorical exog dtypes for skops | M-14 (requiere el commit 2) | Una serie y multiserie con exog categórica |
| 5 | Keep dotted file names and only replace known backend extensions | M-16 (decisión 3a): `save_forecaster` y su docstring; `save-load-forecaster.ipynb` (celda 13) | `'model_v1.2'` y `'f.pkl'` |
| 6 | Detect built-in window feature classes by module when saving | B-07 | Sin aviso con `RollingFeaturesClassification` |
| 7 | Write exported weight functions as UTF-8 | B-08, más S-04 si se confirma (callables sin `__name__` o sin código fuente) | Caso con un `partial` en `__main__` si S-04 se confirma |
| 8 | Add release notes for persistence fixes | Fixed: A-08, A-09, M-14, M-15, B-05 a B-08. Changed: M-16 | |

#### PR 4 · `fix/transforms-and-index-handling`
Título: *Fix transformers, time zones and step handling in index utilities*

| # | Commit | Contenido |
|---|---|---|
| 1 | Fall back to default feature names when get_feature_names_out raises | A-07: `transform_dataframe`, `transform_series`; test con `ForecasterRecursive` y `Pipeline(FunctionTransformer, StandardScaler)` |
| 2 | Return a Series from transform_series for single-row input | M-11; test unitario y de `ForecasterStats` con `last_window` de 1 observación |
| 3 | Do not mutate transformers in transform_series when the series name differs | B-17; test con `Pipeline` |
| 4 | Support tz-aware indexes and indexes without freq in date_to_index_position | M-10 (más los mensajes de error del §3); `predict(steps=str)` y `TimeSeriesFold` con zona horaria y sin `freq` |
| 5 | Accept numpy integers and reject empty steps in prepare_steps_direct | M-04; `np.int64`, tupla, `0` y `[]` |
| 6 | Validate steps against the exog length in exog_to_direct | B-19 |
| 7 | Accept pandas Index and arrays as levels in multiseries predict | B-18 |
| 8 | Add release notes for transform and index fixes | Fixed (M-09 ya va en el PR 1c) |

#### PR 5a · `fix/input-validation`
Título: *Fix input validation edge cases in skforecast.utils*

| # | Commit | Contenido |
|---|---|---|
| 1 | Accept numpy integers and reject booleans in lags and window sizes | B-03, B-20 |
| 2 | Do not fail on weight_func callables without retrievable source | M-06 (con S-04). Se recomienda moverlo al PR 3c (§5.8), pendiente de decisión |
| 3 | Do not mutate user fit_kwargs in check_select_fit_kwargs | B-02, más el nuevo texto del aviso de S-02 |
| 4 | Accept nullable and pyarrow numeric dtypes in check_exog_dtypes | B-09 |
| 5 | Handle nullable dtypes when trimming multiseries NaN | M-07 |
| 6 | Report mismatched frequencies and time zones clearly in check_preprocess_series | B-01, B-16 |
| 7 | Fix unnamed, duplicated and unordered exog names in multiseries preprocessing | B-14, B-15 (más S-06 si se confirma) |
| 8 | Check residuals only for the predicted levels | M-08; reescribir `test_check_residuals_input_ValueError_when_residuals_for_some_level_is_None` |
| 9 | Tighten check_predict_input edge cases | B-11 (ampliado, ver §5.5), B-13, S-01. **B-12 ya está hecho** (PR 1c, commit `29f707e` del usuario) |
| 10 | Fix exog_val handling in ForecasterRnn and stop mutating fit_kwargs | B-21, S-05 (requiere instalar keras en la sesión) |
| 11 | Add release notes for input validation fixes | Fixed |

#### PR 5b · `perf/utils-hot-paths`
Título: *Speed up multiseries predict and check_predict_input, and fix utils docstrings*

| # | Commit | Contenido |
|---|---|---|
| 1 | Make deepcopy_forecaster exception-safe using the deepcopy memo | B-04 + O-07; los 35 tests existentes más el caso de excepción |
| 2 | Speed up check_predict_input | O-05 (números de benchmark en la descripción del PR). **Volver a medir:** el PR 1c cambió la función (ver §5.5); `expand_index` ya no se puede quitar |
| 3 | Speed up last window preparation in multiseries predict | O-04 |
| 4 | Use corrwith in multivariate_time_series_corr | O-06, más `lags` como `np.integer` |
| 5 | Add ExtraTrees to the per-tree fast predict path | O-03, movido desde el PR 2: `ExtraTreesRegressor` en la rama de `RandomForestRegressor` y `ExtraTreeRegressor` en la de `DecisionTreeRegressor`. Salida idéntica, también con NaN; predicción fila a fila x17.5 (§5.7). Test de equivalencia con `estimator.predict` |
| 6 | Fix docstrings in skforecast.utils | Lo que quede del §3 |
| 7 | Add release notes for performance changes | Changed (rendimiento, con las ganancias medidas) y Fixed (B-04) |

### 5.5 Hallazgos nuevos y lecciones de la implementación (PRs 1a a 1c, 2026-10-05)

**Cambios respecto al plan:**
- **A-04:** la decisión 1a (exog ancha estricta) se sustituyó por la opción **b-light**, aprobada por el usuario. En `ForecasterRecursiveMultiSeries.predict`, la exog ancha se alinea por fecha y columna (un `reindex`, con vía rápida si ya está alineada), igual que en `fit`, en el backtesting y con la exog dict. Motivos medidos:
  - `fit` y el backtesting ya eran permisivos;
  - con la exog completa (entrenamiento + futuro), la ancha daba predicciones erróneas sin error;
  - 0 tests rotos y coste nulo si la exog ya está alineada.
  - Cambio de comportamiento documentado en la nota de versión: una exog ancha con índice que no coincide pero valores en orden (p. ej. `RangeIndex` reiniciado) antes se usaba por posición y ahora sus valores son NaN.
- **A-03:** el helper es `_check_exog_alignment(exog_name, exog_index, expected_index, align_by_index)`. `check_predict_input` calcula `expected_index = expand_index(last_window_index, last_step)` una vez. Con `align_by_index=True` (multiserie) comprueba por fecha (`isin`) y lanza `ValueError` si hay fechas duplicadas; con `False` compara posición a posición.
- **B-12:** hecho en el PR 1c (commit del usuario `29f707e`).
- **Nota de versión:** la entrada de M-09 se fusionó con la de #1343 en una sola, con dos sub-viñetas.

**Hallazgos nuevos, no arreglados (candidatos para PRs futuros):**
- **N-02 (bajo, junto con B-11 en el PR 5a):** una Series exog con nombre válido sin el resto de columnas. En los forecasters estrictos da `KeyError: "['e2'] not in index"` sin contexto. En multiserie la columna queda en NaN: con exog ancha solo sale un aviso genérico de NaN y con dict ninguno. El arreglo de B-11 (convertir la Series a frame en el check) haría saltar "Missing columns" (error) o `MissingExogWarning` (multiserie).
- **N-03 (bajo):** en multiserie, una Series exog cuyo nombre no está en `exog_names_in_`, con una variable categórica, da `TypeError: ufunc 'isnan' not supported for the input types` (ancha y dict). Ya pasaba en `0.26.x` en la ruta dict.
- **N-04 (bajo, ruido):** con exog ancha en multiserie, los NaN dan dos avisos (`check_predict_input` y `check_exog(allow_nan=False)` en `_create_predict_inputs`); con dict, uno. Ya pasaba en `0.26.x` con NaN del usuario. Arreglo posible: `allow_nan=True` en esa llamada.
- **N-05 (rendimiento, para O-05 en el PR 5b):** con exog dict con historia completa (500 series × 2000 filas), `check_predict_input` pasa de 21 a 42 ms por el `isin` (~42 µs por serie). El `predict` completo tarda lo mismo porque desaparecen 500 avisos falsos. Optimización posible: `searchsorted` si el índice es monótono.
- **N-06 (teórico):** la vía rápida del helper confía en la igualdad de `freq`; una exog con `freq='24h'` frente a una serie `'D'` con cambio de hora pasaría sin comprobar.
- **N-07 (documentación, contexto IA):** `skills/forecasting-multiple-series/SKILL.md` (líneas 33 y 157) dice que la exog debe tener el mismo formato que `series` (ambos anchos o ambos dict). El user guide `multi-series-with-different-length-and-different_exog.ipynb` (celda 1) dice que se pueden combinar. Corregir con `ai-context-sync`.
- **N-08 (lint preexistente):** `test_recursive_predict_bootstrapping.py` tiene imports sin usar (F401 ×3, F811); `cast_exog_dtypes` tiene E721 (se depreca en el PR 6).
- **Tests de `ForecasterRnn`:** no se ejecutaron para el PR 1c (el cloud no tiene torch/keras). Ejecutarlos en el PR 5a, que ya necesita keras, o pedir al usuario que los corra en local sobre `0.26.x`.

**Lecciones para los PRs siguientes:**
- **Pruebas de propiedades con configuraciones raras.** En el PR 1b el usuario encontró un bug mío en `ForecasterDirectMultiVariate` con la serie objetivo sin lags, una configuración que mi prueba no cubría. Incluir siempre: serie sin lags, transformadores, diferenciación por serie, categóricas, índices con zona horaria.
- **Comparar con la base** el caso de usuario que cambia de comportamiento y documentarlo en la nota de versión (en el 1c, el `RangeIndex` reiniciado).
- **Cada commit pasa sus tests** (worktree por commit) y **dos líneas en blanco entre tests** (comparar con la base).
- **Cloud:** en `foundation` hay 10 fallos por falta de torch, iguales en la base y en la rama; `deep_learning` no se puede ejecutar sin instalar torch/keras.


### 5.6 PR 6: revisión del plan e implementación (2026-10-05)

**Revisión del plan antes de implementar** (entornos con las versiones mínimas: numpy 1.26.4, scikit-learn 1.4.2; tests de `utils`, `preprocessing`, `recursive`, `direct`, `stats` y `model_selection`, sin `slow`):

| | pasan | fallan | errores de recogida |
|---|---|---|---|
| pandas 2.1.4 | 3839 | 42 | 44 |
| pandas 2.2.0 | 4855 | 10 | 0 |
| pandas 2.2.0 + scikit-learn 1.6.0 (rama del PR 6, con `datasets`) | 4886 | 1 | 0 |

- **Corrección de N-01:** los alias `'YE'`/`'QE'`/`'ME'` de `_calendar.py:694` no fallan con pandas 2.1 (son cadenas que se comparan; la lista incluye los alias antiguos). Los errores con `'YE'`/`'ME'` venían de los datos de los tests. Lo que falla en la librería es `include_groups` (`_preprocessing.py:476`, `reshape_series_wide_to_long`) y, **nuevo**, `pd.option_context("future.no_silent_downcasting", True)` (`_calendar.py:849`, `calculate_distance_from_holiday` con NaN en la columna de festivos).
- **N-09 (nuevo, resuelto con scikit-learn>=1.6):** con scikit-learn 1.4, la `X_train` de `create_train_X_y` es de solo lectura y `LinearRegression().fit(X_train, y_train)` da `ValueError: cannot set WRITEABLE flag to True of this array` (7 de los 10 fallos). `forecaster.fit/predict` funcionan. Con 1.5.2 ya no falla.
- **M-13** se resuelve con el mínimo 1.6 (ExtraTree* acepta NaN desde 1.6.0, comprobado con `pr6_nan_trees.py`; `b4/r13*.py` reproduce el error de backtesting con 1.5.2).
- **Fallo que queda con 1.6:** `test_QuantileBinner_is_equivalent_to_KBinsDiscretizer` usa `quantile_method` (scikit-learn 1.7). Solo afecta al test; se deja.
- Otros hallazgos de la revisión, incluidos en el PR: `how-to-install.md` decía `numpy>=1.22`; rama muerta para pandas < 2.2 en `datasets.py:817`.
- `cast_exog_dtypes`: sin usos en ningún fichero del repo ni en ninguna rama publicada (0.8.x a 0.25.x), ni en `main`. El usuario decidió eliminarla.

**Decisiones del usuario (2026-10-05):** eliminar `cast_exog_dtypes`; para las versiones, "es suficiente con aceptar que hay que subir el mínimo e informar de por qué"; subir también `scikit-learn>=1.6`.

**Implementación** (rama `chore/minimum-versions-and-cleanup`; título propuesto: *Require pandas>=2.2 and scikit-learn>=1.6, and remove cast_exog_dtypes*):

| # | Commit | Contenido |
|---|---|---|
| 1 | `08e3db0` Require pandas>=2.2 and scikit-learn>=1.6 | `pyproject.toml` (3 líneas), `how-to-install.md` (también numpy 1.26), `ai_context_header.md` (también `matplotlib<3.12`), ficheros de contexto de IA regenerados, ramas muertas para pandas < 2.2 en `datasets.py` y en 8 ficheros de tests (añadidas en la revisión final), comentario de `_SKLEARN_NAN_TOLERANT_ESTIMATORS` |
| 2 | `3b7fd82` Remove cast_exog_dtypes | Función y su línea de `docs/api/utils.md`. Desaparece el E721 de N-08 |
| 3 | `8e52a08` Report scipy and optional dependencies in show_versions | Añade scipy, statsmodels, matplotlib, torch, lightgbm, xgboost, catboost, skops y cloudpickle; nuevo test |
| 4 | `fea55d7` Add release notes for minimum versions, cast_exog_dtypes and show_versions | Tres entradas en Changed, sin highlight |

Decisiones del usuario tras la revisión final: sin highlight de API Change para la eliminación; la lista de `show_versions` vale; `show_versions` se añade a `docs/api/utils.md` (commit 3) y se enlaza en la nota de versión (commit 4).


### 5.7 PR 2: revisión del plan antes de implementar (2026-10-06)

Reproducido todo sobre `0.26.x` @ `742d309` (pandas 2.3.3, scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10). Scripts en `review/pr2/` (`r_*.py`).

**Estado de cada hallazgo:**

| Hallazgo | ¿Se reproduce? | Medida | ¿El arreglo del plan es el adecuado? |
|---|---|---|---|
| A-05 Pipeline + LightGBM/CatBoost | Sí: `fit` falla en Recursive y Direct; XGB y HGB funcionan (`r_a05.py`) | Con el prefijo `paso__`, `Pipeline(StandardScaler(), LGBMRegressor())` con 10 categorías entrena sin error pero con MAE 5.60 frente a 0.12 (LightGBM trunca los códigos escalados y junta categorías); con `MinMaxScaler`, 5.99 y sin aviso. CatBoost y XGB con escalador fallan (`r_a05_prefix.py`). Con `ColumnTransformer` las columnas se reordenan y los índices apuntarían a otra columna | **No.** Convierte un error claro en resultados erróneos sin aviso en el caso más habitual (escalador delante). Alternativa: no configurar las categóricas nativas si el estimador es un `Pipeline` (pasan codificadas como ordinales, como para cualquier otro estimador). Codificadas como ordinales: MAE 0.11-0.12 con cualquier paso y librería |
| A-06 CatBoostClassifier | Sí: `fit` bien; `predict`, `predict_proba` y backtesting fallan (`r_a06.py`) | | Sí: cast en el bucle de `_recursive_predict` del clasificador, con un helper que lea los índices categóricos del CatBoost entrenado (también lo usa `_build_predict_function`) |
| M-01 `.cat.codes` | Sí: búsqueda one-step-ahead con CatBoost, MAE medio 1.797 frente a 0.936 del backtesting equivalente (`r_m01.py`) | Con `categories=range(n_levels)` en la causa raíz: 0.877; LightGBM sin cambios (0.888) | Sí, solo la causa raíz. La defensa en `cast_catboost_categorical_columns_dataframe` sobra: es el único `pd.Categorical` que llega ahí |
| M-12 reset y avisos | Sí: HGB `categorical_features=[3]` y XGB `feature_types` del usuario se pierden sin aviso con exog numérica; aviso falso en cada refit (`r_m12*.py`) | Avisos falsos con `-W always`: backtesting con refit 4; búsqueda one-step-ahead con `lags_grid=[3, 5, 7]` 2 | **No del todo.** La regla del plan (avisar si el valor previo no es el de por defecto y cambia) quita los del refit, pero no los de la búsqueda con `lags_grid` (el índice cambia con los lags). Un marcador en el estimador tampoco sirve: `forecaster.set_params` clona el estimador y lo pierde. Alternativa: avisar una vez en `__init__` si el usuario trae su propia configuración y `categorical_features` no es `None`, y no avisar en `fit` |
| M-17 CalibratedClassifierCV | Sí, en las 4 familias: el clasificador dice `use_native_categoricals=True`, pero los lags llegan como numéricos (`r_m17_s03.py`). Afecta al ejemplo de la guía de clasificación (celda 45, `CalibratedClassifierCV(HGB)`) | | Sí |
| S-03 TransformedTargetRegressor | **Confirmado** en las 4 familias: la exog categórica llega como numérica | `TTR` y `CalibratedClassifierCV` reenvían al estimador interno los argumentos de `fit`, la matriz `object` y los NaN (`r_wrappers.py`) | Sí, junto con M-17. Requisito: el cast de CatBoost en `predict` tiene que desenvolver también `TTR` y `CalibratedClassifierCV`; si no, `TTR(CatBoost)` y `CalibratedClassifierCV(CatBoost)`, que hoy funcionan, fallarían en `predict` |
| O-03 ExtraTrees | Salida idéntica (también con NaN); predicción fila a fila x17.5 (`r_o03.py`) | | Sí, pero ya no depende de nada y es rendimiento: mejor en el PR 5b |
| A-10 dispositivo | Sí: XGB `'cuda:0'` falla en `predict`; XGB `'gpu'` vuelve como `'cuda'`; LGBM `'cuda'` vuelve como `'gpu'`; `'GPU'`/`'CPU'` dan `KeyError` (`r_a10.py`) | Nuevo: con CatBoost sin entrenar, la rama cambia `task_type='GPU'` a `'CPU'` y no lo restaura (lee `None`); con un modelo entrenado es código muerto | Sí, salvo el `try/finally`: obliga a reindentar los 5 bucles de predicción (unas 370 líneas) para un caso que solo pasa si `predict` se interrumpe a mitad; la consecuencia es que el estimador queda en CPU |

Otros datos:
- XGB sin `device` (`None`) y LightGBM sin `device` en `get_params()`: hoy cada `predict` deja `device='cpu'` fijado. Con el arreglo, un dispositivo sin fijar no se toca.
- CatBoost envuelto y sin categóricas predice bien (`r_n10.py`).
- Las categóricas nativas se publicaron en 0.22.0: cualquier cambio de comportamiento va en Changed.
- `estimator_has_native_nan_support` se deja como está (solo desenvuelve `Pipeline`): añadir `TTR` sería una mejora, no un arreglo.

**Decisión del usuario (2026-10-06): dividir el PR 2 en dos.** Lo complicado (A-05, M-17/S-03 y M-12) gira en torno a una sola pregunta de diseño: cuándo configurar las categóricas nativas de un estimador envuelto. El resto son arreglos locales sin decisiones de diseño.

**Importancia** (frecuencia del caso, gravedad y coste del arreglo):

| Orden | Hallazgo | Por qué | Coste | PR |
|---|---|---|---|---|
| 1 | A-06 | `predict` falla siempre con `CatBoostClassifier` y la configuración por defecto (`features_encoding='auto'`) | Bajo | 2a |
| 2 | M-01 | Métrica errónea sin aviso: la búsqueda puede elegir mal. Caso estrecho: `encoding='ordinal_category'` (el defecto es `'ordinal'`), one-step-ahead y una serie sin datos en test | Muy bajo (una línea) | 2a |
| 3 | A-10 | XGB `'cuda:0'` falla en `predict`; LGBM `'cuda'` pasa a `'gpu'` (otro backend) sin aviso | Bajo (una función) | 2a |
| 4 | A-05 | Error claro; se evita quitando el `Pipeline`. El arreglo bueno cambia el comportamiento de los `Pipeline` con XGB/HGB | Medio; decisión del usuario | 2b |
| 5 | M-17, S-03 | Sin error ni resultados erróneos: las categóricas van como numéricas (MAE 0.11 frente a 0.12 en las mediciones) | Medio-alto: 6 helpers, cast de CatBoost en `predict`, volver a ejecutar la guía de clasificación | 2b |
| 6 | M-12 | Reset de la configuración del usuario (raro) y avisos de más | Medio: los 5 `__init__` | 2b |

#### PR 2a · `fix/catboost-predict-and-device`
Título: *Fix CatBoost categorical features at predict and restore the estimator device*

| # | Commit | Hallazgo | Contenido | Tests |
|---|---|---|---|---|
| 1 | Cast CatBoost categorical features at predict in ForecasterRecursiveClassifier | A-06 | Helper privado en `utils.py` que lee los índices categóricos del CatBoost entrenado (regresor o clasificador). Lo usan `_build_predict_function` (en lugar de comprobar el nombre `CatBoostRegressor` para el cast) y `_recursive_predict` del clasificador (`predict` y `predict_proba`). La rama de `CatBoostRegressor` sin categóricas (restaurar `writeable`) no cambia | `predict`, `predict_proba` y backtesting con `CatBoostClassifier` (`features_encoding='auto'`); tests del helper |
| 2 | Encode multiseries levels with all categories so CatBoost codes do not shift | M-01 | `pd.Categorical(encoded_values, categories=range(len(self.encoding_mapping_)))` en `_create_train_X_y` de `ForecasterRecursiveMultiSeries`. Sin la defensa en `cast_catboost_categorical_columns_dataframe` | Búsqueda one-step-ahead con CatBoost, `encoding='ordinal_category'` y un nivel sin datos en test; códigos de `_level_skforecast` iguales a sus valores |
| 3 | Restore the original estimator device verbatim after predicting | A-10 | `set_cpu_gpu_device`: solo XGBoost y LightGBM (se quita la rama de CatBoost); lee el valor con `get_params()`; pasa el valor tal cual (sin la tabla que cambiaba `'gpu'`/`'cuda'`); un dispositivo sin fijar (`None`) no se toca, porque ya es CPU; valida el prefijo `cpu`/`gpu`/`cuda`; docstring NumPy completo. **Sin `try/finally`** en los llamadores | Restaurar `'cuda:0'` (XGB), `'gpu'` (XGB) y `'cuda'` (LGBM) tras `predict`, sin GPU; dispositivo sin fijar intacto; actualizar `test_set_cpu_gpu_device.py` |
| 4 | Add release notes for CatBoost and device fixes | | Tres entradas en Fixed | |

#### PR 2b · `fix/categorical-wrapped-estimators` (después del 2a)
Título: *Fix native categorical features with Pipeline, CalibratedClassifierCV and TransformedTargetRegressor*

Decisiones pendientes del usuario antes de implementar:
- **A-05:** recomendado, no configurar las categóricas nativas si el estimador es un `Pipeline` (codificación ordinal; los `Pipeline` con XGB/HGB cambian, va en Changed). Alternativa: el prefijo del plan original (descartado por los resultados erróneos con escaladores).
- **M-12:** recomendado, avisar una vez en `__init__` y no avisar en `fit`. Versión reducida: la regla del plan original (quita el aviso del refit, deja los de la búsqueda con `lags_grid`) más la corrección de la nota de la guía.

| # | Commit | Hallazgo |
|---|---|---|
| 1 | Configure native categorical features inside TransformedTargetRegressor and CalibratedClassifierCV | M-17, S-03. `_unwrap_estimator(estimator, fitted=False)`, usado por `configure_*`, `cast_catboost_*`, `_get/_restore_*`, el helper de CatBoost del PR 2a y `_check_categorical_support`. Volver a ejecutar `autoregressive-classification-forecasting.ipynb` |
| 2 | Pass categorical features ordinal-encoded to Pipeline estimators | A-05 (según la decisión) |
| 3 | Warn at initialization when the forecaster manages the estimator categorical parameters | M-12 (según la decisión); corregir la celda 35 de `categorical-features.ipynb` |
| 4 | Add release notes | Fixed (M-17, S-03, M-12) y Changed (A-05) |

**O-03** pasa al PR 5b (commit 5).

#### PR 2a: implementación (2026-10-06): skforecast/skforecast#1349, fusionado (`59822eb`)

Subido el 2026-10-06 09:13 UTC con el OK del usuario: rama `fix/catboost-predict-and-device`, head `3d43b12`, título *Fix CatBoost categorical features at predict and restore the estimator device*. Pie quitado, sesión suscrita a la actividad del PR y check-in de seguridad `trig_01YY7NZYGN5xR8fm96QHBSVY` (10:04 UTC).

Rama `fix/catboost-predict-and-device` desde `origin/0.26.x` (`742d309`):

| # | Commit | Contenido |
|---|---|---|
| 1 | `e12bca2` Cast CatBoost categorical features at predict in ForecasterRecursiveClassifier | Helper `_get_catboost_cat_feature_indices` (exportado en `utils/__init__.py`), usado en `_build_predict_function` (solo para leer los índices) y en `_recursive_predict` del clasificador. Tests: helper (nuevo fichero), `predict` y `predict_proba` con `CatBoostClassifier` (lags y exog categórica), backtesting sin refit |
| 2 | `225050f` Encode multiseries levels with all categories so CatBoost codes do not shift | `categories=range(len(self.encoding_mapping_))`. Tests: códigos en train y test con una serie sin datos en test; parametrización de CatBoost + `ordinal_category` en el test de equivalencia one-step-ahead frente a backtesting (fixture `series_dict_nans`, `id_1002` acaba antes del test) |
| 3 | `febd913` Restore the original estimator device verbatim after predicting | `set_cpu_gpu_device` reescrita. Tests de la función reescritos (16) e integración en `predict` de Recursive y MultiSeries |
| 4 | `3d43b12` Add release notes for CatBoost and device fixes | Tres entradas en Fixed, tras las del PR 1a; sin highlight |

Desviaciones del plan:
- A-10: **sin validación del valor de `device`** (el plan decía validar el prefijo `cpu`/`gpu`/`cuda`). Al restaurar se pasa el valor que tenía el usuario, y XGBoost acepta otros (por ejemplo `'sycl'`): validar haría fallar `predict` al restaurar. Se quita el test del `ValueError` con `'tpu'`.

Verificación:
- Comprobación de propiedades de A-06: en 6 configuraciones (lags contiguos y no contiguos, window features con `mode`, exog categórica con NaN en el futuro, `categorical_features=None`, `features_encoding='categorical'`), `predict` y `predict_proba` coinciden con CatBoost sobre `create_predict_X` convertido como en `fit` (`pr2/p_a06.py`).
- M-01: la búsqueda one-step-ahead coincide con el backtesting (antes, `id_1003` y `id_1004` diferían).
- Los tests nuevos fallan sin el arreglo (5 de A-06, 2 de M-01, 17 de A-10) y pasan con él.
- `verify`: ruff limpio salvo un F401 en `utils/__init__.py` (mismo patrón que las 4 reexportaciones privadas que ya había); tests `utils` 665, `recursive` 1360 (1 skipped), `direct` 899, `model_selection` 786 (sin `slow`), `feature_selection` 69; contexto IA al día; cada commit pasa sus tests en un worktree aparte.

Revisión final (2026-10-06), con fixups en los commits 1 y 3 (SHA de arriba ya actualizados):
- `set_cpu_gpu_device` leía el dispositivo con `get_params()` (unos 90 µs, dos llamadas por `predict`, un 17-33 % de un `predict(1)`). `getattr(estimator, 'device', None)` da el mismo valor (XGBoost siempre tiene el atributo; LightGBM cuando se fija) y vuelve a ser el de antes. Resultado frente a la base: `predict(1)` con XGBoost 0.62 → 0.53 ms (sin dispositivo fijado ya no se llama a `set_params`).
- Quitado el `try/except` de `set_params`: con `xgboost-cpu` (sin CUDA), `set_params(device='cuda')` sobre un modelo entrenado no falla; solo hacía falta para CatBoost.
- El test de dispositivo de Recursive usaba el fixture `y_categorical`; ahora `pd.Series(np.arange(50))`, como los tests simples del fichero.
- Comentario del bucle del clasificador: dice que los NaN pasan a -1, como en `fit`.
- `encoding_mapping_` no se vacía en `_train_test_split_one_step_ahead` (preexistente): con un forecaster ya entrenado con más series, `range(len(...))` incluye categorías sin usar. Inocuo: la búsqueda trabaja sobre una copia y da el mismo MAE.
- Mensaje del commit 3 reescrito (decía que el dispositivo se lee con `get_params`).



### 5.8 PR 3: revisión del plan antes de implementar (2026-10-06)

Reproducido todo sobre `0.26.x` @ `59822eb` (pandas 2.3.3, skops 0.16.0, pyarrow 25.0.1). Scripts en `review/next/`:
- `r_all.py`, `r_rest.py`, `r_o01.py`, `r_s04.py`, `r_pa.py`, `o02.py`: reproducciones;
- `skops_types.py`, `skops_tz.py`, `tz_types.py`, `scan_attrs.py`: qué tipos soporta skops y qué guardan los forecasters;
- `matrix.py`: matriz de 30 casos de guardado y carga con skops;
- `old_gen.py` y `old_load.py`: compatibilidad con ficheros del formato actual (el de 0.23 a 0.25);
- `apply_proto.py` y `pr3_proto.diff`: prototipo de todos los arreglos (se probó en un worktree temporal, ya borrado; sin commits).

**Estado de cada hallazgo:**

| Hallazgo | ¿Se reproduce? | Medida | ¿El arreglo del plan es el adecuado? |
|---|---|---|---|
| A-08 | Sí. Con cambio de hora (Madrid horario de marzo, Nueva York diario) la carga falla. Sin cambio de hora, `Europe/Madrid` vuelve como `UTC+01:00` y, si se predice más allá del cambio de hora, las etiquetas salen con una hora de desfase | | Sí, con tres ajustes: (1) conservar `ZoneInfo`: con `str(tz)` vuelve como pytz, las etiquetas son iguales, pero al combinar las predicciones con los datos del usuario el índice pasa a UTC; (2) error claro al guardar si la zona no se puede reconstruir desde `str(tz)` (`dateutil`, `pytz.FixedOffset`), en lugar de un fichero que no carga; (3) el fallback del formato antiguo: con desfases mezclados `pd.to_datetime` no lanza `ValueError`, devuelve un índice `object`; hay que comprobar el tipo del resultado |
| M-15 | Sí: `500ms` y `100us` | | Sí, lo cubre A-08. Además, `format='ISO8601'` en el fallback recupera los ficheros antiguos de 500 ms, que hoy no cargan |
| B-06 | Sí: carga, pero `predict` falla (`Expected frequency of type <CustomBusinessDay>`) | | Sí: guardar `index.freq` (el objeto). skops ya serializa todos los offsets salvo `pd.DateOffset` (N-11) |
| O-01 | Sí. `ForecasterEquivalentDate` guarda la serie entera en `last_window_` (a propósito, ver abajo). Con 200k filas: guardar 4.4 s, cargar 3.8 s, 58 MB | Con epochs: 0.01 s, 0.01 s, 3.3 MB | Sí |
| A-09 + O-02 | Sí, en los 5 forecasters con `window_features` (falla al guardar). Con `RangeIndex` se guarda, pero falla al cargar | joblib con 500k filas: 16.0 MB → 0.006 MB; `deepcopy` 26.9 → 0.29 ms | Sí. Mejor **quitar la clave `'rolling_obj'`** que dejarla siempre a `None`: sería estado muerto, y solo la comprueban 3 aserciones de tests de `__init__` |
| M-14 | Sí: falla al guardar con exog categórica | | Sí, ampliado a N-10 |
| B-05 | Sí: 228 903 errores de `predict` concurrente en 30 guardados | Con la copia: 0 | Sí. Además simplifica `save_forecaster` (sin `try/finally`) y es lo que necesitan M-14, N-10 y N-11, que descomponen más atributos |
| M-16 | Sí: `model_v1.1` y `model_v1.2` escriben un solo fichero y se carga el segundo modelo | | Sí (decisión 3a). El prototipo da exactamente la tabla de M-16 |
| B-07 | Sí | | **No como estaba.** Decidir por el módulo (`skforecast.`) rompe `test_save_forecaster_warning_when_user_defined_window_features`: las clases de usuario de los tests viven en módulos `skforecast.…` y dejan de avisar. Mejor añadir `'RollingFeaturesClassification'` al conjunto de nombres (una línea) |
| B-08 | Sí, simulando cp1252: con `σ` en la función, `UnicodeEncodeError` al guardar (el modelo ya se ha escrito); con `ñ` en un literal, el `.py` no se puede importar (`SyntaxError`) | | Sí (`encoding='utf-8'`). Sin test específico: en Linux ya es UTF-8 y el test no fallaría en la base |
| S-04 | **No se puede alcanzar:** el constructor ya falla con `partial` o con una instancia invocable (M-06) | | Pasa al PR 5a, junto con M-06 |

**Hallazgos nuevos:**
- **N-10 (alto):** exog con dtype pyarrow (`double[pyarrow]`, admitido con un `DataTypeWarning`) y skops. Se guarda y se carga, pero el primer `predict` mata el proceso (segfault). skops reconstruye `pd.ArrowDtype` como un objeto Cython inválido (`skops_types.py`, `r_pa.py`). Mismo arreglo que M-14.
- **N-11 (medio):** skops no puede serializar `pd.DateOffset(...)` (`TypeError: n argument must be an integer`). Afecta a:
  - `ForecasterEquivalentDate(offset=pd.DateOffset(days=7))`, el uso habitual;
  - cualquier forecaster entrenado con una serie de frecuencia `DateOffset` (p. ej. `pd.DateOffset(months=1)`).

  El resto de offsets funcionan: `D`, `h`, `MS`, `W-SUN`, `B`, `CustomBusinessDay`, `500ms`, `BusinessHour`, `Easter`, etc.
- **Qué no soporta skops** (tipos sueltos, `skops_types.py` y `skops_tz.py`):

  | Tipo | Resultado |
  |---|---|
  | `CategoricalDtype` | Falla al guardar |
  | `DatetimeTZDtype` | Falla al guardar (`RecursionError`) |
  | `ArrowDtype` | Segfault al usarlo tras cargar |
  | `pd.DateOffset` | Falla al guardar |
  | Objetos tz de `pytz` | Fallan al guardar (`RecursionError`); por eso la zona horaria se guarda como texto |
  | `Timestamp` y `Timedelta` | Fallan al cargar, pero ningún atributo de los forecasters los guarda (`scan_attrs.py` recorre su `__dict__`) |

- **`ForecasterEquivalentDate` guarda la serie de entrenamiento entera en `last_window_`.** El comentario del código dice que es por si el `offset` es mayor que los datos disponibles. Es una decisión de diseño y queda fuera del PR.

**Prototipo:**
- **Matriz skops** (`matrix.py`, 30 casos). Cubre:
  - los 6 forecasters con exog numérica, categórica, bool, `Int64` y pyarrow;
  - `window_features`, transformadores, diferenciación e intervalos (conformal con bins y bootstrapping);
  - LightGBM, HGB, XGBoost y CatBoost;
  - índices con zona horaria con y sin cambio de hora, `zoneinfo`, `500ms`, `CustomBusinessDay`, frecuencia `DateOffset` y `RangeIndex`;
  - `ForecasterEquivalentDate`.

  Resultado: en la base, 6 de 30 OK y 1 segfault; en el prototipo, 30 de 30, con predicciones e intervalos idénticos y el mismo dtype del índice.
- **Ficheros guardados con la base** (formato de 0.23 a 0.25): todos cargan con el prototipo, con las mismas predicciones. Los de 500 ms y los de cambio de hora, que hoy no cargan, ahora sí. Los de cambio de hora vuelven con desfase fijo, porque es lo único que guardaban.
- **Tipos de zona horaria:** `str`, `pytz`, `zoneinfo`, `'UTC'` y `datetime.timezone` vuelven idénticos, con el mismo tipo de tz. `dateutil` y `pytz.FixedOffset` dan un `ValueError` claro al guardar.
- **Tests existentes** (`utils`, `preprocessing`, `recursive` y `direct`, sin `slow`): 3266 passed y 3 fallos, los previstos: el formato del payload de `DatetimeIndex` y los dos tests que comprobaban que la descomposición modificaba el forecaster en el sitio.
- **Contexto de IA:** ni `llms-base.txt` ni las skills mencionan `save_forecaster`, así que no hay nada que tocar.

**Diseño del arreglo de skops** (sale del prototipo):
- `_skops_decompose_forecaster` devuelve una copia superficial (`copy(forecaster)`) con los atributos descompuestos. Son los de ahora (`last_window_`, `training_range_`) y además:
  - `exog_dtypes_in_` y `exog_dtypes_out_`;
  - `index_freq_`;
  - `offset`: solo existe en `ForecasterEquivalentDate`.

  `_skops_reconstruct_forecaster` los rehace en el forecaster cargado. Lista explícita de atributos, sin recorrer todo el `__dict__`.
- `_decompose_index` (`DatetimeIndex`):
  - guarda `asi8` (epochs), `unit`, `tz` como texto, `tz_zoneinfo` (bool) y `freq` como objeto (con `_decompose_offset`);
  - `_compose_index` reconstruye con `view('M8[unit]')`, `tz_localize('UTC')` y `tz_convert`;
  - los payloads sin `unit` (formato antiguo) se leen con `format='ISO8601'`, y con `utc=True` si el resultado no es un `DatetimeIndex` (desfases mezclados).
- `_decompose_dtype` y `_compose_dtype`:
  - `CategoricalDtype` → categorías (con `_decompose_index`) y `ordered`;
  - `ArrowDtype` y `DatetimeTZDtype` → `str(dtype)`, que se rehace con `pandas_dtype`;
  - el resto de dtypes no se tocan (skops los soporta).
- `_decompose_offset` y `_compose_offset`: solo `type(x) is pd.DateOffset` → `n`, `normalize` y `kwds`.

**Decisión del usuario (2026-10-06): dividir el PR 3 en 3a y 3b.** Tienen perfiles de revisión distintos:
- **3a** es pequeño, toca el backend por defecto (joblib) y contiene el único cambio de comportamiento (M-16, pérdida de datos sin aviso).
- **3b** son los internos de skops, con un cambio del formato del fichero y su compatibilidad hacia atrás.

Tocan trozos distintos de `save_forecaster`; el único choque esperado es la nota de versión. El 3a va primero porque es rápido de revisar y es el de más impacto para el usuario por defecto.

#### PR 3a · `fix/save-forecaster-file-names`
Título: *Fix file names and custom function export in save_forecaster*

| # | Commit | Hallazgo | Contenido | Tests |
|---|---|---|---|---|
| 1 | Keep dotted file names and only replace known backend extensions | M-16 | `save_forecaster` y el docstring de `file_name`; celda 13 (markdown) de `save-load-forecaster.ipynb`, sin volver a ejecutarlo | Parametrizado con la tabla de M-16: `'model'`, `'model.pkl'` con joblib, `'model_v1.2'`, `'forecaster_2026.10.04'`, `'model.bin'`; dos nombres con puntos ya no se pisan |
| 2 | Do not warn about RollingFeaturesClassification when saving | B-07 | Añadir el nombre al conjunto | Sin `SaveLoadSkforecastWarning` con `RollingFeaturesClassification` |
| 3 | Write exported weight functions as UTF-8 | B-08 | `open(..., encoding='utf-8')` | Sin test específico |
| 4 | Add release notes for save_forecaster fixes | | Changed (M-16); Fixed (B-07, B-08) | |

#### PR 3b · `fix/skops-persistence`
Título: *Fix skops persistence of time zones, window features, categorical exog and DateOffset*

| # | Commit | Hallazgo | Contenido | Tests |
|---|---|---|---|---|
| 1 | Stop storing pandas Rolling objects in RollingFeatures | A-09, O-02 | `transform_batch` de las dos clases con un dict local; se quita la clave `'rolling_obj'` | Ajustar las 3 aserciones de `__init__`; round trip skops con `window_features` en Recursive, Direct, MultiSeries, DirectMultiVariate y Classifier, y con `RangeIndex` |
| 2 | Serialize a shallow copy of the forecaster with skops | B-05 | `_skops_decompose_forecaster` devuelve la copia; `save_forecaster` sin `try/finally` | Reescribir los dos tests de descomposición (el original no cambia); el de no mutación ya existe |
| 3 | Store DatetimeIndex as epochs with unit and time zone for skops | A-08, M-15, B-06, O-01 | Nuevo payload, error con zonas que no se pueden reconstruir y fallback del formato antiguo | Actualizar `test_decompose_index_output[datetime]`; round trips con cambio de hora, zona sin cambio de hora, `zoneinfo`, `500ms` y `CustomBusinessDay`; payloads antiguos (texto, desfases mezclados, fracciones de segundo); `ValueError` con `dateutil` |
| 4 | Serialize categorical and pyarrow exog dtypes for skops | M-14, N-10 | `_decompose_dtype` y `_compose_dtype` | Exog categórica en una serie y multiserie con dict; exog pyarrow |
| 5 | Serialize pandas DateOffset for skops | N-11 | `_decompose_offset` y `_compose_offset` en `index_freq_`, `offset` y la frecuencia del payload. Docstring de `save_forecaster` y celda 15 de la guía: qué se descompone | `ForecasterEquivalentDate` con `DateOffset`; serie con frecuencia `DateOffset(months=1)` |
| 6 | Add release notes for skops fixes | | Fixed: A-09 (con el tamaño de joblib y el `deepcopy`), A-08 (con M-15, B-06 y el tiempo de O-01), M-14 y N-10, N-11, B-05 | |

S-04 pasa al PR 5a (commit 2, junto con M-06).

Ajustes aprobados por el usuario (2026-10-06, con «Empieza el PR 3a»): B-07 por nombre; en el 3b, conservar `ZoneInfo`, el error al guardar con zonas que no se reconstruyen y quitar la clave `'rolling_obj'`; S-04 pasa al PR 5a.

#### PR 3a: implementación (2026-10-06): skforecast/skforecast#1350, fusionado (`7e352bc`)

Subido el 2026-10-06 12:37 UTC con el OK del usuario: título *Fix file names and custom function export in save_forecaster*, head `c4e2422`. Pie quitado, sesión suscrita a la actividad del PR y check-in de seguridad `trig_019wvidUxeHsvtmFUZ6NMzex` (13:29 UTC). Cuerpo en `review/pr3a_body.md`.

**Fusionado** el 2026-10-06 13:28 UTC (merge commit `7e352bc`), con el título original y sin commits del usuario (mismo contenido que `c4e2422`). CI verde (`check`, CodeQL, Analyze). Desuscrito y check-in borrado. La rama remota `fix/save-forecaster-file-names` sigue existiendo (no borrarla sin permiso).

Rama `fix/save-forecaster-file-names` desde `origin/0.26.x` (`59822eb`):

| # | Commit | Contenido |
|---|---|---|
| 1 | `ca07680` Keep dotted file names and only replace known backend extensions | M-16: `save_forecaster` (conjunto `known_extensions`, comparación sin mayúsculas), docstring de `file_name`, celda 13 (markdown) de `save-load-forecaster.ipynb` (diff de una línea, sin volver a ejecutar). Test parametrizado con 7 casos (`tmp_path`), que además carga el fichero con el backend inferido |
| 2 | `da52e19` Do not warn about RollingFeaturesClassification when saving | B-07: el nombre en el conjunto. Test sin aviso con `RollingFeatures` y `RollingFeaturesClassification` |
| 3 | `aa25b66` Write exported weight functions as UTF-8 | B-08. Sin test (en Linux ya es UTF-8) |
| 4 | `c4e2422` Add release notes for save_forecaster fixes | Changed (M-16, al final de Changed); Fixed (B-07, B-08, al final de Fixed). Sin highlight |

Verificación:
- Los tests nuevos fallan en la base: 3 de los 7 casos de M-16 (nombre con puntos, fecha con puntos y extensión desconocida) y el de `RollingFeaturesClassification`.
- B-08 comprobado a mano con un locale ASCII (`LC_ALL=C PYTHONUTF8=0 PYTHONCOERCECLOCALE=0`): la base da `UnicodeEncodeError` y deja un `.py` vacío; la rama lo guarda y se importa con `σ`.
- Las llamadas a `save_forecaster` de la documentación (`save-load-forecaster.ipynb`, `forecaster-in-production.ipynb`) usan la extensión de su backend: no cambian.
- `verify`: ruff limpio; `utils` 674 passed; contexto IA al día; referencias de la nota de versión definidas; cada commit pasa `test_save_load_forecaster.py` en un worktree aparte (41, 43, 43, 43).

Revisión final (2026-10-06), sin cambios en los commits:
- Lógica de nombres comparada con la antigua en 21 nombres × 4 backends (directorios con puntos, ficheros ocultos, nombre acabado en punto, extensiones en mayúsculas, `model.tar.joblib`, rutas relativas): solo cambian los nombres con una extensión no vacía y desconocida, que es lo buscado.
- Convenciones: dos líneas en blanco entre tests, sin trailers en los commits, ruff limpio, sin otras menciones al comportamiento antiguo en docs, skills ni `llms-base.txt` (la página de la API sale del docstring).
- Observaciones que no se cambian: `known_extensions` repite las claves de `extension_backend_map` de `load_forecaster` (cinco extensiones; una constante común sería un refactor); los `.py` de `weight_func` se escriben en el directorio de trabajo y no junto al modelo (diseño previo); los tests nuevos usan `tmp_path` y los antiguos el directorio actual con `os.remove`.

**Exportación de `weight_func` (pregunta del usuario tras la revisión final, 2026-10-06; scripts en `review/next/wfloc*`):**
- **N-12 (medio, nuevo):** el `.py` exportado solo contiene el código de la función, sin los imports que usa. Con la función de la guía (`np.where`), el modelo cargado predice bien, pero al reentrenarlo (`fit`, backtesting con `refit`) falla con `NameError: name 'np' is not defined`.
- **Colisión:** los `.py` se escriben en el directorio de trabajo con el nombre de la función. Dos modelos guardados en `store_a/` y `store_b/` con una función que se llama igual escriben el mismo `./custom_weights.py` y el segundo pisa al primero sin aviso, así que al cargar el modelo A se importaría la función del B.
- **Guardarlo junto al modelo:**
  - a favor: el modelo y sus funciones quedan juntos (se copia la carpeta a producción), no aparecen ficheros sueltos en la raíz del proyecto y se evita la colisión entre carpetas;
  - en contra: es un cambio de comportamiento (Changed). Con el modelo en `models/`, `from custom_weights import custom_weights` da `ModuleNotFoundError`; hace falta `from models.custom_weights import custom_weights` (paquete de espacio de nombres, el nombre de la carpeta debe ser un identificador válido) o añadir la carpeta a `sys.path`.
  - Desde un script lanzado en otro directorio no funciona ninguna de las dos ubicaciones, porque `sys.path[0]` es la carpeta del script y no el directorio de trabajo.
- `backend='cloudpickle'` resuelve las tres cosas: guarda la función dentro del fichero, y el modelo carga y reentrena sin ficheros aparte.
- **Decisión del usuario (2026-10-06):** no va en el PR 3a. Se crea el **PR 3c** (abajo), documentado para hacerlo más adelante, sin implementar.

#### PR 3b: implementación (2026-10-06): skforecast/skforecast#1352, fusionado (`b08b5b6`)

Subido el 2026-10-06 14:40 UTC con el OK del usuario («OK, sube el PR 3b»), que da por buenas las desviaciones del plan; sin highlight. Título *Fix skops persistence of time zones, window features, categorical exog and DateOffset*, head `ac2f777`. Pie quitado, sesión suscrita a la actividad del PR y check-in de seguridad `trig_01JgZurF46feQ4tkZd1iC8oo` (15:32 UTC). Cuerpo en `review/pr3b_body.md`.

**Commit del usuario (2026-10-06 15:18 UTC):** `4b5c3e1` *Fix categorical exog data in skops round-trip test Fix skops round-trip tests: seen categories and no pytz dependency* (un solo commit con las dos correcciones):
1. En `test_save_and_load_forecaster_round_trip_skops_exog_dtypes`, `month` pasa a `day_of_week` (también `int32`) y se añade `assert not predictions.isna().to_numpy().any()`. **Bug mío del test:** con 60 días desde el 1 de enero, el mes 3 de la exog de predicción no se vio al entrenar, el `OrdinalEncoder` lo codificaba como NaN y las predicciones eran NaN; la comparación daba por iguales NaN y NaN, así que el test pasaba sin comprobar nada útil en esos dos casos.
2. Fuera `pytz` del test del `ValueError` (con pandas 3 deja de ser dependencia); quedan `dateutil` y el `datetime.timezone` con nombre.

Revisado: correcto. Los dos ficheros de skops, 107 passed; ruff limpio. Un plugin de pytest (`scratchpad/nancheck/nan_plugin.py`) que intercepta las comparaciones de los round trips no encuentra predicciones NaN en la rama, y en la versión anterior marca exactamente los dos casos categóricos. Descripción del PR actualizada (16 casos de DatetimeIndex, nota sobre los números de versiones mínimas, comprobación de NaN). CI verde en `4b5c3e1` (`check`, CodeQL, Analyze). `0.26.x` avanzó a `fb3bfd3` (#1351, contexto de IA y README): solo coincide `releases.md` y se fusiona sin conflictos.

**Lección:** en los tests de round trip, comprobar que las predicciones de referencia no son NaN; si no, la comparación con el modelo cargado no prueba nada.

**Fusionado** el 2026-10-06 15:21 UTC (merge commit `b08b5b6`), con el contenido de `4b5c3e1`. Desuscrito automáticamente y check-in `trig_01JgZurF46feQ4tkZd1iC8oo` borrado. El usuario borró la rama remota; queda la local `fix/skops-persistence` (no borrarla sin permiso).

Rama `fix/skops-persistence` desde `origin/0.26.x` (`7e352bc`). Título previsto: *Fix skops persistence of time zones, window features, categorical exog and DateOffset*.

| # | Commit | Hallazgos | Contenido |
|---|---|---|---|
| 1 | `75c2381` Stop storing pandas Rolling objects in RollingFeatures | A-09, O-02 | `transform_batch` de las dos clases con un dict local; fuera la clave `'rolling_obj'` (4 tests de `__init__` ajustados, 6 entradas). Tests: `transform_batch` no guarda estado (uno por clase); round trip skops con `window_features` en Recursive (fecha y `RangeIndex`), Direct, MultiSeries, DirectMultiVariate y Classifier |
| 2 | `13ee4cf` Serialize a shallow copy of the forecaster with skops | B-05 | `_skops_decompose_forecaster` devuelve `copy(forecaster)`; `save_forecaster` sin `try/finally`. Los dos tests de descomposición comprueban que el original no cambia |
| 3 | `f0f7dad` Store DatetimeIndex as epochs with unit and time zone for skops | A-08, M-15, B-06, O-01 | Payload `asi8`, `unit`, `tz`, `tz_zoneinfo`, `freq` (objeto); `ValueError` al guardar si la zona no se reconstruye; lectura de los payloads antiguos. Tests: payload (naive y Madrid), `ValueError` (dateutil `tzoffset` y `pytz.FixedOffset`), round trips de índice (Madrid con cambio de hora, zoneinfo, UTC, `datetime.timezone`, `500ms`, unidad `s`, `CustomBusinessDay`), payloads antiguos (naive, fracciones de segundo, un desfase, desfases mezclados) y 5 round trips con forecaster |
| 4 | `4228767` Serialize categorical and pyarrow exog dtypes for skops | M-14, N-10 | `_decompose_dtype` y `_compose_dtype` en `exog_dtypes_in_` y `exog_dtypes_out_`. Tests: round trip de dtypes, dtypes que no cambian, forecaster (Recursive y MultiSeries con dict) con exog categórica (`int32` y texto) y pyarrow |
| 5 | `28048b6` Serialize pandas DateOffset for skops | N-11 | `_decompose_offset` y `_compose_offset` en `index_freq_`, `offset`, `window_size` y la `freq` del payload; docstring de `save_forecaster`, comentario de `load_forecaster` y celda 15 de la guía (solo markdown). Tests: round trip del offset, valores que no cambian, índice con frecuencia `DateOffset`, `ForecasterEquivalentDate` (sin entrenar y entrenado) y Recursive con frecuencia `DateOffset(months=1)` |
| 6 | `ac2f777` Add release notes for skops persistence fixes | | Cinco entradas en Fixed, tras las del PR 3a; sin highlight |

**Desviaciones del plan (para el punto de control):**
1. **Lectura de los ficheros antiguos.** En lugar de `format='ISO8601'` y, si no sale un `DatetimeIndex`, `utc=True`, se leen siempre con `utc=True` y se convierten al desfase del primer timestamp (o sin zona). Con desfases mezclados, el primer `pd.to_datetime` emite un `FutureWarning` en pandas 2.x (que el usuario vería al cargar) y en pandas 3 lanza `ValueError`. El resultado es idéntico para los ficheros sin cambio de hora (misma zona `UTC+01:00` que hoy); los que tienen cambio de hora vuelven con el desfase del primer timestamp en lugar de UTC.
2. **Categorías como array de numpy** (`categories.to_numpy()`) en lugar de `_decompose_index`, que guarda los enteros como lista y los reconstruye como `int64`. `index.month` da `int32`, así que `exog_dtypes_in_` cambiaba tras cargar. skops guarda arrays `int32`, de texto (`object`) y `datetime64` sin tipos no fiables.
3. **`window_size` en la lista de offsets.** En `ForecasterEquivalentDate` es un `DateOffset` hasta `fit` (y tras `set_params`). Sin él, guardar el forecaster sin entrenar falla (comprobado quitándolo: el test falla).
4. `except (KeyError, ValueError)` en la comprobación de la zona horaria, en lugar de `Exception` (pytz y zoneinfo lanzan subclases de `KeyError`).
5. Frase de la celda 15 más corta (regla de docs). No se vuelve a ejecutar el notebook: la lista de tipos no fiables del ejemplo de la guía es la misma en la base y en la rama.

**Límite que queda (también en la base):** un fichero antiguo con un cambio de hora dentro de `last_window_` y frecuencia diaria o mayor no carga (`Inferred frequency None ... does not conform to passed frequency D`). El fichero solo guardaba los desfases y la zona no se puede recuperar. La nota de versión dice "most of those with a daylight saving time change".

**Cambio de comportamiento:** con `dateutil` o `pytz.FixedOffset`, skops ahora lanza `ValueError` al guardar; antes guardaba y cargaba con desfase fijo. Es el diseño aprobado; va en la entrada de Fixed.

**Verificación:**
- Tests nuevos contra la base (o contra el commit anterior, para los del commit 5): commit 1, 14 fallos (6 round trips skops, 2 de estado, 6 de `__init__` esperados); commit 2, 2 fallos; commit 3, 16 fallos (los casos que ya funcionaban, UTC, desfase fijo, sin zona, `RangeIndex` y payloads antiguos sin cambio de hora, pasan, como regresión); commit 4, 2 fallos con exog categórica y 2 segfaults con pyarrow; commit 5, 2 fallos.
- Cada commit pasa `test_save_load_forecaster.py`, `test_skops_decompose_compose.py` y los dos ficheros de `RollingFeatures` en un worktree aparte: 108, 108, 127, 139, 151, 151.
- Matriz skops (`matrix.py`): 30 de 30 en la rama (base: 6 de 30 y un segfault).
- Ficheros antiguos: `old/` (7) cargan igual que con la base (Madrid sin y con cambio de hora, 500 ms y el resto). `old2/` (`old_gen2.py`): Nueva York diario con el cambio en el rango de entrenamiento y Madrid 15 min con el cambio, que fallaban, ahora cargan con las mismas predicciones e instantes. `old3/` (cambio de hora dentro de `last_window_`, diario) falla en las dos.
- Medidas (`o01_time.py`, `o02_size.py`): `ForecasterEquivalentDate` con 200 000 valores en skops, guardar 3.4 s, cargar 3.4 s, 58 MB → 0.02 s, 0.01 s, 3.3 MB. Forecaster con `RollingFeatures` y 500 000 valores en joblib: 16.0 MB → 0.006 MB; `deepcopy` 5.6 → 0.24 ms.
- `verify`: ruff limpio en los ficheros cambiados; `utils` y `preprocessing` 1070 passed; `recursive` y `direct` (sin `slow`) 2259 passed y 1 skipped; contexto de IA al día (ni `llms-base.txt` ni las skills mencionan skops); referencias de la nota de versión definidas. No ejecutados: `model_selection`, `deep_learning`, `stats` y el resto (el cambio de `RollingFeatures` no altera los resultados).
- Revisión del diff: el docstring de `_skops_decompose_forecaster` y el test de dtypes (que compartía la instancia del forecaster entre casos) se corrigieron con un fixup en el commit 4.

**Revisión final (2026-10-06), con dos correcciones en los commits locales:**
1. **Zona horaria con nombre engañoso (regresión de la rama, corregida en el commit 3).** Un `datetime.timezone(timedelta(hours=1), 'CET')` se guardaba como `'CET'` y se reconstruía como la zona CET con horario de verano: las predicciones de julio salían con `+02:00` (en la base cargaba bien, con desfase fijo). La comprobación al guardar ya no se limita a que el nombre se pueda leer (`pd.Timestamp(0, tz=tz)`): reconstruye la zona como lo hace la carga y compara el dtype (`pd.DatetimeTZDtype(unit, tz_rebuilt) == index.dtype`, que usa `tz_compare` de pandas). Probado con 16 tipos de zona: pytz, zoneinfo, UTC y `datetime.timezone` sin nombre pasan; `datetime.timezone` con nombre, dateutil, `pytz.FixedOffset` y `'dateutil/...'` dan `ValueError`. Nuevo caso `CET` en el test del `ValueError`. Mensaje del commit 3 ajustado.
2. **Nota de versión de `CustomBusinessDay`.** En la base fallaba `predict` o, si un festivo caía dentro de `last_window_`, la carga (`cbd_tz_check.py`). Dice ahora "so loading or `predict` failed" (y el mensaje del commit 3 igual).

Comprobado además:
- Versiones mínimas (`venv_pd22`: pandas 2.2.0, numpy 1.26.4, scikit-learn 1.6.0): los cuatro ficheros de tests, 152 passed. skops 0.14.0 (la mínima): los dos ficheros de skops, 108 passed (después se restauró skops 0.16.0 en ese entorno).
- `model_selection` (sin `slow`): 786 passed. Con los de antes (`utils` y `preprocessing` 1071, `recursive` y `direct` 2259), solo quedan sin ejecutar `stats`, `deep_learning`, `foundation`, `drift_detection`, `metrics`, `plot`, `datasets` y `feature_selection`, que no usan este código.
- Matriz skops 30 de 30, ficheros antiguos iguales y cada commit en su worktree (108, 108, 128, 140, 152, 152) sobre la historia final.
- No aparecen tipos nuevos que confiar en skops: los payloads solo tienen `str`, `int`, `bool` y arrays de numpy, y los offsets que se guardan ya estaban en `index_freq_`.
- `ForecasterFoundation` tiene `window_size` y `last_window_` como propiedades de solo lectura, pero skops lo rechaza antes de descomponer (igual que antes).
- `pyarrow` está en los extras de test (`pandas[parquet]`), que es lo que instala la CI; uno de los tests lo usa al recoger el módulo.

Riesgos residuales, sin cambio:
- Ficheros antiguos con cambio de hora dentro de `last_window_` y frecuencia diaria o mayor: siguen sin cargar (también en la base).
- Las zonas que no se reconstruyen desde su nombre dan `ValueError` al guardar con skops; antes se guardaban y volvían con desfase fijo (diseño aprobado; está en la nota).
- Exog con columnas de fecha con zona horaria: el `DatetimeTZDtype` se guarda por nombre, así que vuelve como pytz aunque fuera zoneinfo, y uno de dateutil no cargaría. Los modelos no usan esas columnas directamente; es un caso muy raro.
- Si el segfault de pyarrow volviera, el test mataría el proceso de pytest en lugar de fallar.


#### PR 3c · `fix/weight-func-export` (documentado el 2026-10-06; implementado y fusionado, ver abajo)
Título: *Fix the export of custom weight functions in save_forecaster*

**Qué arregla** (reproducciones en `review/next/wfloc/` y `wfloc2/`):
- **N-12:** el `.py` exportado no incluye los imports que usa la función. El modelo cargado predice, pero `fit` o un backtesting con `refit` dan `NameError: name 'np' is not defined`. Pasa con la función de la guía (`np.where`).
- **Colisión:** los `.py` se escriben en el directorio de trabajo con el nombre de la función. Dos modelos guardados en carpetas distintas con funciones que se llaman igual se pisan sin aviso, y al cargar el primero se importaría la función del segundo.
- **S-04 y M-06**, si se mueven del PR 5a (decisión 3):
  - M-06: un `functools.partial` o una instancia invocable como `weight_func` hacen fallar el constructor (`inspect.getsource`);
  - S-04: una vez arreglado M-06, `save_forecaster` tampoco los trataría bien. Un `partial` tiene `__module__ == 'functools'`, así que no se exportaría ni avisaría; una instancia tiene `__module__ == '__main__'` pero no `__name__`, así que falla al ordenar por nombre.

**Orden:** después del PR 3a, porque toca el mismo bloque de `save_forecaster` (incluida la línea de B-08).

**Decisiones pendientes del usuario, antes de implementar:**
1. **Dónde se escriben los `.py`:**
   - junto al fichero del modelo (**recomendado**: evita la colisión y deja el modelo y sus funciones juntos; va en Changed);
   - o, como ahora, en el directorio de trabajo.

   Con el modelo en `models/`, el import pasa a ser `from models.custom_weights import custom_weights` (o añadir la carpeta a `sys.path`).
2. **Cómo arreglar N-12:**
   - (a) escribir los imports que usa la función;
   - (b) solo avisar y recomendar `cloudpickle`;
   - (c) las dos (**recomendado**).
3. **Mover M-06 y S-04 del PR 5a al 3c** (**recomendado**): todo lo de `weight_func` queda en un PR, y S-04 solo se puede probar con M-06 arreglado. Si no se mueven, se quedan en el commit 2 del 5a.

**Commits (con las recomendaciones):**

| # | Commit | Hallazgo | Contenido | Tests |
|---|---|---|---|---|
| 1 | Write the imports used by exported weight functions | N-12 | Antes del código de la función, escribir los imports de lo que usa, leídos con `inspect.getclosurevars(fun).globals`: módulos (`import numpy as np`) y funciones o clases importadas (`from scipy.stats._stats_py import skew`, con alias si el nombre local es otro). Las demás variables globales (constantes, objetos) no se pueden reconstruir: aviso que las nombra y recomienda `backend='cloudpickle'`. Revisar al implementar los `nonlocals` (closures) y los builtins | Contenido del `.py` con `np`, con una función importada y con una constante global (aviso); importar el `.py` generado y reentrenar el modelo cargado |
| 2 | Save exported weight functions next to the forecaster file | Colisión | Escribir en `Path(file_name).parent`; el aviso muestra la ruta. Docstring de `save_custom_functions`; advertencias de las celdas 25 y 35 de `save-load-forecaster.ipynb`. Los ejemplos de la guía guardan en el directorio de trabajo, así que su código no cambia; volver a ejecutar el notebook si cambia el texto del aviso | Modelo en `tmp_path / 'models'`: el `.py` queda en esa carpeta; dos modelos en carpetas distintas con funciones homónimas no se pisan |
| 3 | Accept functools.partial and callable instances as weight_func | M-06, S-04 | `initialize_weights` no falla si no hay código fuente (`source_code_weight_func = None`). `save_forecaster` no exporta los invocables sin `__name__` o sin código fuente y avisa (recomienda `cloudpickle`); el `partial` se detecta por su función (`.func.__module__`) | Constructor con `partial` e instancia; aviso al guardar; round trip con `cloudpickle` |
| 4 | Add release notes for weight function export | | Fixed: N-12, M-06 y S-04. Changed: ubicación de los `.py` | |

`backend='cloudpickle'` ya resuelve las tres cosas (comprobado: carga y reentrena sin ficheros aparte); la nota de versión y los avisos deben seguir recomendándolo.


#### PR 3c: implementación (2026-10-06): skforecast/skforecast#1353, fusionado (`adbf335`)

Subido el 2026-10-06 15:54 UTC con el OK del usuario («OK, sube el PR 3c»). Título *Fix the export of custom weight functions in save_forecaster*, head `24a1a5b`. Pie quitado, sesión suscrita a la actividad del PR y check-in de seguridad `trig_017QoeEEr47RzQoYtVeokE27` (16:45 UTC). Cuerpo en `review/pr3c_body.md`.

Decisiones del usuario (2026-10-06, «acepto las recomendaciones»): los `.py` junto al fichero del modelo; N-12 con imports y aviso; M-06 y S-04 pasan del PR 5a al 3c.

Rama `fix/weight-func-export` desde `origin/0.26.x` (`b08b5b6`). Título previsto: *Fix the export of custom weight functions in save_forecaster*.

| # | Commit | Hallazgos | Contenido |
|---|---|---|---|
| 1 | `dacb535` Write the imports used by exported weight functions | N-12 | Nuevo `_get_source_with_imports`: lee los nombres globales del bytecode (`LOAD_GLOBAL`/`LOAD_NAME`, también en el código anidado) y escribe `import x as y` o `from m import f as g` si el objeto se recupera del módulo (`sys.modules[m].f is valor`). El resto (constantes, funciones de `__main__`, variables de una función envolvente) va a un `SaveLoadSkforecastWarning` que recomienda `cloudpickle`. Tests: fichero nuevo `test_get_source_with_imports.py` (4 casos), contenido del `.py` con `import numpy as np`, el módulo exportado se importa y funciona, aviso con una constante global |
| 2 | `825442c` Save exported weight functions next to the forecaster file | Colisión | `file_name.parent / f"{nombre}.py"`; el aviso muestra la ruta (igual que antes si el modelo está en el directorio de trabajo, así que el notebook no cambia de salida). Docstring de `save_forecaster` y celda 25 de la guía (markdown). Test: modelo en `models/`, nada en el directorio de trabajo |
| 3 | `5b4167a` Accept functools.partial and callable objects as weight_func | M-06, S-04 | `_get_source_code` (devuelve `None` con `OSError`/`TypeError`) en `initialize_weights`. En `save_forecaster`, un `partial` exporta la función que envuelve; las lambdas y los objetos invocables no se exportan y dan un aviso. Tests: `initialize_weights` con `partial` y objeto; `save_forecaster` con los dos |
| 4 | `24a1a5b` Add release notes for weight function export fixes | | Changed: ubicación de los `.py`. Fixed: N-12; M-06 + S-04 en una entrada (aceptar `partial` y objetos es nuevo en este ciclo, así que su exportación no es un bug publicado, salvo la lambda con skops) |

**Por qué el bytecode y no `inspect.getclosurevars`** (medido con `n12/probe.py`):
- `getclosurevars` toma los nombres de atributo como variables globales: con un global `month` y `index.month` en la función, avisaría en falso.
- `getclosurevars` solo mira el código principal de la función. Los generadores, las lambdas y las comprensiones en Python 3.10 y 3.11 son código anidado: un `np` usado solo ahí quedaría sin import, que es justo el bug N-12.

**Probado:**
- De punta a punta, en procesos separados (`n12/e2e`): guardar con joblib, importar el `.py`, cargar y reentrenar. En el directorio de trabajo y en `models/` (`from models.custom_weights import custom_weights`).
- `partial` con joblib (tras `from w import w`) y con cloudpickle: carga y reentrena (`s04/e2e`).
- `probe_s04.py` con `partial`, objeto y lambda en joblib, pickle y skops:
  - en la base, los dos primeros fallan en el constructor, y la lambda con skops escribe `<lambda>.py`;
  - en la rama, el `partial` exporta `w.py`, el objeto avisa y la lambda con skops avisa;
  - la lambda con joblib o pickle sigue dando el `PicklingError` de pickle, como antes.
- El helper en Python 3.10 (numpy 1.26), 3.12, 3.13 y 3.14 (`pyver/probe_helper.py`): mismos resultados.
- Límite: con numpy 1.26, las ufuncs (`from numpy import exp`) no tienen `__module__`, así que no se escribe su import y van al aviso.
- Tests nuevos contra el commit anterior o la base: N-12, 3 casos de contenido, módulo independiente y aviso; ubicación, 1; M-06 y S-04, 4.
- Cada commit en su worktree, con `test_save_load_forecaster.py`, `test_initialize_weights.py` y `test_get_source_with_imports.py`: 79, 80, 84 y 84 passed, sin `.py` sueltos.
- `utils`: 734 passed. Tests de `recursive` y `direct` con `weight` (sin `slow`): 75 passed.
- ruff limpio salvo el `pandas` sin usar que ya existía en `test_initialize_weights.py`. Contexto IA al día. Nota de versión sin referencias sin definir.

**Riesgos y límites que quedan:**
- El código de una función anidada (closure) se escribe indentado. Ya no funcionaría solo por la variable envolvente, que va al aviso.
- ~~Una función definida en una consola interactiva sin fichero da `OSError` en `inspect.getsource` al guardar (ya pasaba antes).~~ Resuelto por el commit del usuario `4478ba0` (ver abajo).
- Las funciones auxiliares de `__main__` que use la función de pesos no se exportan; van al aviso.

**Revisión final (2026-10-06), con una corrección en el commit 1:**
- **Bug de la rama, corregido:** los nombres que solo aparecen en la firma (anotaciones y valores por defecto) o en un decorador no están en el bytecode de la función, y se evalúan al importar el módulo. Con `def custom_weights(index: pd.DatetimeIndex, cutoff=pd.Timestamp(...))`, el `.py` solo llevaba `import numpy as np` y al importarlo daba `NameError: name 'pd' is not defined` (`n12/annot`).
- **Arreglo:** se compila el código fuente como un módulo (`compile(textwrap.dedent(source), ..., dont_inherit=True)`) y se leen los nombres de ese bytecode. Así entran el cuerpo, la firma y los decoradores. El `dedent` es para las closures, y sus variables libres se excluyen de la búsqueda.
- **`dont_inherit=True`:** sin él, `compile` hereda el `from __future__ import annotations` de `utils.py` y compila las anotaciones como texto. Lo cazó el test nuevo: mi sonda aislada no tenía ese import.
- Nuevo caso `signature` en `test_get_source_with_imports_output`. Falla sin el arreglo y pasa con él.
- Mensaje del commit 1 actualizado.

**Comprobado tras la corrección:**
- La sonda del helper (`pyver/probe_helper.py`, 6 casos: módulos, función con alias, no importables, closure, firma y decorador) da los mismos resultados en Python 3.10, 3.11, 3.12, 3.13 y 3.14.
- `utils`: 735 passed. Tests de `recursive` y `direct` con `weight`: 75 passed. ruff limpio. Contexto IA al día.
- Cada commit en su worktree: 80, 81, 85 y 85 passed, sin `.py` sueltos.
- De punta a punta: directorio de trabajo, `models/`, función anotada, `partial` con joblib y cloudpickle, y `probe_s04.py`. Todo igual que antes.
- Otros notebooks que mencionan `source_code_weight_func` (`weighted-time-series-forecasting`, multiserie, multivariante): solo imprimen el de funciones normales y no cambian.
- `ForecasterRecursiveMultiSeries` rellena las series sin función en `weight_func_` (atributo de entrenamiento) con `_weight_func_all_1`. No afecta a la exportación, que usa `weight_func`.

**Riesgos residuales, sin cambio:**
- Una función envuelta con un decorador basado en `functools.wraps` toma las variables libres del envoltorio, lo que puede dar un aviso de más. No rompe el fichero, porque el decorador se vuelve a aplicar al importarlo, y es muy raro en funciones de pesos.
- Las funciones de una librería se importan desde su módulo real, que puede ser privado (p. ej. `from scipy.stats._stats_py import skew`). Funciona igual que pickle, pero puede romper si la librería mueve ese módulo.


**Commit del usuario (2026-10-06 16:11 UTC):** `4478ba0` *Skip exporting weight functions whose source code is not available.*
1. En `save_forecaster`, una función de `__main__` sin código fuente disponible (`_get_source_code(fun) is None`: consola, `exec`) pasa a la lista de no exportables y sale en el mismo aviso que las lambdas y los objetos invocables. Antes de este commit daba `OSError` después de escribir el fichero del modelo (era el límite conocido de arriba). Texto del aviso ampliado.
2. Ancla de la documentación de los dos avisos de `save_forecaster`: `#saving-and-loading-a-forecaster-model-with-custom-features` (ya no existe) pasa a `#forecaster-with-custom-features` (encabezado *Forecaster with Custom Features*).
3. Fuera el `import pandas as pd` sin usar de `test_initialize_weights.py`.
4. Nota de versión: la entrada de M-06 + S-04 añade la función sin código fuente.
5. Test: caso `no_source` en `test_save_forecaster_save_custom_functions_partial_and_callable_object` (función creada con `exec`).

Revisado: correcto, sin cambios necesarios.
- El caso `no_source` falla en `24a1a5b` con `OSError: could not get source code`; `callable_object` también falla allí, pero solo porque cambia el texto del aviso.
- Consola real de Python 3.12 (`script`): el forecaster se crea (`source_code_weight_func` es `None`), al guardar sale el aviso y solo se escribe `forecaster.joblib`.
- Tests: los 3 ficheros tocados, 86 passed; `utils`, 736 passed; `recursive` y `direct` con `weight`, 75 passed. ruff limpio, ya sin el aviso del `pandas`. Contexto IA al día. Nota de versión sin referencias sin definir.
- El ancla nueva sigue la convención de los otros enlaces del paquete (`#input-data` para *Input data*: minúsculas y guiones). No pude abrir la web para confirmarlo (el proxy da 403).

Observaciones opcionales, sin bloquear:
- En Python 3.13 y 3.14 la consola nueva sí guarda el código fuente (`inspect.getsource` funciona, comprobado con `script`), así que esas funciones se exportan como `.py`. «defined in the Python console», en el aviso y en la nota, vale para Python 3.12 o anterior y para la consola básica (`PYTHON_BASIC_REPL`). Lo mismo con el docstring de `_get_source_code`.
- El nombre del test (`..._partial_and_callable_object`) ya no cubre el tercer caso, y su docstring dice «a function whose source code is not available defined in '__main__'».
- Las salidas guardadas de las celdas 26 y 41 de `save-load-forecaster.ipynb` muestran el ancla vieja en los avisos. Ya estaban desfasadas (líneas `utils.py:2782` y `2806`), y se actualizan cuando se vuelva a ejecutar el notebook.

Descripción del PR actualizada: comportamiento nuevo, ancla, límites, número de tests y ruff. CI verde en `4478ba0` (`check`, CodeQL, Analyze).

**Fusionado** el 2026-10-06 16:24 UTC (merge commit `adbf335`), con el contenido de `4478ba0`; el usuario lo pasó a «ready for review» a las 16:24. Desuscrito automáticamente y check-in `trig_017QoeEEr47RzQoYtVeokE27` borrado. `0.26.x` local actualizada. La rama remota `fix/weight-func-export` sigue existiendo, igual que la local (no borrarlas sin permiso).

### 5.9 PR 7: job de CI con las versiones mínimas (documentado el 2026-10-06; implementado el 2026-10-07)

Petición del usuario (2026-10-06, antes de subir el PR 3b): añadir un PR para un job de CI que instale las versiones mínimas. Es el seguimiento propuesto en §5.6 y §5.3, que el PR 6 dejó fuera.

Rama prevista: `ci/minimum-versions-job`. Título: *Add a CI job that tests the minimum supported versions*. Es independiente del resto de PRs.

**Por qué:**
- La CI nunca instala las versiones mínimas: `unit-tests.yml` instala lo último que resuelve y `unit-tests-latest-deps.yml` fuerza lo último cada lunes.
- Los fallos con los mínimos (M-05, N-01, M-13 y N-09) solo se vieron en el PR 6, con entornos montados a mano.
- Hoy hay un mínimo declarado que no se puede instalar (statsmodels, abajo), y un test que falla con el mínimo de scikit-learn.

**Medido el 2026-10-06** (entorno `scratchpad/venv_min310`, Python 3.10):
- `uv pip compile pyproject.toml --extra test --resolution lowest-direct --python-version 3.10` resuelve exactamente los mínimos de `pyproject.toml`:
  - núcleo: numpy 1.26.0, pandas 2.2.0, scikit-learn 1.6.0, scipy 1.12.0, optuna 4.0.0, joblib 1.3.0, numba 0.59.0, tqdm 4.66.0 y rich 13.9.0;
  - extras de test: statsmodels 0.13.0, matplotlib 3.7.0, lightgbm 4.0.0, xgboost 2.1.0, catboost 1.2, keras 3.0.0, torch 2.4.0, cloudpickle 3.0.0, skops 0.14.0 y pytest 9.1.0;
  - pyarrow no es dependencia directa (llega con `pandas[parquet]`), así que se instala la última (25.0.1).
- **statsmodels `>=0.13` no se puede cumplir en su mínimo:**
  - 0.13.0 no tiene wheel para ninguna versión de Python que admite skforecast (3.10+) y su compilación desde fuente falla (`No module named 'pkg_resources'`);
  - 0.13.1 tiene wheel para 3.10, pero no importa con pandas 2 (`ImportError: cannot import name 'Int64Index' from 'pandas'`);
  - 0.13.2 es la primera que instala e importa con pandas 2.2.
  - La serie 0.13 no tiene wheels para Python 3.12 o superior (allí se resuelve 0.14, sin problema).
- Con los mínimos y statsmodels 0.13.2 (y también con 0.13.5): `stats`, `plot` y `utils` sin `slow`, **1205 passed**; `preprocessing`, **346 passed y 1 fallo**, `test_QuantileBinner_is_equivalent_to_KBinsDiscretizer`, que pasa `quantile_method='linear'` a `KBinsDiscretizer` (argumento de scikit-learn 1.7; ya visto en §5.6).
- No medido:
  - el resto de la suite (`recursive`, `direct`, `model_selection`, `feature_selection`, `drift_detection`, etc.). No la lancé sin permiso del usuario; la primera ejecución del job lo dirá;
  - `deep_learning` y `foundation` con keras 3.0 y torch 2.4: el sandbox no llega al índice de PyTorch y el wheel de PyPI para Linux es la versión CUDA (varios GB).
- Versión de Python del job: **3.10**. Es la mínima de `requires-python`, y numba 0.59, numpy 1.26, scipy 1.12 y pandas 2.2.0 no tienen wheels para 3.13 o superior. Mínimo de Python con mínimos de dependencias es además la combinación que tiene sentido probar.

**Hallazgo nuevo N-13 (bajo):** `plot/plot.py:26`, `deep_learning/_forecaster_rnn.py:64` y `deep_learning/utils.py:46` capturan cualquier excepción al importar las dependencias opcionales y toman la última palabra del mensaje como nombre del paquete (`str(e).split(" ")[-1]`). Si el paquete está instalado pero falla al importar (por ejemplo, statsmodels 0.13.1 con pandas 2), el usuario recibe `ModuleNotFoundError: No module named '(/ruta/.../python3'`, que esconde el error real. Arreglo propuesto: usar `e.name` solo con `ModuleNotFoundError` y dejar pasar las demás excepciones. No es de este PR salvo que el usuario lo decida (decisión 4).

**Diseño propuesto:**
- Nuevo workflow `.github/workflows/unit-tests-min-deps.yml`, con la misma estructura que `unit-tests-latest-deps.yml`: un job, `ubuntu-latest`, Python 3.10, `timeout-minutes: 30`.
- Instalación: `uv pip install --resolution lowest-direct -e ".[test]"`, con `UV_TORCH_BACKEND=cpu`. Lee los mínimos de `pyproject.toml`, así que no hay una lista de versiones que mantener a mano ni que se desincronice.
- Mostrar el entorno con `uv pip list` (y `skforecast.show_versions()`, que ya enumera las dependencias opcionales desde el PR 6).
- Ejecutar `pytest -q -ra -o verbosity_assertions=2 --exclude-warning-annotations`, como el job de tests (decisión 3 para `deep_learning`).
- Disparadores: ver decisión 1.

**Decisiones del usuario, antes de implementar:**
1. **Cuándo se ejecuta:**
   - (a) en los PRs a `main` y a mano (`workflow_dispatch`), igual que `unit-tests.yml`;
   - (b) además, en los PRs a las ramas de versión (`*.x`) **solo si cambian `pyproject.toml` o el propio workflow** (filtro `paths`). Es cuando cambian los mínimos (como en el PR 6), y el coste es bajo: la mayoría de PRs no tocan esos ficheros. **Recomendado.** Sirve además para ver la primera ejecución en el propio PR, que va contra `0.26.x`;
   - (c) además, cada semana (`schedule`), como el de últimas versiones. Con mínimos fijos, el resultado solo cambia si cambia el código, que ya cubren (a) y (b).
2. **Mínimo de statsmodels:**
   - (a) `>=0.13.2`: el mínimo real que funciona (medido). **Recomendado:** es el cambio más pequeño y cierto;
   - (b) `>=0.14`: simplifica (una serie con wheels para todas las versiones de Python), pero deja fuera a quien use 0.13.x en Python 3.10 o 3.11 sin motivo medido.

   Ficheros: `pyproject.toml` (4 líneas: `stats`, `plotting`, `test` y la de `all`), `optional_dependencies` en `utils.py:68`, `docs/quick-start/how-to-install.md` (2 líneas), `tools/ai/ai_context_header.md:40` y regenerar con `ai-context-sync`. Nota de versión en Changed.
3. **`deep_learning` y `foundation` en el job:**
   - (a) incluirlos: es lo que declara `pyproject.toml` (keras 3.0, torch 2.4), pero no está medido y la instalación de torch alarga el job;
   - (b) excluirlos (`--ignore`), como hace macOS con `deep_learning`.

   **Recomendado:** (a) en la primera ejecución; si falla por keras o torch y el arreglo no es inmediato, pasar a (b) y apuntarlo como hallazgo.
4. **N-13:** incluirlo en este PR (es lo que se ve al instalar mínimos que no funcionan) o dejarlo para el PR 5a (validación). **Recomendado:** PR 5a, para que este PR solo toque CI y versiones.

**Commits previstos** (con las recomendaciones):

| # | Commit | Contenido | Comprobación |
|---|---|---|---|
| 1 | Add a CI job with the minimum supported versions | Workflow nuevo con los disparadores de la decisión 1 | La primera ejecución en el PR muestra los fallos reales |
| 2 | Require statsmodels>=0.13.2 | Los ficheros de la decisión 2; ficheros de contexto de IA regenerados | `test_check_optional_dependency.py` (compara `optional_dependencies` con `pyproject.toml`) y `generate_ai_context_files.py --check` |
| 3 | Run test_QuantileBinner_is_equivalent_to_KBinsDiscretizer with scikit-learn < 1.7 | Pasar `quantile_method='linear'` solo con scikit-learn >= 1.7 (antes es el comportamiento por defecto, así que el test sigue comprobando lo mismo; no se salta) | El test pasa con scikit-learn 1.6.0 y con la última |
| 4 | (los que salgan de la primera ejecución) | Un commit por causa. Si un fallo pide subir otro mínimo, se decide con el usuario antes, como en el PR 6 | |
| 5 | Add release notes for the minimum versions job | Changed: el mínimo de statsmodels (con el motivo). El job de CI no es un cambio para el usuario: sin entrada | |

**Otros cambios del mismo PR:**
- `CLAUDE.md:29` dice que los tests unitarios solo se ejecutan en los PRs a `main`: añadir el job nuevo y sus disparadores.
- El texto de `tools/ai/ai_context_header.md` ("it runs in CI on the pull request of each release to `main`") solo cambia si se elige (b) en la decisión 1; entonces regenerar los ficheros de contexto de IA.

**Riesgos:**
- `--resolution lowest-direct` también baja las dependencias de test (pytest 9.1.0, pytest-cov 7.1.0): es lo que declaran y funcionan, pero si un mínimo de test se rompe, el job fallará por la herramienta y no por skforecast. Se arregla subiendo ese mínimo.
- Un paquete que publique una versión que rompa con un mínimo de otro (como pyarrow, que no se fija) puede romper el job sin cambios en skforecast. Se vería también en `unit-tests.yml`.


#### PR 7: implementación (2026-10-07): rama `ci/minimum-versions-job` publicada, sin PR

**Base:** `0.26.x` avanzó a `13d5015` (#1342, optimización de `fit` en multiserie, y #1359, `plot_prediction_distribution` con índice entero). No tocan `pyproject.toml`, los workflows ni el contexto de IA.

**Decisiones del usuario:**
- Disparadores: **solo los PRs a `main`** (y `workflow_dispatch`), como `unit-tests.yml` («solo quiero que este job se ejecute cuando se haga un merge a main»). Se descartó el disparador de `*.x`, que había implementado con un job previo de `git diff` sobre el merge commit, porque un filtro `paths` solo mira los primeros 300 ficheros y un PR de versión cambia 326 (`pyproject.toml` es el 153).
- statsmodels `>=0.13.2` y N-13 al PR 5a (recomendaciones, sin objeción).
- torch se mantiene en el job (pregunta del usuario por el coste en la capa gratuita): el repo es público (minutos gratis e ilimitados; límite de 20 jobs Linux a la vez, el PR de versión usa 15 + 1). En el último PR de versión (0.25.x, run `35009139439`), el paso de instalación de ubuntu 3.10, con torch, tardó 5 s y los tests 4 min 18 s.
- keras `>=3.3` («Sí»), tras medirlo.

**Consecuencia del disparador:** el job no corre en este PR (va contra `0.26.x`) ni se puede lanzar a mano hasta que el workflow esté en `main` (GitHub solo lanza a mano los workflows de la rama por defecto). Por eso la suite completa con los mínimos se ejecuta en local antes del PR.

| # | Commit | Contenido |
|---|---|---|
| 1 | `a32442f` Require statsmodels>=0.13.2 | `pyproject.toml` (`stats`, `plotting`, `all`, `test`), `optional_dependencies`, `how-to-install.md`, `ai_context_header.md`; contexto IA regenerado |
| 2 | `3d17574` Run the QuantileBinner equivalence test with scikit-learn < 1.7 | `quantile_method='linear'` solo con scikit-learn >= 1.7 (antes, `np.percentile` lineal). Con 1.6.0: 1 fallo antes, 44 passed después; con 1.9.1, 44 passed |
| 3 | `abb6d29` Add a CI job that tests the minimum supported versions | `.github/workflows/unit-tests-min-deps.yml`: PRs a `main` y a mano, ubuntu, Python 3.10, `uv pip install --resolution lowest-direct -e ".[test]"` con `UV_TORCH_BACKEND=cpu`, caché con `cache-suffix: min-deps`, mismo pytest que `unit-tests.yml`. `CLAUDE.md`: el job entra en «unit tests ... only on pull requests to `main`» |
| 4 | `b40ce5a` Require keras>=3.3 | `pyproject.toml` (`deeplearning`, `all`, `test`), `optional_dependencies`, `how-to-install.md`, `ai_context_header.md` (dos líneas); contexto IA regenerado |
| 5 | `b51c1a0` Add release notes for the minimum versions of statsmodels and keras | Dos entradas en Changed, después de la de pandas y scikit-learn (sin tocar su highlight) |

**Medido** (entorno `scratchpad/venv_minci`, la misma instalación que el job; torch 2.4.0 de PyPI con CUDA, porque el entorno bloquea `download-r2.pytorch.org`, ver abajo):
- Con los mínimos de `0.26.x`, la instalación falla en statsmodels 0.13.0 (`No module named 'pkg_resources'` al compilar). Con `>=0.13.2` instala exactamente los mínimos.
- **Hallazgo nuevo N-14 (alto, resuelto en el commit 4):** con keras 3.0.0 a 3.2.1 y torch 2.4.0, un modelo de Keras no se puede copiar (`deepcopy`): `ForecasterRnn(estimator=model)` da `TypeError: cannot pickle 'module' object` (3.1.0: `TypeError: DTypePolicy.__new__() missing 1 required positional argument: 'name'`). Tests de `deep_learning` simulando GitHub Actions: keras 3.0.0, 107 failed, 84 passed y 1 error; keras 3.3.0, **192 passed en 13 s**. Sin medir con el backend de TensorFlow (no instalado).
- `test_fit_tensorflow.py` falla en local (3 tests) porque solo comprueba el backend fuera de GitHub Actions (`GITHUB_ACTIONS != 'true'`); con `GITHUB_ACTIONS=true` pasa. No es de las versiones mínimas.
- Cada commit en su worktree: `test_check_optional_dependency.py` y `test_QuantileBinner.py` (47 passed), `generate_ai_context_files.py --check` y `actionlint` limpios. En el head: ruff limpio; `utils` + `preprocessing`, 1095 passed.
- Red: el entorno cloud bloquea `download-r2.pytorch.org` (wheels de torch para CPU). Para usarlos, el usuario puede añadir el host en Network access del entorno.

**Rama publicada** el 2026-10-07 con el OK del usuario («vamos a hacer todos los commit, publicar la rama y, mientras la reviso en local, lanzas los tests»): 5 commits, head `b51c1a0`. Sin PR.

**Suite completa con los mínimos** (el usuario la pidió; misma instalación y comando que el job, `GITHUB_ACTIONS=true`, `KERAS_BACKEND=torch`): **6590 passed, 1 skipped, 1 failed, 1 error en 6 min**. Primero se paró en la recogida (en CI el job abortaría sin ejecutar ningún test); con `--continue-on-collection-errors`:
- **Error, hallazgo nuevo N-15 (medio):** `drift_detection/tests/tests_population_drift/fixture_results_multiseries.joblib` y `fixture_summary_multiseries.joblib` se guardaron con numpy 2 y no cargan con numpy 1.26.0 (`ModuleNotFoundError: No module named 'numpy._core'`). Son los únicos 2 de los 9 fixtures con el problema. numpy 1.26.1 añadió `numpy._core` para leer pickles de numpy 2: los 9 cargan con 1.26.1 y pandas 2.2.0. Caso de usuario medido: un forecaster guardado con `save_forecaster` y numpy 2.5.3 carga con 1.26.1 (mismas predicciones) y falla con 1.26.0. **Decisión del usuario («adelante»):** subir a `numpy>=1.26.1` (recomendado frente a regenerar los fixtures: el problema volvería con cada fixture nuevo guardado con numpy 2, y la copia de prueba no salía idéntica por las tuplas con NaN).
- **Fallo:** `test_evaluate_grid_hyperparameters_stats_warn_when_non_valid_params` comparaba el aviso entero, cuya segunda parte es el error de statsmodels (0.13.2: `Invalid trend method.`). Ahora compara solo la parte de skforecast. Pasa con statsmodels 0.14.6 y 0.13.2.

**Commits nuevos** (encima de los publicados, sin reescribir historia publicada):

| # | Commit | Contenido |
|---|---|---|
| 6 | `3df318f` Require numpy>=1.26.1 | `pyproject.toml`, `how-to-install.md`, `ai_context_header.md`; contexto IA regenerado |
| 7 | `b620a81` Match only the skforecast part of the skipped parameters warning | `test_evaluate_grid_hyperparameters_stats.py` |
| 8 | `ca74405` Add release note for the minimum version of numpy | Una entrada en Changed, después de las de statsmodels y keras |

**Error mío, corregido antes de subir:** el commit de numpy se hizo con `git add -A` mientras corría la suite y se coló `backtesting.gif` (lo crea y lo borra en la raíz un test de `plot`). Se quitó del commit (que no estaba subido). **Lección:** con tests en marcha, añadir siempre ficheros por ruta, nunca `git add -A`.

**Push de los commits 6 a 8:** GitHub devolvía `Internal Server Error` (500) y después 503 en cada push a `ci/minimum-versions-job`, incluso con un solo commit; crear el PR desde esa rama por la API también daba 500; las lecturas funcionaban. A propuesta del usuario, los 8 commits se subieron a una **rama nueva, `ci/minimum-versions-job-v2`**, que GitHub aceptó a la primera: el fallo estaba ligado a la rama original. La remota `ci/minimum-versions-job` (5 commits, `b51c1a0`) y la local siguen existiendo; no borrarlas sin permiso. Trabajo en local en `ci/minimum-versions-job-v2`.

**PR abierto en borrador** el 2026-10-07 15:17 UTC: skforecast/skforecast#1360 (desde `ci/minimum-versions-job-v2`, head `ca74405`), título *Add a CI job for the minimum supported versions and fix the minimums of numpy, statsmodels and keras*. Pie quitado, sesión suscrita y check-in `trig_01JQxEDtAQbrHQBoFeuTax3B` (16:09 UTC). Cuerpo en `review/pr7_body.md`. Los 3 commits también como parches en `review/pr7_patches/`.

**Suite completa en el head** (`ca74405`, numpy 1.26.1, keras 3.3.0): **6597 passed, 1 skipped** (`test_set_params` del clasificador, requiere scikit-learn >= 1.8), 6 min 29 s; sin ficheros sueltos en el repo. Descripción del PR actualizada con el resultado.

**Rama original:** el usuario pidió borrarla («sí, bórrala»). Comprobado antes que sus commits están en la v2. La local está borrada. La remota no se pudo borrar: el push de borrado devuelve HTTP 403 (política del entorno, no se reintenta); la tiene que borrar el usuario desde GitHub o desde su máquina (`git push origin --delete ci/minimum-versions-job`).

CI del PR en verde en `ca74405` (`check`, CodeQL, Analyze); el job nuevo no corre (PR a `0.26.x`).

**Pendiente:** revisión y merge del usuario; borrar la remota `ci/minimum-versions-job` (usuario).


**Fusionado** el 2026-10-07 15:34 UTC (merge commit `2d102b6`), tras pasarlo el usuario a «ready for review». Incluye un commit del usuario, `31de197` *Compare the scikit-learn version with packaging in test_preprocess_repr*: la comparación como texto (`sklearn.__version__ >= "1.7.0"`) fallaría con scikit-learn 1.10; revisado y correcto. Desuscrito automáticamente y check-in `trig_01JQxEDtAQbrHQBoFeuTax3B` borrado. El job correrá por primera vez en el PR de la versión a `main`.

### 5.10 PR 4: transformaciones, índices y `steps` (2026-10-07): skforecast/skforecast#1361, fusionado (`02ae46b`)

Petición del usuario (2026-10-07): «empieza el PR 4». Rama `fix/transforms-and-index-handling` desde `origin/0.26.x` (`13d5015`, con el #1342). Título previsto: *Fix transformers, time zones and step handling in index utilities*.

**Reproducido sobre `13d5015` antes de implementar** (`scratchpad/pr4/repro.py`, `repro2.py`): los 7 hallazgos siguen. M-11 de punta a punta necesita `Sarimax` (único estimador de `ForecasterStats` con `last_window`). M-10b con `TimeSeriesFold`: 2 en lugar de 25, sin aviso.

| # | Commit | Hallazgo | Contenido y tests |
|---|---|---|---|
| 1 | `86e97b3` Fall back to default feature names when get_feature_names_out raises | A-07 | Nuevo `_get_feature_names_out` (devuelve `None` si el método falla con `AttributeError`, `ValueError` o `TypeError`), usado en `transform_dataframe` (cae a `df.columns`) y `transform_series` (cae a `transformed_i`). Tests: `transform_dataframe` y `transform_series` con `Pipeline(FunctionTransformer)`, y `ForecasterRecursive` con `transformer_y` y `transformer_exog` = `Pipeline(FunctionTransformer(log1p), StandardScaler())`, comparado con la misma Pipeline con `feature_names_out='one-to-one'` |
| 2 | `138e3bb` Return a Series from transform_series for single-row input | M-11 | `values_transformed.iloc[:, 0]` en lugar de `squeeze()`. Tests: unitario de una fila con salida pandas y `ForecasterStats(Sarimax)` con `last_window` de 1 observación (igual que con salida numpy) |
| 3 | `d5ef3ea` Do not mutate transformers in transform_series when the series name differs | B-17 | Se renombra la columna de entrada al nombre visto en fit (sin `deepcopy` ni modificar el transformer); la salida de una columna conserva el nombre de la entrada. Test parametrizado: `StandardScaler` y `Pipeline`, con salida numpy y pandas |
| 4 | `93d7ca5` Support tz-aware indexes and indexes without freq in date_to_index_position | M-10 | Fecha sin zona → zona del índice; con otra zona → `tz_convert`; fecha con zona e índice sin zona → `ValueError`. `validation` → `index.searchsorted(target, side='right')` (no necesita `freq`). `prediction` sin `freq` → `ValueError`. Mensajes con `date_literal` y «both included» (3 tests de mensaje actualizados). Tests: unitarios (zona horaria, sin `freq`, índice irregular), `TimeSeriesFold` y `ForecasterRecursive.predict(steps=<fecha>)` |
| 5 | `c919ed5` Accept numpy integers and reject empty steps in prepare_steps_direct | M-04 | `(int, np.integer)`; `< 1` → `ValueError` (mismo mensaje que `check_predict_input`); lista vacía → `ValueError`; otros tipos (tupla, range, array, float) → `TypeError`. Tests unitarios y de `ForecasterDirect.predict` |
| 6 | `488d0e4` Validate steps against the exog length in exog_to_direct | B-19 | `1 <= steps <= n_rows` en `exog_to_direct` y `exog_to_direct_numpy`. Tests: 0 y más pasos que filas |
| 7 | `2d871ae` Accept pandas Index and arrays as levels in multiseries predict | B-18 | `prepare_levels_multiseries` convierte `pd.Index` y `np.ndarray` en lista de `str` (`tolist()`); sirve a multiserie y `ForecasterRnn`. Otros tipos siguen dando el `TypeError` de `check_predict_input`. Tests: unitario y `ForecasterRecursiveMultiSeries.predict` |
| 8 | `08bd732` Add release notes for transform and index fixes | | Fixed: dos de M-10 junto a la entrada de zonas horarias; A-07, M-11, B-17, M-04, B-18 y B-19 al final. Enlaces nuevos: `transform_series`, `exog_to_direct`, `exog_to_direct_numpy` |

**Mediciones y decisiones:**
- **M-10, equivalencia:** `searchsorted` frente al cálculo anterior con `_date_range_from_index`, en 1915 casos con frecuencia (D, h, MS, W, 15min, horaria con zona horaria, diaria con zona horaria en sus dos convenciones con cambio de hora): 0 diferencias (`pr4/m10_equiv.py`).
- **M-10, alcance:** con zona horaria y fecha sin zona, en la base `backtesting_forecaster` da `TypeError` y `grid_search_forecaster` devuelve una tabla vacía (salta todas las combinaciones); en la rama funcionan (`pr4/tz_backtesting.py`).
- **B-17, cambio de nombres:** comparado con la base (`pr4/b17_compare.py`): idéntico con salida de una columna (numpy y pandas, `inverse_transform`, mismo nombre); la `Pipeline` funciona. **Único cambio:** un transformer que expande en varias columnas (`OneHotEncoder`) aplicado con `fit=False` a una serie con otro nombre: columnas `y_A` en lugar de `pred_A`. Solo se alcanza llamando directamente a `transform_series` (los dos llamadores, `ForecasterStats` y `ForecasterRnn`, usan una columna). Está en la nota de versión. Alternativa si el usuario prefiere no cambiarlo: mantener la modificación del transformer y renombrar solo cuando `feature_names_in_` es de solo lectura.
- **A-07:** `transformer_series` del multiserie ya funcionaba en la base con esa Pipeline (va por otro camino); la nota solo cita `transformer_y` y `transformer_exog`.
- **B-18:** en `ForecasterRnn`, un `Index` daba antes un `TypeError` claro; ahora se acepta (medido en `venv_minci`, con keras 3.3).
- **Hallazgo nuevo N-16 (bajo, fuera del PR):** `ForecasterStats` con `Sarimax` y un `last_window` cuyo nombre no es el del entrenamiento falla, también sin transformer y en la base: `ValueError: Columns must match to concatenate along rows.` (statsmodels, en `Sarimax.append`). Por eso B-17 no se alcanza por `ForecasterStats`. Arreglo posible: renombrar `last_window` al nombre de `y` en `ForecasterStats` antes de `append`. Sin PR asignado.
- Los docstrings de los `predict` siguen diciendo `levels : str, list`; aceptar `Index`/array es un arreglo de robustez (sin cambio en el contexto de IA, `--check` al día).

**Desviaciones del plan:**
- B-18 se arregla en `prepare_levels_multiseries` (convertir a lista) en lugar de `if len(levels) == 0` en `preprocess_levels_self_last_window_multiseries`: el código que sigue espera una lista, y así también lo acepta `ForecasterRnn`.
- M-10, `prediction` sin `freq`: el plan dejaba elegir entre inferirla o dar un error; se da un `ValueError` (los índices de los forecasters siempre tienen `freq`, solo se alcanza llamando directamente a la función).
- M-04: tupla, `range` y array dan `TypeError` (no se aceptan), como decía el plan.
- B-17: se mide y documenta el cambio de nombres con transformers que expanden (ver arriba).

**Verificación (skill `verify`):**
- `ruff check` sobre los `.py` cambiados: sin hallazgos. Líneas nuevas de 88 columnas o menos (salvo nombres de test que ya existían).
- Tests, secuenciales, sin `slow`: `utils` 776 passed; `recursive` 1509 passed, 1 skipped; `direct` 902 passed; `model_selection` 794 passed; `deep_learning` 192 passed (en `venv_minci`, keras 3.3 + torch 2.4).
- Cada commit en su worktree con los ficheros de tests que toca: 54, 42, 14, 133, 59, 35 y 62 passed; el de la nota, sin referencias sin definir; sin ficheros sueltos.
- Tests nuevos contra el commit anterior (o la base): fallan antes y pasan después (A-07 3, M-11 2, B-17 2, M-10 14 contando los 3 mensajes actualizados, M-04 12, B-19 4, B-18 4).
- Con las versiones mínimas (numpy 1.26.1, pandas 2.2.0, scikit-learn 1.6.0): los ficheros de tests tocados, 347 passed.
- Contexto IA `--check`: al día. La rama se fusiona sin conflictos con `0.26.x` (`2d102b6`).
- No ejecutados: `stats`, `preprocessing`, `feature_selection`, `foundation`, `drift_detection` y `plot`. Ninguno llama a las funciones tocadas: todos sus llamadores están en `recursive`, `direct`, `model_selection` y `deep_learning`, que sí se ejecutaron (`foundation` tiene su propia validación de `levels`).

**Revisión final (2026-10-07), a petición del usuario; subido durante la revisión con su OK («Puedes ir subiendo la PR mientras haces la revisión»):**
- Antes del push, integrado en sus commits (sin publicar todavía): una línea en blanco de más al final de 4 ficheros de tests y espacios finales en 2 parametrizaciones; comprobación de que las predicciones de referencia no son NaN en los 3 tests que comparan dos llamadas equivalentes (lección del PR 3b); mensaje del commit 5 corregido (`len()` devuelve un `int` de Python, no de numpy).
- **Push** de `fix/transforms-and-index-handling` (head `c7d7c97`) y **PR en borrador** skforecast/skforecast#1361, *Fix transformers, time zones and step handling in index utilities*. Pie quitado, sesión suscrita y check-in `trig_01RQWssu1cyMywSzdcPDMnZK` (16:42 UTC). Cuerpo en `review/pr4_body.md`.
- Hallado después del push, en commits nuevos: los tests insertados en `test_transform_dataframe.py` (A-07) y `test_transform_series.py` (M-11) tenían 3 líneas en blanco antes y 1 después (`3f71669`); la nota de A-07 ahora nombra qué transformers fallaban (`6f5648f`).
- **A-07, medido en todos los forecasters** (`pr4/a07_matrix.py`, `a07_rnn.py`): fallaba `transformer_exog` en todos (Recursive, Direct, MultiSeries, DirectMultiVariate, Classifier, Stats y Rnn) y `transformer_y` en Recursive y Direct; `transformer_series` y el `transformer_y` de Stats ya funcionaban. Ahora todos funcionan.
- **M-10, casos límite** (`pr4/m10_edges.py`): una fecha sin zona que no existe o es ambigua en hora local da `NonExistentTimeError` o `AmbiguousTimeError` de pandas (antes, `TypeError`); se anota como límite conocido. Un índice desordenado se comporta igual que en la base (`ValueError` de rango).
- Código revisado sin otros problemas: sin código muerto (`_date_range_from_index` y `deepcopy` siguen en uso), sin cambios de comportamiento no buscados salvo el de B-17 (documentado), sin `except` más amplios de lo necesario (`_get_feature_names_out` solo cae a los nombres por defecto).
- Descripción del PR actualizada (transformers afectados, límite conocido, commits de la revisión).


**Fusionado (2026-10-07 16:03 UTC):** skforecast/skforecast#1361, merge commit `02ae46b`, con un commit del usuario (`cf3f332`: docstrings de `levels` en los `predict` de multiserie y `ForecasterRnn` (`pandas Index`, `numpy ndarray`); `except` de `_get_feature_names_out` sin `TypeError`; dos líneas en blanco entre los tests de `transform_dataframe`/`transform_series`, que mi commit `3f71669` había dejado mal). Decisión B-17: se quedó el renombrado (recomendado). Desuscrito y check-in `trig_01RQWssu1cyMywSzdcPDMnZK` borrado. **Lecciones:** al aceptar un tipo nuevo en un argumento público, actualizar su docstring en todos los métodos; comprobar las líneas en blanco entre tests con un script, no a ojo.

### 5.11 PR 5a: validación de entradas (2026-10-07): skforecast/skforecast#1362, fusionado (`9c6f0fd`)

Petición del usuario (2026-10-07): «Me gustaría implementar 5a y 5b». Rama `fix/input-validation`, rebasada sobre `origin/0.26.x` `02ae46b` (después del PR 4; se rebasó porque no estaba publicada). 14 commits. Cuerpo del PR en `review/pr5a_body.md`.

**Reproducido sobre `2d102b6` antes de implementar** (`scratchpad/pr5a/repro.py`, `repro2.py`, `repro_rnn.py` en `venv_minci` con keras 3.3): todos los hallazgos del plan siguen. Novedades al reproducir:
- **B-16 es peor de lo que decía el informe:** con series en `UTC` y `Europe/Madrid`, `predict()` devuelve solo una de las series, sin aviso (la otra termina en otro instante y `preprocess_levels_self_last_window_multiseries` la descarta); con una serie sin zona, `fit` da `TypeError`.
- **S-01 confirmado de punta a punta:** `ForecasterRnn` con un `last_window` sin una serie de entrada da las mismas predicciones que si esa serie fuera otra columna (predicciones erróneas sin aviso).
- **B-21:** solo falla con una Series `exog_val` sin nombre (con nombre, `input_to_frame` no mira el mapeo).
- **S-06 confirmado:** una exog de la misma longitud con otras fechas no se reindexaba; en multiserie salía `Different index for series and exog`, y `FoundationModel` la usaba por posición.
- **N-13:** reproducido con un statsmodels falso que lanza `ImportError` al importarse: `No module named '(/root/'`.

| # | Commit | Hallazgos | Contenido |
|---|---|---|---|
| 1 | `ed6daca` Accept numpy integers and reject booleans in lags and window sizes | B-03, B-20 | `initialize_lags`: `np.integer` sí, `bool` no (también dentro de listas/tuplas), lags a `int64`. `initialize_window_features`: `np.integer` sí, `bool` no, `[]` → `ValueError` propio |
| 2 | `98150eb` Do not mutate user fit_kwargs in check_select_fit_kwargs | B-02, S-02 | Filtra `sample_weight` en la comprensión. Aviso propio cuando `fit` acepta `**kwargs` (Pipeline) |
| 3 | `7dd399e` Accept nullable and pyarrow numeric dtypes in check_exog_dtypes | B-09 | `pd.api.types.is_integer_dtype`/`is_float_dtype`, mismo código para Series y DataFrame. Predicciones con categorías `Int32` y `int64[pyarrow]` idénticas a `int64` (LGBM, HGB). `interval[int64]` ahora avisa |
| 4 | `a3860a2` Handle nullable dtypes when trimming multiseries NaN | M-07 | `pd.isna` en `align_series_and_exog_multiseries`. Predicciones iguales a float64 |
| 5 | `5fb1804` Report mismatched frequencies and time zones clearly in check_preprocess_series | B-01, B-16 | Orden por texto si no se pueden comparar; `ValueError` con las zonas horarias (`None` para sin zona; compara por `str(tz)`, así `'UTC'` y `datetime.timezone.utc` son la misma). Docstring corregido |
| 6 | `c9e3f88` Fix unnamed, duplicated and unordered exog names in multiseries preprocessing | B-14, B-15 | `check_exog` antes de `to_frame()`; columnas duplicadas → `ValueError` (ancha y dict); `exog_names_in_` con `dict.fromkeys`. Al endurecer las comparaciones de orden, un test existente esperaba `exog_2` que no estaba en su exog: corregido |
| 7 | `44185d3` Reindex multiseries exog with the same length and different dates | S-06 | Reindexa si el índice difiere (no solo si la longitud difiere); índice con duplicados → `ValueError` |
| 8 | `abe3268` Check residuals only for the predicted levels | M-08 | Solo `levels`; en multiserie, nivel sin residuos → `'_unknown_level'` (mismo `.get` que al predecir). Mensaje «None or empty». Reescrito el test previsto y 2 de `predict_bootstrapping` (mensaje) |
| 9 | `d626bcc` Check the missing exog columns when exog is a Series | B-11, N-02 | La Series se compara como DataFrame de una columna. Efecto lateral: en multiserie una Series con nombre no visto da además el `MissingExogWarning` |
| 10 | `eb2552e` Check that last_window has all the input series in ForecasterRnn | S-01 | La comprobación de DirectMultiVariate también para Rnn. Test unitario y de `predict` de Rnn |
| 11 | `1589407` Warn about missing values only in the part of last_window that is used | B-13 | Últimas `window_size` filas de los niveles a predecir (multiserie) o de `series_names_in_` (DMV usa `X_train_series_names_in_`, Rnn todas); Stats revisa todo |
| 12 | `8a7e5fe` Fix exog_val handling in ForecasterRnn and stop mutating fit_kwargs | B-21, S-05 | `'exog_val': 'exog'` en `input_to_frame` (nuevo `test_input_to_frame.py`); copia de `fit_kwargs` en `__init__` y `set_fit_kwargs` |
| 13 | `4943af5` Show the real error when an optional dependency fails to import | N-13 | Mismo patrón que `skforecast.stats`: `ModuleNotFoundError` con `name` del paquete → mensaje de instalación; cualquier otro error se relanza. Tests con subproceso (finder que oculta el paquete y paquete falso roto) |
| 14 | `40ed950` Add release notes for input validation fixes | | 12 entradas en Fixed (la de exog de multiserie con 4 sub-viñetas) |

**Verificación (tras el rebase):** `ruff` sin hallazgos nuevos (3 imports sin usar preexistentes en tests). Tests: `utils` 853, `recursive` 1516 + 1 skipped, `direct` 902, `model_selection` 794, `plot` 34, `stats` 451, `preprocessing` 347, `feature_selection` 70, `experimental` 78; `foundation` 858 + 10 fallos y `deep_learning` 196 + 3 fallos, **los mismos en la base** (sin torch en el entorno principal; sin TensorFlow para `test_fit_tensorflow.py`). Cada commit en su worktree con sus tests: todos pasan. Ficheros de test tocados con las versiones mínimas (numpy 1.26.1, pandas 2.2.0, sklearn 1.6.0, keras 3.3): 446 passed. Contexto IA al día.

**Fusionado** el 2026-10-07 a las 19:36 UTC (merge commit `9c6f0fd`), aprobado por JavierEscobarOrtiz. Incluye un commit del usuario, `938a388` (Cast numpy window_sizes to int so unsigned integers do not overflow): `initialize_window_features` guarda `int(window_sizes)` y `int(max(window_sizes))`, porque con `np.uint8` `-window_size` desbordaba igual que con los lags; tests en `test_initialize_window_features.py` y en `predict` de `ForecasterRecursive` (compara con un int). También precisa la nota de M-08 («a level that is not in the residuals dict»). Comprobado: los dos ficheros de test pasan (53) y `ruff` sin hallazgos.

**Fuera del PR (no se tocó):** N-03, N-04 (doble aviso de NaN), N-07, N-08 y N-16. Nuevo N-17 (medio, corregido el 2026-10-07): en multiserie, una exog (ancha o dict) con las fechas en orden descendente se queda vacía al alinear (`.loc[first:last]` por etiqueta) y el modelo se entrena **sin exog** (`exog_in_=False`), solo con avisos. El `KeyError: 'e'` de la primera reproducción venía del propio script (accedía a `X['e']`).

### 5.12 PR 5b: rendimiento y docstrings (2026-10-07): skforecast/skforecast#1363, fusionado (`18e5dbaf0`)

Rama `perf/utils-hot-paths` **apilada sobre `fix/input-validation`** (los dos tocan `check_predict_input`); el PR se abriría contra `0.26.x` y mostrará los commits del 5a hasta que este se fusione. 7 commits. Cuerpo en `review/pr5b_body.md`. Mediciones: mínimo de 2-3 ejecuciones alternadas base/cambio (una sola ejecución variaba hasta un 20 %).

| # | Commit | Hallazgo | Medido |
|---|---|---|---|
| 1 | `fce2439` Make deepcopy_forecaster exception-safe using the deepcopy memo | B-04, O-07 | Copias idénticas en 7 forecasters × 3 combinaciones de flags; el original intacto si falla la copia (antes quedaba sin estimador entrenado, residuos ni `last_window_`). Rnn: 156 → 72 ms por copia (modelo de 126k parámetros). Docstring corregido (§3) |
| 2 | `05341f2` Speed up check_predict_input | O-05 | `pd.isna` sobre numpy. `check_predict_input` 116 → 70 µs; `predict` 384 → 333 µs con exog, 200 → 174 sin exog; multiserie 50 series con exog 6.7 → 5.9 ms. **Descartado tras medir:** `searchsorted` en lugar de `isin` (N-05): la comprobación baja 77 → 59 ms, pero `predict` no cambia (la exog se reindexa después) |
| 3 | `1c7fb3c` Speed up last window preparation in multiseries predict | O-04 | `predict(24)`: 50 series 2.6 → 1.8 ms; 500, 17.6 → 7.7; 5000, 349 → 71 ms |
| 4 | `cdfd42b` Use corrwith in multivariate_time_series_corr | O-06 | pearson 22 → 19 ms, spearman 327 → 98, kendall 769 → 87. `lags` `np.integer` arreglado. Docstring de `lags` (§3) |
| 5 | `5b3f3d2` Add ExtraTrees to the per-tree fast predict path | O-03 | `predict(100)` con `ExtraTreesRegressor(100)`: 508 → 33 ms, mismas predicciones (también con NaN) |
| 6 | `f3090b3` Fix docstrings in skforecast.utils | §3 | `check_exog` (invertido), `check_y`, `initialize_differentiator_multiseries`, `check_predict_input`, `align_series_and_exog_multiseries`, `exog_to_direct_numpy`. Ya estaban hechos: `set_cpu_gpu_device`, `show_versions`, comentario de NaN, «ints.Got», matplotlib |
| 7 | `5867a2d` Add release notes for performance changes | | Highlight Enhancement; 5 entradas en Changed (junto al `fit` multiserie); 2 en Fixed; enlace nuevo `multivariate_time_series_corr` |

**Verificación:** `ruff` sin hallazgos nuevos (import sin usar preexistente en `test_deepcopy_forecaster.py`). Tests: `utils` 880, `recursive` 1516 + 1 skipped, `direct` 902, `model_selection` 794, `feature_selection` 70, `plot` 34, `deep_learning` 196 + los 3 de TensorFlow de la base. Cada commit en su worktree: pasa. Ficheros de test tocados con las versiones mínimas: 191 passed. Contexto IA al día.

**Diferencias con el plan:** el beneficio de O-05 es menor que en el informe (×1.6 en la comprobación, -13 % en `predict`): `expand_index` no se puede quitar y el backtesting sin reentrenar ya no es el caso dominante. O-03 da ×15 (informe: ×15). O-04 coincide (×4.9 con 5000 series).

#### Revisión final de 5a y 5b (2026-10-07), a petición del usuario

Correcciones, integradas con fixup en sus commits (nada estaba publicado; el 5b se rebasó sobre el 5a corregido):
- `ForecasterRnn.set_fit_kwargs` copiaba `fit_kwargs` sin comprobar el tipo: con un argumento que no es dict daba `AttributeError` en lugar del `TypeError` de `check_select_fit_kwargs`. Ahora copia solo si es dict, como `__init__` (comprobado: `set_fit_kwargs('x')` da el `TypeError`).
- La nota de B-09 contaba la implementación (`is_integer_dtype`); quitada.
- El highlight del 5b decía que la validación de entradas tarda un 40 % menos en general; solo está medido para `ForecasterRecursive` con exog. Quitado del highlight (la entrada de Changed da las cifras con su caso).
- El test nuevo de `multivariate_time_series_corr` tenía 3 líneas en blanco antes (una línea con espacios al final del fichero de la base).

Comprobaciones sin hallazgos:
- **`check_exog_dtypes` frente a la base**, 23 dtypes × Series/DataFrame: solo cambian `Int32` categórico (aceptado), `UInt8` y `double[pyarrow]` (sin aviso) e `interval` (ahora avisa).
- **`preprocess_levels_self_last_window_multiseries` frente a la base:** 0 diferencias en 400 configuraciones aleatorias (longitudes y finales de ventana, int/float, `RangeIndex`/`DatetimeIndex`, niveles desordenados o no guardados, `input_levels_is_list`).
- **B-13 en `ForecasterDirectMultiVariate`:** `predict` solo usa las columnas de `X_train_series_names_in_` (`last_window.iloc[:, get_indexer(X_train_series_names_in_)]`), las mismas que ahora se comprueban; la serie objetivo sin lags no se usa.
- **Llamadores de `check_residuals_input`:** los tres multiserie pasan `levels` (Rnn pasa `self.levels`, igual que antes).
- **`deepcopy_forecaster` con `memo`:** las claves son ids de objetos vivos del forecaster y los valores se guardan en el propio `memo`, así que no hay colisiones ni objetos liberados durante la copia.
- Los 21 commits pasan sus tests en su worktree; ruff solo da los 4 imports sin usar de la base; notas sin referencias sin definir; contexto IA al día.

Riesgos y decisiones para el usuario (en las descripciones de los PRs):
- Cambios de comportamiento: errores nuevos con booleanos en `lags`, columnas duplicadas en la exog multiserie y series con zonas horarias distintas; un aviso más (`MissingExogWarning`) en multiserie con una Series exog de nombre no visto.
- O-05 es algo más lento con exog de tipos mezclados (categóricas): unos 6 µs por llamada.
- Los tests de N-13 usan subprocesos y un *finder* que oculta el paquete; el helper está repetido en `plot` y `deep_learning` (paquetes de tests distintos).
- **N-18 (bajo, nuevo, fuera de los PRs):** en `ForecasterStats`, un `last_window_exog` Series con nombre válido cuando se entrenó con más exógenas falla en statsmodels (`Columns must match to concatenate along rows.`), de la familia de N-16.

#### Después de abrir el PR (2026-10-07)

- Commit del usuario `5e08b2f` (Remove unused import and document that check_predict_input accepts levels as str): quita el `deepcopy` sin usar de `test_deepcopy_forecaster.py` (36 passed) y vuelve a `levels : str, list` en el docstring de `check_predict_input`.
- Tras el merge del 5a (`9c6f0fd`, que trae además `899b783` de #1367, `ndiffs` de stats), el 5b **combina sin conflictos** con `0.26.x`; su diff se reduce a 7 ficheros. Probado el resultado del merge: `utils` 885 passed (sin `slow`), `predict` de Recursive y MultiSeries y `predict_bootstrapping` de MultiSeries 128 passed; nota de versión sin referencias sin definir ni marcas de conflicto. No hace falta traer el commit `938a388` del 5a a la rama del 5b.

#### Revisión de rendimiento de 5a y 5b (2026-10-07), a petición del usuario

Pregunta del usuario: «¿Alguno de los cambios implementados produce que algunos de los métodos o funciones se vuelvan mucho más lentos?». Medido `0.26.x` (`02ae46b`), 5a (`40ed950`) y 5b (`5867a2d`) en worktrees separados, mínimo de 3 ejecuciones alternadas, 4 núcleos, pandas 2.3.3, con entradas elegidas para perjudicar a los cambios. Scripts en `scratchpad/perfq/` (`bench.py`, `bench2.py`, `trees_micro.py`).

**Más lento, con arreglo barato (commit de O-05, `05341f2`):** `pd.isna(df.to_numpy())` convierte a array `object` los DataFrames con tipos nullable (`Float64`, `Int64`) o pyarrow (`DataFrame.to_numpy()` con varias columnas de extensión → `object`).
1. **`exog` largo con tipos nullable o pyarrow:** la comprobación de NaN, 0,09 → 23 ms con 100.000 filas (×250); `predict(24)` 13 → 29 ms con 100.000 filas, 1,8 → 3,8 ms con 10.000, +5 % con 24. Con columnas categóricas, +27 % con 100.000 filas.
2. **`last_window` largo con tipos nullable pasado por el usuario** (MultiSeries, DMV, Rnn): se convierte la tabla entera antes de cortar las últimas `window_size` filas. MultiSeries con 50.000×50 `Float64`: 5,5 → 116 ms (×20).

Arreglo probado (variante `v3`, worktree `perfq/wt_v3`, 166 tests de `check_predict_input` y `predict` multiserie pasan): cortar las filas antes de `to_numpy` solo si hay más de `window_size` (`len(last_window) > window_size`, `iloc` antes de seleccionar columnas) y volver a `exog_to_check.isna().to_numpy().any()` en `exog`.

| Caso | `0.26.x` | 5a | 5b | 5b + arreglo |
|---|---|---|---|---|
| `check_predict_input`, exog 24 filas `float64` | 112 µs | 136 µs | 80 µs | 98 µs |
| MultiSeries `predict(24)`, 5000 series | 379 ms | 385 ms | 83 ms | 87 ms |
| MultiSeries `predict(24)`, 50 series con exog | 3,8 ms | 3,9 ms | 2,9 ms | 3,0 ms |
| `predict(24)`, exog `Float64` 100.000 filas | 13,3 ms | 13,3 ms | 29,3 ms | 13,4 ms |
| MultiSeries, `last_window` `Float64` 50.000×50 | 5,5 ms | 5,9 ms | 116 ms | 7,0 ms |

El 5a solo hace `check_predict_input` algo más lento (112 → 136 µs: `iloc` y selección de columnas de B-13); con el 5b y el arreglo queda en 98.

**Más lento, se propone dejarlo:**
3. **`ExtraTreesRegressor` con `n_jobs=-1` y muchas filas por llamada** (O-03): el camino por árbol es de un hilo y scikit-learn reparte entre núcleos. MultiSeries 500 series `predict(24)`: 1,1 → 1,4 s; micro con árboles profundos y 5000 filas: 139 → 464 ms (×3,3), más con más núcleos. Pero `ForecasterRecursive.predict(24)` con `n_jobs=-1`: 830 → 10 ms; `predict_bootstrapping(n_boot=500)`: 910 → 80 ms; con `n_jobs=None` siempre igual o más rápido (scikit-learn tarda ~30 ms por llamada en arrancar los hilos). `RandomForestRegressor` tiene el mismo compromiso desde la 0.24.0 (`45f0657`). Elegir por `n_jobs` y filas exigiría un umbral dependiente de la máquina y cambiaría también RandomForest: aparte, si se quiere.
4. **`multivariate_time_series_corr` con datos muy pequeños** (O-06): `corrwith` llama a scipy por lag. 100 filas y 5 lags: spearman 6,8 → 14 ms, pearson 4,2 → 5,5 ms. Con datos reales, más rápido (spearman 2000×24: 512 → 128 ms; 100.000×3: 970 → 515 ms). Función de análisis que se llama una vez.

**Sin cambios de velocidad:** `deepcopy_forecaster` (algo más rápido), `check_exog_dtypes` con 2000 columnas, `fit` multiserie con exog dict (500×2000 y 5000×200), `check_preprocess_exog_multiseries`, `predict` con `last_window` `float64` largo.

**Correcciones de la descripción del PR:** pearson de `multivariate_time_series_corr` no mejora en esta medición (23,5 → 23,3 ms; decía 22 → 19); «(dtypes kept)» en el último `last_window` multiserie solo vale para tipos numpy: las series nullable dan columnas `float64` (antes `Float64`), con predicciones idénticas.

**Fusionado** el 2026-10-07 a las 20:24 UTC por JavierEscobarOrtiz (merge `18e5dbaf0`), sin el arreglo. La regresión (N-19) queda como tarea T1 del documento de traspaso `scratchpad/handoff/handoff_utils_pendiente.md`, junto con todo lo pendiente (PR 2b, N-04, N-07, N-08, N-16, N-17, N-18), con las reproducciones comprobadas sobre `18e5dbaf0`.
