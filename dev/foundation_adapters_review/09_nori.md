# 09. Nori (synthefy-nori)

- **Rama propuesta**: `fix/nori-adapter-alignment`, creada desde `origin/0.27.x`.
- **Prioridad**: media. El adaptador funciona con la última versión del backend; no hay nada roto que corregir con urgencia [probado].
- **Adaptador**: `NoriAdapter` (`skforecast/foundation/_adapters.py`).
- **Ficheros afectados**:
  - `skforecast/foundation/_adapters.py` (`NoriAdapter`: `predict`, `_load_model`, `_to_numpy`, atributo `backend_package` y docstrings).
  - `skforecast/foundation/_foundation_model.py` (solo el docstring de `FoundationModel`: lista de `model_id` de Nori).
  - `skforecast/foundation/tests/tests_foundation_models/test_NoriAdapter.py`, `fixtures_adapters.py` (`FakeNoriRegressor`) y `test_get_model_info.py`.
  - `tools/foundation_compat/adapters.toml` (entrada `[nori]`, solo el valor de `backend_package`).
  - `skills/foundation-forecasting/SKILL.md`, `skills/foundation-forecasting/references/adapter-parameters.md`, `docs/user_guides/foundation-forecasting-models.ipynb` (solo el texto de la celda de la tabla de modelos), `docs/releases/releases.md`.
- **Depende de**: `02_common.md`, solo en parte. La tabla central de ids no soportados (NORI-P1) y la comprobación central de contexto todo NaN (D11) viven allí. La opción (c) de D24 usa el helper opcional de comprobación de versión de 02 (si se adopta D-VER (b)). NORI-P2, NORI-P4, NORI-P5, NORI-P8 y NORI-P3 con D24 (a) no dependen de 02: el PR se puede fusionar antes.
- **Bloquea a**: `10_docs_ci.md`. La entrada de Nori-1-5 en `adapters.toml` y en la matriz del workflow (NORI-P7) debe usar el mismo `backend_package` que fije este PR (NORI-P3); los smoke de orden de cuantiles y de NaN en la exog del contexto se añaden allí. La guía de usuario se reejecuta allí una sola vez.
- **Cierra issue**: ninguno de skforecast.

## 1. Estado en la revisión (2026-11-06)

Base: rama creada desde `origin/0.27.x` (skforecast 0.27.0 en desarrollo), commit `d9b8f4f`.

**Backend** [documentado, salvo donde se indica]:

| Concepto | Valor |
|---|---|
| Versión con la que se escribió el adaptador | synthefy-nori 0.18.2 [inferido por fechas, el historial git local es superficial: `NoriAdapter` salió en skforecast 0.24.0 el 2026-08-24; 0.18.2 es del 2026-08-20 y era la última en PyPI ese día; 0.19.0 es del 2026-08-25] |
| Versión probada en el CI del 2026-11-06 | synthefy-nori 0.23.0 [probado] |
| Última versión en PyPI | 0.23.0 (2026-11-04). Tag upstream `synthefy-nori-v0.23.0` (f88b73f) [verificado en código] |
| Primera versión con Nori 1.5 | 0.22.0: commit 46224d9 (2026-10-20), "Release Nori 1.5 as synthefy-nori 0.22.0 (#185)" [verificado en código, `evidence/nori/upstream_checks.log`] |
| Requisitos de 0.23.0 | `requires_python >=3.9`; `torch>=2.8,<2.14`; `numpy>=2.0`; `pandas>=2.0`; `scikit-learn>=1.4`; `scipy>=1.13`; `numba>=0.60`; `huggingface-hub>=1.0,<2` (Python>=3.10; `>=0.36,<2` en Python<3.10); `synthefy>=7,<9`. Extra `forecasting` (o `timeseries`): `synthefy[forecasting]` |
| Licencia del paquete | Apache-2.0 |

Evidencia: `evidence/nori/pypi_info.log`, `evidence/nori/verification/pypi_info.log`, `evidence/nori/upstream_checks.log`.

**Checkpoints en el Hub** [probado: `hub_info.py` ejecutado en la revisión y en la verificación; `config.json` leídos en la verificación]:

| Repo | sha | Creado / modificado | Tamaño | Licencia, gated | Notas |
|---|---|---|---|---|---|
| `Synthefy/Nori` | 1aca005 | 2026-05-24 / 2026-08-24 | 48.2 MB | apache-2.0, no | 6M (5868069 parámetros), `regression_head` `999-quantile pinball`; checkpoint por defecto del adaptador (`default_model_id`) |
| `Synthefy/Nori-30M` | 63c9f7f | 2026-07-13 / 2026-08-24 | 117.9 MB | apache-2.0, no | 29206833 parámetros, cabeza pinball; no descargado, solo enrutado y licencia |
| `Synthefy/Nori-100M` | 467ae97 | 2026-08-18 / 2026-08-24 | 395.6 MB | apache-2.0, no | 98304698 parámetros, cabeza pinball; no descargado, solo enrutado y licencia |
| `Synthefy/Nori-1-5` | 79afc5a | 2026-10-20 / 2026-10-20 | 83.2 MB | apache-2.0, no | nuevo; 20757832 parámetros, "Model v3: two-phase encoder", cabeza pinball, `min_synthefy_nori_version` 0.22.0 |
| `Synthefy/Nori-Causal` | afdc891 | 2026-10-25 / 2026-10-25 | 121.5 MB | apache-2.0, no | nuevo; `model_type` `noricausal`, salida "(T, T) edge probabilities", fichero `noricausal_31m.pt` (no hay `nori.pt`) |

Evidencia: `evidence/nori/hub_info.log`, `evidence/nori/verification/hub_info.log`, `evidence/nori/verification/hub_configs.log`.

**Entorno de la revisión** [probado]: Python 3.12.3, synthefy-nori 0.23.0, synthefy 8.0.0, torch 2.13.0+cpu, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1, scipy 1.18.1, huggingface-hub 1.33.0, skforecast 0.27.0 editable; Intel Xeon 2.10 GHz, 4 CPU, sin GPU. Versiones exactas en `evidence/nori/requirements-freeze.txt` y `evidence/nori/env.log`. Entornos auxiliares temporales (borrados) con synthefy-nori 0.21.1 y 0.18.2. Código upstream leído en los tags `synthefy-nori-v0.18.2` (ccd166b) y `synthefy-nori-v0.23.0` (f88b73f), más tags antiguos (v0.11.0, v0.15.0, v0.16.0, 0.17.0, 0.18.x, 0.21.1).

**Línea base del CI** [probado]: workflow Foundation Backends Compatibility, run 42212742430, job 127875169149 (rama 0.27.x, 2026-11-06): synthefy-nori 0.23.0, torch 2.13.0+cpu, numpy 2.5.4, pandas 2.3.3, huggingface_hub 1.33.0; contrato 4 passed, smoke 6 passed, `held_back` false. En local: contrato 4, smoke 6 (pesos reales) y 43 tests unitarios de `test_NoriAdapter.py` en verde (`evidence/nori/contract.log`, `evidence/nori/smoke.log`, `evidence/nori/unit_tests.log`). `python tools/foundation_compat/run.py --adapter nori` no se ejecutó en la revisión (el job del mismo día instaló las mismas versiones).

### Antes de empezar

Este plan puede quedar desfasado. Antes de tocar código, desde la raíz del repositorio:

```bash
# Última versión y requisitos del backend
python dev/foundation_adapters_review/evidence/common/pypi_info.py synthefy-nori
# Checkpoints de Synthefy cuyo nombre empieza por "Nori" (sha, fechas, licencia, gated, tamaño)
python dev/foundation_adapters_review/evidence/common/hub_info.py Synthefy --prefix Nori
# Versión mínima que declara Nori-1-5
python -c 'import json, urllib.request; print(json.load(urllib.request.urlopen("https://huggingface.co/Synthefy/Nori-1-5/resolve/main/config.json")).get("min_synthefy_nori_version"))'
# Recrear el entorno de la revisión con la última versión del backend
bash dev/foundation_adapters_review/evidence/common/make_env.sh nori "synthefy-nori" 3.12
~/.venvs/fm-review-nori/bin/python -m pytest tools/foundation_compat/tests/test_contract.py --backend nori -q -p no:cacheprovider
~/.venvs/fm-review-nori/bin/python -m pytest tools/foundation_compat/tests/test_smoke.py --backend nori -q -p no:cacheprovider
~/.venvs/fm-review-nori/bin/python dev/foundation_adapters_review/evidence/nori/compare_direct.py
```

Esperado hoy: 0.23.0, los cinco repos de la tabla, `0.22.0`, contrato 4 passed, smoke 6 passed y diferencias 0.0 en `compare_direct.py` [probado].

Si aparece una versión posterior a 0.23.0:

1. Leer su CHANGELOG o sus notas de versión buscando cambios en `NoriRegressor.__init__`, `fit`, `predict` (`output_type`, `quantiles`), en el tratamiento de NaN en `X` y en `y`, en el registro `NORI_MODELS` de `hf.py` y en la salida (orden de las filas de cuantiles, tipo numpy).
2. Repetir contrato, smoke, `compare_direct.py`, `edge_cases.py` y `verification/verify_nori.py` (sección 4). Si `compare_direct.py` deja de dar 0.0 o "unsorted levels keep order" deja de ser `True`, NORI-P5 no se puede aplicar tal cual; si la sección D de `verify_nori.py` deja de dar predicciones finitas con NaN en `X`, NORI-P2 no se puede aplicar tal cual.
3. Si el rango de torch cambia (hoy `<2.14`), anotarlo en la descripción del PR.

Si aparece un checkpoint nuevo bajo `Synthefy/Nori*`: leer su `config.json` (`model_type`, `regression_head`, `min_synthefy_nori_version`). Si no es un regresor, añadirlo a la tabla central de ids no soportados de `02_common.md`; si exige otra versión mínima, actualizar el valor de NORI-P3; si su cabeza no es de cuantiles, revisar NORI-P4 (la equivalencia de `'mode'` y `'median'` dejaría de ser general).

## 2. Hallazgos

- **NORI-F1** [probado] Con synthefy-nori 0.23.0 pasan el contrato (4), el smoke con pesos reales (6) y los 43 tests unitarios del adaptador; la verificación los reejecutó con el mismo resultado y leyó el job de CI 127875169149 (contrato 4, smoke 6, `held_back` false). Evidencia: `evidence/nori/contract.log`, `evidence/nori/smoke.log`, `evidence/nori/unit_tests.log`, `evidence/nori/verification/contract.log`, `evidence/nori/verification/smoke.log`, `evidence/nori/verification/unit_tests.log`.
- **NORI-F2** [verificado en código] La API que usa el adaptador no ha cambiado entre 0.18.2 (ccd166b) y 0.23.0 (f88b73f): `NoriRegressor(model=..., **nori_config)`, `fit(X, y)`, `predict(X, *, output_type, quantiles)`, con los mismos argumentos posicionales. Solo hay kwargs nuevos (`encode_id_cols`, `large_context_holdout`, `large_context_max_calls`, `multi_target_prediction_strategy` y `multi_target_prediction_policy` en `__init__`; `multi_target_prediction_policy` en `predict`) y ningún valor por defecto del constructor cambia. `model` acepta un nombre de registro o un repo id (`resolve_model_repo` devuelve `NORI_MODELS.get(model, model)` en `hf.py`). Evidencia: `evidence/nori/upstream_checks.log`.
- **NORI-F3** [probado] La salida del adaptador a través de `FoundationModel` coincide con `NoriRegressor` llamado directamente sobre una matriz de features construida de forma independiente (diferencia máxima 0.0) en `point_estimate` `'mean'`, `'median'` y `'mode'` y con cuantiles desordenados (`[0.9, 0.1, 0.5]` en la revisión, `[0.8, 0.2, 0.5, 0.05]` en la verificación), para `Synthefy/Nori` y `Synthefy/Nori-1-5`, con dos conjuntos de datos distintos (horario con 24 pasos; diario con dos covariables, contexto 256 y 12 pasos). Nombre de registro (`nori-6m`, `nori-1-5`) frente a repo id: 0.0; dos llamadas seguidas: 0.0. Matiz [verificado en código]: con el repo id crudo, `get_expected_inference_config_path` devuelve None y el backend no valida la configuración de inferencia esperada, pero Nori-1-5 guarda la suya en el checkpoint (`model_v3_inference.json`) y se usa igual. Evidencia: `evidence/nori/compare_direct.log`, `evidence/nori/verification/verify_nori.log`.
- **NORI-F4** [probado] `Synthefy/Nori-1-5` (sha 79afc5a, 2026-10-20, 83.2 MB, apache-2.0, no gated) se enruta a `NoriAdapter` y funciona con 0.23.0. Su `config.json` declara `min_synthefy_nori_version` 0.22.0; con 0.21.1 falla con un TypeError opaco (`get_x_encoder() got an unexpected keyword argument 'prompt_quantile_ple_bins'`), mientras `Synthefy/Nori` sí funciona. `prompt_quantile_ple_bins` aparece por primera vez en `encoders.py` de v0.22.0 y ningún código de 0.23.0 lee `min_synthefy_nori_version` [verificado en código]. `backend_package = "synthefy-nori"` no fija mínimo. Evidencia: `evidence/nori/old_backend_nori15.log`, `evidence/nori/verification/old_backend_check.log`, `evidence/nori/verification/hub_configs.log`.
- **NORI-F5** [probado] `Synthefy/Nori-Causal` (descubrimiento causal, `model_type` `noricausal`, salida "(T, T) edge probabilities") cae en el prefijo `Synthefy/Nori` de `_resolve_adapter` (comparación con `startswith`); `FoundationModel` se construye sin error y `predict` falla con `RemoteEntryNotFoundError` 404 porque el repo no tiene `nori.pt` (su fichero es `noricausal_31m.pt`). Evidencia: `evidence/nori/compare_direct.log` (secciones 1 y 5), `evidence/nori/verification/verify_nori.log` (secciones A y A2), `evidence/nori/verification/hub_configs.log`.
- **NORI-F6** [probado] Los cinco repos `Synthefy/Nori*` son apache-2.0 y no gated, PyPI declara Apache-2.0 y el LICENSE del repo de código es Apache 2.0; la entrada `"Synthefy/Nori"` de `_MODEL_LICENSES` (`("Apache-2.0", None, False, None)`, aprox. línea 321 de `skforecast/foundation/_utils.py` en d9b8f4f) cubre por prefijo todos los regresores, igual que `requires_hf_auth=False` y `requires_provider_auth=False`. Evidencia: `evidence/nori/hub_info.log`, `evidence/nori/verification/hub_info.log`.
- **NORI-F7** [probado] Revisiones del Hub: `Synthefy/Nori` 1aca005, `Nori-30M` 63c9f7f, `Nori-100M` 467ae97 (los tres modificados el 2026-08-24), `Nori-1-5` 79afc5a (2026-10-20). El backend descarga siempre la revisión `main`: `_resolve_model_path` llama a `download_checkpoint(model=, token=)` sin `revision`, y `NoriRegressor` no tiene argumento de revisión, así que tampoco se puede fijar por `nori_config` [verificado en código, `api.py` v0.23.0]. Un cambio de pesos en `main` llegaría sin aviso, como en el resto de adaptadores. Evidencia: `evidence/nori/hub_info.log`, `evidence/nori/verification/hub_info.log`.
- **NORI-F8** [probado] `point_estimate='mode'` da exactamente lo mismo que `'median'` en todos los checkpoints publicados (ejecutado en 6M y en 1.5): los cuatro regresores declaran `regression_head` `999-quantile pinball`, y en `api.py` v0.23.0 `output_type="mode"` llama a `_predict_point` con `quantile_collapse="median"`, con un comentario que dice que una cabeza de cuantiles no tiene moda nativa [verificado en código]. Los docs lo presentan como una opción distinta. Evidencia: `evidence/nori/compare_direct.log`, `evidence/nori/edge_cases.log`, `evidence/nori/verification/verify_nori.log`, `evidence/nori/verification/hub_configs.log`.
- **NORI-F9** [verificado en código] (corregido por la verificación) El backend devuelve las filas de cuantiles en el orden pedido y acepta duplicados: el bucle `for i, level in enumerate(q_levels)` de `_predict_distribution` existe desde v0.11.0 (no "desde v0.16.0") hasta v0.23.0. Ejecutado en 0.23.0 [probado]: `[0.9, 0.1, 0.5]`, `[0.8, 0.2, 0.5, 0.05]` y los duplicados `[0.5, 0.9, 0.5, 0.1]` vuelven en el orden pedido. La ordenación, deduplicación y reindexado de `NoriAdapter.predict` (y su comentario "Nori does not guarantee that the output column order matches...") son redundantes. Evidencia: `evidence/nori/upstream_checks.log`, `evidence/nori/compare_direct.log`, `evidence/nori/edge_cases.log`, `evidence/nori/verification/verify_nori.log`.
- **NORI-F10** [verificado en código] La rama torch de `NoriAdapter._to_numpy` es inalcanzable: `_predict_point` y `_predict_distribution` hacen `detach().cpu().numpy()` y convierten a numpy float64 dentro del backend al menos desde v0.15.0. Salida real [probado]: `ndarray float64` de forma `(24,)` en punto y `(3, 24)` en cuantiles (revisión), `(12,)` y `(4, 12)` (verificación). Solo la ejercita `test_NoriAdapter_predict_handles_device_tensor_output`. Evidencia: `evidence/nori/upstream_checks.log`, `evidence/nori/compare_direct.log`, `evidence/nori/verification/verify_nori.log`.
- **NORI-F11** [probado] El backend rechaza NaN solo en `y` (0.23.0: `ValueError: y must contain only finite target values`; 0.18.2: `Input y contains NaN.`) y acepta NaN en `X`, en las filas de contexto y en las de consulta, con predicciones finitas (también en 0.18.2). El README upstream lo recomienda ("Leave missing values as NaN") [documentado]. El adaptador descarta además las filas de contexto con alguna covariable NaN: 188 de 256 filas pasadas a `fit` (68 con NaN en `x1`) en la verificación, 150 de 200 (25 % de NaN) en la revisión. Filtrando solo NaN en `y`, se pasan 256 de 256 (200 de 200) y la salida coincide con la llamada directa sobre la matriz con NaN (diferencia 0.0). El docstring del atributo `supports_nan_in_series` de `NoriAdapter` dice que `NoriRegressor` rechaza NaN, lo que es inexacto. Evidencia: `evidence/nori/nan_covariates_check.log`, `evidence/nori/compare_direct.log` (sección 4), `evidence/nori/verification/verify_nori.log` (sección D), `evidence/nori/verification/old_backend_nan_x_0.18.2.log`.
- **NORI-F12** [verificado en código] Desde 0.18.1 (commit 00def60, 2026-08-17, "Unify categorical DataFrame preprocessing", primer tag `synthefy-nori-v0.18.1`) `NoriRegressor` acepta DataFrames con columnas no numéricas (`categorical_columns="auto"`, codificación ordinal ajustada con las filas de contexto, niveles no vistos al código "other", NaN conservado); el adaptador convierte todo a float32 con `_to_float_array` y lanza ValueError con columnas no numéricas (`supports_categorical_covariates=False` es veraz para el adaptador). Con covariables solo numéricas (con NaN), matriz numpy y DataFrame dan exactamente lo mismo (0.0 en punto y cuantiles, para 6M y 1.5) y una columna de texto con un nivel no visto `'z'` en la consulta da predicciones finitas [probado]. Evidencia: `evidence/nori/verification/dataframe_numeric_check.log`, `evidence/nori/edge_cases.log`.
- **NORI-F13** [verificado en código] Con `device=None` el backend elige CUDA, luego MPS y luego CPU (`_default_device` y `_as_device` en `api.py` v0.23.0; MPS automático desde el commit d57faa6); un dispositivo pedido y no disponible lanza un RuntimeError claro [probado en CPU con `'cuda'` y `'mps'`]. El adaptador no usa `_resolve_torch_device` y expone el dispositivo solo vía `nori_config`; es coherente con el backend. CUDA y MPS solo se han leído en código. Evidencia: `evidence/nori/edge_cases.log`.
- **NORI-F14** [probado] En CPU el backend imprime con `print` (no con un warning) "Mixed precision is not supported for CPU inference, so it has been automatically disabled" en cada carga nueva del modelo (`predictor.py:321` en v0.23.0 [verificado en código]); al ser un `print`, `suppress_warnings` no lo silencia [inferido]. La búsqueda de issues upstream sobre "mixed precision" no devolvió ninguno. Evidencia: `evidence/nori/verification/verify_nori.log` (sección E), `evidence/nori/edge_cases.log`, `evidence/nori/upstream_checks.log`.
- **NORI-F15** [verificado en código] El adaptador cumple `.claude/rules/foundation.md` y `_AdapterBase`: el bucle por serie de `NoriAdapter.predict` es inherente al backend (un `fit` por tabla, sin API por lotes entre tablas; el `NoriTSForecaster` upstream también itera por `item_id`, `nori_ts/core.py:267`); no hay agrupación ni relleno por adaptador; los atributos de capacidad son veraces y `predict` devuelve `(steps, n_quantiles)`, también con `steps=1` [probado, `evidence/nori/edge_cases.log`]. La featurización (unas 150 líneas: `_featurize`, `_timestamps`, `_known_future_columns`, `_to_float_array`) es propia de skforecast. `NoriTSForecaster` exige el extra `forecasting` (datasets, gluonts, statsmodels) y tiene abierta la issue upstream #194 ("synthefy.nori_ts fails to import on pandas 3 (ArrayManager was removed)", 2026-10-31; leído solo título y estado). El fallo de import sin el extra (`ModuleNotFoundError: datasets`) no se reprodujo [inferido]. Evidencia: `evidence/nori/upstream_checks.log`, `evidence/nori/edge_cases.log`.
- **NORI-F16** [documentado] synthefy-nori 0.23.0 exige `torch>=2.8,<2.14` y `numpy>=2.0` (metadatos de PyPI). No choca con skforecast (`numpy>=1.26.1`), pero instalar Nori obliga a numpy 2 y una futura torch 2.14 quedará retenida en el entorno de Nori [inferido]. Evidencia: `evidence/nori/pypi_info.log`, `evidence/nori/verification/pypi_info.log`.
- **NORI-F17** [verificado en código] Docs desfasados: ni el docstring de `FoundationModel` (lista de `model_id`, aprox. líneas 83-87 de `_foundation_model.py` en d9b8f4f) ni el de `NoriAdapter` (parámetro `model_id`, aprox. líneas 4423-4427) mencionan `Synthefy/Nori-1-5`, que el README upstream usa en todos sus ejemplos; el docstring de `NoriAdapter` (`point_estimate`, aprox. 4438-4440) y `skills/foundation-forecasting/references/adapter-parameters.md` (sección de `NoriAdapter`, aprox. líneas 165-181) presentan `'mode'` como distinto de `'median'`; el docstring de `supports_nan_in_series` (aprox. 4482-4486) es inexacto. `tools/ai/llms-base.txt` (aprox. línea 495) y `SKILL.md` siguen siendo correctos. Matiz de la verificación: el enlace `https://docs.synthefy.com/nori/` del docstring es el mismo que la insignia Docs del README upstream; solo el enlace de la guía de usuario (celda de la tabla de modelos, celda 11 en la revisión) apunta a `https://docs.synthefy.com/forecasting-api/overview` (API alojada). `docs.synthefy.com` no fue accesible desde la revisión, así que el contenido de ambos enlaces no se ha comprobado.
- **NORI-F18** [verificado en código] `FakeNoriRegressor` (`fixtures_adapters.py`, aprox. líneas 593-635 en d9b8f4f) diverge del backend: `'mode'` devuelve la media (real: mediana), `fit` acepta NaN en `y` (real: ValueError) y no valida los cuantiles (real: ValueError fuera de `(0, 1)`). `test_NoriAdapter_predict_quantiles_shape_and_column_order` (aprox. línea 247 de `test_NoriAdapter.py`) comprueba además que el fake recibe niveles ordenados. `adapters.toml` `[nori]` (aprox. líneas 56-59) solo prueba `Synthefy/Nori` (6M): Nori-1-5 (otra arquitectura, backend mínimo 0.22.0) no tiene cobertura semanal, y los smoke no comprueban el orden con cuantiles desordenados ni NaN en las covariables. `BACKEND_CONTRACTS["nori"]` de `tools/foundation_compat/tests/contracts.py` (aprox. líneas 139-147) refleja exactamente las llamadas reales.
- **NORI-F19** [probado] Backtest de la guía (vic_electricity, `Synthefy/Nori` 6M, exog `Temperature` y `Holiday`, `context_length=500`, pasos de 24) antes y después del prototipo de NORI-P2: MAE 185.9507 en ambos, 720 predicciones, diferencia máxima 0.0 entre variantes y 0.0 frente al CSV de la primera pasada. vic_electricity no tiene NaN en `Temperature` ni `Holiday`, así que solo demuestra que no hay regresión sin NaN; el efecto con NaN se ve en `evidence/nori/verification/verify_nori.log` (sección D). Python 3.12.3, synthefy-nori 0.23.0, torch 2.13.0+cpu, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1, huggingface_hub 1.33.0; Intel Xeon 2.10 GHz, 4 CPU, sin GPU. Evidencia: `evidence/nori/backtest_current.log`, `evidence/nori/backtest_patched.log`, `evidence/nori/backtest_current.csv`, `evidence/nori/backtest_patched.csv`, `evidence/nori/verification/backtest_current.log`, `evidence/nori/verification/backtest_patched.log`.
- **NORI-V1** [verificado en código] `backend_package` no se aplica en tiempo de ejecución: solo se usa en el mensaje del ImportError de `_load_model`, en `get_model_info`, en `tools/foundation_compat/run.py` (instalación) y en `test_backends_compat_config.py` (igualdad con `adapters.toml`). Un mínimo ahí no protege a quien ya tiene synthefy-nori <0.22 frente al TypeError de Nori-1-5 (que persiste con 0.21.1 instalado [probado], `evidence/nori/verification/old_backend_check.log`). El único patrón de comprobación en ejecución es `TimesFM3Adapter` (`importlib.metadata`). Además, el ImportError de `NoriAdapter._load_model` (aprox. línea 4890 en d9b8f4f) escribe `pip install {self.backend_package}` sin comillas, mientras `TabPFNAdapter` y `_import_timesfm` lo entrecomillan; con `>=` sin comillas el shell interpretaría una redirección (hallazgo transversal X-4).
- **NORI-V2** [probado] Con synthefy-nori <0.22, el nombre de registro `nori-1-5` (`nori_config={"model": "nori-1-5"}`) no está en `NORI_MODELS` de 0.21.1, se trata como repo id crudo y falla con un `CheckpointAccessError` "access denied" engañoso, no con un error de versión. Afecta solo a quien use el nombre de registro; el adaptador pasa el repo id. Evidencia: `evidence/nori/verification/old_backend_check.log`.

**Descartado**

- Riesgo original de NORI-P6 ("el preprocesado de DataFrame trata también las columnas numéricas y las predicciones podrían cambiar aunque no haya categóricas"): refutado. `_transform_tabular` (`featurize.py` de synthefy 8.0.0, dependencia `synthefy>=7,<9` de synthefy-nori 0.23.0) pasa las columnas numéricas como float32 sin transformar [verificado en código] y numpy frente a DataFrame numérico da 0.0 [probado, `evidence/nori/verification/dataframe_numeric_check.log`].
- "El orden de cuantiles se respeta desde al menos v0.16.0" (NORI-F9 original): corregido a "desde v0.11.0".

## 3. Cambios a implementar

### NORI-P1. Rechazar `Synthefy/Nori-Causal` al resolver el adaptador (en `02_common.md`)

- **Qué**: no se implementa en este PR. `02_common.md` añade la tabla central de ids no soportados en `_resolve_adapter` (consultada antes de los prefijos), con `Synthefy/Nori-Causal` y su motivo (modelo de descubrimiento causal, no un forecaster); `FoundationModel("Synthefy/Nori-Causal")` debe fallar al construir con un ValueError explicativo. Su test en `test_resolve_adapter.py` comprueba que `Synthefy/Nori`, `Synthefy/Nori-1-5`, `Synthefy/Nori-30M` y `Synthefy/Nori-100M` siguen resolviendo a `NoriAdapter`.
- **Por qué**: NORI-F5.
- **Riesgo**: bajo; hoy ya falla, pero tarde [probado].
- **Rompe compatibilidad**: no.
- **Decisión**: D-REG (en `02_common.md`).
- **En este PR**: nada en código. Al tocar los docstrings (NORI-P4), no listar Nori-Causal como `model_id` válido.

### NORI-P2. Conservar las filas de contexto con covariables NaN

- **Qué**: en `NoriAdapter.predict` (bloque del comentario "NoriRegressor rejects NaN" y de `valid_rows`, aprox. líneas 4831-4842 en d9b8f4f), filtrar solo las filas cuyo objetivo es NaN y pasar las covariables NaN al backend:

  ```python
  y_ctx = series.to_numpy(dtype=float)

  # NoriRegressor rejects NaN in the target only; NaN in the features are
  # handled by the backend. The running-index feature is an absolute offset,
  # so dropping interior rows keeps the remaining rows positioned in time.
  valid_rows = ~np.isnan(y_ctx)
  if not valid_rows.any():
      raise ValueError(
          f"Series '{series_name}' has no observed values in its context. "
          f"NoriAdapter cannot predict it."
      )
  X_ctx = X_ctx[valid_rows]
  y_ctx = y_ctx[valid_rows]
  ```

  Actualizar el docstring del atributo `supports_nan_in_series` en la sección Attributes de `NoriAdapter` (aprox. líneas 4482-4486): `NoriRegressor` rechaza NaN solo en el objetivo, el adaptador descarta las filas de contexto con objetivo NaN y las covariables NaN se pasan al backend. `_featurize` ya conserva NaN al convertir a float32 (`_to_float_array` convierte con `to_numpy(dtype=np.float32)` desde el commit 2282824 de `origin/0.27.x`; en d9b8f4f era `astype(np.float32).to_numpy()`, y ambas expresiones conservan NaN) [verificado en código], y las filas de consulta con NaN ya se pasan hoy tal cual y el backend las acepta [probado, `evidence/nori/verification/verify_nori.log`, sección D].
- **Relación con `02_common.md`** [inferido]: la comprobación central de contexto todo NaN (D11) rechazará antes, en `FoundationModel.predict`, una ventana sin ningún valor observado; el ValueError del adaptador queda solo para llamadas directas al adaptador (los tests unitarios lo usan así). Mantenerlo es barato; quitarlo es aceptable si 02 se fusiona con D11 (a).
- **Por qué**: NORI-F11, NORI-F19.
- **Riesgo**: cambia predicciones solo cuando el contexto tiene covariables futuras conocidas con NaN en filas con objetivo observado [probado: sin NaN en la exog, MAE 185.9507 antes y después y diferencia 0.0]. No exige subir el mínimo del backend: 0.18.2 ya acepta NaN en `X` [probado]. Si un backend futuro dejara de aceptar NaN en `X`, lo detectaría el smoke con NaN en la exog del contexto que añade `10_docs_ci.md` (sección 3.6).
- **Rompe compatibilidad**: no en la API; sí en resultados para usuarios con NaN en la exog histórica (línea en "Before upgrading", ver sección 5).
- **Decisión**: D23.

### NORI-P3. Versión mínima `synthefy-nori>=0.22` (incluye NORI-V1 y NORI-V2)

- **Qué** (según D24):
  - (a) Recomendada: `backend_package: str = "synthefy-nori>=0.22"` en `NoriAdapter` (aprox. línea 4557) y el mismo valor en `[nori]` de `tools/foundation_compat/adapters.toml` (lo exige `test_adapters_toml_entries_match_the_adapter_classes`). Documentar que el mínimo solo cubre instalaciones nuevas (y el CI): quien ya tiene 0.21.x sigue viendo el TypeError opaco con Nori-1-5 [verificado en código, NORI-V1].
  - (b) Solo documentar el mínimo para Nori-1-5, sin tocar `backend_package`.
  - (c) Además de (a), comprobar la versión instalada en `_load_model` con el helper común opcional de `02_common.md` (si se adopta D-VER (b)), para dar un error claro también a las instalaciones existentes. El mínimo sería global (como `backend_package`), aunque 6M funcione con 0.21.1 [probado]; filtrar por checkpoint metería lógica de checkpoints en el adaptador, contra `.claude/rules/foundation.md` [inferido]. Traducir el TypeError del backend dentro del adaptador no se recomienda por el mismo motivo.
  - Con cualquier opción, entrecomillar el comando del ImportError de `NoriAdapter._load_model` (aprox. líneas 4885-4891), con el patrón de `TabPFNAdapter._load_model` y `_import_timesfm` (X-4):

    ```python
    raise ImportError(
        f"{self.backend_package} is required for NoriAdapter. "
        f'Install it with `pip install "{self.backend_package}"`.'
    ) from exc
    ```

  - NORI-V2 no requiere código (el adaptador pasa el repo id crudo); se documenta en NORI-P4.
- **Por qué**: NORI-F4, NORI-F16, NORI-V1, NORI-V2, X-4.
- **Riesgo**: bajo. `test_adapters_toml_backend_package_is_a_PEP_508_requirement` acepta `>=`: `tabpfn-time-series>=1.3` ya pasa ese test [verificado en código].
- **Rompe compatibilidad**: no en la API de skforecast. Con (a) solo cambia el comando sugerido y lo que instalan `run.py` y el CI; con (c), quien tenga synthefy-nori <0.22 recibe un error al cargar cualquier checkpoint, también 6M, que hoy le funciona.
- **Decisión**: D24 (y D-VER para la opción (c)).

### NORI-P4. Documentación de Nori-1-5, `'mode'`, NaN y enlaces

- **Qué**: textos, sin cambio de comportamiento (lugares exactos en la sección 5):
  - Añadir `Synthefy/Nori-1-5` a los docstrings de `NoriAdapter` (parámetro `model_id`) y de `FoundationModel` (lista de `model_id`), a `references/adapter-parameters.md` y a la celda de la tabla de modelos de la guía, con la nota de que necesita synthefy-nori>=0.22 y de que con versiones anteriores el nombre de registro `nori-1-5` falla con un error de acceso engañoso (NORI-V2).
  - Indicar que, con los checkpoints publicados (cabeza de cuantiles), `point_estimate='mode'` devuelve lo mismo que `'median'`.
  - Corregir el texto sobre NaN (ver NORI-P2) en el docstring y en `adapter-parameters.md`.
  - Enlace de la guía: cambiar `https://docs.synthefy.com/forecasting-api/overview` por `https://docs.synthefy.com/nori/`, el mismo del docstring y de la insignia Docs del README upstream [verificado en código]. Abrir ambos enlaces antes de cambiarlo: la revisión no pudo acceder a `docs.synthefy.com`.
- **Por qué**: NORI-F8, NORI-F11, NORI-F17, NORI-V2.
- **Riesgo**: nulo.
- **Rompe compatibilidad**: no.
- **Decisión**: ninguna (el texto del mínimo depende de D24; si D23 = (b), el texto de NaN solo corrige lo que hace hoy el adaptador).

### NORI-P5. Quitar los workarounds redundantes (orden de cuantiles y rama torch)

- **Qué**:
  - En `NoriAdapter.predict`, eliminar el comentario "Nori does not guarantee...", `query_levels = sorted(set(quantile_list))` y `quantile_column_indices` (aprox. líneas 4795-4800 en d9b8f4f), y pedir al backend los niveles tal como llegan:

    ```python
    q = self._model.predict(X_fut, output_type="quantiles", quantiles=quantile_list)
    # Nori returns (n_quantiles, steps) in the requested order;
    # skforecast expects (steps, n_quantiles).
    predictions[series_name] = (
        np.asarray(q, dtype=float).reshape(len(quantile_list), steps).T
    )
    ```

  - Eliminar el método estático `NoriAdapter._to_numpy` (aprox. líneas 4901-4924) y usar `np.asarray(y_hat, dtype=float).reshape(-1, 1)` en la rama de punto. Conservar los `reshape` (upstream corrigió la forma de las predicciones de una sola fila en 3935683, 2026-08-29, "preserve one-row prediction shape" [verificado en código, `evidence/nori/upstream_checks.log`]).
  - Mantener la validación propia de niveles en `(0, 1)` al inicio de `predict`: el backend también la hace (`"quantiles must lie strictly in (0, 1)."` [verificado en código]), pero el mensaje del adaptador es más claro.
- **Por qué**: NORI-F9, NORI-F10.
- **Riesgo**: bajo, sin cambio de predicciones en todo el rango leído: orden y duplicados desde v0.11.0 y salida numpy desde al menos v0.15.0 [verificado en código; probado en 0.23.0]. Las versiones de PyPI anteriores a v0.11.0 no se han revisado; con D24 (a) las instalaciones nuevas quedan fuera de ese rango [inferido]. La red de seguridad con el backend real es el smoke de orden de cuantiles que añade `10_docs_ci.md` (NORI-P7) y, mientras tanto, `compare_direct.py` ("unsorted levels keep order: True").
- **Rompe compatibilidad**: no (`_to_numpy` es privado).
- **Decisión**: ninguna.

### NORI-P6. Covariables categóricas nativas del backend (pospuesta hasta D25)

- **Qué** (solo si D25 = (a)): construir en `NoriAdapter._featurize` un `pandas.DataFrame` con las features numéricas y las columnas de exog con su dtype original, en vez de convertir todo con `_to_float_array`; pasarlo a `NoriRegressor.fit` y `predict`; declarar `supports_categorical_covariates = True`. Cuidar los riesgos reales [verificado en código]: columnas `object` con cardinalidad mayor que `max_categorical_cardinality` lanzan error en el backend, y `predict` exige un DataFrame con las mismas columnas y dtypes que `fit`. Requiere synthefy-nori>=0.18.1, cubierto por NORI-P3 (a).
- **Por qué**: NORI-F12.
- **Riesgo**: con covariables solo numéricas la salida no cambia (0.0) [probado]; solo cambia lo que hoy lanza ValueError (columnas no numéricas).
- **Rompe compatibilidad**: no; amplía lo aceptado. Cambia `get_model_info("Synthefy/Nori").supports_categorical_covariates` y el test `test_NoriAdapter_predict_ValueError_on_non_numeric_covariate`.
- **Decisión**: D25. Si D25 = (b), queda fuera de este PR (sección 8); con (a) puede ir aquí o en un PR posterior.

### NORI-P7. Cobertura de Nori-1-5 en el CI de compatibilidad (en `10_docs_ci.md`)

- **Qué**: no se implementa en este PR. `10_docs_ci.md` añade una segunda entrada en `tools/foundation_compat/adapters.toml` para `Synthefy/Nori-1-5`, la añade también a la matriz `adapter:` de `.github/workflows/foundation-backends.yml` (lo exige `test_workflow_matrix_matches_adapters_toml`) con el mismo `backend_package` que `NoriAdapter` (lo exige `test_adapters_toml_entries_match_the_adapter_classes`), y añade en `tools/foundation_compat/tests/test_smoke.py` un test de orden con cuantiles desordenados.
- **Por qué**: NORI-F4, NORI-F18.
- **Riesgo**: un job más (83 MB de pesos).
- **Rompe compatibilidad**: no.
- **Decisión**: D26.
- **En este PR**: dejar fijado el valor de `backend_package` (NORI-P3) que usarán las dos entradas.

### NORI-P8. Alinear `FakeNoriRegressor` con el backend real

- **Qué**, en `skforecast/foundation/tests/tests_foundation_models/fixtures_adapters.py`:
  - `predict` con `output_type="mode"` devuelve la mediana de `y_` (como el backend con cabeza de cuantiles).
  - `fit` lanza `ValueError("y must contain only finite target values")` si `y` tiene NaN, como 0.23.0, y guarda la matriz recibida (`self.X_ = np.asarray(X, dtype=float)`) para poder comprobar NORI-P2.
  - `predict` lanza `ValueError("quantiles must lie strictly in (0, 1).")` con niveles fuera de `(0, 1)`.
  - Actualizar el docstring de la clase.
- **Por qué**: NORI-F18.
- **Riesgo**: nulo; con el fake estricto, el test de filas con objetivo NaN pasa a detectar una regresión del filtrado.
- **Rompe compatibilidad**: no.
- **Decisión**: ninguna.

### NORI-P9. Delegar la featurización en `synthefy.nori_ts.NoriTSForecaster` (pospuesta hasta D27)

- **Qué** (solo si D27 = (a)): usar `NoriTSForecaster(mode="local", model=...)` y eliminar `_featurize`, `_timestamps`, `_known_future_columns` y `_to_float_array`.
- **Por qué**: NORI-F15 (quitaría unas 150 líneas propias y seguiría las correcciones de features de upstream).
- **Riesgo**: alto [verificado en código]: rompe los parámetros públicos `add_calendar_features` y `n_fourier_terms`, cambia las predicciones (otras features; punto igual a mediana, sin `'mean'`), añade datasets, gluonts y statsmodels, y upstream tiene abierta la issue #194 (pandas 3).
- **Rompe compatibilidad**: sí (parámetros públicos y predicciones).
- **Decisión**: D27; la recomendación es (b), mantener la featurización propia (revisitar si upstream la convierte en la vía principal sin extras pesados), así que no entra en este PR.

### NORI-P10. Checkpoint por defecto (pospuesta hasta D28)

- **Qué** (solo si D28 = (b)): `default_model_id = "Synthefy/Nori-1-5"` en `NoriAdapter`.
- **Por qué**: el README upstream usa `nori-1-5` en todos sus ejemplos y `hf.py` v0.23.0 define `DEFAULT_DOWNLOAD_MODEL_NAME = "nori-1-5"` (CLI de descarga) [verificado en código]; relacionado con NORI-F4 y NORI-F17.
- **Riesgo**: con (b) cambian las salidas de la guía y de las herramientas que usan `default_model_id`, y exige D24 (a) o (c).
- **Rompe compatibilidad**: no en la API; sí en las salidas por defecto.
- **Decisión**: D28; la recomendación es (a), mantener `Synthefy/Nori` 6M (el más pequeño y el que usan CI y docs), así que no entra en este PR.

## 4. Tests

**Tests unitarios** (`skforecast/foundation/tests/tests_foundation_models/`):

- `test_NoriAdapter.py`:
  - NORI-P2: nuevo `test_NoriAdapter_predict_keeps_context_rows_with_nan_covariates`: contexto con NaN en una covariable futura conocida (en filas con objetivo observado) y exog futura; comprobar con `fake.X_` que llegan todas las filas y que la columna de la covariable conserva los NaN. Mantener `test_NoriAdapter_predict_drops_nan_rows_before_fit` (sigue siendo correcto: filas con objetivo NaN) y revisar su docstring. Actualizar el mensaje esperado de `test_NoriAdapter_predict_ValueError_when_all_context_rows_are_nan` al nuevo texto ("has no observed values in its context") y su docstring.
  - NORI-P3: `test_NoriAdapter_load_model_ImportError_when_backend_missing` debe comprobar el comando entrecomillado, por ejemplo `match=re.escape('pip install "synthefy-nori>=0.22"')` con D24 (a) o (c) (con (b), `'pip install "synthefy-nori"'`). Con D24 (c), añadir un test que simule una versión antigua (monkeypatch de la función que lea la versión en el helper de 02) y espere el error claro.
  - NORI-P5: en `test_NoriAdapter_predict_quantiles_shape_and_column_order`, cambiar la aserción final a `fake.last_quantiles == [0.9, 0.1, 0.5]` (niveles en el orden pedido) y su docstring; añadir un caso con niveles duplicados (`[0.5, 0.9, 0.5]`) que devuelva tres columnas en ese orden; eliminar `test_NoriAdapter_predict_handles_device_tensor_output`.
  - NORI-P8: un test de que `point_estimate="mode"` y `"median"` dan lo mismo con el fake (refleja el backend real).
  - NORI-P6 (solo con D25 (a)): sustituir `test_NoriAdapter_predict_ValueError_on_non_numeric_covariate` por tests con categorías no vistas en el horizonte y con NaN.
- `fixtures_adapters.py`: cambios de `FakeNoriRegressor` de NORI-P8.
- `test_get_model_info.py`: en `test_get_model_info_output_for_each_adapter`, la tupla esperada de `"Synthefy/Nori"` pasa a `("NoriAdapter", 4096, "synthefy-nori>=0.22", True, False, False)` con D24 (a) o (c) (y el quinto elemento a `True` solo con D25 (a)).
- `test_backends_compat_config.py`: sin cambios; debe seguir pasando tras actualizar `adapters.toml` (mismo `backend_package` en el adaptador y en `[nori]`, requisito PEP 508, matriz del workflow sin cambios en este PR).
- `test_resolve_adapter.py`: sin cambios aquí; el caso de Nori-Causal lo añade `02_common.md`.

**Compatibilidad** (`tools/foundation_compat/`):

- `adapters.toml`: `[nori]` con `backend_package = "synthefy-nori>=0.22"` (D24 (a) o (c)). La entrada de Nori-1-5 es de `10_docs_ci.md`.
- `tests/contracts.py`: sin cambios; `BACKEND_CONTRACTS["nori"]` sigue reflejando las llamadas (`NoriRegressor` con `model`, `fit` con 2 posicionales, `predict` con 1 posicional y `output_type`, `quantiles`) [verificado en código].
- `tests/test_smoke.py`: sin cambios en este PR. El fichero es común a todos los adaptadores y el workflow solo se lanza en la ejecución semanal y en los PR a `main` (no en los PR a ramas `*.x`) [verificado en código, `.github/workflows/foundation-backends.yml`]; que todos los backends con exog acepten NaN en la exog del contexto no se ha comprobado [inferido]. Por eso el smoke propuesto por NORI-P2 (copia de `exog` con NaN en una de cada cuatro filas, `fit` y `predict(steps=STEPS, exog=exog_future)`, forma, índice y valores finitos; saltado si `get_model_info(model_id).allow_exog` es False) se añade en `10_docs_ci.md`, junto al smoke de orden de cuantiles. La comprobación con el backend real en este PR es `verify_nori.py` (sección D).

**Comprobaciones antes y después** (desde la raíz del repositorio, entorno de la sección 1):

```bash
# Adaptador frente a llamada directa (features construidas de forma independiente)
~/.venvs/fm-review-nori/bin/python dev/foundation_adapters_review/evidence/nori/compare_direct.py
# Verificación independiente, incluida la sección D de NaN en covariables (descarga Nori-1-5, 83 MB)
~/.venvs/fm-review-nori/bin/python dev/foundation_adapters_review/evidence/nori/verification/verify_nori.py
# Casos límite (steps=1, duplicados, RangeIndex, series heterogéneas, mode y median)
~/.venvs/fm-review-nori/bin/python dev/foundation_adapters_review/evidence/nori/edge_cases.py
# NaN en covariables: variante actual frente al prototipo
~/.venvs/fm-review-nori/bin/python dev/foundation_adapters_review/evidence/nori/patch_keep_nan_covariates.py nan-check
```

- `patch_keep_nan_covariates.py` sustituye `NoriAdapter.predict` (variante `patched`) por una copia que llama a `self._to_numpy`: ejecutar `nan-check` después del commit de NORI-P2 y antes del de NORI-P5. Tras NORI-P5 la variante `patched` falla con AttributeError por la falta de `_to_numpy` [verificado en código del script]; usar entonces `verify_nori.py` (sección D), que no depende de métodos privados eliminados.
- Si `02_common.md` ya está fusionado, `verify_nori.py` termina con un ValueError no capturado en la sección A2 (`FoundationModel("Synthefy/Nori-Causal")` falla al construir) después de imprimir las secciones A a D; es lo esperado [inferido]. `compare_direct.py` captura la excepción en su sección 5.

**Backtest de la guía** (NORI-P2 puede alterar predicciones; comprobación de corrección, no benchmark):

```bash
~/.venvs/fm-review-nori/bin/python dev/foundation_adapters_review/evidence/common/backtest_guide.py \
    --model-id Synthefy/Nori --exog --out nori_backtest_after.csv
~/.venvs/fm-review-nori/bin/python -c 'import pandas as pd; a = pd.read_csv("dev/foundation_adapters_review/evidence/nori/backtest_current.csv", index_col=0); b = pd.read_csv("nori_backtest_after.csv", index_col=0); print((a["pred"] - b["pred"]).abs().max())'
```

Esperado: `"mae": 185.9507`, `"n_predictions": 720` y diferencia máxima 0.0 frente a `evidence/nori/backtest_current.csv` (el dataset no tiene NaN en exog). Borrar después `nori_backtest_after.csv`; no se versiona.

**Backend antiguo** (solo para NORI-P3 con D24 (c), o para ver el comportamiento con 0.21.1):

```bash
uv venv ~/.venvs/fm-review-nori-old --python 3.12
uv pip install --python ~/.venvs/fm-review-nori-old/bin/python --torch-backend cpu -e . pytest "synthefy-nori==0.21.1"
# Backend directo: comportamiento de partida
~/.venvs/fm-review-nori-old/bin/python dev/foundation_adapters_review/evidence/nori/verification/old_backend_check.py
# A través de skforecast (para comprobar la opción (c))
~/.venvs/fm-review-nori-old/bin/python -c 'import numpy as np, pandas as pd; from skforecast.foundation import FoundationModel; y = pd.Series(np.arange(100.0), index=pd.date_range("2024-01-01", periods=100, freq="D")); m = FoundationModel(model_id="Synthefy/Nori-1-5"); m.fit(series=y); print(m.predict(steps=3))'
```

`old_backend_check.py` llama solo al backend; hoy imprime `Synthefy/Nori : ok`, `Synthefy/Nori-1-5 : TypeError: get_x_encoder() got an unexpected keyword argument 'prompt_quantile_ple_bins'` y `nori-1-5 : CheckpointAccessError ... access denied` [probado]; no cambia con este PR. La línea con `FoundationModel` muestra hoy el mismo TypeError [inferido]; con D24 (c) debe mostrar el error de versión de skforecast al cargar.

## 5. Documentación y contexto de IA

- **Docstrings** (`skforecast/foundation/_adapters.py`, `NoriAdapter`; leer antes `.github/instructions/docstrings.instructions.md`):
  - Parámetro `model_id` (clase, aprox. líneas 4423-4427 en d9b8f4f, y `__init__`, aprox. 4576-4577): añadir `"Synthefy/Nori-1-5"` con la nota "requires `synthefy-nori>=0.22`". No listar `Synthefy/Nori-Causal`.
  - Parámetro `point_estimate` (aprox. 4438-4440 y 4589-4591): "With the published checkpoints (quantile regression head), `'mode'` returns the same values as `'median'`."
  - Atributo `supports_nan_in_series` (aprox. 4482-4486): texto de NORI-P2.
  - Atributo `backend_package`: sin cambios de texto; el valor cambia con D24.
  - Notes (aprox. 4528-4531): mantener "Covariates must be numeric" salvo con D25 (a).
- **Docstring de `FoundationModel`** (`skforecast/foundation/_foundation_model.py`): añadir `'Synthefy/Nori-1-5'` a la lista "Synthefy Nori (supports `exog`)" (aprox. líneas 83-87). La lista de parámetros de Nori (aprox. 135-138) no cambia. `docs/api/FoundationModel.md` se genera del docstring.
- **Guía de usuario** `docs/user_guides/foundation-forecasting-models.ipynb`, solo la celda Markdown con la tabla HTML de modelos soportados (celda 11 en la revisión; fila `<th scope="row">Nori</th>`):
  - Columna de `model_id`: añadir `Synthefy/Nori-1-5`.
  - Instalación, con D24 (a) o (c): `<code>pip install "synthefy-nori&gt;=0.22"</code>`, `data-copy='pip install "synthefy-nori&gt;=0.22"'` y `aria-label`, con el mismo patrón que la fila de TabPFN-TS.
  - Enlace de documentación: `https://docs.synthefy.com/forecasting-api/overview` a `https://docs.synthefy.com/nori/` (NORI-P4).
  - No reejecutar el notebook: lo hace `10_docs_ci.md` una sola vez. Las celdas de la sección "Synthefy Nori" (celdas 90 a 95 en la revisión) y la tabla comparativa de métricas no cambian.
- **Skill** `skills/foundation-forecasting/`:
  - `SKILL.md`: la línea de instalación `pip install synthefy-nori          # For Nori` pasa a `pip install "synthefy-nori>=0.22"  # For Nori` con D24 (a) o (c); en la tabla "Choosing a Model", la fila `Synthefy/Nori` puede mencionar Nori-1-5. Con D25 (a), corregir también el error común 8 ("neither do Nori, T0, and TS-ICL").
  - `references/adapter-parameters.md`, sección de `NoriAdapter`: añadir Nori-1-5 en la fila de `model_id` y la nota de versión mínima (con NORI-V2) en el párrafo final; en la fila de `point_estimate`, la equivalencia de `'mode'` y `'median'`; en el párrafo de atributos de capacidad (`supports_nan_in_series`, aprox. línea 242), cambiar "Nori drops the NaN rows itself" por "Nori drops the context rows whose target is NaN; NaN in the covariates are passed to the backend".
- **`tools/ai/llms-base.txt`**: la fila de `NoriAdapter` de la tabla de adaptadores (aprox. línea 495) usa el prefijo `Synthefy/Nori` y sigue siendo correcta [verificado en código]; no se cambia salvo que D25 (a) cambie lo dicho sobre categóricas.
- **Regeneración**: seguir el skill `ai-context-sync`: `python tools/ai/generate_ai_context_files.py` y después `python tools/ai/generate_ai_context_files.py --check`. No editar a mano `AGENTS.md`, `llms-full.txt` ni `docs/llms*.txt`.
- **Release note** (`docs/releases/releases.md`, sección 0.27.0, con el skill `release-note`; el enlace al PR se añade al abrirlo):
  - **Changed** (con D23 (a)):

    ```
    + <code>[NoriAdapter]</code> keeps the context rows whose exogenous variables have missing values and passes them to the backend, which accepts them; only the rows with a missing target are dropped. Predictions change only when the historical `exog` of a series has missing values.
    ```

    Con D24 (a) o (c):

    ```
    + <code>[NoriAdapter]</code> requires `synthefy-nori>=0.22`, the first version that loads the Synthefy Nori 1.5 checkpoint (`'Synthefy/Nori-1-5'`), now listed in the documentation. Existing installations are not upgraded automatically: run `pip install -U "synthefy-nori>=0.22"`.
    ```

  - **"Before upgrading"** (resultados que cambian sin cambiar el código; en el bloque `!!! warning`, con la sangría de cuatro espacios de las entradas existentes):

    ```
    + <code>[FoundationModel]</code> and <code>[ForecasterFoundation]</code> with Synthefy Nori and missing values in the historical `exog`: predictions change, because the context rows with missing covariates are no longer dropped.
    ```

  - Nota: la entrada de 0.25.0 ("The rows whose target or covariates are NaN are now dropped before the in-context fit") es de una versión publicada y no se edita.

## 6. Decisiones pendientes

| Id | Decisión | Opciones | Recomendación | Impacto en el plan | Estado |
|---|---|---|---|---|---|
| D23 | NaN en covariables de Nori (NORI-P2) | (a) aplicar; (b) solo corregir el docstring | (a) | Con (b), NORI-P2 se reduce al docstring, sin línea en "Before upgrading" ni smoke con NaN en `10_docs_ci.md` | pendiente |
| D24 | Mínimo de Nori (NORI-P3) | (a) `synthefy-nori>=0.22` con comillas y nota de que solo cubre instalaciones nuevas; (b) solo documentar; (c) comprobación en `_load_model` | (a); (c) si se adopta D-VER | Fija `backend_package`, `adapters.toml`, `test_get_model_info.py`, el comando de instalación de la skill y de la guía, y la entrada de release note; (c) depende del helper de `02_common.md`. Las comillas del ImportError van con cualquier opción | pendiente |
| D25 | Categóricas nativas de Nori (NORI-P6) | (a) adoptar DataFrame con dtypes; (b) mantener solo numéricas | (a) con baja prioridad: con numéricas la salida no cambia (0.0) y solo se amplía lo que hoy falla; cuidar cardinalidad máxima y dtypes en `predict` | Con (a), NORI-P6 entra (aquí o en un PR posterior) y cambian `supports_categorical_covariates`, docs y skill; con (b), fuera de alcance | pendiente |
| D26 | Cobertura CI de Nori-1-5 (NORI-P7) | (a) segunda entrada; (b) sustituir la actual; (c) nada | (a) | Se implementa en `10_docs_ci.md`; aquí solo se fija el `backend_package` común | pendiente |
| D27 | Featurización de Nori (NORI-P9) | (a) delegar en `NoriTSForecaster`; (b) mantener la propia | (b) | Con (b), nada en este PR; con (a), PR aparte con rotura de API | pendiente |
| D28 | Checkpoint por defecto de Nori (NORI-P10) | (a) mantener Synthefy/Nori 6M; (b) Nori-1-5 (cambia salidas y exige D24) | (a) | Con (a), nada; con (b), cambia `default_model_id` y las salidas de docs y herramientas | pendiente |

Decisiones de otros planes que afectan a este: D-REG (NORI-P1) y D-VER (opción (c) de D24), en `02_common.md`; D11 (contexto todo NaN), que decide si el ValueError de NORI-P2 queda solo como defensa para llamadas directas; D-Q (niveles de cuantil a float), sin efecto en la lógica de Nori.

## 7. Criterios de aceptación

- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_NoriAdapter.py -q`: todo en verde (43 tests en la revisión, más los nuevos y menos el de tensores).
- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_get_model_info.py skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py -q`: en verde.
- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/ -q` una vez al terminar (módulo tocado).
- [ ] `~/.venvs/fm-review-nori/bin/python -m pytest tools/foundation_compat/tests/test_contract.py --backend nori -q -p no:cacheprovider`: 4 passed.
- [ ] `~/.venvs/fm-review-nori/bin/python -m pytest tools/foundation_compat/tests/test_smoke.py --backend nori -q -p no:cacheprovider`: 6 passed.
- [ ] `python tools/foundation_compat/run.py --adapter nori`: contrato y smoke en verde, con la instalación del `backend_package` nuevo.
- [ ] `compare_direct.py`: diferencia 0.0 frente a la llamada directa en punto y cuantiles para `Synthefy/Nori` y `Synthefy/Nori-1-5`, `unsorted levels keep order: True`, `mode == median: True`.
- [ ] `verify_nori.py`: secciones B/C con 0.0 en `mean`, `median`, `mode` y cuantiles `[0.8, 0.2, 0.5, 0.05]`; sección D con `current adapter: rows passed to fit = 256, NaN in X = True` (antes: 188, False) y `keep-NaN variant vs direct backend call on X with NaN: max diff 0.000e+00`. La línea `current adapter vs direct call on the rows without NaN` deja de dar 0.0, porque el adaptador ya no descarta esas filas [inferido].
- [ ] `patch_keep_nan_covariates.py nan-check` (entre NORI-P2 y NORI-P5): línea `current` con `200 of 200` (antes: 150 of 200; MAE indicativo 0.2351 en lugar de 0.2534) y `patched vs direct NoriRegressor ... max diff 0.000e+00`.
- [ ] `edge_cases.py`: duplicados y desordenados `[0.9, 0.5, 0.5, 0.1]` con los mismos valores que en `evidence/nori/edge_cases.log` (`q_0.9` 33.985, `q_0.5` 33.526, `q_0.1` 32.965); `mode` igual a `median`; `steps=1` con forma `(1, 2)` en punto y `(1, 3)` en cuantiles.
- [ ] Backtest de la guía: MAE 185.9507, 720 predicciones y diferencia máxima 0.0 frente a `evidence/nori/backtest_current.csv`.
- [ ] Con D24 (c): la línea con `FoundationModel` de la sección 4, en el entorno con synthefy-nori 0.21.1, muestra el error de versión de skforecast en lugar del TypeError.
- [ ] `python tools/ai/generate_ai_context_files.py --check`: sin errores.
- [ ] Skill `verify` en verde (lint de los ficheros tocados, tests afectados, contexto de IA, release notes).
- [ ] Sin cambios fuera de los ficheros de la cabecera; la guía no se reejecuta.

## 8. Fuera de alcance

- NORI-P1 (rechazo de `Synthefy/Nori-Causal`): `02_common.md` (D-REG).
- NORI-P7 (entrada de Nori-1-5 en `adapters.toml` y en la matriz del workflow, smoke de orden de cuantiles) y el smoke con NaN en la exog del contexto propuesto por NORI-P2: `10_docs_ci.md` (D26, D23).
- La comprobación central de contexto todo NaN, el helper común de versión y la conversión de niveles de cuantil a float: `02_common.md`.
- La reejecución de la guía de usuario: `10_docs_ci.md`.
- NORI-P6 salvo D25 (a); NORI-P9 (D27, recomendación: no delegar); NORI-P10 (D28, recomendación: mantener 6M).
- Fijar la revisión del Hub (NORI-F7): ni el backend ni el adaptador lo permiten; queda para una política común de revisión y opciones de carga en todos los adaptadores (D13 de `evidence/cross_cutting.md`), sin propuesta aquí.
- Silenciar el `print` de mixed precision del backend (NORI-F14): no hay propuesta.
- El techo `torch<2.14` del backend (NORI-F16): informativo, no hay propuesta.
- Los `stacklevel` fijos (el aviso de `RangeIndex` de `NoriAdapter.predict` usa `stacklevel=3`): hallazgo transversal XC-8 de la revisión transversal (relacionado con X-6), no se toca aquí.
- CUDA y MPS (solo leídos en código) y los checkpoints 30M y 100M (no descargados): sin cambios previstos.
- Benchmarks de velocidad o de precisión entre versiones del backend o entre checkpoints: fuera del alcance de skforecast. El backtest de la guía solo se usa como comprobación de corrección antes y después.
