# 04. TimesFM 2.5 y TimesFM 3 (timesfm)

- **Rama propuesta**: `fix/timesfm-alignment`, creada desde `origin/0.27.x`.
- **Prioridad**: media. Los dos adaptadores están alineados con la última versión del backend y nada se rompe con timesfm 3.0.2 [probado]; lo urgente para usuarios es el mínimo de TimesFM 2.5 (cuantiles erróneos con timesfm 2.0.x, T25-F6).
- **Adaptadores**: `TimesFM25Adapter` y `TimesFM3Adapter`, los dos servidos por el paquete `timesfm`. Van en un único PR porque comparten el paquete y el helper `_import_timesfm`, donde vive el arreglo de TV-1 [verificado en código].
- **Ficheros afectados**:
  - `skforecast/foundation/_adapters.py`: `_import_timesfm`, `TimesFM25Adapter` (docstring, `backend_package`, `__init__`, `set_params`, `predict` (comentario), `_load_model`, `_ensure_compiled`) y `TimesFM3Adapter` (docstring de clase, `context_length`, `predict`, `_load_model`; `_RESERVED_PREDICT_KWARGS`, `_validate_predict_kwargs` y `predict_kwargs` solo si se adopta D6 (c)).
  - `skforecast/foundation/_foundation_model.py`: solo docstring (lista de kwargs de TimesFM 2.5 y nota de dispositivo).
  - `skforecast/foundation/tests/tests_foundation_models/`: `fixtures_adapters.py`, `test_TimesFM25Adapter.py`, `test_TimesFM3Adapter.py`, `test_get_model_info.py`.
  - `tools/foundation_compat/adapters.toml`, `tools/foundation_compat/tests/contracts.py`, `tools/foundation_compat/tests/test_smoke.py`.
  - `docs/user_guides/foundation-forecasting-models.ipynb` (solo texto de TimesFM, sin re-ejecutar), `skills/foundation-forecasting/SKILL.md`, `skills/foundation-forecasting/references/adapter-parameters.md`, `tools/ai/llms-base.txt`, ficheros generados por `tools/ai/generate_ai_context_files.py`, `docs/releases/releases.md`.
- **Depende de**: `02_common.md`, de forma suave.
  - T25-P3 (ids `-flax` y `-transformers` aceptados por el prefijo) se implementa allí, no aquí.
  - Si allí se adopta D-VER (b), este PR usa su helper común de comprobación de versión dentro de `_import_timesfm` en lugar de una comprobación propia (y declara la cota `>=3.0` también en `TimesFM3Adapter`, porque el helper la lee de `backend_package`). Si 02 aún no se ha fusionado, este PR se puede abrir igual: la comprobación de versión queda dentro de `_import_timesfm` y se sustituye por el helper común cuando exista.
  - 02 también centraliza la conversión de niveles de cuantil a float (D-Q) y la comprobación de contexto todo NaN (TV-2); si se fusiona antes, rebasar este PR sobre él: puede haber conflictos en `test_TimesFM25Adapter.py` y `test_TimesFM3Adapter.py` (validación de cuantiles) [inferido].
- **Bloquea a**: `10_docs_ci.md` (re-ejecución única de la guía y actualización del CI de compatibilidad, después de los textos y el mínimo de este PR).
- **Cierra issue**: ninguna issue de skforecast registrada en la revisión. Referencias upstream (google-research/timesfm) [documentado]: PR #435, #463, #483 y #492 (fusionados); PR #531, #552, #486, #549, #546, #515, #487 y #550 (abiertos); issues #338, #274, #475, #502 y #412 (abiertas; #412 ya resuelta por #435); issue #434 (cerrada); #505 (abierto, creado el 2026-09-11, "Raise when from_pretrained cannot honor use_variate_attention"; la evidencia no registra si es issue o PR); PR #415 (cerrado sin fusionar el 2026-10-31).

## 1. Estado en la revisión (2026-11-06)

Base de la revisión: rama `docs/foundation-adapters-review` desde `origin/0.27.x`, commit `d9b8f4f`, skforecast 0.27.0 en desarrollo [verificado en código].

### Versiones

| Elemento | Probado antes (CI) | Última en la revisión | Notas |
|---|---|---|---|
| `timesfm` (PyPI) | 3.0.2 | 3.0.2 (2026-09-09) | Historial: 1.3.0 (2025-07-06), 2.0.0 (2026-06-05, primera con TimesFM 2.5), 2.0.1 (2026-06-08), 2.0.2 (2026-07-02), 3.0.0 (2026-08-28), 3.0.1 (2026-09-02, sin tag git y su rueda sin la corrección #483), 3.0.2 (2026-09-09) [probado]. Evidencia: `evidence/timesfm25/pypi_timesfm.log`. |
| Código upstream | tag v3.0.2 | master `e51928e` (2026-09-29) | `git diff --stat v3.0.2 origin/master -- src/timesfm` vacío; después del tag solo cambian el README y `src/timesfm3/mlx` [probado]. Evidencia: `evidence/timesfm25/upstream_diff.log`, `evidence/timesfm3/upstream_diff.log`. |
| Requisitos de `timesfm` 3.0.2 | | | `requires_python >=3.10`, `numpy>=1.26.4`, `huggingface_hub>=0.28.0` (en la práctica hace falta >=0.30, ver TV-1), `safetensors>=0.5.3`; extra `torch` con `torch>=2.0.0`; extras `mlx`, `flax` y `xreg` (`jax[cuda]`, `scikit-learn`) [documentado]. Evidencia: `evidence/timesfm25/pypi_timesfm.log`. |
| `google/timesfm-2.5-200m-pytorch` | revisión vigente el 2026-11-06 | sha `1d952420fba8`, sin cambios | Último commit 2025-10-02 (README); pesos en `d418f3e8a8fa` (2025-10-01, "Refreshed model with speed optimizations"); apache-2.0, no gated, 925.2 MB, 232 tensores; `config.json` con `torch_compile: false` [probado]. Evidencia: `evidence/timesfm25/hub_checks.log`, `evidence/timesfm25/verification/hub_rerun.log`. |
| `google/timesfm-2.5-200m-flax` | | sha `b99e9f6c` (2025-09-02) | apache-2.0, no gated; no es cargable por el adaptador (ver T25-F8) [probado]. |
| `google/timesfm-2.5-200m-transformers` | | sha `5a9806b9` (creado 2026-02-19, modificado 2026-04-10) | apache-2.0, no gated, `library_name` transformers; no es cargable por el adaptador (ver T25-F8) [probado]. |
| `google/timesfm-3.0-pytorch` | revisión vigente el 2026-11-06 | sha `43046b85ec22`, sin cambios | Creado 2026-08-24, `lastModified` 2026-09-02 (README, LICENSE, metadatos); mismo blob LFS de `model.safetensors` (sha256 `a7592b0a8432...`) en los 10 commits desde 2026-08-24; licencia `other` (`timesfm-non-commercial-license-v1.0`), no gated, 1322.9 MB; no hay 3.5 ni flax [probado]. Evidencia: `evidence/timesfm3/hub_google_timesfm_recheck.log`, `evidence/timesfm3/verification/hub_commits_verify.log`. |
| Entornos de la revisión | | | Python 3.12.3, timesfm 3.0.2, torch 2.15.0+cpu, huggingface_hub 2.4.0, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1; 4 CPU, sin GPU, Linux [probado]. Evidencia: `evidence/timesfm25/requirements-freeze.txt`, `evidence/timesfm3/requirements-freeze.txt`. |

Entornos auxiliares usados [probado]: `~/.venvs/fm-review-timesfm25-oldhf` (timesfm 2.0.0, 2.0.1, 2.0.2 y 3.0.2 instalados en sucesión con `huggingface_hub==0.36.0`; queda con 2.0.2), `~/.venvs/fm-review-timesfm25-verify` (timesfm 3.0.2 con `huggingface_hub` 0.36.0, 1.0.0 y 0.29.3; queda con 0.29.3) y `~/.venvs/fm-review-timesfm3-v300` (timesfm 3.0.0). Evidencia: `evidence/timesfm25/env_oldhf.log`, `evidence/timesfm3/env_v300.log`.

### Línea base del CI

- Run 42212742430 del workflow Foundation Backends Compatibility, rama 0.27.x, commit `09433c35`, 2026-11-06, CPython 3.12.16, timesfm 3.0.2, torch 2.15.0+cpu, huggingface_hub 2.4.0 [probado].
  - Job 127875168998 (`timesfm25`): contrato 6 passed; smoke 5 passed y 1 skipped (exog no soportado); `held_back: false`, `status: passed`. Evidencia: `evidence/timesfm25/ci_baseline_job_127875168998.log`.
  - Job 127875169049 (`timesfm3`): contrato 3 passed; smoke 6 passed (con LicenseWarning); `held_back: false`, `status: passed`. Evidencia: `evidence/timesfm3/ci_baseline.log`.
- En los entornos de la revisión: contrato y smoke iguales al CI; tests unitarios de `TimesFM25Adapter` 51 passed (58 con la selección `-k timesfm25` de la verificación) y de `TimesFM3Adapter` 57 passed [probado]. Evidencia: `evidence/timesfm25/contract.log`, `evidence/timesfm25/smoke.log`, `evidence/timesfm25/unit_tests.log`, `evidence/timesfm25/verification/rerun_contract_smoke.log`, `evidence/timesfm25/verification/rerun_unit_tests.log`, `evidence/timesfm3/verification/contract_smoke_verify.log`, `evidence/timesfm3/verification/unit_tests_verify.log`.

### Antes de empezar

Este plan puede quedarse desfasado. Desde la raíz del repositorio, antes de tocar código:

```bash
# 1. Versiones de timesfm en PyPI (la revisión vio 3.0.2 como última)
python dev/foundation_adapters_review/evidence/common/pypi_info.py timesfm --since 2026-09-01

# 2. Checkpoints de Google en el Hub (comparar con los shas de la tabla anterior)
python dev/foundation_adapters_review/evidence/common/hub_info.py google --search timesfm
python dev/foundation_adapters_review/evidence/common/hub_info.py --repo google/timesfm-2.5-200m-pytorch
python dev/foundation_adapters_review/evidence/common/hub_info.py --repo google/timesfm-3.0-pytorch

# 3. Issues y PR upstream tocados desde la revisión (estado y fecha de merge)
gh api -X GET search/issues -f q="repo:google-research/timesfm updated:>=2026-11-06" \
    --jq '.items[] | [.number, .state, (.pull_request.merged_at // "-"), .title] | @tsv'

# 4. Entornos de la revisión (solo si no existen: make_env.sh borra el entorno previo)
bash dev/foundation_adapters_review/evidence/common/make_env.sh timesfm25 "timesfm[torch]" 3.12
bash dev/foundation_adapters_review/evidence/common/make_env.sh timesfm3 "timesfm[torch]" 3.12
uv venv ~/.venvs/fm-review-timesfm25-oldhf --python 3.12
uv pip install --python ~/.venvs/fm-review-timesfm25-oldhf/bin/python --torch-backend cpu \
    -e . pytest "timesfm[torch]==2.0.2" "huggingface_hub==0.36.0"
uv venv ~/.venvs/fm-review-timesfm3-v300 --python 3.12
uv pip install --python ~/.venvs/fm-review-timesfm3-v300/bin/python --torch-backend cpu \
    -e . pytest "timesfm[torch]==3.0.0"
```

Qué hacer si algo cambió:

- **Nueva versión de `timesfm` (>3.0.2)**: leer su CHANGELOG y el diff de `src/timesfm/timesfm_2p5/` y `src/timesfm3/torch/` desde v3.0.2; recrear los dos entornos principales y re-ejecutar contrato, smoke, `evidence/timesfm25/compare_direct.py` y `evidence/timesfm3/compare_direct.py` antes de aplicar el plan. Si la diferencia frente a la llamada directa deja de ser 0.0, parar y abrir una revisión nueva.
- **#531 publicado** (mínimo `huggingface_hub>=0.30`): la rama del `TypeError` de `paper_url` de 3.1 sigue siendo útil para quien tenga instalada una versión de timesfm anterior; mantenerla.
- **#552 publicado** (`timesfm.__getattr__` lanza ImportError): el `getattr` dentro de `try/except (AttributeError, ImportError)` de 3.1 ya lo cubre.
- **#549 publicado** (`use_sdpa`): no cambia este plan (TFM3-P4 queda pospuesta, D7).
- **#486 publicado** (contexto todo NaN): no cambia este plan; la comprobación central es de `02_common.md`.
- **#487 publicado** (`window_size`): documentar `window_size` como efectivo en lugar de "sin efecto" (3.5).
- **Checkpoint nuevo** (por ejemplo TimesFM 3.5 o una variante 2.5 nueva): el prefijo `google/timesfm-3.0` no enruta ids `3.5` (`google/timesfm-3.5-pytorch` da `ValueError` "No adapter found") [probado: `evidence/timesfm3/compare_direct.log`, sección "Registry routing"]; añadir soporte queda fuera de este PR y exige su propia revisión.

## 2. Hallazgos

### TimesFM 2.5 (`TimesFM25Adapter`)

- **T25-F1** [probado] timesfm 3.0.2 es la última versión; master no toca `src/timesfm` desde v3.0.2. Contrato 6 passed, smoke 5 passed y 1 skipped (exog no soportado), unitarios 51 passed (58 en la selección de la verificación). Evidencia: `evidence/timesfm25/pypi_timesfm.log`, `evidence/timesfm25/upstream_diff.log`, `evidence/timesfm25/contract.log`, `evidence/timesfm25/smoke.log`, `evidence/timesfm25/unit_tests.log`, `evidence/timesfm25/ci_baseline_job_127875168998.log`.
- **T25-F2 (corregido)** [verificado en código] Las firmas que usa el adaptador (`from_pretrained`, `_from_pretrained`, `compile`, `forecast`, campos de `ForecastConfig`) no cambian entre v2.0.1 y v3.0.2. Entre v2.0.1 y v2.0.2 entran además 1e89ad1 y f83f134 (torch.compile se aplica de verdad a `forward`, antes reasignaba un `self` local sin efecto, y se respeta el kwarg `torch_compile` de `load_checkpoint`) y 2011cfb (descarga previa de `config.json` en `_from_pretrained`); entre v2.0.2 y v3.0.2 solo entran af8f595 y el import de timesfm3. Evidencia: `evidence/timesfm25/upstream_diff.log`.
- **T25-F3** [probado] La salida del adaptador coincide exactamente (diferencia máxima 0.0 en puntual y cuantiles) con `TimesFM_2p5_200M_torch.forecast` directo, con el modelo cargado sin el workaround: en la primera pasada con 3 series de 400, 150 y 300 observaciones (una con NaN internos), `context_length=200`, `steps=24` y cuantiles 0.1, 0.5 y 0.9; en la verificación con 3 series diarias (una con 20 NaN iniciales y 10 internos, otra de 90 puntos con `context_length=320`), `steps=30` y los 9 cuantiles. Evidencia: `evidence/timesfm25/compare_direct.log` (sección 1), `evidence/timesfm25/verification/verify_direct.log` (sección 1).
- **T25-F4** [probado] La predicción puntual es el cuantil 0.5 (canal 5 de la salida de cuantiles, `decode_index=5`), no la media (canal 0, que difiere hasta 0.30). La skill oficial de timesfm (`timesfm-forecasting/SKILL.md` v3.0.2, línea 239) la llama "the median (0.5 quantile)" [documentado]. Dicen "media" tres textos de skforecast [verificado en código]: la tabla de la guía (celda 11, columna "Point forecast": "Mean (dedicated output)"), las Notes de `TimesFM25Adapter` y `skills/foundation-forecasting/references/adapter-parameters.md` línea 53 ("mean (in practice equal to quantile `0.5`)"). Evidencia: `evidence/timesfm25/verification/verify_direct.log` (sección 2), `evidence/timesfm25/compare_direct.log` (sección 2).
- **T25-F5** [probado] La subclase `_TimesFMCompat` de `TimesFM25Adapter._load_model` (absorbe `proxies` y `resume_download`) solo hace falta con timesfm 2.0.0 y 0.30 <= huggingface_hub < 1.0: con huggingface_hub < 0.30 ya falla `import timesfm` (TV-1; `paper_url` también está en la rueda 2.0.0 [verificado en código]). Sin ella, timesfm 2.0.0 con huggingface_hub 0.36.0 da `TypeError ... unexpected keyword argument 'proxies'`; con 2.0.1 + 0.36.0, 3.0.2 + 0.36.0, 3.0.2 + 1.0.0 y 3.0.2 + 2.4.0 la carga directa funciona. Arreglado upstream en 2f1625c (PR #435, fusionado 2026-06-08, desde v2.0.1: `__init__` acepta `**kwargs`). Evidencia: `evidence/timesfm25/check_workaround.log`, `evidence/timesfm25/verification/rerun_workaround.log`.
- **T25-F6** [probado] Con timesfm 2.0.x y `force_flip_invariance=True` (valor por defecto de `ForecastConfig`, el que usa el adaptador), los cuantiles de los pasos >128 (parte autorregresiva) salen colapsados y desordenados: con timesfm 2.0.2 y `steps=384`, el orden q0.1 <= q0.5 <= q0.9 se cumple en el 47 % y el 50 % de los pasos 129-384 de dos series, con anchura media q0.9-q0.1 de 0.024 y 0.046 (frente a 2.941 y 1.260 en los pasos 1-128). Con 3.0.2 están ordenados el 100 % (anchura 0.853 y 0.922). Arreglado en af8f595 (PR #463, merge 3dae50b, solo en v3.0.0 y posteriores). `backend_package` no declara mínimo y el smoke (`STEPS = 12`) no lo detectaría. Evidencia: `evidence/timesfm25/check_flip_ar_quantiles.log`, `evidence/timesfm25/verification/rerun_flip_ar_quantiles.log`.
- **T25-F7** [probado] Sin el extra torch, `import timesfm` funciona (el import de 2.5 está dentro de `try/except ImportError: pass` en `timesfm/__init__.py`) y el adaptador falla con `AttributeError: module 'timesfm' has no attribute 'TimesFM_2p5_200M_torch'` en lugar de un ImportError con la instrucción de instalación. Con timesfm 1.x ocurre lo mismo (1.3.0 no tiene `TimesFM_2p5_200M_torch` ni `ForecastConfig`) [verificado en código]. `TimesFM3Adapter` ya distingue estos casos. Evidencia: `evidence/timesfm25/check_import_errors.log`, `evidence/timesfm25/verification/rerun_import_errors.log`.
- **T25-F8** [probado] El prefijo `google/timesfm-2.5` envía a este adaptador `google/timesfm-2.5-200m-pytorch` (funciona), `-flax` (`RemoteEntryNotFoundError` 404, no tiene `model.safetensors`) y `-transformers` (su cabecera safetensors tiene 272 tensores y faltan 228 de los 232 que espera el módulo). El fallo de `load_state_dict(strict=True)` de `-transformers` tras descargar 925 MB está [verificado en código], no ejecutado. `timesfm-1.0` y `timesfm-2.0` se rechazan y `timesfm-3.0` va a `TimesFM3Adapter`. Los tres repos 2.5 son apache-2.0 y no gated. Evidencia: `evidence/timesfm25/registry_and_loading.log`, `evidence/timesfm25/verification/verify_registry_checkpoints.log`.
- **T25-F9** [probado] El checkpoint `-pytorch` no cambia desde 2025-10-02 (pesos en d418f3e, 2025-10-01). Su `config.json` trae `torch_compile: false`, que el hub mixin inyecta en `__init__`: la instancia cargada no compila aunque el valor por defecto de la clase es `True`. El adaptador no lo fija, así que el comportamiento depende de la revisión del Hub. Evidencia: `evidence/timesfm25/hub_checks.log`, `evidence/timesfm25/compare_direct.log` (sección 3), `evidence/timesfm25/verification/verify_direct.log` (sección 3).
- **T25-F10** [probado] `forecast_config_kwargs` se pasa sin validar: `return_backcast=True` rompe la forma de la salida (`ValueError: could not broadcast input array from shape (318,1) into shape (30,1)` en `FoundationModel`, con `context_length=320`; 216 frente a 24 filas en la primera pasada) y `max_context` da `TypeError ... got multiple values for keyword argument 'max_context'` en el primer `predict`, no en `__init__`. El docstring solo lista 5 claves y no menciona `per_core_batch_size`. `window_size` existe en `ForecastConfig` pero es un TODO upstream (`configs.py` v3.0.2; PR #487 abierto) [verificado en código]. Evidencia: `evidence/timesfm25/verification/verify_direct.log` (sección 4), `evidence/timesfm25/compare_direct.log` (sección 4).
- **T25-F11** [probado] `compile` redondea `max_context` a múltiplos de 32 y `max_horizon` a múltiplos de 128, y exige que la suma no supere 16384: `context_length=16384` falla en el primer `predict` con "Context + horizon must be less than the context limit. 16384 + 128 > 16384"; 16000 funciona. Con `use_continuous_quantile_head` el horizonte compilado se limita a 1024 [verificado en código: `compile()` en `timesfm_2p5_torch.py` v3.0.2, líneas 393-416]. La guía dice "Max context length 16384" sin ese matiz. Evidencia: `evidence/timesfm25/verification/verify_direct.log` (sección 4).
- **T25-F12** [verificado en código] timesfm ofrece covariables para TimesFM 2.5 vía XReg (`forecast_with_covariates`, desde 0d4d514, 2025-10-28; presente en la rueda 2.0.0), no nativas: regresión lineal en JAX (`jnp.linalg.pinv` en `BatchedInContextXRegLinear.fit`) con un único beta para todas las series de la llamada, así que la predicción de una serie depende de las demás del lote; admite covariables dinámicas numéricas y categóricas y estáticas, pero no covariables solo de pasado; requiere `timesfm[xreg]` (`jax[cuda]`, scikit-learn) y `return_backcast=True` (comprobación en `timesfm_2p5_base.py` v3.0.2, línea 240). El commit 6ae67d4 revirtió una regresión por serie. Issues abiertas #338 (fuga de datos) y #274 (dependencia del lote); PR abierto #550 [documentado]. No ejecutado. Evidencia: `evidence/timesfm25/upstream_issues.log`, `evidence/timesfm25/upstream_issue_bodies.log`.
- **T25-F13** [verificado en código] El backend elige `cuda:0` si `torch.cuda.is_available()` y si no CPU (`timesfm_2p5_torch.py` v3.0.2, líneas 72-77), sin argumento de dispositivo (no se puede forzar CPU en una máquina con CUDA), sin MPS (PR #415 cerrado sin fusionar el 2026-10-31 [documentado]) y siempre en float32 (líneas 428-430). El adaptador no expone `device`, coherente con el backend. Evidencia: `evidence/timesfm25/upstream_issues.log`.
- **T25-F14** [probado] `per_core_batch_size` vale 1 por defecto (`forecast` llama al decode compilado una vez por grupo de `global_batch_size`, `timesfm_2p5_base.py` v3.0.2 líneas 167-190 [verificado en código]); con 4 la salida es igual salvo ruido float32 (1.5e-5 en la verificación con 5 series, 7.6e-6 en la primera pasada con otros datos). Ya se puede pasar por `forecast_config_kwargs`, pero no está documentado. Sin medidas de tiempo. Evidencia: `evidence/timesfm25/verification/verify_batch_size.log`, `evidence/timesfm25/compare_direct.log` (sección 5).
- **T25-F15** [verificado en código] Los atributos de capacidad y licencia son veraces con 3.0.2: `SUPPORTED_QUANTILES` coincide con `TimesFM_2p5_200M_Definition.quantiles`; `supports_nan_in_series=True` (el backend quita los NaN iniciales e interpola los internos; probado también con `Float64` y `pd.NA`, salida idéntica [probado]); `requires_equal_length_context=False` (el backend rellena con ceros y máscara); Apache-2.0 en `_MODEL_LICENSES` (clave `google/timesfm-2.5`, `skforecast/foundation/_utils.py` aprox. línea 282 en d9b8f4f), igual que las fichas del Hub y el README upstream ("weights up to version 2.5 remain Apache-2.0"); `requires_hf_auth=False`, `requires_provider_auth=False`, `weights_repo_id=None`, `weights_in_hf_cache=True`. Salvedad: `supports_nan_in_series=True` tiene el caso límite TV-2 (contexto recortado todo NaN), que resuelve `02_common.md`. Evidencia: `evidence/timesfm25/registry_and_loading.log`, `evidence/timesfm25/verification/verify_direct.log` (sección 5).
- **T25-F16** [verificado en código] El adaptador cumple `_AdapterBase` y `.claude/rules/foundation.md`: no rellena, no agrupa ni recorta (`FoundationModel` recorta y el backend rellena internamente); su único bucle por serie construye el diccionario de salida a partir de una sola llamada `forecast`, y `predict` devuelve `(steps, n_quantiles)` por serie.
- **T25-F17** [verificado en código] Huecos de test: `FakeTimesFM25Model` (`fixtures_adapters.py`) devuelve ceros como puntual y `i/10` en el canal `i`, así que no reproduce que la puntual sea el canal 5; su `_FakeForecastConfig.max_horizon=16384` no es realista (está puesto para que `_ensure_compiled` no importe timesfm); no hay tests de claves reservadas de `forecast_config_kwargs` (`return_backcast`, `max_context`); `contracts.py` comprueba `_from_pretrained` solo por el override; `test_smoke.py` usa `STEPS = 12`, sin parte autorregresiva (>128 pasos) ni comprobación de que la puntual sea q0.5. Consecuencia: T25-F6 y un cambio de canales pasarían hoy el CI.

### TimesFM 3 (`TimesFM3Adapter`)

- **TFM3-F1** [probado] 3.0.2 es la última versión (3.0.0 2026-08-28, 3.0.1 2026-09-02 sin tag git y cuya rueda no contiene la corrección #483, 3.0.2 2026-09-09). El extra `[torch]` exige `torch>=2.0.0` y `requires_python>=3.10`, compatible con skforecast. Contrato 3 passed, smoke 6 passed, unitarios 57 passed. Evidencia: `evidence/timesfm3/ci_baseline.log`, `evidence/timesfm3/verification/contract_smoke_verify.log`, `evidence/timesfm3/verification/unit_tests_verify.log`.
- **TFM3-F2** [probado] La API usada (`TimesFM3Forecaster.from_pretrained(..., device=...)`, kwargs de `predict_batch`, `ForecastOutput.forecast` y `.quantiles`, `config.quantiles`) es igual en 3.0.0, 3.0.2 y master; el código pasó a `timesfm3/torch/` con un shim [verificado en código]. Las correcciones de 3.0.2 no son alcanzables por el adaptador [verificado en código]: #483 (commit 0ac3663; alineación de covariables con padding `"none"` y contexto >15360, y placeholders en lotes mixtos) porque el adaptador usa `"edge"` con covariables y grupos homogéneos; #492 (commit 8680259; KV cache) porque `TimesFM3Torch.forward` no pasa `decode_cache`. La salida coincide exactamente (maxdiff 0.0) con `predict_batch` directo sin exog, multiserie de distinta longitud y con covariables pasadas y futuras; con las columnas de covariables en otro orden una llamada directa difiere ~1e-3 sobre valores de ~5600 (0.0009765625 en el pliegue 0 de la guía), porque el adaptador las ordena con `get_exog_signature`; con el mismo orden la igualdad es exacta. Evidencia: `evidence/timesfm3/compare_direct.log` (secciones A, C y D), `evidence/timesfm3/verification/compare_direct_rerun.log`, `evidence/timesfm3/verification/check_guide_fold0.log`.
- **TFM3-F3** [probado] Con timesfm>=3.0.2, `import timesfm3` funciona sin torch (`__getattr__` perezoso, PEP 562): la rama "torch is missing" de `TimesFM3Adapter._load_model` es código muerto y el usuario ve el mensaje genérico "does not provide `TimesFM3Forecaster`. Reinstall with `pip install -U "timesfm[torch]"`". Con 3.0.0 sí sale "torch is missing". La sonda `from timesfm3 import TimesFM3Forecaster` saca el ImportError real ("import of torch halted") en 3.0.0 y 3.0.2. El test `test_TimesFM3Adapter_load_model_ImportError_when_torch_missing` simula el comportamiento de 3.0.0 (`sys.modules["timesfm3"] = None`), así que el fake diverge del backend real. Con el PR abierto #552, `hasattr` dejaría propagar el ImportError de upstream, cuyo mensaje es accionable (`pip install "timesfm[torch]"`): impacto cosmético. Evidencia: `evidence/timesfm3/check_import_errors.log`, `evidence/timesfm3/check_import_errors_v300.log`, `evidence/timesfm3/check_import_fix.log`, `evidence/timesfm3/verification/check_p1_probe.log`.
- **TFM3-F4 (corregido)** [verificado en código] El docstring de `TimesFM3Adapter.predict` dice que `padding_mode="edge"` "matches the default used by TimesFM's own `predict()`": es falso, el valor por defecto de `predict` y `predict_batch` es `"none"` en 3.0.0, 3.0.1, 3.0.2 y master. `TimesFM3Evaluator` usa `"none"` por defecto pero no lo impone: reenvía el argumento a `predict_batch` (`evaluator.py` v3.0.2, líneas 58, 97, 195 y 257); los ejemplos del README no pasan `padding_mode`, así que usan `"none"`. La elección cambia la salida con covariables futuras (maxdiff 0.93, 1.6 % relativo) [probado]. El único efecto protector real de `"edge"` es evitar #483 con timesfm<3.0.2 y contextos >15360. `padding_mode` está en `_RESERVED_PREDICT_KWARGS`, así que el usuario no puede elegirlo. Evidencia: `evidence/timesfm3/compare_direct.log` (sección D), `evidence/timesfm3/verification/compare_direct_rerun.log`.
- **TFM3-F5** [probado] La predicción de una serie depende de la longitud de las demás series del mismo trozo de `per_core_batch_size` (4 por defecto; relleno a la izquierda con máscara hasta el contexto máximo del trozo, redondeado a 32): con la misma longitud la diferencia es 1e-5; una serie de 100 sola frente a acompañada por una de 2048 difiere 0.32 (0.6 %); a través de `FoundationModel`, con series de 2048, 1000, 300 y 100 observaciones sin exog, `predict(levels=[x])` frente a `predict()` filtrado difiere hasta 0.20 (0.39 %). La frase "(`predict(levels=[x])` equals `predict()` filtered to `x`)" solo está en `skills/foundation-forecasting/references/adapter-parameters.md` línea 82; las Notes del docstring y `llms-base.txt` (línea 531) solo afirman la independencia respecto al exog de las otras series, que es correcta (entradas idénticas) [verificado en código]. Evidencia: `evidence/timesfm3/batch_invariance.log`, `evidence/timesfm3/verification/batch_invariance_rerun.log`, `evidence/timesfm3/compare_direct.log` (sección C).
- **TFM3-F6** [probado] Único checkpoint oficial 3.x: `google/timesfm-3.0-pytorch` (sha 43046b85, mismo blob de pesos desde 2026-08-24). Sin 3.5, flax/JAX (issue #475 abierta [documentado]) ni otros tamaños; las copias de terceros (ONNX, GGUF, resubidas) se rechazan por el prefijo. El prefijo `google/timesfm-3.0` también enruta ids inexistentes como `google/timesfm-3.0-flax` (`02_common.md`, X-3, lo registra y decide no añadirlo a la tabla de ids no soportados: el 404 del Hub ya es claro). Evidencia: `evidence/timesfm3/hub_google_timesfm_recheck.log`, `evidence/timesfm3/hub_commits_timesfm3.log`, `evidence/timesfm3/verification/hub_commits_verify.log`.
- **TFM3-F7** [probado] La licencia de la ficha (`other`, `timesfm-non-commercial-license-v1.0`, `license_link` LICENSE, no gated) coincide con `_MODEL_LICENSES["google/timesfm-3.0"]` y con `requires_hf_auth=False`. El texto del LICENSE (sha256 `3e36db7240d2...`) limita el uso a fines no comerciales y no de producción y prohíbe usar las salidas en decisiones comerciales; el README upstream indica que el código es Apache-2.0 y que el uso comercial está disponible vía BigQuery ML [documentado]. El `LicenseWarning` ya dice "commercial or production use", pero `SKILL.md` y `llms-base.txt` solo hablan de "restrict commercial use". Evidencia: `evidence/timesfm3/hub_google_timesfm_recheck.log`.
- **TFM3-F8** [probado] Atributos de capacidad veraces con 3.0.2: `SUPPORTED_QUANTILES` coincide con `config.quantiles` (mediana en el índice 4), `global_context` 15360, sin techo de horizonte (1, 63, 64, 65 y 200 pasos; con exog hasta 100), NaN iniciales, internos y finales, covariables int y bool aceptadas y category rechazada con ValueError, 40 covariables, series de distinta longitud sin relleno previo; `weights_in_hf_cache=True` correcto. `supports_heterogeneous_covariates=False` debe mantenerse: en lotes mixtos el backend usa placeholders `np.zeros_like` sin máscara (PR #546 abierto) y puede fallar al concatenar horizontes (PR #515 abierto) [documentado]. Evidencia: `evidence/timesfm3/compare_direct.log` (secciones A, B, D, E, F y G), `evidence/timesfm3/verification/compare_direct_rerun.log`.
- **TFM3-F9** [verificado en código] No hay código MPS upstream (`git grep -i mps` vacío en `src/timesfm3/torch` v3.0.2); el comportamiento en MPS y CUDA no se pudo comprobar (sin GPU ni Apple Silicon). `_ModelConfig` no tiene dtype y convertir a mano los pesos a bfloat16 falla en CPU ("expected m1 and m2 to have the same dtype") [probado: `evidence/timesfm3/compare_direct.log`, sección H], así que no exponer dtype es coherente. 3.0.2 añade un backend MLX (`timesfm3.mlx`, extra `[mlx]`, misma API) que el adaptador no usa.
- **TFM3-F10 (corregido)** [verificado en código] `TimesFM3Forecaster.from_pretrained(**kwargs)` respeta `per_core_batch_size`, `revision`, `cache_dir`, `local_files_only`, `token` y `force_download`, que el adaptador no transmite (solo se pueden usar con un `model=` precargado). `use_sdpa` no tiene efecto en 3.0.2 (no se lee en `timesfm3_forecaster.py`, solo se define en la línea 86; PR #549 abierto) y se descarta; otros campos de arquitectura (`use_variate_attention`, etc.) los sobrescribe `config.json` (#502, #505). `per_core_batch_size` cambia las predicciones multiserie (TFM3-F5), así que no es una opción de carga inocua. `predict_kwargs` ya cubre `use_symmetric_averaging`, `make_positive`, `use_znorm` y `sort_quantiles`; `TimesFM3Evaluator` usa `use_symmetric_averaging=True` y `make_positive=True`, el adaptador los valores por defecto de `TimesFM3Forecaster` (`False`). Los objetivos multivariantes (contextos 2-D) no se usan.
- **TFM3-F11** [verificado en código] El adaptador cumple `_AdapterBase` y `.claude/rules/foundation.md`: no rellena, no agrupa ni itera para resolver heterogeneidad; el bucle por serie solo traduce covariables y depende de que `FoundationModel` agrupe por `get_exog_signature`; `predict` devuelve `(steps, n_quantiles)` por serie. `contracts.py` cubre `from_pretrained(device)` y los kwargs de `predict_batch`; por diseño no cubre `config.quantiles` ni los campos de `ForecastOutput`, que ejercitan los smoke tests. Evidencia: `evidence/timesfm3/contract.log`, `evidence/timesfm3/smoke.log`.
- **TFM3-F12** [probado] La guía sigue siendo reproducible: MAE de TimesFM 3.0 con exog 162.4352 con timesfm 3.0.2 (la celda 64 muestra 162.435209), pliegues 0 y 29 reproducidos exactamente (maxdiff 0.0); la fila de la tabla de la guía coincide con el código. "roughly 15,360" es en realidad exacto (`_MAX_CONTEXT_LENGTH=15360`) y el backend trunca sin aviso los contextos más largos (`_Query.format` y `batch_context`) [verificado en código]. Evidencia: `evidence/timesfm3/backtest_guide_exog.log`, `evidence/timesfm3/verification/check_guide_fold0.log`.

### Transversales que posee este plan

- **TV-1** [probado con huggingface_hub 0.29.3] timesfm 3.0.2 declara `huggingface_hub>=0.28.0`, pero con 0.28 o 0.29 `import timesfm` lanza `TypeError: ModelHubMixin.__init_subclass__() got an unexpected keyword argument 'paper_url'`, que `_import_timesfm` no captura (solo captura ImportError): lo ven `TimesFM25Adapter` y `TimesFM3Adapter` (en `FoundationModel.predict`). timesfm 2.0.0 tiene el mismo argumento `paper_url` [verificado en código]. PR upstream #531 (sube el mínimo a 0.30.0) abierto [documentado]. Prioridad baja. Evidencia: `evidence/timesfm25/verification/hfhub_029_import.log`.
- **TV-3** [verificado en código] Con un `model` precompilado cuyo `forecast_config.max_context` es menor que `context_length`, el backend recorta el contexto en silencio a `max_context` (`value = value[-context:]` en `forecast`, `timesfm_2p5_base.py` v3.0.2); `TimesFM25Adapter._ensure_compiled` solo compara `max_horizon`. Prioridad baja.

Relacionados, resueltos en otro plan: **TV-2** (contexto recortado todo NaN: `ValueError` críptico "The truth value of an empty array is ambiguous" con numpy>=2.2, o predicción desde ceros sin aviso con numpy<2.2; PR upstream #486 abierto; evidencia `evidence/timesfm25/verification/verify_all_nan_context.log`, `evidence/timesfm25/verification/verify_all_nan_old_numpy.log`) y **X-3** (ids aceptados por prefijo que no cargan) van en `02_common.md`.

### Descartado

- T25-F2 en su forma original: "los únicos cambios entre v2.0.1 y v3.0.2 son af8f595 y el import de timesfm3" (también entran 1e89ad1, f83f134 y 2011cfb en v2.0.2) [verificado en código].
- T25-P1 en su forma original: "misma cota que `TimesFM3Adapter`" (`TimesFM3Adapter` declara `timesfm[torch]` sin cota y solo comprueba la versión en tiempo de ejecución cuando falta `TimesFM3Forecaster`) [verificado en código].
- TFM3-F4: "`TimesFM3Evaluator` impone `padding_mode="none"`" (solo lo usa por defecto y reenvía el argumento) [verificado en código].
- TFM3-F5 / TFM3-P3: que la frase de invariancia esté en las Notes del docstring (solo está en `adapter-parameters.md` línea 82) [verificado en código].
- TFM3-F10: `use_sdpa` como opción útil de carga (no tiene efecto en 3.0.2) [verificado en código].

Preguntas abiertas (no bloquean este PR): causa interna exacta de TFM3-F5 (solo se midió el efecto); si conviene recomendar la `ForecastConfig` del ejemplo de la ficha de TimesFM 2.5 (cuestión de precisión, fuera de alcance).

## 3. Cambios a implementar

Orden sugerido de commits: el test de smoke de 3.6 primero (para poder comprobar que falla con timesfm 2.0.2 antes de que exista el mínimo), luego 3.1 y 3.2 (helper e imports), 3.3 (mínimo y fin del workaround), 3.5 y 3.9 (validación y opciones de TimesFM 2.5), 3.10 a 3.12 (TimesFM 3), el resto de 3.6 (fake y contrato) y por último los textos (3.4, 3.7, 3.11, 3.13).

### 3.1 T25-P5 (con TFM3-P1 y TV-1): helper común de import de timesfm

- **Qué**: convertir `_import_timesfm` (`skforecast/foundation/_adapters.py`, aprox. línea 562 en d9b8f4f) en el único punto de entrada al backend de los dos adaptadores. Debe, en este orden:
  1. importar `timesfm`; si lanza `ImportError`, mantener el mensaje actual ("timesfm is required for {adapter}. Install it with `pip install "{backend_package}"`", ya entrecomillado; lo comprueban `test_TimesFM25Adapter_load_model_ImportError_when_timesfm_not_installed` y `test_TimesFM3Adapter_load_model_ImportError_when_timesfm_not_installed_no_LicenseWarning`);
  2. capturar el `TypeError` cuyo texto contiene `paper_url` y relanzarlo como `ImportError` que pida `huggingface_hub>=0.30` (TV-1); cualquier otro `TypeError` se relanza sin tocar;
  3. comprobar la versión instalada de `timesfm` antes de la sonda (la verificación exige mantener la comprobación de versión antes de la sonda). Con D-VER (b), `_import_timesfm` llama a `_check_backend_min_version(adapter_cls)` (helper común de `02_common.md`, 3.4): el mínimo sale de `adapter_cls.backend_package` (por eso 3.3 declara la cota `>=3.0` en los dos adaptadores), el parámetro `min_version` del esbozo se elimina y el mensaje pasa a ser el del helper ("TimesFM3Adapter requires timesfm>=3.0, but ... is installed"). Con D-VER (a) o (c), si se pasa `min_version`, comprobar con `importlib.metadata.version("timesfm")` y la comparación del major que ya hace hoy `TimesFM3Adapter._load_model` (sin dependencias nuevas; basta para `>=3.0`) [verificado en código]. Si la versión no se puede leer (`PackageNotFoundError`), no bloquear: seguir con la comprobación del atributo [inferido: evita romper instalaciones desde código fuente y los tests unitarios, cuyo entorno no instala timesfm];
  4. si se pasa `attr`, obtenerlo con `getattr` dentro de `try/except (AttributeError, ImportError)` (cubre el PR #552; con #552 fusionado esta parte es defensiva);
  5. si falta, importar la sonda `(módulo, nombre)` con `getattr(importlib.import_module(módulo), nombre)` para sacar la causa real (normalmente torch ausente) y lanzar un `ImportError` con esa causa y `pip install "<backend_package>"`; si la sonda no falla, el mensaje genérico de reinstalación actual (`pip install -U "<backend_package>"`).
- **Sondas**:
  - TimesFM 3: `("timesfm3", "TimesFM3Forecaster")`, equivalente a `from timesfm3 import TimesFM3Forecaster` [probado: lanza el ImportError real en 3.0.0 y 3.0.2, `evidence/timesfm3/check_import_fix.log`, `evidence/timesfm3/verification/check_p1_probe.log`].
  - TimesFM 2.5: `("timesfm.timesfm_2p5.timesfm_2p5_torch", "TimesFM_2p5_200M_torch")`. `timesfm/__init__.py` v3.0.2 importa exactamente `from .timesfm_2p5 import timesfm_2p5_torch` dentro de `try/except ImportError: pass` [verificado en código], así que reimportarlo saca el ImportError de torch [inferido]; se comprueba con `evidence/timesfm25/check_import_errors.py`.
- **Esbozo de la firma** (ilustrativo, no es la implementación):

```python
def _import_timesfm(
    adapter_cls: type,
    attr: str | None = None,
    probe: tuple[str, str] | None = None,
    min_version: str | None = None,
) -> Any:
    try:
        import timesfm
    except ImportError as exc:
        raise ImportError(
            f"timesfm is required for {adapter_cls.__name__}. "
            f'Install it with `pip install "{adapter_cls.backend_package}"`.'
        ) from exc
    except TypeError as exc:
        if "paper_url" not in str(exc):
            raise
        raise ImportError(
            f"{adapter_cls.__name__} needs huggingface_hub>=0.30 to import timesfm. "
            'Upgrade it with `pip install -U "huggingface_hub>=0.30"`.'
        ) from exc
    ...  # min_version check (timesfm>=3.0, see D8), before the probe
    if attr is not None:
        try:
            getattr(timesfm, attr)
        except (AttributeError, ImportError):
            ...  # import `probe` to surface the real cause, then raise ImportError
    return timesfm
```

- Llamadas: `TimesFM25Adapter._load_model` y `TimesFM3Adapter._load_model` con `attr` y `probe`; `min_version="3.0"` (según D8) solo con D-VER (a)/(c): con D-VER (b) el argumento `min_version` del esbozo desaparece y `_import_timesfm` llama a `_check_backend_min_version(adapter_cls)`, que lee el mínimo de `backend_package`; `TimesFM25Adapter._ensure_compiled` sin argumentos opcionales (solo necesita `ForecastConfig`, y la versión ya se comprobó en `_load_model`).
- **Por qué**: T25-F7, TFM3-F3, TV-1.
- **Riesgo**: bajo, solo cambia mensajes de error en la carga [inferido]. La comprobación de versión pasa a ejecutarse siempre en `_load_model` (hoy en TimesFM 3 solo cuando falta la clase); `importlib.metadata.version` es barato [inferido].
- **Rompe compatibilidad**: no (salvo el mínimo de 3.3, que se decide en D8).
- **Decisión**: el umbral de versión depende de D8; el uso del helper de 02 depende de D-VER (de `02_common.md`).

### 3.2 TFM3-P1: sonda de torch ausente en `TimesFM3Adapter._load_model`

- **Qué**: sustituir en `TimesFM3Adapter._load_model` (aprox. líneas 1656-1703 en d9b8f4f) el bloque `if not hasattr(timesfm, "TimesFM3Forecaster")` (comprobación de versión con `importlib.metadata`, `import timesfm3` y los tres `ImportError`) por la llamada al helper de 3.1: `_import_timesfm(type(self), attr="TimesFM3Forecaster", probe=("timesfm3", "TimesFM3Forecaster"), min_version="3.0")` (con D-VER (b), sin `min_version`). Mantener:
  - el mensaje "TimesFM 3.0 requires `timesfm>=3.0`" con timesfm<3 (lo comprueba `test_TimesFM3Adapter_load_model_ImportError_when_timesfm_predates_3`) con D-VER (a)/(c); con D-VER (b) el mensaje pasa a ser el del helper común ("TimesFM3Adapter requires timesfm>=3.0, but ... is installed") y el `match` de ese test se actualiza a ese texto;
  - el texto "torch is missing" en el mensaje de la sonda, o actualizar el `match` de `test_TimesFM3Adapter_load_model_ImportError_when_torch_missing`;
  - que el `LicenseWarning` (`_warn_if_non_commercial`) se emita solo después de que el import tenga éxito.
  Actualizar las Notes de `_load_model`.
- **Por qué**: TFM3-F3.
- **Riesgo**: bajo; solo mensajes [probado: `evidence/timesfm3/verification/check_p1_probe.log`].
- **Rompe compatibilidad**: no.
- **Decisión**: ninguna.

### 3.3 T25-P1: `timesfm>=3.0` para TimesFM 2.5 y quitar `_TimesFMCompat`

- **Qué** (con la recomendación A de D8):
  - `TimesFM25Adapter.backend_package = "timesfm[torch]>=3.0"` (aprox. línea 722 en d9b8f4f) y la misma cadena en `[timesfm25]` de `tools/foundation_compat/adapters.toml` (`test_adapters_toml_entries_match_the_adapter_classes` exige que coincidan, y `test_adapters_toml_backend_package_is_a_PEP_508_requirement` la acepta [verificado en código]). Sería el primer `>=` en timesfm: `TimesFM3Adapter` declara `timesfm[torch]` sin cota [verificado en código]. Con D-VER (b) hay que declarar también `backend_package = "timesfm[torch]>=3.0"` en `TimesFM3Adapter` (aprox. línea 1195 en d9b8f4f) y en `[timesfm3]` de `adapters.toml`, porque el helper común de `02_common.md` (3.4) lee el mínimo de `backend_package` y no acepta una versión explícita: sin la cota no comprobaría nada para TimesFM 3 y `test_TimesFM3Adapter_load_model_ImportError_when_timesfm_predates_3` dejaría de tener código que lo lance [verificado en código]. Solo con D-VER (a) o (c), donde `_import_timesfm` conserva su propia comparación con `min_version`, la cota en `TimesFM3Adapter` es opcional [inferido].
  - En `TimesFM25Adapter._load_model` (aprox. líneas 959-1003): llamar al helper de 3.1 con `attr="TimesFM_2p5_200M_torch"`, la sonda de 3.1 y `min_version="3.0"`; eliminar la clase local `_TimesFMCompat` y su comentario, y cargar con `timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_id, torch_compile=False)` (el `torch_compile=False` es T25-P8, ver 3.9).
  - Con timesfm<3.0, `ImportError` que diga que hace falta `timesfm>=3.0` y por qué (cuantiles erróneos para horizontes >128 con 2.0.x, y sin TimesFM 2.5 con 1.x), con el comando `pip install -U "timesfm[torch]>=3.0"`.
  - Sub-decisión de D8: si se elige UserWarning en lugar de ImportError para 2.0.x, avisar en lugar de fallar; en ese caso `_TimesFMCompat` solo se puede quitar si 2.0.0 se sigue rechazando con ImportError (mínimo duro `>=2.0.1`).
  - En `tools/foundation_compat/tests/contracts.py`, quitar la entrada `{"target": "timesfm:TimesFM_2p5_200M_torch._from_pretrained"}` y su comentario ("Overridden by the adapter to absorb arguments of `huggingface_hub`").
- **Por qué**: T25-F5, T25-F6, T25-F7.
- **Riesgo**: bajo. Quitar el workaround no cambia predicciones: la carga directa sin workaround da una salida idéntica a la del adaptador (0.0) [probado: `evidence/timesfm25/verification/verify_direct.log`], así que no hace falta el backtest de la guía.
- **Rompe compatibilidad**: sí, para quien tenga timesfm<3.0 instalado (tendrá que actualizar). `backend_package` no es una dependencia de skforecast: la cota actúa en el mensaje del ImportError, en `get_model_info`, en `tools/foundation_compat/run.py`, en los docs y en la comprobación en tiempo de ejecución [verificado en código].
- **Decisión**: D8 (y D-VER de 02 para la forma de la comprobación). Con D8 B (`>=2.0.1`) la comparación ya no se resuelve con el major y hace falta comparar versiones completas [inferido].

### 3.4 T25-P2: "media" por "mediana (cuantil 0.5)"

- **Qué**: en las Notes de `TimesFM25Adapter` (aprox. líneas 689-691 en d9b8f4f), sustituir "The point forecast is documented by TimesFM as the mean (in practice the 2.5 checkpoint returns a value equal to quantile 0.5)" por una frase que diga que la puntual es la mediana (cuantil 0.5). Corregir también el comentario de `TimesFM25Adapter.predict` (aprox. línea 944: "idx 0 = mean, 1-9 = q0.1-q0.9") para añadir que `point_forecast` es el canal 5 (mediana). Textos de docs en la sección 5 (celda 11 de la guía y `adapter-parameters.md` línea 53).
- **Por qué**: T25-F4.
- **Riesgo**: nulo.
- **Rompe compatibilidad**: no.
- **Decisión**: ninguna.

### 3.5 T25-P4: validar `forecast_config_kwargs`

- **Qué**: siguiendo el patrón de `TimesFM3Adapter._RESERVED_PREDICT_KWARGS` y `TimesFM3Adapter._validate_predict_kwargs`, añadir a `TimesFM25Adapter` un atributo de clase (por ejemplo `_RESERVED_FORECAST_CONFIG_KWARGS = frozenset({"max_context", "max_horizon", "return_backcast"})`) y un `classmethod` de validación que lance `ValueError` nombrando las claves. Llamarlo en `__init__` y dentro de la función `validate` de `set_params` (donde hoy se normaliza `forecast_config_kwargs` a `{}`), sin copiar el diccionario (lo comprueba `test_TimesFM25Adapter_init_forecast_config_kwargs_stored_by_reference`). `window_size`: la verificación propone rechazarlo o documentarlo; se recomienda documentarlo como sin efecto (TODO upstream, PR #487) en lugar de rechazarlo, para no bloquearlo si upstream lo implementa [inferido]. Documentar `per_core_batch_size` en el docstring del parámetro (ver 3.9). No se recomienda rechazar claves desconocidas con una lista fija de campos de `ForecastConfig` (se desfasaría con cada versión) [inferido].
- **Por qué**: T25-F10.
- **Riesgo**: bajo.
- **Rompe compatibilidad**: solo configuraciones que ya fallaban [probado: `evidence/timesfm25/verification/verify_direct.log`, sección 4].
- **Decisión**: ninguna.

### 3.6 T25-P6: smoke con horizonte >128 y fake fiel

- **Qué**:
  - En `tools/foundation_compat/tests/test_smoke.py`, un test nuevo con `steps` >128 (por ejemplo 200; `N_OBSERVATIONS = 300` y el `max_horizon` por defecto de 512 lo permiten) que compruebe con `predict(quantiles=[0.1, 0.5, 0.9])` que los cuantiles están ordenados en todos los pasos y que `predict` sin cuantiles coincide con la columna q0.5. Limitarlo a `timesfm25` (y opcionalmente `timesfm3`) mediante `backend_config["adapter"]` y `pytest.skip`: otros backends pueden cruzar cuantiles sin que sea un defecto y el horizonte largo encarece el smoke [inferido]. El fixture `model` usa `CONTEXT_LENGTH = 64`; el criterio se probó con `context_length=512` y `steps=384`, así que su poder de detección con los valores del smoke es [inferido] y debe comprobarse (sección 4).
  - En `skforecast/foundation/tests/tests_foundation_models/fixtures_adapters.py`, `FakeTimesFM25Model.forecast` debe devolver como puntual el canal 5 de la salida de cuantiles (0.5) y `from_pretrained` debe aceptar `**kwargs` (para `torch_compile=False`). `_FakeForecastConfig` necesita un atributo `max_context` (el aviso de TV-3 en `_ensure_compiled` lo lee; hoy solo tiene `max_horizon`). Si se baja `max_horizon` de 16384 a un valor realista (por ejemplo 1024), debe seguir siendo mayor que cualquier `steps` de los tests, porque si no `_ensure_compiled` intenta importar timesfm (motivo documentado en el docstring del fake) [verificado en código].
- **Por qué**: T25-F17 (y la detección de T25-F6 y T25-F4 en CI).
- **Riesgo**: algo más de tiempo de smoke en CI [inferido].
- **Rompe compatibilidad**: no. El criterio de orden de cuantiles falla con timesfm 2.0.2 (47 % a 50 % de pasos ordenados) y pasa con 3.0.2 (100 %) [probado: `evidence/timesfm25/verification/rerun_flip_ar_quantiles.log`]; la puntual igual al canal 5 es válida con 3.0.2 [probado].
- **Decisión**: ninguna. La actualización del workflow del CI de compatibilidad, si hiciera falta, es de `10_docs_ci.md`.

### 3.7 T25-P7: XReg de TimesFM 2.5, no implementarlo y matizar los docs

- **Qué** (con la recomendación A de D10): mantener `allow_exog=False` y añadir en las Notes de `TimesFM25Adapter` y en los docs (sección 5) que el modelo no admite covariables de forma nativa; la librería ofrece XReg (`forecast_with_covariates`), una regresión lineal externa en JAX, común a todo el lote (la predicción de una serie dependería de las demás), con el extra `timesfm[xreg]` (`jax[cuda]`), que skforecast no usa.
- **Por qué**: T25-F12.
- **Riesgo**: nulo con A; alto con B.
- **Rompe compatibilidad**: no.
- **Decisión**: D10. Con B (implementar XReg: `allow_exog=True`, `supports_past_only_covariates=False`, `return_backcast` interno, extra opcional `timesfm[xreg]`, elegir el `xreg_mode` por defecto y aceptar la dependencia del lote) haría falta un PR propio (fuera de alcance aquí).

### 3.8 T25-P3: prefijo de TimesFM 2.5 (se implementa en `02_common.md`)

- **Qué**: no se implementa en este PR. `02_common.md` añade la tabla central de ids no soportados (D-REG (a), con `google/timesfm-2.5-200m-flax` y `google/timesfm-2.5-200m-transformers`; `google/timesfm-3.0-flax` queda fuera de la tabla porque no existe en el Hub y el 404 del Hub ya es claro, ver 02 "Descartado") o, si se prefiere para TimesFM 2.5, el prefijo exacto `google/timesfm-2.5-200m-pytorch` en `_ADAPTER_REGISTRY` (aprox. línea 5895 en d9b8f4f) y `TimesFM25Adapter._MODEL_ID_PREFIX`. La clave de licencia `google/timesfm-2.5` sigue coincidiendo porque `_resolve_adapter` y `_get_model_license` usan `startswith` [verificado en código].
- **Coordinación**: si 02 se fusiona antes, este PR no toca la columna de prefijos de `SKILL.md` (línea 166) ni de `llms-base.txt` (línea 489). El test `test_TimesFM25Adapter_set_params_updates_and_resets_model` usa `"google/timesfm-2.5-200m-pytorch-v2"` (aprox. línea 206), que empieza también por el prefijo exacto, así que no habría que cambiarlo [verificado en código].
- **Por qué**: T25-F8.
- **Riesgo** (en 02): bajo; un checkpoint 2.5 futuro con otro nombre exigiría ampliar el registro.
- **Rompe compatibilidad**: no para los ids que hoy funcionan.
- **Decisión**: D9 (y D-REG de 02).

### 3.9 T25-P8 (con TV-3 y T25-F11): opciones del backend no usadas

- **Qué**:
  - Pasar `torch_compile=False` explícito en `TimesFM_2p5_200M_torch.from_pretrained` dentro de `_load_model`, para no depender del `config.json` del Hub. Es seguro: los kwargs de `from_pretrained` prevalecen sobre `config.json` (con `torch_compile=True` la instancia sale compilada [probado: `evidence/timesfm25/compare_direct.log`, sección 3]) y la instancia ya carga hoy con `torch_compile=False`, así que las predicciones no cambian [probado: `evidence/timesfm25/verification/rerun_workaround.log`, `evidence/timesfm25/verification/verify_direct.log` sección 3]. Añadir a `contracts.py` la entrada `{"target": "timesfm:TimesFM_2p5_200M_torch", "kwargs": ["torch_compile"]}` (una clase representa su constructor, que declara `torch_compile: bool = True` desde 2.0.0 [probado: firma impresa en `evidence/timesfm25/verification/rerun_workaround.log`]).
  - Documentar en el docstring de `forecast_config_kwargs` (clase y `__init__`) la clave `per_core_batch_size` (T25-F14) y las claves reservadas de 3.5.
  - Documentar en las Notes de `TimesFM25Adapter` que el backend usa CUDA si está disponible y si no CPU, sin MPS ni forma de forzar CPU, siempre en float32 (T25-F13), y que `context_length` más el horizonte compilado (redondeados a 32 y 128) no puede superar 16384 (T25-F11).
  - TV-3: en `TimesFM25Adapter._ensure_compiled` (aprox. líneas 1005-1051), cuando no se recompila y `forecast_config.max_context < context_length` (solo puede ocurrir con un modelo precompilado por el usuario: cuando compila el adaptador, `max_context` se redondea al alza), emitir un `IgnoredArgumentWarning` (`skforecast.exceptions`) que diga que el contexto efectivo es `max_context`; emitirlo una sola vez por modelo cargado para no repetirlo en cada pliegue del backtesting [inferido]. Documentarlo también en el parámetro `model`. No recompilar el modelo del usuario.
- **Por qué**: T25-F9, T25-F11, T25-F13, T25-F14, TV-3.
- **Riesgo**: bajo [inferido]. torch.compile no se activa en ningún caso.
- **Rompe compatibilidad**: no.
- **Decisión**: ninguna.

### 3.10 TFM3-P2: `padding_mode` de TimesFM 3

- **Qué**:
  - (a), siempre: corregir el docstring de `TimesFM3Adapter.predict` (Notes, aprox. líneas 1434-1444 en d9b8f4f). Quitar "matches the default used by TimesFM's own `predict()`" y decir que `"edge"` con covariables es una elección de skforecast, distinta del valor por defecto de `predict` y `predict_batch` en timesfm (`"none"`, también el que usa por defecto `TimesFM3Evaluator`), y que con timesfm<3.0.2 evita además el desalineamiento de covariables con contextos >15360 corregido en #483.
  - (c), si se adopta: quitar `"padding_mode"` de `TimesFM3Adapter._RESERVED_PREDICT_KWARGS` y tomarlo de `predict_kwargs` sin duplicarlo en la llamada (hoy se pasa explícito junto con `**self.predict_kwargs`, lo que daría `TypeError` por argumento duplicado [verificado en código]):

```python
predict_kwargs = dict(self.predict_kwargs)
padding_mode = predict_kwargs.pop("padding_mode", "edge" if has_covariates else "none")
results = self._model.predict_batch(..., padding_mode=padding_mode, **predict_kwargs)
```

  Actualizar el docstring de `predict_kwargs` (clase y `__init__`, que hoy lista `padding_mode` entre las claves prohibidas) y las Notes de `_validate_predict_kwargs`.
- **Por qué**: TFM3-F4.
- **Riesgo**: (a) y (c) ninguno sobre las predicciones por defecto; (b) cambia predicciones con covariables futuras respecto a 0.26 y 0.27, y con contextos >15360 exigiría timesfm>=3.0.2 [verificado en código].
- **Rompe compatibilidad**: (a) y (c) no; (b) sí (valores predichos con exog futura).
- **Decisión**: D6.

### 3.11 TFM3-P3: corregir la afirmación de invariancia del lote

- **Qué**: en `skills/foundation-forecasting/references/adapter-parameters.md` línea 82, quitar "(`predict(levels=[x])` equals `predict()` filtered to `x`)". En las Notes de la clase `TimesFM3Adapter` (aprox. líneas 1156-1168 en d9b8f4f) no hay que quitar nada: añadir el matiz de que `predict_batch` rellena cada trozo de `per_core_batch_size` series hasta la longitud de la más larga, de modo que la predicción de una serie puede variar ligeramente con la longitud y el orden de las otras series, aunque nunca recibe su exog. El docstring de `test_FoundationModel_TimesFM3_predict_levels_matches_batch_covariates` ya dice que comprueba las covariables recibidas; opcionalmente, añadir que no comprueba la salida.
- **Por qué**: TFM3-F5.
- **Riesgo**: nulo (solo texto).
- **Rompe compatibilidad**: no.
- **Decisión**: ninguna. La alternativa de que `FoundationModel` agrupe por longitud de contexto (atributo de adaptador) no se propone ahora: cambiaría predicciones multiserie y multiplicaría llamadas; requeriría decisión del mantenedor [inferido].

### 3.12 TFM3-P4: opciones de carga de TimesFM 3 (pospuesta)

- **Qué**: con la recomendación (a) de D7, no se cambia código en este PR: `model=` precargado ya permite `revision`, `cache_dir`, `local_files_only`, `token`, `force_download` y `per_core_batch_size`. Si se elige (b) (`model_kwargs` reenviado a `TimesFM3Forecaster.from_pretrained`, incluido en `get_params` y en las claves que reinician el modelo en `set_params`) o (c) (solo `revision`), sería un PR posterior y común a todos los adaptadores; no incluir `use_sdpa` (sin efecto en 3.0.2) y tener en cuenta que `per_core_batch_size` cambia predicciones multiserie (TFM3-F5). Prueba prevista entonces: fake de `from_pretrained` que registre los kwargs y smoke con `revision="43046b85ec22d584a13f8098c2ed39c889e129c2"`.
- **Por qué**: TFM3-F10.
- **Riesgo**: bajo con (b)/(c) (parámetro opcional nuevo) [inferido].
- **Rompe compatibilidad**: no.
- **Decisión**: D7.

### 3.13 TFM3-P5: textos de licencia y de contexto máximo de TimesFM 3

- **Qué**: decir "uso no comercial y no de producción" para TimesFM 3.0 (con la nota de que Google ofrece uso comercial vía BigQuery ML según el README upstream) en `SKILL.md`, `adapter-parameters.md` y `llms-base.txt`; sustituir "roughly 15,360" por "15,360; longer contexts are truncated by the backend" en el docstring de `context_length` de `TimesFM3Adapter` (aprox. línea 1072 en d9b8f4f) y en `adapter-parameters.md` línea 78. Lugares exactos en la sección 5.
- **Por qué**: TFM3-F7, TFM3-F12.
- **Riesgo**: nulo.
- **Rompe compatibilidad**: no.
- **Decisión**: ninguna.

## 4. Tests

### Tests unitarios

`skforecast/foundation/tests/tests_foundation_models/test_TimesFM25Adapter.py`:

- Adaptar `test_TimesFM25Adapter_load_model_no_LicenseWarning_and_noop_when_model_set`: el fake `_FakeV25Base` ya no se envuelve en `_TimesFMCompat`; simular la versión con `monkeypatch.setattr(importlib.metadata, "version", lambda name: "3.0.2")` (o el helper de 02), comprobar que `from_pretrained` recibe `torch_compile=False` y que no hay `LicenseWarning`. Simular la versión hace el test independiente de la timesfm que tenga instalada quien lo ejecute [inferido].
- Mantener `test_TimesFM25Adapter_load_model_ImportError_when_timesfm_not_installed`, `test_TimesFM25Adapter_ensure_compiled_calls_compile_with_actual_steps` y `test_TimesFM25Adapter_ensure_compiled_noop_when_already_compiled` (este último con el `max_context` del fake).
- Nuevos:
  - `test_TimesFM25Adapter_load_model_ImportError_when_timesfm_predates_3` (versión `"2.0.2"`: ImportError que nombra `timesfm>=3.0`, sin `LicenseWarning`), según D8.
  - `test_TimesFM25Adapter_load_model_ImportError_when_torch_missing` (módulo `timesfm` falso sin `TimesFM_2p5_200M_torch`, versión `"3.0.2"`, sonda bloqueada con `sys.modules["timesfm.timesfm_2p5.timesfm_2p5_torch"] = None`: ImportError con el comando de instalación, no `AttributeError`).
  - `test_TimesFM25Adapter_load_model_ImportError_when_huggingface_hub_too_old` (simular que `import timesfm` lanza `TypeError` con `paper_url`, por ejemplo con `monkeypatch` de `builtins.__import__`: ImportError que pide `huggingface_hub>=0.30`; otro `TypeError` se propaga sin cambios).
  - `test_TimesFM25Adapter_init_ValueError_when_forecast_config_kwargs_has_reserved_key` y `test_TimesFM25Adapter_set_params_ValueError_when_forecast_config_kwargs_has_reserved_key`, parametrizados con `max_context`, `max_horizon` y `return_backcast`.
  - `test_TimesFM25Adapter_ensure_compiled_warns_when_model_max_context_below_context_length` (TV-3: `IgnoredArgumentWarning` una sola vez en dos `predict`; sin aviso cuando `max_context >= context_length`).
  - Un test que fije que la puntual es el cuantil 0.5 con el fake actualizado (por ejemplo, `predict(quantiles=None)` igual a la columna de `quantiles=[0.5]`).

`skforecast/foundation/tests/tests_foundation_models/test_TimesFM3Adapter.py`:

- Reescribir `test_TimesFM3Adapter_load_model_ImportError_when_torch_missing` parametrizado en dos casos: el de 3.0.0 actual (`sys.modules["timesfm3"] = None`) y el de 3.0.2 (módulo `timesfm3` cuyo `__getattr__("TimesFM3Forecaster")` lanza ImportError). Esperar en ambos un ImportError que cite la causa (torch) y el comando de instalación, sin `LicenseWarning`.
- Nuevo caso de PR #552: módulo `timesfm` cuyo `__getattr__("TimesFM3Forecaster")` lanza ImportError.
- Mantener `test_TimesFM3Adapter_load_model_ImportError_when_timesfm_predates_3` (con D-VER (b), actualizar su `match` al mensaje del helper común, "TimesFM3Adapter requires timesfm>=3.0, but ... is installed", en lugar de "TimesFM 3.0 requires `timesfm>=3.0`") y `test_TimesFM3Adapter_load_model_ImportError_when_timesfm_not_installed_no_LicenseWarning`.
- En `test_TimesFM3Adapter_load_model_LicenseWarning_and_forwards_device`, simular la versión `"3.0.2"` con `monkeypatch` para que el test no dependa de la timesfm instalada (la comprobación de versión pasa a ejecutarse siempre) [inferido].
- Nuevo test del `TypeError` de `paper_url` (mismo patrón que en TimesFM 2.5).
- Si D6 (c): quitar `"padding_mode"` de la parametrización de `test_TimesFM3Adapter_init_ValueError_when_predict_kwargs_has_reserved_key`, cambiar `test_TimesFM3Adapter_set_params_ValueError_when_predict_kwargs_has_reserved_key` (hoy usa `{"padding_mode": "edge"}`) para que use otra clave reservada, y añadir un test de que el `padding_mode` del usuario llega a `predict_batch` (`FakeTimesFM3Forecaster.last_padding_mode`) sin duplicarse en `last_kwargs` y de que el valor por defecto sigue siendo `"edge"` con covariables y `"none"` sin ellas.

Otros:

- `skforecast/foundation/tests/tests_foundation_models/test_get_model_info.py`: actualizar `backend_package` de `TimesFM25Adapter` (aprox. línea 163 en d9b8f4f) a `"timesfm[torch]>=3.0"`; la línea 95 (TimesFM 3) con D-VER (b) (cota `>=3.0` en `TimesFM3Adapter`); opcional con (a)/(c).
- `skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py`: sin cambios; debe seguir pasando con `adapters.toml` actualizado.
- `fixtures_adapters.py`: `FakeTimesFM25Model` según 3.6 (puntual en el canal 5, `from_pretrained(cls, model_id, **kwargs)`, `_FakeForecastConfig.max_context`).

### Compatibilidad (`tools/foundation_compat/`)

- `adapters.toml`: `[timesfm25] backend_package = "timesfm[torch]>=3.0"` (D8); `[timesfm3] backend_package = "timesfm[torch]>=3.0"` con D-VER (b) (cota `>=3.0` en `TimesFM3Adapter`); opcional con (a)/(c).
- `tests/contracts.py`: quitar `TimesFM_2p5_200M_torch._from_pretrained`; añadir el constructor con `kwargs: ["torch_compile"]`. `timesfm3` sin cambios (`padding_mode` ya está en los kwargs de `predict_batch`).
- `tests/test_smoke.py`: el test de horizonte >128 de 3.6. Comprobar su poder de detección antes de añadir el mínimo de 3.3 (con el mínimo, timesfm 2.0.2 ya no carga): en el commit del test, ejecutarlo en `~/.venvs/fm-review-timesfm25-oldhf` (timesfm 2.0.2) y debe fallar; en `~/.venvs/fm-review-timesfm25` (3.0.2) debe pasar. Si con `CONTEXT_LENGTH = 64` no falla con 2.0.2, construir en ese test un `FoundationModel` propio con más contexto (por ejemplo 512, como en `evidence/timesfm25/check_flip_ar_quantiles.py`).

### Comprobaciones antes y después con la evidencia

Desde la raíz del repositorio, en los entornos de la revisión:

```bash
# TimesFM 2.5: adaptador frente a llamada directa (antes y después: diferencia 0.0)
~/.venvs/fm-review-timesfm25/bin/python dev/foundation_adapters_review/evidence/timesfm25/compare_direct.py
~/.venvs/fm-review-timesfm25/bin/python dev/foundation_adapters_review/evidence/timesfm25/verification/verify_direct.py

# Orden de cuantiles en pasos >128 (3.0.2: 1.00; con 2.0.2, tras D8 A, ImportError de versión)
~/.venvs/fm-review-timesfm25/bin/python dev/foundation_adapters_review/evidence/timesfm25/check_flip_ar_quantiles.py
~/.venvs/fm-review-timesfm25-oldhf/bin/python dev/foundation_adapters_review/evidence/timesfm25/check_flip_ar_quantiles.py

# Carga sin workaround: instalar cada combinación en el entorno auxiliar y ejecutar
# (2.0.0 + 0.36.0, 2.0.1 + 0.36.0, 3.0.2 + 0.36.0, 3.0.2 + 1.0.0), por ejemplo:
uv pip install --python ~/.venvs/fm-review-timesfm25-oldhf/bin/python --torch-backend cpu \
    "timesfm[torch]==2.0.0" "huggingface_hub==0.36.0"
~/.venvs/fm-review-timesfm25-oldhf/bin/python dev/foundation_adapters_review/evidence/timesfm25/check_workaround.py

# Mensajes de import sin torch
~/.venvs/fm-review-timesfm25/bin/python dev/foundation_adapters_review/evidence/timesfm25/check_import_errors.py
~/.venvs/fm-review-timesfm3/bin/python dev/foundation_adapters_review/evidence/timesfm3/check_import_errors.py
~/.venvs/fm-review-timesfm3-v300/bin/python dev/foundation_adapters_review/evidence/timesfm3/check_import_errors.py

# TimesFM 3: adaptador frente a predict_batch e invariancia del lote (no deben cambiar)
~/.venvs/fm-review-timesfm3/bin/python dev/foundation_adapters_review/evidence/timesfm3/compare_direct.py
~/.venvs/fm-review-timesfm3/bin/python dev/foundation_adapters_review/evidence/timesfm3/batch_invariance.py
~/.venvs/fm-review-timesfm3/bin/python dev/foundation_adapters_review/evidence/timesfm3/verification/check_guide_fold0.py
```

Al terminar, dejar `~/.venvs/fm-review-timesfm25-oldhf` con timesfm 2.0.2 y `huggingface_hub==0.36.0`, como estaba.

TV-1 con `huggingface_hub` 0.29.3 (entorno auxiliar de la verificación, que ya queda con esa combinación; recrearlo solo si no existe):

```bash
uv venv ~/.venvs/fm-review-timesfm25-verify --python 3.12
uv pip install --python ~/.venvs/fm-review-timesfm25-verify/bin/python --torch-backend cpu \
    -e . pytest "timesfm[torch]==3.0.2" "huggingface_hub==0.29.3"
~/.venvs/fm-review-timesfm25-verify/bin/python -c 'import numpy as np, pandas as pd
from skforecast.foundation import FoundationModel
y = pd.Series(np.arange(100.0), index=pd.date_range("2024-01-01", periods=100, freq="h"))
m = FoundationModel(model_id="google/timesfm-2.5-200m-pytorch")
m.fit(series=y)
print(m.predict(steps=3))'
```

Backtest de la guía (`evidence/common/backtest_guide.py`): no hace falta en este PR con las recomendaciones (a)+(c) de D6 y A de D8, porque ningún cambio altera predicciones (quitar el workaround y `torch_compile=False` dan 0.0 frente a la llamada directa) [probado]. Solo si se eligiera D6 (b), ejecutarlo antes y después como comprobación de corrección, no como benchmark:

```bash
~/.venvs/fm-review-timesfm3/bin/python dev/foundation_adapters_review/evidence/common/backtest_guide.py \
    --model-id google/timesfm-3.0-pytorch --exog --out dev/foundation_adapters_review/evidence/timesfm3/backtest_after.csv
```

Valor de referencia antes del cambio: MAE 162.4352 (`evidence/timesfm3/backtest_guide_exog.log`).

## 5. Documentación y contexto de IA

### Docstrings (`skforecast/foundation/_adapters.py`)

- `_import_timesfm`: nuevos parámetros (`attr`, `probe` y, solo con D-VER (a)/(c), `min_version`) y errores (3.1).
- `TimesFM25Adapter`: parámetro `model` (contexto efectivo limitado por el `max_context` precompilado, TV-3); parámetro `forecast_config_kwargs` en la clase y en `__init__` (claves reservadas, `per_core_batch_size`, `window_size` sin efecto upstream); Notes (mediana, T25-P2; CUDA o CPU sin MPS y float32; límite 16384 de contexto más horizonte; XReg no usado, T25-P7); Notes de `_load_model` (sin workaround, mínimo `timesfm>=3.0`, `torch_compile=False`); Notes de `_ensure_compiled` (aviso de TV-3); comentario de canales en `predict`.
- `TimesFM3Adapter`: `context_length` ("15,360; longer contexts are truncated by the backend"); `predict_kwargs` en la clase y en `__init__` (solo si D6 (c)); Notes de la clase (matiz de relleno por `per_core_batch_size`, TFM3-P3); Notes de `predict` (`padding_mode`, TFM3-P2); Notes de `_validate_predict_kwargs` (solo si D6 (c)); Notes de `_load_model`.
- `skforecast/foundation/_foundation_model.py`: lista de kwargs de TimesFM 2.5 (aprox. líneas 113-115 en d9b8f4f), opcional, si se quiere citar `per_core_batch_size` dentro de `forecast_config_kwargs`; nota de dispositivo (aprox. líneas 234-235: "No device parameter: TimesFM 2.5, which relies on its backend's own default device selection") con "CUDA if available, otherwise CPU; no MPS".

### Guía de usuario (`docs/user_guides/foundation-forecasting-models.ipynb`)

Solo texto de TimesFM; no re-ejecutar el notebook (lo hace `10_docs_ci.md`).

- Celda 11 en la revisión (markdown con la tabla HTML "Supported Foundation Models"; columnas Model, Provider, Available model IDs, Covariate support (exog), Default context length, Max context length, Max horizon, Point forecast, Cross learning, Backend, Install command, Links):
  - fila "TimesFM 2.5": "Point forecast" de "Mean (dedicated output)" a "Median (0.5 quantile)" (mismo texto que la fila de TimesFM 3.0); "Max context length" de "16384" a "16384 (context + horizon)" o similar (T25-F11); comando de instalación de `pip install timesfm[torch]` a `pip install "timesfm[torch]>=3.0"` en el `<code>`, en `data-copy` y en `aria-label` (D8). Dentro de atributos HTML delimitados por comillas dobles, escribir las comillas como `&quot;` (o usar comillas simples en el comando) y `>` como `&gt;` en el `<code>` [inferido].
  - fila "TimesFM 3.0": "Max context length" de "~15,360" a "15,360" (TFM3-P5); comando de instalación igual al de TimesFM 2.5 solo si se declara la cota también en `TimesFM3Adapter`.
- Celda 53 en la revisión (markdown "### TimesFM 2.5"): opcional, una frase de que el backend usa CUDA o CPU (sin MPS) y de que el modelo no usa covariables (XReg no soportado), T25-P7 y T25-P8.
- Celda 59 en la revisión (markdown "### TimesFM 3.0", "Unlike TimesFM 2.5, this version natively supports exogenous variables"): una frase sobre la licencia no comercial y no de producción (TFM3-P5).
- Celda 17 en la revisión (markdown "Exogenous Variables (`exog`)", con "TimesFM 2.5 and Moirai do not accept exogenous variables") y celda 109 en la revisión (nota al pie de la comparación, "TimesFM-2.5 (200m) and Moirai-2.0-R (small) do not allow the inclusion of exogenous features"): sin cambios salvo que D10 elija B.

### Skill y referencias

- `skills/foundation-forecasting/references/adapter-parameters.md`:
  - Sección TimesFM25Adapter: línea 53 ("Point forecast: mean (in practice equal to quantile `0.5`)" pasa a "median (quantile `0.5`)"); línea 61 (`forecast_config_kwargs`: claves reservadas y `per_core_batch_size`); línea 63 (añadir CUDA o CPU sin MPS, límite de 16384 de contexto más horizonte, requisito `timesfm>=3.0`, XReg no usado).
  - Sección TimesFM3Adapter: línea 78 ("roughly 15,360"); línea 80 (`predict_kwargs`, solo si D6 (c)); línea 82 (quitar la frase de invariancia y añadir el matiz de `per_core_batch_size`); línea 84 ("non-commercial and non-production", con la nota de BigQuery ML).
- `skills/foundation-forecasting/SKILL.md`: línea 49 (tabla de errores frecuentes) y línea 196 (párrafo de licencias): TimesFM 3.0 "non-commercial and non-production"; línea 57 (instalación): `pip install "timesfm[torch]>=3.0"   # For TimesFM 2.5 and 3.0`; línea 185 (`info.backend_package` de TimesFM 3) con D-VER (b) (cota `>=3.0` en `TimesFM3Adapter`); opcional con (a)/(c). La línea 166 (prefijo) es de `02_common.md`.
- `tools/ai/llms-base.txt`: línea 529 (licencias): matiz "non-commercial and non-production" de TimesFM 3.0; línea 507 (`info.backend_package` de TimesFM 3) con D-VER (b) (cota `>=3.0` en `TimesFM3Adapter`); opcional con (a)/(c). La línea 458 solo nombra el paquete `timesfm`, sin comando de instalación: no requiere cambio. La línea 489 (prefijo) es de `02_common.md`. La línea 531 (independencia respecto al exog) es correcta y no se toca aquí.
- Opcional: mencionar el `TypeError` de `paper_url` (TV-1) en la skill `troubleshooting-common-errors` si se quiere documentar además del mensaje.
- Regenerar con el skill `ai-context-sync`: `python tools/ai/generate_ai_context_files.py` y `python tools/ai/generate_ai_context_files.py --check`. No editar a mano los ficheros generados.

### Nota de versión (`docs/releases/releases.md`, sección 0.27.0)

Texto propuesto (ajustar a la decisión tomada y al número de PR).

En la advertencia "Before upgrading", dentro de la lista "And some code needs to be updated:" (indentada con 4 espacios, como el resto de esa lista):

```markdown
    + <code>[TimesFM25Adapter]</code> requires `timesfm>=3.0` (`pip install -U "timesfm[torch]>=3.0"`). With timesfm 2.0.x the quantiles of the steps beyond 128 were collapsed and unordered.
```

En "Changed":

```markdown
+ <code>[TimesFM25Adapter]</code> requires `timesfm[torch]>=3.0` and raises an `ImportError` with older versions, whose quantiles were wrong for horizons longer than 128 steps. The workaround for `timesfm` 2.0.0 with `huggingface_hub<1.0` is removed. ([#XXXX](https://github.com/skforecast/skforecast/pull/XXXX))
```

En "Fixed":

```markdown
+ <code>[TimesFM25Adapter]</code> and <code>[TimesFM3Adapter]</code> raised an `AttributeError`, a generic message or a `TypeError` about `paper_url` instead of an `ImportError` that explains the cause when torch was missing or `huggingface_hub` was older than 0.30. ([#XXXX](https://github.com/skforecast/skforecast/pull/XXXX))
+ <code>[TimesFM25Adapter]</code> raises a `ValueError` when `forecast_config_kwargs` includes `max_context`, `max_horizon` or `return_backcast`, which failed later with a cryptic error, and warns when a pre-compiled `model` has a `max_context` shorter than `context_length`. ([#XXXX](https://github.com/skforecast/skforecast/pull/XXXX))
+ The documentation of <code>[TimesFM25Adapter]</code> described the point forecast as the mean; it is the median (quantile 0.5). The docstring of <code>[TimesFM3Adapter]</code> said that `padding_mode="edge"` is the default of timesfm; the default of timesfm is `"none"`. TimesFM 3.0 weights are documented as non-commercial and non-production. ([#XXXX](https://github.com/skforecast/skforecast/pull/XXXX))
```

Solo si D6 (c), en "Added":

```markdown
+ <code>[TimesFM3Adapter]</code> accepts `padding_mode` in `predict_kwargs`; the default is unchanged (`"edge"` with covariates, `"none"` without them). ([#XXXX](https://github.com/skforecast/skforecast/pull/XXXX))
```

## 6. Decisiones pendientes

| Id | Decisión | Opciones | Recomendación | Impacto en el plan | Estado |
|---|---|---|---|---|---|
| D6 | `padding_mode` de TimesFM 3 (TFM3-P2) | (a) mantener `"edge"` y corregir el docstring; (b) `"none"` como upstream (cambia predicciones; contextos >15360 exigen >=3.0.2); (c) configurable por `predict_kwargs` | (a) + (c) | (a) siempre entra; (c) añade el cambio de `_RESERVED_PREDICT_KWARGS`, el `pop`, docstrings y tests; (b) obliga a ejecutar el backtest de la guía antes y después, a una entrada en "Before upgrading" y a re-ejecutar las celdas de TimesFM 3 en 10 | pendiente |
| D7 | Opciones de carga de TimesFM 3 (TFM3-P4) | (a) nada, `model=` precargado ya lo permite; (b) `model_kwargs` reenviado; (c) solo `revision` | (a) ahora y decidir (b) de forma común para todos los adaptadores | Con (a), sin código en este PR; (b) o (c) serían un PR posterior y común | pendiente |
| D8 | Mínimo de TimesFM 2.5 (T25-P1) | A `>=3.0`, ImportError con <3 y quitar el workaround; B `>=2.0.1` y quitar el workaround (no protege del bug de cuantiles); C solo docs. Sub-decisión: ImportError o UserWarning con 2.0.x | A con ImportError | Fija `backend_package`, `adapters.toml`, `min_version` del helper de 3.1 (solo con D-VER (a)/(c)), el mensaje, los tests de versión, la nota "Before upgrading" y el comando de instalación en guía y skill; con B hay que comparar versiones completas; con C no se quita `_TimesFMCompat` | pendiente |
| D9 | Prefijo de TimesFM 2.5 (T25-P3) | A prefijo exacto `-200m-pytorch`; B rechazar `-flax` y `-transformers` con mensaje; C nada | B, como entrada de la tabla central de D-REG (a); A solo si D-REG se resuelve con (c) | Con B se implementa en `02_common.md` y aquí solo condiciona qué textos de prefijo no tocar; con A el cambio de prefijo (`_ADAPTER_REGISTRY`, `_MODEL_ID_PREFIX`, `test_ADAPTER_REGISTRY_contains_all_expected_entries`) se hace en este PR, como indica 02 (3.2) | pendiente |
| D10 | XReg de TimesFM 2.5 (T25-P7) | A no implementar y matizar docs; B implementar (beta común al lote, jax, `timesfm[xreg]`) | A | Con A, solo textos en este PR; B sería un PR propio con `allow_exog=True` y nuevas dependencias | pendiente |

## 7. Criterios de aceptación

- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_TimesFM25Adapter.py -q` pasa, con los tests nuevos de la sección 4.
- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_TimesFM3Adapter.py -q` pasa (57 o más).
- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_get_model_info.py skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py -q` pasa.
- [ ] Una vez al terminar, `pytest skforecast/foundation/tests/tests_foundation_models/ -q` pasa (carpeta del módulo tocado; sin `-n`).
- [ ] `~/.venvs/fm-review-timesfm25/bin/python -m pytest tools/foundation_compat/tests/test_contract.py --backend timesfm25 -q -p no:cacheprovider`: 6 passed (sale la entrada `_from_pretrained` y entra el constructor con `torch_compile`).
- [ ] `~/.venvs/fm-review-timesfm25/bin/python -m pytest tools/foundation_compat/tests/test_smoke.py --backend timesfm25 -q -p no:cacheprovider`: los 5 actuales y el nuevo de horizonte >128 pasan, 1 skipped (exog). El test nuevo falló en `~/.venvs/fm-review-timesfm25-oldhf` (timesfm 2.0.2) antes de añadir el mínimo.
- [ ] Contrato y smoke con `--backend timesfm3` en `~/.venvs/fm-review-timesfm3`: 3 passed y 6 passed (más el nuevo si se aplica a timesfm3).
- [ ] `python tools/foundation_compat/run.py --adapter timesfm25` y `python tools/foundation_compat/run.py --adapter timesfm3`: `"status": "passed"` en el JSON de `results/` (ignorado por git).
- [ ] `evidence/timesfm25/compare_direct.py` y `evidence/timesfm25/verification/verify_direct.py`: diferencia 0.0 frente a la llamada directa en puntual y cuantiles; "point == ch5: True"; "loaded torch_compile: False"; en la sección 4, `max_context` y `return_backcast` fallan ahora con `ValueError` al construir; `context_length=16384` sigue fallando con el mensaje del backend y 16000 da 30 filas.
- [ ] `evidence/timesfm25/check_flip_ar_quantiles.py`: con timesfm 3.0.2, "steps 129-384: 1.00" en las dos series; con timesfm 2.0.2 (entorno `-oldhf`), ImportError que pide `timesfm>=3.0` (D8 A).
- [ ] `evidence/timesfm25/check_workaround.py`: con timesfm 3.0.2 y `huggingface_hub` 0.36.0 o 1.0.0, el adaptador carga y `torch_compile = False`; con timesfm 2.0.0 y 2.0.1, ImportError de versión en el adaptador (D8 A; la carga directa (a) de 2.0.0 sigue dando el `TypeError` de `proxies`, y la de 2.0.1 sigue cargando).
- [ ] `evidence/timesfm25/check_import_errors.py`: ImportError con la causa y `pip install "timesfm[torch]>=3.0"`, no `AttributeError`.
- [ ] `evidence/timesfm3/check_import_errors.py` con timesfm 3.0.2 y 3.0.0: ImportError que cita torch ("import of torch halted") en las dos versiones, sin `LicenseWarning`.
- [ ] Prueba de TV-1 con `huggingface_hub==0.29.3`: ImportError que pide `huggingface_hub>=0.30`, no `TypeError`.
- [ ] `evidence/timesfm3/compare_direct.py`: maxdiff 0.0 en A, C y D (valores por defecto sin cambios); `evidence/timesfm3/batch_invariance.py`: mismas cifras (0.31692 y 0.20221; solo cambia el texto); `evidence/timesfm3/verification/check_guide_fold0.py`: "MAE from csv 162.43520906024307" y "fold 0 FoundationModel vs csv maxdiff 0.0".
- [ ] `python tools/ai/generate_ai_context_files.py --check` sin errores tras regenerar.
- [ ] Entrada en `docs/releases/releases.md` (sección 0.27.0).
- [ ] Skill `verify` en verde antes del commit.

## 8. Fuera de alcance

- Medir velocidad o precisión entre versiones de timesfm, entre `"edge"` y `"none"`, entre valores de `per_core_batch_size` o con `torch.compile`: no es tarea de skforecast y la revisión no lo usa para decidir.
- Recomendar la `ForecastConfig` del ejemplo de la ficha de TimesFM 2.5 (`normalize_inputs`, `use_continuous_quantile_head`, `fix_quantile_crossing`) o los valores de `TimesFM3Evaluator` (`use_symmetric_averaging=True`, `make_positive=True`): son cuestiones de precisión.
- T25-P3 y la tabla central de ids no soportados (`google/timesfm-2.5-200m-flax`, `-transformers`): `02_common.md` (D9, D-REG).
- TV-2 (contexto todo NaN), la conversión de niveles de cuantil a float (D-Q) y el helper común de versión (D-VER): `02_common.md`.
- Re-ejecutar la guía de usuario y actualizar el workflow del CI de compatibilidad, incluida la frase de `tools/foundation_compat/README.md` sobre el runtime MLX de timesfm "with other signatures" (TFM3-F9): `10_docs_ci.md`.
- La corrección general de la garantía de independencia entre series en `FoundationModel.predict`, `llms-base.txt` y la guía de series heterogéneas (X-8): transversal, no la cubre este plan; aquí solo se corrige la frase de TimesFM 3 en `adapter-parameters.md`.
- XReg para TimesFM 2.5 (D10 B), `padding_mode="none"` por defecto (D6 b) y las opciones de carga de TimesFM 3 (D7 b/c): pospuestos hasta su decisión.
- Backend MLX (`timesfm3.mlx`), soporte MPS o un parámetro de dtype para TimesFM 3: el backend no los ofrece en torch o no se pudieron comprobar (sin GPU ni Apple Silicon).
- Agrupar en `FoundationModel` las series por longitud para TimesFM 3 (alternativa a TFM3-P3): cambiaría predicciones y multiplicaría llamadas; requeriría una decisión aparte.
