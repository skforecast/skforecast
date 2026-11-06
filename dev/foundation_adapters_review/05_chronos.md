# 05. Chronos-2 (chronos-forecasting)

- **Rama propuesta**: `fix/chronos-alignment`, creada desde `origin/0.27.x`.
- **Prioridad**: media. El adaptador funciona con la última versión del backend; los cambios declaran un mínimo de versión que hoy falta, corrigen una covariable mal codificada y precisan textos.
- **Adaptador**: `ChronosAdapter`.
- **Ficheros afectados**:
  - `skforecast/foundation/_adapters.py` (`ChronosAdapter`: atributo `backend_package`, docstrings de clase y de `__init__`, `_load_pipeline`, `_to_covariate_array`, `_build_chronos_input`, validación de `predict_kwargs` en `__init__` y `set_params`).
  - `skforecast/foundation/_foundation_model.py` (solo el docstring de la clase, atributo `adapter`).
  - `skforecast/foundation/tests/tests_foundation_models/test_ChronosAdapter.py`, `test_get_model_info.py` y `test_device_handling.py` (este último solo cambia con D14 (b), salvo un retoque opcional de docstring).
  - `tools/foundation_compat/adapters.toml`, `tools/foundation_compat/tests/contracts.py`.
  - `docs/user_guides/foundation-forecasting-models.ipynb` (solo el texto de Chronos, sin reejecutar).
  - `skills/foundation-forecasting/SKILL.md`, `skills/foundation-forecasting/references/adapter-parameters.md`, `skills/prediction-intervals/SKILL.md`, `tools/ai/llms-base.txt` y los ficheros generados.
  - `docs/releases/releases.md`.
- **Depende de**: `02_common.md` para CHR-P1 (comprobación central de ventana de contexto sin valores observados, decisión D11) y, solo si se adopta D-VER (b), para el helper común de comprobación de versión que usaría CHR-P2. El resto del plan no espera a 02.
- **Bloquea a**: `10_docs_ci.md` (reejecución única de la guía, que refresca la salida de `list_adapters` con el nuevo `backend_package`, y cambios del CI de compatibilidad).
- **Cierra issue**: ninguna de skforecast. Relacionada upstream: amazon-science/chronos-forecasting#544 (abierta), que este PR no cierra.

## 1. Estado en la revisión (2026-11-06)

Base de la revisión: commit `d9b8f4f` de la rama de revisión creada desde `origin/0.27.x` (skforecast 0.27.0 en desarrollo) [verificado en código].

| Elemento | Antes (probado en la revisión) | Última disponible | Fuente |
|---|---|---|---|
| chronos-forecasting | 2.3.2 | 2.3.2, publicada en PyPI el 2026-09-08 (2.3.0 el 2026-06-18, 2.2.0 el 2025-12-08, 2.1.0 el 2025-11-21) [documentado] | `evidence/chronos/pypi_info.log` |
| main upstream | | 2 commits sin publicar: `10afa9e` (memoria en `Chronos2Dataset`, solo `dataset.py`) y `1224faf` (`predict_df`, `df_utils.py`); ninguno toca la API que usa el adaptador [verificado en código] | `evidence/chronos/verification/verification.md` |
| Python | 3.12.3 en los entornos de la revisión; 3.12.16 en el CI | `requires_python >=3.10` en 2.0.0 a 2.3.x [documentado] | `evidence/chronos/pypi_info.log`, `evidence/chronos/requirements-freeze.txt` |
| torch | 2.15.0+cpu | `torch>=2.2,<3` desde 2.1.0 (2.0.0: `>=2.0`) [documentado] | `evidence/chronos/pypi_info.log` |
| transformers | 5.22.1 (entorno principal), 4.57.6 (entorno con 2.1.0 y 2.2.0) | 2.3.x: `transformers>=4.41,<6`; 2.2.x: `<5`; 2.0.0: `>=4.49,<5` [documentado] | `evidence/chronos/pypi_info.log` |
| Otras dependencias de 2.3.x | accelerate 1.15.0, numpy 2.5.4, pandas 2.3.3, huggingface-hub 2.4.0 | `accelerate>=1.1.0,<2`, `numpy<3`, `pandas>=2.0,<4`; scikit-learn desaparece en 2.3.0 [documentado] | `evidence/chronos/pypi_info.log` |

Checkpoints en el Hub (todos apache-2.0, no gated, `Chronos2Pipeline`, contexto 8192, `output_patch_size` 16, `max_output_patches` 64, cuantiles de entrenamiento en [0.01, 0.99]) [probado]:

| Checkpoint | sha | Último cambio | Cuantiles de entrenamiento | Nota |
|---|---|---|---|---|
| `amazon/chronos-2` | `29ec3766d3` | 2026-06-05 (solo README) | 21 | config `f32073949651` y pesos `ddcda3c7508b` idénticos a `autogluon/chronos-2` |
| `autogluon/chronos-2` | `60088152a3` | 2026-06-05 (solo README) | 21 | mismo blob de pesos |
| `autogluon/chronos-2-small` | `ddec01313e` | 2025-12-03 | 13 | checkpoint por defecto del adaptador (`default_model_id`) |
| `autogluon/chronos-2-synth` | `3607918a9f` | 2025-12-03 | 13 | entrenado solo con datos sintéticos |

No hay checkpoints Chronos nuevos de amazon ni de autogluon desde 2025-12-03 [probado]. Evidencia: `evidence/chronos/hub_info.log`, `evidence/chronos/hub_commits.log`, `evidence/chronos/hub_revisions.log`, `evidence/chronos/verification/hub_recheck.log`, `evidence/chronos/verification/configs_recheck.log`.

Línea base del CI [probado]: workflow Foundation Backends Compatibility, run 42212742430, job 127875169074 (rama 0.27.x, commit `09433c35`, 2026-11-06, Python 3.12.16): chronos-forecasting 2.3.2, torch 2.15.0+cpu, transformers 5.22.1, accelerate 1.15.0; contrato 3 passed, smoke 6 passed (2 `InputTypeWarning`), `held_back` false, estado passed. En la revisión, `python tools/foundation_compat/run.py --adapter chronos` también dio passed, contrato y smoke dieron 9 passed y los 55 tests de `test_ChronosAdapter.py` pasaron con el backend real instalado. Evidencia: `evidence/chronos/ci_baseline.log`, `evidence/chronos/compat_run.log`, `evidence/chronos/compat_results/chronos.json`, `evidence/chronos/contract.log`, `evidence/chronos/smoke.log`, `evidence/chronos/unit_tests.log`, `evidence/chronos/verification/contract_smoke_rerun.log`, `evidence/chronos/verification/unit_tests_rerun.log`.

### Antes de empezar

Este plan puede quedar desfasado. Antes de tocar código, desde la raíz del repositorio:

```bash
# Entornos de la revisión (ver evidence/README.md para recrearlos)
bash dev/foundation_adapters_review/evidence/common/make_env.sh chronos "chronos-forecasting" 3.12
uv venv ~/.venvs/fm-review-chronos-old --python 3.12
uv pip install --python ~/.venvs/fm-review-chronos-old/bin/python --torch-backend cpu \
    -e . pytest "chronos-forecasting==2.1.0"

# Última versión publicada y requisitos
python dev/foundation_adapters_review/evidence/common/pypi_info.py chronos-forecasting
# Checkpoints (sha, fecha, licencia, gated); el prefijo se compara con el nombre sin autor
python dev/foundation_adapters_review/evidence/common/hub_info.py amazon --prefix chronos-2
python dev/foundation_adapters_review/evidence/common/hub_info.py autogluon --prefix chronos-2
# Estado de la issue upstream del contexto todo NaN (#544)
gh search issues --repo amazon-science/chronos-forecasting "extreme quantiles" --state open
```

Qué hacer según el resultado:

- **Sigue siendo 2.3.2 y los mismos sha**: seguir el plan tal cual.
- **Hay una versión nueva (2.3.x o 2.4)**: recrear el entorno con `make_env.sh` y volver a ejecutar contrato, smoke, `evidence/chronos/signature_check.py` y `evidence/chronos/compare_direct.py` (comandos en la sección 4). Si las diferencias siguen en 0 y `Chronos2Pipeline.predict` sigue declarando `cross_learning`, el plan vale; anotar la versión nueva en la descripción del PR. Si cambia la firma de `predict` o `predict_quantiles`, revisar CHR-P2, CHR-P3 y CHR-P8 antes de implementarlos. Si la nueva versión arregla #544 o cambia el tratamiento de categóricas (`np.issubdtype(dtype, np.number)`), revisar CHR-F4 y CHR-F5. Si cambian los requisitos de transformers, revisar D12.
- **Hay un checkpoint nuevo** (por ejemplo `autogluon/chronos-2.5`): el prefijo `autogluon/chronos-2` ya lo resolvería a `ChronosAdapter` [inferido a partir de `evidence/chronos/resolve_adapter.log`, que prueba la resolución por prefijo]; ejecutar `compare_direct.py <model_id>` con él y comprobar licencia y `gated` con `hub_info.py` antes de documentarlo. Añadirlo a la tabla de la guía es trabajo aparte si no cabe en este PR.

## 2. Hallazgos

- **CHR-F1** [probado] El adaptador funciona con chronos-forecasting 2.3.2. Su salida es idéntica (max_abs_diff 0) a `Chronos2Pipeline.predict_quantiles` con la misma entrada en `autogluon/chronos-2-small`, `autogluon/chronos-2-synth` y `amazon/chronos-2` (este último probado en la primera pasada; la verificación no lo reejecutó, pero comparte config y blob de pesos con `autogluon/chronos-2`): serie única (caso A); tres series de longitudes 400, 300 y 180 con NaN y covariables numérica, string y bool, con `cross_learning` False y True (caso B); bfloat16 (caso H); covariable solo de pasado. Salida de forma `(steps, n_quantiles)` y dtype float32. Evidencia: `evidence/chronos/compare_direct.log` (casos A, B, H), `evidence/chronos/verification/compare_direct_rerun.log`, `evidence/chronos/capabilities_check.log`, `evidence/chronos/contract.log`, `evidence/chronos/smoke.log`, `evidence/chronos/compat_run.log`, `evidence/chronos/unit_tests.log`.
- **CHR-F2** [verificado en código] Entre v2.2.2 y v2.3.2 no cambian las firmas de `Chronos2Pipeline.predict` ni `predict_quantiles` (solo el tipo de `inputs`, que añade `Sequence[PreparedInput]`). `from_pretrained` acepta `torch_dtype` y `dtype` y los traduce según la versión de transformers (PR #505). 2.3.0 reescribe el preprocesado (`preprocess.py`, PR #493): valida un esquema homogéneo de covariables en cada llamada; elimina scikit-learn (PR #506) y admite transformers 5 (PR #416). El adaptador ya cumple el esquema homogéneo porque `supports_heterogeneous_covariates=False` y `FoundationModel` agrupa las series. Los commits `10afa9e` y `1224faf` de main no afectan. Evidencia: `evidence/chronos/signature_2.3.2.log`, `evidence/chronos/capabilities_check.log` (esquema heterogéneo: `ValueError` "Heterogeneous lists are not supported"), `evidence/chronos/pypi_info.log`.
- **CHR-F3** [probado] El mínimo real del backend es 2.2.0 y no está declarado. `ChronosAdapter.predict` pasa siempre `cross_learning`, introducido en 2.2.0 (PR #418); en 2.0.x y 2.1.x `predict` rechaza los kwargs desconocidos (v2.1.0 `pipeline.py`, aprox. línea 496) y con 2.1.0 cada predicción falla con `TypeError: Unexpected keyword arguments: ['cross_learning']`, también con `torch_dtype="bfloat16"`. Con 2.2.0 funciona (salida `(12, 2)`). `backend_package` y `adapters.toml` no fijan versión y el `ImportError` de `_load_pipeline` dice ">=2.0". El contrato no lo detecta: `cross_learning` figura como `weak_kwarg` de `predict_quantiles`, que lo recibe por `**predict_kwargs`. Además, 2.2.x fija transformers<5 [documentado]. Evidencia: `evidence/chronos/old_backend_2.1.0.log`, `evidence/chronos/old_backend_2.2.0.log`, `evidence/chronos/verification/old_backend_2.1.0_rerun.log`, `evidence/chronos/verification/old_backend_2.2.0_rerun.log`, `evidence/chronos/signature_2.1.0.log`, `evidence/chronos/signature_2.3.2.log`, `evidence/chronos/verification/contract_p3_check.log`.
- **CHR-F4** [probado] Si las últimas `context_length` observaciones de una serie son todas NaN (aunque haya datos antes), `FoundationModel.predict` y `fit` + `predict` pasan esa ventana a Chronos-2, que devuelve q0.1 entre -94680 y -57279 y q0.9 entre 58444 y 86309, sin ningún aviso (con una serie, con un dict de dos series y tras `fit`; serie `tail_nan`, 100 NaN finales con `context_length=64`). skforecast solo rechaza series enteramente NaN (`skforecast/utils/utils.py`, aprox. línea 4446 en d9b8f4f). `backtesting_foundation` sí salta esas series (`skforecast/model_selection/_validation.py`, aprox. líneas 2892-2895), y el comentario del mismo fichero (aprox. líneas 2874-2882) que dice que el camino del usuario las rechaza es falso [verificado en código]. Upstream: issue amazon-science/chronos-forecasting#544, abierta el 2026-11-05 y reproducida sobre 2.3.2 (InstanceNorm con loc=0 y scale=1 y sinh inverso) [documentado]. Es un caso de X-1, que se arregla en `02_common.md`. Evidencia: `evidence/chronos/all_nan_context.log`, `evidence/chronos/verification/all_nan_context_rerun.log`, `evidence/cross_cutting/probe_shared.log`, `evidence/chronos/upstream_issues.log`.
- **CHR-F5** (corregido por la verificación) [probado] Una covariable `pd.Categorical` con categorías numéricas llega al backend como entero (int32 en el caso C) y se trata como numérica: `_to_covariate_array` hace `to_numpy()` y el backend decide numérico o categórico con `np.issubdtype(dtype, np.number)` (v2.3.2 `preprocess.py`, `_stack_covariate`; v2.2.2 `dataset.py`, aprox. línea 133). La salida coincide con la llamada directa con la columna numérica (diferencia 0) y difiere de la llamada con la columna como categoría (diferencia 0.348). Contradice el docstring de `_to_covariate_array` y el atributo `supports_categorical_covariates=True` (el nombre correcto; no `supports_categorical_features`). No es una regresión: igual en 2.2.x. La ruta `predict_df` de upstream sí trata un `pd.Categorical` como categoría (`is_numeric_dtype` es False) [verificado en código]. Mezclar dtypes de la misma columna entre series ya lo rechaza `FoundationModel` con `TypeError` (caso D). Evidencia: `evidence/chronos/compare_direct.log` casos C y D, `evidence/chronos/verification/compare_direct_rerun.log`.
- **CHR-F6** (corregido por la verificación) [probado] `cross_learning` se aplica dentro de cada lote del backend (`batch_size`, 256 por defecto, contando filas de objetivo y covariables), no dentro de todo el grupo de covariables que forma `FoundationModel`. Un lote puede superar `batch_size`, porque se añaden tareas enteras mientras el acumulado sea menor [verificado en código, v2.3.2 `dataset.py` `_generate_sequential_batches`, aprox. líneas 300-311, y `pipeline.py` `predict`, que pone `group_ids` a cero por lote]. Con 4 series y `cross_learning=True`, `predict_kwargs={"batch_size": 2}` cambia la predicción de g0 en 0.540 frente a `batch_size=256`. El docstring upstream de `cross_learning` (v2.3.2 `pipeline.py`) ya dice "Results become dependent on batch size" y recomienda un `batch_size` de alrededor de 100 [verificado en código]. El docstring del adaptador ("applies within each of those groups") y el SKILL son imprecisos. Pregunta abierta: no se probó un grupo de más de 256 filas por la ruta de `FoundationModel` (se forzó `batch_size=2`). Evidencia: `evidence/chronos/compare_direct.log` caso G, `evidence/chronos/verification/compare_direct_rerun.log`.
- **CHR-F7** [verificado en código] `device_map="auto"` no pasa por `_resolve_torch_device` (definido en `_adapters.py`, aprox. línea 34): `ChronosAdapter._load_pipeline` lo reenvía a `BaseChronosPipeline.from_pretrained`, transformers lo delega en accelerate (`transformers/integrations/accelerate.py`, aprox. línea 115; `accelerate.infer_auto_device_map`), que reparte el modelo entre GPUs y descarga a CPU o disco si falta memoria. En un Mac con MPS, `get_max_memory` (accelerate 1.15.0, `utils/modeling.py`) añade "mps" y no añade "cpu", así que todo va a MPS o a disco (la verificación lo anotó como CHR-V4). No equivale a "CUDA > MPS > CPU", como dicen el docstring, `SKILL.md` (línea 82) y `adapter-parameters.md` (línea 42). `T0Adapter._load_model` sí usa `_resolve_torch_device` (aprox. línea 3804 en d9b8f4f). En CPU, el modelo queda en cpu con `hf_device_map=None` [probado, `evidence/chronos/verification/compare_direct_rerun.log`]. No hay GPU en la revisión. Evidencia: `evidence/chronos/compare_direct.log`, `evidence/chronos/verification/verification.md`.
- **CHR-F8** [probado] Los niveles fuera de [0.01, 0.99] se recortan al extremo (`q_0.001 == q_0.01` y `q_0.999 == q_0.99`, diferencia 0) y el backend solo emite un `logger.warning` (v2.3.2 `pipeline.py`, aprox. línea 806), sin `warnings.warn`. `autogluon/chronos-2-small` y `-synth` se entrenaron con 13 cuantiles y `chronos-2` con 21; los intermedios se interpolan. `SUPPORTED_QUANTILES=None` es correcto, pero los docs ("Any in (0, 1)") no mencionan el recorte. Evidencia: `evidence/chronos/compare_direct.log` caso E, `evidence/chronos/hub_revisions.log`, `evidence/chronos/capabilities_check.log`.
- **CHR-F9** [probado] `model_prediction_length` es 1024 (64 x 16: `max_output_patches` 64, `output_patch_size` 16) en los cuatro checkpoints. Con `steps=1040` la predicción es autorregresiva, finita (1040 filas), y el backend emite `UserWarning: We recommend keeping prediction length <= 1024`. La tabla de la guía dice "No hard limit, set via steps". Evidencia: `evidence/chronos/compare_direct.log` caso F, `evidence/chronos/hub_revisions.log`.
- **CHR-F10** [probado] Los cuatro repos (`amazon/chronos-2` `29ec3766d3`, `autogluon/chronos-2` `60088152a3`, `autogluon/chronos-2-small` `ddec01313e`, `autogluon/chronos-2-synth` `3607918a9f`) son apache-2.0 y `gated=False`; los cambios del 2026-06-05 solo tocan el README (config y pesos iguales antes y después). `_resolve_adapter` acepta `amazon/chronos-2*` y `autogluon/chronos-2*` y rechaza Bolt, T5, rutas locales y `s3://`. `_MODEL_LICENSES` (`skforecast/foundation/_utils.py`; Apache-2.0, sin restricción, coincide con los model cards y con el LICENSE del tag v2.3.2), `requires_hf_auth`, `requires_provider_auth`, `weights_repo_id` y `weights_in_hf_cache` son veraces. No hay que cambiar registro ni licencias. Evidencia: `evidence/chronos/hub_info.log`, `evidence/chronos/hub_commits.log`, `evidence/chronos/hub_revisions.log`, `evidence/chronos/resolve_adapter.log`, `evidence/chronos/capabilities_check.log`.
- **CHR-F11** [probado] Las capacidades declaradas son veraces frente a 2.3.2: contexto máximo 8192 (igual al `context_length` por defecto), `allow_exog=True`, covariables solo de pasado (diferencia 0 frente a la llamada directa), `supports_heterogeneous_covariates=False` (el backend lo exige), `requires_equal_length_context=False` (el backend rellena por la izquierda), NaN en el objetivo, `supports_categorical_covariates=True` con la salvedad de CHR-F5. Evidencia: `evidence/chronos/capabilities_check.log`, `evidence/chronos/verification/capabilities_check_rerun.log`, `evidence/chronos/compare_direct.log` caso B.
- **CHR-F12** (corregido por la verificación) [probado] No hay workarounds que quitar y el adaptador cumple `.claude/rules/foundation.md`: no tiene relleno, bucles por serie, fallbacks de MPS ni comprobaciones de versión. La conversión de numéricas, bool y dtypes nullable a float32 en `_to_covariate_array` sigue siendo necesaria (un array bool se trataría como categoría, porque `np.issubdtype(bool, np.number)` es False, ya desde 2.2.x; y el backend no convierte `pd.NA`). Forzar `cross_learning=False` con una sola serie en `ChronosAdapter.predict` es inocuo pero no necesario (diferencia 0.0 entre True y False con una entrada; la verificación lo anotó como CHR-V2). El `squeeze(0)` de la dimensión de variables es necesario. Evidencia: `evidence/chronos/verification/single_series_cross_learning.log`.
- **CHR-F13** [verificado en código] `FakePipeline` (`skforecast/foundation/tests/tests_foundation_models/fixtures_adapters.py`) acepta cualquier kwarg y cualquier esquema y devuelve numpy; el backend real rechaza kwargs desconocidos (`TypeError`) y esquemas heterogéneos (`ValueError`) y devuelve tensores. `test_ChronosAdapter.py` solo prueba un `Categorical` de strings (`test_ChronosAdapter_build_chronos_input_dtype_handling`); no hay tests de `Categorical` numérica, de contexto todo NaN ni de `predict_kwargs` con `cross_learning`, que da `TypeError: ... got multiple values for keyword argument 'cross_learning'` (error de Python, no del backend) [probado]. Evidencia: `evidence/chronos/unit_tests.log`, `evidence/chronos/verification/single_series_cross_learning.log`.
- **CHR-F14** [verificado en código] Textos desfasados o imprecisos:
  - Guía, celda 0: Chronos usa "quantized discrete tokens"; Chronos-2 usa embeddings de parches con InstanceNorm (v2.3.2 `chronos2/model.py`: `input_patch_embedding`, `Patch`, `InstanceNorm`).
  - Guía, celda 11 (tabla): omite `autogluon/chronos-2` y el horizonte recomendado de 1024 ("No hard limit").
  - `skforecast/foundation/_foundation_model.py`, docstring de la clase (línea 152): "`autogluon/chronos-*`" en vez de Chronos-2.
  - `SKILL.md` línea 82 y `adapter-parameters.md` línea 42: "CUDA > MPS > CPU" (ver CHR-F7).
  - `SKILL.md` línea 118 y `adapter-parameters.md` línea 44: descripción imprecisa del cross-learning (ver CHR-F6).
  - `llms-base.txt` línea 488 y `SKILL.md` línea 159: "Any in (0, 1)" sin el recorte (ver CHR-F8).
  - `llms-base.txt` línea 488 y `SKILL.md` línea 165: sin el prefijo `amazon/chronos-2` (ver CHR-V3).
  - Mensaje del `ImportError` de `_load_pipeline`: ">=2.0" (ver CHR-F3).
- **CHR-F15** [verificado en código] No hay opciones nuevas del backend que el adaptador deba exponer: `batch_size`, `context_length`, `limit_prediction_length` (declarados en `predict`), `max_output_patches`, `unrolled_quantiles` y `after_batch` (extraídos de `**kwargs`) ya se pasan por `predict_kwargs`. `dtype` es el nombre nuevo de `torch_dtype`, pero 2.3.x acepta los dos; con `torch_dtype=None`, 2.3.x pasa `dtype="auto"` y el modelo se carga en el dtype del checkpoint (float32 en los cuatro), sin impacto. Con 2.2.0 y transformers 4.57, `torch_dtype` solo produce un log de deprecación [probado, `evidence/chronos/old_backend_2.2.0.log`]. El PR #532 (commit `ff27460`, en 2.3.2) corrige una pérdida de precisión con bfloat16 al desescalar; solo importa porque los docs ponen `torch.bfloat16` como ejemplo.
- **CHR-V1** [verificado en código] En 2.0.x y 2.1.x el cross-learning existía con el nombre `predict_batches_jointly` (v2.1.0 `pipeline.py`, aprox. línea 381; también en `evidence/chronos/signature_2.1.0.log`); se renombró a `cross_learning` en 2.2.0 (commit `3cb8f88`, PR #418) y 2.2+ acepta todavía el nombre viejo con `FutureWarning` (v2.3.2 `pipeline.py`, líneas 570-577). chronos-forecasting 2.2.x fija transformers<5 y el soporte de transformers 5 llega en 2.3.0 (commit `88791cf`, PR #416) [documentado en los metadatos de PyPI, `evidence/chronos/pypi_info.log`]. Afecta a la elección de D12.
- **CHR-V3** [verificado en código] La tabla de adaptadores de `tools/ai/llms-base.txt` (línea 488) solo lista el prefijo `autogluon/chronos-2`, y `SKILL.md` (línea 165) dice `autogluon/chronos-2-*`. Ninguna cubre `amazon/chronos-2`, que el registro acepta (`_ADAPTER_REGISTRY`, aprox. líneas 5898-5899) y que la guía usa; la segunda tampoco cubre `autogluon/chronos-2` sin sufijo. `adapter-parameters.md` (línea 32) sí lista los dos prefijos.

### Descartado

No hay hallazgos refutados; estas afirmaciones de la primera pasada se descartaron al corregir los hallazgos:

- Que forzar `cross_learning=False` con una sola serie sea necesario: es inocuo, diferencia 0.0 (corrección de CHR-F12) [probado, `evidence/chronos/verification/single_series_cross_learning.log`].
- Que un array bool se trate como categoría a partir de 2.3.0: 2.2.x ya hacía lo mismo (corrección de CHR-F12) [verificado en código, v2.2.x `dataset.py` y v2.3.2 `preprocess.py`].
- Que el atributo se llame `supports_categorical_features`: es `supports_categorical_covariates` (corrección de CHR-F5) [verificado en código].
- Que un lote del backend tenga "como máximo" `batch_size` filas: puede superarlo (corrección de CHR-F6) [verificado en código, v2.3.2 `dataset.py` `_generate_sequential_batches`].

## 3. Cambios a implementar

Las líneas citadas son aproximadas y se refieren a `d9b8f4f`. El adaptador sigue siendo fino: nada de lo propuesto añade bucles, relleno ni agrupación en `ChronosAdapter`.

### CHR-P1. Ventanas de contexto sin valores observados (implementado en 02_common.md)

- **Qué**: nada en `ChronosAdapter`. La comprobación central (tras recortar a `context_length`, error si una serie no tiene ningún valor observado) y la corrección del comentario de `skforecast/model_selection/_validation.py` se implementan en `02_common.md`, en `FoundationModel`, para todos los adaptadores, como pide `.claude/rules/foundation.md` [verificado en código]. Aquí solo quedan la comprobación con Chronos real (`evidence/chronos/all_nan_context.py`, sección 4) y, si 02 no lo cubre con Chronos, un caso en `test_predict.py` (sección 4).
- **Por qué**: CHR-F4, CHR-F13.
- **Riesgo**: ninguno en este PR.
- **Rompe compatibilidad**: lo valora `02_common.md` (sí, para quien hoy recibe valores sin sentido, si se elige el `ValueError` de D11 (a)).
- **Decisión**: D11, en `02_common.md`. Si 02 aún no se ha fusionado cuando se abra este PR, no bloquear: el resto es independiente.

### CHR-P2. Declarar el mínimo de chronos-forecasting

- **Qué**:
  - `ChronosAdapter.backend_package = "chronos-forecasting>=2.2"` (o `">=2.3"` / `">=2.3.2"` según D12) en `skforecast/foundation/_adapters.py` (aprox. línea 201).
  - El mismo valor en `[chronos] backend_package` de `tools/foundation_compat/adapters.toml`: `test_adapters_toml_entries_match_the_adapter_classes` exige que coincidan y `test_adapters_toml_backend_package_is_a_PEP_508_requirement` que sea un requisito PEP 508 (ambos en `test_backends_compat_config.py`) [verificado en código].
  - `ChronosAdapter._load_pipeline` (aprox. líneas 446-452): quitar el ">=2.0" fijo del texto y entrecomillar el comando, como ya hacen `_import_timesfm` y `TabPFNAdapter._load_model` [verificado en código]:

    ```python
    raise ImportError(
        f"{self.backend_package} is required for ChronosAdapter. "
        f'Install it with `pip install "{self.backend_package}"`.'
    ) from exc
    ```

  - Actualizar la nota del docstring de `_load_pipeline` ("`chronos-forecasting` >=2.0 must be installed", aprox. líneas 439-440) para que remita a `backend_package`.
  - Solo si D-VER (b) se adopta en `02_common.md`: llamar al helper común de versión al principio de `_load_pipeline`, antes del `from chronos import BaseChronosPipeline`, como fija `02_common.md` (sección 3.4), para que una instalación con 2.0 o 2.1 dé un `ImportError` que nombre la versión mínima. Sin ese helper, el mínimo en `backend_package` solo protege instalaciones nuevas y el CI (X-4) [verificado en código: `backend_package` solo se usa en el mensaje, en `get_model_info` y en `tools/foundation_compat/run.py`]; no escribir una comprobación propia en el adaptador.
  - Con D12 (c) (sin mínimo), en lugar de lo anterior: pasar `cross_learning` solo cuando sea `True` en `ChronosAdapter.predict`. Funcionaría con 2.0 y 2.1 sin cross-learning [inferido], pero con `cross_learning=True` seguiría fallando con 2.0 y 2.1, salvo que se traduzca a `predict_batches_jointly` (CHR-V1), que es otra rama por versión.
- **Por qué**: CHR-F3, CHR-V1, CHR-F15, CHR-F14 (mensaje).
- **Riesgo**: bajo. pip con transformers 5 instalado resolverá 2.3.x aunque el mínimo sea 2.2, porque 2.2.x fija transformers<5 [inferido].
- **Rompe compatibilidad**: no para quien tiene >=2.2, que es lo que instala pip hoy [documentado]. Endurece el requisito del backend opcional; con 2.0 y 2.1 ya fallaba cada predicción.
- **Decisión**: D12 (versión) y D-VER de `02_common.md` (comprobación en ejecución).

### CHR-P3. Contrato: `cross_learning` en `Chronos2Pipeline.predict`

- **Qué**: en `BACKEND_CONTRACTS["chronos"]` de `tools/foundation_compat/tests/contracts.py`, añadir una entrada, con un comentario que explique que el adaptador llega a `predict` a través de `predict_quantiles`. La entrada actual de `predict_quantiles` (con `weak_kwargs`) se mantiene.

  ```python
  # `predict_quantiles` forwards `cross_learning` to `predict` through
  # `**predict_kwargs`: declared in `predict` since chronos-forecasting 2.2.0.
  {
      "target": "chronos:Chronos2Pipeline.predict",
      "kwargs": ["cross_learning"],
  },
  ```

- **Por qué**: CHR-F3. `test_contract.py` exige que los `kwargs` estén declarados en la firma; `Chronos2Pipeline.predict` declara `cross_learning` en 2.3.2 y no en 2.1.0, así que la entrada pasa con 2.3.2 y falla con 2.1.0 [verificado en código, `evidence/chronos/verification/contract_p3_check.log`].
- **Riesgo**: ninguno.
- **Rompe compatibilidad**: no.
- **Decisión**: no necesita.

### CHR-P4. Enviar un `pd.Categorical` como categórico aunque sus categorías sean números

- **Qué**: en `ChronosAdapter._to_covariate_array` (aprox. líneas 461-498), dentro de la rama de `pd.Series` y antes de la comprobación numérica, convertir los `CategoricalDtype` a array `object`:

  ```python
  if isinstance(col_data.dtype, pd.CategoricalDtype):
      return col_data.astype(object).to_numpy()
  ```

  Es exactamente el cambio aplicado como monkeypatch en `evidence/chronos/backtest_patch.py`. En el backend, un array `object` con enteros cae en la rama de categoría (`np.issubdtype(object, np.number)` es False) [verificado en código], que es lo que hace `predict_df` upstream con un `pd.Categorical`. El mismo método sirve a covariables pasadas y futuras (`_build_chronos_input`), así que el cambio cubre las dos [verificado en código]. Opcional, por coherencia: tratar igual un `pd.Categorical` que llegue por la rama de fallback (no pandas Series); `_build_chronos_input` siempre pasa `df[col]`, una Series [verificado en código]. Actualizar los docstrings de `_to_covariate_array` y `_build_chronos_input` para que digan que un `Categorical` se envía como `object` sea cual sea el tipo de sus categorías.
- **Por qué**: CHR-F5 (y la parte de CHR-F13 sobre su cobertura).
- **Riesgo**: cambia predicciones solo para quien usa exog `category` con categorías numéricas (o booleanas, ver abajo). El backtest de la guía (vic_electricity, `autogluon/chronos-2-small`, exog `Temperature` y `Holiday`, `context_length=500`, `steps=24`) da MAE 171.267 antes y después, diferencia máxima entre predicciones 0.0, porque la guía no usa `category` [probado; chronos-forecasting 2.3.2, torch 2.15.0+cpu, transformers 5.22.1, numpy 2.5.4, pandas 2.3.3, Python 3.12.3, Intel Xeon 2.10 GHz, 4 CPU lógicas, Linux; `evidence/chronos/backtest_before.log`, `evidence/chronos/backtest_after.log`, `evidence/chronos/verification/backtest_before_rerun.log`, `evidence/chronos/verification/backtest_after_rerun.log`]. Dos casos límite sin probar contra el backend:
  - Un `Categorical` con valores ausentes se convierte en `object` con `nan`, como una columna string con NaN [inferido]: incluirlo en la comprobación directa de la sección 4.
  - Un `Categorical` de booleanos hoy pasa por la rama numérica (`pd.api.types.is_bool_dtype` devuelve True para un `CategoricalDtype` de booleanos) y llega como float32; con el cambio, al ir antes de esa comprobación, llegaría como `object` y sería categórico [verificado en código de pandas 2.3.3; efecto en el backend inferido]. Es coherente con el objetivo (un `category` es categórico), pero hay que decidirlo explícitamente, fijarlo con un test y citarlo en la release note.
- **Rompe compatibilidad**: cambia resultados para esos usuarios; va en "Before upgrading" de las release notes.
- **Decisión**: D13. Con D13 (b), no tocar el código y corregir solo los docstrings y el SKILL: "solo las categorías no numéricas se tratan como categóricas".

### CHR-P5. Documentar que el cross-learning depende de `batch_size`

- **Qué**: en el docstring de `ChronosAdapter` (parámetro `cross_learning` de la clase, aprox. líneas 110-116, y de `__init__`, aprox. líneas 246-249), en `SKILL.md` (líneas 118-119) y en `adapter-parameters.md` (línea 44), explicar que el cross-learning se aplica dentro de cada lote del backend (`batch_size` de `predict_kwargs`, 256 por defecto, contando filas de objetivo y covariables; un lote puede superarlo porque se añaden series enteras), que el resultado depende de `batch_size`, y que upstream recomienda un `batch_size` de alrededor de 100. Texto orientativo:

  ```text
  Cross-learning is applied within each batch of the backend, not within the
  whole group of series that share covariates: the batch is set by
  `predict_kwargs={"batch_size": ...}` (256 rows by default, target and
  covariates included). Results depend on `batch_size`; the chronos-forecasting
  documentation suggests a value around 100.
  ```

- **Por qué**: CHR-F6. Los textos generales sobre independencia entre series (docstring de `FoundationModel.predict`, `llms-base.txt` línea 531, X-8) no son de este PR.
- **Riesgo**: ninguno.
- **Rompe compatibilidad**: no.
- **Decisión**: no necesita.

### CHR-P6. `device_map="auto"`: documentar el comportamiento real o resolverlo como T0Adapter

- **Qué**:
  - Opción (a), recomendada: corregir la descripción "CUDA > MPS > CPU" en el docstring de `ChronosAdapter` (clase, aprox. líneas 103-107, e `__init__`, aprox. líneas 238-242), en `SKILL.md` (comentario de la línea 82) y en `adapter-parameters.md` (línea 42): "`'auto'` is resolved by Hugging Face accelerate: the model may be split across several GPUs and offloaded to CPU or disk when memory is short; on Apple Silicon it goes to MPS (or disk). Pass `'cuda'`, `'mps'` or `'cpu'` to use a single device." Sin cambio de código. Opcional: precisar el docstring de `test_ChronosAdapter_load_pipeline_passes_device_map_to_from_pretrained` en `test_device_handling.py` ("Chronos handles 'auto' internally" pasa a "accelerate resolves 'auto'").
  - Opción (b): en `ChronosAdapter._load_pipeline`, resolver con `_resolve_torch_device(self.device_map)` antes de `from_pretrained`, como `T0Adapter._load_model` (aprox. línea 3804). `_resolve_torch_device` devuelve tal cual los valores distintos de `"auto"` [verificado en código].
- **Por qué**: CHR-F7, CHR-F14.
- **Riesgo**: (a) ninguno. (b) cambia la colocación del modelo en máquinas con varias GPUs o con poca memoria de GPU, y no se puede probar sin GPU ni MPS.
- **Rompe compatibilidad**: no en CPU; (b) sí cambia la colocación con varias GPUs.
- **Decisión**: D14.

### CHR-P7. Textos de Chronos-2 (incluye CHR-V3)

- **Qué**: los cambios de texto de la sección 5: arquitectura por parches en la celda 0 de la guía; tabla de la guía con `autogluon/chronos-2` y horizonte recomendado de hasta 1024 pasos; docstring de `FoundationModel` (línea 152) con `amazon/chronos-2*` y `autogluon/chronos-2*`; prefijos `amazon/chronos-2` y `autogluon/chronos-2` en `llms-base.txt` (línea 488) y `SKILL.md` (línea 165); recorte de los cuantiles fuera de [0.01, 0.99] en `llms-base.txt` (488), `SKILL.md` (159), `adapter-parameters.md` (línea 34) y `skills/prediction-intervals/SKILL.md` (línea 150, que también dice "any quantile in (0, 1)" para Chronos-2 [verificado en código]); notas de cuantiles y horizonte en el docstring de `ChronosAdapter`.
- **Por qué**: CHR-F8, CHR-F9, CHR-F14, CHR-V3.
- **Riesgo**: ninguno. Otros PR tocan líneas vecinas de los mismos ficheros: 03 y 07 editan también la línea 159 de `SKILL.md`; 02, 03 y 07 (y 08 y 09 según D18 y D25) editan otras filas de la tabla de adaptadores de `llms-base.txt`; 02, 03, 06 y 09 editan otras filas de la tabla de `SKILL.md`. Ningún otro plan toca `skills/prediction-intervals/SKILL.md`. Tocar solo la parte de Chronos-2 y rebasar si hay conflicto.
- **Rompe compatibilidad**: no.
- **Decisión**: no necesita.

### CHR-P8. Completar los tests de ChronosAdapter y reservar los kwargs gestionados por el adaptador

- **Qué**:
  - Tests de `Categorical` numérica (tras CHR-P4) y de `predict_kwargs` con `cross_learning` (sección 4). El de contexto todo NaN depende de CHR-P1 (sección 4).
  - Rechazar en `__init__` y en `set_params` un `predict_kwargs` que contenga argumentos que el adaptador ya pasa, con un `ValueError` claro en lugar del `TypeError` "multiple values" que hoy sale en `predict` [probado, CHR-F13]. Seguir el patrón existente de `TimesFM3Adapter` y `TimestralAdapter` (atributo `_RESERVED_PREDICT_KWARGS` y el classmethod `_validate_predict_kwargs`, llamado desde `__init__` y desde la función `validate` que `set_params` pasa a `_apply_set_params`) [verificado en código]:

    ```python
    _RESERVED_PREDICT_KWARGS: frozenset[str] = frozenset({
        "inputs", "prediction_length", "quantile_levels", "cross_learning",
    })
    ```

    Valorar añadir `predict_batches_jointly`: en 2.2+ su valor sustituye a `cross_learning` con un `FutureWarning` (v2.3.2 `pipeline.py`, líneas 570-577) [verificado en código], así que dentro de `predict_kwargs` saltaría el parámetro `cross_learning` del adaptador.
  - Alternativa: no validar y documentar en el docstring de `predict_kwargs` que esas claves no se admiten.
  - Opcional (relacionado con XC-7): acercar `FakePipeline.predict_quantiles` al backend, rechazando con `TypeError` los kwargs que `Chronos2Pipeline.predict` no acepta (los declarados `batch_size`, `context_length`, `cross_learning`, `limit_prediction_length` y los que extrae de `**kwargs`: `max_output_patches`, `unrolled_quantiles`, `after_batch`, `predict_batches_jointly`) [verificado en código, v2.3.2]. Comprobar antes que ningún test existente pase otros kwargs.
- **Por qué**: CHR-F13.
- **Riesgo**: bajo. El `ValueError` aparece al construir el modelo en vez de al predecir.
- **Rompe compatibilidad**: no en la práctica: esas combinaciones ya fallaban con `TypeError`.
- **Decisión**: el informe no le asigna un id; ver la fila "sin id (CHR-P8)" de la sección 6.

## 4. Tests

Unitarios (sin backend, con `FakePipeline`):

- `skforecast/foundation/tests/tests_foundation_models/test_ChronosAdapter.py`:
  - `test_ChronosAdapter_build_chronos_input_dtype_handling`: añadir un caso `pd.Categorical([0, 1, 2, 3] * 7 + [0, 1])` (30 valores, como el resto) a la parametrización y a `ids`, con kind esperado `"O"` (con D13 (a)); con D13 (b), el caso fija el comportamiento actual (kind `"i"`) y el docstring lo explica.
  - Nuevo `test_ChronosAdapter_to_covariate_array_categorical_with_numeric_categories_returns_object`: categorías enteras, float y booleanas, con y sin valores ausentes; el array devuelto es `object` y conserva los valores (con D13 (a)).
  - Nuevo `test_ChronosAdapter_predict_categorical_numeric_covariate_forwarded_as_object`: `predict` con exog pasada y futura `category` de enteros; `pipeline.last_inputs[0]["past_covariates"]["<col>"]` y `["future_covariates"]["<col>"]` tienen dtype `object`.
  - Nuevo `test_ChronosAdapter_init_ValueError_when_predict_kwargs_has_reserved_keys` (parametrizado con `cross_learning`, `inputs`, `prediction_length`, `quantile_levels`) y un caso más en la parametrización de `test_ChronosAdapter_set_params_ValueError_when_invalid` con `{"predict_kwargs": {"cross_learning": True}}`. Si se elige documentar en lugar de validar, un test que fije el `TypeError` actual no aporta: no añadirlo.
  - Nuevo `test_ChronosAdapter_load_pipeline_ImportError_message_quotes_backend_package`: simular la ausencia de `chronos` con `monkeypatch` de `builtins.__import__` (patrón de `test_NoriAdapter_load_model_ImportError_when_backend_missing` en `test_NoriAdapter.py` y de `test_MoiraiAdapter_load_module_ImportError_no_LicenseWarning` en `test_MoiraiAdapter.py`) y comprobar que el mensaje contiene `pip install "chronos-forecasting>=2.2"` (o el mínimo elegido en D12) y no contiene ">=2.0 is required".
- `skforecast/foundation/tests/tests_foundation_models/test_predict.py`: el test central de contexto todo NaN lo añade `02_common.md`; el fichero ya usa `FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())` [verificado en código]. Si 02 lo escribe con otro adaptador, añadir aquí un caso con Chronos, no en `test_ChronosAdapter.py` (el adaptador recibe la ventana ya recortada).
- `skforecast/foundation/tests/tests_foundation_models/test_get_model_info.py`: actualizar el `backend_package` esperado en `test_get_model_info_output_Chronos` (línea 60) y en la tupla de `autogluon/chronos-2-small` de la parametrización (línea 161).
- `skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py`: sin cambios; debe seguir pasando con el nuevo valor en `adapters.toml`.
- `skforecast/foundation/tests/tests_foundation_models/test_device_handling.py`: solo con D14 (b), cambiar `test_ChronosAdapter_load_pipeline_passes_device_map_to_from_pretrained`, que hoy afirma que `"auto"` llega sin resolver a `BaseChronosPipeline.from_pretrained`, para que compruebe el valor resuelto (parcheando `torch.cuda.is_available` y `torch.backends.mps.is_available`, como los tests de `_resolve_torch_device` del mismo fichero). Con D14 (a), solo el retoque opcional de su docstring.
- Fakes: `FakePipeline` de `fixtures_adapters.py` (opcional, ver CHR-P8). La `FakePipeline` de `skforecast/foundation/tests/tests_forecaster_foundation/fixtures_forecaster_foundation.py` no necesita cambios.

Compatibilidad (`tools/foundation_compat/`):

- `tests/contracts.py`: entrada nueva de CHR-P3. Con 2.3.2, `test_contract.py --backend chronos` pasa de 3 a 4 tests [inferido: un test por entrada más `test_backend_has_contract`]; con 2.1.0 la entrada nueva falla.
- `adapters.toml`: `backend_package` de `[chronos]` (CHR-P2). El workflow `.github/workflows/foundation-backends.yml` no cambia aquí. Un carril del CI de compatibilidad con la versión mínima de cada backend es una subdecisión de D-VER que ningún plan de esta serie implementa (`10_docs_ci.md`, sección 8).
- `tests/test_smoke.py`: sin cambios.

Comprobaciones con el backend real (antes en `origin/0.27.x`, después en la rama):

```bash
PY=~/.venvs/fm-review-chronos/bin/python
OLD=~/.venvs/fm-review-chronos-old/bin/python

# Adaptador frente a llamada directa (casos A a H)
$PY dev/foundation_adapters_review/evidence/chronos/compare_direct.py autogluon/chronos-2-small autogluon/chronos-2-synth
# Firma de predict (base de CHR-P3)
$PY dev/foundation_adapters_review/evidence/chronos/signature_check.py
$OLD dev/foundation_adapters_review/evidence/chronos/signature_check.py
# Backend antiguo (CHR-P2): 2.1.0 y, temporalmente, 2.2.0
$OLD dev/foundation_adapters_review/evidence/chronos/old_backend_check.py
uv pip install --python $OLD "chronos-forecasting==2.2.0"
$OLD dev/foundation_adapters_review/evidence/chronos/old_backend_check.py
uv pip install --python $OLD "chronos-forecasting==2.1.0"
# Contexto todo NaN (CHR-P1, cuando 02 esté fusionado)
$PY dev/foundation_adapters_review/evidence/chronos/all_nan_context.py
# Contrato y smoke
$PY -m pytest tools/foundation_compat/tests/test_contract.py --backend chronos -q -p no:cacheprovider
$OLD -m pytest tools/foundation_compat/tests/test_contract.py --backend chronos -q -p no:cacheprovider
$PY -m pytest tools/foundation_compat/tests/test_smoke.py --backend chronos -q -p no:cacheprovider
python tools/foundation_compat/run.py --adapter chronos
```

Sobre el caso C de `compare_direct.py`: su referencia "categórica" convierte la columna a `str` (`astype(str)`), mientras que el adaptador enviará enteros en un array `object`. Que la diferencia sea 0.0 con el cambio es lo esperado si la codificación categórica del backend solo depende de la identidad de los valores [inferido]; si no da 0.0, añadir una referencia con `astype(object)` sin pasar a `str` y comparar con ella. Ampliar también la comprobación con un `Categorical` que tenga algún valor ausente y con un `Categorical` de booleanos (riesgos de CHR-P4) antes de dar el cambio por bueno. El script vive en `dev/`, fuera del PR; el resultado se cita en la descripción del PR.

Backtest de la guía, solo para CHR-P4 (el único cambio que puede alterar predicciones), como comprobación de corrección antes y después, no como benchmark (el tiempo que imprime no se usa):

```bash
$PY dev/foundation_adapters_review/evidence/common/backtest_guide.py \
    --model-id autogluon/chronos-2-small --exog --out backtest_before.csv   # en origin/0.27.x
$PY dev/foundation_adapters_review/evidence/common/backtest_guide.py \
    --model-id autogluon/chronos-2-small --exog --out backtest_after.csv    # en la rama
$PY -c 'import pandas as pd; a = pd.read_csv("backtest_before.csv"); b = pd.read_csv("backtest_after.csv"); print((a["pred"] - b["pred"]).abs().max())'
```

Resultado esperado: MAE 171.267 en los dos y diferencia máxima 0.0 [probado en la revisión con el monkeypatch de `evidence/chronos/backtest_patch.py`]. Borrar los CSV después; no se versionan.

## 5. Documentación y contexto de IA

Docstrings (`skforecast/foundation/_adapters.py`, `ChronosAdapter`):

- Parámetro `cross_learning` (clase e `__init__`): texto de CHR-P5.
- Parámetro `device_map` (clase e `__init__`): texto de CHR-P6 según D14.
- `predict_kwargs` (clase e `__init__`): claves reservadas (CHR-P8).
- `SUPPORTED_QUANTILES` o `Notes`: los niveles fuera de [0.01, 0.99] devuelven el valor del extremo (el backend solo lo registra en el log); por encima de 1024 pasos la predicción es autorregresiva y el backend emite un `UserWarning`.
- `_load_pipeline`, `_to_covariate_array`, `_build_chronos_input`: textos de CHR-P2 y CHR-P4.
- `skforecast/foundation/_foundation_model.py`, docstring de la clase (atributo `adapter`, línea 152): "`ChronosAdapter` for `amazon/chronos-2*` and `autogluon/chronos-2*` models".

Guía `docs/user_guides/foundation-forecasting-models.ipynb` (solo texto de Chronos; no reejecutar, lo hace `10_docs_ci.md`):

- Celda markdown de introducción "Forecasting with foundation models" (celda 0 en la revisión), párrafo "Models such as **Amazon Chronos**...": sacar Chronos de "quantized discrete tokens (as in Chronos, which applies scalar quantization)" y citar Chronos-2 entre los modelos de "continuous patch embeddings" (Chronos-2 usa parches con InstanceNorm).
- Celda markdown/HTML con la tabla de modelos soportados (celda 11 en la revisión), fila "Chronos": añadir `autogluon/chronos-2` a la lista de ids (`amazon/chronos-2<br>autogluon/chronos-2<br>autogluon/chronos-2-small<br>autogluon/chronos-2-synth`); columna "Max horizon": "1024 recommended (longer horizons are predicted autoregressively, with a warning)" en lugar de "No hard limit, set via <code>steps</code>"; comando de instalación con el mínimo de D12, escapado como en la fila de TabPFN-TS: `<code>pip install "chronos-forecasting&gt;=2.2"</code>`, `data-copy='pip install "chronos-forecasting&gt;=2.2"'` y el `aria-label` correspondiente.
- La salida de `list_adapters(as_frame=True)` (celda de código 13 en la revisión) mostrará el nuevo `backend_package` solo tras la reejecución de `10_docs_ci.md`.

Skills y referencia de IA:

- `skills/foundation-forecasting/SKILL.md`: línea 56 (`pip install "chronos-forecasting>=2.2"`), línea 82 (comentario de `device_map`), líneas 118-119 (cross-learning y `batch_size`), línea 159 (recorte de cuantiles de Chronos-2), línea 165 (prefijos `amazon/chronos-2` y `autogluon/chronos-2`) y, con D13 (a), el punto 8 de errores comunes (línea 285) puede aclarar que un `category` con códigos numéricos se trata como categórico en Chronos-2.
- `skills/foundation-forecasting/references/adapter-parameters.md`: línea 34 (cuantiles), línea 42 (`device_map`), línea 44 (`cross_learning`), y una línea sobre el horizonte recomendado de 1024 y las claves reservadas de `predict_kwargs`.
- `skills/prediction-intervals/SKILL.md` línea 150: recorte de Chronos-2 a [0.01, 0.99].
- `tools/ai/llms-base.txt` línea 488: prefijos `amazon/chronos-2`, `autogluon/chronos-2`; cuantiles "Any in `(0, 1)`; levels outside `[0.01, 0.99]` return the 0.01 or 0.99 value".
- Opcional: `README.md` línea 69 (`# pip install chronos-forecasting`) con el mínimo de D12, por coherencia con la guía.
- Regenerar con el skill `ai-context-sync` (`python tools/ai/generate_ai_context_files.py` y `python tools/ai/generate_ai_context_files.py --check`). No editar a mano `AGENTS.md`, `llms-full.txt` ni `docs/llms*.txt`.

Release notes (`docs/releases/releases.md`, sección 0.27.0 en desarrollo, que hoy agrupa las entradas en **Added**, **Changed** y **Fixed** sin subapartados por área; seguir el skill `release-note` y ajustar a lo decidido en D12, D13, D14, CHR-P8 y al número de PR):

- En **Fixed**:

  ```text
  + <code>[FoundationModel]</code> and <code>[ForecasterFoundation]</code> with Chronos-2 pass an exogenous variable of `category` dtype whose categories are numbers (for example, the day of the week) to the model as a categorical covariate. It was passed as a numeric one. ([#XXXX](https://github.com/skforecast/skforecast/pull/XXXX))

  + Chronos-2 requires `chronos-forecasting>=2.2`: older releases do not accept `cross_learning` and failed on every prediction with a `TypeError`. The install command of the `ImportError` is quoted so that it can be pasted in a shell. ([#XXXX](https://github.com/skforecast/skforecast/pull/XXXX))

  + `ChronosAdapter` raises a `ValueError` when `predict_kwargs` contains an argument the adapter already passes (`cross_learning`, `inputs`, `prediction_length`, `quantile_levels`); it failed later with a `TypeError`. ([#XXXX](https://github.com/skforecast/skforecast/pull/XXXX))
  ```

- En el bloque "Before upgrading", lista "Some results change with this version, without any change in your code":

  ```text
  + <code>[FoundationModel]</code> and <code>[ForecasterFoundation]</code> with Chronos-2 and an exogenous variable of `category` dtype with numeric (or boolean) categories: predictions change, because the variable is now encoded as categorical.
  ```

- En **Changed**, documentación:

  ```text
  + Chronos-2 documentation: cross-learning depends on the `batch_size` of the backend, quantile levels outside `[0.01, 0.99]` return the 0.01 or 0.99 value, horizons above 1024 steps are predicted autoregressively, and `device_map='auto'` is resolved by Hugging Face accelerate. ([#XXXX](https://github.com/skforecast/skforecast/pull/XXXX))
  ```

## 6. Decisiones pendientes

| Id | Decisión | Opciones | Recomendación | Impacto en el plan | Estado |
|---|---|---|---|---|---|
| D12 | Mínimo de chronos-forecasting (CHR-P2) | (a) `>=2.2`, mínimo funcional (2.2.x fija transformers<5); (b) `>=2.3.2`: transformers 5 y arreglo de bfloat16 (#532); una variante es `>=2.3`, que garantiza transformers 5 pero no el arreglo de bfloat16, que llegó en 2.3.2; (c) sin mínimo y pasar `cross_learning` solo si es True | (a), según el informe; (b) si se quiere garantizar transformers 5. La revisión transversal recomienda `>=2.3` por CHR-V1 | Valor de `backend_package`, `adapters.toml`, mensaje del `ImportError`, comandos de instalación, `test_get_model_info.py` y release note. Con (c) no cambia `backend_package` y cambia `ChronosAdapter.predict` | pendiente |
| D13 | `pd.Categorical` con categorías numéricas (CHR-P4) | (a) enviarlo como categórico; (b) mantener el comportamiento y documentarlo | (a): coincide con `predict_df` upstream | Con (a): cambio de código, tests, "Before upgrading" y backtest antes y después; decidir también si un `Categorical` de booleanos pasa a categórico. Con (b): solo docstrings, SKILL y un test que fije el comportamiento | pendiente |
| D14 | `device_map="auto"` (CHR-P6) | (a) documentar que lo resuelve accelerate; (b) resolverlo con `_resolve_torch_device` como `T0Adapter` | (a): sin cambio de código | Con (b): cambio en `_load_pipeline`, cambio del test existente de `test_device_handling.py`, release note y riesgo no comprobable sin GPU | pendiente |
| sin id (CHR-P8) | `predict_kwargs` con argumentos que el adaptador ya pasa | (a) `ValueError` con `_RESERVED_PREDICT_KWARGS` en `__init__` y `set_params`, como `TimesFM3Adapter` y `TimestralAdapter`; (b) solo documentarlo | (a), por coherencia con los adaptadores que ya lo hacen | Con (a): código, dos tests y release note. Con (b): una frase en el docstring | pendiente |

Decisiones de otros planes que afectan a este: D11 (contexto todo NaN, CHR-P1) y D-VER (helper de versión en ejecución, CHR-P2), en `02_common.md`.

## 7. Criterios de aceptación

- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_ChronosAdapter.py -x -q`: todos pasan (55 en la revisión más los nuevos).
- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_get_model_info.py skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py skforecast/foundation/tests/tests_foundation_models/test_device_handling.py -q`: todos pasan.
- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/ -q` una vez al terminar (módulo tocado; no la suite completa).
- [ ] `~/.venvs/fm-review-chronos/bin/python -m pytest skforecast/foundation/tests/tests_foundation_models/test_ChronosAdapter.py -q -p no:cacheprovider`: pasan también con el backend real 2.3.2 instalado.
- [ ] `test_contract.py --backend chronos`: pasa con 2.3.2 (4 tests [inferido]) y falla en la entrada nueva de `Chronos2Pipeline.predict` con 2.1.0.
- [ ] `test_smoke.py --backend chronos` con 2.3.2: 6 passed.
- [ ] `python tools/foundation_compat/run.py --adapter chronos`: status passed, `held_back` false.
- [ ] `compare_direct.py autogluon/chronos-2-small autogluon/chronos-2-synth`: casos A, B (cross_learning False y True) y H con diferencia 0.0 frente a la llamada directa; caso C (con D13 (a)): el adaptador construye un array `object` y la diferencia frente a la llamada directa con la columna como categoría es 0.0 (antes 0.348; ver la nota de la sección 4 sobre la referencia `str`); caso D: el mismo `TypeError` de `FoundationModel`; casos E, F y G sin cambios (recorte a 0.01/0.99 con diferencia 0, `UserWarning` por encima de 1024, 0.540 entre `batch_size` 256 y 2).
- [ ] `old_backend_check.py` en el entorno con 2.1.0: con D-VER (b), `ImportError` que nombra la versión mínima; sin helper, el `TypeError` de siempre (el mínimo solo cubre instalaciones nuevas). Con 2.2.0 (si D12 (a)): OK `(12, 2)`.
- [ ] `all_nan_context.py`, una vez fusionado `02_common.md`: el error definido en D11 que nombra `tail_nan` en los tres casos, en lugar de cuantiles de ±1e5.
- [ ] Backtest de la guía antes y después de CHR-P4: MAE 171.267 en ambos, diferencia máxima 0.0.
- [ ] `python tools/ai/generate_ai_context_files.py --check` sin diferencias.
- [ ] Entrada en `docs/releases/releases.md` (skill `release-note`).
- [ ] Skill `verify` en verde antes del commit.

## 8. Fuera de alcance

- La comprobación central de contexto todo NaN y el comentario de `_validation.py` (CHR-P1, D11): `02_common.md`.
- El helper común de comprobación de versión en tiempo de ejecución (D-VER) y la tabla central de ids no soportados: `02_common.md`.
- Reejecutar la guía de usuario y cualquier cambio en `.github/workflows/foundation-backends.yml`: `10_docs_ci.md`. Un carril del CI de compatibilidad con la versión mínima de cada backend es una subdecisión de D-VER que ningún plan de esta serie implementa (`10_docs_ci.md`, sección 8).
- Los textos generales sobre independencia de las series en un lote (docstring de `FoundationModel.predict`, `llms-base.txt` línea 531, guía de series heterogéneas; X-8): no van en este PR. Las demás invariantes genéricas del smoke (XC-7 de `evidence/cross_cutting.md`) quedan fuera de esta serie de PR (`10_docs_ci.md`, sección 8); si se quieren, van en un PR aparte `chore/foundation-compat-checks` (PR-12 de `evidence/cross_cutting.md`), después de 01 y 03.
- Exponer opciones nuevas del backend (`batch_size`, `max_output_patches`, `unrolled_quantiles`, `dtype`): ya se pasan por `predict_kwargs` o se aceptan (CHR-F15).
- Quitar la rama que fuerza `cross_learning=False` con una sola serie: es inocua (CHR-F12); solo se toca si D12 (c) obliga a reescribir esa línea.
- Cambios en el registro, las licencias o los atributos de capacidad (CHR-F10, CHR-F11).
- Añadir checkpoints nuevos a la guía: no hay ninguno desde 2025-12-03.
- Comprobar `device_map` en CUDA o MPS: no hay GPU en la revisión.
- Medir velocidad o precisión entre versiones del backend, checkpoints o dtypes: fuera del alcance de skforecast. El backtest de la guía solo se usa como comprobación de corrección antes y después de CHR-P4.
