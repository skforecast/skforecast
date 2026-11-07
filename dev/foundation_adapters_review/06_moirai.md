# 06. Moirai-2 (uni2ts)

- **Rama propuesta**: `fix/moirai-alignment`, creada desde `origin/0.27.x`.
- **Prioridad**: media (MOI-P1 media; MOI-P2, MOI-P3, MOI-P4, MOI-P6 y MOI-V1 baja).
- **Adaptadores afectados**: `MoiraiAdapter`. El arreglo compartido de `stacklevel` (MOI-V1: `_warn_if_non_commercial` y `_resolve_torch_device`) está en `02_common.md`, sección 3.5; aquí solo se quita el `stacklevel=7` propio de Moirai.
- **Ficheros afectados**:
  - `skforecast/foundation/_adapters.py` (`MoiraiAdapter`: `_ensure_forecast_obj` deja de pasar `stacklevel=7`).
  - `skforecast/foundation/tests/tests_foundation_models/`: `test_MoiraiAdapter.py`, `test_device_handling.py`, `test_get_model_info.py`, `fixtures_adapters.py`.
  - `tools/foundation_compat/adapters.toml`, `tools/foundation_compat/tests/contracts.py`.
  - `docs/user_guides/foundation-forecasting-models.ipynb` (solo texto de Moirai, sin reejecutar).
  - `skills/foundation-forecasting/SKILL.md`, `skills/foundation-forecasting/references/adapter-parameters.md`, `tools/ai/llms-base.txt` y los ficheros generados.
  - `docs/releases/releases.md`.
- **Depende de**: `02_common.md`. De ahí vienen la comprobación de contexto todo NaN (MOI-P5), el helper opcional de comprobación de versión (D-VER, que usaría MOI-P1), la conversión de los niveles de cuantil a float y la tabla central de ids no soportados. No hay dependencia funcional con 03, 04 ni 05, pero tocan zonas cercanas de `_adapters.py` (y 03 puede tocar `_resolve_torch_device` con TSI-P5): se recomienda fusionar este PR después de ellos, en el orden de la serie, para evitar conflictos [inferido].
- **Bloquea a**: `10_docs_ci.md`, que reejecuta la guía (la salida de la celda 68 en la revisión cambia la ubicación del `LicenseWarning`). No bloquea a 07, 08 ni 09 (sus `stacklevel=3` quedan para un seguimiento posterior a 02, que traslada el helper en la sección 3.5).
- **Cierra issue**: ninguna de skforecast. Referencias upstream (SalesforceAIResearch/uni2ts), todas abiertas en la revisión [probado]:
  - #216 y #256: PR para relajar dependencias y soportar Python 3.13.
  - #251: torch<2.5.
  - #133: tests con torch 2.5.
  - #224: caída de precisión con covariables `feat_dynamic_real`.

## 1. Estado en la revisión (2026-11-06)

**Versiones probadas y última disponible.**

| Elemento | Probado antes (CI) | Último en la revisión |
|:--|:--|:--|
| uni2ts (PyPI) | 2.0.0 | 2.0.0, publicado el 2025-11-04 [probado] |
| Código upstream | tag 2.0.0 = 8062ef5 | `main` cfd46d4 (2026-06-02), sin cambios en `src/` ni en `pyproject.toml` desde 2.0.0 [probado] |
| Checkpoint | `Salesforce/moirai-2.0-R-small` | igual: sha 30f43ff (last_modified 2026-01-29) [probado] |

Detalle de las versiones de uni2ts:

- El wheel de 2.0.0 es idéntico a `src/uni2ts` del tag (`diff -rq`) [probado].
- Versiones anteriores: 1.1.0, 1.1.1 y 1.2.0. Ninguna contiene `uni2ts.model.moirai2` (ausente en los tags 1.1.0 y 1.2.0 y en el wheel 1.1.1) [probado].
- Evidencia: `evidence/moirai/pypi_uni2ts.log`, `evidence/moirai/upstream_diff.log`, `evidence/moirai/verification/verification.md`.

Requisitos de uni2ts 2.0.0 [probado]:

- `requires_python >=3.10`, pero `torch<2.5,>=2.1`: en la práctica solo se resuelve con Python <= 3.12.
- Fija además `scipy~=1.11.3`, `numpy~=1.26.0` y `gluonts~=0.14.3`. Todas las versiones de uni2ts fijan `scipy~=1.11.3` y `numpy~=1.26.0`.
- uni2ts 1.1.1 exige `torch>=2.1` sin techo; 1.2.0 y 2.0.0, `torch>=2.1,<2.5`.
- gluonts 0.14.4 exige `pandas<2.2.0`; gluonts 0.14.3, `pandas<3`.
- Evidencia: `evidence/moirai/pypi_uni2ts.log`, `evidence/moirai/verification/resolve_check.log`.

Checkpoints en el Hub [probado]:

- `Salesforce/moirai-2.0-R-small`:
  - Pesos y `config.json` en el commit f2ed354 (2025-08-06); los commits posteriores solo cambian el README (2025-08-11, 2026-01-16, 2026-01-29).
  - Licencia cc-by-nc-4.0, no gated, sin fichero LICENSE, 45.6 MB.
  - `config.json`: `max_seq_len=512`, `patch_size=16`, `num_predict_token=4`, `quantile_levels` de 0.1 a 0.9.
- `-base` y `-large` dan 404.
- Nuevo `Salesforce/moirai-agent` (2025-12-29): otro tipo de modelo, que el registro rechaza.
- `Salesforce/moirai-1.5-llama-kvcache` (2025-06-26): gated manual, sin licencia declarada, no se enruta a `MoiraiAdapter`.
- Evidencia: `evidence/moirai/hub_info.log`, `evidence/moirai/hub_moirai2_small.log`, `evidence/moirai/hub_moirai2_sizes.log`, `evidence/moirai/hub_collections.log`.

**Línea base del CI** [probado]:

- Run 42212742430 del workflow Foundation Backends Compatibility, job 127875169007 (moirai), 2026-11-06, rama 0.27.x, commit 09433c35.
- Entorno: Python 3.12.16, uni2ts 2.0.0, torch 2.4.1+cpu, numpy 1.26.4, scipy 1.17.1 (override `scipy>=1.12`), pandas 2.3.3, huggingface-hub 2.4.0, gluonts 0.14.3, jax 0.7.1, lightning 2.6.6, einops 0.7.0.
- Resultados: contrato 5 passed; smoke 5 passed y 1 skipped (`test_predict_output_when_exog`, `allow_exog=False`).
- El log del job muestra ya el `LicenseWarning` apuntando a `_adapters.py:2069` (MOI-V1).
- Evidencia: `evidence/moirai/ci_baseline.log`.

**Entorno de la revisión** [probado]:

- `~/.venvs/fm-review-moirai`: Python 3.12.3, skforecast 0.27.0 editable, uni2ts 2.0.0, torch 2.4.1+cpu, numpy 1.26.4, scipy 1.17.1, pandas 2.3.3, huggingface_hub 2.4.0, gluonts 0.14.3.
- Máquina: CPU x86_64 con 4 núcleos, sin GPU.
- Resultados: contrato 5 passed; smoke 5 passed y 1 skipped; unitarios de `test_MoiraiAdapter.py` 36 passed; `test_device_handling.py -k moirai` 12 passed.
- Evidencia: `evidence/moirai/verification/rerun_tests.log`, `evidence/moirai/requirements-freeze.txt`, `evidence/moirai/env.log`.

**Antes de empezar.** Este plan puede quedar desfasado. Comprueba primero si hay versiones o checkpoints nuevos (desde la raíz del repositorio):

```bash
python dev/foundation_adapters_review/evidence/common/pypi_info.py uni2ts --since 2025-11-04
python dev/foundation_adapters_review/evidence/common/hub_info.py Salesforce --prefix moirai
python dev/foundation_adapters_review/evidence/common/hub_info.py --repo Salesforce/moirai-2.0-R-small
# Estado de los PR upstream que cambiarían la instalación
gh pr view 216 --repo SalesforceAIResearch/uni2ts --json state,mergedAt
gh pr view 256 --repo SalesforceAIResearch/uni2ts --json state,mergedAt
```

Recrea el entorno de la revisión (`evidence/README.md`). El script reescribe `evidence/moirai/requirements-freeze.txt`: si no quieres cambiar la evidencia, restaura ese fichero después con `git checkout`.

```bash
UV_OVERRIDE=dev/foundation_adapters_review/evidence/moirai/overrides.txt \
    bash dev/foundation_adapters_review/evidence/common/make_env.sh moirai "uni2ts" 3.12
```

**Si apareció una versión nueva de uni2ts (posterior a 2.0.0):**

1. Recrea el entorno con ella y vuelve a pasar el contrato, el smoke y los unitarios. Ejecuta también `compare_direct.py`, `covariates_probe.py` (D16), `set_params_reload.py` y `misc_checks.py` (sección 4).
2. Mira su `requires_dist`.
   - Si ya no fija `scipy~=1.11.3`: el override de `adapters.toml` (MOI-F2) y la nota del conflicto con pip (MOI-F4) pueden sobrar.
   - Si ya no fija `torch<2.5`: la nota "Python <= 3.12" de MOI-P1 y D15 cambia. Comprueba que en Python 3.13 se resuelve la versión nueva y no 1.1.1.
   - Si ya contiene `moirai2` y cambia las firmas de `Moirai2Module.from_pretrained`, `Moirai2Forecast.__init__`, `hparams_context` o `predict`, el contrato lo detectará: actualiza `contracts.py` y el adaptador.
3. Ajusta el mínimo de `backend_package` solo si la versión nueva trae algo que el adaptador necesite.

**Si aparecieron checkpoints nuevos de Moirai-2 (por ejemplo `-base` o `-large`):**

1. Lee su `config.json` y compara `quantile_levels` con `SUPPORTED_QUANTILES`. MOI-P3 los rechazaría si difieren.
2. Compara `max_seq_len` y `patch_size` con 512 y 16.
3. Actualiza el docstring y la tabla de la guía (MOI-P4). No hay nada que retirar de la tabla central de ids no soportados: `02_common.md` (sección "Descartado") no incluye `-base` ni `-large` en ella.

## 2. Hallazgos

Fuente: sección 3.6 de `evidence/diagnosis_report.md` y `evidence/moirai/verification/verification.md`. Se usan los textos corregidos por la verificación.

- **MOI-F1** [probado] 2.0.0 sigue siendo la última versión de uni2ts.
  - El wheel es idéntico a `src/uni2ts` del tag, y `main` (cfd46d4) solo toca ficheros fuera de `src/` y de `pyproject.toml` (37 ficheros: `project/moirai-agent`, CODEOWNERS, SECURITY.md...).
  - Las firmas de `Moirai2Module.from_pretrained`, `Moirai2Forecast.__init__`, `hparams_context` y `predict` no cambian: no hace falta tocar `contracts.py` por la API.
  - Resultados: contrato 5, smoke 5 + 1 skipped, unitarios 36, device_handling 12.
  - Evidencia: `evidence/moirai/upstream_diff.log`, `evidence/moirai/pypi_uni2ts.log`, `evidence/moirai/contract.log`, `evidence/moirai/smoke.log`, `evidence/moirai/verification/rerun_tests.log`.
- **MOI-F2** [probado] El override `scipy>=1.12` de `tools/foundation_compat/adapters.toml` sigue siendo necesario.
  - uni2ts declara `scipy~=1.11.3` en todas sus versiones, y skforecast exige `scipy>=1.12`: sin override, `uv pip compile` no encuentra solución. Con el override se resuelve scipy 1.17.1.
  - Los PR upstream #216 (relajar dependencias) y #256 (Python 3.13) siguen abiertos.
  - Evidencia: `evidence/moirai/resolve_no_override.log`, `evidence/moirai/verification/resolve_check.log`, `evidence/moirai/upstream_issues.log`, `evidence/moirai/upstream_issue_bodies.log`.
- **MOI-F3** [probado] En Python >= 3.13 el resolvedor instala uni2ts 1.1.1 (con torch 2.15.0+cpu), porque 1.2.0 y 2.0.0 exigen torch<2.5.
  - uni2ts 1.1.1 no tiene `uni2ts.model.moirai2`.
  - El adaptador dice "uni2ts is required for MoiraiAdapter. Install it with `pip install uni2ts`." aunque uni2ts está instalado; la causa real es "No module named 'uni2ts.model.moirai2'".
  - `uni2ts>=2.0` en Python 3.13 no tiene solución. `backend_package="uni2ts"` no expresa el mínimo 2.0.
  - Issues upstream #251 y #133 abiertas.
  - Evidencia: `evidence/moirai/resolve_py313.log`, `evidence/moirai/py313_install.log`, `evidence/moirai/upstream_diff.log`, `evidence/moirai/misc_checks.log` (sección 1), `evidence/moirai/verification/resolve_check.log`, `evidence/moirai/verification/rerun_misc_checks.log`.
- **MOI-F4** [probado] Con pip y Python 3.12, `pip install uni2ts` sobre skforecast instala uni2ts 2.0.0, pero baja scipy a 1.11.4 y pandas a 2.1.4 (gluonts 0.14.4 fija `pandas<2.2.0`).
  - Así incumple los requisitos de skforecast; `pip check` lo reporta. Una predicción básica con Moirai funciona igualmente.
  - Con uv, skforecast en la misma resolución y el override, se eligen gluonts 0.14.3 (`pandas<3`) y pandas 2.3.3.
  - Las docs (SKILL.md línea 58, celda 11 de la guía) dicen solo `pip install uni2ts` y no mencionan el conflicto ni la limitación a Python <= 3.12.
  - Evidencia: `evidence/moirai/pip_user_path.log`, `evidence/moirai/verification/pip_dry_run.log`, `evidence/moirai/requirements-freeze.txt`.
- **MOI-F5 (corregido)** [probado] De Moirai-2, Salesforce solo publica `Salesforce/moirai-2.0-R-small`: pesos sin cambios desde 2025-08-06, y después solo commits de README (2025-08-11, 2026-01-16, 2026-01-29).
  - `-base` y `-large` dan 404, pero el docstring de `MoiraiAdapter` dice `Salesforce/moirai-2.0-R-{small,base,large}` (aprox. línea 1714 en d9b8f4f).
  - `_resolve_adapter` acepta cualquier id que empiece por el prefijo `Salesforce/moirai-2` (entrada `"Salesforce/moirai-2"` de `_ADAPTER_REGISTRY`), no solo `moirai-2.0-R-*`.
  - Rechaza Moirai 1.x, Moirai-MoE, `moirai-agent`, el id en minúsculas y los fine-tunes de terceros.
  - Evidencia: `evidence/moirai/hub_info.log`, `evidence/moirai/hub_moirai2_small.log`, `evidence/moirai/hub_moirai2_sizes.log`, `evidence/moirai/compare_direct.log` (sección 1).
- **MOI-F6 (corregido)** [probado] La licencia de `moirai-2.0-R-small` (cc-by-nc-4.0, no gated, sin fichero LICENSE) coincide con `_MODEL_LICENSES["Salesforce/moirai"]` de `skforecast/foundation/_utils.py`.
  - También son correctos `requires_hf_auth=False`, `requires_provider_auth=False`, `weights_repo_id=None` y `weights_in_hf_cache=True`; el `LicenseWarning` enlaza a la model card.
  - No todos los `Salesforce/moirai*` son cc-by-nc-4.0: `moirai-1.5-llama-kvcache` no declara licencia y es gated manual. No afecta, porque ese id no se enruta a `MoiraiAdapter`.
  - La sección Moirai de la guía (celda 65 en la revisión, "### Moirai") no menciona la licencia; la skill (`SKILL.md` línea 196) y `llms-base.txt` (línea 529) sí.
  - Evidencia: `evidence/moirai/hub_info.log`, `evidence/moirai/hub_moirai2_small.log`, `evidence/moirai/misc_checks.log` (sección 3).
- **MOI-F7** [probado] La salida del adaptador es idéntica (max|diff| = 0.000e+00) a `Moirai2Forecast.predict` directo, con el mismo módulo y el mismo `context_length`.
  - Casos: 2 series iguales; 3 series de longitudes 700/300/40 en una llamada; horizonte 150 (más que `num_predict_token` x `patch_size` = 64); contexto 2048; contexto 12000; serie con NaN interior.
  - Forma `(steps, n_quantiles)` por serie; la columna `pred` es exactamente el cuantil 0.5.
  - Evidencia: `evidence/moirai/compare_direct.log` (secciones 3, 4 y 6), `evidence/moirai/verification/rerun_compare_direct.log`, `evidence/moirai/verification/extra_checks.log`.
- **MOI-F8** [verificado en código] Los atributos de capacidad son veraces.
  - `SUPPORTED_QUANTILES` coincide con `module.quantile_levels` del checkpoint actual [probado].
  - `requires_equal_length_context=False`: cada serie se rellena por separado con `past_is_pad`. Una serie en lote coincide con la serie sola hasta 1.5e-5 [probado].
  - `supports_nan_in_series=True`: la máscara de observados se calcula antes de imputar; NaN interiores y al inicio dan valores finitos [probado].
  - No hay tope duro de contexto: `RotaryProjection._init_freq` se amplía en `forward`. Contextos de 8192 y 12000 y series de 5 y 17 puntos funcionan [probado].
  - El fallback de MPS a CPU sigue justificado: `PackedStdScaler` hace `target.double()` (`packed_scaler.py` línea 46 del tag 2.0.0).
  - Evidencia: `evidence/moirai/compare_direct.log` (secciones 2 a 5), `evidence/moirai/hub_moirai2_small.log`.
- **MOI-F9** [probado] `Moirai2Forecast.predict` con `feat_dynamic_real` (covariables futuras, de longitud T + steps o T) falla con `RuntimeError: The size of tensor a (34) must match the size of tensor b (36)` en `uni2ts/model/moirai2/module.py:144`.
  - `past_feat_dynamic_real` se acepta y cambia la salida 2.137e-02, sin evidencia de que el modelo se entrenara con ellas.
  - `predict()` recorta todos los campos de lista a `context_length` y dice "only support univariate forecast now".
  - `allow_exog=False` y `supports_past_only_covariates=False` siguen siendo correctos. La nota del docstring acierta en la causa; el efecto es un `RuntimeError`.
  - Issue upstream #224 abierta (caída de precisión con `feat_dynamic_real`).
  - Evidencia: `evidence/moirai/covariates_probe.log`, `evidence/moirai/verification/rerun_covariates_probe.log`, `evidence/moirai/upstream_issue_bodies.log`.
- **MOI-F10** [probado] `set_params(context_length=...)` o `set_params(device=...)` pone `_module=None`, y el siguiente `predict` recarga los pesos con `from_pretrained`: 2 llamadas en total.
  - `context_length` es un hiperparámetro del envoltorio `Moirai2Forecast` (`hparams_context` lo acepta), no de los pesos: reutilizar el módulo da diferencia 0.000e+00.
  - Además se descarta un `module` inyectado por el usuario, que pasa a descargarse desde `model_id` (1 llamada a `from_pretrained` donde antes había 0).
  - Código: `MoiraiAdapter.set_params`, aprox. líneas 1884-1917 en d9b8f4f (el reset está en `_reset_module`, aprox. línea 1907).
  - Evidencia: `evidence/moirai/set_params_reload.log`, `evidence/moirai/verification/rerun_set_params_reload.log`.
- **MOI-F11** [verificado en código] `MoiraiAdapter.predict` indexa la salida del backend por posiciones de `SUPPORTED_QUANTILES`.
  - El backend la ordena por `module.quantile_levels`, que viene del `config.json` de cada checkpoint (`Moirai2Module.__init__` acepta `quantile_levels`, por defecto 0.1 a 0.9).
  - Hoy coinciden [probado]. Un checkpoint con otra rejilla daría cuantiles mal etiquetados sin error, y ni el contrato ni el smoke lo comprueban.
  - Evidencia: `evidence/moirai/compare_direct.log` (sección 2).
- **MOI-F12** [probado] Una serie cuya ventana de contexto (128 observaciones en la prueba) es toda NaN da cuantiles en torno a 0 (del orden de 1e-4, por ejemplo q_0.5 = 0.000012) sin aviso: la imputación de uni2ts (`CausalMeanImputation`) rellena con 0 una secuencia entera NaN.
  - Es un caso genérico de `FoundationModel` (X-1, XC-1 en `evidence/cross_cutting.md`).
  - Evidencia: `evidence/moirai/all_nan_context.log`, `evidence/moirai/verification/rerun_all_nan_context.log`.
- **MOI-F13** [probado] El aviso de fallback MPS (`stacklevel=7` fijo en `_ensure_forecast_obj`, aprox. línea 2088 en d9b8f4f) se ve así:
  - Por `ForecasterFoundation.predict`, apunta a la línea del usuario: línea 71 del script.
  - Por `FoundationModel.predict`, apunta un marco por encima: línea 90 del script (la llamada a `main()`), en lugar de la 67.
  - El fallback a CPU funciona con `"mps"`, `"mps:0"` y `"auto"` (MPS simulado, sin hardware).
  - Evidencia: `evidence/moirai/misc_checks.log` (sección 2), `evidence/moirai/verification/rerun_misc_checks.log`.
- **MOI-F14** [verificado en código] Hay textos desfasados o sin respaldo:
  - "Max context length 2048" en la tabla de la guía (celda 11 en la revisión): 2048 es solo el valor por defecto.
  - "Multivariate pretraining" en `skills/foundation-forecasting/SKILL.md` línea 168: la model card no lo respalda y `predict` dice "only support univariate forecast now".
  - `R-{small,base,large}` en el docstring.
  - Instalación con `pip install uni2ts` (SKILL.md línea 58, celda 11) sin aviso de Python <= 3.12 ni del conflicto con pip.
  - El resto (cuantiles fijos, sin exog, horizonte libre, punto = mediana, MPS a CPU, licencia en la skill y en `llms-base.txt`) es correcto.
  - Matiz de la verificación: el checkpoint declara 512 parches x 16 = 8192 observaciones. 12000 funciona sin error y con diferencia 0 frente a la llamada directa [probado]. Si el modelo se entrenó con ese tope no es verificable: el paper está bloqueado en la red de la revisión.
  - Evidencia: `evidence/moirai/hub_moirai2_small.log`, `evidence/moirai/compare_direct.log` (sección 5), `evidence/moirai/verification/extra_checks.log`.
- **MOI-F15** [verificado en código] `FakeMoirai2Forecast` (`fixtures_adapters.py`, aprox. líneas 251-285 en d9b8f4f) reproduce la forma real `(n, 9, steps)` y la interfaz usada (`predict(past_target)`, `hparams_context(prediction_length)`).
  - El real devuelve float32 y el fake float64, sin efecto: `FoundationModel` devuelve float64 [probado].
  - Faltan tests de `module.quantile_levels == SUPPORTED_QUANTILES`, del mensaje con uni2ts < 2.0 y de la conservación del módulo inyectado.
  - Al contrario, `test_MoiraiAdapter_set_params_updates_and_resets_module` afirma que `context_length` y `device` resetean `_module`.
  - Evidencia: `evidence/moirai/unit_tests.log`, `evidence/moirai/misc_checks.log` (sección 3).
- **MOI-V1** [probado] El `LicenseWarning` de Moirai apunta a `skforecast/foundation/_adapters.py:2069`, la línea `self._load_module()` de `_ensure_forecast_obj`, y no a la línea del usuario.
  - Causa: `_warn_if_non_commercial` (`skforecast/foundation/_utils.py`, aprox. línea 439 en d9b8f4f) usa `stacklevel=3` fijo, que solo sube hasta el llamador de `_load_module`. Es la misma familia que MOI-F13 y X-6 (XC-8 en `evidence/cross_cutting.md`).
  - Se ve también en el log del CI (job 127875169007) y en la salida guardada de la guía (celda 68 en la revisión: "Location: .../skforecast/foundation/_adapters.py: 2074", ruta de site-packages de la máquina que la ejecutó) [verificado en código].
  - Es cosmético. El helper se llama desde 6 puntos (`TimesFM25Adapter`, `TimesFM3Adapter`, `MoiraiAdapter`, `TabPFNAdapter`, `TSICLAdapter` y `TimestralAdapter`); avisan los cinco con licencia no comercial (todos menos TimesFM 2.5) [verificado en código].
  - Evidencia: `evidence/moirai/verification/extra_checks.log`, `evidence/moirai/misc_checks.log` (sección 3), `evidence/moirai/ci_baseline.log`.
- **MOI-V2** [verificado en código] El dato correcto para "Max context length" es este: el checkpoint declara `max_seq_len=512` parches x `patch_size=16` = 8192 observaciones, y uni2ts no tiene tope duro (`RotaryProjection` se amplía).
  - `context_length=12000` da una salida finita e idéntica a la llamada directa [probado].
  - Redacción propuesta: "sin tope duro en uni2ts (el checkpoint declara 8192)", en lugar de 2048 o de "sin límite".
  - Evidencia: `evidence/moirai/hub_moirai2_small.log`, `evidence/moirai/verification/extra_checks.log`.

**Descartado**

- "Todos los `Salesforce/moirai*` públicos son cc-by-nc-4.0" (parte original de MOI-F6): `moirai-1.5-llama-kvcache` no declara licencia [probado].

**Preguntas abiertas (no verificables en la revisión)**

- Contexto real de entrenamiento de Moirai-2, y si se entrenó con covariables o en modo multivariante: el paper (arXiv 2511.11698) y el blog están bloqueados.
- Comportamiento real en CUDA y MPS: no hay GPU (MPS solo se simuló).
- Comentarios de las issues upstream: solo se leyeron título, estado y el principio del cuerpo.

## 3. Cambios a implementar

### MOI-P1. `uni2ts>=2.0`, mensaje de import que distinga los casos y documentación de instalación

**Qué:**

1. **Mínimo.** `MoiraiAdapter.backend_package = "uni2ts>=2.0"` (aprox. línea 1820 en d9b8f4f), y lo mismo en `[moirai] backend_package` de `tools/foundation_compat/adapters.toml`. Se mantiene `overrides = ["scipy>=1.12"]` con su comentario (MOI-F2).
   - Precedente: `TabPFNAdapter.backend_package = "tabpfn-time-series>=1.3"`. `run.py` y `test_backends_compat_config.py` ya admiten especificadores PEP 508 [verificado en código].
2. **Mensaje.** En `MoiraiAdapter._load_module` (aprox. líneas 2037-2043 en d9b8f4f), cuando falla `from uni2ts.model.moirai2 import Moirai2Module`, distinguir tres casos:
   - uni2ts no está instalado: mensaje de instalación con el especificador entre comillas.
   - uni2ts está instalado con versión < 2.0: dar la versión instalada, decir que Moirai-2 exige `uni2ts>=2.0` y que uni2ts 2.0 exige torch<2.5, que solo existe para Python <= 3.12.
   - uni2ts >= 2.0 está instalado pero el import falla por otra causa: reemitir con la causa original, como hace `TimesFM3Adapter._load_model`.
   - Con D-VER (b), `_load_module` llama a `_check_backend_min_version(type(self))` de 02 (sección 3.4) al principio, antes del import; el caso uni2ts < 2.0 lo lanza el helper con su mensaje (`MoiraiAdapter requires uni2ts>=2.0, but 1.1.1 is installed. Upgrade with ...`), y el `except ImportError` solo distingue dos casos con la versión que devolvió el helper: `None` (no instalado, mensaje con la nota de Python <= 3.12) y >= 2.0 instalado pero import fallido (reemitir la causa). La nota de Python <= 3.12 no se añade al mensaje genérico de 02: queda en el mensaje de "no instalado" y en las docs (sección 5). Con D-VER (a) o (c), leer la versión con `importlib.metadata.version("uni2ts")` y `PackageNotFoundError` y mantener las tres ramas (patrón de `TimesFM3Adapter._load_model`, aprox. líneas 1658-1696 en d9b8f4f).
3. **Comillas.** Entrecomillar `backend_package` en el mensaje: `pip install "uni2ts>=2.0"`. Sin comillas, el shell redirige la salida a un fichero `=2.0`. `TabPFNAdapter` y `TimesFM3Adapter` ya entrecomillan [verificado en código; corrección de la verificación].
4. **Docs.** Actualizar los comandos de instalación y documentar el conflicto scipy/pandas con pip (sección 5).

Ilustración del mensaje (no es la implementación completa):

```python
except ImportError as exc:
    installed = _installed_version("uni2ts")  # version returned by the helper of 02 (D-VER (b)) or read with importlib.metadata ((a)/(c))
    if installed is None:
        message = (
            "uni2ts is required for MoiraiAdapter. Install it with "
            f'`pip install "{self.backend_package}"` (uni2ts 2.0 needs '
            "Python 3.12 or earlier)."
        )
    elif int(installed.split(".")[0]) < 2:  # only with D-VER (a)/(c): with (b) the helper of 02 raised before the import
        message = (
            f"MoiraiAdapter requires uni2ts>=2.0 (Moirai-2), but uni2ts "
            f"{installed} is installed and does not provide "
            "`uni2ts.model.moirai2`. uni2ts 2.0 requires torch<2.5, available "
            "only for Python 3.12 or earlier. Upgrade with "
            f'`pip install -U "{self.backend_package}"`.'
        )
    else:
        message = (
            f"uni2ts {installed} is installed but `uni2ts.model.moirai2` "
            f"could not be imported ({exc})."
        )
    raise ImportError(message) from exc
```

`_installed_version` es un nombre ilustrativo: el helper de 02, `_check_backend_min_version(type(self))`, recibe la clase del adaptador (no el nombre de la distribución) y se llama antes del import, no dentro del `except`; con D-VER (a) o (c), `importlib.metadata`. La comparación por versión mayor (como `TimesFM3Adapter._load_model`, con `ValueError` tratado como 0) basta aquí.

**Por qué:** MOI-F2, MOI-F3, MOI-F4.

**Riesgo:** bajo.
- Con uni2ts < 2.0 el adaptador ya no funcionaba (no existe `moirai2`).
- `uv pip install "uni2ts>=2.0"` en Python 3.13 falla en la resolución en lugar de instalar 1.1.1 [probado].
- Un mínimo en `backend_package` solo cubre instalaciones nuevas y el CI; las existentes las cubre el mensaje (X-4) [verificado en código].

**Rompe compatibilidad:** no. Los usuarios de Python 3.13+ ya no podían usar el adaptador; ahora ven un error claro.

**Decisión:** D15 (recomendación (c), con comillas). Con D-VER (b), el helper de 02 lanza el caso < 2.0; MOI-P1 solo distingue no instalado e import fallido.

### MOI-P2. No recargar los pesos al cambiar `context_length` o `device`, y conservar el módulo inyectado

**Qué:** en `MoiraiAdapter.set_params` (aprox. líneas 1884-1917 en d9b8f4f), separar los resets que se pasan a `_apply_set_params`:
- `model_id` resetea `_module` y `_forecast_obj`, como hoy (`_reset_module`).
- `context_length` y `device` resetean solo `_forecast_obj`.

`_apply_set_params` ya admite varias tuplas en `resets` y llama a cada una cuyo conjunto de claves cambió [verificado en código]. `_ensure_forecast_obj` reconstruye entonces el envoltorio `Moirai2Forecast` con el `_module` existente, el nuevo `context_length` y el nuevo dispositivo.

```python
def _reset_forecast_obj() -> None:
    self._forecast_obj = None

return _apply_set_params(
    self, params,
    validate=validate,
    resets=(
        ({"model_id"}, _reset_module),
        ({"context_length", "device"}, _reset_forecast_obj),
    ),
)
```

Detalles de la implementación:

- Al cambiar `device`, el `.to(resolved_device)` del envoltorio mueve el módulo en sitio, porque `module` es un submódulo de `Moirai2Forecast` (un `nn.Module`). Es lo mismo que pasa en la primera carga [verificado en código].
- Alternativa equivalente: pasar `context_length` en cada `hparams_context(prediction_length=steps, context_length=...)` de `_run_inference`, que lo acepta [verificado en código], y no resetear nada con `context_length`. Se recomienda la primera, porque no cambia la llamada de `_run_inference` ni el contrato de `hparams_context` [inferido].
- Actualizar el docstring de `set_params` (hoy: "Resets the module and forecast object when `model_id`, `context_length`, or `device` changes").

**Por qué:** MOI-F10.

**Riesgo:** bajo. Las predicciones no cambian: diferencia 0.000e+00 entre el módulo recargado y el reutilizado [probado]. Hoy hay 2 llamadas a `from_pretrained`; medido con uni2ts 2.0.0, torch 2.4.1+cpu, Python 3.12.3, CPU x86_64 de 4 núcleos (`evidence/moirai/set_params_reload.log`).

**Rompe compatibilidad:** no en la API. Hay que actualizar `test_MoiraiAdapter_set_params_updates_and_resets_module` y `test_MoiraiAdapter_set_params_device_resets_forecast_obj`, que afirman el reset de `_module` (sección 4).

**Decisión:** ninguna.

### MOI-P3. Guardarraíl de la rejilla de cuantiles del checkpoint

**Qué:** en `MoiraiAdapter._ensure_forecast_obj`, después de `self._load_module()` y antes de construir `Moirai2Forecast`, comprobar que `self._module.quantile_levels` coincide con `SUPPORTED_QUANTILES` (misma longitud, tolerancia 1e-9). Si no coincide, `ValueError` que nombre el `model_id` y las dos rejillas.

- Va en `_ensure_forecast_obj` y no en `_load_module` para cubrir también un `module` inyectado: `_load_module` no hace nada si `_module` ya está puesto [verificado en código].
- La verificación prefiere el error a recalcular índices desde `module.quantile_levels`, porque `SUPPORTED_QUANTILES` ya se usa para validar los niveles antes de cargar [verificado en código].

```python
checkpoint_levels = [float(q) for q in self._module.quantile_levels]
if len(checkpoint_levels) != len(self.SUPPORTED_QUANTILES) or any(
    abs(a - b) > 1e-9 for a, b in zip(checkpoint_levels, self.SUPPORTED_QUANTILES)
):
    raise ValueError(
        f"The checkpoint '{self.model_id}' returns the quantile levels "
        f"{checkpoint_levels}, but MoiraiAdapter supports only "
        f"{self.SUPPORTED_QUANTILES}."
    )
```

La "comprobación en el smoke" de la propuesta queda cubierta sin un test nuevo: el guardarraíl se ejecuta en cada `predict` del smoke con el checkpoint real, que fallaría si la rejilla cambiara [inferido].

**Relación con 02:** `02_common.md` convierte los niveles a float en `FoundationModel` y no cambia la llamada a `_validate_supported_quantiles` de `MoiraiAdapter.predict`. Si 02 acabara centralizando también la validación contra `SUPPORTED_QUANTILES` (XC-5 en `evidence/cross_cutting.md`), esa llamada se quitaría allí, no aquí.

**Por qué:** MOI-F11, MOI-F15.

**Riesgo:** nulo con el checkpoint actual (rejillas iguales [probado]). Los tests que construyen el envoltorio con un `MagicMock` como módulo necesitan `quantile_levels` explícito: iterar un `MagicMock` da una lista vacía y el guardarraíl lanzaría el error (sección 4) [inferido].

**Rompe compatibilidad:** no.

**Decisión:** ninguna.

### MOI-P4. Correcciones de documentación de Moirai (incluye MOI-V2)

**Qué:** textos de Moirai en el docstring, la guía, la skill y `llms-base.txt`, con los lugares exactos en la sección 5:
- Solo se publica R-small, y el registro acepta el prefijo `Salesforce/moirai-2`.
- Contexto máximo: "sin tope duro en uni2ts; el checkpoint declara 8192".
- Licencia CC-BY-NC-4.0 y `LicenseWarning` en la sección Moirai de la guía.
- Quitar "Multivariate pretraining".
- En el docstring, que las covariables futuras hacen fallar `predict()` de uni2ts con un `RuntimeError`.

**Por qué:** MOI-F5, MOI-F6, MOI-F9, MOI-F14, MOI-V2.

**Riesgo:** nulo. **Rompe compatibilidad:** no. **Decisión:** ninguna. La parte de instalación depende de D15.

### MOI-P5. Contexto todo NaN (implementado en `02_common.md`)

**Qué:** nada en este PR. La comprobación central en `FoundationModel.predict`, después del recorte a `context_length`, se implementa en `02_common.md` junto con CHR-P1 y TV-2. El adaptador no debe tener una comprobación propia (`.claude/rules/foundation.md`).

**Aquí solo se comprueba** que, con 02 ya fusionado, `evidence/moirai/all_nan_context.py` da el error o el aviso que se haya decidido en lugar de cuantiles ~0 (sección 7).

**Por qué:** MOI-F12, X-1. **Decisión:** D11 (en 02).

### MOI-P6 y MOI-V1. `stacklevel` del aviso MPS y del `LicenseWarning`

**Dónde vive el arreglo:** el cambio de código compartido (traslado de `_first_frame_outside_skforecast` y `_stacklevel_user_code` a `skforecast/utils/utils.py`, `_warn_if_non_commercial` y el `stacklevel=None` por defecto de `_resolve_torch_device`) es la sección 3.5 de `02_common.md`, porque afecta a todos los adaptadores; con él queda resuelto MOI-V1. Este PR solo hace la parte propia de Moirai.

**Qué:**

1. **`MoiraiAdapter._ensure_forecast_obj`**: quitar `stacklevel = 7` de la llamada a `_resolve_torch_device` (aprox. línea 2088 en d9b8f4f), para que use el cálculo dinámico de 02.
2. Comprobar con un test que el `LicenseWarning` de Moirai apunta al llamador (lo arregla 02, pero el test de Moirai lo fija).

Notas:
- Si 02 todavía no está fusionado, este punto se deja para después: sin el `stacklevel=None` por defecto de 02, quitar `stacklevel = 7` empeoraría el aviso.
- `TimestralAdapter` pasa `stacklevel=6` a `_resolve_torch_device` y puede quedarse así: Timestral no está en la revisión. Su `LicenseWarning` mejora igual con 02.

**Por qué:** MOI-F13, MOI-V1, X-6.

**Riesgo:** nulo para el resultado. Cambia solo la línea que muestra el aviso MPS.

**Rompe compatibilidad:** no.

**Decisión:** ninguna. Depende de la sección 3.5 de `02_common.md`.

### MOI-P7. Mantener `allow_exog=False` y `supports_past_only_covariates=False`

**Qué:** ningún cambio de código. No activar covariables hasta que uni2ts publique soporte documentado.
- Precisar el docstring (MOI-P4).
- Reejecutar `evidence/moirai/covariates_probe.py` con cada release nueva de uni2ts (sección 4 y "Antes de empezar").

**Por qué:** MOI-F9.

**Riesgo:** ninguno manteniendo el estado actual. **Rompe compatibilidad:** no.

**Decisión:** D16 (recomendación (a)). Las opciones (b) y (c) serían otro PR.

## 4. Tests

**Tests unitarios en `skforecast/foundation/tests/tests_foundation_models/test_MoiraiAdapter.py`:**

- Cambiar `test_MoiraiAdapter_set_params_updates_and_resets_module` (MOI-P2). Hoy está parametrizado con `model_id`, `context_length` y `device`, y afirma `adapter._module is None`. Separarlo en dos:
  - `test_MoiraiAdapter_set_params_model_id_resets_module_and_forecast_obj`: solo `model_id`, comportamiento actual.
  - `test_MoiraiAdapter_set_params_context_length_or_device_keeps_module`: `_module` se conserva (`is` el mismo objeto) y `_forecast_obj is None`.
  - `test_MoiraiAdapter_set_params_no_reset_when_value_unchanged` sigue igual.
- Nuevo `test_MoiraiAdapter_set_params_keeps_injected_module_without_reloading` (MOI-P2).
  - Con `module=FakeMoirai2Module(...)` y `uni2ts.model.moirai2` simulado en `sys.modules`, `set_params(context_length=...)` seguido de `predict` no llama a `from_pretrained` (0 llamadas).
  - El nuevo envoltorio recibe el nuevo `context_length`.
- Nuevo `test_MoiraiAdapter_load_module_ImportError_when_uni2ts_without_moirai2` (MOI-P1).
  - `monkeypatch.setitem(sys.modules, "uni2ts.model.moirai2", None)` y la versión instalada simulada en `"1.1.1"` (parcheando `importlib.metadata.version` donde la lea el helper de 02 o el adaptador).
  - Con D-VER (b): el `ImportError` lo lanza `_check_backend_min_version` antes del import (comprobar que no se llega a `uni2ts.model.moirai2`, o basta con el mensaje), y el mensaje es el del helper: contiene `1.1.1`, `uni2ts>=2.0` y el comando entrecomillado `pip install -U "uni2ts>=2.0"`, sin afirmar "Python 3.12".
  - Con D-VER (a) o (c): el mensaje contiene `1.1.1`, `uni2ts>=2.0`, "Python 3.12" y el comando entrecomillado `pip install -U "uni2ts>=2.0"`.
- Nuevo test del caso "uni2ts >= 2.0 instalado pero el import falla": versión simulada `"2.0.0"`; el mensaje nombra `uni2ts 2.0.0` y la causa original, y `__cause__` es el `ImportError` original.
- Caso "uni2ts no instalado": mensaje con `pip install "uni2ts>=2.0"`. Se puede ampliar `test_MoiraiAdapter_load_module_ImportError_no_LicenseWarning`, que ya simula el import y sigue valiendo [verificado en código; matiz de la verificación]. Simular también la versión como no instalada, para que el test no dependa de si uni2ts está en el entorno.
- Nuevo `test_MoiraiAdapter_ensure_forecast_obj_ValueError_when_checkpoint_quantile_levels_differ` (MOI-P3): un módulo falso con `quantile_levels` de 0.05 a 0.95 da `ValueError` con el mensaje esperado.
- `test_MoiraiAdapter_load_module_LicenseWarning_on_successful_load` (MOI-V1): con 02 fusionado, añadir `record[0].filename == __file__`. Si 02 ya lo añadió (su sección 4 lo propone), no duplicarlo.
- Los tests existentes que usan `Salesforce/moirai-2.0-R-base` y `-R-large` (`test_MoiraiAdapter_get_params_returns_expected_keys_and_values` y el parametrizado de `set_params`) no pasan por `_resolve_adapter`, así que no les afecta la tabla de ids no soportados de 02 [verificado en código]. En los tests nuevos, usar `Salesforce/moirai-2.0-R-small` o un id ficticio.

**Tests en `test_device_handling.py`:**

- `test_MoiraiAdapter_set_params_device_resets_forecast_obj`: cambiar la aserción `adapter._module is None` por la conservación del módulo y actualizar su docstring (MOI-P2).
- `test_MoiraiAdapter_ensure_forecast_obj_calls_to_with_resolved_device`:
  - Quitar `assert kwargs["stacklevel"] == 7` (MOI-P6). Se puede comprobar que no se pasa `stacklevel`, o que vale `None`.
  - Dar a `mock_module` el atributo `quantile_levels = [0.1, ..., 0.9]` (MOI-P3).
- `test_MoiraiAdapter_ensure_forecast_obj_mps_falls_back_to_cpu`: dar `quantile_levels` a `mock_module` y añadir que el aviso apunta al fichero del test (`record[0].filename == __file__`).
- Los tests de `_resolve_torch_device`, de `_warn_if_non_commercial` y del helper trasladado son de `02_common.md` (sección 4).

**Tests en `test_get_model_info.py`:**

- `test_get_model_info_output_for_each_adapter`, aprox. línea 167 en d9b8f4f (igual en `origin/0.27.x`): la tupla esperada de `Salesforce/moirai-2.0-R-small` pasa de `"uni2ts"` a `"uni2ts>=2.0"` (MOI-P1; solo con D15 (b) o (c), con (a) no cambia) [verificado en código].

**Fakes en `fixtures_adapters.py`:**

- Nuevo `FakeMoirai2Module` con `quantile_levels` (por defecto 0.1 a 0.9), un `from_pretrained` de clase que cuente las llamadas, `eval()` y `to()`.
  - Sirve para MOI-P2 y MOI-P3, y para sustituir el `_FakeMoirai2Module` local de `test_MoiraiAdapter_load_module_LicenseWarning_on_successful_load`.
- `FakeMoirai2Forecast`: aceptar los kwargs del constructor real (`module`, `prediction_length`, `context_length`, `target_dim`, `feat_dynamic_real_dim`, `past_feat_dynamic_real_dim`), con valores por defecto para no romper `make_adapter` (que hoy lo construye sin argumentos), y `eval()`/`to()` que devuelvan `self`.
  - Así se puede usar como `Moirai2Forecast` simulado en `sys.modules` y comprobar el `context_length` que recibe.
  - La forma `(n, 9, steps)` sigue igual (MOI-F15).

**Compatibilidad (`tools/foundation_compat/`):**

- `adapters.toml`: `[moirai] backend_package = "uni2ts>=2.0"`. Se mantiene `overrides = ["scipy>=1.12"]` con su comentario. `test_backends_compat_config.py` comprueba que coincide con el atributo de la clase y que es PEP 508 [verificado en código].
- `tests/contracts.py`, entrada `"moirai"`: opcionalmente, `{"target": "uni2ts.model.moirai2:Moirai2Module", "kwargs": ["quantile_levels"]}`, para que el contrato se ponga en rojo si upstream deja de aceptar `quantile_levels` en el constructor. Solo comprueba la firma, no el atributo que lee MOI-P3; el atributo lo cubre el smoke [inferido]. Si se elige la alternativa de MOI-P2 con `hparams_context`, añadir `"context_length"` a los kwargs de `Moirai2Forecast.hparams_context`.
- `tests/test_smoke.py`: sin cambios (ver MOI-P3).

**Comprobaciones antes y después con los scripts de evidencia** (entorno `~/.venvs/fm-review-moirai`, desde la raíz del repositorio):

```bash
PY=~/.venvs/fm-review-moirai/bin/python
$PY dev/foundation_adapters_review/evidence/moirai/compare_direct.py
$PY dev/foundation_adapters_review/evidence/moirai/set_params_reload.py
$PY dev/foundation_adapters_review/evidence/moirai/misc_checks.py
$PY dev/foundation_adapters_review/evidence/moirai/verification/extra_checks.py
$PY dev/foundation_adapters_review/evidence/moirai/covariates_probe.py
$PY dev/foundation_adapters_review/evidence/moirai/all_nan_context.py   # tras fusionar 02
$PY -m pytest tools/foundation_compat/tests/test_contract.py --backend moirai -q -p no:cacheprovider
$PY -m pytest tools/foundation_compat/tests/test_smoke.py --backend moirai -q -p no:cacheprovider
$PY -m pytest skforecast/foundation/tests/tests_foundation_models/test_MoiraiAdapter.py -q -p no:cacheprovider
python tools/foundation_compat/run.py --adapter moirai
```

Los valores esperados están en la sección 7.

**Comprobación de la resolución en Python 3.13** (MOI-P1, no hace falta crear el entorno):

```bash
echo "uni2ts>=2.0" | uv pip compile --python-version 3.13 --torch-backend cpu -
# esperado: "No solution found" (uni2ts>=2.0.0 depends on torch>=2.1,<2.5)
```

**Backtest de la guía:** no hace falta (`dev/foundation_adapters_review/evidence/common/backtest_guide.py`). Ninguna propuesta cambia predicciones: MOI-P2 da diferencia 0.000e+00 [probado]. La comprobación de corrección es `set_params_reload.py`.

## 5. Documentación y contexto de IA

**Docstrings (`skforecast/foundation/_adapters.py`):**

- **`MoiraiAdapter`** (clase, aprox. líneas 1707-1807 en d9b8f4f):
  - `model_id`: sustituir "Must be a `Salesforce/moirai-2.0-R-{small,base,large}` variant." por algo como "Ids starting with `Salesforce/moirai-2` are routed to this adapter; Salesforce publishes only `Salesforce/moirai-2.0-R-small`." (MOI-F5).
  - `context_length`: añadir "uni2ts has no hard limit; the checkpoint declares 8192 observations (512 patches of 16)." (MOI-V2).
  - Notes de covariables: añadir que, con `feat_dynamic_real`, `Moirai2Forecast.predict()` de uni2ts 2.0.0 falla con un `RuntimeError` (MOI-F9).
  - `backend_package` en Attributes, o una línea en Notes: `uni2ts>=2.0`, que exige Python <= 3.12 (MOI-P1).
- **`MoiraiAdapter.__init__`**: la misma nota de `context_length`.
- **`MoiraiAdapter.set_params`**: qué parámetros recargan los pesos (solo `model_id`) y cuáles reconstruyen solo el envoltorio (`context_length`, `device`) (MOI-P2).
- **`_load_module`**: Notes con los tres mensajes de `ImportError` (MOI-P1).
- **`_ensure_forecast_obj`**: Notes con el `ValueError` de la rejilla (MOI-P3).
- Los docstrings de `_resolve_torch_device` y `_warn_if_non_commercial` los actualiza `02_common.md` (sección 5).
- Seguir `.github/instructions/docstrings.instructions.md`.

**Guía `docs/user_guides/foundation-forecasting-models.ipynb`:** solo celdas markdown de Moirai, sin reejecutar el notebook.

- **Celda 11 en la revisión**, tabla HTML "Supported Foundation Models", fila `<th scope="row">Moirai</th>`:
  - Columna "Max context length" (el segundo `<td>2048</td>` de la fila): cambiar `2048` por "No hard limit in uni2ts (the checkpoint declares 8192)" (MOI-F14, MOI-V2). "Default context length" sigue en 2048.
  - Columna "Install command": `<code>pip install "uni2ts&gt;=2.0"</code>`, con `data-copy='pip install "uni2ts&gt;=2.0"'` y `aria-label` como en la fila de TabPFN-TS. Añadir "(Python 3.12 or earlier)" si cabe en la celda (D15).
- **Celda 65 en la revisión** ("### Moirai", texto "A ForecasterFoundation is created using Salesforce's **Moirai-2.0-R-small** model."). Añadir:
  - Que es el único checkpoint de Moirai-2 publicado.
  - Que sus pesos son CC-BY-NC-4.0 y la primera carga emite un `LicenseWarning` (MOI-F6).
  - Con D15 (c), una nota de instalación: uni2ts 2.0 exige Python <= 3.12, y con pip baja scipy a 1.11.x y pandas a 2.1.x (incompatibles con skforecast). Recomendar uv con un override `scipy>=1.12`, como el CI de compatibilidad (MOI-F4).
- La celda 68 en la revisión (salida con el `LicenseWarning` que apunta a `_adapters.py`) **no se toca aquí**: la actualiza la reejecución de `10_docs_ci.md`.

**`skills/foundation-forecasting/SKILL.md`:**

- Línea 58: `pip install "uni2ts>=2.0"           # For Moirai-2 (Python 3.12 or earlier)` (MOI-P1).
- Línea 168 (tabla "Choosing a Model"): sustituir "Multivariate pretraining, probabilistic forecasts" por una descripción respaldada por la model card, por ejemplo "Univariate probabilistic forecasts (fixed deciles), no covariates" (MOI-F14).
- Línea 284 (error 7, "Tuning a parameter that forces a model reload"): quitar Moirai-2 de la lista de `context_length`, y decir que en Moirai-2 `device` tampoco recarga los pesos (MOI-P2).

**`skills/foundation-forecasting/references/adapter-parameters.md`:**

- Sección `## MoiraiAdapter` (aprox. líneas 86-97):
  - Línea 88 (prefijo): añadir que solo se publica `Salesforce/moirai-2.0-R-small`.
  - Fila `context_length`: "no hard limit in uni2ts; the checkpoint declares 8192".
  - Una línea de instalación (`uni2ts>=2.0`, Python <= 3.12).
- Línea 218 (fila `MoiraiAdapter` de la tabla de `set_params`): "Changing these forces a model reload" pasa de "**all of them**" a `model_id` (`context_length` y `device` solo reconstruyen el envoltorio) (MOI-P2).
- Líneas 230 y 231: quitar Moirai-2 de las listas de "reloads the model" y de los ejemplos de recarga, y añadirlo a la de "free" (MOI-P2).

**`tools/ai/llms-base.txt`:**

- Línea 458: en la lista de backends, `uni2ts` pasa a `"uni2ts>=2.0"` (Python 3.12 or earlier) (MOI-P1).
- La fila de la línea 491 (`Salesforce/moirai-2`, No, 2048, deciles) es correcta [verificado en código].
- Línea 529 (licencias): sin cambios.

**Regeneración:** aplicar la skill `ai-context-sync` (`python tools/ai/generate_ai_context_files.py` y después `--check`). No se editan a mano `AGENTS.md`, `llms-full.txt` ni `docs/llms*.txt`.

**Nota de release** (`docs/releases/releases.md`, sección 0.27.0; ajustar con la skill `release-note` y añadir el enlace del PR):

En **Changed** (la skill incluye ahí los mínimos de dependencias y la documentación):

```markdown
+ <code>[MoiraiAdapter]</code> requires `uni2ts>=2.0`, the first version with Moirai-2. When the installed uni2ts does not provide Moirai-2 (versions before 2.0, the ones installed on Python 3.13 or later), the `ImportError` gives the installed version and explains that uni2ts 2.0 needs Python 3.12 or earlier, instead of asking to install uni2ts again. ([#PR](https://github.com/skforecast/skforecast/pull/PR))

+ <code>[MoiraiAdapter]</code>: changing `context_length` or `device` with `set_params` no longer reloads the weights from the Hugging Face Hub and keeps a module passed in `module`; predictions are unchanged. ([#PR](https://github.com/skforecast/skforecast/pull/PR))

+ <code>[MoiraiAdapter]</code> raises a `ValueError` if a checkpoint returns quantile levels other than `[0.1, ..., 0.9]`, instead of labeling them wrongly. ([#PR](https://github.com/skforecast/skforecast/pull/PR))

+ Moirai-2 documentation: only `Salesforce/moirai-2.0-R-small` is published, uni2ts has no hard context limit (the checkpoint declares 8192), the weights are CC-BY-NC-4.0, and uni2ts 2.0 installs only on Python 3.12 or earlier and, with pip, downgrades scipy and pandas below the versions skforecast requires. ([#PR](https://github.com/skforecast/skforecast/pull/PR))
```

En **Fixed**:

```markdown
+ The MPS fallback warning of <code>[MoiraiAdapter]</code> pointed to an internal line of skforecast when the model was used through `FoundationModel.predict`. It now points to the line of user code that triggered the prediction. ([#PR](https://github.com/skforecast/skforecast/pull/PR))
```

La entrada del `LicenseWarning` la lleva `02_common.md` (sección 5); no repetirla aquí.

No hace falta una entrada en "Before upgrading": con uni2ts < 2.0 el adaptador ya fallaba al importar `uni2ts.model.moirai2` [probado], y ninguna predicción cambia.

## 6. Decisiones pendientes

| Id | Decisión | Opciones | Recomendación | Impacto en el plan | Estado |
|:--|:--|:--|:--|:--|:--|
| D15 | Mínimo y mensaje de Moirai (MOI-P1) | (a) solo mejorar el mensaje; (b) mensaje y `uni2ts>=2.0` en `backend_package`, `adapters.toml` y docs; (c) (b) y además documentar Python <= 3.12 y el conflicto scipy 1.11 / pandas < 2.2 de la instalación con pip | (c), con el especificador entre comillas | (a): no cambian `backend_package`, `adapters.toml` ni `test_get_model_info.py`, solo el mensaje. (b): sin las notas de Python y pip en la celda 65, SKILL.md y `llms-base.txt`. (c): el plan completo | pendiente |
| D16 | Covariables en Moirai (MOI-P7) | (a) mantener sin exog; (b) solo pasadas vía `past_feat_dynamic_real` (cambia la salida 2.137e-02 sin evidencia de entrenamiento); (c) llamar a `forward()` con covariables futuras (exige preparar datos en el adaptador, contra `.claude/rules/foundation.md`) | (a) | (a): solo la precisión del docstring y reejecutar `covariates_probe.py` con cada release. (b) o (c): otro PR, con cambios de atributos, tests y docs | pendiente |
| D11 | Contexto todo NaN (MOI-P5), se decide en `02_common.md` | (a) ValueError que nombre las series; (b) warning y predicción NaN para esas series; (c) solo documentar | (a), coherente con el rechazo de series enteramente NaN (la revisión de Moirai proponía warning) | Ninguno aquí: solo cambia el resultado esperado de `all_nan_context.py` en la sección 7 | pendiente |
| D-VER | Política de versión mínima, se decide en `02_common.md` | (a) mínimo solo en `backend_package` (cubre instalaciones nuevas y CI); (b) además un helper común de comprobación en tiempo de ejecución (`importlib.metadata`, patrón de `TimesFM3Adapter`); (c) solo docs | (b), con un helper común pequeño | Con (b), el helper de 02 lanza el caso < 2.0 antes del import y MOI-P1 solo distingue no instalado e import fallido; con (a) o (c), MOI-P1 lee la versión con `importlib.metadata` dentro de `_load_module` y mantiene las tres ramas | pendiente |

## 7. Criterios de aceptación

**Tests unitarios:**

- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_MoiraiAdapter.py -q`: todo pasa, con los tests nuevos de MOI-P1, P2, P3 y V1.
- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_device_handling.py -q`: todo pasa, y el aviso MPS apunta al fichero del test.
- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_get_model_info.py skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py -q`: todo pasa.
- [ ] Una vez al final: `pytest skforecast/foundation/tests/tests_foundation_models/ -q` y las carpetas que indique `python .claude/skills/verify/affected.py`.

**Compatibilidad con el backend real** (`python tools/foundation_compat/run.py --adapter moirai`, o los dos pytest con `--backend moirai` en `~/.venvs/fm-review-moirai`):

- [ ] El entorno se resuelve con uni2ts 2.0.0 y el override `scipy>=1.12`.
- [ ] Contrato: 5 passed (6 si se añade la entrada de `Moirai2Module`).
- [ ] Smoke: 5 passed y 1 skipped (exog), como en la línea base.

**Scripts de evidencia, después del cambio:**

- [ ] `compare_direct.py`: `max|adapter - direct| = 0.000e+00` en todos los casos (2 series, 700/300/40, `steps=150`, contexto 2048, NaN), y `module.quantile_levels` igual a `SUPPORTED_QUANTILES` (`equal: True`).
- [ ] `set_params_reload.py`:
  - "from_pretrained calls after predict with the new context_length: 1" (antes 2).
  - "adapter._module is None after set_params(context_length=1024): False" (antes True).
  - `max|reloaded module - reused module| = 0.000e+00`.
  - "injected module dropped after set_params(context_length=1024): False" (antes True).
  - "injected module, from_pretrained calls after set_params + predict: 0" (antes 1).
  - Si el script deja de reflejar el nuevo flujo, ajustar solo sus mensajes, no los valores comparados.
- [ ] `misc_checks.py`:
  - Sección 1: el script simula la falta de `uni2ts.model.moirai2` con uni2ts 2.0.0 instalado, así que ahora sale el tercer caso de MOI-P1: el mensaje nombra `uni2ts 2.0.0` y la causa original, en lugar de "Install it with `pip install uni2ts`". El caso uni2ts < 2.0 lo cubre el test unitario con la versión simulada.
  - Sección 2: el aviso MPS apunta a la línea 67 del script por `FoundationModel.predict` (antes 90) y a la 71 por `ForecasterFoundation.predict` (como antes), con `"mps"`, `"mps:0"` y `"auto"`. Dispositivo final `cpu`.
  - Sección 3: el `LicenseWarning` ya no muestra `skforecast/foundation/_adapters.py:2069` como ubicación; `pred dtype: float64`.
- [ ] `verification/extra_checks.py`:
  - El `LicenseWarning` apunta a la línea marcada `# user line` del script (línea 44), no a `foundation/_adapters.py:2069`.
  - `context_length=12000`: `finite: True` y `max|adapter - direct| = 0.000e+00`.
- [ ] `covariates_probe.py`: sin cambios respecto a `evidence/moirai/covariates_probe.log` (RuntimeError con `feat_dynamic_real`; `past_feat_dynamic_real` con diferencia 2.137e-02). Confirma que D16 (a) sigue vigente.
- [ ] `all_nan_context.py`, solo con `02_common.md` ya fusionado: la serie `nan_tail` da el error (o el aviso) decidido en D11, no cuantiles ~1e-4.
- [ ] `echo "uni2ts>=2.0" | uv pip compile --python-version 3.13 --torch-backend cpu -` da "No solution found".

**Documentación y cierre:**

- [ ] `python tools/ai/generate_ai_context_files.py --check` sin diferencias tras regenerar.
- [ ] Skill `verify` en verde (lint de los ficheros tocados, tests afectados, contexto de IA, nota de release).
- [ ] Entrada en `docs/releases/releases.md`, sección 0.27.0.

## 8. Fuera de alcance

- **MOI-P5** (contexto todo NaN): se implementa en `02_common.md` (D11).
- **Rechazar al resolver `Salesforce/moirai-2.0-R-base` y `-large`**, que hoy se enrutan y dan 404 en el Hub (X-3): `02_common.md` (sección "Descartado") los deja fuera de la tabla central de ids no soportados (D-REG) porque el error del Hub ya es claro [probado], así que `test_get_model_info.py` (usa `Salesforce/moirai-2.0-R-base`, aprox. línea 298) no cambia. Aquí solo se corrige el docstring.
- **Reejecutar la guía** (la salida de la celda 68 en la revisión y el resto): `10_docs_ci.md`.
- **La clave `python` opcional por entrada en `adapters.toml`** (XC-12 de `evidence/cross_cutting.md`) queda fuera de esta serie de PR (`10_docs_ci.md`, sección 8): el workflow del CI de compatibilidad y `run.py` fijan Python 3.12, la única versión que sirve a uni2ts 2.0 y a tsicl [verificado en código].
- **Los `stacklevel=3` de los avisos reemitidos por TabICL, TabPFN-TS y Nori:** fuera de esta serie de PR (07, 08 y 09 no los tocan); seguimiento posterior a 02 que reutilice el helper trasladado.
- **Los `stacklevel` fijos propios de Timestral (5 y 6):** Timestral no está en esta revisión.
- **Opciones (b) y (c) de D16** (covariables pasadas o futuras en Moirai-2): pospuestas hasta decisión.
- **Una política común de `revision` y `model_kwargs`** para todos los adaptadores (D13 de `evidence/cross_cutting.md`): pospuesta.
- **Medir velocidad o precisión de Moirai-2**, de versiones de uni2ts o de checkpoints, y comparar con otros modelos: fuera del alcance de skforecast. El ahorro de recargas de MOI-P2 no se mide en tiempo: basta el recuento de `from_pretrained` y la diferencia 0.
- **Pruebas en CUDA o MPS reales**, y si Moirai-2 se entrenó con contextos de 8192 o con covariables: no verificables en la revisión.
- **Cambios upstream** (PR #216 y #256, issues #251, #133 y #224): solo se siguen con la comprobación de "Antes de empezar".
