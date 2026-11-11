# 03. TS-ICL (tsicl): capacidades veraces, partición central con exog, cuantiles y horizonte

- **Rama propuesta**: `fix/tsicl-alignment`, creada desde `origin/0.27.x`. Sustituye a las agrupaciones 3 (`fix/tsicl-covariates`) y 8 (`fix/tsicl-quantiles-horizon`) de la sección 5 del informe: aquí todo TS-ICL va en un solo PR.
- **Prioridad**: alta (TSI-P1 y TSI-P2 son de prioridad alta; TSI-P3, TSI-P4, TSI-P7 y TSI-P8, media; TSI-P5 y TSI-P6, baja).
- **Adaptadores afectados**: `TSICLAdapter`. Por el atributo nuevo de TSI-P1, que entra en `_AdapterBase._REQUIRED_CLASS_ATTRIBUTES`, también cambia el cuerpo de clase de todos los adaptadores registrados: `ChronosAdapter`, `TimesFM25Adapter`, `TimesFM3Adapter`, `MoiraiAdapter`, `TabICLAdapter`, `TabPFNAdapter`, `T0Adapter`, `TSICLAdapter`, `NoriAdapter` y `TimestralAdapter` [verificado en código].
- **Ficheros afectados**:
  - Código: `skforecast/foundation/_adapter_base.py`, `skforecast/foundation/_foundation_model.py`, `skforecast/foundation/_adapters.py`, `skforecast/foundation/_utils.py` (solo si el helper de partición vive ahí).
  - Tests: en `skforecast/foundation/tests/tests_foundation_models/`: `fixtures_adapters.py`, `test_TSICLAdapter.py`, `test_predict.py`, `test_AdapterBase.py`, `test_check_exog_columns.py`, `test_get_model_info.py` y, si el helper vive en `_utils.py`, `test_utils.py`; `tools/foundation_compat/tests/test_smoke.py`.
  - Documentación: `docs/user_guides/foundation-forecasting-models.ipynb` (solo texto de TS-ICL, sin reejecutar), `docs/user_guides/foundation-forecasting-with-heterogeneous-series.ipynb` (solo texto markdown de TS-ICL, sin reejecutar), `skills/foundation-forecasting/SKILL.md`, `skills/foundation-forecasting/references/adapter-parameters.md`, `tools/ai/llms-base.txt`, ficheros generados (regeneración), `docs/releases/releases.md`.
- **Depende de**: `02_common.md`, solo de forma parcial:
  - TSI-P3: 02 solo convierte los niveles de cuantil a `float` (D-Q) y pospone la validación central contra `SUPPORTED_QUANTILES` a un PR posterior a 03 (`02_common.md`, sección 8); por tanto `TSICLAdapter.predict` llama a `_validate_supported_quantiles` por su cuenta y ese PR posterior retirará la llamada. La traducción a `Decimal` de TSI-P3 funciona con o sin 02 (también con `int` o `numpy.float32`) [probado, solo la aritmética].
  - TSI-P4: usa el helper común de comprobación de versión si 02 lo introduce (D-VER); si no, una lectura local con `importlib.metadata`.
  - TSI-P1: no depende funcionalmente de 02, pero 02 también toca `FoundationModel.predict` (comprobación de contexto todo NaN, cuantiles), así que hay que rebasar sobre 02 y resolver el conflicto en el mismo método.
  - El resto no depende de 02. `01_t0.md` es independiente; si se fusiona antes, rebasar (cambia el cuerpo de `T0Adapter`, donde este PR añade el atributo nuevo).
- **Bloquea a**: `10_docs_ci.md` (reejecución única de la guía). Los PR `04` a `09` no quedan bloqueados, pero van después en el orden y deben rebasar sobre este: cada clase de adaptador ya declarará aquí el atributo nuevo y sus PR solo tocarán su propio cuerpo de clase.
- **Cierra issue**: ninguna de skforecast. No hay issue sobre TS-ICL; la #1417 abierta es la issue automática del check de compatibilidad (falla por T0; en su tabla tsicl 0.2.1 sale en verde) [documentado]. Hay que abrir una issue upstream en `EDF-Lab/ts-icl` sobre la comparación exacta de floats de los niveles de cuantil (TSI-F7), que no existe [documentado].

## 1. Estado en la revisión (2026-11-06)

Base de la revisión: commit `d9b8f4f` de `origin/0.27.x` (skforecast 0.27.0 en desarrollo). Las líneas citadas en este plan son de ese commit. Desde entonces `origin/0.27.x` ha avanzado; en `skforecast/foundation` solo cambia `_to_covariate_array` de cada adaptador (`col_data.to_numpy(dtype=np.float32)`, commit `2282824`) sin desplazar líneas, y `tools/ai/llms-base.txt` gana 2 líneas hacia la línea 400, así que sus líneas 458, 496, 530 y 531 pasan a ser aprox. 460, 498, 532 y 533 [verificado en código].

| Elemento | Probado en la revisión | Última versión disponible |
|---|---|---|
| `tsicl` (PyPI) | 0.2.1 | 0.2.1 (2026-06-25). Historial: 0.1.0 (2026-06-16), 0.1.1 (2026-06-17), 0.1.3 (2026-06-18), 0.2.0 (2026-06-18), 0.2.1 [documentado] |
| Código upstream (`EDF-Lab/ts-icl`) | tag `v0.2.1` = `61d1e89` | `main` = `349f3ea` (2026-07-17): fusión del PR #1 (commits `a60e7c7` y `2e6624d`, del 2026-07-16, arreglo del rollout con covariables) y revisión de LICENSE, sin release [verificado en código] |
| Checkpoint `taharnbl/TS-ICL` | `tsicl-v1.ckpt` | Mismo fichero. Revisión `19c94031439fb31f36ce395088ee50a6762d3774` (2026-07-07), repo creado el 2026-06-14, 219.2 MB, no gated, licencia `other` / `tsicl-v1-license-v1.0`. El LFS sha256 del checkpoint (prefijo `a67ae9f694c2`, 219150987 bytes) es idéntico en `f01f3869` (2026-06-18) y `19c94031`: el último commit solo cambia LICENSE [probado]. Ningún otro repo en el Hub (`search=tsicl`, `search=ts-icl`, `filter=tsicl`; `author=EDF-Lab` vacío) [probado] |
| Requisitos de `tsicl` 0.2.1 | | `requires_python >=3.12`; `torch>=2.5.1,<2.10`; `numpy>=2.1.3`; `scikit-learn>=1.8.0`; `huggingface-hub>=1.19.0`; `pandas>=2.2.3`; además `einops`, `hydra-core`, `matplotlib`, `seaborn`, `dotenv`. Todas las releases (0.1.0 a 0.2.1) exigen Python >=3.12 y torch <2.10 [documentado]. La última torch publicada es 2.15.0 [documentado] |
| Entorno local | Python 3.12.3, tsicl 0.2.1, torch 2.9.1+cpu, numpy 2.5.4, pandas 2.3.3, huggingface_hub 2.4.0, scikit-learn 1.9.1, skforecast 0.27.0, pytest 9.2.0; Linux x86_64, 4 CPU, sin CUDA ni MPS [probado] | |

Evidencia: `evidence/tsicl/versions.log`, `evidence/tsicl/pypi_info.log`, `evidence/tsicl/hub_info.log`, `evidence/tsicl/hub_license_commits.log`, `evidence/tsicl/verification/hub_revisions.log`, `evidence/tsicl/upstream_diff_v0.2.1_main.log`, `evidence/tsicl/requirements-freeze.txt`.

**Línea base del CI** [documentado]: run 42212742430 del workflow Foundation Backends Compatibility (rama 0.27.x, 2026-11-06), job 127875169087: tsicl 0.2.1, torch 2.9.1+cpu, Python 3.12.16, contrato 3 passed, smoke 6 passed. En local [probado]: contrato 3 passed, smoke 6 passed, unitarios `test_TSICLAdapter.py` 54 passed (`evidence/tsicl/contract.log`, `evidence/tsicl/smoke.log`, `evidence/tsicl/unit_tests.log`, `evidence/tsicl/verification/contract.log`, `evidence/tsicl/verification/tests.log`). El CI está en verde aunque los defectos TSI-F2, F4, F7 y F8 existen (TSI-F14).

**Issues upstream relevantes** (estado del 2026-11-06) [documentado]: PR #1 (arreglo del rollout) cerrado y fusionado el 2026-07-17; abiertas #3 (covariables solo de pasado ignoradas sin `allow_covar_forecast=True`) con PR #4, #5 (conservar las covariables históricas en el caso mixto) con PR #6, #7 (si el entrenamiento cubre covariables solo de pasado), #8 (máscara de covariables compartida en el lote), #9 (mediana fija en la predicción interna de covariables), #10 (exog futura descartada con longitudes distintas), #19 (el tipo de retorno cambia con las longitudes), #20 (`squeeze_output=True` quita todas las dimensiones unitarias), #21 (la predicción interna de covariables sobrescribe el contexto de los lotes siguientes), #23 (`batch_size` no limita las series con listas irregulares); también abiertas #11 a #18 y #22 (validación de entradas), todas del 2026-10-30; PR #2 (`hf_repo_id` configurable) abierto. Ninguna trata la comparación exacta de los niveles de cuantil. Evidencia: `evidence/tsicl/upstream_issues.log`, `evidence/tsicl/upstream_issues_bodies.log`.

### Antes de empezar

Este plan puede quedar desfasado. Desde la raíz del repositorio, recrear el entorno y comprobar si hay versiones nuevas (`make_env.sh` reescribe `evidence/tsicl/requirements-freeze.txt`: no incluir ese cambio en el PR salvo que se quiera actualizar la evidencia):

```bash
bash dev/foundation_adapters_review/evidence/common/make_env.sh tsicl "tsicl" 3.12
PY=~/.venvs/fm-review-tsicl/bin/python
$PY dev/foundation_adapters_review/evidence/common/pypi_info.py tsicl
$PY dev/foundation_adapters_review/evidence/common/pypi_info.py torch
$PY dev/foundation_adapters_review/evidence/common/hub_info.py --repo taharnbl/TS-ICL
$PY dev/foundation_adapters_review/evidence/common/hub_info.py taharnbl --prefix TS-ICL
git ls-remote --tags https://github.com/EDF-Lab/ts-icl
gh api "search/issues?q=repo:EDF-Lab/ts-icl+is:issue" --jq '.items[] | [.number, .state, .title] | @tsv'
gh api "search/issues?q=repo:EDF-Lab/ts-icl+is:pull-request" --jq '.items[] | [.number, .state, .title] | @tsv'
```

Qué hacer si algo cambió:

- **Release de `tsicl` posterior a 0.2.1**: volver a ejecutar `evidence/tsicl/compare_direct.py`, `evidence/tsicl/check_covariates_rollout.py`, `evidence/tsicl/check_quantile_grid.py`, `evidence/tsicl/verification/check_quantile_exact_types.py` y `evidence/tsicl/verification/verify_backend_direct.py` en el entorno nuevo, y revisar `requires_dist` y `requires_python` (TSI-P7). Después:
  - si incluye el PR #1 (rollout), E5 de `check_covariates_rollout.py` debe funcionar con exog: aplicar la opción (b) de D32 (subir el mínimo en `backend_package` y en `tools/foundation_compat/adapters.toml` y no añadir el guard de TSI-P4), o mantener el guard solo para versiones <= 0.2.1;
  - si incluye #4 y #6 (covariables solo de pasado y mixtas), E1 y E2 dejarán de dar 0.000e+00: replantear TSI-P2 (D30) y mantener `supports_past_only_covariates=True` si el efecto es real;
  - si corrige #8 y #10, E3 y E4 deberían dar como mucho ruido de float32 en lote: TSI-P1 sigue siendo correcto, pero D29 puede revisarse;
  - si corrige la comparación de cuantiles, `check_quantile_grid.py` dará 0 rechazos directos: la traducción a `Decimal` de TSI-P3 pasa a ser innecesaria (se puede retirar o dejar, es inocua).
- **Revisión nueva del checkpoint en el Hub**: comprobar si cambia `tsicl-v1.ckpt` (LFS sha256) o aparece otro fichero o repo; si cambia la LICENSE, revisar la entrada `taharnbl/TS-ICL` de `_MODEL_LICENSES` en `skforecast/foundation/_utils.py`.
- **Cambios en `origin/0.27.x`**: rebasar y comprobar que las líneas citadas siguen apuntando a las mismas funciones (buscar por nombre, no por número).

## 2. Hallazgos

Los ids X-n son los del informe (sección 3.10); en `evidence/cross_cutting.md` los mismos hallazgos llevan ids XC-n (X-1 = XC-1, X-2 = XC-5, X-3 = XC-3, X-6 = XC-8, X-8 = XC-6; las fakes y el smoke son XC-7).

- **TSI-F1** [probado] La API del adaptador está alineada con tsicl 0.2.1: contrato 3, smoke 6, unitarios 54. La salida vía `FoundationModel` coincide exactamente (max_abs_diff 0.000e+00) con `TSICL.forecast` directo, con y sin exog, con cuantiles, con NaN en el contexto y con series de longitudes distintas; dos llamadas repetidas dan 0.000e+00. El diff `v0.2.1..main` en `src/` solo toca `_rollout_f`. Evidencia: `evidence/tsicl/compare_direct.log`, `evidence/tsicl/verification/verify_backend_direct.log` (V1), `evidence/tsicl/upstream_diff_v0.2.1_main.log`.
- **TSI-F2** [probado] Las covariables solo de pasado no tienen efecto: la predicción es idéntica (0.000e+00) a no pasar exog; con pasado y futuro cambia (1.347 en E1 y 0.995 en V2). En E1, MAE sin exog 0.4426, solo pasado 0.4426, pasado y futuro 0.2819. Causa: `pipeline.py@v0.2.1` línea 401 (`covar_t = None` sin futuro) y línea 1364 (solo construye las covariables si hay pasado y futuro) [verificado en código]. Con `allow_covar_forecast=True`, que el adaptador no pasa, sí influyen (2.244e-01 frente a sin covariables, porque el backend pronostica antes la covariable, línea 411). `supports_past_only_covariates=True` no es veraz. Issue #3 y PR #4 abiertos. Evidencia: `evidence/tsicl/check_covariates_rollout.log` (E1), `evidence/tsicl/check_registry_options.log`, `evidence/tsicl/verification/verify_backend_direct.log` (V2).
- **TSI-F3** [probado] Con covariables mixtas (una con futuro y otra solo de pasado) la de solo pasado se descarta en silencio: diferencia 0.000e+00 frente a pasar solo la de futuro. Causa: `data_adapter.py@v0.2.1` línea 184 (restringe las claves de `past_covariates` a las de `future_covariates`) [verificado en código]. Issue #5 y PR #6 abiertos. Evidencia: `evidence/tsicl/check_covariates_rollout.log` (E2), `evidence/tsicl/verification/verify_backend_direct.log` (V3).
- **TSI-F4 (corregido)** [probado] Si las series de una llamada llegan al backend con longitudes distintas, este descarta las covariables futuras de todas: s1 junto a una serie más corta da lo mismo que sin exog (0.000e+00) y difiere 1.347 (0.995 en la verificación) de s1 sola con exog; con igual longitud la diferencia es ruido (3.815e-06 en E3, 7.629e-06 en V4). `FoundationModel` agrupa por firma de exog pero no por longitud. Matiz de la verificación: la longitud que cuenta es la que llega tras recortar a `context_length`, así que ocurre en cuanto una serie del grupo es más corta que `context_length` (4096 por defecto, caso frecuente). Causa: `data_adapter.py@v0.2.1` línea 288 (`covars = past_covars` cuando el pasado es lista y el futuro tensor) [verificado en código]; issue #10 abierta. Evidencia: `evidence/tsicl/check_covariates_rollout.log` (E3), `evidence/tsicl/verification/verify_backend_direct.log` (V4).
- **TSI-F5** [probado] Un NaN en la exog de una serie cambia la predicción de otra de la misma llamada: 4.160e-03 (E4) y 7.874e-03 (V5), frente a un ruido sin NaN de 3.8e-06 (E3) y 7.6e-06 (V5). Persiste con series de igual longitud (V5). La máscara de covariables del Perceiver se suma sobre el lote (`perceiver.py@v0.2.1` línea 561, `isnan().sum(0)`) [verificado en código]. Issue #8 abierta. Evidencia: `evidence/tsicl/check_covariates_rollout.log` (E4), `evidence/tsicl/verification/verify_backend_direct.log` (V5).
- **TSI-F6** [probado] Declarar `requires_equal_length_context=True` (relleno NaN en `FoundationModel`) evita la pérdida de exog de la serie larga (s1: 7.629e-06), pero cambia la predicción de la serie rellenada frente a predecirla sola: 8.269e-02 con exog y 2.293e-01 sin exog. Sin relleno: con exog, s1 1.289 y s2 6.746e-01; sin exog, 0.000e+00 en las dos. No es una solución neutra. Evidencia: `evidence/tsicl/check_padding_option.log`, `evidence/tsicl/verification/rerun_check_padding_option.log` (mismos números).
- **TSI-F7 (corregido)** [probado] 8 niveles de `SUPPORTED_QUANTILES` (0.07, 0.14, 0.28, 0.29, 0.55, 0.56, 0.57, 0.58) los rechaza el backend con `ValueError` (`Misspecified arg quantile_levels=[7.000000000000001]`), tanto en `TSICL._get_quantile_indices` como vía `FoundationModel.predict`: `pipeline.py@v0.2.1` líneas 201-203 compara `100*x` con `np.linspace(1, 99, 99)` por pertenencia exacta de floats, y ningún float vecino (±50 ulp) la cumple. Corrección de la verificación: es falso que el adaptador no pueda sortearlo; el backend solo usa el valor para buscar el índice (líneas 201-207, 973, 1027) y, pasando `Decimal(str(q))` o `Fraction(str(q))`, obtiene el índice correcto (6, 13, 27, 28, 54, 55, 56, 57), la llamada funciona, q0.06 y q0.08 salen idénticos a la referencia (0.000e+00) y q0.07 queda entre ambos en todos los pasos. Sin issue upstream (títulos #1 a #23 comprobados). Evidencia: `evidence/tsicl/check_quantile_grid.log`, `evidence/tsicl/verification/check_quantile_exact_types.log`, `evidence/tsicl/verification/verify_backend_direct.log` (V6).
- **TSI-F8** [probado] Con 0.2.1 (`max_target_length=672`, `max_context_length=4096`): `steps=700` sin exog funciona (700 filas finitas); `steps=1344` sin exog falla (`ValueError: Arg prediction_length should be a >0 int, received 0`, `pipeline.py` línea 1270); `steps=700` y `steps=1344` con exog futura fallan (`RuntimeError: Sizes of tensors must match except in dimension 0`, línea 1101). El rollout solo se usa con `steps>672` (línea 1068); la causa está en las líneas 1097 (`pred_len = prediction_length % max_target_length`, 0 en los múltiplos de 672), 1103 y 1131 [verificado en código]. Arreglado en main (`a60e7c7`, `2e6624d`: `pred_len = prediction_length - start` y concatenación con `dim=-2`; PR #1 cerrado el 2026-07-17) sin release [verificado en código, no ejecutado]. Evidencia: `evidence/tsicl/check_covariates_rollout.log` (E5), `evidence/tsicl/verification/verify_backend_direct.log` (V7), `evidence/tsicl/upstream_diff_v0.2.1_main.log`.
- **TSI-F9** [verificado en código] `TSICL._get_device` convierte MPS en CPU en silencio cuando no hay CUDA (`pipeline.py@v0.2.1` líneas 211-217; `_get_device(mps) -> cpu` [probado] en CPU). El adaptador no pasa `mps_fallback_message` a `_resolve_torch_device` (definida en `_adapters.py` aprox. línea 34 en d9b8f4f; `TSICLAdapter.predict` la llama aprox. en la línea 4251), como sí hacen `MoiraiAdapter` (aprox. línea 2083) y `TimestralAdapter` (aprox. línea 5885); solo lo documenta en el docstring y en `adapter-parameters.md` línea 160. MPS real no comprobado (sin hardware). Evidencia: `evidence/tsicl/verification/verify_backend_direct.log` (V8), `evidence/tsicl/check_registry_options.log`.
- **TSI-F10** [probado] `context_length>4096` se recorta a 4096 en el backend sin aviso (`context_length=10000 -> this_context_length=4096`; `pipeline.py@v0.2.1` línea 763, `min(self.max_context_length, context_length)`). El adaptador solo valida entero positivo (`_validate_positive_int` en `TSICLAdapter.__init__` y en la función `validate` de `set_params`). Evidencia: `evidence/tsicl/verification/verify_backend_direct.log` (V8).
- **TSI-F11** [probado] Registro, checkpoint y acceso correctos: `taharnbl/TS-ICL` resuelve a `TSICLAdapter`; `taharnbl/ts-icl` y `EDF-Lab/TS-ICL` se rechazan con `ValueError`; `taharnbl/TS-ICL-v2` resolvería por prefijo y descargaría `tsicl-v1.ckpt`, porque el repo está fijo en el backend (`hf_repo_ID = "taharnbl/TS-ICL"`, `pipeline.py@v0.2.1` línea 88). Hub sin gating, licencia `tsicl-v1-license-v1.0`, igual que `_MODEL_LICENSES`; `weights_repo_id` y `weights_in_hf_cache` coinciden con el `hf_hub_download` del backend. Evidencia: `evidence/tsicl/hub_info.log`, `evidence/tsicl/verification/hub_revisions.log`, `evidence/tsicl/verification/registry_license.log`, `evidence/tsicl/check_registry_options.log`.
- **TSI-F12 (corregido)** [documentado] La LICENSE se revisó (texto "Last Revised: June 30, 2026", commit del Hub del 2026-07-07, igual que la de main salvo el salto de línea final): sigue siendo no comercial. Cambian la definición de "Derivative" (incluye destilar mediante los Outputs) y la redacción de "Non-Commercial Purpose" y de la cláusula de Outputs ("data produced" pasa a "Outputs" en 3.a). La versión del 9 de junio ya incluía el código de inferencia en "TS-ICL Model" y ya restringía los Outputs a fines no comerciales. Sin cambio material para skforecast: la entrada de `_MODEL_LICENSES` y el texto del `LicenseWarning` siguen correctos. Evidencia: `evidence/tsicl/hub_license_commits.log`, `evidence/tsicl/verification/registry_license.log`.
- **TSI-F13** [documentado] Todas las releases de tsicl (0.1.0 a 0.2.1) exigen Python >=3.12 y torch >=2.5.1,<2.10; la 0.2.1 exige además scikit-learn >=1.8.0, numpy >=2.1.3 y huggingface-hub >=1.19.0, por encima de los mínimos de skforecast. skforecast soporta Python 3.10 a 3.14 (`tools/ai/llms-base.txt` línea 13). Ni la guía (celda 11), ni `SKILL.md`, ni `llms-base.txt` lo dicen. Que `pip install tsicl` no encuentre versión en Python 3.10 y 3.11 es [inferido] de `requires_python`; el tope de torch puede chocar con otros backends en el mismo entorno [inferido]. Evidencia: `evidence/tsicl/pypi_info.log`.
- **TSI-F14** [probado] Con `squeeze_output=False` el backend devuelve un `Tensor` de forma `(b, 1, steps, q)` si las series tienen igual longitud (`(2, 1, 24, 1)` en V0) y una lista de `(1, steps, q)` si no (issue upstream #19). `FakeTSICL` (`fixtures_adapters.py` aprox. líneas 290-338) devuelve siempre lista. No hay tests unitarios de la rama tensor ni de TSI-F2 a TSI-F8; el smoke solo comprueba forma y finitud (`STEPS = 12`). Evidencia: `evidence/tsicl/verification/verify_backend_direct.log` (V0), `evidence/tsicl/check_registry_options.log`.
- **TSI-F15** [verificado en código] Opciones del backend que el adaptador no usa (firma de `TSICL.forecast`): `batch_size` (64; solo agrupa series de igual longitud, issue #23), `allow_covar_forecast` (ver TSI-F2; issues #21 y #9), `allow_auto_complete`, `point_estimator` (`"mean"`; irrelevante, el adaptador usa el cuantil 0.5). El PR abierto #2 añadiría `hf_repo_id` a `TSICL`. Evidencia: `evidence/tsicl/check_registry_options.log`, `evidence/tsicl/verification/registry_license.log`.
- **TSI-F16** [verificado en código] El adaptador es fino: un dict por serie (`TSICLAdapter._build_tsicl_input`) y una sola llamada a `forecast` en `TSICLAdapter.predict`, sin bucles de agrupación, relleno ni reintentos. Sus parches propios siguen vigentes: conversión a float32 con `ValueError` para no numéricas (`_to_covariate_array`), `squeeze_output=False` (necesario: `squeeze_output=True` hace un `.squeeze()` total, `pipeline.py@v0.2.1` líneas 779-781, issue #20), caché del dispositivo resuelto (`_resolved_device`) y `LicenseWarning`. No hay workarounds de versión que retirar; lo que falla son atributos de capacidad (TSI-F2, F3, F4, F7). Cumple `.claude/rules/foundation.md`: las correcciones van en atributos y en `FoundationModel`.
- **TSI-F17 (corregido)** [probado] Afirmaciones falsas con tsicl 0.2.1:
  - (a) covariables solo de pasado soportadas: docstring (Notes) de `TSICLAdapter`, `SKILL.md` líneas 130 y 287, `llms-base.txt` línea 530; añadido en la verificación, `adapter-parameters.md` línea 249; añadido en este plan, docstring de la propiedad `FoundationModel.supports_past_only_covariates` (aprox. línea 396, lista TS-ICL entre los `True`) [verificado en código];
  - (b) toda la rejilla 0.01: `SKILL.md` línea 159, `llms-base.txt` línea 496, docstring de `TSICLAdapter`; con el mismo texto, `SKILL.md` líneas 46 y 281 y `adapter-parameters.md` línea 152 [verificado en código];
  - (c) horizonte sin límite: guía principal, celda 11 ("No hard limit, set via steps");
  - (d) independencia entre series: `llms-base.txt` línea 531, docstring de `FoundationModel.predict` (aprox. línea 1114, "never depends on the exog columns of the other series") y el final de `SKILL.md` línea 130 ("so the forecast of a series never depends on the exog of the others") [verificado en código];
  - (e) Python >=3.12 no documentado;
  - añadido en la verificación: `docs/user_guides/foundation-forecasting-with-heterogeneous-series.ipynb` celda 2 ("Not for Chronos-2, TS-ICL and TabICL: the forecast of a series is identical to forecasting it alone") y celda 21 (tabla "Past-only covariates: Yes" y texto "Past-only covariates are supported").

  Evidencia: `evidence/tsicl/check_covariates_rollout.log`, `evidence/tsicl/check_quantile_grid.log`, `evidence/tsicl/verification/verify_backend_direct.log`.
- **TSI-V1** [probado] La guía de series heterogéneas afirma para TS-ICL independencia del lote (celda 2) y soporte de covariables solo de pasado (celda 21), y `adapter-parameters.md` línea 249 repite lo segundo: las dos cosas son falsas con tsicl 0.2.1 (V2 diferencia 0 con solo pasado; V4 0.995 cuando una serie hermana más corta descarta la exog; V5 efecto cruzado del NaN 7.9e-03). Forma parte de X-8. Evidencia: `evidence/tsicl/verification/verify_backend_direct.log`, `evidence/tsicl/verification/verification.md`.
- **TSI-V2** [probado] Los 8 niveles rechazados se pueden servir traduciendo el nivel a un número exacto (`decimal.Decimal(str(q))` o `fractions.Fraction`) antes de llamar al backend: índices 6, 13, 27, 28, 54, 55, 56, 57; q0.06 y q0.08 idénticos a la referencia (0.000e+00); q0.06 <= q0.07 <= q0.08 en todos los pasos. Es un detalle propio del adaptador, retirable cuando upstream corrija la comparación. Evidencia: `evidence/tsicl/verification/check_quantile_exact_types.log`.

Hallazgos transversales relacionados que este plan no posee pero toca: X-8 (garantía de independencia entre series; aquí solo la parte TS-ICL), X-2 (tipo de los niveles de cuantil, en 02), X-3 (`taharnbl/TS-ICL-v2` aceptado por prefijo, en 02), X-6 (stacklevel fijos), X-1 (contexto todo NaN, en 02). Evidencia: `evidence/cross_cutting.md`, `evidence/cross_cutting/probe_shared.log`.

### Descartado

- "El adaptador no puede sortear los niveles de cuantil rechazados traduciendo el valor" (parte de la versión inicial de TSI-F7): refutado, `Decimal` o `Fraction` lo sortean (TSI-V2) [probado, `evidence/tsicl/verification/check_quantile_exact_types.log`].
- "La LICENSE nueva incluye por primera vez el código de inferencia y restringe los Outputs" (versión inicial de TSI-F12): refutado, ya estaba en el texto del 9 de junio [documentado, `evidence/tsicl/hub_license_commits.log`].
- Nota opcional de TSI-P7 "la licencia cubre también el código de inferencia": descartada, no es un cambio nuevo.
- Opción de relleno (`requires_equal_length_context=True`) como arreglo de TSI-F4: descartada por TSI-F6.

## 3. Cambios a implementar

Orden recomendado dentro del PR: TSI-P2, TSI-P1, TSI-P8 (fake), TSI-P3, TSI-P4, TSI-P5, TSI-P6 y, al final, TSI-P7.

### TSI-P1. Partición central de llamadas cuando hay exog (atributo de capacidad nuevo)

- **Qué**:
  - Añadir a `_AdapterBase._REQUIRED_CLASS_ATTRIBUTES` (`skforecast/foundation/_adapter_base.py`, aprox. línea 44 en d9b8f4f) un atributo booleano nuevo, justo después de `"requires_equal_length_context"`. Nombre propuesto con la opción (b) de D29: `requires_single_series_with_exog` (`True`: `FoundationModel` llama al adaptador una vez por cada serie que lleva exog). Con la opción (a) cambiarían semántica (agrupar también por longitud tras el recorte) y nombre (por ejemplo `requires_equal_length_with_exog`). El nombre final es del implementador; el patrón `requires_*` sigue al de `requires_equal_length_context`.
  - Documentarlo en el docstring de `_AdapterBase`, con un párrafo como el de `requires_equal_length_context`.
  - Como `__init_subclass__` exige que cada clase lo declare en su propio cuerpo (los valores heredados no cuentan), declararlo en los 10 adaptadores de `_adapters.py`, junto a `requires_equal_length_context` (aprox. líneas en d9b8f4f): `ChronosAdapter` (195), `TimesFM25Adapter` (716), `TimesFM3Adapter` (1189), `MoiraiAdapter` (1814), `TabICLAdapter` (2254), `TabPFNAdapter` (2900), `T0Adapter` (3532), `TSICLAdapter` (4047), `NoriAdapter` (4551), `TimestralAdapter` (5266). Valor `True` solo en `TSICLAdapter`; `False` en el resto, que conserva el comportamiento actual. Añadir la entrada en la sección Attributes del docstring de cada adaptador, junto a la de `requires_equal_length_context` (que ya existe en los 10). Si antes de este PR se fusiona otro adaptador (TimeH2O, o el PR abierto #1341 de TiRex-2 [documentado]), también debe declararlo.
  - En `FoundationModel.predict` (`_foundation_model.py`, bloque que calcula `series_groups`, aprox. líneas 1242-1251), después de `group_series_by_exog_signature` (o de `[series_names_in]`), y si el adaptador declara el atributo, sacar a una llamada propia cada serie cuya firma (`get_exog_signature`) no es `((), ())`, sea de solo pasado o con futuro. Las series sin exog siguen juntas en una sola llamada: sin covariables no hay pérdida (TSI-F4) ni efecto de la máscara (TSI-F5), y sin exog con longitudes distintas la predicción en lote coincide con la individual (0.000e+00 en `check_padding_option.log`) [probado]. La firma se calcula por serie, no por grupo, para que el atributo sea correcto también si un adaptador lo combina con `supports_heterogeneous_covariates=True`. La lógica vive una sola vez en `FoundationModel` o en un helper de `_utils.py` junto a `group_series_by_exog_signature`, nunca en el adaptador, como pide `.claude/rules/foundation.md`. Ejemplo ilustrativo:

    ```python
    def split_series_with_exog(series_names, context_exog, exog):
        """One group with the series without exog, one group per series with exog."""
        without_exog, with_exog = [], []
        for series_name in series_names:
            signature = get_exog_signature(
                context_exog = context_exog.get(series_name) if context_exog is not None else None,
                exog         = exog.get(series_name) if exog is not None else None,
            )
            if signature == ((), ()):
                without_exog.append(series_name)
            else:
                with_exog.append([series_name])
        return ([without_exog] if without_exog else []) + with_exog


    # In FoundationModel.predict, after series_groups is computed:
    if self.adapter.requires_single_series_with_exog:
        series_groups = [
            part
            for group in series_groups
            for part in split_series_with_exog(group, context_exog, exog)
        ]
    ```

  - La partición se aplica en los dos caminos (`check_inputs=True` y el de `backtesting_foundation`, `check_inputs=False`), porque el cálculo de `series_groups` está fuera de la rama `check_inputs` [verificado en código]. El orden de la salida no cambia: el DataFrame se construye desde `series_names_in`.
  - No se recomienda exponer el atributo en `FoundationModelInfo` (`_model_info.py`) ni como propiedad de `FoundationModel`, igual que `requires_equal_length_context`, que tampoco se expone [verificado en código]. Si se expone, hay que actualizar `test_get_model_info.py` y `test_list_adapters.py`.
  - Actualizar el docstring de `FoundationModel.predict` (aprox. líneas 1104-1120): con el atributo, cada serie con exog va en su propia llamada, así que su predicción no depende de las demás series; distinguir "no recibe la exog de otras series" (lo que garantiza la agrupación) de "la salida es idéntica a predecirla sola" (depende del backend), como pide X-8.
- **Por qué**: TSI-F4 y TSI-F5 (y TSI-F17 (d), TSI-V1, X-8). La opción (b) arregla las dos: TSI-F5 persiste con igual longitud [probado, V5], así que agrupar por longitud no basta. Medidas: igual longitud 7.6e-06 frente a 0.995 con longitudes distintas [probado].
- **Riesgo**: bajo en API; solo cambia la partición de llamadas. Con multiserie y exog, TS-ICL hace una llamada por serie en vez de una por grupo: más llamadas al backend [inferido]; no se mide (fuera de alcance). Afecta a los demás adaptadores solo en que declaran el atributo: con `False` el comportamiento es el actual [inferido; lo confirma la carpeta de tests].
- **Compatibilidad**: no rompe la API pública. Cambia el contrato de `_AdapterBase` (atributo obligatorio nuevo): un adaptador externo que herede de `_AdapterBase` fallaría con `TypeError` al definirse hasta declararlo. Cambian (se corrigen) las predicciones de TS-ICL en multiserie con exog y longitudes distintas tras el recorte o con NaN en la exog; con exog y series de igual longitud cambian solo en ruido de float32 (del orden de 1e-5); la serie única de la guía no cambia [inferido].
- **Decisión**: D29.

### TSI-P2. `supports_past_only_covariates = False` en `TSICLAdapter`

- **Qué**: cambiar el atributo de clase de `TSICLAdapter` (aprox. línea 4044 en d9b8f4f) a `False`. `FoundationModel._check_exog_columns` emitirá entonces `IgnoredArgumentWarning` para las columnas históricas sin valores futuros (`_foundation_model.py` aprox. línea 992); las columnas se siguen reenviando al backend, que las ignora (solo pasado) o las descarta (mixto), así que las predicciones no cambian (diferencia 0 en E1/E2 y V2/V3) [probado]. El aviso solo sale en el camino `check_inputs=True` [verificado en código]. Actualizar el docstring de la propiedad `FoundationModel.supports_past_only_covariates` (aprox. línea 396: TS-ICL pasa de la lista `True` a la lista `False`) y el párrafo Notes de `TSICLAdapter`.
- **Por qué**: TSI-F2, TSI-F3, TSI-F17 (a), TSI-V1.
- **Riesgo**: bajo, solo añade un aviso veraz. Cambia el valor que devuelven `get_model_info("taharnbl/TS-ICL").supports_past_only_covariates` y `list_adapters()` [verificado en código].
- **Compatibilidad**: no rompe; un aviso nuevo para quien pase columnas solo de pasado.
- **Decisión**: D30. La opción (b), pasar `allow_covar_forecast=True`, no cubre el caso mixto (`pipeline.py` línea 411 exige `not covar_on_future`) [verificado en código] y arrastra #21 y #9.

### TSI-P3. Niveles de cuantil de TS-ICL

- **Qué** (opción (d) de D31, recomendada): en `TSICLAdapter.predict`, justo antes de `self._model.forecast`, traducir cada nivel de `query_levels` a un `Decimal` exacto de la rejilla, sin cambiar los nombres de columna (los construye `FoundationModel` con los niveles pedidos por el usuario). El backend solo usa el nivel para buscar su índice, así que el resto de niveles no cambia (índices idénticos) [probado]. Ejemplo ilustrativo:

    ```python
    from decimal import Decimal

    # tsicl<=0.2.1 looks each level up with an exact float comparison of
    # 100 * q against np.linspace(1, 99, 99), which rejects 0.07, 0.14, ...
    # An exact Decimal on the 0.01 grid maps to the same index.
    backend_levels = [Decimal(round(q * 100)) / 100 for q in query_levels]
    ```

  - La forma verificada con el backend es `Decimal(str(q))` [probado]. La variante con `round(q * 100)` da el mismo `Decimal` para los 99 niveles de la rejilla y además acepta un `float` que venga de `numpy.float32` (`float(np.float32(0.07))` es `0.07000000029802322`, cuyo `str` no está en la rejilla): `100 * (Decimal(round(q * 100)) / 100)` pertenece a la rejilla del backend para los 99 niveles y para ese `float32`, y `100 * Decimal(str(q))` no pertenece para el `float32` [probado, solo la aritmética con numpy, sin llamar al backend]. Antes de redondear hay que validar que el nivel está en la rejilla (tolerancia `1e-9` de `_validate_supported_quantiles`, `_utils.py` aprox. línea 64) para no convertir en silencio, por ejemplo, 0.075 en 0.08. Con esa tolerancia, niveles `float32` como `np.float32(0.55)` (error 1.2e-08) o `np.float32(0.9)` (2.4e-08) se rechazan, igual que hoy en TimesFM y Moirai [probado, solo la aritmética]; la política de tolerancia es común y corresponde a 02.
  - Validación: llamar a `_validate_supported_quantiles(quantiles, self.SUPPORTED_QUANTILES, "TS-ICL")` en `TSICLAdapter.predict`, antes de la traducción a `Decimal`, como `TimesFM25Adapter` (aprox. línea 922), `TimesFM3Adapter` (aprox. línea 1448) y `MoiraiAdapter` (aprox. línea 1989) (02 pospone la validación central a un PR posterior a 03, `02_common.md`, sección 8, que retirará esta llamada). Quitar el comentario "Validation of this grid is delegated to the tsicl library" sobre `SUPPORTED_QUANTILES` (aprox. línea 4041) y corregir el párrafo de Notes que dice que el error sale de la librería.
  - Con la opción (a) de D31: filtrar los 8 niveles de `SUPPORTED_QUANTILES` (o generarla con la misma aritmética del backend) y validar; los textos de SKILL, llms y docstring deben listar las excepciones.
  - En ambos casos, abrir una issue upstream en `EDF-Lab/ts-icl` con el ejemplo de `check_quantile_grid.log` (opción (c)).
- **Por qué**: TSI-F7, TSI-V2, TSI-F17 (b).
- **Riesgo**: la opción (d) depende de un detalle interno del backend (el nivel solo se usa para buscar el índice); por eso necesita un test que fije el comportamiento con un fake que reproduzca la comparación exacta, y la comprobación real en el entorno de revisión.
- **Compatibilidad**: no rompe; esos 8 niveles hoy fallan.
- **Decisión**: D31 (y la parte "niveles de cuantil a float" de `02_common.md`, D-Q, para saber qué tipo llega).

### TSI-P4. Horizontes mayores que 672 pasos con tsicl <= 0.2.1

- **Qué**: al principio de `TSICLAdapter.predict`, antes de `_load_model` (para fallar sin descargar pesos), lanzar un `ValueError` claro solo cuando `steps > 672` y (alguna serie de la llamada tiene exog futura, o `steps % 672 == 0`) y la versión instalada de `tsicl` es <= 0.2.1. Con `steps <= 672` el backend nunca usa el rollout (`pipeline.py` línea 1068) [verificado en código], así que no hay que limitar nada. La versión se lee con el helper común de 02 si existe (D-VER) o con `importlib.metadata.version("tsicl")`, como `TimesFM3Adapter._load_model` (aprox. línea 1658). Si la versión no se puede leer (`PackageNotFoundError`, por ejemplo con un `model=` inyectado en tests), no aplicar el guard [inferido: no hay backend tsicl que proteger]; los tests sustituyen la lectura con monkeypatch. Guardar el 672 en una constante privada con comentario (`TSICL.max_target_length` en 0.2.1 [probado]). Ejemplo ilustrativo del mensaje:

    ```python
    raise ValueError(
        f"tsicl {installed} cannot forecast more than 672 steps with future "
        f"exogenous variables, nor a multiple of 672 steps (got steps={steps}). "
        f"Reduce `steps` or forecast in several calls."
    )
    ```

  - El guard es una restricción del backend dependiente de su versión, no manejo genérico de la entrada; por eso vive en el adaptador y no contradice `.claude/rules/foundation.md` [inferido].
  - Cuando upstream publique el PR #1: subir el mínimo en `backend_package` y en `tools/foundation_compat/adapters.toml`, y retirar el guard (D32 (b), PR posterior).
- **Por qué**: TSI-F8, TSI-F17 (c).
- **Riesgo**: bajo. Si el guard comprobara mal la versión, bloquearía horizontes que con una release nueva funcionan; el test debe cubrir una versión > 0.2.1.
- **Compatibilidad**: no rompe; esos casos ya fallan con errores crípticos.
- **Decisión**: D32.

### TSI-P5. Aviso de caída a CPU con MPS

- **Qué**: en `TSICLAdapter.predict`, donde se resuelve `self._resolved_device` (aprox. línea 4251), pasar `mps_fallback_message` a `_resolve_torch_device`, como `MoiraiAdapter` (aprox. línea 2083). Ejemplo: `"MPS device is not supported by TS-ICL: tsicl runs on CPU when CUDA is not available. Falling back to CPU."`. El dispositivo resuelto pasa a ser `"cpu"`, lo mismo que hoy hace el backend en silencio, así que las predicciones no cambian [inferido de TSI-F9]. El aviso sale una vez por instancia, porque el dispositivo se cachea en `_resolved_device` (y se invalida con `set_params(device=...)`). Para `stacklevel`, `_resolve_torch_device` -> `TSICLAdapter.predict` -> `FoundationModel.predict` deja la llamada del usuario a `FoundationModel.predict` en `stacklevel=4` [inferido de la cadena de llamadas]; por `ForecasterFoundation` o `backtesting_foundation` apunta a otro marco (X-6), así que el test no debe fijar la línea. Actualizar el docstring de `device` en la clase y en `__init__`.
- **Por qué**: TSI-F9.
- **Riesgo**: muy bajo. El test existente `test_TSICLAdapter_predict_caches_resolved_device_across_calls` sustituye `_resolve_torch_device` por una función de un solo argumento (`counting_resolve(device)`): hay que darle `**kwargs` o fallará con `TypeError` [verificado en código].
- **Compatibilidad**: no rompe.
- **Decisión**: ninguna.

### TSI-P6. `context_length` mayor que 4096

- **Qué**: documentar en el docstring de `context_length` (clase y `__init__`) que el backend recorta el contexto a 4096 y emitir un `IgnoredArgumentWarning` cuando `context_length > 4096`. Hay dos sitios posibles: la validación de `__init__` y `set_params` (la función `validate` interna), o `predict` antes de llamar al backend. Se prefiere `predict` [inferido], porque `ForecasterFoundation.__init__` clona el estimador con `sklearn.base.clone` [verificado en código] y un aviso en `__init__` se repetiría en cada clonado.
- **Por qué**: TSI-F10.
- **Riesgo**: muy bajo.
- **Compatibilidad**: no rompe (las predicciones no cambian, ya se recortaba).
- **Decisión**: D33.

### TSI-P7. Documentación y contexto de IA de TS-ICL

- **Qué**: documentar Python >=3.12 y torch >=2.5.1,<2.10 (y los mínimos de scikit-learn, numpy y huggingface-hub de tsicl 0.2.1); corregir todas las afirmaciones de TSI-F17 y TSI-V1, de forma coherente con lo que se decida en P1 a P4 y P6; regenerar los ficheros generados. Detalle en la sección 5.
- **Por qué**: TSI-F13, TSI-F17, TSI-V1 (TSI-F12 no requiere cambio).
- **Riesgo**: nulo.
- **Compatibilidad**: no rompe.
- **Decisión**: el texto depende de D29 a D33.

### TSI-P8. Tests y compat de TS-ICL

- **Qué**: un `FakeTSICL` fiel al backend (rama tensor, registro de llamadas, comparación exacta de cuantiles), tests unitarios de P1 a P6 y dos invariantes en el smoke de compatibilidad, activadas por el atributo nuevo. Detalle en la sección 4. `tools/foundation_compat/tests/contracts.py` solo cambia si se elige D30 (b) (añadir `allow_covar_forecast` a los kwargs de `tsicl:TSICL.forecast`).
- **Por qué**: TSI-F14 (y que TSI-F2, F4, F7 y F8 dejen de pasar el CI en verde).
- **Riesgo**: las invariantes nuevas del smoke fallan hasta aplicar P1, así que van en este mismo PR.
- **Compatibilidad**: no rompe.
- **Decisión**: ninguna.

## 4. Tests

Unitarios (entorno principal, sin `tsicl`: todos usan `FakeTSICL`):

- `skforecast/foundation/tests/tests_foundation_models/fixtures_adapters.py`, clase `FakeTSICL`:
  - registrar cada llamada en una lista (por ejemplo `self.calls`, con `inputs` y `quantile_levels`) además de los `last_*` actuales;
  - devolver un array apilado de forma `(b, 1, steps, q)` cuando todas las series (`inputs[i]["target"]`) tienen la misma longitud y una lista de `(1, steps, q)` si no, como el backend real (V0);
  - reproducir la comprobación exacta del backend: `ValueError` si algún `100 * q` no pertenece a `np.linspace(1, 99, 99).tolist()`, para que el test de P3 falle sin la traducción;
  - convertir los niveles a `float` solo para construir los valores devueltos (pueden llegar como `Decimal`).
- `test_TSICLAdapter.py`:
  - `TSICLAdapter.supports_past_only_covariates is False` y el atributo nuevo es `True` (P1, P2);
  - salida correcta con la rama tensor (series de igual longitud) y con la rama lista (P8);
  - los 99 niveles de `SUPPORTED_QUANTILES` funcionan con el fake estricto, las columnas de `FoundationModel.predict` se llaman `q_0.07`, etc., y el fake recibe niveles cuyo `100 * q` está en la rejilla; un nivel fuera de la rejilla (por ejemplo 0.075) da el `ValueError` de skforecast; `float(np.float32(0.07))` funciona (P3, opción (d));
  - `ValueError` del guard con `steps=700` y exog futura, y con `steps=1344` sin exog; sin error con `steps=700` sin exog y con `steps=672` con exog; sin error si se simula una versión de `tsicl` > 0.2.1 o si la versión no se puede leer (monkeypatch de la lectura de versión) (P4);
  - `UserWarning` de MPS y el fake recibe `torch.device("cpu")` al simular `torch.cuda.is_available() = False` y `torch.backends.mps.is_available() = True`, siguiendo `test_TSICLAdapter_predict_caches_resolved_device_across_calls` y `test_MoiraiAdapter_ensure_forecast_obj_mps_falls_back_to_cpu` de `test_device_handling.py`; actualizar `counting_resolve` del primero para aceptar `**kwargs` (P5);
  - `IgnoredArgumentWarning` con `context_length=5000` y ninguno con 4096 (P6).
- `test_predict.py`: con `FoundationModel("taharnbl/TS-ICL", model=FakeTSICL())` y las entradas de `_make_heterogeneous_inputs` (series "full", "none", "past", "full2"), comprobar que hay 4 llamadas (`len(fake.calls)`): "none" sola en la llamada sin exog y "full", "past" y "full2" cada una en la suya; que la salida mantiene el orden de `series_names_in`; y lo mismo en el camino `check_inputs=False`. Añadir un caso con varias series sin exog y longitudes distintas que compartan una llamada. Para adaptadores con el atributo a `False`, `test_predict_groups_series_by_exog_signature_when_adapter_requires_it` (Chronos con `FakePipeline`, 3 llamadas) y `test_predict_single_call_when_adapter_supports_heterogeneous_covariates` deben seguir en verde sin cambios.
- `test_utils.py` (solo si el helper vive en `_utils.py`): orden de los grupos y tratamiento de `context_exog`/`exog` a `None`.
- `test_AdapterBase.py`: actualizar el `err_msg` de `test_AdapterBase_TypeError_when_subclass_misses_required_class_attributes` (la lista de atributos que faltan incluye el nuevo justo después de `'requires_equal_length_context'`) y añadir en `test_AdapterBase_every_registered_adapter_honors_the_contract` la comprobación `isinstance(adapter_cls.<atributo nuevo>, bool)`.
- `test_check_exog_columns.py`: `(TSICLAdapter, True)` pasa a `(TSICLAdapter, False)` en `test_adapters_declare_supports_past_only_covariates`, y `"taharnbl/TS-ICL"` pasa de `_PAST_ONLY_ADAPTERS` a `_FUTURE_ONLY_ADAPTERS` (los tests parametrizados con esas listas cubren entonces el aviso).
- `test_get_model_info.py`: hoy no hay ninguna aserción de `supports_past_only_covariates` para TS-ICL [verificado en código]; añadir una (`False`). Si se decide filtrar niveles (D31 (a)), actualizar el caso `("taharnbl/TS-ICL", tuple(round(0.01 * i, 2) for i in range(1, 100)))` de `test_get_model_info_supported_quantiles` (aprox. línea 209).
- `skforecast/foundation/tests/tests_forecaster_foundation/`: no hay aserciones sobre TS-ICL; se ejecuta la carpeta porque cambia el contrato base.

Compatibilidad (`tools/foundation_compat`):

- `tests/test_smoke.py`: dos tests nuevos que se ejecutan solo si `model.adapter` declara el atributo nuevo a `True` (hoy, `--backend tsicl`): (1) una serie predicha sola da exactamente lo mismo (0.0) que dentro de una llamada con otra serie más corta, ambas con exog pasada y futura; la serie corta debe tener menos observaciones que `CONTEXT_LENGTH = 64` (por ejemplo 40), porque el fallo depende de la longitud tras el recorte (TSI-F4); (2) la exog futura cambia la predicción (diferencia mayor que 0 frente a sin exog; `series_1` depende de `exog_1`). Un tercero opcional: `steps=700` con exog da el `ValueError` del guard mientras tsicl <= 0.2.1. Las demás invariantes genéricas del smoke (XC-7 de `evidence/cross_cutting.md`) quedan fuera de esta serie de PR (`10_docs_ci.md`, sección 8); si se quieren, van en un PR aparte `chore/foundation-compat-checks` (PR-12 de `evidence/cross_cutting.md`), después de 01 y 03.
- `tests/contracts.py`: sin cambios (salvo D30 (b)). `adapters.toml`: sin cambios en este PR (el mínimo se sube en un PR posterior, D32 (b)).

Comprobaciones antes y después con la evidencia (entorno `~/.venvs/fm-review-tsicl`, desde la raíz del repositorio):

```bash
PY=~/.venvs/fm-review-tsicl/bin/python
$PY dev/foundation_adapters_review/evidence/tsicl/compare_direct.py
$PY dev/foundation_adapters_review/evidence/tsicl/check_covariates_rollout.py
$PY dev/foundation_adapters_review/evidence/tsicl/check_quantile_grid.py
$PY dev/foundation_adapters_review/evidence/tsicl/verification/check_quantile_exact_types.py
$PY dev/foundation_adapters_review/evidence/tsicl/verification/verify_backend_direct.py
$PY dev/foundation_adapters_review/evidence/tsicl/check_padding_option.py   # opcional
```

Valores esperados después del cambio (antes: los de la sección 2):

- `compare_direct.py`: caso `no_exog_quantiles`, 0.000e+00 en s1 y s2 (sin exog no se parte la llamada). Caso `exog_point`: la referencia es una llamada directa en lote de dos series de igual longitud y el adaptador hará una llamada por serie, así que la diferencia deja de ser exactamente 0 y será del orden del ruido de lote (3.8e-06 a 7.6e-06 en E3 y V4) [inferido]. No es una regresión; la comprobación de corrección de P1 es E3/E4.
- `check_covariates_rollout.py` (pasa por `FoundationModel`): E1 `past-only vs no exog` 0.000e+00 (sin cambio; el aviso queda oculto por el `warnings.simplefilter("ignore")` del script); E2 0.000e+00; E3 `s1 joint (same length) vs alone` y `s1 joint (s2 shorter) vs alone` 0.000e+00, y `s1 joint (s2 shorter) vs no exog` distinto de 0 (alrededor de 1.347); E4 `s1 joint (s2 exog with NaN) vs alone` 0.000e+00 [inferido: misma entrada y una llamada idéntica a la de la serie sola, y el backend es determinista, `repeat call max_abs_diff=0.000e+00` en `compare_direct.log`]; criterio mínimo del informe: menos de 1e-5. E5: `steps=700 exog=False` sigue dando 700 filas finitas; `steps=1344 exog=False`, `steps=700 exog=True` y `steps=1344 exog=True` dan el `ValueError` de skforecast. Con la opción (a) de D29, E4 seguiría en torno a 4.2e-03.
- `check_quantile_grid.py`: `rejected through FoundationModel.predict: 0` con la opción (d); la línea `rejected by TSICL._get_quantile_indices: 8` no cambia (es el backend directo).
- `verification/check_quantile_exact_types.py`: sin cambios (índices 6, 13, 27, 28, 54, 55, 56, 57; q0.06 y q0.08 0.000e+00).
- `verification/verify_backend_direct.py`: V1 sigue en 0.000e+00 en las tres líneas (sin exog no se parte la llamada; con exog es una sola serie; los niveles 0.05, 0.5, 0.95 mantienen su índice); V0 y V2 a V8 son llamadas directas al backend y no cambian.
- `check_padding_option.py` (opcional): con `requires_equal_length_context=False` y exog, las dos diferencias pasan de 1.289 y 6.746e-01 a 0.000e+00 [inferido]; el bloque con `requires_equal_length_context=True` monkeyparchea el atributo y deja de ser representativo.

Backtest de la guía: no es necesario; ninguna propuesta cambia el caso de la guía (una serie, exog con valores futuros, `steps=24`, `context_length=500`, punto = cuantil 0.5) [verificado en código]. Si se quiere una comprobación de corrección de P1, ejecutar antes y después y comparar las predicciones guardadas (diferencia máxima esperada 0.0 y MAE idéntico):

```bash
$PY dev/foundation_adapters_review/evidence/common/backtest_guide.py --model-id taharnbl/TS-ICL --exog --out <dir_temporal>/tsicl_before.csv
$PY dev/foundation_adapters_review/evidence/common/backtest_guide.py --model-id taharnbl/TS-ICL --exog --out <dir_temporal>/tsicl_after.csv
```

## 5. Documentación y contexto de IA

Docstrings (`skforecast/foundation/`):

- `_adapter_base.py`: párrafo del atributo nuevo en el docstring de `_AdapterBase`.
- `_adapters.py`, `TSICLAdapter`: Notes (covariables solo de pasado ignoradas con tsicl 0.2.1 y aviso; una llamada por serie con exog; rejilla de cuantiles completa con (d) o lista de excepciones con (a), sin decir que el error sale de la librería; límite de 672 pasos con exog o múltiplos de 672; Python >=3.12 y torch <2.10), `context_length` (recorte a 4096 y aviso), `device` (aviso de caída a CPU), entrada del atributo nuevo en Attributes. En los otros 9 adaptadores, solo la entrada del atributo nuevo en Attributes.
- `_foundation_model.py`: propiedad `supports_past_only_covariates` (aprox. línea 396), docstring de `predict` (aprox. líneas 1104-1120).

Guía principal `docs/user_guides/foundation-forecasting-models.ipynb` (solo texto de TS-ICL, sin reejecutar; son celdas markdown):

- Tabla "Supported Foundation Models" (celda 11 en la revisión), fila TS-ICL: la columna "Max context length" dice "No hard limit" y el backend recorta a 4096 (TSI-F10; `SKILL.md` línea 173 ya dice 4096) [verificado en código]; la columna "Max horizon" dice "No hard limit, set via steps" y debe indicar el límite de 672 pasos con exog futura (o múltiplos de 672) con tsicl 0.2.1; en "Install command", añadir la nota de Python >=3.12 y torch <2.10.
- Introducción de la sección "EDF-Lab TS-ICL" (celda 96 en la revisión, "A ForecasterFoundation is created using EDF-Lab TS-ICL model"): una o dos frases con los requisitos de Python y torch, que las covariables solo de pasado se ignoran con aviso y que cada serie con exog se predice en su propia llamada. La salida de las celdas 97 a 100 (una serie, exog con futuro, `steps=24`) no cambia [inferido], así que no hay que reejecutar; la reejecución única es de `10_docs_ci.md`.

Guía de series heterogéneas `docs/user_guides/foundation-forecasting-with-heterogeneous-series.ipynb` (solo texto markdown, sin reejecutar):

- Celda 2 en la revisión (tabla de preguntas, fila "Does batching series together change their forecasts?"): sacar TS-ICL de "Not for Chronos-2, TS-ICL and TabICL" y describirlo según D29; con (b): con exog cada serie se predice en su propia llamada (idéntico a predecirla sola); sin exog, en lote con longitudes distintas 0.000e+00 [probado, `check_padding_option.log`]; con igual longitud sin exog no se ha medido [inferido: ruido de float32 posible].
- Celda 21 en la revisión (tabla de capacidades y texto por adaptador): fila TS-ICL, "Past-only covariates" pasa de "Yes" a "No (ignored with a warning)" y "Batch with different exog columns" de "Grouped" a "One call per series with exog"; en el texto "TS-ICL. As with Chronos-2, ...", quitar "Past-only covariates are supported" y explicar la partición.
- Celda 18 en la revisión (requisito de mismas columnas por lote) sigue siendo cierta; no tocar.

Skill y contexto de IA:

- `skills/foundation-forecasting/SKILL.md`: línea 46 (fila de la tabla de problemas: cuantiles de TS-ICL, igual que la 159); línea 62 (comentario de `pip install tsicl` con Python >=3.12 y torch <2.10); línea 130 (TS-ICL sale de la lista de adaptadores que usan covariables solo de pasado y pasa a la de los que las ignoran con aviso; matizar "so the forecast of a series never depends on the exog of the others"); línea 159 (rejilla 0.01: correcta con (d), con excepciones con (a)); línea 173 (fila de la tabla: "Past & future covariates" pasa a covariables futuras, y límite de horizonte); línea 281 (error frecuente de cuantiles: igual que la 159); línea 287 (error frecuente de predecir sin exog: TS-ICL pasa a la lista de los que ignoran las columnas históricas); añadir un error frecuente con el horizonte de más de 672 pasos con exog.
- `skills/foundation-forecasting/references/adapter-parameters.md`: sección `TSICLAdapter` (líneas 148-165: `allow_exog` "past and future covariates" en la 151, cuantiles en la 152, fila `context_length` con el recorte a 4096, fila `device` (línea 160) con el aviso, requisitos de Python y torch, horizonte); línea 242 (párrafo de los atributos que lee `FoundationModel`: añadir el atributo nuevo); línea 249 (cabecera de la tabla: TS-ICL pasa de la columna `supports_past_only_covariates=True` a la de `False`).
- `tools/ai/llms-base.txt` (líneas en d9b8f4f; aprox. +2 en `origin/0.27.x`): línea 458 (lista de backends: requisitos de tsicl), línea 496 (fila `TSICLAdapter`: "Yes (past & future covariates)" y cuantiles), línea 530 (TS-ICL pasa a la lista de los que ignoran las covariables solo de pasado), línea 531 (independencia entre series: con el atributo nuevo, TS-ICL predice cada serie con exog por separado). La línea 13 (versiones de Python de skforecast) no cambia; el requisito va junto a TS-ICL.
- Regenerar con la skill `ai-context-sync` (no editar a mano `AGENTS.md`, `.github/copilot-instructions.md`, `llms-full.txt` ni `docs/llms*.txt`).
- La corrección general de la garantía de independencia para otros backends (T0, TimesFM 3, Chronos con `cross_learning`) es de sus planes; aquí solo se corrige lo que afecta a TS-ICL y al atributo nuevo.

Nota de release para `docs/releases/releases.md`, sección 0.27.0 (con la skill `release-note`; el texto final depende de D29 a D33; `#PR` es el número del pull request):

```markdown
**Fixed**

+ <code>[FoundationModel]</code>, <code>[ForecasterFoundation]</code> and <code>[backtesting_foundation]</code> with TS-ICL (`'taharnbl/TS-ICL'`) and exogenous variables: when the series of a call reached the backend with different lengths (any series shorter than `context_length`), tsicl 0.2.1 silently dropped the future covariates of every series, and NaN values in the exog of one series changed the forecast of the others. Each series with exogenous variables is now forecast in its own backend call, so its forecast is the same as when it is predicted alone. ([#PR](https://github.com/skforecast/skforecast/pull/PR))
+ <code>[TSICLAdapter]</code> rejected 8 quantile levels of its 0.01 grid (`0.07`, `0.14`, `0.28`, `0.29`, `0.55`, `0.56`, `0.57`, `0.58`) with a `ValueError` from tsicl. Every level of the grid now works. ([#PR](https://github.com/skforecast/skforecast/pull/PR))
+ <code>[TSICLAdapter]</code> raised an unclear `RuntimeError` or `ValueError` from tsicl 0.2.1 when forecasting more than 672 steps with future exogenous variables, or a multiple of 672 steps. It now raises a `ValueError` that explains the limit. ([#PR](https://github.com/skforecast/skforecast/pull/PR))

**Changed**

+ <code>[TSICLAdapter]</code> declares `supports_past_only_covariates=False`: tsicl 0.2.1 ignores historical exog columns without future values (and drops them when other columns have future values), so they now raise an `IgnoredArgumentWarning`. The forecasts do not change. ([#PR](https://github.com/skforecast/skforecast/pull/PR))
+ <code>[TSICLAdapter]</code> warns when an MPS device falls back to CPU and when `context_length` is larger than 4096, the maximum context of the backend. The documentation states that tsicl requires Python >= 3.12 and torch < 2.10. ([#PR](https://github.com/skforecast/skforecast/pull/PR))
```

## 6. Decisiones pendientes

| Id | Decisión | Opciones | Recomendación | Impacto en el plan | Estado |
|---|---|---|---|---|---|
| D29 | Partición de llamadas de TS-ICL con exog (TSI-P1) | (a) agrupar por firma de exog y longitud tras el recorte (`min(len, context_length, 4096)`); TSI-F5 persiste; (b) una serie por llamada cuando hay exog; (c) relleno con `requires_equal_length_context` (descartada por TSI-F6); (d) bucle en el adaptador (contrario a `.claude/rules/foundation.md`) | (b), revisable cuando upstream cierre #8 y #10 | Nombre y semántica del atributo nuevo, lógica en `FoundationModel.predict`, tests de `test_predict.py`, invariantes del smoke y texto de la guía heterogénea (celdas 2 y 21) | pendiente |
| D30 | Covariables solo de pasado en TS-ICL (TSI-P2) | (a) `False` ahora y volver a `True` cuando se publiquen #4 y #6; (b) pasar `allow_covar_forecast=True` (no cubre el caso mixto, arrastra #21 y #9); (c) solo documentar | (a) | Atributo, `test_check_exog_columns.py`, `test_get_model_info.py`, docs y guía heterogénea (celda 21); con (b), además `contracts.py` y el fake | pendiente |
| D31 | Niveles de cuantil de TS-ICL (TSI-P3) | (a) filtrar los 8 niveles de `SUPPORTED_QUANTILES` y validar; (b) mantener y documentar; (c) reportar upstream; (d) traducir a `Decimal` exacto en el adaptador | (d) + (c); (a) si no se acepta depender del detalle interno | Código de `TSICLAdapter.predict` o de `SUPPORTED_QUANTILES`, fake estricto, `test_get_model_info.py` (solo con (a)), textos de cuantiles | pendiente |
| D32 | Horizonte de TS-ICL por encima de 672 pasos (TSI-P4) | (a) guard y documentación ahora; (b) esperar la release con el PR #1 y subir el mínimo; (c) nada | (a) ahora y (b) cuando salga la release | Guard y su test, smoke opcional, textos de horizonte; con (b), `backend_package` y `adapters.toml` en otro PR | pendiente |
| D33 | `context_length > 4096` en TS-ICL (TSI-P6) | (a) aviso; (b) error; (c) solo docstring | docstring + aviso | Aviso y su test, docstrings, fila de la tabla de la guía | pendiente |

Relacionadas, que se deciden en `02_common.md`: D-Q (tipo de los niveles de cuantil) y D-VER (helper común de versión), que solo cambian qué tipo de nivel llega al adaptador en TSI-P3 y cómo se lee la versión en TSI-P4.

## 7. Criterios de aceptación

- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_TSICLAdapter.py -x -q`: todo en verde, con los tests nuevos de P1 a P6 y P8 (más de los 54 actuales).
- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_predict.py skforecast/foundation/tests/tests_foundation_models/test_AdapterBase.py skforecast/foundation/tests/tests_foundation_models/test_check_exog_columns.py skforecast/foundation/tests/tests_foundation_models/test_get_model_info.py -x -q`: en verde.
- [ ] Una vez, al terminar: `pytest skforecast/foundation/tests/tests_foundation_models/ -q` y `pytest skforecast/foundation/tests/tests_forecaster_foundation/ -q` en verde (cambia el contrato base, así que todos los adaptadores deben definir el atributo nuevo).
- [ ] `~/.venvs/fm-review-tsicl/bin/python -m pytest tools/foundation_compat/tests/test_contract.py --backend tsicl -q -p no:cacheprovider`: 3 passed.
- [ ] `~/.venvs/fm-review-tsicl/bin/python -m pytest tools/foundation_compat/tests/test_smoke.py --backend tsicl -q -p no:cacheprovider`: los 6 actuales más las invariantes nuevas, en verde.
- [ ] `python tools/foundation_compat/run.py --adapter tsicl`: en verde.
- [ ] `check_covariates_rollout.py`: E3 (las dos líneas "vs alone") y E4 dan 0.000e+00 con D29 (b) (como mínimo, menos de 1e-5); E1 y E2 siguen en 0.000e+00; E5 da el `ValueError` de skforecast en los tres casos que hoy fallan y 700 filas finitas en `steps=700 exog=False`.
- [ ] `check_quantile_grid.py`: `rejected through FoundationModel.predict: 0` (con D31 (d)); con (a), ningún nivel de la lista declarada rechazado.
- [ ] `verify_backend_direct.py`: V1 sigue en 0.000e+00 en sus tres líneas. `compare_direct.py`: `no_exog_quantiles` en 0.000e+00; `exog_point` solo con ruido de lote (del orden de 1e-5 o menos).
- [ ] Sin `IgnoredArgumentWarning` nuevo en el caso de la guía (una serie con exog pasada y futura).
- [ ] Ficheros de IA regenerados con `ai-context-sync` y el check de contexto de IA en verde.
- [ ] Entrada en `docs/releases/releases.md` (skill `release-note`).
- [ ] Skill `verify` en verde antes del commit.
- [ ] Issue abierta upstream en `EDF-Lab/ts-icl` sobre la comparación exacta de floats de `quantile_levels` (si D31 incluye (c)).

## 8. Fuera de alcance

- Medidas de velocidad o de precisión entre versiones, variantes o tamaños de lote (incluido el coste de hacer una llamada por serie con exog): fuera de alcance para skforecast.
- Volver a `supports_past_only_covariates=True` cuando upstream publique #4 y #6 (D30 (a), PR posterior), y volver a agrupar en lote cuando cierre #8 y #10 (D29).
- Subir el mínimo de `tsicl` y retirar el guard de 672 pasos cuando se publique el PR #1 (D32 (b), PR posterior; el arreglo de main no se ha ejecutado).
- Exponer `batch_size`, `allow_auto_complete`, `point_estimator` o `allow_covar_forecast` (TSI-F15; este último solo si D30 elige (b)), y el `hf_repo_id` del PR #2 upstream.
- `taharnbl/TS-ICL-v2` y otros ids aceptados por prefijo que cargan otros pesos (X-3, TSI-F11): tabla central de ids no soportados o documentación en `02_common.md` (D-REG).
- Contexto todo NaN (X-1), conversión de los niveles de cuantil a `float` (X-2, D-Q), validación central contra `SUPPORTED_QUANTILES` y su tolerancia, y helper común de versión (D-VER): en `02_common.md`.
- Helper común de `stacklevel` (X-6): no forma parte de este PR.
- Reejecución de la guía principal: `10_docs_ci.md`. Las demás invariantes genéricas del smoke (XC-7 de `evidence/cross_cutting.md`) quedan fuera de esta serie de PR (`10_docs_ci.md`, sección 8); si se quieren, van en un PR aparte `chore/foundation-compat-checks` (PR-12 de `evidence/cross_cutting.md`), después de 01 y 03. La clave `python` opcional por entrada en `adapters.toml` (XC-12 de `evidence/cross_cutting.md`) queda fuera de esta serie de PR (`10_docs_ci.md`, sección 8): el workflow del CI de compatibilidad y `run.py` fijan Python 3.12, la única versión que sirve a uni2ts 2.0 y a tsicl [verificado en código].
- Texto de la licencia: TSI-F12 no exige cambios.
- Comportamiento real en CUDA y MPS (sin hardware en la revisión) y la pregunta de si el modelo se entrenó con covariables solo de pasado (issue #7; el paper no se pudo consultar).
