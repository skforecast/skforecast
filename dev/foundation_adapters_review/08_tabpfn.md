# 08. TabPFN-TS (tabpfn-time-series)

- **Rama propuesta**: `fix/tabpfn-alignment`, creada desde `origin/0.27.x`. El pull request va contra `0.27.x`.
- **Prioridad**: media. El adaptador funciona con la última versión del backend; no hay rotura [probado: `evidence/tabpfn/ci_baseline.log`].
- **Adaptador**: `TabPFNAdapter`.
- **Ficheros afectados**:
  - `skforecast/foundation/_adapters.py`: clase `TabPFNAdapter` (docstrings, `predict`, `_get_future_timestamps`; según las decisiones, también `__init__`, `get_params`, `set_params`, `_load_model` y los atributos de clase) y los imports `contextlib` e `io` del principio del módulo (aprox. líneas 13-14 en d9b8f4f), que quedan sin uso.
  - `skforecast/foundation/_utils.py`: aquí solo se usan los helpers que crea 07; si se toma D19 (B), se cambia además el comentario de `_MODEL_LICENSES`.
  - `skforecast/foundation/tests/tests_foundation_models/fixtures_adapters.py` (`FakeTabPFNTSPipeline`).
  - `skforecast/foundation/tests/tests_foundation_models/test_TabPFNAdapter.py`.
  - Según las decisiones: `skforecast/foundation/_foundation_model.py` (solo docstring), `skforecast/foundation/tests/tests_foundation_models/test_get_model_info.py`, `tools/foundation_compat/adapters.toml` y `tools/foundation_compat/tests/contracts.py`.
  - Documentación: `skills/foundation-forecasting/references/adapter-parameters.md`, `skills/foundation-forecasting/SKILL.md` (solo con D18 B o D19 B), el texto de la celda 11 de `docs/user_guides/foundation-forecasting-models.ipynb` (solo con D19 B) y `docs/releases/releases.md`.
- **Depende de**:
  - `07_tabicl.md`, que crea en `skforecast/foundation/_utils.py` el helper común que reemite los warnings ocultos por la redirección de stderr y el arreglo común de los timestamps sintéticos sin `pd.Timedelta(days=n)`. Este PR solo los aplica a `TabPFNAdapter`.
  - `02_common.md`, solo si se elige D19 (B) junto con la comprobación de versión en tiempo de ejecución (helper opcional de D-VER).
- **Bloquea a**: `10_docs_ci.md`, que re-ejecuta una sola vez la guía (incluidas las celdas de TabPFN-TS, TPF-P4) y añade la caché de `~/.cache/tabpfn` al CI (TPF-P7).
- **Cierra issue**: ninguna conocida. En el repositorio upstream no hay issue sobre desactivar tqdm ni sobre estos puntos [documentado: `evidence/tabpfn/upstream_issues.log`]; no se ha encontrado una issue de skforecast asociada [inferido].

## 1. Estado en la revisión (2026-11-06)

**Versiones**

| Componente | Probado antes (CI) | Última versión en la revisión | Requisitos |
|---|---|---|---|
| `tabpfn-time-series` | 1.4.0 | 1.4.0 (2026-10-28). La anterior es 1.3.0 (2026-09-17) | `requires_python>=3.10`; 1.4.0 exige `tabpfn>=9.2.0` y 1.3.0 exige `tabpfn>=9.0.0` [documentado: `evidence/tabpfn/pypi_tabpfn-time-series.log`, `evidence/tabpfn/verification/verification.md`] |
| `tabpfn` | 9.2.0 | 9.2.0 (2026-10-12). Antes: 9.1.0 (2026-10-02) y 9.0.0 (2026-09-15) | `requires_python>=3.10`, `torch>=2.5` [documentado: `evidence/tabpfn/pypi_tabpfn.log`] |
| `tabpfn-client` | 0.7.1 | 0.7.1 (0.7.0 y 0.7.1 del 2026-10-28 y 2026-10-30) | Sus cambios son internos (jobs en segundo plano, `thinking_effort`) y sigue el alias `v3.5_default`; no afectan al adaptador, que no llama a `tabpfn-client` directamente [documentado: CHANGELOG del sdist, recogido en `evidence/tabpfn/verification/verification.md`] |
| `tabpfn-common-utils` | 0.2.23 | 0.2.23 | [documentado: `evidence/tabpfn/pypi_tabpfn-common-utils.log`] |

Las versiones exactas del entorno de la revisión (Python 3.12.3, torch 2.15.0+cpu, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.0, gluonts 0.17.0, fev 0.10.0, huggingface-hub 1.33.0) están en `evidence/tabpfn/requirements-freeze.txt` y `evidence/tabpfn/env.log` [probado].

**Checkpoints en el Hub**

- `Prior-Labs/tabpfn_3_5` es el que usa el adaptador (`weights_repo_id`):
  - sha `06bf2ba35c80a92a3b9abb436b99cf49e7a0365e`, creado y modificado el 2026-09-09, no gated, licencia `tabpfn-3-5-license-v1.0`.
  - Ficheros: `tabpfn-v3.5-20260909.safetensors` (876027932 bytes, el de regresión), `tabpfn-v3.5-20260909_multiclass.safetensors` (876028348 bytes) y `tabpfn-v3.5-fast-20260909.safetensors` (334181708 bytes).
  - Sin repos más nuevos de Prior-Labs para series temporales (el listado completo del autor incluye `tabpfn_2_5`, `tabpfn_2_6`, `tabpfn_3`, `TabPFN-v2-clf`, `TabPFN-v2-reg` y `demo`).

  [probado: `evidence/tabpfn/hub_info.log`, `evidence/tabpfn/verification/hub_rerun.log`]
- `Prior-Labs/TabPFN-v2-reg` es público, no gated y no pide `TABPFN_TOKEN` con tabpfn 9.2.0. Datos: sha `02a44705ff7790b50cded404bc59adb94da456a1`, última modificación 2026-10-28, licencia `priorlabs-1-1`. Su fichero `tabpfn-v2-regressor.ckpt` (unos 44 MB) es el checkpoint real que la verificación usa para las pruebas sin token. No es el checkpoint por defecto del adaptador [probado: `evidence/tabpfn/hub_info.log`, `evidence/tabpfn/verification/real_weights_v2_check.log`].

**Línea base del CI**

- Run 42212742430 del workflow Foundation Backends Compatibility (rama 0.27.x, commit `09433c3`, 2026-11-06), job `tabpfn` 127875169000. Entorno: Python 3.12.16, ubuntu-26.04, tabpfn-time-series 1.4.0, tabpfn 9.2.0, torch 2.15.0+cpu.
- Resultados: contrato 5 passed, smoke 6 passed, `latest_on_pypi` 1.4.0, `held_back` false.
- La caché de `~/.cache/huggingface` restaurada ocupa 8332 B y la guardada 7547 B.
- El smoke muestra dos warnings: el `LicenseWarning` de skforecast y `UserWarning: Running TabPFN locally on CPU; inference will be slow for long contexts...` (`tabpfn_time_series/predictor.py:105`).

[probado: `evidence/tabpfn/ci_baseline.log`, `evidence/tabpfn/verification/verification.md`]

**Entorno de la revisión**

- Con 1.4.0/9.2.0 pasan los 58 tests unitarios de `test_TabPFNAdapter.py` y los 5 del contrato [probado: `evidence/tabpfn/unit_tests.log`, `evidence/tabpfn/contract.log`].
- El smoke falla con `TabPFNLicenseError` (1 failed, 1 passed en `smoke.log`, ejecutado con `-x`; 5 failed, 1 passed en la re-ejecución completa de `verification/contract_rerun.log`, con 1.3.0/9.0.0 y con 1.4.0/9.2.0) porque **en la revisión no había `TABPFN_TOKEN`**: los pesos TabPFN-3.5 no se descargan [probado: `evidence/tabpfn/smoke.log`, `evidence/tabpfn/verification/contract_rerun.log`].
- No hay GPU: CUDA y MPS solo se han leído en código.

**Antes de empezar.** Este plan puede quedarse desfasado. Desde la raíz del repositorio:

```bash
python dev/foundation_adapters_review/evidence/common/pypi_info.py tabpfn-time-series
python dev/foundation_adapters_review/evidence/common/pypi_info.py tabpfn
python dev/foundation_adapters_review/evidence/common/pypi_info.py tabpfn-client
python dev/foundation_adapters_review/evidence/common/hub_info.py --repo Prior-Labs/tabpfn_3_5
python dev/foundation_adapters_review/evidence/common/hub_info.py --repo Prior-Labs/TabPFN-v2-reg
python dev/foundation_adapters_review/evidence/common/hub_info.py Prior-Labs
# Recrear el entorno (usar ">=1.4" si se decide D19 B). Ojo: el script
# sobrescribe evidence/tabpfn/requirements-freeze.txt; no lo incluyas en el PR.
bash dev/foundation_adapters_review/evidence/common/make_env.sh tabpfn "tabpfn-time-series>=1.3" 3.12
# Último resultado del job tabpfn en el CI de compatibilidad
gh run list --workflow foundation-backends.yml --branch 0.27.x --limit 3
```

Comprueba también que el PR de `07_tabicl.md` esté fusionado en `0.27.x`, porque este PR usa sus helpers, y anota el nombre y la firma que fijó.

Si ha salido una versión nueva de `tabpfn-time-series` o de `tabpfn`:

1. Lee el CHANGELOG desde 1.4.0 (o 9.2.0).
2. Comprueba si cambian:
   - la firma de `TabPFNTSPipeline.__init__` y de `predict_df` (corre el contrato);
   - `TABPFN_TS_DEFAULT_FEATURES`;
   - la lógica de `compile` en `TabPFNTSPipeline.__init__`;
   - la versión de pesos fijada en `defaults.py` (`TABPFN_MODEL_VERSION`). Si cambian los pesos, revisa también `weights_repo_id`, `_MODEL_LICENSES` y `tools/check_foundation_models_metadata.py`;
   - si tqdm se puede desactivar upstream. Si se puede, es preferible a la redirección de stderr de TPF-P1.
3. Repite las comprobaciones de la sección 4 (`compare_direct.py`, `feature_level_check.py`, `real_weights_v2_check.py`).
4. Actualiza esta sección y la recomendación de D19.

## 2. Hallazgos

- **TPF-F1** [probado] Con tabpfn-time-series 1.4.0 y tabpfn 9.2.0:
  - pasan el contrato (5) y los unitarios (58);
  - el smoke falla en la revisión solo por falta de `TABPFN_TOKEN` (`TabPFNLicenseError`);
  - en el CI, con token, el smoke pasó con las mismas versiones (6 passed).

  Evidencia: `evidence/tabpfn/contract.log`, `evidence/tabpfn/unit_tests.log`, `evidence/tabpfn/smoke.log`, `evidence/tabpfn/ci_baseline.log`, `evidence/tabpfn/verification/contract_rerun.log`, `evidence/tabpfn/verification/unit_tests_rerun.log`.
- **TPF-F2 (corregido)** [probado] La traducción de entradas y salidas es exacta.
  - La comparación original (6 casos: punto; cuantiles [0.1, 0.5, 0.9]; cuantiles fuera de la rejilla por defecto [0.025, 0.975]; 2 series de distinta longitud con NaN; exog conocida heterogénea en cuantiles y en punto) da 0.0, pero no era discriminante: un TabPFN-3.5 diminuto con pesos aleatorios (ModelSpecs) da una predicción constante en el horizonte (std 0) e insensible a la exog (exog+3 y sin exog: diferencia 0.0).
  - Queda probada a nivel de features: los frames que el adaptador pasa a `predict_df`, featurizados con el preprocesado del backend (`_featurize_context_future`), son idénticos a los de una entrada directa con 2 series, NaN y exog heterogénea.
  - Queda probada también con el checkpoint público `Prior-Labs/TabPFN-v2-reg`: adaptador frente a llamada directa 0.0 (con y sin exog); un cambio de +3 en la exog futura mueve la predicción 13.3456; el control con la exog desplazada 1 paso difiere 7.0137.
  - No está probada con TabPFN-3.5.

  Evidencia: `evidence/tabpfn/compare_direct.log`, `evidence/tabpfn/verification/rerun_compare_direct.log`, `evidence/tabpfn/verification/nontrivial_check.log`, `evidence/tabpfn/verification/feature_level_check.log`, `evidence/tabpfn/verification/real_weights_v2_check.log`.
- **TPF-F3** [verificado en código] Entre v1.3.0 y v1.4.0, la API que usa el adaptador solo tiene adiciones:
  - `TabPFNTSPipeline.__init__` añade `num_workers_per_gpu=1` y `compile=True`;
  - `predict_df`, `data_preparation.py`, `ts_dataframe.py`, `cpu_worker.py` y `model_adapters/base.py` no cambian;
  - `TabPFNMode.LOCAL/CLIENT` y `tabpfn_output_selection` en {mean, median, mode} siguen igual.

  Sí cambian `worker/model_adapters/tabpfn_adapter.py` (soporte de ModelSpecs) y `gpu_worker.py` (pools por GPU y paso de kwargs, ver TPF-V1), sin efecto en la API que usa el adaptador. Evidencia: `evidence/tabpfn/capabilities_check.log` (firma instalada) y `evidence/tabpfn/old_backend_1.3.0.log` (firma de 1.3.0).
- **TPF-F4** [verificado en código] En 1.4.0 los features temporales por defecto son `[KnownCovariateLagFeature, RunningIndexFeature, CalendarFeature, AutoSeasonalFeature]` (`TABPFN_TS_DEFAULT_FEATURES`, PR upstream #195).
  - El docstring de `temporal_features` del adaptador (aprox. líneas 2789-2793 en d9b8f4f) enumera los 3 antiguos; el docstring upstream de `temporal_features` en v1.4.0 también está desfasado.
  - Con exog aparecen las columnas `x1_lag_1`, `x1_lag_24` y `x1_mean_24`.
  - Que con exog la misma versión de skforecast prediga distinto con 1.3 y con 1.4 es [inferido]: cambian los features de entrada, no se ha medido.

  Evidencia: `evidence/tabpfn/compare_direct.log`, `evidence/tabpfn/old_backend_1.3.0.log`.
- **TPF-F5** [verificado en código] En CUDA, 1.4.0 compila por defecto. Lo hace en `TabPFNTSPipeline.__init__` con `if compile and not is_client and torch.cuda.is_available(): tabpfn_model_config = {**tabpfn_model_config, "enable_torch_compile": True}`.
  - El adaptador no pasa `compile` (`_load_model`, aprox. líneas 3229-3241 en d9b8f4f), así que se sobrescribe un `tabpfn_model_config={"enable_torch_compile": False}` del usuario.
  - Según el docstring upstream, la compilación tarda ~1 min por proceso y las predicciones pueden variar por redondeo [documentado].
  - Solo se evita con `model=TabPFNTSPipeline(compile=False)`, una vía documentada como solo para tests.
  - No probado: no hay GPU.

  Evidencia: `evidence/tabpfn/capabilities_check.log` (sección 3).
- **TPF-F6 (corregido)** [probado] Con `show_progress=False` (el defecto), el `contextlib.redirect_stderr(io.StringIO())` que oculta tqdm oculta también la visualización de los warnings del backend, por ejemplo `"Some covariate values in future_tsdf are missing (NaN)"` (`tabpfn_time_series/pipeline.py` v1.4.0, aprox. línea 157).
  - El warning se sigue emitiendo, y `pytest.warns` y `catch_warnings(record=True)` lo capturan. Lo que se pierde es su visualización, porque `warnings.showwarning` escribe en el `sys.stderr` redirigido; para el usuario típico (script o notebook) el aviso no se ve.
  - Con `show_progress=True` el aviso sí llega a stderr.
  - Upstream no ofrece desactivar tqdm (`cpu_worker.py` y `gpu_worker.py` de v1.4.0 sin `disable`, sin issue upstream), así que la redirección sigue siendo necesaria.

  Evidencia: `evidence/tabpfn/capabilities_check.log` (sección 1), `evidence/tabpfn/verification/rerun_capabilities_check.log`, `evidence/tabpfn/verification/test_p1_pytest_warns.log`, `evidence/tabpfn/upstream_issues.log`.
- **TPF-F7** [probado] El mínimo actual `tabpfn-time-series>=1.3` funciona:
  - con 1.3.0 y tabpfn 9.0.0 (el par más bajo permitido) pasan el contrato (5) y los unitarios (58);
  - la firma de 1.3.0 no tiene `compile` ni `num_workers_per_gpu`, pero sí `max_featurize_rows`, y usa 3 features por defecto;
  - ambas versiones fijan TabPFN-3.5 (`TABPFN_MODEL_VERSION` `"v3.5"` en LOCAL, `v3.5_default` en CLIENT).

  Evidencia: `evidence/tabpfn/old_backend_1.3.0.log`, `evidence/tabpfn/verification/old_backend_1.3.0_rerun.log`, `evidence/tabpfn/verification/contract_rerun.log`, `evidence/tabpfn/verification/old_backend_env.log` (`evidence/tabpfn/old_backend_1.3.0_env.log` está vacío).
- **TPF-F8** [probado] `weights_repo_id = "Prior-Labs/tabpfn_3_5"` es correcto (sha 06bf2ba35c80, 2026-09-09, no gated).
  - El mismo repo trae una variante fast (334 MB) que se puede seleccionar con `tabpfn_model_config["model_path"]`.
  - tabpfn 9.2.0 resuelve `V3_5` y `V3_5_FAST` a esos ficheros (`model_loading.py`, `get_v3_5` y `get_v3_5_fast`, aprox. líneas 213-232 de v9.2.0) [verificado en código].

  Evidencia: `evidence/tabpfn/hub_info.log`, `evidence/tabpfn/verification/hub_rerun.log`.
- **TPF-F9** [probado] La licencia `tabpfn-3-5-license-v1.0` (no comercial y no productiva; secciones 1.c, 2.b y 2.d del LICENSE, sha256 `9570fdd687bb...`) coincide con la entrada `"priorlabs/tabpfn"` de `_MODEL_LICENSES` (`commercial_use_restricted=True`). `tools/check_foundation_models_metadata.py` da OK para `priorlabs/tabpfn-ts`. Evidencia: `evidence/tabpfn/metadata_check.log`, `evidence/tabpfn/verification/metadata_check_rerun.log`.
- **TPF-F10** [verificado en código] `requires_provider_auth=True` y `requires_hf_auth=False` son veraces:
  - tabpfn 9.2.0 llama a `ensure_license_accepted` en `_download_model` (`model_loading.py`, aprox. líneas 557-570) antes de descargar `V3_5` y `V3_5_FAST`, lo que exige `TABPFN_TOKEN` o login en el navegador;
  - el repo del Hub no está gated.

  Evidencia: `evidence/tabpfn/smoke.log`, `evidence/tabpfn/hub_info.log`.
- **TPF-F11** [verificado en código] `weights_in_hf_cache=False` es veraz: tabpfn descarga con `hf_hub_download(local_dir=...)` (`_try_huggingface_downloads`, aprox. líneas 323-330 de `model_loading.py` v9.2.0) a su propia caché (`~/.cache/tabpfn` en Linux, o `$XDG_CACHE_HOME/tabpfn`, o `TABPFN_MODEL_CACHE_DIR`).
  - El workflow `.github/workflows/foundation-backends.yml` solo cachea `~/.cache/huggingface` (paso "Cache model weights", aprox. línea 94), que para tabpfn ocupa 8332 B.
  - Que cada ejecución vuelva a descargar los 876 MB es [inferido] (no hay log de descarga), aunque es coherente con el tamaño de la caché.

  Evidencia: `evidence/tabpfn/ci_baseline.log`.
- **TPF-F12** [probado] Cualquier `priorlabs/tabpfn*` (por ejemplo `priorlabs/tabpfn-v2`, `priorlabs/tabpfn-3.5` o `priorlabs/tabpfn-ts-3`) resuelve a `TabPFNAdapter` y carga TabPFN-3.5, porque `_resolve_adapter` usa el prefijo `priorlabs/tabpfn`.
  - Los ids del Hub (`Prior-Labs/tabpfn_3_5`, `Prior-Labs/tabpfn-ts`), `priorlabs/TabPFN-TS` y `PriorLabs/tabpfn-ts` dan `ValueError`.
  - Ya está documentado en el docstring y en `adapter-parameters.md`.
  - Es parte de X-3, que se trata en `02_common.md`; para TabPFN basta con la documentación.

  Evidencia: `evidence/tabpfn/resolve_adapter.log`, `evidence/tabpfn/verification/resolve_rerun.log`.
- **TPF-F13** [probado] El índice diario sintético para series con `RangeIndex` sigue siendo necesario: 1.4.0 acepta timestamps enteros, pero los convierte a `datetime64[ns]` como nanosegundos desde 1970, lo que deja sin sentido los features de calendario, la frecuencia y la estacionalidad. Evidencia: `evidence/tabpfn/probe_int_timestamps.log`, `evidence/tabpfn/verification/rerun_probe_int_timestamps.log`.
- **TPF-F14 (corregido)** [probado] Con numpy 2.5.4 y pandas 2.3.3 (la última 2.3.x; skforecast fija `pandas<3`), el constructor `pd.Timedelta(days=n)` emite un `DeprecationWarning` ("The 'generic' unit for NumPy timedelta is deprecated").
  - Lo emite el constructor (también `pd.Timedelta(days=5.0)`), no la suma.
  - `pd.Timedelta(n, unit="D")` y `pd.date_range("2000-01-01", periods=n + steps, freq="D")[n:]` no avisan y dan las mismas fechas (n=1, 5, 200).
  - Es un `DeprecationWarning`, oculto por defecto fuera de `__main__`.
  - Aparece en `TabPFNAdapter._get_future_timestamps` (aprox. línea 3305 en d9b8f4f) y sale 2 veces en los tests de `RangeIndex` de `test_TabPFNAdapter.py`.

  Evidencia: `evidence/tabpfn/unit_tests.log`, `evidence/tabpfn/verification/rerun_probe_int_timestamps.log`, `evidence/cross_cutting/probe_shared.log` (sección 5), `evidence/tabpfn/verification/verification.md` (n=1, 5, 200 y `unit="D"`).
- **TPF-F15 (corregido)** [probado] Son veraces los atributos de capacidad `supports_heterogeneous_covariates=True`, `requires_equal_length_context=False` y `supports_nan_in_series=True`.
  - La prueba original con pesos aleatorios no discriminaba. Con pesos v2, una serie sola o en lote con otra que tiene una covariable extra da 0.0, con los features de 1.4 y con los de 1.3.
  - El backend elimina las filas con target NaN. Si una serie tiene como mucho 1 valor válido, rellena con 0 sin avisar (`_handle_missing_values` de `pipeline.py` v1.4.0) [verificado en código]. FoundationModel ya rechaza las series enteramente NaN [probado: `evidence/tabpfn/capabilities_check.log`, sección 2].

  Evidencia: `evidence/tabpfn/heterogeneous_check.log`, `evidence/tabpfn/verification/real_weights_v2_check.log`.
- **TPF-F16** [probado] Las covariables `category` y `str` pasan por el adaptador y el pipeline sin error.
  - Con pesos v2 se usan con sentido. Con una covariable que fija un salto de nivel, la media de la predicción en los tramos bajo y alto es 9.91 y 39.93 con `category` o `str`, frente a 9.91 y 39.91 con la versión numérica 0/1 y a 10.08 y 10.12 sin exog.
  - El CHANGELOG de TabPFN 9.0.0 (#1245, #1240) dice que `category` se trata como categórica y que los strings se codifican [documentado].
  - El README de 1.4.0 dice que `KnownCovariateLagFeature` no crea columnas para covariables string o categóricas [documentado].
  - Con TabPFN-3.5 sigue sin verificar.

  Evidencia: `evidence/tabpfn/compare_direct.log`, `evidence/tabpfn/verification/categorical_v2_check.log`.
- **TPF-F17** [verificado en código] Las salidas de TabPFN-TS de la guía son anteriores a 1.3:
  - celda 80: skforecast 0.24.0, Python 3.13.14, fecha 2026-09-04;
  - celda 81: LicenseWarning con el texto antiguo "TabPFN License v1.0" y enlace a `Prior-Labs/tabpfn_3/blob/main/LICENSE`;
  - celda 83: MAE 186.416404;
  - celda 108: tabla comparativa (MAE de TabPFN-TS 186.4164).

  En cambio, siguen siendo correctos la tabla de la celda 11, el texto de la celda 77, la salida de `list_adapters` de la celda 13 (ya muestra `tabpfn_3_5`), `adapter-parameters.md` (línea 123) y `llms-base.txt` (líneas 493, 511, 514, 529-530).
- **TPF-F18** [verificado en código] La selección de dispositivo coincide con el docstring ("CUDA > MPS > CPU selected automatically by the library"):
  - en local con CUDA se usa `GPUParallelWorker`;
  - con MPS o CPU se usa `CPUParallelWorker`, y tabpfn elige el dispositivo con `device="auto"` (`predictor.py` v1.4.0 `_select_local_worker_class`, `utils.py` v9.2.0 `infer_devices`).
- **TPF-F19** [verificado en código] El adaptador no expone `num_workers_per_gpu` (1.4.0, solo CUDA local), `compile` (1.4.0) ni `max_featurize_rows` de `TabPFNTSPipeline`: `_load_model` construye un `kwargs` fijo (aprox. líneas 3229-3241 en d9b8f4f). `tabpfn_model_config` sí permite pasar `model_path` (incluido el checkpoint fast), `device`, `n_estimators` e `inference_precision`. Evidencia: `evidence/tabpfn/verification/rerun_capabilities_check.log` (sección 3).
- **TPF-F20** [verificado en código] Hay huecos en los tests:
  - `FakeTabPFNTSPipeline` (`fixtures_adapters.py`, aprox. línea 519) no acepta `num_workers_per_gpu`, `compile` ni `max_featurize_rows` (este último existe desde 1.3) y no emite warnings, así que ningún test cubre TPF-F6;
  - el contrato (`contracts.py`, entrada `"tabpfn"`) cubre las 4 llamadas del adaptador, pero el tipo de las columnas de cuantil solo lo comprueba el smoke;
  - `run.py` (aprox. líneas 310-321) marca el smoke como `skipped` si falta `TABPFN_TOKEN`, así que sin token nada comprueba la salida real.
- **TPF-V1** [verificado en código] En 1.3.0, con varias GPU y al menos tantas series como workers, `GPUParallelWorker.predict` llama a `_prediction_routine_per_gpu` sin `**kwargs`, así que se usan los cuantiles por defecto.
  - El adaptador hace `group[quantile_list]` en `predict` y daría `KeyError` con niveles fuera de la rejilla por defecto, por ejemplo `interval=[5, 95]`.
  - 1.4.0 lo arregla (#197; CHANGELOG: "Custom quantiles are no longer replaced by the defaults when predicting with several GPU workers").
  - No probado: no hay GPU. Evidencia: `evidence/tabpfn/verification/verification.md` (hallazgo nuevo V1).
- **TPF-V2** [documentado] El mínimo actual permite tabpfn 9.0.0, que no tiene dos arreglos de 9.1.0:
  - #1310: TabPFN-3.5 extrapolaba demasiado en features fuera de rango, sobre todo de tiempo como `year`; cambia predicciones;
  - #1215: cuantiles de `FullSupportBarDistribution`.

  1.4.0 exige `tabpfn>=9.2.0`. Evidencia: `evidence/tabpfn/verification/verification.md` (hallazgo nuevo V2), `evidence/tabpfn/pypi_tabpfn-time-series.log`.
- **TPF-V3** [probado] Hay una forma de tener un modelo real y sensible a covariables y timestamps sin token: el checkpoint público `Prior-Labs/TabPFN-v2-reg` (`tabpfn-v2-regressor.ckpt`, 44 MB), pasado como `tabpfn_model_config={"model_path": <ckpt>}`. tabpfn 9.2.0 no pide licencia para V2 (V2 no está en `_HF_REPOS` de `_download_model`). El ModelSpecs aleatorio no sirve para tests de alineación. Evidencia: `evidence/tabpfn/verification/real_weights_v2_check.log`, `evidence/tabpfn/verification/nontrivial_check.log`.
- **TPF-V4** [probado] El mismo `pd.Timestamp("2000-01-01") + pd.Timedelta(days=n)` está en `TabICLAdapter._get_future_timestamps` (aprox. línea 2631 en d9b8f4f). Lo arregla `07_tabicl.md`, que crea el arreglo común que aquí se aplica a TabPFN. Evidencia: `evidence/tabpfn/verification/rerun_probe_int_timestamps.log`, `evidence/cross_cutting/probe_shared.log`. Nota: en `evidence/tabpfn/verification/verification.md` la etiqueta V4 designa el hecho de que el fake no acepta `max_featurize_rows`; ese hecho se recoge aquí en TPF-F20.

Hallazgos transversales relacionados, de los que este plan no es responsable:
- X-5: `pd.Timedelta(days=n)`. Lo trata 07.
- X-7: patrón `redirect_stderr`. Lo trata 07.
- X-1: contexto todo NaN. Lo trata 02.
- X-2: cuantiles a float. Lo trata 02; no exige cambios en `TabPFNAdapter`, que busca las columnas con los mismos niveles que recibe [inferido].
- X-3: ids no soportados. Lo trata 02.
- X-4: un mínimo en `backend_package` no protege instalaciones existentes. Lo trata 02 (D-VER); afecta a TPF-P8.

**Descartado**
- "TabPFN-TS-3 sigue en Prior-Labs/tabpfn_3": no verificado. Upstream solo cita "TabPFN-TS-3 ship config" en el docstring de `max_context_length`, y los pesos que usa son `Prior-Labs/tabpfn_3_5` [verificado en código, docstring de `max_context_length` en tabpfn-time-series 1.4.0].
- Las igualdades obtenidas con el TabPFN-3.5 aleatorio (ModelSpecs) no sirven como prueba de alineación de exog ni de timestamps (ver TPF-F2, TPF-F15 y TPF-V3) [probado].
- Un test de TPF-P1 basado en `pytest.warns` no discrimina: ya pasa sin el arreglo (`evidence/tabpfn/verification/test_p1_pytest_warns.log`, 2 passed) [probado].
- La afirmación inicial de que el mínimo `>=1.3` no tiene riesgos se queda corta (ver TPF-V1 y TPF-V2) [verificado en código y documentado].
- Quedan abiertos, sin verificar:
  - el efecto real de las covariables `category` con TabPFN-3.5;
  - el comportamiento de `compile` en CUDA;
  - el modo `client`.

## 3. Cambios a implementar

### TPF-P1. Reemitir los warnings del backend ocultos por la redirección de stderr

- **Qué**:
  - En `TabPFNAdapter.predict` (aprox. líneas 3175-3185 en d9b8f4f), sustituye el bloque `_stderr_cm = contextlib.redirect_stderr(io.StringIO()) if not self.show_progress else contextlib.nullcontext()` por el context manager común que crea `07_tabicl.md` en `skforecast/foundation/_utils.py`. La revisión transversal propuso el nombre `_suppress_backend_progress()`; usa el nombre que fije 07.
  - Ese context manager graba los warnings con `warnings.catch_warnings(record=True)` dentro de la redirección y los reemite con `warnings.warn_explicit`, conservando `filename` y `lineno`, cuando stderr ya está restaurado.
  - Tras este cambio y el de 07, `contextlib` e `io` ya no se usan en `_adapters.py` (hoy solo aparecen en las líneas 2512-2514 de TabICL y 3176-3178 de TabPFN) [verificado en código]: elimina `import contextlib` e `import io` (aprox. líneas 13-14) si 07 no lo hizo.
  - Actualiza el docstring de `show_progress` (clase, aprox. líneas 2794-2797, e `__init__`, aprox. líneas 2951-2954) para decir que solo se oculta la barra y que los warnings del backend se siguen mostrando.

  Ejemplo ilustrativo (el nombre y la firma los fija 07):

  ```python
  with _suppress_backend_progress(self.show_progress):
      result_df = self._model.predict_df(
                      context_df = context_df,
                      future_df  = future_df,
                      quantiles  = tabpfn_quantiles,
                  )
  ```
- **Por qué**: TPF-F6, TPF-F20 (X-7).
- **Riesgo**: bajo. Hay cuatro matices:
  - `warn_explicit` sin `registry` no deduplica, así que en backtesting el aviso sale en cada fold, aunque respeta los filtros del usuario [inferido].
  - `catch_warnings` no es thread-safe [inferido].
  - Los warnings emitidos en procesos worker (loky) no pasan por la redirección [inferido].
  - El CI muestra otro warning del backend, `UserWarning: Running TabPFN locally on CPU; inference will be slow...` (`tabpfn_time_series/predictor.py:105`) [probado: `evidence/tabpfn/ci_baseline.log`]. Si se emite dentro de `predict_df`, el usuario en CPU pasará a verlo con `show_progress=False` [inferido]. Comprueba en el PR dónde se emite. No lo filtres en el adaptador: el usuario puede filtrarlo.
- **Rompe compatibilidad**: no. Solo cambia lo que se muestra; las predicciones no cambian.
- **Decisión**: ninguna.

### TPF-P3. Docstrings de `temporal_features` y de la versión fijada

- **Qué**:
  - En la docstring de la clase `TabPFNAdapter` (resumen, aprox. líneas 2753-2756, que enumera "running index, calendar features, automatically detected seasonal features", y parámetro `temporal_features`, aprox. líneas 2789-2793) y en la de `__init__` (aprox. líneas 2947-2950), deja de enumerar los features por defecto. Di que, con `None`, se usan los de la versión instalada de TabPFN-TS y que desde 1.4.0 incluyen `KnownCovariateLagFeature` (lags de las covariables conocidas).
  - Opcional, en la docstring de `weights_repo_id` (aprox. líneas 2849-2855, "pinned by tabpfn-time-series 1.3"), en el comentario de `_MODEL_LICENSES` en `_utils.py` (aprox. líneas 307-308) y en `adapter-parameters.md` (línea 123): di que tanto 1.3 como 1.4 fijan `Prior-Labs/tabpfn_3_5` (TPF-F7). Si se toma D19 (B), estos textos cambian según TPF-P8.
- **Por qué**: TPF-F4, TPF-F7.
- **Riesgo**: nulo (solo documentación).
- **Rompe compatibilidad**: no.
- **Decisión**: ninguna.

### TPF-P5. Timestamps sintéticos sin `pd.Timedelta(days=n)`

- **Qué**: en `TabPFNAdapter._get_future_timestamps` (aprox. línea 3305 en d9b8f4f), usa el arreglo común que introduce `07_tabicl.md` para `TabICLAdapter._get_future_timestamps`. Si 07 crea un helper en `_utils.py`, ambos adaptadores lo llaman y desaparece la duplicación. Forma equivalente sin aviso [probado]:

  ```python
  timestamps = pd.date_range("2000-01-01", periods=n + steps, freq="D")[n:]
  # alternativa mínima: pd.Timestamp("2000-01-01") + pd.Timedelta(n, unit="D")
  ```
- **Por qué**: TPF-F14, TPF-V4 (X-5).
- **Riesgo**: nulo. Genera las mismas fechas (n=1, 5, 200) y solo afecta al camino de `RangeIndex` [probado].
- **Rompe compatibilidad**: no. No cambia predicciones.
- **Decisión**: ninguna.

### TPF-P9. Cerrar los huecos de tests del adaptador

- **Qué**:
  - `FakeTabPFNTSPipeline.__init__` (`fixtures_adapters.py`, aprox. línea 534) acepta `**kwargs` y guarda todos los argumentos recibidos (por ejemplo en `self.init_kwargs`), para no divergir de la firma real (`num_workers_per_gpu`, `compile`, `max_featurize_rows`).
  - Añade al fake una forma de emitir un warning dentro de `predict_df`, por ejemplo un atributo `warning_message` que, si no es `None`, lanza `warnings.warn(self.warning_message, UserWarning)`.
  - Los tests unitarios inyectan el fake con `model=`, y entonces `_load_model` no lo construye [verificado en código]. Para comprobar los argumentos que `_load_model` pasa al constructor, sigue el patrón de `test_TabPFNAdapter_load_model_LicenseWarning_on_successful_load` (aprox. líneas 778-817 de `test_TabPFNAdapter.py`): un módulo falso en `sys.modules["tabpfn_time_series"]` con `TabPFNMode` y con `TabPFNTSPipeline = FakeTabPFNTSPipeline`.
  - Tests nuevos en `test_TabPFNAdapter.py` (ver sección 4).
  - Opcional, fuera de la suite unitaria: una comprobación adaptador frente a llamada directa que no use el ModelSpecs aleatorio. Puede comparar a nivel de features (como `feature_level_check.py`, que usa la función privada `_featurize_context_future` del backend) o usar el checkpoint público `Prior-Labs/TabPFN-v2-reg`, sin token (como `real_weights_v2_check.py`).
    - Tiene que poder ejecutarse sin `TABPFN_TOKEN`: `run.py` salta el nivel smoke completo cuando falta el secreto [verificado en código].
    - La ubicación se decide en el PR. Si se descarta, deja los dos scripts de evidencia como procedimiento manual documentado en la descripción del PR.
- **Por qué**: TPF-F20, TPF-F2, TPF-V3.
- **Riesgo**: bajo. La ruta ModelSpecs exige `tabpfn-time-series>=1.4` y no discrimina; no la uses [probado].
- **Rompe compatibilidad**: no.
- **Decisión**: ninguna.

### TPF-P2. Opciones nuevas de `TabPFNTSPipeline` (`compile`, `num_workers_per_gpu`, `max_featurize_rows`)

- **Qué** (según D17):
  - **(A)**:
    - Nuevo parámetro `pipeline_kwargs: dict[str, Any] | None = None` en `TabPFNAdapter.__init__`. Se guarda por referencia, como `tabpfn_model_config` (`clone` de scikit-learn exige no copiarlo; `test_clone.py` cubre ese patrón), lo devuelve `get_params` y lo valida `set_params` (dict o `None`).
    - Se añade al conjunto de parámetros que reinician `_model` en la llamada a `_apply_set_params` de `set_params` (aprox. líneas 3055-3065).
    - En `_load_model` se mezcla con `kwargs` solo si no está vacío.
    - Se rechazan con `ValueError` las claves que construye el adaptador (`max_context_length`, `tabpfn_mode`, `tabpfn_output_selection`, `tabpfn_model_config`, `temporal_features`), siguiendo el patrón de `TimesFM3Adapter._RESERVED_PREDICT_KWARGS` y `TimesFM3Adapter._validate_predict_kwargs` (aprox. líneas 1199 y 1255).
  - **(B)**: parámetro `compile: bool | None = None`, que se pasa a `TabPFNTSPipeline` solo si no es `None`, con el mismo tratamiento en `get_params`, `set_params` y el reinicio de `_model`.
  - **(C)**: solo documentar que en CUDA 1.4 compila por defecto y que la única vía es `model=TabPFNTSPipeline(compile=False)`.

  Ejemplo ilustrativo de (A) en `_load_model`:

  ```python
  if self.pipeline_kwargs:
      kwargs.update(self.pipeline_kwargs)
  self._model = TabPFNTSPipeline(**kwargs)
  ```
- **Por qué**: TPF-F5, TPF-F19.
- **Riesgo**:
  - No se puede verificar sin GPU.
  - Pasar `compile` a 1.3 da `TypeError` [verificado en código]; por eso solo se pasa cuando el usuario lo fija, y así se mantiene `>=1.3` si D19 queda en (A). Con (A), las claves válidas dependen de la versión instalada del backend; documéntalo.
- **Rompe compatibilidad**: no. Es un parámetro opcional nuevo y con el valor por defecto no cambia nada.
- **Decisión**: D17. Si sigue pendiente al abrir el PR, esta propuesta queda fuera (sección 8).

### TPF-P6. `supports_categorical_covariates`

- **Qué** (según D18):
  - (A) mantener `supports_categorical_covariates = False` (aprox. línea 2898);
  - (B) pasarlo a `True` y actualizar el docstring del atributo, `test_get_model_info.py` (tupla de `priorlabs/tabpfn-ts`, aprox. líneas 170-171) y los textos de la sección 5 que dicen que solo Chronos-2 y Timestral aceptan categóricas.
- **Por qué**: TPF-F16.
- **Riesgo**: con (B) se declararía una capacidad no verificada con el checkpoint por defecto (TabPFN-3.5). El impacto es bajo: el atributo solo alimenta `get_model_info`/`list_adapters` y la etiqueta `supports_categorical_features` de `ForecasterFoundation` (`skforecast/foundation/_forecaster_foundation.py`, aprox. línea 195) [verificado en código].
- **Rompe compatibilidad**: no.
- **Decisión**: D18. Si sigue pendiente, no se toca.

### TPF-P8. Versión mínima de `tabpfn-time-series`

- **Qué** (según D19):
  - **(A)**: mantener `backend_package = "tabpfn-time-series>=1.3"` y documentar los límites con 1.3: TPF-V1 con varias GPU y TPF-V2.
  - **(B)**: `backend_package = "tabpfn-time-series>=1.4"` en `TabPFNAdapter` (aprox. línea 2906) y en la entrada `[tabpfn]` de `tools/foundation_compat/adapters.toml`. Además:
    - actualizar `test_get_model_info.py` (aprox. línea 171);
    - actualizar el comentario de `_MODEL_LICENSES` en `_utils.py` (aprox. líneas 307-308), la docstring de `weights_repo_id` y los textos de la sección 5.
  - Con (B) y la opción (b) de D-VER, añade en `_load_model` la comprobación de versión en tiempo de ejecución con el helper opcional de `02_common.md`. Un mínimo en `backend_package` no protege instalaciones existentes (X-4) [verificado en código].
- **Por qué**: TPF-F7, TPF-F4, TPF-F5, TPF-V1, TPF-V2.
- **Riesgo**: con (B), `tabpfn>=9.2.0` pasa a ser obligatorio [documentado], y quien tenga 1.3 debe actualizar. Con exog, las predicciones pueden cambiar al pasar de 1.3 a 1.4 por los nuevos features por defecto [inferido].
- **Rompe compatibilidad**: (B) endurece el requisito del backend opcional. La API de skforecast no cambia.
- **Decisión**: D19.

### TPF-P4. Re-ejecutar las celdas de TabPFN-TS de la guía con `TABPFN_TOKEN`

- **Se implementa en `10_docs_ci.md`**, que re-ejecuta la guía una sola vez tras los PR de adaptador. Las celdas son la 78 a la 83 y la tabla de la 108 (numeración de la revisión).
- **Por qué**: TPF-F17.
- **Criterio**: el LicenseWarning de la salida debe decir "TabPFN-3.5 License v1.0 (non-commercial without an enterprise license)" y enlazar a `Prior-Labs/tabpfn_3_5`. El MAE de la tabla cambiará; no es una comparación de versiones.
- Este PR no re-ejecuta el notebook.

### TPF-P7. Cachear `~/.cache/tabpfn` en el CI de compatibilidad

- **Se implementa en `10_docs_ci.md`**: añadir `~/.cache/tabpfn` al `path` de `actions/cache` (paso "Cache model weights") en `.github/workflows/foundation-backends.yml` (aprox. línea 94).
- **Por qué**: TPF-F11.
- **Riesgo**: 876 MB más en el presupuesto de caché de Actions, compartido con los demás adaptadores [inferido]. Ninguno para los usuarios.

## 4. Tests

**Tests unitarios en `skforecast/foundation/tests/tests_foundation_models/test_TabPFNAdapter.py`**

- **TPF-P1**: test nuevo, por ejemplo `test_TabPFNAdapter_predict_backend_warning_shown_outside_redirect_when_show_progress_False`.
  - El fake (inyectado con `model=`) emite un `UserWarning` dentro de `predict_df`.
  - Se comprueba que `warnings.showwarning` se llama cuando `sys.stderr` ya es el original. Se hace con `monkeypatch.setattr(warnings, "showwarning", recorder)` y un `recorder` que guarda `sys.stderr is original_stderr`.
  - Sin el arreglo, la única llamada ocurre con el `StringIO` redirigido.
  - No sirve `pytest.warns`: pasa ya hoy [probado: `evidence/tabpfn/verification/test_p1_pytest_warns.log`]. Tampoco `capsys` sin más: el plugin de warnings de pytest intercepta la visualización y no escribe en stderr [inferido].
  - Si el `monkeypatch` de `showwarning` no se comporta igual dentro del plugin de warnings de pytest en alguna versión de Python [inferido], usa un subproceso como en `evidence/tabpfn/verification/test_p1_pytest_warns.py` (`test_warning_text_does_not_reach_stderr`), comprobando que el texto sí llega a stderr.
- **TPF-P1**: test con `show_progress=True` que compruebe que el warning sigue llegando una sola vez, sin duplicarse.
- **TPF-P5**:
  - Extiende `test_TabPFNAdapter_predict_range_index_timestamps_are_datetime` (y `test_TabPFNAdapter_predict_range_index_issues_warning_and_returns_arrays`) con `@pytest.mark.filterwarnings("error::DeprecationWarning")`, o crea un test nuevo.
  - Comprueba que los timestamps de `fake.last_future_df` son `pd.date_range("2000-01-01", periods=n + steps, freq="D")[n:]`.
  - Hoy esos tests dan 2 `DeprecationWarning` en `_adapters.py:3305` [probado: `evidence/tabpfn/unit_tests.log`].
- **TPF-P9**: test de `_load_model` con el módulo falso en `sys.modules` (patrón de `test_TabPFNAdapter_load_model_LicenseWarning_on_successful_load`) que compruebe que `adapter._model.init_kwargs` contiene exactamente `max_context_length`, `tabpfn_mode` y `tabpfn_output_selection` (más `tabpfn_model_config` y `temporal_features` cuando se fijan), y que el fake acepta además `compile`, `num_workers_per_gpu` y `max_featurize_rows` sin error.
- **TPF-P2** (si D17 es A o B), con el mismo módulo falso:
  - el parámetro se reenvía solo si no es `None`, y con `None` el fake no recibe la clave;
  - `set_params` con un valor distinto reinicia `_model`, y con el mismo valor no lo reinicia (ver `test_TabPFNAdapter_set_params_resets_model_when_param_changes` y `test_TabPFNAdapter_set_params_no_reset_when_value_unchanged`);
  - con (A), las claves reservadas dan `ValueError` en `__init__` y en `set_params`;
  - `get_params` incluye la clave nueva: actualiza `test_TabPFNAdapter_get_params_returns_expected_keys_and_values` y `test_TabPFNAdapter_init_default_params`;
  - con (A), añade un caso `pipeline_kwargs` a `test_clone_round_trip_with_non_empty_config_dict` de `test_clone.py` si se quiere cubrir el almacenamiento por referencia.

**Otros tests**
- `test_get_model_info.py` (`test_get_model_info_output_for_each_adapter`): actualiza la tupla de `priorlabs/tabpfn-ts` si D19 es B (`"tabpfn-time-series>=1.4"`) o si D18 es B (quinto elemento `True`).
- `test_backends_compat_config.py`: sin cambios. Debe seguir pasando, porque comprueba que `adapters.toml` coincide con `backend_package`.

**Fixtures**
- `FakeTabPFNTSPipeline` en `fixtures_adapters.py` (aprox. líneas 519-583): `**kwargs` guardados en `__init__` y warning opcional en `predict_df` (TPF-P9).

**Compatibilidad (`tools/foundation_compat`)**
- `tests/contracts.py`, entrada `"tabpfn"` (aprox. líneas 114-131). `weak_kwargs` no sirve aquí: según la cabecera de `contracts.py` es para argumentos que el backend toma por `**kwargs`, y `test_contract.py` también exige que la firma los acepte (`bind_partial`); la firma de 1.3.0 no tiene `compile` ni `**kwargs`, así que el contrato fallaría con 1.3 [verificado en código: `evidence/tabpfn/old_backend_1.3.0.log`]. Por tanto:
  - con D17 (B) y D19 (B), añade `"compile"` a `kwargs` de la llamada `tabpfn_time_series:TabPFNTSPipeline` (1.4 lo declara explícitamente);
  - con D17 (B) y D19 (A), no lo añadas: solo se pasa cuando el usuario lo fija, y queda cubierto por el unitario;
  - con D17 (A) no hay nombres fijos que añadir.
- `adapters.toml`, entrada `[tabpfn]`: `backend_package` según D19.
- Smoke (`tests/test_smoke.py --backend tabpfn`): necesita `TABPFN_TOKEN`, que no estaba disponible en la revisión. Se ejecuta en el CI (secreto `TABPFN_TOKEN`) o en local con el token del mantenedor.

**Comprobaciones de evidencia antes y después** (desde la raíz del repositorio, con el entorno `~/.venvs/fm-review-tabpfn`; recréalo con `make_env.sh` como en la sección 1 si no existe):

```bash
export TABPFN_DISABLE_TELEMETRY=1 TABPFN_NO_BROWSER=1
PY=~/.venvs/fm-review-tabpfn/bin/python
# TPF-P1: sección 1, hoy False con show_progress=False y True con True
$PY dev/foundation_adapters_review/evidence/tabpfn/capabilities_check.py
# TPF-P1: hoy 2 passed; después, test_warning_text_does_not_reach_stderr debe fallar
$PY -m pytest -q -p no:cacheprovider dev/foundation_adapters_review/evidence/tabpfn/verification/test_p1_pytest_warns.py
# Corrección de la traducción (no debe cambiar)
$PY dev/foundation_adapters_review/evidence/tabpfn/compare_direct.py
$PY dev/foundation_adapters_review/evidence/tabpfn/verification/feature_level_check.py
# Pesos reales sin token: descargar antes el checkpoint v2 (44 MB, ver el docstring del script)
$PY dev/foundation_adapters_review/evidence/tabpfn/verification/real_weights_v2_check.py
# Contrato y unitarios
$PY -m pytest tools/foundation_compat/tests/test_contract.py --backend tabpfn -q -p no:cacheprovider
$PY -m pytest skforecast/foundation/tests/tests_foundation_models/test_TabPFNAdapter.py -q -p no:cacheprovider
```

Con D19 (A), repite el contrato y los unitarios en un entorno auxiliar con `tabpfn-time-series==1.3.0` y `tabpfn==9.0.0` (la revisión usó un entorno temporal así, ver `evidence/tabpfn/verification/old_backend_env.log`), siguiendo el patrón de entornos auxiliares de `evidence/README.md`.

`dev/foundation_adapters_review/evidence/common/backtest_guide.py` **no aplica**:
- ninguna propuesta cambia las predicciones con el uso por defecto [inferido: TPF-P1 solo cambia la visualización, TPF-P5 genera las mismas fechas (probado), TPF-P2 con `None` no cambia la llamada];
- además necesita los pesos de TabPFN-3.5, es decir, `TABPFN_TOKEN`.

## 5. Documentación y contexto de IA

- **Docstrings** (`skforecast/foundation/_adapters.py`, `TabPFNAdapter`):
  - `temporal_features` y el resumen de la clase (TPF-P3);
  - `show_progress` en la clase y en `__init__` (TPF-P1);
  - `weights_repo_id` (TPF-P3 o TPF-P8);
  - el atributo `supports_categorical_covariates`, si D18 es B;
  - el parámetro nuevo de TPF-P2 en la clase, `__init__`, `get_params` y `set_params`, si D17 lo introduce. En ese caso, añádelo también a la lista de parámetros de TabPFN-TS de la docstring de `FoundationModel` (`skforecast/foundation/_foundation_model.py`, aprox. líneas 126-131);
  - la entrada `supports_nan_in_series` de la docstring de la clase: añadir que el backend descarta las filas con target NaN y que, con como mucho un valor observado en la ventana de contexto, rellena la serie con 0 sin avisar (TPF-F15) [verificado en código].

  Sigue `.github/instructions/docstrings.instructions.md`.
- **Guía de usuario** `docs/user_guides/foundation-forecasting-models.ipynb`: solo se editan textos (`source`) y no se re-ejecuta el notebook.
  - Con D19 (B): en la tabla HTML de modelos soportados (celda 11 en la revisión, fila "TabPFN-TS"), cambia `pip install "tabpfn-time-series&gt;=1.3"` a `&gt;=1.4` en el `<code>`, en `data-copy` y en `aria-label`.
  - La salida de `list_adapters` (celda 13, que muestra `backend_package` y `supports_categorical_covariates`) y las celdas 78 a 83 y 108 las actualiza `10_docs_ci.md` al re-ejecutar (TPF-P4).
  - Sin D19 (B), este PR no toca la guía.
- **`skills/foundation-forecasting/references/adapter-parameters.md`**, sección `TabPFNAdapter`:
  - fila `temporal_features` (línea 129): di que `None` usa los features por defecto de la versión instalada, con lags de las covariables conocidas desde 1.4;
  - fila `show_progress` (línea 130): solo oculta la barra y los warnings del backend se siguen mostrando;
  - fila `model_id` (línea 123): versión fijada según TPF-P3/TPF-P8;
  - con D19 (A), una nota con los límites de 1.3 (TPF-V1 con varias GPU, TPF-V2);
  - si D17 lo introduce, fila del parámetro nuevo y su entrada en la tabla de `set_params` (línea 220);
  - con D18 (B), la frase de la línea 243 ("`supports_categorical_covariates` is `True` only for Chronos-2 and Timestral");
  - una frase con el mismo texto que la entrada `supports_nan_in_series` de la docstring sobre el relleno con 0 (TPF-F15).
- **`skills/foundation-forecasting/SKILL.md`**: con D19 (B), línea 60 `pip install "tabpfn-time-series>=1.4"`; con D18 (B), la línea 285 ("Only Chronos-2 and Timestral accept them natively"). Nada más que cambiar [verificado en código].
- **`tools/ai/llms-base.txt`**: sin cambios. Las líneas 458, 493, 511, 514 y 529-530 siguen siendo correctas [verificado en código]. Con D18 (B), valora mencionar las categóricas en la fila de TabPFNAdapter (línea 493). Si D17 introduce un parámetro y se quiere citar, hazlo en esa misma tabla.
- **Regeneración**: tras tocar `skills/` o `llms-base.txt`, sigue la skill `ai-context-sync` (`python tools/ai/generate_ai_context_files.py` y `--check`). No edites a mano los ficheros generados.
- **Release notes** (`docs/releases/releases.md`, sección `0.27.0`; las entradas van en las listas **Fixed**, **Changed** y **Added**, sin subapartado por área, y los cambios que alteran resultados o exigen actualizar código van también en "Before upgrading"). Texto propuesto (sustituye `NNNN` por el número del PR):
  - **Fixed**. Si 07 ya tiene una entrada para TabICL, amplíala con "TabICL and TabPFN-TS" en lugar de añadir otra:

    `+ <code>[FoundationModel]</code> and <code>[ForecasterFoundation]</code> with TabPFN-TS show the warnings of the backend, for example about missing values in the future covariates, when `show_progress=False` (the default); they were hidden together with the progress bar. The synthetic dates used for series with a `RangeIndex` no longer raise a `DeprecationWarning` with numpy 2.5. ([#NNNN](https://github.com/skforecast/skforecast/pull/NNNN))`
  - **Changed**, solo con D19 (B):

    `+ <code>[TabPFNAdapter]</code> requires `tabpfn-time-series>=1.4` (and therefore `tabpfn>=9.2.0`): with 1.3 and several GPUs, quantile levels outside the default grid could raise a `KeyError`, and `tabpfn` 9.0 lacks fixes in the extrapolation of time features and in the quantiles. ([#NNNN](https://github.com/skforecast/skforecast/pull/NNNN))`

    En "Before upgrading", en la lista de código que hay que actualizar: `+ TabPFN-TS: upgrade to `tabpfn-time-series>=1.4`. Its default temporal features include lags of the known covariates, so forecasts with `exog` can differ from those obtained with 1.3.`
  - **Added**, solo con D17 (A):

    `+ <code>[TabPFNAdapter]</code> accepts `pipeline_kwargs`, forwarded to `TabPFNTSPipeline`, for example `pipeline_kwargs={"compile": False}` to disable `torch.compile` on CUDA with `tabpfn-time-series>=1.4`. ([#NNNN](https://github.com/skforecast/skforecast/pull/NNNN))`
  - **Changed**, solo con D18 (B):

    `+ <code>[TabPFNAdapter]</code> declares `supports_categorical_covariates=True`: `category` and `str` covariates are passed to TabPFN, which handles them natively. ([#NNNN](https://github.com/skforecast/skforecast/pull/NNNN))`

  Usa la skill `release-note` para el formato final.

## 6. Decisiones pendientes

| Id | Decisión | Opciones | Recomendación | Impacto en el plan | Estado |
|---|---|---|---|---|---|
| D17 | Opciones de `TabPFNTSPipeline` (TPF-P2) | (A) `pipeline_kwargs` reenviado tal cual; (B) `compile` explícito; (C) solo documentar | (A): menor mantenimiento, cubre `compile`, `num_workers_per_gpu`, `max_featurize_rows` y opciones futuras sin tocar el adaptador en cada versión. Pasar solo si no es `None` | Con (A) o (B): parámetro nuevo, tests, docstrings (también la de `FoundationModel`), docs y release note (Added). Con (B) y D19 (B), además `"compile"` en `kwargs` de `contracts.py`. Con (C): solo texto en el docstring y en `adapter-parameters.md` | pendiente |
| D18 | Categóricas en TabPFN (TPF-P6) | (A) mantener `False` hasta probar con TabPFN-3.5 y token; (B) `True` por el changelog de TabPFN 9.0 y la prueba con v2 | (A) | Con (A) no se toca nada. Con (B): atributo, `test_get_model_info.py`, docs (`adapter-parameters.md` línea 243, `SKILL.md` línea 285) y release note | pendiente |
| D19 | Mínimo de `tabpfn-time-series` (TPF-P8) | (A) mantener `>=1.3` y documentar límites; (B) `>=1.4` (arrastra `tabpfn>=9.2.0`) | (B): con 1.3 hay `KeyError` con varias GPU (TPF-V1) y tabpfn 9.0 sin #1310/#1215 (TPF-V2). La revisión inicial recomendaba (A) | Con (B): `backend_package`, `adapters.toml`, `test_get_model_info.py`, comentario de `_utils.py`, guía (celda 11), `SKILL.md`, `adapter-parameters.md` y release note (Changed y Before upgrading); comprobación en tiempo de ejecución solo si D-VER (b) en `02_common.md`. Con (A): documentar TPF-V1 y TPF-V2 en `adapter-parameters.md` y probar el contrato con 1.3.0/9.0.0 (la revisión transversal recomienda un carril de compatibilidad con la versión mínima, solo de contrato) | pendiente |

## 7. Criterios de aceptación

- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_TabPFNAdapter.py -q` pasa. Son los 58 tests previos más los nuevos, sin `DeprecationWarning` de `_adapters.py` en el resumen (hoy hay 2 en la línea 3305).
- [ ] Si se tocan, `pytest skforecast/foundation/tests/tests_foundation_models/test_get_model_info.py skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py skforecast/foundation/tests/tests_foundation_models/test_clone.py -q` pasa.
- [ ] El test nuevo de TPF-P1 falla con el código de d9b8f4f y pasa con el arreglo: es discriminante.
- [ ] `capabilities_check.py`, sección 1: `True` con `show_progress=False` (hoy `False`) y `True` con `show_progress=True`.
- [ ] `test_p1_pytest_warns.py`: `test_pytest_warns_sees_warning_even_with_redirect` sigue pasando y `test_warning_text_does_not_reach_stderr` pasa a fallar (1 failed, 1 passed), porque el texto ya llega a stderr.
- [ ] `compare_direct.py`: diferencia 0.0 frente a la llamada directa en los 6 casos.
- [ ] `feature_level_check.py`: `featurized train and test frames identical (adapter vs direct): True`.
- [ ] `real_weights_v2_check.py`, con el checkpoint v2 y sin token:
  - adaptador frente a directo 0.0, con y sin exog;
  - sensibilidad a la exog 13.3456;
  - control con la exog desplazada 7.0137;
  - serie sola frente a en lote 0.0 con los features de 1.4 y de 1.3.
- [ ] `~/.venvs/fm-review-tabpfn/bin/python -m pytest tools/foundation_compat/tests/test_contract.py --backend tabpfn -q`: 5 passed con 1.4.0/9.2.0. Con D19 (A), también 5 passed con 1.3.0/9.0.0.
- [ ] `python tools/foundation_compat/run.py --adapter tabpfn`: contrato `passed`. El smoke da `passed` con `TABPFN_TOKEN`, o `skipped` ("no TABPFN_TOKEN") sin él.
- [ ] El job `tabpfn` del workflow Foundation Backends Compatibility, con token, está en verde: contrato 5 passed y smoke 6 passed.
- [ ] Si se tocó `skills/` o `llms-base.txt`, `python tools/ai/generate_ai_context_files.py --check` está sin diferencias.
- [ ] Hay entrada en `docs/releases/releases.md`, coherente con la de 07.
- [ ] `evidence/tabpfn/requirements-freeze.txt` no aparece modificado en el PR.
- [ ] La skill `verify` está en verde antes del commit.

## 8. Fuera de alcance

- **TPF-P4** (re-ejecutar las celdas 78 a 83 y 108 de la guía con `TABPFN_TOKEN`) y **TPF-P7** (caché de `~/.cache/tabpfn` en el CI): van en `10_docs_ci.md`.
- **TPF-P2** (D17), **TPF-P6** (D18) y **TPF-P8** (D19) quedan pospuestas si su decisión sigue pendiente al abrir el PR. En ese caso, este PR incluye solo TPF-P1, TPF-P3, TPF-P5 y TPF-P9.
- El helper común de reemisión de warnings y el de timestamps sintéticos se crean en `07_tabicl.md`; aquí solo se aplican.
- Van en `02_common.md`:
  - la comprobación central de contexto todo NaN (X-1). TabPFN rellena con 0 cuando una serie tiene como mucho 1 valor válido [verificado en código]; ese caso no lo cubre la comprobación de 02 y aquí solo se documenta (sección 5) [inferido];
  - la conversión de cuantiles a float (X-2);
  - la tabla de ids no soportados (X-3; para `priorlabs/tabpfn-v2` y similares basta con la documentación actual, TPF-F12);
  - el helper de comprobación de versión (D-VER, X-4).
- El `stacklevel` común de los avisos (X-6) no se toca aquí.
- Quedan fuera los benchmarks de velocidad o precisión: TabPFN-TS 1.3 frente a 1.4, `compile` frente a sin compilar, checkpoint fast frente al normal, `n_estimators`, `context_length`. No son objetivo de skforecast.
- No se prueba aquí, por falta de GPU o de token: el modo `client`, el comportamiento en CUDA y MPS, y el efecto de las covariables categóricas con TabPFN-3.5.
- Pedir upstream una opción para desactivar tqdm, que permitiría quitar la redirección de stderr, es una mejora opcional que no forma parte del PR.
