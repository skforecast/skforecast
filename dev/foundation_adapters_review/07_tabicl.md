# 07. TabICL (tabicl)

- **Rama propuesta**: `fix/tabicl-alignment`, creada desde `origin/0.27.x`.
- **Prioridad**: media, la que asigna la revisión. El adaptador está alineado con tabicl 2.2.0 [probado]. Los problemas son de otro tipo: warnings del backend ocultos con `show_progress=False`, niveles de cuantil `int` o `numpy.float32` rechazados, ningún mínimo declarado (2.0.2 falla) y textos desfasados [probado].
- **Adaptadores afectados**: `TabICLAdapter`. Este PR crea también dos helpers comunes en `skforecast/foundation/_utils.py`, que `TabPFNAdapter` adoptará en `08_tabpfn.md`.
- **Ficheros afectados**:
  - `skforecast/foundation/_utils.py` (helpers nuevos).
  - `skforecast/foundation/_adapters.py` (`TabICLAdapter`: `predict`, `_load_model`, `_get_future_timestamps`, `backend_package` y docstrings).
  - `skforecast/foundation/tests/tests_foundation_models/fixtures_adapters.py` (`FakeTabICLForecaster`).
  - `skforecast/foundation/tests/tests_foundation_models/test_TabICLAdapter.py`, `test_utils.py` y `test_get_model_info.py` (este último solo con D20).
  - `tools/foundation_compat/adapters.toml` (solo con D20).
  - `docs/user_guides/foundation-forecasting-models.ipynb` (solo el texto de TabICL, sin re-ejecutar).
  - `skills/foundation-forecasting/SKILL.md`, `skills/foundation-forecasting/references/adapter-parameters.md`, `tools/ai/llms-base.txt` y los ficheros generados a partir de ellos.
  - `docs/releases/releases.md`.
- **Depende de**: `02_common.md`, por dos decisiones que se toman allí: D-Q (conversión central de los niveles de cuantil a `float`, que cubriría TAB-P2) y D-VER (helper común de comprobación de versión en tiempo de ejecución, que TAB-P3 necesita para dar un error claro). Si 02 no está fusionado o se eligió la otra opción, seguir las ramas alternativas descritas en TAB-P2 y TAB-P3. No depende de `01_t0.md`.
- **Bloquea a**: `08_tabpfn.md`, que aplica a TabPFN los dos helpers creados aquí, y `10_docs_ci.md`, que re-ejecuta la guía y actualiza el CI de compatibilidad.
- **Cierra issue**: ninguna issue de skforecast conocida.

## 1. Estado en la revisión (2026-11-06)

Base de la revisión: commit `d9b8f4f` de `origin/0.27.x`, con skforecast 0.27.0 en desarrollo [verificado en código].

### Versiones

| Elemento | Probado antes | Última en la revisión |
|---|---|---|
| tabicl (PyPI) | 2.2.0, en el CI y en el entorno de revisión [probado] | 2.2.0, publicada el 2026-09-02 [probado, evidence/tabicl/verification/pypi_tabicl.log]. Historial: 2.0.0 (2026-02-12), 2.0.1 (2026-02-14), 2.0.2 (2026-02-23), 2.0.3 (2026-03-02), 2.1.0 (2026-04-21), 2.1.1 (2026-04-29), 2.2.0 (2026-09-02) [probado, mismo log] |
| main upstream | | 2 commits sin release tras v2.2.0: `c91f00d` (2026-10-09) y `d83091b` (2026-10-27); HEAD `d83091b` [probado]. No son 20: esos 20 commits son v2.1.1..v2.2.0 [probado] |
| Mínimo real | | 2.0.3: con 2.0.2 falla el constructor [probado, evidence/tabicl/verification/old_backend_2.0.2.log, evidence/tabicl/verification/old_backend_2.0.3.log, evidence/tabicl/verification/old_backend_2.0.3_smoke.log] |
| Requisitos de tabicl 2.2.0 | | `requires_python >=3.10` y `torch>=2.2`. El extra `forecast` pide `pandas>=2.1.2`, `statsmodels>=0.14.5` y `matplotlib>=3.10.3`, y ya no incluye gluonts [documentado, evidence/tabicl/pypi_tabicl.log, evidence/tabicl/verification/pypi_tabicl.log] (de 2.0.2 a 2.1.1 el `pyproject.toml` pedía `gluonts>=0.16.0`; retirado en 2.2.0 por el PR #108, `CHANGES.md` de v2.2.0) [verificado en código, tags v2.0.2 a v2.2.0 del clon upstream de "Antes de empezar"; evidence/tabicl/upstream_issues.log]. Son compatibles con las dependencias de skforecast [verificado en código] |
| Entorno de revisión | Python 3.12.3, tabicl 2.2.0, torch 2.15.0+cpu, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1, huggingface-hub 2.4.0, statsmodels 0.15.0, matplotlib 3.11.2, einops 0.8.2; Linux x86_64, 4 CPU, sin GPU [probado, evidence/tabicl/requirements-freeze.txt, evidence/tabicl/env.log] | |

### Checkpoints en el Hub

- `jingang/TabICL`: sha `4dcd344` (último commit el 2026-02-18, que solo cambia el README; los checkpoints se subieron en `d700cf0` el 2026-02-12). Licencia bsd-3-clause, no gated, 441.3 MB en total. Contiene 3 checkpoints de clasificación y 1 de regresión (`tabicl-regressor-v2-20260212.ckpt`) [probado, evidence/tabicl/verification/hub.log, evidence/tabicl/hub_jingang_TabICL.log].
- tabicl 2.2.0 fija `repo_id="jingang/TabICL"` y el checkpoint por defecto `tabicl-regressor-v2-20260212.ckpt` en `src/tabicl/_sklearn/regressor.py` [verificado en código].
- `soda-inria/tabicl` no existe en el Hub (404). En la búsqueda "tabicl", los demás repos son de terceros (eugenehp, general-forecast, AMFORGE...), salvo `jingang/TabICL-clf`, del mismo autor [probado, evidence/tabicl/verification/hub.log].
- No hay ningún checkpoint nuevo desde el 2026-02-12 [probado, evidence/tabicl/verification/hub.log].

### Línea base del CI

Run 42212742430 del workflow Foundation Backends Compatibility, job 127875169082 "tabicl", rama 0.27.x, commit `09433c3`, 2026-11-06, CPython 3.12.16 [probado, evidence/tabicl/ci_baseline.log]:

- Versiones instaladas: tabicl 2.2.0, torch 2.15.0+cpu, numpy 2.5.4, pandas 2.3.3, huggingface_hub 2.4.0.
- `test_contract.py`: 3 passed. `test_smoke.py`: 6 passed, con 2 `InputTypeWarning`.
- `latest_on_pypi: 2.2.0`, `held_back: false`, `status: passed`. Caché HF restaurada de unos 101 MB.

En el entorno de revisión, contrato y smoke dan 9 passed y `test_TabICLAdapter.py` da 50 passed con 4 warnings: 2 `InputTypeWarning` y 2 `DeprecationWarning` de `_adapters.py:2631` (`pd.Timedelta(days=n)`) [probado, evidence/tabicl/verification/contract_smoke_2.2.0.log, evidence/tabicl/verification/unit_tests.log, evidence/tabicl/unit_tests.log].

### Antes de empezar

Este plan puede quedar desfasado. Antes de tocar código, desde la raíz del repositorio:

```bash
# Última versión de tabicl, requires_python y requisitos del extra forecast
python dev/foundation_adapters_review/evidence/common/pypi_info.py tabicl
# Checkpoints: repo de pesos, otros repos del autor y ausencia de soda-inria/tabicl
python dev/foundation_adapters_review/evidence/common/hub_info.py --repo jingang/TabICL
python dev/foundation_adapters_review/evidence/common/hub_info.py jingang --prefix TabICL
python dev/foundation_adapters_review/evidence/common/hub_info.py --repo soda-inria/tabicl
# Commits sin release en main upstream (esperado: c91f00d y d83091b)
git clone https://github.com/soda-inria/tabicl ~/fm-review-src/tabicl  # solo si no existe
git -C ~/fm-review-src/tabicl fetch --tags origin
git -C ~/fm-review-src/tabicl log --oneline v2.2.0..origin/main
# Estado de la issue #172 y del PR #174 (alias "fourier")
gh api search/issues -X GET -f q="repo:soda-inria/tabicl fourier"
# Recrear el entorno de revisión si no existe
bash dev/foundation_adapters_review/evidence/common/make_env.sh tabicl "tabicl[forecast]" 3.12
```

`make_env.sh` sobrescribe `evidence/tabicl/requirements-freeze.txt`; no incluir ese cambio en el PR salvo que se quiera actualizar la evidencia [verificado en código].

Si apareció una versión nueva de tabicl o un checkpoint nuevo:

1. Recrear `~/.venvs/fm-review-tabicl` con `make_env.sh` y correr contrato, smoke y tests unitarios (comandos de la sección 7).
2. Leer `CHANGES.md` de la nueva tag y comprobar en el código upstream:
   - las firmas de `TabICLForecaster.__init__` y de `predict_df` (`src/tabicl/forecast/_forecaster.py`);
   - la validación de cuantiles (`ForecastEngine._validate_quantiles` en `src/tabicl/forecast/_engine.py`, que en 2.2.0 exige `isinstance(q, float)`);
   - si la barra tqdm de `src/tabicl/forecast/_dispatch.py` admite ya `disable`; en ese caso, preferir esa opción a la redirección de stderr y simplificar TAB-P1;
   - si se arregló el alias `"fourier"` (PR #174);
   - si se fijó `huggingface_hub<2` (el commit `c91f00d` se titula así pero no lo hace, TAB-F5).
3. Re-ejecutar `compare_direct.py` y `verification/verify_tabicl.py` y comparar con los valores de la sección 2.
4. Revisar la recomendación de D20 con esa información. El mínimo recomendado solo cambia si la nueva versión arregla algo que el adaptador necesita.

## 2. Hallazgos

Las rutas de evidencia son relativas a este fichero.

- **TAB-F1** [probado] tabicl 2.2.0 es la última versión. Contrato (3 passed) y smoke (6 passed) pasan en el entorno de revisión y en el CI. En main upstream hay 2 commits sin publicar (`c91f00d` y `d83091b`), no 20. Evidencia: evidence/tabicl/verification/contract_smoke_2.2.0.log, evidence/tabicl/ci_baseline.log, evidence/tabicl/verification/pypi_tabicl.log.
- **TAB-F2** [probado] `backend_package = "tabicl[forecast]"` no declara mínimo. `tabicl.forecast` aparece en v2.0.2 (commit `808809a`), pero allí el parámetro del constructor se llama `output_selection` y no `point_estimate`. Con 2.0.2 el contrato da 1 failed y 2 passed (`TypeError: got an unexpected keyword argument 'point_estimate'`). Con 2.0.3 pasan el contrato (3) y el smoke (6); el cambio de nombre está en CHANGES 2.0.3 [documentado]. Matiz de la verificación: la 2.0.2 solo fue la última versión entre el 2026-02-23 y el 2026-03-02, así que el impacto práctico es pequeño. Evidencia: evidence/tabicl/verification/old_backend_2.0.2.log, evidence/tabicl/verification/old_backend_2.0.3.log, evidence/tabicl/verification/old_backend_2.0.3_smoke.log.
- **TAB-F3** [verificado en código] Las firmas que usa el adaptador son iguales en v2.0.3, v2.1.0, v2.1.1 y v2.2.0: `TabICLForecaster(max_context_length, temporal_features, point_estimate, tabicl_config)` y `predict_df(context_df, future_df, quantiles)`. `temporal_features` acepta también cadenas (`list[str | TimeTransform]`) desde 2.1.0. CHANGES 2.1.0 (PR #84) recomienda importar desde el nivel superior de `tabicl`, pero `tabicl.forecast` no lleva prefijo `_`, está en `__all__` y exporta `TabICLForecaster` en todas esas versiones. No hace falta cambiar el import.
- **TAB-F4** [verificado en código] 2.2.0 trae tres cambios que afectan al módulo forecast y al adaptador: quita gluonts del extra (PR #108), corrige un off-by-one de `DatetimeEncoder` (PR #151, issue #136) y cambia el orden de dispositivos por defecto a CUDA, XPU, MPS, CPU (PR #144). Que la corrección de `DatetimeEncoder` cambie las predicciones respecto a <2.2 es [inferido]: no se midió, y medirlo queda fuera de alcance. Ninguno de los tres cambios exige tocar el adaptador.
- **TAB-F5** [verificado en código] El commit `c91f00d` de main se titula "Pin huggingface_hub to version <2", pero su diff de `pyproject.toml` solo añade `tokenizers>=0.20`: no hay pin. Con huggingface-hub 2.4.0 funciona [probado, evidence/tabicl/verification/contract_smoke_2.2.0.log]. Vigilar la próxima release.
- **TAB-F6** [probado] `jingang/TabICL` (bsd-3-clause, no gated) no cambia de ficheros desde el 2026-02-12 y tiene un único checkpoint de regresión. `soda-inria/tabicl` no existe en el Hub. `weights_repo_id = "jingang/TabICL"` y `weights_in_hf_cache = True` son correctos. Evidencia: evidence/tabicl/verification/hub.log, evidence/tabicl/hub_jingang_TabICL.log.
- **TAB-F7 (corregido)** [probado] `model_id` no se pasa nunca al backend. Cualquier id con el prefijo `soda-inria/tabicl` (por ejemplo `soda-inria/tabicl-v1-does-not-exist`, `soda-inria/tabicl-v2` o `soda-inria/tabicl-regressor-v2-20260212`) se resuelve a `TabICLAdapter` y predice con el checkpoint v2. En cambio, el repo real `jingang/TabICL` se rechaza con ValueError, y también `soda-inria/TabICL`, porque el prefijo distingue mayúsculas. El checkpoint solo se elige con `tabicl_config["checkpoint_version"]`. Corrección de la verificación: el texto impreciso está sobre todo en el docstring del adaptador ("HuggingFace model ID"). `tools/ai/llms-base.txt` línea 62 es una frase genérica sobre todos los adaptadores, y la línea 513 ya dice que los pesos vienen de `jingang/TabICL` [verificado en código]. Evidencia: evidence/tabicl/resolve_adapter.log, evidence/tabicl/verification/verify_tabicl.log (sección "F7 model_id resolution"), evidence/tabicl/capabilities_check.log.
- **TAB-F8 (corregido)** [probado] La salida del adaptador coincide con una llamada directa a `TabICLForecaster.predict_df` sobre la misma entrada:
  - casos A (serie única con exog y cuantiles, y punto medio) y C (dos series con covariables distintas): `max_abs_diff = 0.0`;
  - caso B (tres series de distinta longitud): en la ejecución original dio 0.0, pero al re-ejecutarlo da 3.8e-6 y 1.9e-6 (allclose True), no 0 exacto. Además, su tramo NaN (posiciones 50 a 59 de 200) caía fuera de la ventana de 128 filas, así que el caso no ejercitaba NaN;
  - con 10 NaN dentro de una ventana de 64 filas, el adaptador y `predict_df` sobre la misma ventana dan `max_abs_diff = 0.0`.

  La conclusión de que la traducción es correcta se mantiene. Evidencia: evidence/tabicl/compare_direct.log, evidence/tabicl/verification/rerun_compare_direct.log, evidence/tabicl/verification/verify_tabicl.log (sección "F8 NaN inside the context window").
- **TAB-F9** [probado] Los atributos de capacidad son veraces con 2.2.0:
  - `requires_equal_length_context=False`: el backend reparte por `item_id`.
  - `supports_heterogeneous_covariates=False`: el backend usa la intersección de columnas en un solo frame.
  - `supports_past_only_covariates=False`: esas columnas se descartan y `FoundationModel` avisa con `IgnoredArgumentWarning`.
  - `supports_nan_in_series=True`: el backend descarta las filas NaN y, si queda como mucho un valor válido, rellena con 0.
  - `supports_categorical_covariates=False`: una covariable de texto (directa) o `category` (vía `FoundationModel`) falla en el backend con "NumPy arrays passed to TabICL must be castable to a numeric dtype".
  - `requires_hf_auth=False` y `requires_provider_auth=False`.

  Evidencia: evidence/tabicl/capabilities_check.log, evidence/tabicl/verification/verify_tabicl.log (sección "F9/F14").
- **TAB-F10** [probado] `FoundationModel` acepta niveles `int` (`[0, 0.5, 1]`) y `numpy.float32` y los reenvía con su tipo. tabicl los rechaza con `ValueError: Quantiles must be a list of floats` (`_engine.py` v2.2.0, línea 111, `isinstance(q, float)` [verificado en código]). `numpy.float64` pasa porque es subclase de `float`, y una tupla de floats funciona. Con `float(q)` las predicciones son idénticas a las de niveles float. Con float32 las columnas salen como `q_0.10000000149011612`, por el f-string de `FoundationModel.predict` y no por el adaptador. Relacionado con X-2, que es de `02_common.md`. Evidencia: evidence/tabicl/capabilities_check.log, evidence/tabicl/verification/verify_tabicl.log (sección "F10/P2"), evidence/cross_cutting/probe_shared.log (sección 2).
- **TAB-F11** [verificado en código] tabicl no tiene límite duro de contexto ni de horizonte. En `_prepare_context` (v2.2.0), `max_context_length` solo recorta con `slice_by_timestep`, y lo hace después de descartar los NaN. En cambio, la guía (celda 11 en la revisión, tabla "Supported Foundation Models") dice "Default context length 4096 | Max context length 4096" para TabICL.
- **TAB-F12** [verificado en código] El adaptador no expone `device`: se pasa con `tabicl_config={"device": ...}`, probado con `"cpu"` [probado, evidence/tabicl/capabilities_check.log]. `SeriesDispatcher` solo mira `torch.cuda.is_available()`: sin CUDA (también con MPS o XPU) va por CPU con joblib/loky y `min(cpu_count, 8)` workers. Cada `TabICLRegressor` elige dispositivo en el orden CUDA, XPU, MPS, CPU (`DEFAULT_DEVICE_PREFERENCE` en `_torch_devices.py`) y cae a CPU si `mps_possibly_faulty()`. Es coherente con el docstring de `FoundationModel` (aprox. líneas 226 a 235 en d9b8f4f). CUDA y MPS no se pudieron ejecutar.
- **TAB-F13** [probado] La licencia BSD-3-Clause de `_MODEL_LICENSES["soda-inria/tabicl"]` (`skforecast/foundation/_utils.py`, aprox. línea 301 en d9b8f4f) es correcta. El `LICENSE` de v2.2.0 (líneas 30 a 35) indica que `src/tabicl/forecast` es derivado de TabPFN-TS bajo Apache-2.0; eso afecta al código, no a los pesos [documentado]. La issue upstream #83 (licencia de los pesos) está cerrada, pero sus comentarios no se pudieron leer. Evidencia: evidence/tabicl/hub_jingang_TabICL.log, evidence/tabicl/verification/upstream_issues.log.
- **TAB-F14** [probado] El índice diario sintético desde 2000-01-01 para series con `RangeIndex` sigue siendo necesario. tabicl aplica `pd.to_datetime` a los timestamps enteros (`_ts_dataframe.py` v2.2.0, línea 189 [verificado en código]) y los convierte en nanosegundos desde 1970 (`Timestamp("1970-01-01 00:00:00.000000120")`). No hay issue upstream sobre esto. Evidencia: evidence/tabicl/capabilities_check.log, evidence/tabicl/verification/verify_tabicl.log (sección "F9/F14"), evidence/tabicl/upstream_issues.log.
- **TAB-F15** [probado] Con `show_progress=False`, el valor por defecto, la redirección de stderr que oculta la barra tqdm oculta también el `UserWarning` del backend "Some covariate values in future_tsdf are missing (NaN)", emitido en `_forecaster.py:260` (`_align_covariates`), que el docstring promete. Con `show_progress=True` sí se ve. La redirección sigue siendo necesaria [verificado en código]: tabicl no ofrece desactivar tqdm (`_dispatch.py` lo usa sin `disable`) y `TQDM_DISABLE` solo se lee al importar tqdm (`envwrap`). `TabPFNAdapter` usa el mismo patrón (X-7, TPF-F6, `_adapters.py` aprox. línea 3176). Evidencia: evidence/tabicl/stderr_redirect_check.log, evidence/tabicl/verification/verify_tabicl.log (secciones "F15").
- **TAB-F16** [verificado en código] El adaptador es fino y cumple el contrato de `_AdapterBase` (líneas 2124 a 2748 de `_adapters.py` en d9b8f4f). Solo construye frames largos por serie, usa el índice sintético para `RangeIndex` y redirige stderr. No hay padding, agrupación, clipping, fallback de MPS ni comprobación de versión.
- **TAB-F17** [verificado en código] No quedan opciones útiles del backend sin usar:
  - `TabICLForecaster` no expone el batching ni el `num_workers` de `SeriesDispatcher`.
  - `device`, `n_estimators`, `use_amp`, `random_state`, `checkpoint_version`, `model_path`, `allow_auto_download` y `n_jobs` se alcanzan vía `tabicl_config`.
  - El regresor calcula siempre los cuantiles (`output_type` con punto y `"quantiles"`), así que pedir los 9 por defecto con `quantiles=None` no cuesta más.
  - `kv_cache` no aportaría, porque hay un `fit` y un `predict` por serie [inferido].
- **TAB-F18** [verificado en código] Textos imprecisos:
  - (a) Guía, celda 11: contexto máximo 4096.
  - (b) Docstring de `TabICLAdapter`, con cuatro problemas:
    - `model_id` aparece como "HuggingFace model ID";
    - `show_progress` cita solo la barra `GPU 0: ...`, cuando en CPU es "Predicting time series";
    - dice que la exog futura con NaN se acepta "with a warning", que con el valor por defecto no se ve;
    - `temporal_features` solo admite `TimeTransform`, cuando con tabicl >=2.1 acepta también cadenas. `["index", "datetime"]` funciona y `["fourier"]` da TypeError en 2.2.0 (issue #172 y PR #174 abiertos) [probado, evidence/tabicl/verification/verify_tabicl.log, sección "F18", evidence/tabicl/verification/upstream_issues.log].
  - (c) `references/adapter-parameters.md` línea 112: el mismo problema con `temporal_features`.
  - (d) `SKILL.md` línea 284 y `adapter-parameters.md` línea 230 dicen que `context_length` "reloads the model" en TabICL. En realidad, `set_params` solo recrea `TabICLForecaster`, y el checkpoint se carga con `torch.load` dentro de `TabICLRegressor.fit` (`regressor.py:357/414`), una vez por serie y por llamada.
  - (e) "(0, 1)" en el docstring de `SUPPORTED_QUANTILES` y en `llms-base.txt` línea 492 (ver TAB-V2).
- **TAB-F19** [probado] Los 50 tests unitarios pasan. `FakeTabICLForecaster` (`fixtures_adapters.py`, aprox. línea 460 en d9b8f4f) no valida el tipo de los cuantiles, no emite warnings y no descarta NaN, así que los tests no detectan ni TAB-F10 ni TAB-F15. `contracts.py` solo comprueba los kwargs aceptados. El compat detecta la rotura de 2.0.2, pero solo se ejecuta con la última versión: ningún test cubre una versión mínima. Evidencia: evidence/tabicl/verification/unit_tests.log.
- **TAB-V1** [probado] `FoundationModel` recorta el contexto a las últimas `context_length` filas, NaN incluidos, y después tabicl descarta las filas NaN, así que el modelo ve menos de `context_length` observaciones. Una llamada directa con la serie completa conserva las últimas `max_context_length` observaciones válidas. En la prueba, el adaptador da una diferencia de 0.3516 frente a esa llamada directa, y de 0.0 frente a `predict_df` sobre la misma ventana de 64 filas con 10 NaN. Es una decisión centralizada de `FoundationModel`, común a todos los adaptadores; basta con documentarla. Evidencia: evidence/tabicl/verification/verify_tabicl.log (sección "F8").
- **TAB-V2** [probado] El docstring de `SUPPORTED_QUANTILES` de `TabICLAdapter` y `llms-base.txt` línea 492 dicen "(0, 1)". Sin embargo, tabicl acepta 0.0 y 1.0 (valida `0 <= q <= 1`), y las Notes del mismo docstring ya dicen `[0, 1]`. Evidencia: evidence/tabicl/verification/verify_tabicl.log (sección "F10/P2": columnas `q_0`, `q_0.5`, `q_1`).

Hallazgos relacionados de otros planes, solo como contexto:

- X-2 (tipos de cuantil): `02_common.md`.
- X-5 y X-7 (`pd.Timedelta(days=n)` y `redirect_stderr` en los dos adaptadores tabulares): este PR crea los helpers y `08_tabpfn.md` los aplica a TabPFN.
- TPF-F6, TPF-F14, TPF-V4, TPF-P1 y TPF-P5: `08_tabpfn.md`.

Descartado:

- "20 commits sin release en main upstream": son 2, `c91f00d` y `d83091b` [probado].
- La evidencia original de TAB-P1 (`proto_fixes.py`) no demostraba la reemisión. Usaba un `catch_warnings(record=True)` externo, que captura el warning también sin el parche (probe 10 de evidence/tabicl/capabilities_check.log). La propuesta se mantiene, con la prueba en subproceso de la verificación [probado].

## 3. Cambios a implementar

### TAB-P1. Reemitir los warnings del backend ocultos por la redirección de stderr (helper común)

- **Qué**:
  - Crear en `skforecast/foundation/_utils.py` un context manager común. El nombre propuesto es `_suppress_backend_progress(show_progress: bool)`; hoy no existe [verificado en código].
    - Con `show_progress=False`, redirige stderr a un `io.StringIO()` para ocultar tqdm y graba dentro los warnings con `warnings.catch_warnings(record=True)`.
    - Al salir, ya restaurado stderr, reemite cada warning con `warnings.warn_explicit`, conservando la categoría, el fichero y la línea originales.
    - Con `show_progress=True` no hace nada.
  - En `TabICLAdapter.predict` (aprox. líneas 2511 a 2521 en d9b8f4f), sustituir el bloque `_stderr_cm = (contextlib.redirect_stderr(io.StringIO()) ...)` / `with _stderr_cm:` por `with _suppress_backend_progress(self.show_progress):`.
  - `TabPFNAdapter` adopta el helper en `08_tabpfn.md` (TPF-P1), no en este PR. Por eso los imports `contextlib` e `io` de `_adapters.py` se quedan aquí, porque los sigue usando `TabPFNAdapter.predict` (aprox. línea 3176) [verificado en código]. Se quitan en 08.

  Esbozo:

  ```python
  @contextlib.contextmanager
  def _suppress_backend_progress(show_progress: bool):
      if show_progress:
          yield
          return
      records = []
      try:
          with contextlib.redirect_stderr(io.StringIO()):
              with warnings.catch_warnings(record=True) as records:
                  yield
      finally:
          for record in records:
              warnings.warn_explicit(
                  message=record.message,
                  category=record.category,
                  filename=record.filename,
                  lineno=record.lineno,
                  source=record.source,
              )
  ```

  No llamar a `warnings.simplefilter("always")` dentro del helper. Así se respetan los filtros del usuario: un warning con filtro `"ignore"` no se graba, y `warn_explicit` vuelve a aplicar los filtros al reemitir [inferido]. El `try/finally`, para reemitir también cuando el backend lanza una excepción, es una sugerencia [inferido].
- **Por qué**: TAB-F15, y TAB-F19 (los tests no lo detectan). Es el mismo patrón que TPF-F6 (X-7).
- **Riesgo**: bajo. En un subproceso sin `catch_warnings` externo, con el prototipo el `UserWarning` aparece en stderr y la barra sigue oculta; sin el prototipo no aparece nada [probado, evidence/tabicl/verification/verify_tabicl.log, sección "P1 prototype"]. El warning de NaN en covariables futuras se emite en el proceso padre (`_forecaster.py:260`, `_align_covariates`) [probado, mismo log]. Matices [inferido]:
  - `warn_explicit` sin `registry` no deduplica por ubicación, así que en un backtesting el aviso saldrá en cada fold, respetando los filtros. Se puede valorar pasar un `registry` propio del helper.
  - `catch_warnings` no es thread-safe.
  - Los warnings que se emitan dentro de los workers de loky (varias series en CPU) no pasan por la redirección del proceso padre.
- **Rompe compatibilidad**: no. Solo aparecen warnings que antes se perdían. Las predicciones no deberían cambiar, porque el helper no toca ni las entradas ni la salida de `predict_df` [inferido].
- **Decisión**: ninguna.

### TPF-P5 aplicado a TabICL (TPF-F14, TPF-V4, X-5). Timestamps sintéticos sin `pd.Timedelta(days=n)` (helper común)

El id pertenece a `08_tabpfn.md`. Se incluye aquí porque este PR crea el helper y TabICL tiene el mismo código.

- **Qué**:
  - Crear en `skforecast/foundation/_utils.py` un helper común. El nombre propuesto es `_synthetic_daily_index(n_context: int, steps: int = 0)`; hoy no existe [verificado en código]. Devuelve el índice diario sintético desde 2000-01-01, para el contexto (`steps=0`) o para el horizonte.
  - En `TabICLAdapter._get_future_timestamps` (aprox. línea 2631 en d9b8f4f), sustituir `pd.Timestamp("2000-01-01") + pd.Timedelta(days=n)` por el helper.
  - Opcionalmente, usarlo también en `TabICLAdapter._get_timestamps` (aprox. línea 2569), que ya usa `pd.date_range("2000-01-01", periods=len(series), freq="D")` y no avisa [verificado en código].

  Forma propuesta:

  ```python
  pd.date_range("2000-01-01", periods=n_context + steps, freq="D")[n_context:]
  ```

  Una alternativa mínima equivalente es `pd.Timedelta(n, unit="D")`.
- **Por qué**: el constructor `pd.Timedelta(days=n)` (la forma con keyword, no la suma) emite `DeprecationWarning` ("The 'generic' unit for NumPy timedelta is deprecated") con numpy 2.5.4 y pandas 2.3.3. El corte de `pd.date_range` y `pd.Timedelta(n, unit="D")` no avisan y dan las mismas fechas [probado con n=5, evidence/cross_cutting/probe_shared.log sección 5, evidence/tabpfn/verification/rerun_probe_int_timestamps.log]. Hoy el aviso sale en dos tests de `test_TabICLAdapter.py` [probado, evidence/tabicl/unit_tests.log].
- **Riesgo**: nulo. Solo afecta al camino de `RangeIndex`, da las mismas fechas y no cambia predicciones [inferido a partir de la igualdad de fechas probada].
- **Rompe compatibilidad**: no.
- **Decisión**: ninguna.

### TAB-P2. Niveles de cuantil a float

- **Qué**: depende de D-Q, que se decide en `02_common.md`.
  - Con D-Q (a), la conversión `float(q)` vive en `FoundationModel.predict` (02) y este PR no toca `TabICLAdapter.predict`. Aquí solo se añaden la validación en el fake (TAB-P6) y un test de extremo a extremo con niveles `int` y `numpy.float32`.
  - Con D-Q (b), o si 02 no está fusionado, la conversión se hace en `TabICLAdapter.predict` (aprox. línea 2470 en d9b8f4f), y las columnas del resultado se seleccionan con la misma lista:

    ```python
    quantile_list = [float(q) for q in quantiles] if quantiles is not None else None
    ```

    Así `tabicl_quantiles` y `group[quantile_list]` usan floats. Los nombres de columna con float32 (`q_0.10000000149011612`) no se pueden arreglar en el adaptador, porque salen del f-string de `FoundationModel.predict` (aprox. línea 1310) [probado].
- **Por qué**: TAB-F10 y X-2.
- **Riesgo**: mínimo. Las predicciones son idénticas a las de niveles float, y con niveles int las columnas salen como `q_0`, `q_0.5` y `q_1` [probado, evidence/tabicl/verification/verify_tabicl.log, evidence/tabicl/proto_fixes.log].
- **Rompe compatibilidad**: no; llamadas que hoy fallan pasan a funcionar.
- **Decisión**: D-Q (`02_common.md`).

### TAB-P3. Versión mínima de tabicl

- **Qué**:
  - Fijar el mínimo según D20 en `TabICLAdapter.backend_package` (aprox. línea 2260 en d9b8f4f): `"tabicl[forecast]>=2.2.0"` con (b) o `"tabicl[forecast]>=2.0.3"` con (a).
  - Poner el mismo valor en `backend_package` de la entrada `[tabicl]` de `tools/foundation_compat/adapters.toml`. Lo exige `test_adapters_toml_entries_match_the_adapter_classes` en `test_backends_compat_config.py` [verificado en código].
  - Entrecomillar el comando del ImportError de `TabICLAdapter._load_model` (aprox. línea 2559). Hoy es `` f"Install it with `pip install {self.backend_package}`." `` sin comillas, y con `>=` el shell redirigiría la salida [verificado en código]. El mensaje ya interpola `backend_package`, así que no hay que cambiarlo a mano. Es el mismo patrón que `TabPFNAdapter._load_model` (aprox. línea 3225) y `_import_timesfm`:

    ```python
    f'Install it with `pip install "{self.backend_package}"`.'
    ```

    `02_common.md` no toca este mensaje ni ningún adaptador (sección 3.4 y sección 8 de 02), así que las comillas se ponen siempre en este PR.
  - Con D-VER (b), llamar a `_check_backend_min_version(type(self))` de `02_common.md` (sección 3.4) al principio de `TabICLAdapter._load_model`; el helper lee la distribución (`tabicl`) y el mínimo del `backend_package` fijado en el primer punto (D20), sin argumentos propios. Si D-VER elige la subdecisión de argumentos explícitos, pasar `"tabicl"` y el mínimo de D20. Así, quien ya tenga 2.0.2 recibe un error claro en lugar del TypeError del constructor. El precedente es `TimesFM3Adapter._load_model` (aprox. líneas 1629 a 1696, con `importlib.metadata` desde la línea 1658).
- **Por qué**: TAB-F2, TAB-F4 y TAB-F19. Un mínimo en `backend_package` solo cambia el comando de instalación, la documentación y el entorno de compat. Con 2.0.2, `tabicl.forecast` se importa bien, el ImportError no salta y el TypeError sigue igual [verificado en código] (X-4).
- **Riesgo**: bajo. Con (b), quien tenga de 2.0.3 a 2.1.1 debe actualizar.
- **Rompe compatibilidad**: con (b), sí, para usuarios con tabicl <2.2 (anotarlo en "Before upgrading"). Con (a), solo para 2.0.2, que ya no funciona.
- **Decisión**: D20, y D-VER (`02_common.md`) para la comprobación en tiempo de ejecución.

### TAB-P4. Papel de `model_id`

- **Qué**:
  - Con D21 (a), documentarlo en el docstring de `TabICLAdapter`: el parámetro y el atributo `model_id` (aprox. líneas 2131, 2161 y 2280 en d9b8f4f). Hay que decir que `model_id` solo selecciona el adaptador por el prefijo `soda-inria/tabicl`, que los pesos vienen siempre de `jingang/TabICL` y que el checkpoint se elige con `tabicl_config["checkpoint_version"]`. `adapter-parameters.md` línea 107 y `llms-base.txt` línea 513 ya lo dicen, y `llms-base.txt` línea 62 no necesita cambio [verificado en código].
  - Con D21 (b), el rechazo de los sufijos iría a la tabla central de ids no soportados de `02_common.md` (D-REG) o a un prefijo exacto en `_ADAPTER_REGISTRY`, con tests en `test_resolve_adapter.py`.
- **Por qué**: TAB-F7, TAB-F6 y X-3.
- **Riesgo**: ninguno con (a). (b) rompería ids con sufijo que hoy se resuelven, como `soda-inria/tabicl-v2` o `soda-inria/tabicl-regressor-v2-20260212` [probado, evidence/tabicl/resolve_adapter.log].
- **Rompe compatibilidad**: no con (a).
- **Decisión**: D21.

### TAB-P5. Textos de docstrings, guía y skill (incluye TAB-V2)

- **Qué**: corregir los textos de TAB-F11 y TAB-F18, cambiar "(0, 1)" por `[0, 1]` (TAB-V2) y añadir la nota de TAB-V1. La lista exacta de sitios está en la sección 5. Después, regenerar el contexto de IA.
- **Por qué**: TAB-F11, TAB-F18, TAB-F7, TAB-V1 y TAB-V2.
- **Riesgo**: ninguno.
- **Rompe compatibilidad**: no.
- **Decisión**: ninguna. El texto de `model_id` depende de D21 y el comando de instalación, de D20.

### TAB-P6. `FakeTabICLForecaster` fiel al backend y tests

- **Qué**: en `skforecast/foundation/tests/tests_foundation_models/fixtures_adapters.py`, `FakeTabICLForecaster.predict_df` debe:
  - validar los cuantiles como tabicl: lista de `float` en `[0, 1]`, y `ValueError("Quantiles must be a list of floats")` en otro caso;
  - poder emitir un `UserWarning` y escribir un texto de progreso en `sys.stderr`, por ejemplo con argumentos opcionales del constructor del fake (`warning_message=None`, `progress_text=None`), para probar TAB-P1.

  Los tests existentes usan niveles float, y `predict` ya pasa una lista de floats por defecto, así que no se ven afectados [verificado en código]. No hace falta que el fake descarte NaN, porque los tests de NaN viven en `FoundationModel`.
- **Por qué**: TAB-F19, TAB-F10 y TAB-F15.
- **Riesgo**: ninguno.
- **Rompe compatibilidad**: no.
- **Decisión**: ninguna.

### TAB-V1. `context_length` cuenta filas, NaN incluidos (solo documentación)

- **Qué**: solo documentación, en el parámetro `context_length` de `TabICLAdapter` y en la fila `context_length` de TabICL en `adapter-parameters.md` (línea 109). No se cambia el recorte de `FoundationModel`.
- **Por qué**: TAB-V1.
- **Riesgo**: ninguno.
- **Rompe compatibilidad**: no.
- **Decisión**: ninguna.

## 4. Tests

Tests unitarios, sin backend real:

- `skforecast/foundation/tests/tests_foundation_models/test_utils.py` (añadir los dos helpers al import de `skforecast.foundation._utils`):
  - `_suppress_backend_progress` con `show_progress=False`:
    - un warning emitido dentro se reemite al salir, con su categoría, fichero y línea;
    - el texto escrito en `sys.stderr` dentro no llega a `capsys.readouterr().err`;
    - el warning se muestra cuando `sys.stderr` ya está restaurado (ver la técnica de `showwarning` más abajo);
    - con un filtro `"ignore"` del usuario para esa categoría no se reemite nada;
    - si el cuerpo lanza una excepción, se propaga y los warnings grabados se reemiten.
  - `_suppress_backend_progress` con `show_progress=True`: el texto de stderr y el warning llegan sin cambios.
  - `_synthetic_daily_index`: mismas fechas que `pd.Timestamp("2000-01-01") + pd.Timedelta(n, unit="D")` para n=1, 5 y 200, sin `DeprecationWarning` (con `warnings.simplefilter("error", DeprecationWarning)` dentro del test).
- `skforecast/foundation/tests/tests_foundation_models/test_TabICLAdapter.py`:
  - Nuevo test de reemisión con `show_progress=False`, en el que el fake emite el `UserWarning` de NaN y escribe progreso en stderr. Ojo: `pytest.warns` pasa también sin el arreglo [probado en TabPFN, evidence/tabpfn/verification/test_p1_pytest_warns.log], y `capsys` no ve el warning porque pytest lo graba. El test debe comprobar lo que se pierde. Opción recomendada [inferido]: dentro de `warnings.catch_warnings()`, poner `warnings.simplefilter("always")` y sustituir `warnings.showwarning` por una función que guarde el `sys.stderr` activo en el momento de la llamada. Después, comprobar que se llamó una vez y que ese stderr no era el `io.StringIO` de la redirección (sin el arreglo se llama con el `StringIO` activo). Alternativa: un subproceso que ejecute la predicción con el fake y compruebe su stderr, como hace `evidence/tabicl/stderr_redirect_check.py`.
  - Test de que el texto de progreso sigue oculto con `show_progress=False` y visible con `True` (`capsys`).
  - Test con niveles `[0, 0.5, 1]` y `np.float32` a través de `FoundationModel`, con el fake inyectado (`model=FakeTabICLForecaster()`): el fake recibe `float` y las columnas son `q_0`, `q_0.5` y `q_1`. Con D-Q (b), añadir el test equivalente que llama directamente a `adapter.predict`.
  - Ampliar `test_TabICLAdapter_predict_range_index_timestamps_are_datetime` (aprox. línea 624 en d9b8f4f) para comprobar las fechas exactas del horizonte (2000-01-01 + n días en adelante) y que no hay `DeprecationWarning`. Hoy ese test y `test_TabICLAdapter_predict_range_index_issues_warning_and_returns_arrays` emiten el aviso [probado, evidence/tabicl/unit_tests.log].
  - Con D20, adaptar `test_TabICLAdapter_predict_ImportError_when_tabicl_not_installed` (aprox. línea 700) para comprobar el mensaje entrecomillado con el mínimo. El `match="tabicl\\[forecast\\]"` actual seguiría pasando [verificado en código], así que hay que endurecerlo. Con D-VER (b), añadir un test con `importlib.metadata.version` simulado a `"2.0.2"` que espere el error claro.
- `skforecast/foundation/tests/tests_foundation_models/test_get_model_info.py` (aprox. línea 169, `("TabICLAdapter", 4096, "tabicl[forecast]", True, False, False)`): con D20, actualizar al nuevo `backend_package`.
- `skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py`: sin cambios; debe seguir pasando con `adapters.toml` actualizado.
- `skforecast/foundation/tests/tests_foundation_models/test_TabPFNAdapter.py`: sin cambios en este PR; debe seguir pasando.

Compatibilidad (`tools/foundation_compat/`):

- `tools/foundation_compat/tests/contracts.py` (entrada `"tabicl"`): sin cambios, porque las firmas no cambian entre 2.0.3 y 2.2.0 [verificado en código].
- `tools/foundation_compat/adapters.toml`: con D20, `backend_package` de `[tabicl]` igual al del adaptador.
- Smoke: sin cambios obligatorios. El test genérico de orden de cuantiles va en `10_docs_ci.md` (sección 3.6); el fake fiel al backend es TAB-P6 aquí. Las demás invariantes genéricas del smoke (XC-7 de `evidence/cross_cutting.md`) quedan fuera de esta serie de PR (`10_docs_ci.md`, sección 8); si se quieren, van en un PR aparte `chore/foundation-compat-checks` (PR-12 de `evidence/cross_cutting.md`), después de 01 y 03.
- Comprobación manual del mínimo, porque no hay CI de versiones mínimas. Con D20 (a), recrear un entorno con 2.0.3 y correr contrato y smoke. Con (b), el mínimo coincide con la versión que ya prueba el CI. Para comprobar D-VER (b), usar un entorno con 2.0.2:

  ```bash
  uv venv ~/.venvs/fm-review-tabicl-2.0.2 --python 3.12
  uv pip install --python ~/.venvs/fm-review-tabicl-2.0.2/bin/python --torch-backend cpu -e . pytest "tabicl[forecast]==2.0.2"
  ~/.venvs/fm-review-tabicl-2.0.2/bin/python -m pytest tools/foundation_compat/tests/test_contract.py --backend tabicl -q -p no:cacheprovider
  ```

  En la revisión, ese contrato dio 1 failed y 2 passed [probado, evidence/tabicl/verification/old_backend_2.0.2.log]. Borrar el entorno al terminar.

Evidencias a re-ejecutar antes y después del cambio, con el entorno de revisión:

```bash
~/.venvs/fm-review-tabicl/bin/python dev/foundation_adapters_review/evidence/tabicl/stderr_redirect_check.py
~/.venvs/fm-review-tabicl/bin/python dev/foundation_adapters_review/evidence/tabicl/verification/verify_tabicl.py
~/.venvs/fm-review-tabicl/bin/python dev/foundation_adapters_review/evidence/tabicl/capabilities_check.py
~/.venvs/fm-review-tabicl/bin/python dev/foundation_adapters_review/evidence/tabicl/compare_direct.py
~/.venvs/fm-review-tabicl/bin/python dev/foundation_adapters_review/evidence/tabicl/resolve_adapter.py
```

El backtest de la guía (`dev/foundation_adapters_review/evidence/common/backtest_guide.py`) no aplica, porque ningún cambio de este PR altera predicciones:

- TAB-P2 da resultados idénticos a los de niveles float [probado, evidence/tabicl/proto_fixes.log, evidence/tabicl/verification/verify_tabicl.log];
- TAB-P1 no toca entradas ni salidas [inferido];
- el helper de fechas da las mismas fechas [probado con n=5];
- TAB-P3 no cambia nada con 2.2.0 [inferido].

`compare_direct.py` basta como comprobación de corrección.

## 5. Documentación y contexto de IA

Docstrings en `skforecast/foundation/_adapters.py` (docstring de clase y de `__init__` de `TabICLAdapter`; líneas aproximadas en d9b8f4f):

- `model_id` (líneas 2131, 2161 y 2280): sustituir "HuggingFace model ID" por el texto de TAB-P4: selecciona el adaptador, los pesos vienen de `jingang/TabICL` y el checkpoint se elige con `tabicl_config["checkpoint_version"]`.
- `context_length` (líneas 2136 y 2285): añadir que cuenta filas, NaN incluidos, y que tabicl descarta después las filas NaN, así que el modelo puede ver menos observaciones (TAB-V1).
- `temporal_features` (líneas 2149 y 2298): aceptar instancias de `TimeTransform` o, con tabicl >=2.1, nombres registrados como `"index"` o `"datetime"` [probado]. Antes de enumerar más nombres, consultar el registro de transformaciones de tabicl. Avisar de que `"fourier"` no funciona en tabicl 2.2.0 (issue upstream #172) y que hay que pasar una instancia de `FourierEncoder` con `periods` [inferido del TypeError "missing 1 required positional argument: 'periods'"]. Con D20 (b), quitar la condición "tabicl >=2.1".
- `show_progress` (líneas 2154 a 2156 y 2302 a 2305): describir la barra como en `TabPFNAdapter` (`Predicting time series: ...` en CPU, `GPU 0: ...` en GPU) y añadir que los warnings del backend se muestran igualmente.
- Atributo `SUPPORTED_QUANTILES` (líneas 2194 a 2196): cambiar "(0, 1)" por `[0, 1]` (TAB-V2).
- Notes (línea 2231): mantener "NaN values in the future covariates are accepted by TabICL with a warning", que con TAB-P1 pasa a ser cierto también con `show_progress=False`.
- Docstring de `FoundationModel` (`_foundation_model.py`, líneas 121 a 125, entrada TabICLv2): sin cambios obligatorios.

Guía `docs/user_guides/foundation-forecasting-models.ipynb`. Solo se edita el texto de TabICL y no se re-ejecuta el notebook; eso lo hace `10_docs_ci.md`.

- Tabla HTML "Supported Foundation Models" (celda 11 en la revisión, columnas "Default context length", "Max context length", "Max horizon", "Install command"), fila TabICL:
  - "Max context length" pasa de `4096` a "No hard limit", el mismo estilo que su "Max horizon" ("No hard limit, set via `steps`"). "Default context length" se mantiene en 4096.
  - Con D20, el `<code>`, el `data-copy` y el `aria-label` del botón de instalación pasan a `pip install "tabicl[forecast]>=2.2.0"` (o `>=2.0.3`).
- Sin cambios en este PR: la sección TabICL (celdas 71 a 76 en la revisión: `FoundationModel(model_id="soda-inria/tabicl", context_length=500, show_progress=False)`, fit, predict, predict_interval y backtesting con MAE 215.802569), la celda 45 (influencia de `context_length`, que incluye `soda-inria/tabicl` entre los modelos con exog) y la tabla comparativa (celda 108).

Skill y referencias:

- `skills/foundation-forecasting/SKILL.md`:
  - Línea 59 (`pip install tabicl[forecast]       # For TabICL`): entrecomillar, como las líneas de TimesFM y TabPFN-TS, y añadir el mínimo de D20 si se decide.
  - Línea 284: quitar TabICL de los adaptadores en los que `context_length` recarga el modelo. Solo se recrea `TabICLForecaster`, y los pesos se cargan en cada llamada en cualquier caso.
  - Línea 159: agrupa TabICL con los adaptadores que aceptan "any quantile in `(0, 1)`", pero para TabICL es `[0, 1]`. Ajustarlo solo si se puede sin cambiar lo que dice de los demás adaptadores.
- `skills/foundation-forecasting/references/adapter-parameters.md`:
  - Línea 103 ("**Quantiles**: any value in `(0, 1)`" de TabICL): cambiar a `[0, 1]` (TAB-V2).
  - Línea 107 (`model_id`): ya es correcta. Se puede añadir `tabicl_config["checkpoint_version"]` para ser explícito.
  - Línea 109 (`context_length`): añadir la nota de TAB-V1.
  - Línea 112 (`temporal_features`): el mismo texto que el docstring.
  - Línea 113 (`show_progress`): añadir que los warnings del backend se muestran igualmente.
  - Línea 219 (tabla de `set_params`, columna "Changing these forces a model reload"): para TabICL, aclarar que esos parámetros recrean `TabICLForecaster`, pero que ninguno recarga pesos de forma distinta, porque los pesos se cargan en cada llamada [verificado en código].
  - Líneas 230 y 231: quitar TabICL de la lista de "reloads the model" y de los ejemplos de recarga, y añadirlo a la lista de "free".
- `tools/ai/llms-base.txt`:
  - Línea 492: "Any in `(0, 1)`" pasa a "Any in `[0, 1]`" en la fila de TabICL (TAB-V2).
  - Línea 458: lista los paquetes sin versión; cambiarla solo si se añaden mínimos a todos (fuera de este PR).
  - Líneas 62 y 513: sin cambios.
- Regenerar con el skill `ai-context-sync`: `python tools/ai/generate_ai_context_files.py` y `python tools/ai/generate_ai_context_files.py --check`. No editar a mano `AGENTS.md`, `llms-full.txt`, `docs/llms*.txt` ni `.github/copilot-instructions.md`.

Entradas de release notes en `docs/releases/releases.md`, sección 0.27.0 (o la versión en desarrollo), redactadas con el skill `release-note`. Texto propuesto:

**Fixed**:

```markdown
+ <code>[TabICLAdapter]</code> with `show_progress=False`, the default, hid the warnings of tabicl (for example about missing values in the future covariates) together with its progress bar. The progress bar is still hidden and the warnings are shown. ([#XXXX](https://github.com/skforecast/skforecast/pull/XXXX))

+ <code>[TabICLAdapter]</code> raised `ValueError: Quantiles must be a list of floats` with integer quantile levels (`quantiles=[0, 0.5, 1]`) or `numpy.float32` levels. ([#XXXX](https://github.com/skforecast/skforecast/pull/XXXX))

+ <code>[TabICLAdapter]</code> no longer raises a NumPy `DeprecationWarning` (generic timedelta unit) when the series have a `RangeIndex`. ([#XXXX](https://github.com/skforecast/skforecast/pull/XXXX))
```

La segunda entrada solo va aquí si D-Q es (b). Con (a), la escribe `02_common.md` para todos los adaptadores.

**Changed** (solo si se decide D20; ajustar la versión a `>=2.0.3` con la opción (a)):

```markdown
+ <code>[TabICLAdapter]</code> requires `tabicl[forecast]>=2.2.0`: install it with `pip install "tabicl[forecast]>=2.2.0"`. Older versions are rejected with a clear error. ([#XXXX](https://github.com/skforecast/skforecast/pull/XXXX))
```

La frase "Older versions are rejected with a clear error" solo va si se adopta D-VER (b). Con D20 (b), añadir también una línea en "Before upgrading", bajo "And some code needs to be updated":

```markdown
+ <code>[TabICLAdapter]</code> needs `tabicl[forecast]>=2.2.0`: upgrade it with `pip install -U "tabicl[forecast]>=2.2.0"`.
```

## 6. Decisiones pendientes

| Id | Decisión | Opciones | Recomendación | Impacto en el plan | Estado |
|---|---|---|---|---|---|
| D20 | Mínimo de TabICL (TAB-P3) | (a) `tabicl[forecast]>=2.0.3`, el mínimo real probado; (b) `>=2.2.0`, la única versión que prueba el CI, sin gluonts y con la corrección de `DatetimeEncoder`; (c) solo documentar | (b). Un error claro para las instalaciones existentes exige además la comprobación en tiempo de ejecución (D-VER). (a) si se prioriza la compatibilidad | Cambia `backend_package`, `adapters.toml`, `test_get_model_info.py`, el test del ImportError, el comando de la guía (celda 11), `SKILL.md` línea 59 y las release notes ("Changed" y, con (b), "Before upgrading") | pendiente |
| D21 | `model_id` de TabICL (TAB-P4) | (a) documentar; (b) aceptar solo el id exacto `soda-inria/tabicl`; (c) mapear ids a `checkpoint_version` | (a), porque solo existe un checkpoint de regresión | Con (a), solo docstring. Con (b), tabla de ids de `02_common.md` (D-REG) o prefijo exacto y tests en `test_resolve_adapter.py`; rompe ids con sufijo | pendiente |
| D-Q | Tipo de los niveles de cuantil (TAB-P2, X-2); se decide en `02_common.md` | (a) convertir a float en `FoundationModel`; (b) solo en `TabICLAdapter` | (a), manteniendo los nombres de columna pedidos (`02_common.md`, 3.3). La conversión no corrige los nombres con `float32` [probado, `02_common.md`, sección Descartado] | Con (a), este PR solo añade el fake y un test de extremo a extremo. Con (b), cambia `TabICLAdapter.predict` y la entrada de release notes va aquí | pendiente |
| D-VER | Política de versión mínima (X-4); se decide en `02_common.md` | (a) mínimo solo en `backend_package` (cubre instalaciones nuevas y CI); (b) además, helper común de comprobación en tiempo de ejecución (`importlib.metadata`, patrón de `TimesFM3Adapter`); (c) solo docs | (b), con un helper común pequeño | Con (b), `TabICLAdapter._load_model` llama al helper, se añade un test con versión simulada y la release note dice "rejected with a clear error" | pendiente |

## 7. Criterios de aceptación

- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_TabICLAdapter.py -q`: pasan todos (50 en la revisión más los nuevos). El resumen de warnings ya no incluye `DeprecationWarning` de `_adapters.py` (en la revisión había 4 warnings: 2 `InputTypeWarning` y 2 `DeprecationWarning`).
- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_TabICLAdapter.py -q -W "error::DeprecationWarning"`: pasa. Antes del cambio fallan los dos tests de `RangeIndex` con numpy 2.5 [inferido de evidence/tabicl/unit_tests.log].
- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_utils.py -q`: pasan, incluidos los tests de los dos helpers.
- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_get_model_info.py skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py skforecast/foundation/tests/tests_foundation_models/test_TabPFNAdapter.py -q`: pasan.
- [ ] Contrato y smoke en el entorno de revisión: `~/.venvs/fm-review-tabicl/bin/python -m pytest tools/foundation_compat/tests/test_contract.py --backend tabicl -q -p no:cacheprovider` da 3 passed, y lo mismo con `test_smoke.py` da 6 passed.
- [ ] `python tools/foundation_compat/run.py --adapter tabicl`: `"held_back": false` y `"status": "passed"`. No añadir el directorio `results/` al commit.
- [ ] `stderr_redirect_check.py`: en la sección `show_progress=False`, la salida contiene "UserWarning: Some covariate values in future_tsdf are missing (NaN)" y no contiene "Predicting time series" (en la revisión solo imprimía "done"). La sección `show_progress=True` queda igual que en evidence/tabicl/stderr_redirect_check.log.
- [ ] `verify_tabicl.py`:
  - en "F15 NaN in future exog, show_progress=False", el warning aparece en stderr y la línea dice "tqdm bar on stderr=False";
  - en "F10/P2", los niveles `int [0, 0.5, 1]` y `np.float32` dan OK cuando TAB-P2 esté aplicado (aquí o en 02);
  - en "F8", "max_abs_diff (same 64-row window incl. 10 NaN): 0.0".
- [ ] `capabilities_check.py`: las líneas `[quantiles=[0, 0.5, 1] (int levels)]` y `[quantiles as numpy float32]` pasan de ERROR a OK (con TAB-P2 aplicado). El resto de líneas queda igual que en evidence/tabicl/capabilities_check.log.
- [ ] `compare_direct.py`: casos A y C con `max_abs_diff=0.000e+00` frente a la llamada directa. Caso B con `allclose=True` y diferencia máxima de 0.0 o del orden de 3.8e-6 (ruido float32 visto en la verificación).
- [ ] `resolve_adapter.py`: con D21 (a), igual que evidence/tabicl/resolve_adapter.log.
- [ ] Con D-VER (b), en el entorno con tabicl 2.0.2, `FoundationModel("soda-inria/tabicl")` con `fit` y `predict` lanza un error que nombra la versión instalada, el mínimo y `pip install -U "tabicl[forecast]>=..."`, en lugar del TypeError de `point_estimate`.
- [ ] `python tools/ai/generate_ai_context_files.py --check` termina sin errores tras regenerar.
- [ ] Entradas añadidas en `docs/releases/releases.md`.
- [ ] Skill `verify` en verde.

## 8. Fuera de alcance

- Aplicar los dos helpers a `TabPFNAdapter` (TPF-P1, TPF-P5 y TPF-V4 en TabPFN) y quitar entonces los imports `contextlib` e `io` de `_adapters.py`: `08_tabpfn.md`.
- La conversión central de cuantiles (si D-Q es (a)), la comprobación central de contexto todo NaN, la tabla central de ids no soportados (D-REG) y el helper de versión (D-VER): `02_common.md`.
- Re-ejecutar la guía (celdas 71 a 76 y 108; el MAE de TabICL en la revisión es 215.802569), el test genérico de orden de cuantiles (sección 3.6) y cualquier cambio del CI de compatibilidad: `10_docs_ci.md`. La prueba de versiones mínimas (TAB-F19) la deja fuera `10_docs_ci.md` (sección 8) y queda ligada a D-VER. Las demás invariantes genéricas del smoke (XC-7 de `evidence/cross_cutting.md`) quedan fuera de esta serie de PR (`10_docs_ci.md`, sección 8); si se quieren, van en un PR aparte `chore/foundation-compat-checks` (PR-12 de `evidence/cross_cutting.md`), después de 01 y 03.
- Cambiar el recorte de `context_length` en `FoundationModel` para que cuente observaciones válidas (TAB-V1). Aquí solo se documenta; cambiarlo sería una decisión común a todos los adaptadores.
- Exponer `device` o `num_workers` (TAB-F12, TAB-F17): `device` ya se alcanza vía `tabicl_config` y el backend no expone `num_workers`.
- Documentar o usar el alias `"fourier"` hasta que upstream publique el PR #174.
- Cambiar el import a `from tabicl import TabICLForecaster` (TAB-F3): no hace falta.
- El `stacklevel=3` fijo del warning de `RangeIndex` (XC-8 de la revisión transversal): fuera de este plan.
- El comportamiento real en CUDA, XPU y MPS: no se puede verificar sin GPU.
- Benchmarks de velocidad o de precisión entre versiones de tabicl o entre checkpoints, por ejemplo el efecto de la corrección de `DatetimeEncoder` de 2.2.0: fuera del alcance de skforecast.
