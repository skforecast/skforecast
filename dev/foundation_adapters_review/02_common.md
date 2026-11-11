# 02. Cambios comunes en FoundationModel y el registro

- **Rama propuesta**: `fix/foundation-common-guards`, creada desde `origin/0.27.x`. Si se prefiere dividir en dos PR (ver "División opcional en dos PR" al final de la sección 3): `fix/foundation-common-guards` (guardas: X-1, X-2, X-3, X-8) y `chore/foundation-backend-version-helper` (helpers: X-4, X-6), las dos desde `origin/0.27.x`.
- **Prioridad**: alta para X-1 (predicciones absurdas sin aviso en al menos tres backends); media para X-2, X-3 y X-4; baja para X-6 y X-8.
- **Adaptadores afectados**: todos, a través del código común (`FoundationModel`, `_resolve_adapter`, `_utils.py`). No se edita el código propio de ningún adaptador: la tabla de ids no soportados vive junto a `_ADAPTER_REGISTRY` y el único cambio en `_adapters.py` fuera de ella es la firma de `_resolve_torch_device`.
- **Ficheros afectados**:
  - `skforecast/foundation/_foundation_model.py` (`FoundationModel.predict` y su docstring).
  - `skforecast/foundation/_adapters.py` (`_ADAPTER_REGISTRY`, `_resolve_adapter`, tabla nueva `_UNSUPPORTED_MODEL_IDS`, `import re`, firma de `_resolve_torch_device`).
  - `skforecast/foundation/_utils.py` (helpers nuevos y `_warn_if_non_commercial`).
  - `skforecast/model_selection/_validation.py` (solo un comentario).
  - `skforecast/utils/utils.py` y `skforecast/utils/__init__.py` (helper de `stacklevel` trasladado desde `model_selection/_utils.py` e `import os`, 3.5; si se divide en dos PR, 3.5 va en `chore/foundation-backend-version-helper`).
  - `skforecast/model_selection/_utils.py` (solo los imports del helper trasladado, 3.5).
  - Tests de `skforecast/foundation/tests/tests_foundation_models/` y `skforecast/foundation/tests/tests_forecaster_foundation/test_predict.py`, y test nuevo en `skforecast/utils/tests/tests_utils/` (3.5).
  - `docs/user_guides/foundation-forecasting-with-heterogeneous-series.ipynb` (solo celdas markdown), `skills/foundation-forecasting/SKILL.md`, `tools/ai/llms-base.txt`, `docs/releases/releases.md`.
- **Depende de**: ninguno. Es independiente de 01 (T0), que es urgente y puede fusionarse antes o después (ver 3.2 y 3.4).
- **Bloquea a**:
  - 04 a 09: usan el helper de versión de 3.4 para declarar y comprobar su mínimo (si D-VER se resuelve con la opción (b)); 06 usa además el helper de `stacklevel` de 3.5 para el aviso MPS de Moirai (MOI-P6).
  - 04 (T25-P3) y 09 (NORI-P1): sus ids no soportados quedan resueltos aquí (3.2); con D9 opción A el cambio de prefijo de TimesFM 2.5 pasa a 04.
  - 07: TAB-P2 queda resuelto aquí (3.3) si D-Q se resuelve con (a).
  - 03 (de forma débil): recibe los niveles de cuantil ya convertidos a `float` para su traducción a `Decimal` (TSI-P3) y rebasa sobre el texto de X-8 (mismas líneas de `llms-base.txt` y `SKILL.md`).
  - 10: re-ejecuta la guía principal y actualiza el CI de compatibilidad con todo integrado.
  - 01 no queda bloqueado (ver 3.2 y 3.4); solo comparte con 3.6 la línea 130 de `SKILL.md` y la 531 de `llms-base.txt`.
- **Cierra issue**: ninguna de skforecast. Relacionadas upstream: amazon-science/chronos-forecasting#544 (abierta) y el PR google-research/timesfm#486 (abierto); la comprobación central de 3.1 protege a los usuarios de skforecast sin esperar a que se publiquen [inferido].

## 1. Estado en la revisión (2026-11-06)

**Base.** Rama desde `origin/0.27.x`, commit `d9b8f4f`, skforecast 0.27.0 en desarrollo [verificado en código]. Las sondas del código común se ejecutaron con Python 3.12.3, numpy 2.5.4, pandas 2.3.3 y el fake de Chronos-2 de los tests (`FakePipeline` de `skforecast/foundation/tests/tests_foundation_models/fixtures_adapters.py`), sin pesos (evidence/cross_cutting/probe_shared.log) [probado].

**Versiones de backend implicadas en los hallazgos de este plan** (todas eran la última publicada el 2026-11-06):

| Backend | Versión probada | Última | Python y torch de la prueba | Relevancia para este plan |
|:--|:--|:--|:--|:--|
| chronos-forecasting | 2.3.2 | 2.3.2 (2026-09-08) | Python 3.12.3, torch 2.15.0+cpu | Contexto todo NaN: cuantiles de ±1e5 (issue upstream #544 abierta) [probado] |
| uni2ts | 2.0.0 | 2.0.0 (2025-11-04) | Python 3.12.3, torch 2.4.1+cpu, numpy 1.26.4 (uni2ts exige Python <= 3.12) | Contexto todo NaN: cuantiles en torno a 0 [probado] |
| timesfm | 3.0.2 | 3.0.2 (2026-09-09) | Python 3.12.3; numpy 2.5.4, y 1.26.4, 2.1.3 y 2.2.0 en la prueba de numpy | Contexto todo NaN: `ValueError` críptico con numpy >= 2.2, ceros con numpy < 2.2 (PR upstream #486 abierto) [probado] |
| tabicl | 2.2.0 | 2.2.0 (2026-09-02) | Python 3.12.3, torch 2.15.0+cpu | Rechaza niveles de cuantil `int` y `numpy.float32` [probado] |
| synthefy-nori | 0.23.0 | 0.23.0 (2026-11-04) | Python 3.12.3, torch 2.13.0+cpu | `Synthefy/Nori-Causal` cae en el prefijo [probado] |
| tfc-t0 | 0.6.0 | 0.6.0 (2026-10-13) | Python 3.12.3, torch 2.15.0+cpu | Los 4 repos ONNX caen en el prefijo [probado] |

skforecast admite numpy >= 1.26.1 y Python >= 3.10 [verificado en código: `pyproject.toml`]. Esto importa en dos puntos: el comportamiento de TimesFM 2.5 con un contexto todo NaN cambia con numpy 2.2 (TV-2), y `warnings.warn(skip_file_prefixes=...)` exige Python 3.12, por lo que no sirve para X-6 [documentado].

**Repos del Hub que entran en la tabla de ids no soportados** (evidence/nori/hub_info.log, evidence/t0/hub_info.log, evidence/timesfm25/hub_google_timesfm.log) [probado]:

| Repo | sha | Última modificación | Motivo |
|:--|:--|:--|:--|
| `Synthefy/Nori-Causal` | afdc891 | 2026-10-25 | Modelo de descubrimiento causal (fichero `noricausal_31m.pt`, sin `nori.pt`; `model_type` noricausal) |
| `theforecastingcompany/t0-alpha-onnx-fp16` | 0490724 | 2026-10-18 | `library_name` onnxruntime, otro esquema de `config.json` |
| `theforecastingcompany/t0-alpha-onnx-int8` | 95c0ad0 | 2026-10-18 | idem |
| `theforecastingcompany/t0-beta-onnx-fp16` | feac67c | 2026-10-18 | idem |
| `theforecastingcompany/t0-beta-onnx-int8` | e6ae677 | 2026-10-18 | idem |
| `google/timesfm-2.5-200m-flax` | b99e9f6 | 2025-09-02 | Checkpoint orbax sin `model.safetensors`: la carga da 404 |
| `google/timesfm-2.5-200m-transformers` | 5a9806b | 2026-04-10 | Formato transformers: 272 tensores con otros nombres, faltan 228 de los 232 que espera el módulo |

Los ids que sí deben seguir resolviendo: `Synthefy/Nori` (1aca005), `Synthefy/Nori-1-5` (79afc5a), `Synthefy/Nori-30M` (63c9f7f), `Synthefy/Nori-100M` (467ae97), `theforecastingcompany/t0-alpha` (7419278), `theforecastingcompany/t0-beta` (941e522), `theforecastingcompany/t0-gamma` (e08aee0), `google/timesfm-2.5-200m-pytorch` (1d95242) [probado].

**Línea base del CI.** Run 42212742430 del workflow Foundation Backends Compatibility (rama 0.27.x, 2026-11-06): todos los jobs en verde salvo T0 (job 127875169126, por T0-F1 y T0-F2, que resuelve el plan 01) [probado]. El CI de compatibilidad no ejercita ninguno de los casos de este plan: el smoke usa contextos sin NaN final, niveles `float` e ids válidos [verificado en código: `tools/foundation_compat/tests/test_smoke.py`].

**Antes de empezar.** Este plan puede quedar desfasado. Comprobar desde la raíz del repositorio:

```bash
# Repos nuevos que caigan en un prefijo registrado y no sean forecasters
python dev/foundation_adapters_review/evidence/common/hub_info.py Synthefy --prefix Nori
python dev/foundation_adapters_review/evidence/common/hub_info.py theforecastingcompany --prefix t0
python dev/foundation_adapters_review/evidence/common/hub_info.py google --search timesfm
python dev/foundation_adapters_review/evidence/common/hub_info.py Salesforce --prefix moirai-2
# Versiones nuevas de los backends cuyos fallos motivan las guardas
python dev/foundation_adapters_review/evidence/common/pypi_info.py chronos-forecasting
python dev/foundation_adapters_review/evidence/common/pypi_info.py timesfm
python dev/foundation_adapters_review/evidence/common/pypi_info.py uni2ts
python dev/foundation_adapters_review/evidence/common/pypi_info.py tabicl
python dev/foundation_adapters_review/evidence/common/pypi_info.py synthefy-nori
python dev/foundation_adapters_review/evidence/common/pypi_info.py tfc-t0
# Estado actual del código común (debe reproducir evidence/cross_cutting/probe_shared.log)
~/.venvs/fm-review-chronos/bin/python dev/foundation_adapters_review/evidence/cross_cutting/probe_shared.py
```

(Para Google se usa `--search timesfm` y no `--prefix`, porque el listado por autor tiene un límite de 500 repos.)

Qué hacer si algo cambió:
- Un repo nuevo que cae en un prefijo y no carga (otro modelo causal, otra exportación ONNX o flax): añadirlo a `_UNSUPPORTED_MODEL_IDS` con su motivo y a los tests parametrizados. Un checkpoint nuevo que sí es forecaster y coincide con un patrón de la tabla: estrechar el patrón.
- Una release de chronos-forecasting o timesfm que corrija el contexto todo NaN: la comprobación central se mantiene, porque cubre a todos los adaptadores y a las versiones antiguas [inferido].
- Una release de tabicl que acepte `int`: la conversión a `float` se mantiene por las versiones antiguas y porque es inocua [inferido].
- Si `probe_shared.py` ya no reproduce el log (alguien arregló algo en 0.27.x), revisar qué partes de este plan siguen vigentes antes de implementar.

Recrear los entornos: evidence/README.md (sección "Recrear los entornos").

## 2. Hallazgos

Equivalencia de ids. Este plan usa las ids X-n del informe (evidence/diagnosis_report.md, sección 3.10) y las decisiones de su sección 6. La revisión transversal (evidence/cross_cutting.md) usa XC-n con otra numeración: X-1 = XC-1, X-2 = XC-5 (parte de tipos), X-3 = XC-3, X-4 = XC-2, X-5 = XC-9, X-6 = XC-8, X-7 = XC-4, X-8 = XC-6. Sus decisiones también llevan otra numeración: D1 de cross_cutting.md es D11 aquí, D7 de cross_cutting.md es D-REG, y D6 de cross_cutting.md contiene D-VER más los mínimos por backend. Atención: D11 de cross_cutting.md es otra decisión (`padding_mode` de TimesFM 3, D6 del informe, propiedad de 04).

### Hallazgos propios de este plan

- **X-1** [probado] Una ventana de contexto toda NaN llega al backend por el camino del usuario sin ningún aviso de skforecast. `FoundationModel` solo rechaza las series enteramente NaN (`check_preprocess_series`, `skforecast/utils/utils.py`, aprox. línea 4445 en d9b8f4f) y después recorta con `iloc[-self.context_length:]` (`FoundationModel._check_preprocess_context`, aprox. línea 591 en d9b8f4f) sin comprobar que quede algún valor observado [verificado en código]. Con `context_length=50` y las últimas 80 observaciones NaN, `predict` se ejecuta y el backend recibe un contexto todo NaN (evidence/cross_cutting/probe_shared.log, sección 1). Resultado por backend:
  - Chronos-2: q0.1 entre -94680 y -57279 y q0.9 entre 58444 y 86309, sin warnings, igual con una serie, con dos series y con el contexto guardado en `fit` (CHR-F4; evidence/chronos/all_nan_context.log).
  - Moirai-2: cuantiles en torno a 0, por ejemplo q0.5 de 0.000012 en el primer paso (MOI-F12; evidence/moirai/all_nan_context.log).
  - TimesFM 2.5: falla o predice ceros según numpy (TV-2).
  - TabICL y TabPFN-TS: con como mucho un valor válido rellenan con 0 (`_impute_missing_targets` de tabicl v2.2.0, `_handle_missing_values` de tabpfn-time-series v1.4.0) [verificado en código].

  El comentario de `skforecast/model_selection/_validation.py` (aprox. líneas 2874-2882 en d9b8f4f) afirma que "an all-NaN context is rejected on the user-facing path", lo cual es falso [verificado en código]. `backtesting_foundation` sí salta esas series en cada fold (filtro `level_series.notna().any()`, aprox. línea 2895 en d9b8f4f) [verificado en código; test `test_output_backtesting_foundation_series_skipped_when_context_window_all_nan` en `skforecast/model_selection/tests/tests_validation/test_backtesting_foundation.py`]. La celda 2 de `docs/user_guides/foundation-forecasting-with-heterogeneous-series.ipynb` dice "A context window that is entirely NaN is rejected" [verificado en código].
- **TV-2** [probado] Con TimesFM 2.5, si el contexto recortado de una serie es todo NaN, `linear_interpolation` de timesfm hace `if non_nans_values:` sobre un array vacío: con numpy >= 2.2 todo el `predict` falla con "ValueError: The truth value of an empty array is ambiguous" (también para las demás series de la llamada); con numpy < 2.2 (1.26.4 y 2.1.3 probados) rellena con ceros y predice `[0. 0. 0. ...]` sin aviso de skforecast (numpy solo emite un `DeprecationWarning`). PR upstream #486 abierto. La verificación recomienda la comprobación central en `FoundationModel`, no en el adaptador. Evidencia: evidence/timesfm25/verification/verify_all_nan_context.log, evidence/timesfm25/verification/verify_all_nan_old_numpy.log, evidence/timesfm25/verification/verification.md.
- **X-2** [probado] `FoundationModel.predict` solo comprueba `0 <= q <= 1` (aprox. líneas 1135-1145 en d9b8f4f) y reenvía los niveles con su tipo: `[0, 0.5, 1]` llega al backend como `['int', 'float', 'int']` y `[np.float32(0.1), np.float32(0.9)]` como `['float32', 'float32']` (evidence/cross_cutting/probe_shared.log, sección 2). tabicl rechaza ambos con "ValueError: Quantiles must be a list of floats" (TAB-F10; evidence/tabicl/capabilities_check.log, probes "quantiles=[0, 0.5, 1] (int levels)" y "quantiles as numpy float32"); `numpy.float64` pasa por ser subclase de `float` (evidence/tabicl/verification/verification.md). Con `float(q)` las predicciones son idénticas a pedir niveles `float` (evidence/tabicl/proto_fixes.log: "equal to float levels: True"). Con `float32` las columnas salen como `q_0.10000000149011612`; ese nombre lo produce el f-string de `FoundationModel.predict` (`col_names`, aprox. línea 1310 en d9b8f4f), no el backend (evidence/tabicl/verification/verification.md).
- **X-3** [probado] Los prefijos del registro aceptan ids que no son forecasters cargables. `_resolve_adapter` (aprox. línea 5914 en d9b8f4f) devuelve el primer prefijo de `_ADAPTER_REGISTRY` con `startswith` [verificado en código]. Resuelven a un adaptador: `Synthefy/Nori-Causal`, `theforecastingcompany/t0-alpha-onnx-int8`, `google/timesfm-2.5-200m-flax`, `google/timesfm-2.5-200m-transformers`, `google/timesfm-3.0-flax`, `soda-inria/tabicl-v1-does-not-exist`, `priorlabs/tabpfn-v2`, `Salesforce/moirai-2.0-R-large` y `taharnbl/TS-ICL-v2` (evidence/cross_cutting/probe_shared.log, sección 3). Consecuencia por id:
  - `Synthefy/Nori-Causal`: `FoundationModel` se construye y `predict` falla con RemoteEntryNotFoundError 404, porque no hay `nori.pt` (NORI-F5; evidence/nori/hub_info.log) [probado].
  - Los 4 repos ONNX de T0: `get_model_info` los resuelve a `T0Adapter` con licencia Apache-2.0, y al cargar, el `TypeError` de `from_pretrained` se convierte en un `OSError` que dice que la configuración "could not be downloaded", aunque sí se descarga (T0-F7; evidence/t0/registry_and_loading_060.log, evidence/t0/check_load_errors_060.log, evidence/t0/verification/load_errors.log) [probado].
  - `google/timesfm-2.5-200m-flax`: RemoteEntryNotFoundError 404 [probado]. `-transformers`: faltan 228 de los 232 tensores [probado]; el fallo de `load_state_dict(strict=True)` tras descargar 925 MB está [verificado en código], no ejecutado (T25-F8; evidence/timesfm25/registry_and_loading.log, evidence/timesfm25/verification/verify_registry_checkpoints.log).
  - `google/timesfm-3.0-flax` y `Salesforce/moirai-2.0-R-large` (o `-base`) no existen en el Hub y fallan con un 404 propio del Hub (TFM3-F6, MOI-F5) [probado].
  - TabICL, TabPFN-TS y TS-ICL: el `model_id` es solo una etiqueta y se cargan los pesos fijos del backend (TAB-F7, TPF-F12, TSI-F11) [probado].
  - Las entradas de `_MODEL_LICENSES` (`skforecast/foundation/_utils.py`) se buscan por prefijo más largo (`_get_model_license`), así que una tabla de exclusiones o un prefijo de registro más estrecho no les afecta (XC-3 de cross_cutting.md; verificación de T25-P3) [verificado en código].
- **X-4** [probado / verificado en código] Solo `TabPFNAdapter` declara un mínimo en `backend_package` (`"tabpfn-time-series>=1.3"`; evidence/cross_cutting/probe_shared.log, sección 4). Un mínimo ahí no protege instalaciones existentes: `backend_package` solo se usa en el mensaje del `ImportError`, en `_build_model_info` (`_model_info.py`, aprox. línea 173; alimenta `get_model_info` y `list_adapters`), en `tools/foundation_compat/run.py` (que lo lee de `adapters.toml`) y en los tests de coherencia con `adapters.toml` (evidence/nori/verification/verification.md, NORI-P3 corregido; evidence/tabicl/verification/verification.md, TAB-P3) [verificado en código]. El único patrón de comprobación en ejecución es `TimesFM3Adapter._load_model` con `importlib.metadata.version("timesfm")` (aprox. líneas 1656-1696 en d9b8f4f), que se ejecuta después del import y solo cuando falta `TimesFM3Forecaster` [verificado en código]. Los mensajes del `ImportError` de Chronos (aprox. línea 451), Moirai (2042), TabICL (2559), T0 (3787), TSICL (4300) y Nori (4890) no entrecomillan `backend_package`; con `>=` el shell redirigiría la salida a un fichero (`=2.0`) (evidence/moirai/verification/verification.md, MOI-P1) [verificado en código]. Entrecomillan ya `_import_timesfm`, TimesFM3, TabPFN y Timestral [verificado en código]. Chronos escribe además ">=2.0" fijo en el texto [verificado en código]. `packaging` ya se importa de forma incondicional en `skforecast/utils/utils.py` (línea 16 en d9b8f4f, `from packaging.requirements import Requirement`, usado por `_find_optional_dependency` y `check_optional_dependency`), y `skforecast/foundation/_utils.py` importa de `..utils`, así que cualquier import de `skforecast.foundation` ya carga `packaging`; igual en origin/0.27.x [verificado en código]. `pyproject.toml` no lo declara (llega vía optuna, `packaging>=20.0`) [probado, según evidence/cross_cutting.md, XC-2]. El compat solo prueba la última versión de cada backend: ningún mínimo declarado se prueba en CI (TAB-F19 según XC-2) [verificado en código].
- **X-5** [probado] `pd.Timestamp("2000-01-01") + pd.Timedelta(days=n)` emite un `DeprecationWarning` con numpy 2.5.4 y pandas 2.3.3 (el constructor de `Timedelta`, no la suma); `pd.date_range("2000-01-01", periods=n + steps, freq="D")[n:]` no avisa y da las mismas fechas. Mismo código en `TabICLAdapter._get_future_timestamps` y `TabPFNAdapter._get_future_timestamps` (aprox. líneas 2631 y 3305 en d9b8f4f). Evidencia: evidence/cross_cutting/probe_shared.log (sección 5), evidence/tabpfn/verification/rerun_probe_int_timestamps.log. **Se implementa en 07** (ver 3.7).
- **X-6** [probado] `_warn_if_non_commercial` usa `stacklevel=3` fijo (`_utils.py`, aprox. línea 439 en d9b8f4f) y se llama desde la carga perezosa: en Moirai el `LicenseWarning` apunta a `skforecast/foundation/_adapters.py:2069` (`self._load_module()` dentro de `MoiraiAdapter._ensure_forecast_obj`), no a la línea del usuario; también se ve en el log del CI (job 127875169007) (MOI-V1; evidence/moirai/verification/verification.md). El aviso de fallback MPS de Moirai (`stacklevel=7`, aprox. línea 2088) apunta a la línea del usuario por `ForecasterFoundation.predict` y un marco por encima por `FoundationModel.predict` (MOI-F13; evidence/moirai/misc_checks.log, sección 2). Hay más valores fijos en avisos de adaptadores: 3 en TabICL, TabPFN y Nori, 5 y 6 en Timestral [verificado en código].
- **X-7** [probado] Con `show_progress=False` (valor por defecto), `contextlib.redirect_stderr(io.StringIO())` en TabICL y TabPFN (aprox. líneas 2512 y 3176 en d9b8f4f) oculta la barra tqdm y también la visualización de los warnings del backend: el warning se emite y `pytest.warns` lo captura, pero no llega a stderr (TPF-F6, TAB-F15; evidence/tabpfn/verification/test_p1_pytest_warns.log, evidence/tabicl/stderr_redirect_check.log). **Se implementa en 07** (ver 3.7).
- **X-8** [probado] La garantía documentada de que la predicción de una serie no depende de las demás es falsa o tiene matices en cuatro backends:
  - TS-ICL: con longitudes distintas se pierde la exog futura y un NaN en la exog de una serie altera otra (TSI-F4, TSI-F5; evidence/tsicl/check_covariates_rollout.log).
  - T0: covariables heterogéneas convertidas en ceros con máscara MISSING y relleno leído como MISSING (T0-F4, T0-F5; evidence/t0/options_060.log, evidence/t0/compare_direct_060.log).
  - TimesFM 3.0: relleno de cada trozo de `per_core_batch_size` (TFM3-F5; evidence/timesfm3/batch_invariance.log).
  - Chronos-2 con `cross_learning=True`: depende de `batch_size` (CHR-F6).

  Textos afectados del código común: el docstring de `FoundationModel.predict` (Notes, aprox. línea 1114 en d9b8f4f: "so the prediction of a series never depends on the exog columns of the other series"), `tools/ai/llms-base.txt` línea 531 y la frase equivalente de `skills/foundation-forecasting/SKILL.md` línea 130 ("so the forecast of a series never depends on the exog of the others") [verificado en código]. La parte que `FoundationModel` sí garantiza es que una serie no recibe columnas de exog de otras series (agrupación por firma con `group_series_by_exog_signature` cuando `supports_heterogeneous_covariates` es `False`) [verificado en código].

### Hallazgos de otros planes que este plan resuelve o usa

- **CHR-F4** (propiedad de 05) [probado]: contexto todo NaN en Chronos-2; lo resuelve 3.1 (CHR-P1).
- **MOI-F12** (propiedad de 06) [probado]: contexto todo NaN en Moirai-2; lo resuelve 3.1 (MOI-P5).
- **TAB-F10** (propiedad de 07) [probado]: niveles `int` y `float32` rechazados por tabicl; lo resuelve 3.3 (TAB-P2).
- **NORI-F5** (09), **T0-F7** (01), **T25-F8** (04) [probado]: ids que el prefijo acepta y no cargan; los resuelve 3.2 (NORI-P1, T0-P5, T25-P3).
- **MOI-V1** y **MOI-F13** (06) [probado]: `stacklevel` fijo; 3.5 arregla `_warn_if_non_commercial` y aporta el helper, 06 lo aplica al aviso MPS (MOI-P6).

### Descartado

- "Convertir a `float` en `FoundationModel` arregla también los nombres de columna con `float32`" (recomendación de D-Q en el informe, marcada [inferido]): es falso. `f"q_{float(np.float32(0.1))}"` sigue dando `q_0.10000000149011612`, porque `float()` conserva el valor binario del `float32` [probado con numpy 2.5.4; coherente con evidence/tabicl/verification/verification.md].
- "TimesFM3Adapter declara `timesfm>=3.0` en `backend_package`" (versión inicial de T25-P1): corregido, declara `"timesfm[torch]"` sin cota y solo comprueba la versión en ejecución cuando falta `TimesFM3Forecaster` [verificado en código].
- Añadir a la tabla de ids no soportados `google/timesfm-3.0-flax`, `Salesforce/moirai-2.0-R-base` y `-large`: no existen en el Hub y el error del Hub ya es claro [probado]; no se incluyen.
- Añadir a la tabla los sufijos de TabICL, TabPFN-TS y TS-ICL: el id es una etiqueta y los pesos son fijos; basta documentarlo en 07, 08 y 03 (D21 y equivalentes) [verificado en código].

## 3. Cambios a implementar

### 3.1 CHR-P1 / MOI-P5 / TV-2 (X-1): rechazar en FoundationModel las ventanas de contexto sin valores observados

- **Qué.** En `FoundationModel.predict` (`skforecast/foundation/_foundation_model.py`), justo después del filtrado por `levels` (aprox. línea 1169 en d9b8f4f) y antes de la comprobación de `supports_nan_in_series` (aprox. línea 1229) y del relleno de `_left_pad_context`, comprobar que el contexto de cada serie que se va a predecir tenga al menos un valor observado. Se aplica en todos los caminos (contexto guardado en `fit`, `context` del usuario y `check_inputs=False`): es una operación barata [inferido] y en backtesting nunca salta, porque `backtesting_foundation` ya descarta esas series antes de llamar a `predict` [verificado en código]. Con la opción (a) de D11:

  ```python
  series_without_observations = [
      series_name
      for series_name in series_names_in
      if not context[series_name].notna().any()
  ]
  if series_without_observations:
      raise ValueError(
          f"The context window of series {series_without_observations} has no "
          f"observed values: their last `context_length` "
          f"({self.context_length}) observations are all NaN. Use a larger "
          f"`context_length`, impute the series or exclude them with `levels`."
      )
  ```

  Conviene extraerlo a un helper privado de `_utils.py` (por ejemplo `_check_context_has_observations(context, series_names_in, context_length)`) para testearlo aislado. `fit` no cambia: el usuario puede guardar un contexto así y predecir después con otro `context` [inferido]. Corregir el comentario de `skforecast/model_selection/_validation.py` (aprox. líneas 2874-2882): `FoundationModel.predict` rechaza esas series en todos los caminos; el backtesting las filtra antes para saltarlas en el fold en lugar de fallar.
- **Por qué.** X-1, TV-2, CHR-F4, MOI-F12. `.claude/rules/foundation.md` pide que el tratamiento genérico viva una vez en `FoundationModel`, no en cada adaptador.
- **Riesgo.** Bajo. Afecta a todos los adaptadores, pero solo a entradas que hoy dan valores sin sentido o un error críptico [probado]. Con varias series, la opción (a) hace fallar toda la llamada: quien hoy recibe predicciones válidas para las demás series junto con basura para una tendrá que usar `levels` o corregir los datos [inferido]. El orden importa en un caso: una serie toda NaN en la ventana con un adaptador `supports_nan_in_series=False` recibirá el error nuevo en lugar del de NaN; el test existente `test_predict_ValueError_when_context_has_nan_and_adapter_does_not_support_it` no se ve afectado (su serie `s2` tiene valores observados) [verificado en código].
- **Rompe compatibilidad.** Sí, solo para esas entradas (pasan de valores absurdos a un `ValueError`). Hay que anotarlo en la release y en "Before upgrading".
- **Decisión.** D11. Con (b) el cambio es otro: excluir esas series de las llamadas al adaptador, rellenar sus filas con NaN en `pred_matrix` y emitir un `MissingValuesWarning` que las nombre. Con (c) solo se corrigen el comentario y los textos.

### 3.2 NORI-P1 / T0-P5 / T25-P3 (X-3): tabla central de ids no soportados, consultada antes de los prefijos

- **Qué.** En `skforecast/foundation/_adapters.py`, junto a `_ADAPTER_REGISTRY` (aprox. línea 5895 en d9b8f4f), una tabla nueva `_UNSUPPORTED_MODEL_IDS` (patrón de expresión regular y motivo; requiere `import re`, que el módulo no importa hoy) que `_resolve_adapter` consulta antes de recorrer los prefijos:

  ```python
  _UNSUPPORTED_MODEL_IDS: dict[str, str] = {
      r"Synthefy/Nori-Causal": (
          "it is a causal discovery model, not a forecaster"
      ),
      r"theforecastingcompany/t0-[^/]*-onnx": (
          "ONNX exports are run with onnxruntime, not with tfc-t0; use the "
          "PyTorch checkpoint, e.g. 'theforecastingcompany/t0-beta'"
      ),
      r"google/timesfm-2\.5-200m-(flax|transformers)": (
          "only the PyTorch checkpoint 'google/timesfm-2.5-200m-pytorch' can "
          "be loaded by the timesfm package"
      ),
  }
  ```

  `_resolve_adapter` hace `re.match(pattern, model_id)` para cada entrada y lanza `ValueError(f"'{model_id}' is not supported: {reason}.")`. Como `FoundationModel.__init__` (aprox. línea 301), `FoundationModel.set_params(model_id=...)` (aprox. línea 1378) y `get_model_info` (`skforecast/foundation/_model_info.py`, aprox. línea 218) ya llaman a `_resolve_adapter`, los tres fallan al construir, sin descargas [verificado en código]. `list_adapters` no cambia (recorre `_ADAPTER_REGISTRY`) [verificado en código]. Actualizar el docstring de `_resolve_adapter`. No se toca `_MODEL_LICENSES` ni el prefijo de ningún adaptador. Coordinación con 01: si 01 ya añadió un rechazo de `-onnx` dentro de `T0Adapter._load_model` (T0-P5 en su forma por adaptador), este PR lo retira en favor de la entrada de la tabla; si no, 01 no necesita hacerlo. 09 no necesita tocar `NoriAdapter` para NORI-P1.
- **Por qué.** X-3, NORI-F5, T0-F7, T25-F8.
- **Riesgo.** Bajo: solo afecta a ids que hoy fallan más tarde [probado]. Instanciar directamente `TimesFM25Adapter(model_id="google/timesfm-2.5-200m-flax")` (clase privada) seguiría pasando `_validate_model_id_prefix` [verificado en código]; no es un camino documentado [inferido].
- **Rompe compatibilidad.** No: esos ids ya fallaban.
- **Decisión.** D-REG. Si para TimesFM 2.5 se elige el prefijo exacto `google/timesfm-2.5-200m-pytorch` (D9, opción A, propiedad de 04), la entrada de TimesFM sale de esta tabla y el cambio de prefijo se hace en 04 (afecta a la clave de `_ADAPTER_REGISTRY`, a `TimesFM25Adapter._MODEL_ID_PREFIX` y a `test_ADAPTER_REGISTRY_contains_all_expected_entries`).

### 3.3 TAB-P2 (X-2): niveles de cuantil a `float` antes de llamar al adaptador

- **Qué.** En `FoundationModel.predict`, tras la validación de rango (aprox. líneas 1135-1145 en d9b8f4f), construir `quantiles_adapter = [float(q) for q in quantiles]` (o `None`) y pasarlo en la llamada `self.adapter.predict(..., quantiles=quantiles_adapter)` (aprox. línea 1286). Los nombres de columna se siguen construyendo con los niveles que pidió el usuario (`col_names = [f"q_{q}" for q in quantiles]`, aprox. línea 1310), para no cambiar `q_0` por `q_0.0` con niveles enteros [probado: `f"q_{0}"` da `q_0` y `f"q_{float(0)}"` da `q_0.0`]. El orden se conserva, y `FoundationModel` asigna la salida del adaptador por posición, así que no hay que tocar ningún adaptador [verificado en código: `pred_matrix[:, i, :] = raw_predictions[series_name]`].
- **Por qué.** X-2, TAB-F10.
- **Riesgo.** Mínimo: predicciones idénticas a pedir niveles `float` [probado en evidence/tabicl/proto_fixes.log]. Para TS-ICL, 03 traduce después cada nivel a un `Decimal` exacto de la rejilla en el adaptador (`Decimal(round(q * 100)) / 100`, TSI-P3, D31), y no a `Decimal(str(q))`, porque `str(float(np.float32(0.1)))` es `"0.10000000149011612"` y no coincide con la rejilla [probado con numpy 2.5.4]; conviene que su test cubra un nivel que llega desde `np.float32`.
- **Rompe compatibilidad.** No.
- **Decisión.** D-Q. Con (b) el cambio se hace solo en `TabICLAdapter` y pasa a 07.

### 3.4 X-4: helper común y opcional de versión mínima del backend y del comando de instalación

- **Qué.** En `skforecast/foundation/_utils.py`, helpers privados que reutilizan `packaging` (ya importado en ejecución por `skforecast/utils/utils.py`) e `importlib.metadata`:

  ```python
  def _check_backend_min_version(adapter_cls: type) -> str | None:
      # req = Requirement(adapter_cls.backend_package)
      # installed = importlib.metadata.version(req.name); returns None if the
      # distribution metadata is not found (the import step of the adapter
      # reports "not installed"); raises ImportError if
      # not req.specifier.contains(installed, prereleases=True).

  def _pip_install_command(backend_package: str, upgrade: bool = False) -> str:
      # Always quoted: '`pip install -U "uni2ts>=2.0"`'
  ```

  El mínimo se declara una sola vez en el código, en el atributo de clase `backend_package`, que ya alimenta el mensaje del `ImportError` y `get_model_info`; `tools/foundation_compat/adapters.toml` repite el valor y `test_adapters_toml_entries_match_the_adapter_classes` exige que coincidan, mientras `test_adapters_toml_backend_package_is_a_PEP_508_requirement` lo valida como requisito PEP 508 (los dos en `test_backends_compat_config.py`) [verificado en código]. Mensaje de "versión antigua": `f"{adapter_cls.__name__} requires {req.name}{req.specifier}, but {installed} is installed. Upgrade with {_pip_install_command(adapter_cls.backend_package, upgrade=True)}."`. Cada adaptador llamará a `_check_backend_min_version(type(self))` al principio de su método de carga (`_load_model`, `_load_module` o equivalente), antes del import del backend, para que un backend antiguo cuyo import falla también reciba el mensaje de versión [inferido], y siempre antes de `_warn_if_non_commercial` (como `TimesFM3Adapter`, que hoy comprueba la versión después del import y antes del aviso de licencia). Distinguir "falta el runtime o el extra" (torch en T0 y TimesFM, `uni2ts.model.moirai2` en Moirai) sigue siendo una sonda propia de cada adaptador, que usará `_pip_install_command` para su mensaje.

  **En este PR no se declara ningún mínimo ni se modifica ningún adaptador.** Cada plan de adaptador declara el suyo y adopta el helper:

  | Backend | Plan | Decisión | Recomendación del informe |
  |:--|:--|:--|:--|
  | tfc-t0 | 01 | D1 | `tfc-t0[torch]>=0.6` (el tope `<0.7` lo decide quien lleva #1385) |
  | tsicl | 03 | D32 | sin mínimo por ahora (guard de horizonte); subir el mínimo cuando se publique el arreglo del rollout; entrecomillar el mensaje |
  | timesfm (TimesFM 2.5; también simplifica la sonda de TimesFM 3) | 04 | D8 | `>=3.0` con `ImportError` |
  | chronos-forecasting | 05 | D12 | `>=2.2`; `>=2.3.2` si se quiere garantizar transformers 5 (cross_cutting.md recomienda `>=2.3`) |
  | uni2ts | 06 | D15 | `>=2.0` más documentar Python <= 3.12 y el conflicto con pip |
  | tabicl | 07 | D20 | `>=2.2.0` |
  | tabpfn-time-series | 08 | D19 | `>=1.4` |
  | synthefy-nori | 09 | D24 | `>=0.22` |

  01 es urgente e independiente: si se fusiona antes que este plan, hace su propia comprobación (patrón de `TimesFM3Adapter`) y un PR posterior la pasa al helper; si este plan va antes, 01 usa el helper directamente.
- **Por qué.** X-4. Fusiona la parte común de T25-P1, T25-P5, TV-1, TFM3-P1, CHR-P2, MOI-P1, TAB-P3, NORI-P3, NORI-V1, TPF-P8 y la parte (b) de TSI-P4; las partes propias de cada adaptador siguen en su plan.
- **Riesgo.** Bajo en este PR: código nuevo sin llamadas desde los adaptadores. `importlib.metadata` debería encontrar la distribución con su nombre normalizado (`tfc-t0` frente a `tfc_t0`) [inferido; cubrir con test]. `specifier.contains(installed, prereleases=True)` sigue PEP 440: `2.0.0rc1` no cumple `>=2.0` y se informa como versión antigua; los topes (`<0.7`) y las versiones locales (`2.15.0+cpu`) se evalúan correctamente [probado con packaging 26.3].
- **Rompe compatibilidad.** No en este PR. Sí en cada PR de adaptador para quien tenga un backend por debajo del mínimo, que lo anotará en su release note.
- **Decisión.** D-VER. Con (a) este apartado se reduce a `_pip_install_command` (comillas) y los mínimos solo protegen instalaciones nuevas; con (c) desaparece.

### 3.5 MOI-V1 (X-6; base de MOI-P6): `stacklevel` dinámico para los avisos emitidos desde el código común

- **Qué.** Reutilizar el helper que ya existe en `skforecast/model_selection/_utils.py` en vez de escribir otro:
  1. **Trasladar el helper.** Mover `_first_frame_outside_skforecast` y `_stacklevel_user_code` (aprox. líneas 3406-3457 en d9b8f4f) a `skforecast/utils/utils.py`.
     - Añadir `import os` a `skforecast/utils/utils.py` (hoy no lo importa; `sys` sí) y los dos nombres a la lista explícita de privados de `skforecast/utils/__init__.py`, porque `from .utils import *` no exporta nombres con guion bajo [verificado en código].
     - `model_selection/_utils.py` importa los dos nombres desde `..utils` (ya importa de allí): `_first_frame_outside_skforecast` también se llama directamente en `_SearchWarnings` para registrar la posición del usuario. Sus usos actuales en `model_selection/_utils.py` no cambian (5 en d9b8f4f, contando esa llamada directa; `origin/0.27.x` añade otros con el commit `522d11d` y los avisos nuevos de la búsqueda, 9 en total, todos en el mismo módulo, y desplaza el helper hacia la línea 4035) [verificado en código].
     - `_stacklevel_user_code` ya calcula el `stacklevel` que apunta al primer marco fuera del paquete, y los tests cuentan como código de fuera [verificado en código]. La ruta del paquete se calcula con `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`, que da `skforecast/` desde las dos ubicaciones [verificado en código].
     - No se puede importar desde `model_selection` en `foundation/_utils.py`: `skforecast/model_selection/_validation.py` importa `..foundation._utils` y habría un import circular; `foundation/_utils.py` ya importa de `..utils` [verificado en código].
  2. **`_warn_if_non_commercial`** (`skforecast/foundation/_utils.py`, aprox. línea 439): sustituir `stacklevel=3` por `stacklevel=_stacklevel_user_code()`, calculado dentro de la función. Así el aviso apunta a la línea del usuario por cualquier vía (`FoundationModel.predict`, `ForecasterFoundation.predict`, backtesting), sin depender de la profundidad de la carga perezosa. Es común a todos los adaptadores, así que cambia aquí y no en un PR de adaptador.
  3. **`_resolve_torch_device`** (`skforecast/foundation/_adapters.py`, aprox. líneas 34-77): cambiar `stacklevel: int = 2` por `stacklevel: int | None = None`. Con `None`, `warnings.warn(mps_fallback_message, stacklevel=_stacklevel_user_code())`. Actualizar su docstring. Los valores explícitos actuales (Moirai `stacklevel=7`, aprox. línea 2088; Timestral `stacklevel=6`, aprox. línea 5890) siguen funcionando igual hasta que su PR los quite, y `test_MoiraiAdapter_ensure_forecast_obj_calls_to_with_resolved_device` (que comprueba `kwargs["stacklevel"] == 7`) sigue pasando hasta 06 [verificado en código].
  - `skip_file_prefixes` de `warnings.warn` no sirve: exige Python 3.12 y skforecast admite `>=3.10` (`pyproject.toml`) [verificado en código].
- **Por qué.** X-6, MOI-V1, MOI-F13.
- **Riesgo.** Nulo para las predicciones; solo cambia el fichero y la línea que muestra el aviso. Hay que ejecutar los tests de `model_selection` que comprueban `w.filename` (sección 4). Los filtros de warnings que el usuario defina por módulo pueden dejar de coincidir si filtraban por `skforecast.foundation._adapters` [inferido].
- **Rompe compatibilidad.** No.
- **Decisión.** Ninguna. Alternativa, si se prefiere: sacar 3.5 a un PR pequeño propio (como el PR-13 `chore/foundation-warning-stacklevel` de `evidence/cross_cutting.md`). Quién lo adopta después: 06 quita `stacklevel=7` del fallback MPS de Moirai (MOI-P6); 03 no debe pasar un `stacklevel` fijo si añade `mps_fallback_message` a TS-ICL (TSI-P5); los `stacklevel=3` de los avisos de `predict` de TabICL, TabPFN-TS y Nori quedan para un seguimiento posterior a 02 con el helper (07, 08 y 09 no los tocan); Timestral queda fuera de esta revisión.

### 3.6 X-8: separar en los textos comunes "no recibe la exog de otras series" de "la salida es idéntica a predecirla sola"

- **Qué.** Solo texto, en los sitios comunes a todos los adaptadores:
  - Docstring de `FoundationModel.predict`, Notes (aprox. líneas 1107-1119 en d9b8f4f): sustituir "so the prediction of a series never depends on the exog columns of the other series" por una frase como "so a series never receives the exog columns of other series. Whether its forecast is identical to forecasting it alone depends on the backend (see the Notes of each adapter)".
  - `tools/ai/llms-base.txt` línea 531 y `skills/foundation-forecasting/SKILL.md` línea 130: el mismo matiz en lugar de "so the forecast of a series never depends on the exog of the others".
- **Por qué.** X-8.
- **Riesgo.** Nulo.
- **Rompe compatibilidad.** No.
- **Decisión.** Ninguna. Las correcciones propias de cada backend van en su plan: T0 en 01 (comentario de `_adapters.py` aprox. línea 3718, T0-F5; o en el PR de invariancia de lote de T0 si 01 lo separa), TS-ICL en 03 (celdas 2 y 21 de la guía de series heterogéneas, TSI-F17 y TSI-P7; el atributo nuevo de agrupación de TSI-P1, D29), TimesFM 3.0 en 04 (`adapter-parameters.md` línea 82, TFM3-P3) y Chronos-2 en 05 (CHR-P5). 01 y 03 tocan las mismas líneas (línea 130 de `SKILL.md` y línea 531 de `llms-base.txt`, donde la lista de adaptadores agrupados y la frase de X-8 están en la misma línea): el que se fusione en segundo lugar rebasa y conserva ambos cambios.

### 3.7 Reparto del resto de hallazgos transversales

| Id (cross_cutting.md) | Dónde se implementa | Motivo |
|:--|:--|:--|
| X-1 (XC-1), X-2 (XC-5, tipos), X-3 (XC-3), X-4 (XC-2, parte común), X-6 (XC-8, parte común), X-8 (XC-6, textos comunes) | **Este plan** (3.1 a 3.6) | Código o texto que vive una sola vez |
| X-5 (XC-9) | **07**: helper común de timestamps sintéticos en `_utils.py` (TPF-P5, TPF-V4), adoptado por 08 | Solo lo usan TabICL y TabPFN; 07 introduce los helpers compartidos con 08 |
| X-7 (XC-4) | **07**: context manager común en `_utils.py` (por ejemplo `_suppress_backend_progress()`, que reemite con `warnings.warn_explicit`), adoptado por 08 (TAB-P1, TPF-P1) | Idem |
| XC-5, validación central contra `SUPPORTED_QUANTILES` | Pospuesto (sección 8) | No hay defecto que corregir: los adaptadores que validan lo hacen bien; los 8 niveles fallidos de TS-ICL los resuelve 03 (D31) |
| XC-7 (tests y fakes) | Fakes en cada plan de adaptador; el test genérico de orden de cuantiles en **10** (sección 3.6). Las demás invariantes genéricas del smoke (XC-7 de `evidence/cross_cutting.md`) quedan fuera de esta serie de PR (`10_docs_ci.md`, sección 8); si se quieren, van en un PR aparte `chore/foundation-compat-checks` (PR-12 de `evidence/cross_cutting.md`), después de 01 y 03 | Las fakes son propias de cada backend |
| XC-10 (dispositivo) | **03** (TSI-P5) y **05** (CHR-P6) | No hace falta cambiar `_resolve_torch_device` salvo el `stacklevel` de 3.5 |
| XC-11 (licencias) | Textos en **04** (TFM3-P5), **06** (MOI-P4) y **08** (TPF-P4) | `_MODEL_LICENSES` es correcta |
| XC-12 (infraestructura de compat) | **10** | Un carril del CI de compatibilidad con la versión mínima de cada backend es una subdecisión de D-VER que ningún plan de esta serie implementa (`10_docs_ci.md`, sección 8) |

### División opcional en dos PR

1. `fix/foundation-common-guards`: 3.1, 3.2, 3.3 y 3.6. Cambia el comportamiento visible y lleva las release notes de **Fixed**.
2. `chore/foundation-backend-version-helper`: 3.4 y 3.5. Helpers internos; solo 3.5 cambia algo visible (la línea que muestra el `LicenseWarning`).

Dependencias si se divide: 01, 04 y 09 se apoyan en el primero para sus ids no soportados (3.2), 03 y 07 en la conversión a `float` (3.3); 04 a 09 se apoyan en el segundo para el helper de versión y 06 para el de `stacklevel`. Ambos salen de `origin/0.27.x` y no se pisan: tocan funciones distintas [inferido].

## 4. Tests

Ficheros bajo `skforecast/foundation/tests/tests_foundation_models/` salvo que se indique otra ruta.

- **`test_predict.py`** (3.1 y 3.3), con `FakePipeline` de `fixtures_adapters.py`:
  - `test_predict_ValueError_when_context_window_has_no_observed_values`: contexto guardado en `fit` y `context` pasado en `predict`; serie única y multiserie (el mensaje nombra solo la serie afectada).
  - `test_predict_no_error_when_series_without_observations_is_excluded_by_levels`.
  - `test_predict_ValueError_when_context_window_has_no_observed_values_and_check_inputs_False`.
  - `test_predict_forwards_quantiles_as_float_to_adapter`: `[0, 0.5, 1]` y `[np.float32(0.1), np.float32(0.9)]`; los elementos de `FakePipeline.last_quantile_levels` son todos `float` y las columnas no cambian (`q_0`, `q_0.5`, `q_1`; `q_0.10000000149011612`, `q_0.8999999761581421`).
- **`skforecast/foundation/tests/tests_forecaster_foundation/test_predict.py`**: un test de que `ForecasterFoundation.predict` propaga el `ValueError` de 3.1.
- **`skforecast/model_selection/tests/tests_validation/test_backtesting_foundation.py`**: no se modifica; `test_output_backtesting_foundation_series_skipped_when_context_window_all_nan` debe seguir pasando.
- **`test_resolve_adapter.py`** (3.2):
  - Quitar `("google/timesfm-2.5-200m-flax", TimesFM25Adapter)` de `test_resolve_adapter_returns_correct_class` (es el único sitio del repositorio que usa uno de los ids de la tabla) [verificado en código].
  - `test_resolve_adapter_ValueError_when_model_id_is_unsupported`, parametrizado con los 7 ids de la sección 1; comprueba el motivo y, además, que cada id empieza por un prefijo de `_ADAPTER_REGISTRY` (si no, la entrada sería código muerto).
  - `test_resolve_adapter_resolves_supported_ids_sharing_an_unsupported_prefix`: `Synthefy/Nori`, `Synthefy/Nori-1-5`, `Synthefy/Nori-30M`, `Synthefy/Nori-100M`, `theforecastingcompany/t0-alpha`, `theforecastingcompany/t0-beta`, `theforecastingcompany/t0-gamma`, `google/timesfm-2.5-200m-pytorch`.
- **`test_init.py`**, **`test_set_params.py`** y **`test_get_model_info.py`**: `FoundationModel("Synthefy/Nori-Causal")`, `set_params(model_id="google/timesfm-2.5-200m-flax")` y `get_model_info("theforecastingcompany/t0-beta-onnx-int8")` lanzan `ValueError` con el motivo.
- **`test_utils.py`** (3.1, 3.4 y 3.5):
  - `_check_context_has_observations` (si se extrae el helper): sin series afectadas no lanza nada; con una o varias, `ValueError` que las nombra.
  - `_check_backend_min_version` con `monkeypatch` de `importlib.metadata.version` y una clase de prueba con `backend_package = "fakepkg[extra]>=2.0"`: versión menor da `ImportError` con el comando entrecomillado; igual o mayor devuelve la versión; `"2.0.0rc1"` da `ImportError`; con `backend_package = "fakepkg>=0.6,<0.7"` y `"0.7.1"` instalada, `ImportError`; `PackageNotFoundError` devuelve `None`.
  - `_pip_install_command`: siempre con comillas dobles, con y sin `-U`.
  - `test_warn_if_non_commercial_points_to_caller`: `record[0].filename == __file__`, mismo patrón que `test_ForecasterFoundation_Timestral_FeatureOutOfRangeWarning_points_to_user_call` en `test_TimestralAdapter.py`. `test_warn_if_non_commercial_warns_for_registered_prefixes` sigue igual.
- **`test_MoiraiAdapter.py`**: ampliar `test_MoiraiAdapter_load_module_LicenseWarning_on_successful_load` con `record[0].filename == __file__`.
- **`test_device_handling.py`**: `_resolve_torch_device(..., stacklevel=None)` con MPS simulado apunta al llamador; los tests `test_resolve_torch_device_*` actuales y los que pasan `stacklevel` explícito no cambian.
- **Helper trasladado (3.5)**: test nuevo en `skforecast/utils/tests/tests_utils/` (por ejemplo `test_stacklevel_user_code.py`): llamado directamente desde el test da 1 más los marcos de skforecast saltados, y un aviso emitido desde una función de skforecast con ese `stacklevel` apunta al test; los ficheros de `tests` cuentan como fuera del paquete. Ejecutar además los tests de `skforecast/model_selection/tests/` que comprueban `filename == __file__` en `origin/0.27.x` (la lista cambió tras `522d11d`, que añadió tres llamadores de `_stacklevel_user_code`, y `test_warn_n_evaluated_search_multiseries.py` pasó a llamarse `test_warn_n_evaluated_search.py`): `tests_utils/test_search_warnings.py`, `tests_utils/test_warn_n_evaluated_search.py`, `tests_utils/test_sample_param_grid.py`, `tests_utils/test_check_one_step_ahead_input.py`, `tests_utils/test_initialize_levels_model_selection_multiseries.py` y `tests_search/test_bayesian_search_forecaster.py` [verificado en código]. Comprobar la lista con `git grep -l __file__ origin/0.27.x -- skforecast/model_selection/tests` antes de empezar (`test_optimize_study.py` usa `__file__` como pathname de un `LogRecord`, no es una comprobación de filename de un aviso).
- **Fakes**: no hace falta cambiar ninguna en este plan. 07 hará que `FakeTabICLForecaster` valide `isinstance(q, float)` (TAB-P6).
- **`tools/foundation_compat/`**: sin cambios en `tests/contracts.py` ni en `adapters.toml` (ningún backend cambia de firma ni de mínimo en este plan).
- **Smoke de sanidad** (no debe cambiar): `python tools/foundation_compat/run.py --adapter chronos` y `~/.venvs/fm-review-tabicl/bin/python -m pytest tools/foundation_compat/tests/test_smoke.py --backend tabicl -q -p no:cacheprovider`.
- **Scripts de evidencia, antes y después** (desde la raíz del repositorio):

  ```bash
  ~/.venvs/fm-review-chronos/bin/python dev/foundation_adapters_review/evidence/cross_cutting/probe_shared.py
  ~/.venvs/fm-review-chronos/bin/python dev/foundation_adapters_review/evidence/chronos/all_nan_context.py
  ~/.venvs/fm-review-moirai/bin/python dev/foundation_adapters_review/evidence/moirai/all_nan_context.py
  ~/.venvs/fm-review-timesfm25/bin/python dev/foundation_adapters_review/evidence/timesfm25/verification/verify_all_nan_context.py
  ~/.venvs/fm-review-tabicl/bin/python dev/foundation_adapters_review/evidence/tabicl/capabilities_check.py
  ~/.venvs/fm-review-timesfm25/bin/python dev/foundation_adapters_review/evidence/timesfm25/registry_and_loading.py
  ~/.venvs/fm-review-t0/bin/python dev/foundation_adapters_review/evidence/t0/registry_and_loading.py
  ```

  (`t0/check_load_errors.py` llama al backend directamente, sin skforecast, así que no sirve para comprobar este cambio.) Los resultados esperados están en la sección 7.
- **Backtest de la guía**: no hace falta. 3.1 solo afecta a ventanas sin ningún valor observado y 3.3 no cambia el valor de los niveles `float` [inferido]. Si se quiere una comprobación de corrección (no un benchmark): `~/.venvs/fm-review-chronos/bin/python dev/foundation_adapters_review/evidence/common/backtest_guide.py --model-id autogluon/chronos-2-small --exog --out dev/foundation_adapters_review/evidence/chronos/backtest_after_02.csv` debe dar MAE 171.267, igual que evidence/chronos/backtest_before.log (chronos-forecasting 2.3.2, torch 2.15.0+cpu, Python 3.12.3) [probado antes del cambio].

## 5. Documentación y contexto de IA

- **Docstrings.**
  - `FoundationModel.predict`: en `quantiles`, añadir que los niveles se pasan al backend como `float` de Python y que los nombres de columna conservan el valor pedido; en Notes, añadir que una ventana de contexto (las últimas `context_length` observaciones) sin ningún valor observado lanza `ValueError` (3.1), y el matiz de X-8 (3.6).
  - `_resolve_adapter`: tabla de ids no soportados.
  - `_warn_if_non_commercial` y `_resolve_torch_device`: nuevo comportamiento de `stacklevel`.
  - Helpers nuevos de `_utils.py`: docstring NumPy completo.
- **`docs/user_guides/foundation-forecasting-with-heterogeneous-series.ipynb`** (solo markdown, no hace falta re-ejecutar):
  - Tabla "In short" (celda 2 en la revisión), fila "Can the target contain NaN?": con D11 (a) la frase "A context window that is entirely NaN is rejected" pasa a ser cierta; añadir "with a `ValueError` naming the series". Con (b), reescribirla. La fila "Does batching series together change their forecasts?" es de 03 y 05.
  - Sección "How it works" (celda 18 en la revisión), viñeta "NaN in the data": tras "No backend can use a context window that is entirely NaN", añadir que `FoundationModel.predict` lo rechaza antes de llamar al backend.
  - "Alignment and grouping before calling the backend" (celda 20 en la revisión), paso 5 "NaN in the context is checked": añadir que, para cualquier backend, una ventana sin valores observados lanza un `ValueError` que nombra la serie.
  - "Requirements common to every backend" (celda 21 en la revisión), viñeta "A series cannot be **entirely NaN**": ampliarla a "nor can its last `context_length` observations".
- **`docs/user_guides/foundation-forecasting-models.ipynb`**: sin cambios en este plan; 10 la re-ejecuta.
- **`skills/foundation-forecasting/SKILL.md`**:
  - Sección "Multi-Series (Global Zero-Shot Model)" (aprox. líneas 99-105): una frase sobre el contexto sin valores observados.
  - Línea 130 (sección "With Exogenous Variables"): matiz de X-8.
  - Tras "The adapter is resolved automatically from the `model_id` prefix" (aprox. línea 176): "A few ids that match a prefix are rejected because they are not loadable forecasters: `Synthefy/Nori-Causal`, the ONNX exports of TFC-T0 and `google/timesfm-2.5-200m-flax` / `-transformers`".
  - "Common Mistakes": un punto nuevo sobre el `ValueError` cuando las últimas `context_length` observaciones de una serie son todas NaN.
  - `references/adapter-parameters.md`: sin cambios en este plan.
- **`tools/ai/llms-base.txt`**: línea 531 (X-8 más la frase del contexto sin valores observados) y, tras la tabla de adaptadores (aprox. líneas 485-496), la frase de ids rechazados.
- **Regeneración**: skill `ai-context-sync` (`python tools/ai/generate_ai_context_files.py` y después `--check`). No editar a mano `AGENTS.md`, `llms-full.txt` ni `docs/llms*.txt`.
- **Release note** (`docs/releases/releases.md`, sección 0.27.0; usar la skill `release-note`). En **Fixed**:

  ```markdown
  + <code>[FoundationModel]</code> and <code>[ForecasterFoundation]</code> passed to the backend, without any warning, a series whose last `context_length` observations were all NaN: Chronos-2 returned quantiles around ±1e5, Moirai-2 values close to 0, and TimesFM 2.5 failed with a cryptic error (numpy >= 2.2) or predicted zeros (numpy < 2.2). They now raise a `ValueError` naming those series. <code>[backtesting_foundation]</code> already skipped them in each fold and is unchanged. ([#PR](https://github.com/skforecast/skforecast/pull/PR))

  + Quantile levels given as integers (`quantiles=[0, 0.5, 1]`) or numpy `float32` are now passed to the backend as Python floats. TabICL rejected them with `Quantiles must be a list of floats`. The names of the prediction columns are unchanged. ([#PR](https://github.com/skforecast/skforecast/pull/PR))

  + <code>[FoundationModel]</code> now raises a `ValueError` when it is created with an id that matches a supported prefix but is not a loadable forecaster: `Synthefy/Nori-Causal` (a causal discovery model), the ONNX exports of TFC-T0 (`theforecastingcompany/t0-*-onnx-*`) and `google/timesfm-2.5-200m-flax` / `-transformers`. They were accepted and failed later with a 404 or a misleading error, in some cases after downloading the weights. ([#PR](https://github.com/skforecast/skforecast/pull/PR))
  ```

  En "Before upgrading", bajo "And some code needs to be updated":

  ```markdown
  + <code>[FoundationModel]</code> and <code>[ForecasterFoundation]</code> raise a `ValueError` in `predict` when the last `context_length` observations of a series are all NaN, instead of returning meaningless forecasts: use a larger `context_length`, impute the series or exclude it with `levels`.
  ```

  Y, si entra 3.5, en **Changed**:

  ```markdown
  + The `LicenseWarning` of foundation models with non-commercial weights now points to the line of the user's code instead of an internal line of skforecast. ([#PR](https://github.com/skforecast/skforecast/pull/PR))
  ```

  3.4 no lleva release note propia: es interno hasta que cada PR de adaptador declare su mínimo y escriba su entrada.

## 6. Decisiones pendientes

| Id | Decisión | Opciones | Recomendación | Impacto en el plan |
|:--|:--|:--|:--|:--|
| D11 | Contexto sin valores observados (X-1: CHR-P1, MOI-P5, TV-2) | (a) `ValueError` que nombre las series; (b) aviso y predicción NaN para esas series, como hace el backtesting al saltarlas; (c) solo documentarlo | (a): coherente con el rechazo actual de series enteramente NaN y no cambia `backtesting_foundation`. La revisión de Moirai proponía un aviso | (a) es el código de 3.1; (b) cambia 3.1 (exclusión, relleno NaN y `MissingValuesWarning`), los tests, la release note y el texto de la celda 2; (c) deja 3.1 en el comentario y los textos. **pendiente** |
| D-REG | Ids aceptados por prefijo que no cargan (X-3: NORI-P1, T0-P5, T25-P3) | (a) tabla central de ids no soportados con motivo en `_resolve_adapter`, consultada antes de los prefijos; (b) comprobaciones en cada adaptador; (c) prefijos exactos; (d) documentar | (a), y (c) solo para TimesFM 2.5 si se prefiere (D9, de 04) | (a) es 3.2; con (c) para TimesFM la entrada sale de la tabla y el cambio de prefijo va a 04; (b) mueve el trabajo a 01, 04 y 09 y contradice `.claude/rules/foundation.md`. **pendiente** |
| D-Q | Tipo de los niveles de cuantil (X-2, TAB-P2) | (a) convertir a `float` en `FoundationModel`; (b) solo en `TabICLAdapter`. Subdecisión añadida por este plan: mantener los nombres de columna pedidos o normalizar los `float32` a su representación corta (`float(str(q))`, que da `q_0.1`) | (a), manteniendo los nombres pedidos. La afirmación del informe de que (a) arregla los nombres con `float32` está descartada (sección 2) | (a) es 3.3; (b) mueve el cambio a 07; normalizar los nombres cambiaría columnas visibles y necesitaría su propia release note. **pendiente** |
| D-VER | Política de versión mínima (X-4) | (a) mínimo solo en `backend_package` (cubre instalaciones nuevas y CI); (b) además un helper común de comprobación en ejecución (`importlib.metadata`, patrón de `TimesFM3Adapter`); (c) solo documentar. Subdecisión: mínimo derivado de `backend_package` (este plan) o argumentos explícitos (`_require_backend(adapter_cls, distribution, min_version, extra_hint)`, propuesta de XC-2) | (b) con un helper pequeño y el mínimo derivado de `backend_package`, para no repetir lógica de versiones en cada adaptador | (b) es 3.4; con (a) 3.4 se reduce a `_pip_install_command`; con (c) desaparece. Los mínimos concretos (D1, D8, D12, D15, D19, D20, D24, D32) se deciden en cada plan de adaptador. **pendiente** |

Relacionada, propiedad de otro plan: D9 (prefijo de TimesFM 2.5, plan 04) interactúa con D-REG como se indica arriba.

## 7. Criterios de aceptación

- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_predict.py -x -q` pasa, con los tests nuevos de 3.1 y 3.3.
- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_resolve_adapter.py skforecast/foundation/tests/tests_foundation_models/test_utils.py skforecast/foundation/tests/tests_foundation_models/test_init.py skforecast/foundation/tests/tests_foundation_models/test_set_params.py skforecast/foundation/tests/tests_foundation_models/test_get_model_info.py skforecast/foundation/tests/tests_foundation_models/test_device_handling.py skforecast/foundation/tests/tests_foundation_models/test_MoiraiAdapter.py skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py -q` pasa.
- [ ] `pytest skforecast/foundation/tests/tests_forecaster_foundation/ -q` pasa.
- [ ] Si entra 3.5: `pytest skforecast/utils/tests/tests_utils/test_stacklevel_user_code.py -q` (o el nombre elegido) y `pytest skforecast/model_selection/tests/tests_utils/test_search_warnings.py skforecast/model_selection/tests/tests_utils/test_warn_n_evaluated_search.py skforecast/model_selection/tests/tests_utils/test_sample_param_grid.py skforecast/model_selection/tests/tests_utils/test_check_one_step_ahead_input.py skforecast/model_selection/tests/tests_utils/test_initialize_levels_model_selection_multiseries.py skforecast/model_selection/tests/tests_search/test_bayesian_search_forecaster.py -q` pasan.
- [ ] `pytest skforecast/model_selection/tests/tests_validation/test_backtesting_foundation.py -q` pasa sin cambios en el fichero.
- [ ] Una vez al final: `pytest skforecast/foundation/tests/tests_foundation_models/ -q`.
- [ ] `probe_shared.py` (comando de la sección 4): la sección 1 muestra `raised: ValueError` con el nombre de la serie (`y`); la sección 2 muestra tipos `['float', 'float', 'float']` y `['float', 'float']`, con columnas `['level', 'q_0', 'q_0.5', 'q_1']` y `['level', 'q_0.10000000149011612', 'q_0.8999999761581421']` (sin cambios); la sección 3 da `ValueError` para `Synthefy/Nori-Causal`, `theforecastingcompany/t0-alpha-onnx-int8`, `google/timesfm-2.5-200m-flax` y `google/timesfm-2.5-200m-transformers`, y los otros cinco ids resuelven como antes.
- [ ] `chronos/all_nan_context.py`: los tres casos lanzan un `ValueError` que nombra `tail_nan`, en lugar de cuantiles de ±1e5.
- [ ] `moirai/all_nan_context.py`: `ValueError` que nombra `nan_tail`, en lugar de cuantiles en torno a 0.
- [ ] `timesfm25/verification/verify_all_nan_context.py`: la línea "context fully NaN (context_length=64)" muestra el `ValueError` de skforecast que nombra `gap` (ya no "The truth value of an empty array is ambiguous"); "context partly NaN (context_length=128)" sigue en `OK, NaN in output: False`. La línea "backend helpers" llama al backend directamente y no cambia.
- [ ] `tabicl/capabilities_check.py`: los probes "quantiles=[0, 0.5, 1] (int levels)" y "quantiles as numpy float32" pasan a `OK ['level', 'q_0', 'q_0.5', 'q_1']` y `OK ['level', 'q_0.10000000149011612', 'q_0.8999999761581421']` (antes `ERROR ValueError: Quantiles must be a list of floats`); el resto de probes no cambia.
- [ ] `timesfm25/registry_and_loading.py`: en la sección 1, `-flax` y `-transformers` dan `ValueError` con el motivo de la tabla y `-pytorch` sigue en `TimesFM25Adapter`; la sección 3 (instancia `TimesFM25Adapter` directamente, sin pasar por la tabla) sigue dando el 404.
- [ ] `t0/registry_and_loading.py`: `t0-alpha`, `t0-beta` y `t0-gamma` siguen en `T0Adapter` con licencia Apache-2.0; los 4 repos ONNX dan `ValueError` con el motivo de la tabla, y "Loading an ONNX repository through FoundationModel" muestra ese `ValueError` al construir, en lugar del `OSError`. La última sección (`torch_dtype`) no cambia: depende de 01.
- [ ] `python -c "from skforecast.foundation import FoundationModel; FoundationModel('Synthefy/Nori-Causal')"` lanza `ValueError: 'Synthefy/Nori-Causal' is not supported: it is a causal discovery model, not a forecaster.` sin descargar nada.
- [ ] Sanidad del compat: `python tools/foundation_compat/run.py --adapter chronos` en verde, y el smoke de tabicl en verde.
- [ ] `python tools/ai/generate_ai_context_files.py --check` sin diferencias tras regenerar.
- [ ] Skill `verify` en verde (`python .claude/skills/verify/affected.py` da los ficheros afectados; ruff sin errores en los ficheros tocados; comprobaciones de docs del notebook de series heterogéneas).
- [ ] Entradas en `docs/releases/releases.md`, sección 0.27.0 (Fixed y "Before upgrading"; Changed si entra 3.5).
- [ ] Sin cambios en ningún método de adaptador: `git diff origin/0.27.x -- skforecast/foundation/_adapters.py` solo muestra `import re`, la firma y el docstring de `_resolve_torch_device`, la tabla nueva y `_resolve_adapter`.

## 8. Fuera de alcance

- Mínimos concretos de cada backend y adopción del helper de versión en cada método de carga, incluido entrecomillar los `ImportError` de Chronos, Moirai, TabICL, T0, TSICL y Nori: planes 01 y 03 a 09 (D1, D8, D12, D15, D19, D20, D24, D32).
- Eliminación de `_TimesFMCompat` y simplificación de la sonda de import de `TimesFM3Adapter` y de `_import_timesfm` (T25-P1, T25-P5, TFM3-P1, TV-1): plan 04.
- `pyproject.toml` no declara `packaging` aunque `skforecast/utils/utils.py` lo importa de forma incondicional [verificado en código]; declararlo es un cambio aparte y trivial para el mantenedor.
- Helpers de timestamps sintéticos (X-5) y de reemisión de warnings ocultos por `redirect_stderr` (X-7): plan 07, adoptados por 08.
- Validación central de los niveles contra `SUPPORTED_QUANTILES` y retirada de `_validate_supported_quantiles` de los adaptadores que la llaman (TimesFM 2.5, TimesFM 3.0, Moirai y Timestral) (parte de XC-5): pospuesta a un PR posterior a 03, 04 y 06, que tocan esas mismas zonas. Es la opción centralizada que recomienda `.claude/rules/foundation.md` y el diseño final recomendado, pero hoy no corrige ningún defecto [verificado en código].
- Traducción a `Decimal` de los niveles de TS-ICL (TSI-P3, D31) y el atributo de agrupación de TS-ICL (TSI-P1, D29): plan 03.
- Textos de independencia propios de cada backend (T0, TS-ICL, TimesFM 3.0, Chronos-2): planes 01, 03, 04 y 05.
- Rechazar o documentar los sufijos arbitrarios de TabICL, TabPFN-TS y TS-ICL (D21 y equivalentes): planes 07, 08 y 03.
- El relleno con 0 de TabICL y TabPFN-TS cuando la ventana tiene un único valor observado (comportamiento del backend, [verificado en código]): este plan solo rechaza las ventanas sin ninguno; documentarlo, si se quiere, corresponde a 07 y 08 [inferido].
- Que `context_length` cuente filas con NaN incluidos (TAB-V1, decisión común): no se cambia aquí; su documentación va en 07.
- Demás cambios de `tools/foundation_compat` (XC-12) y el test genérico de orden de cuantiles del smoke (`10_docs_ci.md`, sección 3.6): plan 10. Las demás invariantes genéricas del smoke (XC-7 de `evidence/cross_cutting.md`) quedan fuera de esta serie de PR (`10_docs_ci.md`, sección 8); si se quieren, van en un PR aparte `chore/foundation-compat-checks` (PR-12 de `evidence/cross_cutting.md`), después de 01 y 03. Un carril del CI de compatibilidad con la versión mínima de cada backend es una subdecisión de D-VER que ningún plan de esta serie implementa (`10_docs_ci.md`, sección 8).
- Quitar el `stacklevel` fijo de los avisos propios de Moirai (MOI-P6, plan 06), de TabICL, TabPFN-TS y Nori (fuera de esta serie de PR: seguimiento posterior a 02 que reutilice el helper de 3.5; 07, 08 y 09 no lo tocan) y de Timestral (fuera de esta revisión). Tampoco se revisan aquí otros `stacklevel` fijos del código común que la revisión no midió: los de `FoundationModel` en `_foundation_model.py` (`fit`, `_exog_to_dict` para `exog` en formato largo, `_prepare_future_exog`, `_check_exog_columns` y los dos `IgnoredArgumentWarning` de `predict`), el de `ForecasterFoundation.fit` y el `MissingValuesWarning` de `align_context_exog` en `_utils.py` [verificado en código]. Un PR posterior puede adoptar el helper de 3.5 en todos ellos de una vez.
- Una política común de `revision` y `model_kwargs` (D13 de cross_cutting.md; TFM3-P4, T0-F12, NORI-F7): pospuesta.
- Cualquier medida de velocidad o de precisión entre versiones, variantes o checkpoints: fuera del alcance de skforecast. El único backtest que se menciona es una comprobación de corrección antes y después.
