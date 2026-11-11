# Verificación adversarial: TabPFNAdapter (tabpfn-time-series 1.4.0 / tabpfn 9.2.0)

Fecha: 2026-11-06. Entorno: ~/.venvs/fm-review-tabpfn (Python 3.12.3, tabpfn-time-series 1.4.0,
tabpfn 9.2.0, tabpfn-client 0.7.1, torch 2.15.0+cpu, numpy 2.5.4, pandas 2.3.3, skforecast 0.27.0
editable). Entorno temporal ~/.venvs/fm-review-tabpfn-130v (tabpfn-time-series 1.3.0 + tabpfn 9.0.0),
borrado al terminar. Máquina: 4 CPU, sin GPU. Sin TABPFN_TOKEN: los pesos TabPFN-3.5 no se descargan.
No se han ejecutado medidas de velocidad ni comparaciones de precisión (fuera de alcance).

## Qué se ha re-ejecutado

| Script / log | Qué comprueba |
|---|---|
| `rerun_compare_direct.log` | compare_direct.py del investigador: diferencias 0.0 en los 6 casos. |
| `nontrivial_check.py/.log` | El TabPFN-3.5 diminuto aleatorio da una predicción constante en el horizonte (std 0) e insensible a la exog (exog+3 y sin exog: diferencia 0.0). Sus igualdades no prueban la alineación de exog ni de timestamps. |
| `feature_level_check.py/.log` | Captura los frames que el adapter pasa a `predict_df` y los featuriza con `_featurize_context_future` del backend: idénticos a los de una entrada directa (2 series, NaN, exog heterogénea). |
| `real_weights_v2_check.py/.log` | Con pesos reales sin token (checkpoint público TabPFN-v2 de Prior-Labs/TabPFN-v2-reg, 44 MB, no es el checkpoint por defecto del adapter): adapter = llamada directa (0.0), la predicción sí depende de la exog (13.3) y el control con exog desplazada 1 paso sí difiere (7.0). s1 sola o junto a s2 con covariable extra: 0.0 con los features de 1.4 y con los de 1.3. |
| `categorical_v2_check.py/.log` | Con pesos v2: una covariable category o str que fija un salto de nivel se usa igual que su versión numérica 0/1. |
| `rerun_capabilities_check.log` | Sección 1: el warning del backend no aparece en stderr con show_progress=False y sí con True. |
| `test_p1_pytest_warns.py/.log` | `pytest.warns` detecta el warning también con show_progress=False (el warning se emite; lo que se pierde es su visualización). |
| `contract_rerun.log`, `unit_tests_rerun.log`, `old_backend_1.3.0_rerun.log`, `old_backend_env.log` | Contrato 5 passed con 1.3.0/9.0.0 y con 1.4.0/9.2.0; unit tests 58 passed en ambos; smoke falla aquí por TabPFNLicenseError (sin token). |
| `rerun_probe_int_timestamps.log` | Timestamps enteros convertidos a datetime64[ns]; DeprecationWarning de `pd.Timedelta(days=n)`. |
| `hub_rerun.log`, `metadata_check_rerun.log`, `resolve_rerun.log` | Hub (sha 06bf2ba35c80, ficheros y tamaños), metadatos OK, resolución de model_id. |

Además: log del job 127875169000 (run 42212742430) leído por la API de GitHub: tabpfn 9.2.0 instalado,
contrato 5 passed, smoke 6 passed, caché de pesos restaurada 8332 B y guardada 7547 B.

## Veredictos

- TPF-F1 confirmado [probado].
- TPF-F2 corregido [probado]: la conclusión se mantiene, pero la evidencia del investigador no era
  discriminante (modelo aleatorio con salida constante e insensible a la exog). Re-probado con
  feature_level_check y con pesos reales TabPFN-v2.
- TPF-F3 confirmado [verificado en código]. Nota: sí cambian `worker/model_adapters/tabpfn_adapter.py`
  (soporte de ModelSpecs) y `gpu_worker.py` (ver hallazgo nuevo V1).
- TPF-F4 confirmado [verificado en código]; el cambio de predicciones entre 1.3 y 1.4 con exog es
  inferido (cambian los features de entrada).
- TPF-F5 confirmado [verificado en código]; existe la vía `model=TabPFNTSPipeline(compile=False)`,
  documentada como solo para tests.
- TPF-F6 corregido [probado]: el warning se emite; lo que se pierde es su visualización por defecto en
  stderr. `catch_warnings(record=True)` y `pytest.warns` lo siguen capturando.
- TPF-F7 confirmado [probado] (re-ejecutado con un entorno 1.3.0/9.0.0 nuevo).
- TPF-F8, F9, F10, F11, F12, F13, F17, F18, F19, F20 confirmados.
- TPF-F14 corregido [probado]: el aviso lo emite el constructor `pd.Timedelta(days=n)` (forma con
  keyword), no la suma; `pd.Timedelta(n, unit="D")` no avisa. El mismo código está en
  `TabICLAdapter._get_future_timestamps` (skforecast/foundation/_adapters.py línea 2631).
- TPF-F15 corregido [probado]: misma observación que F2; re-probado con pesos v2.
- TPF-F16 confirmado [probado]; nueva evidencia con pesos v2: category y str se usan con sentido.

Propuestas: P1 corregida (el test propuesto con `pytest.warns` pasa ya hoy sin el arreglo; hay que
comprobar stderr con capsys/capfd o `warnings.showwarning`). P2, P3, P4, P6, P7 confirmadas. P5
corregida (incluir TabICLAdapter; alternativa `pd.Timedelta(n, unit="D")`). P8 corregida (hay
argumentos nuevos para subir el mínimo, ver V1 y V2). P9 corregida (usar el checkpoint público v2 en
lugar del ModelSpecs aleatorio, que no discrimina).

## Hallazgos nuevos

- V1 [verificado en código]: en tabpfn-time-series 1.3.0, `GPUParallelWorker.predict` con varias GPU y
  al menos tantas series como workers llama a `_prediction_routine_per_gpu` sin `**kwargs`, así que se
  usan los cuantiles por defecto. El adapter hace `group[quantile_list]`: con cuantiles fuera de la
  rejilla por defecto (por ejemplo interval=[5, 95]) daría KeyError. 1.4.0 lo arregla (#197, CHANGELOG
  "Custom quantiles are no longer replaced by the defaults when predicting with several GPU workers").
  No probado (sin GPU). Relevante para P8.
- V2 [documentado]: el mínimo permitido (1.3.0) acepta tabpfn 9.0.0, que no tiene los arreglos de 9.1.0
  #1310 (TabPFN-3.5 extrapolaba mal en features de tiempo repetidos como `year`; cambia predicciones) y
  #1215 (cuantiles de FullSupportBarDistribution). 1.4.0 exige tabpfn>=9.2.0.
- V3 [probado]: el ModelSpecs aleatorio no sirve para tests de alineación; el checkpoint público
  Prior-Labs/TabPFN-v2-reg se descarga sin TABPFN_TOKEN con tabpfn 9.2.0 (V2 no está en `_HF_REPOS` de
  `_download_model`) y da un modelo sensible a exog y timestamps.
- V4 [verificado en código]: `FakeTabPFNTSPipeline.__init__` tampoco acepta `max_featurize_rows` (1.3).
- tabpfn-client 0.7.0/0.7.1 (2026-10-28/30) [documentado, CHANGELOG del sdist]: cambios internos de
  la API (jobs en segundo plano, thinking_effort); el alias v3.5_default sigue. Nada que afecte al
  adapter, que no llama a tabpfn-client directamente.
