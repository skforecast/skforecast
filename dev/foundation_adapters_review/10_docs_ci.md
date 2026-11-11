# 10. Guía de usuario y CI de compatibilidad

- **Rama propuesta**: `docs/foundation-guide-refresh`, creada desde `origin/0.27.x`.
- **Prioridad**: media para la re-ejecución de la guía (TPF-P4 es media en el informe); baja para los cambios del CI (TPF-P7 y NORI-P7 son de prioridad baja en el informe) [documentado]. Es el último PR de la serie: se abre cuando 01 a 09 están fusionados y, por lo explicado en 3.6, conviene fusionarlo poco antes del PR de release a `main`.
- **Adaptadores afectados**: en la guía, los nueve de la revisión (`T0Adapter`, `TSICLAdapter`, `TimesFM25Adapter`, `TimesFM3Adapter`, `ChronosAdapter`, `MoiraiAdapter`, `TabPFNAdapter`, `TabICLAdapter`, `NoriAdapter`), más `TimestralAdapter`, que no se revisó pero tiene sección propia en la guía y fila en la tabla comparativa [verificado en código]. En el CI de compatibilidad: `TabPFNAdapter` (caché de pesos) y `NoriAdapter` (segunda entrada para Nori-1-5).
- **Ficheros afectados**:
  - `docs/user_guides/foundation-forecasting-models.ipynb` (re-ejecución completa y repaso de las celdas comunes).
  - `docs/user_guides/foundation-forecasting-with-heterogeneous-series.ipynb` (repaso de las celdas 2 y 21 y re-ejecución).
  - `.github/workflows/foundation-backends.yml` (ruta de caché de TabPFN y matriz).
  - `tools/foundation_compat/adapters.toml`, `tools/foundation_compat/tests/contracts.py`, `tools/foundation_compat/tests/test_smoke.py`, `tools/foundation_compat/README.md`.
  - `skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py`.
  - `docs/releases/releases.md`.
- **Depende de**: 01 (`01_t0.md`), 02 (`02_common.md`), 03 (`03_tsicl.md`), 04 (`04_timesfm.md`), 05 (`05_chronos.md`), 06 (`06_moirai.md`), 07 (`07_tabicl.md`), 08 (`08_tabpfn.md`) y 09 (`09_nori.md`), todos fusionados en `0.27.x`. Esos PR cambian el texto de cada adaptador en la guía, los `backend_package` (mínimos de versión, también en `adapters.toml`) y los atributos de capacidad; este PR re-ejecuta, comprueba la coherencia final y hace los cambios de CI comunes.
- **Bloquea a**: ninguno.
- **Cierra issue**: ninguna.

### Por qué es un PR aparte

Cada PR de adaptador (01, 03 a 09) edita solo el texto de su adaptador en la guía (su fila de la tabla de la celda 11, su sección y, si toca, su fila de la tabla de la guía de series heterogéneas) y no re-ejecuta el cuaderno. Motivos:

- Una re-ejecución cambia las salidas y el `execution_count` de todas las celdas de código del `.ipynb` [inferido]: nueve PR que re-ejecutan el mismo cuaderno chocarían en el JSON aunque cada uno toque una sección distinta.
- La re-ejecución es cara: exige un único kernel con los diez backends instalados, porque la tabla de la celda 108 lee las variables `metrics_*` de todas las secciones [verificado en código]; el `TABPFN_TOKEN` del mantenedor para las celdas de TabPFN-TS (celda 78 comprueba `os.getenv("TABPFN_TOKEN")`) [verificado en código]; y un entorno que solo se resuelve forzando requisitos declarados de uni2ts (sección 1.4) [documentado].
- La tabla comparativa de la celda 108 solo es coherente si todos sus MAE salen de la misma ejecución, con las versiones de los backends que exigen los PR ya fusionados [inferido]. Hoy no lo es (sección 1.3).

Los cambios del CI de compatibilidad (TPF-P7, NORI-P7) van aquí porque tocan ficheros compartidos por todos los adaptadores (`adapters.toml`, la matriz del workflow, `contracts.py`, `test_smoke.py`), que los PR de adaptador también editan (mínimos de versión, contratos); hacerlos al final evita conflictos [inferido].

## 1. Estado en la revisión (2026-11-06)

Base de la revisión: rama `docs/foundation-adapters-review` desde `origin/0.27.x`, commit `d9b8f4f` (skforecast 0.27.0 en desarrollo) [verificado en código].

### 1.1 Versiones de los backends y checkpoints

| Adaptador | Versión probada antes | Última versión (fecha) | Checkpoints en el Hub | Requisitos relevantes |
|---|---|---|---|---|
| T0Adapter | tfc-t0 0.3.2, última que funciona según la issue #1385; el CI del 2026-11-06 instaló 0.6.0 y falló [probado] | 0.6.0 (2026-10-13) [documentado] | t0-alpha, t0-beta (necesita >=0.5) y t0-gamma (necesita >=0.6), apache-2.0, no gated, un solo commit por repo (2026-10-15); 4 repos ONNX que el prefijo acepta pero no cargan [probado] | runtime torch en el extra `tfc-t0[torch]`, que solo existe desde 0.6.0 [probado] |
| TSICLAdapter | tsicl 0.2.1 [probado] | 0.2.1 (2026-06-25); arreglo del rollout en main sin publicar [verificado en código] | `taharnbl/TS-ICL`, un solo checkpoint, mismo LFS sha256 antes y después del último commit, que solo cambia LICENSE (2026-07-07) [probado] | Python>=3.12, torch>=2.5.1,<2.10, numpy>=2.1.3, huggingface-hub>=1.19.0 (TSI-F13) [documentado] |
| TimesFM25Adapter | timesfm 3.0.2 [probado] | 3.0.2 (2026-09-09) [documentado] | `google/timesfm-2.5-200m-pytorch`, pesos en d418f3e (2025-10-01), sin cambios desde 2025-10-02 [probado] | sin mínimo declarado; con timesfm 2.0.x los cuantiles de los pasos >128 son erróneos (T25-F6) [probado]; D8 recomienda >=3.0 |
| TimesFM3Adapter | timesfm 3.0.2 [probado] | 3.0.2 (2026-09-09) [documentado] | `google/timesfm-3.0-pytorch`, sha 43046b85, mismo blob de pesos desde 2026-08-24 [probado] | timesfm declara huggingface_hub>=0.28.0, pero con 0.28 o 0.29 `import timesfm` falla (TV-1) [probado] |
| ChronosAdapter | chronos-forecasting 2.3.2 [probado] | 2.3.2 (2026-09-08) [documentado] | `amazon/chronos-2`, `autogluon/chronos-2`, `-small` y `-synth`, ninguno nuevo desde 2025-12-03 [probado] | mínimo real 2.2.0 (CHR-F3) [probado]; 2.2.x fija transformers<5 [documentado] |
| MoiraiAdapter | uni2ts 2.0.0 [probado] | 2.0.0 (2025-11-04) [documentado] | solo `Salesforce/moirai-2.0-R-small`, pesos sin cambios desde 2025-08-06; `-base` y `-large` dan 404 [probado] | torch>=2.1,<2.5, numpy~=1.26.0, scipy~=1.11.3, gluonts~=0.14.3 [documentado]; en Python>=3.13 se instala uni2ts 1.1.1 sin Moirai-2 (MOI-F3) [probado] |
| TabPFNAdapter | tabpfn-time-series 1.4.0 + tabpfn 9.2.0 [probado] | 1.4.0 (2026-10-28) [documentado] | `Prior-Labs/tabpfn_3_5`, sha 06bf2ba35c80 (2026-09-09), no gated; pesos de regresión `tabpfn-v3.5-20260909.safetensors` (876 MB); la descarga exige `TABPFN_TOKEN` [probado] | tabpfn>=9.2.0, gluonts>=0.16.0, datasets>=4 [documentado] |
| TabICLAdapter | tabicl 2.2.0 [probado] | 2.2.0 (2026-09-02); 2 commits en main sin publicar (c91f00d, d83091b) [probado] | `jingang/TabICL`, sin cambios de ficheros desde 2026-02-12 [probado] | mínimo real 2.0.3 (TAB-F2) [probado]; D20 recomienda >=2.2.0 |
| NoriAdapter | synthefy-nori 0.23.0 [probado] | 0.23.0 (2026-11-04) [documentado] | `Synthefy/Nori` (6M, 48.2 MB) y `Synthefy/Nori-1-5` (sha 79afc5a, 2026-10-20, 83.2 MB, apache-2.0, no gated, `min_synthefy_nori_version` 0.22.0 en su `config.json`) [probado] | torch>=2.8,<2.14, numpy>=2.0 (NORI-F16) [documentado] |

Evidencia: tabla resumen y secciones 3.x de `evidence/diagnosis_report.md`; `evidence/moirai/pypi_uni2ts.log`, `evidence/tsicl/pypi_info.log`, `evidence/nori/pypi_info.log`, `evidence/tabpfn/pypi_tabpfn-time-series.log`, `evidence/nori/hub_info.log`, `evidence/tabpfn/hub_info.log`.

### 1.2 Línea base del CI de compatibilidad

Run 42212742430 del workflow "Foundation Backends Compatibility" (rama 0.27.x, commit 09433c35, 2026-11-06, CPython 3.12.16) [probado]:

- t0 (job 127875169126): instaló tfc-t0 0.6.0 sin torch (`"torch": null`); contrato 2 fallidos y 1 pasado, smoke sin ejecutar, estado `failed` [probado]. Evidencia: `evidence/t0/ci_baseline_job_127875169126.log`. Lo arregla 01.
- timesfm25 (job 127875168998): contrato 6, smoke 5 + 1 skipped [probado] (`evidence/timesfm25/ci_baseline_job_127875168998.log`). timesfm3: contrato 3, smoke 6 [probado] (`evidence/timesfm3/ci_baseline.log`). chronos: contrato 3, smoke 6 [probado] (`evidence/chronos/ci_baseline.log`). moirai: contrato 5, smoke 5 + 1 skipped (`test_predict_output_when_exog`, `allow_exog=False`) [probado] (`evidence/moirai/ci_baseline.log`). tabicl: contrato 3, smoke 6 [probado] (`evidence/tabicl/ci_baseline.log`).
- tsicl: contrato 3, smoke 6; nori: contrato 4, smoke 6 [probado] (tabla resumen de `evidence/diagnosis_report.md`).
- tabpfn: contrato 5, smoke 6 con `TABPFN_TOKEN`; la caché de `~/.cache/huggingface` del job tabpfn restauró 8332 B ("Cache Size: ~0 MB") y guardó 7547 B [probado]. Evidencia: `evidence/tabpfn/ci_baseline.log`, `evidence/tabpfn/verification/verification.md`.
- timestral: fuera de la revisión.

### 1.3 Estado de la guía de usuario en la revisión

Numeración de celdas desde 0, como en la revisión.

- `docs/user_guides/foundation-forecasting-models.ipynb` tiene 111 celdas; la tabla comparativa es la celda 108 ("# Comparison of backtesting metrics"), que construye un `pd.DataFrame` con `metrics_chronos`, `metrics_timesfm`, `metrics_timesfm3`, `metrics_moirai`, `metrics_tabicl`, `metrics_tabpfn`, `metrics_t0`, `metrics_nori`, `metrics_tsicl` y `metrics_timestral`; la celda 107 ("### Model comparison") la introduce y la 109 es la nota "\* TimesFM-2.5 (200m) and Moirai-2.0-R (small) do not allow the inclusion of exogenous features." [verificado en código].
- La salida de la celda 108 muestra (MAE): Timestral (209M) 135.6705, TS-ICL 140.8760, TFC T0 145.5262, TimesFM-3.0 162.4352, Chronos-2 (small) 171.2670, Nori 185.7599, TabPFN-TS 186.4164, TimesFM-2.5 (200m)* 191.9283, Moirai-2.0-R (small)* 201.8164, TabICL 215.8026 [verificado en código].
- Metadatos del cuaderno: Python 3.12.3; la celda 19 ("# Libraries") imprime `torch version: 2.9.1+cpu`, sin CUDA ni MPS [verificado en código]. Los `execution_count` van de 2 a 69 sin huecos [verificado en código].
- Aun así, las salidas no corresponden a una misma versión del código: la celda 13 (`list_adapters(as_frame=True)`) ya muestra `Prior-Labs/tabpfn_3_5`, mientras que el LicenseWarning de la celda 81 dice "released under TabPFN License v1.0" y enlaza `Prior-Labs/tabpfn_3/blob/main/LICENSE` (TPF-F17) [verificado en código]. `_MODEL_LICENSES` (en `skforecast/foundation/_utils.py`, entrada `"priorlabs/tabpfn"`) da hoy "TabPFN-3.5 License v1.0 (non-commercial without an enterprise license)" con el enlace a `https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE` [verificado en código].
- La "Location" de ese LicenseWarning apunta a un `site-packages` de otra máquina (Windows), a `_adapters.py:2692`, que en `d9b8f4f` cae dentro de `TabICLAdapter`: la guía se ejecutó con una instalación no editable de otra versión de skforecast [verificado en código].
- Comprobaciones de corrección hechas en la revisión con la configuración de la tabla (vic_electricity horario, entrenamiento hasta 2014-11-30 23:59, folds de 24 pasos sin reentrenar, `context_length=500`, MAE), reproducida por `evidence/common/backtest_guide.py` [verificado en código]. Son referencias para comprobar que la re-ejecución es coherente, no comparaciones de precisión:
  - Chronos-2 (small) con exog: MAE 171.267, igual que la guía; chronos-forecasting 2.3.2, torch 2.15.0+cpu, Python 3.12.3, Linux, 4 CPU [probado] (`evidence/chronos/backtest_before.log`).
  - TimesFM-3.0 con exog: MAE 162.4352, igual que la guía; pliegues 0 y 29 reproducidos exactamente; torch 2.15.0+cpu (TFM3-F12) [probado] (`evidence/timesfm3/backtest_guide_exog.log`, `evidence/timesfm3/verification/check_guide_fold0.log`). Como la guía obtuvo esos dos valores con torch 2.9.1+cpu, la versión de torch no los altera [inferido].
  - T0 (t0-alpha) con exog: MAE 147.5073, y 147.5073 con `trim_leading_missing=True` (diferencia máxima 0.0 en 720 predicciones); tfc-t0 0.6.0, torch 2.15.0+cpu [probado] (`evidence/t0/backtest_guide_trim.log`). Se obtuvo con `evidence/t0/backtest_t0.py`, que usa los datos y folds de `backtest_guide.py` pero traduce la llamada del adaptador a 0.6.0 con `evidence/t0/t0_shim.py`, porque el `T0Adapter` de `d9b8f4f` no funciona con 0.6.0 [verificado en código]. La guía muestra 145.5262, obtenido con un backend anterior a 0.4 según el aviso `pip install "tfc-t0<0.4"` de la propia guía [inferido].
  - Nori (`Synthefy/Nori`) con exog: MAE 185.9507 antes y después de NORI-P2 (diferencia máxima 0.0 en 720 predicciones); synthefy-nori 0.23.0, torch 2.13.0+cpu; el backtest tardó 238.84 s antes de NORI-P2 y 330.86 s después, en 4 CPU [probado] (`evidence/nori/backtest_current.log`, `evidence/nori/backtest_patched.log`). La guía muestra 185.7599; la versión de synthefy-nori de esa ejecución no consta [inferido].
  - Moirai, TabICL, TabPFN-TS, TimesFM 2.5, TS-ICL y Timestral: sin referencia de la revisión para la tabla.

### 1.4 Entorno único para la guía: requisitos incompatibles

La guía se ejecuta en un solo kernel y los requisitos declarados de los backends no se pueden satisfacer a la vez [documentado]:

- uni2ts 2.0.0 declara `torch<2.5,>=2.1`, `numpy~=1.26.0`, `scipy~=1.11.3`, `gluonts~=0.14.3`, `datasets~=2.17.1`, `einops==0.7.*`, `hydra-core==1.3` y `python-dotenv==1.0.0` (`evidence/moirai/pypi_uni2ts.log`).
- tsicl 0.2.1 declara Python>=3.12, `torch<2.10,>=2.5.1`, `numpy>=2.1.3`, `einops>=0.8.0` y `hydra-core>=1.3.2` (`evidence/tsicl/pypi_info.log`, TSI-F13).
- synthefy-nori 0.23.0 declara `torch<2.14,>=2.8` y `numpy>=2.0` (`evidence/nori/pypi_info.log`, NORI-F16).
- tabpfn-time-series 1.4.0 declara `gluonts>=0.16.0`, `datasets>=4`, `pandas>=2.2` y `python-dotenv>=1.1.0` (`evidence/tabpfn/pypi_tabpfn-time-series.log`).
- skforecast exige scipy>=1.12; el CI ya pasa a uni2ts el override `scipy>=1.12` (MOI-F2) [probado]. Con pip, `pip install uni2ts` baja scipy a 1.11.4 y pandas a 2.1.4 (gluonts 0.14.4 fija pandas<2.2) (MOI-F4) [probado].

Consecuencia: torch 2.8.x o 2.9.x es la única serie dentro de los rangos de tsicl y synthefy-nori, y uni2ts solo se instala en ese kernel forzando sus requisitos de torch, numpy, scipy, gluonts, datasets, einops, hydra-core y python-dotenv [inferido]. La ejecución anterior de la guía usó torch 2.9.1+cpu (celda 19), así que también instaló uni2ts fuera de sus requisitos declarados [inferido]. Moirai con torch>=2.5 no se probó en la revisión: su entorno usa torch 2.4.1+cpu, numpy 1.26.4, scipy 1.17.1 (override) y gluonts 0.14.3 [probado].

### 1.5 Antes de empezar

Este plan puede quedar desfasado. Antes de crear la rama:

1. Comprobar que 01 a 09 están fusionados en `0.27.x` y que el último run programado del CI de compatibilidad está en verde. El run programado se lanza desde `main` y comprueba la rama de release más alta [verificado en código: comentario y job `setup` de `.github/workflows/foundation-backends.yml`], así que hay que mirar el resumen del run, que dice la rama comprobada:
   ```bash
   git fetch origin
   gh pr list --base 0.27.x --state merged --search "foundation" --limit 20
   gh run list --workflow foundation-backends.yml --event schedule --limit 3
   gh run view <run-id>
   ```
2. Volver a consultar la última versión de cada backend (los scripts solo usan la biblioteca estándar):
   ```bash
   for pkg in tfc-t0 tsicl timesfm chronos-forecasting uni2ts tabpfn-time-series tabpfn tabicl synthefy-nori; do
       python dev/foundation_adapters_review/evidence/common/pypi_info.py "$pkg" --since 2026-11-06
   done
   ```
3. Volver a consultar los checkpoints del Hub:
   ```bash
   H=dev/foundation_adapters_review/evidence/common/hub_info.py
   python $H theforecastingcompany --prefix t0
   python $H taharnbl --prefix TS-ICL
   python $H google --prefix timesfm
   python $H amazon --prefix chronos-2
   python $H autogluon --prefix chronos-2
   python $H Salesforce --prefix moirai-2
   python $H Prior-Labs
   python $H --repo jingang/TabICL
   python $H Synthefy --prefix Nori
   ```
4. Qué hacer si algo cambió:
   - Versión nueva de un backend: si el último run del CI con esa versión está en verde, re-ejecutar la guía con ella (la guía debe mostrar la última versión) y anotarla en la descripción del PR. Si el CI falla, no arreglar el adaptador en este PR: abrir o actualizar la issue `backend-compat` y esperar a un PR del adaptador.
   - Cambian los requisitos de torch, numpy, gluonts, datasets, einops, hydra-core o python-dotenv de uni2ts, tsicl, synthefy-nori o tabpfn-time-series: recalcular los overrides de la sección 3.1.
   - `Synthefy/Nori-1-5` cambia de sha o de `min_synthefy_nori_version`: revisar el `backend_package` de la sección 3.6.
   - Checkpoint nuevo (por ejemplo Moirai-2 base o un TimesFM nuevo): no se añade aquí; es material de una revisión del adaptador.

## 2. Hallazgos

Este plan no es dueño de ningún id de la revisión. Implementa o comprueba los siguientes, cuyo detalle completo está en el plan del adaptador correspondiente:

- **TPF-F17** [verificado en código] Las salidas de TabPFN-TS de la guía (celdas 80 a 83 y el MAE 186.4164 de la celda 108) son anteriores a tabpfn-time-series 1.3: el LicenseWarning de la celda 81 tiene el texto antiguo ("TabPFN License v1.0") y enlaza `Prior-Labs/tabpfn_3`; se generaron con otro backend y otros pesos. El resto de textos de TabPFN (fila de la celda 11, celda 77, `skills/foundation-forecasting/references/adapter-parameters.md` línea 123, `tools/ai/llms-base.txt`) es correcto. Evidencia: `evidence/tabpfn/verification/verification.md`, sección 3.7 de `evidence/diagnosis_report.md`. Dueño: 08; lo implementa este plan (TPF-P4).
- **TPF-F11** [verificado en código] `weights_in_hf_cache=False` es veraz: tabpfn 9.2.0 descarga los pesos con `hf_hub_download(local_dir=...)` a su propia caché (`get_cache_dir` en `model_loading.py`: `~/.cache/tabpfn` en Linux, o `$XDG_CACHE_HOME/tabpfn`, o `TABPFN_MODEL_CACHE_DIR`), y el workflow solo cachea `~/.cache/huggingface` (8332 B restaurados y 7547 B guardados para tabpfn). Que cada ejecución vuelva a descargar los 876 MB de pesos es [inferido] (no hay log de descarga), coherente con el tamaño de la caché. Evidencia: `evidence/tabpfn/ci_baseline.log`, `evidence/tabpfn/verification/verification.md`. Dueño: 08; lo implementa este plan (TPF-P7).
- **NORI-F4** [probado] `Synthefy/Nori-1-5` (sha 79afc5a, 2026-10-20, 83.2 MB, apache-2.0, no gated) se enruta a `NoriAdapter` y funciona con 0.23.0; su `config.json` declara `min_synthefy_nori_version` 0.22.0, y con 0.21.1 falla con "TypeError: get_x_encoder() got an unexpected keyword argument 'prompt_quantile_ple_bins'" (con 0.21.1 el nombre de registro `nori-1-5` da además un CheckpointAccessError engañoso). Ningún código de 0.23.0 lee `min_synthefy_nori_version`. Evidencia: `evidence/nori/old_backend_nori15.log`, `evidence/nori/verification/old_backend_check.log`. Dueño: 09 (NORI-P3, D24); este plan añade su cobertura en el CI (NORI-P7).
- **NORI-F18** [verificado en código] El fake `FakeNoriRegressor` diverge del backend (lo trata 09, NORI-P8); `adapters.toml` solo prueba `Synthefy/Nori` (6M), así que Nori-1-5, otra arquitectura con otro mínimo del backend, no tiene cobertura semanal; y los smoke no comprueban el orden con cuantiles desordenados. Evidencia: `evidence/nori/verification/verification.md`. Dueño: 09; la parte del CI la implementa este plan (NORI-P7).
- **T0-F13** [verificado en código] La frase de `tools/foundation_compat/README.md` sobre un runtime MLX "with other signatures" está desfasada desde tfc-t0 0.6.0 (MLX ejecuta el mismo `predict`). Evidencia: sección 3.1 de `evidence/diagnosis_report.md`; XC-12 de `evidence/cross_cutting.md`. Dueño: 01 (T0-P6); este plan la corrige solo si sigue en el README al empezar.
- **TFM3-F9** [verificado en código] timesfm 3.0.2 añade un backend MLX (`timesfm3.mlx`) que el adaptador no usa; según XC-12 los dos runtimes MLX tienen la misma API, así que la misma frase del README tampoco describe bien timesfm. Evidencia: sección 3.4 de `evidence/diagnosis_report.md`; XC-12 de `evidence/cross_cutting.md`. Dueño: 01 solo quita `tfc-t0` de la frase (T0-P6); 04 deja la mitad de `timesfm` fuera de su alcance (su sección 8), así que este plan reescribe la frase restante (3.7).
- **TSI-F17 (corregido)** [probado] y **X-8** [probado] La guía de series heterogéneas afirma en la celda 2 que juntar series en lote no cambia la predicción ("Not for Chronos-2, TS-ICL and TabICL") y en la celda 21 "Past-only covariates: Yes" para TS-ICL; ambas cosas son falsas para TS-ICL con tsicl 0.2.1, y la independencia del lote es falsa para T0 con covariables heterogéneas y tiene matices en TimesFM 3 (relleno por `per_core_batch_size`), Chronos-2 (cross-learning por `batch_size`) y T0 (relleno leído como MISSING). Evidencia: `evidence/tsicl/verification/verification.md`, XC-6 de `evidence/cross_cutting.md`. Dueños: 03 (TS-ICL), 01 (T0), 04 (TimesFM 3), 05 (Chronos-2). Este plan solo comprueba que el texto final de esas celdas es coherente tras todos los PR.
- **X-1** [probado] Una ventana de contexto toda NaN llega hoy al backend (Chronos ±1e5, Moirai ~0, TimesFM 2.5 ValueError críptico o ceros según numpy), sin aviso de skforecast; la celda 2 de la guía de series heterogéneas dice "A context window that is entirely NaN is rejected", que solo será cierto tras 02 (D11). Evidencia: `evidence/cross_cutting/probe_shared.log`. Dueño: 02; este plan comprueba el texto final y re-ejecuta esa guía.

Observaciones de la preparación de este plan (sin id en el informe; se tratan aquí porque solo afectan a ficheros comunes):

- [verificado en código] `tools/foundation_compat/tests/conftest.py` parametriza el contrato con `BACKEND_CONTRACTS.get(<clave de adapters.toml>)` y `test_backend_has_contract` (en `test_contract.py`) falla si la clave no está: una segunda entrada en `adapters.toml` exige también una clave en `tools/foundation_compat/tests/contracts.py`, además de la matriz del workflow (la verificación de NORI-P7 solo citaba la matriz). El README ya lo dice para un adaptador nuevo ("the contract test until the second is").
- [verificado en código] `test_adapters_toml_entries_match_the_adapter_classes` exige que cada entrada declare el mismo `backend_package` que la clase del adaptador; `test_adapters_toml_covers_every_registered_adapter` ya admite más de una entrada por adaptador ("at least one entry"); `run.py` acepta como `--adapter` cualquier clave de `adapters.toml` y `summarize.py` escribe una fila por clave.
- [verificado en código] `summarize.py` recorre las entradas del `adapters.toml` de la rama comprobada y cuenta como fallo ("no result") una entrada sin JSON de resultado. El run programado usa el workflow de `main` [documentado: GitHub ejecuta los workflows programados desde la rama por defecto] y comprueba `0.27.x`; hasta que la release lleve este PR a `main`, la matriz de `main` no tendrá `nori15` y cada run programado contará `nori15: no result` como fallo y abrirá o actualizará la issue `backend-compat` [inferido]. Ver 3.6.
- [verificado en código] `tools/foundation_compat/README.md` dice que el workflow "runs one job per adapter"; con una segunda entrada de Nori será un job por entrada de `adapters.toml`.
- [verificado en código] La celda 25 de la guía principal enlaza la referencia de API solo de `ChronosAdapter`, `TimesFM25Adapter`, `TimesFM3Adapter`, `MoiraiAdapter` y `TabICLAdapter`; faltan `TabPFNAdapter`, `T0Adapter`, `NoriAdapter`, `TSICLAdapter` y `TimestralAdapter`, que sí están en `docs/api/FoundationModel.md`.
- [verificado en código] `tools/docs/execute_notebooks/execute_notebooks.py` excluye de los barridos de carpeta los dos cuadernos foundation (`EXCLUDE_NOTEBOOKS`, "Specific environment"), pero ejecuta un cuaderno pasado explícitamente; el tiempo máximo por celda es `EXECUTION_TIMEOUT = 600` segundos y el kernel es `KERNEL_NAME = "python3"`.
- [verificado en código] Ningún otro fichero del repositorio fuera de `dev/` cita los MAE de la tabla de la celda 108 (búsqueda de los diez valores en `.md`, `.txt`, `.ipynb` y `.py`).

Descartado o abierto:

- "TabPFN-TS-3 sigue en Prior-Labs/tabpfn_3": no verificado (upstream solo cita "TabPFN-TS-3 ship config" en un docstring; los pesos que usa son `Prior-Labs/tabpfn_3_5`); no afecta a este plan, que usa el enlace de `_MODEL_LICENSES`.

## 3. Cambios a implementar

### 3.1 TPF-P4. Re-ejecutar la guía principal una sola vez (incluidas las celdas de TabPFN-TS con `TABPFN_TOKEN`)

- **Qué**: re-ejecutar `docs/user_guides/foundation-forecasting-models.ipynb` completo, una vez, con las versiones de los backends que exigen los PR ya fusionados, de modo que todas las salidas (incluidas las celdas 78 a 83 de TabPFN-TS y la tabla de la celda 108) salgan del mismo kernel y de la misma versión de skforecast. TPF-P4 pedía re-ejecutar solo las celdas 78 a 83 y la 108 con tabpfn-time-series 1.4.0 [documentado]; aquí se amplía a todo el cuaderno porque las salidas actuales ya mezclan versiones (sección 1.3) y porque 01 a 09 cambian salidas de otras secciones (por ejemplo `list_adapters` en la celda 13 y el MAE de T0) [inferido].
- **Entorno** (propuesta no ejecutada en la revisión) [inferido]:
  - Python 3.12 (la única versión que sirve a la vez a tsicl, >=3.12, y a uni2ts 2.0.0, que en la práctica exige <=3.12), en CPU como la ejecución anterior, con skforecast del checkout en modo editable.
  - Los backends con los mismos especificadores que el CI, leídos de `tools/foundation_compat/adapters.toml` (que, tras los PR de adaptador, coinciden con cada `backend_package`), en un fichero de requisitos fuera del repositorio.
  - Overrides de uv deducidos de la sección 1.4 (el resolvedor puede pedir más; si es así, añadirlos y anotarlos en el PR): `torch>=2.8,<2.10`, `numpy>=2.1.3`, `scipy>=1.12`, `gluonts>=0.16`, `datasets>=4`, `einops>=0.8`, `hydra-core>=1.3.2`, `python-dotenv>=1.1.0`.
  ```bash
  uv venv ~/.venvs/skforecast-fm-guide --python 3.12
  python -c 'import tomllib; d = tomllib.load(open("tools/foundation_compat/adapters.toml", "rb")); print("\n".join(sorted({v["backend_package"] for v in d.values()})))' \
      > ~/.venvs/skforecast-fm-guide-requirements.txt
  printf "torch>=2.8,<2.10\nnumpy>=2.1.3\nscipy>=1.12\ngluonts>=0.16\ndatasets>=4\neinops>=0.8\nhydra-core>=1.3.2\npython-dotenv>=1.1.0\n" \
      > ~/.venvs/skforecast-fm-guide-overrides.txt
  uv pip install --python ~/.venvs/skforecast-fm-guide/bin/python --torch-backend cpu \
      --override ~/.venvs/skforecast-fm-guide-overrides.txt \
      -e ".[plotting]" papermill ipykernel -r ~/.venvs/skforecast-fm-guide-requirements.txt
  ```
  - Comprobar que el kernel importa el checkout y no un `site-packages` (la ejecución anterior no lo hacía, sección 1.3): `cd docs/user_guides && ~/.venvs/skforecast-fm-guide/bin/python -c "import skforecast; print(skforecast.__file__)"`, y la línea `Kernel skf` de la cabecera del ejecutor (avisa cuando no es el código del repositorio) [verificado en código]. Con `ipykernel` instalado en el entorno, el kernelspec `python3` se resuelve al intérprete que lanza el ejecutor salvo que haya un kernelspec `python3` de usuario que lo sobrescriba [inferido]; la línea `Kernel` de la cabecera lo confirma.
- **Ejecución**, desde la raíz del repositorio y con el token del mantenedor en el entorno (nunca en el cuaderno ni en el PR). `TABPFN_DISABLE_TELEMETRY=1 TABPFN_NO_BROWSER=1` son las variables que la revisión usó con TabPFN (`evidence/README.md`) [documentado]:
  ```bash
  export TABPFN_TOKEN=...   # token de Prior Labs del mantenedor
  TABPFN_DISABLE_TELEMETRY=1 TABPFN_NO_BROWSER=1 \
      ~/.venvs/skforecast-fm-guide/bin/python tools/docs/execute_notebooks/execute_notebooks.py \
      user_guides/foundation-forecasting-models.ipynb
  ```
  El cuaderno está en `EXCLUDE_NOTEBOOKS`, por eso hay que pasarlo explícitamente [verificado en código]. El log queda en `tools/docs/execute_notebooks/logs/` (ignorado por git) [verificado en código].
- **Qué revisar en las salidas**:
  - celda 78 imprime `True`;
  - celda 81: LicenseWarning con "TabPFN-3.5 License v1.0 (non-commercial without an enterprise license)" y enlace a `https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE` [verificado en código: `_MODEL_LICENSES`]; su "Location" no apunta a un `site-packages`;
  - celda 13 (`list_adapters(as_frame=True)`) con los `backend_package` y atributos de capacidad finales de 01 a 09 (incluido el atributo nuevo de agrupación de 03, si `list_adapters` lo muestra);
  - celda 19 con la versión de torch del entorno;
  - celda 108 con diez filas numéricas, coherentes con la sección 4.3.
- **Por qué**: TPF-F17; coherencia de la tabla de la celda 108 tras 01 a 09.
- **Riesgo**: cambian los MAE mostrados de TabPFN-TS (pesos TabPFN-3.5 en lugar de los anteriores a 1.3) y de T0 (backend 0.6), y pueden cambiar los de otros modelos por las versiones nuevas o por decisiones de otros planes (por ejemplo D6 (b) en TimesFM 3 cambia predicciones) [inferido]; no es una comparación entre versiones y no se presenta como tal. Moirai se ejecuta con torch, numpy, gluonts y datasets fuera de lo que declara uni2ts: hay que comprobarlo (sección 4.3) antes de publicar [inferido]. Una celda que supere 600 s aborta el cuaderno [verificado en código]; el backtest de Nori tardó 238.84 s antes de NORI-P2 y 330.86 s después, en 4 CPU, en la revisión [probado] (`evidence/nori/backtest_current.log`, `evidence/nori/backtest_patched.log`), y el de Timestral (209M) no se midió.
- **Rompe compatibilidad**: no.
- **Decisiones**: ninguna propia; los textos que se re-ejecutan dependen de D19 (comando de instalación de TabPFN-TS en la celda 11) y de las decisiones de cada PR de adaptador.

### 3.2 Repaso de coherencia de las celdas comunes de la guía principal (sin id)

Las celdas que no pertenecen a ningún adaptador no las edita ningún PR de adaptador. Antes de re-ejecutar:

- **Celda 11** (tabla HTML "Supported Foundation Models", celda 11 en la revisión): cada PR de adaptador edita su fila. Aquí solo se comprueba, fila a fila, que el comando de instalación coincide con el `backend_package` final de cada adaptador y va entrecomillado cuando lleva `>=`, `<` o `@` (en la revisión, la fila de T0 mostraba `pip install "tfc-t0<0.4"` y la de TabPFN-TS `pip install "tabpfn-time-series>=1.3"`) [verificado en código]. Cada comando aparece tres veces: en `<code>`, en el atributo `data-copy` del botón y en su `aria-label`; los caracteres especiales se escapan (`&lt;`, `&gt;`, `&quot;`) o el atributo va entre comillas simples, como en la fila actual de T0 [verificado en código]; el `aria-label` de la fila de TabPFN-TS no lleva hoy las comillas (`Copy command: pip install tabpfn-time-series&gt;=1.3`, también en `origin/0.27.x`) y se corrige aquí [verificado en código]. Si alguna fila no coincide, corregirla aquí y anotarlo en la descripción del PR.
- **Celda 13** (`list_adapters`): se regenera con la re-ejecución; no tiene texto propio.
- **Celda 14** (Tip "All models run on the CPU... The MPS backend is also detected automatically"): no se toca salvo que contradiga lo que dejen 03 (MPS en TS-ICL) y 04 (TimesFM 2.5 sin MPS); si es así, una frase corta que remita a la tabla [inferido].
- **Celda 17** ("Exogenous Variables (`exog`)"): la lista de modelos con covariables (Chronos, TimesFM 3.0, TabICL, TabPFN-TS, TFC-T0, Nori, TS-ICL, Timestral) sigue siendo correcta mientras ningún PR cambie `allow_exog` [inferido]; comprobarlo con la salida nueva de la celda 13.
- **Celda 25** (argumentos adicionales del adaptador): completar la lista de enlaces a la referencia de API con `TabPFNAdapter`, `T0Adapter`, `NoriAdapter`, `TSICLAdapter` y `TimestralAdapter`, con el mismo formato que los existentes (`../api/FoundationModel.html#skforecast.foundation._adapters.<Clase>`) [verificado en código].
- **Celdas 107 y 109** (introducción de la tabla y nota "\* TimesFM-2.5 (200m) and Moirai-2.0-R (small) do not allow the inclusion of exogenous features."): siguen siendo correctas mientras D10 (XReg de TimesFM 2.5) y D16 (covariables en Moirai) se resuelvan como recomienda la revisión [inferido]. La celda 110 (aviso de fuga de datos) no cambia.
- No se añaden a la guía tablas de versiones ni explicaciones de la re-ejecución (regla de guías simples de `.claude/rules/docs.md`); las versiones usadas van en la descripción del PR.

### 3.3 Guía de series heterogéneas (sin id; comprueba TSI-F17, X-1 y X-8)

- **Qué**: tras 01 a 09, leer las celdas 2 ("## In short") y 21 ("## Reference: requirements and limitations per backend") de `docs/user_guides/foundation-forecasting-with-heterogeneous-series.ipynb` y comprobar que dicen lo mismo que los atributos y el comportamiento finales [verificado en código: textos actuales]:
  - celda 2, fila "Does batching series together change their forecasts?" (hoy "Not for Chronos-2, TS-ICL and TabICL"): coherente con 01, 03, 04 y 05 (X-8, TSI-F17);
  - celda 2, fila "Can the target contain NaN?" ("A context window that is entirely NaN is rejected") y la lista común de la celda 21 ("A series cannot be entirely NaN"): coherentes con 02 (X-1);
  - celda 21, tabla de capacidades: fila de TS-ICL ("Past-only covariates: Yes", "Grouped") y su nota ("Past-only covariates are supported"), según 03; fila y nota de TFC-T0 ("Single call", "pools the columns of all series"), según 01 (T0-P3, D3); fila y nota de Synthefy Nori ("Rows with NaN target or NaN feature dropped by the adapter"), según 09 (NORI-P2, D23); nota de Chronos-2 sobre `batch_size` si 05 la añade;
  - celdas 19 y 20 ("How it works"): si 03 añade un atributo de capacidad nuevo, que la lista de atributos sea coherente.
  Si un PR dejó una frase incoherente, corregirla aquí con el texto que fijó ese PR.
- **Re-ejecutar** el cuaderno una vez con el mismo entorno: usa solo `autogluon/chronos-2-small` (celda 11) con covariables numéricas (`sin_day_of_week`, `cos_day_of_week`, `air_temperature`, `wind_speed`) [verificado en código], así que no se esperan cambios de predicción [inferido]. La re-ejecución comprueba que la validación nueva de 02 (contexto todo NaN) no rompe el `predict` de la celda 12 ni el backtesting de la celda 15 con los bloques NaN de `id_1003`, y que los textos de las celdas 13 y 16 siguen describiendo la salida.
  ```bash
  ~/.venvs/skforecast-fm-guide/bin/python tools/docs/execute_notebooks/execute_notebooks.py \
      user_guides/foundation-forecasting-with-heterogeneous-series.ipynb
  ```
- **Por qué**: TSI-F17, X-1, X-8.
- **Riesgo**: bajo [inferido]. **Rompe compatibilidad**: no.

### 3.4 Otras páginas que mencionan adaptadores (sin cambio previsto)

- `docs/user_guides/comparing-forecasting-models.ipynb`: usa solo `FoundationModel("autogluon/chronos-2-small", device_map="cpu")` con covariables y sin columnas `category` [verificado en código]; está en `EXCLUDE_NOTEBOOKS` ("Specific environment (keras, chronos)") [verificado en código]. No se esperan cambios de predicción por la serie [inferido]; no se re-ejecuta, basta `--check`.
- `docs/introduction-forecasting/introduction-forecasting.md`, `docs/user_guides/global-forecasting-overview.md`, `docs/user_guides/probabilistic-forecasting-overview.md`, `docs/quick-start/ai-assisted-forecasting.md`, `docs/skforecast-ai/skforecast-ai.md` y los cuadernos `input-data`, `calendar-features`, `save-load-forecaster`, `migrate-from-prophet` y `hyperparameter-tuning-and-lags-selection`: solo nombran los modelos o `ForecasterFoundation` [verificado en código]; no cambian salvo que un PR renombre un modelo.
- `docs/api/FoundationModel.md`: ya lista los diez adaptadores [verificado en código]; los cambios de docstring de cada PR se publican solos.

### 3.5 TPF-P7. Cachear los pesos de TabPFN en el workflow

- **Qué**: en `.github/workflows/foundation-backends.yml`, paso "Cache model weights" del job `backend`, añadir `~/.cache/tabpfn` a `path` (ruta concreta según la verificación) y un comentario:
  ```yaml
      - name: Cache model weights
        uses: actions/cache@v6
        with:
          # TabPFN stores its weights in its own cache, not in the Hugging Face one.
          path: |
            ~/.cache/huggingface
            ~/.cache/tabpfn
          key: foundation-weights-${{ matrix.adapter }}-${{ hashFiles('tools/foundation_compat/adapters.toml') }}
          restore-keys: |
            foundation-weights-${{ matrix.adapter }}-
  ```
  Alternativa equivalente: fijar `TABPFN_MODEL_CACHE_DIR` a una carpeta dentro de `~/.cache/huggingface` en el paso "Check the adapter against its backend" [verificado en código: `get_cache_dir` de tabpfn 9.2.0]; se prefiere la ruta explícita porque no depende de una variable del backend [inferido].
- **Por qué**: TPF-F11.
- **Riesgo**: unos 876 MB más en el presupuesto de caché de Actions del repositorio, compartido con las demás entradas [inferido]. Cambiar `path` cambia la versión de las cachés de `actions/cache`, de modo que no se restaura ninguna caché anterior aunque coincida `restore-keys` [documentado], y la clave también cambia porque incluye `hashFiles('tools/foundation_compat/adapters.toml')` y este PR cambia ese fichero [verificado en código]: la primera ejecución descarga de nuevo los pesos de todas las entradas [inferido]. Para los demás jobs `~/.cache/tabpfn` no existe y solo se guarda la otra ruta [inferido]. El run programado usa el YAML de `main`, así que la caché nueva solo se aplica a la ejecución semanal tras la release [inferido].
- **Rompe compatibilidad**: no (solo herramientas). **Decisiones**: ninguna.

### 3.6 NORI-P7. Segunda entrada de compatibilidad para Nori-1-5 y orden de cuantiles (D26)

- **Qué** (opción (a) de D26):
  - `tools/foundation_compat/adapters.toml`: nueva tabla con el mismo `adapter` y `backend_package` que `[nori]`; si 09 adoptó D24 (a), ese valor es `synthefy-nori>=0.22` en las dos entradas [verificado en código: `test_adapters_toml_entries_match_the_adapter_classes`]:
    ```toml
    # Second checkpoint of NoriAdapter: another architecture that needs a newer
    # backend (min_synthefy_nori_version 0.22.0 in its config.json).
    [nori15]
    adapter = "NoriAdapter"
    backend_package = "synthefy-nori>=0.22"  # same value as NoriAdapter.backend_package
    model_id = "Synthefy/Nori-1-5"
    ```
    Si 02 introduce la tabla central de ids no soportados (D-REG), comprobar que `Synthefy/Nori-1-5` sigue aceptado (solo debe rechazarse `Synthefy/Nori-Causal`, NORI-P1).
  - `.github/workflows/foundation-backends.yml`: añadir `nori15` a `matrix.adapter` (lo exige `test_workflow_matrix_matches_adapters_toml`, que lee la matriz en una sola línea `adapter: [...]`) [verificado en código].
  - `tools/foundation_compat/tests/contracts.py`: alias para la nueva clave, con un comentario, porque el backend y sus llamadas son los mismos [verificado en código: `conftest.py`, `test_backend_has_contract`]:
    ```python
    # Same backend calls as `nori`: only the checkpoint of the smoke tests changes.
    BACKEND_CONTRACTS["nori15"] = BACKEND_CONTRACTS["nori"]
    ```
  - `tools/foundation_compat/tests/test_smoke.py`: la parte de NORI-P7 sobre el orden de los cuantiles, genérica para todas las entradas: un test `test_predict_quantiles_output_when_unordered_quantiles` que llame a `forecaster.predict_quantiles(steps=STEPS, quantiles=[0.9, 0.1, 0.5])` sobre una serie y compruebe que las columnas son `["level", "q_0.9", "q_0.1", "q_0.5"]` y que `q_0.1 <= q_0.5 <= q_0.9` en cada fila. Los tres niveles están en el `SUPPORTED_QUANTILES` de todos los adaptadores que declaran uno (TimesFM 2.5, TimesFM 3, Moirai, TS-ICL, Timestral) y no están entre los 8 niveles que tsicl 0.2.1 rechaza [verificado en código; `evidence/tsicl/check_quantile_grid.log`]. `FoundationModel` nombra las columnas en el orden pedido (`col_names = [f"q_{q}" for q in quantiles]` en `predict`, aprox. línea 1310 en d9b8f4f) [verificado en código], y la revisión probó con Nori que los niveles desordenados vuelven en el orden pedido, con diferencia 0.0 frente a la llamada directa [probado] (`evidence/nori/compare_direct.log`, `evidence/nori/verification/verify_nori.log`). Que pase en todas las entradas es [inferido]: si falla en alguna, es un hallazgo nuevo; abrir una issue y no arreglar el adaptador en este PR.
  - `tools/foundation_compat/tests/test_smoke.py`: el smoke con NaN en la exog del contexto propuesto por NORI-P2 (09, sección 4), solo si 09 adoptó D23 (a): un test `test_predict_output_when_exog_has_nan` que, saltado si `get_model_info(backend_config["model_id"]).allow_exog` es False (como `test_predict_output_when_exog`), haga `fit` con una copia de `exog` con NaN en una de cada cuatro filas y `predict(steps=STEPS, exog=exog_future)`, y compruebe forma, índice y valores finitos. Con Nori lo demostró `verify_nori.py` (sección D) [probado] (`evidence/nori/verification/verify_nori.log`); que todos los backends con exog acepten NaN en la exog del contexto no se ha comprobado [inferido]: si falla en alguna entrada, es un hallazgo nuevo; abrir una issue y no arreglar el adaptador en este PR.
  - `skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py`: test nuevo `test_contracts_cover_every_adapters_toml_entry`, que cargue `tools/foundation_compat/tests/contracts.py` por ruta (`importlib.util.spec_from_file_location`; el fichero no importa nada [verificado en código]) y compruebe `set(BACKEND_CONTRACTS) == set(load_adapters())`, para que una entrada sin contrato falle en los tests unitarios y no solo en el CI semanal [inferido]. Opcional: comprobar que dos entradas del mismo adaptador no repiten `model_id`.
- **Por qué**: NORI-F4, NORI-F18 (y la corrección de la verificación a NORI-P7: una entrada nueva obliga también a la matriz del workflow).
- **Riesgo**:
  - un job más por semana (83.2 MB de pesos, cacheados en `~/.cache/huggingface` con su propia clave `foundation-weights-nori15-...`) [inferido];
  - entre la fusión de este PR en `0.27.x` y la release, el run programado (YAML de `main`, sin `nori15` en la matriz) comprobará `0.27.x`, cuyo `adapters.toml` sí tiene `nori15`; `summarize.py` lo contará como "no result" y abrirá o actualizará la issue `backend-compat` [inferido a partir de `summarize.py` y del workflow, verificados en código]. Mitigación recomendada: fusionar este PR justo antes del PR de release a `main` (es el último de la serie) y, si entra un run programado entre medias, anotarlo en la issue; el PR de release a `main` ya ejecuta el workflow con la matriz nueva [verificado en código: disparador `pull_request` a `main`];
  - si D24 no se adoptó, la entrada instala la última synthefy-nori (0.23.0 en la revisión, con la que Nori-1-5 funciona) [probado], pero no protege a quien tenga <0.22.
- **Rompe compatibilidad**: no. **Depende de**: D26 (y D24 para el valor de `backend_package`).

### 3.7 README de `tools/foundation_compat` (sin id; recoge T0-F13 y TFM3-F9 si siguen pendientes)

- Sección "Workflow": "runs one job per adapter" pasa a "runs one job per entry of `adapters.toml`" [verificado en código].
- Sección "Maintenance": añadir un punto para un segundo checkpoint de un adaptador que necesita su propia comprobación (otra arquitectura u otro mínimo del backend): nueva entrada en `adapters.toml` con el mismo `adapter` y `backend_package`, su clave en la matriz del workflow y un alias en `tests/contracts.py`; el test unitario `test_backends_compat_config.py` falla hasta completar los tres (con el test nuevo de 3.6). Añadir que el run programado usa la matriz de `main`, así que una entrada nueva solo se comprueba cada semana tras la release.
- Una frase sobre la caché: los pesos se guardan entre ejecuciones en `~/.cache/huggingface` y, para TabPFN, en `~/.cache/tabpfn`.
- Tras 01, que solo quita `tfc-t0`, la frase queda como "`timesfm` ship an MLX runtime with other signatures" y sigue desfasada (TFM3-F9, 04 no la toca); reescribirla: "On macOS the result may differ from CI (Linux), for example in the device each backend selects. CI is the reference." [inferido].
- **Riesgo**: nulo. **Rompe compatibilidad**: no.

## 4. Tests

### 4.1 Tests unitarios

- `skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py`: los cuatro tests actuales (`test_adapters_toml_covers_every_registered_adapter`, `test_adapters_toml_entries_match_the_adapter_classes`, `test_adapters_toml_backend_package_is_a_PEP_508_requirement`, `test_workflow_matrix_matches_adapters_toml`) deben seguir pasando con la entrada `nori15`; añadir `test_contracts_cover_every_adapters_toml_entry` (3.6).
  ```bash
  pytest skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py -q
  ```
- No hay cambios en fakes ni en tests de adaptadores: los hacen 01 a 09.

### 4.2 Compatibilidad (`tools/foundation_compat`)

- Contrato y smoke de la entrada nueva y de la antigua en el entorno de Nori de la revisión (se recrea con `bash dev/foundation_adapters_review/evidence/common/make_env.sh nori "synthefy-nori" 3.12`, o con el requisito que fije D24; ver `evidence/README.md`):
  ```bash
  PY=~/.venvs/fm-review-nori/bin/python
  $PY -m pytest tools/foundation_compat/tests/test_contract.py tools/foundation_compat/tests/test_smoke.py --backend nori15 -q -p no:cacheprovider
  $PY -m pytest tools/foundation_compat/tests/test_contract.py tools/foundation_compat/tests/test_smoke.py --backend nori -q -p no:cacheprovider
  ```
- Flujo completo como en el CI: `python tools/foundation_compat/run.py --adapter nori15` y `python tools/foundation_compat/run.py --adapter nori`.
- Los smoke nuevos (cuantiles desordenados y, con D23 (a), NaN en la exog del contexto) en cada entrada: `python tools/foundation_compat/run.py --adapter <clave>` para `chronos`, `timesfm25`, `timesfm3`, `moirai`, `tabicl`, `t0`, `nori`, `nori15`, `tsicl` y `timestral`; `tabpfn` solo con `TABPFN_TOKEN` (sin él, los smoke se marcan como skipped) [verificado en código: README de `tools/foundation_compat`].
- Workflow: los PR a `0.27.x` no lo ejecutan [verificado en código]; lanzarlo a mano sobre la rama (Actions > Foundation Backends Compatibility > Run workflow), que usa el YAML de la rama, dos veces seguidas. En la primera debe haber un job por entrada, incluido `nori15`, todos en verde; en la segunda, el paso "Cache model weights" del job `tabpfn` debe restaurar del orden de 876 MB en lugar de 8332 B [inferido].

### 4.3 Comprobaciones de corrección de la re-ejecución (no son benchmarks)

Tras la re-ejecución, contrastar la celda 108 con las referencias de la sección 1.3:

- Chronos-2 (small) 171.2670 y TimesFM-3.0 162.4352 si chronos-forecasting y timesfm siguen en 2.3.2 y 3.0.2 y D6 no se resolvió con (b) (la revisión y la guía los obtuvieron con torch 2.15.0 y 2.9.1) [probado en la revisión].
- Nori cerca de 185.9507 (valor de la revisión con synthefy-nori 0.23.0, en Linux, con torch 2.13.0+cpu); una diferencia pequeña por la versión de torch del kernel o por el sistema operativo es posible [inferido]. Una diferencia grande indica un problema del entorno de la guía, no un cambio de precisión: parar y revisar el entorno.
- T0: el MAE depende de la versión de tfc-t0 del kernel y el cambio no es una regresión. 172.27 con tfc-t0 >= 0.6.1 (la que se instala con Python >= 3.11) y 147.51 con 0.6.0 (la que pip instala con Python 3.10); la guía mostraba 145.5262 con 0.3.2. Toda la diferencia entre 0.6.0 y 0.6.1 viene de `drop_constant_covariates`, que 0.6.1 añade a `T0Forecaster.predict` con `True` por defecto y que el adaptador no pasa (decisión de #1385): con `drop_constant_covariates=False` en 0.6.1 el MAE vuelve a 147.51, y solo con Temperature (181.33) o sin exógenas (185.56) las dos versiones coinciden. Del 1 al 23 de diciembre `Holiday` vale 0 en todo el contexto y en el horizonte, y aun así en 0.6.0 esa columna baja el MAE de 168.2 a 138.5: la ventaja de 0.6.0 viene casi entera de una columna sin información. Solo el 24 de diciembre (primer festivo en el horizonte con contexto constante) se pierde información real (205.6 con 0.6.0, 263.7 con 0.6.1); desde el 25 las dos versiones coinciden día a día. T0 cambia de posición en la tabla de la celda 108: decirlo en la descripción del PR para que no parezca una regresión [probado] (vic_electricity, t0-alpha, `context_length=500`, 30 folds de 24 pasos, CPU; Python 3.14.3 con tfc-t0 0.6.1 y torch 2.15.0, Python 3.10.20 con tfc-t0 0.6.0 y torch 2.14.1; macOS arm64).
- Moirai, ejecutado fuera de los requisitos de uni2ts (sección 1.4): comparar el backtest de la configuración de la guía (sin `--exog`, como la celda 70) en el entorno de la revisión, que respeta esos requisitos (torch 2.4.1+cpu, numpy 1.26.4; se recrea con `UV_OVERRIDE=dev/foundation_adapters_review/evidence/moirai/overrides.txt bash dev/foundation_adapters_review/evidence/common/make_env.sh moirai "uni2ts" 3.12`), con el del entorno de la guía. Los CSV van fuera del repositorio:
  ```bash
  S=dev/foundation_adapters_review/evidence/common/backtest_guide.py
  ~/.venvs/fm-review-moirai/bin/python $S --model-id Salesforce/moirai-2.0-R-small --out ~/moirai_reference.csv
  ~/.venvs/skforecast-fm-guide/bin/python $S --model-id Salesforce/moirai-2.0-R-small --out ~/moirai_guide_env.csv
  python -c 'import pandas as pd; a = pd.read_csv("~/moirai_reference.csv"); b = pd.read_csv("~/moirai_guide_env.csv"); print((a["pred"] - b["pred"]).abs().max())'
  ```
  Esperado: el MAE impreso por el segundo comando igual al de la celda 108 para Moirai, y una diferencia máxima de predicción del orden del redondeo de float32 frente al entorno de referencia [inferido]. Si no es así, no publicar la sección de Moirai con ese entorno: anotarlo en el PR y en el seguimiento de Moirai (D15).
- Las versiones exactas del kernel (Python, torch, numpy, pandas y cada backend) se copian del log del ejecutor (cabecera) a la descripción del PR.

### 4.4 Documentación

```bash
python tools/docs/execute_notebooks/execute_notebooks.py --check
python tools/docs/hooks/page_meta.py
python tools/ai/generate_ai_context_files.py --check
```

## 5. Documentación y contexto de IA

- **Docstrings**: ninguno.
- **`docs/user_guides/foundation-forecasting-models.ipynb`**: re-ejecución completa (3.1); celda 25 (lista de enlaces a la API, celda 25 en la revisión) completada con los cinco adaptadores que faltan; comprobación de las filas de la tabla HTML de la celda 11 (comandos de instalación) y de las celdas 14, 17, 107 y 109 (3.2). Los textos de cada sección de modelo no se tocan salvo incoherencia evidente con su PR.
- **`docs/user_guides/foundation-forecasting-with-heterogeneous-series.ipynb`**: comprobación de las celdas 2 ("In short") y 21 (tabla de capacidades por adaptador y notas) y re-ejecución (3.3).
- **`skills/foundation-forecasting/SKILL.md`, `skills/foundation-forecasting/references/adapter-parameters.md` y `tools/ai/llms-base.txt`**: sin cambios previstos; ninguno cita los MAE de la guía [verificado en código]. Si el repaso encuentra una frase que contradice los atributos finales y que ningún PR de adaptador cambió, corregirla aquí y regenerar con la skill `ai-context-sync` (`python tools/ai/generate_ai_context_files.py`).
- **`tools/foundation_compat/README.md`**: 3.7.
- **Release note** (`docs/releases/releases.md`, sección 0.27.0, **Changed**, junto a las entradas de documentación). Los cambios del CI y de `tools/` no llevan entrada según la skill `release-note` [verificado en código]; la documentación solo lleva entrada si un lector lo nota, y la tabla comparativa regenerada lo es [inferido]. Si 08 ya dejó una entrada sobre la guía de TabPFN-TS, fusionarlas. Texto propuesto:
  ```markdown
  + The outputs of the foundation models user guide, including the comparison table of the backtesting metrics, are regenerated with the latest version of every backend, and TabPFN-TS now runs with the TabPFN-3.5 weights. [User guide](../user_guides/foundation-forecasting-models.ipynb#model-comparison) ([#NNNN](https://github.com/skforecast/skforecast/pull/NNNN))
  ```

## 6. Decisiones pendientes

| Id | Decisión | Opciones | Recomendación | Impacto en el plan | Estado |
|---|---|---|---|---|---|
| D26 | Cobertura CI de Nori-1-5 (NORI-P7) | (a) segunda entrada; (b) sustituir la actual; (c) nada | (a) | Con (a), sección 3.6 tal cual. Con (b), cambiar `model_id` de `[nori]` a `Synthefy/Nori-1-5` sin tocar matriz ni contratos (y sin el riesgo de "no result"). Con (c), se omite 3.6 salvo los dos smoke de `test_smoke.py`, que no dependen de D26 | pendiente |
| D24 | Mínimo de Nori (NORI-P3) | (a) `synthefy-nori>=0.22` con comillas y nota de que solo cubre instalaciones nuevas; (b) solo documentar; (c) comprobación en `_load_model` | (a); (c) si se adopta D-VER | Fija el `backend_package` de las dos entradas de Nori en `adapters.toml` y el comando de la fila de Nori en la celda 11. Lo decide 09 | pendiente |
| D23 | NaN en covariables de Nori (NORI-P2) | (a) aplicar; (b) solo corregir el docstring | (a) | Con (a), 3.6 añade el smoke con NaN en la exog del contexto; con (b), no se añade. Lo decide 09 | pendiente |
| D19 | Mínimo de TabPFN-TS (TPF-P8) | (A) mantener `>=1.3` y documentar límites; (B) `>=1.4` (arrastra tabpfn>=9.2.0) | B: con 1.3 hay KeyError con varias GPU (TPF-V1) y tabpfn 9.0 sin #1310/#1215 (TPF-V2) | Fija el comando de instalación de TabPFN-TS en la celda 11 y el requisito del entorno de la guía (en la revisión la última es 1.4.0 en ambos casos). Lo decide 08 | pendiente |
| D6 | `padding_mode` de TimesFM 3 (TFM3-P2) | (a) mantener "edge" y corregir el docstring; (b) "none" como upstream (cambia predicciones); (c) configurable por `predict_kwargs` | (a) + (c) | Con (b), el MAE de TimesFM-3.0 de la celda 108 deja de ser 162.4352 y la referencia de 4.3 no vale. Lo decide 04 | pendiente |
| D15 | Mínimo y mensaje de Moirai (MOI-P1) | (a) solo mensaje; (b) mensaje + `uni2ts>=2.0`; (c) (b) + documentar Python<=3.12 y conflicto con pip | (c), con comillas | Fija el requisito de uni2ts del entorno de la guía; es el seguimiento donde anotar un fallo de Moirai en el entorno común (4.3). Lo decide 06 | pendiente |

## 7. Criterios de aceptación

- [ ] 01 a 09 fusionados en `0.27.x` y el último run programado del CI de compatibilidad en verde sobre `0.27.x` (sección 1.5).
- [ ] `pytest skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py -q`: todos pasan, incluido `test_contracts_cover_every_adapters_toml_entry`.
- [ ] `python tools/foundation_compat/run.py --adapter nori15`: código de salida 0, contrato y smoke `passed`, `held_back` false.
- [ ] `python tools/foundation_compat/run.py --adapter nori`: sigue en `passed` con el smoke nuevo de cuantiles desordenados.
- [ ] Workflow lanzado a mano sobre la rama: un job por entrada (incluido `nori15`), todos en verde; en la segunda ejecución, la caché del job `tabpfn` restaura del orden de 876 MB en lugar de 8332 B.
- [ ] `execute_notebooks.py user_guides/foundation-forecasting-models.ipynb`: código de salida 0, sin errores en el log; la cabecera (`Kernel skf`) muestra que el kernel importa skforecast del checkout.
- [ ] Celda 78 imprime `True`; el LicenseWarning de la celda 81 dice "TabPFN-3.5 License v1.0 (non-commercial without an enterprise license)" y enlaza `Prior-Labs/tabpfn_3_5`; ninguna "Location" de un warning apunta a un `site-packages`.
- [ ] Celda 108 con diez filas numéricas; Chronos-2 (small) 171.2670 y TimesFM-3.0 162.4352 si sus backends no cambiaron de versión (y D6 no es (b)); Nori cerca de 185.9507; T0 172.27 con tfc-t0 >= 0.6.1 (147.51 con 0.6.0; no es una regresión, ver 4.3); Moirai coherente con el backtest del entorno `fm-review-moirai` (4.3).
- [ ] Celda 13 (`list_adapters`) refleja los `backend_package` y atributos finales de 01 a 09.
- [ ] Celda 25 enlaza los diez adaptadores; los comandos de instalación de la celda 11 coinciden con los `backend_package` finales, bien entrecomillados en `<code>`, `data-copy` y `aria-label`.
- [ ] `execute_notebooks.py user_guides/foundation-forecasting-with-heterogeneous-series.ipynb`: código de salida 0; celdas 2 y 21 coherentes con los atributos finales.
- [ ] `python tools/docs/execute_notebooks/execute_notebooks.py --check`: código de salida 0.
- [ ] `python tools/docs/hooks/page_meta.py` y `python tools/ai/generate_ai_context_files.py --check` sin errores.
- [ ] Entrada en `docs/releases/releases.md` (sección 0.27.0, Changed).
- [ ] Skill `verify` en verde antes del commit.
- [ ] Descripción del PR con las versiones de Python, torch, numpy, pandas y de cada backend usadas en la re-ejecución, los overrides aplicados, y la nota sobre el run programado entre la fusión y la release (3.6).

## 8. Fuera de alcance

- Benchmarks de velocidad o de precisión entre versiones de un backend o entre variantes de un modelo: fuera del alcance de skforecast. La tabla de la celda 108 se regenera; no se analiza la diferencia con la anterior.
- Re-ejecutar la guía en GPU o medir tiempos.
- Cambios de código en adaptadores o en `FoundationModel`: van en 01 a 09. Si la re-ejecución revela un fallo, se abre una issue y se arregla en un PR del adaptador.
- Invariantes del smoke propias de cada adaptador (TSI-P8 en 03, T25-P6 en 04) y los fakes (por ejemplo NORI-P8 en 09): en sus planes. Las demás invariantes genéricas del smoke (XC-7 de `evidence/cross_cutting.md`) quedan fuera de esta serie de PR (`10_docs_ci.md`, sección 8); si se quieren, van en un PR aparte `chore/foundation-compat-checks` (PR-12 de `evidence/cross_cutting.md`), después de 01 y 03: como anota `evidence/cross_cutting.md` para su PR-12, T0 y TS-ICL las fallarían antes de 01 y 03. Este plan solo añade el smoke de cuantiles desordenados y, con D23 (a), el de NaN en la exog del contexto (3.6).
- Un carril del CI con la versión mínima de cada backend (sub-decisión de la política de mínimos, D-VER en el informe y D6 en `evidence/cross_cutting.md`, donde se recomienda hacerlo solo para el nivel de contrato) y una clave `python` opcional por entrada en `adapters.toml` (XC-12 de `evidence/cross_cutting.md`): no se hacen aquí.
- Una matriz del workflow generada desde `adapters.toml` (evitaría el desfase entre la matriz de `main` y la rama comprobada descrito en 3.6): no se hace aquí.
- Re-ejecutar `docs/user_guides/comparing-forecasting-models.ipynb` (3.4).
- Revisar `TimestralAdapter`: solo se re-ejecuta su sección de la guía.
- Checkpoints nuevos aparecidos después de la revisión (1.5, paso 4).
- Decisiones pospuestas de otros planes que afectarían a la guía si se tomaran de otra forma: D10 (XReg de TimesFM 2.5), D16 (covariables en Moirai), D18 (categóricas en TabPFN), D25 (categóricas nativas de Nori), D28 (checkpoint por defecto de Nori).
