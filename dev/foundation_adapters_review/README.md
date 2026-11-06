# Planes de implementación: alineación de los adaptadores foundation con sus backends

Índice de los planes de implementación que salen de la revisión del 2026-11-06 de los adaptadores de `skforecast.foundation`. Cada plan corresponde a un pull request futuro y es autocontenido: para implementarlo basta con abrir ese fichero y la evidencia que enlaza.

Cómo recrear los entornos de la revisión y volver a ejecutar sus comprobaciones está en [evidence/README.md](evidence/README.md). Ese fichero no se copia aquí.

## 1. Propósito y alcance

- **Objetivo**: implementar, en un PR por adaptador más un PR de cambios comunes y otro final de guía y CI, la alineación de los adaptadores foundation con la última versión de sus backends y de sus checkpoints que encontró la revisión del 2026-11-06 [documentado] (`evidence/diagnosis_report.md`, sección 1).
- **Base de la revisión**: `origin/0.27.x`, commit `d9b8f4f` (skforecast 0.27.0 en desarrollo) [verificado en código]. Las líneas que citan los planes son de ese commit y se dan como "aprox. línea N en d9b8f4f". `origin/0.27.x` ha avanzado después: por ejemplo, el commit `2282824` cambia `_to_covariate_array` en cada adaptador (conversión a `float32` sin `Series` intermedia) [verificado en código]. Hay que localizar el código por nombre de clase y método, no por número de línea.
- **Adaptadores en el alcance (9)**: `T0Adapter`, `TSICLAdapter`, `TimesFM25Adapter`, `TimesFM3Adapter`, `ChronosAdapter`, `MoiraiAdapter`, `TabPFNAdapter`, `TabICLAdapter` y `NoriAdapter` [documentado].
- **Excluidos**: Timestral y TimeH2O, a petición del mantenedor [documentado]. `TimestralAdapter` solo aparece en los planes de forma indirecta: recibe el atributo de capacidad nuevo de `03_tsicl.md` y su sección de la guía se re-ejecuta en `10_docs_ci.md` [verificado en código].
- **Fuera de alcance**: medir si un modelo o una versión del backend es más rápido, más lento o más preciso (tiempos, barridos de `batch_size`, dtype, `torch.compile`, comparaciones de precisión entre versiones o variantes) [documentado]. Sí entran las comprobaciones de corrección: la salida del adaptador debe coincidir con una llamada directa al backend sobre la misma entrada, y el backtest de la guía (`evidence/common/backtest_guide.py`) se usa solo como comprobación antes y después de un cambio que pueda alterar predicciones.

## 2. Resumen por adaptador

Valores de la sección 2 de `evidence/diagnosis_report.md`, ya corregidos por la verificación.

| Adaptador | Versión probada antes | Última versión | Estado | Prioridad | Plan |
|---|---|---|---|---|---|
| `T0Adapter` | tfc-t0 0.3.2, la última que funciona según la issue #1385; el CI del 2026-11-06 instaló 0.6.0 y falló [probado] | tfc-t0 0.6.0 (2026-10-13) [documentado] | roto [probado] | alta | [01_t0.md](01_t0.md) |
| `TSICLAdapter` | tsicl 0.2.1 (CI: contrato 3, smoke 6) [probado] | tsicl 0.2.1 (2026-06-25); `main` con el arreglo del rollout sin publicar [verificado en código] | desfasado [probado] | alta | [03_tsicl.md](03_tsicl.md) |
| `TimesFM25Adapter` | timesfm 3.0.2 (CI: contrato 6, smoke 5 más 1 skipped) [probado] | timesfm 3.0.2 (2026-09-09) [documentado] | alineado [probado] | media | [04_timesfm.md](04_timesfm.md) |
| `TimesFM3Adapter` | timesfm 3.0.2 (CI: contrato 3, smoke 6) [probado] | timesfm 3.0.2 (2026-09-09) [documentado] | alineado [probado] | media | [04_timesfm.md](04_timesfm.md) |
| `ChronosAdapter` | chronos-forecasting 2.3.2 (CI: contrato 3, smoke 6) [probado] | chronos-forecasting 2.3.2 (2026-09-08) [documentado] | alineado [probado] | media | [05_chronos.md](05_chronos.md) |
| `MoiraiAdapter` | uni2ts 2.0.0 (CI: contrato y smoke en verde) [probado] | uni2ts 2.0.0 (2025-11-04) [documentado] | alineado [probado] | media | [06_moirai.md](06_moirai.md) |
| `TabICLAdapter` | tabicl 2.2.0 (CI: contrato 3, smoke 6) [probado] | tabicl 2.2.0 (2026-09-02); 2 commits en `main` sin publicar [verificado en código] | alineado [probado] | media | [07_tabicl.md](07_tabicl.md) |
| `TabPFNAdapter` | tabpfn-time-series 1.4.0 con tabpfn 9.2.0 (CI: contrato 5, smoke 6) [probado] | tabpfn-time-series 1.4.0 (2026-10-28) con tabpfn 9.2.0 [documentado] | alineado [probado] | media | [08_tabpfn.md](08_tabpfn.md) |
| `NoriAdapter` | synthefy-nori 0.23.0 (CI: contrato 4, smoke 6) [probado] | synthefy-nori 0.23.0 (2026-11-04) [documentado] | alineado [probado] | media | [09_nori.md](09_nori.md) |

"Alineado" significa que el adaptador funciona con la última versión del backend; no quiere decir que no haya cambios pendientes (mínimos de versión sin declarar, textos desfasados, casos límite). Los cambios comunes a varios adaptadores están en [02_common.md](02_common.md) y la re-ejecución de la guía y el CI de compatibilidad en [10_docs_ci.md](10_docs_ci.md).

## 3. Orden de los PR

Todas las ramas se crean desde `origin/0.27.x` y todos los PR se abren contra `0.27.x`, nunca contra `main` [documentado] (`CLAUDE.md`, sección "Git workflow").

| Nº | Fichero | Rama propuesta | Prioridad | Depende de | Contenido |
|---|---|---|---|---|---|
| 01 | [01_t0.md](01_t0.md) | `fix/t0-tfc-t0-0.6` | alta y urgente | nada | T0 con `tfc-t0[torch]>=0.6`: API nueva (`model_input`, `quantile_levels`), extra torch y mínimo con ImportError claro, relleno como PAD (`trim_leading_missing=True`), `supports_heterogeneous_covariates=False`, `torch_dtype`, fakes, contrato y docs; cierra skforecast/skforecast#1385. |
| 02 | [02_common.md](02_common.md) | `fix/foundation-common-guards` (opcionalmente dividido con `chore/foundation-backend-version-helper`) | alta (X-1); media (X-2, X-3, X-4); baja (X-6, X-8) | nada | Código común en `FoundationModel`, el registro y `_utils.py`: error con contexto todo NaN, tabla central de ids no soportados, niveles de cuantil a `float`, helper opcional de comprobación de versión, `stacklevel` de los avisos y matiz de la garantía de independencia entre series. |
| 03 | [03_tsicl.md](03_tsicl.md) | `fix/tsicl-alignment` | alta | 02 (parcial) | TS-ICL: atributo de capacidad nuevo y partición central con exog, `supports_past_only_covariates=False`, niveles de cuantil vía `Decimal`, guarda de horizonte >672 con exog futura, docs de Python>=3.12 y torch<2.10, tests y smoke. |
| 04 | [04_timesfm.md](04_timesfm.md) | `fix/timesfm-alignment` | media | 02 (suave) | TimesFM 2.5 y 3: helper de import de `timesfm` (TV-1), `timesfm>=3.0` para 2.5 y retirada de `_TimesFMCompat`, "mediana" en lugar de "media", `padding_mode`, kwargs reservados y docs. |
| 05 | [05_chronos.md](05_chronos.md) | `fix/chronos-alignment` | media | 02 (CHR-P1 y, con D-VER (b), el helper de versión) | Chronos-2: mínimo `chronos-forecasting>=2.2`, contrato de `cross_learning`, `pd.Categorical` numérica enviada como categórica, `device_map` y textos. |
| 06 | [06_moirai.md](06_moirai.md) | `fix/moirai-alignment` | media | 02 | Moirai-2: `uni2ts>=2.0` y mensaje claro en Python>=3.13, no recargar pesos en `set_params`, guarda de niveles de cuantil, `stacklevel` del aviso MPS y docs. |
| 07 | [07_tabicl.md](07_tabicl.md) | `fix/tabicl-alignment` | media | 02 (D-Q y D-VER) | TabICL: crea los helpers comunes (reemitir warnings ocultos por `redirect_stderr`; timestamps sin `pd.Timedelta(days=n)`), mínimo de tabicl, niveles de cuantil y docs. |
| 08 | [08_tabpfn.md](08_tabpfn.md) | `fix/tabpfn-alignment` | media | 07; 02 solo con D19 (B) y D-VER | TabPFN-TS: aplica los helpers de 07, docstring de `temporal_features`, opciones de `TabPFNTSPipeline` (D17), categóricas (D18), mínimo (D19) y fakes. |
| 09 | [09_nori.md](09_nori.md) | `fix/nori-adapter-alignment` | media | 02 (parcial) | Nori: conservar las filas de contexto con covariables NaN, `synthefy-nori>=0.22`, quitar workarounds redundantes, docs con Nori-1-5 y fake. |
| 10 | [10_docs_ci.md](10_docs_ci.md) | `docs/foundation-guide-refresh` | media (guía); baja (CI) | 01 a 09 fusionados | Re-ejecución única de la guía de usuario (incluidas las celdas de TabPFN-TS, con `TABPFN_TOKEN`), repaso de la guía de series heterogéneas, caché de `~/.cache/tabpfn`, entrada de Nori-1-5 y otros cambios del CI de compatibilidad. |

**Dependencias** [inferido salvo donde se indica]:

- **01 es independiente y urgente**: `T0Adapter` está roto con la instalación por defecto (`pip install tfc-t0` instala 0.6.0 sin torch) [probado]. No espera a 02. Si 01 se fusiona antes que 02, la entrada de los ids ONNX de T0 en la tabla central de ids no soportados y, si se adopta D-VER (b), la migración de la comprobación de versión de T0 al helper común se hacen en 02 sobre el código ya fusionado de 01.
- **02 va antes de los PR de adaptador que lo citan** (03 a 09): allí viven la comprobación de contexto todo NaN (D11), la tabla de ids no soportados (D-REG), la conversión de cuantiles (D-Q) y el helper de versión (D-VER). Cada plan de adaptador describe qué hacer si 02 aún no está fusionado o si se eligió otra opción.
- **07 va antes de 08**: 07 crea en `skforecast/foundation/_utils.py` los helpers que 08 aplica a `TabPFNAdapter`.
- **03 antes de 04 a 09 en el orden de fusión**: su atributo de capacidad nuevo entra en `_AdapterBase._REQUIRED_CLASS_ATTRIBUTES` y obliga a declararlo en el cuerpo de clase de todos los adaptadores [verificado en código]; los PR siguientes deben rebasar sobre él.
- **03 a 09 en serie, no en paralelo**: tocan zonas cercanas de `skforecast/foundation/_adapters.py`, `fixtures_adapters.py`, `tools/foundation_compat/adapters.toml`, `tools/ai/llms-base.txt` y `skills/foundation-forecasting/`.
- **10 va el último**, cuando 01 a 09 están fusionados en `0.27.x`, y conviene fusionarlo poco antes del PR de release a `main`.
- **`stacklevel` de los avisos**: el arreglo compartido (`_warn_if_non_commercial`, `_resolve_torch_device` y el traslado de `_stacklevel_user_code` a `skforecast/utils/utils.py`) está solo en `02_common.md`, sección 3.5; `06_moirai.md` se limita a quitar el `stacklevel=7` propio del aviso MPS de Moirai.

**Regla para la guía de usuario**: cada PR de adaptador (01 y 03 a 09) edita solo el texto de su propio adaptador en `docs/user_guides/foundation-forecasting-models.ipynb` (su fila de la tabla de modelos, su sección y, si toca, su fila de la guía de series heterogéneas) y **no re-ejecuta** el cuaderno. 10 lo re-ejecuta una sola vez. Motivos: una re-ejecución cambia salidas y `execution_count` de todas las celdas de código, así que varios PR que re-ejecutan chocarían en el JSON [inferido]; la tabla comparativa final lee las variables `metrics_*` de todas las secciones y exige un único kernel con todos los backends y el `TABPFN_TOKEN` [verificado en código].

## 4. Decisiones pendientes

Ninguna decisión está tomada: todas están **pendientes** del mantenedor. Los ids son los de la sección 6 de `evidence/diagnosis_report.md` (no existen D4 ni D22 en esa numeración). `evidence/cross_cutting.md` usa otra numeración propia (XC-n para los hallazgos y D1 a D14 para sus decisiones) [verificado en código]; los planes usan siempre los ids del informe (X-n, TV-n, Dn, D-REG, D-VER, D-Q).

| Id | Decisión | Plan donde se describe | Recomendación | Estado |
|---|---|---|---|---|
| D1 | Mínimo de T0 (T0-P1) | 01 | (a) `tfc-t0[torch]>=0.6` | pendiente |
| D2 | Relleno de lote en T0 (T0-P2) | 01 | (a) `trim_leading_missing=True` | pendiente |
| D3 | Covariables heterogéneas en T0 (T0-P3) | 01 | (a) `supports_heterogeneous_covariates=False`; complementaria de D2 | pendiente |
| D5 | Opciones de inferencia de T0 (T0-P8) | 01 | (a) posponer | pendiente |
| D6 | `padding_mode` de TimesFM 3 (TFM3-P2) | 04 | (a) mantener "edge" y corregir el docstring, más (c) configurable por `predict_kwargs` | pendiente |
| D7 | Opciones de carga de TimesFM 3 (TFM3-P4) | 04 | (a) nada por ahora; decidir (b) `model_kwargs` de forma común | pendiente |
| D8 | Mínimo de TimesFM 2.5 (T25-P1) | 04 | A: `>=3.0` con ImportError y sin `_TimesFMCompat` | pendiente |
| D9 | Prefijo de TimesFM 2.5 (T25-P3) | 02 y 04 | B, como entrada de la tabla central de D-REG (a) (02, sección 3.2); A (prefijo exacto `google/timesfm-2.5-200m-pytorch`) solo si D-REG se resuelve con (c) | pendiente |
| D10 | XReg de TimesFM 2.5 (T25-P7) | 04 | A: no implementar y matizar docs | pendiente |
| D11 | Contexto todo NaN (CHR-P1, MOI-P5, TV-2) | 02 (citada en 05, 06 y 09) | (a) ValueError que nombre las series | pendiente |
| D12 | Mínimo de Chronos (CHR-P2) | 05 | (a) `>=2.2`; (b) `>=2.3.2` si se quiere garantizar transformers 5 | pendiente |
| D13 | `pd.Categorical` numérica en Chronos (CHR-P4) | 05 | (a) enviarla como categórica | pendiente |
| D14 | `device_map="auto"` en Chronos (CHR-P6) | 05 | (a) documentar el comportamiento de accelerate | pendiente |
| D15 | Mínimo y mensaje de Moirai (MOI-P1) | 06 | (c) mensaje, `uni2ts>=2.0`, Python<=3.12 y conflicto con pip, con comillas | pendiente |
| D16 | Covariables en Moirai (MOI-P7) | 06 | (a) mantener sin exog | pendiente |
| D17 | Opciones de `TabPFNTSPipeline` (TPF-P2) | 08 | A: `pipeline_kwargs` reenviado solo si no es None | pendiente |
| D18 | Categóricas en TabPFN (TPF-P6) | 08 | A: mantener False hasta probar con TabPFN-3.5 y token | pendiente |
| D19 | Mínimo de TabPFN-TS (TPF-P8) | 08 | B: `>=1.4` (arrastra tabpfn>=9.2.0) | pendiente |
| D20 | Mínimo de TabICL (TAB-P3) | 07 | (b) `>=2.2.0`, con comprobación en ejecución si se adopta D-VER | pendiente |
| D21 | Papel de `model_id` en TabICL (TAB-P4) | 07 | (a) documentar | pendiente |
| D23 | NaN en covariables de Nori (NORI-P2) | 09 | (a) aplicar | pendiente |
| D24 | Mínimo de Nori (NORI-P3) | 09 | (a) `synthefy-nori>=0.22` con comillas; (c) si se adopta D-VER | pendiente |
| D25 | Categóricas nativas de Nori (NORI-P6) | 09 | (a) adoptar DataFrame con dtypes, con baja prioridad | pendiente |
| D26 | Cobertura CI de Nori-1-5 (NORI-P7) | 09 (se implementa en 10) | (a) segunda entrada | pendiente |
| D27 | Featurización de Nori (NORI-P9) | 09 | (b) mantener la propia | pendiente |
| D28 | Checkpoint por defecto de Nori (NORI-P10) | 09 | (a) mantener `Synthefy/Nori` 6M | pendiente |
| D29 | Agrupación de TS-ICL con exog (TSI-P1) | 03 | (b) una serie por llamada, revisable cuando upstream cierre #8 y #10 | pendiente |
| D30 | Covariables solo de pasado en TS-ICL (TSI-P2) | 03 | (a) False hasta que se publiquen #4 y #6 | pendiente |
| D31 | Niveles de cuantil de TS-ICL (TSI-P3) | 03 | (d) traducir a un `Decimal` exacto de la rejilla (`Decimal(round(q * 100)) / 100`, ver 03) más (c) reportar upstream | pendiente |
| D32 | Horizonte de TS-ICL (TSI-P4) | 03 | (a) guarda y docs ahora; después (b) subir el mínimo | pendiente |
| D33 | `context_length>4096` en TS-ICL (TSI-P6) | 03 | docstring más aviso | pendiente |
| D-REG | Ids aceptados por prefijo que no cargan (X-3: NORI-P1, T0-P5, T25-P3) | 02 (citada en 01, 04 y 09) | (a) tabla central de ids no soportados en `_resolve_adapter`; (c) solo para TimesFM 2.5 si se prefiere | pendiente |
| D-VER | Política de versión mínima (X-4) | 02 (citada en 03 a 09) | (b) mínimo en `backend_package` más un helper común pequeño de comprobación en ejecución | pendiente |
| D-Q | Tipo de los niveles de cuantil (X-2, TAB-P2) | 02 (citada en 03 y 07) | (a) convertir a `float` en `FoundationModel` | pendiente |

## 5. Cómo usar un plan

1. **Antes de empezar**: seguir el paso "Antes de empezar" de la sección 1 del plan. Los planes describen el estado del 2026-11-06 y pueden quedar desfasados: comprobar en PyPI y en el Hub si hay versiones nuevas, por ejemplo:

   ```bash
   python dev/foundation_adapters_review/evidence/common/pypi_info.py tfc-t0
   python dev/foundation_adapters_review/evidence/common/hub_info.py theforecastingcompany --prefix t0
   python dev/foundation_adapters_review/evidence/common/hub_info.py --repo google/timesfm-3.0-pytorch
   ```

   Si apareció una versión o un checkpoint nuevo, actuar como indique el plan (por lo general: recrear el entorno, repetir las comprobaciones de la sección 4 del plan y ajustar mínimos y textos antes de implementar). Recrear los entornos de la revisión con [evidence/README.md](evidence/README.md).
2. **Rama**: crear la rama propuesta desde `origin/0.27.x` (la rama de release `X.Y.x` se deriva de `__version__` en `skforecast/__init__.py`: 0.27.0 da `0.27.x`) y rebasar sobre los PR de la serie ya fusionados.
3. **Implementar** la sección 3 del plan, respetando `.claude/rules/foundation.md`: los adaptadores son finos y el tratamiento genérico va a `FoundationModel`. Las decisiones de la sección 6 del plan deben estar tomadas antes de implementar las propuestas que dependen de ellas.
4. **Tests**: ejecutar los tests de la sección 4 del plan (solo los ficheros tocados, sin `-n` y sin lanzar la suite completa) y las comprobaciones de evidencia indicadas, con el intérprete del entorno de la revisión del adaptador (`~/.venvs/fm-review-<key>/bin/python`).
5. **Criterios de aceptación**: repasar la sección 7 del plan (por ejemplo `python tools/foundation_compat/run.py --adapter <key>` y los valores esperados de los scripts de evidencia).
6. **Documentación y contexto de IA**: aplicar la sección 5 del plan; si cambian `tools/ai/llms-base.txt` o `skills/`, regenerar los ficheros generados con la skill `ai-context-sync` (nunca editar a mano `AGENTS.md`, `llms-full.txt` ni `docs/llms*.txt`).
7. **Release note**: añadir la entrada propuesta en la sección 5 del plan a `docs/releases/releases.md`, en la sección de la versión en desarrollo (skill `release-note`).
8. **Verificación**: ejecutar la skill `verify` antes de dar el cambio por terminado y antes del commit.
9. **PR**: abrirlo contra la rama de release `0.27.x` (nunca contra `main`), con `Closes #NNNN` si el plan cierra una issue (por ejemplo `Closes #1385` en 01).

## 6. Estructura de la carpeta

```
dev/foundation_adapters_review/
├── README.md               este índice
├── 01_t0.md                T0 (tfc-t0), urgente
├── 02_common.md            cambios comunes en FoundationModel, el registro y _utils.py
├── 03_tsicl.md             TS-ICL (tsicl)
├── 04_timesfm.md           TimesFM 2.5 y TimesFM 3 (timesfm)
├── 05_chronos.md           Chronos-2 (chronos-forecasting)
├── 06_moirai.md            Moirai-2 (uni2ts)
├── 07_tabicl.md            TabICL (tabicl), crea los helpers que usa 08
├── 08_tabpfn.md            TabPFN-TS (tabpfn-time-series)
├── 09_nori.md              Nori (synthefy-nori)
├── 10_docs_ci.md           re-ejecución de la guía y CI de compatibilidad
└── evidence/
    ├── README.md           entornos de la revisión y cómo reproducir las comprobaciones
    ├── diagnosis_report.md diagnóstico completo y verificado (hallazgos, propuestas, decisiones)
    ├── cross_cutting.md    revisión transversal del código común
    ├── cross_cutting/      sondas del código común (probe_shared.py y su log)
    ├── common/             make_env.sh, pypi_info.py, hub_info.py, backtest_guide.py
    └── <key>/              una carpeta por adaptador: scripts, logs, requirements-freeze.txt
        └── verification/   segunda pasada independiente: verification.md, scripts y logs
```

Claves de adaptador (`<key>`, iguales a las entradas de `tools/foundation_compat/adapters.toml`): `t0`, `tsicl`, `timesfm25`, `timesfm3`, `chronos`, `moirai`, `tabpfn`, `tabicl`, `nori` [verificado en código].

Fuentes, por orden de autoridad: `evidence/<key>/verification/verification.md` (confirma, corrige o refuta cada hallazgo y propuesta) y `evidence/diagnosis_report.md` (solo da como hechos los hallazgos confirmados o corregidos, en su forma corregida); `evidence/cross_cutting.md` es la base de los cambios comunes de 02. Los planes solo recogen hallazgos confirmados o corregidos; los refutados aparecen en la sub-lista "Descartado" de cada plan.

## 7. Leyenda de etiquetas y límites de la revisión

Cada afirmación de los planes lleva una etiqueta:

- **[probado]**: ejecutado en la revisión o en la verificación.
- **[verificado en código]**: leído en el código del backend o de skforecast (se cita fichero y tag o commit).
- **[documentado]**: lo dice una fuente escrita (CHANGELOG, README, model card, metadatos de PyPI, issue), sin comprobarlo.
- **[inferido]**: deducción sin prueba directa.

Los identificadores de los hallazgos y propuestas se conservan del informe: `<ADAPTADOR>-Fn` (hallazgos), `<ADAPTADOR>-Pn` (propuestas), `<ADAPTADOR>-Vn` (hallazgos o propuestas añadidos por la verificación), `X-n` (transversales), `TV-n` (transversales de TimesFM), `Dn`, `D-REG`, `D-VER` y `D-Q` (decisiones).

Límites de la revisión [documentado] (`evidence/diagnosis_report.md`, sección 1, "Lo que no se ha podido comprobar"):

- arXiv y los blogs de las empresas estaban bloqueados por la red: no se usaron papers ni anuncios (afecta, por ejemplo, al contexto de entrenamiento de Moirai-2 y a si TS-ICL se entrenó con covariables solo de pasado).
- La API REST de GitHub solo estaba disponible para skforecast/skforecast: el código upstream se leyó con `git clone`, tags, `git log`, CHANGELOG y README, y las issues upstream con la búsqueda de GitHub (título, estado y cuerpo, sin comentarios).
- No hay GPU: el comportamiento en CUDA, MPS, XPU y MLX solo se ha leído en código.
- No hay `TABPFN_TOKEN`: los pesos de TabPFN-3.5 no se descargan, así que el smoke real de TabPFN no se ejecutó en la revisión (sí en el CI). La alineación de TabPFN-TS se comprobó a nivel de features y con el checkpoint público `Prior-Labs/TabPFN-v2-reg`.
- Ningún checkpoint del alcance está gated en el Hub; TabPFN-3.5 exige aceptar la licencia de Prior Labs mediante token propio.
- El código sin publicar de `main` upstream de tsicl (arreglo del rollout, PR #1) no se ejecutó.
- `tools/foundation_compat/run.py` solo se ejecutó para Chronos; para el resto, el CI del mismo día (run 42212742430 del workflow Foundation Backends Compatibility, rama 0.27.x) lo ejecutó con las mismas versiones, y en la revisión se corrieron contrato y smoke directamente.
- Todas las pruebas se hicieron en Linux x86_64 con 4 CPU, sin GPU, Python 3.12.3 y torch CPU (versiones exactas por entorno en `evidence/README.md` y en cada `evidence/<key>/requirements-freeze.txt`) [probado].
