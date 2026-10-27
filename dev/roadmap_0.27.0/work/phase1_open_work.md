# Fase 1: inventario del trabajo abierto (skforecast 0.27.0)

Fecha: 2026-10-17. Base: `origin/0.26.x` (`__version__ = "0.26.0"`, publicada el 2026-10-08).
Fuentes: `git fetch origin` y ramas remotas con commits posteriores al 2026-07-17; GitHub
`skforecast/skforecast` (MCP de GitHub y REST con `gh api`). GraphQL no está disponible en la
sesión; las discusiones se leyeron por REST (`/repos/.../discussions`), que sí funciona.

## 1. Ramas remotas activas (últimos 3 meses)

| Rama | Último commit | Qué hace frente a `origin/0.26.x` | Estado | ¿0.27.0? |
|:--|:--|:--|:--|:--|
| `perf/ets-numba-compile` (PR [#1400](https://github.com/skforecast/skforecast/pull/1400)) | 2026-10-14 | Reescribe `_ets_nelder_mead` sin expresiones de arrays ni `argsort` (nuevo `_sort_simplex` con inserción estable) en `skforecast/stats/exponential_smoothing/_ets_base.py`. La primera compilación de numba de `Ets` baja de 12 s a 7 s (4 núcleos), mismas estimaciones. 4 commits, 2 ficheros (+102/-39). | Terminada, PR abierto, nota de versión escrita. | Sí. Encaja en "stats: solo rendimiento". Conflicto trivial en `releases.md` con #1402 (las dos crean la sección 0.27.0). |
| `fix/fetch-dataset-website-visits-m5` (PR [#1402](https://github.com/skforecast/skforecast/pull/1402)) | 2026-10-14 | `skforecast/datasets/datasets.py`: corrige `date_format` de `website_visits` (`%d/%m/%y`) e `index_col` de `m5` (`datetime`); libera el pool de pyarrow y usa `df.pop` en vez de `set_index` (m5 pasa de más de 15 GB a unos 10 GB). Tests nuevos en `test_fetch_dataset.py`. | Terminada, PR abierto. | Sí (bug evidente: los dos datasets se devolvían sin `DatetimeIndex` y con aviso `Could not preprocess dataset`). |
| `feature_synthetic_data_generator` | 2026-08-19 | Generador sintético de paneles horarios de gasolineras: `skforecast/experimental/_synthetic_data.py` (1053 líneas: `StationConfig`, `SeasonalEffectsConfig`, `generate_station`, `generate_gas_station_panel`, `summarize`), tests (291 líneas), `docs/api/experimental.md`, metodología en `dev/SYNTHETIC_DATA_METHODOLOGY.md` y notebook de 9949 líneas. Basada en una base antigua: 288 commits por detrás de `0.26.x`; el diff con `...` arrastra 142 ficheros por la base vieja, el trabajo propio son 8 commits sobre `1b4930665`. `experimental/__init__.py` no lo exporta. | WIP (último commit "ongoing"), sin PR. | Opcional. No es foco (datasets/utils), pero útil para docs y benchmarks. Requiere rebase sobre `0.26.x` y decidir si va en `experimental` o en `datasets`. |
| `ci/minimum-versions-job` | 2026-10-07 | 0 commits por delante de `0.26.x`: ya fusionada (job de versiones mínimas y nota de statsmodels/keras). | Fusionada. | Ya está. Se puede borrar la rama. |
| `dependabot/github_actions/astral-sh/setup-uv-10.3.0` (PR [#1404](https://github.com/skforecast/skforecast/pull/1404)) | 2026-10-14 | Sube `astral-sh/setup-uv` de 10.2.0 a 10.3.0 en 7 workflows. Apunta a `main`. | Lista. | Mantenimiento de CI; fusionar o reabrir contra `0.26.x`. |
| `review/deep-learning-module` | 2026-10-14 | Deep learning: solo handoff y experimentos en `dev/` (`dev/handoff_deep_learning.md`), sin cambios en `skforecast/`. | Fuera de alcance. | Fuera de alcance (no analizada). |

Ramas más antiguas que la ventana (no analizadas): `feature_optimizations` (2026-05-08), `docs_dev` (2026-01-25).

## 2. Issues, PRs y discusiones por tema

Open issues: 18 en total. Open PRs: 9. Discusiones: 34 en total en el repo, solo 3 en los últimos 9 meses
(D1119, D1153, D1191).

### Tema A. Backtesting multiserie con series cortas, que empiezan tarde o con nombres no estándar (se repite)

| Ref | Tipo | Estado | Resumen |
|:--|:--|:--|:--|
| [#1406](https://github.com/skforecast/skforecast/issues/1406) | Bug | Abierto, PR [#1407](https://github.com/skforecast/skforecast/pull/1407) (externo, `mergeable=blocked`) | `backtesting_forecaster_multiseries` descarta en silencio una serie que no se predice en ningún fold; los agregados `average`, `weighted_average`, `pooling` se calculan sin ella. Resultado silenciosamente engañoso. El PR añade `MissingValuesWarning`. |
| [#1410](https://github.com/skforecast/skforecast/issues/1410) | Bug | Abierto, sin PR | `_extract_data_folds_multiseries` (`skforecast/model_selection/_utils.py`, en torno a las líneas 1006 y 1027) mantiene una serie con exactamente `window_size` valores y `fit` falla con `refit`. Ocurre en 1 de cada `steps` casos con series que empiezan en fecha aleatoria. |
| [#1357](https://github.com/skforecast/skforecast/issues/1357) | Bug | Abierto, PR [#1408](https://github.com/skforecast/skforecast/pull/1408) (externo, del 2026-10-17) | `ForecasterRecursiveMultiSeries` falla con `KeyError` si los nombres de las series son enteros (muy habitual con IDs numéricos de tienda o producto). El PR convierte los IDs a `str` en `check_preprocess_series` (+201 líneas). |
| [#1356](https://github.com/skforecast/skforecast/issues/1356) | Bug menor | Abierto, sin PR | El orden de `X_train_series_names_in_` y de los residuos depende de `encoding`. Solo consistencia. |
| [#1354](https://github.com/skforecast/skforecast/issues/1354) | Bug | Abierto, PR [#1389](https://github.com/skforecast/skforecast/pull/1389) (externo) | `set_in_sample_residuals` falla con CatBoost y exógenas categóricas en `ForecasterRecursiveMultiSeries`. |
| [#1196](https://github.com/skforecast/skforecast/issues/1196) | Feature | Cerrado | `min_non_nan_last_window` para no descartar un nivel por un solo NaN en la última ventana. |

### Tema B. Evaluación de intervalos de predicción en backtesting (se repite, desde 2024)

| Ref | Tipo | Estado | Resumen |
|:--|:--|:--|:--|
| [#1409](https://github.com/skforecast/skforecast/issues/1409) | Feature | Abierto, sin PR | Añadir `coverage` e `interval_width` a las métricas de backtesting cuando `interval` tiene dos extremos, por nivel y en los agregados. `calculate_coverage` ya existe en `skforecast.metrics`. |
| [D1119](https://github.com/skforecast/skforecast/discussions/1119) | Discusión | Respondida | Usuario nuevo que calcula a mano métricas por horizonte (`steps` 1, 7, 14, 30) alineando índices. Hoy la salida de backtesting tiene columna `fold` (`skforecast/model_selection/_validation.py:660`) pero no una columna de paso del horizonte: quick win directo. |
| [#1253](https://github.com/skforecast/skforecast/issues/1253) | Feature | Cerrado (ya existen `weighted_interval_score` y Winkler) | Métricas de intervalos: existen, pero no están integradas en backtesting. |
| [#885](https://github.com/skforecast/skforecast/issues/885), [#902](https://github.com/skforecast/skforecast/issues/902) | Preguntas | Cerrados (15 y 8 comentarios) | Confusión con la cobertura de los intervalos y resultados iguales con distintos setups de backtesting. |
| [#1142](https://github.com/skforecast/skforecast/issues/1142) | Bug | Cerrado | Intervalos conformal con `differentiation` crecen linealmente con el horizonte. |
| [#1365](https://github.com/skforecast/skforecast/issues/1365) / PR #1369 | Bug (stats) | Cerrado, corregido en 0.26.0 | `Arima.sigma2_` sin corrección por grados de libertad: intervalos más estrechos que R. |
| PR #1372 | PR (stats) | Cerrado sin fusionar | Varianza analítica de ETS con error multiplicativo: correcta, pero el intervalo normal construido encima no mejora; se descartó. |

### Tema C. Modelos foundation: nuevos adaptadores y compatibilidad con backends (se repite)

| Ref | Tipo | Estado | Resumen |
|:--|:--|:--|:--|
| [#1385](https://github.com/skforecast/skforecast/issues/1385) | Bug | Abierto, sin PR | `T0Adapter` roto con `tfc-t0>=0.4` (0.6.0 es la última): `predict()` ya no acepta `context` ni `quantiles` y desde 0.6.0 torch es un extra (`tfc-t0[torch]`). La llamada sigue igual en `skforecast/foundation/_adapters.py:3724-3729`. Roto "de fábrica" con `pip install tfc-t0`. |
| [#1190](https://github.com/skforecast/skforecast/issues/1190) + PR [#1341](https://github.com/skforecast/skforecast/pull/1341) | Feature | Abierto (11 comentarios), PR externo `mergeable=blocked`, +1376 líneas | Adaptador TiRex-2 (sustituye a #1340, cerrado). |
| [#1206](https://github.com/skforecast/skforecast/issues/1206), [#1221](https://github.com/skforecast/skforecast/issues/1221), [#1252](https://github.com/skforecast/skforecast/issues/1252) | Feature | Cerrados (implementados: TabPFN, T0, Nori) | Peticiones de nuevos modelos foundation, varias de los propios autores de los modelos. |

### Tema D. Exógenas con disponibilidad temporal real ("as-of") en backtesting

| Ref | Tipo | Estado | Resumen |
|:--|:--|:--|:--|
| [#1131](https://github.com/skforecast/skforecast/issues/1131) | Feature | Abierto (3 comentarios), sin PR | Backtesting con exógenas que son a su vez previsiones (meteorología, mercado): usar solo el valor publicado antes de cada origen de previsión. Sin esto, el backtest usa el valor final y hay fuga de información. Los mantenedores lo aplazaron "a la release siguiente" (marzo de 2026). Un tercer usuario aportó una propuesta de `exog_builder`. |

### Tema E. Búsqueda de hiperparámetros

| Ref | Tipo | Estado | Resumen |
|:--|:--|:--|:--|
| [#1355](https://github.com/skforecast/skforecast/issues/1355) | Bug | Abierto, PR [#1388](https://github.com/skforecast/skforecast/pull/1388) (externo) | `grid_search` y `random_search` fallan con `lags=None` y solo window features. |
| [#995](https://github.com/skforecast/skforecast/issues/995) | Pregunta | Abierto desde 2025-07 | Cómo usar el pruning de Optuna en `bayesian_search_forecaster`. |

### Tema F. Window features y features personalizadas

| Ref | Tipo | Estado | Resumen |
|:--|:--|:--|:--|
| [D1153](https://github.com/skforecast/skforecast/discussions/1153) | Discusión | Respondida (14 comentarios) | El usuario intenta meter una asimetría móvil en `transformer_series`; la receta de window features personalizadas para multiserie le da errores con sus datos (520 series) hasta varios intercambios. Señal de que faltan estadísticos habituales en `RollingFeatures` (asimetría) y de que los errores de las clases personalizadas son difíciles de diagnosticar. |
| [#1319](https://github.com/skforecast/skforecast/issues/1319) | Bug | Cerrado | `RollingFeatures` con `kwargs_stats=None` lanzaba `TypeError`. |

### Tema G. Migración desde otras librerías

| Ref | Tipo | Estado | Resumen |
|:--|:--|:--|:--|
| [D1191](https://github.com/skforecast/skforecast/discussions/1191) | Discusión | Respondida | Migrar de `sktime.make_reduction` (detrend, deseasonalize, differencer) a skforecast. |
| [#974](https://github.com/skforecast/skforecast/issues/974) | Issue | Abierto desde 2025-06 | Hoja de ruta para que skforecast sea el motor de reducción de sktime. |
| En `0.26.x` ya hay guía de migración desde Prophet (commits del 2026-10-13/14). | | | |

### Tema H. Varios (abiertos, baja prioridad)

- [#1386](https://github.com/skforecast/skforecast/issues/1386) + PR [#1387](https://github.com/skforecast/skforecast/pull/1387): `plot_residuals` ignora `fig` si hay otra figura activa (+10/-5, listo para revisar).
- [#1084](https://github.com/skforecast/skforecast/issues/1084): soporte de `cudf.pandas` (sin actividad desde abril).
- [#1139](https://github.com/skforecast/skforecast/issues/1139): tests fallidos por la resolución de fechas de pandas (marzo, etiqueta `dependencies`).
- [#854](https://github.com/skforecast/skforecast/issues/854) diferenciación fraccional; [#626](https://github.com/skforecast/skforecast/issues/626) forecaster multivariante de un modelo: antiguos, sin actividad.
- PRs cerrados sin fusionar con decisión tomada: #1172 (`py.typed`, descartado tras medir), #1334 (import diferido de numba, tests fallidos), #1313 (Optuna 5, dependabot ignorado; el entorno de nube ya usa `optuna 5.0.0` sin problemas).

### Deep learning

1 issue abierto ([#1065](https://github.com/skforecast/skforecast/issues/1065)) y 3 cerrados en el periodo (#1087, #861, #776). Fuera de alcance.

## 3. Temas recurrentes (voz directa de los usuarios)

1. **Backtesting multiserie con series heterogéneas** (6 refs: #1406, #1410, #1357, #1356, #1354, #1196). Paneles reales con series que empiezan tarde, son cortas o tienen IDs enteros. Hay 3 PRs externos abiertos (#1407, #1408, #1389). #1406 es un resultado silenciosamente engañoso en los agregados.
2. **Evaluación de intervalos y métricas por horizonte** (8 refs: #1409, D1119, #1253, #885, #902, #1142, #1365, PR #1372). Pregunta recurrente desde 2024: ¿son buenos mis intervalos? ¿cuánto empeora el error con el horizonte? Las piezas existen (`calculate_coverage`, `crps_from_quantiles`, `weighted_interval_score`, columna `fold`), falta integrarlas en backtesting y añadir la columna de paso.
3. **Modelos foundation** (6 refs: #1385, #1190, PR #1341, #1206, #1221, #1252). Demanda alta de nuevos adaptadores; y riesgo de rotura por cambios de API de los backends (#1385, T0 roto hoy).
4. **Exógenas como previsiones y fuga de información** (#1131, más el patrón de preguntas sobre exógenas futuras). Una sola issue, pero es fuga silenciosa en un flujo de producción habitual.
5. **Window features personalizadas** (D1153, #1319, #648 antiguo). Fricción para crear estadísticos que `RollingFeatures` no trae.
6. **Migración desde sktime y Prophet** (D1191, #974; guía de Prophet ya hecha).

## 4. Candidatos inmediatos para 0.27.0

- Fusionar #1400 (ETS) y #1402 (datasets), resolviendo el conflicto de la sección 0.27.0 en `releases.md`.
- Revisar y fusionar PRs externos de bugs: #1407 (#1406), #1408 (#1357), #1389 (#1354), #1388 (#1355), #1387 (#1386).
- Corregir #1410 (comparación `<` frente a `<=` de `window_size` en `_extract_data_folds_multiseries`) y #1385 (T0 con `tfc-t0>=0.4`).
- Implementar #1409 junto con una columna de paso del horizonte en las predicciones de backtesting (D1119).
- Decidir TiRex-2 (PR #1341) y el alcance de #1131 (exógenas as-of).
