# Scripts de la revisión independiente

Acompañan a [../REVISION.md](../REVISION.md). Ninguno modifica `skforecast/`. Todos empiezan con `import _env` (bloquea `pyarrow` con `NO_PYARROW=1` y da las versiones) y se lanzan en los tres entornos con:

```bash
bash run3.sh script.py [args]     # pandas 2.3, pandas 3 sin pyarrow, pandas 3 con pyarrow
```

`run3.sh` pone en `PYTHONPATH` el árbol medido (`SK_TREE`, por defecto `/home/user/sk_perf`, un worktree de `perf/multiseries-predict-backtesting` en `ab62227`) y `OMP_NUM_THREADS=1`. Los scripts `dnp2_*.py` se lanzan dos veces, con `SK_TREE` apuntando a un worktree de `0.28.x` (`a049e5c`) y a otro con `dnp2_proto.patch` aplicado; nombran sus salidas con el árbol y el entorno (`_env.tree()` y `_env.tag()`). El entorno de pandas 2 se crea como dice el anexo de `INFORME.md`, añadiendo `xgboost catboost`.

| Fichero | Qué es | Sección de REVISION.md |
|---|---|---|
| `fase1_nota_a_ciegas.md` | Nota de la fase 1, guardada en un commit antes de leer el informe | 1 |
| `proto_bt_once.py` | Prototipo de backtesting multiserie sin refit que valida y prepara una vez y devuelve las mismas salidas | 2 |
| `proto_bt_once_cases.py` | 25 casos que el techo no cubre: salidas y avisos frente al código actual. `REFIT=0\|1\|k` y `PROTO=once\|blocks` eligen refit y prototipo | 2.2, 9 |
| `time_bt_once.py` | Desglose del tiempo actual y tiempo del prototipo (medir a solas) | 2.1, 2.3 |
| `proto_bt_blocks.py` | Segundo prototipo: refit, preparación compartible entre candidatos, métricas en numpy | 9 |
| `time_bt_blocks.py`, `bench_data.py` | Tiempos y memoria del segundo prototipo (cada variante en un proceso) | 9.1, 9.3, 9.7 |
| `time_search.py` | Búsqueda de hiperparámetros: hoy frente al prototipo por candidato | 9.2 |
| `time_fit_parts.py` | De qué está hecho un `fit` dentro del backtesting | 9.1, 9.2 |
| `time_p5.py` | P5 sobre el backtesting univariante completo | 9.5 |
| `real_data.py` | Casos reales de tres artículos de cienciadedatos.net (py44, py59, py39): datos, preprocesado, forecasters y folds. `python real_data.py download DIR` (con `pyarrow`) baja los datos una vez; después, `REAL_DATA_DIR=DIR` | 10 |
| `time_real_cases.py` | Prueba del prototipo con esos casos: llamada completa hoy frente a `proto_bt_blocks.py`, con el tiempo del modelo separado; en el caso de una serie, P5 | 10 |
| `time_real_search.py` | Variante de py59: la búsqueda con `TimeSeriesFold` en lugar de `OneStepAheadFold`, hoy frente al prototipo con la preparación compartida | 10 |
| `time_real_osa_search.py` | La búsqueda de py59 con `OneStepAheadFold` tal como la hace el artículo: reparto entre modelo, matrices, métricas y ajuste final (STUDY-osa-search-cost) | 10.6 |
| `check_real_mae.py` | Fidelidad de `real_data.py` a los artículos: métricas del código actual junto a las que publica cada artículo | 10.1 |
| `dnp2_proto.patch` | Prototipo de la opción (a') de D-NP2 como parche sobre `0.28.x` (`a049e5c`): numpy al estimador en el `fit` de multiserie, aviso, índice del nivel, nombres traducidos a índices y variante de CatBoost. Se aplica en un worktree aparte (`git apply`); es un prototipo, no la implementación | 11.1 |
| `dnp2_grid.py`, `dnp2_cmp.py` | Rejilla de 336 configuraciones con el código actual y con el prototipo, y su comparación (también la de `dnp2_dtypes.py`) | 11.2 |
| `dnp2_extra.py` | Receta antigua, configuración del usuario por índice y por nombre, `ordinal_category` con `None`, estimadores envueltos, nombres del estimador, 13 estimadores y forecasters guardados | 11.2, 11.4 |
| `dnp2_dtypes.py` | Exog de 14 dtypes con cada árbol | 11.2 |
| `dnp2_today_matrix.py`, `dnp2_xgb_default.py` | Qué hace hoy cada forecaster con una `category` sin gestionar, y XGBoost con sus parámetros por defecto | 11.3 |
| `dnp2_time_fit.py`, `dnp2_run_perf.sh` | Tiempo y pico de memoria del `fit` de multiserie, hoy y con el prototipo (cada variante en un proceso, a solas) | 11.6 |
| `proto_bt_batched.py` | Alternativa descartada: todos los folds en una pasada | 6.2 |
| `time_small.py` | Coste por fold univariante, P2, P7, P17 | 4, 6.1 |
| `spot_checks.py` | Comprobaciones propias de dos hallazgos de los subagentes | 3.2, 4 |
| `p1_*.py` | P1 y D-NP2 (subagente; `p1_common.py` tiene el prototipo del arreglo) | 3.1 |
| `p1c_*.py` | Revisión adversarial de la opción (c) de D-NP2 (subagente; `p1c_common.py` tiene el prototipo `option_c` y el conjunto de 30 series). | 3.1 |
| `p2_*.py` | P2 y D-NP3 (subagente; `p2_proto.py` tiene el prototipo) | 3.2 |
| `misc_*.py` | I3 a I11 y P4 a P17 (subagente) | 4, 5 |
| `results/` | Salidas de las medidas de tiempo que cita la revisión. En las de casos reales (`real_*.txt`), `_1thread` es LightGBM con `LGBM_N_JOBS=1` (salidas exactas) y `_n300`, la muestra de 300 edificios (`REAL_N_SERIES=300`); sin sufijo, LightGBM con los hilos por defecto de los artículos. Las salidas de los scripts de corrección (`p1_*`, `p1c_*`, `p2_*`, `misc_*`) no se guardan: se regeneran con `run3.sh`. Las de D-NP2 (`dnp2_*.txt`) sí se guardan, también las de corrección: `dnp2_perf_*.txt` son las medidas de 11.6 y `dnp2_tests.txt`, los tests que fallan con el prototipo | |
