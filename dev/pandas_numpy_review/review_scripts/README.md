# Scripts de la revisión independiente

Acompañan a [../REVISION.md](../REVISION.md). Ninguno modifica `skforecast/`. Todos empiezan con `import _env` (bloquea `pyarrow` con `NO_PYARROW=1` y da las versiones) y se lanzan en los tres entornos con:

```bash
bash run3.sh script.py [args]     # pandas 2.3, pandas 3 sin pyarrow, pandas 3 con pyarrow
```

`run3.sh` pone en `PYTHONPATH` el árbol medido (`SK_TREE`, por defecto `/home/user/sk_perf`, un worktree de `perf/multiseries-predict-backtesting` en `ab62227`) y `OMP_NUM_THREADS=1`. El entorno de pandas 2 se crea como dice el anexo de `INFORME.md`, añadiendo `xgboost catboost`.

| Fichero | Qué es | Sección de REVISION.md |
|---|---|---|
| `fase1_nota_a_ciegas.md` | Nota de la fase 1, guardada en un commit antes de leer el informe | 1 |
| `proto_bt_once.py` | Prototipo de backtesting multiserie sin refit que valida y prepara una vez y devuelve las mismas salidas | 2 |
| `proto_bt_once_cases.py` | 25 casos que el techo no cubre: salidas y avisos frente al código actual | 2.2 |
| `time_bt_once.py` | Desglose del tiempo actual y tiempo del prototipo (medir a solas) | 2.1, 2.3 |
| `proto_bt_batched.py` | Alternativa descartada: todos los folds en una pasada | 6.2 |
| `time_small.py` | Coste por fold univariante, P2, P7, P17 | 4, 6.1 |
| `spot_checks.py` | Comprobaciones propias de dos hallazgos de los subagentes | 3.2, 4 |
| `p1_*.py` | P1 y D-NP2 (subagente; `p1_common.py` tiene el prototipo del arreglo) | 3.1 |
| `p1c_*.py` | Revisión adversarial de la opción (c) de D-NP2 (subagente; `p1c_common.py` tiene el prototipo `option_c` y el conjunto de 30 series). Salidas en `results/p1c_*.txt` | 3.1 |
| `p2_*.py` | P2 y D-NP3 (subagente; `p2_proto.py` tiene el prototipo) | 3.2 |
| `misc_*.py` | I3 a I11 y P4 a P17 (subagente) | 4, 5 |
| `results/` | Salidas de las medidas de tiempo y de los scripts `p2_*` | |
