# Scripts del estudio pandas frente a numpy

Ninguno modifica el paquete: los prototipos sustituyen funciones en memoria o llaman a métodos privados. Usan el `skforecast` del repositorio (instalado en modo editable). Todos admiten los tres entornos del informe; para pandas 3 sin `pyarrow`, `ceiling_bt_ms.py` acepta `NO_PYARROW=1`.

## En la raíz (fases 2 y 3)

| Script | Qué mide o comprueba | Sección |
|---|---|---|
| `ceiling_bt_ms.py N EXOG EST [FOLDS] [STEPS]` | Backtesting multiserie sin refit actual frente al bucle en numpy con la exog codificada una vez; comprueba que las predicciones son idénticas | §5.1 |
| `run_ceiling.sh` | La matriz de `ceiling_bt_ms.py` (Ridge y LightGBM, 100 y 600 series, exog ninguna, float y mezclada, tres entornos). Rutas de los entornos al principio del fichero | §5.1 |
| `preencode.py N` | Codificar la exog de todas las series una vez: serie a serie, tabla concatenada y búsqueda por códigos | §5.1 |
| `capture_fit.py` | Qué matriz recibe `estimator.fit` en cada forecaster según el dtype de la exog | §4.1 |
| `ms_numpy_fit.py` | `fit` multiserie con numpy en vez del DataFrame: mismas predicciones y tiempo | §4.1 |
| `ct_names.py` | `Pipeline` con `ColumnTransformer` por nombre en `ForecasterRecursive` y multiserie | §4.1 |
| `fn_warn.py` | Avisos de nombres de columnas en el `predict` multiserie | §4.1 |
| `verify_cat_bug.py` | Bug P1: LightGBM, `categorical_features=None`, `category` de enteros | §4.2 |
| `bool_exog_fit.py` | `fit` de `ForecasterRecursive` con una exog `bool`: tiempo y memoria | P2 |
| `obj_vs_float.py` | Ridge y LightGBM entrenados con un array `object` frente a float: mismo modelo | P2 |
| `ctxy_dtypes.py` | Dtypes que devuelve `create_train_X_y` con una exog `bool` | P2 |
| `float64_na_fit.py` | Exog `Float64` con NA en `fit` | I10 |
| `eqdate_nan.py` | `ForecasterEquivalentDate` con NaN según el tipo de `offset` | I3 |

## Inventario de la fase 1

- `single_series/`: familia recursiva de una serie (`common.py` define los casos; `predict_bench.py`, `predict_prof.py`, `bt_bench.py`, `search_bench.py`, `micro.py`).
- `direct/`: `ForecasterDirect` y `ForecasterDirectMultiVariate` (`bench_direct.py`, `bench_dmv.py`, `micro_*.py`, `prof.py`, `sites.py`).
- `multiseries/`: `fit`, one-step-ahead y residuos de multiserie (`prof_fit.py`, `micro_fit.py`, `prof_osa.py`, `oos.py`, `catcheck2.py` con el bug P1, y otros auxiliares).
- `results/`: salidas de `run_ceiling.sh` y de los perfiles de la fase 1.

Son scripts de usar y medir, no código mantenido: si cambia la firma de un método privado, hay que adaptarlos.
