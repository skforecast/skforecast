# Scripts de medida

Miden tiempos, perfilan por líneas y comprueban la identidad de resultados de `ForecasterRecursiveMultiSeries` en tres entornos: pandas 2, pandas 3 con `pyarrow` y pandas 3 sin `pyarrow` (el mismo entorno de pandas 3 con la importación de `pyarrow` bloqueada). Nada se escribe dentro del repositorio.

## Preparación

```bash
export MS_WORK=/c/tmp/ms_work                      # carpeta de trabajo fuera del repositorio
export MS_PY_P2=<python del entorno skforecast_27_py13>
export MS_PY_P3=<python del entorno skforecast_27_py13_p3>
mkdir -p "$MS_WORK/base_tree"
git archive cb03fbe96 skforecast | tar -x -C "$MS_WORK/base_tree"   # árbol base
```

El árbol que se compara con la base es el repositorio, salvo que `MS_NEW_TREE` apunte a otra carpeta (por ejemplo, una copia con un prototipo). Las medidas de tiempo necesitan la máquina libre y enchufada; el proceso se fija al procesador lógico 12 (`MS_CPU` para cambiarlo, `cores.py` para ver cuáles son rápidos).

## Identidad

```bash
bash run_identity.sh base base            # una vez: resultados del árbol base
bash run_identity.sh rama new compare     # la rama, comparada con la base
```

Dos baterías por entorno: `identity_ms.py` (4591 casos de los forecasters de varias series, 1 a 2 minutos) e `identity_trims.py` (2911 casos del bloque anterior, que cubre el resto de forecasters). El comparador exige valores bit a bit, dtypes con su almacenamiento, atributos de índices y columnas, avisos (categoría, texto y fichero al que apuntan) y errores.

## Tiempos

```bash
bash run_bench.sh <etiqueta> 2 "base new" @core      # 24 casos, 20 y 100 series
bash run_bench.sh <etiqueta> 2 "base new" @adverse   # 12 casos que no deben empeorar
bash run_bench.sh <etiqueta> 1 "base new" @scale     # 600 y 1000 series
python agg_ms.py "$MS_WORK/out/<etiqueta>" --gain    # base frente a rama
python agg_ms.py "$MS_WORK/out/<etiqueta>"           # tres entornos, coste fijo y por serie
```

En lugar de un conjunto se puede pasar una expresión regular de nombres de casos (`python bench_ms.py --list` los enumera). `bench_mv.py` mide `ForecasterDirectMultiVariate`.

## Perfil por líneas

```bash
PIN=1 bash run_prof.sh perfil_base base predict 20 wide
PIN=1 bash run_prof.sh perfil_rama new predict 20 wide
python linediff.py "$MS_WORK/out/perfil_rama" predict_20_wide            # pandas 2 frente a pandas 3
python protodiff.py "$MS_WORK/out/perfil_base" "$MS_WORK/out/perfil_rama" predict_20_wide
```

`repro_skip_folds.py` reproduce el fallo de `TimeSeriesFold` con `refit=int` y `skip_folds` (ver "Hallazgos sin tocar" en `SEGUIMIENTO.md`); no forma parte de las medidas.

`lineprof.py` instrumenta solo el código de skforecast (`sys.monitoring`) y da, por línea, las veces que se ejecuta y su tiempo, con las llamadas a pandas, numpy y scikit-learn que hace y sin las funciones de skforecast a las que llama. El perfil infla los tiempos alrededor de un 16 %; sirve para repartir el coste entre líneas, no como medida absoluta. `counts.py` lista las líneas cuyo número de ejecuciones crece con el número de series y `str_spy.py` las que crean o convierten arrays de texto en pandas 3.
