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

Una tercera batería, `identity_dtypes.py` (312 casos, unos segundos), pasa ventanas con columnas que no son float (`category`, `string`, texto, `object`, `bool`, `boolean` e `Int64` con NA, `int`, `datetime` y mezcladas) por `predict` y por `_create_predict_inputs` sin comprobar las entradas, y anota si la ventana del usuario cambia. No va en las de siempre; se lanza con `BATTERIES`:

Una cuarta, `identity_exog.py` (424 casos, 396 sin `pyarrow`, unos segundos), hace lo mismo con la exog: `int`, `Int64`, `Float64` y `boolean` con y sin NA, dtypes de `pyarrow`, `bool`, `category` de enteros y de texto, texto, `object`, `datetime`, varias columnas de tipos distintos y un diccionario cuyas series dan las filas a predecir con dtypes distintos. Pasa cada una por `predict` (ancha, diccionario con un objeto compartido o uno por serie, con más filas que `steps`, con las columnas en otro orden, sin la exog de alguna serie y formato largo; ventana dada y guardada) y por backtesting, con Ridge y LightGBM, y anota si la exog del usuario cambia. Con NA en las filas a predecir todos los casos fallan en la base (pandas no convierte su NA a float): cubren los errores.

Una quinta, `identity_labels.py` (unos 3500 casos, varios minutos), pasa una exog en diccionario con nombres de columna que no son texto (enteros, float, NaN, `None`, booleanos mezclados, `numpy.str_`, fechas con zona, tuplas en un `MultiIndex`), en índices de columnas de varios tipos (`object`, `category`, `string` con y sin `pyarrow`, `str` con `pyarrow`), con los mismos nombres en `fit` y `predict` o solo iguales (1 y 1.0, `True` y 1, `numpy.str_` y `str`, la misma hora en dos zonas), y exog con el índice con nombre o sin frecuencia. Cubre el cambio 4 (`reindex_like` solo conserva las etiquetas del usuario cuando los dos ejes son `identical`). Ordena los nombres de los conjuntos que imprimen algunos avisos, porque con nombres NaN su orden depende de las direcciones de memoria.

`probe_labels7.py` (unos minutos por entorno, no importa skforecast) es la sonda de etiquetas del cambio 7: comprueba, con nombres de columna y tipos de índice como los de `identity_labels.py`, qué construcciones baratas de las columnas de la exog son `identical` a las de la plantilla vacía que se creaba antes, y que `reindex(index=..., columns=...)` da lo mismo que `reindex_like` con ella, errores incluidos. La batería `labels` no distingue esas construcciones (solo cambia el camino, no las salidas), así que la prueba es esta sonda. Se lanza suelta en cada entorno: `[NO_PYARROW=1] python probe_labels7.py`.

`probe_labels4b.py` (de 10 a 15 minutos por entorno; los tres pueden ir a la vez, no son tiempos; no importa skforecast) es la sonda del cambio 4b: compara, errores incluidos, la exog de un nivel que devuelve el camino nuevo (índice `equals` al de las predicciones y columnas `identical`: copia superficial con el índice de las predicciones y las columnas del entrenamiento) con la del código del cambio 4 y con un `reindex`, con nombres y tipos de columnas como los de `probe_labels7.py` e índices del usuario con nombre, sin `freq`, con otra `freq` igual, `object`, otra unidad, zona horaria o tipo, más largos, más cortos o desplazados; también que el objeto del usuario no cambia. Como en el 7, ni la batería `labels` ni los tests ven una diferencia de etiquetas (no llega a las predicciones): fue esta sonda la que encontró que hay que poner también las columnas del entrenamiento (`None` y NaN son `identical` en pandas 2). `edges4b.py` pasa los casos límite del 4b por la API pública (`predict`, `predict_interval` y backtesting) en el árbol de `PYTHONPATH`; `edges4b_compare.py` compara dos árboles.

`probe_labels3.py` (unos minutos por entorno; no importa skforecast) es la sonda del cambio 3: con columnas de la ventana y nombres a comprobar de muchos tipos (texto en índices `object`, `category`, `string` y `str`, enteros, float, booleanos, mezclados, `None`, NaN, `pd.NA`, NaT, fechas con y sin zona, tuplas, `numpy.str_`, columnas duplicadas) y listas de nombres en otro orden, con sustitutos iguales para Python, duplicados, subconjuntos o booleanos, compara el resultado de seleccionar las columnas (`df[names]`, errores incluidos) con el de comprobar la ventana entera cuando cada condición candidata lo permite. Fue la que mostró que el conjunto del plan no basta (booleanos frente a enteros, `None` y `pd.NA` en pandas 3) y que la guarda de duplicados sobra. `edges3.py` pasa los casos límite del 3 por la API pública del multiserie (`predict`, `predict_interval`, backtesting), de `ForecasterDirectMultiVariate` y de `ForecasterRnn` (necesita keras) en el árbol de `PYTHONPATH`, y compara dos árboles con `--compare`; hay que fijar `PYTHONHASHSEED`.

Desde el cambio 4, cada cambio pasa las cinco:

```bash
BATTERIES="dtypes exog labels" bash run_identity.sh base base      # una vez
BATTERIES="ms trims dtypes exog labels" bash run_identity.sh rama new compare
```

## Tiempos

```bash
bash run_bench.sh <etiqueta> 2 "base new" @core      # 34 casos, 20 y 100 series
bash run_bench.sh <etiqueta> 2 "base new" @adverse   # 14 casos que no deben empeorar
bash run_bench.sh <etiqueta> 1 "base new" @scale     # 600 y 1000 series
python agg_ms.py "$MS_WORK/out/<etiqueta>" --gain    # base frente a rama
python agg_ms.py "$MS_WORK/out/<etiqueta>"           # tres entornos, coste fijo y por serie
```

Los casos con `exogdt=mixed` usan una exog con una columna float, una `int` y una `category` de enteros (variables de calendario), en lugar de tres float; están en `@core` (20 y 100 series) y en `@scale` (600 y 1000).

El adverso `levels=5` (predecir 5 niveles con una ventana de todas las series, que no toma los caminos rápidos de los cambios 2 y 3) está con 20, 100 y 1000 series; el de 20 se añadió con el cambio 3 y no está en `@adverse`.

Los casos `missing=1|lgbm` (20 y 100 series, fuera de los conjuntos) pasan un diccionario sin la exog de la última serie, cuyos predictores de exog quedan NaN (por eso LightGBM): es el adverso del cambio 7, el único caso en el que se crea la plantilla vacía de exog.

Los casos del cambio 4b (20 y 100 series, fuera de los conjuntos): `predict` con exog en diccionario sin `freq` (`nofreq`), backtesting con exog ancha o en diccionario con el índice con nombre (`named`) o sin `freq`, y el adverso `early=1|named` (otras fechas con nombre, que ahora compara las fechas antes del `reindex`). Expresiones regulares de las tandas del 4b en `$MS_WORK/cambio4b_regex.sh` (fuera del repositorio).

El árbol `base` es el de la PR (`cb03fbe96`). Para aislar la ganancia de un cambio, se compara con el árbol del commit anterior: `git archive <commit> skforecast | tar -x -C "$MS_WORK/base_<commit>"` y `MS_BASE_TREE="$MS_WORK/base_<commit>"` al lanzar `run_bench.sh`.

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
