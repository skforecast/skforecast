# skforecast 0.28.0 y pandas 3: qué entró, por qué y qué queda abierto

Fecha: 2026-11-20, sobre `0.28.x` en `bdc9ae521`; el estado del ROADMAP es el del commit que
añade este informe. Para revisar la migración sin leer las PRs.

## 0. Cómo leer este informe

La Parte I es el resumen. La Parte II tiene un anexo por PR, en orden de fusión. El
apéndice A recoge la medición final de rendimiento. **p2** es pandas 2.3.3 y **p3**, pandas
3.1.1. Con pandas 3, el texto (`str`) usa `pyarrow` si está instalado; "sin `pyarrow`" es un
`pip install skforecast` limpio. Las medidas son de Windows 11, Python 3.13.18 y
scikit-learn 1.9.1, en entornos que solo difieren en pandas, salvo que se diga otra cosa.
Cada cifra cita su fuente: una PR (`#nnnn`, su descripción), un commit o estos documentos,
legibles con `git show` aunque se borren:

| Abreviatura | Documento |
|---|---|
| P3 §N | `bdc9ae521:dev/pandas3_migration/INFORME.md` (estudio y diario de la migración) |
| R28 | `bdc9ae521:dev/roadmap_0.28.0/roadmap.md` (roadmap de la 0.28.0, congelado) |
| MSPERF §N | `30195692f:dev/multiseries_predict_perf/MEDIDAS.md` |
| ROADMAP, FAQ | `dev/ROADMAP.md` y `docs/faq/pandas-2-and-pandas-3.md` |

---

# Parte I. Resumen ejecutivo

## 1. En una página

skforecast 0.28.0 funciona con pandas 2.3 y con pandas 3 (`pandas>=2.3, <4.0`) con el mismo
código. Con los mismos datos, predicciones, intervalos y métricas son iguales bit a bit en
las dos versiones, salvo dos casos documentados en la FAQ (sección 3). Lo que sí cambia son
los tipos que pandas 3 cambia para todos: texto `str` en lugar de `object`, fechas en
microsegundos, índices con zona horaria que pierden la frecuencia en `tz_convert` y arrays
de solo lectura.

Lo que depende de la versión de pandas vive en un módulo privado,
`skforecast/utils/_pandas_compat.py`, y en dos puntos de `utils.py`: una pista en un mensaje
de error y una nota en `load_forecaster`. No hay más ramas por versión en el paquete
(sección 4).

Son 15 PRs: 4 en `0.27.x` (el arnés de verificación y tres arreglos previos) y 11 en
`0.28.x`. Otras 4 PRs de rendimiento, que nacen de la migración pero no son parte de ella,
recuperan en `predict` y en el backtesting lo que pandas 3 pierde por cada objeto de pandas
que se crea. Por el camino se arreglaron errores que también afectan a pandas 2 y que están
en la 0.27.0 publicada. Los más serios: transformadores que modificaban los datos del
usuario y daban métricas erróneas sin avisar (#1504) y `reshape_series_wide_to_long`
perdiendo filas (#1497).

La verificación se apoya en un arnés que compara bit a bit las salidas de cada PR con su
base, en pandas 2 y en pandas 3, en la suite completa en las dos versiones, en cuatro
revisiones independientes y en la CI completa (sección 6). En rendimiento, pandas 3 queda a la
par de pandas 2 en `fit`; en las matrices de entrenamiento, de 0,79 a 1,20 veces con `pyarrow`
(0,73 a 1,11 sin él), salvo la de solo lags (2,5 veces, deliberado); y por detrás en `predict`
en bucle, sobre todo con `pyarrow` instalado (sección 7).

Queda abierto (secciones 8 y 9):

- relanzar la CI completa, que no corre desde el 2026-11-16;
- confirmar con un A/B dos pérdidas de pandas 3 que vio un informe externo y que el arnés no
  mide: las búsquedas de hiperparámetros y la matriz multiserie con exógenas de texto;
- decidir si hace falta una revisión independiente del código fusionado después;
- pasar `benchmarks/` y 5 notebooks a pandas 3;
- cuatro errores previos, independientes de pandas (#1506 a #1509);
- sin fecha, retirar pandas 2, bloqueado por `gluonts`.

## 2. Qué trae pandas 3 y dónde chocaba con skforecast

pandas 3.0 cambia cuatro comportamientos por defecto que tocan a skforecast:

- **Copy-on-Write.** Todo objeto derivado de otro se comporta como una copia, y pandas
  copia solo cuando hace falta. Tres consecuencias para skforecast:
  - el argumento `copy` de `concat` y `astype` deja de tener sentido y emite un
    `Pandas4Warning` en pandas 3.1;
  - el constructor de `DataFrame` copia por defecto un array de numpy que recibe (pandas 2
    no lo hacía), y eso era casi toda la diferencia de rendimiento en las matrices de
    entrenamiento (sección 7);
  - `to_numpy()` y `.values` pueden devolver vistas de solo lectura.
- **Texto `str`.** Las columnas de texto son de un dtype propio, `str`, en lugar de
  `object`. Se guarda en Arrow si `pyarrow` está instalado y en objetos de Python si no.
  Los valores ausentes son `NaN`.
- **Resolución de fechas.** Un índice creado a partir de texto o con `date_range` es
  `datetime64[us]`, no `[ns]`. Las fechas son las mismas; cambia el dtype.
- **Zonas horarias y `Day`.** `tz_convert` descarta las frecuencias que no son de duración
  fija (`'D'`, `'W'`, `'MS'`...), y `Day` pasa a ser un día de calendario, no 24 horas.

Antes de empezar, con pandas 3.1.0 fallaban 94 tests de `0.28.x` (91 por pandas 3 y los 3
locales de TensorFlow), y el paquete emitía 1748 `Pandas4Warning` desde 30 sitios (#1495).
El estudio, sobre `0.27.x` y otra fecha, agrupó sus 90 fallos de pandas en tres familias
(P3 §2.1): resolución de fechas (46 tests), dtype `str` (32) y zonas horarias, `Day` y
comparación de frecuencias (12). La mayoría eran expectativas de tests escritas para pandas
2; unos pocos, errores reales del paquete.

## 3. Qué cambia para el usuario

### 3.1 Versiones e instalación

- **Mínimos nuevos.** Python 3.11 (#1496); numpy 2.0, pandas 2.3, scipy 1.13, numba 0.60,
  statsmodels 0.14.6 y matplotlib 3.8.4 (#1500); techo `pandas<4.0` (#1501). statsmodels
  0.14.5 no se importa con pandas 3.1, de ahí su mínimo.
- **Actualizar.** `pip install -U skforecast` conserva el pandas instalado, pero sube numpy
  y pandas si son anteriores a los mínimos. Quien necesite numpy 1.x o pandas 2.2 debe fijar
  `skforecast<0.28` (#1500, #1501). Un entorno nuevo recibe pandas 3.
- **Backends que siguen en pandas 2.** `MoiraiAdapter` y `TabPFNAdapter` dependen de
  `gluonts`, que exige `pandas<3`: instalarlos baja pandas a 2.3.3. Los otros nueve backends
  foundation resuelven pandas 3.1.1 (#1501). `uni2ts` (Moirai) declara numpy 1.26 y scipy
  1.11, pero funciona con numpy 2; la guía de modelos foundation explica cómo instalarlo
  (#1500).

### 3.2 Tipos en las salidas

- **Texto `str`** en las columnas de texto, en la columna `level`, en `exog_dtypes_in_` y en
  los nombres de columna de `X_train` (decisión D4: se acepta, no se normaliza a `object`).
  La única diferencia de resultado: `None` en una columna de texto es una categoría propia en
  pandas 2 y `NaN` en pandas 3, así que un estimador que no admite NaN (`Ridge`) ajusta en
  pandas 2 y falla en pandas 3 (FAQ).
- **Fechas en `[us]`.** Con pandas 3 las salidas conservan la unidad del índice de entrada;
  con pandas 2 las predicciones salen en `[ns]` (#1510). Las fechas son las mismas.
- **Alias de frecuencia retirados.** `PopulationDriftDetector(chunk_size='M')` falla con
  pandas 3 (hay que usar `'ME'`); el error muestra ahora el de pandas, que lo dice (#1510).

### 3.3 Zonas horarias

En pandas 3, un índice creado en UTC con `freq='D'` y convertido a hora local con
`tz_convert` llega sin frecuencia, y el forecaster lanza el `ValueError` de siempre ("has a
pandas DatetimeIndex without a frequency"). Con pandas 3, el mensaje añade una pista: usar
`freq='24h'` (diario) o `'168h'` (semanal), que pandas conserva y dan las mismas predicciones
que `'D'` en pandas 2, o dejar el índice en UTC (#1497). No se infiere la frecuencia por el
usuario (D6, sección 5.2).

Primera excepción a "mismos resultados":
`ForecasterEquivalentDate(offset=pd.offsets.Day(n))` con datos horarios con zona predice
distinto tras un cambio de hora, porque `Day` es ahora un día de calendario (23 o 25 horas
ese día). `offset=24 * n` da lo mismo en las dos versiones (FAQ).

### 3.4 Arrays de solo lectura

Con Copy-on-Write, `to_numpy()` de un `DataFrame` (también de las salidas de skforecast) y
los arrays de `out_sample_residuals_` pueden ser de solo lectura; `to_numpy(copy=True)` o
`.copy()` dan uno escribible (FAQ).

La matriz que devuelve el `create_train_X_y` público es escribible siempre con pandas 3. Con
pandas 2 se comporta como en la 0.27.0: con solo lags contiguos y un estimador que no pide
una matriz contigua en C (en el clasificador, con cualquier estimador) es una vista de solo
lectura; en el resto de casos, escribible (#1502, D13).

### 3.5 Guardar y cargar forecasters

- De pandas 2 a pandas 3 funciona con todos los backends, salvo un caso: un forecaster
  entrenado con pandas 2 sobre fechas diarias convertidas desde UTC (`freq='D'`) solo acepta un
  `last_window` con `'D'` (con `'24h'` lanza `TypeError`), y pandas 3 no admite `'D'` en esas
  fechas si cruzan un cambio de hora. Hay que reentrenarlo con `'24h'` (FAQ).
- De pandas 3 a pandas 2.3, solo con skops. joblib, pickle y cloudpickle fallan dentro de
  pandas con errores sin explicación (un `AssertionError` vacío, un `NotImplementedError` o
  un `TypeError` de `StringDtype`). Con pandas 2, `load_forecaster` añade a esos errores una
  nota que lo explica, sin cambiar su tipo (#1497, D8).
- Con skops, la lista `trusted` depende de la versión de pandas que guardó el fichero (por
  ejemplo, `pandas.core.indexes.datetimes.DatetimeIndex` en pandas 2 y
  `pandas.DatetimeIndex` en pandas 3), así que una lista escrita para una versión no sirve
  para los ficheros de la otra (#1510, FAQ).

### 3.6 Arreglos que cambian resultados también con pandas 2

Salieron de la migración, pero son errores de versiones publicadas, también de la 0.27.0:

- **Transformadores que escriben en su entrada** (`StandardScaler(copy=False)`, también
  dentro de un `Pipeline`, o un `FunctionTransformer` con una función in situ; con pandas 3,
  también un `ColumnTransformer` que los contenga) modificaban `y`, `series` y `exog` del
  usuario. Dos `predict` seguidos con la misma `exog` daban distinto; en las reproducciones de
  #1504, un backtesting con refit daba
  MAE 37,1 en lugar de 0,446, y una búsqueda one-step-ahead con diferenciación, 6,68 en lugar
  de 0,245 (#1504). La copia nueva cambia en pandas 2 algunos valores en el orden del
  redondeo (hasta 2,6e-13 relativo); no lleva nota de versión.
- **`reshape_series_wide_to_long` sin frecuencia** asumía datos diarios: con datos horarios
  perdía filas y con mensuales añadía filas de NaN (también en las salidas `'long'` de
  `TimeSeriesSplitter`). Ahora infiere la frecuencia; si no puede (datos diarios con huecos,
  o menos de 3 fechas), lanza `ValueError` y pide `data.asfreq('D')` (#1497, D15). Es el
  único arreglo que rompe un caso que antes funcionaba.
- **Validación de frecuencias.** `TimeSeriesFold` y `OneStepAheadFold` lanzaban `TypeError`
  en lugar del `ValueError` que lista las frecuencias cuando las series de un diccionario
  tenían frecuencias no comparables (`'D'` y `'MS'`). `expand_index` daba fechas un día antes
  con índices mensuales con zona y sin frecuencia al oeste de UTC (#1497).
- **`ForecasterRecursiveMultiSeries`** lanzaba `TypeError` cuando una exógena de texto tenía
  dtypes de texto distintos entre series (`string` y `object`; con pandas 3, `str` y
  `object`, porque una columna toda vacía es `object`), y las búsquedas one-step-ahead con
  `encoding='onehot'` y nombres de serie enteros emitían un `FutureWarning` de pandas
  (#1497).
- **`calculate_distance_from_holiday` con zona horaria** daba horas en lugar de días, o un
  día de menos al cruzar el cambio de hora de primavera (#1505).

Antes de la 0.27.0 se arreglaron, en `0.27.x`, dos errores de `PopulationDriftDetector` con
categorías sin usar (`ValueError` de chi2 en `fit`, `KeyError` en `predict` y chi2 no
determinista entre procesos; #1445, #1448). Ya están publicados en la 0.27.0, así que no
cambian nada para quien la tenga.

## 4. Cómo queda el código

Este es el mapa para revisar la migración en el código de `0.28.x`.

### 4.1 El módulo de compatibilidad (#1495)

`skforecast/utils/_pandas_compat.py` es el único módulo que lee la versión de pandas. Tiene
una constante y tres funciones:

| Nombre | Qué hace | Dónde se usa |
|---|---|---|
| `PANDAS_GE_3` | `True` con pandas 3 o posterior | Las dos ramas de `utils.py` (4.2) y los tests condicionados (4.5) |
| `concat(objs, **kwargs)` | `pd.concat`, pasando `copy=False` solo en pandas 2 | Dos veces en `ForecasterRecursiveMultiSeries`, al construir la matriz de entrenamiento (`_assemble_train_X_y` y la concatenación de las exógenas) |
| `astype(obj, dtypes, changed_only=False)` | `astype`, con `copy=False` solo en pandas 2. Con `changed_only=True` convierte solo las columnas cuyo dtype cambia y, si ninguna cambia, devuelve el mismo objeto | `create_train_X_y` (con `changed_only=True`) y `create_predict_X` de `ForecasterRecursive`, `ForecasterRecursiveClassifier`, `ForecasterDirect` y `ForecasterDirectMultiVariate`, `create_predict_X` de `ForecasterRecursiveMultiSeries` y `reshape_exog_long_to_dict` (con `changed_only=True`) |
| `infer_freq(index)` | `pd.infer_freq` devolviendo siempre un offset (el comportamiento de pandas 4) y sin el aviso de pandas 3.1 | Dos veces en `_infer_freq_from_index` (4.3) |

Por qué existe, en lugar de quitar `copy=False` en todas las versiones: en pandas 2, quitarlo
costaba hasta 1,9 veces el tiempo y un 22 % más de memoria en las matrices multiserie
(P3 §4.5). El `compat.is_string_dtype` que proponía el estudio no se creó, porque
`pd.api.types.is_string_dtype` ya da lo mismo en las dos versiones.

### 4.2 Las dos ramas por versión fuera del módulo (#1497)

Las dos están en `skforecast/utils/utils.py`:

- **`_TZ_MISSING_FREQ_HINT`**: el texto de la pista de la sección 3.3. Vale `""` con pandas 2,
  donde `tz_convert` conserva la frecuencia. Se añade al `ValueError` de índice con zona sin
  frecuencia en tres validaciones: `check_extract_values_and_index` (`fit` y `predict`),
  `check_preprocess_series` (series en diccionario) y `_extract_index` en
  `model_selection/_split.py` (los folds).
- **`_add_note_pandas_3_file`**: añade con `exc.add_note` la nota de la sección 3.5, solo con
  pandas 2, en los dos bloques `except (AssertionError, NotImplementedError, TypeError)` de
  `load_forecaster` (joblib, pickle y cloudpickle, y skops).

### 4.3 Frecuencias

- **`_infer_freq_from_index`** (`utils.py`): infiere la frecuencia de un índice sin ella. Si
  el índice tiene zona y avanza en pasos fijos de UTC (creado en UTC y convertido), la infiere
  sobre las fechas en UTC, porque en hora local daría una frecuencia errónea (`'ME'` en lugar
  de `'MS'` al oeste de UTC) o ninguna. La lógica de UTC venía de la 0.26 (#1343, para
  `expand_index`); #1497 la sacó a esta función y la usa también en
  `reshape_series_wide_to_long`, y #1505 en `calculate_distance_from_holiday`.
- **Orden de las frecuencias** en el mensaje "Found frequencies" (`_split.py` y
  `check_preprocess_series`): `sorted(..., key=str)`, porque pandas 3 no sabe comparar
  algunos offsets (`'<' not supported between instances of 'Day' and 'Day'`) y así el texto
  es el mismo en las dos versiones (#1497, D10).

### 4.4 Copias nuevas, y una que se quitó

- **Matriz pública** (#1502). En `create_train_X_y` de `ForecasterRecursive`,
  `ForecasterRecursiveClassifier`, `ForecasterDirect` y `ForecasterDirectMultiVariate`, el
  `DataFrame` se crea con `copy = False if X_train.flags.writeable else None`. Una matriz
  escribible es un array nuevo, y la copia por defecto de pandas 3 sobra. Una de solo lectura
  es una vista (de la copia interna de `y`, o de la salida de una window feature) y conserva
  el comportamiento de cada versión: pandas 2 no la copia, pandas 3 sí. `fit` no pasa por
  ese constructor.
- **Exógenas antes de codificar las categóricas** (#1502, #1530). Sin `transformer_exog`,
  `exog` puede ser el objeto del usuario, y asignarle las columnas codificadas lo modificaría.
  Antes se hacía `exog.copy()`; ahora `exog.copy(deep=False)`, que basta porque la
  asignación sustituye las columnas en lugar de escribir en sus arrays. Son 9 sitios en
  `fit` y `predict` de los cinco forecasters, y uno más en el `predict` multiserie (#1530).
- **Entrada de los transformadores** (#1504). `transform_numpy`, `transform_series` y
  `transform_dataframe` copian siempre su entrada antes de llamar al transformador (D14). En
  `transform_series` la copia es profunda a propósito: scikit-learn vuelve escribibles los
  arrays de solo lectura de pandas 3, así que una copia superficial dejaría escribir en los
  datos del usuario (comentario en el código).

### 4.5 Tests, CI y empaquetado

- **Tests.** Son independientes de la versión (#1499): los que dependen del dtype de texto
  comparan con `STR_DTYPE = pd.Series(['a']).dtype` (6 ficheros) y 8 ficheros usan
  `PANDAS_GE_3` para las expectativas que difieren por versión (18 líneas, ninguna como
  `skipif` de módulo entero).
- **CI.**
  - `unit-tests.yml`: Python 3.11 a 3.14 en Linux, Windows y macOS, todo con pandas 3, más
    una celda con `pandas<3` (Python 3.12, Linux).
  - `unit-tests-min-deps.yml`: Python 3.11 con los mínimos (pandas 2.3.0, numpy 2.0.0,
    scikit-learn 1.6.0).
  - `unit-tests-latest-deps.yml`: convierte en error todo `Pandas4Warning` nacido en
    skforecast o en sus tests (D5). Solo se lanza a mano o cada semana sobre `main`, así que
    hasta publicar la 0.28.0 solo vigila `0.28.x` si se lanza a mano; en `main` (código de la
    0.27, con pandas 3) falla con 91 tests desde el 2026-11-16 (run 43912433207).
  - `unit-tests.yml` y `unit-tests-min-deps.yml` no corren en las PRs a `0.28.x`: se lanzan a
    mano y corren en la PR de la versión a `main`.
- **Empaquetado.** `pyproject.toml`: `requires-python = ">=3.11"`, `numpy>=2.0`,
  `pandas>=2.3, <4.0`; el extra `test` lleva `pandas[parquet]>=2.3`, sin techo (punto
  abierto, sección 9).

## 5. Decisiones y su porqué

### 5.1 D1 a D15

El estudio (P3 §8) planteó 15 decisiones con opciones y una recomendación; cada una se
cerró en la PR que la necesitaba, y varias cambiaron por el camino. La tabla da el estado
final; la sección 5.2 explica las que no se entienden en una línea.

| Id | Tema | Recomendación del estudio | Lo fusionado |
|---|---|---|---|
| D1 | Python y rango de pandas | Python 3.10, `pandas>=2.2, <4.0` | Python 3.11 (#1496), `pandas>=2.3, <4.0` (#1500, #1501) |
| D2 | Mínimo de pandas 2 | 2.2.0 | 2.3, junto con numpy 2.0 (#1500) |
| D3 | Dónde va lo que depende de la versión | Módulo `_pandas_compat` | Igual (#1495), sin `is_string_dtype` |
| D4 | dtype `str` en las salidas | Aceptarlo | Igual |
| D5 | `Pandas4Warning` en CI | Error en el job semanal | Igual, con `-o filterwarnings` (#1501) |
| D6 | Índice con zona y sin `freq` | Documentar `'24h'` y mejorar el mensaje | Igual, la pista solo con pandas 3 (#1497) |
| D7 | Dos copias con `encoding=None` en pandas 3 | Nada | Nada |
| D8 | Guardar en una versión y cargar en otra | Documentar y nota en `load_forecaster` | Igual (#1497, #1510) |
| D9 | pandas de `benchmarks/` | `pandas==2.3.3` hasta la release | Igual; pendiente (MAINT-bench-pandas3) |
| D10 | Orden de las frecuencias en los mensajes | Ordenar por duración | `sorted(key=str)` (#1497) |
| D11 | Rama de destino | `0.27.x` | `0.28.x`; los arreglos de errores publicados, a `0.27.x` |
| D12 | MASE y RMSSE con NaN (`stack`) | `stack(future_stack=True)` | Nada: ya resuelto en la 0.27.0 |
| D13 | Matriz de solo lags | `copy=False` siempre | `copy=False` solo si la matriz es escribible (#1502) |
| D14 | Cuándo copiar la entrada del transformador | Si `transformer.copy` no es `True` | Siempre (#1504) |
| D15 | `reshape_series_wide_to_long` sin `freq` | `ValueError` | Inferirla; `ValueError` si no se puede (#1497) |

### 5.2 Las decisiones que conviene entender

**D1 y D2: Python 3.11, numpy 2.0 y pandas 2.3.** La recomendación inicial del estudio era
no dejar fuera a nadie: Python 3.10 y pandas 2.2. Pero Python 3.10 perdió soporte el
2026-10-01, antes de publicar, y pandas 3 ya exige 3.11: el 2026-11-03 se eligió 3.11, en su
propia PR (#1496). Con numpy 1.x permitido, el rango `pandas<4.0` dejaba instalar pandas 3.0
con numpy 1.26, una combinación que ningún job probaba y que en Windows daba 255 fallos, unos
220 por columnas enteras `int32` de numpy 1.x (R28). numpy 2.0 exige al menos pandas 2.2.2
(2.2.0 y 2.2.1 no funcionan con él); se eligió 2.3, la única serie 2.x que aún recibe
arreglos. Efecto lateral: skops
funciona en los dos sentidos sin código propio (con pandas 2.2 no cargaba un fichero de
pandas 3 con exógenas de texto).

**D3: un módulo de compatibilidad.** Las alternativas eran comprobar `pd.__version__` donde
hiciera falta, o no tener rama y aceptar en pandas 2 la copia que evita `copy=False`. La
segunda costaba hasta 1,9 veces el tiempo en las matrices multiserie (P3 §4.5). El módulo
deja la rama en un sitio y hace trivial retirarla con pandas 2.

**D5: el filtro de avisos.** La idea es que una deprecación de pandas falle en CI antes de
pandas 4, pero solo la que nace en skforecast. La forma obvia,
`-W error::pandas.errors.Pandas4Warning:skforecast`, no filtra nada: pytest exige que el
módulo coincida entero y ningún aviso se atribuye al módulo `skforecast` a secas. Sin
módulo, falla al importar statsmodels 0.14.6, que emite 131 avisos con pandas 3.1.1. La
forma fusionada, `-o "filterwarnings=error::pandas.errors.Pandas4Warning:skforecast"`,
filtra por prefijo e incluye los tests; a cambio sustituye la lista `filterwarnings` del
`pyproject.toml`, hoy vacía (P3 §8).

**D6: no inferir la frecuencia por el usuario.** La alternativa era inferirla en UTC en las
validaciones y aceptar el índice sin `freq`. Se descartó porque skforecast exige `freq`
explícita por diseño, y porque inferirla adivinaría la convención del usuario y añadiría una
rama en cuatro sitios. `expand_index` sí la infiere en UTC, pero solo para construir fechas
futuras, no para validar.

**D7: nada que hacer.** Con `encoding=None` y exógenas, pandas 3 copia dos veces la matriz
multiserie porque `drop` es perezoso. La diferencia (1,11 veces) estaba dentro del ruido
entre procesos y por debajo de la de un caso sin `drop` (1,18), así que no había ganancia
que justificar un cambio (P3 §8).

**D11: todo en `0.28.x`.** El plan inicial era la 0.27.0. El 2026-11-04 se movió a la 0.28:
la migración solo sirve completa (hasta el pin de #1501 ningún usuario nota nada), iba a
tardar semanas y no debía retrasar la 0.27.0, y sus cambios de mínimos y de resultados
encajan mejor en una minor nueva. Los errores de versiones publicadas que salieron por el
camino se arreglaron en `0.27.x` (#1445, #1448).

**D12: nada que hacer.** El estudio vio que MASE y RMSSE daban distinto en pandas 2 y 3 con
NaN en la serie objetivo, por un `stack()`. El commit `c8a59be5a`, de la 0.27.0, ya había
reescrito esas métricas sin `stack`.

**D13: la matriz de solo lags.** Con solo lags contiguos, la matriz es una vista de solo
lectura de la copia interna de `y`. Con pandas 2 se envuelve sin copiar; con pandas 3 el
constructor la copia (2,5 veces el tiempo y 6 veces la memoria, apéndice A). El estudio
proponía pasar siempre `copy=False`, para que pandas 3 hiciera lo mismo que pandas 2. Una
revisión de #1502 vio que con pandas 3 esa matriz de solo lectura rechazaba más operaciones
que la de pandas 2 (`X_train.loc[:, col] = value` da `TypeError` en lugar de un
`FutureWarning`, y `fillna(inplace=True)` falla) y que podía compartir memoria con la salida
de una window feature propia. Se dejó la copia de pandas 3 para ese caso: es la única
matriz de entrenamiento que sigue claramente peor con pandas 3, y es deliberado.

**D14: copiar siempre.** El estudio proponía copiar solo si `transformer.copy` no es `True`.
Con 25 transformadores probados, esa condición copia igual con `Pipeline`,
`ColumnTransformer`, `FunctionTransformer` y los transformadores propios (la mayoría no
modifica nada), solo ahorra la copia con los escaladores por defecto y no protege de un
transformador propio que declara `copy=True` y escribe en su entrada. La variante `copy is
False` no detecta un `Pipeline` ni un `FunctionTransformer`. Copiar siempre es más simple y
cuesta poco: de 4 a 7 ms en un `fit` de 200 000 × 10 (unos 200 ms) y unos 10 µs por
`predict`, sin cambio en el pico de memoria (P3 §8, #1504).

**D15: inferir la frecuencia en `reshape_series_wide_to_long`.** El estudio proponía un
`ValueError` siempre que el índice no tuviera `freq`. Eso rompía el caso más común, un índice
diario sin `freq` leído con `read_csv`, que funcionaba. Inferirla arregla los casos horario y
mensual y mantiene el diario; solo el diario con huecos pasa a lanzar `ValueError`.

### 5.3 Cómo se organizó el trabajo, y lo que el estudio dice y no se fusionó

El estudio (P3 §7) planeaba 8 PRs. Las PR 1 y 2 se fusionaron en una (#1495). La PR 3 se
partió en dos: las copias que sobran (#1502) y los transformadores (#1504). Python 3.11
(#1496) y los mínimos de numpy y pandas (#1500) fueron PRs propias antes del pin (#1501). La
PR 8, que descomponía `pd.StringDtype` para que skops cargara en pandas 2.2 los ficheros de
pandas 3, dejó de hacer falta al subir el mínimo a 2.3 (D2). #1498 fue un arreglo de la CI
de mínimos (`pyarrow` 26 no funciona con numpy 1.x), necesario hasta #1500.

P3 §1, §7 y §8 son el plan de octubre y no recogen lo fusionado: difieren en D1, D2 y D10
a D15, en el `compat.is_string_dtype` que no se creó y en sus cifras de rendimiento, que son
del prototipo. P3 §11.3 dice que el backtesting llama a `predict` con `check_inputs=False`;
para la predicción puntual es falso: `_fit_predict_forecaster` pasa
`check_inputs=True if interval is None else False`, así que cada fold valida sus entradas.

## 6. Cómo se verificó

**Identidad bit a bit.** El arnés de #1444 (en `dev/pandas3_migration/harness/`, se borrará
con la carpeta) ejecuta `fit`, `predict`, intervalos, backtesting, búsquedas, el detector de
deriva y guardar y cargar sobre una batería de casos. De cada salida guarda una huella
estricta (valores, dtypes, índices y nombres) y compara la rama con su base, en pandas 2 y en
pandas 3. También comprueba que las entradas del usuario no cambian, que dos `predict`
seguidos coinciden y qué salidas comparten memoria con una entrada. Pasó de 205 salidas
(#1495) a 258 (#1505).

- Identidad estricta en #1495, #1502, #1505, #1512, #1516 y #1530 (sus descripciones), y en
  #1497 y #1501 (P3 §10).
- En #1504 difieren solo las salidas de los dos casos que arregla (los transformadores in
  situ): 5 de 257 en p2 y 3 en p3. Los cambios de redondeo de pandas 2 (sección 3.6)
  salieron de una batería aparte de 110 salidas.
- p2 frente a p3 con el mismo árbol: ninguna salida con valores distintos; 125 de 258
  difieren solo en el dtype (`str`, `[us]`), como se espera (#1505).
- Las PRs de rendimiento añadieron baterías propias, sin diferencias: 2911 casos en #1516 y
  cinco de hasta 8106 en #1530.

**Tests.** De los 94 tests que fallaban con pandas 3.1.0 (#1495), 3 eran locales de
TensorFlow; #1497 arregló 6 por código y #1499 hizo independientes de la versión los otros
85. Los `Pandas4Warning` del paquete pasaron de 1748 a 0 (#1495); desde #1501, el job de
latest-deps los convierte en error.
Suite completa con #1530: 8499 pasan en p2 y 8492 en p3, con los mismos 3 fallos locales
de TensorFlow, que pasan en CI.

**CI.** Como en las PRs a `0.28.x` no corren tests, los cuatro workflows se lanzaron a mano.
La última vez fue sobre #1505 (`12898d190`, 2026-11-16) y los cuatro pasaron: Python 3.11 a
3.14 en los tres sistemas con pandas 3.1.1, la celda `pandas<3`, las versiones mínimas y 11
backends foundation (P3 §10).

**Revisiones independientes** (R28): hasta el 2026-11-15 todas las revisiones las hizo quien
escribió el código, así que se añadieron cuatro en sesiones nuevas, que no partían de sus
conclusiones, más una CI completa; cada hallazgo se comprobó contra el código antes de
cambiar nada:

1. `/code-review high` de #1502: encontró el problema de la matriz de solo lectura y cambió
   D13.
2. `/code-review high` de #1504: dos errores más, arreglados en la misma PR.
3. La CI completa sobre la rama.
4. `/code-review ultra` de toda la migración: un arreglo, en #1505.
5. Una revisión que ejecutó en p2 y p3 el código que depende de la versión: 11 hallazgos,
   ninguno introducido por la migración. Arreglos en #1505, documentación en #1510 y cuatro
   issues (#1506 a #1509).

## 7. Rendimiento

### 7.1 pandas 3 frente a pandas 2 en la 0.28.x final

Mismo árbol, razón p3 / p2 de los mínimos de 31 benchmarks del arnés (más de 1 es que pandas
3 tarda más). Protocolo y tabla completa en el apéndice A.

| Grupo | p3 con `pyarrow` | p3 sin `pyarrow` | Memoria p3 / p2 | En #1501 |
|---|---|---|---|---|
| `create_train_X_y` público de una serie (7) | 0,79 a 1,20 | 0,73 a 1,11 | 1,00 a 1,14 | 1,33 a 3,01 |
| Ídem, solo lags (vista de solo lectura, D13) | 2,54 | 2,53 | 6,24 | No medido |
| `fit` (7) | 0,96 a 1,05 | 0,94 a 1,04 | 1,00 a 1,14 | 1,00 a 1,04 |
| Multiserie, `create_train_X_y` y `fit` (5) | 0,99 a 1,14 | 0,95 a 1,04 | 0,76 a 1,00 | 0,94 a 1,14 |
| Bucles de `predict` (5) | 1,20 a 1,43 | 1,03 a 1,15 | Menos de 1 MiB | 1,06 a 1,32 |
| Backtesting (3) | 0,96 a 1,16 | 0,96 a 1,04 | 1,00 a 1,03 | 1,06 a 1,13 |

Los tres benchmarks que faltan (`expand_index`, window features y un auxiliar multiserie)
están en el apéndice A. "En #1501" es la misma comparación sobre el árbol de #1501, antes de
las PRs de copias y de rendimiento.

Cómo leerlo:

- **Matrices.** Casi toda la diferencia era la copia del constructor de `DataFrame` (sección
  2), que #1502 quitó. Queda la matriz de solo lags, por decisión (D13).
- **`fit`** está a la par: domina el estimador.
- **`predict` en bucle** es lo que más pierde aún. Es coste por cada objeto de pandas que se
  crea (constructores, indexación), no un punto caliente, y crece con `pyarrow` (7.2).
- La suite de `benchmarks/` de la CI, medida antes de las PRs de rendimiento, daba una media
  geométrica p3 / p2 de 1,03 sin `pyarrow` (R28).
- El arnés no cubre las búsquedas de hiperparámetros ni la matriz multiserie con muchas
  series y exógenas de texto en diccionario. Un informe externo vio pérdidas en las dos, sin
  confirmar aquí (sección 8.1).

### 7.2 Las PRs de rendimiento (#1512, #1516, #1519, #1530)

No son de la migración, pero nacen de ella. Con `pyarrow` instalado, pandas 3 guarda en Arrow
los nombres de columnas e índices de texto, y cada operación con ellos cuesta más: el
`predict` multiserie con exógenas en formato largo y 100 series perdía un 51 % frente a
pandas 2 (MSPERF §1). Cada PR quita pasos que devolvían los mismos datos, con resultados
idénticos, y acelera las dos versiones:

| PR | Qué quita | Ejemplo (base → PR) |
|---|---|---|
| #1512 | `groupby` y copia al pasar series y exógenas de formato largo a diccionario | 5000 series: 1060 → 430 ms en p2, 1793 → 607 ms en p3 con `pyarrow` |
| #1516 | Índice de las predicciones calculado dos veces, recortes y comprobaciones repetidas en `predict`; `pd.concat` en el backtesting | Backtesting de 1000 folds: 393 → 283 ms en p2, 523 → 341 ms en p3 con `pyarrow` |
| #1519 | La validación de scikit-learn al aplicar un `StandardScaler` ya ajustado | Una llamada: 84 → 2,4 µs; backtesting de 1000 folds: 424 → 192 ms (Linux, p3) |
| #1530 | Copias, reindexados y selecciones redundantes en `predict` y backtesting multiserie | Backtesting por fold, `exog` ancha, 100 series: 38,0 → 8,1 ms en p2, 42,1 → 8,7 ms en p3 con `pyarrow` |

Según #1516, en los forecasters de una serie, p3 con esa PR ya es más rápido que p2 sin
ella. Lo que queda está en la fila PERF-predict-numpy del ROADMAP: su bloque A fue #1516;
el bloque B, pendiente, es que el backtesting de `ForecasterRecursive` y `ForecasterDirect`
no valide cada fold (5 a 7 ms por backtesting de 50 folds, medido el 2026-11-19).

## 8. Riesgos y temas abiertos

### 8.1 Antes de publicar la 0.28.0

Salen del ROADMAP, sobre todo de su sección 1:

- **CI completa.** No se ha relanzado desde `12898d190` (comprobado con `gh run list`). Lo
  fusionado después (#1503, #1510 a #1516, #1519 y #1530) no ha corrido con versiones
  mínimas, ni en Linux ni en macOS, ni con la celda `pandas<3`. La PR de la versión a `main`
  lanza `unit-tests.yml` y el job de mínimos (y `foundation-backends.yml` si cambia
  `skforecast/foundation/`), pero no latest-deps, y un fallo ahí se ve tarde.
- **REVIEW-post-1516.** El código fusionado en `0.28.x` después del 2026-11-16 (#1511 a
  #1519 y lo posterior, incluido #1530) no tiene una revisión independiente de conjunto;
  #1516, #1519 y #1530 tuvieron la suya. Falta decidir si se hace antes de la PR de la
  versión.
- **Pérdidas de pandas 3 sin confirmar.** Un informe externo del 2026-11-20 (evaluación de
  #1530 con 33 artículos de cienciadedatos.net, Linux, pandas 3.1.1 con `pyarrow`) vio dos,
  ninguna medida aquí:
  - las búsquedas de cinco artículos (LightGBM, XGBoost y HistGradientBoosting) tardaban
    del 10 al 20 % más con pandas 3, en una sola ronda. Con la variabilidad de LightGBM hace
    falta un A/B; queda en la fila MAINT-pandas3-final-crossenv;
  - con las entradas de un artículo (1578 series, 22 exógenas en diccionario, 3 de texto),
    `create_train_X_y` multiserie pasaba de 2,5 s con pandas 2 a 3,2 s con pandas 3, por
    tres costes: la copia profunda de la `exog` de cada serie en
    `check_preprocess_exog_multiseries`, el `Index` de texto que crean las window features
    en cada llamada y las conversiones del texto Arrow en el `OrdinalEncoder`. Son 0,7 s,
    el 1 % de un `fit` con LightGBM (PERF-ms-train-pandas3, por decidir).
- **MAINT-bench-pandas3.** `benchmarks/` sigue con `pandas==2.3.3` (D9). Para no mezclar el
  cambio de pandas con los de código, se hará con una medición puente: una tanda con pandas
  2.3.3; una PR que cambia solo ese pin y guarda la versión de `pyarrow`; una segunda tanda
  con pandas 3 sin `pyarrow`, como en la CI; y una tercera con `pyarrow`, que es lo que verán
  muchos usuarios. Un `benchmark.joblib` escrito con pandas 3 no se abre con pandas 2, así
  que el notebook de la FAQ que los lee tiene que pasar también a pandas 3.
- **MAINT-docs-pandas3.** Reejecutar los 5 notebooks cuyas salidas cambian con pandas 3:
  `[ns]` a `[us]` en tres y `object` a `str` en dos. Se hará cuando el entorno de la
  documentación use pandas 3.
- **REL-conda-forge-028.** Llevar al feedstock de conda-forge los mínimos y el techo nuevos.
- **REL-skforecast-ai-python311.** Pedir a `skforecast-ai` que suba su mínimo a Python 3.11
  y actualizar `docs/skforecast-ai/skforecast-ai.md`, que aún dice 3.10 (consecuencia de
  #1496).

### 8.2 Errores previos, independientes de pandas (issues abiertas)

Salieron en la revisión 5 y no los introdujo la migración:

- #1506: un estimador que escribe en su entrada da predicciones erróneas en
  `ForecasterDirect` (el equivalente de #1504, pero en el estimador en lugar del
  transformador).
- #1507: horas en lugar de días en las distancias de `calculate_distance_from_holiday` con
  frecuencias quincenales o de trimestre y año laborables.
- #1508: exógenas numéricas tratadas como categóricas si `transformer_exog` devuelve numpy y
  hay una columna de texto.
- #1509: dtypes booleanos distintos entre series en `reshape_exog_long_to_dict`.

### 8.3 Sin fecha

- **MAINT-drop-pandas2: retirar pandas 2.** Bloqueado por `gluonts` (backends de Moirai y
  TabPFN). Desaparecerían `PANDAS_GE_3`, las ramas de `concat` y `astype`, la pista y la
  nota condicionadas, los tests condicionados y la celda `pandas<3`. Se quedan las copias de
  #1502 y #1504 y la inferencia de frecuencia en UTC, que son el comportamiento correcto con
  pandas 3. La lista detallada está en P3 §10, "Al retirar pandas 2"; el ROADMAP señala dos
  erratas en ella (`compat.infer_freq` tiene 2 usos, no 3, y el orden de las frecuencias es
  `sorted(key=str)`, no por duración) y pide moverla a esa fila antes de borrar la carpeta.
- **MAINT-save-versions.** Guardar con el forecaster las versiones de pandas, numpy,
  scikit-learn y Python, y avisar al cargarlo solo si cambia una versión mayor (P3 §12).
- **PERF-predict-numpy, bloque B.** Validar una sola vez en el backtesting (sección 7.2).

### 8.4 Menores

- **El filtro de `Pandas4Warning`** (MAINT-latest-deps-pandas4): cuando salga pandas 4 habrá
  que cambiarlo, porque el job semanal lo instala pese al techo.
- **codecov** (MAINT-codecov-pandas2): mide con pandas 3, así que las ramas de pandas 2 de
  `_pandas_compat.py` saldrán sin cubrir.
- **El extra `test`** (MAINT-test-deps): `pandas[parquet]>=2.3` no tiene el techo `<4.0` del
  core.
- **Un test bayesiano inestable** (TEST-flaky-bayesian-threads): falló una vez en el job de
  mínimos, probablemente por una carrera entre los hilos de los trials.
- **Tres errores previos que solo ocurren con pandas 2** (BUG-pandas2-shared-memory):
  `X_train` e `y_train` comparten memoria; `X_train` comparte memoria con la salida de una
  window feature propia; y `filter_train_X_y_for_step` sin exógenas cambia el índice del
  `X_train` del usuario. Sin issue; desaparecen al retirar pandas 2.
- **Una hora de diferencia en pandas 2** entre las fechas que valida `check_predict_input` y
  las que predicen los forecasters multiserie, con índices diarios anclados en UTC cuando las
  predicciones cruzan un cambio de hora (anotado en #1516; ya estaba en `0.28.x`).
- **La nota de `load_forecaster` en la rama de skops** se contradice con pandas 2.3: dice que
  un fichero de pandas 3 se carga en pandas 2.3 con skops, y aparece justo cuando esa carga
  falla. Se deja hasta retirar pandas 2.

## 9. Lo que queda por decidir

Todo lo anterior tiene dueño y momento salvo esto, que es decisión de los mantenedores:

1. **REVIEW-post-1516**: si hace falta una revisión independiente del código fusionado
   después del 2026-11-16 antes de la PR de la versión, y de qué alcance.
2. **CI antes de la PR de la versión**: relanzar los cuatro workflows sobre `0.28.x` ahora,
   o esperar a la PR a `main` (que no lanza latest-deps).
3. **MAINT-test-deps**: si el extra `test` lleva `pandas[parquet]>=2.3, <4.0`.
4. **BUG-pandas2-shared-memory**: si se arreglan o se esperan a retirar pandas 2.
5. **MAINT-save-versions**: si se implementa y en qué versión.
6. **MAINT-drop-pandas2**: con qué criterio se retira pandas 2 cuando `gluonts` lo permita.
7. **PERF-ms-train-pandas3**: si entra en la 0.28.0, después de medirlo en las tres
   configuraciones (pandas 2, y pandas 3 con y sin `pyarrow`).

---

# Parte II. Anexos por PR

Un anexo por PR, en orden de fusión. Cada uno tiene la misma estructura: el problema, qué
cambió, decisiones y alternativas descartadas, cómo se verificó y qué quedó pendiente. Las
cifras salen de la descripción de la PR salvo que se cite otra fuente. Cuando se dice que
una PR "pasó el arnés", quiere decir identidad estricta de todas sus salidas frente a la
base, en pandas 2 y en pandas 3.

## Anexo II.1. En `0.27.x`, antes de empezar

### #1443: expectativas de dos tests de `backtesting_stats`

Fusionada el 2026-11-03 en `0.27.x` (`9d1a6579e`).

- **Problema.** Dos tests de `test_backtesting_stats.py`
  (`..._multiple_estimators_refit_{False,True}_with_mocked`) fallaban en Windows y Linux, con
  pandas 2 y con pandas 3. Cuando `38fd338e7` (#1437) añadió las columnas `n_evaluated`,
  `step` e `y_true` al backtesting, no se actualizaron sus expectativas. Tienen `skipif` en
  macOS, por eso nadie lo vio. Sin arreglarlos, la migración no podía tener la suite en verde
  (P3 §10, "Revisión del 2026-11-03").
- **Qué cambió.** Solo el test: añade las tres columnas a los valores esperados y reordena
  `estimator_id`. La PR no tiene descripción.
- **Pendiente.** Nada.

### #1444: el arnés de verificación

Fusionada el 2026-11-03 en `0.27.x` (`c49bfcd60`). Solo cambia `dev/`.

- **Problema.** El estudio había medido con un arnés no versionado. Cada PR de la migración
  tenía que demostrar que sus resultados no cambian y que el rendimiento se mantiene, en
  pandas 2 y en pandas 3, con un método repetible.
- **Qué cambió.** `dev/pandas3_migration/harness/`:
  - `identity_cases.py` ejecuta los casos de identidad y guarda dos huellas SHA-256 de cada
    salida: la estricta (valores, dtypes, índices y su `freq`) y la normalizada (solo valores).
    Comprueba además que las entradas no cambian, que dos `predict` seguidos coinciden, qué
    salidas comparten memoria con una entrada y qué avisos de deprecación nacen en
    `skforecast/`.
  - `bench_hotspots.py` mide tiempo y pico de `tracemalloc`.
  - `run_ab.py` exporta la base con `git archive` y alterna procesos de los dos árboles en
    cada entorno.
  - Al fusionarse tenía 30 casos, 205 salidas y 24 benchmarks (P3 §10); al final, 35 casos y
    31 benchmarks (`bdc9ae521:dev/pandas3_migration/harness/README.md`) y 258 salidas (#1505).
- **Decisiones.**
  - No hay huellas de referencia versionadas: son bit a bit y dependen de la plataforma, así
    que base y rama se calculan siempre en la misma máquina.
  - El A/B alterna procesos de dos copias del paquete; parchear el código dentro del mismo
    proceso, como hacía el estudio, había dado dos resultados falsos.
  - El RSS no se mide, porque en Windows no se puede reiniciar su pico entre benchmarks.
- **Verificación.** 205 de 205 salidas iguales entre dos ejecuciones y entre base y rama.
  Control nulo (la misma rama como base y como cabeza): razón de mínimos entre 0,94 y 1,09.
  Una diferencia dentro de esa banda no es significativa (P3 §10).
- **Pendiente.** Se borrará con `dev/pandas3_migration/`. El ROADMAP pide decidir antes si
  pasa a `benchmarks/` o a `tools/`, porque lo usan también otras tareas de rendimiento.

### #1445: `PopulationDriftDetector` con categorías sin usar, y chi2 determinista

Fusionada el 2026-11-03 en `0.27.x` (`5c2ff0258`). Errores de la 0.26 y 0.27 que salieron al
construir los casos del arnés (P3 §10).

- **Problema.**
  - Con una columna `category` que tiene categorías sin observaciones, `value_counts()`
    devuelve recuentos 0. Si una categoría no aparecía ni en la referencia ni en el bloque,
    la tabla de contingencia tenía una columna de ceros y `chi2_contingency` lanzaba
    `ValueError` en `fit` (también podía pasar en `predict`).
  - `predict` alineaba las categorías con un `set`, así que su orden dependía de
    `PYTHONHASHSEED` y los últimos bits de `chi2_statistic` cambiaban entre sesiones.
- **Qué cambió.**
  - Un helper privado quita las columnas a cero antes del estadístico. El resultado es el
    mismo que sin la categoría sin usar, y los atributos y la distancia de Jensen-Shannon no
    cambian.
  - Las categorías siguen un orden estable: primero las de referencia y después las nuevas,
    en el orden de `value_counts` (de más a menos frecuente). La descripción de la PR dice
    "por orden de aparición"; el código (`_population_drift.py`, `predict`) y P3 §10 dicen
    `value_counts`. Los resultados que no fallaban no cambian.
- **Verificación.** Tests nuevos en `test_fit.py` (con `threshold_method` `std` y `quantile`) y
  `test_predict.py`, con valores fijos y contra los mismos datos sin la categoría, bit a bit.
  Nota de versión en 0.27.0.
- **Pendiente.** Nada.

### #1448: `PopulationDriftDetector` con un nivel de series categórico

Fusionada el 2026-11-04 en `0.27.x` (`d3db96fa7`). Venía de la PR 1 del plan y se separó
para publicarla en la 0.27.0 (D11).

- **Problema.** Con varias series, si el nivel de series del `MultiIndex` es `category` con
  categorías sin usar, `groupby(level=0)` de pandas 2 las recorre como grupos vacíos: `fit`
  creaba un detector vacío por cada una (con avisos) y `predict` lanzaba
  `KeyError: 'ks_statistic'`. pandas 3 usa `observed=True` por defecto y no fallaba: era un
  error de pandas 2 que la migración destapó.
- **Qué cambió.** `groupby(level=0, observed=True)` en `fit` y `predict` del detector, y lo
  mismo en `FoundationModel._exog_to_dict`, que devolvía un `DataFrame` vacío en lugar de
  `None` para la categoría sin usar. Los resultados son los de un nivel de texto.
- **Verificación.** Tres tests nuevos, que fallan en `0.27.x` y pasan aquí. Los tests de
  `drift_detection` y `foundation` pasan con pandas 2.2.3, 2.3.3 y 3.1.0. Nota de versión
  en 0.27.0.
- **Pendiente.** Nada.

## Anexo II.2. En `0.28.x`: la migración

### #1495: renombrados y módulo de compatibilidad

Fusionada el 2026-11-12 (`e4cf42c19`). PR 1 y PR 2 del plan, juntas porque ninguna es
visible para el usuario. Sin nota de versión.

- **Problema.** El paquete emitía 1748 `Pandas4Warning` desde 30 sitios con pandas 3.1.0, por
  APIs que pandas 3 depreca y por el argumento `copy` de `concat` y `astype`.
- **Qué cambió.**
  - Renombrados:
    - `dayofweek` y `dayofyear` pasan a `day_of_week` y `day_of_year`;
    - `isinstance(x, pd.DateOffset)` pasa a `isinstance(x, pd.offsets.BaseOffset)`, igual en
      pandas 2;
    - `freq='d'` pasa a `'D'` en un ejemplo;
    - el relleno de festivos de `calculate_distance_from_holiday` se hace en numpy, sin la
      opción `future.no_silent_downcasting`.
  - Módulo `_pandas_compat` (sección 4.1): `concat`, `astype(changed_only)` e `infer_freq`.
  - `_population_drift.py`: `astype("float64", copy=False)` pasa a
    `to_numpy(dtype="float64")`.
- **Decisiones.** D3 (a). `changed_only=True` solo en los `create_train_X_y` y en
  `reshape_exog_long_to_dict`: convierte solo las columnas que cambian, como hace pandas por
  dentro, y evita que pandas 3 parta la matriz en un bloque por columna. Los
  `create_predict_X` pasan el diccionario completo, sin recorrer los dtypes.
- **Verificación.**
  - Con pandas 3.1.0, los mismos 94 fallos en la rama y en la base, y 0 `Pandas4Warning` del
    paquete en la rama.
  - Pasó el arnés: 205 salidas.
  - Tiempo, rama / base (mediana): de 0,85 a 1,035 con pandas 2.3.3 y de 0,86 a 1,036 con
    pandas 3.1.0. La ganancia viene de
    no llamar a `astype` cuando nada cambia.
- **Pendiente.** Los 94 fallos, para #1497 y #1499; Python 3.11 se separó a #1496.

### #1496: Python 3.11 como mínimo

Fusionada el 2026-11-12 (`4016ede06`).

- **Problema.** Python 3.10 perdió soporte en octubre de 2026 (anunciado en las notas de la
  0.27.0), y pandas 3 exige 3.11.
- **Qué cambió.**
  - `requires-python = ">=3.11"` y la CI empieza en 3.11 (`unit-tests.yml`, el job de mínimos
    y la prueba de humo de `publish-pypi.yml`).
  - statsmodels pasa de 0.13.2 a 0.13.3, el primero con ruedas para 3.11; #1500 lo subió
    después a 0.14.6.
  - Se quita `tomli` del extra `test` (se usa `tomllib`), y la nota de Python 3.10 de
    `T0Adapter`.
  - Con Python 3.10, pip sigue instalando la 0.27.0.
- **Verificación.** Tests de foundation y utils (2032 pasan) y regeneración del contexto de IA.
  La suite completa y los mínimos no se ejecutaron en local.
- **Pendiente.** El feedstock de conda-forge (REL-conda-forge-028).
  `docs/skforecast-ai/skforecast-ai.md` sigue diciendo 3.10 a propósito: describe otro
  paquete.

### #1497: frecuencias y mensajes de error robustos a pandas 3

Fusionada el 2026-11-13 (`e1c928b9f`). Es la PR con más cambios de comportamiento.

- **Problema.** En pandas 3, `tz_convert` descarta las frecuencias de calendario, `asfreq` sin
  frecuencia asume diaria y el texto es `str` (pero `object` si todos los valores faltan).
- **Qué cambió.**
  - `reshape_series_wide_to_long`: infiere la frecuencia de las fechas ordenadas y lanza
    `ValueError` si no puede, en lugar de asumir diaria. Ya no renombra el índice de entrada.
  - `TimeSeriesSplitter` (salidas `'long'`): cada partición toma la frecuencia de sus series.
  - `expand_index`: `_infer_freq_from_index`, que infiere en UTC si el índice avanza en pasos
    fijos de UTC.
  - Pista de pandas 3 en los errores de índice sin frecuencia (`_TZ_MISSING_FREQ_HINT`), y
    `sorted(key=str)` en "Found frequencies".
  - `check_preprocess_exog_multiseries` compara como uno los dtypes de texto (`str`,
    `object`, `string`).
  - Se quita un `fillna("_unknown_level")` de `_train_test_split_one_step_ahead`, que no
    rellenaba nada y daba un `FutureWarning` con `onehot` y nombres enteros.
  - `backtesting_foundation`: un fold vacío usa `dtype=str` en `level`.
  - Nota en `load_forecaster` con pandas 2 (`_add_note_pandas_3_file`).
- **Decisiones.**
  - D6: pista, sin inferir la frecuencia.
  - D8: nota en `load_forecaster`, sin cambiar el tipo del error.
  - D10: orden por texto (más simple que por duración y con el mismo texto en las dos
    versiones).
  - D15: inferir la frecuencia, en lugar de un `ValueError` general que rompía el diario sin
    `freq`.
- **Cambios también con pandas 2**, con nota de versión:
  - sin `freq`, `reshape_series_wide_to_long` lanza `ValueError` con datos diarios con huecos o
    con menos de 3 filas (antes funcionaba porque asumía diaria), y los días laborables pasan
    a `B` en lugar de filas diarias con NaN los fines de semana;
  - una exógena `string` en una serie y `object` en otra ya no da `TypeError`;
  - `TimeSeriesFold.split` y `OneStepAheadFold.split` dan el `ValueError` con la lista de
    frecuencias en lugar de un `TypeError`, y el orden de esa lista cambia.
- **Verificación.**
  - Suite con pandas 3.1.0: de 94 fallos a 88, ninguno nuevo.
  - Ficheros reales guardados con pandas 3.1.0 y cargados en 2.3.3 y 2.2.3: joblib y pickle
    fallan con la nota; skops carga en 2.3.3.
  - `expand_index` sin frecuencia, 1,11 veces el tiempo (0,08 ms por llamada). La inferencia
    nunca se ejecuta en `fit`, `predict` ni el backtesting, porque los forecasters exigen
    `freq`.
- **Pendiente.** Los fallos de los tests, para #1499. La suite completa no se repitió tras los
  dos últimos cambios.

### #1498: `pyarrow<26` en el job de mínimos

Fusionada el 2026-11-13 (`5eed5d875`). Solo CI.

- **Problema.** Con Python 3.11 como mínimo (#1496), uv resolvía `pyarrow` 26, que necesita
  numpy 2 pero no lo declara, junto al numpy mínimo de entonces (1.26.1). La recolección de
  los tests que leen `.parquet` fallaba.
- **Qué cambió.** `unit-tests-min-deps.yml` pasa `pyarrow<26` como restricción
  (`--constraints`). Como requisito, `--resolution lowest-direct` habría instalado
  `pyarrow` 10.0.1.
- **Decisiones.** Subir numpy a 2.0 aquí se descartó, porque obligaba a subir pandas: era
  cambiar los mínimos del usuario para arreglar un entorno de test. Se hizo después, por sus
  propios motivos, en #1500.
- **Verificación.** El job, lanzado en la rama, pasó en verde (8107 tests).
- **Pendiente.** Quitar la restricción con numpy 2.0 como mínimo: lo hizo #1500.

### #1499: tests independientes de la versión

Fusionada el 2026-11-13 (`541786939`). Solo tests.

- **Problema.** Con pandas 3.1.0 fallaban 85 tests cuyas expectativas estaban escritas para
  pandas 2, y los propios tests emitían 20 `Pandas4Warning`.
- **Qué cambió, por causa.**
  - Resolución: los fixtures multiserie, guardados con pandas 2, siguen en `[ns]`, así que las
    expectativas de esos tests se crean en `[ns]`.
  - Texto: `STR_DTYPE = pd.Series(['a']).dtype` en lugar de `object`.
  - Nombres de clase en los mensajes: se construyen desde la clase (`f"{pd.Series}"`).
  - Zonas horarias: el indicador `tz_zoneinfo` de skops se deriva de lo que construye pandas.
  - Índices creados en UTC y convertidos: `'D'` solo con pandas 2 (7 casos se omiten con
    pandas 3) y `'24h'` en las dos.
  - Avisos de los tests: comparaciones de índices sin `freq`, `exact='equiv'` explícito y
    listas en lugar de tuplas.
- **Verificación.** Suite completa con pandas 3.1.0 y `-W error::pandas.errors.Pandas4Warning`:
  ningún fallo relacionado con pandas. CI en la rama, en verde: mínimos (8114) y
  latest-deps (8107; antes, 85 fallos por versión de Python).
- **Pendiente.** Activar el filtro de avisos en CI (#1501).

### #1500: numpy 2.0 y pandas 2.3 como mínimos

Fusionada el 2026-11-13 (`2256bf287`). Decisión D2 (c), tomada ese día.

- **Problema.** numpy 1.x ya no recibe arreglos y no tiene ruedas para Python 3.13 y 3.14.
  Con numpy 1.x como mínimo, `pandas<4.0` permitía pandas 3.0 con numpy 1.26, que ningún job
  probaba.
- **Qué cambió.**
  - Mínimos: numpy 2.0, pandas 2.3 (2.2.0 y 2.2.1 no funcionan con numpy 2), scipy 1.13,
    numba 0.60, statsmodels 0.14.6 (el primero que se importa con pandas 3.1) y matplotlib
    3.8.4. En el extra `test`: catboost 1.2.8, lightgbm 4.4 y `pandas[parquet]` 2.3.
  - Se quita la restricción de `pyarrow` de #1498 y restos de numpy 1.x (el respaldo de
    `np.trapz` en `crps_from_quantiles`).
  - Moirai: `uni2ts` declara numpy 1.26 pero funciona con numpy 2 (cuantiles bit a bit
    iguales con 1.26.4, 2.0.2 y 2.5.4). Se documenta su instalación en dos pasos y el
    `ImportError` del adaptador la explica.
- **Para el usuario.** Instalar la 0.28.0 sube numpy y pandas si son anteriores (pip no vuelve
  a la 0.27.0, como sí hace con la versión de Python); para seguir con numpy 1.x o pandas 2.2,
  `skforecast<0.28`.
- **Decisiones.** Se descartó retirar el adaptador de Moirai. La subida va antes del pin
  (#1501) porque cambia lo que prueba su matriz de CI.
- **Verificación.**
  - Mínimos en local (numpy 2.0.0, pandas 2.3.0, scikit-learn 1.6.0...): los 78 ficheros de
    test que usan lightgbm, xgboost o catboost pasan.
  - Moirai: contrato y prueba de humo con los pesos reales.
  - Los otros diez backends resuelven las mismas versiones.
  - Un forecaster guardado con numpy 1.26.4 carga con 2.5.4 y al revés.
- **Pendiente.** El feedstock de conda-forge. Un test bayesiano falló una vez en el job de
  mínimos (TEST-flaky-bayesian-threads; P3 §10, "DEP-numpy2-pandas23, el 2026-11-13").

### #1501: el pin `pandas<4.0`

Fusionada el 2026-11-14 (`046e511cd`). No cambia ningún fichero de `skforecast/`. Desde aquí
los usuarios pueden instalar pandas 3.

- **Qué cambió.**
  - `pyproject.toml`: `pandas>=2.3, <4.0`.
  - `unit-tests.yml`: las 12 celdas instalan pandas 3, y una nueva, `pandas<3` (Python 3.12,
    Linux), prueba pandas 2 con el resto de dependencias al día. La clave `pandas` tiene que
    estar en la matriz: un `include` con una clave nueva se fusionaría con la celda 3.12/Linux
    existente en lugar de añadir otra.
  - `unit-tests-latest-deps.yml`: el filtro de `Pandas4Warning` (D5, sección 5.2).
  - Guía de instalación, contexto de IA y notas de versión.
- **Para el usuario.** `pip install -U skforecast` conserva el pandas instalado; un entorno
  nuevo recibe pandas 3.1.1. Instalar los backends de `MoiraiAdapter` o `TabPFNAdapter`
  (`gluonts`) baja pandas a 2.3.3.
- **Verificación.**
  - Suite completa en p2 y p3 (con el filtro), solo con los 3 fallos de TensorFlow.
  - Control del filtro con un paquete temporal: falla con un aviso nacido en `skforecast.*`,
    en un test o en una deprecación real de pandas, y no con uno de otro paquete.
  - CI lanzada a mano, toda en verde: las 12 celdas, `pandas<3`, mínimos (pandas 2.3.0),
    latest-deps y los 11 backends foundation (9 con pandas 3.1.1; Moirai y TabPFN con 2.3.3).
  - Comparación p3 / p2: la columna "En #1501" de la sección 7.1.
- **Pendiente.**
  - Quitar la copia de las matrices, que pasaba a verla el usuario (#1502, "tiene que entrar
    antes de la 0.28.0").
  - codecov mide con pandas 3; `benchmarks/` sigue en pandas 2.3.3 (D9); falta el feedstock.
  - El filtro, cuando salga pandas 4.

### #1502: copias que sobran con pandas 3

Fusionada el 2026-11-15 (`3db7f86f4`). Mitad de rendimiento de la PR 3 del plan. Sin nota de
versión: con pandas 2 no cambia nada.

- **Problema.** Con pandas 3, el `create_train_X_y` público tardaba de 1,33 a 3,01 veces lo de
  pandas 2 (#1501), por la copia que hace el constructor de `DataFrame` de un array de numpy.
- **Qué cambió.**
  - En `create_train_X_y` de los cuatro forecasters con matriz numpy:
    `copy = False if X_train.flags.writeable else None`. `fit` no pasa por ese constructor: la
    ganancia es solo del `create_train_X_y` público.
  - En 9 sitios, `exog.copy()` pasa a `exog.copy(deep=False)` antes de codificar las
    categóricas (sección 4.4).
- **Decisiones.** D13, con el cambio que provocó la revisión independiente (sección 5.2). La
  primera versión pasaba `copy=False` siempre.
- **Verificación.**
  - Pasó el arnés: 257 salidas, sin pares nuevos que compartan memoria.
  - pandas 3, rama / base: el `create_train_X_y` público baja a 0,36 a 0,77 del tiempo y a
    0,57 a 0,84 del pico de memoria, salvo con solo lags (copia deliberada).
  - pandas 2: memoria idéntica, tiempo de 0,94 a 1,02.
  - Mutaciones: sin la copia superficial fallan los 10 tests nuevos de categóricas; con
    `copy=False` siempre, los 7 casos de matriz escribible.
- **Pendiente.**
  - La suite completa no se repitió tras el cambio de la revisión; sí las carpetas de los
    cinco forecasters.
  - Dos hallazgos de la revisión pasaron a #1504.

### #1504: los transformadores ya no modifican los datos del usuario

Fusionada el 2026-11-15 (`658bf68b7`). Segunda mitad de la PR 3 del plan. El error más serio
de todos los que salieron en la migración.

- **Problema.** `transform_numpy`, `transform_series` y `transform_dataframe` pasaban al
  transformador los datos de quien llamaba, que a menudo son los del usuario. Un transformador
  que escribe en su entrada (un escalador con `copy=False`, un `Pipeline` que lo contiene, un
  `FunctionTransformer` in situ; con pandas 3, también un `ColumnTransformer`) o uno que la
  devuelve tal cual (`FunctionTransformer()` identidad) producía:
  - entradas del usuario modificadas en `fit`, `predict` y `set_out_sample_residuals`, en
    siete forecasters;
  - métricas erróneas sin aviso: backtesting con refit, MAE 37,1 en lugar de 0,446;
    búsqueda one-step-ahead con diferenciación, 6,68 en lugar de 0,245; tres candidatos
    idénticos con MAE 0,89, 116,9 y 411,3;
  - errores: búsquedas que descartaban todos los candidatos, y un `TypeError` en el segundo
    `predict` con categóricas.

  Copy-on-Write no lo evita: scikit-learn vuelve escribible el array de solo lectura antes de
  escribir.
- **Qué cambió.**
  - Las tres funciones copian su entrada cuando hay transformador.
  - Dos llamadas directas a `inverse_transform` en `model_selection/_utils.py` pasan a
    `transform_numpy` (las encontró la revisión independiente).
- **Decisiones.** D14, copiar siempre. Alternativas descartadas:
  - copiar si `transformer.copy` no es `True`: copia igual con `Pipeline` o
    `FunctionTransformer` y solo ahorra la copia barata de los escaladores por defecto;
  - copiar si `copy is False`: no detecta `Pipeline` ni `FunctionTransformer`;
  - copiar en cada llamador: 56 llamadas en 8 ficheros;
  - detectar la modificación y lanzar un error: rompería código que funciona.
- **Verificación.**
  - Arnés, base frente a rama: 252 de 257 salidas iguales en p2 y 254 de 257 en p3. Las que
    difieren son las de los dos casos que arregla.
  - Batería aparte de 110 salidas: en pandas 2 cambian 42 en el orden del redondeo (hasta
    2,6e-13 relativo), porque la copia deja los datos contiguos por columna, como ya los crea
    pandas 3. Con la rama, p2 y p3 dan lo mismo bit a bit.
  - 42 casos de test nuevos; sin el arreglo fallan 35 de los 67 casos nuevos o modificados del
    primer commit.
  - Coste de la copia: 7,3 ms en p2 y 4,2 ms en p3 para 200 000 × 10; de 10 a 16 µs por
    `predict`; `predict_loop_scaler` 1,037 veces en pandas 2.
- **Pendiente.**
  - Copias de objetos que el forecaster ya posee: medidas y aceptadas, porque evitarlas pide
    un parámetro público.
  - El equivalente para el estimador: #1506.

### #1505: festivos con zona horaria y dos problemas de pandas 3 en `predict`

Fusionada el 2026-11-16 (`12898d190`). Arreglos de las dos revisiones de toda la rama. Sobre
su commit de fusión se lanzaron los cuatro workflows, en verde (R28); no se han vuelto a
lanzar desde entonces (`gh run list`, sección 8.1).

- **Qué cambió.**
  1. `calculate_distance_from_holiday` con un índice en pasos fijos de UTC daba horas en lugar
     de días: siempre con pandas 3, con pandas 2 solo sin frecuencia. Ahora usa
     `_infer_freq_from_index`.
  2. Con un índice en hora local, perdía un día al cruzar el cambio de hora de primavera (33 en
     lugar de 34), en las dos versiones y en la 0.27.0. Ahora cuenta días entre fechas
     locales cuando la unidad es el día y el índice no avanza en pasos de UTC.
  3. Lo mismo con `date_column`. Además, las fechas con zona ya no pasan por un array de
     objetos `Timestamp`: con un índice diario en hora local de 18 000 filas, de 105 a 6,6 ms
     en p2 y de 156 a 3,9 ms en p3.
  4. Con pandas 3, un hueco en `exog` no se detectaba en `predict` con índices en pasos de UTC,
     porque `None == None` aceptaba las dos frecuencias. Ahora `_check_exog_alignment` exige
     que `exog` tenga frecuencia para tomar el atajo.
  5. Con pandas 3, `ForecasterDirectMultiVariate.predict` pasaba a las window features un
     array de solo lectura. Ahora pasa una copia, como `ForecasterDirect`.
  6. Tests que no fijaban dos condiciones, un docstring y dos comentarios.
  7. Efecto colateral de 1: un índice sin frecuencia y con menos de tres fechas ya no lanza el
     `ValueError` de pandas ("Need at least 3 dates"); avisa y devuelve horas, como dice el
     docstring. Los puntos 1 a 3 y este van en la nota de versión (Fixed); 4 y 5 no, porque
     solo ocurren con pandas 3, que ninguna versión publicada admite.
- **Decisiones.** Quitar la zona en todos los casos, el arreglo más simple, rompía las fechas
  en pasos de UTC del cambio de otoño, que eran correctas.
- **Verificación.**
  - 1260 índices contra la distancia en días de calendario: erróneos de 281 a 12 en p2 y de
    392 a 24 en p3, sin que cambie ninguno que era correcto. Los 12 que quedan no tienen
    frecuencia inferible y avisan, como antes.
  - Pasó el arnés: 258 salidas. p2 frente a p3: 125 difieren solo en el dtype.
  - Suite completa en p2 y p3.
- **Pendiente.** Los hallazgos ajenos a pandas pasaron a las issues #1506 a #1509; los de
  documentación, a #1510. La PR midió que `changed_only=True` haría más rápido
  `create_predict_X` (10 exógenas float: de 1,97 a 0,56 ms en p2 y de 2,73 a 0,76 ms en p3),
  pero dejó el parámetro como estaba; quitó la frase errónea del docstring de
  `_pandas_compat.astype`.

### #1510: documentación y el error de `chunk_size`

Fusionada el 2026-11-16 (`c18f6705c`). PR 7 del plan.

- **Qué cambió.**
  - Página nueva de la FAQ, `docs/faq/pandas-2-and-pandas-3.md`: texto, fechas, zonas
    horarias, `ForecasterEquivalentDate`, arrays de solo lectura y persistencia.
  - Frases con enlace en cinco guías, y `day_of_week` en tres notebooks (solo celdas
    Markdown y código; las salidas guardadas siguen valiendo).
  - Skills (`troubleshooting-common-errors`, `forecasting-single-series`,
    `baseline-forecasting`, `drift-detection`) y `llms-base.txt`.
  - Notas de versión: el titular y la entrada de #1501 ya no prometen "las mismas
    predicciones" sin matices, sino que remiten a la página; dos viñetas nuevas en "Before
    upgrading" y dos entradas en Changed.
  - Las dos excepciones a "mismos resultados" que documenta la página:
    `ForecasterEquivalentDate(offset=pd.offsets.Day(n))` con datos horarios con zona (en el
    ejemplo, `[336, 337, 338, 339]` con pandas 2 y `[337, 338, 339, 340]` o un `ValueError`
    con pandas 3; `offset=24 * n` da lo mismo en las dos), y `None` en columnas de texto
    (categoría con pandas 2, `NaN` con pandas 3: con `Ridge`, pandas 3 falla).
  - `PopulationDriftDetector`: con un `chunk_size` que pandas no acepta, el error decía
    `Got <class 'str'>`. Ahora muestra el valor y el error de pandas, que con pandas 3 y `'M'`
    nombra `'ME'`.
- **Decisiones.**
  - No se documenta `predict(check_inputs=False)` para acelerar bucles: es de uso interno.
  - No se guarda en skforecast una tabla de alias de pandas.
- **Verificación.** Cada afirmación de la página se ejecutó con pandas 2.3.3 y 3.1.1. Tests
  del detector en p2 y p3; `execute_notebooks.py --check` en los 8 notebooks cambiados; anclas
  de la página contra todos los enlaces.
- **Pendiente.** Reejecutar los notebooks cuyas salidas cambian con pandas 3
  (MAINT-docs-pandas3). No se compiló la documentación con `mkdocs`.

## Anexo II.3. Relacionadas: rendimiento

### #1512: series y exógenas en formato largo, sin `groupby`

Fusionada el 2026-11-16 (`c65284868`). Fila PERF-long-to-dict.

- **Problema.** `check_preprocess_series`, `check_preprocess_exog_multiseries` y
  `_create_predict_inputs` copiaban todo el `DataFrame` y hacían `groupby(level=0)` y
  `droplevel(0)` por serie: de 1 a 2 s fijos por `fit`, backtesting o búsqueda con miles de
  series, más con `pyarrow`.
- **Qué cambió.**
  - Helper privado `_split_long_format`: trabaja con los códigos enteros del `MultiIndex`,
    ordena una vez y corta cada serie, reproduciendo lo que devolvía `groupby`.
  - Con `series_ids`, solo convierte las series pedidas: predecir una serie con una `exog`
    larga de 5000 pasa de 368 a 4 ms en p2.
  - Los IDs de una `exog` en diccionario se buscan en un `set` (la lista crecía con el cuadrado
    del número de series).
  - Solo cambia el comportamiento con entradas inválidas o marginales:
    - un `MultiIndex` de más de dos niveles da un `ValueError` claro, en lugar de un error
      sin relación;
    - `series` en formato largo con el nombre de la primera columna duplicado usa la primera
      columna, como dice el aviso, en lugar de lanzar `TypeError: 'str' object is not
      callable`;
    - en `predict`, el índice de cada `exog` larga queda sin nombre, como ya pasaba en `fit`;
      las predicciones no cambian.
- **Verificación.**
  - 123 entradas raras, el helper frente a `groupby` en 18 y 10 salidas de extremo a extremo,
    en p2, en p3 con el almacenamiento de texto `python` y en p3 con `pyarrow`: iguales,
    salvo los casos inválidos de arriba.
  - Pasó el arnés: 258 salidas.
  - Cifras de la sección 7.2.
- **Pendiente.** PERF-reshape-long-to-dict: el mismo patrón en `reshape_series_long_to_dict` y
  `reshape_exog_long_to_dict`.

### #1516: `predict` en bucle, sin trabajo repetido

Fusionada el 2026-11-17 (`267784c8b`). Bloque A de PERF-predict-numpy.

- **Problema.** En un bucle de `predict` con un modelo ligero, casi todo el tiempo es pandas.
  Con pandas 3, 1000 folds de un paso tardaban de 1,13 a 1,16 veces lo de pandas 2 sin
  `pyarrow` y 1,30 veces con él.
- **Qué cambió.**
  - `check_predict_input` devuelve el índice de los pasos que ya construía (`None` sin `exog`),
    y los forecasters lo reutilizan en lugar de llamar otra vez a `expand_index`.
  - En `_create_predict_inputs` de `ForecasterRecursive`, `ForecasterRecursiveClassifier` y
    `ForecasterDirect`:
    - no recorta `last_window` si ya tiene `window_size` filas;
    - compara los dtypes como listas;
    - se salta `check_exog` cuando los valores son enteros, booleanos o float sin NaN.
  - `ForecasterDirectMultiVariate` recibe la reutilización del índice y los dos cambios de
    `exog`; `ForecasterRecursiveMultiSeries`, solo la reutilización del índice.
  - `backtesting_forecaster` une las predicciones puntuales concatenando valores e índices en
    lugar de `pd.concat`.
  - En los dos forecasters multiserie, el índice solo se reutiliza si `last_window` tiene
    exactamente `window_size` filas. Así se conserva la salida en un caso de pandas 2 en que
    validación y predicción difieren una hora.
- **Verificación.**
  - 2911 casos (predicciones, índices, avisos y errores) iguales en p2, p3 con y sin
    `pyarrow`.
  - La revisión independiente comparó 30 342 escenarios en p2 y 24 940 en p3: iguales, salvo
    dos diferencias marginales documentadas (un aviso duplicado con una opción retirada de
    pandas 2, y el tipo de error con `check_inputs=False` sin ajustar).
  - Pasó el arnés (258 salidas) antes de los últimos cambios de la revisión; el código final
    se comprobó con la batería de 2911 casos y las carpetas de tests de los módulos tocados.
    La suite completa no se repitió tras el rebase.
  - Un `predict` con exógenas en pandas 2, de un 30 % a un 4 % más rápido según el
    forecaster.
- **Pendiente.**
  - Bloque B: validar solo el primer fold. Es la mayor ganancia restante, pero un error ahí
    sería silencioso.
  - La diferencia de una hora en pandas 2 (sección 8.4).
  - PERF-multiseries-names-pyarrow, que dio #1530.

### #1519: `StandardScaler` sin la validación de scikit-learn

Fusionada el 2026-11-18 (`3c0711b8a`). Fila PERF-transform-numpy-scalers. No depende de
pandas, pero ataca el mismo coste fijo de `predict`.

- **Problema.** Con los arrays cortos de un forecaster, casi todo el tiempo de `transform` e
  `inverse_transform` es validación de scikit-learn: de 58 a 92 µs por llamada, con menos de
  3 µs de aritmética. En `ForecasterRecursiveMultiSeries` con 300 series, el 30 % de
  `predict`.
- **Qué cambió.** `_transform_numpy_fast_path`, llamado desde `transform_numpy` con
  `fit=False`.
  Aplica las operaciones de `StandardScaler` en el mismo orden, sobre una copia, solo si se
  cumple todo:
  - el escalador es exactamente `StandardScaler`, ajustado con una columna y con atributos
    float64;
  - el array es exactamente un `numpy.ndarray` float64, no vacío, sin infinitos, y 1D (o 2D
    solo en la inversa).

  En cualquier otro caso sigue el camino de scikit-learn, con sus mismos errores.
- **Decisiones.** float32 queda fuera, porque scikit-learn cambió su aritmética en 1.8.
  `MinMaxScaler`, `RobustScaler` y `Pipeline` quedan para otra PR.
- **Verificación.**
  - Igual en bytes que el camino anterior en 1176 casos de contrato y 14 070 de fuzzing, con
    scikit-learn 1.6.0 y 1.9.1; con 1.9.1 y pandas 3, 20 casos difieren solo en el indicador
    de solo lectura (ver "Pendiente").
  - 17 mutaciones detectadas por los tests.
  - 45 forecasters guardados con la 0.27.0 y la 0.28.0 dan las mismas predicciones.
  - Medido en Linux con pandas 3: un `predict` de 300 series, de 162 a 114 ms.
- **Pendiente.**
  - Un test avisa si una versión futura de scikit-learn cambia la aritmética; 13 casos
    dependen del texto de sus errores.
  - La suite completa no se ejecutó.
  - Diferencia conocida: con `transform_output="pandas"` y pandas 3, scikit-learn devuelve un
    array de solo lectura y el camino rápido uno escribible.

### #1530: `predict` y backtesting multiserie

Fusionada el 2026-11-20 (`a989a27ba`). Fila PERF-multiseries-names-pyarrow; medidas en
MSPERF.

- **Problema.** Con pandas 3, y más con `pyarrow`, cada objeto de pandas creado a partir de
  una lista de nombres cuesta más, y el `predict` multiserie creaba varios por serie y por
  llamada.
- **Qué cambió.** Pasos que solo devolvían los mismos datos:
  - reutilizar el conjunto de entrenamiento de folds con los mismos límites;
  - no seleccionar ni copiar `last_window` si sus columnas ya son los niveles;
  - no reindexar la `exog` de cada serie si ya está alineada;
  - comprobar una sola vez una `exog` compartida por varias series;
  - `np.tile` para una `exog` ancha;
  - cortar la `exog` larga desde un índice por posición;
  - no seleccionar columnas en `check_predict_input` cuando son exactamente las de la ventana.

  Cada cambio conserva el camino anterior en los casos que no cubre. También acelera
  `ForecasterDirectMultiVariate`: `predict` del 6 al 23 % y el backtesting por fold con
  `exog` ancha del 44 al 86 %, según el número de series.
- **Verificación.**
  - Cinco baterías (hasta 8106 casos) y sondas de etiquetas (hasta 27 000 combinaciones) en
    p2 y p3 con y sin `pyarrow`: 0 diferencias.
  - Pasó el arnés: 258 salidas.
  - Suite completa: 8499 tests pasan en p2 y 8492 en p3.
  - Backtesting por fold con 100 series y `exog` ancha: de 38,0 a 8,1 ms en p2.
  - `predict_bootstrapping` salió un 18 % más lento en el benchmark oficial en los dos
    entornos. Solo llama una vez a `_create_predict_inputs`, que es más rápido; medido en 24
    procesos aislados, cada proceso cae en un modo rápido o lento con el mismo código, y las
    medias de los dos árboles coinciden (4,27 frente a 4,17 s en p2).
- **Pendiente.**
  - Dos errores previos que se dejaron igual para no cambiar resultados: `levels=[]` con
    `last_window` y `exog` da un error de pandas, y una `exog` que mezcla una columna float y
    una anulable (`Int64`, `Float64`, `boolean`) con NA en los pasos da un `TypeError`.
  - Ni mínimos ni Linux ni macOS: los cubre la CI de la PR de la versión.

---

# Apéndice A. Medición final de pandas 3 frente a pandas 2

Medida del 2026-11-20 para la sección 7 (fila MAINT-pandas3-final-crossenv del ROADMAP).
Los resultados en bruto no se versionan; esta tabla es la copia que se conserva.

- **Árbol**: `bdc9ae521` (la `0.28.x` con #1530), el mismo en los tres entornos, por
  `PYTHONPATH`.
- **Entornos**: Windows 11, Python 3.13.18, numpy 2.5.5, scikit-learn 1.9.1, scipy 1.18.1,
  numba 0.68.0, lightgbm 4.7.0, pyarrow 26.0.1. Dos entornos conda que solo difieren en
  pandas (2.3.3 y 3.1.1; antes de medir se igualaron numpy, pyarrow, optuna, pytz y tzdata,
  que difería en versiones de parche). "Sin pa": el entorno de pandas 3.1.1 con la
  importación de `pyarrow` bloqueada (un buscador en `sys.meta_path`), de modo que el texto
  usa el almacenamiento `python`, como en un `pip install skforecast` limpio.
- **Protocolo**: `bench_hotspots.py --bench all` (31 benchmarks), un proceso por entorno y
  ronda, 4 rondas con el orden de los entornos invertido en cada una, 2 llamadas de
  calentamiento y 5 medidas por proceso (20 por entorno). Todos los procesos fijados al
  procesador lógico 12 con prioridad alta, y BLAS y OpenMP con un hilo. Razón de los
  mínimos; "Por ronda" es el intervalo de las razones de los mínimos de cada ronda (p3 con
  `pyarrow`). Memoria: pico de `tracemalloc` por encima de lo que había antes de la llamada,
  mínimo de 2, en un proceso por entorno.
- **Condiciones**: CPU de toda la máquina antes de cada benchmark, mediana 3 % y máximo 20 %,
  con el portátil enchufado. La ronda 1 de "sin pa" coincidió con carga (CPU del 12 al 20 %
  antes de varios benchmarks): sus mínimos salen hasta 1,67 veces los de las otras tres
  rondas, que coinciden entre sí; el mínimo de 20 no la usa.
- **Columna #1501**: la medida de entonces (pandas 3.1.1 con `pyarrow` frente a 2.3.3,
  numpy 2.5.4, sin fijar núcleo ni hilos, sobre el árbol de #1501); "nuevo" son los 7
  benchmarks que añadió #1502.

| Benchmark | p2 (ms) | p3 con pa (ms) | p3 sin pa (ms) | p3 con pa / p2 | Por ronda | p3 sin pa / p2 | p2 (MiB) | p3 con pa (MiB) | p3 sin pa (MiB) | #1501, p3 / p2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `ms_create_train_X_y_block` | 57,4 | 65,1 | 59,4 | 1,14 | 1,13 a 1,22 | 1,04 | 71,16 | 69,79 | 69,78 | 1,14 |
| `ms_create_train_X_y_concat` | 39,1 | 38,6 | 36,9 | 0,99 | 0,94 a 1,01 | 0,95 | 88,07 | 73,10 | 73,09 | 0,94 |
| `ms_create_train_X_y_onehot_dict` | 53,4 | 52,9 | 51,1 | 0,99 | 0,97 a 1,02 | 0,96 | 67,00 | 67,04 | 67,03 | 1,01 |
| `ms_fit_linear` | 225,7 | 237,9 | 227,1 | 1,05 | 0,95 a 1,14 | 1,01 | 140,54 | 140,60 | 140,59 | 1,02 |
| `ms_fit_linear_encoding_none` | 217,7 | 215,3 | 210,5 | 0,99 | 0,82 a 1,16 | 0,97 | 181,26 | 137,62 | 137,61 | 1,00 |
| `ms_get_level_row_slices` | 0,3 | 0,4 | 0,3 | 1,37 | 1,32 a 1,37 | 1,23 | 0,19 | 0,19 | 0,19 | 1,34 |
| `ms_predict_levels` | 3,3 | 4,5 | 3,4 | 1,35 | 1,23 a 1,41 | 1,03 | 0,36 | 0,32 | 0,34 | 1,06 |
| `recursive_create_train_X_y` | 46,4 | 53,3 | 47,3 | 1,15 | 1,13 a 1,17 | 1,02 | 59,13 | 59,14 | 59,14 | 1,85 |
| `recursive_create_train_X_y_float` | 15,4 | 16,0 | 15,9 | 1,04 | 1,03 a 1,07 | 1,03 | 48,07 | 48,07 | 48,07 | 3,01 |
| `direct_create_train_X_y_float` | 14,0 | 14,1 | 13,6 | 1,01 | 0,99 a 1,03 | 0,97 | 37,02 | 41,60 | 41,60 | 1,76 |
| `recursive_fit_linear` | 124,9 | 126,3 | 126,1 | 1,01 | 0,99 a 1,03 | 1,01 | 90,60 | 90,60 | 90,60 | 1,01 |
| `direct_create_train_X_y` | 32,1 | 38,4 | 35,5 | 1,20 | 1,16 a 1,26 | 1,11 | 62,04 | 66,49 | 66,48 | 1,54 |
| `direct_multivariate_create_train_X_y` | 35,7 | 39,8 | 37,8 | 1,11 | 1,10 a 1,29 | 1,06 | 71,81 | 72,92 | 72,91 | 1,53 |
| `classifier_create_train_X_y` | 40,7 | 46,0 | 41,9 | 1,13 | 1,11 a 1,24 | 1,03 | 30,43 | 34,66 | 30,43 | 1,33 |
| `predict_loop` | 41,3 | 53,1 | 45,2 | 1,29 | 1,27 a 1,34 | 1,09 | 0,03 | 0,06 | 0,07 | 1,32 |
| `expand_index_check_predict_input` | 241,6 | 319,2 | 289,6 | 1,32 | 1,29 a 1,56 | 1,20 | 0,01 | 0,22 | 0,20 | 1,32 |
| `backtesting_recursive` | 155,4 | 149,3 | 148,9 | 0,96 | 0,93 a 0,98 | 0,96 | 7,36 | 7,38 | 7,36 | 1,13 |
| `backtesting_recursive_refit` | 70,5 | 75,1 | 70,2 | 1,06 | 1,06 a 1,10 | 1,00 | 0,58 | 0,60 | 0,59 | 1,06 |
| `backtesting_multiseries` | 210,1 | 244,2 | 217,7 | 1,16 | 1,14 a 1,18 | 1,04 | 9,63 | 9,87 | 9,79 | 1,08 |
| `window_features` | 33,5 | 30,9 | 30,4 | 0,92 | 0,91 a 0,93 | 0,91 | 18,33 | 13,74 | 13,74 | 0,92 |
| `fit_recursive` | 66,6 | 69,9 | 67,0 | 1,05 | 1,04 a 1,11 | 1,01 | 46,92 | 46,92 | 46,92 | 1,04 |
| `fit_direct` | 178,5 | 183,5 | 177,1 | 1,03 | 1,02 a 1,08 | 0,99 | 53,04 | 53,05 | 53,05 | 1,03 |
| `fit_direct_multivariate` | 988,6 | 991,1 | 987,2 | 1,00 | 1,00 a 1,01 | 1,00 | 185,77 | 185,78 | 185,78 | 1,01 |
| `fit_classifier` | 338,1 | 341,0 | 343,2 | 1,01 | 0,97 a 1,03 | 1,02 | 29,06 | 33,04 | 28,81 | 1,00 |
| `recursive_create_train_X_y_noexog` | 4,2 | 10,7 | 10,7 | 2,54 | 2,49 a 2,62 | 2,53 | 6,36 | 39,68 | 39,68 | nuevo |
| `predict_loop_categorical` | 195,8 | 279,9 | 205,3 | 1,43 | 1,39 a 1,46 | 1,05 | 0,18 | 0,26 | 0,24 | nuevo |
| `predict_loop_scaler` | 149,5 | 198,3 | 171,5 | 1,33 | 1,27 a 1,33 | 1,15 | 0,09 | 0,14 | 0,14 | nuevo |
| `predict_loop_pipeline` | 233,2 | 280,5 | 254,5 | 1,20 | 1,16 a 1,22 | 1,09 | 0,63 | 0,81 | 0,80 | nuevo |
| `fit_recursive_transformers` | 56,8 | 59,7 | 59,1 | 1,05 | 1,00 a 1,05 | 1,04 | 45,31 | 45,31 | 45,31 | nuevo |
| `recursive_create_train_X_y_wide_exog` | 88,4 | 70,1 | 64,4 | 0,79 | 0,77 a 0,80 | 0,73 | 140,19 | 140,20 | 140,19 | nuevo |
| `fit_recursive_wide_exog` | 462,0 | 441,2 | 434,9 | 0,96 | 0,95 a 0,96 | 0,94 | 178,14 | 178,15 | 178,14 | nuevo |
