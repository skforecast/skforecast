# Qué cambia en skforecast 0.28 frente a la 0.27

**Estado: revisión completa (partes 1 a 5).** Este documento cubre la migración a pandas 3
y las dependencias (parte 1), la paralelización (parte 2), el rendimiento de `predict` y del
backtesting (parte 3), las correcciones de errores (parte 4) y la exógena como matriz float
y las variables categóricas con un `Pipeline` (parte 5).

## 0. Cómo leer este documento

- Solo recoge **cambios en el código de la librería**: qué cambia, por qué y qué efecto tiene
  para quien usa skforecast. Los cambios en los tests y en la CI se mencionan en una línea; el
  módulo `experimental`, también.
- Cada cambio cita sus pull requests (`#nnnn`). El detalle está en su descripción en GitHub.
- **pandas 2** es pandas 2.3 y **pandas 3** es pandas 3.1. Las cifras de rendimiento son de
  Linux con Python 3.12, numpy 2.5 y scikit-learn 1.9, salvo que se diga otra cosa (las de
  paralelización dicen siempre el número de CPU; las de rendimiento son de las pull
  requests, sección 2.8).
- La revisión se hizo del 2026-11-23 al 2026-11-25 sobre la rama `0.28.x`, leyendo el
  código de cada pull request y comprobando con experimentos lo que no estaba claro.


## 1. En una página

skforecast 0.28 funciona con **pandas 2.3 y pandas 3** con el mismo código. Con los mismos
datos, predicciones, intervalos y métricas son iguales en las dos versiones, salvo dos casos
documentados (sección 3). Lo que cambia son los tipos que pandas 3 cambia para todas las
librerías: texto `str` en lugar de `object`, fechas en microsegundos, índices con zona
horaria que pierden la frecuencia al convertirlos y arrays de solo lectura.

La migración obligó a tocar tres zonas de la librería, y en las tres se aprovechó para
arreglar errores que **ya estaban en la 0.27 y afectan también a pandas 2**:

1. **Frecuencias y zonas horarias.** La librería ya no supone una frecuencia diaria cuando
   no la tiene: la infiere, también en UTC, o falla con un mensaje que explica qué hacer.
2. **Copias de los datos.** Los transformadores ya no modifican los datos del usuario. Antes,
   uno que escribe en su entrada (`StandardScaler(copy=False)`, un `FunctionTransformer` in
   situ) daba **métricas erróneas sin aviso** en backtesting y búsquedas.
3. **Dependencias.** Python 3.11, numpy 2.0 y pandas 2.3 pasan a ser los mínimos.

La **paralelización** del backtesting y de las búsquedas (`n_jobs`) se rehízo para que vaya
más rápido sin cambiar ningún resultado: el forecaster se envía a los procesos una sola vez,
nunca se arrancan más procesos que folds, y `n_jobs='auto'` paraleliza un backtesting sin
refit cuando las predicciones tardan lo bastante. Con 16 CPU, una búsqueda de XGBoost con
refit es 1,8 veces más rápida y un backtesting de HistGradientBoosting sin refit, 2,4 veces
(en la primera llamada; en las siguientes, 1,9 y 3,8 veces).
De paso se arreglaron fallos de la 0.27: el backend `threading` daba métricas erróneas o
errores con refit, y los `n_jobs` negativos no seguían la regla de joblib. A cambio, algunos
modelos con arrays grandes (KNN) van más lentos con `n_jobs` entero, y una carpeta temporal
de joblib no escribible hace fallar casos que antes funcionaban, con un error que explica
cómo resolverlo.

**`predict` y el backtesting son más rápidos**, con resultados idénticos. El mayor cambio es
en multiserie: con 100 series, `predict` tarda de un 28 a un 82 % menos y cada fold de un
backtesting sin refit, de un 53 a un 81 % menos. Con un `StandardScaler`, el escalador se
aplica unas 30 veces más deprisa, lo que hace un backtesting de muchos folds cortos hasta
2,2 veces más rápido. En los forecasters de una serie, un backtesting sin refit de muchos
folds tarda un 30 % menos, y el formato largo se convierte de 2 a 3 veces más deprisa. Estos
cambios compensan el sobrecoste de pandas 3: con los forecasters de una serie, pandas 3 con
la 0.28 va más rápido que pandas 2 antes de ella.

**Correcciones de errores que ya estaban en la 0.27.** En multiserie con `RangeIndex`, una
exógena de predicción con el índice reseteado daba **predicciones erróneas sin aviso**; ahora
da un error que explica cómo arreglarla (a cambio, deja de aceptar una exógena que incluya
el periodo de entrenamiento). `ForecasterRnn` empieza cada ajuste desde los pesos iniciales:
el backtesting con refit ya no depende de los folds anteriores ni da **métricas optimistas**
con un forecaster ya ajustado, y `set_params` y las búsquedas de hiperparámetros funcionan
(antes descartaban todos los candidatos). En el clasificador, el backtesting con refit ya no
predice clases erróneas cuando las clases cambian entre folds ni depende de `n_jobs`, y
`predict_proba` ya no falla cuando una clase solo aparece en los primeros valores de la
serie. Una exógena con columnas `bool`, nullable (`Int64`, `boolean`...) o de pyarrow junto
a otras numéricas llega al modelo como una matriz float: `fit` es unas 4 veces más rápido
con un modelo lineal, `predict` ya no falla con HistGradientBoosting y una categórica, y un
`pd.NA` da el error de valores ausentes en lugar de un `TypeError`. Por último, con un
`Pipeline` como estimador ya no se configuran las categóricas nativas en su último paso:
con LightGBM y CatBoost fallaba, y con HistGradientBoosting detrás de un escalador trataba
como categorías los valores escalados. A cambio, un `Pipeline` que termina en
HistGradientBoosting o XGBoost da otro modelo.

Para publicar quedan un mensaje de error claro para los `ForecasterRnn` guardados con la
0.27 (hoy, un `AttributeError`), varios fallos previos de fechas con zona horaria que ya
estaban en la 0.27 y reejecutar cinco notebooks de la documentación con pandas 3 (sección
5). Las correcciones de documentación P1 ya se hicieron (#1539). Fuera de esta revisión, el
roadmap selecciona además para la 0.28.0 varios fallos previos de categóricas (sección 5).


## 2. Los cambios, por área

### 2.1 Versiones mínimas y pandas 3 (#1496, #1500, #1501)

**Qué cambia.**

- Python 3.11 como mínimo (`requires-python = ">=3.11"`).
- Mínimos del núcleo: numpy 2.0, pandas 2.3 (`pandas>=2.3, <4.0`), scipy 1.13 y numba 0.60.
  Opcionales: statsmodels 0.14.6 y matplotlib 3.8.4.
- Restos de numpy 1.x eliminados (`np.trapz` pasa a `np.trapezoid` en las métricas).
- El adaptador de Moirai explica en su `ImportError` cómo instalarlo en dos pasos (su paquete
  declara numpy 1.26, pero funciona con numpy 2).

**Por qué.** Python 3.10 dejó de tener soporte en octubre de 2026 y pandas 3 exige 3.11.
numpy 1.x ya no recibe arreglos y no tiene ruedas para Python 3.13 y 3.14. Con numpy 1.x como
mínimo, `pandas<4.0` habría permitido una combinación (pandas 3 con numpy 1.26) que no se
probaba. El techo `<4.0` protege de los cambios que anuncia pandas 4.

**Efecto para el usuario.**

- Con Python 3.10, pip sigue instalando la 0.27.
- Instalar la 0.28 sube numpy y pandas si son anteriores a los mínimos. Comprobado con
  `pip install --dry-run`:
  - con pandas 2.3, `pip install -U skforecast` conserva pandas 2.3;
  - con pandas anterior a 2.3, sube a **pandas 3**, no a 2.3;
  - `uv pip install -U` y `--upgrade-strategy eager` suben a pandas 3 también desde 2.3.

  Para quedarse en pandas 2: `pip install -U skforecast "pandas<3"`.
- Los backends de `MoiraiAdapter` y `TabPFNAdapter` (`gluonts`) todavía exigen pandas 2:
  instalarlos baja pandas.

### 2.2 Un único punto de contacto con la versión de pandas (#1495, #1497)

**Qué cambia.**

- Nuevo módulo privado `skforecast/utils/_pandas_compat.py`, el único que lee la versión de
  pandas. Envuelve tres operaciones cuyo comportamiento cambia o se depreca en pandas 3:
  `concat`, `astype` (con una opción para convertir solo las columnas que cambian) e
  `infer_freq` (que en pandas 3 devuelve un objeto en lugar de un texto).
- Renombrados de APIs que pandas 3 depreca: `dayofweek` y `dayofyear` pasan a `day_of_week` y
  `day_of_year`; las comprobaciones de tipo usan `pd.offsets.BaseOffset`.
- Fuera del módulo solo hay dos puntos que dependen de la versión, los dos en mensajes: una
  pista en los errores de índice sin frecuencia y una nota en `load_forecaster`.

**Por qué.** Con pandas 3, el paquete emitía unos 1700 avisos `Pandas4Warning` desde 30
sitios. Concentrar la compatibilidad en un módulo hace que retirar pandas 2 en el futuro sea
cambiar un fichero, no buscar ramas por todo el paquete.

**Efecto para el usuario.** Ningún aviso de pandas nacido en skforecast. Como efecto
secundario, las matrices de entrenamiento se crean algo más rápido: `astype` ya no se llama
cuando ninguna columna cambia de tipo.

### 2.3 Frecuencias y zonas horarias (#1497, #1505)

**El problema de pandas 3.** Al convertir un índice de zona horaria (`tz_convert`), pandas 3
descarta las frecuencias de calendario (`'D'`, `'W'`, `'MS'`), porque un día de calendario ya
no son siempre 24 horas. Un índice diario creado en UTC y mostrado en hora de Madrid se queda
sin frecuencia. Además, `asfreq` sin frecuencia ya no puede suponer nada.

**Qué cambia.**

- **Inferencia de la frecuencia** (`_infer_freq_from_index`). Se infiere en hora local y,
  si el índice tiene zona horaria, también en UTC. La de UTC gana solo si el índice avanza
  en pasos fijos de UTC (lo decide `_is_utc_anchored_index` a partir de la hora del día que
  se mantiene constante al cruzar un cambio de hora).
- **Fechas futuras** (`expand_index`). Se generan con la misma convención que el índice: en
  UTC si avanza en pasos UTC, en hora local si no. Antes, un índice sin frecuencia se
  infería solo en hora local, lo que falla con un índice convertido de UTC que cruza un
  cambio de hora (el caso habitual con pandas 3, que descarta la frecuencia).
- **`reshape_series_wide_to_long`.** Infiere la frecuencia de las fechas y lanza un
  `ValueError` si no puede, en lugar de suponer frecuencia diaria. Ya no renombra el índice
  de la entrada.
- **Mensajes de error.** Cuando un índice con zona horaria no tiene frecuencia, el error
  explica la causa de pandas 3 y propone `freq='24h'`.
- **Huecos en `exog` en `predict`** (#1505). Un atajo daba por buena una `exog` que empezaba
  en el primer paso y tenía "la misma frecuencia" que los pasos predichos. Con pandas 3 las
  dos frecuencias podían ser `None`, y `None == None` aceptaba una `exog` con huecos: el
  forecaster usaba valores de otras fechas. Ahora el atajo exige una frecuencia, y si no la
  hay se comparan las fechas una a una.

**Por qué.** Sin estos cambios, con pandas 3, los forecasters fallaban con datos diarios
convertidos de UTC, y `reshape_series_wide_to_long` perdía filas en silencio al suponer
frecuencia diaria (también con pandas 2).

**Efecto para el usuario.**

- Con datos en pasos UTC convertidos a hora local, la recomendación es `freq='24h'` (o
  `'168h'` semanal): pandas la conserva en las dos versiones.
- `reshape_series_wide_to_long` sin frecuencia lanza `ValueError` con datos diarios con
  huecos o con menos de 3 fechas, y los días laborables pasan a frecuencia `'B'` en lugar de
  filas diarias con NaN los fines de semana. Es un cambio también con pandas 2.
- `TimeSeriesFold.split` y `OneStepAheadFold.split` dan un `ValueError` con la lista de
  frecuencias encontradas en lugar de un `TypeError`.

### 2.4 Columnas de texto (#1497)

**Qué cambia.** Donde la librería comparaba tipos de las exógenas entre series, `str`,
`object` y `string` cuentan como el mismo tipo de texto. Se quita un relleno (`fillna`) que no
rellenaba nada y daba un aviso de pandas.

**Por qué.** pandas 3 guarda el texto como `str`, pero como `object` si todos los valores
faltan, así que dos series con la misma columna podían tener tipos distintos.

**Efecto para el usuario.**

- Una exógena `string` en una serie y `object` en otra ya no da `TypeError` (también con
  pandas 2).
- Con pandas 3, `None` en una columna de texto es un valor ausente (`NaN`); con pandas 2 era
  una categoría propia. Con un estimador que no admite NaN (Ridge), el mismo código funciona
  con pandas 2 y falla con pandas 3. Está documentado; la librería no lo cambia.

### 2.5 Copias: los datos del usuario y el rendimiento (#1502, #1504, #1505)

Es el bloque más importante en corrección y en rendimiento.

**a) Los transformadores ya no modifican los datos del usuario (#1504).**

- **Qué cambia.** Las tres funciones por las que pasa todo transformador (`transform_numpy`,
  `transform_series` y `transform_dataframe`) entregan al transformador una copia de la
  entrada. Dos sitios de la búsqueda one-step-ahead que llamaban al transformador
  directamente pasan por estas funciones.
- **Por qué.** Un transformador que escribe en su entrada (`StandardScaler(copy=False)`, un
  `Pipeline` que lo contiene, un `FunctionTransformer` in situ) modificaba `y`, `exog` o el
  conjunto de test compartido por los candidatos de una búsqueda. Copy-on-Write de pandas 3
  no lo evita, porque scikit-learn vuelve escribibles los arrays de pandas antes de escribir.
  **El fallo existía en la 0.27.** Comprobado con ella:
  - backtesting con `refit=True`: MAE 4,30 en lugar de 1,55, y la `y` del usuario modificada;
  - búsqueda one-step-ahead con tres candidatos idénticos: MAE 0,95, 82,7 y 507,9 en lugar
    de 0,76 los tres;
  - con diferenciación, los tres candidatos daban el mismo valor erróneo (3,36 en lugar de
    0,78), así que comparar candidatos no lo delataba.

  Con un `FunctionTransformer()` identidad y variables categóricas, el segundo `predict`
  fallaba.
- **Efecto para el usuario.** Métricas correctas con cualquier transformador. El coste de la
  copia es de un 0 a un 5 % en `fit`, `predict` y backtesting, y de un 12 % en la creación de
  la matriz multiserie con pandas 2. Con pandas 2, algunos resultados cambian en el
  redondeo (hasta 1e-15), y pandas 2 y pandas 3 pasan a dar exactamente lo mismo.

**b) Menos copias innecesarias con pandas 3 (#1502).**

- **Qué cambia.**
  - El `create_train_X_y` público construye su `DataFrame` sin copiar la matriz cuando esta
    ya es un array nuevo. pandas 3 copia por defecto cualquier array que recibe; pandas 2,
    no.
  - Antes de codificar las variables categóricas sin `transformer_exog`, la copia de la
    `exog` pasa de profunda a superficial. La superficial protege igual al usuario, porque
    asignar las columnas codificadas las sustituye sin escribir en sus arrays.
- **Por qué.** Con pandas 3, el `create_train_X_y` público era de 1,3 a 3 veces más lento que
  con pandas 2.
- **Efecto para el usuario.** Con pandas 3, `create_train_X_y` (y `select_features`, que lo
  usa) tarda de 0,26 a 0,34 veces lo que tardaba, con 0,57 veces la memoria, y queda a la par
  de pandas 2. La matriz devuelta sigue siendo modificable. `fit` no usa este constructor y
  no cambia. Resultados idénticos.

**c) Array escribible para las window features en `ForecasterDirectMultiVariate.predict`
(#1505).** Con pandas 3 y sin `transformer_series`, la ventana llegaba a las window features
como un array de solo lectura; una window feature que necesita escribir (código compilado)
fallaba. Ahora recibe una copia, como en los demás forecasters.

### 2.6 Guardar y cargar forecasters (#1497, #1536)

**Qué cambia.** `load_forecaster` añade una nota a dos errores de carga entre versiones de
pandas. El tipo del error no cambia en ninguno de los dos.

- Un forecaster guardado con pandas 3 (joblib o pickle) que se carga con pandas 2: la nota
  explica la causa y que skops funciona entre versiones (#1497).
- Un forecaster guardado con pandas 2 (joblib, pickle o cloudpickle) con una zona horaria con
  nombre (`'Europe/Madrid'`) que se carga con pandas 3 sin `pytz` instalado: el
  `ModuleNotFoundError` dice que hay que instalar `pytz` (#1536).

**Por qué.** pandas 2 no puede reconstruir algunos objetos de pandas 3, y el error original
(un `AssertionError` vacío) no daba ninguna pista. En el otro sentido, pandas 2 guarda esas
zonas horarias como objetos de `pytz`, y pandas 3 ya no instala `pytz`: el error no decía por
qué un forecaster de la 0.27 dejaba de cargar.

**Efecto para el usuario.** Los dos casos siguen sin cargar, pero el error dice qué hacer.
Cargar sin `pytz` (reconstruyendo la zona horaria con `zoneinfo`) queda como mejora futura.

### 2.7 Paralelización del backtesting y las búsquedas (#1503, #1513, #1535, #1538)

Afecta a `backtesting_forecaster`, `backtesting_forecaster_multiseries`,
`backtesting_stats` y a todas las búsquedas de hiperparámetros, que los usan. Ningún cambio
altera predicciones ni métricas: una revisión independiente comparó `n_jobs=1` con
`n_jobs>1` en más de 600 casos sin diferencias.

**a) El forecaster se envía a los procesos una sola vez (#1503).**

- **Qué cambia.** El forecaster se serializa una vez, en el hilo principal, y viaja como un
  único array de bytes. Si pasa de 1 MB, joblib lo escribe una vez en un fichero temporal que
  comparten todos los procesos. Cada fold lo deserializa y trabaja con su propia copia.
- **Por qué.** joblib lo serializaba una vez por fold, desde un hilo secundario: el coste
  crecía con el número de folds y, en Linux, serializar XGBoost o LightGBM desde ese hilo
  arrancaba un segundo pool de hilos de OpenMP, que hacía hasta 5 veces más lentos los `fit`
  posteriores de XGBoost en la misma sesión.
- **Efecto para el usuario.** Más rápido con muchos folds y modelos grandes, y sin el
  frenazo de XGBoost después de un backtesting en paralelo. Dos costes aceptados:
  - modelos que guardan arrays grandes (KNN, SVR, procesos gaussianos), con `n_jobs`
    entero y sin refit, van de 2 a 3 veces más lentos que antes desde la segunda llamada,
    porque cada fold copia el modelo (con `n_jobs='auto'` van en un proceso y no lo notan);
  - si la carpeta temporal de joblib no se puede escribir (punto e), falla un caso que
    antes funcionaba.

**b) Nunca más procesos que folds (#1503).**

- **Qué cambia.** El número de procesos se limita al de folds, también con `n_jobs` entero.
- **Por qué.** Solo se limitaba con `n_jobs='auto'` en un backtesting suelto, y las
  búsquedas convierten `'auto'` en un entero antes: una búsqueda de 6 folds arrancaba
  `cpu_count() - 1` procesos que esperaban sin hacer nada.
- **Efecto para el usuario.** Menos tiempo y memoria cuando hay menos folds que CPU. Con 16
  CPU, una búsqueda de XGBoost con refit (3 candidatos, 6 folds) pasa de 47 s a 26 s en la
  primera llamada y de 45 s a 24 s en las siguientes, con los puntos a y b juntos.

**c) `n_jobs='auto'` sin refit decide por el tiempo de predicción (#1503, #1538).**

- **Qué cambia.** En `backtesting_forecaster` sin refit, con un `HistGradientBoostingRegressor`
  o un `CatBoostRegressor` y 4 folds o más, se predicen los dos primeros folds en el proceso
  principal y, con el más rápido de los dos, se estima cuánto tardarán los demás. Se
  paralelizan solo si la estimación supera 6 s (Linux y macOS) o 12 s (Windows). En la 0.27,
  un backtesting sin refit iba siempre en un proceso. El multiserie sin refit sigue igual.
- **Por qué.** Estos estimadores predicen pocas filas despacio, y con muchos folds compensa
  repartirlos. Se usan dos folds porque la primera predicción de un proceso paga costes
  únicos (crear el controlador de hilos, compilar con numba las window features) y, sola,
  inflaba la estimación hasta 10 veces: en la primera llamada de una sesión se arrancaban
  procesos para décimas de segundo de trabajo (#1538 lo corrigió antes de publicar).
- **Efecto para el usuario.** Con 16 CPU, un backtesting de HistGradientBoosting sin refit
  de 100 folds pasa de 19 s a 8 s en la primera llamada y a 5 s en las siguientes. Con
  muchos folds pequeños se mantiene el tiempo de la 0.27 (1000 folds de un paso con 4 CPU:
  1,0 a 1,1 s). Entre 4 y 6 s de trabajo en Linux con pocas CPU, el umbral es algo alto y se
  queda en un proceso cuando el paralelo iría un poco más rápido (5,2 s frente a 4,1 s con 4
  CPU, medido en esta revisión en una máquina compartida: orientativo); nunca es más lento
  que la 0.27, que siempre usaba un proceso.

**d) El backend `threading` ya no comparte el forecaster entre folds (#1503).**

- **Qué cambia.** Con `joblib.parallel_config(backend="threading")`, cada fold trabaja con su
  propia copia del forecaster.
- **Por qué.** Todos los folds recibían el mismo objeto: con refit lo ajustaban a la vez, y
  los de `backtesting_stats` también predecían a la vez. **El fallo existía en la 0.27.**
  Comprobado con ella: con refit y `n_jobs=4`, error con HistGradientBoosting
  (`ValueError`) y con Ridge (`NotFittedError`); la PR documenta también métricas erróneas
  sin aviso.
- **Efecto para el usuario.** Resultados idénticos a `n_jobs=1`. Un forecaster que no se
  puede serializar ahora da error con ese backend (usar `n_jobs=1`).

**e) La carpeta temporal de joblib (#1535).**

- **Qué cambia.** Si joblib no puede escribir en su carpeta temporal (`JOBLIB_TEMP_FOLDER`
  mal puesta, sistema de solo lectura, disco lleno), el `PicklingError` del backtesting
  explica la causa y tres soluciones: `JOBLIB_TEMP_FOLDER`, `joblib.parallel_config(max_nbytes=None)`
  o `n_jobs=1`. No se reintenta automáticamente: el fichero compartido es de 1,6 a 2 veces
  más rápido con forecasters grandes y el reintento complicaba el código.
- **Por qué.** Desde el punto a, un forecaster que pasa de 1 MB en total necesita esa
  carpeta aunque ninguno de sus arrays pase de 1 MB (un RandomForest son muchos árboles
  pequeños). En la 0.27 no la necesitaba. El error original no decía la causa.
- **Efecto para el usuario.** Con la carpeta temporal rota, un caso que funcionaba en la
  0.27 ahora falla, con un mensaje que dice cómo resolverlo. Comprobado: RandomForest de
  2,1 MB, `n_jobs=2`, sin refit; la 0.27 da resultados y la 0.28, el error explicado.

**f) `n_jobs` entero con la regla de joblib (#1513).**

- **Qué cambia.** En los tres backtesting, las búsquedas, `ForecasterDirect` y
  `ForecasterDirectMultiVariate`:
  - un `n_jobs` negativo usa `cpu_count() + 1 + n_jobs` procesos (`-2`, todas las CPU menos
    una), como joblib; antes, cualquier negativo usaba todas;
  - `n_jobs=0`, que joblib no admite, da `ValueError`; antes usaba todas las CPU, o una con
    un aviso;
  - `refit=0` cuenta como `False`: con `n_jobs` entero ya no se trata como refit intermitente
    (un proceso y un aviso), sino que va en paralelo;
  - las búsquedas comprueban `n_jobs` una vez, antes del primer candidato. Antes, un valor
    inválido hacía fallar cada candidato y la búsqueda devolvía resultados vacíos con un
    aviso.
- **Por qué.** Eran fallos de la 0.27, encontrados en la revisión de #1503.
- **Efecto para el usuario.** `-2` y siguientes usan un proceso menos que antes. Los
  resultados no cambian. Un entero de NumPy (`np.int64`) sigue dando `TypeError`, como en la
  0.27.

### 2.8 Rendimiento de `predict` y del backtesting (#1512, #1516, #1519, #1530)

Cuatro cambios que solo quitan trabajo repetido. **Ninguno cambia resultados, avisos ni
errores de uso normal**: cada pull request compara sus salidas bit a bit con la versión
anterior en pandas 2 y 3, y esta revisión lo comprobó de nuevo en los casos dudosos. Las
cifras son de las pull requests (en Windows, salvo las de `StandardScaler`, en Linux); no se
han vuelto a medir. Con pandas 3, y más con `pyarrow`, cada objeto de pandas que se crea a
partir de una lista de nombres (índices, columnas, `reindex`, `concat`) cuesta más que con
pandas 2. Por eso estos cambios importan más con pandas 3: compensan su sobrecoste. La
conversión de la exógena a una matriz float (#1542), que también acelera `fit`, está en la
sección 2.9 e) porque además corrige errores.

**a) `predict` y backtesting de `ForecasterRecursiveMultiSeries` (#1530).**

- **Qué cambia.** Cada paso que solo devolvería los mismos datos se salta: en backtesting, el
  entrenamiento se extrae una vez para todos los folds con los mismos límites (sin refit,
  todos); la ventana no se recorta ni se copia si sus columnas ya son las series a predecir;
  la exógena no se reindexa si ya está alineada; una exógena compartida por varias series se
  comprueba, alinea y recorta una sola vez, y la exógena ancha se repite con numpy en lugar de
  concatenarse una vez por serie. Fuera de esos casos se sigue el camino de antes.
- **Por qué.** El `predict` multiserie creaba varios objetos de pandas por serie y por
  llamada: con un modelo ligero, la mayor parte del tiempo de `predict` y de cada fold era
  eso, no el modelo.
- **Efecto para el usuario.** Con 100 series, `predict` tarda de un 28 a un 82 % menos según
  el formato de la exógena (con exógena ancha, de 1,9 a 0,7 ms con pandas 2) y cada fold de
  un backtesting sin refit, de un 53 a un 81 % menos (con exógena ancha, de 38 a 8 ms). Con
  refit, donde domina el ajuste, de un 3 a un 12 %. `ForecasterDirectMultiVariate` también
  mejora (su backtesting con exógena ancha, de un 44 a un 86 % menos). `fit` no cambia.

**b) `StandardScaler` sin la validación de scikit-learn (#1519).**

- **Qué cambia.** Con un `StandardScaler` como `transformer_y` o `transformer_series` (el
  caso más habitual), el escalador ya ajustado se aplica con la misma aritmética que
  scikit-learn y en el mismo orden, pero sin su validación de la entrada. Cualquier otro caso
  sigue por scikit-learn: otros escaladores, datos `float32`, infinitos, arrays vacíos o un
  escalador sin ajustar, así que sus errores son los mismos.
- **Por qué.** El escalador se aplica una o dos veces por serie en cada predicción, sobre
  arrays cortos. En ellos, casi todo el coste de scikit-learn (de 57 a 84 µs por llamada) es
  validar la entrada; la cuenta ocupa menos de 3 µs. Con 300 series era un 30 % de `predict`.
- **Efecto para el usuario.** Cada aplicación pasa a 2 o 3 µs. Un backtesting de 1000 folds
  de un paso con `ForecasterRecursive` es 2,2 veces más rápido, y el `predict` multiserie con
  300 series, de 1,2 a 1,4 veces. El resultado depende de que scikit-learn no cambie la
  aritmética de `StandardScaler`; un test la compara con la versión instalada.

**c) Menos trabajo repetido en cada `predict` (#1516).**

- **Qué cambia.** Las fechas de los pasos a predecir se calculan una vez (antes, dos), la
  ventana no se recorta si ya tiene el tamaño justo, y la comprobación de valores ausentes de
  la exógena se salta cuando sus valores no pueden tenerlos. El backtesting de una serie une
  las predicciones de los folds con numpy en lugar de `pd.concat`. Afecta sobre todo a
  `ForecasterRecursive`, `ForecasterDirect` y `ForecasterRecursiveClassifier`.
- **Por qué.** En un bucle de `predict` con un modelo ligero (backtesting sin refit con
  muchos folds cortos), el tiempo se iba en objetos de pandas.
- **Efecto para el usuario.** Un backtesting sin refit de 1000 folds de un paso tarda de un
  27 a un 35 % menos en los forecasters de una serie, y de un 2 a un 5 % menos en los de
  varias series. Con estos forecasters, pandas 3 con la 0.28 va más rápido que pandas 2 con
  la versión anterior. Queda una mejora mayor para más adelante: validar solo el primer fold
  del backtesting cuando los datos lo garantizan (PERF-predict-numpy).

**d) Formato largo a diccionario (#1512).**

- **Qué cambia.** Las series y las exógenas en formato largo (un DataFrame con un
  MultiIndex de serie y fecha) se separan por serie ordenando las filas una sola vez, en
  lugar de copiar todo el DataFrame y agruparlo. En `fit` y en `predict` solo se convierten
  las series que se usan.
- **Por qué.** Con miles de series, la conversión costaba de 1 a 2 s fijos en cada `fit`,
  backtesting o búsqueda.
- **Efecto para el usuario.** Con 5000 series, la conversión de las series es de 2 a 3 veces
  más rápida y la de las exógenas, de 1,4 a 1,8 veces. Predecir una serie con una exógena
  larga de 5000 series baja de 368 ms a 4 ms (pandas 2). Dos entradas no válidas cambian de
  error: un MultiIndex de tres niveles da ahora un `ValueError` claro, y una primera columna
  con el nombre repetido funciona (antes, `TypeError`). Sigue recomendándose pasar las series
  como diccionario.

### 2.9 Correcciones de errores (#1511, #1514, #1515, #1540, #1542, #1544)

Errores que ya estaban en la 0.27 y afectan igual a pandas 2 y 3. Los de a), b) y c) daban
resultados erróneos sin aviso; los de d), e) y f), sobre todo errores, con algún caso de
resultado erróneo que se indica.

**a) Exógena con `RangeIndex` en `ForecasterRecursiveMultiSeries` (#1540).**

- **Qué cambia.** Con series de `RangeIndex`, los métodos de predicción dan un `ValueError` si
  la exógena tiene etiquetas anteriores al primer paso predicho. El mensaje propone las dos
  soluciones: seleccionar desde el primer paso (`exog.loc[primer_paso:]`) o poner un índice
  que empiece en él. Con un diccionario, cada serie se comprueba con su exógena y el mensaje
  la nombra.
- **Por qué.** Desde la 0.26, este forecaster alinea la exógena por las etiquetas de su índice.
  Con fechas no hay confusión posible, pero con `RangeIndex` las etiquetas son posiciones:
  una exógena futura con el índice reseteado (0, 1, 2...) tiene las etiquetas del
  entrenamiento, así que se usaban **los valores de otros pasos sin ningún aviso**.
- **Efecto para el usuario.** El caso erróneo pasa a ser un error. A cambio, con `RangeIndex`
  deja de aceptarse una exógena que incluya el periodo de entrenamiento, que por su índice no
  se distingue de una reseteada, y una exógena común a series que terminan en posiciones
  distintas cuando se predice la que termina más tarde. Sin cambios: `DatetimeIndex`, el resto
  de forecasters (usan la exógena por posición) y el backtesting. El mismo problema en `fit`
  queda pendiente (sección 5).

**b) `ForecasterRnn`: cada ajuste desde los pesos iniciales (#1515).**

- **Qué cambia.** El forecaster guarda los pesos y la configuración de compilación del modelo
  al crearse, y cada `fit` vuelve a ellos con un optimizador nuevo. El argumento nuevo
  `warm_start=True` mantiene el comportamiento anterior (cada ajuste continúa desde el
  anterior), útil para acelerar los refits con `EarlyStopping`. `set_params` cambia los
  parámetros de compilación (por ejemplo, el optimizador) conservando los demás.
- **Por qué.** El modelo se entrenaba sobre sí mismo: en el backtesting con refit cada fold
  dependía de los anteriores, y un forecaster ya ajustado con toda la serie empezaba cada fold
  desde un modelo que había visto el test, con **métricas optimistas**. Además, `set_params`
  fallaba siempre, así que las búsquedas de hiperparámetros con `ForecasterRnn` descartaban
  todos los candidatos y no devolvían resultados.
- **Efecto para el usuario.** Cambian los resultados de los ajustes repetidos y del
  backtesting con refit (en el ejemplo de la pull request, el MAE del forecaster ya ajustado
  pasa de 0,357, optimista, a 0,463). Las búsquedas funcionan. Los forecasters guardados con la
  0.27 siguen prediciendo, pero hay que crearlos de nuevo para reentrenarlos, hacer
  backtesting con `initial_train_size` o llamar a `set_params`; hoy fallan con un
  `AttributeError` sin explicación (sección 5).

**c) Clases del clasificador en el backtesting con refit (#1511).**

- **Qué cambia.** En el backtesting de `ForecasterRecursiveClassifier`, cada fold elige la
  clase predicha con las clases de su propio modelo. La probabilidad de una clase que el
  modelo de un fold no conoce es 0 en lugar de NaN.
- **Por qué.** La clase se elegía al final, con las clases del forecaster del proceso
  principal. Cuando una clase entraba o salía del entrenamiento entre folds, se predecía una
  clase errónea en todos los pasos de un fold o nunca una clase nueva, y el resultado
  **dependía de `n_jobs`**. Las búsquedas con refit, que puntúan con esa clase, también.
- **Efecto para el usuario.** Sin refit, o cuando las clases no cambian, nada. En los demás
  casos cambian las métricas, que ahora coinciden con ajustar el forecaster a mano en cada
  fold y no dependen de `n_jobs`.

**d) Clasificador con una clase que solo aparece en los lags (#1514).**

- **Qué cambia.** Cuando el estimador no conoce una clase, su probabilidad es 0 y la clase
  predicha se lee correctamente. `get_feature_importances` etiqueta cada fila de coeficientes
  con su clase; con un clasificador lineal binario da una fila, la de la clase positiva, como
  scikit-learn.
- **Por qué.** El forecaster codifica las clases con toda la serie (también los primeros
  valores, que solo se usan como lags), pero el estimador solo las aprende a partir de ahí. Si
  una clase solo estaba al principio, `predict_proba` fallaba. Ocurre, por ejemplo, en el fold
  de backtesting en que una clase sale de la ventana de entrenamiento. `get_feature_importances`
  fallaba por la misma causa y, además, con cualquier clasificador lineal binario.
- **Efecto para el usuario.** Esos casos funcionan; con todas las clases conocidas la salida
  es la misma. Con `XGBClassifier` el caso sigue fallando en `fit` (sección 5).

**e) La exógena llega al modelo como una matriz float (#1542).**

- **Qué cambia.** Los forecasters convierten la exógena a una matriz float64, con NaN en los
  valores ausentes, cuando todas sus columnas son numéricas, booleanas o `category` de
  enteros y alguna es `bool` (junto a números), nullable (`Int64`, `boolean`...), de
  pyarrow o `category`. La decisión depende solo de los tipos de las columnas. En el resto de
  casos (todo float, enteros y float, texto, fechas...), nada cambia. Afecta a `fit` y
  `predict` de `ForecasterRecursive`, `ForecasterRecursiveClassifier`, `ForecasterDirect` y
  `ForecasterDirectMultiVariate`, y a los métodos de predicción de
  `ForecasterRecursiveMultiSeries`. `create_train_X_y` y `create_predict_X` siguen
  devolviendo los tipos originales.
- **Por qué.** Con esas columnas, pandas devuelve una matriz de tipo `object`: el modelo va
  más lento y usa más memoria, `predict` fallaba con HistGradientBoosting y una categórica
  (`ufunc 'isnan' not supported`; el `fit` funcionaba) y un `pd.NA` daba
  `TypeError: float() argument must be ... not 'NAType'` en `fit` y `predict`.
- **Efecto para el usuario.** Con 100.000 filas y una columna `bool`, `fit` es unas 4 veces
  más rápido con un modelo lineal (un 15 % con LightGBM) y el pico de memoria al crear la
  matriz de entrenamiento es 2,5 veces menor. Los errores de HistGradientBoosting y `pd.NA` desaparecen: un
  valor ausente da el error de valores ausentes de siempre, o NaN si el modelo los acepta.
  Un caso cambia de resultado: una `category` de enteros que no está en
  `categorical_features`, con valores ausentes y junto a columnas enteras, llegaba al modelo
  con −9,2·10¹⁸ en lugar del ausente, **sin aviso**; ahora es NaN, así que un modelo que no
  acepta NaN da error. Los forecasters con HistGradientBoosting guardados con una versión
  anterior siguen fallando en `predict` hasta reajustarlos.

**f) Categóricas nativas con un `Pipeline` como estimador (#1544).**

- **Qué cambia.** Con un `Pipeline`, el forecaster ya no configura las categóricas nativas
  en su último paso. Las variables categóricas llegan a él con su codificación ordinal, como
  columnas numéricas, igual que con cualquier modelo sin soporte nativo. Si el usuario
  configura él mismo el último paso, esa configuración se respeta. En
  `ForecasterRecursiveClassifier`, `features_encoding='categorical'` con un `Pipeline` da
  un error que explica el motivo.
- **Por qué.** Los pasos anteriores del `Pipeline` pueden transformar o reordenar las
  columnas, y la configuración no llegaba bien: con LightGBM y CatBoost, `fit` fallaba (los
  argumentos iban a `Pipeline.fit` sin el prefijo del paso); con XGBoost detrás de un
  escalador fallaba también, y HistGradientBoosting detrás de un escalador trataba como
  categorías los valores escalados, **sin aviso**. Afecta a las exógenas categóricas, a
  `encoding='ordinal_category'` en multiserie y a los lags del clasificador.
- **Efecto para el usuario.** Esos pipelines funcionan. Un `Pipeline` que termina en
  HistGradientBoosting o en XGBoost (sin escalador) da otro modelo, porque pierde las
  categóricas nativas; en los casos medidos, el error cambia poco y en los dos sentidos.
  Para conservarlas, hay que pasar el modelo sin el `Pipeline`. Los forecasters guardados
  con una versión anterior predicen igual, pero fallan si se reajustan sin la columna
  categórica: hay que crearlos de nuevo.

### 2.10 Otros cambios menores

- `calculate_distance_from_holiday` cuenta días de calendario con fechas con zona horaria
  (antes perdía un día en primavera o daba horas en lugar de días) y es mucho más rápida
  (#1495, #1505).
- `PopulationDriftDetector`: con un `chunk_size` que pandas no acepta, el error muestra el
  valor y el error de pandas, que con `'M'` en pandas 3 dice que se use `'ME'` (#1510).
- Documentación (#1510): página nueva "pandas 2 and pandas 3" en la FAQ, notas en cinco
  guías, skills y contexto de IA, y notas de versión con las excepciones.
- Los paneles de información de los datasets y el resumen de `RangeDriftDetector` tienen el
  mismo ancho fijo que los avisos: 88 caracteres, o el de la consola si es más estrecha. Las
  URL largas se parten en dos líneas, pero siguen siendo un único enlace (#1493).
- Módulo `experimental`: `TimeSeriesSplitter` toma la frecuencia de cada serie en sus salidas
  en formato largo (#1497) y ya no pierde los valores de series con la misma frecuencia y
  fechas desplazadas (#1534, que corrige una regresión introducida por #1497).
- Tests y CI (#1498, #1499, #1501): los tests no dependen de la versión de pandas, la CI
  prueba pandas 2 y pandas 3, y un job semanal falla ante cualquier aviso de deprecación de
  pandas nacido en skforecast. La acción `setup-uv` de la CI pasa a la versión 10.4.0
  (#1545).


## 3. Cambios que pueden romper código o cambiar resultados

Las filas de paralelización, rendimiento y correcciones no dependen de la versión de pandas.
Los cambios de rendimiento (sección 2.8) no cambian resultados; solo dos entradas no válidas
cambian de error.

| Cambio | Con pandas 2 | Con pandas 3 | Qué hacer |
|---|---|---|---|
| Mínimos: Python 3.11, numpy 2.0, pandas 2.3 | sí | sí | Actualizar o quedarse en `skforecast<0.28` |
| Texto `str`, fechas `[us]` y arrays de solo lectura en las salidas | no | sí | Comprobar texto con `pd.api.types.is_string_dtype`; `to_numpy(copy=True)` para modificar |
| Índice convertido de UTC sin frecuencia: `ValueError` | no | sí | `freq='24h'` o `'168h'` |
| `reshape_series_wide_to_long` sin frecuencia: `ValueError` con huecos o menos de 3 fechas; días laborables como `'B'` | sí | sí | Dar la frecuencia (`asfreq`) |
| Métricas con transformadores que escriben en su entrada | cambian (antes eran erróneas) | cambian | Ninguna: ahora son correctas |
| `None` en columnas de texto | categoría propia | valor ausente | `np.nan` o un texto como `'unknown'` |
| `ForecasterEquivalentDate` con `pd.offsets.Day(n)` y datos horarios con zona horaria | 24 horas | día de calendario (u error cerca de un cambio de hora) | Offset entero `24 * n` |
| Forecasters guardados con pandas 3 (joblib, pickle) | no cargan | — | skops entre versiones |
| `calculate_distance_from_holiday` con zona horaria | puede cambiar un día | puede cambiar un día | Ninguna: ahora es correcto |
| `TimeSeriesFold.split` con frecuencias mezcladas | `ValueError` en lugar de `TypeError` | igual | Capturar `ValueError` |
| Forecaster guardado con pandas 2 con zona horaria con nombre, cargado sin `pytz` | — | no carga (ahora el error lo dice) | `pip install pytz` |
| `n_jobs=0` | `ValueError` | igual | `1`, `-1` o `'auto'` |
| `n_jobs` negativo distinto de `-1` | un proceso menos que antes (regla de joblib) | igual | Ninguna |
| Búsqueda con `n_jobs` inválido (por ejemplo, `np.int64`) | error antes del primer candidato, no resultados vacíos | igual | `int(n_jobs)` |
| `refit=0` con `n_jobs` entero | en paralelo y sin aviso | igual | Ninguna |
| Backend `threading` con refit | métricas correctas (antes, erróneas o error) | igual | Ninguna |
| Backend `threading` con un forecaster que no se puede serializar | error | igual | `n_jobs=1` |
| Carpeta temporal de joblib no escribible y forecaster de más de 1 MB | falla (en la 0.27 funcionaba), con un error que explica la solución | igual | `JOBLIB_TEMP_FOLDER`, `parallel_config(max_nbytes=None)` o `n_jobs=1` |
| KNN, SVR o procesos gaussianos con `n_jobs` entero y sin refit | de 2 a 3 veces más lentos (mismos resultados) | igual | `n_jobs='auto'` o `n_jobs=1` |
| Formato largo con un MultiIndex de tres o más niveles | `ValueError` claro (antes, otro error) | igual | Dos niveles: serie y fecha |
| Series en formato largo con el nombre de la primera columna repetido | funciona con la primera columna (antes, `TypeError`) | igual | Ninguna |
| Multiserie con `RangeIndex` y exógena de predicción que empieza antes del primer paso (reseteada, con el histórico o común a series que terminan en posiciones distintas) | `ValueError` (antes, predicciones erróneas o correctas sin aviso, según el caso) | igual | `exog.loc[primer_paso:]` o un índice que empiece en el primer paso |
| `ForecasterRnn`: ajustes repetidos y backtesting con refit | cambian: cada ajuste empieza desde los pesos iniciales | igual | `warm_start=True` para encadenar los ajustes como antes |
| `ForecasterRnn` guardado con la 0.27 | predice; `fit`, `set_params` y backtesting con `initial_train_size` dan `AttributeError` | igual | Crearlo de nuevo |
| Backtesting y búsquedas del clasificador con refit, con clases que cambian entre folds | cambian (antes eran erróneas y dependían de `n_jobs`); probabilidad 0 en lugar de NaN | igual | Ninguna: ahora son correctos |
| Exógena con una `category` de enteros fuera de `categorical_features`, con valores ausentes y junto a columnas enteras | NaN (antes, −9,2·10¹⁸ sin aviso): error si el modelo no acepta NaN | igual | Declararla en `categorical_features` o imputar los ausentes |
| Forecaster con HistGradientBoosting, una categórica y una exógena `bool`, guardado con una versión anterior | sigue fallando en `predict` | igual | Reajustarlo |
| `Pipeline` que termina en HistGradientBoosting o XGBoost, con variables categóricas | otro modelo: sin categóricas nativas | igual | Pasar el modelo sin el `Pipeline` para conservarlas |
| Forecaster con un `Pipeline` y categóricas, guardado con una versión anterior | predice igual; falla si se reajusta con otras columnas | igual | Crearlo de nuevo |
| `ForecasterRecursiveClassifier` con `features_encoding='categorical'` y un `Pipeline` | `ValueError` al crearlo | igual | `'auto'` u `'ordinal'` |


## 4. Cómo se verificó

Las pull requests comparan sus salidas bit a bit con la versión anterior en pandas 2 y 3, y
una revisión independiente de toda la 0.28 no encontró diferencias de resultados frente a la
0.27 en unos 2000 escenarios de forecasters, backtesting y búsquedas. En esta revisión se
reprodujeron con experimentos los fallos que se arreglan (también con el código de la 0.27),
las afirmaciones de rendimiento principales y los casos dudosos. En la parte 2 se comprobó
además en Linux lo que las pull requests solo midieron en Windows (#1535 y #1538). En la
parte 3 se volvió a comprobar la igualdad de resultados donde cabía alguna duda: la
conversión del formato largo frente a la de la 0.27 (25 casos), la comprobación de valores
ausentes que se salta (12 tipos de exógena), la unión de folds sin `pd.concat` (20 casos),
el camino rápido de `StandardScaler` frente a scikit-learn, por bytes y avisos (66 casos), y
42 backtesting multiserie antes y después de #1530 (series anchas y en diccionario, varias
exógenas, refit, `skip_folds`, `n_jobs=2` e intervalos). En todos, cero diferencias en pandas
2 y 3, y los datos del usuario quedan intactos. En la parte 4 se reprodujo cada error con el
código de la 0.27 y se comprobó su arreglo en pandas 2 y 3: exógenas con `RangeIndex`
(reseteada, con el histórico, en orden descendente, en diccionario y con series que terminan
en posiciones distintas); `ForecasterRnn` (los pesos iniciales no cambian al entrenar, la
copia del backtesting vuelve a ellos, dos ajustes con la misma semilla coinciden,
`set_params`, forecasters guardados con la 0.27 y cinco tipos de función de pérdida, que se
comportan como en la 0.27; solo con pandas 2 y el backend torch); el backtesting del
clasificador (igual con `n_jobs` 1 y 2, y igual que ajustar a mano cada fold), y las clases
que solo están en los lags con seis estimadores. En la parte 5 se compararon el código de
antes y el de después de cada pull request en pandas 2 y 3, con salidas idénticas entre las
dos versiones. Para la exógena como matriz float: el tipo de la matriz, el tiempo de `fit`,
los errores de HistGradientBoosting y de `pd.NA` (también con la 0.27), la `category` de
enteros con ausentes, los tipos de `create_train_X_y` y `create_predict_X`, multiserie con
exógena ancha y en diccionario, y un forecaster guardado. Para los `Pipeline`: LightGBM,
CatBoost, XGBoost e HistGradientBoosting sin envolver, en un `Pipeline` y detrás de un
escalador, en cuatro forecasters; la configuración propia del usuario; forecasters
guardados, y una búsqueda con `lags_grid`. Sin envolver, los resultados son idénticos.


## 5. Pendiente antes de publicar

Ordenado por importancia. Los identificadores son filas de `dev/ROADMAP.md`. Los tres P1 que
salieron de la revisión ya están resueltos: la carga sin `pytz` (#1536, sección 2.6), la
estimación inflada de `n_jobs='auto'` en la primera llamada (#1538, sección 2.7 c) y las
correcciones de documentación (DOC-028-review, #1539: "Before upgrading", la FAQ de pandas 2
y 3, la nota de la carpeta temporal de joblib, y el error de `reshape_series_wide_to_long`,
que ahora sugiere también `asfreq('B')`).

1. **BUG-rnn-old-pickle (P2, nuevo de la 0.28, muy barato).** Un `ForecasterRnn` guardado con
   la 0.27 da un `AttributeError` sin explicación en `fit`, `set_params` y el backtesting con
   `initial_train_size` (sección 2.9 b). Las notas de versión lo documentan, pero falta un
   error que diga que hay que crearlo de nuevo, y que salte antes de que el `fit` fallido deje
   el forecaster como no ajustado (hoy `predict` deja de funcionar hasta volver a cargarlo).
2. **BUG-backtesting-ytrue-missing-date (P2, previo, barato).** Si una fecha predicha no está
   en `y`, el backtesting toma la **última observación** como valor real y la métrica sale
   mal sin aviso. Reproducido con el caso siguiente.
3. **BUG-daily-utc-steps-ambiguous (P2, previo).** Datos diarios en pasos UTC a una hora que
   no es medianoche UTC (por ejemplo, 12:00 UTC en Madrid): si la ventana no contiene un
   cambio de hora, las fechas predichas que lo cruzan salen desplazadas una hora. Con pandas 2
   y `freq='D'`, el backtesting da métricas erróneas (MAE 3,80 en lugar de 3,54); con
   `freq='24h'` sale bien.
4. **BUG-utc-midnight-anchored-freq y BUG-wide-to-long-inferred-freq (P2, previos).** Dos
   casos en los que la convención o la frecuencia inferidas no son las de los datos, y
   `reshape_series_wide_to_long` o `expand_index` pierden valores o predicen otras fechas.
5. **BUG-skops-tz-dst (P2, previo).** skops guarda pero no carga un forecaster diario de UTC a
   hora local cuya ventana cruza un cambio de hora.
6. **BUG-ms-fit-exog-rangeindex (P2, previo).** El mismo problema de la sección 2.9 a, en
   `fit`: con series de `RangeIndex`, una exógena con el índice reseteado se alinea con otras
   filas sin aviso.
7. **PERF-one-step-ahead-inverse (P3).** La búsqueda one-step-ahead multiserie deshace la
   transformación con una llamada directa al transformador. Hoy es segura solo por copias
   implícitas; pasarla por `transform_numpy` la protegería y la haría de 2 a 2,6 veces más
   rápida. Es la parte que dejó fuera el camino rápido de `StandardScaler` (sección 2.8 b).
8. Menores de la parte 1: ISSUE-1507 (`calculate_distance_from_holiday` da horas con
   frecuencias semimensuales y de días laborables) y MAINT-docs-pandas3 (reejecutar cinco
   notebooks con pandas 3).
9. De la parte 2, ninguno impide publicar. Quedan como mejoras (P3 y P4): evitar la copia de
   los modelos con arrays grandes y revisar los umbrales de `n_jobs='auto'` por plataforma
   (PERF-n_jobs-next), reintentar sin la carpeta temporal (FE-njobs-temp-folder-retry), que
   los forecasters Direct no guarden el número de CPU de la máquina donde se crearon
   (BUG-direct-n_jobs-resolved-at-init) y un test inestable con `threading`.
10. De la parte 3, ninguno impide publicar: no hay fallos nuevos. Quedan dos fallos previos de
    la 0.27 que encontró #1530, los dos con datos no válidos y un mensaje de error poco útil:
    `predict` multiserie con `levels=[]`, ventana y exógena da un error de pandas
    (BUG-ms-predict-empty-levels-window, P4), y una exógena que mezcla float y una columna
    nullable con valores ausentes daba `TypeError` en lugar del error de valores ausentes.
    Este último ya está arreglado (#1542, sección 2.9 e). Mejoras para después:
    validar solo el primer fold del backtesting (PERF-predict-numpy) y usar la conversión
    nueva del formato largo en `reshape_series_long_to_dict` y `reshape_exog_long_to_dict`
    (PERF-reshape-long-to-dict).
11. De la parte 4, ninguno más impide publicar: no hay fallos nuevos que cambien resultados.
    Quedan fallos previos que encontraron las pull requests: `XGBClassifier` sigue fallando
    en `fit` si una clase solo está en los lags y no es la última
    (BUG-xgbclassifier-lag-only-class, P3), series de `RangeIndex` con paso negativo y exógena
    (BUG-ms-exog-negative-step, P4), y dos del módulo experimental (BUG-splitter-positional-cuts
    y BUG-splitter-pandas2-dst-union, P2). Como documentación, avisar de que con
    `SVC(probability=True)` la clase de `predict` y la de `predict_proba` (y la del backtesting)
    pueden no coincidir (DOC-classifier-predict-vs-proba, P4).
12. De la parte 5, ninguno impide publicar: no hay fallos nuevos. Quedan límites previos
    que encontraron las pull requests: CatBoost dentro de un envoltorio que no copia la
    matriz (un `Pipeline` sin otro paso, o `TransformedTargetRegressor`) falla en el
    `predict` multiserie y en los intervalos por bootstrapping
    (BUG-predict-readonly-wrapped-catboost, P2); los enteros mayores que 2^53 de una exógena
    que se convierte a float salen redondeados en `create_train_X_y` y `create_predict_X`
    (BUG-create-X-int-overflow, P3), y LightGBM con categóricas nativas y pocas filas puede
    dar un modelo peor que sin ellas (DOC-lightgbm-categorical-few-rows, P3).

**Fuera del alcance de esta revisión.** El roadmap fija para la 0.28.0 (selección del
2026-11-25) varios arreglos de fallos que ya estaban en la 0.27 y que no son regresiones:
el principal es BUG-ms-fit-dataframe-categorical (P1: en `ForecasterRecursiveMultiSeries`,
con `categorical_features=None` y una exógena `category` cuyos valores no coinciden con sus
códigos, LightGBM y XGBoost dan **predicciones erróneas sin aviso**). Antes, una medida con
datos reales de BUG-catboost-refit-categorical (CatBoost con categóricas que se reajusta
empeora sin aviso; P1 si se confirma), y después, BUG-predict-readonly-wrapped-catboost y el
plan de BUG-04. #1542 y #1544 eran requisitos del primero.


## 6. Alcance

La revisión cubre todas las pull requests de la 0.28 que cambian código de la librería. Las
restantes (#1494, #1518, #1521, #1523 a #1526, #1528, #1531 y #1545) no lo cambian. #1536,
#1538, #1539 y #1540 se unieron durante la revisión y están incluidas en sus partes; #1542 y
#1544, después de terminarla, y forman la parte 5.
