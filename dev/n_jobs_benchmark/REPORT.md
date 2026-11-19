# Estrategia de `n_jobs='auto'`: informe final

> **Estado a 2026-11-19.** Las propuestas 1 a 3 están fusionadas en `0.28.x` (#1503), y
> los fallos previos, en #1511, #1513, #1514 y #1515 (salvo `XGBClassifier` con una clase
> solo en los lags). Lo pendiente (siguientes pasos, propuestas 4 a 6, umbral en macOS)
> está en [`dev/ROADMAP.md`](../ROADMAP.md), filas PERF-n_jobs-next, NJOBS-cpu-count,
> DOC-njobs-faq y BUG-xgbclassifier-lag-only-class. El 2026-11-19 se borraron `NOTES.md`,
> `prototype_serialization/` y los datos en bruto de `results/` (`results.jsonl` y
> `machine.json`): se leen con `git show 3c0711b8a:dev/n_jobs_benchmark/<ruta>`. Se
> conservan los `README.md`, `summary.md` y scripts de comparación de `results/` y los
> scripts de medida (los de comparación necesitan recuperar antes su `results.jsonl`).

13 de noviembre de 2026. Sustituye a las conclusiones provisionales de
`NOTES.md`, que conserva el detalle del prototipo de serialización.

**Estado (15 de noviembre de 2026).** Las propuestas 1 (también en el
backtesting multiserie), 2 y 3 (versión mínima) están implementadas en la rama
`perf/n-jobs-auto-strategy`. Las secciones Conclusión, Causa común y
Propuestas describen el comportamiento anterior a esos cambios; el resultado
de cada propuesta está en sus apartados "Medido" y en `results/*/README.md`
(Linux en la EC2 y Windows en `results/win-22cpu/`). El resto de propuestas
queda para pull requests posteriores.

**Revisión final (16 de noviembre de 2026).** Una revisión independiente
encontró que la serialización única hacía hasta 10 veces más lentos los
modelos con arrays grandes (KNN, RandomForest), porque joblib no podía
compartir los bytes como memmap. Se arregló en la rama con cinco commits: el
forecaster serializado viaja como array de numpy; no se serializa si joblib
usa un proceso; se deserializa antes de registrar los avisos del fold; y
`backtesting_stats` también serializa una vez (con hilos, sus folds
compartían el forecaster). Medido en `results/local-4vcpu-review/README.md`.
Lo que queda fuera está en "Siguientes pasos" y en "Fallos previos
encontrados en la revisión final".

## Conclusión

`n_jobs='auto'` decide bien: en los casos medidos queda entre un 2 % y un 6 %
del mejor resultado posible eligiendo entre un proceso y `cpu_count() - 1`.
Ninguna regla actual debe cambiarse por otra regla. Lo que se puede mejorar
está debajo de las reglas: el forecaster ajustado se serializa una vez por
fold, desde un hilo secundario. Eso explica los tres casos en los que `'auto'`
pierde, y deja además los ajustes posteriores de XGBoost hasta 3,8 veces más
lentos durante el resto de la sesión.

## Datos

- Máquina: EC2 r5a.4xlarge, AMD EPYC 7571, 16 CPU (8 núcleos), 124 GB,
  Ubuntu 26.04. Tamaños de 8 y 4 CPU emulados con la afinidad.
- Python 3.12.13, skforecast 0.27.0 (commit `6fa488c`), numpy 2.5.4,
  pandas 2.3.3, scikit-learn 1.9.1, joblib 1.6.0, lightgbm 4.7.0,
  xgboost 3.4.1, catboost 1.2.10, numba 0.68.1.
- Datos en `results/ip-10-2-1-64/` (nombre de la máquina): `results.jsonl`
  con cada medida y `summary.md` con las tablas.
- 724 medidas, ninguna fallida. Etapas 1 a 6 completas: los casos principales
  tienen dos repeticiones en los tres tamaños, y los casos extendidos
  (búsqueda con refit, multiserie, variantes de hilos, número de procesos),
  una repetición en los tres tamaños.
- Calidad: en las 352 medidas en paralelo, la primera llamada arrancó sus
  procesos y las siguientes reutilizaron los mismos. CPU ocupada por otros
  procesos: 3,2 % como máximo. Las dos repeticiones difieren un 0,4 % de
  mediana y un 1,7 % en el percentil 90, así que una diferencia del 5 % ya es
  real. Las predicciones son idénticas en serie y en paralelo.

Tiempos en segundos. "En frío" es la primera llamada de un proceso nuevo (un
script); "en caliente", las siguientes (un notebook).

## Suma de todos los casos

| CPU | Llamada | Casos | Siempre 1 | Siempre en paralelo | `'auto'` | El mejor de los dos | `'auto'` frente al mejor |
|---|---|---|---|---|---|---|---|
| 16 | En frío | 35 | 576,6 | 605,6 | 409,0 | 388,3 | +5,3 % |
| 16 | En caliente | 34 | 569,8 | 445,6 | 369,7 | 348,7 | +6,0 % |
| 8 | En frío | 37 | 646,8 | 547,7 | 446,6 | 438,6 | +1,8 % |
| 8 | En caliente | 37 | 646,5 | 500,7 | 419,0 | 408,8 | +2,5 % |
| 4 | En frío | 39 | 852,0 | 786,6 | 692,3 | 667,0 | +3,8 % |
| 4 | En caliente | 39 | 851,8 | 709,7 | 656,5 | 634,6 | +3,5 % |

Paralelizar siempre llega a tardar 17 veces más en un caso (LightGBM con sus
hilos). Usar siempre un proceso, hasta 5 veces más.

## Reglas que aciertan

| Regla | Dato (serie frente a paralelo) |
|---|---|
| Lineales, un proceso | Ridge con refit: 0,14 frente a 2,4 en frío |
| LightGBM con sus hilos, un proceso | 5,2 frente a 92,2 (6 folds con refit, 16 CPU) |
| Sin refit, un proceso: LightGBM de un hilo | 3,3 frente a 12,7 en frío y 7,4 a 8,7 en caliente |
| Sin refit, un proceso: RandomForest | 21,9 frente a 36,3 en frío y 35,0 en caliente |
| Sin refit, un proceso: XGBoost | 2,9 frente a 6,0 (16 CPU); 4,1 frente a 6,2 (4 CPU) |
| Sin refit, un proceso: CatBoost, backtesting suelto | 3,9 frente a 5,5 en frío (2,4 en caliente) |
| Con refit, en paralelo: HistGB | 60,2 frente a 15,8 (16 CPU), 20,0 (8) y 32,9 (4) |
| Con refit, en paralelo: LightGBM de un hilo | 44,6 frente a 11,6 (16 CPU), 13,9 (8) y 20,0 (4) |
| Con refit, en paralelo: RandomForest, 6 folds | 116,5 frente a 42,2 (16 CPU) y 61,3 (4) |
| Con refit, en paralelo: CatBoost | De 0,98 a 1,5 veces: gana poco, no pierde |
| Excepción de HistGB en búsquedas sin refit | 26,8 frente a 20,7 en frío y 18,6 en caliente |
| Umbral de carga de 2 s | Punto de equilibrio medido: de 3,0 a 4,3 s |
| `min(n_jobs, n_tasks)` | 6 folds, HistGB: 9,3 y 7,2 con 6 procesos; 10,0 y 9,7 con 15 |
| `cpu_count() - 1` frente a la mitad de los CPU | Igual o mejor en todos los casos con refit |

Sobre el umbral: entre 2 y 4 s de trabajo, `'auto'` paraleliza y en frío
pierde menos de 1 s, y en caliente gana. Subirlo a 3 s cambiaría un resultado
por el otro. No hay motivo para tocarlo.

Arrancar los procesos cuesta 2,2 s con 2 a 8 procesos y 3,1 s con 15, igual
con o sin window features.

## Dónde pierde `'auto'`

| Caso | `'auto'` | La alternativa | Tiempo de más |
|---|---|---|---|
| XGBoost, búsqueda con refit (3 candidatos, 6 folds, 16 CPU) | 46,7 (paralelo) | 32,0 (serie) | 46 % |
| La misma búsqueda, 4 CPU | 63,5 (paralelo) | 54,1 (serie) | 17 %; con 8 CPU empatan (39,0 y 39,1) |
| XGBoost, 6 folds con refit, 4 CPU | 21,7 (paralelo) | 17,9 (serie) | 21 % |
| XGBoost, 6 folds con refit, 16 CPU, en caliente | 12,3 (paralelo) | 10,6 (serie) | 17 % |
| XGBoost, multiserie con refit, 16 CPU, en caliente | 12,7 (paralelo, medido con 15 procesos; `'auto'` usa 8) | 9,3 (serie) | 36 % |
| XGBoost, multiserie con refit, 4 CPU | 14,9 (paralelo) | 10,5 (serie) | 42 %; con 8 CPU gana 1,2 veces |
| HistGB, backtesting suelto sin refit (100 folds) | 18,9 (serie) | 13,9 en frío, 11,8 en caliente | 36 % en frío, 60 % en caliente |
| CatBoost, búsqueda sin refit, en frío | 9,3 (paralelo) | 7,7 (serie) | 10 a 16 %; en caliente gana del 9 al 23 % |

Con XGBoost, paralelizar sí gana en el backtesting con refit de 16 folds: 1,57
veces con 16 CPU, 1,22 con 8 y 1,04 con 4. En el resto de casos con refit
empata o pierde, en los tres tamaños.

**Coste oculto.** Después de un backtesting en paralelo, los ajustes de XGBoost
en el proceso principal tardan 3,8 veces más con 16 CPU, 2,1 con 8 y 1,4 con
4. Dura el resto de la sesión, así que afecta también al ajuste final del
modelo que el usuario haga después. Es lo que hunde la búsqueda con refit:
cada candidato hace su primer ajuste en el proceso principal. Con LightGBM con
sus hilos pasa lo mismo (hasta 9 veces), pero `'auto'` nunca lo paraleliza.

**Sin refit, paralelizar no escala.** La fase de predicción de HistGB tarda lo
mismo con 3, 7 y 15 procesos. Con LightGBM de un hilo tarda 10 veces más en
paralelo que en serie.

## Causa común

En `_validation.py`, cada fold recibe el forecaster ajustado como argumento.
joblib lo serializa una vez por fold, y lo hace desde un hilo secundario.

- El coste por fold explica que la predicción no escale.
- Serializar un modelo de XGBoost o LightGBM desde ese hilo crea un segundo
  equipo de hilos de OpenMP, y eso ralentiza los ajustes posteriores.

El prototipo de `prototype_serialization/` lo confirma (4 vCPU, 3 procesos,
100 folds, fase de predicción):

| Estimador | En serie | En paralelo, hoy | Serializado una vez | Una vez y reutilizado por worker |
|---|---|---|---|---|
| HistGB | 9,09 | 5,78 | 3,92 | 3,65 |
| LightGBM de un hilo | 0,31 | 3,50 | 0,52 | 0,25 |
| XGBoost | 0,66 | 0,51 | 0,68 | 0,38 |
| CatBoost | 1,02 | 0,49 | 0,45 | 0,42 |
| RandomForest | 1,24 | 5,87 | 4,98 | 0,80 (como array compartido) |

Serializando en el hilo principal, el ajuste posterior de XGBoost deja de
ralentizarse (de 1,64 veces a entre 0,98 y 1,06) y el de LightGBM también (de
4,64 a entre 0,85 y 1,03).

## Propuestas, por impacto

### 1. Serializar el forecaster una vez, en el hilo principal

- **Qué:** en el backtesting, convertir el forecaster a bytes una sola vez
  antes de `Parallel` y enviar esos bytes a los folds. En los folds con refit
  basta el forecaster sin ajustar.
- **Qué arregla:** el coste oculto de XGBoost (toda la tabla de pérdidas de
  XGBoost) y el coste por fold sin refit.
- **Lo que no está medido:** el prototipo llama a `forecaster.predict`, no al
  backtesting real, y no cubre los folds con refit. La ganancia en los casos de
  XGBoost con refit es una predicción razonada, no una medida.
- **Riesgo:** bajo en la variante simple (bytes por llamada, sin reutilizar en
  el worker). La reutilización por worker da más, pero retiene memoria y
  necesita invalidación: mejor dejarla para una segunda fase.
- **Validación:** este mismo benchmark antes y después, en la máquina de 16
  CPU. Las predicciones deben ser idénticas.
- **Medido (rama `perf/serialize-forecaster-once`, variante simple):**
  predicciones idénticas; medidas en paralelo un 21 % más rápidas en caliente
  con 16 CPU y un 13 % con 8; ajuste posterior de XGBoost de 6,4 a 1,7 s; la
  búsqueda con refit de XGBoost pasa de 45,1 a 31,0 s en caliente (un proceso:
  31,5). Detalle en `results/ec2-proposal1/README.md` y, en 4 vCPU, en
  `results/local-4vcpu-proposal1/README.md`.
- **Medido en multiserie (rama `perf/n-jobs-auto-strategy`, commit e3eadd9):**
  el mismo cambio en `_backtesting_forecaster_multiseries`. Con XGBoost y 16
  CPU, en caliente, el backtesting sin refit en paralelo pasa de 8,9 a 3,2 s y
  con refit de 10,1 a 5,4 (lo que elige `'auto'`); el ajuste posterior de
  XGBoost deja de ser hasta 5,3 veces más lento. Sin refit, en frío, un proceso
  sigue ganando, así que `'auto'` no cambia. Predicciones idénticas. Detalle en
  `results/ec2-ms-serialization/README.md`.
- **Medido tras la revisión final (4 vCPU, `n_jobs=3` sin refit):** con los
  bytes como array de numpy que joblib comparte como memmap, RandomForest de
  73 MB con 480 folds pasa de 89 a 29 s (0.28.x llenaba `/dev/shm`) y KNN de
  38 MB con 480 folds de 43,7 a 10,6 s en frío (0.28.x: 5,1). LightGBM con 2000
  folds, 10,8 s (0.28.x: 116,6). Detalle en
  `results/local-4vcpu-review/README.md`.

### 2. No arrancar más procesos que folds, también en las búsquedas y con `n_jobs` entero

- **Qué:** hoy `min(n_jobs, n_tasks)` solo se aplica al backtesting suelto con
  `'auto'`. Las búsquedas y un `n_jobs` entero usan todos los procesos aunque
  haya menos folds.
- **Evidencia:** con 6 folds y 16 CPU, 15 procesos tardan 9,7 en caliente y 6
  procesos 7,2 (HistGB); con LightGBM de un hilo, 8,1 frente a 6,2. La búsqueda
  de 3 candidatos de 6 folds paga 5 a 6 s de arranque en lugar de 3. Además,
  con menos procesos cada worker recibe más hilos.
- **Ganancia esperada:** del 10 al 25 % cuando hay menos folds que CPU; ninguna
  en el resto.
- **Hipótesis sin verificar:** el coste extra viene de que, con más procesos
  que folds, cada llamada cae en workers que aún no han importado skforecast.
- **Riesgo:** bajo. El número de folds es constante dentro de una búsqueda, así
  que no provoca reinicios de los workers. Falta medirlo directamente en una
  búsqueda.
- **Medido (rama `perf/n-jobs-auto-strategy`, commit 8e9223b, sobre las
  propuestas 1 y 3):** `n_jobs = min(n_jobs, folds)` en los dos backtestings,
  así que cubre las búsquedas y el `n_jobs` entero. Con 16 CPU, la búsqueda con
  refit de 3 candidatos de 6 folds con `n_jobs=15` (lo que resuelve `'auto'`)
  pasa de 15 procesos a 6 y es
  del 16 al 30 % más rápida en frío; en caliente, XGBoost pasa de 31,0 a
  23,6 s (un proceso: 32,8), HistGB de 22,5 a 21,2 y LightGBM de un hilo no
  cambia. El backtesting con `n_jobs=15` es del 25 al 37 % más rápido en
  caliente. La memoria de los workers baja a menos de la mitad. Con 8 CPU (7
  procesos frente a 6), sin diferencias. Los resultados encajan con la
  hipótesis de los workers sin estrenar y, en caliente, con los hilos extra
  que joblib da a cada worker. Detalle en `results/ec2-proposal2/README.md`.

### 3. Sin refit: decidir por carga en lugar de por lista de estimadores

- **Qué:** sustituir "sin refit, un proceso" y la lista de excepciones (HistGB,
  CatBoost) por el tiempo de predicción del primer fold multiplicado por el
  número de folds.
- **Evidencia:** HistGB tarda del 36 al 60 % más en el backtesting suelto de
  100 folds, y en multiserie sin refit le pasa lo contrario: paralelizar pierde
  (4,3 frente a 8,2 en frío y 5,7 en caliente). Una lista por estimador no
  distingue los dos casos; la carga sí.
- **Depende de la propuesta 1:** con la serialización actual el umbral saldría
  distinto para cada estimador. Hay que hacerla después y calibrar con este
  benchmark.
- **Con la propuesta 1 medida:** el backtesting suelto de HistGB sin refit
  tarda 19,1 s con un proceso (lo que elige `'auto'`) y 8,0 en frío y 4,8 en
  caliente en paralelo con 16 CPU. XGBoost, LightGBM con un hilo y CatBoost
  siguen perdiendo en frío y ganan poco en caliente; RandomForest sigue
  perdiendo. Ver `results/ec2-proposal1/README.md`.
- **Medido (rama `perf/n-jobs-auto-prediction-workload`, versión mínima):**
  sin refit, con HistGB o CatBoost, se predice el primer fold en el proceso
  principal y se paraleliza el resto si la predicción estimada supera 6 s
  (Linux y macOS) o 12 s (Windows). HistGB con `'auto'` pasa de 19,2 a 8,1 s
  en frío y de 19,1 a 5,0 en caliente con 16 CPU; CatBoost sigue en un
  proceso. Detalle en `results/ec2-proposal3/README.md`.
- **Medido en Windows (22 CPU, `results/win-22cpu/README.md`):** con todos
  los cambios, la búsqueda de XGBoost con refit y `'auto'` pasa de 26,9 / 20,1
  s a 16,3 / 13,7 (en frío / en caliente), y el backtesting con refit y
  `n_jobs` entero tarda la mitad en caliente. Sin refit, el paralelo compensa
  en frío a partir de unos 7 s de predicción, pero se mantiene el umbral de
  12 s (una máquina, una repetición, riesgo asimétrico). Cerca del umbral, la
  decisión cambia entre llamadas: el primer fold de la primera llamada es más
  lento y empuja la estimación por encima de 12 s (21 procesos), y las
  siguientes vuelven a uno aunque los workers ya estén arrancados (8,0 s en
  lugar de 2,5). Mejora futura: no contar el arranque de los workers cuando ya
  están en marcha. El ajuste posterior de XGBoost no se ralentiza en Windows.

### 4. Decisión de criterio: `cpu_count()` en lugar de `cpu_count() - 1`

- **Evidencia:** con 16 folds y refit, usar todos los CPU en lugar de todos
  menos uno tarda menos en las máquinas de 8 y 4 CPU, que emulan un hilo por
  núcleo (segundos en frío):

  | Estimador | 8 CPU: 7 procesos | 8 CPU: 8 procesos | 4 CPU: 3 procesos | 4 CPU: 4 procesos |
  |---|---|---|---|---|
  | HistGB | 20,0 | 16,7 | 32,9 | 27,6 |
  | LightGBM de un hilo | 13,9 | 11,8 | 20,0 | 17,5 |
  | XGBoost | 28,2 | 21,4 | 44,9 | 37,5 |

  Con 16 CPU (dos hilos por núcleo), 15 y 16 procesos empatan.
- **Matices:** 16 folds se reparten exactos entre 8 y entre 4 procesos, lo que
  favorece a esa opción; con 6 folds y 4 CPU no habría diferencia. Y la máquina
  emulada conserva núcleos libres para el sistema, que un portátil real de 4
  núcleos no tiene.
- **Lo que se decide:** dejar un CPU libre mantiene el equipo usable mientras
  corre el backtesting, a cambio de un 15 a 32 % más de tiempo en máquinas sin
  hilos por núcleo. Es una decisión de criterio, no un error de la regla. La
  alternativa barata es documentar que `n_jobs=-1` da ese margen.

### 5. Documentación

- Con refit y 8 o más núcleos, LightGBM con `n_jobs=1` y los folds en paralelo
  gana a LightGBM con sus hilos: 8,6 frente a 14,0 con 16 CPU. Con 8 CPU
  empatan y con 4 ganan los hilos (15,3 frente a 17,8).
- HistGB predice a 7 ms por paso con 500 iteraciones, 28 veces más lento que
  LightGBM. En un forecaster recursivo sin refit, casi todo su tiempo es
  predicción.

### 6. Opcional: XGBoost con un hilo al predecir pocas filas

En Windows con 22 CPU ganaba del 22 al 32 %; en Linux, del 3 al 7 %. Falta
medir el umbral de filas en Windows
(`python extra/bench_xgb_predict_threads.py micro`).

## Qué no cambiar

- No añadir XGBoost a la lista de "un proceso" en Linux antes de medir la
  propuesta 1. Paralelizar gana hasta 1,57 veces con 16 folds; lo que falla es
  el coste oculto, y se arregla en su origen. Si la propuesta 1 no se hiciera o
  no bastara, sí habría que añadirlo: con refit, XGBoost en paralelo solo gana
  con claridad en 3 de las 12 combinaciones medidas de caso y tamaño de
  máquina.
- No añadir condiciones nuevas por sistema operativo.
- No detectar si los procesos ya están arrancados en esta fase: la ganancia
  se limita a cargas por debajo del umbral (1,8 frente a 0,5 s) y depende de
  la API privada de joblib. La medida de Windows (`results/win-22cpu/`, 8,0
  frente a 2,5 s cerca del umbral sin refit) la deja como posible mejora
  futura, sopesando esos dos inconvenientes.
- La excepción de CatBoost en búsquedas sin refit: pierde poco en frío y gana
  poco en caliente.

## Límites

- Linux: un solo hardware (AMD). Windows: un portátil de 22 CPU, con una
  repetición por medida y solo los casos de `bench_windows.py`
  (`results/win-22cpu/`); el informe original medía en Windows que
  paralelizar XGBoost ganaba 2 veces.
- macOS queda fuera. Lo que depende de joblib debería trasladarse; lo que
  depende de OpenMP (XGBoost y LightGBM en paralelo), no necesariamente.
- Un tamaño de datos (5.112 filas, 500 árboles). Las reglas dependen del
  trabajo por fold, que los casos cubren de 0,1 a 175 s.
- Los casos extendidos tienen una sola repetición por tamaño de máquina.
- Las máquinas de 8 y 4 CPU son emuladas: un hilo por núcleo, con el resto de
  la máquina real libre para el sistema.
- No se han medido los intervalos de predicción, los modelos estadísticos ni
  el refit intermitente.

## Siguientes pasos (fuera de esta pull request, sin decidir)

Ideas por evaluar, no compromisos. Cada una necesita sus propias medidas antes
de decidir si se hace.

- **Modelos con arrays grandes sin copiarlos en cada fold.** Tras la revisión
  final, KNN de 38 MB sin refit y con `n_jobs=3` sigue de 2 a 3 veces más
  lento que en 0.28.x en caliente (480 folds: 8,6 frente a 3,2 s;
  `results/local-4vcpu-review/README.md`): cada fold copia el modelo al
  deserializarlo, y 0.28.x enviaba sus arrays como memmaps de solo lectura que
  los workers usaban sin copiar. Afecta a los modelos que guardan arrays
  grandes (KNN, SVR, procesos gaussianos), con `n_jobs` entero y sin refit;
  con `'auto'` van en un proceso. Alternativa: pickle con protocolo 5 y
  buffers fuera de banda (`buffer_callback`), de modo que joblib comparta
  cada array grande como memmap y `pickle.loads(..., buffers=...)` lo use sin
  copiarlo. Con hilos habría que copiar los buffers, para que cada fold tenga
  su copia, y los arrays grandes volverían a ser de solo lectura en los
  workers, como en 0.28.x. Falta: prototipo y medidas.
- **No contar el arranque si los workers ya están en marcha.** En Windows,
  cerca del umbral sin refit, `'auto'` vuelve a un proceso en las llamadas
  siguientes aunque los 21 workers ya estén arrancados (8,0 s en lugar de 2,5;
  `results/win-22cpu/README.md`). Inconvenientes: depende del estado interno de
  loky (API privada) y haría que la decisión dependa de lo ejecutado antes en
  la sesión. Falta: diseñar cómo detectarlo y medir el beneficio en Linux y
  Windows.
- **Umbral de predicción en Windows (12 s).** En el portátil de 22 CPU el
  paralelo compensa en frío desde unos 7 s, pero con una repetición y mucho
  ruido. Falta: más máquinas Windows (con menos núcleos, con antivirus) y más
  repeticiones, con predicciones de unos 6, 8 y 10 s.
- **XGBoost con refit y pocos CPU.** Antes de los cambios, con 4 CPU, el
  paralelo perdía con 6 folds (21,7 frente a 17,9 s) y en multiserie (14,9
  frente a 10,5). Con 16 folds y 4 CPU ya ganaba tras la propuesta 1
  (`results/local-4vcpu-proposal1/`). Falta: repetir los casos de 6 folds y
  multiserie con 4 CPU sobre la rama actual antes de pensar en una regla.
- **Caché del forecaster en cada worker.** Evitaría deserializarlo en cada
  fold, a cambio de retener memoria en los workers y de necesitar invalidarla
  entre llamadas. Pesa con muchos folds cortos y `n_jobs` entero sin refit
  (con `'auto'` estos casos van en un proceso): según la revisión final,
  LightGBM con 2000 folds pasa unos 14 de sus 15 s deserializando (21 ms por
  fold), y KNN con 480 folds sigue 3 veces más lento que en 0.28.x, que le
  enviaba los arrays del modelo como memmaps sin copiarlos. Falta: diseñar la
  invalidación (por llamada) y medirlo.
- **`cpu_count()` frente a `cpu_count() - 1`** (propuesta 4). Decisión de
  criterio; la alternativa barata es documentar `n_jobs=-1`.
- **`mmap_mode='c'` de `backtesting_stats` con Sarimax.** Se añadió porque el
  filtro de Kalman de un Sarimax ajustado no se podía deserializar desde los
  memmaps de solo lectura de joblib. Ahora el forecaster viaja serializado una
  vez y se deserializa desde bytes, así que quizá ya no haga falta; se ha
  mantenido para `y` y `exog`. Quitarlo evitaría que loky reinicie los workers
  al alternar con otro `mmap_mode`. Falta: probar un Sarimax con refit y una
  serie de más de 1 MB sin él.
- **Reinicio de los workers de loky por el tope de folds.** loky solo reutiliza
  sus workers si no cambia el número de hilos que joblib da a cada uno
  (`cpu_count() // n_jobs`). Con el tope, ese número depende de los folds:
  dentro de una búsqueda no cambia (de ahí la ganancia de la propuesta 2), pero
  dos backtestings seguidos con distinto número de folds y el mismo `n_jobs`
  entero, o un `ForecasterDirect` que ajusta en paralelo con otro `n_jobs`,
  arrancan workers nuevos en cada llamada (revisión final, 4 CPU: unos 2 s
  frente a 0,2 s en 0.28.x; con 3 folds, en cambio, la rama ganaba). Falta:
  medir si compensa aplicar el tope solo cuando no cambia el número de hilos.
- **Backend de hilos y clases definidas en `__main__`.** Con un backend de
  hilos, cada fold deserializa el forecaster en el proceso principal, y
  cloudpickle vuelve a aplicar a la clase original el estado guardado de una
  clase definida en `__main__` (un estimador propio en un notebook): sus
  métodos se quedan con una copia de los globales del momento. Un cambio
  posterior de un global que usen no se ve. Alternativa: con hilos, copiar con
  `deepcopy` en lugar de serializar, lo que exige saber si el backend activo
  usa hilos (API privada de joblib).
- **Avisos de los folds terminados cuando otro fold falla.** Con varios
  procesos, si un fold lanza un error se pierden los avisos de los folds ya
  terminados (el error es el mismo). Con `'auto'` sin refit y HGB o CatBoost,
  donde 0.28.x usaba un proceso, se ven ahora menos avisos que con `n_jobs=1`.
  Alternativa: `return_as="generator"` y reemitir los avisos antes de relanzar
  el error.
- **XGBoost con un hilo al predecir pocas filas** (propuesta 6). Falta medir
  el umbral de filas en Windows.

## Fallos previos encontrados en la revisión final (fuera de la paralelización)

La revisión final de la pull request #1503 (16 de noviembre de 2026) encontró
tres fallos que ya estaban en 0.28.x y no dependen de la estrategia de
`n_jobs`, y al arreglar el primero apareció un cuarto. Cada uno va en su
propia pull request: a 16 de noviembre de 2026 están arreglados todos menos el
de `ForecasterRnn`, y queda una limitación de XGBoost en el de las clases que
solo están en los lags.

- **`ForecasterRecursiveClassifier` con clases que aparecen entre folds.** En
  `_backtesting_forecaster`, la columna `pred` se calcula con
  `forecaster.classes_` del proceso principal: con `n_jobs=1`, las del último
  refit; en paralelo, las del ajuste inicial. Las columnas de probabilidad de
  una clase que no existía en un fold quedan en NaN, y `argmax` elige esa
  clase en todo el fold. En el caso de la revisión, la exactitud es 0,40 con
  `n_jobs=1` y 0,67 con `n_jobs=2`. Arreglado en la pull request #1511: cada
  fold calcula `pred` con las clases de su modelo, y una clase que su modelo no
  conoce tiene probabilidad 0. También cubre el caso contrario, una clase que
  deja el entrenamiento (con `n_jobs` mayor que 1, se predecía en todo el fold).
- **`ForecasterRecursiveClassifier` con una clase que solo está en los lags.**
  Si una clase solo aparece en los primeros `window_size` valores del
  entrenamiento, el encoder la cuenta como clase y el estimador no la ve como
  objetivo: `predict_proba` falla con `ValueError: could not broadcast input
  array from shape (2,) into shape (3,)`. Lo reproducen un `fit` y un
  `predict_proba`, sin backtesting; en un backtesting con refit puede ocurrir
  cuando una clase está a punto de salir de la ventana de entrenamiento.
  Encontrado al comprobar la pull request #1511, que no lo arregla. Arreglado
  en la pull request #1514: `_recursive_predict` coloca las probabilidades del
  estimador en las columnas de sus clases (`estimator.classes_` son los
  códigos del encoder), da 0 a las que no conoce y elige la clase con el
  argmax de la fila completa. Antes, además, tomaba la posición de la columna
  como el código de la clase. El encoder no cambia: también codifica los lags.
  **Pendiente:** `XGBClassifier` exige que las clases del objetivo sean
  `0, 1, ..., k-1`. Si la clase que solo está en los lags no es el último
  código, falla en `fit`, antes de llegar a ese código (`ValueError: Invalid
  classes inferred from unique values of y. Expected: [0 1], got [0 2]`, con
  xgboost 3.4.1); si es el último, funciona desde #1514. Arreglarlo exige
  entrenar el estimador con los códigos del objetivo renumerados y deshacerlo
  al predecir: toca `fit`, las dos ramas de `_recursive_predict`, la búsqueda
  one-step-ahead y `create_train_X_y`, que es público (esfuerzo M).
- **`ForecasterRnn` con refit.** keras continúa el entrenamiento en cada
  `fit`: con `n_jobs=1`, cada fold parte de los pesos del anterior; en
  paralelo, de los del ajuste inicial (MAE de 3,16 frente a 3,61). Arreglo
  propuesto: reiniciar los pesos en `ForecasterRnn.fit`, o documentarlo.
- **`refit=0` y `n_jobs` negativos.** Con `n_jobs` entero, `refit=0` se trata
  como refit intermitente (avisa y usa un proceso), aunque
  `select_n_jobs_backtesting` ya lo normaliza a `False`. Los `n_jobs`
  negativos y `n_jobs=0` se convierten en `cpu_count()`: `-2` da 4 procesos
  con 4 CPU, donde joblib usaría 3, y `0`, que joblib rechaza, también da 4.
  Arreglado en la pull request #1513: `refit=0` se trata como `False`, y un
  `n_jobs` entero se resuelve con la regla de joblib
  (`max(cpu_count() + 1 + n_jobs, 1)`) en una función privada,
  `_resolve_n_jobs`. No se usa `joblib.effective_n_jobs`, cuyo resultado
  depende del backend activo. `n_jobs=0` lanza un `ValueError`, y las búsquedas
  lo comprueban antes de evaluar ningún candidato.
