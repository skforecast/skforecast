# Notas del estudio de `n_jobs`: hallazgos preliminares

Estado a 12 de noviembre de 2026. Son conclusiones provisionales, a la espera
de la segunda repetición del benchmark en la EC2. Sirven para retomar el
análisis final sin depender de la conversación en la que se hicieron.

## Origen

Se revisó un informe que comparaba `n_jobs='auto'` entre skforecast 0.26.0 y
0.27.0 en Windows (22 CPU) y proponía cambiar las reglas. La revisión concluyó
que sus tres problemas eran uno solo (la regla decide como si cada backtesting
arrancara los procesos, y el informe medía con los procesos ya arrancados) y
que hacía falta medir en frío y en caliente por separado. De ahí sale
`bench_n_jobs.py`.

## Ejecución en la EC2 (parcial)

- Máquina: r5a.4xlarge, AMD EPYC 7571, 16 CPU (8 núcleos con SMT), 124 GB,
  Ubuntu 26.04. Python 3.12.13, lightgbm 4.7.0, xgboost 3.4.1,
  catboost 1.2.10, scikit-learn 1.9.1, joblib 1.6.0, numpy 2.5.4, pandas 2.3.3.
- Resultados en `results/ip-10-2-1-64/`. Primera subida: 376 medidas (etapas 1
  a 3 completas, y parte de la 4), sin fallos, con el protocolo de frío y
  caliente cumplido en todas y predicciones idénticas en serie y en paralelo.
- La máquina se reinició en el minuto 228; se reanudó con 405 medidas
  pendientes. La mayoría de los casos tienen una sola repetición.

## Reglas de `'auto'` que aciertan

| Regla | Dato |
|---|---|
| Lineales, un proceso | Ridge en paralelo solo pierde |
| LightGBM con sus hilos, un proceso | En paralelo, 92 s frente a 5 s (6 folds con refit, 16 CPU) |
| Sin refit, un proceso (LightGBM de un hilo, RandomForest) | En paralelo tardan de 1,6 a 2,6 veces más |
| Umbral de carga de 2 s | Punto de equilibrio medido: de 3 a 4,3 s de trabajo real en serie |
| `min(n_jobs, n_tasks)` | Pedir 15 procesos para 6 folds cuesta unos 2 s más en cada llamada |
| Con refit, en paralelo (HistGB, LightGBM de un hilo, RandomForest) | De 2 a 5 veces más rápido |

Coste de arrancar los procesos: 2,2 s con 2 a 8 procesos, 3,1 s con 15. Con o
sin window features es el mismo. Cambiar el número de procesos entre llamadas
reinicia los workers (de 2,3 a 3,7 s) cuando cambia `cpu_count() // n_jobs`.

## Problemas encontrados

1. **XGBoost en paralelo en Linux.** Con refit gana de 1,04 (4 CPU) a 1,5
   veces (16 CPU) en un backtesting suelto, pero pierde en la búsqueda con
   refit (46,7 s frente a 32,0 s en serie) y con pocos folds (16,8 s frente a
   10,6 s en las llamadas siguientes). Tras una ejecución en paralelo, los
   ajustes de XGBoost en el proceso principal tardan 3,7 veces más con 16 CPU,
   2,1 con 8 y 1,4 con 4. En Windows el informe original medía lo contrario
   (paralelizar ganaba 2 veces).
2. **Sin refit, paralelizar no escala.** La fase de predicción de HistGB tarda
   9,7 s en paralelo con 3, con 7 y con 15 procesos (16,8 s en serie). Con
   LightGBM de un hilo, 6 s en paralelo frente a 0,6 s en serie.
3. **HistGB sin refit en un backtesting suelto:** `'auto'` usa un proceso y
   pierde (18,9 s frente a 13,9 s en frío y 11,8 s en caliente), igual en los
   tres tamaños de máquina.
4. **HistGB predice despacio:** 7 ms por paso con 500 iteraciones, 28 veces
   más que LightGBM. De los 19 s de su backtesting sin refit, 17 son
   predicción.

## Causa de fondo y prototipo

Hipótesis: el forecaster ajustado se serializa una vez por fold, y lo hace un
hilo secundario de joblib. Eso explica el coste por fold (problema 2) y la
ralentización posterior de XGBoost y LightGBM (problema 1).

Prototipo en `prototype_serialization/` (fuera de skforecast, 100 folds de 24
pasos, máquina Linux de 4 vCPU compartidas, 3 procesos, mejor de 2 llamadas
con los procesos arrancados, cada combinación en un proceso nuevo).

Segundos de la fase de predicción:

| Estimador | En serie | En paralelo, como hoy | Serializado una vez | Serializado una vez y reutilizado por worker |
|---|---|---|---|---|
| HistGB | 9,09 | 5,78 | 3,92 | 3,65 |
| LightGBM de un hilo | 0,31 | 3,50 | 0,52 | 0,25 |
| LightGBM | 0,40 | 1,17 | 0,61 | 0,25 |
| XGBoost | 0,66 | 0,51 | 0,68 | 0,38 |
| CatBoost | 1,02 | 0,49 | 0,45 | 0,42 |
| RandomForest | 1,24 | 5,87 | 4,98 | 5,00 |

RandomForest (40 MB) baja a 0,80 s si los bytes se comparten como un array de
numpy, que joblib pasa a los workers por un mapa de memoria. Con esa variante,
XGBoost da 0,36 s, CatBoost 0,37 s, HistGB 3,68 s y LightGBM 0,29 a 0,36 s.

Ajuste posterior a la ejecución en paralelo, frente al previo:

| Estimador | Como hoy | Serializado en el hilo principal |
|---|---|---|
| XGBoost | 1,64 veces más lento | Sin cambio (0,98 a 1,06) |
| LightGBM con sus hilos | 4,64 veces más lento | Sin cambio (0,85 a 1,03) |

Conclusión: la hipótesis se confirma. Serializar el forecaster una vez, en el
hilo principal, y enviar los bytes elimina el coste por fold y la
ralentización posterior. Con ello, paralelizar la predicción deja de perder en
todos los estimadores.

Límites del prototipo: llama a `forecaster.predict` por fold, no al
backtesting real; no cubre los folds con refit (habría que enviar el
forecaster sin ajustar); la reutilización por worker necesita diseño (memoria
retenida, invalidación); y el arranque en frío sigue existiendo, así que el
umbral de carga sigue haciendo falta.

## Otros hallazgos

- **CatBoost en búsquedas sin refit** (hoy en paralelo): pierde del 10 al 16 %
  en frío y gana del 9 al 23 % en caliente. Marginal con 3 candidatos de 40
  folds.
- **Hilos al predecir.**
  - XGBoost con un hilo: en Linux gana del 3 al 7 %, y solo con una fila. En
    Windows con 22 CPU ganaba del 22 al 32 % (`extra/`, script
    `bench_xgb_predict_threads.py`: 1,63 s con 22 hilos, 1,27 s limitando en
    cada llamada con threadpoolctl, 1,11 s con una copia del booster con
    `nthread=1`).
  - CatBoost con `thread_count=1`: un 15 % con 16 CPU hasta 100 filas, nada
    con 8 y 4.
  - LightGBM: sin cambio con una fila; con más filas ganan sus hilos.
  - HistGB: su límite actual de un hilo da 1,23 veces con 16 CPU. Limitar una
    vez por bucle en lugar de en cada llamada no aporta en Linux.
- **LightGBM, hilos frente a procesos** (16 folds con refit): con 16 CPU gana
  un hilo por modelo y folds en paralelo (8,6 s frente a 14,0 s); con 8
  empatan; con 4 ganan los hilos (15,3 s frente a 17,8 s).
- **Número de procesos:** `cpu_count() - 1` es tan bueno o mejor que la mitad
  y que todos los CPU.
- **Detectar los procesos ya arrancados:** la ganancia se concentra por debajo
  del umbral (hasta unos 3 s por llamada).

## Pendiente

1. Recibir los resultados finales de la EC2 y confirmar estas cifras con la
   segunda repetición.
2. Informe final con las propuestas ordenadas por impacto.
3. Propuesta principal: implementar la serialización única en el backtesting
   y medirla con este benchmark, antes y después, en la máquina de 16 CPU.
   Después, revisar si la regla "sin refit, un proceso" y la lista de
   excepciones pueden sustituirse por un criterio de carga.
4. Decidir sobre XGBoost en Linux si el punto 3 no se aborda.
5. Opcional: umbral de filas de XGBoost en Windows
   (`python bench_xgb_predict_threads.py micro`).
