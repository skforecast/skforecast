# Seguimiento: `ForecasterRecursiveMultiSeries` más rápido en `predict` y backtesting

Lo primero que hay que leer al retomar el trabajo. El plan completo está en [PLAN.md](PLAN.md) y las medidas en [MEDIDAS.md](MEDIDAS.md).

## Cómo retomar

1. Rama `perf/multiseries-predict-backtesting`, creada desde `origin/0.28.x` (`c694e1915`) sin upstream. Fila del roadmap: PERF-multiseries-names-pyarrow.
2. Mirar en la tabla de estado cuál es el siguiente cambio pendiente. Se implementa uno cada vez, en el orden de la tabla, siguiendo el "Ciclo de cada cambio" de `PLAN.md`.
3. Ningún cambio se da por hecho sin la revisión y el visto bueno de Joaquín. El commit lo hace él o lo pide de forma explícita para ese cambio. Sin push hasta que lo pida.
4. Preparar las medidas como explica [scripts/README.md](scripts/README.md). Los resultados de la batería de identidad sobre el árbol base se generan una vez y se reutilizan.
5. Al cerrar un cambio: actualizar su fila y, si hubo decisiones, el registro de abajo, dentro del mismo commit.

## Estado

Estados: pendiente, en revisión, hecho (con el hash del commit), descartado.

| # | Cambio | Función | Estado | Commit | Ganancia medida al implementarlo (20 series; p2 / p3 sin pa / p3 con pa) |
|---|---|---|---|---|---|
| 0 | Carpeta `dev/multiseries_predict_perf/` (plan, medidas, scripts) | | en revisión | | |
| 1 | No recortar el conjunto de entrenamiento en cada fold si sus límites no cambian | `_extract_data_folds_multiseries` | pendiente | | |
| 2 | No reordenar ni copiar la ventana si sus columnas ya son los niveles | `_create_predict_inputs` | pendiente | | |
| 3 | No seleccionar columnas de la ventana si ya son las que hay que comprobar | `check_predict_input` | pendiente | | |
| 4 | No realinear la exog de una serie ya alineada | `_create_predict_inputs` | pendiente | | |
| 5 | Repetir la exog ancha con numpy en vez de `pd.concat` | `_create_predict_inputs` | pendiente | | |
| 6 | Exog compartida por varias series: cortarla, comprobarla y compararla una vez | `_extract_data_folds_multiseries`, `check_predict_input`, `_check_exog_alignment`, `_create_predict_inputs` | pendiente | | |
| 7 | Crear la plantilla vacía de exog solo si hace falta | `_create_predict_inputs` | pendiente | | |
| 8 | Cortar cada serie de un formato largo sin el MultiIndex | `_split_long_format` | pendiente | | |
| 9 | No cortar la exog ancha si ya tiene `steps` filas | `_create_predict_inputs` | pendiente | | |
| 10 | Reutilizar el índice de columnas de la ventana | `_create_predict_inputs` | pendiente | | |
| 11 | Ventana guardada: no extraer la última fecha de cada serie si comparten índice | `preprocess_levels_self_last_window_multiseries` | pendiente | | |
| B2 | Calcular una vez la primera fecha esperada de la exog | `check_predict_input`, `_check_exog_alignment` | candidato, sin prototipar | | |
| B1 | Montar la última ventana de cada fold a partir de los valores | `_extract_data_folds_multiseries` | candidato, sin prototipar | | |
| B3 | Comprobar los NaN de la exog de cada serie sobre sus valores numpy | `check_predict_input` | candidato, sin prototipar | | |
| | Nota de versión y fila del roadmap | | pendiente | | |
| | Verificación final (identidad, tiempos, benchmark oficial, `verify`, suite completa p2 y p3, arnés) | | pendiente | | |

Dependencias: el 7 necesita el 4; el 6 necesita el 1 y el 4. El resto son independientes.

## Registro de decisiones

- **2026-11-17, alcance.** Solo cambios pequeños, nada de rediseñar `predict`. Condiciones: resultados, índices, dtypes (en especial la columna `level`), avisos y errores idénticos bit a bit; pandas 2 no puede empeorar en ningún caso.
- **2026-11-17, corte.** Un cambio entra si gana al menos un 3 % en dos de los tres entornos. Bloque A: con 20 series. Bloque B: con 100 series, informando también con 20, 600 y 1000.
- **2026-11-17, bloque B.** En la misma PR, después de verificar el bloque A. Cada candidato se prototipa y mide antes de decidir.
- **2026-11-17, forma de trabajo.** Un commit por cambio, con revisión detallada de cada uno antes de pasar al siguiente. Los tests de cada cambio van en su commit.
- **2026-11-17, una sola PR.** La suite completa con pandas 2 y 3 y el arnés se pasan una vez, al final.
- **2026-11-17, `ForecasterDirectMultiVariate`.** Solo recibe el cambio 3, que está en código compartido. Su `predict` no se toca.
- **2026-11-17, fuera por medida.** El `isna` repetido de la exog ancha (2,7 / 3,1 / 2,8 %), construir de otra forma la columna `level` de la salida (menos del 1 %) y memorizar la conversión de columnas por objeto (en pandas 3 cada corte de filas crea un índice de columnas nuevo).
- **2026-11-17, plan guardado en `dev/`.** Para retomarlo y actualizarlo entre sesiones. La carpeta se borra al fusionar la PR.

## Hallazgos sin tocar

- `predict` con `levels=[]`, ventana dada y exog lanza `No objects to concatenate`, un error de pandas, en lugar de uno propio. Los cambios conservan ese comportamiento. Candidato a issue.
- La batería del bloque anterior (`identity_trims.py`) da 653 casos con error en pandas 3 y 382 en pandas 2 sobre el árbol base. No viene de este trabajo; sin analizar.
- Rediseñar la extracción de folds sobre una matriz única de todas las series dejaría el coste por fold casi independiente del número de series. Es un rediseño; candidato para PERF-predict-numpy.
