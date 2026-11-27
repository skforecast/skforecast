# BUG-ms-fit-dataframe-categorical: revisiones y prototipos

Material de la sesión del 2026-11-26 y 2026-11-27 sobre la rama
`fix/ms-fit-numpy-estimator` (el estimador de `ForecasterRecursiveMultiSeries`
recibe un array numpy en `fit`). El estado, las decisiones y los pasos
siguientes están en [`dev/handoff_ms_fit_categorical.md`](../../handoff_ms_fit_categorical.md).

## Contenido

| Carpeta | Qué es |
|---|---|
| `informes/01_revision_rama_*.md` | Primera revisión independiente de la rama (corrección, rendimiento, si es una mejora). Dos revisores sin el contexto de la sesión, uno con Fable 5.2 y otro con Opus 5.5 |
| `informes/02_dudas_diseno_*.md` | Segunda revisión, sobre tres dudas de diseño: si la gestión de categóricas es demasiado amplia, si conviene crear las matrices directamente en numpy, y si conviene arreglar primero con pandas. Los mismos dos modelos, sin las conclusiones del autor |
| `arreglo_pandas/` | El arreglo mínimo con pandas que implementó cada revisor sobre la base (`a7c7a869d`), como diff. `fable_variante_codigos.diff` se aplica sobre `fable_minimo.diff`: no convierte la columna cuando sus códigos ya son sus valores |
| `scripts_dudas_fable/`, `scripts_dudas_opus/` | Scripts de la segunda revisión: matriz de escenarios sobre base, rama y arreglo mínimo, casos sueltos, parámetros por nombre, ida y vuelta de dtypes, copia en la frontera. Incluye los resúmenes de la matriz (`summary_*.md`) y las salidas pequeñas (`extra*.txt`) |
| `scripts_revision_fable/`, `scripts_revision_opus/` | Scripts de la primera revisión: batería de equivalencia entre base y rama, medidas de tiempo y memoria, guardado y carga, búsquedas |

El script que reproduce la regresión de los parámetros por nombre que no son
categóricos está en [`../repro/ms_byname_params.py`](../repro/ms_byname_params.py).

## Cómo leerlo

- Los informes citan rutas de la carpeta temporal de la sesión como
  `<scratchpad>/...`. Equivalencias: `<scratchpad>/indep_review` es
  `scripts_revision_fable/`, `<scratchpad>/indep_review_opus` es
  `scripts_revision_opus/`, `<scratchpad>/doubts_a` es `scripts_dudas_fable/` y
  `<scratchpad>/doubts_b` es `scripts_dudas_opus/`.
- No se conservan las copias de los árboles (base, rama, arreglo mínimo), los
  registros de pytest, los resultados en JSON ni los pickles de las baterías.
  Los informes recogen sus cifras.
- Los scripts son material de referencia. Los lanzadores (`*.sh`) tienen
  `<scratchpad>`, `<repo>` y `<conda>` en lugar de las rutas locales, y los
  scripts esperan `PYTHONPATH` apuntando al árbol que se quiere probar.
- Los diff de `arreglo_pandas/` son la referencia para reimplementar el
  arreglo, no un parche para aplicar tal cual: son prototipos sin tests propios
  y sus primeras versiones rompían CatBoost con `cat_features` en el
  constructor.
- La carpeta `base028` que citan los informes no era idéntica al HEAD: dos
  ficheros de test (`test_predict.py` y `test_set_in_sample_residuals.py` de
  multiserie) eran los de la rama. El código fuente sí coincidía, y los dos
  revisores de la segunda revisión usaron los tests del HEAD.
