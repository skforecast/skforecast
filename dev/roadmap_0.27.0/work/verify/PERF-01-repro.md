# PERF-01: verificación

**Veredicto: keep (esfuerzo M), con alcance acotado.** Se confirma el cuello de botella: `_backtesting_foundation` llama al modelo una vez por fold y prepara cada serie en cada fold con slicing por etiquetas de pandas. Agrupar los folds en una sola llamada da 3.4x en el caso más común (una serie, muchos folds) con el mismo resultado (dif. máx. 3.8e-06, misma MAE), pero en CPU la ganancia se diluye rápido al aumentar el número de series (1.4x con 5 series, 1.1x con 50). La segunda parte (sobrecarga pandas) solo pesa con miles de series: 0.87 ms por serie y fold sin exog y 1.33 ms con exog, que el prototipo baja a 0.14 y 0.41 ms. Sin solape con `perf/ets-numba-compile` (solo toca `stats/exponential_smoothing/_ets_base.py` y `releases.md`).

Script: `dev/roadmap_0.27.0/repro/PERF-01.py` (incluye el prototipo `backtesting_foundation_batched`). Entorno: Python 3.12.3, numpy 2.5.4, pandas 2.3.3, numba 0.68.0, statsmodels 0.15.0, torch 2.14.1+cpu (4 hilos), chronos-forecasting 2.3.2, transformers 5.20.1, skforecast 0.26.0; 4 CPU, sin GPU.

**Aviso:** otros agentes comparten las 4 CPU (carga media entre 1.3 y 3 durante las medidas buenas). Una primera ejecución del caso C con carga 8 dio 103 s frente a 38 s y se descartó; la que se muestra se repitió con carga ~2. Mediana de 3 repeticiones. numba no está en este camino (no hay `njit` en `foundation/` ni en `model_selection/_utils.py`), así que no hay tiempo de compilación que separar; la carga del modelo y la primera predicción (3.3 s) se miden aparte.

## Resultados medidos

| Caso (chronos-2-small, CPU) | Actual | Agrupado (prototipo) | Mejora | Igualdad |
|---|---|---|---|---|
| A. 1 serie horaria, 6000 obs, 200 folds x 24 pasos, L=512 | 5.68 s [5.66, 7.66, 5.68] | 1.68 s [1.68, 1.60, 1.75] | 3.4x | 4800 filas, dif. 3.8e-06, MAE 0.811941 en ambos |
| C. 5 series horarias, 100 folds x 24, L=512 | 5.66 s | 4.00 s | 1.4x | dif. 3.8e-06 |
| B. 50 series diarias, 30 folds x 7, L=256 | 7.27 s | 6.50 s | 1.1x (dentro del ruido) | dif. 2.9e-06 |

En A, una llamada a `predict` con un solo contexto cuesta 28.7 ms; 200 llamadas suman 5.74 s, prácticamente todo el backtesting actual: con una serie la sobrecarga pandas es despreciable y el coste es la llamada al modelo con lote de tamaño 1.

| Sobrecarga sin modelo (adapter.predict simulado), 2000 series diarias, 10 folds x 7 | Actual | Prototipo | Mejora |
|---|---|---|---|
| Sin exog | 17.45 s (1.74 s/fold, 0.87 ms por serie y fold) | 2.70 s (0.14 ms) | 6.5x |
| Exog de 2 columnas | 26.69 s (2.67 s/fold, 1.33 ms por serie y fold) | 8.13 s (0.41 ms) | 3.3x |

Como referencia, el modelo en CPU cuesta unos 4.3 ms por serie y fold (caso B), así que con miles de series la sobrecarga es un 17 a 24% del total en CPU. En GPU el modelo es mucho más rápido y la sobrecarga dominaría (estimación, no medido: no hay GPU).

## Cuello de botella (cProfile, `prof_*.txt` en el scratchpad)

- Una llamada al modelo por fold: `skforecast/model_selection/_validation.py:2521` (bucle) y `:2591-2609` (`predict_quantiles`/`predict`). La candidata citaba 2577-2594.
- Sin exog, la mitad del tiempo perfilado está en `_extract_data_folds_multiseries` (`model_selection/_utils.py:1022-1034`: `isna().to_numpy().all()`, `first_valid_index`/`last_valid_index` sobre todo el histórico en cada fold, y un `pd.DataFrame(series_last_window)` por fold que el backtesting foundation no usa). El resto es la reconstrucción del contexto en `_validation.py:2540-2553`, que repite el trabajo. Entre ambas suman 100 000 slices `.loc` por etiqueta y 82 000 `notna()` para 20 000 pares serie y fold.
- Con exog, además, `FoundationModel._prepare_future_exog` (`foundation/_foundation_model.py:870,877`): `expand_index` (un `date_range`) y `isnull().any(axis=None)` por serie, y otro `expand_index` por serie al construir el índice largo (`:1261`). Esto también afecta a `predict` normal con muchas series.

## Optimización propuesta

1. En `_backtesting_foundation`, sustituir `_extract_data_folds_multiseries` por un cálculo vectorizado por serie (una vez): posiciones de los límites de cada fold con `index.searchsorted`, primer valor válido y sumas acumuladas de `notna` para decidir con aritmética si la serie tiene datos en el contexto y en el test; después, `iloc` posicional.
2. Construir los contextos (y `context_exog`/`exog`) de todos los folds como pseudo-series con clave `(serie, fold)` y llamar a `FoundationModel.predict` en trozos (constante interna, p. ej. 1024 pseudo-series, para no inflar memoria en adaptadores que crean un array denso como T0); repartir por fold y recortar `gap`. Excepciones: con `cross_learning=True` (Chronos) las series del lote comparten información, así que mezclar folds de la misma serie filtraría el futuro: mantener una llamada por fold. El último fold incompleto (`allow_incomplete_fold`) tiene menos pasos: va en una llamada aparte.
3. En `FoundationModel.predict`/`_prepare_future_exog`: cachear `expand_index` por última marca temporal (todas las pseudo-series de un fold comparten horizonte) y comprobar NaN con numpy. Beneficia también a `predict` con miles de series.

Ganancia esperada: 3 a 3.5x en backtesting y `bayesian_search_foundation` de una serie (por trial), 1.1 a 1.4x con varias series en CPU, y 3 a 6x en la parte de preparación con miles de series. Los adaptadores que llaman al backend una vez por lote (Chronos-2, TimesFM 2.5/3.0, Moirai-2, T0, TS-ICL) se benefician del punto 2; TabICL, TabPFN-TS y Nori recorren las series una a una y solo ganan con los puntos 1 y 3. Solo se ha medido Chronos-2.

Riesgos y pruebas: tests con adaptador simulado que comparen salida agrupada y por fold (con `levels`, folds omitidos, `gap`, último fold incompleto, exog, series que empiezan tarde con contextos de longitud distinta y `cross_learning=True`). Sin cambios de API.
