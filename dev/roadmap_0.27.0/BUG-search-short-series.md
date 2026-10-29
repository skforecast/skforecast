# BUG-search-short-series: las búsquedas multiserie comparan candidatos sobre series distintas

Fecha: 2026-10-19. Base: rama `0.27.x` (con #1415 y #1418 fusionados).
Estado: **diseño propuesto (opción c2 con escalado propio de cada serie), prototipo validado
contra skforecast y medido con datos reales (M5). Quedan las mediciones de la sección 8.2
antes o durante la implementación.** Prioridad: **muy alta**; da resultados erróneos en
silencio en un uso habitual de los modelos globales.

> **2026-10-29: la propuesta c2 queda sustituida** por la de
> [`BUG-search-short-series-transparency.md`](BUG-search-short-series-transparency.md)
> (informar de cuántos puntos evalúa cada candidato y ordenar los avisos de las búsquedas).
> Este documento queda como investigación: el problema, los experimentos 1 a 5 y las
> opciones descartadas. c2 queda aparcada como posible opción explícita del usuario.

Actualizado el 2026-10-22 con su relación con #1432 (NaN en los predictores de los modelos
lineales y aviso de predicciones NaN en las métricas): secciones 2.2, 7, 7.1, 8.2 y 10.

Entorno de medida: Python 3.14.3, numpy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1,
lightgbm, xgboost y catboost del entorno local de desarrollo. Scripts en
`dev/roadmap_0.27.0/repro/BUG-search-short-series_*.py` (sección 9).

---

## 1. Resumen

- `grid_search_forecaster_multiseries`, `random_search_forecaster_multiseries` y
  `bayesian_search_forecaster_multiseries` evalúan cada candidato sobre los pares
  (serie, fold) que ese candidato puede predecir. Cuando los candidatos tienen distinto
  `window_size` (distintos `lags` o `window_features`), una ventana grande deja fuera las
  series cortas, o sus primeros folds, y **la métrica agregada que se optimiza se calcula
  sobre puntos distintos en cada candidato**. Pasa igual con `TimeSeriesFold` y con
  `OneStepAheadFold`.
- El usuario no se entera: `grid_search` y `random_search` silencian los avisos de
  skforecast después del primer candidato, y el one-step-ahead no avisa nunca.
- Medido: el ranking se invierte con datos sintéticos (sección 2.3) y con ventas reales de
  M5 (sección 6.4.2); con muchas series cortas la ventaja de una ventana larga se exagera
  hasta el doble o cambia de signo según los datos (sección 2.4).
- Propuesta: **opción c2**. Con estimadores que aceptan NaN (LightGBM, HGBR, XGBoost,
  CatBoost), predecir las series con menos historia que la ventana dejando a NaN los lags que
  faltan, en lugar de descartarlas. El entrenamiento no cambia. **Requisito medido: cada
  serie con algún valor se escala con su propio `transformer_series` aunque no aporte filas
  de entrenamiento.** Con el escalado actual de los niveles desconocidos (`_unknown_level`),
  las predicciones de las series cortas son mucho peores que un naive (sección 6.4.1). Todos
  los candidatos predicen todos los puntos y la comparación es justa sin tocar la lógica de
  las búsquedas. Con estimadores que no aceptan NaN, avisar pronto y una sola vez, sin
  fallback.

## 2. El problema

### 2.1 Mecanismo

1. `_extract_data_folds_multiseries` (`skforecast/model_selection/_utils.py`) deja fuera de
   un fold la serie que tiene `window_size` valores o menos en su entrenamiento (frontera
   fijada en #1418) o cuya última ventana es más corta que `window_size` o tiene NaN que el
   estimador no admite.
2. `window_size` depende del candidato: con `lags_grid = {3, 24}`, `lags=24` necesita 25
   valores en el entrenamiento de cada fold, `lags=3` solo 4.
3. Cada candidato se evalúa sobre lo que predice, y la búsqueda ordena por esa métrica.
4. One-step-ahead: `ForecasterRecursiveMultiSeries._train_test_split_one_step_ahead`
   aplica el mismo filtro (`levels_last_window`), así que el sesgo es el mismo.

### 2.2 Los avisos no llegan

- `backtesting_forecaster_multiseries` avisa desde #1415 de los pares (serie, fold) no
  predichos. Pero en `_evaluate_grid_hyperparameters_multiseries`
  (`skforecast/model_selection/_search.py`), después de evaluar el primer candidato se
  ejecuta `warnings.filterwarnings('ignore', ...)` para todas las categorías de skforecast.
  Si el primer candidato tiene una ventana corta, el aviso no sale nunca. En los
  experimentos 1 y 2: **0 avisos**.
- `bayesian_search_forecaster_multiseries` no silencia: el aviso puede repetirse en cada
  trial, con textos distintos (incluyen el `window_size`), así que Python no los agrupa.
- El one-step-ahead no tiene aviso.
- El aviso que añade #1432 sufre el mismo silenciado: el `MissingValuesWarning` de
  `_calculate_metrics_backtesting_multiseries` cuando hay predicciones NaN en fechas con valor
  real, con su número por serie. En grid y random solo sale para el primer candidato; en
  bayesian, en cada trial cuyo recuento cambie (va en el texto, así que Python no los agrupa).

### 2.3 Experimento 1: el ranking se invierte

Script `BUG-search-short-series_01_ranking_flip.py`. Dos series AR(1) largas sin
estacionalidad (S01, S10) y una serie corta y ruidosa (HARD, 80 valores, 13 en el
entrenamiento inicial). `Ridge(alpha=1e-6)`, `lags_grid = {"lags_3": 3, "lags_24": 24}`.

MAE (menor es mejor):

| candidato | S01 | S10 | HARD | `average` que optimiza la búsqueda |
|---|---|---|---|---|
| `lags_3` | **2.00** | **1.37** | 8.84 | 4.07 |
| `lags_24` | 2.49 | 2.41 | NaN (descartada) | **2.45** (la búsqueda elige este) |

`lags_3` es mejor en las series comunes, pero la búsqueda elige `lags_24` solo porque no
evalúa HARD. Con `OneStepAheadFold`: 1.79 frente a 4.19, la misma elección. 0 avisos en las dos.

### 2.4 Experimento 2: muchas series cortas

Script `BUG-search-short-series_02_options.py`. 5 series largas (desde 2016) y 15 cortas
(productos nuevos que empiezan entre 2019 y 2021, más ruidosas). Mensuales, estacionalidad
12, `Ridge`, `refit=True`, 3 folds de 12 meses desde 2021-07.

```
           2016      2017      2018      2019      2020      2021      2022      2023
long_0     #################################+++++++++++++++  train fold 0:  66 obs
new_00     .............................####+++++++++++++++  train fold 0:   9 obs
new_03     ............................#####+++++++++++++++  train fold 0:  10 obs
new_06     .....................############+++++++++++++++  train fold 0:  25 obs
```
(`.` sin datos, `#` entrenamiento del fold 0, `+` test)

| candidato | series con alguna métrica | puntos de test cubiertos | MAE medio por serie hoy |
|---|---|---|---|
| `lags_3` | 20/20 | 100% | 20.34 |
| `lags_12` | 20/20 | 90% | 12.02 |
| `lags_24` | 20/20 | 66% | **11.83** (gana) |

Dos conclusiones:

- `lags_24` se evalúa solo sobre el 66% de los puntos, los fáciles: series largas y folds
  en los que las series nuevas ya tienen historia.
- **Las 20 series tienen métricas en los tres candidatos.** El sesgo está en los pares
  (serie, fold), no en las series: mirar qué series tienen métricas NaN no lo detecta.

En el experimento 3 (HGBR, 6 semillas), la ventaja de `lags_24` sobre `lags_12` que da la
búsqueda de hoy va de -0.61 a +1.77, con signo opuesto al de una comparación justa en 2 de
las 6 semillas (sección 6.2). **El sesgo no tiene una dirección fija**: depende de qué
puntos fáciles o difíciles quedan fuera.

## 3. Cómo lo resuelven otras librerías

Fuentes: hilos del foro de Nixtla con respuestas de mantenedores (no documentación oficial)
y memoria para GluonTS (no confirmado).

1. **Fallar en lugar de descartar.** mlforecast: cada serie necesita más de
   `n_windows * h` muestras después de quitar los NaN de los lags. Según un hilo, 0.13.0
   descartaba las series cortas con un aviso y desde 0.13.2 lanza un error; el remedio es
   quitarlas antes ([hilo 1](https://linen.nixtla.io/t/13551184/hi-everyone-i-have-a-problem-while-running-the-ml-forecast-c),
   [hilo 2](https://linen.nixtla.io/t/9607843/this-message-was-deleted)).
2. **Rellenar y marcar.** neuralforecast: `start_padding_enabled=True` rellena con ceros el
   principio de las series cortas
   ([hilo](https://linen.nixtla.io/t/16816679/hi-team-i-wanted-to-update-my-earlier-question-with-some-add),
   [docs](https://nixtlaverse.nixtla.io/neuralforecast/common.base_windows.html)).
   GluonTS rellena el contexto y marca lo no observado con una máscara (sin confirmar).
3. **Conjunto de evaluación fijado por el problema**, como en competiciones tipo M5: todas
   las series necesitan predicción.

Principio común: **el conjunto de evaluación lo definen los datos o el problema, nunca el
candidato.** skforecast lo incumple porque descarta en silencio.

## 4. Criterios para elegir

1. Medir solo lo que el usuario pidió: las predicciones del propio candidato.
2. Nada invisible ni arbitrario: ningún componente oculto ni decisiones que el usuario no ha tomado.
3. Mismos puntos de evaluación para todos los candidatos.
4. Funcionar igual en grid, random y bayesian (en bayesian los lags se eligen dentro de cada
   trial, `search_space(trial)`, así que no se conocen antes de empezar).
5. Coste razonable (tiempo de entrenamiento, compatibilidad con los resultados actuales).

## 5. Opciones evaluadas y por qué se descartan

### A. Avisar desde la búsqueda (sin cambiar resultados)

Guardar por candidato qué series tienen métricas NaN y avisar una vez al final si no son
las mismas. **Descartada**:
- No detecta el sesgo por folds (experimento 2: 20/20 series en todos los candidatos).
- No es accionable: el usuario solo tiene el resultado de la búsqueda.
- Si se arregla la comparación, el aviso cambia de significado y hay que rehacerlo.

### a. Excluir el candidato que no cubre todos los puntos

**Descartada.** Con muchas series cortas, toda ventana larga deja algún punto sin cubrir.
En el experimento 2 solo sobrevive `lags_3`, el peor con diferencia (20.34 frente a ~12).

### B. Conjunto común: evaluar todos los candidatos sobre los pares que puede predecir el de mayor `window_size`

Experimento 2: elige `lags_12` (11.64 frente a 11.83). **Descartada**:
- Ignora para todos los candidatos el 34% de los puntos, justo el arranque de las series
  nuevas, que suele ser lo que más interesa.
- En bayesian el `window_size` máximo no se conoce de antemano. Recorrer el `search_space`
  en seco con un trial falso solo funciona con lags categóricos y espacios no condicionales.

### d. Fallback (por ejemplo naive) en los pares no cubiertos

Experimento 2: elige `lags_12` (12.53 frente a 13.48), con el 100% de los puntos.
**Descartada**:
- Cambia lo que se mide: "candidato + un naive que nadie ha pedido". El fallback es
  arbitrario y el ranking depende de él.
- Componente invisible: el usuario no ve qué parte de la métrica viene del fallback.
- Incoherente con el backtesting: el mejor candidato tendría una métrica distinta en
  `backtesting_forecaster_multiseries`, salvo que el backtesting también aplique el fallback.
- Premia la cobertura: parte del resultado es "qué candidato necesita menos fallback".

### c1. Rellenar con NaN también en el entrenamiento

Entrenar también con las filas en las que faltan lags porque son anteriores al **inicio real**
de la serie (nunca porque la ventana de entrenamiento las corta, que serían filas de cold
start falsas) y predecir con NaN. Con `L` lags, cada serie añade hasta `L - 1` filas
(`y = 2.º valor ← [1.er valor, NaN × (L - 1)]`, ...). **Descartada frente a c2**:
- Misma precisión que c2 con los cuatro estimadores (sección 6.2).
- Duplica las filas de entrenamiento en el experimento 4 (×2.07). En general se pasa de
  `n - L` a `n - 1` filas por serie: con datos diarios, 365 lags y 2 años, casi el doble, y
  las filas nuevas son casi todo NaN.
- Cambia el entrenamiento para todas las series y, con él, todos los resultados actuales.
- No mejora los puntos que hoy ya se predicen (diferencias de ±0.3 sin dirección clara).

Queda como alternativa si la sección 8.2 muestra que c2 falla en algún caso.

## 6. Propuesta: c2

### 6.1 Definición

- **Entrenamiento:** igual que hoy. Solo filas con todos los lags observados; una serie con
  `window_size` valores o menos no aporta filas.
- **Escalado (requisito, sección 6.4.1):** toda serie con algún valor observado ajusta su
  propio `transformer_series`, aunque no aporte filas. No se usa `_unknown_level` para ella.
  El encoding de la serie puede seguir siendo el de un nivel desconocido: medido, da igual
  (sección 6.4.1).
- **Predicción:** con un estimador que acepte NaN (`estimator_has_native_nan_support`), una
  serie con al menos un valor observado antes del test se predice aunque tenga menos de
  `window_size` valores. Los lags (y `window_features`) que faltan van a NaN.
- **Sin soporte de NaN:** se mantiene el descarte y se avisa pronto y una sola vez. En grid y
  random se puede calcular antes del bucle qué candidatos no cubrirán qué series; en
  bayesian, en el primer trial que descarte. El aviso dice qué series y qué hacer: usar un
  estimador que acepte NaN o quitarlas de `levels`.

Ejemplo con `lags=3` (`long`: 6 valores, `new`: 2 valores):

| | entrenamiento | predicción del siguiente valor de `new` |
|---|---|---|
| hoy | solo filas completas de `long` | no se predice (descartada) |
| c2 | igual que hoy (más el transformer propio de `new`) | `[12, 10, NaN]` |
| c1 | además `52 ← [50, NaN, NaN]`, `55 ← [52, 50, NaN]`, `12 ← [10, NaN, NaN]` | `[12, 10, NaN]` |

### 6.2 Resultados de la simulación (experimentos 3 y 4)

**Validación de la simulación** (script `_03_option_c_hgbr.py`): la implementación manual de
"hoy" coincide exactamente con skforecast con `transformer_series=StandardScaler()`
(semilla 42, `lags` 3/12/24: 15.23 / 12.10 / 11.50 en los dos), y con la misma cobertura
(100% / 90% / 64%). La simulación escala cada serie con sus propios valores, es decir, la
variante c2_own de la sección 6.4.

**Ventaja de `lags_24` sobre `lags_12`** (MAE de `lags_12` menos MAE de `lags_24`; HGBR;
c1 de este script rellena también desde el corte de la ventana):

| semilla | hoy | c1 | c2 | cobertura de `lags_24` hoy |
|---|---|---|---|---|
| 42 | +0.60 | +0.26 | +0.52 | 64% |
| 0 | +0.90 | +0.42 | +0.30 | 62% |
| 1 | **-0.61** | +0.19 | -0.05 | 60% |
| 2 | +1.77 | +0.96 | +0.91 | 62% |
| 3 | **-0.29** | +0.37 | +0.57 | 70% |
| 4 | +0.97 | 0.00 | +0.51 | 64% |

**Cuatro estimadores** (script `_04_c1_vs_c2.py`, c1 solo desde el inicio real, c2 con
encoding NaN para las series fuera del entrenamiento; media de 3 semillas, `lags=24`, MAE):

| estimador | cold start: naive | cold start: c1 | cold start: c2 | puntos que hoy ya se predicen: hoy | ídem: c1 | filas c1 / hoy |
|---|---|---|---|---|---|---|
| HGBR | 19.86 | 13.91 | 14.06 | 10.62 | 10.51 | ×2.07 |
| LightGBM | 19.86 | 13.92 | 13.80 | 10.64 | 10.44 | ×2.07 |
| XGBoost | 19.86 | 14.17 | 14.24 | 10.55 | 10.62 | ×2.07 |
| CatBoost | 19.86 | 13.59 | 13.39 | 10.11 | 10.14 | ×2.07 |

("Cold start" son los ~220 de 600 puntos que `lags_24` no puede predecir hoy.)

- c2 y c1 empatan: la diferencia en cold start es menor de 0.3 y cambia de signo. Las dos
  mejoran al naive en torno a un 30%.
- La regla con la que cada estimador trata un NaN que no vio al entrenar (HGBR: rama con
  más muestras; LightGBM: lo convierte en 0; XGBoost: rama por defecto; CatBoost: mínimo)
  no da predicciones malas con ninguno de los cuatro.

### 6.3 Por qué c2

- Cumple los criterios 1 a 4: solo predicciones del candidato, sin fallback, mismos puntos
  para todos y sin depender de conocer los candidatos.
- No añade filas de entrenamiento; solo un transformer más por serie corta. **Las
  predicciones de los puntos que hoy ya se predicen no cambian** (medido con el prototipo:
  diferencia máxima 0 con LightGBM y HGBR, `lags` 12 y 24).
- Encaja con lo que skforecast ya hace: admitir NaN en la última ventana con estimadores que
  los aceptan, y predecir niveles desconocidos.
- Lo implícito (cómo trata el estimador un NaN no visto) está medido y es documentable:
  "las series con menos historia que la ventana se predicen con los lags que faltan a NaN; el
  estimador los trata como cualquier valor ausente".

c2 no cambia las predicciones existentes, pero **sí cambia las métricas** del backtesting y
de las búsquedas con estimadores que aceptan NaN, porque entran puntos que hoy se descartan.
El ranking de búsquedas existentes puede cambiar. Necesita una línea en "Before upgrading".

### 6.4 Prototipo con skforecast real (experimento 5)

Script `BUG-search-short-series_05_c2_prototype.py`: `fit` y `predict` reales de
`ForecasterRecursiveMultiSeries`, fold a fold, sin cambiar la librería. `predict` ya acepta
una última ventana rellenada con NaN hasta `window_size` y niveles que no estuvieron en el
entrenamiento. Variantes:

- **drop**: la regla de hoy. **Validado: mismas predicciones que
  `backtesting_forecaster_multiseries`** (540 y 384 filas con `lags` 12 y 24, diferencia
  máxima 0).
- **c2_pooled**: c2 tal como lo haría skforecast hoy. La serie corta es un nivel desconocido
  y se escala con el transformer `_unknown_level`, ajustado sobre todas las series
  concatenadas en `_create_train_X_y`.
- **c2_own**: c2 con cada serie escalada con sus propios valores (fuera de skforecast, con
  `transformer_series=None`).

#### 6.4.1 Datos sintéticos (los de los experimentos 2 a 4, 3 semillas)

**Escalado** (`lags=24`, encoding `ordinal`, MAE en cold start; naive: 19.86):

| estimador | c2_pooled | c2_own | ventaja de `lags_24`: hoy | c2_pooled | c2_own |
|---|---|---|---|---|---|
| HGBR | 34.32 | 14.06 | +0.30 | -5.52 | +0.26 |
| LightGBM | 31.23 | 13.80 | +0.20 | -4.46 | +0.27 |
| XGBoost | 32.61 | 14.24 | +0.21 | -5.01 | +0.10 |
| CatBoost | 32.36 | 13.39 | +0.17 | -5.32 | +0.16 |

- **c2_pooled es inservible**: en cold start es mucho peor que el naive (31 a 34 frente a
  19.9) y castiga a la ventana larga (-4.5 a -5.5), es decir, sustituye un sesgo por otro.
  Las series tienen niveles de 50 a 150, y el escalado común las pone fuera de la escala
  con la que el modelo aprendió.
- **c2_own reproduce la simulación** (13.4 a 14.2, los mismos números de la sección 6.2).
  De ahí el requisito de la sección 6.1.

**Encodings** (MAE en cold start, `lags=24`):

| estimador | encoding | c2_pooled | c2_own |
|---|---|---|---|
| LightGBM | `ordinal` | 31.23 | 13.80 |
| LightGBM | `ordinal_category` | 30.73 | 13.71 |
| LightGBM | `onehot` | 31.34 | 13.71 |
| LightGBM | `None` | 19.56 | 13.72 |
| HGBR | `ordinal` | 34.32 | 14.06 |
| HGBR | `ordinal_category` | 34.28 | 14.09 |
| HGBR | `onehot` | 34.12 | 14.19 |
| HGBR | `None` | 23.40 | 14.14 |

Con escalado propio, el encoding da igual: el nivel puede seguir siendo desconocido para el
encoding. Ningún encoding da error al predecir un nivel desconocido.

**`window_features`.** Hallazgo previo a c2: `RollingFeatures.transform` (camino de
predicción) quita los NaN de la ventana e **ignora `min_periods`**
(`skforecast/preprocessing/_preprocessing.py`, `transform`), mientras que `transform_batch`
(entrenamiento) usa el rolling de pandas con `min_periods`, que por defecto es la ventana.
Con `[NaN × 3, 1, 2, 4]` y ventana 6, al predecir `roll_mean_6` = 2.33 y `roll_sum_6` = 7
(calculados con 3 valores), y al entrenar los dos serían NaN. La asimetría existe hoy, sin
c2, cuando la última ventana tiene NaN y el estimador los acepta; c2 la hace frecuente.
Medido con LightGBM, `lags=6` y `RollingFeatures(["mean", "std", "sum"], 24)`, MAE en cold
start (naive: 19.86):

| camino de predicción | c2_pooled | c2_own |
|---|---|---|
| ignora los NaN (hoy) | 36.81 | 14.93 |
| NaN si hay menos de `min_periods` valores (parche) | 36.10 | 14.60 |

Hacer coherente `transform` mejora poco (2%), pero es lo correcto y no depende de c2.

**Diferenciación** (LightGBM, `lags=12`, MAE en cold start):

| `differentiation` | hoy (puntos cubiertos) | c2_pooled | c2_own | naive |
|---|---|---|---|---|
| `None` | 12.17 | 30.28 | 13.31 | 18.52 |
| 1 | 14.43 | 18.40 | 16.31 | 18.46 |

Con relleno al principio de la ventana, la diferenciación funciona: no hay predicciones NaN
(el relleno no se propaga porque la inversa parte del último valor observado) y c2_own mejora
al naive un 12%. Hoy `differentiation` fuerza `dropna_last_window=True` por los NaN
**intermedios**; los NaN de relleno, al principio, no tienen ese problema.

**Intervalos: no concluyente.** Conformal al 80% con residuos in-sample y LightGBM: cobertura
del 4% al 5% en todos los puntos, también en los que hoy ya se predicen. Los residuos
in-sample de LightGBM sobreajustado son casi cero y los intervalos salen estrechísimos, así
que la medición no dice nada de c2. Lo único que se ve: el ancho en cold start es 1.65 veces
el de los puntos normales con c2_own y 2.6 veces con c2_pooled. Hay que repetirlo con
residuos out-of-sample (sección 8.2).

#### 6.4.2 Datos reales: M5

Tienda CA_1 de M5 (`fetch_dataset("m5")`): 40 productos a la venta desde 2011 y 60 lanzados
entre enero y noviembre de 2014 (lanzamiento = primer día con `sell_price`). Backtesting de
julio a diciembre de 2014, `steps=28`, `refit=True`, 7 folds, 17 224 puntos de test; con
`lags=56` hoy quedan fuera 1476 (8.6%). Ventas diarias intermitentes (media de ~1 unidad).
MAE medio por serie:

| estimador | variante | `lags=7` | `lags=28` | `lags=56` | cobertura con `lags=56` |
|---|---|---|---|---|---|
| LightGBM | hoy | 1.122 | 1.105 | **1.087** | 91.4% |
| LightGBM | c2_pooled | **1.130** | 1.156 | 1.211 | 100% |
| LightGBM | c2_own | 1.121 | 1.109 | **1.103** | 100% |
| HGBR | hoy | 1.173 | 1.107 | **1.101** | 91.4% |
| HGBR | c2_pooled | 1.180 | **1.138** | 1.184 | 100% |
| HGBR | c2_own | 1.172 | **1.110** | 1.113 | 100% |

MAE en los puntos de cold start (los que `lags=56` deja fuera hoy):

| | `lags=7` | `lags=28` | `lags=56` |
|---|---|---|---|
| LightGBM c2_pooled | 1.019 | 1.307 | 1.836 |
| LightGBM c2_own | 0.915 | 0.907 | 0.914 |
| HGBR c2_pooled | 1.003 | 1.149 | 1.573 |
| HGBR c2_own | 0.913 | 0.903 | 0.901 |
| naive (último valor) | 1.087 | | |
| media de los últimos 28 días | 0.876 | | |

- **Con HGBR el ranking cambia con datos reales**: hoy gana `lags=56`; con una comparación
  justa (c2_own) gana `lags=28`. Con LightGBM sigue ganando `lags=56`, pero su ventaja
  sobre `lags=28` baja de 0.018 a 0.006 (3 veces menos). Las diferencias son pequeñas
  porque en este panel solo el 8.6% de los puntos es cold start.
- **c2_pooled vuelve a fallar**: penaliza las ventanas largas y con LightGBM elige `lags=7`.
- **c2_own en cold start** mejora al naive del último valor un 17%, pero queda un 3% o 4%
  por detrás de la media de los últimos 28 días, un baseline fuerte para ventas
  intermitentes. El objetivo de c2 es una comparación justa, no ganar a los baselines, pero
  conviene saberlo.

## 7. Alcance de la implementación

| Pieza | Hoy | Con c2 |
|---|---|---|
| `_extract_data_folds_multiseries` | descarta la serie con última ventana corta o no entrenada | no descartarla si el estimador acepta NaN |
| `ForecasterRecursiveMultiSeries.fit` / `_create_train_X_y` | `ValueError` si una serie tiene `window_size` valores o menos; `_unknown_level` se ajusta sobre todas las series concatenadas | aceptar esas series para ajustar su `transformer_series` propio, sin aportar filas (requisito de la sección 6.1) |
| `ForecasterRecursiveMultiSeries.predict` y `check_predict_input` (`skforecast/utils/utils.py`, `len(last_window) < window_size` da `ValueError`) | exige `window_size` valores | admitir una `last_window` más corta, rellenada con NaN, con estimadores que aceptan NaN. Si no, el backtesting haría algo que `predict` no permite |
| `_train_test_split_one_step_ahead` | descarta igual | mismo cambio |
| `RollingFeatures.transform` | ignora NaN y `min_periods` | respetar `min_periods` como `transform_batch` (arreglo independiente de c2, sección 6.4.1) |
| Diferenciación | fuerza `dropna_last_window=True` | distinguir el relleno inicial (válido, medido) de los NaN intermedios (siguen descartándose) |
| Aviso de #1415 (`_warn_levels_not_predicted_multiseries`) | causas `train` y `last_window` | con c2 casi no salta con estimadores con NaN; revisar sus textos |
| `MissingValuesWarning` de `check_predict_input` por NaN en la última ventana | salta | no avisar por los NaN de relleno, o cada backtesting avisará |
| Búsquedas | silencian tras el primer candidato | aviso temprano y único para estimadores sin NaN |
| Mensajes de error que describen la regla de predicción: `ValueError` de #1427 cuando ningún fold tiene niveles que predecir, y los añadidos el 2026-10-21 (backtesting sin nada evaluable, en #1427; one-step-ahead sin series que entrenar o evaluar, en #1430) | citan la regla actual (más de `window_size` valores en el entrenamiento, última ventana completa) | revisar los textos: con estimadores que aceptan NaN la regla cambia |
| Búsqueda con `OneStepAheadFold` (`_evaluate_grid_hyperparameters_multiseries` y `_objective` de bayesian, `_search.py`) | `_train_test_split_one_step_ahead` se llama por candidato de `lags` fuera del `try`: un candidato sin nada que evaluar aborta toda la búsqueda | tratarlo por candidato ("Parameters skipped"), como con `TimeSeriesFold`; hace falta para los estimadores sin NaN, que mantienen el descarte |
| Camino rápido de los modelos lineales en `_build_predict_function` (#1432) | `predict` lanza `ValueError` (`Input X contains NaN in the predictors [...] at step ... (fecha)`) si un predictor es NaN. `_backtesting_forecaster_multiseries` y `_backtesting_forecaster` activan `_allow_nan_predictions` en su copia del forecaster, que `_build_predict_function` lee del forecaster, y entonces obtienen predicciones NaN en lugar del error | sin cambios, siempre que el relleno con NaN de la última ventana se limite a estimadores con `estimator_has_native_nan_support`. Si una serie rellenada llegara a un modelo lineal, `predict` daría ese error y el backtesting multiserie predicciones NaN con aviso, en lugar del descarte con aviso temprano que pide c2. Añadir un test: modelo lineal con una serie corta, que se sigue descartando |
| Aviso de predicciones NaN en las métricas multiserie (#1432, `_calculate_metrics_backtesting_multiseries`) | avisa del número de predicciones NaN por serie en fechas con valor real. Con estimadores sin NaN, las series cortas se descartan antes de predecir, así que no lo disparan | con c2 las series cortas se predicen con valores finitos y no deben dispararlo (comprobarlo en los tests). Su texto ("missing values in the last window or in `exog`") sigue valiendo |
| Silenciado de avisos en grid y random tras el primer candidato (sección 2.2) | oculta también el aviso de #1432 | el aviso temprano y único de c2 debería cubrir también los candidatos con predicciones NaN, o agregarse por candidato, para que el usuario sepa en cuáles faltan puntos (sección 7.1). Señal robusta, que no depende de avisos: QW-12 del roadmap, el número de predicciones evaluadas por serie en el backtesting; va antes y bloquea este bug, que lo lleva a las búsquedas por candidato (decisión del 2026-10-29) |
| Docs, `llms-base.txt`, skills (`forecasting-multiple-series`, `hyperparameter-optimization`, `backtesting-configuration`), release note con "Before upgrading" | | cambio de comportamiento público: `ai-context-sync` |

Fuera de alcance: `ForecasterDirectMultiVariate` y `ForecasterRnn` (usan todas las series como
predictores y no descartan series de la última ventana), `ForecasterFoundation` (el contexto
no depende del candidato).

Esfuerzo estimado: M a L.

### 7.1 Caso relacionado: `ForecasterDirectMultiVariate` con huecos en las series

Fuera del alcance de c2, pero es el mismo sesgo y hay que tenerlo en cuenta al arreglar el
aviso de las búsquedas.

- **Mecanismo.** ForecasterDirectMultiVariate usa todas las series como predictores, así que
  no descarta series de la última ventana: un fold cuya ventana contiene un hueco se predice
  igualmente. Que el hueco caiga dentro de la ventana depende del `window_size` del candidato.
  Medido (2026-10-22): dos series de 120 días, un NaN en `b` 5 valores antes del final del
  primer entrenamiento, `steps=10`, `refit=False`. Con `Ridge`, `lags=3` se evalúa sobre 60
  puntos y `lags=12` sobre 50 (el fold afectado da 10 predicciones NaN). Con
  `HistGradientBoostingRegressor`, los dos predicen los 60: no hay sesgo con estimadores que
  aceptan NaN.
- **Antes de #1432.** Con un modelo lineal (camino rápido `np.dot`), ese fold daba
  predicciones NaN que las métricas descartaban en silencio. Con estimadores que rechazan
  NaN por su cuenta (SVR, KNN), el backtesting fallaba entero.
- **Desde #1432.** Con un modelo lineal, el backtesting multiserie sigue dando NaN en ese
  fold (`_allow_nan_predictions`) y avisa del número de predicciones descartadas por serie.
  En grid y random el aviso solo sale para el primer candidato (sección 2.2), así que la
  búsqueda puede seguir comparando candidatos sobre puntos distintos sin que el usuario lo
  vea. Con huecos en `exog` no hay sesgo: la fecha afectada es la misma para cualquier
  `lags`, y todos los candidatos pierden los mismos puntos.
- **Al arreglar este bug.** Como mínimo, que el aviso por candidato de las búsquedas
  (sección 7) incluya este caso. Una comparación justa con estimadores sin NaN necesitaría
  el conjunto común de la opción B, descartada en general (sección 5); decidir si merece la
  pena para este caso o basta con avisar.

## 8. Mediciones

### 8.1 Hechas (2026-10-19)

| Punto | Resultado | Dónde |
|---|---|---|
| Escalado de las series que no entran en el entrenamiento | `_unknown_level` hace c2 inservible; el escalado propio lo arregla. Pasa a ser un requisito | 6.4.1, 6.4.2 |
| `window_features` | asimetría entre entrenamiento y predicción, previa a c2; arreglarla mejora un 2% | 6.4.1 |
| Encodings | con escalado propio dan igual; ninguno falla con un nivel desconocido | 6.4.1 |
| Datos reales | M5: el ranking cambia con HGBR; c2_own mejora al naive del último valor pero no a la media de 28 días | 6.4.2 |
| Diferenciación | el relleno inicial funciona; solo los NaN intermedios necesitan el descarte | 6.4.1 |
| Prototipo en skforecast | validado contra `backtesting_forecaster_multiseries`; las predicciones de hoy no cambian | 6.4 |

### 8.2 Pendientes

Por orden de riesgo:

1. **Intervalos de predicción** (bootstrapping, conformal, binned) con **residuos
   out-of-sample**: los in-sample de los árboles sobreajustados no permiten evaluar la
   cobertura (sección 6.4.1). Medir la cobertura en cold start frente al resto, y decidir qué
   residuos usa una serie corta (los suyos, pocos o ninguno, o los de `_unknown_level`).
2. **Más datos reales**: M5 tiene solo un 8.6% de puntos de cold start en el periodo
   elegido. Probar un periodo o un panel con más altas recientes, y otra frecuencia (por
   ejemplo semanal).
3. **Exógenas** en series cortas (`exog` por serie que empieza más tarde).
4. **Otras configuraciones de backtesting:** `refit=False` (la serie corta no entra nunca en
   el entrenamiento, solo se predice), `refit` entero, `fixed_train_size=False`, `gap > 0`,
   `fold_stride < steps`, `initial_train_size=None`.
5. **`OneStepAheadFold`**: repetir el experimento 5 por el camino de
   `_train_test_split_one_step_ahead`.
6. **Bayesian con `lags` en `search_space`:** cuántas veces sale el aviso temprano con
   estimadores sin NaN, y que salga solo una vez.
7. **Tiempos:** c2 no añade filas de entrenamiento, pero predice más series; medirlo con
   muchas series cortas.
8. **Series muy cortas:** con 1 o 2 valores, el transformer propio se ajusta con casi nada
   (desviación 0 o inestable). Decidir un mínimo o qué hacer cuando la desviación es 0.
9. **Avisos de #1432 en las búsquedas:** con c2 y el aviso temprano implementados, comprobar
   que el aviso de predicciones NaN sale una vez por candidato afectado en grid y random, que
   no se repite sin control en bayesian, y que no salta por las series cortas que c2 predice
   (secciones 7 y 7.1).

## 9. Scripts

En `dev/roadmap_0.27.0/repro/`:

| Script | Qué muestra | Tiempo aprox. |
|---|---|---|
| `BUG-search-short-series_01_ranking_flip.py` | experimento 1: inversión del ranking y 0 avisos (`TimeSeriesFold` y `OneStepAheadFold`) | segundos |
| `BUG-search-short-series_02_options.py` | experimento 2: hoy frente a a, B y d con 20 series | segundos |
| `BUG-search-short-series_03_option_c_hgbr.py [semilla]` | experimento 3: validación de la simulación contra skforecast, c1 y c2 con HGBR | ~2 min por semilla |
| `BUG-search-short-series_04_c1_vs_c2.py [semillas...]` | experimento 4: c1 (inicio real) y c2 con HGBR, LightGBM, XGBoost y CatBoost | ~2 min con 3 semillas |
| `BUG-search-short-series_05_c2_prototype.py synthetic` | experimento 5: prototipo con skforecast real; validación, escalado, encodings, `window_features`, intervalos y diferenciación | ~10 min |
| `BUG-search-short-series_05_c2_prototype.py m5 <m5.parquet>` | experimento 5 con M5 (CA_1). El parquet sale de `fetch_dataset("m5").reset_index().to_parquet(...)`, ~116 MB | ~10 min |

CatBoost escribe `catboost_info/` en el directorio de trabajo con los scripts 3 y 4: borrarlo
después (el script 5 lo evita con `allow_writing_files=False`).

## 10. Relación con el roadmap

- **ISSUE-1406 (#1415):** el aviso del backtesting y el helper por pares (serie, fold) son la
  base para detectar qué queda fuera. Con c2 hay que revisar sus textos.
- **ISSUE-1410 (#1418):** fija la frontera `window_size` del entrenamiento que usa c2.
- **QW-09 (`coverage`, `interval_width`):** las métricas de intervalo siguen la misma regla de
  qué puntos entran; con c2 entran más. El punto 1 de la sección 8.2 le afecta directamente.
- **FE-04 (baseline multiserie con los mismos folds):** mismo principio (comparar sobre los
  mismos puntos). El baseline tiene su propia frontera de series cortas.
- **#1427 y #1430 (fusionados el 2026-10-21):** el backtesting multiserie salta los folds sin
  series que entrenar o predecir, y `weighted_average` deja fuera los niveles predichos sin
  valores observados; `initial_train_size` se valida contra el span de las series, y el test
  del one-step-ahead (`_train_test_split_one_step_ahead`, que c2 también modifica) solo incluye
  las series con valores observados tras el entrenamiento. Con c2 se saltarán menos folds con
  estimadores que aceptan NaN; los textos de sus errores están en la sección 7.
- **#1432 (en revisión):** con un modelo lineal de scikit-learn, un predictor NaN hace que
  `predict` lance un `ValueError` con el predictor, el paso, la fecha y la serie, en lugar de
  devolver NaN. El backtesting no falla al predecir: su copia del forecaster tiene
  `_allow_nan_predictions=True` y devuelve NaN en esas fechas (así, un NaN que solo afecta a
  los pasos descartados del `gap` no importa). Las métricas multiserie avisan del número de
  predicciones NaN por serie en fechas con valor real; las de una serie fallan con un mensaje
  claro (causa probable y remedio). Tres puntos de contacto con este bug: el silenciado de las búsquedas
  oculta ese aviso (sección 2.2), el relleno de c2 debe limitarse a estimadores que aceptan
  NaN para no chocar con el error de `predict` (sección 7), y ForecasterDirectMultiVariate
  con huecos tiene el mismo sesgo entre candidatos (sección 7.1).
