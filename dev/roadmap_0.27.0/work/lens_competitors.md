# Lente: huecos frente a otras librerías (solo ML y foundation)

Fecha: 2026-10-18. Rama analizada: 0.26.x (`__version__ = "0.26.0"`).
Librerías de referencia: darts, sktime, mlforecast, autogluon-timeseries, statsforecast, neuralforecast, TimeGPT.
Entorno de los experimentos: Python 3.12, lightgbm 4.7.0, scikit-learn 1.9.1, skforecast 0.26.0 (rama local).
Scripts: `/tmp/claude-0/-home-user-skforecast/8505cf39-716d-5c4c-adf6-669c7ec65b16/scratchpad/competitors/exp_seasonal.py` y `exp_decay.py`.

## Resumen

La mayoría de los "huecos clásicos" frente a la competencia **ya están cubiertos** en skforecast (ver tabla de descartes). Quedan cinco huecos reales que encajan con la filosofía (estimadores scikit-learn, API simple):

| # | Propuesta | Tipo | Esfuerzo | Evidencia clave |
|---|-----------|------|----------|-----------------|
| 1 | Ventanas estacionales en `RollingFeatures` (media de la misma hora de los últimos 7 días / mismo día de las últimas 4 semanas) | feature | M | MAE 49,75 a 47,69 (-4,1 %) en bike_sharing horario |
| 2 | Baseline `ForecasterEquivalentDate` en `backtesting_forecaster_multiseries` | quick win | S/M | Hoy lanza `TypeError`; la guía oficial de comparación hace un bucle a mano |
| 3 | Espacios de búsqueda por defecto (`search_space="auto"`) | quick win | S/M | 16 `def search_space` en 8 notebooks; mlforecast trae `AutoLightGBM`, `AutoXGBoost`... |
| 4 | Variables estáticas por serie (`static_features`) en modelos globales | feature | M | darts, mlforecast, autogluon y neuralforecast lo tienen; skforecast no (grep vacío) |
| 5 | Pesos de decaimiento temporal integrados (`weight_func` exponencial) | quick win | S | MAE 49,75 a 48,33 (-2,8 %) con semivida 365 días; darts trae `sample_weight="exponential"` |

## 1. Ventanas estacionales en `RollingFeatures`

**Qué falta.** `RollingFeatures` (`skforecast/preprocessing/_preprocessing.py:1071`, `__init__` en :1133) solo calcula estadísticos sobre ventanas contiguas `[t-w, t-1]`. No hay forma de calcular "media de la misma hora en los últimos 7 días" o "media del mismo día de la semana en las últimas 4 semanas". mlforecast lo trae (`SeasonalRollingMean/Std/Min/Max/Quantile`, y `lag_transforms={7: [RollingMean(4)]}` para aplicar ventanas sobre un retardo); darts y sktime (`WindowSummarizer` con `lag_feature`) permiten ventanas desplazadas. grep de `season`, `offset` o `shift` en la clase: sin resultados.

**Por qué encaja.** Es un predictor estándar en datos horarios y diarios con estacionalidad múltiple (energía, movilidad, ventas). Hoy exige una clase propia, que además **no se guarda** con `save_forecaster` (aviso en `skforecast/utils/utils.py:3599-3611`) ni es compatible con skops: fricción directa en producción.

**Medido.** bike_sharing (`users`, horario), entrenamiento hasta 2012-08-31, backtesting de 4 meses con `steps=24`, LightGBM 500 árboles, lags 1..24 y 168, exógenas de calendario y meteo:

| Configuración | MAE |
|---|---|
| A: lags + `RollingFeatures(mean 24, mean 168)` | 49,750 |
| B: A + media estacional 24x7 + media estacional 168x4 (clase propia) | **47,691 (-4,1 %)** |

**Antes** (lo que hoy escribe el usuario, ~20 líneas):

```python
class SeasonalRolling:
    def __init__(self, season_length, n_seasons):
        self.s, self.k = season_length, n_seasons
        self.window_sizes = season_length * n_seasons
        self.features_names = [f"seas_mean_{season_length}x{n_seasons}"]
    def transform_batch(self, X):
        feat = sum(X.shift(j * self.s) for j in range(1, self.k + 1)) / self.k
        return feat.to_frame(self.features_names[0]).iloc[self.window_sizes:]
    def transform(self, X):
        idx = np.arange(X.shape[0] - self.s, -1, -self.s)
        return np.nanmean(X[idx], axis=0).reshape(-1, 1) if X.ndim == 2 else np.array([np.nanmean(X[idx])])

window_features = [RollingFeatures(["mean"], [24]), SeasonalRolling(24, 7), SeasonalRolling(168, 4)]
```

**Después:**

```python
window_features = RollingFeatures(
    stats        = ["mean", "mean", "mean"],
    window_sizes = [24, 7, 4],
    season_length= [None, 24, 168],   # None = ventana contigua (comportamiento actual)
)
```

**Cambio.** Nuevo parámetro `season_length` (int o lista) en `RollingFeatures`: si se indica, la ventana toma `X[-s], X[-2s], ..., X[-w*s]` y `max_window_size = w*s`. Afecta a `transform_batch` (pandas: `rolling` sobre la serie remuestreada por fase, o suma de `shift`) y a `transform` (indexado con paso `-s`, vectorizable). Nombres `roll_mean_7_s24`. Opcional en la misma PR: estadístico `quantile`. Extiende parcialmente DISC-1153 (window features personalizadas), pero es otra necesidad: no es un estadístico nuevo, es otra geometría de ventana.

## 2. Baseline `ForecasterEquivalentDate` para varias series

**Qué falta.** `ForecasterEquivalentDate` es solo de serie única (`"allowed_input_types_series": ["pandas.Series"]`, `skforecast/recursive/_forecaster_equivalent_date.py:246`). `backtesting_forecaster_multiseries` lo rechaza (`skforecast/model_selection/_validation.py:1698`). Reproducido:

```
TypeError `forecaster` must be of type ['ForecasterRecursiveMultiSeries', 'ForecasterDirectMultiVariate', 'ForecasterRnn'], for all other types of forecasters use the functions available in the `model_selection` module. Got ForecasterEquivalentDate
```

darts (`NaiveSeasonal` sobre listas de series), statsforecast (`SeasonalNaive`) y autogluon (`SeasonalNaive`) evalúan el baseline en paneles de miles de series con una llamada.

**Por qué importa.** Comparar el modelo global con un naive estacional es la primera buena práctica (skill `baseline-forecasting`, que no cubre multiserie). La propia guía `docs/user_guides/comparing-forecasting-models.ipynb` (líneas 1752 y 1775 del JSON) hace un bucle por serie y concatena a mano, y pierde las métricas agregadas (`average`, `weighted_average`, `pooling`) que sí obtiene el modelo global.

**Antes:**

```python
preds = []
for level in series.columns:
    _, p = backtesting_forecaster(
        forecaster=ForecasterEquivalentDate(offset=7), y=series[level], cv=cv,
        metric="mean_absolute_error", show_progress=False,
    )
    preds.append(p.assign(level=level))
preds = pd.concat(preds)
# métricas por serie y agregadas: a mano
```

**Después:**

```python
metrics, preds = backtesting_forecaster_multiseries(
    forecaster=ForecasterEquivalentDate(offset=7), series=series_dict, cv=cv,
    metric=["mean_absolute_error", "mean_absolute_scaled_error"],
)
```

**Cambio.** Aceptar `ForecasterEquivalentDate` en `backtesting_forecaster_multiseries` iterando por nivel (es trivial de ajustar, no hay modelo) y reutilizando el cálculo de métricas multiserie existente; admitir `series` en DataFrame ancho, largo o dict y series de distinta longitud. Usuarios: flujo habitual en multiserie (todo el que compara con baseline).

## 3. Espacios de búsqueda por defecto

**Qué falta.** `bayesian_search_forecaster` exige `search_space: Callable` (`skforecast/model_selection/_search.py:561`). No hay espacios predefinidos (grep de `default_search_space` vacío). El usuario copia el ejemplo de la skill (`skills/hyperparameter-optimization/SKILL.md:62-69`) o de alguno de los 16 `def search_space` de 8 notebooks. mlforecast ofrece `AutoLightGBM`, `AutoXGBoost`, `AutoCatboost`, `AutoRandomForest`, `AutoRidge`, `AutoLasso`... con espacios `lightgbm_space(trial)` etc., y además un espacio de lags derivado de `season_length`; autogluon tiene presets.

**Antes:**

```python
def search_space(trial):
    return {
        "lags": trial.suggest_categorical("lags", [24, 48, [1, 2, 3, 23, 24, 168]]),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
    }
results, _ = bayesian_search_forecaster(forecaster, y=y, cv=cv, search_space=search_space, metric="mean_absolute_error")
```

**Después:**

```python
results, _ = bayesian_search_forecaster(forecaster, y=y, cv=cv, search_space="auto", metric="mean_absolute_error")
# o bien, para ampliar:
from skforecast.model_selection import get_default_search_space
space = get_default_search_space(forecaster, lags=[24, 48, 168])
```

**Cambio.** Diccionario de espacios por clase de estimador (LightGBM, XGBoost, CatBoost, HistGradientBoosting, RandomForest/ExtraTrees, Ridge/Lasso/ElasticNet) que devuelve una función compatible con optuna; resolverlo cuando `search_space == "auto"` en las tres funciones bayesianas (`_search.py:732,780,1807,1869`). Para Pipelines, prefijo automático con el nombre del último paso. Error claro si el estimador no tiene espacio. Sin lags por defecto salvo que se pasen (decidir con cuidado: es opinable). Usuarios: flujo habitual (la búsqueda bayesiana aparece en casi todas las guías).

## 4. Variables estáticas por serie en modelos globales

**Qué falta.** No existe soporte ni documentación de variables estáticas (grep de `static` en `skforecast/recursive/_forecaster_recursive_multiseries.py`, `docs/user_guides`, `skills` y `tools/ai/llms-base.txt`: vacío). mlforecast (`static_features`), darts (`static_covariates`, `use_static_covariates=True` por defecto), autogluon (`static_features`) y neuralforecast (`stat_exog_list`) lo traen.

**Por qué importa.** En paneles retail o de sensores (tipo de tienda, región, categoría de producto, potencia instalada) estas variables permiten generalizar entre series y, sobre todo, predecir series nuevas (cold start, que skforecast ya soporta con `_unknown_level`, `_forecaster_recursive_multiseries.py:971`) con algo más que el nivel desconocido del encoding. Hoy hay que repetir cada atributo en todas las filas de la exógena, también en las del horizonte de predicción.

**Antes:**

```python
static = pd.DataFrame({"store_type": [...], "size": [...]}, index=series_names)
exog_dict = {
    sid: pd.DataFrame(np.repeat(static.loc[[sid]].to_numpy(), len(idx_full), axis=0),
                      index=idx_full, columns=static.columns).join(exog_dyn[sid])
    for sid in series_names
}
# y repetir lo mismo para el índice futuro antes de cada predict / backtesting
```

**Después:**

```python
forecaster.fit(series=series_dict, exog=exog_dyn, static_features=static)
forecaster.predict(steps=14, exog=exog_dyn_future)                     # reutiliza static
forecaster.predict(steps=14, levels=["new"], last_window=lw, static_features=static_new)
```

**Cambio.** Guardar `static_features_` (DataFrame indexado por serie) en `fit` de `ForecasterRecursiveMultiSeries` (y `bayesian/grid_search`, `backtesting_forecaster_multiseries`, `select_features_multiseries` lo propagan). Se añaden como columnas constantes en `_create_train_X_y` y en la matriz de predicción; columnas categóricas con el mismo tratamiento que la exógena categórica. Esfuerzo M por la propagación a backtesting y búsquedas. Usuarios: flujo habitual en multiserie con metadatos; nicho en serie única (no aplica).

## 5. Pesos de decaimiento temporal integrados

**Qué falta.** `weight_func` acepta solo funciones propias (`skforecast/utils/utils.py:297-320`); el único ejemplo de la guía es excluir un periodo (`docs/user_guides/weighted-time-series-forecasting.ipynb`, `def custom_weights`). darts ofrece `fit(sample_weight="linear" | "exponential")`. Además, una función del usuario definida en `__main__` exige `save_custom_functions` y avisa al guardar (`utils.py:3590-3596`).

**Medido** (misma configuración A del punto 1):

| Semivida | MAE |
|---|---|
| sin pesos | 49,750 |
| 365 días | **48,332 (-2,8 %)** |
| 180 días | 49,388 (-0,7 %) |
| 90 días | 51,334 (+3,2 %) |

La ganancia depende mucho de la semivida: por eso conviene un helper con parámetro explícito y documentado, no un valor por defecto.

**Antes:**

```python
def decay(index):
    age = (index.max() - index) / pd.Timedelta(days=1)
    return 0.5 ** (age / 365)
forecaster = ForecasterRecursive(estimator=LGBMRegressor(), lags=24, weight_func=decay)
```

**Después:**

```python
from skforecast.utils import TimeDecayWeights
forecaster = ForecasterRecursive(estimator=LGBMRegressor(), lags=24,
                                 weight_func=TimeDecayWeights(half_life="365D"))
```

**Cambio.** Clase invocable importable (serializable sin código fuente) con `half_life` (entero en pasos o cadena de pandas) y `kind` (`"exponential"`, `"linear"`), relativa al último instante del entrenamiento (en backtesting con `refit` se recalcula en cada fold). Documentar en la guía de pesos. Usuarios: nicho medio (series con cambios de régimen o deriva lenta).

## Descartes (ya existe o no compensa)

| Idea (competidor) | Estado en skforecast | Evidencia |
|---|---|---|
| Calendario y festivos (darts `add_holidays`, mlforecast `date_features`) | Existe | `CalendarFeatures`, `create_calendar_features`, `calculate_distance_from_holiday` (`skforecast/preprocessing/_calendar.py`, exportados en `preprocessing/__init__.py`) |
| EWM como window feature (mlforecast `ExponentiallyWeightedMean`) | Existe | `RollingFeatures` stat `'ewm'` (`_preprocessing.py:1078`) |
| Expanding (mlforecast `ExpandingMean`) | No existe; incompatible con la ventana fija de la predicción recursiva y de poco valor | |
| Box-Cox / escaladores locales por serie (mlforecast `LocalBoxCox`, `LocalStandardScaler`) | Existe | `transformer_series` admite dict por serie (`_forecaster_recursive_multiseries.py:414`), y `PowerTransformer` sirve para Box-Cox |
| Diferenciación por serie | Existe | `differentiation: int | dict` (`_forecaster_recursive_multiseries.py:419`) |
| Diferenciación estacional (mlforecast `Differences([24])`, sktime `Differencer(lags)`) | No existe (`TimeSeriesDifferentiator(order, window_size)`, `_preprocessing.py:112-115`) | Sin demanda en issues (solo #854, fraccional); los lags estacionales ya capturan la estacionalidad en ML. Baja prioridad |
| Estrategias de refit en CV (darts `retrain`) | Existe | `TimeSeriesFold(refit=bool|int, fixed_train_size, fold_stride, skip_folds, gap)` (`_split.py:103-116`) |
| Importancia por horizonte | Existe | `ForecasterDirect.get_feature_importances(step)` (`skforecast/direct/_forecaster_direct.py:3567`) |
| Lags y ventanas de exógenas (darts `lags_future_covariates`) | Documentado con feature-engine | `docs/user_guides/exogenous-variables.ipynb` (LagFeatures + WindowFeatures) |
| Predicción de series nuevas con modelo global (mlforecast `new_df`) | Existe | `_unknown_level` (`_forecaster_recursive_multiseries.py:971`) |
| Foundation zero-shot con covariables pasadas y futuras | Existe | `supports_past_only_covariates` (`skforecast/foundation/_adapters.py:170`) |
| Ensembles de forecasters (darts `NaiveEnsembleModel`, sktime `EnsembleForecaster`, autogluon `WeightedEnsemble`) | No existe a nivel forecaster (sí `StackingRegressor` a nivel estimador, guía `stacking-ensemble-models-forecasting.ipynb`) | Medido: media de dos LightGBM distintos 49,41 frente a 49,75 del mejor (-0,7 %); con naive estacional 50,57 (peor). Ganancia pequeña; no justifica API nueva ahora. Podría reevaluarse con ML + foundation (no medido: sin torch en el entorno) |
| Reconciliación jerárquica (hierarchicalforecast, darts, sktime) | No existe | Sin demanda en issues (búsqueda semántica "hierarchical reconciliation": ningún resultado relevante); esfuerzo L; hierarchicalforecast puede consumir las predicciones de skforecast. Como mucho, una guía |
| Fine-tuning de foundation (TimeGPT `finetune_steps`, autogluon) | Ya planificado | PLAN-CKPT y PLAN-TTM |

## Fricciones observadas

- La guía oficial `comparing-forecasting-models.ipynb` es un buen termómetro de fricción: escribe una función `summarize` propia (cobertura, MASE: ya identificado en GAP-backtesting-interval-metrics), renombra `q_0.1/q_0.9` de foundation a `lower_bound/upper_bound` (ya identificado en GAP-foundation-backtesting-interval) y hace bucles por serie para los baselines (propuesta 2).
- Las window features propias son el mecanismo de extensión, pero no se guardan con el forecaster (`utils.py:3599-3611`); cada estadístico que se integra en `RollingFeatures` elimina esa fricción de producción.
- Las window features propias en Python puro ralentizan el backtesting: la configuración B tardó 79,0 s frente a 51,9 s de A (aunque B también tiene dos variables más).
