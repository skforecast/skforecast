# Lente: usuario de modelos foundation (skforecast 0.26.0 -> 0.27.0)

Fecha: 2026-10-17. Rama analizada: `0.26.x`.

## Cómo se ha evaluado

- Lectura de `skforecast/foundation/` (`_forecaster_foundation.py`, `_foundation_model.py`, `_adapters.py`, `_utils.py`, `_model_info.py`), de `backtesting_foundation` (`skforecast/model_selection/_validation.py:2357-2903`), de `bayesian_search_foundation` (`skforecast/model_selection/_search.py:2091`), de `deepcopy_forecaster` (`skforecast/utils/utils.py:4941`), de la skill `skills/foundation-forecasting/` y de los tests.
- Ejecución real en CPU con `autogluon/chronos-2-small` (`chronos-forecasting` instalado con `uv pip install`, torch 2.14.1+cpu, transformers 5.20.1, Python 3.12). La descarga desde Hugging Face funcionó. No se probaron otros backends (TimesFM, Moirai, TabICL, TabPFN, T0, Nori, TS-ICL).
- Scripts en `/tmp/claude-0/-home-user-skforecast/8505cf39-716d-5c4c-adf6-669c7ec65b16/scratchpad/foundation/` (`s1_basic.py` a `s11_q.py`).

El flujo básico funciona bien de punta a punta: `fit`, `predict`, `predict_interval`, `predict_quantiles` y `backtesting_foundation` con `quantiles` (10 folds mensuales en 0,2 s una vez cargado el modelo; la primera carga tarda unos 13 s incluida la descarga).

## Candidatos (ordenados por valor)

### 1. BUG: un pipeline o modelo precargado (p. ej. Chronos-2 ajustado con fine-tuning) se descarta en silencio al crear `ForecasterFoundation`

- Todos los adaptadores aceptan un objeto precargado (`pipeline=` en Chronos, `model=` o `module=` en el resto: `_adapters.py:185,703,1175,1835,2267,2908,3532,4043,4542`), pero `get_params` no lo incluye (`_foundation_model.py:1295-1321`, decisión documentada para no copiar pesos). `ForecasterFoundation.__init__` hace `clone(estimator)` (`_forecaster_foundation.py:160`) y `deepcopy_forecaster` también clona (`utils.py:4999`), así que el objeto se pierde y se recarga el checkpoint base desde `model_id`.
- Reproducido (`s4_pipeline.py`): fine-tuning de 30 pasos con `Chronos2Pipeline.fit`, después `FoundationModel(..., pipeline=ft)`.
  - `FoundationModel` directo (ajustado): `[10.231 9.79 9.795 9.487 9.152 10.464 10.704]`
  - `ForecasterFoundation`: `[12.083 12.797 11.061 8.647 6.958 7.668 9.977]`
  - Pipeline base: `[12.083 12.797 11.061 ...]`, idéntico al forecaster. `f.estimator.adapter._pipeline` vale `None` tras el `__init__`.
- El propio fixture de tests lo sabe (`tests_forecaster_foundation/fixtures_forecaster_foundation.py:200-203`: "a pipeline passed to the constructor would not survive that clone"), pero el usuario no recibe ningún aviso. Además, un checkpoint local no se puede usar porque `_resolve_adapter` exige un prefijo de Hugging Face (`_adapters.py:5132`). Hoy no hay forma pública de usar un modelo ajustado con `ForecasterFoundation`, `backtesting_foundation` ni `bayesian_search_foundation`.
- Arreglo propuesto (S): en `ForecasterFoundation.__init__` (y en `deepcopy_forecaster` para foundation) pasar la referencia del objeto precargado al adaptador clonado (sin copiar pesos), o como mínimo lanzar un aviso o un error. Completa PLAN-CKPT (`finetuned_checkpoint`).

### 2. QUICK WIN: aviso cuando se ajusta con exog y se predice sin ella (Chronos-2, TS-ICL y TimesFM 3.0)

- Los adaptadores con `supports_past_only_covariates=True` usan entonces la exog solo como covariable pasada, sin aviso (`_foundation_model.py:950-993`, solo avisa para los otros adaptadores). Los forecasters ML lanzan `ValueError`.
- Reproducido (`s9_exog_forget.py`), exog `promo` con efecto +5: sin `exog` en `predict` no se emite ningún aviso y en los días con promo la predicción es 10,08 / 10,03 en lugar de 15,08 / 15,27. `ForecasterRecursive` en el mismo caso: `ValueError: Forecaster trained with exogenous variable/s...`.
- Antes: `forecaster.fit(series=y, exog=exog_train); forecaster.predict(steps=7)` # sin aviso, la promo se ignora.
- Después: el mismo código emite `MissingExogWarning` ("`promo` se usa solo como covariable pasada; pasa `exog` con valores futuros si se conocen"). `MissingExogWarning` ya existe en `skforecast/exceptions`.
- Afecta al flujo habitual con exógenas (calendario, promociones, festivos). Esfuerzo S.

### 3. HUECO: covariables solo pasadas en `backtesting_foundation` (fuga de información)

- `predict` permite covariables solo pasadas (columna en `context_exog` sin valores futuros), y la skill lo anuncia, pero `backtesting_foundation` pasa siempre todas las columnas de `exog` del tramo de test como covariables futuras (`_validation.py:2556-2575`). Con una variable observada a posteriori (temperatura real, ventas de otra serie) el backtest usa el futuro.
- Medido (`s5_pastonly.py`, Chronos-2 small, 20 folds de 7 días): MAE con la columna como futura 0,484; bucle honesto con la columna solo pasada 2,992; sin exog 2,99. El backtest sobrestima la precisión 6 veces.
- Antes: escribir un bucle a mano con `predict(context=..., context_exog=...)` por fold.
- Después: `backtesting_foundation(..., exog=exog, past_only_exog=['x_observed'])` (y el mismo parámetro en `bayesian_search_foundation`), que elimina esas columnas de `exog_test`.
- Esfuerzo S. Completa en parte ISSUE-1131 (exógenas as-of): resuelve el caso de variables que no se conocen en absoluto en el futuro.

### 4. RENDIMIENTO: agrupar los folds en una sola llamada al modelo y reducir la sobrecarga pandas por serie en `backtesting_foundation`

- `backtesting_foundation` llama al modelo una vez por fold (`_validation.py:2577-2594`). Los modelos foundation son zero-shot y los contextos de todos los folds se pueden enviar como un único lote.
- Medido en CPU (`s3_perf1.py`), 1 serie horaria, 200 folds de 24 pasos: backtesting 5,92 s (30 ms por fold); mismas predicciones en una sola llamada agrupada: 1,64 s (3,6 veces más rápido, diferencia máxima 3,8e-06). Con 50 series y 30 folds (`s2_perf.py`): 11,33 s frente a 9,52 s (ya van agrupadas por fold). En GPU la ganancia debería ser mayor (no medido).
- Sobrecarga pandas independiente del modelo (adaptador simulado, `s6_overhead.py`, `s7_overhead_noexog.py`): 2000 series diarias, 10 folds: 18,3 s sin exog (1,8 s por fold) y 28,3 s con 2 columnas exog (2,8 s por fold). Los puntos calientes son `_extract_data_folds_multiseries` (2,0 s por fold), `_prepare_future_exog` (1,1 s por fold) y la reconstrucción del contexto con `.loc` y `notna` por serie.
- Hay que respetar `cross_learning=True` (no mezclar folds en el mismo lote, porque los contextos posteriores contienen el test de folds anteriores) y el tamaño de lote del backend. Esfuerzo M.

### 5. QUICK WIN: `save_forecaster` guarda los pesos del modelo si ya se ha predicho

- Los adaptadores cargan el backend de forma perezosa en `self._pipeline` o `self._model` y nada lo excluye del pickle.
- Medido (`s10_save.py`, chronos-2-small): 0,01 MB si se guarda antes de predecir, 111,8 MB tras un `predict`. Con chronos-2 base o TimesFM serían cientos de MB, y el pickle queda atado a la versión de torch y al dispositivo.
- Antes: `forecaster.predict(steps=7); save_forecaster(forecaster, "f.joblib")` # 112 MB.
- Después: el mismo código produce un fichero de 0,01 MB y el modelo se recarga al primer `predict` (implementar `__getstate__` en `_AdapterBase` que vacía el objeto cargado si no lo pasó el usuario; coordinar con el candidato 1).
- Afecta a quien guarda el forecaster para producción tras validarlo (flujo habitual pero no mayoritario). Esfuerzo S.

### 6. FUNCIONALIDAD: calibración conformal de los cuantiles nativos

- Los intervalos nativos de Chronos-2 small infracubren en los tres datasets medidos (`s8_coverage.py`, intervalo nominal del 80%, cuantiles 0,1 y 0,9 en backtesting): h2o mensual 0,750 (n=96), bike_sharing horario 0,711 (n=1440), fuel_consumption 0,688 (n=96).
- Los forecasters ML tienen `method='conformal'` con residuos fuera de muestra, pero `ForecasterFoundation.predict_interval` solo tiene cuantiles nativos (`probabilistic_methods: ["quantile_native"]`, `_forecaster_foundation.py:200`).
- Propuesta: `set_out_sample_residuals` alimentado desde `backtesting_foundation` y un `method='conformal'` (o un ajuste tipo CQR por nivel y horizonte) en `predict_interval`. Esfuerzo M. Encaja con GAP-backtesting-interval-metrics (que mide la cobertura) y la corrige.

## Otras fricciones observadas (no priorizadas)

- `bayesian_search_foundation` clona el forecaster (`_search.py:2248`), así que recarga los pesos una vez por búsqueda; si se busca sobre `model_id`, se recarga en cada cambio de checkpoint (documentado en la skill, Common Mistakes 7). Un caché por `model_id` lo evitaría.
- Con una sola serie, `predict` devuelve un DataFrame con columna `level` y no una Series como los forecasters ML; comparar ML y foundation obliga a remodelar la salida.
- Con `interval=0.999` Chronos-2 recorta los cuantiles a [0,01, 0,99] y solo avisa por `logging`, no con un warning de skforecast; los nombres internos de cuantiles salen como `q_0.0005000000000000004` por el cálculo `0.5 - interval/2`. Cosmético.
- No hay ninguna guía de fine-tuning ni de guardado de un forecaster foundation en la skill ni en los notebooks.
- Ya identificados y no repetidos aquí: `backtesting_foundation` sin `interval` ni métricas probabilísticas, falta de `grid_search_foundation`, T0 roto (#1385), TiRex-2 (#1341), TTM y `finetuned_checkpoint`.
