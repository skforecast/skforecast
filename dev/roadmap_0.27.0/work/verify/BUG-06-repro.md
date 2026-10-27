# BUG-06: verificación

**Veredicto: keep (esfuerzo M), prioridad media.** Se reproduce con la configuración por defecto de TabICL y Nori (`point_estimate='mean'`) y con TabPFN si `point_estimate` no es `'median'`: `predict()` devuelve la media, mientras que `predict_interval()['pred']` y la métrica de `backtesting_foundation(..., quantiles=...)` usan el cuantil 0.5. No es un valor absurdo (la mediana es un punto válido), pero es una inconsistencia silenciosa de métricas en el flujo habitual: la misma configuración da dos MAE distintos según se pidan o no cuantiles, el `point_estimate` que ajusta `bayesian_search_foundation` (evalúa sin cuantiles) se ignora al pedir intervalos, y en ML `predict_interval()['pred']` es exactamente `predict()`.

Script: `dev/roadmap_0.27.0/repro/BUG-06.py` (parte A con un backend Nori falso, determinista y sin red; parte B referencia ML; parte C TabICL real si está instalado). Entorno: skforecast 0.26.0, tabicl 2.2.0 (checkpoint `tabicl-regressor-v2-20260212.ckpt`, instalado con `uv pip install --no-deps`), torch 2.14.1+cpu, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1.

## Resultados medidos

| Caso | `predict()` | `predict_interval()['pred']` / métrica con cuantiles |
|---|---|---|
| A. Nori falso, serie lognormal | pred 1.438 | pred 1.003; MAE backtesting 1.1693 sin cuantiles frente a 1.0834 con cuantiles; con cuantiles desaparece la columna `pred` |
| A. Mismo forecaster con `point_estimate='median'` | pred 1.003 | pred 1.003 (idéntico al caso `'mean'`: el parámetro no influye en `predict_interval`) |
| B. `ForecasterRecursive` (bootstrapping) | [1.619, 1.66, 1.338] | idéntico (referencia esperada) |
| C. TabICL real, demanda Poisson intermitente diaria (8 folds de 7) | MAE 0.6994 | MAE 0.6760 (+3.5 %); diferencia media entre puntos 0.244 (31 % de la demanda media) |

Medición adicional con TabICL real (script `scratchpad/bug06/real_bt.py`, mismo protocolo, `refit=False`):

| Serie | MAE sin cuantiles (media) | MAE con cuantiles (q_0.5) | Diferencia |
|---|---|---|---|
| h2o mensual (suave) | 0.0713 | 0.0703 | +1.5 % |
| bike_sharing `users` horaria | 40.85 | 42.86 | -4.7 % |
| items_sales `item_2` diaria | 2.856 | 2.750 | +3.9 % |
| Poisson intermitente diaria | 0.699 | 0.676 | +3.5 % |

El signo cambia según la serie, así que la diferencia basta para invertir una comparación entre modelos o entre valores de `point_estimate`.

**Corrección a la candidata:** TimesFM 2.5 **no** está afectado. En timesfm (`timesfm_2p5_torch.py:513` y `timesfm_2p5_flax.py:598`, versión 3.0.2) `forecast` devuelve `full_forecast[..., 5]`, es decir, el cuantil 0.5, y el adaptador pide el índice `round(0.5 * 10) = 5`. Sí es errónea la documentación: la tabla de la guía dice "Mean (dedicated output)" para TimesFM 2.5 (`docs/user_guides/foundation-forecasting-models.ipynb:466`) y el comentario de `_adapters.py:916` induce a pensarlo. Afectados reales: TabICL y Nori por defecto, TabPFN con `point_estimate` en `'mean'` o `'mode'`. Chronos-2, TimesFM 2.5, TimesFM 3, Moirai-2, T0 y TS-ICL usan la mediana en ambos caminos.

## Causa raíz

- `skforecast/foundation/_forecaster_foundation.py:946-960`: `predict_interval` fuerza el cuantil 0.5, llama a `predict_quantiles` y renombra `q_0.5` a `pred`.
- `skforecast/model_selection/_validation.py:2590-2599` (llama a `predict_quantiles`, sin `pred`), `:2650-2653` (multiserie, `metric_col = 'q_0.5'`) y `:2683-2686` (una serie, `y_pred = ...['q_0.5']`); `:2851-2859` añade 0.5 con un `IgnoredArgumentWarning`.
- Adaptadores: `_adapters.py:2528-2531` (TabICL toma `target`, la media, solo sin cuantiles), `:4826-4835` (Nori, `output_type=self.point_estimate` solo sin cuantiles), `:3187` y `:3229` (TabPFN, `target` según `tabpfn_output_selection`).
- `skforecast/foundation/_foundation_model.py:1041,1058`: el docstring dice "point forecast (median)", falso para TabICL y Nori.
- `skforecast/model_selection/_search.py:2328`: `bayesian_search_foundation` evalúa sin cuantiles, sobre `predict()`.

## Arreglo propuesto

Centralizado en `FoundationModel` (regla del módulo):

1. Contrato: `FoundationModel.predict(..., quantiles, include_point=False)`. Con `include_point=True` devuelve `pred` junto a las columnas `q_*`. Si el punto del adaptador es la mediana (nuevo atributo o propiedad del adaptador, por ejemplo `point_is_median`, `True` por defecto y `self.point_estimate == 'median'` en TabICL, TabPFN y Nori), `FoundationModel` copia `q_0.5` (añadiéndolo a la consulta si falta) sin tocar el adaptador. Si no, pide al adaptador el punto en la misma llamada: TabICL ya lo calcula a la vez (`tabicl/forecast/_engine.py:68-70`, `output_type=[point_estimate, "quantiles"]`) y TabPFN-TS devuelve `target` junto a los cuantiles; Nori necesita una segunda llamada a `predict` sobre el mismo ajuste en contexto.
2. `predict_interval`: pedir solo `[lower, upper]` con `include_point=True` y usar ese `pred`; deja de forzar 0.5.
3. `_backtesting_foundation`: con `quantiles`, conservar `pred` y medir sobre ella (salida `['level', 'fold', 'pred', 'q_...']`); quitar el 0.5 forzado y su aviso. Encaja con GAP-foundation-backtesting-interval: si se añade `interval` antes, `pred` debe ser ya el punto de `predict()`.
4. Docs: docstrings de `_foundation_model.py:1041,1058`, tabla de la guía (TimesFM 2.5: mediana), comentario de `_adapters.py:916`, skill `foundation-forecasting` y nota "Changed" en `releases.md` (cambian las métricas con cuantiles para TabICL y Nori).
5. Tests con un adaptador falso cuya media difiera de la mediana: `predict_interval()['pred'] == predict()['pred']`, misma métrica con y sin cuantiles, columna `pred` presente.

Alternativa mínima (S) si no entra en la 0.27: documentar que `pred` de `predict_interval` y la métrica con cuantiles son la mediana y avisar en `backtesting_foundation` cuando `point_estimate != 'median'`. Es peor: deja la asimetría con ML y la traslada a la futura API con `interval`.
