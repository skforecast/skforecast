# Plan: Custom GPT — Skforecast Forecasting Assistant

> Guía paso a paso para crear y publicar el Custom GPT de skforecast en ChatGPT.
>
> **Estado**: Draft
> **Fecha**: 2026-03-09
> **Prerequisitos completados**: `llms-full.txt` generado (3,579 líneas), 12 skills, docs page

---

## Tabla de contenidos

1. [Decisiones de diseño](#1-decisiones-de-diseño)
2. [Preparación del knowledge file](#2-preparación-del-knowledge-file)
3. [System prompt](#3-system-prompt)
4. [Configuración en ChatGPT](#4-configuración-en-chatgpt)
5. [Conversation starters](#5-conversation-starters)
6. [Testing y validación](#6-testing-y-validación)
7. [Publicación en GPT Store](#7-publicación-en-gpt-store)
8. [Mantenimiento](#8-mantenimiento)

---

## 1. Decisiones de diseño

### Perfil de usuario objetivo

El Custom GPT cubre **dos perfiles** que no tienen acceso al repo:

| Perfil | Necesita | Qué espera |
|--------|----------|-------------|
| **Analista con algo de código** | Workflows completos copy-paste | Código funcional que pueda ejecutar en Colab/Jupyter |
| **No-coder** | Respuestas directas | "¿Cuánto vendo mañana?" sin ver código |

Los devs con IDE ya están cubiertos por los archivos de contexto del repo.

### Capacidades a habilitar

| Capacidad | Por qué |
|-----------|---------|
| **Code Interpreter** (obligatorio) | Ejecutar código Python en sandbox, generar gráficos, procesar CSVs del usuario |
| **File Upload** | El usuario sube su CSV/Excel y el GPT lo analiza |
| **Web Search** | Buscar documentación actualizada si algo no está en el knowledge file |
| **DALL-E** | No necesario — desactivar para no confundir |

### Knowledge file: `llms-full.txt`

- **Tamaño**: ~3,579 líneas, ~120 KB — dentro del límite de ChatGPT (hasta ~2M caracteres / ~50 archivos)
- **Contenido**: API completa + 12 workflows + referencia de métodos
- **Un solo archivo** es preferible a múltiples fragmentos: ChatGPT retrieval funciona mejor con un documento coherente que con fragmentos sueltos

### Nombre

Opciones (máx 50 caracteres en GPT Store):

| Opción | Pros | Contras |
|--------|------|---------|
| **Skforecast** | Nombre oficial, conciso | Poco descriptivo para quienes no lo conocen |
| **Skforecast - Time Series Forecasting** | Descriptivo + nombre oficial | Un poco largo |
| **Time Series Forecasting with Skforecast** | SEO-friendly, descriptivo | Nombre de librería al final |

**Recomendación**: "Skforecast - Time Series Forecasting"

---

## 2. Preparación del knowledge file

### 2.1 Generar la versión más reciente

```bash
python tools/ai/generate_ai_context_files.py
python tools/ai/generate_ai_context_files.py --check
```

### 2.2 Archivo a subir

Usar directamente: **`llms-full.txt`** (raíz del repo)

No se necesita packaging adicional. El archivo ya contiene:
- Descripción general y versión
- Estructura del proyecto
- Imports y API de todos los forecasters
- 12 workflows completos (single series, multi-series, hyperparameter tuning, etc.)
- Referencia de métodos con firmas
- Errores comunes y soluciones

### 2.3 Verificación pre-upload

Antes de subir, verificar que:
- [ ] La versión en el archivo coincide con la versión publicada en PyPI
- [ ] Los imports funcionan en un entorno limpio
- [ ] No contiene marcadores `AUTO-GENERATED` (no molestar al usuario — **mantenerlos**: son solo comentarios HTML, ChatGPT los ignora)

---

## 3. System prompt

El system prompt es lo más crítico. Define la personalidad, restricciones y comportamiento.

### Principios del system prompt

1. **Conciso** — ChatGPT tiene límite de ~8,000 caracteres para instructions. Ser directo.
2. **Referir al knowledge** — No repetir lo que ya está en `llms-full.txt`. Decirle que lo consulte.
3. **Code Interpreter first** — Siempre que sea posible, ejecutar código, no solo mostrarlo.
4. **Adaptarse al usuario** — Si sube un CSV, analizarlo. Si pregunta en español, responder en español.
5. **Defensivo** — Validar que el código funcione antes de entregarlo. Manejar errores comunes.

### System prompt (draft)

```
You are an expert time series forecasting assistant powered by skforecast, a Python library for time series forecasting with machine learning. Your knowledge file contains the complete API reference and workflows.

## ABSOLUTE RULES (never violate these)

- RUN CODE IMMEDIATELY. Do not describe what you will do. Do not say "I'll start by..." or "Let me...". Write the code, execute it, and show the output. Act first, explain after.
- NEVER fabricate results, metrics, or numbers you haven't computed
- ALWAYS consult your knowledge file for imports, parameters, and API usage

## Core behavior

1. Install skforecast at the start of every session using this exact command:
   ```
   pip install skforecast --quiet
   ```
   If installation fails (timeout, build error), retry with:
   ```
   pip install skforecast --quiet --no-cache-dir
   ```
   If it still fails, install without optional compiled dependencies:
   ```
   pip install skforecast --quiet --no-cache-dir --no-deps
   pip install joblib tqdm rich optuna
   ```
   The sandbox already has numpy, pandas, scikit-learn, scipy, and matplotlib pre-installed.
   For LightGBM: `pip install lightgbm --quiet`. For statsmodels (ARIMA/ETS): `pip install statsmodels --quiet`
2. When the user uploads a CSV/Excel, immediately load it and show .head(), .info(), missing values, and a plot — no preamble
3. Adapt your language to match the user's language (Spanish, English, etc.)
4. Be concise and action-oriented — don't end every response with a menu of options

## Workflow for forecasting requests

CRITICAL: Work step by step. Execute ONE step, show the result, explain it briefly, and then move to the next step. NEVER dump the entire pipeline in a single response. The user needs to see and understand each result before proceeding.

1. **Load & explore**: Read the data, identify the datetime column, set it as index with frequency. Show .head(), .info(), missing values count, and a plot of the series. STOP here and explain what you see (trend, seasonality, anomalies, frequency)
2. **Recommend a forecaster**: Based on what you observed, recommend a forecaster and explain why. Then train a baseline model with sensible defaults (e.g., lags=seasonal_period). Show feature importance or lag significance if relevant
3. **Evaluate with backtesting**: Run backtesting with TimeSeriesFold. Show the metric and a plot of predicted vs actual. Explain whether the result is good or not for this data
4. **Predict**: Generate forecasts with prediction intervals and plot them. Explain the intervals

Only suggest hyperparameter tuning, rolling features, or model comparison if the user asks for it or the baseline results are clearly poor. Don't overwhelm with optimization before the user has seen basic results.

## Key technical rules

- skforecast is a MACHINE LEARNING library for time series. Always default to ML models with ForecasterRecursive as the first recommendation. Statistical models (ARIMA, ETS) via ForecasterStats are an alternative to compare against, not the default choice
- Default estimator: Use LightGBM (LGBMRegressor) as the first choice — it is fast, handles large datasets well, and works within the Code Interpreter time limits. RandomForest is too slow for hourly or large datasets in this sandbox. Install with: pip install lightgbm --quiet
- The Code Interpreter sandbox has limited CPU and short timeouts. Keep models lightweight: max 100-200 trees, avoid n_jobs=-1 (only 1 CPU available). For datasets over 5,000 rows, LightGBM is strongly preferred over RandomForest or XGBoost
- skforecast requires a pandas Series/DataFrame with DatetimeIndex and frequency set (use .asfreq())
- NaN values must be handled before fitting
- For prediction intervals, use predict_interval() with method='bootstrapping' or 'conformal'
- For multi-series, use ForecasterRecursiveMultiSeries with encoding='ordinal' as default
- For hyperparameter tuning, prefer bayesian_search_forecaster with TimeSeriesFold
- NEVER use old class names (ForecasterAutoreg, ForecasterAutoregMultiSeries) — always use current names

## What you can do

- Analyze uploaded time series data (CSV, Excel)
- Build forecasting models with any sklearn-compatible regressor
- Compare multiple models and strategies
- Generate prediction intervals
- Tune hyperparameters automatically
- Explain results and methodology
- Create publication-ready plots

## What you should NOT do

- NEVER narrate your plan before acting. Wrong: "I'll load your data and then..." Right: [immediately run the code and show output]
- NEVER fabricate metrics, numbers, or approximate results you haven't computed
- NEVER end responses with long menus of emoji options
- Don't recommend other forecasting libraries unless the user's problem is outside skforecast's scope
- Don't fabricate parameters or methods not in your knowledge file
- Don't skip data exploration — always show the data first
- Don't present results without backtesting validation
```

### Notas sobre el system prompt

- **~2,000 caracteres** — bien dentro del límite
- No repite contenido de `llms-full.txt` — solo define comportamiento
- `pip install skforecast` es necesario porque el sandbox de Code Interpreter empieza limpio cada sesión
- El prompt es en inglés porque ChatGPT respeta mejor las instrucciones en inglés, pero se adapta al idioma del usuario vía punto 4

---

## 4. Configuración en ChatGPT

### Paso a paso en chat.openai.com

1. Ir a **[chat.openai.com/gpts/mine](https://chat.openai.com/gpts/mine)** → **Create a GPT**
2. Pestaña **Configure** (no usar el chat wizard, es menos preciso)
3. Rellenar:

| Campo | Valor |
|-------|-------|
| **Name** | Skforecast - Time Series Forecasting |
| **Description** | Expert assistant for time series forecasting with Python and skforecast. Upload your data (CSV/Excel) and get forecasts, model comparison, hyperparameter tuning, and prediction intervals. Works with any sklearn-compatible model. |
| **Instructions** | Pegar el [system prompt del punto 3](#system-prompt-draft) |
| **Conversation starters** | Ver [punto 5](#5-conversation-starters) |
| **Knowledge** | Subir `llms-full.txt` |
| **Capabilities** | ✅ Code Interpreter & Data Analysis, ✅ Web Search, ❌ Image Generation |
| **Actions** | Ninguna (no necesitamos API calls externas) |

4. **Profile picture**: Usar el logo de skforecast (buscar en `images/` o `docs/img/`)

### Verificar la imagen de perfil

```bash
# Buscar logos disponibles en el repo
ls images/
ls docs/img/
```

---

## 5. Conversation starters

Los conversation starters son los botones que ve el usuario al abrir el GPT. Deben cubrir los casos de uso principales:

| Starter | Perfil objetivo | Qué demuestra |
|---------|----------------|----------------|
| 📈 "Forecast my time series — I'll upload a CSV" | Analista | File upload + workflow completo |
| 🔍 "Help me choose the right forecaster for my problem" | Analista | Guía de decisión (skill choosing-a-forecaster) |
| 📊 "Show me how to forecast electricity demand hourly" | Analista | Workflow con dataset built-in |
| ⚡ "Compare multiple forecasting models on my data" | Analista avanzado | Backtesting + comparación |

---

## 6. Testing y validación

### 6.1 Test suite mínima

Antes de publicar, probar estos **10 prompts** y verificar que el GPT responde correctamente:

#### Funcionalidad básica

| # | Prompt | Resultado esperado |
|---|--------|--------------------|
| 1 | "Forecast my time series" + subir un CSV con columna date y columna value | Carga datos, plot, entrena ForecasterRecursive, backtesting, predicción con gráfico |
| 2 | "¿Qué forecaster debería usar si tengo 50 tiendas con datos diarios?" | Recomienda ForecasterRecursiveMultiSeries, explica encoding |
| 3 | "Show me a complete example with the bike sharing dataset" | Usa `fetch_dataset('bike_sharing')`, incluye exog variables |
| 4 | "Tune hyperparameters for my model" + CSV | bayesian_search_forecaster con TimeSeriesFold |
| 5 | "I need prediction intervals for my forecast" | predict_interval con method='bootstrapping' |

#### Robustez

| # | Prompt | Resultado esperado |
|---|--------|--------------------|
| 6 | Subir CSV con fechas desordenadas y NaNs | Detecta problemas, ordena, imputa o avisa |
| 7 | "Use ForecasterAutoreg" (nombre obsoleto) | Corrige a ForecasterRecursive |
| 8 | "Forecast with ARIMA" | Usa ForecasterStats con Arima, no statsmodels directo |
| 9 | Subir Excel con múltiples hojas | Pregunta qué hoja usar o analiza todas |
| 10 | "Forecast the next 365 days" con datos horarios | Gestiona el horizonte largo razonablemente |

#### Idioma

| # | Prompt | Resultado esperado |
|---|--------|--------------------|
| 11 | "Predice mis ventas para el próximo mes" + CSV | Responde en español, mismo workflow |

### 6.2 Criterios de aprobación

- [ ] Los 11 prompts producen código que **ejecuta sin errores** en Code Interpreter
- [ ] Los imports son correctos (no usa nombres obsoletos)
- [ ] Siempre hace backtesting antes de presentar predicciones como fiables
- [ ] Los gráficos son legibles
- [ ] Responde en el idioma del usuario
- [ ] No inventa parámetros que no existen

### 6.3 Datasets de prueba

Preparar 3-4 CSVs para testing:

| Dataset | Características | Para probar |
|---------|----------------|-------------|
| Ventas mensuales (single series) | 120 filas, sin exog, mensual | Caso básico |
| Energía horaria (con exog) | 8,760 filas, 3 exog, horaria | Exog variables, alta frecuencia |
| Multi-tienda (multi-series) | 3 series, diaria | ForecasterRecursiveMultiSeries |
| Datos "sucios" | NaNs, fechas desordenadas, frecuencia irregular | Robustez |

Se pueden generar con los datasets built-in de skforecast:
```python
from skforecast.datasets import fetch_dataset
# Guardar como CSV para subir al GPT durante tests
fetch_dataset('h2o').to_csv('test_monthly.csv')
fetch_dataset('bike_sharing')[['y', 'temp', 'atemp', 'hum']].to_csv('test_hourly_exog.csv')
fetch_dataset('items_sales').to_csv('test_multiseries.csv')
```

---

## 7. Publicación en GPT Store

### Requisitos previos (OpenAI)

- [ ] Cuenta ChatGPT Plus/Team/Enterprise
- [ ] Perfil de builder verificado (Settings → Builder profile → domain o social link)
- [ ] GPT testeado y estable

### Pasos

1. En la configuración del GPT, sección **Sharing** → seleccionar **"Public"**
2. Completar:
   - **Category**: "Programming" o "Data Analysis"
   - **Tags**: time series, forecasting, machine learning, python, skforecast
3. **Save & Publish**

### SEO en GPT Store

- El **nombre** y la **description** son lo más importante para discovery
- Incluir keywords: "time series", "forecasting", "prediction", "machine learning", "Python"
- La description actual ya los incluye

### Responder a reviews/feedback

- Monitorizar los ratings periódicamente
- Actualizar el knowledge file cuando salga nueva versión

---

## 8. Mantenimiento

### Proceso de actualización (cada release de skforecast)

```
1. Regenerar llms-full.txt:
   python tools/ai/generate_ai_context_files.py

2. En chat.openai.com/gpts/mine → editar GPT:
   - Knowledge → eliminar llms-full.txt anterior
   - Knowledge → subir nuevo llms-full.txt
   - Verificar que la versión en el system prompt sigue siendo correcta
   - Save

3. Ejecutar 3-4 prompts del test suite para verificar
```

### Automatización posible (futuro)

OpenAI tiene una API de GPTs (`/v1/assistants`) que permite actualizar knowledge files programáticamente. Se podría integrar en el CI:

```bash
# Hipotético — cuando OpenAI lo soporte completamente
python tools/ai/update_custom_gpt.py --knowledge llms-full.txt
```

Por ahora, la actualización manual es rápida (~5 minutos) y solo ocurre en cada release.

---

## Checklist resumen

| Paso | Acción | Estado |
|------|--------|--------|
| 1 | Regenerar `llms-full.txt` con versión actual | ⬜ |
| 2 | Crear GPT en ChatGPT con configuración del punto 4 | ⬜ |
| 3 | Pegar system prompt | ⬜ |
| 4 | Subir `llms-full.txt` como knowledge | ⬜ |
| 5 | Configurar conversation starters | ⬜ |
| 6 | Subir logo de skforecast como profile picture | ⬜ |
| 7 | Ejecutar 11 prompts del test suite | ⬜ |
| 8 | Corregir problemas encontrados en testing | ⬜ |
| 9 | Re-test tras correcciones | ⬜ |
| 10 | Verificar builder profile en OpenAI | ⬜ |
| 11 | Publicar como Public en GPT Store | ⬜ |
| 12 | Añadir link al GPT en docs y README | ⬜ |
