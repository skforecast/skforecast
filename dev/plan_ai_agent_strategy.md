# Skforecast AI Agent & Vibe-Coding Strategy

> Plan para maximizar la adopción de skforecast a través de asistentes AI, cubriendo desde desarrolladores que vibecodean hasta usuarios sin conocimientos de programación.

**Fecha de creación**: 2026-02-27  
**Estado**: Draft  
**Última revisión**: -

---

## Tabla de contenidos

1. [Resumen ejecutivo](#resumen-ejecutivo)
2. [Perfiles de usuario objetivo](#perfiles-de-usuario-objetivo)
3. [Opciones evaluadas](#opciones-evaluadas)
4. [Plan de acción por prioridad](#plan-de-acción-por-prioridad)
5. [Detalle de cada iniciativa](#detalle-de-cada-iniciativa)
6. [Comunicación y distribución](#comunicación-y-distribución)
7. [Qué NO hacer](#qué-no-hacer)

---

## Resumen ejecutivo

Hay tres perfiles de usuario que queremos cubrir:
1. **Desarrolladores que vibecodean**: Ya usan un IDE + LLM, necesitan que el LLM sepa usar skforecast correctamente.
2. **Analistas con algo de código**: Quieren copiar/pegar código funcional generado por AI.
3. **No-coders**: Quieren respuestas ("¿cuánto voy a vender mañana?") sin ver código.

La estrategia es **distribuir el mismo conocimiento de skforecast en múltiples formatos**, cada uno adaptado a una plataforma y perfil de usuario.

---

## Perfiles de usuario objetivo

| Perfil | Herramienta habitual | Qué necesita | Solución |
|--------|---------------------|--------------|----------|
| Dev vibecoder | VS Code, Cursor, Claude Code | Que el LLM genere código correcto de skforecast | Archivos de contexto en el repo |
| Analista | ChatGPT, Colab | Código end-to-end listo para ejecutar | Custom GPT + instrucciones ricas |
| No-coder | ChatGPT, Claude Desktop | Respuesta directa sin ver código | Custom GPT con Code Interpreter / MCP |

---

## Opciones evaluadas

| Opción | Pros | Contras | Veredicto |
|--------|------|---------|-----------|
| **Archivos de contexto en repo** (copilot-instructions.md, AGENTS.md, .cursor/rules) | Coste cero, máximo alcance para devs, se mantiene con la API | Limitado a devs con IDE | ✅ HACER (Prioridad 1) |
| **Custom GPT (ChatGPT)** | Alcance masivo, funciona para no-coders, coste cero de desarrollo | Requiere ChatGPT Plus, datos van a OpenAI, límites sandbox | ✅ HACER (Prioridad 1) |
| **Google Gem + Claude Project** | Cubre otros ecosistemas | Menor alcance que ChatGPT | ✅ HACER (Prioridad 2) |
| **MCP Server con tools** | Datos locales, sin límites, multi-plataforma | Desarrollo significativo (2-4 semanas), usuario necesita Python | ✅ HACER (Prioridad 3) |
| **Agente custom con backend propio** | Control total de la experiencia | Caro, requiere infra, compite con tools existentes del usuario | ❌ NO HACER |

---

## Plan de acción por prioridad

### 🔴 Prioridad 1 — Impacto inmediato, coste mínimo (1-2 días)

| # | Tarea | Estado | Notas |
|---|-------|--------|-------|
| 1.1 | Crear `AGENTS.md` en la raíz del repo | ⬜ Pendiente | Para Claude Code y OpenAI Codex. Contenido basado en copilot-instructions.md |
| 1.2 | Crear `.cursor/rules/skforecast.mdc` | ⬜ Pendiente | Para usuarios de Cursor IDE |
| 1.3 | Enriquecer instrucciones con workflows end-to-end | ⬜ Pendiente | Añadir a copilot-instructions.md, AGENTS.md y llms.txt |
| 1.4 | Añadir guía de decisión de forecaster | ⬜ Pendiente | Tabla "qué forecaster usar según el caso" |
| 1.5 | Añadir sección de errores comunes | ⬜ Pendiente | Errores frecuentes que cometen los LLMs |
| 1.6 | Crear Custom GPT en ChatGPT (Code Interpreter) | ⬜ Pendiente | System prompt + llms.txt como knowledge |
| 1.7 | Probar Custom GPT con datasets reales | ⬜ Pendiente | Validar que genera código correcto y respuestas claras |

### 🟡 Prioridad 2 — Ampliar ecosistema (1 semana)

| # | Tarea | Estado | Notas |
|---|-------|--------|-------|
| 2.1 | Crear Google Gem para Gemini | ⬜ Pendiente | Mismas instrucciones adaptadas |
| 2.2 | Crear Claude Project con instrucciones | ⬜ Pendiente | Para usuarios de Claude |
| 2.3 | Publicar Custom GPT en GPT Store | ⬜ Pendiente | Requiere pruebas previas (1.7) |
| 2.4 | Añadir links en README, docs y web | ⬜ Pendiente | Sección "AI Assistants" |
| 2.5 | Crear notebook de demos para el Custom GPT | ⬜ Pendiente | Casos de uso típicos documentados |

### 🟢 Prioridad 3 — MCP Server (1-2 meses)

| # | Tarea | Estado | Notas |
|---|-------|--------|-------|
| 3.1 | Diseñar API de tools de alto nivel | ⬜ Pendiente | Ver sección detallada abajo |
| 3.2 | Implementar tool `load_and_analyze_data` | ⬜ Pendiente | Detectar columnas, frecuencia, missing values |
| 3.3 | Implementar tool `forecast` | ⬜ Pendiente | Train + predict + intervals en una sola llamada |
| 3.4 | Implementar tool `compare_models` | ⬜ Pendiente | Comparar varios enfoques automáticamente |
| 3.5 | Implementar tool `explain_forecast` | ⬜ Pendiente | Tendencia, estacionalidad, importancia de features |
| 3.6 | Empaquetar como `skforecast-mcp` en PyPI | ⬜ Pendiente | Instalable con `pip install skforecast-mcp` |
| 3.7 | Documentar configuración para Claude Desktop / VS Code | ⬜ Pendiente | Guía step-by-step |
| 3.8 | Tests y validación con datasets variados | ⬜ Pendiente | |

---

## Detalle de cada iniciativa

### 1. Archivos de contexto en el repositorio

#### Estado actual (ya existe)
- `.github/copilot-instructions.md` — Muy completo en referencia de API
- `llms.txt` — Buen resumen para cualquier LLM

#### Archivos a crear

**`AGENTS.md`** (raíz del repo):
- Leído automáticamente por Claude Code y OpenAI Codex
- Contenido igual a copilot-instructions.md + workflows end-to-end

**`.cursor/rules/skforecast.mdc`**:
- Leído automáticamente por Cursor IDE
- Formato con frontmatter:
```yaml
---
description: Rules for working with skforecast time series forecasting library
globs: ["**/*.py", "**/*.ipynb"]
---
```

#### Contenido a añadir en todos los archivos de instrucciones

**A) Workflows end-to-end** (el usuario dice QUÉ quiere, no CÓMO):

Se deben incluir scripts completos y funcionales para estos escenarios:
- "Quiero predecir una serie temporal" → script completo con ForecasterRecursive
- "Tengo varias series y quiero predecirlas todas" → script con ForecasterRecursiveMultiSeries
- "Quiero encontrar los mejores hiperparámetros" → script con bayesian_search_forecaster
- "Quiero usar ARIMA" → script con ForecasterStats + Arima
- "Quiero predecir con intervalos de confianza" → script con predict_interval + backtesting

Cada workflow debe incluir: imports, instalación, carga de datos, split, entrenamiento, predicción, evaluación y visualización.

**B) Guía de decisión de forecaster**:

| Situación | Forecaster | Por qué |
|-----------|-----------|---------|
| Una serie, caso general | ForecasterRecursive | Default, rápido y flexible |
| Una serie, horizonte largo | ForecasterDirect | Modelo independiente por step |
| Múltiples series relacionadas | ForecasterRecursiveMultiSeries | Patrones compartidos |
| Necesito ARIMA/ETS/SARIMAX | ForecasterStats + Arima/Ets/Sarimax | Modelos estadísticos |
| Múltiples inputs → un output | ForecasterDirectMultiVariate | Multivariante |
| Clasificación (sube/baja) | ForecasterRecursiveClassifier | Predicción categórica |
| Baseline rápido | ForecasterEquivalentDate | Referencia de fechas equivalentes |
| Deep learning (RNN/LSTM) | ForecasterRnn | Redes neuronales, requiere Keras |

**C) Errores comunes que los LLMs suelen cometer**:

| Error | Causa | Solución |
|-------|-------|----------|
| `"y must have a frequency"` | DatetimeIndex sin freq | `data = data.asfreq('D')` |
| `"exog must have same index as y"` | Exog no cubre horizonte | Asegurar que exog tiene fechas futuras |
| `ImportError: ForecasterAutoreg` | Import de API antigua depreciada | `from skforecast.recursive import ForecasterRecursive` |
| `ImportError: ForecasterAutoregMultiSeries` | Import de API antigua depreciada | `from skforecast.recursive import ForecasterRecursiveMultiSeries` |
| NaN en predictions | Missing values en datos | `data = data.fillna(...)` o interpolación antes de fit |
| `steps` no definidos en Direct | ForecasterDirect requiere steps en init | `ForecasterDirect(..., steps=10)` |
| `exog` shape mismatch | Exog de predict no tiene el mismo número de columnas | Verificar que exog tenga las mismas features que en fit |
| Usar `cv=TimeSeriesFold(...)` sin `steps` | `steps` es obligatorio | Siempre pasar `steps=N` a TimeSeriesFold |

---

### 2. Custom GPT (ChatGPT con Code Interpreter)

#### System prompt

```
You are the official Skforecast AI assistant. You help users perform time series 
forecasting using the skforecast Python library, even if they have zero programming 
experience.

## Your behavior:
1. When a user asks a forecasting question, use Code Interpreter to run real Python 
   code with skforecast. Install it silently with: !pip install skforecast lightgbm -q
2. NEVER show code to the user unless they explicitly ask for it.
3. Answer in PLAIN LANGUAGE with the prediction, confidence interval, and a chart.
4. If the user uploads a file, auto-detect the date column, frequency, and target.
5. If accuracy is poor, say so honestly and suggest what data would help.
6. Always validate with backtesting before giving a final prediction.

## When the user uploads a file:
- Auto-detect date column (parse dates), frequency, and target variable
- If ambiguous, ask ONE simple question to clarify (don't ask multiple questions)
- Handle missing values automatically (interpolation)
- Default to LightGBM + ForecasterRecursive with lag selection up to 2x seasonal period

## Response format for predictions:
"Based on your historical data, the forecast for [period] is **[value]** 
(estimated range: [lower] – [upper]).
The model's average error on historical data is [metric in intuitive units]."

Always include a clean chart showing: historical data, forecast, and confidence interval.

## When the user asks for code:
Generate complete, runnable code compatible with Google Colab. Include all imports 
and !pip install commands at the top.

## Critical rules:
- Use ONLY current skforecast v0.21.0+ API (see knowledge files)
- NEVER use deprecated imports like ForecasterAutoreg
- Always set datetime index with frequency
- Always include backtesting validation to report model accuracy
- If the data has < 30 observations, warn that predictions will be unreliable
```

#### Knowledge files a subir
- `llms.txt` (el que ya tenemos)
- Opcionalmente: 1-2 notebooks de ejemplo exportados como .py o .md

#### Publicación
- Nombre: **"Skforecast - Time Series Forecasting"**
- Descripción: "Upload your data (CSV/Excel) and ask any forecasting question. Get predictions, confidence intervals, and visualizations — no coding required."
- Logo: Logo de skforecast
- Publicar en GPT Store como público

---

### 3. Google Gem + Claude Project

Mismo contenido que el Custom GPT, adaptado a cada plataforma:

- **Google Gem**: Pegamos las instrucciones en la configuración del Gem. No tiene Code Interpreter equivalente, así que se limita a generar código para que el usuario lo ejecute en Colab.
- **Claude Project**: Subimos llms.txt y el prompt como instrucciones del proyecto. Claude no ejecuta código, pero genera código muy preciso si tiene el contexto.

---

### 4. MCP Server (`skforecast-mcp`)

#### Filosofía de diseño
- Tools de **ALTO NIVEL**: el LLM no programa skforecast, lo usa como herramienta
- Cada tool es una operación completa (no exponer .fit()/.predict() por separado)
- El LLM decide cuándo usar cada tool basándose en la pregunta del usuario

#### Tools a implementar

```python
@tool
def load_and_analyze_data(file_path: str) -> dict:
    """
    Load a CSV/Excel file and return a summary of its contents.
    
    Returns: {
        columns: [...],
        date_column: str,
        numeric_columns: [...],
        frequency: str,
        date_range: {start: str, end: str},
        n_rows: int,
        missing_values: {column: count},
        basic_stats: {column: {mean, std, min, max}}
    }
    """

@tool
def forecast(
    file_path: str,
    target_column: str,
    date_column: str,
    steps: int,
    frequency: str = "auto",
    exog_columns: list[str] | None = None,
    confidence_level: float = 0.80
) -> dict:
    """
    Train a forecaster and return predictions with confidence intervals.
    Automatically selects model, lags, and validates with backtesting.
    
    Returns: {
        predictions: [{date: str, value: float, lower: float, upper: float}],
        model_used: str,
        lags_used: list[int],
        backtesting_metric: {name: str, value: float},
        plot_path: str  # Path to saved chart
    }
    """

@tool
def compare_models(
    file_path: str,
    target_column: str,
    date_column: str,
    steps: int,
    frequency: str = "auto"
) -> dict:
    """
    Compare multiple forecasting approaches and return a ranked results table.
    Tests: LightGBM, RandomForest, Ridge, ARIMA, ETS.
    
    Returns: {
        results: [{model: str, metric: float, training_time: float}],
        best_model: str,
        recommendation: str
    }
    """

@tool
def explain_forecast(
    file_path: str,
    target_column: str,
    date_column: str,
    frequency: str = "auto"
) -> dict:
    """
    Analyze a time series and explain its patterns.
    
    Returns: {
        trend: str,  # "increasing", "decreasing", "stable"
        seasonality: {detected: bool, period: int, description: str},
        feature_importance: [{feature: str, importance: float}],
        stationarity: {is_stationary: bool, adf_pvalue: float},
        summary: str  # Human-readable paragraph
    }
    """
```

#### Estructura del paquete

```
skforecast-mcp/
├── pyproject.toml
├── README.md
├── src/
│   └── skforecast_mcp/
│       ├── __init__.py
│       ├── server.py          # MCP server entry point
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── load_data.py
│       │   ├── forecast.py
│       │   ├── compare.py
│       │   └── explain.py
│       └── utils/
│           ├── __init__.py
│           ├── auto_detect.py  # Auto-detect columns, frequency
│           └── plotting.py     # Generate charts
└── tests/
```

#### Distribución
- Publicar en PyPI como `skforecast-mcp`
- El usuario instala: `pip install skforecast-mcp`
- Configuración en Claude Desktop (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "skforecast": {
      "command": "python",
      "args": ["-m", "skforecast_mcp"]
    }
  }
}
```
- Configuración en VS Code (settings.json):
```json
{
  "mcp": {
    "servers": {
      "skforecast": {
        "command": "python",
        "args": ["-m", "skforecast_mcp"]
      }
    }
  }
}
```

---

## Comunicación y distribución

### En el README del repo

Añadir una sección visible:

```markdown
## 🤖 AI-Assisted Forecasting

**No coding experience?** Use our AI assistants:
- **ChatGPT**: [Skforecast GPT](link) — Upload your CSV and ask your question
- **Any LLM**: Pass [https://skforecast.org/llms.txt](https://skforecast.org/llms.txt) as context

**Developer using AI coding tools?** Skforecast works out of the box:
- **VS Code Copilot**: Instructions auto-loaded from `.github/copilot-instructions.md`
- **Cursor**: Instructions auto-loaded from `.cursor/rules/`
- **Claude Code / Codex**: Instructions auto-loaded from `AGENTS.md`
```

### En la documentación web (skforecast.org)

Crear una página "AI Assistants" en la sección de user guides con:
- Links a todos los agentes
- Instrucciones para usar skforecast con cada herramienta AI
- Ejemplos de prompts que funcionan bien

### En redes sociales / blog

Post de lanzamiento: *"Ahora puedes hacer forecasting sin escribir código. Sube tu CSV a nuestro GPT y pregúntale cuánto vas a vender mañana."*

---

## Qué NO hacer

| Idea | Por qué no |
|------|-----------|
| Crear un agente custom con backend/servidor propio | Coste de infra enorme para un proyecto OSS, el usuario tiene que confiar en un tercero con su API key |
| Exponer la API de bajo nivel de skforecast como tools del MCP | Demasiado complejo para que un LLM lo use correctamente en nombre de un no-coder |
| Construir una web app completa para no-coders | Eso es un producto entero (ya existe Skforecast Studio) |
| Mantener un solo formato de instrucciones | Cada plataforma tiene su convención; hay que cubrirlas todas |
| Crear instrucciones solo con referencia de API | Los LLMs necesitan workflows completos end-to-end, no solo documentación |

---

## Cronograma estimado

| Semana | Tareas |
|--------|--------|
| **Semana 1** | 1.1-1.5: Crear AGENTS.md, .cursor/rules, enriquecer instrucciones |
| **Semana 1** | 1.6-1.7: Crear y probar Custom GPT |
| **Semana 2** | 2.1-2.5: Google Gem, Claude Project, publicar GPT, actualizar docs |
| **Semana 3-4** | Recoger feedback de usuarios y iterar sobre instrucciones y GPT |
| **Mes 2-3** | 3.1-3.8: Diseñar, implementar y publicar MCP Server |

---

## Métricas de éxito

- Número de usos del Custom GPT (GPT Store analytics)
- Issues / PRs que mencionen AI-assisted usage
- Descargas de `skforecast-mcp` en PyPI (cuando se lance)
- Feedback cualitativo de usuarios en GitHub Discussions / Twitter
- Reducción de issues causados por imports deprecados (señal de que los LLMs generan código correcto)
