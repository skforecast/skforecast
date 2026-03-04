# Skforecast AI Agent & Vibe-Coding Strategy

> Plan para maximizar la adopción de skforecast a través de asistentes AI, cubriendo desde desarrolladores que vibecodean hasta usuarios sin conocimientos de programación.

**Fecha de creación**: 2026-02-27  
**Estado**: Draft  
**Última revisión**: 2026-03-03

---

## Tabla de contenidos

1. [Resumen ejecutivo](#resumen-ejecutivo)
2. [Perfiles de usuario objetivo](#perfiles-de-usuario-objetivo)
3. [Principio fundamental: fuente única de verdad](#principio-fundamental-fuente-única-de-verdad)
4. [Opciones evaluadas](#opciones-evaluadas)
5. [Plan de acción por prioridad](#plan-de-acción-por-prioridad)
6. [Detalle de cada iniciativa](#detalle-de-cada-iniciativa)
7. [Comunicación y distribución](#comunicación-y-distribución)
8. [Qué NO hacer](#qué-no-hacer)

---

## Resumen ejecutivo

Hay tres perfiles de usuario que queremos cubrir:
1. **Desarrolladores que vibecodean**: Ya usan un IDE + LLM, necesitan que el LLM sepa usar skforecast correctamente.
2. **Analistas con algo de código**: Quieren copiar/pegar código funcional generado por AI.
3. **No-coders**: Quieren respuestas ("¿cuánto voy a vender mañana?") sin ver código.

La estrategia es **mantener el conocimiento de skforecast en dos archivos fuente** (`llms.txt` y `llms-full.txt`) y **generar automáticamente** todos los archivos de contexto para cada plataforma mediante un script. Esto garantiza consistencia y elimina el riesgo de desincronización cuando la API cambie.

### Canales de impacto (de mayor a menor alcance)

| Canal | Alcance | Quién lo ve |
|-------|---------|-------------|
| **Training data** (docs web, Stack Overflow, Kaggle, blogs) | Todo usuario de cualquier LLM | Usuarios que nunca han oído de skforecast |
| **`llms.txt` en skforecast.org** | LLMs con web search (ChatGPT, Perplexity, etc.) | Usuarios que piden usar skforecast a un LLM |
| **Custom GPTs / Gems / Projects** | Usuarios de ChatGPT/Gemini/Claude | Usuarios que buscan soluciones de forecasting |
| **Archivos de contexto en repo** | Devs con el repo clonado en un IDE | Contribuidores y devs que trabajan con el código |
| **MCP Server** | Usuarios con Python + IDE compatible | Power users y analistas |

---

## Perfiles de usuario objetivo

| Perfil | Herramienta habitual | Qué necesita | Solución |
|--------|---------------------|--------------|----------|
| Dev vibecoder | VS Code, Cursor, Claude Code | Que el LLM genere código correcto de skforecast | Archivos de contexto en el repo (auto-generados) |
| Analista | ChatGPT, Colab | Código end-to-end listo para ejecutar | Custom GPT + llms-full.txt como knowledge |
| No-coder | ChatGPT, Claude Desktop | Respuesta directa sin ver código | Custom GPT con Code Interpreter / MCP Server |

---

## Principio fundamental: fuente única de verdad

### El problema

Múltiples plataformas requieren el mismo conocimiento en formatos distintos:
- `.github/copilot-instructions.md` (VS Code Copilot)
- `AGENTS.md` (Claude Code, OpenAI Codex)
- `.claude/CLAUDE.md` (Claude Code)
- `.cursor/rules/skforecast.mdc` (Cursor IDE)
- `llms.txt` / `llms-full.txt` (web, cualquier LLM)
- Knowledge files del Custom GPT

Si se mantienen manualmente, **se desincronizarán inevitablemente** cuando la API cambie (nueva versión, imports nuevos, parámetros modificados).

### La solución: 2 archivos fuente + 1 script

Se mantienen manualmente **solo 3 archivos**:

| Archivo fuente | Contenido | Para quién |
|----------------|-----------|-----------|
| `llms.txt` | Resumen corto de skforecast (~730 líneas) con API reference, imports, ejemplos básicos | Cualquier LLM (web, IDEs, Custom GPTs) |
| `llms-full.txt` | Versión exhaustiva: todo lo de `llms.txt` + workflows end-to-end completos, guía de decisión de forecaster, errores comunes de LLMs, mejores prácticas | LLMs que necesitan contexto profundo |
| `tools/ai_context_header.md` | Sección pequeña específica para desarrollo del repo: cómo correr tests, code style, estructura de contribución | Solo devs trabajando dentro del repo |

Un script `tools/generate_ai_context_files.py` genera automáticamente:

```
Archivos fuente (mantenidos manualmente):
  llms.txt                        ← Resumen de API (~730 líneas)
  llms-full.txt                   ← Referencia completa + workflows
  tools/ai_context_header.md      ← Sección de desarrollo del repo

Script genera (NO editar manualmente, regenerar con el script):
  llms-full.txt                         → docs/llms-full.txt (web: skforecast.org/llms-full.txt)
  llms.txt                              → docs/llms.txt (web: skforecast.org/llms.txt)
  ai_context_header + llms-full.txt     → .github/copilot-instructions.md
  ai_context_header + llms-full.txt     → AGENTS.md
  ai_context_header + llms-full.txt     → .claude/CLAUDE.md
  ai_context_header + llms-full.txt     → .windsurfrules
  frontmatter + ai_context_header
    + llms-full.txt                     → .cursor/rules/skforecast.mdc
```

### Flujo de actualización (cada release)

```
1. Editar llms.txt y/o llms-full.txt con los cambios de API
2. Ejecutar: python tools/generate_ai_context_files.py
3. Commit de todos los archivos generados
4. Subir llms-full.txt actualizado al Custom GPT como knowledge file
```

### Contenido del script `tools/generate_ai_context_files.py`

```python
"""
Generate AI context files for all platforms from source files.

Source files (maintained manually):
    - llms.txt                    : Short API summary for any LLM
    - llms-full.txt               : Comprehensive reference + workflows
    - tools/ai_context_header.md  : Repo-dev-specific instructions (testing, style)

Generated files (DO NOT edit manually):
    - .github/copilot-instructions.md
    - AGENTS.md
    - .claude/CLAUDE.md
    - .windsurfrules
    - .cursor/rules/skforecast.mdc
    - docs/llms.txt
    - docs/llms-full.txt
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Source files
LLMS_TXT = ROOT / "llms.txt"
LLMS_FULL = ROOT / "llms-full.txt"
DEV_HEADER = ROOT / "tools" / "ai_context_header.md"

AUTOGEN_NOTICE = (
    "<!-- AUTO-GENERATED FILE. DO NOT EDIT MANUALLY. -->\n"
    "<!-- Source: llms-full.txt + tools/ai_context_header.md -->\n"
    "<!-- Regenerate with: python tools/generate_ai_context_files.py -->\n\n"
)

CURSOR_FRONTMATTER = (
    "---\n"
    "description: Rules for working with skforecast time series forecasting library\n"
    "globs: [\"**/*.py\", \"**/*.ipynb\"]\n"
    "---\n\n"
)

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  Generated: {path.relative_to(ROOT)}")

def main():
    llms_txt = read(LLMS_TXT)
    llms_full = read(LLMS_FULL)
    dev_header = read(DEV_HEADER) if DEV_HEADER.exists() else ""
    
    # Dev files = header + full content
    dev_content = AUTOGEN_NOTICE + dev_header + "\n\n" + llms_full

    print("Generating AI context files...")

    # 1. .github/copilot-instructions.md
    write(ROOT / ".github" / "copilot-instructions.md", dev_content)

    # 2. AGENTS.md (Claude Code, OpenAI Codex)
    write(ROOT / "AGENTS.md", dev_content)

    # 3. .claude/CLAUDE.md
    write(ROOT / ".claude" / "CLAUDE.md", dev_content)

    # 4. .windsurfrules (Windsurf / Codeium)
    write(ROOT / ".windsurfrules", dev_content)

    # 5. .cursor/rules/skforecast.mdc
    cursor_content = CURSOR_FRONTMATTER + dev_content
    write(ROOT / ".cursor" / "rules" / "skforecast.mdc", cursor_content)

    # 6. docs/llms.txt (for web: skforecast.org/llms.txt)
    write(ROOT / "docs" / "llms.txt", llms_txt)

    # 7. docs/llms-full.txt (for web: skforecast.org/llms-full.txt)
    write(ROOT / "docs" / "llms-full.txt", llms_full)

    print("\nDone! Remember to update the Custom GPT knowledge file manually.")

if __name__ == "__main__":
    main()
```

---

## Opciones evaluadas

| Opción | Pros | Contras | Veredicto |
|--------|------|---------|-----------|
| **Archivos de contexto en repo** (auto-generados) | Coste cero, máximo alcance para devs, se mantiene con la API, fuente única | Limitado a devs con IDE | ✅ HACER (Prioridad 1) |
| **`llms-full.txt` en skforecast.org** | Funciona para CUALQUIER LLM en cualquier plataforma, incluso web | Requiere que el LLM busque la URL o el usuario la pegue | ✅ HACER (Prioridad 1) |
| **Custom GPT (ChatGPT)** | Alcance masivo, funciona para no-coders, coste cero de desarrollo | Requiere ChatGPT Plus, datos van a OpenAI, límites sandbox | ✅ HACER (Prioridad 1) |
| **Google Gem + Claude Project** | Cubre otros ecosistemas | Menor alcance que ChatGPT | ✅ HACER (Prioridad 2) |
| **Estrategia de contenido para training data** | Impacto a largo plazo: los LLMs recomiendan skforecast sin contexto extra | Requiere esfuerzo continuo, resultados lentos | ✅ HACER (Prioridad 2) |
| **MCP Server con tools** | Datos locales, sin límites, multi-plataforma (todos los IDEs) | Desarrollo significativo, usuario necesita Python | ✅ HACER (Prioridad 2-3, empezar MVP) |
| **Agente custom con backend propio** | Control total de la experiencia | Caro, requiere infra, compite con tools existentes del usuario | ❌ NO HACER |

---

## Plan de acción por prioridad

### 🔴 Prioridad 1 — Fuente única + distribución inmediata (1-2 días)

| # | Tarea | Estado | Notas |
|---|-------|--------|-------|
| 1.0 | Crear `llms-full.txt` a partir de `llms.txt` + contenido de `copilot-instructions.md` | ⬜ Pendiente | Fusionar y enriquecer con workflows, guía de decisión, errores comunes |
| 1.1 | Crear `tools/ai_context_header.md` con info de desarrollo | ⬜ Pendiente | Testing, code style, contribución — solo para devs del repo |
| 1.2 | Crear script `tools/generate_ai_context_files.py` | ⬜ Pendiente | Genera todos los archivos de contexto desde las fuentes |
| 1.3 | Ejecutar script → generar `AGENTS.md`, `.claude/CLAUDE.md`, `.windsurfrules`, `.cursor/rules/skforecast.mdc`, actualizar `copilot-instructions.md` | ⬜ Pendiente | Verificar que cada archivo se lee correctamente en su plataforma |
| 1.4 | Copiar `llms.txt` y `llms-full.txt` a `docs/` para la web | ⬜ Pendiente | Verificar que MkDocs los sirve como estáticos en skforecast.org |
| 1.5 | Enriquecer `llms-full.txt` con workflows end-to-end | ⬜ Pendiente | Scripts completos para los 5 escenarios más comunes |
| 1.6 | Añadir guía de decisión de forecaster a `llms-full.txt` | ⬜ Pendiente | Tabla "qué forecaster usar según el caso" |
| 1.7 | Añadir sección de errores comunes a `llms-full.txt` | ⬜ Pendiente | Errores frecuentes que cometen los LLMs |
| 1.8 | Crear Custom GPT en ChatGPT (Code Interpreter) | ⬜ Pendiente | System prompt + `llms-full.txt` como knowledge |
| 1.9 | Probar Custom GPT con datasets reales | ⬜ Pendiente | Validar que genera código correcto y respuestas claras |

### 🟡 Prioridad 2 — Ampliar ecosistema + contenido (1-2 semanas)

| # | Tarea | Estado | Notas |
|---|-------|--------|-------|
| 2.1 | Crear Google Gem para Gemini | ⬜ Pendiente | Mismas instrucciones adaptadas |
| 2.2 | Crear Claude Project con instrucciones | ⬜ Pendiente | Para usuarios de Claude |
| 2.3 | Publicar Custom GPT en GPT Store | ⬜ Pendiente | Requiere pruebas previas (1.9) |
| 2.4 | Añadir links en README, docs y web | ⬜ Pendiente | Sección "AI Assistants" |
| 2.5 | Crear notebook de demos para el Custom GPT | ⬜ Pendiente | Casos de uso típicos documentados |
| 2.6 | Estrategia de contenido: responder en Stack Overflow con skforecast | ⬜ Continuo | Buscar preguntas de forecasting y responder usando skforecast |
| 2.7 | Publicar 2-3 notebooks en Kaggle con buen SEO | ⬜ Pendiente | Forecasting competiciones populares con skforecast |
| 2.8 | MCP Server MVP: implementar tools `load_and_analyze_data` + `forecast` | ⬜ Pendiente | Solo 2 tools como prueba de concepto |
| 2.9 | Añadir "actualizar archivos AI" al checklist de release | ⬜ Pendiente | En CONTRIBUTING.md o release process docs |
| 2.10 | Crear workflow `ai-context-drift-check.yml` | ⬜ Pendiente | Cron semanal: detecta cambios API, abre PR acumulativa |

### 🟢 Prioridad 3 — MCP Server completo (1-2 meses)

| # | Tarea | Estado | Notas |
|---|-------|--------|-------|
| 3.1 | Diseñar API de tools de alto nivel | ⬜ Pendiente | Ver sección detallada abajo |
| 3.2 | Implementar tool `compare_models` | ⬜ Pendiente | Comparar varios enfoques automáticamente |
| 3.3 | Implementar tool `explain_forecast` | ⬜ Pendiente | Tendencia, estacionalidad, importancia de features |
| 3.4 | Empaquetar como `skforecast-mcp` en PyPI | ⬜ Pendiente | Instalable con `pip install skforecast-mcp` |
| 3.5 | Documentar configuración para Claude Desktop / VS Code | ⬜ Pendiente | Guía step-by-step |
| 3.6 | Tests y validación con datasets variados | ⬜ Pendiente | |

---

## Detalle de cada iniciativa

### 1. Archivos de contexto: fuente única y generación automática

#### Arquitectura de archivos

```
skforecast/
├── llms.txt                              ← FUENTE 1: Resumen corto (mantener manualmente)
├── llms-full.txt                         ← FUENTE 2: Referencia completa (mantener manualmente)
├── tools/
│   ├── ai_context_header.md              ← FUENTE 3: Header de desarrollo (mantener manualmente)
│   └── generate_ai_context_files.py      ← Script que genera todo
│
│  ── ARCHIVOS GENERADOS (NO editar manualmente) ──
├── .github/copilot-instructions.md       ← Generado: header + llms-full.txt
├── AGENTS.md                             ← Generado: header + llms-full.txt
├── .claude/CLAUDE.md                     ← Generado: header + llms-full.txt
├── .windsurfrules                        ← Generado: header + llms-full.txt
├── .cursor/rules/skforecast.mdc          ← Generado: frontmatter + header + llms-full.txt
└── docs/
    ├── llms.txt                          ← Generado: copia para web
    └── llms-full.txt                     ← Generado: copia para web
```

#### ¿Quién lee cada archivo?

| Archivo | Plataforma | Cuándo se lee | Audiencia |
|---------|-----------|---------------|-----------|
| `skforecast.org/llms.txt` | Cualquier LLM con web search | Cuando busca info de skforecast | Todos |
| `skforecast.org/llms-full.txt` | Cualquier LLM cuando el usuario pega la URL | Bajo demanda | Todos |
| `.github/copilot-instructions.md` | VS Code / GitHub Copilot | Auto al abrir el repo | Devs con repo clonado |
| `AGENTS.md` | Claude Code, OpenAI Codex | Auto al abrir el repo | Devs con repo clonado |
| `.claude/CLAUDE.md` | Claude Code | Auto al abrir el repo | Devs con repo clonado |
| `.windsurfrules` | Windsurf / Codeium | Auto al abrir el repo | Devs con repo clonado |
| `.cursor/rules/skforecast.mdc` | Cursor IDE | Auto al abrir el repo | Devs con repo clonado |

**Nota importante**: Los archivos del repo solo sirven para devs que clonan el repositorio. Para el usuario típico que hace `pip install skforecast` y trabaja en su propio proyecto, **lo que importa es la versión web** (`skforecast.org/llms.txt`) y el **training data** del LLM (documentación, blogs, Stack Overflow).

#### Contenido de `tools/ai_context_header.md`

```markdown
# Skforecast — Development Context

## For Contributors Working Inside This Repository

### Testing
```bash
pytest skforecast/recursive/tests/ -vv
pytest --cov=skforecast --cov-report=html
```

### Code Style
- NumPy-style docstrings
- Type hints for function signatures
- PEP 8 compliant (max line length 88, enforced by ruff)
- Relative imports within package

### Dependencies
Core: numpy>=1.24, pandas>=1.5, scikit-learn>=1.2, scipy>=1.3.2, optuna>=2.10, joblib>=1.1, numba>=0.59, tqdm>=4.57, rich>=13.9
Optional: statsmodels>=0.12 (stats), matplotlib>=3.3 (plotting), keras>=3.0 (deep learning)

---

# Skforecast — Complete API & Workflow Reference

(The content below is the full `llms-full.txt` and applies to any user of skforecast)
```

#### Contenido a incluir en `llms-full.txt` (añadido respecto a `llms.txt`)

**A) Workflows end-to-end** (el usuario dice QUÉ quiere, no CÓMO):

Se deben incluir scripts completos y funcionales para estos escenarios:
- "Quiero predecir una serie temporal" → script completo con ForecasterRecursive
- "Tengo varias series y quiero predecirlas todas" → script con ForecasterRecursiveMultiSeries
- "Quiero encontrar los mejores hiperparámetros" → script con bayesian_search_forecaster
- "Quiero usar ARIMA" → script con ForecasterStats + Arima
- "Quiero predecir con intervalos de confianza" → script con predict_interval + backtesting

Cada workflow debe incluir: imports, carga de datos, split, entrenamiento, predicción, evaluación y visualización.

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
2. Show a brief summary of what you're doing (e.g., "Training a LightGBM model with 
   24 lags on your monthly data..."), then show the results.
3. Answer in PLAIN LANGUAGE with the prediction, confidence interval, and a chart.
4. Include code in a collapsed/brief block so interested users can see and adapt it.
5. If the user uploads a file, auto-detect the date column, frequency, and target.
6. If accuracy is poor, say so honestly and suggest what data would help.
7. Always validate with backtesting before giving a final prediction.

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
- `llms-full.txt` (referencia completa con workflows)

#### Publicación
- Nombre: **"Skforecast - Time Series Forecasting"**
- Descripción: "Upload your data (CSV/Excel) and ask any forecasting question. Get predictions, confidence intervals, and visualizations — no coding required."
- Logo: Logo de skforecast
- Publicar en GPT Store como público

---

### 3. Google Gem + Claude Project

Mismo contenido que el Custom GPT, adaptado a cada plataforma:

- **Google Gem**: Pegamos las instrucciones en la configuración del Gem. No tiene Code Interpreter equivalente, así que se limita a generar código para que el usuario lo ejecute en Colab.
- **Claude Project**: Subimos `llms-full.txt` y el prompt como instrucciones del proyecto. Claude no ejecuta código, pero genera código muy preciso si tiene el contexto.

---

### 4. Estrategia de contenido para training data

**Por qué es importante**: Los archivos de contexto solo sirven cuando están en el workspace del IDE o cuando el usuario pega la URL. Lo que REALMENTE hace que un LLM "sepa" skforecast **sin ningún contexto extra** es el training data: contenido público en la web que el LLM vio durante su entrenamiento.

| Canal | Acción | Impacto |
|-------|--------|---------|
| **Stack Overflow** | Buscar preguntas de time series forecasting y responder usando skforecast | Alto — peso enorme en training data de LLMs |
| **Kaggle notebooks** | Publicar 2-3 notebooks en competiciones populares de forecasting | Alto — muy indexado y visible |
| **Blog posts** | Publicar tutoriales en Medium / Towards Data Science | Medio — buen SEO |
| **Documentation web** | Ya existe y es completa | Ya cubierto |
| **GitHub Discussions** | Responder preguntas de usuarios con ejemplos completos | Medio |

**Regla**: Cada pieza de contenido público debe usar los **imports actuales** correctos. Un solo blog post con `from skforecast.ForecasterAutoreg import ForecasterAutoreg` que un LLM memorice causa miles de errores a usuarios.

---

### 5. MCP Server (`skforecast-mcp`)

#### Filosofía de diseño
- Tools de **ALTO NIVEL**: el LLM no programa skforecast, lo usa como herramienta
- Cada tool es una operación completa (no exponer .fit()/.predict() por separado)
- El LLM decide cuándo usar cada tool basándose en la pregunta del usuario
- Las imágenes se retornan como base64 en el resultado (no como rutas a archivos)

#### Tools a implementar

**MVP (Prioridad 2):**

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
        plot_base64: str  # Chart as base64-encoded PNG
    }
    """
```

**Completo (Prioridad 3):**

```python
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
│           └── plotting.py     # Generate charts as base64
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

### 6. Workflow de detección de cambios API (`ai-context-drift-check`)

#### Problema

Entre releases, se van acumulando cambios en la rama de desarrollo (ej. `0.21.x`) que afectan a la API pública: nuevos parámetros, exports, deprecaciones. Es fácil olvidar reflejarlos en `llms.txt` / `llms-full.txt` cuando llega el momento del release.

#### Solución: GitHub Action con cron semanal + PR acumulativa

Un workflow que se ejecuta cada lunes, analiza el diff de la rama de desarrollo contra el último release tag, y mantiene **una sola PR abierta** que se va engordando con los cambios detectados cada semana.

```
Semana 1: workflow detecta cambios → crea PR con tools/ai_changes_pending.md
Semana 2: workflow detecta más cambios → actualiza la misma PR (append)
Semana 3: sin cambios API → no toca la PR
Release: tú mergeas la PR → usas la lista para actualizar llms.txt/llms-full.txt
```

#### Archivo controlado: `tools/ai_changes_pending.md`

```markdown
# Pending API changes for AI context files

> This file is auto-updated weekly by the `ai-context-drift-check` workflow.
> After updating `llms.txt` / `llms-full.txt`, clear the sections below and
> keep only this header.

---

## Detected 2026-03-04 (0.21.x, commits abc123..def456)

### Modified signatures
- `bayesian_search_forecaster()`: new param `suppress_warnings: bool = False`
- `backtesting_forecaster()`: new param `use_binned_residuals: bool = True`

### New exports
- `skforecast.preprocessing.ConformalIntervalCalibrator`

### Dependency changes
- `optuna>=2.10` → `optuna>=3.0`

---

## Detected 2026-03-11 (0.21.x, commits def456..789abc)

### Modified signatures
- `RollingFeatures.__init__()`: param `window_sizes` renamed to `windows`

### Deprecated
- `check_exog` → use `validate_exog`

---
```

#### Qué detecta el workflow (sin LLM, puro `git diff` + `grep`)

| Categoría | Cómo lo detecta |
|-----------|----------------|
| Exports nuevos/eliminados | `git diff` en `*/__init__.py` — líneas `+`/`-` con imports |
| Firmas modificadas | `git diff` en `*.py` — líneas con `def ` que cambian |
| Archivos nuevos en módulos públicos | `git diff --name-status` — archivos con status `A` |
| Cambios en dependencias | `git diff` en `pyproject.toml` |
| Deprecaciones | `git diff` en `*.py` — líneas con `deprecated` (case-insensitive) |

Excluye automáticamente: `tests/`, `docs/`, `dev/`, `tools/`, `benchmarks/`.

#### Workflow: `.github/workflows/ai-context-drift-check.yml`

```yaml
name: AI Context Drift Check

on:
  schedule:
    - cron: '0 8 * * 1'  # Every Monday at 8:00 UTC
  workflow_dispatch:       # Manual trigger

permissions:
  contents: write
  pull-requests: write

jobs:
  check-drift:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for tags

      - name: Detect API changes
        id: detect
        run: |
          # Find latest release tag
          LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
          if [ -z "$LAST_TAG" ]; then
            echo "No tags found, skipping."
            echo "has_changes=false" >> $GITHUB_OUTPUT
            exit 0
          fi

          DEV_BRANCH="${{ github.ref_name }}"
          RANGE="${LAST_TAG}..${DEV_BRANCH}"
          TODAY=$(date +%Y-%m-%d)
          COMMITS_SHORT=$(git log --oneline $RANGE -- '*.py' 'pyproject.toml' | head -5)
          COMMIT_START=$(echo $LAST_TAG | cut -c1-7)
          COMMIT_END=$(git rev-parse --short HEAD)

          # Collect changes
          CHANGES=""

          # 1. Modified signatures (public functions)
          SIGS=$(git diff $RANGE -- 'skforecast/**/*.py' \
            ':!skforecast/**/tests/**' \
            | grep -E '^\+.*def [a-z_]+\(' \
            | grep -v '__' \
            | sed 's/^+//' | sed 's/^[[:space:]]*/- /' || true)
          if [ -n "$SIGS" ]; then
            CHANGES="${CHANGES}\n### Modified signatures\n${SIGS}\n"
          fi

          # 2. New exports
          EXPORTS=$(git diff $RANGE -- 'skforecast/**/__init__.py' \
            | grep '^+' | grep -v '^\+\+\+' \
            | grep -E '(import|from)' \
            | sed 's/^+//' | sed 's/^[[:space:]]*/- /' || true)
          if [ -n "$EXPORTS" ]; then
            CHANGES="${CHANGES}\n### New/modified exports\n${EXPORTS}\n"
          fi

          # 3. New files
          NEW_FILES=$(git diff --name-status $RANGE -- 'skforecast/**/*.py' \
            ':!skforecast/**/tests/**' \
            | grep '^A' | awk '{print "- " $2}' || true)
          if [ -n "$NEW_FILES" ]; then
            CHANGES="${CHANGES}\n### New files\n${NEW_FILES}\n"
          fi

          # 4. Dependency changes
          DEPS=$(git diff $RANGE -- 'pyproject.toml' \
            | grep -E '^\+.*>=' \
            | sed 's/^+//' | sed 's/^[[:space:]]*/- /' || true)
          if [ -n "$DEPS" ]; then
            CHANGES="${CHANGES}\n### Dependency changes\n${DEPS}\n"
          fi

          # 5. Deprecations
          DEPRECATED=$(git diff $RANGE -- 'skforecast/**/*.py' \
            ':!skforecast/**/tests/**' \
            | grep -i 'deprecated' | grep '^\+' \
            | sed 's/^+//' | sed 's/^[[:space:]]*/- /' | head -10 || true)
          if [ -n "$DEPRECATED" ]; then
            CHANGES="${CHANGES}\n### Deprecated\n${DEPRECATED}\n"
          fi

          if [ -z "$CHANGES" ]; then
            echo "No API changes detected."
            echo "has_changes=false" >> $GITHUB_OUTPUT
            exit 0
          fi

          # Build new section
          SECTION="## Detected ${TODAY} (${DEV_BRANCH}, commits ${COMMIT_START}..${COMMIT_END})\n${CHANGES}\n---\n"
          echo "$SECTION" > /tmp/new_changes.txt
          echo "has_changes=true" >> $GITHUB_OUTPUT

      - name: Update pending changes file
        if: steps.detect.outputs.has_changes == 'true'
        run: |
          FILE="tools/ai_changes_pending.md"
          if [ ! -f "$FILE" ]; then
            cat > "$FILE" << 'EOF'
          # Pending API changes for AI context files

          > This file is auto-updated weekly by the `ai-context-drift-check` workflow.
          > After updating `llms.txt` / `llms-full.txt`, clear the sections below and
          > keep only this header.

          ---

          EOF
          fi
          # Append new section
          cat /tmp/new_changes.txt >> "$FILE"

      - name: Create or update PR
        if: steps.detect.outputs.has_changes == 'true'
        uses: peter-evans/create-pull-request@v6
        with:
          branch: ai-context-drift-check
          title: "🔄 AI context files — API changes detected"
          body: |
            Automated weekly check detected API changes that may need to be
            reflected in `llms.txt` / `llms-full.txt`.

            Review `tools/ai_changes_pending.md` for details.

            **When ready:**
            1. Merge this PR
            2. Update `llms.txt` / `llms-full.txt` using the list (VS Code + Copilot)
            3. Run `python tools/generate_ai_context_files.py`
            4. Clear `tools/ai_changes_pending.md` (keep only header)
          labels: documentation, ai-context
          commit-message: "docs: update ai_changes_pending.md with detected API changes"
```

#### Flujo de trabajo completo

```
┌─────────────────────────────────────────────────┐
│  GitHub Actions (cron lunes, gratis repos pub.)  │
│  Detecta diff API → actualiza PR acumulativa     │
└──────────────────────┬──────────────────────────┘
                       │ notificación (cuando quieras)
                       ▼
┌─────────────────────────────────────────────────┐
│  Mergeas la PR → ahora ai_changes_pending.md    │
│  tiene la lista de cambios en tu rama local      │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  VS Code + Copilot Pro (empresa)                 │
│  "Mira tools/ai_changes_pending.md y actualiza   │
│   llms-full.txt con estos cambios"               │
│  → Copilot aplica los cambios, tú revisas        │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  python tools/generate_ai_context_files.py       │
│  Limpiar ai_changes_pending.md → commit todo     │
└─────────────────────────────────────────────────┘
```

**Coste total**: 0€ (GitHub Actions es gratis para repos públicos, `peter-evans/create-pull-request` es una action gratuita, no requiere LLM).

#### Opciones descartadas para este workflow

| Opción | Por qué se descarta |
|--------|-------------------|
| `.github/instructions/*.md` (Copilot instrucciones por carpeta) | Se inyectan **además de** `copilot-instructions.md` — con el mismo contenido duplicaría tokens. Solo útil en monorepos con zonas muy distintas (React + Python + Terraform). skforecast es temáticamente uniforme |
| `.github/agents/*.md` (Copilot custom agents `@nombre`) | No se auto-inyectan — el usuario debe invocar `@agent-name`. Solo funcionan en Copilot Chat. Muy nicho, casi ningún proyecto OSS los usa todavía. Considerar en el futuro si emerge un caso de uso claro (ej. `@skforecast-reviewer`) |
| Que el workflow edite directamente `llms-full.txt` con un LLM | Riesgo de que el LLM malinterprete un cambio, invente sintaxis o meta ruido. La fuente de verdad no debe editarse automáticamente |
| Abrir Issues en vez de PR | Una PR con archivo controlado queda en el repo, es visible en el diff, y es acumulativa. Un Issue es fácil de ignorar y se pierde entre otros |

---

## Comunicación y distribución

### En el README del repo

Añadir una sección visible:

```markdown
## 🤖 AI-Assisted Forecasting

**No coding experience?** Use our AI assistants:
- **ChatGPT**: [Skforecast GPT](link) — Upload your CSV and ask your question
- **Any LLM**: Pass [https://skforecast.org/llms-full.txt](https://skforecast.org/llms-full.txt) as context

**Developer using AI coding tools?** Skforecast works out of the box with:
- **VS Code Copilot**: Instructions auto-loaded from `.github/copilot-instructions.md`
- **Cursor**: Instructions auto-loaded from `.cursor/rules/`
- **Claude Code**: Instructions auto-loaded from `AGENTS.md` and `.claude/CLAUDE.md`
- **OpenAI Codex**: Instructions auto-loaded from `AGENTS.md`
```

### En la documentación web (skforecast.org)

Crear una página "AI Assistants" en la sección de user guides con:
- Links a todos los agentes
- Instrucciones para usar skforecast con cada herramienta AI
- Ejemplos de prompts que funcionan bien
- Link a `llms-full.txt` con instrucción: "Pega esta URL en tu LLM para que sepa usar skforecast"

### En redes sociales / blog

Post de lanzamiento: *"Ahora puedes hacer forecasting sin escribir código. Sube tu CSV a nuestro GPT y pregúntale cuánto vas a vender mañana."*

---

## Qué NO hacer

| Idea | Por qué no |
|------|-----------|
| Crear un agente custom con backend/servidor propio | Coste de infra enorme para un proyecto OSS, el usuario tiene que confiar en un tercero con su API key |
| Exponer la API de bajo nivel de skforecast como tools del MCP | Demasiado complejo para que un LLM lo use correctamente en nombre de un no-coder |
| Construir una web app completa para no-coders | Eso es un producto entero (ya existe Skforecast Studio) |
| Editar manualmente los archivos generados | Se sobreescribirán la próxima vez que se ejecute el script. Editar SOLO las fuentes |
| Crear instrucciones solo con referencia de API | Los LLMs necesitan workflows completos end-to-end, no solo documentación |
| Ignorar el training data | Los archivos de contexto solo sirven en IDEs. Lo que hace que un LLM conozca skforecast a nivel global es el contenido público (docs, SO, Kaggle, blogs) |

---

## Cronograma estimado

| Semana | Tareas |
|--------|--------|
| **Semana 1** | 1.0-1.4: Crear llms-full.txt, header, script, generar archivos, publicar en web |
| **Semana 1** | 1.5-1.7: Enriquecer llms-full.txt con workflows, guía de decisión, errores comunes |
| **Semana 1** | 1.8-1.9: Crear y probar Custom GPT |
| **Semana 2** | 2.1-2.5: Google Gem, Claude Project, publicar GPT, actualizar docs |
| **Semana 2** | 2.8: MCP Server MVP (2 tools) |
| **Semana 2** | 2.10: Crear workflow `ai-context-drift-check.yml` |
| **Semana 3-4** | 2.6-2.7: Stack Overflow, Kaggle notebooks |
| **Semana 3-4** | Recoger feedback de usuarios y iterar sobre instrucciones y GPT |
| **Mes 2-3** | 3.1-3.6: MCP Server completo, publicar en PyPI |

---

## Proceso de mantenimiento (cada release)

| Paso | Acción |
|------|--------|
| 1 | Mergear la PR acumulativa de `ai-context-drift-check` (si hay cambios pendientes) |
| 2 | Revisar `tools/ai_changes_pending.md` — usar como checklist de qué actualizar |
| 3 | Actualizar `llms.txt` y `llms-full.txt` con los cambios (VS Code + Copilot Pro) |
| 4 | Limpiar `tools/ai_changes_pending.md` (dejar solo el header) |
| 5 | Ejecutar `python tools/generate_ai_context_files.py` |
| 6 | Commit de todos los archivos generados junto con el release |
| 7 | Subir `llms-full.txt` actualizado al Custom GPT como knowledge file |
| 8 | Verificar que `skforecast.org/llms.txt` y `llms-full.txt` están actualizados |

---

## Métricas de éxito

- Número de usos del Custom GPT (GPT Store analytics)
- Issues / PRs que mencionen AI-assisted usage
- Descargas de `skforecast-mcp` en PyPI (cuando se lance)
- Feedback cualitativo de usuarios en GitHub Discussions / Twitter
- Reducción de issues causados por imports deprecados (señal de que los LLMs generan código correcto)
- Posición de skforecast en respuestas de LLMs cuando se pregunta "best Python library for time series forecasting"

---

## Consideración pendiente: tamaño de `llms-full.txt` en archivos de IDE

Los archivos de IDE (`.github/copilot-instructions.md`, `AGENTS.md`, `.claude/CLAUDE.md`, `.windsurfrules`, `.cursor/rules/skforecast.mdc`) se **auto-inyectan en cada prompt**. Si `llms-full.txt` crece demasiado (>2000 líneas), puede causar:

- **Gasto innecesario de tokens** en cada interacción
- **Truncamiento** en plataformas con límites de contexto más estrictos
- **Dilución** del contenido importante entre demasiado texto

**Evaluar cuando `llms.txt` y `llms-full.txt` tengan su contenido definitivo.** Si `llms-full.txt` queda en un tamaño razonable (<2000 líneas), usar directamente en los archivos de IDE como está planificado. Si crece mucho, cambiar los archivos de IDE para que usen `llms.txt` (corto) + dev header, y reservar `llms-full.txt` solo para:
- Web (`skforecast.org/llms-full.txt`)
- Knowledge file del Custom GPT
- Claude Project / Google Gem

El script se adaptaría fácilmente: cambiar la variable fuente de `llms_full` a `llms_txt` para los archivos de IDE.

```python
# Si llms-full.txt es demasiado grande, cambiar esto en el script:
dev_content = AUTOGEN_NOTICE + dev_header + "\n\n" + llms_full   # ← actual
dev_content = AUTOGEN_NOTICE + dev_header + "\n\n" + llms_txt    # ← alternativa si es muy grande
```
