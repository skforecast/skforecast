# Skforecast AI Agent & Vibe-Coding Strategy

> Plan para maximizar la adopción de skforecast a través de asistentes AI, cubriendo desde desarrolladores que vibecodean hasta usuarios sin conocimientos de programación.

**Fecha de creación**: 2026-02-27  
**Estado**: Draft  
**Última revisión**: 2026-03-04

---

## Tabla de contenidos

1. [Resumen ejecutivo](#resumen-ejecutivo)
2. [Perfiles de usuario objetivo](#perfiles-de-usuario-objetivo)
3. [Principio fundamental: fuente única de verdad](#principio-fundamental-fuente-única-de-verdad)
4. [Opciones evaluadas](#opciones-evaluadas)
5. [Plan de acción por prioridad](#plan-de-acción-por-prioridad)
6. [Detalle de cada iniciativa](#detalle-de-cada-iniciativa)
   - [1. Archivos de contexto](#1-archivos-de-contexto-fuente-única-y-generación-automática)
   - [2. Custom GPT](#2-custom-gpt-chatgpt-con-code-interpreter)
   - [3. Google Gem + Claude Project](#3-google-gem--claude-project)
   - [4. Estrategia de contenido](#4-estrategia-de-contenido-para-training-data)
   - [5. MCP Server](#5-mcp-server-skforecast-mcp)
   - [6. Workflow de detección de cambios API](#6-workflow-de-detección-de-cambios-api-ai-context-drift-check)
7. [Comunicación y distribución](#comunicación-y-distribución)
8. [Qué NO hacer](#qué-no-hacer)
9. [Cronograma estimado](#cronograma-estimado)
10. [Proceso de mantenimiento](#proceso-de-mantenimiento-cada-release)
11. [Métricas de éxito](#métricas-de-éxito)
12. [Validación: cómo probar que funciona](#validación-cómo-probar-que-funciona)

---

## Resumen ejecutivo

Hay tres perfiles de usuario que queremos cubrir:
1. **Desarrolladores que vibecodean**: Ya usan un IDE + LLM, necesitan que el LLM sepa usar skforecast correctamente.
2. **Analistas con algo de código**: Quieren copiar/pegar código funcional generado por AI.
3. **No-coders**: Quieren respuestas ("¿cuánto voy a vender mañana?") sin ver código.

La estrategia es **mantener el conocimiento de skforecast en un archivo fuente** (`llms.txt`) **y una carpeta modular `skills/`** con workflows independientes siguiendo el [Agent Skills spec](https://agentskills.io). Un script **genera automáticamente** `llms-full.txt` (concatenando `llms.txt` + skills) y todos los archivos de contexto para cada plataforma. Esto garantiza consistencia, elimina duplicación (~60% del contenido anterior de `llms-full.txt` era idéntico a `llms.txt`), y habilita *progressive disclosure* (los agentes cargan solo los skills que necesitan).

### Canales de impacto y audiencia

| Canal | Alcance | Quién lo ve | Cuándo se lee |
|-------|---------|-------------|---------------|
| **Training data** (docs web, Stack Overflow, Kaggle, blogs) | Todo usuario de cualquier LLM | Usuarios que nunca han oído de skforecast | Entrenamiento del modelo |
| **`skforecast.org/llms.txt`** | LLMs con web search (ChatGPT, Perplexity, etc.) | Usuarios que piden usar skforecast a un LLM | Cuando busca info de skforecast |
| **`skforecast.org/llms-full.txt`** | Cualquier LLM cuando el usuario pega la URL | Todos | Bajo demanda |
| **Custom GPTs / Gems / Projects** | Usuarios de ChatGPT/Gemini/Claude | Usuarios que buscan soluciones de forecasting | Al iniciar conversación |
| **Archivos de contexto en repo** (`.github/copilot-instructions.md`, `AGENTS.md`, etc.) | Devs con el repo clonado en un IDE | Contribuidores y devs que trabajan con el código | Auto al abrir el repo |
| **MCP Server** | Usuarios con Python + IDE compatible | Power users y analistas | Al invocar tools |

---

## Perfiles de usuario objetivo

| Perfil | Herramienta habitual | Qué necesita | Solución |
|--------|---------------------|--------------|----------|
| Dev vibecoder | VS Code, Cursor, Claude Code | Que el LLM genere código correcto de skforecast | Archivos de contexto en el repo (auto-generados) |
| Analista | ChatGPT, Colab | Workflows completos copy-paste listos para ejecutar | Custom GPT + llms-full.txt como knowledge |
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

### La solución: `llms.txt` + `skills/` + 1 script

Se mantienen manualmente:

| Fuente | Contenido | Formato | Para quién |
|--------|-----------|---------|-----------|
| `llms.txt` | Core reference: estructura del proyecto, imports, API overview, ejemplos básicos | Markdown plano (~730 líneas) | IDEs (auto-inyectado), web, cualquier LLM |
| `skills/` | Carpeta con N skills modulares: workflows, guía de decisión, errores comunes, API detallada | [Agent Skills spec](https://agentskills.io) — directorio con `SKILL.md` | Agentes AI avanzados, web (via `llms-full.txt`) |
| `tools/ai_context_header.md` | Header de desarrollo: tests, code style, contribución | Markdown plano (~30 líneas) | Solo devs del repo |

**`llms-full.txt` ya NO se mantiene manualmente** — se genera automáticamente concatenando `llms.txt` + todos los `SKILL.md` (sin frontmatter). Esto elimina la duplicación que existía (~60% del contenido era idéntico).

**Decisión de diseño — por qué `skills/` en vez de un monolito**:

1. **Eliminación de duplicación**: Antes, `llms-full.txt` repetía ~430 líneas idénticas de `llms.txt`. Con skills, cada pieza de contenido existe en un solo lugar.
2. **Progressive disclosure**: Un agente AI carga `llms.txt` (~730 líneas) por defecto; solo activa skills específicos cuando necesita un workflow concreto. Esto reduce el consumo de tokens a lo estrictamente necesario.
3. **Mantenimiento modular**: Editar un workflow de "prediction intervals" no requiere navegar un archivo de >2000 líneas. Cada skill es independiente y auto-contenido.
4. **Distribución universal**: Gracias a [Vercel Skills CLI](https://github.com/vercel-labs/skills) (`npx skills add`), los skills se instalan automáticamente en 40+ agentes (Claude Code, Cursor, Copilot, Codex, Windsurf, Gemini CLI, etc.).
5. **Estándar abierto**: El [Agent Skills spec](https://agentskills.io/specification) es un estándar emergente con adopción creciente — no atamos el contenido a un proveedor.

**Decisión de diseño — IDE context**: Los archivos de IDE usan `llms.txt` (corto), **no** `llms-full.txt`. `llms.txt` (~730 líneas) ya contiene la referencia de API, imports y ejemplos — suficiente para un dev con IDE que tiene el código delante. Inyectar `llms-full.txt` (>2000 líneas) en cada prompt causaría gasto innecesario de tokens, truncamiento, y dilución del contenido importante.

**Versionado**: Incluir una línea prominente al inicio de `llms.txt`: `"This document is for skforecast v0.21.0+. If you are using an older version, check the documentation at skforecast.org."`. Mantener versiones archivadas probablemente no compensa el esfuerzo.

Un script `tools/generate_ai_context_files.py` genera automáticamente `llms-full.txt` y todos los archivos derivados. Ver [sección 1: Arquitectura de archivos](#arquitectura-de-archivos) para el mapeo completo.

### Flujo de actualización (cada release)

```
1. Editar llms.txt y/o skills/ con los cambios de API
2. Ejecutar: python tools/generate_ai_context_files.py
   → Genera llms-full.txt (llms.txt + skills sin frontmatter)
   → Genera archivos de IDE (header + llms.txt)
   → Copia llms.txt y llms-full.txt a docs/
3. Commit de todos los archivos generados (incluido llms-full.txt)
4. Subir llms-full.txt actualizado al Custom GPT como knowledge file
```

### Script `tools/generate_ai_context_files.py` — diseño

El script debe implementar esta lógica:

```python
# 1. Ensamblar llms-full.txt = llms.txt + todos los SKILL.md + references/ (sin frontmatter YAML)
#    Orden explícito: workflows primero (mayor uso), referencia técnica al final.
#    Los LLMs dan más peso al contenido que aparece primero.
SKILL_ORDER = [
    "forecasting-single-series",
    "forecasting-multiple-series",
    "statistical-models",
    "hyperparameter-optimization",
    "prediction-intervals",
    "feature-engineering",
    "feature-selection",
    "drift-detection",
    "deep-learning-forecasting",
    "choosing-a-forecaster",
    "common-errors",
    "complete-api-reference",  # Al final: contenido denso de referencia
]

skills_content = ""
for skill_name in SKILL_ORDER:
    skill_dir = Path("skills") / skill_name
    skill_md = skill_dir / "SKILL.md"
    if skill_md.exists():
        raw = skill_md.read_text()
        body = strip_yaml_frontmatter(raw)
        skills_content += f"\n\n{'=' * 80}\n"
        skills_content += f"# SKILL: {skill_name}\n"
        skills_content += f"{'=' * 80}\n\n"
        skills_content += body.strip()
        # Incluir archivos de references/ si existen
        refs_dir = skill_dir / "references"
        if refs_dir.exists():
            for ref_file in sorted(refs_dir.glob("*.md")):
                ref_body = ref_file.read_text().strip()
                skills_content += f"\n\n---\n### Reference: {ref_file.stem}\n\n"
                skills_content += ref_body

llms_full = llms_txt + "\n\n" + skills_content

# 2. Archivos de IDE: header + llms.txt (corto)
dev_content = AUTOGEN_NOTICE_IDE + dev_header + "\n\n" + llms_txt

# 3. Copiar llms.txt y llms-full.txt a docs/ para web
```

Donde `AUTOGEN_NOTICE` tiene dos variantes según el archivo destino:

Para archivos IDE (header + llms.txt):
```
<!-- AUTO-GENERATED FILE. DO NOT EDIT MANUALLY. -->
<!-- Source: llms.txt + tools/ai_context_header.md -->
<!-- Regenerate with: python tools/generate_ai_context_files.py -->
```

Para `llms-full.txt` (llms.txt + skills):
```
<!-- AUTO-GENERATED FILE. DO NOT EDIT MANUALLY. -->
<!-- Source: llms.txt + skills/ -->
<!-- Regenerate with: python tools/generate_ai_context_files.py -->
```

**Funcionalidad requerida del script:**

| Función | Descripción |
|---------|-------------|
| Ensamblaje `llms-full.txt` | Concatena `llms.txt` + todos los `skills/*/SKILL.md` (sin frontmatter YAML) + `references/*.md` si existen, en orden explícito definido en `SKILL_ORDER` |
| Generación IDE | Lee header + `llms.txt`, genera los 7 archivos de IDE (ver arquitectura) |
| Cursor frontmatter | Añade `---\ndescription: ...\nglobs: ...\n---\n` al archivo de Cursor |
| Validación de skills | Verificar que cada `SKILL.md` tiene frontmatter YAML válido (`name`, `description` requeridos), < 500 líneas, y está listado en `SKILL_ORDER` |
| Validación de fuentes | Verificar que `llms.txt` contiene los imports de los `__init__.py` públicos, y que la versión coincide con `skforecast/__init__.py` |
| Modo `--check` | Sin escribir archivos, falla si los archivos generados están desactualizados (para CI) |

---

## Opciones evaluadas

| Opción | Pros | Contras | Veredicto |
|--------|------|---------|-----------|
| **Archivos de contexto en repo** (auto-generados) | Coste cero, máximo alcance para devs, se mantiene con la API, fuente única | Limitado a devs con IDE | ✅ HACER (Prioridad 1) |
| **`llms-full.txt` en skforecast.org** | Funciona para CUALQUIER LLM en cualquier plataforma, incluso web | Requiere que el LLM busque la URL o el usuario la pegue | ✅ HACER (Prioridad 1) |
| **Custom GPT (ChatGPT)** | Alcance masivo, funciona para no-coders, coste cero de desarrollo | Requiere ChatGPT Plus, datos van a OpenAI, límites sandbox | ✅ HACER (Prioridad 1) |
| **Google Gem + Claude Project** | Cubre otros ecosistemas | Menor alcance que ChatGPT | ✅ HACER (Prioridad 2) |
| **Estrategia de contenido para training data** | Impacto a largo plazo: los LLMs recomiendan skforecast sin contexto extra | Requiere esfuerzo continuo, resultados lentos | ✅ HACER (Prioridad 2) |
| **MCP Server con tools** | Datos locales, sin límites, multi-plataforma (todos los IDEs) | Desarrollo significativo, ecosistema joven, adopción baja fuera de Claude Desktop | ✅ HACER (Prioridad 3) |
| **Agente custom con backend propio** | Control total de la experiencia | Caro, requiere infra, compite con tools existentes del usuario | ❌ NO HACER |

---

## Plan de acción por prioridad

**Convenciones**: 🤖 = ejecutable por agente AI, 👤 = requiere acción manual humana, 🤖/👤 = el agente prepara, el humano valida. Las tareas están agrupadas en **fases con dependencias explícitas**. Dentro de cada fase, las tareas son independientes entre sí y el agente puede ejecutarlas en cualquier orden.

### 🔴 Prioridad 1 — Fuente única + distribución inmediata (2-3 semanas)

#### Fase A — Investigación (sin escritura de archivos)

| # | Tarea | Ejecutor | Estado | Notas |
|---|-------|----------|--------|-------|
| 1.1 | Auditar `.github/copilot-instructions.md` actual | 🤖 | ⬜ | Comparar las 865 líneas actuales vs. header + llms.txt; documentar contenido que se pierde o cambia. **Hacerlo primero**: el resultado informa qué contenido falta en `llms.txt` o debe ir en skills |

#### Fase B — Infraestructura y config (dependencia: Fase A)

| # | Tarea | Ejecutor | Estado | Notas |
|---|-------|----------|--------|-------|
| 1.2 | Crear `tools/ai_context_header.md` con info de desarrollo | 🤖 | ⬜ | Testing, code style, contribución — solo para devs del repo |
| 1.3 | Crear `.gitattributes` para archivos generados | 🤖 | ⬜ | `linguist-generated=true` para colapsar diffs en GitHub |
| 1.4 | Añadir URL de `llms.txt` al `pyproject.toml` | 🤖 | ⬜ | `[project.urls] "LLM Context" = "https://skforecast.org/llms.txt"` |
| 1.5 | Crear script `tools/generate_ai_context_files.py` | 🤖 | ⬜ | Ensambla llms-full.txt (llms.txt + skills + references), genera archivos IDE, valida skills + imports, modo `--check`. Se escribe ahora pero se **ejecuta en Fase D** (necesita skills) |

#### Fase C — Contenido de skills (dependencia: Fase A; paralelizable con Fase B)

El agente crea cada directorio `skills/{name}/` y su `SKILL.md` en un solo paso (no crear directorios vacíos primero).

| # | Tarea | Ejecutor | Estado | Notas |
|---|-------|----------|--------|-------|
| 1.6 | Crear 5 skills de workflows end-to-end | 🤖 | ⬜ | `forecasting-single-series`, `forecasting-multiple-series`, `hyperparameter-optimization`, `statistical-models`, `prediction-intervals` |
| 1.7 | Crear skill `choosing-a-forecaster` | 🤖 | ⬜ | Guía de decisión: "qué forecaster usar según el caso" |
| 1.8 | Crear skill `common-errors` | 🤖 | ⬜ | Errores frecuentes de LLMs + soluciones |
| 1.9 | Crear 4 skills técnicos | 🤖 | ⬜ | `feature-engineering`, `feature-selection`, `drift-detection`, `deep-learning-forecasting` |
| 1.10 | Crear skill `complete-api-reference` con `references/` | 🤖 | ⬜ | Índice en SKILL.md + firmas detalladas en `references/method-signatures.md` |

#### Fase D — Ensamblaje y validación (dependencia: Fases B + C completadas)

| # | Tarea | Ejecutor | Estado | Notas |
|---|-------|----------|--------|-------|
| 1.11 | Ejecutar script → generar todos los archivos (IDE + docs + llms-full.txt) | 🤖 | ⬜ | Verificar que cada archivo generado es correcto. El script ya copia a `docs/` |
| 1.12 | Validar cada skill en al menos 1 agente | 🤖/👤 | ⬜ | Probar cada skill con un prompt relevante en Copilot o Claude Code |
| 1.13 | Verificar que MkDocs sirve `llms.txt` y `llms-full.txt` | 👤 | ⬜ | Verificar URLs en skforecast.org después del deploy |

#### Fase E — Distribución (dependencia: Fase D; requiere acción manual)

| # | Tarea | Ejecutor | Estado | Notas |
|---|-------|----------|--------|-------|
| 1.14 | Crear Custom GPT en ChatGPT (Code Interpreter) | 👤 | ⬜ | System prompt + `llms-full.txt` generado como knowledge |
| 1.15 | Probar Custom GPT con datasets reales | 👤 | ⬜ | Validar que genera código correcto y respuestas claras |

### 🟡 Prioridad 2 — Ampliar ecosistema + contenido (2-3 semanas)

#### Fase F — Automatización y CI (dependencia: P1 Fase D; 🤖 agente)

| # | Tarea | Ejecutor | Estado | Notas |
|---|-------|----------|--------|-------|
| 2.1 | Crear workflow `ai-context-drift-check.yml` | 🤖 | ⬜ | Cron semanal: detecta cambios API, abre PR acumulativa |
| 2.2 | Añadir step de CI para validar archivos generados | 🤖 | ⬜ | `python tools/generate_ai_context_files.py --check` en CI existente |
| 2.3 | Añadir "actualizar archivos AI" al checklist de release | 🤖 | ⬜ | En CONTRIBUTING.md o release process docs |
| 2.4 | Configurar `robots.txt` y `sitemap.xml` para `llms.txt` | 🤖 | ⬜ | `Allow: /llms.txt`, `Allow: /llms-full.txt` + enlace en sitemap |
| 2.5 | Crear `.github/instructions/tests.md` para Copilot | 🤖 | ⬜ | Glob: `**/tests/**`. Analizar tests existentes del repo para extraer patrones reales. Ver detalle abajo |

#### Fase G — Contenido complementario (dependencia: P1 Fase D; 🤖 prepara, 👤 publica)

| # | Tarea | Ejecutor | Estado | Notas |
|---|-------|----------|--------|-------|
| 2.6 | Crear set de prompts de prueba para validar GPT y contexto | 🤖 | ⬜ | 5-10 prompts con respuestas esperadas; el humano los ejecuta periódicamente |
| 2.7 | Añadir links en README, docs y web | 🤖 | ⬜ | Sección "AI Assistants" con links a GPT, skills, llms.txt |
| 2.8 | Crear notebook de demos para el Custom GPT | 🤖 | ⬜ | Casos de uso típicos documentados |
| 2.9 | Crear 2-3 notebooks para Kaggle con buen SEO | 🤖/👤 | ⬜ | Agente crea el contenido, humano publica en Kaggle |

#### Fase H — Plataformas externas (dependencia: P1 Fase E; 👤 manual)

| # | Tarea | Ejecutor | Estado | Notas |
|---|-------|----------|--------|-------|
| 2.10 | Crear Google Gem para Gemini | 👤 | ⬜ | Mismas instrucciones adaptadas |
| 2.11 | Crear Claude Project con instrucciones | 👤 | ⬜ | Para usuarios de Claude |
| 2.12 | Publicar Custom GPT en GPT Store | 👤 | ⬜ | Requiere pruebas previas (1.15) |
| 2.13 | Estrategia de contenido: responder en Stack Overflow | 👤 | ⬜ Continuo | Buscar preguntas de forecasting y responder usando skforecast |

### 🟢 Prioridad 3 — MCP Server (1-2 meses)

| # | Tarea | Ejecutor | Estado | Notas |
|---|-------|----------|--------|-------|
| 3.1 | Diseñar API de tools de alto nivel | 🤖/👤 | ⬜ | Definir firmas, parámetros y respuestas **antes** de implementar. Ver sección detallada abajo |
| 3.2 | MCP Server MVP: implementar tools `load_and_analyze_data` + `forecast` | 🤖 | ⬜ | Solo 2 tools como prueba de concepto |
| 3.3 | Implementar tool `compare_models` | 🤖 | ⬜ | Comparar varios enfoques automáticamente |
| 3.4 | Implementar tool `explain_forecast` | 🤖 | ⬜ | Tendencia, estacionalidad, importancia de features |
| 3.5 | Tests y validación con datasets variados | 🤖 | ⬜ | Antes de empaquetar |
| 3.6 | Empaquetar como `skforecast-mcp` en PyPI | 🤖 | ⬜ | Instalable con `pip install skforecast-mcp` |
| 3.7 | Documentar configuración para Claude Desktop / VS Code | 🤖 | ⬜ | Guía step-by-step |

---

## Detalle de cada iniciativa

### 1. Archivos de contexto: fuente única y generación automática

#### Arquitectura de archivos

```
skforecast/
├── llms.txt                              ← FUENTE 1: Core reference (mantener manualmente)
├── skills/                               ← FUENTE 2: Skills modulares (mantener manualmente)
│   ├── forecasting-single-series/
│   │   └── SKILL.md
│   ├── forecasting-multiple-series/
│   │   └── SKILL.md
│   ├── hyperparameter-optimization/
│   │   └── SKILL.md
│   ├── statistical-models/
│   │   └── SKILL.md
│   ├── prediction-intervals/
│   │   └── SKILL.md
│   ├── feature-engineering/
│   │   └── SKILL.md
│   ├── feature-selection/
│   │   └── SKILL.md
│   ├── drift-detection/
│   │   └── SKILL.md
│   ├── deep-learning-forecasting/
│   │   └── SKILL.md
│   ├── choosing-a-forecaster/
│   │   └── SKILL.md
│   ├── common-errors/
│   │   └── SKILL.md
│   └── complete-api-reference/
│       ├── SKILL.md
│       └── references/
│           └── method-signatures.md  ← Contenido largo (>500 líneas), referenciado desde SKILL.md
├── tools/
│   ├── ai_context_header.md              ← FUENTE 3: Header de desarrollo (mantener manualmente)
│   └── generate_ai_context_files.py      ← Script que genera todo
│
│  ── ARCHIVOS GENERADOS (NO editar manualmente) ──
├── llms-full.txt                         ← Generado: llms.txt + todos los SKILL.md (sin frontmatter)
├── .github/copilot-instructions.md       ← Generado: header + llms.txt (corto)
├── AGENTS.md                             ← Generado: header + llms.txt (corto)
├── .claude/CLAUDE.md                     ← Generado: header + llms.txt (corto)
├── .windsurfrules                        ← Generado: header + llms.txt (corto)
├── .cursor/rules/skforecast.mdc          ← Generado: frontmatter + header + llms.txt (corto)
├── .github/instructions/
│   └── tests.md                          ← Manual: convenciones de testing (solo Copilot, glob **/tests/**)
├── .gitattributes                        ← Marcar archivos generados como linguist-generated
└── docs/
    ├── llms.txt                          ← Generado: copia para web
    └── llms-full.txt                     ← Generado: copia para web
```

**`.gitattributes`** para que los PRs no muestren diffs enormes de archivos auto-generados:
```
llms-full.txt                    linguist-generated=true
.github/copilot-instructions.md  linguist-generated=true
AGENTS.md                        linguist-generated=true
.claude/CLAUDE.md                linguist-generated=true
.windsurfrules                   linguist-generated=true
.cursor/rules/skforecast.mdc     linguist-generated=true
docs/llms.txt                    linguist-generated=true
docs/llms-full.txt               linguist-generated=true
```

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

(The content below is the full `llms.txt` and applies to any user of skforecast)
```

#### `.github/instructions/tests.md` — Convenciones de testing para Copilot

Archivo **mantenido manualmente** (no generado por el script). Solo lo lee GitHub Copilot, y solo cuando se editan archivos que coincidan con el glob `**/tests/**`.

**Proceso para crearlo**: El agente que implemente esta tarea debe **analizar los tests existentes** del repo (ej. `skforecast/recursive/tests/`, `skforecast/direct/tests/`, `skforecast/preprocessing/tests/`) para extraer los patrones reales que usa el proyecto. No inventar convenciones — documentar las que ya se siguen.

**Principios clave que debe reflejar el archivo:**

1. **Tests justos y necesarios**: No repetir el mismo test múltiples veces para chequear cosas distintas cuando se pueden agrupar los checks en un único test.
2. **Parametrizar**: Usar `@pytest.mark.parametrize` cuando el mismo test se ejecuta con diferentes inputs/configuraciones en lugar de escribir N tests casi idénticos.
3. **Agrupación de asserts**: Si un test verifica un objeto con múltiples propiedades (ej. el resultado de `fit()`), agrupar los asserts en un solo test en vez de crear un test por cada propiedad.
4. **No tests triviales**: No testear lo que ya testea scikit-learn o pandas. Testear el comportamiento específico de skforecast.

**Contenido a extraer del análisis de tests existentes:**
- Estructura de carpetas y naming (`test_{modulo}_{metodo}_{escenario}.py` o similar)
- Fixtures compartidos (¿se usan `conftest.py`? ¿a qué nivel?)
- Patrones de parametrización que ya se usan
- Cómo se manejan los warnings esperados
- Qué tipos de assertions se prefieren (`pd.testing.assert_frame_equal`, `np.testing.assert_array_almost_equal`, etc.)
- Convenciones para test data (¿se crean inline o se cargan de fixtures?)

---

#### Skills: diseño y mejores prácticas

Los skills siguen el [Agent Skills spec](https://agentskills.io/specification), un estándar abierto para contenido modular consumible por agentes AI.

##### Estructura de cada skill

```
skills/{skill-name}/
├── SKILL.md              ← Archivo principal (< 500 líneas recomendado)
└── references/           ← Opcional: contenido largo referenciado desde SKILL.md
    └── method-signatures.md
```

##### Formato de `SKILL.md`

Cada `SKILL.md` tiene 2 partes:

1. **Frontmatter YAML** (metadatos):
```yaml
---
name: forecasting-single-series
description: >
  Teaches the agent how to forecast a single time series using
  ForecasterRecursive with scikit-learn compatible estimators.
  Covers training, prediction, backtesting and evaluation.
---
```

2. **Cuerpo Markdown** (instrucciones para el agente):
```markdown
# Forecasting a Single Time Series

## When to use
Use ForecasterRecursive when you have a single time series...

## Complete workflow
\```python
from skforecast.recursive import ForecasterRecursive
...
\```

## Common mistakes
...
```

##### Reglas de frontmatter

| Campo | Requerido | Restricciones |
|-------|-----------|---------------|
| `name` | ✅ | Max 64 chars, lowercase, solo `a-z`, `0-9`, `-` |
| `description` | ✅ | Max 1024 chars, tercera persona ("Teaches the agent..."), sin mencionar versiones |
| `license` | ❌ | String (ej. "MIT") |
| `compatibility` | ❌ | Lista de agentes compatibles |

##### Convenciones de naming

- **Gerundio o sustantivo compuesto**: `forecasting-single-series`, `hyperparameter-optimization`
- **Lowercase con hyphens**: nunca `camelCase` ni `snake_case`
- **Descriptivo y corto**: el nombre debe indicar qué hace el skill sin abrir el archivo

##### Progressive disclosure (carga bajo demanda)

```
Nivel 1: name + description    ← Se carga al inicio (lista de skills disponibles)
Nivel 2: SKILL.md completo     ← Se carga cuando el agente activa el skill
Nivel 3: references/*.md       ← Se carga solo si el agente necesita detalle adicional
```

Los agentes que soportan progressive disclosure (Claude Code, Cursor, etc.) solo cargan el archivo completo cuando es relevante para la tarea del usuario. Esto reduce tokens consumidos vs. inyectar todo el contenido en cada prompt.

##### Buenas prácticas para escribir skills (fuente: Anthropic + Agent Skills spec)

1. **Concisión**: Cada línea debe justificar su existencia. Preguntarse "¿Realmente necesita esto el agente?" antes de añadir contenido.
2. **Tercera persona en `description`**: "Teaches the agent how to..." — no "Learn how to..." ni "You can...".
3. **Un nivel de profundidad en references**: `SKILL.md` puede referenciar archivos en `references/`, pero esos archivos NO deben referenciar otros. Cadenas de referencias confunden a los agentes.
4. **Workflows con pasos claros**: Cada workflow debe seguir el patrón: imports → datos → configuración → fit → predict → evaluación.
5. **Ejemplos ejecutables**: Todo bloque de código debe ser copy-pasteable y funcional (no pseudo-código).
6. **No duplicar `llms.txt`**: Los skills asumen que el agente ya tiene el contenido de `llms.txt`. No repetir imports, estructura del proyecto, ni API overview.
7. **Terminología consistente**: Usar los mismos nombres que la documentación oficial (ej. "forecaster", no "model"; "backtesting", no "walk-forward validation").
8. **Idioma inglés**: Todos los skills deben escribirse en inglés (contenido, frontmatter y comentarios de código). El público objetivo es global.

> **Nota sobre la estabilidad del Agent Skills spec**: El spec es emergente (v0.x). Si cambia el formato del frontmatter o la estructura de directorios, la adaptación sería local al script de generación y a los propios skills — no afecta a `llms.txt` ni a los archivos de IDE. El riesgo se mitiga porque el formato subyacente es Markdown + YAML, que son estándares maduros.

##### Catálogo de skills a crear

| Skill | Descripción (frontmatter) | Contenido principal | Líneas est. |
|-------|--------------------------|---------------------|-------------|
| `forecasting-single-series` | Teaches how to forecast a single time series using ForecasterRecursive | Workflow completo: imports, datos, fit, predict, backtesting, plot | ~150 |
| `forecasting-multiple-series` | Teaches how to forecast multiple related series using ForecasterRecursiveMultiSeries | Workflow con series dict/DataFrame, encoding, predict por levels | ~150 |
| `hyperparameter-optimization` | Teaches how to find optimal hyperparameters using grid, random, and Bayesian search | grid_search, random_search, bayesian_search con TimeSeriesFold | ~200 |
| `statistical-models` | Teaches how to use ARIMA, ETS, SARIMAX and ARAR via ForecasterStats | Arima, Auto-ARIMA, Ets, Sarimax con fit/predict | ~150 |
| `prediction-intervals` | Teaches how to generate prediction intervals using bootstrapping and conformal methods | predict_interval, métodos, backtesting con intervals | ~150 |
| `feature-engineering` | Teaches how to use RollingFeatures, DateTimeFeatureTransformer and differencing | RollingFeatures, TimeSeriesDifferentiator, DateTimeFeatureTransformer | ~120 |
| `feature-selection` | Teaches how to select optimal lags, window features and exogenous variables | select_features, select_features_multiseries con RFECV | ~100 |
| `drift-detection` | Teaches how to detect data drift in production using range and population detectors | RangeDriftDetector, PopulationDriftDetector | ~120 |
| `deep-learning-forecasting` | Teaches how to use ForecasterRnn with Keras for RNN/LSTM forecasting | create_and_compile_model, ForecasterRnn | ~120 |
| `choosing-a-forecaster` | Helps decide which forecaster to use based on the user's data and requirements | Tabla de decisión, diagrama de flujo, recomendaciones | ~100 |
| `common-errors` | Lists common errors LLMs make with skforecast and their solutions | Tabla error-causa-solución, deprecated imports | ~80 |
| `complete-api-reference` | Complete constructor and method signatures for all forecasters and functions | Índice de firmas + referencia a `references/method-signatures.md` | ~100 + refs |

**Nota**: El skill `complete-api-reference` usa `references/method-signatures.md` para el contenido largo (firmas detalladas de todos los métodos). El `SKILL.md` principal contiene un índice resumido y referencia al archivo de detalle. Esto cumple la recomendación de < 500 líneas por `SKILL.md`.

##### Distribución de skills

Los skills se distribuyen automáticamente a través de múltiples canales:

| Canal | Mecanismo | Acción del usuario |
|-------|-----------|-------------------|
| **Vercel Skills CLI** | `npx skills add` detecta `skills/` en el repo | El dev ejecuta el comando en su proyecto |
| **Clonado del repo** | `skills/` ya está en el repo | Disponible automáticamente para agentes locales |
| **`llms-full.txt`** | El script concatena todos los skills en un solo archivo | El usuario pega la URL o lo sube como knowledge |

---

#### Contenido de los skills (especificación del contenido)

El contenido que antes iba en `llms-full.txt` como un monolito ahora se distribuye en skills independientes. A continuación se detalla qué debe contener cada grupo de skills.

**A) Skills de workflows end-to-end** (5 skills, el usuario dice QUÉ quiere, no CÓMO):

Cada skill es un workflow completo y auto-contenido. Se deben incluir scripts funcionales para estos escenarios:

| Skill | Trigger del usuario | Contenido del workflow |
|-------|-------------------|----------------------|
| `forecasting-single-series` | "Quiero predecir una serie temporal" | ForecasterRecursive + backtesting + plot |
| `forecasting-multiple-series` | "Tengo varias series y quiero predecirlas todas" | ForecasterRecursiveMultiSeries + encoding + predict por levels |
| `hyperparameter-optimization` | "Quiero encontrar los mejores hiperparámetros" | bayesian_search_forecaster + TimeSeriesFold + resultados |
| `statistical-models` | "Quiero usar ARIMA" | ForecasterStats + Arima/Ets/Sarimax + Auto-ARIMA |
| `prediction-intervals` | "Quiero predecir con intervalos de confianza" | predict_interval + bootstrapping/conformal + backtesting |

Cada workflow debe incluir: imports, carga de datos, split, entrenamiento, predicción, evaluación y visualización.

**B) Skill `choosing-a-forecaster`** — Guía de decisión:

| Situación | Forecaster | Por qué | Datos mínimos recomendados |
|-----------|-----------|--------|---------------------------|
| Una serie, caso general | ForecasterRecursive | Default, rápido y flexible | ≥50 observaciones |
| Una serie, horizonte largo | ForecasterDirect | Modelo independiente por step | ≥100 observaciones |
| Múltiples series relacionadas | ForecasterRecursiveMultiSeries | Patrones compartidos | ≥50 obs/serie, ≥3 series |
| Necesito ARIMA/ETS/SARIMAX | ForecasterStats + Arima/Ets/Sarimax | Modelos estadísticos | ≥30 observaciones |
| Múltiples inputs → un output | ForecasterDirectMultiVariate | Multivariante | ≥100 observaciones |
| Clasificación (sube/baja) | ForecasterRecursiveClassifier | Predicción categórica | ≥100 observaciones |
| Baseline rápido | ForecasterEquivalentDate | Referencia de fechas equivalentes | ≥2 ciclos estacionales |
| Deep learning (RNN/LSTM) | ForecasterRnn | Redes neuronales, requiere Keras | ≥500 observaciones |

Los valores de "Datos mínimos" son orientativos — dependen del caso de uso, la frecuencia y la complejidad del patrón.

**C) Skill `common-errors`** — Errores comunes que los LLMs suelen cometer:

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

**D) Skills técnicos** (4 skills — feature engineering, selection, drift detection, deep learning):

Cada skill técnico cubre una funcionalidad específica de skforecast. A diferencia de los skills de workflow (que son end-to-end), estos son skills de referencia que se activan cuando el agente necesita una funcionalidad concreta.

| Skill | Contenido clave |
|-------|----------------|
| `feature-engineering` | `RollingFeatures` con stats/window_sizes, `TimeSeriesDifferentiator` (incluyendo inverse_transform), `DateTimeFeatureTransformer` con features list. Ejemplo de combinación de los 3 en un forecaster |
| `feature-selection` | `select_features()` y `select_features_multiseries()` con `RFECV`. Parámetros `select_only`, `force_inclusion`, `subsample`. Cómo actualizar el forecaster tras la selección |
| `drift-detection` | `RangeDriftDetector` (fit con series/exog, predict con last_window). `PopulationDriftDetector` (chunk_size, threshold, threshold_method). Cuándo usar cada uno |
| `deep-learning-forecasting` | `create_and_compile_model()` con layers, `ForecasterRnn` con fit/predict. Requisitos (keras>=3.0). Ejemplo mínimo funcional |

**E) Skill `complete-api-reference`** — Referencia completa de API:

Este skill es un caso especial: el `SKILL.md` principal (~100 líneas) contiene un índice organizado por módulo con las firmas resumidas, y referencia `references/method-signatures.md` para las firmas completas con todos los parámetros y tipos.

Estructura:
```
complete-api-reference/
├── SKILL.md                      ← Índice: clases y funciones agrupadas por módulo, firma resumida
└── references/
    └── method-signatures.md      ← Firmas completas: constructor + métodos con todos los params y tipos
```

El contenido de `references/method-signatures.md` se extrae de `llms.txt` (sección de API) y se amplía con firmas de métodos que no están en `llms.txt` (ej. `set_params()`, `get_feature_importances()`, `set_out_sample_residuals()`).

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
- `llms-full.txt` (generado automáticamente: core reference + todos los skills)

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

**MVP (Prioridad 3):**

```python
@tool
def load_and_analyze_data(file_path: str) -> dict:
    """Load a CSV/Excel file and return a summary (columns, frequency, date range, stats, missing values)."""

@tool
def forecast(file_path: str, target_column: str, date_column: str, steps: int,
             frequency: str = "auto", exog_columns: list[str] | None = None,
             confidence_level: float = 0.80) -> dict:
    """Train a forecaster and return predictions with confidence intervals.
    Auto-selects model/lags, validates with backtesting. Returns predictions, metric, plot_base64."""
```

**Completo (Prioridad 3, posterior al MVP):**

```python
@tool
def compare_models(file_path, target_column, date_column, steps, frequency="auto") -> dict:
    """Compare LightGBM, RandomForest, Ridge, ARIMA, ETS and return ranked results."""

@tool
def explain_forecast(file_path, target_column, date_column, frequency="auto") -> dict:
    """Analyze trend, seasonality, feature importance, stationarity. Returns human-readable summary."""
```

#### Estructura del paquete

```
skforecast-mcp/
├── pyproject.toml
├── README.md
├── src/skforecast_mcp/
│   ├── __init__.py
│   ├── server.py              # MCP server entry point
│   ├── tools/{load_data, forecast, compare, explain}.py
│   └── utils/{auto_detect, plotting}.py
└── tests/
```

#### Distribución
- Publicar en PyPI como `skforecast-mcp`
- El usuario instala: `pip install skforecast-mcp`
- Configuración en Claude Desktop / VS Code:
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

---

### 6. Workflow de detección de cambios API (`ai-context-drift-check`)

#### Problema

Entre releases, se van acumulando cambios en la rama de desarrollo que afectan a la API pública. Es fácil olvidar reflejarlos en `llms.txt` y/o `skills/` cuando llega el momento del release.

#### Solución: GitHub Action con cron semanal + PR acumulativa

Un workflow que se ejecuta cada lunes, analiza el diff de la rama de desarrollo contra el último release tag, y mantiene **una sola PR abierta** que se va engordando con los cambios detectados cada semana.

```
Semana 1: workflow detecta cambios → crea PR con tools/ai_changes_pending.md
Semana 2: workflow detecta más cambios → actualiza la misma PR (append)
Semana 3: sin cambios API → no toca la PR
Release: tú mergeas la PR → usas la lista para actualizar llms.txt y/o skills/
```

#### Qué detecta (sin LLM, puro `git diff` + `grep`)

| Categoría | Cómo lo detecta |
|-----------|----------------|
| Exports nuevos/eliminados | `git diff` en `*/__init__.py` — líneas `+`/`-` con imports |
| Firmas modificadas | `git diff` en `*.py` — líneas con `def ` que cambian |
| Archivos nuevos en módulos públicos | `git diff --name-status` — archivos con status `A` |
| Cambios en dependencias | `git diff` en `pyproject.toml` |
| Deprecaciones | `git diff` en `*.py` — líneas con `deprecated` (case-insensitive) |

Excluye automáticamente: `tests/`, `docs/`, `dev/`, `tools/`, `benchmarks/`.

#### Workflow: `.github/workflows/ai-context-drift-check.yml`

Estructura principal del workflow:

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
          fetch-depth: 0          # Full history for tags
      - name: Detect API changes  # git diff last_tag..HEAD, collect sigs/exports/deps/deprecations
        id: detect
        run: ...                  # See full implementation in tools/
      - name: Update pending changes file
        if: steps.detect.outputs.has_changes == 'true'
        run: ...                  # Append to tools/ai_changes_pending.md
      - name: Create or update PR
        if: steps.detect.outputs.has_changes == 'true'
        uses: peter-evans/create-pull-request@v6
        with:
          branch: ai-context-drift-check
          title: "🔄 AI context files — API changes detected"
          labels: documentation, ai-context
```

#### Archivo controlado: `tools/ai_changes_pending.md`

Ejemplo del formato de detección:

```markdown
## Detected 2026-03-04 (0.21.x, commits abc123..def456)

### Modified signatures
- `bayesian_search_forecaster()`: new param `suppress_warnings: bool = False`

### New exports
- `skforecast.preprocessing.ConformalIntervalCalibrator`

### Dependency changes
- `optuna>=2.10` → `optuna>=3.0`
```

#### Flujo de trabajo completo

```
GitHub Actions (cron lunes)        →  Detecta diff API → actualiza PR acumulativa
        ↓
Mergeas la PR                      →  ai_changes_pending.md en tu rama local
        ↓
VS Code + Copilot                  →  "Mira ai_changes_pending.md y actualiza:
                                       - llms.txt si cambian imports, API o estructura
                                       - skills/ si cambian workflows o ejemplos"
        ↓
python tools/generate_ai_context_files.py  →  Limpiar ai_changes_pending.md → commit
```

**Coste total**: 0€ (GitHub Actions gratis para repos públicos, `peter-evans/create-pull-request` gratuita, no requiere LLM).

#### Opciones descartadas para este workflow

| Opción | Por qué se descarta |
|--------|-------------------|
| `.github/instructions/*.md` para contexto general | Se inyectan **además de** `copilot-instructions.md` — con el mismo contenido duplicaría tokens. Sin embargo, **sí se usa para contexto especializado** como `.github/instructions/tests.md` (ver tarea 2.13) que solo se inyecta al editar archivos de test |
| `.github/agents/*.md` (Copilot custom agents `@nombre`) | No se auto-inyectan — el usuario debe invocar `@agent-name`. Muy nicho. Considerar en el futuro |
| Que el workflow edite `llms.txt` / `skills/` con un LLM | Riesgo de que el LLM malinterprete un cambio. Las fuentes de verdad no deben editarse automáticamente |
| Abrir Issues en vez de PR | Una PR con archivo controlado queda en el repo, es visible en el diff, y es acumulativa. Un Issue es fácil de ignorar |

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
- **Any AI agent**: `npx skills add` — installs skforecast skills for 40+ agents automatically
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

### Discoverabilidad web de `llms.txt`

Tres acciones para que los crawlers de LLMs (GPTBot, ClaudeBot, etc.) encuentren `llms.txt`:

1. **`robots.txt`**: Añadir `Allow: /llms.txt` y `Allow: /llms-full.txt` explícitamente.
2. **`sitemap.xml`**: Incluir ambos archivos como URLs en el sitemap.
3. **HTML `<link>` en el `<head>`**: Añadir en la configuración de MkDocs:
   ```html
   <link rel="help" type="text/plain" href="/llms.txt" />
   ```
   Esto sigue el estándar emergente de `llms.txt` y permite descubrimiento automático.

### En redes sociales / blog

Post de lanzamiento: *"Ahora puedes hacer forecasting sin escribir código. Sube tu CSV a nuestro GPT y pregúntale cuánto vas a vender mañana."*

---

## Qué NO hacer

| Idea | Por qué no |
|------|-----------|
| Crear un agente custom con backend/servidor propio | Coste de infra enorme para un proyecto OSS, el usuario tiene que confiar en un tercero con su API key |
| Exponer la API de bajo nivel de skforecast como tools del MCP | Demasiado complejo para que un LLM lo use correctamente en nombre de un no-coder |
| Construir una web app completa para no-coders | Eso es un producto entero (ya existe Skforecast Studio) |
| Editar manualmente los archivos generados (incluido `llms-full.txt`) | Se sobreescribirán la próxima vez que se ejecute el script. Editar SOLO las fuentes (`llms.txt` y `skills/`) |
| Crear instrucciones solo con referencia de API | Los LLMs necesitan workflows completos end-to-end, no solo documentación |
| Ignorar el training data | Los archivos de contexto solo sirven en IDEs. Lo que hace que un LLM conozca skforecast a nivel global es el contenido público (docs, SO, Kaggle, blogs) |

---

## Cronograma estimado

| Semana | Fase | Tareas | Ejecutor principal |
|--------|------|--------|--------------------|
| **Semana 1** | A+B | 1.1-1.5: Auditar copilot-instructions.md, crear header, .gitattributes, pyproject.toml, script | 🤖 |
| **Semana 1-2** | C | 1.6-1.10: Crear los 12 skills (workflows, técnicos, referencia) | 🤖 |
| **Semana 2** | D | 1.11-1.13: Ejecutar script, validar skills, verificar web | 🤖/👤 |
| **Semana 2-3** | E | 1.14-1.15: Crear y probar Custom GPT | 👤 |
| **Semana 3** | F | 2.1-2.5: Workflow drift, CI check, release checklist, robots.txt, tests.md | 🤖 |
| **Semana 3-4** | G | 2.6-2.9: Prompts de prueba, links en docs, notebook demos, Kaggle | 🤖/👤 |
| **Semana 4** | H | 2.10-2.13: Google Gem, Claude Project, publicar GPT, Stack Overflow | 👤 |
| **Semana 4-5** | — | Recoger feedback de usuarios y iterar sobre instrucciones y GPT | 👤 |
| **Mes 2-3** | — | 3.1-3.7: MCP Server diseño + MVP + completo, publicar en PyPI | 🤖/👤 |

---

## Proceso de mantenimiento (cada release)

| Paso | Acción |
|------|--------|
| 1 | Mergear la PR acumulativa de `ai-context-drift-check` (si hay cambios pendientes) |
| 2 | Revisar `tools/ai_changes_pending.md` — usar como checklist de qué actualizar |
| 3 | Actualizar `llms.txt` (imports, API, estructura) y/o `skills/` (workflows, ejemplos) según los cambios detectados |
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
- Adopción de skills: clones del repo con `skills/`, uso de `npx skills add` (si Vercel publica analytics)

---

## Validación: cómo probar que funciona

Crear un set de **5-10 prompts de prueba** con respuestas esperadas y ejecutarlos periódicamente:

| # | Prompt de prueba | Respuesta esperada | Dónde probar |
|---|-----------------|--------------------|--------------|
| 1 | "Importa ForecasterRecursive" | `from skforecast.recursive import ForecasterRecursive` | Custom GPT, IDE con contexto |
| 2 | "Predice una serie temporal mensual" | Script completo con ForecasterRecursive + backtesting | Custom GPT |
| 3 | "¿Qué forecaster uso para 5 series relacionadas?" | ForecasterRecursiveMultiSeries con explicación | Custom GPT, Claude Project |
| 4 | "Importa ForecasterAutoreg" | Debe corregir a ForecasterRecursive | Custom GPT, IDE con contexto |
| 5 | "Haz grid search con validación cruzada" | Script con grid_search_forecaster + TimeSeriesFold | Custom GPT |
| 6 | "Tengo NaN en mis datos" | Recomendar interpolación/fillna ANTES de fit | Custom GPT |
| 7 | "Predice con intervalos de confianza" | predict_interval con method y backtesting | Custom GPT |

**Frecuencia**: Ejecutar después de cada actualización de `llms.txt`, `skills/`, o del Custom GPT. También útil como smoke test antes de publicar en GPT Store.

**Validación de archivos de IDE**: Verificar que cada plataforma lee su archivo correctamente:
- VS Code: Abrir repo, preguntar a Copilot algo de skforecast, verificar que usa imports correctos
- Cursor: Idem con Cursor
- Claude Code: `claude` en el repo, preguntar y verificar
