# AI Context System

This directory contains the source files and generation script that power all AI-assisted development in skforecast. Every AI context file in the repository — IDE instructions, LLM references, and documentation copies — is generated from the files here.

## File types at a glance

VS Code + GitHub Copilot recognises several file conventions. Each serves a different role:

| File type | Location | What it is | Best-practice objective |
|-----------|----------|------------|------------------------|
| **copilot-instructions.md** | `.github/copilot-instructions.md` | Global project context injected into **every** Copilot Chat conversation. Contains the API reference, code style, and testing conventions that Copilot should always know about. | Provide the AI with a persistent baseline of project-specific knowledge so it never generates code that contradicts the architecture, style, or API conventions of the repo. |
| **instructions** | `.github/instructions/*.instructions.md` | Targeted coding conventions that activate **only when the open file matches** an `applyTo` glob in the YAML frontmatter (e.g. docstring rules for `*.py`, testing rules for `tests/`). | Keep the global context lean by extracting domain-specific rules into separate files that load only when relevant, avoiding prompt bloat while ensuring precision in specialized tasks. |
| **prompts** | `.github/prompts/*.prompt.md` | Reusable prompt templates the developer runs **on demand** from the Copilot Chat prompt picker. Used for review checklists and guided workflows. | Standardize repetitive developer workflows (reviews, audits, migrations) into shareable, version-controlled templates that any team member can invoke identically. |
| **skills** | `skills/*/SKILL.md` | Self-contained workflow guides that the AI agent **discovers automatically** when the user's question matches the skill description. Each skill covers one topic end-to-end. | Encode deep domain knowledge (decision trees, pitfalls, end-to-end examples) in modular documents the agent retrieves on demand, enabling expert-level guidance without overloading the base context. |
| **agents** | `.github/agents/*.agent.md` | Custom agent modes that appear in the Copilot Chat **agent picker**. Each defines a specialized AI persona with its own system prompt, allowed tools, and constraints. | Create purpose-built agent configurations for recurring workflows (e.g. a "reviewer" agent that only reads files and runs tests, or a "docs" agent restricted to documentation folders) so developers get a tailored experience without manual prompt engineering. |
| **AGENTS.md** | `AGENTS.md` (repo root) | Equivalent to `copilot-instructions.md` but for IDEs that follow the AGENTS.md convention (Claude Code, Codex CLI, Aider). Same content, different standard. | Ensure consistent AI behaviour across different IDEs and tools by providing the same project context through each tool's native convention. |

## How VS Code / GitHub Copilot uses these files

When you open the skforecast repo in VS Code with GitHub Copilot, different files are loaded into the AI's context at different times. Understanding when each file type activates is key to understanding the system.

### copilot-instructions.md (always active)

`.github/copilot-instructions.md` is **injected into every Copilot Chat conversation** automatically. The user does nothing — VS Code reads this file and appends it to the system prompt. This is the baseline context that Copilot always has about skforecast: project structure, API overview, code style, and testing conventions.

- **When**: every conversation, every message
- **Scope**: the entire repository
- **Content**: `ai_context_header.md` (dev conventions) + `llms-base.txt` (API reference)

### .instructions.md files (auto-activated by file pattern)

Files in `.github/instructions/` have an `applyTo` glob in their YAML frontmatter. VS Code automatically adds the matching instruction file to the context **only when the user is editing a file that matches the pattern**. This keeps the context focused — test conventions only appear when writing tests.

| File | Pattern | Active when editing |
|------|---------|---------------------|
| `testing.instructions.md` | `**/tests/**` | Any file inside a `tests/` directory |
| `docstrings.instructions.md` | `skforecast/**/*.py` | Any Python file in the package |

- **When**: only when the active file matches `applyTo`
- **Scope**: additive — loaded on top of `copilot-instructions.md`
- **Content**: detailed conventions that would be too verbose for the global context

### .prompt.md files (user-invoked)

Files in `.github/prompts/` are **reusable prompt templates** that the user explicitly runs via the Copilot Chat prompt picker (type `/` or use the attachment button). They are never loaded automatically.

| File | Purpose |
|------|---------|
| `review-llms-base.prompt.md` | Checklist to review `llms-base.txt` against actual source code |
| `review-skill.prompt.md` | Checklist to review a skill folder for API correctness |

- **When**: only when the user explicitly selects the prompt
- **Scope**: single conversation
- **Content**: structured review checklists with references to source files

### Skills (discovered on demand)

Files in `skills/*/SKILL.md` are **specialized workflow guides** that AI agents can discover and load when the user asks about a specific topic. In VS Code, Copilot reads skills from the `skills/` directory when their description matches the user's question.

- **When**: on demand, when the agent determines a skill is relevant
- **Scope**: single conversation
- **Content**: end-to-end workflows with decision trees, code examples, and pitfalls

### .agent.md files (custom agent modes)

Files in `.github/agents/` define **custom agent modes** that appear in the Copilot Chat agent picker (the model/mode dropdown). Each `.agent.md` file creates a specialized AI persona with its own system prompt and, optionally, restricted tools or scoped instructions.

- **When**: only when the user selects the agent mode from the picker
- **Scope**: entire conversation while that mode is active
- **Content**: a YAML frontmatter (`name`, `description`, `tools`) plus a system-level prompt body that shapes the agent's behaviour

### AGENTS.md (other IDEs)

`AGENTS.md` at the repo root serves the same purpose as `copilot-instructions.md` but for IDEs that follow the AGENTS.md convention (Claude Code, Codex CLI, Aider, and others). It contains identical content.

- **When**: automatically on project open, depending on the IDE
- **Scope**: entire repository

### Summary: context loading order

```
Always loaded:
  └─ .github/copilot-instructions.md        (global API + code style)

Loaded when file pattern matches:
  └─ .github/instructions/docstrings.md      (when editing *.py)
  └─ .github/instructions/testing.md         (when editing tests/)

Loaded on demand by AI agent:
  └─ skills/*/SKILL.md                       (topic-specific workflows)

Loaded when user selects agent mode:
  └─ .github/agents/*.agent.md               (custom agent personas)

Loaded when user explicitly invokes:
  └─ .github/prompts/review-*.prompt.md      (review checklists)
```

## Strategy

The AI context system follows three principles:

1. **Single source of truth** — All generated files derive from `llms-base.txt` and `ai_context_header.md`. Edit the source, regenerate, done.
2. **Layered context** — Global context is always present, specialized context loads only when relevant, avoiding prompt bloat.
3. **CI-enforced consistency** — A GitHub Actions workflow runs `--check` mode on every PR to prevent stale generated files from reaching master.

## Source files (human-maintained)

These are the only files you edit directly:

| File | Lines | Purpose |
|------|-------|---------|
| `tools/ai/llms-base.txt` | ~730 | Core API reference: all forecasters, imports, examples, workflows |
| `tools/ai/ai_context_header.md` | ~95 | Dev-only context: testing commands, code style, dependencies |
| `llms.txt` (root) | ~120 | Public index per [llmstxt.org](https://llmstxt.org) spec with links to docs |
| `skills/*/SKILL.md` | 12 skills | Modular workflow guides, one per topic |
| `skills/*/references/*.md` | 6 files | Supplementary reference tables for some skills |
| `.github/instructions/*.md` | 2 files | Pattern-matched coding conventions |
| `.github/prompts/*.md` | 2 files | Reusable review checklists |

## Generated files (do not edit)

All marked with `<!-- AUTO-GENERATED -->` header and tracked in `.gitattributes` as `linguist-generated=true` (collapsed in GitHub PR diffs).

| File | Source | Content |
|------|--------|---------|
| `.github/copilot-instructions.md` | header + llms-base | IDE context for GitHub Copilot |
| `AGENTS.md` | header + llms-base | IDE context for Claude Code, Codex, Aider |
| `llms-full.txt` | llms-base + 12 skills | Complete LLM reference (~2500 lines) |
| `docs/llms.txt` | copy of root `llms.txt` | Served at skforecast.org/llms.txt |
| `docs/llms-full.txt` | copy of `llms-full.txt` | Served at skforecast.org/llms-full.txt |

## Skills

12 self-contained workflow guides in `skills/`. Each has a `SKILL.md` with YAML frontmatter (`name`, `description`) and optional `references/` subfolder.

| Skill | References | Topic |
|-------|-----------|-------|
| `forecasting-single-series` | — | ForecasterRecursive / ForecasterDirect |
| `forecasting-multiple-series` | — | ForecasterRecursiveMultiSeries global model |
| `statistical-models` | `model-parameters.md` | ARIMA, SARIMAX, ETS, ARAR |
| `hyperparameter-optimization` | `search-parameters.md` | Grid, random, Bayesian search |
| `prediction-intervals` | `interval-compatibility.md` | Bootstrapping, conformal, quantile |
| `feature-engineering` | `rolling-stats-reference.md` | RollingFeatures, calendar features |
| `feature-selection` | — | RFECV, SelectFromModel |
| `drift-detection` | — | RangeDriftDetector, PopulationDriftDetector |
| `deep-learning-forecasting` | `architecture-options.md` | ForecasterRnn, LSTM/GRU |
| `choosing-a-forecaster` | — | Decision guide for forecaster selection |
| `troubleshooting-common-errors` | — | Common mistakes and fixes |
| `complete-api-reference` | `method-signatures.md` | All constructor and method signatures |

## Generation script

```bash
# Generate all files
python tools/ai/generate_ai_context_files.py

# CI mode: fail if any generated file is stale or missing
python tools/ai/generate_ai_context_files.py --check

# Validate URLs in llms.txt are reachable
python tools/ai/generate_ai_context_files.py --check-urls

# Skip specific URL patterns during validation
python tools/ai/generate_ai_context_files.py --check-urls --ignore-urls llms-full.txt
```

### What `--check` validates

| Check | What it verifies |
|-------|-----------------|
| **Skill structure** | Every `skills/*/SKILL.md` has valid YAML frontmatter, `name` matches directory, body ≤ 500 lines |
| **Version consistency** | `Version:` in `llms-base.txt` matches `__version__` in `skforecast/__init__.py` |
| **Imports consistency** | Every public export in subpackage `__init__.py` files appears as an import in `llms-base.txt` |
| **File freshness** | Each generated file matches what the script would produce right now |

### CI enforcement

`.github/workflows/ai-context-check.yml` runs `--check` on every pull request targeting master. If any generated file is stale, the PR check fails with a message indicating which files need regeneration.

## File map

```
skforecast/
├── llms.txt                              # Public index (human-maintained)
├── llms-full.txt                         # Complete reference (generated)
├── AGENTS.md                             # IDE context (generated)
├── .gitattributes                        # Marks generated files
├── .github/
│   ├── copilot-instructions.md           # IDE context (generated)
│   ├── instructions/
│   │   ├── docstrings.instructions.md    # → skforecast/**/*.py
│   │   └── testing.instructions.md       # → **/tests/**
│   ├── prompts/
│   │   ├── review-llms-base.prompt.md    # Review checklist for llms-base.txt
│   │   └── review-skill.prompt.md        # Review checklist for skills
│   └── workflows/
│       └── ai-context-check.yml          # CI: validates generated files
├── skills/
│   ├── forecasting-single-series/
│   │   └── SKILL.md
│   ├── complete-api-reference/
│   │   ├── SKILL.md
│   │   └── references/
│   │       └── method-signatures.md
│   └── ... (10 more skills)
├── tools/ai/
│   ├── README.md                         # This file
│   ├── llms-base.txt                     # Core API reference (source)
│   ├── ai_context_header.md              # Dev conventions (source)
│   └── generate_ai_context_files.py      # Generation + validation script
└── docs/
    ├── llms.txt                          # Website copy (generated)
    └── llms-full.txt                     # Website copy (generated)
```

## Common tasks

**Add a new public class or function to the API:**

1. Edit `tools/ai/llms-base.txt` — add the import line and any relevant docs
2. Run `python tools/ai/generate_ai_context_files.py`
3. Commit the source change and all regenerated files

**Add a new skill:**

1. Create `skills/<skill-name>/SKILL.md` with YAML frontmatter (`name`, `description`)
2. Add the skill name to `SKILL_ORDER` in `generate_ai_context_files.py`
3. Optionally add `skills/<skill-name>/references/*.md` for supplementary content
4. Run `python tools/ai/generate_ai_context_files.py`

**Add a new instruction file:**

1. Create `.github/instructions/<name>.instructions.md` with YAML frontmatter (`description`, `applyTo`)
2. No regeneration needed — VS Code picks it up directly

**Review a skill for correctness:**

1. Open Copilot Chat and invoke the `review-skill` prompt
2. Specify which skill to review

**Bump the version:**

1. Update `__version__` in `skforecast/__init__.py`
2. Update `Version:` in `tools/ai/llms-base.txt`
3. Regenerate — the `--check` validation will catch mismatches
