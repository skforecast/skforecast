---
paths:
  - "tools/ai/**"
  - "skills/**"
  - "llms.txt"
  - "llms-full.txt"
  - "docs/llms*.txt"
  - "AGENTS.md"
  - ".github/copilot-instructions.md"
  - "context7.json"
  - ".claude-plugin/**"
---

# AI context files

These files are produced by `tools/ai/generate_ai_context_files.py`; any manual edit is lost on the next run (and a PreToolUse hook blocks it):

| Generated file | Edit this instead |
|:---------------|:------------------|
| `AGENTS.md`, `.github/copilot-instructions.md` | `tools/ai/ai_context_header.md` (dev rules), `tools/ai/llms-base.txt` (API reference) |
| `llms-full.txt`, `docs/llms-full.txt` | `tools/ai/llms-base.txt` plus `skills/*/SKILL.md` and `references/` |
| `docs/llms.txt` | `llms.txt` at the repository root |
| The `SKILLS-LIST` block in `CLAUDE.md` | The set of folders in `skills/` |

After editing any source, regenerate and verify:

```bash
python tools/ai/generate_ai_context_files.py           # regenerate all
python tools/ai/generate_ai_context_files.py --check   # what CI runs
```

`.github/workflows/ai-context-check.yml` fails any pull request (to `main` or `*.x`) whose generated files are stale.

Gotchas:

- Public API change: grep the function or class name in `tools/ai/llms-base.txt`, `skills/*/SKILL.md` and `skills/*/references/*.md`, and fix every example (signature, defaults, return values). `--check` only detects stale generated files, not outdated examples in the sources.
- Adding a skill: create `skills/<name>/SKILL.md`, add `<name>` to `SKILL_ORDER` in the generator, then regenerate. A skill body must stay at or below 500 lines.
- Bumping the version: update `__version__` in `skforecast/__init__.py`, `Version:` in `tools/ai/llms-base.txt` and the plugin `version` in `.claude-plugin/marketplace.json`. The `--check` run compares them. `CITATION.cff` has no version on purpose (Zenodo takes it from the GitHub release).
- `context7.json` and `.claude-plugin/marketplace.json` are maintained by hand and validated by `--check` (see "Distribution to user agents" in `tools/ai/README.md`). There is deliberately no copy of `skills/` inside the pip package.
- Adding a public export: it must also appear as an import in `tools/ai/llms-base.txt`, which `--check` verifies against each subpackage `__init__.py`.
- The skills in `skills/` are for skforecast users (they are published inside `llms-full.txt`). Contributor workflows live in `.claude/skills/` and are not part of the generated files.
- Full description of the system: `tools/ai/README.md`.
