See @AGENTS.md for all project behavioral rules, coding standards, and operating instructions.
`AGENTS.md` is generated: to change those rules, edit `tools/ai/ai_context_header.md` and regenerate (see below).

## On-demand references (read only when relevant to the task)

- **Task workflows** — [skills/](skills/) contains one folder per workflow. Before implementing or explaining a workflow that matches one of these, read the matching `SKILL.md` (and its `references/` if present).
  <!-- SKILLS-LIST:START - generated, do not edit by hand -->
  autocorrelation-and-lag-selection, backtesting-configuration, baseline-forecasting, choosing-a-forecaster, complete-api-reference, deep-learning-forecasting, drift-detection, feature-engineering, feature-selection, forecasting-multiple-series, forecasting-single-series, foundation-forecasting, hyperparameter-optimization, metric-selection, prediction-intervals, statistical-models, troubleshooting-common-errors
  <!-- SKILLS-LIST:END -->
- **Writing or updating tests** — read [.github/instructions/testing.instructions.md](.github/instructions/testing.instructions.md) before touching anything under `**/tests/**`.
- **Writing or updating docstrings** — read [.github/instructions/docstrings.instructions.md](.github/instructions/docstrings.instructions.md) before adding or editing NumPy-style docstrings on public APIs.
- **Deep API details not covered in AGENTS.md** — [docs/llms-full.txt](docs/llms-full.txt) is the fullest reference; consult it for parameter-level questions or features missing from the embedded reference.

## Generated files: never edit directly

These files are produced by `tools/ai/generate_ai_context_files.py` and any manual edit is lost on the next run:

| Generated file | Edit this instead |
|:---------------|:------------------|
| `AGENTS.md`, `.github/copilot-instructions.md` | `tools/ai/ai_context_header.md` (dev rules), `tools/ai/llms-base.txt` (API reference) |
| `llms-full.txt`, `docs/llms-full.txt` | `tools/ai/llms-base.txt` plus `skills/*/SKILL.md` |
| `docs/llms.txt` | `llms.txt` at the repository root |
| The `SKILLS-LIST` block in this file | The set of folders in `skills/` |

After editing any source, regenerate and verify:

```bash
python tools/ai/generate_ai_context_files.py           # regenerate all
python tools/ai/generate_ai_context_files.py --check   # what CI runs
```

[.github/workflows/ai-context-check.yml](.github/workflows/ai-context-check.yml) fails any pull request whose generated files are stale.

Related gotchas:

- Adding a skill: create `skills/<name>/SKILL.md`, add `<name>` to `SKILL_ORDER` in the generator, then regenerate. A skill body must stay at or below 500 lines.
- Bumping the version: update `__version__` in `skforecast/__init__.py` and `Version:` in `tools/ai/llms-base.txt`. The `--check` run compares them.
- Adding a public export: it must also appear as an import in `tools/ai/llms-base.txt`, which `--check` verifies against each subpackage `__init__.py`.
- Full description of the system: [tools/ai/README.md](tools/ai/README.md).

## Git workflow

- Feature and fix branches target the current release branch (`X.Y.z`, currently `0.25.x`), not `main`. The release branch is merged into `main` at release time.
- The default branch on GitHub is `main`. Older local clones may still resolve `origin/HEAD` to `origin/master`, which is stale.
- Tests, coverage and the AI context check only run on pull requests targeting `main` (`unit-tests.yml`, `codecov.yml`, `ai-context-check.yml`), so run them locally while working on a release branch. `unit-tests-latest-deps.yml` is weekly and `benchmarks.yml` is manual.
- Update [changelog.md](changelog.md) for user facing changes.

## Documentation

- Sources live in [docs/](docs/) as Markdown and Jupyter notebooks, wired together by [mkdocs.yml](mkdocs.yml).
- Notebooks are committed with their outputs. Re-execute them with `python tools/execute_docs/execute_docs_notebooks.py [subdir_or_notebook]`, which runs papermill and writes warning logs to `tools/execute_docs/logs/`. Notebooks listed in `SLOW_NOTEBOOKS` inside that script are skipped unless `--include-slow` is passed or the notebook is given explicitly.
- Executing the whole `docs/` tree is slow, so pass the specific subdirectory or notebook that changed.

## Files NOT to use as context

- `.github/copilot-instructions.md` — duplicate of `AGENTS.md` (auto-generated for Copilot).
- `.github/prompts/*` — Copilot review prompts, not general guidance.
- `dev/` — scratch notebooks and benchmarks, not maintained code. Do not treat as an example of project conventions.
- `build/`, `dist/`, `site/`, `.venv/` — build artifacts, git-ignored, may contain stale copies of the package.
