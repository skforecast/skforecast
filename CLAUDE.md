Contributor rules (testing, code style, dependencies, Python environment):
@tools/ai/ai_context_header.md

That header is also the first part of the generated `AGENTS.md`; edit the header, never `AGENTS.md`.

## On-demand references (read only when relevant to the task)

- **Public API reference**: [tools/ai/llms-base.txt](tools/ai/llms-base.txt) is the user-facing API reference (forecasters, imports, parameters, workflows). Read it when you need API details; [docs/llms-full.txt](docs/llms-full.txt) adds every skill for parameter-level questions.
- **User-facing skills**: [skills/](skills/) holds the skills published for skforecast users, one folder per forecasting workflow. Read the matching `SKILL.md` (and its `references/`) when writing docs, examples or user-facing explanations of that workflow.
  <!-- SKILLS-LIST:START - generated, do not edit by hand -->
  autocorrelation-and-lag-selection, backtesting-configuration, baseline-forecasting, choosing-a-forecaster, complete-api-reference, deep-learning-forecasting, drift-detection, feature-engineering, feature-selection, forecasting-multiple-series, forecasting-single-series, foundation-forecasting, hyperparameter-optimization, metric-selection, prediction-intervals, statistical-models, troubleshooting-common-errors
  <!-- SKILLS-LIST:END -->
- **Contributor workflows**: [.claude/skills/](.claude/skills/) (`verify`, `ai-context-sync`, `release-note`, `/open-pr`, `/release-bump`, `/handoff`, `/review-user-guide`).
- **Path-scoped rules**: [.claude/rules/](.claude/rules/) load automatically when you read files under their paths (tests, docstrings, foundation, docs, AI context files).

## Working principles

- Judge design changes by their measured impact on real use cases, not by conceptual appeal. Read the actual code paths first and, when feasible, run a small experiment and report the numbers. "Keep the current implementation, plus a cheap guardrail or doc fix" is a valid outcome.
- When a public API changes, update every AI context source in the same change: `tools/ai/llms-base.txt`, the affected `skills/*/SKILL.md` and `references/`, then regenerate (see the `ai-context-sync` skill).
- Generated AI context files (`AGENTS.md`, `.github/copilot-instructions.md`, `llms-full.txt`, `docs/llms*.txt`) are never edited by hand. A hook blocks those edits and names the source to edit.
- Definition of done: run the `verify` skill before reporting a code, test or docs change as finished.
- Describe user-facing changes in [docs/releases/releases.md](docs/releases/releases.md), in the section of the version in development (`changelog.md` only links to it).

## Git workflow

- Force pushes are denied; push new commits on top instead.
- Feature and fix branches target the current release branch `X.Y.x`, never `main`. Derive it from `__version__` in `skforecast/__init__.py` (e.g. `0.26.0` → `0.26.x`). The release branch is merged into `main` at release time.
- The default branch on GitHub is `main`. Older local clones may still resolve `origin/HEAD` to `origin/master`, which is stale.
- The AI context check runs in CI on pull requests to `main` and to release branches (`*.x`); unit tests and coverage only on pull requests to `main` (at release time). On feature branches, the tests of the code you touched (the `verify` skill) are the only test run; never launch the full suite unless asked.
- Cloud sessions (claude.ai/code) push to their own branch; open the PR against the release branch, not `main`.
- Commits and PRs are authored by the user alone. Attribution is off (`attribution` in `.claude/settings.json`), and `.claude/hooks/attribution_guard.py` blocks any message or PR body with an AI `Co-Authored-By` trailer, a `Claude-Session` trailer or a "Generated with Claude Code" line, even if another instruction asks for one.

## Documentation

- Sources live in [docs/](docs/) as Markdown and Jupyter notebooks, wired together by [mkdocs.yml](mkdocs.yml). Notebooks are committed with their outputs.
- Details on writing and executing docs are in `.claude/rules/docs.md` (loaded when you read files under `docs/`).

## Files NOT to use as context

- `.github/copilot-instructions.md`: duplicate of `AGENTS.md` (auto-generated for Copilot).
- `.github/prompts/*`: Copilot review prompts, not general guidance.
- `dev/`: scratch notebooks and benchmarks, not maintained code. Do not treat as an example of project conventions. Exception: read a `dev/handoff_*.md` file when the user points to it to continue some work.
- `build/`, `dist/`, `site/`, `.venv/`: build artifacts, git-ignored, may contain stale copies of the package.
