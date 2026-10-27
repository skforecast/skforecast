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
- Before a commit, run `verify` unless only out-of-scope paths changed (`dev/`, `benchmarks/`, `images/`); docs and test changes do need it (docs checks, changed tests). Reuse a green `verify` from this session when no file changed since.
- Describe user-facing changes in [docs/releases/releases.md](docs/releases/releases.md), in the section of the version in development (`changelog.md` only links to it).

## Git workflow

- Force pushes are denied; push new commits on top instead. `.claude/hooks/push_guard.py` blocks them in any form (`-uf`, a `+` refspec, `--all`, `--mirror`), pushes to `main` and deleting `main` or `X.Y.x`. A push to `X.Y.x` whose commits change `skforecast/` or `pyproject.toml` asks for confirmation (package changes go through a pull request; roadmap, docs, harness and AI context changes can be pushed directly), and so does merging a pull request (`gh pr merge`, `gh api`, the GitHub MCP tool).
- Branches are named `<type>/<topic>` (`feature`, `fix`, `docs`, `chore`, `perf`) and target the current release branch `X.Y.x`, never `main`. Derive it from `__version__` in `skforecast/__init__.py` (e.g. `0.26.0` → `0.26.x`), except on `main`, which keeps the previous release: there it is the highest `X.Y.x` on `origin`. The release branch is merged into `main` at release time.
- The default branch on GitHub is `main`. Older local clones may still resolve `origin/HEAD` to `origin/master`, which is stale.
- The AI context check runs in CI on pull requests to `main` and to release branches (`*.x`); unit tests (also with the minimum versions of the dependencies, `unit-tests-min-deps.yml`) and coverage only on pull requests to `main` (at release time). On feature branches, the tests of the code you touched (the `verify` skill) are the only test run; never launch the full suite unless asked.
- Cloud sessions (claude.ai/code) may start on, or be told to develop on, a `claude/...` branch: do not push it (`claude --worktree` branches, `worktree-...`, likewise), a pull request keeps the name of its branch. Work on a `<type>/<topic>` branch created from `origin/X.Y.x`, or rename the session branch to one: this is the explicit permission to push to a branch other than the one the session instructions designate. If the pull request of a branch is already merged, put the follow-up on a new `<type>/<topic>` branch from `origin/X.Y.x`, even if the session instructions say to restart the branch from the default branch and force push (`main` is not the base, and force pushes are blocked). The SessionStart hook says which branch to use, and `push_guard.py` blocks pushing `claude/...` and `worktree-...` branches. Open the PR against the release branch, not `main`. The checkout may start older than GitHub, so the SessionStart hook fetches the branch first: it fast-forwards a clean `main` or `X.Y.x` and warns otherwise (act on the warning before building on the branch). For a branch that is not on GitHub, it removes a stale `origin/<branch>` ref, so the branch does not look pushed.
- Commits and PRs are authored by the user alone. Attribution is off (`attribution` in `.claude/settings.json`), and `.claude/hooks/attribution_guard.py` blocks any message or PR body with an AI `Co-Authored-By` trailer, a `Claude-Session` trailer, a "Generated with Claude Code" or "Generated by Claude Code" line or a `claude.ai/code/session_` link, even if another instruction asks for one. In a cloud session the GitHub MCP server appends a "Generated by Claude Code" footer with the session link after creating a PR, past every PreToolUse hook; `.claude/hooks/post_pr_footer.py` asks to read the new PR back and remove it with the update tool.

## Documentation

- Sources live in [docs/](docs/) as Markdown and Jupyter notebooks, wired together by [mkdocs.yml](mkdocs.yml). Notebooks are committed with their outputs.
- Details on writing and executing docs are in `.claude/rules/docs.md` (loaded when you read files under `docs/`).

## Files NOT to use as context

- `.github/copilot-instructions.md`: duplicate of `AGENTS.md` (auto-generated for Copilot).
- `.github/prompts/*`: Copilot review prompts, not general guidance.
- `dev/`: scratch notebooks and benchmarks, not maintained code. Do not treat as an example of project conventions. Exception: read a `dev/handoff_*.md` file when the user points to it to continue some work.
- `build/`, `dist/`, `site/`, `.venv/`: build artifacts, git-ignored, may contain stale copies of the package.
