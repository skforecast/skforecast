See @AGENTS.md for all project behavioral rules, coding standards, and operating instructions.

## On-demand references (read only when relevant to the task)

- **Task workflows** — [skills/](skills/) contains one folder per workflow (choosing-a-forecaster, complete-api-reference, deep-learning-forecasting, drift-detection, feature-engineering, feature-selection, forecasting-multiple-series, forecasting-single-series, foundation-forecasting, hyperparameter-optimization, prediction-intervals, statistical-models, troubleshooting-common-errors). Before implementing or explaining a workflow that matches one of these, read the matching `SKILL.md` (and its `references/` if present).
- **Writing or updating tests** — read [.github/instructions/testing.instructions.md](.github/instructions/testing.instructions.md) before touching anything under `**/tests/**`.
- **Writing or updating docstrings** — read [.github/instructions/docstrings.instructions.md](.github/instructions/docstrings.instructions.md) before adding or editing NumPy-style docstrings on public APIs.
- **Deep API details not covered in AGENTS.md** — [docs/llms-full.txt](docs/llms-full.txt) is the fullest reference; consult it for parameter-level questions or features missing from the embedded reference.

## Files NOT to use as context

- `.github/copilot-instructions.md` — duplicate of `AGENTS.md` (auto-generated for Copilot).
- `.github/prompts/*` — Copilot review prompts, not general guidance.