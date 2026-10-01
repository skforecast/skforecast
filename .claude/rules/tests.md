---
paths:
  - "**/tests/**"
---

# Tests

Before writing or updating tests, read `.github/instructions/testing.instructions.md` (shared with Copilot).

## How much to run

The suite has more than 6000 tests; collecting all of them alone takes about 30 seconds, while a single test file usually runs in a few seconds. Never run the whole suite locally: CI runs it on every pull request to `main` and to release branches (`*.x`), on 3 operating systems and 5 Python versions.

1. While iterating: only the test files of the code you changed, stopping at the first failure. After a fix, rerun only what failed.
   ```bash
   pytest skforecast/recursive/tests/tests_forecaster_recursive/test_predict.py -x -q
   pytest skforecast/recursive/tests/tests_forecaster_recursive/ -q --lf
   ```
2. Once, before opening a pull request: the test package of each subpackage you touched (e.g. `pytest skforecast/recursive/tests -q`).
3. To run the full suite on a branch without opening a PR, trigger CI instead: `gh workflow run unit-tests.yml --ref <branch>`.

## Rules

- Run tests sequentially. Never pass `-n` (pytest-xdist): it saturates the machine, and some tests in `model_selection/tests/tests_search` write output files that collide when run in parallel.
- Use `-q` and keep the default `--tb=short`. Add `-v` only to inspect specific test ids; per-test output of large runs floods the context.
- Avoid launching several heavy Python processes at once (for example three backtesting scripts in one response); run them one after another.
