---
paths:
  - "**/tests/**"
---

# Tests

Before writing or updating tests, read `.github/instructions/testing.instructions.md` (shared with Copilot).

## How much to run

The suite has more than 6000 tests; collecting all of them alone takes about 30 seconds, while a single test file usually runs in a few seconds. CI runs the full suite only on the pull request of each release to `main`, so on feature branches the tests you run are the safety net: run the right ones, once, and say what you did not cover.

1. While iterating: only the test files of the methods you changed, stopping at the first failure. After a fix, rerun only what failed.
   ```bash
   pytest skforecast/recursive/tests/tests_forecaster_recursive/test_predict.py -x -q
   pytest skforecast/recursive/tests/tests_forecaster_recursive/ -q --lf
   ```
2. Once, when the change is done (`verify` skill): the test folder of each module you touched (`skforecast/<pkg>/tests/tests_<module>/`).
3. Shared code (`skforecast/utils/`, `skforecast/base/`, `skforecast/model_selection/_utils.py`, `skforecast/preprocessing/`): also run the test folders of the callers whose behavior you changed, and list the packages you did not run in the report.
4. Never launch the full suite on your own. If you think it is needed (broad refactor, dependency change), say so and let the user decide.

## Rules

- Run tests sequentially. Never pass `-n` (pytest-xdist): it saturates the machine, and some tests in `model_selection/tests/tests_search` write output files that collide when run in parallel.
- Use `-q` and keep the default `--tb=short`. Add `-v` only to inspect specific test ids; per-test output of large runs floods the context.
- Avoid launching several heavy Python processes at once (for example three backtesting scripts in one response); run them one after another.
