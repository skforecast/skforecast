# Skforecast: Development Context

## For Contributors Working Inside This Repository

### Testing

```bash
pytest path/to/test_file.py -x -q                 # Iterate on the touched test file
pytest path/to/tests/ -q --lf                     # Rerun only the last failures
pytest skforecast/<pkg>/tests/tests_<module>/ -q  # Touched module, once when done
```

Do not run the full suite (6000+ tests) unless the user asks for it; it runs in
CI on the pull request of each release to `main`. Run tests sequentially: do not
use `-n` (pytest-xdist), it saturates the machine and some search tests write
output files that collide in parallel runs.

Markers: `@pytest.mark.slow` for long-running tests (skip with `-m "not slow"`).

### Code Style

- NumPy-style docstrings
- Type hints for function signatures
- PEP 8 compliant (max line length 88, enforced by ruff)
- Double quotes for strings (ruff `quote-style = "double"`)
- Relative imports within package
- When generating code comments, docstrings, and documentation, do not use en dashes (–), or em dashes (—). Use commas, colons, semicolons, or parentheses for punctuation instead.

### Commits and pull requests

Commits and pull requests carry only the author's identity: no
`Co-Authored-By` trailer for an AI agent, no session link trailer, no
"Generated with" line and no mention of the AI assistant in the message. The
author identity comes from git config or `GIT_AUTHOR_*` and `GIT_COMMITTER_*`;
do not override it.

### Dependencies

Core: numpy>=1.26, pandas>=2.2,<3.0, scikit-learn>=1.6, scipy>=1.12, optuna>=4.0, joblib>=1.3, numba>=0.59, tqdm>=4.66, rich>=13.9
Optional: statsmodels>=0.13.2,<0.15 (stats), matplotlib>=3.7,<3.12 (plotting), keras>=3.0,<4.0 (deep learning)

### Python environment

Local machines: environments are managed with conda. Run every Python command
(tests, scripts, notebooks, `pip install`, etc.) in the conda environment that
is currently active. Do not run `conda env list` to ask which environment to
use, and do not use the `.venv` directory at the repository root. If the shell
does not inherit the active environment (`$CONDA_DEFAULT_ENV` is empty), source
your shell profile first, or call the interpreter through `conda run -n <env>`.

Cloud sessions (`CLAUDE_CODE_REMOTE=true`, e.g. claude.ai/code): there is no
conda. A SessionStart hook installs the package with the `test` extras into a
virtual environment outside the repository and puts it on `PATH`, so call
`python` and `pytest` directly. To make the install faster, it skips the deep
learning packages (`torch`, `keras`): the deep learning tests do not need to
pass unless you work on that code. When they are needed (e.g. `ForecasterRnn`
or foundation models), set `SKFORECAST_CLOUD_DL=1` in the cloud environment
variables, or install them in the session with
`uv pip install torch "keras>=3.0,<4.0" --torch-backend cpu` (`uv` is in
`~/.local/bin` if it is not on `PATH`).
