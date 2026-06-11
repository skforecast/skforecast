# Skforecast — Development Context

## For Contributors Working Inside This Repository

### Testing

```bash
pytest skforecast/recursive/tests/ -vv           # Run a specific module's tests
pytest --cov=skforecast --cov-report=html         # Coverage report
pytest -n auto                                    # Parallel execution (pytest-xdist)
```

Markers: `@pytest.mark.slow` for long-running tests (skip with `-m "not slow"`).

### Code Style

- NumPy-style docstrings
- Type hints for function signatures
- PEP 8 compliant (max line length 88, enforced by ruff)
- Double quotes for strings (ruff `quote-style = "double"`)
- Relative imports within package

### Dependencies

Core: numpy>=1.26, pandas>=2.1,<3.0, scikit-learn>=1.4, scipy>=1.12, optuna>=4.0, joblib>=1.3, numba>=0.59, tqdm>=4.66, rich>=13.9
Optional: statsmodels>=0.13,<0.15 (stats), matplotlib>=3.7,<3.11 + seaborn>=0.12,<0.14 (plotting), keras>=3.0,<4.0 (deep learning)

### Python environment

Before running any Python command (tests, scripts, notebooks, `pip install`, etc.)
for the first time in a session, run `conda env list` and ask which environment to
use. Do not assume the active environment. Once the user confirms an environment,
reuse it for the rest of the session without asking again.

---

# Skforecast — Complete API & Workflow Reference

(The content below is the full `llms-base.txt` and applies to any user of skforecast)
