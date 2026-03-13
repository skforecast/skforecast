---
description: 'Use when writing, updating, or reviewing tests for skforecast. Covers test structure, naming, fixtures, parametrize patterns, assertion style, grouping, and expected-value conventions for the skforecast package.'
applyTo: '**/tests/**'
---
# Skforecast Testing Guidelines

## File Organization

- One test file per public method or logical unit: `test_<method_name>.py`.
- Each test directory has an `__init__.py` (empty).
- File header comment: `# Unit test <method_name> <ForecasterClass>` (or just `# Unit test <function_name>` for utility/standalone modules).
- No `conftest.py` — the only exception is `plot/` tests which use `conftest.py` for the matplotlib `Agg` backend.

### Directory nesting

- **Forecaster modules** (`recursive/`, `direct/`): `tests/tests_<forecaster>/` with fixtures in the same directory.
- **model_selection**: `tests/` holds shared fixtures at the top level; subdirectories (`tests_search/`, `tests_validation/`, `tests_split/`) import them with `..fixtures_model_selection`.
- **Utility/standalone modules** (`utils/`, `preprocessing/`, `metrics/`, `drift_detection/`): `tests/tests_<module>/` with optional `fixtures_<module>.py`.

## Fixtures

- Fixtures are **module-level variables** (not `@pytest.fixture`) defined in a separate `fixtures_*.py` file.
- Import them explicitly with relative imports:
  ```python
  from .fixtures_forecaster_recursive import y, exog, exog_predict, data
  # For model_selection subdirectories:
  from ..fixtures_model_selection import y, exog
  ```
- Fixture data is generated from a **fixed seed** then stored as hardcoded `np.array` inside the fixtures file. The original random code is kept as a comment for reproducibility.
- Common fixture variables: `y` (pd.Series), `exog` (pd.Series), `exog_predict` (pd.Series with shifted index), `data` (pd.Series with DatetimeIndex).
- For simple preprocessor/utility tests, small fixture arrays can be defined as module-level variables directly in the test file when a separate fixtures file is unnecessary.

## Imports

Standard import order at the top of every test file:
```python
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# ... other sklearn imports as needed
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive

# Fixtures
from .fixtures_forecaster_recursive import y, exog
```

## Naming Conventions

- Test functions: `test_<method>_<scenario>` — descriptive, snake_case.
  - Error tests: `test_<method>_<ErrorType>_when_<condition>`.
  - Output tests: `test_<method>_output_when_<condition>`.
  - Attribute tests: `test_<method>_<attribute>_correctly_stored`.
- Always include a **docstring** explaining what the test does.
- Docstrings use multi-line format with the text starting on the line after the opening `"""` and the closing `"""` on its own line:
  ```python
  def test_example():
      """
      Test that the function returns the expected value.
      """
  ```
- Parametrize ids use `lambda` for readable test names:
  ```python
  ids=lambda dt: f'lags, window_features, expected: {dt}'
  ```

## Parametrize & Grouping

Minimize total test count by **grouping related checks** and **parametrizing** variations. When multiple input types exercise the same code branch, combine them in a single parametrized test rather than creating one test function per type.

### Parametrize for variations of the same logic
```python
@pytest.mark.parametrize(
    'lags', 
    [3, [1, 2, 3], np.array([1, 2, 3]), range(1, 4)],
    ids=lambda lags: f'lags: {lags}'
)
def test_set_lags_with_different_inputs(lags):
    ...
```

### Parametrize for multiple error-raising inputs
```python
@pytest.mark.parametrize(
    'dif', 
    [0, 0.5, 1.5, 'not_int'],
    ids=lambda dif: f'differentiation: {dif}'
)
def test_init_ValueError_when_differentiation_not_valid(dif):
    ...
```

### Group multiple assertions in a single test
When multiple assertions verify facets of the same logical behavior or scenario, group them in a single test function — not only when checking object attributes:
```python
def test_init_window_size_correctly_stored(lags, window_features, expected):
    ...
    assert forecaster.window_size == expected
    if lags:
        np.testing.assert_array_almost_equal(forecaster.lags, ...)
        assert forecaster.lags_names == [...]
        assert forecaster.max_lag == lags
    else:
        assert forecaster.lags is None
```

### Parametrize the forecaster configuration for "does not modify input" tests
```python
@pytest.mark.parametrize(
    'forecaster_kwargs',
    [
        {'estimator': LinearRegression(), 'lags': 5},
        {'estimator': LinearRegression(), 'lags': 5,
         'window_features': RollingFeatures(stats=['mean'], window_sizes=3)},
        {'estimator': LinearRegression(), 'lags': 5,
         'window_features': RollingFeatures(stats=['mean'], window_sizes=3),
         'transformer_y': StandardScaler(), 'transformer_exog': StandardScaler()},
        {'estimator': LinearRegression(), 'lags': 5,
         'window_features': RollingFeatures(stats=['mean'], window_sizes=3),
         'transformer_y': StandardScaler(), 'transformer_exog': StandardScaler(),
         'differentiation': 1},
    ],
    ids=['base', 'window_features', 'transformers', 'differentiation']
)
def test_method_does_not_modify_y_exog(forecaster_kwargs):
    ...
```

## Assertions

### Comparing pandas objects (preferred)
```python
pd.testing.assert_frame_equal(results, expected)
pd.testing.assert_series_equal(results, expected)
```

### Comparing numpy arrays
```python
np.testing.assert_array_almost_equal(results, expected)
np.testing.assert_array_equal(results, expected)
```

### Scalar and simple comparisons
```python
assert results == expected
assert forecaster.attribute is None
assert isinstance(results, np.ndarray)
```

### Scalar float comparisons
```python
assert np.isclose(result, expected_value)
```

### Never use `pytest.approx` for array comparisons — use `np.testing.assert_array_almost_equal`.

## Expected Values

- **Hardcoded expected values**: Always compare results against a pre-computed expected value, not a dynamically computed one. The expected values should be hardcoded `np.array([...])` or `pd.Series(...)` / `pd.DataFrame(...)`.
- Build the full expected object (DataFrame, Series, array) with correct `index`, `columns`, `name`, and `dtype` to match the output exactly.
- For tuple outputs (like `_create_train_X_y`), check each element separately:
  ```python
  pd.testing.assert_frame_equal(results[0], expected[0])
  pd.testing.assert_series_equal(results[1], expected[1])
  assert isinstance(results[2], type(None))
  ```

## Testing Errors and Warnings

- Use `re.escape()` for the expected error message, and `pytest.raises` with `match`:
  ```python
  err_msg = re.escape('Exact error message text here')
  with pytest.raises(ValueError, match=err_msg):
      ...
  ```
- For warnings use `pytest.warns`:
  ```python
  warn_msg = re.escape('Exact warning message text here')
  with pytest.warns(UserWarning, match=warn_msg):
      ...
  ```
- Custom skforecast warnings are imported from `skforecast.exceptions` (e.g., `MissingValuesWarning`).
- When the result inside a warning block matters, capture it:
  ```python
  with pytest.warns(UserWarning, match=warn_msg):
      results = forecaster.get_feature_importances()
      assert results is expected
  ```

## Testing "Does Not Modify Input"

A common and important pattern: verify that `fit`/`predict` does not mutate the input data.
```python
y_local = y.copy()
exog_local = exog.copy()
y_copy = y_local.copy()
exog_copy = exog_local.copy()

forecaster.fit(y=y_local, exog=exog_local)

pd.testing.assert_series_equal(y_local, y_copy)
pd.testing.assert_series_equal(exog_local, exog_copy)
```

## Test Progression Within a File

Tests in a file follow this order:
1. **Error/validation tests** — ValueError, TypeError, NotFittedError, warnings.
2. **Basic output tests** — simplest scenario (no exog, no transformers).
3. **Feature-rich tests** — with exog, transformers, window_features, differentiation.
4. **Does-not-modify-input tests** (parametrized across configurations).
5. **Edge cases** — special dtypes, pipelines, custom weight functions.

## Module-Specific Patterns

### model_selection tests
- Disable tqdm progress bars at module level:
  ```python
  from tqdm import tqdm
  from functools import partialmethod
  tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
  ```
- Mark long-running tests (e.g., Bayesian search) with `@pytest.mark.slow`.
- Fixtures live in `tests/fixtures_model_selection.py`; subdirectories import with `..fixtures_model_selection`.

### preprocessing / utils tests
- For simple transformer/utility functions, small fixture arrays can be defined as module-level variables directly in the test file (e.g., `y`, `y_diff_1` in `test_TimeSeriesDifferentiator.py`).
- Group all validation errors for `__init__` / `_validate_params` into a single test function using a dict of param sets.

### plot tests
- Only module using `conftest.py` — sets matplotlib `Agg` backend via `@pytest.fixture(autouse=True)`.

## Common Estimators in Tests

- `LinearRegression()` — default estimator for most tests (deterministic, simple).
- `RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123)` — for feature importance tests.
- `LGBMRegressor(verbose=-1, random_state=123)` — for window features and categorical tests.
- `HistGradientBoostingRegressor()` — for categorical native support tests.
- `make_pipeline(StandardScaler(), LinearRegression())` — for pipeline tests.
- `object()` — when the estimator is not used (e.g., testing init, binner).

## Style

- PEP 8 compliant, max line length 88.
- Keyword arguments aligned vertically for readability in forecaster instantiation:
  ```python
  forecaster = ForecasterRecursive(
                   estimator        = LinearRegression(),
                   lags             = 5,
                   transformer_y    = transformer_y,
                   transformer_exog = transformer_exog,
               )
  ```
- NumPy arrays use aligned formatting when multiline.
