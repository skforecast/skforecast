---
description: 'Use when writing, updating, or reviewing docstrings in skforecast source code. Covers NumPy-style format, section order, parameter/return formatting, type annotations, deprecation notices, version tags, and cross-reference conventions.'
applyTo: 'skforecast/**/*.py'
---
# Skforecast Docstring Guidelines

## Format

NumPy-style docstrings. Every public class and public method/function must have a docstring.

## Before Modifying a Docstring

Before writing or modifying a docstring, **read the existing docstrings** in the same file and the neighboring parameters to match the established style exactly. Do not reformat existing content that you are not changing.

## Common Mistakes — Do NOT

These are the most frequent errors. Violating any of these rules is always wrong in skforecast.

| Wrong | Correct | Why |
|-------|---------|-----|
| ` ``True`` ` (double backticks) | `` `True` `` (single backticks) | Skforecast never uses rST double-backtick literals |
| `pd.Series` | `pandas Series` | Docstring types use readable names, not aliases |
| `pd.DataFrame` | `pandas DataFrame` | Same as above |
| `np.ndarray` | `numpy ndarray` | Same as above |
| `:class:\`OrdinalEncoder\`` | `OrdinalEncoder` | No rST cross-reference directives |
| Adding `Raises` section | *(omit it)* | Skforecast docstrings never include Raises |
| Adding `Warnings` section | *(omit it)* | Skforecast docstrings never include Warnings |
| Adding `See Also` section | *(omit it)* | Skforecast docstrings never include See Also |
| Adding `Yields` section | *(omit it)* | Skforecast docstrings never include Yields |

**Additional rules:**
- NEVER use double backticks (`` `` ``) for any inline code or value. Always single backticks.
- NEVER add docstrings, comments, or type annotations to code you did not change.
- NEVER invent parameters or attributes that do not exist in the actual code.

## Section Order

Follow this exact section order (omit sections that don't apply):

1. **Summary** — one-line or short paragraph
2. **Parameters** — constructor or function arguments
3. **Attributes** — class-level only (after Parameters in classes)
4. **Returns** — what the method/function returns
5. **Notes** — implementation details, caveats, behavioral notes
6. **References** — numbered references using `.. [1]` syntax

## Summary

- First line: concise description of the class/method purpose.
- Separated from sections by a blank line.
- For classes: describe what the class does, not how to use it.
- For methods: describe what the method does, starting with a verb (e.g., "Training Forecaster.", "Predict n steps ahead.").

## Parameters Section

```python
Parameters
----------
y : pandas Series
    Training time series.
exog : pandas Series, pandas DataFrame, default None
    Exogenous variable/s included as predictor/s. Must have the same
    number of observations as `y` and their indexes must be aligned.
steps : int, str, pandas Timestamp
    Number of steps to predict. 

    - If steps is int, number of steps to predict. 
    - If str or pandas Datetime, the prediction will be up to that date.
```

### Rules

- **Type line format**: `name : type[, type[, ...]][, default value]`
- **Default values**: written as `default None`, `default True`, `default 123`, `default 'auto'` — always on the type line, not in the description.
- **Description indentation**: 4 spaces from the left margin (one level deeper than the parameter name).
- **Sub-items** (enumerated options): insert a blank line between the description and the first bullet. Bullets use the same indentation as the description text. Continuation lines for a bullet align with the dash (`-`), **not** indented further to align with the text after the dash. No blank lines between consecutive bullets.

  Correct example (inside a class docstring, 4-space base indent from `"""`):
  ```
      encoding : str, None, default 'ordinal'
          Encoding used to identify the different series.
  ​
          - If `'ordinal'`, a single column is created with integer values from 0
          to n_series - 1.
          - If `'onehot'`, a binary column is created for each series.
          - If None, no column is created to identify the series. Internally, the
          series are identified as an integer from 0 to n_series - 1, but no column
          is created in the training matrices.
  ```
  Notice: continuation line `to n_series - 1.` starts at the same column as the `-` dash, not at the column of the text after `- `.

- **Backticks**: always single backticks — never double. Use for parameter names, values, and attribute references (`y`, `None`, `self.last_window_`, `True`, `False`).
- **Multi-line descriptions**: continuation lines align with the first line of the description (same indent level as description start).
- **Type naming conventions** (critical — these are the most common source of errors):
  - `pandas Series`, `pandas DataFrame` — NEVER `pd.Series` or `pd.DataFrame`
  - `numpy ndarray` — NEVER `np.ndarray`
  - `str`, `int`, `float`, `bool`, `dict`, `list`, `tuple`, `Callable`, `object`
  - Union types separated by commas: `int, list, numpy ndarray, range`
  - For complex union types: `str | Callable | list[str | Callable]` in the signature, but `str, Callable, list` in the docstring

## Attributes Section

Only in class docstrings, after Parameters.

```python
Attributes
----------
lags : numpy ndarray
    Lags used as predictors.
is_fitted : bool
    Tag to identify if the estimator has been fitted (trained).
```

### Rules

- Same format as Parameters but **without default values**.
- Include all public attributes the user might inspect after fitting.
- Private attributes (prefixed `_`) are included only if they are part of the API (e.g., `_probabilistic_mode`).
- Trailing-underscore attributes (e.g., `last_window_`, `in_sample_residuals_`) are sklearn-convention fitted attributes — always document them.

## Returns Section

```python
Returns
-------
predictions : pandas Series
    Predicted values.
```

Or for multiple returns:

```python
Returns
-------
X_train : pandas DataFrame
    Training values (predictors).
y_train : pandas Series
    Values of the time series related to each row of `X_train`.
```

### Rules

- Format: `name : type` followed by indented description.
- For `None` returns: `Returns\n-------\nNone`
- For tuple returns, document each element separately (don't write `tuple`).
- For complex return structures (DataFrame with specific columns), describe the columns in the description body:
  ```
  predictions : pandas DataFrame
      Values predicted by the forecaster and their estimated interval.

      - pred: predictions.
      - lower_bound: lower bound of the interval.
      - upper_bound: upper bound of the interval.
  ```

## Notes Section

Use for behavioral details, caveats, or relationships between parameters:

```python
Notes
-----
Note on `fold_stride` vs. `steps`:

- If `fold_stride == steps`, test sets are placed back-to-back without overlap. 
```

## References Section

Use numbered RST references:

```python
References
----------
.. [1] Forecasting: Principles and Practice (3rd ed) Rob J Hyndman and George Athanasopoulos.
       https://otexts.com/fpp3/prediction-intervals.html

.. [2] MAPIE - Model Agnostic Prediction Interval Estimator.
       https://mapie.readthedocs.io/en/stable/
```

Reference in text with `[1]_`.

## Cross-References to Other Classes

- **In the type position** (after `:` on a parameter/attribute line): use plain text, no backticks.
  ```
  differentiator : TimeSeriesDifferentiator
  categorical_encoder : sklearn OrdinalEncoder
  ```
- **In description body**: use single backticks for the class/object name.
  ```
      `OrdinalEncoder` used internally to encode categorical features.
  ```
- **Never** use rST cross-reference directives (`:class:`, `:func:`, `:meth:`, `:ref:`). Use plain single backticks only.

## Version and Deprecation Tags

- **New parameters**: add `**New in version X.Y.Z**` as the last line of the parameter description, indented at the same level:
  ```
  binner_kwargs : dict, default None
      Additional arguments to pass to the `QuantileBinner`.
      **New in version 0.14.0**
  ```

- **Deprecated parameters**: add as a separate parameter entry at the end of Parameters:
  ```
  regressor : estimator or pipeline compatible with the scikit-learn API
      **Deprecated**, alias for `estimator`.
  ```

## Type Hints (Signatures)

Function signatures use Python type hints (`|` union syntax, `list[...]`, `dict[...]`). These are **separate** from docstring types:

```python
def fit(
    self,
    y: pd.Series,
    exog: pd.Series | pd.DataFrame | None = None,
    store_last_window: bool = True,
) -> None:
```

- Use `|` for unions in signatures (not `Union[]`).
- Signature types are more precise (`pd.Series`); docstring types use readable names (`pandas Series`).
