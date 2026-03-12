---
description: 'Use when writing, updating, or reviewing docstrings in skforecast source code. Covers NumPy-style format, section order, parameter/return formatting, type annotations, deprecation notices, version tags, and cross-reference conventions.'
applyTo: 'skforecast/**/*.py'
---
# Skforecast Docstring Guidelines

## Format

NumPy-style docstrings. Every public class and public method/function must have a docstring.

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
- **Sub-items** (enumerated options): use a blank line before the list, then `- If \`value\`: description` with 4-space indentation, matching the description indentation.
- **Backticks**: use single backticks for parameter names, values, and attribute references (`` `y` ``, `` `None` ``, `` `self.last_window_` ``).
- **Multi-line descriptions**: continuation lines align with the first line of the description (4-space indent).
- **Type naming conventions**:
  - `pandas Series`, `pandas DataFrame` (not `pd.Series`)
  - `numpy ndarray` (not `np.ndarray`)
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
