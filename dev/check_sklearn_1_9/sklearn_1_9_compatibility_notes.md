# scikit-learn 1.9 compatibility notes for skforecast

Reference: https://scikit-learn.org/stable/whats_new/v1.9.html (released June 2026).

This document summarizes the scikit-learn 1.9.0 changes that are relevant to
skforecast and lists the concrete points to verify on the skforecast side. It
is a maintenance aid, not user-facing documentation. The automated smoke test
lives in `dev/check_sklearn_compatibility.py`.

Context: skforecast pins `scikit-learn>=1.4` (`pyproject.toml`) and wraps
scikit-learn regressors/classifiers, relying on `clone`, estimator tags,
`Pipeline`, `ColumnTransformer`, `OrdinalEncoder`/`OneHotEncoder`, `RFE`/`RFECV`,
and the scorer/metric machinery.

---

## 1. Main scikit-learn 1.9 changes relevant to skforecast

### 1.1 Dependencies and dataframe handling

- **New `narwhals` dependency.** scikit-learn now uses narwhals internally to
  support dataframe input/output in the `set_output` API. The dataframe
  interchange protocol (`__dataframe__`), previously used for non-pandas
  dataframes, is being retired (deprecated upstream by polars).
- **New config key `sparse_interface`.** `set_config(sparse_interface="sparray")`
  makes sklearn return SciPy sparse arrays instead of sparse matrices. Default
  is still `"spmatrix"`, but will switch to `"sparray"` in a few releases.
- **`utils.check_array` now rejects pandas `StringDtype` when `dtype="numeric"`.**
  Under pandas 3, string columns use `StringDtype` (not `object`), which
  previously slipped through and now raises `ValueError`.

### 1.2 New deprecations (API changes)

- **`criterion="friedman_mse"` deprecated in trees** (`DecisionTreeRegressor`,
  `ExtraTreeRegressor`, `RandomForestRegressor`, `ExtraTreesRegressor`); use
  `"squared_error"`. Also the `criterion` parameter itself is deprecated on
  `GradientBoostingRegressor` / `GradientBoostingClassifier`.
- **`TargetEncoder`: `shuffle` and `random_state` deprecated** (removed in 1.11);
  pass a CV generator via `cv` instead.
- **`SVC` / `NuSVC`: `probability` parameter deprecated** (removed in 1.11; not
  thread-safe). Use `CalibratedClassifierCV(..., ensemble=False)`. `probA_` /
  `probB_` attributes also deprecated.
- **`log_loss` / `d2_log_loss_score`: `y_pred` deprecated in favor of `y_proba`.**
- **`lasso_path` / `enet_path`: `n_alphas` deprecated.**
- **`LogisticRegressionCV`: default `scoring` will change** from accuracy to
  `"neg_log_loss"` in 1.11; positional `sample_weight` in `.score()` deprecated.
- **Positional args deprecated** in `confusion_matrix_at_thresholds`
  (`pos_label`, `sample_weight`).

### 1.3 Behavioural / numeric changes ("changed models")

- **`sample_weight` all-zeros now raises `ValueError`** across all estimators and
  weight-validating metrics.
- **RandomForest / ExtraTrees `sample_weight` semantics changed:** weights are now
  used to draw samples; a float `max_samples` is a fraction of
  `sample_weight.sum()` (not `X.shape[0]`), and float `max_samples > 1.0` /
  integer `> n_samples` are now allowed. Fitted models may differ.
- **GradientBoosting `"friedman_mse"` impurity scaling fixed** (now uses
  `"squared_error"`); trees may differ in edge cases.
- **`LogisticRegression(solver="lbfgs")` computes the gradient at float32** when
  fit on float32 data (previously implicitly upcast to float64). Cast to float64
  to reproduce old numerics.
- **`SGDOneClassSVM` alpha formulation corrected** (`alpha = nu`); `coef_`,
  `offset_`, and predictions may change. `SGDClassifier` NaN fix in multiclass.
- **`RidgeCV`: `auto` is now equivalent to `eigen`** and LOO errors are numerically
  stable in the small-alpha regime.
- **`PowerTransformer(method="yeo-johnson")`** now uses `scipy.stats.yeojohnson`;
  minor numeric deviations.

### 1.4 Determinism / robustness (mostly positive)

- **`RFE` now uses stable sorting** when ranking feature importances, so feature
  selection is deterministic across runs when importances are tied (selected
  features in tied cases may differ from earlier versions).
- **`SelectFromModel` / `RFE` support sparse feature importances** via
  `importance_getter`.
- **`GroupKFold` uses stable sorting**; `StratifiedGroupKFold` raises when
  `n_splits > n_groups`.
- **`Pipeline` / `FeatureUnion` / `ColumnTransformer`** raise clearer errors when
  an estimator class (not instance) is passed; several HTML-repr improvements.
- **`utils.get_tags`** gives a clearer error when passed a class instead of an
  instance.

### 1.5 Low relevance to skforecast

- Experimental callback API (`ProgressBar`, `ScoringMonitor`).
- Extended Array API support across estimators/metrics.
- Metadata routing fixes.
- Many module-specific fixes (cluster, manifold, inspection, neighbors, etc.).

---

## 2. Points to check on the skforecast side

### 2.1 Core wrapping and cloning

- [ ] `clone`, `get_params`, `set_params` on all forecasters and wrapped
      estimators (covered by `check_sklearn_compatibility.py`).
- [ ] Estimator tags path: skforecast uses `__sklearn_tags__` with a legacy
      `_get_tags` fallback. Confirm no class-vs-instance misuse now that
      `utils.get_tags` raises on classes.

### 2.2 Transformers, exog and categoricals

- [ ] `transformer_y` / `transformer_series` / `transformer_exog` with
      `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`,
      `OrdinalEncoder`.
- [ ] `OneHotEncoder` with `set_output(transform="pandas")` requires
      `sparse_output=False` (relevant given the `sparse_interface` config).
- [ ] Built-in `categorical_features="auto"` (internal `OrdinalEncoder`) across
      LightGBM / XGBoost / CatBoost / HistGradientBoosting.
- [ ] String exog columns under pandas 3 (`StringDtype`): confirm skforecast
      coerces/encodes before any `check_array(dtype="numeric")` path.

### 2.3 Feature selection

- [ ] `select_features` / `select_features_multiseries` with `RFE` / `RFECV`.
      Stable sorting may change which lags/exog are selected in tied cases;
      review any tests asserting exact selected-feature sets.

### 2.4 Metrics and probabilistic outputs

- [ ] skforecast metrics that mirror sklearn (pinball / CRPS / coverage): verify
      no reliance on the deprecated `log_loss(y_pred=...)` signature and check
      the `d2_pinball_score` / `d2_absolute_error_score` quantile-method change
      if used anywhere.
- [ ] `predict_interval` / `predict_quantiles` bootstrapping and conformal paths
      (remember `store_in_sample_residuals=True` for conformal at fit time).

### 2.5 Reproducibility of expected values in tests

- [ ] RandomForest / ExtraTrees: `sample_weight` + `max_samples` semantics
      changed. Recheck any hard-coded expected predictions in tests that use
      these estimators with `weight_func` or `max_samples`.
- [ ] GradientBoosting / tree estimators: avoid `criterion="friedman_mse"` in
      examples, tests, and fixtures; switch to `"squared_error"`.
- [ ] LogisticRegression on float32 input: expected-value drift in classifier
      forecaster tests.

### 2.6 Weights

- [ ] `weight_func` / `series_weights`: ensure skforecast never forwards an
      all-zero `sample_weight` (now a hard `ValueError`).

### 2.7 Dependency surface

- [ ] Confirm `narwhals` is pulled in transitively by scikit-learn (no direct
      pin needed) and does not conflict with skforecast's environment.
- [ ] No skforecast code relies on the deprecated `__dataframe__` interchange
      protocol.

---

## 3. Status

Smoke test `dev/check_sklearn_compatibility.py` run against scikit-learn 1.9.0
(Python 3.14, numpy 2.4.6, pandas 2.3.3): 12/12 checks pass, no scikit-learn
deprecation warnings emitted. Remaining items above are review/audit checks for
tests, fixtures, and examples rather than confirmed breakages.
