# Unit tests for bayesian_search_foundation
# ==============================================================================
import os
import re
import pytest
import numpy as np
import pandas as pd
from copy import deepcopy
from functools import partialmethod
from unittest.mock import patch

import optuna
from tqdm import tqdm

from skforecast.model_selection import bayesian_search_foundation
from skforecast.model_selection._split import TimeSeriesFold, OneStepAheadFold
from skforecast.recursive import ForecasterRecursive
from sklearn.linear_model import Ridge

# Fixtures from the foundation test suite
from skforecast.foundation.tests.tests_forecaster_foundation.fixtures_forecaster_foundation import (
    FakePipeline,
    make_forecaster,
    y,
    series_wide,
    exog,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

PATCH_DEEPCOPY = "skforecast.model_selection._search.deepcopy_forecaster"


# ==============================================================================
# TypeError: wrong forecaster type
# ==============================================================================

def test_TypeError_bayesian_search_foundation_wrong_forecaster_type():
    """
    TypeError is raised when forecaster is not a ForecasterFoundation instance.
    """
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=2)
    cv = TimeSeriesFold(steps=5, initial_train_size=40)

    def search_space(trial):
        return {"context_length": trial.suggest_categorical("context_length", [24, 48])}

    err_msg = re.escape(
        "`forecaster` must be a `ForecasterFoundation` instance. "
        "Got ForecasterRecursive."
    )
    with pytest.raises(TypeError, match=err_msg):
        bayesian_search_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            n_trials=2,
            return_best=False,
        )


# ==============================================================================
# TypeError: wrong cv type
# ==============================================================================

def test_TypeError_bayesian_search_foundation_cv_is_OneStepAheadFold():
    """
    TypeError is raised when cv is OneStepAheadFold (not supported for foundation).
    """
    forecaster = make_forecaster()
    cv = OneStepAheadFold(initial_train_size=40)

    def search_space(trial):
        return {"context_length": trial.suggest_categorical("context_length", [24, 48])}

    with patch(PATCH_DEEPCOPY, side_effect=deepcopy):
        err_msg = re.escape(
            "`cv` must be a `TimeSeriesFold` instance. Got OneStepAheadFold. "
            "`OneStepAheadFold` is not supported for `ForecasterFoundation`."
        )
        with pytest.raises(TypeError, match=err_msg):
            bayesian_search_foundation(
                forecaster=forecaster,
                series=y,
                cv=cv,
                search_space=search_space,
                metric="mean_absolute_error",
                n_trials=2,
                return_best=False,
            )


# ==============================================================================
# ValueError: search_space key mismatch
# ==============================================================================

def test_ValueError_bayesian_search_foundation_search_space_key_mismatch():
    """
    ValueError is raised when search_space dict keys do not match trial.params keys.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(steps=5, initial_train_size=40)

    def search_space(trial):
        # Dict key 'context_length' but suggest name is 'ctx_len' -> mismatch
        return {"context_length": trial.suggest_categorical("ctx_len", [24, 48])}

    with patch(PATCH_DEEPCOPY, side_effect=deepcopy):
        err_msg = re.escape(
            "`search_space` dict keys must match the names passed to "
            "`trial.suggest_*()`."
        )
        with pytest.raises(ValueError, match=err_msg):
            bayesian_search_foundation(
                forecaster=forecaster,
                series=y,
                cv=cv,
                search_space=search_space,
                metric="mean_absolute_error",
                n_trials=2,
                return_best=False,
            )


# ==============================================================================
# Single-series: basic shape / column / sort check
# ==============================================================================

def test_bayesian_search_foundation_single_series():
    """
    Single pd.Series: results DataFrame has correct shape, columns, and is
    sorted ascending by the metric. study is an optuna Study.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(steps=5, initial_train_size=40)
    n_trials = 2

    def search_space(trial):
        return {
            "context_length": trial.suggest_categorical("context_length", [24, 48])
        }

    with patch(PATCH_DEEPCOPY, side_effect=deepcopy):
        results, study = bayesian_search_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            n_trials=n_trials,
            return_best=False,
            show_progress=False,
        )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == n_trials

    expected_cols = {"trial_number", "levels", "params", "mean_absolute_error", "context_length"}
    assert expected_cols.issubset(set(results.columns))

    assert "lags" not in results.columns

    # Sorted ascending for regression
    metric_vals = results["mean_absolute_error"].tolist()
    assert metric_vals == sorted(metric_vals)

    assert isinstance(study, optuna.Study)

    # levels column contains the resolved level name
    assert all(lv == ["y"] for lv in results["levels"])


# ==============================================================================
# Multi-series: aggregated metric column names
# ==============================================================================

def test_bayesian_search_foundation_multi_series_aggregated_columns():
    """
    Multi-series (series_wide): when add_aggregated_metric is True the results
    DataFrame contains aggregated metric columns (e.g. metric__weighted_average).
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(steps=5, initial_train_size=40)
    n_trials = 2

    def search_space(trial):
        return {
            "context_length": trial.suggest_categorical("context_length", [24, 48])
        }

    with patch(PATCH_DEEPCOPY, side_effect=deepcopy):
        results, study = bayesian_search_foundation(
            forecaster=forecaster,
            series=series_wide,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            n_trials=n_trials,
            return_best=False,
            show_progress=False,
        )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == n_trials

    # Aggregated metric columns must be present
    agg_cols = {
        "mean_absolute_error__weighted_average",
        "mean_absolute_error__average",
        "mean_absolute_error__pooling",
    }
    assert agg_cols.issubset(set(results.columns))

    # No bare 'mean_absolute_error' column (it was expanded to aggregated names)
    assert "mean_absolute_error" not in results.columns

    assert "lags" not in results.columns

    assert isinstance(study, optuna.Study)


# ==============================================================================
# return_best: original forecaster is refitted with the best params
# ==============================================================================

def test_bayesian_search_foundation_return_best():
    """
    When return_best=True the original forecaster is refitted with the best
    parameters found during the search.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(steps=5, initial_train_size=40)
    n_trials = 2

    def search_space(trial):
        return {
            "context_length": trial.suggest_categorical("context_length", [24, 48])
        }

    with patch(PATCH_DEEPCOPY, side_effect=deepcopy):
        results, _ = bayesian_search_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            n_trials=n_trials,
            return_best=True,
            show_progress=False,
        )

    best_context_length = results.loc[0, "context_length"]
    assert forecaster.context_length == best_context_length
    assert forecaster.is_fitted


# ==============================================================================
# Callable metric: column name matches function __name__
# ==============================================================================

def test_bayesian_search_foundation_callable_metric():
    """
    When a callable is passed as metric the results column name is the
    function's __name__ attribute.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(steps=5, initial_train_size=40)
    n_trials = 2

    def my_mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def search_space(trial):
        return {
            "context_length": trial.suggest_categorical("context_length", [24, 48])
        }

    with patch(PATCH_DEEPCOPY, side_effect=deepcopy):
        results, study = bayesian_search_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            search_space=search_space,
            metric=my_mae,
            n_trials=n_trials,
            return_best=False,
            show_progress=False,
        )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == n_trials
    assert "my_mae" in results.columns
    assert "lags" not in results.columns


# ==============================================================================
# Exog passthrough: search completes without error when exog is provided
# ==============================================================================

def test_bayesian_search_foundation_with_exog():
    """
    When exog is provided the search completes and returns valid results.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(steps=5, initial_train_size=40)
    n_trials = 2

    def search_space(trial):
        return {
            "context_length": trial.suggest_categorical("context_length", [24, 48])
        }

    with patch(PATCH_DEEPCOPY, side_effect=deepcopy):
        results, study = bayesian_search_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            exog=exog,
            n_trials=n_trials,
            return_best=False,
            show_progress=False,
        )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == n_trials
    assert "mean_absolute_error" in results.columns


# ==============================================================================
# List of metrics: all metric columns present in results
# ==============================================================================

def test_bayesian_search_foundation_list_of_metrics():
    """
    When metric is a list all metric names appear as columns in results and
    the DataFrame is sorted by the first metric.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(steps=5, initial_train_size=40)
    n_trials = 2

    def search_space(trial):
        return {
            "context_length": trial.suggest_categorical("context_length", [24, 48])
        }

    with patch(PATCH_DEEPCOPY, side_effect=deepcopy):
        results, _ = bayesian_search_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            search_space=search_space,
            metric=["mean_absolute_error", "mean_squared_error"],
            n_trials=n_trials,
            return_best=False,
            show_progress=False,
        )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == n_trials
    assert "mean_absolute_error" in results.columns
    assert "mean_squared_error" in results.columns
    assert "lags" not in results.columns

    mae_vals = results["mean_absolute_error"].tolist()
    assert mae_vals == sorted(mae_vals)


# ==============================================================================
# Custom aggregate_metric: only requested aggregations appear
# ==============================================================================

def test_bayesian_search_foundation_custom_aggregate_metric():
    """
    When aggregate_metric is a custom subset only those aggregations appear
    as columns; the others are absent.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(steps=5, initial_train_size=40)
    n_trials = 2

    def search_space(trial):
        return {
            "context_length": trial.suggest_categorical("context_length", [24, 48])
        }

    with patch(PATCH_DEEPCOPY, side_effect=deepcopy):
        results, _ = bayesian_search_foundation(
            forecaster=forecaster,
            series=series_wide,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            aggregate_metric=["average"],
            n_trials=n_trials,
            return_best=False,
            show_progress=False,
        )

    assert isinstance(results, pd.DataFrame)
    assert "mean_absolute_error__average" in results.columns
    assert "mean_absolute_error__weighted_average" not in results.columns
    assert "mean_absolute_error__pooling" not in results.columns
    assert "lags" not in results.columns


# ==============================================================================
# ValueError: invalid aggregate_metric value
# ==============================================================================

def test_ValueError_bayesian_search_foundation_invalid_aggregate_metric():
    """
    ValueError is raised when aggregate_metric contains an unsupported value.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(steps=5, initial_train_size=40)

    def search_space(trial):
        return {"context_length": trial.suggest_categorical("context_length", [24, 48])}

    allowed = ["average", "weighted_average", "pooling"]
    err_msg = re.escape(
        f"Allowed `aggregate_metric` are: {allowed}. Got: ['invalid_method']."
    )
    with pytest.raises(ValueError, match=err_msg):
        bayesian_search_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            aggregate_metric=["invalid_method"],
            n_trials=2,
            return_best=False,
        )


# ==============================================================================
# ValueError: duplicate metric names
# ==============================================================================

def test_ValueError_bayesian_search_foundation_duplicate_metric_names():
    """
    ValueError is raised when two metrics in the list resolve to the same name.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(steps=5, initial_train_size=40)

    def search_space(trial):
        return {"context_length": trial.suggest_categorical("context_length", [24, 48])}

    err_msg = re.escape(
        "When `metric` is a `list`, each metric name must be unique."
    )
    with pytest.raises(ValueError, match=err_msg):
        bayesian_search_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            search_space=search_space,
            metric=["mean_absolute_error", "mean_absolute_error"],
            n_trials=2,
            return_best=False,
        )


# ==============================================================================
# return_best with multi-series: forecaster refitted on multi-series data
# ==============================================================================

def test_bayesian_search_foundation_return_best_multi_series():
    """
    When return_best=True with multi-series input the original forecaster is
    refitted with the best parameters found during the search.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(steps=5, initial_train_size=40)
    n_trials = 2

    def search_space(trial):
        return {
            "context_length": trial.suggest_categorical("context_length", [24, 48])
        }

    with patch(PATCH_DEEPCOPY, side_effect=deepcopy):
        results, _ = bayesian_search_foundation(
            forecaster=forecaster,
            series=series_wide,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            n_trials=n_trials,
            return_best=True,
            show_progress=False,
        )

    best_context_length = results.loc[0, "context_length"]
    assert forecaster.context_length == best_context_length
    assert forecaster.is_fitted


# ==============================================================================
# suppress_warnings smoke test
# ==============================================================================

def test_bayesian_search_foundation_suppress_warnings():
    """
    suppress_warnings=True completes without error and returns valid results.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(steps=5, initial_train_size=40)
    n_trials = 2

    def search_space(trial):
        return {
            "context_length": trial.suggest_categorical("context_length", [24, 48])
        }

    with patch(PATCH_DEEPCOPY, side_effect=deepcopy):
        results, study = bayesian_search_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            n_trials=n_trials,
            return_best=False,
            show_progress=False,
            suppress_warnings=True,
        )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == n_trials
    assert isinstance(study, optuna.Study)


# ==============================================================================
# output_file: results file is created on disk
# ==============================================================================

def test_bayesian_search_foundation_output_file():
    """
    When output_file is provided the file is created on disk.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(steps=5, initial_train_size=40)
    n_trials = 2

    def search_space(trial):
        return {
            "context_length": trial.suggest_categorical("context_length", [24, 48])
        }

    output_file = "test_bayesian_search_foundation_output_file.txt"
    with patch(PATCH_DEEPCOPY, side_effect=deepcopy):
        _ = bayesian_search_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            n_trials=n_trials,
            return_best=False,
            show_progress=False,
            output_file=output_file,
        )

    assert os.path.isfile(output_file)
    os.remove(output_file)


# ==============================================================================
# kwargs_create_study: custom sampler is accepted
# ==============================================================================

def test_bayesian_search_foundation_kwargs_create_study():
    """
    When kwargs_create_study is provided it is forwarded to optuna.create_study.
    """
    from optuna.samplers import TPESampler

    forecaster = make_forecaster()
    cv = TimeSeriesFold(steps=5, initial_train_size=40)
    n_trials = 2

    def search_space(trial):
        return {
            "context_length": trial.suggest_categorical("context_length", [24, 48])
        }

    kwargs_create_study = {"sampler": TPESampler(seed=123, n_startup_trials=1)}
    with patch(PATCH_DEEPCOPY, side_effect=deepcopy):
        results, study = bayesian_search_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            n_trials=n_trials,
            return_best=False,
            show_progress=False,
            kwargs_create_study=kwargs_create_study,
        )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == n_trials
    assert isinstance(study, optuna.Study)


# ==============================================================================
# kwargs_study_optimize: gc_after_trial flag is accepted
# ==============================================================================

def test_bayesian_search_foundation_kwargs_study_optimize():
    """
    When kwargs_study_optimize is provided it is forwarded to study.optimize.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(steps=5, initial_train_size=40)
    n_trials = 2

    def search_space(trial):
        return {
            "context_length": trial.suggest_categorical("context_length", [24, 48])
        }

    kwargs_study_optimize = {"gc_after_trial": True}
    with patch(PATCH_DEEPCOPY, side_effect=deepcopy):
        results, study = bayesian_search_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            n_trials=n_trials,
            return_best=False,
            show_progress=False,
            kwargs_study_optimize=kwargs_study_optimize,
        )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == n_trials
    assert isinstance(study, optuna.Study)


# ==============================================================================
# levels filtering: only requested series appear in results
# ==============================================================================

def test_bayesian_search_foundation_levels_filtering():
    """
    When levels is provided only those series appear in the results levels column.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(steps=5, initial_train_size=40)
    n_trials = 2

    def search_space(trial):
        return {
            "context_length": trial.suggest_categorical("context_length", [24, 48])
        }

    with patch(PATCH_DEEPCOPY, side_effect=deepcopy):
        results, _ = bayesian_search_foundation(
            forecaster=forecaster,
            series=series_wide,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            levels=["series_1"],
            n_trials=n_trials,
            return_best=False,
            show_progress=False,
        )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == n_trials
    assert all(lv == ["series_1"] for lv in results["levels"])


# ==============================================================================
# verbose=True with return_best: print is produced and forecaster is fitted
# ==============================================================================

def test_bayesian_search_foundation_verbose_return_best(capsys):
    """
    When verbose=True and return_best=True the best-found parameters are printed
    and the original forecaster is refitted.
    """
    forecaster = make_forecaster()
    cv = TimeSeriesFold(steps=5, initial_train_size=40)
    n_trials = 2

    def search_space(trial):
        return {
            "context_length": trial.suggest_categorical("context_length", [24, 48])
        }

    with patch(PATCH_DEEPCOPY, side_effect=deepcopy):
        results, _ = bayesian_search_foundation(
            forecaster=forecaster,
            series=y,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            n_trials=n_trials,
            return_best=True,
            show_progress=False,
            verbose=True,
        )

    captured = capsys.readouterr()
    assert "best-found parameters" in captured.out
    assert forecaster.is_fitted
