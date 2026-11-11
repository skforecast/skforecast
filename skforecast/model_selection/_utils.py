################################################################################
#                     skforecast.model_selection._utils                        #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################


from __future__ import annotations
from typing import Callable, Generator, Any
import logging
import re
import sys
import threading
import warnings
from itertools import chain
import numpy as np
import pandas as pd
from joblib import cpu_count
from sklearn.exceptions import NotFittedError
from sklearn.linear_model._base import LinearModel
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from ..exceptions import (
    IgnoredArgumentWarning,
    LongTrainingWarning,
    MissingValuesWarning,
    OneStepAheadValidationWarning
)
from ..metrics import (
    add_y_train_argument,
    _any_metric_needs_y_train,
    _get_metric,
    calculate_coverage
)
from ..metrics.metrics import (
    _metric_greater_is_better,
    _naive_scales,
    _scaled_metric_per_origin,
    _is_scaled_metric
)
from ..utils import (
    _date_range_from_index,
    check_interval,
    date_to_index_position,
    cast_catboost_categorical_columns_dataframe,
    _copy_rows_to_check,
    _check_in_place_fit,
    align_series_and_exog_multiseries,
    _series_too_short_multiseries,
    estimator_has_native_nan_support,
    _first_frame_outside_skforecast,
    _find_stack_level,
    transform_numpy
)

# NOTE: Names of the metrics added to the backtesting metrics when the
# predictions include `lower_bound` and `upper_bound`, see
# `_calculate_interval_metrics`.
INTERVAL_METRIC_NAMES = ['coverage', 'interval_width']

# NOTE: Name of the column of the backtesting metrics with the number of
# points that enter the metrics (prediction and true value not missing, after
# removing the predictions of overlapping folds), see `_count_evaluated`.
N_EVALUATED_NAME = 'n_evaluated'

# NOTE: The searches warn when a candidate evaluates fewer points than this
# fraction of the points of the most complete candidate, see
# `_warn_n_evaluated_search`. Below 5% of the points left out, the
# bias of the metric is usually smaller than the differences between
# candidates.
N_EVALUATED_WARNING_THRESHOLD = 0.95
# NOTE: Start of the warning of the levels not predicted in the folds of the
# multi-series backtesting. The multi-series searches leave it out when the
# warning of the points evaluated of the candidates covers it.
LEVELS_NOT_PREDICTED_WARNING = (
    "Some levels are not predicted in folds where they have observed values"
)


def initialize_lags_grid(
    forecaster: object, 
    lags_grid: (
        list[int | list[int] | np.ndarray | range | None]
        | dict[str, int | list[int] | np.ndarray | range | None]
        | None
    ) = None,
) -> tuple[dict[str, int | list[int] | np.ndarray | range | None], str]:
    """
    Initialize lags grid and lags label for model selection. 

    Parameters
    ----------
    forecaster : Forecaster
        Forecaster model. ForecasterRecursive, ForecasterDirect, 
        ForecasterRecursiveMultiSeries, ForecasterDirectMultiVariate.
    lags_grid : list, dict, default None
        Lists of lags to try, containing int, lists, numpy ndarray, range
        objects or `None`. A `None` element evaluates the forecaster without
        lags, using only its `window_features` (a `ValueError` is raised if
        it has none). If `dict`, the keys are used as labels in the `results`
        DataFrame, and the values are used as the lists of lags to try. If
        `None`, the lags of the forecaster are used.

    Returns
    -------
    lags_grid : dict
        Dictionary with lags configuration for each iteration.
    lags_label : str
        Label for lags representation in the results object.

    """

    if not isinstance(lags_grid, (list, dict, type(None))):
        raise TypeError(
            f"`lags_grid` argument must be a list, dict or None. "
            f"Got {type(lags_grid)}."
        )

    lags_label = 'values'
    if isinstance(lags_grid, list):
        lags_grid = {f'{lags}': lags for lags in lags_grid}
    elif lags_grid is None:
        if forecaster.lags is None:
            lags_grid = {'None': None}
        else:
            lags = [int(lag) for lag in forecaster.lags]  # Required since numpy 2.0
            lags_grid = {f'{lags}': lags}
    else:
        lags_label = 'keys'

    if (
        any(lags is None for lags in lags_grid.values())
        and forecaster.window_features is None
    ):
        raise ValueError(
            "`lags_grid` contains `None`, a candidate without lags, but the "
            "forecaster has no `window_features` to create the predictors. Set "
            "`window_features` in the forecaster or remove `None` from `lags_grid`."
        )

    return lags_grid, lags_label


def _check_initial_train_size(
    initial_train_size: int,
    initial_train_size_input: int | str | pd.Timestamp,
    data_length: int,
    data_length_name: str = "the length of `y`",
    window_size: int | None = None,
    allow_window_size: bool = False
) -> None:
    """
    Check that `initial_train_size` leaves data to train with and data to
    evaluate. The training set needs more than `window_size` observations (the
    first `window_size` ones only create the predictors of the first row), and
    the rest of the data is evaluated. The error says which limit is not met,
    with the value received.

    Parameters
    ----------
    initial_train_size : int
        Number of observations of the first training set (`initial_train_size`
        of `cv` as an integer).
    initial_train_size_input : int, str, pandas Timestamp
        `initial_train_size` of `cv` as passed by the user, shown in the error.
    data_length : int
        Number of observations of the data.
    data_length_name : str, default 'the length of `y`'
        Name of the length of the data in the error.
    window_size : int, default None
        `window_size` of the forecaster. If `None`, the training set only needs
        one observation (`ForecasterFoundation`, whose `window_size` is the
        maximum context of the model, and the initial check of the searches
        with a `OneStepAheadFold`, whose candidates are checked one by one).
    allow_window_size : bool, default False
        If `True`, `initial_train_size` can be equal to `window_size`
        (`ForecasterStats`, whose `window_size` is ignored).

    Returns
    -------
    None

    """

    if window_size is None:
        min_size = 1
        lower_message = "greater than 0"
    elif allow_window_size:
        min_size = window_size
        lower_message = (
            f"greater than or equal to the `window_size` of the forecaster "
            f"({window_size})"
        )
    else:
        min_size = window_size + 1
        lower_message = (
            f"greater than the `window_size` of the forecaster ({window_size})"
        )

    if isinstance(initial_train_size_input, (str, pd.Timestamp)):
        received = (
            f"Got '{initial_train_size_input}', {initial_train_size} observations "
            f"in the training set."
        )
    else:
        received = f"Got {initial_train_size_input}."

    if window_size is not None and min_size >= data_length:
        raise ValueError(
            f"The `window_size` of the forecaster ({window_size}) leaves no valid "
            f"`initial_train_size`, which must be "
            f"{'at least' if allow_window_size else 'greater than'} the "
            f"`window_size` and smaller than {data_length_name} ({data_length}). "
            f"Reduce the `window_size` (lags, window features, differentiation...) "
            f"or use more data."
        )
    if initial_train_size < min_size:
        raise ValueError(
            f"`initial_train_size` must be {lower_message}, so that the first "
            f"training set has data to train with. {received}"
        )
    if initial_train_size >= data_length:
        raise ValueError(
            f"`initial_train_size` must be smaller than {data_length_name} "
            f"({data_length}), so that there is data left to evaluate. {received}"
        )


def check_backtesting_input(
    forecaster: object,
    cv: object,
    metric: str | Callable | list[str | Callable],
    add_aggregated_metric: bool = True,
    y: pd.Series | None = None,
    series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame] = None,
    exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
    interval: float | list[float] | tuple[float, ...] | str | object | None = None,
    interval_method: str = 'bootstrapping',    
    alpha: float | None = None,
    n_boot: int = 250,
    use_in_sample_residuals: bool = True,
    use_binned_residuals: bool = True,
    random_state: int = 123,
    return_predictors: bool = False,
    freeze_params: bool = True,
    n_jobs: int | str = 'auto',
    show_progress: bool = True,
    suppress_warnings: bool = False
) -> None:
    """
    This is a helper function to check most inputs of backtesting functions in 
    modules `model_selection`.

    Parameters
    ----------
    forecaster : Forecaster
        Forecaster model.
    cv : TimeSeriesFold
        TimeSeriesFold object with the information needed to split the data into folds.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
    add_aggregated_metric : bool, default True
        If `True`, the aggregated metrics (average, weighted average and pooling)
        over all levels are also returned (only multiseries).
    y : pandas Series, default None
        Training time series for uni-series forecasters.
    series : pandas DataFrame, dict, default None
        Training time series for multi-series forecasters.
    exog : pandas Series, pandas DataFrame, dict, default None
        Exogenous variables.
    interval : float, list, tuple, str, object, default None
        Specifies whether probabilistic predictions should be estimated and the 
        method to use. The following options are supported:

        - If `float`, represents the nominal (expected) coverage (between 0 and 1). 
        For instance, `interval=0.95` corresponds to `[0.025, 0.975]` quantiles.
        - If `list` or `tuple`: Sequence of quantiles to compute, each value must 
        be between 0 and 1 inclusive. For example, a 95% confidence interval can 
        be specified as `interval = [0.025, 0.975]` or multiple quantiles (e.g. 0.1, 
        0.5 and 0.9) as `interval = [0.1, 0.5, 0.9]`.
        - If 'bootstrapping' (str): `n_boot` bootstrapping predictions will be generated.
        - If scipy.stats distribution object, the distribution parameters will
        be estimated for each prediction.
        - If None, no probabilistic predictions are estimated.
    interval_method : str, default 'bootstrapping'
        Technique used to estimate prediction intervals. Available options:

        + 'bootstrapping': Bootstrapping is used to generate prediction 
        intervals.
        + 'conformal': Employs the conformal prediction split method for 
        interval estimation.
    alpha : float, default None
        The confidence intervals used in ForecasterStats are (1 - alpha) %. 
    n_boot : int, default `250`
        Number of bootstrapping iterations to perform when estimating prediction
            intervals.
    use_in_sample_residuals : bool, default True
        If `True`, residuals from the training data are used as proxy of prediction 
        error to create prediction intervals.  If `False`, out_sample_residuals 
        are used if they are already stored inside the forecaster.
    use_binned_residuals : bool, default True
        If `True`, residuals are selected based on the predicted values 
        (binned selection).
        If `False`, residuals are selected randomly.
    random_state : int, default `123`
        Seed for the random number generator to ensure reproducibility.
    return_predictors : bool, default False
        If `True`, the predictors used to make the predictions are also returned.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_fit_forecaster.
    freeze_params : bool, default True
        Determines whether to freeze the model parameters after the first fit
        for estimators that perform automatic model selection (`Arima` with
        `order=None` or `seasonal_order=None`, `Ets` with `model='ZZZ'`).

        - If `True`, the parameters selected in the first fit (`best_params_`)
        are reused in all subsequent refits, so the automatic selection runs only
        once. Arima keeps the orders, intercept, drift and Box-Cox lambda, and
        re-estimates the coefficients in each refit. Ets keeps the model type,
        smoothing parameters and Box-Cox lambda, and re-estimates the initial
        states.
        - If `False`, automatic model selection is performed independently in each
        refit, allowing parameters to adapt across folds. This increases runtime
        and adds an `estimator_params` column to the output with the model
        selected in each fold.
    show_progress : bool, default True
        Whether to show a progress bar.
    suppress_warnings: bool, default False
        If `True`, skforecast warnings will be suppressed during the backtesting 
        process. See skforecast.exceptions.warn_skforecast_categories for more
        information.

    Returns
    -------
    None
    
    """

    forecaster_name = type(forecaster).__name__
    cv_name = type(cv).__name__

    if cv_name != "TimeSeriesFold":
        raise TypeError(f"`cv` must be a 'TimeSeriesFold' object. Got '{cv_name}'.")

    steps = cv.steps
    initial_train_size = cv.initial_train_size
    gap = cv.gap
    allow_incomplete_fold = cv.allow_incomplete_fold
    refit = cv.refit

    forecasters_uni = [
        "ForecasterRecursive",
        "ForecasterDirect",
        "ForecasterStats",
        "ForecasterEquivalentDate",
        "ForecasterRecursiveClassifier"
    ]
    forecasters_direct = [
        "ForecasterDirect",
        "ForecasterDirectMultiVariate",
        "ForecasterRnn"
    ]
    forecasters_multi_no_dict = [
        "ForecasterDirectMultiVariate",
        "ForecasterRnn",
    ]
    forecasters_multi_dict = [
        "ForecasterRecursiveMultiSeries",
        "ForecasterFoundation"
    ]
    # NOTE: ForecasterStats has interval but not with bootstrapping or conformal
    # NOTE: ForecasterRnn has interval with conformal but not bootstrapping
    forecasters_boot_conformal = [
        "ForecasterRecursive",
        "ForecasterDirect",
        "ForecasterRecursiveMultiSeries",
        "ForecasterDirectMultiVariate",
        "ForecasterEquivalentDate",
    ]
    forecasters_return_predictors = [
        "ForecasterRecursive",
        "ForecasterDirect",
        "ForecasterRecursiveMultiSeries",
        "ForecasterDirectMultiVariate",
        "ForecasterRecursiveClassifier"
    ]

    if forecaster_name in forecasters_uni:
        if not isinstance(y, pd.Series):
            raise TypeError("`y` must be a pandas Series.")
        data_name = 'y'
        data_length_name = "the length of `y`"
        index = cv._extract_index(y)

    elif forecaster_name in forecasters_multi_no_dict:
        if not isinstance(series, pd.DataFrame):
            raise TypeError("`series` must be a pandas DataFrame.")
        data_name = 'series'
        data_length_name = "the length of `series`"
        index = cv._extract_index(series)

    elif forecaster_name in forecasters_multi_dict:

        # NOTE: Checks are not need as they are done in the function
        # `check_preprocess_series` that is used before `check_backtesting_input`
        # in the backtesting function.

        # NOTE: The folds are built over the span of all series (from the
        # earliest to the latest index), which is longer than any of them when
        # they do not cover the same period.
        data_name = 'series'
        data_length_name = "the length of the span of `series`"
        index = cv._extract_index(series)

    data_length = len(index)

    if exog is not None:
        if forecaster_name in forecasters_multi_dict:
            # NOTE: Checks are not need as they are done in the function
            # `check_preprocess_exog_multiseries` that is used before
            # `check_backtesting_input` in the backtesting function.
            pass
        else:
            if not isinstance(exog, (pd.Series, pd.DataFrame)):
                raise TypeError(
                    f"`exog` must be a pandas Series, DataFrame or None. Got {type(exog)}."
                )

    if not isinstance(metric, (str, Callable, list)):
        raise TypeError(
            f"`metric` must be a string, a callable function, or a list containing "
            f"multiple strings and/or callables. Got {type(metric)}."
        )

    if forecaster_name == "ForecasterEquivalentDate" and isinstance(
        forecaster.offset, pd.tseries.offsets.DateOffset
    ):
        # NOTE: Checks when initial_train_size is not None cannot be done here
        # because the forecaster is not fitted yet and we don't know the
        # window_size since pd.DateOffset is not a fixed window size.
        if initial_train_size is None:
            raise ValueError(
                f"`initial_train_size` must be an integer greater than "
                f"the `window_size` of the forecaster ({forecaster.window_size}) "
                f"and smaller than {data_length_name} ({data_length}) or "
                f"a date within this range of the index."
            )
    elif initial_train_size is not None:
        initial_train_size = date_to_index_position(
                                 index        = index, 
                                 date_input   = initial_train_size, 
                                 method       = 'validation',
                                 date_literal = 'initial_train_size'
                             )
        # NOTE: For ForecasterFoundation, window_size equals context_length,
        # the *maximum* context the model accepts, not a minimum training size:
        # initial_train_size only needs to be at least 1. The window_size of
        # ForecasterStats is ignored, initial_train_size can be equal to it.
        _check_initial_train_size(
            initial_train_size       = initial_train_size,
            initial_train_size_input = cv.initial_train_size,
            data_length              = data_length,
            data_length_name         = data_length_name,
            window_size              = (
                None if forecaster_name == 'ForecasterFoundation'
                else forecaster.window_size
            ),
            allow_window_size        = forecaster_name == 'ForecasterStats'
        )
        if allow_incomplete_fold:
            # At least one observation after the gap to allow incomplete fold
            if data_length <= initial_train_size + gap:
                raise ValueError(
                    f"`{data_name}` must have more than `initial_train_size + gap` "
                    f"observations to create at least one fold.\n"
                    f"    Time series length: {data_length}\n"
                    f"    Required > {initial_train_size + gap}\n"
                    f"    initial_train_size: {initial_train_size}\n"
                    f"    gap: {gap}\n"
                )
        else:
            # At least one complete fold
            if data_length < initial_train_size + gap + steps:
                raise ValueError(
                    f"`{data_name}` must have at least `initial_train_size + gap + steps` "
                    f"observations to create a minimum of one complete fold "
                    f"(allow_incomplete_fold=False).\n"
                    f"    Time series length: {data_length}\n"
                    f"    Required >= {initial_train_size + gap + steps}\n"
                    f"    initial_train_size: {initial_train_size}\n"
                    f"    gap: {gap}\n"
                    f"    steps: {steps}\n"
                )
    else:
        if forecaster_name in [
            'ForecasterStats', 'ForecasterFoundation', 'ForecasterEquivalentDate'
        ]:
            raise ValueError(
                f"When using {forecaster_name}, `initial_train_size` must be an "
                f"integer smaller than {data_length_name} ({data_length})."
            )
        else:
            if not forecaster.is_fitted:
                raise NotFittedError(
                    "`forecaster` must be already trained if no `initial_train_size` "
                    "is provided."
                )
            if refit:
                raise ValueError(
                    "`refit` is only allowed when `initial_train_size` is not `None`."
                )

    if forecaster_name == 'ForecasterStats' and cv.skip_folds is not None:
        raise ValueError(
            "`skip_folds` is not allowed for ForecasterStats. Set it to `None`."
        )

    # NOTE: The names are those of the columns of the metrics table. A callable
    # without `__name__` (for example, a `functools.partial`) is named after
    # the wrapper of `add_y_train_argument`.
    metric_names = [
        m if isinstance(m, str) else getattr(add_y_train_argument(m), '__name__', None)
        for m in (metric if isinstance(metric, list) else [metric])
    ]
    duplicated_names = list(dict.fromkeys(
        name for name in metric_names
        if name is not None and metric_names.count(name) > 1
    ))
    if duplicated_names:
        raise ValueError(
            f"Each metric name must be unique, they are the columns of the "
            f"metrics table. Duplicated names: {duplicated_names}. Give each "
            f"function a distinct `__name__` (for example, "
            f"`create_mean_pinball_loss` includes the quantile in the name)."
        )
    if N_EVALUATED_NAME in metric_names:
        raise ValueError(
            f"The backtesting metrics include the column '{N_EVALUATED_NAME}' "
            f"with the number of points evaluated. Rename the metric "
            f"'{N_EVALUATED_NAME}' passed in `metric`."
        )

    # NOTE: `backtesting_foundation` passes its `quantiles` as `interval`.
    if _get_interval_bound_quantiles(interval) is not None or alpha is not None:
        reserved_names = [name for name in metric_names if name in INTERVAL_METRIC_NAMES]
        if reserved_names:
            raise ValueError(
                f"When `interval` has a lower and an upper bound, or `alpha` is "
                f"used, the backtesting metrics include {INTERVAL_METRIC_NAMES}. "
                f"Rename the metric(s) {reserved_names} passed in `metric`."
            )

    if not isinstance(add_aggregated_metric, bool):
        raise TypeError("`add_aggregated_metric` must be a boolean: `True`, `False`.")
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 0:
        raise TypeError(f"`n_boot` must be an integer greater than 0. Got {n_boot}.")
    if not isinstance(use_in_sample_residuals, bool):
        raise TypeError("`use_in_sample_residuals` must be a boolean: `True`, `False`.")
    if not isinstance(use_binned_residuals, bool):
        raise TypeError("`use_binned_residuals` must be a boolean: `True`, `False`.")
    if not isinstance(random_state, (int, np.integer)) or random_state < 0:
        raise TypeError(f"`random_state` must be an integer greater than 0. Got {random_state}.")
    if not isinstance(return_predictors, bool):
        raise TypeError("`return_predictors` must be a boolean: `True`, `False`.")
    if not isinstance(freeze_params, bool):
        raise TypeError("`freeze_params` must be a boolean: `True`, `False`.")
    if not isinstance(n_jobs, int) and n_jobs != 'auto':
        raise TypeError(f"`n_jobs` must be an integer or `'auto'`. Got {n_jobs}.")
    if not isinstance(show_progress, bool):
        raise TypeError("`show_progress` must be a boolean: `True`, `False`.")
    if not isinstance(suppress_warnings, bool):
        raise TypeError("`suppress_warnings` must be a boolean: `True`, `False`.")

    if interval is not None or alpha is not None:
        
        if forecaster_name in forecasters_boot_conformal:

            if interval_method == 'conformal':
                if not isinstance(interval, (float, list, tuple)):
                    raise TypeError(
                        f"When `interval_method` is 'conformal', `interval` must "
                        f"be a float or a list/tuple defining a symmetric interval. "
                        f"Got {type(interval)}."
                    )
                if isinstance(interval, (list, tuple)) and len(interval) != 2:
                    raise ValueError(
                        f"When `interval_method` is 'conformal', `interval` must be "
                        f"a float or a list/tuple of exactly 2 values (lower and "
                        f"upper quantile) defining a symmetric interval. Multiple "
                        f"quantiles (e.g. `interval=[0.1, 0.5, 0.9]`) are only "
                        f"supported when `interval_method='bootstrapping'`. "
                        f"Got {interval}."
                    )
            elif interval_method == 'bootstrapping':
                if (
                    not isinstance(interval, (float, list, tuple, str))
                    and (not hasattr(interval, "_pdf") or not callable(getattr(interval, "fit", None)))
                ):                
                    raise TypeError(
                        f"When `interval_method` is 'bootstrapping', `interval` "
                        f"must be a float, a list or tuple of floats, a "
                        f"scipy.stats distribution object (with methods `_pdf` and "
                        f"`fit`) or the string 'bootstrapping'. Got {type(interval)}."
                    )
                if isinstance(interval, (list, tuple)):
                    if len(interval) == 2:
                        check_interval(
                            interval                   = interval,
                            ensure_symmetric_intervals = False
                        )
                    else:
                        for q in interval:
                            if (q < 0.) or (q > 1.):
                                raise ValueError(
                                    f"When `interval` is a list or tuple, all values must be "
                                    f"between 0 and 1 inclusive. Got {q} in {interval}."
                                )
                elif isinstance(interval, float):
                    if (interval <= 0.) or (interval >= 1.):
                        raise ValueError(
                            f"When `interval` is a float, it must be between 0 and 1 "
                            f"exclusive. Got {interval}."
                        )
                elif isinstance(interval, str):
                    if interval != 'bootstrapping':
                        raise ValueError(
                            f"When `interval` is a string, it must be 'bootstrapping'."
                            f"Got {interval}."
                        )
            else:
                raise ValueError(
                    f"`interval_method` must be 'bootstrapping' or 'conformal'. "
                    f"Got {interval_method}."
                )
        else:
            if forecaster_name == 'ForecasterRecursiveClassifier':
                raise ValueError(
                    f"`interval` is not supported for {forecaster_name}. Class "
                    f"probabilities are returned by default during backtesting, "
                    f"set `interval=None`."
                )
            if forecaster_name == 'ForecasterFoundation':
                check_interval(quantiles=interval)
            else:
                check_interval(interval=interval, alpha=alpha)

    if return_predictors and forecaster_name not in forecasters_return_predictors:
        raise ValueError(
            f"`return_predictors` is only allowed for forecasters of type "
            f"{forecasters_return_predictors}. Got {forecaster_name}."
        )

    if forecaster_name in forecasters_direct and forecaster.max_step < steps + gap:
        raise ValueError(
            f"When using a {forecaster_name}, the combination of steps "
            f"+ gap ({steps + gap}) cannot be greater than the `steps` parameter "
            f"declared when the forecaster is initialized ({forecaster.max_step})."
        )


def check_one_step_ahead_input(
    forecaster: object,
    cv: object,
    metric: str | Callable | list[str | Callable],
    y: pd.Series | None = None,
    series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame] = None,
    exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
    show_progress: bool = True,
    suppress_warnings: bool = False
) -> None:
    """
    This is a helper function to check most inputs of hyperparameter tuning
    functions in modules `model_selection` when using a `OneStepAheadFold`.

    Parameters
    ----------
    forecaster : Forecaster
        Forecaster model.
    cv : OneStepAheadFold
        OneStepAheadFold object with the information needed to split the data into folds.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
    y : pandas Series, default None
        Training time series for uni-series forecasters.
    series : pandas DataFrame, dict, default None
        Training time series for multi-series forecasters.
    exog : pandas Series, pandas DataFrame, dict, default None
        Exogenous variables.
    show_progress : bool, default True
        Whether to show a progress bar.
    suppress_warnings: bool, default False
        If `True`, skforecast warnings will be suppressed during the hyperparameter 
        search. See skforecast.exceptions.warn_skforecast_categories for more
        information.

    Returns
    -------
    None
    
    """

    forecaster_name = type(forecaster).__name__
    cv_name = type(cv).__name__

    if cv_name != "OneStepAheadFold":
        raise TypeError(f"`cv` must be a 'OneStepAheadFold' object. Got '{cv_name}'.")

    initial_train_size = cv.initial_train_size

    forecasters_one_step_ahead = [
        "ForecasterRecursive",
        "ForecasterDirect",
        "ForecasterRecursiveClassifier",
        'ForecasterRecursiveMultiSeries',
        'ForecasterDirectMultiVariate'
    ]
    if forecaster_name not in forecasters_one_step_ahead:
        raise TypeError(
            f"Only forecasters of type {forecasters_one_step_ahead} are allowed "
            f"when using `cv` of type `OneStepAheadFold`. Got {forecaster_name}."
        )

    forecasters_uni = [
        "ForecasterRecursive",
        "ForecasterDirect",
        "ForecasterRecursiveClassifier"
    ]
    forecasters_multi_no_dict = [
        "ForecasterDirectMultiVariate",
    ]
    forecasters_multi_dict = [
        "ForecasterRecursiveMultiSeries"
    ]

    if forecaster_name in forecasters_uni:
        if not isinstance(y, pd.Series):
            raise TypeError(f"`y` must be a pandas Series. Got {type(y)}")
        data_length_name = "the length of `y`"
        index = cv._extract_index(y)

    elif forecaster_name in forecasters_multi_no_dict:
        if not isinstance(series, pd.DataFrame):
            raise TypeError(f"`series` must be a pandas DataFrame. Got {type(series)}")
        data_length_name = "the length of `series`"
        index = cv._extract_index(series)

    elif forecaster_name in forecasters_multi_dict:

        # NOTE: Checks are not need as they are done in the function
        # `check_preprocess_series` that is used before `check_one_step_ahead_input`
        # in the backtesting function.

        # NOTE: The training and test sets are split over the span of all series
        # (from the earliest to the latest index), which is longer than any of
        # them when they do not cover the same period.
        data_length_name = "the length of the span of `series`"
        index = cv._extract_index(series)

    data_length = len(index)

    if exog is not None:
        if forecaster_name in forecasters_multi_dict:
            # NOTE: Checks are not need as they are done in the function 
            # `check_preprocess_exog_multiseries` that is used before 
            # `check_backtesting_input` in the backtesting function.
            pass
        else:
            if not isinstance(exog, (pd.Series, pd.DataFrame)):
                raise TypeError(
                    f"`exog` must be a pandas Series, DataFrame or None. Got {type(exog)}."
                )

    if not isinstance(metric, (str, Callable, list)):
        raise TypeError(
            f"`metric` must be a string, a callable function, or a list containing "
            f"multiple strings and/or callables. Got {type(metric)}."
        )

    initial_train_size = date_to_index_position(
                             index        = index,
                             date_input   = initial_train_size,
                             method       = 'validation',
                             date_literal = 'initial_train_size'
                         )
    # NOTE: The `window_size` is checked for each candidate of the search
    # (`_train_test_split_one_step_ahead_search`), since the candidates can
    # have a different `window_size` than the forecaster passed.
    _check_initial_train_size(
        initial_train_size       = initial_train_size,
        initial_train_size_input = cv.initial_train_size,
        data_length              = data_length,
        data_length_name         = data_length_name,
        window_size              = None
    )

    if not isinstance(show_progress, bool):
        raise TypeError("`show_progress` must be a boolean: `True`, `False`.")
    if not isinstance(suppress_warnings, bool):
        raise TypeError("`suppress_warnings` must be a boolean: `True`, `False`.")
    
    if not suppress_warnings:
        warnings.warn(
            "One-step-ahead predictions are used for faster model comparison, but they "
            "may not fully represent multi-step prediction performance. It is recommended "
            "to backtest the final model for a more accurate multi-step performance "
            "estimate.",
            OneStepAheadValidationWarning,
            stacklevel = _find_stack_level()
        )


def select_n_jobs_backtesting(
    forecaster: object,
    refit: bool | int
) -> int:
    """
    Select the optimal number of jobs to use in the backtesting process. This
    selection is based on heuristics and is not guaranteed to be optimal.

    The number of jobs is chosen as follows:

    - If `refit` is an integer other than 0 or 1, then `n_jobs = 1`. The
    forecaster fitted in a fold is reused in the following ones, so the folds
    cannot run in parallel.
    - If forecaster is not 'ForecasterRecursive', 'ForecasterRecursiveClassifier'
    or 'ForecasterRecursiveMultiSeries', then `n_jobs = 1`. 'ForecasterDirect'
    and 'ForecasterDirectMultiVariate' parallelize the fit of their steps
    instead (see `skforecast.utils.select_n_jobs_fit_forecaster`), and
    'ForecasterStats', 'ForecasterEquivalentDate' and 'ForecasterRnn' are not
    parallelized.
    - If `refit` is `False` (or 0), then `n_jobs = 1`, unless estimator, or the
    last step of a `Pipeline`, is a `HistGradientBoostingRegressor` or a
    `CatBoostRegressor`. The folds only predict with the forecaster fitted
    before them, and sending the fitted forecaster to the workers usually costs
    more than the predictions they share. These two estimators predict a few
    rows slowly and their fitted models are small, so their predictions pay
    for sending them.
    - If estimator, or the last step of a `Pipeline`, is a linear model of
    scikit-learn (a subclass of `LinearModel`, such as `Ridge`), then
    `n_jobs = 1`. It is fitted and predicts in less time than the workers
    take to start. `LogisticRegression` is not a subclass of `LinearModel`:
    it is fitted iteratively, so it is parallelized.
    - If estimator, or the last step of a `Pipeline`, is a `LGBMRegressor` or
    `LGBMClassifier` with internal `n_jobs != 1`, then `n_jobs = 1`. LightGBM
    already parallelizes its operations with threads, and more processes
    would compete for the same cores.
    - Otherwise, `n_jobs = max(1, cpu_count() - 1)`.

    The hyperparameter searches use these rules for every candidate. A single
    backtesting (`backtesting_forecaster`, `backtesting_forecaster_multiseries`)
    then limits the result by its workload with
    `skforecast.model_selection._utils.limit_n_jobs_by_workload`.

    Parameters
    ----------
    forecaster : Forecaster
        Forecaster model.
    refit : bool, int
        If the forecaster is refitted during the backtesting process.

    Returns
    -------
    n_jobs : int
        The number of jobs to run in parallel.
    
    """

    refit = False if refit == 0 else refit
    if not isinstance(refit, bool) and refit != 1:
        return 1

    if type(forecaster).__name__ not in {
        "ForecasterRecursive",
        "ForecasterRecursiveClassifier",
        "ForecasterRecursiveMultiSeries",
    }:
        return 1

    estimator = forecaster.estimator
    if isinstance(estimator, Pipeline):
        estimator = estimator[-1]

    # NOTE: Without refit, each task receives the fitted forecaster and only
    # predicts, so parallelizing only pays off when the prediction is slow and
    # the fitted model is cheap to send to the workers. That is the case of
    # `HistGradientBoostingRegressor` and `CatBoostRegressor` (high fixed cost
    # per predict call, models of a few hundred KB): their searches without
    # refit were 1.5 to 3 times faster in parallel, at the cost of starting the
    # workers in small searches. LightGBM, XGBoost and RandomForest did not
    # gain. Review this exception if these libraries make their prediction
    # faster. A single backtesting without refit still uses one job
    # (`limit_n_jobs_by_workload`).
    if (
        refit is False
        and type(estimator).__name__
        not in {"HistGradientBoostingRegressor", "CatBoostRegressor"}
    ):
        return 1

    if isinstance(estimator, LinearModel):
        return 1

    if (
        type(estimator).__name__ in {"LGBMRegressor", "LGBMClassifier"}
        and estimator.n_jobs != 1
    ):
        return 1

    return max(1, cpu_count() - 1)


# NOTE: Workload, in seconds, below which a single backtesting with
# `n_jobs='auto'` runs with one job, because starting the workers costs more
# than it saves. The workload is estimated as the duration of the initial fit
# times the number of refits, which gives about 80% of the real sequential
# time. Measured with 21 workers that import skforecast, parallelizing pays
# off from about 5 s of real sequential work on Windows (4 s of estimated
# workload), from about 2.8 s on Linux (2 s) and from about 2.4 s on macOS
# (2 s, Apple M5 with 9 workers). Other platforms use the value of Windows.
_MIN_PARALLEL_WORKLOAD_SECONDS = {"win32": 4.0, "linux": 2.0, "darwin": 2.0}
_MIN_PARALLEL_WORKLOAD_SECONDS_DEFAULT = 4.0


def limit_n_jobs_by_workload(
    n_jobs: int,
    n_tasks: int,
    n_fits: int,
    fit_seconds: float | None
) -> int:
    """
    Limit the number of jobs selected with `n_jobs='auto'` by the workload of
    a single backtesting. Starting the workers of `joblib` takes seconds (more
    on Windows than on Linux), so parallelizing only pays off when the folds
    have enough work to share. The result is never greater than `n_jobs`.

    - If `n_jobs` is 1, there is at most one task or no task fits the
    forecaster (no refit), then `n_jobs = 1`. Predicting the folds of a fitted
    forecaster takes less time than the workers take to start.
    - If the estimated workload, `n_fits * fit_seconds`, is below the minimum
    of the platform (4 seconds on Windows, 2 seconds on Linux and macOS), then
    `n_jobs = 1`. If the time of the initial fit is not known
    (`fit_seconds=None`), the workload is not estimated.
    - Otherwise, `n_jobs = min(n_jobs, n_tasks)`: no more workers are started
    than there are tasks.

    Parameters
    ----------
    n_jobs : int
        Number of jobs selected with `select_n_jobs_backtesting`.
    n_tasks : int
        Number of tasks (folds) sent to `joblib.Parallel`.
    n_fits : int
        Number of fits made in the tasks (folds with refit), without the
        initial fit.
    fit_seconds : float, None
        Duration of the initial fit, in seconds. `None` if the forecaster
        was not fitted before the folds.

    Returns
    -------
    n_jobs : int
        The number of jobs to run in parallel.

    """

    if n_jobs <= 1 or n_tasks <= 1 or n_fits == 0:
        return 1

    if fit_seconds is not None:
        min_workload = _MIN_PARALLEL_WORKLOAD_SECONDS.get(
            sys.platform, _MIN_PARALLEL_WORKLOAD_SECONDS_DEFAULT
        )
        if n_fits * fit_seconds < min_workload:
            return 1

    return min(n_jobs, n_tasks)


def _train_test_split_one_step_ahead_search(
    forecaster: object,
    cv: object,
    initial_train_size: int,
    y: pd.Series | None = None,
    series: pd.DataFrame | dict[str, pd.Series] | None = None,
    exog: pd.Series | pd.DataFrame | dict[str, pd.DataFrame | None] | None = None,
    levels: list[str] | None = None,
    max_levels: int = 5
) -> tuple:
    """
    Training and test matrices of a candidate of a search with a
    `OneStepAheadFold`, created with `_train_test_split_one_step_ahead` of the
    forecaster. Before the split, `initial_train_size` is checked against the
    `window_size` of the candidate. In multi-series searches, the split is
    rejected if no level of `levels` can be evaluated in the test set, with
    the cause of each level. The errors are raised for the candidate, so the
    search can skip it.

    Parameters
    ----------
    forecaster : ForecasterRecursive, ForecasterDirect, ForecasterRecursiveMultiSeries, ForecasterDirectMultiVariate
        Forecaster with the lags and parameters of the candidate.
    cv : OneStepAheadFold
        OneStepAheadFold object of the search.
    initial_train_size : int
        `initial_train_size` of `cv` as an integer.
    y : pandas Series, default None
        Time series of the single series searches.
    series : pandas DataFrame, dict, default None
        Time series of the multi-series searches (a dict, already checked, for
        ForecasterRecursiveMultiSeries).
    exog : pandas Series, pandas DataFrame, dict, default None
        Exogenous variables.
    levels : list, default None
        Levels of the multi-series searches.
    max_levels : int, default 5
        Maximum number of levels whose cause is given in the error when no
        level can be evaluated.

    Returns
    -------
    split : tuple
        Output of `_train_test_split_one_step_ahead` of the forecaster.

    """

    forecaster_name = type(forecaster).__name__
    if y is not None:
        data = y
        data_length_name = "the length of `y`"
    else:
        data = series
        data_length_name = (
            "the length of the span of `series`"
            if forecaster_name == "ForecasterRecursiveMultiSeries"
            else "the length of `series`"
        )
    _check_initial_train_size(
        initial_train_size       = initial_train_size,
        initial_train_size_input = cv.initial_train_size,
        data_length              = len(cv._extract_index(data)),
        data_length_name         = data_length_name,
        window_size              = forecaster.window_size
    )

    if y is not None:
        return forecaster._train_test_split_one_step_ahead(
                   y=y, initial_train_size=initial_train_size, exog=exog
               )

    split = forecaster._train_test_split_one_step_ahead(
                series=series, initial_train_size=initial_train_size, exog=exog
            )
    y_test, X_test_encoding = split[3], split[5]
    if forecaster_name == "ForecasterRecursiveMultiSeries":
        is_observed = pd.notna(y_test).to_numpy()
        levels_test = set(np.asarray(X_test_encoding)[is_observed])
        has_points = any(level in levels_test for level in levels)
        # NOTE: Same rule as `_train_test_split_one_step_ahead` of the forecaster.
        dropna_last_window = (
            forecaster.differentiation_max is not None
            or not estimator_has_native_nan_support(forecaster.estimator)
        )
    else:
        has_points = bool(pd.notna(y_test).any())
        dropna_last_window = False
    if not has_points:
        span_index = cv._extract_index(series)
        causes = [
            _cause_level_not_evaluated(
                level              = level,
                y                  = series[level],
                span_index         = span_index,
                initial_train_size = initial_train_size,
                window_size        = forecaster.window_size,
                dropna_last_window = dropna_last_window
            )
            for level in levels[:max_levels]
        ]
        causes_repr = [cause for cause, _ in causes]
        n_more = len(levels) - max_levels
        if n_more > 0:
            causes_repr.append(f"and {n_more} more level{'s' if n_more > 1 else ''}")
        # NOTE: The conditions are explained only when the cause of a level is
        # not one of them.
        conditions = (
            " A level is evaluated if it has more than `window_size` observations "
            "in the training set, its last window can be used to predict, and it "
            "has observed values in the test set."
            if any(is_generic for _, is_generic in causes)
            else ""
        )
        raise ValueError(
            f"No level of `levels` can be evaluated in the test set with this "
            f"`window_size` ({forecaster.window_size}). {'; '.join(causes_repr)}."
            f"{conditions} Review `levels`, `initial_train_size` of `cv` or the "
            f"span of the series."
        )

    return split


def _cause_level_not_evaluated(
    level: str,
    y: pd.Series,
    span_index: pd.Index,
    initial_train_size: int,
    window_size: int,
    dropna_last_window: bool
) -> tuple[str, bool]:
    """
    Why a level cannot be evaluated in the test set of a `OneStepAheadFold`,
    checked in the order of `_train_test_split_one_step_ahead`: too few
    observations in the training set (after removing the leading and trailing
    NaNs), a last window that cannot be used to predict, or no observed values
    in the test set. If none of them applies, a generic cause is returned.

    Parameters
    ----------
    level : str
        Name of the level.
    y : pandas Series
        Time series of the level.
    span_index : pandas Index
        Index of the span of all the series.
    initial_train_size : int
        `initial_train_size` of `cv` as an integer.
    window_size : int
        `window_size` of the candidate.
    dropna_last_window : bool
        Whether a last window with NaNs is left out (the estimator does not
        support NaNs, or the forecaster has `differentiation`).

    Returns
    -------
    cause : str
        Cause, a sentence that starts with the name of the level, for the
        error of the search.
    is_generic : bool
        Whether the cause is the generic one.

    """

    train_end = span_index[initial_train_size - 1]
    y_train = y.loc[:train_end]
    first_valid, last_valid = y_train.first_valid_index(), y_train.last_valid_index()
    n_train = 0 if first_valid is None else len(y_train.loc[first_valid:last_valid])
    if n_train <= window_size:
        return (
            f"'{level}' has {n_train} observation{'' if n_train == 1 else 's'} "
            f"in the training set, not more than the `window_size`"
        ), False

    if y.index[-1] < train_end:
        return (
            f"'{level}' ends before the end of the training set, so it cannot be "
            f"predicted after it"
        ), False

    last_window = y.loc[span_index[initial_train_size - window_size]:train_end]
    if dropna_last_window and last_window.isna().to_numpy().any():
        return (
            f"'{level}' has missing values in its last window (the `window_size` "
            f"observations before the test set)"
        ), False

    if not y.loc[span_index[initial_train_size]:].notna().to_numpy().any():
        return f"'{level}' has no observed values in the test set", False

    return (
        f"'{level}' has no row in the test set without missing values in `y` or "
        f"in its lags"
    ), True


def _get_test_positions(
    y: pd.Series,
    test_index: pd.Index
) -> np.ndarray:
    """
    Positions in `y` of the dates (or index values) of the one-step-ahead test
    set. The true values with differentiation, and the forecast origins of the
    scaled metrics, are read from `y` at these positions.

    Parameters
    ----------
    y : pandas Series
        Time series, in its original scale.
    test_index : pandas Index
        Index of the rows of the test set.

    Returns
    -------
    test_positions : numpy ndarray
        Position in `y` of each row of the test set.

    """

    test_positions = y.index.get_indexer(test_index)
    if (test_positions < 0).any():
        n_missing = int((test_positions < 0).sum())
        raise ValueError(
            f"{n_missing} dates of the test set are not in the index of the "
            f"series '{y.name}', so its true values cannot be read from it. "
            f"First missing date: {test_index[test_positions < 0][0]}."
        )

    return test_positions


def _revert_differentiation_one_step_ahead(
    y_observed: np.ndarray,
    diff_true: np.ndarray,
    diff_pred: np.ndarray,
    transformer: object | None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Revert the differentiation of one-step-ahead predictions. Each prediction
    is made from the observed values before it, so, in the transformed scale,
    it is the observed value minus the observed difference plus the predicted
    one:

    `level_pred_t = level_true_t - diff_true_t + diff_pred_t`

    This holds for any order of differentiation: the error of the level is the
    error of the difference.

    The `inverse_transform_next_window` method of the differentiator is not
    used: it chains the predictions, as `predict` does, adding each predicted
    difference to the previous prediction, so the error of each prediction
    would be carried into the following ones. The observed values are taken
    from the series rather than rebuilt from the differences, since the rows
    with missing values are left out of the test set.

    Parameters
    ----------
    y_observed : numpy ndarray
        Observed values of the series at the predicted positions, in the
        original scale.
    diff_true : numpy ndarray
        Target of the estimator at the same positions (the transformed and
        differentiated series).
    diff_pred : numpy ndarray
        Predictions of the estimator at the same positions.
    transformer : object, None
        Fitted transformer of the series (`transformer_y`, or the transformer
        of the level in `transformer_series_`), or `None`.

    Returns
    -------
    y_true : numpy ndarray
        Observed values in the original scale.
    y_pred : numpy ndarray
        One-step-ahead predictions in the original scale.

    """

    y_true = np.asarray(y_observed, dtype=float).ravel()
    level_true = transform_numpy(
                     array       = y_true,
                     transformer = transformer,
                     fit         = False
                 ).ravel()
    level_pred = (
        level_true
        - np.asarray(diff_true, dtype=float).ravel()
        + np.asarray(diff_pred, dtype=float).ravel()
    )
    y_pred = transform_numpy(
                 array             = level_pred,
                 transformer       = transformer,
                 fit               = False,
                 inverse_transform = True
             ).ravel()

    return y_true, y_pred


def _calculate_metrics_one_step_ahead(
    forecaster: object,
    metrics: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_weight: np.ndarray | None,
    fit_kwargs: dict[str, Any],
    y: pd.Series,
    test_index: pd.Index
) -> tuple[list, int]:
    """
    Calculate metrics when predictions are one-step-ahead. When forecaster is
    of type ForecasterDirect only the estimator for step 1 is used. Data is
    expected as numpy arrays already filtered and ready for fitting.

    Parameters
    ----------
    forecaster : object
        Forecaster model.
    metrics : list
        List of metrics.
    X_train : numpy ndarray
        Predictor values used to train the model.
    y_train : numpy ndarray
        Target values related to each row of `X_train`.
    X_test : numpy ndarray
        Predictor values used to test the model.
    y_test : numpy ndarray
        Target values related to each row of `X_test`.
    sample_weight : numpy ndarray, None
        Precomputed sample weights for training. `None` if no weight function
        is defined.
    fit_kwargs : dict
        Precomputed keyword arguments for `estimator.fit`. Can be empty dict.
    y : pandas Series
        Time series used in the search, in its original scale. With
        `differentiation`, its values at `test_index` are the true values, from
        which the predictions are reverted one step ahead (see
        `_revert_differentiation_one_step_ahead`). The metrics with the
        argument `y_train` use its values: the scaled metrics scale the error
        of each prediction with the observations before it (its forecast origin
        is its own position, since it is predicted one step ahead), and the
        rest receive the observations before the first prediction.
    test_index : pandas Index
        Index of `y` of each row of `X_test`.

    Returns
    -------
    metric_values : list
        List with metric values.
    n_evaluated : int
        Number of points that enter the metrics: true value and prediction
        not missing.
    
    """

    if type(forecaster).__name__ == 'ForecasterDirect':
        # NOTE: Only step 1 is optimized in one-step-ahead validation.
        estimator = forecaster.estimators_[1]
    else:
        estimator = forecaster.estimator

    if sample_weight is not None:
        estimator.fit(
            X             = X_train,
            y             = y_train,
            sample_weight = sample_weight,
            **fit_kwargs
        )
    else:
        estimator.fit(X=X_train, y=y_train, **fit_kwargs)

    y_true = y_test
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning
        )
        y_pred = estimator.predict(X_test).ravel()

    # NOTE: Each prediction is made one step ahead with the observed values,
    # so its forecast origin is its own position in `y`.
    test_positions = _get_test_positions(y=y, test_index=test_index)

    if forecaster.differentiation is not None:
        y_true, y_pred = _revert_differentiation_one_step_ahead(
                             y_observed  = y.to_numpy()[test_positions],
                             diff_true   = y_test,
                             diff_pred   = y_pred,
                             transformer = forecaster.transformer_y
                         )
    elif forecaster.transformer_y is not None:
        y_true = forecaster.transformer_y.inverse_transform(y_true.reshape(-1, 1))
        y_pred = forecaster.transformer_y.inverse_transform(y_pred.reshape(-1, 1))

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    y_train, scale_abs, scale_sq = _get_y_train_and_scales(
                                       y       = y,
                                       origins = test_positions,
                                       metrics = metrics
                                   )
    metric_values, n_undefined = _calculate_metric_values(
                                     metrics   = metrics,
                                     y_true    = y_true,
                                     y_pred    = y_pred,
                                     y_train   = y_train,
                                     scale_abs = scale_abs,
                                     scale_sq  = scale_sq
                                 )
    _warn_undefined_scale({None: (n_undefined, len(y_true))})
    n_evaluated = _count_evaluated(y_true=y_true, y_pred=y_pred)

    return metric_values, n_evaluated


def _initialize_levels_model_selection_multiseries(
    forecaster: object, 
    series:  pd.DataFrame | dict[str, pd.Series | pd.DataFrame],
    levels: str | list | None = None
) -> list:
    """
    Initialize levels for model_selection multi-series functions.

    Parameters
    ----------
    forecaster : ForecasterRecursiveMultiSeries, ForecasterDirectMultiVariate, ForecasterRnn, ForecasterFoundation
        Forecaster model.
    series : pandas DataFrame, dict
        Training time series.
    levels : str, list, default None
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account. The resulting metric will be
        the average of the optimization of all levels.

    Returns
    -------
    levels : list
        List of levels to be used in model_selection multi-series functions.
    
    """

    multi_series_forecasters_with_levels = [
        'ForecasterRecursiveMultiSeries', 
        'ForecasterRnn',
        'ForecasterFoundation'
    ]

    if type(forecaster).__name__ in multi_series_forecasters_with_levels  \
        and not isinstance(levels, (str, list, type(None))):
        raise TypeError(
            f"`levels` must be a `list` of column names, a `str` of a column "
            f"name or `None` when using a forecaster of type "
            f"{multi_series_forecasters_with_levels}. If the forecaster is of "
            f"type `ForecasterDirectMultiVariate`, this argument is ignored."
        )

    if type(forecaster).__name__ == 'ForecasterDirectMultiVariate':
        if levels and levels != forecaster.level and levels != [forecaster.level]:
            warnings.warn(
                f"`levels` argument have no use when the forecaster is of type "
                f"`ForecasterDirectMultiVariate`. The level of this forecaster "
                f"is '{forecaster.level}', to predict another level, change "
                f"the `level` argument when initializing the forecaster. \n",
                IgnoredArgumentWarning,
                stacklevel = _find_stack_level()
            )
        levels = [forecaster.level]
    else:
        if levels is None:
            # Forecaster could be untrained, so self.series_col_names cannot be used.
            if isinstance(series, pd.DataFrame):
                levels = list(series.columns)
            else:
                levels = list(series.keys())
        else:
            if isinstance(levels, str):
                levels = [levels]
            
            if isinstance(series, pd.DataFrame):
                available_levels = list(series.columns)
            else:
                available_levels = list(series.keys())

            if not set(levels).issubset(set(available_levels)):
                raise ValueError(
                    f"Levels {levels} not found in `series`, available levels are "
                    f"{available_levels}. Review `levels` argument."
                )

    return levels


def _extract_data_folds_multiseries(
    series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame],
    folds: list,
    span_index: pd.DatetimeIndex | pd.RangeIndex,
    window_size: int,
    exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
    dropna_last_window: bool = False,
    externally_fitted: bool = False
) -> Generator[
        tuple[
            pd.DataFrame | dict[str, pd.Series | pd.DataFrame],
            pd.DataFrame,
            list[str],
            pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None,
            pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None,
            list[list[int]]
        ],
        None,
        None
    ]:
    """
    Select the data from series and exog that corresponds to each fold created 
    using the split method of the OneStepAheadFold or TimeSeriesFold objects.

    Parameters
    ----------
    series : pandas DataFrame, dict
        Time series.
    folds : list
        Folds created using the split method of the OneStepAheadFold or 
        TimeSeriesFold objects.
    span_index : pandas DatetimeIndex, pandas RangeIndex
        Full index from the minimum to the maximum index among all series.
    window_size : int
        Size of the window needed to create the predictors. A series with
        `window_size` values or fewer in the training set of a fold is left
        out of that fold, since `fit` needs more than `window_size` values to
        create at least one training row.
    exog : pandas Series, pandas DataFrame, dict, default None
        Exogenous variables.
    dropna_last_window : bool, default False
        If `True`, drop the columns of the last window that have NaN values.
    externally_fitted : bool, default False
        Flag indicating whether the forecaster is already trained. Only used when 
        `initial_train_size` is None and `refit` is False.

    Yield
    -----
    series_train : pandas DataFrame, dict
        Time series corresponding to the training set of the fold.
    series_last_window: pandas DataFrame
        Time series corresponding to the last window of the fold.
    levels_last_window: list
        Levels of the time series present in the last window of the fold.
    exog_train: pandas Series, pandas DataFrame, dict, None
        Exogenous variable corresponding to the training set of the fold.
    exog_test: pandas Series, pandas DataFrame, dict, None
        Exogenous variable corresponding to the test set of the fold.
    fold: list
        List containing the indexes of that fold.

    """

    is_series_dict = isinstance(series, dict)
    is_exog_dict = isinstance(exog, dict)

    for fold in folds:
        train_iloc_start       = fold[1][0]
        train_iloc_end         = fold[1][1]
        last_window_iloc_start = fold[2][0]
        last_window_iloc_end   = fold[2][1]
        test_iloc_start        = fold[3][0]
        test_iloc_end          = fold[3][1]

        if is_series_dict or is_exog_dict:
            # Subtract 1 to the iloc indexes to get the loc indexes
            train_loc_start       = span_index[train_iloc_start]
            train_loc_end         = span_index[train_iloc_end - 1]
            last_window_loc_start = span_index[last_window_iloc_start]
            last_window_loc_end   = span_index[last_window_iloc_end - 1]
            test_loc_start        = span_index[test_iloc_start]
            test_loc_end          = span_index[test_iloc_end - 1]

        if not is_series_dict:
            series_train = series.iloc[train_iloc_start:train_iloc_end, ]

            series_to_drop = []
            for col in series_train.columns:
                if series_train[col].isna().to_numpy().all():
                    series_to_drop.append(col)
                else:
                    first_valid_index = series_train[col].first_valid_index()
                    last_valid_index = series_train[col].last_valid_index()
                    if (
                        len(series_train[col].loc[first_valid_index:last_valid_index])
                        <= window_size
                    ):
                        series_to_drop.append(col)

            series_last_window = series.iloc[
                last_window_iloc_start:last_window_iloc_end,
            ]
            
            series_train = series_train.drop(columns=series_to_drop)
            if not externally_fitted:
                series_last_window = series_last_window.drop(columns=series_to_drop)
        else:
            series_train = {}
            series_last_window = {}
            for k, v in series.items():
                v_train = v.loc[train_loc_start:train_loc_end]
                if not v_train.isna().to_numpy().all():
                    first_valid_index = v_train.first_valid_index()
                    last_valid_index  = v_train.last_valid_index()
                    if first_valid_index is not None and last_valid_index is not None:
                        v_train = v_train.loc[first_valid_index : last_valid_index]
                        if len(v_train) > window_size:
                            series_train[k] = v_train
                
                v_last_window = v.loc[last_window_loc_start:last_window_loc_end]
                if ((externally_fitted or k in series_train) and len(v_last_window) >= window_size):
                    series_last_window[k] = v_last_window

            series_last_window = pd.DataFrame(series_last_window)

        if dropna_last_window:
            series_last_window = series_last_window.dropna(axis=1, how="any")
        
        levels_last_window = list(series_last_window.columns)

        if exog is not None:
            if not is_exog_dict:
                exog_train = exog.iloc[train_iloc_start:train_iloc_end, ]
                exog_test = exog.iloc[test_iloc_start:test_iloc_end, ]
            else:
                exog_train = {}
                exog_test = {}
                for k, v in exog.items():
                    if v is None:
                        exog_train[k] = None
                        exog_test[k] = None
                    else:
                        v_train = v.loc[train_loc_start:train_loc_end]
                        if len(v_train) > 0:
                            exog_train[k] = v_train
                        if externally_fitted or k in exog_train:
                            v_test = v.loc[test_loc_start:test_loc_end]
                            if not v_test.empty:
                                exog_test[k] = v_test
        else:
            exog_train = None
            exog_test = None

        yield series_train, series_last_window, levels_last_window, exog_train, exog_test, fold


def _select_fit_folds_multiseries(
    folds: list,
    levels_predict_folds: list[list[str]]
) -> list[bool]:
    """
    Select the folds of a multi-series backtesting in which the forecaster is
    fitted. A fold with refit (`fold[5]` is `True`) trains the model that
    predicts it and the following folds without refit. That model is only
    fitted if at least one of these folds has a level to predict, so no
    forecaster is trained with an empty training set (no series with more
    than `window_size` values) or only to be discarded.

    Parameters
    ----------
    folds : list
        Folds created using the split method of the TimeSeriesFold object,
        before the initial fit sets `fold[5]` of the first fold to `False`.
    levels_predict_folds : list
        Levels predicted in each fold.

    Returns
    -------
    fit_folds : list
        `True` for the folds in which the forecaster is fitted.

    """

    fit_folds = [False] * len(folds)
    refit_fold = None
    for i, (fold, levels_predict) in enumerate(zip(folds, levels_predict_folds)):
        if fold[5] is True:
            refit_fold = i
        if refit_fold is not None and levels_predict:
            fit_folds[refit_fold] = True

    return fit_folds


def _warn_levels_not_predicted_multiseries(
    forecaster: object,
    series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame],
    folds: list,
    span_index: pd.DatetimeIndex | pd.RangeIndex,
    levels: list[str],
    levels_predict_folds: list[list[str]],
    levels_train_folds: list[list[str]] | None = None
) -> None:
    """
    Warn about the levels that are not predicted in a fold of the multi-series
    backtesting although they have observed values in its test set, so their
    metrics, and the aggregated metrics, do not include that fold. The levels
    are grouped by the reason why they are not predicted:

    - The level is not in the training set of the fold: it has `window_size`
    observations or fewer, so the forecaster is not trained with it.
    - The level is in the training set of the fold, or the training set is not
    used, but its last window cannot be used to predict: it has missing values
    and the estimator does not support them natively or the forecaster uses
    differentiation. For `ForecasterFoundation`, its context window has no
    observed values.

    A level without observed values in the test set of a fold (for example, a
    series that starts after it) is not reported, there is nothing to evaluate.
    A fold in which no level is predicted is skipped (no predictions). If a
    level has observed values in its test set, the fold is also listed.

    Parameters
    ----------
    forecaster : object
        Forecaster used in the backtesting.
    series : pandas DataFrame, dict
        Time series.
    folds : list
        Folds created using the split method of the TimeSeriesFold object.
    span_index : pandas DatetimeIndex, pandas RangeIndex
        Full index from the minimum to the maximum index among all series.
    levels : list
        Levels to predict.
    levels_predict_folds : list
        Levels predicted in each fold.
    levels_train_folds : list, default None
        Levels of the training set of each fold. If `None`, the training set
        is not used (the forecaster is fitted before the backtesting, or it is
        not trained) and every level not predicted is attributed to the last
        window.

    Returns
    -------
    None

    """

    if levels_train_folds is None:
        levels_train_folds = [None] * len(folds)

    not_predicted = {'train': {}, 'last_window': {}}
    skipped_folds = []
    for fold, levels_train, levels_predict in zip(
        folds, levels_train_folds, levels_predict_folds
    ):
        levels_train = set(levels_train) if levels_train is not None else None
        levels_predict = set(levels_predict)
        test_no_gap_loc_start = span_index[fold[4][0]]
        test_no_gap_loc_end   = span_index[fold[4][1] - 1]
        for level in levels:
            if level in levels_predict:
                continue
            y_test = series[level].loc[test_no_gap_loc_start:test_no_gap_loc_end]
            if not y_test.notna().to_numpy().any():
                continue
            reason = (
                'train' if levels_train is not None and level not in levels_train
                else 'last_window'
            )
            not_predicted[reason].setdefault(level, []).append(fold[0])
            if not levels_predict and fold[0] not in skipped_folds:
                skipped_folds.append(fold[0])

    if not not_predicted['train'] and not not_predicted['last_window']:
        return

    if type(forecaster).__name__ == 'ForecasterFoundation':
        reason_last_window = "No observed values in the context window"
    elif forecaster.differentiation_max is not None:
        reason_last_window = (
            "Missing values in the last window, they are not allowed with "
            "differentiation"
        )
    else:
        reason_last_window = (
            f"Missing values in the last window, "
            f"{type(forecaster.estimator).__name__} does not support them"
        )
    reason_messages = {
        'train': (
            f"Too short for the `window_size` ({forecaster.window_size}) in the "
            f"training set"
        ),
        'last_window': reason_last_window
    }

    levels_predicted = set(chain(*levels_predict_folds))
    lines = []
    for reason, levels_folds in not_predicted.items():
        if levels_folds:
            levels_info = [
                f"{level!r} (never predicted)" if level not in levels_predicted
                else f"{level!r} (folds {', '.join(map(str, fold_numbers))})"
                for level, fold_numbers in levels_folds.items()
            ]
            lines.append(f"  - {reason_messages[reason]}: {', '.join(levels_info)}.")

    blocks = [
        f"{LEVELS_NOT_PREDICTED_WARNING}, so their metrics, and the aggregated "
        f"metrics, do not include those folds. Levels never predicted have NaN "
        f"metrics and are left out of the aggregated metrics.",
        "Levels not predicted:\n" + "\n".join(lines)
    ]
    if skipped_folds:
        blocks.append(
            f"Folds skipped, no level can be predicted in them: "
            f"{', '.join(map(str, skipped_folds))}."
        )

    warnings.warn("\n\n".join(blocks), MissingValuesWarning)


def _get_interval_bound_quantiles(
    interval: float | list[float] | tuple[float, ...] | str | object | None
) -> list[float] | None:
    """
    Quantiles of the lower and upper bounds of the prediction interval defined
    by `interval`: a float (nominal coverage), or a list or tuple of two
    quantiles, optionally with the median (0.5). Any other value (more
    quantiles, 'bootstrapping', a distribution or None) does not define a
    single interval and returns None.

    Parameters
    ----------
    interval : float, list, tuple, str, object, None
        Argument `interval` of the backtesting functions (or `quantiles` in
        `backtesting_foundation`).

    Returns
    -------
    bound_quantiles : list, None
        Quantiles of the lower and upper bounds, sorted, or None.

    """

    if isinstance(interval, float):
        return [0.5 - interval / 2, 0.5 + interval / 2]

    if isinstance(interval, (list, tuple)):
        bound_quantiles = list(interval)
        if len(bound_quantiles) == 3 and 0.5 in bound_quantiles:
            bound_quantiles.remove(0.5)
        if len(bound_quantiles) == 2:
            return sorted(bound_quantiles)

    return None


def _get_interval_columns(
    columns: pd.Index | list[str],
    interval: float | list[float] | tuple[float, ...] | str | object | None
) -> list[str] | None:
    """
    Columns of the backtesting predictions with the lower and upper bounds of
    the prediction interval, used to calculate the interval metrics
    (`INTERVAL_METRIC_NAMES`): `lower_bound` and `upper_bound` if they exist,
    otherwise the quantile columns (`q_<quantile>`) of the bounds defined by
    `interval` (see `_get_interval_bound_quantiles`).

    Parameters
    ----------
    columns : pandas Index, list
        Columns of the backtesting predictions.
    interval : float, list, tuple, str, object, None
        Argument `interval` of the backtesting functions (or `quantiles` in
        `backtesting_foundation`).

    Returns
    -------
    interval_columns : list, None
        Names of the lower and upper bound columns, or None if the predictions
        do not define a single interval.

    """

    if {'lower_bound', 'upper_bound'}.issubset(columns):
        return ['lower_bound', 'upper_bound']

    bound_quantiles = _get_interval_bound_quantiles(interval)
    if bound_quantiles is not None:
        interval_columns = [f"q_{q}" for q in bound_quantiles]
        if set(interval_columns).issubset(columns):
            return interval_columns

    return None


def _calculate_interval_metrics(
    y_true: np.ndarray | pd.Series,
    lower_bound: np.ndarray | pd.Series,
    upper_bound: np.ndarray | pd.Series,
) -> list[float]:
    """
    Calculate the coverage and the mean width of a prediction interval
    defined by its lower and upper bounds.

    Rows with a missing value in `y_true`, `lower_bound` or `upper_bound`
    are ignored. If no row is left, both metrics are NaN.

    Parameters
    ----------
    y_true : numpy ndarray, pandas Series
        True values of the target variable.
    lower_bound : numpy ndarray, pandas Series
        Lower bound of the interval.
    upper_bound : numpy ndarray, pandas Series
        Upper bound of the interval.

    Returns
    -------
    interval_metrics : list
        Coverage (share of true values inside the interval, bounds included)
        and mean width (mean of `upper_bound - lower_bound`, in the units of
        the series), in the order of `INTERVAL_METRIC_NAMES`.

    """

    y_true = np.asarray(y_true, dtype=float)
    lower_bound = np.asarray(lower_bound, dtype=float)
    upper_bound = np.asarray(upper_bound, dtype=float)

    mask = ~(np.isnan(y_true) | np.isnan(lower_bound) | np.isnan(upper_bound))
    if not mask.any():
        return [np.nan, np.nan]

    coverage = calculate_coverage(
                   y_true      = y_true[mask],
                   lower_bound = lower_bound[mask],
                   upper_bound = upper_bound[mask]
               )
    interval_width = np.mean(upper_bound[mask] - lower_bound[mask])

    return [float(coverage), float(interval_width)]


def _count_evaluated(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series
) -> int:
    """
    Number of points that enter the backtesting metrics: those with a true
    value and a prediction, both not missing. The predictions must be those
    used for the metrics, after removing the repeated dates of overlapping
    folds (`fold_stride < steps`).

    Parameters
    ----------
    y_true : numpy ndarray, pandas Series
        True values of the target variable.
    y_pred : numpy ndarray, pandas Series
        Predicted values, aligned by position with `y_true`.

    Returns
    -------
    n_evaluated : int
        Number of points evaluated.

    """

    is_evaluated = pd.notna(np.asarray(y_true)) & pd.notna(np.asarray(y_pred))

    return int(is_evaluated.sum())


def _get_y_true_multiseries(
    series: pd.DataFrame | dict[str, pd.Series],
    index: pd.MultiIndex
) -> np.ndarray:
    """
    True values of the multi-series backtesting predictions: the value of
    the series `level` at the date `idx` of each pair of `index`. Pairs
    without an observed value (missing values, or dates outside the series)
    are NaN.

    Parameters
    ----------
    series : pandas DataFrame, dict
        Time series, in wide format or as a dict of pandas Series.
    index : pandas MultiIndex
        Index of the predictions, with levels `idx` (date) and `level`
        (series). It may include repeated pairs (overlapping folds).

    Returns
    -------
    y_true : numpy ndarray
        True values, aligned by position with `index`.

    """

    # NOTE: The codes of the MultiIndex are used to group the rows by series,
    # so the dates and the names of the series of every row are not
    # materialized. Each series is looked up only at the dates of its rows:
    # time and memory grow with the number of rows, also with series that
    # only cover part of the test set.
    position_idx = index.names.index('idx')
    position_level = index.names.index('level')
    dates = index.levels[position_idx]
    date_codes = index.codes[position_idx]
    levels = index.levels[position_level]
    level_codes = index.codes[position_level]

    y_true = np.full(len(index), np.nan)
    rows_by_level = np.split(
        np.argsort(level_codes, kind='stable'),
        np.cumsum(np.bincount(level_codes, minlength=len(levels)))[:-1]
    )
    for level, rows in zip(levels, rows_by_level):
        if rows.size == 0:
            continue
        level_series = series[level]
        locations = level_series.index.get_indexer(dates[date_codes[rows]])
        is_found = locations >= 0
        y_true[rows[is_found]] = level_series.to_numpy(dtype=float)[locations[is_found]]

    return y_true


def _check_nan_predictions_backtesting(
    y_true: pd.Series,
    y_pred: pd.Series,
    estimator_id: str | None = None,
    differentiation: int | None = None
) -> None:
    """
    Raise an error if a backtesting prediction of a single series is NaN at a
    date with an observed value. The metrics cannot be calculated with it, and
    leaving it out would calculate them with fewer values than predicted.

    Parameters
    ----------
    y_true : pandas Series
        True values of the series, aligned by position with `y_pred`.
    y_pred : pandas Series
        Backtesting predictions.
    estimator_id : str, default None
        Identifier of the estimator that made the predictions, included in
        the message when the forecaster has several estimators.
    differentiation : int, default None
        Order of the differentiation of the forecaster. If not `None`, the
        message also includes a missing value among the last values of the
        series before a fold, from which the differentiation is reverted, as
        a cause.

    Returns
    -------
    None

    """

    is_nan_prediction = y_pred.isna().to_numpy() & y_true.notna().to_numpy()
    if is_nan_prediction.any():
        of_estimator = (
            f" of estimator '{estimator_id}'" if estimator_id is not None else ""
        )
        dates = y_pred.index[is_nan_prediction]
        nan_predictors = (
            "a missing value in the series or in `exog`, used as a predictor of "
            "these dates, with an estimator that does not accept NaN (for "
            "example, a linear model)"
        )
        if differentiation is None:
            cause = f"The most likely cause is {nan_predictors}"
            remedy = "Fill the missing values, or use an estimator that accepts them"
        else:
            # NOTE: The differentiation is reverted from the last values of the
            # series, which no predictor reads, so a missing one gives NaN
            # predictions with any estimator.
            last_values = (
                "a missing last value of the series" if differentiation == 1
                else f"a missing value among the last {differentiation} values "
                     f"of the series"
            )
            cause = (
                f"The most likely causes are {nan_predictors}, or {last_values} "
                f"before the fold, from which the differentiation is reverted "
                f"(with any estimator)"
            )
            remedy = (
                "Fill the missing values, or, in the first case, use an "
                "estimator that accepts them"
            )
        raise ValueError(
            f"{len(dates)} backtesting predictions{of_estimator} are NaN at "
            f"dates with observed values (first: {dates[0]}), so the metrics "
            f"cannot be calculated. {cause}. {remedy}, for example "
            f"HistGradientBoostingRegressor or LGBMRegressor."
        )


def _get_y_train_and_scales(
    y: pd.Series,
    origins: np.ndarray,
    metrics: list[Callable]
) -> tuple[pd.Series | None, np.ndarray | None, np.ndarray | None]:
    """
    Values that the metrics need from the training data, given the position
    in `y` of the forecast origin of each point evaluated (the end of the last
    window of its fold in backtesting, and the position of the point itself in
    one-step-ahead validation).

    - `mean_absolute_scaled_error` and `root_mean_squared_scaled_error` scale
    the error of each point with the error of the one-step naive forecast over
    the observations before its origin, `y[:origin]`. The scale depends only on
    the series and the origin, not on the forecaster (`window_size`, `refit`,
    `fixed_train_size`), so the scaled metrics of two forecasters are
    comparable.
    - The rest of the metrics with the argument `y_train` receive the
    observations before the first origin.

    Parameters
    ----------
    y : pandas Series
        Values of the series, in the coordinates of `origins`.
    origins : numpy ndarray
        Position in `y` of the forecast origin of each point evaluated.
    metrics : list
        Metrics to calculate, already processed by `add_y_train_argument`.

    Returns
    -------
    y_train : pandas Series, None
        Observations before the first forecast origin, from the first observed
        value. `None` if no metric other than the scaled metrics needs it.
    scale_abs : numpy ndarray, None
        Mean absolute error of the naive forecast before the origin of each
        point. `None` if no metric is a scaled metric.
    scale_sq : numpy ndarray, None
        Mean squared error of the naive forecast before the origin of each
        point. `None` if no metric is a scaled metric.

    """

    if not _any_metric_needs_y_train(metrics):
        return None, None, None

    origins = np.asarray(origins, dtype=int)

    # NOTE: The scaled metrics of skforecast use the scales, not `y_train`,
    # so it is only built for the other metrics with the argument `y_train`.
    y_train = None
    if any(
        getattr(m, '_needs_y_train', True) and not _is_scaled_metric(m)
        for m in metrics
    ):
        first_origin = int(origins.min()) if origins.size > 0 else 0
        y_train = y.iloc[:first_origin]
        # NOTE: `y_train` starts at the first observed value (the span of
        # several series may start before it), and includes the intercepted NaNs.
        first_valid = y_train.first_valid_index()
        if first_valid is not None:
            y_train = y_train.loc[first_valid:]

    scale_abs = scale_sq = None
    if any(_is_scaled_metric(m) for m in metrics):
        scale_abs, scale_sq = _naive_scales(y=y.to_numpy(dtype=float), origins=origins)

    return y_train, scale_abs, scale_sq


def _calculate_metric_values(
    metrics: list[Callable],
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_train: pd.Series | np.ndarray | None = None,
    scale_abs: np.ndarray | None = None,
    scale_sq: np.ndarray | None = None
) -> tuple[list[float], int]:
    """
    Calculate the value of each metric. `mean_absolute_scaled_error` and
    `root_mean_squared_scaled_error` scale each error with the scale of its
    forecast origin (`scale_abs`, `scale_sq`), and the rest of the metrics
    receive `y_train`.

    Parameters
    ----------
    metrics : list
        Metrics to calculate, with the argument `y_train`.
    y_true : pandas Series, numpy ndarray
        True values.
    y_pred : pandas Series, numpy ndarray
        Predicted values.
    y_train : pandas Series, numpy ndarray, default None
        Observations before the first forecast origin, passed to the metrics
        with the argument `y_train` other than the scaled metrics.
    scale_abs : numpy ndarray, default None
        Mean absolute error of the naive forecast at the origin of each point.
        If `None`, the scaled metrics receive `y_train` as the other metrics.
    scale_sq : numpy ndarray, default None
        Mean squared error of the naive forecast at the origin of each point.

    Returns
    -------
    metric_values : list
        Value of each metric.
    n_undefined : int
        Number of points left out of the scaled metrics because their scale
        is missing or zero.

    """

    metric_values = []
    n_undefined = 0
    for m in metrics:
        if scale_abs is not None and _is_scaled_metric(m):
            value, n_undefined = _scaled_metric_per_origin(
                metric    = m,
                y_true    = y_true,
                y_pred    = y_pred,
                scale_abs = scale_abs,
                scale_sq  = scale_sq
            )
        else:
            value = m(y_true=y_true, y_pred=y_pred, y_train=y_train)
        metric_values.append(value)

    return metric_values, n_undefined


def _warn_undefined_scale(
    n_undefined: dict[str | None, tuple[int, int]]
) -> None:
    """
    Warn that some points are left out of `mean_absolute_scaled_error` and
    `root_mean_squared_scaled_error` because the scale of their forecast
    origin is missing or zero.

    Parameters
    ----------
    n_undefined : dict
        Points left out and points evaluated, `(n_undefined, n_evaluated)`,
        by level (`None` for a single series). Levels without points left out
        are ignored.

    Returns
    -------
    None

    """

    n_undefined = {k: v for k, v in n_undefined.items() if v[0] > 0}
    if not n_undefined:
        return

    if list(n_undefined) == [None]:
        n, total = n_undefined[None]
        where = f"{n} of the {total} points evaluated"
    else:
        where = ", ".join(
            f"'{level}' ({n} of {total})" for level, (n, total) in n_undefined.items()
        )
        where = f"some points evaluated of the levels {where}"
    warnings.warn(
        f"The scale of `mean_absolute_scaled_error` and "
        f"`root_mean_squared_scaled_error` (the error of the one-step naive "
        f"forecast over the observations before the forecast origin) is zero "
        f"or missing for {where}, so they are left out of these metrics. It "
        f"happens when the series is constant, or has no two consecutive "
        f"observed values, before the forecast origin. If no point is left, "
        f"the metric is NaN.",
        RuntimeWarning,
        stacklevel = _find_stack_level()
    )


def _calculate_metrics_backtesting(
    predictions: pd.DataFrame,
    metrics: list[Callable],
    y_train: pd.Series | None = None,
    interval_columns: list[str] | None = None,
    pred_column: str = 'pred',
    estimator_id: str | None = None,
    differentiation: int | None = None,
    scale_abs: np.ndarray | None = None,
    scale_sq: np.ndarray | None = None
) -> list[float]:
    """
    Calculate the backtesting metrics of a single series: the metrics, the
    interval metrics (`INTERVAL_METRIC_NAMES`) if `interval_columns` is given,
    and the number of points evaluated (`N_EVALUATED_NAME`), in this order.

    The dates without an observed value (`y_true` NaN) are left out of the
    metrics, as in the multi-series backtesting. A NaN prediction at a date
    with an observed value raises an error first (see
    `_check_nan_predictions_backtesting`), so the points evaluated are the
    dates with an observed value.

    Parameters
    ----------
    predictions : pandas DataFrame
        Backtesting predictions used for the metrics, without the repeated
        dates of overlapping folds, and of a single estimator. Columns
        `y_true` and `pred_column`, and the columns in `interval_columns`.
        Its rows are not modified.
    metrics : list
        Metrics to calculate, with the argument `y_train`.
    y_train : pandas Series, default None
        Observations before the first forecast origin, passed to the metrics
        with the argument `y_train` other than `mean_absolute_scaled_error`
        and `root_mean_squared_scaled_error`.
    interval_columns : list, default None
        Names of the lower and upper bound columns of the prediction interval.
        If `None`, the interval metrics are not calculated.
    pred_column : str, default 'pred'
        Name of the column with the point predictions.
    estimator_id : str, default None
        Identifier of the estimator that made the predictions, included in
        the error of NaN predictions when the forecaster has several
        estimators.
    differentiation : int, default None
        Order of the differentiation of the forecaster, used in the error of
        NaN predictions.
    scale_abs : numpy ndarray, default None
        Scale of `mean_absolute_scaled_error` for each row of `predictions`:
        the mean absolute error of the one-step naive forecast before its
        forecast origin (see `_get_y_train_and_scales`). If `None`, the
        scaled metrics receive `y_train`.
    scale_sq : numpy ndarray, default None
        Scale of `root_mean_squared_scaled_error` for each row of
        `predictions`: the mean squared error of the one-step naive forecast
        before its forecast origin.

    Returns
    -------
    metric_values : list
        Value of each metric, then the coverage and the mean width of the
        interval if `interval_columns` is given, and the number of points
        evaluated.

    """

    y_true = predictions['y_true']
    y_pred = predictions[pred_column]
    _check_nan_predictions_backtesting(
        y_true          = y_true,
        y_pred          = y_pred,
        estimator_id    = estimator_id,
        differentiation = differentiation
    )

    is_observed = y_true.notna().to_numpy()
    if not is_observed.any():
        raise ValueError(
            "The series has no observed values in the test set of the folds, "
            "so the metrics cannot be calculated. Review `initial_train_size` "
            "of `cv`, or the span of the series."
        )
    y_true = y_true.loc[is_observed]
    y_pred = y_pred.loc[is_observed]
    if scale_abs is not None:
        scale_abs = np.asarray(scale_abs)[is_observed]
        scale_sq = np.asarray(scale_sq)[is_observed]

    metric_values, n_undefined = _calculate_metric_values(
                                     metrics   = metrics,
                                     y_true    = y_true,
                                     y_pred    = y_pred,
                                     y_train   = y_train,
                                     scale_abs = scale_abs,
                                     scale_sq  = scale_sq
                                 )
    _warn_undefined_scale({None: (n_undefined, len(y_true))})
    if interval_columns is not None:
        metric_values.extend(
            _calculate_interval_metrics(
                y_true      = y_true,
                lower_bound = predictions[interval_columns[0]].loc[is_observed],
                upper_bound = predictions[interval_columns[1]].loc[is_observed]
            )
        )
    metric_values.append(_count_evaluated(y_true=y_true, y_pred=y_pred))

    return metric_values


def _calculate_metrics_backtesting_multiseries(
    series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame],
    predictions: pd.DataFrame,
    folds: list[list[int]] | tqdm,
    span_index: pd.DatetimeIndex | pd.RangeIndex,
    metrics: list,
    levels: list,
    add_aggregated_metric: bool = True
) -> pd.DataFrame:
    """   
    Calculate metrics for each level and also for all levels aggregated using
    average, weighted average or pooling.

    - 'average': the average (arithmetic mean) of all levels.
    - 'weighted_average': the average of the metrics weighted by the number of
    points evaluated in each level (`n_evaluated`).
    - 'pooling': the values of all levels are pooled and then the metric is
    calculated.

    If `predictions` includes the columns `lower_bound` and `upper_bound`, the
    coverage and the mean width of the interval (`INTERVAL_METRIC_NAMES`) are
    added after the metrics, for each level and aggregated in the same way.

    The last column, `n_evaluated` (`N_EVALUATED_NAME`), is the number of
    points that enter the metrics of each level: those with a true value and
    a prediction, both not missing. Its aggregated value is the sum over the
    levels.

    `mean_absolute_scaled_error` and `root_mean_squared_scaled_error` scale
    the error of each point with the error of the one-step naive forecast of
    its series over the observations before its forecast origin (the end of
    the last window of its fold). Their pooled value is the mean of the scaled
    errors of all the points of all the levels (the root of the mean of the
    squared scaled errors for the RMSSE). The rest of the metrics with the
    argument `y_train` receive the observations of the series before the first
    forecast origin.

    Parameters
    ----------
    series : pandas DataFrame, dict
        Series data used for backtesting. Only used for the metrics with the
        argument `y_train`.
    predictions : pandas DataFrame
        MultiIndex (`idx`, `level`) DataFrame generated during the backtesting
        multiseries process, without the repeated dates of overlapping folds.
        Columns `fold` (fold of each prediction), `y_true` (true values) and
        `y_pred` (point predictions), and optionally `lower_bound` and
        `upper_bound`.
    folds : list, tqdm
        Folds created during the backtesting process.
    span_index : pandas DatetimeIndex, pandas RangeIndex
        Full index from the minimum to the maximum index among all series.
    metrics : list
        List of metrics to calculate.
    levels : list
        Levels to calculate the metrics.
    add_aggregated_metric : bool, default True
        If `True`, and multiple series (`levels`) are predicted, the aggregated
        metrics (average, weighted average and pooled) are also returned.

        - 'average': the average (arithmetic mean) of all levels.
        - 'weighted_average': the average of the metrics weighted by the number of
        points evaluated in each level.
        - 'pooling': the values of all levels are pooled and then the metric is
        calculated.

    Returns
    -------
    metrics_levels : pandas DataFrame
        Value(s) of the metric(s) and number of points evaluated.
    
    """

    metric_names = [m.__name__ for m in metrics]
    needs_y_train = _any_metric_needs_y_train(metrics)
    add_interval_metrics = {'lower_bound', 'upper_bound'}.issubset(predictions.columns)
    output_names = (
        metric_names + INTERVAL_METRIC_NAMES if add_interval_metrics else metric_names
    )

    levels_in_predictions = predictions.index.get_level_values('level').unique()
    predictions = predictions.swaplevel()

    # NOTE: The predictions of the dates without observed values are expected
    # to be NaN. A NaN prediction of a date with an observed value (the
    # forecaster could not predict it) is left out of the metrics, which are
    # then calculated with fewer values than predicted, so it is warned.
    is_nan_prediction = (
        predictions['y_true'].notna() & predictions['y_pred'].isna()
    )
    if is_nan_prediction.any():
        n_nan_predictions = (
            is_nan_prediction[is_nan_prediction]
            .groupby(level='level', sort=False)
            .size()
            .to_dict()
        )
        warnings.warn(
            f"Some predictions are NaN at dates with observed values, so the "
            f"metrics leave them out. NaN predictions per series: "
            f"{n_nan_predictions} ({int(is_nan_prediction.sum())} in total). "
            f"The forecaster could not predict these dates, for example because "
            f"of missing values in the last window or in `exog`.",
            MissingValuesWarning
        )

    # NOTE: Rows are dropped only by `y_true` and `y_pred`, so the metrics are
    # the same with and without the interval bounds. The rows left are the
    # points evaluated (`n_evaluated`).
    y_true_y_pred = predictions.dropna(
        axis=0, how="any", subset=['y_true', 'y_pred']
    )

    y_true_y_pred_grouped = (
        y_true_y_pred
        .reset_index(level="level")
        .groupby(by="level", sort=False, as_index=False)
    )

    # NOTE: The scale of each point depends on its series and its forecast
    # origin (the end of the last window of its fold, a position of
    # `span_index`), so it is calculated for every point evaluated, aligned
    # with the rows of `y_true_y_pred`.
    y_train_levels = {}
    scale_abs = scale_sq = None
    if needs_y_train:
        origin_by_fold = {fold[0]: fold[2][1] for fold in folds}
        origins = y_true_y_pred['fold'].map(origin_by_fold).to_numpy()
        is_scaled = any(_is_scaled_metric(m) for m in metrics)
        if is_scaled:
            scale_abs = np.full(len(y_true_y_pred), np.nan)
            scale_sq = np.full(len(y_true_y_pred), np.nan)
        for level, positions in y_true_y_pred_grouped.indices.items():
            y_level = series[level]
            if isinstance(y_level, pd.DataFrame):
                y_level = y_level.iloc[:, 0]
            y_level = y_level.reindex(span_index)
            y_train_level, scale_abs_level, scale_sq_level = _get_y_train_and_scales(
                y       = y_level,
                origins = origins[positions],
                metrics = metrics
            )
            y_train_levels[level] = y_train_level
            if is_scaled:
                scale_abs[positions] = scale_abs_level
                scale_sq[positions] = scale_sq_level

    metrics_levels = []
    n_evaluated_levels = []
    n_undefined_levels = {}
    for level in levels:
        if level in levels_in_predictions and level in y_true_y_pred_grouped.groups:
            group = y_true_y_pred_grouped.get_group(level)
            positions = y_true_y_pred_grouped.indices[level]
            y_true = group['y_true']
            y_pred = group['y_pred']
            metrics_level, n_undefined = _calculate_metric_values(
                metrics   = metrics,
                y_true    = y_true,
                y_pred    = y_pred,
                y_train   = y_train_levels.get(level),
                scale_abs = scale_abs[positions] if scale_abs is not None else None,
                scale_sq  = scale_sq[positions] if scale_sq is not None else None
            )
            n_undefined_levels[level] = (n_undefined, len(group))
            if add_interval_metrics:
                metrics_level.extend(
                    _calculate_interval_metrics(
                        y_true      = y_true,
                        lower_bound = group['lower_bound'],
                        upper_bound = group['upper_bound']
                    )
                )
            metrics_levels.append(metrics_level)
            n_evaluated_levels.append(len(group))
        else:
            metrics_levels.append([None for _ in output_names])
            n_evaluated_levels.append(0)

    _warn_undefined_scale(n_undefined_levels)

    metrics_levels = pd.DataFrame(
                         data    = metrics_levels,
                         columns = output_names
                     )
    metrics_levels.insert(0, 'levels', levels)
    metrics_levels[N_EVALUATED_NAME] = n_evaluated_levels

    if len(levels) < 2:
        add_aggregated_metric = False

    if add_aggregated_metric:

        # NOTE: The aggregated number of points evaluated is the sum over the
        # levels, the same for the three aggregations.
        n_evaluated = int(metrics_levels[N_EVALUATED_NAME].sum())

        # aggregation: average
        average = metrics_levels[output_names].mean(skipna=True)
        average = average.to_frame().transpose()
        average['levels'] = 'average'
        average[N_EVALUATED_NAME] = n_evaluated

        # aggregation: weighted_average
        # NOTE: The weights are the points evaluated in each level. A level
        # without points evaluated (not predicted, or predicted without
        # observed values in its test set) has no metric, so it is left out of
        # the weighted average, as it is of the average and the pooling.
        metrics_levels_no_missing = (
            metrics_levels.loc[metrics_levels[N_EVALUATED_NAME] > 0]
        )
        weighted_averages = {}
        for col in output_names:
            weighted_averages[col] = (
                np.average(
                    metrics_levels_no_missing[col],
                    weights = metrics_levels_no_missing[N_EVALUATED_NAME]
                )
                if not metrics_levels_no_missing.empty
                else np.nan
            )
        weighted_average = pd.DataFrame(weighted_averages, index=[0])
        weighted_average['levels'] = 'weighted_average'
        weighted_average[N_EVALUATED_NAME] = n_evaluated

        # aggregation: pooling
        # NOTE: The scaled metrics are pooled with the scale of each point
        # (the mean of the scaled errors of all the points). The rest of the
        # metrics with `y_train` receive the training values of all levels.
        y_true = y_true_y_pred.loc[:, 'y_true'].droplevel("level")
        y_pred = y_true_y_pred.loc[:, 'y_pred'].droplevel("level")
        y_train_values = [v for v in y_train_levels.values() if v is not None]
        y_train_pooled = pd.concat(y_train_values) if y_train_values else None
        pooled, _ = _calculate_metric_values(
                        metrics   = metrics,
                        y_true    = y_true,
                        y_pred    = y_pred,
                        y_train   = y_train_pooled,
                        scale_abs = scale_abs,
                        scale_sq  = scale_sq
                    )

        if add_interval_metrics:
            pooled.extend(
                _calculate_interval_metrics(
                    y_true      = y_true,
                    lower_bound = y_true_y_pred['lower_bound'],
                    upper_bound = y_true_y_pred['upper_bound']
                )
            )

        pooled = pd.DataFrame([pooled], columns=output_names)
        pooled['levels'] = 'pooling'
        pooled[N_EVALUATED_NAME] = n_evaluated

        metrics_levels = pd.concat(
            [metrics_levels, average, weighted_average, pooled],
            axis=0, ignore_index=True
        )

    return metrics_levels


def _predict_and_calculate_metrics_one_step_ahead_multiseries(
    forecaster: object,
    series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame],
    X_train: pd.DataFrame,
    y_train: pd.Series | dict[int, pd.Series],
    X_test: pd.DataFrame,
    y_test: pd.Series | dict[int, pd.Series],
    X_train_encoding: pd.Series | pd.Index,
    X_test_encoding: pd.Series | pd.Index,
    levels: list[str],
    metrics: list[str | Callable],
    add_aggregated_metric: bool = True,
    sample_weight: np.ndarray | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    return_predictions: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """   
    One-step-ahead predictions and metrics for each level and also for all levels
    aggregated using average, weighted average or pooling.
    Input matrices (X_train, y_train, X_train_encoding, X_test, y_test, X_test_encoding)
    should have been generated using the forecaster._train_test_split_one_step_ahead().

    - 'average': the average (arithmetic mean) of all levels.
    - 'weighted_average': the average of the metrics weighted by the number of
    points evaluated in each level.
    - 'pooling': the values of all levels are pooled and then the metric is
    calculated.

    Parameters
    ----------
    forecaster : object
        Forecaster model.
    series : pandas DataFrame, dict
        Series data used to train and test the forecaster, in their original
        scale. With `differentiation`, the values of each level at the test
        dates are the true values, from which the predictions are reverted one
        step ahead (see `_revert_differentiation_one_step_ahead`). The metrics
        with the argument `y_train` also use its values.
    X_train : pandas DataFrame
        Training matrix.
    y_train : pandas Series, dict
        Target values of the training set.
    X_test : pandas DataFrame
        Test matrix.
    y_test : pandas Series, dict
        Target values of the test set.
    X_train_encoding : pandas Series, pandas Index
        ForecasterRecursiveMultiSeries: series identifiers for each row of `X_train`.
        ForecasterDirectMultiVariate: Index of the training observations for step 1.
    X_test_encoding : pandas Series, pandas Index
        ForecasterRecursiveMultiSeries: series identifiers for each row of `X_test`.
        ForecasterDirectMultiVariate: Index of the test observations for step 1.
    levels : list
        Levels to calculate the metrics.
    metrics : list
        List of metrics to calculate.
    add_aggregated_metric : bool, default True
        If `True`, and multiple series (`levels`) are predicted, the aggregated
        metrics (average, weighted average and pooled) are also returned.

        - 'average': the average (arithmetic mean) of all levels.
        - 'weighted_average': the average of the metrics weighted by the number of
        points evaluated in each level.
        - 'pooling': the values of all levels are pooled and then the metric is
        calculated.
    sample_weight : numpy ndarray, default None
        Precomputed sample weights for training. `None` if no weight function
        is defined.
    fit_kwargs : dict, default None
        Precomputed keyword arguments for `estimator.fit`. Can be empty dict
        or `None`.
    return_predictions : bool, default True
        If `True`, return the predictions DataFrame. If `False`, return `None`
        instead of predictions to avoid unnecessary computation.

    Returns
    -------
    metrics_levels : pandas DataFrame
        Value(s) of the metric(s). The last column, `n_evaluated`
        (`N_EVALUATED_NAME`), is the number of points that enter the metrics
        of each level: those with a true value and a prediction, both not
        missing. Its aggregated value is the sum over the levels, as in
        `_calculate_metrics_backtesting_multiseries`.
    predictions : pandas DataFrame, None
        Value of predictions for each level. `None` if `return_predictions`
        is `False`.
    
    """

    # NOTE: All this checks can be deleted as they are done in the public
    # function that calls this private function.
    # if not isinstance(series, (pd.DataFrame, dict)):
    #     raise TypeError(
    #         "`series` must be a pandas DataFrame or a dictionary of pandas "
    #         "DataFrames."
    #     )
    # if not isinstance(X_train, pd.DataFrame):
    #     raise TypeError(f"`X_train` must be a pandas DataFrame. Got: {type(X_train)}")
    # if not isinstance(y_train, (pd.Series, dict)):
    #     raise TypeError(
    #         f"`y_train` must be a pandas Series or a dictionary of pandas Series. "
    #         f"Got: {type(y_train)}"
    #     )        
    # if not isinstance(X_test, pd.DataFrame):
    #     raise TypeError(f"`X_test` must be a pandas DataFrame. Got: {type(X_test)}")
    # if not isinstance(y_test, (pd.Series, dict)):
    #     raise TypeError(
    #         f"`y_test` must be a pandas Series or a dictionary of pandas Series. "
    #         f"Got: {type(y_test)}"
    #     )
    # if not isinstance(X_train_encoding, pd.Series):
    #     raise TypeError(
    #         f"`X_train_encoding` must be a pandas Series. Got: {type(X_train_encoding)}"
    #     )
    # if not isinstance(X_test_encoding, pd.Series):
    #     raise TypeError(
    #         f"`X_test_encoding` must be a pandas Series. Got: {type(X_test_encoding)}"
    #     )
    # if not isinstance(levels, list):
    #     raise TypeError(f"`levels` must be a list. Got: {type(levels)}")
    # if not isinstance(metrics, list):
    #     raise TypeError(f"`metrics` must be a list. Got: {type(metrics)}")
    # if not isinstance(add_aggregated_metric, bool):
    #     raise TypeError(
    #         f"`add_aggregated_metric` must be a boolean. Got: {type(add_aggregated_metric)}"
    #     )

    if fit_kwargs is None:
        fit_kwargs = {}

    metrics = [
        _get_metric(metric=m)
        if isinstance(m, str)
        else add_y_train_argument(m) 
        for m in metrics
    ]
    metric_names = [(m if isinstance(m, str) else m.__name__) for m in metrics]
    needs_y_train = _any_metric_needs_y_train(metrics)

    if isinstance(series[levels[0]].index, pd.DatetimeIndex):
        freq = series[levels[0]].index.freq
    else:
        freq = series[levels[0]].index.step

    # ForecasterDirectMultiVariate path (numpy)
    # ==========================================================================
    # X_train_encoding and X_test_encoding are pandas Index objects.
    if type(forecaster).__name__ == 'ForecasterDirectMultiVariate':
        
        # NOTE: Only step 1 is optimized in one-step-ahead validation.
        step = 1
        estimator = forecaster.estimators_[step]
        level = forecaster.level
        test_index = X_test_encoding

        if sample_weight is not None:
            estimator.fit(
                X=X_train, y=y_train,
                sample_weight=sample_weight, **fit_kwargs
            )
        else:
            estimator.fit(X=X_train, y=y_train, **fit_kwargs)

        y_true = y_test
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning
            )
            y_pred = estimator.predict(X_test).ravel()

        # NOTE: Each prediction is made one step ahead with the observed
        # values, so its forecast origin is its own position in the series.
        y_level = series[level]
        test_positions = _get_test_positions(y=y_level, test_index=test_index)

        differentiator = (
            forecaster.differentiator_[level]
            if forecaster.differentiation is not None
            else None
        )
        if differentiator is not None:
            y_true, y_pred = _revert_differentiation_one_step_ahead(
                                 y_observed  = y_level.to_numpy()[test_positions],
                                 diff_true   = y_test,
                                 diff_pred   = y_pred,
                                 transformer = (
                                     forecaster.transformer_series_[level]
                                     if forecaster.transformer_series is not None
                                     else None
                                 )
                             )
        elif forecaster.transformer_series is not None:
            transformer = forecaster.transformer_series_[level]
            y_true = transformer.inverse_transform(
                y_true.reshape(-1, 1)
            ).ravel()
            y_pred = transformer.inverse_transform(
                y_pred.reshape(-1, 1)
            ).ravel()

        y_train_metric, scale_abs, scale_sq = _get_y_train_and_scales(
                                                  y       = y_level,
                                                  origins = test_positions,
                                                  metrics = metrics
                                              )
        metrics_level, n_undefined = _calculate_metric_values(
                                         metrics   = metrics,
                                         y_true    = y_true,
                                         y_pred    = y_pred,
                                         y_train   = y_train_metric,
                                         scale_abs = scale_abs,
                                         scale_sq  = scale_sq
                                     )
        _warn_undefined_scale({level: (n_undefined, len(y_true))})

        metrics_levels = pd.DataFrame(
            data=[metrics_level], columns=metric_names
        )
        metrics_levels.insert(0, 'levels', [level])
        metrics_levels[N_EVALUATED_NAME] = _count_evaluated(
            y_true=y_true, y_pred=y_pred
        )

        if return_predictions:
            predictions = pd.DataFrame({level: y_pred}, index=test_index)
            predictions.index.name = None
            if isinstance(test_index, pd.DatetimeIndex):
                predictions = predictions.reindex(
                    _date_range_from_index(
                        index = predictions.index,
                        start = predictions.index[0],
                        end   = predictions.index[-1],
                        freq  = freq
                    )
                )
            else:
                predictions = predictions.reindex(
                    pd.RangeIndex(
                        start=min(test_index),
                        stop=max(test_index) + 1,
                        step=freq
                    )
                )
        else:
            predictions = None

        return metrics_levels, predictions

    # ForecasterRecursiveMultiSeries path (pandas)
    # ==========================================================================
    # X_train_encoding and X_test_encoding are series identifiers for each row 
    # of X_train and X_test, respectively.
    # NOTE: X_train and X_test are created once by `_train_test_split_one_step_ahead`
    # and reused by every candidate of the search. The cast below only copies
    # them for CatBoost, so it is checked that the estimator does not modify
    # X_train in place, which would change the metrics of the next candidates.
    feature_names = X_train.columns.to_list()
    X_train = cast_catboost_categorical_columns_dataframe(
        X=X_train, fit_kwargs=fit_kwargs,
        estimator=forecaster.estimator, feature_names=feature_names,
    )
    X_test = cast_catboost_categorical_columns_dataframe(
        X=X_test, fit_kwargs=fit_kwargs,
        estimator=forecaster.estimator, feature_names=feature_names,
    )

    X_train_rows = _copy_rows_to_check(X_train)

    if sample_weight is not None:
        forecaster.estimator.fit(
            X             = X_train,
            y             = y_train,
            sample_weight = sample_weight,
            **fit_kwargs
        )
    else:
        forecaster.estimator.fit(X=X_train, y=y_train, **fit_kwargs)

    _check_in_place_fit(X=X_train, X_rows=X_train_rows)

    predictions_per_level = pd.DataFrame(
        {
            'y_true': y_test,
            'y_pred': forecaster.estimator.predict(X_test),
            '_level_skforecast': X_test_encoding,
        },
        index=y_test.index,
    ).groupby('_level_skforecast')
    predictions_per_level = {key: group for key, group in predictions_per_level}

    # NOTE: Each prediction is made one step ahead with the observed values,
    # so its forecast origin is its own position in its series.
    series_per_level = {}
    test_positions_per_level = {}
    if needs_y_train or forecaster.differentiation is not None:
        for level in predictions_per_level:
            y_level = series[level]
            if isinstance(y_level, pd.DataFrame):
                y_level = y_level.iloc[:, 0]
            series_per_level[level] = y_level
            test_positions_per_level[level] = _get_test_positions(
                                                  y          = y_level,
                                                  test_index = predictions_per_level[level].index
                                              )

    for level, predictions_level in predictions_per_level.items():
        differentiator = (
            forecaster.differentiator_[level]
            if forecaster.differentiation is not None
            else None
        )
        if differentiator is not None:
            y_true, y_pred = _revert_differentiation_one_step_ahead(
                                 y_observed  = series_per_level[level].to_numpy()[
                                                   test_positions_per_level[level]
                                               ],
                                 diff_true   = predictions_level["y_true"].to_numpy(),
                                 diff_pred   = predictions_level["y_pred"].to_numpy(),
                                 transformer = (
                                     forecaster.transformer_series_[level]
                                     if forecaster.transformer_series is not None
                                     else None
                                 )
                             )
            predictions_level["y_true"] = y_true
            predictions_level["y_pred"] = y_pred
        elif forecaster.transformer_series is not None:
            transformer = forecaster.transformer_series_[level]
            predictions_level["y_true"] = transformer.inverse_transform(
                predictions_level[["y_true"]]
            )
            predictions_level["y_pred"] = transformer.inverse_transform(
                predictions_level[["y_pred"]]
            )

    y_train_per_level = {}
    scales_per_level = {}
    if needs_y_train:
        for level in predictions_per_level:
            y_train_level, scale_abs_level, scale_sq_level = _get_y_train_and_scales(
                y       = series_per_level[level],
                origins = test_positions_per_level[level],
                metrics = metrics
            )
            y_train_per_level[level] = y_train_level
            scales_per_level[level] = (scale_abs_level, scale_sq_level)

    metrics_levels = []
    n_evaluated_levels = []
    n_undefined_levels = {}
    for level in levels:
        if level in predictions_per_level:
            y_true = predictions_per_level[level].loc[:, 'y_true']
            y_pred = predictions_per_level[level].loc[:, 'y_pred']
            scale_abs, scale_sq = scales_per_level.get(level, (None, None))
            metrics_level, n_undefined = _calculate_metric_values(
                metrics   = metrics,
                y_true    = y_true,
                y_pred    = y_pred,
                y_train   = y_train_per_level.get(level),
                scale_abs = scale_abs,
                scale_sq  = scale_sq
            )
            n_undefined_levels[level] = (n_undefined, len(y_true))
            metrics_levels.append(metrics_level)
            n_evaluated_levels.append(_count_evaluated(y_true=y_true, y_pred=y_pred))
        else:
            metrics_levels.append([None for _ in metrics])
            n_evaluated_levels.append(0)

    _warn_undefined_scale(n_undefined_levels)

    metrics_levels = pd.DataFrame(
                         data    = metrics_levels,
                         columns = metric_names
                     )
    metrics_levels.insert(0, 'levels', levels)
    metrics_levels[N_EVALUATED_NAME] = n_evaluated_levels

    if len(levels) < 2:
        add_aggregated_metric = False

    if add_aggregated_metric:

        # NOTE: The aggregated number of points evaluated is the sum over the
        # levels, the same for the three aggregations.
        n_evaluated = int(metrics_levels[N_EVALUATED_NAME].sum())

        # aggregation: average
        average = metrics_levels[metric_names].mean(skipna=True)
        average = average.to_frame().transpose()
        average['levels'] = 'average'
        average[N_EVALUATED_NAME] = n_evaluated

        # aggregation: weighted_average
        # NOTE: Weighted by the points evaluated in each level, as in
        # `_calculate_metrics_backtesting_multiseries`.
        metrics_levels_no_missing = (
            metrics_levels.loc[metrics_levels[N_EVALUATED_NAME] > 0]
        )
        weighted_averages = {}
        for col in metric_names:
            weighted_averages[col] = (
                np.average(
                    metrics_levels_no_missing[col],
                    weights = metrics_levels_no_missing[N_EVALUATED_NAME]
                )
                if not metrics_levels_no_missing.empty
                else np.nan
            )
        weighted_average = pd.DataFrame(weighted_averages, index=[0])
        weighted_average['levels'] = 'weighted_average'
        weighted_average[N_EVALUATED_NAME] = n_evaluated

        # aggregation: pooling
        # NOTE: `X_test` includes every series of the forecaster, so only the
        # predictions of `levels` are pooled.
        # NOTE: The scaled metrics are pooled with the scale of each point
        # (the mean of the scaled errors of all the points).
        levels_pooled = [k for k in predictions_per_level if k in levels]
        predictions_pooled = pd.concat(
            [predictions_per_level[k] for k in levels_pooled]
        )
        y_train_pooled = scale_abs_pooled = scale_sq_pooled = None
        if needs_y_train:
            y_train_values = [
                y_train_per_level[k] for k in levels_pooled
                if y_train_per_level[k] is not None
            ]
            if y_train_values:
                y_train_pooled = pd.concat(y_train_values)
            if scales_per_level[levels_pooled[0]][0] is not None:
                scale_abs_pooled = np.concatenate(
                    [scales_per_level[k][0] for k in levels_pooled]
                )
                scale_sq_pooled = np.concatenate(
                    [scales_per_level[k][1] for k in levels_pooled]
                )

        pooled, _ = _calculate_metric_values(
                        metrics   = metrics,
                        y_true    = predictions_pooled['y_true'],
                        y_pred    = predictions_pooled['y_pred'],
                        y_train   = y_train_pooled,
                        scale_abs = scale_abs_pooled,
                        scale_sq  = scale_sq_pooled
                    )
        pooled = pd.DataFrame([pooled], columns=metric_names)
        pooled['levels'] = 'pooling'
        pooled[N_EVALUATED_NAME] = n_evaluated

        metrics_levels = pd.concat(
            [metrics_levels, average, weighted_average, pooled],
            axis=0, ignore_index=True
        )

    if return_predictions:
        predictions = (
            pd.concat(predictions_per_level.values())
            .loc[:, ["y_pred", "_level_skforecast"]]
            .pivot(columns="_level_skforecast", values="y_pred")
            .rename_axis(columns=None, index=None)
        )
        if isinstance(X_test.index, pd.DatetimeIndex):
            predictions = predictions.reindex(
                _date_range_from_index(
                    index = predictions.index,
                    start = predictions.index[0],
                    end   = predictions.index[-1],
                    freq  = freq
                )
            )
        else:
            predictions = predictions.reindex(
                pd.RangeIndex(
                    start=min(X_test.index), stop=max(X_test.index) + 1, step=freq
                )
            )
    else:
        predictions = None

    return metrics_levels, predictions


def _make_lags_hashable(lags, sentinel: object) -> int | tuple | None:
    """
    Convert a lags value into a hashable type suitable for use as a
    dictionary key in the split cache of bayesian search functions.

    Parameters
    ----------
    lags : int, list, numpy ndarray, range, dict, None, or sentinel
        Lag configuration to convert. Supported types:

        - `int`: returned as-is.
        - `list`, `numpy ndarray`, `range`: converted to `tuple`.
        - `dict`: converted to a sorted `tuple` of `(key, value)`
          pairs, where each value is itself converted to `tuple` when
          it is a list, ndarray, or range.
        - `None`: returned as-is (valid lags value in skforecast).
        - *sentinel*: returned as-is (indicates lags were not present in
          the search space).
    sentinel : object
        Sentinel object used to distinguish "lags not included in the
        search space" from `lags=None`. Each caller must pass its own
        `_SENTINEL` instance created with `object()`.

    Returns
    -------
    lags : int, tuple, None
        A hashable representation of `lags` that can be used as a
        dictionary key.
    
    """

    if lags is sentinel or lags is None:
        return lags
    if isinstance(lags, (list, np.ndarray, range)):
        return tuple(lags)
    if isinstance(lags, dict):
        return tuple(
            (k, tuple(v) if isinstance(v, (list, np.ndarray, range)) else v)
            for k, v in sorted(lags.items())
        )
    return lags


def _get_search_direction(metric: Callable, forecaster: object) -> str:
    """
    Get the direction in which a hyperparameter search optimizes a metric.
    It is inferred from the metric (see `_metric_greater_is_better`). If the
    direction of the metric is unknown, it is set from the task of the
    forecaster: `'minimize'` for regression and `'maximize'` for
    classification.

    Parameters
    ----------
    metric : Callable
        Metric used to rank the candidates of the search, as returned by
        `_get_metric` or `add_y_train_argument`.
    forecaster : object
        Forecaster model.

    Returns
    -------
    direction : str
        `'minimize'` or `'maximize'`.

    """

    greater_is_better = _metric_greater_is_better(metric)
    if greater_is_better is None:
        greater_is_better = (
            forecaster.__skforecast_tags__['forecaster_task'] != 'regression'
        )

    return 'maximize' if greater_is_better else 'minimize'


def _check_study_direction(study: object, metric: Callable) -> str:
    """
    Get the direction of an optuna study, which is the direction used to rank
    the trials of a bayesian search. A warning is raised if it contradicts the
    known direction of the metric (for example, `'minimize'` with
    `accuracy_score`), usually because of the `direction` passed in
    `kwargs_create_study`.

    Parameters
    ----------
    study : optuna Study
        Optuna study of the search.
    metric : Callable
        Metric optimized by the study.

    Returns
    -------
    direction : str
        `'minimize'` or `'maximize'`.

    """

    direction = study.direction.name.lower()
    greater_is_better = _metric_greater_is_better(metric)
    if greater_is_better is not None:
        metric_direction = 'maximize' if greater_is_better else 'minimize'
        if direction != metric_direction:
            warnings.warn(
                f"The optuna study direction is '{direction}', but greater values "
                f"of the metric `{getattr(metric, '__name__', metric)}` are "
                f"{'better' if greater_is_better else 'worse'}, so it should be "
                f"'{metric_direction}'. The search, the order of the results and "
                f"`return_best` follow the study direction. Remove `direction` "
                f"from `kwargs_create_study` to use the direction of the metric.",
                UserWarning,
                stacklevel = _find_stack_level()
            )

    return direction


def _select_metrics_search_multiseries(
    metrics: pd.DataFrame,
    metric_names: list[str],
    rows: list
) -> np.ndarray:
    """
    Values of the metrics of a multi-series backtesting used by the searches:
    the columns `metric_names` of the rows `rows` (aggregations, or a level if
    the metrics are not aggregated), flattened metric by metric and, within
    each metric, in the order of `rows`. It is the order of the columns
    `<metric>__<aggregation>` of the search results. Metrics are selected by
    name, so other columns of the metrics table are ignored.

    Parameters
    ----------
    metrics : pandas DataFrame
        Metrics table of `backtesting_forecaster_multiseries`,
        `backtesting_foundation` or
        `_predict_and_calculate_metrics_one_step_ahead_multiseries`, with the
        column `levels`.
    metric_names : list
        Names of the metrics, columns of `metrics`.
    rows : list
        Values of the column `levels` to select, in the order of the results.

    Returns
    -------
    metric_values : numpy ndarray
        Values of the metrics, of length `len(metric_names) * len(rows)`.

    """

    metric_values = (
        metrics
        .set_index('levels')
        .loc[rows, metric_names]
        .to_numpy()
        .T
        .ravel()
    )

    return metric_values


def _check_metric_names_search(metric_names: list[str]) -> None:
    """
    Check the names of the metrics of a search before evaluating any
    candidate: they must be unique, and `n_evaluated` (`N_EVALUATED_NAME`) is
    reserved for the number of points evaluated of the backtesting. Otherwise,
    the backtesting of every candidate would raise an error.

    Parameters
    ----------
    metric_names : list
        Names of the metrics.

    Returns
    -------
    None

    """

    if len(metric_names) != len(set(metric_names)):
        raise ValueError(
            "When `metric` is a `list`, each metric name must be unique."
        )
    if N_EVALUATED_NAME in metric_names:
        raise ValueError(
            f"The backtesting metrics include the column '{N_EVALUATED_NAME}' "
            f"with the number of points evaluated. Rename the metric "
            f"'{N_EVALUATED_NAME}' passed in `metric`."
        )


def _get_n_evaluated_search_multiseries(
    metrics: pd.DataFrame,
    row: str,
    levels: list[str]
) -> tuple[int, np.ndarray]:
    """
    Number of points evaluated of a candidate of a multi-series search: the
    column `n_evaluated` (`N_EVALUATED_NAME`) of the metrics table, in the row
    used for the metrics of the results (an aggregation, whose value is the
    sum over the levels, or the level if the metrics are not aggregated), and
    in the row of each level.

    Parameters
    ----------
    metrics : pandas DataFrame
        Metrics table of `backtesting_forecaster_multiseries` or
        `_predict_and_calculate_metrics_one_step_ahead_multiseries`, with the
        columns `levels` and `n_evaluated`.
    row : str
        Value of the column `levels` used for the metrics of the results.
    levels : list
        Levels of the search.

    Returns
    -------
    n_evaluated : int
        Number of points evaluated.
    n_evaluated_levels : numpy ndarray
        Number of points evaluated of each level, in the order of `levels`.

    """

    n_evaluated = metrics.set_index('levels')[N_EVALUATED_NAME]
    n_evaluated_levels = n_evaluated.loc[levels].to_numpy(dtype=int)
    n_evaluated = int(n_evaluated.loc[row])

    return n_evaluated, n_evaluated_levels


def _format_lags_warning(lags: Any) -> str:
    """
    Short representation of the lags of a candidate for the warnings of the
    searches: arrays with more than 6 lags show the first two and the last.

    Parameters
    ----------
    lags : Any
        Lags of the candidate: numpy ndarray, dict of them
        (`ForecasterDirectMultiVariate`), a label or `None`.

    Returns
    -------
    lags_repr : str
        Representation of the lags.

    """

    if isinstance(lags, dict):
        lags_repr = ", ".join(
            f"{k!r}: {_format_lags_warning(v)}" for k, v in lags.items()
        )
        return "{" + lags_repr + "}"
    if isinstance(lags, (np.ndarray, list, range)):
        lags = [int(lag) for lag in lags]
        if len(lags) > 6:
            return f"[{lags[0]}, {lags[1]}, ..., {lags[-1]}]"
        return str(lags)

    return str(lags)


def _format_names_warning(
    names: list,
    max_names: int | None = None,
    details: dict | None = None
) -> str:
    """
    Representation of a list of names for the warnings of the searches. If
    `max_names` is given and the list is longer, only the first two and the
    last names are shown.

    Parameters
    ----------
    names : list
        Names to show.
    max_names : int, default None
        Maximum number of names shown in full.
    details : dict, default None
        Detail of each name, shown in parentheses after it.

    Returns
    -------
    names_repr : str
        Representation of the names, without brackets.

    """

    if max_names is not None and len(names) > max_names:
        names = [*names[:2], ..., names[-1]]

    return ", ".join(
        "..." if name is ...
        else repr(name) if details is None
        else f"{name!r} ({details[name]})"
        for name in names
    )


def _is_affected_n_evaluated(n_evaluated: list[int] | np.ndarray) -> np.ndarray:
    """
    Candidates that evaluate less than `N_EVALUATED_WARNING_THRESHOLD` (95%)
    of the points of the most complete candidate.

    Parameters
    ----------
    n_evaluated : list, numpy ndarray
        Number of points evaluated of each candidate.

    Returns
    -------
    is_affected : numpy ndarray
        Whether each candidate evaluates fewer points than the threshold.

    """

    n_evaluated = np.asarray(n_evaluated)
    if n_evaluated.size == 0:
        return np.zeros(0, dtype=bool)

    return n_evaluated < N_EVALUATED_WARNING_THRESHOLD * n_evaluated.max()


def _n_evaluated_covers_levels_not_predicted(
    n_evaluated: list[int] | np.ndarray,
    n_evaluated_levels: list[list[int]] | np.ndarray
) -> bool:
    """
    Whether the warning of the points evaluated of a multi-series search
    (`_warn_n_evaluated_search`) covers the warning of the levels
    not predicted raised in the backtesting of the candidates, so the search
    can leave the latter out. It does when the warning of the points
    evaluated is raised and every level has points evaluated in some
    candidate: the levels not predicted by some candidates are those that are
    not evaluated on the same points by every candidate, and the warning
    gives the action. A level without points evaluated in any candidate is
    not in the warning of the points evaluated, so then both are shown.

    Parameters
    ----------
    n_evaluated : list, numpy ndarray
        Number of points evaluated of each candidate.
    n_evaluated_levels : list, numpy ndarray
        Number of points evaluated of each candidate (rows) and level
        (columns).

    Returns
    -------
    covers : bool
        Whether the warning of the levels not predicted can be left out.

    """

    if not _is_affected_n_evaluated(n_evaluated).any():
        return False
    n_evaluated_levels = np.asarray(n_evaluated_levels)

    return bool((n_evaluated_levels.max(axis=0) > 0).all())


def _warn_n_evaluated_search(
    n_evaluated: list[int] | np.ndarray,
    lags: list[Any],
    candidate_ids: list[int],
    n_evaluated_levels: list[list[int]] | np.ndarray | None = None,
    levels: list[str] | None = None,
    candidate_name: str = "candidate",
    best_is_first: bool = True,
    max_lines: int = 5,
    max_ids: int = 5,
    max_levels: int = 10
) -> None:
    """
    Warn, once at the end of a search, when some candidates evaluate fewer
    points than the most complete one, so their metrics are not comparable. A
    candidate is reported when it evaluates less than
    `N_EVALUATED_WARNING_THRESHOLD` (95%) of the points of the most complete
    candidate. The candidates must be in the order of the ranking, the first
    one is the best.

    In multi-series searches (`levels` is not `None`), a level is left out of
    the metrics of a candidate when the candidate cannot predict it. In single
    series searches, it happens with a `OneStepAheadFold` when the rows with
    missing values in `y` or in its lags are dropped
    (`dropna_from_series=True`): more rows are dropped with larger windows.

    The candidates reported are grouped by `lags` and number of points
    evaluated, which usually depends only on the window of the candidate, and
    identified by their row in the results (grid and random search) or their
    trial number (Bayesian search).

    In multi-series searches, the warning also gives the levels that every
    candidate evaluates on the same points (the same number of points as the
    most complete candidate in that level): passing them in `levels` makes
    the candidates comparable.
    They are listed if they are at most `max_levels`; otherwise, if the rest
    are at most `max_levels`, the rest are listed instead. Levels without
    points evaluated in any candidate are left out.

    Parameters
    ----------
    n_evaluated : list, numpy ndarray
        Number of points evaluated of each candidate, in the order of the
        ranking.
    lags : list
        Lags of each candidate, in the order of the ranking.
    candidate_ids : list
        Identifier of each candidate, in the order of the ranking: its row in
        the results if `candidate_name` is 'candidate', its trial number if it
        is 'trial'.
    n_evaluated_levels : list, numpy ndarray, default None
        Number of points evaluated of each candidate (rows, in the order of the
        ranking) and level (columns, in the order of `levels`). Only used in
        multi-series searches.
    levels : list, default None
        Levels of a multi-series search. `None` in single series searches.
    candidate_name : str, default 'candidate'
        Name of a candidate in the warning ('candidate' or 'trial').
    best_is_first : bool, default True
        Whether the first candidate is the best of the ranking. If `False` (the
        best candidate is not among the candidates), the warning does not say
        whether the best candidate is affected.
    max_lines : int, default 5
        Maximum number of groups of candidates listed in the warning.
    max_ids : int, default 5
        Maximum number of identifiers listed in each group of candidates.
    max_levels : int, default 10
        Maximum number of levels listed in the warning.

    Returns
    -------
    None

    """

    n_evaluated = np.asarray(n_evaluated)
    is_affected = _is_affected_n_evaluated(n_evaluated)
    if not is_affected.any():
        return

    n_evaluated_max = int(n_evaluated.max())

    candidates_name = f"{candidate_name}s"
    affected = np.flatnonzero(is_affected)
    if len(affected) == 1:
        summary = (
            f"1 of {len(n_evaluated)} {candidates_name} is evaluated on fewer "
            f"points than the most complete one, so its metrics are not "
            f"comparable."
        )
        best_message = (
            "It is the best candidate of the ranking, so the ranking may favor it."
            if is_affected[0]
            else "It is not the best candidate of the ranking."
        )
    else:
        summary = (
            f"{len(affected)} of {len(n_evaluated)} {candidates_name} are "
            f"evaluated on fewer points than the most complete one, so their "
            f"metrics are not comparable."
        )
        best_message = (
            "The best candidate of the ranking is one of them, so the ranking may "
            "favor it."
            if is_affected[0]
            else "The best candidate of the ranking is not one of them."
        )
    if best_is_first:
        summary += f" {best_message}"

    remove_message = "remove the candidates with larger windows"
    levels_block = None
    if levels is None:
        action = (
            f"The rows with missing values in `y` or in its lags are dropped "
            f"(`dropna_from_series=True`), more with larger windows. To compare "
            f"the candidates on the same points, fill the gaps of `y` or "
            f"{remove_message}."
        )
    elif len(levels) < 2:
        action = f"To compare the candidates on the same points, {remove_message}."
    else:
        n_evaluated_levels = np.asarray(n_evaluated_levels)
        n_evaluated_levels_max = n_evaluated_levels.max(axis=0)
        is_evaluated = n_evaluated_levels_max > 0
        is_same_points = (n_evaluated_levels == n_evaluated_levels_max).all(axis=0)
        levels_same = [
            level for level, same, evaluated
            in zip(levels, is_same_points, is_evaluated) if same and evaluated
        ]
        levels_other = [
            level for level, same, evaluated
            in zip(levels, is_same_points, is_evaluated) if not same and evaluated
        ]
        if not levels_same:
            action = (
                f"No series is evaluated on the same points by every candidate, "
                f"so restricting `levels` does not help: {remove_message}."
            )
        else:
            action = (
                f"To compare the candidates on the same points, pass the series "
                f"listed below in `levels` or {remove_message}."
            )
            if len(levels_same) > max_levels and len(levels_other) > max_levels:
                action = (
                    f"To compare the candidates on the same points, pass in "
                    f"`levels` only the series that every candidate evaluates on "
                    f"the same points (counted below) or {remove_message}."
                )
            if len(levels_same) <= max_levels:
                levels_line = f"levels=[{_format_names_warning(levels_same)}]"
            elif len(levels_other) <= max_levels:
                levels_line = (
                    f"levels: every series except "
                    f"[{_format_names_warning(levels_other)}]"
                )
            else:
                levels_line = (
                    f"levels: {len(levels_same):,} of "
                    f"{len(levels_same) + len(levels_other):,} series (the series "
                    f"left out include "
                    f"{_format_names_warning(levels_other, max_names=max_levels)})"
                )
            levels_block = (
                f"Series that every candidate evaluates on the same points:\n"
                f"  {levels_line}"
            )

    groups = {}
    for i in affected:
        key = (_format_lags_warning(lags[i]), int(n_evaluated[i]))
        groups.setdefault(key, []).append(candidate_ids[i])

    id_name = "row" if candidate_name == "candidate" else candidate_name
    lines = []
    for (lags_repr, n), ids in list(groups.items())[:max_lines]:
        if len(ids) == 1:
            ids_repr = f"{id_name} {ids[0]}"
        else:
            ids_list = ", ".join(str(id_) for id_ in ids[:max_ids])
            if len(ids) > max_ids:
                ids_list += ", ..."
            if id_name != candidate_name:
                ids_list = f"{id_name}s {ids_list}"
            ids_repr = f"{len(ids)} {candidates_name} ({ids_list})"
        lines.append(
            f"  - lags: {lags_repr}, {ids_repr}: {n:,} ({n / n_evaluated_max:.1%})"
        )
    if len(groups) > max_lines:
        n_more = sum(len(ids) for ids in list(groups.values())[max_lines:])
        lines.append(
            f"  ... and {n_more} more {candidate_name if n_more == 1 else candidates_name}"
        )
    candidates_block = (
        f"{candidates_name.capitalize()} with less than "
        f"{N_EVALUATED_WARNING_THRESHOLD:.0%} of the {n_evaluated_max:,} points "
        f"of the most complete one:\n" + "\n".join(lines)
    )

    blocks = [
        f"{summary} {action} See the column `{N_EVALUATED_NAME}` of the results.",
        candidates_block,
    ]
    if levels_block is not None:
        blocks.append(levels_block)

    warnings.warn(
        "\n\n".join(blocks), MissingValuesWarning, stacklevel=_find_stack_level()
    )


def _drop_series_too_short_refit(
    forecaster: object,
    series: dict[str, pd.Series] | pd.DataFrame,
    exog: dict[str, pd.DataFrame | None] | pd.Series | pd.DataFrame | None = None
) -> tuple[
    dict[str, pd.Series] | pd.DataFrame,
    dict[str, pd.DataFrame | None] | pd.Series | pd.DataFrame | None
]:
    """
    Leave out of the refit of the best candidate of a multi-series search the
    series that `fit` of `ForecasterRecursiveMultiSeries` rejects: they have
    no more observations than the `window_size`, after removing their leading
    and trailing NaNs. The backtesting of the candidate does not predict them
    either. A warning lists them, with their number of observations: the
    forecaster returned is not trained with them. It is a `RuntimeWarning`,
    not silenced by `suppress_warnings`, since it describes the forecaster
    returned. Other forecasters are returned unchanged.

    Parameters
    ----------
    forecaster : ForecasterRecursiveMultiSeries, ForecasterDirectMultiVariate
        Forecaster with the lags and parameters of the best candidate.
    series : dict, pandas DataFrame
        Time series of the search (a dict, already checked, for
        ForecasterRecursiveMultiSeries).
    exog : dict, pandas Series, pandas DataFrame, default None
        Exogenous variables of the search. They are filtered only if they are
        a dict.

    Returns
    -------
    series : dict, pandas DataFrame
        Time series without the series too short.
    exog : dict, pandas Series, pandas DataFrame, None
        Exogenous variables without the series too short.

    """

    if type(forecaster).__name__ != "ForecasterRecursiveMultiSeries":
        return series, exog

    # NOTE: `align_series_and_exog_multiseries` replaces the values of the
    # dict, so a copy is passed. Without exog, it only trims the series.
    series_aligned, _ = align_series_and_exog_multiseries(
                            series_dict = dict(series),
                            exog_dict   = {k: None for k in series}
                        )
    series_too_short = _series_too_short_multiseries(
                           series_dict = series_aligned,
                           window_size = forecaster.window_size
                       )
    if not series_too_short:
        return series, exog

    series = {k: v for k, v in series.items() if k not in series_too_short}
    if isinstance(exog, dict):
        exog = {k: v for k, v in exog.items() if k not in series_too_short}

    n_series = len(series_too_short)
    n_observations = {
        k: f"{len(series_aligned[k])} observation"
           f"{'' if len(series_aligned[k]) == 1 else 's'}"
        for k in series_too_short
    }
    names_repr = _format_names_warning(
        series_too_short, max_names=10, details=n_observations
    )
    warnings.warn(
        f"{n_series} series {'is' if n_series == 1 else 'are'} left out of the "
        f"refit of the best candidate: {'it has' if n_series == 1 else 'they have'} "
        f"no more observations than the `window_size` ({forecaster.window_size}), "
        f"too few to train the forecaster, and the candidate did not predict "
        f"{'it' if n_series == 1 else 'them'} in the search either. The "
        f"forecaster returned is not trained with "
        f"{'it' if n_series == 1 else 'them'} (not in `series_names_in_`).\n\n"
        f"Series left out: {names_repr}.",
        RuntimeWarning,
        stacklevel = _find_stack_level()
    )

    return series, exog


def _sample_param_grid(
    param_distributions: dict,
    n_iter: int,
    random_state: int,
    grid_search_name: str,
    lags_grid: bool = True
) -> list[dict]:
    """
    Sample the combinations of parameters of a random search with
    scikit-learn's `ParameterSampler`. When the total space of parameters is
    smaller than `n_iter`, the warning of scikit-learn, which refers to
    `GridSearchCV`, is replaced by one that refers to the grid search of
    skforecast.

    Parameters
    ----------
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and distributions or
        lists of parameters to try.
    n_iter : int
        Number of parameter settings that are sampled.
    random_state : int
        Seed of the sampling.
    grid_search_name : str
        Name of the grid search of skforecast equivalent to the random search.
    lags_grid : bool, default True
        Whether the random search evaluates the combinations for each value of
        `lags_grid`.

    Returns
    -------
    param_grid : list
        Combinations of parameters sampled.

    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category = UserWarning,
            message  = "The total space of parameters"
        )
        param_grid = list(
            ParameterSampler(
                param_distributions, n_iter=n_iter, random_state=random_state
            )
        )

    # NOTE: ParameterSampler returns fewer combinations than `n_iter` only when
    # every distribution is a list and the space is smaller than `n_iter`.
    if len(param_grid) < n_iter:
        for_each_lags = " for each value of `lags_grid`" if lags_grid else ""
        warnings.warn(
            f"The total space of parameters ({len(param_grid)}) is smaller than "
            f"`n_iter` ({n_iter}), so the search evaluates the {len(param_grid)} "
            f"combinations{for_each_lags}. For an exhaustive search, use "
            f"`{grid_search_name}`.",
            UserWarning,
            stacklevel = _find_stack_level()
        )

    return param_grid


def _warn_skipped_candidates_search(
    skipped: list[tuple[Any, dict, Exception, int | None]],
    n_candidates: int,
    n_params: int | None = None,
    candidate_name: str = "candidate",
    max_errors: int = 5,
    max_candidates: int = 5
) -> None:
    """
    Warn, once at the end of a search, about the candidates skipped because
    their evaluation raised an error, so they are not in the results. The
    candidates are grouped by error. When every combination of parameters of
    a value of `lags` raises the same error, they are summarized in one line.
    The warning points to the code of the user that called the search.

    Parameters
    ----------
    skipped : list
        Candidates skipped, in the order of the search, as tuples
        `(lags, params, error, candidate_id)`. `lags` is `None` in the searches
        without `lags_grid`. `candidate_id` identifies the candidate in the
        warning (the trial number in Bayesian searches), `None` if the
        candidate is identified by its lags and parameters.
    n_candidates : int
        Number of candidates of the search.
    n_params : int, default None
        Number of combinations of parameters evaluated for each value of
        `lags`. If `None`, the candidates are not summarized by `lags`.
    candidate_name : str, default 'candidate'
        Name of a candidate in the warning ('candidate' or 'trial').
    max_errors : int, default 5
        Maximum number of errors listed in the warning.
    max_candidates : int, default 5
        Maximum number of candidates listed for each error.

    Returns
    -------
    None

    """

    if not skipped:
        return

    n_skipped = len(skipped)
    if n_skipped == n_candidates:
        consequence = "so the results are empty"
    elif n_skipped == 1:
        consequence = "so it is not in the results"
    else:
        consequence = "so they are not in the results"
    verb = "raises" if n_skipped == 1 else "raise"
    summary = (
        f"Parameters skipped: {n_skipped} of {n_candidates} {candidate_name}s "
        f"{verb} an error, {consequence}."
    )

    # NOTE: The trials of Optuna run in threads can be skipped in any order.
    skipped = sorted(
        skipped, key=lambda candidate: -1 if candidate[3] is None else candidate[3]
    )
    errors = {}
    for lags, params, error, candidate_id in skipped:
        error_message = f"{type(error).__name__}: {error}"
        errors.setdefault(error_message, []).append((lags, params, candidate_id))

    blocks = [summary]
    for error_message, candidates in list(errors.items())[:max_errors]:

        lines = []
        lags_reprs = {}
        for lags, params, candidate_id in candidates:
            id_repr = (
                "" if candidate_id is None else f"{candidate_name} {candidate_id}, "
            )
            lags_reprs.setdefault(
                None if lags is None else _format_lags_warning(lags), []
            ).append((id_repr, params))
        for lags_repr, params_list in lags_reprs.items():
            if lags_repr is None:
                lines.extend(
                    f"{id_repr}params: {params}" for id_repr, params in params_list
                )
            elif n_params is not None and n_params > 1 and len(params_list) == n_params:
                lines.append(f"lags: {lags_repr} ({n_params} candidates)")
            else:
                lines.extend(
                    f"{id_repr}lags: {lags_repr}, params: {params}"
                    for id_repr, params in params_list
                )

        candidates_lines = [f"  - {line}" for line in lines[:max_candidates]]
        if len(lines) > max_candidates:
            candidates_lines.append(f"  ... and {len(lines) - max_candidates} more")
        blocks.append(error_message + "\n" + "\n".join(candidates_lines))

    if len(errors) > max_errors:
        blocks.append(f"... and {len(errors) - max_errors} more errors")

    warnings.warn(
        "\n\n".join(blocks), RuntimeWarning, stacklevel=_find_stack_level()
    )


class _SearchWarnings:
    """
    Context manager that records the warnings raised while the candidates of
    a search are evaluated, so each of them is shown once at the end of the
    search, with the number of candidates in which it is raised, instead of
    once per candidate.

    - The warnings are recorded with the filters active when the context is
    entered (the user's and those of `suppress_warnings`): a warning ignored
    by them is not recorded. The actions that show a warning once are replaced
    by `"always"`, also the default action of Python, so the repetitions are
    recorded and counted.
    - `LongTrainingWarning` is shown when it is raised the first time, with
    the number of fits of the whole search, so the user can stop the search.
    The rest are discarded.
    - The warnings point to the code of the user that called the search
    (the first frame outside skforecast when the context is entered).
    - Each warning is tagged with the candidate being evaluated, set with
    `set_candidate`. The candidate is stored per thread, so the trials of
    Optuna run in threads (`n_jobs` of `study.optimize`) are tagged correctly.
    Warnings raised with no candidate (for example, during the refit of the
    best candidate) are shown, but not counted.
    - The candidates skipped because their evaluation raised an error are
    added with `add_skipped` and reported in a single warning.

    The warnings raised in the folds of the backtesting that run in the worker
    processes of joblib (`n_jobs` greater than 1, which `n_jobs='auto'`
    selects for some forecasters) are shown again in the main process, after
    the folds (`_reemit_worker_warnings`), so they are recorded too.

    Parameters
    ----------
    n_candidates : int
        Number of candidates (or trials) of the search, used to report the
        number of fits of the whole search in `LongTrainingWarning`.
    candidate_name : str, default 'candidate'
        Name of a candidate in the warnings ('candidate' or 'trial').

    Attributes
    ----------
    n_candidates : int
        Number of candidates (or trials) of the search.
    candidate_name : str
        Name of a candidate in the warnings ('candidate' or 'trial').
    records : list
        Warnings recorded, as tuples `(candidate, category, text, message)`.
    skipped : list
        Candidates skipped, as tuples `(lags, params, error, candidate_id)`.

    """

    def __init__(
        self, n_candidates: int, candidate_name: str = "candidate"
    ) -> None:

        self.n_candidates = n_candidates
        self.candidate_name = candidate_name
        self.records = []
        self.skipped = []
        self._local = threading.local()
        self._lock = threading.Lock()
        self._long_training_shown = False

    def __enter__(self) -> _SearchWarnings:

        self._catch_warnings = warnings.catch_warnings()
        self._catch_warnings.__enter__()

        # NOTE: The actions that show a warning once ('default', 'module',
        # 'once') are replaced by 'always', so the repetitions are recorded and
        # counted. 'ignore' and 'error' are kept. The filters are restored on
        # exit, so the warnings are shown with the original ones.
        filters = list(warnings.filters)
        warnings.resetwarnings()
        for action, message, category, module, lineno in reversed(filters):
            warnings.filterwarnings(
                action   = "always" if action in ("default", "module", "once") else action,
                message  = getattr(message, "pattern", message) or "",
                category = category,
                module   = getattr(module, "pattern", module) or "",
                lineno   = lineno
            )
        warnings.simplefilter("always", append=True)
        self._showwarning = warnings.showwarning
        warnings.showwarning = self._record

        # NOTE: Code of the user that called the search, where LongTrainingWarning
        # points to. It is found here, in the main thread, because the trials of
        # Optuna can run in other threads.
        frame, _ = _first_frame_outside_skforecast(sys._getframe(1))
        self._user_location = (
            (frame.f_code.co_filename, frame.f_lineno) if frame is not None else None
        )

        return self

    def __exit__(self, *exc_info) -> None:

        self._catch_warnings.__exit__(*exc_info)

    def set_candidate(self, candidate: Any) -> None:
        """
        Set the candidate being evaluated in the current thread.

        Parameters
        ----------
        candidate : Any
            Identifier of the candidate, `None` outside the evaluation of the
            candidates.

        Returns
        -------
        None

        """

        self._local.candidate = candidate

    def add_skipped(
        self,
        lags: Any,
        params: dict,
        error: Exception,
        candidate_id: int | None = None
    ) -> None:
        """
        Add a candidate skipped because its evaluation raised an error. It can
        be called from the threads of Optuna.

        Parameters
        ----------
        lags : Any
            Lags of the candidate, `None` in the searches without `lags_grid`.
        params : dict
            Parameters of the candidate.
        error : Exception
            Error raised.
        candidate_id : int, default None
            Identifier of the candidate in the warning (the trial number in
            Bayesian searches). If `None`, the candidate is identified by its
            lags and parameters.

        Returns
        -------
        None

        """

        with self._lock:
            self.skipped.append((lags, params, error, candidate_id))

    def discard(self, category: type[Warning], text_start: str) -> None:
        """
        Discard the warnings recorded of a category whose text starts with
        `text_start`, for example, a warning covered by another one of the
        search.

        Parameters
        ----------
        category : Warning
            Category of the warnings to discard.
        text_start : str
            Start of the text of the warnings to discard.

        Returns
        -------
        None

        """

        self.records = [
            record for record in self.records
            if not (issubclass(record[1], category) and record[2].startswith(text_start))
        ]

    def _record(
        self,
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: str | None = None
    ) -> None:
        """
        Replacement of `warnings.showwarning` that records the warning with
        the candidate of the current thread. The first `LongTrainingWarning`
        is shown at once and the rest are discarded.

        Parameters
        ----------
        message : Warning, str
            Warning raised.
        category : Warning
            Category of the warning.
        filename : str
            File where the warning is raised.
        lineno : int
            Line where the warning is raised.
        file : file, default None
            File where the warning is written, passed to `warnings.showwarning`.
        line : str, default None
            Line of code of the warning, passed to `warnings.showwarning`.

        Returns
        -------
        None

        """

        if issubclass(category, LongTrainingWarning):
            with self._lock:
                if self._long_training_shown:
                    return
                self._long_training_shown = True
            message = self._long_training_message(message)
            if self._user_location is not None:
                filename, lineno = self._user_location
                line = None
            self._showwarning(message, category, filename, lineno, file, line)
            return

        text = getattr(message, "message", None)
        if not isinstance(text, str):
            text = str(message)
        candidate = getattr(self._local, "candidate", None)
        with self._lock:
            self.records.append((candidate, category, text, message))

    def _long_training_message(self, message: Warning | str) -> Warning | str:
        """
        Rewrite the `LongTrainingWarning` of the backtesting of a candidate
        with the number of fits of the whole search. If the number of fits
        cannot be read from the message, it is returned unchanged.

        Parameters
        ----------
        message : Warning, str
            `LongTrainingWarning` of the backtesting of a candidate.

        Returns
        -------
        message : Warning, str
            Warning with the number of fits of the whole search.

        """

        text = getattr(message, "message", str(message))
        match = re.search(r"fit (\d+) times( \(\d+ folds \* \d+ estimators\))?", text)
        if match is None:
            return message

        n_fits = int(match.group(1))
        n_candidates_name = (
            f"{self.n_candidates} {self.candidate_name}"
            f"{'' if self.n_candidates == 1 else 's'}"
        )
        return LongTrainingWarning(
            f"Each {self.candidate_name} fits the forecaster about {n_fits:,} "
            f"times{match.group(2) or ''}, so the search ({n_candidates_name}) "
            f"fits it about {n_fits * self.n_candidates:,} times. This can take "
            f"substantial amounts of time. If not feasible, try with "
            f"`refit = False`."
        )

    def emit(self, n_candidates: int | None = None, n_params: int | None = None) -> None:
        """
        Show the warnings recorded, each once, and the candidates skipped.
        Call it after exiting the context, so the warnings go through the
        filters of the user.

        - A warning with the same text in every candidate is shown as it is.
        - A warning with the same text in some candidates is shown with the
        number of candidates, after its first sentence.
        - Warnings with the same first sentence and different texts (for
        example, with the series or the counts of each candidate) are shown
        once: the text of the first one, with the number of candidates and
        "details of the first one" after its first sentence.
        - Warnings raised only outside the candidates (for example, in the
        refit or when the training matrices of a `OneStepAheadFold` are
        created for each value of `lags`) are shown as they are if they have
        the same text. Those with the same first sentence and different texts
        (for example, with the counts of each value of `lags`) are shown once:
        the text of the first one, with the number of different texts and
        "details of the first one" after its first sentence.

        The warnings point to the code of the user that called the search.

        Parameters
        ----------
        n_candidates : int, default None
            Number of candidates evaluated. If `None`, `n_candidates` of the
            initialization.
        n_params : int, default None
            Number of combinations of parameters evaluated for each value of
            `lags`, to summarize the candidates skipped by `lags`.

        Returns
        -------
        None

        """

        if n_candidates is None:
            n_candidates = self.n_candidates
        candidates_name = f"{self.candidate_name}s"

        groups = {}
        for candidate, category, text, message in self.records:
            first_sentence = _split_first_sentence(text)[0]
            texts = groups.setdefault((category, first_sentence), {})
            candidates, _ = texts.setdefault(text, (set(), message))
            if candidate is not None:
                candidates.add(candidate)

        for (category, _), texts in groups.items():

            n = len(set().union(*(candidates for candidates, _ in texts.values())))
            text, (_, message) = next(iter(texts.items()))
            if n == 0:
                if len(texts) > 1:
                    first_sentence, rest = _split_first_sentence(text)
                    text = (
                        f"{first_sentence[:-1]} ({len(texts)} times in the search, "
                        f"details of the first one).{rest}"
                    )
                _warn_recorded(text, category, message)
                continue

            if len(texts) == 1 and n == n_candidates:
                _warn_recorded(text, category, message)
                continue

            details = "" if len(texts) == 1 else "; details of the first one"
            first_sentence, rest = _split_first_sentence(text)
            text = (
                f"{first_sentence[:-1]} (in {n} of {n_candidates} "
                f"{candidates_name}{details}).{rest}"
            )
            _warn_recorded(text, category, message)

        _warn_skipped_candidates_search(
            skipped        = self.skipped,
            n_candidates   = n_candidates,
            n_params       = n_params,
            candidate_name = self.candidate_name
        )


class _TrialEvaluationError(Exception):
    """
    Error raised when the evaluation of a trial of a Bayesian search fails.
    It is added to the argument `catch` of `study.optimize`, so Optuna marks
    the trial as failed and the search continues. The original error is the
    cause (`__cause__`) of this one.
    """


def _skip_trial(
    search_warnings: _SearchWarnings, trial: object, error: Exception
) -> _TrialEvaluationError:
    """
    Skip a trial of a Bayesian search whose evaluation raised an error: add
    it to the candidates skipped of the search, identified by its number, and
    return the `_TrialEvaluationError` to raise, so Optuna marks the trial as
    failed (`raise _skip_trial(...) from error`).

    Parameters
    ----------
    search_warnings : _SearchWarnings
        Warnings of the search.
    trial : optuna Trial
        Trial skipped.
    error : Exception
        Error raised by the evaluation of the trial.

    Returns
    -------
    trial_error : _TrialEvaluationError
        Error to raise.

    """

    search_warnings.add_skipped(
        lags=None, params=trial.params, error=error, candidate_id=trial.number
    )

    return _TrialEvaluationError(f"{type(error).__name__}: {error}")


class _FailedTrialsLogFilter(logging.Filter):
    """
    Filter of the logger of Optuna that runs the trials
    (`optuna.study._optimize`) that drops the two messages of each failed
    trial ("Trial N failed with parameters..." with the traceback, and "Trial
    N failed with value None."), unless its error is in the `catch` of the
    user. The trials that fail with `_TrialEvaluationError` are reported in
    the warning of the candidates skipped of the search, and the other errors
    are raised by `study.optimize`, so their messages would repeat them. The
    errors in the `catch` of the user are only reported by these messages.

    Parameters
    ----------
    catch : tuple, default ()
        Errors in the `catch` of the user, whose messages are kept.

    Attributes
    ----------
    catch : tuple
        Errors in the `catch` of the user, whose messages are kept.

    """

    def __init__(self, catch: tuple[type[Exception], ...] = ()) -> None:

        super().__init__()
        self.catch = catch
        self._trials = set()

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Whether the record is logged.

        Parameters
        ----------
        record : logging.LogRecord
            Record of the logger.

        Returns
        -------
        keep : bool
            `False` for the records of the trials that fail with an error that
            is not in the `catch` of the user (`_TrialEvaluationError` or an
            error that stops the search).

        """

        match = re.match(r"Trial (\d+) failed", record.getMessage())
        if match is None:
            return True
        error = record.exc_info[1] if record.exc_info else None
        if error is not None and (
            isinstance(error, _TrialEvaluationError)
            or not isinstance(error, self.catch)
        ):
            self._trials.add(match.group(1))
            return False

        return match.group(1) not in self._trials


def _optimize_study(
    study: object,
    objective: Callable,
    n_trials: int,
    kwargs_study_optimize: dict,
    filter_logs: bool = True
) -> None:
    """
    Run `study.optimize` for a Bayesian search. The trials whose evaluation
    fails (`_TrialEvaluationError`, see `_skip_trial`) are marked as failed by
    Optuna and the search continues: `_TrialEvaluationError` is added to the
    argument `catch` of the user. Other errors, for example in the
    `search_space`, stop the search. The warning of Optuna about the choices
    of a categorical distribution is ignored.

    Parameters
    ----------
    study : optuna Study
        Study to optimize.
    objective : Callable
        Objective function of the search.
    n_trials : int
        Number of trials.
    kwargs_study_optimize : dict
        Other arguments of `study.optimize`, `catch` included.
    filter_logs : bool, default True
        If `True`, the messages of Optuna about the failed trials are not
        logged, except for the errors in the `catch` of the user: the trials
        that fail with `_TrialEvaluationError` are reported in the warning of
        the candidates skipped, and the other errors are raised (see
        `_FailedTrialsLogFilter`).

    Returns
    -------
    None

    """

    kwargs_study_optimize = dict(kwargs_study_optimize)
    catch = kwargs_study_optimize.pop("catch", ())
    catch = (catch,) if isinstance(catch, type) else tuple(catch)

    # NOTE: The messages are logged by the child logger of Optuna that runs
    # the trials; a filter of the logger `optuna` does not apply to them.
    logger = logging.getLogger("optuna.study._optimize")
    log_filter = _FailedTrialsLogFilter(catch=catch) if filter_logs else None
    if log_filter is not None:
        logger.addFilter(log_filter)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category = UserWarning,
                message  = "Choices for a categorical distribution should be*"
            )
            study.optimize(
                objective,
                n_trials = n_trials,
                catch    = (*catch, _TrialEvaluationError),
                **kwargs_study_optimize
            )
    finally:
        if log_filter is not None:
            logger.removeFilter(log_filter)


def _split_first_sentence(text: str) -> tuple[str, str]:
    """
    Split a text in its first sentence, ending in a period, and the rest. If
    the text has no period followed by a space or a line break, the whole text
    is the first sentence and a period is added.

    Parameters
    ----------
    text : str
        Text to split.

    Returns
    -------
    first_sentence : str
        First sentence, ending in a period.
    rest : str
        Rest of the text, with the leading space or line break.

    """

    match = re.search(r"\.(?=\s)", text)
    if match is None:
        text = text.rstrip()
        return (text if text.endswith(".") else f"{text}."), ""

    return text[:match.end()], text[match.end():]


def _warn_recorded(text: str, category: type[Warning], message: Warning | str) -> None:
    """
    Show a warning recorded by `_SearchWarnings` with a new text and the same
    category, pointing to the code of the user. If the category cannot be
    built with a text, the original message is shown.

    Parameters
    ----------
    text : str
        Text of the warning.
    category : Warning
        Category of the warning.
    message : Warning, str
        Original message of the warning, shown if the category cannot be built
        with `text`.

    Returns
    -------
    None

    """

    try:
        new_message = category(text)
    except Exception:
        new_message = message

    warnings.warn(new_message, stacklevel=_find_stack_level())
