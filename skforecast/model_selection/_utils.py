################################################################################
#                     skforecast.model_selection._utils                        #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################


from __future__ import annotations
from typing import Callable, Generator, Any
import warnings
from itertools import chain
import numpy as np
import pandas as pd
from joblib import cpu_count
from sklearn.exceptions import NotFittedError
from sklearn.linear_model._base import LinearModel, LinearClassifierMixin
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from ..exceptions import (
    IgnoredArgumentWarning,
    MissingValuesWarning,
    OneStepAheadValidationWarning
)
from ..metrics import (
    add_y_train_argument,
    _any_metric_needs_y_train,
    _get_metric,
    calculate_coverage
)
from ..metrics.metrics import _metric_greater_is_better
from ..utils import (
    _date_range_from_index,
    check_interval,
    date_to_index_position,
    cast_catboost_categorical_columns_dataframe,
    _copy_rows_to_check,
    _check_in_place_fit,
)

# NOTE: Names of the metrics added to the backtesting metrics when the
# predictions include `lower_bound` and `upper_bound`, see
# `_calculate_interval_metrics`.
INTERVAL_METRIC_NAMES = ['coverage', 'interval_width']


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
        if forecaster_name == 'ForecasterFoundation':
            # NOTE: For ForecasterFoundation, window_size equals context_length 
            # which is the *maximum* context the model accepts, not a minimum 
            # required training size. Only validate that initial_train_size is 
            # at least 1 and smaller than data_length.
            if initial_train_size < 1 or initial_train_size >= data_length:
                raise ValueError(
                    f"If `initial_train_size` is an integer, it must be greater than "
                    f"0 and smaller than {data_length_name} ({data_length}). "
                    f"If it is a date, it must be within this range of the index."
                )
        elif initial_train_size < forecaster.window_size or initial_train_size >= data_length:
            raise ValueError(
                f"If `initial_train_size` is an integer, it must be greater than "
                f"the `window_size` of the forecaster ({forecaster.window_size}) "
                f"and smaller than {data_length_name} ({data_length}). If "
                f"it is a date, it must be within this range of the index."
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

    # NOTE: `backtesting_foundation` passes its `quantiles` as `interval`.
    if _get_interval_bound_quantiles(interval) is not None or alpha is not None:
        metric_names = [
            m if isinstance(m, str) else getattr(m, '__name__', None)
            for m in (metric if isinstance(metric, list) else [metric])
        ]
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
    if initial_train_size < forecaster.window_size or initial_train_size >= data_length:
        raise ValueError(
            f"If `initial_train_size` is an integer, it must be greater than "
            f"the `window_size` of the forecaster ({forecaster.window_size}) "
            f"and smaller than {data_length_name} ({data_length}). If "
            f"it is a date, it must be within this range of the index."
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
            "estimate.", OneStepAheadValidationWarning
        )


def select_n_jobs_backtesting(
    forecaster: object,
    refit: bool | int
) -> int:
    """
    Select the optimal number of jobs to use in the backtesting process. This
    selection is based on heuristics and is not guaranteed to be optimal.

    The number of jobs is chosen as follows:

    - If `refit` is an integer, then `n_jobs = 1`. This is because parallelization doesn't 
    work with intermittent refit.
    - If forecaster is 'ForecasterRecursive' and estimator is a linear estimator, 
    then `n_jobs = 1`.
    - If forecaster is 'ForecasterRecursive' and estimator is not a linear 
    estimator then `n_jobs = max(1, cpu_count() - 1)`.
    - If forecaster is 'ForecasterDirect' or 'ForecasterDirectMultiVariate'
    and `refit = True`, then `n_jobs = max(1, cpu_count() - 1)`.
    - If forecaster is 'ForecasterDirect' or 'ForecasterDirectMultiVariate'
    and `refit = False`, then `n_jobs = 1`.
    - If forecaster is 'ForecasterRecursiveMultiSeries', then `n_jobs = max(1, cpu_count() - 1)`.
    - If forecaster is 'ForecasterStats' or 'ForecasterEquivalentDate', 
    then `n_jobs = 1`.
    - If estimator is a `LGBMRegressor(n_jobs=1)`, then `n_jobs = max(1, cpu_count() - 1)`.
    - If estimator is a `LGBMRegressor` with internal n_jobs != 1, then `n_jobs = 1`.
    This is because `lightgbm` is highly optimized for gradient boosting and
    parallelizes operations at a very fine-grained level, making additional
    parallelization unnecessary and potentially harmful due to resource contention.

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

    forecaster_name = type(forecaster).__name__

    if forecaster_name in ('ForecasterStats', 'ForecasterFoundation'):
        n_jobs = 1
        return n_jobs

    if isinstance(forecaster.estimator, Pipeline):
        estimator = forecaster.estimator[-1]
    else:
        estimator = forecaster.estimator

    refit = False if refit == 0 else refit
    if not isinstance(refit, bool) and refit != 1:
        n_jobs = 1
    else:
        if forecaster_name in {'ForecasterRecursive', 'ForecasterRecursiveClassifier'}:
            if isinstance(estimator, (LinearModel, LinearClassifierMixin)):
                n_jobs = 1
            elif type(estimator).__name__ in {'LGBMRegressor', 'LGBMClassifier'}:
                n_jobs = max(1, cpu_count() - 1) if estimator.n_jobs == 1 else 1
            else:
                n_jobs = max(1, cpu_count() - 1)
        elif forecaster_name in {'ForecasterDirect', 'ForecasterDirectMultiVariate'}:
            # Parallelization is applied during the fitting process.
            n_jobs = 1
        elif forecaster_name in {'ForecasterRecursiveMultiSeries'}:
            if type(estimator).__name__ == 'LGBMRegressor':
                n_jobs = max(1, cpu_count() - 1) if estimator.n_jobs == 1 else 1
            else:
                n_jobs = max(1, cpu_count() - 1)
        elif forecaster_name in {'ForecasterEquivalentDate'}:
            n_jobs = 1
        else:
            n_jobs = 1

    return n_jobs


def _calculate_metrics_one_step_ahead(
    forecaster: object,
    metrics: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_weight: np.ndarray | None,
    fit_kwargs: dict[str, Any]
) -> list:
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

    Returns
    -------
    metric_values : list
        List with metric values.
    
    """

    needs_y_train = _any_metric_needs_y_train(metrics)

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

    if not needs_y_train:
        y_train = None

    if forecaster.differentiation is not None:
        y_true = forecaster.differentiator.inverse_transform_next_window(y_true)
        y_pred = forecaster.differentiator.inverse_transform_next_window(y_pred)
        if needs_y_train:
            y_train = forecaster.differentiator.inverse_transform_training(y_train)

    if forecaster.transformer_y is not None:
        y_true = forecaster.transformer_y.inverse_transform(y_true.reshape(-1, 1))
        y_pred = forecaster.transformer_y.inverse_transform(y_pred.reshape(-1, 1))
        if needs_y_train:
            y_train = forecaster.transformer_y.inverse_transform(y_train.reshape(-1, 1))

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    if needs_y_train:
        y_train = y_train.ravel()
    
    metric_values = [
        m(y_true=y_true, y_pred=y_pred, y_train=y_train) for m in metrics
    ]

    return metric_values


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
                IgnoredArgumentWarning
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
            f"`window_size` ({forecaster.window_size}) observations or fewer "
            f"in the training set"
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
    if skipped_folds:
        lines.append(
            f"  - Folds skipped, no level can be predicted in them: "
            f"{', '.join(map(str, skipped_folds))}."
        )

    warnings.warn(
        "Some levels are not predicted in folds where they have observed values, "
        "so their metrics, and the aggregated metrics, do not include those "
        "folds. Levels never predicted have NaN metrics and are left out of the "
        "aggregated metrics.\n" + "\n".join(lines),
        MissingValuesWarning
    )


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


def _calculate_metrics_backtesting_multiseries(
    series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame],
    predictions: pd.DataFrame,
    folds: list[list[int]] | tqdm,
    span_index: pd.DatetimeIndex | pd.RangeIndex,
    window_size: int,
    metrics: list,
    levels: list,
    add_aggregated_metric: bool = True
) -> pd.DataFrame:
    """   
    Calculate metrics for each level and also for all levels aggregated using
    average, weighted average or pooling.

    - 'average': the average (arithmetic mean) of all levels.
    - 'weighted_average': the average of the metrics weighted by the number of
    predicted values of each level.
    - 'pooling': the values of all levels are pooled and then the metric is
    calculated.

    If `predictions` includes the columns `lower_bound` and `upper_bound`, the
    coverage and the mean width of the interval (`INTERVAL_METRIC_NAMES`) are
    added after the metrics, for each level and aggregated in the same way.

    Parameters
    ----------
    series : pandas DataFrame, dict
        Series data used for backtesting.
    predictions : pandas DataFrame
        MultiIndex DataFrame generated during the backtesting multiseries process.
        The first column is the point prediction used to calculate the metrics.
        It may also include the columns `lower_bound` and `upper_bound`.
    folds : list, tqdm
        Folds created during the backtesting process.
    span_index : pandas DatetimeIndex, pandas RangeIndex
        Full index from the minimum to the maximum index among all series.
    window_size : int
        Size of the window used by the forecaster to create the predictors.
        This is used remove the first `window_size` (differentiation included) 
        values from y_train since they are not part of the training matrix.
    metrics : list
        List of metrics to calculate.
    levels : list
        Levels to calculate the metrics.
    add_aggregated_metric : bool, default True
        If `True`, and multiple series (`levels`) are predicted, the aggregated
        metrics (average, weighted average and pooled) are also returned.

        - 'average': the average (arithmetic mean) of all levels.
        - 'weighted_average': the average of the metrics weighted by the number of
        predicted values of each level.
        - 'pooling': the values of all levels are pooled and then the metric is
        calculated.

    Returns
    -------
    metrics_levels : pandas DataFrame
        Value(s) of the metric(s).
    
    """

    # NOTE: All this checks can be deleted as they are done in the public
    # function that calls this private function.
    # if not isinstance(series, (pd.DataFrame, dict)):
    #     raise TypeError(
    #         "`series` must be a pandas DataFrame or a dictionary of pandas "
    #         "DataFrames."
    #     )
    # if not isinstance(predictions, pd.DataFrame):
    #     raise TypeError("`predictions` must be a pandas DataFrame.")
    # if not isinstance(folds, (list, tqdm)):
    #     raise TypeError("`folds` must be a list or a tqdm object.")
    # if not isinstance(span_index, (pd.DatetimeIndex, pd.RangeIndex)):
    #     raise TypeError("`span_index` must be a pandas DatetimeIndex or pandas RangeIndex.")
    # if not isinstance(window_size, (int, np.integer)):
    #     raise TypeError("`window_size` must be an integer.")
    # if not isinstance(metrics, list):
    #     raise TypeError("`metrics` must be a list.")
    # if not isinstance(levels, list):
    #     raise TypeError("`levels` must be a list.")
    # if not isinstance(add_aggregated_metric, bool):
    #     raise TypeError("`add_aggregated_metric` must be a boolean.")
    
    metric_names = [m.__name__ for m in metrics]
    needs_y_train = _any_metric_needs_y_train(metrics)
    add_interval_metrics = {'lower_bound', 'upper_bound'}.issubset(predictions.columns)
    output_names = (
        metric_names + INTERVAL_METRIC_NAMES if add_interval_metrics else metric_names
    )

    levels_in_predictions = predictions.index.get_level_values('level').unique()

    if isinstance(series, pd.DataFrame) and not isinstance(series.index, pd.MultiIndex):
        series = series.stack()
        series.index = series.index.swaplevel()
        series.index.names = ['level', 'idx']
        series = series.to_frame('y_true')
    else:
        series = pd.concat(series, names = ['level', 'idx']).to_frame('y_true')
    
    predictions = predictions.swaplevel()
    predictions = predictions.rename(columns={predictions.columns[0]: 'y_pred'})

    # NOTE: Rows are dropped only by `y_true` and `y_pred`, so the metrics are
    # the same with and without the interval bounds.
    y_true_y_pred = pd.merge(
        series,
        predictions,
        left_index  = True,
        right_index = True,
        how         = "inner",
    )

    # NOTE: The predictions of the dates without observed values are expected
    # to be NaN. A NaN prediction of a date with an observed value (the
    # forecaster could not predict it) is left out of the metrics, which are
    # then calculated with fewer values than predicted, so it is warned.
    is_nan_prediction = (
        y_true_y_pred['y_true'].notna() & y_true_y_pred['y_pred'].isna()
    )
    if is_nan_prediction.any():
        n_nan_predictions = (
            is_nan_prediction[is_nan_prediction]
            .groupby(level='level', sort=False)
            .size()
            .to_dict()
        )
        warnings.warn(
            f"{int(is_nan_prediction.sum())} predictions are NaN at dates with "
            f"observed values and are left out of the metrics, which are "
            f"calculated with fewer values than predicted. Number of NaN "
            f"predictions per series: {n_nan_predictions}. The forecaster "
            f"could not predict these dates, for example because of missing "
            f"values in the last window or in `exog`.",
            MissingValuesWarning
        )

    y_true_y_pred = y_true_y_pred.dropna(
        axis=0, how="any", subset=['y_true', 'y_pred']
    )

    if needs_y_train:
        train_indexes = []
        for i, fold in enumerate(folds):
            fit_fold = fold[-1]
            if i == 0 or fit_fold:
                train_iloc_start = fold[1][0]
                train_iloc_end = fold[1][1]
                train_indexes.append(np.arange(train_iloc_start, train_iloc_end))
        
        train_indexes = np.unique(np.concatenate(train_indexes))
        train_indexes = span_index[train_indexes]
        train_indexes = pd.MultiIndex.from_product([
            levels_in_predictions,
            train_indexes,
        ])
        series_train = series.loc[series.index.isin(train_indexes)]
        # NOTE: Exclude first window_size observations used to create predictors
        series_train = series_train[
            series_train.groupby(level="level").cumcount() >= window_size
        ]
        series_train_grouped = (
            series_train
            .reset_index(level="level")
            .groupby(by="level", sort=False, as_index=False)
        )

    y_true_y_pred_grouped = (
        y_true_y_pred
        .reset_index(level="level")
        .groupby(by="level", sort=False, as_index=False)
    )
    metrics_levels = []
    for level in levels:
        if level in levels_in_predictions and level in y_true_y_pred_grouped.groups:
            group = y_true_y_pred_grouped.get_group(level)
            y_true = group['y_true']
            y_pred = group['y_pred']
            if needs_y_train:
                # NOTE: y_train includes the intercepted NaNs
                y_train = series_train_grouped.get_group(level)['y_true']
            else:
                y_train = None
            metrics_level = [
                m(y_true=y_true, y_pred=y_pred, y_train=y_train)
                for m in metrics
            ]
            if add_interval_metrics:
                metrics_level.extend(
                    _calculate_interval_metrics(
                        y_true      = y_true,
                        lower_bound = group['lower_bound'],
                        upper_bound = group['upper_bound']
                    )
                )
            metrics_levels.append(metrics_level)
        else:
            metrics_levels.append([None for _ in output_names])

    metrics_levels = pd.DataFrame(
                         data    = metrics_levels,
                         columns = output_names
                     )
    metrics_levels.insert(0, 'levels', levels)

    if len(levels) < 2:
        add_aggregated_metric = False

    if add_aggregated_metric:

        # aggregation: average
        average = metrics_levels.drop(columns='levels').mean(skipna=True)
        average = average.to_frame().transpose()
        average['levels'] = 'average'

        # aggregation: weighted_average
        n_predictions_levels = (
            predictions
            .groupby(level="level", sort=False)["y_pred"]
            .apply(lambda x: x.notna().sum())
            .reset_index(name="n_predictions")
            .rename(columns={"level": "levels"})
        )
        # NOTE: A level predicted without observed values in its test set (for
        # example, a series that ends at the end of the training set) has no
        # metric, so it is left out of the weighted average, as it is of the
        # average and the pooling.
        n_predictions_levels = n_predictions_levels[
            n_predictions_levels['levels'].isin(y_true_y_pred_grouped.groups)
        ]
        metrics_levels_no_missing = (
            metrics_levels.merge(n_predictions_levels, on='levels', how='inner')
        )
        weighted_averages = {}
        for col in output_names:
            weighted_averages[col] = (
                np.average(
                    metrics_levels_no_missing[col],
                    weights = metrics_levels_no_missing['n_predictions']
                )
                if not metrics_levels_no_missing.empty
                else np.nan
            )
        weighted_average = pd.DataFrame(weighted_averages, index=[0])
        weighted_average['levels'] = 'weighted_average'

        # aggregation: pooling
        scaled_metrics = [
            'mean_absolute_scaled_error', 
            'root_mean_squared_scaled_error'
        ]

        y_true = y_true_y_pred.loc[:, 'y_true'].droplevel("level")
        y_pred = y_true_y_pred.loc[:, 'y_pred'].droplevel("level")
        if needs_y_train:
            y_train = series_train.loc[:, 'y_true'].droplevel("level")
            if any(name in scaled_metrics for name in metric_names):
                series_train_list = [
                    group['y_true'] 
                    for _, group in series_train.groupby('level', sort=False)
                ]
        else:
            y_train = None

        pooled = []
        for m, m_name in zip(metrics, metric_names):
            if needs_y_train and m_name in scaled_metrics:
                pooled.append(
                    m(y_true=y_true, y_pred=y_pred, y_train=series_train_list)
                )
            else:
                pooled.append(
                    m(y_true=y_true, y_pred=y_pred, y_train=y_train)
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
    predicted values of each level.
    - 'pooling': the values of all levels are pooled and then the metric is
    calculated.

    Parameters
    ----------
    forecaster : object
        Forecaster model.
    series : pandas DataFrame, dict
        Series data used to train and test the forecaster.
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
        predicted values of each level.
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
        Value(s) of the metric(s).
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

        if needs_y_train:
            y_train_metric = y_train.copy()
        else:
            y_train_metric = None

        if forecaster.differentiation is not None:
            differentiator = forecaster.differentiator_[level]
            if differentiator is not None:
                y_true = differentiator.inverse_transform_next_window(y_true)
                y_pred = differentiator.inverse_transform_next_window(y_pred)
                if needs_y_train:
                    y_train_metric = differentiator.inverse_transform_training(
                        y_train_metric
                    )

        if forecaster.transformer_series is not None:
            transformer = forecaster.transformer_series_[level]
            y_true = transformer.inverse_transform(
                y_true.reshape(-1, 1)
            ).ravel()
            y_pred = transformer.inverse_transform(
                y_pred.reshape(-1, 1)
            ).ravel()
            if needs_y_train:
                y_train_metric = transformer.inverse_transform(
                    y_train_metric.reshape(-1, 1)
                ).ravel()

        metrics_level = [
            m(y_true=y_true, y_pred=y_pred, y_train=y_train_metric)
            for m in metrics
        ]

        metrics_levels = pd.DataFrame(
            data=[metrics_level], columns=metric_names
        )
        metrics_levels.insert(0, 'levels', [level])

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

    if needs_y_train:
        
        y_train_per_level = pd.DataFrame(
            {"y_train": y_train, "_level_skforecast": X_train_encoding},
            index=y_train.index,
        ).groupby("_level_skforecast")

        # NOTE: Interleaved Nan values were excluded fom y_train. They are restored
        if isinstance(series[levels[0]].index, pd.DatetimeIndex):
            y_train_per_level = {
                key: group.reindex(
                    _date_range_from_index(
                        index = group.index,
                        start = group.index[0],
                        end   = group.index[-1],
                        freq  = freq
                    )
                )
                for key, group in y_train_per_level
            }
        else:
            y_train_per_level = {
                key: group.reindex(
                    pd.RangeIndex(
                        start=min(group.index), stop=max(group.index) + 1, step=freq
                    )
                )
                for key, group in y_train_per_level
            }

    if forecaster.differentiation is not None:
        for level in predictions_per_level:
            differentiator = forecaster.differentiator_[level]
            if differentiator is not None:
                predictions_per_level[level]["y_true"] = (
                    differentiator.inverse_transform_next_window(
                        predictions_per_level[level]["y_true"].to_numpy()
                    )
                )
                predictions_per_level[level]["y_pred"] = (
                    differentiator.inverse_transform_next_window(
                        predictions_per_level[level]["y_pred"].to_numpy()
                    )   
                )
                if needs_y_train:
                    y_train_per_level[level]["y_train"] = (
                        differentiator.inverse_transform_training(
                            y_train_per_level[level]["y_train"].to_numpy()
                        )
                    )

    if forecaster.transformer_series is not None:
        for level in predictions_per_level:
            transformer = forecaster.transformer_series_[level]
            predictions_per_level[level]["y_true"] = transformer.inverse_transform(
                predictions_per_level[level][["y_true"]]
            )
            predictions_per_level[level]["y_pred"] = transformer.inverse_transform(
                predictions_per_level[level][["y_pred"]]
            )
            if needs_y_train:
                y_train_per_level[level]["y_train"] = transformer.inverse_transform(
                    y_train_per_level[level][["y_train"]]
                )

    metrics_levels = []
    for level in levels:
        if level in predictions_per_level:
            y_true = predictions_per_level[level].loc[:, 'y_true']
            y_pred = predictions_per_level[level].loc[:, 'y_pred']
            if needs_y_train:
                y_train = y_train_per_level[level].loc[:, 'y_train']
            else:
                y_train = None
            metrics_level = [
                m(y_true=y_true, y_pred=y_pred, y_train=y_train) 
                for m in metrics
            ]
            metrics_levels.append(metrics_level)
        else:
            metrics_levels.append([None for _ in metrics])

    metrics_levels = pd.DataFrame(
                         data    = metrics_levels,
                         columns = metric_names
                     )
    metrics_levels.insert(0, 'levels', levels)

    if len(levels) < 2:
        add_aggregated_metric = False

    if add_aggregated_metric:

        # aggregation: average
        average = metrics_levels.drop(columns='levels').mean(skipna=True)
        average = average.to_frame().transpose()
        average['levels'] = 'average'

        # aggregation: weighted_average
        n_predictions_levels = {
            k: v['y_pred'].notna().sum()
            for k, v in predictions_per_level.items()
        }
        n_predictions_levels = pd.DataFrame(
            n_predictions_levels.items(),
            columns=['levels', 'n_predictions']
        )
        metrics_levels_no_missing = (
            metrics_levels.merge(n_predictions_levels, on='levels', how='inner')
        )
        weighted_averages = {}
        for col in metric_names:
            weighted_averages[col] = np.average(
                metrics_levels_no_missing[col],
                weights = metrics_levels_no_missing['n_predictions']
            )
        weighted_average = pd.DataFrame(weighted_averages, index=[0])
        weighted_average['levels'] = 'weighted_average'

        # aggregation: pooling
        scaled_metrics = ['mean_absolute_scaled_error', 'root_mean_squared_scaled_error']
        predictions_pooled = pd.concat(predictions_per_level.values())
        if needs_y_train:
            y_train_pooled = pd.concat(
                [v for k, v in y_train_per_level.items() if k in predictions_per_level]
            )
            if any(name in scaled_metrics for name in metric_names):
                list_y_train_by_level = [
                    v['y_train'].to_numpy()
                    for k, v in y_train_per_level.items()
                    if k in predictions_per_level
                ]
        else:
            y_train_pooled = None

        pooled = []
        y_true = predictions_pooled['y_true']
        y_pred = predictions_pooled['y_pred']
        for m, m_name in zip(metrics, metric_names):
            if needs_y_train and m_name in scaled_metrics:
                pooled.append(
                    m(y_true=y_true, y_pred=y_pred, y_train=list_y_train_by_level)
                )
            else:
                pooled.append(
                    m(
                        y_true=y_true,
                        y_pred=y_pred,
                        y_train=(
                            y_train_pooled["y_train"]
                            if y_train_pooled is not None
                            else None
                        ),
                    )
                )
        pooled = pd.DataFrame([pooled], columns=metric_names)
        pooled['levels'] = 'pooling'

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
                UserWarning
            )

    return direction
