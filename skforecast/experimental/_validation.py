################################################################################
#                  skforecast.model_selection._validation                      #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
from typing import Callable
from copy import deepcopy
from itertools import chain
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from tqdm.auto import tqdm
from ..stats import Arar
from ..metrics import add_y_train_argument, _get_metric
from ..exceptions import LongTrainingWarning, IgnoredArgumentWarning
from ..model_selection._split import TimeSeriesFold
from ..model_selection._utils import (
    _initialize_levels_model_selection_multiseries,
    check_backtesting_input,
    select_n_jobs_backtesting,
    _extract_data_folds_multiseries,
    _calculate_metrics_backtesting_multiseries
)
from ..utils import (
    check_preprocess_series,
    check_preprocess_exog_multiseries,
    set_skforecast_warnings
)


def _backtesting_stats(
    forecaster: object,
    y: pd.Series,
    metric: str | Callable | list[str | Callable],
    cv: TimeSeriesFold,
    exog: pd.Series | pd.DataFrame | None = None,
    alpha: float | None = None,
    interval: list[float] | tuple[float] | None = None,
    n_jobs: int | str = 'auto',
    suppress_warnings_fit: bool = False,
    verbose: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting of ForecasterSarimax.
    
    A copy of the original forecaster is created so that it is not modified during 
    the process.
    
    Parameters
    ----------
    forecaster : ForecasterSarimax
        Forecaster model.
    y : pandas Series
        Training time series.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    cv : TimeSeriesFold
        TimeSeriesFold object with the information needed to split the data into folds.
    exog : pandas Series, pandas DataFrame, default None
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    alpha : float, default 0.05
        The confidence intervals for the forecasts are (1 - alpha) %.
        If both, `alpha` and `interval` are provided, `alpha` will be used.
    interval : list, tuple, default None
        Confidence of the prediction interval estimated. The values must be
        symmetric. Sequence of percentiles to compute, which must be between 
        0 and 100 inclusive. For example, interval of 95% should be as 
        `interval = [2.5, 97.5]`. If both, `alpha` and `interval` are 
        provided, `alpha` will be used.
    n_jobs : int, 'auto', default 'auto'
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default False
        Print number of folds and index of training and validation sets used 
        for backtesting.
    suppress_warnings_fit : bool, default False
        If `True`, warnings generated during fitting will be ignored.
    show_progress : bool, default True
        Whether to show a progress bar.

    Returns
    -------
    metric_values : pandas DataFrame
        Value(s) of the metric(s).
    backtest_predictions : pandas DataFrame
        Value of predictions. The  DataFrame includes the following columns:

        - fold: Indicates the fold number where the prediction was made.
        - pred: Predicted values for the corresponding series and time steps.

        If `interval` is not `None`, additional columns are included:

        - lower_bound: lower bound of the interval.
        - upper_bound: upper bound of the interval.

        Depending on the relation between `steps` and `fold_stride`, the output
        may include repeated indexes (if `fold_stride < steps`) or gaps
        (if `fold_stride > steps`). See Notes below for more details.

    Notes
    -----
    Note on `fold_stride` vs. `steps`:

    - If `fold_stride == steps`, test sets are placed back-to-back without overlap. 
    Each observation appears only once in the output DataFrame, so the index is unique.
    - If `fold_stride < steps`, test sets overlap. Multiple forecasts are generated 
    for the same observations and, therefore, the output DataFrame contains repeated 
    indexes.
    - If `fold_stride > steps`, there are gaps between consecutive test sets. 
    Some observations in the series will not have associated predictions, so 
    the output DataFrame has non-contiguous indexes.
    
    """

    forecaster = deepcopy(forecaster)
    cv = deepcopy(cv)

    if isinstance(forecaster.regressor, Arar):
        if cv.refit is False:
            warnings.warn(
                "If `ForecasterStats` uses `Arar` as regressor, `cv.refit` must be "
                "`True` since predictions must start from the end of the training set."
                " Setting `cv.refit = True`.",
                IgnoredArgumentWarning
            )
            cv.refit = True

    cv.set_params({
        'window_size': forecaster.window_size,
        'return_all_indexes': False,
        'verbose': verbose
    })

    refit = cv.refit
    overlapping_folds = cv.overlapping_folds
    
    if refit == False:
        if n_jobs != 'auto' and n_jobs != 1:
            warnings.warn(
                "If `refit = False`, `n_jobs` is set to 1 to avoid unexpected "
                "results during parallelization.",
                IgnoredArgumentWarning
            )
        n_jobs = 1
    else:
        if n_jobs == 'auto':        
            n_jobs = select_n_jobs_backtesting(
                         forecaster = forecaster,
                         refit      = refit
                     )
        elif not isinstance(refit, bool) and refit != 1 and n_jobs != 1:
            warnings.warn(
                "If `refit` is an integer other than 1 (intermittent refit). `n_jobs` "
                "is set to 1 to avoid unexpected results during parallelization.",
                IgnoredArgumentWarning
            )
            n_jobs = 1
        else:
            n_jobs = n_jobs if n_jobs > 0 else cpu_count()

    if not isinstance(metric, list):
        metrics = [
            _get_metric(metric=metric)
            if isinstance(metric, str)
            else add_y_train_argument(metric)
        ]
    else:
        metrics = [
            _get_metric(metric=m)
            if isinstance(m, str)
            else add_y_train_argument(m) 
            for m in metric
        ]
    
    folds = cv.split(X=y, as_pandas=False)
    initial_train_size = cv.initial_train_size
    steps = cv.steps
    gap = cv.gap

    # NOTE: initial_train_size cannot be None because of append method in Sarimax
    # NOTE: This allows for parallelization when `refit` is `False`. The initial 
    # Forecaster fit occurs outside of the auxiliary function.
    exog_train = exog.iloc[:initial_train_size, ] if exog is not None else None
    forecaster.fit(
        y                 = y.iloc[:initial_train_size, ],
        exog              = exog_train,
        suppress_warnings = suppress_warnings_fit
    )
    folds[0][5] = False
    
    if refit:
        n_of_fits = int(len(folds) / refit)
        if n_of_fits > 50:
            warnings.warn(
                f"The forecaster will be fit {n_of_fits} times. This can take substantial "
                f"amounts of time. If not feasible, try with `refit = False`.\n",
                LongTrainingWarning
            )
       
    folds_tqdm = tqdm(folds) if show_progress else folds

    def _fit_predict_forecaster(
        y, exog, forecaster, alpha, interval, fold, steps, gap
    ) -> pd.DataFrame:
        """
        Fit the forecaster and predict `steps` ahead. This is an auxiliary 
        function used to parallelize the backtesting_forecaster function.
        """

        # In each iteration the model is fitted before making predictions. 
        # if fixed_train_size the train size doesn't increase but moves by `steps` 
        # in each iteration. if False the train size increases by `steps` in each 
        # iteration.
        train_iloc_start = fold[1][0]
        train_iloc_end   = fold[1][1]
        test_iloc_start  = fold[3][0]
        test_iloc_end    = fold[3][1]

        if refit:
            last_window_iloc_start = fold[1][1]  # Same as train_iloc_end
            last_window_iloc_end   = fold[2][1]
        else:
            last_window_iloc_end   = fold[3][0]  # test_iloc_start
            last_window_iloc_start = last_window_iloc_end - steps

        if fold[5] is False:
            # When the model is not fitted, last_window and last_window_exog must 
            # be updated to include the data needed to make predictions.
            last_window_y = y.iloc[last_window_iloc_start:last_window_iloc_end]
            last_window_exog = exog.iloc[last_window_iloc_start:last_window_iloc_end] if exog is not None else None 
        else:
            # The model is fitted before making predictions. If `fixed_train_size`  
            # the train size doesn't increase but moves by `steps` in each iteration. 
            # If `False` the train size increases by `steps` in each  iteration.
            y_train = y.iloc[train_iloc_start:train_iloc_end, ]
            exog_train = exog.iloc[train_iloc_start:train_iloc_end, ] if exog is not None else None
            
            last_window_y = None
            last_window_exog = None

            forecaster.fit(y=y_train, exog=exog_train, suppress_warnings=suppress_warnings_fit)

        next_window_exog = exog.iloc[test_iloc_start:test_iloc_end, ] if exog is not None else None

        # After the first fit, Sarimax must use the last windows stored in the model
        if fold == folds[0]:
            last_window_y = None
            last_window_exog = None

        steps = len(range(test_iloc_start, test_iloc_end))
        if alpha is None and interval is None:
            pred = forecaster.predict(
                       steps            = steps,
                       last_window      = last_window_y,
                       last_window_exog = last_window_exog,
                       exog             = next_window_exog
                   )
        else:
            pred = forecaster.predict_interval(
                       steps            = steps,
                       exog             = next_window_exog,
                       alpha            = alpha,
                       interval         = interval,
                       last_window      = last_window_y,
                       last_window_exog = last_window_exog
                   )

        pred = pred.iloc[gap:, ]            
        
        return pred

    backtest_predictions = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict_forecaster)(
            y=y,
            exog=exog,
            forecaster=forecaster,
            alpha=alpha,
            interval=interval,
            fold=fold,
            steps=steps,
            gap=gap,
        )
        for fold in folds_tqdm
    )
    fold_labels = [
        np.repeat(fold[0], backtest_predictions[i].shape[0]) for i, fold in enumerate(folds)
    ]
    
    backtest_predictions = pd.concat(backtest_predictions)
    if isinstance(backtest_predictions, pd.Series):
        backtest_predictions = pd.DataFrame(backtest_predictions)
    backtest_predictions.insert(0, 'fold', np.concatenate(fold_labels))

    train_indexes = []
    for i, fold in enumerate(folds):
        fit_fold = fold[-1]
        if i == 0 or fit_fold:
            train_iloc_start = fold[1][0]
            train_iloc_end = fold[1][1]
            train_indexes.append(np.arange(train_iloc_start, train_iloc_end))
    
    train_indexes = np.unique(np.concatenate(train_indexes))
    y_train = y.iloc[train_indexes]

    backtest_predictions_for_metrics = backtest_predictions
    if overlapping_folds:
        backtest_predictions_for_metrics = (
            backtest_predictions_for_metrics
            .loc[~backtest_predictions_for_metrics.index.duplicated(keep='last')]
        )

    metric_values = [
        m(
            y_true = y.loc[backtest_predictions_for_metrics.index],
            y_pred = backtest_predictions_for_metrics['pred'],
            y_train = y_train
        ) 
        for m in metrics
    ]

    metric_values = pd.DataFrame(
        data    = [metric_values],
        columns = [m.__name__ for m in metrics]
    )

    return metric_values, backtest_predictions


def backtesting_stats(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold,
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    alpha: float | None = None,
    interval: list[float] | tuple[float] | None = None,
    n_jobs: int | str = 'auto',
    verbose: bool = False,
    suppress_warnings_fit: bool = False,
    show_progress: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting of ForecasterSarimax.
    
    A copy of the original forecaster is created so that it is not modified during 
    the process.

    Parameters
    ----------
    forecaster : ForecasterSarimax
        Forecaster model.
    y : pandas Series
        Training time series.
    cv : TimeSeriesFold
        TimeSeriesFold object with the information needed to split the data into folds.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    exog : pandas Series, pandas DataFrame, default None
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    alpha : float, default 0.05
        The confidence intervals for the forecasts are (1 - alpha) %.
        If both, `alpha` and `interval` are provided, `alpha` will be used.
    interval : list, tuple, default None
        Confidence of the prediction interval estimated. The values must be
        symmetric. Sequence of percentiles to compute, which must be between 
        0 and 100 inclusive. For example, interval of 95% should be as 
        `interval = [2.5, 97.5]`. If both, `alpha` and `interval` are 
        provided, `alpha` will be used.
    n_jobs : int, 'auto', default 'auto'
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting. 
    verbose : bool, default False
        Print number of folds and index of training and validation sets used 
        for backtesting.
    suppress_warnings_fit : bool, default False
        If `True`, warnings generated during fitting will be ignored.
    show_progress : bool, default True
        Whether to show a progress bar.

    Returns
    -------
    metric_values : pandas DataFrame
        Value(s) of the metric(s).
    backtest_predictions : pandas DataFrame
        Value of predictions. The  DataFrame includes the following columns:

        - fold: Indicates the fold number where the prediction was made.
        - pred: Predicted values for the corresponding series and time steps.

        If `interval` is not `None`, additional columns are included:
        
        - lower_bound: lower bound of the interval.
        - upper_bound: upper bound of the interval.

        Depending on the relation between `steps` and `fold_stride`, the output
        may include repeated indexes (if `fold_stride < steps`) or gaps
        (if `fold_stride > steps`). See Notes below for more details.

    Notes
    -----
    Note on `fold_stride` vs. `steps`:

    - If `fold_stride == steps`, test sets are placed back-to-back without overlap. 
    Each observation appears only once in the output DataFrame, so the index is unique.
    - If `fold_stride < steps`, test sets overlap. Multiple forecasts are generated 
    for the same observations and, therefore, the output DataFrame contains repeated 
    indexes.
    - If `fold_stride > steps`, there are gaps between consecutive test sets. 
    Some observations in the series will not have associated predictions, so 
    the output DataFrame has non-contiguous indexes.
    
    """
    
    if type(forecaster).__name__ not in ['ForecasterSarimax', 'ForecasterStats']:
        raise TypeError(
            "`forecaster` must be of type `ForecasterSarimax`, for all other "
            "types of forecasters use the functions available in the other "
            "`model_selection` modules."
        )
    
    check_backtesting_input(
        forecaster            = forecaster,
        cv                    = cv,
        y                     = y,
        metric                = metric,
        interval              = interval,
        alpha                 = alpha,
        n_jobs                = n_jobs,
        show_progress         = show_progress,
        suppress_warnings_fit = suppress_warnings_fit
    )
    
    metric_values, backtest_predictions = _backtesting_stats(
        forecaster            = forecaster,
        y                     = y,
        cv                    = cv,
        metric                = metric,
        exog                  = exog,
        alpha                 = alpha,
        interval              = interval,
        n_jobs                = n_jobs,
        verbose               = verbose,
        suppress_warnings_fit = suppress_warnings_fit,
        show_progress         = show_progress
    )

    return metric_values, backtest_predictions
