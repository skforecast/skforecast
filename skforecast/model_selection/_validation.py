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
    manage_warnings,
    deepcopy_forecaster
)
from ..foundational._utils import check_preprocess_series_type


def _prepare_fold_data(
    folds: list[list],
    y: pd.Series,
    exog: pd.Series | pd.DataFrame | None
) -> list[dict]:
    """
    Pre-slice `y` and `exog` for each fold to minimize IPC serialization
    cost when using `joblib.Parallel`.

    For `refit=False` folds (`fold[5] is False`), only `last_window_y`
    and `exog_test` are needed. For `refit=True` folds (`fold[5] is True`),
    `y_train`, `exog_train`, and `exog_test` are included.

    Parameters
    ----------
    folds : list of list
        Fold metadata as produced by `TimeSeriesFold.split`.
        Each fold: `[fold_number, [train_start, train_end],
        [lw_start, lw_end], [test_start, test_end],
        [test_start_with_gap, test_end_with_gap], fit_forecaster]`.
    y : pandas Series
        Training time series.
    exog : pandas Series, pandas DataFrame, or None
        Exogenous variables (or None).

    Returns
    -------
    fold_data : list of dict
        One dict per fold with keys `'y_train'`, `'last_window_y'`,
        `'exog_train'`, `'exog_test'`. Unused entries are `None`.
    
    """

    fold_data = []
    for fold in folds:
        if fold[5] is False:
            # No refit: worker only needs last_window + test exog
            data = {
                'y_train': None,
                'last_window_y': y.iloc[fold[2][0]:fold[2][1]],
                'exog_train': None,
                'exog_test': (
                    exog.iloc[fold[3][0]:fold[3][1]] if exog is not None else None
                ),
            }
        else:
            # Refit: worker needs training data + test exog
            data = {
                'y_train': y.iloc[fold[1][0]:fold[1][1]],
                'last_window_y': None,
                'exog_train': (
                    exog.iloc[fold[1][0]:fold[1][1]] if exog is not None else None
                ),
                'exog_test': (
                    exog.iloc[fold[3][0]:fold[3][1]] if exog is not None else None
                ),
            }
        fold_data.append(data)

    return fold_data


def _fit_predict_forecaster(
    fold: list,
    y_train: pd.Series | None,
    last_window_y: pd.Series | None,
    exog_train: pd.Series | pd.DataFrame | None,
    exog_test: pd.Series | pd.DataFrame | None,
    forecaster: object,
    store_in_sample_residuals: bool,
    gap: int,
    interval: float | list[float] | tuple[float] | str | object | None,
    interval_method: str,
    n_boot: int,
    use_in_sample_residuals: bool,
    use_binned_residuals: bool,
    out_sample_residuals_: np.ndarray | None,
    out_sample_residuals_by_bin_: dict[int, np.ndarray] | None,
    random_state: int,
    return_predictors: bool,
    is_regression: bool,
    suppress_warnings: bool
) -> pd.DataFrame:
    """
    Fit the forecaster and predict `steps` ahead. This is a module-level
    auxiliary function used to parallelize `_backtesting_forecaster`.

    Defined at module level (instead of as a nested closure) so that
    `joblib.Parallel` can serialize it efficiently with `pickle` rather
    than `cloudpickle`, avoiding unnecessary closure overhead.

    Receives pre-sliced data from `_prepare_fold_data` to minimize 
    Inter-Process Communication (IPC) serialization cost.

    Parameters
    ----------
    fold : list
        Fold metadata as produced by `TimeSeriesFold.split`.
    y_train : pandas Series or None
        Pre-sliced training time series. `None` when `fold[5] is False`
        (no refit).
    last_window_y : pandas Series or None
        Pre-sliced last window of the time series. `None` when
        `fold[5] is True` (refit).
    exog_train : pandas Series, pandas DataFrame, or None
        Pre-sliced training exogenous variables. `None` when no refit
        or when no exogenous variables are used.
    exog_test : pandas Series, pandas DataFrame, or None
        Pre-sliced test exogenous variables. `None` when no exogenous
        variables are used.
    forecaster : object
        Forecaster model.
    store_in_sample_residuals : bool
        Whether to store in-sample residuals during `fit()`.
    gap : int
        Number of observations between training end and test start.
    interval : float, list, tuple, str, object, or None
        Interval specification for probabilistic predictions.
    interval_method : str
        Method for probabilistic predictions ('bootstrapping' or 'conformal').
    n_boot : int
        Number of bootstrap samples.
    use_in_sample_residuals : bool
        Whether to use in-sample residuals for intervals.
    use_binned_residuals : bool
        Whether to bin residuals by predicted value.
    out_sample_residuals_ : np.ndarray, default None
        Pre-validated out-of-sample residuals to restore after each `fit()` call 
        (which resets them to `None`).
    out_sample_residuals_by_bin_ : dict, default None
        Pre-validated out-of-sample residuals indexed by predicted-value bin.
    random_state : int
        Random seed.
    return_predictors : bool
        Whether to return predictor values.
    is_regression : bool
        Whether the forecaster is a regression model.
    suppress_warnings : bool, default False
        If `True`, skforecast warnings are suppressed during execution.
        See `skforecast.exceptions.warn_skforecast_categories` for the
        list of warnings that are suppressed.

    Returns
    -------
    pred : pandas DataFrame
        Predictions for the fold.
    
    """

    test_iloc_start = fold[3][0]
    test_iloc_end   = fold[3][1]

    if fold[5] is True:
        forecaster.fit(
            y                         = y_train,
            exog                      = exog_train,
            store_in_sample_residuals = store_in_sample_residuals,
            suppress_warnings         = suppress_warnings
        )
        if out_sample_residuals_ is not None:
            forecaster.out_sample_residuals_ = out_sample_residuals_
        if out_sample_residuals_by_bin_ is not None:
            forecaster.out_sample_residuals_by_bin_ = out_sample_residuals_by_bin_

    steps = test_iloc_end - test_iloc_start
    if type(forecaster).__name__ == 'ForecasterDirect' and gap > 0:
        # Select only the steps that need to be predicted if gap > 0
        test_no_gap_iloc_start = fold[4][0]
        test_no_gap_iloc_end   = fold[4][1]
        n_steps = test_no_gap_iloc_end - test_no_gap_iloc_start
        steps = list(range(gap + 1, gap + 1 + n_steps))

    preds = []
    if is_regression:
        if interval is not None:
            kwargs_interval = {
                'steps': steps,
                'last_window': last_window_y,
                'exog': exog_test,
                'n_boot': n_boot,
                'use_in_sample_residuals': use_in_sample_residuals,
                'use_binned_residuals': use_binned_residuals,
                'random_state': random_state,
                'suppress_warnings': suppress_warnings
            }
            if interval_method == 'bootstrapping':
                if interval == 'bootstrapping':
                    pred = forecaster.predict_bootstrapping(**kwargs_interval)
                elif isinstance(interval, (float, list, tuple)):
                    if isinstance(interval, float):
                        quantiles = [0.5 - interval / 2, 0.5 + interval / 2]
                    else:
                        quantiles = [q / 100 for q in interval]
                    pred = forecaster.predict_quantiles(quantiles=quantiles, **kwargs_interval)
                    if len(quantiles) == 2:
                        pred.columns = ['lower_bound', 'upper_bound']
                    else:
                        pred.columns = [f'p_{p}' for p in interval]
                else:
                    pred = forecaster.predict_dist(distribution=interval, **kwargs_interval)
                
                preds.append(pred)
            else:
                pred = forecaster.predict_interval(
                    method='conformal', interval=interval, **kwargs_interval
                )
                preds.append(pred)

        # NOTE: This is done after probabilistic predictions to avoid repeating 
        # the same checks.
        if interval is None or interval_method != 'conformal':
            pred = forecaster.predict(
                       steps             = steps,
                       last_window       = last_window_y,
                       exog              = exog_test,
                       check_inputs      = True if interval is None else False,
                       suppress_warnings = suppress_warnings
                   )
            preds.insert(0, pred)
    else:
        pred = forecaster.predict_proba(
                   steps             = steps,
                   last_window       = last_window_y,
                   exog              = exog_test,
                   suppress_warnings = suppress_warnings
               )
        preds.append(pred)

    if return_predictors:
        pred = forecaster.create_predict_X(
                   steps             = steps,
                   last_window       = last_window_y,
                   exog              = exog_test,
                   check_inputs      = False,
                   suppress_warnings = suppress_warnings
               )
        preds.append(pred)

    if len(preds) == 1:
        pred = preds[0]
    else:
        pred = pd.concat(preds, axis=1)

    if type(forecaster).__name__ != 'ForecasterDirect' and gap > 0:
        pred = pred.iloc[gap:, ]
    
    return pred


@manage_warnings
def _backtesting_forecaster(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold,
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    interval: float | list[float] | tuple[float] | str | object | None = None,
    interval_method: str = 'bootstrapping',
    n_boot: int = 250,
    use_in_sample_residuals: bool = True,
    use_binned_residuals: bool = True,
    random_state: int = 123,
    return_predictors: bool = False,
    n_jobs: int | str = 'auto',
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting of forecaster model following the folds generated by the TimeSeriesFold
    class and using the metric(s) provided.

    If `forecaster` is already trained and `initial_train_size` is set to `None` in the
    TimeSeriesFold class, no initial train will be done and all data will be used
    to evaluate the model. However, the first `len(forecaster.last_window)` observations
    are needed to create the initial predictors, so no predictions are calculated for
    them.
    
    A copy of the original forecaster is created so that it is not modified during the
    process.
    
    Parameters
    ----------
    forecaster : ForecasterRecursive, ForecasterDirect, ForecasterEquivalentDate, ForecasterRecursiveClassifier
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
    interval : float, list, tuple, str, object, default None
        Specifies whether probabilistic predictions should be estimated and the 
        method to use. The following options are supported:

        - If `float`, represents the nominal (expected) coverage (between 0 and 1). 
        For instance, `interval=0.95` corresponds to `[2.5, 97.5]` percentiles.
        - If `list` or `tuple`: Sequence of percentiles to compute, each value must 
        be between 0 and 100 inclusive. For example, a 95% confidence interval can 
        be specified as `interval = [2.5, 97.5]` or multiple percentiles (e.g. 10, 
        50 and 90) as `interval = [10, 50, 90]`.
        - If 'bootstrapping' (str): `n_boot` bootstrapping predictions will be generated.
        - If scipy.stats distribution object, the distribution parameters will
        be estimated for each prediction.
        - If None, no probabilistic predictions are estimated.
    interval_method : str, default 'bootstrapping'
        Technique used to estimate prediction intervals. Available options:

        - 'bootstrapping': Bootstrapping is used to generate prediction 
        intervals [1]_.
        - 'conformal': Employs the conformal prediction split method for 
        interval estimation [2]_.
    n_boot : int, default 250
        Number of bootstrapping iterations to perform when estimating prediction
        intervals.
    use_in_sample_residuals : bool, default True
        If `True`, residuals from the training data are used as proxy of
        prediction error to create predictions. 
        If `False`, out of sample residuals (calibration) are used. 
        Out-of-sample residuals must be precomputed using Forecaster's
        `set_out_sample_residuals()` method.
    use_binned_residuals : bool, default True
        If `True`, residuals are selected based on the predicted values 
        (binned selection).
        If `False`, residuals are selected randomly.
    random_state : int, default 123
        Seed for the random number generator to ensure reproducibility.
    return_predictors : bool, default False
        If `True`, the predictors used to make the predictions are also returned.
    n_jobs : int, 'auto', default 'auto'
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default False
        Print number of folds and index of training and validation sets used 
        for backtesting.
    show_progress : bool, default True
        Whether to show a progress bar.
    suppress_warnings : bool, default False
        If `True`, skforecast warnings will be suppressed during the backtesting 
        process. See skforecast.exceptions.warn_skforecast_categories for more
        information.

    Returns
    -------
    metric_values : pandas DataFrame
        Value(s) of the metric(s).
    backtest_predictions : pandas DataFrame
        Value of predictions. The DataFrame includes the following columns:

        - fold: Indicates the fold number where the prediction was made.
        - pred: Predicted values for the corresponding series and time steps.

        If `interval` is not `None`, additional columns are included depending on the method:
        
        - For `float`: Columns `lower_bound` and `upper_bound`.
        - For `list` or `tuple` of 2 elements: Columns `lower_bound` and `upper_bound`.
        - For `list` or `tuple` with multiple percentiles: One column per percentile 
        (e.g., `p_10`, `p_50`, `p_90`).
        - For `'bootstrapping'`: One column per bootstrapping iteration 
        (e.g., `pred_boot_0`, `pred_boot_1`, ..., `pred_boot_n`).
        - For `scipy.stats` distribution objects: One column for each estimated 
        parameter of the distribution (e.g., `loc`, `scale`).

        If `return_predictors` is `True`, one column per predictor is created.

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

    References
    ----------
    .. [1] Forecasting: Principles and Practice (3rd ed) Rob J Hyndman and George Athanasopoulos.
           https://otexts.com/fpp3/prediction-intervals.html
    
    .. [2] MAPIE - Model Agnostic Prediction Interval Estimator.
           https://mapie.readthedocs.io/en/stable/theoretical_description_regression.html#the-split-method
    
    """

    need_out_sample_residuals = (
        interval is not None and not use_in_sample_residuals
    )

    if cv.initial_train_size is not None:
        forecaster = deepcopy_forecaster(
            forecaster, include_out_sample_residuals=need_out_sample_residuals
        )
    else:
        forecaster = deepcopy(forecaster)
    is_regression = forecaster.__skforecast_tags__['forecaster_task'] == 'regression'
    cv = deepcopy(cv)

    cv.set_params({
        'window_size': forecaster.window_size,
        'differentiation': forecaster.differentiation_max,
        'return_all_indexes': False,
        'verbose': verbose
    })

    refit = cv.refit
    overlapping_folds = cv.overlapping_folds

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

    store_in_sample_residuals = True if use_in_sample_residuals else False
    if interval is None:
        forecaster._probabilistic_mode = False
    elif use_binned_residuals:
        forecaster._probabilistic_mode = 'binned'
    else:
        forecaster._probabilistic_mode = 'no_binned'

    folds = cv.split(X=y, as_pandas=False)
    initial_train_size = cv.initial_train_size
    window_size = cv.window_size
    gap = cv.gap

    # Save out-of-sample residuals before any fit() call. Since fit() resets
    # them to None, they must be preserved and restored after each fit so that
    # probabilistic predictions with use_in_sample_residuals=False keep working.
    out_sample_residuals_ = None
    out_sample_residuals_by_bin_ = None
    if need_out_sample_residuals:
        if use_binned_residuals:
            out_sample_residuals_by_bin_ = forecaster.out_sample_residuals_by_bin_
        else:
            out_sample_residuals_ = forecaster.out_sample_residuals_

    if initial_train_size is not None:
        # NOTE: This allows for parallelization when `refit` is `False`. The initial 
        # Forecaster fit occurs outside of the auxiliary function.
        exog_train = exog.iloc[:initial_train_size, ] if exog is not None else None
        forecaster.fit(
            y                         = y.iloc[:initial_train_size, ],
            exog                      = exog_train,
            store_in_sample_residuals = store_in_sample_residuals,
            suppress_warnings         = suppress_warnings
        )
        if out_sample_residuals_ is not None:
            forecaster.out_sample_residuals_ = out_sample_residuals_
        if out_sample_residuals_by_bin_ is not None:
            forecaster.out_sample_residuals_by_bin_ = out_sample_residuals_by_bin_
        folds[0][5] = False

    if refit:
        n_of_fits = int(len(folds) / refit)
        if type(forecaster).__name__ != 'ForecasterDirect' and n_of_fits > 50:
            warnings.warn(
                f"The forecaster will be fit {n_of_fits} times. This can take substantial"
                f" amounts of time. If not feasible, try with `refit = False`.\n",
                LongTrainingWarning
            )
        elif type(forecaster).__name__ == 'ForecasterDirect' and n_of_fits * forecaster.max_step > 50:
            warnings.warn(
                f"The forecaster will be fit {n_of_fits * forecaster.max_step} times "
                f"({n_of_fits} folds * {forecaster.max_step} estimators). This can take "
                f"substantial amounts of time. If not feasible, try with `refit = False`.\n",
                LongTrainingWarning
            )

    fold_data_list = _prepare_fold_data(folds, y, exog)
    fold_items = list(zip(folds, fold_data_list))
    if show_progress:
        fold_items = tqdm(fold_items)

    kwargs_fit_predict_forecaster = {
        "forecaster": forecaster,
        "store_in_sample_residuals": store_in_sample_residuals,
        "gap": gap,
        "interval": interval,
        "interval_method": interval_method,
        "n_boot": n_boot,
        "use_in_sample_residuals": use_in_sample_residuals,
        "use_binned_residuals": use_binned_residuals,
        "out_sample_residuals_": out_sample_residuals_,
        "out_sample_residuals_by_bin_": out_sample_residuals_by_bin_,
        "random_state": random_state,
        "return_predictors": return_predictors,
        'is_regression': is_regression,
        "suppress_warnings": suppress_warnings
    }
    backtest_predictions = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict_forecaster)(
            fold          = fold,
            y_train       = fold_data['y_train'],
            last_window_y = fold_data['last_window_y'],
            exog_train    = fold_data['exog_train'],
            exog_test     = fold_data['exog_test'],
            **kwargs_fit_predict_forecaster
        )
        for fold, fold_data in fold_items
    )
    fold_labels = [
        np.repeat(fold[0], backtest_predictions[i].shape[0]) for i, fold in enumerate(folds)
    ]

    backtest_predictions = pd.concat(backtest_predictions)
    if isinstance(backtest_predictions, pd.Series):
        backtest_predictions = backtest_predictions.to_frame()

    if not is_regression:
        proba_cols = [f"{cls}_proba" for cls in forecaster.classes_]
        idx_max = backtest_predictions[proba_cols].to_numpy().argmax(axis=1)
        backtest_predictions.insert(0, "pred", np.array(forecaster.classes_)[idx_max])

    backtest_predictions.insert(0, 'fold', np.concatenate(fold_labels))

    train_indexes = []
    for i, fold in enumerate(folds):
        fit_fold = fold[-1]
        if i == 0 or fit_fold:
            # NOTE: When using a scaled metric, `y_train` doesn't include the
            # first window_size observations used to create the predictors and/or
            # rolling features.
            train_iloc_start = fold[1][0] + window_size
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

    y_true = y.loc[backtest_predictions_for_metrics.index]
    y_pred = backtest_predictions_for_metrics['pred']
    metric_values = [[
        m(y_true=y_true, y_pred=y_pred, y_train=y_train) 
        for m in metrics
    ]]

    metric_values = pd.DataFrame(
        data    = metric_values,
        columns = [m.__name__ for m in metrics]
    )

    return metric_values, backtest_predictions


def backtesting_forecaster(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold,
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    interval: float | list[float] | tuple[float] | str | object | None = None,
    interval_method: str = 'bootstrapping',
    n_boot: int = 250,
    use_in_sample_residuals: bool = True,
    use_binned_residuals: bool = True,
    random_state: int = 123,
    return_predictors: bool = False,
    n_jobs: int | str = 'auto',
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting of forecaster model following the folds generated by the TimeSeriesFold
    class and using the metric(s) provided.

    If `forecaster` is already trained and `initial_train_size` is set to `None` in the
    TimeSeriesFold class, no initial train will be done and all data will be used
    to evaluate the model. However, the first `len(forecaster.last_window)` observations
    are needed to create the initial predictors, so no predictions are calculated for
    them.
    
    A copy of the original forecaster is created so that it is not modified during 
    the process.

    Parameters
    ----------
    forecaster : ForecasterRecursive, ForecasterDirect, ForecasterEquivalentDate, ForecasterRecursiveClassifier
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
    interval : float, list, tuple, str, object, default None
        Specifies whether probabilistic predictions should be estimated and the 
        method to use. The following options are supported:

        - If `float`, represents the nominal (expected) coverage (between 0 and 1). 
        For instance, `interval=0.95` corresponds to `[2.5, 97.5]` percentiles.
        - If `list` or `tuple`: Sequence of percentiles to compute, each value must 
        be between 0 and 100 inclusive. For example, a 95% confidence interval can 
        be specified as `interval = [2.5, 97.5]` or multiple percentiles (e.g. 10, 
        50 and 90) as `interval = [10, 50, 90]`.
        - If 'bootstrapping' (str): `n_boot` bootstrapping predictions will be generated.
        - If scipy.stats distribution object, the distribution parameters will
        be estimated for each prediction.
        - If None, no probabilistic predictions are estimated.
    interval_method : str, default 'bootstrapping'
        Technique used to estimate prediction intervals. Available options:

        - 'bootstrapping': Bootstrapping is used to generate prediction 
        intervals [1]_.
        - 'conformal': Employs the conformal prediction split method for 
        interval estimation [2]_.
    n_boot : int, default 250
        Number of bootstrapping iterations to perform when estimating prediction
        intervals.
    use_in_sample_residuals : bool, default True
        If `True`, residuals from the training data are used as proxy of
        prediction error to create predictions. 
        If `False`, out of sample residuals (calibration) are used. 
        Out-of-sample residuals must be precomputed using Forecaster's
        `set_out_sample_residuals()` method.
    use_binned_residuals : bool, default True
        If `True`, residuals are selected based on the predicted values 
        (binned selection).
        If `False`, residuals are selected randomly.
    random_state : int, default 123
        Seed for the random number generator to ensure reproducibility.
    return_predictors : bool, default False
        If `True`, the predictors used to make the predictions are also returned.
    n_jobs : int, 'auto', default 'auto'
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default False
        Print number of folds and index of training and validation sets used 
        for backtesting.
    show_progress : bool, default True
        Whether to show a progress bar.
    suppress_warnings : bool, default False
        If `True`, skforecast warnings will be suppressed during the backtesting 
        process. See skforecast.exceptions.warn_skforecast_categories for more
        information.

    Returns
    -------
    metric_values : pandas DataFrame
        Value(s) of the metric(s).
    backtest_predictions : pandas DataFrame
        Value of predictions. The DataFrame includes the following columns:

        - fold: Indicates the fold number where the prediction was made.
        - pred: Predicted values for the corresponding series and time steps.

        If `interval` is not `None`, additional columns are included depending on the method:
        
        - For `float`: Columns `lower_bound` and `upper_bound`.
        - For `list` or `tuple` of 2 elements: Columns `lower_bound` and `upper_bound`.
        - For `list` or `tuple` with multiple percentiles: One column per percentile 
        (e.g., `p_10`, `p_50`, `p_90`).
        - For `'bootstrapping'`: One column per bootstrapping iteration 
        (e.g., `pred_boot_0`, `pred_boot_1`, ..., `pred_boot_n`).
        - For `scipy.stats` distribution objects: One column for each estimated 
        parameter of the distribution (e.g., `loc`, `scale`).

        If `return_predictors` is `True`, one column per predictor is created.

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

    References
    ----------
    .. [1] Forecasting: Principles and Practice (3rd ed) Rob J Hyndman and George Athanasopoulos.
           https://otexts.com/fpp3/prediction-intervals.html
    
    .. [2] MAPIE - Model Agnostic Prediction Interval Estimator.
           https://mapie.readthedocs.io/en/stable/theoretical_description_regression.html#the-split-method
    
    """

    forecasters_allowed = [
        'ForecasterRecursive', 
        'ForecasterDirect',
        'ForecasterEquivalentDate',
        'ForecasterRecursiveClassifier'
    ]
    
    if type(forecaster).__name__ not in forecasters_allowed:
        raise TypeError(
            f"`forecaster` must be of type {forecasters_allowed}. For all other "
            f"types of forecasters use the other functions available in the "
            f"`model_selection` module."
        )
    
    check_backtesting_input(
        forecaster              = forecaster,
        cv                      = cv,
        y                       = y,
        metric                  = metric,
        interval                = interval,
        interval_method         = interval_method,
        n_boot                  = n_boot,
        use_in_sample_residuals = use_in_sample_residuals,
        use_binned_residuals    = use_binned_residuals,
        random_state            = random_state,
        return_predictors       = return_predictors,
        n_jobs                  = n_jobs,
        show_progress           = show_progress,
        suppress_warnings       = suppress_warnings
    )
    
    metric_values, backtest_predictions = _backtesting_forecaster(
        forecaster              = forecaster,
        y                       = y,
        cv                      = cv,
        metric                  = metric,
        exog                    = exog,
        interval                = interval,
        interval_method         = interval_method,
        n_boot                  = n_boot,
        use_in_sample_residuals = use_in_sample_residuals,
        use_binned_residuals    = use_binned_residuals,
        random_state            = random_state,
        return_predictors       = return_predictors,
        n_jobs                  = n_jobs,
        verbose                 = verbose,
        show_progress           = show_progress,
        suppress_warnings       = suppress_warnings
    )

    return metric_values, backtest_predictions


def _fit_predict_forecaster_multiseries(
    data_fold: tuple,
    forecaster: object,
    store_in_sample_residuals: bool,
    levels: list[str],
    gap: int,
    interval: float | list[float] | tuple[float] | str | object | None,
    interval_method: str,
    n_boot: int,
    use_in_sample_residuals: bool,
    use_binned_residuals: bool,
    out_sample_residuals_: dict[str, np.ndarray] | None,
    out_sample_residuals_by_bin_: dict[str, dict[int, np.ndarray]] | None,
    random_state: int,
    return_predictors: bool,
    suppress_warnings: bool
) -> tuple[pd.DataFrame, list[str]]:
    """
    Fit the forecaster and predict `steps` ahead. This is a module-level
    auxiliary function used to parallelize `_backtesting_forecaster_multiseries`.

    Defined at module level (instead of as a nested closure) so that
    `joblib.Parallel` can serialize it efficiently with `pickle` rather
    than `cloudpickle`, avoiding unnecessary closure overhead.

    Parameters
    ----------
    data_fold : tuple
        Pre-extracted data for the fold as produced by
        `_extract_data_folds_multiseries`. Contains:
        `(series_train, last_window_series, last_window_levels,
        exog_train, exog_test, fold)`.
        For `refit=False` folds, `series_train` and `exog_train` are `None`.
    forecaster : object
        Forecaster model.
    store_in_sample_residuals : bool
        Whether to store in-sample residuals during `fit()`.
    levels : list of str
        Time series levels to predict.
    gap : int
        Number of observations between training end and test start.
    interval : float, list, tuple, str, object, or None
        Interval specification for probabilistic predictions.
    interval_method : str
        Method for probabilistic predictions ('bootstrapping' or 'conformal').
    n_boot : int
        Number of bootstrap samples.
    use_in_sample_residuals : bool
        Whether to use in-sample residuals for intervals.
    use_binned_residuals : bool
        Whether to bin residuals by predicted value.
    out_sample_residuals_ : dict, default None
        Pre-validated out-of-sample residuals to restore after each `fit()` call 
        (which resets them to `None`).
    out_sample_residuals_by_bin_ : dict, default None
        Pre-validated out-of-sample residuals indexed by predicted-value bin.
    random_state : int
        Random seed.
    return_predictors : bool
        Whether to return predictor values.
    suppress_warnings : bool
        Whether to suppress skforecast warnings.

    Returns
    -------
    pred : pandas DataFrame
        Predictions for the fold.
    levels_predict : list of str
        Levels predicted in this fold.
    
    """

    (
        series_train,
        last_window_series,
        last_window_levels,
        exog_train,
        exog_test,
        fold
    ) = data_fold

    if fold[5] is True:
        forecaster.fit(
            series                    = series_train, 
            exog                      = exog_train,
            store_last_window         = last_window_levels,
            store_in_sample_residuals = store_in_sample_residuals,
            suppress_warnings         = suppress_warnings
        )
        if out_sample_residuals_ is not None:
            forecaster.out_sample_residuals_ = out_sample_residuals_
        if out_sample_residuals_by_bin_ is not None:
            forecaster.out_sample_residuals_by_bin_ = out_sample_residuals_by_bin_

    if type(forecaster).__name__ == 'ForecasterDirectMultiVariate' and gap > 0:
        # Select only the steps that need to be predicted if gap > 0
        test_no_gap_iloc_start = fold[4][0]
        test_no_gap_iloc_end   = fold[4][1]
        n_steps = test_no_gap_iloc_end - test_no_gap_iloc_start
        steps = list(range(gap + 1, gap + 1 + n_steps))
    else:
        # test_iloc_end - test_iloc_start
        steps = fold[3][1] - fold[3][0]

    preds = []
    levels_predict = [level for level in levels if level in last_window_levels]
    if interval is not None:
        kwargs_interval = {
            'steps': steps,
            'levels': levels_predict,
            'last_window': last_window_series,
            'exog': exog_test,
            'n_boot': n_boot,
            'use_in_sample_residuals': use_in_sample_residuals,
            'use_binned_residuals': use_binned_residuals,
            'random_state': random_state,
            'suppress_warnings': suppress_warnings
        }
        if interval_method == 'bootstrapping':
            if interval == 'bootstrapping':
                pred = forecaster.predict_bootstrapping(**kwargs_interval)
            elif isinstance(interval, (float, list, tuple)):
                if isinstance(interval, float):
                    quantiles = [0.5 - interval / 2, 0.5 + interval / 2]
                else:
                    quantiles = [q / 100 for q in interval]
                pred = forecaster.predict_quantiles(quantiles=quantiles, **kwargs_interval)
                if len(quantiles) == 2:
                    pred.columns = ['level', 'lower_bound', 'upper_bound']
                else:
                    pred.columns = ['level'] + [f'p_{p}' for p in interval]
            else:
                pred = forecaster.predict_dist(distribution=interval, **kwargs_interval)
             
            # NOTE: Remove column 'level' as it already exists from predict()
            preds.append(pred.iloc[:, 1:])
        else:
            pred = forecaster.predict_interval(
                method='conformal', interval=interval, **kwargs_interval
            )
            preds.append(pred)

    # NOTE: This is done after probabilistic predictions to avoid repeating 
    # the same checks.
    if interval is None or interval_method != 'conformal':
        pred = forecaster.predict(
                   steps             = steps, 
                   levels            = levels_predict, 
                   last_window       = last_window_series,
                   exog              = exog_test,
                   suppress_warnings = suppress_warnings,
                   check_inputs      = True if interval is None else False
               )
        preds.insert(0, pred)

    if return_predictors:
        # NOTE: ForecasterRnn is not allowed for return_predictors as it 
        # returns two DataFrames, X_predict, exog_predict.
        # NOTE: Remove column 'level' as it already exists from predict()
        pred = forecaster.create_predict_X(
                   steps             = steps,
                   levels            = levels_predict, 
                   last_window       = last_window_series,
                   exog              = exog_test,
                   suppress_warnings = suppress_warnings,
                   check_inputs      = False
               ).iloc[:, 1:]
        preds.append(pred)

    if len(preds) == 1:
        pred = preds[0]
    else:
        pred = pd.concat(preds, axis=1)

    # TODO: Check when long format with multiple levels in multivariate
    if type(forecaster).__name__ != 'ForecasterDirectMultiVariate' and gap > 0:
        pred = pred.iloc[len(levels_predict) * gap:, :]

    return pred, levels_predict


@manage_warnings
def _backtesting_forecaster_multiseries(
    forecaster: object,
    series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame],
    cv: TimeSeriesFold,
    metric: str | Callable | list[str | Callable],
    levels: str | list[str] | None = None,
    add_aggregated_metric: bool = True,
    exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
    interval: float | list[float] | tuple[float] | str | object | None = None,
    interval_method: str = 'conformal',
    n_boot: int = 250,
    use_in_sample_residuals: bool = True,
    use_binned_residuals: bool = True,
    random_state: int = 123,
    return_predictors: bool = False,
    n_jobs: int | str = 'auto',
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting of forecaster model following the folds generated by the TimeSeriesFold
    class and using the metric(s) provided.

    If `forecaster` is already trained and `initial_train_size` is set to `None` in the
    TimeSeriesFold class, no initial train will be done and all data will be used
    to evaluate the model. However, the first `len(forecaster.last_window)` observations
    are needed to create the initial predictors, so no predictions are calculated for
    them.
    
    A copy of the original forecaster is created so that it is not modified during the
    process.
    
    Parameters
    ----------
    forecaster : ForecasterRecursiveMultiSeries, ForecasterDirectMultiVariate, ForecasterRnn
        Forecaster model.
    series : pandas DataFrame, dict
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
    levels : str, list, default None
        Time series to be predicted. If `None` all levels will be predicted.
    add_aggregated_metric : bool, default True
        If `True`, and multiple series (`levels`) are predicted, the aggregated
        metrics (average, weighted average and pooled) are also returned.

        - 'average': the average (arithmetic mean) of all levels.
        - 'weighted_average': the average of the metrics weighted by the number of
        predicted values of each level.
        - 'pooling': the values of all levels are pooled and then the metric is
        calculated.
    exog : pandas Series, pandas DataFrame, dict, default None
        Exogenous variables.
    interval : float, list, tuple, str, object, default None
        Specifies whether probabilistic predictions should be estimated and the 
        method to use. The following options are supported:

        - If `float`, represents the nominal (expected) coverage (between 0 and 1). 
        For instance, `interval=0.95` corresponds to `[2.5, 97.5]` percentiles.
        - If `list` or `tuple`: Sequence of percentiles to compute, each value must 
        be between 0 and 100 inclusive. For example, a 95% confidence interval can 
        be specified as `interval = [2.5, 97.5]` or multiple percentiles (e.g. 10, 
        50 and 90) as `interval = [10, 50, 90]`.
        - If 'bootstrapping' (str): `n_boot` bootstrapping predictions will be generated.
        - If scipy.stats distribution object, the distribution parameters will
        be estimated for each prediction.
        - If None, no probabilistic predictions are estimated.
    interval_method : str, default 'conformal'
        Technique used to estimate prediction intervals. Available options:

        - 'bootstrapping': Bootstrapping is used to generate prediction 
        intervals [1]_.
        - 'conformal': Employs the conformal prediction split method for 
        interval estimation [2]_.
    n_boot : int, default 250
        Number of bootstrapping iterations to perform when estimating prediction
        intervals.
    use_in_sample_residuals : bool, default True
        If `True`, residuals from the training data are used as proxy of
        prediction error to create predictions. 
        If `False`, out of sample residuals (calibration) are used. 
        Out-of-sample residuals must be precomputed using Forecaster's
        `set_out_sample_residuals()` method.
    use_binned_residuals : bool, default True
        If `True`, residuals are selected based on the predicted values 
        (binned selection).
        If `False`, residuals are selected randomly.
    random_state : int, default 123
        Seed for the random number generator to ensure reproducibility.
    return_predictors : bool, default False
        If `True`, the predictors used to make the predictions are also returned.
    n_jobs : int, 'auto', default 'auto'
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default False
        Print number of folds and index of training and validation sets used 
        for backtesting.
    show_progress : bool, default True
        Whether to show a progress bar.
    suppress_warnings: bool, default False
        If `True`, skforecast warnings will be suppressed during the backtesting 
        process. See skforecast.exceptions.warn_skforecast_categories for more
        information.

    Returns
    -------
    metrics_levels : pandas DataFrame
        Value(s) of the metric(s). Index are the levels and columns the metrics.
    backtest_predictions : pandas DataFrame
        Long-format DataFrame containing the predicted values for each series. The 
        DataFrame includes the following columns:
        
        - `level`: Identifier for the time series or level being predicted.
        - fold: Indicates the fold number where the prediction was made.
        - `pred`: Predicted values for the corresponding series and time steps.

        If `interval` is not `None`, additional columns are included depending on the method:
        
        - For `float`: Columns `lower_bound` and `upper_bound`.
        - For `list` or `tuple` of 2 elements: Columns `lower_bound` and `upper_bound`.
        - For `list` or `tuple` with multiple percentiles: One column per percentile 
        (e.g., `p_10`, `p_50`, `p_90`).
        - For `'bootstrapping'`: One column per bootstrapping iteration 
        (e.g., `pred_boot_0`, `pred_boot_1`, ..., `pred_boot_n`).
        - For `scipy.stats` distribution objects: One column for each estimated 
        parameter of the distribution (e.g., `loc`, `scale`).

        If `return_predictors` is `True`, one column per predictor is created.

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

    References
    ----------
    .. [1] Forecasting: Principles and Practice (3rd ed) Rob J Hyndman and George Athanasopoulos.
           https://otexts.com/fpp3/prediction-intervals.html
    
    .. [2] MAPIE - Model Agnostic Prediction Interval Estimator.
           https://mapie.readthedocs.io/en/stable/theoretical_description_regression.html#the-split-method

    """

    need_out_sample_residuals = (
        interval is not None and not use_in_sample_residuals
    )

    if cv.initial_train_size is not None:
        forecaster = deepcopy_forecaster(
            forecaster, include_out_sample_residuals=need_out_sample_residuals
        )
    else:
        forecaster = deepcopy(forecaster)
    cv = deepcopy(cv)

    cv.set_params({
        'window_size': forecaster.window_size,
        'differentiation': forecaster.differentiation_max,
        'return_all_indexes': False,
        'verbose': verbose
    })

    refit = cv.refit
    overlapping_folds = cv.overlapping_folds

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

    levels = _initialize_levels_model_selection_multiseries(
                 forecaster = forecaster,
                 series     = series,
                 levels     = levels
             )

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

    store_in_sample_residuals = True if use_in_sample_residuals else False
    if interval is None:
        forecaster._probabilistic_mode = False
    elif use_binned_residuals:
        forecaster._probabilistic_mode = 'binned'
    else:
        forecaster._probabilistic_mode = 'no_binned'

    folds = cv.split(X=series, as_pandas=False)
    span_index = cv._extract_index(X=series)
    initial_train_size = cv.initial_train_size
    gap = cv.gap

    # Save out-of-sample residuals before any fit() call. Since fit() resets
    # them to None, they must be preserved and restored after each fit so that
    # probabilistic predictions with use_in_sample_residuals=False keep working.
    out_sample_residuals_ = None
    out_sample_residuals_by_bin_ = None
    if need_out_sample_residuals:
        if use_binned_residuals:
            out_sample_residuals_by_bin_ = forecaster.out_sample_residuals_by_bin_
        else:
            out_sample_residuals_ = forecaster.out_sample_residuals_

    if initial_train_size is not None:
        # NOTE: This allows for parallelization when `refit` is `False`. The initial 
        # Forecaster fit occurs outside of the auxiliary function.
        data_fold = _extract_data_folds_multiseries(
                        series             = series,
                        folds              = [folds[0]],
                        span_index         = span_index,
                        window_size        = forecaster.window_size,
                        exog               = exog,
                        dropna_last_window = forecaster.dropna_from_series,
                        externally_fitted  = False
                    )
        series_train, _, last_window_levels, exog_train, _, _ = next(data_fold)
        forecaster.fit(
            series                    = series_train,
            exog                      = exog_train,
            store_last_window         = last_window_levels,
            store_in_sample_residuals = store_in_sample_residuals,
            suppress_warnings         = suppress_warnings
        )
        if out_sample_residuals_ is not None:
            forecaster.out_sample_residuals_ = out_sample_residuals_
        if out_sample_residuals_by_bin_ is not None:
            forecaster.out_sample_residuals_by_bin_ = out_sample_residuals_by_bin_
        folds[0][5] = False
        
    if refit:
        n_of_fits = int(len(folds) / refit)
        if type(forecaster).__name__ != 'ForecasterDirectMultiVariate' and n_of_fits > 50:
            warnings.warn(
                f"The forecaster will be fit {n_of_fits} times. This can take substantial "
                f"amounts of time. If not feasible, try with `refit = False`.\n",
                LongTrainingWarning,
            )
        elif type(forecaster).__name__ == 'ForecasterDirectMultiVariate' and n_of_fits * forecaster.max_step > 50:
            warnings.warn(
                f"The forecaster will be fit {n_of_fits * forecaster.max_step} times "
                f"({n_of_fits} folds * {forecaster.max_step} estimators). This can take "
                f"substantial amounts of time. If not feasible, try with `refit = False`.\n",
                LongTrainingWarning
            )

    if show_progress:
        folds = tqdm(folds)
        
    externally_fitted = True if initial_train_size is None else False
    data_folds = _extract_data_folds_multiseries(
                     series             = series,
                     folds              = folds,
                     span_index         = span_index,
                     window_size        = forecaster.window_size,
                     exog               = exog,
                     dropna_last_window = forecaster.dropna_from_series,
                     externally_fitted  = externally_fitted
                 )

    # Strip series_train and exog_train for refit=False folds to minimize
    # IPC serialization cost when using joblib.Parallel.
    data_folds_list = [
        (None, lw, levels_lw, None, e_test, fold) if fold[5] is False
        else (s_train, lw, levels_lw, e_train, e_test, fold)
        for s_train, lw, levels_lw, e_train, e_test, fold in data_folds
    ]

    kwargs_fit_predict_forecaster = {
        "forecaster": forecaster,
        "store_in_sample_residuals": store_in_sample_residuals,
        "levels": levels,
        "gap": gap,
        "interval": interval,
        "interval_method": interval_method,
        "n_boot": n_boot,
        "use_in_sample_residuals": use_in_sample_residuals,
        "use_binned_residuals": use_binned_residuals,
        "out_sample_residuals_": out_sample_residuals_,
        "out_sample_residuals_by_bin_": out_sample_residuals_by_bin_,
        "random_state": random_state,
        "return_predictors": return_predictors,
        "suppress_warnings": suppress_warnings
    }
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict_forecaster_multiseries)(
            data_fold=data_fold, **kwargs_fit_predict_forecaster
        )
        for data_fold in data_folds_list
    )

    backtest_predictions = [result[0] for result in results]
    fold_labels = [
        np.repeat(fold[0], backtest_predictions[i].shape[0]) for i, fold in enumerate(folds)
    ]
    backtest_predictions = pd.concat(backtest_predictions, axis=0)
    backtest_predictions.insert(0, 'fold', np.concatenate(fold_labels))
    backtest_levels = set(chain(*[result[1] for result in results]))
    
    backtest_predictions = (
        backtest_predictions
        .rename_axis('idx', axis=0)
        .set_index('level', append=True)
    )

    backtest_predictions_grouped = backtest_predictions.groupby('level', sort=False)
    for level, indices in backtest_predictions_grouped.groups.items():
        if level in backtest_levels:
            valid_index = series[level].dropna().index
            valid_index = pd.MultiIndex.from_product([valid_index, [level]], names=['idx', 'level'])
            no_valid_index = indices.difference(valid_index, sort=False)
            backtest_predictions.loc[no_valid_index, 'pred'] = np.nan

    backtest_predictions_for_metrics = backtest_predictions
    if overlapping_folds:
        backtest_predictions_for_metrics = (
            backtest_predictions_for_metrics
            .loc[~backtest_predictions_for_metrics.index.duplicated(keep='last')]
        )

    metrics_levels = _calculate_metrics_backtesting_multiseries(
        series                = series,
        predictions           = backtest_predictions_for_metrics[['pred']],
        folds                 = folds,
        span_index            = span_index,
        window_size           = forecaster.window_size,
        metrics               = metrics,
        levels                = levels,
        add_aggregated_metric = add_aggregated_metric
    )

    backtest_predictions = (
        backtest_predictions
        .reset_index('level')
        .rename_axis(None, axis=0)
    )

    return metrics_levels, backtest_predictions


@manage_warnings
def backtesting_forecaster_multiseries(
    forecaster: object,
    series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame],
    cv: TimeSeriesFold,
    metric: str | Callable | list[str | Callable],
    levels: str | list[str] | None = None,
    add_aggregated_metric: bool = True,
    exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
    interval: float | list[float] | tuple[float] | str | object | None = None,
    interval_method: str = 'conformal',
    n_boot: int = 250,
    use_in_sample_residuals: bool = True,
    use_binned_residuals: bool = True,
    random_state: int = 123,
    return_predictors: bool = False,
    n_jobs: int | str = 'auto',
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting of forecaster model following the folds generated by the TimeSeriesFold
    class and using the metric(s) provided.

    If `forecaster` is already trained and `initial_train_size` is set to `None` in the
    TimeSeriesFold class, no initial train will be done and all data will be used
    to evaluate the model. However, the first `len(forecaster.last_window)` observations
    are needed to create the initial predictors, so no predictions are calculated for
    them.
    
    A copy of the original forecaster is created so that it is not modified during 
    the process.

    Parameters
    ----------
    forecaster : ForecasterRecursiveMultiSeries, ForecasterDirectMultiVariate, ForecasterRnn
        Forecaster model.
    series : pandas DataFrame, dict
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
    levels : str, list, default None
        Time series to be predicted. If `None` all levels will be predicted.
    add_aggregated_metric : bool, default True
        If `True`, and multiple series (`levels`) are predicted, the aggregated
        metrics (average, weighted average and pooled) are also returned.

        - 'average': the average (arithmetic mean) of all levels.
        - 'weighted_average': the average of the metrics weighted by the number of
        predicted values of each level.
        - 'pooling': the values of all levels are pooled and then the metric is
        calculated.
    exog : pandas Series, pandas DataFrame, dict, default None
        Exogenous variables.
    interval : float, list, tuple, str, object, default None
        Specifies whether probabilistic predictions should be estimated and the 
        method to use. The following options are supported:

        - If `float`, represents the nominal (expected) coverage (between 0 and 1). 
        For instance, `interval=0.95` corresponds to `[2.5, 97.5]` percentiles.
        - If `list` or `tuple`: Sequence of percentiles to compute, each value must 
        be between 0 and 100 inclusive. For example, a 95% confidence interval can 
        be specified as `interval = [2.5, 97.5]` or multiple percentiles (e.g. 10, 
        50 and 90) as `interval = [10, 50, 90]`.
        - If 'bootstrapping' (str): `n_boot` bootstrapping predictions will be generated.
        - If scipy.stats distribution object, the distribution parameters will
        be estimated for each prediction.
        - If None, no probabilistic predictions are estimated.
    interval_method : str, default 'conformal'
        Technique used to estimate prediction intervals. Available options:

        - 'bootstrapping': Bootstrapping is used to generate prediction 
        intervals [1]_.
        - 'conformal': Employs the conformal prediction split method for 
        interval estimation [2]_.
    n_boot : int, default 250
        Number of bootstrapping iterations to perform when estimating prediction 
        intervals.
    use_in_sample_residuals : bool, default True
        If `True`, residuals from the training data are used as proxy of
        prediction error to create predictions. 
        If `False`, out of sample residuals (calibration) are used. 
        Out-of-sample residuals must be precomputed using Forecaster's
        `set_out_sample_residuals()` method.
    use_binned_residuals : bool, default True
        If `True`, residuals are selected based on the predicted values 
        (binned selection).
        If `False`, residuals are selected randomly.
    random_state : int, default 123
        Seed for the random number generator to ensure reproducibility.
    return_predictors : bool, default False
        If `True`, the predictors used to make the predictions are also returned.
    n_jobs : int, 'auto', default 'auto'
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default False
        Print number of folds and index of training and validation sets used 
        for backtesting.
    show_progress : bool, default True
        Whether to show a progress bar.
    suppress_warnings: bool, default False
        If `True`, skforecast warnings will be suppressed during the backtesting 
        process. See skforecast.exceptions.warn_skforecast_categories for more
        information.

    Returns
    -------
    metrics_levels : pandas DataFrame
        Value(s) of the metric(s). Index are the levels and columns the metrics.
    backtest_predictions : pandas DataFrame
        Long-format DataFrame containing the predicted values for each series. The 
        DataFrame includes the following columns:
        
        - `level`: Identifier for the time series or level being predicted.
        - fold: Indicates the fold number where the prediction was made.
        - `pred`: Predicted values for the corresponding series and time steps.

        If `interval` is not `None`, additional columns are included depending on the method:
        
        - For `float`: Columns `lower_bound` and `upper_bound`.
        - For `list` or `tuple` of 2 elements: Columns `lower_bound` and `upper_bound`.
        - For `list` or `tuple` with multiple percentiles: One column per percentile 
        (e.g., `p_10`, `p_50`, `p_90`).
        - For `'bootstrapping'`: One column per bootstrapping iteration 
        (e.g., `pred_boot_0`, `pred_boot_1`, ..., `pred_boot_n`).
        - For `scipy.stats` distribution objects: One column for each estimated 
        parameter of the distribution (e.g., `loc`, `scale`).

        If `return_predictors` is `True`, one column per predictor is created.

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

    References
    ----------
    .. [1] Forecasting: Principles and Practice (3rd ed) Rob J Hyndman and George Athanasopoulos.
           https://otexts.com/fpp3/prediction-intervals.html
    
    .. [2] MAPIE - Model Agnostic Prediction Interval Estimator.
           https://mapie.readthedocs.io/en/stable/theoretical_description_regression.html#the-split-method
    
    """

    multi_series_forecasters = [
        'ForecasterRecursiveMultiSeries', 
        'ForecasterDirectMultiVariate',
        'ForecasterRnn'
    ]

    forecaster_name = type(forecaster).__name__

    if forecaster_name not in multi_series_forecasters:
        raise TypeError(
            f"`forecaster` must be of type {multi_series_forecasters}, "
            f"for all other types of forecasters use the functions available in "
            f"the `model_selection` module. Got {forecaster_name}"
        )

    if forecaster_name == 'ForecasterRecursiveMultiSeries':
        series, series_indexes = check_preprocess_series(series)
        if exog is not None:
            series_names_in_ = list(series.keys())
            exog_dict = {serie: None for serie in series_names_in_}
            exog, _ = check_preprocess_exog_multiseries(
                          series_names_in_  = series_names_in_,
                          series_index_type = type(series_indexes[series_names_in_[0]]),
                          exog              = exog,
                          exog_dict         = exog_dict
                      )

    check_backtesting_input(
        forecaster              = forecaster,
        cv                      = cv,
        metric                  = metric,
        add_aggregated_metric   = add_aggregated_metric,
        series                  = series,
        exog                    = exog,
        interval                = interval,
        interval_method         = interval_method,
        n_boot                  = n_boot,
        use_in_sample_residuals = use_in_sample_residuals,
        use_binned_residuals    = use_binned_residuals,
        random_state            = random_state,
        return_predictors       = return_predictors,
        n_jobs                  = n_jobs,
        show_progress           = show_progress,
        suppress_warnings       = suppress_warnings
    )

    metrics_levels, backtest_predictions = _backtesting_forecaster_multiseries(
        forecaster              = forecaster,
        series                  = series,
        cv                      = cv,
        levels                  = levels,
        metric                  = metric,
        add_aggregated_metric   = add_aggregated_metric,
        exog                    = exog,
        interval                = interval,
        interval_method         = interval_method,
        n_boot                  = n_boot,
        use_in_sample_residuals = use_in_sample_residuals,
        use_binned_residuals    = use_binned_residuals,
        random_state            = random_state,
        return_predictors       = return_predictors,
        n_jobs                  = n_jobs,
        verbose                 = verbose,
        show_progress           = show_progress,
        suppress_warnings       = suppress_warnings
    )

    return metrics_levels, backtest_predictions


def _fit_predict_forecaster_stats(
    fold: list,
    forecaster: object,
    y: pd.Series,
    exog: pd.Series | pd.DataFrame | None,
    steps: int,
    gap: int,
    alpha: float | None,
    interval: list[float] | tuple[float] | None,
    refit: bool | int,
    folds: list,
    freeze_params: bool,
    suppress_warnings: bool
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """
    Fit the forecaster and predict `steps` ahead. This is a module-level
    auxiliary function used to parallelize `_backtesting_stats`.

    Defined at module level (instead of as a nested closure) so that
    `joblib.Parallel` can serialize it efficiently with `pickle` rather
    than `cloudpickle`, avoiding unnecessary closure overhead.

    Parameters
    ----------
    fold : list
        Fold metadata as produced by `TimeSeriesFold.split`.
    forecaster : object
        Forecaster model (ForecasterStats).
    y : pandas Series
        Full training time series.
    exog : pandas Series, pandas DataFrame, or None
        Full exogenous variable/s.
    steps : int
        Number of steps to predict.
    gap : int
        Number of observations between training end and test start.
    alpha : float or None
        Confidence level for prediction intervals.
    interval : list, tuple, or None
        Percentiles for prediction intervals.
    refit : bool or int
        Whether to refit in each fold.
    folds : list
        All folds metadata (needed to identify the first fold).
    freeze_params : bool
        Whether estimator params are frozen after first fit.
    suppress_warnings : bool
        Whether to suppress skforecast warnings.

    Returns
    -------
    pred : pandas DataFrame
        Predictions for the fold.
    estimator_names_ : numpy ndarray or None
        Estimator names repeated for each step. `None` if `freeze_params`
        is `True`.

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

        forecaster.fit(y=y_train, exog=exog_train, suppress_warnings=suppress_warnings)

    exog_test = exog.iloc[test_iloc_start:test_iloc_end, ] if exog is not None else None

    # After the first fit, Sarimax must use the last windows stored in the model
    if fold == folds[0]:
        last_window_y = None
        last_window_exog = None

    steps = len(range(test_iloc_start, test_iloc_end))
    if alpha is None and interval is None:
        pred = forecaster.predict(
                   steps             = steps,
                   last_window       = last_window_y,
                   last_window_exog  = last_window_exog,
                   exog              = exog_test,
                   suppress_warnings = suppress_warnings
               )
    else:
        pred = forecaster.predict_interval(
                   steps             = steps,
                   exog              = exog_test,
                   alpha             = alpha,
                   interval          = interval,
                   last_window       = last_window_y,
                   last_window_exog  = last_window_exog,
                   suppress_warnings = suppress_warnings
               )

    if gap > 0:
        pred = pred.iloc[forecaster.n_estimators * gap:, :]

    estimator_names_ = None
    if not freeze_params:
        estimator_names_ = np.repeat(forecaster.estimator_names_, steps - gap)

    return pred, estimator_names_


@manage_warnings
def _backtesting_stats(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold,
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    alpha: float | None = None,
    interval: list[float] | tuple[float] | None = None,
    freeze_params: bool = True,
    n_jobs: int | str = 'auto',
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting of ForecasterStats.
    
    A copy of the original forecaster is created so that it is not modified during 
    the process.
    
    Parameters
    ----------
    forecaster : ForecasterStats
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
    alpha : float, default None
        The confidence intervals for the forecasts are (1 - alpha) %.
        If both, `alpha` and `interval` are provided, `alpha` will be used.
    interval : list, tuple, default None
        Confidence of the prediction interval estimated. The values must be
        symmetric. Sequence of percentiles to compute, which must be between 
        0 and 100 inclusive. For example, interval of 95% should be as 
        `interval = [2.5, 97.5]`. If both, `alpha` and `interval` are 
        provided, `alpha` will be used.
    freeze_params : bool, default True
        Determines whether to freeze the model parameters after the first fit
        for estimators that perform automatic model selection.

        - If `True`, the model parameters found during the first fit (e.g., order 
        and seasonal_order for Arima, or smoothing parameters for Ets) are reused
        in all subsequent refits. This avoids re-running the automatic selection
        procedure in each fold and reduces runtime.
        - If `False`, automatic model selection is performed independently in each
        refit, allowing parameters to adapt across folds. This increases runtime
        and adds a `params` column to the output with the parameters selected per
        fold.
    n_jobs : int, 'auto', default 'auto'
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default False
        Print number of folds and index of training and validation sets used 
        for backtesting.
    show_progress : bool, default True
        Whether to show a progress bar.
    suppress_warnings: bool, default False
        If `True`, skforecast warnings will be suppressed during the backtesting 
        process. See skforecast.exceptions.warn_skforecast_categories for more
        information.

    Returns
    -------
    metric_values : pandas DataFrame
        Value(s) of the metric(s).
    backtest_predictions : pandas DataFrame
        Value of predictions. The DataFrame includes the following columns:

        - fold: Indicates the fold number where the prediction was made.
        - pred: Predicted values for the corresponding series and time steps.

        If `interval` is not `None`, additional columns are included:

        - lower_bound: lower bound of the interval.
        - upper_bound: upper bound of the interval.

        If `freeze_params` is `False`, an additional column is included:

        - estimator_params: parameters used in the estimator for each fold.

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

    forecaster = deepcopy_forecaster(forecaster)
    cv = deepcopy(cv)

    # NOTE: Only skforecast.Sarimax allows refit=False, if other estimators are 
    # present, refit must be True.
    all_sarimax = all(
        est_type == 'skforecast.stats._sarimax.Sarimax' 
        for est_type in forecaster.estimator_types
    )
    if not all_sarimax and not cv.refit:
        warnings.warn(
            "Estimators different from `skforecast.stats.Sarimax` require refitting "
            "since predictions must start from the end of the training set. "
            "`refit` is set to `True`, regardless of the value provided.",
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
    
    if not refit:
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

    if alpha is not None or interval is not None:
        ids_not_support_interval = [
            est_id for est_id, est_type in zip(forecaster.estimator_ids, forecaster.estimator_types)
            if est_type not in forecaster.estimators_support_interval
        ]
        if ids_not_support_interval:
            warnings.warn(
                f"The following estimators do not support prediction intervals "
                f"and will be excluded from backtesting: {ids_not_support_interval}.",
                IgnoredArgumentWarning
            )
            forecaster.remove_estimators(ids_not_support_interval)

    # NOTE: initial_train_size cannot be None because of append method in Sarimax
    exog_train = exog.iloc[:initial_train_size, ] if exog is not None else None
    forecaster.fit(
        y                 = y.iloc[:initial_train_size, ],
        exog              = exog_train,
        suppress_warnings = suppress_warnings
    )
    folds[0][5] = False

    if freeze_params and refit:
        for estimator in forecaster.estimators_:
            if hasattr(estimator, 'best_params_') and estimator.best_params_:
                estimator._set_params(**estimator.best_params_)
    
    if refit:
        n_of_fits = int(len(folds) / refit)
        if n_of_fits > 50:
            warnings.warn(
                f"The forecaster will be fit {n_of_fits} times. This can take substantial "
                f"amounts of time. If not feasible, try with `refit = False`.\n",
                LongTrainingWarning
            )
       
    folds_tqdm = tqdm(folds) if show_progress else folds

    kwargs_fit_predict_forecaster = {
        "forecaster": forecaster,
        "y": y,
        "exog": exog,
        "steps": steps,
        "gap": gap,
        "alpha": alpha,
        "interval": interval,
        "refit": refit,
        "folds": folds,
        "freeze_params": freeze_params,
        "suppress_warnings": suppress_warnings
    }
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict_forecaster_stats)(
            fold=fold, **kwargs_fit_predict_forecaster
        )
        for fold in folds_tqdm
    )

    backtest_predictions = [result[0] for result in results]
    fold_labels = [
        np.repeat(fold[0], backtest_predictions[i].shape[0]) for i, fold in enumerate(folds)
    ]
    backtest_predictions = pd.concat(backtest_predictions, axis=0)
    if isinstance(backtest_predictions, pd.Series):
        backtest_predictions = pd.DataFrame(backtest_predictions)
    backtest_predictions.insert(0, 'fold', np.concatenate(fold_labels))
    if not freeze_params:
        estimator_names_ = [result[1] for result in results]
        backtest_predictions['estimator_params'] = np.concatenate(estimator_names_)

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

    if forecaster.n_estimators == 1:
        y_true = y.loc[backtest_predictions_for_metrics.index]
        y_pred = backtest_predictions_for_metrics['pred']
        metric_values = [[
            m(y_true=y_true, y_pred=y_pred, y_train=y_train)
            for m in metrics
        ]]

        metric_values = pd.DataFrame(
            data    = metric_values,
            columns = [m.__name__ for m in metrics]
        )
    else:
        unique_indices = backtest_predictions_for_metrics.index.unique()
        y_true = y.loc[unique_indices]

        grouped = backtest_predictions_for_metrics.groupby('estimator_id', sort=False)
        metric_values = [
            [
                m(y_true=y_true, y_pred=group['pred'], y_train=y_train)
                for m in metrics
            ]
            for _, group in grouped
        ]

        metric_values = pd.DataFrame(
            data    = metric_values,
            columns = [m.__name__ for m in metrics]
        )
        metric_values.insert(0, 'estimator_id', forecaster.estimator_ids)

    return metric_values, backtest_predictions


def backtesting_stats(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold,
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    alpha: float | None = None,
    interval: list[float] | tuple[float] | None = None,
    freeze_params: bool = True,
    n_jobs: int | str = 'auto',
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting of ForecasterStats.
    
    A copy of the original forecaster is created so that it is not modified during 
    the process.

    Parameters
    ----------
    forecaster : ForecasterStats
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
    alpha : float, default None
        The confidence intervals for the forecasts are (1 - alpha) %.
        If both, `alpha` and `interval` are provided, `alpha` will be used.
    interval : list, tuple, default None
        Confidence of the prediction interval estimated. The values must be
        symmetric. Sequence of percentiles to compute, which must be between 
        0 and 100 inclusive. For example, interval of 95% should be as 
        `interval = [2.5, 97.5]`. If both, `alpha` and `interval` are 
        provided, `alpha` will be used.
    freeze_params : bool, default True
        Determines whether to freeze the model parameters after the first fit
        for estimators that perform automatic model selection.

        - If `True`, the model parameters found during the first fit (e.g., order 
        and seasonal_order for Arima, or smoothing parameters for Ets) are reused
        in all subsequent refits. This avoids re-running the automatic selection
        procedure in each fold and reduces runtime.
        - If `False`, automatic model selection is performed independently in each
        refit, allowing parameters to adapt across folds. This increases runtime
        and adds a `params` column to the output with the parameters selected per
        fold.
    n_jobs : int, 'auto', default 'auto'
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting. 
    verbose : bool, default False
        Print number of folds and index of training and validation sets used 
        for backtesting.
    show_progress : bool, default True
        Whether to show a progress bar.
    suppress_warnings: bool, default False
        If `True`, skforecast warnings will be suppressed during the backtesting 
        process. See skforecast.exceptions.warn_skforecast_categories for more
        information.

    Returns
    -------
    metric_values : pandas DataFrame
        Value(s) of the metric(s).
    backtest_predictions : pandas DataFrame
        Value of predictions. The DataFrame includes the following columns:

        - fold: Indicates the fold number where the prediction was made.
        - pred: Predicted values for the corresponding series and time steps.

        If `interval` is not `None`, additional columns are included:
        
        - lower_bound: lower bound of the interval.
        - upper_bound: upper bound of the interval.

        If `freeze_params` is `False`, an additional column is included:

        - estimator_params: parameters used in the estimator for each fold.

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
    
    if type(forecaster).__name__  != 'ForecasterStats':
        raise TypeError(
            "`forecaster` must be of type `ForecasterStats`. For all other "
            "types of forecasters use the other functions available in the "
            "`model_selection` module."
        )
    
    check_backtesting_input(
        forecaster        = forecaster,
        cv                = cv,
        y                 = y,
        metric            = metric,
        interval          = interval,
        alpha             = alpha,
        freeze_params     = freeze_params,
        n_jobs            = n_jobs,
        show_progress     = show_progress,
        suppress_warnings = suppress_warnings
    )
    
    metric_values, backtest_predictions = _backtesting_stats(
        forecaster        = forecaster,
        y                 = y,
        cv                = cv,
        metric            = metric,
        exog              = exog,
        alpha             = alpha,
        interval          = interval,
        freeze_params     = freeze_params,
        n_jobs            = n_jobs,
        verbose           = verbose,
        show_progress     = show_progress,
        suppress_warnings = suppress_warnings
    )

    return metric_values, backtest_predictions


@manage_warnings
def _backtesting_foundational(
    forecaster: object,
    series: pd.Series | pd.DataFrame | dict,
    cv: TimeSeriesFold,
    metric: str | Callable | list[str | Callable],
    add_aggregated_metric: bool = True,
    levels: str | list[str] | None = None,
    exog: pd.Series | pd.DataFrame | dict | None = None,
    interval: list[float] | tuple[float] | None = None,
    quantiles: list[float] | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting of ForecasterFoundational.

    The original forecaster is used directly (no copy): refit is always
    disabled for foundational models and every fold passes `last_window`
    explicitly, so `self._history` is never modified during the fold loop.
    The only state change is the initial `fit` call that stores the training
    context window.

    Parameters
    ----------
    forecaster : ForecasterFoundational
        Forecaster model.
    series : pandas Series, pandas DataFrame, or dict
        Training time series. A single `pd.Series` runs in single-series
        mode; a wide `pd.DataFrame` or `dict[str, pd.Series]` runs in
        multi-series mode.
    cv : TimeSeriesFold
        TimeSeriesFold object with the information needed to split the data
        into folds.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.

        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and
        `y_train` (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    add_aggregated_metric : bool, default True
        If `True`, and multiple series (multi-series mode) are predicted,
        the aggregated metrics (average, weighted average and pooled) are
        also returned.
    levels : str, list of str, default None
        Series to predict and evaluate. Only used in multi-series mode. If
        `None`, all series seen at fit time are used.
    exog : pandas Series, pandas DataFrame, dict, default None
        Exogenous variable/s included as predictor/s. Must cover the full
        time range of `series` including the forecast horizon of each fold.
    interval : list, tuple, default None
        Confidence of the prediction interval estimated. Sequence of two
        percentiles to compute (e.g. `[10, 90]` for an 80 % interval).
        Cannot be provided together with `quantiles`.
    quantiles : list, default None
        Sequence of quantile levels (between 0 and 1 inclusive) to estimate
        (e.g. `[0.1, 0.5, 0.9]`). Cannot be provided together with
        `interval`.
    verbose : bool, default False
        Print number of folds and index of training and validation sets used
        for backtesting.
    show_progress : bool, default True
        Whether to show a progress bar.
    suppress_warnings : bool, default False
        If `True`, skforecast warnings will be suppressed during the
        backtesting process. See
        skforecast.exceptions.warn_skforecast_categories for more
        information.

    Returns
    -------
    metric_values : pandas DataFrame
        Value(s) of the metric(s).
    backtest_predictions : pandas DataFrame
        Value of predictions and (optionally) prediction intervals or
        quantiles.

    """

    cv = deepcopy(cv)

    is_multiseries, series_names, series_norm = check_preprocess_series_type(series)

    if levels is not None and is_multiseries:
        levels = [levels] if isinstance(levels, str) else list(levels)
    else:
        levels = None

    cv.set_params({
        'window_size': forecaster.window_size,
        'return_all_indexes': False,
        'verbose': verbose,
    })

    refit = cv.refit
    fixed_train_size = cv.fixed_train_size
    overlapping_folds = cv.overlapping_folds

    if refit is not False:
        warnings.warn(
            "`refit` has no effect on `ForecasterFoundational`. Foundational models "
            "are zero-shot and do not learn from training data.",
            IgnoredArgumentWarning
        )
        refit = False
        cv.set_params({'refit': False})

    if fixed_train_size is True:
        fixed_train_size = False
        cv.set_params({'fixed_train_size': False})

    if not isinstance(metric, list):
        metrics = [
            _get_metric(metric=metric) if isinstance(metric, str)
            else add_y_train_argument(metric)
        ]
    else:
        metrics = [
            _get_metric(metric=m) if isinstance(m, str)
            else add_y_train_argument(m)
            for m in metric
        ]

    # Reference series used for cv.split() and index extraction.
    if is_multiseries:
        ref_series = (
            series_norm.iloc[:, 0]
            if isinstance(series_norm, pd.DataFrame)
            else next(iter(series_norm.values()))
        )
    else:
        ref_series = series_norm

    folds = cv.split(X=ref_series, as_pandas=False)
    span_index = cv._extract_index(X=series_norm)
    initial_train_size = cv.initial_train_size

    def _slice_series(s, i, j):
        """Slice a pd.Series, pd.DataFrame or dict of pd.Series by iloc."""
        if isinstance(s, dict):
            return {k: v.iloc[i:j] for k, v in s.items()}
        return s.iloc[i:j]

    def _slice_exog(e, i, j):
        """Slice exog (pd.Series, pd.DataFrame, dict, or None) by iloc."""
        if e is None:
            return None
        if isinstance(e, dict):
            return {k: (v.iloc[i:j] if v is not None else None) for k, v in e.items()}
        return e.iloc[i:j]

    if initial_train_size is not None:
        train_start, train_end = folds[0][1]
        forecaster.fit(
            series=_slice_series(series_norm, train_start, train_end),
            exog=_slice_exog(exog, train_start, train_end),
        )
        folds[0][5] = False

    if refit:
        n_of_fits = int(len(folds) / refit)
        if n_of_fits > 50:
            warnings.warn(
                f"The forecaster will be fit {n_of_fits} times. This can take "
                f"substantial amounts of time. If not feasible, try with "
                f"`refit = False`.\n",
                LongTrainingWarning,
            )

    folds_tqdm = tqdm(folds) if show_progress else folds
    backtest_predictions = []

    for fold in folds_tqdm:
        fold_number = fold[0]
        train_start, train_end = fold[1]
        window_start, window_end = fold[2]
        test_gap_start, test_gap_end = fold[3]
        test_start, test_end = fold[4]
        should_refit = fold[5]

        steps_with_gap = test_gap_end - test_gap_start
        last_window = _slice_series(series_norm, window_start, window_end)
        last_window_exog = _slice_exog(exog, window_start, window_end)
        exog_test = _slice_exog(exog, test_gap_start, test_gap_end)

        if should_refit:
            forecaster.fit(
                series=_slice_series(series_norm, train_start, train_end),
                exog=_slice_exog(exog, train_start, train_end),
            )

        if quantiles is not None:
            pred = forecaster.predict_quantiles(
                steps=steps_with_gap,
                levels=levels,
                quantiles=quantiles,
                exog=exog_test,
                last_window=last_window,
                last_window_exog=last_window_exog,
            )
        elif interval is not None:
            pred = forecaster.predict_interval(
                steps=steps_with_gap,
                levels=levels,
                interval=interval,
                exog=exog_test,
                last_window=last_window,
                last_window_exog=last_window_exog,
            )
        else:
            pred = forecaster.predict(
                steps=steps_with_gap,
                levels=levels,
                exog=exog_test,
                last_window=last_window,
                last_window_exog=last_window_exog,
            )

        if isinstance(pred, pd.Series):
            pred = pred.to_frame(name='pred')

        # Slice to actual test period (remove gap rows if gap > 0).
        test_index = ref_series.iloc[test_start:test_end].index
        if is_multiseries:
            pred = pred[pred.index.isin(test_index)]
        else:
            pred = pred.loc[test_index]

        pred.insert(0, 'fold', fold_number)
        backtest_predictions.append(pred)

    backtest_predictions = pd.concat(backtest_predictions, axis=0)

    # Collect training indexes used across all fits (for y_train in metrics).
    train_indexes = []
    for i, fold in enumerate(folds):
        if i == 0 or fold[-1]:
            train_indexes.append(np.arange(fold[1][0], fold[1][1]))
    train_indexes = np.unique(np.concatenate(train_indexes))

    if is_multiseries:
        # Convert to (idx, level) MultiIndex required by
        # _calculate_metrics_backtesting_multiseries.
        backtest_predictions = (
            backtest_predictions
            .rename_axis('idx', axis=0)
            .set_index('level', append=True)
        )
        backtest_predictions_for_metrics = backtest_predictions
        if overlapping_folds:
            backtest_predictions_for_metrics = (
                backtest_predictions_for_metrics
                .loc[~backtest_predictions_for_metrics.index.duplicated(keep='last')]
            )

        # For quantile-only output, derive 'pred' from the median quantile.
        if 'pred' not in backtest_predictions_for_metrics.columns and quantiles is not None:
            q_col = (
                'q_0.5' if 0.5 in quantiles
                else f"q_{min(quantiles, key=lambda q: abs(q - 0.5))}"
            )
            backtest_predictions_for_metrics = backtest_predictions_for_metrics.copy()
            backtest_predictions_for_metrics['pred'] = (
                backtest_predictions_for_metrics[q_col]
            )

        metrics_levels = _calculate_metrics_backtesting_multiseries(
            series=series_norm,
            predictions=backtest_predictions_for_metrics[['pred']],
            folds=folds,
            span_index=span_index,
            window_size=forecaster.window_size,
            metrics=metrics,
            levels=levels if levels is not None else series_names,
            add_aggregated_metric=add_aggregated_metric,
        )

        backtest_predictions = (
            backtest_predictions
            .reset_index('level')
            .rename_axis(None, axis=0)
        )
    else:
        backtest_predictions_for_metrics = backtest_predictions
        if overlapping_folds:
            backtest_predictions_for_metrics = (
                backtest_predictions_for_metrics
                .loc[~backtest_predictions_for_metrics.index.duplicated(keep='last')]
            )

        if 'pred' in backtest_predictions_for_metrics.columns:
            y_pred = backtest_predictions_for_metrics['pred']
        else:
            q_col = (
                'q_0.5' if 0.5 in quantiles
                else f"q_{min(quantiles, key=lambda q: abs(q - 0.5))}"
            )
            y_pred = backtest_predictions_for_metrics[q_col]

        y_train = ref_series.iloc[train_indexes]
        y_true = ref_series.loc[backtest_predictions_for_metrics.index]

        metric_values = [[
            m(y_true=y_true, y_pred=y_pred, y_train=y_train)
            for m in metrics
        ]]
        metrics_levels = pd.DataFrame(
            data=metric_values,
            columns=[m.__name__ for m in metrics],
        )

    return metrics_levels, backtest_predictions


def backtesting_foundational(
    forecaster: object,
    series: pd.Series | pd.DataFrame | dict,
    cv: TimeSeriesFold,
    metric: str | Callable | list[str | Callable],
    add_aggregated_metric: bool = True,
    levels: str | list[str] | None = None,
    exog: pd.Series | pd.DataFrame | dict | None = None,
    interval: list[float] | tuple[float] | None = None,
    quantiles: list[float] | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting of ForecasterFoundational.

    The original forecaster is modified in-place (fitted on the initial
    training slice) but its loaded model weights are preserved across the
    entire backtesting run. Since foundational models are zero-shot, refit
    is always disabled and per-fold predictions receive `last_window`
    explicitly, so the stored context is not consulted or modified during
    the fold loop.

    Parameters
    ----------
    forecaster : ForecasterFoundational
        Forecaster model.
    series : pandas Series, pandas DataFrame, or dict
        Training time series. A single `pd.Series` runs in single-series
        mode. A wide `pd.DataFrame`, a long-format `pd.DataFrame` with a
        MultiIndex (series IDs in the first level, `DatetimeIndex` in the
        second), or a `dict[str, pd.Series]` runs in multi-series mode.
        Long-format DataFrames are normalised internally to a dict before
        processing.
    cv : TimeSeriesFold
        TimeSeriesFold object with the information needed to split the data
        into folds.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.

        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and
        `y_train` (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    add_aggregated_metric : bool, default True
        If `True`, and multiple series (multi-series mode) are predicted,
        the aggregated metrics (average, weighted average and pooled) are
        also returned.

        - 'average': the average (arithmetic mean) of all levels.
        - 'weighted_average': the average of the metrics weighted by the
        number of predicted values of each level.
        - 'pooling': the values of all levels are pooled and then the metric
        is calculated.
    levels : str, list of str, default None
        Series to predict and evaluate. Only used in multi-series mode. If
        `None`, all series seen at fit time are used.
    exog : pandas Series, pandas DataFrame, dict, default None
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `series` and should be aligned so that
        `series[i]` is regressed on `exog[i]`. Must also cover the
        forecast horizon of each fold.
    interval : list, tuple, default None
        Confidence of the prediction interval estimated. Sequence of two
        percentiles to compute, which must be between 0 and 100 inclusive.
        For example, an 80 % interval should be specified as
        `interval = [10, 90]`. Cannot be provided together with
        `quantiles`.
    quantiles : list, default None
        Sequence of quantile levels (between 0 and 1 inclusive) to estimate.
        For example, `quantiles = [0.1, 0.5, 0.9]`. Cannot be provided
        together with `interval`.
    verbose : bool, default False
        Print number of folds and index of training and validation sets used
        for backtesting.
    show_progress : bool, default True
        Whether to show a progress bar.
    suppress_warnings : bool, default False
        If `True`, skforecast warnings will be suppressed during the
        backtesting process. See
        skforecast.exceptions.warn_skforecast_categories for more
        information.

    Returns
    -------
    metric_values : pandas DataFrame
        Value(s) of the metric(s).
    backtest_predictions : pandas DataFrame
        Value of predictions. The DataFrame includes the following columns:

        - fold: Indicates the fold number where the prediction was made.
        - pred: Predicted values (when `interval` and `quantiles` are
        `None`, or when `interval` is provided).

        If `interval` is provided, additional columns are included:

        - lower_bound: lower bound of the interval.
        - upper_bound: upper bound of the interval.

        If `quantiles` is provided, one column per quantile is included
        (e.g. `q_0.1`, `q_0.5`, `q_0.9`).

        In multi-series mode, a `level` column identifies the series.

        Depending on the relation between `steps` and `fold_stride`, the
        output may include repeated indexes (if `fold_stride < steps`) or
        gaps (if `fold_stride > steps`). See Notes below for more details.

    Notes
    -----
    Note on `fold_stride` vs. `steps`:

    - If `fold_stride == steps`, test sets are placed back-to-back without
    overlap. Each observation appears only once in the output DataFrame, so
    the index is unique.
    - If `fold_stride < steps`, test sets overlap. Multiple forecasts are
    generated for the same observations and, therefore, the output DataFrame
    contains repeated indexes.
    - If `fold_stride > steps`, there are gaps between consecutive test sets.
    Some observations in the series will not have associated predictions, so
    the output DataFrame has non-contiguous indexes.

    """

    if type(forecaster).__name__ != 'ForecasterFoundational':
        raise TypeError(
            "`forecaster` must be of type `ForecasterFoundational`. For all "
            "other types of forecasters use the other functions available in "
            "the `model_selection` module."
        )

    _, _, series_norm = check_preprocess_series_type(series)

    if interval is not None and quantiles is not None:
        raise ValueError(
            "`interval` and `quantiles` cannot be provided simultaneously. "
            "Use `interval` for a prediction interval (e.g. [10, 90]) or "
            "`quantiles` for specific quantile levels (e.g. [0.1, 0.5, 0.9])."
        )

    if quantiles is not None:
        if not isinstance(quantiles, (list, tuple)) or not all(
            isinstance(q, (int, float)) and 0 <= q <= 1 for q in quantiles
        ):
            raise TypeError(
                "`quantiles` must be a list or tuple of floats in the range "
                f"[0, 1]. Got {quantiles}."
            )

    check_backtesting_input(
        forecaster             = forecaster,
        cv                     = cv,
        series                 = series_norm,
        metric                 = metric,
        add_aggregated_metric  = add_aggregated_metric,
        exog                   = exog,
        interval               = interval,
        show_progress          = show_progress,
        suppress_warnings      = suppress_warnings,
    )

    metric_values, backtest_predictions = _backtesting_foundational(
        forecaster            = forecaster,
        series                = series_norm,
        cv                    = cv,
        metric                = metric,
        add_aggregated_metric = add_aggregated_metric,
        levels                = levels,
        exog                  = exog,
        interval              = interval,
        quantiles             = quantiles,
        verbose               = verbose,
        show_progress         = show_progress,
        suppress_warnings     = suppress_warnings,
    )

    return metric_values, backtest_predictions
