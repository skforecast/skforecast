################################################################################
#                  skforecast.model_selection._validation                      #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################


from __future__ import annotations
from typing import Callable, Any
from copy import deepcopy
from itertools import chain
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from tqdm.auto import tqdm
from ..metrics import add_y_train_argument, _any_metric_needs_y_train, _get_metric
from ..exceptions import (
    LongTrainingWarning,
    IgnoredArgumentWarning
)
from ..model_selection._split import TimeSeriesFold
from ..model_selection._utils import (
    _initialize_levels_model_selection_multiseries,
    check_backtesting_input,
    select_n_jobs_backtesting,
    _extract_data_folds_multiseries,
    _select_fit_folds_multiseries,
    _warn_levels_not_predicted_multiseries,
    _calculate_metrics_backtesting_multiseries,
    _check_nan_predictions_backtesting,
    _get_interval_columns,
    _calculate_interval_metrics,
    INTERVAL_METRIC_NAMES
)
from ..utils import (
    check_preprocess_series,
    check_preprocess_exog_multiseries,
    align_series_and_exog_multiseries,
    manage_warnings,
    deepcopy_forecaster,
    estimator_has_native_nan_support
)
from ..foundation._utils import check_preprocess_series_foundation


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


def _restore_out_sample_residuals(
    forecaster: object,
    out_sample_residuals_: np.ndarray | dict | None,
    out_sample_residuals_by_bin_: dict | None,
    binner_: object | dict | None
) -> None:
    """
    Restore the out-of-sample residuals set by the user, and the binner that
    binned them, after a `fit()` call. `fit()` resets the residuals to `None`
    and refits the binner on the in-sample predictions of the new training
    set. Without restoring the binner, the residuals stored for each bin would
    be selected with a binner whose bins are different from the ones that
    created them.

    Parameters
    ----------
    forecaster : object
        Forecaster model.
    out_sample_residuals_ : numpy ndarray, dict, default None
        Out-of-sample residuals to restore. Ignored if `None`.
    out_sample_residuals_by_bin_ : dict, default None
        Out-of-sample residuals indexed by predicted-value bin to restore.
        Ignored if `None`.
    binner_ : object, dict, default None
        Binner (or dict of binners, one per level) fitted before the first
        `fit()` call. Ignored if `None`.

    Returns
    -------
    None

    """

    if out_sample_residuals_ is not None:
        forecaster.out_sample_residuals_ = out_sample_residuals_
    if out_sample_residuals_by_bin_ is not None:
        forecaster.out_sample_residuals_by_bin_ = out_sample_residuals_by_bin_
    if binner_ is not None:
        # NOTE: A fresh copy is restored every time. Forecasters with a single
        # binner refit it in place in `fit()`, so restoring the saved object
        # itself would let the next fit modify it.
        forecaster.binner = deepcopy(binner_)
        if isinstance(forecaster.binner, dict):
            forecaster.binner_intervals_ = {
                level: binner.intervals_
                for level, binner in forecaster.binner.items()
            }
        else:
            forecaster.binner_intervals_ = forecaster.binner.intervals_


def _fit_predict_forecaster(
    fold: list,
    y_train: pd.Series | None,
    last_window_y: pd.Series | None,
    exog_train: pd.Series | pd.DataFrame | None,
    exog_test: pd.Series | pd.DataFrame | None,
    forecaster: object,
    store_in_sample_residuals: bool,
    gap: int,
    interval: float | list[float] | tuple[float, ...] | str | object | None,
    interval_method: str,
    n_boot: int,
    use_in_sample_residuals: bool,
    use_binned_residuals: bool,
    out_sample_residuals_: np.ndarray | None,
    out_sample_residuals_by_bin_: dict[int, np.ndarray] | None,
    binner_: object | None,
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
    binner_ : object, default None
        Binner fitted by the user before backtesting, restored after each
        `fit()` call (which refits it) so that `out_sample_residuals_by_bin_`
        is used with the binner that created it.
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
        _restore_out_sample_residuals(
            forecaster                   = forecaster,
            out_sample_residuals_        = out_sample_residuals_,
            out_sample_residuals_by_bin_ = out_sample_residuals_by_bin_,
            binner_                      = binner_
        )

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
                        interval = [0.5 - interval / 2, 0.5 + interval / 2]
                    pred = forecaster.predict_quantiles(quantiles=interval, **kwargs_interval)
                    if len(interval) == 2:
                        pred.columns = ['lower_bound', 'upper_bound']
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
    interval: float | list[float] | tuple[float, ...] | str | object | None = None,
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
        TimeSeriesFold object with the information needed to split the data into
        folds. Its `window_size` is taken from the forecaster, and the object
        passed is not modified.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: name of the metric, for example `'mean_absolute_error'`
        or `'root_mean_squared_error'`. All accepted names are listed in
        https://skforecast.org/latest/user_guides/metrics.html.
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
        Value(s) of the metric(s). If `interval` has a lower and an upper bound
        (a `float`, or a `list` or `tuple` of 2 quantiles, optionally with the
        median 0.5), two more columns are added after the metrics:

        - coverage: share of true values inside the interval (bounds included).
        - interval_width: mean of `upper_bound - lower_bound`, in the units of
        the series.

        **New in version 0.27.0**
    backtest_predictions : pandas DataFrame
        Value of predictions. The DataFrame includes the following columns:

        - fold: Indicates the fold number where the prediction was made.
        - pred: Predicted values for the corresponding series and time steps.

        If `interval` is not `None`, additional columns are included depending on the method:
        
        - For `float`: Columns `lower_bound` and `upper_bound`.
        - For `list` or `tuple` of 2 elements: Columns `lower_bound` and `upper_bound`.
        - For `list` or `tuple` with multiple quantiles: One column per quantile 
        (e.g., `q_0.1`, `q_0.5`, `q_0.9`).
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
           https://mapie.readthedocs.io/en/stable/content/conformal-prediction/regression/#2-the-split-method
    
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

    # NOTE: The steps of the `gap` are predicted and then discarded, so a
    # missing predictor (for example, a gap in `exog`) that only affects them
    # must not stop the backtesting: a linear model returns NaN predictions
    # instead of raising. NaN predictions of scored dates raise before the
    # metrics, in `_check_nan_predictions_backtesting`. The forecaster is a
    # copy, so the attribute does not reach the user's forecaster. It is set
    # even if the forecaster does not have it: forecasters saved with previous
    # versions lack it, and the rest of forecasters do not read it.
    forecaster._allow_nan_predictions = True

    folds = cv.split(X=y, as_pandas=False)
    initial_train_size = cv.initial_train_size_as_int
    window_size = cv.window_size
    gap = cv.gap

    # Save out-of-sample residuals, and the binner that binned them, before any
    # fit() call. Since fit() resets the residuals to None and refits the binner,
    # they must be preserved and restored after each fit so that probabilistic
    # predictions with use_in_sample_residuals=False keep working and the
    # residuals of each bin are selected with the binner that created them.
    # Both residual attributes are restored regardless of `use_binned_residuals`
    # because some forecasters access both of them when predicting.
    out_sample_residuals_ = None
    out_sample_residuals_by_bin_ = None
    binner_ = None
    if need_out_sample_residuals:
        out_sample_residuals_ = forecaster.out_sample_residuals_
        out_sample_residuals_by_bin_ = forecaster.out_sample_residuals_by_bin_
        binner_ = deepcopy(forecaster.binner)

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
        _restore_out_sample_residuals(
            forecaster                   = forecaster,
            out_sample_residuals_        = out_sample_residuals_,
            out_sample_residuals_by_bin_ = out_sample_residuals_by_bin_,
            binner_                      = binner_
        )
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
        "binner_": binner_,
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

    backtest_predictions_for_metrics = backtest_predictions
    if overlapping_folds:
        backtest_predictions_for_metrics = (
            backtest_predictions_for_metrics
            .loc[~backtest_predictions_for_metrics.index.duplicated(keep='last')]
        )
    
    y_true = y.loc[backtest_predictions_for_metrics.index]
    y_pred = backtest_predictions_for_metrics['pred']
    _check_nan_predictions_backtesting(
        y_true=y_true, y_pred=y_pred, differentiation=forecaster.differentiation
    )

    if _any_metric_needs_y_train(metrics):
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
    else:
        y_train = None

    metric_values = [
        m(y_true=y_true, y_pred=y_pred, y_train=y_train)
        for m in metrics
    ]
    metric_names = [m.__name__ for m in metrics]

    interval_columns = _get_interval_columns(
                           columns  = backtest_predictions_for_metrics.columns,
                           interval = interval
                       )
    if interval_columns is not None:
        metric_values.extend(
            _calculate_interval_metrics(
                y_true      = y_true,
                lower_bound = backtest_predictions_for_metrics[interval_columns[0]],
                upper_bound = backtest_predictions_for_metrics[interval_columns[1]]
            )
        )
        metric_names += INTERVAL_METRIC_NAMES

    metric_values = pd.DataFrame(
        data    = [metric_values],
        columns = metric_names
    )

    return metric_values, backtest_predictions


def backtesting_forecaster(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold,
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    interval: float | list[float] | tuple[float, ...] | str | object | None = None,
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
        TimeSeriesFold object with the information needed to split the data into
        folds. Its `window_size` is taken from the forecaster, and the object
        passed is not modified.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: name of the metric, for example `'mean_absolute_error'`
        or `'root_mean_squared_error'`. All accepted names are listed in
        https://skforecast.org/latest/user_guides/metrics.html.
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
        For instance, `interval=0.95` corresponds to `[0.025, 0.975]` quantiles.
        - If `list` or `tuple`: Sequence of quantiles to compute, each value must 
        be between 0 and 1 inclusive. For example, a 95% confidence interval can 
        be specified as `interval = [0.025, 0.975]` or multiple quantiles (e.g. 0.1, 
        0.5 and 0.9) as `interval = [0.1, 0.5, 0.9]`.
        - If 'bootstrapping' (str): `n_boot` bootstrapping predictions will be generated.
        - If scipy.stats distribution object, the distribution parameters will
        be estimated for each prediction.
        - If None, no probabilistic predictions are estimated.

        **Changed in version 0.23.0:** `interval` is now expressed as
        quantiles (0-1) instead of percentiles (0-100). Passing percentiles
        is not longer supported and will raise a `ValueError`.
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
        Value(s) of the metric(s). If `interval` has a lower and an upper bound
        (a `float`, or a `list` or `tuple` of 2 quantiles, optionally with the
        median 0.5), two more columns are added after the metrics:

        - coverage: share of true values inside the interval (bounds included).
        - interval_width: mean of `upper_bound - lower_bound`, in the units of
        the series.

        **New in version 0.27.0**
    backtest_predictions : pandas DataFrame
        Value of predictions. The DataFrame includes the following columns:

        - fold: Indicates the fold number where the prediction was made.
        - pred: Predicted values for the corresponding series and time steps.

        If `interval` is not `None`, additional columns are included depending on the method:
        
        - For `float`: Columns `lower_bound` and `upper_bound`.
        - For `list` or `tuple` of 2 elements: Columns `lower_bound` and `upper_bound`.
        - For `list` or `tuple` with multiple quantiles: One column per quantile 
        (e.g., `q_0.1`, `q_0.5`, `q_0.9`).
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
           https://mapie.readthedocs.io/en/stable/content/conformal-prediction/regression/#2-the-split-method
    
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
    interval: float | list[float] | tuple[float, ...] | str | object | None,
    interval_method: str,
    n_boot: int,
    use_in_sample_residuals: bool,
    use_binned_residuals: bool,
    out_sample_residuals_: dict[str, np.ndarray] | None,
    out_sample_residuals_by_bin_: dict[str, dict[int, np.ndarray]] | None,
    binner_: dict[str, Any] | None,
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
        The forecaster is fitted only if `series_train` is not `None`. For
        the folds without fit, `series_train` and `exog_train` are `None`.
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
    binner_ : dict, default None
        Binners fitted by the user before backtesting, one per level, restored
        after each `fit()` call (which refits them) so that
        `out_sample_residuals_by_bin_` is used with the binners that created it.
    random_state : int
        Random seed.
    return_predictors : bool
        Whether to return predictor values.
    suppress_warnings : bool
        Whether to suppress skforecast warnings.

    Returns
    -------
    pred : pandas DataFrame, None
        Predictions for the fold. `None` if no level is predicted in the fold
        (it is only fitted for the following folds without refit).
    
    """

    (
        series_train,
        last_window_series,
        last_window_levels,
        exog_train,
        exog_test,
        fold
    ) = data_fold

    if series_train is not None:
        forecaster.fit(
            series                    = series_train, 
            exog                      = exog_train,
            store_last_window         = last_window_levels,
            store_in_sample_residuals = store_in_sample_residuals,
            suppress_warnings         = suppress_warnings
        )
        _restore_out_sample_residuals(
            forecaster                   = forecaster,
            out_sample_residuals_        = out_sample_residuals_,
            out_sample_residuals_by_bin_ = out_sample_residuals_by_bin_,
            binner_                      = binner_
        )

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
    # NOTE: The levels not predicted in the fold are reported once, after all
    # folds, by `_warn_levels_not_predicted_multiseries`. A warning raised here
    # would be lost when the folds run in parallel worker processes.
    last_window_levels_set = set(last_window_levels)
    levels_predict = [level for level in levels if level in last_window_levels_set]
    if not levels_predict:
        return None

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
                    interval = [0.5 - interval / 2, 0.5 + interval / 2]
                pred = forecaster.predict_quantiles(quantiles=interval, **kwargs_interval)
                if len(interval) == 2:
                    pred.columns = ['level', 'lower_bound', 'upper_bound']
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

    return pred


@manage_warnings
def _backtesting_forecaster_multiseries(
    forecaster: object,
    series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame],
    cv: TimeSeriesFold,
    metric: str | Callable | list[str | Callable],
    levels: str | list[str] | None = None,
    add_aggregated_metric: bool = True,
    exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
    interval: float | list[float] | tuple[float, ...] | str | object | None = None,
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
        TimeSeriesFold object with the information needed to split the data into
        folds. Its `window_size` is taken from the forecaster, and the object
        passed is not modified.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: name of the metric, for example `'mean_absolute_error'`
        or `'root_mean_squared_error'`. All accepted names are listed in
        https://skforecast.org/latest/user_guides/metrics.html.
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
        For instance, `interval=0.95` corresponds to `[0.025, 0.975]` quantiles.
        - If `list` or `tuple`: Sequence of quantiles to compute, each value must 
        be between 0 and 1 inclusive. For example, a 95% confidence interval can 
        be specified as `interval = [0.025, 0.975]` or multiple quantiles (e.g. 0.1, 
        0.5 and 0.9) as `interval = [0.1, 0.5, 0.9]`.
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
        A level that is not predicted in any fold (for example, a series
        shorter than the window size of the forecaster) has NaN metrics and is
        left out of the aggregated metrics. A `MissingValuesWarning` names the
        levels not predicted in a fold where they have observed values. A fold
        in which no level can be predicted is skipped (no predictions, and the
        forecaster is not trained only for it). If no fold can be predicted, a
        `ValueError` is raised.
        If `interval` has a lower and an upper bound (a `float`, or a `list` or
        `tuple` of 2 quantiles, optionally with the median 0.5), two more 
        columns are added after the metrics, for each level and aggregated in 
        the same way as the other metrics:

        - coverage: share of true values inside the interval (bounds included).
        - interval_width: mean of `upper_bound - lower_bound`, in the units of
        the series.

        **New in version 0.27.0**
    backtest_predictions : pandas DataFrame
        Long-format DataFrame containing the predicted values for each series. The 
        DataFrame includes the following columns:
        
        - level: Identifier for the time series or level being predicted.
        - fold: Indicates the fold number where the prediction was made.
        - pred: Predicted values for the corresponding series and time steps.

        If `interval` is not `None`, additional columns are included depending on the method:
        
        - For `float`: Columns `lower_bound` and `upper_bound`.
        - For `list` or `tuple` of 2 elements: Columns `lower_bound` and `upper_bound`.
        - For `list` or `tuple` with multiple quantiles: One column per quantile 
        (e.g., `q_0.1`, `q_0.5`, `q_0.9`).
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
           https://mapie.readthedocs.io/en/stable/content/conformal-prediction/regression/#2-the-split-method

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

    # NOTE: A missing predictor (for example, a gap in `exog`, or in the last
    # window of ForecasterDirectMultiVariate) does not stop the backtesting: a
    # linear model returns NaN predictions for the dates that cannot be
    # predicted, instead of raising, so the rest of series and dates are still
    # evaluated. The NaN predictions of dates with observed values are warned
    # in `_calculate_metrics_backtesting_multiseries`. The forecaster is a copy,
    # so the attribute does not reach the user's forecaster. It is set even if
    # the forecaster does not have it: forecasters saved with previous versions
    # lack it, and ForecasterRnn does not read it.
    forecaster._allow_nan_predictions = True

    folds = cv.split(X=series, as_pandas=False)
    span_index = cv._extract_index(X=series)
    initial_train_size = cv.initial_train_size_as_int
    gap = cv.gap

    # ForecasterDirectMultiVariate and ForecasterRnn use all series to create the
    # predictors, so no series can be dropped from the last window. Folds whose last
    # window has NaNs are still predicted and return NaN predictions (or raise, if the
    # estimator rejects NaNs).
    if type(forecaster).__name__ in {'ForecasterDirectMultiVariate', 'ForecasterRnn'}:
        dropna_last_window = False
    else:
        # Levels with NaNs in the last window are kept when the estimator natively
        # supports NaNs. Otherwise they are dropped, and if no level survives the fold
        # produces no predictions. Differentiation is a hard exclusion: `numpy.diff`
        # expands the NaN footprint and the inverse transform propagates a single NaN
        # across the whole forecast horizon.
        dropna_last_window = (
            forecaster.differentiation_max is not None
            or not estimator_has_native_nan_support(forecaster.estimator)
        )
    
    # Save out-of-sample residuals, and the binner that binned them, before any
    # fit() call. Since fit() resets the residuals to None and refits the binner,
    # they must be preserved and restored after each fit so that probabilistic
    # predictions with use_in_sample_residuals=False keep working and the
    # residuals of each bin are selected with the binner that created them.
    # Both residual attributes are restored regardless of `use_binned_residuals`
    # because some forecasters access both of them when predicting.
    out_sample_residuals_ = None
    out_sample_residuals_by_bin_ = None
    binner_ = None
    if need_out_sample_residuals:
        out_sample_residuals_ = forecaster.out_sample_residuals_
        out_sample_residuals_by_bin_ = forecaster.out_sample_residuals_by_bin_
        binner_ = deepcopy(forecaster.binner)

    externally_fitted = True if initial_train_size is None else False
    data_folds_generator = _extract_data_folds_multiseries(
                               series             = series,
                               folds              = folds,
                               span_index         = span_index,
                               window_size        = forecaster.window_size,
                               exog               = exog,
                               dropna_last_window = dropna_last_window,
                               externally_fitted  = externally_fitted
                           )

    # NOTE: A fold without levels to predict (no level is in its training set
    # or has a valid last window) is skipped. The forecaster is only fitted in a
    # refit fold if its model predicts at least one level, in that fold or in
    # the following folds without refit, so `fit` never receives an empty
    # training set. The levels of the training set of each fold are kept to
    # report why a level is not predicted. Only the training sets of the refit
    # folds are kept, the other folds are predicted with the model of the
    # previous refit.
    data_folds = []
    levels_train_folds = []
    levels_predict_folds = []
    for s_train, lw, levels_lw, e_train, e_test, fold in data_folds_generator:
        levels_train_folds.append(list(s_train))
        levels_lw_set = set(levels_lw)
        levels_predict_folds.append(
            [level for level in levels if level in levels_lw_set]
        )
        if fold[5] is not True:
            s_train, e_train = None, None
        data_folds.append((s_train, lw, levels_lw, e_train, e_test, fold))

    if not any(levels_predict_folds):
        n_folds_no_train = 0 if externally_fitted else sum(
            not set(levels).intersection(levels_train)
            for levels_train in levels_train_folds
        )
        raise ValueError(
            f"No fold of the backtesting has a level to predict. In every fold, "
            f"each level has `window_size` ({forecaster.window_size}) observations "
            f"or fewer in the training set, or its last window cannot be used to "
            f"predict (fewer than `window_size` observations, or missing values "
            f"that are not supported). Folds without levels in the training set: "
            f"{n_folds_no_train} of {len(folds)}. Review `initial_train_size`, "
            f"`fixed_train_size` and `refit` of `cv`, `levels`, or the span of "
            f"the series."
        )
    # NOTE: A level predicted in a fold is evaluated only if it has observed
    # values in its test set (for example, not if the series ends at the end
    # of the training set). If no level is evaluated, no metric can be
    # calculated, so it fails before fitting any forecaster.
    levels_evaluate_any = any(
        series[level].loc[span_index[fold[4][0]]:span_index[fold[4][1] - 1]].notna().any()
        for fold, levels_predict in zip(folds, levels_predict_folds)
        for level in levels_predict
    )
    if not levels_evaluate_any:
        raise ValueError(
            "No level predicted in the backtesting has observed values in the "
            "test set of the folds in which it is predicted, so the metrics "
            "cannot be calculated. Review `initial_train_size`, `fixed_train_size` "
            "and `refit` of `cv`, `levels`, or the span of the series."
        )
    fit_folds = _select_fit_folds_multiseries(
                    folds                = folds,
                    levels_predict_folds = levels_predict_folds
                )

    if refit:
        # NOTE: The initial fit is included, the skipped fits are not.
        n_of_fits = sum(fit_folds)
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

    if initial_train_size is not None:
        # NOTE: This allows for parallelization when `refit` is `False`. The initial 
        # Forecaster fit occurs outside of the auxiliary function.
        if fit_folds[0]:
            series_train, _, last_window_levels, exog_train, _, _ = data_folds[0]
            forecaster.fit(
                series                    = series_train,
                exog                      = exog_train,
                store_last_window         = last_window_levels,
                store_in_sample_residuals = store_in_sample_residuals,
                suppress_warnings         = suppress_warnings
            )
            _restore_out_sample_residuals(
                forecaster                   = forecaster,
                out_sample_residuals_        = out_sample_residuals_,
                out_sample_residuals_by_bin_ = out_sample_residuals_by_bin_,
                binner_                      = binner_
            )
        fit_folds[0] = False
        folds[0][5] = False

    # Strip series_train and exog_train for the folds without fit to minimize
    # IPC serialization cost when using joblib.Parallel. Skipped folds (no fit
    # and no levels to predict) are not sent to the workers.
    data_folds_list = []
    for (s_train, lw, levels_lw, e_train, e_test, fold), fit_fold, levels_predict in zip(
        data_folds, fit_folds, levels_predict_folds
    ):
        if fit_fold:
            data_folds_list.append((s_train, lw, levels_lw, e_train, e_test, fold))
        elif levels_predict:
            data_folds_list.append((None, lw, levels_lw, None, e_test, fold))

    fold_numbers = [data_fold[5][0] for data_fold in data_folds_list]
    if show_progress:
        data_folds_list = tqdm(data_folds_list)

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
        "binner_": binner_,
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

    # NOTE: A fold that is only fitted, because its model predicts the
    # following folds without refit, returns no predictions (`None`).
    fold_labels = [
        np.repeat(fold_number, pred.shape[0])
        for fold_number, pred in zip(fold_numbers, results)
        if pred is not None
    ]
    backtest_predictions = pd.concat(
        [pred for pred in results if pred is not None], axis=0
    )
    backtest_predictions.insert(0, 'fold', np.concatenate(fold_labels))
    backtest_levels = set(chain(*levels_predict_folds))

    _warn_levels_not_predicted_multiseries(
        forecaster           = forecaster,
        series               = series,
        folds                = folds,
        span_index           = span_index,
        levels               = levels,
        levels_predict_folds = levels_predict_folds,
        levels_train_folds   = None if externally_fitted else levels_train_folds
    )

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

    predictions_for_metrics = backtest_predictions_for_metrics[['pred']]
    interval_columns = _get_interval_columns(
                           columns  = backtest_predictions_for_metrics.columns,
                           interval = interval
                       )
    if interval_columns is not None:
        predictions_for_metrics = (
            backtest_predictions_for_metrics[['pred', *interval_columns]]
            .set_axis(['pred', 'lower_bound', 'upper_bound'], axis=1)
        )

    metrics_levels = _calculate_metrics_backtesting_multiseries(
        series                = series,
        predictions           = predictions_for_metrics,
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
    interval: float | list[float] | tuple[float, ...] | str | object | None = None,
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
        TimeSeriesFold object with the information needed to split the data into
        folds. Its `window_size` is taken from the forecaster, and the object
        passed is not modified.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: name of the metric, for example `'mean_absolute_error'`
        or `'root_mean_squared_error'`. All accepted names are listed in
        https://skforecast.org/latest/user_guides/metrics.html.
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
        For instance, `interval=0.95` corresponds to `[0.025, 0.975]` quantiles.
        - If `list` or `tuple`: Sequence of quantiles to compute, each value must 
        be between 0 and 1 inclusive. For example, a 95% confidence interval can 
        be specified as `interval = [0.025, 0.975]` or multiple quantiles (e.g. 0.1, 
        0.5 and 0.9) as `interval = [0.1, 0.5, 0.9]`.
        - If 'bootstrapping' (str): `n_boot` bootstrapping predictions will be generated.
        - If scipy.stats distribution object, the distribution parameters will
        be estimated for each prediction.
        - If None, no probabilistic predictions are estimated.

        **Changed in version 0.23.0:** `interval` is now expressed as
        quantiles (0-1) instead of percentiles (0-100). Passing percentiles
        is not longer supported and will raise a `ValueError`.
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
        A level that is not predicted in any fold (for example, a series
        shorter than the window size of the forecaster) has NaN metrics and is
        left out of the aggregated metrics. A `MissingValuesWarning` names the
        levels not predicted in a fold where they have observed values. A fold
        in which no level can be predicted is skipped (no predictions, and the
        forecaster is not trained only for it). If no fold can be predicted, a
        `ValueError` is raised.
        If `interval` has a lower and an upper bound (a `float`, or a `list` or
        `tuple` of 2 quantiles, optionally with the median 0.5), two more 
        columns are added after the metrics, for each level and aggregated in 
        the same way as the other metrics:

        - coverage: share of true values inside the interval (bounds included).
        - interval_width: mean of `upper_bound - lower_bound`, in the units of
        the series.

        **New in version 0.27.0**
    backtest_predictions : pandas DataFrame
        Long-format DataFrame containing the predicted values for each series. The 
        DataFrame includes the following columns:
        
        - level: Identifier for the time series or level being predicted.
        - fold: Indicates the fold number where the prediction was made.
        - pred: Predicted values for the corresponding series and time steps.

        If `interval` is not `None`, additional columns are included depending on the method:
        
        - For `float`: Columns `lower_bound` and `upper_bound`.
        - For `list` or `tuple` of 2 elements: Columns `lower_bound` and `upper_bound`.
        - For `list` or `tuple` with multiple quantiles: One column per quantile 
        (e.g., `q_0.1`, `q_0.5`, `q_0.9`).
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
           https://mapie.readthedocs.io/en/stable/content/conformal-prediction/regression/#2-the-split-method
    
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
    interval: list[float] | tuple[float, ...] | None,
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
    interval : list, tuple, or None
        Quantiles of the prediction interval (`alpha` is already translated
        into `interval` by `backtesting_stats`).
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
    if interval is None:
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
                   interval          = interval,
                   last_window       = last_window_y,
                   last_window_exog  = last_window_exog,
                   suppress_warnings = suppress_warnings
               )

    # NOTE: With a single estimator and no interval, `pred` is a pandas Series.
    if gap > 0:
        pred = pred.iloc[forecaster.n_estimators * gap:]

    # NOTE: Predictions are in long format with estimators interleaved per step,
    # so names are tiled (not repeated) to stay aligned with `estimator_id`.
    estimator_names_ = None
    if not freeze_params:
        estimator_names_ = np.tile(forecaster.estimator_names_, steps - gap)

    return pred, estimator_names_


@manage_warnings
def _backtesting_stats(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold,
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    interval: list[float] | tuple[float, ...] | None = None,
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
        TimeSeriesFold object with the information needed to split the data into
        folds. Its `window_size` is taken from the forecaster, and the object
        passed is not modified. With estimators other than `Sarimax`,
        `refit` is set to `True`.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: name of the metric, for example `'mean_absolute_error'`
        or `'root_mean_squared_error'`. All accepted names are listed in
        https://skforecast.org/latest/user_guides/metrics.html.
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    exog : pandas Series, pandas DataFrame, default None
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    interval : list, tuple, default None
        Confidence of the prediction interval estimated. The values must be
        symmetric. Sequence of quantiles to compute, which must be between
        0 and 1 inclusive. For example, interval of 95% should be as
        `interval = [0.025, 0.975]`. If `None`, no intervals are estimated.
        `backtesting_stats` translates its `alpha` argument into `interval`
        before calling this function.
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
        Value(s) of the metric(s), one row per estimator if the forecaster
        has several. If `interval` (or `alpha`) is not `None`, two more 
        columns are added after the metrics:

        - coverage: share of true values inside the interval (bounds included).
        - interval_width: mean of `upper_bound - lower_bound`, in the units of
        the series.

        **New in version 0.27.0**
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
    # present, refit must be True. An integer refit other than 1 (intermittent
    # refit) also predicts some folds without refitting. True == 1 in Python.
    all_sarimax = all(
        est_type == 'skforecast.stats._sarimax.Sarimax'
        for est_type in forecaster.estimator_types
    )
    if not all_sarimax and cv.refit != 1:
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
    initial_train_size = cv.initial_train_size_as_int
    steps = cv.steps
    gap = cv.gap

    if interval is not None:
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

    # NOTE: Each refit clones the unfitted estimators in `forecaster.estimators`,
    # so the parameters selected in the first fit are set on them (the forecaster
    # is a copy). Both lists follow the same order.
    if freeze_params and refit:
        for estimator, fitted_estimator in zip(
            forecaster.estimators, forecaster.estimators_
        ):
            best_params = getattr(fitted_estimator, 'best_params_', None)
            if best_params:
                estimator.set_params(**best_params)

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

    if _any_metric_needs_y_train(metrics):
        train_indexes = []
        for i, fold in enumerate(folds):
            fit_fold = fold[-1]
            if i == 0 or fit_fold:
                train_iloc_start = fold[1][0]
                train_iloc_end = fold[1][1]
                train_indexes.append(np.arange(train_iloc_start, train_iloc_end))
        
        train_indexes = np.unique(np.concatenate(train_indexes))
        y_train = y.iloc[train_indexes]
    else:
        y_train = None

    backtest_predictions_for_metrics = backtest_predictions
    if overlapping_folds:
        # With several estimators, each date has one row per estimator, so the
        # duplicates are identified by the pair (date, estimator_id).
        if forecaster.n_estimators == 1:
            is_duplicated = backtest_predictions_for_metrics.index.duplicated(keep='last')
        else:
            is_duplicated = (
                backtest_predictions_for_metrics
                .set_index('estimator_id', append=True)
                .index.duplicated(keep='last')
            )
        backtest_predictions_for_metrics = (
            backtest_predictions_for_metrics.loc[~is_duplicated]
        )

    interval_columns = _get_interval_columns(
                           columns  = backtest_predictions_for_metrics.columns,
                           interval = interval
                       )
    add_interval_metrics = interval_columns is not None
    metric_names = [m.__name__ for m in metrics]
    if add_interval_metrics:
        metric_names += INTERVAL_METRIC_NAMES

    if forecaster.n_estimators == 1:
        y_true = y.loc[backtest_predictions_for_metrics.index]
        y_pred = backtest_predictions_for_metrics['pred']
        _check_nan_predictions_backtesting(y_true=y_true, y_pred=y_pred)
        metric_values = [
            m(y_true=y_true, y_pred=y_pred, y_train=y_train)
            for m in metrics
        ]
        if add_interval_metrics:
            metric_values.extend(
                _calculate_interval_metrics(
                    y_true      = y_true,
                    lower_bound = backtest_predictions_for_metrics[interval_columns[0]],
                    upper_bound = backtest_predictions_for_metrics[interval_columns[1]]
                )
            )

        metric_values = pd.DataFrame(
            data    = [metric_values],
            columns = metric_names
        )
    else:
        unique_indices = backtest_predictions_for_metrics.index.unique()
        y_true = y.loc[unique_indices]

        grouped = backtest_predictions_for_metrics.groupby('estimator_id', sort=False)
        metric_values = []
        for estimator_id, group in grouped:
            _check_nan_predictions_backtesting(
                y_true       = y_true,
                y_pred       = group['pred'],
                estimator_id = estimator_id
            )
            metric_values_estimator = [
                m(y_true=y_true, y_pred=group['pred'], y_train=y_train)
                for m in metrics
            ]
            if add_interval_metrics:
                metric_values_estimator.extend(
                    _calculate_interval_metrics(
                        y_true      = y_true,
                        lower_bound = group[interval_columns[0]],
                        upper_bound = group[interval_columns[1]]
                    )
                )
            metric_values.append(metric_values_estimator)

        metric_values = pd.DataFrame(
            data    = metric_values,
            columns = metric_names
        )
        metric_values.insert(0, 'estimator_id', forecaster.estimator_ids)

    return metric_values, backtest_predictions


def backtesting_stats(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold,
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    interval: list[float] | tuple[float, ...] | None = None,
    alpha: float | None = None,
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
        TimeSeriesFold object with the information needed to split the data into
        folds. Its `window_size` is taken from the forecaster, and the object
        passed is not modified. With estimators other than `Sarimax`,
        `refit` is set to `True`.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: name of the metric, for example `'mean_absolute_error'`
        or `'root_mean_squared_error'`. All accepted names are listed in
        https://skforecast.org/latest/user_guides/metrics.html.
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    exog : pandas Series, pandas DataFrame, default None
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    interval : list, tuple, default None
        Confidence of the prediction interval estimated. The values must be
        symmetric. Sequence of quantiles to compute, which must be between 
        0 and 1 inclusive. For example, interval of 95% should be as 
        `interval = [0.025, 0.975]`. If both `alpha` and `interval` are
        `None`, no intervals are estimated.

        **Changed in version 0.23.0:** `interval` is now expressed as
        quantiles (0-1) instead of percentiles (0-100). Passing percentiles
        is not longer supported and will raise a `ValueError`.

        **Changed in version 0.27.0:** passing both `alpha` and `interval`
        raises a `ValueError` (previously, `alpha` took precedence), and
        `interval` comes before `alpha` in the signature.
    alpha : float, default None
        Alias of `interval` expressed as the significance level: the
        confidence intervals for the forecasts are (1 - alpha) %. For
        example, `alpha=0.05` is equivalent to `interval=[0.025, 0.975]`.
        It cannot be used together with `interval`.
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
        Value(s) of the metric(s), one row per estimator if the forecaster
        has several. If `interval` (or `alpha`) is not `None`, two more 
        columns are added after the metrics:

        - coverage: share of true values inside the interval (bounds included).
        - interval_width: mean of `upper_bound - lower_bound`, in the units of
        the series.

        **New in version 0.27.0**
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
    
    if alpha is not None and interval is not None:
        raise ValueError(
            "`alpha` and `interval` cannot be used together. Use only one "
            "of them: `alpha=0.05` is equivalent to `interval=[0.025, 0.975]`."
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

    # NOTE: `alpha` is an alias of `interval`. It is translated into the
    # equivalent `interval` so that the rest of the process only uses `interval`.
    if alpha is not None:
        interval = [alpha / 2, 1 - alpha / 2]

    metric_values, backtest_predictions = _backtesting_stats(
        forecaster        = forecaster,
        y                 = y,
        cv                = cv,
        metric            = metric,
        exog              = exog,
        interval          = interval,
        freeze_params     = freeze_params,
        n_jobs            = n_jobs,
        verbose           = verbose,
        show_progress     = show_progress,
        suppress_warnings = suppress_warnings
    )

    return metric_values, backtest_predictions


@manage_warnings
def _backtesting_foundation(
    forecaster: object,
    series: dict[str, pd.Series],
    cv: TimeSeriesFold,
    metric: str | Callable | list[str | Callable],
    levels: str | list[str] | None = None,
    add_aggregated_metric: bool = True,
    exog: dict[str, pd.Series | pd.DataFrame | None] | None = None,
    quantiles: list[float] | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting of ForecasterFoundation.

    The original forecaster is used directly (no copy) and is not modified:
    `fit` is never called (`cv` is forced to `refit=True` only so that the
    training span of each fold expands) and every fold passes `context`
    explicitly, so neither the fit state nor `context_` change during the
    fold loop.

    Parameters
    ----------
    forecaster : ForecasterFoundation
        Forecaster model.
    series : dict[str, pd.Series]
        Training time series. A single `pd.Series` runs in single-series
        mode; a wide `pd.DataFrame` or `dict[str, pd.Series]` runs in
        multi-series mode.
    cv : TimeSeriesFold
        TimeSeriesFold object with the information needed to split the data into
        folds. Its `window_size` is taken from the forecaster (the context
        length), and the object passed is not modified. `refit` and
        `fixed_train_size` are ignored: the context grows with each fold.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.

        - If `string`: name of the metric, for example `'mean_absolute_error'`
        or `'root_mean_squared_error'`. All accepted names are listed in
        https://skforecast.org/latest/user_guides/metrics.html.
        - If `Callable`: Function with arguments `y_true`, `y_pred` and
        `y_train` (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    levels : str, list, default None
        Time series to be predicted and evaluated. Must be a subset of the
        names in `series`, otherwise a `ValueError` is raised. If `None`,
        all series are used.
    add_aggregated_metric : bool, default True
        If `True`, and multiple series (`levels`) are predicted, the aggregated
        metrics (average, weighted average and pooled) are also returned.

        - 'average': the average (arithmetic mean) of all levels.
        - 'weighted_average': the average of the metrics weighted by the number of
        predicted values of each level.
        - 'pooling': the values of all levels are pooled and then the metric is
        calculated.
    exog : dict[str, pd.Series | pd.DataFrame | None] | None, default None
        Exogenous variable/s included as predictor/s. Must cover the full
        time range of `series` including the forecast horizon of each fold.
    quantiles : list, default None
        Sequence of quantile levels (between 0 and 1 inclusive) to estimate.
        For example, `quantiles = [0.1, 0.5, 0.9]`.
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
    metrics_levels : pandas DataFrame
        Value(s) of the metric(s).
        In multi-series mode, a level that is not predicted in any fold (for
        example, a series with no observed values in its context window) has
        NaN metrics and is left out of the aggregated metrics. A
        `MissingValuesWarning` names the levels not predicted in a fold where
        they have observed values.
        If `quantiles` has 2 quantiles, optionally with the median 0.5 (for
        example, `[0.1, 0.9]` or `[0.1, 0.5, 0.9]`), two more columns are 
        added after the metrics, computed with those 2 quantiles as the bounds
        of the interval (for each level and aggregated in multi-series mode):

        - coverage: share of true values inside the interval (bounds included).
        - interval_width: mean of `upper_bound - lower_bound`, in the units of
        the series.

        **New in version 0.27.0**
    backtest_predictions : pandas DataFrame
        Value of predictions. The DataFrame includes the following columns:

        - fold: Indicates the fold number where the prediction was made.
        - level: Identifies the series.
        - pred: Predicted values.

        If `quantiles` is provided, one column per quantile is included
        (e.g. `q_0.1`, `q_0.5`, `q_0.9`) instead of pred.

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

    cv = deepcopy(cv)

    if cv.refit is True or cv.fixed_train_size is False:
        warnings.warn(
            f"Foundation models are zero-shot and do not learn from training data. "
            f"Arguments `refit` and `fixed_train_size` have no effect and are "
            f"ignored. The context window grows with each fold up to the maximum "
            f"context length ({forecaster.context_length}).",
            IgnoredArgumentWarning
        )
    cv.refit = True
    cv.fixed_train_size = False

    series_names = list(series.keys())
    is_multiseries = len(series_names) > 1

    levels = _initialize_levels_model_selection_multiseries(
                 forecaster = forecaster,
                 series     = series,
                 levels     = levels
             )

    cv.set_params({
        'window_size': forecaster.window_size,
        'return_all_indexes': False,
        'verbose': verbose,
    })
    overlapping_folds = cv.overlapping_folds

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

    folds = cv.split(X=series, as_pandas=False)
    span_index = cv._extract_index(X=series)
    gap = cv.gap

    # NOTE: Foundation models create no predictors from past values, so their
    # window size is 0: a series with at least one observed value in the train
    # span of a fold is used as context. `forecaster.window_size` is the maximum
    # context length, not the minimum, and it must not be used here.
    data_folds = _extract_data_folds_multiseries(
                     series             = series,
                     folds              = folds,
                     span_index         = span_index,
                     window_size        = 0,
                     exog               = exog,
                     dropna_last_window = False,
                     externally_fitted  = False
                 )

    data_folds_tqdm = tqdm(data_folds, total=len(folds)) if show_progress else data_folds
    
    backtest_predictions = []
    levels_predict_folds = []
    for context, _, _, context_exog, exog_test, fold in data_folds_tqdm:

        fold_number = fold[0]
        test_gap_start, test_gap_end = fold[3]

        steps_with_gap = test_gap_end - test_gap_start
        train_loc_end = span_index[fold[1][1] - 1]
        test_loc_start = span_index[fold[4][0]]
        test_loc_end = span_index[fold[4][1] - 1]

        # NOTE: `_extract_data_folds_multiseries` trims each series to its
        # last valid value, but the context of a foundation model must end
        # at the end of the train span so that the predictions start at the
        # test start: the context is rebuilt from the first valid value up to
        # the train end, trailing NaN included. A series is predicted in this
        # fold only if it has an actual value in the test window and its
        # context window is not entirely NaN (an all-NaN context is rejected
        # on the user-facing path but would reach the adapter here because
        # `check_inputs=False`).
        context = {
            series_name: (
                series[series_name]
                .loc[level_series.index[0]:train_loc_end]
                .iloc[-forecaster.context_length :]
            )
            for series_name, level_series in context.items()
            if series[series_name].loc[test_loc_start:test_loc_end].notna().any()
        }
        context = {
            series_name: level_series
            for series_name, level_series in context.items()
            if level_series.notna().any()
        }
        levels_predict = [level for level in levels if level in context]
        levels_predict_folds.append(levels_predict)
        if not levels_predict:
            # NOTE: As in `_backtesting_forecaster_multiseries`, the fold is
            # skipped. It contributes an empty frame so that the output keeps
            # its columns and the metrics are `None` if every fold is skipped
            # (`_backtesting_forecaster_multiseries` raises a ValueError
            # instead, it has no forecaster to train). The levels not
            # predicted are reported after all folds.
            col_names = ["pred"] if quantiles is None else [f"q_{q}" for q in quantiles]
            pred = pd.DataFrame(
                {
                    "level": pd.Series(dtype=object),
                    **{col: pd.Series(dtype=float) for col in col_names},
                },
                index=span_index[:0],
            )
            pred.insert(1, 'fold', fold_number)
            backtest_predictions.append(pred)
            continue

        if exog is not None:
            context_exog = {
                series_name: (
                    series_exog.iloc[-forecaster.context_length :]
                    if series_exog is not None
                    else None
                )
                for series_name, series_exog in context_exog.items()
            }
        else:
            context_exog = None

        if quantiles is not None:
            pred = forecaster.predict_quantiles(
                       steps        = steps_with_gap,
                       levels       = levels_predict,
                       context      = context,
                       context_exog = context_exog,
                       exog         = exog_test,
                       quantiles    = quantiles,
                       check_inputs = False
                   )
        else:
            pred = forecaster.predict(
                       steps        = steps_with_gap,
                       levels       = levels_predict,
                       context      = context,
                       context_exog = context_exog,
                       exog         = exog_test,
                       check_inputs = False
                   )
        
        pred = pred.iloc[len(levels_predict) * gap:, :]

        pred.insert(1, 'fold', fold_number)
        backtest_predictions.append(pred)

    backtest_predictions = pd.concat(backtest_predictions, axis=0)

    _warn_levels_not_predicted_multiseries(
        forecaster           = forecaster,
        series               = series,
        folds                = folds,
        span_index           = span_index,
        levels               = levels,
        levels_predict_folds = levels_predict_folds
    )

    if is_multiseries:
        backtest_levels = set(backtest_predictions['level'].unique())

        # Convert to (idx, level) MultiIndex required by
        # _calculate_metrics_backtesting_multiseries.
        backtest_predictions = (
            backtest_predictions
            .rename_axis('idx', axis=0)
            .set_index('level', append=True)
        )

        if quantiles is not None:
            mask_cols = [c for c in backtest_predictions.columns if c.startswith('q_')]
        else:
            mask_cols = ['pred']

        backtest_predictions_grouped = backtest_predictions.groupby('level', sort=False)
        for level, indices in backtest_predictions_grouped.groups.items():
            if level in backtest_levels:
                valid_index = series[level].dropna().index
                valid_index = pd.MultiIndex.from_product(
                    [valid_index, [level]], names=['idx', 'level']
                )
                no_valid_index = indices.difference(valid_index, sort=False)
                backtest_predictions.loc[no_valid_index, mask_cols] = np.nan

        backtest_predictions_for_metrics = backtest_predictions
        if overlapping_folds:
            backtest_predictions_for_metrics = (
                backtest_predictions_for_metrics
                .loc[~backtest_predictions_for_metrics.index.duplicated(keep='last')]
            )

        # For quantile-only output, use the median quantile as the metric target.
        if quantiles is not None:
            metric_col = 'q_0.5'
        else:
            metric_col = 'pred'

        predictions_for_metrics = backtest_predictions_for_metrics[[metric_col]]
        interval_columns = _get_interval_columns(
                               columns  = backtest_predictions_for_metrics.columns,
                               interval = quantiles
                           )
        if interval_columns is not None:
            predictions_for_metrics = (
                backtest_predictions_for_metrics[[metric_col, *interval_columns]]
                .set_axis([metric_col, 'lower_bound', 'upper_bound'], axis=1)
            )

        metrics_levels = _calculate_metrics_backtesting_multiseries(
            series                = series,
            predictions           = predictions_for_metrics,
            folds                 = folds,
            span_index            = span_index,
            window_size           = 0,  # Not used here 
            metrics               = metrics,
            levels                = levels,
            add_aggregated_metric = add_aggregated_metric,
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
        
        y_true = series[series_names[0]].loc[backtest_predictions_for_metrics.index]
        # For quantile-only output, derive 'pred' from the median quantile.
        if quantiles is not None:
            y_pred = backtest_predictions_for_metrics['q_0.5']
        else:
            y_pred = backtest_predictions_for_metrics['pred']
        _check_nan_predictions_backtesting(y_true=y_true, y_pred=y_pred)
        
        if _any_metric_needs_y_train(metrics):
            train_indexes = []
            for i, fold in enumerate(folds):
                if i == 0 or fold[-1]:
                    train_indexes.append(np.arange(fold[1][0], fold[1][1]))
            train_indexes = np.unique(np.concatenate(train_indexes))
            y_train = series[series_names[0]].iloc[train_indexes]
        else:
            y_train = None

        metric_values = [
            m(y_true=y_true, y_pred=y_pred, y_train=y_train)
            for m in metrics
        ]
        metric_names = [m.__name__ for m in metrics]

        interval_columns = _get_interval_columns(
                               columns  = backtest_predictions_for_metrics.columns,
                               interval = quantiles
                           )
        if interval_columns is not None:
            metric_values.extend(
                _calculate_interval_metrics(
                    y_true      = y_true,
                    lower_bound = backtest_predictions_for_metrics[interval_columns[0]],
                    upper_bound = backtest_predictions_for_metrics[interval_columns[1]]
                )
            )
            metric_names += INTERVAL_METRIC_NAMES

        metrics_levels = pd.DataFrame(
            data    = [metric_values],
            columns = metric_names,
        )

    return metrics_levels, backtest_predictions


@manage_warnings
def backtesting_foundation(
    forecaster: object,
    series: pd.Series | pd.DataFrame | dict,
    cv: TimeSeriesFold,
    metric: str | Callable | list[str | Callable],
    levels: str | list[str] | None = None,
    add_aggregated_metric: bool = True,
    exog: pd.Series | pd.DataFrame | dict | None = None,
    quantiles: list[float] | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting of ForecasterFoundation.

    The original forecaster is used directly (no copy) and is not modified:
    its loaded model weights are reused across the entire backtesting run
    and `fit` is never called. Since foundation models are zero-shot, `cv`
    is forced to `refit=True` and `fixed_train_size=False` only so that the
    context expands with each fold, and per-fold predictions receive
    `context` explicitly, so neither the fit state nor the stored context
    change during the fold loop.

    Parameters
    ----------
    forecaster : ForecasterFoundation
        Forecaster model.
    series : pandas Series, pandas DataFrame, or dict
        Training time series. A single `pd.Series` runs in single-series
        mode. A wide `pd.DataFrame`, a long-format `pd.DataFrame` with a
        MultiIndex (series IDs in the first level, `DatetimeIndex` in the
        second), or a `dict[str, pd.Series]` runs in multi-series mode.
        Long-format DataFrames are normalised internally to a dict before
        processing.
    cv : TimeSeriesFold
        TimeSeriesFold object with the information needed to split the data into
        folds. Its `window_size` is taken from the forecaster (the context
        length), and the object passed is not modified. `refit` and
        `fixed_train_size` are ignored: the context grows with each fold.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.

        - If `string`: name of the metric, for example `'mean_absolute_error'`
        or `'root_mean_squared_error'`. All accepted names are listed in
        https://skforecast.org/latest/user_guides/metrics.html.
        - If `Callable`: Function with arguments `y_true`, `y_pred` and
        `y_train` (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    levels : str, list, default None
        Time series to be predicted and evaluated. Must be a subset of the
        names in `series`, otherwise a `ValueError` is raised. If `None`,
        all series are used.
    add_aggregated_metric : bool, default True
        If `True`, and multiple series (`levels`) are predicted, the aggregated
        metrics (average, weighted average and pooled) are also returned.

        - 'average': the average (arithmetic mean) of all levels.
        - 'weighted_average': the average of the metrics weighted by the
        number of predicted values of each level.
        - 'pooling': the values of all levels are pooled and then the metric
        is calculated.
    exog : pandas Series, pandas DataFrame, dict, default None
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `series` and should be aligned so that
        `series[i]` is regressed on `exog[i]`. Must also cover the
        forecast horizon of each fold.
    quantiles : list, default None
        Sequence of quantile levels (between 0 and 1 inclusive) to estimate.
        For example, `quantiles = [0.1, 0.5, 0.9]`.
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
    metrics_levels : pandas DataFrame
        Value(s) of the metric(s).
        In multi-series mode, a level that is not predicted in any fold (for
        example, a series with no observed values in its context window) has
        NaN metrics and is left out of the aggregated metrics. A
        `MissingValuesWarning` names the levels not predicted in a fold where
        they have observed values.
        If `quantiles` has 2 quantiles, optionally with the median 0.5 (for
        example, `[0.1, 0.9]` or `[0.1, 0.5, 0.9]`), two more columns are 
        added after the metrics, computed with those 2 quantiles as the bounds
        of the interval (for each level and aggregated in multi-series mode):

        - coverage: share of true values inside the interval (bounds included).
        - interval_width: mean of `upper_bound - lower_bound`, in the units of
        the series.

        **New in version 0.27.0**
    backtest_predictions : pandas DataFrame
        Value of predictions. The DataFrame includes the following columns:

        - fold: Indicates the fold number where the prediction was made.
        - level: Identifies the series.
        - pred: Predicted values.

        If `quantiles` is provided, one column per quantile is included
        (e.g. `q_0.1`, `q_0.5`, `q_0.9`) instead of pred.

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

    if type(forecaster).__name__ != 'ForecasterFoundation':
        raise TypeError(
            "`forecaster` must be of type `ForecasterFoundation`. For all "
            "other types of forecasters use the other functions available in "
            "the `model_selection` module."
        )

    series_dict, series_indexes = check_preprocess_series_foundation(series)
    series_names_in_ = list(series_dict.keys())

    exog_dict = None
    if exog is not None:
        exog_dict, _ = check_preprocess_exog_multiseries(
            series_names_in_  = series_names_in_,
            series_index_type = type(series_indexes[series_names_in_[0]]),
            exog              = exog,
            exog_dict         = {series_name: None for series_name in series_names_in_},
        )

        # NOTE: As no trim is applied to the series, it is only needed to align exog.
        series_dict, exog_dict = align_series_and_exog_multiseries(
                                     series_dict      = series_dict,
                                     exog_dict        = exog_dict,
                                     trim_series_nan  = False,
                                 )

    if quantiles is not None:
        if 0.5 not in quantiles:
            warnings.warn(
                "The median quantile (0.5) is required to compute the main "
                "predictions and metrics. Since it is not included in `quantiles`, "
                "it will be added automatically.",
                IgnoredArgumentWarning
            )
            quantiles = sorted(quantiles + [0.5])

    check_backtesting_input(
        forecaster             = forecaster,
        cv                     = cv,
        series                 = series_dict,
        metric                 = metric,
        add_aggregated_metric  = add_aggregated_metric,
        exog                   = exog_dict,
        interval               = quantiles,
        show_progress          = show_progress,
        suppress_warnings      = suppress_warnings,
    )

    metrics_levels, backtest_predictions = _backtesting_foundation(
        forecaster            = forecaster,
        series                = series_dict,
        cv                    = cv,
        metric                = metric,
        levels                = levels,
        add_aggregated_metric = add_aggregated_metric,
        exog                  = exog_dict,
        quantiles             = quantiles,
        verbose               = verbose,
        show_progress         = show_progress,
        suppress_warnings     = suppress_warnings,
    )

    return metrics_levels, backtest_predictions
