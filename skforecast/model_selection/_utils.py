################################################################################
#                     skforecast.model_selection._utils                        #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Union, Tuple, Optional, Callable, Generator
from copy import deepcopy
import warnings
import numpy as np
import pandas as pd
from joblib import cpu_count
from tqdm.auto import tqdm
from sklearn.pipeline import Pipeline
import sklearn.linear_model
from sklearn.exceptions import NotFittedError

from ..exceptions import IgnoredArgumentWarning
from ..metrics import add_y_train_argument, _get_metric
from ..utils import check_interval


def initialize_lags_grid(
    forecaster: object, 
    lags_grid: Optional[Union[list, dict]] = None
) -> Tuple[dict, str]:
    """
    Initialize lags grid and lags label for model selection. 

    Parameters
    ----------
    forecaster : Forecaster
        Forecaster model. ForecasterRecursive, ForecasterDirect, 
        ForecasterRecursiveMultiSeries, ForecasterDirectMultiVariate.
    lags_grid : list, dict, default `None`
        Lists of lags to try, containing int, lists, numpy ndarray, or range 
        objects. If `dict`, the keys are used as labels in the `results` 
        DataFrame, and the values are used as the lists of lags to try.

    Returns
    -------
    lags_grid : dict
        Dictionary with lags configuration for each iteration.
    lags_label : str
        Label for lags representation in the results object.

    """

    if not isinstance(lags_grid, (list, dict, type(None))):
        raise TypeError(
            (f"`lags_grid` argument must be a list, dict or None. "
             f"Got {type(lags_grid)}.")
        )

    lags_label = 'values'
    if isinstance(lags_grid, list):
        lags_grid = {f'{lags}': lags for lags in lags_grid}
    elif lags_grid is None:
        lags = [int(lag) for lag in forecaster.lags]  # Required since numpy 2.0
        lags_grid = {f'{lags}': lags}
    else:
        lags_label = 'keys'

    return lags_grid, lags_label


def check_backtesting_input(
    forecaster: object,
    cv: object,
    metric: Union[str, Callable, list],
    add_aggregated_metric: bool = True,
    y: Optional[pd.Series] = None,
    series: Optional[Union[pd.DataFrame, dict]] = None,
    exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
    interval: Optional[list] = None,
    alpha: Optional[float] = None,
    n_boot: int = 250,
    random_state: int = 123,
    use_in_sample_residuals: bool = True,
    use_binned_residuals: bool = False,
    n_jobs: Union[int, str] = 'auto',
    show_progress: bool = True,
    suppress_warnings: bool = False,
    suppress_warnings_fit: bool = False
) -> None:
    """
    This is a helper function to check most inputs of backtesting functions in 
    modules `model_selection`, `model_selection_multiseries` and 
    `model_selection_sarimax`.

    Parameters
    ----------
    forecaster : Forecaster
        Forecaster model.
    cv : TimeSeriesFold
        TimeSeriesFold object with the information needed to split the data into folds.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
    add_aggregated_metric : bool, default `True`
        If `True`, the aggregated metrics (average, weighted average and pooling)
        over all levels are also returned (only multiseries).
    y : pandas Series, default `None`
        Training time series for uni-series forecasters.
    series : pandas DataFrame, dict, default `None`
        Training time series for multi-series forecasters.
    exog : pandas Series, pandas DataFrame, dict, default `None`
        Exogenous variables.
    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive.
    alpha : float, default `None`
        The confidence intervals used in ForecasterSarimax are (1 - alpha) %. 
    n_boot : int, default `250`
        Number of bootstrapping iterations used to estimate prediction
        intervals.
    random_state : int, default `123`
        Sets a seed to the random generator, so that boot intervals are always 
        deterministic.
    use_in_sample_residuals : bool, default `True`
        If `True`, residuals from the training data are used as proxy of prediction 
        error to create prediction intervals.  If `False`, out_sample_residuals 
        are used if they are already stored inside the forecaster.
    use_binned_residuals : bool, default `False`
        If `True`, residuals used in each bootstrapping iteration are selected
        conditioning on the predicted values. If `False`, residuals are selected
        randomly without conditioning on the predicted values.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the fuction
        skforecast.utils.select_n_jobs_fit_forecaster.
        **New in version 0.9.0**
    show_progress : bool, default `True`
        Whether to show a progress bar.
    suppress_warnings: bool, default `False`
        If `True`, skforecast warnings will be suppressed during the backtesting 
        process. See skforecast.exceptions.warn_skforecast_categories for more
        information.
    suppress_warnings_fit : bool, default `False`
        If `True`, warnings generated during fitting will be ignored. Only 
        `ForecasterSarimax`.

    Returns
    -------
    None
    
    """

    forecaster_name = type(forecaster).__name__
    cv_name = type(cv).__name__

    if cv_name != "TimeSeriesFold":
        raise TypeError(
            (f"`cv` must be a TimeSeriesFold object. Got {cv_name}.")
        )

    steps = cv.steps
    initial_train_size = cv.initial_train_size
    gap = cv.gap
    allow_incomplete_fold = cv.allow_incomplete_fold
    refit = cv.refit

    forecasters_uni = [
        "ForecasterRecursive",
        "ForecasterDirect",
        "ForecasterSarimax",
        "ForecasterEquivalentDate",
    ]
    forecasters_multi = [
        "ForecasterDirectMultiVariate",
        "ForecasterRnn",
    ]
    forecasters_multi_dict = [
        "ForecasterRecursiveMultiSeries"
    ]

    if forecaster_name in forecasters_uni:
        if not isinstance(y, pd.Series):
            raise TypeError("`y` must be a pandas Series.")
        data_name = 'y'
        data_length = len(y)

    elif forecaster_name in forecasters_multi:
        if not isinstance(series, pd.DataFrame):
            raise TypeError("`series` must be a pandas DataFrame.")
        data_name = 'series'
        data_length = len(series)
    
    elif forecaster_name in forecasters_multi_dict:
        if not isinstance(series, (pd.DataFrame, dict)):
            raise TypeError(
                (f"`series` must be a pandas DataFrame or a dict of DataFrames or Series. "
                 f"Got {type(series)}.")
            )
        
        data_name = 'series'
        if isinstance(series, dict):
            not_valid_series = [
                k 
                for k, v in series.items()
                if not isinstance(v, (pd.Series, pd.DataFrame))
            ]
            if not_valid_series:
                raise TypeError(
                    (f"If `series` is a dictionary, all series must be a named "
                     f"pandas Series or a pandas DataFrame with a single column. "
                     f"Review series: {not_valid_series}")
                )
            not_valid_index = [
                k 
                for k, v in series.items()
                if not isinstance(v.index, pd.DatetimeIndex)
            ]
            if not_valid_index:
                raise ValueError(
                    (f"If `series` is a dictionary, all series must have a Pandas "
                     f"DatetimeIndex as index with the same frequency. "
                     f"Review series: {not_valid_index}")
                )

            indexes_freq = [f'{v.index.freq}' for v in series.values()]
            indexes_freq = sorted(set(indexes_freq))
            if not len(indexes_freq) == 1:
                raise ValueError(
                    (f"If `series` is a dictionary, all series must have a Pandas "
                     f"DatetimeIndex as index with the same frequency. "
                     f"Found frequencies: {indexes_freq}")
                )
            data_length = max([len(series[serie]) for serie in series])
        else:
            data_length = len(series)

    if exog is not None:
        if forecaster_name in forecasters_multi_dict:
            if not isinstance(exog, (pd.Series, pd.DataFrame, dict)):
                raise TypeError(
                    (f"`exog` must be a pandas Series, DataFrame, dictionary of pandas "
                     f"Series/DataFrames or None. Got {type(exog)}.")
                )
            if isinstance(exog, dict):
                not_valid_exog = [
                    k 
                    for k, v in exog.items()
                    if not isinstance(v, (pd.Series, pd.DataFrame, type(None)))
                ]
                if not_valid_exog:
                    raise TypeError(
                        (f"If `exog` is a dictionary, All exog must be a named pandas "
                         f"Series, a pandas DataFrame or None. Review exog: {not_valid_exog}")
                    )
        else:
            if not isinstance(exog, (pd.Series, pd.DataFrame)):
                raise TypeError(
                    (f"`exog` must be a pandas Series, DataFrame or None. Got {type(exog)}.")
                )

    if hasattr(forecaster, 'differentiation'):
        if forecaster.differentiation != cv.differentiation:
            raise ValueError(
                (f"The differentiation included in the forecaster "
                 f"({forecaster.differentiation}) differs from the differentiation "
                 f"included in the cv ({cv.differentiation}). Set the same value "
                 f"for both.")
            )

    if not isinstance(metric, (str, Callable, list)):
        raise TypeError(
            (f"`metric` must be a string, a callable function, or a list containing "
             f"multiple strings and/or callables. Got {type(metric)}.")
        )

    if forecaster_name == "ForecasterEquivalentDate" and isinstance(
        forecaster.offset, pd.tseries.offsets.DateOffset
    ):
        if initial_train_size is None:
            raise ValueError(
                (f"`initial_train_size` must be an integer greater than "
                 f"the `window_size` of the forecaster ({forecaster.window_size}) "
                 f"and smaller than the length of `{data_name}` ({data_length}).")
            )
    elif initial_train_size is not None:
        if initial_train_size < forecaster.window_size or initial_train_size >= data_length:
            raise ValueError(
                (f"If used, `initial_train_size` must be an integer greater than "
                 f"the `window_size` of the forecaster ({forecaster.window_size}) "
                 f"and smaller than the length of `{data_name}` ({data_length}).")
            )
        if initial_train_size + gap >= data_length:
            raise ValueError(
                (f"The combination of initial_train_size {initial_train_size} and "
                 f"gap {gap} cannot be greater than the length of `{data_name}` "
                 f"({data_length}).")
            )
    else:
        if forecaster_name in ['ForecasterSarimax', 'ForecasterEquivalentDate']:
            raise ValueError(
                (f"`initial_train_size` must be an integer smaller than the "
                 f"length of `{data_name}` ({data_length}).")
            )
        else:
            if not forecaster.is_fitted:
                raise NotFittedError(
                    ("`forecaster` must be already trained if no `initial_train_size` "
                     "is provided.")
                )
            if refit:
                raise ValueError(
                    "`refit` is only allowed when `initial_train_size` is not `None`."
                )

    if forecaster_name == 'ForecasterSarimax' and cv.skip_folds is not None:
        raise ValueError(
            "`skip_folds` is not allowed for ForecasterSarimax. Set it to `None`."
        )

    if not isinstance(add_aggregated_metric, bool):
        raise TypeError("`add_aggregated_metric` must be a boolean: `True`, `False`.")
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 0:
        raise TypeError(f"`n_boot` must be an integer greater than 0. Got {n_boot}.")
    if not isinstance(random_state, (int, np.integer)) or random_state < 0:
        raise TypeError(f"`random_state` must be an integer greater than 0. Got {random_state}.")
    if not isinstance(use_in_sample_residuals, bool):
        raise TypeError("`use_in_sample_residuals` must be a boolean: `True`, `False`.")
    if not isinstance(use_binned_residuals, bool):
        raise TypeError("`use_binned_residuals` must be a boolean: `True`, `False`.")
    if not isinstance(n_jobs, int) and n_jobs != 'auto':
        raise TypeError(f"`n_jobs` must be an integer or `'auto'`. Got {n_jobs}.")
    if not isinstance(show_progress, bool):
        raise TypeError("`show_progress` must be a boolean: `True`, `False`.")
    if not isinstance(suppress_warnings, bool):
        raise TypeError("`suppress_warnings` must be a boolean: `True`, `False`.")
    if not isinstance(suppress_warnings_fit, bool):
        raise TypeError("`suppress_warnings_fit` must be a boolean: `True`, `False`.")

    if interval is not None or alpha is not None:
        check_interval(interval=interval, alpha=alpha)

    if not allow_incomplete_fold and data_length - (initial_train_size + gap) < steps:
        raise ValueError(
            (f"There is not enough data to evaluate {steps} steps in a single "
             f"fold. Set `allow_incomplete_fold` to `True` to allow incomplete folds.\n"
             f"    Data available for test : {data_length - (initial_train_size + gap)}\n"
             f"    Steps                   : {steps}")
        )


def select_n_jobs_backtesting(
    forecaster: object,
    refit: Union[bool, int]
) -> int:
    """
    Select the optimal number of jobs to use in the backtesting process. This
    selection is based on heuristics and is not guaranteed to be optimal.

    The number of jobs is chosen as follows:

    - If `refit` is an integer, then `n_jobs = 1`. This is because parallelization doesn't 
    work with intermittent refit.
    - If forecaster is 'ForecasterRecursive' and regressor is a linear regressor, 
    then `n_jobs = 1`.
    - If forecaster is 'ForecasterRecursive' and regressor is not a linear 
    regressor and `refit = True`, then `n_jobs = cpu_count() - 1`.
    - If forecaster is 'ForecasterRecursive' and regressor is not a linear 
    regressor and `refit = False`, then `n_jobs = 1`.
    - If forecaster is 'ForecasterDirect' or 'ForecasterDirectMultiVariate'
    and `refit = True`, then `n_jobs = cpu_count() - 1`.
    - If forecaster is 'ForecasterDirect' or 'ForecasterDirectMultiVariate'
    and `refit = False`, then `n_jobs = 1`.
    - If forecaster is 'ForecasterRecursiveMultiSeries', then `n_jobs = cpu_count() - 1`.
    - If forecaster is 'ForecasterSarimax' or 'ForecasterEquivalentDate', 
    then `n_jobs = 1`.
    - If regressor is a `LGBMRegressor`, then `n_jobs = 1`. This is because `lightgbm` 
    is highly optimized for gradient boosting and parallelizes operations at a very 
    fine-grained level, making additional parallelization unnecessary and 
    potentially harmful due to resource contention.

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

    if isinstance(forecaster.regressor, Pipeline):
        regressor_name = type(forecaster.regressor[-1]).__name__
    else:
        regressor_name = type(forecaster.regressor).__name__

    linear_regressors = [
        regressor_name
        for regressor_name in dir(sklearn.linear_model)
        if not regressor_name.startswith('_')
    ]
    
    refit = False if refit == 0 else refit
    if not isinstance(refit, bool) and refit != 1:
        n_jobs = 1
    else:
        if forecaster_name in ['ForecasterRecursive']:
            if regressor_name in linear_regressors or regressor_name == 'LGBMRegressor':
                n_jobs = 1
            else:
                n_jobs = cpu_count() - 1 if refit else 1
        elif forecaster_name in ['ForecasterDirect', 'ForecasterDirectMultiVariate']:
            n_jobs = 1
        elif forecaster_name in ['ForecasterRecursiveMultiSeries']:
            if regressor_name == 'LGBMRegressor':
                n_jobs = 1
            else:
                n_jobs = cpu_count() - 1
        elif forecaster_name in ['ForecasterSarimax', 'ForecasterEquivalentDate']:
            n_jobs = 1
        else:
            n_jobs = 1

    return n_jobs


def _initialize_levels_model_selection_multiseries(
    forecaster: object, 
    series: Union[pd.DataFrame, dict],
    levels: Optional[Union[str, list]] = None
) -> list:
    """
    Initialize levels for model_selection_multiseries functions.

    Parameters
    ----------
    forecaster : ForecasterRecursiveMultiSeries, ForecasterDirectMultiVariate, ForecasterRnn
        Forecaster model.
    series : pandas DataFrame, dict
        Training time series.
    levels : str, list, default `None`
        level (`str`) or levels (`list`) at which the forecaster is optimized. 
        If `None`, all levels are taken into account. The resulting metric will be
        the average of the optimization of all levels.

    Returns
    -------
    levels : list
        List of levels to be used in model_selection_multiseries functions.
    
    """

    multi_series_forecasters_with_levels = [
        'ForecasterRecursiveMultiSeries', 
        'ForecasterRnn'
    ]

    if type(forecaster).__name__ in multi_series_forecasters_with_levels  \
        and not isinstance(levels, (str, list, type(None))):
        raise TypeError(
            (f"`levels` must be a `list` of column names, a `str` of a column "
             f"name or `None` when using a forecaster of type "
             f"{multi_series_forecasters_with_levels}. If the forecaster is of "
             f"type `ForecasterDirectMultiVariate`, this argument is ignored.")
        )

    if type(forecaster).__name__ == 'ForecasterDirectMultiVariate':
        if levels and levels != forecaster.level and levels != [forecaster.level]:
            warnings.warn(
                (f"`levels` argument have no use when the forecaster is of type "
                 f"`ForecasterDirectMultiVariate`. The level of this forecaster "
                 f"is '{forecaster.level}', to predict another level, change "
                 f"the `level` argument when initializing the forecaster. \n"),
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
        elif isinstance(levels, str):
            levels = [levels]

    return levels


def _extract_data_folds_multiseries(
    series: Union[pd.Series, pd.DataFrame, dict],
    folds: list,
    span_index: Union[pd.DatetimeIndex, pd.RangeIndex],
    window_size: int,
    exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
    dropna_last_window: bool = False,
    externally_fitted: bool = False
) -> Generator[
        Tuple[
            Union[pd.Series, pd.DataFrame, dict],
            pd.DataFrame,
            list,
            Optional[Union[pd.Series, pd.DataFrame, dict]],
            Optional[Union[pd.Series, pd.DataFrame, dict]],
            list
        ],
        None,
        None
    ]:
    """
    Select the data from series and exog that corresponds to each fold created using the
    skforecast.model_selection._create_backtesting_folds function.

    Parameters
    ----------
    series : pandas Series, pandas DataFrame, dict
        Time series.
    folds : list
        Folds created using the skforecast.model_selection._create_backtesting_folds
        function.
    span_index : pandas DatetimeIndex, pandas RangeIndex
        Full index from the minimum to the maximum index among all series.
    window_size : int
        Size of the window needed to create the predictors.
    exog : pandas Series, pandas DataFrame, dict, default `None`
        Exogenous variables.
    dropna_last_window : bool, default `False`
        If `True`, drop the columns of the last window that have NaN values.
    externally_fitted : bool, default `False`
        Flag indicating whether the forecaster is already trained. Only used when 
        `initial_train_size` is None and `refit` is False.

    Yield
    -----
    series_train : pandas Series, pandas DataFrame, dict
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
        Fold created using the skforecast.model_selection._create_backtesting_folds

    """

    for fold in folds:
        train_iloc_start       = fold[0][0]
        train_iloc_end         = fold[0][1]
        last_window_iloc_start = fold[1][0]
        last_window_iloc_end   = fold[1][1]
        test_iloc_start        = fold[2][0]
        test_iloc_end          = fold[2][1]

        if isinstance(series, dict) or isinstance(exog, dict):
            # Substract 1 to the iloc indexes to get the loc indexes
            train_loc_start       = span_index[train_iloc_start]
            train_loc_end         = span_index[train_iloc_end - 1]
            last_window_loc_start = span_index[last_window_iloc_start]
            last_window_loc_end   = span_index[last_window_iloc_end - 1]
            test_loc_start        = span_index[test_iloc_start]
            test_loc_end          = span_index[test_iloc_end - 1]

        if isinstance(series, pd.DataFrame):
            series_train = series.iloc[train_iloc_start:train_iloc_end, ]

            series_to_drop = []
            for col in series_train.columns:
                if series_train[col].isna().all():
                    series_to_drop.append(col)
                else:
                    first_valid_index = series_train[col].first_valid_index()
                    last_valid_index = series_train[col].last_valid_index()
                    if (
                        len(series_train[col].loc[first_valid_index:last_valid_index])
                        < window_size
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
            for k in series.keys():
                v = series[k].loc[train_loc_start:train_loc_end]
                if not v.isna().all():
                    first_valid_index = v.first_valid_index()
                    last_valid_index  = v.last_valid_index()
                    if first_valid_index is not None and last_valid_index is not None:
                        v = v.loc[first_valid_index : last_valid_index]
                        if len(v) >= window_size:
                            series_train[k] = v

            series_last_window = {}
            for k, v in series.items():
                v = series[k].loc[last_window_loc_start:last_window_loc_end]
                if ((externally_fitted or k in series_train) and len(v) >= window_size):
                    series_last_window[k] = v

            series_last_window = pd.DataFrame(series_last_window)

        if dropna_last_window:
            series_last_window = series_last_window.dropna(axis=1, how="any")
            # TODO: add the option to drop the series without minimum non NaN values.
            # Similar to how pandas does in the rolling window function.
        
        levels_last_window = list(series_last_window.columns)

        if exog is not None:
            if isinstance(exog, (pd.Series, pd.DataFrame)):
                exog_train = exog.iloc[train_iloc_start:train_iloc_end, ]
                exog_test = exog.iloc[test_iloc_start:test_iloc_end, ]
            else:
                exog_train = {
                    k: v.loc[train_loc_start:train_loc_end] 
                    for k, v in exog.items()
                }
                exog_train = {k: v for k, v in exog_train.items() if len(v) > 0}

                exog_test = {
                    k: v.loc[test_loc_start:test_loc_end]
                    for k, v in exog.items()
                    if externally_fitted or k in exog_train
                }

                exog_test = {k: v for k, v in exog_test.items() if len(v) > 0}
        else:
            exog_train = None
            exog_test = None

        yield series_train, series_last_window, levels_last_window, exog_train, exog_test, fold


def _calculate_metrics_multiseries(
    series: Union[pd.DataFrame, dict],
    predictions: pd.DataFrame,
    folds: Union[list, tqdm],
    span_index: Union[pd.DatetimeIndex, pd.RangeIndex],
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

    Parameters
    ----------
    series : pandas DataFrame, dict
        Series data used for backtesting.
    predictions : pandas DataFrame
        Predictions generated during the backtesting process.
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
    add_aggregated_metric : bool, default `True`
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

    if not isinstance(series, (pd.DataFrame, dict)):
        raise TypeError(
            ("`series` must be a pandas DataFrame or a dictionary of pandas "
             "DataFrames.")
        )
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("`predictions` must be a pandas DataFrame.")
    if not isinstance(folds, (list, tqdm)):
        raise TypeError("`folds` must be a list or a tqdm object.")
    if not isinstance(span_index, (pd.DatetimeIndex, pd.RangeIndex)):
        raise TypeError("`span_index` must be a pandas DatetimeIndex or pandas RangeIndex.")
    if not isinstance(window_size, (int, np.integer)):
        raise TypeError("`window_size` must be an integer.")
    if not isinstance(metrics, list):
        raise TypeError("`metrics` must be a list.")
    if not isinstance(levels, list):
        raise TypeError("`levels` must be a list.")
    if not isinstance(add_aggregated_metric, bool):
        raise TypeError("`add_aggregated_metric` must be a boolean.")
    
    metric_names = [(m if isinstance(m, str) else m.__name__) for m in metrics]

    y_true_pred_levels = []
    y_train_levels = []
    for level in levels:
        y_true_pred_level = None
        y_train = None
        if level in predictions.columns:
            # TODO: avoid merges inside the loop, instead merge outside and then filter
            y_true_pred_level = pd.merge(
                series[level],
                predictions[level],
                left_index  = True,
                right_index = True,
                how         = "inner",
            ).dropna(axis=0, how="any")
            y_true_pred_level.columns = ['y_true', 'y_pred']

            train_indexes = []
            for i, fold in enumerate(folds):
                fit_fold = fold[-1]
                if i == 0 or fit_fold:
                    train_iloc_start = fold[0][0]
                    train_iloc_end = fold[0][1]
                    train_indexes.append(np.arange(train_iloc_start, train_iloc_end))
            train_indexes = np.unique(np.concatenate(train_indexes))
            train_indexes = span_index[train_indexes]
            y_train = series[level].loc[series[level].index.intersection(train_indexes)]

        y_true_pred_levels.append(y_true_pred_level)
        y_train_levels.append(y_train)
            
    metrics_levels = []
    for i, level in enumerate(levels):
        if y_true_pred_levels[i] is not None and not y_true_pred_levels[i].empty:
            metrics_level = [
                m(
                    y_true = y_true_pred_levels[i].iloc[:, 0],
                    y_pred = y_true_pred_levels[i].iloc[:, 1],
                    y_train = y_train_levels[i].iloc[window_size:]  # Exclude observations used to create predictors
                )
                for m in metrics
            ]
            metrics_levels.append(metrics_level)
        else:
            metrics_levels.append([None for _ in metrics])

    metrics_levels = pd.DataFrame(
                         data    = metrics_levels,
                         columns = [m if isinstance(m, str) else m.__name__
                                    for m in metrics]
                     )
    metrics_levels.insert(0, 'levels', levels)

    if len(levels) < 2:
        add_aggregated_metric = False
    
    if add_aggregated_metric:

        # aggragation: average
        average = metrics_levels.drop(columns='levels').mean(skipna=True)
        average = average.to_frame().transpose()
        average['levels'] = 'average'

        # aggregation: weighted_average
        weighted_averages = {}
        n_predictions_levels = (
            predictions
            .notna()
            .sum()
            .to_frame(name='n_predictions')
            .reset_index(names='levels')
        )
        metrics_levels_no_missing = (
            metrics_levels.merge(n_predictions_levels, on='levels', how='inner')
        )
        for col in metric_names:
            weighted_averages[col] = np.average(
                metrics_levels_no_missing[col],
                weights=metrics_levels_no_missing['n_predictions']
            )
        weighted_average = pd.DataFrame(weighted_averages, index=[0])
        weighted_average['levels'] = 'weighted_average'

        # aggregation: pooling
        y_true_pred_levels, y_train_levels = zip(
            *[
                (a, b.iloc[window_size:])  # Exclude observations used to create predictors
                for a, b in zip(y_true_pred_levels, y_train_levels)
                if a is not None
            ]
        )
        y_train_levels = list(y_train_levels)
        y_true_pred_levels = pd.concat(y_true_pred_levels)
        y_train_levels_concat = pd.concat(y_train_levels)

        pooled = []
        for m, m_name in zip(metrics, metric_names):
            if m_name in ['mean_absolute_scaled_error', 'root_mean_squared_scaled_error']:
                pooled.append(
                    m(
                        y_true = y_true_pred_levels.loc[:, 'y_true'],
                        y_pred = y_true_pred_levels.loc[:, 'y_pred'],
                        y_train = y_train_levels
                    )
                )
            else:
                pooled.append(
                    m(
                        y_true = y_true_pred_levels.loc[:, 'y_true'],
                        y_pred = y_true_pred_levels.loc[:, 'y_pred'],
                        y_train = y_train_levels_concat
                    )
                )
        pooled = pd.DataFrame([pooled], columns=metric_names)
        pooled['levels'] = 'pooling'

        metrics_levels = pd.concat(
            [metrics_levels, average, weighted_average, pooled],
            axis=0,
            ignore_index=True
        )

    return metrics_levels


def _predict_and_calculate_metrics_multiseries_one_step_ahead(
    forecaster: object,
    series: Union[pd.DataFrame, dict],
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, dict],
    X_train_encoding: pd.Series,
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, dict],
    X_test_encoding: pd.Series,
    levels: list,
    metrics: list,
    add_aggregated_metric: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """   
    One-step-ahead predictions and metrics for each level and also for all levels
    aggregated using average, weighted average or pooling.
    Input matrices (X_train, y_train, X_test, y_test, X_train_encoding, y_test_encoding)
    are should have been generated using the forecaster._train_test_split_one_step_ahead().

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
    X_train_encoding : pandas Series
            Series identifiers for each row of `X_train`.
    X_test : pandas DataFrame
        Test matrix.
    y_test : pandas Series, dict
        Target values of the test set.
    X_test_encoding : pandas Series
        Series identifiers for each row of `X_test`.
    metrics : list
        List of metrics to calculate.
    levels : list
        Levels to calculate the metrics.
    add_aggregated_metric : bool, default `True`
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
    backtest_predictions : pandas Dataframe
        Value of predictions for each level.
    
    """

    if not isinstance(series, (pd.DataFrame, dict)):
        raise TypeError(
            ("`series` must be a pandas DataFrame or a dictionary of pandas "
             "DataFrames.")
        )
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(f"`X_train` must be a pandas DataFrame. Got: {type(X_train)}")
    if not isinstance(y_train, (pd.Series, dict)):
        raise TypeError(
            (f"`y_train` must be a pandas Series or a dictionary of pandas Series. "
                f"Got: {type(y_train)}")
        )        
    if not isinstance(X_train_encoding, pd.Series):
        raise TypeError(
            (f"`X_train_encoding` must be a pandas Series. Got: {type(X_train_encoding)}")
        )
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"`X_test` must be a pandas DataFrame. Got: {type(X_test)}")
    if not isinstance(y_test, (pd.Series, dict)):
        raise TypeError(
            (f"`y_test` must be a pandas Series or a dictionary of pandas Series. "
             f"Got: {type(y_test)}")
        )
    if not isinstance(X_test_encoding, pd.Series):
        raise TypeError(
            (f"`y_test_encoding` must be a pandas Series. Got: {type(X_test_encoding)}")
        )
    if not isinstance(metrics, list):
        raise TypeError("`metrics` must be a list.")
    if not isinstance(levels, list):
        raise TypeError("`levels` must be a list.")
    if not isinstance(add_aggregated_metric, bool):
        raise TypeError("`add_aggregated_metric` must be a boolean.")
    
    metrics = [
        _get_metric(metric=m)
        if isinstance(m, str)
        else add_y_train_argument(m) 
        for m in metrics
    ]
    metric_names = [(m if isinstance(m, str) else m.__name__) for m in metrics]

    if isinstance(series[levels[0]].index, pd.DatetimeIndex):
        freq = series[levels[0]].index.freq
    else:
        freq = series[levels[0]].index.step

    if type(forecaster).__name__ == 'ForecasterDirectMultiVariate':
        step = 1
        X_train, y_train = forecaster.filter_train_X_y_for_step(
                               step    = step,
                               X_train = X_train,
                               y_train = y_train
                           )
        X_test, y_test = forecaster.filter_train_X_y_for_step(
                             step    = step,  
                             X_train = X_test,
                             y_train = y_test
                         )                 
        forecaster.regressors_[step].fit(X_train, y_train)
        pred = forecaster.regressors_[step].predict(X_test)
    else:
        forecaster.regressor.fit(X_train, y_train)
        pred = forecaster.regressor.predict(X_test)

    predictions_per_level = pd.DataFrame(
        {
            'y_true': y_test,
            'y_pred': pred,
            '_level_skforecast': X_test_encoding,
        },
        index=y_test.index,
    ).groupby('_level_skforecast')
    predictions_per_level = {key: group for key, group in predictions_per_level}

    y_train_per_level = pd.DataFrame(
        {"y_train": y_train, "_level_skforecast": X_train_encoding},
        index=y_train.index,
    ).groupby("_level_skforecast")
    # Interleaved Nan values were excluded fom y_train. They are reestored
    y_train_per_level = {key: group.asfreq(freq) for key, group in y_train_per_level}

    if forecaster.differentiation is not None:
        for level in predictions_per_level:
            differentiator = deepcopy(forecaster.differentiator_[level])
            differentiator.initial_values = [
                series[level].iloc[
                    forecaster.window_size - forecaster.differentiation
                ]
            ]
            predictions_per_level[level]["y_pred"] = differentiator.inverse_transform_next_window(
                predictions_per_level[level]["y_pred"].to_numpy()
            )
            predictions_per_level[level]["y_true"] = differentiator.inverse_transform_next_window(
                predictions_per_level[level]["y_true"].to_numpy()
            )
            y_train_per_level[level]["y_true"] = differentiator.inverse_transform(
                y_train_per_level[level]["y_train"].to_numpy()
            )[forecaster.differentiation:]

    if forecaster.transformer_series is not None:
        for level in predictions_per_level:
            transformer = forecaster.transformer_series_[level]
            predictions_per_level[level]["y_pred"] = transformer.inverse_transform(
                predictions_per_level[level][["y_pred"]]
            )
            predictions_per_level[level]["y_true"] = transformer.inverse_transform(
                predictions_per_level[level][["y_true"]]
            )
            y_train_per_level[level]["y_train"] = transformer.inverse_transform(
                y_train_per_level[level][["y_train"]]
            )

    metrics_levels = []
    for level in levels:
        if level in predictions_per_level:
            metrics_level = [
                m(
                    y_true = predictions_per_level[level].loc[:, 'y_true'],
                    y_pred = predictions_per_level[level].loc[:, 'y_pred'],
                    y_train = y_train_per_level[level].loc[:, 'y_train']
                )
                for m in metrics
            ]
            metrics_levels.append(metrics_level)
        else:
            metrics_levels.append([None for _ in metrics])

    metrics_levels = pd.DataFrame(
                         data    = metrics_levels,
                         columns = [m if isinstance(m, str) else m.__name__
                                    for m in metrics]
                     )
    metrics_levels.insert(0, 'levels', levels)

    if len(levels) < 2:
        add_aggregated_metric = False

    if add_aggregated_metric:

        # aggragation: average
        average = metrics_levels.drop(columns='levels').mean(skipna=True)
        average = average.to_frame().transpose()
        average['levels'] = 'average'

        # aggregation: weighted_average
        weighted_averages = {}
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
        for col in metric_names:
            weighted_averages[col] = np.average(
                metrics_levels_no_missing[col],
                weights=metrics_levels_no_missing['n_predictions']
            )
        weighted_average = pd.DataFrame(weighted_averages, index=[0])
        weighted_average['levels'] = 'weighted_average'

        # aggregation: pooling
        list_y_train_by_level = [
            v['y_train'].to_numpy()
            for k, v in y_train_per_level.items()
            if k in predictions_per_level
        ]
        predictions_pooled = pd.concat(predictions_per_level.values())
        y_train_pooled = pd.concat(
            [v for k, v in y_train_per_level.items() if k in predictions_per_level]
        )
        pooled = []
        for m, m_name in zip(metrics, metric_names):
            if m_name in ['mean_absolute_scaled_error', 'root_mean_squared_scaled_error']:
                pooled.append(
                    m(
                        y_true = predictions_pooled['y_true'],
                        y_pred = predictions_pooled['y_pred'],
                        y_train = list_y_train_by_level
                    )
                )
            else:
                pooled.append(
                    m(
                        y_true = predictions_pooled['y_true'],
                        y_pred = predictions_pooled['y_pred'],
                        y_train = y_train_pooled['y_train']
                    )
                )
        pooled = pd.DataFrame([pooled], columns=metric_names)
        pooled['levels'] = 'pooling'

        metrics_levels = pd.concat(
            [metrics_levels, average, weighted_average, pooled],
            axis=0,
            ignore_index=True
        )

    predictions = (
        pd.concat(predictions_per_level.values())
        .loc[:, ["y_pred", "_level_skforecast"]]
        .pivot(columns="_level_skforecast", values="y_pred")
        .rename_axis(columns=None, index=None)
    )
    predictions = predictions.asfreq(X_test.index.freq)

    return metrics_levels, predictions
