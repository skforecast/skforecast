################################################################################
#                     skforecast.model_selection._search                       #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import os
import logging
from typing import Callable
import warnings
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import ParameterGrid, ParameterSampler
from ..exceptions import warn_skforecast_categories
from ..model_selection._split import TimeSeriesFold, OneStepAheadFold
from ..experimental._validation import (
    backtesting_stats
)
from ..metrics import add_y_train_argument, _get_metric
from ..model_selection._utils import (
    check_one_step_ahead_input,
    initialize_lags_grid,
    _initialize_levels_model_selection_multiseries,
    _calculate_metrics_one_step_ahead,
    _predict_and_calculate_metrics_one_step_ahead_multiseries
)
from ..utils import (
    initialize_lags, 
    date_to_index_position, 
    check_preprocess_series,
    check_preprocess_exog_multiseries,
    set_skforecast_warnings
)


def grid_search_stats(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold,
    param_grid: dict,
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    return_best: bool = True,
    n_jobs: int | str = 'auto',
    verbose: bool = False,
    suppress_warnings_fit: bool = False,
    show_progress: bool = True,
    output_file: str | None = None
) -> pd.DataFrame:
    """
    Exhaustive search over specified parameter values for a ForecasterSarimax object.
    Validation is done using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterSarimax
        Forecaster model.
    y : pandas Series
        Training time series. 
    cv : TimeSeriesFold
        TimeSeriesFold object with the information needed to split the data into folds.
        **New in version 0.14.0**
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values.
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
    return_best : bool, default True
        Refit the `forecaster` using the best found parameters on the whole data.
    n_jobs : int, 'auto', default 'auto'
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default False
        Print number of folds used for cv or backtesting.
    suppress_warnings_fit : bool, default False
        If `True`, warnings generated during fitting will be ignored.
    show_progress : bool, default True
        Whether to show a progress bar.
    output_file : str, default None
        Specifies the filename or full path where the results should be saved. 
        The results will be saved in a tab-separated values (TSV) format. If 
        `None`, the results will not be saved to a file.

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

        - column params: parameters configuration for each iteration.
        - column metric: metric value estimated for each iteration.
        - additional n columns with param = value.
    
    """

    param_grid = list(ParameterGrid(param_grid))

    results = _evaluate_grid_hyperparameters_stats(
        forecaster            = forecaster,
        y                     = y,
        cv                    = cv,
        param_grid            = param_grid,
        metric                = metric,
        exog                  = exog,
        return_best           = return_best,
        n_jobs                = n_jobs,
        verbose               = verbose,
        suppress_warnings_fit = suppress_warnings_fit,
        show_progress         = show_progress,
        output_file           = output_file
    )

    return results


def random_search_stats(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold,
    param_distributions: dict,
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    n_iter: int = 10,
    random_state: int = 123,
    return_best: bool = True,
    n_jobs: int | str = 'auto',
    verbose: bool = False,
    suppress_warnings_fit: bool = False,
    show_progress: bool = True,
    output_file: str | None = None
) -> pd.DataFrame:
    """
    Random search over specified parameter values or distributions for a Forecaster 
    object. Validation is done using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterSarimax
        Forecaster model.
    y : pandas Series
        Training time series. 
    cv : TimeSeriesFold
        TimeSeriesFold object with the information needed to split the data into folds.
        **New in version 0.14.0**
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and 
        distributions or lists of parameters to try.
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
    n_iter : int, default 10
        Number of parameter settings that are sampled. 
        n_iter trades off runtime vs quality of the solution.
    random_state : int, default 123
        Sets a seed to the random sampling for reproducible output.
    return_best : bool, default True
        Refit the `forecaster` using the best found parameters on the whole data.
    n_jobs : int, 'auto', default 'auto'
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default False
        Print number of folds used for cv or backtesting.
    suppress_warnings_fit : bool, default False
        If `True`, warnings generated during fitting will be ignored.
    show_progress : bool, default True
        Whether to show a progress bar.
    output_file : str, default None
        Specifies the filename or full path where the results should be saved. 
        The results will be saved in a tab-separated values (TSV) format. If 
        `None`, the results will not be saved to a file.

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

        - column params: parameters configuration for each iteration.
        - column metric: metric value estimated for each iteration.
        - additional n columns with param = value.
    
    """

    param_grid = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))

    results = _evaluate_grid_hyperparameters_stats(
        forecaster            = forecaster,
        y                     = y,
        cv                    = cv,
        param_grid            = param_grid,
        metric                = metric,
        exog                  = exog,
        return_best           = return_best,
        n_jobs                = n_jobs,
        verbose               = verbose,
        suppress_warnings_fit = suppress_warnings_fit,
        show_progress         = show_progress,
        output_file           = output_file
    )

    return results


def _evaluate_grid_hyperparameters_stats(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold,
    param_grid: dict,
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    return_best: bool = True,
    n_jobs: int | str = 'auto',
    verbose: bool = False,
    suppress_warnings_fit: bool = False,
    show_progress: bool = True,
    output_file: str | None = None
) -> pd.DataFrame:
    """
    Evaluate parameter values for a Forecaster object using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterSarimax
        Forecaster model.
    y : pandas Series
        Training time series. 
    cv : TimeSeriesFold
        TimeSeriesFold object with the information needed to split the data into folds.
        **New in version 0.14.0**
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values.
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
    return_best : bool, default True
        Refit the `forecaster` using the best found parameters on the whole data.
    n_jobs : int, 'auto', default 'auto'
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default False
        Print number of folds used for cv or backtesting.
    suppress_warnings_fit : bool, default False
        If `True`, warnings generated during fitting will be ignored.
    show_progress : bool, default True
        Whether to show a progress bar.
    output_file : str, default None
        Specifies the filename or full path where the results should be saved. 
        The results will be saved in a tab-separated values (TSV) format. If 
        `None`, the results will not be saved to a file.

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

        - column params: parameters configuration for each iteration.
        - column metric: metric value estimated for each iteration.
        - additional n columns with param = value.

    """

    if return_best and exog is not None and (len(exog) != len(y)):
        raise ValueError(
            f"`exog` must have same number of samples as `y`. "
            f"length `exog`: ({len(exog)}), length `y`: ({len(y)})"
        )

    if not isinstance(metric, list):
        metric = [metric] 
    metric_dict = {(m if isinstance(m, str) else m.__name__): [] 
                   for m in metric}
    
    if len(metric_dict) != len(metric):
        raise ValueError(
            "When `metric` is a `list`, each metric name must be unique."
        )

    print(f"Number of models compared: {len(param_grid)}.")

    if show_progress:
        param_grid = tqdm(param_grid, desc='params grid', position=0)
    
    if output_file is not None and os.path.isfile(output_file):
        os.remove(output_file)
    
    params_list = []
    for params in param_grid:

        try:
            forecaster.set_params(params)
            metric_values = backtesting_stats(
                                forecaster            = forecaster,
                                y                     = y,
                                cv                    = cv,
                                metric                = metric,
                                exog                  = exog,
                                alpha                 = None,
                                interval              = None,
                                n_jobs                = n_jobs,
                                verbose               = verbose,
                                suppress_warnings_fit = suppress_warnings_fit,
                                show_progress         = False
                            )[0]
            metric_values = metric_values.iloc[0, :].to_list()
            warnings.filterwarnings(
                'ignore', category=RuntimeWarning, message= "The forecaster will be fit.*"
            )
            
            params_list.append(params)
            for m, m_value in zip(metric, metric_values):
                m_name = m if isinstance(m, str) else m.__name__
                metric_dict[m_name].append(m_value)
        except Exception as e:
            warnings.warn(
                f"Exception raised for parameters {params}.\n"
                f"Parameters skipped. Exception: {e}",
                RuntimeWarning
            )
            continue
        
        if output_file is not None:
            header = ['params', *metric_dict.keys(), *params.keys()]
            row = [params, *metric_values, *params.values()]
            if not os.path.isfile(output_file):
                with open(output_file, 'w', newline='') as f:
                    f.write('\t'.join(header) + '\n')
                    f.write('\t'.join([str(r) for r in row]) + '\n')
            else:
                with open(output_file, 'a', newline='') as f:
                    f.write('\t'.join([str(r) for r in row]) + '\n')

    results = pd.DataFrame({
                  'params': params_list,
                  **metric_dict
              })
    
    results = (
        results
        .sort_values(by=list(metric_dict.keys())[0], ascending=True)
        .reset_index(drop=True)
    )
    results = pd.concat([results, results['params'].apply(pd.Series)], axis=1)
    
    if return_best:
        
        best_params = results.loc[0, 'params']
        best_metric = results.loc[0, list(metric_dict.keys())[0]]
        forecaster.set_params(best_params)
        forecaster.fit(y=y, exog=exog, suppress_warnings=suppress_warnings_fit)
        
        print(
            f"`Forecaster` refitted using the best-found parameters, "
            f"and the whole data set: \n"
            f"  Parameters: {best_params}\n"
            f"  Backtesting metric: {best_metric}\n"
        )
            
    return results
