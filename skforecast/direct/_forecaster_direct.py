################################################################################
#                               ForecasterDirect                               #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
from typing import Callable
import warnings
import sys
import numpy as np
import pandas as pd
import inspect
from copy import copy, deepcopy
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from joblib import Parallel, delayed, cpu_count

import skforecast
from ..base import ForecasterBase
from ..exceptions import DataTransformationWarning, ResidualsUsageWarning
from ..utils import (
    initialize_lags,
    initialize_window_features,
    initialize_weights,
    check_select_fit_kwargs,
    check_y,
    check_exog,
    get_exog_dtypes,
    check_exog_dtypes,
    prepare_steps_direct,
    check_predict_input,
    check_residuals_input,
    check_interval,
    preprocess_y,
    preprocess_last_window,
    preprocess_exog,
    input_to_frame,
    exog_to_direct,
    exog_to_direct_numpy,
    expand_index,
    transform_numpy,
    transform_dataframe,
    select_n_jobs_fit_forecaster
)
from ..preprocessing import TimeSeriesDifferentiator, QuantileBinner


class ForecasterDirect(ForecasterBase):
    """
    This class turns any regressor compatible with the scikit-learn API into a
    autoregressive direct multi-step forecaster. A separate model is created for
    each forecast time step. See documentation for more details.
    
    Parameters
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
    steps : int
        Maximum number of future steps the forecaster will predict when using
        method `predict()`. Since a different model is created for each step,
        this value should be defined before training.
    lags : int, list, numpy ndarray, range, default None
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
    
        - `int`: include lags from 1 to `lags` (included).
        - `list`, `1d numpy ndarray` or `range`: include only lags present in 
        `lags`, all elements must be int.
        - `None`: no lags are included as predictors. 
    window_features : object, list, default None
        Instance or list of instances used to create window features. Window features
        are created from the original time series and are included as predictors.
    transformer_y : object transformer (preprocessor), default None
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
        The transformation is applied to `y` before training the forecaster. 
    transformer_exog : object transformer (preprocessor), default None
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    weight_func : Callable, default None
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        Ignored if `regressor` does not have the argument `sample_weight` in its `fit`
        method. The resulting `sample_weight` cannot have negative values.
    differentiation : int, default None
        Order of differencing applied to the time series before training the forecaster.
        If `None`, no differencing is applied. The order of differentiation is the number
        of times the differencing operation is applied to a time series. Differencing
        involves computing the differences between consecutive data points in the series.
        Before returning a prediction, the differencing operation is reversed.
    fit_kwargs : dict, default None
        Additional arguments to be passed to the `fit` method of the regressor.
    binner_kwargs : dict, default None
        Additional arguments to pass to the `QuantileBinner` used to discretize 
        the residuals into k bins according to the predicted values associated 
        with each residual. Available arguments are: `n_bins`, `method`, `subsample`,
        `random_state` and `dtype`. Argument `method` is passed internally to the
        function `numpy.percentile`.
        **New in version 0.15.0**
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_fit_forecaster.
    forecaster_id : str, int, default None
        Name used as an identifier of the forecaster.
    
    Attributes
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
        An instance of this regressor is trained for each step. All of them 
        are stored in `self.regressors_`.
    regressors_ : dict
        Dictionary with regressors trained for each step. They are initialized 
        as a copy of `regressor`.
    steps : int
        Number of future steps the forecaster will predict when using method
        `predict()`. Since a different model is created for each step, this value
        should be defined before training.
    lags : numpy ndarray
        Lags used as predictors.
    lags_names : list
        Names of the lags used as predictors.
    max_lag : int
        Maximum lag included in `lags`.
    window_features : list
        Class or list of classes used to create window features.
    window_features_names : list
        Names of the window features to be included in the `X_train` matrix.
    window_features_class_names : list
        Names of the classes used to create the window features.
    max_size_window_features : int
        Maximum window size required by the window features.
    window_size : int
        The window size needed to create the predictors. It is calculated as the 
        maximum value between `max_lag` and `max_size_window_features`. If 
        differentiation is used, `window_size` is increased by n units equal to 
        the order of differentiation so that predictors can be generated correctly.
    transformer_y : object transformer (preprocessor)
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
        The transformation is applied to `y` before training the forecaster.
    transformer_exog : object transformer (preprocessor)
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    weight_func : Callable
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        Ignored if `regressor` does not have the argument `sample_weight` in its `fit`
        method. The resulting `sample_weight` cannot have negative values.
    source_code_weight_func : str
        Source code of the custom function used to create weights.
    differentiation : int
        Order of differencing applied to the time series before training the 
        forecaster.
    differentiation_max : int
        Maximum order of differentiation. For this Forecaster, it is equal to
        the value of the `differentiation` parameter.
    differentiator : TimeSeriesDifferentiator
        Skforecast object used to differentiate the time series.
    last_window_ : pandas DataFrame
        This window represents the most recent data observed by the predictor
        during its training phase. It contains the values needed to predict the
        next step immediately after the training data. These values are stored
        in the original scale of the time series before undergoing any transformations
        or differentiation. When `differentiation` parameter is specified, the
        dimensions of the `last_window_` are expanded as many values as the order
        of differentiation. For example, if `lags` = 7 and `differentiation` = 1,
        `last_window_` will have 8 values.
    index_type_ : type
        Type of index of the input used in training.
    index_freq_ : str
        Frequency of Index of the input used in training.
    training_range_ : pandas Index
        First and last values of index of the data used during training.
    exog_in_ : bool
        If the forecaster has been trained using exogenous variable/s.
    exog_names_in_ : list
        Names of the exogenous variables used during training.
    exog_type_in_ : type
        Type of exogenous data (pandas Series or DataFrame) used in training.
    exog_dtypes_in_ : dict
        Type of each exogenous variable/s used in training. If `transformer_exog` 
        is used, the dtypes are calculated after the transformation.
    X_train_window_features_names_out_ : list
        Names of the window features included in the matrix `X_train` created
        internally for training.
    X_train_exog_names_out_ : list
        Names of the exogenous variables included in the matrix `X_train` created
        internally for training. It can be different from `exog_names_in_` if
        some exogenous variables are transformed during the training process.
    X_train_direct_exog_names_out_ : list
        Same as `X_train_exog_names_out_` but using the direct format. The same 
        exogenous variable is repeated for each step.
    X_train_features_names_out_ : list
        Names of columns of the matrix created internally for training.
    fit_kwargs : dict
        Additional arguments to be passed to the `fit` method of the regressor.
    in_sample_residuals_ : dict
        Residuals of the model when predicting training data. Only stored up 
        to 10_000 values per step in the form `{step: residuals}`. If 
        `transformer_y` is not `None`, residuals are stored in the 
        transformed scale. If `differentiation` is not `None`, residuals are 
        stored after differentiation.
    in_sample_residuals_by_bin_ : dict
        In sample residuals binned according to the predicted value each residual
        is associated with. The number of residuals stored per bin is limited to 
        `10_000 // self.binner.n_bins_` in the form `{bin: residuals}`. If 
        `transformer_y` is not `None`, residuals are stored in the transformed 
        scale. If `differentiation` is not `None`, residuals are stored after 
        differentiation. 
        **New in version 0.15.0**
    out_sample_residuals_ : dict
        Residuals of the model when predicting non-training data. Only stored up 
        to 10_000 values per step in the form `{step: residuals}`. Use 
        `set_out_sample_residuals()` method to set values. If `transformer_y` 
        is not `None`, residuals are stored in the transformed scale. If 
        `differentiation` is not `None`, residuals are stored after differentiation. 
    out_sample_residuals_by_bin_ : dict
        Out of sample residuals binned according to the predicted value each residual
        is associated with. The number of residuals stored per bin is limited to 
        `10_000 // self.binner.n_bins_` in the form `{bin: residuals}`. If 
        `transformer_y` is not `None`, residuals are stored in the transformed 
        scale. If `differentiation` is not `None`, residuals are stored after 
        differentiation. 
        **New in version 0.15.0**
    binner : skforecast.preprocessing.QuantileBinner
        `QuantileBinner` used to discretize residuals into k bins according 
        to the predicted values associated with each residual.
        **New in version 0.15.0**
    binner_intervals_ : dict
        Intervals used to discretize residuals into k bins according to the predicted
        values associated with each residual.
        **New in version 0.15.0**
    binner_kwargs : dict
        Additional arguments to pass to the `QuantileBinner`.
        **New in version 0.15.0**
    creation_date : str
        Date of creation.
    is_fitted : bool
        Tag to identify if the regressor has been fitted (trained).
    fit_date : str
        Date of last fit.
    skforecast_version : str
        Version of skforecast library used to create the forecaster.
    python_version : str
        Version of python used to create the forecaster.
    n_jobs : int, 'auto'
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_fit_forecaster.
    forecaster_id : str, int
        Name used as an identifier of the forecaster.

    Notes
    -----
    A separate model is created for each forecasting time step. It is important to
    note that all models share the same parameter and hyperparameter configuration.
     
    """
    # TODO: REview docstring residuals and binner, adapt to n steps

    def __init__(
        self,
        regressor: object,
        steps: int,
        lags: int | list[int] | np.ndarray[int] | range[int] | None = None,
        window_features: object | list[object] | None = None,
        transformer_y: object | None = None,
        transformer_exog: object | None = None,
        weight_func: Callable | None = None,
        differentiation: int | None = None,
        fit_kwargs: dict[str, object] | None = None,
        binner_kwargs: dict[str, object] | None = None,
        n_jobs: int | str = 'auto',
        forecaster_id: str | int | None = None
    ) -> None:
        
        self.regressor                          = copy(regressor)
        self.steps                              = steps
        self.transformer_y                      = transformer_y
        self.transformer_exog                   = transformer_exog
        self.weight_func                        = weight_func
        self.source_code_weight_func            = None
        self.differentiation                    = differentiation
        self.differentiation_max                = None
        self.differentiator                     = None
        self.last_window_                       = None
        self.index_type_                        = None
        self.index_freq_                        = None
        self.training_range_                    = None
        self.exog_in_                           = False
        self.exog_names_in_                     = None
        self.exog_type_in_                      = None
        self.exog_dtypes_in_                    = None
        self.X_train_window_features_names_out_ = None
        self.X_train_exog_names_out_            = None
        self.X_train_direct_exog_names_out_     = None
        self.X_train_features_names_out_        = None
        self.in_sample_residuals_               = None
        self.out_sample_residuals_              = None
        self.in_sample_residuals_by_bin_        = None
        self.out_sample_residuals_by_bin_       = None
        self.creation_date                      = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.is_fitted                          = False
        self.fit_date                           = None
        self.skforecast_version                 = skforecast.__version__
        self.python_version                     = sys.version.split(" ")[0]
        self.forecaster_id                      = forecaster_id

        if not isinstance(steps, int):
            raise TypeError(
                f"`steps` argument must be an int greater than or equal to 1. "
                f"Got {type(steps)}."
            )

        if steps < 1:
            raise ValueError(
                f"`steps` argument must be greater than or equal to 1. Got {steps}."
            )

        self.regressors_ = {step: clone(self.regressor) for step in range(1, steps + 1)}
        self.lags, self.lags_names, self.max_lag = initialize_lags(type(self).__name__, lags)
        self.window_features, self.window_features_names, self.max_size_window_features = (
            initialize_window_features(window_features)
        )
        if self.window_features is None and self.lags is None:
            raise ValueError(
                "At least one of the arguments `lags` or `window_features` "
                "must be different from None. This is required to create the "
                "predictors used in training the forecaster."
            )
        
        self.window_size = max(
            [ws for ws in [self.max_lag, self.max_size_window_features] 
             if ws is not None]
        )
        self.window_features_class_names = None
        if window_features is not None:
            self.window_features_class_names = [
                type(wf).__name__ for wf in self.window_features
            ]

        self.in_sample_residuals_ = {step: None for step in range(1, steps + 1)}

        self.weight_func, self.source_code_weight_func, _ = initialize_weights(
            forecaster_name = type(self).__name__, 
            regressor       = regressor, 
            weight_func     = weight_func, 
            series_weights  = None
        )

        if differentiation is not None:
            if not isinstance(differentiation, int) or differentiation < 1:
                raise ValueError(
                    f"Argument `differentiation` must be an integer equal to or "
                    f"greater than 1. Got {differentiation}."
                )
            self.differentiation = differentiation
            self.differentiation_max = differentiation
            self.window_size += differentiation
            self.differentiator = TimeSeriesDifferentiator(
                order=differentiation, window_size=self.window_size
            )

        self.fit_kwargs = check_select_fit_kwargs(
                              regressor  = regressor,
                              fit_kwargs = fit_kwargs
                          )
        
        self.binner_kwargs = binner_kwargs
        if binner_kwargs is None:
            self.binner_kwargs = {
                'n_bins': 10, 'method': 'linear', 'subsample': 200000,
                'random_state': 789654, 'dtype': np.float64
            }
        self.binner = QuantileBinner(**self.binner_kwargs)
        self.binner_intervals_ = None

        if n_jobs == 'auto':
            self.n_jobs = select_n_jobs_fit_forecaster(
                              forecaster_name = type(self).__name__,
                              regressor       = self.regressor
                          )
        else:
            if not isinstance(n_jobs, int):
                raise TypeError(
                    f"`n_jobs` must be an integer or `'auto'`. Got {type(n_jobs)}."
                )
            self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()

    def __repr__(
        self
    ) -> str:
        """
        Information displayed when a ForecasterDirect object is printed.
        """

        (
            params,
            _,
            _,
            exog_names_in_,
            _,
        ) = self._preprocess_repr(
                regressor      = self.regressor,
                exog_names_in_ = self.exog_names_in_
            )
        
        params = self._format_text_repr(params)
        exog_names_in_ = self._format_text_repr(exog_names_in_)

        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Regressor: {type(self.regressor).__name__} \n"
            f"Lags: {self.lags} \n"
            f"Window features: {self.window_features_names} \n"
            f"Window size: {self.window_size} \n"
            f"Maximum steps to predict: {self.steps} \n"
            f"Exogenous included: {self.exog_in_} \n"
            f"Exogenous names: {exog_names_in_} \n"
            f"Transformer for y: {self.transformer_y} \n"
            f"Transformer for exog: {self.transformer_exog} \n"
            f"Weight function included: {True if self.weight_func is not None else False} \n"
            f"Differentiation order: {self.differentiation} \n"
            f"Training range: {self.training_range_.to_list() if self.is_fitted else None} \n"
            f"Training index type: {str(self.index_type_).split('.')[-1][:-2] if self.is_fitted else None} \n"
            f"Training index frequency: {self.index_freq_ if self.is_fitted else None} \n"
            f"Regressor parameters: {params} \n"
            f"fit_kwargs: {self.fit_kwargs} \n"
            f"Creation date: {self.creation_date} \n"
            f"Last fit date: {self.fit_date} \n"
            f"Skforecast version: {self.skforecast_version} \n"
            f"Python version: {self.python_version} \n"
            f"Forecaster id: {self.forecaster_id} \n"
        )

        return info

    def _repr_html_(self):
        """
        HTML representation of the object.
        The "General Information" section is expanded by default.
        """

        (
            params,
            _,
            _,
            exog_names_in_,
            _,
        ) = self._preprocess_repr(
                regressor      = self.regressor,
                exog_names_in_ = self.exog_names_in_
            )

        style, unique_id = self._get_style_repr_html(self.is_fitted)
        
        content = f"""
        <div class="container-{unique_id}">
            <h2>{type(self).__name__}</h2>
            <details open>
                <summary>General Information</summary>
                <ul>
                    <li><strong>Regressor:</strong> {type(self.regressor).__name__}</li>
                    <li><strong>Lags:</strong> {self.lags}</li>
                    <li><strong>Window features:</strong> {self.window_features_names}</li>
                    <li><strong>Window size:</strong> {self.window_size}</li>
                    <li><strong>Maximum steps to predict:</strong> {self.steps}</li>
                    <li><strong>Exogenous included:</strong> {self.exog_in_}</li>
                    <li><strong>Weight function included:</strong> {self.weight_func is not None}</li>
                    <li><strong>Differentiation order:</strong> {self.differentiation}</li>
                    <li><strong>Creation date:</strong> {self.creation_date}</li>
                    <li><strong>Last fit date:</strong> {self.fit_date}</li>
                    <li><strong>Skforecast version:</strong> {self.skforecast_version}</li>
                    <li><strong>Python version:</strong> {self.python_version}</li>
                    <li><strong>Forecaster id:</strong> {self.forecaster_id}</li>
                </ul>
            </details>
            <details>
                <summary>Exogenous Variables</summary>
                <ul>
                    {exog_names_in_}
                </ul>
            </details>
            <details>
                <summary>Data Transformations</summary>
                <ul>
                    <li><strong>Transformer for y:</strong> {self.transformer_y}</li>
                    <li><strong>Transformer for exog:</strong> {self.transformer_exog}</li>
                </ul>
            </details>
            <details>
                <summary>Training Information</summary>
                <ul>
                    <li><strong>Training range:</strong> {self.training_range_.to_list() if self.is_fitted else 'Not fitted'}</li>
                    <li><strong>Training index type:</strong> {str(self.index_type_).split('.')[-1][:-2] if self.is_fitted else 'Not fitted'}</li>
                    <li><strong>Training index frequency:</strong> {self.index_freq_ if self.is_fitted else 'Not fitted'}</li>
                </ul>
            </details>
            <details>
                <summary>Regressor Parameters</summary>
                <ul>
                    {params}
                </ul>
            </details>
            <details>
                <summary>Fit Kwargs</summary>
                <ul>
                    {self.fit_kwargs}
                </ul>
            </details>
            <p>
                <a href="https://skforecast.org/{skforecast.__version__}/api/forecasterdirect.html">&#128712 <strong>API Reference</strong></a>
                &nbsp;&nbsp;
                <a href="https://skforecast.org/{skforecast.__version__}/user_guides/direct-multi-step-forecasting.html">&#128462 <strong>User Guide</strong></a>
            </p>
        </div>
        """

        # Return the combined style and content
        return style + content

    def _create_lags(
        self, 
        y: np.ndarray,
        X_as_pandas: bool = False,
        train_index: pd.Index | None = None
    ) -> tuple[np.ndarray | pd.DataFrame | None, np.ndarray]:
        """
        Create the lagged values and their target variable from a time series.
        
        Note that the returned matrix `X_data` contains the lag 1 in the first 
        column, the lag 2 in the in the second column and so on.
        
        Parameters
        ----------
        y : numpy ndarray
            Training time series values.
        X_as_pandas : bool, default False
            If `True`, the returned matrix `X_data` is a pandas DataFrame.
        train_index : pandas Index, default None
            Index of the training data. It is used to create the pandas DataFrame
            `X_data` when `X_as_pandas` is `True`.

        Returns
        -------
        X_data : numpy ndarray, pandas DataFrame, None
            Lagged values (predictors).
        y_data : numpy ndarray
            Values of the time series related to each row of `X_data`.
        
        """
        
        n_rows = len(y) - self.window_size - (self.steps - 1)
        
        X_data = None
        if self.lags is not None:
            X_data = np.full(
                shape=(n_rows, len(self.lags)), fill_value=np.nan, order='F', dtype=float
            )
            for i, lag in enumerate(self.lags):
                X_data[:, i] = y[self.window_size - lag : -(lag + self.steps - 1)] 

            if X_as_pandas:
                X_data = pd.DataFrame(
                             data    = X_data,
                             columns = self.lags_names,
                             index   = train_index
                         )

        y_data = np.full(
            shape=(n_rows, self.steps), fill_value=np.nan, order='F', dtype=float
        )
        for step in range(self.steps):
            y_data[:, step] = y[self.window_size + step : self.window_size + step + n_rows]
        
        return X_data, y_data

    def _create_window_features(
        self, 
        y: pd.Series,
        train_index: pd.Index,
        X_as_pandas: bool = False,
    ) -> tuple[list[np.ndarray | pd.DataFrame], list[str]]:
        """
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        train_index : pandas Index
            Index of the training data. It is used to create the pandas DataFrame
            `X_train_window_features` when `X_as_pandas` is `True`.
        X_as_pandas : bool, default False
            If `True`, the returned matrix `X_train_window_features` is a 
            pandas DataFrame.

        Returns
        -------
        X_train_window_features : list
            List of numpy ndarrays or pandas DataFrames with the window features.
        X_train_window_features_names_out_ : list
            Names of the window features.
        
        """

        len_train_index = len(train_index)
        X_train_window_features = []
        X_train_window_features_names_out_ = []
        for wf in self.window_features:
            X_train_wf = wf.transform_batch(y)
            if not isinstance(X_train_wf, pd.DataFrame):
                raise TypeError(
                    f"The method `transform_batch` of {type(wf).__name__} "
                    f"must return a pandas DataFrame."
                )
            X_train_wf = X_train_wf.iloc[-len_train_index:]
            if not len(X_train_wf) == len_train_index:
                raise ValueError(
                    f"The method `transform_batch` of {type(wf).__name__} "
                    f"must return a DataFrame with the same number of rows as "
                    f"the input time series - (`window_size` + (`steps` - 1)): {len_train_index}."
                )
            X_train_wf.index = train_index
            
            X_train_window_features_names_out_.extend(X_train_wf.columns)
            if not X_as_pandas:
                X_train_wf = X_train_wf.to_numpy()     
            X_train_window_features.append(X_train_wf)

        return X_train_window_features, X_train_window_features_names_out_

    def _create_train_X_y(
        self,
        y: pd.Series,
        exog: pd.Series | pd.DataFrame | None = None
    ) -> tuple[
        pd.DataFrame, 
        dict[int, pd.Series], 
        list[str], 
        list[str], 
        list[str], 
        dict[str, type]
    ]:
        """
        Create training matrices from univariate time series and exogenous
        variables. The resulting matrices contain the target variable and 
        predictors needed to train all the regressors (one per step).
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and their indexes must be aligned.

        Returns
        -------
        X_train : pandas DataFrame
            Training values (predictors) for each step. Note that the index 
            corresponds to that of the last step. It is updated for the corresponding 
            step in the `filter_train_X_y_for_step` method.
        y_train : dict
            Values of the time series related to each row of `X_train` for each 
            step in the form {step: y_step_[i]}.
        exog_names_in_ : list
            Names of the exogenous variables used during training.
        X_train_exog_names_out_ : list
            Names of the exogenous variables included in the matrix `X_train` created
            internally for training. It can be different from `exog_names_in_` if
            some exogenous variables are transformed during the training process.
        X_train_features_names_out_ : list
            Names of the columns of the matrix created internally for training.
        exog_dtypes_in_ : dict
            Type of each exogenous variable/s used in training. If `transformer_exog` 
            is used, the dtypes are calculated before the transformation.
        
        """

        check_y(y=y)
        y = input_to_frame(data=y, input_name='y')

        if len(y) < self.window_size + self.steps:
            raise ValueError(
                f"Minimum length of `y` for training this forecaster is "
                f"{self.window_size + self.steps}. Reduce the number of "
                f"predicted steps, {self.steps}, or the maximum "
                f"window_size, {self.window_size}, if no more data is available.\n"
                f"    Length `y`: {len(y)}.\n"
                f"    Max step : {self.steps}.\n"
                f"    Max window size: {self.window_size}.\n"
                f"    Lags window size: {self.max_lag}.\n"
                f"    Window features window size: {self.max_size_window_features}."
            )

        fit_transformer = False if self.is_fitted else True
        y = transform_dataframe(
                df                = y, 
                transformer       = self.transformer_y,
                fit               = fit_transformer,
                inverse_transform = False,
            )
        y_values, y_index = preprocess_y(y=y)

        if self.differentiation is not None:
            if not self.is_fitted:
                y_values = self.differentiator.fit_transform(y_values)
            else:
                differentiator = copy(self.differentiator)
                y_values = differentiator.fit_transform(y_values)

        exog_names_in_ = None
        exog_dtypes_in_ = None
        categorical_features = False
        if exog is not None:
            check_exog(exog=exog, allow_nan=True)
            exog = input_to_frame(data=exog, input_name='exog')

            y_index_no_ws = y_index[self.window_size:]
            len_y = len(y_values)
            len_y_no_ws = len_y - self.window_size
            len_exog = len(exog)
            if not len_exog == len_y and not len_exog == len_y_no_ws:
                raise ValueError(
                    f"Length of `exog` must be equal to the length of `y` (if index is "
                    f"fully aligned) or length of `y` - `window_size` (if `exog` "
                    f"starts after the first `window_size` values).\n"
                    f"    `exog`              : ({exog.index[0]} -- {exog.index[-1]})  (n={len_exog})\n"
                    f"    `y`                 : ({y.index[0]} -- {y.index[-1]})  (n={len_y})\n"
                    f"    `y` - `window_size` : ({y_index_no_ws[0]} -- {y_index_no_ws[-1]})  (n={len_y_no_ws})"
                )
            
            # NOTE: Need here for filter_train_X_y_for_step to work without fitting
            self.exog_in_ = True
            exog_names_in_ = exog.columns.to_list()
            exog_dtypes_in_ = get_exog_dtypes(exog=exog)

            exog = transform_dataframe(
                       df                = exog,
                       transformer       = self.transformer_exog,
                       fit               = fit_transformer,
                       inverse_transform = False
                   )

            check_exog_dtypes(exog, call_check_exog=True)
            categorical_features = (
                exog.select_dtypes(include=np.number).shape[1] != exog.shape[1]
            )

            _, exog_index = preprocess_exog(exog=exog, return_values=False)
            if len_exog == len_y:
                if not (exog_index == y_index).all():
                    raise ValueError(
                        "When `exog` has the same length as `y`, the index of "
                        "`exog` must be aligned with the index of `y` "
                        "to ensure the correct alignment of values."
                    )
                # The first `self.window_size` positions have to be removed from 
                # exog since they are not in X_train.
                exog = exog.iloc[self.window_size:, ]
            else:
                if not (exog_index == y_index_no_ws).all():
                    raise ValueError(
                        "When `exog` doesn't contain the first `window_size` observations, "
                        "the index of `exog` must be aligned with the index of `y` minus "
                        "the first `window_size` observations to ensure the correct "
                        "alignment of values."
                    )
        
        X_train = []
        X_train_features_names_out_ = []
        train_index = y_index[self.window_size + (self.steps - 1):]
        len_train_index = len(train_index)
        X_as_pandas = True if categorical_features else False

        X_train_lags, y_train = self._create_lags(
            y=y_values, X_as_pandas=X_as_pandas, train_index=train_index
        )
        if X_train_lags is not None:
            X_train.append(X_train_lags)
            X_train_features_names_out_.extend(self.lags_names)
        
        X_train_window_features_names_out_ = None
        if self.window_features is not None:
            n_diff = 0 if self.differentiation is None else self.differentiation
            end_wf = None if self.steps == 1 else -(self.steps - 1)
            y_window_features = pd.Series(
                y_values[n_diff:end_wf], index=y_index[n_diff:end_wf]
            )
            X_train_window_features, X_train_window_features_names_out_ = (
                self._create_window_features(
                    y           = y_window_features, 
                    X_as_pandas = X_as_pandas, 
                    train_index = train_index
                )
            )
            X_train.extend(X_train_window_features)
            X_train_features_names_out_.extend(X_train_window_features_names_out_)

        # NOTE: Need here for filter_train_X_y_for_step to work without fitting
        self.X_train_window_features_names_out_ = X_train_window_features_names_out_

        X_train_exog_names_out_ = None
        if exog is not None:
            X_train_exog_names_out_ = exog.columns.to_list()
            if X_as_pandas:
                exog_direct, X_train_direct_exog_names_out_ = exog_to_direct(
                    exog=exog, steps=self.steps
                )
                exog_direct.index = train_index
            else:
                exog_direct, X_train_direct_exog_names_out_ = exog_to_direct_numpy(
                    exog=exog, steps=self.steps
                )

            # NOTE: Need here for filter_train_X_y_for_step to work without fitting
            self.X_train_direct_exog_names_out_ = X_train_direct_exog_names_out_

            X_train_features_names_out_.extend(self.X_train_direct_exog_names_out_)
            X_train.append(exog_direct)
        
        if len(X_train) == 1:
            X_train = X_train[0]
        else:
            if X_as_pandas:
                X_train = pd.concat(X_train, axis=1)
            else:
                X_train = np.concatenate(X_train, axis=1)
                
        if X_as_pandas:
            X_train.index = train_index
        else:
            X_train = pd.DataFrame(
                          data    = X_train,
                          index   = train_index,
                          columns = X_train_features_names_out_
                      )

        y_train = {
            step: pd.Series(
                      data  = y_train[:, step - 1], 
                      index = y_index[self.window_size + step - 1:][:len_train_index],
                      name  = f"y_step_{step}"
                  )
            for step in range(1, self.steps + 1)
        }
        
        return (
            X_train,
            y_train,
            exog_names_in_,
            X_train_exog_names_out_,
            X_train_features_names_out_,
            exog_dtypes_in_
        )

    def create_train_X_y(
        self,
        y: pd.Series,
        exog: pd.Series | pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
        """
        Create training matrices from univariate time series and exogenous
        variables. The resulting matrices contain the target variable and 
        predictors needed to train all the regressors (one per step).
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and their indexes must be aligned.

        Returns
        -------
        X_train : pandas DataFrame
            Training values (predictors) for each step. Note that the index 
            corresponds to that of the last step. It is updated for the corresponding 
            step in the `filter_train_X_y_for_step` method.
        y_train : dict
            Values of the time series related to each row of `X_train` for each 
            step in the form {step: y_step_[i]}.
        
        """

        output = self._create_train_X_y(y=y, exog=exog)

        X_train = output[0]
        y_train = output[1]

        return X_train, y_train

    def filter_train_X_y_for_step(
        self,
        step: int,
        X_train: pd.DataFrame,
        y_train: dict[int, pd.Series],
        remove_suffix: bool = False
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Select the columns needed to train a forecaster for a specific step.  
        The input matrices should be created using `create_train_X_y` method. 
        This method updates the index of `X_train` to the corresponding one 
        according to `y_train`. If `remove_suffix=True` the suffix "_step_i" 
        will be removed from the column names. 

        Parameters
        ----------
        step : int
            Step for which columns must be selected. Starts at 1.
        X_train : pandas DataFrame
            Dataframe created with the `create_train_X_y` method, first return.
        y_train : dict
            Dict created with the `create_train_X_y` method, second return.
        remove_suffix : bool, default False
            If True, suffix "_step_i" is removed from the column names.

        Returns
        -------
        X_train_step : pandas DataFrame
            Training values (predictors) for the selected step.
        y_train_step : pandas Series
            Values of the time series related to each row of `X_train`.

        """

        if (step < 1) or (step > self.steps):
            raise ValueError(
                f"Invalid value `step`. For this forecaster, minimum value is 1 "
                f"and the maximum step is {self.steps}."
            )

        y_train_step = y_train[step]

        # Matrix X_train starts at index 0.
        if not self.exog_in_:
            X_train_step = X_train
        else:
            n_lags = len(self.lags) if self.lags is not None else 0
            n_window_features = (
                len(self.X_train_window_features_names_out_) if self.window_features is not None else 0
            )
            idx_columns_autoreg = np.arange(n_lags + n_window_features)
            n_exog = len(self.X_train_direct_exog_names_out_) / self.steps
            idx_columns_exog = (
                np.arange((step - 1) * n_exog, (step) * n_exog) + idx_columns_autoreg[-1] + 1
            )
            idx_columns = np.concatenate((idx_columns_autoreg, idx_columns_exog))
            X_train_step = X_train.iloc[:, idx_columns]

        X_train_step.index = y_train_step.index

        if remove_suffix:
            X_train_step.columns = [
                col_name.replace(f"_step_{step}", "")
                for col_name in X_train_step.columns
            ]
            y_train_step.name = y_train_step.name.replace(f"_step_{step}", "")

        return X_train_step, y_train_step

    def _train_test_split_one_step_ahead(
        self,
        y: pd.Series,
        initial_train_size: int,
        exog: pd.Series | pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, dict[int, pd.Series], pd.DataFrame, dict[int, pd.Series]]:
        """
        Create matrices needed to train and test the forecaster for one-step-ahead
        predictions.

        Parameters
        ----------
        y : pandas Series
            Training time series.
        initial_train_size : int
            Initial size of the training set. It is the number of observations used
            to train the forecaster before making the first prediction.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and their indexes must be aligned.
        
        Returns
        -------
        X_train : pandas DataFrame
            Predictor values used to train the model.
        y_train : dict
            Values of the time series related to each row of `X_train` for each 
            step in the form {step: y_step_[i]}.
        X_test : pandas DataFrame
            Predictor values used to test the model.
        y_test : dict
            Values of the time series related to each row of `X_test` for each 
            step in the form {step: y_step_[i]}.
        
        """

        is_fitted = self.is_fitted
        self.is_fitted = False
        X_train, y_train, *_ = self._create_train_X_y(
            y    = y.iloc[: initial_train_size],
            exog = exog.iloc[: initial_train_size] if exog is not None else None
        )

        test_init = initial_train_size - self.window_size
        self.is_fitted = True
        X_test, y_test, *_ = self._create_train_X_y(
            y    = y.iloc[test_init:],
            exog = exog.iloc[test_init:] if exog is not None else None
        )

        self.is_fitted = is_fitted

        return X_train, y_train, X_test, y_test

    def create_sample_weights(
        self,
        X_train: pd.DataFrame,
    ) -> np.ndarray:
        """
        Create weights for each observation according to the forecaster's attribute
        `weight_func`.

        Parameters
        ----------
        X_train : pandas DataFrame
            Dataframe created with `create_train_X_y` and `filter_train_X_y_for_step`
            methods, first return.

        Returns
        -------
        sample_weight : numpy ndarray
            Weights to use in `fit` method.

        """

        sample_weight = None

        if self.weight_func is not None:
            sample_weight = self.weight_func(X_train.index)

        if sample_weight is not None:
            if np.isnan(sample_weight).any():
                raise ValueError(
                    "The resulting `sample_weight` cannot have NaN values."
                )
            if np.any(sample_weight < 0):
                raise ValueError(
                    "The resulting `sample_weight` cannot have negative values."
                )
            if np.sum(sample_weight) == 0:
                raise ValueError(
                    "The resulting `sample_weight` cannot be normalized because "
                    "the sum of the weights is zero."
                )

        return sample_weight

    def fit(
        self,
        y: pd.Series,
        exog: pd.Series | pd.DataFrame | None = None,
        store_last_window: bool = True,
        store_in_sample_residuals: bool = True,
        random_state: int = 123
    ) -> None:
        """
        Training Forecaster.

        Additional arguments to be passed to the `fit` method of the regressor 
        can be added with the `fit_kwargs` argument when initializing the forecaster.

        Parameters
        ----------
        y : pandas Series
            Training time series.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and their indexes must be aligned so
            that y[i] is regressed on exog[i].
        store_last_window : bool, default True
            Whether or not to store the last window (`last_window_`) of training data.
        store_in_sample_residuals : bool, default True
            If `True`, in-sample residuals will be stored in the forecaster object
            after fitting (`in_sample_residuals_` and `in_sample_residuals_by_bin_`
            attributes).
        random_state : int, default 123
            Set a seed for the random generator so that the stored sample 
            residuals are always deterministic.

        Returns
        -------
        None
        
        """

        # Reset values in case the forecaster has already been fitted.
        self.last_window_                       = None
        self.index_type_                        = None
        self.index_freq_                        = None
        self.training_range_                    = None
        self.exog_in_                           = False
        self.exog_names_in_                     = None
        self.exog_type_in_                      = None
        self.exog_dtypes_in_                    = None
        self.X_train_window_features_names_out_ = None
        self.X_train_exog_names_out_            = None
        self.X_train_direct_exog_names_out_     = None
        self.X_train_features_names_out_        = None
        self.in_sample_residuals_               = {step: None for step in range(1, self.steps + 1)}
        self.in_sample_residuals_by_bin_        = None
        self.binner_intervals_                  = None
        self.is_fitted                          = False
        self.fit_date                           = None

        (
            X_train,
            y_train,
            exog_names_in_,
            X_train_exog_names_out_,
            X_train_features_names_out_,
            exog_dtypes_in_
        ) = self._create_train_X_y(y=y, exog=exog)

        def fit_forecaster(regressor, X_train, y_train, step, store_in_sample_residuals, random_state):
            """
            Auxiliary function to fit each of the forecaster's regressors in parallel.

            Parameters
            ----------
            regressor : object
                Regressor to be fitted.
            X_train : pandas DataFrame
                Dataframe created with the `create_train_X_y` method, first return.
            y_train : dict
                Dict created with the `create_train_X_y` method, second return.
            step : int
                Step of the forecaster to be fitted.
            store_in_sample_residuals : bool
                If `True`, in-sample residuals will be stored in the forecaster object
                after fitting (`in_sample_residuals_` and `in_sample_residuals_by_bin_`
                attributes).
            random_state : int, default 123
                Set a seed for the random generator so that the stored sample 
                residuals are always deterministic.
            
            Returns
            -------
            Tuple with the step, fitted regressor, in-sample residuals, true values
            and predicted values for the step.

            """

            X_train_step, y_train_step = self.filter_train_X_y_for_step(
                                             step          = step,
                                             X_train       = X_train,
                                             y_train       = y_train,
                                             remove_suffix = True
                                         )
            sample_weight = self.create_sample_weights(X_train=X_train_step)
            if sample_weight is not None:
                regressor.fit(
                    X             = X_train_step,
                    y             = y_train_step,
                    sample_weight = sample_weight,
                    **self.fit_kwargs
                )
            else:
                regressor.fit(
                    X = X_train_step,
                    y = y_train_step,
                    **self.fit_kwargs
                )

            # This is done to save time during fit in functions such as backtesting()
            if store_in_sample_residuals:
                y_true_step = y_train_step.to_numpy()
                y_pred_step = regressor.predict(X_train_step)
                residuals = y_true_step - y_pred_step

                if len(residuals) > 10_000:
                    rng = np.random.default_rng(seed=random_state)
                    residuals = residuals[
                        rng.integers(low=0, high=len(residuals), size=10_000)
                    ]
            else:
                y_true_step = None
                y_pred_step = None
                residuals = None

            return step, regressor, residuals, y_true_step, y_pred_step

        results_fit = (
            Parallel(n_jobs=self.n_jobs)
            (delayed(fit_forecaster)
            (
                regressor                 = copy(self.regressor),
                X_train                   = X_train,
                y_train                   = y_train,
                step                      = step,
                store_in_sample_residuals = store_in_sample_residuals,
                random_state              = random_state
            )
            for step in range(1, self.steps + 1))
        )

        self.regressors_ = {step: regressor for step, regressor, *_ in results_fit}

        if store_in_sample_residuals:
            self.in_sample_residuals_ = {
                step: residuals 
                for step, _, residuals, *_ in results_fit
            }

            y_true, y_pred = zip(*[(y_true, y_pred) for *_, y_true, y_pred in results_fit])
            self._binning_in_sample_residuals(
                y_true       = np.concatenate(y_true),
                y_pred       = np.concatenate(y_pred),
                random_state = random_state
            )
        
        self.X_train_features_names_out_ = X_train_features_names_out_

        self.is_fitted = True
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range_ = preprocess_y(y=y, return_values=False)[1][[0, -1]]
        self.index_type_ = type(X_train.index)
        if isinstance(X_train.index, pd.DatetimeIndex):
            self.index_freq_ = X_train.index.freqstr
        else: 
            self.index_freq_ = X_train.index.step

        if exog is not None:
            self.exog_in_ = True
            self.exog_type_in_ = type(exog)
            self.exog_names_in_ = exog_names_in_
            self.exog_dtypes_in_ = exog_dtypes_in_
            self.X_train_exog_names_out_ = X_train_exog_names_out_

        if store_last_window:
            self.last_window_ = (
                y.iloc[-self.window_size:]
                .copy()
                .to_frame(name=y.name if y.name is not None else 'y')
            )

    def _binning_in_sample_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        random_state: int = 123
    ) -> None:
        """
        Bin residuals according to the predicted value each residual is
        associated with. First a `skforecast.preprocessing.QuantileBinner` object
        is fitted to the predicted values. Then, residuals are binned according
        to the predicted value each residual is associated with. Residuals are
        stored in the forecaster object as `in_sample_residuals_` and
        `in_sample_residuals_by_bin_`.

        `y_true` and `y_pred` assumed to be differentiated and or transformed
        according to the attributes `differentiation` and `transformer_y`.
        The number of residuals stored per bin is limited to 
        `10_000 // self.binner.n_bins_`. The total number of residuals stored is
        `10_000`.
        **New in version 0.15.0**

        Parameters
        ----------
        y_true : numpy ndarray
            True values of the time series.
        y_pred : numpy ndarray
            Predicted values of the time series.
        random_state : int, default 123
            Set a seed for the random generator so that the stored sample 
            residuals are always deterministic.

        Returns
        -------
        None
        
        """

        residuals = y_true - y_pred
        data = pd.DataFrame({'prediction': y_pred, 'residuals': residuals})

        data['bin'] = self.binner.fit_transform(y_pred).astype(int)
        self.in_sample_residuals_by_bin_ = (
            data.groupby('bin')['residuals'].apply(np.array).to_dict()
        )

        rng = np.random.default_rng(seed=random_state)
        max_sample = 10_000 // self.binner.n_bins_
        for k, v in self.in_sample_residuals_by_bin_.items():
            if len(v) > max_sample:
                sample = v[rng.integers(low=0, high=len(v), size=max_sample)]
                self.in_sample_residuals_by_bin_[k] = sample

        self.binner_intervals_ = self.binner.intervals_

    def _create_predict_inputs(
        self,
        steps: int | list[int] | None = None,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        predict_probabilistic: bool = False,
        use_in_sample_residuals: bool = True,
        use_binned_residuals: bool = False,
        check_inputs: bool = True
    ) -> tuple[list[np.ndarray], list[str], list[int], pd.Index]:
        """
        Create the inputs needed for the prediction process.
        
        Parameters
        ----------
        steps : int, list, None, default None
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as were defined at 
            initialization.
        last_window : pandas Series, pandas DataFrame, default None
            Series values used to create the predictors (lags) needed to 
            predict `steps`.
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.
        predict_probabilistic : bool, default False
            If `True`, the necessary checks for probabilistic predictions will be 
            performed.
        use_in_sample_residuals : bool, default True
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. 
            If `False`, out of sample residuals (calibration) are used. 
            Out-of-sample residuals must be precomputed using Forecaster's
            `set_out_sample_residuals()` method.
        use_binned_residuals : bool, default False
            If `True`, residuals are selected based on the predicted values 
            (binned selection).
            If `False`, residuals are selected randomly.
        check_inputs : bool, default True
            If `True`, the input is checked for possible warnings and errors 
            with the `check_predict_input` function. This argument is created 
            for internal use and is not recommended to be changed.

        Returns
        -------
        Xs : list
            List of numpy arrays with the predictors for each step.
        Xs_col_names : list
            Names of the columns of the matrix created internally for prediction.
        steps : list
            Steps to predict.
        prediction_index : pandas Index
            Index of the predictions.
        
        """

        steps = prepare_steps_direct(
                    max_step = self.steps,
                    steps    = steps
                )

        if last_window is None:
            last_window = self.last_window_

        if check_inputs:
            check_predict_input(
                forecaster_name = type(self).__name__,
                steps           = steps,
                is_fitted       = self.is_fitted,
                exog_in_        = self.exog_in_,
                index_type_     = self.index_type_,
                index_freq_     = self.index_freq_,
                window_size     = self.window_size,
                last_window     = last_window,
                exog            = exog,
                exog_type_in_   = self.exog_type_in_,
                exog_names_in_  = self.exog_names_in_,
                interval        = None,
                max_steps       = self.steps
            )

            if predict_probabilistic:
                check_residuals_input(
                    forecaster_name              = type(self).__name__,
                    use_in_sample_residuals      = use_in_sample_residuals,
                    in_sample_residuals_         = self.in_sample_residuals_,
                    out_sample_residuals_        = self.out_sample_residuals_,
                    use_binned_residuals         = use_binned_residuals,
                    in_sample_residuals_by_bin_  = self.in_sample_residuals_by_bin_,
                    out_sample_residuals_by_bin_ = self.out_sample_residuals_by_bin_,
                    steps                        = steps
                )

        last_window = last_window.iloc[-self.window_size:].copy()
        last_window_values, last_window_index = preprocess_last_window(
                                                    last_window = last_window
                                                )

        last_window_values = transform_numpy(
                                 array             = last_window_values,
                                 transformer       = self.transformer_y,
                                 fit               = False,
                                 inverse_transform = False
                             )
        if self.differentiation is not None:
            last_window_values = self.differentiator.fit_transform(last_window_values)

        X_autoreg = []
        Xs_col_names = []
        if self.lags is not None:
            X_lags = last_window_values[-self.lags]
            X_autoreg.append(X_lags)
            Xs_col_names.extend(self.lags_names)

        if self.window_features is not None:
            n_diff = 0 if self.differentiation is None else self.differentiation
            X_window_features = np.concatenate(
                [
                    wf.transform(last_window_values[n_diff:])
                    for wf in self.window_features
                ]
            )
            X_autoreg.append(X_window_features)
            Xs_col_names.extend(self.X_train_window_features_names_out_)

        X_autoreg = np.concatenate(X_autoreg).reshape(1, -1)
        if exog is not None:
            exog = input_to_frame(data=exog, input_name='exog')
            exog = exog.loc[:, self.exog_names_in_]
            exog = transform_dataframe(
                       df                = exog,
                       transformer       = self.transformer_exog,
                       fit               = False,
                       inverse_transform = False
                   )
            check_exog_dtypes(exog=exog)
            exog_values, _ = exog_to_direct_numpy(
                                 exog  = exog.to_numpy()[:max(steps)],
                                 steps = max(steps)
                             )
            exog_values = exog_values[0]
            
            n_exog = exog.shape[1]
            Xs = [
                np.concatenate(
                    [
                        X_autoreg,
                        exog_values[(step - 1) * n_exog : step * n_exog].reshape(1, -1),
                    ],
                    axis=1
                )
                for step in steps
            ]
            # HACK: This is not the best way to do it. Can have any problem
            # if the exog_columns are not in the same order as the
            # self.window_features_names.
            Xs_col_names = Xs_col_names + exog.columns.to_list()
        else:
            Xs = [X_autoreg] * len(steps)

        prediction_index = expand_index(
                               index = last_window_index,
                               steps = max(steps)
                           )[np.array(steps) - 1]
        if isinstance(last_window_index, pd.DatetimeIndex) and np.array_equal(
            steps, np.arange(min(steps), max(steps) + 1)
        ):
            prediction_index.freq = last_window_index.freq
        
        # HACK: Why no use self.X_train_features_names_out_ as Xs_col_names?
        return Xs, Xs_col_names, steps, prediction_index

    def create_predict_X(
        self,
        steps: int | list[int] | None = None,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Create the predictors needed to predict `steps` ahead.
        
        Parameters
        ----------
        steps : int, list, None, default None
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as were defined at 
            initialization.
        last_window : pandas Series, pandas DataFrame, default None
            Series values used to create the predictors (lags) needed to 
            predict `steps`.
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.

        Returns
        -------
        X_predict : pandas DataFrame
            Pandas DataFrame with the predictors for each step. The index 
            is the same as the prediction index.
        
        """
        
        (
            Xs,
            Xs_col_names,
            steps,
            prediction_index
        ) = self._create_predict_inputs(
                steps        = steps,
                last_window  = last_window,
                exog         = exog
            )

        X_predict = pd.DataFrame(
                        data    = np.concatenate(Xs, axis=0), 
                        columns = Xs_col_names, 
                        index   = prediction_index
                    )
        
        if self.transformer_y is not None or self.differentiation is not None:
            warnings.warn(
                "The output matrix is in the transformed scale due to the "
                "inclusion of transformations or differentiation in the Forecaster. "
                "As a result, any predictions generated using this matrix will also "
                "be in the transformed scale. Please refer to the documentation "
                "for more details: "
                "https://skforecast.org/latest/user_guides/training-and-prediction-matrices.html",
                DataTransformationWarning
            )

        return X_predict

    def predict(
        self,
        steps: int | list[int] | None = None,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        check_inputs: bool = True
    ) -> pd.Series:
        """
        Predict n steps ahead.

        Parameters
        ----------
        steps : int, list, None, default None
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as were defined at 
            initialization.
        last_window : pandas Series, pandas DataFrame, default None
            Series values used to create the predictors (lags) needed to 
            predict `steps`.
            If `last_window = None`, the values stored in` self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.
        check_inputs : bool, default True
            If `True`, the input is checked for possible warnings and errors 
            with the `check_predict_input` function. This argument is created 
            for internal use and is not recommended to be changed.

        Returns
        -------
        predictions : pandas Series
            Predicted values.

        """

        (
            Xs,
            _,
            steps,
            prediction_index
        ) = self._create_predict_inputs(
                steps        = steps,
                last_window  = last_window,
                exog         = exog,
                check_inputs = check_inputs,
            )

        regressors = [self.regressors_[step] for step in steps]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", 
                message="X does not have valid feature names", 
                category=UserWarning
            )
            predictions = np.array([
                regressor.predict(X).ravel()[0] 
                for regressor, X in zip(regressors, Xs)
            ])

        if self.differentiation is not None:
            predictions = self.differentiator.inverse_transform_next_window(predictions)

        predictions = transform_numpy(
                          array             = predictions,
                          transformer       = self.transformer_y,
                          fit               = False,
                          inverse_transform = True
                      )

        predictions = pd.Series(
                          data  = predictions,
                          index = prediction_index,
                          name  = 'pred'
                      )

        return predictions

    def predict_bootstrapping(
        self,
        steps: int | list[int] | None = None,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        n_boot: int = 250,
        use_in_sample_residuals: bool = True,
        use_binned_residuals: bool = False,
        random_state: int = 123
    ) -> pd.DataFrame:
        """
        Generate multiple forecasting predictions using a bootstrapping process. 
        By sampling from a collection of past observed errors (the residuals),
        each iteration of bootstrapping generates a different set of predictions. 
        See the References section for more information. 
        
        Parameters
        ----------
        steps : int, list, None, default None
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as were defined at 
            initialization.
        last_window : pandas Series, pandas DataFrame, default None
            Series values used to create the predictors (lags) needed to 
            predict `steps`.
            If `last_window = None`, the values stored in` self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.
        n_boot : int, default 250
            Number of bootstrapping iterations to perform when estimating prediction
            intervals.
        use_in_sample_residuals : bool, default True
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. 
            If `False`, out of sample residuals (calibration) are used. 
            Out-of-sample residuals must be precomputed using Forecaster's
            `set_out_sample_residuals()` method.
        use_binned_residuals : bool, default False
            If `True`, residuals are selected based on the predicted values 
            (binned selection).
            If `False`, residuals are selected randomly.
        random_state : int, default 123
            Seed for the random number generator to ensure reproducibility.

        Returns
        -------
        boot_predictions : pandas DataFrame
            Predictions generated by bootstrapping.
            Shape: (steps, n_boot)

        References
        ----------
        .. [1] Forecasting: Principles and Practice (3rd ed) Rob J Hyndman and George Athanasopoulos.
               https://otexts.com/fpp3/prediction-intervals.html

        """

        (
            Xs,
            _,
            steps,
            prediction_index
        ) = self._create_predict_inputs(
                steps                   = steps, 
                last_window             = last_window, 
                exog                    = exog,
                predict_probabilistic   = True, 
                use_in_sample_residuals = use_in_sample_residuals,
                use_binned_residuals    = use_binned_residuals
            )

        if use_in_sample_residuals:
            residuals = self.in_sample_residuals_
            residuals_by_bin = self.in_sample_residuals_by_bin_
        else:
            residuals = self.out_sample_residuals_
            residuals_by_bin = self.out_sample_residuals_by_bin_

        # NOTE: Since residuals are {step/bin: residuals}, more n_boot iterations
        # than the number of residuals for the step/bin with more residuals, 
        # doesn't add any new information to the bootstrapping process.
        if use_binned_residuals:
            recommended_n_boot = np.max([v.size for v in residuals_by_bin.values()])
        else:
            recommended_n_boot = np.max([v.size for v in residuals.values()])
        
        if n_boot > recommended_n_boot:
            warnings.warn(
                f"`n_boot`, {n_boot}, is greater than the number of available "
                f"residuals. More than {recommended_n_boot} iterations don't "
                f"add new information to the bootstrapping process, but increase "
                f"the computational cost.",
                ResidualsUsageWarning
            )

        # NOTE: Predictors and Residuals are transformed and differenced.  
        regressors = [self.regressors_[step] for step in steps]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", 
                message="X does not have valid feature names", 
                category=UserWarning
            )
            predictions = np.array([
                regressor.predict(X).ravel()[0] 
                for regressor, X in zip(regressors, Xs)
            ])

        boot_predictions = np.tile(predictions, (n_boot, 1)).T
        boot_columns = [f"pred_boot_{i}" for i in range(n_boot)]
        
        rng = np.random.default_rng(seed=random_state)
        for i, step in enumerate(steps):

            if use_binned_residuals:
                predicted_bin = self.binner.transform(predictions[i]).item()
                step_residuals = residuals_by_bin[predicted_bin]
            else:
                step_residuals = residuals[step]
            len_step_residuals = len(step_residuals)

            # NOTE: If n_boot != len_step_residuals, upsample or downsample the 
            # residuals from the step/bin to match n_boot.
            if len_step_residuals != n_boot:
                step_residuals = step_residuals[
                    rng.integers(low=0, high=len_step_residuals, size=n_boot)
                ]
            
            boot_predictions[i, :] = boot_predictions[i, :] + step_residuals

        if self.differentiation is not None:
            boot_predictions = (
                self.differentiator.inverse_transform_next_window(boot_predictions)
            )

        if self.transformer_y:
            boot_predictions = np.apply_along_axis(
                                   func1d            = transform_numpy,
                                   axis              = 0,
                                   arr               = boot_predictions,
                                   transformer       = self.transformer_y,
                                   fit               = False,
                                   inverse_transform = True
                               )
    
        boot_predictions = pd.DataFrame(
                               data    = boot_predictions,
                               index   = prediction_index,
                               columns = boot_columns
                           )
        
        return boot_predictions
    
    def _predict_interval_conformal(
        self,
        steps: int | list[int] | None = None,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        nominal_coverage: float = 0.95,
        use_in_sample_residuals: bool = True,
        use_binned_residuals: bool = False
    ) -> pd.DataFrame:
        """
        Generate prediction intervals using the conformal prediction 
        split method [1]_.

        Parameters
        ----------
        steps : int, list, None, default None
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as were defined at 
            initialization.
        last_window : pandas Series, pandas DataFrame, default None
            Series values used to create the predictors (lags) needed to 
            predict `steps`.
            If `last_window = None`, the values stored in` self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.
        nominal_coverage : float, default 0.95
            Nominal coverage, also known as expected coverage, of the prediction
            intervals. Must be between 0 and 1.
        use_in_sample_residuals : bool, default True
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. 
            If `False`, out of sample residuals (calibration) are used. 
            Out-of-sample residuals must be precomputed using Forecaster's
            `set_out_sample_residuals()` method.
        use_binned_residuals : bool, default False
            If `True`, residuals are selected based on the predicted values 
            (binned selection).
            If `False`, residuals are selected randomly.

        Returns
        -------
        predictions : pandas DataFrame
            Values predicted by the forecaster and their estimated interval.

            - pred: predictions.
            - lower_bound: lower bound of the interval.
            - upper_bound: upper bound of the interval.

        References
        ----------
        .. [1] MAPIE - Model Agnostic Prediction Interval Estimator.
               https://mapie.readthedocs.io/en/stable/theoretical_description_regression.html#the-split-method

        """

        (
            Xs,
            _,
            steps,
            prediction_index
        ) = self._create_predict_inputs(
                steps                   = steps, 
                last_window             = last_window, 
                exog                    = exog,
                predict_probabilistic   = True, 
                use_in_sample_residuals = use_in_sample_residuals,
                use_binned_residuals    = use_binned_residuals
            )

        if use_in_sample_residuals:
            residuals = self.in_sample_residuals_
            residuals_by_bin = self.in_sample_residuals_by_bin_
        else:
            residuals = self.out_sample_residuals_
            residuals_by_bin = self.out_sample_residuals_by_bin_

        # NOTE: Predictors and Residuals are transformed and differenced.  
        regressors = [self.regressors_[step] for step in steps]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", 
                message="X does not have valid feature names", 
                category=UserWarning
            )
            predictions = np.array([
                regressor.predict(X).ravel()[0] 
                for regressor, X in zip(regressors, Xs)
            ])
        
        if use_binned_residuals:
            correction_factor_by_bin = {
                k: np.quantile(np.abs(v), nominal_coverage)
                for k, v in residuals_by_bin.items()
            }
            replace_func = np.vectorize(lambda x: correction_factor_by_bin[x])
            predictions_bin = self.binner.transform(predictions)
            correction_factor = replace_func(predictions_bin)
            lower_bound = predictions - correction_factor
            upper_bound = predictions + correction_factor
        else:
            for step in steps:
                correction_factor = np.quantile(np.abs(residuals[step]), nominal_coverage)
                lower_bound = predictions - correction_factor
                upper_bound = predictions + correction_factor

        predictions = np.column_stack([predictions, lower_bound, upper_bound])

        if self.differentiation is not None:
            predictions = (
                self.differentiator.inverse_transform_next_window(predictions)
            )
        
        if self.transformer_y:
            predictions = np.apply_along_axis(
                              func1d            = transform_numpy,
                              axis              = 0,
                              arr               = predictions,
                              transformer       = self.transformer_y,
                              fit               = False,
                              inverse_transform = True
                          )
        
        predictions = pd.DataFrame(
                          data    = predictions,
                          index   = prediction_index,
                          columns = ["pred", "lower_bound", "upper_bound"]
                      )

        return predictions

    def predict_interval(
        self,
        steps: int | list[int] | None = None,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        method: str = 'bootstrapping',
        interval: float | list[float] | tuple[float] = [5, 95],
        n_boot: int = 250,
        use_in_sample_residuals: bool = True,
        use_binned_residuals: bool = False,
        random_state: int = 123
    ) -> pd.DataFrame:
        """
        Predict n steps ahead and estimate prediction intervals using either 
        bootstrapping or conformal prediction methods. Refer to the References 
        section for additional details on these methods.
        
        Parameters
        ----------
        steps : int, list, None, default None
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as were defined at 
            initialization.
        last_window : pandas Series, pandas DataFrame, default None
            Series values used to create the predictors (lags) needed to 
            predict `steps`.
            If `last_window = None`, the values stored in` self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.
        method : str, default 'bootstrapping'
            Technique used to estimate prediction intervals. Available options:

            - 'bootstrapping': Bootstrapping is used to generate prediction 
            intervals [1]_.
            - 'conformal': Employs the conformal prediction split method for 
            interval estimation [2]_.
        interval : float, list, tuple, default [5, 95]
            Confidence level of the prediction interval. Interpretation depends 
            on the method used:
            
            - If `float`, represents the nominal (expected) coverage (between 0 
            and 1). For instance, `interval=0.95` corresponds to `[2.5, 97.5]` 
            percentiles.
            - If `list` or `tuple`, defines the exact percentiles to compute, which 
            must be between 0 and 100 inclusive. For example, interval 
            of 95% should be as `interval = [2.5, 97.5]`.
            - When using `method='conformal'`, the interval must be a float or 
            a list/tuple defining a symmetric interval.
        n_boot : int, default 250
            Number of bootstrapping iterations to perform when estimating prediction
            intervals.
        use_in_sample_residuals : bool, default True
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. 
            If `False`, out of sample residuals (calibration) are used. 
            Out-of-sample residuals must be precomputed using Forecaster's
            `set_out_sample_residuals()` method.
        use_binned_residuals : bool, default False
            If `True`, residuals are selected based on the predicted values 
            (binned selection).
            If `False`, residuals are selected randomly.
        random_state : int, default 123
            Seed for the random number generator to ensure reproducibility.

        Returns
        -------
        predictions : pandas DataFrame
            Values predicted by the forecaster and their estimated interval.

            - pred: predictions.
            - lower_bound: lower bound of the interval.
            - upper_bound: upper bound of the interval.

        References
        ----------
        .. [1] Forecasting: Principles and Practice (3rd ed) Rob J Hyndman and George Athanasopoulos.
               https://otexts.com/fpp3/prediction-intervals.html
        
        .. [2] MAPIE - Model Agnostic Prediction Interval Estimator.
               https://mapie.readthedocs.io/en/stable/theoretical_description_regression.html#the-split-method
    
        """

        if method == "bootstrapping":
            
            if isinstance(interval, (list, tuple)):
                check_interval(interval=interval, ensure_symmetric_intervals=False)
                interval = np.array(interval) / 100
            else:
                check_interval(alpha=interval, alpha_literal='interval')
                interval = np.array([0.5 - interval / 2, 0.5 + interval / 2])

            boot_predictions = self.predict_bootstrapping(
                                   steps                   = steps,
                                   last_window             = last_window,
                                   exog                    = exog,
                                   n_boot                  = n_boot,
                                   random_state            = random_state,
                                   use_in_sample_residuals = use_in_sample_residuals,
                                   use_binned_residuals    = use_binned_residuals
                               )
    
            predictions = self.predict(
                              steps        = steps,
                              last_window  = last_window,
                              exog         = exog,
                              check_inputs = False
                          )
    
            predictions_interval = boot_predictions.quantile(q=interval, axis=1).transpose()
            predictions_interval.columns = ['lower_bound', 'upper_bound']
            predictions = pd.concat((predictions, predictions_interval), axis=1)

        elif method == 'conformal':

            if isinstance(interval, (list, tuple)):
                check_interval(interval=interval, ensure_symmetric_intervals=True)
                nominal_coverage = (interval[1] - interval[0]) / 100
            else:
                check_interval(alpha=interval, alpha_literal='interval')
                nominal_coverage = interval
            
            predictions = self._predict_interval_conformal(
                              steps                   = steps,
                              last_window             = last_window,
                              exog                    = exog,
                              nominal_coverage        = nominal_coverage,
                              use_in_sample_residuals = use_in_sample_residuals,
                              use_binned_residuals    = use_binned_residuals
                          )
        else:
            raise ValueError(
                f"Invalid `method` '{method}'. Choose 'bootstrapping' or 'conformal'."
            )

        return predictions

    def predict_quantiles(
        self,
        steps: int | list[int] | None = None,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        quantiles: list[float] | tuple[float] = [0.05, 0.5, 0.95],
        n_boot: int = 250,
        use_in_sample_residuals: bool = True,
        use_binned_residuals: bool = False,
        random_state: int = 123
    ) -> pd.DataFrame:
        """
        Bootstrapping based predicted quantiles.
        
        Parameters
        ----------
        steps : int, list, None, default None
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as were defined at 
            initialization.
        last_window : pandas Series, pandas DataFrame, default None
            Series values used to create the predictors (lags) needed to 
            predict `steps`.
            If `last_window = None`, the values stored in` self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.
        quantiles : list, tuple, default [0.05, 0.5, 0.95]
            Sequence of quantiles to compute, which must be between 0 and 1 
            inclusive. For example, quantiles of 0.05, 0.5 and 0.95 should be as 
            `quantiles = [0.05, 0.5, 0.95]`.
        n_boot : int, default 250
            Number of bootstrapping iterations to perform when estimating quantiles.
        use_in_sample_residuals : bool, default True
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. 
            If `False`, out of sample residuals (calibration) are used. 
            Out-of-sample residuals must be precomputed using Forecaster's
            `set_out_sample_residuals()` method.
        use_binned_residuals : bool, default False
            If `True`, residuals are selected based on the predicted values 
            (binned selection).
            If `False`, residuals are selected randomly.
        random_state : int, default 123
            Seed for the random number generator to ensure reproducibility.

        Returns
        -------
        predictions : pandas DataFrame
            Quantiles predicted by the forecaster.

        References
        ----------
        .. [1] Forecasting: Principles and Practice (3rd ed) Rob J Hyndman and George Athanasopoulos.
               https://otexts.com/fpp3/prediction-intervals.html
        
        """
        
        check_interval(quantiles=quantiles)

        predictions = self.predict_bootstrapping(
                          steps                   = steps,
                          last_window             = last_window,
                          exog                    = exog,
                          n_boot                  = n_boot,
                          random_state            = random_state,
                          use_in_sample_residuals = use_in_sample_residuals,
                          use_binned_residuals    = use_binned_residuals
                      )

        predictions = predictions.quantile(q=quantiles, axis=1).transpose()
        predictions.columns = [f'q_{q}' for q in quantiles]

        return predictions
    
    def predict_dist(
        self,
        distribution: object,
        steps: int | list[int] | None = None,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        n_boot: int = 250,
        use_in_sample_residuals: bool = True,
        use_binned_residuals: bool = False,
        random_state: int = 123
    ) -> pd.DataFrame:
        """
        Fit a given probability distribution for each step. After generating 
        multiple forecasting predictions through a bootstrapping process, each 
        step is fitted to the given distribution.
        
        Parameters
        ----------
        distribution : object
            A distribution object from scipy.stats with methods `_pdf` and `fit`. 
            For example scipy.stats.norm.
        steps : int, list, None, default None
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as were defined at 
            initialization.
        last_window : pandas Series, pandas DataFrame, default None
            Series values used to create the predictors (lags) needed to 
            predict `steps`.
            If `last_window = None`, the values stored in` self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.
        n_boot : int, default 250
            Number of bootstrapping iterations to perform when estimating prediction
            intervals.
        use_in_sample_residuals : bool, default True
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. 
            If `False`, out of sample residuals (calibration) are used. 
            Out-of-sample residuals must be precomputed using Forecaster's
            `set_out_sample_residuals()` method.
        use_binned_residuals : bool, default False
            If `True`, residuals are selected based on the predicted values 
            (binned selection).
            If `False`, residuals are selected randomly.
        random_state : int, default 123
            Seed for the random number generator to ensure reproducibility.

        Returns
        -------
        predictions : pandas DataFrame
            Distribution parameters estimated for each step.

        References
        ----------
        .. [1] Forecasting: Principles and Practice (3rd ed) Rob J Hyndman and George Athanasopoulos.
               https://otexts.com/fpp3/prediction-intervals.html

        """

        if not hasattr(distribution, "_pdf") or not callable(getattr(distribution, "fit", None)):
            raise TypeError(
                "`distribution` must be a valid probability distribution object "
                "from scipy.stats, with methods `_pdf` and `fit`."
            )
        
        predictions = self.predict_bootstrapping(
                          steps                   = steps,
                          last_window             = last_window,
                          exog                    = exog,
                          n_boot                  = n_boot,
                          random_state            = random_state,
                          use_in_sample_residuals = use_in_sample_residuals,
                          use_binned_residuals    = use_binned_residuals
                      )       

        param_names = [
            p for p in inspect.signature(distribution._pdf).parameters
            if not p == 'x'
        ] + ["loc", "scale"]

        predictions[param_names] = (
            predictions.apply(
                lambda x: distribution.fit(x), axis=1, result_type='expand'
            )
        )
        predictions = predictions[param_names]

        return predictions

    def set_params(
        self, 
        params: dict[str, object]
    ) -> None:
        """
        Set new values to the parameters of the scikit learn model stored in the
        forecaster. It is important to note that all models share the same 
        configuration of parameters and hyperparameters.
        
        Parameters
        ----------
        params : dict
            Parameters values.

        Returns
        -------
        None
        
        """

        self.regressor = clone(self.regressor)
        self.regressor.set_params(**params)
        self.regressors_ = {
            step: clone(self.regressor)
            for step in range(1, self.steps + 1)
        }

    def set_fit_kwargs(
        self, 
        fit_kwargs: dict[str, object]
    ) -> None:
        """
        Set new values for the additional keyword arguments passed to the `fit` 
        method of the regressor.
        
        Parameters
        ----------
        fit_kwargs : dict
            Dict of the form {"argument": new_value}.

        Returns
        -------
        None
        
        """

        self.fit_kwargs = check_select_fit_kwargs(self.regressor, fit_kwargs=fit_kwargs)

    def set_lags(
        self, 
        lags: int | list[int] | np.ndarray[int] | range[int] | None = None
    ) -> None:
        """
        Set new value to the attribute `lags`. Attributes `lags_names`, 
        `max_lag` and `window_size` are also updated.
        
        Parameters
        ----------
        lags : int, list, numpy ndarray, range, default None
            Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1. 
        
            - `int`: include lags from 1 to `lags` (included).
            - `list`, `1d numpy ndarray` or `range`: include only lags present in 
            `lags`, all elements must be int.
            - `None`: no lags are included as predictors. 

        Returns
        -------
        None
        
        """

        if self.window_features is None and lags is None:
            raise ValueError(
                "At least one of the arguments `lags` or `window_features` "
                "must be different from None. This is required to create the "
                "predictors used in training the forecaster."
            )
        
        self.lags, self.lags_names, self.max_lag = initialize_lags(type(self).__name__, lags)
        self.window_size = max(
            [ws for ws in [self.max_lag, self.max_size_window_features] 
             if ws is not None]
        )
        if self.differentiation is not None:
            self.window_size += self.differentiation
            self.differentiator.set_params(window_size=self.window_size)

    def set_window_features(
        self, 
        window_features: object | list[object] | None = None
    ) -> None:
        """
        Set new value to the attribute `window_features`. Attributes 
        `max_size_window_features`, `window_features_names`, 
        `window_features_class_names` and `window_size` are also updated.
        
        Parameters
        ----------
        window_features : object, list, default None
            Instance or list of instances used to create window features. Window features
            are created from the original time series and are included as predictors.

        Returns
        -------
        None
        
        """

        if window_features is None and self.lags is None:
            raise ValueError(
                "At least one of the arguments `lags` or `window_features` "
                "must be different from None. This is required to create the "
                "predictors used in training the forecaster."
            )
        
        self.window_features, self.window_features_names, self.max_size_window_features = (
            initialize_window_features(window_features)
        )
        self.window_features_class_names = None
        if window_features is not None:
            self.window_features_class_names = [
                type(wf).__name__ for wf in self.window_features
            ] 
        self.window_size = max(
            [ws for ws in [self.max_lag, self.max_size_window_features] 
             if ws is not None]
        )
        if self.differentiation is not None:
            self.window_size += self.differentiation   
            self.differentiator.set_params(window_size=self.window_size)

    def set_out_sample_residuals(
        self,
        y_true: dict[int, np.ndarray | pd.Series],
        y_pred: dict[int, np.ndarray | pd.Series],
        append: bool = False,
        random_state: int = 36987
    ) -> None:
        """
        Set new values to the attribute `out_sample_residuals_`. Out of sample
        residuals are meant to be calculated using observations that did not
        participate in the training process. `y_true` and `y_pred` are expected
        to be in the original scale of the time series. Residuals are calculated
        as `y_true` - `y_pred`, after applying the necessary transformations and
        differentiations if the forecaster includes them (`self.transformer_y`
        and `self.differentiation`).

        A total of 10_000 residuals are stored in the attribute `out_sample_residuals_`.
        If the number of residuals is greater than 10_000, a random sample of
        10_000 residuals is stored.
        
        Parameters
        ----------
        y_true : dict
            Dictionary of numpy ndarrays or pandas Series with the true values of
            the time series for each model in the form {step: y_true}.
        y_pred : dict
            Dictionary of numpy ndarrays or pandas Series with the predicted values
            of the time series for each model in the form {step: y_pred}.
        append : bool, default False
            If `True`, new residuals are added to the once already stored in the
            attribute `out_sample_residuals_`. If after appending the new residuals,
            the limit of 10_000 samples is exceeded, a random sample of 10_000 is
            kept.
        random_state : int, default 36987
            Sets a seed to the random sampling for reproducible output.

        Returns
        -------
        None

        """

        if not self.is_fitted:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `set_out_sample_residuals()`."
            )

        if not isinstance(y_true, dict):
            raise TypeError(
                f"`y_true` must be a dictionary of numpy ndarrays or pandas Series. "
                f"Got {type(y_true)}."
            )

        if not isinstance(y_pred, dict):
            raise TypeError(
                f"`y_pred` must be a dictionary of numpy ndarrays or pandas Series. "
                f"Got {type(y_pred)}."
            )
        
        if not set(y_true.keys()) == set(y_pred.keys()):
            raise ValueError(
                f"`y_true` and `y_pred` must have the same keys. "
                f"Got {set(y_true.keys())} and {set(y_pred.keys())}."
            )
        
        for k in y_true.keys():
            if not isinstance(y_true[k], (np.ndarray, pd.Series)):
                raise TypeError(
                    f"Values of `y_true` must be numpy ndarrays or pandas Series. "
                    f"Got {type(y_true[k])} for step {k}."
                )
            if not isinstance(y_pred[k], (np.ndarray, pd.Series)):
                raise TypeError(
                    f"Values of `y_pred` must be numpy ndarrays or pandas Series. "
                    f"Got {type(y_pred[k])} for step {k}."
                )
            if len(y_true[k]) != len(y_pred[k]):
                raise ValueError(
                    f"`y_true` and `y_pred` must have the same length. "
                    f"Got {len(y_true[k])} and {len(y_pred[k])} for step {k}."
                )
            if isinstance(y_true[k], pd.Series) and isinstance(y_pred[k], pd.Series):
                if not y_true[k].index.equals(y_pred[k].index):
                    raise ValueError(
                        f"When containing pandas Series, elements in `y_true` and "
                        f"`y_pred` must have the same index. Error in step {k}."
                    )
        
        steps_to_update = set(range(1, self.steps + 1)).intersection(set(y_pred.keys()))
        if not steps_to_update:
            raise ValueError(
                "Provided keys in `y_pred` and `y_true` do not match any step. "
                "Residuals cannot be updated."
            )
        
        y_true = deepcopy(y_true)
        y_pred = deepcopy(y_pred)
        if self.differentiation is not None:
            differentiator = copy(self.differentiator)
            differentiator.set_params(window_size=None)
        
        residuals = {}
        for step in steps_to_update:
            if isinstance(y_true[step], pd.Series):
                y_true[step] = y_true[step].to_numpy()
            if isinstance(y_pred[step], pd.Series):
                y_pred[step] = y_pred[step].to_numpy()
            if self.transformer_y:
                y_true[step] = transform_numpy(
                                   array             = y_true[step],
                                   transformer       = self.transformer_y,
                                   fit               = False,
                                   inverse_transform = False
                               )
                y_pred[step] = transform_numpy(
                                   array             = y_pred[step],
                                   transformer       = self.transformer_y,
                                   fit               = False,
                                   inverse_transform = False
                               )
            if self.differentiation is not None:
                y_true[step] = differentiator.fit_transform(y_true[step])[self.differentiation:]
                y_pred[step] = differentiator.fit_transform(y_pred[step])[self.differentiation:]

            residuals[step] = y_true[step] - y_pred[step]

        y_true = np.concatenate(list(y_true.values()))
        y_pred = np.concatenate(list(y_pred.values()))
        data = pd.DataFrame(
            {'prediction': y_pred, 'residuals': y_true - y_pred}
        ).dropna()
        y_pred = data['prediction'].to_numpy()
        residuals_all_steps = data['residuals'].to_numpy()

        data['bin'] = self.binner.transform(y_pred).astype(int)
        residuals_by_bin = data.groupby('bin')['residuals'].apply(np.array).to_dict()
        
        out_sample_residuals = (
            {step: None for step in range(1, self.steps + 1)}
            if self.out_sample_residuals_ is None
            else self.out_sample_residuals_
        )

        rng = np.random.default_rng(seed=random_state)
        for step, residuals_step in residuals.items():
            if append and out_sample_residuals[step] is not None:
                out_sample_residuals_step = np.concatenate(
                    [out_sample_residuals[step], residuals_step]
                )
            else:
                out_sample_residuals_step = residuals_step
            
            if len(out_sample_residuals_step) > 10_000:
                out_sample_residuals_step = rng.choice(
                    out_sample_residuals_step, size=10_000, replace=False
                )
            
            out_sample_residuals[step] = out_sample_residuals_step

        out_sample_residuals_by_bin = (
            {} 
            if self.out_sample_residuals_by_bin_ is None
            else self.out_sample_residuals_by_bin_
        )
        if append:
            for k, v in residuals_by_bin.items():
                if k in out_sample_residuals_by_bin:
                    out_sample_residuals_by_bin[k] = np.concatenate(
                        (out_sample_residuals_by_bin[k], v)
                    )
                else:
                    out_sample_residuals_by_bin[k] = v
        else:
            out_sample_residuals_by_bin = residuals_by_bin

        max_samples = 10_000 // self.binner.n_bins_
        for k, v in out_sample_residuals_by_bin.items():
            if len(v) > max_samples:
                sample = rng.choice(a=v, size=max_samples, replace=False)
                out_sample_residuals_by_bin[k] = sample

        in_sample_residuals_by_bin = (
            {}
            if self.in_sample_residuals_by_bin_ is None
            else self.in_sample_residuals_by_bin_
        )
        for k in in_sample_residuals_by_bin.keys():
            if k not in out_sample_residuals_by_bin:
                out_sample_residuals_by_bin[k] = np.array([])

        empty_bins = [
            k for k, v in out_sample_residuals_by_bin.items() 
            if v.size == 0
        ]
        if empty_bins:
            warnings.warn(
                f"The following bins have no out of sample residuals: {empty_bins}. "
                f"No predicted values fall in the interval "
                f"{[self.binner_intervals_[bin] for bin in empty_bins]}. "
                f"Empty bins will be filled with a random sample of residuals.",
                ResidualsUsageWarning
            )
            for k in empty_bins:
                out_sample_residuals_by_bin[k] = rng.choice(
                    a       = residuals_all_steps,
                    size    = min(max_samples, len(residuals_all_steps)),
                    replace = False
                )

        self.out_sample_residuals_ = out_sample_residuals
        self.out_sample_residuals_by_bin_ = out_sample_residuals_by_bin

    def get_feature_importances(
        self, 
        step: int,
        sort_importance: bool = True
    ) -> pd.DataFrame:
        """
        Return feature importance of the model stored in the forecaster for a
        specific step. Since a separate model is created for each forecast time
        step, it is necessary to select the model from which retrieve information.
        Only valid when regressor stores internally the feature importances in
        the attribute `feature_importances_` or `coef_`. Otherwise, it returns  
        `None`.

        Parameters
        ----------
        step : int
            Model from which retrieve information (a separate model is created 
            for each forecast time step). First step is 1.
        sort_importance: bool, default True
            If `True`, sorts the feature importances in descending order.

        Returns
        -------
        feature_importances : pandas DataFrame
            Feature importances associated with each predictor.
        
        """

        if not isinstance(step, int):
            raise TypeError(f"`step` must be an integer. Got {type(step)}.")
        
        if not self.is_fitted:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `get_feature_importances()`."
            )

        if (step < 1) or (step > self.steps):
            raise ValueError(
                f"The step must have a value from 1 to the maximum number of steps "
                f"({self.steps}). Got {step}."
            )

        if isinstance(self.regressor, Pipeline):
            estimator = self.regressors_[step][-1]
        else:
            estimator = self.regressors_[step]

        n_lags = len(self.lags) if self.lags is not None else 0
        n_window_features = (
            len(self.window_features_names) if self.window_features is not None else 0
        )
        idx_columns_autoreg = np.arange(n_lags + n_window_features)
        if not self.exog_in_:
            idx_columns = idx_columns_autoreg
        else:
            n_exog = len(self.X_train_direct_exog_names_out_) / self.steps
            idx_columns_exog = (
                np.arange((step - 1) * n_exog, (step) * n_exog) + idx_columns_autoreg[-1] + 1
            )
            idx_columns = np.concatenate((idx_columns_autoreg, idx_columns_exog))
        
        idx_columns = [int(x) for x in idx_columns]  # Required since numpy 2.0
        feature_names = [self.X_train_features_names_out_[i].replace(f"_step_{step}", "") 
                         for i in idx_columns]

        if hasattr(estimator, 'feature_importances_'):
            feature_importances = estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            feature_importances = estimator.coef_
        else:
            warnings.warn(
                f"Impossible to access feature importances for regressor of type "
                f"{type(estimator)}. This method is only valid when the "
                f"regressor stores internally the feature importances in the "
                f"attribute `feature_importances_` or `coef_`."
            )
            feature_importances = None

        if feature_importances is not None:
            feature_importances = pd.DataFrame({
                                      'feature': feature_names,
                                      'importance': feature_importances
                                  })
            if sort_importance:
                feature_importances = feature_importances.sort_values(
                                          by='importance', ascending=False
                                      )

        return feature_importances
