################################################################################
#                               ForecasterDirect                               #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Union, Tuple, Optional, Callable, Any
import warnings
import sys
import numpy as np
import pandas as pd
import inspect
from copy import copy
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from joblib import Parallel, delayed, cpu_count

import skforecast
from ..base import ForecasterBase
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
from ..preprocessing import TimeSeriesDifferentiator


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
    lags : int, list, numpy ndarray, range, default `None`
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
    
        - `int`: include lags from 1 to `lags` (included).
        - `list`, `1d numpy ndarray` or `range`: include only lags present in 
        `lags`, all elements must be int.
        - `None`: no lags are included as predictors. 
    window_features : object, list, default `None`
        Instance or list of instances used to create window features. Window features
        are created from the original time series and are included as predictors.
    transformer_y : object transformer (preprocessor), default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
        The transformation is applied to `y` before training the forecaster. 
    transformer_exog : object transformer (preprocessor), default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    weight_func : Callable, default `None`
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        Ignored if `regressor` does not have the argument `sample_weight` in its `fit`
        method. The resulting `sample_weight` cannot have negative values.
    differentiation : int, default `None`
        Order of differencing applied to the time series before training the forecaster.
        If `None`, no differencing is applied. The order of differentiation is the number
        of times the differencing operation is applied to a time series. Differencing
        involves computing the differences between consecutive data points in the series.
        Differentiation is reversed in the output of `predict()` and `predict_interval()`.
        **WARNING: This argument is newly introduced and requires special attention. It
        is still experimental and may undergo changes.**
    fit_kwargs : dict, default `None`
        Additional arguments to be passed to the `fit` method of the regressor.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_fit_forecaster.
    forecaster_id : str, int, default `None`
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
        Residuals of the models when predicting training data. Only stored up to
        1000 values per model in the form `{step: residuals}`. If `transformer_y` 
        is not `None`, residuals are stored in the transformed scale.
    out_sample_residuals_ : dict
        Residuals of the models when predicting non training data. Only stored
        up to 1000 values per model in the form `{step: residuals}`. If `transformer_y` 
        is not `None`, residuals are assumed to be in the transformed scale. Use 
        `set_out_sample_residuals()` method to set values.
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
    binner : Ignored
        Not used, present here for API consistency by convention.
    binner_intervals_ : Ignored
        Not used, present here for API consistency by convention.
    binner_kwargs : Ignored
        Not used, present here for API consistency by convention.
    in_sample_residuals_by_bin_ : Ignored
        Not used, present here for API consistency by convention.
    out_sample_residuals_by_bin_ : Ignored
        Not used, present here for API consistency by convention.

    Notes
    -----
    A separate model is created for each forecasting time step. It is important to
    note that all models share the same parameter and hyperparameter configuration.
     
    """

    def __init__(
        self,
        regressor: object,
        steps: int,
        lags: Optional[Union[int, list, np.ndarray, range]] = None,
        window_features: Optional[Union[object, list]] = None,
        transformer_y: Optional[object] = None,
        transformer_exog: Optional[object] = None,
        weight_func: Optional[Callable] = None,
        differentiation: Optional[int] = None,
        fit_kwargs: Optional[dict] = None,
        n_jobs: Union[int, str] = 'auto',
        forecaster_id: Optional[Union[str, int]] = None
    ) -> None:
        
        self.regressor                          = copy(regressor)
        self.steps                              = steps
        self.transformer_y                      = transformer_y
        self.transformer_exog                   = transformer_exog
        self.weight_func                        = weight_func
        self.source_code_weight_func            = None
        self.differentiation                    = differentiation
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
        self.creation_date                      = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.is_fitted                          = False
        self.fit_date                           = None
        self.skforecast_version                 = skforecast.__version__
        self.python_version                     = sys.version.split(" ")[0]
        self.forecaster_id                      = forecaster_id
        self.binner                             = None
        self.binner_intervals_                  = None
        self.binner_kwargs                      = None
        self.in_sample_residuals_by_bin_        = None
        self.out_sample_residuals_by_bin_       = None

        if not isinstance(steps, int):
            raise TypeError(
                (f"`steps` argument must be an int greater than or equal to 1. "
                 f"Got {type(steps)}.")
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
                ("At least one of the arguments `lags` or `window_features` "
                 "must be different from None. This is required to create the "
                 "predictors used in training the forecaster.")
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

        if self.differentiation is not None:
            if not isinstance(differentiation, int) or differentiation < 1:
                raise ValueError(
                    (f"Argument `differentiation` must be an integer equal to or "
                     f"greater than 1. Got {differentiation}.")
                )
            self.window_size += self.differentiation
            self.differentiator = TimeSeriesDifferentiator(order=self.differentiation)

        self.weight_func, self.source_code_weight_func, _ = initialize_weights(
            forecaster_name = type(self).__name__, 
            regressor       = regressor, 
            weight_func     = weight_func, 
            series_weights  = None
        )

        self.fit_kwargs = check_select_fit_kwargs(
                              regressor  = regressor,
                              fit_kwargs = fit_kwargs
                          )

        self.in_sample_residuals_ = {step: None for step in range(1, steps + 1)}
        self.out_sample_residuals_ = None

        if n_jobs == 'auto':
            self.n_jobs = select_n_jobs_fit_forecaster(
                              forecaster_name = type(self).__name__,
                              regressor_name  = type(self.regressor).__name__,
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
            f"Regressor: {self.regressor} \n"
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
                    <li><strong>Regressor:</strong> {self.regressor}</li>
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
                <a href="https://skforecast.org/{skforecast.__version__}/api/forecasterdirect#forecasterdirect.html">&#128712 <strong>API Reference</strong></a>
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
        train_index: Optional[pd.Index] = None
    ) -> Tuple[Optional[Union[np.ndarray, pd.DataFrame]], np.ndarray]:
        """
        Create the lagged values and their target variable from a time series.
        
        Note that the returned matrix `X_data` contains the lag 1 in the first 
        column, the lag 2 in the in the second column and so on.
        
        Parameters
        ----------
        y : numpy ndarray
            Training time series values.
        X_as_pandas : bool, default `False`
            If `True`, the returned matrix `X_data` is a pandas DataFrame.
        train_index : pandas Index, default `None`
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
    ) -> Tuple[list, list]:
        """
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        train_index : pandas Index
            Index of the training data. It is used to create the pandas DataFrame
            `X_train_window_features` when `X_as_pandas` is `True`.
        X_as_pandas : bool, default `False`
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
                    (f"The method `transform_batch` of {type(wf).__name__} "
                     f"must return a pandas DataFrame.")
                )
            X_train_wf = X_train_wf.iloc[-len_train_index:]
            if not len(X_train_wf) == len_train_index:
                raise ValueError(
                    (f"The method `transform_batch` of {type(wf).__name__} "
                     f"must return a DataFrame with the same number of rows as "
                     f"the input time series - (`window_size` + (`steps` - 1)): {len_train_index}.")
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
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None
    ) -> Tuple[pd.DataFrame, dict, list, list, list, dict]:
        """
        Create training matrices from univariate time series and exogenous
        variables. The resulting matrices contain the target variable and 
        predictors needed to train all the regressors (one per step).
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        exog : pandas Series, pandas DataFrame, default `None`
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
            # TODO: Check if this check can be checked vs y_train. As happened
            # in base on data (more data from y than exog, but can be aligned
            # because of the window_size.
            if len(exog) != len(y):
                raise ValueError(
                    (f"`exog` must have same number of samples as `y`. "
                     f"length `exog`: ({len(exog)}), length `y`: ({len(y)})")
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
            if not (exog_index[:len(y_index)] == y_index).all():
                raise ValueError(
                    ("Different index for `y` and `exog`. They must be equal "
                     "to ensure the correct alignment of values.")
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
            # The first `self.window_size` positions have to be removed from exog
            # since they are not in X_train.
            X_train_exog_names_out_ = exog.columns.to_list()
            if X_as_pandas:
                exog_to_train, X_train_direct_exog_names_out_ = exog_to_direct(
                    exog=exog, steps=self.steps
                )
                exog_to_train = exog_to_train.iloc[-len_train_index:, :]
                exog_to_train.index = train_index
            else:
                exog_to_train, X_train_direct_exog_names_out_ = exog_to_direct_numpy(
                    exog=exog, steps=self.steps
                )
                exog_to_train = exog_to_train[-len_train_index:, :]

            # NOTE: Need here for filter_train_X_y_for_step to work without fitting
            self.X_train_direct_exog_names_out_ = X_train_direct_exog_names_out_

            X_train_features_names_out_.extend(self.X_train_direct_exog_names_out_)
            X_train.append(exog_to_train)
        
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
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Create training matrices from univariate time series and exogenous
        variables. The resulting matrices contain the target variable and 
        predictors needed to train all the regressors (one per step).
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        exog : pandas Series, pandas DataFrame, default `None`
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
        y_train: dict,
        remove_suffix: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select the columns needed to train a forecaster for a specific step.  
        The input matrices should be created using `create_train_X_y` method. 
        This method updates the index of `X_train` to the corresponding one 
        according to `y_train`. If `remove_suffix=True` the suffix "_step_i" 
        will be removed from the column names. 

        Parameters
        ----------
        step : int
            Step for which columns must be selected selected. Starts at 1.
        X_train : pandas DataFrame
            Dataframe created with the `create_train_X_y` method, first return.
        y_train : dict
            Dict created with the `create_train_X_y` method, second return.
        remove_suffix : bool, default `False`
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
                (f"Invalid value `step`. For this forecaster, minimum value is 1 "
                 f"and the maximum step is {self.steps}.")
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
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None
    ) -> Tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
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
        exog : pandas Series, pandas DataFrame, default `None`
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
        Crate weights for each observation according to the forecaster's attribute
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
                    ("The resulting `sample_weight` cannot be normalized because "
                     "the sum of the weights is zero.")
                )

        return sample_weight

    def fit(
        self,
        y: pd.Series,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
        store_last_window: bool = True,
        store_in_sample_residuals: bool = True
    ) -> None:
        """
        Training Forecaster.

        Additional arguments to be passed to the `fit` method of the regressor 
        can be added with the `fit_kwargs` argument when initializing the forecaster.

        Parameters
        ----------
        y : pandas Series
            Training time series.
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and their indexes must be aligned so
            that y[i] is regressed on exog[i].
        store_last_window : bool, default `True`
            Whether or not to store the last window (`last_window_`) of training data.
        store_in_sample_residuals : bool, default `True`
            If `True`, in-sample residuals will be stored in the forecaster object
            after fitting (`in_sample_residuals_` attribute).

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

        def fit_forecaster(regressor, X_train, y_train, step, store_in_sample_residuals):
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
                after fitting (`in_sample_residuals_` attribute).
            
            Returns
            -------
            Tuple with the step, fitted regressor and in-sample residuals.

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
                residuals = (
                    (y_train_step - regressor.predict(X_train_step))
                ).to_numpy()

                if len(residuals) > 1000:
                    # Only up to 1000 residuals are stored
                    rng = np.random.default_rng(seed=123)
                    residuals = rng.choice(
                                    a       = residuals, 
                                    size    = 1000, 
                                    replace = False
                                )
            else:
                residuals = None

            return step, regressor, residuals

        results_fit = (
            Parallel(n_jobs=self.n_jobs)
            (delayed(fit_forecaster)
            (
                regressor                 = copy(self.regressor),
                X_train                   = X_train,
                y_train                   = y_train,
                step                      = step,
                store_in_sample_residuals = store_in_sample_residuals
            )
            for step in range(1, self.steps + 1))
        )

        self.regressors_ = {step: regressor 
                            for step, regressor, _ in results_fit}

        if store_in_sample_residuals:
            self.in_sample_residuals_ = {step: residuals 
                                         for step, _, residuals in results_fit}
        
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

    def _create_predict_inputs(
        self,
        steps: Optional[Union[int, list]] = None,
        last_window: Optional[Union[pd.Series, pd.DataFrame]] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
        check_inputs: bool = True
    ) -> Tuple[list, list, list, pd.Index]:
        """
        Create the inputs needed for the prediction process.
        
        Parameters
        ----------
        steps : int, list, None, default `None`
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as were defined at 
            initialization.
        last_window : pandas Series, pandas DataFrame, default `None`
            Series values used to create the predictors (lags) needed to 
            predict `steps`.
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.
        check_inputs : bool, default `True`
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
        steps: Optional[Union[int, list]] = None,
        last_window: Optional[Union[pd.Series, pd.DataFrame]] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Create the predictors needed to predict `steps` ahead.
        
        Parameters
        ----------
        steps : int, list, None, default `None`
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as were defined at 
            initialization.
        last_window : pandas Series, pandas DataFrame, default `None`
            Series values used to create the predictors (lags) needed to 
            predict `steps`.
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.

        Returns
        -------
        X_predict : pandas DataFrame
            Pandas DataFrame with the predictors for each step. The index 
            is the same as the prediction index.
        
        """

        Xs, Xs_col_names, steps, prediction_index = self._create_predict_inputs(
            steps=steps, last_window=last_window, exog=exog
        )

        X_predict = pd.DataFrame(
                        data    = np.concatenate(Xs, axis=0), 
                        columns = Xs_col_names, 
                        index   = prediction_index
                    )

        return X_predict

    def predict(
        self,
        steps: Optional[Union[int, list]] = None,
        last_window: Optional[Union[pd.Series, pd.DataFrame]] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
        check_inputs: bool = True
    ) -> pd.Series:
        """
        Predict n steps ahead.

        Parameters
        ----------
        steps : int, list, None, default `None`
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as were defined at 
            initialization.
        last_window : pandas Series, pandas DataFrame, default `None`
            Series values used to create the predictors (lags) needed to 
            predict `steps`.
            If `last_window = None`, the values stored in` self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.
        check_inputs : bool, default `True`
            If `True`, the input is checked for possible warnings and errors 
            with the `check_predict_input` function. This argument is created 
            for internal use and is not recommended to be changed.

        Returns
        -------
        predictions : pandas Series
            Predicted values.

        """

        Xs, _, steps, prediction_index = self._create_predict_inputs(
            steps=steps, last_window=last_window, exog=exog, check_inputs=check_inputs
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
        steps: Optional[Union[int, list]] = None,
        last_window: Optional[pd.Series] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
        n_boot: int = 250,
        random_state: int = 123,
        use_in_sample_residuals: bool = True,
        use_binned_residuals: Any = None
    ) -> pd.DataFrame:
        """
        Generate multiple forecasting predictions using a bootstrapping process. 
        By sampling from a collection of past observed errors (the residuals),
        each iteration of bootstrapping generates a different set of predictions. 
        See the Notes section for more information. 
        
        Parameters
        ----------
        steps : int, list, None, default `None`
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as were defined at 
            initialization.
        last_window : pandas Series, default `None`
            Series values used to create the predictors (lags) needed to 
            predict `steps`.
            If `last_window = None`, the values stored in` self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.
        n_boot : int, default `250`
            Number of bootstrapping iterations used to estimate predictions.
        random_state : int, default `123`
            Sets a seed to the random generator, so that boot predictions are always 
            deterministic.
        use_in_sample_residuals : bool, default `True`
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. If `False`, out of sample 
            residuals are used. In the latter case, the user should have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).
        use_binned_residuals : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        boot_predictions : pandas DataFrame
            Predictions generated by bootstrapping.
            Shape: (steps, n_boot)

        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp3/prediction-intervals.html#prediction-intervals-from-bootstrapped-residuals
        Forecasting: Principles and Practice (3nd ed) Rob J Hyndman and George Athanasopoulos.

        """

        if self.is_fitted:
            
            steps = prepare_steps_direct(
                        steps    = steps,
                        max_step = self.steps
                    )
            
            if use_in_sample_residuals:
                if not set(steps).issubset(set(self.in_sample_residuals_.keys())):
                    raise ValueError(
                        (f"Not `forecaster.in_sample_residuals_` for steps: "
                         f"{set(steps) - set(self.in_sample_residuals_.keys())}.")
                    )
                residuals = self.in_sample_residuals_
            else:
                if self.out_sample_residuals_ is None:
                    raise ValueError(
                        ("`forecaster.out_sample_residuals_` is `None`. Use "
                         "`use_in_sample_residuals=True` or the "
                         "`set_out_sample_residuals()` method before predicting.")
                    )
                else:
                    if not set(steps).issubset(set(self.out_sample_residuals_.keys())):
                        raise ValueError(
                            (f"No `forecaster.out_sample_residuals_` for steps: "
                             f"{set(steps) - set(self.out_sample_residuals_.keys())}. "
                             f"Use method `set_out_sample_residuals()`.")
                        )
                residuals = self.out_sample_residuals_
            
            check_residuals = (
                "forecaster.in_sample_residuals_" if use_in_sample_residuals
                else "forecaster.out_sample_residuals_"
            )
            for step in steps:
                print(step)
                if residuals[step] is None:
                    raise ValueError(
                        (f"forecaster residuals for step {step} are `None`. "
                         f"Check {check_residuals}.")
                    )
                elif any(x is None or np.isnan(x) for x in residuals[step]):
                    raise ValueError(
                        (f"forecaster residuals for step {step} contains `None` values. "
                         f"Check {check_residuals}.")
                    )

        Xs, _, steps, prediction_index = self._create_predict_inputs(
            steps=steps, last_window=last_window, exog=exog
        )

        # NOTE: Predictions must be transformed and differenced before adding residuals
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
            sampled_residuals = residuals[step][
                rng.integers(low=0, high=len(residuals[step]), size=n_boot)
            ]
            boot_predictions[i, :] = boot_predictions[i, :] + sampled_residuals

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

    def predict_interval(
        self,
        steps: Optional[Union[int, list]] = None,
        last_window: Optional[pd.Series] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
        interval: list = [5, 95],
        n_boot: int = 250,
        random_state: int = 123,
        use_in_sample_residuals: bool = True,
        use_binned_residuals: Any = None
    ) -> pd.DataFrame:
        """
        Bootstrapping based predicted intervals.
        Both predictions and intervals are returned.
        
        Parameters
        ----------
        steps : int, list, None, default `None`
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as were defined at 
            initialization.
        last_window : pandas Series, default `None`
            Series values used to create the predictors (lags) needed to 
            predict `steps`.
            If `last_window = None`, the values stored in` self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.
        interval : list, default `[5, 95]`
            Confidence of the prediction interval estimated. Sequence of 
            percentiles to compute, which must be between 0 and 100 inclusive. 
            For example, interval of 95% should be as `interval = [2.5, 97.5]`.
        n_boot : int, default `250`
            Number of bootstrapping iterations used to estimate predictions.
        random_state : int, default `123`
            Sets a seed to the random generator, so that boot predictions are always 
            deterministic.
        use_in_sample_residuals : bool, default `True`
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. If `False`, out of sample 
            residuals are used. In the latter case, the user should have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).
        use_binned_residuals : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        predictions : pandas DataFrame
            Values predicted by the forecaster and their estimated interval.

            - pred: predictions.
            - lower_bound: lower bound of the interval.
            - upper_bound: upper bound of the interval.

        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp2/prediction-intervals.html
        Forecasting: Principles and Practice (2nd ed) Rob J Hyndman and
        George Athanasopoulos.
        
        """

        check_interval(interval=interval)

        boot_predictions = self.predict_bootstrapping(
                               steps                   = steps,
                               last_window             = last_window,
                               exog                    = exog,
                               n_boot                  = n_boot,
                               random_state            = random_state,
                               use_in_sample_residuals = use_in_sample_residuals
                           )

        predictions = self.predict(
                          steps        = steps,
                          last_window  = last_window,
                          exog         = exog,
                          check_inputs = False
                      )

        interval = np.array(interval) / 100
        predictions_interval = boot_predictions.quantile(q=interval, axis=1).transpose()
        predictions_interval.columns = ['lower_bound', 'upper_bound']
        predictions = pd.concat((predictions, predictions_interval), axis=1)

        return predictions

    def predict_quantiles(
        self,
        steps: Optional[Union[int, list]] = None,
        last_window: Optional[pd.Series] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
        quantiles: list = [0.05, 0.5, 0.95],
        n_boot: int = 250,
        random_state: int = 123,
        use_in_sample_residuals: bool = True,
        use_binned_residuals: Any = None
    ) -> pd.DataFrame:
        """
        Bootstrapping based predicted quantiles.
        
        Parameters
        ----------
        steps : int, list, None, default `None`
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as were defined at 
            initialization.
        last_window : pandas Series, default `None`
            Series values used to create the predictors (lags) needed to 
            predict `steps`.
            If `last_window = None`, the values stored in` self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.
        quantiles : list, default `[0.05, 0.5, 0.95]`
            Sequence of quantiles to compute, which must be between 0 and 1 
            inclusive. For example, quantiles of 0.05, 0.5 and 0.95 should be as 
            `quantiles = [0.05, 0.5, 0.95]`.
        n_boot : int, default `250`
            Number of bootstrapping iterations used to estimate quantiles.
        random_state : int, default `123`
            Sets a seed to the random generator, so that boot quantiles are always 
            deterministic.
        use_in_sample_residuals : bool, default `True`
            If `True`, residuals from the training data are used as proxy of
            prediction error to create prediction quantiles. If `False`, out of
            sample residuals are used. In the latter case, the user should have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).
        use_binned_residuals : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        predictions : pandas DataFrame
            Quantiles predicted by the forecaster.

        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp2/prediction-intervals.html
        Forecasting: Principles and Practice (2nd ed) Rob J Hyndman and
        George Athanasopoulos.
        
        """
        
        check_interval(quantiles=quantiles)

        boot_predictions = self.predict_bootstrapping(
                               steps                   = steps,
                               last_window             = last_window,
                               exog                    = exog,
                               n_boot                  = n_boot,
                               random_state            = random_state,
                               use_in_sample_residuals = use_in_sample_residuals
                           )

        predictions = boot_predictions.quantile(q=quantiles, axis=1).transpose()
        predictions.columns = [f'q_{q}' for q in quantiles]

        return predictions
    
    def predict_dist(
        self,
        distribution: object,
        steps: Optional[Union[int, list]] = None,
        last_window: Optional[pd.Series] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
        n_boot: int = 250,
        random_state: int = 123,
        use_in_sample_residuals: bool = True,
        use_binned_residuals: Any = None
    ) -> pd.DataFrame:
        """
        Fit a given probability distribution for each step. After generating 
        multiple forecasting predictions through a bootstrapping process, each 
        step is fitted to the given distribution.
        
        Parameters
        ----------
        distribution : Object
            A distribution object from scipy.stats.
        steps : int, list, None, default `None`
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined when initializing the forecaster. Starts at 1.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as were defined at 
            initialization.
        last_window : pandas Series, default `None`
            Series values used to create the predictors (lags) needed to 
            predict `steps`.
            If `last_window = None`, the values stored in` self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.
        n_boot : int, default `250`
            Number of bootstrapping iterations used to estimate predictions.
        random_state : int, default `123`
            Sets a seed to the random generator, so that boot predictions are always 
            deterministic.
        use_in_sample_residuals : bool, default `True`
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. If `False`, out of sample 
            residuals are used. In the latter case, the user should have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).
        use_binned_residuals : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        predictions : pandas DataFrame
            Distribution parameters estimated for each step.

        """
        
        boot_samples = self.predict_bootstrapping(
                           steps                   = steps,
                           last_window             = last_window,
                           exog                    = exog,
                           n_boot                  = n_boot,
                           random_state            = random_state,
                           use_in_sample_residuals = use_in_sample_residuals
                       )       

        param_names = [p for p in inspect.signature(distribution._pdf).parameters
                       if not p == 'x'] + ["loc", "scale"]
        param_values = np.apply_along_axis(
                           lambda x: distribution.fit(x),
                           axis = 1,
                           arr  = boot_samples
                       )
        predictions = pd.DataFrame(
                          data    = param_values,
                          columns = param_names,
                          index   = boot_samples.index
                      )

        return predictions

    def set_params(
        self, 
        params: dict
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
        fit_kwargs: dict
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
        lags: Optional[Union[int, list, np.ndarray, range]] = None
    ) -> None:
        """
        Set new value to the attribute `lags`. Attributes `lags_names`, 
        `max_lag` and `window_size` are also updated.
        
        Parameters
        ----------
        lags : int, list, numpy ndarray, range, default `None`
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
                ("At least one of the arguments `lags` or `window_features` "
                 "must be different from None. This is required to create the "
                 "predictors used in training the forecaster.")
            )
        
        self.lags, self.lags_names, self.max_lag = initialize_lags(type(self).__name__, lags)
        self.window_size = max(
            [ws for ws in [self.max_lag, self.max_size_window_features] 
             if ws is not None]
        )
        if self.differentiation is not None:
            self.window_size += self.differentiation

    def set_window_features(
        self, 
        window_features: Optional[Union[object, list]] = None
    ) -> None:
        """
        Set new value to the attribute `window_features`. Attributes 
        `max_size_window_features`, `window_features_names`, 
        `window_features_class_names` and `window_size` are also updated.
        
        Parameters
        ----------
        window_features : object, list, default `None`
            Instance or list of instances used to create window features. Window features
            are created from the original time series and are included as predictors.

        Returns
        -------
        None
        
        """

        if window_features is None and self.lags is None:
            raise ValueError(
                ("At least one of the arguments `lags` or `window_features` "
                 "must be different from None. This is required to create the "
                 "predictors used in training the forecaster.")
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

    def set_out_sample_residuals(
        self,
        y_true: dict,
        y_pred: dict,
        append: bool = False,
        random_state: int = 36987
    ) -> None:
        """
        Set new values to the attribute `out_sample_residuals_`. Out of sample
        residuals are meant to be calculated using observations that did not
        participate in the training process.
        
        Parameters
        ----------
        y_true : dict
            Dictionary of numpy ndarrays or pandas series with the true values of
            the time series for each model in the form {step: y_true}.
        y_pred : dict
            Dictionary of numpy ndarrays or pandas series with the predicted values
            of the time series for each model in the form {step: y_pred}.
        append : bool, default `False`
            If `True`, new residuals are added to the once already stored in the
            attribute `out_sample_residuals_`. If after appending the new residuals,
            the limit of 10000 samples is exceeded, a random sample of 10000 is
            kept.
        random_state : int, default `123`
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
                f"`y_true` must be a dictionary of numpy ndarrays or pandas series. "
                f"Got {type(y_true)}."
            )

        if not isinstance(y_pred, dict):
            raise TypeError(
                f"`y_pred` must be a dictionary of numpy ndarrays or pandas series. "
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
                    f"Values of `y_true` must be numpy ndarrays or pandas series. "
                    f"Got {type(y_true[k])}."
                )
            if not isinstance(y_pred[k], (np.ndarray, pd.Series)):
                raise TypeError(
                    f"Values of `y_pred` must be numpy ndarrays or pandas series. "
                    f"Got {type(y_pred[k])}."
                )
            if len(y_true[k]) != len(y_pred[k]):
                raise ValueError(
                    f"{k} must have the same length in `y_true` and `y_pred`. "
                    f"Got {len(y_true[k])} and {len(y_pred[k])}."
                )
            if isinstance(y_true[k], pd.Series) and isinstance(y_pred[k], pd.Series):
                if not y_true[k].index.equals(y_pred[k].index):
                    raise ValueError(
                       "When containing pandas series, elements in `y_true` and "
                        "must have the same index."
                    )
        
        if self.out_sample_residuals_ is None:
            self.out_sample_residuals_ = {
                step: None for step in range(1, self.steps + 1)
            }
        
        steps_to_update = set(range(1, self.steps + 1)).intersection(set(y_pred.keys()))
        if not steps_to_update:
            warnings.warn(
                "Provided keys in `y_pred` and `y_true` do not match any step. "
                "Residuals are not updated."
            )

        residuals = {}
        rng = np.random.default_rng(seed=random_state)
        y_true = y_true.copy()
        y_pred = y_pred.copy()
        if self.differentiation is not None:
            differentiator = copy(self.differentiator)
        for k in steps_to_update:
            if isinstance(y_true[k], pd.Series):
                y_true[k] = y_true[k].to_numpy()
            if isinstance(y_pred[k], pd.Series):
                y_pred[k] = y_pred[k].to_numpy()
            if self.transformer_y:
                y_true[k] = transform_numpy(
                                array             = y_true[k],
                                transformer       = self.transformer_y,
                                fit               = False,
                                inverse_transform = False
                            )
                y_pred[k] = transform_numpy(
                                array             = y_pred[k],
                                transformer       = self.transformer_y,
                                fit               = False,
                                inverse_transform = False
                            )
            if self.differentiation is not None:
                y_true[k] = differentiator.fit_transform(y_true[k])[self.differentiation:]
                y_pred[k] = differentiator.fit_transform(y_pred[k])[self.differentiation:]

            residuals[k] = y_true[k] - y_pred[k]

        for key, value in residuals.items():
            if append and self.out_sample_residuals_[key] is not None:
                value = np.concatenate((
                            self.out_sample_residuals_[key],
                            value
                        ))
            if len(value) > 10000:
                value = rng.choice(value, size=10000, replace=False)
            self.out_sample_residuals_[key] = value

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
        sort_importance: bool, default `True`
            If `True`, sorts the feature importances in descending order.

        Returns
        -------
        feature_importances : pandas DataFrame
            Feature importances associated with each predictor.
        
        """

        if not isinstance(step, int):
            raise TypeError(
                f"`step` must be an integer. Got {type(step)}."
            )
        
        if not self.is_fitted:
            raise NotFittedError(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `get_feature_importances()`.")
            )

        if (step < 1) or (step > self.steps):
            raise ValueError(
                (f"The step must have a value from 1 to the maximum number of steps "
                 f"({self.steps}). Got {step}.")
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
                (f"Impossible to access feature importances for regressor of type "
                 f"{type(estimator)}. This method is only valid when the "
                 f"regressor stores internally the feature importances in the "
                 f"attribute `feature_importances_` or `coef_`.")
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
