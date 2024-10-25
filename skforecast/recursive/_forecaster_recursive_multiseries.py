################################################################################
#                         ForecasterRecursiveMultiSeries                       #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Union, Tuple, Optional, Callable
import warnings
import sys
import numpy as np
import pandas as pd
from copy import copy
import inspect
import sklearn
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import skforecast
from ..base import ForecasterBase
from ..exceptions import (
    MissingValuesWarning,
    UnknownLevelWarning,
    IgnoredArgumentWarning
)
from ..utils import (
    initialize_lags,
    initialize_window_features,
    initialize_weights,
    initialize_transformer_series,
    check_select_fit_kwargs,
    check_preprocess_series,
    check_preprocess_exog_multiseries,
    align_series_and_exog_multiseries,
    prepare_levels_multiseries,
    preprocess_levels_self_last_window_multiseries,
    prepare_residuals_multiseries,
    get_exog_dtypes,
    check_exog_dtypes,
    check_predict_input,
    check_interval,
    preprocess_last_window,
    expand_index,
    transform_numpy,
    transform_series,
    transform_dataframe,
    set_skforecast_warnings
)
from ..preprocessing import TimeSeriesDifferentiator
from ..model_selection._utils import _extract_data_folds_multiseries


class ForecasterRecursiveMultiSeries(ForecasterBase):
    """
    This class turns any regressor compatible with the scikit-learn API into a
    recursive autoregressive (multi-step) forecaster for multiple series.
    
    Parameters
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
    lags : int, list, numpy ndarray, range, default `None`
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
    
        - `int`: include lags from 1 to `lags` (included).
        - `list`, `1d numpy ndarray` or `range`: include only lags present in 
        `lags`, all elements must be int.
        - `None`: no lags are included as predictors. 
    window_features : object, list, default `None`
        Instance or list of instances used to create window features. Window features
        are created from the original time series and are included as predictors.
    encoding : str, None, default `'ordinal'`
        Encoding used to identify the different series. 
        
        - If `'ordinal'`, a single column is created with integer values from 0 
        to n_series - 1. 
        - If `'ordinal_category'`, a single column is created with integer 
        values from 0 to n_series - 1 and the column is transformed into 
        pandas.category dtype so that it can be used as a categorical variable. 
        - If `'onehot'`, a binary column is created for each series.
        - If None, no column is created to identify the series. Internally, the
        series are identified as an integer from 0 to n_series - 1, but no column
        is created in the training matrices.
        **Changed to 'ordinal' in version 0.14.0**
    transformer_series : transformer (preprocessor), dict, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and 
        inverse_transform. Transformation is applied to each `series` before training 
        the forecaster. ColumnTransformers are not allowed since they do not have 
        inverse_transform method.

        - If single transformer: it is cloned and applied to all series. 
        - If `dict` of transformers: a different transformer can be used for each series.
    transformer_exog : transformer, default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    weight_func : Callable, dict, default `None`
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates. 
        Ignored if `regressor` does not have the argument `sample_weight` in its 
        `fit` method. See Notes section for more details on the use of the weights.

        - If single function: it is applied to all series. 
        - If `dict` {'series_column_name' : Callable}: a different function can be
        used for each series, a weight of 1 is given to all series not present in 
        `weight_func`.
    series_weights : dict, default `None`
        Weights associated with each series {'series_column_name' : float}. It is only
        applied if the `regressor` used accepts `sample_weight` in its `fit` method. 
        See Notes section for more details on the use of the weights.

        - If a `dict` is provided, a weight of 1 is given to all series not present
        in `series_weights`.
        - If `None`, all levels have the same weight.
    differentiation : int, default `None`
        Order of differencing applied to the time series before training the forecaster.
        If `None`, no differencing is applied. The order of differentiation is the number
        of times the differencing operation is applied to a time series. Differencing
        involves computing the differences between consecutive data points in the series.
        Differentiation is reversed in the output of `predict()` and `predict_interval()`.
        **WARNING: This argument is newly introduced and requires special attention. It
        is still experimental and may undergo changes.**
    dropna_from_series : bool, default `False`
        Determine whether NaN detected in the training matrices will be dropped.

        - If `True`, drop NaNs in X_train and same rows in y_train.
        - If `False`, leave NaNs in X_train and warn the user.
        **New in version 0.12.0**
    fit_kwargs : dict, default `None`
        Additional arguments to be passed to the `fit` method of the regressor.
    forecaster_id : str, int, default `None`
        Name used as an identifier of the forecaster.
    
    Attributes
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
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
    encoding : str
        Encoding used to identify the different series. 
        
        - If `'ordinal'`, a single column is created with integer values from 0 
        to n_series - 1. 
        - If `'ordinal_category'`, a single column is created with integer 
        values from 0 to n_series - 1 and the column is transformed into 
        pandas.category dtype so that it can be used as a categorical variable. 
        - If `'onehot'`, a binary column is created for each series.
        - If None, no column is created to identify the series. Internally, the
        series are identified as an integer from 0 to n_series - 1, but no column
        is created in the training matrices.
        **Changed to 'ordinal' in version 0.14.0**
    encoder : sklearn.preprocessing
        Scikit-learn preprocessing encoder used to encode the series.
        **New in version 0.12.0**
    encoding_mapping_ : dict
        Mapping of the encoding used to identify the different series.
    transformer_series : transformer (preprocessor), dict
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and 
        inverse_transform. Transformation is applied to each `series` before training 
        the forecaster. ColumnTransformers are not allowed since they do not have 
        inverse_transform method.

        - If single transformer: it is cloned and applied to all series. 
        - If `dict` of transformers: a different transformer can be used for each series.
    transformer_series_ : dict
        Dictionary with the transformer for each series. It is created cloning the 
        objects in `transformer_series` and is used internally to avoid overwriting.
    transformer_exog : transformer (preprocessor)
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    weight_func : Callable, dict
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates. 
        Ignored if `regressor` does not have the argument `sample_weight` in its 
        `fit` method. See Notes section for more details on the use of the weights.

        - If single function: it is applied to all series. 
        - If `dict` {'series_column_name' : Callable}: a different function can be
        used for each series, a weight of 1 is given to all series not present 
        in `weight_func`.
    weight_func_ : dict
        Dictionary with the `weight_func` for each series. It is created cloning the 
        objects in `weight_func` and is used internally to avoid overwriting.
    source_code_weight_func : str, dict
        Source code of the custom function(s) used to create weights.
    series_weights : dict
        Weights associated with each series {'series_column_name' : float}. It is only
        applied if the `regressor` used accepts `sample_weight` in its `fit` method. 
        See Notes section for more details on the use of the weights.

        - If a `dict` is provided, a weight of 1 is given to all series not present
        in `series_weights`.
        - If `None`, all levels have the same weight.
    series_weights_ : dict
        Weights associated with each series.It is created as a clone of `series_weights`
        and is used internally to avoid overwriting.
    differentiation : int
        Order of differencing applied to the time series before training the 
        forecaster.
    differentiator : TimeSeriesDifferentiator
        Skforecast object used to differentiate the time series.
    differentiator_ : dict
        Dictionary with the `differentiator` for each series. It is created cloning the
        objects in `differentiator` and is used internally to avoid overwriting.
    dropna_from_series : bool
        Determine whether NaN detected in the training matrices will be dropped.
    last_window_ : dict
        Last window of training data for each series. It stores the values 
        needed to predict the next `step` immediately after the training data. 
        These values are stored in the original scale of the time series before 
        undergoing any transformations or differentiation. When `differentiation` 
        parameter is specified, the dimensions of the `last_window_` are expanded 
        as many values as the order of differentiation. For example, if 
        `lags` = 7 and `differentiation` = 1, `last_window_` will have 8 values.
    index_type_ : type
        Type of index of the input used in training.
    index_freq_ : str
        Frequency of Index of the input used in training.
    training_range_: dict
        First and last values of index of the data used during training for each 
        series.
    series_names_in_ : list
        Names of the series (levels) provided by the user during training.
    exog_in_ : bool
        If the forecaster has been trained using exogenous variable/s.
    exog_names_in_ : list
        Names of the exogenous variables used during training.
    exog_type_in_ : type
        Type of exogenous data (pandas Series, DataFrame or dict) used in training.
    exog_dtypes_in_ : dict
        Type of each exogenous variable/s used in training. If `transformer_exog` 
        is used, the dtypes are calculated before the transformation.
    X_train_series_names_in_ : list
        Names of the series (levels) included in the matrix `X_train` created
        internally for training. It can be different from `series_names_in_` if
        some series are dropped during the training process because of NaNs or 
        because they are not present in the training period.
    X_train_window_features_names_out_ : list
        Names of the window features included in the matrix `X_train` created
        internally for training.
    X_train_exog_names_out_ : list
        Names of the exogenous variables included in the matrix `X_train` created
        internally for training. It can be different from `exog_names_in_` if
        some exogenous variables are transformed during the training process.
    X_train_features_names_out_ : list
        Names of columns of the matrix created internally for training.
    fit_kwargs : dict
        Additional arguments to be passed to the `fit` method of the regressor.
    in_sample_residuals_ : dict
        Residuals of the model when predicting training data. Only stored up to
        1000 values in the form `{level: residuals}`. If `transformer_series` 
        is not `None`, residuals are stored in the transformed scale.
    out_sample_residuals_ : dict
        Residuals of the model when predicting non-training data. Only stored
        up to 1000 values in the form `{level: residuals}`. If `transformer_series` 
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
    forecaster_id : str, int
        Name used as an identifier of the forecaster.

    Notes
    -----
    The weights are used to control the influence that each observation has on the
    training of the model. `ForecasterRecursiveMultiSeries` accepts two types of weights. 
    If the two types of weights are indicated, they are multiplied to create the final
    weights. The resulting `sample_weight` cannot have negative values.

    - `series_weights` : controls the relative importance of each series. If a 
    series has twice as much weight as the others, the observations of that series 
    influence the training twice as much. The higher the weight of a series 
    relative to the others, the more the model will focus on trying to learn 
    that series.
    - `weight_func` : controls the relative importance of each observation 
    according to its index value. For example, a function that assigns a lower 
    weight to certain dates.
    
    """

    def __init__(
        self,
        regressor: object,
        lags: Optional[Union[int, list, np.ndarray, range]] = None,
        window_features: Optional[Union[object, list]] = None,
        encoding: Optional[str] = 'ordinal',
        transformer_series: Optional[Union[object, dict]] = None,
        transformer_exog: Optional[object] = None,
        weight_func: Optional[Union[Callable, dict]] = None,
        series_weights: Optional[dict] = None,
        differentiation: Optional[int] = None,
        dropna_from_series: bool = False,
        fit_kwargs: Optional[dict] = None,
        forecaster_id: Optional[Union[str, int]] = None
    ) -> None:

        self.regressor                          = copy(regressor)
        self.encoding                           = encoding
        self.encoder                            = None
        self.encoding_mapping_                  = {}
        self.transformer_series                 = transformer_series
        self.transformer_series_                = None
        self.transformer_exog                   = transformer_exog
        self.weight_func                        = weight_func
        self.weight_func_                       = None
        self.source_code_weight_func            = None
        self.series_weights                     = series_weights
        self.series_weights_                    = None
        self.differentiation                    = differentiation
        self.differentiator                     = None
        self.differentiator_                    = None
        self.dropna_from_series                 = dropna_from_series
        self.last_window_                       = None
        self.index_type_                        = None
        self.index_freq_                        = None
        self.training_range_                    = None
        self.series_names_in_                   = None
        self.exog_in_                           = False
        self.exog_names_in_                     = None
        self.exog_type_in_                      = None
        self.exog_dtypes_in_                    = None 
        self.X_train_series_names_in_           = None
        self.X_train_window_features_names_out_ = None
        self.X_train_exog_names_out_            = None
        self.X_train_features_names_out_        = None
        self.in_sample_residuals_               = None
        self.out_sample_residuals_              = None
        self.creation_date                      = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.is_fitted                          = False
        self.fit_date                           = None
        self.skforecast_version                 = skforecast.__version__
        self.python_version                     = sys.version.split(" ")[0]
        self.forecaster_id                      = forecaster_id

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

        self.weight_func, self.source_code_weight_func, self.series_weights = (
            initialize_weights(
                forecaster_name = type(self).__name__,
                regressor       = regressor,
                weight_func     = weight_func,
                series_weights  = series_weights,
            )
        )

        if self.differentiation is not None:
            if not isinstance(differentiation, int) or differentiation < 1:
                raise ValueError(
                    (f"Argument `differentiation` must be an integer equal to or "
                     f"greater than 1. Got {differentiation}.")
                )
            self.window_size += self.differentiation
            self.differentiator = TimeSeriesDifferentiator(order=self.differentiation)

        self.fit_kwargs = check_select_fit_kwargs(
                              regressor  = regressor,
                              fit_kwargs = fit_kwargs
                          )

        if self.encoding not in ['ordinal', 'ordinal_category', 'onehot', None]:
            raise ValueError(
                (f"Argument `encoding` must be one of the following values: 'ordinal', "
                 f"'ordinal_category', 'onehot' or None. Got '{self.encoding}'.")
            )

        if self.encoding == 'onehot':
            self.encoder = OneHotEncoder(
                               categories    = 'auto',
                               sparse_output = False,
                               drop          = None,
                               dtype         = int
                           ).set_output(transform='pandas')
        else:
            self.encoder = OrdinalEncoder(
                               categories = 'auto',
                               dtype      = int
                           ).set_output(transform='pandas')

        scaling_regressors = tuple(
            member[1]
            for member in inspect.getmembers(sklearn.linear_model, inspect.isclass)
            + inspect.getmembers(sklearn.svm, inspect.isclass)
        )

        if self.transformer_series is None and isinstance(regressor, scaling_regressors):
            warnings.warn(
                ("When using a linear model, it is recommended to use a transformer_series "
                 "to ensure all series are in the same scale. You can use, for example, a "
                 "`StandardScaler` from sklearn.preprocessing.")
            )

        if isinstance(self.transformer_series, dict):
            if self.encoding is None:
                raise TypeError(
                    ("When `encoding` is None, `transformer_series` must be a single "
                     "transformer (not `dict`) as it is applied to all series.")
                )
            if '_unknown_level' not in self.transformer_series.keys():
                raise ValueError(
                    ("If `transformer_series` is a `dict`, a transformer must be "
                     "provided to transform series that do not exist during training. "
                     "Add the key '_unknown_level' to `transformer_series`. "
                     "For example: {'_unknown_level': your_transformer}.")
                )


    def __repr__(
        self
    ) -> str:
        """
        Information displayed when a ForecasterRecursiveMultiSeries object is printed.
        """
        
        (
            params,
            training_range_,
            series_names_in_,
            exog_names_in_,
            transformer_series,
        ) = [
            self._format_text_repr(value) 
            for value in self._preprocess_repr(
                regressor          = self.regressor,
                training_range_    = self.training_range_,
                series_names_in_   = self.series_names_in_,
                exog_names_in_     = self.exog_names_in_,
                transformer_series = self.transformer_series,
            )
        ]

        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Regressor: {self.regressor} \n"
            f"Lags: {self.lags} \n"
            f"Window features: {self.window_features_names} \n"
            f"Window size: {self.window_size} \n"
            f"Series encoding: {self.encoding} \n"
            f"Series names (levels): {series_names_in_} \n"
            f"Exogenous included: {self.exog_in_} \n"
            f"Exogenous names: {exog_names_in_} \n"
            f"Transformer for series: {transformer_series} \n"
            f"Transformer for exog: {self.transformer_exog} \n"
            f"Weight function included: {True if self.weight_func is not None else False} \n"
            f"Series weights: {self.series_weights} \n"
            f"Differentiation order: {self.differentiation} \n"
            f"Training range: {training_range_} \n"
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
            training_range_,
            series_names_in_,
            exog_names_in_,
            transformer_series,
        ) = self._preprocess_repr(
                regressor          = self.regressor,
                training_range_    = self.training_range_,
                series_names_in_   = self.series_names_in_,
                exog_names_in_     = self.exog_names_in_,
                transformer_series = self.transformer_series,
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
                    <li><strong>Series encoding:</strong> {self.encoding}</li>
                    <li><strong>Exogenous included:</strong> {self.exog_in_}</li>
                    <li><strong>Weight function included:</strong> {self.weight_func is not None}</li>
                    <li><strong>Series weights:</strong> {self.series_weights}</li>
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
                    <li><strong>Transformer for series:</strong> {transformer_series}</li>
                    <li><strong>Transformer for exog:</strong> {self.transformer_exog}</li>
                </ul>
            </details>
            <details>
                <summary>Training Information</summary>
                <ul>
                    <li><strong>Series names (levels):</strong> {series_names_in_}</li>
                    <li><strong>Training range:</strong> {training_range_}</li>
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
                <a href="https://skforecast.org/{skforecast.__version__}/api/forecastermultiseries#forecasterrecursivemultiseries.html">&#128712 <strong>API Reference</strong></a>
                &nbsp;&nbsp;
                <a href="https://skforecast.org/{skforecast.__version__}/user_guides/independent-multi-time-series-forecasting.html">&#128462 <strong>User Guide</strong></a>
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

        X_data = None
        if self.lags is not None:
            n_rows = len(y) - self.window_size
            X_data = np.full(
                shape=(n_rows, len(self.lags)), fill_value=np.nan, order='F', dtype=float
            )
            for i, lag in enumerate(self.lags):
                X_data[:, i] = y[self.window_size - lag: -lag]

            if X_as_pandas:
                X_data = pd.DataFrame(
                             data    = X_data,
                             columns = self.lags_names,
                             index   = train_index
                         )

        y_data = y[self.window_size:]

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
                     f"the input time series - `window_size`: {len_train_index}.")
                )
            if not (X_train_wf.index == train_index).all():
                raise ValueError(
                    (f"The method `transform_batch` of {type(wf).__name__} "
                     f"must return a DataFrame with the same index as "
                     f"the input time series - `window_size`.")
                )
            
            X_train_window_features_names_out_.extend(X_train_wf.columns)
            if not X_as_pandas:
                X_train_wf = X_train_wf.to_numpy()     
            X_train_window_features.append(X_train_wf)

        return X_train_window_features, X_train_window_features_names_out_


    def _create_train_X_y_single_series(
        self,
        y: pd.Series,
        ignore_exog: bool,
        exog: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, list, pd.DataFrame, pd.Series]:
        """
        Create training matrices from univariate time series and exogenous
        variables. This method does not transform the exog variables.
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        ignore_exog : bool
            If `True`, `exog` is ignored.
        exog : pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.

        Returns
        -------
        X_train_lags : pandas DataFrame
            Training values of lags.
            Shape: (len(y) - self.max_lag, len(self.lags))
        X_train_window_features_names_out_ : list
            Names of the window features.
        X_train_exog : pandas DataFrame
            Training values of exogenous variables.
            Shape: (len(y) - self.max_lag, len(exog.columns))
        y_train : pandas Series
            Values (target) of the time series related to each row of `X_train`.
            Shape: (len(y) - self.max_lag, )
        
        """

        series_name = y.name
        if len(y) <= self.window_size:
            raise ValueError(
                f"Length of '{series_name}' must be greater than the maximum window size "
                f"needed by the forecaster.\n"
                f"    Length '{series_name}': {len(y)}.\n"
                f"    Max window size: {self.window_size}.\n"
                f"    Lags window size: {self.max_lag}.\n"
                f"    Window features window size: {self.max_size_window_features}."
            )

        if self.encoding is None:
            fit_transformer = False
            transformer_series = self.transformer_series_['_unknown_level']
        else:
            fit_transformer = False if self.is_fitted else True
            transformer_series = self.transformer_series_[series_name]

        y = transform_series(
                series            = y,
                transformer       = transformer_series,
                fit               = fit_transformer,
                inverse_transform = False
            )

        y_values = y.to_numpy()
        y_index = y.index

        if self.differentiation is not None:
            if not self.is_fitted:
                y_values = self.differentiator_[series_name].fit_transform(y_values)
            else:
                differentiator = copy(self.differentiator_[series_name])
                y_values = differentiator.fit_transform(y_values)

        X_train_autoreg = []
        train_index = y_index[self.window_size:]

        X_train_lags, y_train = self._create_lags(
            y=y_values, X_as_pandas=True, train_index=train_index
        )
        if X_train_lags is not None:
            X_train_autoreg.append(X_train_lags)
        
        X_train_window_features_names_out_ = None
        if self.window_features is not None:
            n_diff = 0 if self.differentiation is None else self.differentiation
            y_window_features = pd.Series(y_values[n_diff:], index=y_index[n_diff:])
            X_train_window_features, X_train_window_features_names_out_ = (
                self._create_window_features(
                    y=y_window_features, X_as_pandas=True, train_index=train_index
                )
            )
            X_train_autoreg.extend(X_train_window_features)

        if len(X_train_autoreg) == 1:
            X_train_autoreg = X_train_autoreg[0]
        else:
            X_train_autoreg = pd.concat(X_train_autoreg, axis=1)
        
        X_train_autoreg['_level_skforecast'] = series_name

        if ignore_exog:
            X_train_exog = None
        else:
            if exog is not None:
                # The first `self.window_size` positions have to be removed from exog
                # since they are not in X_train_autoreg.
                X_train_exog = exog.iloc[self.window_size:, ]
            else:
                X_train_exog = pd.DataFrame(
                                   data    = np.nan,
                                   columns = ['_dummy_exog_col_to_keep_shape'],
                                   index   = train_index
                               )

        y_train = pd.Series(
                      data  = y_train,
                      index = train_index,
                      name  = 'y'
                  )

        return X_train_autoreg, X_train_window_features_names_out_, X_train_exog, y_train


    def _create_train_X_y(
        self,
        series: Union[pd.DataFrame, dict],
        exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
        store_last_window: Union[bool, list] = True,
    ) -> Tuple[pd.DataFrame, pd.Series, dict, list, list, list, list, list, dict, dict]:
        """
        Create training matrices from multiple time series and exogenous
        variables. See Notes section for more details depending on the type of
        `series` and `exog`.
        
        Parameters
        ----------
        series : pandas DataFrame, dict
            Training time series.
        exog : pandas Series, pandas DataFrame, dict, default `None`
            Exogenous variable/s included as predictor/s.
        store_last_window : bool, list, default `True`
            Whether or not to store the last window (`last_window_`) of training data.

            - If `True`, last window is stored for all series. 
            - If `list`, last window is stored for the series present in the list.
            - If `False`, last window is not stored.

        Returns
        -------
        X_train : pandas DataFrame
            Training values (predictors).
        y_train : pandas Series
            Values of the time series related to each row of `X_train`.
        series_indexes : dict
            Dictionary with the index of each series.
        series_names_in_ : list
            Names of the series (levels) provided by the user during training.
        X_train_series_names_in_ : list
            Names of the series (levels) included in the matrix `X_train` created
            internally for training. It can be different from `series_names_in_` if
            some series are dropped during the training process because of NaNs or
            because they are not present in the training period.
        exog_names_in_ : list
            Names of the exogenous variables used during training.
        X_train_window_features_names_out_ : list
            Names of the window features included in the matrix `X_train` created
            internally for training.
        X_train_exog_names_out_ : list
            Names of the exogenous variables included in the matrix `X_train` created
            internally for training. It can be different from `exog_names_in_` if
            some exogenous variables are transformed during the training process.
        exog_dtypes_in_ : dict
            Type of each exogenous variable/s used in training. If `transformer_exog` 
            is used, the dtypes are calculated before the transformation.
        last_window_ : dict
            Last window of training data for each series. It stores the values 
            needed to predict the next `step` immediately after the training data.

        Notes
        -----
        - If `series` is a pandas DataFrame and `exog` is a pandas Series or 
        DataFrame, each exog is duplicated for each series. Exog must have the
        same index as `series` (type, length and frequency).
        - If `series` is a pandas DataFrame and `exog` is a dict of pandas Series 
        or DataFrames. Each key in `exog` must be a column in `series` and the 
        values are the exog for each series. Exog must have the same index as 
        `series` (type, length and frequency).
        - If `series` is a dict of pandas Series, `exog` must be a dict of pandas
        Series or DataFrames. The keys in `series` and `exog` must be the same.
        All series and exog must have a pandas DatetimeIndex with the same 
        frequency.
        
        """

        series_dict, series_indexes = check_preprocess_series(series=series)
        input_series_is_dict = isinstance(series, dict)
        series_names_in_ = list(series_dict.keys())

        if self.is_fitted and not set(series_names_in_).issubset(set(self.series_names_in_)):
            raise ValueError(
                (f"Once the Forecaster has been trained, `series` must contain "
                 f"the same series names as those used during training:\n"
                 f" Got      : {series_names_in_}\n"
                 f" Expected : {self.series_names_in_}")
            )

        exog_dict = {serie: None for serie in series_names_in_}
        exog_names_in_ = None
        X_train_exog_names_out_ = None
        if exog is not None:
            exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                                            input_series_is_dict = input_series_is_dict,
                                            series_indexes       = series_indexes,
                                            series_names_in_     = series_names_in_,
                                            exog                 = exog,
                                            exog_dict            = exog_dict
                                        )

            if self.is_fitted:
                if self.exog_names_in_ is None:
                    raise ValueError(
                        ("Once the Forecaster has been trained, `exog` must be `None` "
                         "because no exogenous variables were added during training.")
                    )
                else:
                    if not set(exog_names_in_) == set(self.exog_names_in_):
                        raise ValueError(
                            (f"Once the Forecaster has been trained, `exog` must contain "
                             f"the same exogenous variables as those used during training:\n"
                             f" Got      : {exog_names_in_}\n"
                             f" Expected : {self.exog_names_in_}")
                        )

        if not self.is_fitted:
            self.transformer_series_ = initialize_transformer_series(
                                           forecaster_name    = type(self).__name__,
                                           series_names_in_   = series_names_in_,
                                           encoding           = self.encoding,
                                           transformer_series = self.transformer_series
                                       )

        if self.differentiation is None:
            self.differentiator_ = {serie: None for serie in series_names_in_}
        else:
            if not self.is_fitted:
                self.differentiator_ = {serie: copy(self.differentiator)
                                        for serie in series_names_in_}

        series_dict, exog_dict = align_series_and_exog_multiseries(
                                     series_dict          = series_dict,
                                     input_series_is_dict = input_series_is_dict,
                                     exog_dict            = exog_dict
                                 )
        
        if not self.is_fitted and self.transformer_series_['_unknown_level'] is not None:
            self.transformer_series_['_unknown_level'].fit(
                np.concatenate(list(series_dict.values())).reshape(-1, 1)
            )

        ignore_exog = True if exog is None else False
        input_matrices = [
            [series_dict[k], exog_dict[k], ignore_exog]
             for k in series_dict.keys()
        ]

        X_train_autoreg_buffer = []
        X_train_exog_buffer = []
        y_train_buffer = []
        for matrices in input_matrices:

            (
                X_train_autoreg,
                X_train_window_features_names_out_,
                X_train_exog,
                y_train
            ) = self._create_train_X_y_single_series(
                y           = matrices[0],
                exog        = matrices[1],
                ignore_exog = matrices[2],
            )

            X_train_autoreg_buffer.append(X_train_autoreg)
            X_train_exog_buffer.append(X_train_exog)
            y_train_buffer.append(y_train)

        X_train = pd.concat(X_train_autoreg_buffer, axis=0)
        y_train = pd.concat(y_train_buffer, axis=0)

        if self.is_fitted:
            encoded_values = self.encoder.transform(X_train[['_level_skforecast']])
        else:
            encoded_values = self.encoder.fit_transform(X_train[['_level_skforecast']])
            for i, code in enumerate(self.encoder.categories_[0]):
                self.encoding_mapping_[code] = i

        X_train = pd.concat([
                      X_train.drop(columns='_level_skforecast'),
                      encoded_values
                  ], axis=1)

        if self.encoding == 'onehot':
            X_train.columns = X_train.columns.str.replace('_level_skforecast_', '')
        elif self.encoding == 'ordinal_category':
            X_train['_level_skforecast'] = (
                X_train['_level_skforecast'].astype('category')
            )

        del encoded_values

        exog_dtypes_in_ = None
        if exog is not None:

            X_train_exog = pd.concat(X_train_exog_buffer, axis=0)
            if '_dummy_exog_col_to_keep_shape' in X_train_exog.columns:
                X_train_exog = (
                    X_train_exog.drop(columns=['_dummy_exog_col_to_keep_shape'])
                )

            exog_names_in_ = X_train_exog.columns.to_list()
            exog_dtypes_in_ = get_exog_dtypes(exog=X_train_exog)

            fit_transformer = False if self.is_fitted else True
            X_train_exog = transform_dataframe(
                               df                = X_train_exog,
                               transformer       = self.transformer_exog,
                               fit               = fit_transformer,
                               inverse_transform = False
                           )

            check_exog_dtypes(X_train_exog, call_check_exog=False)
            if not (X_train_exog.index == X_train.index).all():
                raise ValueError(
                    ("Different index for `series` and `exog` after transformation. "
                     "They must be equal to ensure the correct alignment of values.")
                )

            X_train_exog_names_out_ = X_train_exog.columns.to_list()
            X_train = pd.concat([X_train, X_train_exog], axis=1)

        if y_train.isnull().any():
            mask = y_train.notna().to_numpy()
            y_train = y_train.iloc[mask]
            X_train = X_train.iloc[mask,]
            warnings.warn(
                ("NaNs detected in `y_train`. They have been dropped because the "
                 "target variable cannot have NaN values. Same rows have been "
                 "dropped from `X_train` to maintain alignment. This is caused by "
                 "series with interspersed NaNs."),
                 MissingValuesWarning
            )

        if self.dropna_from_series:
            if np.any(X_train.isnull().to_numpy()):
                mask = X_train.notna().all(axis=1).to_numpy()
                X_train = X_train.iloc[mask, ]
                y_train = y_train.iloc[mask]
                warnings.warn(
                    ("NaNs detected in `X_train`. They have been dropped. If "
                     "you want to keep them, set `forecaster.dropna_from_series = False`. "
                     "Same rows have been removed from `y_train` to maintain alignment. "
                     "This caused by series with interspersed NaNs."),
                     MissingValuesWarning
                )
        else:
            if np.any(X_train.isnull().to_numpy()):
                warnings.warn(
                    ("NaNs detected in `X_train`. Some regressors do not allow "
                     "NaN values during training. If you want to drop them, "
                     "set `forecaster.dropna_from_series = True`."),
                     MissingValuesWarning
                )

        if X_train.empty:
            raise ValueError(
                ("All samples have been removed due to NaNs. Set "
                 "`forecaster.dropna_from_series = False` or review `exog` values.")
            )
        
        if self.encoding == 'onehot':
            X_train_series_names_in_ = [
                col for col in series_names_in_ if X_train[col].sum() > 0
            ]
        else:
            unique_levels = X_train['_level_skforecast'].unique()
            X_train_series_names_in_ = [
                k for k, v in self.encoding_mapping_.items()
                if v in unique_levels
            ]

        # The last time window of training data is stored so that lags needed as
        # predictors in the first iteration of `predict()` can be calculated.
        last_window_ = None
        if store_last_window:

            series_to_store = (
                X_train_series_names_in_ if store_last_window is True else store_last_window
            )

            series_not_in_series_dict = set(series_to_store) - set(X_train_series_names_in_)
            if series_not_in_series_dict:
                warnings.warn(
                    (f"Series {series_not_in_series_dict} are not present in "
                     f"`series`. No last window is stored for them."),
                    IgnoredArgumentWarning
                )
                series_to_store = [s for s in series_to_store 
                                   if s not in series_not_in_series_dict]

            if series_to_store:
                last_window_ = {
                    k: v.iloc[-self.window_size:].copy()
                    for k, v in series_dict.items()
                    if k in series_to_store
                }

        return (
            X_train,
            y_train,
            series_indexes,
            series_names_in_,
            X_train_series_names_in_,
            exog_names_in_,
            X_train_window_features_names_out_,
            X_train_exog_names_out_,
            exog_dtypes_in_,
            last_window_
        )


    def create_train_X_y(
        self,
        series: Union[pd.DataFrame, dict],
        exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
        suppress_warnings: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create training matrices from multiple time series and exogenous
        variables. See Notes section for more details depending on the type of
        `series` and `exog`.
        
        Parameters
        ----------
        series : pandas DataFrame, dict
            Training time series.
        exog : pandas Series, pandas DataFrame, dict, default `None`
            Exogenous variable/s included as predictor/s.
        suppress_warnings : bool, default `False`
            If `True`, skforecast warnings will be suppressed during the creation
            of the training matrices. See skforecast.exceptions.warn_skforecast_categories 
            for more information.

        Returns
        -------
        X_train : pandas DataFrame
            Training values (predictors).
        y_train : pandas Series
            Values (target) of the time series related to each row of `X_train`.

        Notes
        -----
        - If `series` is a pandas DataFrame and `exog` is a pandas Series or 
        DataFrame, each exog is duplicated for each series. Exog must have the
        same index as `series` (type, length and frequency).
        - If `series` is a pandas DataFrame and `exog` is a dict of pandas Series 
        or DataFrames. Each key in `exog` must be a column in `series` and the 
        values are the exog for each series. Exog must have the same index as 
        `series` (type, length and frequency).
        - If `series` is a dict of pandas Series, `exog`must be a dict of pandas
        Series or DataFrames. The keys in `series` and `exog` must be the same.
        All series and exog must have a pandas DatetimeIndex with the same 
        frequency.
        
        """

        set_skforecast_warnings(suppress_warnings, action='ignore')

        output = self._create_train_X_y(
                     series            = series, 
                     exog              = exog, 
                     store_last_window = False
                 )

        X_train = output[0]
        y_train = output[1]

        if self.encoding is None:
            X_train = X_train.drop(columns='_level_skforecast')
        
        set_skforecast_warnings(suppress_warnings, action='default')

        return X_train, y_train


    def _train_test_split_one_step_ahead(
        self,
        series: Union[pd.DataFrame, dict],
        initial_train_size: int,
        exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Create matrices needed to train and test the forecaster for one-step-ahead
        predictions.

        Parameters
        ----------
        series : pandas DataFrame, dict
            Training time series.
        initial_train_size : int
            Initial size of the training set. It is the number of observations used
            to train the forecaster before making the first prediction.
        exog : pandas Series, pandas DataFrame, dict, default `None`
            Exogenous variable/s included as predictor/s.
        
        Returns
        -------
        X_train : pandas DataFrame
            Predictor values used to train the model.
        y_train : pandas Series
            Target values related to each row of `X_train`.
        X_test : pandas DataFrame
            Predictor values used to test the model.
        y_test : pandas Series
            Target values related to each row of `X_test`.
        X_train_encoding : pandas Series
            Series identifiers for each row of `X_train`.
        X_test_encoding : pandas Series
            Series identifiers for each row of `X_test`.
        
        """

        if isinstance(series, dict):
            freqs = [s.index.freq for s in series.values() if s.index.freq is not None]
            if not freqs:
                raise ValueError("At least one series must have a frequency.")
            if not all(f == freqs[0] for f in freqs):
                raise ValueError(
                    "All series with frequency must have the same frequency."
                )
            min_index = min([v.index[0] for v in series.values()])
            max_index = max([v.index[-1] for v in series.values()])
            span_index = pd.date_range(start=min_index, end=max_index, freq=freqs[0])
        else:
            span_index = series.index

        fold = [
            [0, initial_train_size],
            [initial_train_size - self.window_size, initial_train_size],
            [initial_train_size - self.window_size, len(span_index)],
            [0, 0],  # Dummy value
            True
        ]
        data_fold = _extract_data_folds_multiseries(
                        series             = series,
                        folds              = [fold],
                        span_index         = span_index,
                        window_size        = self.window_size,
                        exog               = exog,
                        dropna_last_window = self.dropna_from_series,
                        externally_fitted  = False
                    )
        series_train, _, levels_last_window, exog_train, exog_test, _ = next(data_fold)

        start_test_idx = initial_train_size - self.window_size
        if isinstance(series, pd.DataFrame):
            series_test = series.iloc[start_test_idx:, :]
            series_test = series_test.loc[:, levels_last_window]
            series_test = series_test.dropna(axis=1, how='all')
        elif isinstance(series, dict):
            start_test_date = span_index[start_test_idx]
            series_test = {
                k: v.loc[v.index >= start_test_date]
                for k, v in series.items()
                if k in levels_last_window and not v.empty and not v.isna().all()
            }
       
        _is_fitted = self.is_fitted
        _series_names_in_ = self.series_names_in_
        _exog_names_in_ = self.exog_names_in_

        self.is_fitted = False
        X_train, y_train, _, series_names_in_, _, exog_names_in_, *_ = (
            self._create_train_X_y(
                series            = series_train,
                exog              = exog_train,
                store_last_window = False
            )
        )
        
        self.series_names_in_ = series_names_in_
        if exog is not None:
            self.exog_names_in_ = exog_names_in_
        self.is_fitted = True

        X_test, y_test, *_ = self._create_train_X_y(
                                 series            = series_test,
                                 exog              = exog_test,
                                 store_last_window = False
                             )
        self.is_fitted = _is_fitted
        self.series_names_in_ = _series_names_in_
        self.exog_names_in_ = _exog_names_in_

        if self.encoding in ["ordinal", "ordinal_category"]:
            X_train_encoding = self.encoder.inverse_transform(
                X_train[["_level_skforecast"]]
            ).ravel()
            X_test_encoding = self.encoder.inverse_transform(
                X_test[["_level_skforecast"]]
            ).ravel()
        elif self.encoding == 'onehot':
            X_train_encoding = self.encoder.inverse_transform(
                X_train.loc[:, self.encoding_mapping_.keys()]
            ).ravel()
            X_test_encoding = self.encoder.inverse_transform(
                X_test.loc[:, self.encoding_mapping_.keys()]
            ).ravel()
        else:
            X_train_encoding = self.encoder.inverse_transform(
                X_train[["_level_skforecast"]]
            ).ravel()
            X_test_encoding = self.encoder.inverse_transform(
                X_test[["_level_skforecast"]]
            ).ravel()
            X_train = X_train.drop(columns="_level_skforecast")
            X_test = X_test.drop(columns="_level_skforecast")
        
        X_train_encoding = pd.Series(
            data=X_train_encoding, index=X_train.index
        ).fillna("_unknown_level")
        X_test_encoding = pd.Series(
            data=X_test_encoding, index=X_test.index
        ).fillna("_unknown_level")
        
        return X_train, y_train, X_test, y_test, X_train_encoding, X_test_encoding


    def create_sample_weights(
        self,
        series_names_in_: list,
        X_train: pd.DataFrame
    ) -> np.ndarray:
        """
        Crate weights for each observation according to the forecaster's attributes
        `series_weights` and `weight_func`. The resulting weights are product of both
        types of weights.

        Parameters
        ----------
        series_names_in_ : list
            Names of the series (levels) used during training.
        X_train : pandas DataFrame
            Dataframe created with the `create_train_X_y` method, first return.

        Returns
        -------
        weights : numpy ndarray
            Weights to use in `fit` method.
        
        """

        weights = None
        weights_samples = None
        weights_series = None

        if self.series_weights is not None:
            # Series not present in series_weights have a weight of 1 in all their samples.
            # Keys in series_weights not present in series are ignored.
            series_not_in_series_weights = (
                set(series_names_in_) - set(self.series_weights.keys())
            )
            if series_not_in_series_weights:
                warnings.warn(
                    (f"{series_not_in_series_weights} not present in `series_weights`. "
                     f"A weight of 1 is given to all their samples."),
                     IgnoredArgumentWarning
                )
            self.series_weights_ = {col: 1. for col in series_names_in_}
            self.series_weights_.update(
                (k, v)
                for k, v in self.series_weights.items()
                if k in self.series_weights_
            )

            if self.encoding == "onehot":
                weights_series = [
                    np.repeat(self.series_weights_[serie], sum(X_train[serie]))
                    for serie in series_names_in_
                ]
            else:
                weights_series = [
                    np.repeat(
                        self.series_weights_[serie],
                        sum(X_train["_level_skforecast"] == self.encoding_mapping_[serie]),
                    )
                    for serie in series_names_in_
                ]

            weights_series = np.concatenate(weights_series)

        if self.weight_func is not None:
            if isinstance(self.weight_func, Callable):
                self.weight_func_ = {col: copy(self.weight_func)
                                     for col in series_names_in_}
            else:
                # Series not present in weight_func have a weight of 1 in all their samples
                series_not_in_weight_func = (
                    set(series_names_in_) - set(self.weight_func.keys())
                )
                if series_not_in_weight_func:
                    warnings.warn(
                        (f"{series_not_in_weight_func} not present in `weight_func`. "
                         f"A weight of 1 is given to all their samples."),
                         IgnoredArgumentWarning
                    )
                self.weight_func_ = {col: lambda x: np.ones_like(x, dtype=float) 
                                     for col in series_names_in_}
                self.weight_func_.update(
                    (k, v)
                    for k, v in self.weight_func.items()
                    if k in self.weight_func_
                )

            weights_samples = []
            for key in self.weight_func_.keys():
                if self.encoding == "onehot":
                    idx = X_train.index[X_train[key] == 1.0]
                else:
                    idx = X_train.index[
                            X_train["_level_skforecast"] == self.encoding_mapping_[key]
                         ]
                weights_samples.append(self.weight_func_[key](idx))
            weights_samples = np.concatenate(weights_samples)

        if weights_series is not None:
            weights = weights_series
            if weights_samples is not None:
                weights = weights * weights_samples
        else:
            if weights_samples is not None:
                weights = weights_samples

        if weights is not None:
            if np.isnan(weights).any():
                raise ValueError(
                    "The resulting `weights` cannot have NaN values."
                )
            if np.any(weights < 0):
                raise ValueError(
                    "The resulting `weights` cannot have negative values."
                )
            if np.sum(weights) == 0:
                raise ValueError(
                    ("The resulting `weights` cannot be normalized because "
                     "the sum of the weights is zero.")
                )

        return weights


    def fit(
        self,
        series: Union[pd.DataFrame, dict],
        exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
        store_last_window: Union[bool, list] = True,
        store_in_sample_residuals: bool = True,
        suppress_warnings: bool = False
    ) -> None:
        """
        Training Forecaster. See Notes section for more details depending on 
        the type of `series` and `exog`.

        Additional arguments to be passed to the `fit` method of the regressor 
        can be added with the `fit_kwargs` argument when initializing the forecaster.
        
        Parameters
        ----------
        series : pandas DataFrame, dict
            Training time series.
        exog : pandas Series, pandas DataFrame, dict, default `None`
            Exogenous variable/s included as predictor/s.
        store_last_window : bool, list, default `True`
            Whether or not to store the last window (`last_window_`) of training data.

            - If `True`, last window is stored for all series. 
            - If `list`, last window is stored for the series present in the list.
            - If `False`, last window is not stored.
        store_in_sample_residuals : bool, default `True`
            If `True`, in-sample residuals will be stored in the forecaster object
            after fitting (`in_sample_residuals_` attribute).
        suppress_warnings : bool, default `False`
            If `True`, skforecast warnings will be suppressed during the training 
            process. See skforecast.exceptions.warn_skforecast_categories for more
            information.

        Returns
        -------
        None

        Notes
        -----
        - If `series` is a pandas DataFrame and `exog` is a pandas Series or 
        DataFrame, each exog is duplicated for each series. Exog must have the
        same index as `series` (type, length and frequency).
        - If `series` is a pandas DataFrame and `exog` is a dict of pandas Series 
        or DataFrames. Each key in `exog` must be a column in `series` and the 
        values are the exog for each series. Exog must have the same index as 
        `series` (type, length and frequency).
        - If `series` is a dict of pandas Series, `exog`must be a dict of pandas
        Series or DataFrames. The keys in `series` and `exog` must be the same.
        All series and exog must have a pandas DatetimeIndex with the same 
        frequency.
        
        """

        set_skforecast_warnings(suppress_warnings, action='ignore')

        # Reset values in case the forecaster has already been fitted.
        self.last_window_                       = None
        self.index_type_                        = None
        self.index_freq_                        = None
        self.training_range_                    = None
        self.series_names_in_                   = None
        self.exog_in_                           = False
        self.exog_names_in_                     = None
        self.exog_type_in_                      = None
        self.exog_dtypes_in_                    = None
        self.X_train_series_names_in_           = None
        self.X_train_window_features_names_out_ = None
        self.X_train_exog_names_out_            = None
        self.X_train_features_names_out_        = None
        self.in_sample_residuals_               = None
        self.is_fitted                          = False
        self.fit_date                           = None

        (
            X_train,
            y_train,
            series_indexes,
            series_names_in_,
            X_train_series_names_in_,
            exog_names_in_,
            X_train_window_features_names_out_,
            X_train_exog_names_out_,
            exog_dtypes_in_,
            last_window_
        ) = self._create_train_X_y(
                series=series, exog=exog, store_last_window=store_last_window
            )

        sample_weight = self.create_sample_weights(
                            series_names_in_ = series_names_in_,
                            X_train          = X_train
                        )

        X_train_regressor = (
            X_train
            if self.encoding is not None
            else X_train.drop(columns="_level_skforecast")
        )
        if sample_weight is not None:
            self.regressor.fit(
                X             = X_train_regressor,
                y             = y_train,
                sample_weight = sample_weight,
                **self.fit_kwargs
            )
        else:
            self.regressor.fit(X=X_train_regressor, y=y_train, **self.fit_kwargs)

        self.series_names_in_ = series_names_in_
        self.X_train_series_names_in_ = X_train_series_names_in_
        self.X_train_window_features_names_out_ = X_train_window_features_names_out_
        self.X_train_features_names_out_ = X_train_regressor.columns.to_list()

        self.is_fitted = True
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range_ = {k: v[[0, -1]] for k, v in series_indexes.items()}
        self.index_type_ = type(series_indexes[series_names_in_[0]])
        if isinstance(series_indexes[series_names_in_[0]], pd.DatetimeIndex):
            self.index_freq_ = series_indexes[series_names_in_[0]].freqstr
        else:
            self.index_freq_ = series_indexes[series_names_in_[0]].step

        if exog is not None:
            self.exog_in_ = True
            self.exog_names_in_ = exog_names_in_
            self.exog_type_in_ = type(exog)
            self.exog_dtypes_in_ = exog_dtypes_in_
            self.X_train_exog_names_out_ = X_train_exog_names_out_

        in_sample_residuals_ = {}
        if store_in_sample_residuals:

            residuals = (y_train - self.regressor.predict(X_train_regressor)).to_numpy()
            
            rng = np.random.default_rng(seed=123)
            if self.encoding is not None:
                for col in X_train_series_names_in_:
                    if self.encoding == 'onehot':
                        mask = X_train[col].to_numpy() == 1.
                    else:
                        encoded_value = self.encoding_mapping_[col]
                        mask = X_train['_level_skforecast'].to_numpy() == encoded_value
                    
                    residuals_col = residuals[mask]
                    if len(residuals_col) > 1000:
                        residuals_col = rng.choice(
                                            a       = residuals_col,
                                            size    = 1000,
                                            replace = False
                                        )
                    in_sample_residuals_[col] = residuals_col
            
            if len(residuals) > 1000:
                in_sample_residuals_['_unknown_level'] = rng.choice(
                                                            a       = residuals,
                                                            size    = 1000,
                                                            replace = False
                                                        )
            else:
                in_sample_residuals_['_unknown_level'] = residuals
        else:
            if self.encoding is not None:
                for col in X_train_series_names_in_:
                    in_sample_residuals_[col] = None
            in_sample_residuals_['_unknown_level'] = None

        self.in_sample_residuals_ = in_sample_residuals_

        if store_last_window:
            self.last_window_ = last_window_
        
        set_skforecast_warnings(suppress_warnings, action='default')


    def _create_predict_inputs(
        self,
        steps: int,
        levels: Optional[Union[str, list]] = None,
        last_window: Optional[pd.DataFrame] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
        predict_boot: bool = False,
        use_in_sample_residuals: bool = True,
        check_inputs: bool = True
    ) -> Tuple[pd.DataFrame, Optional[dict], list, pd.Index, Optional[dict]]:
        """
        Create the inputs needed for the first iteration of the prediction 
        process. As this is a recursive process, the last window is updated at 
        each iteration of the prediction process.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        levels : str, list, default `None`
            Time series to be predicted. If `None` all levels whose last window
            ends at the same datetime index will be predicted together.
        last_window : pandas DataFrame, default `None`
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.
        predict_boot : bool, default `False`
            If `True`, residuals are returned to generate bootstrapping predictions.
        use_in_sample_residuals : bool, default `True`
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. If `False`, out of sample 
            residuals are used. In the latter case, the user should have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).
        check_inputs : bool, default `True`
            If `True`, the input is checked for possible warnings and errors 
            with the `check_predict_input` function. This argument is created 
            for internal use and is not recommended to be changed.

        Returns
        -------
        last_window : pandas DataFrame
            Series values used to create the predictors needed in the first 
            iteration of the prediction (t + 1).
        exog_values_dict : dict, None
            Exogenous variable/s included as predictor/s for each series in 
            each step. The keys are the steps and the values are numpy arrays
            where each column is an exog and each row a series (level).
        levels : list
            Names of the series (levels) to be predicted.
        prediction_index : pandas Index
            Index of the predictions.
        residuals : dict, None
            Residuals used to generate bootstrapping predictions for each level 
            in the form `{level: residuals}`. If `predict_boot = False`, 
            `residuals` is `None`.
        
        """

        levels, input_levels_is_list = prepare_levels_multiseries(
            X_train_series_names_in_=self.X_train_series_names_in_, levels=levels
        )

        if self.is_fitted and last_window is None:
            levels, last_window = preprocess_levels_self_last_window_multiseries(
                                      levels               = levels,
                                      input_levels_is_list = input_levels_is_list,
                                      last_window_         = self.last_window_
                                  )
            
        if self.is_fitted and predict_boot:
            residuals = prepare_residuals_multiseries(
                            levels                  = levels,
                            use_in_sample_residuals = use_in_sample_residuals,
                            encoding                = self.encoding,
                            in_sample_residuals_    = self.in_sample_residuals_,
                            out_sample_residuals_   = self.out_sample_residuals_
                        )
        else:
            residuals = None

        if check_inputs:
            check_predict_input(
                forecaster_name  = type(self).__name__,
                steps            = steps,
                is_fitted        = self.is_fitted,
                exog_in_         = self.exog_in_,
                index_type_      = self.index_type_,
                index_freq_      = self.index_freq_,
                window_size      = self.window_size,
                last_window      = last_window,
                exog             = exog,
                exog_type_in_    = self.exog_type_in_,
                exog_names_in_   = self.exog_names_in_,
                interval         = None,
                levels           = levels,
                series_names_in_ = self.series_names_in_,
                encoding         = self.encoding
            )

        last_window = last_window.iloc[
            -self.window_size :, last_window.columns.get_indexer(levels)
        ].copy()
        _, last_window_index = preprocess_last_window(
                                   last_window   = last_window,
                                   return_values = False
                               )
        prediction_index = expand_index(
                               index = last_window_index,
                               steps = steps
                           )

        if exog is not None:
            if isinstance(exog, dict):
                # Empty dataframe to be filled with the exog values of each level
                empty_exog = pd.DataFrame(
                                 data  = {col: pd.Series(dtype=dtype)
                                          for col, dtype in self.exog_dtypes_in_.items()},
                                 index = prediction_index
                             )
            else:
                if isinstance(exog, pd.Series):
                    exog = exog.to_frame()
                
                exog = transform_dataframe(
                           df                = exog,
                           transformer       = self.transformer_exog,
                           fit               = False,
                           inverse_transform = False
                       )
                check_exog_dtypes(exog=exog)
                exog_values = exog.to_numpy()[:steps]
        else:
            exog_values = None
        
        exog_values_all_levels = []
        for level in levels:
            last_window_level = last_window[level].to_numpy()
            last_window_level = transform_numpy(
                array             = last_window_level,
                transformer       = self.transformer_series_.get(level, self.transformer_series_['_unknown_level']),
                fit               = False,
                inverse_transform = False
            )
   
            if self.differentiation is not None:
                if level not in self.differentiator_.keys():
                    self.differentiator_[level] = copy(self.differentiator)
                last_window_level = (
                    self.differentiator_[level].fit_transform(last_window_level)
                )
            
            last_window[level] = last_window_level

            if isinstance(exog, dict):
                # Fill the empty dataframe with the exog values of each level
                # and transform them if necessary
                exog_values = exog.get(level, None)
                if exog_values is not None:
                    if isinstance(exog_values, pd.Series):
                        exog_values = exog_values.to_frame()

                    exog_values = empty_exog.fillna(exog_values)
                    exog_values = transform_dataframe(
                                      df                = exog_values,
                                      transformer       = self.transformer_exog,
                                      fit               = False,
                                      inverse_transform = False
                                  )
                    
                    check_exog_dtypes(
                        exog      = exog_values,
                        series_id = f"`exog` for series '{level}'"
                    )
                    exog_values = exog_values.to_numpy()
                else:
                    exog_values = empty_exog.to_numpy(copy=True)
            
            exog_values_all_levels.append(exog_values)

        if exog is not None:
            # Exog is transformed into a dict where each key is a step and each value
            # is a numpy array where each column is an exog and each row a series
            exog_values_all_levels = np.concatenate(exog_values_all_levels)
            exog_values_dict = {}
            for i in range(steps):
                exog_values_dict[i + 1] = exog_values_all_levels[i::steps, :]
        else:
            exog_values_dict = None

        return last_window, exog_values_dict, levels, prediction_index, residuals


    def _recursive_predict(
        self,
        steps: int,
        levels: list,
        last_window: pd.DataFrame,
        exog_values_dict: Optional[dict] = None,
        residuals: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict n steps for one or multiple levels. It is an iterative process
        in which, each prediction, is used as a predictor for the next step.

        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        levels : list
            Time series to be predicted.
        last_window : pandas DataFrame
            Series values used to create the predictors needed in the first 
            iteration of the prediction (t + 1).
        exog_values_dict : dict, default `None`
            Exogenous variable/s included as predictor/s for each series in 
            each step. The keys are the steps and the values are numpy arrays
            where each column is an exog and each row a series (level).
        residuals : numpy ndarray, default `None`
            Residuals used to generate bootstrapping predictions in the form
            (steps, levels).

        Returns
        -------
        predictions : numpy ndarray
            Predicted values.

        """

        n_levels = len(levels)
        n_lags = len(self.lags) if self.lags is not None else 0
        n_window_features = (
            len(self.X_train_window_features_names_out_)
            if self.window_features is not None
            else 0
        )
        n_autoreg = n_lags + n_window_features
        n_exog = len(self.X_train_exog_names_out_) if exog_values_dict is not None else 0

        if self.encoding is not None:
            if self.encoding == "onehot":
                levels_encoded = np.zeros(
                    (n_levels, len(self.X_train_series_names_in_)), dtype=float
                )
                for i, level in enumerate(levels):
                    if level in self.X_train_series_names_in_:
                        levels_encoded[i, self.X_train_series_names_in_.index(level)] = 1.
            else:
                levels_encoded = np.array(
                    [self.encoding_mapping_.get(level, None) for level in levels],
                    dtype="float64"
                ).reshape(-1, 1)
            levels_encoded_shape = levels_encoded.shape[1]
        else:
            levels_encoded_shape = 0

        features_shape = n_autoreg + levels_encoded_shape + n_exog
        features = np.full(
            shape=(n_levels, features_shape), fill_value=np.nan, order='F', dtype=float
        )
        if self.encoding is not None:
            features[:, n_autoreg: n_autoreg + levels_encoded_shape] = levels_encoded

        predictions = np.full(
            shape=(steps, n_levels), fill_value=np.nan, order='C', dtype=float
        )
        last_window = np.concatenate((last_window.to_numpy(), predictions), axis=0)

        for i in range(steps):
            
            if self.lags is not None:
                features[:, :n_lags] = last_window[
                    -self.lags - (steps - i), :
                ].transpose()
            if self.window_features is not None:
                features[:, n_lags:n_autoreg] = np.concatenate(
                    [
                        wf.transform(last_window[i:-(steps - i), :]) 
                        for wf in self.window_features
                    ],
                    axis=1
                )
            if exog_values_dict is not None:
                features[:, -n_exog:] = exog_values_dict[i + 1]

            pred = self.regressor.predict(features)
            
            if residuals is not None:
                pred += residuals[i, :]
            
            predictions[i, :] = pred 

            # Update `last_window` values. The first position is discarded and 
            # the new prediction is added at the end.
            last_window[-(steps - i), :] = pred

        return predictions


    def create_predict_X(
        self,
        steps: int,
        levels: Optional[Union[str, list]] = None,
        last_window: Optional[pd.DataFrame] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
        suppress_warnings: bool = False
    ) -> dict:
        """
        Create the predictors needed to predict `steps` ahead. As it is a recursive
        process, the predictors are created at each iteration of the prediction 
        process.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        levels : str, list, default `None`
            Time series to be predicted. If `None` all levels whose last window
            ends at the same datetime index will be predicted together.
        last_window : pandas DataFrame, default `None`
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.
        suppress_warnings : bool, default `False`
            If `True`, skforecast warnings will be suppressed during the prediction 
            process. See skforecast.exceptions.warn_skforecast_categories for more
            information.

        Returns
        -------
        X_predict_dict : dict
            Dict in the form `{level: X_predict}` with the predictors for each 
            step and series. The index is the same as the prediction index.
        
        """

        set_skforecast_warnings(suppress_warnings, action='ignore')

        (
            last_window,
            exog_values_dict,
            levels,
            prediction_index,
            _
        ) = self._create_predict_inputs(
            steps        = steps,
            levels       = levels,
            last_window  = last_window,
            exog         = exog
        )
  
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", 
                message="X does not have valid feature names", 
                category=UserWarning
            )
            predictions = self._recursive_predict(
                              steps            = steps,
                              levels           = levels,
                              last_window      = last_window,
                              exog_values_dict = exog_values_dict
                          )
        
        X_predict_dict = {}
        if self.lags is not None:
            idx_lags = np.arange(-steps, 0)[:, None] - self.lags
        len_X_train_series_names_in_ = len(self.X_train_series_names_in_)
        exog_shape = len(self.X_train_exog_names_out_) if exog is not None else 0

        for i, level in enumerate(levels):
            
            X_predict_level = []
            full_predictors = np.concatenate(
                (last_window[level].to_numpy(), predictions[:, i])
            )

            if self.lags is not None:
                X_predict_level.append(full_predictors[idx_lags + len(full_predictors)])

            if self.window_features is not None:
                X_window_features = np.full(
                    shape      = (steps, len(self.X_train_window_features_names_out_)), 
                    fill_value = np.nan, 
                    order      = 'C',
                    dtype      = float
                )
                for i in range(steps):
                    X_window_features[i, :] = np.concatenate(
                        [wf.transform(full_predictors[i:-(steps - i)]) 
                         for wf in self.window_features]
                    )
                X_predict_level.append(X_window_features)

            if self.encoding is not None:
                if self.encoding == 'onehot':
                    level_encoded = np.zeros(
                                        shape = (1, len_X_train_series_names_in_),
                                        dtype = float
                                    )
                    level_encoded[0][self.X_train_series_names_in_.index(level)] = 1.
                else:
                    level_encoded = np.array(
                                        [self.encoding_mapping_.get(level, None)],
                                        dtype = 'float64'
                                    )

                level_encoded = np.tile(level_encoded, (steps, 1))
                X_predict_level.append(level_encoded)
            
            if exog is not None:
                exog_cols = np.full(
                    shape=(steps, exog_shape), fill_value=np.nan, order='C', dtype=float
                )
                for j in range(steps):
                    exog_cols[j, :] = exog_values_dict[j + 1][i, :]
                X_predict_level.append(exog_cols)

            X_predict_dict[level] = pd.DataFrame(
                                        data    = np.concatenate(X_predict_level, axis=1),
                                        columns = self.X_train_features_names_out_,
                                        index   = prediction_index
                                    )
        
        set_skforecast_warnings(suppress_warnings, action='default')

        return X_predict_dict


    def predict(
        self,
        steps: int,
        levels: Optional[Union[str, list]] = None,
        last_window: Optional[pd.DataFrame] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
        suppress_warnings: bool = False,
        check_inputs: bool = True
    ) -> pd.DataFrame:
        """
        Predict n steps ahead. It is an recursive process in which, each prediction,
        is used as a predictor for the next step. Only levels whose last window
        ends at the same datetime index can be predicted together.

        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        levels : str, list, default `None`
            Time series to be predicted. If `None` all levels whose last window
            ends at the same datetime index will be predicted together.
        last_window : pandas DataFrame, default `None`
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, dict, default `None`
            Exogenous variable/s included as predictor/s.
        suppress_warnings : bool, default `False`
            If `True`, skforecast warnings will be suppressed during the prediction 
            process. See skforecast.exceptions.warn_skforecast_categories for more
            information.
        check_inputs : bool, default `True`
            If `True`, the input is checked for possible warnings and errors 
            with the `check_predict_input` function. This argument is created 
            for internal use and is not recommended to be changed.

        Returns
        -------
        predictions : pandas DataFrame
            Predicted values, one column for each level.

        """

        set_skforecast_warnings(suppress_warnings, action='ignore')

        (
            last_window,
            exog_values_dict,
            levels,
            prediction_index,
            _
        ) = self._create_predict_inputs(
            steps        = steps,
            levels       = levels,
            last_window  = last_window,
            exog         = exog,
            check_inputs = check_inputs
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", 
                message="X does not have valid feature names", 
                category=UserWarning
            )
            predictions = self._recursive_predict(
                              steps            = steps,
                              levels           = levels,
                              last_window      = last_window,
                              exog_values_dict = exog_values_dict
                          )
        
        for i, level in enumerate(levels):
            if self.differentiation is not None:
                predictions[:, i] = (
                    self
                    .differentiator_[level]
                    .inverse_transform_next_window(predictions[:, i])
                )

            predictions[:, i] = transform_numpy(
                array             = predictions[:, i],
                transformer       = self.transformer_series_.get(level, self.transformer_series_['_unknown_level']),
                fit               = False,
                inverse_transform = True
            )

        predictions = pd.DataFrame(
                          data    = predictions,
                          index   = prediction_index,
                          columns = levels
                      )
        
        set_skforecast_warnings(suppress_warnings, action='default')

        return predictions


    def predict_bootstrapping(
        self,
        steps: int,
        levels: Optional[Union[str, list]] = None,
        last_window: Optional[pd.DataFrame] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
        n_boot: int = 250,
        random_state: int = 123,
        use_in_sample_residuals: bool = True,
        suppress_warnings: bool = False
    ) -> dict:
        """
        Generate multiple forecasting predictions using a bootstrapping process. 
        By sampling from a collection of past observed errors (the residuals),
        each iteration of bootstrapping generates a different set of predictions. 
        Only levels whose last window ends at the same datetime index can be 
        predicted together. See the Notes section for more information. 
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        levels : str, list, default `None`
            Time series to be predicted. If `None` all levels whose last window
            ends at the same datetime index will be predicted together.
        last_window : pandas DataFrame, default `None`
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, dict, default `None`
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
        suppress_warnings : bool, default `False`
            If `True`, skforecast warnings will be suppressed during the prediction 
            process. See skforecast.exceptions.warn_skforecast_categories for more
            information.

        Returns
        -------
        boot_predictions : dict
            Predictions generated by bootstrapping for each level.
            {level: pandas DataFrame, shape (steps, n_boot)}

        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp3/prediction-intervals.html#prediction-intervals-from-bootstrapped-residuals
        Forecasting: Principles and Practice (3nd ed) Rob J Hyndman and George Athanasopoulos.

        """

        set_skforecast_warnings(suppress_warnings, action='ignore')

        (
            last_window,
            exog_values_dict,
            levels,
            prediction_index,
            residuals
        ) = self._create_predict_inputs(
            steps                   = steps,
            levels                  = levels,
            last_window             = last_window,
            exog                    = exog,
            predict_boot            = True,
            use_in_sample_residuals = use_in_sample_residuals
        )

        n_levels = len(levels)
        rng = np.random.default_rng(seed=random_state)
        sample_residuals = np.full(
                               shape      = (steps, n_boot, n_levels),
                               fill_value = np.nan,
                               order      = 'F',
                               dtype      = float
                           )
        for i, level in enumerate(levels):
            sample_residuals[:, :, i] = rng.choice(
                                            a       = residuals[level],
                                            size    = (steps, n_boot),
                                            replace = True
                                        )
        
        boot_columns = []
        boot_predictions_full = np.full(
                                    shape      = (steps, n_levels, n_boot),
                                    fill_value = np.nan,
                                    order      = 'F',
                                    dtype      = float
                                )
        
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", 
                message="X does not have valid feature names", 
                category=UserWarning
            )
            for i in range(n_boot):

                boot_columns.append(f"pred_boot_{i}")
                boot_predictions_full[:, :, i] = self._recursive_predict(
                    steps            = steps,
                    levels           = levels,
                    last_window      = last_window,
                    exog_values_dict = exog_values_dict,
                    residuals        = sample_residuals[:, i, :]
                )

        boot_predictions = {}
        for i, level in enumerate(levels):

            if self.differentiation is not None:
                boot_predictions_full[:, i, :] = (
                    self.differentiator_[level]
                    .inverse_transform_next_window(boot_predictions_full[:, i, :])
                )
            
            transformer_level = self.transformer_series_.get(
                                    level,
                                    self.transformer_series_['_unknown_level']
                                )
            if transformer_level is not None:
                boot_predictions_full[:, i, :] = np.apply_along_axis(
                    func1d            = transform_numpy,
                    axis              = 0,
                    arr               = boot_predictions_full[:, i, :],
                    transformer       = transformer_level,
                    fit               = False,
                    inverse_transform = True
                )
    
            boot_predictions[level] = pd.DataFrame(
                                          data    = boot_predictions_full[:, i, :],
                                          index   = prediction_index,
                                          columns = boot_columns
                                      )
        
        set_skforecast_warnings(suppress_warnings, action='default')

        return boot_predictions


    def predict_interval(
        self,
        steps: int,
        levels: Optional[Union[str, list]] = None,
        last_window: Optional[pd.DataFrame] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
        interval: list = [5, 95],
        n_boot: int = 250,
        random_state: int = 123,
        use_in_sample_residuals: bool = True,
        suppress_warnings: bool = False
    ) -> pd.DataFrame:
        """
        Iterative process in which, each prediction, is used as a predictor
        for the next step and bootstrapping is used to estimate prediction
        intervals. Both predictions and intervals are returned.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        levels : str, list, default `None`
            Time series to be predicted. If `None` all levels whose last window
            ends at the same datetime index will be predicted together.
        last_window : pandas DataFrame, default `None`
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, dict, default `None`
            Exogenous variable/s included as predictor/s.
        interval : list, default `[5, 95]`
            Confidence of the prediction interval estimated. Sequence of 
            percentiles to compute, which must be between 0 and 100 inclusive. 
            For example, interval of 95% should be as `interval = [2.5, 97.5]`.
        n_boot : int, default `250`
            Number of bootstrapping iterations used to estimate prediction 
            intervals.
        random_state : int, default `123`
            Sets a seed to the random generator, so that boot predictions are always 
            deterministic.
        use_in_sample_residuals : bool, default `True`
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. If `False`, out of sample 
            residuals are used. In the latter case, the user should have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).
        suppress_warnings : bool, default `False`
            If `True`, skforecast warnings will be suppressed during the prediction 
            process. See skforecast.exceptions.warn_skforecast_categories for more
            information.

        Returns
        -------
        predictions : pandas DataFrame
            Values predicted by the forecaster and their estimated interval.

            - level: predictions.
            - level_lower_bound: lower bound of the interval.
            - level_upper_bound: upper bound of the interval.

        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp2/prediction-intervals.html
        Forecasting: Principles and Practice (2nd ed) Rob J Hyndman and
        George Athanasopoulos.

        """

        set_skforecast_warnings(suppress_warnings, action='ignore')

        check_interval(interval=interval)

        boot_predictions = self.predict_bootstrapping(
                               steps                   = steps,
                               levels                  = levels,
                               last_window             = last_window,
                               exog                    = exog,
                               n_boot                  = n_boot,
                               random_state            = random_state,
                               use_in_sample_residuals = use_in_sample_residuals,
                               suppress_warnings       = suppress_warnings
                           )
        
        preds = self.predict(
                    steps             = steps,
                    levels            = levels,
                    last_window       = last_window,
                    exog              = exog,
                    suppress_warnings = suppress_warnings,
                    check_inputs      = False
                )

        interval = np.array(interval) / 100
        predictions = []

        for level in preds.columns:
            preds_interval = boot_predictions[level].quantile(q=interval, axis=1).transpose()
            preds_interval.columns = [f'{level}_lower_bound', f'{level}_upper_bound']
            predictions.append(preds[level])
            predictions.append(preds_interval)

        predictions = pd.concat(predictions, axis=1)
        
        set_skforecast_warnings(suppress_warnings, action='default')

        return predictions


    def predict_quantiles(
        self,
        steps: int,
        levels: Optional[Union[str, list]] = None,
        last_window: Optional[pd.DataFrame] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
        quantiles: list = [0.05, 0.5, 0.95],
        n_boot: int = 250,
        random_state: int = 123,
        use_in_sample_residuals: bool = True,
        suppress_warnings: bool = False
    ) -> pd.DataFrame:
        """
        Calculate the specified quantiles for each step. After generating 
        multiple forecasting predictions through a bootstrapping process, each 
        quantile is calculated for each step.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        levels : str, list, default `None`
            Time series to be predicted. If `None` all levels whose last window
            ends at the same datetime index will be predicted together.
        last_window : pandas DataFrame, default `None`
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, dict, default `None`
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
            prediction error to create quantiles. If `False`, out of sample 
            residuals are used. In the latter case, the user should have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).
        suppress_warnings : bool, default `False`
            If `True`, skforecast warnings will be suppressed during the prediction 
            process. See skforecast.exceptions.warn_skforecast_categories for more
            information.

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

        set_skforecast_warnings(suppress_warnings, action='ignore')

        check_interval(quantiles=quantiles)

        boot_predictions = self.predict_bootstrapping(
                               steps                   = steps,
                               levels                  = levels,
                               last_window             = last_window,
                               exog                    = exog,
                               n_boot                  = n_boot,
                               random_state            = random_state,
                               use_in_sample_residuals = use_in_sample_residuals,
                               suppress_warnings       = suppress_warnings
                           )

        predictions = []

        for level in boot_predictions.keys():
            preds_quantiles = (
                boot_predictions[level].quantile(q=quantiles, axis=1).transpose()
            )
            preds_quantiles.columns = [f'{level}_q_{q}' for q in quantiles]
            predictions.append(preds_quantiles)

        predictions = pd.concat(predictions, axis=1)
        
        set_skforecast_warnings(suppress_warnings, action='default')

        return predictions


    def predict_dist(
        self,
        steps: int,
        distribution: object,
        levels: Optional[Union[str, list]] = None,
        last_window: Optional[pd.DataFrame] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame, dict]] = None,
        n_boot: int = 250,
        random_state: int = 123,
        use_in_sample_residuals: bool = True,
        suppress_warnings: bool = False
    ) -> pd.DataFrame:
        """
        Fit a given probability distribution for each step. After generating 
        multiple forecasting predictions through a bootstrapping process, each 
        step is fitted to the given distribution.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        distribution : Object
            A distribution object from scipy.stats. For example scipy.stats.norm.
        levels : str, list, default `None`
            Time series to be predicted. If `None` all levels whose last window
            ends at the same datetime index will be predicted together.
        last_window : pandas DataFrame, default `None`
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, dict, default `None`
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
        suppress_warnings : bool, default `False`
            If `True`, skforecast warnings will be suppressed during the prediction 
            process. See skforecast.exceptions.warn_skforecast_categories for more
            information.

        Returns
        -------
        predictions : pandas DataFrame
            Distribution parameters estimated for each step and level.

        """

        set_skforecast_warnings(suppress_warnings, action='ignore')

        boot_samples = self.predict_bootstrapping(
                           steps                   = steps,
                           levels                  = levels,
                           last_window             = last_window,
                           exog                    = exog,
                           n_boot                  = n_boot,
                           random_state            = random_state,
                           use_in_sample_residuals = use_in_sample_residuals,
                           suppress_warnings       = suppress_warnings
                       )

        param_names = [
            p for p in inspect.signature(distribution._pdf).parameters if not p == "x"
        ] + ["loc", "scale"]
        predictions = []

        for level in boot_samples.keys():
            param_values = np.apply_along_axis(
                lambda x: distribution.fit(x), axis=1, arr=boot_samples[level]
            )
            level_param_names = [f'{level}_{p}' for p in param_names]

            pred_level = pd.DataFrame(
                             data    = param_values,
                             columns = level_param_names,
                             index   = boot_samples[level].index
                         )

            predictions.append(pred_level)

        predictions = pd.concat(predictions, axis=1)
        
        set_skforecast_warnings(suppress_warnings, action='default')

        return predictions


    def set_params(
        self, 
        params: dict
    ) -> None:
        """
        Set new values to the parameters of the scikit learn model stored in the
        forecaster.
        
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


    def set_out_sample_residuals(
        self, 
        residuals: dict,
        append: bool = True,
        transform: bool = True,
        random_state: int = 123
    ) -> None:
        """
        Set new values to the attribute `out_sample_residuals_`. Out of sample
        residuals are meant to be calculated using observations that did not
        participate in the training process.
        
        Parameters
        ----------
        residuals : dict
            Dictionary of numpy ndarrays with the residuals of each level in the
            form {level: residuals}. If len(residuals) > 1000, only a random 
            sample of 1000 values are stored. Keys must be the same as `levels`.
        append : bool, default `True`
            If `True`, new residuals are added to the once already stored in the
            attribute `out_sample_residuals_`. Once the limit of 1000 values is
            reached, no more values are appended. If False, `out_sample_residuals_`
            is overwritten with the new residuals.
        transform : bool, default `True`
            If `True`, new residuals are transformed using self.transformer_series.
        random_state : int, default `123`
            Sets a seed to the random sampling for reproducible output.
        
        Returns
        -------
        None

        """

        if not isinstance(residuals, dict) or not all(
            isinstance(x, np.ndarray) for x in residuals.values()
        ):
            raise TypeError(
                (f"`residuals` argument must be a dict of numpy ndarrays in the form "
                 "`{level: residuals}`. "
                 f"Got {type(residuals)}.")
            )

        if not self.is_fitted:
            raise NotFittedError(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `set_out_sample_residuals()`.")
            )
        
        if self.encoding is None:
            self.out_sample_residuals_ = {'_unknown_level': None}
            if list(residuals.keys()) != ['_unknown_level']:
                warnings.warn(
                    ("As `encoding` is set to `None`, no distinction between levels "
                     "is made. All residuals are stored in the '_unknown_level' key."),
                     UnknownLevelWarning
                )
                residuals = [v for v in residuals.values() if v is not None]
                if residuals:
                    residuals = np.concatenate(residuals)
                residuals = {'_unknown_level': residuals}
              
        else:
            if self.out_sample_residuals_ is None:
                self.out_sample_residuals_ = {level: None for level in self.series_names_in_}

            if not set(self.out_sample_residuals_.keys()).issubset(set(residuals.keys())):
                warnings.warn(
                    f"Only residuals of levels " 
                    f"{set(self.out_sample_residuals_.keys()).intersection(set(residuals.keys()))} "
                    f"are updated.", IgnoredArgumentWarning
                )
            residuals = {
                k: v 
                for k, v in residuals.items() 
                if k in self.out_sample_residuals_.keys() and k != '_unknown_level'
            }

        for level, value in residuals.items():

            residuals_level = value
            transformer_level = self.transformer_series_[level]
            level_str = f"level '{level}'" if self.encoding is not None else 'all levels'

            if not transform and transformer_level is not None:
                warnings.warn(
                    (f"Argument `transform` is set to `False` but forecaster was "
                     f"trained using a transformer {transformer_level} "
                     f"for {level_str}. Ensure that the new residuals are "
                     f"already transformed or set `transform=True`.")
                )

            if transform and self.transformer_series_ and transformer_level:
                warnings.warn(
                    (f"Residuals will be transformed using the same transformer used "
                     f"when training the forecaster for {level_str} : "
                     f"({transformer_level}). Ensure that the new "
                     f"residuals are on the same scale as the original time series.")
                )
                residuals_level = transform_numpy(
                    array             = residuals_level,
                    transformer       = transformer_level,
                    fit               = False,
                    inverse_transform = False
                )

            if len(residuals_level) > 1000:
                rng = np.random.default_rng(seed=random_state)
                residuals_level = rng.choice(a=residuals_level, size=1000, replace=False)

            if append and self.out_sample_residuals_[level] is not None:
                free_space = max(0, 1000 - len(self.out_sample_residuals_[level]))
                if len(residuals_level) < free_space:
                    residuals_level = np.hstack((
                                          self.out_sample_residuals_[level],
                                          residuals_level
                                      ))
                else:
                    residuals_level = np.hstack((
                                          self.out_sample_residuals_[level],
                                          residuals_level[:free_space]
                                      ))

            self.out_sample_residuals_[level] = residuals_level

        if self.encoding is not None:
            residuals_unknown_level = [
                v for k, v in self.out_sample_residuals_.items() 
                if v is not None and k != '_unknown_level'
            ]
            if residuals_unknown_level:
                residuals_unknown_level = np.concatenate(residuals_unknown_level)
                if len(residuals_unknown_level) > 1000:
                    rng = np.random.default_rng(seed=random_state)
                    residuals_unknown_level = rng.choice(
                                                  a       = residuals_unknown_level,
                                                  size    = 1000,
                                                  replace = False
                                              )
            else:
                residuals_unknown_level = None
            
            self.out_sample_residuals_['_unknown_level'] = residuals_unknown_level


    def get_feature_importances(
        self,
        sort_importance: bool = True
    ) -> pd.DataFrame:
        """
        Return feature importances of the regressor stored in the
        forecaster. Only valid when regressor stores internally the feature
        importances in the attribute `feature_importances_` or `coef_`.

        Parameters
        ----------
        sort_importance: bool, default `True`
            If `True`, sorts the feature importances in descending order.

        Returns
        -------
        feature_importances : pandas DataFrame
            Feature importances associated with each predictor.
        
        """

        if not self.is_fitted:
            raise NotFittedError(
                ("This forecaster is not fitted yet. Call `fit` with appropriate "
                 "arguments before using `get_feature_importances()`.")
            )

        if isinstance(self.regressor, Pipeline):
            estimator = self.regressor[-1]
        else:
            estimator = self.regressor

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
                                      'feature': self.X_train_features_names_out_,
                                      'importance': feature_importances
                                  })
            if sort_importance:
                feature_importances = feature_importances.sort_values(
                                          by='importance', ascending=False
                                      )

        return feature_importances
