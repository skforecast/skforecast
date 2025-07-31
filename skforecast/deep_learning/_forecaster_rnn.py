################################################################################
#                                ForecasterRnn                                 #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

from __future__ import annotations

import sys
import warnings
from copy import deepcopy
from typing import Any

import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler

import skforecast

from ..base import ForecasterBase
from ..exceptions import DataTransformationWarning
from ..utils import (
    check_exog,
    check_interval,
    check_predict_input,
    check_residuals_input,
    check_select_fit_kwargs,
    check_y,
    check_extract_values_and_index,
    expand_index,
    get_exog_dtypes,
    get_style_repr_html,
    initialize_lags,
    initialize_transformer_series,
    input_to_frame,
    prepare_levels_multiseries,
    prepare_steps_direct,
    set_skforecast_warnings,
    transform_dataframe,
    transform_numpy,
    transform_series,
)


# TODO. Test Interval
# TODO. Test Grid search
# TODO. Include window features
# TODO. Include differentiation
# TODO. Include binner residuals
class ForecasterRnn(ForecasterBase):
    """
    This class turns any regressor compatible with the Keras API into a
    Keras RNN multi-series multi-step forecaster. A unique model is created
    to forecast all time steps and series. Keras enables workflows on top of
    either JAX, TensorFlow, or PyTorch. See documentation for more details.

    Parameters
    ----------
    regressor : regressor or pipeline compatible with the Keras API
        An instance of a regressor or pipeline compatible with the Keras API.
    levels : str, list
        Name of one or more time series to be predicted. This determine the series
        the forecaster will be handling. If `None`, all series used during training
        will be available for prediction.
    lags : int, list, numpy ndarray, range
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
    
        - `int`: include lags from 1 to `lags` (included).
        - `list`, `1d numpy ndarray` or `range`: include only lags present in 
        `lags`, all elements must be int.
    transformer_series : transformer (preprocessor), dict, default None
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and 
        inverse_transform. Transformation is applied to each `series` before training 
        the forecaster. ColumnTransformers are not allowed since they do not have 
        inverse_transform method.

        - If single transformer: it is cloned and applied to all series. 
        - If `dict` of transformers: a different transformer can be used for each series.
    transformer_exog : transformer, default None
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    fit_kwargs : dict, default `None`
        Additional arguments to be passed to the `fit` method of the regressor.
    forecaster_id : str, int, default `None`
        Name used as an identifier of the forecaster.

    Attributes
    ----------
    regressor : regressor or pipeline compatible with the Keras API
        An instance of a regressor or pipeline compatible with the Keras API.
        An instance of this regressor is trained for each step. All of them
        are stored in `self.regressors_`.
    levels : str, list
        Name of one or more time series to be predicted. This determine the series
        the forecaster will be handling. If `None`, all series used during training
        will be available for prediction.
    layers_names : list
        Names of the layers in the Keras model used as regressor.
    steps : numpy ndarray
        Future steps the forecaster will predict when using prediction methods.
    max_step : int
        Maximum step the forecaster is able to predict. It is the maximum value
        included in `steps`.
    lags : numpy ndarray
        Lags used as predictors.
    max_lag : int
        Maximum lag included in `lags`.
    window_size : int
        Size of the window needed to create the predictors.
    transformer_series : object, dict
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and
        inverse_transform. Transformation is applied to each `series` before training
        the forecaster. ColumnTransformers are not allowed since they do not have
        inverse_transform method.
    transformer_series_ : dict
        Dictionary with the transformer for each series. It is created cloning the
        objects in `transformer_series` and is used internally to avoid overwriting.
    transformer_exog : transformer
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    last_window_ : pandas DataFrame
        This window represents the most recent data observed by the predictor
        during its training phase. It contains the values needed to predict the
        next step immediately after the training data. These values are stored
        in the original scale of the time series before undergoing any transformations
        or differentiation.
    index_type_ : type
        Type of index of the input used in training.
    index_freq_ : str
        Frequency of Index of the input used in training.
    training_range_: pandas Index
        First and last values of index of the data used during training.
    series_names_in_ : list
        Names of the series used during training.
    n_series_in : int
        Number of series used during training.
    n_levels_out : int
        Number of levels (series) to be predicted by the forecaster.
    exog_in_ : bool
        If the forecaster has been trained using exogenous variable/s.
    exog_names_in_ : list
        Names of the exogenous variables used during training.
    exog_type_in_ : type
        Type of exogenous variable/s used in training.
    exog_dtypes_in_ : dict
        Type of each exogenous variable/s used in training before the transformation
        applied by `transformer_exog`. If `transformer_exog` is not used, it
        is equal to `exog_dtypes_out_`.
    exog_dtypes_out_ : dict
        Type of each exogenous variable/s used in training after the transformation 
        applied by `transformer_exog`. If `transformer_exog` is not used, it 
        is equal to `exog_dtypes_in_`.
    X_train_dim_names_ : dict
        Labels for the multi-dimensional arrays created internally for training.
    y_train_dim_names_ : dict
        Labels for the multi-dimensional arrays created internally for training.
    series_val : pandas DataFrame
        Values of the series used for validation during training.
    exog_val : pandas DataFrame
        Values of the exogenous variables used for validation during training.
    history : dict
        Dictionary with the history of the training of each step. It is created
        internally to avoid overwriting.
    fit_kwargs : dict
        Additional arguments to be passed to the `fit` method of the regressor.
    in_sample_residuals_ : dict
        Residuals of the model when predicting training data. Only stored up 
        to 10_000 values per step in the form `{step: residuals}`. If 
        `transformer_series` is not `None`, residuals are stored in the 
        transformed scale.
    out_sample_residuals_ : dict
        Residuals of the model when predicting non-training data. Only stored up 
        to 10_000 values per step in the form `{step: residuals}`. Use 
        `set_out_sample_residuals()` method to set values. If `transformer_series` 
        is not `None`, residuals are stored in the transformed scale.
    creation_date : str
        Date of creation.
    is_fitted : bool
        Tag to identify if the regressor has been fitted (trained).
    fit_date : str
        Date of last fit.
    keras_backend_ : str
        Keras backend used to fit the forecaster. It can be 'tensorflow', 'torch' 
        or 'jax'.
    skforecast_version : str
        Version of skforecast library used to create the forecaster.
    python_version : str
        Version of python used to create the forecaster.
    forecaster_id : str, int
        Name used as an identifier of the forecaster.
    _probabilistic_mode: str, bool
        Private attribute used to indicate whether the forecaster should perform 
        some calculations during backtesting.
    dropna_from_series : Ignored
        Not used, present here for API consistency by convention.
    encoding : Ignored
        Not used, present here for API consistency by convention.
    differentiation : Ignored
        Not used, present here for API consistency by convention.
    differentiation_max : Ignored
        Not used, present here for API consistency by convention.
    differentiator : Ignored
        Not used, present here for API consistency by convention.
    differentiator_ : Ignored
        Not used, present here for API consistency by convention.

    """

    def __init__(
        self,
        regressor: object,
        levels: str | list[str],
        lags: int | list[int] | np.ndarray[int] | range[int],
        transformer_series: object | dict[str, object] | None = MinMaxScaler(
            feature_range=(0, 1)
        ),
        transformer_exog: object | None = MinMaxScaler(feature_range=(0, 1)),
        fit_kwargs: dict[str, object] | None = {},
        forecaster_id: str | int | None = None
    ) -> None:
        
        self.regressor = deepcopy(regressor)
        self.levels = None
        self.transformer_series = transformer_series
        self.transformer_series_ = None
        self.transformer_exog = transformer_exog
        self.max_lag = None
        self.window_size = None
        self.last_window_ = None
        self.index_type_ = None
        self.index_freq_ = None
        self.training_range_ = None
        self.series_names_in_ = None
        self.exog_names_in_ = None
        self.exog_type_in_ = None
        self.exog_dtypes_in_ = None
        self.X_train_dim_names_ = None
        self.y_train_dim_names_ = None
        self.history_ = None 
        self.is_fitted = False
        self.creation_date = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M:%S")
        self.fit_date = None
        self.keras_backend_ = None
        self.skforecast_version = skforecast.__version__
        self.python_version = sys.version.split(" ")[0]
        self.forecaster_id = forecaster_id
        self._probabilistic_mode = "no_binned"
        self.weight_func = None  # Ignored in this forecaster
        self.source_code_weight_func = None  # Ignored in this forecaster
        self.dropna_from_series = False  # Ignored in this forecaster
        self.encoding = None  # Ignored in this forecaster
        self.differentiation = None  # Ignored in this forecaster
        self.differentiation_max = None  # Ignored in this forecaster
        self.differentiator = None  # Ignored in this forecaster
        self.differentiator_ = None  # Ignored in this forecaster

        layer_init = self.regressor.layers[0]
        layer_end = self.regressor.layers[-1]
        self.layers_names = [layer.name for layer in self.regressor.layers]
        
        self.lags, self.lags_names, self.max_lag = initialize_lags(
            type(self).__name__, lags
        )
        n_lags_regressor = layer_init.output.shape[1]
        if len(self.lags) != n_lags_regressor:
            raise ValueError(
                f"Number of lags ({len(self.lags)}) does not match the number of "
                f"lags expected by the regressor architecture ({n_lags_regressor})."
            )
        
        self.window_size = self.max_lag

        self.steps = np.arange(layer_end.output.shape[1]) + 1
        self.max_step = np.max(self.steps)

        if isinstance(levels, str):
            self.levels = [levels]
        elif isinstance(levels, list):
            self.levels = levels
        else:
            raise TypeError(
                f"`levels` argument must be a string or list. Got {type(levels)}."
            )
        
        self.n_series_in = self.regressor.get_layer('series_input').output.shape[-1]
        self.n_levels_out = self.regressor.get_layer('output_dense_td_layer').output.shape[-1]
        self.exog_in_ = True if "exog_input" in self.layers_names else False
        if self.exog_in_:
            self.n_exog_in = self.regressor.get_layer('exog_input').output.shape[-1]
        else:
            self.n_exog_in = None
            # NOTE: This is needed because the Reshape layer changes the output 
            # shape in _create_and_compile_model_no_exog
            self.n_levels_out = int(self.n_levels_out / self.max_step)

        if not len(self.levels) == self.n_levels_out:
            raise ValueError(
                f"Number of levels ({len(self.levels)}) does not match the number of "
                f"levels expected by the regressor architecture ({self.n_levels_out})."
            )

        self.series_val = None
        self.exog_val = None
        if "series_val" in fit_kwargs:
            if not isinstance(fit_kwargs["series_val"], pd.DataFrame):
                raise TypeError(
                    f"`series_val` must be a pandas DataFrame. "
                    f"Got {type(fit_kwargs['series_val'])}."
                )
            self.series_val = fit_kwargs.pop("series_val")            

            if self.exog_in_:
                if "exog_val" not in fit_kwargs.keys():
                    raise ValueError(
                        "If `series_val` is provided, `exog_val` must also be "
                        "provided using the `fit_kwargs` argument when the "
                        "regressor has exogenous variables."
                    )
                else:
                    if not isinstance(fit_kwargs["exog_val"], (pd.Series, pd.DataFrame)):
                        raise TypeError(
                            f"`exog_val` must be a pandas Series or DataFrame. "
                            f"Got {type(fit_kwargs['exog_val'])}."
                        )
                    self.exog_val = input_to_frame(
                        data=fit_kwargs.pop("exog_val"), input_name='exog_val'
                    )

        self.in_sample_residuals_ = None
        self.in_sample_residuals_by_bin_ = None  # Ignored in this forecaster
        self.out_sample_residuals_ = None
        self.out_sample_residuals_by_bin_ = None  # Ignored in this forecaster

        self.fit_kwargs = check_select_fit_kwargs(
            regressor=self.regressor, fit_kwargs=fit_kwargs
        )

    def __repr__(self) -> str:
        """
        Information displayed when a ForecasterRnn object is printed.
        """

        params = str(self.regressor.get_config())
        compile_config = str(self.regressor.get_compile_config())
        
        (
            _,
            _,
            series_names_in_,
            exog_names_in_,
            transformer_series,
        ) = [
            self._format_text_repr(value) 
            for value in self._preprocess_repr(
                regressor          = None,
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
            f"Layers names: {self.layers_names} \n"
            f"Lags: {self.lags} \n"
            f"Window size: {self.window_size} \n"
            f"Maximum steps to predict: {self.steps} \n"
            f"Series names: {series_names_in_} \n"
            f"Target series (levels): {self.levels} \n"
            f"Exogenous included: {self.exog_in_} \n"
            f"Exogenous names: {exog_names_in_} \n"
            f"Transformer for series: {transformer_series} \n"
            f"Transformer for exog: {self.transformer_exog} \n"
            f"Training range: {self.training_range_.to_list() if self.is_fitted else None} \n"
            f"Training index type: {str(self.index_type_).split('.')[-1][:-2] if self.is_fitted else None} \n"
            f"Training index frequency: {self.index_freq_ if self.is_fitted else None} \n"
            f"Regressor parameters: {params} \n"
            f"Compile parameters: {compile_config} \n"
            f"fit_kwargs: {self.fit_kwargs} \n"
            f"Creation date: {self.creation_date} \n"
            f"Last fit date: {self.fit_date} \n"
            f"Keras backend: {self.keras_backend_} \n"
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

        params = str(self.regressor.get_config())
        compile_config = str(self.regressor.get_compile_config())

        (
            _,
            _,
            series_names_in_,
            exog_names_in_,
            transformer_series,
        ) = self._preprocess_repr(
                regressor          = None,
                series_names_in_   = self.series_names_in_,
                exog_names_in_     = self.exog_names_in_,
                transformer_series = self.transformer_series,
            )

        style, unique_id = get_style_repr_html(self.is_fitted)
        
        content = f"""
        <div class="container-{unique_id}">
            <h2>{type(self).__name__}</h2>
            <details open>
                <summary>General Information</summary>
                <ul>
                    <li><strong>Regressor:</strong> {type(self.regressor).__name__}</li>
                    <li><strong>Layers names:</strong> {self.layers_names}</li>
                    <li><strong>Lags:</strong> {self.lags}</li>
                    <li><strong>Window size:</strong> {self.window_size}</li>
                    <li><strong>Maximum steps to predict:</strong> {self.steps}</li>
                    <li><strong>Exogenous included:</strong> {self.exog_in_}</li>
                    <li><strong>Creation date:</strong> {self.creation_date}</li>
                    <li><strong>Last fit date:</strong> {self.fit_date}</li>
                    <li><strong>Keras backend:</strong> {self.keras_backend_}</li>
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
                    <li><strong>Series names:</strong> {series_names_in_}</li>
                    <li><strong>Target series (levels):</strong> {self.levels}</li>
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
                <summary>Compile Parameters</summary>
                <ul>
                    {compile_config}
                </ul>
            </details>
            <details>
                <summary>Fit Kwargs</summary>
                <ul>
                    {self.fit_kwargs}
                </ul>
            </details>
            <p>
                <a href="https://skforecast.org/{skforecast.__version__}/api/forecasterrnn.html">&#128712 <strong>API Reference</strong></a>
                &nbsp;&nbsp;
                <a href="https://skforecast.org/{skforecast.__version__}/user_guides/forecasting-with-deep-learning-rnn-lstm.html">&#128462 <strong>User Guide</strong></a>
            </p>
        </div>
        """

        # Return the combined style and content
        return style + content

    # TODO: CREATE_LAGS_AND_STEPS
    def _create_lags(
        self, 
        y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transforms a 1d array into a 3d array (X) and a 3d array (y). Each row
        in X is associated with a value of y and it represents the lags that
        precede it.

        Notice that, the returned matrix X_data, contains the lag 1 in the first
        column, the lag 2 in the second column and so on.

        Parameters
        ----------
        y : numpy ndarray
            1d numpy ndarray Training time series.

        Returns
        -------
        X_data : numpy ndarray
            3d numpy ndarray with the lagged values (predictors).
            Shape: (samples - max(lags), len(lags))
        y_data : numpy ndarray
            3d numpy ndarray with the values of the time series related to each
            row of `X_data` for each step.
            Shape: (len(max_step), samples - max(lags))

        """

        n_rows = len(y) - self.window_size - self.max_step + 1  # rows of y_data

        X_data = np.full(
            shape=(n_rows, (self.window_size)), fill_value=np.nan, order="F", dtype=float
        )
        for i, lag in enumerate(range(self.window_size - 1, -1, -1)):
            X_data[:, i] = y[self.window_size - lag - 1 : -(lag + self.max_step)]

        # Get lags index
        X_data = X_data[:, self.lags - 1]

        y_data = np.full(
            shape=(n_rows, self.max_step), fill_value=np.nan, order="F", dtype=float
        )
        for step in range(self.max_step):
            y_data[:, step] = y[self.window_size + step : self.window_size + step + n_rows]

        # Get steps index
        y_data = y_data[:, self.steps - 1]

        return X_data, y_data

    def _create_train_X_y(
        self,
        series: pd.DataFrame,
        exog: pd.Series | pd.DataFrame | None = None
    ) -> tuple[
        np.ndarray, 
        np.ndarray, 
        np.ndarray, 
        dict[int, list], 
        list[str], 
        dict[str, type], 
        dict[str, type]
    ]:
        """
        Create training matrices. The resulting multi-dimensional matrices contain
        the target variable and predictors needed to train the model.

        Parameters
        ----------
        series : pandas DataFrame
            Training time series.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `series` and their indexes must be aligned.

        Returns
        -------
        X_train : numpy ndarray
            Training values (predictors) for each step. The resulting array has
            3 dimensions: (n_observations, n_lags, n_series)
        exog_train: numpy ndarray
            Value of exogenous variables aligned with X_train. (n_observations, n_exog)
        y_train : numpy ndarray
            Values (target) of the time series related to each row of `X_train`.
            The resulting array has 3 dimensions: (n_observations, n_steps, n_levels)
        dimension_names : dict
            Labels for the multi-dimensional arrays created internally for training.
        exog_names_in_ : list
            Names of the exogenous variables included in the training matrices.
        exog_dtypes_in_ : dict
            Type of each exogenous variable/s used in training before the transformation
            applied by `transformer_exog`. If `transformer_exog` is not used, it
            is equal to `exog_dtypes_out_`.
        exog_dtypes_out_ : dict
            Type of each exogenous variable/s used in training after the transformation 
            applied by `transformer_exog`. If `transformer_exog` is not used, it 
            is equal to `exog_dtypes_in_`.

        """

        if not isinstance(series, pd.DataFrame):
            raise TypeError(f"`series` must be a pandas DataFrame. Got {type(series)}.")

        _, series_index = check_extract_values_and_index(
            data=series, data_label="`series`", return_values=False
        )
        series_names_in_ = list(series.columns)
        if not len(series_names_in_) == self.n_series_in:
            raise ValueError(
                f"Number of series in `series` ({len(series_names_in_)}) "
                f"does not match the number of series expected by the model "
                f"architecture ({self.n_series_in})."
            )

        if not set(self.levels).issubset(set(series_names_in_)):
            raise ValueError(
                f"`levels` defined when initializing the forecaster must be "
                f"included in `series` used for training. "
                f"{set(self.levels) - set(series_names_in_)} not found."
            )

        if len(series) < self.window_size + self.max_step:
            raise ValueError(
                f"Minimum length of `series` for training this forecaster is "
                f"{self.window_size + self.max_step}. Reduce the number of "
                f"predicted steps, {self.max_step}, or the maximum "
                f"lag, {self.max_lag}, if no more data is available.\n"
                f"    Length `series`: {len(series)}.\n"
                f"    Max step : {self.max_step}.\n"
                f"    Lags window size: {self.max_lag}."
            )
        
        if exog is None and self.exog_in_:
            raise ValueError(
                "The regressor architecture expects exogenous variables "
                "during training. Provide `exog` argument."
            )

        fit_transformer = False
        if not self.is_fitted:
            fit_transformer = True
            self.transformer_series_ = initialize_transformer_series(
                                           forecaster_name    = type(self).__name__,
                                           series_names_in_   = series_names_in_,
                                           transformer_series = self.transformer_series
                                       )

        # Step 1: Create lags for all columns
        X_train = []
        y_train = []

        # TODO: Add method argument to calculate lags and/or steps
        for serie in series_names_in_:
            x = series[serie]
            check_y(y=x)
            x = transform_series(
                series=x,
                transformer=self.transformer_series_[serie],
                fit=fit_transformer,
                inverse_transform=False,
            )
            X, _ = self._create_lags(x)
            X_train.append(X)

        for level in self.levels:
            y = series[level]
            check_y(y=y)
            y = transform_series(
                series=y,
                transformer=self.transformer_series_[level],
                fit=fit_transformer,
                inverse_transform=False,
            )

            _, y = self._create_lags(y)
            y_train.append(y)

        X_train = np.stack(X_train, axis=2)
        y_train = np.stack(y_train, axis=2)

        train_index = series_index[
            self.max_lag : (len(series_index) - self.max_step + 1)
        ]
        dimension_names = {
            "X_train": {
                0: train_index,
                1: self.lags_names[::-1],
                2: series_names_in_,
            },
            "y_train": {
                0: train_index,
                1: [f"step_{step}" for step in self.steps],
                2: self.levels,
            },
        }

        if exog is not None:

            check_exog(exog=exog, allow_nan=False)
            exog = input_to_frame(data=exog, input_name='exog')
            _, exog_index = check_extract_values_and_index(
                data=exog, data_label='`exog`', ignore_freq=True, return_values=False
            )

            if len(exog.columns) != self.n_exog_in:
                raise ValueError(
                    f"Number of columns in `exog` ({len(exog.columns)}) "
                    f"does not match the number of exogenous variables expected "
                    f"by the model architecture ({self.n_exog_in})."
                )
            
            series_index_no_ws = series_index[self.window_size:]
            len_series = len(series)
            len_series_no_ws = len_series - self.window_size
            len_exog = len(exog)
            if not len_exog == len_series and not len_exog == len_series_no_ws:
                raise ValueError(
                    f"Length of `exog` must be equal to the length of `series` (if "
                    f"index is fully aligned) or length of `series` - `window_size` "
                    f"(if `exog` starts after the first `window_size` values).\n"
                    f"    `exog`                   : ({exog_index[0]} -- {exog_index[-1]})  (n={len_exog})\n"
                    f"    `series`                 : ({series.index[0]} -- {series.index[-1]})  (n={len_series})\n"
                    f"    `series` - `window_size` : ({series_index_no_ws[0]} -- {series_index_no_ws[-1]})  (n={len_series_no_ws})"
                )
            
            exog_names_in_ = exog.columns.to_list()
            if len(set(exog_names_in_) - set(series_names_in_)) != len(exog_names_in_):
                raise ValueError(
                    f"`exog` cannot contain a column named the same as one of "
                    f"the series (column names of series).\n"
                    f"  `series` columns : {series_names_in_}.\n"
                    f"  `exog`   columns : {exog_names_in_}."
                )
            
            print(exog)
            
            exog_n_dim_in = len(exog_names_in_)
            exog_dtypes_in_ = get_exog_dtypes(exog=exog)
            exog = transform_dataframe(
                df=exog,
                transformer=self.transformer_exog,
                fit=fit_transformer,
                inverse_transform=False,
            )
            exog_n_dim_out = len(exog.columns)
            exog_dtypes_out_ = get_exog_dtypes(exog=exog)

            print(exog_n_dim_in)
            print(exog_n_dim_out)
            print(exog)

            if exog_n_dim_in != exog_n_dim_out:
                raise ValueError(
                    f"Number of columns in `exog` after transformation ({exog_n_dim_out}) "
                    f"does not match the number of columns before transformation ({exog_n_dim_in}). "
                    f"The ForecasterRnn does not support transformations that "
                    f"change the number of columns in `exog`. Preprocess `exog` "
                    f"before passing it to the `create_and_compile_model` function."
                )

            if len_exog == len_series:
                if not (exog_index == series_index).all():
                    raise ValueError(
                        "When `exog` has the same length as `series`, the index "
                        "of `exog` must be aligned with the index of `series` "
                        "to ensure the correct alignment of values."
                    )
            else:
                if not (exog_index == series_index_no_ws).all():
                    raise ValueError(
                        "When `exog` doesn't contain the first `window_size` "
                        "observations, the index of `exog` must be aligned with "
                        "the index of `series` minus the first `window_size` "
                        "observations to ensure the correct alignment of values."
                    )
            
            exog_train = []
            for _, exog_name in enumerate(exog.columns):
                _, exog_step = self._create_lags(exog[exog_name])
                exog_train.append(exog_step)
                
            exog_train = np.stack(exog_train, axis=2)

            dimension_names["exog_train"] = {
                0: train_index,
                1: [f"step_{step}" for step in self.steps],
                2: exog.columns.to_list(),
            }
        else:
            exog_train = None
            exog_names_in_ = None
            exog_dtypes_in_ = None
            exog_dtypes_out_ = None
            dimension_names["exog_train"] = {
                0: None,
                1: None,
                2: None
            }

        return (
            X_train, 
            exog_train, 
            y_train, 
            dimension_names,
            exog_names_in_,
            exog_dtypes_in_,
            exog_dtypes_out_
        )
    
    def create_train_X_y(
        self,
        series: pd.DataFrame,
        exog: pd.Series | pd.DataFrame | None = None,
        suppress_warnings: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, list]]:
        """
        Create training matrices. The resulting multi-dimensional matrices contain
        the target variable and predictors needed to train the model.

        Parameters
        ----------
        series : pandas DataFrame
            Training time series.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `series` and their indexes must be aligned.
        suppress_warnings : bool, default False
            If `True`, skforecast warnings will be suppressed during the creation
            of the training matrices. See skforecast.exceptions.warn_skforecast_categories 
            for more information.

        Returns
        -------
        X_train : numpy ndarray
            Training values (predictors) for each step. The resulting array has
            3 dimensions: (n_observations, n_lags, n_series)
        exog_train: numpy ndarray
            Value of exogenous variables aligned with X_train. (n_observations, n_exog)
        y_train : numpy ndarray
            Values (target) of the time series related to each row of `X_train`.
            The resulting array has 3 dimensions: (n_observations, n_steps, n_levels)
        dimension_names : dict
            Labels for the multi-dimensional arrays created internally for training.

        """

        set_skforecast_warnings(suppress_warnings, action='ignore')

        output = self._create_train_X_y(series=series, exog=exog)

        X_train = output[0]
        exog_train = output[1]
        y_train = output[2]
        dimension_names = output[3]
        
        set_skforecast_warnings(suppress_warnings, action='default')

        return X_train, exog_train, y_train, dimension_names

    def fit(
        self,
        series: pd.DataFrame,
        exog: pd.Series | pd.DataFrame = None,
        store_last_window: bool = True,
        store_in_sample_residuals: bool = False,
        random_state: int = 123,
        suppress_warnings: bool = False
    ) -> None:
        """
        Training Forecaster.

        Additional arguments to be passed to the `fit` method of the regressor
        can be added with the `fit_kwargs` argument when initializing the forecaster.

        Parameters
        ----------
        series : pandas DataFrame
            Training time series.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `series` and their indexes must be aligned so
            that series[i] is regressed on exog[i].
        store_last_window : bool, default True
            Whether or not to store the last window (`last_window_`) of training data.
        store_in_sample_residuals : bool, default False
            If `True`, in-sample residuals will be stored in the forecaster object
            after fitting (`in_sample_residuals_` attribute).
        random_state : int, default 123
            Set a seed for the random generator so that the stored sample 
            residuals are always deterministic.
        suppress_warnings : bool, default False
            If `True`, skforecast warnings will be suppressed during the prediction
            process. See skforecast.exceptions.warn_skforecast_categories for more
            information.
        
        Returns
        -------
        None

        """

        set_skforecast_warnings(suppress_warnings, action="ignore")

        # Reset values in case the forecaster has already been fitted.
        self.last_window_ = None
        self.index_type_ = None
        self.index_freq_ = None
        self.training_range_ = None
        self.series_names_in_ = None
        self.exog_names_in_ = None
        self.exog_type_in_ = None
        self.exog_dtypes_in_ = None
        self.exog_dtypes_out_ = None
        self.X_train_dim_names_ = None
        self.y_train_dim_names_ = None
        self.exog_train_dim_names_ = None
        self.in_sample_residuals_ = None
        self.is_fitted = False
        self.fit_date = None
        self.keras_backend_ = keras.backend.backend()
            
        (
            X_train,
            exog_train,
            y_train,
            dimension_names,
            exog_names_in_,
            exog_dtypes_in_,
            exog_dtypes_out_,
        ) = self._create_train_X_y(series=series, exog=exog)

        # NOTE: Need here to avoid refitting the transformer_series_ with the 
        # validation data.
        self.is_fitted = True
        series_names_in_ = dimension_names["X_train"][2]

        if self.keras_backend_ == "torch":
            
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using '{self.keras_backend_}' backend with device: {device}")

            torch_device = torch.device(device)
            X_train = torch.tensor(X_train).to(torch_device)
            y_train = torch.tensor(y_train).to(torch_device)
            if exog_train is not None:
                exog_train = torch.tensor(exog_train).to(torch_device)

        if self.series_val is not None:
            series_val = self.series_val[series_names_in_]
            if exog is not None:
                exog_val = self.exog_val[exog_names_in_]
            else:
                exog_val = None
            
            X_val, exog_val, y_val, *_ = self._create_train_X_y(
                series=series_val, exog=exog_val
            )
            if self.keras_backend_ == "torch":
                X_val = torch.tensor(X_val).to(torch_device)
                y_val = torch.tensor(y_val).to(torch_device)
                if exog_val is not None:
                    exog_val = torch.tensor(exog_val).to(torch_device)

            if self.exog_val is not None:
                history = self.regressor.fit(
                    x=[X_train, exog_train],
                    y=y_train,
                    validation_data=([X_val, exog_val], y_val),
                    **self.fit_kwargs,
                )
            else:
                history = self.regressor.fit(
                    x=X_train,
                    y=y_train,
                    validation_data=(X_val, y_val),
                    **self.fit_kwargs,
                )

        else:
            history = self.regressor.fit(
                x=X_train if exog_train is None else [X_train, exog_train],
                y=y_train,
                **self.fit_kwargs,
            )

        # TODO: Include binning in the forecaster
        self.in_sample_residuals_ = {}
        if store_in_sample_residuals:

            # NOTE: Convert to numpy array if using torch backend
            if self.keras_backend_ == "torch":
                y_train = y_train.detach().cpu().numpy()

            residuals = y_train - self.regressor.predict(
                x=X_train if exog_train is None else [X_train, exog_train], verbose=0
            )

            residuals = np.concatenate(
                [residuals[:, i, :] for i, step in enumerate(self.steps)]
            )

            rng = np.random.default_rng(seed=random_state)
            for i, level in enumerate(self.levels):
                residuals_level = residuals[:, i]
                if len(residuals_level) > 10_000:
                    residuals_level = residuals_level[
                        rng.integers(low=0, high=len(residuals_level), size=10_000)
                    ]
                self.in_sample_residuals_[level] = residuals_level
        else:
            for level in self.levels:
                self.in_sample_residuals_[level] = None
        
        self.series_names_in_ = series_names_in_
        self.X_train_series_names_in_ = series_names_in_
        self.X_train_dim_names_ = dimension_names["X_train"]
        self.y_train_dim_names_ = dimension_names["y_train"]
        self.history_ = history.history

        self.fit_date = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M:%S")
        self.training_range_ = series.index[[0, -1]]
        self.index_type_ = type(series.index)
        if isinstance(series.index, pd.DatetimeIndex):
            self.index_freq_ = series.index.freqstr
        else:
            self.index_freq_ = series.index.step

        if exog is not None:
            # NOTE: self.exog_in_ is determined by the regressor architecture and
            # set during initialization.
            self.exog_names_in_ = exog_names_in_
            self.exog_type_in_ = type(exog)
            self.exog_dtypes_in_ = exog_dtypes_in_
            self.exog_dtypes_out_ = exog_dtypes_out_
            self.exog_train_dim_names_ = dimension_names["exog_train"]
            self.X_train_exog_names_out_ = dimension_names["exog_train"][2]
            self.X_train_features_names_out_ = dimension_names["X_train"][1] + dimension_names["exog_train"][2]
        else:
            self.X_train_features_names_out_ = dimension_names["X_train"][1]

        if store_last_window:
            self.last_window_ = series.iloc[-self.max_lag :, :].copy()

        set_skforecast_warnings(suppress_warnings, action="default")

    def _create_predict_inputs(
        self,
        steps: int | list[int] | None = None,
        levels: str | list[str] | None = None,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        predict_probabilistic: bool = False,
        use_in_sample_residuals: bool = True,
        check_inputs: bool = True
    ) -> tuple[list[np.ndarray], dict[str, dict], list[int], list[str], pd.Index]:
        """
        Create the inputs needed for the prediction process.
        
        Parameters
        ----------
        steps : int, list, default None
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined in the regressor architecture.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as defined in the regressor
            architecture.
        levels : str, list, default None
            Name(s) of the time series to be predicted. It must be included
            in `levels`, defined when initializing the forecaster. If `None`, all
            all series used during training will be available for prediction.
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
        check_inputs : bool, default True
            If `True`, the input is checked for possible warnings and errors 
            with the `check_predict_input` function. This argument is created 
            for internal use and is not recommended to be changed.

        Returns
        -------
        X : list
            List of numpy arrays needed for prediction. The first element
            is the matrix of lags and the second element is the matrix of
            exogenous variables.
        X_predict_dimension_names : dict
            Labels for the multi-dimensional arrays created internally for prediction.
        steps : list
            Steps to predict.
        levels : list
            Levels (series) to predict.
        prediction_index : pandas Index
            Index of the predictions.
        
        """

        levels, _ = prepare_levels_multiseries(
            X_train_series_names_in_=self.levels, levels=levels
        )

        steps = prepare_steps_direct(
                    max_step = self.steps,
                    steps    = steps
                )

        if last_window is None:
            last_window = self.last_window_

        if check_inputs:
            check_predict_input(
                forecaster_name=type(self).__name__,
                steps=steps,
                is_fitted=self.is_fitted,
                exog_in_=self.exog_in_,
                index_type_=self.index_type_,
                index_freq_=self.index_freq_,
                window_size=self.window_size,
                last_window=last_window,
                exog=exog,
                exog_names_in_=self.exog_names_in_,
                interval=None,
                max_steps=self.max_step,
                levels=levels,
                levels_forecaster=self.levels,
                series_names_in_=self.series_names_in_,
            )

            if predict_probabilistic:
                # TODO: Check for this forecaster
                check_residuals_input(
                    forecaster_name              = type(self).__name__,
                    use_in_sample_residuals      = use_in_sample_residuals,
                    in_sample_residuals_         = self.in_sample_residuals_,
                    out_sample_residuals_        = self.out_sample_residuals_,
                    use_binned_residuals         = False,
                    in_sample_residuals_by_bin_  = None,
                    out_sample_residuals_by_bin_ = None,
                    levels                       = self.levels
                )

        last_window = last_window.iloc[
            -self.window_size :, last_window.columns.get_indexer(self.series_names_in_)
        ].copy()

        last_window_values = last_window.to_numpy()
        last_window_matrix = np.full(
            shape=last_window.shape, fill_value=np.nan, order='F', dtype=float
        )
        for idx_series, series in enumerate(self.series_names_in_):
            last_window_series = last_window_values[:, idx_series]
            last_window_series = transform_numpy(
                array=last_window_series,
                transformer=self.transformer_series_[series],
                fit=False,
                inverse_transform=False,
            )
            last_window_matrix[:, idx_series] = last_window_series

        X = [np.reshape(last_window_matrix, (1, self.max_lag, last_window.shape[1]))]
        X_predict_dimension_names = {
            "X_autoreg": {
                0: "batch",
                1: self.lags_names[::-1],
                2: self.X_train_series_names_in_
            }
        }

        if exog is not None:

            exog = input_to_frame(data=exog, input_name='exog')
            exog = transform_dataframe(
                df=exog,
                transformer=self.transformer_exog,
                fit=False,
                inverse_transform=False,
            )

            exog_pred = exog.to_numpy()[:self.max_step]

            # NOTE: This is done to ensure that the exogenous variables
            # have the same number of rows as the maximum step to predict 
            # during backtesting when the last fold is incomplete 
            if len(exog_pred) < self.max_step:
                exog_pred = np.concatenate(
                    [
                        exog_pred,
                        np.full(
                            shape=(self.max_step - len(exog_pred), exog_pred.shape[1]),
                            fill_value=0.,
                            dtype=float
                        )
                    ],
                    axis=0
                )

            exog_pred = np.expand_dims(exog_pred, axis=0)
            X.append(exog_pred)

            X_predict_dimension_names["exog_pred"] = {
                0: "batch",
                1: [f"step_{step}" for step in self.steps],
                2: self.X_train_exog_names_out_
            }
        
        prediction_index = expand_index(
                               index = last_window.index,
                               steps = max(steps)
                           )[np.array(steps) - 1]
        if isinstance(last_window.index, pd.DatetimeIndex) and np.array_equal(
            steps, np.arange(min(steps), max(steps) + 1)
        ):
            prediction_index.freq = last_window.index.freq

        return X, X_predict_dimension_names, steps, levels, prediction_index

    def create_predict_X(
        self,
        steps: int | list[int] | None = None,
        levels: str | list[str] | None = None,
        last_window: pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        suppress_warnings: bool = False,
        check_inputs: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """
        Create the predictors needed to predict `steps` ahead.
        
        Parameters
        ----------
        steps : int, list, default None
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined in the regressor architecture.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as defined in the regressor
            architecture.
        levels : str, list, default None
            Name(s) of the time series to be predicted. It must be included
            in `levels`, defined when initializing the forecaster. If `None`, all
            all series used during training will be available for prediction.
        last_window : pandas DataFrame, default None
            Series values used to create the predictors (lags) needed to 
            predict `steps`.
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.
        suppress_warnings : bool, default False
            If `True`, skforecast warnings will be suppressed during the prediction 
            process. See skforecast.exceptions.warn_skforecast_categories for more
            information.
        check_inputs : bool, default True
            If `True`, the input is checked for possible warnings and errors 
            with the `check_predict_input` function. This argument is created 
            for internal use and is not recommended to be changed.

        Returns
        -------
        X_predict : pandas DataFrame
            Pandas DataFrame with the predictors for each step.
        exog_predict : pandas DataFrame
            Pandas DataFrame with the exogenous variables for each step.
        
        """

        set_skforecast_warnings(suppress_warnings, action='ignore')

        (
            X,
            X_predict_dimension_names,
            *_
        ) = self._create_predict_inputs(
                steps        = steps,
                levels       = levels,
                last_window  = last_window,
                exog         = exog,
                check_inputs = check_inputs
            )

        X_predict = pd.DataFrame(
                        data    = X[0][0], 
                        columns = X_predict_dimension_names['X_autoreg'][2],
                        index   = X_predict_dimension_names['X_autoreg'][1] 
                    )
        
        exog_predict = None
        if self.exog_in_:
            exog_predict = pd.DataFrame(
                data    = X[1][0], 
                columns = X_predict_dimension_names['exog_pred'][2],
                index   = X_predict_dimension_names['exog_pred'][1]
            )
            # NOTE: not needed in this forecaster
            # categorical_features = any(
            #     not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) 
            #     for dtype in set(self.exog_dtypes_out_)
            # )
            # if categorical_features:
            #     X_predict = X_predict.astype(self.exog_dtypes_out_)
        
        if self.transformer_series is not None:
            warnings.warn(
                "The output matrix is in the transformed scale due to the "
                "inclusion of transformations in the Forecaster. "
                "As a result, any predictions generated using this matrix will also "
                "be in the transformed scale. Please refer to the documentation "
                "for more details: "
                "https://skforecast.org/latest/user_guides/training-and-prediction-matrices.html",
                DataTransformationWarning
            )
        
        set_skforecast_warnings(suppress_warnings, action='default')

        return X_predict, exog_predict

    def predict(
        self,
        steps: int | list[int] | None = None,
        levels: str | list[str] | None = None,
        last_window: pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        suppress_warnings: bool = False,
        check_inputs: bool = True
    ) -> pd.DataFrame:
        """
        Predict n steps ahead

        Parameters
        ----------
        steps : int, list, default None
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined in the regressor architecture.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as defined in the regressor
            architecture.
        levels : str, list, default None
            Name(s) of the time series to be predicted. It must be included
            in `levels`, defined when initializing the forecaster. If `None`, all
            all series used during training will be available for prediction.
        last_window : pandas DataFrame, default `None`
            Series values used to create the predictors (lags) needed in the
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.
        suppress_warnings : bool, default `False`
            If `True`, skforecast warnings will be suppressed during the fitting
            process. See skforecast.exceptions.warn_skforecast_categories for more
            information.
        check_inputs : bool, default True
            If `True`, the input is checked for possible warnings and errors 
            with the `check_predict_input` function. This argument is created 
            for internal use and is not recommended to be changed.

        Returns
        -------
        predictions : pandas DataFrame
            Predicted values.

        """

        set_skforecast_warnings(suppress_warnings, action="ignore")

        (
            X,
            _,
            steps,
            levels,
            prediction_index
        ) = self._create_predict_inputs(
                steps        = steps,
                levels       = levels,
                last_window  = last_window,
                exog         = exog,
                check_inputs = check_inputs
            )

        predictions = self.regressor.predict(
            X[0] if not self.exog_in_ else X, verbose=0
        )
        predictions = np.reshape(
            predictions, (predictions.shape[1], predictions.shape[2])
        )[np.array(steps) - 1]

        for i, level in enumerate(self.levels):
            # NOTE: The inverse transformation is applied only if the level
            # is included in the levels to predict.
            if level in levels:
                predictions[:, i] = transform_numpy(
                    array             = predictions[:, i],
                    transformer       = self.transformer_series_[level],
                    fit               = False,
                    inverse_transform = True
                )

        n_steps, n_levels = predictions.shape
        predictions = pd.DataFrame(
            {"level": np.tile(self.levels, n_steps), "pred": predictions.ravel()},
            index = np.repeat(prediction_index, n_levels),
        )
        predictions = predictions[predictions['level'].isin(levels)]

        set_skforecast_warnings(suppress_warnings, action="default")

        return predictions
    
    def _predict_interval_conformal(
        self,
        steps: int | list[int] | None = None,
        levels: str | list[str] | None = None,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        nominal_coverage: float = 0.95,
        use_in_sample_residuals: bool = True
    ) -> pd.DataFrame:
        """
        Generate prediction intervals using the conformal prediction 
        split method [1]_.

        Parameters
        ----------
        steps : int, list, default None
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined in the regressor architecture.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as defined in the regressor
            architecture.
        levels : str, list, default None
            Name(s) of the time series to be predicted. It must be included
            in `levels`, defined when initializing the forecaster. If `None`, all
            all series used during training will be available for prediction.
        last_window : pandas Series, pandas DataFrame, default None
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
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
            X,
            _,
            steps,
            levels,
            prediction_index
        ) = self._create_predict_inputs(
                steps                   = steps,
                levels                  = levels,
                last_window             = last_window,
                exog                    = exog,
                predict_probabilistic   = True,
                use_in_sample_residuals = use_in_sample_residuals
            )

        if use_in_sample_residuals:
            residuals = self.in_sample_residuals_
        else:
            residuals = self.out_sample_residuals_

        predictions = self.regressor.predict(
            X[0] if not self.exog_in_ else X, verbose=0
        )
        predictions = np.reshape(
            predictions, (predictions.shape[1], predictions.shape[2])
        )[np.array(steps) - 1]
        
        n_steps = len(steps)
        n_levels = len(self.levels)
        correction_factor = np.full(
            shape=(n_steps, n_levels), fill_value=np.nan, order='C', dtype=float
        )
        for i, level in enumerate(self.levels):
            # NOTE: The correction factor is calculated only for the levels
            # included in the levels to predict.
            if level in levels:
                correction_factor[:, i] = np.quantile(
                    np.abs(residuals[level]), nominal_coverage
                )
            else:
                correction_factor[:, i] = 0.

        lower_bound = predictions - correction_factor
        upper_bound = predictions + correction_factor

        # NOTE: Create a 3D array with shape (n_levels, intervals, steps)
        predictions = np.array([predictions, lower_bound, upper_bound]).swapaxes(0, 2)

        for i, level in enumerate(self.levels):
            # NOTE: The inverse transformation is applied only if the level
            # is included in the levels to predict.
            if level in levels:
                transformer_level = self.transformer_series_[level]
                if transformer_level is not None:
                    predictions[i, :, :] = np.apply_along_axis(
                        func1d            = transform_numpy,
                        axis              = 0,
                        arr               = predictions[i, :, :],
                        transformer       = transformer_level,
                        fit               = False,
                        inverse_transform = True
                    )
        
        predictions = pd.DataFrame(
                          data    = predictions.swapaxes(0, 1).reshape(-1, 3),
                          index   = np.repeat(prediction_index, len(self.levels)),
                          columns = ["pred", "lower_bound", "upper_bound"]
                      )
        predictions.insert(0, 'level', np.tile(self.levels, n_steps))
        predictions = predictions[predictions['level'].isin(levels)]

        return predictions

    def predict_interval(
        self,
        steps: int | list[int] | None = None,
        levels: str | list[str] | None = None,
        last_window: pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        method: str = 'conformal',
        interval: float | list[float] | tuple[float] = [5, 95],
        use_in_sample_residuals: bool = True,
        suppress_warnings: bool = False,
        n_boot: Any = None,
        use_binned_residuals: Any = None,
        random_state: Any = None,
    ) -> pd.DataFrame:
        """
        Predict n steps ahead and estimate prediction intervals using conformal 
        prediction method. Refer to the References section for additional details.
        
        Parameters
        ----------
        steps : int, list, default None
            Predict n steps. The value of `steps` must be less than or equal to the 
            value of steps defined in the regressor architecture.
        
            - If `int`: Only steps within the range of 1 to int are predicted.
            - If `list`: List of ints. Only the steps contained in the list 
            are predicted.
            - If `None`: As many steps are predicted as defined in the regressor
            architecture.
        levels : str, list, default None
            Name(s) of the time series to be predicted. It must be included
            in `levels`, defined when initializing the forecaster. If `None`, all
            all series used during training will be available for prediction.
        last_window : pandas DataFrame, default None
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, dict, default None
            Exogenous variable/s included as predictor/s.
        method : str, default 'conformal'
            Employs the conformal prediction split method for interval estimation [1]_.
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
        use_in_sample_residuals : bool, default True
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. 
            If `False`, out of sample residuals (calibration) are used. 
            Out-of-sample residuals must be precomputed using Forecaster's
            `set_out_sample_residuals()` method.
        suppress_warnings : bool, default False
            If `True`, skforecast warnings will be suppressed during the prediction 
            process. See skforecast.exceptions.warn_skforecast_categories for more
            information.
        n_boot : Ignored
            Not used, present here for API consistency by convention.
        use_binned_residuals : Ignored
            Not used, present here for API consistency by convention.
        random_state : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        predictions : pandas DataFrame
            Long-format DataFrame with the predictions and the lower and upper
            bounds of the estimated interval. The columns are `level`, `pred`,
            `lower_bound`, `upper_bound`.

        References
        ----------        
        .. [1] MAPIE - Model Agnostic Prediction Interval Estimator.
               https://mapie.readthedocs.io/en/stable/theoretical_description_regression.html#the-split-method
    
        """

        set_skforecast_warnings(suppress_warnings, action='ignore')

        if method == "conformal":

            if isinstance(interval, (list, tuple)):
                check_interval(interval=interval, ensure_symmetric_intervals=True)
                nominal_coverage = (interval[1] - interval[0]) / 100
            else:
                check_interval(alpha=interval, alpha_literal='interval')
                nominal_coverage = interval
            
            predictions = self._predict_interval_conformal(
                              steps                   = steps,
                              levels                  = levels,
                              last_window             = last_window,
                              exog                    = exog,
                              nominal_coverage        = nominal_coverage,
                              use_in_sample_residuals = use_in_sample_residuals
                          )
        else:
            raise ValueError(
                f"Invalid `method` '{method}'. Only 'conformal' is available."
            )
        
        set_skforecast_warnings(suppress_warnings, action='default')

        return predictions

    def plot_history(
        self,
        ax: matplotlib.axes.Axes = None,
        exclude_first_iteration: bool = False,
        **fig_kw,
    ) -> matplotlib.figure.Figure:
        """
        Plots the training and validation loss curves from the given history object stored
        in the ForecasterRnn.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default `None`
            Pre-existing ax for the plot. Otherwise, call matplotlib.pyplot.subplots()
            internally.
        exclude_first_iteration : bool, default `False`
            Whether to exclude the first epoch from the plot.
        fig_kw : dict
            Other keyword arguments are passed to matplotlib.pyplot.subplots().

        Returns
        -------
        fig: matplotlib.figure.Figure
            Matplotlib Figure.
        
        """

        if ax is None:
            fig, ax = plt.subplots(1, 1, **fig_kw)
        else:
            fig = ax.get_figure()

        # Setting up the plot style
        if self.history_ is None:
            raise ValueError("ForecasterRnn has not been fitted yet.")

        # Determine the range of epochs to plot, excluding the first one if specified
        epoch_range = range(1, len(self.history_["loss"]) + 1)
        if exclude_first_iteration:
            epoch_range = range(2, len(self.history_["loss"]) + 1)

        # Plotting training loss
        ax.plot(
            epoch_range,
            self.history_["loss"][
                exclude_first_iteration:
            ],  # Skip first element if specified
            color="b",
            label="Training Loss",
        )

        # Plotting validation loss
        if "val_loss" in self.history_:
            ax.plot(
                epoch_range,
                self.history_["val_loss"][
                    exclude_first_iteration:
                ],  # Skip first element if specified
                color="r",
                label="Validation Loss",
            )

        # Labeling the axes and adding a title
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")

        # Adding a legend
        ax.legend()

        # Displaying grid for better readability
        ax.grid(True, linestyle="--", alpha=0.7)

        # Setting x-axis ticks to integers only
        ax.set_xticks(epoch_range)

        return fig

    # TODO create testing
    def set_params(self, params: dict) -> None:  
        """
        Set new values to the parameters of the scikit-learn model stored in the
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
        self.regressor.reset_states()
        self.regressor.compile(**params)

    def set_fit_kwargs(self, fit_kwargs: dict) -> None:
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

    def set_lags(self, lags: Any) -> None:
        """
        Not used, present here for API consistency by convention.

        Returns
        -------
        None

        """

        pass

    def set_in_sample_residuals(
        self,
        series: pd.DataFrame,
        exog: pd.Series | pd.DataFrame = None,
        random_state: int = 123,
        suppress_warnings: bool = False
    ) -> None:
        """
        Set in-sample residuals in case they were not calculated during the
        training process. 
        
        In-sample residuals are calculated as the difference between the true 
        values and the predictions made by the forecaster using the training 
        data. The following internal attributes are updated:

        + `in_sample_residuals_`: Dictionary containing a numpy ndarray with the
        residuals for each series in the form `{series: residuals}`.

        A total of 10_000 residuals are stored in the attribute `in_sample_residuals_`.
        If the number of residuals is greater than 10_000, a random sample of
        10_000 residuals is stored. The number of residuals stored per bin is
        limited to `10_000 // self.binner.n_bins_`.
        
        Parameters
        ----------
        series : pandas DataFrame
            Training time series.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.
        random_state : int, default 123
            Sets a seed to the random sampling for reproducible output.
        suppress_warnings : bool, default False
            If `True`, skforecast warnings will be suppressed during the sampling 
            process. See skforecast.exceptions.warn_skforecast_categories for more
            information.

        Returns
        -------
        None

        """

        set_skforecast_warnings(suppress_warnings, action='ignore')

        if not self.is_fitted:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `set_in_sample_residuals()`."
            )

        if not isinstance(series, pd.DataFrame):
            raise TypeError(
                f"`series` must be a pandas DataFrame. Got {type(series)}."
            )
        
        series_index_range = check_extract_values_and_index(
            data=series, data_label='`series`', return_values=False
        )[1][[0, -1]]
        if not series_index_range.equals(self.training_range_):
            raise IndexError(
                f"The index range of `series` does not match the range "
                f"used during training. Please ensure the index is aligned "
                f"with the training data.\n"
                f"    Expected : {self.training_range_}\n"
                f"    Received : {series_index_range}"
            )
            
        (
            X_train,
            exog_train,
            y_train,
            dimension_names,
            *_
        ) = self._create_train_X_y(series=series, exog=exog)
        
        if exog is not None:
            X_train_features_names_out_ = dimension_names["X_train"][1] + dimension_names["exog_train"][2]
        else:
            X_train_features_names_out_ = dimension_names["X_train"][1]
        
        if not X_train_features_names_out_ == self.X_train_features_names_out_:
            raise ValueError(
                f"Feature mismatch detected after matrix creation. The features "
                f"generated from the provided data do not match those used during "
                f"the training process. To correctly set in-sample residuals, "
                f"ensure that the same data and preprocessing steps are applied.\n"
                f"    Expected output : {self.X_train_features_names_out_}\n"
                f"    Current output  : {X_train_features_names_out_}"
            )
        
        # TODO: Include binning in the forecaster
        self.in_sample_residuals_ = {}
        residuals = y_train - self.regressor.predict(
            x=X_train if exog_train is None else [X_train, exog_train], verbose=0
        )
        residuals = np.concatenate(
            [residuals[:, i, :] for i, step in enumerate(self.steps)]
        )

        rng = np.random.default_rng(seed=random_state)
        for i, level in enumerate(self.levels):
            residuals_level = residuals[:, i]
            if len(residuals_level) > 10_000:
                residuals_level = residuals_level[
                    rng.integers(low=0, high=len(residuals_level), size=10_000)
                ]
            self.in_sample_residuals_[level] = residuals_level

        set_skforecast_warnings(suppress_warnings, action='default')

    def set_out_sample_residuals(
        self,
        y_true: dict[str, np.ndarray | pd.Series],
        y_pred: dict[str, np.ndarray | pd.Series],
        append: bool = False,
        random_state: int = 123
    ) -> None:
        """
        Set new values to the attribute `out_sample_residuals_`. Out of sample
        residuals are meant to be calculated using observations that did not
        participate in the training process. `y_true` and `y_pred` are expected
        to be in the original scale of the time series. Residuals are calculated
        as `y_true` - `y_pred`, after applying the necessary transformations and
        differentiations if the forecaster includes them (`self.transformer_series`
        and `self.differentiation`).

        A total of 10_000 residuals are stored in the attribute `out_sample_residuals_`.
        If the number of residuals is greater than 10_000, a random sample of
        10_000 residuals is stored.
        
        Parameters
        ----------
        y_true : dict
            Dictionary of numpy ndarrays or pandas Series with the true values of
            the time series for each series in the form {series: y_true}.
        y_pred : dict
            Dictionary of numpy ndarrays or pandas Series with the predicted values
            of the time series for each series in the form {series: y_pred}.
        append : bool, default False
            If `True`, new residuals are added to the once already stored in the
            attribute `out_sample_residuals_`. If after appending the new residuals,
            the limit of 10_000 samples is exceeded, a random sample of 10_000 is
            kept.
        random_state : int, default 123
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
                    f"Got {type(y_true[k])} for series {k}."
                )
            if not isinstance(y_pred[k], (np.ndarray, pd.Series)):
                raise TypeError(
                    f"Values of `y_pred` must be numpy ndarrays or pandas Series. "
                    f"Got {type(y_pred[k])} for series {k}."
                )
            if len(y_true[k]) != len(y_pred[k]):
                raise ValueError(
                    f"`y_true` and `y_pred` must have the same length. "
                    f"Got {len(y_true[k])} and {len(y_pred[k])} for series {k}."
                )
            if isinstance(y_true[k], pd.Series) and isinstance(y_pred[k], pd.Series):
                if not y_true[k].index.equals(y_pred[k].index):
                    raise ValueError(
                        f"When containing pandas Series, elements in `y_true` and "
                        f"`y_pred` must have the same index. Error in series {k}."
                    )
        
        series_to_update = set(y_pred.keys()).intersection(set(self.levels))
        if not series_to_update:
            raise ValueError(
                f"Provided keys in `y_pred` and `y_true` do not match any of the "
                f"target time series in the forecaster, {self.levels}. Residuals "
                f"cannot be updated."
            )
        
        if self.out_sample_residuals_ is None:
            self.out_sample_residuals_ = {level: None for level in self.levels}
        
        rng = np.random.default_rng(seed=random_state)
        for level in series_to_update:

            y_true_level = deepcopy(y_true[level])
            y_pred_level = deepcopy(y_pred[level])
            if not isinstance(y_true_level, np.ndarray):
                y_true_level = y_true_level.to_numpy()
            if not isinstance(y_pred_level, np.ndarray):
                y_pred_level = y_pred_level.to_numpy()

            if self.transformer_series:
                y_true_level = transform_numpy(
                                   array             = y_true_level,
                                   transformer       = self.transformer_series_[level],
                                   fit               = False,
                                   inverse_transform = False
                               )
                y_pred_level = transform_numpy(
                                   array             = y_pred_level,
                                   transformer       = self.transformer_series_[level],
                                   fit               = False,
                                   inverse_transform = False
                               )

            data = pd.DataFrame(
                {'prediction': y_pred_level, 'residuals': y_true_level - y_pred_level}
            ).dropna()
            residuals = data['residuals'].to_numpy()

            out_sample_residuals = self.out_sample_residuals_.get(level, np.array([]))
            out_sample_residuals = (
                np.array([]) 
                if out_sample_residuals is None
                else out_sample_residuals
            )
            if append:
                out_sample_residuals = np.concatenate([out_sample_residuals, residuals])
            else:
                out_sample_residuals = residuals

            if len(out_sample_residuals) > 10_000:
                out_sample_residuals = rng.choice(
                    a       = out_sample_residuals, 
                    size    = 10_000, 
                    replace = False
                )

            self.out_sample_residuals_[level] = out_sample_residuals
