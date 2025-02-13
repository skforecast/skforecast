################################################################################
#                                ForecasterRnn                                 #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

from __future__ import annotations
import inspect
import sys
import warnings
from copy import copy, deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import skforecast
from skforecast.utils.utils import transform_dataframe

from ..base import ForecasterBase
from ..exceptions import (
    DataTransformationWarning,
    IgnoredArgumentWarning,
    MissingValuesWarning,
    ResidualsUsageWarning,
    UnknownLevelWarning
)
from ..utils import (
    check_interval,
    check_predict_input,
    check_select_fit_kwargs,
    check_y,
    check_exog,
    check_exog_dtypes,
    input_to_frame,
    expand_index,
    preprocess_last_window,
    preprocess_y,
    set_skforecast_warnings,
    transform_numpy,
    transform_series,
)


# TODO. Test Interval
# TODO. Test Grid search
class ForecasterRnn(ForecasterBase):
    """
    This class turns any regressor compatible with the Keras API into a
    Keras RNN multi-serie multi-step forecaster. A unique model is created
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
    steps : int, list, str, default `'auto'`
        Steps to be predicted. If 'auto', steps used are from 1 to N, where N is
        extracted from the output layer `self.regressor.layers[-1].output_shape[1]`.
    lags : int, list, numpy ndarray, range, str, default 'auto'
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
    
        -`auto`: lags used are from 1 to N, where N is extracted from the input
        layer `self.regressor.layers[0].input_shape[0][1]`.
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
    steps : numpy ndarray
        Future steps the forecaster will predict when using prediction methods.
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
    exog_in_ : bool
        If the forecaster has been trained using exogenous variable/s.
    exog_names_in_ : list
        Names of the exogenous variables used during training.
    exog_type_in_ : type
        Type of exogenous variable/s used in training.
    exog_dtypes_in_ : dict
        Type of each exogenous variable/s used in training. If `transformer_exog` 
        is used, the dtypes are calculated after the transformation.
    X_train_dim_names_ : dict
        Labels for the multi-dimensional arrays created internally for training.
    y_train_dim_names_ : dict
        Labels for the multi-dimensional arrays created internally for training.
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
    skforcast_version : str
        Version of skforecast library used to create the forecaster.
    python_version : str
        Version of python used to create the forecaster.
    forecaster_id : str, int
        Name used as an identifier of the forecaster.
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
        levels: Union[str, list],
        steps: Optional[Union[int, list, str]] = "auto",
        lags: Optional[Union[int, list, str]] = "auto",
        transformer_series: Optional[Union[object, dict]] = MinMaxScaler(
            feature_range=(0, 1)
        ),
        transformer_exog: Optional[Union[object, dict]] = MinMaxScaler(
            feature_range=(0, 1)
        ),
        fit_kwargs: Optional[dict] = {},
        forecaster_id: Optional[Union[str, int]] = None,
    ) -> None:
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
        self.exog_in_ = False
        self.exog_names_in_ = None
        self.exog_type_in_ = None
        self.exog_dtypes_in_ = None
        self.X_train_dim_names_ = None
        self.y_train_dim_names_ = None
        self.history_ = None 
        self.is_fitted = False
        self.creation_date = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M:%S")
        self.fit_date = None
        self.skforecast_version = skforecast.__version__
        self.python_version = sys.version.split(" ")[0]
        self.forecaster_id = forecaster_id
        self.weight_func = None # Ignored in this forecaster
        self.source_code_weight_func = None # Ignored in this forecaster
        self.dropna_from_series = False  # Ignored in this forecaster
        self.encoding = None  # Ignored in this forecaster
        self.differentiation = None  # Ignored in this forecaster
        self.differentiation_max = None  # Ignored in this forecaster
        self.differentiator = None  # Ignored in this forecaster
        self.differentiator_ = None  # Ignored in this forecaster

        # Infer parameters from the model
        self.regressor = deepcopy(regressor)
        layer_init = self.regressor.layers[0]

        if lags == "auto":
            if keras.__version__ < "3.0":
                self.lags = np.arange(layer_init.input_shape[0][1]) + 1
            else:
                self.lags = np.arange(layer_init.output.shape[1]) + 1

        elif isinstance(lags, int):
            self.lags = np.arange(lags) + 1
        elif isinstance(lags, list):
            self.lags = np.array(lags)
        else:
            raise TypeError(
                f"`lags` argument must be an int, list or 'auto'. Got {type(lags)}."
            )

        self.max_lag = np.max(self.lags)
        self.window_size = self.max_lag

        layer_end = self.regressor.layers[-1]
        if steps == "auto":
            if keras.__version__ < "3.0":
                self.steps = np.arange(layer_end.output_shape[1]) + 1
                self.n_series = layer_end.output_shape[-1]
            else:
                self.steps = np.arange(layer_end.output.shape[1]) + 1
                self.n_series = layer_end.output.shape[-1]
        elif isinstance(steps, int):
            self.steps = np.arange(steps) + 1
        elif isinstance(steps, list):
            self.steps = np.array(steps)
        else:
            raise TypeError(
                f"`steps` argument must be an int, list or 'auto'. Got {type(steps)}."
            )

        self.max_step = np.max(self.steps)
        if keras.__version__ < "3.0":
            self.outputs = layer_end.output_shape[-1]
        else:
            self.outputs = layer_end.output.shape[-1]

        if not isinstance(levels, (list, str, type(None))):
            raise TypeError(
                f"`levels` argument must be a string, list or. Got {type(levels)}."
            )

        if isinstance(levels, str):
            self.levels = [levels]
        elif isinstance(levels, list):
            self.levels = levels
        else:
            raise TypeError(
                f"`levels` argument must be a string or a list. Got {type(levels)}."
            )

        self.series_val = None
        self.exog_val = None

        # TODO check that series_val & exog_val should be both available if exog is not None
        if "series_val" in fit_kwargs:
            self.series_val = fit_kwargs["series_val"]
            fit_kwargs.pop("series_val")

        if "exog_val" in fit_kwargs:
            self.exog_val = fit_kwargs["exog_val"]
            fit_kwargs.pop("exog_val")

        self.in_sample_residuals_ = {step: None for step in self.steps}
        self.out_sample_residuals_ = None

        self.fit_kwargs = check_select_fit_kwargs(
            regressor=self.regressor, fit_kwargs=fit_kwargs
        )

    def __repr__(self) -> str:
        """
        Information displayed when a ForecasterRnn object is printed.
        """

        if isinstance(self.regressor, Pipeline):
            name_pipe_steps = tuple(
                name + "__" for name in self.regressor.named_steps.keys()
            )
            params = {
                key: value
                for key, value in self.regressor.get_params().items()
                if key.startswith(name_pipe_steps)
            }
        else:
            params = self.regressor.get_config()
            compile_config = self.regressor.get_compile_config()

        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Regressor: {self.regressor} \n"
            f"Target series, levels: {self.levels} \n"
            f"Maximum steps predicted: {self.steps} \n"
            f"Lags: {self.lags} \n"
            f"Window size: {self.window_size} \n"
            f"Multivariate series: {self.series_names_in_} \n"
            f"Exogenous included: {self.exog_in_} \n"
            f"Exogenous names: {self.exog_names_in_} \n"
            f"Transformer for series: {self.transformer_series} \n"
            f"Transformer for exog: {self.transformer_exog} \n"
            f"Training range: {self.training_range_.to_list() if self.is_fitted else None} \n"
            f"Training index type: {str(self.index_type_).split('.')[-1][:-2] if self.is_fitted else None} \n"
            f"Training index frequency: {self.index_freq_ if self.is_fitted else None} \n"
            f"Model parameters: {params} \n"
            f"Compile parameters: {compile_config} \n"
            f"fit_kwargs: {self.fit_kwargs} \n"
            f"Creation date: {self.creation_date} \n"
            f"Last fit date: {self.fit_date} \n"
            f"Skforecast version: {self.skforecast_version} \n"
            f"Python version: {self.python_version} \n"
            f"Forecaster id: {self.forecaster_id} \n"
        )

        return info

    def _create_lags(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

        n_splits = len(y) - self.max_lag - self.max_step + 1  # rows of y_data
        if n_splits <= 0:
            raise ValueError(
                (
                    f"The maximum lag ({self.max_lag}) must be less than the length "
                    f"of the series minus the maximum of steps ({len(y) - self.max_step})."
                )
            )

        X_data = np.full(
            shape=(n_splits, (self.max_lag)), fill_value=np.nan, order="F", dtype=float
        )
        for i, lag in enumerate(range(self.max_lag - 1, -1, -1)):
            X_data[:, i] = y[self.max_lag - lag - 1 : -(lag + self.max_step)]

        y_data = np.full(
            shape=(n_splits, self.max_step), fill_value=np.nan, order="F", dtype=float
        )
        for step in range(self.max_step):
            y_data[:, step] = y[self.max_lag + step : self.max_lag + step + n_splits]

        # Get lags index
        X_data = X_data[:, self.lags - 1]

        # Get steps index
        y_data = y_data[:, self.steps - 1]

        return X_data, y_data

    def create_train_X_y(
        self,
        series: pd.DataFrame,
        exog: pd.Series | pd.DataFrame | None = None
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
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
        X_train : np.ndarray
            Training values (predictors) for each step. The resulting array has
            3 dimensions: (time_points, n_lags, n_series)
        exog_train: np.ndarray
            Value of exogenous variables aligned with X_train. (time_points, n_exog)
        y_train : np.ndarray
            Values (target) of the time series related to each row of `X_train`
            The resulting array has 3 dimensions: (time_points, n_steps, n_levels)
        dimension_names : dict
            Labels for the multi-dimensional arrays created internally for training

        """
        # TODO: 
        # check that number of series match self.n_series

        if not isinstance(series, pd.DataFrame):
            raise TypeError(f"`series` must be a pandas DataFrame. Got {type(series)}.")

        series_names_in_ = list(series.columns)

        if not set(self.levels).issubset(set(series.columns)):
            raise ValueError(
                (
                    f"`levels` defined when initializing the forecaster must be included "
                    f"in `series` used for trainng. {set(self.levels) - set(series.columns)} "
                    f"not found."
                )
            )

        if len(series) < self.max_lag + self.max_step:
            raise ValueError(
                (
                    f"Minimum length of `series` for training this forecaster is "
                    f"{self.max_lag + self.max_step}. Got {len(series)}. Reduce the "
                    f"number of predicted steps, {self.max_step}, or the maximum "
                    f"lag, {self.max_lag}, if no more data is available."
                )
            )

        if self.transformer_series is None:
            self.transformer_series_ = {serie: None for serie in series_names_in_}
        elif not isinstance(self.transformer_series, dict):
            self.transformer_series_ = {
                serie: clone(self.transformer_series) for serie in series_names_in_
            }
        else:
            self.transformer_series_ = {serie: None for serie in series_names_in_}
            # Only elements already present in transformer_series_ are updated
            self.transformer_series_.update(
                (k, v)
                for k, v in deepcopy(self.transformer_series).items()
                if k in self.transformer_series_
            )
            series_not_in_transformer_series = set(series.columns) - set(
                self.transformer_series.keys()
            )
            if series_not_in_transformer_series:
                warnings.warn(
                    (
                        f"{series_not_in_transformer_series} not present in "
                        f"`transformer_series`. No transformation is applied to "
                        f"these series."
                    ),
                    IgnoredArgumentWarning,
                )

        # Step 1: Create lags for all columns
        X_train = []
        y_train = []

        for i, serie in enumerate(series.columns):
            x = series[serie]
            check_y(y=x)
            x = transform_series(
                series=x,
                transformer=self.transformer_series_[serie],
                fit=True,
                inverse_transform=False,
            )
            X, _ = self._create_lags(x)
            X_train.append(X)

        for i, serie in enumerate(self.levels):
            y = series[serie]
            check_y(y=y)
            y = transform_series(
                series=y,
                transformer=self.transformer_series_[serie],
                fit=True,
                inverse_transform=False,
            )

            _, y = self._create_lags(y)
            y_train.append(y)

        X_train = np.stack(X_train, axis=2)
        y_train = np.stack(y_train, axis=2)

        train_index = series.index.to_list()[
            self.max_lag : (len(series.index.to_list()) - self.max_step + 1)
        ]
        dimension_names = {
            "X_train": {
                0: train_index,
                1: ["lag_" + str(l) for l in self.lags],
                2: series.columns.to_list(),
            },
            "y_train": {
                0: train_index,
                1: ["step_" + str(l) for l in self.steps],
                2: self.levels,
            },
        }

        if exog is not None:
            check_exog(exog=exog, allow_nan=False)
            exog = input_to_frame(data=exog, input_name='exog')
            #TODO: mejorar el mensaje de error si exog no tiene alguno de los
            # indices en train_index
            exog_train = exog.loc[train_index, :]
            
            exog_train = transform_dataframe(
                df=exog_train,
                transformer=self.transformer_exog,
                fit=True,
                inverse_transform=False,
            )

            dimension_names["exog_train"] = {
                0: train_index,
                1: exog_train.columns.to_list(),
            }
        else:
            exog_train = None
            dimension_names["exog_train"] = {
                0: None,
                1: None,
            }

        return X_train, exog_train, y_train, dimension_names

    def fit(
        self,
        series: pd.DataFrame,
        store_in_sample_residuals: bool = True,
        exog: pd.DataFrame = None,
        suppress_warnings: bool = False,
        store_last_window: str = "Ignored",
    ) -> None:
        """
        Training Forecaster.

        Additional arguments to be passed to the `fit` method of the regressor
        can be added with the `fit_kwargs` argument when initializing the forecaster.

        Parameters
        ----------
        series : pandas DataFrame
            Training time series.
        store_in_sample_residuals : bool, default `True`
            If `True`, in-sample residuals will be stored in the forecaster object
            after fitting (`in_sample_residuals_` attribute).
        exog : Ignored
            Not used, present here for API consistency by convention.
        suppress_warnings : bool, default `False`
            If `True`, skforecast warnings will be suppressed during the prediction
            process. See skforecast.exceptions.warn_skforecast_categories for more
            information.
        store_last_window : Ignored
            Not used, present here for API consistency by convention.
        device : str, default `auto`
            Torch device. if auto `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
        Returns
        -------
        None

        """

        set_skforecast_warnings(suppress_warnings, action="ignore")

        # Reset values in case the forecaster has already been fitted.
        self.index_type_ = None
        self.index_freq_ = None
        self.last_window_ = None
        self.exog_in_ = None
        self.exog_type_in_ = None
        self.exog_dtypes_in_ = None
        self.exog_names_in_ = None
        self.series_names_in_ = None
        self.X_train_dim_names_ = None
        self.y_train_dim_names_ = None
        self.exog_train_dim_names_ = None
        self.in_sample_residuals_ = None
        self.is_fitted = False
        self.training_range_ = None

        X_train, exog_train, y_train, dimension_names = self.create_train_X_y(
            series=series, exog=exog
        )
        self.X_train_dim_names_    = dimension_names["X_train"]
        self.y_train_dim_names_    = dimension_names["y_train"]
        self.exog_train_dim_names_ = dimension_names["exog_train"]
        self.series_names_in_      = dimension_names["X_train"][2]
        self.exog_names_in_        = dimension_names["exog_train"][1]

        if keras.__version__ > "3.0" and keras.backend.backend() == "torch":
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch_device = torch.device(device)

            print(f"Using device: {device}")
            X_train = torch.tensor(X_train).to(torch_device)
            y_train = torch.tensor(y_train).to(torch_device)
            if exog_train is not None:
                exog_train = torch.tensor(exog_train.to_numpy()).to(torch_device)

        if self.series_val is not None:
            series_val = self.series_val[self.series_names_in_]
            if exog is not None:
                # TODO: raise error if exog_val do not exist
                exog_val = self.exog_val[self.exog_names_in_]
            else:
                exog_val = None
            X_val, exog_val, y_val, _ = self.create_train_X_y(
                series=series_val, exog=exog_val
            )
            if keras.__version__ > "3.0" and keras.backend.backend() == "torch":
                X_val = torch.tensor(X_val).to(torch_device)
                y_val = torch.tensor(y_val).to(torch_device)
                if exog_val:
                    exog_val = torch.tensor(exog_val).to(torch_device)

            if exog_val:
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
            if exog_train is not None:
                history = self.regressor.fit(
                    x=[X_train, exog_train],
                    y=y_train,
                    **self.fit_kwargs,
                )
            else:
                history = self.regressor.fit(
                    x=X_train,
                    y=y_train,
                    **self.fit_kwargs,
                )

        self.history_ = history.history
        self.is_fitted = True
        self.fit_date = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M:%S")
        _, y_index = preprocess_y(y=series[self.levels], return_values=False)
        self.training_range_ = y_index[[0, -1]]
        self.index_type_ = type(y_index)
        if isinstance(y_index, pd.DatetimeIndex):
            self.index_freq_ = y_index.freqstr
        else:
            self.index_freq_ = y_index.step

        self.last_window_ = series.iloc[-self.max_lag :, :].copy()

        set_skforecast_warnings(suppress_warnings, action="default")

        if store_in_sample_residuals:
            residuals = y_train - self.regressor.predict(x=X_train, verbose=0)
            self.in_sample_residuals_ = {
                step: residuals[:, i, :] for i, step in enumerate(self.steps)
            }

    def predict(
        self,
        steps: Optional[Union[int, list]] = None,
        levels: Optional[Union[str, list]] = None,
        last_window: Optional[pd.DataFrame] = None,
        exog: pd.Series | pd.DataFrame | None = None,
        suppress_warnings: bool = False,
    ) -> pd.DataFrame:
        """
        Predict n steps ahead

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
        levels : str, list, default `None`
            Name of one or more time series to be predicted. It must be included
            in `levels` defined when initializing the forecaster. If `None`, all
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

        Returns
        -------
        predictions : pandas DataFrame
            Predicted values.

        """

        set_skforecast_warnings(suppress_warnings, action="ignore")

        if levels is None:
            levels = self.levels
        elif isinstance(levels, str):
            levels = [levels]
        if isinstance(steps, int):
            steps = list(np.arange(steps) + 1)
        elif steps is None:
            if isinstance(self.steps, int):
                steps = list(np.arange(self.steps) + 1)
            elif isinstance(self.steps, (list, np.ndarray)):
                steps = list(np.array(self.steps))
        elif isinstance(steps, list):
            steps = list(np.array(steps))

        for step in steps:
            if not isinstance(step, (int, np.int64, np.int32)):
                raise TypeError(
                    (
                        f"`steps` argument must be an int, a list of ints or `None`. "
                        f"Got {type(steps)}."
                    )
                )

        if last_window is None:
            last_window = self.last_window_

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
            exog_type_in_=None,
            exog_names_in_=None,
            interval=None,
            max_steps=self.max_step,
            levels=levels,
            levels_forecaster=self.levels,
            series_names_in_=self.series_names_in_,
        )

        last_window = last_window.iloc[-self.window_size :,].copy()

        for serie_name in self.series_names_in_:
            last_window_serie = transform_series(
                series=last_window[serie_name],
                transformer=self.transformer_series_[serie_name],
                fit=False,
                inverse_transform=False,
            )
            last_window_values, last_window_index = preprocess_last_window(
                last_window=last_window_serie
            )
            last_window.loc[:, serie_name] = last_window_values

        X = np.reshape(last_window.to_numpy(), (1, self.max_lag, last_window.shape[1]))
        if not exog:
            predictions = self.regressor.predict(X, verbose=0)
        else:
            if isinstance(exog, pd.Series):
                exog = exog.to_frame()
            
            exog = exog.loc[:, self.exog_names_in_]
            exog = transform_dataframe(
                        df                = exog,
                        transformer       = self.transformer_exog,
                        fit               = False,
                        inverse_transform = False
                    )
            check_exog_dtypes(exog=exog)
            exog_values = exog.to_numpy()[:steps]
            predictions = self.regressor.predict([X, exog_values], verbose=0)
        
        predictions_reshaped = np.reshape(
                predictions, (predictions.shape[1], predictions.shape[2])
            )


        # if len(self.levels) == 1:
        #     predictions_reshaped = np.reshape(predictions, (predictions.shape[1], 1))
        # else:
        #     predictions_reshaped = np.reshape(
        #         predictions, (predictions.shape[1], predictions.shape[2])
        #     )
        idx = expand_index(index=last_window_index, steps=max(steps))

        predictions = pd.DataFrame(
            data=predictions_reshaped[np.array(steps) - 1],
            columns=self.levels,
            index=idx[np.array(steps) - 1],
        )
        predictions = predictions[levels]

        for serie in levels:
            x = predictions[serie]
            check_y(y=x)
            x = transform_series(
                series=x,
                transformer=self.transformer_series_[serie],
                fit=False,
                inverse_transform=True,
            )
            predictions.loc[:, serie] = x

        # Temporal standardization. Pending full refactoring
        predictions = (
            predictions.melt(var_name="level", value_name="pred", ignore_index=False)
            .reset_index()
            .sort_values(by=["index", "level"])
            .set_index("index")
            .rename_axis(None, axis=0)
        )

        set_skforecast_warnings(suppress_warnings, action="default")

        return predictions

    def predict_bootstrapping(
        self,
        steps: Optional[Union[int, list]] = None,
        last_window: Optional[pd.DataFrame] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
        n_boot: int = 250,
        random_state: int = 123,
        use_in_sample_residuals: bool = True,
        suppress_warnings: bool = False,
        levels: Any = None,
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
        levels : str, list, default None
            Time series to be predicted. If `None` all levels whose last window
            ends at the same datetime index will be predicted together.
        last_window : pandas DataFrame, default None
            Series values used to create the predictors (lags) needed in the
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, dict, default None
            Exogenous variable/s included as predictor/s.
        n_boot : int, default 250
            Number of bootstrapping iterations used to estimate predictions.
        random_state : int, default 123
            Sets a seed to the random generator, so that boot predictions are always
            deterministic.
        use_in_sample_residuals : bool, default True
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. If `False`, out of sample
            residuals are used. In the latter case, the user should have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).
        suppress_warnings : bool, default False
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

        set_skforecast_warnings(suppress_warnings, action="ignore")

        if levels is None:
            levels = self.levels
        elif isinstance(levels, str):
            levels = [levels]

        if isinstance(steps, int):
            steps = list(np.arange(steps) + 1)
        elif steps is None:
            if isinstance(self.steps, int):
                steps = list(np.arange(self.steps) + 1)
            elif isinstance(self.steps, (list, np.ndarray)):
                steps = list(np.array(self.steps))
        elif isinstance(steps, list):
            steps = list(np.array(steps))

        if use_in_sample_residuals:
            if not set(steps).issubset(set(self.in_sample_residuals_.keys())):
                raise ValueError(
                    f"Not `forecaster.in_sample_residuals_` for steps: "
                    f"{set(steps) - set(self.in_sample_residuals_.keys())}."
                )
            residuals = self.in_sample_residuals_
        else:
            if self.out_sample_residuals_ is None:
                raise ValueError(
                    "`forecaster.out_sample_residuals_` is `None`. Use "
                    "`use_in_sample_residuals=True` or the "
                    "`set_out_sample_residuals()` method before predicting."
                )
            else:
                if not set(steps).issubset(set(self.out_sample_residuals_.keys())):
                    raise ValueError(
                        f"Not `forecaster.out_sample_residuals_` for steps: "
                        f"{set(steps) - set(self.out_sample_residuals_.keys())}. "
                        f"Use method `set_out_sample_residuals()`."
                    )
            residuals = self.out_sample_residuals_

        check_residuals = (
            "forecaster.in_sample_residuals_"
            if use_in_sample_residuals
            else "forecaster.out_sample_residuals_"
        )
        for step in steps:
            if residuals[step] is None:
                raise ValueError(
                    f"forecaster residuals for step {step} are `None`. "
                    f"Check {check_residuals}."
                )
            elif any(element is None for element in residuals[step]) or np.any(
                np.isnan(residuals[step])
            ):
                raise ValueError(
                    f"forecaster residuals for step {step} contains `None` "
                    f"or `NaNs` values. Check {check_residuals}."
                )

        if last_window is None:
            last_window = self.last_window_

        check_predict_input(
            forecaster_name=type(self).__name__,
            steps=steps,
            is_fitted=self.is_fitted,
            exog_in_=self.exog_in_,
            index_type_=self.index_type_,
            index_freq_=self.index_freq_,
            window_size=self.window_size,
            last_window=last_window,
            exog=None,
            exog_type_in_=None,
            exog_names_in_=None,
            interval=None,
            max_steps=self.max_step,
            levels=levels,
            levels_forecaster=self.levels,
            series_names_in_=self.series_names_in_,
        )

        last_window = last_window.iloc[-self.window_size :,].copy()

        for serie_name in self.series_names_in_:
            last_window_serie = transform_series(
                series=last_window[serie_name],
                transformer=self.transformer_series_[serie_name],
                fit=False,
                inverse_transform=False,
            )
            last_window_values, last_window_index = preprocess_last_window(
                last_window=last_window_serie
            )
            last_window.loc[:, serie_name] = last_window_values

        X = np.reshape(last_window.to_numpy(), (1, self.max_lag, last_window.shape[1]))
        prediction_index = expand_index(index=last_window_index, steps=max(steps))

        predictions = self.regressor.predict(X, verbose=0)
        predictions = np.squeeze(predictions, axis=0)

        boot_predictions = {}
        boot_columns = [f"pred_boot_{i}" for i in range(n_boot)]
        rng = np.random.default_rng(seed=random_state)

        for j, level in enumerate(levels):
            boot_level = np.tile(predictions[:, j], (n_boot, 1)).T

            for i, step in enumerate(steps):
                sampled_residuals = residuals[step][
                    rng.integers(low=0, high=len(residuals[step]), size=n_boot), j
                ]
                boot_level[i, :] += sampled_residuals

            if self.transformer_series_[level]:
                boot_level = np.apply_along_axis(
                    func1d=transform_numpy,
                    axis=0,
                    arr=boot_level,
                    transformer=self.transformer_series_[level],
                    fit=False,
                    inverse_transform=True,
                )

            boot_level = pd.DataFrame(
                data=boot_level[np.array(steps) - 1],
                index=prediction_index,
                columns=boot_columns,
            )

            boot_predictions[level] = boot_level

        # Temporal standardization. Pending full code refactoring:
        boot_predictions = (
            pd.concat(
                [value.assign(level=key) for key, value in boot_predictions.items()]
            )
            .reset_index()
            .sort_values(by=["index", "level"])
            .set_index("index")
            .rename_axis(None, axis=0)
        )
        boot_predictions = boot_predictions[
            ["level"]
            + [col for col in boot_predictions.columns if col not in ["level", "index"]]
        ]
        if (
            isinstance(boot_predictions.index, pd.DatetimeIndex)
            and boot_predictions.index.freq is not None
        ):
            boot_predictions.index.freq = None

        set_skforecast_warnings(suppress_warnings, action="default")

        return boot_predictions

    def predict_interval(
        self,
        steps: int,
        levels: str | list[str] | None = None,
        last_window: pd.DataFrame | None = None,
        exog: (
            pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None
        ) = None,
        interval: list[float] | tuple[float] = [5, 95],
        n_boot: int = 250,
        random_state: int = 123,
        use_in_sample_residuals: bool = True,
        suppress_warnings: bool = False,
    ) -> pd.DataFrame:
        """
        Iterative process in which, each prediction, is used as a predictor
        for the next step and bootstrapping is used to estimate prediction
        intervals. Both predictions and intervals are returned.

        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        levels : str, list, default None
            Time series to be predicted. If `None` all levels whose last window
            ends at the same datetime index will be predicted together.
        last_window : pandas DataFrame, default None
            Series values used to create the predictors (lags) needed in the
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, dict, default None
            Exogenous variable/s included as predictor/s.
        interval : list, tuple, default `[5, 95]`
            Confidence of the prediction interval estimated. Sequence of
            percentiles to compute, which must be between 0 and 100 inclusive.
            For example, interval of 95% should be as `interval = [2.5, 97.5]`.
        n_boot : int, default 250
            Number of bootstrapping iterations used to estimate prediction
            intervals.
        random_state : int, default 123
            Sets a seed to the random generator, so that boot predictions are always
            deterministic.
        use_in_sample_residuals : bool, default True
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. If `False`, out of sample
            residuals are used. In the latter case, the user should have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).
        suppress_warnings : bool, default False
            If `True`, skforecast warnings will be suppressed during the prediction
            process. See skforecast.exceptions.warn_skforecast_categories for more
            information.

        Returns
        -------
        predictions : pandas DataFrame
            Long-format DataFrame with the predictions and the lower and upper
            bounds of the estimated interval. The columns are `level`, `pred`,
            `lower_bound`, `upper_bound`.

        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp2/prediction-intervals.html
        Forecasting: Principles and Practice (2nd ed) Rob J Hyndman and
        George Athanasopoulos.

        """

        set_skforecast_warnings(suppress_warnings, action="ignore")

        check_interval(interval=interval)

        boot_predictions = self.predict_bootstrapping(
            steps=steps,
            levels=levels,
            last_window=last_window,
            exog=exog,
            n_boot=n_boot,
            random_state=random_state,
            use_in_sample_residuals=use_in_sample_residuals,
            suppress_warnings=suppress_warnings,
        )

        predictions = self.predict(
            steps=steps,
            levels=levels,
            last_window=last_window,
            exog=exog,
            suppress_warnings=suppress_warnings,
            # check_inputs=False
        )

        interval = np.array(interval) / 100
        boot_predictions[["lower_bound", "upper_bound"]] = (
            boot_predictions.iloc[:, 1:].quantile(q=interval, axis=1).transpose()
        )

        predictions = pd.concat(
            [predictions, boot_predictions[["lower_bound", "upper_bound"]]], axis=1
        )

        set_skforecast_warnings(suppress_warnings, action="default")

        return predictions

    def predict_quantiles(
        self,
        steps: int,
        levels: str | list[str] | None = None,
        last_window: pd.DataFrame | None = None,
        exog: (
            pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None
        ) = None,
        quantiles: list[float] | tuple[float] = [0.05, 0.5, 0.95],
        n_boot: int = 250,
        random_state: int = 123,
        use_in_sample_residuals: bool = True,
        suppress_warnings: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate the specified quantiles for each step. After generating
        multiple forecasting predictions through a bootstrapping process, each
        quantile is calculated for each step.

        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        levels : str, list, default None
            Time series to be predicted. If `None` all levels whose last window
            ends at the same datetime index will be predicted together.
        last_window : pandas DataFrame, default None
            Series values used to create the predictors (lags) needed in the
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, dict, default None
            Exogenous variable/s included as predictor/s.
        quantiles : list, tuple, default [0.05, 0.5, 0.95]
            Sequence of quantiles to compute, which must be between 0 and 1
            inclusive. For example, quantiles of 0.05, 0.5 and 0.95 should be as
            `quantiles = [0.05, 0.5, 0.95]`.
        n_boot : int, default 250
            Number of bootstrapping iterations used to estimate quantiles.
        random_state : int, default 123
            Sets a seed to the random generator, so that boot quantiles are always
            deterministic.
        use_in_sample_residuals : bool, default True
            If `True`, residuals from the training data are used as proxy of
            prediction error to create quantiles. If `False`, out of sample
            residuals are used. In the latter case, the user should have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).
        suppress_warnings : bool, default False
            If `True`, skforecast warnings will be suppressed during the prediction
            process. See skforecast.exceptions.warn_skforecast_categories for more
            information.

        Returns
        -------
        predictions : pandas DataFrame
            Long-format DataFrame with the quantiles predicted by the forecaster.
            For example, if `quantiles = [0.05, 0.5, 0.95]`, the columns are
            `level`, `q_0.05`, `q_0.5`, `q_0.95`.

        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp2/prediction-intervals.html
        Forecasting: Principles and Practice (2nd ed) Rob J Hyndman and
        George Athanasopoulos.

        """

        set_skforecast_warnings(suppress_warnings, action="ignore")

        check_interval(quantiles=quantiles)

        predictions = self.predict_bootstrapping(
            steps=steps,
            levels=levels,
            last_window=last_window,
            exog=exog,
            n_boot=n_boot,
            random_state=random_state,
            use_in_sample_residuals=use_in_sample_residuals,
            suppress_warnings=suppress_warnings,
        )

        quantiles_cols = [f"q_{q}" for q in quantiles]
        predictions[quantiles_cols] = (
            predictions.iloc[:, 1:].quantile(q=quantiles, axis=1).transpose()
        )
        predictions = predictions[["level"] + quantiles_cols]

        set_skforecast_warnings(suppress_warnings, action="default")

        return predictions

    def predict_dist(
        self,
        steps: int,
        distribution: object,
        levels: str | list[str] | None = None,
        last_window: pd.DataFrame | None = None,
        exog: (
            pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None
        ) = None,
        n_boot: int = 250,
        random_state: int = 123,
        use_in_sample_residuals: bool = True,
        suppress_warnings: bool = False,
    ) -> pd.DataFrame:
        """
        Fit a given probability distribution for each step. After generating
        multiple forecasting predictions through a bootstrapping process, each
        step is fitted to the given distribution.

        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        distribution : object
            A distribution object from scipy.stats with methods `_pdf` and `fit`.
            For example scipy.stats.norm.
        levels : str, list, default None
            Time series to be predicted. If `None` all levels whose last window
            ends at the same datetime index will be predicted together.
        last_window : pandas DataFrame, default None
            Series values used to create the predictors (lags) needed in the
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, dict, default None
            Exogenous variable/s included as predictor/s.
        n_boot : int, default 250
            Number of bootstrapping iterations used to estimate predictions.
        random_state : int, default 123
            Sets a seed to the random generator, so that boot predictions are always
            deterministic.
        use_in_sample_residuals : bool, default True
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. If `False`, out of sample
            residuals are used. In the latter case, the user should have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).
        suppress_warnings : bool, default False
            If `True`, skforecast warnings will be suppressed during the prediction
            process. See skforecast.exceptions.warn_skforecast_categories for more
            information.

        Returns
        -------
        predictions : pandas DataFrame
            Long-format DataFrame with the parameters of the fitted distribution
            for each step. The columns are `level`, `param_0`, `param_1`, ...,
            `param_n`, where `param_i` are the parameters of the distribution.

        """

        if not hasattr(distribution, "_pdf") or not callable(
            getattr(distribution, "fit", None)
        ):
            raise TypeError(
                "`distribution` must be a valid probability distribution object "
                "from scipy.stats, with methods `_pdf` and `fit`."
            )

        set_skforecast_warnings(suppress_warnings, action="ignore")

        predictions = self.predict_bootstrapping(
            steps=steps,
            levels=levels,
            last_window=last_window,
            exog=exog,
            n_boot=n_boot,
            random_state=random_state,
            use_in_sample_residuals=use_in_sample_residuals,
            suppress_warnings=suppress_warnings,
        )

        param_names = [
            p for p in inspect.signature(distribution._pdf).parameters if not p == "x"
        ] + ["loc", "scale"]

        predictions[param_names] = predictions.iloc[:, 1:].apply(
            lambda x: distribution.fit(x), axis=1, result_type="expand"
        )
        predictions = predictions[["level"] + param_names]

        set_skforecast_warnings(suppress_warnings, action="default")

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

    # def predict_bootstrapping(
    #     self,
    #     steps: Optional[Union[int, list]] = None,
    #     last_window: Optional[pd.DataFrame] = None,
    #     exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    #     n_boot: int = 250,
    #     random_state: int = 123,
    #     use_in_sample_residuals: bool = True,
    #     levels: Any = None,
    # ) -> pd.DataFrame:
    #     """
    #     Generate multiple forecasting predictions using a bootstrapping process.
    #     By sampling from a collection of past observed errors (the residuals),
    #     each iteration of bootstrapping generates a different set of predictions.
    #     See the Notes section for more information.

    #     Parameters
    #     ----------
    #     steps : int, list, None, default `None`
    #         Predict n steps. The value of `steps` must be less than or equal to the
    #         value of steps defined when initializing the forecaster. Starts at 1.

    #             - If `int`: Only steps within the range of 1 to int are predicted.
    #             - If `list`: List of ints. Only the steps contained in the list
    #             are predicted.
    #             - If `None`: As many steps are predicted as were defined at
    #             initialization.
    #     last_window : pandas DataFrame, default `None`
    #         Series values used to create the predictors (lags) needed in the
    #         first iteration of the prediction (t + 1).
    #         If `last_window = None`, the values stored in` self.last_window` are
    #         used to calculate the initial predictors, and the predictions start
    #         right after training data.
    #     exog : pandas Series, pandas DataFrame, default `None`
    #         Exogenous variable/s included as predictor/s.
    #     n_boot : int, default `250`
    #         Number of bootstrapping iterations used to estimate prediction
    #         intervals.
    #     random_state : int, default `123`
    #         Sets a seed to the random generator, so that boot intervals are always
    #         deterministic.
    #     use_in_sample_residuals : bool, default `True`
    #         If `True`, residuals from the training data are used as proxy of
    #         prediction error to create prediction intervals. If `False`, out of
    #         sample residuals are used. In the latter case, the user should have
    #         calculated and stored the residuals within the forecaster (see
    #         `set_out_sample_residuals()`).
    #     levelss : Ignored
    #         Not used, present here for API consistency by convention.

    #     Returns
    #     -------
    #     boot_predictions : pandas DataFrame
    #         Predictions generated by bootstrapping.
    #         Shape: (steps, n_boot)

    #     Notes
    #     -----
    #     More information about prediction intervals in forecasting:
    #     https://otexts.com/fpp3/prediction-intervals.html#prediction-intervals-from-bootstrapped-residuals
    #     Forecasting: Principles and Practice (3nd ed) Rob J Hyndman and George Athanasopoulos.

    #     """

    #     if isinstance(steps, int):
    #         steps = list(np.arange(steps) + 1)
    #     elif steps is None:
    #         steps = list(np.arange(self.steps) + 1)
    #     elif isinstance(steps, list):
    #         steps = list(np.array(steps))

    #     if use_in_sample_residuals:
    #         if not set(steps).issubset(set(self.in_sample_residuals.keys())):
    #             raise ValueError(
    #                 (
    #                     f"Not `forecaster.in_sample_residuals` for steps: "
    #                     f"{set(steps) - set(self.in_sample_residuals.keys())}."
    #                 )
    #             )
    #         residuals = self.in_sample_residuals
    #     else:
    #         if self.out_sample_residuals is None:
    #             raise ValueError(
    #                 (
    #                     "`forecaster.out_sample_residuals` is `None`. Use "
    #                     "`in_sample_residuals=True` or method `set_out_sample_residuals()` "
    #                     "before `predict_interval()`, `predict_bootstrapping()` or "
    #                     "`predict_dist()`."
    #                 )
    #             )
    #         else:
    #             if not set(steps).issubset(set(self.out_sample_residuals.keys())):
    #                 raise ValueError(
    #                     (
    #                         f"Not `forecaster.out_sample_residuals` for steps: "
    #                         f"{set(steps) - set(self.out_sample_residuals.keys())}. "
    #                         f"Use method `set_out_sample_residuals()`."
    #                     )
    #                 )
    #         residuals = self.out_sample_residuals

    #     check_residuals = (
    #         "forecaster.in_sample_residuals"
    #         if in_sample_residuals
    #         else "forecaster.out_sample_residuals"
    #     )
    #     for step in steps:
    #         if residuals[step] is None:
    #             raise ValueError(
    #                 (
    #                     f"forecaster residuals for step {step} are `None`. "
    #                     f"Check {check_residuals}."
    #                 )
    #             )
    #         elif (residuals[step] == None).any():
    #             raise ValueError(
    #                 (
    #                     f"forecaster residuals for step {step} contains `None` values. "
    #                     f"Check {check_residuals}."
    #                 )
    #             )

    #     predictions = self.predict(steps=steps, last_window=last_window, exog=exog)

    #     # Predictions must be in the transformed scale before adding residuals
    #     predictions = transform_dataframe(
    #         df=predictions,
    #         transformer=self.transformer_series_[self.levels],
    #         fit=False,
    #         inverse_transform=False,
    #     )
    #     boot_predictions = pd.concat([predictions] * n_boot, axis=1)
    #     boot_predictions.columns = [f"pred_boot_{i}" for i in range(n_boot)]

    #     for i, step in enumerate(steps):
    #         rng = np.random.default_rng(seed=random_state)
    #         sample_residuals = rng.choice(a=residuals[step], size=n_boot, replace=True)
    #         boot_predictions.iloc[i, :] = boot_predictions.iloc[i, :] + sample_residuals

    #     if self.transformer_series_[self.levels]:
    #         for col in boot_predictions.columns:
    #             boot_predictions[col] = transform_series(
    #                 series=boot_predictions[col],
    #                 transformer=self.transformer_series_[self.levels],
    #                 fit=False,
    #                 inverse_transform=True,
    #             )

    #     return boot_predictions

    # def predict_interval(
    #     self,
    #     steps: Optional[Union[int, list]] = None,
    #     last_window: Optional[pd.DataFrame] = None,
    #     exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    #     interval: list = [5, 95],
    #     n_boot: int = 250,
    #     random_state: int = 123,
    #     in_sample_residuals: bool = True,
    #     levelss: Any = None,
    # ) -> pd.DataFrame:
    #     """
    #     Bootstrapping based prediction intervals.
    #     Both predictions and intervals are returned.

    #     Parameters
    #     ----------
    #     steps : int, list, None, default `None`
    #         Predict n steps. The value of `steps` must be less than or equal to the
    #         value of steps defined when initializing the forecaster. Starts at 1.

    #             - If `int`: Only steps within the range of 1 to int are predicted.
    #             - If `list`: List of ints. Only the steps contained in the list
    #             are predicted.
    #             - If `None`: As many steps are predicted as were defined at
    #             initialization.
    #     last_window : pandas DataFrame, default `None`
    #         Series values used to create the predictors (lags) needed in the
    #         first iteration of the prediction (t + 1).
    #         If `last_window = None`, the values stored in` self.last_window` are
    #         used to calculate the initial predictors, and the predictions start
    #         right after training data.
    #     exog : pandas Series, pandas DataFrame, default `None`
    #         Exogenous variable/s included as predictor/s.
    #     interval : list, tuple, default `[5, 95]`
    #         Confidence of the prediction interval estimated. Sequence of
    #         percentiles to compute, which must be between 0 and 100 inclusive.
    #         For example, interval of 95% should be as `interval = [2.5, 97.5]`.
    #     n_boot : int, default `250`
    #         Number of bootstrapping iterations used to estimate prediction
    #         intervals.
    #     random_state : int, default `123`
    #         Sets a seed to the random generator, so that boot intervals are always
    #         deterministic.
    #     in_sample_residuals : bool, default `True`
    #         If `True`, residuals from the training data are used as proxy of
    #         prediction error to create prediction intervals. If `False`, out of
    #         sample residuals are used. In the latter case, the user should have
    #         calculated and stored the residuals within the forecaster (see
    #         `set_out_sample_residuals()`).
    #     levelss : Ignored
    #         Not used, present here for API consistency by convention.

    #     Returns
    #     -------
    #     predictions : pandas DataFrame
    #         Values predicted by the forecaster and their estimated interval.

    #             - pred: predictions.
    #             - lower_bound: lower bound of the interval.
    #             - upper_bound: upper bound of the interval.

    #     Notes
    #     -----
    #     More information about prediction intervals in forecasting:
    #     https://otexts.com/fpp2/prediction-intervals.html
    #     Forecasting: Principles and Practice (2nd ed) Rob J Hyndman and
    #     George Athanasopoulos.

    #     """

    #     check_interval(interval=interval)

    #     predictions = self.predict(steps=steps, last_window=last_window, exog=exog)

    #     boot_predictions = self.predict_bootstrapping(
    #         steps=steps,
    #         last_window=last_window,
    #         exog=exog,
    #         n_boot=n_boot,
    #         random_state=random_state,
    #         in_sample_residuals=in_sample_residuals,
    #     )

    #     interval = np.array(interval) / 100
    #     predictions_interval = boot_predictions.quantile(q=interval, axis=1).transpose()
    #     predictions_interval.columns = ["lower_bound", "upper_bound"]
    #     predictions = pd.concat((predictions, predictions_interval), axis=1)

    #     return predictions

    # def predict_dist(
    #     self,
    #     distribution: object,
    #     steps: Optional[Union[int, list]] = None,
    #     last_window: Optional[pd.DataFrame] = None,
    #     exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    #     n_boot: int = 250,
    #     random_state: int = 123,
    #     in_sample_residuals: bool = True,
    #     levelss: Any = None,
    # ) -> pd.DataFrame:
    #     """
    #     Fit a given probability distribution for each step. After generating
    #     multiple forecasting predictions through a bootstrapping process, each
    #     step is fitted to the given distribution.

    #     Parameters
    #     ----------
    #     distribution : Object
    #         A distribution object from scipy.stats.
    #     steps : int, list, None, default `None`
    #         Predict n steps. The value of `steps` must be less than or equal to the
    #         value of steps defined when initializing the forecaster. Starts at 1.

    #             - If `int`: Only steps within the range of 1 to int are predicted.
    #             - If `list`: List of ints. Only the steps contained in the list
    #             are predicted.
    #             - If `None`: As many steps are predicted as were defined at
    #             initialization.
    #     last_window : pandas DataFrame, default `None`
    #         Series values used to create the predictors (lags) needed in the
    #         first iteration of the prediction (t + 1).
    #         If `last_window = None`, the values stored in` self.last_window` are
    #         used to calculate the initial predictors, and the predictions start
    #         right after training data.
    #     exog : pandas Series, pandas DataFrame, default `None`
    #         Exogenous variable/s included as predictor/s.
    #     n_boot : int, default `250`
    #         Number of bootstrapping iterations used to estimate prediction
    #         intervals.
    #     random_state : int, default `123`
    #         Sets a seed to the random generator, so that boot intervals are always
    #         deterministic.
    #     in_sample_residuals : bool, default `True`
    #         If `True`, residuals from the training data are used as proxy of
    #         prediction error to create prediction intervals. If `False`, out of
    #         sample residuals are used. In the latter case, the user should have
    #         calculated and stored the residuals within the forecaster (see
    #         `set_out_sample_residuals()`).
    #     levelss : Ignored
    #         Not used, present here for API consistency by convention.

    #     Returns
    #     -------
    #     predictions : pandas DataFrame
    #         Distribution parameters estimated for each step.

    #     """

    #     boot_samples = self.predict_bootstrapping(
    #         steps=steps,
    #         last_window=last_window,
    #         exog=exog,
    #         n_boot=n_boot,
    #         random_state=random_state,
    #         in_sample_residuals=in_sample_residuals,
    #     )

    #     param_names = [
    #         p for p in inspect.signature(distribution._pdf).parameters if not p == "x"
    #     ] + ["loc", "scale"]
    #     param_values = np.apply_along_axis(
    #         lambda x: distribution.fit(x), axis=1, arr=boot_samples
    #     )
    #     predictions = pd.DataFrame(
    #         data=param_values, columns=param_names, index=boot_samples.index
    #     )

    #     return predictions

    def set_params(self, params: dict) -> None:  # TODO testear
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

    # def set_out_sample_residuals(
    #     self,
    #     residuals: np.ndarray,
    #     append: bool = True,
    #     transform: bool = True,
    #     random_state: int = 123,
    # ) -> None:
    #     """
    #     Set new values to the attribute `out_sample_residuals`. Out of sample
    #     residuals are meant to be calculated using observations that did not
    #     participate in the training process.

    #     Parameters
    #     ----------
    #     residuals : numpy ndarray
    #         Values of residuals. If len(residuals) > 1000, only a random sample
    #         of 1000 values are stored.
    #     append : bool, default `True`
    #         If `True`, new residuals are added to the once already stored in the
    #         attribute `out_sample_residuals`. Once the limit of 1000 values is
    #         reached, no more values are appended. If False, `out_sample_residuals`
    #         is overwritten with the new residuals.
    #     transform : bool, default `True`
    #         If `True`, new residuals are transformed using self.transformer_y.
    #     random_state : int, default `123`
    #         Sets a seed to the random sampling for reproducible output.

    #     Returns
    #     -------
    #     None

    #     """

    #     if not isinstance(residuals, np.ndarray):
    #         raise TypeError(
    #             f"`residuals` argument must be `numpy ndarray`. Got {type(residuals)}."
    #         )

    #     if not transform and self.transformer_y is not None:
    #         warnings.warn(
    #             (
    #                 f"Argument `transform` is set to `False` but forecaster was trained "
    #                 f"using a transformer {self.transformer_y}. Ensure that the new residuals "
    #                 f"are already transformed or set `transform=True`."
    #             )
    #         )

    #     if transform and self.transformer_y is not None:
    #         warnings.warn(
    #             (
    #                 f"Residuals will be transformed using the same transformer used "
    #                 f"when training the forecaster ({self.transformer_y}). Ensure that the "
    #                 f"new residuals are on the same scale as the original time series."
    #             )
    #         )

    #         residuals = transform_series(
    #             series=pd.Series(residuals, name="residuals"),
    #             transformer=self.transformer_y,
    #             fit=False,
    #             inverse_transform=False,
    #         ).to_numpy()

    #     if len(residuals) > 1000:
    #         rng = np.random.default_rng(seed=random_state)
    #         residuals = rng.choice(a=residuals, size=1000, replace=False)

    #     if append and self.out_sample_residuals is not None:
    #         free_space = max(0, 1000 - len(self.out_sample_residuals))
    #         if len(residuals) < free_space:
    #             residuals = np.hstack((self.out_sample_residuals, residuals))
    #         else:
    #             residuals = np.hstack(
    #                 (self.out_sample_residuals, residuals[:free_space])
    #             )

    #     self.out_sample_residuals = residuals
