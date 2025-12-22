################################################################################
#                             ForecasterStats                                  #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import warnings
import sys
import pandas as pd
from copy import copy
import textwrap
import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from .. import __version__
from ..exceptions import IgnoredArgumentWarning
from ..utils import (
    check_y,
    check_exog,
    check_exog_dtypes,
    check_predict_input,
    expand_index,
    get_exog_dtypes,
    transform_series,
    transform_numpy,
    transform_dataframe,
    get_style_repr_html,
    initialize_estimator
)


class ForecasterStats():
    """
    This class turns statistical model into a Forecaster compatible with the skforecast API.
    Supported statistical models are: skforecast.stats.Sarimax, skforecast.stats.ARAR,
    aeon.forecasting.stats.ARIMA and aeon.forecasting.stats.ETS.
    
    Parameters
    ----------
    estimator : object
        A statistical model instance. Supported models are:
        
        - skforecast.stats.Sarimax
        - skforecast.stats.ARAR
        - aeon.forecasting.stats.ARIMA
        - aeon.forecasting.stats.ETS
    transformer_y : object transformer (preprocessor), default None
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
        The transformation is applied to `y` before training the forecaster. 
    transformer_exog : object transformer (preprocessor), default None
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    fit_kwargs : dict, default None
        Additional arguments to be passed to the `fit` method of the estimator. 
        When using the skforecast Sarimax model, the fit kwargs should be passed 
        using the model parameter `sm_fit_kwargs` and not this one.
    forecaster_id : str, int, default None
        Name used as an identifier of the forecaster.
    regressor : estimator or pipeline compatible with the Keras API
        **Deprecated**, alias for `estimator`.
    
    Attributes
    ----------
    estimator : object
        A statistical model instance.
    params: dict
        Parameters of the statistical model.
    transformer_y : object transformer (preprocessor)
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
        The transformation is applied to `y` before training the forecaster.
    transformer_exog : object transformer (preprocessor)
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    window_size : int
        Not used, present here for API consistency by convention.
    last_window_ : pandas Series
        Last window the forecaster has seen during training. It stores the
        values needed to predict the next `step` immediately after the training data. In the
        statistical models it stores all the training data.
    extended_index_ : pandas Index
        Index the forecaster has seen during training.
    index_type_ : type
        Type of index of the input used in training.
    index_freq_ : str
        Frequency of Index of the input used in training.
    training_range_ : pandas Index
        First and last values of index of the data used during training.
    series_name_in_ : str
        Names of the series provided by the user during training.
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
    X_train_exog_names_out_ : list
        Names of the exogenous variables included in the matrix `X_train` created
        internally for training. It can be different from `exog_names_in_` if
        some exogenous variables are transformed during the training process.
    fit_kwargs : dict
        Additional arguments to be passed to the `fit` method of the estimator.
    creation_date : str
        Date of creation.
    is_fitted : bool
        Tag to identify if the estimator has been fitted (trained).
    fit_date : str
        Date of last fit.
    valid_estimator_types : list
        List of valid estimator types.
    skforecast_version : str
        Version of skforecast library used to create the forecaster.
    python_version : str
        Version of python used to create the forecaster.
    forecaster_id : str, int
        Name used as an identifier of the forecaster.
    __skforecast_tags__ : dict
        Tags associated with the forecaster.
    
    """
    
    def __init__(
        self,
        estimator: object = None,
        transformer_y: object | None = None,
        transformer_exog: object | None = None,
        fit_kwargs: dict[str, object] | None = None,
        forecaster_id: str | int | None = None,
        regressor: object = None
    ) -> None:
        
        self.estimator               = copy(initialize_estimator(estimator, regressor))
        self.estimator_type          = None
        self.transformer_y           = transformer_y
        self.transformer_exog        = transformer_exog
        self.window_size             = 1
        self.last_window_            = None
        self.extended_index_         = None
        self.index_type_             = None
        self.index_freq_             = None
        self.training_range_         = None
        self.series_name_in_         = None
        self.exog_in_                = False
        self.exog_names_in_          = None
        self.exog_type_in_           = None
        self.exog_dtypes_in_         = None
        self.exog_dtypes_out_        = None
        self.X_train_exog_names_out_ = None
        self.creation_date           = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.is_fitted               = False
        self.fit_date                = None
        self.skforecast_version      = __version__
        self.python_version          = sys.version.split(" ")[0]
        self.forecaster_id           = forecaster_id
        self.valid_estimator_types   = [
            'skforecast.stats._sarimax.Sarimax',
            'skforecast.stats._arar.Arar',
            'skforecast.stats._ets.Ets',
            'aeon.forecasting.stats._arima.ARIMA',
            'aeon.forecasting.stats._ets.ETS'
        ]
        self.estimators_support_exog  = [
            'skforecast.stats._sarimax.Sarimax',
            'skforecast.stats._arar.Arar',
        ]

        estimator_type = f"{type(estimator).__module__}.{type(estimator).__name__}"
        if estimator_type not in self.valid_estimator_types:
            raise TypeError(
                f"`estimator` must be an instance of type {self.valid_estimator_types}. "
                f"Got '{type(estimator)}'."
            )
        self.estimator_type = estimator_type

        self.params = self.estimator.get_params(deep=True)
        self.fit_kwargs = fit_kwargs if fit_kwargs is not None else {}

        self.__skforecast_tags__ = {
            "library": "skforecast",
            "forecaster_name": "ForecasterStats",
            "forecaster_task": "regression",
            "forecasting_scope": "single-series",  # single-series | global
            "forecasting_strategy": "recursive",   # recursive | direct | deep_learning
            "index_types_supported": ["pandas.RangeIndex", "pandas.DatetimeIndex"],
            "requires_index_frequency": True,

            "allowed_input_types_series": ["pandas.Series"],
            "supports_exog": True,
            "allowed_input_types_exog": ["pandas.Series", "pandas.DataFrame"],
            "handles_missing_values_series": False, 
            "handles_missing_values_exog": False, 

            "supports_lags": False,
            "supports_window_features": False,
            "supports_transformer_series": True,
            "supports_transformer_exog": True,
            "supports_weight_func": False,
            "supports_differentiation": False,

            "prediction_types": ["point", "interval"],
            "supports_probabilistic": True,
            "probabilistic_methods": ["distribution"],
            "handles_binned_residuals": False
        }

    def __setstate__(self, state: dict) -> None:
        """
        Custom __setstate__ to ensure backward compatibility when unpickling
        Forecaster objects created with older versions of skforecast.

        Parameters
        ----------
        state : dict
            The state dictionary from the pickled object.

        Returns
        -------
        None

        """

        # Migration: 'regressor' renamed to 'estimator' in version 0.18.0
        if 'regressor' in state and 'estimator' not in state:
            state['estimator'] = state.pop('regressor')

        self.__dict__.update(state)

    @property
    def regressor(self):
        warnings.warn(
            "The `regressor` attribute is deprecated and will be removed in future "
            "versions. Use `estimator` instead.",
            FutureWarning
        )
        return self.estimator
    
    def _preprocess_repr(self) -> tuple[str, str]:
        """
        Format text for __repr__ method.

        Returns
        -------
        text : str
            Formatted text.

        """
        params = str(self.params)
        if len(params) > 58:
            params = "\n    " + textwrap.fill(
                params, width=80, subsequent_indent="    "
            )

        exog_names_in_ = None
        if self.exog_names_in_ is not None:
            exog_names_in_ = copy(self.exog_names_in_)
            if len(exog_names_in_) > 50:
                exog_names_in_ = exog_names_in_[:50] + ["..."]
            exog_names_in_ = ", ".join(exog_names_in_)
            if len(exog_names_in_) > 58:
                exog_names_in_ = "\n    " + textwrap.fill(
                    exog_names_in_, width=80, subsequent_indent="    "
                )
        
        return params, exog_names_in_

    def __repr__(
        self
    ) -> str:
        """
        Information displayed when a ForecasterStats object is printed.
        """

        params, exog_names_in_ = self._preprocess_repr()

        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Estimator: {self.estimator} \n"
            f"Series name: {self.series_name_in_} \n"
            f"Exogenous included: {self.exog_in_} \n"
            f"Exogenous names: {exog_names_in_} \n"
            f"Transformer for y: {self.transformer_y} \n"
            f"Transformer for exog: {self.transformer_exog} \n"
            f"Training range: {self.training_range_.to_list() if self.is_fitted else None} \n"
            f"Training index type: {str(self.index_type_).split('.')[-1][:-2] if self.is_fitted else None} \n"
            f"Training index frequency: {self.index_freq_ if self.is_fitted else None} \n"
            f"Estimator parameters: {params} \n"
            f"fit_kwargs: {self.fit_kwargs} \n"
            f"Creation date: {self.creation_date} \n"
            f"Last fit date: {self.fit_date} \n"
            f"Index seen by the forecaster: {self.extended_index_} \n"
            f"Skforecast version: {self.skforecast_version} \n"
            f"Python version: {self.python_version} \n"
            f"Forecaster id: {self.forecaster_id} \n"
        )

        return info

    def _repr_html_(
        self
    ) -> str:
        """
        HTML representation of the object.
        The "General Information" section is expanded by default.
        """

        params, exog_names_in_ = self._preprocess_repr()
        style, unique_id = get_style_repr_html(self.is_fitted)

        # Get details if applicable to extended estimator name
        estimator_details = ""
        if self.estimator_type == 'skforecast.stats._ets.Ets' and self.is_fitted:
            estimator_details = (
                f"({self.estimator.config_['error']}"
                f"{self.estimator.config_['trend']}"
                f"{self.estimator.config_['season']})"
            )

        content = f"""
        <div class="container-{unique_id}">
            <p style="font-size: 1.5em; font-weight: bold; margin-block-start: 0.83em; margin-block-end: 0.83em;">{type(self).__name__}</p>
            <details open>
                <summary>General Information</summary>
                <ul>
                    <li><strong>Estimator:</strong> {type(self.estimator).__name__}{estimator_details}</li>                  
                    <li><strong>Window size:</strong> {self.window_size}</li>
                    <li><strong>Series name:</strong> {self.series_name_in_}</li>
                    <li><strong>Exogenous included:</strong> {self.exog_in_}</li>
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
                <summary>Estimator Parameters</summary>
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
                <a href="https://skforecast.org/{__version__}/api/forecasterstats.html">&#128712 <strong>API Reference</strong></a>
                &nbsp;&nbsp;
                <a href="https://skforecast.org/{__version__}/user_guides/forecasting-sarimax-arima.html">&#128462 <strong>User Guide</strong></a>
            </p>
        </div>
        """

        return style + content

    def fit(
        self,
        y: pd.Series,
        exog: pd.Series | pd.DataFrame | None = None,
        store_last_window: bool = True,
        suppress_warnings: bool = False
    ) -> None:
        """
        Training Forecaster.

        Additional arguments to be passed to the `fit` method of the estimator 
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
        suppress_warnings : bool, default False
            If `True`, warnings generated during fitting will be ignored.

        Returns
        -------
        None
        
        """

        check_y(y=y)
        if exog is not None:
            if self.estimator_type not in self.estimators_support_exog:
                warnings.warn(
                    f"The estimator {self.estimator_type} does not support exogenous variables, "
                    f"they will be ignored during fit.",
                    IgnoredArgumentWarning
                )
            if len(exog) != len(y):
                raise ValueError(
                    f"`exog` must have same number of samples as `y`. "
                    f"length `exog`: ({len(exog)}), length `y`: ({len(y)})"
                )
            check_exog(exog=exog)

        # Reset values in case the forecaster has already been fitted.
        self.last_window_            = None
        self.extended_index_         = None
        self.index_type_             = None
        self.index_freq_             = None
        self.training_range_         = None
        self.series_name_in_         = None
        self.exog_in_                = False
        self.exog_names_in_          = None
        self.exog_type_in_           = None
        self.exog_dtypes_in_         = None
        self.exog_dtypes_out_        = None
        self.X_train_exog_names_out_ = None
        self.in_sample_residuals_    = None
        self.is_fitted               = False
        self.fit_date                = None
        
        if exog is not None:
            self.exog_in_ = True
            self.exog_type_in_ = type(exog)
            self.exog_dtypes_in_ = get_exog_dtypes(exog=exog)
            self.exog_names_in_ = \
                exog.columns.to_list() if isinstance(exog, pd.DataFrame) else [exog.name]

        y = transform_series(
                series            = y,
                transformer       = self.transformer_y,
                fit               = True,
                inverse_transform = False
            )

        if exog is not None:
            if isinstance(exog, pd.Series):
                exog = exog.to_frame()
            
            exog = transform_dataframe(
                       df                = exog,
                       transformer       = self.transformer_exog,
                       fit               = True,
                       inverse_transform = False
                   )
            
            check_exog_dtypes(exog, call_check_exog=True)
            self.exog_dtypes_out_ = get_exog_dtypes(exog=exog)
            self.X_train_exog_names_out_ = exog.columns.to_list()

        # Prepare fit_kwargs, warning for SARIMAX if needed
        fit_kwargs_to_use = self.fit_kwargs.copy() if self.fit_kwargs else {}
        if fit_kwargs_to_use and self.estimator_type == 'skforecast.stats._sarimax.Sarimax':
            warnings.warn(
                "When using the skforecast Sarimax estimator, fit kwargs should be passed "
                "using the parameter `sm_fit_kwargs` during estimator initialization, "
                "not via ForecasterStats `fit_kwargs`. The provided `fit_kwargs` will be ignored.",
                IgnoredArgumentWarning
            )
            fit_kwargs_to_use = {}
        
        if suppress_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.estimator.fit(y=y, exog=exog, **fit_kwargs_to_use)
        else:
            self.estimator.fit(y=y, exog=exog, **fit_kwargs_to_use)

        self.is_fitted = True
        self.series_name_in_ = y.name if y.name is not None else 'y'
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range_ = y.index[[0, -1]]
        self.index_type_ = type(y.index)
        if isinstance(y.index, pd.DatetimeIndex):
            self.index_freq_ = y.index.freqstr
        else: 
            self.index_freq_ = y.index.step

        if store_last_window:
            self.last_window_ = y.copy()

        if self.estimator_type == 'skforecast.stats._sarimax.Sarimax':
            self.extended_index_ = self.estimator.sarimax_res.fittedvalues.index.copy()
        else:
            self.extended_index_ = y.index
        self.params = self.estimator.get_params(deep=True)

    def _create_predict_inputs(
        self,
        steps: int,
        last_window: pd.Series | None = None,
        last_window_exog: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None
    ) -> tuple[pd.Series, pd.DataFrame | None, pd.DataFrame | None]:
        """
        Create inputs needed for the first iteration of the prediction process. 
        Since it is a recursive process, last window is updated at each 
        iteration of the prediction process.
        
        Parameters
        ----------
        steps : int
            Number of steps to predict. 
        last_window : pandas Series, default None
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window_` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        last_window_exog : pandas Series, pandas DataFrame, default None
            Values of the exogenous variables aligned with `last_window`. Only
            needed when `last_window` is not None and the forecaster has been
            trained including exogenous variables. Used to make predictions 
            unrelated to the original data. Values have to start at the end 
            of the training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.

        Returns
        -------
        last_window : pandas Series
            Series predictors.
        last_window_exog : pandas DataFrame, None
            Values of the exogenous variables aligned with `last_window`.
        exog : pandas DataFrame, None
            Exogenous variable/s included as predictor/s.
        
        """

        # Needs to be a new variable to avoid arima_res_.append when using 
        # self.last_window. It already has it stored.
        last_window_check = last_window if last_window is not None else self.last_window_

        check_predict_input(
            forecaster_name  = type(self).__name__,
            steps            = steps,
            is_fitted        = self.is_fitted,
            exog_in_         = self.exog_in_,
            index_type_      = self.index_type_,
            index_freq_      = self.index_freq_,
            window_size      = self.window_size,
            last_window      = last_window_check,
            last_window_exog = last_window_exog,
            exog             = exog,
            exog_names_in_   = self.exog_names_in_,
            interval         = None,
            alpha            = None
        )
        
        # If not last_window is provided, last_window needs to be None
        if last_window is not None:
            last_window = last_window.copy()

        # When last_window_exog is provided but no last_window
        if last_window is None and last_window_exog is not None:
            raise ValueError(
                "To make predictions unrelated to the original data, both "
                "`last_window` and `last_window_exog` must be provided."
            )

        # Check if forecaster needs exog
        if last_window is not None and last_window_exog is None and self.exog_in_:
            raise ValueError(
                "Forecaster trained with exogenous variable/s. To make predictions "
                "unrelated to the original data, same variable/s must be provided "
                "using `last_window_exog`."
            )

        if last_window is not None:
            # If predictions do not follow directly from the end of the training 
            # data. The internal statsmodels SARIMAX model needs to be updated 
            # using its append method. The data needs to start at the end of the 
            # training series.

            # Check index append values
            expected_index = expand_index(index=self.extended_index_, steps=1)[0]
            if expected_index != last_window.index[0]:
                raise ValueError(
                    f"To make predictions unrelated to the original data, `last_window` "
                    f"has to start at the end of the index seen by the forecaster.\n"
                    f"    Series last index         : {self.extended_index_[-1]}.\n"
                    f"    Expected index            : {expected_index}.\n"
                    f"    `last_window` index start : {last_window.index[0]}."
                )
            
            last_window = transform_series(
                              series            = last_window,
                              transformer       = self.transformer_y,
                              fit               = False,
                              inverse_transform = False
                          )
            
            # Transform last_window_exog
            if last_window_exog is not None:
                # check index last_window_exog
                if expected_index != last_window_exog.index[0]:
                    raise ValueError(
                        f"To make predictions unrelated to the original data, `last_window_exog` "
                        f"has to start at the end of the index seen by the forecaster.\n"
                        f"    Series last index              : {self.extended_index_[-1]}.\n"
                        f"    Expected index                 : {expected_index}.\n"
                        f"    `last_window_exog` index start : {last_window_exog.index[0]}."
                    )

                if isinstance(last_window_exog, pd.Series):
                    last_window_exog = last_window_exog.to_frame()
            
                last_window_exog = transform_dataframe(
                                       df                = last_window_exog,
                                       transformer       = self.transformer_exog,
                                       fit               = False,
                                       inverse_transform = False
                                   )
                
        # Exog
        if exog is not None:
            if isinstance(exog, pd.Series):
                exog = exog.to_frame()

            exog = transform_dataframe(
                       df                = exog,
                       transformer       = self.transformer_exog,
                       fit               = False,
                       inverse_transform = False
                   )  
            exog = exog.iloc[:steps, ]

        return last_window, last_window_exog, exog

    def predict(
        self,
        steps: int,
        last_window: pd.Series | None = None,
        last_window_exog: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None
    ) -> pd.Series:
        """
        Forecast future values.

        Generate predictions (forecasts) n steps in the future. Note that if 
        exogenous variables were used in the model fit, they will be expected 
        for the predict procedure and will fail otherwise.
        
        When predicting using `last_window` and `last_window_exog`, they must
        start right after the end of the index seen by the forecaster during
        training.
        
        Parameters
        ----------
        steps : int
            Number of steps to predict. 
        last_window : pandas Series, default None
            Series values used to create the predictors needed in the 
            predictions. Used to make predictions unrelated to the original data. 
            Values have to start at the end of the training data.
        last_window_exog : pandas Series, pandas DataFrame, default None
            Values of the exogenous variables aligned with `last_window`. Only
            needed when `last_window` is not None and the forecaster has been
            trained including exogenous variables. Used to make predictions 
            unrelated to the original data. Values have to start at the end 
            of the training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.

        Returns
        -------
        predictions : pandas Series
            Predicted values.
        
        """

        last_window, last_window_exog, exog = self._create_predict_inputs(
                                                  steps            = steps,
                                                  last_window      = last_window,
                                                  last_window_exog = last_window_exog,
                                                  exog             = exog,
                                              )
        
        if last_window is not None:
            if self.estimator_type == 'skforecast.stats._sarimax.Sarimax':
                self.estimator.append(
                    y     = last_window,
                    exog  = last_window_exog,
                    refit = False
                )
                self.extended_index_ = self.estimator.sarimax_res.fittedvalues.index
            else:
                raise ValueError(
                    "`last_window` is only supported for the skforecast.Sarimax estimator (statsmodels)."
                    "For other models, predictions must follow directly after the end of the "
                    "training data."
                )

        # Dictionary dispatch for estimator-specific prediction methods
        predict_dispatch = {
            'skforecast.stats._sarimax.Sarimax': self._predict_sarimax,
            'skforecast.stats._arar.Arar': self._predict_skforecast_stats,
            'skforecast.stats._ets.Ets': self._predict_skforecast_stats,
            'aeon.forecasting.stats._arima.ARIMA': self._predict_aeon,
            'aeon.forecasting.stats._ets.ETS': self._predict_aeon
        }
        
        predict_method = predict_dispatch[self.estimator_type]
        predictions = predict_method(steps, exog)

        return predictions

    def _predict_sarimax(self, steps: int, exog: pd.Series | pd.DataFrame | None) -> pd.Series:
        """Generate predictions using SARIMAX model."""
        predictions = self.estimator.predict(steps=steps, exog=exog).iloc[:, 0]
        predictions = transform_series(
            series=predictions,
            transformer=self.transformer_y,
            fit=False,
            inverse_transform=True
        )
        predictions.name = 'pred'
        return predictions

    def _predict_skforecast_stats(self, steps: int, exog: pd.Series | pd.DataFrame | None) -> pd.Series:
        """Generate predictions using skforecast ARAR/ETS models."""
        predictions = self.estimator.predict(steps=steps, exog=exog)
        predictions = transform_numpy(
            array=predictions,
            transformer=self.transformer_y,
            fit=False,
            inverse_transform=True
        )
        predictions_index = expand_index(index=self.extended_index_, steps=steps)
        return pd.Series(predictions, index=predictions_index, name='pred')

    def _predict_aeon(self, steps: int, exog: pd.Series | pd.DataFrame | None) -> pd.Series:
        """Generate predictions using AEON models."""
        predictions = self.estimator.iterative_forecast(
            y=self.last_window_.to_numpy(),
            prediction_horizon=steps
        )
        predictions = transform_numpy(
            array=predictions,
            transformer=self.transformer_y,
            fit=False,
            inverse_transform=True
        )
        predictions_index = expand_index(index=self.extended_index_, steps=steps)
        return pd.Series(predictions, index=predictions_index, name='pred')

    def predict_interval(
        self,
        steps: int,
        last_window: pd.Series | None = None,
        last_window_exog: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        alpha: float = 0.05,
        interval: list[float] | tuple[float] | None = None,
    ) -> pd.DataFrame:
        """
        Forecast future values and their confidence intervals.

        Generate predictions (forecasts) n steps in the future with confidence
        intervals. Note that if exogenous variables were used in the model fit, 
        they will be expected for the predict procedure and will fail otherwise.

        When predicting using `last_window` and `last_window_exog`, the internal
        statsmodels SARIMAX will be updated using its append method. To do this,
        `last_window` data must start at the end of the index seen by the 
        forecaster, this is stored in forecaster.extended_index_.

        Check https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMAResults.append.html
        to know more about statsmodels append method.

        Parameters
        ----------
        steps : int
            Number of steps to predict. 
        last_window : pandas Series, default None
            Series values used to create the predictors needed in the 
            predictions. Used to make predictions unrelated to the original data. 
            Values have to start at the end of the training data.
        last_window_exog : pandas Series, pandas DataFrame, default None
            Values of the exogenous variables aligned with `last_window`. Only
            need when `last_window` is not None and the forecaster has been
            trained including exogenous variables.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.
        alpha : float, default 0.05
            The confidence intervals for the forecasts are (1 - alpha) %.
            If both, `alpha` and `interval` are provided, `alpha` will be used.
        interval : list, tuple, default None
            Confidence of the prediction interval estimated. The values must be
            symmetric. Sequence of percentiles to compute, which must be between 
            0 and 100 inclusive. For example, interval of 95% should be as 
            `interval = [2.5, 97.5]`. If both, `alpha` and `interval` are 
            provided, `alpha` will be used.

        Returns
        -------
        predictions : pandas DataFrame
            Values predicted by the forecaster and their estimated interval.

            - pred: predictions.
            - lower_bound: lower bound of the interval.
            - upper_bound: upper bound of the interval.

        """

        # If interval and alpha take alpha, if interval transform to alpha
        if alpha is None:
            if 100 - interval[1] != interval[0]:
                raise ValueError(
                    f"When using `interval` in ForecasterStats, it must be symmetrical. "
                    f"For example, interval of 95% should be as `interval = [2.5, 97.5]`. "
                    f"Got {interval}."
                )
            alpha = 2 * (100 - interval[1]) / 100

        last_window, last_window_exog, exog = self._create_predict_inputs(
                                                  steps            = steps,
                                                  last_window      = last_window,
                                                  last_window_exog = last_window_exog,
                                                  exog             = exog,
                                              )

        if last_window is not None:
            if self.estimator_type == 'skforecast.stats._sarimax.Sarimax':
                self.estimator.append(
                    y     = last_window,
                    exog  = last_window_exog,
                    refit = False
                )
                self.extended_index_ = self.estimator.sarimax_res.fittedvalues.index
            else:
                raise NotImplementedError(
                    f"Prediction intervals with `last_window` parameter is only "
                    f"supported for SARIMAX models. For {self.estimator_type}, "
                    f"predictions with `last_window` are not yet implemented."
                    f"It is required that predictions follow directly after the "
                    f"end of the training data."
                )

        # Get following n steps predictions with intervals
        # Dictionary dispatch for estimator-specific prediction interval methods
        predict_interval_dispatch = {
            'skforecast.stats._sarimax.Sarimax': self._predict_interval_sarimax,
            'skforecast.stats._arar.Arar': self._predict_interval_skforecast_stats,
            'skforecast.stats._ets.Ets': self._predict_interval_skforecast_stats
        }
        
        if self.estimator_type in ['aeon.forecasting.stats._arima.ARIMA', 'aeon.forecasting.stats._ets.ETS']:
            raise NotImplementedError(
                "Prediction intervals is not implemented for AEON ARIMA and ETS models yet."
            )
        
        predict_interval_method = predict_interval_dispatch[self.estimator_type]
        predictions = predict_interval_method(steps, exog, alpha)

        if self.transformer_y:
            # Transform all columns with the same transformer
            predictions_values = self.transformer_y.inverse_transform(predictions.values)
            if hasattr(predictions_values, 'toarray'):
                predictions_values = predictions_values.toarray()
            predictions = pd.DataFrame(
                predictions_values,
                index=predictions.index,
                columns=predictions.columns
            )

        return predictions

    def _predict_interval_sarimax(
        self, 
        steps: int, 
        exog: pd.Series | pd.DataFrame | None, 
        alpha: float
    ) -> pd.DataFrame:
        """Generate prediction intervals using SARIMAX model."""
        return self.estimator.predict(
            steps=steps,
            exog=exog,
            return_conf_int=True,
            alpha=alpha
        )

    def _predict_interval_skforecast_stats(
        self, 
        steps: int, 
        exog: pd.Series | pd.DataFrame | None, 
        alpha: float
    ) -> pd.DataFrame:
        """Generate prediction intervals using skforecast ARAR/ETS models."""
        predictions = self.estimator.predict_interval(
            steps=steps,
            exog=exog,
            level=[100 * (1 - alpha)],
            as_frame=True
        )
        predictions_index = expand_index(index=self.extended_index_, steps=steps)
        predictions.index = predictions_index
        predictions.columns = ['pred', 'lower_bound', 'upper_bound']
        return predictions

    def set_params(
        self, 
        params: dict[str, object]
    ) -> None:
        """
        Set new values to the parameters of the model stored in the forecaster.
        
        Parameters
        ----------
        params : dict
            Parameters values.

        Returns
        -------
        None
        
        """

        self.estimator = clone(self.estimator)
        self.estimator.set_params(**params)
        self.params = self.estimator.get_params(deep=True)

    def set_fit_kwargs(
        self, 
        fit_kwargs: dict[str, object]
    ) -> None:
        """
        Set new values for the additional keyword arguments passed to the `fit` 
        method of the estimator.
        
        Parameters
        ----------
        fit_kwargs : dict
            Dict of the form {"argument": new_value}.

        Returns
        -------
        None
        
        """

        warnings.warn(
            "When using the skforecast Sarimax model, the fit kwargs should "
            "be passed using the model parameter `sm_fit_kwargs`.",
            IgnoredArgumentWarning
        )
        self.fit_kwargs = {}

    def get_feature_importances(
        self,
        sort_importance: bool = True
    ) -> pd.DataFrame:
        """
        Return feature importances of the estimator stored in the forecaster.

        Parameters
        ----------
        sort_importance: bool, default True
            If `True`, sorts the feature importances in descending order.

        Returns
        -------
        feature_importances : pandas DataFrame
            Feature importances associated with each predictor.

        """

        if not self.is_fitted:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `get_feature_importances()`."
            )
        
        # Dictionary dispatch for estimator-specific feature importance methods
        feature_importances_dispatch = {
            'skforecast.stats._sarimax.Sarimax': self._get_feature_importances_sarimax,
            'skforecast.stats._arar.Arar': self._get_feature_importances_arar,
            'skforecast.stats._ets.Ets': self._get_feature_importances_ets,
            'aeon.forecasting.stats._arima.ARIMA': self._get_feature_importances_aeon_arima,
            'aeon.forecasting.stats._ets.ETS': self._get_feature_importances_aeon_ets
        }
        
        get_importances_method = feature_importances_dispatch[self.estimator_type]
        feature_importances = get_importances_method()

        if sort_importance:
            feature_importances = feature_importances.sort_values(
                                      by='importance', ascending=False
                                  ).reset_index(drop=True)

        return feature_importances

    def _get_feature_importances_sarimax(self) -> pd.DataFrame:
        """Get feature importances for SARIMAX model."""
        feature_importances = self.estimator.params().to_frame().reset_index()
        feature_importances.columns = ['feature', 'importance']
        return feature_importances

    def _get_feature_importances_arar(self) -> pd.DataFrame:
        """Get feature importances for ARAR model."""
        return pd.DataFrame({
            'feature': [f'lag_{lag}' for lag in self.estimator.lags_],
            'importance': self.estimator.coef_
        })

    def _get_feature_importances_ets(self) -> pd.DataFrame:
        """Get feature importances for ETS model."""
        features = ['alpha (level)']
        importances = [self.estimator.params_['alpha']]
        
        if self.estimator.config_['trend'] != 'N':
            features.append('beta (trend)')
            importances.append(self.estimator.params_['beta'])
        
        if self.estimator.config_['season'] != 'N':
            features.append('gamma (seasonal)')
            importances.append(self.estimator.params_['gamma'])
        
        if self.estimator.config_['damped']:
            features.append('phi (damping)')
            importances.append(self.estimator.params_['phi'])
        
        return pd.DataFrame({
            'feature': features,
            'importance': importances
        })

    def _get_feature_importances_aeon_arima(self) -> pd.DataFrame:
        """Get feature importances for AEON ARIMA model."""
        return pd.DataFrame({
            'feature': [f'lag_{lag}' for lag in range(1, self.estimator.p + 1)] + ["ma", "intercept"],
            'importance': np.concatenate([self.estimator.phi_, self.estimator.theta_, [self.estimator.c_]])
        })

    def _get_feature_importances_aeon_ets(self) -> pd.DataFrame:
        """Get feature importances for AEON ETS model."""
        warnings.warn("Feature importances is not available for the AEON ETS model.")
        return pd.DataFrame(columns=['feature', 'importance'])

    def get_info_criteria(
        self, 
        criteria: str = 'aic', 
        method: str = 'standard'
    ) -> float:
        """
        Get the selected information criteria.

        Check https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.info_criteria.html
        to know more about statsmodels info_criteria method.

        Parameters
        ----------
        criteria : str, default 'aic'
            The information criteria to compute. Valid options are {'aic', 'bic',
            'hqic'}.
        method : str, default 'standard'
            The method for information criteria computation. Default is 'standard'
            method; 'lutkepohl' computes the information criteria as in Lütkepohl
            (2007).

        Returns
        -------
        metric : float
            The value of the selected information criteria.

        """

        # Dictionary dispatch for estimator-specific info criteria methods
        info_criteria_dispatch = {
            'skforecast.stats._sarimax.Sarimax': self._get_info_criteria_sarimax,
            'skforecast.stats._arar.Arar': self._get_info_criteria_arar,
            'skforecast.stats._ets.Ets': self._get_info_criteria_ets,
            'aeon.forecasting.stats._arima.ARIMA': self._get_info_criteria_aeon,
            'aeon.forecasting.stats._ets.ETS': self._get_info_criteria_aeon
        }
        
        get_criteria_method = info_criteria_dispatch[self.estimator_type]
        metric = get_criteria_method(criteria, method)
        
        return metric

    def _get_info_criteria_sarimax(self, criteria: str, method: str) -> float:
        """Get information criteria for SARIMAX model."""
        if criteria not in ['aic', 'bic', 'hqic']:
            raise ValueError(
                "Invalid value for `criteria`. Valid options are 'aic', 'bic', "
                "and 'hqic'."
            )
        
        if method not in ['standard', 'lutkepohl']:
            raise ValueError(
                "Invalid value for `method`. Valid options are 'standard' and "
                "'lutkepohl'."
            )
        
        return self.estimator.get_info_criteria(criteria=criteria, method=method)

    def _get_info_criteria_arar(self, criteria: str, method: str) -> float:
        """Get information criteria for ARAR model."""
        if criteria not in ['aic', 'bic']:
            raise ValueError(
                "Invalid value for `criteria`. Valid options are 'aic' and 'bic' "
                "for ARAR model."
            )
        
        if method != 'standard':
            raise ValueError(
                "Invalid value for `method`. Only 'standard' is supported for "
                "ARAR model."
            )
        
        return self.estimator.aic_ if criteria == 'aic' else self.estimator.bic_

    def _get_info_criteria_ets(self, criteria: str, method: str) -> float:
        """Get information criteria for skforecast ETS model."""
        if criteria not in ['aic', 'bic']:
            raise ValueError(
                "Invalid value for `criteria`. Valid options are 'aic' and 'bic' "
                "for ETS model."
            )
        
        if method != 'standard':
            raise ValueError(
                "Invalid value for `method`. Only 'standard' is supported for "
                "ETS model."
            )
        
        return self.estimator.model_.aic if criteria == 'aic' else self.estimator.model_.bic

    def _get_info_criteria_aeon(self, criteria: str, method: str) -> float:
        """Get information criteria for AEON models."""
        if criteria != 'aic':
            raise ValueError(
                "Invalid value for `criteria`. Only 'aic' is supported for "
                "AEON models."
            )
        return self.estimator.aic_

    def summary(self) -> None:
        """
        Show forecaster information.
        
        Parameters
        ----------
        self

        Returns
        -------
        None
        
        """
        
        print(self)

    def reduce_memory(self) -> None:
        """
        Reduce memory usage by removing internal arrays of the estimator not
        needed for prediction. This method only works for estimators that
        expose the method `reduce_memory()`.
        The arrays removed depend on the specific estimator used.
        
        Returns
        -------
        None
        
        """
        estimators_with_reduce_memory = [
            'skforecast.stats._arar.Arar',
            'skforecast.stats._ets.Ets'
        ]
        if not self.is_fitted:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `reduce_memory()`."
            )
        if hasattr(self.estimator, 'reduce_memory'):
            self.estimator.reduce_memory()
        else:
            raise NotImplementedError(
                f"The estimator {self.estimator_type} does not implement "
                f"`reduce_memory()` method. Currently, this method is only "
                f"available for the following estimators: "
                f"{estimators_with_reduce_memory}."
            )