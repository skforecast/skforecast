################################################################################
#                             ForecasterStats                                  #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
from typing import Any
import warnings
import sys
from copy import copy
import textwrap
import numpy as np
import pandas as pd
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
    set_skforecast_warnings,
    initialize_estimator
)


# TODO: Get estimator info, que sea una tabla con los ids, con los nombres, etc
class ForecasterStats():
    """
    This class turns statistical models into a Forecaster compatible with the 
    skforecast API. It supports single or multiple statistical models for the 
    same time series, enabling model comparison and ensemble predictions.
    
    Supported statistical models are: skforecast.stats.Sarimax, skforecast.stats.Arima,
    skforecast.stats.Arar, skforecast.stats.Ets, aeon.forecasting.stats.ARIMA and
    aeon.forecasting.stats.ETS.
    
    Parameters
    ----------
    estimator : object, list of objects
        A statistical model instance or a list of statistical model instances. 
        When a list is provided, all models are fitted to the same time series 
        and predictions from all models are returned. Supported models are:
        
        - skforecast.stats.Arima 
        - skforecast.stats.Arar
        - skforecast.stats.Ets
        - skforecast.stats.Sarimax (statsmodels wrapper)
        - sktime.forecasting.ARIMA (pdmarima wrapper)
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
    forecaster_id : str, int, default None
        Name used as an identifier of the forecaster.
    regressor : estimator or pipeline compatible with the Keras API
        **Deprecated**, alias for `estimator`.
    fit_kwargs : Ignored
        Not used, present here for API consistency by convention.
    
    Attributes
    ----------
    estimators : list
        List of the original statistical model instances provided by the user 
        without being fitted.
    estimators_ : list
        List of statistical model instances. These are the estimators that will
        be trained when `fit()` is called.
    estimator_ids : list
        Unique identifiers for each estimator, generated from estimator types and 
        numeric suffixes to handle duplicates (e.g., 'skforecast.Arima', 
        'skforecast.Arima_2', 'skforecast.Ets'). Used to identify predictions 
        from each model.
    estimator_names_ : list
        Descriptive names for each estimator including the fitted model configuration
        (e.g., 'Arima(1,1,1)(0,0,0)[12]', 'Ets(AAA)', etc.). This is updated 
        after fitting to reflect the selected model.
    estimator_types_ : tuple
        Full qualified type string for each estimator (e.g., 
        'skforecast.stats._arima.Arima').
    estimator_params_ : dict
        Dictionary containing the parameters of each estimator.
    n_estimators : int
        Number of estimators in the forecaster.
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
        Index the forecaster has seen during training and prediction. This 
        attribute's initial value is the index of the training data, but this 
        is extended after predictions are made using an external 'last_window'.
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
    creation_date : str
        Date of creation.
    is_fitted : bool
        Tag to identify if the estimator has been fitted (trained).
    fit_date : str
        Date of last fit.
    valid_estimator_types : tuple
        Valid estimator types.
    estimators_support_last_window : tuple
        Estimators that support last_window argument in prediction methods.
    estimators_support_exog : tuple
        Estimators that support exogenous variables.
    estimators_support_interval : tuple
        Estimators that support prediction intervals.
    estimators_support_reduce_memory : tuple
        Estimators that support reduce memory method.
    _predict_dispatch : dict
        Dictionary dispatch for estimator-specific predict methods.
    _predict_interval_dispatch : dict
        Dictionary dispatch for estimator-specific predict_interval methods.
    _feature_importances_dispatch : dict
        Dictionary dispatch for estimator-specific feature importance methods.
    _info_criteria_dispatch : dict
        Dictionary dispatch for estimator-specific information criteria methods.
    skforecast_version : str
        Version of skforecast library used to create the forecaster.
    python_version : str
        Version of python used to create the forecaster.
    forecaster_id : str, int
        Name used as an identifier of the forecaster.
    __skforecast_tags__ : dict
        Tags associated with the forecaster.
    fit_kwargs : Ignored
        Not used, present here for API consistency by convention.
    
    """
    
    def __init__(
        self,
        estimator: object | list[object] = None,
        transformer_y: object | None = None,
        transformer_exog: object | None = None,
        forecaster_id: str | int | None = None,
        regressor: object = None,
        fit_kwargs: Any = None,
    ) -> None:
        
        # Valid estimator types (class-level constant)
        self.valid_estimator_types = (
            'skforecast.stats._arima.Arima',
            'skforecast.stats._arar.Arar',
            'skforecast.stats._ets.Ets',
            'skforecast.stats._sarimax.Sarimax',
            'aeon.forecasting.stats._arima.ARIMA',
            'aeon.forecasting.stats._ets.ETS',
            'sktime.forecasting.arima._pmdarima.ARIMA'
        )

        # TODO: Remove 0.20. Handle deprecated 'regressor' argument
        estimator = initialize_estimator(estimator, regressor)
        
        if not isinstance(estimator, list):
            estimator = [estimator]
        else:
            if len(estimator) == 0:
                raise ValueError("`estimator` list cannot be empty.")
        
        # Validate all estimators and collect types
        estimator_types_ = []
        for i, est in enumerate(estimator):
            est_type = f"{type(est).__module__}.{type(est).__name__}"
            if est_type not in self.valid_estimator_types:
                raise TypeError(
                    f"Estimator at index {i} must be an instance of type "
                    f"{self.valid_estimator_types}. Got '{type(est)}'."
                )
            estimator_types_.append(est_type)
        
        # TODO: Review window_size for statistical models
        # TODO Review _search functions, they only work for single estimator
        # TODO: Decide if include 'aggregate' parameter for multiple estimators, it
        # aggregates predictions from all estimators.
        self.estimators              = estimator
        self.estimators_             = [copy(est) for est in self.estimators]
        self.estimator_ids           = self._generate_ids(self.estimators)
        self.estimator_names_        = [None] * len(self.estimators)
        self.estimator_types_        = estimator_types_
        self.n_estimators            = len(self.estimators)
        self.estimator_params_       = None
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
        self.fit_kwargs              = None  # Ignored, present for API consistency

        self.estimators_support_last_window = (
            'skforecast.stats._sarimax.Sarimax',
        )
        self.estimators_support_exog = (
            'skforecast.stats._arima.Arima',
            'skforecast.stats._arar.Arar',
            'skforecast.stats._sarimax.Sarimax',
            'sktime.forecasting.arima._pmdarima.ARIMA',
        )
        self.estimators_support_interval = (
            'skforecast.stats._arima.Arima',
            'skforecast.stats._arar.Arar',
            'skforecast.stats._ets.Ets',
            'skforecast.stats._sarimax.Sarimax',
            'sktime.forecasting.arima._pmdarima.ARIMA'
        )
        self.estimators_support_reduce_memory = (
            'skforecast.stats._arima.Arima',
            'skforecast.stats._arar.Arar',
            'skforecast.stats._ets.Ets'
        )
        self._predict_dispatch = {
            'skforecast.stats._arima.Arima': self._predict_skforecast_stats,
            'skforecast.stats._arar.Arar': self._predict_skforecast_stats,
            'skforecast.stats._ets.Ets': self._predict_skforecast_stats,
            'skforecast.stats._sarimax.Sarimax': self._predict_sarimax,
            'aeon.forecasting.stats._arima.ARIMA': self._predict_aeon,
            'aeon.forecasting.stats._ets.ETS': self._predict_aeon,
            'sktime.forecasting.arima._pmdarima.ARIMA': self._predict_sktime_arima
        }
        self._predict_interval_dispatch = {
            'skforecast.stats._arima.Arima': self._predict_interval_skforecast_stats,
            'skforecast.stats._arar.Arar': self._predict_interval_skforecast_stats,
            'skforecast.stats._ets.Ets': self._predict_interval_skforecast_stats,
            'skforecast.stats._sarimax.Sarimax': self._predict_interval_sarimax,
            'sktime.forecasting.arima._pmdarima.ARIMA': self._predict_interval_sktime_arima,
        }
        self._feature_importances_dispatch = {
            'skforecast.stats._arima.Arima': self._get_feature_importances_arima,
            'skforecast.stats._arar.Arar': self._get_feature_importances_arar,
            'skforecast.stats._ets.Ets': self._get_feature_importances_ets,
            'skforecast.stats._sarimax.Sarimax': self._get_feature_importances_sarimax,
            'aeon.forecasting.stats._arima.ARIMA': self._get_feature_importances_aeon_arima,
            'aeon.forecasting.stats._ets.ETS': self._get_feature_importances_aeon_ets,
            'sktime.forecasting.arima._pmdarima.ARIMA': self._get_feature_importances_sktime_arima
        }
        self._info_criteria_dispatch = {
            'skforecast.stats._arima.Arima': self._get_info_criteria_arima,
            'skforecast.stats._arar.Arar': self._get_info_criteria_arar,
            'skforecast.stats._ets.Ets': self._get_info_criteria_ets,
            'skforecast.stats._sarimax.Sarimax': self._get_info_criteria_sarimax,
            'aeon.forecasting.stats._arima.ARIMA': self._get_info_criteria_aeon,
            'aeon.forecasting.stats._ets.ETS': self._get_info_criteria_aeon,
            'sktime.forecasting.arima._pmdarima.ARIMA': self._get_info_criteria_sktime_arima
        }

        # TODO: Review, multiple_estimator flag?
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
        return self.estimators

    def _generate_ids(self, estimators: list) -> list[str]:
        """
        Generate unique ids for a list of estimators. Handles duplicate ids by 
        appending a numeric suffix.
        
        Parameters
        ----------
        estimators : list
            List of statistical model instances.
        
        Returns
        -------
        ids : list[str]
            List of unique ids for each estimator.
        
        """

        ids = []
        id_counts = {}
        for est in estimators:

            base_id = (
                f"{type(est).__module__.split('.')[0]}.{type(est).__name__}"
            )
            
            # Track occurrences and add suffix for duplicates
            if base_id in id_counts:
                id_counts[base_id] += 1
                unique_id = f"{base_id}_{id_counts[base_id]}"
            else:
                id_counts[base_id] = 1
                unique_id = base_id
            
            ids.append(unique_id)
        
        return ids

    def get_estimator(self, id: str) -> object:
        """
        Get a specific estimator by its id.
        
        Parameters
        ----------
        id : str
            The id of the estimator to retrieve.
        
        Returns
        -------
        estimator : object
            The requested estimator instance.
        
        """
        
        if id not in self.estimator_ids:
            raise KeyError(
                f"No estimator with id '{id}'. "
                f"Available estimators: {self.estimator_ids}"
            )
        
        idx = self.estimator_ids.index(id)

        return self.estimators_[idx]
    
    def get_estimator_ids(self) -> list[str]:
        """
        Get the ids of all estimators in the forecaster.
        
        Returns
        -------
        estimator_ids : list[str]
            List of estimator ids.
        
        """

        return self.estimator_ids
    
    def remove_estimator(self, ids: str | list[str]) -> None:
        """
        Remove one or more estimators by their ids.
        
        Parameters
        ----------
        ids : str, list[str]
            The ids of the estimators to remove.
        
        Returns
        -------
        None
        
        """

        if isinstance(ids, str):
            ids = [ids]
        
        missing_ids = [id for id in ids if id not in self.estimator_ids]
        if missing_ids:
            raise KeyError(
                f"No estimator(s) with id '{missing_ids}'. "
                f"Available estimators: {self.estimator_ids}"
            )
            
        for id in ids:
            idx = self.estimator_ids.index(id)
            del self.estimators[idx]
            del self.estimators_[idx]
            del self.estimator_ids[idx]
            del self.estimator_names_[idx]
            del self.estimator_types_[idx]
            self.n_estimators -= 1

    def _preprocess_repr(self) -> tuple[list[str], str]:
        """
        Format text for __repr__ method.

        Returns
        -------
        estimator_params : list[str]
            List of formatted parameters for each estimator.
        exog_names_in_ : str
            Formatted exogenous variable names.

        """
        
        # Format parameters for each estimator
        estimator_params = []
        if self.estimator_params_ is not None:
            for id in self.estimator_ids:
                params = str(self.estimator_params_[id])
                if len(params) > 58:
                    params = "\n        " + textwrap.fill(
                        params, width=76, subsequent_indent="        "
                    )
                estimator_params.append(f"{id}: {params}")

        # Format exogenous variable names
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
        
        return estimator_params, exog_names_in_

    def __repr__(
        self
    ) -> str:
        """
        Information displayed when a ForecasterStats object is printed.
        """

        estimator_params, exog_names_in_ = self._preprocess_repr()
        params_list = "\n    ".join(estimator_params)

        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Estimators: {self.estimator_ids} \n"
            f"Series name: {self.series_name_in_} \n"
            f"Exogenous included: {self.exog_in_} \n"
            f"Exogenous names: {exog_names_in_} \n"
            f"Transformer for y: {self.transformer_y} \n"
            f"Transformer for exog: {self.transformer_exog} \n"
            f"Training range: {self.training_range_.to_list() if self.is_fitted else None} \n"
            f"Training index type: {str(self.index_type_).split('.')[-1][:-2] if self.is_fitted else None} \n"
            f"Training index frequency: {self.index_freq_ if self.is_fitted else None} \n"
            f"Estimator parameters: \n    {params_list} \n"
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

        estimator_params, exog_names_in_ = self._preprocess_repr()
        style, unique_id = get_style_repr_html(self.is_fitted)

        # Build estimators list
        estimators_html = "<ul>"
        for est_id, est_name in zip(self.estimator_ids, self.estimator_names_):
            if est_name is not None:
                estimators_html += f"<li>{est_id}: {est_name}</li>"
            else:
                estimators_html += f"<li>{est_id}</li>"
        estimators_html += "</ul>"

        # Build parameters section
        if len(estimator_params) == 1:
            params_html = f"<ul><li>{estimator_params[0]}</li></ul>"
        else:
            params_html = "<ul>"
            for param in estimator_params:
                params_html += f"<li>{param}</li>"
            params_html += "</ul>"

        content = f"""
        <div class="container-{unique_id}">
            <p style="font-size: 1.5em; font-weight: bold; margin-block-start: 0.83em; margin-block-end: 0.83em;">{type(self).__name__}</p>
            <details open>
                <summary>General Information</summary>
                <ul>
                    <li><strong>Estimators:</strong> {estimators_html}</li>                  
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
                {params_html}
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

        Fits all estimators to the same time series. Each estimator is trained
        independently on the transformed data.
        
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

        set_skforecast_warnings(suppress_warnings, action='ignore')

        self.estimators_             = [copy(est) for est in self.estimators]
        self.estimator_names_        = [None] * len(self.estimators)
        self.estimator_params_       = None
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

        check_y(y=y)
        
        if exog is not None:
            
            # NaNs are checked later
            check_exog(exog=exog)
            if len(exog) != len(y):
                raise ValueError(
                    f"`exog` must have same number of samples as `y`. "
                    f"length `exog`: ({len(exog)}), length `y`: ({len(y)})"
                )
            
            unsupported_exog = [
                id for id, est_type in zip(self.estimator_ids, self.estimator_types_)
                if est_type not in self.estimators_support_exog
            ]
            if unsupported_exog:
                warnings.warn(
                    f"The following estimators do not support exogenous variables and "
                    f"will ignore them during fit: {unsupported_exog}",
                    IgnoredArgumentWarning
                )

        y = transform_series(
                series            = y,
                transformer       = self.transformer_y,
                fit               = True,
                inverse_transform = False
            )

        if exog is not None:

            # NOTE: This must be here, before transforming exog
            self.exog_in_ = True
            self.exog_type_in_ = type(exog)
            self.exog_names_in_ = (
                exog.columns.to_list() if isinstance(exog, pd.DataFrame) else [exog.name]
            )
            self.exog_dtypes_in_ = get_exog_dtypes(exog=exog)

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

        if suppress_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for estimator in self.estimators_:
                    estimator.fit(y=y, exog=exog)
        else:
            for estimator in self.estimators_:
                estimator.fit(y=y, exog=exog)

        self.is_fitted = True

        for i, estimator in enumerate(self.estimators_):
            # Check if estimator has estimator_name_ attribute (skforecast models)
            if hasattr(estimator, 'estimator_name_') and estimator.estimator_name_ is not None:
                self.estimator_names_[i] = estimator.estimator_name_
            else:
                self.estimator_names_[i] = f"{type(estimator).__module__.split('.')[0]}.{type(estimator).__name__}"

        self.estimator_params_ = {
            est_id: est.get_params() 
            for est_id, est in zip(self.estimator_ids, self.estimators_)
        }
        self.series_name_in_ = y.name if y.name is not None else 'y'
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range_ = y.index[[0, -1]]
        self.index_type_ = type(y.index)
        if isinstance(y.index, pd.DatetimeIndex):
            self.index_freq_ = y.index.freqstr
        else: 
            self.index_freq_ = y.index.step

        # TODO: Check when multiple series are supported
        if store_last_window:
            self.last_window_ = y.copy()

        # Set extended_index_ based on first SARIMAX estimator or default to y.index
        first_sarimax = next(
            (est for est, est_type in zip(self.estimators_, self.estimator_types_)
             if est_type == 'skforecast.stats._sarimax.Sarimax'),
            None
        )
        if first_sarimax is not None:
            self.extended_index_ = first_sarimax.sarimax_res.fittedvalues.index.copy()
        else:
            self.extended_index_ = y.index

        set_skforecast_warnings(suppress_warnings, action='default')

    def _create_predict_inputs(
        self,
        steps: int,
        last_window: pd.Series | None = None,
        last_window_exog: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None
    ) -> tuple[pd.Series, pd.DataFrame | None, pd.DataFrame | None, pd.Index]:
        """
        Create and validate inputs needed for the prediction process.

        This method prepares the inputs required by the predict methods,
        including validation of `last_window` and `exog`, and applying
        transformations if configured.
        
        Parameters
        ----------
        steps : int
            Number of steps to predict. 
        last_window : pandas Series, default None
            Series values used to create the predictors needed in the 
            predictions. If `last_window = None`, the values stored in 
            `self.last_window_` are used. 
            
            When provided, `last_window` must start right after the end of the 
            index seen by the forecaster during training. This is only supported 
            for skforecast.Sarimax estimator.
        last_window_exog : pandas Series, pandas DataFrame, default None
            Values of the exogenous variables aligned with `last_window`. Only
            needed when `last_window` is not None and the forecaster has been
            trained including exogenous variables. Must start at the end 
            of the training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.

        Returns
        -------
        last_window : pandas Series
            Transformed series values for prediction.
        last_window_exog : pandas DataFrame, None
            Transformed exogenous variables aligned with `last_window`.
        exog : pandas DataFrame, None
            Transformed exogenous variable/s for prediction.
        prediction_index : pandas Index
            Index for the predicted values, starting right after the end of 
            the training data.
        
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
            expected_index = expand_index(index=self.extended_index_, steps=1)[0]
            if expected_index != last_window.index[0]:
                raise ValueError(
                    f"To make predictions unrelated to the original data, `last_window` "
                    f"has to start at the end of the index seen by the forecaster.\n"
                    f"    Series last index         : {self.extended_index_[-1]}.\n"
                    f"    Expected index            : {expected_index}.\n"
                    f"    `last_window` index start : {last_window.index[0]}."
                )
            
            last_window = last_window.copy()
            last_window = transform_series(
                              series            = last_window,
                              transformer       = self.transformer_y,
                              fit               = False,
                              inverse_transform = False
                          )
            
            if last_window_exog is not None:
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
        
        # Prediction index starting right after the end of the training data
        prediction_index = expand_index(index=self.last_window_.index, steps=steps)

        return last_window, last_window_exog, exog, prediction_index

    def _check_append_last_window(
        self,
        steps: int,
        last_window: pd.Series,
        last_window_exog: pd.DataFrame | None
    ) -> pd.Index:
        """
        Handle the last_window logic for prediction methods.

        This method validates that SARIMAX estimators exist, warns about 
        unsupported estimators, appends the last_window data to SARIMAX 
        estimators, and returns the updated prediction index.

        Parameters
        ----------
        steps : int
            Number of steps to predict.
        last_window : pandas Series
            Transformed series values for prediction.
        last_window_exog : pandas DataFrame, None
            Transformed exogenous variables aligned with `last_window`.

        Returns
        -------
        prediction_index : pandas Index
            Updated index for the predicted values.

        """

        sarimax_indices = [
            i for i, estimator_type in enumerate(self.estimator_types_) 
            if estimator_type == 'skforecast.stats._sarimax.Sarimax'
        ]
        if not sarimax_indices:
            raise NotImplementedError(
                "Prediction with `last_window` parameter is only supported for "
                "skforecast.Sarimax estimator. The forecaster does not contain any "
                "estimator that supports this feature."
            )
        
        unsupported_last_window = [
            id for id, estimator_type in zip(self.estimator_ids, self.estimator_types_)
            if estimator_type not in self.estimators_support_last_window
        ]
        if unsupported_last_window:
            warnings.warn(
                f"Prediction with `last_window` is not implemented for estimators: {unsupported_last_window}. "
                f"These estimators will be skipped. Available estimators for prediction "
                f"using `last_window` are: {list(self.estimators_support_last_window)}.",
                IgnoredArgumentWarning
            )
        
        for i in sarimax_indices:
            self.estimators_[i].append(
                y     = last_window,
                exog  = last_window_exog,
                refit = False
            )

        self.extended_index_ = self.estimators_[sarimax_indices[0]].sarimax_res.fittedvalues.index
        prediction_index = expand_index(index=self.extended_index_, steps=steps)

        return prediction_index

    def predict(
        self,
        steps: int,
        last_window: pd.Series | None = None,
        last_window_exog: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        suppress_warnings: bool = False
    ) -> pd.Series | pd.DataFrame:
        """
        Forecast future values.

        Generate predictions (forecasts) n steps in the future using all 
        fitted estimators. If exogenous variables were used during training, 
        they must be provided for prediction.
        
        When using `last_window` and `last_window_exog`, they must start right 
        after the end of the index seen by the forecaster during training. 
        This feature is only supported for skforecast.Sarimax estimator; 
        other estimators will ignore `last_window` and predict from the end 
        of the training data.
        
        Parameters
        ----------
        steps : int
            Number of steps to predict. 
        last_window : pandas Series, default None
            Series values used to create the predictors needed in the 
            predictions. Used to make predictions unrelated to the original data. 
            Values must start at the end of the training data. Only supported 
            for skforecast.Sarimax estimator.
        last_window_exog : pandas Series, pandas DataFrame, default None
            Values of the exogenous variables aligned with `last_window`. Only
            needed when `last_window` is not None and the forecaster has been
            trained including exogenous variables. Values must start at the end 
            of the training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.
        suppress_warnings : bool, default False
            If `True`, skforecast warnings will be suppressed during the prediction 
            process. See skforecast.exceptions.warn_skforecast_categories for more
            information.

        Returns
        -------
        predictions : pandas Series, pandas DataFrame
            Predicted values from all estimators:

            - For multiple estimators: long format DataFrame with columns 
            'estimator_id' (estimator id) and 'pred' (predicted value).
            - For a single estimator: pandas Series with predicted values.
        
        """

        set_skforecast_warnings(suppress_warnings, action='ignore')

        last_window, last_window_exog, exog, prediction_index = (
            self._create_predict_inputs(
                steps            = steps,
                last_window      = last_window,
                last_window_exog = last_window_exog,
                exog             = exog,
            )
        )
        
        if last_window is not None:
            prediction_index = self._check_append_last_window(
                steps            = steps,
                last_window      = last_window,
                last_window_exog = last_window_exog
            )

        all_predictions = []
        estimator_ids = []
        for estimator, est_id, est_type in zip(
            self.estimators_, self.estimator_ids, self.estimator_types_
        ):
            if last_window is not None and est_type not in self.estimators_support_last_window:
                continue
            
            pred_func = self._predict_dispatch[est_type]
            preds = pred_func(estimator=estimator, steps=steps, exog=exog)
            all_predictions.append(preds)
            estimator_ids.append(est_id)

        predictions = transform_numpy(
                          array             = np.concatenate(all_predictions),
                          transformer       = self.transformer_y,
                          fit               = False,
                          inverse_transform = True
                      )

        if len(self.estimator_ids) == 1:
            predictions = pd.Series(
                              data  = predictions.ravel(),
                              index = prediction_index,
                              name  = 'pred'
                          )
        else:
            predictions = pd.DataFrame(
                {"estimator_id": np.repeat(estimator_ids, steps), "pred": predictions.ravel()},
                index = np.tile(prediction_index, len(estimator_ids)),
            )
        
        set_skforecast_warnings(suppress_warnings, action='default')

        return predictions

    def _predict_sarimax(
        self, 
        estimator: object, 
        steps: int, 
        exog: pd.Series | pd.DataFrame | None
    ) -> np.ndarray:
        """Generate predictions using SARIMAX statsmodels model."""
        preds = estimator.predict(steps=steps, exog=exog)['pred'].to_numpy()
        return preds

    def _predict_skforecast_stats(
        self, 
        estimator: object, 
        steps: int, 
        exog: pd.Series | pd.DataFrame | None
    ) -> np.ndarray:
        """Generate predictions using skforecast Arima/Arar/Ets models."""
        preds = estimator.predict(steps=steps, exog=exog)
        return preds

    def _predict_aeon(
        self, 
        estimator: object, 
        steps: int, 
        exog: pd.Series | pd.DataFrame | None
    ) -> np.ndarray:
        """Generate predictions using AEON models."""
        preds = estimator.iterative_forecast(
            y = self.last_window_.to_numpy(),
            prediction_horizon = steps
        )
        return preds

    def _predict_sktime_arima(
        self,
        estimator: object,
        steps: int,
        exog: pd.Series | pd.DataFrame | None
    ) -> np.ndarray:
        """Generate predictions using sktime ARIMA model."""
        fh = np.arange(1, steps + 1)
        preds = estimator.predict(fh=fh, X=exog).to_numpy()
        return preds

    def predict_interval(
        self,
        steps: int,
        last_window: pd.Series | None = None,
        last_window_exog: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        alpha: float = 0.05,
        interval: list[float] | tuple[float] | None = None,
        suppress_warnings: bool = False
    ) -> pd.DataFrame:
        """
        Forecast future values and their confidence intervals.

        Generate predictions (forecasts) n steps in the future with confidence
        intervals using fitted estimators that support prediction intervals. 
        If exogenous variables were used during training, they must be provided 
        for prediction.
        
        Estimators that do not support prediction intervals will be skipped 
        with a warning. Supported estimators for intervals are the ones listed
        in the attribute `estimators_support_intervals`.

        When using `last_window` and `last_window_exog`, they must start right 
        after the end of the index seen by the forecaster during training. 
        This feature is only supported for skforecast.Sarimax estimator; 
        other estimators will ignore `last_window` and predict from the end 
        of the training data.

        Parameters
        ----------
        steps : int
            Number of steps to predict. 
        last_window : pandas Series, default None
            Series values used to create the predictors needed in the 
            predictions. Used to make predictions unrelated to the original data. 
            Values must start at the end of the training data. Only supported 
            for skforecast.Sarimax estimator.
        last_window_exog : pandas Series, pandas DataFrame, default None
            Values of the exogenous variables aligned with `last_window`. Only
            needed when `last_window` is not None and the forecaster has been
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
        suppress_warnings : bool, default False
            If `True`, skforecast warnings will be suppressed during the prediction 
            process. See skforecast.exceptions.warn_skforecast_categories for more
            information.

        Returns
        -------
        predictions : pandas DataFrame
            Predicted values from estimators that support intervals and their 
            estimated intervals:
            
            - For multiple estimators: long format DataFrame with columns
              'estimator_id', 'pred', 'lower_bound', 'upper_bound'.
            - For a single estimator: DataFrame with columns
              'pred', 'lower_bound', 'upper_bound'.

        """

        set_skforecast_warnings(suppress_warnings, action='ignore')

        # If interval and alpha take alpha, if interval transform to alpha
        if alpha is None:
            if 100 - interval[1] != interval[0]:
                raise ValueError(
                    f"When using `interval` in ForecasterStats, it must be symmetrical. "
                    f"For example, interval of 95% should be as `interval = [2.5, 97.5]`. "
                    f"Got {interval}."
                )
            alpha = 2 * (100 - interval[1]) / 100

        last_window, last_window_exog, exog, prediction_index = (
            self._create_predict_inputs(
                steps            = steps,
                last_window      = last_window,
                last_window_exog = last_window_exog,
                exog             = exog,
            )
        )

        if last_window is not None:
            prediction_index = self._check_append_last_window(
                steps            = steps,
                last_window      = last_window,
                last_window_exog = last_window_exog
            )

        unsupported_interval = [
            id for id, est_type in zip(self.estimator_ids, self.estimator_types_)
            if est_type not in self.estimators_support_interval
        ]
        if unsupported_interval:
            warnings.warn(
                f"Interval prediction is not implemented for estimators: {unsupported_interval}. "
                f"These estimators will be skipped. Available estimators for prediction "
                f"intervals are: {list(self.estimators_support_interval)}.",
                IgnoredArgumentWarning
            )
        
        all_predictions = []
        estimator_ids = []
        for estimator, est_id, est_type in zip(
            self.estimators_, self.estimator_ids, self.estimator_types_
        ):
            
            if est_type not in self.estimators_support_interval:
                continue
            if last_window is not None and est_type not in self.estimators_support_last_window:
                continue
                
            pred_func = self._predict_interval_dispatch[est_type]
            preds = pred_func(estimator=estimator, steps=steps, exog=exog, alpha=alpha)
            all_predictions.append(preds)
            estimator_ids.append(est_id)

        predictions = transform_numpy(
                          array             = np.concatenate(all_predictions),
                          transformer       = self.transformer_y,
                          fit               = False,
                          inverse_transform = True
                      )

        predictions = pd.DataFrame(
                          data  = predictions,
                          index = np.tile(prediction_index, len(estimator_ids)),
                          columns = ['pred', 'lower_bound', 'upper_bound']
                      )
        
        if len(self.estimator_ids) > 1:
            predictions.insert(0, 'estimator_id', np.repeat(estimator_ids, steps))
        else:
            # This is done to restore the frequency
            predictions.index = prediction_index
        
        set_skforecast_warnings(suppress_warnings, action='default')

        return predictions

    def _predict_interval_sarimax(
        self,
        estimator: object,
        steps: int, 
        exog: pd.Series | pd.DataFrame | None,
        alpha: float
    ) -> np.ndarray:
        """Generate prediction intervals using SARIMAX statsmodels model."""
        preds = estimator.predict(
            steps=steps, exog=exog, return_conf_int=True, alpha=alpha
        ).to_numpy()
        return preds

    def _predict_interval_skforecast_stats(
        self,
        estimator: object,
        steps: int, 
        exog: pd.Series | pd.DataFrame | None,
        alpha: float
    ) -> np.ndarray:
        """Generate prediction intervals using skforecast Arima/Arar/Ets models."""
        preds = estimator.predict_interval(
            steps    = steps,
            exog     = exog,
            level    = [100 * (1 - alpha)],
            as_frame = False
        )
        return preds
    
    def _predict_interval_sktime_arima(
        self,
        estimator: object,
        steps: int,
        exog: pd.Series | pd.DataFrame | None,
        alpha: float
    ) -> np.ndarray:
        """Generate prediction intervals using sktime ARIMA model."""
        fh = np.arange(1, steps + 1)
        preds = estimator.predict_interval(fh=fh, X=exog, coverage=1 - alpha).to_numpy()
        return preds

    # TODO: Add get_params and set_params for each estimator when multiple estimators are supported
    def set_params(
        self, 
        params: dict[str, object] | dict[str, dict[str, object]]
    ) -> None:
        """
        Set new values to the parameters of the model stored in the forecaster.
        
        Parameters
        ----------
        params : dict
            Parameters values. The expected format depends on the number of
            estimators in the forecaster:
            
            - Single estimator: A dictionary with parameter names as keys 
            and their new values as values.
            - Multiple estimators: A dictionary where each key is an 
            estimator id (as shown in `estimator_ids`) and each value 
            is a dictionary of parameters for that estimator.

        Returns
        -------
        None
        
        """
        
        if self.n_estimators == 1:
            # Single estimator: params is a simple dict of parameter values
            self.estimators[0] = clone(self.estimators[0])
            self.estimators[0].set_params(**params)
        else:
            # Multiple estimators: params must be a dict of dicts keyed by estimator name
            if not isinstance(params, dict):
                raise TypeError(
                    f"`params` must be a dictionary. Got {type(params).__name__}."
                )
            
            provided_ids = set(params.keys())
            valid_ids = set(self.estimator_ids)
            invalid_ids = provided_ids - valid_ids
            if invalid_ids == provided_ids:
                raise ValueError(
                    f"None of the provided estimator ids {list(invalid_ids)} "
                    f"match the available estimator ids: {self.estimator_ids}."
                )
            if invalid_ids:
                warnings.warn(
                    f"The following estimator ids do not match any estimator "
                    f"in the forecaster and will be ignored: {list(invalid_ids)}. "
                    f"Available estimator ids are: {self.estimator_ids}.",
                    IgnoredArgumentWarning
                )
            
            for est_id, est_params in params.items():
                if est_id in valid_ids:
                    idx = self.estimator_ids.index(est_id)
                    self.estimators[idx] = clone(self.estimators[idx])
                    self.estimators[idx].set_params(**est_params)

    def set_fit_kwargs(
        self, 
        fit_kwargs: Any = None
    ) -> None:
        """
        Set new values for the additional keyword arguments passed to the `fit` 
        method of the estimator.
        
        Parameters
        ----------
        fit_kwargs : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        None
        
        """

        if self.estimator_type == 'skforecast.stats._sarimax.Sarimax':
            warnings.warn(
                "When using the skforecast Sarimax model, the fit kwargs should "
                "be passed using the model parameter `sm_fit_kwargs`.",
                IgnoredArgumentWarning
            )

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
        feature_importances = []
        for estimator, estimator_type, estimator_id in zip(
            self.estimators_, self.estimator_types_, self.estimator_ids
        ):
            get_importances_method = self._feature_importances_dispatch[estimator_type]
            importance = get_importances_method(estimator)
            if importance is not None:
                importance.insert(0, 'estimator_id', estimator_id)
                feature_importances.append(importance)

        feature_importances = pd.concat(feature_importances, ignore_index=True)

        if sort_importance:
            feature_importances = feature_importances.sort_values(
                                      by=['estimator_id', 'importance'],
                                      ascending=False
                                  ).reset_index(drop=True)

        return feature_importances

    @staticmethod
    def _get_feature_importances_sarimax(estimator) -> pd.DataFrame:
        """Get feature importances for SARIMAX statsmodels model."""

        return estimator.get_feature_importances()
    
    @staticmethod
    def _get_feature_importances_arima(estimator) -> pd.DataFrame:
        """Get feature importances for Arima model."""

        return estimator.get_feature_importances()

    @staticmethod
    def _get_feature_importances_arar(estimator) -> pd.DataFrame:
        """Get feature importances for Arar model."""
        
        return estimator.get_feature_importances()

    @staticmethod
    def _get_feature_importances_ets(estimator) -> pd.DataFrame:
        """Get feature importances for Eta model."""

        return estimator.get_feature_importances()

    @staticmethod
    def _get_feature_importances_aeon_arima(estimator) -> pd.DataFrame:
        """Get feature importances for AEON ARIMA model."""
        return pd.DataFrame({
            'feature': [f'lag_{lag}' for lag in range(1, estimator.p + 1)] + ["ma", "intercept"],
            'importance': np.concatenate([estimator.phi_, estimator.theta_, [estimator.c_]])
        })

    @staticmethod
    def _get_feature_importances_aeon_ets(estimator) -> pd.DataFrame:
        """Get feature importances for AEON ETS model."""
        warnings.warn("Feature importances is not available for the AEON ETS model.")
        return pd.DataFrame(columns=['feature', 'importance'])
    
    @staticmethod
    def _get_feature_importances_sktime_arima(estimator) -> pd.DataFrame:
        """Get feature importances for sktime ARIMA model."""
        feature_importances = estimator._forecaster.params().to_frame().reset_index()
        feature_importances.columns = ['feature', 'importance']
        return feature_importances

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

        if not self.is_fitted:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `get_info_criteria()`."
            )
        info_criteria = []
        for estimator, estimator_type in zip(self.estimators_, self.estimator_types_):
            get_criteria_method = self._info_criteria_dispatch[estimator_type]
            value = get_criteria_method(estimator, criteria, method)
            info_criteria.append(value)

        results = pd.DataFrame({
            'estimator_id': self.estimator_ids,
            'criteria': criteria,
            'value': info_criteria
        })
        
        return results

    @staticmethod
    def _get_info_criteria_sarimax(estimator, criteria: str, method: str) -> float:
        """Get information criteria for SARIMAX statsmodels model."""
       
        return estimator.get_info_criteria(criteria=criteria, method=method)
    
    @staticmethod
    def _get_info_criteria_arima(estimator, criteria: str, method: str) -> float:
        """Get information criteria for Arima model."""

        return estimator.get_info_criteria(criteria=criteria)

    @staticmethod
    def _get_info_criteria_arar(estimator, criteria: str, method: str) -> float:
        """Get information criteria for Arar model."""
        
        return estimator.get_info_criteria(criteria=criteria)

    @staticmethod
    def _get_info_criteria_ets(estimator, criteria: str, method: str) -> float:
        """Get information criteria for skforecast Ets model."""
        
        return estimator.get_info_criteria(criteria=criteria)
    
    @staticmethod
    def _get_info_criteria_sktime_arima(estimator, criteria: str, method: str) -> float:
        """Get information criteria for sktime ARIMA model."""
        if criteria not in {'aic', 'bic', 'hqic'}:
            raise ValueError("`criteria` must be one of {'aic','bic','hqic'}")
        if method not in {'standard', 'lutkepohl'}:
            raise ValueError("`method` must be either 'standard' or 'lutkepohl'")

        return estimator._forecaster.arima_res_.info_criteria(criteria=criteria, method=method)

    @staticmethod
    def _get_info_criteria_aeon(estimator, criteria: str, method: str) -> float:
        """Get information criteria for AEON models."""
        if criteria != 'aic':
            raise ValueError(
                "Invalid value for `criteria`. Only 'aic' is supported for "
                "AEON models."
            )
        
        return estimator.aic_

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
        
        if not self.is_fitted:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `reduce_memory()`."
            )

        unsupported_reduce_memory = [
            est_id for est_id, estimator_type in zip(self.estimator_ids, self.estimator_types_)
            if estimator_type not in self.estimators_support_reduce_memory
        ]
        if unsupported_reduce_memory:
            warnings.warn(
                f"Memory reduction is not implemented for estimators: {unsupported_reduce_memory}. "
                f"These estimators will be skipped. Available estimators for memory "
                f"reduction are: {list(self.estimators_support_reduce_memory)}.",
                IgnoredArgumentWarning
            )
            
        for estimator, est_type in zip(self.estimators_, self.estimator_types_):
            if est_type in self.estimators_support_reduce_memory:
                estimator.reduce_memory()
