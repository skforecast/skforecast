################################################################################
#                                ForecasterBase                                #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import html
import textwrap
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


_CAT_BADGE = (
    '<span title="categorical" style="background-color: #FFA726; color: white; '
    'border-radius: 3px; padding: 1px 5px; font-size: 0.75em; '
    'margin-left: 4px; font-weight: bold; letter-spacing: 0.03em; '
    'vertical-align: middle;">CAT</span>'
)


class ForecasterBase(ABC):
    """
    Base class for all forecasters in skforecast. All forecasters should specify
    all the parameters that can be set at the class level in their `__init__`.     
    """

    def _preprocess_repr(
        self,
        estimator: object | None = None,
        training_range_: dict[str, str] | None = None,
        series_names_in_: list[str] | None = None,
        exog_names_in_: list[str] | None = None,
        transformer_series: object | dict[str, object] | None = None,
        categorical_features_names_in_: list[str] | None = None,
        as_html: bool = False
    ) -> tuple[str, str | None, str | None, str | None, str | None]:
        """
        Prepare the information to be displayed when a Forecaster object is printed.

        Parameters
        ----------
        estimator : object, default None
            Estimator object.
        training_range_ : dict, default None
            Training range. Only used for `ForecasterRecursiveMultiSeries`.
        series_names_in_ : list, default None
            Names of the series used in the forecaster. Only used for `ForecasterRecursiveMultiSeries`.
        exog_names_in_ : list, default None
            Names of the exogenous variables used in the forecaster.
        transformer_series : object, dict, default None
            Transformer used in the series. Only used for `ForecasterRecursiveMultiSeries`.
        categorical_features_names_in_ : list, default None
            Names of the exogenous variables considered as categorical.
            Only used when `as_html` is `True`.
        as_html : bool, default False
            If `True`, exogenous variable names are formatted as HTML with
            CAT badges and html-escaping.

        Returns
        -------
        params : str, None
            Parameters of the estimator.
        training_range_ : str, None
            Training range. Only used for `ForecasterRecursiveMultiSeries`.
        series_names_in_ : str, None
            Names of the series used in the forecaster. Only used for multi-series forecasters.
        exog_names_in_ : str, None
            Names of the exogenous variables used in the forecaster.
        transformer_series : str, None
            Transformer used in the series. Only used for multi-series forecasters.
        
        """

        if estimator is not None:
            if isinstance(estimator, Pipeline):
                name_pipe_steps = tuple(name + "__" for name in estimator.named_steps.keys())
                params = {
                    key: value for key, value in estimator.get_params().items() 
                    if key.startswith(name_pipe_steps)
                }
            else:
                params = estimator.get_params()
            params = str(params)
        else:
            params = None

        if training_range_ is not None:
            training_range_ = [
                f"'{k}': {v.astype(str).to_list()}" 
                for k, v in training_range_.items()
            ]
            if len(training_range_) > 10:
                training_range_ = training_range_[:5] + ['...'] + training_range_[-5:]
            training_range_ = ", ".join(training_range_)

        max_items = 50
        if series_names_in_ is not None:
            if len(series_names_in_) > max_items:
                series_names_in_ = (
                    series_names_in_[: max_items // 2]
                    + ["..."]
                    + series_names_in_[-max_items // 2 :]
                )
            series_names_in_ = ", ".join(series_names_in_)

        if exog_names_in_ is not None:
            if len(exog_names_in_) > max_items:
                exog_names_in_ = (
                    exog_names_in_[: max_items // 2]
                    + ['...']
                    + exog_names_in_[-max_items // 2 :]
                )
            if as_html:
                cat_set = set(categorical_features_names_in_ or [])
                parts = []
                for name in exog_names_in_:
                    if name == '...':
                        parts.append('<em style="color: #999;">\u2026</em>')
                    elif name in cat_set:
                        parts.append(f'{html.escape(str(name))}{_CAT_BADGE}')
                    else:
                        parts.append(html.escape(str(name)))
                exog_names_in_ = ', '.join(parts)
            else:
                exog_names_in_ = ', '.join(exog_names_in_)
        
        if transformer_series is not None:
            if isinstance(transformer_series, dict):
                transformer_series = [
                    f"'{k}': {v}" 
                    for k, v in transformer_series.items()
                ]
                if len(transformer_series) > 10:
                    transformer_series = transformer_series[:5] + ["..."] + transformer_series[-5:]
                transformer_series = ", ".join(transformer_series)
            else:
                transformer_series = str(transformer_series)        

        return params, training_range_, series_names_in_, exog_names_in_, transformer_series

    def _format_text_repr(
        self, 
        text: str, 
        max_text_length: int = 58,
        width: int = 80, 
        indent: str = "    "
    ) -> str:
        """
        Format text for __repr__ method.

        Parameters
        ----------
        text : str
            Text to format.
        max_text_length : int, default 58
            Maximum length of the text before wrapping.
        width : int, default 80
            Maximum width of the text.
        indent : str, default "    "
            Indentation of the text.
        
        Returns
        -------
        text : str
            Formatted text.

        """

        if text is not None and len(text) > max_text_length:
            text = "\n    " + textwrap.fill(
                str(text), width=width, subsequent_indent=indent
            )
        
        return text

    @abstractmethod
    def create_train_X_y(
        self,
        y: pd.Series,
        exog: pd.Series | pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Create training matrices from univariate time series and exogenous
        variables.

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
            Training values (predictors).
            Shape: (len(y) - self.max_lag, len(self.lags))
        y_train : pandas Series
            Values (target) of the time series related to each row of `X_train`.
            Shape: (len(y) - self.max_lag, )
        
        """
        
        pass

    @abstractmethod
    def fit(
        self,
        y: pd.Series,
        exog: pd.Series | pd.DataFrame | None = None
    ) -> None:
        """
        Training Forecaster.
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and their indexes must be aligned so
            that y[i] is regressed on exog[i].

        Returns
        -------
        None
        
        """
        
        pass

    @abstractmethod        
    def predict(
        self,
        steps: int,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None
    ) -> pd.Series:
        """
        Predict n steps ahead.
        
        Parameters
        ----------
        steps : int
            Number of steps to predict. 
        last_window : pandas Series, pandas DataFrame, default None
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.

        Returns
        -------
        predictions : pandas Series
            Predicted values.
        
        """

        pass
        
    @abstractmethod
    def set_params(self, params: dict[str, object]) -> None:
        """
        Set new values to the parameters of the scikit-learn model stored in the
        forecaster.
        
        Parameters
        ----------
        params : dict
            Parameters values.

        Returns
        -------
        None
        
        """
        
        pass
        
    def set_lags(
        self, 
        lags: int | list[int] | np.ndarray[int] | range[int] | None = None
    ) -> None:
        """
        Set new value to the attribute `lags`.
        Attributes `max_lag` and `window_size` are also updated.
        
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
        
        pass

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

        pass

    def get_tags(self) -> dict[str, Any]:
        """
        Return the tags that characterize the behavior of the forecaster.

        Returns
        -------
        skforecast_tags : dict
            Dictionary with forecaster tags.

        """

        return self.__skforecast_tags__

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
        
        print(self.__repr__())
