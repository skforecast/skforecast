################################################################################
#                                ForecasterBase                                #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional
import uuid
import textwrap
from IPython.display import display, HTML
import pandas as pd
from sklearn.pipeline import Pipeline


class ForecasterBase(ABC):
    """
    Base class for all forecasters in skforecast. All forecasters should specify
    all the parameters that can be set at the class level in their ``__init__``.     
    """

    def _preprocess_repr(
        self,
        regressor: object,
        training_range_: Optional[dict] = None,
        series_names_in_: Optional[list] = None,
        exog_names_in_: Optional[list] = None,
        transformer_series: Optional[Union[str, dict]] = None
    ) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[Union[str, object]]]:
        """
        Prepare the information to be displayed when a Forecaster object is printed.

        Parameters
        ----------
        regressor : object
            Regressor object.
        training_range_ : dict, default `None`
            Training range. Only used for `ForecasterRecursiveMultiSeries`.
        series_names_in_ : list, default `None`
            Names of the series used in the forecaster. Only used for `ForecasterRecursiveMultiSeries`.
        exog_names_in_ : list, default `None`
            Names of the exogenous variables used in the forecaster.
        transformer_series : str, dict, default `None`
            Transformer used in the series. Only used for `ForecasterRecursiveMultiSeries`.
        
        """

        if isinstance(regressor, Pipeline):
            name_pipe_steps = tuple(name + "__" for name in regressor.named_steps.keys())
            params = {key: value for key, value in regressor.get_params().items() 
                      if key.startswith(name_pipe_steps)}
        else:
            params = regressor.get_params()
        params = str(params)

        if training_range_ is not None:
            training_range_ = [
                f"'{k}': {v.astype(str).to_list()}" 
                for k, v in training_range_.items()
            ]
            if len(training_range_) > 10:
                training_range_ = training_range_[:5] + ['...'] + training_range_[-5:]
            training_range_ = ", ".join(training_range_)

        if series_names_in_ is not None:
            if len(series_names_in_) > 50:
                series_names_in_ = series_names_in_[:25] + ["..."] + series_names_in_[-25:]
            series_names_in_ = ", ".join(series_names_in_)

        if exog_names_in_ is not None:
            if len(exog_names_in_) > 50:
                exog_names_in_ = exog_names_in_[:25] + ["..."] + exog_names_in_[-25:]
            exog_names_in_ = ", ".join(exog_names_in_)
        
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
        max_text_length : int, default `58`
            Maximum length of the text before wrapping.
        width : int, default `80`
            Maximum width of the text.
        indent : str, default `"    "`
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
    
    def _get_style_repr_html(
        self, 
        is_fitted: bool
    ) -> Tuple[str, str]:
        """
        Return style and unique_id for HTML representation.

        Parameters
        ----------
        is_fitted : bool
            If the forecaster has been fitted.
        
        Returns
        -------
        style : str
            CSS style.
        unique_id : str
            Unique id for the HTML container.
        
        """

        unique_id = str(uuid.uuid4()).replace('-', '')
        background_color = "#f0f8ff" if is_fitted else "#f9f1e2"
        section_color = "#b3dbfd" if is_fitted else "#fae3b3"

        style = f"""
        <style>
            .container-{unique_id} {{
                font-family: 'Arial', sans-serif;
                font-size: 0.9em;
                color: #333;
                border: 1px solid #ddd;
                background-color: {background_color};
                padding: 5px 15px;
                border-radius: 8px;
                max-width: 600px;
                #margin: auto;
            }}
            .container-{unique_id} h2 {{
                font-size: 1.5em;
                color: #222;
                border-bottom: 2px solid #ddd;
                padding-bottom: 5px;
                margin-bottom: 15px;
            }}
            .container-{unique_id} details {{
                margin: 10px 0;
            }}
            .container-{unique_id} summary {{
                font-weight: bold;
                font-size: 1.1em;
                cursor: pointer;
                margin-bottom: 5px;
                background-color: {section_color};
                padding: 5px;
                border-radius: 5px;
            }}
            .container-{unique_id} summary:hover {{
                background-color: #e0e0e0;
            }}
            .container-{unique_id} ul {{
                font-family: 'Courier New', monospace;
                list-style-type: none;
                padding-left: 20px;
                margin: 10px 0;
            }}
            .container-{unique_id} li {{
                margin: 5px 0;
                font-family: 'Courier New', monospace;
            }}
            .container-{unique_id} li strong {{
                font-weight: bold;
                color: #444;
            }}
            .container-{unique_id} li::before {{
                content: "- ";
                color: #666;
            }}
            .container-{unique_id} a {{
                color: #001633;
                text-decoration: none;
            }}
            .container-{unique_id} a:hover {{
                color: #359ccb; 
            }}
        </style>
        """

        return style, unique_id

    @abstractmethod
    def create_train_X_y(
        self,
        y: pd.Series,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create training matrices from univariate time series and exogenous
        variables.

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
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None
    ) -> None:
        """
        Training Forecaster.
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        exog : pandas Series, pandas DataFrame, default `None`
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
        last_window: Optional[pd.Series] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None
    ) -> pd.Series:
        """
        Predict n steps ahead.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        last_window : pandas Series, default `None`
            Series values used to create the predictors (lags) needed in the 
            first iteration of the prediction (t + 1).
            If `last_window = None`, the values stored in `self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
        exog : pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s.

        Returns
        -------
        predictions : pandas Series
            Predicted values.
        
        """

        pass
        
    @abstractmethod
    def set_params(self, params: dict) -> None:
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
        
        pass
        
    def set_lags(self, lags: int) -> None:
        """
        Set new value to the attribute `lags`.
        Attributes `max_lag` and `window_size` are also updated.
        
        Parameters
        ----------
        lags : int, list, numpy ndarray, range
            Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.

            - `int`: include lags from 1 to `lags` (included).
            - `list`, `1d numpy ndarray` or `range`: include only lags present in 
            `lags`, all elements must be int.

        Returns
        -------
        None
        
        """
        
        pass

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
        
        # If environment supports HTML (like Jupyter Notebook)
        if hasattr(self, '_repr_html_') and callable(getattr(self, '_repr_html_')):
            display(HTML(self._repr_html_()))
        else:
            # Fall back to __repr__ if _repr_html_ is not available
            print(self.__repr__())
