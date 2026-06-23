################################################################################
#                             ForecasterFoundation                             #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import html
import sys
import textwrap
import warnings
import pandas as pd
from sklearn.exceptions import NotFittedError

from .. import __version__
from ..exceptions import IgnoredArgumentWarning
from ..utils import (
    _normalize_interval_scale,
    check_interval,
    get_style_repr_html,
)
from ._foundation_model import FoundationModel


class ForecasterFoundation:
    """
    Forecaster that wraps a `FoundationModel` [1]_ for full skforecast ecosystem
    compatibility: backtesting, model selection, etc.

    Unlike ML-based forecasters, there is no training step — the underlying
    foundation models are zero-shot. `fit` only stores the context
    (recent observations) and records index metadata. Predictions are generated 
    directly by the model's `predict_quantiles` pipeline.

    Supports both single-series and multi-series modes. Pass a `pandas.Series`
    to `fit` for single-series forecasting or a wide `pandas.DataFrame`, a
    long-format `pandas.DataFrame` (MultiIndex), or a `dict[str, pd.Series]`
    for multi-series (global-model) forecasting.

    Parameters
    ----------
    estimator : FoundationModel
        A configured `FoundationModel` instance, e.g.
        `FoundationModel("autogluon/chronos-2-small", context_length=512)`.
        See `FoundationModel` for the list of supported `model_id` values
        and adapter-specific parameters.
    forecaster_id : str, int, default None
        Name used as an identifier of the forecaster.

    Attributes
    ----------
    estimator : FoundationModel
        The `FoundationModel` instance provided by the user.
    model_id : str
        HuggingFace model ID. Delegates to `estimator.model_id`.
    context_ : dict
        Per-series dict of pandas Series containing the last `context_length`
        observations from the training data. Delegates to
        `estimator.context_`. `None` before fitting.
    context_exog_ : dict
        Per-series dict of pandas DataFrame containing the last
        `context_length` exogenous variables from the training data.
        Delegates to `estimator.context_exog_`. `None` before fitting or
        if no exogenous variables were provided.
    last_window_ : dict
        Alias for `context_`.
    last_window_exog_ : dict
        Alias for `context_exog_`.
    context_length : int
        Maximum number of historical observations used as context. Delegates
        to `estimator.context_length`.
    window_size : int
        Desired number of historical observations used as context by the
        model. Always equals `context_length`.
    index_type_ : type
        Type of index of the input used in training. Delegates to
        `estimator.index_type_`.
    index_freq_ : pandas DateOffset, int
        Frequency of the index of the input used in training. A
        `pandas.DateOffset` for `DatetimeIndex`; the `step` integer
        for `RangeIndex`. Delegates to `estimator.index_freq_`.
    context_range_ : dict
        First and last values of index of the data used during training.
        A `dict` keyed by series name with `pandas.Index` values.
        Delegates to `estimator.context_range_`.
    series_names_in_ : list
        Names of the series (levels) provided by the user during training.
        Delegates to `estimator.series_names_in_`.
    is_multiple_series_ : bool
        Whether the forecaster was fitted with multiple series. Delegates
        to `estimator.is_multiple_series_`.
    exog_in_ : bool
        If the forecaster has been trained using exogenous variable/s.
        Delegates to `estimator.exog_in_`.
    exog_names_in_ : list
        Names of the exogenous variables used during training. Delegates
        to `estimator.exog_names_in_`.
    exog_names_in_per_series_ : dict
        Names of the exogenous variables used during training for each
        series. Delegates to `estimator.exog_names_in_per_series_`.
    exog_type_in_ : type
        Type of exogenous variable/s used in training. Delegates to
        `estimator.exog_type_in_`.
    creation_date : str
        Date of creation.
    is_fitted : bool
        Tag to identify if the forecaster has been fitted (trained).
    fit_date : str
        Date of last fit. Delegates to `estimator.fit_date`.
    skforecast_version : str
        Version of skforecast library used to create the forecaster.
    python_version : str
        Version of python used to create the forecaster.
    forecaster_id : str, int
        Name used as an identifier of the forecaster.
    __skforecast_tags__ : dict
        Tags associated with the forecaster.

    References
    ----------
    .. [1] FoundationModel and adapters:
           https://skforecast.org/latest/api/foundationmodel.html

    """

    def __init__(
        self,
        estimator: FoundationModel,
        forecaster_id: str | int | None = None,
    ) -> None:

        if not isinstance(estimator, FoundationModel):
            raise TypeError(
                f"`estimator` must be a `FoundationModel` instance. "
                f"Got {type(estimator)}."
            )

        self.estimator          = estimator
        self.creation_date      = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.is_fitted          = False
        self.skforecast_version = __version__
        self.python_version     = sys.version.split(" ")[0]
        self.forecaster_id      = forecaster_id

        self.__skforecast_tags__ = {
            "library": "skforecast",
            "forecaster_name": "ForecasterFoundation",
            "forecaster_task": "regression",
            "forecasting_scope": "single-series | global",
            "forecasting_strategy": "foundation",
            "multiple_estimators": False, 
            "index_types_supported": ["pandas.RangeIndex", "pandas.DatetimeIndex"],
            "requires_index_frequency": True,

            "allowed_input_types_series": [
                "pandas.Series",
                "pandas.DataFrame",
                "long-format pandas.DataFrame",
                "dict[str, pandas.Series]",
            ],
            "supports_exog": True,
            "allowed_input_types_exog": [
                "pandas.Series",
                "pandas.DataFrame",
                "long-format pandas.DataFrame",
                "dict[str, pandas.Series | pandas.DataFrame | None]",
            ],
            "handles_missing_values_series": True,
            "handles_missing_values_exog": True,

            "supports_lags": False,
            "supports_window_features": False,
            "supports_calendar_features": False,
            "supports_transformer_series": False,
            "supports_transformer_exog": False,
            "supports_categorical_features": True,
            "supports_weight_func": False,
            "supports_differentiation": False,

            "prediction_types": ["point", "interval", "quantiles"],
            "supports_probabilistic": True,
            "probabilistic_methods": ["quantile_native"],
            "handles_binned_residuals": False,
        }

    @property
    def context_length(self) -> int:
        """
        Maximum number of historical observations used as context.

        Returns
        -------
        context_length : int
            Maximum context length. Delegates to `estimator.context_length`.
        """
        return self.estimator.context_length

    @property
    def model_id(self) -> str:
        """
        HuggingFace model ID.

        Returns
        -------
        model_id : str
            HuggingFace model ID. Delegates to `estimator.model_id`.
        """
        return self.estimator.model_id

    @property
    def window_size(self) -> int:
        """
        Desired number of historical observations used as context by the
        model. Always equals `context_length`.

        Returns
        -------
        window_size : int
            Context window size. Delegates to `estimator.context_length`.
        """
        return self.estimator.context_length

    @property
    def context_(self) -> dict[str, pd.Series] | None:
        """
        Per-series context stored during `fit`.

        Returns
        -------
        context_ : dict, None
            Per-series dict of pandas Series containing the last
            `context_length` observations from the training data.
            Delegates to `estimator.context_`. `None` before fitting.
        """
        return self.estimator.context_ if self.is_fitted else None

    @property
    def last_window_(self) -> dict[str, pd.Series] | None:
        """
        Alias for `context_`.

        Returns
        -------
        last_window_ : dict, None
            Per-series dict of pandas Series. Alias for `context_`.
        """
        return self.context_

    @property
    def context_exog_(self) -> dict[str, pd.DataFrame] | None:
        """
        Per-series exogenous context stored during `fit`.

        Returns
        -------
        context_exog_ : dict, None
            Per-series dict of pandas DataFrame containing the last
            `context_length` exogenous variables from the training data.
            Delegates to `estimator.context_exog_`. `None` before fitting
            or if no exogenous variables were provided.
        """
        return self.estimator.context_exog_ if self.is_fitted else None

    @property
    def last_window_exog_(self) -> dict[str, pd.DataFrame] | None:
        """
        Alias for `context_exog_`.

        Returns
        -------
        last_window_exog_ : dict, None
            Per-series dict of pandas DataFrame. Alias for `context_exog_`.
        """
        return self.context_exog_

    @property
    def index_type_(self) -> type | None:
        """
        Type of index of the input used in training.

        Returns
        -------
        index_type_ : type, None
            Index type. Delegates to `estimator.index_type_`.
        """
        return self.estimator.index_type_

    @property
    def index_freq_(self) -> object:
        """
        Frequency of the index of the input used in training.

        Returns
        -------
        index_freq_ : pandas DateOffset, int, None
            Index frequency. Delegates to `estimator.index_freq_`.
        """
        return self.estimator.index_freq_

    @property
    def context_range_(self) -> dict[str, pd.Index] | None:
        """
        First and last values of index of the data used during training.

        Returns
        -------
        context_range_ : dict, None
            Per-series index range. Delegates to `estimator.context_range_`.
        """
        return self.estimator.context_range_

    @property
    def series_names_in_(self) -> list[str] | None:
        """
        Names of the series (levels) provided by the user during training.

        Returns
        -------
        series_names_in_ : list, None
            Series names. Delegates to `estimator.series_names_in_`.
        """
        return self.estimator.series_names_in_

    @property
    def is_multiple_series_(self) -> bool:
        """
        Whether the forecaster was fitted with multiple series.

        Returns
        -------
        is_multiple_series_ : bool
            Delegates to `estimator.is_multiple_series_`.
        """
        return self.estimator.is_multiple_series_

    @property
    def exog_in_(self) -> bool:
        """
        If the forecaster has been trained using exogenous variable/s.

        Returns
        -------
        exog_in_ : bool
            Delegates to `estimator.exog_in_`.
        """
        return self.estimator.exog_in_

    @property
    def exog_names_in_(self) -> list[str] | None:
        """
        Names of the exogenous variables used during training.

        Returns
        -------
        exog_names_in_ : list, None
            Delegates to `estimator.exog_names_in_`.
        """
        return self.estimator.exog_names_in_

    @property
    def exog_names_in_per_series_(self) -> dict | None:
        """
        Names of the exogenous variables used during training for each series.

        Returns
        -------
        exog_names_in_per_series_ : dict, None
            Delegates to `estimator.exog_names_in_per_series_`.
        """
        return self.estimator.exog_names_in_per_series_

    @property
    def exog_type_in_(self) -> type | None:
        """
        Type of exogenous variable/s used in training.

        Returns
        -------
        exog_type_in_ : type, None
            Delegates to `estimator.exog_type_in_`.
        """
        return self.estimator.exog_type_in_

    @property
    def fit_date(self) -> str | None:
        """
        Date of last fit.

        Returns
        -------
        fit_date : str, None
            Delegates to `estimator.fit_date`.
        """
        return self.estimator.fit_date

    @staticmethod
    def _truncate_names(
        names: list[str] | None,
        max_items: int = 50,
    ) -> list[str] | None:
        """
        Truncate a list of names for display.

        Returns the first and last `max_items // 2` elements (joined by
        `'...'`) when the list exceeds `max_items`. Returns a shallow
        copy so callers can mutate the result freely.

        Parameters
        ----------
        names : list, None
            Names to truncate. If `None`, returns `None`.
        max_items : int, default 50
            Maximum number of names to keep before truncation.

        Returns
        -------
        truncated : list, None
            Truncated list, or `None` if `names` is `None`.

        """

        if names is None:
            return None

        names = list(names)
        if len(names) > max_items:
            half = max_items // 2
            names = names[:half] + ["..."] + names[-half:]

        return names

    @staticmethod
    def _format_names_repr(
        names: list[str] | None,
        max_items: int = 50,
        max_text_length: int = 58,
    ) -> str | None:
        """
        Format a list of names for text `__repr__`.

        Truncates the list to the first and last `max_items // 2` elements
        (joined by `'...'`) when it exceeds `max_items`, then wraps the
        resulting string with `textwrap.fill` when it exceeds
        `max_text_length` characters.

        Parameters
        ----------
        names : list, None
            Names to format. If `None`, returns `None`.
        max_items : int, default 50
            Maximum number of names to display before truncation.
        max_text_length : int, default 58
            Maximum length of the joined string before text wrapping.

        Returns
        -------
        formatted : str, None
            Formatted string, or `None` if `names` is `None`.

        """

        names = ForecasterFoundation._truncate_names(names, max_items)
        if names is None:
            return None

        formatted = ", ".join(names)
        if len(formatted) > max_text_length:
            formatted = "\n    " + textwrap.fill(
                formatted, width=80, subsequent_indent="    "
            )

        return formatted

    def __repr__(self) -> str:
        """
        Information displayed when a ForecasterFoundation object is printed.
        """

        series_names_in_ = self._format_names_repr(self.series_names_in_)
        exog_names_in_ = self._format_names_repr(self.exog_names_in_)

        if self.is_fitted:
            context_range_repr = {
                k: v.to_list() for k, v in self.context_range_.items()
            }
        else:
            context_range_repr = None

        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Model ID: {self.model_id} \n"
            f"Context length: {self.context_length} \n"
            f"Series names: {series_names_in_} \n"
            f"Exogenous included: {self.exog_in_} \n"
            f"Exogenous names: {exog_names_in_} \n"
            f"Context range: {context_range_repr} \n"
            f"Training index type: {str(self.index_type_).split('.')[-1][:-2] if self.is_fitted else None} \n"
            f"Training index frequency: {self.index_freq_ if self.is_fitted else None} \n"
            f"Creation date: {self.creation_date} \n"
            f"Last fit date: {self.fit_date} \n"
            f"Skforecast version: {self.skforecast_version} \n"
            f"Python version: {self.python_version} \n"
            f"Forecaster id: {self.forecaster_id} \n"
        )

        return info

    def _repr_html_(self) -> str:
        """
        HTML representation of the object.
        The "General Information" section is expanded by default.
        """

        style, unique_id = get_style_repr_html(self.is_fitted)

        exog_names = self._truncate_names(self.exog_names_in_)
        if exog_names is not None:
            parts = []
            for n in exog_names:
                if n == "...":
                    parts.append('<em style="color: #999;">\u2026</em>')
                else:
                    parts.append(html.escape(str(n)))
            exog_names_html = ", ".join(parts)
        else:
            exog_names_html = str(None)

        series_names = self._truncate_names(self.series_names_in_)
        if series_names is not None:
            series_value_html = ", ".join(html.escape(str(n)) for n in series_names)
        else:
            series_value_html = str(self.series_names_in_)

        if self.is_fitted:
            context_range_parts = [
                f"'{k}': {v.astype(str).to_list()}"
                for k, v in self.context_range_.items()
            ]
            if len(context_range_parts) > 10:
                context_range_parts = (
                    context_range_parts[:5] + ["..."] + context_range_parts[-5:]
                )
            context_range_html = ", ".join(context_range_parts)
        else:
            context_range_html = "Not fitted"

        params_html = "".join(
            f"<li><strong>{html.escape(str(k))}:</strong> {html.escape(str(v))}</li>"
            for k, v in self.estimator.adapter.get_params().items()
            if k != "model_id"
        )

        content = f"""
        <div class="container-{unique_id}">
            <p style="font-size: 1.5em; font-weight: bold; margin-block-start: 0.83em; margin-block-end: 0.83em;">{type(self).__name__}</p>
            <details open>
                <summary>General Information</summary>
                <ul>
                    <li><strong>Model ID:</strong> {self.model_id}</li>
                    <li><strong>Context length:</strong> {self.context_length}</li>
                    <li><strong>Window size:</strong> {self.window_size}</li>
                    <li><strong>Series names:</strong> {series_value_html}</li>
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
                <p style="margin: 0.2em 0 0.2em 1.5em;">{exog_names_html}</p>
            </details>
            <details>
                <summary>Training Information</summary>
                <ul>
                    <li><strong>Context range:</strong> {context_range_html}</li>
                    <li><strong>Training index type:</strong> {str(self.index_type_).split('.')[-1][:-2] if self.is_fitted else 'Not fitted'}</li>
                    <li><strong>Training index frequency:</strong> {html.escape(str(self.index_freq_)) if self.is_fitted else 'Not fitted'}</li>
                </ul>
            </details>
            <details>
                <summary>Model Parameters</summary>
                <ul>
                    {params_html}
                </ul>
            </details>
            <p>
                <a href="https://skforecast.org/{__version__}/api/forecasterfoundation.html">&#128214; <strong>API Reference</strong></a>
                &nbsp;&nbsp;
                <a href="https://skforecast.org/{__version__}/user_guides/foundation-forecasting-models.html">&#128221; <strong>User Guide</strong></a>
            </p>
        </div>
        """

        return style + content

    def fit(
        self,
        series: pd.Series | pd.DataFrame | dict[str, pd.Series],
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.Series | pd.DataFrame | None]
            | None
        ) = None,
    ) -> None:
        """
        Training Forecaster.

        Stores index metadata and delegates context storage to the underlying
        adapter. No model training occurs since foundation model is zero-shot.

        Parameters
        ----------
        series : pandas Series, pandas DataFrame, dict
            Training time series.

            - If `pandas.Series`: single-series mode.
            - If wide `pandas.DataFrame` (one column per series): multi-series
              mode.
            - If long-format `pandas.DataFrame` with a MultiIndex (first level =
              series IDs, second level = `DatetimeIndex`): multi-series
              mode. Internally converted to a dict. An `InputTypeWarning` is
              issued; consider passing a dict directly for better performance.
            - If `dict[str, pd.Series]`: multi-series mode.
        exog : pandas Series, pandas DataFrame, dict, default None
            Historical exogenous variables aligned to `series`. At prediction
            time they are forwarded to the underlying adapter as past
            (historical) covariates, using the adapter-specific covariate
            format.

            In single-series mode: `pd.Series` or `pd.DataFrame` aligned to
            `series`.

            In multi-series mode: a `dict[str, pd.Series | pd.DataFrame | None]`
            with one entry per series, a single `pd.Series` / `pd.DataFrame`
            broadcast to all series, or a long-format `pd.DataFrame` with a
            MultiIndex (first level = series IDs, second level =
            `DatetimeIndex`). Long-format inputs are converted to a `dict`
            internally; an `InputTypeWarning` is issued.

        Returns
        -------
        None

        """

        self.is_fitted = False

        if exog is not None and not self.estimator.allow_exog:
            warnings.warn(
                f"The model '{self.estimator.model_id}' does not support "
                f"exogenous variables. `exog` will be ignored.",
                IgnoredArgumentWarning,
                stacklevel=2,
            )
            exog = None

        self.estimator.fit(series=series, exog=exog)
        self.is_fitted = True

    def predict(
        self,
        steps: int,
        levels: str | list[str] | None = None,
        context: pd.Series | pd.DataFrame | dict[str, pd.Series] | None = None,
        context_exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None,
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.Series | pd.DataFrame | None]
            | None
        ) = None,
        check_inputs: bool = True,
    ) -> pd.DataFrame:
        """
        Predict n steps ahead.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        levels : str, list, default None
            Subset of series to predict. If `None`, all series in `context` are 
            predicted. 
        context : pandas Series, pandas DataFrame, dict, default None
            Context override for backtesting. When provided, replaces the
            context stored at fit time. In single-series mode pass a
            `pd.Series`; in multi-series mode pass a wide `pd.DataFrame` or a
            `dict[str, pd.Series]`. If longer than `context_length`, only the
            last `context_length` observations are used. If shorter, all
            available observations are passed as-is and the model handles the
            reduced context gracefully.
        context_exog : pandas Series, pandas DataFrame, dict, default None
            Historical exogenous variables aligned to `context` (past
            covariates, mapped to the adapter-specific covariate format).
        exog : pandas Series, pandas DataFrame, dict, default None
            Future-known exogenous variables for the forecast horizon (future
            covariates, mapped to the adapter-specific covariate format).
            Must cover exactly `steps` steps for each series.
        check_inputs : bool, default True
            If `True`, the `context` and `context_exog` inputs are validated
            and normalized. If `False`, `context` must already be a
            `dict[str, pandas Series]` and `context_exog` must be a
            `dict[str, pandas DataFrame | None]` or `None`. This argument
            is created for internal use and is not recommended to be changed.

        Returns
        -------
        predictions : pandas DataFrame
            Long-format DataFrame with columns `['level', 'pred']`.
            The index repeats each forecast timestamp once per series.

        Notes
        -----
        Foundation models are pre-trained and do not learn from the data passed
        to `fit`. The `fit` method only stores context (the last `context_length`
        observations) and metadata. This leads to four distinct behaviors
        depending on the combination of `is_fitted` and `context`:

        - **Not fitted, `context=None`**: raises `NotFittedError`. There is no
        context available for prediction.
        - **Fitted, `context=None`**: uses the context and `context_exog_` stored
        during `fit`. If the user supplies `context_exog`, it is ignored with a
        warning.
        - **Not fitted, `context` provided (zero-shot mode)**: The model uses
        `context` and `context_exog` (if provided) as context for prediction.
        - **Fitted, `context` provided**: Stored context is ignored, the
        provided `context` and `context_exog` (if provided) are used for
        prediction.

        """

        if not self.is_fitted and context is None:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `predict()`, or pass `context`."
            )

        predictions = self.estimator.predict(
                          steps        = steps,
                          context      = context,
                          context_exog = context_exog,
                          exog         = exog,
                          quantiles    = None,
                          levels       = levels,
                          check_inputs = check_inputs,
                      )

        return predictions

    def predict_interval(
        self,
        steps: int,
        levels: str | list[str] | None = None,
        context: pd.Series | pd.DataFrame | dict[str, pd.Series] | None = None,
        context_exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None,
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.Series | pd.DataFrame | None]
            | None
        ) = None,
        interval: float | list[float] | tuple[float] = [0.1, 0.9],
        check_inputs: bool = True,
    ) -> pd.DataFrame:
        """
        Predict n steps ahead with prediction intervals.

        Prediction intervals are derived directly from the underlying
        foundation model's native quantile output — no bootstrapping or
        residual estimation is used.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        levels : str, list, default None
            Subset of series to predict. If `None`, all series in `context` are 
            predicted. 
        context : pandas Series, pandas DataFrame, dict, default None
            Context override for backtesting.
        context_exog : pandas Series, pandas DataFrame, dict, default None
            Historical exog aligned to `context` (past covariates).
        exog : pandas Series, pandas DataFrame, dict, default None
            Future-known exogenous variables for the forecast horizon
            (future covariates).
        interval : float, list, tuple, default [0.1, 0.9]
            Confidence level of the prediction interval. Interpretation depends 
            on the method used:
            
            - If `float`, represents the nominal (expected) coverage (between 0 
            and 1). For instance, `interval=0.95` corresponds to `[0.025, 0.975]` 
            quantiles.
            - If `list` or `tuple`, defines the exact quantiles to compute, which 
            must be between 0 and 1 inclusive. For example, interval 
            of 95% should be as `interval = [0.025, 0.975]`.

            **Changed in version 0.23.0:** `interval` is now expressed as
            quantiles (0-1) instead of percentiles (0-100). Passing percentiles
            is deprecated and emits a `FutureWarning`.
        check_inputs : bool, default True
            If `True`, the `context` and `context_exog` inputs are validated
            and normalized. If `False`, `context` must already be a
            `dict[str, pandas Series]` and `context_exog` must be a
            `dict[str, pandas DataFrame | None]` or `None`. This argument
            is created for internal use and is not recommended to be changed.

        Returns
        -------
        predictions : pandas DataFrame
            Long-format DataFrame with columns `['level', 'pred', 'lower_bound',
            'upper_bound']`.

        Notes
        -----
        Foundation models are pre-trained and do not learn from the data passed
        to `fit`. The `fit` method only stores context (the last `context_length`
        observations) and metadata. This leads to four distinct behaviors
        depending on the combination of `is_fitted` and `context`:

        - **Not fitted, `context=None`**: raises `NotFittedError`. There is no
        context available for prediction.
        - **Fitted, `context=None`**: uses the context and `context_exog_` stored
        during `fit`. If the user supplies `context_exog`, it is ignored with a
        warning.
        - **Not fitted, `context` provided (zero-shot mode)**: The model uses
        `context` and `context_exog` (if provided) as context for prediction.
        - **Fitted, `context` provided**: Stored context is ignored, the
        provided `context` and `context_exog` (if provided) are used for
        prediction.

        """

        if not self.is_fitted and context is None:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `predict_interval()`, or pass `context`."
            )

        if isinstance(interval, (list, tuple)):
            interval = _normalize_interval_scale(interval)
            check_interval(interval=interval, ensure_symmetric_intervals=False)
        else:
            check_interval(alpha=interval, alpha_literal='interval')
            interval = [0.5 - interval / 2, 0.5 + interval / 2]

        lower_q, upper_q = float(interval[0]), float(interval[1])

        # Always include the median (0.5) so 'pred' is the central forecast.
        quantiles = sorted({lower_q, 0.5, upper_q})

        predictions = self.predict_quantiles(
                          steps        = steps,
                          levels       = levels,
                          context      = context,
                          context_exog = context_exog,
                          exog         = exog,
                          quantiles    = quantiles,
                          check_inputs = check_inputs,
                      )

        predictions = predictions[['level', f'q_{0.5}', f'q_{lower_q}', f'q_{upper_q}']]
        predictions.columns = ['level', 'pred', 'lower_bound', 'upper_bound']

        return predictions

    def predict_quantiles(
        self,
        steps: int,
        levels: str | list[str] | None = None,
        context: pd.Series | pd.DataFrame | dict[str, pd.Series] | None = None,
        context_exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None,
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.Series | pd.DataFrame | None]
            | None
        ) = None,
        quantiles: list[float] | tuple[float] = [0.1, 0.5, 0.9],
        check_inputs: bool = True,
    ) -> pd.DataFrame:
        """
        Predict n steps ahead at specified quantile levels.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        levels : str, list, default None
            Subset of series to predict. If `None`, all series in `context` are 
            predicted. 
        context : pandas Series, pandas DataFrame, dict, default None
            Context override for backtesting.
        context_exog : pandas Series, pandas DataFrame, dict, default None
            Historical exog aligned to `context` (past covariates).
        exog : pandas Series, pandas DataFrame, dict, default None
            Future-known exogenous variables for the forecast horizon
            (future covariates).
        quantiles : list, tuple, default [0.1, 0.5, 0.9]
            Quantile levels to forecast. Values must be in the range (0, 1).
        check_inputs : bool, default True
            If `True`, the `context` and `context_exog` inputs are validated
            and normalized. If `False`, `context` must already be a
            `dict[str, pandas Series]` and `context_exog` must be a
            `dict[str, pandas DataFrame | None]` or `None`. This argument
            is created for internal use and is not recommended to be changed.

        Returns
        -------
        predictions : pandas DataFrame
            Long-format DataFrame with columns `['level', 'q_0.1', 'q_0.5', ...]`.

        Notes
        -----
        Foundation models are pre-trained and do not learn from the data passed
        to `fit`. The `fit` method only stores context (the last `context_length`
        observations) and metadata. This leads to four distinct behaviors
        depending on the combination of `is_fitted` and `context`:

        - **Not fitted, `context=None`**: raises `NotFittedError`. There is no
        context available for prediction.
        - **Fitted, `context=None`**: uses the context and `context_exog_` stored
        during `fit`. If the user supplies `context_exog`, it is ignored with a
        warning.
        - **Not fitted, `context` provided (zero-shot mode)**: The model uses
        `context` and `context_exog` (if provided) as context for prediction.
        - **Fitted, `context` provided**: Stored context is ignored, the
        provided `context` and `context_exog` (if provided) are used for
        prediction.

        """

        if not self.is_fitted and context is None:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `predict_quantiles()`, or pass `context`."
            )

        predictions = self.estimator.predict(
                          steps        = steps,
                          context      = context,
                          context_exog = context_exog,
                          exog         = exog,
                          quantiles    = list(quantiles),
                          levels       = levels,
                          check_inputs = check_inputs,
                      )

        return predictions

    def set_params(self, params: dict[str, object]) -> None:
        """
        Set new values to the parameters of the underlying estimator.

        After calling this method, the forecaster is reset to an unfitted state.

        Parameters
        ----------
        params : dict
            Parameters values.

        Returns
        -------
        None

        """

        self.estimator.set_params(**params)
        self.is_fitted = False

    def summary(self) -> None:
        """
        Show forecaster information.

        Returns
        -------
        None

        """

        print(self.__repr__())
