################################################################################
#                             ForecasterFoundation                             #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import sys
import textwrap
import warnings
from copy import copy
import pandas as pd
from sklearn.exceptions import NotFittedError

from .. import __version__
from ..exceptions import IgnoredArgumentWarning
from ..utils import (
    check_interval,
    get_style_repr_html,
)
from ._foundation_model import FoundationModel


class ForecasterFoundation:
    """
    Forecaster that wraps a `FoundationModel` for full skforecast ecosystem
    compatibility: backtesting, model selection, etc.

    Unlike ML-based forecasters, there is no training step — the underlying
    foundation models are zero-shot. `fit` only stores the history
    as context and records index metadata. Predictions are generated directly
    by the model's `predict_quantiles` pipeline.

    Supports both single-series and multi-series modes. Pass a `pandas.Series`
    to `fit` for single-series forecasting or a wide `pandas.DataFrame`, a
    long-format `pandas.DataFrame` (MultiIndex), or a `dict[str, pd.Series]`
    for multi-series (global-model) forecasting.

    Parameters
    ----------
    estimator : FoundationModel
        A configured `FoundationModel` instance, e.g.
        `FoundationModel("autogluon/chronos-2-small", context_length=512)`.
    forecaster_id : str, int, default None
        Name used as an identifier of the forecaster.

    Attributes
    ----------
    estimator : FoundationModel
        The `FoundationModel` instance provided by the user.
    context_length : int
        Maximum number of historical observations used as context. Mirrors
        `estimator.context_length`. Updated when `set_params` is called.
    model_id : str
        HuggingFace model ID. Mirrors `estimator.model_id`. Updated when
        `set_params` is called.
    window_size : int
        Number of historical observations provided to the model as context in
        each backtesting fold. Always equals `context_length`. Unlike ML
        forecasters where `window_size` is the strict minimum required to build
        features, here it represents the *desired* context size: backtesting
        passes up to `context_length` observations per fold so the model
        receives as much history as possible. When fewer observations are
        available (e.g. early folds), all available data is passed and the
        model handles shorter input gracefully.
    last_window_ : None
        Intentionally `None` — `ForecasterFoundation` never stores training
        data directly; the adapter's internal `_history` is used instead.
    index_type_ : type
        Type of index of the input used in training.
    index_freq_ : pandas.DateOffset or int
        Frequency of the index of the input used in training. A
        `pandas.DateOffset` for `DatetimeIndex`; the `step` integer
        for `RangeIndex`.
    training_range_ : pandas Index or dict
        First and last values of the index of the data used during training.
        `pandas.Index` in single-series mode; `dict[str, pandas.Index]` in
        multi-series mode.
    series_names_in_ : list of str
        Names of all series seen during training. In single-series mode this
        is a one-element list.
    exog_in_ : bool
        `True` if the forecaster has been trained with exogenous variables.
    exog_names_in_ : list
        Names of the exogenous variables used during training.
    exog_type_in_ : type
        Type of exogenous variable/s used in training.
    fit_date : str
        Date of last fit.
    is_fitted : bool
        Tag to identify if the forecaster has been fitted (trained).
    creation_date : str
        Date of creation.
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
        estimator: FoundationModel,
        forecaster_id: str | int | None = None,
    ) -> None:

        if not isinstance(estimator, FoundationModel):
            raise TypeError(
                f"`estimator` must be a `FoundationModel` instance. "
                f"Got {type(estimator)}."
            )

        self.estimator                 = estimator
        self.forecaster_id             = forecaster_id
        self.last_window_              = None
        self.index_type_               = None
        self.index_freq_               = None
        self.training_range_           = None
        self.series_names_in_          = None
        self.is_multiple_series_       = False
        self.exog_in_                  = False
        self.exog_names_in_            = None
        self.exog_names_in_per_series_ = None
        self.exog_type_in_             = None
        self.is_fitted                 = False
        self.fit_date                  = None
        self.creation_date             = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.skforecast_version        = __version__
        self.python_version            = sys.version.split(" ")[0]
        self.context_length            = estimator.context_length
        self.model_id                  = estimator.model_id
        self.window_size               = estimator.context_length

        self.__skforecast_tags__ = {
            "library": "skforecast",
            "forecaster_name": "ForecasterFoundation",
            "forecaster_task": "regression",
            "forecasting_scope": "single-series|multi-series",
            "forecasting_strategy": "foundation",
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
            "handles_missing_values_series": False,
            "handles_missing_values_exog": False,

            "supports_lags": False,
            "supports_window_features": False,
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

    def __repr__(self) -> str:
        """
        Information displayed when a ForecasterFoundation object is printed.
        """

        exog_names_in_ = None
        if self.exog_names_in_ is not None:
            names = copy(self.exog_names_in_)
            if len(names) > 50:
                names = names[:50] + ["..."]
            exog_names_in_ = ", ".join(names)
            if len(exog_names_in_) > 58:
                exog_names_in_ = "\n    " + textwrap.fill(
                    exog_names_in_, width=80, subsequent_indent="    "
                )

        if self.is_fitted and self.is_multiple_series_:
            training_range_repr = {
                k: v.to_list() for k, v in self.training_range_.items()
            }
        elif self.is_fitted:
            training_range_repr = self.training_range_.to_list()
        else:
            training_range_repr = None

        series_repr = self.series_names_in_
        series_label = "Series names"

        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Model: {self.model_id} \n"
            f"Context length: {self.context_length} \n"
            f"{series_label}: {series_repr} \n"
            f"Exogenous included: {self.exog_in_} \n"
            f"Exogenous names: {exog_names_in_} \n"
            f"Training range: {training_range_repr} \n"
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

        exog_names_html = None
        if self.exog_names_in_ is not None:
            names = copy(self.exog_names_in_)
            if len(names) > 50:
                names = names[:50] + ["..."]
            exog_names_html = "".join(f"<li>{n}</li>" for n in names)

        if self.is_fitted and self.is_multiple_series_:
            training_range_html = "".join(
                f"<li><strong>{k}:</strong> {v.to_list()}</li>"
                for k, v in self.training_range_.items()
            )
            training_range_html = f"<ul>{training_range_html}</ul>"
        elif self.is_fitted:
            training_range_html = str(self.training_range_.to_list())
        else:
            training_range_html = "Not fitted"

        series_label_html = "Series names"
        series_value_html = str(self.series_names_in_)

        content = f"""
        <div class="container-{unique_id}">
            <p style="font-size: 1.5em; font-weight: bold; margin-block-start: 0.83em; margin-block-end: 0.83em;">{type(self).__name__}</p>
            <details open>
                <summary>General Information</summary>
                <ul>
                    <li><strong>Model:</strong> {self.model_id}</li>
                    <li><strong>Context length:</strong> {self.context_length}</li>
                    <li><strong>Window size:</strong> {self.window_size}</li>
                    <li><strong>{series_label_html}:</strong> {series_value_html}</li>
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
                    {exog_names_html}
                </ul>
            </details>
            <details>
                <summary>Training Information</summary>
                <ul>
                    <li><strong>Training range:</strong> {training_range_html}</li>
                    <li><strong>Training index type:</strong> {str(self.index_type_).split('.')[-1][:-2] if self.is_fitted else 'Not fitted'}</li>
                    <li><strong>Training index frequency:</strong> {self.index_freq_ if self.is_fitted else 'Not fitted'}</li>
                </ul>
            </details>
            <details>
                <summary>Model Parameters</summary>
                <ul>
                    {''.join(f'<li><strong>{k}:</strong> {v}</li>' for k, v in self.estimator.adapter.get_params().items() if k != 'model_id')}
                </ul>
            </details>
            <p>
                <a href="https://skforecast.org/{__version__}/api/forecasterfoundation.html">&#128712 <strong>API Reference</strong></a>
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
    ) -> ForecasterFoundation:
        """
        Fit the forecaster.

        Stores index metadata and delegates history storage to the underlying
        adapter. No model training occurs since foundation model is zero-shot.

        Parameters
        ----------
        series : pandas Series, pandas DataFrame, or dict of pandas Series
            Training time series.

            - `pandas.Series`: single-series mode.
            - Wide `pandas.DataFrame` (one column per series): multi-series
              mode.
            - Long-format `pandas.DataFrame` with a MultiIndex (first level =
              series IDs, second level = `DatetimeIndex`): multi-series
              mode. Internally converted to a dict. An `InputTypeWarning` is
              issued; consider passing a dict directly for better performance.
            - `dict[str, pd.Series]`: multi-series mode.
        exog : pandas Series, pandas DataFrame, dict, default None
            Historical exogenous variables aligned to `series`. These map to
            `past_covariates` in the Chronos-2 input at prediction time.

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
        self : ForecasterFoundation

        """

        self.last_window_              = None
        self.index_type_               = None
        self.index_freq_               = None
        self.training_range_           = None
        self.series_names_in_          = None
        self.is_multiple_series_       = False
        self.exog_in_                  = False
        self.exog_names_in_            = None
        self.exog_names_in_per_series_ = None
        self.exog_type_in_             = None
        self.is_fitted                 = False
        self.fit_date                  = None

        if exog is not None and not self.estimator.allow_exogenous:
            warnings.warn(
                f"The model '{self.estimator.model_id}' does not support "
                f"exogenous variables. `exog` will be ignored.",
                IgnoredArgumentWarning,
                stacklevel=2,
            )
            exog = None

        self.estimator.fit(series=series, exog=exog)

        self.series_names_in_    = self.estimator.series_names_in_
        self.is_multiple_series_ = self.estimator.is_multiple_series_

        self.exog_in_                  = self.estimator.exog_in_
        self.exog_names_in_            = self.estimator.exog_names_in_
        self.exog_names_in_per_series_ = self.estimator.exog_names_in_per_series_ 
        if exog is not None and self.exog_in_:
            self.exog_type_in_ = type(exog)
        
        self.is_fitted       = True
        self.fit_date        = self.estimator.fit_date
        self.training_range_ = self.estimator.training_range_
        self.index_type_     = self.estimator.index_type_
        self.index_freq_     = self.estimator.index_freq_

    def predict(
        self,
        steps: int,
        levels: str | list[str] | None = None,
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.Series | pd.DataFrame | None]
            | None
        ) = None,
        last_window: pd.Series | pd.DataFrame | dict[str, pd.Series] | None = None,
        last_window_exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Forecast future values.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        levels : str, list of str, default None
            Series to predict. Only used in multi-series mode. If `None`,
            all series seen at fit time are predicted.
        exog : pandas Series, pandas DataFrame, dict, default None
            Future-known exogenous variables for the forecast horizon. Maps to
            `future_covariates` in Chronos-2. Must cover exactly `steps` steps
            for each series.
        last_window : pandas Series, pandas DataFrame, dict, default None
            Context override for backtesting. When provided, replaces the
            history stored at fit time. In single-series mode pass a
            `pd.Series`; in multi-series mode pass a wide `pd.DataFrame` or a
            `dict[str, pd.Series]`. If longer than `context_length`, only the
            last `context_length` observations are used. If shorter, all
            available observations are passed as-is and the model handles the
            reduced context gracefully.
        last_window_exog : pandas Series, pandas DataFrame, dict, default None
            Historical exogenous variables aligned to `last_window`. Maps to
            `past_covariates` in Chronos-2.

        Returns
        -------
        predictions : pandas DataFrame
            Long-format `pd.DataFrame` with columns `['level', 'pred']`.
            The index repeats each forecast timestamp once per series.

        """

        if not self.is_fitted:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `predict()`."
            )

        predictions = self.estimator.predict(
            steps=steps,
            exog=exog,
            quantiles=None,
            last_window=last_window,
            last_window_exog=last_window_exog,
        )

        if levels is not None:
            if isinstance(levels, str):
                levels = [levels]
            predictions = predictions[predictions["level"].isin(levels)]

        return predictions

    def predict_interval(
        self,
        steps: int,
        interval: list[float] | tuple[float] = [10, 90],
        levels: str | list[str] | None = None,
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.Series | pd.DataFrame | None]
            | None
        ) = None,
        last_window: pd.Series | pd.DataFrame | dict[str, pd.Series] | None = None,
        last_window_exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None,
    ) -> pd.DataFrame:
        """
        Forecast future values with prediction intervals.

        Prediction intervals are derived directly from Chronos-2's native
        quantile output — no bootstrapping or residual estimation is used.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        interval : list, tuple, default [10, 90]
            Confidence of the prediction interval. Sequence of two percentiles
            `[lower, upper]`, e.g. `[10, 90]` for an 80 % interval.
            Values must be between 0 and 100 inclusive.
        levels : str, list of str, default None
            Series to predict. Only used in multi-series mode. If `None`,
            all series seen at fit time are predicted.
        exog : pandas Series, pandas DataFrame, dict, default None
            Future-known exogenous variables (`future_covariates`).
        last_window : pandas Series, pandas DataFrame, dict, default None
            Context override for backtesting.
        last_window_exog : pandas Series, pandas DataFrame, dict, default None
            Historical exog aligned to `last_window`.

        Returns
        -------
        predictions : pandas DataFrame
            Long-format columns `['level', 'pred', 'lower_bound', 'upper_bound']`.

        """

        if not self.is_fitted:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `predict_interval()`."
            )

        if isinstance(interval, (int, float)):
            check_interval(alpha=interval, alpha_literal='interval')
            interval = [(0.5 - interval / 2) * 100, (0.5 + interval / 2) * 100]

        if len(interval) != 2:
            raise ValueError(
                f"`interval` must be a sequence of exactly two values [lower, upper]. "
                f"Got {len(interval)} values."
            )
        lower_pct, upper_pct = float(interval[0]), float(interval[1])
        if not (0 <= lower_pct < upper_pct <= 100):
            raise ValueError(
                f"`interval` values must satisfy 0 <= lower < upper <= 100. "
                f"Got [{lower_pct}, {upper_pct}]."
            )

        lower_q = lower_pct / 100
        upper_q = upper_pct / 100
        # Always include the median (0.5) so 'pred' is the central forecast.
        quantiles = sorted({lower_q, 0.5, upper_q})

        df = self.estimator.predict(
            steps=steps,
            exog=exog,
            quantiles=quantiles,
            last_window=last_window,
            last_window_exog=last_window_exog,
        )
        if levels is not None:
            if isinstance(levels, str):
                levels = [levels]
            df = df[df["level"].isin(levels)]
        result = pd.DataFrame(index=df.index)
        result['level']       = df['level']
        result['pred']        = df[f'q_{0.5}']
        result['lower_bound'] = df[f'q_{lower_q}']
        result['upper_bound'] = df[f'q_{upper_q}']
        return result

    def predict_quantiles(
        self,
        steps: int,
        quantiles: list[float] | tuple[float] = [0.1, 0.5, 0.9],
        levels: str | list[str] | None = None,
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.Series | pd.DataFrame | None]
            | None
        ) = None,
        last_window: pd.Series | pd.DataFrame | dict[str, pd.Series] | None = None,
        last_window_exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None,
    ) -> pd.DataFrame:
        """
        Forecast future values at specified quantile levels.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        quantiles : list of float, default [0.1, 0.5, 0.9]
            Quantile levels to forecast. Values must be in the range (0, 1).
        levels : str, list of str, default None
            Series to predict. Only used in multi-series mode. If `None`,
            all series seen at fit time are predicted.
        exog : pandas Series, pandas DataFrame, dict, default None
            Future-known exogenous variables (`future_covariates`).
        last_window : pandas Series, pandas DataFrame, dict, default None
            Context override for backtesting.
        last_window_exog : pandas Series, pandas DataFrame, dict, default None
            Historical exog aligned to `last_window`.

        Returns
        -------
        predictions : pandas DataFrame
            Long-format columns `['level', 'q_0.1', 'q_0.5', ...]`.

        """

        if not self.is_fitted:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `predict_quantiles()`."
            )

        predictions = self.estimator.predict(
            steps=steps,
            exog=exog,
            quantiles=list(quantiles),
            last_window=last_window,
            last_window_exog=last_window_exog,
        )
        if levels is not None:
            if isinstance(levels, str):
                levels = [levels]
            predictions = predictions[predictions["level"].isin(levels)]
        return predictions

    def set_params(self, params: dict) -> None:
        """
        Set new values to the parameters of the underlying estimator.

        After calling this method, the forecaster is reset to an unfitted state.
        The `fit` method must be called before prediction.

        Parameters
        ----------
        params : dict
            Parameter names and their new values. Valid keys depend on the
            underlying adapter. See the adapter's `set_params` for the
            full list of accepted parameters.

        Returns
        -------
        None

        """

        self.estimator.set_params(**params)

        self.context_length = self.estimator.context_length
        self.model_id       = self.estimator.model_id
        self.window_size    = self.estimator.context_length

        self.is_fitted        = False
        self.fit_date         = None
        self.training_range_  = None
        self.index_type_      = None
        self.index_freq_      = None
        self.last_window_     = None
        self.series_names_in_ = None
        self.is_multiple_series_  = False
        self.exog_in_         = False
        self.exog_names_in_            = None
        self.exog_names_in_per_series_  = None
        self.exog_type_in_             = None

    def summary(self) -> None:
        """
        Show forecaster information.

        Returns
        -------
        None

        """

        print(self)
