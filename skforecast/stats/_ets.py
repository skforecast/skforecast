################################################################################
#                                      ETS                                     #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd
from dataclasses import asdict
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .exponential_smoothing._ets_base import (
    ets,
    auto_ets,
    forecast_ets
)
from ._utils import check_memory_reduced


class Ets(BaseEstimator, RegressorMixin):
    """
    Scikit-learn style wrapper for the ETS (Error, Trend, Seasonality) model.

    This estimator treats a univariate time series as input. Call `fit(y)`
    with a 1D array-like of observations in time order, then produce
    out-of-sample forecasts via `predict(steps)` and prediction intervals
    via `predict_interval(steps, level=...)`. In-sample diagnostics are
    available through `fitted_`, `residuals_()` and `summary()`.

    Parameters
    ----------
    m : int, default=1
        Seasonal period (e.g., 12 for monthly data with yearly seasonality).
    model : str, None, default "ZZZ"
        Three-letter model specification (e.g., "ANN", "AAA", "MAM"):
        - First letter: Error type (A=Additive, M=Multiplicative, Z=Auto)
        - Second letter: Trend type (N=None, A=Additive, M=Multiplicative, Z=Auto)
        - Third letter: Season type (N=None, A=Additive, M=Multiplicative, Z=Auto)
        Use "ZZZ" or None for automatic model selection.
    damped : bool or None, default None
        Whether to use damped trend. If None, both damped and non-damped
        models are tried (only when model="ZZZ").
    alpha : float, optional
        Smoothing parameter for level (0 < alpha < 1). If None, estimated.
    beta : float, optional
        Smoothing parameter for trend (0 < beta < alpha). If None, estimated.
    gamma : float, optional
        Smoothing parameter for seasonality (0 < gamma < 1-alpha). If None, estimated.
    phi : float, optional
        Damping parameter (0 < phi < 1). If None, estimated.
    lambda_param : float, optional
        Box-Cox transformation parameter. If None, no transformation applied.
    lambda_auto : bool, default=False
        If True, automatically select optimal Box-Cox lambda parameter.
    bias_adjust : bool, default=True
        Apply bias adjustment when back-transforming forecasts.
    bounds : str, default="both"
        Parameter bounds type: "usual", "admissible", or "both".
    seasonal : bool, default=True
        Allow seasonal models (only used with model="ZZZ").
    trend : bool, optional
        Allow trend models. If None, automatically determined (only with model="ZZZ").
    ic : {"aic", "aicc", "bic"}, default="aicc"
        Information criterion for model selection (only with model="ZZZ").
    allow_multiplicative : bool, default=True
        Allow multiplicative error and season models (only with model="ZZZ").
    allow_multiplicative_trend : bool, default=False
        Allow multiplicative trend models (only with model="ZZZ").

    Attributes
    ----------
    m : int
        Seasonal period (e.g., 12 for monthly data with yearly seasonality).
    model : str
        Three-letter model specification (e.g., "ANN", "AAA", "MAM"). Each letter
        represents error, trend, and season types respectively, using A (Additive),
        M (Multiplicative), N (None), or Z (Auto-select).
    damped : bool or None
        Whether to apply damping to the trend component. If None with model="ZZZ",
        both damped and non-damped models are evaluated during automatic selection.
    alpha : float or None
        User-provided smoothing parameter for the level component (0 < alpha < 1).
        When None, the parameter is estimated during fitting.
    beta : float or None
        User-provided smoothing parameter for the trend component (0 < beta < alpha).
        When None, the parameter is estimated during fitting if trend is present.
    gamma : float or None
        User-provided smoothing parameter for the seasonal component (0 < gamma < 1-alpha).
        When None, the parameter is estimated during fitting if seasonality is present.
    phi : float or None
        User-provided damping parameter (0 < phi < 1). When None, the parameter is
        estimated during fitting if damped trend is used.
    lambda_param : float or None
        Box-Cox transformation parameter applied to the time series before fitting.
        When None, no transformation is applied unless lambda_auto is True.
    lambda_auto : bool
        Whether to automatically determine the optimal Box-Cox transformation parameter
        during model fitting.
    bias_adjust : bool
        Whether to apply bias adjustment when back-transforming forecasts from the
        Box-Cox transformed scale to the original scale.
    bounds : str
        Type of parameter bounds used during optimization: "usual" for standard bounds,
        "admissible" for stability-ensuring bounds, or "both" for their intersection.
    seasonal : bool
        Whether seasonal models are considered during automatic model selection
        (only applicable when model="ZZZ").
    trend : bool or None
        Whether trend models are considered during automatic model selection. When None
        with model="ZZZ", this is determined automatically based on the data.
    ic : {"aic", "aicc", "bic"}
        Information criterion used to compare and select the best model during automatic
        model selection (only applicable when model="ZZZ").
    allow_multiplicative : bool
        Whether multiplicative error and seasonal components are allowed during automatic
        model selection (only applicable when model="ZZZ").
    allow_multiplicative_trend : bool
        Whether multiplicative trend components are allowed during automatic model
        selection (only applicable when model="ZZZ").
    model_ : ETSModel or None
        The fitted ETS model object containing parameters, diagnostics, and state space
        representation. Available after calling `fit()`.
    model_config_ : dict or None
        Dictionary containing the model configuration including error type, trend type,
        seasonal type, damping flag, and seasonal period. Available after calling `fit()`.
    params_ : dict or None
        Dictionary of estimated model parameters including smoothing parameters (alpha, 
        beta, gamma, phi) and initial state values. Available after calling `fit()`.
    aic_ : float or None
        Akaike Information Criterion of the fitted model, measuring the quality of fit
        while penalizing model complexity. Available after calling `fit()`.
    bic_ : float or None
        Bayesian Information Criterion of the fitted model, similar to AIC but with a
        stronger penalty for model complexity. Available after calling `fit()`.
    y_train_ : ndarray of shape (n_samples,) or None
        The original training time series used to fit the model.
    fitted_values_ : ndarray of shape (n_samples,) or None
        One-step-ahead in-sample fitted values from the model.
    in_sample_residuals_ : ndarray of shape (n_samples,) or None
        In-sample residuals calculated as the difference between observed values and
        fitted values.
    n_features_in_ : int or None
        Number of features (time series) seen during `fit()`. For ETS, this is always 1
        as it handles univariate time series. Available after calling `fit()`.
    is_memory_reduced : bool
        Flag indicating whether `reduce_memory()` has been called to clear diagnostic
        arrays (y_train_, fitted_values_, in_sample_residuals_).
    is_fitted : bool
        Flag indicating whether the model has been successfully fitted to data.
    estimator_id : str
        String identifier for the model configuration (e.g., "Ets(ANN)").
    
    """

    def __init__(
        self,
        m: int = 1,
        model: str | None = "ZZZ",
        damped: bool | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        phi: float | None = None,
        lambda_param: float | None = None,
        lambda_auto: bool = False,
        bias_adjust: bool = True,
        bounds: str = "both",
        seasonal: bool = True,
        trend: bool | None = None,
        ic: Literal["aic", "aicc", "bic"] = "aicc",
        allow_multiplicative: bool = True,
        allow_multiplicative_trend: bool = False,
    ):

        self.m                          = m
        self.model                      = model if model is not None else "ZZZ"
        self.damped                     = damped
        self.alpha                      = alpha
        self.beta                       = beta
        self.gamma                      = gamma
        self.phi                        = phi
        self.lambda_param               = lambda_param
        self.lambda_auto                = lambda_auto
        self.bias_adjust                = bias_adjust
        self.bounds                     = bounds
        self.seasonal                   = seasonal
        self.trend                      = trend
        self.ic                         = ic
        self.allow_multiplicative       = allow_multiplicative
        self.allow_multiplicative_trend = allow_multiplicative_trend
        
        self.model_                     = None
        self.model_config_              = None
        self.params_                    = None
        self.aic_                       = None
        self.bic_                       = None
        self.y_train_                   = None
        self.fitted_values_             = None
        self.in_sample_residuals_       = None
        self.n_features_in_             = None
        self.is_memory_reduced          = False
        self.is_fitted                  = False
        self.estimator_id               = f"Ets({self.model})"

    def __repr__(self) -> str:
        """
        Information displayed when an Ets object is printed.
        """
        return self.estimator_id

    def fit(self, y: pd.Series | np.ndarray, exog: None = None) -> Ets:
        """
        Fit the ETS model to a univariate time series.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Time-ordered numeric sequence.
        exog : Ignored
            Exogenous variables. Ignored, present for API compatibility.

        Returns
        -------
        self : Ets
            Fitted estimator.
            
        """

        self.model_               = None
        self.model_config_        = None
        self.params_              = None
        self.aic_                 = None
        self.bic_                 = None
        self.y_train_             = None
        self.fitted_values_       = None
        self.in_sample_residuals_ = None
        self.n_features_in_       = None
        self.is_memory_reduced    = False
        self.is_fitted            = False
        
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("`y` must be a pandas Series or numpy ndarray.")
        
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 2 and y.shape[1] == 1:
            # Allow (n, 1) shaped arrays and squeeze to 1D
            y = y.ravel()
        elif y.ndim != 1:
            raise ValueError("`y` must be a 1D array-like sequence.")
        if len(y) < 1:
            raise ValueError("`y` is too short to fit ETS model.")

        # Automatic model selection
        if self.model == "ZZZ":
            self.model_ = auto_ets(
                y,
                m                          = self.m,
                seasonal                   = self.seasonal,
                trend                      = self.trend,
                damped                     = self.damped,
                ic                         = self.ic,
                allow_multiplicative       = self.allow_multiplicative,
                allow_multiplicative_trend = self.allow_multiplicative_trend,
                lambda_auto                = self.lambda_auto,
                verbose                    = False,
            )
        else:
            # Fit specific model
            damped_param = False if self.damped is None else self.damped
            self.model_ = ets(
                y,
                m            = self.m,
                model        = self.model,
                damped       = damped_param,
                alpha        = self.alpha,
                beta         = self.beta,
                gamma        = self.gamma,
                phi          = self.phi,
                lambda_param = self.lambda_param,
                lambda_auto  = self.lambda_auto,
                bias_adjust  = self.bias_adjust,
                bounds       = self.bounds,
            )

        # Extract model attributes (use references to avoid duplicating arrays)
        self.model_config_        = asdict(self.model_.config)
        self.params_              = asdict(self.model_.params)
        self.aic_                 = self.model_.aic
        self.bic_                 = self.model_.bic
        self.y_train_             = self.model_.y_original
        self.fitted_values_       = self.model_.fitted
        self.in_sample_residuals_ = self.model_.residuals
        self.n_features_in_       = 1
        self.is_fitted            = True

        model_name = f"{self.model_config_['error']}{self.model_config_['trend']}{self.model_config_['season']}"
        if self.model_config_['damped'] and self.model_config_['trend'] != "N":
            model_name = f"{self.model_config_['error']}{self.model_config_['trend']}d{self.model_config_['season']}"

        self.estimator_id = f"Ets({model_name})"

        return self

    def predict(self, steps: int, exog: None = None) -> np.ndarray:
        """
        Generate mean forecasts steps ahead.

        Parameters
        ----------
        steps : int
            Forecast horizon (must be > 0).
        exog : None
            Exogenous variables. Ignored, present for API compatibility.

        Returns
        -------
        mean : ndarray of shape (steps,)
            Point forecasts for steps 1..h.
        """
        check_is_fitted(self, "model_")
        if not isinstance(steps, (int, np.integer)) or steps <= 0:
            raise ValueError("`steps` must be a positive integer.")

        result = forecast_ets(
            self.model_,
            h           = steps,
            bias_adjust = self.bias_adjust,
            level       = None
        )
        return result["mean"]

    def predict_interval(
        self,
        steps: int = 1,
        level: list[float] | tuple[float, ...] = (80, 95),
        as_frame: bool = True,
        exog: None = None,
    ) -> pd.DataFrame | dict:
        """
        Forecast with prediction intervals.

        Parameters
        ----------
        steps : int, default=1
            Forecast horizon.
        level : list or tuple of float, default=(80, 95)
            Confidence levels in percent.
        as_frame : bool, default=True
            If True, return a tidy DataFrame with columns:
            'mean', 'lower_<L>', 'upper_<L>' for each level L.
            If False, return raw dict.
        exog : None
            Exogenous variables. Ignored, present for API compatibility.

        Returns
        -------
        DataFrame or dict
            If as_frame=True: DataFrame indexed by step (1..steps).
            Else: dict with keys 'mean', 'lower_XX', 'upper_XX'.
        """
        check_is_fitted(self, "model_")
        if not isinstance(steps, (int, np.integer)) or steps <= 0:
            raise ValueError("`steps` must be a positive integer.")

        result = forecast_ets(
            self.model_,
            h           = steps,
            bias_adjust = self.bias_adjust,
            level       = list(level)
        )

        if not as_frame:
            return result

        # Convert to DataFrame
        idx = pd.RangeIndex(1, steps + 1, name="step")
        df = pd.DataFrame({"mean": result["mean"]}, index=idx)

        for lv in level:
            lv_int = int(lv)
            if f"lower_{lv_int}" in result:
                df[f"lower_{lv_int}"] = result[f"lower_{lv_int}"]
                df[f"upper_{lv_int}"] = result[f"upper_{lv_int}"]

        return df

    def get_residuals(self) -> np.ndarray:
        """
        Get in-sample residuals (observed - fitted) from the ETS model.

        Returns
        -------
        residuals : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "model_")
        check_memory_reduced(self, 'residuals_')
        return self.in_sample_residuals_

    def get_fitted_values(self) -> np.ndarray:
        """
        Get in-sample fitted values from the ETS model.

        Returns
        -------
        fitted : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "model_")
        check_memory_reduced(self, 'fitted_')
        return self.fitted_values_

    def summary(self) -> None:
        """
        Print a summary of the fitted ETS model.
        """
        check_is_fitted(self, "model_")
        check_memory_reduced(self, 'summary')

        print("Ets Model Summary")
        print("=" * 60)
        print(f"Model: {self.estimator_id}")
        print(f"Number of observations: {len(self.y_train_)}")
        print(f"Seasonal period (m): {self.model_config_['m']}")
        print()

        print("Smoothing parameters:")
        print(f"  alpha (level):       {self.params_['alpha']:.4f}")
        if self.model_config_['trend'] != "N":
            print(f"  beta (trend):        {self.params_['beta']:.4f}")
        if self.model_config_['season'] != "N":
            print(f"  gamma (seasonal):    {self.params_['gamma']:.4f}")
        if self.model_config_['damped']:
            print(f"  phi (damping):       {self.params_['phi']:.4f}")
        print()

        print("Initial states:")
        print(f"  Level (l0):          {self.params_['init_states'][0]:.4f}")
        if self.model_config_['trend'] != "N" and len(self.params_['init_states']) > 1:
            print(f"  Trend (b0):          {self.params_['init_states'][1]:.4f}")
        print()

        print("Model fit statistics:")
        print(f"  sigma^2:             {self.model_.sigma2:.6f}")
        print(f"  Log-likelihood:      {self.model_.loglik:.2f}")
        print(f"  AIC:                 {self.aic_:.2f}")
        print(f"  BIC:                 {self.bic_:.2f}")
        print()

        print("Residual statistics:")
        print(f"  Mean:                {np.mean(self.in_sample_residuals_):.6f}")
        print(f"  Std Dev:             {np.std(self.in_sample_residuals_, ddof=1):.6f}")
        print(f"  MAE:                 {np.mean(np.abs(self.in_sample_residuals_)):.6f}")
        print(f"  RMSE:                {np.sqrt(np.mean(self.in_sample_residuals_**2)):.6f}")
        print()

        print("Time Series Summary Statistics:")
        print(f"  Mean:                {np.mean(self.y_train_):.4f}")
        print(f"  Std Dev:             {np.std(self.y_train_, ddof=1):.4f}")
        print(f"  Min:                 {np.min(self.y_train_):.4f}")
        print(f"  25%:                 {np.percentile(self.y_train_, 25):.4f}")
        print(f"  Median:              {np.median(self.y_train_):.4f}")
        print(f"  75%:                 {np.percentile(self.y_train_, 75):.4f}")
        print(f"  Max:                 {np.max(self.y_train_):.4f}")

    def get_score(self, y: None = None) -> float:
        """
        R^2 using in-sample fitted values.

        Parameters
        ----------
        y : ignored
            Present for API compatibility.

        Returns
        -------
        score : float
            Coefficient of determination.
        """
        check_is_fitted(self, "model_")
        check_memory_reduced(self, 'score')
        y = self.y_train_
        fitted = self.fitted_values_

        # Handle NaN values if any
        mask = ~(np.isnan(y) | np.isnan(fitted))
        if mask.sum() < 2:
            return float("nan")

        ss_res = np.sum((y[mask] - fitted[mask]) ** 2)
        ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2) + np.finfo(float).eps
        return 1.0 - ss_res / ss_tot

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "m": self.m,
            "model": self.model,
            "damped": self.damped,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "phi": self.phi,
            "lambda_param": self.lambda_param,
            "lambda_auto": self.lambda_auto,
            "bias_adjust": self.bias_adjust,
            "bounds": self.bounds,
            "seasonal": self.seasonal,
            "trend": self.trend,
            "ic": self.ic,
            "allow_multiplicative": self.allow_multiplicative,
            "allow_multiplicative_trend": self.allow_multiplicative_trend,
        }

    def set_params(self, **params) -> Ets:
        """
        Set the parameters of this estimator and reset the fitted state.
        
        This method resets the estimator to its unfitted state whenever parameters
        are changed, requiring the model to be refitted before making predictions.

        Parameters
        ----------
        **params : dict
            Estimator parameters. Valid parameter keys are: 'm', 'model', 'damped',
            'alpha', 'beta', 'gamma', 'phi', 'lambda_param', 'lambda_auto',
            'bias_adjust', 'bounds', 'seasonal', 'trend', 'ic', 'allow_multiplicative',
            'allow_multiplicative_trend'.

        Returns
        -------
        self : Ets
            The estimator with updated parameters and reset state.
            
        Raises
        ------
        ValueError
            If any parameter key is invalid.
        """
        # Validate parameter keys
        valid_params = {
            'm', 'model', 'damped', 'alpha', 'beta', 'gamma', 'phi',
            'lambda_param', 'lambda_auto', 'bias_adjust', 'bounds',
            'seasonal', 'trend', 'ic', 'allow_multiplicative',
            'allow_multiplicative_trend'
        }
        for key in params.keys():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{key}' for estimator {self.__class__.__name__}. "
                    f"Valid parameters are: {sorted(valid_params)}"
                )
        
        # Set the parameters
        for key, value in params.items():
            setattr(self, key, value)
        
        # Reset fitted state - model needs to be refitted with new parameters
        self.model_               = None
        self.model_config_        = None
        self.params_              = None
        self.y_train_             = None
        self.fitted_values_       = None
        self.in_sample_residuals_ = None
        self.n_features_in_       = None
        self.is_memory_reduced    = False
        self.is_fitted            = False
        
        return self

    def reduce_memory(self) -> Ets:
        """
        Reduce memory usage by removing internal arrays not needed for prediction.
        This method clears memory-heavy arrays that are only needed for diagnostics
        but not for prediction. After calling this method, the following methods
        will raise an error:
        
        - fitted_(): In-sample fitted values
        - residuals_(): In-sample residuals
        - score(): R² coefficient
        - summary(): Model summary statistics
        
        Prediction methods remain fully functional:
        
        - predict(): Point forecasts
        - predict_interval(): Prediction intervals
        
        Returns
        -------
        self : Ets
            The estimator with reduced memory usage.
        
        """
        check_is_fitted(self, "model_")
        
        # Clear arrays at Ets level
        self.y_train_ = None
        self.fitted_values_ = None
        self.in_sample_residuals_ = None
        
        # Clear arrays at ETSModel level
        if hasattr(self, 'model_'):
            self.model_.fitted = None
            self.model_.residuals = None
            self.model_.y_original = None
        
        self.is_memory_reduced = True
        
        return self
