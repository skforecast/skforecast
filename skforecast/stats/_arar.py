################################################################################
#                                     ARAR                                     #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
from typing import Any
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from ..exceptions import ExogenousInterpretationWarning
from .arar._arar_base import (
    arar,
    forecast,
    fitted_arar
)
from ._utils import (
    check_is_fitted, 
    check_memory_reduced, 
    FastLinearRegression
)


class Arar(BaseEstimator, RegressorMixin):
    """
    Scikit-learn style wrapper for the ARAR time-series model.

    This estimator treats a univariate sequence as "the feature".
    Call `fit(y)` with a 1D array-like of observations in time order, then
    produce out-of-sample forecasts via `predict(steps)` and prediction intervals
    via `predict_interval(steps, level=...)`. In-sample diagnostics are available
    through `fitted_`, `residuals_()` and `summary()`.

    Parameters
    ----------
    max_ar_depth : int, default None
        Maximum AR depth considered for the (1, i, j, k) AR selection stage.
    max_lag : int, default None
        Maximum lag used when estimating autocovariances.
    safe : bool, default True
        If True, falls back to a mean-only model on numerical issues or very
        short series; otherwise errors are raised.

    Attributes
    ----------
    max_ar_depth : int or None
        Maximum AR depth considered for the (1, i, j, k) AR selection stage during 
        model fitting. When None, a default value is determined automatically based 
        on the series length.
    max_lag : int or None
        Maximum lag used when estimating autocovariances during the memory-shortening 
        step. When None, a default value is determined automatically based on the 
        series length.
    safe : bool
        Whether to use safe mode. When True, the model falls back to a mean-only 
        forecast on numerical issues or very short series. When False, errors are 
        raised instead.
    model_ : tuple or None
        Raw tuple returned by the underlying ARAR algorithm containing: 
        (Y, best_phi, best_lag, sigma2, psi, sbar, max_ar_depth, max_lag). 
        Available after calling `fit()`.
    coef_ : ndarray of shape (4,) or None
        Estimated AR coefficients for the selected lags (1, i, j, k). Some 
        coefficients may be zero if the corresponding lag was not selected. 
        Available after calling `fit()`.
    lags_ : tuple or None
        Selected lag indices (1, i, j, k) used in the AR model, where each 
        represents which past observations contribute to the forecast. 
        Available after calling `fit()`.
    sigma2_ : float or None
        Estimated innovation variance (one-step-ahead forecast error variance) 
        from the fitted ARAR model. Available after calling `fit()`.
    psi_ : ndarray or None
        Memory-shortening filter coefficients used to transform the original 
        series into one with shorter memory before AR fitting. Available after 
        calling `fit()`.
    sbar_ : float or None
        Mean of the memory-shortened series, used as the long-run mean in 
        forecasting. Available after calling `fit()`.
    aic_ : float or None
        Akaike Information Criterion measuring model fit quality while penalizing 
        complexity. For models with exogenous variables, this is an approximate 
        calculation that treats the two-step procedure (regression + ARAR) as 
        independent stages, which may underestimate total model complexity. 
        Available after calling `fit()`.
    bic_ : float or None
        Bayesian Information Criterion, similar to AIC but with a stronger penalty 
        for model complexity. For models with exogenous variables, this is an 
        approximate calculation that treats the two-step procedure (regression + 
        ARAR) as independent stages, which may underestimate total model complexity. 
        Available after calling `fit()`.
    exog_model_ : FastLinearRegression or None
        Fitted linear regression model for exogenous variables. When exogenous 
        variables are provided during fitting, this model captures their linear 
        relationship with the target series. Available after calling `fit()` with 
        exogenous variables.
    coef_exog_ : ndarray of shape (n_exog_features,) or None
        Coefficients from the exogenous variables regression model, excluding the 
        intercept. Available after calling `fit()` with exogenous variables.
    n_exog_features_in_ : int or None
        Number of exogenous features used during fitting. Zero if no exogenous 
        variables were provided. Available after calling `fit()`.
    y_train_ : ndarray of shape (n_samples,) or None
        Original training time series used to fit the model.
    fitted_values_ : ndarray of shape (n_samples,) or None
        One-step-ahead in-sample fitted values. The first k-1 values may be NaN 
        where k is the largest lag used.
    in_sample_residuals_ : ndarray of shape (n_samples,) or None
        In-sample residuals calculated as the difference between observed values 
        and fitted values.
    n_features_in_ : int or None
        Number of features (time series) seen during `fit()`. For ARAR, this is 
        always 1 as it handles univariate time series (present for scikit-learn 
        compatibility). Available after calling `fit()`.
    is_memory_reduced : bool
        Flag indicating whether `reduce_memory()` has been called to clear diagnostic 
        arrays (y_train_, fitted_values_, in_sample_residuals_).
    is_fitted : bool
        Flag indicating whether the model has been successfully fitted to data.
    estimator_name_ : str
        String identifier of the fitted model configuration (e.g., "Arar(lags=[1,2,3])"). 
        This is updated after fitting to reflect the selected model.
    
    Notes
    -----
    When exogenous variables are provided during fitting, the model uses a
    two-step approach (regression followed by ARAR on residuals). In this
    approach, the target series is first regressed on the exogenous variables
    using a linear regression model. The residuals from this regression,
    representing the portion of the series not explained by the exogenous
    variables, are then modeled using the ARAR model.

    This design allows the influence of exogenous variables to be incorporated
    prior to applying the ARAR model, rather than within the ARAR dynamics
    themselves.

    This two-step approach is necessary because the ARAR model is inherently
    univariate and does not natively support exogenous variables. By separating
    the regression step, the method preserves the original ARAR formulation
    while still capturing the effects of external predictors.

    However, this approach carries important assumptions and implications:

    - The relationship between the target series and the exogenous variables is
    assumed to be linear and time-invariant.
    - The ARAR model is applied only to the residual process, meaning its
    parameters describe the dynamics of the series after removing the
    contribution of exogenous variables.
    - As a result, the interpretability of the ARAR parameters changes: they no
    longer describe the full data-generating process, but rather the behavior
    of the unexplained component.

    Despite these limitations, this strategy provides a practical and
    computationally efficient way to incorporate exogenous information into an
    otherwise univariate ARAR framework.

    """

    def __init__(
        self, 
        max_ar_depth: int | None = None, 
        max_lag: int | None = None, 
        safe: bool = True
    ):
        self.max_ar_depth           = max_ar_depth
        self.max_lag                = max_lag
        self.safe                   = safe
        self.lags_                  = None
        self.sigma2_                = None
        self.psi_                   = None
        self.sbar_                  = None
  
        self.model_                 = None
        self.coef_                  = None
        self.aic_                   = None
        self.bic_                   = None
        self.exog_model_            = None
        self.coef_exog_             = None
        self.n_exog_features_in_    = None
        self.y_train_               = None
        self.fitted_values_         = None
        self.in_sample_residuals_   = None
        self.n_features_in_         = None
        self.is_memory_reduced      = False
        self.is_fitted              = False
        self.estimator_name_        = "Arar()"

    def __repr__(self) -> str:
        """
        Information displayed when an Arar object is printed.
        """
        return self.estimator_name_

    def fit(
        self, 
        y: np.ndarray | pd.Series, 
        exog: np.ndarray | pd.Series | pd.DataFrame | None = None,
        suppress_warnings: bool = False
    ) -> "Arar":
        """
        Fit the ARAR model to a univariate time series.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Time-ordered numeric sequence.
        exog : Series, DataFrame, or ndarray of shape (n_samples, n_exog_features), default None
            Exogenous variables to include in the model. See Notes section for details
            on how exogenous variables are handled.
        suppress_warnings : bool, default False
            If True, suppresses the warning about exogenous variables affecting model
            interpretation.

        Returns
        -------
        self : Arar
            Fitted estimator.

        Notes
        -----
        When exogenous variables are provided during fitting, the model uses a
        two-step approach (regression followed by ARAR on residuals). In this
        approach, the target series is first regressed on the exogenous variables
        using a linear regression model. The residuals from this regression,
        representing the portion of the series not explained by the exogenous
        variables, are then modeled using the ARAR model.

        This design allows the influence of exogenous variables to be incorporated
        prior to applying the ARAR model, rather than within the ARAR dynamics
        themselves.

        This two-step approach is necessary because the ARAR model is inherently
        univariate and does not natively support exogenous variables. By separating
        the regression step, the method preserves the original ARAR formulation
        while still capturing the effects of external predictors.

        However, this approach carries important assumptions and implications:

        - The relationship between the target series and the exogenous variables is
        assumed to be linear and time-invariant.
        - The ARAR model is applied only to the residual process, meaning its
        parameters describe the dynamics of the series after removing the
        contribution of exogenous variables.
        - As a result, the interpretability of the ARAR parameters changes: they no
        longer describe the full data-generating process, but rather the behavior
        of the unexplained component.

        Despite these limitations, this strategy provides a practical and
        computationally efficient way to incorporate exogenous information into an
        otherwise univariate ARAR framework.

        """

        self.lags_                = None
        self.sigma2_              = None
        self.psi_                 = None
        self.sbar_                = None

        self.model_               = None
        self.coef_                = None
        self.aic_                 = None
        self.bic_                 = None
        self.exog_model_          = None
        self.coef_exog_           = None
        self.n_exog_features_in_  = None
        self.y_train_             = None
        self.fitted_values_       = None
        self.in_sample_residuals_ = None
        self.n_features_in_       = None
        self.is_memory_reduced    = False
        self.is_fitted            = False

        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("`y` must be a pandas Series or numpy ndarray.")
        
        if not isinstance(exog, (type(None), pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError("`exog` must be None, a pandas Series, pandas DataFrame, or numpy ndarray.")
        
        y = np.asarray(y, dtype=float)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        elif y.ndim != 1:
            raise ValueError("`y` must be a 1D array-like sequence.")
        
        series_to_arar = y

        if exog is not None:
            if not suppress_warnings:
                warnings.warn(
                    "Exogenous variables are being handled using a two-step approach: "
                    "(1) linear regression on exog, (2) ARAR on residuals. "
                    "This affects model interpretation:\n"
                    "  - ARAR coefficients (coef_) describe residual dynamics, not the original series\n"
                    "  - Pred intervals reflect only ARAR uncertainty, not exog regression uncertainty\n"
                    "  - Assumes a linear, time-invariant relationship between exog and target\n"
                    "For more details, see the fit() method's Notes section of ARAR class. ",
                    ExogenousInterpretationWarning
                )
            
            exog = np.asarray(exog, dtype=float)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            elif exog.ndim != 2:
                raise ValueError("`exog` must be 1D or 2D.")
            
            if len(exog) != len(y):
                raise ValueError(f"Length of exog ({len(exog)}) must match length of y ({len(y)})")

            self.exog_model_ = FastLinearRegression()
            self.exog_model_.fit(exog, y)
            self.coef_exog_ = self.exog_model_.coef_
            series_to_arar = y - self.exog_model_.predict(exog)

        if series_to_arar.size < 2 and not self.safe:
            raise ValueError("Series too short to fit ARAR when safe=False.")

        self.model_ = arar(
            series_to_arar, max_ar_depth=self.max_ar_depth, max_lag=self.max_lag, safe=self.safe
        )

        (Y, best_phi, best_lag, sigma2, psi, sbar, max_ar_depth, max_lag) = self.model_

        self.max_ar_depth        = max_ar_depth
        self.max_lag             = max_lag
        self.lags_               = tuple(best_lag)
        self.sigma2_             = float(sigma2)
        self.psi_                = np.asarray(psi, dtype=float)
        self.sbar_               = float(sbar)
        self.coef_               = np.asarray(best_phi, dtype=float)
        self.y_train_            = y
        self.n_exog_features_in_ = exog.shape[1] if exog is not None else 0
        self.n_features_in_      = 1       
        self.is_memory_reduced   = False
        self.is_fitted           = True

        arar_fitted = fitted_arar(self.model_)["fitted"]
        if self.exog_model_ is not None:
            exog_fitted = self.exog_model_.predict(exog)
            self.fitted_values_ = exog_fitted + arar_fitted
        else:
            self.fitted_values_ = arar_fitted
        
        # Residuals: original y minus fitted values
        self.in_sample_residuals_ = y - self.fitted_values_

        # Compute AIC and BIC
        # Note: For models with exogenous variables, this is an approximate calculation
        # that treats the two-step procedure (regression + ARAR) as independent stages.
        # This may underestimate model complexity. Use these criteria primarily for
        # comparing models with the same exogenous structure.
        largest_lag = max(self.lags_)
        valid_residuals = self.in_sample_residuals_[largest_lag:]
        # Remove NaN values for AIC/BIC calculation
        valid_residuals = valid_residuals[~np.isnan(valid_residuals)]
        n = len(valid_residuals)
        if n > 0:
            # Count parameters:
            # - ARAR: 4 AR coefficients + 1 mean parameter (sbar) + 1 variance (sigma2) = 6
            # - Exog: n_exog coefficients + 1 intercept (if exog present)
            # Note: We count all 4 AR coefficients even if some are zero, as they were
            # selected during model fitting. The variance parameter sigma2 is also estimated.
            k_arar = 6  # 4 AR coefficients + sbar + sigma2
            k_exog = (self.n_exog_features_in_ + 1) if self.exog_model_ is not None else 0  # +1 for intercept
            k = k_arar + k_exog
            sigma2 = max(np.sum(valid_residuals ** 2) / n, 1e-12)  # Ensure positive
            loglik = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1)
            self.aic_ = -2 * loglik + 2 * k
            self.bic_ = -2 * loglik + k * np.log(n)
        else:
            self.aic_ = np.nan
            self.bic_ = np.nan

        self.estimator_name_ = f"Arar(lags={self.lags_})"

        return self
    
    @check_is_fitted
    def predict(
        self, 
        steps: int, 
        exog: np.ndarray | pd.Series | pd.DataFrame | None = None
    ) -> np.ndarray:
        """
        Generate mean forecasts steps ahead.

        Parameters
        ----------
        steps : int
            Forecast horizon (must be > 0)
        exog : ndarray, Series or DataFrame of shape (steps, n_exog_features), default None
            Exogenous variables for prediction.

        Returns
        -------
        predictions : ndarray of shape (h,)
            Point forecasts for steps 1..h.
        
        """

        if not isinstance(steps, (int, np.integer)) or steps <= 0:
            raise ValueError("`steps` must be a positive integer.")

        # Forecast ARAR component
        predictions = forecast(self.model_, h=steps)["mean"]

        if self.exog_model_ is None and exog is not None:
            raise ValueError(
                "Model was fitted without exog, but `exog` was provided for prediction. "
                "Please refit the model with exogenous variables."
            )

        if self.exog_model_ is not None:
            if exog is None:
                raise ValueError("Model was fitted with exog, so `exog` is required for prediction.")
            exog = np.asarray(exog, dtype=float)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            elif exog.ndim != 2:
                raise ValueError("`exog` must be 1D or 2D.")
            
            # Check feature consistency
            if exog.shape[1] != self.n_exog_features_in_:
                raise ValueError(f"Mismatch in exogenous features: fitted with {self.n_exog_features_in_}, got {exog.shape[1]}.")
            
            if len(exog) != steps:
                raise ValueError(f"Length of exog ({len(exog)}) must match steps ({steps}).")

            # Forecast Regression component
            exog_pred = self.exog_model_.predict(exog)
            predictions = predictions + exog_pred
        
        return predictions

    @check_is_fitted
    def predict_interval(
        self,
        steps: int = 1,
        level=(80, 95),
        as_frame: bool = True,
        exog: np.ndarray | pd.Series | pd.DataFrame | None = None
    ) -> np.ndarray | pd.DataFrame:
        """
        Forecast with symmetric normal-theory prediction intervals.

        Parameters
        ----------
        steps : int, default 1
            Forecast horizon.
        level : iterable of int, default (80, 95)
            Confidence levels in percent.
        as_frame : bool, default True
            If True, return a tidy DataFrame with columns 'mean', 'lower_<L>',
            'upper_<L>' for each level L. If False, return a NumPy ndarray.
        exog : ndarray, Series or DataFrame of shape (steps, n_exog_features), default None
            Exogenous variables for prediction.

        Returns
        -------
        predictions : numpy ndarray, pandas DataFrame
            If as_frame=True, pandas DataFrame with columns 'mean', 'lower_<L>',
            'upper_<L>' for each level L. If as_frame=False, numpy ndarray.
            
        Notes
        -----
        When exogenous variables are used, prediction intervals account only for 
        ARAR forecast uncertainty and do not include uncertainty from the regression 
        coefficients. This may result in **undercoverage** (actual coverage < nominal level).

        """

        if not isinstance(steps, (int, np.integer)) or steps <= 0:
            raise ValueError("`steps` must be a positive integer.")
            
        raw_preds = forecast(self.model_, h=steps, level=level)
        
        if self.exog_model_ is None and exog is not None:
            raise ValueError(
                "Model was fitted without exog, but `exog` was provided for prediction. "
                "Please refit the model with exogenous variables."
            )
        
        if self.exog_model_ is not None:
            if exog is None:
                raise ValueError("Model was fitted with exog, so `exog` is required for prediction.")
            exog = np.asarray(exog, dtype=float)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            elif exog.ndim != 2:
                raise ValueError("`exog` must be 1D or 2D.")

            # Check feature consistency
            if exog.shape[1] != self.n_exog_features_in_:
                raise ValueError(
                    f"Mismatch in exogenous features: fitted with {self.n_exog_features_in_}, "
                    f"got {exog.shape[1]}.")
            
            if len(exog) != steps:
                raise ValueError(f"Length of exog ({len(exog)}) must match steps ({steps}).")

            exog_pred = self.exog_model_.predict(exog)
            
            raw_preds["mean"] = raw_preds["mean"] + exog_pred
            # Broadcast the exog prediction across confidence columns
            raw_preds["upper"] = raw_preds["upper"] + exog_pred[:, np.newaxis]
            raw_preds["lower"] = raw_preds["lower"] + exog_pred[:, np.newaxis]

        levels = raw_preds["level"]
        n_levels = len(levels)
        cols = [raw_preds["mean"]]
        for i in range(n_levels):
            cols.append(raw_preds["lower"][:, i])
            cols.append(raw_preds["upper"][:, i])
        
        predictions = np.column_stack(cols)

        if as_frame:
            col_names = ["mean"]
            for level in levels:
                level = int(level)
                col_names.append(f"lower_{level}")
                col_names.append(f"upper_{level}")
            
            predictions = pd.DataFrame(
                predictions, columns=col_names, index=pd.RangeIndex(1, steps + 1, name="step")
            )
        
        return predictions

    @check_is_fitted
    def get_residuals(self) -> np.ndarray:
        """
        Get in-sample residuals (observed - fitted) from the ARAR model.

        Returns
        -------
        residuals : ndarray of shape (n_samples,)

        """

        check_memory_reduced(self, method_name='get_residuals')
        return self.in_sample_residuals_

    @check_is_fitted
    def get_fitted_values(self) -> np.ndarray:
        """
        Get in-sample fitted values from the ARAR model.

        Returns
        -------
        fitted : ndarray of shape (n_samples,)

        """

        check_memory_reduced(self, method_name='get_fitted_values')
        return self.fitted_values_

    @check_is_fitted
    def get_score(self, y: Any = None) -> float:
        """
        R^2 using in-sample fitted values (ignores initial NaNs).

        Parameters
        ----------
        y : ignored
            Present for API compatibility.

        Returns
        -------
        score : float
            Coefficient of determination.
        
        """

        check_memory_reduced(self, method_name='get_score')

        y = self.y_train_
        fitted = self.fitted_values_

        mask = ~np.isnan(fitted)
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
        deep : bool, default True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        
        """
        
        return {
            "max_ar_depth": self.max_ar_depth,
            "max_lag": self.max_lag,
            "safe": self.safe
        }
    
    @check_is_fitted
    def get_feature_importances(self) -> pd.DataFrame:
        """Get feature importances for Arar model."""
        importances = pd.DataFrame({
            'feature': [f'lag_{lag}' for lag in self.lags_],
            'importance': self.coef_
        })

        if self.coef_exog_ is not None:
            exog_importances = pd.DataFrame({
                'feature': [f'exog_{i}' for i in range(self.coef_exog_.shape[0])],
                'importance': self.coef_exog_
            })
            importances = pd.concat([importances, exog_importances], ignore_index=True)
            warnings.warn(
                    "Exogenous variables are being handled using a two-step approach: "
                    "(1) linear regression on exog, (2) ARAR on residuals. "
                    "This affects model interpretation:\n"
                    "  - ARAR coefficients (coef_) describe residual dynamics, not the original series\n"
                    "  - Exogenous coefficients (coef_exog_) describe exogenous impact on original series",
                ExogenousInterpretationWarning
            )

        return importances
    
    @check_is_fitted
    def get_info_criteria(self, criteria: str) -> float:
        """
        Get information criteria.

        Parameters
        ----------
        criteria : str
            Information criterion to retrieve. Valid options are 'aic' and 'bic'.
        Returns
        -------
        info_criteria : float
            Value of the requested information criterion.

        """
        if criteria not in {'aic', 'bic'}:
            raise ValueError(
                "Invalid value for `criteria`. Valid options are 'aic' and 'bic' "
                "for ARAR model."
            )
        
        if criteria == 'aic':
            value = self.aic_
        else:
            value = self.bic_
        
        return value
    
    def set_params(self, **params) -> "Arar":
        """
        Set the parameters of this estimator and reset the fitted state.
        
        This method resets the estimator to its unfitted state whenever parameters
        are changed, requiring the model to be refitted before making predictions.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters. Valid parameter keys are 'max_ar_depth', 'max_lag',
            and 'safe'.
        
        Returns
        -------
        self : Arar
            The estimator with updated parameters and reset state.
        
        Raises
        ------
        ValueError
            If any parameter key is invalid.
        
        """

        valid_params = {'max_ar_depth', 'max_lag', 'safe'}
        for key in params.keys():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{key}' for estimator {self.__class__.__name__}. "
                    f"Valid parameters are: {valid_params}"
                )
        
        for key, value in params.items():
            setattr(self, key, value)
        
        # Reset fitted state
        self.lags_                  = None
        self.sigma2_                = None
        self.psi_                   = None
        self.sbar_                  = None
  
        self.model_                 = None
        self.coef_                  = None
        self.aic_                   = None
        self.bic_                   = None
        self.exog_model_            = None
        self.coef_exog_             = None
        self.n_exog_features_in_    = None
        self.y_train_               = None
        self.fitted_values_         = None
        self.in_sample_residuals_   = None
        self.n_features_in_         = None
        self.is_memory_reduced      = False
        self.is_fitted              = False
        self.estimator_name_        = "Arar()"
        
        return self
    
    @check_is_fitted
    def summary(self) -> None:
        """
        Print a simple textual summary of the fitted Arar model.
        """
        
        print(f"{self.estimator_name_} Model Summary")
        print("------------------")
        print(f"Selected AR lags:                         {self.lags_}")
        print(f"AR coefficients (phi):                    {np.round(self.coef_, 4)}")
        print(f"Residual variance (sigma^2):              {self.sigma2_:.4f}")
        print(f"Mean of shortened series (sbar):          {self.sbar_:.4f}")
        print(f"Length of memory-shortening filter (psi): {len(self.psi_)}")

        if not self.is_memory_reduced:
            print("\nTime Series Summary Statistics")
            print(f"Number of observations: {len(self.y_train_)}")
            print(f"Mean:                   {np.mean(self.y_train_):.4f}")
            print(f"Std Dev:                {np.std(self.y_train_, ddof=1):.4f}")
            print(f"Min:                    {np.min(self.y_train_):.4f}")
            print(f"25%:                    {np.percentile(self.y_train_, 25):.4f}")
            print(f"Median:                 {np.median(self.y_train_):.4f}")
            print(f"75%:                    {np.percentile(self.y_train_, 75):.4f}")
            print(f"Max:                    {np.max(self.y_train_):.4f}")
        
        print("\nModel Diagnostics")
        print(f"AIC: {self.aic_:.4f}")
        print(f"BIC: {self.bic_:.4f}")
        
        if self.exog_model_ is not None:
            print("\nExogenous Model (Linear Regression)")
            print("-----------------------------------")
            print(f"Number of features: {self.n_exog_features_in_}")
            print(f"Intercept: {self.exog_model_.intercept_:.4f}")
            print(f"Coefficients: {np.round(self.exog_model_.coef_, 4)}")

    @check_is_fitted
    def reduce_memory(self) -> "Arar":
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
        self : Arar
            The estimator with reduced memory usage.
        
        """
        
        self.fitted_values_ = None
        self.in_sample_residuals_ = None

        self.is_memory_reduced = True
        
        return self
