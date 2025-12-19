################################################################################
#                                 ARAR                                         #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from .arar._arar_base import (
    arar,
    forecast,
    fitted_arar
)
from ._utils import check_memory_reduced, FastLinearRegression
from ..exceptions import ExogenousInterpretationWarning

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
    max_ar_depth : int, default=None
        Maximum AR depth considered for the (1, i, j, k) AR selection stage.
    max_lag : int, default=None
        Maximum lag used when estimating autocovariances.
    safe : bool, default=True
        If True, falls back to a mean-only model on numerical issues or very
        short series; otherwise errors are raised.

    Attributes
    ----------
    max_ar_depth : int,
        Maximum AR depth considered for the (1, i, j, k) AR selection stage.
    max_lag : int
        Maximum lag used when estimating autocovariances.
    safe : bool
        If True, falls back to a mean-only model on numerical issues or very
        short series; otherwise errors are raised.
    model_ : tuple
        Raw tuple returned by `arar(...)`: (Y, best_phi, best_lag, sigma2, psi, sbar).
    y_ : ndarray of shape (n_samples,)
        Original training series (float).
    coef_ : ndarray of shape (4,)
        Selected AR coefficients for lags (1, i, j, k).
    lags_ : tuple
        Selected lags (1, i, j, k).
    sigma2_ : float
        Innovation variance.
    psi_ : ndarray
        Memory-shortening filter.
    sbar_ : float
        Mean of shortened series.
    exog_model_ : FastLinearRegression
        The fitted regression model for the exogenous variables.
    coef_exog_ : ndarray of shape (n_exog_features,)
        Coefficients of the exogenous variables regression model.
    n_features_in_ : int
        Number of features in the target series (always 1, for sklearn compatibility).
    n_exog_features_in_ : int
        Number of exogenous features seen during fitting (0 if no exog provided).
    fitted_values_ : ndarray of shape (n_samples,)
        In-sample fitted values (NaN for first k-1 terms).
    residuals_in_ : ndarray of shape (n_samples,)
        In-sample residuals (observed - fitted).
    aic_ : float
        Akaike Information Criterion. For models with exogenous variables, this is 
        an approximate calculation that treats the two-step procedure (regression + 
        ARAR) as independent. This may underestimate model complexity. Use primarily 
        for comparing models with the same exogenous structure.
    bic_ : float
        Bayesian Information Criterion. For models with exogenous variables, this is 
        an approximate calculation that treats the two-step procedure (regression + 
        ARAR) as independent. This may underestimate model complexity. Use primarily 
        for comparing models with the same exogenous structure.
    memory_reduced_ : bool
        Flag indicating whether reduce_memory() has been called.
    
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

    def __init__(self, max_ar_depth: int | None = None, max_lag: int | None = None, safe: bool = True):
        self.max_ar_depth = max_ar_depth
        self.max_lag = max_lag
        self.safe = safe
        self.model_ = None
        self.n_features_in_ = None
        self.y_ = None
        self.coef_ = None
        self.lags_ = None
        self.sigma2_ = None
        self.psi_ = None
        self.sbar_ = None
        self.exog_model_ = None
        self.coef_exog_ = None
        self.n_exog_features_in_ = None
        self.memory_reduced_ = False

    def fit(self, y: pd.Series | np.ndarray, exog: pd.Series | pd.DataFrame | np.ndarray | None = None, 
            suppress_warnings: bool = False) -> "Arar":
        """
        Fit the ARAR model to a univariate time series.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Time-ordered numeric sequence.
        exog : Series, DataFrame, or ndarray of shape (n_samples, n_exog_features), default=None
            Exogenous variables to include in the model. See Notes section for details
            on how exogenous variables are handled.
        suppress_warnings : bool, default=False
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
        self.exog_model_ = None

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

        self.model_ = arar(series_to_arar, max_ar_depth=self.max_ar_depth, max_lag=self.max_lag, safe=self.safe)

        (Y, best_phi, best_lag, sigma2, psi, sbar, max_ar_depth, max_lag) = self.model_

        self.y_ = y
        self.coef_ = np.asarray(best_phi, dtype=float)
        self.lags_ = tuple(best_lag)
        self.sigma2_ = float(sigma2)
        self.psi_ = np.asarray(psi, dtype=float)
        self.sbar_ = float(sbar)
        self.max_ar_depth = max_ar_depth
        self.max_lag = max_lag
        self.n_exog_features_in_ = exog.shape[1] if exog is not None else 0
        self.n_features_in_ = 1
        self.memory_reduced_ = False

        arar_fitted = fitted_arar(self.model_)["fitted"]
        if self.exog_model_ is not None:
            exog_fitted = self.exog_model_.predict(exog)
            self.fitted_values_ = exog_fitted + arar_fitted
        else:
            self.fitted_values_ = arar_fitted
        
        # Residuals: original y minus fitted values
        self.residuals_in_ = y - self.fitted_values_

        # Compute AIC and BIC
        # Note: For models with exogenous variables, this is an approximate calculation
        # that treats the two-step procedure (regression + ARAR) as independent stages.
        # This may underestimate model complexity. Use these criteria primarily for
        # comparing models with the same exogenous structure.
        largest_lag = max(self.lags_)
        valid_residuals = self.residuals_in_[largest_lag:]
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

        return self
    
    def predict(self, steps: int, exog: pd.Series | pd.DataFrame | np.ndarray | None = None) -> np.ndarray:
        """
        Generate mean forecasts steps ahead.

        Parameters
        ----------
        steps : int
            Forecast horizon (must be > 0)
        exog : Series, DataFrame, or ndarray of shape (steps, n_exog_features), default=None
            Exogenous variables for prediction.

        Returns
        -------
        mean : ndarray of shape (h,)
            Point forecasts for steps 1..h.
        """
        check_is_fitted(self, "model_")
        if not isinstance(steps, (int, np.integer)) or steps <= 0:
            raise ValueError("`steps` must be a positive integer.")

        # Forecast ARAR component
        arar_pred = forecast(self.model_, h=steps)["mean"]

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
            arar_pred = arar_pred + exog_pred
        
        return arar_pred

    def predict_interval(
        self,
        steps: int = 1,
        level=(80, 95),
        as_frame: bool = True,
        exog: pd.Series | pd.DataFrame | np.ndarray | None = None
    ) -> pd.DataFrame | dict:
        """
        Forecast with symmetric normal-theory prediction intervals.

        Parameters
        ----------
        steps : int, default=1
            Forecast horizon.
        level : iterable of int, default=(80, 95)
            Confidence levels in percent.
        as_frame : bool, default=True
            If True, return a tidy DataFrame with columns:
            'mean', 'lower_<L>', 'upper_<L>' for each level L.
        exog : Series, DataFrame, or ndarray of shape (steps, n_exog_features), default=None
            Exogenous variables for prediction.

        Returns
        -------
        DataFrame or dict
            If as_frame=True: DataFrame indexed by step (1..steps).
            Else: the raw dict from `forecast`.
            
        Notes
        -----
        When exogenous variables are used, prediction intervals account only for 
        ARAR forecast uncertainty and do not include uncertainty from the regression 
        coefficients. This may result in **undercoverage** (actual coverage < nominal level).
        """
        check_is_fitted(self, "model_")
        out = forecast(self.model_, h=steps, level=level)
        
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
            
            out["mean"] = out["mean"] + exog_pred
            # Broadcast the exog prediction across confidence columns
            out["upper"] = out["upper"] + exog_pred[:, np.newaxis]
            out["lower"] = out["lower"] + exog_pred[:, np.newaxis]

        if not as_frame:
            return out

        idx = pd.RangeIndex(1, steps + 1, name="step")
        df = pd.DataFrame({"mean": out["mean"]}, index=idx)
        for i, L in enumerate(out["level"]):
            df[f"lower_{L}"] = out["lower"][:, i]
            df[f"upper_{L}"] = out["upper"][:, i]
        return df

    def get_residuals(self) -> np.ndarray:
        """
        Get in-sample residuals (observed - fitted) from the ARAR model.

        Returns
        -------
        residuals : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "model_")
        check_memory_reduced(self, 'residuals_')
        return self.residuals_in_

    def get_fitted_values(self) -> np.ndarray:
        """
        Get in-sample fitted values from the ARAR model.

        Returns
        -------
        fitted : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "model_")
        check_memory_reduced(self, 'fitted_')
        return self.fitted_values_

    def summary(self) -> None:
        """
        Print a simple textual summary of the fitted ARAR model.
        """
        check_is_fitted(self, "model_")
        check_memory_reduced(self, 'summary')
        
        print("ARAR Model Summary")
        print("------------------")
        print(f"Number of observations: {len(self.y_)}")
        print(f"Selected AR lags: {self.lags_}")
        print(f"AR coefficients (phi): {np.round(self.coef_, 4)}")
        print(f"Residual variance (sigma^2): {self.sigma2_:.4f}")
        print(f"Mean of shortened series (sbar): {self.sbar_:.4f}")
        print(f"Length of memory-shortening filter (psi): {len(self.psi_)}")

        print("\nTime Series Summary Statistics")
        print(f"Mean: {np.mean(self.y_):.4f}")
        print(f"Std Dev: {np.std(self.y_, ddof=1):.4f}")
        print(f"Min: {np.min(self.y_):.4f}")
        print(f"25%: {np.percentile(self.y_, 25):.4f}")
        print(f"Median: {np.median(self.y_):.4f}")
        print(f"75%: {np.percentile(self.y_, 75):.4f}")
        print(f"Max: {np.max(self.y_):.4f}")
        
        print("\nModel Diagnostics")
        print(f"AIC: {self.aic_:.4f}")
        print(f"BIC: {self.bic_:.4f}")
        
        if self.exog_model_ is not None:
            print("\nExogenous Model (Linear Regression)")
            print("-----------------------------------")
            print(f"Number of features: {self.n_exog_features_in_}")
            print(f"Intercept: {self.exog_model_.intercept_:.4f}")
            print(f"Coefficients: {np.round(self.exog_model_.coef_, 4)}")

    def get_score(self, y=None) -> float:
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
        check_is_fitted(self, "model_")
        check_memory_reduced(self, 'score')
        y = self.y_
        fitted = self.fitted_values_
        mask = ~np.isnan(fitted)
        if mask.sum() < 2:
            return float("nan")
        ss_res = np.sum((y[mask] - fitted[mask]) ** 2)
        ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2) + np.finfo(float).eps
        return 1.0 - ss_res / ss_tot

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
        check_is_fitted(self, "model_")
        
        # Clear arrays at Arar level
        self.fitted_values_ = None
        self.residuals_in_ = None

        self.memory_reduced_ = True
        
        return self
