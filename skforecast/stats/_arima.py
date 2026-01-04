################################################################################
#                                 ARIMA                                        #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.    #
################################################################################
# coding=utf-8

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from .arima._arima_base import (
    arima,
    predict_arima
)
from ._utils import check_memory_reduced


class Arima(BaseEstimator, RegressorMixin):
    """
    Scikit-learn style wrapper for the ARIMA (AutoRegressive Integrated Moving Average) model.

    This estimator treats a univariate time series as input. Call `fit(y)` with 
    a 1D array-like of observations in time order, then produce out-of-sample 
    forecasts via `predict(steps)` and prediction intervals via `predict_interval(steps, level=...)`. 
    In-sample diagnostics are available through `fitted_`, `residuals_()` and `summary()`.

    Parameters
    ----------
    order : tuple of int, default=(0, 0, 0)
        The (p, d, q) order of the non-seasonal ARIMA model:
        - p: AR order (number of lag observations)
        - d: Degree of differencing (number of times to difference the series)
        - q: MA order (size of moving average window)
    seasonal_order : tuple of int, default=(0, 0, 0)
        The (P, D, Q) order of the seasonal component:
        - P: Seasonal AR order
        - D: Seasonal differencing order
        - Q: Seasonal MA order
    m : int, default=1
        Seasonal period (e.g., 12 for monthly data with yearly seasonality, 
        4 for quarterly data). Set to 1 for non-seasonal models.
    include_mean : bool, default=True
        Whether to include a mean/intercept term in the model. Only applies 
        when there is no differencing (d=0 and D=0).
    transform_pars : bool, default=True
        Whether to transform parameters to ensure stationarity and invertibility 
        during optimization.
    method : str, default="CSS-ML"
        Estimation method. Options:
        - "CSS-ML": Conditional sum of squares for initial values, then maximum likelihood
        - "ML": Maximum likelihood only
        - "CSS": Conditional sum of squares only
    n_cond : int, optional
        Number of initial observations to use for conditional sum of squares. 
        If None, defaults to max(p + d*m + P*m, q + Q*m).
    SSinit : str, default="Gardner1980"
        Method for state-space initialization. Options:
        - "Gardner1980": Gardner's method (default, more numerically stable)
        - "Rossignol2011": Rossignol's method (alternative)
    optim_method : str, default="BFGS"
        Optimization method passed to scipy.optimize.minimize. Common options 
        include "BFGS", "L-BFGS-B", "Nelder-Mead", etc.
    optim_control : dict, optional
        Additional options passed to the optimizer (e.g., maxiter, ftol).
    kappa : float, default=1e6
        Prior variance for diffuse states in the Kalman filter.

    Attributes
    ----------
    model_ : dict
        Dictionary containing the fitted ARIMA model with keys:
        - 'y': Original training series
        - 'fitted': In-sample fitted values
        - 'coef': Coefficient DataFrame
        - 'sigma2': Innovation variance
        - 'var_coef': Variance-covariance matrix
        - 'loglik': Log-likelihood
        - 'aic': Akaike Information Criterion
        - 'bic': Bayesian Information Criterion
        - 'arma': ARIMA specification [p, q, P, Q, m, d, D]
        - 'residuals': Model residuals
        - 'converged': Convergence status
        - 'model': State-space model dict
        - 'method': Estimation method string
    y_train_ : ndarray of shape (n_samples,)
        Original training series.
    coef_ : ndarray
        Flattened array of fitted coefficients (AR, MA, xreg, intercept).
    coef_names_ : list of str
        Names of coefficients in coef_.
    sigma2_ : float
        Innovation variance (residual variance).
    loglik_ : float
        Log-likelihood of the fitted model.
    aic_ : float
        Akaike Information Criterion.
    bic_ : float
        Bayesian Information Criterion.
    arma_ : list of int
        ARIMA specification: [p, q, P, Q, m, d, D].
    converged_ : bool
        Whether the optimization converged successfully.
    n_features_in_ : int
        Number of features in the target series (always 1, for sklearn compatibility).
    n_exog_features_in_ : int
        Number of exogenous features seen during fitting (0 if no exog provided).
    fitted_values_ : ndarray of shape (n_samples,)
        In-sample fitted values.
    in_sample_residuals_ : ndarray of shape (n_samples,)
        In-sample residuals (observed - fitted).
    var_coef_ : ndarray
        Variance-covariance matrix of coefficients.
    is_memory_reduced : bool
        Flag indicating whether reduce_memory() has been called.
    estimator_id : str
        String identifier for the model configuration (e.g., "Arima(1,1,1)(0,0,0)[1]").

    Notes
    -----
    The ARIMA model supports exogenous regressors (xreg) which are incorporated 
    directly into the likelihood function, unlike the two-step approach used in 
    the ARAR model. This means the exogenous variables are modeled jointly with 
    the ARMA errors, providing a more integrated treatment.

    The model uses a state-space representation and the Kalman filter for 
    likelihood computation and forecasting, which allows handling of missing 
    values and provides efficient recursive prediction.

    Examples
    --------
    >>> from skforecast.stats import Arima
    >>> import numpy as np
    >>> 
    >>> # Generate sample data
    >>> np.random.seed(42)
    >>> y = np.cumsum(np.random.randn(100)) + 50
    >>> 
    >>> # Fit ARIMA(1,1,1) model
    >>> model = Arima(order=(1, 1, 1))
    >>> model.fit(y)
    >>> 
    >>> # Forecast 10 steps ahead
    >>> predictions = model.predict(steps=10)
    >>> 
    >>> # Get prediction intervals
    >>> intervals = model.predict_interval(steps=10, level=(80, 95))
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (0, 0, 0),
        seasonal_order: tuple[int, int, int] = (0, 0, 0),
        m: int = 1,
        include_mean: bool = True,
        transform_pars: bool = True,
        method: str = "CSS-ML",
        n_cond: int | None = None,
        SSinit: str = "Gardner1980",
        optim_method: str = "BFGS",
        optim_control: dict | None = None,
        kappa: float = 1e6,
    ):

        if len(order) != 3:
            raise ValueError(
                f"`order` must be a tuple of length 3, got length {len(order)}"
            )
        if len(seasonal_order) != 3:
            raise ValueError(
                f"`seasonal_order` must be a tuple of length 3, got length {len(seasonal_order)}"
            )
        
        self.order             = order
        self.seasonal_order    = seasonal_order
        self.m                 = m
        self.include_mean      = include_mean
        self.transform_pars    = transform_pars
        self.method            = method
        self.n_cond            = n_cond
        self.SSinit            = SSinit
        self.optim_method      = optim_method
        self.optim_control     = optim_control
        self.kappa             = kappa
        self.is_memory_reduced = False

        p, d, q = self.order
        P, D, Q = self.seasonal_order
        if P == 0 and D == 0 and Q == 0:
            self.estimator_id = f"Arima({p},{d},{q})"
        else:
            self.estimator_id = f"Arima({p},{d},{q})({P},{D},{Q})[{self.m}]"

    def __repr__(self) -> str:
        """
        Information displayed when an Arima object is printed.
        """
        return self.estimator_id

    def fit(
        self, 
        y: pd.Series | np.ndarray, 
        exog: pd.Series | pd.DataFrame | np.ndarray | None = None
    ) -> "Arima":
        """
        Fit the ARIMA model to a univariate time series.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Time-ordered numeric sequence.
        exog : Series, DataFrame, or ndarray of shape (n_samples, n_exog_features), default=None
            Exogenous regressors to include in the model. These are incorporated 
            directly into the ARIMA likelihood function.

        Returns
        -------
        self : Arima
            Fitted estimator.

        Notes
        -----
        Unlike the ARAR model, ARIMA natively supports exogenous regressors which 
        are incorporated directly into the likelihood function. This provides a 
        more integrated treatment where the exogenous variables and ARMA dynamics 
        are jointly estimated.
        """
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("`y` must be a pandas Series or numpy array.")
        
        if not isinstance(exog, (type(None), pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError("`exog` must be a pandas Series, DataFrame, numpy array, or None.")
        
        y = np.asarray(y, dtype=float)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        elif y.ndim != 1:
            raise ValueError("`y` must be 1-dimensional.")
        
        if exog is not None:
            exog = np.asarray(exog, dtype=float)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            elif exog.ndim != 2:
                raise ValueError("`exog` must be 1- or 2-dimensional.")
            
            if len(exog) != len(y):
                raise ValueError(
                    f"Length of `exog` ({len(exog)}) does not match length of `y` ({len(y)})."
                )
        
        self.model_ = arima(
            x              = y,
            m              = self.m,
            order          = self.order,
            seasonal       = self.seasonal_order,
            xreg           = exog,
            include_mean   = self.include_mean,
            transform_pars = self.transform_pars,
            fixed          = None,
            init           = None,
            method         = self.method,
            n_cond         = self.n_cond,
            SSinit         = self.SSinit,
            optim_method   = self.optim_method,
            optim_control  = self.optim_control,
            kappa          = self.kappa
        )
        
        # Extract and store model attributes
        self.y_train_             = self.model_['y']
        self.coef_                = self.model_['coef'].values.flatten()
        self.coef_names_          = list(self.model_['coef'].columns)
        self.sigma2_              = self.model_['sigma2']
        self.loglik_              = self.model_['loglik']
        self.aic_                 = self.model_['aic']
        self.bic_                 = self.model_['bic']
        self.arma_                = self.model_['arma']
        self.converged_           = self.model_['converged']
        self.fitted_values_       = self.model_['fitted']
        self.in_sample_residuals_ = self.model_['residuals']
        self.var_coef_            = self.model_['var_coef']
        self.n_exog_features_in_  = exog.shape[1] if exog is not None else 0
        self.n_features_in_       = 1
        self.is_memory_reduced    = False
        
        return self

    def predict(
        self, 
        steps: int, 
        exog: pd.Series | pd.DataFrame | np.ndarray | None = None
    ) -> np.ndarray:
        """
        Generate mean forecasts steps ahead.

        Parameters
        ----------
        steps : int
            Forecast horizon (must be > 0).
        exog : Series, DataFrame, or ndarray of shape (steps, n_exog_features), default=None
            Exogenous regressors for the forecast period. Must have the same 
            number of features as used during fitting.

        Returns
        -------
        mean : ndarray of shape (steps,)
            Point forecasts for steps 1..steps.

        Raises
        ------
        ValueError
            If model hasn't been fitted, steps <= 0, or exog shape is incorrect.
        """
        check_is_fitted(self, "model_")
        
        if not isinstance(steps, (int, np.integer)) or steps <= 0:
            raise ValueError("`steps` must be a positive integer.")
        
        # Convert exog to numpy array if needed
        if exog is not None:
            exog = np.asarray(exog, dtype=float)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            elif exog.ndim != 2:
                raise ValueError("`exog` must be 1- or 2-dimensional.")
            
            if len(exog) != steps:
                raise ValueError(
                    f"Length of `exog` ({len(exog)}) must match `steps` ({steps})."
                )
            
            if exog.shape[1] != self.n_exog_features_in_:
                raise ValueError(
                    f"Number of exogenous features ({exog.shape[1]}) does not match "
                    f"the number used during fitting ({self.n_exog_features_in_})."
                )
        elif self.n_exog_features_in_ > 0:
            raise ValueError(
                f"Model was fitted with {self.n_exog_features_in_} exogenous features, "
                f"but `exog` was not provided for prediction."
            )
        
        # Generate predictions using base implementation
        result = predict_arima(
            model   = self.model_,
            n_ahead = steps,
            newxreg = exog,
            se_fit  = False
        )
        
        return result['pred']

    def predict_interval(
        self,
        steps: int = 1,
        level: list[float] | tuple[float, ...] | None = None,
        alpha: float | None = None,
        as_frame: bool = True,
        exog: pd.Series | pd.DataFrame | np.ndarray | None = None,
    ) -> pd.DataFrame | np.ndarray:
        """
        Forecast with prediction intervals.

        Parameters
        ----------
        steps : int, default=1
            Forecast horizon.
        level : list or tuple of float, default=None
            Confidence levels in percent (e.g., 80 for 80% intervals).
            If None and alpha is None, defaults to (80, 95).
            Cannot be specified together with `alpha`.
        alpha : float, default=None
            The significance level for the prediction interval. 
            If specified, the confidence interval will be (1 - alpha) * 100%.
            For example, alpha=0.05 gives 95% intervals.
            Cannot be specified together with `level`.
        as_frame : bool, default=True
            If True, return a tidy DataFrame with columns:
            'mean', 'lower_<L>', 'upper_<L>' for each level L.
            If False, return a NumPy ndarray with columns ordered as in the DataFrame.
        exog : Series, DataFrame, or ndarray of shape (steps, n_exog_features), default=None
            Exogenous regressors for the forecast period.

        Returns
        -------
        DataFrame or ndarray
            If as_frame=True: DataFrame indexed by step (1..steps) with columns:
            'mean', 'lower_<L>', 'upper_<L>' for each level L.
            If as_frame=False: NumPy ndarray with columns ordered as in the DataFrame.

        Raises
        ------
        ValueError
            If model hasn't been fitted, steps <= 0, or exog shape is incorrect.

        Notes
        -----
        Prediction intervals are computed using the standard errors from the 
        Kalman filter and assuming normally distributed innovations. The intervals 
        fully account for both parameter uncertainty (through the variance-covariance 
        matrix) and forecast uncertainty.
        """
        check_is_fitted(self, "model_")
        
        if not isinstance(steps, (int, np.integer)) or steps <= 0:
            raise ValueError("`steps` must be a positive integer.")
        
        # Handle level and alpha parameters
        if level is not None and alpha is not None:
            raise ValueError(
                "Cannot specify both `level` and `alpha`. Use one or the other."
            )
        
        if alpha is not None:
            if not 0 < alpha < 1:
                raise ValueError("`alpha` must be between 0 and 1.")
            level = [(1 - alpha) * 100]
        elif level is None:
            level = (80, 95)
        
        # Ensure level is iterable
        if isinstance(level, (int, float, np.number)):
            level = [level]
        else:
            level = list(level)
        
        # Convert exog to numpy array if needed
        if exog is not None:
            exog = np.asarray(exog, dtype=float)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            elif exog.ndim != 2:
                raise ValueError("`exog` must be 1- or 2-dimensional.")
            
            if len(exog) != steps:
                raise ValueError(
                    f"Length of `exog` ({len(exog)}) must match `steps` ({steps})."
                )
            
            if exog.shape[1] != self.n_exog_features_in_:
                raise ValueError(
                    f"Number of exogenous features ({exog.shape[1]}) does not match "
                    f"the number used during fitting ({self.n_exog_features_in_})."
                )
        elif self.n_exog_features_in_ > 0:
            raise ValueError(
                f"Model was fitted with {self.n_exog_features_in_} exogenous features, "
                f"but `exog` was not provided for prediction."
            )
        
        # Generate predictions with standard errors
        result = predict_arima(
            model   = self.model_,
            n_ahead = steps,
            newxreg = exog,
            se_fit  = True
        )
        
        mean = np.asarray(result['pred'])
        se = np.asarray(result['se'])
        levels = list(level)
        n_levels = len(levels)

        # Preallocate and compute intervals
        lower = np.empty((steps, n_levels), dtype=float)
        upper = np.empty((steps, n_levels), dtype=float)
        for i, lv in enumerate(levels):
            alpha_lvl = 1 - lv / 100
            z = norm.ppf(1 - alpha_lvl / 2)
            lower[:, i] = mean - z * se
            upper[:, i] = mean + z * se

        results = np.empty((steps, 1 + 2 * n_levels), dtype=float)
        results[:, 0] = mean
        for i in range(n_levels):
            results[:, 1 + 2 * i] = lower[:, i]
            results[:, 1 + 2 * i + 1] = upper[:, i]


        if as_frame:
            idx = pd.RangeIndex(1, steps + 1, name="step")
            col_names = ["mean"]
            for level in levels:
                col_names.append(f"lower_{int(level)}")
                col_names.append(f"upper_{int(level)}")
            results = pd.DataFrame(results, index=idx, columns=col_names)

        return results

    def get_residuals(self) -> np.ndarray:
        """
        Get in-sample residuals (observed - fitted) from the ARIMA model.

        Returns
        -------
        residuals : ndarray of shape (n_samples,)
            In-sample residuals.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        RuntimeError
            If reduce_memory() has been called (residuals are no longer available).
        """
        check_is_fitted(self, "model_")
        check_memory_reduced(self, 'get_residuals')
        return self.in_sample_residuals_

    def get_fitted_values(self) -> np.ndarray:
        """
        Get in-sample fitted values from the ARIMA model.

        Returns
        -------
        fitted : ndarray of shape (n_samples,)
            In-sample fitted values.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        RuntimeError
            If reduce_memory() has been called (fitted values are no longer available).
        """
        check_is_fitted(self, "model_")
        check_memory_reduced(self, 'get_fitted_values')
        return self.fitted_values_

    def summary(self) -> None:
        """
        Print a summary of the fitted ARIMA model.

        Includes model specification, coefficients, fit statistics, and residual diagnostics.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        RuntimeError
            If reduce_memory() has been called (summary information is no longer available).
        """
        check_is_fitted(self, "model_")
        check_memory_reduced(self, 'summary')
                
        print("ARIMA Model Summary")
        print("=" * 60)
        print(f"Model: {self.estimator_id}")
        print(f"Method: {self.model_['method']}")
        print(f"Number of observations: {len(self.y_train_)}")
        print(f"Converged: {self.converged_}")
        print()
        
        print("Coefficients:")
        print("-" * 60)
        for i, name in enumerate(self.coef_names_):
            # Extract standard error from variance-covariance matrix
            if self.var_coef_ is not None and i < len(self.var_coef_):
                se = np.sqrt(self.var_coef_[i, i])
                t_stat = self.coef_[i] / se if se > 0 else np.nan
                print(f"  {name:15s}: {self.coef_[i]:10.4f}  (SE: {se:8.4f}, t: {t_stat:8.2f})")
            else:
                print(f"  {name:15s}: {self.coef_[i]:10.4f}")
        print()
        
        print("Model fit statistics:")
        print(f"  sigma^2:             {self.sigma2_:.6f}")
        print(f"  Log-likelihood:      {self.loglik_:.2f}")
        print(f"  AIC:                 {self.aic_:.2f}")
        if self.bic_ is not None:
            print(f"  BIC:                 {self.bic_:.2f}")
        else:
            print(f"  BIC:                 N/A")
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

    def params(self) -> np.ndarray:
        """
        Get the parameters of the fitted model.
        
        The order of variables is the AR coefficients, MA coefficients, 
        seasonal AR coefficients, seasonal MA coefficients, exogenous 
        coefficients (if any), and intercept (if included).

        Returns
        -------
        params : ndarray
            The parameters of the model.
        
        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        """
        check_is_fitted(self, "model_")
        return self.coef_

    def get_score(self, y: None = None) -> float:
        """
        Compute R^2 score using in-sample fitted values.

        Parameters
        ----------
        y : ignored
            Present for API compatibility with sklearn.

        Returns
        -------
        score : float
            Coefficient of determination (R^2).

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        RuntimeError
            If reduce_memory() has been called (score cannot be computed).
        """
        check_is_fitted(self, "model_")
        check_memory_reduced(self, 'get_score')
        
        y = self.y_train_
        fitted = self.fitted_values_
        
        # Handle NaN values if any
        mask = ~(np.isnan(y) | np.isnan(fitted))
        if mask.sum() < 2:
            return np.nan
        
        ss_res = np.sum((y[mask] - fitted[mask]) ** 2)
        ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2) + np.finfo(float).eps
        return 1.0 - ss_res / ss_tot

    def get_info_criteria(self, criteria: str = 'aic') -> float:
        """
        Get the selected information criterion.

        Parameters
        ----------
        criteria : str, default='aic'
            The information criterion to retrieve. Valid options are 
            {'aic', 'bic'}.

        Returns
        -------
        metric : float
            The value of the selected information criterion.
        
        Raises
        ------
        ValueError
            If an invalid criteria is specified.
        NotFittedError
            If the model has not been fitted.
        """
        check_is_fitted(self, "model_")
        
        if criteria not in ['aic', 'bic']:
            raise ValueError(
                f"Invalid value for `criteria`: '{criteria}'. "
                f"Valid options are 'aic' and 'bic'."
            )
        
        if criteria == 'aic':
            return self.aic_
        elif criteria == 'bic':
            if self.bic_ is None:
                raise ValueError(
                    "BIC is not available for this model. This may occur when "
                    "the model did not converge or other estimation issues."
                )
            return self.bic_

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
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "m": self.m,
            "include_mean": self.include_mean,
            "transform_pars": self.transform_pars,
            "method": self.method,
            "n_cond": self.n_cond,
            "SSinit": self.SSinit,
            "optim_method": self.optim_method,
            "optim_control": self.optim_control,
            "kappa": self.kappa,
        }

    def set_params(self, **params) -> "Arima":
        """
        Set the parameters of this estimator and reset the fitted state.
        
        This method resets the estimator to its unfitted state whenever parameters
        are changed, requiring the model to be refitted before making predictions.

        Parameters
        ----------
        **params : dict
            Estimator parameters. Valid parameter keys are: 'order', 'seasonal_order',
            'm', 'include_mean', 'transform_pars', 'method', 'n_cond', 'SSinit',
            'optim_method', 'optim_control', 'kappa'.

        Returns
        -------
        self : Arima
            The estimator with updated parameters and reset state.
            
        Raises
        ------
        ValueError
            If any parameter key is invalid.
        """
        # Validate parameter keys
        valid_params = {
            'order', 'seasonal_order', 'm', 'include_mean', 'transform_pars',
            'method', 'n_cond', 'SSinit', 'optim_method', 'optim_control', 'kappa'
        }
        for key in params.keys():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{key}'. Valid parameters are: {valid_params}"
                )
        
        # Set the parameters
        for key, value in params.items():
            setattr(self, key, value)
        
        # Reset fitted state - remove fitted attributes if they exist
        fitted_attrs = [
            'model_', 'y_train_', 'coef_', 'coef_names_', 'sigma2_', 'loglik_',
            'aic_', 'bic_', 'arma_', 'converged_', 'fitted_values_',
            'in_sample_residuals_', 'var_coef_', 'n_features_in_', 'n_exog_features_in_'
        ]
        for attr in fitted_attrs:
            if hasattr(self, attr):
                delattr(self, attr)
        
        self.is_memory_reduced = False
        
        return self

    def reduce_memory(self) -> "Arima":
        """
        Free memory by deleting large attributes after fitting.

        This method removes fitted values, residuals, and other intermediate 
        results that are not strictly necessary for prediction. After calling 
        this method, certain diagnostic functions (like get_residuals(), 
        get_fitted_values(), summary()) will no longer work, but prediction 
        methods will continue to function.

        Call this method only if you need to reduce memory usage and don't 
        need access to diagnostic information.

        Returns
        -------
        self : Arima
            The estimator with reduced memory footprint.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.

        Warnings
        --------
        This operation is irreversible. You will need to refit the model to 
        access the removed attributes.
        """
        check_is_fitted(self, "model_")
        
        if self.is_memory_reduced:
            warnings.warn(
                "Memory has already been reduced. No further reduction possible.",
                UserWarning
            )
            return self
        
        # Delete large memory-consuming attributes, but keep model_ for predictions
        attrs_to_delete = [
            'y_train_', 'fitted_values_', 'in_sample_residuals_', 'var_coef_'
        ]
        
        for attr in attrs_to_delete:
            if hasattr(self, attr):
                delattr(self, attr)
        
        self.is_memory_reduced = True
        
        warnings.warn(
            "Memory reduced. Diagnostic methods (get_residuals, get_fitted_values, "
            "summary, get_score) are no longer available. Prediction methods remain functional.",
            UserWarning
        )
        
        return self
