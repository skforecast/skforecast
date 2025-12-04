"""
ARIMA Model Implementation from Scratch
Optimized with Numba JIT compilation for maximum performance.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from numba import jit
import warnings


class ARIMA:
    """
    AutoRegressive Integrated Moving Average (ARIMA) model.
    
    This implementation uses Conditional Least Squares (CLS) for parameter estimation
    and is optimized with Numba JIT compilation for computational efficiency.
    
    Parameters
    ----------
    order : tuple of int, default=(1, 0, 0)
        The (p, d, q) order of the model:
        - p: AR order (autoregressive)
        - d: Differencing order (integration)
        - q: MA order (moving average)
    
    Attributes
    ----------
    coef_ : ndarray
        Fitted coefficients [AR coefficients, MA coefficients, intercept]
    ar_coef_ : ndarray
        Autoregressive coefficients
    ma_coef_ : ndarray
        Moving average coefficients
    exog_coef_ : ndarray
        Exogenous variable coefficients (estimated in closed form)
    intercept_ : float
        Intercept term
    sigma2_ : float
        Residual variance
    residuals_ : ndarray
        Model residuals from differenced series
    y_diff_ : ndarray
        Differenced series used for fitting
    diff_initial_values_ : list of float
        Initial values needed for inverse differencing (memory efficient)
    is_fitted_ : bool
        Whether the model has been fitted
    """
    
    def __init__(self, order=(1, 0, 0)):
        """
        Initialize ARIMA model.
        
        Parameters
        ----------
        order : tuple of int, default=(1, 0, 0)
            The (p, d, q) order of the model
        """
        self.order = order
        self.p, self.d, self.q = order
        
        # Validate order
        if self.p < 0 or self.d < 0 or self.q < 0:
            raise ValueError("Order parameters must be non-negative integers")
        if self.p == 0 and self.q == 0:
            raise ValueError("At least one of p or q must be greater than 0")
        
        # Model parameters (fitted)
        self.coef_ = None
        self.ar_coef_ = None
        self.ma_coef_ = None
        self.exog_coef_ = None
        self.intercept_ = None
        self.sigma2_ = None
        self.residuals_ = None
        self.y_diff_ = None
        self.exog_trimmed_ = None  # Exog trimmed to match differenced y (NOT differenced)
        self.diff_initial_values_ = None
        self.is_fitted_ = False
        
    def fit(self, y, exog=None):
        """
        Fit ARIMA model to training data using Conditional Least Squares.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Training time series data
        exog : array-like of shape (n_samples, n_features), optional
            Exogenous variables. NOT differenced - trimmed to match differenced y.
            Beta coefficients estimated in closed form during optimization.
            
        Returns
        -------
        self : ARIMA
            Fitted estimator
        """
        # Convert to numpy array and validate
        y = np.asarray(y, dtype=np.float64).flatten()
        
        if len(y) < max(self.p, self.q) + self.d + 1:
            raise ValueError(
                f"Time series too short. Need at least {max(self.p, self.q) + self.d + 1} observations"
            )
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("Input contains NaN or infinite values")
        
        # Process exogenous variables
        if exog is not None:
            exog = np.asarray(exog, dtype=np.float64)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            if len(exog) != len(y):
                raise ValueError(f"exog length {len(exog)} does not match y length {len(y)}")
            if np.any(np.isnan(exog)) or np.any(np.isinf(exog)):
                raise ValueError("exog contains NaN or infinite values")
        
        # Apply differencing and store only necessary initial values
        self.y_diff_, self.diff_initial_values_ = self._difference_with_initial(y, self.d)
        
        # Trim exog to match differenced y length (don't difference it)
        if exog is not None:
            self.exog_trimmed_ = exog[self.d:]
        else:
            self.exog_trimmed_ = None
        
        # Estimate parameters using Conditional Least Squares
        self._fit_cls()
        
        self.is_fitted_ = True
        return self
    
    def predict(self, steps=1, exog=None):
        """
        Generate forecasts for future time steps.
        
        Parameters
        ----------
        steps : int, default=1
            Number of steps ahead to forecast
        exog : array-like of shape (steps, n_features), optional
            Future exogenous variables (in original form, not differenced).
            Required if model was fit with exog.
            
        Returns
        -------
        forecasts : ndarray of shape (steps,)
            Forecasted values
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling predict")
        
        if steps < 1:
            raise ValueError("steps must be at least 1")
        
        # Validate exog
        if self.exog_trimmed_ is not None and exog is None:
            raise ValueError("Model was fitted with exog, must provide exog for prediction")
        if self.exog_trimmed_ is None and exog is not None:
            raise ValueError("Model was fitted without exog, cannot use exog for prediction")
        
        # Process future exog
        exog_future = None
        if exog is not None:
            exog = np.asarray(exog, dtype=np.float64)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            if len(exog) != steps:
                raise ValueError(f"exog length {len(exog)} does not match steps {steps}")
            if exog.shape[1] != self.exog_trimmed_.shape[1]:
                raise ValueError(f"exog has {exog.shape[1]} features, expected {self.exog_trimmed_.shape[1]}")
            exog_future = exog
        
        # Generate forecasts on differenced scale
        forecasts_diff = self._forecast_diff(steps, exog_future)
        
        # Inverse differencing to get original scale
        forecasts = self._inverse_difference(forecasts_diff, self.diff_initial_values_, self.d)
        
        return forecasts
    
    def predict_interval(self, steps=1, alpha=0.05):
        """
        Generate forecasts with prediction intervals.
        
        Uses approximate prediction intervals based on the residual variance
        from Conditional Least Squares estimation. The intervals account for
        forecast error variance that grows with the forecast horizon.
        
        Parameters
        ----------
        steps : int, default=1
            Number of steps ahead to forecast
        alpha : float, default=0.05
            Significance level for prediction intervals.
            Default 0.05 gives 95% prediction intervals.
            
        Returns
        -------
        forecasts : ndarray of shape (steps,)
            Point forecasts
        lower : ndarray of shape (steps,)
            Lower bounds of prediction intervals
        upper : ndarray of shape (steps,)
            Upper bounds of prediction intervals
            
        Notes
        -----
        The prediction intervals are approximate and based on:
        1. Residual variance from CLS estimation
        2. Forecast error variance that accumulates over time
        3. Normal distribution assumption for forecast errors
        
        For exact intervals, Maximum Likelihood Estimation would be required.
        However, these approximate intervals are typically very close to exact
        intervals for practical purposes.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling predict_interval")
        
        if steps < 1:
            raise ValueError("steps must be at least 1")
        
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be between 0 and 1")
        
        # Generate point forecasts
        forecasts = self.predict(steps)
        
        # Compute forecast error standard deviations on differenced scale
        forecast_std_diff = self._compute_forecast_std(steps)
        
        # Critical value from standard normal distribution
        z_critical = norm.ppf(1 - alpha / 2)
        
        # Compute intervals on differenced scale
        forecasts_diff = self._forecast_diff(steps)
        lower_diff = forecasts_diff - z_critical * forecast_std_diff
        upper_diff = forecasts_diff + z_critical * forecast_std_diff
        
        # Apply inverse differencing to get intervals in original scale
        lower = self._inverse_difference(lower_diff, self.diff_initial_values_, self.d)
        upper = self._inverse_difference(upper_diff, self.diff_initial_values_, self.d)
        
        return forecasts, lower, upper
    
    def _difference_with_initial(self, y, d):
        """
        Apply differencing transformation and store initial values for inversion.
        
        This method is memory-efficient: instead of storing the entire original series,
        it only stores the last value at each differencing level (d values total).
        
        Parameters
        ----------
        y : ndarray
            Time series to difference
        d : int
            Order of differencing
            
        Returns
        -------
        y_diff : ndarray
            Differenced series
        initial_values : list of float
            Last value at each differencing level [level_0, level_1, ..., level_{d-1}]
        """
        if d == 0:
            return y, []
        
        initial_values = []
        y_diff = y.copy()
        
        # Apply differencing d times, storing the last value at each level
        for _ in range(d):
            initial_values.append(y_diff[-1])
            y_diff = np.diff(y_diff)
        
        return y_diff, initial_values
    
    def _inverse_difference(self, y_diff, initial_values, d):
        """
        Inverse differencing to recover original scale.
        
        Uses only the stored initial values (last value at each level) instead of
        the full original series. This is memory-efficient for large datasets.
        
        Mathematical explanation:
        - For d=1: y_t = y_{t-1} + diff_t, so we need the last original value
        - For d=2: Apply integration twice, each time using the last value from the previous level
        
        Parameters
        ----------
        y_diff : ndarray
            Differenced forecasts
        initial_values : list of float
            Last values at each differencing level [level_0, level_1, ..., level_{d-1}]
        d : int
            Order of differencing
            
        Returns
        -------
        y : ndarray
            Forecasts in original scale
        """
        if d == 0:
            return y_diff
        
        # Integrate back from level d to level 0
        y = y_diff.copy()
        for level in range(d):
            # Get the last value from level (d - level - 1)
            # initial_values[i] contains the last value from differencing level i
            last_val = initial_values[d - level - 1]
            # Cumulative sum and add the last value
            y = last_val + np.cumsum(y)
        
        return y
    
    def _fit_cls(self):
        """
        Fit model using Conditional Least Squares (CLS).
        
        This method estimates AR and MA parameters by minimizing
        the sum of squared residuals.
        For ARIMAX: Only AR/MA parameters are optimized.
        Beta coefficients are estimated in closed form (OLS) during each evaluation.
        """
        y = self.y_diff_
        n = len(y)
        
        # Remove mean for centered estimation
        y_mean = np.mean(y)
        y_centered = y - y_mean
        
        # Initial parameter guess (only AR and MA, not beta)
        n_params = self.p + self.q
        initial_params = np.zeros(n_params)
        
        # Initialize AR parameters with OLS estimates if possible
        if self.p > 0:
            initial_params[:self.p] = self._estimate_ar_ols(y_centered)
        
        # Optimize using scipy (only AR/MA parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # result = minimize(
            #     self._cls_objective,
            #     initial_params,
            #     args=(y_centered,),
            #     method='L-BFGS-B',
            #     bounds=[(-0.99, 0.99)] * self.p + [(-0.99, 0.99)] * self.q,
            #     options={'maxiter': 1000}
            # )
            result = minimize(
                self._cls_val_and_grad,
                initial_params,
                args=(y_centered, self.exog_trimmed_),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.99, 0.99)] * self.p + [(-0.99, 0.99)] * self.q,
                options={'maxiter': 1000}
            )
        
        params = result.x
        
        # Extract AR/MA coefficients
        self.ar_coef_ = params[:self.p] if self.p > 0 else np.array([])
        self.ma_coef_ = params[self.p:] if self.q > 0 else np.array([])
        
        # Compute residuals to get beta in closed form
        self.residuals_ = self._compute_residuals(y_centered, params, self.exog_trimmed_)
        
        # Beta was computed during residual calculation, extract it
        if self.exog_trimmed_ is not None:
            # Re-estimate beta one final time with optimal AR/MA
            self.exog_coef_ = self._estimate_beta_closed_form(y_centered, params, self.exog_trimmed_)
        else:
            self.exog_coef_ = np.array([])
        
        # For differenced models (d >= 1), do not include drift/intercept
        if self.d == 0:
            self.intercept_ = y_mean
        else:
            self.intercept_ = 0.0
        
        # Store all coefficients
        self.coef_ = np.concatenate([self.ar_coef_, self.ma_coef_, self.exog_coef_, [self.intercept_]])
        
        # Compute final residuals and variance
        self.residuals_ = self._compute_residuals(y_centered, params, self.exog_trimmed_)
        self.sigma2_ = np.var(self.residuals_)

    def _cls_val_and_grad(self, params, y, exog=None):
        """
        Helper to return both objective value and gradient for scipy.minimize.
        
        For ARIMAX: estimates beta in closed form given AR/MA params,
        then computes SSE and gradients w.r.t. AR/MA only.
        """
        ar_coef = params[:self.p] if self.p > 0 else np.array([])
        ma_coef = params[self.p:] if self.q > 0 else np.array([])
        
        if exog is not None:
            # Estimate beta in closed form given current AR/MA
            beta = self._estimate_beta_closed_form(y, params, exog)
            return _compute_objective_and_gradient_jit_exog_profile(y, exog, ar_coef, ma_coef, beta, self.p, self.q)
        else:
            return _compute_objective_and_gradient_jit(y, ar_coef, ma_coef, self.p, self.q)
    
    def _estimate_beta_closed_form(self, y, ar_ma_params, exog):
        """
        Estimate exog coefficients (beta) in closed form via OLS.
        
        Given AR/MA parameters, compute ARIMA residuals, then regress
        y - ARIMA_fit on exog to get beta.
        
        Parameters
        ----------
        y : ndarray
            Centered time series
        ar_ma_params : ndarray
            Current AR and MA parameters
        exog : ndarray
            Centered exogenous variables
            
        Returns
        -------
        beta : ndarray
            OLS estimates of exog coefficients
        """
        ar_coef = ar_ma_params[:self.p] if self.p > 0 else np.array([])
        ma_coef = ar_ma_params[self.p:] if self.q > 0 else np.array([])
        
        # Compute what y would be without exog effect (ARIMA component only)
        n = len(y)
        start_idx = max(self.p, self.q)
        
        # Get ARIMA fitted values (without exog)
        y_arima = np.zeros(n)
        residuals_temp = np.zeros(n)
        
        for t in range(start_idx, n):
            ar_term = 0.0
            for i in range(self.p):
                ar_term += ar_coef[i] * y[t - i - 1]
            
            ma_term = 0.0
            for i in range(self.q):
                ma_term += ma_coef[i] * residuals_temp[t - i - 1]
            
            y_arima[t] = ar_term + ma_term
            residuals_temp[t] = y[t] - y_arima[t]
        
        # Residual from ARIMA: y - ARIMA_fit
        y_residual = y[start_idx:] - y_arima[start_idx:]
        X = exog[start_idx:]
        
        # OLS: beta = (X'X)^-1 X'y_residual
        try:
            beta = np.linalg.lstsq(X, y_residual, rcond=1e-10)[0]
        except:
            beta = np.zeros(X.shape[1])
        
        return beta
    
    def _estimate_ar_ols(self, y):
        """
        Estimate AR parameters using OLS (Ordinary Least Squares).
        
        Parameters
        ----------
        y : ndarray
            Centered time series
            
        Returns
        -------
        ar_coef : ndarray
            Initial AR coefficient estimates
        """
        if self.p == 0:
            return np.array([])
        
        n = len(y)
        # Create lagged design matrix
        X = np.zeros((n - self.p, self.p))
        for i in range(self.p):
            X[:, i] = y[self.p - i - 1:n - i - 1]
        
        y_target = y[self.p:]
        
        # OLS estimation with regularization for stability
        try:
            ar_coef = np.linalg.lstsq(X, y_target, rcond=1e-10)[0]
            # Clip to ensure stationarity
            ar_coef = np.clip(ar_coef, -0.9, 0.9)
        except:
            ar_coef = np.zeros(self.p)
        
        return ar_coef
    
    def _cls_objective(self, params, y):
        """
        Objective function for Conditional Least Squares optimization.
        
        Parameters
        ----------
        params : ndarray
            Parameters [AR coefficients, MA coefficients]
        y : ndarray
            Centered time series
            
        Returns
        -------
        sse : float
            Sum of squared errors
        """
        residuals = self._compute_residuals(y, params)
        return np.sum(residuals ** 2)
    
    def _compute_residuals(self, y, params, exog=None):
        """
        Compute residuals using numba-optimized function.
        
        Parameters
        ----------
        y : ndarray
            Centered time series
        params : ndarray
            Parameters [AR coefficients, MA coefficients]
        exog : ndarray, optional
            Centered exogenous variables
            
        Returns
        -------
        residuals : ndarray
            Model residuals
        """
        ar_coef = params[:self.p] if self.p > 0 else np.array([])
        ma_coef = params[self.p:] if self.q > 0 else np.array([])
        
        if exog is not None:
            beta = self._estimate_beta_closed_form(y, params, exog)
            return _compute_residuals_jit_exog(y, exog, ar_coef, ma_coef, beta, self.p, self.q)
        else:
            return _compute_residuals_jit(y, ar_coef, ma_coef, self.p, self.q)
    
    def _forecast_diff(self, steps, exog_future=None):
        """
        Generate forecasts on differenced scale.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        exog_future : ndarray, optional
            Future exogenous variables (not differenced)
            
        Returns
        -------
        forecasts : ndarray
            Forecasts on differenced scale
        """
        # Only pass the last p values for AR and last q residuals for MA
        n = len(self.y_diff_)
        y_centered = self.y_diff_ - self.intercept_
        
        # Extract only necessary historical values
        y_last = y_centered[-self.p:] if self.p > 0 else np.array([])
        residuals_last = self.residuals_[-self.q:] if self.q > 0 else np.array([])
        
        if exog_future is not None:
            # Use exog as-is (no centering needed)
            return _forecast_diff_jit_exog(
                y_last, residuals_last, exog_future, self.ar_coef_, self.ma_coef_,
                self.exog_coef_, self.p, self.q, steps, self.intercept_
            )
        else:
            return _forecast_diff_jit(
                y_last, residuals_last, self.ar_coef_, self.ma_coef_, 
                self.p, self.q, steps, self.intercept_
            )
    
    def _compute_forecast_std(self, steps):
        """
        Compute forecast error standard deviations for each step ahead.
        
        Uses the approximate formula for ARMA forecast error variance,
        which accounts for the propagation of uncertainty through the model.
        
        Parameters
        ----------
        steps : int
            Number of forecast steps
            
        Returns
        -------
        std : ndarray of shape (steps,)
            Standard deviation of forecast errors at each horizon
        """
        # Compute cumulative variance for each forecast horizon
        # For ARMA models, forecast variance grows with horizon
        forecast_var = _compute_forecast_variance_jit(
            self.ar_coef_, self.ma_coef_, self.p, self.q, steps, self.sigma2_
        )
        
        return np.sqrt(forecast_var)


# Numba-optimized functions for maximum performance

@jit(nopython=True, cache=True, fastmath=True)
def _compute_residuals_jit(y, ar_coef, ma_coef, p, q):
    """
    Compute residuals using Numba JIT compilation for speed.
    
    Uses conditional likelihood approach (conditions on first max(p, q) observations).
    
    Parameters
    ----------
    y : ndarray
        Time series (centered)
    ar_coef : ndarray
        AR coefficients
    ma_coef : ndarray
        MA coefficients
    p : int
        AR order
    q : int
        MA order
        
    Returns
    -------
    residuals : ndarray
        Model residuals
    """
    n = len(y)
    start_idx = max(p, q)
    residuals = np.zeros(n)
    
    # Initialize residuals for conditioning period
    for i in range(start_idx):
        residuals[i] = 0.0
    
    # Compute residuals iteratively
    for t in range(start_idx, n):
        # AR component
        ar_term = 0.0
        for i in range(p):
            ar_term += ar_coef[i] * y[t - i - 1]
        
        # MA component
        ma_term = 0.0
        for i in range(q):
            ma_term += ma_coef[i] * residuals[t - i - 1]
        
        # Residual at time t
        residuals[t] = y[t] - ar_term - ma_term
    
    return residuals


@jit(nopython=True, cache=True, fastmath=True)
def _forecast_diff_jit(y_last, residuals_last, ar_coef, ma_coef, p, q, steps, intercept):
    """
    Generate forecasts using Numba JIT compilation.
    
    Memory-efficient implementation: only uses the last p values and last q residuals
    instead of the full historical arrays.
    
    Parameters
    ----------
    y_last : ndarray
        Last p values of historical time series (centered), shape (p,)
    residuals_last : ndarray
        Last q historical residuals, shape (q,)
    ar_coef : ndarray
        AR coefficients
    ma_coef : ndarray
        MA coefficients
    p : int
        AR order
    q : int
        MA order
    steps : int
        Number of forecast steps
    intercept : float
        Intercept term
        
    Returns
    -------
    forecasts : ndarray
        Forecasted values
    """
    forecasts = np.zeros(steps)
    
    # Create buffers for rolling window of values and residuals
    # We only need to track the last max(p, q) values
    buffer_size = max(p, q)
    y_buffer = np.zeros(buffer_size + steps)
    residuals_buffer = np.zeros(buffer_size + steps)
    
    # Initialize buffers with historical values
    if p > 0:
        y_buffer[:p] = y_last
    if q > 0:
        residuals_buffer[:q] = residuals_last
    
    # Generate forecasts iteratively
    for h in range(steps):
        # AR component - weighted sum of previous values
        ar_term = 0.0
        for i in range(p):
            ar_term += ar_coef[i] * y_buffer[p - 1 - i + h]
        
        # MA component - weighted sum of previous residuals
        # Future residuals have expectation 0
        ma_term = 0.0
        for i in range(q):
            if h > i:
                # Use forecasted residual (which is 0)
                pass
            else:
                # Use historical residual
                ma_term += ma_coef[i] * residuals_buffer[q - 1 - i + h]
        
        # Forecast at step h
        forecast_centered = ar_term + ma_term
        
        # Store in buffer for next iteration
        if p > 0:
            y_buffer[p + h] = forecast_centered
        
        # Add back intercept
        forecasts[h] = forecast_centered + intercept
    
    return forecasts


@jit(nopython=True, cache=True, fastmath=True)
def _compute_forecast_variance_jit(ar_coef, ma_coef, p, q, steps, sigma2):
    """
    Compute forecast error variance for each forecast horizon.
    
    Uses the recursive formula for ARMA forecast error variance.
    The variance accumulates as the forecast horizon increases due to
    the propagation of uncertainty through the model.
    
    Parameters
    ----------
    ar_coef : ndarray
        AR coefficients
    ma_coef : ndarray
        MA coefficients
    p : int
        AR order
    q : int
        MA order
    steps : int
        Number of forecast steps
    sigma2 : float
        Residual variance
        
    Returns
    -------
    forecast_var : ndarray of shape (steps,)
        Forecast error variance at each horizon
    """
    forecast_var = np.zeros(steps)
    
    # For ARMA models, we compute psi weights (impulse response coefficients)
    # These represent how a shock propagates through the model over time
    max_lag = max(p, q) + steps
    psi = np.zeros(max_lag)
    psi[0] = 1.0
    
    # Compute psi weights recursively
    # psi_j = sum(phi_i * psi_{j-i}) + theta_j for j >= 1
    for j in range(1, max_lag):
        psi_j = 0.0
        
        # AR contribution
        for i in range(min(p, j)):
            psi_j += ar_coef[i] * psi[j - i - 1]
        
        # MA contribution (only for j <= q)
        if j <= q:
            psi_j += ma_coef[j - 1]
        
        psi[j] = psi_j
    
    # Compute cumulative variance: Var(forecast_h) = sigma2 * sum(psi_i^2 for i < h)
    for h in range(steps):
        var_h = 0.0
        for i in range(h + 1):
            var_h += psi[i] ** 2
        forecast_var[h] = sigma2 * var_h
    
    return forecast_var


def check_stationarity(y):
    """
    Simple check for stationarity using variance ratio test.
    
    Parameters
    ----------
    y : array-like
        Time series to check
        
    Returns
    -------
    is_stationary : bool
        True if series appears stationary
    """
    y = np.asarray(y)
    n = len(y)
    
    # Split into two halves and compare variances
    mid = n // 2
    var1 = np.var(y[:mid])
    var2 = np.var(y[mid:])
    
    # If variances are similar, likely stationary
    ratio = max(var1, var2) / (min(var1, var2) + 1e-10)
    
    return ratio < 3.0


@jit(nopython=True, cache=True, fastmath=True)
def _compute_objective_and_gradient_jit(y, ar_coef, ma_coef, p, q):
    """
    Compute SSE and Gradient simultaneously in a single pass.
    
    Derivatives are computed recursively:
    d_eps_t/d_phi = -y_{t-k} - sum(theta * d_eps_{t-j}/d_phi)
    d_eps_t/d_theta = -eps_{t-k} - sum(theta * d_eps_{t-j}/d_theta)
    """
    n = len(y)
    n_params = p + q
    
    # Initialize
    sse = 0.0
    total_grad = np.zeros(n_params)
    residuals = np.zeros(n)
    
    # Buffer for historical gradients of residuals: shape (q, n_params)
    # grad_buffer[j] stores d_eps_{t-1-j} / d_params
    # We only need history for MA terms because of the recursive definition
    grad_buffer = np.zeros((max(1, q), n_params))
    
    start_idx = max(p, q)
    
    # Initialize residuals (assume 0 for pre-sample)
    # (residuals array is already zero-init)

    for t in range(start_idx, n):
        # --- 1. Compute Residual ---
        ar_term = 0.0
        for i in range(p):
            ar_term += ar_coef[i] * y[t - i - 1]
            
        ma_term = 0.0
        for i in range(q):
            ma_term += ma_coef[i] * residuals[t - i - 1]
            
        residuals[t] = y[t] - ar_term - ma_term
        sse += residuals[t]**2
        
        # --- 2. Compute Gradient ---
        curr_grad = np.zeros(n_params)
        
        # Derivatives w.r.t AR params (phi_k)
        # d_eps_t/d_phi_k = -y_{t-k} - sum(theta_j * d_eps_{t-j}/d_phi_k)
        for k in range(p):
            val = -y[t - k - 1]
            # Recursive part (only if MA terms exist)
            for j in range(q):
                val -= ma_coef[j] * grad_buffer[j, k]
            curr_grad[k] = val
            
        # Derivatives w.r.t MA params (theta_k)
        # d_eps_t/d_theta_k = -eps_{t-k} - sum(theta_j * d_eps_{t-j}/d_theta_k)
        for k in range(q):
            val = -residuals[t - k - 1]
            # Recursive part
            for j in range(q):
                val -= ma_coef[j] * grad_buffer[j, p + k]
            curr_grad[p + k] = val
            
        # Accumulate total gradient: d(SSE)/d_param = sum(2 * eps_t * d_eps_t/d_param)
        for i in range(n_params):
            total_grad[i] += 2 * residuals[t] * curr_grad[i]
            
        # Update buffer (shift right to make room for current t)
        # grad_buffer[0] becomes the gradient at t-1
        if q > 0:
            for j in range(q - 1, 0, -1):
                grad_buffer[j] = grad_buffer[j-1]
            grad_buffer[0] = curr_grad

    return sse, total_grad


@jit(nopython=True, cache=True, fastmath=True)
def _compute_residuals_jit_exog(y, exog, ar_coef, ma_coef, exog_coef, p, q):
    """
    Compute residuals with exogenous variables.
    Beta (exog_coef) is pre-computed in closed form.
    """
    n = len(y)
    start_idx = max(p, q)
    residuals = np.zeros(n)
    
    for i in range(start_idx):
        residuals[i] = 0.0
    
    for t in range(start_idx, n):
        # Exog component
        exog_term = 0.0
        for k in range(len(exog_coef)):
            exog_term += exog_coef[k] * exog[t, k]
        
        # AR component
        ar_term = 0.0
        for i in range(p):
            ar_term += ar_coef[i] * y[t - i - 1]
        
        # MA component
        ma_term = 0.0
        for i in range(q):
            ma_term += ma_coef[i] * residuals[t - i - 1]
        
        residuals[t] = y[t] - exog_term - ar_term - ma_term
    
    return residuals


@jit(nopython=True, cache=True, fastmath=True)
def _compute_objective_and_gradient_jit_exog_profile(y, exog, ar_coef, ma_coef, exog_coef, p, q):
    """
    Compute SSE and gradient for ARIMAX with profile likelihood.
    
    Beta is estimated in closed form (passed in), gradients computed only for AR/MA.
    Gradient accounts for the implicit dependence of beta on AR/MA parameters.
    """
    n = len(y)
    n_params = p + q  # Only AR and MA parameters
    
    sse = 0.0
    total_grad = np.zeros(n_params)
    residuals = np.zeros(n)
    
    grad_buffer = np.zeros((max(1, q), n_params))
    start_idx = max(p, q)
    
    for t in range(start_idx, n):
        # --- 1. Compute Residual ---
        exog_term = 0.0
        for k in range(len(exog_coef)):
            exog_term += exog_coef[k] * exog[t, k]
        
        ar_term = 0.0
        for i in range(p):
            ar_term += ar_coef[i] * y[t - i - 1]
        
        ma_term = 0.0
        for i in range(q):
            ma_term += ma_coef[i] * residuals[t - i - 1]
        
        residuals[t] = y[t] - exog_term - ar_term - ma_term
        sse += residuals[t]**2
        
        # --- 2. Compute Gradient (only for AR/MA) ---
        curr_grad = np.zeros(n_params)
        
        # d_eps/d_phi (AR coefficients)
        for k in range(p):
            val = -y[t - k - 1]
            for j in range(q):
                val -= ma_coef[j] * grad_buffer[j, k]
            curr_grad[k] = val
        
        # d_eps/d_theta (MA coefficients)
        for k in range(q):
            val = -residuals[t - k - 1]
            for j in range(q):
                val -= ma_coef[j] * grad_buffer[j, p + k]
            curr_grad[p + k] = val
        
        # Accumulate gradient
        for i in range(n_params):
            total_grad[i] += 2 * residuals[t] * curr_grad[i]
        
        # Update buffer
        if q > 0:
            for j in range(q - 1, 0, -1):
                grad_buffer[j] = grad_buffer[j-1]
            grad_buffer[0] = curr_grad
    
    return sse, total_grad


@jit(nopython=True, cache=True, fastmath=True)
def _forecast_diff_jit_exog(y_last, residuals_last, exog_future, ar_coef, ma_coef, exog_coef, p, q, steps, intercept):
    """
    Generate forecasts with exogenous variables.
    """
    forecasts = np.zeros(steps)
    buffer_size = max(p, q)
    y_buffer = np.zeros(buffer_size + steps)
    residuals_buffer = np.zeros(buffer_size + steps)
    
    if p > 0:
        y_buffer[:p] = y_last
    if q > 0:
        residuals_buffer[:q] = residuals_last
    
    for h in range(steps):
        # Exog component
        exog_term = 0.0
        for k in range(len(exog_coef)):
            exog_term += exog_coef[k] * exog_future[h, k]
        
        # AR component
        ar_term = 0.0
        for i in range(p):
            ar_term += ar_coef[i] * y_buffer[p - 1 - i + h]
        
        # MA component
        ma_term = 0.0
        for i in range(q):
            if h > i:
                pass
            else:
                ma_term += ma_coef[i] * residuals_buffer[q - 1 - i + h]
        
        forecast_centered = exog_term + ar_term + ma_term
        
        if p > 0:
            y_buffer[p + h] = forecast_centered
        
        forecasts[h] = forecast_centered + intercept
    
    return forecasts
