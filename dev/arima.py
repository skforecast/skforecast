import numpy as np
import numba as nb
from scipy.optimize import minimize

@nb.jit(nopython=True)
def _diff_series(y, d):
    """Apply differencing d times."""
    for _ in range(d):
        y = np.diff(y)
    return y

@nb.jit(nopython=True)
def _undiff_series(diffed, last_values, d):
    """Undifference the series for any d."""
    if d == 0:
        return diffed
    undiffed = diffed.copy()
    for _ in range(d):
        # Reverse the last differencing step
        temp = np.zeros(len(undiffed) + 1)
        temp[0] = last_values[-1]  # Start with the last original value
        for i in range(1, len(temp)):
            temp[i] = temp[i-1] + undiffed[i-1]
        undiffed = temp[1:]  # Shift to match length
        last_values = last_values[:-1] if len(last_values) > 1 else last_values  # Update for next level
    return undiffed

@nb.jit(nopython=True)
def _arima_log_likelihood(params, y, exog, p, q, k):
    """Compute negative log-likelihood for ARIMAX parameters."""
    phi = params[:p]
    theta = params[p:p+q]
    beta = params[p+q:p+q+k]
    sigma2 = params[-1]
    n = len(y)
    eps = np.zeros(n)
    ll = 0.0
    
    # Initialize residuals for t < max(p, q) to 0 (backcasting approximation)
    for t in range(max(p, q)):
        eps[t] = 0.0
    
    for t in range(max(p, q), n):
        ar_sum = 0.0
        for i in range(p):
            if t - i - 1 >= 0:
                ar_sum += phi[i] * y[t - i - 1]
        ma_sum = 0.0
        for i in range(q):
            if t - i - 1 >= 0:
                ma_sum += theta[i] * eps[t - i - 1]
        exog_sum = 0.0
        if k > 0:
            for i in range(k):
                exog_sum += beta[i] * exog[t, i]
        eps[t] = y[t] - ar_sum - ma_sum - exog_sum
        ll += 0.5 * np.log(2 * np.pi * sigma2) + eps[t]**2 / (2 * sigma2)
    
    return ll

def _fit_arima_params(y, exog, p, q, k):
    """Fit ARIMAX parameters using MLE."""
    def objective(params):
        return _arima_log_likelihood(params, y, exog, p, q, k)
    
    bounds = [(-1, 1)] * (p + q) + [(-np.inf, np.inf)] * k + [(1e-6, None)]
    res = minimize(objective, np.zeros(p + q + k + 1), bounds=bounds, method='L-BFGS-B')
    if not res.success:
        raise ValueError("Optimization failed: " + res.message)
    return res.x

@nb.jit(nopython=True)
def _predict_arima(y, exog, exog_future, phi, theta, beta, sigma2, n_steps, p, q, k, simulate=False):
    """Recursive prediction on differenced series with exog."""
    n = len(y)
    predictions = np.zeros(n_steps)
    eps = np.zeros(n + n_steps)
    
    # Compute initial residuals (same as before)
    for t in range(max(p, q), n):
        ar_sum = 0.0
        for i in range(p):
            ar_sum += phi[i] * y[t - i - 1]
        ma_sum = 0.0
        for i in range(q):
            ma_sum += theta[i] * eps[t - i - 1]
        exog_sum = 0.0
        if k > 0:
            for i in range(k):
                exog_sum += beta[i] * exog[t, i]
        eps[t] = y[t] - ar_sum - ma_sum - exog_sum
    
    # Predict
    for t in range(n_steps):
        ar_sum = 0.0
        for i in range(p):
            idx = n + t - i - 1
            if idx >= 0:
                ar_sum += phi[i] * (y[idx] if idx < n else predictions[t - i - 1])
        ma_sum = 0.0
        for i in range(q):
            idx = n + t - i - 1
            if idx >= 0:
                ma_sum += theta[i] * eps[idx]
        exog_sum = 0.0
        if k > 0:
            for i in range(k):
                exog_sum += beta[i] * exog_future[t, i]
        predictions[t] = ar_sum + ma_sum + exog_sum
        eps[n + t] = np.random.normal(0, np.sqrt(sigma2)) if simulate else 0.0  # Deterministic by default
    
    return predictions

class ARIMA:
    def __init__(self, p=1, d=0, q=0):
        self.p = p
        self.d = d
        self.q = q
        self.phi = None
        self.theta = None
        self.beta = None
        self.sigma2 = None
        self.y_diff = None
        self.last_values = None
        self.y_original = None
        self.exog = None
        self.k = 0  # Number of exogenous variables
    
    def fit(self, y, exog=None):
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if exog is not None:
            if not isinstance(exog, np.ndarray):
                exog = np.array(exog)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            self.k = exog.shape[1]
            if len(exog) != len(y):
                raise ValueError("Exog must have same length as y")
            self.exog = exog
        else:
            self.k = 0
            self.exog = np.zeros((len(y), 0))
        
        if len(y) < self.p + self.q + 1:
            raise ValueError("Time series too short for given p and q")
        
        self.y_original = y.copy()
        self.y_diff = _diff_series(y.copy(), self.d)
        if self.d > 0:
            self.last_values = y[-self.d:]
        params = _fit_arima_params(self.y_diff, self.exog, self.p, self.q, self.k)
        self.phi = params[:self.p]
        self.theta = params[self.p:self.p+self.q]
        self.beta = params[self.p+self.q:self.p+self.q+self.k]
        self.sigma2 = params[-1]
    
    def predict(self, n_steps, exog_future=None, simulate=False):
        if self.phi is None:
            raise ValueError("Model not fitted")
        if self.k > 0 and exog_future is None:
            raise ValueError("Exog_future required for prediction with exogenous variables")
        if exog_future is not None:
            if not isinstance(exog_future, np.ndarray):
                exog_future = np.array(exog_future)
            if exog_future.ndim == 1:
                exog_future = exog_future.reshape(-1, 1)
            if exog_future.shape[1] != self.k:
                raise ValueError("Exog_future must have same number of features as fit exog")
            if exog_future.shape[0] != n_steps:
                raise ValueError("Exog_future must have n_steps rows")
        else:
            exog_future = np.zeros((n_steps, self.k))
        
        predictions_diff = _predict_arima(self.y_diff, self.exog, exog_future, self.phi, self.theta, self.beta, self.sigma2, n_steps, self.p, self.q, self.k, simulate)
        if self.d == 0:
            return predictions_diff
        else:
            return _undiff_series(predictions_diff, self.last_values, self.d)
    
    def summary(self):
        """Print model summary."""
        print(f"ARIMAX({self.p},{self.d},{self.q}) with {self.k} exogenous variables")
        print(f"AR coefficients: {self.phi}")
        print(f"MA coefficients: {self.theta}")
        print(f"Exog coefficients: {self.beta}")
        print(f"Sigma^2: {self.sigma2}")