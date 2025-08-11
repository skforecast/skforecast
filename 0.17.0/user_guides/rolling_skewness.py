# Custom class to create rolling skewness features
# ==============================================================================
import numpy as np
import pandas as pd
from scipy.stats import skew


class RollingSkewness():
    """
    Custom class to create rolling skewness features.
    """

    def __init__(self, window_sizes, features_names: list = 'rolling_skewness'):
        
        if not isinstance(window_sizes, list):
            window_sizes = [window_sizes]
        self.window_sizes = window_sizes
        self.features_names = features_names

    def transform_batch(self, X: pd.Series) -> pd.DataFrame:
        
        rolling_obj = X.rolling(window=self.window_sizes[0], center=False, closed='left')
        rolling_skewness = rolling_obj.skew()
        rolling_skewness = pd.DataFrame({
                               self.features_names: rolling_skewness
                           }).dropna()

        return rolling_skewness

    def transform(self, X: np.ndarray) -> np.ndarray:
        
        X = X[~np.isnan(X)]
        if len(X) > 0:
            rolling_skewness = np.array([skew(X, bias=False)])
        else:
            rolling_skewness = np.nan
        
        return rolling_skewness


class RollingSkewnessMultiSeries():
    """
    Custom class to create rolling skewness features for multiple series.
    """

    def __init__(self, window_sizes, features_names: list = 'rolling_skewness'):
        
        if not isinstance(window_sizes, list):
            window_sizes = [window_sizes]
        self.window_sizes = window_sizes
        self.features_names = features_names

    def transform_batch(self, X: pd.Series) -> pd.DataFrame:
        
        rolling_obj = X.rolling(window=self.window_sizes[0], center=False, closed='left')
        rolling_skewness = rolling_obj.skew()
        rolling_skewness = pd.DataFrame({
                               self.features_names: rolling_skewness
                           }).dropna()

        return rolling_skewness

    def transform(self, X: np.ndarray) -> np.ndarray:
        
        X_dim = X.ndim
        if X_dim == 1:
            n_series = 1  # Only one series
            X = X.reshape(-1, 1)
        else:
            n_series = X.shape[1]  # Series (levels) to be predicted (present in last_window)
        
        n_stats = 1  # Only skewness is calculated
        rolling_skewness = np.full(
            shape=(n_series, n_stats), fill_value=np.nan, dtype=float
        )
        for i in range(n_series):
            if len(X) > 0:
                rolling_skewness[i, :] = skew(X[:, i], bias=False)
            else:
                rolling_skewness[i, :] = np.nan      

        if X_dim == 1:
            rolling_skewness = rolling_skewness.flatten()  
        
        return rolling_skewness
