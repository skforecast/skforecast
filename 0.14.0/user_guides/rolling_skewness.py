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
