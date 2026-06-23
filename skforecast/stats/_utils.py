################################################################################
#                             skforecast.stats._utils                          #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import warnings
import numpy as np
from sklearn.exceptions import NotFittedError


def _normalize_level(
    level: float | int | list[float] | tuple[float, ...]
) -> list[float]:
    """
    Normalize confidence level(s) to the 0-1 coverage scale.

    Detection rules, applied to the level value(s):

    - All values in `(0, 1]`: already coverage proportions, returned unchanged.
    - All values in `(1, 100]`: legacy percentiles. They are divided by 100 and
    a `FutureWarning` is emitted.
    - Mixed (some value `<= 1` and some value `> 1`): the scale is ambiguous and
    a `ValueError` is raised.

    Parameters
    ----------
    level : float, int, list, tuple
        Confidence level(s), either as coverage proportions (0-1) or as legacy
        percentiles (0-100).

    Returns
    -------
    level : list
        Confidence level(s) expressed as coverage proportions in the 0-1 scale.

    """

    if isinstance(level, (int, float, np.number)):
        values = [float(level)]
    else:
        values = [float(v) for v in level]

    any_above_one = any(v > 1 for v in values)
    any_le_one = any(v <= 1 for v in values)

    if any_above_one and any_le_one:
        raise ValueError(
            "`level` mixes values <= 1 and > 1, so the scale is ambiguous. Use "
            "coverage proportions in the (0, 1] range, e.g. `level=[0.8, 0.95]`."
        )

    if any_above_one:
        warnings.warn(
            "Passing `level` as percentiles (0-100) is deprecated. Use coverage "
            "proportions (0-1) instead. For example, use `level=[0.8, 0.95]` "
            "instead of `level=[80, 95]`. Percentile support will be removed in "
            "skforecast 0.24.0.",
            FutureWarning
        )
        return [v / 100 for v in values]

    return values


def check_is_fitted(func):
    """
    This decorator checks if the model is fitted before using the desired method.

    Parameters
    ----------
    func : Callable
        Function to wrap.
    
    Returns
    -------
    wrapper : wrapper
        Function wrapped.

    """

    def wrapper(self, *args, **kwargs):

        if not self.is_fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. Call "
                f"'fit' with appropriate arguments before using this estimator."
            )
        
        result = func(self, *args, **kwargs)
        
        return result
    
    return wrapper


def check_memory_reduced(estimator: object, method_name: str) -> None:
    """
    Check if estimator memory has been reduced and raise informative error.
    
    Parameters
    ----------
    estimator : object
        Estimator instance to check.
    method_name : str
        Name of the method being called (for error message).
        
    Raises
    ------
    ValueError
        If estimator.is_memory_reduced is True.
    
    """

    if getattr(estimator, 'is_memory_reduced', False):
                
        message = (
            f"Cannot call {method_name}(): model memory has been reduced via "
            f"reduce_memory() to reduce memory usage. "
            f"Refit the model to restore full functionality."
        )
        raise ValueError(message)


class FastLinearRegression:
    """
    Fast linear regression with using numpy linalg.solve as primary method and
    numpy lstsq as fallback method in case of multicollinearity. This class is
    designed to be a lightweight alternative to sklearn's LinearRegression.
    
    Attributes
    ----------
    intercept_ : float
        The intercept term
    coef_ : np.ndarray
        The coefficient array
    """
    
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None
    
    def fit(self, X, y):
        """
        Fit the linear regression model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray
            Target values of shape (n_samples,)
            
        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        try:
            # Try fastest method: closed-form solution
            XtX = X_with_intercept.T @ X_with_intercept
            coefficients = np.linalg.solve(XtX, X_with_intercept.T @ y)
            
        except np.linalg.LinAlgError:
            # Fallback to lstsq (handles rank-deficient matrices)
            coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        
        self.intercept_ = coefficients[0]
        self.coef_ = coefficients[1:]
        
        return self
    
    def predict(self, X):
        """
        Predict using the linear model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
            
        Returns
        -------
        y_pred : np.ndarray
            Predicted values of shape (n_samples,)
        """
        if self.intercept_ is None or self.coef_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_
