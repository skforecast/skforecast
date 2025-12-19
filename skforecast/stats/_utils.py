################################################################################
#                             skforecast.stats._utils                          #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import numpy as np

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
        If estimator.memory_reduced_ is True.
    """
    if getattr(estimator, 'memory_reduced_', False):
                
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
