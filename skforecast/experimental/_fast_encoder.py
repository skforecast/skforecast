################################################################################
#                               experimental                                   #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import pandas as pd
import numpy as np

class FastOrdinalEncoder:  # pragma: no cover
    """
    Encode categorical values as an integer array, with integer values
    from 0 to n_categories - 1.

    This encoder mimics the behavior of sklearn's OrdinalEncoder but during the
    fit, categories are not learned from the data. Instead, the user must provide
    a list of unique categories. This is useful when the categories are known
    beforehand and the data is large.

    Parameters
    ----------

    Attributes
    ----------
    categories_ : np.ndarray
        Unique categories in the data.
    category_map_ : dict
        Mapping of categories to integers.
    inverse_category_map_ : dict
        Mapping of integers to categories.
    unknown_value : int | float, default=-1
        Value to use for unknown categories.
    
    """

    def __init__(self, unknown_value: int | float = -1):

        self.unknown_value = unknown_value
        self.categories_ = None
        self.category_map_ = None
        self.inverse_category_map_ = None
        
    def fit(self, categories: list | np.ndarray) -> None:
        """
        Fit the encoder using the provided categories.

        Parameters
        ----------
        categories : list | np.ndarray
            Unique categories used to fit the encoder.
        """

        if not isinstance(categories, (list, np.ndarray)):
            raise ValueError("Categories must be a list or numpy array.")
        if len(categories) == 0:
            raise ValueError("Categories cannot be empty.")

        self.categories_ = np.sort(categories)
        self.category_map_ = {category: idx for idx, category in enumerate(self.categories_)}
        self.inverse_category_map_ = {idx: category for idx, category in enumerate(self.categories_)}
    
    def transform(self, X: np.ndarray | pd.Series) -> pd.Series:
        """
        Transform the data to ordinal values using direct indexing.

        Parameters
        ----------
        X : np.ndarray | pd.Series
            Input data to transform.

        Returns
        -------
        pd.Series
            Transformed data with ordinal values.

        """

        if self.categories_ is None:
            raise ValueError(
                "The encoder has not been fitted yet. Call 'fit' before 'transform'."
            )
        if not isinstance(X, (np.ndarray, pd.Series)):
            raise ValueError("Input data must be a numpy array or pandas Series.")
        
        encoded_data = pd.Series(X).map(self.category_map_)

        return encoded_data
    
    def inverse_transform(self, X: np.ndarray | pd.Series) -> pd.Series:
        """
        Inverse transform the encoded data back to original categories.

        Parameters
        ----------
        X : np.ndarray | pd.Series
            Encoded data to inverse transform.

        Returns
        -------
        pd.Series
            Inverse transformed data with original categories.
        """

        if self.categories_ is None:
            raise ValueError(
                "The encoder has not been fitted yet. Call 'fit' before 'inverse_transform'."
            )
        if not isinstance(X, (np.ndarray, pd.Series)):
            raise ValueError("Input data must be a numpy array or pandas Series.")
        
        inverse_encoded_data = (
            pd.Series(X)
            .map(self.inverse_category_map_)
        )

        return inverse_encoded_data