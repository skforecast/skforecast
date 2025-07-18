################################################################################
#                               experimental                                   #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

import pandas as pd
import numpy as np


def calculate_distance_from_holiday(
    df: pd.DataFrame, 
    holiday_column: str = 'is_holiday',
    date_column: str = 'date',
    fill_na: int | float = 0.
) -> pd.DataFrame: # pragma: no cover
    """
    Calculate the number of days to the next holiday and the number of days since 
    the last holiday.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the holiday data.
    holiday_column : str, default 'is_holiday'
        The name of the column indicating holidays (True/False), by default 'is_holiday'.
    date_column : str, default 'date'
        The name of the column containing the dates, by default 'date'.
    fill_na : int, float, default 0.
        Value to fill for NaN values in the output columns, by default 0.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional columns for days to the next holiday ('days_to_holiday') 
        and days since the last holiday ('days_since_holiday').
    
    Notes
    -----
    The function assumes that the input `df` contains a boolean column indicating holidays
    and a date column. It calculates the number of days to the next holiday and the number of
    days since the last holiday for each date in the date column.

    """

    df = df.reset_index(drop=True)
    df[date_column] = pd.to_datetime(df[date_column])
    
    dates = df[date_column].to_numpy()
    holiday_dates = df.loc[df[holiday_column], date_column].to_numpy()
    holiday_dates_sorted = np.sort(holiday_dates)

    # For next holiday (right side)
    next_idx = np.searchsorted(holiday_dates_sorted, dates, side='left')
    has_next = next_idx < len(holiday_dates_sorted)
    days_to_holiday = np.full(len(dates), np.nan)
    days_to_holiday[has_next] = (
        holiday_dates_sorted[next_idx[has_next]] - dates[has_next]
    ).astype('timedelta64[D]').astype(int)

    # For previous holiday (left side)
    prev_idx = np.searchsorted(holiday_dates_sorted, dates, side='right') - 1
    has_prev = prev_idx >= 0
    days_since_holiday = np.full(len(dates), np.nan)
    days_since_holiday[has_prev] = (
        dates[has_prev] - holiday_dates_sorted[prev_idx[has_prev]]
    ).astype('timedelta64[D]').astype(int)

    df["days_to_holiday"] = pd.Series(days_to_holiday, dtype="Int64").fillna(fill_na)
    df["days_since_holiday"] = pd.Series(days_since_holiday, dtype="Int64").fillna(fill_na)
    
    return df


class FastOrdinalEncoder: # pragma: no cover
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
    unknown_value : int | float, default=np.nan
        Value to use for unknown categories.
    
    """

    #TODO: changes needed in ForecasterRecursiveMultiSeries to use this encoder:
        # if self.is_fitted:
        #     encoded_values = self.encoder.transform(X_train[['_level_skforecast']])
        # else:
        #     if self.encoding == 'onehot':
        #         encoded_values = self.encoder.fit_transform(X_train[['_level_skforecast']])
        #         for i, code in enumerate(self.encoder.categories_[0]):
        #             self.encoding_mapping_[code] = i
        #     else:
        #         self.encoder.fit(categories=series_names_in_)
        #         encoded_values = self.encoder.transform(X_train['_level_skforecast'])
        #         self.encoding_mapping_ = self.encoder.category_map_.copy()

    def __init__(self, unknown_value: int | float = np.nan):

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
    
    def transform(self, X: np.ndarray | pd.Series | pd.DataFrame) -> pd.Series:
        """
        Transform the data to ordinal values using direct indexing.

        Parameters
        ----------
        X : np.ndarray, pd.Series, pd.DataFrame
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
        if not isinstance(X, (np.ndarray, pd.Series, pd.DataFrame)):
            raise ValueError("Input data must be a numpy array or pandas Series or DataFrame.")
        
        if isinstance(X, pd.DataFrame):
            if X.shape[1] > 1:
                raise ValueError("Input DataFrame must have only one column.")
            X = X.iloc[:, 0]
        
        encoded_data = pd.Series(X).map(self.category_map_)
        encoded_data = encoded_data.to_frame()

        return encoded_data
    
    def inverse_transform(self, X: np.ndarray | pd.Series | pd.DataFrame) -> np.ndarray:
        """
        Inverse transform the encoded data back to original categories.

        Parameters
        ----------
        X : np.ndarray, pd.Series, pd.DataFrame
            Encoded data to inverse transform.

        Returns
        -------
        np.ndarray
            Inverse transformed data with original categories.
        """

        if self.categories_ is None:
            raise ValueError(
                "The encoder has not been fitted yet. Call 'fit' before 'inverse_transform'."
            )
        if not isinstance(X, (np.ndarray, pd.Series, pd.DataFrame)):
            raise ValueError("Input data must be a numpy array or pandas Series or DataFrame.")

        if isinstance(X, pd.DataFrame):
            if X.shape[1] > 1:
                raise ValueError("Input DataFrame must have only one column.")
            X = X.iloc[:, 0]

        inverse_encoded_data = (
            pd.Series(X)
            .map(self.inverse_category_map_)
            .fillna(self.unknown_value)
        ).to_numpy().reshape(-1, 1)

        return inverse_encoded_data
