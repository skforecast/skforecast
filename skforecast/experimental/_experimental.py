################################################################################
#                            experimental                                      #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8


import pandas as pd
import numpy as np

def calculate_distance_from_holiday(df, holiday_column='is_holiday', date_column='date', fill_na=0):
    """
    Calculate the number of days to the next holiday and the number of days since the last holiday.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the holiday data.
    holiday_column : str, optional
        The name of the column indicating holidays (True/False), by default 'is_holiday'.
    date_column : str, optional
        The name of the column containing the dates, by default 'date'.
    fill_na : int or float, optional
        Value to fill for NaN values in the output columns, by default 0.

    Returns
    -------
    pd.DataFrame
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
    days_to_holiday[has_next] = (holiday_dates_sorted[next_idx[has_next]] - dates[has_next]).astype('timedelta64[D]').astype(int)

    # For previous holiday (left side)
    prev_idx = np.searchsorted(holiday_dates_sorted, dates, side='right') - 1
    has_prev = prev_idx >= 0
    days_since_holiday = np.full(len(dates), np.nan)
    days_since_holiday[has_prev] = (dates[has_prev] - holiday_dates_sorted[prev_idx[has_prev]]).astype('timedelta64[D]').astype(int)

    df["days_to_holiday"] = pd.Series(days_to_holiday, dtype="Int64").fillna(fill_na)
    df["days_since_holiday"] = pd.Series(days_since_holiday, dtype="Int64").fillna(fill_na)
    
    return df