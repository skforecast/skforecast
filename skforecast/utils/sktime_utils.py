################################################################################
#                      skforecast.utils.sktime_utils                          #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8


import warnings
from typing import Union, Any, Optional
import pandas as pd
import datetime
import numpy as np
from ..utils import check_optional_dependency


try:
    from sktime.transformations.base import BaseTransformer
except Exception as e:
    package_name = str(e).split(" ")[-1].replace("'", "")
    check_optional_dependency(package_name=package_name)


def nparray_to_df(
        data: np.array,
        first_date: datetime.datetime,
        columns: list,
        period: str) -> pd.DataFrame:
    """
    Given data as an numpy array, the first date and column names, convert an
    numpy array to a Pandas DataFrame with a DatetimeIndex.

    Parameters
    ----------
    data : np.array
        Input data as an np array
    first_date : datetime.datetime
        The first date to be used in the index for the DataFrame
    columns : list
        The names of the columns in the DataFrame
    period : str
        The string representing time series frequency

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the data

    """
    dates = pd.date_range(first_date, periods=data.shape[0], freq=period)
    return pd.DataFrame(data=data, index=dates, columns=columns)


class StartSktimePipe(BaseTransformer):

    """
    This class converts input data to a format accepted by sktime transformers.
    An object of this class should be placed at the start of the pipeline
    containing sktime transformers.

    Parameters
    ----------
    period : str
        The string representing time series frequency
    columns : list
        The names of the columns in the DataFrame
    first_date : datetime.datetime
        The first date to be used in the index for the DataFrame

    Attributes
    ----------
    period : str
        The string representing time series frequency
    columns : list
        The names of the columns in the DataFrame
    first_date : datetime.datetime
        The first date to be used in the index for the DataFrame

    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "capability:inverse_transform": True,
    }

    def __init__(self) -> None:
        self.period = None
        self.columns = None
        self.first_date = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None):
        """
        Fits the transformer. Stores values required to convert a
        numpy array to a DataFrame with a DatetimeIndex.

        Parameters
        ----------
        X : pandas DataFrame
            Time series used to fit the transformer
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : StartSktimePipe

        """
        self.first_date = X.index[0]
        self.period = X.index.freqstr
        self.columns = X.columns
        return self

    def transform(self, X, y=None):
        """
        Converts a numpy array, a pandas Series or pandas DataFrame with
        DatetimeIndex to a pandas DataFrame with a PeriodIndex required by
        sktime transformers.

        Parameters
        ----------
        X : numpy ndarray or a pandas DataFrame with a DatetimeIndex.
            Time series to be tranformed with sktime transformers.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X : pandas DataFrame
            A pandas DataFrame with a PeriodIndex.

        """
        if type(X) == np.ndarray:
            X = nparray_to_df(X, self.first_date, self.columns, self.period)
        if type(X.index) != pd.core.indexes.period.PeriodIndex:
            X = X.copy()
            X.index = X.index.to_period(self.period)
        return X

    def inverse_transform(self, X, y=None):
        """
        Converts a pandas Series or pandas DataFrame with a PeriodIndex
        produced by sktime transformers to a pandas DataFrame with
        DatetimeIndex.

        Parameters
        ----------
        X : A pandas Series or DataFrame with PeriodIndex.
            Time series output by an sktime transformer to be
            reverse-tranformed.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X : pandas DataFrame
            A pandas DataFrame with a PeriodIndex.

        """
        if type(X.index) != pd.core.indexes.datetimes.DatetimeIndex:
            X = X.copy()
            X.index = X.index.to_timestamp(how="end").date.astype('datetime64[ns]')
            X = X.asfreq(self.period)
        return X


class EndSktimePipe(BaseTransformer):

    """
    This class converts data produced by sktime transformers into a format
    accepted by skforecast objects.
    An object of this class should be placed at the end of the pipeline
    containing sktime transformers.

    Parameters
    ----------
    period : str
        The string representing time series frequency
    columns : list
        The names of the columns in the DataFrame
    first_date : datetime.datetime
        The first date to be used in the index for the DataFrame

    Attributes
    ----------
    period : str
        The string representing time series frequency
    columns : list
        The names of the columns in the DataFrame
    first_date : datetime.datetime
        The first date to be used in the index for the DataFrame

    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "capability:inverse_transform": True,
    }

    def __init__(self) -> None:
        self.period = None
        self.columns = None
        self.first_date = None

    def fit(self, X, y=None):
        """
        Fits the transformer. Stores values required to convert a
        numpy array to a DataFrame with a DatetimeIndex.

        Parameters
        ----------
        X : pandas DataFrame
            Time series used to fit the transformer
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : EndSktimePipe

        """
        self.first_date = X.index.to_timestamp(how="end").date.astype('datetime64[ns]')[0]
        self.period = X.index.freqstr
        self.columns = X.columns
        return self

    def transform(self, X, y=None):
        """
        Converts a pandas DataFrame with a PeriodIndex produced by sktime
        transformers to a pandas DataFrame with a DatetimeIndex.

        Parameters
        ----------
        X : A pandas DataFrame with a PeriodIndex.
            Time series that has/have been tranformed with sktime
            transformers.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X : pandas DataFrame
            A pandas DataFrame with a DatetimeIndex.

        """
        if type(X.index) != pd.core.indexes.datetimes.DatetimeIndex:
            X = X.copy()
            X.index = X.index.to_timestamp(how="end").date.astype('datetime64[ns]')
            X = X.asfreq(self.period)
        return X

    def inverse_transform(self, X, y=None):
        """
        Converts a numpy array, a pandas Series or pandas DataFrame with
        DatetimeIndex to a pandas DataFrame with a PeriodIndex required by
        sktime transformers.

        Parameters
        ----------
        X : numpy ndarray or a pandas DataFrame with DatetimeIndex.
            Time series to be reverse-tranformed with sktime transformers.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X : pandas DataFrame
            A pandas DataFrame with a PeriodIndex.

        """

        if type(X) == np.ndarray:
            X = nparray_to_df(X, self.first_date, self.columns, self.period)
        if type(X.index) != pd.core.indexes.period.PeriodIndex:
            X = X.copy()
            X.index = X.index.to_period(self.period)
        return X
