################################################################################
#                           ForecasterEquivalentDate                           #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Union, Optional, Callable, Any
import warnings
import sys
import numpy as np
import pandas as pd

import skforecast
from ..utils import check_predict_input
from ..utils import preprocess_y
from ..utils import preprocess_last_window
from ..utils import expand_index


class ForecasterEquivalentDate():
    """
    This forecaster predicts future values based on the most recent equivalent
    date. It also allows to aggregate multiple past values of the equivalent
    date using a function (e.g. mean, median, max, min, etc.). The equivalent
    date is calculated by moving back in time a specified number of steps (offset).
    The offset can be defined as an integer or as a pandas DateOffset. This
    approach is useful as a baseline, but it is a simplistic method and may not
    capture complex underlying patterns.
    
    Parameters
    ----------
    offset : int, pandas.tseries.offsets.DateOffset
        Number of steps to go back in time to find the most recent equivalent
        date to the target period.
        If `offset` is an integer, it represents the number of steps to go back
        in time. For example, if the frequency of the time series is daily, 
        `offset = 7` means that the most recent data similar to the target
        period is the value observed 7 days ago.
        Pandas DateOffsets can also be used to move forward a given number of 
        valid dates. For example, Bday(2) can be used to move back two business 
        days. If the date does not start on a valid date, it is first moved to a 
        valid date. For example, if the date is a Saturday, it is moved to the 
        previous Friday. Then, the offset is applied. If the result is a non-valid 
        date, it is moved to the next valid date. For example, if the date
        is a Sunday, it is moved to the next Monday. 
        For more information about offsets, see
        https://pandas.pydata.org/docs/reference/offset_frequency.html.
    n_offsets : int, default `1`
        Number of equivalent dates (multiple of offset) used in the prediction.
        If `n_offsets` is greater than 1, the values at the equivalent dates are
        aggregated using the `agg_func` function. For example, if the frequency
        of the time series is daily, `offset = 7`, `n_offsets = 2` and
        `agg_func = np.mean`, the predicted value will be the mean of the values
        observed 7 and 14 days ago.
    agg_func : Callable, default `np.mean`
        Function used to aggregate the values of the equivalent dates when the
        number of equivalent dates (`n_offsets`) is greater than 1.
    forecaster_id : str, int, default `None`
        Name used as an identifier of the forecaster.
    
    Attributes
    ----------
    offset : int, pandas.tseries.offsets.DateOffset
        Number of steps to go back in time to find the most recent equivalent
        date to the target period.
        If `offset` is an integer, it represents the number of steps to go back
        in time. For example, if the frequency of the time series is daily, 
        `offset = 7` means that the most recent data similar to the target
        period is the value observed 7 days ago.
        Pandas DateOffsets can also be used to move forward a given number of 
        valid dates. For example, Bday(2) can be used to move back two business 
        days. If the date does not start on a valid date, it is first moved to a 
        valid date. For example, if the date is a Saturday, it is moved to the 
        previous Friday. Then, the offset is applied. If the result is a non-valid 
        date, it is moved to the next valid date. For example, if the date
        is a Sunday, it is moved to the next Monday. 
        For more information about offsets, see
        https://pandas.pydata.org/docs/reference/offset_frequency.html.
    n_offsets : int
        Number of equivalent dates (multiple of offset) used in the prediction.
        If `offset` is greater than 1, the value at the equivalent dates is
        aggregated using the `agg_func` function. For example, if the frequency
        of the time series is daily, `offset = 7`, `n_offsets = 2` and
        `agg_func = np.mean`, the predicted value will be the mean of the values
        observed 7 and 14 days ago.
    agg_func : Callable
        Function used to aggregate the values of the equivalent dates when the
        number of equivalent dates (`n_offsets`) is greater than 1.
    window_size : int
        Number of past values needed to include the last equivalent dates according
        to the `offset` and `n_offsets`.
    last_window_ : pandas Series
        This window represents the most recent data observed by the predictor
        during its training phase. It contains the past values needed to include
        the last equivalent date according the `offset` and `n_offsets`.
    index_type_ : type
        Type of index of the input used in training.
    index_freq_ : str
        Frequency of Index of the input used in training.
    training_range_ : pandas Index
        First and last values of index of the data used during training.
    creation_date : str
        Date of creation.
    is_fitted : bool
        Tag to identify if the regressor has been fitted (trained).
    fit_date : str
        Date of last fit.
    skforecast_version : str
        Version of skforecast library used to create the forecaster.
    python_version : str
        Version of python used to create the forecaster.
    forecaster_id : str, int
        Name used as an identifier of the forecaster.
    regressor : Ignored
        Not used, present here for API consistency by convention.
    differentiation : Ignored
        Not used, present here for API consistency by convention.

    """
    
    def __init__(
        self,
        offset: Union[int, pd.tseries.offsets.DateOffset],
        n_offsets: int = 1,
        agg_func: Callable = np.mean,
        forecaster_id: Optional[Union[str, int]] = None
    ) -> None:
        
        self.offset             = offset
        self.n_offsets          = n_offsets
        self.agg_func           = agg_func
        self.last_window_       = None
        self.index_type_        = None
        self.index_freq_        = None
        self.training_range_    = None
        self.creation_date      = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.is_fitted          = False
        self.fit_date           = None
        self.skforecast_version = skforecast.__version__
        self.python_version     = sys.version.split(" ")[0]
        self.forecaster_id      = forecaster_id
        self.regressor          = None
        self.differentiation    = None
       
        if not isinstance(self.offset, (int, pd.tseries.offsets.DateOffset)):
            raise TypeError(
                ("`offset` must be an integer greater than 0 or a "
                 "pandas.tseries.offsets. Find more information about offsets in "
                 "https://pandas.pydata.org/docs/reference/offset_frequency.html")
            )
        
        self.window_size = self.offset * self.n_offsets

    def __repr__(
        self
    ) -> str:
        """
        Information displayed when a Forecaster object is printed.
        """

        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Offset: {self.offset} \n"
            f"Number of offsets: {self.n_offsets} \n"
            f"Aggregation function: {self.agg_func.__name__} \n"
            f"Window size: {self.window_size} \n"
            f"Training range: {self.training_range_.to_list() if self.is_fitted else None} \n"
            f"Training index type: {str(self.index_type_).split('.')[-1][:-2] if self.is_fitted else None} \n"
            f"Training index frequency: {self.index_freq_ if self.is_fitted else None} \n"
            f"Creation date: {self.creation_date} \n"
            f"Last fit date: {self.fit_date} \n"
            f"Skforecast version: {self.skforecast_version} \n"
            f"Python version: {self.python_version} \n"
            f"Forecaster id: {self.forecaster_id} \n"
        )

        return info

    def fit(
        self,
        y: pd.Series,
        exog: Any = None,
        store_in_sample_residuals: Any = None
    ) -> None:
        """
        Training Forecaster.
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        exog : Ignored
            Not used, present here for API consistency by convention.
        store_in_sample_residuals : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        None
        
        """

        if isinstance(self.offset, pd.tseries.offsets.DateOffset):
            if not isinstance(y.index, pd.DatetimeIndex):
                raise TypeError(
                    ("If `offset` is a pandas DateOffset, the index of `y` must be a "
                     "pandas DatetimeIndex with frequency.")
                )
            elif y.index.freq is None:
                raise TypeError(
                    ("If `offset` is a pandas DateOffset, the index of `y` must be a "
                     "pandas DatetimeIndex with frequency.")
                )
        
        # Reset values in case the forecaster has already been fitted.
        self.last_window_    = None
        self.index_type_     = None
        self.index_freq_     = None
        self.training_range_ = None
        self.is_fitted       = False

        _, y_index = preprocess_y(y=y, return_values=False)

        if isinstance(self.offset, pd.tseries.offsets.DateOffset):
            # Calculate the window_size in steps for compatibility with the
            # check_predict_input function. This is not a exact calculation
            # because the offset follows the calendar rules and the distance
            # between two dates may not be constant.
            first_valid_index = (y_index[-1] - self.offset * self.n_offsets)

            try:
                window_size_idx_start = y_index.get_loc(first_valid_index)
                window_size_idx_end = y_index.get_loc(y_index[-1])
                self.window_size = window_size_idx_end - window_size_idx_start
            except KeyError:
                raise ValueError(
                    (f"The length of `y` ({len(y)}), must be greater than or equal "
                     f"to the window size ({self.window_size}). This is because  "
                     f"the offset ({self.offset}) is larger than the available "
                     f"data. Try to decrease the size of the offset ({self.offset}), "
                     f"the number of n_offsets ({self.n_offsets}) or increase the "
                     f"size of `y`.")
                )
        else:
            if len(y) < self.window_size:
                raise ValueError(
                    (f"The length of `y` ({len(y)}), must be greater than or equal "
                     f"to the window size ({self.window_size}). This is because  "
                     f"the offset ({self.offset}) is larger than the available "
                     f"data. Try to decrease the size of the offset ({self.offset}), "
                     f"the number of n_offsets ({self.n_offsets}) or increase the "
                     f"size of `y`.")
                )
        
        self.is_fitted = True
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range_ = y_index[[0, -1]]
        self.index_type_ = type(y_index)
        self.index_freq_ = (
            y_index.freqstr if isinstance(y_index, pd.DatetimeIndex) else y_index.step
        )
        
        # The last time window of training data is stored so that equivalent
        # dates are available when calling the `predict` method.
        # Store the whole series to avoid errors when the offset is larger 
        # than the data available.
        self.last_window_ = y.copy()

    def predict(
        self,
        steps: int,
        last_window: Optional[pd.Series] = None,
        exog: Any = None,
    ) -> pd.Series:
        """
        Predict n steps ahead.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        last_window : pandas Series, default `None`
            Past values needed to select the last equivalent dates according to 
            the offset. If `last_window = None`, the values stored in 
            `self.last_window_` are used and the predictions start immediately 
            after the training data.
        exog : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        predictions : pandas Series
            Predicted values.
        
        """

        if last_window is None:
            last_window = self.last_window_

        check_predict_input(
            forecaster_name = type(self).__name__,
            steps           = steps,
            is_fitted       = self.is_fitted,
            exog_in_        = False,
            index_type_     = self.index_type_,
            index_freq_     = self.index_freq_,
            window_size     = self.window_size,
            last_window     = last_window
        )

        last_window = last_window.copy()

        last_window_values, last_window_index = preprocess_last_window(
                                                    last_window = last_window
                                                )
        prediction_index = expand_index(index=last_window_index, steps=steps)
        
        if isinstance(self.offset, int):

            equivalent_indexes = np.tile(
                                     np.arange(-self.offset, 0),
                                     int(np.ceil(steps / self.offset))
                                 )
            equivalent_indexes = equivalent_indexes[:steps]

            if self.n_offsets == 1:
                equivalent_values = last_window_values[equivalent_indexes]
                predictions = equivalent_values.ravel()

            if self.n_offsets > 1:
                equivalent_indexes = [equivalent_indexes - n * self.offset 
                                      for n in np.arange(self.n_offsets)]
                equivalent_indexes = np.vstack(equivalent_indexes)
                equivalent_values = last_window_values[equivalent_indexes]
                predictions = np.apply_along_axis(
                                  self.agg_func,
                                  axis = 0,
                                  arr  = equivalent_values
                              )
            
            predictions = pd.Series(
                              data  = predictions,
                              index = prediction_index,
                              name  = 'pred'
                          )

        if isinstance(self.offset, pd.tseries.offsets.DateOffset):

            predictions_index = prediction_index
            max_allowed_date = last_window_index[-1]

            # For every date in predictions_index, calculate the n offsets
            offset_dates = []
            for date in predictions_index:
                selected_offsets = []
                while len(selected_offsets) < self.n_offsets:
                    offset_date = date - self.offset
                    if offset_date <= max_allowed_date:
                        selected_offsets.append(offset_date)
                    date = offset_date
                offset_dates.append(selected_offsets)
            
            offset_dates = np.array(offset_dates)
    
            # Select the values of the time series corresponding to the each
            # offset date. If the offset date is not in the time series, the
            # value is set to NaN.
            equivalent_values = (
                last_window.
                reindex(offset_dates.ravel())
                .to_numpy()
                .reshape(-1, self.n_offsets)
            )
            equivalent_values = pd.DataFrame(
                                    data    = equivalent_values,
                                    index   = predictions_index,
                                    columns = [f'offset_{i}' 
                                               for i in range(self.n_offsets)]
                                )
            
            # Error if all values are missing
            if equivalent_values.isnull().all().all():
                raise ValueError(
                    (f"All equivalent values are missing. This is caused by using "
                     f"an offset ({self.offset}) larger than the available data. "
                     f"Try to decrease the size of the offset ({self.offset}), "
                     f"the number of n_offsets ({self.n_offsets}) or increase the "
                     f"size of `last_window`. In backtesting, this error may be "
                     f"caused by using an `initial_train_size` too small.")
                )
            
            # Warning if equivalent values are missing
            incomplete_offsets = equivalent_values.isnull().any(axis=1)
            incomplete_offsets = incomplete_offsets[incomplete_offsets].index
            if not incomplete_offsets.empty:
                warnings.warn(
                    (f"Steps: {incomplete_offsets.strftime('%Y-%m-%d').to_list()} "
                     f"are calculated with less than {self.n_offsets} n_offsets. "
                     f"To avoid this, increase the `last_window` size or decrease "
                     f"the number of n_offsets. The current configuration requires " 
                     f"a total offset of {self.offset * self.n_offsets}.")
                )
            
            aggregate_values = equivalent_values.apply(self.agg_func, axis=1)
            predictions = aggregate_values.rename('pred')
        
        return predictions

    def summary(self) -> None:
        """
        Show forecaster information.
        
        Parameters
        ----------
        self

        Returns
        -------
        None
        
        """
        
        print(self)
