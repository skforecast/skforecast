################################################################################
#                           ForecasterEquivalentDate                           #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
from typing import Callable, Any
import warnings
import sys
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from .. import __version__
from ..exceptions import MissingValuesWarning, ResidualsUsageWarning
from ..utils import (
    check_y,
    check_predict_input,
    check_residuals_input,
    check_interval,
    check_extract_values_and_index,
    expand_index,
    get_style_repr_html
)
from ..preprocessing import QuantileBinner


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
    n_offsets : int, default 1
        Number of equivalent dates (multiple of offset) used in the prediction.
        If `n_offsets` is greater than 1, the values at the equivalent dates are
        aggregated using the `agg_func` function. For example, if the frequency
        of the time series is daily, `offset = 7`, `n_offsets = 2` and
        `agg_func = np.mean`, the predicted value will be the mean of the values
        observed 7 and 14 days ago.
    agg_func : Callable, default np.mean
        Function used to aggregate the values of the equivalent dates when the
        number of equivalent dates (`n_offsets`) is greater than 1.
    binner_kwargs : dict, default None
        Additional arguments to pass to the `QuantileBinner` used to discretize 
        the residuals into k bins according to the predicted values associated 
        with each residual. Available arguments are: `n_bins`, `method`, `subsample`,
        `random_state` and `dtype`. Argument `method` is passed internally to the
        function `numpy.percentile`.
        **New in version 0.17.0**
    forecaster_id : str, int, default None
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
    series_name_in_ : str
        Names of the series provided by the user during training.
    in_sample_residuals_ : numpy ndarray
        Residuals of the model when predicting training data. Only stored up to
        10_000 values. If `transformer_y` is not `None`, residuals are stored in
        the transformed scale. If `differentiation` is not `None`, residuals are
        stored after differentiation.
    in_sample_residuals_by_bin_ : dict
        In sample residuals binned according to the predicted value each residual
        is associated with. The number of residuals stored per bin is limited to 
        `10_000 // self.binner.n_bins_` in the form `{bin: residuals}`. If 
        `transformer_y` is not `None`, residuals are stored in the transformed 
        scale. If `differentiation` is not `None`, residuals are stored after 
        differentiation. 
    out_sample_residuals_ : numpy ndarray
        Residuals of the model when predicting non-training data. Only stored up to
        10_000 values. Use `set_out_sample_residuals()` method to set values. If 
        `transformer_y` is not `None`, residuals are stored in the transformed 
        scale. If `differentiation` is not `None`, residuals are stored after 
        differentiation.
    out_sample_residuals_by_bin_ : dict
        Out of sample residuals binned according to the predicted value each residual
        is associated with. The number of residuals stored per bin is limited to 
        `10_000 // self.binner.n_bins_` in the form `{bin: residuals}`. If 
        `transformer_y` is not `None`, residuals are stored in the transformed 
        scale. If `differentiation` is not `None`, residuals are stored after 
        differentiation. 
    binner : skforecast.preprocessing.QuantileBinner
        `QuantileBinner` used to discretize residuals into k bins according 
        to the predicted values associated with each residual.
    binner_intervals_ : dict
        Intervals used to discretize residuals into k bins according to the predicted
        values associated with each residual.
    binner_kwargs : dict
        Additional arguments to pass to the `QuantileBinner`.
    creation_date : str
        Date of creation.
    is_fitted : bool
        Tag to identify if the estimator has been fitted (trained).
    fit_date : str
        Date of last fit.
    skforecast_version : str
        Version of skforecast library used to create the forecaster.
    python_version : str
        Version of python used to create the forecaster.
    forecaster_id : str, int
        Name used as an identifier of the forecaster.
    __skforecast_tags__ : dict
        Tags associated with the forecaster.
    _probabilistic_mode: str, bool
        Private attribute used to indicate whether the forecaster should perform 
        some calculations during backtesting.
    estimator : Ignored
        Not used, present here for API consistency by convention.
    differentiation : Ignored
        Not used, present here for API consistency by convention.
    differentiation_max : Ignored
        Not used, present here for API consistency by convention.

    """
    
    def __init__(
        self,
        offset: int | pd.tseries.offsets.DateOffset,
        n_offsets: int = 1,
        agg_func: Callable = np.mean,
        binner_kwargs: dict[str, object] | None = None,
        forecaster_id: str | int | None = None
    ) -> None:
        
        self.offset                       = offset
        self.n_offsets                    = n_offsets
        self.agg_func                     = agg_func
        self.last_window_                 = None
        self.index_type_                  = None
        self.index_freq_                  = None
        self.training_range_              = None
        self.series_name_in_              = None
        self.in_sample_residuals_         = None
        self.out_sample_residuals_        = None
        self.in_sample_residuals_by_bin_  = None
        self.out_sample_residuals_by_bin_ = None
        self.creation_date                = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.is_fitted                    = False
        self.fit_date                     = None
        self.skforecast_version           = __version__
        self.python_version               = sys.version.split(" ")[0]
        self.forecaster_id                = forecaster_id
        self._probabilistic_mode          = "binned"
        self.estimator                    = None
        self.differentiation              = None
        self.differentiation_max          = None
       
        if not isinstance(self.offset, (int, pd.tseries.offsets.DateOffset)):
            raise TypeError(
                "`offset` must be an integer greater than 0 or a "
                "pandas.tseries.offsets. Find more information about offsets in "
                "https://pandas.pydata.org/docs/reference/offset_frequency.html"
            )
        
        self.window_size = self.offset * self.n_offsets

        self.binner_kwargs = binner_kwargs
        if binner_kwargs is None:
            self.binner_kwargs = {
                'n_bins': 10, 'method': 'linear', 'subsample': 200000,
                'random_state': 789654, 'dtype': np.float64
            }
        self.binner = QuantileBinner(**self.binner_kwargs)
        self.binner_intervals_ = None
        
        self.__skforecast_tags__ = {
            "library": "skforecast",
            "forecaster_name": "ForecasterEquivalentDate",
            "forecaster_task": "regression",
            "forecasting_scope": "single-series",  # single-series | global
            "forecasting_strategy": "recursive",   # recursive | direct | deep_learning
            "index_types_supported": ["pandas.RangeIndex", "pandas.DatetimeIndex"],
            "requires_index_frequency": True,

            "allowed_input_types_series": ["pandas.Series"],
            "supports_exog": False,
            "allowed_input_types_exog": [],
            "handles_missing_values_series": False, 
            "handles_missing_values_exog": False, 

            "supports_lags": False,
            "supports_window_features": False,
            "supports_transformer_series": False,
            "supports_transformer_exog": False,
            "supports_weight_func": False,
            "supports_differentiation": False,

            "prediction_types": ["point", "interval"],
            "supports_probabilistic": True,
            "probabilistic_methods": ["conformal"],
            "handles_binned_residuals": True
        }

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
            f"Series name: {self.series_name_in_} \n"
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

    def _repr_html_(self):
        """
        HTML representation of the object.
        The "General Information" section is expanded by default.
        """

        style, unique_id = get_style_repr_html(self.is_fitted)
        
        content = f"""
        <div class="container-{unique_id}">
            <p style="font-size: 1.5em; font-weight: bold; margin-block-start: 0.83em; margin-block-end: 0.83em;">{type(self).__name__}</p>
            <details open>
                <summary>General Information</summary>
                <ul>
                    <li><strong>Estimator:</strong> {type(self.estimator).__name__}</li>
                    <li><strong>Offset:</strong> {self.offset}</li>
                    <li><strong>Number of offsets:</strong> {self.n_offsets}</li>
                    <li><strong>Aggregation function:</strong> {self.agg_func.__name__}</li>
                    <li><strong>Window size:</strong> {self.window_size}</li>
                    <li><strong>Creation date:</strong> {self.creation_date}</li>
                    <li><strong>Last fit date:</strong> {self.fit_date}</li>
                    <li><strong>Skforecast version:</strong> {self.skforecast_version}</li>
                    <li><strong>Python version:</strong> {self.python_version}</li>
                    <li><strong>Forecaster id:</strong> {self.forecaster_id}</li>
                </ul>
            </details>
            <details>
                <summary>Training Information</summary>
                <ul>
                    <li><strong>Training range:</strong> {self.training_range_.to_list() if self.is_fitted else 'Not fitted'}</li>
                    <li><strong>Training index type:</strong> {str(self.index_type_).split('.')[-1][:-2] if self.is_fitted else 'Not fitted'}</li>
                    <li><strong>Training index frequency:</strong> {self.index_freq_ if self.is_fitted else 'Not fitted'}</li>
                </ul>
            </details>
            <p>
                <a href="https://skforecast.org/{__version__}/api/forecasterequivalentdate.html">&#128712 <strong>API Reference</strong></a>
                &nbsp;&nbsp;
                <a href="https://skforecast.org/{__version__}/user_guides/forecasting-baseline.html">&#128462 <strong>User Guide</strong></a>
            </p>
        </div>
        """

        return style + content

    def fit(
        self,
        y: pd.Series,
        store_in_sample_residuals: bool = False,
        random_state: int = 123,
        exog: Any = None
    ) -> None:
        """
        Training Forecaster.
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        store_in_sample_residuals : bool, default False
            If `True`, in-sample residuals will be stored in the forecaster object
            after fitting (`in_sample_residuals_` and `in_sample_residuals_by_bin_`
            attributes).
            If `False`, only the intervals of the bins are stored.
        random_state : int, default 123
            Set a seed for the random generator so that the stored sample 
            residuals are always deterministic.
        exog : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        None
        
        """
    
        if not isinstance(y, pd.Series):
            raise TypeError(
                f"`y` must be a pandas Series with a DatetimeIndex or a RangeIndex. "
                f"Found {type(y)}."
            )

        if isinstance(self.offset, pd.tseries.offsets.DateOffset):
            if not isinstance(y.index, pd.DatetimeIndex):
                raise TypeError(
                    "If `offset` is a pandas DateOffset, the index of `y` must be a "
                    "pandas DatetimeIndex with frequency."
                )
            elif y.index.freq is None:
                raise TypeError(
                    "If `offset` is a pandas DateOffset, the index of `y` must be a "
                    "pandas DatetimeIndex with frequency."
                )
        
        # Reset values in case the forecaster has already been fitted.
        self.last_window_    = None
        self.index_type_     = None
        self.index_freq_     = None
        self.training_range_ = None
        self.series_name_in_ = None
        self.is_fitted       = False

        _, y_index = check_extract_values_and_index(
            data=y, data_label='`y`', return_values=False
        )

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
                    f"The length of `y` ({len(y)}), must be greater than or equal "
                    f"to the window size ({self.window_size}). This is because  "
                    f"the offset ({self.offset}) is larger than the available "
                    f"data. Try to decrease the size of the offset ({self.offset}), "
                    f"the number of `n_offsets` ({self.n_offsets}) or increase the "
                    f"size of `y`."
                )
        else:
            if len(y) <= self.window_size:
                raise ValueError(
                    f"Length of `y` must be greater than the maximum window size "
                    f"needed by the forecaster. This is because  "
                    f"the offset ({self.offset}) is larger than the available "
                    f"data. Try to decrease the size of the offset ({self.offset}), "
                    f"the number of `n_offsets` ({self.n_offsets}) or increase the "
                    f"size of `y`.\n"
                    f"    Length `y`: {len(y)}.\n"
                    f"    Max window size: {self.window_size}.\n"
                )
        
        self.is_fitted = True
        self.series_name_in_ = y.name if y.name is not None else 'y'
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range_ = y_index[[0, -1]]
        self.index_type_ = type(y_index)
        self.index_freq_ = (
            y_index.freq if isinstance(y_index, pd.DatetimeIndex) else y_index.step
        )

        # NOTE: This is done to save time during fit in functions such as backtesting()
        if self._probabilistic_mode is not False:
            self._binning_in_sample_residuals(
                y                         = y,
                store_in_sample_residuals = store_in_sample_residuals,
                random_state              = random_state
            )
        
        # The last time window of training data is stored so that equivalent
        # dates are available when calling the `predict` method.
        # Store the whole series to avoid errors when the offset is larger 
        # than the data available.
        self.last_window_ = y.copy()

    def _binning_in_sample_residuals(
        self,
        y: pd.Series,
        store_in_sample_residuals: bool = False,
        random_state: int = 123
    ) -> None:
        """
        Bin residuals according to the predicted value each residual is
        associated with. First a `skforecast.preprocessing.QuantileBinner` object
        is fitted to the predicted values. Then, residuals are binned according
        to the predicted value each residual is associated with. Residuals are
        stored in the forecaster object as `in_sample_residuals_` and
        `in_sample_residuals_by_bin_`.

        The number of residuals stored per bin is limited to 
        `10_000 // self.binner.n_bins_`. The total number of residuals stored is
        `10_000`.
        **New in version 0.17.0**

        Parameters
        ----------
        y : pandas Series
            Training time series.
        store_in_sample_residuals : bool, default False
            If `True`, in-sample residuals will be stored in the forecaster object
            after fitting (`in_sample_residuals_` and `in_sample_residuals_by_bin_`
            attributes).
            If `False`, only the intervals of the bins are stored.
        random_state : int, default 123
            Set a seed for the random generator so that the stored sample 
            residuals are always deterministic.

        Returns
        -------
        None
        
        """
        
        if isinstance(self.offset, pd.tseries.offsets.DateOffset):
            y_preds = []
            for n_off in range(1, self.n_offsets + 1):
                idx = y.index - self.offset * n_off
                mask = idx >= y.index[0]
                y_pred = y.loc[idx[mask]]
                y_pred.index = y.index[-mask.sum():]
                y_preds.append(y_pred)

            y_preds = pd.concat(y_preds, axis=1).to_numpy()
            y_true = y.to_numpy()[-len(y_preds):]

        else:
            y_preds = [
                y.shift(self.offset * n_off)[self.window_size:]
                for n_off in range(1, self.n_offsets + 1)
            ]
            y_preds = np.column_stack(y_preds)
            y_true = y.to_numpy()[self.window_size:]

        y_pred = np.apply_along_axis(
                     self.agg_func,
                     axis = 1,
                     arr  = y_preds
                 )

        residuals = y_true - y_pred

        if self._probabilistic_mode == "binned":
            data = pd.DataFrame(
                {'prediction': y_pred, 'residuals': residuals}
            ).dropna()
            y_pred = data['prediction'].to_numpy()
            residuals = data['residuals'].to_numpy()

            self.binner.fit(y_pred)
            self.binner_intervals_ = self.binner.intervals_
    
        if store_in_sample_residuals:
            rng = np.random.default_rng(seed=random_state)
            if self._probabilistic_mode == "binned":
                data['bin'] = self.binner.transform(y_pred).astype(int)
                self.in_sample_residuals_by_bin_ = (
                    data.groupby('bin')['residuals'].apply(np.array).to_dict()
                )

                max_sample = 10_000 // self.binner.n_bins_
                for k, v in self.in_sample_residuals_by_bin_.items():
                    if len(v) > max_sample:
                        sample = v[rng.integers(low=0, high=len(v), size=max_sample)]
                        self.in_sample_residuals_by_bin_[k] = sample

                for k in self.binner_intervals_.keys():
                    if k not in self.in_sample_residuals_by_bin_:
                        self.in_sample_residuals_by_bin_[k] = np.array([])

                empty_bins = [
                    k for k, v in self.in_sample_residuals_by_bin_.items() 
                    if v.size == 0
                ]
                if empty_bins:
                    empty_bin_size = min(max_sample, len(residuals))
                    for k in empty_bins:
                        self.in_sample_residuals_by_bin_[k] = rng.choice(
                            a       = residuals,
                            size    = empty_bin_size,
                            replace = False
                        )
   
            if len(residuals) > 10_000:
                residuals = residuals[
                    rng.integers(low=0, high=len(residuals), size=10_000)
                ]

            self.in_sample_residuals_ = residuals

    def predict(
        self,
        steps: int,
        last_window: pd.Series | None = None,
        check_inputs: bool = True,
        exog: Any = None
    ) -> pd.Series:
        """
        Predict n steps ahead.
        
        Parameters
        ----------
        steps : int
            Number of steps to predict. 
        last_window : pandas Series, default None
            Past values needed to select the last equivalent dates according to 
            the offset. If `last_window = None`, the values stored in 
            `self.last_window_` are used and the predictions start immediately 
            after the training data.
        check_inputs : bool, default True
            If `True`, the input is checked for possible warnings and errors 
            with the `check_predict_input` function. This argument is created 
            for internal use and is not recommended to be changed.
        exog : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        predictions : pandas Series
            Predicted values.
        
        """

        if last_window is None:
            last_window = self.last_window_

        if check_inputs:
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

        prediction_index = expand_index(index=last_window.index, steps=steps)
        
        if isinstance(self.offset, int):
            
            last_window_values = last_window.to_numpy(copy=True).ravel()
            equivalent_indexes = np.tile(
                                     np.arange(-self.offset, 0),
                                     int(np.ceil(steps / self.offset))
                                 )
            equivalent_indexes = equivalent_indexes[:steps]

            if self.n_offsets == 1:
                equivalent_values = last_window_values[equivalent_indexes]
                predictions = equivalent_values.ravel()

            if self.n_offsets > 1:
                equivalent_indexes = [
                    equivalent_indexes - n * self.offset 
                    for n in np.arange(self.n_offsets)
                ]
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
            
            last_window = last_window.copy()
            max_allowed_date = last_window.index[-1]

            # For every date in prediction_index, calculate the n offsets
            offset_dates = []
            for date in prediction_index:
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
                                    index   = prediction_index,
                                    columns = [f'offset_{i}' for i in range(self.n_offsets)]
                                )
            
            # Error if all values are missing
            if equivalent_values.isnull().all().all():
                raise ValueError(
                    f"All equivalent values are missing. This is caused by using "
                    f"an offset ({self.offset}) larger than the available data. "
                    f"Try to decrease the size of the offset ({self.offset}), "
                    f"the number of `n_offsets` ({self.n_offsets}) or increase the "
                    f"size of `last_window`. In backtesting, this error may be "
                    f"caused by using an `initial_train_size` too small."
                )
            
            # Warning if equivalent values are missing
            incomplete_offsets = equivalent_values.isnull().any(axis=1)
            incomplete_offsets = incomplete_offsets[incomplete_offsets].index
            if not incomplete_offsets.empty:
                warnings.warn(
                    f"Steps: {incomplete_offsets.strftime('%Y-%m-%d').to_list()} "
                    f"are calculated with less than {self.n_offsets} `n_offsets`. "
                    f"To avoid this, increase the `last_window` size or decrease "
                    f"the number of `n_offsets`. The current configuration requires " 
                    f"a total offset of {self.offset * self.n_offsets}.",
                    MissingValuesWarning
                )
            
            aggregate_values = equivalent_values.apply(self.agg_func, axis=1)
            predictions = aggregate_values.rename('pred')
        
        return predictions

    def predict_interval(
        self,
        steps: int,
        last_window: pd.Series | None = None,
        method: str = 'conformal',
        interval: float | list[float] | tuple[float] = [5, 95],
        use_in_sample_residuals: bool = True,
        use_binned_residuals: bool = True,
        random_state: Any = None,
        exog: Any = None,
        n_boot: Any = None
    ) -> pd.DataFrame:
        """
        Predict n steps ahead and estimate prediction intervals using conformal 
        prediction method. Refer to the References section for additional 
        details on this method.
        
        Parameters
        ----------
        steps : int
            Number of steps to predict.
        last_window : pandas Series, default None
            Past values needed to select the last equivalent dates according to 
            the offset. If `last_window = None`, the values stored in 
            `self.last_window_` are used and the predictions start immediately 
            after the training data.
        method : str, default 'conformal'
            Technique used to estimate prediction intervals. Available options:

            - 'conformal': Employs the conformal prediction split method for 
            interval estimation [1]_.
        interval : float, list, tuple, default [5, 95]
            Confidence level of the prediction interval. Interpretation depends 
            on the method used:
            
            - If `float`, represents the nominal (expected) coverage (between 0 
            and 1). For instance, `interval=0.95` corresponds to `[2.5, 97.5]` 
            percentiles.
            - If `list` or `tuple`, defines the exact percentiles to compute, which 
            must be between 0 and 100 inclusive. For example, interval 
            of 95% should be as `interval = [2.5, 97.5]`.
            - When using `method='conformal'`, the interval must be a float or 
            a list/tuple defining a symmetric interval.
        use_in_sample_residuals : bool, default True
            If `True`, residuals from the training data are used as proxy of
            prediction error to create predictions. 
            If `False`, out of sample residuals (calibration) are used. 
            Out-of-sample residuals must be precomputed using Forecaster's
            `set_out_sample_residuals()` method.
        use_binned_residuals : bool, default True
            If `True`, residuals are selected based on the predicted values 
            (binned selection).
            If `False`, residuals are selected randomly.
        random_state : Ignored
            Not used, present here for API consistency by convention.
        exog : Ignored
            Not used, present here for API consistency by convention.
        n_boot : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        predictions : pandas DataFrame
            Values predicted by the forecaster and their estimated interval.

            - pred: predictions.
            - lower_bound: lower bound of the interval.
            - upper_bound: upper bound of the interval.

        References
        ----------        
        .. [1] MAPIE - Model Agnostic Prediction Interval Estimator.
               https://mapie.readthedocs.io/en/stable/theoretical_description_regression.html#the-split-method
    
        """

        if method != 'conformal':
            raise ValueError(
                f"Method '{method}' is not supported. Only 'conformal' is available."
            )

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

        check_residuals_input(
            forecaster_name              = type(self).__name__,
            use_in_sample_residuals      = use_in_sample_residuals,
            in_sample_residuals_         = self.in_sample_residuals_,
            out_sample_residuals_        = self.out_sample_residuals_,
            use_binned_residuals         = use_binned_residuals,
            in_sample_residuals_by_bin_  = self.in_sample_residuals_by_bin_,
            out_sample_residuals_by_bin_ = self.out_sample_residuals_by_bin_
        )

        if isinstance(interval, (list, tuple)):
            check_interval(interval=interval, ensure_symmetric_intervals=True)
            nominal_coverage = (interval[1] - interval[0]) / 100
        else:
            check_interval(alpha=interval, alpha_literal='interval')
            nominal_coverage = interval
        
        if use_in_sample_residuals:
            residuals = self.in_sample_residuals_
            residuals_by_bin = self.in_sample_residuals_by_bin_
        else:
            residuals = self.out_sample_residuals_
            residuals_by_bin = self.out_sample_residuals_by_bin_
        
        prediction_index = expand_index(index=last_window.index, steps=steps)
        
        if isinstance(self.offset, int):
            
            last_window_values = last_window.to_numpy(copy=True).ravel()
            equivalent_indexes = np.tile(
                                     np.arange(-self.offset, 0),
                                     int(np.ceil(steps / self.offset))
                                 )
            equivalent_indexes = equivalent_indexes[:steps]

            if self.n_offsets == 1:
                equivalent_values = last_window_values[equivalent_indexes]
                predictions = equivalent_values.ravel()

            if self.n_offsets > 1:
                equivalent_indexes = [
                    equivalent_indexes - n * self.offset 
                    for n in np.arange(self.n_offsets)
                ]
                equivalent_indexes = np.vstack(equivalent_indexes)
                equivalent_values = last_window_values[equivalent_indexes]
                predictions = np.apply_along_axis(
                                  self.agg_func,
                                  axis = 0,
                                  arr  = equivalent_values
                              )

        if isinstance(self.offset, pd.tseries.offsets.DateOffset):
            
            last_window = last_window.copy()
            max_allowed_date = last_window.index[-1]

            # For every date in prediction_index, calculate the n offsets
            offset_dates = []
            for date in prediction_index:
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
                                    index   = prediction_index,
                                    columns = [f'offset_{i}' for i in range(self.n_offsets)]
                                )
            
            # Error if all values are missing
            if equivalent_values.isnull().all().all():
                raise ValueError(
                    f"All equivalent values are missing. This is caused by using "
                    f"an offset ({self.offset}) larger than the available data. "
                    f"Try to decrease the size of the offset ({self.offset}), "
                    f"the number of `n_offsets` ({self.n_offsets}) or increase the "
                    f"size of `last_window`. In backtesting, this error may be "
                    f"caused by using an `initial_train_size` too small."
                )
            
            # Warning if equivalent values are missing
            incomplete_offsets = equivalent_values.isnull().any(axis=1)
            incomplete_offsets = incomplete_offsets[incomplete_offsets].index
            if not incomplete_offsets.empty:
                warnings.warn(
                    f"Steps: {incomplete_offsets.strftime('%Y-%m-%d').to_list()} "
                    f"are calculated with less than {self.n_offsets} `n_offsets`. "
                    f"To avoid this, increase the `last_window` size or decrease "
                    f"the number of `n_offsets`. The current configuration requires " 
                    f"a total offset of {self.offset * self.n_offsets}.",
                    MissingValuesWarning
                )
            
            aggregate_values = equivalent_values.apply(self.agg_func, axis=1)
            predictions = aggregate_values.to_numpy()
        
        if use_binned_residuals:
            correction_factor_by_bin = {
                k: np.quantile(np.abs(v), nominal_coverage)
                for k, v in residuals_by_bin.items()
            }
            replace_func = np.vectorize(lambda x: correction_factor_by_bin[x])
            predictions_bin = self.binner.transform(predictions)
            correction_factor = replace_func(predictions_bin)
        else:
            correction_factor = np.quantile(np.abs(residuals), nominal_coverage)
            
        lower_bound = predictions - correction_factor
        upper_bound = predictions + correction_factor
        predictions = np.column_stack([predictions, lower_bound, upper_bound])

        predictions = pd.DataFrame(
                          data    = predictions,
                          index   = prediction_index,
                          columns = ["pred", "lower_bound", "upper_bound"]
                      )
        
        return predictions

    def set_in_sample_residuals(
        self,
        y: pd.Series,
        random_state: int = 123,
        exog: Any = None
    ) -> None:
        """
        Set in-sample residuals in case they were not calculated during the
        training process. 
        
        In-sample residuals are calculated as the difference between the true 
        values and the predictions made by the forecaster using the training 
        data. The following internal attributes are updated:

        + `in_sample_residuals_`: residuals stored in a numpy ndarray.
        + `binner_intervals_`: intervals used to bin the residuals are calculated
        using the quantiles of the predicted values.
        + `in_sample_residuals_by_bin_`: residuals are binned according to the
        predicted value they are associated with and stored in a dictionary, where
        the keys are the intervals of the predicted values and the values are
        the residuals associated with that range. 

        A total of 10_000 residuals are stored in the attribute `in_sample_residuals_`.
        If the number of residuals is greater than 10_000, a random sample of
        10_000 residuals is stored. The number of residuals stored per bin is
        limited to `10_000 // self.binner.n_bins_`.
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        random_state : int, default 123
            Sets a seed to the random sampling for reproducible output.
        exog : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        None

        """

        if not self.is_fitted:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `set_in_sample_residuals()`."
            )
        
        check_y(y=y)
        y_index_range = check_extract_values_and_index(
            data=y, data_label='`y`', return_values=False
        )[1][[0, -1]]
        if not y_index_range.equals(self.training_range_):
            raise IndexError(
                f"The index range of `y` does not match the range "
                f"used during training. Please ensure the index is aligned "
                f"with the training data.\n"
                f"    Expected : {self.training_range_}\n"
                f"    Received : {y_index_range}"
            )
        
        self._binning_in_sample_residuals(
            y                         = y,
            store_in_sample_residuals = True,
            random_state              = random_state
        )

    def set_out_sample_residuals(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        append: bool = False,
        random_state: int = 123
    ) -> None:
        """
        Set new values to the attribute `out_sample_residuals_`. Out of sample
        residuals are meant to be calculated using observations that did not
        participate in the training process. Two internal attributes are updated:

        + `out_sample_residuals_`: residuals stored in a numpy ndarray.
        + `out_sample_residuals_by_bin_`: residuals are binned according to the
        predicted value they are associated with and stored in a dictionary, where
        the keys are the  intervals of the predicted values and the values are
        the residuals associated with that range. If a bin binning is empty, it
        is filled with a random sample of residuals from other bins. This is done
        to ensure that all bins have at least one residual and can be used in the
        prediction process.

        A total of 10_000 residuals are stored in the attribute `out_sample_residuals_`.
        If the number of residuals is greater than 10_000, a random sample of
        10_000 residuals is stored. The number of residuals stored per bin is
        limited to `10_000 // self.binner.n_bins_`.
        
        Parameters
        ----------
        y_true : numpy ndarray, pandas Series
            True values of the time series from which the residuals have been
            calculated.
        y_pred : numpy ndarray, pandas Series
            Predicted values of the time series.
        append : bool, default False
            If `True`, new residuals are added to the once already stored in the
            forecaster. If after appending the new residuals, the limit of
            `10_000 // self.binner.n_bins_` values per bin is reached, a random
            sample of residuals is stored.
        random_state : int, default 123
            Sets a seed to the random sampling for reproducible output.

        Returns
        -------
        None

        """

        if not self.is_fitted:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `set_out_sample_residuals()`."
            )

        if not isinstance(y_true, (np.ndarray, pd.Series)):
            raise TypeError(
                f"`y_true` argument must be `numpy ndarray` or `pandas Series`. "
                f"Got {type(y_true)}."
            )
        
        if not isinstance(y_pred, (np.ndarray, pd.Series)):
            raise TypeError(
                f"`y_pred` argument must be `numpy ndarray` or `pandas Series`. "
                f"Got {type(y_pred)}."
            )
        
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"`y_true` and `y_pred` must have the same length. "
                f"Got {len(y_true)} and {len(y_pred)}."
            )
        
        if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
            if not y_true.index.equals(y_pred.index):
                raise ValueError(
                    "`y_true` and `y_pred` must have the same index."
                )
        
        if not isinstance(y_pred, np.ndarray):
            y_pred = y_pred.to_numpy()
        if not isinstance(y_true, np.ndarray):
            y_true = y_true.to_numpy()
        
        data = pd.DataFrame(
            {'prediction': y_pred, 'residuals': y_true - y_pred}
        ).dropna()
        y_pred = data['prediction'].to_numpy()
        residuals = data['residuals'].to_numpy()

        data['bin'] = self.binner.transform(y_pred).astype(int)
        residuals_by_bin = data.groupby('bin')['residuals'].apply(np.array).to_dict()

        out_sample_residuals = (
            np.array([]) 
            if self.out_sample_residuals_ is None
            else self.out_sample_residuals_
        )
        out_sample_residuals_by_bin = (
            {} 
            if self.out_sample_residuals_by_bin_ is None
            else self.out_sample_residuals_by_bin_
        )
        if append:
            out_sample_residuals = np.concatenate([out_sample_residuals, residuals])
            for k, v in residuals_by_bin.items():
                if k in out_sample_residuals_by_bin:
                    out_sample_residuals_by_bin[k] = np.concatenate(
                        (out_sample_residuals_by_bin[k], v)
                    )
                else:
                    out_sample_residuals_by_bin[k] = v
        else:
            out_sample_residuals = residuals
            out_sample_residuals_by_bin = residuals_by_bin

        max_samples = 10_000 // self.binner.n_bins_
        rng = np.random.default_rng(seed=random_state)
        for k, v in out_sample_residuals_by_bin.items():
            if len(v) > max_samples:
                sample = rng.choice(a=v, size=max_samples, replace=False)
                out_sample_residuals_by_bin[k] = sample

        bin_keys = (
            []
            if self.binner_intervals_ is None
            else self.binner_intervals_.keys()
        )
        for k in bin_keys:
            if k not in out_sample_residuals_by_bin:
                out_sample_residuals_by_bin[k] = np.array([])

        empty_bins = [
            k for k, v in out_sample_residuals_by_bin.items() 
            if v.size == 0
        ]
        if empty_bins:
            warnings.warn(
                f"The following bins have no out of sample residuals: {empty_bins}. "
                f"No predicted values fall in the interval "
                f"{[self.binner_intervals_[bin] for bin in empty_bins]}. "
                f"Empty bins will be filled with a random sample of residuals.",
                ResidualsUsageWarning
            )
            empty_bin_size = min(max_samples, len(out_sample_residuals))
            for k in empty_bins:
                out_sample_residuals_by_bin[k] = rng.choice(
                    a       = out_sample_residuals,
                    size    = empty_bin_size,
                    replace = False
                )

        if len(out_sample_residuals) > 10_000:
            out_sample_residuals = rng.choice(
                a       = out_sample_residuals, 
                size    = 10_000, 
                replace = False
            )

        self.out_sample_residuals_ = out_sample_residuals
        self.out_sample_residuals_by_bin_ = out_sample_residuals_by_bin

    def get_tags(self) -> dict[str, Any]:
        """
        Return the tags that characterize the behavior of the forecaster.

        Returns
        -------
        skforecast_tags : dict
            Dictionary with forecaster tags.

        """

        return self.__skforecast_tags__

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
