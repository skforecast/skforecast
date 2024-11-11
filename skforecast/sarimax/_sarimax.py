################################################################################
#                                 Sarimax                                      #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Union, Optional
import warnings
import inspect
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _check_fitted(func):
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

        if not self.fitted:
            raise NotFittedError(
                ("Sarimax instance is not fitted yet. Call `fit` with "
                 "appropriate arguments before using this method.")
            )
        
        result = func(self, *args, **kwargs)
        
        return result
    
    return wrapper


class Sarimax(BaseEstimator, RegressorMixin):
    """
    A universal scikit-learn style wrapper for statsmodels SARIMAX.

    This class wraps the statsmodels.tsa.statespace.sarimax.SARIMAX model to 
    follow the scikit-learn style. The following docstring is based on the 
    statmodels documentation and it is highly recommended to visit their site 
    for the best level of detail.

    https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.html

    Parameters
    ----------
    order : tuple, default `(1, 0, 0)`
        The (p,d,q) order of the model for the number of AR parameters, differences, 
        and MA parameters. 
        
        - `d` must be an integer indicating the integration order of the process.
        - `p` and `q` may either be an integers indicating the AR and MA orders 
        (so that all lags up to those orders are included) or else iterables 
        giving specific AR and / or MA lags to include.
    seasonal_order : tuple, default `(0, 0, 0, 0)`
        The (P,D,Q,s) order of the seasonal component of the model for the AR 
        parameters, differences, MA parameters, and periodicity. 
        
        - `D` must be an integer indicating the integration order of the process.
        - `P` and `Q` may either be an integers indicating the AR and MA orders 
        (so that all lags up to those orders are included) or else iterables 
        giving specific AR and / or MA lags to include. 
        - `s` is an integer giving the periodicity (number of periods in season),
        often it is 4 for quarterly data or 12 for monthly data.
    trend : str, default `None`
        Parameter controlling the deterministic trend polynomial `A(t)`.
            
        - `'c'` indicates a constant (i.e. a degree zero component of the 
        trend polynomial).
        - `'t'` indicates a linear trend with time.
        - `'ct'` indicates both, `'c'` and `'t'`. 
        - Can also be specified as an iterable defining the non-zero polynomial 
        exponents to include, in increasing order. For example, `[1,1,0,1]` 
        denotes `a + b*t + ct^3`.
    measurement_error : bool, default `False`
        Whether or not to assume the endogenous observations `y` were measured 
        with error.
    time_varying_regression : bool, default `False`
        Used when an explanatory variables, `exog`, are provided to select whether 
        or not coefficients on the exogenous regressors are allowed to vary over time.
    mle_regression : bool, default `True`
        Whether or not to use estimate the regression coefficients for the
        exogenous variables as part of maximum likelihood estimation or through
        the Kalman filter (i.e. recursive least squares). If 
        `time_varying_regression` is `True`, this must be set to `False`.
    simple_differencing : bool, default `False`
        Whether or not to use partially conditional maximum likelihood
        estimation. 
        
        - If `True`, differencing is performed prior to estimation, which 
        discards the first `s*D + d` initial rows but results in a smaller 
        state-space formulation. 
        - If `False`, the full SARIMAX model is put in state-space form so 
        that all datapoints can be used in estimation.
    enforce_stationarity : bool, default `True`
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model.
    enforce_invertibility : bool, default `True`
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model.
    hamilton_representation : bool, default `False`
        Whether or not to use the Hamilton representation of an ARMA process
        (if `True`) or the Harvey representation (if `False`).
    concentrate_scale : bool, default `False`
        Whether or not to concentrate the scale (variance of the error term)
        out of the likelihood. This reduces the number of parameters estimated
        by maximum likelihood by one, but standard errors will then not
        be available for the scale parameter.
    trend_offset : int, default `1`
        The offset at which to start time trend values. Default is 1, so that
        if `trend='t'` the trend is equal to 1, 2, ..., nobs. Typically is only
        set when the model created by extending a previous dataset.
    use_exact_diffuse : bool, default `False`
        Whether or not to use exact diffuse initialization for non-stationary
        states. Default is `False` (in which case approximate diffuse
        initialization is used).
    method : str, default `'lbfgs'`
        The method determines which solver from scipy.optimize is used, and it 
        can be chosen from among the following strings:

        - `'newton'` for Newton-Raphson
        - `'nm'` for Nelder-Mead
        - `'bfgs'` for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
        - `'lbfgs'` for limited-memory BFGS with optional box constraints
        - `'powell'` for modified Powell`s method
        - `'cg'` for conjugate gradient
        - `'ncg'` for Newton-conjugate gradient
        - `'basinhopping'` for global basin-hopping solver
    maxiter : int, default `50`
        The maximum number of iterations to perform.
    start_params : numpy ndarray, default `None`
        Initial guess of the solution for the loglikelihood maximization. 
        If `None`, the default is given by regressor.start_params.
    disp : bool, default `False`
        Set to `True` to print convergence messages.
    sm_init_kwargs : dict, default `{}`
        Additional keyword arguments to pass to the statsmodels SARIMAX model 
        when it is initialized.
    sm_fit_kwargs : dict, default `{}` 
        Additional keyword arguments to pass to the `fit` method of the
        statsmodels SARIMAX model. The statsmodels SARIMAX.fit parameters 
        `method`, `max_iter`, `start_params` and `disp` have been moved to the 
        initialization of this model and will have priority over those provided 
        by the user using via `sm_fit_kwargs`.
    sm_predict_kwargs : dict, default `{}`
        Additional keyword arguments to pass to the `get_forecast` method of the
        statsmodels SARIMAXResults object.

    Attributes
    ----------
    order : tuple
        The (p,d,q) order of the model for the number of AR parameters, differences, 
        and MA parameters.
    seasonal_order : tuple
        The (P,D,Q,s) order of the seasonal component of the model for the AR 
        parameters, differences, MA parameters, and periodicity.
    trend : str
        Deterministic trend polynomial `A(t)`.
    measurement_error : bool
        Whether or not to assume the endogenous observations `y` were measured 
        with error.
    time_varying_regression : bool
        Used when an explanatory variables, `exog`, are provided to select whether 
        or not coefficients on the exogenous regressors are allowed to vary over time.
    mle_regression : bool
        Whether or not to use estimate the regression coefficients for the
        exogenous variables as part of maximum likelihood estimation or through
        the Kalman filter (i.e. recursive least squares). If 
        `time_varying_regression` is `True`, this must be set to `False`.
    simple_differencing : bool
        Whether or not to use partially conditional maximum likelihood
        estimation.
    enforce_stationarity : bool
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model.
    enforce_invertibility : bool
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model.
    hamilton_representation : bool
        Whether or not to use the Hamilton representation of an ARMA process
        (if `True`) or the Harvey representation (if `False`).
    concentrate_scale : bool
        Whether or not to concentrate the scale (variance of the error term)
        out of the likelihood. This reduces the number of parameters estimated
        by maximum likelihood by one, but standard errors will then not
        be available for the scale parameter.
    trend_offset : int
        The offset at which to start time trend values.
    use_exact_diffuse : bool
        Whether or not to use exact diffuse initialization for non-stationary
        states.
    method : str
        The method determines which solver from scipy.optimize is used.
    maxiter : int
        The maximum number of iterations to perform.
    start_params : numpy ndarray
        Initial guess of the solution for the loglikelihood maximization.
    disp : bool
        Set to `True` to print convergence messages.
    sm_init_kwargs : dict
        Additional keyword arguments to pass to the statsmodels SARIMAX model 
        when it is initialized.
    sm_fit_kwargs : dict
        Additional keyword arguments to pass to the `fit` method of the
        statsmodels SARIMAX model.
    sm_predict_kwargs : dict
        Additional keyword arguments to pass to the `get_forecast` method of the
        statsmodels SARIMAXResults object.
    _sarimax_params : dict
        Parameters of this model that can be set with the `set_params` method.
    output_type : str
        Format of the object returned by the predict method. This is set 
        automatically according to the type of `y` used in the fit method to 
        train the model, `'numpy'` or `'pandas'`.
    sarimax : object
        The statsmodels.tsa.statespace.sarimax.SARIMAX object created.
    fitted : bool
        Tag to identify if the regressor has been fitted (trained).
    sarimax_res : object
        The resulting statsmodels.tsa.statespace.sarimax.SARIMAXResults object 
        created by statsmodels after fitting the SARIMAX model.
    training_index : pandas Index
        Index of the training series as long as it is a pandas Series or Dataframe.

    """

    def __init__(
        self,
        order: tuple = (1, 0, 0),
        seasonal_order: tuple = (0, 0, 0, 0),
        trend: str = None,
        measurement_error: bool = False,
        time_varying_regression: bool = False,
        mle_regression: bool = True,
        simple_differencing: bool = False,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        hamilton_representation: bool = False,
        concentrate_scale: bool = False,
        trend_offset: int = 1,
        use_exact_diffuse: bool = False,
        dates = None,
        freq = None,
        missing = 'none',
        validate_specification: bool = True,
        method: str = 'lbfgs',
        maxiter: int = 50,
        start_params: np.ndarray = None,
        disp: bool = False,
        sm_init_kwargs: dict = {},
        sm_fit_kwargs: dict = {},
        sm_predict_kwargs: dict = {}
    ) -> None:

        self.order                   = order
        self.seasonal_order          = seasonal_order
        self.trend                   = trend
        self.measurement_error       = measurement_error
        self.time_varying_regression = time_varying_regression
        self.mle_regression          = mle_regression
        self.simple_differencing     = simple_differencing
        self.enforce_stationarity    = enforce_stationarity
        self.enforce_invertibility   = enforce_invertibility
        self.hamilton_representation = hamilton_representation
        self.concentrate_scale       = concentrate_scale
        self.trend_offset            = trend_offset
        self.use_exact_diffuse       = use_exact_diffuse
        self.dates                   = dates
        self.freq                    = freq
        self.missing                 = missing
        self.validate_specification  = validate_specification
        self.method                  = method
        self.maxiter                 = maxiter
        self.start_params            = start_params
        self.disp                    = disp

        # Create the dictionaries with the additional statsmodels parameters to be  
        # used during the init, fit and predict methods. Note that the statsmodels 
        # SARIMAX.fit parameters `method`, `max_iter`, `start_params` and `disp` 
        # have been moved to the initialization of this model and will have 
        # priority over those provided by the user using via `sm_fit_kwargs`.
        self.sm_init_kwargs    = sm_init_kwargs
        self.sm_fit_kwargs     = sm_fit_kwargs
        self.sm_predict_kwargs = sm_predict_kwargs

        # Params that can be set with the `set_params` method
        _, _, _, _sarimax_params = inspect.getargvalues(inspect.currentframe())
        _sarimax_params.pop("self")
        self._sarimax_params = _sarimax_params

        self._consolidate_kwargs()

        # Create Results Attributes 
        self.output_type    = None
        self.sarimax        = None
        self.fitted         = False
        self.sarimax_res    = None
        self.training_index = None


    def __repr__(
        self
    ) -> str:
        """
        Information displayed when a Sarimax object is printed.
        """

        p, d, q = self.order
        P, D, Q, m = self.seasonal_order
        
        return f"Sarimax({p},{d},{q})({P},{D},{Q})[{m}]"


    def _consolidate_kwargs(
        self
    ) -> None:
        """
        Create the dictionaries to be used during the init, fit, and predict methods.
        Note that the parameters in this model's initialization take precedence 
        over those provided by the user using via the statsmodels kwargs dicts.

        Parameters
        ----------
        self
        
        Returns
        -------
        None
        
        """
        
        # statsmodels.tsa.statespace.SARIMAX parameters
        _init_kwargs = self.sm_init_kwargs.copy()
        _init_kwargs.update({
           'order': self.order,
           'seasonal_order': self.seasonal_order,
           'trend': self.trend,
           'measurement_error': self.measurement_error,
           'time_varying_regression': self.time_varying_regression,
           'mle_regression': self.mle_regression,
           'simple_differencing': self.simple_differencing,
           'enforce_stationarity': self.enforce_stationarity,
           'enforce_invertibility': self.enforce_invertibility,
           'hamilton_representation': self.hamilton_representation,
           'concentrate_scale': self.concentrate_scale,
           'trend_offset': self.trend_offset,
           'use_exact_diffuse': self.use_exact_diffuse,
           'dates': self.dates,
           'freq': self.freq,
           'missing': self.missing,
           'validate_specification': self.validate_specification
        })
        self._init_kwargs = _init_kwargs

        # statsmodels.tsa.statespace.SARIMAX.fit parameters
        _fit_kwargs = self.sm_fit_kwargs.copy()
        _fit_kwargs.update({
           'method': self.method,
           'maxiter': self.maxiter,
           'start_params': self.start_params,
           'disp': self.disp,
        })        
        self._fit_kwargs = _fit_kwargs

        # statsmodels.tsa.statespace.SARIMAXResults.get_forecast parameters
        self._predict_kwargs = self.sm_predict_kwargs.copy()


    def _create_sarimax(
        self,
        endog: Union[np.ndarray, pd.Series, pd.DataFrame],
        exog: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None
    ) -> None:
        """
        A helper method to create a new statsmodel SARIMAX model.

        Additional keyword arguments to pass to the statsmodels SARIMAX model 
        when it is initialized can be added with the `init_kwargs` argument 
        when initializing the model.

        Parameters
        ----------
        endog : numpy ndarray, pandas Series, pandas DataFrame
            The endogenous variable.
        exog : numpy ndarray, pandas Series, pandas DataFrame, default `None`
            The exogenous variables.
        
        Returns
        -------
        None

        """

        self.sarimax = SARIMAX(endog=endog, exog=exog, **self._init_kwargs)


    def fit(
        self,
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        exog: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None
    ) -> None:
        """
        Fit the model to the data.

        Additional keyword arguments to pass to the `fit` method of the
        statsmodels SARIMAX model can be added with the `fit_kwargs` argument 
        when initializing the model.

        Parameters
        ----------
        y : numpy ndarray, pandas Series, pandas DataFrame
            Training time series.
        exog : numpy ndarray, pandas Series, pandas DataFrame, default `None`
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and their indexes must be aligned so
            that y[i] is regressed on exog[i].

        Returns
        -------
        None

        """

        # Reset values in case the model has already been fitted.
        self.output_type    = None
        self.sarimax_res    = None
        self.fitted         = False
        self.training_index = None

        self.output_type = 'numpy' if isinstance(y, np.ndarray) else 'pandas'
        
        self._create_sarimax(endog=y, exog=exog)
        self.sarimax_res = self.sarimax.fit(**self._fit_kwargs)
        self.fitted = True
        
        if self.output_type == 'pandas':
            self.training_index = y.index


    @_check_fitted
    def predict(
        self,
        steps: int,
        exog: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None, 
        return_conf_int: bool = False,
        alpha: float = 0.05
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Forecast future values and, if desired, their confidence intervals.

        Generate predictions (forecasts) n steps in the future with confidence
        intervals. Note that if exogenous variables were used in the model fit, 
        they will be expected for the predict procedure and will fail otherwise.

        Additional keyword arguments to pass to the `get_forecast` method of the
        statsmodels SARIMAX model can be added with the `predict_kwargs` argument 
        when initializing the model.

        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        exog : numpy ndarray, pandas Series, pandas DataFrame, default `None`
            Value of the exogenous variable/s for the next steps. The number of 
            observations needed is the number of steps to predict. 
        return_conf_int : bool, default `False`
            Whether to get the confidence intervals of the forecasts.
        alpha : float, default `0.05`
            The confidence intervals for the forecasts are (1 - alpha) %.

        Returns
        -------
        predictions : numpy ndarray, pandas DataFrame
            Values predicted by the forecaster and their estimated interval. The 
            output type is the same as the type of `y` used in the fit method.

            - pred: predictions.
            - lower_bound: lower bound of the interval. (if `return_conf_int`)
            - upper_bound: upper bound of the interval. (if `return_conf_int`)

        """

        # This is done because statsmodels doesn't allow `exog` length greater than
        # the number of steps
        if exog is not None and len(exog) > steps:
            warnings.warn(
                (f"When predicting using exogenous variables, the `exog` parameter "
                 f"must have the same length as the number of predicted steps. Since "
                 f"len(exog) > steps, only the first {steps} observations are used.")
            )
            exog = exog[:steps]

        predictions = self.sarimax_res.get_forecast(
                          steps = steps,
                          exog  = exog,
                          **self._predict_kwargs
                      )
        
        if not return_conf_int:
            predictions = predictions.predicted_mean
            if self.output_type == 'pandas':
                predictions = predictions.rename("pred").to_frame()
        else:
            if self.output_type == 'numpy':
                predictions = np.column_stack(
                                  [predictions.predicted_mean,
                                   predictions.conf_int(alpha=alpha)]
                              )
            else:
                predictions = pd.concat((
                                  predictions.predicted_mean,
                                  predictions.conf_int(alpha=alpha)),
                                  axis = 1
                              )
                predictions.columns = ['pred', 'lower_bound', 'upper_bound']

        return predictions


    @_check_fitted
    def append(
        self,
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        exog: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
        refit: bool = False,
        copy_initialization: bool = False,
        **kwargs
    ) -> None:
        """
        Recreate the results object with new data appended to the original data.

        Creates a new result object applied to a dataset that is created by 
        appending new data to the end of the model's original data. The new 
        results can then be used for analysis or forecasting.

        Parameters
        ----------
        y : numpy ndarray, pandas Series, pandas DataFrame
            New observations from the modeled time-series process.
        exog : numpy ndarray, pandas Series, pandas DataFrame, default `None`
            New observations of exogenous regressors, if applicable. Must have 
            the same number of observations as `y` and their indexes must be 
            aligned so that y[i] is regressed on exog[i].
        refit : bool, default `False`
            Whether to re-fit the parameters, based on the combined dataset.
        copy_initialization : bool, default `False`
            Whether or not to copy the initialization from the current results 
            set to the new model. 
        **kwargs
            Keyword arguments may be used to modify model specification arguments 
            when created the new model object.

        Returns
        -------
        None

        Notes
        -----
        The `y` and `exog` arguments to this method must be formatted in the same 
        way (e.g. Pandas Series versus Numpy array) as were the `y` and `exog` 
        arrays passed to the original model.

        The `y` argument to this method should consist of new observations that 
        occurred directly after the last element of `y`. For any other kind of 
        dataset, see the apply method.

        This method will apply filtering to all of the original data as well as 
        to the new data. To apply filtering only to the new data (which can be 
        much faster if the original dataset is large), see the extend method.

        https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.mlemodel.MLEResults.append.html#statsmodels.tsa.statespace.mlemodel.MLEResults.append

        """

        fit_kwargs = self._fit_kwargs if refit else None

        self.sarimax_res = self.sarimax_res.append(
                               endog               = y,
                               exog                = exog,
                               refit               = refit,
                               copy_initialization = copy_initialization,
                               fit_kwargs          = fit_kwargs,
                               **kwargs
                           )


    @_check_fitted
    def apply(
        self,
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        exog: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
        refit: bool = False,
        copy_initialization: bool = False,
        **kwargs
    ) -> None:
        """
        Apply the fitted parameters to new data unrelated to the original data.

        Creates a new result object using the current fitted parameters, applied 
        to a completely new dataset that is assumed to be unrelated to the model's
        original data. The new results can then be used for analysis or forecasting.

        Parameters
        ----------
        y : numpy ndarray, pandas Series, pandas DataFrame
            New observations from the modeled time-series process.
        exog : numpy ndarray, pandas Series, pandas DataFrame, default `None`
            New observations of exogenous regressors, if applicable. Must have 
            the same number of observations as `y` and their indexes must be 
            aligned so that y[i] is regressed on exog[i].
        refit : bool, default `False`
            Whether to re-fit the parameters, using the new dataset.
        copy_initialization : bool, default `False`
            Whether or not to copy the initialization from the current results 
            set to the new model. 
        **kwargs
            Keyword arguments may be used to modify model specification arguments 
            when created the new model object.

        Returns
        -------
        None

        Notes
        -----
        The `y` argument to this method should consist of new observations that 
        are not necessarily related to the original model's `y` dataset. For 
        observations that continue that original dataset by follow directly after 
        its last element, see the append and extend methods.

        https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.mlemodel.MLEResults.apply.html#statsmodels.tsa.statespace.mlemodel.MLEResults.apply

        """

        fit_kwargs = self._fit_kwargs if refit else None

        self.sarimax_res = self.sarimax_res.apply(
                               endog               = y,
                               exog                = exog,
                               refit               = refit,
                               copy_initialization = copy_initialization,
                               fit_kwargs          = fit_kwargs,
                               **kwargs
                           )


    @_check_fitted
    def extend(
        self,
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        exog: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
        **kwargs
    ) -> None:
        """
        Recreate the results object for new data that extends the original data.

        Creates a new result object applied to a new dataset that is assumed to 
        follow directly from the end of the model's original data. The new 
        results can then be used for analysis or forecasting.

        Parameters
        ----------
        y : numpy ndarray, pandas Series, pandas DataFrame
            New observations from the modeled time-series process.
        exog : numpy ndarray, pandas Series, pandas DataFrame, default `None`
            New observations of exogenous regressors, if applicable. Must have 
            the same number of observations as `y` and their indexes must be 
            aligned so that y[i] is regressed on exog[i].
        **kwargs
            Keyword arguments may be used to modify model specification arguments 
            when created the new model object.

        Returns
        -------
        None

        Notes
        -----
        The `y` argument to this method should consist of new observations that 
        occurred directly after the last element of the model's original `y` 
        array. For any other kind of dataset, see the apply method.

        This method will apply filtering only to the new data provided by the `y` 
        argument, which can be much faster than re-filtering the entire dataset. 
        However, the returned results object will only have results for the new 
        data. To retrieve results for both the new data and the original data, 
        see the append method.

        https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.mlemodel.MLEResults.extend.html#statsmodels.tsa.statespace.mlemodel.MLEResults.extend

        """

        self.sarimax_res = self.sarimax_res.extend(
                               endog = y,
                               exog  = exog,
                               **kwargs
                           )


    def set_params(
        self, 
        **params: dict
    ) -> None:
        """
        Set new values to the parameters of the regressor.
        
        Parameters
        ----------
        params : dict
            Parameters values.

        Returns
        -------
        None

        """
        
        params = {k: v for k, v in params.items() if k in self._sarimax_params}
        for key, value in params.items():
            setattr(self, key, value)

        self._consolidate_kwargs()

        # Reset values in case the model has already been fitted.
        self.output_type    = None
        self.sarimax_res    = None
        self.fitted         = False
        self.training_index = None


    @_check_fitted
    def params(
        self
    ) -> Union[np.ndarray, pd.Series]:
        """
        Get the parameters of the model. The order of variables is the trend
        coefficients, the `k_exog` exogenous coefficients, the `k_ar` AR 
        coefficients, and finally the `k_ma` MA coefficients.

        Returns
        -------
        params : numpy ndarray, pandas Series
            The parameters of the model.
        
        """

        return self.sarimax_res.params


    @_check_fitted
    def summary(
        self,
        alpha: float = 0.05,
        start: int = None
    ) -> object:
        """
        Get a summary of the SARIMAXResults object.
        
        Parameters
        ----------
        alpha : float, default `0.05`
            The confidence intervals for the forecasts are (1 - alpha) %.
        start : int, default `None`
            Integer of the start observation.

        Returns
        -------
        summary : Summary instance
            This holds the summary table and text, which can be printed or 
            converted to various output formats.
        
        """

        return self.sarimax_res.summary(alpha=alpha, start=start)


    @_check_fitted
    def get_info_criteria(
        self,
        criteria: str = 'aic',
        method: str = 'standard'
    ) -> float:
        """
        Get the selected information criteria.

        Check https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.info_criteria.html
        to know more about statsmodels info_criteria method.

        Parameters
        ----------
        criteria : str, default `'aic'`
            The information criteria to compute. Valid options are {'aic', 'bic',
            'hqic'}.
        method : str, default `'standard'`
            The method for information criteria computation. Default is 'standard'
            method; 'lutkepohl' computes the information criteria as in Lütkepohl
            (2007).

        Returns
        -------
        metric : float
            The value of the selected information criteria.
        
        """

        if criteria not in ['aic', 'bic', 'hqic']:
            raise ValueError(
                ("Invalid value for `criteria`. Valid options are 'aic', 'bic', "
                 "and 'hqic'.")
            )
        
        if method not in ['standard', 'lutkepohl']:
            raise ValueError(
                ("Invalid value for `method`. Valid options are 'standard' and "
                 "'lutkepohl'.")
            )
        
        metric = self.sarimax_res.info_criteria(criteria=criteria, method=method)
        
        return metric