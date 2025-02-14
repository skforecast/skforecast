# Contributing to skforecast

## How to Contribute

Skforecast is a community-driven open-source project that relies on contributions from people like you. Every contribution, no matter how big or small, can make a significant impact on the project. Even if you've never contributed to an open-source project before, don't worry! Skforecast is a great place to start. Your help will be appreciated and welcomed with gratitude.

**Recent Enhancement Example**  
*Custom Loss Function Support*  
We recently enhanced model flexibility by enabling custom loss functions in our forecasting models. Contributors can now:
- Pass custom loss functions directly to `create_and_compile()` in ForecasterRNN
- Use either string identifiers for predefined losses or custom callables
- Maintain compatibility with existing compilation arguments
- Validate loss function signatures during model compilation

Primarily, skforecast development consists of adding and creating new *Forecasters*, new validation strategies, or improving the performance of the current code. However, there are many other ways to contribute:

- Submit a bug report or feature request on [GitHub Issues](https://github.com/skforecast/skforecast/issues).
- Contribute a Jupyter notebook to our [examples](https://skforecast.org/latest/examples/examples_english.html).
- Write [unit or integration tests](https://docs.pytest.org/en/latest/) for our project.
- Answer questions on our issues, Stack Overflow, and elsewhere.
- Translate our documentation into another language.
- Write a blog post, tweet, or share our project with others.

As you can see, there are lots of ways to get involved and we would be very happy for you to join us! Before you start, please open an issue with a brief proposal description so we can align.

Visit our [authors section](https://skforecast.org/latest/authors/authors.html) to meet all the contributors to skforecast.


## Testing

To run the test suite, first install the testing dependencies that are located in the main folder:

```bash
$ pip install -r requirements_test.txt
```

All unit tests can be run at once as follows from the root of the project:

```bash
$ pytest -vv
```

Tests take some time to run. Therefore, during normal development, it is recommended to run only the desired tests from the test file being written:

```bash
$ pytest new_module/tests/test_module.py
```

This will go a long way to ensure that the new code does not affect existing library functionality.

## Documentation

Docstring documentation must be included in every class and function. Skforecast uses MkDocs to build the documentation and follows the numpydoc format (as does scikit-learn). The location of the docstring should be just below the class definition, here are two examples:

```python
class ForecasterRecursive(ForecasterBase):
    """
    This class turns any regressor compatible with the scikit-learn API into a
    recursive autoregressive (multi-step) forecaster.
    
    Parameters
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
    lags : int, list, numpy ndarray, range, default `None`
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
    
        - `int`: include lags from 1 to `lags` (included).
        - `list`, `1d numpy ndarray` or `range`: include only lags present in 
        `lags`, all elements must be int.
        - `None`: no lags are included as predictors. 
    window_features : object, list, default `None`
        Instance or list of instances used to create window features. Window features
        are created from the original time series and are included as predictors.
    transformer_y : object transformer (preprocessor), default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
        The transformation is applied to `y` before training the forecaster. 
    transformer_exog : object transformer (preprocessor), default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    weight_func : Callable, default `None`
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        Ignored if `regressor` does not have the argument `sample_weight` in its `fit`
        method. The resulting `sample_weight` cannot have negative values.
    differentiation : int, default `None`
        Order of differencing applied to the time series before training the forecaster.
        If `None`, no differencing is applied. The order of differentiation is the number
        of times the differencing operation is applied to a time series. Differencing
        involves computing the differences between consecutive data points in the series.
        Differentiation is reversed in the output of `predict()` and `predict_interval()`.
        **WARNING: This argument is newly introduced and requires special attention. It
        is still experimental and may undergo changes.**
    fit_kwargs : dict, default `None`
        Additional arguments to be passed to the `fit` method of the regressor.
    binner_kwargs : dict, default `None`
        Additional arguments to pass to the `QuantileBinner` used to discretize 
        the residuals into k bins according to the predicted values associated 
        with each residual. Available arguments are: `n_bins`, `method`, `subsample`,
        `random_state` and `dtype`. Argument `method` is passed internally to the
        fucntion `numpy.percentile`.
        **New in version 0.14.0**
    forecaster_id : str, int, default `None`
        Name used as an identifier of the forecaster.
    
    Attributes
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
    lags : numpy ndarray
        Lags used as predictors.
    lags_names : list
        Names of the lags used as predictors.
    max_lag : int
        Maximum lag included in `lags`.
    window_features : list
        Class or list of classes used to create window features.
    window_features_names : list
        Names of the window features to be included in the `X_train` matrix.
    window_features_class_names : list
        Names of the classes used to create the window features.
    max_size_window_features : int
        Maximum window size required by the window features.
    window_size : int
        The window size needed to create the predictors. It is calculated as the 
        maximum value between `max_lag` and `max_size_window_features`. If 
        differentiation is used, `window_size` is increased by n units equal to 
        the order of differentiation so that predictors can be generated correctly.
    transformer_y : object transformer (preprocessor)
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
        The transformation is applied to `y` before training the forecaster.
    transformer_exog : object transformer (preprocessor)
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.
    weight_func : Callable
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        Ignored if `regressor` does not have the argument `sample_weight` in its `fit`
        method. The resulting `sample_weight` cannot have negative values.
    differentiation : int
        Order of differencing applied to the time series before training the forecaster.
        If `None`, no differencing is applied. The order of differentiation is the number
        of times the differencing operation is applied to a time series. Differencing
        involves computing the differences between consecutive data points in the series.
        Differentiation is reversed in the output of `predict()` and `predict_interval()`.
        **WARNING: This argument is newly introduced and requires special attention. It
        is still experimental and may undergo changes.**
        **New in version 0.10.0**
    binner : sklearn.preprocessing.KBinsDiscretizer
        `KBinsDiscretizer` used to discretize residuals into k bins according 
        to the predicted values associated with each residual.
        **New in version 0.12.0**
    binner_intervals_ : dict
        Intervals used to discretize residuals into k bins according to the predicted
        values associated with each residual.
        **New in version 0.12.0**
    binner_kwargs : dict
        Additional arguments to pass to the `QuantileBinner` used to discretize 
        the residuals into k bins according to the predicted values associated 
        with each residual. Available arguments are: `n_bins`, `method`, `subsample`,
        `random_state` and `dtype`. Argument `method` is passed internally to the
        fucntion `numpy.percentile`.
        **New in version 0.14.0**
    source_code_weight_func : str
        Source code of the custom function used to create weights.
    differentiation : int
        Order of differencing applied to the time series before training the 
        forecaster.
    differentiator : TimeSeriesDifferentiator
        Skforecast object used to differentiate the time series.
    last_window_ : pandas DataFrame
        This window represents the most recent data observed by the predictor
        during its training phase. It contains the values needed to predict the
        next step immediately after the training data. These values are stored
        in the original scale of the time series before undergoing any transformations
        or differentiation. When `differentiation` parameter is specified, the
        dimensions of the `last_window_` are expanded as many values as the order
        of differentiation. For example, if `lags` = 7 and `differentiation` = 1,
        `last_window_` will have 8 values.
    index_type_ : type
        Type of index of the input used in training.
    index_freq_ : str
        Frequency of Index of the input used in training.
    training_range_ : pandas Index
        First and last values of index of the data used during training.
    exog_in_ : bool
        If the forecaster has been trained using exogenous variable/s.
    exog_names_in_ : list
        Names of the exogenous variables used during training.
    exog_type_in_ : type
        Type of exogenous data (pandas Series or DataFrame) used in training.
    exog_dtypes_in_ : dict
        Type of each exogenous variable/s used in training. If `transformer_exog` 
        is used, the dtypes are calculated before the transformation.
    X_train_window_features_names_out_ : list
        Names of the window features included in the matrix `X_train` created
        internally for training.
    X_train_exog_names_out_ : list
        Names of the exogenous variables included in the matrix `X_train` created
        internally for training. It can be different from `exog_names_in_` if
        some exogenous variables are transformed during the training process.
    X_train_features_names_out_ : list
        Names of columns of the matrix created internally for training.
    fit_kwargs : dict
        Additional arguments to be passed to the `fit` method of the regressor.
    in_sample_residuals_ : numpy ndarray
        Residuals of the model when predicting training data. Only stored up to
        10_000 values. If `transformer_y` is not `None`, residuals are stored in
        the transformed scale. If `differentiation` is not `None`, residuals are
        stored after differentiation.
    in_sample_residuals_by_bin_ : dict
        In sample residuals binned according to the predicted value each residual
        is associated with. If `transformer_y` is not `None`, residuals are stored
        in the transformed scale. If `differentiation` is not `None`, residuals are
        stored after differentiation. The number of residuals stored per bin is
        limited to `10_000 // self.binner.n_bins_`.
        **New in version 0.14.0**
    out_sample_residuals_ : numpy ndarray
        Residuals of the model when predicting non training data. Only stored up to
        10_000 values. If `transformer_y` is not `None`, residuals are stored in
        the transformed scale. If `differentiation` is not `None`, residuals are
        stored after differentiation.
    out_sample_residuals_by_bin_ : dict
        Out of sample residuals binned according to the predicted value each residual
        is associated with. If `transformer_y` is not `None`, residuals are stored
        in the transformed scale. If `differentiation` is not `None`, residuals are
        stored after differentiation. The number of residuals stored per bin is
        limited to `10_000 // self.binner.n_bins_`.
        **New in version 0.12.0**
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
    
    """
```

```python
def preprocess_y(
    y: pd.Series
) -> Tuple[np.ndarray, pd.Index]:
    """
    Return values and index of series separately. Index is overwritten 
    according to the next rules:
    
        - If index is of type `DatetimeIndex` and has frequency, nothing is 
        changed.
        - If index is of type `RangeIndex`, nothing is changed.
        - If index is of type `DatetimeIndex` but has no frequency, a 
        `RangeIndex` is created.
        - If index is not of type `DatetimeIndex`, a `RangeIndex` is created.
    
    Parameters
    ----------
    y : pandas Series, pandas DataFrame
        Time series.
    return_values : bool, default `True`
        If `True` return the values of `y` as numpy ndarray. This option is 
        intended to avoid copying data when it is not necessary.

    Returns
    -------
    y_values : None, numpy ndarray
        Numpy array with values of `y`.
    y_index : pandas Index
        Index of `y` modified according to the rules.
    
    """
```