# Contributing to skforecast

## How to Contribute

Skforecast is a community-driven open-source project that relies on contributions from people like you. Every contribution, no matter how big or small, can make a significant impact on the project. Even if you've never contributed to an open-source project before, don't worry! Skforecast is a great place to start. Your help will be appreciated and welcomed with gratitude.

Primarily, skforecast development consists of adding and creating new *Forecasters*, new validation strategies, or improving the performance of the current code. However, there are many other ways to contribute:

- Submit a bug report or feature request on [GitHub Issues](https://github.com/skforecast/skforecast/issues).
- Contribute a Jupyter notebook to our [examples](https://skforecast.org/latest/examples/examples).
- Write [unit or integration tests](https://docs.pytest.org/en/latest/) for our project.
- Answer questions on our issues, Stack Overflow, and elsewhere.
- Translate our documentation into another language.
- Write a blog post, tweet, or share our project with others.

As you can see, there are lots of ways to get involved and we would be very happy for you to join us! Before you start, please open an issue with a brief proposal description so we can align.

Visit our [authors section](https://skforecast.org/latest/authors/authors) to meet all the contributors to skforecast.


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
class ForecasterAutoreg(ForecasterBase):
    """
    This class turns any regressor compatible with the scikit-learn API into a
    recursive autoregressive (multi-step) forecaster.
    
    Parameters
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API
    lags : int, list, numpy ndarray, range
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1. 
    
            - `int`: include lags from 1 to `lags` (included).
            - `list`, `1d numpy ndarray` or `range`: include only lags present in 
            `lags`, all elements must be int.
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
    fit_kwargs : dict, default `None`
        Additional arguments to be passed to the `fit` method of the regressor.
        **New in version 0.8.0**
    forecaster_id : str, int, default `None`
        Name used as an identifier of the forecaster.
        **New in version 0.7.0**
    
    Attributes
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.
    lags : numpy ndarray
        Lags used as predictors.
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
    source_code_weight_func : str
        Source code of the custom function used to create weights.
    max_lag : int
        Maximum value of lag included in `lags`.
    window_size : int
        Size of the window needed to create the predictors. It is equal to `max_lag`.
    last_window : pandas Series
        Last window the forecaster has seen during training. It stores the
        values needed to predict the next `step` immediately after the training data.
    index_type : type
        Type of index of the input used in training.
    index_freq : str
        Frequency of Index of the input used in training.
    training_range : pandas Index
        First and last values of index of the data used during training.
    included_exog : bool
        If the forecaster has been trained using exogenous variable/s.
    exog_type : type
        Type of exogenous data (pandas Series or DataFrame) used in training.
    exog_dtypes : dict
        Type of each exogenous variable/s used in training. If `transformer_exog` 
        is used, the dtypes are calculated after the transformation.
    exog_col_names : list
        Names of columns of `exog` if `exog` used in training was a pandas
        DataFrame.
    X_train_col_names : list
        Names of columns of the matrix created internally for training.
    fit_kwargs : dict
        Additional arguments to be passed to the `fit` method of the regressor.
        **New in version 0.8.0**
    in_sample_residuals : numpy ndarray
        Residuals of the model when predicting training data. Only stored up to
        1000 values. If `transformer_y` is not `None`, residuals are stored in the
        transformed scale.
    out_sample_residuals : numpy ndarray
        Residuals of the model when predicting non training data. Only stored
        up to 1000 values. If `transformer_y` is not `None`, residuals
        are assumed to be in the transformed scale. Use `set_out_sample_residuals` 
        method to set values.
    fitted : bool
        Tag to identify if the regressor has been fitted (trained).
    creation_date : str
        Date of creation.
    fit_date : str
        Date of last fit.
    skforcast_version : str
        Version of skforecast library used to create the forecaster.
    python_version : str
        Version of python used to create the forecaster.
    forecaster_id : str, int default `None`
        Name used as an identifier of the forecaster.
        **New in version 0.7.0**
     
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