# Changelog

All significant changes to this project are documented in this release file.

| Legend                                                     |                                       |
|:-----------------------------------------------------------|:--------------------------------------|
| <span class="badge text-bg-feature">Feature</span>         | New feature                           |
| <span class="badge text-bg-enhancement">Enhancement</span> | Improvement in existing functionality |
| <span class="badge text-bg-api-change">API Change</span>   | Changes in the API                    |
| <span class="badge text-bg-danger">Fix</span>              | Bug fix                               |


## 0.14.0 <small>In development</small> { id="0.14.0" }

The main changes in this release are:

This release has undergone a major refactoring to improve the performance of the library. Visit the [migration guide](https://skforecast.org/latest/user_guides/migration-guide.html) section for more information.

+ <span class="badge text-bg-feature">Feature</span> Window features can be added to the training matrix using the `window_features` argument in all forecasters. You can use the <code>[RollingFeatures]</code> class to create these features or create your own object.

+ <span class="badge text-bg-feature">Feature</span> <code>[model_selection]</code> functions now have a new argument `cv`. This argument expect an object of type <code>[TimeSeriesFold]</code> or <code>[OneStepAheadFold]</code> which allows to define the validation strategy using the arguments `initial_train_size`, `steps`, `gap`, `refit`, `fixed_train_size`, `skip_folds` and `allow_incomplete_folds`.

+ <span class="badge text-bg-feature">Feature</span> Hyperparameter search now allows to follow a one-step-ahead validation strategy using a <code>[OneStepAheadFold]</code> as `cv` argument in the <code>[model_selection]</code> functions.

+ <span class="badge text-bg-enhancement">Enhancement</span> Refactor the prediction process in <code>[ForecasterRecursiveMultiSeries]</code> to improve performance when predicting multiple series.

+ <span class="badge text-bg-enhancement">Enhancement</span> The bootstrapping process in the `predict_bootstrapping` method of all forecasters has been optimized to improve performance. This may result in slightly different results when using the same seed as in previous versions.

+ <span class="badge text-bg-api-change">API Change</span> package structure has been changed to improve code organization. The forecasters have been grouped into the `recursive`, `direct` amd `deep_learning` modules. Visit the [migration guide](https://skforecast.org/latest/user_guides/migration-guide.html) section for more information.

+ <span class="badge text-bg-api-change">API Change</span> <code>[ForecasterAutoregCustom]</code> has been deprecated. Window features can be added using the `window_features` argument in the <code>[ForecasterRecursive]</code>.

+ <span class="badge text-bg-api-change">API Change</span> The `pmdarima.ARIMA` regressor is no longer supported by the <code>[ForecasterSarimax]</code>. You can use the skforecast <code>[Sarimax]</code> model or, to continue using it, use skforecast 0.13.0 or lower.

+ <span class="badge text-bg-danger">Fix</span> Fixed a bug where the `create_predict_X` method in recursive Forecasters did not correctly generate the matrix correctly when using transformations and/or differentiations


**Added**

+ Added `numba>=0.59` as hard dependency.

+ Added `window_features` argument to all forecasters. This argument allows the user to add window features to the training matrix. See <code>[RollingFeatures]</code>.

+ Hyperparameter search now allows to follow a one-step-ahead validation strategy using a <code>[OneStepAheadFold]</code> as `cv` argument in the <code>[model_selection]</code> functions.

+ Differentiation has been extended to all forecasters. The `differentiation` argument has been added to all forecasters to model the n-order differentiated time series.

+ Create `transform_numpy` function in the <code>[utils]</code> module to carry out the transformation of the modeled time series and exogenous variables as numpy arrays.

+ `random_state` argument in the `fit` method of <code>[ForecasterRecursive]</code> to set a seed for the random generator so that the stored sample residuals are always deterministic.

+ New private method `_train_test_split_one_step_ahead` in all forecasters.

+ New private function `_calculate_metrics_one_step_ahead` to <code>[model_selection]</code> module to calculate the metrics when predicting one step ahead.

+ The `steps` argument in the predict method of the <code>[ForecasterRecursive]</code> can now be a str or a pandas datetime. If so, the method will predict up to the specified date. (contribution by @imMoya).


**Changed**

+ <code>[ForecasterAutoregCustom]</code> has been deprecated. Window features can be added using the `window_features` argument in the <code>[ForecasterRecursive]</code>.

+ Refactor `recursive_predict` in <code>[ForecasterRecursiveMultiSeries]</code> to predict all series at once and include option of adding residuals. This improves performance when predicting multiple series.

+ Refactor `predict_bootstrapping` in all Forecasters. The bootstrapping process has been optimized to improve performance. This may result in slightly different results when using the same seed as in previous versions.

+ Change the default value of `encoding` to `ordinal` in <code>[ForecasterRecursiveMultiSeries]</code>. This will avoid conflicts if the regressor does not support categorical variables by default.

+ Removed argument `engine` from <code>[bayesian_search_forecaster]</code> and <code>[bayesian_search_forecaster_multiseries]</code>.

+ The `pmdarima.ARIMA` regressor is no longer supported by the <code>[ForecasterSarimax]</code>. You can use the skforecast <code>[Sarimax]</code> model or, to continue using it, use skforecast 0.13.0 or lower.

+ `initialize_lags` in <code>[utils]</code> now returns the maximum lag, `max_lag`.

+ Removed attribute `window_size_diff` from all Forecasters. The window size extended by the order of differentiation is now calculated on `window_size`.

+ `lags` can be `None` when initializing any Forecaster that includes window features.

+ <code>[model_selection]</code> module has been divided internally into different modules to improve code organization (`_validation`, `_search`, `_split`).

+ Functions from `model_selection_multiseries` and `model_selection_sarimax` modules have been moved to the <code>[model_selection]</code> module.

+ <code>[model_selection]</code> functions now have a new argument `cv`. This argument expect an object of type <code>[TimeSeriesFold]</code> or <code>[OneStepAheadFold]</code> which allows to define the validation strategy using the arguments `initial_train_size`, `steps`, `gap`, `refit`, `fixed_train_size`, `skip_folds` and `allow_incomplete_folds`.

+ Added <code>[feature_selection]</code> module. The functions <code>[select_features]</code> and <code>[select_features_multiseries]</code> have been moved to this module.

+ `exog_to_direct` and `exog_to_direct_numpy` in <code>[utils]</code> now returns a the names of the columns of the transformed exogenous variables.

+ Renamed attributes in all Forecasters:

    + `encoding_mapping` has been renamed to `encoding_mapping_`.

    + `last_window` has been renamed to `last_window_`.

    + `index_type` has been renamed to `index_type_`.

    + `index_freq` has been renamed to `index_freq_`.

    + `training_range` has been renamed to `training_range_`.

    + `series_col_names` has been renamed to `series_names_in_`.

    + `included_exog` has been renamed to `exog_in_`.

    + `exog_type` has been renamed to `exog_type_in_`.

    + `exog_dtypes` has been renamed to `exog_dtypes_in_`.

    + `exog_col_names` has been renamed to `exog_names_in_`.

    + `series_X_train` has been renamed to `X_train_series_names_in_`.

    + `X_train_col_names` has been renamed to `X_train_features_names_out_`.

    + `binner_intervals` has been renamed to `binner_intervals_`.

    + `in_sample_residuals` has been renamed to `in_sample_residuals_`.

    + `out_sample_residuals` has been renamed to `out_sample_residuals_`.

    + `fitted` has been renamed to `is_fitted`.

+ Renamed arguments in different functions and methods:

    + `in_sample_residuals` has been renamed to `use_in_sample_residuals`.

    + `binned_residuals` has been renamed to `use_binned_residuals`.

    + `series_col_names` has been renamed to `series_names_in_` in the `check_predict_input`, `check_preprocess_exog_multiseries` and `initialize_transformer_series` functions in the <code>[utils]</code> module.

    + `series_X_train` has been renamed to `X_train_series_names_in_` in the `prepare_levels_multiseries` function in the <code>[utils]</code> module.

    + `exog_col_names` has been renamed to `exog_names_in_` in the `check_predict_input` and `check_preprocess_exog_multiseries` functions in the <code>[utils]</code> module.

    + `index_type` has been renamed to `index_type_` in the `check_predict_input` function in the <code>[utils]</code> module.

    + `index_freq` has been renamed to `index_freq_` in the `check_predict_input` function in the <code>[utils]</code> module.

    + `included_exog` has been renamed to `exog_in_` in the `check_predict_input` function in the <code>[utils]</code> module.

    + `exog_type` has been renamed to `exog_type_in_` in the `check_predict_input` function in the <code>[utils]</code> module.

    + `exog_dtypes` has been renamed to `exog_dtypes_in_` in the `check_predict_input` function in the <code>[utils]</code> module.

    + `fitted` has been renamed to `is_fitted` in the `check_predict_input` function in the <code>[utils]</code> module.

    + `use_in_sample` has been renamed to `use_in_sample_residuals` in the `prepare_residuals_multiseries` function in the <code>[utils]</code> module.

    + `in_sample_residuals` has been renamed to `use_in_sample_residuals` in the <code>[backtesting_forecaster]</code>, <code>[backtesting_forecaster_multiseries]</code> and `check_backtesting_input` (<code>[utils]</code> module) functions.

   + `binned_residuals` has been renamed to `use_binned_residuals` in the <code>[backtesting_forecaster]</code> function.

    + `in_sample_residuals` has been renamed to `in_sample_residuals_` in the `prepare_residuals_multiseries` function in the <code>[utils]</code> module.

    + `out_sample_residuals` has been renamed to `out_sample_residuals_` in the `prepare_residuals_multiseries` function in the <code>[utils]</code> module.

    + `last_window` has been renamed to `last_window_` in the `preprocess_levels_self_last_window_multiseries` function in the <code>[utils]</code> module.


**Fixed**

+ Fixed a bug where the `create_predict_X` method in recursive Forecasters did not correctly generate the matrix correctly when using transformations and/or differentiations.


## 0.13.0 <small>Aug 01, 2024</small> { id="0.13.0" }

The main changes in this release are:

+ <span class="badge text-bg-feature">Feature</span> Global Forecasters <code>[ForecasterAutoregMultiSeries]</code> and <code>[ForecasterAutoregMultiSeriesCustom]</code> are able to [predict series not seen during training](https://skforecast.org/latest/user_guides/independent-multi-time-series-forecasting.html#forecasting-unknown-series). This is useful when the user wants to predict a new series that was not included in the training data.

+ <span class="badge text-bg-feature">Feature</span> `encoding` can be set to `None` in Global Forecasters <code>[ForecasterAutoregMultiSeries]</code> and <code>[ForecasterAutoregMultiSeriesCustom]</code>. This option does [not add the encoded series ids](https://skforecast.org/latest/user_guides/independent-multi-time-series-forecasting#series-encoding-in-multi-series) to the regressor training matrix.

+ <span class="badge text-bg-feature">Feature</span> New `create_predict_X` method in all recursive and direct Forecasters to allow the user to inspect the matrix passed to the predict method of the regressor.

+ <span class="badge text-bg-feature">Feature</span> New module <code>[metrics]</code> with functions to calculate metrics for time series forecasting such as <code>[mean_absolute_scaled_error]</code> and <code>[root_mean_squared_scaled_error]</code>. Visit [Time Series Forecasting Metrics](https://skforecast.org/latest/user_guides/metrics.html) for more information.

+ <span class="badge text-bg-feature">Feature</span> New argument `add_aggregated_metric` in <code>[backtesting_forecaster_multiseries]</code> to include, in addition to the metrics for each level, the aggregated metric of all levels using the average (arithmetic mean), weighted average (weighted by the number of predicted values of each level) or pooling (the values of all levels are pooled and then the metric is calculated).

+ <span class="badge text-bg-feature">Feature</span> New argument `skip_folds` in <code>[model_selection]</code> and <code>[model_selection_multiseries]</code> functions. It allows the user to [skip some folds during backtesting](https://skforecast.org/latest/user_guides/backtesting#backtesting-with-skip-folds), which can be useful to speed up the backtesting process and thus the hyperparameter search.

+ <span class="badge text-bg-api-change">API Change</span> backtesting procedures now pass the training series to the metric functions so it can be used to calculate metrics that depend on the training series.

+ <span class="badge text-bg-api-change">API Change</span> Changed the default value of the `transformer_series` argument to `None` in the Global Forecasters <code>[ForecasterAutoregMultiSeries]</code> and <code>[ForecasterAutoregMultiSeriesCustom]</code>. In most cases, tree-based models are used as regressors in these forecasters, so no transformation is applied by default as it is not necessary.

**Added**

+ Support for `python 3.12`.

+ `keras` has been added as an optional dependency, tag deeplearning, to use the <code>[ForecasterRnn]</code>.

+ `PyTorch` backend for the <code>[ForecasterRnn]</code>.

+ New `create_predict_X` method in all recursive and direct Forecasters to allow the user to inspect the matrix passed to the predict method of the regressor.

+ New `_create_predict_inputs` method in all Forecasters to unify the inputs of the predict methods.

+ New plot function <code>[plot_prediction_intervals]</code> in the <code>[plot]</code> module to plot predicted intervals.

+ New module <code>[metrics]</code> with functions to calculate metrics for time series forecasting such as `mean_absolute_scaled_error` and `root_mean_squared_scaled_error`.

+ New argument `skip_folds` in <code>[model_selection]</code> and <code>[model_selection_multiseries]</code> functions. It allows the user to skip some folds during backtesting, which can be useful to speed up the backtesting process and thus the hyperparameter search.

+ New function <code>[plot_prediction_intervals]</code> in module <code>[plot]</code>.

+ Global Forecasters <code>[ForecasterAutoregMultiSeries]</code> and <code>[ForecasterAutoregMultiSeriesCustom]</code> are able to predict series not seen during training. This is useful when the user wants to predict a new series that was not included in the training data.

+ `encoding` can be set to `None` in Global Forecasters <code>[ForecasterAutoregMultiSeries]</code> and <code>[ForecasterAutoregMultiSeriesCustom]</code>. This option does not add the encoded series ids to the regressor training matrix.

+ New argument `add_aggregated_metric` in <code>[backtesting_forecaster_multiseries]</code> to include, in addition to the metrics for each level, the aggregated metric of all levels using the average (arithmetic mean), weighted average (weighted by the number of predicted values of each level) or pooling (the values of all levels are pooled and then the metric is calculated).

+ New argument `aggregate_metric` in <code>[grid_search_forecaster_multiseries]</code>, <code>[random_search_forecaster_multiseries]</code> and <code>[bayesian_search_forecaster_multiseries]</code> to select the aggregation method used to combine the metric(s) of all levels during the hyperparameter search. The available methods are: mean (arithmetic mean), weighted (weighted by the number of predicted values of each level) and pool (the values of all levels are pooled and then the metric is calculated). If more than one metric and/or aggregation method is used, all are reported in the results, but the first of each is used to select the best model.

+ New class <code>[DateTimeFeatureTransformer]</code> and function <code>[create_datetime_features]</code> in the <code>[preprocessing]</code> module to create datetime and calendar features from a datetime index.

**Changed**

+ Deprecated `python 3.8` compatibility.

+ Update [project dependencies](https://skforecast.org/latest/quick-start/how-to-install).

+ Change default value of `n_bins` when initializing <code>[ForecasterAutoreg]</code> from 15 to 10.

+ Refactor `_recursive_predict` in all recursive forecasters.

+ Change default value of `transformer_series` when initializing <code>[ForecasterAutoregMultiSeries]</code> and <code>[ForecasterAutoregMultiSeriesCustom]</code> from `StandardScaler()` to `None`.

+ Function `_get_metric` moved from <code>[model_selection]</code> to <code>[metrics]</code>.

+ Change information message when `verbose` is `True` in <code>[backtesting_forecaster]</code> and <code>[backtesting_forecaster_multiseries]</code>.

+ `select_n_jobs_backtesting` and `select_n_jobs_fit` in <code>[utils]</code> return `n_jobs = 1` if regressor is `LGBMRegressor`. This is because `lightgbm` is highly optimized for gradient boosting and parallelizes operations at a very fine-grained level, making additional parallelization unnecessary and potentially harmful due to resource contention.

+ `metric_values` returned by <code>[backtesting_forecaster]</code> and <code>[backtesting_sarimax]</code> is a `pandas DataFrame` with one column per metric instead of a `list`.

**Fixed**

+ Bug fix in <code>[backtesting_forecaster_multiseries]</code> using a <code>[ForecasterAutoregMultiSeries]</code> or <code>[ForecasterAutoregMultiSeriesCustom]</code> that includes differentiation.


## 0.12.1 <small>May 20, 2024</small> { id="0.12.1" }

<span class="badge text-bg-danger">Fix</span> This is a minor release to fix a bug.

**Added**


**Changed**


**Fixed**

+ Bug fix when storing `last_window` using a [`ForecasterAutoregMultiSeries`] that includes differentiation.


## 0.12.0 <small>May 05, 2024</small> { id="0.12.0" }

The main changes in this release are:

+ <span class="badge text-bg-feature">Feature</span> Multiseries forecaster (Global Models) can be trained using [series of different lengths and with different exogenous variables](https://skforecast.org/latest/user_guides/multi-series-with-different-length-and-different_exog) per series.

+ <span class="badge text-bg-feature">Feature</span> New functionality to [select features](https://skforecast.org/latest/user_guides/feature-selection) using scikit-learn selectors ([`select_features`](https://skforecast.org/latest/api/model_selection#skforecast.model_selection.model_selection.select_features) and [`select_features_multiseries`](https://skforecast.org/latest/api/model_selection_multiseries#skforecast.model_selection_multiseries.model_selection_multiseries.select_features_multiseries)).

+ <span class="badge text-bg-feature">Feature</span> Added new forecaster [`ForecasterRnn`](https://skforecast.org/latest/api/forecasterrnn) to create forecasting models based on [deep learning](https://skforecast.org/latest/user_guides/forecasting-with-deep-learning-rnn-lstm) (RNN and LSTM).

+ <span class="badge text-bg-feature">Feature</span> New method to [predict intervals conditioned on the range of the predicted values](https://skforecast.org/latest/user_guides/probabilistic-forecasting#intervals-conditioned-on-predicted-values-binned-residuals). This is can help to improve the interval coverage when the residuals are not homoscedastic ([`ForecasterAutoreg`](https://skforecast.org/latest/api/forecasterautoreg)).

+ <span class="badge text-bg-enhancement">Enhancement</span> [Bayesian hyperparameter search](https://skforecast.org/latest/user_guides/independent-multi-time-series-forecasting#hyperparameter-tuning-and-lags-selection-multi-series) is now available for all multiseries forecasters using `optuna` as the search engine.

+ <span class="badge text-bg-enhancement">Enhancement</span> All Recursive Forecasters are now able to [differentiate the time series](https://skforecast.org/latest/faq/time-series-differentiation) before modeling it.

+ <span class="badge text-bg-api-change">API Change</span> Changed the default value of the `transformer_series` argument to use a `StandardScaler()` in the Global Forecasters ([`ForecasterAutoregMultiSeries`](https://skforecast.org/latest/api/forecastermultiseries), [`ForecasterAutoregMultiSeriesCustom`](https://skforecast.org/latest/api/forecastermultiseriescustom) and [`ForecasterAutoregMultiVariate`](https://skforecast.org/latest/api/forecastermultivariate)).

**Added**

+ Added `bayesian_search_forecaster_multiseries` function to `model_selection_multiseries` module. This function performs a Bayesian hyperparameter search for the `ForecasterAutoregMultiSeries`, `ForecasterAutoregMultiSeriesCustom`, and `ForecasterAutoregMultiVariate` using `optuna` as the search engine.

+ `ForecasterAutoregMultiVariate` allows to include None when lags is a dict so that a series does not participate in the construction of X_train.

+ The `output_file` argument has been added to the hyperparameter search functions in the `model_selection`, `model_selection_multiseries` and `model_selection_sarimax` modules to save the results of the hyperparameter search in a tab-separated values (TSV) file.

+ New argument `binned_residuals` in method `predict_interval` allows to condition the bootstraped residuals on range of the predicted values. 

+ Added `save_custom_functions` argument to the `save_forecaster` function in the `utils` module. If `True`, save custom functions used in the forecaster (`fun_predictors` and `weight_func`) as .py files. Custom functions must be available in the environment where the forecaster is loaded.

+ Added `select_features` and `select_features_multiseries` functions to the `model_selection` and `model_selection_multiseries` modules to perform feature selection using scikit-learn selectors.

+ Added `sort_importance` argument to `get_feature_importances` method in all Forecasters. If `True`, sort the feature importances in descending order.

+ Added `initialize_lags_grid` function to `model_selection` module. This function initializes the lags to be used in the hyperparameter search functions in `model_selection` and `model_selection_multiseries`.

+ Added `_initialize_levels_model_selection_multiseries` function to `model_selection_multiseries` module. This function initializes the levels of the series to be used in the model selection functions.

+ Added `set_dark_theme` function to the `plot` module to set a dark theme for matplotlib plots.

+ Allow tuple type for `lags` argument in all Forecasters.

+ Argument `differentiation` in all Forecasters to model the n-order differentiated time series.

+ Added `window_size_diff` attribute to all Forecasters. It stores the size of the window (`window_size`) extended by the order of differentiation. Added  to all Forecasters for API consistency.

+ Added `store_last_window` parameter to `fit` method in Forecasters. If `True`, store the last window of the training data.

+ Added `utils.set_skforecast_warnings` function to set the warnings of the skforecast package.

+ Added new forecaster `ForecasterRnn` to create forecasting models based on deep learning (RNN and LSTM).

+ Added new function `create_and_compile_model` to module `skforecast.ForecasterRnn.utils` to help to create and compile a RNN or LSTM models to be used in `ForecasterRnn`.

**Changed**

+ Deprecated argument `lags_grid` in `bayesian_search_forecaster`. Use `search_space` to define the candidate values for the lags. This allows the lags to be optimized along with the other hyperparameters of the regressor in the bayesian search.

+ `n_boot` argument in `predict_interval`changed from 500 to 250.

+ Changed the default value of the `transformer_series` argument to use a `StandardScaler()` in the Global Forecasters (`ForecasterAutoregMultiSeries`, `ForecasterAutoregMultiSeriesCustom` and `ForecasterAutoregMultiVariate`).

+ Refactor `utils.select_n_jobs_backtesting` to use the forecaster directly instead of `forecaster_name` and `regressor_name`.

+ Remove `_backtesting_forecaster_verbose` in model_selection in favor of `_create_backtesting_folds`, (deprecated since 0.8.0).

**Fixed**

+ Small bug in `utils.select_n_jobs_backtesting`, rename `ForecasterAutoregMultiseries` to `ForecasterAutoregMultiSeries`.


## 0.11.0 <small>Nov 16, 2023</small> { id="0.11.0" }

The main changes in this release are:

+ New `predict_quantiles` method in all Autoreg Forecasters to calculate the specified quantiles for each step.

+ Create `ForecasterBaseline.ForecasterEquivalentDate`, a Forecaster to create simple model that serves as a basic reference for evaluating the performance of more complex models.

**Added**

+ Added `skforecast.datasets` module. It contains functions to load data for our examples and user guides.

+ Added `predict_quantiles` method to all Autoreg Forecasters.

+ Added `SkforecastVersionWarning` to the `exception` module. This warning notify that the skforecast version installed in the environment differs from the version used to initialize the forecaster when using `load_forecaster`.

+ Create `ForecasterBaseline.ForecasterEquivalentDate`, a Forecaster to create simple model that serves as a basic reference for evaluating the performance of more complex models.

**Changed**

+ Enhance the management of internal copying in skforecast to minimize the number of copies, thereby accelerating data processing.

**Fixed**

+ Rename `self.skforcast_version` attribute to `self.skforecast_version` in all Forecasters.

+ Fixed a bug where the `create_train_X_y` method did not correctly align lags and exogenous variables when the index was not a Pandas index in all Forecasters.


## 0.10.1 <small>Sep 26, 2023</small> { id="0.10.1" }

This is a minor release to fix a bug when using `grid_search_forecaster`, `random_search_forecaster` or `bayesian_search_forecaster` with a Forecaster that includes differentiation.

**Added**


**Changed**


**Fixed**

+ Bug fix `grid_search_forecaster`, `random_search_forecaster` or `bayesian_search_forecaster` with a Forecaster that includes differentiation.


## 0.10.0 <small>Sep 07, 2023</small> { id="0.10.0" }

The main changes in this release are:

+ New `Sarimax.Sarimax` model. A wrapper of `statsmodels.SARIMAX` that follows the scikit-learn API and can be used with the `ForecasterSarimax`.

+ Added `differentiation` argument to `ForecasterAutoreg` and `ForecasterAutoregCustom` to model the n-order differentiated time series using the new skforecast preprocessor `TimeSeriesDifferentiator`.

**Added**

+ New `Sarimax.Sarimax` model. A wrapper of `statsmodels.SARIMAX` that follows the scikit-learn API.

+ Added `skforecast.preprocessing.TimeSeriesDifferentiator` to preprocess time series by differentiating or integrating them (reverse differentiation).

+ Added `differentiation` argument to `ForecasterAutoreg` and `ForecasterAutoregCustom` to model the n-order differentiated time series.

**Changed**

+ Refactor `ForecasterSarimax` to work with both skforecast Sarimax and pmdarima ARIMA models.

+ Replace `setup.py` with `pyproject.toml`.

**Fixed**


## 0.9.1 <small>Jul 14, 2023</small> { id="0.9.1" }

The main changes in this release are:

+ Fix imports in `skforecast.utils` module to correctly import `sklearn.linear_model` into the `select_n_jobs_backtesting` and `select_n_jobs_fit_forecaster` functions.

**Added**

**Changed**

**Fixed**

+ Fix imports in `skforecast.utils` module to correctly import `sklearn.linear_model` into the `select_n_jobs_backtesting` and `select_n_jobs_fit_forecaster` functions.


## 0.9.0 <small>Jul 09, 2023</small> { id="0.9.0" }

The main changes in this release are:

+ `ForecasterAutoregDirect` and `ForecasterAutoregMultiVariate` include the `n_jobs` argument in their `fit` method, allowing multi-process parallelization for improved performance.

+ All backtesting and grid search functions have been extended to include the `n_jobs` argument, allowing multi-process parallelization for improved performance.

+ Argument `refit` now can be also an `integer` in all backtesting dependent functions in modules `model_selection`, `model_selection_multiseries`, and `model_selection_sarimax`. This allows the Forecaster to be trained every this number of iterations.

+ `ForecasterAutoregMultiSeries` and `ForecasterAutoregMultiSeriesCustom` can be trained using series of different lengths. This means that the model can handle datasets with different numbers of data points in each series.

**Added**

+ Support for `scikit-learn 1.3.x`.

+ Argument `n_jobs='auto'` to `fit` method in `ForecasterAutoregDirect` and `ForecasterAutoregMultiVariate` to allow multi-process parallelization.

+ Argument `n_jobs='auto'` to all backtesting dependent functions in modules `model_selection`, `model_selection_multiseries` and `model_selection_sarimax` to allow multi-process parallelization.

+ Argument `refit` now can be also an `integer` in all backtesting dependent functions in modules `model_selection`, `model_selection_multiseries`, and `model_selection_sarimax`. This allows the Forecaster to be trained every this number of iterations.

+ `ForecasterAutoregMultiSeries` and `ForecasterAutoregMultiSeriesCustom` allow to use series of different lengths for training.

+ Added `show_progress` to grid search functions.

+ Added functions `select_n_jobs_backtesting` and `select_n_jobs_fit_forecaster` to `utils` to select the number of jobs to use during multi-process parallelization.

**Changed**

+ Remove `get_feature_importance` in favor of `get_feature_importances` in all Forecasters, (deprecated since 0.8.0).

+ The `model_selection._create_backtesting_folds` function now also returns the last window indices and whether or not to train the forecaster.

+ The `model_selection` functions `_backtesting_forecaster_refit` and `_backtesting_forecaster_no_refit` have been unified in `_backtesting_forecaster`.

+ The `model_selection_multiseries` functions `_backtesting_forecaster_multiseries_refit` and `_backtesting_forecaster_multiseries_no_refit` have been unified in `_backtesting_forecaster_multiseries`.

+ The `model_selection_sarimax` functions `_backtesting_refit_sarimax` and `_backtesting_no_refit_sarimax` have been unified in `_backtesting_sarimax`.

+ `utils.preprocess_y` allows a pandas DataFrame as input.

**Fixed**

+ Ensure reproducibility of Direct Forecasters when using `predict_bootstrapping`, `predict_dist` and `predict_interval` with a `list` of steps.

+ The `create_train_X_y` method returns a dict of pandas Series as `y_train` in `ForecasterAutoregDirect` and `ForecasterAutoregMultiVariate`. This ensures that each series has the appropriate index according to the step to be trained.

+ The `filter_train_X_y_for_step` method in `ForecasterAutoregDirect` and `ForecasterAutoregMultiVariate` now updates the index of `X_train_step` to ensure correct alignment with `y_train_step`.


## 0.8.1 <small>May 27, 2023</small> { id="0.8.1" }

**Added**

- Argument `store_in_sample_residuals=True` in `fit` method added to all forecasters to speed up functions such as backtesting.

**Changed**

- Refactor `utils.exog_to_direct` and `utils.exog_to_direct_numpy` to increase performance.

**Fixed**

- `utils.check_exog_dtypes` now compares the `dtype.name` instead of the `dtype`. (suggested by Metaming https://github.com/Metaming)


## 0.8.0 <small>May 16, 2023</small> { id="0.8.0" }

**Added**

+ Added the `fit_kwargs` argument to all forecasters to allow the inclusion of additional keyword arguments passed to the regressor's `fit` method.

+ Added the `set_fit_kwargs` method to set the `fit_kwargs` attribute.
  
+ Support for `pandas 2.0.x`.

+ Added `exceptions` module with custom warnings.

+ Added function `utils.check_exog_dtypes` to issue a warning if exogenous variables are one of type `init`, `float`, or `category`. Raise Exception if `exog` has categorical columns with non integer values.

+ Added function `utils.get_exog_dtypes` to get the data types of the exogenous variables included during the training of the forecaster model. 

+ Added function `utils.cast_exog_dtypes` to cast data types of the exogenous variables using a dictionary as a mapping.

+ Added function `utils.check_select_fit_kwargs` to check if the argument `fit_kwargs` is a dictionary and select only the keys used by the `fit` method of the regressor.

+ Added function `model_selection._create_backtesting_folds` to provide train/test indices (position) for backtesting functions.

+ Added argument `gap` to functions in `model_selection`, `model_selection_multiseries` and `model_selection_sarimax` to omit observations between training and prediction.

+ Added argument `show_progress` to functions `model_selection.backtesting_forecaster`, `model_selection_multiseries.backtesting_forecaster_multiseries` and `model_selection_sarimax.backtesting_forecaster_sarimax` to indicate weather to show a progress bar.

+ Added argument `remove_suffix`, default `False`, to the method `filter_train_X_y_for_step()` in `ForecasterAutoregDirect` and `ForecasterAutoregMultiVariate`. If `remove_suffix=True` the suffix "_step_i" will be removed from the column names of the training matrices.

**Changed**

+ Rename optional dependency package `statsmodels` to `sarimax`. Now only `pmdarima` will be installed, `statsmodels` is no longer needed.

+ Rename `get_feature_importance()` to `get_feature_importances()` in all Forecasters. `get_feature_importance()` method will me removed in skforecast 0.9.0.

+ Refactor `get_feature_importances()` in all Forecasters.

+ Remove `model_selection_statsmodels` in favor of `ForecasterSarimax` and `model_selection_sarimax`, (deprecated since 0.7.0).

+ Remove attributes `create_predictors` and `source_code_create_predictors` in favor of `fun_predictors` and `source_code_fun_predictors` in `ForecasterAutoregCustom`, (deprecated since 0.7.0).

+ The `utils.check_exog` function now includes a new optional parameter, `allow_nan`, that controls whether a warning should be issued if the input `exog` contains NaN values. 

+ `utils.check_exog` is applied before and after `exog` transformations.

+ The `utils.preprocess_y` function now includes a new optional parameter, `return_values`, that controls whether to return a numpy ndarray with the values of y or not. This new option is intended to avoid copying data when it is not necessary.

+ The `utils.preprocess_exog` function now includes a new optional parameter, `return_values`, that controls whether to return a numpy ndarray with the values of y or not. This new option is intended to avoid copying data when it is not necessary.

+ Replaced `tqdm.tqdm` by `tqdm.auto.tqdm`.

+ Refactor `utils.exog_to_direct`.

**Fixed**

+ The dtypes of exogenous variables are maintained when generating the training matrices with the `create_train_X_y` method in all the Forecasters.


## 0.7.0 <small>Mar 21, 2023</small> { id="0.7.0" }

**Added**

+ Class `ForecasterAutoregMultiSeriesCustom`.

+ Class `ForecasterSarimax` and `model_selection_sarimax` (wrapper of [pmdarima](http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA)).
  
+ Method `predict_interval()` to `ForecasterAutoregDirect` and `ForecasterAutoregMultiVariate`.

+ Method `predict_bootstrapping()` to all forecasters, generate multiple forecasting predictions using a bootstrapping process.

+ Method `predict_dist()` to all forecasters, fit a given probability distribution for each step using a bootstrapping process.

+ Function `plot_prediction_distribution` in module `plot`.

+ Alias `backtesting_forecaster_multivariate` for `backtesting_forecaster_multiseries` in `model_selection_multiseries` module.

+ Alias `grid_search_forecaster_multivariate` for `grid_search_forecaster_multiseries` in `model_selection_multiseries` module.

+ Alias `random_search_forecaster_multivariate` for `random_search_forecaster_multiseries` in `model_selection_multiseries` module.

+ Attribute `forecaster_id` to all Forecasters.

**Changed**

+ Deprecated `python 3.7` compatibility.

+ Added `python 3.11` compatibility.

+ `model_selection_statsmodels` is deprecated in favor of `ForecasterSarimax` and `model_selection_sarimax`. It will be removed in version 0.8.0.

+ Remove `levels_weights` argument in `grid_search_forecaster_multiseries` and `random_search_forecaster_multiseries`, deprecated since version 0.6.0. Use `series_weights` and `weights_func` when creating the forecaster instead.

+ Attributes `create_predictors` and `source_code_create_predictors` renamed to `fun_predictors` and `source_code_fun_predictors` in `ForecasterAutoregCustom`. Old names will be removed in version 0.8.0.

+ Remove engine `'skopt'` in `bayesian_search_forecaster` in favor of engine `'optuna'`. To continue using it, use skforecast 0.6.0.

+ `in_sample_residuals` and `out_sample_residuals` are stored as numpy ndarrays instead of pandas series.

+ In `ForecasterAutoregMultiSeries`, `set_out_sample_residuals()` is now expecting a `dict` for the `residuals` argument instead of a `pandas DataFrame`.

+ Remove the `scikit-optimize` dependency.

**Fixed**

+ Remove operator `**` in `set_params()` method for all forecasters.

+ Replace `getfullargspec` in favor of `inspect.signature` (contribution by @jordisilv).


## 0.6.0 <small>Nov 30, 2022</small> { id="0.6.0" }

**Added**

+ Class `ForecasterAutoregMultivariate`.

+ Function `initialize_lags` in `utils` module  to create lags values in the initialization of forecasters (applies to all forecasters).

+ Function `initialize_weights` in `utils` module to check and initialize arguments `series_weights`and `weight_func` (applies to all forecasters).

+ Argument `weights_func` in all Forecasters to allow weighted time series forecasting. Individual time based weights can be assigned to each value of the series during the model training.

+ Argument `series_weights` in `ForecasterAutoregMultiSeries` to define individual weights each series.

+ Include argument `random_state` in all Forecasters `set_out_sample_residuals` methods for random sampling with reproducible output.

+ In `ForecasterAutoregMultiSeries`, `predict` and `predict_interval` methods allow the simultaneous prediction of multiple levels.

+ `backtesting_forecaster_multiseries` allows backtesting multiple levels simultaneously.

+ `metric` argument can be a list in `grid_search_forecaster_multiseries`, `random_search_forecaster_multiseries`. If `metric` is a `list`, multiple metrics will be calculated. (suggested by Pablo Dávila Herrero https://github.com/Pablo-Davila)

+ Function `multivariate_time_series_corr` in module `utils`.

+ Function `plot_multivariate_time_series_corr` in module `plot`.
  
**Changed**

+ `ForecasterAutoregDirect` allows to predict specific steps.

+ Remove `ForecasterAutoregMultiOutput` in favor of `ForecasterAutoregDirect`, (deprecated since 0.5.0).

+ Rename function `exog_to_multi_output` to `exog_to_direct` in `utils` module.

+ In `ForecasterAutoregMultiSeries`, rename parameter `series_levels` to `series_col_names`.

+ In `ForecasterAutoregMultiSeries` change type of `out_sample_residuals` to a `dict` of numpy ndarrays.

+ In `ForecasterAutoregMultiSeries`, delete argument `level` from method `set_out_sample_residuals`.

+ In `ForecasterAutoregMultiSeries`, `level` argument of `predict` and `predict_interval` renamed to `levels`.

+ In `backtesting_forecaster_multiseries`, `level` argument of `predict` and `predict_interval` renamed to `levels`.

+ In `check_predict_input` function, argument `level` renamed to `levels` and `series_levels` renamed to `series_col_names`.

+ In `backtesting_forecaster_multiseries`, `metrics_levels` output is now a pandas DataFrame.

+ In `grid_search_forecaster_multiseries` and `random_search_forecaster_multiseries`, argument `levels_weights` is deprecated since version 0.6.0, and will be removed in version 0.7.0. Use `series_weights` and `weights_func` when creating the forecaster instead.

+ Refactor `_create_lags_` in `ForecasterAutoreg`, `ForecasterAutoregDirect` and `ForecasterAutoregMultiSeries`. (suggested by Bennett https://github.com/Bennett561)

+ Refactor `backtesting_forecaster` and `backtesting_forecaster_multiseries`.

+ In `ForecasterAutoregDirect`, `filter_train_X_y_for_step` now starts at 1 (before 0).

+ In `ForecasterAutoregDirect`, DataFrame `y_train` now start with 1, `y_step_1` (before `y_step_0`).

+ Remove `cv_forecaster` from module `model_selection`.

**Fixed**

+ In `ForecasterAutoregMultiSeries`, argument `last_window` predict method now works when it is a pandas DataFrame.

+ In `ForecasterAutoregMultiSeries`, fix bug transformers initialization.


## 0.5.1 <small>Oct 05, 2022</small> { id="0.5.1" }

**Added**

+ Check that `exog` and `y` have the same length in `_evaluate_grid_hyperparameters` and `bayesian_search_forecaster` to avoid fit exception when `return_best`.

+ Check that `exog` and `series` have the same length in `_evaluate_grid_hyperparameters_multiseries` to avoid fit exception when `return_best`.

**Changed**

+ Argument `levels_list` in `grid_search_forecaster_multiseries`, `random_search_forecaster_multiseries` and `_evaluate_grid_hyperparameters_multiseries` renamed to `levels`.

**Fixed**

+ `ForecasterAutoregMultiOutput` updated to match `ForecasterAutoregDirect`.

+ Fix Exception to raise when `level_weights` does not add up to a number close to 1.0 (before was exactly 1.0) in `grid_search_forecaster_multiseries`, `random_search_forecaster_multiseries` and `_evaluate_grid_hyperparameters_multiseries`.

+ `Create_train_X_y` in `ForecasterAutoregMultiSeries` now works when the forecaster is not fitted.


## 0.5.0 <small>Sep 23, 2022</small> { id="0.5.0" }

**Added**

+ New arguments `transformer_y` (`transformer_series` for multiseries) and `transformer_exog` in all forecaster classes. It is for transforming (scaling, max-min, ...) the modeled time series and exogenous variables inside the forecaster.

+ Functions in utils `transform_series` and `transform_dataframe` to carry out the transformation of the modeled time series and exogenous variables.

+ Functions `_backtesting_forecaster_verbose`, `random_search_forecaster`, `_evaluate_grid_hyperparameters`, `bayesian_search_forecaster`, `_bayesian_search_optuna` and `_bayesian_search_skopt` in model_selection.

+ Created `ForecasterAutoregMultiSeries` class for modeling multiple time series simultaneously.

+ Created module `model_selection_multiseries`. Functions: `_backtesting_forecaster_multiseries_refit`, `_backtesting_forecaster_multiseries_no_refit`, `backtesting_forecaster_multiseries`, `grid_search_forecaster_multiseries`, `random_search_forecaster_multiseries` and `_evaluate_grid_hyperparameters_multiseries`.

+ Function `_check_interval` in utils. (suggested by Thomas Karaouzene https://github.com/tkaraouzene)

+ `metric` can be a list in `backtesting_forecaster`, `grid_search_forecaster`, `random_search_forecaster`, `backtesting_forecaster_multiseries`. If `metric` is a `list`, multiple metrics will be calculated. (suggested by Pablo Dávila Herrero https://github.com/Pablo-Davila)

+ Skforecast works with python 3.10.

+ Functions `save_forecaster` and `load_forecaster` to module utils.

+ `get_feature_importance()` method checks if the forecast is fitted.

**Changed**

+ `backtesting_forecaster` change default value of argument `fixed_train_size: bool=True`.

+ Remove argument `set_out_sample_residuals` in function `backtesting_forecaster` (deprecated since 0.4.2).

+ `backtesting_forecaster` verbose now includes fold size.

+ `grid_search_forecaster` results include the name of the used metric as column name.

+ Remove `get_coef` method from `ForecasterAutoreg`, `ForecasterAutoregCustom` and `ForecasterAutoregMultiOutput` (deprecated since 0.4.3).

+ `_get_metric` now allows `mean_squared_log_error`.

+ `ForecasterAutoregMultiOutput` has been renamed to `ForecasterAutoregDirect`. `ForecasterAutoregMultiOutput` will be removed in version 0.6.0.

+ `check_predict_input` updated to check `ForecasterAutoregMultiSeries` inputs.

+ `set_out_sample_residuals` has a new argument `transform` to transform the residuals before being stored.

**Fixed**

+ `fit` now stores `last_window` values with len = forecaster.max_lag in ForecasterAutoreg and ForecasterAutoregCustom.

+ `in_sample_residuals` stored as a `pd.Series` when `len(residuals) > 1000`.


## 0.4.3 <small>Mar 18, 2022</small> { id="0.4.3" }

**Added**

+ Checks if all elements in lags are `int` when creating ForecasterAutoreg and ForecasterAutoregMultiOutput.

+ Add `fixed_train_size: bool=False` argument to `backtesting_forecaster` and `backtesting_sarimax`

**Changed**

+ Rename `get_metric` to `_get_metric`.

+ Functions in model_selection module allow custom metrics.

+ Functions in model_selection_statsmodels module allow custom metrics.

+ Change function `set_out_sample_residuals` (ForecasterAutoreg and ForecasterAutoregCustom), `residuals` argument must be a `pandas Series` (was `numpy ndarray`).

+ Returned value of backtesting functions (model_selection and model_selection_statsmodels) is now a `float` (was `numpy ndarray`).

+ `get_coef` and `get_feature_importance` methods unified in `get_feature_importance`.

**Fixed**

+ Requirements versions.

+ Method `fit` doesn't remove `out_sample_residuals` each time the forecaster is fitted.

+ Added random seed to residuals downsampling (ForecasterAutoreg and ForecasterAutoregCustom)


## 0.4.2 <small>Jan 08, 2022</small> { id="0.4.2" }

**Added**

+ Increased verbosity of function `backtesting_forecaster()`.

+ Random state argument in `backtesting_forecaster()`.

**Changed**

+ Function `backtesting_forecaster()` do not modify the original forecaster.

+ Deprecated argument `set_out_sample_residuals` in function `backtesting_forecaster()`.

+ Function `model_selection.time_series_spliter` renamed to `model_selection.time_series_splitter`

**Fixed**

+ Methods `get_coef` and `get_feature_importance` of `ForecasterAutoregMultiOutput` class return proper feature names.


## 0.4.1 <small>Dec 13, 2021</small> { id="0.4.1" }

**Added**

**Changed**

**Fixed**

+ `fit` and `predict` transform pandas series and dataframes to numpy arrays if regressor is XGBoost.


## 0.4.0 <small>Dec 10, 2021</small> { id="0.4.0" }

Version 0.4 has undergone a huge code refactoring. Main changes are related to input-output formats (only pandas series and dataframes are allowed although internally numpy arrays are used for performance) and model validation methods (unified into backtesting with and without refit).

**Added**

+ `ForecasterBase` as parent class

**Changed**

+ Argument `y` must be pandas Series. Numpy ndarrays are not allowed anymore.

+ Argument `exog` must be pandas Series or pandas DataFrame. Numpy ndarrays are not allowed anymore.

+ Output of `predict` is a pandas Series with index according to the steps predicted.

+ Scikitlearn pipelines are allowed as regressors.

+ `backtesting_forecaster` and `backtesting_forecaster_intervals` have been combined in a single function.

    + It is possible to backtest forecasters already trained.
    + `ForecasterAutoregMultiOutput` allows incomplete folds.
    + It is possible to update `out_sample_residuals` with backtesting residuals.
    
+ `cv_forecaster` has the option to update `out_sample_residuals` with backtesting residuals.

+ `backtesting_sarimax_statsmodels` and `cv_sarimax_statsmodels` have been combined in a single function.

+ `gridsearch_forecaster` use backtesting as validation strategy with the option of refit.

+ Extended information when printing `Forecaster` object.

+ All static methods for checking and preprocessing inputs moved to module utils.

+ Remove deprecated class `ForecasterCustom`.

**Fixed**


## 0.3.0 <small>Sep 01, 2021</small> { id="0.3.0" }

**Added**

+ New module model_selection_statsmodels to cross-validate, backtesting and grid search AutoReg and SARIMAX models from statsmodels library:
    + `backtesting_autoreg_statsmodels`
    + `cv_autoreg_statsmodels`
    + `backtesting_sarimax_statsmodels`
    + `cv_sarimax_statsmodels`
    + `grid_search_sarimax_statsmodels`
    
+ Added attribute window_size to `ForecasterAutoreg` and `ForecasterAutoregCustom`. It is equal to `max_lag`.

**Changed**

+ `cv_forecaster` returns cross-validation metrics and cross-validation predictions.
+ Added an extra column for each parameter in the dataframe returned by `grid_search_forecaster`.
+ statsmodels 0.12.2 added to requirements

**Fixed**


## 0.2.0 <small>Aug 26, 2021</small> { id="0.2.0" }

**Added**


+ Multiple exogenous variables can be passed as pandas DataFrame.

+ Documentation at https://joaquinamatrodrigo.github.io/skforecast/

+ New unit test

+ Increased typing

**Changed**

+ New implementation of `ForecasterAutoregMultiOutput`. The training process in the new version creates a different X_train for each step. See [Direct multi-step forecasting](https://github.com/skforecast/skforecast#introduction) for more details. Old versión can be acces with `skforecast.deprecated.ForecasterAutoregMultiOutput`.

**Fixed**


## 0.1.9 <small>Jul 27, 2021</small> { id="0.1.9" }

**Added**

+ Logging total number of models to fit in `grid_search_forecaster`.

+ Class `ForecasterAutoregCustom`.

+ Method `create_train_X_y` to facilitate access to the training data matrix created from `y` and `exog`.

**Changed**


+ New implementation of `ForecasterAutoregMultiOutput`. The training process in the new version creates a different X_train for each step. See [Direct multi-step forecasting](https://github.com/skforecast/skforecast#introduction) for more details. Old versión can be acces with `skforecast.deprecated.ForecasterAutoregMultiOutput`.

+ Class `ForecasterCustom` has been renamed to `ForecasterAutoregCustom`. However, `ForecasterCustom` will still remain to keep backward compatibility.

+ Argument `metric` in `cv_forecaster`, `backtesting_forecaster`, `grid_search_forecaster` and `backtesting_forecaster_intervals` changed from 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error' to 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'.

+ Check if argument `metric` in `cv_forecaster`, `backtesting_forecaster`, `grid_search_forecaster` and `backtesting_forecaster_intervals` is one of 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'.

+ `time_series_spliter` doesn't include the remaining observations in the last complete fold but in a new one when `allow_incomplete_fold=True`. Take in consideration that incomplete folds with few observations could overestimate or underestimate the validation metric.

**Fixed**

+ Update lags of  `ForecasterAutoregMultiOutput` after `grid_search_forecaster`.


## 0.1.8.1 <small>May 17, 2021</small> { id="0.1.8.1" }

**Added**

+ `set_out_sample_residuals` method to store or update out of sample residuals used by `predict_interval`.

**Changed**

+ `backtesting_forecaster_intervals` and `backtesting_forecaster` print number of steps per fold.

+ Only stored up to 1000 residuals.

+ Improved verbose in `backtesting_forecaster_intervals`.

**Fixed**

+ Warning of inclompleted folds when using `backtesting_forecast` with a  `ForecasterAutoregMultiOutput`.

+ `ForecasterAutoregMultiOutput.predict` allow exog data longer than needed (steps).

+ `backtesting_forecast` prints correctly the number of folds when remainder observations are cero.

+ Removed named argument X in `self.regressor.predict(X)` to allow using XGBoost regressor.

+ Values stored in `self.last_window` when training `ForecasterAutoregMultiOutput`. 


## 0.1.8 <small>Apr 02, 2021</small> { id="0.1.8" }

**Added**

- Class `ForecasterAutoregMultiOutput.py`: forecaster with direct multi-step predictions.
- Method `ForecasterCustom.predict_interval` and  `ForecasterAutoreg.predict_interval`: estimate prediction interval using bootstrapping.
- `skforecast.model_selection.backtesting_forecaster_intervals` perform backtesting and return prediction intervals.
 
**Changed**

 
**Fixed**


## 0.1.7 <small>Mar 19, 2021</small> { id="0.1.7" }

**Added**

- Class `ForecasterCustom`: same functionalities as `ForecasterAutoreg` but allows custom definition of predictors.
 
**Changed**

- `grid_search forecaster` adapted to work with objects `ForecasterCustom` in addition to `ForecasterAutoreg`.
 
**Fixed**
 
 
## 0.1.6 <small>Mar 14, 2021</small> { id="0.1.6" }

**Added**

- Method `get_feature_importances` to `skforecast.ForecasterAutoreg`.
- Added backtesting strategy in `grid_search_forecaster`.
- Added `backtesting_forecast` to `skforecast.model_selection`.
 
**Changed**

- Method `create_lags` return a matrix where the order of columns match the ascending order of lags. For example, column 0 contains the values of the minimum lag used as predictor.
- Renamed argument `X` to `last_window` in method `predict`.
- Renamed `ts_cv_forecaster` to `cv_forecaster`.
 
**Fixed**


## 0.1.4 <small>Feb 15, 2021</small> { id="0.1.4" }
  
**Added**

- Method `get_coef` to `skforecast.ForecasterAutoreg`.
 
**Changed**

 
**Fixed**



<!-- Links to API Reference -->
<!-- Forecasters -->
[ForecasterRecursive]: https://skforecast.org/latest/api/forecasterrecursive
[ForecasterDirect]: https://skforecast.org/latest/api/forecasterdirect
[ForecasterRecursiveMultiSeries]: https://skforecast.org/latest/api/forecastermultiseries
[ForecasterDirectMultiVariate]: https://skforecast.org/latest/api/forecastermultivariate
[ForecasterRNN]: https://skforecast.org/latest/api/forecasterrnn
[ForecasterSarimax]: https://skforecast.org/latest/api/forecastersarimax
[Sarimax]: https://skforecast.org/latest/api/sarimax
[ForecasterEquivalentDate]: https://skforecast.org/latest/api/forecasterbaseline#skforecast.ForecasterBaseline.ForecasterEquivalentDate

<!-- metrics -->
[metrics]: https://skforecast.org/latest/api/metrics
[add_y_train_argument]: https://skforecast.org/latest/api/metrics#skforecast.metrics.metrics.add_y_train_argument
[mean_absolute_scaled_error]: https://skforecast.org/latest/api/metrics#skforecast.metrics.metrics.mean_absolute_scaled_error
[root_mean_squared_scaled_error]: https://skforecast.org/latest/api/metrics#skforecast.metrics.metrics.root_mean_squared_scaled_error

<!-- model_selection -->
[model_selection]: https://skforecast.org/latest/api/model_selection
[backtesting_forecaster]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection._validation.backtesting_forecaster
[grid_search_forecaster]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection.model_selection.grid_search_forecaster
[random_search_forecaster]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection.model_selection.random_search_forecaster
[bayesian_search_forecaster]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection.model_selection.bayesian_search_forecaster

[backtesting_forecaster_multiseries]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection._validation.backtesting_forecaster_multiseries
[grid_search_forecaster_multiseries]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection._search.grid_search_forecaster_multiseries
[random_search_forecaster_multiseries]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection._search.random_search_forecaster_multiseries
[bayesian_search_forecaster_multiseries]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection._search.bayesian_search_forecaster_multiseries

[backtesting_sarimax]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection._validation.backtesting_sarimax
[grid_search_sarimax]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection._search.grid_search_sarimax
[random_search_sarimax]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection._search.random_search_sarimax

[BaseFold]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection._split.basefold
[TimeSeriesFold]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection._split.timeseriesfold
[OneStepAheadFold]: https://skforecast.org/latest/api/model_selection#skforecast.model_selection._split.onestepaheadfold

<!-- feature_selection -->
[feature_selection]: https://skforecast.org/latest/api/feature_selection
[select_features]: https://skforecast.org/latest/api/feature_selection#skforecast.feature_selection.feature_selection.select_features
[select_features_multiseries]: https://skforecast.org/latest/api/feature_selection#skforecast.feature_selection.feature_selection.select_features_multiseries

<!-- preprocessing -->
[preprocessing]: https://skforecast.org/latest/api/preprocessing
[TimeSeriesDifferentiator]: https://skforecast.org/latest/api/preprocessing#skforecast.preprocessing.preprocessing.TimeSeriesDifferentiator
[series_long_to_dict]: https://skforecast.org/latest/api/preprocessing#skforecast.preprocessing.preprocessing.series_long_to_dict
[exog_long_to_dict]: https://skforecast.org/latest/api/preprocessing#skforecast.preprocessing.preprocessing.exog_long_to_dict
[DateTimeFeatureTransformer]: https://skforecast.org/latest/api/preprocessing#skforecast.preprocessing.preprocessing.DateTimeFeatureTransformer
[create_datetime_features]: https://skforecast.org/latest/api/preprocessing#skforecast.preprocessing.preprocessing.create_datetime_features
[RollingFeatures]: https://skforecast.org/latest/api/preprocessing#skforecast.preprocessing.preprocessing.RollingFeatures

<!-- plot -->
[plot]: https://skforecast.org/latest/api/plot
[set_dark_theme]: https://skforecast.org/latest/api/plot#skforecast.plot.plot.set_dark_theme
[plot_residuals]: https://skforecast.org/latest/api/plot#skforecast.plot.plot.plot_residuals
[plot_multivariate_time_series_corr]: https://skforecast.org/latest/api/plot#skforecast.plot.plot.plot_multivariate_time_series_corr
[plot_prediction_distribution]: https://skforecast.org/latest/api/plot#skforecast.plot.plot.plot_prediction_distribution
[plot_prediction_intervals]: https://skforecast.org/latest/api/plot#skforecast.plot.plot.plot_prediction_intervals

<!-- datasets -->
[datasets]: https://skforecast.org/latest/api/datasets
[fetch_dataset]: https://skforecast.org/latest/api/datasets#skforecast.datasets.fetch_dataset
[load_demo_dataset]: https://skforecast.org/latest/api/datasets#skforecast.datasets.load_demo_dataset

<!-- utils -->
[utils]: https://skforecast.org/latest/api/utils


<!-- OLD -->
[ForecasterAutoreg]: https://skforecast.org/0.13.0/api/forecasterautoreg
[ForecasterAutoregCustom]: https://skforecast.org/0.13.0/api/forecasterautoregcustom
[ForecasterAutoregDirect]: https://skforecast.org/0.13.0/api/forecasterautoregdirect
[ForecasterAutoregMultiSeries]: https://skforecast.org/0.13.0/api/forecastermultiseries
[ForecasterAutoregMultiSeriesCustom]: https://skforecast.org/0.13.0/api/forecastermultiseriescustom
[ForecasterAutoregMultiVariate]: https://skforecast.org/0.13.0/api/forecastermultivariate
[model_selection_multiseries]: https://skforecast.org/0.13.0/api/model_selection_multiseries
[model_selection_sarimax]: https://skforecast.org/0.13.0/api/model_selection_sarimax
