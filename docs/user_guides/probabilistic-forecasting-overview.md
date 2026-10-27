# Probabilistic forecasting

When trying to predict future values, most forecasting models estimate a single value, usually the expected (mean) value of the series at each future step. This is called point forecasting. Although knowing the expected value of a time series in advance is useful in almost any business case, this type of prediction does not provide any information about the confidence of the model or the uncertainty of the prediction.

Probabilistic forecasting, as opposed to point forecasting, is a family of techniques that estimate the distribution of the outcome rather than a single future value. This type of forecasting provides much richer information because it allows the creation of prediction intervals, the range of likely values where the true value may fall. More formally, a prediction interval defines the interval within which the true value of the response variable is expected to be found with a given probability.

Skforecast implements four families of methods for probabilistic forecasting, each summarized below and explained in detail in its own user guide:

+ [**Bootstrapped residuals**](probabilistic-forecasting-bootstrapped-residuals.ipynb): simulate many possible future paths by adding resampled past errors to the predictions.

+ [**Conformal prediction**](probabilistic-forecasting-conformal-prediction.ipynb): build the interval by adding and subtracting a quantile of the past errors to the point forecast. Conformal methods can also be used to [calibrate intervals](probabilistic-forecasting-conformal-calibration.ipynb) obtained with other techniques.

+ [**Quantile regression**](probabilistic-forecasting-quantile-regression.ipynb): train models that directly predict the quantiles that define the bounds of the interval.

+ [**Foundation models**](foundation-forecasting-models.ipynb): pre-trained models that predict the quantiles of a new series directly, with no residuals and no training.

<div class="skf-anim-embed">
  <iframe src="../animations/probabilistic-methods.html" title="Animation: how each probabilistic forecasting method builds the interval" loading="lazy" allowfullscreen></iframe>
</div>

Bootstrapping and conformal prediction build the interval from the past errors of the model (residuals), so the interval is only reliable when those residuals are representative of the errors the model will make in the future: the section [Residuals](#residuals) explains which ones to use. The last sections compare the methods and explain how to evaluate the resulting intervals.


## Bootstrapped residuals

Bootstrapping is a statistical technique that allows for estimating the distribution of a statistic by resampling the data with replacement. Applied to forecasting, the idea is to simulate many possible futures. In each simulation (bootstrapping iteration), a residual (a past error of the model) is drawn at random and added to the prediction of each step. In recursive forecasters, this noisy prediction is also used as a lag to predict the next step, so the errors accumulate along the forecast horizon, as they would in reality. Repeating this process hundreds of times produces a collection of plausible trajectories, whose spread reflects the uncertainty of the forecast.

+ [`predict_bootstrapping()`](https://skforecast.org/latest/api/forecasterrecursive#skforecast.recursive._forecaster_recursive.ForecasterRecursive.predict_bootstrapping): returns the simulated trajectories themselves, a pandas `DataFrame` with one row per predicted step and one column per bootstrapping iteration. For example, with `steps=24` and `n_boot=500`, the output has shape `(24, 500)`: each column is one possible future, and each row contains 500 plausible values for that step.

The other three methods call `predict_bootstrapping()` internally and summarize, row by row, the distribution of the simulated values:

+ [`predict_interval(method='bootstrapping')`](https://skforecast.org/latest/api/forecasterrecursive#skforecast.recursive._forecaster_recursive.ForecasterRecursive.predict_interval): the lower and upper bounds of the interval, computed as quantiles of the simulated values (for example, the 0.1 and 0.9 quantiles for an 80% interval).

+ [`predict_quantiles()`](https://skforecast.org/latest/api/forecasterrecursive#skforecast.recursive._forecaster_recursive.ForecasterRecursive.predict_quantiles): any list of quantiles of the simulated values.

+ [`predict_dist()`](https://skforecast.org/latest/api/forecasterrecursive#skforecast.recursive._forecaster_recursive.ForecasterRecursive.predict_dist): the parameters of a parametric distribution fitted to the simulated values. Any of the continuous distributions available in [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions) can be used.

These methods are available in [`ForecasterRecursive`](https://skforecast.org/latest/api/forecasterrecursive), [`ForecasterDirect`](https://skforecast.org/latest/api/forecasterdirect), [`ForecasterRecursiveMultiSeries`](https://skforecast.org/latest/api/forecasterrecursivemultiseries) and [`ForecasterDirectMultiVariate`](https://skforecast.org/latest/api/forecasterdirectmultivariate).

Discover how to use these methods in [Probabilistic forecasting with bootstrapped residuals](probabilistic-forecasting-bootstrapped-residuals.ipynb).


## Conformal prediction

Conformal prediction is a framework for constructing prediction intervals that are guaranteed to contain the true value with at least a specified probability (coverage probability), on average over new observations (marginal coverage), under the assumption that the data are exchangeable. In time series this assumption holds only approximately, and for multi-step forecasts the coverage is not guaranteed, so the empirical coverage should always be validated (see [Evaluating prediction intervals](#evaluating-prediction-intervals)).

It works by combining the predictions of a point forecasting model with its past residuals (differences between previous predictions and actual values). In the split conformal approach, the absolute residuals are computed on a calibration set (data not used to train the model), and their quantile at the nominal coverage level, $\hat{q}$, is calculated. The prediction interval is then built around the point forecast as $\hat{y} \pm \hat{q}$. In skforecast, the calibration residuals are the out-of-sample residuals stored in the forecaster; with the default in-sample residuals, the interval is not a split conformal interval and tends to be too narrow (see [Residuals](#residuals)).

Skforecast implements [Split Conformal Prediction (SCP)](https://mapie.readthedocs.io/en/stable/content/conformal-prediction/regression/#2-the-split-method) through the [`predict_interval(method='conformal')`](https://skforecast.org/latest/api/forecasterrecursive#skforecast.recursive._forecaster_recursive.ForecasterRecursive.predict_interval) method. Conformal prediction is available in the same forecasters as bootstrapping, and it is the only probabilistic method available in [`ForecasterRnn`](https://skforecast.org/latest/api/forecasterrnn) and [`ForecasterEquivalentDate`](https://skforecast.org/latest/api/forecasterequivalentdate).

Discover how to use conformal methods in [Probabilistic forecasting with conformal prediction](probabilistic-forecasting-conformal-prediction.ipynb).

Conformal methods can also calibrate prediction intervals generated by other techniques, such as quantile regression or bootstrapped residuals. In this case, the conformal method widens or narrows the intervals so that their empirical coverage on a calibration set matches the nominal coverage. Skforecast provides this functionality through the [`ConformalIntervalCalibrator`](https://skforecast.org/latest/api/preprocessing#skforecast.preprocessing._preprocessing.ConformalIntervalCalibrator) transformer.

Discover how to perform this calibration in [Probabilistic forecasting, conformal calibration](probabilistic-forecasting-conformal-calibration.ipynb).


## Quantile regression

Quantile regression is a technique for estimating the conditional quantiles of a response variable. Instead of minimizing the squared error, the model minimizes the quantile loss (also known as pinball loss), which penalizes underestimation and overestimation asymmetrically, so that the predictions converge to the requested quantile. By combining the predictions of two quantile estimators, an interval can be constructed, with each model estimating one of the bounds of the interval. For example, models trained on $Q = 0.1$ and $Q = 0.9$ produce a nominal 80% prediction interval ($90\% - 10\% = 80\%$).

If a machine learning algorithm capable of modeling quantiles is used as the `estimator` in a forecaster, the `predict` method returns predictions for the quantile the estimator was trained on. Examples of such estimators are LightGBM (`LGBMRegressor(objective='quantile', alpha=0.9)`) and scikit-learn's `GradientBoostingRegressor(loss='quantile', alpha=0.9)` or `HistGradientBoostingRegressor(loss='quantile', quantile=0.9)`. By creating two forecasters, each configured with a different quantile, their predictions can be combined to generate a prediction interval. As explained in [Which method to use?](#which-method-to-use), a direct forecaster such as [`ForecasterDirect`](https://skforecast.org/latest/api/forecasterdirect) is preferred for this method.

Discover how to use this method in [Probabilistic forecasting with quantile regression](probabilistic-forecasting-quantile-regression.ipynb).


## Foundation models

Foundation models are large models pre-trained on a vast number of time series, which forecast a new series without being trained on it (zero-shot). Most of them are probabilistic by design: besides the point forecast, they predict a set of quantiles of the future values. The interval comes directly from the model, so it needs no residuals, no calibration set and no additional training. This makes them a strong option when the series is short or new, or when there are no residuals representative of the errors the model will make in the future.

[`ForecasterFoundation`](https://skforecast.org/latest/api/forecasterfoundation) gives access to these quantiles with the same API as the other forecasters, for a single series or for many:

+ [`predict_interval()`](https://skforecast.org/latest/api/forecasterfoundation#skforecast.foundation._forecaster_foundation.ForecasterFoundation.predict_interval): the lower and upper bounds of the interval, which are the two quantiles predicted by the model for the requested interval.

+ [`predict_quantiles()`](https://skforecast.org/latest/api/forecasterfoundation#skforecast.foundation._forecaster_foundation.ForecasterFoundation.predict_quantiles): any list of the quantiles the model supports.

The quantiles available depend on the model: some accept any quantile, whereas others only return a fixed set of them, which limits the widest interval they can provide.

Discover how to use these models in [Forecasting with foundation models](foundation-forecasting-models.ipynb).


## Residuals

Bootstrapping and conformal prediction use the residuals of the model as an estimate of the errors it will make in the future. Two types of residuals can be used, selected with the `use_in_sample_residuals` argument:

+ **In-sample residuals** (`use_in_sample_residuals=True`, the default): the errors of the model on the same data used to train it. They are always available, but the model has already seen these observations, so they underestimate the future error and produce intervals that are too narrow. For example, in the [bootstrapped residuals user guide](probabilistic-forecasting-bootstrapped-residuals.ipynb), a nominal 80% interval built with in-sample residuals achieves an empirical coverage of around 61%.

+ **Out-of-sample residuals** (`use_in_sample_residuals=False`): the errors of the model on data not used to train it. They are obtained by backtesting on a validation (calibration) set, never on the test set, and stored in the forecaster with [`set_out_sample_residuals()`](https://skforecast.org/latest/api/forecasterrecursive#skforecast.recursive._forecaster_recursive.ForecasterRecursive.set_out_sample_residuals). They are the recommended option. Fitting the forecaster again discards them, so they must be calculated again after every refit.

By default (`use_binned_residuals=True`), the residuals are conditioned on the predicted value: they are grouped into bins according to the prediction that generated them, and each new prediction uses the residuals of its bin. The width of the interval then adapts to the magnitude of the prediction. For example, when the error grows with the level of the series, the intervals are narrow for low predictions and wide for high ones.

<div class="skf-anim-embed">
  <iframe src="../animations/probabilistic-residuals.html" title="Animation: in-sample, out-of-sample and binned residuals" loading="lazy" allowfullscreen></iframe>
</div>


## Which method to use?

There is no definitive answer to this question, as the resulting coverage may vary depending on the dataset and estimator. The table summarizes the main differences between the methods:

| Method | Requires | Shape of the interval | Output | Computational cost |
|:-------|:---------|:----------------------|:-------|:-------------------|
| Bootstrapped residuals | Out-of-sample residuals | Asymmetric if the residuals are; widens along the horizon in recursive forecasters | Simulated paths, intervals, quantiles and distributions | Grows with `n_boot` and the number of series |
| Conformal prediction | Out-of-sample residuals | Symmetric around the forecast; does not grow with the horizon | Intervals | The lowest |
| Quantile regression | An estimator that supports a quantile loss | Asymmetric; the predicted quantiles may cross | One quantile per forecaster | One forecaster trained per quantile |
| Foundation models | A pre-trained model; no training and no residuals | Set by the model; can be asymmetric | Intervals and quantiles | No training, but the inference of a large model: a GPU is recommended |

Some general guidelines can be followed:

+ **Bootstrapped residuals**: This method achieves good results in most cases and is the only one that returns the full predictive distribution (simulated paths, any quantile or a parametric distribution). In recursive forecasters, the errors are propagated along the horizon, so the intervals widen with the number of steps. However, when the residuals already come from multi-step predictions, this propagation can produce intervals that are too wide: in the [global models user guide](probabilistic-forecasting-global-models.ipynb), bootstrapping reaches an empirical coverage of 96% to 98% for a nominal 80% interval. Its cost is negligible for a single series (milliseconds), but it grows with `n_boot` and with the number of series: forecasting 300 series is about 35 times slower than with conformal prediction.

+ **Conformal prediction**: The fastest method, since no simulation is needed, and its intervals are often narrower than those of bootstrapping, with an empirical coverage close to the nominal one (about 84% for a nominal 80% interval in the global models user guide). Its drawbacks: the interval is symmetric, so a bias of the model is not corrected, and the same correction factor is applied to all the steps of the horizon, so when the errors grow with the horizon, the intervals tend to be too wide for the first steps and too narrow for the last ones.

+ **Quantile regression**: An appropriate choice when the estimator is capable of modeling quantiles. However, it has three drawbacks. First, one forecaster must be trained for each quantile, which can be computationally expensive. Second, since each model is trained independently, nothing prevents the predicted quantiles from crossing (for example, a lower bound above the upper bound). Third, in recursive forecasters the quantile predictions are fed back as lags, so the predictions of later steps are built on quantile values instead of expected values. For this reason, a direct strategy ([`ForecasterDirect`](https://skforecast.org/latest/api/forecasterdirect)) is preferred. The coverage of these intervals can be corrected with [conformal calibration](probabilistic-forecasting-conformal-calibration.ipynb).

+ **Foundation models**: The only method with nothing to train and no residuals to collect, so it is the natural choice for short or new series and a strong candidate in any other case. The same model forecasts any series, and its interval does not have to be symmetric around the forecast. Its drawbacks: it needs more compute than a machine learning model (usually a GPU), some models only return a fixed set of quantiles, and a model trained with features of the domain can still be more accurate on a given dataset. As with the other methods, its coverage should be validated with backtesting.

A comparison of bootstrapping, conformal prediction and quantile regression on the same dataset can be found in the article [Probabilistic forecasting with machine learning](https://cienciadedatos.net/documentos/py42-probabilistic-forecasting).


## Evaluating prediction intervals

A good prediction interval is calibrated (its empirical coverage is close to the nominal one) and sharp (it is as narrow as possible). The coverage ([`calculate_coverage()`](https://skforecast.org/latest/api/metrics#skforecast.metrics.calculate_coverage)) and the width of the interval measure each property separately and are easy to interpret, but neither of them can rank models on its own: a wider interval always achieves a higher coverage. To compare models or tune hyperparameters, use a proper scoring rule that combines both aspects:

+ [`winkler_score()`](https://skforecast.org/latest/api/metrics#skforecast.metrics.winkler_score): a single interval, scored by its width plus a penalty for each observation that falls outside it.

+ [`weighted_interval_score()`](https://skforecast.org/latest/api/metrics#skforecast.metrics.weighted_interval_score), [`crps_from_quantiles()`](https://skforecast.org/latest/api/metrics#skforecast.metrics.crps_from_quantiles) and [`crps_from_predictions()`](https://skforecast.org/latest/api/metrics#skforecast.metrics.crps_from_predictions): several intervals or the whole predictive distribution.

+ [`create_mean_pinball_loss()`](https://skforecast.org/latest/api/metrics#skforecast.metrics.create_mean_pinball_loss): a single quantile, for example to tune the hyperparameters of a quantile regression model.

<div class="skf-anim-embed">
  <iframe src="../animations/probabilistic-evaluation.html" title="Animation: coverage and width of three prediction intervals for the same forecast" loading="lazy" allowfullscreen></iframe>
</div>

The intervals must be evaluated on data not used to train the model. The [`backtesting_forecaster()`](https://skforecast.org/latest/api/model_selection#skforecast.model_selection._validation.backtesting_forecaster) and [`backtesting_forecaster_multiseries()`](https://skforecast.org/latest/api/model_selection#skforecast.model_selection._validation.backtesting_forecaster_multiseries) functions return the intervals of every fold with the `interval` and `interval_method` arguments. When `interval` defines one interval (for example, `[0.1, 0.9]` or `[0.1, 0.5, 0.9]`), the metrics also include its empirical `coverage` and its mean width (`interval_width`), for each series and aggregated in multi-series backtesting. [`backtesting_foundation()`](https://skforecast.org/latest/api/model_selection#skforecast.model_selection._validation.backtesting_foundation) does the same with its `quantiles` argument.

Discover how to use these metrics in [Metrics in probabilistic forecasting](probabilistic-forecasting-metrics.ipynb) and [Continuous Ranked Probability Score (CRPS)](../faq/probabilistic-forecasting-crps-score.ipynb).

!!! warning

    As Rob J Hyndman explains in his [blog](https://robjhyndman.com/hyndsight/narrow-pi/), in real-world problems, almost all prediction intervals are too narrow. For example, nominal 95% intervals may only provide coverage between 71% and 87%. This is a well-known phenomenon that arises because they do not account for all sources of uncertainty. With forecasting models, there are at least four sources of uncertainty:

    + The random error term
    + The parameter estimates
    + The choice of model for the historical data
    + The continuation of the historical data generating process into the future

    When producing prediction intervals for time series models, generally only the first of these sources is taken into account. Therefore, it is advisable to use test data to validate the empirical coverage of the interval and not only rely on the expected one.


## Other probabilistic forecasting options

Besides the methods above, skforecast offers probabilistic forecasts in other contexts:

+ **Global models**: bootstrapping, conformal prediction and quantile regression can also be applied to models trained on multiple series. In global forecasters, `predict_interval()` uses conformal prediction by default. See [Probabilistic forecasting: global models](probabilistic-forecasting-global-models.ipynb).

+ **Statistical models**: [`ForecasterStats`](https://skforecast.org/latest/api/forecasterstats) returns the prediction intervals calculated by the underlying statistical model (ARIMA, SARIMAX, ETS, ARAR) through its `predict_interval()` method.

!!! tip

    For more examples on how to use probabilistic forecasting, check out the following articles:

    + [Probabilistic forecasting with machine learning](https://cienciadedatos.net/documentos/py42-probabilistic-forecasting)
    + [Probabilistic forecasting: prediction intervals for multi-step time series forecasting](https://cienciadedatos.net/documentos/py60-probabilistic-forecasting-prediction-intervals-multi-step-forecasting)
    + [Continuous Ranked Probability Score (CRPS) in probabilistic forecasting](../faq/probabilistic-forecasting-crps-score.ipynb)
    + [Calibration of probabilistic forecasting intervals](../faq/probabilistic-forecasting-calibrate-intervals.ipynb)
