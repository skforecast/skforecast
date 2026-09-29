# Global forecasting

Many forecasting problems involve not one time series but many: sales of hundreds of products, demand in dozens of stores, readings from a fleet of sensors. There are two ways to model them.

+ **Local models**: one model per series. Each model only learns from the history of its own series.

+ **Global models**: a single model trained on all the series at once. It learns the patterns they share, such as seasonality, trends or the effect of holidays.

Global models are often more accurate than local ones when the series are related, because every series benefits from what the model learns from the others. This matters most for short or noisy series, which do not have enough history to be modeled on their own. Global models are also easier to maintain: one model to train, tune and deploy instead of hundreds.

<div class="skf-anim-embed">
  <iframe src="../animations/global-forecasting.html" title="Animation: local and global forecasting models" loading="lazy" allowfullscreen></iframe>
</div>

Skforecast implements several global forecasters, each suited to a different kind of problem.

| Forecaster | How the series are used | Typical use |
|:-----------|:------------------------|:------------|
| [`ForecasterRecursiveMultiSeries`](independent-multi-time-series-forecasting.ipynb) | Each series is predicted from its own past values. The model is shared by all of them. | Many related series: products, stores, customers |
| [`ForecasterDirectMultiVariate`](dependent-multi-series-multivariate-forecasting.ipynb) | One series is predicted from the past values of all the series. | A few series that influence each other: sensors of a machine |
| [`ForecasterFoundation`](foundation-forecasting-models.ipynb) | A pre-trained model forecasts many series in one call, without training. | Accurate forecasts without training, even for short or new series |
| [`ForecasterRnn`](forecasting-with-deep-learning-rnn-lstm.ipynb) | A recurrent neural network learns from several series at once. | Large datasets with complex, non-linear patterns |

All of them produce both point forecasts and probabilistic forecasts (prediction intervals). Foundation models return them directly, while the other forecasters estimate them from the errors of the model, with bootstrapping or conformal prediction. Discover how in [Probabilistic global models](probabilistic-forecasting-global-models.ipynb).


### Independent multi-series forecasting

In independent multi-series forecasting, a single model is trained with all the series, but each series is predicted only from its own past values (and its exogenous variables). The series are related through the model, not through their values: sales of two products in the same store may not depend on each other, yet both follow the same weekly pattern and react to the same holidays.

`ForecasterRecursiveMultiSeries` implements this strategy with any scikit-learn compatible regressor, such as LightGBM or XGBoost. By default, an identifier of each series is added as a feature (`encoding`), so the model can learn what is common to all series and what is specific to each one.

<div class="skf-anim-embed">
  <iframe src="../animations/independent-multi-series.html" title="Animation: independent multi-series forecasting" loading="lazy" allowfullscreen></iframe>
</div>

Discover how to use it in [Independent multi-time series forecasting](independent-multi-time-series-forecasting.ipynb).


### Dependent multi-series forecasting (multivariate)

In dependent multi-series forecasting, also known as multivariate forecasting, the series are assumed to influence each other. The past values of every series are used as features to predict one of them. A typical example is the set of sensors of an industrial compressor, where flow, temperature and pressure depend on each other over time.

`ForecasterDirectMultiVariate` implements this strategy. It predicts one series (`level`) and uses the direct strategy, with one model for each step of the forecast horizon.

<div class="skf-anim-embed">
  <iframe src="../animations/dependent-multi-series.html" title="Animation: dependent multi-series forecasting" loading="lazy" allowfullscreen></iframe>
</div>

Discover how to use it in [Dependent multivariate series forecasting](dependent-multi-series-multivariate-forecasting.ipynb).


### Foundation models

Foundation models, such as Chronos-2, TimesFM or Moirai-2, are pre-trained on large collections of time series. They are global by design: a single model forecasts any number of series in one call, without training (zero-shot). Most of them also accept exogenous variables.

`ForecasterFoundation` integrates them with the rest of skforecast, so they can be backtested and compared with any other forecaster.

<div class="skf-anim-embed">
  <iframe src="../animations/foundation-models.html" title="Animation: forecasting with foundation models" loading="lazy" allowfullscreen></iframe>
</div>

Discover how to use them in [Forecasting with foundation models](foundation-forecasting-models.ipynb).


### Deep learning

Recurrent neural networks (RNN and LSTM) can learn from several series at once and predict several series at the same time. They can capture complex, non-linear patterns, but they need more data and more tuning than the other options.

Discover how to use them in [Deep learning, Recurrent Neural Networks](forecasting-with-deep-learning-rnn-lstm.ipynb).


## Real-world series

Real collections of series are rarely clean: products launched at different dates, exogenous variables that only exist for some series, or gaps in the data. `ForecasterRecursiveMultiSeries` and `ForecasterFoundation` are the most flexible: they handle series of different lengths, different exogenous variables for each series and missing values.

Discover how in:

+ [Series with different lengths and different exogenous variables](multi-series-with-different-length-and-different_exog.ipynb)
+ [Foundation models with heterogeneous series](foundation-forecasting-with-heterogeneous-series.ipynb)
+ [Handling missing values](handling-missing-values.ipynb)


## Evaluation and tuning

Global forecasters are evaluated with backtesting, which returns the metric of each series and an aggregated metric for all of them. Use `backtesting_forecaster_multiseries` for the forecasters that are trained, and `backtesting_foundation` for foundation models, which predict each fold from its context without training.

Discover how in [Backtesting](backtesting.ipynb) and [Hyperparameter tuning and lags selection](hyperparameter-tuning-and-lags-selection.ipynb).


## Which one to use?

There is no single answer, since it depends on the data. However, some general guidelines can be followed:

+ **`ForecasterRecursiveMultiSeries`**: the best starting point when there are many related series. It is fast, scales to thousands of series and handles series of different lengths.

+ **`ForecasterDirectMultiVariate`**: when there are a few series and the past of some of them helps to predict the others.

+ **`ForecasterFoundation`**: accurate forecasts out of the box, with no model to train or tune, even for short or new series. Foundation models are among the best performers in public benchmarks, but they need more computing resources to predict (usually a GPU), and a model trained on your data with well-designed features can still be more accurate for a specific problem. It requires installing the library of the foundation model.

+ **`ForecasterRnn`**: when there is a large amount of data and the patterns are too complex for the other options.

In every case, compare the candidates with backtesting on the same data before choosing one.

!!! tip

    For more examples of global forecasting models, check out the following articles:

    + [Global Forecasting Models I: Multi-series forecasting](https://www.cienciadedatos.net/documentos/py44-multi-series-forecasting-skforecast.html)
    + [Global Forecasting Models II: Comparative analysis of single and multi-series forecasting](https://www.cienciadedatos.net/documentos/py53-global-forecasting-models.html)
    + [Global Forecasting Models III: Modeling thousand time series with a single global model](https://www.cienciadedatos.net/documentos/py59-scalable-forecasting-models.html)
    + [Global Forecasting Models IV: A step by step guide using Kaggle sticker sales data](https://cienciadedatos.net/documentos/py66-forecasting-sticker-sales-kaggle.html)
