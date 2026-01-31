# Introduction to forecasting

## Time series and forecasting

A **time series** is a sequence of data arranged chronologically and spaced at equal or irregular intervals. The **forecasting** process consists of predicting the future value of a time series, either by modeling the series solely based on its past behavior (autoregressive) or by incorporating other external variables.

## Machine learning for forecasting

To apply machine learning models to forecasting problems, the **time series needs to be transformed into a matrix** where each value is associated with a specific time window (known as **lags**) that precedes it. In the context of time series, a lag with respect to a time step *t* is defined as the value of the series at previous time steps. For instance, lag 1 represents the value at time step *t-1*, while lag *m* represents the value at time step *t-m*.

This transformation is essential for machine learning models to **capture the dependencies and patterns** that exist between past and future values in a time series. By using lags as input features, machine learning models can learn from the past and make predictions about future values. The number of **lags** used as input features in the matrix is an **important hyperparameter** that needs to be carefully tuned to obtain the best performance of the model.

<p style="text-align: center">
    <img src="../img/transform_timeseries.gif" style="width: 500px;">
    <br>
    <font size="2.5"> <i>Time series transformation into a matrix of 5 lags and a vector with the value of the series that follows each row of the matrix</i></font>
</p>

This type of transformation also allows to include additional variables.

<p style="text-align: center">
    <img src="../img/matrix_transformation_with_exog_variable.png" style="width: 500px;">
    <br>
    <font size="2.5"> <i>Time series transformation including an exogenous variable</i></font>
</p>

Once data have been rearranged into the new shape, any regression model can be trained to predict the next value (step) of the series. During model training, every row is considered a separate data instance, where values at lags 1, 2, ... *p* are considered predictors for the target quantity of the time series at time step *p+1*. 

<p style="text-align: center">
    <img src="../img/diagram-training-forecaster.png" style="width: 600px;">
    <br>
    <font size="2.5"> <i>Diagram of training a machine learning model with time series data</i></font>
</p>


## Single-step forecasting

Single-step prediction is used when the goal is to predict only the next value of the series.

<p style="text-align: center">
    <img src="../img/diagram-single-step-forecasting.png" style="width: 500px;">
    <br>
    <font size="2.5"> <i>Diagram of single-step forecasting</i></font>
</p>


## Multi-step forecasting

When working with time series, it is seldom needed to predict only the next element in the series (*t+1*). Instead, the most common goal is to predict a whole future interval (*t+1, ..., t+n*)  or a far point in time (*t+n*). Several strategies allow generating this type of prediction.


### Recursive multi-step forecasting

Since the value *t(n-1)* is required to predict *t(n)*, and *t(n-1)* is unknown, a recursive process is applied in which, **each new prediction, is based on the previous one**. This process is known as **recursive forecasting** or recursive multi-step forecasting and can be easily generated with the [`ForecasterRecursive`](../user_guides/autoregressive-forecaster.ipynb) class.

<p style="text-align: center">
    <img src="../img/diagram-recursive-mutistep-forecasting.png" style="width: 500px">
    <br>
    <font size="2.5"> <i>Diagram of recursive multi-step forecasting</i></font>
</p>
<p style="text-align: center">
    <img src="../img/recursive-forecasting-gif.gif" style="width: 500px; padding: 10px; background-color: white; border-radius: 4px;">
    <br>
    <font size="2.5"> <i>Recursive forecasting</i></font>
</p>


### Direct multi-step forecasting

Direct multi-step forecasting consists of **training a different model for each step** of the forecast horizon. For example, to predict the next 5 values of a time series, 5 different models are trained, one for each step. As a result, the **predictions are independent of each other**. This entire process is automated in the [`ForecasterDirect`](../user_guides/direct-multi-step-forecasting.ipynb) class. 

<p style="text-align: center">
    <img src="../img/diagram-direct-multi-step-forecasting.png" style="width: 500px">
    <br>
    <font size="2.5"> <i>Diagram of direct multi-step forecasting</i></font>
</p>
<p style="text-align: center">
    <img src="../img/direct-forecasting-gif.gif" style="width: 500px; padding: 10px; background-color: white; border-radius: 4px;">
    <br>
    <font size="2.5"> <i>Direct forecasting</i></font>
</p>


### Multiple output forecasting

Some machine learning models, such as long short-term memory (LSTM) neural networks, can **predict multiple values of a sequence simultaneously** (one-shot). This strategy implemented in the <code>ForecasterRnn</code> class.


## Forecasting Classification

While time series forecasting is predominantly treated as a regression problem, where the goal is to predict a continuous numerical value (e.g., temperature, sales units, price), there are scenarios where the objective is to predict a **categorical variable** or a discrete state. This approach is known as **Forecasting Classification**.

In this context, the time series transformation into a matrix of lags remains exactly the same as in regression. The key difference lies in the target variable, which represents a class or label rather than a continuous number.

Common use cases include:

- **Directional prediction:** Predicting whether a stock price will go "Up" or "Down" (Binary classification).

- **Event detection:** Predicting if a system will be in a "Normal", "Warning", or "Critical" state (Multi-class classification).

- **Threshold crossing:** Forecasting if electricity demand will exceed a specific limit (Binary classification).

The [`ForecasterRecursiveClassifier`](../user_guides/autoregressive-classification-forecasting.ipynb) class in skforecast is designed specifically to handle these scenarios, allowing users to apply standard scikit-learn classifiers to time series problems.


## Forecasting with statistical models

While machine learning models require transforming the time series into a matrix of lags, **statistical models** such as **ARIMA**, **SARIMAX**, and **ETS** operate directly on the time series data. These models are based on well-established statistical theory and have been widely used for decades in forecasting applications.

Key characteristics of statistical models:

- **No matrix transformation required:** They work directly with the sequential structure of the time series.

- **Interpretable parameters:** The model coefficients have clear statistical meaning.

- **Built-in uncertainty quantification:** Prediction intervals are derived from the model's probabilistic framework.

- **Automatic parameter selection:** Algorithms like **AutoARIMA** can automatically find optimal hyperparameters using information criteria (AIC, BIC).

Statistical models are particularly well-suited for **univariate forecasting** with clear trend and seasonal patterns, and when **interpretability** is a priority. However, they may struggle with highly complex, non-linear relationships where machine learning approaches excel.

The [`ForecasterStats`](../user_guides/forecasting-statistical-models.html) class in skforecast provides a unified interface for statistical models, allowing them to be used with the same API as machine learning forecasters, including backtesting and hyperparameter tuning capabilities.


## Global forecasting models

**Univariate time series forecasting** focuses on modeling a single time series as a linear or nonlinear function of its own past values (lags), using historical observations to predict future ones. 

**Global forecasting** builds a single predictive model that considers all time series simultaneously. This approach seeks to **learn the shared patterns that underlie the different series**, helping to reduce the influence of noise present in individual time series. It is computationally efficient, easier to maintain, and often yields more robust generalization across series. Two main strategies are used to implement global forecasting models.

### Independent Multi-Series Forecasting

In **independent multi-series forecasting**, a single model is trained using all time series, but each series is treated independently—past values of one series are not used to predict another. Modeling them together is still beneficial when the series share similar temporal dynamics. For example, sales of products A and B in the same store may not be directly related, but both are influenced by the same underlying store-level patterns.

<p style="text-align: center">
    <img src="../img/forecaster_multi_series_train_matrix_diagram.png" style="width: 700px">
    <br>
    <font size="2.5"> <i>Transformation of two time series and an exogenous variable into the matrices needed to train a machine learning model in a multi-series context</i></font>
</p>

To predict the next *n* steps, the strategy of [recursive multi-step forecasting](../introduction-forecasting/introduction-forecasting.md#recursive-multi-step-forecasting) is applied

<p style="text-align: center">
    <img src="../img/forecaster_multi_series_prediction_diagram.png" style="width: 700px">
    <br>
    <font size="2.5"> <i>Diagram of recursive forecasting with multiple independent time series</i></font>
</p>

The [`ForecasterRecursiveMultiSeries`](../user_guides/independent-multi-time-series-forecasting.ipynb) class cover this process. 

### Dependent Multi-Series Forecasting (multivariate time series)

In **dependent multi-series forecasting** (also known as **multivariate time series forecasting**), all series are modeled jointly under the assumption that each series depends not only on its own past values, but also on the past values of the other series. The forecaster is expected to learn both the individual dynamics of each series and the relationships between them.

A typical example is the set of sensor readings (such as flow, temperature, and pressure) collected from an industrial machine like a compressor, where the variables influence each other over time.

<p style="text-align: center">
    <img src="../img/forecaster_multivariate_train_matrix_diagram.png" style="width: 700px">
    <br>
    <font size="2.5"> <i>Transformation of two time series and an exogenous variable into the matrices needed to train a machine learning model in a multi-variate-series context</i></font>
</p>

Using the [`ForecasterDirectMultiVariate`](../user_guides/dependent-multi-series-multivariate-forecasting.ipynb) class, it is possible to easily build machine learning models for **multivariate forecasting**.

### Forecasting with Deep Learning

**Deep learning** models have become increasingly popular for time series forecasting, especially when traditional statistical approaches struggle to capture **non-linear relationships or complex temporal patterns**. By leveraging neural network architectures, deep learning methods can automatically learn features and dependencies directly from raw data, offering significant advantages for large datasets, multivariate time series, and problems where classic models fall short.

Using the [`ForecasterRnn`](../user_guides/forecasting-with-deep-learning-rnn-lstm.ipynb) class, it is possible to easily build machine learning models using **deep learning**.


## Forecasters

A **Forecaster** object in the skforecast library is a comprehensive **container that provides essential functionality and methods** for training a forecasting model and generating predictions for future points in time.

The **skforecast** library offers a **variety of forecaster** types, each tailored to specific requirements such as single or multiple time series, direct or recursive strategies, or custom predictors. Regardless of the specific forecaster type, all instances share the same API.

| Forecaster                   | Single series | Multiple series | Recursive strategy | Direct strategy | Probabilistic prediction | Time series differentiation | Exogenous features | Window features |
|:-----------------------------|:-------------:|:---------------:|:------------------:|:---------------:|:------------------------:|:---------------------------:|:------------------:|:---------------:|
|[ForecasterRecursive]            |✔️||✔️||✔️|✔️|✔️|✔️|
|[ForecasterDirect]               |✔️|||✔️|✔️|✔️|✔️|✔️|
|[ForecasterRecursiveMultiSeries] ||✔️|✔️||✔️|✔️|✔️|✔️|
|[ForecasterDirectMultiVariate]   ||✔️||✔️|✔️|✔️|✔️|✔️|
|[ForecasterRNN]                  |✔️|✔️||✔️|✔️||✔️||
|[ForecasterStats]                |✔️||✔️||✔️|✔️|✔️||
|[ForecasterRecursiveClassifier]  |✔️||✔️||✔️||✔️|✔️|

[ForecasterRecursive]: ../user_guides/autoregressive-forecaster.ipynb
[ForecasterDirect]: ../user_guides/direct-multi-step-forecasting.ipynb
[ForecasterRecursiveMultiSeries]: ../user_guides/independent-multi-time-series-forecasting.ipynb
[ForecasterDirectMultiVariate]: ../user_guides/dependent-multi-series-multivariate-forecasting.ipynb
[ForecasterRNN]: ../user_guides/forecasting-with-deep-learning-rnn-lstm.ipynb
[ForecasterStats]: ../user_guides/forecasting-sarimax-arima.ipynb
[ForecasterRecursiveClassifier]: ../user_guides/autoregressive-classification-forecasting.ipynb

To understand what can be done when initializing a forecaster with **skforecast** visit [Forecaster parameters](../quick-start/forecaster-parameters.md) and [Forecaster attributes](../quick-start/forecaster-attributes.ipynb).


## Exogenous variables (features)

Exogenous variables are predictors that are independent of the model being used for forecasting, and **their future values must be known** in order to include them in the prediction process. The inclusion of exogenous variables can **enhance the accuracy of forecasts**.

In skforecast, [exogenous variables](../user_guides/exogenous-variables.ipynb) can be easily included as predictors in all forecasting models. To ensure that their effects are accurately accounted for, it is crucial to include these variables during both the training and prediction phases. This will help to optimize the accuracy of forecasts and provide more reliable predictions.

<p style="text-align: center">
    <img src="../img/matrix_transformation_with_exog_variable.png" style="width: 550px;">
    <br>
    <font size="2.5"> <i>Time series transformation including an exogenous variable</i></font>
</p>


## Backtesting forecasting models

In time series forecasting, the process of *backtesting* consists of **evaluating the performance** of a predictive model by applying it retrospectively to **historical data**. Therefore, it is a special type of cross-validation applied to the previous period(s).

The purpose of backtesting is to evaluate the accuracy and effectiveness of a model and **identify any potential issues or areas of improvement**. By testing the model on historical data, one can assess how well it performs on data that it has not seen before. This is an important step in the modeling process, as it helps to **ensure that the model is robust and reliable**.

Backtesting can be done using a **variety of techniques**, such as simple train-test splits or more sophisticated methods like rolling windows or expanding windows. The choice of method depends on the specific needs of the analysis and the characteristics of the time series data.

Overall, backtesting is an **essential step** in the development of a time series forecasting model. By rigorously testing the model on historical data, one can improve its accuracy and ensure that it is effective at predicting future values of the time series.


### Backtesting without refit

Backtesting without refit is a strategy where the **model is trained only once** and used sequentially without updating it, following the temporal order of the data. This approach is advantageous as it is much faster than other methods that require retraining the model each time. However, the model may lose its predictive power over time as it does not incorporate the latest information available.

<p style="text-align: center">
    <img src="../img/diagram-backtesting-no-refit.png" style="width: 500px;">
</p>

<p style="text-align: center">
    <img src="../img/time-series-backtesting-forecasting-no-refit.gif" style="width: 650px;">
    <br>
    <font size="2.5"> <i>Backtesting without refit</i></font>
</p>


### Backtesting with refit and increasing training size (fixed origin)

In this approach, the **model is trained before making predictions each time**, and all available data up to that point is used in the training process. This differs from standard cross-validation, where the data is randomly distributed between training and validation sets.

Instead of randomizing the data, this backtesting sequentially increases the size of the training set while maintaining the temporal order of the data. By doing this, the model can be tested on progressively larger amounts of historical data, providing a more accurate assessment of its predictive capabilities.

<p style="text-align: center">
    <img src="../img/diagram-backtesting-refit.png" style="width: 500px;">
</p>

<p style="text-align: center">
    <img src="../img/time-series-backtesting-forecasting-with-refit.gif" style="width: 650px;">
    <br>
    <font size="2.5"> <i>Backtesting with refit and increasing training size (fixed origin)</i></font>
</p>


### Backtesting with refit and fixed training size (rolling origin)

In this approach, the **model is trained using a fixed window of past observations**, and the testing is performed on a rolling basis, where the training window is moved forward in time. The size of the training window is kept constant, allowing for the model to be tested on different sections of the data. This technique is particularly useful when there is a limited amount of data available, or when the data is non-stationary, and the model's performance may vary over time. Is also known as time series cross-validation or walk-forward validation.

<p style="text-align: center">
    <img src="../img/diagram-backtesting-refit-fixed-train-size.png" style="width: 500px;">
</p>

<p style="text-align: center">
    <img src="../img/time-series-backtesting-forecasting-refit-fixed-train-size.gif" style="width: 650px;">
    <br>
    <font size="2.5"> <i>Backtesting with refit and fixed training size (rolling origin)</i></font>
</p>


### Backtesting with intermittent refit

The model is **retrained every *n* iterations**, which is useful when the retraining frequency differs from the prediction frequency. For example, a model may be updated weekly but used to generate daily forecasts.

This strategy usually achieves a **good balance** between the **computational cost** of retraining and **avoiding model degradation**. It can be applied with either a fixed or rolling origin, offering flexibility in how the model adapts to new data.

<p style="text-align: center">
    <img src="../img/time-series-backtesting-forecasting-intermittent-refit.gif" style="width: 650px;">
    <br>
    <font size="2.5"> <i>Backtesting with intermittent refit</i></font>
</p>


### Backtesting including gap

This approach introduces a **time gap between the training and test sets**, replicating a scenario where predictions cannot be made immediately after the end of the training data.

For example, consider the goal of predicting the 24 hours of day D+1, but the predictions need to be made at 11:00 to allow sufficient flexibility. At 11:00 on day D, the task is to forecast hours [12 - 23] of the same day and hours [0 - 23] of day D+1. Thus, a total of 36 hours into the future must be predicted, with only the last 24 hours to be stored.

<p style="text-align: center">
    <img src="../img/time-series-backtesting-forecasting-with-gap.gif" style="width: 650px;">
    <br>
    <font size="2.5"> <i>Backtesting with refit and gap</i></font>
</p>


## Backtesting with fold stride

This method involves **controlling how far the test set moves forward between folds**. For example, you might want to forecast the next 30 days but generate a new forecast every 7 days. In this setup, each test window is 30 days long (`steps=30`), while the stride between folds is one week (`fold_stride=7`). This means forecasts overlap, and the same observations are predicted multiple times, which provides a richer view of model consistency and stability across different forecast origins.

<p style="text-align: center">
    <img src="../img/time-series-backtesting-forecasting-fold-stride.gif" style="width: 650px;">
    <br>
    <font size="2.5"> <i>Backtesting with fold stride</i></font>
</p>


### Which strategy should I use?

To ensure an accurate evaluation of your model and gain confidence in its predictive performance on new data, it is critical to employ an **appropriate backtesting strategy**.

The choice depends on **factors** such as the use case, available computing resources, and the time interval between predictions. These will determine both the **prediction horizon** and the **refit strategy**.

+ **Prediction horizon**: Suppose you need to forecast the users of an app every Monday for the whole week. In this case, each iteration of backtesting would be a seven-step prediction, representing the seven days of the week.
  
+ **Refit strategy**: Continuing with the example above, at the end of each week you must decide whether to update the model. Retraining with additional data can improve accuracy but increases time and computational cost. A reasonable approach is to retrain only when the error metric shows a consistent upward trend. This behavior can be effectively simulated with the backtesting framework.

As an illustration, backtesting is performed using the data from this [skforecast example](https://www.cienciadedatos.net/documentos/py29-forecasting-electricity-power-demand-python.html). The same backtest is run with `steps=24` (predicting 24 hours ahead) and different `refit` strategies: `False` (no refit between predictions), re-fitting every `30` days, every `14` days, every `7` days, and `True` (re-fitting after every prediction). Notice that the significant increase in computation time does not necessarily correspond to lower error.

| refit value | execution time (s) | metric |
|------------:|-------------------:|-------:|
|       False |                1.4 |  262.5 |
|          30 |                4.0 |  263.4 |
|          14 |                6.3 |  262.5 |
|           7 |               11.1 |  261.4 |
|        True |               69.1 |  258.3 |

!!! tip

    **Best practice:** The reliability of backtesting results depends on how well the setup mirrors the real-world use case. The closer the backtesting process is to the actual conditions under which the model will operate — including prediction horizon, refit frequency, and data availability — the more trustworthy the estimated performance metrics will be.

For a code example illustrating the backtesting process, refer to the [Backtesting user guide](../user_guides/backtesting.ipynb).
