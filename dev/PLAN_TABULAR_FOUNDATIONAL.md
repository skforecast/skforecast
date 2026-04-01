# Technical Analysis: Tabular Foundation Models in Skforecast

This analysis evaluates the integration of Tabular Foundation Models (TFMs)—specifically **TabPFN-2.5** and **TabICLv2**—into the `skforecast` framework based on the provided source code for recursive and direct forecasting strategies.

---

## 1. Core API Compatibility
All three forecaster classes analyzed (`ForecasterRecursive`, `ForecasterRecursiveMultiSeries`, and `ForecasterDirect`) require an `estimator` that is compatible with the **scikit-learn API**.

* **Standard Interface**: The estimator must implement `.fit(X, y)` and `.predict(X)`.
* **TFM Alignment**: TabPFN-2.5 and TabICLv2 function as scikit-learn wrappers, making them technically compatible for assignment to the `estimator` parameter in any of these classes.
* **In-Context Learning (ICL)**: Unlike traditional regressors, "fitting" these models primarily involves storing the training data as a context window for the transformer, which then performs inference during the "predict" stage.

---

## 2. Analysis by Forecaster Class

### **A. ForecasterRecursive**
This class turns any compatible estimator into a recursive autoregressive forecaster.
* **Execution Logic**: The `_recursive_predict` method uses an iterative loop to predict one step at a time, using each prediction as an input for the next.
* **Compatibility Challenge**: Because TFMs are transformer-based, executing a `.predict()` call for every step in a long horizon (e.g., 100 steps) will be computationally expensive compared to traditional tree-based models.
* **Feature Names**: The class manages feature names through `X_train_features_names_out_`, which aligns with the TFM's need for structured input.

### **B. ForecasterRecursiveMultiSeries**
This class enables global forecasting across multiple time series using a single estimator.
* **Global Modeling**: It utilizes `encoding` (ordinal or one-hot) to identify different series "levels".
* **TFM Strength**: TFMs are exceptionally strong at handling categorical variables and learning cross-series patterns, making them suitable for this "Global Model" approach.
* **Scaling Limits**: This class often handles large volumes of data. Users must ensure the total number of rows across all series does not exceed the TFM's maximum context window (e.g., 50,000 for TabPFN-2.5).

### **C. ForecasterDirect**
This class trains a separate model for each specific forecast step.
* **Efficiency**: It avoids the recursive loop by calling the estimator once for each step in the horizon.
* **Implementation**: It stores a dictionary of clones in `self.estimators_`, with one instance for each step.
* **Optimal Strategy**: For TFMs, this is the most recommended strategy. It allows the foundation model to perform a single high-accuracy inference pass for a specific horizon rather than 100 sequential passes.

---

## 3. Compatibility Matrix

| Feature | Skforecast Requirement | TFM Compatibility |
| :--- | :--- | :--- |
| **Fit/Predict API** | Must implement standard sklearn methods. | **Full**: Both models are built as sklearn wrappers. |
| **Categorical Input** | Handled via `encoding` in MultiSeries. | **Superior**: TFMs handle categorical data natively. |
| **Probabilistic** | Supports bootstrapping and conformal methods. | **Redundant**: TFMs output native probability distributions. |
| **Exogenous Data** | Supported via matrix concatenation. | **Full**: TFMs treat `exog` as standard sequence features. |
| **Differentiation** | Applies order-based diffing to `y`. | **Supported**: Foundation models see the stationarized data. |

---

## 4. Implementation Recommendations

1. **Prioritize `ForecasterDirect`**: Use this strategy to minimize the number of transformer forward passes, which reduces total inference time.
2. **Cold-Start Scenarios**: Use `ForecasterRecursiveMultiSeries` when you have many series with very little historical data, as the TFM's pre-trained "prior" excels here.
3. **Data Volume Management**: Be mindful of the `window_size` and sample count; TFMs are memory-intensive and have strict maximum row limits.
4. **Probabilistic Forecasting**: While `skforecast` supports bootstrapping, consider utilizing the TFM's native probabilistic output for better calibration.

## 4. Detected Bottlenecks and Performance Issues

The integration of TFMs into `skforecast` introduces several specific performance trade-offs compared to traditional Gradient Boosted Trees.

### **A. Sequential Inference Latency**
* **The Problem**: In `ForecasterRecursive`, the `_recursive_predict` loop triggers a full transformer forward pass for every predicted step. 
* **The Bottleneck**: Since transformer complexity is non-linear relative to context size, inference time grows significantly as the forecast horizon and training history increase.

### **B. Memory Footprint of "Hard Copy" Context**
* **The Problem**: TFMs do not learn weights during fit; they store the training data as "context."
* **The Bottleneck**: When `.fit()` is called, the model stores a transformed **hard copy** of the data—often as high-dimensional KV-cache tensors. For models like TabPFN-2.5, this can cost ~48.8 KB of memory per cell, quickly leading to Out-of-Memory (OOM) errors on large datasets or small VRAM GPUs.

### **C. Multi-Series Scaling Constraints**
* **The Problem**: `ForecasterRecursiveMultiSeries` concatenates all series into a single global training matrix. 
* **The Bottleneck**: Users may hit the **50,000-sample limit** of TabPFN-2.5 or the **100,000-sample limit** of TabICLv2 much faster than they would with traditional models.

### **D. Parallelization vs. VRAM Contention**
* **The Problem**: `ForecasterDirect` fits $N$ independent models in parallel using `joblib`. 
* **The Bottleneck**: While this speeds up training on CPUs, if the TFMs are configured to use a GPU, parallelizing multiple foundation model instances can lead to severe VRAM contention and execution crashes.

---