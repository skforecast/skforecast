---
title: Forecasting with ci-proto
description: Forecast with ci-proto in skforecast using past and known-future covariates, categorical inputs, calendar dates, native quantiles and long contexts.
---

# Forecasting with ci-proto

[ci-proto](https://huggingface.co/ContinuousIntelligence/ci-proto) is a pretrained
forecasting model from Continuous Intelligence. The adapter uses the published
Python backend, supporting context windows of up to **2 million observations**
and automatic selection between variants tuned for shorter and longer histories.

## Installation and first forecast

Install the optional backend separately:

~~~bash
pip install skforecast "continuousintelligence>=0.2.0"
~~~

Use Python 3.11 to 3.14, as required by skforecast. On Linux x86-64 the backend
dependencies include the CUDA compiler toolchain. CPU inference is supported
through the same API.

~~~python
import numpy as np
import pandas as pd
from skforecast.foundation import FoundationModel, ForecasterFoundation

index = pd.date_range("2026-01-01", periods=384, freq="h")
sales = pd.Series(
    20 + np.sin(np.arange(384) * 2 * np.pi / 24),
    index=index,
    name="sales",
)
model = FoundationModel(
    model_id="ContinuousIntelligence/ci-proto",
    device="cpu",
    # revision="<full Hugging Face commit SHA>",
)
forecaster = ForecasterFoundation(estimator=model)
forecaster.fit(series=sales)
prediction = forecaster.predict(steps=24)
interval = forecaster.predict_interval(steps=24, interval=[0.1, 0.9])
quantiles = forecaster.predict_quantiles(steps=24, quantiles=[0.1, 0.5, 0.9])
~~~

The backend is loaded at the first prediction. Fitting stores observations and
metadata without training model weights. A point forecast is the median. The
published checkpoint supplies the nine deciles from 0.1 to 0.9; other quantiles
are rejected. Quantile rows are matched against the loaded checkpoint's grid.

## Covariates and multiple series

Past-only columns in historical exog become past covariates. Columns also
provided in future exog become known-future covariates, with their history
followed by their future values. Numeric, boolean, nullable and categorical
columns are supported. Categorical labels receive per-series codes consistent
over history and horizon; missing values remain missing.

Each series is forecast independently using its own history, covariates and
forecast origin. Series may have different lengths and different covariate
columns.

DatetimeIndex inputs supply actual history and future timestamps in epoch
nanoseconds, so the tensor backend uses UTC for calendar features. Monthly and
other calendar frequencies are expanded using pandas offsets. RangeIndex
inputs have no calendar features.

## Context window

ci-proto supports context windows of up to **2 million observations**. By
default the adapter retains and forwards all supplied history. The backend
automatically selects the variant tuned for shorter contexts up to 16,384
observations, or the variant tuned for longer contexts. The switch at 16,384
does not crop the history, and no extra setting is needed to use longer inputs.

`context_length=None`, the default, keeps all available history during fitting,
prediction and backtesting. Set a positive `context_length` only when you want
to impose a limit on retained history.

For evaluation use the existing skforecast
[foundation backtesting workflow](foundation-forecasting-models.ipynb).

## Licence

The code and weights use the
[ci-proto Code and Model Licence](https://github.com/continuousintelligence/ci/blob/main/LICENCE).

**In brief:** Free for research and commercial use, subject to the attribution
and notification requirements in Section 10. Redistribution and use in
externally available products or services require public attribution and a
one-time notification to Continuous Intelligence. Purely internal use is exempt
from these additional requirements.
