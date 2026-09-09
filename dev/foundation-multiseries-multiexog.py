# Libraries
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.foundation import FoundationModel, ForecasterFoundation
from skforecast.preprocessing import (
    reshape_series_long_to_dict, 
    reshape_exog_long_to_dict, 
)
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_foundation,
)

# Load time series of multiple lengths and exogenous variables
# ==============================================================================
series = pd.read_csv(
    'https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/demo_multi_series.csv'
)
exog = pd.read_csv(
    'https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/demo_multi_series_exog.csv'
)

series['timestamp'] = pd.to_datetime(series['timestamp'])
exog['timestamp'] = pd.to_datetime(exog['timestamp'])


# Transform series and exog to dictionaries
# ==============================================================================
series_dict = reshape_series_long_to_dict(
    data      = series,
    series_id = 'series_id',
    index     = 'timestamp',
    values    = 'value',
    freq      = 'D'
)

exog_dict = reshape_exog_long_to_dict(
    data      = exog,
    series_id = 'series_id',
    index     = 'timestamp',
    freq      = 'D'
)


# Drop some exogenous variables for series 'id_1000' and 'id_1003'
# ==============================================================================
# Some exogenous variables are intentionally omitted for series 1 and 3 to illustrate that each series can use a different set of exogenous variables.
exog_dict['id_1000'] = exog_dict['id_1000'].drop(columns=['air_temperature', 'wind_speed'])
exog_dict['id_1003'] = exog_dict['id_1003'].drop(columns=['cos_day_of_week'])


# Partition data in train and test
# ==============================================================================
end_train = '2016-07-31 23:59:00'
series_dict_train = {k: v.loc[: end_train,] for k, v in series_dict.items()}
exog_dict_train   = {k: v.loc[: end_train,] for k, v in exog_dict.items()}
series_dict_test  = {k: v.loc[end_train:,] for k, v in series_dict.items()}
exog_dict_test    = {k: v.loc[end_train:,] for k, v in exog_dict.items()}


# Description of each partition
# ==============================================================================
for k in series_dict.keys():
    print(f"{k}:")
    try:
        print(
            f"\tTrain: len={len(series_dict_train[k])}, {series_dict_train[k].index[0]}"
            f" --- {series_dict_train[k].index[-1]}"
        )
    except IndexError:
        print("\tTrain: len=0")
    try:
        print(
            f"\tTest : len={len(series_dict_test[k])}, {series_dict_test[k].index[0]}"
            f" --- {series_dict_test[k].index[-1]}"
        )
    except IndexError:
        print("\tTest : len=0")


# Exogenous variables for each series
# ==============================================================================
for k in series_dict.keys():
    print(f"{k}:")
    try:
        print(f"\t{exog_dict[k].columns.to_list()}")
    except IndexError:
        print("\tNo exogenous variables")


# Fit and Predict forecaster
# ==============================================================================
model_ids = [
    "autogluon/chronos-2-small",
    "google/timesfm-3.0-pytorch",
    "google/timesfm-2.5-200m-pytorch",
    "soda-inria/tabicl",
    "priorlabs/tabpfn-ts",
    "theforecastingcompany/t0-alpha",
    "Synthefy/Nori",
    "taharnbl/TS-ICL"
]


for model_id in model_ids:
    try:
        estimator = FoundationModel(model_id=model_id, context_length=500)
        forecaster = ForecasterFoundation(estimator=estimator)
        forecaster.fit(series=series_dict_train, exog=exog_dict_train)
        predictions = forecaster.predict(steps=5, exog=exog_dict_test)
        print(predictions.head(9))
    except Exception as e:
        # Now you will know exactly which model threw the error
        print(f"FAILED - Error with {model_id}: {e}")