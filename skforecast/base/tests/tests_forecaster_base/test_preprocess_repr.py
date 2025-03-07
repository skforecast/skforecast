# Unit test _preprocess_repr ForecasterBase
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from skforecast.recursive import ForecasterRecursiveMultiSeries

# Fixtures
np.random.seed(1)
n_cols = 60
n_data = 10
data = np.random.normal(loc = 0, scale = 1, size = (n_data, n_cols))
cols = [f"col_{i}" for i in range(n_cols)]
exog_cols = [f"exog_{i}" for i in range(n_cols)]
index = pd.date_range(start = "2000-01-01", periods = n_data, freq = "MS")

series = pd.DataFrame(data, columns = cols, index = index)
exog = pd.DataFrame(data, columns = exog_cols, index = index)


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(), 
                          {k: StandardScaler() for k in list(series.columns) + ['_unknown_level']}], 
                         ids = lambda ts: f'transformer_series: {ts}')
def test_output_preprocess_repr(transformer_series):
    """
    Test matrix of lags created properly when lags is 3, steps is 1 and y is
    np.arange(10).
    """    
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=2, transformer_series=transformer_series
    )
    forecaster.fit(series=series, exog=exog)
    results = forecaster._preprocess_repr(
                  regressor          = forecaster.regressor, 
                  training_range_    = forecaster.training_range_, 
                  series_names_in_   = forecaster.series_names_in_, 
                  exog_names_in_     = forecaster.exog_names_in_, 
                  transformer_series = forecaster.transformer_series
              )

    expected = [
        "{'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': False}",
        "'col_0': ['2000-01-01', '2000-10-01'], 'col_1': ['2000-01-01', '2000-10-01'], 'col_2': ['2000-01-01', '2000-10-01'], 'col_3': ['2000-01-01', '2000-10-01'], 'col_4': ['2000-01-01', '2000-10-01'], ..., 'col_55': ['2000-01-01', '2000-10-01'], 'col_56': ['2000-01-01', '2000-10-01'], 'col_57': ['2000-01-01', '2000-10-01'], 'col_58': ['2000-01-01', '2000-10-01'], 'col_59': ['2000-01-01', '2000-10-01']",
        "col_0, col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9, col_10, col_11, col_12, col_13, col_14, col_15, col_16, col_17, col_18, col_19, col_20, col_21, col_22, col_23, col_24, ..., col_35, col_36, col_37, col_38, col_39, col_40, col_41, col_42, col_43, col_44, col_45, col_46, col_47, col_48, col_49, col_50, col_51, col_52, col_53, col_54, col_55, col_56, col_57, col_58, col_59",
        "exog_0, exog_1, exog_2, exog_3, exog_4, exog_5, exog_6, exog_7, exog_8, exog_9, exog_10, exog_11, exog_12, exog_13, exog_14, exog_15, exog_16, exog_17, exog_18, exog_19, exog_20, exog_21, exog_22, exog_23, exog_24, ..., exog_35, exog_36, exog_37, exog_38, exog_39, exog_40, exog_41, exog_42, exog_43, exog_44, exog_45, exog_46, exog_47, exog_48, exog_49, exog_50, exog_51, exog_52, exog_53, exog_54, exog_55, exog_56, exog_57, exog_58, exog_59",
        "StandardScaler()"
    ]
    if isinstance(transformer_series, dict):
        expected[-1] = "'col_0': StandardScaler(), 'col_1': StandardScaler(), 'col_2': StandardScaler(), 'col_3': StandardScaler(), 'col_4': StandardScaler(), ..., 'col_56': StandardScaler(), 'col_57': StandardScaler(), 'col_58': StandardScaler(), 'col_59': StandardScaler(), '_unknown_level': StandardScaler()"

    for r, e in zip(results, expected):
        assert r == e
