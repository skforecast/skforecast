import cProfile
import pstats
import io
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from skforecast.datasets import fetch_dataset
from skforecast.recursive import ForecasterRecursive
from skforecast.preprocessing import CalendarFeatures

def main():
    data = fetch_dataset('bike_sharing', raw=True)
    data = data[['date_time', 'users', 'holiday', 'weather', 'temp', 'atemp', 'hum', 'windspeed']]
    data['date_time'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H:%M:%S')
    data = data.set_index('date_time')
    data = data.asfreq('h')
    data = data.sort_index()

    # Create forecaster without window_features
    calendar_transformer = CalendarFeatures(
        features = ['month', 'week', 'day_of_week', 'hour'],
        encoding = 'cyclical',
        keep_original_columns = False,
    )
    forecaster = ForecasterRecursive(
        estimator       = LGBMRegressor(random_state=15926, verbose=-1, n_estimators=10),
        lags            = 24,
        calendar_features=calendar_transformer,
    )
    
    y_train = data['users'].iloc[:-100]
    exog_train = data.drop(columns="users").iloc[:-100]
    exog_predict = data.drop(columns="users").tail(100)

    # Warmup
    forecaster.fit(y=y_train, exog=exog_train)
    forecaster.predict(steps=100, exog=exog_predict)
    
    print("--- Profiling FIT ---")
    pr_fit = cProfile.Profile()
    pr_fit.enable()
    for _ in range(5):
        forecaster.fit(y=y_train, exog=exog_train)
    pr_fit.disable()
    s_fit = io.StringIO()
    ps_fit = pstats.Stats(pr_fit, stream=s_fit).sort_stats('cumulative')
    ps_fit.print_stats(50)
    print(s_fit.getvalue())
    
    print("--- Profiling PREDICT ---")
    pr_pred = cProfile.Profile()
    pr_pred.enable()
    for _ in range(5):
        forecaster.predict(steps=100, exog=exog_predict)
    pr_pred.disable()
    s_pred = io.StringIO()
    ps_pred = pstats.Stats(pr_pred, stream=s_pred).sort_stats('cumulative')
    ps_pred.print_stats(50)
    print(s_pred.getvalue())

if __name__ == "__main__":
    main()
