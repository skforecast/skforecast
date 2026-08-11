from line_profiler import LineProfiler
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from skforecast.datasets import fetch_dataset
from skforecast.recursive import ForecasterRecursive
from skforecast.preprocessing import CalendarFeatures
from skforecast.preprocessing import RollingFeatures
from skforecast.preprocessing._calendar import create_calendar_features

def main():
    data = fetch_dataset('bike_sharing', raw=True)
    data = data[['date_time', 'users', 'holiday', 'weather', 'temp', 'atemp', 'hum', 'windspeed']]
    data['date_time'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H:%M:%S')
    data = data.set_index('date_time')
    data = data.asfreq('h')
    data = data.sort_index()

    window_features = RollingFeatures(stats=["mean"], window_sizes=24 * 3)
    calendar_transformer = CalendarFeatures(
        features = ['month', 'week', 'day_of_week', 'hour'],
        encoding = 'cyclical',
        keep_original_columns = False,
    )
    forecaster = ForecasterRecursive(
        estimator       = LGBMRegressor(random_state=15926, verbose=-1, n_estimators=10),
        lags            = 24,
        window_features = window_features,
        calendar_features=calendar_transformer,
    )
    
    y_train = data['users'].iloc[:-100]
    exog_train = data.drop(columns="users").iloc[:-100]
    exog_predict = data.drop(columns="users").tail(100)

    lp = LineProfiler()
    # Add methods to profile
    lp.add_function(forecaster._create_train_X_y)
    lp.add_function(forecaster._recursive_predict)
    lp.add_function(window_features.transform)
    lp.add_function(window_features._transform_vectorized)
    lp.add_function(calendar_transformer.fit)
    lp.add_function(calendar_transformer.transform)
    lp.add_function(create_calendar_features)

    def run():
        forecaster.fit(y=y_train, exog=exog_train)
        forecaster.predict(steps=100, exog=exog_predict)
    
    lp_wrapper = lp(run)
    lp_wrapper()
    lp.print_stats()

if __name__ == "__main__":
    main()
