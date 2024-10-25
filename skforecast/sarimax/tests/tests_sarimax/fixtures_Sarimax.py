# Fixtures Sarimax
# ==============================================================================
import numpy as np
import pandas as pd

# Fixtures
# From 'https://raw.githubusercontent.com/skforecast/skforecast/master/data/h2o_exog.csv'

# Series y from observation 0 to 50
y = pd.Series(
        data = np.array(
                   [0.379808  , 0.361801  , 0.410534  , 0.48338867, 0.47546342,
                    0.53476104, 0.56860613, 0.59522329, 0.77125778, 0.7515028 ,
                    0.38755434, 0.42728322, 0.41389018, 0.42885882, 0.47012642,
                    0.50920969, 0.558443  , 0.60151406, 0.6329471 , 0.69960539,
                    0.96308051, 0.81932534, 0.4376698 , 0.50612127, 0.47049117,
                    0.51069626, 0.54051379, 0.55811892, 0.67285206, 0.68589738,
                    0.68969198, 0.74130358, 0.81330763, 0.80311257, 0.47525824,
                    0.5525723 , 0.52710782, 0.56124982, 0.58897764, 0.62313362,
                    0.74083723, 0.72537176, 0.81580302, 0.81400947, 0.92665305,
                    0.93727594, 0.52876165, 0.55933994, 0.57787166, 0.61492741]
               ),
        name = 'y',
        index = pd.RangeIndex(start=0, stop=50, step=1)
    )

# Series y from observation 50 to 100
y_lw = pd.Series(
        data = np.array(
                   [0.59418877, 0.70775844, 0.71950195, 0.74432369, 0.80485511,
                    0.78854235, 0.9710894 , 0.84683354, 0.46382252, 0.48527317,
                    0.5280586 , 0.56233647, 0.5885704 , 0.66948036, 0.67799365,
                    0.76299549, 0.79972374, 0.77052192, 0.99438934, 0.80054443,
                    0.49055721, 0.52440799, 0.53664948, 0.55209054, 0.60336564,
                    0.68124538, 0.67807535, 0.79489265, 0.7846239 , 0.8130087 ,
                    0.9777323 , 0.89308148, 0.51269597, 0.65299589, 0.5739764 ,
                    0.63923842, 0.70387188, 0.77064824, 0.84618588, 0.89272889,
                    0.89789988, 0.94728069, 1.05070727, 0.96965567, 0.57329151,
                    0.61850684, 0.61899573, 0.66520922, 0.72652015, 0.85586494]),
        name = 'y',
        index = pd.RangeIndex(start=50, stop=100, step=1)
    )

# Series exog_2 from observation 0 to 50
exog = pd.Series(
           data = np.array(
                      [1.1660294 , 1.1178592 , 1.0679422 , 1.09737593, 1.12219902,
                       1.15318963, 1.19455065, 1.23148851, 1.28906233, 1.34427021,
                       1.31482887, 1.28656429, 1.25029766, 1.18181787, 1.1255426 ,
                       1.14987367, 1.17610562, 1.2136304 , 1.25444805, 1.30034385,
                       1.39111801, 1.44329448, 1.41052563, 1.38516046, 1.33933762,
                       1.24886077, 1.19309846, 1.21718828, 1.25053444, 1.29361568,
                       1.32941483, 1.36957278, 1.42061053, 1.44666263, 1.4045348 ,
                       1.37711086, 1.33427171, 1.28386015, 1.24103316, 1.27060824,
                       1.30826123, 1.34791401, 1.39882465, 1.44383102, 1.50453491,
                       1.54382265, 1.50450063, 1.45320801, 1.40598045, 1.34363532]
                  ),
           name = 'exog',
           index = pd.RangeIndex(start=0, stop=50, step=1)
       )

# Series exog_2 from observation 50 to 100
exog_lw = pd.Series(
           data = np.array([
                      1.27501789, 1.31081724, 1.34284965, 1.37614005, 1.41412559,
                      1.45299631, 1.5056625 , 1.53112882, 1.47502858, 1.4111122 ,
                      1.35901545, 1.27726486, 1.22561223, 1.2667438 , 1.3052879 ,
                      1.35227527, 1.39975273, 1.43614303, 1.50112483, 1.52563498,
                      1.47114733, 1.41608418, 1.36930969, 1.28084993, 1.24141417,
                      1.27955181, 1.31028528, 1.36193391, 1.40844058, 1.4503692 ,
                      1.50966658, 1.55266781, 1.49622847, 1.46990287, 1.42209641,
                      1.35439763, 1.31655571, 1.36814617, 1.40678416, 1.47053466,
                      1.52226695, 1.57094872, 1.62696052, 1.65165448, 1.587767  ,
                      1.5318884 , 1.4662314 , 1.38913179, 1.34050469, 1.39701938]
                  ),
           name = 'exog',
           index = pd.RangeIndex(start=50, stop=100, step=1)
       )

# Series exog_2 from observation 50 to 60
exog_predict = pd.Series(
                   data = np.array(
                              [1.27501789, 1.31081724, 1.34284965, 1.37614005, 1.41412559,
                               1.45299631, 1.5056625 , 1.53112882, 1.47502858, 1.4111122 ]
                          ),
                   name = 'exog',
                   index = pd.RangeIndex(start=50, stop=60, step=1)
               )

# Series exog_2 from observation 100 to 110
exog_lw_predict = pd.Series(
                   data = np.array(
                              [1.44651487, 1.48776549, 1.54580785, 1.58822301, 1.6196549 ,
                               1.65521912, 1.5922988 , 1.5357284 , 1.47770322, 1.41592127]
                          ),
                   name = 'exog',
                   index = pd.RangeIndex(start=100, stop=110, step=1)
               )


# Datetime Series
y_datetime = pd.Series(data=y.to_numpy())
y_datetime.index = pd.date_range(start='2000', periods=50, freq='YE')
y_datetime.name = 'y'

y_lw_datetime = pd.Series(data=y_lw.to_numpy())
y_lw_datetime.index = pd.date_range(start='2050', periods=50, freq='YE')
y_lw_datetime.name = 'y'

exog_datetime = pd.Series(data=exog.to_numpy())
exog_datetime.index = pd.date_range(start='2000', periods=50, freq='YE')
exog_datetime.name = 'exog'

exog_lw_datetime = pd.Series(data=exog_lw.to_numpy())
exog_lw_datetime.index = pd.date_range(start='2050', periods=50, freq='YE')
exog_lw_datetime.name = 'exog'

exog_predict_datetime = pd.Series(data=exog_predict.to_numpy())
exog_predict_datetime.index = pd.date_range(start='2050', periods=10, freq='YE')
exog_predict_datetime.name = 'exog'

exog_lw_predict_datetime = pd.Series(data=exog_lw_predict.to_numpy())
exog_lw_predict_datetime.index = pd.date_range(start='2100', periods=10, freq='YE')
exog_lw_predict_datetime.name = 'exog'


# Pandas DataFrames
df_exog = pd.DataFrame({
              'exog_1': exog.to_numpy(),
              'exog_2': ['a', 'b']*25}
          )

df_exog_lw = pd.DataFrame({
                 'exog_1': exog_lw.to_numpy(),
                 'exog_2': ['a', 'b']*25}
             )
df_exog_lw.index = pd.RangeIndex(start=50, stop=100)

df_exog_predict = pd.DataFrame({
                      'exog_1': exog_predict.to_numpy(),
                      'exog_2': ['a', 'b']*5}
                  )
df_exog_predict.index = pd.RangeIndex(start=50, stop=60)

df_exog_lw_predict = pd.DataFrame({
                         'exog_1': exog_lw_predict.to_numpy(),
                         'exog_2': ['a', 'b']*5}
                     )
df_exog_lw_predict.index = pd.RangeIndex(start=100, stop=110)

df_exog_datetime = df_exog.copy()
df_exog_datetime.index = pd.date_range(start='2000', periods=50, freq='YE')

df_exog_lw_datetime = df_exog_lw.copy()
df_exog_lw_datetime.index = pd.date_range(start='2050', periods=50, freq='YE')

df_exog_predict_datetime = df_exog_predict.copy()
df_exog_predict_datetime.index = pd.date_range(start='2050', periods=10, freq='YE')

df_exog_lw_predict_datetime = df_exog_lw_predict.copy()
df_exog_lw_predict_datetime.index = pd.date_range(start='2100', periods=10, freq='YE')