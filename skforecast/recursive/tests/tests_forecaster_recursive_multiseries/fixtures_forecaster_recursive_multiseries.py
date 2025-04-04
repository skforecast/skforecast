# Fixtures _forecaster_recursive_multiseries
# ==============================================================================
import numpy as np
import pandas as pd

# Fixtures
# np.random.seed(123)
# series_1 = np.random.rand(50)
# series_2 = np.random.rand(50)
# exog = np.random.rand(50)
series = pd.DataFrame(
             {'1': pd.Series(np.array(
                       [0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897,
                        0.42310646, 0.9807642 , 0.68482974, 0.4809319 , 0.39211752,
                        0.34317802, 0.72904971, 0.43857224, 0.0596779 , 0.39804426,
                        0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
                        0.63440096, 0.84943179, 0.72445532, 0.61102351, 0.72244338,
                        0.32295891, 0.36178866, 0.22826323, 0.29371405, 0.63097612,
                        0.09210494, 0.43370117, 0.43086276, 0.4936851 , 0.42583029,
                        0.31226122, 0.42635131, 0.89338916, 0.94416002, 0.50183668,
                        0.62395295, 0.1156184 , 0.31728548, 0.41482621, 0.86630916,
                        0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453])
                   ), 
              '2': pd.Series(np.array(
                       [0.12062867, 0.8263408 , 0.60306013, 0.54506801, 0.34276383,
                        0.30412079, 0.41702221, 0.68130077, 0.87545684, 0.51042234,
                        0.66931378, 0.58593655, 0.6249035 , 0.67468905, 0.84234244,
                        0.08319499, 0.76368284, 0.24366637, 0.19422296, 0.57245696,
                        0.09571252, 0.88532683, 0.62724897, 0.72341636, 0.01612921,
                        0.59443188, 0.55678519, 0.15895964, 0.15307052, 0.69552953,
                        0.31876643, 0.6919703 , 0.55438325, 0.38895057, 0.92513249,
                        0.84167   , 0.35739757, 0.04359146, 0.30476807, 0.39818568,
                        0.70495883, 0.99535848, 0.35591487, 0.76254781, 0.59317692,
                        0.6917018 , 0.15112745, 0.39887629, 0.2408559 , 0.34345601])
                   )
             }
         )

series_2 = series.copy()
series_2.columns = ['l1', 'l2']
series_2.index = pd.date_range(start='2000-01-01', periods=len(series_2), freq='D')

series_as_dict = series_2.copy().to_dict(orient='series')


exog = pd.DataFrame(
           {'exog_1': pd.Series(np.array(
                          [0.51312815, 0.66662455, 0.10590849, 0.13089495, 0.32198061,
                           0.66156434, 0.84650623, 0.55325734, 0.85445249, 0.38483781,
                           0.3167879 , 0.35426468, 0.17108183, 0.82911263, 0.33867085,
                           0.55237008, 0.57855147, 0.52153306, 0.00268806, 0.98834542,
                           0.90534158, 0.20763586, 0.29248941, 0.52001015, 0.90191137,
                           0.98363088, 0.25754206, 0.56435904, 0.80696868, 0.39437005,
                           0.73107304, 0.16106901, 0.60069857, 0.86586446, 0.98352161,
                           0.07936579, 0.42834727, 0.20454286, 0.45063649, 0.54776357,
                           0.09332671, 0.29686078, 0.92758424, 0.56900373, 0.457412  ,
                           0.75352599, 0.74186215, 0.04857903, 0.7086974 , 0.83924335])
                      ),
            'exog_2': ['a']*25 + ['b']*25}
       )

exog_as_dict = {
    'l1': exog.copy(),
    'l2': exog['exog_1'].copy()
}

exog_as_dict_datetime = {
    'l1': exog.copy(),
    'l2': exog['exog_1'].copy()
}
exog_as_dict_datetime['l1'].index = pd.date_range(start='2000-01-01', periods=len(exog), freq='D')
exog_as_dict_datetime['l2'].index = pd.date_range(start='2000-01-01', periods=len(exog), freq='D')

exog_predict = exog.copy()
exog_predict.index = pd.RangeIndex(start=50, stop=100)


def expected_df_to_long_format(
    df: pd.DataFrame, method: str = "predict"
) -> pd.DataFrame:
    """
    Convert DataFrame with predictions (one column per level) to long format.
    """

    if method == "predict":
        df = (
            df.melt(var_name="level", value_name="pred", ignore_index=False)
            .reset_index()
            .sort_values(by=["index", "level"])
            .set_index("index")
            .rename_axis(None, axis=0)
        )
    elif method == "bootstrapping":
        df = (
            pd.concat([value.assign(level=key) for key, value in df.items()])
            .reset_index()
            .sort_values(by=["index", "level"])
            .set_index("index")
            .rename_axis(None, axis=0)
        )
        df = df[
            ["level"] + [col for col in df.columns if col not in ["level", "index"]]
        ]
        if isinstance(df.index, pd.DatetimeIndex) and df.index.freq is not None:
            df.index.freq = None
    elif method == "interval":
        df = df.melt(var_name="level", value_name="pred", ignore_index=False).reset_index()
        df['level_aux'] = df['level'].str.replace(r'_lower_bound|_upper_bound', '', regex=True)
        df['bound_type'] = df['level'].str.extract(r'(lower_bound|upper_bound)$', expand=False).fillna('pred')

        df = (
            df.pivot_table(index=["index", "level_aux"], columns="bound_type", values="pred")
            .reset_index()
            .sort_values(by=["index", "level_aux"])
            .set_index("index")
            .rename_axis(None, axis=0)
            .rename_axis(None, axis=1)
            .rename(columns={"level_aux": "level"})
            [['level', 'pred', 'lower_bound', 'upper_bound']]
        )

    return df
