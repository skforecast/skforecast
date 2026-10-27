"""
Which DatetimeIndex constructions give datetime64[us] (or [s]) in pandas 3
instead of the datetime64[ns] of pandas 2 (whatsnew 3.0.0, "Datetime/timedelta
resolution inference").

    bash dev/pandas3_migration/run_in_both.sh experiments/exp_datetime_resolution.py
"""
import pandas as pd, numpy as np, datetime as dt
print("pandas", pd.__version__)
print("date_range str      ", pd.date_range("2020-01-01", periods=3, freq="D").dtype)
print("date_range str time ", pd.date_range("2020-01-01 00:00:00", periods=3, freq="h").dtype)
print("date_range Timestamp", pd.date_range(pd.Timestamp("2020-01-01"), periods=3, freq="D").dtype)
print("date_range datetime ", pd.date_range(dt.datetime(2020, 1, 1), periods=3, freq="D").dtype)
print("DatetimeIndex str   ", pd.DatetimeIndex(["2020-01-01", "2020-01-02"]).dtype)
print("to_datetime str     ", pd.to_datetime(["2020-01-01", "2020-01-02"]).dtype)
print("to_datetime str ns  ", pd.to_datetime(["2020-01-01 00:00:00.123456789"]).dtype)
print("Timestamp str unit  ", pd.Timestamp("2020-01-01").unit)
print("read_csv parse      ", pd.read_csv(__import__("io").StringIO("d,v\n2020-01-01,1\n2020-01-02,2"), parse_dates=["d"]).d.dtype)
print("Series of datetimes ", pd.Series([dt.datetime(2020, 1, 1)]).dtype)
print("np datetime64[D]    ", pd.DatetimeIndex(np.array(["2020-01-01"], dtype="datetime64[D]")).dtype)
print("asfreq              ", pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"])).asfreq("D").index.dtype)
print("tz-aware date_range ", pd.date_range("2020-01-01", periods=3, freq="D", tz="Europe/Madrid").dtype)
