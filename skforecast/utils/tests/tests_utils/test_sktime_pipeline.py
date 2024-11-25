# Unit test sktime_pipeline
# ==============================================================================

import pytest
import datetime
import numpy as np
import pandas as pd
from skforecast.utils import StartSktimePipe, EndSktimePipe


def test_StartSktimePipe_transforms_input_as_df_with_datetimeindex_WSUN():
    """
    Check if the input DataFrame with specified Offset date is transformed to a dataframe with a PeriodIndex
    """
    df = pd.DataFrame(
        index=pd.DatetimeIndex(['2011-01-02', '2011-01-09', '2011-01-16', '2011-01-23',
               '2011-01-30', '2011-02-06', '2011-02-13', '2011-02-20',
               '2011-02-27', '2011-03-06'],
              dtype='datetime64[ns]', name='date_time', freq='W-SUN'),
        data=np.array([[17.], [25.], [39.], [22.], [33.], [39.], [39.], [17.], [34.], [52.]]),
        columns=["users"]
    )

    pipe = StartSktimePipe(datetime.datetime(2011, 1, 2), ["users"], "W-SUN")

    output = pipe.transform(df)

    expected = pd.DataFrame(
        index=pd.PeriodIndex(['2010-12-27/2011-01-02', '2011-01-03/2011-01-09',
             '2011-01-10/2011-01-16', '2011-01-17/2011-01-23',
             '2011-01-24/2011-01-30', '2011-01-31/2011-02-06',
             '2011-02-07/2011-02-13', '2011-02-14/2011-02-20',
             '2011-02-21/2011-02-27', '2011-02-28/2011-03-06'],
            dtype='period[W-SUN]', name='date_time'),
        data=np.array([[17.], [25.], [39.], [22.], [33.], [39.], [39.], [17.], [34.], [52.]]),
        columns=["users"]
    )
    pd.testing.assert_frame_equal(output, expected)


def test_StartSktimePipe_transforms_input_as_df_with_datetimeindex_inferred_freq():
    """
    Check if the input DataFrame with inferred Offset date is transformed to a dataframe with a PeriodIndex
    """
    df = pd.DataFrame(
        index=pd.DatetimeIndex(['2011-01-02', '2011-01-09', '2011-01-16', '2011-01-23',
               '2011-01-30', '2011-02-06', '2011-02-13', '2011-02-20',
               '2011-02-27', '2011-03-06'],
              dtype='datetime64[ns]', name='date_time'),
        data=np.array([[17.], [25.], [39.], [22.], [33.], [39.], [39.], [17.], [34.], [52.]]),
        columns=["users"]
    )

    pipe = StartSktimePipe(datetime.datetime(2011, 1, 2), ["users"], "W-SUN")

    output = pipe.transform(df)

    expected = pd.DataFrame(
        index=pd.PeriodIndex(['2010-12-27/2011-01-02', '2011-01-03/2011-01-09',
             '2011-01-10/2011-01-16', '2011-01-17/2011-01-23',
             '2011-01-24/2011-01-30', '2011-01-31/2011-02-06',
             '2011-02-07/2011-02-13', '2011-02-14/2011-02-20',
             '2011-02-21/2011-02-27', '2011-02-28/2011-03-06'],
            dtype='period[W-SUN]', name='date_time'),
        data=np.array([[17.], [25.], [39.], [22.], [33.], [39.], [39.], [17.], [34.], [52.]]),
        columns=["users"]
    )
    pd.testing.assert_frame_equal(output, expected)


def test_StartSktimePipe_transforms_input_as_nparray():
    """
    Check if the input numpy array is transformed to a dataframe with a PeriodIndex
    """
    a = np.array([[17.], [25.], [39.], [22.], [33.], [39.], [39.], [17.], [34.], [52.]])

    pipe = StartSktimePipe(datetime.datetime(2011, 1, 2), ["users"], "W-SUN")

    output = pipe.transform(a)

    expected = pd.DataFrame(
        index=pd.PeriodIndex(['2010-12-27/2011-01-02', '2011-01-03/2011-01-09',
             '2011-01-10/2011-01-16', '2011-01-17/2011-01-23',
             '2011-01-24/2011-01-30', '2011-01-31/2011-02-06',
             '2011-02-07/2011-02-13', '2011-02-14/2011-02-20',
             '2011-02-21/2011-02-27', '2011-02-28/2011-03-06'],
            dtype='period[W-SUN]'),
        data=np.array([[17.], [25.], [39.], [22.], [33.], [39.], [39.], [17.], [34.], [52.]]),
        columns=["users"]
    )
    pd.testing.assert_frame_equal(output, expected)


def test_StartSktimePipe_transforms_input_as_series():
    """
    Check if the input Series is transformed to a dataframe with a PeriodIndex
    """

    s = pd.Series(index=pd.DatetimeIndex(['2011-01-02', '2011-01-09', '2011-01-16', '2011-01-23',
               '2011-01-30', '2011-02-06', '2011-02-13', '2011-02-20',
               '2011-02-27', '2011-03-06'],
               dtype='datetime64[ns]', freq='W-SUN'),
            data=np.array([17., 25., 39., 22., 33., 39., 39., 17., 34., 52.])
    )

    pipe = StartSktimePipe(datetime.datetime(2011, 1, 2), ["users"], "W-SUN")

    output = pipe.transform(s)

    expected = pd.Series(
        index=pd.PeriodIndex(['2010-12-27/2011-01-02', '2011-01-03/2011-01-09',
             '2011-01-10/2011-01-16', '2011-01-17/2011-01-23',
             '2011-01-24/2011-01-30', '2011-01-31/2011-02-06',
             '2011-02-07/2011-02-13', '2011-02-14/2011-02-20',
             '2011-02-21/2011-02-27', '2011-02-28/2011-03-06'],
            dtype='period[W-SUN]'),
        data=np.array([17., 25., 39., 22., 33., 39., 39., 17., 34., 52.])
    )
    pd.testing.assert_series_equal(output, expected)


def test_StartSktimePipe_inverse_transform():
    """
    Check if the input DataFrame with a PeriodIndex is transformed to a dataframe with a DatetimeIndex
    """
    df = pd.DataFrame(
        index=pd.PeriodIndex(['2010-12-27/2011-01-02', '2011-01-03/2011-01-09',
             '2011-01-10/2011-01-16', '2011-01-17/2011-01-23',
             '2011-01-24/2011-01-30', '2011-01-31/2011-02-06',
             '2011-02-07/2011-02-13', '2011-02-14/2011-02-20',
             '2011-02-21/2011-02-27', '2011-02-28/2011-03-06'],
            dtype='period[W-SUN]'),
        data=np.array([[17.], [25.], [39.], [22.], [33.], [39.], [39.], [17.], [34.], [52.]]),
        columns=["users"]
    )

    pipe = StartSktimePipe(datetime.datetime(2011, 1, 2), ["users"], "W-SUN")

    output = pipe.inverse_transform(df)

    expected = pd.DataFrame(
        index=pd.DatetimeIndex(['2011-01-02', '2011-01-09', '2011-01-16', '2011-01-23',
               '2011-01-30', '2011-02-06', '2011-02-13', '2011-02-20',
               '2011-02-27', '2011-03-06'],
              dtype='datetime64[ns]', freq='W-SUN'),
        data=np.array([[17.], [25.], [39.], [22.], [33.], [39.], [39.], [17.], [34.], [52.]]),
        columns=["users"]
    )

    pd.testing.assert_frame_equal(output, expected)


def test_EndSktimePipe_transform():
    """
    Check if the input DataFrame with a PeriodIndex is transformed to a dataframe with a DatetimeIndex
    """
    df = pd.DataFrame(
        index=pd.PeriodIndex(['2010-12-27/2011-01-02', '2011-01-03/2011-01-09',
             '2011-01-10/2011-01-16', '2011-01-17/2011-01-23',
             '2011-01-24/2011-01-30', '2011-01-31/2011-02-06',
             '2011-02-07/2011-02-13', '2011-02-14/2011-02-20',
             '2011-02-21/2011-02-27', '2011-02-28/2011-03-06'],
            dtype='period[W-SUN]'),
        data=np.array([[17.], [25.], [39.], [22.], [33.], [39.], [39.], [17.], [34.], [52.]]),
        columns=["users"]
    )

    pipe = EndSktimePipe(datetime.datetime(2011, 1, 2), ["users"], "W-SUN")

    output = pipe.transform(df)

    expected = pd.DataFrame(
        index=pd.DatetimeIndex(['2011-01-02', '2011-01-09', '2011-01-16', '2011-01-23',
               '2011-01-30', '2011-02-06', '2011-02-13', '2011-02-20',
               '2011-02-27', '2011-03-06'],
              dtype='datetime64[ns]', freq='W-SUN'),
        data=np.array([[17.], [25.], [39.], [22.], [33.], [39.], [39.], [17.], [34.], [52.]]),
        columns=["users"]
    )

    pd.testing.assert_frame_equal(output, expected)


def test_EndSktimePipe_inverse_transforms_input_as_df_with_datetimeindex_WSUN():
    """
    Check if the input DataFrame with a DatetimeIndex with specified Offset date
    is inverse-transformed to a dataframe with a PeriodIndex
    """
    df = pd.DataFrame(
        index=pd.DatetimeIndex(['2011-01-02', '2011-01-09', '2011-01-16', '2011-01-23',
               '2011-01-30', '2011-02-06', '2011-02-13', '2011-02-20',
               '2011-02-27', '2011-03-06'],
              dtype='datetime64[ns]', name='date_time', freq='W-SUN'),
        data=np.array([[17.], [25.], [39.], [22.], [33.], [39.], [39.], [17.], [34.], [52.]]),
        columns=["users"]
    )

    pipe = EndSktimePipe(datetime.datetime(2011, 1, 2), ["users"], "W-SUN")

    output = pipe.inverse_transform(df)

    expected = pd.DataFrame(
        index=pd.PeriodIndex(['2010-12-27/2011-01-02', '2011-01-03/2011-01-09',
             '2011-01-10/2011-01-16', '2011-01-17/2011-01-23',
             '2011-01-24/2011-01-30', '2011-01-31/2011-02-06',
             '2011-02-07/2011-02-13', '2011-02-14/2011-02-20',
             '2011-02-21/2011-02-27', '2011-02-28/2011-03-06'],
            dtype='period[W-SUN]', name='date_time'),
        data=np.array([[17.], [25.], [39.], [22.], [33.], [39.], [39.], [17.], [34.], [52.]]),
        columns=["users"]
    )
    pd.testing.assert_frame_equal(output, expected)


def test_EndSktimePipe_inverse_transforms_input_as_df_with_datetimeindex_inferred_freq():
    """
    Check if the input DataFrame with a DatetimeIndex and inferred OffsetDate
    is inverse-transformed to a dataframe with a PeriodIndex
    """
    df = pd.DataFrame(
        index=pd.DatetimeIndex(['2011-01-02', '2011-01-09', '2011-01-16', '2011-01-23',
               '2011-01-30', '2011-02-06', '2011-02-13', '2011-02-20',
               '2011-02-27', '2011-03-06'],
              dtype='datetime64[ns]', name='date_time'),
        data=np.array([[17.], [25.], [39.], [22.], [33.], [39.], [39.], [17.], [34.], [52.]]),
        columns=["users"]
    )

    pipe = EndSktimePipe(datetime.datetime(2011, 1, 2), ["users"], "W-SUN")

    output = pipe.inverse_transform(df)

    expected = pd.DataFrame(
        index=pd.PeriodIndex(['2010-12-27/2011-01-02', '2011-01-03/2011-01-09',
             '2011-01-10/2011-01-16', '2011-01-17/2011-01-23',
             '2011-01-24/2011-01-30', '2011-01-31/2011-02-06',
             '2011-02-07/2011-02-13', '2011-02-14/2011-02-20',
             '2011-02-21/2011-02-27', '2011-02-28/2011-03-06'],
            dtype='period[W-SUN]', name='date_time'),
        data=np.array([[17.], [25.], [39.], [22.], [33.], [39.], [39.], [17.], [34.], [52.]]),
        columns=["users"]
    )
    pd.testing.assert_frame_equal(output, expected)


def test_EndSktimePipe_inverse_transforms_input_as_nparray():
    """
    Check if the input numpy array is inverse-transformed to a dataframe with a PeriodIndex
    """
    a = np.array([[17.], [25.], [39.], [22.], [33.], [39.], [39.], [17.], [34.], [52.]])

    pipe = EndSktimePipe(datetime.datetime(2011, 1, 2), ["users"], "W-SUN")

    output = pipe.inverse_transform(a)

    expected = pd.DataFrame(
        index=pd.PeriodIndex(['2010-12-27/2011-01-02', '2011-01-03/2011-01-09',
             '2011-01-10/2011-01-16', '2011-01-17/2011-01-23',
             '2011-01-24/2011-01-30', '2011-01-31/2011-02-06',
             '2011-02-07/2011-02-13', '2011-02-14/2011-02-20',
             '2011-02-21/2011-02-27', '2011-02-28/2011-03-06'],
            dtype='period[W-SUN]'),
        data=np.array([[17.], [25.], [39.], [22.], [33.], [39.], [39.], [17.], [34.], [52.]]),
        columns=["users"]
    )
    pd.testing.assert_frame_equal(output, expected)


def test_EndSktimePipe_inverse_transforms_input_as_series():
    """
    Check if the input series is inverse-transformed to a dataframe with a PeriodIndex
    """

    s = pd.Series(index=pd.DatetimeIndex(['2011-01-02', '2011-01-09', '2011-01-16', '2011-01-23',
               '2011-01-30', '2011-02-06', '2011-02-13', '2011-02-20',
               '2011-02-27', '2011-03-06'],
               dtype='datetime64[ns]', freq='W-SUN'),
            data=np.array([17., 25., 39., 22., 33., 39., 39., 17., 34., 52.])
    )

    pipe = EndSktimePipe(datetime.datetime(2011, 1, 2), ["users"], "W-SUN")

    output = pipe.inverse_transform(s)

    expected = pd.Series(
        index=pd.PeriodIndex(['2010-12-27/2011-01-02', '2011-01-03/2011-01-09',
             '2011-01-10/2011-01-16', '2011-01-17/2011-01-23',
             '2011-01-24/2011-01-30', '2011-01-31/2011-02-06',
             '2011-02-07/2011-02-13', '2011-02-14/2011-02-20',
             '2011-02-21/2011-02-27', '2011-02-28/2011-03-06'],
            dtype='period[W-SUN]'),
        data=np.array([17., 25., 39., 22., 33., 39., 39., 17., 34., 52.])
    )
    pd.testing.assert_series_equal(output, expected)
