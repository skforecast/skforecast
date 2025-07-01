# Unit test predict_interval ForecasterEquivalentDate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.exceptions import MissingValuesWarning
from skforecast.recursive import ForecasterEquivalentDate

# Fixtures
from .fixtures_forecaster_equivalent_date import y


def test_predict_interval_ValueError_when_method_is_not_conformal():
    """
    Test ValueError is raised when method is not 'conformal'.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)

    err_msg = re.escape(
        "Method 'bootstrapping' is not supported. Only 'conformal' is available."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(steps=3, method='bootstrapping')


def test_predict_interval_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)

    err_msg = re.escape(
        'This Forecaster instance is not fitted yet. Call `fit` with '
        'appropriate arguments before using predict.'
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict_interval(steps=3)


def test_predict_interval_5_steps_with_int_offset_offset_1_n_offsets_1():
    """
    Test predict_interval method with int offset, offset=1 and n_offsets=1.
    """
    forecaster = ForecasterEquivalentDate(
                     offset    = 1,
                     n_offsets = 1
                 )
    forecaster.fit(y=y, store_in_sample_residuals=True)
    predictions = forecaster.predict_interval(steps=5)

    expected = pd.DataFrame(
                   data = np.array([
                              [0.61289453, 0.24063212, 0.98515694],
                              [0.61289453, 0.24063212, 0.98515694],
                              [0.61289453, 0.24063212, 0.98515694],
                              [0.61289453, 0.24063212, 0.98515694],
                              [0.61289453, 0.24063212, 0.98515694]]),
                   index = pd.date_range(start='2000-02-20', periods=5, freq='D'),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_5_steps_with_int_offset_7_n_offsets_1():
    """
    Test predict_interval method with int offset, offset=7 and n_offsets=1.
    """
    forecaster = ForecasterEquivalentDate(
                     offset    = 7,
                     n_offsets = 1
                 )
    forecaster.fit(y=y, store_in_sample_residuals=True)
    predictions = forecaster.predict_interval(steps=5)

    expected = pd.DataFrame(
                   data = np.array([
                              [0.41482621, 0.0359672 , 0.79368522],
                              [0.86630916, 0.03617552, 1.6964428 ],
                              [0.25045537, 0.05372066, 0.44719008],
                              [0.48303426, 0.24535917, 0.72070935],
                              [0.98555979, 0.15542615, 1.81569343]]),
                   index = pd.date_range(start='2000-02-20', periods=5, freq='D'),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_7_steps_with_int_offset_7_n_offsets_2_no_binned():
    """
    Test predict_interval method with int offset, offset=7 and n_offsets=2.
    """
    forecaster = ForecasterEquivalentDate(
                     offset    = 7,
                     n_offsets = 2,
                     agg_func  = np.mean
                 )
    forecaster.fit(y=y, store_in_sample_residuals=True)
    predictions = forecaster.predict_interval(steps=7, use_binned_residuals=False)

    expected = pd.DataFrame(
                   data = np.array([
                              [0.42058876,  0.02043732,  0.8207402 ],
                              [0.87984916,  0.47969772,  1.2800006 ],
                              [0.5973077 ,  0.19715626,  0.99745913],
                              [0.49243547,  0.09228403,  0.89258691],
                              [0.80475637,  0.40460493,  1.20490781],
                              [0.31755176, -0.08259968,  0.7177032 ],
                              [0.46509001,  0.06493857,  0.86524144]]),
                   index = pd.date_range(start='2000-02-20', periods=7, freq='D'),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_7_steps_with_int_offset_5_n_offsets_1_using_last_window():
    """
    Test predict_interval method with int offset, offset=1 and n_offsets=1, using last_window.
    """
    last_window = pd.Series(
        data  = np.array([0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759]),
        index = pd.date_range(start='2000-08-10', periods=5, freq='D'),
        name  = 'y'
    )

    forecaster = ForecasterEquivalentDate(
                     offset    = 5,
                     n_offsets = 1
                 )
    forecaster.fit(y=y)
    forecaster.set_in_sample_residuals(y=y)
    predictions = forecaster.predict_interval(steps=7, last_window=last_window)

    expected = pd.DataFrame(
                   data = np.array([
                              [0.73799541,  0.2065472 ,  1.26944362],
                              [0.18249173, -0.43727373,  0.80225719],
                              [0.17545176, -0.4443137 ,  0.79521722],
                              [0.53155137,  0.09276482,  0.97033792],
                              [0.53182759,  0.17518645,  0.88846873],
                              [0.73799541,  0.2065472 ,  1.26944362],
                              [0.18249173, -0.43727373,  0.80225719]]),
                   index = pd.date_range(start='2000-08-15', periods=7, freq='D'),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_7_steps_with_int_offset_5_n_offsets_2_using_last_window():
    """
    Test predict_interval method with int offset, offset=1 and n_offsets=1, using last_window.
    """
    last_window = pd.Series(
        data = np.array([0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
                         0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]),
        index = pd.date_range(start='2000-08-05', periods=10, freq='D'),
        name = 'y'
    )
    forecaster = ForecasterEquivalentDate(
                     offset    = 5,
                     n_offsets = 2,
                     agg_func  = np.mean
                 )
    forecaster.fit(y=y, store_in_sample_residuals=True)
    predictions = forecaster.predict_interval(steps=7, last_window=last_window)
    
    expected = pd.DataFrame(
                   data = np.array([
                              [0.49422539,  0.05874109,  0.92970969],
                              [0.332763  , -0.18714832,  0.85267431],
                              [0.58050578,  0.23107803,  0.92993352],
                              [0.52551824,  0.09003394,  0.96100255],
                              [0.57236106,  0.22293331,  0.92178881],
                              [0.49422539,  0.05874109,  0.92970969],
                              [0.332763  , -0.18714832,  0.85267431]]),
                   index = pd.date_range(start='2000-08-15', periods=7, freq='D'),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_ValueError_when_all_equivalent_values_are_missing():
    """
    Test ValueError when all equivalent values are missing because the offset is
    too large.
    """
    y = pd.Series(
        np.arange(7), 
        index=pd.date_range(start='2000-02-25', periods=7, freq='D')
    )

    forecaster = ForecasterEquivalentDate(
                     offset    = pd.offsets.MonthEnd(1),
                     n_offsets = 1
                 )
    forecaster.fit(y=y, store_in_sample_residuals=True)

    last_window = pd.Series(
        np.arange(5), 
        index=pd.date_range(start='2000-02-10', periods=5, freq='D')
    )

    err_msg = re.escape(
        "All equivalent values are missing. This is caused by using "
        "an offset (<MonthEnd>) larger than the available data. "
        "Try to decrease the size of the offset (<MonthEnd>), "
        "the number of `n_offsets` (1) or increase the "
        "size of `last_window`. In backtesting, this error may be "
        "caused by using an `initial_train_size` too small."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(steps=3, last_window=last_window)


def test_predict_interval_MissingValuesWarning_when_any_equivalent_values_are_missing():
    """
    Test MissingValuesWarning when some equivalent values are missing because the 
    offset is too large.
    """
    y = pd.Series(
        np.arange(51), 
        index=pd.date_range(start='2000-02-25', periods=51, freq='D')
    )
    
    forecaster = ForecasterEquivalentDate(
                     offset    = pd.offsets.MonthEnd(1),
                     n_offsets = 2,
                     agg_func  = np.mean
                 )
    forecaster.fit(y=y, store_in_sample_residuals=True)

    last_window = pd.Series(
        np.arange(50), 
        index=pd.date_range(start='2000-02-10', periods=50, freq='D')
    )

    warn_msg = re.escape(
        "Steps: ['2000-03-31', '2000-04-01', '2000-04-02'] "
        "are calculated with less than 2 `n_offsets`. "
        "To avoid this, increase the `last_window` size or decrease "
        "the number of `n_offsets`. The current configuration requires " 
        "a total offset of <2 * MonthEnds>."
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        predictions = forecaster.predict_interval(steps=3, last_window=last_window)

    expected = pd.Series(
        data  = np.array([19., 19., 19.]),
        index = pd.date_range(start='2000-03-31', periods=3, freq='D'),
        name  = 'pred'
    )

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_7_steps_with_offset_Day_5_n_offsets_1_using_last_window():
    """
    Test predict_interval method with offset pd.offsets.Day() and 
    n_offsets=1, using last_window.
    """
    last_window = pd.Series(
        data = np.array([0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759]),
        index = pd.date_range(start='2000-08-10', periods=5, freq='D'),
        name = 'y'
    )

    forecaster = ForecasterEquivalentDate(
                     offset    = pd.offsets.Day(5),
                     n_offsets = 1
                 )
    forecaster.fit(y=y, store_in_sample_residuals=True)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    forecaster.out_sample_residuals_by_bin_ = forecaster.in_sample_residuals_by_bin_
    predictions = forecaster.predict_interval(
        steps=7, last_window=last_window, use_in_sample_residuals=False
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [0.73799541,  0.2065472 ,  1.26944362],
                              [0.18249173, -0.43727373,  0.80225719],
                              [0.17545176, -0.4443137 ,  0.79521722],
                              [0.53155137,  0.09276482,  0.97033792],
                              [0.53182759,  0.17518645,  0.88846873],
                              [0.73799541,  0.2065472 ,  1.26944362],
                              [0.18249173, -0.43727373,  0.80225719]]),
                   index = pd.date_range(start='2000-08-15', periods=7, freq='D'),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )

    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_7_steps_with_offset_Day_5_n_offsets_2_using_last_window():
    """
    Test predict_interval method with offset pd.offsets.Day(5) and n_offsets=2,
    using last_window.
    """
    y_monthly = y.copy()
    y_monthly.index = pd.date_range(start='1990-01-01', periods=50, freq='MS')
    last_window = pd.Series(
        data = np.array([0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
                         0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]),
        index = pd.date_range(start='2000-08-05', periods=10, freq='MS'),
        name = 'y'
    )

    forecaster = ForecasterEquivalentDate(
                     offset    = pd.DateOffset(months=2),
                     n_offsets = 2,
                     agg_func  = np.mean
                 )
    forecaster.fit(y=y_monthly, store_in_sample_residuals=True)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    forecaster.out_sample_residuals_by_bin_ = forecaster.in_sample_residuals_by_bin_
    predictions = forecaster.predict_interval(
        steps=7, last_window=last_window, use_in_sample_residuals=False, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [0.50125969, 0.00291078, 0.99960861],
                              [0.79922716, 0.30087825, 1.29757608],
                              [0.50125969, 0.00291078, 0.99960861],
                              [0.79922716, 0.30087825, 1.29757608],
                              [0.50125969, 0.00291078, 0.99960861],
                              [0.79922716, 0.30087825, 1.29757608],
                              [0.50125969, 0.00291078, 0.99960861]]),
                   index = pd.date_range(start='2001-07-01', periods=7, freq='MS'),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )

    pd.testing.assert_frame_equal(predictions, expected)
