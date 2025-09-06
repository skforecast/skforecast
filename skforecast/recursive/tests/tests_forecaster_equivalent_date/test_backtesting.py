# Unit test backtesting ForecasterEquivalentDate
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.recursive import ForecasterEquivalentDate
from skforecast.model_selection._validation import backtesting_forecaster
from skforecast.model_selection._split import TimeSeriesFold

# Fixtures
from .fixtures_forecaster_equivalent_date import y


@pytest.mark.parametrize("initial_train_size", 
                         [len(y) - 20, "2000-01-30 00:00:00"],
                         ids=lambda init: f'initial_train_size: {init}')
def test_backtesting_with_ForecasterEquivalentDate(initial_train_size):
    """
    Test backtesting with ForecasterEquivalentDate.
    """
    forecaster = ForecasterEquivalentDate(
                     offset    = pd.DateOffset(days=10),
                     n_offsets = 2 
                 )
    cv = TimeSeriesFold(
             initial_train_size = initial_train_size,
             steps              = 5,
             refit              = True,
         )

    metric, predictions = backtesting_forecaster(
        forecaster = forecaster,
        y          = y,
        cv         = cv,
        metric     = 'mean_absolute_error',
        verbose    = False,
        n_jobs     = 'auto'
    )

    expected_metric = pd.DataFrame({'mean_absolute_error': [0.2537094475]})
    expected = pd.DataFrame(
        data    = np.array([0.48878949, 0.78924075, 0.58151378, 0.3353507, 0.56024382,
                            0.53047716, 0.27214019, 0.20185749, 0.41263271, 0.58140185,
                            0.36325295, 0.64156648, 0.57765904, 0.5523543, 0.57413684,
                            0.31761006, 0.39406999, 0.56082619, 0.61893703, 0.5664064]),
        index   = pd.date_range(start='2000-01-31', periods=20, freq='D'),
        columns = ['pred']
    )
    expected.insert(0, 'fold', [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)

    pd.testing.assert_frame_equal(metric, expected_metric)
    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize("initial_train_size", 
                         [len(y) - 20, "2000-01-30 23:59:00"],
                         ids=lambda init: f'initial_train_size: {init}')
def test_backtesting_with_ForecasterEquivalentDate_interval(initial_train_size):
    """
    Test backtesting with ForecasterEquivalentDate and interval prediction.
    """
    forecaster = ForecasterEquivalentDate(
                     offset    = pd.DateOffset(days=3),
                     n_offsets = 3 
                 )
    cv = TimeSeriesFold(
             initial_train_size = initial_train_size,
             steps              = 3,
             refit              = False,
         )

    metric, predictions = backtesting_forecaster(
        forecaster      = forecaster,
        y               = y,
        cv              = cv,
        metric          = 'mean_absolute_error',
        interval        = [10, 90],
        interval_method = 'conformal',
        verbose         = False,
        n_jobs          = 'auto'
    )
    
    expected_metric = pd.DataFrame({'mean_absolute_error': [0.2273233665]})
    expected = pd.DataFrame(
                   data = np.array([
                              [0.60004613,  0.1257829 ,  1.07430936],
                              [0.44704276,  0.240286  ,  0.65379952],
                              [0.5345961 ,  0.34670492,  0.72248727],
                              [0.34760385, -0.09290349,  0.78811119],
                              [0.35012471, -0.09038263,  0.79063205],
                              [0.47454251,  0.32686905,  0.62221597],
                              [0.27135109, -0.16915625,  0.71185843],
                              [0.38441517,  0.06673199,  0.70209835],
                              [0.45803337,  0.25127661,  0.66479012],
                              [0.33738045, -0.10312689,  0.77788779],
                              [0.58430687,  0.11004364,  1.0585701 ],
                              [0.562428  ,  0.35692325,  0.76793275],
                              [0.4739577 ,  0.32628424,  0.62163116],
                              [0.64772413,  0.45862738,  0.83682089],
                              [0.45734655,  0.25058979,  0.6641033 ],
                              [0.41515782,  0.20840107,  0.62191458],
                              [0.64405611,  0.45495935,  0.83315286],
                              [0.64202919,  0.45293244,  0.83112595],
                              [0.35652584, -0.0839815 ,  0.79703318],
                              [0.50727114,  0.35959768,  0.6549446 ]]),
                   index = pd.date_range(start='2000-01-31', periods=20, freq='D'),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    expected.insert(0, 'fold', [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6])

    pd.testing.assert_frame_equal(metric, expected_metric)
    pd.testing.assert_frame_equal(predictions, expected)
