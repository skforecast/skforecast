# Unit test backtesting ForecasterEquivalentDate
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.recursive import ForecasterEquivalentDate
from skforecast.model_selection._validation import backtesting_forecaster
from skforecast.model_selection._split import TimeSeriesFold

# Fixtures
from .fixtures_forecaster_equivalent_date import y


def test_backtesting_with_ForecasterEquivalentDate():
    """
    Test backtesting with ForecasterEquivalentDate.
    """
    forecaster = ForecasterEquivalentDate(
                     offset    = pd.DateOffset(days=10),
                     n_offsets = 2 
                 )
    cv = TimeSeriesFold(
        initial_train_size = 30,
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

    pd.testing.assert_frame_equal(metric, expected_metric)
    pd.testing.assert_frame_equal(predictions, expected)
