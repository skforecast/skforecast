# Unit test _encode_levels_onehot ForecasterRecursiveMultiSeries
# ==============================================================================
import numpy as np
from sklearn.linear_model import LinearRegression
from ....recursive import ForecasterRecursiveMultiSeries

# Fixtures
from .fixtures_forecaster_recursive_multiseries import (
    series_dict_unordered,
    exog_dict_unordered
)


def test_encode_levels_onehot_output_when_series_unordered_dropped_and_unknown():
    """
    Test the one-hot encoding of the levels when the series are not in
    alphabetical order ('c', 'a', 'd', 'b'), one series ('d') has no rows in
    X_train (it has no exog and `dropna_from_series=True`) and one level is
    not known. The columns follow `encoding_mapping_` (alphabetical), not the
    order of the series, and the dropped and unknown levels are encoded with
    zeros, as their columns have no ones in the training matrix.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 2,
                     encoding           = 'onehot',
                     dropna_from_series = True
                 )
    forecaster.fit(
        series=series_dict_unordered, exog=exog_dict_unordered,
        suppress_warnings=True
    )
    results = forecaster._encode_levels_onehot(['c', 'a', 'd', 'b', 'unknown'])

    expected = np.array([
        [0., 0., 1., 0.],
        [1., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 0.]
    ])

    assert forecaster.encoding_mapping_ == {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    assert forecaster.X_train_series_names_in_ == ['c', 'a', 'b']
    np.testing.assert_array_equal(results, expected)
